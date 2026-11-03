"""Auto-instrumentation for OpenAI-compatible clients (NVIDIA NIM, vLLM, OpenAI).

    client = instrument_openai(AsyncOpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=...))

Every `chat.completions.create` call becomes an `llm` span with model, provider, token usage,
messages and completion (content capture follows the tracer's policy). Streaming is supported: the
span closes when the stream is consumed, with usage taken from the final chunk when present.
"""

from __future__ import annotations

import functools
import inspect
from typing import Any

from argus import semconv
from argus.sdk.tracer import SpanRecord, _content, _current, get_tracer


def provider_for(base_url: str) -> str:
    url = base_url.lower()
    if "nvidia" in url:
        return "nvidia_nim"
    if "openai.com" in url:
        return "openai"
    if "localhost" in url or "127.0.0.1" in url:
        return "self_hosted"
    return "openai_compatible"


def _start(kwargs: dict[str, Any], provider: str) -> SpanRecord:
    model = kwargs.get("model", "unknown")
    record = get_tracer().start_span(
        f"chat {model}",
        kind="llm",
        attributes={
            semconv.OPERATION: "chat",
            semconv.SYSTEM: provider,
            semconv.REQUEST_MODEL: model,
        },
    )
    record.set(semconv.INPUT, _content(kwargs.get("messages", [])))
    return record


def _finish(record: SpanRecord, response: Any) -> None:
    record.set(semconv.RESPONSE_MODEL, getattr(response, "model", None))
    usage = getattr(response, "usage", None)
    if usage is not None:
        record.set(semconv.INPUT_TOKENS, getattr(usage, "prompt_tokens", None))
        record.set(semconv.OUTPUT_TOKENS, getattr(usage, "completion_tokens", None))
    choices = getattr(response, "choices", None) or []
    if choices:
        message = choices[0].message
        tool_calls = [tc.function.name for tc in (getattr(message, "tool_calls", None) or [])]
        record.set(semconv.OUTPUT, _content(message.content or {"tool_calls": tool_calls}))


class _StreamWrapper:
    """Wraps sync or async streams; accumulates text and closes the span at the end."""

    def __init__(self, stream: Any, record: SpanRecord) -> None:
        self._stream, self._record = stream, record
        self._parts: list[str] = []

    def _observe(self, chunk: Any) -> None:
        if getattr(chunk, "usage", None):
            self._record.set(semconv.INPUT_TOKENS, chunk.usage.prompt_tokens)
            self._record.set(semconv.OUTPUT_TOKENS, chunk.usage.completion_tokens)
        self._record.set(semconv.RESPONSE_MODEL, getattr(chunk, "model", None))
        for choice in getattr(chunk, "choices", None) or []:
            if choice.delta and choice.delta.content:
                self._parts.append(choice.delta.content)

    def _close(self) -> None:
        self._record.set(semconv.OUTPUT, "".join(self._parts)[:8000])
        get_tracer().end_span(self._record)

    def __iter__(self) -> Any:
        try:
            for chunk in self._stream:
                self._observe(chunk)
                yield chunk
        finally:
            self._close()

    async def __aiter__(self) -> Any:
        try:
            async for chunk in self._stream:
                self._observe(chunk)
                yield chunk
        finally:
            self._close()


def instrument_openai(client: Any) -> Any:
    completions = client.chat.completions
    original = completions.create
    provider = provider_for(str(getattr(client, "base_url", "")))
    if getattr(original, "_argus_wrapped", False):
        return client

    if inspect.iscoroutinefunction(original):

        @functools.wraps(original)
        async def acreate(*args: Any, **kwargs: Any) -> Any:
            record = _start(kwargs, provider)
            token = _current.set(record)
            try:
                response = await original(*args, **kwargs)
            except BaseException as exc:
                record.record_error(exc)
                get_tracer().end_span(record)
                raise
            finally:
                _current.reset(token)
            if kwargs.get("stream"):
                return _StreamWrapper(response, record)
            _finish(record, response)
            get_tracer().end_span(record)
            return response

        acreate._argus_wrapped = True  # type: ignore[attr-defined]
        completions.create = acreate
    else:

        @functools.wraps(original)
        def create(*args: Any, **kwargs: Any) -> Any:
            record = _start(kwargs, provider)
            token = _current.set(record)
            try:
                response = original(*args, **kwargs)
            except BaseException as exc:
                record.record_error(exc)
                get_tracer().end_span(record)
                raise
            finally:
                _current.reset(token)
            if kwargs.get("stream"):
                return _StreamWrapper(response, record)
            _finish(record, response)
            get_tracer().end_span(record)
            return response

        create._argus_wrapped = True  # type: ignore[attr-defined]
        completions.create = create
    return client
