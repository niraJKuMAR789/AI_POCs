"""LangChain / LangGraph callback handler that emits Argus spans.

    graph.invoke(state, config={"callbacks": [ArgusCallbackHandler()]})

LangGraph nodes arrive as chain events, chat models as LLM events and tools as tool events; the
LangChain run tree (run_id / parent_run_id) becomes the span tree. If the graph runs inside an
Argus span (e.g. a `@trace_agent` function), that span becomes the root's parent.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

from argus import semconv
from argus.sdk.tracer import SpanRecord, _content, args_hash, current_span, get_tracer


class ArgusCallbackHandler(BaseCallbackHandler):
    def __init__(self) -> None:
        self._spans: dict[UUID, SpanRecord] = {}

    def _start(
        self, run_id: UUID, parent_run_id: UUID | None, name: str, kind: str, attributes: dict[str, Any]
    ) -> SpanRecord:
        parent = self._spans.get(parent_run_id) if parent_run_id else None
        record = get_tracer().start_span(name, kind=kind, attributes=attributes, parent=parent or current_span())
        self._spans[run_id] = record
        return record

    def _end(self, run_id: UUID, output: Any = None, error: BaseException | None = None) -> None:
        record = self._spans.pop(run_id, None)
        if record is None:
            return
        if error is not None:
            record.record_error(error)
        elif output is not None:
            record.set(semconv.OUTPUT, _content(output))
        get_tracer().end_span(record)

    @staticmethod
    def _name(serialized: dict[str, Any] | None, kwargs: dict[str, Any], default: str) -> str:
        return kwargs.get("name") or (serialized or {}).get("name") or default

    # chains / graph nodes
    def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        record = self._start(run_id, parent_run_id, self._name(serialized, kwargs, "chain"), "chain", {})
        record.set(semconv.INPUT, _content(inputs))

    def on_chain_end(self, outputs: Any, *, run_id: UUID, **kwargs: Any) -> None:
        self._end(run_id, outputs)

    def on_chain_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._end(run_id, error=error)

    # chat models / LLMs
    def on_chat_model_start(
        self,
        serialized: dict[str, Any] | None,
        messages: list[list[Any]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        params = kwargs.get("invocation_params") or {}
        model = params.get("model") or params.get("model_name") or self._name(serialized, kwargs, "chat_model")
        record = self._start(
            run_id,
            parent_run_id,
            f"chat {model}",
            "llm",
            {
                semconv.OPERATION: "chat",
                semconv.REQUEST_MODEL: model,
                semconv.SYSTEM: (kwargs.get("metadata") or {}).get("ls_provider"),
            },
        )
        record.set(semconv.INPUT, _content([[getattr(m, "content", str(m)) for m in batch] for batch in messages]))

    def on_llm_start(
        self,
        serialized: dict[str, Any] | None,
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        record = self._start(
            run_id, parent_run_id, self._name(serialized, kwargs, "llm"), "llm", {semconv.OPERATION: "text_completion"}
        )
        record.set(semconv.INPUT, _content(prompts))

    def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        record = self._spans.get(run_id)
        if record is not None:
            usage = (getattr(response, "llm_output", None) or {}).get("token_usage") or {}
            generations = getattr(response, "generations", None) or []
            first = generations[0][0] if generations and generations[0] else None
            message = getattr(first, "message", None)
            meta = getattr(message, "usage_metadata", None) or {}
            record.set(semconv.INPUT_TOKENS, meta.get("input_tokens") or usage.get("prompt_tokens"))
            record.set(semconv.OUTPUT_TOKENS, meta.get("output_tokens") or usage.get("completion_tokens"))
            text = getattr(first, "text", None) if first else None
            self._end(run_id, text or "")
        else:
            self._end(run_id)

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._end(run_id, error=error)

    # tools
    def on_tool_start(
        self,
        serialized: dict[str, Any] | None,
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        name = self._name(serialized, kwargs, "tool")
        record = self._start(
            run_id,
            parent_run_id,
            name,
            "tool",
            {
                semconv.OPERATION: "execute_tool",
                semconv.TOOL_NAME: name,
                semconv.TOOL_ARGS_HASH: args_hash(kwargs.get("inputs") or input_str),
            },
        )
        record.set(semconv.INPUT, _content(kwargs.get("inputs") or input_str))

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        self._end(run_id, getattr(output, "content", output))

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._end(run_id, error=error)

    # retrievers
    def on_retriever_start(
        self,
        serialized: dict[str, Any] | None,
        query: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        record = self._start(
            run_id,
            parent_run_id,
            self._name(serialized, kwargs, "retriever"),
            "retrieval",
            {semconv.OPERATION: "retrieval"},
        )
        record.set(semconv.INPUT, query)

    def on_retriever_end(self, documents: Any, *, run_id: UUID, **kwargs: Any) -> None:
        record = self._spans.get(run_id)
        if record is not None:
            record.set(semconv.RETRIEVED_DOCS, len(documents))
        self._end(run_id, [getattr(d, "page_content", str(d))[:200] for d in documents])

    def on_retriever_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._end(run_id, error=error)
