"""Argus tracing SDK.

A small, dependency-free tracer for agentic apps. Spans nest automatically through `contextvars`
(so async tasks and threads started with `contextvars.copy_context()` keep their parent), carry
GenAI semantic-convention attributes, and are exported as OTLP/JSON in background batches.

    from argus.sdk import init, span, trace_agent, trace_tool

    init(endpoint="http://localhost:8003", api_key="argus_...", service="support-bot")

    @trace_agent("support-bot")
    async def handle(question): ...

    @trace_tool("search_kb")
    def search_kb(query): ...

    with span("rerank", kind="retrieval") as s:
        s.set(RETRIEVED_DOCS, 5)
"""

from __future__ import annotations

import contextvars
import functools
import hashlib
import inspect
import json
import os
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, TypeVar

from argus import semconv

F = TypeVar("F", bound=Callable[..., Any])
_current: contextvars.ContextVar[SpanRecord | None] = contextvars.ContextVar("argus_span", default=None)


def _hex(nbytes: int) -> str:
    return os.urandom(nbytes).hex()


@dataclass
class SpanRecord:
    name: str
    kind: str
    trace_id: str
    span_id: str
    parent_span_id: str | None
    start_ns: int
    end_ns: int | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"
    status_message: str = ""

    def set(self, key: str, value: Any) -> SpanRecord:
        if value is not None:
            self.attributes[key] = value
        return self

    def set_many(self, values: dict[str, Any]) -> SpanRecord:
        for key, value in values.items():
            self.set(key, value)
        return self

    def record_error(self, exc: BaseException) -> None:
        self.status = "error"
        self.status_message = f"{type(exc).__name__}: {exc}"[:500]
        self.attributes[semconv.ERROR_TYPE] = type(exc).__name__

    @property
    def duration_ms(self) -> float:
        return ((self.end_ns or time.time_ns()) - self.start_ns) / 1e6


class Tracer:
    def __init__(
        self,
        service: str,
        exporter: Any | None = None,
        *,
        capture_content: bool = True,
        resource: dict[str, Any] | None = None,
    ) -> None:
        self.service = service
        self.exporter = exporter
        self.capture_content = capture_content
        self.resource = {semconv.SERVICE_NAME: service, **(resource or {})}

    def start_span(
        self,
        name: str,
        *,
        kind: str = "chain",
        attributes: dict[str, Any] | None = None,
        parent: SpanRecord | None = None,
    ) -> SpanRecord:
        """Low-level API for callback-style integrations; pair with `end_span`."""
        parent = parent if parent is not None else _current.get()
        return SpanRecord(
            name=name,
            kind=kind,
            trace_id=parent.trace_id if parent else _hex(16),
            span_id=_hex(8),
            parent_span_id=parent.span_id if parent else None,
            start_ns=time.time_ns(),
            attributes={semconv.SPAN_KIND: kind, **(attributes or {})},
        )

    def end_span(self, record: SpanRecord) -> None:
        record.end_ns = time.time_ns()
        if not self.capture_content:
            record.attributes.pop(semconv.INPUT, None)
            record.attributes.pop(semconv.OUTPUT, None)
        if self.exporter is not None:
            self.exporter.export(record, self.resource)

    @contextmanager
    def span(self, name: str, *, kind: str = "chain", attributes: dict[str, Any] | None = None) -> Iterator[SpanRecord]:
        record = self.start_span(name, kind=kind, attributes=attributes)
        token = _current.set(record)
        try:
            yield record
        except BaseException as exc:
            record.record_error(exc)
            raise
        finally:
            _current.reset(token)
            self.end_span(record)


_tracer: Tracer = Tracer("unconfigured")
_override: contextvars.ContextVar[Tracer | None] = contextvars.ContextVar("argus_tracer", default=None)
_lock = threading.Lock()


def init(
    *,
    endpoint: str | None = None,
    api_key: str | None = None,
    service: str = "agent-app",
    capture_content: bool = True,
    resource: dict[str, Any] | None = None,
    exporter: Any | None = None,
) -> Tracer:
    """Configure the global tracer. Pass `exporter` directly (e.g. InMemoryExporter in tests)."""
    global _tracer
    from argus.sdk.exporter import HTTPExporter

    with _lock:
        if exporter is None and endpoint:
            exporter = HTTPExporter(endpoint, api_key=api_key)
        _tracer = Tracer(service, exporter, capture_content=capture_content, resource=resource)
    return _tracer


def get_tracer() -> Tracer:
    return _override.get() or _tracer


@contextmanager
def use_tracer(tracer: Tracer) -> Iterator[Tracer]:
    """Route spans in this context to `tracer` without touching the global one (libraries, multi-tenant)."""
    token = _override.set(tracer)
    try:
        yield tracer
    finally:
        _override.reset(token)


def current_span() -> SpanRecord | None:
    return _current.get()


@contextmanager
def span(name: str, *, kind: str = "chain", attributes: dict[str, Any] | None = None) -> Iterator[SpanRecord]:
    with get_tracer().span(name, kind=kind, attributes=attributes) as record:
        yield record


def _content(value: Any, limit: int = 8000) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text[:limit]


def args_hash(args: Any) -> str:
    """Stable hash of tool arguments: lets the backend detect agents repeating the same call."""
    return hashlib.sha256(json.dumps(args, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _decorator(name: str, kind: str, extra: Callable[[dict[str, Any]], dict[str, Any]]) -> Callable[[F], F]:
    def wrap(fn: F) -> F:
        signature = inspect.signature(fn)

        def bound(*a: Any, **kw: Any) -> dict[str, Any]:
            try:
                b = signature.bind_partial(*a, **kw)
                return {k: v for k, v in b.arguments.items() if k not in ("self", "cls")}
            except TypeError:
                return {}

        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_inner(*a: Any, **kw: Any) -> Any:
                arguments = bound(*a, **kw)
                with span(name, kind=kind, attributes=extra(arguments)) as s:
                    s.set(semconv.INPUT, _content(arguments))
                    result = await fn(*a, **kw)
                    s.set(semconv.OUTPUT, _content(result))
                    return result

            return async_inner  # type: ignore[return-value]

        @functools.wraps(fn)
        def inner(*a: Any, **kw: Any) -> Any:
            arguments = bound(*a, **kw)
            with span(name, kind=kind, attributes=extra(arguments)) as s:
                s.set(semconv.INPUT, _content(arguments))
                result = fn(*a, **kw)
                s.set(semconv.OUTPUT, _content(result))
                return result

        return inner  # type: ignore[return-value]

    return wrap


def trace_agent(name: str, *, version: str | None = None) -> Callable[[F], F]:
    return _decorator(
        name,
        "agent",
        lambda _: {semconv.OPERATION: "invoke_agent", semconv.AGENT_NAME: name, semconv.AGENT_VERSION: version},
    )


def trace_tool(name: str) -> Callable[[F], F]:
    return _decorator(
        name,
        "tool",
        lambda args: {
            semconv.OPERATION: "execute_tool",
            semconv.TOOL_NAME: name,
            semconv.TOOL_ARGS_HASH: args_hash(args),
        },
    )


def trace_retrieval(name: str) -> Callable[[F], F]:
    return _decorator(name, "retrieval", lambda _: {semconv.OPERATION: "retrieval"})
