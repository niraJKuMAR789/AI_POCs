"""Argus instrumentation for Atlas, Quill and Aegis, added from the outside.

None of the three apps imports Argus. Each one depends on a small provider protocol (`generate` /
`structured` / `stream`, `complete`, `complete_json`) and a few internal components (reranker, SQL guard,
executor, rules engine). Wrapping those objects at their seams gives full traces (agent → LLM calls →
tools) without touching application code. The same approach works for any app built against interfaces.

Token counts: the provider protocols return text or parsed objects, not usage blocks, so tokens are
estimated at ~4 characters per token and flagged with `argus.usage.estimated`.
"""

from __future__ import annotations

import contextvars
import json
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import contextmanager
from typing import Any

from argus import semconv
from argus.sdk.tracer import SpanRecord, span
from pydantic import BaseModel

ESTIMATED = "argus.usage.estimated"
PURPOSE = "argus.llm.purpose"

# The deployment currently serving traffic for an app (agent/prompt version). Set by the workload when a
# release or rollback happens; stamped on every span so Argus can contrast versions.
deployment: contextvars.ContextVar[dict[str, str] | None] = contextvars.ContextVar("mission_deployment", default=None)


def current_deployment() -> dict[str, str]:
    return deployment.get() or {}


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def messages_text(messages: Any) -> str:
    parts = []
    for m in messages or []:
        content = m.get("content", "") if isinstance(m, dict) else str(m)
        if isinstance(content, list):
            content = " ".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
        parts.append(f"{m.get('role', '') if isinstance(m, dict) else ''}: {content}")
    return "\n".join(parts)


def _render(value: Any) -> str:
    if isinstance(value, BaseModel):
        return value.model_dump_json()
    return value if isinstance(value, str) else json.dumps(value, default=str)


def _stamp(s: SpanRecord) -> None:
    dep = current_deployment()
    s.set(semconv.AGENT_VERSION, dep.get("agent_version"))
    s.set(semconv.PROMPT_VERSION, dep.get("prompt_version"))


class TracedLLM:
    """Proxy for any of the three LLM protocols; every call becomes a GenAI `chat` span."""

    METHODS = frozenset({"generate", "structured", "complete", "complete_json"})

    def __init__(self, inner: Any, *, model: str, system: str) -> None:
        self._inner, self._model, self._system = inner, model, system

    def __getattr__(self, name: str) -> Any:
        target = getattr(self._inner, name)
        if name in self.METHODS:
            return self._wrap(name, target)
        if name == "stream":
            return self._wrap_stream(target)
        return target

    def _attributes(self, purpose: str) -> dict[str, Any]:
        return {
            semconv.OPERATION: "chat",
            semconv.SYSTEM: self._system,
            semconv.REQUEST_MODEL: self._model,
            semconv.RESPONSE_MODEL: self._model,
            PURPOSE: purpose,
            ESTIMATED: True,
        }

    def _wrap(self, method: str, target: Callable[..., Any]) -> Callable[..., Any]:
        async def call(messages: Any, *args: Any, **kwargs: Any) -> Any:
            schema = args[0] if args else kwargs.get("schema")
            purpose = schema.__name__ if isinstance(schema, type) else method
            prompt = messages_text(messages)
            with span(f"chat {self._model}", kind="llm", attributes=self._attributes(purpose)) as s:
                _stamp(s)
                s.set(semconv.INPUT, prompt[:8000]).set(semconv.INPUT_TOKENS, estimate_tokens(prompt))
                result = await target(messages, *args, **kwargs)
                text = _render(result)
                s.set(semconv.OUTPUT, text[:8000]).set(semconv.OUTPUT_TOKENS, estimate_tokens(text))
                return result

        return call

    def _wrap_stream(self, target: Callable[..., AsyncIterator[str]]) -> Callable[..., AsyncIterator[str]]:
        async def stream(messages: Any, *args: Any, **kwargs: Any) -> AsyncIterator[str]:
            prompt = messages_text(messages)
            with span(f"chat {self._model}", kind="llm", attributes=self._attributes("stream")) as s:
                _stamp(s)
                s.set(semconv.INPUT, prompt[:8000]).set(semconv.INPUT_TOKENS, estimate_tokens(prompt))
                parts: list[str] = []
                async for token in target(messages, *args, **kwargs):
                    parts.append(token)
                    yield token
                text = "".join(parts)
                s.set(semconv.OUTPUT, text[:8000]).set(semconv.OUTPUT_TOKENS, estimate_tokens(text))

        return stream


@contextmanager
def tool_span(name: str, args: dict[str, Any] | None = None) -> Iterator[SpanRecord]:
    from argus.sdk.tracer import args_hash

    attributes = {semconv.OPERATION: "execute_tool", semconv.TOOL_NAME: name, semconv.TOOL_ARGS_HASH: args_hash(args)}
    with span(name, kind="tool", attributes=attributes) as s:
        _stamp(s)
        if args is not None:
            s.set(semconv.INPUT, _render(args)[:8000])
        yield s


@contextmanager
def agent_span(app: str, question: str, **attributes: Any) -> Iterator[SpanRecord]:
    attrs = {semconv.OPERATION: "invoke_agent", semconv.AGENT_NAME: app, **attributes}
    with span(app, kind="agent", attributes=attrs) as s:
        _stamp(s)
        s.set(semconv.INPUT, question[:8000])
        yield s


class TracedReranker:
    """Atlas's reranker (NeMo Retriever reranking NIM in production) as a `rerank` tool span."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.name = getattr(inner, "name", "reranker")

    async def rerank(self, query: str, texts: list[str], top_n: int) -> list[tuple[int, float]]:
        with tool_span("rerank", {"query": query, "candidates": len(texts), "top_n": top_n}) as s:
            ranked = await self._inner.rerank(query, texts, top_n)
            s.set(semconv.OUTPUT, json.dumps([[i, round(score, 4)] for i, score in ranked]))
            return ranked


class TracedEmbedder:
    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def embed_query(self, text: str) -> list[float]:
        with tool_span("embed_query", {"text": text}):
            return list(await self._inner.embed_query(text))


class TracedGuard:
    """Quill's AST guard. A rejected query is the guard doing its job, so the span stays `ok` and the
    verdict goes into attributes; Argus sees the repair loop as extra steps and tokens."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def check(self, sql: str, policy: Any, context: dict[str, Any]) -> Any:
        with tool_span("sql_guard", {"sql": sql, "role": getattr(policy, "name", "")}) as s:
            verdict = self._inner.check(sql, policy, context)
            s.set("quill.guard.ok", bool(verdict.ok))
            s.set("quill.guard.rewrites", len(getattr(verdict, "rewrites", []) or []))
            s.set(semconv.OUTPUT, verdict.feedback() if not verdict.ok else (verdict.sql or ""))
            return verdict


class TracedDatabase:
    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def query(self, sql: str, *, max_rows: int) -> Any:
        with tool_span("execute_sql", {"sql": sql, "max_rows": max_rows}) as s:
            result = self._inner.query(sql, max_rows=max_rows)
            s.set("quill.rows", len(result.rows)).set(semconv.OUTPUT, f"{len(result.rows)} rows")
            return result


def instrument_rules_engine(engine: Any) -> None:
    """Aegis compiles its LangGraph around the engine instance, so patch the bound method in place."""
    original = engine.evaluate

    def evaluate(claim: Any) -> list[Any]:
        with tool_span("rules_engine", {"claim_id": claim.claim_id, "lines": len(claim.lines)}) as s:
            findings = list(original(claim))
            s.set("aegis.findings", len(findings))
            s.set(semconv.OUTPUT, json.dumps([f.rule_id for f in findings]))
            return findings

    engine.evaluate = evaluate
