"""Ingest pipeline: OTLP → normalized spans → governance (PII, policy) → cost → store → trace rollups.

Spans of one trace can arrive across many batches and out of order (children usually end before
their parents), so trace rollups are recomputed for every trace touched by a batch.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from argus import semconv
from argus.governance.pii import redact
from argus.governance.policy import Policy, PolicyBook, Pricing, span_violations, trace_violations
from argus.ingest.otlp import decode
from argus.models import Span, TraceSummary, Violation
from argus.store.sqlite import Store


@dataclass
class IngestResult:
    accepted: int
    duplicates: int
    rejected: int
    violations: int
    traces_updated: int


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


_EXCEPTION_RE = re.compile(r"^([A-Za-z_][\w.]*(?:Error|Exception|Timeout|Unavailable|Failure)?)\s*:")


def _exception_name(message: str) -> str | None:
    """'TimeoutError: upstream ...' -> 'TimeoutError'; free text like 'child span failed' -> None."""
    match = _EXCEPTION_RE.match(message or "")
    return match.group(1) if match else None


def normalize(raw: dict[str, Any], project_id: str) -> Span | None:
    if not raw["trace_id"] or not raw["span_id"]:
        return None
    a = dict(raw["attributes"])
    operation = a.get(semconv.OPERATION)
    kind = a.get(semconv.SPAN_KIND) or semconv.OPERATION_TO_KIND.get(str(operation), "chain")
    if kind not in semconv.KINDS:
        kind = "chain"
    pop_text = lambda *keys: next((str(a.pop(k)) for k in keys if a.get(k) is not None), None)  # noqa: E731
    return Span(
        project_id=project_id,
        trace_id=raw["trace_id"],
        span_id=raw["span_id"],
        parent_span_id=raw["parent_span_id"],
        name=raw["name"] or "span",
        kind=kind,
        service=str(raw["resource"].get(semconv.SERVICE_NAME, "")),
        start_ns=raw["start_ns"],
        end_ns=max(raw["end_ns"], raw["start_ns"]),
        status=raw["status"],
        status_message=raw["status_message"][:500],
        error_type=a.get(semconv.ERROR_TYPE) or _exception_name(raw["status_message"])
        if raw["status"] == "error"
        else a.get(semconv.ERROR_TYPE),
        model=a.get(semconv.RESPONSE_MODEL) or a.get(semconv.REQUEST_MODEL),
        provider=a.get(semconv.SYSTEM),
        tool_name=a.get(semconv.TOOL_NAME),
        tool_args_hash=a.get(semconv.TOOL_ARGS_HASH),
        agent_name=a.get(semconv.AGENT_NAME),
        agent_version=a.get(semconv.AGENT_VERSION),
        prompt_version=a.get(semconv.PROMPT_VERSION),
        input_tokens=_int(a.get(semconv.INPUT_TOKENS)),
        output_tokens=_int(a.get(semconv.OUTPUT_TOKENS)),
        input_text=pop_text(semconv.INPUT, "gen_ai.prompt", "input.value"),
        output_text=pop_text(semconv.OUTPUT, "gen_ai.completion", "output.value"),
        attributes=a,
    )


def apply_content_policy(span: Span, policy: Policy) -> Span:
    """Scan and redact content and string attributes; enforce capture/block policy before storage."""
    found: Counter[str] = Counter()
    for field in ("input_text", "output_text"):
        text = getattr(span, field)
        if text:
            redacted, counts = redact(text)
            found.update(counts)
            if policy.pii == "redact":
                setattr(span, field, redacted)
    for key, value in list(span.attributes.items()):
        if isinstance(value, str):
            redacted, counts = redact(value)
            if counts:
                found.update(counts)
                if policy.pii != "allow":
                    span.attributes[key] = redacted
    span.pii_found = dict(found)
    if not policy.capture_content or (policy.pii == "block" and found):
        span.input_text = span.output_text = None
    return span


def summarize(project_id: str, trace_id: str, spans: list[Span]) -> TraceSummary:
    by_id = {s.span_id: s for s in spans}
    roots = [s for s in spans if not s.parent_span_id or s.parent_span_id not in by_id]
    true_roots = [s for s in spans if not s.parent_span_id]
    root = min(true_roots or roots, key=lambda s: s.start_ns)

    def depth(s: Span) -> int:
        d, seen = 1, set()
        while s.parent_span_id and s.parent_span_id in by_id and s.span_id not in seen:
            seen.add(s.span_id)
            s, d = by_id[s.parent_span_id], d + 1
        return d

    calls = Counter((s.tool_name, s.tool_args_hash) for s in spans if s.kind == "tool")
    return TraceSummary(
        project_id=project_id,
        trace_id=trace_id,
        root_name=root.name,
        service=root.service,
        start_ns=min(s.start_ns for s in spans),
        end_ns=max(s.end_ns for s in spans),
        duration_ms=(max(s.end_ns for s in spans) - min(s.start_ns for s in spans)) / 1e6,
        span_count=len(spans),
        llm_calls=sum(s.kind == "llm" for s in spans),
        tool_calls=sum(s.kind == "tool" for s in spans),
        input_tokens=sum(s.input_tokens for s in spans),
        output_tokens=sum(s.output_tokens for s in spans),
        cost_usd=round(sum(s.cost_usd for s in spans), 8),
        error_count=sum(s.status == "error" for s in spans),
        max_depth=max(depth(s) for s in spans),
        max_tool_repeat=max(calls.values(), default=0),
        pii_spans=sum(bool(s.pii_found) for s in spans),
        has_root=bool(true_roots),
        models=sorted({s.model for s in spans if s.model}),
        tools=sorted({s.tool_name for s in spans if s.tool_name}),
    )


class Ingestor:
    def __init__(self, store: Store, policies: PolicyBook, pricing: Pricing) -> None:
        self.store, self.policies, self.pricing = store, policies, pricing

    def policy_for(self, project_id: str) -> Policy:
        project = self.store.project(project_id)
        return self.policies.for_project(project["name"] if project else "")

    def ingest(self, project_id: str, payload: dict[str, Any]) -> IngestResult:
        policy = self.policy_for(project_id)
        spans, rejected = [], 0
        for raw in decode(payload):
            span = normalize(raw, project_id)
            if span is None:
                rejected += 1
                continue
            span.cost_usd = self.pricing.cost(span.model, span.input_tokens, span.output_tokens)
            spans.append(apply_content_policy(span, policy))
        inserted = self.store.insert_spans(spans)
        violations: list[Violation] = [v for s in spans for v in span_violations(s, policy)]
        touched = sorted({s.trace_id for s in spans})
        for trace_id in touched:
            trace_spans = self.store.spans_for_trace(project_id, trace_id)
            summary = summarize(project_id, trace_id, trace_spans)
            self.store.upsert_trace(summary)
            if summary.has_root:
                violations += trace_violations(summary, policy)
        violations += self._budget(project_id, spans, policy)
        self.store.add_violations(violations)
        return IngestResult(
            accepted=inserted,
            duplicates=len(spans) - inserted,
            rejected=rejected,
            violations=len(violations),
            traces_updated=len(touched),
        )

    def _budget(self, project_id: str, spans: list[Span], policy: Policy) -> list[Violation]:
        out = []
        for day in {datetime.fromtimestamp(s.start_ns / 1e9, UTC).date() for s in spans}:
            day_ns = int(datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp()) * 10**9
            spent = self.store.daily_cost(project_id, day_ns)
            if spent > policy.daily_budget_usd:
                out.append(
                    Violation(
                        project_id=project_id,
                        trace_id=f"budget:{day}",
                        span_id=None,
                        policy="daily_budget",
                        severity="high",
                        ts_ns=day_ns,
                        message=f"spend ${spent:.2f} on {day} exceeds budget ${policy.daily_budget_usd:.2f}",
                    )
                )
        return out
