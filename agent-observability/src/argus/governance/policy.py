"""Per-project governance policies, pricing, and the checks applied at ingest."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from argus.models import Span, TraceSummary, Violation


class Policy(BaseModel):
    capture_content: bool = True
    pii: str = "redact"
    allowed_models: list[str] = Field(default_factory=list)
    forbidden_tools: list[str] = Field(default_factory=list)
    max_tokens_per_call: int = 32000
    max_cost_per_trace_usd: float = 0.5
    max_steps_per_trace: int = 60
    daily_budget_usd: float = 100.0
    retention_days: int = 30
    content_retention_days: int = 7


class PolicyBook:
    def __init__(self, default: Policy, overrides: dict[str, dict]) -> None:
        self.default, self.overrides = default, overrides

    @classmethod
    def load(cls, path: Path) -> PolicyBook:
        raw = yaml.safe_load(path.read_text()) or {}
        return cls(Policy.model_validate(raw.get("default", {})), raw.get("projects", {}) or {})

    def for_project(self, name: str) -> Policy:
        return self.default.model_copy(update=self.overrides.get(name, {}))


@dataclass(frozen=True)
class Pricing:
    default: tuple[float, float]
    models: dict[str, tuple[float, float]]

    @classmethod
    def load(cls, path: Path) -> Pricing:
        raw = yaml.safe_load(path.read_text())
        d = raw["default"]
        return cls(
            (d["input"], d["output"]), {m: (v["input"], v["output"]) for m, v in (raw.get("models") or {}).items()}
        )

    def cost(self, model: str | None, input_tokens: int, output_tokens: int) -> float:
        if not model or not (input_tokens or output_tokens):
            return 0.0
        rate_in, rate_out = self.models.get(model, self.default)
        return (input_tokens * rate_in + output_tokens * rate_out) / 1_000_000


def span_violations(span: Span, policy: Policy) -> list[Violation]:
    out = []

    def add(name: str, severity: str, message: str) -> None:
        out.append(
            Violation(
                project_id=span.project_id,
                trace_id=span.trace_id,
                span_id=span.span_id,
                policy=name,
                severity=severity,
                message=message,
                ts_ns=span.start_ns,
            )
        )

    if span.kind == "llm" and span.model and policy.allowed_models and span.model not in policy.allowed_models:
        add("unapproved_model", "high", f"model {span.model} is not on the approved list")
    if span.kind == "tool" and span.tool_name in policy.forbidden_tools:
        add("forbidden_tool", "critical", f"forbidden tool {span.tool_name} was executed")
    if span.input_tokens + span.output_tokens > policy.max_tokens_per_call:
        total = span.input_tokens + span.output_tokens
        add("token_limit", "medium", f"{total} tokens exceed the per-call limit of {policy.max_tokens_per_call}")
    if span.pii_found:
        detail = ", ".join(f"{k}x{v}" for k, v in sorted(span.pii_found.items()))
        severity = "critical" if {"API_KEY", "PRIVATE_KEY", "BEARER_TOKEN"} & set(span.pii_found) else "high"
        action = {"redact": "redacted", "block": "content dropped", "allow": "stored as-is"}[policy.pii]
        add("sensitive_data", severity, f"sensitive data in span content ({detail}); {action}")
    return out


def trace_violations(trace: TraceSummary, policy: Policy) -> list[Violation]:
    out = []
    if trace.cost_usd > policy.max_cost_per_trace_usd:
        out.append(
            Violation(
                project_id=trace.project_id,
                trace_id=trace.trace_id,
                span_id=None,
                policy="trace_cost_limit",
                severity="medium",
                ts_ns=trace.start_ns,
                message=f"trace cost ${trace.cost_usd:.4f} exceeds ${policy.max_cost_per_trace_usd:.2f}",
            )
        )
    if trace.span_count > policy.max_steps_per_trace:
        out.append(
            Violation(
                project_id=trace.project_id,
                trace_id=trace.trace_id,
                span_id=None,
                policy="step_limit",
                severity="high",
                ts_ns=trace.start_ns,
                message=f"{trace.span_count} steps exceed the limit of {policy.max_steps_per_trace}",
            )
        )
    return out
