"""Backend domain models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Span(BaseModel):
    project_id: str
    trace_id: str
    span_id: str
    parent_span_id: str | None = None
    name: str
    kind: str = "chain"
    service: str = ""
    start_ns: int
    end_ns: int
    status: str = "ok"
    status_message: str = ""
    error_type: str | None = None
    model: str | None = None
    provider: str | None = None
    tool_name: str | None = None
    tool_args_hash: str | None = None
    agent_name: str | None = None
    agent_version: str | None = None
    prompt_version: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    input_text: str | None = None
    output_text: str | None = None
    pii_found: dict[str, int] = Field(default_factory=dict)
    attributes: dict[str, Any] = Field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return (self.end_ns - self.start_ns) / 1e6


class TraceSummary(BaseModel):
    project_id: str
    trace_id: str
    root_name: str
    service: str
    start_ns: int
    end_ns: int
    duration_ms: float
    span_count: int
    llm_calls: int
    tool_calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    error_count: int
    max_depth: int
    max_tool_repeat: int
    pii_spans: int = 0
    has_root: bool
    models: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)


class Violation(BaseModel):
    id: int | None = None
    project_id: str
    trace_id: str | None
    span_id: str | None
    policy: str
    severity: str
    message: str
    ts_ns: int


class ChangeEvent(BaseModel):
    """A deploy / config change (model swap, prompt version, agent release). Used for RCA correlation."""

    id: int | None = None
    project_id: str = ""
    ts_ns: int
    kind: str = Field(description="deploy | config | model_change | prompt_change | incident_note")
    service: str = ""
    description: str
    attributes: dict[str, Any] = Field(default_factory=dict)


class Anomaly(BaseModel):
    id: int | None = None
    project_id: str
    detector: str
    group_key: str
    metric: str
    window_start_ns: int
    window_end_ns: int
    observed: float
    baseline: float
    score: float
    severity: str
    details: dict[str, Any] = Field(default_factory=dict)
    incident_id: int | None = None


class Incident(BaseModel):
    id: int | None = None
    project_id: str
    group_key: str
    title: str
    status: str = "open"
    severity: str
    start_ns: int
    end_ns: int
    detectors: list[str] = Field(default_factory=list)
    anomaly_count: int = 0
