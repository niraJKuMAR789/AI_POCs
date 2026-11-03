"""RCA report schema and evidence validation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from argus.investigate.tools import InvestigationTools

Category = Literal[
    "provider_degradation",
    "tool_failure",
    "prompt_regression",
    "agent_regression",
    "model_change",
    "data_leak",
    "traffic_surge",
    "unknown",
]


class Evidence(BaseModel):
    kind: Literal["trace", "change", "anomaly", "component", "attribute", "project"]
    ref: str = Field(description="trace id, change id, anomaly id, component name, 'dimension=value' or project")
    detail: str = ""


class RCAReport(BaseModel):
    summary: str
    category: Category
    cause_dimension: str = Field(
        description="model | tool_name | prompt_version | agent_version | error_type | traffic"
    )
    cause_value: str
    evidence: list[Evidence] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)


def validate(report: RCAReport, tools: InvestigationTools, incident_id: int) -> list[str]:
    """Every piece of evidence must exist in this project's data; otherwise the report is flagged."""
    errors = []
    inc = tools.incident(incident_id)
    anomaly_ids = {str(a.id) for a in tools.store.anomalies(tools.project_id, incident_id)}
    change_ids = {str(c["change_id"]) for c in tools.recent_changes(incident_id, lookback_min=24 * 60)}
    contrast = tools.attribute_contrast(incident_id, "all", top=200)["top"]
    attributes = {r["attribute"] for r in contrast}
    components = {c["component"] for c in tools.component_breakdown(incident_id, top=200)}
    for ev in report.evidence:
        if ev.kind == "trace":
            rows = tools.store.query(
                "SELECT 1 FROM traces WHERE project_id = ? AND trace_id = ?", (tools.project_id, ev.ref)
            )
            if not rows:
                errors.append(f"trace {ev.ref} does not exist")
        elif ev.kind == "change" and ev.ref not in change_ids:
            errors.append(f"change {ev.ref} is not a change event near this incident")
        elif ev.kind == "anomaly" and ev.ref not in anomaly_ids:
            errors.append(f"anomaly {ev.ref} is not part of incident {incident_id}")
        elif ev.kind == "attribute" and ev.ref not in attributes:
            errors.append(f"attribute {ev.ref} does not occur during the incident")
        elif ev.kind == "component" and ev.ref not in components:
            errors.append(f"component {ev.ref} does not occur during the incident")
    if report.cause_dimension != "traffic" and f"{report.cause_dimension}={report.cause_value}" not in attributes:
        errors.append(f"cause {report.cause_dimension}={report.cause_value} does not occur during the incident")
    if not report.evidence:
        errors.append("report cites no evidence")
    del inc
    return errors
