"""Deterministic root-cause analysis: rules over the analysis tools. A transparent baseline for the
LLM investigator, and the fallback when no model is configured."""

from __future__ import annotations

from argus.investigate.report import Category, Evidence, RCAReport
from argus.investigate.tools import InvestigationTools

ACTIONS = {
    "provider_degradation": [
        "Fail over to a secondary model/endpoint for the affected model",
        "Check the provider status page and NIM endpoint health",
    ],
    "tool_failure": ["Check the failing tool's upstream dependency", "Add retries with backoff or a circuit breaker"],
    "prompt_regression": ["Roll back the prompt change", "Add a token-budget check to prompt CI"],
    "agent_regression": ["Roll back the agent release", "Add a max-repeat guard on identical tool calls"],
    "model_change": ["Revert to the approved model", "Require approval for model configuration changes"],
    "data_leak": ["Mask sensitive fields in the tool's output", "Review access logs for exposed content"],
    "traffic_surge": ["Confirm the surge is expected", "Check autoscaling headroom and rate limits"],
}


def _relative(minutes_before: float) -> str:
    if minutes_before >= 0:
        return f"{minutes_before:g} min before the incident window"
    return f"{-minutes_before:g} min into the incident window"


def investigate(tools: InvestigationTools, incident_id: int) -> RCAReport:
    inc = tools.incident(incident_id)
    detectors = set(inc.detectors)
    changes = tools.recent_changes(incident_id)
    components = tools.component_breakdown(incident_id)
    contrast = tools.attribute_contrast(incident_id)
    samples = tools.sample_traces(incident_id, k=2)
    evidence = [Evidence(kind="trace", ref=s["trace_id"], detail="representative outlier trace") for s in samples]
    top_attr = contrast["top"][0] if contrast["top"] else None

    def attr_cause(prefer: tuple[str, ...]) -> tuple[str, str] | None:
        for row in contrast["top"]:
            dim, value = row["attribute"].split("=", 1)
            if dim in prefer and row["lift"] > 0.1:
                evidence.append(
                    Evidence(
                        kind="attribute",
                        ref=row["attribute"],
                        detail=f"{row['outlier_share']:.0%} of outlier traces vs "
                        f"{row['baseline_share']:.0%} at baseline",
                    )
                )
                return dim, value
        return None

    def change_matching(dim: str, value: str) -> None:
        for c in changes:
            if str(c["attributes"].get(dim)) == value:
                evidence.append(
                    Evidence(
                        kind="change",
                        ref=str(c["change_id"]),
                        detail=f"{c['description']} ({_relative(c['minutes_before_incident'])})",
                    )
                )

    def version_cause() -> tuple[Category, tuple[str, str]]:
        found = attr_cause(("prompt_version", "model", "agent_version"))
        if not found:
            return "unknown", ("unknown", "unknown")
        by_dimension: dict[str, Category] = {"prompt_version": "prompt_regression", "model": "model_change"}
        change_matching(*found)
        return by_dimension.get(found[0], "agent_regression"), found

    category: Category = "unknown"
    cause = ("unknown", "unknown")
    if "sensitive_data" in detectors:
        category, cause = "data_leak", attr_cause(("tool_name",)) or ("tool_name", "unknown")
    elif detectors & {"tool_loop", "steps"} and "tool_loop" in detectors:
        category, cause = "agent_regression", attr_cause(("agent_version",)) or ("agent_version", "unknown")
        change_matching(*cause)
    elif detectors & {"error_rate", "tool_error_rate", "new_model"}:
        # A model that is new in the failing spans explains errors better than whichever tool they surfaced in.
        # "New" means it served no calls before the incident, not merely that it had no errors back then.
        baseline_calls = {c["component"]: c["baseline"]["calls"] for c in components}
        new_model = next(
            (
                r
                for r in contrast["top"]
                if r["attribute"].startswith("model=")
                and r["lift"] > 0.5
                and r["baseline_share"] < 0.05
                and not baseline_calls.get(f"llm:{r['attribute'].split('=', 1)[1]}")
            ),
            None,
        )
        if new_model or "new_model" in detectors:
            found = attr_cause(("model",))
            if found:
                category, cause = "model_change", found
                change_matching(*found)
        if category == "unknown":
            # Localize by component: the model or tool whose error rate rose the most. Failing LLM calls on a
            # model that was already in use point at the provider, not at the agent.
            failing = sorted(
                (c for c in components if c["incident"]["error_rate"] - c["baseline"]["error_rate"] >= 0.05),
                key=lambda c: c["incident"]["error_rate"] - c["baseline"]["error_rate"],
                reverse=True,
            )
            if failing:
                top = failing[0]
                kind, name = top["component"].split(":", 1)
                evidence.append(
                    Evidence(
                        kind="component",
                        ref=top["component"],
                        detail=f"error rate {top['incident']['error_rate']:.0%} vs "
                        f"{top['baseline']['error_rate']:.0%} at baseline",
                    )
                )
                category = "provider_degradation" if kind == "llm" else "tool_failure"
                cause = ("model", name) if kind == "llm" else ("tool_name", name)
                change_matching(*cause)
            else:
                found = attr_cause(("tool_name",)) or attr_cause(("error_type",))
                if found:
                    category, cause = "tool_failure", found
    elif detectors & {"cost", "tokens"}:
        category, cause = version_cause()
    elif detectors & {"latency", "llm_latency"}:
        # The per-model latency detector already localizes the slow component; trust it first. Otherwise take
        # the component whose latency grew the most (a fixed "2x" cut-off misses moderate slowdowns).
        localized = [
            a.group_key.split(":", 1)[1]
            for a in tools.store.anomalies(tools.project_id, incident_id)
            if a.detector == "llm_latency"
        ]
        slowed = sorted(
            (c for c in components if (c["latency_ratio"] or 0) >= 1.3),
            key=lambda c: c["latency_ratio"] or 0,
            reverse=True,
        )
        slow = next((c for c in components if localized and c["component"] == f"llm:{localized[0]}"), None)
        slow = slow or (slowed[0] if slowed else None)
        if slow:
            evidence.append(
                Evidence(
                    kind="component", ref=slow["component"], detail=f"p50 latency {slow['latency_ratio']}x baseline"
                )
            )
            kind, name = slow["component"].split(":", 1)
            category = "provider_degradation" if kind == "llm" else "tool_failure"
            cause = ("model", name) if kind == "llm" else ("tool_name", name)
            change_matching(*cause)
            for other in tools.other_projects_affected(incident_id):
                evidence.append(Evidence(kind="project", ref=other["project"], detail=other["title"]))
    elif "steps" in detectors:
        # More steps per trace without a latency regression: usually a prompt or agent change (e.g. repair loops).
        category, cause = version_cause()
    elif "throughput" in detectors:
        category, cause = "traffic_surge", ("traffic", "surge")
    for a in tools.store.anomalies(tools.project_id, incident_id)[:3]:
        evidence.append(
            Evidence(kind="anomaly", ref=str(a.id), detail=f"{a.metric}: {a.observed:.4g} vs {a.baseline:.4g}")
        )
    confidence = 0.8 if category != "unknown" and cause[1] != "unknown" else 0.3
    summary = (
        (f"{inc.title}: most consistent with {category.replace('_', ' ')} ({cause[0]}={cause[1]}).")
        if category != "unknown"
        else f"{inc.title}: cause undetermined."
    )
    if top_attr is None and category == "unknown":
        summary += " No distinguishing attributes found."
    return RCAReport(
        summary=summary,
        category=category,
        cause_dimension=cause[0],
        cause_value=cause[1],
        evidence=evidence,
        recommended_actions=ACTIONS.get(category, ["Escalate to on-call"]),
        confidence=confidence,
    )
