"""LLM investigator: a framework-free tool-calling loop over the analysis tools.

The model (NVIDIA NIM, OpenAI-compatible function calling) decides which analyses to run, then calls
`submit_report`. The loop has a step budget, tool errors are returned to the model rather than
raised, and the final report's evidence is validated against the data. The investigator is itself
instrumented with the Argus SDK, so its own runs show up as traces ("dogfooding").
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from argus import semconv
from argus.investigate.report import RCAReport, validate
from argus.investigate.tools import InvestigationTools
from argus.sdk.instrument.openai import instrument_openai
from argus.sdk.tracer import Tracer, _content, span, use_tracer

SYSTEM = """You are an SRE investigating an incident in an agentic AI application, using observability data.
Work like a careful on-call engineer:
1. Start with incident_overview, then compare_windows.
2. Use component_breakdown to see which model or tool changed, attribute_contrast to see which attribute values
   are over-represented in the bad traces, recent_changes for deploys, other_projects_affected for shared causes.
3. Distinguish symptom from cause: a slow agent is a symptom; the slow model endpoint or a new release is a cause.
   Correlated deploys are only causal if their attributes match the outlier traces.
4. Call submit_report exactly once. Every evidence item must reference ids or attribute values returned by tools.
Categories: provider_degradation, tool_failure, prompt_regression, agent_regression, model_change, data_leak,
traffic_surge, unknown. Prefer "unknown" with low confidence over guessing."""


def _schema(name: str, description: str, props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": props, "required": required},
        },
    }


INCIDENT = {"incident_id": {"type": "integer"}}
TOOL_SCHEMAS = [
    _schema(
        "incident_overview",
        "Incident and its anomalies (detector, metric, observed vs baseline).",
        INCIDENT,
        ["incident_id"],
    ),
    _schema(
        "compare_windows", "Aggregate stats during the incident vs the preceding baseline.", INCIDENT, ["incident_id"]
    ),
    _schema(
        "component_breakdown",
        "Per model/tool latency, error rate and tokens: incident vs baseline.",
        INCIDENT,
        ["incident_id"],
    ),
    _schema(
        "attribute_contrast",
        "Attribute values over-represented in outlier traces (errors, slow, loops, sensitive, expensive, all).",
        {**INCIDENT, "outlier": {"type": "string", "default": "auto"}},
        ["incident_id"],
    ),
    _schema("recent_changes", "Deploy/config change events before and during the incident.", INCIDENT, ["incident_id"]),
    _schema("sample_traces", "Representative outlier traces (ids and summaries).", INCIDENT, ["incident_id"]),
    _schema("get_trace", "Span tree of one trace (no prompt content).", {"trace_id": {"type": "string"}}, ["trace_id"]),
    _schema(
        "other_projects_affected",
        "Overlapping incidents of the same kind in other projects of the org.",
        INCIDENT,
        ["incident_id"],
    ),
    {
        "type": "function",
        "function": {
            "name": "submit_report",
            "description": "Submit the final root-cause report.",
            "parameters": RCAReport.model_json_schema(),
        },
    },
]


@dataclass
class InvestigationResult:
    report: RCAReport | None
    validation_errors: list[str]
    steps: int
    transcript: list[dict[str, Any]] = field(default_factory=list)


class LLMInvestigator:
    def __init__(self, client: Any, model: str, *, max_steps: int = 10, tracer: Tracer | None = None) -> None:
        self.client = instrument_openai(client)
        self.model, self.max_steps, self.tracer = model, max_steps, tracer

    async def investigate(self, tools: InvestigationTools, incident_id: int) -> InvestigationResult:
        if self.tracer is None:
            return await self._investigate(tools, incident_id)
        with use_tracer(self.tracer):
            return await self._investigate(tools, incident_id)

    async def _investigate(self, tools: InvestigationTools, incident_id: int) -> InvestigationResult:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Investigate incident {incident_id}."},
        ]
        with span(
            "investigate_incident",
            kind="agent",
            attributes={
                semconv.OPERATION: "invoke_agent",
                semconv.AGENT_NAME: "argus-investigator",
                "argus.incident.id": incident_id,
            },
        ) as root:
            for step in range(1, self.max_steps + 1):
                force = step == self.max_steps
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=TOOL_SCHEMAS,
                    temperature=0.0,
                    tool_choice={"type": "function", "function": {"name": "submit_report"}} if force else "auto",
                )
                message = response.choices[0].message
                messages.append(message.model_dump(exclude_none=True))
                if not message.tool_calls:
                    messages.append({"role": "user", "content": "Use the tools, then call submit_report."})
                    continue
                for call in message.tool_calls:
                    name, raw_args = call.function.name, call.function.arguments or "{}"
                    if name == "submit_report":
                        try:
                            report = RCAReport.model_validate_json(raw_args)
                        except ValidationError as exc:
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": call.id,
                                    "content": f"Invalid report: {exc.errors()[:3]}. Fix and resubmit.",
                                }
                            )
                            continue
                        errors = validate(report, tools, incident_id)
                        if errors:
                            report.confidence = min(report.confidence, 0.3)
                        root.set("argus.investigation.category", report.category)
                        return InvestigationResult(report, errors, step, messages)
                    with span(
                        name, kind="tool", attributes={semconv.OPERATION: "execute_tool", semconv.TOOL_NAME: name}
                    ) as tool_span:
                        try:
                            result = getattr(tools, name)(**json.loads(raw_args))
                            content = json.dumps(result, default=str)[:6000]
                        except Exception as exc:  # report tool errors to the model; don't crash the loop
                            tool_span.record_error(exc)
                            content = json.dumps({"error": f"{type(exc).__name__}: {exc}"})
                        tool_span.set(semconv.OUTPUT, _content(content, 2000))
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": content})
        return InvestigationResult(None, ["step budget exhausted without a report"], self.max_steps, messages)
