"""Synthetic agentic workload with labeled, injected incidents.

Three fictional apps emit realistic span trees (agent → retrieval/tools/LLM calls) through the real
OTLP path. Incidents are injected with known start, end, affected project and root cause, which
makes detection and root-cause analysis measurable. All data is synthetic.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from typing import Any

from argus import semconv
from argus.models import ChangeEvent
from argus.sdk.otlp import encode
from argus.sdk.tracer import SpanRecord, args_hash

MIN = 60 * 10**9
SUPER, NANO, EMBED = (
    "nvidia/nemotron-3-super-120b-a12b",
    "nvidia/nemotron-3-nano-30b-a3b",
    "nvidia/llama-nemotron-embed-1b-v2",
)
UNAPPROVED = "deepseek-ai/deepseek-v3.2"
APPS = ("support-bot", "research-agent", "claims-agent")


@dataclass(frozen=True)
class InjectedIncident:
    id: str
    # provider_degradation | tool_failure | prompt_regression | agent_regression | model_change | data_leak |
    # traffic_surge
    kind: str
    projects: tuple[str, ...]
    start_min: int
    end_min: int
    cause_key: str  # attribute dimension that explains it, e.g. "model", "tool_name", "prompt_version"
    cause_value: str
    expected_detectors: tuple[str, ...]
    description: str


def default_incidents(total_min: int) -> list[InjectedIncident]:
    """Incidents spread over the run, each after a clean baseline period."""
    at = lambda frac: int(total_min * frac)  # noqa: E731
    return [
        InjectedIncident(
            "I1",
            "provider_degradation",
            APPS,
            at(0.30),
            at(0.30) + 45,
            "model",
            SUPER,
            ("latency", "llm_latency"),
            "NIM endpoint for Nemotron Super slows ~5x",
        ),
        InjectedIncident(
            "I2",
            "tool_failure",
            ("support-bot",),
            at(0.45),
            at(0.45) + 30,
            "tool_name",
            "search_kb",
            ("error_rate", "tool_error_rate"),
            "knowledge-base search returns 503s",
        ),
        InjectedIncident(
            "I3",
            "prompt_regression",
            ("support-bot",),
            at(0.62),
            at(0.62) + 90,
            "prompt_version",
            "v2",
            ("cost", "tokens"),
            "prompt v2 inlines the whole FAQ: 3.5x input tokens",
        ),
        InjectedIncident(
            "I4",
            "agent_regression",
            ("research-agent",),
            at(0.52),
            at(0.52) + 60,
            "agent_version",
            "v1.4",
            ("tool_loop", "steps"),
            "agent v1.4 retries web_search with identical arguments",
        ),
        InjectedIncident(
            "I5",
            "model_change",
            ("claims-agent",),
            at(0.70),
            at(0.70) + 50,
            "model",
            UNAPPROVED,
            ("error_rate", "new_model"),
            "config switches claims-agent to an unapproved model",
        ),
        InjectedIncident(
            "I6",
            "data_leak",
            ("claims-agent",),
            at(0.38),
            at(0.38) + 40,
            "tool_name",
            "get_claim",
            ("sensitive_data",),
            "get_claim starts returning member email and SSN",
        ),
        InjectedIncident(
            "I7",
            "traffic_surge",
            ("support-bot",),
            at(0.85),
            at(0.85) + 30,
            "traffic",
            "surge",
            ("throughput",),
            "4x traffic (marketing email); service stays healthy",
        ),
    ]


@dataclass
class Simulator:
    total_min: int = 12 * 60
    start_ns: int = 1_767_225_600 * 10**9  # 2026-01-01T00:00Z (simulated clock)
    rate_per_min: float = 2.5
    seed: int = 7
    intensity: float = 1.0  # incident magnitude: 1.0 = obvious, ~0.3 = subtle
    incidents: list[InjectedIncident] | None = None  # None = the default incident schedule; [] = none

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)
        if self.incidents is None:
            self.incidents = default_incidents(self.total_min)

    def active(self, project: str, minute: float, kind: str) -> bool:
        return any(
            i.kind == kind and project in i.projects and i.start_min <= minute < i.end_min for i in self.incidents or []
        )

    def changes(self) -> dict[str, list[ChangeEvent]]:
        """Deploy/config events: the true causes, plus unrelated noise deploys."""
        out: dict[str, list[ChangeEvent]] = {p: [] for p in APPS}
        ts = lambda m: self.start_ns + int(m * MIN)  # noqa: E731
        for i in self.incidents or []:
            if i.kind == "prompt_regression":
                out["support-bot"] += [
                    ChangeEvent(
                        ts_ns=ts(i.start_min - 1),
                        kind="prompt_change",
                        service="support-bot",
                        description="Deploy prompt template v2 (inline FAQ context)",
                        attributes={"prompt_version": "v2"},
                    ),
                    ChangeEvent(
                        ts_ns=ts(i.end_min),
                        kind="prompt_change",
                        service="support-bot",
                        description="Roll back prompt template to v1",
                        attributes={"prompt_version": "v1"},
                    ),
                ]
            elif i.kind == "agent_regression":
                out["research-agent"] += [
                    ChangeEvent(
                        ts_ns=ts(i.start_min - 2),
                        kind="deploy",
                        service="research-agent",
                        description="Release research-agent v1.4 (new retry policy)",
                        attributes={"agent_version": "v1.4"},
                    ),
                    ChangeEvent(
                        ts_ns=ts(i.end_min),
                        kind="deploy",
                        service="research-agent",
                        description="Roll back research-agent to v1.3",
                        attributes={"agent_version": "v1.3"},
                    ),
                ]
            elif i.kind == "model_change":
                out["claims-agent"] += [
                    ChangeEvent(
                        ts_ns=ts(i.start_min - 1),
                        kind="model_change",
                        service="claims-agent",
                        description=f"Switch review model to {UNAPPROVED}",
                        attributes={"model": UNAPPROVED},
                    ),
                    ChangeEvent(
                        ts_ns=ts(i.end_min),
                        kind="model_change",
                        service="claims-agent",
                        description=f"Revert review model to {SUPER}",
                        attributes={"model": SUPER},
                    ),
                ]
        for project in APPS:  # noise: harmless deploys that a naive "last change" heuristic would blame
            for m in sorted(self.rng.sample(range(60, self.total_min - 30), 3)):
                out[project].append(
                    ChangeEvent(
                        ts_ns=ts(m),
                        kind="deploy",
                        service=project,
                        description=f"Routine dependency bump #{m}",
                        attributes={},
                    )
                )
        return out

    # --- span construction ------------------------------------------------------------------

    def _ids(self) -> tuple[str, str]:
        return f"{self.rng.getrandbits(128):032x}", f"{self.rng.getrandbits(64):016x}"

    def _llm(
        self,
        trace_id: str,
        parent: str,
        t: int,
        project: str,
        minute: float,
        model: str,
        in_tok: int,
        out_tok: int,
        name: str = "chat",
        prompt_version: str | None = None,
    ) -> tuple[SpanRecord, int]:
        latency_ms = (250 + 1.8 * out_tok + 0.04 * in_tok) * self.rng.lognormvariate(0, 0.25)
        if model == SUPER and self.active(project, minute, "provider_degradation"):
            latency_ms *= (1 + 4 * self.intensity) * self.rng.uniform(0.8, 1.2)
        record = SpanRecord(
            name=f"{name} {model}",
            kind="llm",
            trace_id=trace_id,
            span_id=self._ids()[1],
            parent_span_id=parent,
            start_ns=t,
            end_ns=t + int(latency_ms * 1e6),
            attributes={
                semconv.SPAN_KIND: "llm",
                semconv.OPERATION: "chat",
                semconv.SYSTEM: "nvidia_nim",
                semconv.REQUEST_MODEL: model,
                semconv.RESPONSE_MODEL: model,
                semconv.INPUT_TOKENS: in_tok,
                semconv.OUTPUT_TOKENS: out_tok,
                semconv.PROMPT_VERSION: prompt_version,
                semconv.INPUT: "[system + context]",
                semconv.OUTPUT: "[model answer]",
            },
        )
        return record, record.end_ns or t

    def _tool(
        self,
        trace_id: str,
        parent: str,
        t: int,
        tool: str,
        args: dict[str, Any],
        ms: float,
        error: str | None = None,
        output: str = "ok",
    ) -> tuple[SpanRecord, int]:
        end = t + int(ms * self.rng.lognormvariate(0, 0.3) * 1e6)
        record = SpanRecord(
            name=tool,
            kind="tool",
            trace_id=trace_id,
            span_id=self._ids()[1],
            parent_span_id=parent,
            start_ns=t,
            end_ns=end,
            attributes={
                semconv.SPAN_KIND: "tool",
                semconv.OPERATION: "execute_tool",
                semconv.TOOL_NAME: tool,
                semconv.TOOL_ARGS_HASH: args_hash(args),
                semconv.INPUT: json.dumps(args),
                semconv.OUTPUT: output,
            },
        )
        if error:
            record.status, record.status_message = "error", f"{error}: upstream failure"
            record.attributes[semconv.ERROR_TYPE] = error
        return record, end

    def trace(self, project: str, minute: float) -> list[SpanRecord]:
        rng = self.rng
        trace_id, root_id = self._ids()
        t0 = self.start_ns + int(minute * MIN)
        spans: list[SpanRecord] = []
        t = t0 + int(rng.uniform(2, 10) * 1e6)
        version = "v1.4" if project == "research-agent" and self.active(project, minute, "agent_regression") else "v1.3"
        if project == "support-bot":
            prompt = "v2" if self.active(project, minute, "prompt_regression") else "v1"
            q = f"q{rng.randint(1, 5000)}"
            error = (
                "ServiceUnavailable"
                if self.active(project, minute, "tool_failure") and rng.random() < 0.55 * self.intensity
                else None
            )
            if error is None and rng.random() < 0.004:
                error = "Timeout"  # background noise
            s, t = self._tool(trace_id, root_id, t, "search_kb", {"query": q}, 140, error)
            spans.append(s)
            in_tok = int(rng.gauss(1400, 200) * (1 + 2.5 * self.intensity if prompt == "v2" else 1))
            s, t = self._llm(
                trace_id,
                root_id,
                t,
                project,
                minute,
                SUPER,
                max(200, in_tok),
                int(rng.gauss(260, 60)),
                prompt_version=prompt,
            )
            spans.append(s)
            if rng.random() < 0.1:
                s, t = self._tool(trace_id, root_id, t, "create_ticket", {"q": q}, 220)
                spans.append(s)
        elif project == "research-agent":
            s, t = self._llm(trace_id, root_id, t, project, minute, SUPER, int(rng.gauss(900, 120)), 180, "plan")
            spans.append(s)
            looping = version == "v1.4" and rng.random() < 0.35 * self.intensity
            topic = f"topic-{rng.randint(1, 900)}"
            for i in range(rng.randint(7, 10) if looping else rng.randint(2, 4)):
                query = topic if looping else f"{topic}-{i}"
                s, t = self._tool(trace_id, root_id, t, "web_search", {"query": query}, 450)
                spans.append(s)
                if not looping:
                    s, t = self._tool(trace_id, root_id, t, "fetch_page", {"url": f"https://example.org/{query}"}, 300)
                    spans.append(s)
                s, t = self._llm(
                    trace_id, root_id, t, project, minute, NANO, int(rng.gauss(2200, 300)), 220, "summarize"
                )
                spans.append(s)
            s, t = self._llm(trace_id, root_id, t, project, minute, SUPER, int(rng.gauss(3000, 400)), 600, "write")
            spans.append(s)
        else:  # claims-agent
            leak = self.active(project, minute, "data_leak") and rng.random() < self.intensity
            claim_id = f"C{rng.randint(10_000, 99_999)}"
            output = (
                (
                    f"claim {claim_id}: member jane.doe{rng.randint(1, 99)}@example.com, SSN "
                    f"{rng.randint(100, 665)}-{rng.randint(10, 99)}-{rng.randint(1000, 9999)}"
                )
                if leak
                else f"claim {claim_id}: member M{rng.randint(1000, 9999)}, 2 lines"
            )
            s, t = self._tool(trace_id, root_id, t, "get_claim", {"claim_id": claim_id}, 90, output=output)
            spans.append(s)
            model = UNAPPROVED if self.active(project, minute, "model_change") else SUPER
            s, t = self._llm(trace_id, root_id, t, project, minute, model, int(rng.gauss(1800, 250)), 300, "review")
            if leak:
                s.attributes[semconv.INPUT] = f"Review this claim. Tool output: {output}"
            if model == UNAPPROVED and rng.random() < 0.3 * self.intensity:
                s.status, s.status_message = "error", "OutputParserException: response was not valid JSON"
                s.attributes[semconv.ERROR_TYPE] = "OutputParserException"
            spans.append(s)
            if rng.random() < 0.15:
                s, t = self._tool(trace_id, root_id, t, "flag_claim", {"claim_id": claim_id}, 60)
                spans.append(s)
        root = SpanRecord(
            name=project,
            kind="agent",
            trace_id=trace_id,
            span_id=root_id,
            parent_span_id=None,
            start_ns=t0,
            end_ns=t + int(rng.uniform(2, 8) * 1e6),
            attributes={
                semconv.SPAN_KIND: "agent",
                semconv.OPERATION: "invoke_agent",
                semconv.AGENT_NAME: project,
                semconv.AGENT_VERSION: version,
            },
        )
        if any(s.status == "error" for s in spans):
            root.status, root.status_message = "error", "child span failed"
        return [*spans, root]

    def run(self) -> dict[str, list[dict[str, Any]]]:
        """OTLP payloads per project, batched by simulated minute (as a real exporter would)."""
        out: dict[str, list[dict[str, Any]]] = {p: [] for p in APPS}
        for minute in range(self.total_min):
            for project in APPS:
                surge = 1 + 3 * self.intensity if self.active(project, minute, "traffic_surge") else 1
                rate = self.rate_per_min * surge
                n = _poisson(self.rng, rate)
                batch: list[tuple[SpanRecord, dict[str, Any]]] = []
                for _ in range(n):
                    for record in self.trace(project, minute + self.rng.random()):
                        batch.append((record, {semconv.SERVICE_NAME: project, semconv.DEPLOYMENT_ENV: "prod"}))
                if batch:
                    out[project].append(encode(batch))
        return out


def _poisson(rng: random.Random, lam: float) -> int:
    threshold, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= threshold:
            return k
        k += 1
