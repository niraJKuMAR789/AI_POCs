"""Windowed anomaly detectors for agentic workloads.

Each detector turns a window of traces/spans into a metric and compares it with a rolling baseline of
previous windows for the same group (project, model or tool). Baselines skip windows already
flagged for that metric, so a sustained incident doesn't become the new normal.

  continuous metrics: robust z-score (median/MAD) AND a minimum relative change
  rates (errors, loops, sensitive data): exact binomial tail vs the pooled baseline rate
  counts (throughput): Poisson tail vs the baseline mean
  novelty: a model or tool never seen in the project before
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from argus.detect.stats import binom_sf, poisson_sf, quantile, robust_z
from argus.models import Anomaly, Span, TraceSummary

WINDOW_NS = 10 * 60 * 10**9


@dataclass
class Window:
    start_ns: int
    end_ns: int
    traces: list[TraceSummary]
    spans: list[Span]


@dataclass
class Config:
    baseline_windows: int = 18
    min_baseline: int = 6
    min_traces: int = 8
    z_threshold: float = 4.0
    min_rel_change: float = 0.5
    p_threshold: float = 1e-4
    loop_repeat: int = 4
    warmup_windows: int = 6  # novelty detection starts after this many windows


def severity_for(score: float) -> str:
    return "critical" if score >= 12 else "high" if score >= 7 else "medium"


@dataclass
class _History:
    """Per (detector, group, metric) series of past window values that were NOT anomalous."""

    values: dict[tuple[str, str, str], list[Any]] = field(default_factory=lambda: defaultdict(list))

    def baseline(self, key: tuple[str, str, str], n: int) -> list[Any]:
        return self.values[key][-n:]

    def add(self, key: tuple[str, str, str], value: Any) -> None:
        self.values[key].append(value)


class DetectorSuite:
    """Stateful: feed windows in time order with `observe`; returns anomalies for each window."""

    def __init__(self, project_id: str, config: Config | None = None) -> None:
        self.project_id = project_id
        self.cfg = config or Config()
        self.history = _History()
        self.seen_models: set[str] = set()
        self.seen_tools: set[str] = set()
        self.windows_seen = 0

    # --- primitives -------------------------------------------------------------------------

    def _continuous(
        self,
        w: Window,
        detector: str,
        group: str,
        metric: str,
        value: float | None,
        details: dict[str, Any] | None = None,
    ) -> Anomaly | None:
        if value is None:
            return None
        key = (detector, group, metric)
        base = self.history.baseline(key, self.cfg.baseline_windows)
        anomaly = None
        if len(base) >= self.cfg.min_baseline:
            z, center = robust_z(value, base)
            rel = (value - center) / center if center else 0.0
            if z >= self.cfg.z_threshold and rel >= self.cfg.min_rel_change:
                anomaly = Anomaly(
                    project_id=self.project_id,
                    detector=detector,
                    group_key=group,
                    metric=metric,
                    window_start_ns=w.start_ns,
                    window_end_ns=w.end_ns,
                    observed=value,
                    baseline=center,
                    score=round(z, 2),
                    severity=severity_for(z),
                    details={"relative_change": round(rel, 3), **(details or {})},
                )
        if anomaly is None:
            self.history.add(key, value)
        return anomaly

    def _rate(
        self, w: Window, detector: str, group: str, metric: str, k: int, n: int, details: dict[str, Any] | None = None
    ) -> Anomaly | None:
        if n == 0:
            return None
        key = (detector, group, metric)
        base = self.history.baseline(key, self.cfg.baseline_windows)
        anomaly = None
        if len(base) >= self.cfg.min_baseline:
            bk, bn = sum(b[0] for b in base), sum(b[1] for b in base)
            p0 = max((bk + 0.5) / (bn + 1), 0.002)  # smoothed: an all-zero baseline still yields a usable rate
            p_value = binom_sf(k, n, p0)
            if p_value < self.cfg.p_threshold and k >= 3 and k / n >= 2 * p0:
                score = min(30.0, -__import__("math").log10(max(p_value, 1e-300)))
                anomaly = Anomaly(
                    project_id=self.project_id,
                    detector=detector,
                    group_key=group,
                    metric=metric,
                    window_start_ns=w.start_ns,
                    window_end_ns=w.end_ns,
                    observed=round(k / n, 4),
                    baseline=round(p0, 4),
                    score=round(score, 2),
                    severity=severity_for(score),
                    details={"events": k, "total": n, "p_value": p_value, **(details or {})},
                )
        if anomaly is None:
            self.history.add(key, (k, n))
        return anomaly

    # --- the suite --------------------------------------------------------------------------------

    def observe(self, w: Window) -> list[Anomaly]:
        self.windows_seen += 1
        out: list[Anomaly | None] = []
        traces, spans, cfg = w.traces, w.spans, self.cfg
        enough = len(traces) >= cfg.min_traces
        by_root: dict[str, list[TraceSummary]] = defaultdict(list)
        for t in traces:
            by_root[t.root_name].append(t)
        for root, ts in by_root.items():
            group = f"agent:{root}"
            out.append(self._throughput(w, group, len(ts)))
            if len(ts) < cfg.min_traces:
                continue
            out.append(
                self._continuous(
                    w, "latency", group, "trace_duration_p95_ms", quantile([t.duration_ms for t in ts], 0.95)
                )
            )
            out.append(self._continuous(w, "cost", group, "cost_per_trace_usd", sum(t.cost_usd for t in ts) / len(ts)))
            out.append(
                self._continuous(
                    w, "steps", group, "spans_per_trace_p95", quantile([float(t.span_count) for t in ts], 0.95)
                )
            )
            out.append(
                self._rate(w, "error_rate", group, "trace_error_rate", sum(t.error_count > 0 for t in ts), len(ts))
            )
            out.append(
                self._rate(
                    w,
                    "tool_loop",
                    group,
                    "looping_trace_rate",
                    sum(t.max_tool_repeat >= cfg.loop_repeat for t in ts),
                    len(ts),
                    {"repeat_threshold": cfg.loop_repeat},
                )
            )
            out.append(
                self._rate(
                    w, "sensitive_data", group, "traces_with_sensitive_data", sum(t.pii_spans > 0 for t in ts), len(ts)
                )
            )
            llm_spans = [s for s in spans if s.kind == "llm" and s.trace_id in {t.trace_id for t in ts}]
            if llm_spans:
                out.append(
                    self._continuous(
                        w,
                        "tokens",
                        group,
                        "input_tokens_per_llm_call",
                        sum(s.input_tokens for s in llm_spans) / len(llm_spans),
                    )
                )
        if enough:
            out += self._per_component(w, spans)
        out += self._novelty(w, spans)
        return [a for a in out if a is not None]

    def _throughput(self, w: Window, group: str, count: int) -> Anomaly | None:
        key = ("throughput", group, "traces_per_window")
        base = self.history.baseline(key, self.cfg.baseline_windows)
        anomaly = None
        if len(base) >= self.cfg.min_baseline:
            lam = max(sum(base) / len(base), 1.0)
            p_value = poisson_sf(count, lam)
            if p_value < self.cfg.p_threshold and count >= 2 * lam:
                score = min(30.0, -__import__("math").log10(max(p_value, 1e-300)))
                anomaly = Anomaly(
                    project_id=self.project_id,
                    detector="throughput",
                    group_key=group,
                    metric="traces_per_window",
                    window_start_ns=w.start_ns,
                    window_end_ns=w.end_ns,
                    observed=count,
                    baseline=round(lam, 2),
                    score=round(score, 2),
                    severity="medium",
                    details={"p_value": p_value},
                )
        if anomaly is None:
            self.history.add(key, count)
        return anomaly

    def _per_component(self, w: Window, spans: list[Span]) -> list[Anomaly | None]:
        out: list[Anomaly | None] = []
        by_model: dict[str, list[Span]] = defaultdict(list)
        by_tool: dict[str, list[Span]] = defaultdict(list)
        for s in spans:
            if s.kind == "llm" and s.model:
                by_model[s.model].append(s)
            elif s.kind == "tool" and s.tool_name:
                by_tool[s.tool_name].append(s)
        for model, ss in by_model.items():
            if len(ss) >= self.cfg.min_traces:
                # Latency per output token isolates provider slowness from longer answers.
                per_token = [s.duration_ms / max(1, s.output_tokens) for s in ss]
                out.append(
                    self._continuous(
                        w,
                        "llm_latency",
                        f"model:{model}",
                        "ms_per_output_token_p50",
                        quantile(per_token, 0.5),
                        {"calls": len(ss)},
                    )
                )
        for tool, ss in by_tool.items():
            if len(ss) >= self.cfg.min_traces:
                out.append(
                    self._rate(
                        w,
                        "tool_error_rate",
                        f"tool:{tool}",
                        "tool_error_rate",
                        sum(s.status == "error" for s in ss),
                        len(ss),
                    )
                )
        return out

    def _novelty(self, w: Window, spans: list[Span]) -> list[Anomaly | None]:
        out: list[Anomaly | None] = []
        warm = self.windows_seen > self.cfg.warmup_windows
        for attr, seen, label in (("model", self.seen_models, "new_model"), ("tool_name", self.seen_tools, "new_tool")):
            values = {getattr(s, attr) for s in spans if getattr(s, attr)}
            for value in sorted(values - seen):
                if warm:
                    first = min((s for s in spans if getattr(s, attr) == value), key=lambda s: s.start_ns)
                    out.append(
                        Anomaly(
                            project_id=self.project_id,
                            detector=label,
                            group_key=f"{attr}:{value}",
                            metric="first_seen",
                            window_start_ns=w.start_ns,
                            window_end_ns=w.end_ns,
                            observed=1,
                            baseline=0,
                            score=5.0,
                            severity="medium",
                            details={"value": value, "first_trace": first.trace_id, "first_span": first.span_id},
                        )
                    )
                seen.add(value)
        return out


DETECTOR_FAMILIES: dict[str, Callable[..., Any]] = {}
