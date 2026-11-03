"""Analysis tools for incident investigation (used by the deterministic RCA and the LLM agent).

All tools are read-only and project-scoped. They return metrics, ids and attribute values, never
prompt/completion content, so the investigator never needs content-read access.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from argus.detect.engine import FAMILIES, family
from argus.detect.stats import quantile
from argus.models import Incident, Span, TraceSummary
from argus.store.sqlite import Store

MIN_NS = 60 * 10**9
DIMENSIONS = ("model", "tool_name", "error_type", "agent_version", "prompt_version", "provider")
OUTLIER_FOR_DETECTOR = {
    "latency": "slow",
    "llm_latency": "slow",
    "steps": "loops",
    "tool_loop": "loops",
    "cost": "expensive",
    "tokens": "expensive",
    "error_rate": "errors",
    "tool_error_rate": "errors",
    "new_model": "errors",
    "new_tool": "errors",
    "sensitive_data": "sensitive",
    "throughput": "all",
}


@dataclass
class Slice:
    traces: list[TraceSummary]
    spans: dict[str, list[Span]]  # trace_id -> spans


class InvestigationTools:
    def __init__(self, store: Store, project_id: str, baseline_min: int = 120) -> None:
        self.store, self.project_id, self.baseline_min = store, project_id, baseline_min

    # --- data access ----------------------------------------------------------------------------

    def incident(self, incident_id: int) -> Incident:
        inc = self.store.incident(self.project_id, incident_id)
        if inc is None:
            raise KeyError(f"incident {incident_id} not found in this project")
        return inc

    def _slice(self, start: int, end: int, root: str | None = None) -> Slice:
        traces = self.store.traces(self.project_id, start, end, root_name=root)
        spans: dict[str, list[Span]] = defaultdict(list)
        ids = [t.trace_id for t in traces]
        for i in range(0, len(ids), 500):
            chunk = ids[i : i + 500]
            for r in self.store.query(
                f"SELECT * FROM spans WHERE project_id = ? AND trace_id IN ({','.join('?' * len(chunk))})",
                (self.project_id, *chunk),
            ):
                spans[r["trace_id"]].append(self.store._span(r))
        return Slice(traces, spans)

    def _baseline_range(self, inc: Incident) -> tuple[int, int]:
        """The window before the incident, skipping back past any other incident that overlaps it."""
        end = inc.start_ns
        for other in sorted(self.store.incidents(self.project_id), key=lambda i: i.end_ns, reverse=True):
            if other.id != inc.id and other.start_ns < end <= other.end_ns + 10 * MIN_NS:
                end = other.start_ns
        return end - self.baseline_min * MIN_NS, end

    def _lead_detector(self, inc: Incident) -> str:
        primary = [d for d in inc.detectors if d not in ("new_model", "new_tool", "latency", "throughput")]
        return (primary or inc.detectors or ["latency"])[0]

    # --- tools ----------------------------------------------------------------------------------

    def incident_overview(self, incident_id: int) -> dict[str, Any]:
        inc = self.incident(incident_id)
        anomalies = self.store.anomalies(self.project_id, incident_id)
        return {
            "incident": inc.model_dump(),
            "duration_min": (inc.end_ns - inc.start_ns) // MIN_NS,
            "anomalies": [
                {
                    "id": a.id,
                    "detector": a.detector,
                    "group": a.group_key,
                    "metric": a.metric,
                    "observed": a.observed,
                    "baseline": a.baseline,
                    "score": a.score,
                    "minute_offset": (a.window_start_ns - inc.start_ns) // MIN_NS,
                }
                for a in anomalies
            ],
        }

    def compare_windows(self, incident_id: int) -> dict[str, Any]:
        inc = self.incident(incident_id)
        cur, base = self._slice(inc.start_ns, inc.end_ns), self._slice(*self._baseline_range(inc))

        def stats(sl: Slice, minutes: float) -> dict[str, float]:
            ts = sl.traces
            n = max(1, len(ts))
            return {
                "traces": len(ts),
                "traces_per_min": round(len(ts) / max(minutes, 1), 2),
                "p50_ms": round(quantile([t.duration_ms for t in ts], 0.5), 1),
                "p95_ms": round(quantile([t.duration_ms for t in ts], 0.95), 1),
                "error_rate": round(sum(t.error_count > 0 for t in ts) / n, 4),
                "cost_per_trace": round(sum(t.cost_usd for t in ts) / n, 6),
                "spans_per_trace": round(sum(t.span_count for t in ts) / n, 2),
                "looping_rate": round(sum(t.max_tool_repeat >= 4 for t in ts) / n, 4),
                "sensitive_rate": round(sum(t.pii_spans > 0 for t in ts) / n, 4),
            }

        b0, b1 = self._baseline_range(inc)
        return {
            "incident": stats(cur, (inc.end_ns - inc.start_ns) / MIN_NS),
            "baseline": stats(base, (b1 - b0) / MIN_NS),
        }

    def component_breakdown(self, incident_id: int, top: int = 8) -> list[dict[str, Any]]:
        """Per model / tool: latency, error rate and tokens, incident vs baseline, biggest change first."""
        inc = self.incident(incident_id)
        cur, base = self._slice(inc.start_ns, inc.end_ns), self._slice(*self._baseline_range(inc))

        def components(sl: Slice) -> dict[str, list[Span]]:
            out: dict[str, list[Span]] = defaultdict(list)
            for spans in sl.spans.values():
                for s in spans:
                    if s.kind == "llm" and s.model:
                        out[f"llm:{s.model}"].append(s)
                    elif s.kind == "tool" and s.tool_name:
                        out[f"tool:{s.tool_name}"].append(s)
            return out

        cur_c, base_c = components(cur), components(base)
        rows = []
        for name, ss in cur_c.items():
            bs = base_c.get(name, [])

            def m(spans: list[Span]) -> dict[str, float]:
                n = max(1, len(spans))
                return {
                    "calls": len(spans),
                    "p50_ms": round(quantile([s.duration_ms for s in spans], 0.5), 1),
                    "error_rate": round(sum(s.status == "error" for s in spans) / n, 4),
                    "avg_input_tokens": round(sum(s.input_tokens for s in spans) / n, 1),
                }

            now, before = m(ss), m(bs)
            ratios = {k: round(now[k] / before[k], 2) if before[k] else None for k in ("p50_ms", "avg_input_tokens")}
            change = max(
                abs((now["p50_ms"] / before["p50_ms"]) - 1) if before["p50_ms"] else 1.0,
                abs(now["error_rate"] - before["error_rate"]) * 10,
                abs((now["avg_input_tokens"] / before["avg_input_tokens"]) - 1) if before["avg_input_tokens"] else 0.0,
            )
            rows.append(
                {
                    "component": name,
                    "incident": now,
                    "baseline": before,
                    "new_in_incident": not bs,
                    "latency_ratio": ratios["p50_ms"],
                    "tokens_ratio": ratios["avg_input_tokens"],
                    "change_score": round(change, 3),
                }
            )
        return sorted(rows, key=lambda r: float(r["change_score"]), reverse=True)[:top]  # type: ignore[arg-type]

    def attribute_contrast(self, incident_id: int, outlier: str = "auto", top: int = 8) -> dict[str, Any]:
        """BubbleUp-style contrast: which attribute values are over-represented in outlier traces
        during the incident compared with traces in the baseline window."""
        inc = self.incident(incident_id)
        if outlier == "auto":
            outlier = OUTLIER_FOR_DETECTOR.get(self._lead_detector(inc), "all")
        cur, base = self._slice(inc.start_ns, inc.end_ns), self._slice(*self._baseline_range(inc))
        p95_base = quantile([t.duration_ms for t in base.traces], 0.95)
        cost_base = quantile([t.cost_usd for t in base.traces], 0.95)
        rules = {
            "errors": lambda t: t.error_count > 0,
            "slow": lambda t: t.duration_ms > p95_base,
            "loops": lambda t: t.max_tool_repeat >= 4,
            "sensitive": lambda t: t.pii_spans > 0,
            "expensive": lambda t: t.cost_usd > cost_base,
            "all": lambda t: True,
        }
        is_out = rules[outlier]
        out_traces = [t for t in cur.traces if is_out(t)]

        def values(sl: Slice, traces: list[TraceSummary]) -> Counter[str]:
            c: Counter[str] = Counter()
            for t in traces:
                pairs = set()
                for s in sl.spans.get(t.trace_id, []):
                    # Contrast the spans that carry the symptom, not every span in an outlier trace.
                    relevant = True
                    if outlier == "errors":
                        relevant = s.status == "error" and s.kind != "agent"
                    elif outlier == "sensitive":
                        relevant = bool(s.pii_found)
                    for d in DIMENSIONS:
                        v = getattr(s, d)
                        if v and relevant:
                            pairs.add(f"{d}={v}")
                c.update(pairs)
            return c

        oc, bc = values(cur, out_traces), values(base, base.traces)
        n_out, n_base = max(1, len(out_traces)), max(1, len(base.traces))
        rows = [
            {
                "attribute": k,
                "outlier_share": round(v / n_out, 3),
                "baseline_share": round(bc[k] / n_base, 3),
                "lift": round(v / n_out - bc[k] / n_base, 3),
            }
            for k, v in oc.items()
        ]
        rows.sort(key=lambda r: (r["lift"], r["outlier_share"]), reverse=True)
        return {
            "outlier_definition": outlier,
            "outlier_traces": len(out_traces),
            "baseline_traces": len(base.traces),
            "top": rows[:top],
        }

    def recent_changes(self, incident_id: int, lookback_min: int = 120) -> list[dict[str, Any]]:
        inc = self.incident(incident_id)
        events = self.store.changes(self.project_id, inc.start_ns - lookback_min * MIN_NS, inc.end_ns)
        return [
            {
                "change_id": e.id,
                "kind": e.kind,
                "description": e.description,
                "attributes": e.attributes,
                "minutes_before_incident": round((inc.start_ns - e.ts_ns) / MIN_NS, 1),
            }
            for e in events
        ]

    def sample_traces(self, incident_id: int, outlier: str = "auto", k: int = 3) -> list[dict[str, Any]]:
        inc = self.incident(incident_id)
        contrast = self.attribute_contrast(incident_id, outlier)
        rule = contrast["outlier_definition"]
        traces = self.store.traces(self.project_id, inc.start_ns, inc.end_ns)
        key = {
            "slow": lambda t: -t.duration_ms,
            "errors": lambda t: -t.error_count,
            "loops": lambda t: -t.max_tool_repeat,
            "sensitive": lambda t: -t.pii_spans,
            "expensive": lambda t: -t.cost_usd,
        }.get(rule, lambda t: 0)
        return [
            {
                "trace_id": t.trace_id,
                "duration_ms": round(t.duration_ms, 1),
                "spans": t.span_count,
                "errors": t.error_count,
                "cost_usd": round(t.cost_usd, 6),
                "max_tool_repeat": t.max_tool_repeat,
                "models": t.models,
                "tools": t.tools,
            }
            for t in sorted(traces, key=key)[:k]
        ]

    def get_trace(self, trace_id: str) -> list[dict[str, Any]]:
        spans = self.store.spans_for_trace(self.project_id, trace_id)
        if not spans:
            raise KeyError(f"trace {trace_id} not found in this project")
        return [
            {
                "span_id": s.span_id,
                "parent": s.parent_span_id,
                "name": s.name,
                "kind": s.kind,
                "model": s.model,
                "tool": s.tool_name,
                "duration_ms": round(s.duration_ms, 1),
                "status": s.status,
                "error_type": s.error_type,
                "input_tokens": s.input_tokens,
                "sensitive": s.pii_found,
            }
            for s in spans
        ]

    def other_projects_affected(self, incident_id: int) -> list[dict[str, Any]]:
        """Same-org incidents overlapping in time with the same kind of problem: a shared dependency."""
        inc = self.incident(incident_id)
        me = self.store.project(self.project_id)
        assert me is not None
        fam = {family(d) for d in inc.detectors}
        out = []
        for p in self.store.projects():
            if p["id"] == self.project_id or p["org"] != me["org"]:
                continue
            for other in self.store.incidents(p["id"]):
                if (
                    other.start_ns < inc.end_ns
                    and inc.start_ns < other.end_ns
                    and fam & {family(d) for d in other.detectors}
                ):
                    out.append(
                        {
                            "project": p["name"],
                            "incident_id": other.id,
                            "title": other.title,
                            "detectors": other.detectors,
                        }
                    )
        return out


__all__ = ["DIMENSIONS", "FAMILIES", "InvestigationTools"]
