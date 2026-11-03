"""Detection engine: windows the trace store, runs the detector suite, groups anomalies into incidents."""

from __future__ import annotations

from dataclasses import dataclass, field

from argus.detect.detectors import WINDOW_NS, Config, DetectorSuite, Window
from argus.models import Anomaly, Incident
from argus.store.sqlite import Store

TITLES = {
    "latency": "Latency regression",
    "llm_latency": "LLM provider slowdown",
    "cost": "Cost per trace spike",
    "tokens": "Prompt token inflation",
    "steps": "Runaway agent steps",
    "error_rate": "Error rate spike",
    "tool_error_rate": "Tool failure spike",
    "tool_loop": "Agent tool-call loop",
    "sensitive_data": "Sensitive data in agent content",
    "throughput": "Traffic surge",
    "new_model": "New model in use",
    "new_tool": "New tool in use",
}
SEVERITY_ORDER = {"medium": 1, "high": 2, "critical": 3}
FAMILIES = {
    "performance": {"latency", "llm_latency", "steps", "tool_loop", "throughput"},
    "spend": {"cost", "tokens"},
    "errors": {"error_rate", "tool_error_rate", "new_model", "new_tool"},
    "data": {"sensitive_data"},
}


def family(detector: str) -> str:
    return next((f for f, members in FAMILIES.items() if detector in members), detector)


@dataclass
class DetectionEngine:
    store: Store
    config: Config = field(default_factory=Config)
    merge_gap_windows: int = 1
    _suites: dict[str, DetectorSuite] = field(default_factory=dict)
    _cursor: dict[str, int] = field(default_factory=dict)

    def run(self, project_id: str, until_ns: int | None = None) -> list[Anomaly]:
        """Process all complete windows up to `until_ns` (default: the latest data). Idempotent per window."""
        span_range = self.store.time_range(project_id)
        if span_range is None:
            return []
        first, last = span_range
        until = until_ns if until_ns is not None else last
        suite = self._suites.setdefault(project_id, DetectorSuite(project_id, self.config))
        cursor = self._cursor.get(project_id, first - first % WINDOW_NS)
        found: list[Anomaly] = []
        while cursor + WINDOW_NS <= until:
            window = self._window(project_id, cursor, cursor + WINDOW_NS)
            for anomaly in suite.observe(window):
                anomaly.id = self.store.add_anomaly(anomaly)
                if anomaly.id is not None:
                    found.append(anomaly)
            cursor += WINDOW_NS
        self._cursor[project_id] = cursor
        self._group(project_id, found)
        return found

    def _window(self, project_id: str, start: int, end: int) -> Window:
        traces = self.store.traces(project_id, start, end)
        ids = [t.trace_id for t in traces]
        spans = []
        for i in range(0, len(ids), 500):
            chunk = ids[i : i + 500]
            rows = self.store.query(
                f"SELECT * FROM spans WHERE project_id = ? AND trace_id IN ({','.join('?' * len(chunk))})",
                (project_id, *chunk),
            )
            spans += [self.store._span(r) for r in rows]
        return Window(start, end, traces, spans)

    def _group(self, project_id: str, anomalies: list[Anomaly]) -> None:
        """Attach anomalies to an open incident that overlaps in time, else open a new one."""
        gap = self.merge_gap_windows * WINDOW_NS
        open_incidents = self.store.incidents(project_id, status="open")
        for a in sorted(anomalies, key=lambda a: a.window_start_ns):
            # Concurrent anomalies are one incident; back-to-back ones only if they are the same kind of problem.
            target = next(
                (
                    i
                    for i in open_incidents
                    if a.window_start_ns < i.end_ns
                    or (a.window_start_ns <= i.end_ns + gap and family(a.detector) in {family(d) for d in i.detectors})
                ),
                None,
            )
            if target is None:
                target = Incident(
                    project_id=project_id,
                    group_key=a.group_key,
                    title="",
                    severity=a.severity,
                    start_ns=a.window_start_ns,
                    end_ns=a.window_end_ns,
                )
                open_incidents.append(target)
            target.end_ns = max(target.end_ns, a.window_end_ns)
            target.detectors = sorted(set(target.detectors) | {a.detector})
            target.anomaly_count += 1
            if SEVERITY_ORDER[a.severity] > SEVERITY_ORDER[target.severity]:
                target.severity = a.severity
            lead = max(target.detectors, key=lambda d: (d not in ("new_model", "new_tool", "throughput"), d))
            target.title = f"{TITLES.get(lead, lead)} ({', '.join(target.detectors)})"
            target.id = self.store.save_incident(target)
            assert a.id is not None
            self.store.set_anomaly_incident(a.id, target.id)
        # Resolve incidents with no anomaly in the last two windows processed.
        cursor = self._cursor.get(project_id, 0)
        for inc in self.store.incidents(project_id, status="open"):
            if inc.end_ns + 2 * WINDOW_NS <= cursor:
                inc.status = "resolved"
                self.store.save_incident(inc)
