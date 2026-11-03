"""SQLite storage: tenants and keys, spans, trace rollups, governance, change events, anomalies,
incidents, investigations and the content-access log. WAL mode; every query is project-scoped."""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import threading
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from argus.models import Anomaly, ChangeEvent, Incident, Span, TraceSummary, Violation

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (id TEXT PRIMARY KEY, org TEXT NOT NULL, name TEXT NOT NULL UNIQUE);
CREATE TABLE IF NOT EXISTS api_keys (
    key_hash TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id), name TEXT NOT NULL,
    scopes TEXT NOT NULL, created_ns INTEGER NOT NULL, revoked INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS spans (
    project_id TEXT NOT NULL, span_id TEXT NOT NULL, trace_id TEXT NOT NULL, parent_span_id TEXT,
    name TEXT, kind TEXT, service TEXT, start_ns INTEGER, end_ns INTEGER, duration_ms REAL, status TEXT,
    status_message TEXT, error_type TEXT, model TEXT, provider TEXT, tool_name TEXT, tool_args_hash TEXT,
    agent_name TEXT, agent_version TEXT, prompt_version TEXT, input_tokens INTEGER, output_tokens INTEGER,
    cost_usd REAL, input_text TEXT, output_text TEXT, pii_found TEXT, attributes TEXT,
    PRIMARY KEY (project_id, span_id));
CREATE INDEX IF NOT EXISTS idx_spans_trace ON spans(project_id, trace_id);
CREATE INDEX IF NOT EXISTS idx_spans_time ON spans(project_id, start_ns);
CREATE TABLE IF NOT EXISTS traces (
    project_id TEXT NOT NULL, trace_id TEXT NOT NULL, root_name TEXT, service TEXT, start_ns INTEGER,
    end_ns INTEGER, duration_ms REAL, span_count INTEGER, llm_calls INTEGER, tool_calls INTEGER,
    input_tokens INTEGER, output_tokens INTEGER, cost_usd REAL, error_count INTEGER, max_depth INTEGER,
    max_tool_repeat INTEGER, pii_spans INTEGER, has_root INTEGER, models TEXT, tools TEXT,
    PRIMARY KEY (project_id, trace_id));
CREATE INDEX IF NOT EXISTS idx_traces_time ON traces(project_id, start_ns);
CREATE TABLE IF NOT EXISTS violations (
    id INTEGER PRIMARY KEY, project_id TEXT, trace_id TEXT, span_id TEXT, policy TEXT, severity TEXT,
    message TEXT, ts_ns INTEGER, UNIQUE (project_id, trace_id, span_id, policy));
CREATE TABLE IF NOT EXISTS change_events (
    id INTEGER PRIMARY KEY, project_id TEXT, ts_ns INTEGER, kind TEXT, service TEXT, description TEXT,
    attributes TEXT);
CREATE TABLE IF NOT EXISTS anomalies (
    id INTEGER PRIMARY KEY, project_id TEXT, detector TEXT, group_key TEXT, metric TEXT,
    window_start_ns INTEGER, window_end_ns INTEGER, observed REAL, baseline REAL, score REAL, severity TEXT,
    details TEXT, incident_id INTEGER, UNIQUE (project_id, detector, group_key, metric, window_start_ns));
CREATE TABLE IF NOT EXISTS incidents (
    id INTEGER PRIMARY KEY, project_id TEXT, group_key TEXT, title TEXT, status TEXT, severity TEXT,
    start_ns INTEGER, end_ns INTEGER, detectors TEXT, anomaly_count INTEGER);
CREATE TABLE IF NOT EXISTS investigations (
    id INTEGER PRIMARY KEY, incident_id INTEGER, project_id TEXT, created_ns INTEGER, model TEXT,
    report TEXT, validation_errors TEXT);
CREATE TABLE IF NOT EXISTS content_access_log (
    id INTEGER PRIMARY KEY, ts_ns INTEGER, project_id TEXT, key_name TEXT, trace_id TEXT, purpose TEXT);
"""

SPAN_COLUMNS = [
    "project_id",
    "span_id",
    "trace_id",
    "parent_span_id",
    "name",
    "kind",
    "service",
    "start_ns",
    "end_ns",
    "duration_ms",
    "status",
    "status_message",
    "error_type",
    "model",
    "provider",
    "tool_name",
    "tool_args_hash",
    "agent_name",
    "agent_version",
    "prompt_version",
    "input_tokens",
    "output_tokens",
    "cost_usd",
    "input_text",
    "output_text",
    "pii_found",
    "attributes",
]


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


class Store:
    def __init__(self, path: Path | str) -> None:
        self.path = str(path)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.executescript(SCHEMA)
        self._lock = threading.RLock()

    def query(self, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, tuple(params)).fetchall()

    # --- tenants & keys ---------------------------------------------------------------------

    def create_project(self, name: str, org: str = "default") -> str:
        project_id = "prj_" + hashlib.sha256(f"{org}/{name}".encode()).hexdigest()[:12]
        with self._lock, self._conn:
            self._conn.execute("INSERT OR IGNORE INTO projects VALUES (?, ?, ?)", (project_id, org, name))
        return project_id

    def project(self, project_id: str) -> sqlite3.Row | None:
        rows = self.query("SELECT * FROM projects WHERE id = ?", (project_id,))
        return rows[0] if rows else None

    def projects(self) -> list[sqlite3.Row]:
        return self.query("SELECT * FROM projects ORDER BY name")

    def create_key(self, project_id: str, name: str, scopes: list[str]) -> str:
        key = "argus_" + secrets.token_urlsafe(24)
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO api_keys VALUES (?, ?, ?, ?, ?, 0)",
                (hash_key(key), project_id, name, ",".join(scopes), time.time_ns()),
            )
        return key  # shown once; only the hash is stored

    def resolve_key(self, key: str) -> sqlite3.Row | None:
        rows = self.query("SELECT * FROM api_keys WHERE key_hash = ? AND revoked = 0", (hash_key(key),))
        return rows[0] if rows else None

    # --- spans & traces ---------------------------------------------------------------------

    def insert_spans(self, spans: list[Span]) -> int:
        rows = []
        for s in spans:
            d = s.model_dump()
            d["duration_ms"] = s.duration_ms
            d["pii_found"] = json.dumps(s.pii_found)
            d["attributes"] = json.dumps(s.attributes, default=str)
            rows.append(tuple(d[c] for c in SPAN_COLUMNS))
        marks = ",".join("?" * len(SPAN_COLUMNS))
        with self._lock, self._conn:
            before = self._conn.total_changes
            self._conn.executemany(f"INSERT OR IGNORE INTO spans ({','.join(SPAN_COLUMNS)}) VALUES ({marks})", rows)
            return self._conn.total_changes - before

    def spans_for_trace(self, project_id: str, trace_id: str) -> list[Span]:
        rows = self.query(
            "SELECT * FROM spans WHERE project_id = ? AND trace_id = ? ORDER BY start_ns", (project_id, trace_id)
        )
        return [self._span(r) for r in rows]

    @staticmethod
    def _span(row: sqlite3.Row) -> Span:
        d = dict(row)
        d.pop("duration_ms", None)
        d["pii_found"] = json.loads(d["pii_found"] or "{}")
        d["attributes"] = json.loads(d["attributes"] or "{}")
        return Span.model_validate(d)

    def upsert_trace(self, t: TraceSummary) -> None:
        d = t.model_dump()
        d["models"], d["tools"] = json.dumps(t.models), json.dumps(t.tools)
        d["has_root"] = int(t.has_root)
        cols = list(d)
        with self._lock, self._conn:
            self._conn.execute(
                f"INSERT OR REPLACE INTO traces ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                tuple(d.values()),
            )

    def traces(
        self,
        project_id: str,
        start_ns: int | None = None,
        end_ns: int | None = None,
        root_name: str | None = None,
        limit: int = 100_000,
        order: str = "ASC",
    ) -> list[TraceSummary]:
        sql = "SELECT * FROM traces WHERE project_id = ? AND has_root = 1"
        params: list[Any] = [project_id]
        if start_ns is not None:
            sql += " AND start_ns >= ?"
            params.append(start_ns)
        if end_ns is not None:
            sql += " AND start_ns < ?"
            params.append(end_ns)
        if root_name:
            sql += " AND root_name = ?"
            params.append(root_name)
        sql += f" ORDER BY start_ns {'DESC' if order == 'DESC' else 'ASC'} LIMIT ?"
        params.append(limit)
        out = []
        for r in self.query(sql, params):
            d = dict(r)
            d["models"], d["tools"] = json.loads(d["models"]), json.loads(d["tools"])
            d["has_root"] = bool(d["has_root"])
            out.append(TraceSummary.model_validate(d))
        return out

    def time_range(self, project_id: str) -> tuple[int, int] | None:
        row = self.query("SELECT MIN(start_ns), MAX(end_ns) FROM traces WHERE project_id = ?", (project_id,))[0]
        return (row[0], row[1]) if row[0] is not None else None

    # --- governance -----------------------------------------------------------------------------

    def add_violations(self, violations: list[Violation]) -> None:
        with self._lock, self._conn:
            self._conn.executemany(
                "INSERT OR IGNORE INTO violations (project_id, trace_id, span_id, policy, severity, message, ts_ns) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (v.project_id, v.trace_id, v.span_id or "", v.policy, v.severity, v.message, v.ts_ns)
                    for v in violations
                ],
            )

    def violations(self, project_id: str, limit: int = 500) -> list[Violation]:
        rows = self.query(
            "SELECT * FROM violations WHERE project_id = ? ORDER BY ts_ns DESC LIMIT ?", (project_id, limit)
        )
        return [Violation.model_validate({**dict(r), "span_id": r["span_id"] or None}) for r in rows]

    def daily_cost(self, project_id: str, day_start_ns: int) -> float:
        row = self.query(
            "SELECT COALESCE(SUM(cost_usd), 0) FROM spans WHERE project_id = ? AND start_ns >= ? AND start_ns < ?",
            (project_id, day_start_ns, day_start_ns + 86_400 * 10**9),
        )[0]
        return float(row[0])

    def log_content_access(self, project_id: str, key_name: str, trace_id: str, purpose: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO content_access_log (ts_ns, project_id, key_name, trace_id, purpose) "
                "VALUES (?, ?, ?, ?, ?)",
                (time.time_ns(), project_id, key_name, trace_id, purpose),
            )

    def apply_retention(self, project_id: str, now_ns: int, retention_days: int, content_days: int) -> dict[str, int]:
        day = 86_400 * 10**9
        with self._lock, self._conn:
            scrubbed = self._conn.execute(
                "UPDATE spans SET input_text = NULL, output_text = NULL WHERE project_id = ? AND start_ns < ? "
                "AND (input_text IS NOT NULL OR output_text IS NOT NULL)",
                (project_id, now_ns - content_days * day),
            ).rowcount
            deleted = self._conn.execute(
                "DELETE FROM spans WHERE project_id = ? AND start_ns < ?", (project_id, now_ns - retention_days * day)
            ).rowcount
            self._conn.execute(
                "DELETE FROM traces WHERE project_id = ? AND start_ns < ?", (project_id, now_ns - retention_days * day)
            )
        return {"content_scrubbed": scrubbed, "spans_deleted": deleted}

    # --- change events, anomalies, incidents ------------------------------------------------------

    def add_change(self, event: ChangeEvent) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO change_events (project_id, ts_ns, kind, service, description, attributes) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    event.project_id,
                    event.ts_ns,
                    event.kind,
                    event.service,
                    event.description,
                    json.dumps(event.attributes),
                ),
            )
            return int(cur.lastrowid or 0)

    def changes(self, project_id: str, start_ns: int = 0, end_ns: int = 2**62) -> list[ChangeEvent]:
        rows = self.query(
            "SELECT * FROM change_events WHERE project_id = ? AND ts_ns >= ? AND ts_ns < ? ORDER BY ts_ns",
            (project_id, start_ns, end_ns),
        )
        return [ChangeEvent.model_validate({**dict(r), "attributes": json.loads(r["attributes"])}) for r in rows]

    def add_anomaly(self, a: Anomaly) -> int | None:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO anomalies (project_id, detector, group_key, metric, window_start_ns, "
                "window_end_ns, observed, baseline, score, severity, details, incident_id) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    a.project_id,
                    a.detector,
                    a.group_key,
                    a.metric,
                    a.window_start_ns,
                    a.window_end_ns,
                    a.observed,
                    a.baseline,
                    a.score,
                    a.severity,
                    json.dumps(a.details, default=str),
                    a.incident_id,
                ),
            )
            return int(cur.lastrowid or 0) if cur.rowcount else None

    def anomalies(self, project_id: str, incident_id: int | None = None) -> list[Anomaly]:
        sql = "SELECT * FROM anomalies WHERE project_id = ?"
        params: list[Any] = [project_id]
        if incident_id is not None:
            sql += " AND incident_id = ?"
            params.append(incident_id)
        return [
            Anomaly.model_validate({**dict(r), "details": json.loads(r["details"])})
            for r in self.query(sql + " ORDER BY window_start_ns", params)
        ]

    def set_anomaly_incident(self, anomaly_id: int, incident_id: int) -> None:
        with self._lock, self._conn:
            self._conn.execute("UPDATE anomalies SET incident_id = ? WHERE id = ?", (incident_id, anomaly_id))

    def save_incident(self, inc: Incident) -> int:
        values = (
            inc.project_id,
            inc.group_key,
            inc.title,
            inc.status,
            inc.severity,
            inc.start_ns,
            inc.end_ns,
            json.dumps(inc.detectors),
            inc.anomaly_count,
        )
        with self._lock, self._conn:
            if inc.id is None:
                cur = self._conn.execute(
                    "INSERT INTO incidents (project_id, group_key, title, status, severity, start_ns, end_ns, "
                    "detectors, anomaly_count) VALUES (?,?,?,?,?,?,?,?,?)",
                    values,
                )
                return int(cur.lastrowid or 0)
            self._conn.execute(
                "UPDATE incidents SET project_id=?, group_key=?, title=?, status=?, severity=?, start_ns=?, end_ns=?, "
                "detectors=?, anomaly_count=? WHERE id = ?",
                (*values, inc.id),
            )
            return inc.id

    def incidents(self, project_id: str, status: str | None = None) -> list[Incident]:
        sql = "SELECT * FROM incidents WHERE project_id = ?"
        params: list[Any] = [project_id]
        if status:
            sql += " AND status = ?"
            params.append(status)
        return [
            Incident.model_validate({**dict(r), "detectors": json.loads(r["detectors"])})
            for r in self.query(sql + " ORDER BY start_ns", params)
        ]

    def incident(self, project_id: str, incident_id: int) -> Incident | None:
        rows = self.query("SELECT * FROM incidents WHERE project_id = ? AND id = ?", (project_id, incident_id))
        return (
            Incident.model_validate({**dict(rows[0]), "detectors": json.loads(rows[0]["detectors"])}) if rows else None
        )

    def save_investigation(
        self, project_id: str, incident_id: int, model: str, report: dict[str, Any], errors: list[str]
    ) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO investigations (incident_id, project_id, created_ns, model, report, validation_errors) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (incident_id, project_id, time.time_ns(), model, json.dumps(report), json.dumps(errors)),
            )
            return int(cur.lastrowid or 0)

    def investigations(self, project_id: str, incident_id: int) -> list[dict[str, Any]]:
        rows = self.query(
            "SELECT * FROM investigations WHERE project_id = ? AND incident_id = ? ORDER BY id",
            (project_id, incident_id),
        )
        return [
            {**dict(r), "report": json.loads(r["report"]), "validation_errors": json.loads(r["validation_errors"])}
            for r in rows
        ]
