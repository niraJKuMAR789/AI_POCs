"""SQLite executor with defense in depth beneath the AST guard.

* reads use a separate `mode=ro` connection, so even a guard bypass cannot write
* a progress handler aborts queries that exceed the time budget
* results are capped at fetch time as well as by the rewritten LIMIT
"""

from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class QueryTimeoutError(RuntimeError):
    pass


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[tuple[Any, ...]]
    elapsed_ms: float
    truncated: bool = False
    plan_warnings: list[str] = field(default_factory=list)

    def as_records(self, limit: int | None = None) -> list[dict[str, Any]]:
        rows = self.rows if limit is None else self.rows[:limit]
        return [dict(zip(self.columns, r, strict=True)) for r in rows]

    def to_markdown(self, limit: int = 20) -> str:
        if not self.columns:
            return "(no columns)"
        head = "| " + " | ".join(self.columns) + " |\n|" + "---|" * len(self.columns)
        body = "\n".join("| " + " | ".join(_fmt(v) for v in row) + " |" for row in self.rows[:limit])
        more = f"\n… {len(self.rows) - limit} more rows" if len(self.rows) > limit else ""
        return f"{head}\n{body}{more}"


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:,.4f}".rstrip("0").rstrip(".")
    return "NULL" if value is None else str(value)


class Database:
    def __init__(self, path: Path, *, timeout_s: float = 5.0, large_tables: frozenset[str] = frozenset()) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Database not found: {path}. Run `quill init-db` first.")
        self.path = path
        self.timeout_s = timeout_s
        self.large_tables = large_tables
        self._local = threading.local()
        self._write_lock = threading.Lock()

    def _reader(self) -> sqlite3.Connection:
        conn = getattr(self._local, "reader", None)
        if conn is None:
            conn = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, check_same_thread=False)
            self._local.reader = conn
        return conn

    def query(self, sql: str, *, max_rows: int = 1000) -> QueryResult:
        conn = self._reader()
        deadline = time.monotonic() + self.timeout_s
        conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
        started = time.perf_counter()
        try:
            cursor = conn.execute(sql)
            rows = cursor.fetchmany(max_rows + 1)
        except sqlite3.OperationalError as exc:
            if "interrupted" in str(exc):
                raise QueryTimeoutError(f"query exceeded {self.timeout_s}s budget") from exc
            raise
        finally:
            conn.set_progress_handler(None, 0)
        columns = [d[0] for d in cursor.description or []]
        return QueryResult(
            columns=columns,
            rows=[tuple(r) for r in rows[:max_rows]],
            elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
            truncated=len(rows) > max_rows,
        )

    def execute_write(self, sql: str) -> int:
        """Run a guard-approved INSERT on a dedicated read-write connection."""
        with self._write_lock, sqlite3.connect(self.path) as conn:
            return conn.execute(sql).rowcount

    def plan_warnings(self, sql: str) -> list[str]:
        """Flag full scans of large tables that have no usable index (EXPLAIN QUERY PLAN)."""
        try:
            plan = self._reader().execute(f"EXPLAIN QUERY PLAN {sql}").fetchall()
        except sqlite3.Error:
            return []
        warnings = []
        for row in plan:
            detail = str(row[-1])
            if detail.startswith("SCAN "):
                table = detail.split()[1]
                if table in self.large_tables and "USING" not in detail:
                    warnings.append(f"full scan of large table '{table}'")
        return sorted(set(warnings))
