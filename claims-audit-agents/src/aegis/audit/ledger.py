"""Tamper-evident audit ledger: an append-only, hash-chained event log in SQLite.

Each event's hash covers its content and the previous event's hash, so editing or deleting any
historical event breaks verification of every later event.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

GENESIS = "0" * 64
_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_events (
    seq INTEGER PRIMARY KEY,
    ts TEXT NOT NULL,
    claim_id TEXT NOT NULL,
    actor TEXT NOT NULL,
    event TEXT NOT NULL,
    payload TEXT NOT NULL,
    prev_hash TEXT NOT NULL,
    hash TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_claim ON audit_events(claim_id);
"""


def _digest(prev: str, ts: str, claim_id: str, actor: str, event: str, payload: str) -> str:
    return hashlib.sha256("\x1f".join((prev, ts, claim_id, actor, event, payload)).encode()).hexdigest()


class AuditLedger:
    def __init__(self, path: Path | str) -> None:
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._lock = threading.Lock()

    def append(self, claim_id: str, actor: str, event: str, payload: dict[str, Any]) -> str:
        body = json.dumps(payload, sort_keys=True, default=str)
        with self._lock, self._conn:
            row = self._conn.execute("SELECT hash FROM audit_events ORDER BY seq DESC LIMIT 1").fetchone()
            prev = row[0] if row else GENESIS
            ts = datetime.now(UTC).isoformat()
            digest = _digest(prev, ts, claim_id, actor, event, body)
            self._conn.execute(
                "INSERT INTO audit_events (ts, claim_id, actor, event, payload, prev_hash, hash) "
                "VALUES (?,?,?,?,?,?,?)",
                (ts, claim_id, actor, event, body, prev, digest),
            )
        return digest

    def events(self, claim_id: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT seq, ts, claim_id, actor, event, payload, prev_hash, hash FROM audit_events"
        rows = (
            self._conn.execute(query + " WHERE claim_id = ? ORDER BY seq", (claim_id,))
            if claim_id
            else self._conn.execute(query + " ORDER BY seq")
        )
        keys = ("seq", "ts", "claim_id", "actor", "event", "payload", "prev_hash", "hash")
        return [{**dict(zip(keys, r, strict=True)), "payload": json.loads(r[5])} for r in rows]

    def pending_claims(self) -> list[str]:
        """Claims adjudicated as pended and not yet decided by a human (the review queue)."""
        rows = self._conn.execute(
            "SELECT DISTINCT claim_id FROM audit_events a WHERE event = 'adjudicated' "
            "AND json_extract(payload, '$.outcome') = 'pended_for_review' "
            "AND NOT EXISTS (SELECT 1 FROM audit_events h "
            "WHERE h.claim_id = a.claim_id AND h.event = 'human_decision') "
            "ORDER BY seq"
        )
        return [r[0] for r in rows]

    def verify(self) -> tuple[bool, int | None]:
        """Recompute the whole chain. Returns (ok, first_bad_seq)."""
        prev = GENESIS
        for seq, ts, claim_id, actor, event, payload, prev_hash, digest in self._conn.execute(
            "SELECT seq, ts, claim_id, actor, event, payload, prev_hash, hash FROM audit_events ORDER BY seq"
        ):
            if prev_hash != prev or _digest(prev, ts, claim_id, actor, event, payload) != digest:
                return False, seq
            prev = digest
        return True, None
