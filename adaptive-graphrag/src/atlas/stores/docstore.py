"""SQLite document/chunk store: source of truth for chunk text and provenance."""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from atlas.models import Chunk

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY, source TEXT NOT NULL, title TEXT, checksum TEXT NOT NULL, ingested_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY, doc_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    source TEXT, title TEXT, section TEXT, text TEXT NOT NULL, position INTEGER, metadata TEXT
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
"""


class DocStore:
    def __init__(self, path: Path | str) -> None:
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.executescript(_SCHEMA)
        self._lock = threading.Lock()

    def checksum(self, doc_id: str) -> str | None:
        row = self._conn.execute("SELECT checksum FROM documents WHERE id = ?", (doc_id,)).fetchone()
        return row[0] if row else None

    def replace_document(self, doc_id: str, source: str, title: str, checksum: str, chunks: Iterable[Chunk]) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            self._conn.execute(
                "INSERT INTO documents VALUES (?, ?, ?, ?, ?)",
                (doc_id, source, title, checksum, datetime.now(UTC).isoformat()),
            )
            self._conn.executemany(
                "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (c.id, c.doc_id, c.source, c.title, c.section, c.text, c.position, json.dumps(c.metadata))
                    for c in chunks
                ],
            )

    def get(self, ids: list[str]) -> dict[str, Chunk]:
        if not ids:
            return {}
        marks = ",".join("?" * len(ids))
        rows = self._conn.execute(f"SELECT * FROM chunks WHERE id IN ({marks})", ids).fetchall()
        return {row[0]: self._row(row) for row in rows}

    def all_chunks(self) -> list[Chunk]:
        return [self._row(r) for r in self._conn.execute("SELECT * FROM chunks ORDER BY doc_id, position")]

    def documents(self) -> list[dict[str, str]]:
        rows = self._conn.execute(
            "SELECT d.id, d.source, d.title, d.ingested_at, COUNT(c.id) FROM documents d "
            "LEFT JOIN chunks c ON c.doc_id = d.id GROUP BY d.id ORDER BY d.source"
        ).fetchall()
        keys = ("id", "source", "title", "ingested_at", "chunks")
        return [dict(zip(keys, r, strict=True)) for r in rows]

    def count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    @staticmethod
    def _row(row: tuple) -> Chunk:
        return Chunk(
            id=row[0],
            doc_id=row[1],
            source=row[2],
            title=row[3],
            section=row[4],
            text=row[5],
            position=row[6],
            metadata=json.loads(row[7] or "{}"),
        )
