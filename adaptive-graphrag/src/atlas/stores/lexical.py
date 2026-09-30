"""In-process BM25 index over chunks (the sparse half of hybrid retrieval)."""

from __future__ import annotations

import threading

from rank_bm25 import BM25Okapi

from atlas.models import Chunk
from atlas.text import tokenize


class BM25Index:
    def __init__(self) -> None:
        self._ids: list[str] = []
        self._bm25: BM25Okapi | None = None
        self._lock = threading.Lock()

    def build(self, chunks: list[Chunk]) -> None:
        corpus = [tokenize(f"{c.title} {c.section} {c.text}") for c in chunks]
        with self._lock:
            self._ids = [c.id for c in chunks]
            self._bm25 = BM25Okapi(corpus) if corpus else None

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        terms = tokenize(query)
        with self._lock:
            if self._bm25 is None or not terms:
                return []
            scores = self._bm25.get_scores(terms)
            ids = self._ids
        ranked = sorted(zip(ids, scores, strict=True), key=lambda x: x[1], reverse=True)
        return [(cid, float(s)) for cid, s in ranked[:k] if s > 0]
