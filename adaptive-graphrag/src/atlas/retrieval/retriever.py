"""Hybrid + graph retrieval.

local mode:
    query ─┬─ dense (NeMo Retriever embeddings → Qdrant) ──┐
           ├─ sparse (BM25) ────────────────────────────────┼─ RRF ─ rerank (NeMo reranker) ─ top-n chunks
           └─ entity linking → k-hop subgraph ─┬─ provenance chunks ┘
                                               └─ graph facts context
global mode:
    query → community-report vectors → top-k community summaries (+ local evidence)
"""

from __future__ import annotations

import asyncio
import time

from atlas.ingest.chunking import heading_path
from atlas.kb import KnowledgeBase
from atlas.logging import get_logger
from atlas.models import Chunk, RetrievedContext, normalize_entity_name
from atlas.retrieval.fusion import reciprocal_rank_fusion

log = get_logger(__name__)
_ENTITY_MIN_SCORE = 0.35


class Retriever:
    def __init__(self, kb: KnowledgeBase) -> None:
        self.kb = kb
        self.s = kb.settings

    async def retrieve(self, query: str, mode: str = "local") -> list[RetrievedContext]:
        started = time.perf_counter()
        query_vector = await self.kb.providers.embedder.embed_query(query)
        if mode == "global":
            communities, local = await asyncio.gather(
                self._communities(query_vector), self._local(query, query_vector, top_n=3)
            )
            contexts = communities + local
        else:
            contexts = await self._local(query, query_vector, top_n=self.s.rerank_top_n)
        log.info("retrieve", mode=mode, n=len(contexts), ms=round((time.perf_counter() - started) * 1000, 1))
        return contexts

    async def _local(self, query: str, query_vector: list[float], *, top_n: int) -> list[RetrievedContext]:
        dense, (graph_ctx, graph_chunk_ids) = await asyncio.gather(
            self.kb.chunk_vectors.search(query_vector, self.s.dense_k),
            self._graph(query, query_vector),
        )
        sparse = self.kb.bm25.search(query, self.s.sparse_k)
        fused = reciprocal_rank_fusion([[cid for cid, _, _ in dense], [cid for cid, _ in sparse], graph_chunk_ids])
        candidate_ids = [cid for cid, _ in fused[: max(top_n * 4, 20)]]
        chunks = self.kb.docstore.get(candidate_ids)
        candidates: list[Chunk] = [chunks[cid] for cid in candidate_ids if cid in chunks]
        ranked = await self.kb.providers.reranker.rerank(
            query, [f"{c.section}\n{c.text}" if c.section else c.text for c in candidates], top_n
        )
        contexts = [self._chunk_context(candidates[i], score) for i, score in ranked]
        return ([graph_ctx] if graph_ctx else []) + contexts

    async def _graph(self, query: str, query_vector: list[float]) -> tuple[RetrievedContext | None, list[str]]:
        keys = await self._link_entities(query, query_vector)
        if not keys:
            return None, []
        relations = await asyncio.to_thread(self.kb.graph.neighborhood, keys, self.s.graph_hops, self.s.graph_max_facts)
        entities = await asyncio.to_thread(self.kb.graph.entities, keys)
        if not relations and not entities:
            return None, []
        lines = [f"- {e.name} ({e.type}): {e.description}" for e in entities if e.description]
        lines += [f"- {r.as_fact()}" for r in relations]
        provenance = list(dict.fromkeys(cid for r in relations for cid in r.source_chunks))
        context = RetrievedContext(
            id="graph:" + ",".join(keys[:5]),
            kind="graph",
            text="Knowledge-graph facts:\n" + "\n".join(lines),
            source="knowledge-graph",
            title=f"Graph neighbourhood of {', '.join(e.name for e in entities[:4]) or keys[0]}",
            metadata={"entities": keys, "facts": len(relations)},
        )
        return context, provenance

    async def _link_entities(self, query: str, query_vector: list[float]) -> list[str]:
        """Entity linking: exact mention match, then embedding similarity over entity descriptions."""
        words = normalize_entity_name(query).split()
        ngrams = [" ".join(words[i : i + n]) for n in (4, 3, 2, 1) for i in range(len(words) - n + 1)]
        linked = [g for g in ngrams if len(g) >= 3 and g in self.kb.entity_keys]
        hits = await self.kb.entity_vectors.search(query_vector, 8)
        linked += [key for key, score, _ in hits if score >= _ENTITY_MIN_SCORE]
        return list(dict.fromkeys(linked))[:6]

    async def _communities(self, query_vector: list[float]) -> list[RetrievedContext]:
        hits = await self.kb.community_vectors.search(query_vector, self.s.community_top_k)
        return [
            RetrievedContext(
                id=f"community:{key}",
                kind="community",
                text=payload["summary"],
                source="community-report",
                title=payload["title"],
                score=score,
                metadata={"size": payload.get("size", 0)},
            )
            for key, score, payload in hits
        ]

    @staticmethod
    def _chunk_context(chunk: Chunk, score: float) -> RetrievedContext:
        return RetrievedContext(
            id=chunk.id,
            kind="chunk",
            text=chunk.text,
            source=chunk.source,
            title=heading_path(chunk),
            score=score,
            metadata={"doc_id": chunk.doc_id, "position": chunk.position},
        )
