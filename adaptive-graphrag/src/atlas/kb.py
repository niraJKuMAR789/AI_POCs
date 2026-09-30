"""KnowledgeBase: owns every store and implements the ingestion pipeline.

documents -> structure-aware chunks -> NeMo Retriever embeddings -> Qdrant
                                    -> SQLite docstore + BM25
                                    -> LLM entity/relation extraction -> graph store
graph -> Louvain communities -> LLM community reports -> Qdrant (global search)
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import Iterable
from dataclasses import dataclass, field

from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient

from atlas.config import Settings
from atlas.ingest.chunking import chunk_document, embedding_text
from atlas.ingest.communities import build_communities
from atlas.ingest.extraction import extract_graph
from atlas.logging import get_logger
from atlas.models import Entity, SourceDocument
from atlas.providers import Providers, build_providers
from atlas.stores.docstore import DocStore
from atlas.stores.graph import GraphStore, Neo4jGraphStore, NetworkXGraphStore
from atlas.stores.lexical import BM25Index
from atlas.stores.vector import QdrantVectorStore

log = get_logger(__name__)
_EMBED_BATCH = 64


class IngestReport(BaseModel):
    documents: int = 0
    skipped: int = 0
    chunks: int = 0
    entities: int = 0
    relations: int = 0
    communities: int = 0
    seconds: float = 0.0


@dataclass
class KnowledgeBase:
    settings: Settings
    providers: Providers
    docstore: DocStore
    graph: GraphStore
    chunk_vectors: QdrantVectorStore
    entity_vectors: QdrantVectorStore
    community_vectors: QdrantVectorStore
    qdrant: AsyncQdrantClient
    bm25: BM25Index = field(default_factory=BM25Index)
    entity_keys: set[str] = field(default_factory=set)
    _topics: str | None = None

    @classmethod
    async def open(cls, settings: Settings, providers: Providers | None = None) -> KnowledgeBase:
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        if settings.qdrant_url:
            api_key = settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
            qdrant = AsyncQdrantClient(url=settings.qdrant_url, api_key=api_key)
        else:
            qdrant = AsyncQdrantClient(path=str(settings.data_dir / "qdrant"))
        graph: GraphStore
        if settings.graph_backend == "neo4j":
            graph = Neo4jGraphStore(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password.get_secret_value())
        else:
            graph = NetworkXGraphStore(settings.data_dir / "graph.json")
        kb = cls(
            settings=settings,
            providers=providers or build_providers(settings),
            docstore=DocStore(settings.data_dir / "docstore.sqlite"),
            graph=graph,
            chunk_vectors=QdrantVectorStore(qdrant, settings.collection),
            entity_vectors=QdrantVectorStore(qdrant, f"{settings.collection}_entities"),
            community_vectors=QdrantVectorStore(qdrant, f"{settings.collection}_communities"),
            qdrant=qdrant,
        )
        kb.bm25.build(kb.docstore.all_chunks())
        kb.entity_keys = {e.key for e in await asyncio.to_thread(graph.entities)}
        return kb

    async def close(self) -> None:
        self.graph.close()
        await self.qdrant.close()

    # --- ingestion ---------------------------------------------------------------

    async def ingest(self, documents: Iterable[SourceDocument], *, force: bool = False) -> IngestReport:
        started = time.perf_counter()
        report = IngestReport()
        new_chunks = []
        for doc in documents:
            checksum = hashlib.sha256(doc.text.encode()).hexdigest()
            if not force and self.docstore.checksum(doc.id) == checksum:
                report.skipped += 1
                continue
            chunks = chunk_document(doc, chunk_size=self.settings.chunk_size, chunk_overlap=self.settings.chunk_overlap)
            await self.chunk_vectors.delete_where("doc_id", doc.id)
            await self._embed_and_upsert(
                self.chunk_vectors,
                ids=[c.id for c in chunks],
                texts=[embedding_text(c) for c in chunks],
                payloads=[
                    {"doc_id": c.doc_id, "source": c.source, "title": c.title, "section": c.section} for c in chunks
                ],
            )
            self.docstore.replace_document(doc.id, doc.source, doc.title, checksum, chunks)
            new_chunks.extend(chunks)
            report.documents += 1
            log.info("ingest.document", source=doc.source, chunks=len(chunks))
        report.chunks = len(new_chunks)
        if new_chunks:
            self.bm25.build(self.docstore.all_chunks())
            if self.settings.extract_graph:
                entities, relations = await extract_graph(self.providers.generator, new_chunks)
                await asyncio.to_thread(self.graph.upsert, entities, relations)
                await self._index_entities(entities)
                self.entity_keys.update(e.key for e in entities)
                report.entities, report.relations = len(entities), len(relations)
                report.communities = await self.rebuild_communities()
        self._topics = None
        report.seconds = round(time.perf_counter() - started, 2)
        log.info("ingest.done", **report.model_dump())
        return report

    async def rebuild_communities(self) -> int:
        communities = await build_communities(self.graph, self.providers.generator)
        await self.community_vectors.clear()
        await self._embed_and_upsert(
            self.community_vectors,
            ids=[c.id for c in communities],
            texts=[f"{c.title}\n{c.summary}" for c in communities],
            payloads=[{"title": c.title, "summary": c.summary, "size": len(c.entities)} for c in communities],
        )
        return len(communities)

    async def _index_entities(self, entities: list[Entity]) -> None:
        await self._embed_and_upsert(
            self.entity_vectors,
            ids=[e.key for e in entities],
            texts=[f"{e.name}: {e.description}" if e.description else e.name for e in entities],
            payloads=[{"name": e.name, "type": e.type} for e in entities],
        )

    async def _embed_and_upsert(
        self, store: QdrantVectorStore, *, ids: list[str], texts: list[str], payloads: list[dict]
    ) -> None:
        for start in range(0, len(ids), _EMBED_BATCH):
            end = start + _EMBED_BATCH
            vectors = await self.providers.embedder.embed_documents(texts[start:end])
            await store.upsert(ids[start:end], vectors, payloads[start:end])

    # --- introspection -------------------------------------------------------------

    def topics(self, limit: int = 12) -> str:
        """Short description of corpus coverage, fed to the router for better routing."""
        if self._topics is None:
            titles = [c.title for c in self.graph.communities()[:limit]]
            if not titles:
                titles = sorted({d["title"] for d in self.docstore.documents()})[:limit]
            self._topics = "; ".join(titles)
        return self._topics

    def stats(self) -> dict[str, int]:
        return {
            "documents": len(self.docstore.documents()),
            "chunks": self.docstore.count(),
            **self.graph.stats(),
        }
