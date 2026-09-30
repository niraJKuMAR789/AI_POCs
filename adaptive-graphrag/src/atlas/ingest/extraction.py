"""LLM-driven knowledge-graph extraction with structured outputs + entity resolution."""

from __future__ import annotations

import asyncio

from atlas.ingest.chunking import embedding_text
from atlas.logging import get_logger
from atlas.models import Chunk, Entity, Relation, normalize_entity_name
from atlas.prompts import extract_messages
from atlas.providers.base import LLM
from atlas.schemas import ExtractionResult

log = get_logger(__name__)


async def extract_graph(llm: LLM, chunks: list[Chunk], *, concurrency: int = 8) -> tuple[list[Entity], list[Relation]]:
    semaphore = asyncio.Semaphore(concurrency)

    async def one(chunk: Chunk) -> tuple[Chunk, ExtractionResult | None]:
        async with semaphore:
            try:
                return chunk, await llm.structured(extract_messages(embedding_text(chunk)), ExtractionResult)
            except Exception as exc:  # one bad chunk must not fail the whole ingestion
                log.warning("extraction.failed", chunk=chunk.id, error=str(exc))
                return chunk, None

    results = await asyncio.gather(*(one(c) for c in chunks))
    return resolve(results)


def resolve(results: list[tuple[Chunk, ExtractionResult | None]]) -> tuple[list[Entity], list[Relation]]:
    """Merge per-chunk extractions: dedupe entities by normalized name, keep provenance."""
    entities: dict[str, Entity] = {}
    relations: dict[tuple[str, str, str], Relation] = {}
    for chunk, result in results:
        if result is None:
            continue
        for e in result.entities:
            key = normalize_entity_name(e.name)
            if not key:
                continue
            existing = entities.get(key)
            if existing is None:
                entities[key] = Entity(
                    name=e.name.strip(),
                    type=e.type.upper(),
                    description=e.description,
                    source_chunks=[chunk.id],
                )
            else:
                if chunk.id not in existing.source_chunks:
                    existing.source_chunks.append(chunk.id)
                if len(e.description) > len(existing.description):
                    existing.description = e.description
        for r in result.relations:
            src, tgt = normalize_entity_name(r.source), normalize_entity_name(r.target)
            if not src or not tgt or src == tgt:
                continue
            rtype = r.type.upper().replace(" ", "_")
            rel_key = (src, tgt, rtype)
            if rel_key in relations:
                rel = relations[rel_key]
                rel.weight += 1
                if chunk.id not in rel.source_chunks:
                    rel.source_chunks.append(chunk.id)
            else:
                relations[rel_key] = Relation(
                    source=r.source.strip(),
                    target=r.target.strip(),
                    type=rtype,
                    description=r.description,
                    source_chunks=[chunk.id],
                )
    return list(entities.values()), list(relations.values())
