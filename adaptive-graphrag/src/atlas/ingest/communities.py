"""GraphRAG-style community detection (Louvain) and LLM community reports.

Community reports power "global" questions ("what are the main themes?") that no
single chunk can answer: they are embedded and retrieved like documents.
"""

from __future__ import annotations

import asyncio

import networkx as nx

from atlas.logging import get_logger
from atlas.models import Community, Entity, Relation, stable_id
from atlas.prompts import community_messages
from atlas.providers.base import LLM
from atlas.schemas import CommunityReport
from atlas.stores.graph import GraphStore

log = get_logger(__name__)


def detect_communities(
    edges: list[tuple[str, str, float]], *, min_size: int = 3, max_communities: int = 64, seed: int = 42
) -> list[list[str]]:
    graph = nx.Graph()
    for u, v, w in edges:
        prev = graph.get_edge_data(u, v, {"weight": 0.0})["weight"]
        graph.add_edge(u, v, weight=prev + w)
    if graph.number_of_edges() == 0:
        return []
    parts = nx.community.louvain_communities(graph, weight="weight", resolution=1.0, seed=seed)
    ranked = sorted((sorted(p) for p in parts if len(p) >= min_size), key=len, reverse=True)
    return ranked[:max_communities]


async def build_communities(
    graph: GraphStore, llm: LLM, *, concurrency: int = 4, max_entities: int = 20
) -> list[Community]:
    groups = await asyncio.to_thread(lambda: detect_communities(graph.weighted_edges()))
    semaphore = asyncio.Semaphore(concurrency)

    async def report(members: list[str]) -> Community | None:
        entities: list[Entity] = await asyncio.to_thread(graph.entities, members)
        entities.sort(key=lambda e: len(e.source_chunks), reverse=True)
        names = [f"{e.name} ({e.type}): {e.description}".strip(": ") for e in entities[:max_entities]]
        facts: list[Relation] = await asyncio.to_thread(graph.neighborhood, members[:max_entities], 1, 30)
        async with semaphore:
            try:
                result = await llm.structured(community_messages(names, [f.as_fact() for f in facts]), CommunityReport)
            except Exception as exc:
                log.warning("community.report_failed", error=str(exc))
                return None
        return Community(id=stable_id(*members), entities=members, title=result.title, summary=result.summary)

    communities = [c for c in await asyncio.gather(*(report(g) for g in groups)) if c]
    await asyncio.to_thread(graph.save_communities, communities)
    return communities
