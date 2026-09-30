"""Knowledge-graph stores.

`NetworkXGraphStore` is embedded (persisted as JSON) for zero-infra local runs and
tests. `Neo4jGraphStore` is the production backend. Both expose the same
synchronous interface; async callers dispatch through `asyncio.to_thread`.
"""

from __future__ import annotations

import json
import threading
from collections import deque
from pathlib import Path
from typing import Protocol

import networkx as nx

from atlas.models import Community, Entity, Relation, normalize_entity_name


def _merge_list(a: list[str], b: list[str]) -> list[str]:
    return list(dict.fromkeys([*a, *b]))


class GraphStore(Protocol):
    def upsert(self, entities: list[Entity], relations: list[Relation]) -> None: ...
    def entities(self, keys: list[str] | None = None) -> list[Entity]: ...
    def neighborhood(self, keys: list[str], hops: int, limit: int) -> list[Relation]: ...
    def weighted_edges(self) -> list[tuple[str, str, float]]: ...
    def save_communities(self, communities: list[Community]) -> None: ...
    def communities(self) -> list[Community]: ...
    def stats(self) -> dict[str, int]: ...
    def clear(self) -> None: ...
    def close(self) -> None: ...


class NetworkXGraphStore:
    def __init__(self, path: Path | None = None) -> None:
        self._path = path
        self._lock = threading.RLock()
        self._graph = nx.MultiDiGraph()
        self._communities: list[Community] = []
        if path and path.exists():
            payload = json.loads(path.read_text())
            self._graph = nx.node_link_graph(payload["graph"], directed=True, multigraph=True, edges="edges")
            self._communities = [Community.model_validate(c) for c in payload.get("communities", [])]

    def _persist(self) -> None:
        if self._path is None:
            return
        payload = {
            "graph": nx.node_link_data(self._graph, edges="edges"),
            "communities": [c.model_dump() for c in self._communities],
        }
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload))
        tmp.replace(self._path)

    def upsert(self, entities: list[Entity], relations: list[Relation]) -> None:
        with self._lock:
            for e in entities:
                self._upsert_node(e.key, e.name, e.type, e.description, e.source_chunks)
            for r in relations:
                src, tgt = normalize_entity_name(r.source), normalize_entity_name(r.target)
                if not src or not tgt or src == tgt:
                    continue
                for key, name in ((src, r.source), (tgt, r.target)):
                    if key not in self._graph:
                        self._upsert_node(key, name, "CONCEPT", "", r.source_chunks)
                if self._graph.has_edge(src, tgt, key=r.type):
                    data = self._graph.edges[src, tgt, r.type]
                    data["weight"] += r.weight
                    data["source_chunks"] = _merge_list(data["source_chunks"], r.source_chunks)
                    if len(r.description) > len(data["description"]):
                        data["description"] = r.description
                else:
                    self._graph.add_edge(
                        src,
                        tgt,
                        key=r.type,
                        type=r.type,
                        description=r.description,
                        weight=r.weight,
                        source_chunks=list(r.source_chunks),
                    )
            self._persist()

    def _upsert_node(self, key: str, name: str, type_: str, description: str, chunks: list[str]) -> None:
        if key in self._graph:
            node = self._graph.nodes[key]
            node["source_chunks"] = _merge_list(node["source_chunks"], chunks)
            if len(description) > len(node["description"]):
                node["description"] = description
            if node["type"] == "CONCEPT" and type_ != "CONCEPT":
                node["type"] = type_
        else:
            self._graph.add_node(key, name=name, type=type_, description=description, source_chunks=list(chunks))

    def _entity(self, key: str) -> Entity:
        n = self._graph.nodes[key]
        return Entity(name=n["name"], type=n["type"], description=n["description"], source_chunks=n["source_chunks"])

    def entities(self, keys: list[str] | None = None) -> list[Entity]:
        with self._lock:
            selected = self._graph.nodes if keys is None else [k for k in keys if k in self._graph]
            return [self._entity(k) for k in selected]

    def neighborhood(self, keys: list[str], hops: int, limit: int) -> list[Relation]:
        """Breadth-first expansion; closer and heavier edges rank first."""
        with self._lock:
            undirected = self._graph.to_undirected(as_view=True)
            depth = {k: 0 for k in keys if k in self._graph}
            queue = deque(depth)
            while queue:
                node = queue.popleft()
                if depth[node] >= hops:
                    continue
                for nbr in undirected.neighbors(node):
                    if nbr not in depth:
                        depth[nbr] = depth[node] + 1
                        queue.append(nbr)
            scored: list[tuple[float, Relation]] = []
            for u, v, data in self._graph.edges(data=True):
                if u in depth and v in depth:
                    proximity = min(depth[u], depth[v])
                    rel = Relation(
                        source=self._graph.nodes[u]["name"],
                        target=self._graph.nodes[v]["name"],
                        type=data["type"],
                        description=data["description"],
                        weight=data["weight"],
                        source_chunks=data["source_chunks"],
                    )
                    scored.append((data["weight"] / (1 + proximity), rel))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [r for _, r in scored[:limit]]

    def weighted_edges(self) -> list[tuple[str, str, float]]:
        with self._lock:
            return [(u, v, float(d["weight"])) for u, v, d in self._graph.edges(data=True)]

    def save_communities(self, communities: list[Community]) -> None:
        with self._lock:
            self._communities = communities
            self._persist()

    def communities(self) -> list[Community]:
        with self._lock:
            return list(self._communities)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "entities": self._graph.number_of_nodes(),
                "relations": self._graph.number_of_edges(),
                "communities": len(self._communities),
            }

    def clear(self) -> None:
        with self._lock:
            self._graph.clear()
            self._communities = []
            self._persist()

    def close(self) -> None:
        pass


class Neo4jGraphStore:
    """Neo4j backend. Uses a single :REL relationship with a `type` property so
    arbitrary LLM-extracted predicates work without APOC dynamic relationship types."""

    def __init__(self, uri: str, user: str, password: str, database: str | None = None) -> None:
        from neo4j import GraphDatabase

        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._db = database
        self._run("CREATE CONSTRAINT entity_key IF NOT EXISTS FOR (e:Entity) REQUIRE e.key IS UNIQUE")

    def _run(self, query: str, **params: object) -> list[dict]:
        records, _, _ = self._driver.execute_query(query, params, database_=self._db)
        return [r.data() for r in records]

    def upsert(self, entities: list[Entity], relations: list[Relation]) -> None:
        self._run(
            """
            UNWIND $rows AS row
            MERGE (e:Entity {key: row.key})
            ON CREATE SET e.name = row.name, e.type = row.type, e.description = row.description,
                          e.source_chunks = row.chunks
            ON MATCH SET e.description = CASE WHEN size(row.description) > size(coalesce(e.description, ''))
                                              THEN row.description ELSE e.description END,
                         e.source_chunks = coalesce(e.source_chunks, []) +
                             [c IN row.chunks WHERE NOT c IN coalesce(e.source_chunks, [])]
            """,
            rows=[
                {
                    "key": e.key,
                    "name": e.name,
                    "type": e.type,
                    "description": e.description,
                    "chunks": e.source_chunks,
                }
                for e in entities
            ],
        )
        rows = []
        for r in relations:
            src, tgt = normalize_entity_name(r.source), normalize_entity_name(r.target)
            if src and tgt and src != tgt:
                rows.append(
                    {
                        "src": src,
                        "src_name": r.source,
                        "tgt": tgt,
                        "tgt_name": r.target,
                        "type": r.type,
                        "description": r.description,
                        "weight": r.weight,
                        "chunks": r.source_chunks,
                    }
                )
        self._run(
            """
            UNWIND $rows AS row
            MERGE (a:Entity {key: row.src}) ON CREATE SET a.name = row.src_name, a.type = 'CONCEPT',
                a.description = '', a.source_chunks = row.chunks
            MERGE (b:Entity {key: row.tgt}) ON CREATE SET b.name = row.tgt_name, b.type = 'CONCEPT',
                b.description = '', b.source_chunks = row.chunks
            MERGE (a)-[r:REL {type: row.type}]->(b)
            ON CREATE SET r.description = row.description, r.weight = row.weight, r.source_chunks = row.chunks
            ON MATCH SET r.weight = r.weight + row.weight,
                         r.source_chunks = r.source_chunks + [c IN row.chunks WHERE NOT c IN r.source_chunks]
            """,
            rows=rows,
        )

    def entities(self, keys: list[str] | None = None) -> list[Entity]:
        where = "WHERE e.key IN $keys" if keys is not None else ""
        rows = self._run(
            f"MATCH (e:Entity) {where} RETURN e.name AS name, e.type AS type, "
            "e.description AS description, e.source_chunks AS source_chunks",
            keys=keys or [],
        )
        return [Entity.model_validate(r) for r in rows]

    def neighborhood(self, keys: list[str], hops: int, limit: int) -> list[Relation]:
        hops = max(1, min(int(hops), 4))  # bounded; variable-length bounds can't be parameters
        rows = self._run(
            f"""
            MATCH (s:Entity) WHERE s.key IN $keys
            MATCH p = (s)-[:REL*1..{hops}]-(:Entity)
            UNWIND relationships(p) AS r
            WITH r, min(length(p)) AS dist
            RETURN startNode(r).name AS source, endNode(r).name AS target, r.type AS type,
                   r.description AS description, r.weight AS weight, r.source_chunks AS source_chunks
            ORDER BY r.weight / dist DESC LIMIT $limit
            """,
            keys=keys,
            limit=limit,
        )
        return [Relation.model_validate(r) for r in rows]

    def weighted_edges(self) -> list[tuple[str, str, float]]:
        rows = self._run("MATCH (a:Entity)-[r:REL]->(b:Entity) RETURN a.key AS a, b.key AS b, r.weight AS w")
        return [(r["a"], r["b"], float(r["w"])) for r in rows]

    def save_communities(self, communities: list[Community]) -> None:
        self._run("MATCH (c:Community) DETACH DELETE c")
        self._run(
            """
            UNWIND $rows AS row
            CREATE (c:Community {id: row.id, level: row.level, title: row.title, summary: row.summary,
                                 entities: row.entities})
            WITH c, row UNWIND row.entities AS key
            MATCH (e:Entity {key: key}) MERGE (e)-[:IN_COMMUNITY]->(c)
            """,
            rows=[c.model_dump() for c in communities],
        )

    def communities(self) -> list[Community]:
        rows = self._run(
            "MATCH (c:Community) RETURN c.id AS id, c.level AS level, c.title AS title, "
            "c.summary AS summary, c.entities AS entities"
        )
        return [Community.model_validate(r) for r in rows]

    def stats(self) -> dict[str, int]:
        row = self._run(
            "CALL () { MATCH (e:Entity) RETURN count(e) AS entities } "
            "CALL () { MATCH ()-[r:REL]->() RETURN count(r) AS relations } "
            "CALL () { MATCH (c:Community) RETURN count(c) AS communities } "
            "RETURN entities, relations, communities"
        )[0]
        return {k: int(v) for k, v in row.items()}

    def clear(self) -> None:
        self._run("MATCH (n) WHERE n:Entity OR n:Community DETACH DELETE n")

    def close(self) -> None:
        self._driver.close()
