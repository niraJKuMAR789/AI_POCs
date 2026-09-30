from pathlib import Path

from atlas.ingest.communities import detect_communities
from atlas.models import Community, Entity, Relation
from atlas.stores.graph import NetworkXGraphStore


def _store(path: Path | None = None) -> NetworkXGraphStore:
    store = NetworkXGraphStore(path)
    store.upsert(
        [Entity(name="A", type="SERVICE", description="Service A", source_chunks=["c1"])],
        [
            Relation(source="A", target="B", type="DEPENDS_ON", source_chunks=["c1"]),
            Relation(source="B", target="C", type="DEPENDS_ON", source_chunks=["c2"]),
            Relation(source="C", target="D", type="OWNED_BY", source_chunks=["c3"]),
        ],
    )
    return store


def test_neighborhood_respects_hops() -> None:
    store = _store()
    one_hop = {(r.source, r.target) for r in store.neighborhood(["a"], hops=1, limit=10)}
    two_hop = {(r.source, r.target) for r in store.neighborhood(["a"], hops=2, limit=10)}
    assert one_hop == {("A", "B")}
    assert two_hop == {("A", "B"), ("B", "C")}


def test_upsert_merges_edge_weight_and_provenance() -> None:
    store = _store()
    store.upsert([], [Relation(source="a", target="b", type="DEPENDS_ON", source_chunks=["c9"])])
    rel = store.neighborhood(["a"], hops=1, limit=10)[0]
    assert rel.weight == 2.0 and rel.source_chunks == ["c1", "c9"]


def test_persistence_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "graph.json"
    store = _store(path)
    store.save_communities([Community(id="x", entities=["a", "b"], title="T", summary="S")])
    reloaded = NetworkXGraphStore(path)
    assert reloaded.stats() == {"entities": 4, "relations": 3, "communities": 1}
    assert reloaded.entities(["a"])[0].description == "Service A"


def test_detect_communities_separates_clusters() -> None:
    edges = [
        ("a", "b", 1.0),
        ("b", "c", 1.0),
        ("a", "c", 1.0),
        ("x", "y", 1.0),
        ("y", "z", 1.0),
        ("x", "z", 1.0),
        ("c", "x", 0.1),
    ]
    groups = detect_communities(edges, min_size=3)
    assert sorted(map(sorted, groups)) == [["a", "b", "c"], ["x", "y", "z"]]
