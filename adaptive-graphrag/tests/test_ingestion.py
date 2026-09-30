from atlas.ingest.chunking import chunk_document, embedding_text
from atlas.ingest.extraction import resolve
from atlas.ingest.loaders import load_text
from atlas.kb import KnowledgeBase
from atlas.models import Chunk
from atlas.schemas import ExtractedEntity, ExtractedRelation, ExtractionResult

DOC = """# Guide

Intro paragraph.

## Setup

### GPUs

Install drivers.

## Usage

Run it.
"""


def test_chunking_tracks_heading_path_and_is_deterministic() -> None:
    doc = load_text(DOC, "guide.md")
    chunks = chunk_document(doc)
    assert [c.section for c in chunks] == ["Guide", "Guide > Setup > GPUs", "Guide > Usage"]
    assert embedding_text(chunks[1]).startswith("Guide > Setup > GPUs\n\nInstall drivers.")
    assert [c.id for c in chunks] == [c.id for c in chunk_document(doc)]


def test_resolve_merges_entities_and_counts_relations() -> None:
    c1 = Chunk(id="c1", doc_id="d", source="s", title="t", text="x")
    c2 = Chunk(id="c2", doc_id="d", source="s", title="t", text="y")
    r1 = ExtractionResult(
        entities=[ExtractedEntity(name="Heron API", type="product", description="Gateway.")],
        relations=[ExtractedRelation(source="Heron API", target="Fleet Core", type="owned by")],
    )
    r2 = ExtractionResult(
        entities=[ExtractedEntity(name="heron api", type="PRODUCT", description="The public REST gateway.")],
        relations=[ExtractedRelation(source="heron  API", target="fleet core", type="OWNED_BY")],
    )
    entities, relations = resolve([(c1, r1), (c2, r2), (c2, None)])
    assert len(entities) == 1
    assert entities[0].description == "The public REST gateway."
    assert entities[0].source_chunks == ["c1", "c2"]
    assert len(relations) == 1 and relations[0].type == "OWNED_BY" and relations[0].weight == 2


async def test_ingest_builds_all_indexes_and_is_idempotent(kb: KnowledgeBase) -> None:
    stats = kb.stats()
    assert stats["documents"] == 9
    assert stats["chunks"] > 20
    assert stats["entities"] > 20 and stats["relations"] > 20 and stats["communities"] >= 1
    from tests.conftest import CORPUS

    from atlas.ingest.loaders import iter_documents

    report = await kb.ingest(iter_documents(CORPUS))
    assert report.documents == 0 and report.skipped == 9


async def test_reingest_changed_document_replaces_chunks(kb: KnowledgeBase) -> None:
    before = kb.stats()["chunks"]
    await kb.ingest([load_text("# Temp\n\nFirst version.", "temp.md")])
    await kb.ingest([load_text("# Temp\n\nSecond version, now different.", "temp.md")])
    texts = [c.text for c in kb.docstore.all_chunks() if c.source == "temp.md"]
    assert texts == ["Second version, now different."]
    assert kb.stats()["chunks"] == before + 1
