from atlas.kb import KnowledgeBase
from atlas.retrieval.retriever import Retriever


async def test_local_retrieval_finds_relevant_chunk(kb: KnowledgeBase) -> None:
    contexts = await Retriever(kb).retrieve("What certification does Kestrel have for driverless trucks?")
    chunk_sources = [c.source for c in contexts if c.kind == "chunk"]
    assert chunk_sources[0] == "08_kestrel_product_sheet.md"
    assert len(chunk_sources) <= kb.settings.rerank_top_n


async def test_entity_mention_adds_graph_context(kb: KnowledgeBase) -> None:
    contexts = await Retriever(kb).retrieve("What happened in INC-2041?")
    graph = [c for c in contexts if c.kind == "graph"]
    assert graph and "inc-2041" in graph[0].metadata["entities"]


async def test_global_retrieval_returns_community_reports(kb: KnowledgeBase) -> None:
    contexts = await Retriever(kb).retrieve("main themes", mode="global")
    assert any(c.kind == "community" for c in contexts)
