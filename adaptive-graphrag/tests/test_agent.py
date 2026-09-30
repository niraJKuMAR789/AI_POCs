from tests.conftest import ScriptedLLM

from atlas.agent import AtlasAgent
from atlas.agent.graph import UNVERIFIED_NOTE
from atlas.kb import KnowledgeBase
from atlas.schemas import DocumentGrades, GroundednessCheck, QueryAnalysis


def nodes(result) -> list[str]:  # type: ignore[no-untyped-def]
    return [step.node for step in result.trace]


async def test_local_question_runs_full_crag_pipeline(kb: KnowledgeBase) -> None:
    result = await AtlasAgent(kb).ask("Which ISO standard is Kestrel certified to?")
    assert result.route == "local"
    path = nodes(result)
    assert path[:3] == ["analyze", "retrieve", "grade"]
    assert path[-3:] == ["generate", "reflect", "finalize"]
    assert set(path) <= {"analyze", "retrieve", "grade", "rewrite", "generate", "reflect", "finalize"}
    assert "ISO 3691-4" in result.answer
    assert result.grounded
    assert result.citations and all(c.source for c in result.citations)


async def test_direct_route_skips_retrieval(kb: KnowledgeBase) -> None:
    result = await AtlasAgent(kb).ask("hello")
    assert result.route == "direct"
    assert nodes(result) == ["analyze", "generate", "finalize"]
    assert result.citations == []


async def test_weak_evidence_triggers_bounded_rewrites(kb: KnowledgeBase, llm: ScriptedLLM) -> None:
    llm.overrides[DocumentGrades] = lambda: DocumentGrades(relevant_ids=[])
    result = await AtlasAgent(kb).ask("What is the Kafka retention period?")
    assert nodes(result).count("rewrite") == kb.settings.max_rewrites
    assert nodes(result).count("retrieve") == kb.settings.max_rewrites + 1
    assert "does not contain enough information" in result.answer


async def test_ungrounded_draft_is_regenerated_then_flagged(kb: KnowledgeBase, llm: ScriptedLLM) -> None:
    llm.overrides[GroundednessCheck] = lambda: GroundednessCheck(
        grounded=False, answers_question=True, unsupported_claims=["made-up claim"]
    )
    agent = AtlasAgent(kb)
    events = [e async for e in agent.astream("What is the Kafka retention period?")]
    types = [e["type"] for e in events]
    assert types.count("reset") == kb.settings.max_generation_retries
    final = events[-1]["result"]
    assert final["grounded"] is False and final["answer"].endswith(UNVERIFIED_NOTE)
    generate_steps = [e for e in events if e["type"] == "step" and e["node"] == "generate"]
    assert len(generate_steps) == kb.settings.max_generation_retries + 1


async def test_sub_questions_fan_out_retrieval(kb: KnowledgeBase, llm: ScriptedLLM) -> None:
    llm.overrides[QueryAnalysis] = lambda: QueryAnalysis(
        standalone_question="Who owns Telemetry Ingest and what caused INC-2041?",
        route="local",
        sub_questions=["Who owns Telemetry Ingest?", "What caused INC-2041?"],
    )
    result = await AtlasAgent(kb).ask("Who owns Telemetry Ingest and what caused INC-2041?")
    retrieve = next(s for s in result.trace if s.node == "retrieve")
    assert len(retrieve.data["queries"]) == 3


async def test_conversation_memory_is_threaded(kb: KnowledgeBase, llm: ScriptedLLM) -> None:
    agent = AtlasAgent(kb)
    first = await agent.ask("What does Telemetry Ingest do?")
    await agent.ask("Who is on call for it?", thread_id=first.thread_id)
    state = await agent.graph.aget_state({"configurable": {"thread_id": first.thread_id}})
    assert len(state.values["messages"]) == 4
    assert "Telemetry Ingest" in state.values["standalone"]  # follow-up was made self-contained
    assert len(state.values["trace"]) < 10  # trace resets per turn


async def test_stream_emits_tokens_steps_and_final(kb: KnowledgeBase) -> None:
    events = [e async for e in AtlasAgent(kb).astream("What is the Kafka retention period?")]
    tokens = "".join(e["text"] for e in events if e["type"] == "token")
    assert events[-1]["type"] == "final"
    assert tokens.strip() == events[-1]["result"]["answer"].strip()
    assert {e["node"] for e in events if e["type"] == "step"} >= {"analyze", "retrieve", "generate"}
