"""The Atlas agent: an adaptive, corrective, self-reflective GraphRAG workflow in LangGraph.

    START → analyze ─┬─ direct ──────────────────────────────┐
                     ├─ web ─→ web_search ───────────────────┤
                     └─ local|global → retrieve → grade ─┬───┤→ generate → reflect ─┬→ finalize → END
                              ↑                          │   │      ↑               │
                              └──────── rewrite ←────────┘   │      └─ regenerate ──┤
                                           ↑ (insufficient)  │                      │
                                           └─────────────────┴──── (not useful) ────┘

* Adaptive RAG: the router picks a retrieval strategy per question (local / global / web / direct)
  and decomposes multi-hop questions into sub-questions retrieved in parallel.
* Corrective RAG (CRAG): retrieved evidence is graded; weak evidence triggers query rewriting and,
  if still insufficient, a web-search fallback.
* Self-RAG: the draft answer is checked for groundedness and usefulness; unsupported drafts are
  regenerated with feedback, unhelpful ones trigger another retrieval round.
"""

from __future__ import annotations

import asyncio
import functools
import re
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Literal, TypeVar, cast

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from atlas.agent.state import AgentState
from atlas.kb import KnowledgeBase
from atlas.logging import get_logger
from atlas.models import AnswerResult, Citation, RetrievedContext, TraceStep
from atlas.prompts import (
    analyze_messages,
    direct_messages,
    generate_messages,
    grade_messages,
    grounding_messages,
    rewrite_messages,
)
from atlas.providers.base import Message
from atlas.retrieval.retriever import Retriever
from atlas.schemas import DocumentGrades, GroundednessCheck, QueryAnalysis, RewrittenQuery
from atlas.web.search import NullWebSearch, TavilySearch, WebSearch

log = get_logger(__name__)

NodeFn = TypeVar("NodeFn", bound=Callable[..., Awaitable[dict[str, Any]]])
_CITATION_RE = re.compile(r"\[(\d+)\]")
UNVERIFIED_NOTE = "\n\n_Note: parts of this answer could not be fully verified against the sources._"


def traced(name: str) -> Callable[[NodeFn], NodeFn]:
    """Time a node and append a TraceStep built from the `_summary`/`_data` keys it returns."""

    def decorator(fn: NodeFn) -> NodeFn:
        @functools.wraps(fn)
        async def wrapper(self: AtlasAgent, state: AgentState) -> dict[str, Any]:
            started = time.perf_counter()
            update = await fn(self, state)
            step = TraceStep(
                node=name,
                summary=update.pop("_summary", ""),
                latency_ms=round((time.perf_counter() - started) * 1000, 1),
                data=update.pop("_data", {}),
            )
            log.info("agent.step", node=name, summary=step.summary, ms=step.latency_ms)
            return {**update, "trace": [step.model_dump()]}

        return cast(NodeFn, wrapper)

    return decorator


def _to_messages(history: list[AnyMessage]) -> list[Message]:
    out: list[Message] = []
    for m in history:
        role: Literal["user", "assistant"] = "user" if isinstance(m, HumanMessage) else "assistant"
        out.append({"role": role, "content": str(m.content)})
    return out


def _ctx(items: list[dict[str, Any]]) -> list[RetrievedContext]:
    return [RetrievedContext.model_validate(c) for c in items]


class AtlasAgent:
    def __init__(
        self,
        kb: KnowledgeBase,
        *,
        web: WebSearch | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
    ) -> None:
        self.kb = kb
        self.settings = kb.settings
        self.providers = kb.providers
        self.retriever = Retriever(kb)
        if web is None and self.settings.web_search_enabled:
            assert self.settings.tavily_api_key is not None
            web = TavilySearch(self.settings.tavily_api_key.get_secret_value())
        self.web: WebSearch = web or NullWebSearch()
        self.web_enabled = not isinstance(self.web, NullWebSearch)
        self.graph = self._build().compile(checkpointer=checkpointer or InMemorySaver())

    # --- graph wiring ---------------------------------------------------------------

    def _build(self) -> StateGraph:
        g = StateGraph(AgentState)
        g.add_node("analyze", self.analyze)
        g.add_node("retrieve", self.retrieve)
        g.add_node("grade", self.grade)
        g.add_node("rewrite", self.rewrite)
        g.add_node("web_search", self.web_search)
        g.add_node("generate", self.generate)
        g.add_node("reflect", self.reflect)
        g.add_node("finalize", self.finalize)

        g.add_edge(START, "analyze")
        g.add_conditional_edges(
            "analyze",
            self._after_analyze,
            {"retrieve": "retrieve", "web_search": "web_search", "generate": "generate"},
        )
        g.add_edge("retrieve", "grade")
        g.add_conditional_edges(
            "grade",
            self._after_grade,
            {"generate": "generate", "rewrite": "rewrite", "web_search": "web_search"},
        )
        g.add_edge("rewrite", "retrieve")
        g.add_edge("web_search", "generate")
        g.add_conditional_edges("generate", self._after_generate, {"reflect": "reflect", "finalize": "finalize"})
        g.add_conditional_edges(
            "reflect",
            self._after_reflect,
            {"finalize": "finalize", "generate": "generate", "rewrite": "rewrite"},
        )
        g.add_edge("finalize", END)
        return g

    def _after_analyze(self, state: AgentState) -> str:
        route = state["route"]
        if route == "direct":
            return "generate"
        return "web_search" if route == "web" else "retrieve"

    def _after_grade(self, state: AgentState) -> str:
        evidence = [c for c in state["contexts"] if c["kind"] in ("chunk", "community")]
        if len(evidence) >= self.settings.min_relevant_docs:
            return "generate"
        if state["rewrites"] < self.settings.max_rewrites:
            return "rewrite"
        return "web_search" if self.web_enabled else "generate"

    def _after_generate(self, state: AgentState) -> str:
        return "finalize" if state["route"] == "direct" else "reflect"

    def _after_reflect(self, state: AgentState) -> str:
        if state["grounded"] and state["answers_question"]:
            return "finalize"
        if not state["grounded"] and state["generation_attempts"] <= self.settings.max_generation_retries:
            return "generate"
        if not state["answers_question"] and state["rewrites"] < self.settings.max_rewrites:
            return "rewrite"
        return "finalize"

    # --- nodes -----------------------------------------------------------------------

    @traced("analyze")
    async def analyze(self, state: AgentState) -> dict[str, Any]:
        history = _to_messages(state["messages"][:-1])
        analysis = await self.providers.fast.structured(
            analyze_messages(state["question"], history, self.kb.topics()), QueryAnalysis
        )
        route = analysis.route
        if route == "web" and not self.web_enabled:
            route = "local"  # no web tool configured: fall back to the knowledge base
        standalone = analysis.standalone_question.strip() or state["question"]
        return {
            "standalone": standalone,
            "route": route,
            "sub_questions": analysis.sub_questions[:3],
            "queries": [standalone],
            "_summary": f"route={route}"
            + (f", {len(analysis.sub_questions)} sub-questions" if analysis.sub_questions else ""),
            "_data": {
                "standalone": standalone,
                "sub_questions": analysis.sub_questions,
                "rationale": analysis.rationale,
            },
        }

    @traced("retrieve")
    async def retrieve(self, state: AgentState) -> dict[str, Any]:
        # First pass fans out over sub-questions; later passes use the latest rewrite only.
        queries = [state["queries"][-1]]
        if state["rewrites"] == 0:
            queries = list(dict.fromkeys([*queries, *state.get("sub_questions", [])]))
        mode = "global" if state["route"] == "global" else "local"
        results = await asyncio.gather(*(self.retriever.retrieve(q, mode) for q in queries))
        merged: dict[str, RetrievedContext] = {}
        for ctx in (c for batch in results for c in batch):
            if ctx.id not in merged or ctx.score > merged[ctx.id].score:
                merged[ctx.id] = ctx
        candidates = list(merged.values())
        kinds = {k: sum(c.kind == k for c in candidates) for k in ("chunk", "graph", "community")}
        return {
            "candidates": [c.model_dump() for c in candidates],
            "_summary": f"{len(candidates)} candidates from {len(queries)} quer{'y' if len(queries) == 1 else 'ies'}",
            "_data": {"queries": queries, "mode": mode, **kinds},
        }

    @traced("grade")
    async def grade(self, state: AgentState) -> dict[str, Any]:
        candidates = _ctx(state["candidates"])
        if not candidates:
            return {"contexts": [], "_summary": "no candidates"}
        ids = [f"d{i}" for i in range(1, len(candidates) + 1)]
        grades = await self.providers.fast.structured(
            grade_messages(state["standalone"], candidates, ids), DocumentGrades
        )
        keep = set(grades.relevant_ids)
        relevant = [c for doc_id, c in zip(ids, candidates, strict=True) if doc_id in keep]
        # Web context already collected in this turn (if any) is preserved across rounds.
        prior_web = [c for c in state.get("contexts", []) if c["kind"] == "web"]
        return {
            "contexts": prior_web + [c.model_dump() for c in relevant],
            "_summary": f"kept {len(relevant)}/{len(candidates)}",
            "_data": {"reasoning": grades.reasoning},
        }

    @traced("rewrite")
    async def rewrite(self, state: AgentState) -> dict[str, Any]:
        result = await self.providers.fast.structured(
            rewrite_messages(state["standalone"], state["queries"]), RewrittenQuery
        )
        query = result.query.strip() or state["standalone"]
        return {
            "queries": [*state["queries"], query],
            "rewrites": state["rewrites"] + 1,
            "_summary": f"rewrite #{state['rewrites'] + 1}: {query}",
        }

    @traced("web_search")
    async def web_search(self, state: AgentState) -> dict[str, Any]:
        try:
            results = await self.web.search(state["standalone"])
        except Exception as exc:
            log.warning("web_search.failed", error=str(exc))
            results = []
        existing = state.get("contexts", [])
        return {
            "contexts": existing + [r.model_dump() for r in results],
            "_summary": f"{len(results)} web results",
        }

    @traced("generate")
    async def generate(self, state: AgentState) -> dict[str, Any]:
        writer = get_stream_writer()
        attempts = state.get("generation_attempts", 0)
        if attempts:
            writer({"type": "reset"})
        history = _to_messages(state["messages"][:-1])
        if state["route"] == "direct":
            messages = direct_messages(state["question"], history)
        else:
            messages = generate_messages(state["standalone"], _ctx(state["contexts"]), history, state.get("feedback"))
        parts: list[str] = []
        async for token in self.providers.generator.stream(messages):
            parts.append(token)
            writer({"type": "token", "text": token})
        answer = "".join(parts).strip()
        n_sources = len(state.get("contexts", []))
        return {
            "generation": answer,
            "generation_attempts": attempts + 1,
            "_summary": f"attempt {attempts + 1}, {len(answer.split())} words, {n_sources} sources",
        }

    @traced("reflect")
    async def reflect(self, state: AgentState) -> dict[str, Any]:
        contexts = _ctx(state["contexts"])
        if not contexts:
            return {"grounded": True, "answers_question": False, "_summary": "no evidence to check against"}
        check = await self.providers.fast.structured(
            grounding_messages(state["standalone"], state["generation"], contexts), GroundednessCheck
        )
        feedback = "; ".join(check.unsupported_claims) if check.unsupported_claims else None
        return {
            "grounded": check.grounded,
            "answers_question": check.answers_question,
            "feedback": feedback,
            "_summary": f"grounded={check.grounded}, useful={check.answers_question}",
            "_data": {"unsupported_claims": check.unsupported_claims},
        }

    @traced("finalize")
    async def finalize(self, state: AgentState) -> dict[str, Any]:
        answer = state["generation"]
        grounded = state.get("grounded", True) or state["route"] == "direct"
        if not grounded:
            answer += UNVERIFIED_NOTE
        return {
            "generation": answer,
            "grounded": grounded,
            "messages": [AIMessage(content=answer)],
            "_summary": "done",
        }

    # --- public API ---------------------------------------------------------------------

    @staticmethod
    def _turn_input(question: str) -> dict[str, Any]:
        return {
            "messages": [HumanMessage(content=question)],
            "question": question,
            "sub_questions": [],
            "queries": [],
            "candidates": [],
            "contexts": [],
            "rewrites": 0,
            "generation_attempts": 0,
            "feedback": None,
            "grounded": False,
            "answers_question": False,
            "trace": None,
        }

    @staticmethod
    def _config(thread_id: str) -> RunnableConfig:
        return {"configurable": {"thread_id": thread_id}, "recursion_limit": 40}

    async def ask(self, question: str, thread_id: str | None = None) -> AnswerResult:
        thread_id = thread_id or uuid.uuid4().hex
        state = await self.graph.ainvoke(self._turn_input(question), self._config(thread_id))
        return self._result(state, thread_id)

    async def astream(self, question: str, thread_id: str | None = None) -> AsyncIterator[dict[str, Any]]:
        """Yield `step`, `token`, `reset` events, then a single `final` event with the AnswerResult."""
        thread_id = thread_id or uuid.uuid4().hex
        config = self._config(thread_id)
        async for mode, payload in self.graph.astream(
            self._turn_input(question), config, stream_mode=["updates", "custom"]
        ):
            if mode == "custom":
                yield cast(dict[str, Any], payload)
            else:
                for update in cast(dict[str, dict[str, Any] | None], payload).values():
                    for step in (update or {}).get("trace", []) or []:
                        yield {"type": "step", **step}
        snapshot = await self.graph.aget_state(config)
        yield {"type": "final", "result": self._result(snapshot.values, thread_id).model_dump()}

    @staticmethod
    def _result(state: dict[str, Any], thread_id: str) -> AnswerResult:
        contexts = _ctx(state.get("contexts", []))
        answer = state.get("generation", "")
        cited = {int(n) for n in _CITATION_RE.findall(answer)}
        citations = [
            Citation(label=str(i), source=c.source, title=c.title, kind=c.kind, snippet=c.text[:280])
            for i, c in enumerate(contexts, start=1)
            if i in cited
        ]
        return AnswerResult(
            question=state.get("question", ""),
            answer=answer,
            route=state.get("route", ""),
            citations=citations,
            grounded=bool(state.get("grounded", False)),
            trace=[TraceStep.model_validate(t) for t in state.get("trace", [])],
            thread_id=thread_id,
        )
