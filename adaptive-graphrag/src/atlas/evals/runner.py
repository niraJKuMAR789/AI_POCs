"""Evaluation runner: Atlas agent vs. a naive dense-only RAG baseline on a golden set.

Writes a Markdown report (committed to the repo as evidence) plus raw JSON results.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from atlas.agent import AtlasAgent
from atlas.config import get_settings
from atlas.evals.metrics import abstained, correctness, faithfulness, reciprocal_rank, source_recall
from atlas.ingest.loaders import iter_documents
from atlas.kb import KnowledgeBase
from atlas.logging import get_logger
from atlas.models import RetrievedContext
from atlas.prompts import generate_messages

log = get_logger(__name__)


class EvalItem(BaseModel):
    id: str
    type: str
    question: str
    reference: str
    expected_sources: list[str]


class EvalRow(BaseModel):
    id: str
    type: str
    system: str
    answer: str
    correctness: float
    faithfulness: float
    recall: float | None
    mrr: float | None
    abstained: bool
    latency_s: float
    route: str = ""
    steps: int = 0


async def naive_rag(kb: KnowledgeBase, question: str) -> tuple[str, list[RetrievedContext]]:
    """Baseline: dense top-k → single generation. No hybrid search, graph, rerank, grading or reflection."""
    vector = await kb.providers.embedder.embed_query(question)
    hits = await kb.chunk_vectors.search(vector, kb.settings.rerank_top_n)
    chunks = kb.docstore.get([cid for cid, _, _ in hits])
    contexts = [
        RetrievedContext(id=c.id, kind="chunk", text=c.text, source=c.source, title=c.title)
        for cid, _, _ in hits
        if (c := chunks.get(cid))
    ]
    answer = await kb.providers.generator.generate(generate_messages(question, contexts, []))
    return answer, contexts


async def atlas_rag(agent: AtlasAgent, question: str) -> tuple[str, list[RetrievedContext], str, int]:
    thread_id = uuid.uuid4().hex
    result = await agent.ask(question, thread_id)
    state = await agent.graph.aget_state({"configurable": {"thread_id": thread_id}})
    contexts = [RetrievedContext.model_validate(c) for c in state.values.get("contexts", [])]
    return result.answer, contexts, result.route, len(result.trace)


def _sources(contexts: list[RetrievedContext]) -> list[str]:
    """Ordered document sources; graph facts are attributed through their provenance chunks."""
    return list(dict.fromkeys(c.source for c in contexts if c.kind in ("chunk", "web")))


async def _score(
    kb: KnowledgeBase,
    item: EvalItem,
    system: str,
    answer: str,
    contexts: list[RetrievedContext],
    latency: float,
    route: str = "",
    steps: int = 0,
) -> EvalRow:
    judge = kb.providers.generator
    sources = _sources(contexts)
    corr, faith = await asyncio.gather(
        correctness(judge, item.question, answer, item.reference), faithfulness(judge, answer, contexts)
    )
    is_abstain = abstained(answer)
    if item.type == "unanswerable":
        corr = 1.0 if is_abstain else 0.0
    return EvalRow(
        id=item.id,
        type=item.type,
        system=system,
        answer=answer,
        correctness=corr,
        faithfulness=faith,
        recall=source_recall(sources, item.expected_sources),
        mrr=reciprocal_rank(sources, item.expected_sources),
        abstained=is_abstain,
        latency_s=round(latency, 2),
        route=route,
        steps=steps,
    )


async def evaluate(kb: KnowledgeBase, items: list[EvalItem], concurrency: int = 4) -> list[EvalRow]:
    agent = AtlasAgent(kb)
    semaphore = asyncio.Semaphore(concurrency)

    async def run(item: EvalItem) -> list[EvalRow]:
        async with semaphore:
            t0 = time.perf_counter()
            base_answer, base_ctx = await naive_rag(kb, item.question)
            t1 = time.perf_counter()
            answer, ctx, route, steps = await atlas_rag(agent, item.question)
            t2 = time.perf_counter()
            rows = await asyncio.gather(
                _score(kb, item, "naive-rag", base_answer, base_ctx, t1 - t0),
                _score(kb, item, "atlas", answer, ctx, t2 - t1, route, steps),
            )
            log.info("eval.item", id=item.id, atlas=rows[1].correctness, baseline=rows[0].correctness)
            return list(rows)

    results = await asyncio.gather(*(run(i) for i in items))
    return [row for rows in results for row in rows]


def _mean(values: list[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    return round(statistics.fmean(clean), 3) if clean else None


def summarize(rows: list[EvalRow]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for system in dict.fromkeys(r.system for r in rows):
        rs = [r for r in rows if r.system == system]
        latencies = sorted(r.latency_s for r in rs)
        summary[system] = {
            "correctness": _mean([r.correctness for r in rs]),
            "faithfulness": _mean([r.faithfulness for r in rs]),
            "source_recall": _mean([r.recall for r in rs]),
            "mrr": _mean([r.mrr for r in rs]),
            "multi_hop_correctness": _mean([r.correctness for r in rs if r.type == "multi-hop"]),
            "abstention_accuracy": _mean([float(r.abstained) for r in rs if r.type == "unanswerable"]),
            "latency_p50_s": latencies[len(latencies) // 2],
            "latency_p95_s": latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))],
        }
    return summary


def render_report(rows: list[EvalRow], summary: dict[str, dict[str, Any]], meta: dict[str, str]) -> str:
    metrics = list(next(iter(summary.values())).keys())
    systems = list(summary)
    lines = ["# Atlas evaluation report", ""]
    lines += [f"- **{k}**: `{v}`" for k, v in meta.items()]
    lines += [
        "",
        "## Summary",
        "",
        "| metric | " + " | ".join(systems) + " |",
        "|---|" + "---|" * len(systems),
    ]
    for m in metrics:
        lines.append(f"| {m} | " + " | ".join(str(summary[s][m]) for s in systems) + " |")
    lines += [
        "",
        "## Per question",
        "",
        "| id | type | system | correctness | faithfulness | recall | route | latency (s) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda r: (r.id, r.system)):
        lines.append(
            f"| {r.id} | {r.type} | {r.system} | {r.correctness:.2f} | {r.faithfulness:.2f} | "
            f"{'-' if r.recall is None else f'{r.recall:.2f}'} | {r.route or '-'} | {r.latency_s} |"
        )
    return "\n".join(lines) + "\n"


async def run_eval(dataset: Path, out: Path, corpus: Path = Path("data/corpus")) -> dict[str, dict[str, Any]]:
    settings = get_settings()
    items = [EvalItem.model_validate_json(line) for line in dataset.read_text().splitlines() if line.strip()]
    kb = await KnowledgeBase.open(settings)
    try:
        if kb.stats()["chunks"] == 0:
            await kb.ingest(iter_documents(corpus))
        rows = await evaluate(kb, items)
    finally:
        await kb.close()
    summary = summarize(rows)
    meta = {
        "provider": settings.provider,
        "generator": settings.generator_model if not settings.offline else "offline-heuristic",
        "fast_model": settings.fast_model if not settings.offline else "offline-heuristic",
        "embeddings": settings.embedding_model if not settings.offline else "hash-embed-512",
        "reranker": settings.rerank_model if not settings.offline else "term-overlap",
        "questions": str(len(items)),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_report(rows, summary, meta))
    out.with_suffix(".json").write_text(json.dumps([r.model_dump() for r in rows], indent=2))
    print(json.dumps(summary, indent=2))
    return summary
