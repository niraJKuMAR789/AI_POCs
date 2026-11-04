"""Adapters that boot Atlas, Quill and Aegis, instrument them for Argus, and serve one request at a time.

Each app runs exactly as it does standalone (its own settings, data and provider factory). The adapter
only (1) swaps provider objects for traced and fault-injectable wrappers and (2) opens the root agent span.
`provider="nvidia"` uses NVIDIA NIM through each app's own client; `"offline"` uses each app's
deterministic baseline, so the whole drill runs anywhere.
"""

from __future__ import annotations

import dataclasses
import json
import random
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from argus import semconv

from mission_control.faults import Fault, PromptRegression, ProviderTimeouts, SlowReranker
from mission_control.instrument import (
    TracedDatabase,
    TracedEmbedder,
    TracedGuard,
    TracedLLM,
    TracedReranker,
    agent_span,
    instrument_rules_engine,
)
from mission_control.timeline import Timeline

Provider = Literal["nvidia", "offline"]


def project_root(module: Any) -> Path:
    """Repository directory of an editable-installed sibling project (src/<pkg>/__init__.py → root)."""
    return Path(module.__file__).resolve().parents[2]


def _system(provider: Provider) -> str:
    return "nvidia_nim" if provider == "nvidia" else "offline"


def _model(provider: Provider, configured: str, inner: Any) -> str:
    # Label spans with what actually served the call: the NIM model id, or the offline baseline's name.
    return configured if provider == "nvidia" else str(getattr(inner, "name", "offline"))


@dataclass
class AppRuntime:
    name: str
    items: list[Any]
    handle: Any  # async (item, rng) -> dict of KPIs
    close: Any
    models: dict[str, str] = field(default_factory=dict)


# --- Atlas: adaptive GraphRAG --------------------------------------------------------------------------


async def boot_atlas(workdir: Path, provider: Provider, timeline: Timeline, faults: list[Fault]) -> AppRuntime:
    import atlas
    from atlas.agent import AtlasAgent
    from atlas.config import Settings
    from atlas.ingest.loaders import iter_documents
    from atlas.kb import KnowledgeBase

    root = project_root(atlas)
    settings = Settings(provider=provider, data_dir=workdir / "atlas", _env_file=None)  # type: ignore[call-arg]
    kb = await KnowledgeBase.open(settings)
    await kb.ingest(iter_documents(root / "data" / "corpus"))  # ingestion is not part of the drill's traffic

    p = kb.providers
    reranker: Any = p.reranker
    for fault in faults:
        if fault.app == "atlas" and fault.cause_value == "rerank":
            reranker = SlowReranker(reranker, timeline, fault, delay_s=fault_delay(provider, 0.35, 2.5))
    generator_model = _model(provider, settings.generator_model, p.generator)
    fast_model = _model(provider, settings.fast_model, p.fast)
    kb.providers = dataclasses.replace(
        p,
        generator=TracedLLM(p.generator, model=generator_model, system=_system(provider)),
        fast=TracedLLM(p.fast, model=fast_model, system=_system(provider)),
        embedder=TracedEmbedder(p.embedder),
        reranker=TracedReranker(reranker),
    )
    agent = AtlasAgent(kb)
    questions = [json.loads(line)["question"] for line in (root / "evals" / "golden.jsonl").read_text().splitlines()]

    async def handle(question: str, rng: random.Random) -> dict[str, Any]:
        with agent_span("atlas", question) as s:
            result = await agent.ask(question, uuid.uuid4().hex)
            s.set("atlas.route", result.route).set("atlas.grounded", result.grounded)
            s.set("atlas.citations", len(result.citations)).set(semconv.OUTPUT, result.answer[:8000])
            return {"route": result.route, "grounded": result.grounded}

    return AppRuntime("atlas", questions, handle, kb.close, {"generator": generator_model, "fast": fast_model})


# --- Quill: governed text-to-SQL -----------------------------------------------------------------------

ROLES: list[tuple[str, dict[str, Any], float]] = [
    ("analyst", {}, 0.7),
    ("claims_auditor", {}, 0.15),
    ("provider_portal", {"provider_id": 7}, 0.15),
]


async def boot_quill(workdir: Path, provider: Provider, timeline: Timeline, faults: list[Fault]) -> AppRuntime:
    import quill
    from quill.config import Settings
    from quill.db.generate import generate
    from quill.llm.factory import Models
    from quill.pipeline import Text2SQL

    root = project_root(quill)
    db_path = workdir / "quill" / "claims.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    committed = root / "data" / "claims.db"
    if committed.exists():
        shutil.copy(committed, db_path)
    else:
        generate(db_path)
    settings = Settings(  # type: ignore[call-arg]
        provider=provider,
        db_path=db_path,
        semantic_layer=root / "config" / "semantic_layer.yaml",
        policies=root / "config" / "policies.yaml",
        examples_path=root / "data" / "splits" / "train.jsonl",
        _env_file=None,
    )
    pipeline = Text2SQL.from_settings(settings)
    sql_llm: Any = pipeline.models.sql
    for fault in faults:
        if fault.app == "quill" and fault.cause_dimension == "prompt_version":
            sql_llm = PromptRegression(sql_llm, version=fault.cause_value, rate=0.6, seed=11)
    sql_model = _model(provider, settings.sql_model, pipeline.models.sql)
    answer_model = _model(provider, settings.answer_model, pipeline.models.answer)
    pipeline.models = Models(
        sql=TracedLLM(sql_llm, model=sql_model, system=_system(provider)),
        answer=TracedLLM(pipeline.models.answer, model=answer_model, system=_system(provider)),
    )
    pipeline.guard = TracedGuard(pipeline.guard)  # type: ignore[assignment]
    pipeline.db = TracedDatabase(pipeline.db)  # type: ignore[assignment]
    questions = [
        json.loads(line)["question"] for line in (root / "data" / "splits" / "test.jsonl").read_text().splitlines()
    ]

    async def handle(question: str, rng: random.Random) -> dict[str, Any]:
        role, context, _ = rng.choices(ROLES, weights=[w for *_, w in ROLES])[0]
        with agent_span("quill", question, **{"quill.role": role}) as s:
            result = await pipeline.ask(question, role=role, context=context)
            s.set("quill.status", result.status).set("quill.attempts", len(result.attempts))
            s.set(semconv.OUTPUT, (result.executed_sql or result.answer)[:8000])
            return {"status": result.status, "attempts": len(result.attempts)}

    return AppRuntime("quill", questions, handle, lambda: None, {"sql": sql_model, "answer": answer_model})


# --- Aegis: multi-agent claims auditor -----------------------------------------------------------------


SEEDS = {"clinical": 31, "coding": 37}


async def boot_aegis(workdir: Path, provider: Provider, timeline: Timeline, faults: list[Fault]) -> AppRuntime:
    import aegis
    from aegis.config import Settings
    from aegis.models import Claim
    from aegis.service import AuditService, build_llms

    root = project_root(aegis)
    settings = Settings(  # type: ignore[call-arg]
        provider=provider,
        reference=root / "config" / "reference.yaml",
        policy_manual=root / "config" / "policy_manual.md",
        registry=root / "data" / "registry.json",
        state_dir=workdir / "aegis",
        _env_file=None,
    )
    models = {"clinical": settings.clinical_model, "coding": settings.coding_model}
    llms: dict[str, Any] = {}
    labels: dict[str, str] = {}
    for role, inner in build_llms(settings).items():
        wrapped: Any = inner
        for fault in faults:
            if fault.app == "aegis" and fault.expected_category == "provider_degradation":
                wrapped = ProviderTimeouts(
                    wrapped, timeline, fault, rate=0.7, delay_s=fault_delay(provider, 0.05, 5.0), seed=SEEDS[role]
                )
        labels[role] = _model(provider, models[role], inner)
        llms[role] = TracedLLM(wrapped, model=labels[role], system=_system(provider))
    service = await AuditService.create(settings, llms=llms, durable=False)
    instrument_rules_engine(service.engine)
    rows = [json.loads(line) for line in (root / "data" / "claims_labeled.jsonl").read_text().splitlines()]
    claims = [Claim.model_validate(r["claim"]) for r in rows]
    counter = iter(range(10**9))

    async def handle(claim: Claim, rng: random.Random) -> dict[str, Any]:
        claim = claim.model_copy(update={"claim_id": f"{claim.claim_id}-{next(counter):05d}"})
        with agent_span("aegis", f"audit claim {claim.claim_id}", **{"aegis.claim_id": claim.claim_id}) as s:
            result = await service.audit(claim)
            escalated = result.status == "pending_review"
            s.set("aegis.status", result.status)
            outcome = result.decision.outcome.value if result.decision else "pending_review"
            s.set(semconv.OUTPUT, outcome)
            return {"status": result.status, "escalated": escalated}

    return AppRuntime("aegis", claims, handle, service.close, labels)


def fault_delay(provider: Provider, offline_s: float, nim_s: float) -> float:
    """Injected delays are sized to the provider's normal latency so the fault is equally 'big' in both."""
    return offline_s if provider == "offline" else nim_s


BOOT = {"atlas": boot_atlas, "quill": boot_quill, "aegis": boot_aegis}
