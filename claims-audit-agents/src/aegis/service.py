"""AuditService: wires reference data, rules, agents, ledger and the LangGraph workflow together."""

from __future__ import annotations

from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import BaseModel

from aegis.adjudicate import Thresholds
from aegis.agents.reviewers import ReviewerAgent
from aegis.audit.ledger import AuditLedger
from aegis.config import Settings
from aegis.llm.base import LLM
from aegis.models import AuditDecision, Claim, LineOutcome
from aegis.policy.manual import PolicyManual
from aegis.reference import Reference
from aegis.rules.engine import RulesEngine
from aegis.workflow.graph import build_graph, to_decision


class PendingReview(BaseModel):
    claim_id: str
    lines: list[dict[str, Any]]
    findings: list[dict[str, Any]]
    reviews: list[dict[str, Any]]


class AuditResult(BaseModel):
    status: str  # "decided" | "pending_review"
    decision: AuditDecision | None = None
    pending: PendingReview | None = None


def build_llms(settings: Settings) -> dict[str, LLM]:
    if settings.offline:
        from aegis.llm.offline import OfflineLLM

        return {"clinical": OfflineLLM(), "coding": OfflineLLM()}
    from aegis.llm.openai_compat import OpenAICompatibleLLM

    secret = settings.nvidia_api_key if settings.provider == "nvidia" else settings.api_key
    key = secret.get_secret_value() if secret else None
    if settings.provider == "nvidia" and not key:
        raise RuntimeError("NVIDIA_API_KEY is not set (build.nvidia.com) or use AEGIS_PROVIDER=offline")
    guided = settings.provider == "nvidia"
    return {
        "clinical": OpenAICompatibleLLM(
            settings.clinical_model, base_url=settings.base_url, api_key=key, guided_json=guided
        ),
        "coding": OpenAICompatibleLLM(
            settings.coding_model, base_url=settings.base_url, api_key=key, guided_json=guided
        ),
    }


@dataclass
class AuditService:
    settings: Settings
    reference: Reference
    manual: PolicyManual
    ledger: AuditLedger
    llms: dict[str, LLM]
    checkpointer: BaseCheckpointSaver = field(default_factory=InMemorySaver)
    _stack: AsyncExitStack | None = None

    @classmethod
    async def create(
        cls, settings: Settings, *, llms: dict[str, LLM] | None = None, durable: bool = True
    ) -> AuditService:
        settings.state_dir.mkdir(parents=True, exist_ok=True)
        stack = AsyncExitStack()
        checkpointer: BaseCheckpointSaver = InMemorySaver()
        if durable:
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

            checkpointer = await stack.enter_async_context(
                AsyncSqliteSaver.from_conn_string(str(settings.state_dir / "checkpoints.sqlite"))
            )
        service = cls(
            settings=settings,
            reference=Reference.load(settings.reference, settings.registry),
            manual=PolicyManual.load(settings.policy_manual),
            ledger=AuditLedger(settings.state_dir / "ledger.sqlite"),
            llms=llms or build_llms(settings),
            checkpointer=checkpointer,
            _stack=stack,
        )
        service._compile()
        return service

    def _compile(self) -> None:
        agents = {name: ReviewerAgent(name, llm, self.manual, self.reference) for name, llm in self.llms.items()}
        self.engine = RulesEngine(self.reference)
        self.graph = build_graph(
            self.engine,
            agents,
            self.ledger,
            Thresholds(self.settings.override_threshold, self.settings.uphold_threshold),
            self.checkpointer,
        )

    async def close(self) -> None:
        if self._stack:
            await self._stack.aclose()

    @staticmethod
    def _config(claim_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": claim_id}}

    async def audit(self, claim: Claim) -> AuditResult:
        config = self._config(claim.claim_id)
        existing = await self.graph.aget_state(config)
        if existing.values:
            return await self.status(claim.claim_id)  # idempotent: one audit per claim id
        await self.graph.ainvoke({"claim": claim.model_dump(mode="json")}, config)
        return await self.status(claim.claim_id)

    async def status(self, claim_id: str) -> AuditResult:
        snapshot = await self.graph.aget_state(self._config(claim_id))
        if not snapshot.values:
            raise KeyError(f"unknown claim {claim_id}")
        for task in snapshot.tasks:
            for intr in task.interrupts:
                return AuditResult(status="pending_review", pending=PendingReview.model_validate(intr.value))
        return AuditResult(status="decided", decision=to_decision(snapshot.values))

    async def resume(self, claim_id: str, *, reviewer: str, decisions: dict[int, str], note: str = "") -> AuditResult:
        current = await self.status(claim_id)
        if current.pending is None:
            raise ValueError(f"claim {claim_id} is not awaiting review")
        expected = {line["line_no"] for line in current.pending.lines}
        if set(decisions) != expected:
            raise ValueError(f"decisions must cover exactly lines {sorted(expected)} (hard denials are final)")
        allowed = {LineOutcome.PAY.value, LineOutcome.DENY.value}
        if any(v not in allowed for v in decisions.values()):
            raise ValueError("decision values must be 'pay' or 'deny'")
        await self.graph.ainvoke(
            Command(
                resume={"reviewer": reviewer, "decisions": {str(k): v for k, v in decisions.items()}, "note": note}
            ),
            self._config(claim_id),
        )
        return await self.status(claim_id)
