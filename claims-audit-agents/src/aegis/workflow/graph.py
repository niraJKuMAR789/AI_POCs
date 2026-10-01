"""LangGraph orchestration of the audit.

    START → intake → rules ─┬─ no soft findings ──────────────────────┐
                            └─ Send per reviewer (parallel):          │
                                 clinical reviewer ─┐                 │
                                 coding auditor  ───┴→ adjudicate ←───┘
                                                         ├─ all lines decided → finalize → END
                                                         └─ lines need review → human_review (interrupt) → finalize

Human review is a LangGraph `interrupt`. With the SQLite checkpointer the paused audit survives
restarts and is resumed by claim id. Every step is written to the hash-chained audit ledger.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, cast

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send, interrupt
from typing_extensions import TypedDict

from aegis.adjudicate import Thresholds, adjudicate
from aegis.agents.reviewers import ReviewerAgent
from aegis.audit.ledger import AuditLedger
from aegis.models import (
    AgentReview,
    AuditDecision,
    Claim,
    ClaimOutcome,
    Finding,
    LineDecision,
    LineOutcome,
    Severity,
    claim_outcome,
)
from aegis.rules.engine import RulesEngine


class AuditState(TypedDict, total=False):
    claim: dict[str, Any]
    findings: list[dict[str, Any]]
    reviews: Annotated[list[dict[str, Any]], operator.add]
    lines: list[dict[str, Any]]
    outcome: str
    human_reviewer: str | None


class ReviewTask(TypedDict):
    claim: dict[str, Any]
    reviewer: str
    findings: list[dict[str, Any]]


def summarize(decision: AuditDecision) -> str:
    head = f"Claim {decision.claim_id}: {decision.outcome.value.replace('_', ' ')}."
    parts = []
    for line in decision.lines:
        reason = f" ({line.reasons[0]})" if line.reasons and line.outcome != LineOutcome.PAY else ""
        units = f", {line.allowed_units} unit(s) allowed" if line.outcome == LineOutcome.ADJUST else ""
        parts.append(f"line {line.line_no} {line.cpt}: {line.outcome.value}{units}{reason}")
    return head + " " + "; ".join(parts)


def build_graph(
    engine: RulesEngine,
    agents: dict[str, ReviewerAgent],
    ledger: AuditLedger,
    thresholds: Thresholds,
    checkpointer: BaseCheckpointSaver,
) -> Any:

    async def intake(state: AuditState) -> dict[str, Any]:
        claim = Claim.model_validate(state["claim"])
        ledger.append(claim.claim_id, "system", "claim_received", claim.model_dump(mode="json"))
        return {"claim": claim.model_dump(mode="json"), "reviews": []}

    async def rules(state: AuditState) -> dict[str, Any]:
        claim = Claim.model_validate(state["claim"])
        findings = engine.evaluate(claim)
        ledger.append(
            claim.claim_id,
            "rules-engine",
            "rules_evaluated",
            {"findings": [f.model_dump(mode="json") for f in findings]},
        )
        return {"findings": [f.model_dump(mode="json") for f in findings]}

    def route(state: AuditState) -> list[Send] | str:
        findings = [Finding.model_validate(f) for f in state["findings"]]
        if any(f.severity == Severity.HARD and f.line_no is None for f in findings):
            return "adjudicate"  # claim-level hard denial: nothing for agents to change
        hard_denied = {f.line_no for f in findings if f.severity == Severity.HARD and f.adjusted_units is None}
        tasks: dict[str, list[dict[str, Any]]] = {}
        for f in findings:  # soft findings on lines already denied by a hard rule need no review
            if f.severity == Severity.SOFT and f.reviewer in agents and f.line_no not in hard_denied:
                tasks.setdefault(f.reviewer, []).append(f.model_dump(mode="json"))
        if not tasks:
            return "adjudicate"
        return [Send("review", ReviewTask(claim=state["claim"], reviewer=r, findings=fs)) for r, fs in tasks.items()]

    async def review(task: ReviewTask) -> dict[str, Any]:
        claim = Claim.model_validate(task["claim"])
        findings = [Finding.model_validate(f) for f in task["findings"]]
        agent = agents[task["reviewer"]]
        reviews = await agent.review(claim, findings)
        ledger.append(
            claim.claim_id,
            f"agent:{agent.name}",
            "agent_review",
            {"model": agent.llm.name, "reviews": [r.model_dump(mode="json") for r in reviews]},
        )
        return {"reviews": [r.model_dump(mode="json") for r in reviews]}

    async def adjudicate_node(state: AuditState) -> dict[str, Any]:
        claim = Claim.model_validate(state["claim"])
        lines = adjudicate(
            claim,
            [Finding.model_validate(f) for f in state["findings"]],
            [AgentReview.model_validate(r) for r in state.get("reviews", [])],
            thresholds,
        )
        outcome = claim_outcome(lines)
        ledger.append(
            claim.claim_id,
            "adjudicator",
            "adjudicated",
            {"outcome": outcome.value, "lines": [ln.model_dump(mode="json") for ln in lines]},
        )
        return {"lines": [ln.model_dump(mode="json") for ln in lines], "outcome": outcome.value}

    def needs_human(state: AuditState) -> str:
        return "human_review" if state["outcome"] == ClaimOutcome.PENDED.value else "finalize"

    async def human_review(state: AuditState) -> dict[str, Any]:
        lines = [LineDecision.model_validate(ln) for ln in state["lines"]]
        pending = [ln.model_dump(mode="json") for ln in lines if ln.outcome == LineOutcome.REVIEW]
        # Nothing is written before interrupt(): this node re-runs from the top when resumed.
        answer = interrupt(
            {
                "claim_id": state["claim"]["claim_id"],
                "lines": pending,
                "findings": state["findings"],
                "reviews": state.get("reviews", []),
            }
        )
        decided = {int(k): LineOutcome(v) for k, v in answer["decisions"].items()}
        updated = []
        for ln in lines:
            if ln.outcome == LineOutcome.REVIEW:
                ln.outcome = decided[ln.line_no]
                ln.allowed_units = 0 if ln.outcome == LineOutcome.DENY else (ln.allowed_units or 1)
                ln.reasons.append(f"human ({answer['reviewer']}): {answer.get('note', '')}".strip())
                ln.decided_by = f"human:{answer['reviewer']}"
            updated.append(ln)
        outcome = claim_outcome(updated)
        ledger.append(
            state["claim"]["claim_id"],
            f"human:{answer['reviewer']}",
            "human_decision",
            {"decisions": answer["decisions"], "note": answer.get("note", ""), "outcome": outcome.value},
        )
        return {
            "lines": [ln.model_dump(mode="json") for ln in updated],
            "outcome": outcome.value,
            "human_reviewer": answer["reviewer"],
        }

    async def finalize(state: AuditState) -> dict[str, Any]:
        decision = to_decision(dict(state))
        ledger.append(
            decision.claim_id, "system", "finalized", {"outcome": decision.outcome.value, "summary": decision.summary}
        )
        return {}

    g = StateGraph(AuditState)
    g.add_node("intake", intake)
    g.add_node("rules", rules)
    g.add_node("review", cast(Any, review))  # invoked via Send with a ReviewTask payload
    g.add_node("adjudicate", adjudicate_node)
    g.add_node("human_review", human_review)
    g.add_node("finalize", finalize)
    g.add_edge(START, "intake")
    g.add_edge("intake", "rules")
    g.add_conditional_edges("rules", route, ["review", "adjudicate"])
    g.add_edge("review", "adjudicate")
    g.add_conditional_edges("adjudicate", needs_human, ["human_review", "finalize"])
    g.add_edge("human_review", "finalize")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer)


def to_decision(state: dict[str, Any]) -> AuditDecision:
    decision = AuditDecision(
        claim_id=state["claim"]["claim_id"],
        outcome=ClaimOutcome(state["outcome"]),
        lines=[LineDecision.model_validate(ln) for ln in state["lines"]],
        findings=[Finding.model_validate(f) for f in state["findings"]],
        reviews=[AgentReview.model_validate(r) for r in state.get("reviews", [])],
        human_reviewer=state.get("human_reviewer"),
    )
    decision.summary = summarize(decision)
    return decision
