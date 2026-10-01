import sqlite3
from typing import Any

import pytest
from tests.conftest import ScriptedReviewer, pick, scripted_service

from aegis.config import Settings
from aegis.models import ClaimOutcome, Resolution
from aegis.service import AuditService


async def test_clean_claim_is_approved_without_agents(settings: Settings, labeled: list[dict[str, Any]]) -> None:
    reviewer = ScriptedReviewer()
    svc = await scripted_service(settings, reviewer)
    result = await svc.audit(pick(labeled, "clean_diabetes_visit"))
    assert result.decision and result.decision.outcome == ClaimOutcome.APPROVED
    assert reviewer.calls == []  # no LLM cost for clean claims
    await svc.close()


async def test_hard_denial_never_reaches_agents(settings: Settings, labeled: list[dict[str, Any]]) -> None:
    reviewer = ScriptedReviewer()
    svc = await scripted_service(settings, reviewer)
    result = await svc.audit(pick(labeled, "timely_filing"))
    assert result.decision and result.decision.outcome == ClaimOutcome.DENIED and reviewer.calls == []
    await svc.close()


async def test_parallel_reviewers_on_one_claim(settings: Settings, labeled: list[dict[str, Any]]) -> None:
    claim = pick(labeled, next(r["scenario"] for r in labeled if r["scenario"].startswith("mri_brain_necessity")))
    claim = claim.model_copy(
        update={
            "lines": [*claim.lines, claim.lines[0].model_copy(update={"line_no": 3, "cpt": "99215", "charge": 240.0})]
        }
    )
    claim.lines[0].cpt = "99213"
    reviewer = ScriptedReviewer(resolution=Resolution.UPHOLD, confidence=0.9)
    svc = await scripted_service(settings, reviewer)
    result = await svc.audit(claim)
    assert result.decision is not None
    assert {r.reviewer for r in result.decision.reviews} == {"clinical", "coding"}
    assert len(reviewer.calls) == 2
    await svc.close()


async def test_human_review_survives_restart(settings: Settings, labeled: list[dict[str, Any]]) -> None:
    claim = pick(labeled, "mri_brain_no_note")
    reviewer = ScriptedReviewer(resolution=Resolution.ESCALATE)
    first = await scripted_service(settings, reviewer, durable=True)
    pending = await first.audit(claim)
    assert pending.status == "pending_review" and pending.pending
    assert [ln["line_no"] for ln in pending.pending.lines] == [2]
    assert first.ledger.pending_claims() == [claim.claim_id]
    await first.close()

    second = await scripted_service(settings, reviewer, durable=True)  # new process, same state dir
    with pytest.raises(ValueError, match="exactly lines"):
        await second.resume(claim.claim_id, reviewer="rn.lee", decisions={1: "deny", 2: "pay"})
    done = await second.resume(claim.claim_id, reviewer="rn.lee", decisions={2: "deny"}, note="no documentation")
    assert done.decision and done.decision.outcome == ClaimOutcome.PARTIAL
    assert done.decision.lines[1].decided_by == "human:rn.lee"
    assert second.ledger.pending_claims() == []
    events = [e["event"] for e in second.ledger.events(claim.claim_id)]
    assert events == ["claim_received", "rules_evaluated", "agent_review", "adjudicated", "human_decision", "finalized"]
    await second.close()


async def test_audit_is_idempotent(service: AuditService, labeled: list[dict[str, Any]]) -> None:
    claim = pick(labeled, "clean_cardiology")
    await service.audit(claim)
    await service.audit(claim)
    assert [e["event"] for e in service.ledger.events(claim.claim_id)].count("claim_received") == 1


async def test_ledger_detects_tampering(service: AuditService, labeled: list[dict[str, Any]]) -> None:
    for scenario in ("clean_cardiology", "duplicate", "excess_units"):
        await service.audit(pick(labeled, scenario))
    assert service.ledger.verify() == (True, None)
    conn = sqlite3.connect(service.settings.state_dir / "ledger.sqlite")
    conn.execute("UPDATE audit_events SET payload = replace(payload, 'denied', 'approved') WHERE seq = 8")
    conn.commit()
    ok, bad = service.ledger.verify()
    assert not ok and bad == 8
