"""Deterministic adjudication: combines rule findings and validated agent reviews into line decisions.

The LLM never decides directly. An agent's proposal is applied only when it is admissible and
confident enough; anything else goes to a human.
"""

from __future__ import annotations

from dataclasses import dataclass

from aegis.models import AgentReview, Claim, Finding, LineDecision, LineOutcome, Resolution, Severity


@dataclass(frozen=True)
class Thresholds:
    override: float = 0.80  # minimum confidence to pay a line on an agent's override
    uphold: float = 0.70  # minimum confidence to deny a line on an agent's uphold


def adjudicate(
    claim: Claim, findings: list[Finding], reviews: list[AgentReview], thresholds: Thresholds | None = None
) -> list[LineDecision]:
    thresholds = thresholds or Thresholds()
    claim_hard = [f for f in findings if f.line_no is None and f.severity == Severity.HARD]
    by_key = {(r.rule_id, r.line_no): r for r in reviews}
    decisions = []
    for line in claim.lines:
        if claim_hard:
            decisions.append(
                LineDecision(
                    line_no=line.line_no,
                    cpt=line.cpt,
                    outcome=LineOutcome.DENY,
                    reasons=[f"{f.rule_id}: {f.message}" for f in claim_hard],
                )
            )
            continue
        own = [f for f in findings if f.line_no == line.line_no]
        hard = [f for f in own if f.severity == Severity.HARD]
        denying = [f for f in hard if f.adjusted_units is None]
        if denying:
            decisions.append(
                LineDecision(
                    line_no=line.line_no,
                    cpt=line.cpt,
                    outcome=LineOutcome.DENY,
                    reasons=[f"{f.rule_id}: {f.message}" for f in denying],
                )
            )
            continue
        units = min([line.units] + [f.adjusted_units for f in hard if f.adjusted_units is not None])
        outcome = LineOutcome.ADJUST if units < line.units else LineOutcome.PAY
        reasons = [f"{f.rule_id}: {f.message}" for f in hard]
        decided_by = "rules"
        for finding in (f for f in own if f.severity == Severity.SOFT):
            review = by_key.get((finding.rule_id, finding.line_no))
            verdict = _apply(review, thresholds)
            reasons.append(f"{finding.rule_id}: {_reason(review, verdict)}")
            if review is not None:
                decided_by = f"agent:{review.reviewer}"
            if verdict == LineOutcome.DENY or outcome == LineOutcome.DENY:
                outcome = LineOutcome.DENY
            elif verdict == LineOutcome.REVIEW:
                outcome = LineOutcome.REVIEW
        decisions.append(
            LineDecision(
                line_no=line.line_no,
                cpt=line.cpt,
                outcome=outcome,
                allowed_units=0 if outcome == LineOutcome.DENY else units,
                reasons=reasons,
                decided_by=decided_by,
            )
        )
    return decisions


def _apply(review: AgentReview | None, t: Thresholds) -> LineOutcome:
    if review is None or review.resolution == Resolution.ESCALATE:
        return LineOutcome.REVIEW
    if review.resolution == Resolution.OVERRIDE:
        ok = review.confidence >= t.override and not review.validation_errors
        return LineOutcome.PAY if ok else LineOutcome.REVIEW
    return LineOutcome.DENY if review.confidence >= t.uphold else LineOutcome.REVIEW


def _reason(review: AgentReview | None, verdict: LineOutcome) -> str:
    if review is None:
        return "no agent review available; sent to human review"
    detail = f"{review.reviewer} reviewer {review.resolution.value} ({review.confidence:.2f}): {review.rationale}"
    if review.validation_errors:
        detail += f" [validation: {'; '.join(review.validation_errors)}]"
    return f"{detail} -> {verdict.value}"
