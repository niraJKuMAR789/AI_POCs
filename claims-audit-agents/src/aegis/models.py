"""Claims-audit domain model."""

from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, Field


class ClaimLine(BaseModel):
    line_no: int
    cpt: str
    units: int = Field(ge=1)
    charge: float = Field(ge=0)
    modifiers: list[str] = Field(default_factory=list)


class Claim(BaseModel):
    claim_id: str
    member_id: str
    provider_npi: str
    service_date: date
    submitted_date: date
    place_of_service: str = Field(description="CMS place-of-service code, e.g. 11 office, 22 outpatient, 23 ER")
    diagnosis_codes: list[str] = Field(min_length=1)
    lines: list[ClaimLine] = Field(min_length=1)
    prior_auth_number: str | None = None
    referral_number: str | None = None
    clinical_note: str | None = None


class Severity(StrEnum):
    HARD = "hard"  # deterministic outcome; never sent to an LLM
    SOFT = "soft"  # documentation can change the outcome; reviewed by an agent


class Finding(BaseModel):
    rule_id: str
    title: str
    severity: Severity
    line_no: int | None = Field(default=None, description="None = applies to the whole claim")
    message: str
    policy_section: str
    reviewer: str | None = Field(default=None, description="Agent that reviews soft findings: clinical | coding")
    adjusted_units: int | None = None


class Resolution(StrEnum):
    UPHOLD = "uphold"  # the finding stands: the line is not payable
    OVERRIDE = "override"  # documentation satisfies policy: the line is payable
    ESCALATE = "escalate"  # not enough certainty: send to a human


class AgentReview(BaseModel):
    """Validated output of a reviewer agent for one finding."""

    rule_id: str
    line_no: int | None
    reviewer: str
    resolution: Resolution
    confidence: float = Field(ge=0, le=1)
    rationale: str
    policy_citations: list[str] = Field(default_factory=list)
    evidence_quotes: list[str] = Field(default_factory=list)
    validation_errors: list[str] = Field(default_factory=list)


class LineOutcome(StrEnum):
    PAY = "pay"
    ADJUST = "adjust"  # pay reduced units
    DENY = "deny"
    REVIEW = "review"  # pending human decision


class LineDecision(BaseModel):
    line_no: int
    cpt: str
    outcome: LineOutcome
    allowed_units: int = 0
    reasons: list[str] = Field(default_factory=list)
    decided_by: str = "rules"


class ClaimOutcome(StrEnum):
    APPROVED = "approved"
    PARTIAL = "partially_approved"
    DENIED = "denied"
    PENDED = "pended_for_review"


class AuditDecision(BaseModel):
    claim_id: str
    outcome: ClaimOutcome
    lines: list[LineDecision]
    findings: list[Finding]
    reviews: list[AgentReview] = Field(default_factory=list)
    summary: str = ""
    human_reviewer: str | None = None


def claim_outcome(lines: list[LineDecision]) -> ClaimOutcome:
    outcomes = {line.outcome for line in lines}
    if LineOutcome.REVIEW in outcomes:
        return ClaimOutcome.PENDED
    if outcomes == {LineOutcome.DENY}:
        return ClaimOutcome.DENIED
    if outcomes == {LineOutcome.PAY}:
        return ClaimOutcome.APPROVED
    return ClaimOutcome.PARTIAL
