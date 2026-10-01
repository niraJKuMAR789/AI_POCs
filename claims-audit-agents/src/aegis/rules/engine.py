"""Deterministic payer rules.

Each rule is a small pure function over (claim, reference data) returning findings. HARD findings
have a fixed outcome that no agent can change; SOFT findings depend on clinical documentation and are
routed to a reviewer agent ("clinical" or "coding").
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from aegis.models import Claim, Finding, Severity
from aegis.npi import is_valid_npi
from aegis.reference import Reference

RuleFn = Callable[[Claim, Reference], list[Finding]]


@dataclass(frozen=True)
class Rule:
    rule_id: str
    title: str
    fn: RuleFn


RULES: list[Rule] = []


def rule(rule_id: str, title: str) -> Callable[[RuleFn], RuleFn]:
    def register(fn: RuleFn) -> RuleFn:
        RULES.append(Rule(rule_id, title, fn))
        return fn

    return register


def _finding(
    rule_id: str,
    severity: Severity,
    message: str,
    section: str,
    line_no: int | None = None,
    reviewer: str | None = None,
    adjusted_units: int | None = None,
) -> Finding:
    title = next((r.title for r in RULES if r.rule_id == rule_id), rule_id)
    return Finding(
        rule_id=rule_id,
        title=title,
        severity=severity,
        line_no=line_no,
        message=message,
        policy_section=section,
        reviewer=reviewer,
        adjusted_units=adjusted_units,
    )


def _is_emergency(claim: Claim, ref: Reference) -> bool:
    return claim.place_of_service in ref.emergency_pos


@rule("R001", "Member eligibility")
def eligibility(claim: Claim, ref: Reference) -> list[Finding]:
    member = ref.members.get(claim.member_id)
    if member is None:
        return [_finding("R001", Severity.HARD, f"Member {claim.member_id} not found.", "AEG-1.2")]
    if not member.active_on(claim.service_date):
        return [
            _finding(
                "R001",
                Severity.HARD,
                f"Coverage not active on {claim.service_date} "
                f"(coverage {member.coverage_start} to {member.coverage_end or 'open'}).",
                "AEG-1.2",
            )
        ]
    return []


@rule("R002", "Timely filing")
def timely_filing(claim: Claim, ref: Reference) -> list[Finding]:
    days = (claim.submitted_date - claim.service_date).days
    if days > ref.timely_filing_days:
        return [
            _finding(
                "R002",
                Severity.HARD,
                f"Submitted {days} days after service; limit is {ref.timely_filing_days}.",
                "AEG-1.1",
            )
        ]
    return []


@rule("R003", "Duplicate claim line")
def duplicates(claim: Claim, ref: Reference) -> list[Finding]:
    out = []
    for line in claim.lines:
        dups = ref.duplicates(claim.member_id, claim.provider_npi, claim.service_date, line.cpt, claim.claim_id)
        if dups:
            out.append(
                _finding(
                    "R003",
                    Severity.HARD,
                    f"{line.cpt} duplicates line on claim {dups[0].claim_id}.",
                    "AEG-1.3",
                    line.line_no,
                )
            )
    return out


@rule("R004", "Prior authorization")
def prior_auth(claim: Claim, ref: Reference) -> list[Finding]:
    out = []
    for line in claim.lines:
        info = ref.cpt.get(line.cpt)
        if not info or not info.requires_auth:
            continue
        if ref.find_auth(claim.prior_auth_number, claim.member_id, line.cpt, claim.service_date):
            continue
        if _is_emergency(claim, ref):
            out.append(
                _finding(
                    "R004",
                    Severity.SOFT,
                    f"{line.cpt} performed in the ER without prior authorization; payable if an "
                    "emergency condition is documented.",
                    "AEG-3.2",
                    line.line_no,
                    reviewer="clinical",
                )
            )
        else:
            out.append(
                _finding(
                    "R004",
                    Severity.HARD,
                    f"{line.cpt} requires prior authorization; none valid for this member, code and date.",
                    "AEG-3.1",
                    line.line_no,
                )
            )
    return out


@rule("R005", "Medical necessity (diagnosis support)")
def medical_necessity(claim: Claim, ref: Reference) -> list[Finding]:
    out = []
    for line in claim.lines:
        info = ref.cpt.get(line.cpt)
        if not info or not info.necessity:
            continue
        if any(dx.startswith(prefix) for dx in claim.diagnosis_codes for prefix in info.necessity):
            continue
        out.append(
            _finding(
                "R005",
                Severity.SOFT,
                f"No diagnosis on the claim ({', '.join(claim.diagnosis_codes)}) supports {line.cpt}; "
                "payable only if documentation meets procedure-specific criteria.",
                info.policy or "AEG-4.1",
                line.line_no,
                reviewer="clinical",
            )
        )
    return out


@rule("R006", "Procedure-to-procedure (bundling) edit")
def bundling(claim: Claim, ref: Reference) -> list[Finding]:
    out = []
    codes = {line.cpt for line in claim.lines}
    for edit in ref.ncci:
        if edit.column1 not in codes:
            continue
        for line in claim.lines:
            if line.cpt != edit.column2:
                continue
            has_modifier = bool({"59", "XS"} & set(line.modifiers))
            if edit.modifier_allowed and has_modifier:
                out.append(
                    _finding(
                        "R006",
                        Severity.SOFT,
                        f"{line.cpt} billed with {edit.column1} using modifier 59/XS; payable only if a "
                        "distinct procedural service is documented.",
                        "AEG-5.1",
                        line.line_no,
                        reviewer="coding",
                    )
                )
            else:
                reason = "edit does not allow a modifier" if not edit.modifier_allowed else "no 59/XS modifier"
                out.append(
                    _finding(
                        "R006",
                        Severity.HARD,
                        f"{line.cpt} is a component of {edit.column1} at the same encounter ({reason}).",
                        "AEG-5.1",
                        line.line_no,
                    )
                )
    return out


@rule("R007", "Medically unlikely units")
def units(claim: Claim, ref: Reference) -> list[Finding]:
    out = []
    for line in claim.lines:
        info = ref.cpt.get(line.cpt)
        if info and line.units > info.mue:
            out.append(
                _finding(
                    "R007",
                    Severity.HARD,
                    f"{line.units} units of {line.cpt} exceed the daily maximum of {info.mue}.",
                    "AEG-5.2",
                    line.line_no,
                    adjusted_units=info.mue,
                )
            )
    return out


@rule("R008", "Age-specific service")
def age_limits(claim: Claim, ref: Reference) -> list[Finding]:
    member = ref.members.get(claim.member_id)
    if member is None:
        return []
    age = member.age_on(claim.service_date)
    out = []
    for line in claim.lines:
        info = ref.cpt.get(line.cpt)
        if not info:
            continue
        if (info.age_min is not None and age < info.age_min) or (info.age_max is not None and age > info.age_max):
            out.append(
                _finding(
                    "R008",
                    Severity.HARD,
                    f"{line.cpt} is defined for ages {info.age_min or 0}-{info.age_max or 'any'}; member is {age}.",
                    "AEG-6.1",
                    line.line_no,
                )
            )
    return out


@rule("R009", "HMO out-of-network without referral")
def hmo_network(claim: Claim, ref: Reference) -> list[Finding]:
    member, provider = ref.members.get(claim.member_id), ref.providers.get(claim.provider_npi)
    if not member or not provider or member.plan_type != "HMO" or provider.network == "in_network":
        return []
    if _is_emergency(claim, ref) or ref.has_referral(claim.referral_number, claim.member_id, claim.provider_npi):
        return []
    return [
        _finding(
            "R009",
            Severity.HARD,
            f"HMO member seen by out-of-network provider {provider.name} without a referral.",
            "AEG-2.1",
        )
    ]


@rule("R010", "Charge outlier")
def charge_outlier(claim: Claim, ref: Reference) -> list[Finding]:
    out = []
    for line in claim.lines:
        info = ref.cpt.get(line.cpt)
        if info and line.charge / line.units > ref.outlier_multiple * info.ref_cost:
            out.append(
                _finding(
                    "R010",
                    Severity.SOFT,
                    f"{line.cpt} billed at ${line.charge / line.units:,.2f}/unit, more than "
                    f"{ref.outlier_multiple:g}x the ${info.ref_cost:,.2f} reference.",
                    "AEG-5.4",
                    line.line_no,
                    reviewer="coding",
                )
            )
    return out


@rule("R011", "High-level E/M documentation")
def em_level(claim: Claim, ref: Reference) -> list[Finding]:
    out = []
    for line in claim.lines:
        info = ref.cpt.get(line.cpt)
        if info and info.review_level:
            out.append(
                _finding(
                    "R011",
                    Severity.SOFT,
                    f"{line.cpt} requires high-complexity medical decision making.",
                    "AEG-5.3",
                    line.line_no,
                    reviewer="coding",
                )
            )
    return out


@rule("R012", "Provider NPI")
def provider_npi(claim: Claim, ref: Reference) -> list[Finding]:
    if not is_valid_npi(claim.provider_npi):
        return [_finding("R012", Severity.HARD, f"NPI {claim.provider_npi} fails check-digit validation.", "AEG-1.4")]
    if claim.provider_npi not in ref.providers:
        return [_finding("R012", Severity.HARD, f"NPI {claim.provider_npi} is not an enrolled provider.", "AEG-1.4")]
    return []


class RulesEngine:
    def __init__(self, reference: Reference, rules: list[Rule] | None = None) -> None:
        self.reference = reference
        self.rules = rules if rules is not None else RULES

    def evaluate(self, claim: Claim) -> list[Finding]:
        findings: list[Finding] = []
        for r in self.rules:
            findings.extend(r.fn(claim, self.reference))
        return findings
