"""Benchmark: rules-only vs rules + agents vs LLM-only on the labeled claim set.

Metrics:
  * line accuracy and claim accuracy on automatically decided claims
  * straight-through processing (STP): share of claims decided without a human
  * judgment accuracy: claims whose correct outcome depends on reading the clinical note
  * overpayment / underpayment: billed dollars on lines paid that should be denied (and vice versa)
  * rule precision/recall against the injected anomalies
  * audit-ledger integrity
"""

from __future__ import annotations

import json
import statistics
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from aegis.adjudicate import adjudicate
from aegis.config import Settings
from aegis.llm.base import LLM
from aegis.models import Claim, LineOutcome
from aegis.service import AuditService


@dataclass
class Item:
    scenario: str
    expected: dict[int, str]
    rules: set[str]
    judgment: bool
    claim: Claim


@dataclass
class SystemScore:
    name: str
    claims: int = 0
    decided: int = 0
    correct_claims: int = 0
    correct_lines: int = 0
    decided_lines: int = 0
    judgment_total: int = 0
    judgment_correct: int = 0
    judgment_escalated: int = 0
    overpaid: float = 0.0
    underpaid: float = 0.0
    latencies: list[float] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def row(self) -> dict[str, str]:
        pct = lambda a, b: f"{100 * a / b:.1f}%" if b else "-"  # noqa: E731
        lat = sorted(self.latencies)
        return {
            "STP (no human)": pct(self.decided, self.claims),
            "claim accuracy (auto-decided)": pct(self.correct_claims, self.decided),
            "line accuracy (auto-decided)": pct(self.correct_lines, self.decided_lines),
            "judgment accuracy (auto-decided)": pct(
                self.judgment_correct, self.judgment_total - self.judgment_escalated
            ),
            "judgment escalated to human": pct(self.judgment_escalated, self.judgment_total),
            "overpaid $ (should deny, paid)": f"${self.overpaid:,.0f}",
            "underpaid $ (should pay, denied)": f"${self.underpaid:,.0f}",
            "latency p50": f"{statistics.median(lat) * 1000:.0f} ms" if lat else "-",
        }


def load_items(path: Path) -> list[Item]:
    items = []
    for line in path.read_text().splitlines():
        r = json.loads(line)
        items.append(
            Item(
                r["scenario"],
                {int(k): v for k, v in r["expected_lines"].items()},
                set(r["expected_rules"]),
                r["needs_judgment"],
                Claim.model_validate(r["claim"]),
            )
        )
    return items


def score(sc: SystemScore, item: Item, outcomes: dict[int, str] | None) -> None:
    sc.claims += 1
    if item.judgment:
        sc.judgment_total += 1
    if outcomes is None or LineOutcome.REVIEW.value in outcomes.values():
        if item.judgment:
            sc.judgment_escalated += 1
        return
    sc.decided += 1
    charges = {ln.line_no: ln.charge for ln in item.claim.lines}
    all_ok = True
    for line_no, expected in item.expected.items():
        got = outcomes.get(line_no)
        sc.decided_lines += 1
        if got == expected:
            sc.correct_lines += 1
            continue
        all_ok = False
        if expected == "deny" and got in ("pay", "adjust"):
            sc.overpaid += charges[line_no]
        elif expected in ("pay", "adjust") and got == "deny":
            sc.underpaid += charges[line_no]
    sc.correct_claims += all_ok
    if item.judgment and all_ok:
        sc.judgment_correct += 1


class LLMOnlyLine(BaseModel):
    line_no: int
    outcome: str  # pay | adjust | deny
    reason: str = ""


class LLMOnlyDecision(BaseModel):
    lines: list[LLMOnlyLine]


async def llm_only(llm: LLM, service: AuditService, claim: Claim) -> dict[int, str]:
    ref = service.reference
    member, provider = ref.members.get(claim.member_id), ref.providers.get(claim.provider_npi)
    auth = ref.find_auth(claim.prior_auth_number, claim.member_id, claim.lines[0].cpt, claim.service_date)
    facts = {
        "member": member.model_dump(mode="json") if member else "not found",
        "member_age": member.age_on(claim.service_date) if member else None,
        "provider": provider.model_dump(mode="json") if provider else "not enrolled",
        "valid_authorization_for_first_line": bool(auth),
        "duplicates": [
            h.model_dump(mode="json")
            for ln in claim.lines
            for h in ref.duplicates(claim.member_id, claim.provider_npi, claim.service_date, ln.cpt, claim.claim_id)
        ],
        "codes": {ln.cpt: ref.cpt[ln.cpt].model_dump() for ln in claim.lines if ln.cpt in ref.cpt},
        "ncci_edits": [e.model_dump() for e in ref.ncci],
    }
    manual = service.settings.policy_manual.read_text()
    messages = [
        {
            "role": "system",
            "content": "You adjudicate health insurance claims strictly according to the policy "
            "manual. Decide each line: pay, adjust (reduce units) or deny.",
        },
        {
            "role": "user",
            "content": f"<policy_manual>\n{manual}\n</policy_manual>\n<facts>\n{json.dumps(facts)}\n"
            f"</facts>\n<claim>\n{claim.model_dump_json()}\n</claim>",
        },
    ]
    decision = await llm.complete_json(messages, LLMOnlyDecision)  # type: ignore[arg-type]
    return {ln.line_no: ln.outcome for ln in decision.lines}


async def run_eval(settings: Settings, data: Path, out: Path, *, include_llm_only: bool = True) -> dict[str, Any]:
    items = load_items(data)
    work = Path(tempfile.mkdtemp(prefix="aegis-eval-"))
    service = await AuditService.create(settings.model_copy(update={"state_dir": work}), durable=False)
    rules_only, agents, llm_base = SystemScore("rules only"), SystemScore("rules + agents"), SystemScore("LLM only")
    rule_tp = rule_fp = rule_fn = 0
    try:
        for item in items:
            findings = service.engine.evaluate(item.claim)
            fired = {f.rule_id for f in findings}
            rule_tp += len(fired & item.rules)
            rule_fp += len(fired - item.rules)
            rule_fn += len(item.rules - fired)
            base = adjudicate(item.claim, findings, [])
            score(rules_only, item, {ln.line_no: ln.outcome.value for ln in base})

            started = time.perf_counter()
            result = await service.audit(item.claim)
            agents.latencies.append(time.perf_counter() - started)
            score(
                agents,
                item,
                None if result.decision is None else {ln.line_no: ln.outcome.value for ln in result.decision.lines},
            )

            if include_llm_only and not settings.offline:
                started = time.perf_counter()
                try:
                    outcomes = await llm_only(service.llms["clinical"], service, item.claim)
                except Exception as exc:
                    llm_base.errors.append(str(exc)[:200])
                    outcomes = None
                llm_base.latencies.append(time.perf_counter() - started)
                score(llm_base, item, outcomes)
        ledger_ok, _ = service.ledger.verify()
    finally:
        await service.close()

    systems = [rules_only, agents] + ([llm_base] if llm_base.claims else [])
    rows = {s.name: s.row() for s in systems}
    metrics = list(next(iter(rows.values())))
    model = service.llms["clinical"].name
    lines = [
        "# Aegis evaluation report",
        "",
        f"- **provider**: `{settings.provider}`  **reviewer model**: `{model}`",
        f"- **claims**: {len(items)} ({sum(i.judgment for i in items)} need clinical/coding judgment)",
        f"- **rules engine vs injected anomalies**: precision {rule_tp / max(1, rule_tp + rule_fp):.3f}, "
        f"recall {rule_tp / max(1, rule_tp + rule_fn):.3f}",
        f"- **audit ledger hash chain**: {'verified' if ledger_ok else 'BROKEN'}",
        "",
        "| metric | " + " | ".join(rows) + " |",
        "|---|" + "---|" * len(rows),
    ]
    lines += [f"| {m} | " + " | ".join(rows[s][m] for s in rows) + " |" for m in metrics]
    if not llm_base.claims:
        lines += ["", "_LLM-only baseline requires a model provider (skipped in offline mode)._"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    summary = {name: row for name, row in rows.items()}
    print("\n".join(lines))
    return summary
