"""Execution-verified NL→SQL dataset over the synthetic claims warehouse.

Each template pairs several paraphrases with a gold SQL query; parameters are sampled
from the warehouse's real value domains. Every instance is executed and kept only if
the gold SQL runs. Splits hold out whole templates for `test`, so test accuracy measures
generalization to unseen query shapes rather than memorized ones.
"""

from __future__ import annotations

import itertools
import json
import random
import sqlite3
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from quill.db.generate import DENIAL_REASONS, SPECIALTIES, STATES

POOLS: dict[str, list[Any]] = {
    "status": ["paid", "denied", "pending", "adjusted"],
    "state": STATES,
    "specialty": SPECIALTIES,
    "year": ["2024", "2025"],
    "tier": ["Bronze", "Silver", "Gold", "Platinum"],
    "plan_type": ["HMO", "PPO", "EPO"],
    "claim_type": ["professional", "institutional", "pharmacy"],
    "n": [3, 5, 10],
    "proc_category": ["Imaging", "Lab", "Surgery", "Evaluation", "Behavioral", "Oncology"],
    "dx_category": ["Circulatory", "Endocrine", "Respiratory", "Musculoskeletal", "Mental health"],
    "pct": [10, 12, 15],
    "k": [12, 15, 18],
    "d": [15, 30, 60],
    "reason": DENIAL_REASONS,
}


@dataclass(frozen=True)
class Template:
    id: str
    difficulty: str
    questions: tuple[str, ...]
    sql: str


TEMPLATES: list[Template] = [
    Template(
        "count_by_status",
        "easy",
        (
            "How many {status} claims are there?",
            "Count the claims with status {status}.",
            "What's the number of claims that are {status}?",
        ),
        "SELECT COUNT(*) FROM claims WHERE status = '{status}'",
    ),
    Template(
        "members_in_state",
        "easy",
        (
            "How many members live in {state}?",
            "Number of enrollees residing in {state}",
            "Count members whose state is {state}.",
        ),
        "SELECT COUNT(*) FROM members WHERE state = '{state}'",
    ),
    Template(
        "providers_by_specialty_state",
        "easy",
        (
            "List the {specialty} providers in {state}.",
            "Which {specialty} practices are located in {state}?",
            "Show names and NPIs of {specialty} providers practicing in {state}.",
        ),
        "SELECT provider_name, npi FROM providers WHERE specialty = '{specialty}' AND state = '{state}' "
        "ORDER BY provider_name",
    ),
    Template(
        "total_paid_year",
        "easy",
        (
            "What was the total amount paid for services in {year}?",
            "Total plan spend on claims with a service date in {year}",
            "How much did the plan pay in total for {year} dates of service?",
        ),
        "SELECT SUM(paid_amount) FROM claims WHERE strftime('%Y', service_date) = '{year}'",
    ),
    Template(
        "avg_premium_tier",
        "easy",
        (
            "What is the average monthly premium for {tier} plans?",
            "Average premium of {tier} tier plans",
            "On average, how much do {tier} plans cost per month?",
        ),
        "SELECT AVG(monthly_premium) FROM plans WHERE metal_tier = '{tier}'",
    ),
    Template(
        "claim_type_submitted_year",
        "easy",
        (
            "How many {claim_type} claims were submitted in {year}?",
            "Count {claim_type} claims with a submission date in {year}.",
            "Number of {claim_type} claims filed during {year}",
        ),
        "SELECT COUNT(*) FROM claims WHERE claim_type = '{claim_type}' AND strftime('%Y', submitted_date) = '{year}'",
    ),
    Template(
        "denial_rate_by_network",
        "medium",
        (
            "What is the denial rate for in-network versus out-of-network providers?",
            "Compare the share of denied claims by provider network status.",
            "Denial rate split by network status",
        ),
        "SELECT p.network_status, AVG(CASE WHEN c.status = 'denied' THEN 1.0 ELSE 0 END) AS denial_rate "
        "FROM claims c JOIN providers p ON p.provider_id = c.provider_id GROUP BY p.network_status",
    ),
    Template(
        "top_denial_reasons",
        "medium",
        (
            "What are the top {n} denial reasons?",
            "List the {n} most common reasons claims get denied.",
            "Which {n} denial reasons occur most often?",
        ),
        "SELECT denial_reason, COUNT(*) AS n FROM claims WHERE status = 'denied' GROUP BY denial_reason "
        "ORDER BY n DESC LIMIT {n}",
    ),
    Template(
        "paid_by_specialty_year",
        "medium",
        (
            "Show total paid amount by provider specialty for {year}.",
            "How much was paid to each specialty in {year}, highest first?",
            "Break down {year} plan spend by provider specialty.",
        ),
        "SELECT p.specialty, SUM(c.paid_amount) AS total_paid FROM claims c JOIN providers p "
        "ON p.provider_id = c.provider_id WHERE strftime('%Y', c.service_date) = '{year}' "
        "GROUP BY p.specialty ORDER BY total_paid DESC",
    ),
    Template(
        "claims_by_dx_category",
        "medium",
        (
            "How many claims are there for each diagnosis category?",
            "Count claims per primary diagnosis category.",
            "Claim volume by diagnosis category, largest first",
        ),
        "SELECT d.category, COUNT(*) AS claims FROM claims c JOIN diagnoses d ON d.code = c.primary_diagnosis_code "
        "GROUP BY d.category ORDER BY claims DESC",
    ),
    Template(
        "avg_billed_plan_type",
        "medium",
        (
            "What is the average billed amount per claim for members on {plan_type} plans?",
            "Average claim billed amount for {plan_type} enrollees",
            "For members in {plan_type} plans, what's the mean billed amount per claim?",
        ),
        "SELECT AVG(c.billed_amount) FROM claims c JOIN members m ON m.member_id = c.member_id "
        "JOIN plans pl ON pl.plan_id = m.plan_id WHERE pl.plan_type = '{plan_type}'",
    ),
    Template(
        "monthly_claims_year",
        "medium",
        (
            "Show the number of claims per month in {year}.",
            "Monthly claim counts for {year} by service date",
            "How many claims were there each month of {year}?",
        ),
        "SELECT strftime('%Y-%m', service_date) AS month, COUNT(*) AS claims FROM claims "
        "WHERE strftime('%Y', service_date) = '{year}' GROUP BY month ORDER BY month",
    ),
    Template(
        "top_providers_paid",
        "medium",
        (
            "Which {n} providers were paid the most in total?",
            "Top {n} providers by total paid amount",
            "List the {n} providers with the highest total payments.",
        ),
        "SELECT p.provider_name, SUM(c.paid_amount) AS total_paid FROM claims c JOIN providers p "
        "ON p.provider_id = c.provider_id GROUP BY p.provider_id ORDER BY total_paid DESC LIMIT {n}",
    ),
    Template(
        "units_by_proc_category",
        "medium",
        (
            "How many units of {proc_category} procedures were billed?",
            "Total billed units for {proc_category} services",
            "Sum of units across all {proc_category} claim lines",
        ),
        "SELECT SUM(l.units) FROM claim_lines l JOIN procedures pr ON pr.code = l.procedure_code "
        "WHERE pr.category = '{proc_category}'",
    ),
    Template(
        "submit_lag_by_type",
        "medium",
        (
            "What is the average number of days between service and submission for each claim type?",
            "Average filing lag in days by claim type",
            "How long, on average, do providers take to submit each type of claim?",
        ),
        "SELECT claim_type, AVG(julianday(submitted_date) - julianday(service_date)) AS avg_days "
        "FROM claims GROUP BY claim_type",
    ),
    Template(
        "denial_rate_state_tier",
        "hard",
        (
            "What is the denial rate by member state for members on {tier} plans?",
            "For {tier} plan members, break down the denial rate by the state they live in.",
            "Denial rate per member state among {tier} tier enrollees",
        ),
        "SELECT m.state, AVG(CASE WHEN c.status = 'denied' THEN 1.0 ELSE 0 END) AS denial_rate FROM claims c "
        "JOIN members m ON m.member_id = c.member_id JOIN plans pl ON pl.plan_id = m.plan_id "
        "WHERE pl.metal_tier = '{tier}' GROUP BY m.state ORDER BY denial_rate DESC",
    ),
    Template(
        "specialties_high_denial",
        "hard",
        (
            "Which specialties have a denial rate above {pct}%?",
            "List provider specialties whose share of denied claims exceeds {pct} percent.",
            "Find specialties with more than {pct}% of claims denied.",
        ),
        "SELECT p.specialty, AVG(CASE WHEN c.status = 'denied' THEN 1.0 ELSE 0 END) AS denial_rate FROM claims c "
        "JOIN providers p ON p.provider_id = c.provider_id GROUP BY p.specialty "
        "HAVING denial_rate > {pct} / 100.0 ORDER BY denial_rate DESC",
    ),
    Template(
        "members_many_claims",
        "hard",
        (
            "How many members had more than {k} claims in {year}?",
            "Count members with over {k} claims during {year}.",
            "Number of members whose {year} claim count exceeds {k}",
        ),
        "SELECT COUNT(*) FROM (SELECT member_id FROM claims WHERE strftime('%Y', service_date) = '{year}' "
        "GROUP BY member_id HAVING COUNT(*) > {k})",
    ),
    Template(
        "paid_share_dx_category",
        "hard",
        (
            "For each diagnosis category, what share of total paid amount does it represent?",
            "What percentage of all plan payments goes to each diagnosis category?",
            "Share of total paid by diagnosis category",
        ),
        "SELECT d.category, SUM(c.paid_amount) / (SELECT SUM(paid_amount) FROM claims) AS share FROM claims c "
        "JOIN diagnoses d ON d.code = c.primary_diagnosis_code GROUP BY d.category ORDER BY share DESC",
    ),
    Template(
        "top_procs_out_of_network",
        "hard",
        (
            "Which {n} procedures have the highest total billed amount at out-of-network providers?",
            "Top {n} procedures by billed dollars from out-of-network providers",
            "List the {n} procedures out-of-network providers billed the most for.",
        ),
        "SELECT pr.description, SUM(l.line_billed) AS billed FROM claim_lines l "
        "JOIN claims c ON c.claim_id = l.claim_id JOIN providers p ON p.provider_id = c.provider_id "
        "JOIN procedures pr ON pr.code = l.procedure_code "
        "WHERE p.network_status = 'out_of_network' GROUP BY pr.code ORDER BY billed DESC LIMIT {n}",
    ),
    Template(
        "yoy_paid_plan_type",
        "hard",
        (
            "How did total paid change from 2024 to 2025 for each plan type?",
            "Compare 2024 and 2025 total payments by plan type.",
            "Year-over-year total paid (2024 vs 2025) per plan type",
        ),
        "SELECT pl.plan_type, SUM(CASE WHEN strftime('%Y', c.service_date) = '2024' THEN c.paid_amount ELSE 0 END) "
        "AS paid_2024, SUM(CASE WHEN strftime('%Y', c.service_date) = '2025' THEN c.paid_amount ELSE 0 END) "
        "AS paid_2025 "
        "FROM claims c JOIN members m ON m.member_id = c.member_id JOIN plans pl ON pl.plan_id = m.plan_id "
        "GROUP BY pl.plan_type",
    ),
    Template(
        "paid_ratio_type_network",
        "hard",
        (
            "What is the paid-to-billed ratio for each claim type and network status?",
            "Reimbursement rate by claim type and provider network status",
            "For every claim type and network status combination, compute paid over billed.",
        ),
        "SELECT c.claim_type, p.network_status, SUM(c.paid_amount) / SUM(c.billed_amount) AS ratio FROM claims c "
        "JOIN providers p ON p.provider_id = c.provider_id GROUP BY c.claim_type, p.network_status",
    ),
    Template(
        "stale_pending",
        "hard",
        (
            "How many claims were still pending more than {d} days after submission as of 2026-01-01?",
            "As of January 1 2026, count pending claims submitted over {d} days earlier.",
            "Number of pending claims older than {d} days on 2026-01-01",
        ),
        "SELECT COUNT(*) FROM claims WHERE status = 'pending' "
        "AND julianday('2026-01-01') - julianday(submitted_date) > {d}",
    ),
    Template(
        "top_dx_paid_state",
        "hard",
        (
            "Which diagnosis has the highest average paid amount for members in {state}?",
            "For {state} members, what diagnosis costs the plan the most on average per claim?",
            "Top diagnosis by average paid per claim among members living in {state}",
        ),
        "SELECT d.description, AVG(c.paid_amount) AS avg_paid FROM claims c "
        "JOIN members m ON m.member_id = c.member_id JOIN diagnoses d ON d.code = c.primary_diagnosis_code "
        "WHERE m.state = '{state}' GROUP BY d.code "
        "ORDER BY avg_paid DESC LIMIT 1",
    ),
]

HELD_OUT_TEMPLATES = frozenset(
    {"avg_premium_tier", "avg_billed_plan_type", "specialties_high_denial", "paid_ratio_type_network"}
)


@dataclass(frozen=True)
class Example:
    id: str
    template: str
    difficulty: str
    question: str
    sql: str
    split: str


def _fields(template: Template) -> list[str]:
    import string

    names = {f for text in (*template.questions, template.sql) for _, f, _, _ in string.Formatter().parse(text) if f}
    return sorted(names)


def _bindings(template: Template, rng: random.Random, limit: int) -> Iterator[dict[str, Any]]:
    fields = _fields(template)
    combos = list(itertools.product(*(POOLS[f] for f in fields)))
    rng.shuffle(combos)
    for combo in combos[:limit]:
        yield dict(zip(fields, combo, strict=True))


def build_examples(db_path: Path, *, per_template: int = 30, seed: int = 13) -> list[Example]:
    rng = random.Random(seed)
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    examples: list[Example] = []
    for template in TEMPLATES:
        for b_idx, binding in enumerate(_bindings(template, rng, per_template)):
            sql = template.sql.format(**binding)
            try:
                conn.execute(sql).fetchall()
            except sqlite3.Error as exc:  # a broken gold query is a bug, not data
                raise RuntimeError(f"gold SQL failed for {template.id}: {exc}\n{sql}") from exc
            if template.id in HELD_OUT_TEMPLATES:
                split = "test"
            else:
                split = rng.choices(["train", "dev", "test"], [80, 10, 10])[0]
            for q_idx, question in enumerate(template.questions):
                examples.append(
                    Example(
                        id=f"{template.id}-{b_idx}-{q_idx}",
                        template=template.id,
                        difficulty=template.difficulty,
                        question=question.format(**binding),
                        sql=sql,
                        split=split,
                    )
                )
    conn.close()
    return examples


def write_splits(examples: list[Example], out_dir: Path) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for split in ("train", "dev", "test"):
        rows = [e for e in examples if e.split == split]
        counts[split] = len(rows)
        with (out_dir / f"{split}.jsonl").open("w") as fh:
            for e in rows:
                fh.write(json.dumps(asdict(e)) + "\n")
    return counts


def load_split(path: Path) -> list[Example]:
    return [Example(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()]
