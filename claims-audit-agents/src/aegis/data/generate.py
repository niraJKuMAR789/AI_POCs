"""Synthetic registry + labeled claims with injected, ground-truth anomalies.

Every claim comes from a scenario with known expected line outcomes, so the audit can be scored
exactly. Scenarios needing judgment have several clinical-note phrasings, including "hard negatives"
that mention a policy criterion only to negate it ("no focal neurological deficit").
All people, providers and clinical notes are fictional.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from aegis.npi import make_npi

SPECIALTIES = [
    "Family Medicine",
    "Neurology",
    "Orthopedics",
    "Cardiology",
    "Physical Therapy",
    "Oncology",
    "Endocrinology",
    "Emergency Medicine",
]
LAST = ["Garcia", "Nguyen", "Patel", "Kim", "Okafor", "Silva", "Cohen", "Haddad", "Reyes", "Tanaka"]


@dataclass
class Registry:
    members: list[dict[str, Any]] = field(default_factory=list)
    providers: list[dict[str, Any]] = field(default_factory=list)
    authorizations: list[dict[str, Any]] = field(default_factory=list)
    referrals: list[dict[str, Any]] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Labeled:
    claim: dict[str, Any]
    scenario: str
    expected_lines: dict[int, str]  # line_no -> pay | adjust | deny
    expected_rules: list[str]
    needs_judgment: bool


# --- clinical note pools: (text, supports_policy) ---------------------------------------------
NOTES: dict[str, list[tuple[str, bool]]] = {
    "mri_brain": [
        (
            "Patient reports sudden onset severe headache that began during exercise, described as the worst "
            "headache of her life. Exam shows right-sided weakness. MRI ordered to evaluate.",
            True,
        ),
        (
            "Two weeks of progressively worsening headache. Fundoscopic exam shows papilledema. Neurology "
            "recommends MRI brain.",
            True,
        ),
        (
            "New headache in a patient undergoing chemotherapy for breast cancer. Focal neurological deficit noted "
            "with left arm drift.",
            True,
        ),
        (
            "Chronic tension-type headaches for 3 years, unchanged in character. Neurological examination is normal "
            "with no focal deficit. Patient requests imaging for reassurance.",
            False,
        ),
        (
            "Intermittent headaches over several months. Denies sudden onset. No focal neurological deficit and no "
            "papilledema on exam. MRI ordered per patient preference.",
            False,
        ),
        ("Stable migraine pattern controlled with triptans. Normal neurological examination.", False),
    ],
    "mri_lumbar": [
        (
            "Low back pain for 9 weeks. Completed 6 weeks of physical therapy and NSAIDs without improvement. "
            "Progressive motor weakness of left ankle dorsiflexion noted.",
            True,
        ),
        (
            "Acute low back pain with new urinary retention and saddle anesthesia, concerning for cauda equina "
            "syndrome.",
            True,
        ),
        ("Back pain in a patient with a history of prostate cancer; rule out metastatic disease.", True),
        (
            "Three weeks of low back pain. Has not completed six weeks of conservative therapy. No progressive motor "
            "weakness, no bowel or bladder dysfunction, no history of cancer.",
            False,
        ),
        (
            "Low back pain for 10 days after lifting boxes. No therapy tried yet. Denies bowel or bladder dysfunction. "
            "No weakness.",
            False,
        ),
        (
            "Two weeks of mechanical low back pain. Patient has not started physical therapy. Strength 5/5 "
            "throughout, no red flags.",
            False,
        ),
    ],
    "er_auth": [
        (
            "Brought to the emergency department after a motor vehicle collision with head trauma and acute "
            "neurological change (new confusion). Emergent MRI obtained.",
            True,
        ),
        (
            "Sudden severe symptoms: acute onset aphasia and right facial droop 40 minutes before arrival. Stroke "
            "protocol imaging performed emergently.",
            True,
        ),
        (
            "Patient presented to the ER requesting an MRI for chronic knee pain because the outpatient wait was "
            "long. No acute findings.",
            False,
        ),
        (
            "Walked into the ER for a scheduled-type study. No trauma, no sudden severe symptoms and no acute "
            "neurological change; neuro exam at baseline.",
            False,
        ),
    ],
    "modifier_distinct": [
        (
            "Arthroscopic partial medial meniscectomy, right knee. In a separate procedure, corticosteroid injection "
            "of the contralateral left knee for osteoarthritis.",
            True,
        ),
        (
            "Right knee arthroscopy with meniscectomy. Separately, injection performed into the left knee, a different "
            "anatomic site, for symptomatic osteoarthritis.",
            True,
        ),
        (
            "Right knee arthroscopy with meniscectomy. At the end of the case the same right knee was injected with "
            "bupivacaine for postoperative pain.",
            False,
        ),
        (
            "Right knee arthroscopy with meniscectomy and injection of the operative knee. The contralateral knee was "
            "examined and is normal; no separate session or different anatomic site.",
            False,
        ),
    ],
    "em_high": [
        (
            "Severe exacerbation of asthma with oxygen saturation 89%. Decision regarding hospitalization discussed "
            "with patient; admitted for observation.",
            True,
        ),
        (
            "Patient on amiodarone and warfarin requiring drug therapy with intensive monitoring for toxicity; "
            "INR 5.2, dose adjusted.",
            True,
        ),
        ("Routine follow-up of stable hypertension. Blood pressure controlled. Refilled medications.", False),
        ("Annual medication review; diabetes well controlled, A1c 6.8. No changes made.", False),
        (
            "Mild asthma symptoms, no severe exacerbation. No decision regarding hospitalization was needed; "
            "continue current inhaler.",
            False,
        ),
    ],
    "outlier": [
        ("Bilateral procedure performed; both knees injected under ultrasound guidance.", True),
        ("Unusually prolonged service of 95 minutes due to significant complications from vasovagal syncope.", True),
        ("Standard single-joint injection without complications.", False),
        ("Not a bilateral procedure: single right knee injection, routine duration, no complications.", False),
    ],
}


def _npi(rng: random.Random) -> str:
    return make_npi(str(rng.choice([1, 2])) + "".join(str(rng.randint(0, 9)) for _ in range(8)))


def build(n_per_scenario: int = 6, seed: int = 11) -> tuple[Registry, list[Labeled]]:
    rng = random.Random(seed)
    reg = Registry()
    for i in range(1, 121):
        dob = date(1945, 1, 1) + timedelta(days=rng.randint(0, 365 * 70))
        terminated = date(2025, 6, 30) if i % 15 == 0 else None
        reg.members.append(
            {
                "member_id": f"M{i:05d}",
                "date_of_birth": dob.isoformat(),
                "gender": rng.choice("FM"),
                "plan_type": "HMO" if i % 3 == 0 else "PPO",
                "coverage_start": "2024-01-01",
                "coverage_end": terminated and terminated.isoformat(),
            }
        )
    for i in range(30):
        specialty = SPECIALTIES[i % len(SPECIALTIES)]
        reg.providers.append(
            {
                "npi": _npi(rng),
                "name": f"{rng.choice(LAST)} {specialty}",
                "specialty": specialty,
                "network": "out_of_network" if i % 6 == 5 else "in_network",
            }
        )

    active = [m for m in reg.members if not m["coverage_end"]]
    by_specialty = {
        s: [p for p in reg.providers if p["specialty"] == s and p["network"] == "in_network"] for s in SPECIALTIES
    }
    oon = [p for p in reg.providers if p["network"] == "out_of_network"]
    labeled: list[Labeled] = []
    counter = iter(range(1, 100_000))

    def age(member: dict[str, Any], day: date) -> int:
        dob = date.fromisoformat(member["date_of_birth"])
        return day.year - dob.year - ((day.month, day.day) < (dob.month, dob.day))

    def claim(
        member: dict[str, Any],
        provider: dict[str, Any],
        dx: list[str],
        lines: list[tuple[str, int, float, list[str]]],
        *,
        pos: str = "11",
        lag: int = 10,
        auth: str | None = None,
        referral: str | None = None,
        note: str | None = None,
        day: date | None = None,
    ) -> dict[str, Any]:
        service = day or date(2025, 3, 1) + timedelta(days=rng.randint(0, 90))
        return {
            "claim_id": f"C{next(counter):06d}",
            "member_id": member["member_id"],
            "provider_npi": provider["npi"],
            "service_date": service.isoformat(),
            "submitted_date": (service + timedelta(days=lag)).isoformat(),
            "place_of_service": pos,
            "diagnosis_codes": dx,
            "lines": [
                {"line_no": i + 1, "cpt": c, "units": u, "charge": ch, "modifiers": m}
                for i, (c, u, ch, m) in enumerate(lines)
            ],
            "prior_auth_number": auth,
            "referral_number": referral,
            "clinical_note": note,
        }

    def add_auth(member: dict[str, Any], cpt: str, c: dict[str, Any]) -> str:
        number = f"PA{len(reg.authorizations) + 1:06d}"
        day = date.fromisoformat(c["service_date"])
        reg.authorizations.append(
            {
                "auth_number": number,
                "member_id": member["member_id"],
                "cpt": cpt,
                "valid_from": (day - timedelta(days=10)).isoformat(),
                "valid_to": (day + timedelta(days=20)).isoformat(),
            }
        )
        return number

    def ppo_member() -> dict[str, Any]:
        return rng.choice([m for m in active if m["plan_type"] == "PPO" and 18 <= age(m, date(2025, 4, 1)) < 64])

    def emit(c: dict[str, Any], scenario: str, expected: dict[int, str], rules: list[str], judgment: bool) -> None:
        labeled.append(Labeled(c, scenario, expected, rules, judgment))

    for _ in range(n_per_scenario):
        fm = rng.choice(by_specialty["Family Medicine"])
        # Clean claims: no findings.
        m = ppo_member()
        emit(
            claim(m, fm, ["E11.9"], [("99213", 1, 140.0, []), ("83036", 1, 38.0, []), ("36415", 1, 12.0, [])]),
            "clean_diabetes_visit",
            {1: "pay", 2: "pay", 3: "pay"},
            [],
            False,
        )
        emit(
            claim(
                ppo_member(),
                rng.choice(by_specialty["Cardiology"]),
                ["I10"],
                [("99214", 1, 210.0, []), ("93000", 1, 75.0, [])],
            ),
            "clean_cardiology",
            {1: "pay", 2: "pay"},
            [],
            False,
        )
        # Hard administrative failures.
        termed = rng.choice([x for x in reg.members if x["coverage_end"]])
        emit(
            claim(termed, fm, ["J06.9"], [("99213", 1, 130.0, [])], day=date(2025, 9, 15)),
            "ineligible_member",
            {1: "deny"},
            ["R001"],
            False,
        )
        emit(
            claim(ppo_member(), fm, ["I10"], [("99213", 1, 130.0, [])], lag=rng.randint(95, 200)),
            "timely_filing",
            {1: "deny"},
            ["R002"],
            False,
        )
        m = ppo_member()
        first = claim(m, fm, ["I10"], [("99214", 1, 190.0, [])])
        reg.history.append(
            {
                "member_id": m["member_id"],
                "provider_npi": fm["npi"],
                "service_date": first["service_date"],
                "cpt": "99214",
                "claim_id": f"H{len(reg.history) + 1:06d}",
            }
        )
        emit(first, "duplicate", {1: "deny"}, ["R003"], False)
        bad = dict(fm)
        bad["npi"] = fm["npi"][:9] + str((int(fm["npi"][9]) + 1) % 10)
        emit(claim(ppo_member(), bad, ["I10"], [("99213", 1, 130.0, [])]), "invalid_npi", {1: "deny"}, ["R012"], False)
        hmo = rng.choice([x for x in active if x["plan_type"] == "HMO"])
        emit(
            claim(hmo, rng.choice(oon), ["M25.561"], [("99213", 1, 150.0, [])]),
            "hmo_out_of_network",
            {1: "deny"},
            ["R009"],
            False,
        )
        # Line-level hard edits.
        emit(
            claim(
                ppo_member(),
                rng.choice(by_specialty["Orthopedics"]),
                ["M54.50"],
                [("99213", 1, 120.0, []), ("72148", 1, 1500.0, [])],
            ),
            "missing_prior_auth",
            {1: "pay", 2: "deny"},
            ["R004", "R005"],
            False,
        )
        emit(
            claim(
                ppo_member(),
                rng.choice(by_specialty["Cardiology"]),
                ["R07.9"],
                [("93000", 1, 70.0, []), ("93005", 1, 30.0, [])],
            ),
            "unbundled_ecg",
            {1: "pay", 2: "deny"},
            ["R006"],
            False,
        )
        emit(
            claim(
                ppo_member(),
                rng.choice(by_specialty["Physical Therapy"]),
                ["M54.50"],
                [("97110", 6, 210.0, []), ("97140", 2, 66.0, [])],
            ),
            "excess_units",
            {1: "adjust", 2: "pay"},
            ["R007"],
            False,
        )
        young = rng.choice([x for x in active if age(x, date(2025, 4, 1)) < 60])
        emit(claim(young, fm, ["Z00.00"], [("G0439", 1, 175.0, [])]), "age_limit", {1: "deny"}, ["R008"], False)

        # Judgment scenarios: soft findings whose correct outcome depends on the note.
        def judged(
            scenario: str, pool: str, build_claim: Any, line_no: int, rules: list[str], others: dict[int, str]
        ) -> None:
            text, supports = rng.choice(NOTES[pool])
            c = build_claim(text)
            emit(
                c,
                f"{scenario}_{'supported' if supports else 'unsupported'}",
                {**others, line_no: "pay" if supports else "deny"},
                rules,
                True,
            )

        def mri_brain(note: str) -> dict[str, Any]:
            m = ppo_member()
            c = claim(
                m,
                rng.choice(by_specialty["Neurology"]),
                ["R51.9"],
                [("99214", 1, 180.0, []), ("70553", 1, 1600.0, [])],
                note=note,
            )
            c["prior_auth_number"] = add_auth(m, "70553", c)
            return c

        def mri_lumbar(note: str) -> dict[str, Any]:
            m = ppo_member()
            c = claim(m, rng.choice(by_specialty["Orthopedics"]), ["M54.50"], [("72148", 1, 1100.0, [])], note=note)
            c["prior_auth_number"] = add_auth(m, "72148", c)
            return c

        def er_mri(note: str) -> dict[str, Any]:
            return claim(
                ppo_member(),
                rng.choice(by_specialty["Emergency Medicine"]),
                ["G45.9"],
                [("70553", 1, 1700.0, [])],
                pos="23",
                note=note,
            )

        def arthro(note: str) -> dict[str, Any]:
            m = ppo_member()
            c = claim(
                m,
                rng.choice(by_specialty["Orthopedics"]),
                ["S83.241A", "M17.12"],
                [("29881", 1, 5200.0, []), ("20610", 1, 140.0, ["59"])],
                pos="22",
                note=note,
            )
            c["prior_auth_number"] = add_auth(m, "29881", c)
            return c

        def em_high(note: str, provider: dict[str, Any] = fm) -> dict[str, Any]:
            return claim(ppo_member(), provider, ["J45.41"], [("99215", 1, 260.0, [])], note=note)

        def outlier(note: str) -> dict[str, Any]:
            return claim(
                ppo_member(), rng.choice(by_specialty["Orthopedics"]), ["M17.11"], [("20610", 1, 520.0, [])], note=note
            )

        judged("mri_brain_necessity", "mri_brain", mri_brain, 2, ["R005"], {1: "pay"})
        judged("mri_lumbar_necessity", "mri_lumbar", mri_lumbar, 1, ["R005"], {})
        judged("er_without_auth", "er_auth", er_mri, 1, ["R004"], {})
        judged("modifier_59", "modifier_distinct", arthro, 2, ["R006"], {1: "pay"})
        judged("em_level_99215", "em_high", em_high, 1, ["R011"], {})
        judged("charge_outlier", "outlier", outlier, 1, ["R010"], {})
        # No documentation at all: a soft finding cannot be overridden.
        emit(mri_brain(""), "mri_brain_no_note", {1: "pay", 2: "deny"}, ["R005"], True)
        labeled[-1].claim["clinical_note"] = None
    return reg, labeled


def write(out_dir: Path, n_per_scenario: int = 8, seed: int = 11) -> dict[str, int]:
    reg, labeled = build(n_per_scenario, seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "registry.json").write_text(json.dumps(reg.__dict__, indent=1))
    with (out_dir / "claims_labeled.jsonl").open("w") as fh:
        for item in labeled:
            fh.write(
                json.dumps(
                    {
                        "scenario": item.scenario,
                        "expected_lines": item.expected_lines,
                        "expected_rules": item.expected_rules,
                        "needs_judgment": item.needs_judgment,
                        "claim": item.claim,
                    }
                )
                + "\n"
            )
    return {"members": len(reg.members), "providers": len(reg.providers), "claims": len(labeled)}
