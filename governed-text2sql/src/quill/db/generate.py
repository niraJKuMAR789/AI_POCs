"""Deterministic synthetic healthcare-claims warehouse (SQLite).

All people, providers and claims are fictional and generated from a fixed seed, so
the database (and therefore every eval and training example) is reproducible.
"""

from __future__ import annotations

import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

SCHEMA = """
CREATE TABLE plans (
    plan_id INTEGER PRIMARY KEY,
    plan_name TEXT NOT NULL,
    plan_type TEXT NOT NULL CHECK (plan_type IN ('HMO', 'PPO', 'EPO')),
    metal_tier TEXT NOT NULL CHECK (metal_tier IN ('Bronze', 'Silver', 'Gold', 'Platinum')),
    monthly_premium REAL NOT NULL
);
CREATE TABLE members (
    member_id INTEGER PRIMARY KEY,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    date_of_birth TEXT NOT NULL,
    gender TEXT NOT NULL CHECK (gender IN ('F', 'M', 'X')),
    state TEXT NOT NULL,
    plan_id INTEGER NOT NULL REFERENCES plans(plan_id),
    enrollment_date TEXT NOT NULL,
    ssn_last4 TEXT NOT NULL
);
CREATE TABLE providers (
    provider_id INTEGER PRIMARY KEY,
    npi TEXT NOT NULL UNIQUE,
    provider_name TEXT NOT NULL,
    specialty TEXT NOT NULL,
    state TEXT NOT NULL,
    network_status TEXT NOT NULL CHECK (network_status IN ('in_network', 'out_of_network'))
);
CREATE TABLE diagnoses (
    code TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    category TEXT NOT NULL
);
CREATE TABLE procedures (
    code TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    category TEXT NOT NULL,
    base_cost REAL NOT NULL
);
CREATE TABLE claims (
    claim_id INTEGER PRIMARY KEY,
    member_id INTEGER NOT NULL REFERENCES members(member_id),
    provider_id INTEGER NOT NULL REFERENCES providers(provider_id),
    service_date TEXT NOT NULL,
    submitted_date TEXT NOT NULL,
    claim_type TEXT NOT NULL CHECK (claim_type IN ('professional', 'institutional', 'pharmacy')),
    status TEXT NOT NULL CHECK (status IN ('paid', 'denied', 'pending', 'adjusted')),
    billed_amount REAL NOT NULL,
    allowed_amount REAL NOT NULL,
    paid_amount REAL NOT NULL,
    denial_reason TEXT,
    primary_diagnosis_code TEXT NOT NULL REFERENCES diagnoses(code)
);
CREATE TABLE claim_lines (
    line_id INTEGER PRIMARY KEY,
    claim_id INTEGER NOT NULL REFERENCES claims(claim_id),
    procedure_code TEXT NOT NULL REFERENCES procedures(code),
    units INTEGER NOT NULL,
    line_billed REAL NOT NULL,
    line_paid REAL NOT NULL
);
CREATE TABLE audit_flags (
    flag_id INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_id INTEGER NOT NULL REFERENCES claims(claim_id),
    reason TEXT NOT NULL,
    flagged_by TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_claims_member ON claims(member_id);
CREATE INDEX idx_claims_provider ON claims(provider_id);
CREATE INDEX idx_claims_service_date ON claims(service_date);
CREATE INDEX idx_claims_status ON claims(status);
CREATE INDEX idx_lines_claim ON claim_lines(claim_id);
"""

STATES = ["TX", "CA", "NY", "FL", "IL", "PA", "OH", "GA", "WA", "AZ"]
FIRST = [
    "Ava",
    "Liam",
    "Mia",
    "Noah",
    "Zoe",
    "Ethan",
    "Isla",
    "Lucas",
    "Aria",
    "Mateo",
    "Nora",
    "Kai",
    "Leah",
    "Omar",
    "Priya",
    "Ravi",
    "Sofia",
    "Wei",
    "Yara",
    "Diego",
    "Hana",
    "Ivan",
    "Jade",
    "Kofi",
]
LAST = [
    "Garcia",
    "Nguyen",
    "Patel",
    "Smith",
    "Kim",
    "Okafor",
    "Silva",
    "Cohen",
    "Haddad",
    "Ivanova",
    "Johnson",
    "Lopez",
    "Martin",
    "Reyes",
    "Sato",
    "Tanaka",
    "Walker",
    "Zhang",
    "Brown",
    "Ali",
]
SPECIALTIES = [
    "Cardiology",
    "Orthopedics",
    "Dermatology",
    "Family Medicine",
    "Pediatrics",
    "Oncology",
    "Radiology",
    "Endocrinology",
    "Psychiatry",
    "Emergency Medicine",
]
DIAGNOSES = [
    ("E11.9", "Type 2 diabetes mellitus without complications", "Endocrine"),
    ("I10", "Essential (primary) hypertension", "Circulatory"),
    ("J06.9", "Acute upper respiratory infection", "Respiratory"),
    ("M54.5", "Low back pain", "Musculoskeletal"),
    ("F41.1", "Generalized anxiety disorder", "Mental health"),
    ("K21.9", "Gastro-esophageal reflux disease", "Digestive"),
    ("S83.5", "Sprain of cruciate ligament of knee", "Injury"),
    ("C50.9", "Malignant neoplasm of breast", "Neoplasm"),
    ("L40.0", "Psoriasis vulgaris", "Skin"),
    ("I25.10", "Atherosclerotic heart disease", "Circulatory"),
    ("J45.909", "Unspecified asthma", "Respiratory"),
    ("E78.5", "Hyperlipidemia", "Endocrine"),
]
PROCEDURES = [
    ("99213", "Office visit, established patient, low complexity", "Evaluation", 110.0),
    ("99214", "Office visit, established patient, moderate complexity", "Evaluation", 165.0),
    ("99285", "Emergency department visit, high severity", "Emergency", 650.0),
    ("93000", "Electrocardiogram with interpretation", "Diagnostic", 60.0),
    ("71046", "Chest X-ray, two views", "Imaging", 95.0),
    ("70553", "MRI brain with and without contrast", "Imaging", 1400.0),
    ("80053", "Comprehensive metabolic panel", "Lab", 45.0),
    ("83036", "Hemoglobin A1c", "Lab", 30.0),
    ("29881", "Knee arthroscopy with meniscectomy", "Surgery", 4200.0),
    ("90834", "Psychotherapy, 45 minutes", "Behavioral", 140.0),
    ("96413", "Chemotherapy infusion, first hour", "Oncology", 900.0),
    ("17000", "Destruction of benign skin lesion", "Dermatology", 120.0),
]
# Diagnosis → plausible procedures and specialties, so the data has realistic structure to query.
DX_PROCS = {
    "E11.9": ["99213", "83036", "80053"],
    "I10": ["99213", "93000", "80053"],
    "J06.9": ["99213", "71046"],
    "M54.5": ["99214", "70553"],
    "F41.1": ["90834", "99213"],
    "K21.9": ["99213", "99214"],
    "S83.5": ["29881", "99285", "70553"],
    "C50.9": ["96413", "99214", "80053"],
    "L40.0": ["17000", "99213"],
    "I25.10": ["93000", "99214", "99285"],
    "J45.909": ["99213", "71046"],
    "E78.5": ["80053", "99213"],
}
DX_SPECIALTY = {
    "E11.9": "Endocrinology",
    "I10": "Family Medicine",
    "J06.9": "Family Medicine",
    "M54.5": "Orthopedics",
    "F41.1": "Psychiatry",
    "K21.9": "Family Medicine",
    "S83.5": "Orthopedics",
    "C50.9": "Oncology",
    "L40.0": "Dermatology",
    "I25.10": "Cardiology",
    "J45.909": "Pediatrics",
    "E78.5": "Cardiology",
}
DENIAL_REASONS = [
    "Prior authorization missing",
    "Not medically necessary",
    "Out-of-network provider",
    "Duplicate claim",
    "Coding error",
    "Coverage terminated",
]
PLANS = [
    (1, "Helios Basic HMO", "HMO", "Bronze", 289.0),
    (2, "Helios Select PPO", "PPO", "Silver", 412.0),
    (3, "Helios Plus EPO", "EPO", "Silver", 365.0),
    (4, "Helios Premier PPO", "PPO", "Gold", 538.0),
    (5, "Helios Elite PPO", "PPO", "Platinum", 701.0),
    (6, "Helios Value HMO", "HMO", "Gold", 455.0),
]


def generate(path: Path, *, members: int = 2000, providers: int = 300, claims: int = 25000, seed: int = 7) -> Path:
    rng = random.Random(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    conn.executemany("INSERT INTO plans VALUES (?, ?, ?, ?, ?)", PLANS)
    conn.executemany("INSERT INTO diagnoses VALUES (?, ?, ?)", DIAGNOSES)
    conn.executemany("INSERT INTO procedures VALUES (?, ?, ?, ?)", PROCEDURES)

    member_rows = []
    for mid in range(1, members + 1):
        dob = date(1940, 1, 1) + timedelta(days=rng.randint(0, 365 * 80))
        enrolled = date(2021, 1, 1) + timedelta(days=rng.randint(0, 365 * 4))
        member_rows.append(
            (
                mid,
                rng.choice(FIRST),
                rng.choice(LAST),
                dob.isoformat(),
                rng.choices("FMX", [49, 49, 2])[0],
                rng.choice(STATES),
                rng.choices([p[0] for p in PLANS], [25, 25, 15, 18, 7, 10])[0],
                enrolled.isoformat(),
                f"{rng.randint(0, 9999):04d}",
            )
        )
    conn.executemany("INSERT INTO members VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", member_rows)

    provider_rows = []
    by_specialty: dict[str, list[int]] = {s: [] for s in SPECIALTIES}
    for pid in range(1, providers + 1):
        specialty = SPECIALTIES[(pid - 1) % len(SPECIALTIES)] if pid <= len(SPECIALTIES) else rng.choice(SPECIALTIES)
        by_specialty[specialty].append(pid)
        provider_rows.append(
            (
                pid,
                f"{1000000000 + pid * 7919:010d}",
                f"{rng.choice(LAST)} {specialty} Associates",
                specialty,
                rng.choice(STATES),
                rng.choices(["in_network", "out_of_network"], [85, 15])[0],
            )
        )
    conn.executemany("INSERT INTO providers VALUES (?, ?, ?, ?, ?, ?)", provider_rows)
    network = {row[0]: row[5] for row in provider_rows}
    proc_cost = {p[0]: p[3] for p in PROCEDURES}

    claim_rows, line_rows = [], []
    line_id = 1
    for cid in range(1, claims + 1):
        dx = rng.choice(DIAGNOSES)[0]
        pool = by_specialty[DX_SPECIALTY[dx]] if rng.random() < 0.8 else [p[0] for p in provider_rows]
        pid = rng.choice(pool)
        service = date(2024, 1, 1) + timedelta(days=rng.randint(0, 365 * 2 - 1))
        submitted = service + timedelta(days=rng.randint(1, 45))
        claim_type = rng.choices(["professional", "institutional", "pharmacy"], [70, 22, 8])[0]
        procs = rng.sample(DX_PROCS[dx], k=rng.randint(1, len(DX_PROCS[dx])))
        lines, billed_total = [], 0.0
        for code in procs:
            units = rng.choices([1, 2, 3], [80, 15, 5])[0]
            billed = round(proc_cost[code] * units * rng.uniform(1.1, 1.9), 2)
            billed_total += billed
            lines.append((code, units, billed))
        deny_p = 0.08 + (0.17 if network[pid] == "out_of_network" else 0) + (0.05 if billed_total > 3000 else 0)
        roll = rng.random()
        status = "denied" if roll < deny_p else rng.choices(["paid", "pending", "adjusted"], [86, 8, 6])[0]
        allowed = round(billed_total * rng.uniform(0.55, 0.85), 2)
        paid_ratio = {"paid": rng.uniform(0.75, 0.95), "adjusted": rng.uniform(0.4, 0.7)}.get(status, 0.0)
        paid = round(allowed * paid_ratio, 2)
        reason = None
        if status == "denied":
            reason = (
                "Out-of-network provider"
                if network[pid] == "out_of_network" and rng.random() < 0.6
                else rng.choice(DENIAL_REASONS)
            )
        claim_rows.append(
            (
                cid,
                rng.randint(1, members),
                pid,
                service.isoformat(),
                submitted.isoformat(),
                claim_type,
                status,
                round(billed_total, 2),
                allowed,
                paid,
                reason,
                dx,
            )
        )
        for code, units, billed in lines:
            share = billed / billed_total if billed_total else 0
            line_rows.append((line_id, cid, code, units, billed, round(paid * share, 2)))
            line_id += 1
    conn.executemany("INSERT INTO claims VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", claim_rows)
    conn.executemany("INSERT INTO claim_lines VALUES (?, ?, ?, ?, ?, ?)", line_rows)
    conn.commit()
    conn.execute("ANALYZE")
    conn.close()
    return path
