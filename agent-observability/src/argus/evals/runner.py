"""Benchmark: detection and root-cause accuracy on simulated workloads with injected incidents.

Sweeps incident intensity × seeds. For each run: simulate → ingest via OTLP → detect → investigate,
then score against the injected ground truth. Also re-scans everything that was persisted to prove
that no raw sensitive data was stored.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from argus.config import Settings
from argus.governance.pii import scan
from argus.service import Argus
from argus.simulate.workload import MIN, InjectedIncident, Simulator

KIND_TO_CATEGORY = {
    "provider_degradation": "provider_degradation",
    "tool_failure": "tool_failure",
    "prompt_regression": "prompt_regression",
    "agent_regression": "agent_regression",
    "model_change": "model_change",
    "data_leak": "data_leak",
    "traffic_surge": "traffic_surge",
}


@dataclass
class RunScore:
    intensity: float
    seed: int
    expected: int = 0
    detected: int = 0
    argus_incidents: int = 0
    matched_incidents: int = 0
    delays_min: list[float] = field(default_factory=list)
    rca_category_ok: int = 0
    rca_cause_ok: int = 0
    rca_valid: int = 0
    stored_pii_leaks: int = 0
    per_kind: dict[str, list[bool]] = field(default_factory=dict)
    misses: list[str] = field(default_factory=list)


async def run_once(intensity: float, seed: int, minutes: int, use_llm: bool, base: Settings) -> RunScore:
    work = Path(tempfile.mkdtemp(prefix="argus-eval-"))
    argus = Argus(base.model_copy(update={"db_path": work / "argus.db"}))
    sim = Simulator(total_min=minutes, seed=seed, intensity=intensity)
    projects = {
        name: argus.store.create_project(name, org="acme") for name in ("support-bot", "research-agent", "claims-agent")
    }
    for name, payloads in sim.run().items():
        for payload in payloads:
            argus.ingestor.ingest(projects[name], payload)
    for name, events in sim.changes().items():
        for e in events:
            e.project_id = projects[name]
            argus.store.add_change(e)
    argus.detect_all()

    score = RunScore(intensity, seed)
    matched: set[int] = set()
    for inj in sim.incidents or []:
        for name in inj.projects:
            ok = await _score_incident(argus, projects[name], sim, inj, score, matched, use_llm)
            score.per_kind.setdefault(inj.kind, []).append(ok)
            if not ok:
                score.misses.append(f"{inj.id}:{inj.kind}@{name}")
    score.argus_incidents = sum(len(argus.store.incidents(pid)) for pid in projects.values())
    score.matched_incidents = len(matched)
    score.stored_pii_leaks = _stored_leaks(argus)
    return score


async def _score_incident(
    argus: Argus,
    project_id: str,
    sim: Simulator,
    inj: InjectedIncident,
    score: RunScore,
    matched: set[int],
    use_llm: bool,
) -> bool:
    score.expected += 1
    lo = sim.start_ns + (inj.start_min - 10) * MIN
    hi = sim.start_ns + (inj.end_min + 20) * MIN
    candidates = [
        i
        for i in argus.store.incidents(project_id)
        if i.start_ns < hi and lo < i.end_ns and set(i.detectors) & set(inj.expected_detectors)
    ]
    if not candidates:
        return False
    inc = min(candidates, key=lambda i: i.start_ns)
    assert inc.id is not None
    matched.add(inc.id)
    score.detected += 1
    first = min(
        a.window_end_ns for a in argus.store.anomalies(project_id, inc.id) if a.detector in inj.expected_detectors
    )
    score.delays_min.append(max(0.0, (first - (sim.start_ns + inj.start_min * MIN)) / MIN))
    result = await argus.investigate(project_id, inc.id, use_llm=use_llm)
    report = result["report"]
    category_ok = report.get("category") == KIND_TO_CATEGORY[inj.kind]
    score.rca_category_ok += category_ok
    score.rca_cause_ok += category_ok and str(report.get("cause_value")) == inj.cause_value
    score.rca_valid += not result["validation_errors"]
    return True


def _stored_leaks(argus: Argus) -> int:
    """Independent check: re-scan every persisted text field with the PII/secret detectors."""
    leaks = 0
    for row in argus.store.query("SELECT input_text, output_text, attributes FROM spans"):
        for value in row:
            if value:
                leaks += sum(1 for kind, _, _ in scan(value) if kind != "IP_ADDRESS")
    return leaks


def _pct(a: float, b: float) -> str:
    return f"{100 * a / b:.1f}%" if b else "-"


def render(scores: list[RunScore], meta: dict[str, str]) -> str:
    lines = (
        ["# Argus evaluation report", ""]
        + [f"- **{k}**: {v}" for k, v in meta.items()]
        + [
            "",
            "## Detection and root cause by incident intensity",
            "",
            "| intensity | runs | recall | precision | median delay | RCA category | RCA exact cause | evidence valid |"
            " stored PII leaks |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
    )
    kinds: dict[str, dict[float, list[bool]]] = {}
    for intensity in sorted({s.intensity for s in scores}, reverse=True):
        group = [s for s in scores if s.intensity == intensity]
        exp, det = sum(s.expected for s in group), sum(s.detected for s in group)
        delays = [d for s in group for d in s.delays_min]
        median_delay = f"{statistics.median(delays):.0f} min" if delays else "-"
        lines.append(
            f"| {intensity:g} | {len(group)} | {_pct(det, exp)} | "
            f"{_pct(sum(s.matched_incidents for s in group), sum(s.argus_incidents for s in group))} | "
            f"{median_delay} | "
            f"{_pct(sum(s.rca_category_ok for s in group), det)} | {_pct(sum(s.rca_cause_ok for s in group), det)} | "
            f"{_pct(sum(s.rca_valid for s in group), det)} | {sum(s.stored_pii_leaks for s in group)} |"
        )
        for s in group:
            for kind, hits in s.per_kind.items():
                kinds.setdefault(kind, {}).setdefault(intensity, []).extend(hits)
    intensities = sorted({s.intensity for s in scores}, reverse=True)
    lines += [
        "",
        "## Recall by incident type",
        "",
        "| incident type | " + " | ".join(f"intensity {i:g}" for i in intensities) + " |",
        "|---|" + "---|" * len(intensities),
    ]
    for kind, by in kinds.items():
        lines.append(
            f"| {kind} | " + " | ".join(_pct(sum(by.get(i, [])), len(by.get(i, []))) for i in intensities) + " |"
        )
    misses = [f"{s.intensity:g}/seed {s.seed}: {m}" for s in scores for m in s.misses]
    if misses:
        lines += ["", "## Missed incidents", ""] + [f"- {m}" for m in misses]
    return "\n".join(lines) + "\n"


async def run_eval(
    settings: Settings, out: Path, *, intensities: list[float], seeds: list[int], minutes: int, use_llm: bool
) -> list[RunScore]:
    started = time.perf_counter()
    scores = []
    for intensity in intensities:
        for seed in seeds:
            scores.append(await run_once(intensity, seed, minutes, use_llm, settings))
            print(
                f"intensity={intensity} seed={seed}: {scores[-1].detected}/{scores[-1].expected} detected, "
                f"{scores[-1].argus_incidents} incidents",
                flush=True,
            )
    meta = {
        "investigator": settings.investigator_model if use_llm else "deterministic RCA (rules over analysis tools)",
        "workload": f"3 simulated agent apps, {minutes // 60} h each run, 7 injected incident types "
        f"(9 incident×project pairs per run)",
        "seeds": ", ".join(map(str, seeds)),
        "wall time": f"{time.perf_counter() - started:.0f} s",
    }
    await asyncio.to_thread(_write, out, render(scores, meta), scores)
    return scores


def _write(out: Path, report: str, scores: list[RunScore]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    out.with_suffix(".json").write_text(json.dumps([s.__dict__ for s in scores], indent=2, default=str))


def main_sync(settings: Settings, out: Path, quick: bool, use_llm: bool) -> None:
    if quick:
        asyncio.run(run_eval(settings, out, intensities=[1.0], seeds=[1], minutes=720, use_llm=use_llm))
    else:
        asyncio.run(
            run_eval(settings, out, intensities=[1.0, 0.5, 0.25, 0.15], seeds=[1, 2, 3], minutes=720, use_llm=use_llm)
        )
