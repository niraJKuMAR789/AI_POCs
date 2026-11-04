"""The fault drill: one Argus org, three production apps, a day of traffic, three injected incidents.

run_drill()
  boot Atlas, Quill, Aegis (instrumented)         one Argus project each, org "helix"
  replay N minutes of Poisson traffic per app      real requests on a virtual clock
    + deploys, a prompt release and rollback        recorded as Argus change events
    + three injected faults                         Argus is not told about them
  Argus: detect → group into incidents → RCA       deterministic or LLM investigator
  score against the injected faults                detection, delay, root cause, false alarms
"""

from __future__ import annotations

import inspect
import json
import random
import shutil
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import argus
from argus.config import Settings as ArgusSettings
from argus.models import ChangeEvent, Incident
from argus.sdk.tracer import Tracer, use_tracer
from argus.service import Argus, InProcessExporter

from mission_control.apps import BOOT, AppRuntime, Provider, project_root
from mission_control.faults import Fault
from mission_control.instrument import deployment
from mission_control.timeline import MINUTE_NS, ShiftingExporter, Timeline

ORG = "helix"
MISSION_ROOT = Path(__file__).resolve().parents[2]
WINDOW_MIN = 10


@dataclass(frozen=True)
class Change:
    minute: float
    app: str
    kind: str
    service: str
    description: str
    attributes: dict[str, Any] = field(default_factory=dict, compare=False)
    deploy: dict[str, str] | None = field(default=None, compare=False)  # new deployment state, if any


@dataclass
class Scenario:
    minutes: int
    faults: list[Fault]
    changes: list[Change]
    initial: dict[str, dict[str, str]]  # app → deployment at minute 0


def default_scenario(minutes: int = 360) -> Scenario:
    """Three incidents, each after a clean baseline and in a different app, with different signatures:
    a slow dependency after a deploy, a prompt release, and a provider-side outage with no deploy at all."""
    at = lambda frac: round(minutes * frac)  # noqa: E731
    f1, f2, f3 = at(0.40), at(0.58), at(0.78)
    faults = [
        Fault(
            "F1",
            "atlas",
            "Reranker latency after retrieval deploy",
            f1,
            f1 + 45,
            "tool_failure",
            "tool_name",
            "rerank",
            "atlas-retrieval 1.4.0 moved the reranker to a shared GPU pool; each rerank call slows down",
        ),
        Fault(
            "F2",
            "quill",
            "Prompt sql-v2 regression",
            f2,
            f2 + 45,
            "prompt_regression",
            "prompt_version",
            "sql-v2",
            "sql-v2 prompt is ~3x longer and nudges the model to select member PII; the guard rejects those "
            "queries and the repair loop adds an extra LLM call",
        ),
        Fault(
            "F3",
            "aegis",
            "Reviewer model endpoint timeouts",
            f3,
            f3 + 40,
            "provider_degradation",
            "model",
            "{aegis.clinical}",
            "70% of reviewer calls time out upstream; no deploy, nothing changed on our side",
        ),
    ]
    changes = [
        Change(
            at(0.15),
            "atlas",
            "deploy",
            "atlas-api",
            "atlas-api 1.3.2: citation formatting fix",
            {"agent_version": "atlas-1.3.2"},
            {"agent_version": "atlas-1.3.2"},
        ),
        Change(
            f1,
            "atlas",
            "deploy",
            "atlas-retrieval",
            "atlas-retrieval 1.4.0: reranker moved to shared GPU pool",
            {"tool_name": "rerank", "version": "1.4.0"},
        ),
        Change(f1 + 45, "atlas", "config", "atlas-retrieval", "reranker pinned back to dedicated pool"),
        Change(
            f2,
            "quill",
            "prompt_change",
            "quill-api",
            "SQL prompt sql-v2 rolled out to 100%",
            {"prompt_version": "sql-v2"},
            {"agent_version": "quill-2.1.0", "prompt_version": "sql-v2"},
        ),
        Change(
            f2 + 45,
            "quill",
            "prompt_change",
            "quill-api",
            "rollback: SQL prompt back to sql-v1",
            {"rollback": True},
            {"agent_version": "quill-2.1.0", "prompt_version": "sql-v1"},
        ),
    ]
    initial = {
        "atlas": {"agent_version": "atlas-1.3.1"},
        "quill": {"agent_version": "quill-2.1.0", "prompt_version": "sql-v1"},
        "aegis": {"agent_version": "aegis-0.9.0", "prompt_version": "reviewer-v3"},
    }
    return Scenario(minutes, faults, changes, initial)


@dataclass
class DrillConfig:
    workdir: Path
    provider: Provider = "offline"
    minutes: int = 360
    # Requests per minute per app. Claims arrive in batches from clearing houses, so Aegis sees more volume.
    rates: dict[str, float] = field(default_factory=lambda: {"atlas": 1.2, "quill": 1.2, "aegis": 2.0})
    seed: int = 7
    llm_rca: bool = False


@dataclass
class FaultScore:
    fault: Fault
    detected: bool = False
    delay_min: float | None = None
    incidents: list[int] = field(default_factory=list)
    rca_correct: bool = False
    rca: dict[str, Any] | None = None


@dataclass
class DrillResult:
    config: DrillConfig
    scenario: Scenario
    argus: Argus
    projects: dict[str, str]
    runtimes: dict[str, AppRuntime]
    requests: list[dict[str, Any]]
    incidents: dict[str, list[Incident]]
    investigations: dict[int, dict[str, Any]]
    scores: list[FaultScore]
    false_alarms: list[Incident]
    wall_s: float


def _arrivals(rng: random.Random, minutes: int, rate: float) -> list[float]:
    out, t = [], rng.expovariate(rate)
    while t < minutes:
        out.append(t)
        t += rng.expovariate(rate)
    return out


async def run_drill(cfg: DrillConfig, scenario: Scenario | None = None) -> DrillResult:
    scenario = scenario or default_scenario(cfg.minutes)
    started = time.perf_counter()
    if cfg.workdir.exists():
        shutil.rmtree(cfg.workdir)
    cfg.workdir.mkdir(parents=True)

    argus_root = project_root(argus)
    service = Argus(
        ArgusSettings(  # type: ignore[call-arg]
            db_path=cfg.workdir / "argus.db",
            policies=MISSION_ROOT / "config" / "policies.yaml",
            pricing=argus_root / "config" / "pricing.yaml",
            provider="nvidia" if cfg.llm_rca else "offline",
            _env_file=None,
        )
    )
    now = time.time_ns()
    window = WINDOW_MIN * MINUTE_NS
    start_ns = (now - scenario.minutes * MINUTE_NS) // window * window  # the drill ends "now"
    timeline = Timeline(start_ns)

    runtimes: dict[str, AppRuntime] = {}
    tracers: dict[str, Tracer] = {}
    exporters: dict[str, InProcessExporter] = {}
    projects: dict[str, str] = {}
    for name, boot in BOOT.items():
        runtimes[name] = await boot(cfg.workdir, cfg.provider, timeline, scenario.faults)
        projects[name] = service.store.create_project(name, org=ORG)
        exporters[name] = InProcessExporter(service.ingestor, projects[name])
        tracers[name] = Tracer(
            name,
            ShiftingExporter(exporters[name]),
            resource={"deployment.environment": "production", "service.namespace": ORG},
        )
    faults = [_resolve(f, runtimes) for f in scenario.faults]
    # Read-only dashboard keys, one per tenant (stored hashed in Argus; plaintext only in the local workdir).
    keys = {
        app: service.store.create_key(pid, "mission-dashboard", ["read", "investigate"])
        for app, pid in projects.items()
    }
    (cfg.workdir / "dashboard_keys.json").write_text(json.dumps(keys, indent=2))

    for change in scenario.changes:
        service.store.add_change(
            ChangeEvent(
                project_id=projects[change.app],
                ts_ns=timeline.at(change.minute),
                kind=change.kind,
                service=change.service,
                description=change.description,
                attributes=change.attributes,
            )
        )

    rng = random.Random(cfg.seed)
    schedule = sorted((minute, app) for app in runtimes for minute in _arrivals(rng, scenario.minutes, cfg.rates[app]))
    requests: list[dict[str, Any]] = []
    for minute, app in schedule:
        runtime = runtimes[app]
        item = rng.choice(runtime.items)
        token = deployment.set(_deployment_at(scenario, app, minute))
        t0 = time.perf_counter()
        error = None
        try:
            with timeline.request_at(minute), use_tracer(tracers[app]):
                kpis = await runtime.handle(item, rng)
        except Exception as exc:  # an app failure is data for Argus, not a reason to stop the drill
            kpis, error = {}, f"{type(exc).__name__}: {exc}"
        finally:
            deployment.reset(token)
        exporters[app].flush()
        requests.append(
            {"app": app, "minute": minute, "latency_ms": (time.perf_counter() - t0) * 1000, "error": error, **kpis}
        )

    service.detect_all()
    incidents = {app: service.store.incidents(pid) for app, pid in projects.items()}
    investigations: dict[int, dict[str, Any]] = {}
    for app, incs in incidents.items():
        for inc in incs:
            assert inc.id is not None
            investigations[inc.id] = await service.investigate(projects[app], inc.id, use_llm=cfg.llm_rca)

    scores, matched = [], set()
    for fault in faults:
        score = _score(fault, service, projects[fault.app], incidents[fault.app], investigations, timeline)
        matched.update(score.incidents)
        scores.append(score)
    false_alarms = [i for incs in incidents.values() for i in incs if i.id not in matched]

    for runtime in runtimes.values():
        closing = runtime.close()
        if inspect.isawaitable(closing):
            await closing
    return DrillResult(
        cfg,
        scenario,
        service,
        projects,
        runtimes,
        requests,
        incidents,
        investigations,
        scores,
        false_alarms,
        time.perf_counter() - started,
    )


def _resolve(fault: Fault, runtimes: dict[str, AppRuntime]) -> Fault:
    """Fill in expected causes that depend on the provider (e.g. which model serves Aegis's reviewers)."""
    if fault.cause_value.startswith("{") and fault.cause_value.endswith("}"):
        app, role = fault.cause_value[1:-1].split(".")
        from dataclasses import replace

        return replace(fault, cause_value=runtimes[app].models[role])
    return fault


def _deployment_at(scenario: Scenario, app: str, minute: float) -> dict[str, str]:
    state = dict(scenario.initial.get(app, {}))
    for change in sorted(scenario.changes, key=lambda c: c.minute):
        if change.app == app and change.deploy is not None and change.minute <= minute:
            state.update(change.deploy)
    return state


def _score(
    fault: Fault,
    service: Argus,
    project_id: str,
    incidents: list[Incident],
    investigations: dict[int, dict[str, Any]],
    timeline: Timeline,
) -> FaultScore:
    lo, hi = timeline.at(fault.start_min), timeline.at(fault.end_min + WINDOW_MIN)
    score = FaultScore(fault)
    overlapping = [i for i in incidents if i.start_ns < hi and lo < i.end_ns]
    if not overlapping:
        return score
    score.detected = True
    score.incidents = [i.id for i in overlapping if i.id is not None]
    first = min(
        a.window_end_ns for i in overlapping for a in service.store.anomalies(project_id, i.id) if a.window_end_ns >= lo
    )
    score.delay_min = round((first - lo) / MINUTE_NS, 1)
    for inc in overlapping:
        assert inc.id is not None
        report = investigations[inc.id]["report"]
        if report.get("category") == fault.expected_category and report.get("cause_value") == fault.cause_value:
            score.rca_correct, score.rca = True, {"incident": inc.id, **report}
            break
    if score.rca is None and overlapping[0].id is not None:
        score.rca = {"incident": overlapping[0].id, **investigations[overlapping[0].id]["report"]}
    return score


def kpis(result: DrillResult) -> dict[str, dict[str, Any]]:
    """Each app's own health metrics inside vs. outside its fault window: what the business would notice."""
    out: dict[str, dict[str, Any]] = {}
    for fault in (s.fault for s in result.scores):
        rows = [r for r in result.requests if r["app"] == fault.app]
        during = [r for r in rows if fault.active(r["minute"])]
        before = [r for r in rows if r["minute"] < fault.start_min]
        entry: dict[str, Any] = {
            "p50_latency_ms": (_p50(before, "latency_ms"), _p50(during, "latency_ms")),
            "requests": (len(before), len(during)),
        }
        if fault.app == "quill":
            entry["avg_sql_attempts"] = (_mean(before, "attempts"), _mean(during, "attempts"))
        if fault.app == "aegis":
            entry["sent_to_human_review"] = (_share(before, "escalated"), _share(during, "escalated"))
        out[fault.id] = entry
    return out


def _p50(rows: list[dict[str, Any]], key: str) -> float:
    return round(statistics.median([r[key] for r in rows]), 1) if rows else 0.0


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    vals = [r[key] for r in rows if key in r]
    return round(statistics.fmean(vals), 2) if vals else 0.0


def _share(rows: list[dict[str, Any]], key: str) -> float:
    vals = [bool(r.get(key)) for r in rows if key in r]
    return round(sum(vals) / len(vals), 3) if vals else 0.0


Mode = Literal["offline", "nvidia"]
