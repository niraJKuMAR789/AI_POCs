from __future__ import annotations

from pathlib import Path

import pytest

from argus.config import Settings
from argus.service import Argus
from argus.simulate.workload import SUPER, InjectedIncident, Simulator

ROOT = Path(__file__).resolve().parents[1]

TEST_INCIDENTS = [
    InjectedIncident(
        "T1",
        "provider_degradation",
        ("support-bot", "research-agent", "claims-agent"),
        150,
        195,
        "model",
        SUPER,
        ("latency", "llm_latency"),
        "slow model",
    ),
    InjectedIncident(
        "T2",
        "tool_failure",
        ("support-bot",),
        240,
        270,
        "tool_name",
        "search_kb",
        ("error_rate", "tool_error_rate"),
        "kb outage",
    ),
    InjectedIncident(
        "T3", "data_leak", ("claims-agent",), 240, 280, "tool_name", "get_claim", ("sensitive_data",), "leak"
    ),
    InjectedIncident(
        "T4", "agent_regression", ("research-agent",), 250, 310, "agent_version", "v1.4", ("tool_loop", "steps"), "loop"
    ),
    InjectedIncident("T5", "traffic_surge", ("support-bot",), 320, 350, "traffic", "surge", ("throughput",), "surge"),
]


def make_settings(tmp: Path) -> Settings:
    return Settings(
        db_path=tmp / "argus.db",
        policies=ROOT / "config" / "policies.yaml",
        pricing=ROOT / "config" / "pricing.yaml",
        provider="offline",
        _env_file=None,
    )  # type: ignore[call-arg]


@pytest.fixture
def argus(tmp_path: Path) -> Argus:
    return Argus(make_settings(tmp_path))


@pytest.fixture(scope="session")
def simulated(tmp_path_factory: pytest.TempPathFactory) -> tuple[Argus, Simulator, dict[str, str]]:
    """One shared 6-hour simulated workload with injected incidents, ingested and detected."""
    argus = Argus(make_settings(tmp_path_factory.mktemp("sim")))
    sim = Simulator(total_min=360, seed=3, incidents=list(TEST_INCIDENTS))
    projects = {n: argus.store.create_project(n, org="acme") for n in ("support-bot", "research-agent", "claims-agent")}
    for name, payloads in sim.run().items():
        for payload in payloads:
            argus.ingestor.ingest(projects[name], payload)
    for name, events in sim.changes().items():
        for e in events:
            e.project_id = projects[name]
            argus.store.add_change(e)
    argus.detect_all()
    return argus, sim, projects
