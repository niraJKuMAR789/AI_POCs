import random

from argus.detect.detectors import WINDOW_NS, Config, DetectorSuite, Window
from argus.detect.stats import binom_sf, poisson_sf, quantile, robust_z
from argus.models import TraceSummary
from argus.service import Argus
from argus.simulate.workload import MIN, Simulator


def test_stats() -> None:
    assert quantile([1, 2, 3, 4], 0.5) == 2.5
    z, center = robust_z(30, [10, 11, 9, 10, 12, 10])
    assert center == 10 and z > 10
    assert abs(binom_sf(0, 10, 0.3) - 1.0) < 1e-12 and binom_sf(11, 10, 0.3) == 0.0
    assert binom_sf(8, 10, 0.1) < 1e-6 < binom_sf(2, 10, 0.1)
    assert poisson_sf(40, 10) < 1e-9 < poisson_sf(12, 10)


def _window(i: int, durations: list[float], errors: int = 0) -> Window:
    traces = [
        TraceSummary(
            project_id="p",
            trace_id=f"{i}-{j}",
            root_name="bot",
            service="bot",
            start_ns=i * WINDOW_NS,
            end_ns=i * WINDOW_NS + 1,
            duration_ms=d,
            span_count=4,
            llm_calls=1,
            tool_calls=1,
            input_tokens=100,
            output_tokens=10,
            cost_usd=0.001,
            error_count=int(j < errors),
            max_depth=2,
            max_tool_repeat=1,
            has_root=True,
        )
        for j, d in enumerate(durations)
    ]
    return Window(i * WINDOW_NS, (i + 1) * WINDOW_NS, traces, [])


def test_suite_flags_spikes_and_ignores_noise() -> None:
    rng = random.Random(0)
    suite = DetectorSuite("p", Config())
    normal = lambda: [rng.gauss(1000, 120) for _ in range(25)]  # noqa: E731
    flagged = [a for i in range(30) for a in suite.observe(_window(i, normal()))]
    assert flagged == []  # 30 windows of noise: no false positives
    spike = suite.observe(_window(30, [d * 4 for d in normal()], errors=12))
    assert {a.detector for a in spike} == {"latency", "error_rate"}
    # Baselines exclude flagged windows, so a sustained incident keeps being detected.
    still = suite.observe(_window(31, [d * 4 for d in normal()]))
    assert any(a.detector == "latency" for a in still)


def test_engine_detects_injected_incidents_without_false_alarms(simulated) -> None:  # type: ignore[no-untyped-def]
    argus, sim, projects = simulated
    for inj in sim.incidents:
        for name in inj.projects:
            lo, hi = sim.start_ns + (inj.start_min - 10) * MIN, sim.start_ns + (inj.end_min + 20) * MIN
            hits = [
                i
                for i in argus.store.incidents(projects[name])
                if i.start_ns < hi and lo < i.end_ns and set(i.detectors) & set(inj.expected_detectors)
            ]
            assert hits, f"{inj.id} not detected in {name}"
    surge = next(i for i in argus.store.incidents(projects["support-bot"]) if "throughput" in i.detectors)
    assert set(surge.detectors) == {"throughput"}  # a healthy surge must not page as latency/errors
    total = sum(len(argus.store.incidents(p)) for p in projects.values())
    assert total == 7  # T1 in three projects + T2 + T3 + T4 + T5: nothing else


def test_detection_is_idempotent(simulated) -> None:  # type: ignore[no-untyped-def]
    argus, _, projects = simulated
    before = len(argus.store.anomalies(projects["support-bot"]))
    assert argus.engine.run(projects["support-bot"]) == []
    assert len(argus.store.anomalies(projects["support-bot"])) == before


def test_new_model_novelty(argus: Argus) -> None:
    sim = Simulator(total_min=200, seed=5, incidents=[])
    pid = argus.store.create_project("claims-agent", org="acme")
    for p in sim.run()["claims-agent"]:
        argus.ingestor.ingest(pid, p)
    argus.engine.run(pid)
    assert not [a for a in argus.store.anomalies(pid) if a.detector == "new_model"]
