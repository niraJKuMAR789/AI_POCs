"""End-to-end: boot all three apps, replay four hours of traffic with three injected faults, and require
Argus to detect and explain every one of them without false alarms."""

from __future__ import annotations

from pathlib import Path

from mission_control.drill import DrillConfig, kpis, run_drill
from mission_control.report import render


async def test_argus_catches_and_explains_every_injected_fault(tmp_path: Path) -> None:
    result = await run_drill(DrillConfig(workdir=tmp_path / "mission", minutes=240, seed=3))

    assert not [r for r in result.requests if r["error"]], "apps must degrade, not crash"
    for score in result.scores:
        assert score.detected, f"{score.fault.id} was not detected"
        assert score.delay_min is not None and score.delay_min <= 20
        assert score.rca_correct, f"{score.fault.id}: wrong root cause {score.rca}"
    assert result.false_alarms == []

    impact = kpis(result)
    assert impact["F3"]["sent_to_human_review"][1] > impact["F3"]["sent_to_human_review"][0]
    assert impact["F2"]["avg_sql_attempts"][1] > 1.0
    report = render([result])
    assert "None: every incident Argus opened maps to an injected fault." in report


async def test_apps_are_isolated_tenants_in_one_org(tmp_path: Path) -> None:
    result = await run_drill(DrillConfig(workdir=tmp_path / "mission", minutes=60, seed=1))
    store = result.argus.store
    names = {p["name"]: p["org"] for p in store.projects()}
    assert {"atlas", "quill", "aegis"} <= set(names)
    assert {names[a] for a in ("atlas", "quill", "aegis")} == {"helix"}
    for app, pid in result.projects.items():
        rows = store.query("SELECT DISTINCT service FROM traces WHERE project_id = ?", (pid,))
        assert [r["service"] for r in rows] == [app]
