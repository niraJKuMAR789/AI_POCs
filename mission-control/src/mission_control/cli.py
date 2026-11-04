"""`mission-control drill` runs the fault drill; `mission-control dashboard` opens Argus's UI on its data."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import structlog


def _quiet() -> None:
    """The apps log every agent step; for a 1,500-request drill only the summary is useful."""
    logging.basicConfig(level=logging.WARNING)
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))


async def _drill(args: argparse.Namespace) -> None:
    from mission_control.drill import DrillConfig, DrillResult, run_drill
    from mission_control.report import render

    results: list[DrillResult] = []
    for i, seed in enumerate(args.seeds):
        workdir = args.workdir if i == 0 else args.workdir.with_name(f"{args.workdir.name}-seed{seed}")
        cfg = DrillConfig(
            workdir=workdir, provider=args.provider, minutes=args.minutes, seed=seed, llm_rca=args.llm_rca
        )
        result = await run_drill(cfg)
        results.append(result)
        found = sum(s.detected for s in result.scores)
        right = sum(s.rca_correct for s in result.scores)
        print(
            f"seed {seed}: {len(result.requests):,} requests in {result.wall_s:.0f} s · "
            f"detected {found}/{len(result.scores)} · root cause {right}/{len(result.scores)} · "
            f"false alarms {len(result.false_alarms)}"
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(results))
    print(f"report → {args.out}\nArgus data → {args.workdir / 'argus.db'} (mission-control dashboard)")


def _dashboard(args: argparse.Namespace) -> None:
    """Argus API on the drill's database + the Argus Streamlit UI, logged in as one app's tenant."""
    import argus

    from mission_control.apps import project_root
    from mission_control.drill import MISSION_ROOT

    keys = json.loads((args.workdir / "dashboard_keys.json").read_text())
    root = project_root(argus)
    env = {
        **os.environ,
        "ARGUS_DB_PATH": str((args.workdir / "argus.db").resolve()),
        "ARGUS_POLICIES": str(MISSION_ROOT / "config" / "policies.yaml"),
        "ARGUS_PRICING": str(root / "config" / "pricing.yaml"),
        "ARGUS_API_URL": f"http://localhost:{args.api_port}",
        "ARGUS_API_KEY": keys[args.app],
    }
    api = subprocess.Popen([sys.executable, "-m", "argus.cli", "serve", "--port", str(args.api_port)], env=env)
    try:
        ui = [sys.executable, "-m", "streamlit", "run", str(root / "ui" / "app.py"), "--server.port", str(args.port)]
        print(f"Argus API :{args.api_port} · dashboard http://localhost:{args.port} (tenant: {args.app})")
        subprocess.call(ui, env=env)
    finally:
        api.terminate()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="mission-control", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    d = sub.add_parser("drill", help="Run the three apps on one Argus with injected faults and score it")
    d.add_argument("--provider", choices=["offline", "nvidia"], default=os.getenv("MISSION_PROVIDER", "offline"))
    d.add_argument("--minutes", type=int, default=360, help="virtual duration of the replayed traffic")
    d.add_argument("--seeds", type=int, nargs="+", default=[7])
    d.add_argument("--llm-rca", action="store_true", help="investigate with Argus's LLM agent (NVIDIA NIM)")
    d.add_argument("--workdir", type=Path, default=Path(".mission"))
    d.add_argument("--out", type=Path, default=Path("reports/drill_report.md"))

    u = sub.add_parser("dashboard", help="Open the Argus dashboard on the drill's data")
    u.add_argument("--workdir", type=Path, default=Path(".mission"))
    u.add_argument("--app", choices=["atlas", "quill", "aegis"], default="aegis", help="tenant to view")
    u.add_argument("--port", type=int, default=8504)
    u.add_argument("--api-port", type=int, default=8013)

    args = parser.parse_args(argv)
    if args.command == "drill":
        _quiet()
        asyncio.run(_drill(args))
    else:
        _dashboard(args)


if __name__ == "__main__":
    main()
