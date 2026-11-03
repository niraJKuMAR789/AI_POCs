"""`argus` CLI."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from argus.config import get_settings


def _demo(minutes: int, seed: int, intensity: float) -> None:
    """Simulate three agent apps with injected incidents into the local Argus database, then detect."""
    from argus.service import Argus
    from argus.simulate.workload import Simulator

    argus = Argus(get_settings())
    sim = Simulator(total_min=minutes, seed=seed, intensity=intensity)
    projects = {}
    for name, payloads in sim.run().items():
        projects[name] = argus.store.create_project(name, org="acme")
        for payload in payloads:
            argus.ingestor.ingest(projects[name], payload)
    for name, events in sim.changes().items():
        for e in events:
            e.project_id = projects[name]
            argus.store.add_change(e)
    print(f"ingested {minutes} simulated minutes; new anomalies: {argus.detect_all()}")
    for name, pid in projects.items():
        key = argus.store.create_key(pid, "demo", ["ingest", "read", "read:content", "investigate"])
        print(f"  {name:15s} {pid}  api key: {key}")
        for inc in argus.store.incidents(pid):
            print(f"    incident #{inc.id} [{inc.severity}] {inc.title}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="argus", description="Observability and governance for agentic apps")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("demo", help="Load a simulated workload with injected incidents and run detection")
    p.add_argument("--minutes", type=int, default=720)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--intensity", type=float, default=1.0)
    p = sub.add_parser("serve", help="Run the API (collector + query)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8003)
    sub.add_parser("detect", help="Run anomaly detection for all projects")
    p = sub.add_parser("investigate", help="Investigate an incident")
    p.add_argument("project_id")
    p.add_argument("incident_id", type=int)
    p.add_argument("--llm", action="store_true", help="Use the NIM LLM investigator instead of deterministic RCA")
    p = sub.add_parser("create-key", help="Create a project and an API key")
    p.add_argument("project")
    p.add_argument("--org", default="default")
    p.add_argument("--scopes", default="ingest,read,investigate")
    p = sub.add_parser("retention", help="Apply retention policies (scrub old content, delete old spans)")
    p.add_argument("--now-ns", type=int)
    p = sub.add_parser("eval", help="Detection + RCA benchmark on simulated workloads")
    p.add_argument("--out", type=Path, default=Path("reports/eval_report.md"))
    p.add_argument("--quick", action="store_true")
    p.add_argument("--llm", action="store_true")
    p = sub.add_parser("mcp", help="MCP server (stdio); scoped by ARGUS_MCP_API_KEY")
    args = parser.parse_args(argv)

    if args.command == "demo":
        _demo(args.minutes, args.seed, args.intensity)
    elif args.command == "serve":
        import uvicorn

        uvicorn.run("argus.api.app:create_app", factory=True, host=args.host, port=args.port)
    elif args.command == "detect":
        from argus.service import Argus

        print(f"new anomalies: {Argus(get_settings()).detect_all()}")
    elif args.command == "investigate":
        from argus.service import Argus

        result = asyncio.run(Argus(get_settings()).investigate(args.project_id, args.incident_id, use_llm=args.llm))
        print(json.dumps(result, indent=2))
    elif args.command == "create-key":
        from argus.store.sqlite import Store

        settings = get_settings()
        settings.db_path.parent.mkdir(parents=True, exist_ok=True)
        store = Store(settings.db_path)
        pid = store.create_project(args.project, args.org)
        print(json.dumps({"project_id": pid, "api_key": store.create_key(pid, "cli", args.scopes.split(","))}))
    elif args.command == "retention":
        import time

        from argus.service import Argus

        argus = Argus(get_settings())
        now = args.now_ns or time.time_ns()
        for project in argus.store.projects():
            policy = argus.ingestor.policies.for_project(project["name"])
            print(
                project["name"],
                argus.store.apply_retention(project["id"], now, policy.retention_days, policy.content_retention_days),
            )
    elif args.command == "eval":
        from argus.evals.runner import main_sync

        main_sync(get_settings(), args.out, args.quick, args.llm)
    elif args.command == "mcp":
        from argus.mcp_server import main as mcp_main

        mcp_main()


if __name__ == "__main__":
    main()
