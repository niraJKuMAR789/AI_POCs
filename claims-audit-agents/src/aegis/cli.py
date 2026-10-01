"""`aegis` CLI."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from aegis.config import get_settings


async def _audit(raw_json: str) -> None:
    from aegis.models import Claim
    from aegis.service import AuditService

    service = await AuditService.create(get_settings())
    try:
        raw = json.loads(raw_json)
        claim = Claim.model_validate(raw.get("claim", raw))
        result = await service.audit(claim)
        if result.decision:
            print(result.decision.summary)
            for review in result.decision.reviews:
                print(
                    f"  [{review.reviewer}] {review.rule_id} line {review.line_no}: {review.resolution.value} "
                    f"({review.confidence:.2f}) {review.rationale}"
                )
        else:
            print(json.dumps(result.pending.model_dump(mode="json") if result.pending else {}, indent=2))
    finally:
        await service.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="aegis", description="Multi-agent claims auditor")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("generate", help="Generate the synthetic registry and labeled claims")
    p.add_argument("--out", type=Path, default=Path("data"))
    p.add_argument("--per-scenario", type=int, default=8)
    p = sub.add_parser("audit", help="Audit a claim JSON file")
    p.add_argument("path", type=Path)
    p = sub.add_parser("eval", help="Benchmark rules-only vs rules+agents vs LLM-only")
    p.add_argument("--data", type=Path, default=Path("data/claims_labeled.jsonl"))
    p.add_argument("--out", type=Path, default=Path("reports/eval_report.md"))
    p.add_argument("--no-llm-only", action="store_true")
    p = sub.add_parser("serve", help="FastAPI server")
    p.add_argument("--port", type=int, default=8002)
    p.add_argument("--host", default="127.0.0.1")
    p = sub.add_parser("mcp", help="MCP server (stdio, or --http)")
    p.add_argument("--http", action="store_true")
    p.add_argument("--port", type=int, default=8767)
    args = parser.parse_args(argv)

    if args.command == "generate":
        from aegis.data.generate import write

        print(write(args.out, args.per_scenario))
    elif args.command == "audit":
        asyncio.run(_audit(args.path.read_text()))
    elif args.command == "eval":
        from aegis.evals.runner import run_eval

        asyncio.run(run_eval(get_settings(), args.data, args.out, include_llm_only=not args.no_llm_only))
    elif args.command == "serve":
        import uvicorn

        uvicorn.run("aegis.api.app:create_app", factory=True, host=args.host, port=args.port)
    elif args.command == "mcp":
        from aegis.mcp_server import main as mcp_main

        mcp_main(http=args.http, port=args.port)


if __name__ == "__main__":
    main()
