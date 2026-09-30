"""`atlas` command-line interface."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from atlas.config import get_settings
from atlas.logging import configure_logging


async def _ingest(path: Path, force: bool) -> None:
    from atlas.ingest.loaders import iter_documents
    from atlas.kb import KnowledgeBase

    kb = await KnowledgeBase.open(get_settings())
    try:
        report = await kb.ingest(iter_documents(path), force=force)
        print(json.dumps(report.model_dump(), indent=2))
        print(json.dumps(kb.stats(), indent=2))
    finally:
        await kb.close()


async def _ask(question: str, show_trace: bool) -> None:
    from atlas.agent import AtlasAgent
    from atlas.kb import KnowledgeBase

    kb = await KnowledgeBase.open(get_settings())
    try:
        agent = AtlasAgent(kb)
        async for event in agent.astream(question):
            if event["type"] == "token":
                sys.stdout.write(event["text"])
                sys.stdout.flush()
            elif event["type"] == "reset":
                sys.stdout.write("\n[regenerating after failed grounding check]\n")
            elif event["type"] == "step" and show_trace:
                sys.stderr.write(f"  · {event['node']:<10} {event['summary']} ({event['latency_ms']} ms)\n")
            elif event["type"] == "final":
                result = event["result"]
                print(f"\n\nroute={result['route']} grounded={result['grounded']}")
                for c in result["citations"]:
                    print(f"  [{c['label']}] {c['title']} ({c['source']})")
    finally:
        await kb.close()


async def _eval(dataset: Path, out: Path) -> None:
    from atlas.evals.runner import run_eval

    await run_eval(dataset, out)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="atlas", description="Adaptive GraphRAG research agent")
    parser.add_argument("--log-level", default="WARNING")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="Ingest a file or directory (md, txt, pdf)")
    p.add_argument("path", type=Path)
    p.add_argument("--force", action="store_true", help="Re-ingest unchanged documents")

    p = sub.add_parser("ask", help="Ask a question (streams the answer)")
    p.add_argument("question")
    p.add_argument("--trace", action="store_true", help="Print agent steps to stderr")

    p = sub.add_parser("eval", help="Run the evaluation suite and write a Markdown report")
    p.add_argument("--dataset", type=Path, default=Path("evals/golden.jsonl"))
    p.add_argument("--out", type=Path, default=Path("reports/eval_report.md"))

    p = sub.add_parser("serve", help="Run the FastAPI server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)

    p = sub.add_parser("mcp", help="Run the MCP server (stdio by default)")
    p.add_argument("--http", action="store_true", help="Serve Streamable HTTP instead of stdio")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)

    args = parser.parse_args(argv)
    configure_logging(args.log_level, json=args.command in {"serve", "mcp"})

    if args.command == "ingest":
        asyncio.run(_ingest(args.path, args.force))
    elif args.command == "ask":
        asyncio.run(_ask(args.question, args.trace))
    elif args.command == "eval":
        asyncio.run(_eval(args.dataset, args.out))
    elif args.command == "serve":
        import uvicorn

        uvicorn.run("atlas.api.app:create_app", factory=True, host=args.host, port=args.port)
    elif args.command == "mcp":
        from atlas.mcp_server import main as mcp_main

        mcp_main(http=args.http, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
