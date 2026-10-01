"""`quill` CLI."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from quill.config import get_settings


def _init_db(args: argparse.Namespace) -> None:
    from quill.data.questions import build_examples, write_splits
    from quill.db.generate import generate

    settings = get_settings()
    db = generate(settings.db_path, claims=args.claims)
    counts = write_splits(build_examples(db, per_template=args.per_template), settings.examples_path.parent)
    print(f"database: {db}\nsplits: {counts}")


async def _ask(args: argparse.Namespace) -> None:
    from quill.pipeline import Text2SQL

    pipeline = Text2SQL.from_settings(get_settings())
    result = await pipeline.ask(args.question, role=args.role, context=json.loads(args.context))
    print(f"status: {result.status}")
    for i, attempt in enumerate(result.attempts, 1):
        print(f"attempt {i}: {attempt.outcome} {attempt.feedback}")
    if result.executed_sql:
        print(f"\nSQL:\n{result.executed_sql}\n")
    if result.columns:
        from quill.db.engine import QueryResult

        print(QueryResult(result.columns, [tuple(r) for r in result.rows], 0, result.truncated).to_markdown())
    print(f"\n{result.answer}")


async def _chat(args: argparse.Namespace) -> None:
    from quill.agent import build_agent, mcp_tools
    from quill.pipeline import Text2SQL

    pipeline = Text2SQL.from_settings(get_settings())
    context = json.loads(args.context)

    async def loop(agent: Any) -> None:
        while True:
            try:
                message = (await asyncio.to_thread(input, "\nyou › ")).strip()
            except EOFError:
                return
            if message in {"exit", "quit"}:
                return
            run = await agent.arun(message, session_id="cli")
            while run.is_paused:
                for req in run.active_requirements:
                    if req.needs_confirmation:
                        answer = await asyncio.to_thread(
                            input, f"approve {req.tool_execution.tool_name}({req.tool_execution.tool_args})? [y/N] "
                        )
                        req.confirm() if answer.lower() == "y" else req.reject()
                run = await agent.acontinue_run(run_response=run, session_id="cli")
            print(f"\nquill › {run.content}")

    if args.mcp:
        async with mcp_tools(args.role, context) as tools:
            await loop(build_agent(pipeline, role=args.role, context=context, tools=[tools]))
    else:
        await loop(build_agent(pipeline, role=args.role, context=context))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="quill", description="Governed text-to-SQL agent")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init-db", help="Generate the synthetic warehouse and the NL→SQL splits")
    p.add_argument("--claims", type=int, default=25000)
    p.add_argument("--per-template", type=int, default=40)

    for name, help_text in (
        ("ask", "One-shot question through the governed pipeline"),
        ("chat", "Conversational Agno agent"),
    ):
        p = sub.add_parser(name, help=help_text)
        if name == "ask":
            p.add_argument("question")
        else:
            p.add_argument("--mcp", action="store_true", help="Use tools over MCP instead of in-process")
        p.add_argument("--role", default="analyst")
        p.add_argument("--context", default="{}", help="JSON tenant context, e.g. '{\"provider_id\": 7}'")

    p = sub.add_parser("eval", help="Execution accuracy + red-team report")
    p.add_argument("--split", type=Path, default=Path("data/splits/test.jsonl"))
    p.add_argument("--out", type=Path, default=Path("reports/eval_report.md"))
    p.add_argument("--limit", type=int)

    p = sub.add_parser("serve", help="FastAPI server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8001)

    p = sub.add_parser("mcp", help="MCP server bound to QUILL_MCP_ROLE")
    p.add_argument("--http", action="store_true")
    p.add_argument("--port", type=int, default=8766)

    args = parser.parse_args(argv)
    if args.command == "init-db":
        _init_db(args)
    elif args.command == "ask":
        asyncio.run(_ask(args))
    elif args.command == "chat":
        asyncio.run(_chat(args))
    elif args.command == "eval":
        from quill.evals.runner import run_eval

        asyncio.run(run_eval(get_settings(), args.split, args.out, args.limit))
    elif args.command == "serve":
        import uvicorn

        uvicorn.run("quill.api.app:create_app", factory=True, host=args.host, port=args.port)
    elif args.command == "mcp":
        from quill.mcp_server import main as mcp_main

        mcp_main(http=args.http, port=args.port)


if __name__ == "__main__":
    main()
