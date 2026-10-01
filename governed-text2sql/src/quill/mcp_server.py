"""MCP server exposing the governed warehouse.

Each server process is bound to ONE role (and its tenant context) by its launcher via
environment variables. MCP clients never see credentials and cannot choose their own
privileges: every query, including raw SQL, passes the same AST guard.

    QUILL_MCP_ROLE=analyst quill mcp
    QUILL_MCP_ROLE=provider_portal QUILL_MCP_CONTEXT='{"provider_id": 7}' quill mcp --http
"""

from __future__ import annotations

import json
import os
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp_types import ToolAnnotations

from quill.config import Settings, get_settings
from quill.pipeline import AskResult, Text2SQL

READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)


def _compact(result: AskResult, max_rows: int = 50) -> dict[str, Any]:
    payload = result.model_dump(
        include={
            "status",
            "answer",
            "executed_sql",
            "columns",
            "row_count",
            "truncated",
            "guard_rewrites",
            "plan_warnings",
        }
    )
    payload["rows"] = result.rows[:max_rows]
    payload["attempts"] = [f"{a.outcome}: {a.feedback}" if a.feedback else a.outcome for a in result.attempts]
    return payload


def build_server(
    settings: Settings | None = None,
    *,
    pipeline: Text2SQL | None = None,
    role: str | None = None,
    context: dict[str, Any] | None = None,
) -> MCPServer:
    settings = settings or get_settings()
    role = role or os.getenv("QUILL_MCP_ROLE") or settings.default_role
    bound_role: str = role
    context = context if context is not None else json.loads(os.getenv("QUILL_MCP_CONTEXT", "{}"))
    t2s = pipeline or Text2SQL.from_settings(settings)
    policy = t2s.policy(bound_role)

    server = MCPServer(
        name="quill-sql",
        title="Quill governed SQL",
        instructions=(
            f"Governed access to a health-claims warehouse as role '{role}': {policy.description} "
            "Use `ask` for natural-language questions. Use `describe_schema` before writing SQL for `run_sql`. "
            "Restricted columns are hidden and every query is checked against the role policy."
        ),
    )

    @server.tool(annotations=READ_ONLY)
    async def ask(question: str) -> dict[str, Any]:
        """Answer a business question: generates SQL, enforces the role policy, executes it and explains the
        result. Returns the executed SQL, rows and a short answer."""
        return _compact(await t2s.ask(question, role=bound_role, context=context))

    @server.tool(annotations=READ_ONLY)
    def describe_schema(question: str | None = None) -> str:
        """Annotated schema visible to this role. Pass a question to get only the relevant tables."""
        if question:
            return t2s.visible_schema(question, policy)[1]
        tables = [t for t in policy.tables if t in t2s.layer.tables]
        return t2s.layer.ddl(tables, hidden_columns=policy.denied)

    @server.tool(annotations=READ_ONLY)
    def list_metrics() -> dict[str, dict[str, str]]:
        """Certified business metric definitions (use these SQL expressions for consistency)."""
        return {name: {"description": m.description, "sql": m.sql} for name, m in t2s.layer.metrics.items()}

    @server.tool(annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False, idempotent_hint=True))
    def run_sql(sql: str) -> dict[str, Any]:
        """Execute a SQL query under this role's policy. Disallowed tables or columns are rejected with an
        explanation; tenant filters and row limits are applied automatically. Writes are not executed here."""
        return _compact(t2s.run_sql(sql, role=bound_role, context=context))

    @server.tool(annotations=READ_ONLY)
    def whoami() -> dict[str, Any]:
        """The role, tenant context and limits this server enforces."""
        return {
            "role": role,
            "context": context,
            "tables": policy.tables,
            "max_rows": policy.max_rows,
            "write_tables": policy.write_tables,
        }

    return server


def main(http: bool = False, host: str = "127.0.0.1", port: int = 8766) -> None:
    server = build_server()
    if http:
        server.run("streamable-http", host=host, port=port)
    else:
        server.run()
