"""Agno analyst agent: a conversational layer over the governed pipeline.

The agent plans multi-step analyses (decompose, query, compare, explain) using tools that all
route through `Text2SQL` and its guard, so its access is never wider than the caller's role.
Two tool transports:
  * in-process toolkit (default, lowest latency)
  * MCP: the agent talks to `quill mcp` over stdio, the same way an external client would
Writes (flagging claims) pause the run for human confirmation using Agno's `requires_confirmation`.
"""

from __future__ import annotations

import json
from typing import Any

from agno.agent import Agent
from agno.db.in_memory import InMemoryDb
from agno.guardrails import PromptInjectionGuardrail
from agno.models.base import Model
from agno.tools import Toolkit, tool

from quill.config import Settings
from quill.pipeline import Text2SQL

INSTRUCTIONS = [
    "You are Quill, a data analyst for a health-insurance claims warehouse.",
    "Answer data questions by calling `ask_data`; break multi-part questions into several focused calls.",
    "Use `describe_schema` when you need to know which tables and columns exist.",
    "Report numbers exactly as returned, and show the SQL you relied on in a collapsed `sql` block.",
    "If a tool says data is restricted for your role, explain that plainly; never try to work around it.",
    "To flag a claim for audit, call `flag_claim`. A human must approve it before it runs.",
]


class QuillTools(Toolkit):
    def __init__(self, pipeline: Text2SQL, role: str, context: dict[str, Any]) -> None:
        self.pipeline, self.role, self.context = pipeline, role, context
        tools: list[Any] = [self.ask_data, self.describe_schema]
        if "audit_flags" in pipeline.policy(role).write_tables:
            tools.append(self.flag_claim)
        super().__init__(name="quill", tools=tools)

    async def ask_data(self, question: str) -> str:
        """Answer one focused data question against the claims warehouse.

        Args:
            question: A single, specific analytical question in plain English.
        """
        result = await self.pipeline.ask(question, role=self.role, context=self.context)
        return json.dumps(
            {
                "status": result.status,
                "answer": result.answer,
                "sql": result.executed_sql,
                "columns": result.columns,
                "rows": result.rows[:25],
                "row_count": result.row_count,
                "truncated": result.truncated,
            },
            default=str,
        )

    def describe_schema(self, question: str = "") -> str:
        """Describe the tables and columns available to you, optionally only those relevant to a question.

        Args:
            question: Optional question used to select the relevant tables.
        """
        policy = self.pipeline.policy(self.role)
        if question:
            return self.pipeline.visible_schema(question, policy)[1]
        tables = [t for t in policy.tables if t in self.pipeline.layer.tables]
        return self.pipeline.layer.ddl(tables, hidden_columns=policy.denied)

    @tool(requires_confirmation=True)
    def flag_claim(self, claim_id: int, reason: str) -> str:
        """Flag a claim for audit review. Requires human approval.

        Args:
            claim_id: The claim to flag.
            reason: Why the claim should be reviewed.
        """
        safe_reason = reason.replace("'", "''")[:500]
        sql = (
            f"INSERT INTO audit_flags (claim_id, reason, flagged_by) "
            f"VALUES ({int(claim_id)}, '{safe_reason}', 'quill-agent:{self.role}')"
        )
        written = self.pipeline.confirm_write(sql, role=self.role, context=self.context)
        return f"Flagged claim {claim_id} ({written} row written)."


def default_model(settings: Settings) -> Model:
    from agno.models.nvidia import Nvidia

    key = settings.nvidia_api_key.get_secret_value() if settings.nvidia_api_key else None
    return Nvidia(id=settings.sql_model, api_key=key, base_url=settings.base_url, temperature=0.0)


def build_agent(
    pipeline: Text2SQL,
    *,
    role: str,
    context: dict[str, Any] | None = None,
    model: Model | None = None,
    tools: list[Any] | None = None,
) -> Agent:
    return Agent(
        name="Quill",
        model=model or default_model(pipeline.settings),
        tools=tools if tools is not None else [QuillTools(pipeline, role, context or {})],
        instructions=INSTRUCTIONS,
        additional_context=f"Your data role is '{role}': {pipeline.policy(role).description}",
        db=InMemoryDb(),
        add_history_to_context=True,
        num_history_runs=5,
        pre_hooks=[PromptInjectionGuardrail()],
        tool_call_limit=8,
        markdown=True,
    )


def mcp_tools(role: str, context: dict[str, Any] | None = None) -> Any:
    """MCP transport: spawn `quill mcp` bound to the role (use as `async with mcp_tools(...) as t:`)."""
    import os

    from agno.tools.mcp import MCPTools

    env = {**os.environ, "QUILL_MCP_ROLE": role, "QUILL_MCP_CONTEXT": json.dumps(context or {})}
    return MCPTools(command="quill mcp", env=env, include_tools=["ask", "describe_schema", "list_metrics"])
