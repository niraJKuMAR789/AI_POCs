"""MCP server: ask an assistant "why is support-bot slow?" and let it use Argus.

The server is bound to one project by the API key it is launched with (ARGUS_MCP_API_KEY), with
that key's scopes. Content is never returned through MCP.
"""

from __future__ import annotations

import os
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp_types import ToolAnnotations

from argus.config import get_settings
from argus.investigate.tools import InvestigationTools
from argus.service import Argus

READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)


def build_server(argus: Argus, api_key: str) -> MCPServer:
    row = argus.store.resolve_key(api_key)
    if row is None:
        raise PermissionError("invalid ARGUS_MCP_API_KEY")
    project_id, scopes = row["project_id"], set(row["scopes"].split(","))
    project = argus.store.project(project_id)
    assert project is not None
    tools = InvestigationTools(argus.store, project_id)
    server = MCPServer(
        name="argus",
        title="Argus agent observability",
        instructions=f"Observability for project '{project['name']}': traces, incidents, governance.",
    )

    @server.tool(annotations=READ_ONLY)
    def list_incidents(status: str | None = None) -> list[dict[str, Any]]:
        """Detected incidents (open or resolved) with their detectors and severity."""
        return [i.model_dump() for i in argus.store.incidents(project_id, status)]

    @server.tool(annotations=READ_ONLY)
    def incident_overview(incident_id: int) -> dict[str, Any]:
        """Anomalies that make up an incident: metric, observed vs baseline, score."""
        return tools.incident_overview(incident_id)

    @server.tool(annotations=READ_ONLY)
    def component_breakdown(incident_id: int) -> list[dict[str, Any]]:
        """Which models/tools changed latency, errors or tokens during an incident."""
        return tools.component_breakdown(incident_id)

    @server.tool(annotations=READ_ONLY)
    def attribute_contrast(incident_id: int, outlier: str = "auto") -> dict[str, Any]:
        """Attribute values over-represented in bad traces during an incident."""
        return tools.attribute_contrast(incident_id, outlier)

    @server.tool(annotations=READ_ONLY)
    def recent_changes(incident_id: int) -> list[dict[str, Any]]:
        """Deploys and config changes before an incident."""
        return tools.recent_changes(incident_id)

    @server.tool(annotations=READ_ONLY)
    def get_trace(trace_id: str) -> list[dict[str, Any]]:
        """Span tree of a trace (metrics only; no prompt content)."""
        return tools.get_trace(trace_id)

    @server.tool(annotations=READ_ONLY)
    def governance_violations(limit: int = 50) -> list[dict[str, Any]]:
        """Recent policy violations: sensitive data, unapproved models, budgets, limits."""
        return [v.model_dump() for v in argus.store.violations(project_id, limit)]

    if "investigate" in scopes or "admin" in scopes:

        @server.tool(annotations=ToolAnnotations(read_only_hint=False, idempotent_hint=False, open_world_hint=False))
        async def investigate(incident_id: int) -> dict[str, Any]:
            """Run the root-cause investigator on an incident and store its report."""
            return await argus.investigate(project_id, incident_id)

    return server


def main() -> None:
    key = os.environ.get("ARGUS_MCP_API_KEY", "")
    build_server(Argus(get_settings()), key).run()
