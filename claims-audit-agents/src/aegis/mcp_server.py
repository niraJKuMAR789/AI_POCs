"""MCP server: lets an assistant (Claude Desktop, Cursor, an agent) audit claims, search policy and
work the review queue. Human decisions are a separate, non-read-only tool so hosts can gate them."""

from __future__ import annotations

from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp_types import ToolAnnotations

from aegis.config import Settings, get_settings
from aegis.models import Claim
from aegis.service import AuditService

READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)


def build_server(service: AuditService) -> MCPServer:
    server = MCPServer(
        name="aegis-claims-audit",
        title="Aegis claims audit",
        instructions="Audit medical claims against the plan's payment policy. Hard rule violations are final; "
        "soft findings are reviewed by specialist agents and, when uncertain, by a human.",
    )

    @server.tool(annotations=ToolAnnotations(read_only_hint=False, idempotent_hint=True, open_world_hint=False))
    async def audit_claim(claim: dict[str, Any]) -> dict[str, Any]:
        """Audit a claim (JSON matching the Claim schema). Returns the decision or the pending-review payload.
        Re-submitting the same claim_id returns the existing result."""
        result = await service.audit(Claim.model_validate(claim))
        return result.model_dump(mode="json")

    @server.tool(annotations=READ_ONLY)
    async def claim_status(claim_id: str) -> dict[str, Any]:
        """Current decision or pending-review state for a claim."""
        return (await service.status(claim_id)).model_dump(mode="json")

    @server.tool(annotations=READ_ONLY)
    def search_policy(query: str, k: int = 3) -> list[dict[str, str]]:
        """Search the payment policy manual. Returns citable sections (id, title, text)."""
        return [{"id": s.section_id, "title": s.title, "text": s.text} for s in service.manual.search(query, k)]

    @server.tool(annotations=READ_ONLY)
    def review_queue() -> list[str]:
        """Claim ids waiting for a human decision."""
        return service.ledger.pending_claims()

    @server.tool(annotations=READ_ONLY)
    def audit_trail(claim_id: str) -> dict[str, Any]:
        """Hash-chained audit events for a claim, with chain verification."""
        ok, bad = service.ledger.verify()
        return {"chain_verified": ok, "first_bad_seq": bad, "events": service.ledger.events(claim_id)}

    @server.tool(annotations=ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=False))
    async def submit_human_review(claim_id: str, reviewer: str, decisions: dict[str, str], note: str = "") -> dict:
        """Record a human reviewer's pay/deny decision for each pending line (keys are line numbers)."""
        result = await service.resume(
            claim_id, reviewer=reviewer, decisions={int(k): v for k, v in decisions.items()}, note=note
        )
        return result.model_dump(mode="json")

    return server


def main(http: bool = False, port: int = 8767, settings: Settings | None = None) -> None:
    import anyio

    async def run() -> None:
        service = await AuditService.create(settings or get_settings())
        server = build_server(service)
        try:
            if http:
                await server.run_streamable_http_async(port=port)
            else:
                await server.run_stdio_async()
        finally:
            await service.close()

    anyio.run(run)
