"""Web-search fallback for CRAG (Tavily). Results are marked as untrusted `web` context."""

from __future__ import annotations

from typing import Protocol

import httpx

from atlas.models import RetrievedContext, stable_id


class WebSearch(Protocol):
    async def search(self, query: str, k: int = 4) -> list[RetrievedContext]: ...


class NullWebSearch:
    async def search(self, query: str, k: int = 4) -> list[RetrievedContext]:
        return []


class TavilySearch:
    def __init__(self, api_key: str, *, timeout: float = 15.0) -> None:
        self._api_key = api_key
        self._timeout = timeout

    async def search(self, query: str, k: int = 4) -> list[RetrievedContext]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                "https://api.tavily.com/search",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={"query": query, "max_results": k, "search_depth": "advanced"},
            )
            response.raise_for_status()
        return [
            RetrievedContext(
                id=f"web:{stable_id(r['url'])}",
                kind="web",
                text=r.get("content", ""),
                source=r["url"],
                title=r.get("title", r["url"]),
                score=float(r.get("score", 0.0)),
            )
            for r in response.json().get("results", [])
        ]
