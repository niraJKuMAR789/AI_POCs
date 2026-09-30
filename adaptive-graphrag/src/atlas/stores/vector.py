"""Qdrant vector store (embedded on-disk mode locally, server mode in Docker/K8s)."""

from __future__ import annotations

import uuid
import warnings
from typing import Any

from qdrant_client import AsyncQdrantClient, models

_NAMESPACE = uuid.UUID("7d1f3c1e-5b8a-4a51-9d7e-2f6d0e8b4c11")


class QdrantVectorStore:
    def __init__(self, client: AsyncQdrantClient, collection: str) -> None:
        self._client = client
        self.collection = collection
        self._ready = False

    async def _ensure(self, dimension: int) -> None:
        if self._ready:
            return
        if not await self._client.collection_exists(self.collection):
            await self._client.create_collection(
                self.collection,
                vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
            )
            with warnings.catch_warnings():  # embedded Qdrant ignores payload indexes; server mode uses them
                warnings.simplefilter("ignore")
                await self._client.create_payload_index(
                    self.collection, field_name="doc_id", field_schema=models.PayloadSchemaType.KEYWORD
                )
        self._ready = True

    async def upsert(self, ids: list[str], vectors: list[list[float]], payloads: list[dict[str, Any]]) -> None:
        if not ids:
            return
        await self._ensure(len(vectors[0]))
        points = [
            models.PointStruct(id=str(uuid.uuid5(_NAMESPACE, i)), vector=v, payload={**p, "key": i})
            for i, v, p in zip(ids, vectors, payloads, strict=True)
        ]
        await self._client.upsert(self.collection, points=points, wait=True)

    async def delete_where(self, field: str, value: str) -> None:
        if not await self._client.collection_exists(self.collection):
            return
        await self._client.delete(
            self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(must=[models.FieldCondition(key=field, match=models.MatchValue(value=value))])
            ),
        )

    async def clear(self) -> None:
        if await self._client.collection_exists(self.collection):
            await self._client.delete_collection(self.collection)
        self._ready = False

    async def search(self, vector: list[float], k: int) -> list[tuple[str, float, dict[str, Any]]]:
        if not await self._client.collection_exists(self.collection):
            return []
        response = await self._client.query_points(self.collection, query=vector, limit=k, with_payload=True)
        return [(str(p.payload["key"]), float(p.score), dict(p.payload)) for p in response.points if p.payload]
