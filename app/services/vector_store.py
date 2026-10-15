"""Qdrant-backed vector store."""

from __future__ import annotations

from dataclasses import dataclass

from qdrant_client import AsyncQdrantClient, models


@dataclass(frozen=True, slots=True)
class VectorRecord:
    id: str
    vector: list[float]
    document_id: str
    filename: str
    chunk_index: int
    text: str


@dataclass(frozen=True, slots=True)
class SearchHit:
    id: str
    score: float
    document_id: str
    filename: str
    chunk_index: int
    text: str


class QdrantVectorStore:
    """Thin, typed wrapper around the async Qdrant client."""

    def __init__(self, client: AsyncQdrantClient, collection: str, dimension: int) -> None:
        self._client = client
        self._collection = collection
        self._dimension = dimension

    async def ensure_collection(self) -> None:
        if not await self._client.collection_exists(self._collection):
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=models.VectorParams(
                    size=self._dimension, distance=models.Distance.COSINE
                ),
            )
            await self._client.create_payload_index(
                collection_name=self._collection,
                field_name="document_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

    async def upsert(self, records: list[VectorRecord]) -> None:
        points = [
            models.PointStruct(
                id=r.id,
                vector=r.vector,
                payload={
                    "document_id": r.document_id,
                    "filename": r.filename,
                    "chunk_index": r.chunk_index,
                    "text": r.text,
                },
            )
            for r in records
        ]
        for i in range(0, len(points), 256):
            await self._client.upsert(collection_name=self._collection, points=points[i : i + 256])

    async def search(
        self, vector: list[float], top_k: int, score_threshold: float | None = None
    ) -> list[SearchHit]:
        response = await self._client.query_points(
            collection_name=self._collection,
            query=vector,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
        )
        hits: list[SearchHit] = []
        for point in response.points:
            payload = point.payload or {}
            hits.append(
                SearchHit(
                    id=str(point.id),
                    score=float(point.score),
                    document_id=str(payload.get("document_id", "")),
                    filename=str(payload.get("filename", "")),
                    chunk_index=int(payload.get("chunk_index", 0)),
                    text=str(payload.get("text", "")),
                )
            )
        return hits

    async def delete_document(self, document_id: str) -> None:
        await self._client.delete(
            collection_name=self._collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id", match=models.MatchValue(value=document_id)
                        )
                    ]
                )
            ),
        )

    async def close(self) -> None:
        await self._client.close()
