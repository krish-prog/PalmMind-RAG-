"""Embedding providers behind a small async protocol."""

from __future__ import annotations

import asyncio
from typing import Any, Protocol

from openai import AsyncOpenAI, OpenAIError

from app.core.exceptions import ExternalServiceError


class Embedder(Protocol):
    model_name: str
    dimension: int

    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...


class OpenAIEmbedder:
    """OpenAI (or compatible) embeddings API, batched."""

    def __init__(
        self,
        api_key: str,
        model_name: str,
        dimension: int,
        base_url: str | None = None,
        batch_size: int = 64,
    ) -> None:
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.dimension = dimension
        self._batch_size = batch_size

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        try:
            for i in range(0, len(texts), self._batch_size):
                batch = texts[i : i + self._batch_size]
                response = await self._client.embeddings.create(model=self.model_name, input=batch)
                vectors.extend(item.embedding for item in response.data)
        except OpenAIError as exc:
            raise ExternalServiceError(f"Embedding request failed: {exc}") from exc
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_documents([text]))[0]


class FastEmbedEmbedder:
    """Local CPU embeddings via `fastembed` (no API key needed). Install with `.[local]`."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", dimension: int = 384) -> None:
        from fastembed import TextEmbedding  # optional dependency

        self._model: Any = TextEmbedding(model_name=model_name)
        self.model_name = model_name
        self.dimension = dimension

    def _embed(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._model.embed(texts)]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self._embed, texts)

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_documents([text]))[0]
