"""Deterministic test doubles: no network, no API keys."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Callable
from typing import Any

from app.services.llm import LLMResponse, Message, ToolSpec

_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "is",
        "are",
        "it",
        "its",
        "in",
        "on",
        "at",
        "to",
        "of",
        "and",
        "or",
        "for",
        "do",
        "does",
        "what",
        "when",
        "who",
        "how",
        "they",
        "their",
        "this",
        "that",
        "with",
        "by",
        "be",
    ]
)


class HashEmbedder:
    """Bag-of-words hashing embedder: similar texts → similar vectors."""

    model_name = "test-hash"
    dimension = 256

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dimension
        for raw in re.findall(r"[a-z0-9]+", text.lower()):
            if raw in _STOPWORDS:
                continue
            token = raw.rstrip("s") or raw  # crude stemming: opens -> open
            bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dimension
            vec[bucket] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


Script = Callable[[list[Message], list[ToolSpec] | None], LLMResponse]


class ScriptedLLM:
    """Returns responses from a queue (or a callable) and records every call."""

    def __init__(self, responses: list[LLMResponse | Script] | None = None) -> None:
        self.responses: list[LLMResponse | Script] = list(responses or [])
        self.calls: list[dict[str, Any]] = []

    async def chat(
        self, messages: list[Message], tools: list[ToolSpec] | None = None
    ) -> LLMResponse:
        self.calls.append({"messages": messages, "tools": tools})
        if not self.responses:
            return LLMResponse(content="ok")
        nxt = self.responses.pop(0)
        return nxt(messages, tools) if callable(nxt) else nxt
