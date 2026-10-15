"""Request/response models for the ingestion API."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class ChunkingStrategy(StrEnum):
    """Available chunking strategies.

    - ``fixed_size``: sliding character window with overlap, snapped to word boundaries.
      Predictable chunk sizes; good for uniform, unstructured text.
    - ``sentence``: paragraph/sentence-aware packing with sentence-level overlap.
      Never cuts a sentence in half; better semantic coherence per chunk.
    """

    FIXED_SIZE = "fixed_size"
    SENTENCE = "sentence"


class DocumentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    content_type: str
    size_bytes: int
    char_count: int
    chunking_strategy: ChunkingStrategy
    chunk_size: int
    chunk_overlap: int
    num_chunks: int
    embedding_model: str
    created_at: datetime


class DocumentList(BaseModel):
    items: list[DocumentRead]
    total: int
