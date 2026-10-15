"""Request/response models for the conversational RAG API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.booking import BookingRead


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = Field(
        default=None,
        max_length=64,
        pattern=r"^[A-Za-z0-9_-]+$",
        description="Omit to start a new conversation; reuse the returned id for follow-ups.",
    )
    top_k: int | None = Field(default=None, ge=1, le=20)


class SourceChunk(BaseModel):
    document_id: str
    filename: str
    chunk_index: int
    score: float
    text: str


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    standalone_query: str
    sources: list[SourceChunk]
    booking: BookingRead | None = None


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatHistory(BaseModel):
    session_id: str
    messages: list[ChatMessage]
