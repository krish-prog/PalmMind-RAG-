"""Conversational RAG API."""

from __future__ import annotations

from fastapi import APIRouter, status

from app.api.deps import ContainerDep, DbDep
from app.schemas.booking import BookingRead
from app.schemas.chat import ChatHistory, ChatMessage, ChatRequest, ChatResponse, SourceChunk

router = APIRouter(prefix="/chat", tags=["Conversational RAG"])


@router.post("", response_model=ChatResponse, summary="Ask a question or book an interview")
async def chat(container: ContainerDep, db: DbDep, payload: ChatRequest) -> ChatResponse:
    result = await container.rag.chat(db, payload.message, payload.session_id, payload.top_k)
    return ChatResponse(
        session_id=result.session_id,
        answer=result.answer,
        standalone_query=result.standalone_query,
        sources=[
            SourceChunk(
                document_id=h.document_id,
                filename=h.filename,
                chunk_index=h.chunk_index,
                score=round(h.score, 4),
                text=h.text,
            )
            for h in result.sources
        ],
        booking=BookingRead.model_validate(result.booking) if result.booking else None,
    )


@router.get("/{session_id}/history", response_model=ChatHistory, summary="Get session history")
async def get_history(container: ContainerDep, session_id: str) -> ChatHistory:
    messages = await container.memory.get_history(session_id)
    return ChatHistory(
        session_id=session_id,
        messages=[ChatMessage.model_validate(m) for m in messages],
    )


@router.delete(
    "/{session_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Clear session memory"
)
async def clear_history(container: ContainerDep, session_id: str) -> None:
    await container.memory.clear(session_id)
