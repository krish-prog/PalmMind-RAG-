"""Custom conversational RAG pipeline (no RetrievalQAChain / framework chains).

Per turn:
1. Load recent history for the session from Redis.
2. Rewrite the user's message into a standalone query using that history
   (so follow-ups like "what about its pricing?" retrieve correctly).
3. Embed the query and retrieve top-k chunks from Qdrant.
4. Ask the LLM to answer from the retrieved context, with a `book_interview` tool.
5. If the LLM calls the tool, validate the arguments (Pydantic), persist the booking,
   return the result to the LLM and let it write the confirmation. Invalid arguments
   are sent back as a tool error so the LLM asks the user to correct them.
6. Save the exchange to Redis.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import date

from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Booking
from app.schemas.booking import BookingCreate
from app.services.booking import BOOK_INTERVIEW_TOOL, BOOK_INTERVIEW_TOOL_NAME, BookingService
from app.services.embeddings import Embedder
from app.services.llm import LLMClient, Message, ToolCall
from app.services.memory import RedisChatMemory, StoredMessage
from app.services.vector_store import QdrantVectorStore, SearchHit

logger = logging.getLogger(__name__)

CONDENSE_PROMPT = (
    "Rewrite the user's latest message as a single standalone search query that can be "
    "understood without the conversation. Resolve pronouns and references using the "
    "history. Return ONLY the rewritten query, no preamble. If it is already standalone, "
    "return it unchanged."
)

ANSWER_PROMPT = """You are a helpful assistant for a document knowledge base.
Today's date is {today}.

Answer the user's question using ONLY the context below. Cite sources inline as [1], [2].
If the context does not contain the answer, say so plainly instead of guessing.

You can also book interviews. When the user wants to book one, collect their full name,
email, date and time. Ask only for what is still missing, then call `book_interview`.
Never invent booking details. Small talk and booking turns do not need the context.

Context:
{context}"""


@dataclass(frozen=True, slots=True)
class RAGResult:
    session_id: str
    answer: str
    standalone_query: str
    sources: list[SearchHit]
    booking: Booking | None


def format_context(hits: list[SearchHit]) -> str:
    if not hits:
        return "(no relevant context found)"
    return "\n\n".join(
        f"[{i}] (source: {h.filename}, chunk {h.chunk_index})\n{h.text}"
        for i, h in enumerate(hits, start=1)
    )


class RAGService:
    def __init__(
        self,
        *,
        llm: LLMClient,
        embedder: Embedder,
        vector_store: QdrantVectorStore,
        memory: RedisChatMemory,
        top_k: int,
        score_threshold: float,
    ) -> None:
        self._llm = llm
        self._embedder = embedder
        self._store = vector_store
        self._memory = memory
        self._top_k = top_k
        self._score_threshold = score_threshold

    async def _condense(self, history: list[StoredMessage], message: str) -> str:
        if not history:
            return message
        transcript = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:])
        response = await self._llm.chat(
            [
                {"role": "system", "content": CONDENSE_PROMPT},
                {
                    "role": "user",
                    "content": f"History:\n{transcript}\n\nLatest message: {message}",
                },
            ]
        )
        rewritten = (response.content or "").strip()
        return rewritten or message

    async def _retrieve(self, query: str, top_k: int) -> list[SearchHit]:
        vector = await self._embedder.embed_query(query)
        return await self._store.search(vector, top_k, self._score_threshold)

    async def _handle_booking(
        self, db: AsyncSession, call: ToolCall, session_id: str
    ) -> tuple[Booking | None, str]:
        """Validate and persist a booking. Returns (booking, tool-result JSON)."""
        try:
            data = BookingCreate.model_validate(call.arguments)
        except ValidationError as exc:
            errors = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
            return None, json.dumps({"status": "error", "errors": errors})
        booking = await BookingService.create(db, data, session_id)
        return booking, json.dumps(
            {
                "status": "confirmed",
                "booking_id": booking.id,
                "name": booking.name,
                "email": booking.email,
                "date": booking.interview_date.isoformat(),
                "time": booking.interview_time.strftime("%H:%M"),
            }
        )

    async def chat(
        self, db: AsyncSession, message: str, session_id: str | None, top_k: int | None = None
    ) -> RAGResult:
        session_id = session_id or uuid.uuid4().hex
        history = await self._memory.get_history(session_id)

        query = await self._condense(history, message)
        hits = await self._retrieve(query, top_k or self._top_k)

        messages: list[Message] = [
            {
                "role": "system",
                "content": ANSWER_PROMPT.format(
                    today=date.today().isoformat(), context=format_context(hits)
                ),
            },
            *({"role": m["role"], "content": m["content"]} for m in history),
            {"role": "user", "content": message},
        ]

        response = await self._llm.chat(messages, tools=[BOOK_INTERVIEW_TOOL])
        booking: Booking | None = None

        if response.tool_calls:
            messages.append(
                {
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                        }
                        for c in response.tool_calls
                    ],
                }
            )
            for call in response.tool_calls:
                if call.name == BOOK_INTERVIEW_TOOL_NAME and booking is None:
                    booking, result = await self._handle_booking(db, call, session_id)
                else:
                    result = json.dumps({"status": "error", "errors": "unknown or duplicate call"})
                messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
            response = await self._llm.chat(messages)

        answer = (response.content or "").strip() or "Sorry, I couldn't generate a response."
        await self._memory.append_exchange(session_id, message, answer)
        logger.info("chat session=%s hits=%d booking=%s", session_id, len(hits), bool(booking))

        return RAGResult(
            session_id=session_id,
            answer=answer,
            standalone_query=query,
            sources=hits,
            booking=booking,
        )
