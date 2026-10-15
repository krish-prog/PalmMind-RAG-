"""Interview booking: the LLM tool definition plus persistence."""

from __future__ import annotations

from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Booking
from app.schemas.booking import BookingCreate

BOOK_INTERVIEW_TOOL_NAME = "book_interview"

BOOK_INTERVIEW_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": BOOK_INTERVIEW_TOOL_NAME,
        "description": (
            "Book an interview. Call ONLY when the user explicitly wants to book and has "
            "provided all four fields: full name, email, date and time. Resolve relative "
            "dates (e.g. 'next Monday') using today's date from the system prompt."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Candidate's full name"},
                "email": {"type": "string", "description": "Candidate's email address"},
                "date": {"type": "string", "description": "Interview date, YYYY-MM-DD"},
                "time": {"type": "string", "description": "Interview time, 24h HH:MM"},
            },
            "required": ["name", "email", "date", "time"],
            "additionalProperties": False,
        },
    },
}


class BookingService:
    @staticmethod
    async def create(session: AsyncSession, data: BookingCreate, session_id: str) -> Booking:
        booking = Booking(
            name=data.name,
            email=str(data.email).lower(),
            interview_date=data.date,
            interview_time=data.time,
            session_id=session_id,
        )
        session.add(booking)
        await session.commit()
        await session.refresh(booking)
        return booking

    @staticmethod
    async def list_bookings(
        session: AsyncSession, limit: int, offset: int
    ) -> tuple[list[Booking], int]:
        total = await session.scalar(select(func.count()).select_from(Booking)) or 0
        rows = await session.scalars(
            select(Booking).order_by(Booking.created_at.desc()).limit(limit).offset(offset)
        )
        return list(rows), total
