"""Read-only access to bookings captured by the chat agent."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query

from app.api.deps import DbDep
from app.schemas.booking import BookingList, BookingRead
from app.services.booking import BookingService

router = APIRouter(prefix="/bookings", tags=["Bookings"])


@router.get("", response_model=BookingList, summary="List interview bookings")
async def list_bookings(
    db: DbDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> BookingList:
    items, total = await BookingService.list_bookings(db, limit, offset)
    return BookingList(items=[BookingRead.model_validate(b) for b in items], total=total)
