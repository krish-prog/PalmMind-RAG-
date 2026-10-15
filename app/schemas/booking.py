"""Booking models. `BookingCreate` also validates the LLM's tool-call arguments."""

from __future__ import annotations

import datetime as dt

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator


class BookingCreate(BaseModel):
    """Validated interview booking request (produced by the LLM tool call)."""

    name: str = Field(min_length=2, max_length=255)
    email: EmailStr
    date: dt.date
    time: dt.time

    @field_validator("name")
    @classmethod
    def _strip_name(cls, value: str) -> str:
        return " ".join(value.split())

    @field_validator("date")
    @classmethod
    def _not_in_past(cls, value: dt.date) -> dt.date:
        if value < dt.date.today():
            raise ValueError("interview date cannot be in the past")
        return value


class BookingRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    email: str
    interview_date: dt.date
    interview_time: dt.time
    session_id: str
    status: str
    created_at: dt.datetime


class BookingList(BaseModel):
    items: list[BookingRead]
    total: int
