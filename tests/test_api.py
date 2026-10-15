from __future__ import annotations

import io
from datetime import date, timedelta

import pytest
from httpx import AsyncClient
from pypdf import PdfWriter

from app.services.llm import LLMResponse, ToolCall
from tests.fakes import ScriptedLLM

DOC = (
    "Palm Mind Technology is an AI company in Kathmandu. "
    "It builds conversational chatbots for businesses.\n\n"
    "The office opens at 9 AM and closes at 6 PM, Sunday to Friday. "
    "The company offers health insurance and flexible remote work."
)


async def upload(client: AsyncClient, strategy: str = "sentence", text: str = DOC) -> dict:
    resp = await client.post(
        "/api/v1/documents",
        files={"file": ("company.txt", text.encode(), "text/plain")},
        data={"chunking_strategy": strategy, "chunk_size": "120", "chunk_overlap": "20"},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


# ---------------------------------------------------------------- ingestion


@pytest.mark.parametrize("strategy", ["fixed_size", "sentence"])
async def test_upload_txt_both_strategies(client: AsyncClient, strategy: str) -> None:
    doc = await upload(client, strategy)
    assert doc["chunking_strategy"] == strategy
    assert doc["num_chunks"] >= 2

    listed = (await client.get("/api/v1/documents")).json()
    assert listed["total"] == 1
    assert (await client.get(f"/api/v1/documents/{doc['id']}")).status_code == 200


async def test_upload_rejects_unsupported_type(client: AsyncClient) -> None:
    resp = await client.post(
        "/api/v1/documents", files={"file": ("x.docx", b"hello", "application/octet-stream")}
    )
    assert resp.status_code == 415


async def test_upload_rejects_empty_and_bad_overlap(client: AsyncClient) -> None:
    empty = await client.post(
        "/api/v1/documents", files={"file": ("e.txt", b"   \n ", "text/plain")}
    )
    assert empty.status_code == 422
    bad = await client.post(
        "/api/v1/documents",
        files={"file": ("a.txt", DOC.encode(), "text/plain")},
        data={"chunk_size": "200", "chunk_overlap": "200"},
    )
    assert bad.status_code == 400


async def test_upload_blank_pdf_is_rejected(client: AsyncClient) -> None:
    buf = io.BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.write(buf)
    resp = await client.post(
        "/api/v1/documents", files={"file": ("blank.pdf", buf.getvalue(), "application/pdf")}
    )
    assert resp.status_code == 422


async def test_delete_document(client: AsyncClient) -> None:
    doc = await upload(client)
    assert (await client.delete(f"/api/v1/documents/{doc['id']}")).status_code == 204
    assert (await client.get(f"/api/v1/documents/{doc['id']}")).status_code == 404


# ---------------------------------------------------------------- RAG chat


async def test_chat_retrieves_relevant_context(client: AsyncClient, llm: ScriptedLLM) -> None:
    await upload(client)
    llm.responses = [LLMResponse(content="The office opens at 9 AM [1].")]

    resp = await client.post("/api/v1/chat", json={"message": "When does the office open?"})
    body = resp.json()
    assert resp.status_code == 200
    assert body["answer"] == "The office opens at 9 AM [1]."
    assert "9 AM" in body["sources"][0]["text"]
    # retrieved context was injected into the system prompt, booking tool offered
    system_prompt = llm.calls[0]["messages"][0]["content"]
    assert "9 AM" in system_prompt
    assert llm.calls[0]["tools"][0]["function"]["name"] == "book_interview"


async def test_multi_turn_memory_and_query_rewrite(client: AsyncClient, llm: ScriptedLLM) -> None:
    await upload(client)
    llm.responses = [
        LLMResponse(content="It builds chatbots."),
        LLMResponse(content="What employee benefits does the company offer?"),  # condense
        LLMResponse(content="Health insurance and remote work."),
    ]
    first = (await client.post("/api/v1/chat", json={"message": "What does Palm Mind do?"})).json()
    sid = first["session_id"]

    second = (
        await client.post(
            "/api/v1/chat", json={"message": "what benefits do they offer?", "session_id": sid}
        )
    ).json()
    assert second["standalone_query"] == "What employee benefits does the company offer?"
    assert "insurance" in second["sources"][0]["text"]

    # final answer call received prior turns from Redis
    answer_msgs = llm.calls[2]["messages"]
    assert {"role": "user", "content": "What does Palm Mind do?"} in answer_msgs
    assert {"role": "assistant", "content": "It builds chatbots."} in answer_msgs

    history = (await client.get(f"/api/v1/chat/{sid}/history")).json()
    assert len(history["messages"]) == 4

    assert (await client.delete(f"/api/v1/chat/{sid}")).status_code == 204
    assert (await client.get(f"/api/v1/chat/{sid}/history")).json()["messages"] == []


# ---------------------------------------------------------------- booking


async def test_booking_via_tool_call(client: AsyncClient, llm: ScriptedLLM) -> None:
    day = (date.today() + timedelta(days=3)).isoformat()
    llm.responses = [
        LLMResponse(
            content=None,
            tool_calls=[
                ToolCall(
                    id="call_1",
                    name="book_interview",
                    arguments={
                        "name": "Krishna Acharya",
                        "email": "KRISH@example.com",
                        "date": day,
                        "time": "14:30",
                    },
                )
            ],
        ),
        LLMResponse(content=f"Booked for {day} at 14:30."),
    ]
    resp = await client.post(
        "/api/v1/chat",
        json={"message": f"Book me: Krishna Acharya, krish@example.com, {day} 2:30pm"},
    )
    body = resp.json()
    assert body["booking"]["name"] == "Krishna Acharya"
    assert body["booking"]["email"] == "krish@example.com"
    assert body["booking"]["interview_time"] == "14:30:00"

    # tool result was fed back to the LLM
    tool_msg = llm.calls[1]["messages"][-1]
    assert tool_msg["role"] == "tool" and '"confirmed"' in tool_msg["content"]

    bookings = (await client.get("/api/v1/bookings")).json()
    assert bookings["total"] == 1


async def test_invalid_booking_is_reported_back_not_saved(
    client: AsyncClient, llm: ScriptedLLM
) -> None:
    llm.responses = [
        LLMResponse(
            content=None,
            tool_calls=[
                ToolCall(
                    id="c1",
                    name="book_interview",
                    arguments={
                        "name": "Krishna",
                        "email": "not-an-email",
                        "date": "2000-01-01",
                        "time": "10:00",
                    },
                )
            ],
        ),
        LLMResponse(content="That email looks invalid and the date is in the past."),
    ]
    body = (await client.post("/api/v1/chat", json={"message": "book it"})).json()
    assert body["booking"] is None
    tool_msg = llm.calls[1]["messages"][-1]
    assert '"error"' in tool_msg["content"] and "email" in tool_msg["content"]
    assert (await client.get("/api/v1/bookings")).json()["total"] == 0


async def test_chat_validation(client: AsyncClient) -> None:
    assert (await client.post("/api/v1/chat", json={"message": ""})).status_code == 422
    bad_sid = await client.post("/api/v1/chat", json={"message": "hi", "session_id": "a b"})
    assert bad_sid.status_code == 422
