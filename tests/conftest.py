from __future__ import annotations

from collections.abc import AsyncIterator

import fakeredis
import pytest
from httpx import ASGITransport, AsyncClient
from qdrant_client import AsyncQdrantClient

from app.container import Container, build_container
from app.core.config import Settings
from app.main import create_app
from tests.fakes import HashEmbedder, ScriptedLLM


@pytest.fixture
def llm() -> ScriptedLLM:
    return ScriptedLLM()


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        database_url=f"sqlite+aiosqlite:///{tmp_path}/test.db",
        qdrant_url=None,
        retrieval_score_threshold=0.0,
    )


@pytest.fixture
async def client(settings: Settings, llm: ScriptedLLM) -> AsyncIterator[AsyncClient]:
    async def factory(s: Settings) -> Container:
        return await build_container(
            s,
            embedder=HashEmbedder(),
            llm=llm,
            redis=fakeredis.FakeAsyncRedis(decode_responses=True),
            qdrant=AsyncQdrantClient(location=":memory:"),
        )

    app = create_app(settings, factory)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac
