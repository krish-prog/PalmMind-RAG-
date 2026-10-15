"""Composition root: builds and tears down all long-lived dependencies."""

from __future__ import annotations

from dataclasses import dataclass

from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.db.session import create_engine, create_session_factory, init_db
from app.services.embeddings import Embedder, FastEmbedEmbedder, OpenAIEmbedder
from app.services.ingestion import IngestionService
from app.services.llm import LLMClient, OpenAICompatibleLLM
from app.services.memory import RedisChatMemory
from app.services.rag import RAGService
from app.services.vector_store import QdrantVectorStore


@dataclass
class Container:
    settings: Settings
    engine: AsyncEngine
    session_factory: async_sessionmaker[AsyncSession]
    vector_store: QdrantVectorStore
    memory: RedisChatMemory
    ingestion: IngestionService
    rag: RAGService

    async def close(self) -> None:
        await self.memory.close()
        await self.vector_store.close()
        await self.engine.dispose()


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedding_provider == "fastembed":
        return FastEmbedEmbedder(settings.embedding_model, settings.embedding_dim)
    return OpenAIEmbedder(
        api_key=settings.openai_api_key or settings.llm_api_key,
        model_name=settings.embedding_model,
        dimension=settings.embedding_dim,
        base_url=settings.openai_base_url,
    )


def build_llm(settings: Settings) -> LLMClient:
    return OpenAICompatibleLLM(
        api_key=settings.llm_api_key or settings.openai_api_key,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        base_url=settings.llm_base_url,
    )


async def build_container(
    settings: Settings,
    *,
    embedder: Embedder | None = None,
    llm: LLMClient | None = None,
    redis: Redis | None = None,
    qdrant: AsyncQdrantClient | None = None,
) -> Container:
    """Wire everything together. Overrides let tests inject fakes."""
    engine = create_engine(settings.database_url)
    await init_db(engine)

    embedder = embedder or build_embedder(settings)
    qdrant = qdrant or (
        AsyncQdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
        if settings.qdrant_url
        else AsyncQdrantClient(location=":memory:")
    )
    vector_store = QdrantVectorStore(qdrant, settings.qdrant_collection, embedder.dimension)
    await vector_store.ensure_collection()

    memory = RedisChatMemory(
        redis or Redis.from_url(settings.redis_url, decode_responses=True),
        max_turns=settings.chat_history_max_turns,
        ttl_seconds=settings.chat_session_ttl_seconds,
    )

    return Container(
        settings=settings,
        engine=engine,
        session_factory=create_session_factory(engine),
        vector_store=vector_store,
        memory=memory,
        ingestion=IngestionService(embedder, vector_store),
        rag=RAGService(
            llm=llm or build_llm(settings),
            embedder=embedder,
            vector_store=vector_store,
            memory=memory,
            top_k=settings.retrieval_top_k,
            score_threshold=settings.retrieval_score_threshold,
        ),
    )
