"""Application settings, loaded from environment variables / `.env`."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed, validated configuration for the whole service."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "PalmMind RAG API"
    log_level: str = "INFO"

    # --- SQL metadata store (SQLite by default, Postgres in docker-compose) ---
    database_url: str = "sqlite+aiosqlite:///./data/app.db"

    # --- Redis chat memory ---
    redis_url: str = "redis://localhost:6379/0"
    chat_history_max_turns: int = Field(default=10, ge=1)
    chat_session_ttl_seconds: int = Field(default=60 * 60 * 24, ge=60)

    # --- Qdrant vector store ---
    qdrant_url: str | None = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "documents"

    # --- Embeddings ---
    embedding_provider: Literal["openai", "fastembed"] = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    # --- LLM (any OpenAI-compatible endpoint: OpenAI, Groq, Ollama, ...) ---
    llm_api_key: str = ""
    llm_base_url: str | None = None
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.2

    openai_api_key: str = ""
    openai_base_url: str | None = None

    # --- Ingestion / retrieval ---
    max_upload_mb: int = Field(default=20, ge=1)
    default_chunk_size: int = Field(default=800, ge=100)
    default_chunk_overlap: int = Field(default=120, ge=0)
    retrieval_top_k: int = Field(default=5, ge=1, le=50)
    retrieval_score_threshold: float = Field(default=0.2, ge=0.0, le=1.0)


@lru_cache
def get_settings() -> Settings:
    """Return a cached settings instance."""
    return Settings()
