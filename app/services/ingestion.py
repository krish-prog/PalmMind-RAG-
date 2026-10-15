"""Ingestion pipeline: extract → chunk → embed → store vectors → save metadata."""

from __future__ import annotations

import logging
import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import EmptyDocumentError, NotFoundError
from app.db.models import Document, DocumentChunk
from app.schemas.documents import ChunkingStrategy
from app.services.chunking import get_chunker
from app.services.embeddings import Embedder
from app.services.extraction import extract_text
from app.services.vector_store import QdrantVectorStore, VectorRecord

logger = logging.getLogger(__name__)


class IngestionService:
    def __init__(self, embedder: Embedder, vector_store: QdrantVectorStore) -> None:
        self._embedder = embedder
        self._store = vector_store

    async def ingest(
        self,
        session: AsyncSession,
        *,
        filename: str,
        content_type: str,
        data: bytes,
        strategy: ChunkingStrategy,
        chunk_size: int,
        chunk_overlap: int,
    ) -> Document:
        text = await extract_text(filename, data)
        chunks = get_chunker(strategy, chunk_size, chunk_overlap).split(text)
        if not chunks:
            raise EmptyDocumentError("Document produced no chunks.")

        vectors = await self._embedder.embed_documents([c.text for c in chunks])
        document_id = str(uuid.uuid4())
        chunk_ids = [str(uuid.uuid4()) for _ in chunks]

        await self._store.upsert(
            [
                VectorRecord(
                    id=chunk_id,
                    vector=vector,
                    document_id=document_id,
                    filename=filename,
                    chunk_index=chunk.index,
                    text=chunk.text,
                )
                for chunk_id, chunk, vector in zip(chunk_ids, chunks, vectors, strict=True)
            ]
        )

        document = Document(
            id=document_id,
            filename=filename,
            content_type=content_type,
            size_bytes=len(data),
            char_count=len(text),
            chunking_strategy=strategy.value,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            num_chunks=len(chunks),
            embedding_model=self._embedder.model_name,
            chunks=[
                DocumentChunk(id=cid, chunk_index=c.index, char_count=len(c.text))
                for cid, c in zip(chunk_ids, chunks, strict=True)
            ],
        )
        session.add(document)
        try:
            await session.commit()
        except Exception:
            # keep the two stores consistent if the SQL write fails
            await session.rollback()
            await self._store.delete_document(document_id)
            raise

        logger.info("Ingested %s (%s chunks, strategy=%s)", filename, len(chunks), strategy)
        return document

    @staticmethod
    async def list_documents(
        session: AsyncSession, limit: int, offset: int
    ) -> tuple[list[Document], int]:
        total = await session.scalar(select(func.count()).select_from(Document)) or 0
        rows = await session.scalars(
            select(Document).order_by(Document.created_at.desc()).limit(limit).offset(offset)
        )
        return list(rows), total

    @staticmethod
    async def get_document(session: AsyncSession, document_id: str) -> Document:
        document = await session.get(Document, document_id)
        if document is None:
            raise NotFoundError(f"Document '{document_id}' not found")
        return document

    async def delete_document(self, session: AsyncSession, document_id: str) -> None:
        document = await self.get_document(session, document_id)
        await self._store.delete_document(document_id)
        await session.delete(document)
        await session.commit()
