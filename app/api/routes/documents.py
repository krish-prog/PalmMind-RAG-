"""Document Ingestion API."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, Form, Query, UploadFile, status

from app.api.deps import ContainerDep, DbDep
from app.core.exceptions import AppError, FileTooLargeError
from app.schemas.documents import ChunkingStrategy, DocumentList, DocumentRead
from app.services.extraction import detect_extension

router = APIRouter(prefix="/documents", tags=["Document Ingestion"])


@router.post(
    "",
    response_model=DocumentRead,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a .pdf or .txt file and index it",
)
async def upload_document(
    container: ContainerDep,
    db: DbDep,
    file: Annotated[UploadFile, File(description=".pdf or .txt file")],
    chunking_strategy: Annotated[ChunkingStrategy, Form()] = ChunkingStrategy.SENTENCE,
    chunk_size: Annotated[int | None, Form(ge=100, le=8000)] = None,
    chunk_overlap: Annotated[int | None, Form(ge=0, le=2000)] = None,
) -> DocumentRead:
    settings = container.settings
    filename = file.filename or "upload"
    detect_extension(filename)  # fail fast before reading the body

    max_bytes = settings.max_upload_mb * 1024 * 1024
    data = await file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise FileTooLargeError(f"File exceeds {settings.max_upload_mb} MB limit")

    size = chunk_size or settings.default_chunk_size
    overlap = settings.default_chunk_overlap if chunk_overlap is None else chunk_overlap
    if overlap >= size:
        raise AppError("chunk_overlap must be smaller than chunk_size")

    document = await container.ingestion.ingest(
        db,
        filename=filename,
        content_type=file.content_type or "application/octet-stream",
        data=data,
        strategy=chunking_strategy,
        chunk_size=size,
        chunk_overlap=overlap,
    )
    return DocumentRead.model_validate(document)


@router.get("", response_model=DocumentList, summary="List ingested documents")
async def list_documents(
    container: ContainerDep,
    db: DbDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> DocumentList:
    items, total = await container.ingestion.list_documents(db, limit, offset)
    return DocumentList(items=[DocumentRead.model_validate(d) for d in items], total=total)


@router.get("/{document_id}", response_model=DocumentRead, summary="Get document metadata")
async def get_document(container: ContainerDep, db: DbDep, document_id: str) -> DocumentRead:
    return DocumentRead.model_validate(await container.ingestion.get_document(db, document_id))


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document and its vectors",
)
async def delete_document(container: ContainerDep, db: DbDep, document_id: str) -> None:
    await container.ingestion.delete_document(db, document_id)
