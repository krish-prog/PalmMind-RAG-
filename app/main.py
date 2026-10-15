"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import bookings, chat, documents
from app.container import Container, build_container
from app.core.config import Settings, get_settings
from app.core.exceptions import AppError
from app.core.logging import configure_logging

ContainerFactory = Callable[[Settings], Awaitable[Container]]


def create_app(
    settings: Settings | None = None, container_factory: ContainerFactory | None = None
) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    factory = container_factory or build_container

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.container = await factory(settings)
        try:
            yield
        finally:
            await app.state.container.close()

    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        description="Document ingestion + custom conversational RAG with interview booking.",
        lifespan=lifespan,
    )

    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})

    @app.exception_handler(ValueError)
    async def _value_error(_: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.get("/health", tags=["Health"])
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(documents.router, prefix="/api/v1")
    app.include_router(chat.router, prefix="/api/v1")
    app.include_router(bookings.router, prefix="/api/v1")
    return app


app = create_app()
