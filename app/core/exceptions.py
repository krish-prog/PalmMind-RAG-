"""Domain exceptions, mapped to HTTP responses in `app.main`."""

from __future__ import annotations


class AppError(Exception):
    """Base class for expected, user-facing errors."""

    status_code: int = 400

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class UnsupportedFileTypeError(AppError):
    status_code = 415


class FileTooLargeError(AppError):
    status_code = 413


class EmptyDocumentError(AppError):
    status_code = 422


class NotFoundError(AppError):
    status_code = 404


class ExternalServiceError(AppError):
    status_code = 502
