"""Text extraction from uploaded files (.pdf, .txt)."""

from __future__ import annotations

import asyncio
import io
import re
from pathlib import PurePath

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.core.exceptions import EmptyDocumentError, UnsupportedFileTypeError

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".txt"})


def detect_extension(filename: str) -> str:
    """Return the lower-cased extension, or raise if unsupported."""
    ext = PurePath(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{ext or 'none'}'. Allowed: {sorted(SUPPORTED_EXTENSIONS)}"
        )
    return ext


def _normalize(text: str) -> str:
    """Collapse noisy whitespace while keeping paragraph breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_pdf(data: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
    except (PdfReadError, ValueError) as exc:
        raise EmptyDocumentError(f"Could not read PDF: {exc}") from exc
    return "\n\n".join(pages)


def _extract_txt(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


async def extract_text(filename: str, data: bytes) -> str:
    """Extract normalized text from a supported file. PDF parsing runs in a worker thread."""
    ext = detect_extension(filename)
    raw = await asyncio.to_thread(_extract_pdf, data) if ext == ".pdf" else _extract_txt(data)
    text = _normalize(raw)
    if not text:
        raise EmptyDocumentError(
            "No extractable text found (scanned/image-only PDFs need OCR, which is out of scope)."
        )
    return text
