"""Two selectable chunking strategies, behind a common interface."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from app.schemas.documents import ChunkingStrategy

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")


@dataclass(frozen=True, slots=True)
class Chunk:
    index: int
    text: str


class Chunker(Protocol):
    def split(self, text: str) -> list[Chunk]: ...


@dataclass(frozen=True, slots=True)
class FixedSizeChunker:
    """Sliding window of ~``chunk_size`` characters with ``chunk_overlap`` overlap.

    Window edges are moved back to the nearest whitespace so words are never split.
    """

    chunk_size: int
    chunk_overlap: int

    def __post_init__(self) -> None:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

    def split(self, text: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        start, length = 0, len(text)
        while start < length:
            end = min(start + self.chunk_size, length)
            if end < length:
                boundary = text.rfind(" ", start + self.chunk_size // 2, end)
                newline = text.rfind("\n", start + self.chunk_size // 2, end)
                end = max(boundary, newline) if max(boundary, newline) > start else end
            piece = text[start:end].strip()
            if piece:
                chunks.append(Chunk(index=len(chunks), text=piece))
            if end >= length:
                break
            next_start = end - self.chunk_overlap
            # snap overlap start forward to a word boundary
            space = text.find(" ", next_start, end)
            start = space + 1 if space != -1 else next_start
        return chunks


@dataclass(frozen=True, slots=True)
class SentenceChunker:
    """Pack whole sentences (respecting paragraphs) up to ``chunk_size`` characters.

    Consecutive chunks share trailing sentences totalling at most ``chunk_overlap``
    characters, so context carries across boundaries without cutting a sentence.
    """

    chunk_size: int
    chunk_overlap: int

    @staticmethod
    def _sentences(text: str) -> list[str]:
        sentences: list[str] = []
        for paragraph in re.split(r"\n\s*\n", text):
            flat = " ".join(paragraph.split())
            if flat:
                sentences.extend(s.strip() for s in _SENTENCE_SPLIT.split(flat) if s.strip())
        return sentences

    def _hard_wrap(self, sentence: str) -> list[str]:
        """Fallback for a single sentence longer than ``chunk_size``."""
        return [c.text for c in FixedSizeChunker(self.chunk_size, 0).split(sentence)]

    def split(self, text: str) -> list[Chunk]:
        units: list[str] = []
        for sentence in self._sentences(text):
            units.extend(
                self._hard_wrap(sentence) if len(sentence) > self.chunk_size else [sentence]
            )

        chunks: list[Chunk] = []
        current: list[str] = []
        size = 0
        for unit in units:
            if current and size + len(unit) + 1 > self.chunk_size:
                chunks.append(Chunk(index=len(chunks), text=" ".join(current)))
                overlap: list[str] = []
                overlap_size = 0
                for prev in reversed(current):
                    if overlap_size + len(prev) + 1 > self.chunk_overlap:
                        break
                    overlap.insert(0, prev)
                    overlap_size += len(prev) + 1
                current, size = overlap, overlap_size
            current.append(unit)
            size += len(unit) + 1
        if current:
            chunks.append(Chunk(index=len(chunks), text=" ".join(current)))
        return chunks


def get_chunker(strategy: ChunkingStrategy, chunk_size: int, chunk_overlap: int) -> Chunker:
    """Factory: return the chunker for the selected strategy."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    match strategy:
        case ChunkingStrategy.FIXED_SIZE:
            return FixedSizeChunker(chunk_size, chunk_overlap)
        case ChunkingStrategy.SENTENCE:
            return SentenceChunker(chunk_size, chunk_overlap)
