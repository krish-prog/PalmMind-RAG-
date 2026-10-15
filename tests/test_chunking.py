import pytest

from app.schemas.documents import ChunkingStrategy
from app.services.chunking import FixedSizeChunker, SentenceChunker, get_chunker

TEXT = (
    "Palm Mind builds AI products. It is based in Kathmandu. "
    "The company works on chatbots and RAG systems.\n\n"
    "Interviews are held on weekdays. Candidates should bring a portfolio. "
) * 10


def test_fixed_size_respects_limit_and_overlaps() -> None:
    chunks = FixedSizeChunker(chunk_size=200, chunk_overlap=40).split(TEXT)
    assert len(chunks) > 3
    assert all(len(c.text) <= 200 for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    # consecutive chunks share text
    assert chunks[0].text[-20:].split()[-1] in chunks[1].text


def test_fixed_size_never_splits_words() -> None:
    words = set(TEXT.split())
    for chunk in FixedSizeChunker(chunk_size=150, chunk_overlap=30).split(TEXT):
        for w in chunk.text.split():
            assert w in words


def test_sentence_chunker_keeps_whole_sentences() -> None:
    chunks = SentenceChunker(chunk_size=200, chunk_overlap=60).split(TEXT)
    assert len(chunks) > 3
    for c in chunks:
        assert c.text.endswith((".", "!", "?"))
        assert len(c.text) <= 200


def test_sentence_chunker_hard_wraps_giant_sentence() -> None:
    giant = "word " * 500
    chunks = SentenceChunker(chunk_size=200, chunk_overlap=20).split(giant)
    assert all(len(c.text) <= 200 for c in chunks)


def test_factory_and_validation() -> None:
    assert isinstance(get_chunker(ChunkingStrategy.FIXED_SIZE, 300, 50), FixedSizeChunker)
    assert isinstance(get_chunker(ChunkingStrategy.SENTENCE, 300, 50), SentenceChunker)
    with pytest.raises(ValueError):
        get_chunker(ChunkingStrategy.SENTENCE, 100, 100)
