"""
Basic tests for document_processor.chunk_text.

Run with:
    pytest tests/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.document_processor import chunk_text  # noqa: E402


def test_chunk_text_basic():
    text = "word " * 1000  # long text, no punctuation
    chunks = chunk_text(text, source="test.txt", page=1, chunk_size=100, overlap=20)

    assert len(chunks) > 1
    for c in chunks:
        assert c.source == "test.txt"
        assert c.page == 1
        assert len(c.text) <= 100


def test_chunk_text_empty():
    chunks = chunk_text("", source="empty.txt", page=None)
    assert chunks == []


def test_chunk_text_short_text_single_chunk():
    chunks = chunk_text("A short sentence.", source="short.txt", page=None, chunk_size=800)
    assert len(chunks) == 1
    assert chunks[0].text == "A short sentence."


def test_chunk_ids_are_unique_for_different_pages():
    c1 = chunk_text("same text", source="a.txt", page=1)
    c2 = chunk_text("same text", source="a.txt", page=2)
    assert c1[0].chunk_id != c2[0].chunk_id
