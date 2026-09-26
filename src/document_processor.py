"""
document_processor.py

Loads raw course materials (PDF, TXT, MD) from a folder and splits them
into overlapping text chunks suitable for embedding + retrieval.

Each chunk keeps metadata about where it came from (file name, page
number where applicable) so answers can later be cited back to a source.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from pypdf import PdfReader


@dataclass
class Chunk:
    """A single unit of text ready to be embedded, plus its provenance."""

    text: str
    source: str                 # file name the chunk came from
    page: int | None = None     # page number (PDFs only), else None
    chunk_id: str = field(default="")

    def __post_init__(self) -> None:
        if not self.chunk_id:
            page_part = f"p{self.page}" if self.page is not None else "p0"
            self.chunk_id = f"{self.source}::{page_part}::{hash(self.text) & 0xFFFFFFFF}"


def _read_pdf(path: Path) -> List[tuple[str, int]]:
    """Return a list of (page_text, page_number) tuples for a PDF file."""
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((text, i))
    return pages


def _read_text_file(path: Path) -> List[tuple[str, int]]:
    """Return the whole file as a single 'page' (page number None)."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [(text, None)]


def load_documents(data_dir: str | Path) -> List[tuple[str, str, int | None]]:
    """
    Walk `data_dir` and return a list of (source_filename, page_text, page_number)
    tuples for every supported file found (.pdf, .txt, .md).
    """
    data_dir = Path(data_dir)
    results: List[tuple[str, str, int | None]] = []

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        try:
            if suffix == ".pdf":
                for text, page_num in _read_pdf(path):
                    results.append((path.name, text, page_num))
            elif suffix in (".txt", ".md"):
                for text, page_num in _read_text_file(path):
                    results.append((path.name, text, page_num))
            # silently skip unsupported file types
        except Exception as exc:  # noqa: BLE001 - surface but don't crash ingestion
            print(f"[document_processor] Skipping {path.name}: {exc}")

    return results


def chunk_text(
    text: str,
    source: str,
    page: int | None,
    chunk_size: int = 800,
    overlap: int = 150,
) -> List[Chunk]:
    """
    Split `text` into overlapping chunks of roughly `chunk_size` characters.

    Overlap helps preserve context across chunk boundaries so an answer
    that straddles two chunks isn't lost during retrieval.
    """
    text = " ".join(text.split())  # normalize whitespace
    if not text:
        return []

    chunks: List[Chunk] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk_str = text[start:end]
        chunks.append(Chunk(text=chunk_str, source=source, page=page))
        if end == n:
            break
        start = end - overlap  # step forward, keeping overlap

    return chunks


def process_directory(
    data_dir: str | Path,
    chunk_size: int = 800,
    overlap: int = 150,
) -> List[Chunk]:
    """
    Full pipeline: load every document in `data_dir` and return all chunks.
    This is the main entry point used by the ingestion step.
    """
    raw_pages = load_documents(data_dir)
    all_chunks: List[Chunk] = []

    for source, text, page in raw_pages:
        all_chunks.extend(chunk_text(text, source, page, chunk_size, overlap))

    return all_chunks


if __name__ == "__main__":
    # Quick manual smoke test: point at ./data and print a summary.
    here = Path(__file__).resolve().parent.parent
    chunks = process_directory(here / "data")
    print(f"Loaded {len(chunks)} chunks from {here / 'data'}")
    for c in chunks[:3]:
        print(f"- [{c.source} p.{c.page}] {c.text[:80]}...")
