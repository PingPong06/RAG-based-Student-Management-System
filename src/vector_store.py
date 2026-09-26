"""
vector_store.py

Thin wrapper around ChromaDB that:
  - embeds text chunks locally (sentence-transformers, no API cost)
  - persists the vector DB to disk so you don't re-embed every run
  - exposes a simple add / query interface

Using a local embedding model keeps ingestion free and fast, and is a
common real-world pattern: embed locally, generate answers with a
frontier LLM.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import chromadb
from chromadb.utils import embedding_functions

from .document_processor import Chunk

DEFAULT_PERSIST_DIR = "chroma_db"
DEFAULT_COLLECTION = "student_notes"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # small, fast, good quality/speed tradeoff


class VectorStore:
    def __init__(
        self,
        persist_dir: str | Path = DEFAULT_PERSIST_DIR,
        collection_name: str = DEFAULT_COLLECTION,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        self.persist_dir = str(persist_dir)
        self.client = chromadb.PersistentClient(path=self.persist_dir)

        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: List[Chunk]) -> int:
        """Embed and store a list of Chunk objects. Returns count added."""
        if not chunks:
            return 0

        self.collection.upsert(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[
                {"source": c.source, "page": c.page if c.page is not None else -1}
                for c in chunks
            ],
        )
        return len(chunks)

    def query(self, question: str, top_k: int = 5) -> List[dict]:
        """
        Return the top_k most relevant chunks for `question` as a list of
        dicts: {text, source, page, distance}.
        """
        results = self.collection.query(query_texts=[question], n_results=top_k)

        hits = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for text, meta, dist in zip(docs, metas, dists):
            hits.append(
                {
                    "text": text,
                    "source": meta.get("source", "unknown"),
                    "page": meta.get("page", -1),
                    "distance": dist,
                }
            )
        return hits

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        """Delete all stored vectors (useful when re-ingesting from scratch)."""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
