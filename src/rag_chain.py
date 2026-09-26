"""
rag_chain.py

Ties retrieval (VectorStore) together with generation (Anthropic Claude)
to answer student questions grounded in their own course material, with
source citations.
"""

from __future__ import annotations

import os
from typing import List

from anthropic import Anthropic

from .vector_store import VectorStore

DEFAULT_MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are a patient, encouraging study assistant helping a student \
understand their own course material. You will be given retrieved excerpts from the \
student's notes/textbook along with the student's question.

Rules:
- Answer using ONLY the provided context. If the context doesn't contain the answer, \
say so clearly and suggest what topic the student should look up instead of guessing.
- Explain concepts simply, as if tutoring, not just quoting the source.
- After your answer, list the sources you used in the format: [source_file, page].
- Keep answers focused and student-friendly — avoid unnecessary jargon.
"""


def _format_context(hits: List[dict]) -> str:
    blocks = []
    for h in hits:
        page = h["page"] if h["page"] not in (None, -1) else "N/A"
        blocks.append(f"[Source: {h['source']}, page {page}]\n{h['text']}")
    return "\n\n---\n\n".join(blocks)


class RAGChain:
    def __init__(self, store: VectorStore, api_key: str | None = None, model: str = DEFAULT_MODEL):
        self.store = store
        self.client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model

    def answer(self, question: str, top_k: int = 5) -> dict:
        """
        Retrieve relevant chunks and generate a grounded answer.
        Returns {"answer": str, "sources": List[dict]}
        """
        hits = self.store.query(question, top_k=top_k)

        if not hits:
            return {
                "answer": "I couldn't find anything relevant in your uploaded materials. "
                "Try uploading notes covering this topic first.",
                "sources": [],
            }

        context = _format_context(hits)
        user_message = f"Context from the student's materials:\n\n{context}\n\nQuestion: {question}"

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        answer_text = "".join(
            block.text for block in response.content if block.type == "text"
        )

        return {"answer": answer_text, "sources": hits}

    def summarize_topic(self, topic: str, top_k: int = 8) -> str:
        """Summarize everything the student's materials say about a topic."""
        hits = self.store.query(topic, top_k=top_k)
        if not hits:
            return f"No material found on '{topic}' in your uploaded notes."

        context = _format_context(hits)
        prompt = (
            f"Summarize the following excerpts into clear, well-organized study notes "
            f"on the topic '{topic}'. Use headings and bullet points.\n\n{context}"
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1200,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in response.content if block.type == "text")
