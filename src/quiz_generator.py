"""
quiz_generator.py

Generates a short multiple-choice practice quiz from the student's own
uploaded material on a given topic, using the same retrieval step as
rag_chain.py but a generation prompt tuned for structured quiz output.
"""

from __future__ import annotations

import json
import os
from typing import List, TypedDict

from anthropic import Anthropic

from .vector_store import VectorStore

DEFAULT_MODEL = "claude-sonnet-4-6"


class QuizQuestion(TypedDict):
    question: str
    options: List[str]
    correct_index: int
    explanation: str


QUIZ_SYSTEM_PROMPT = """You generate multiple-choice practice quizzes for students, \
based only on the excerpts of course material you are given.

Respond with ONLY valid JSON (no markdown fences, no commentary) matching this shape:
{
  "questions": [
    {
      "question": "...",
      "options": ["...", "...", "...", "..."],
      "correct_index": 0,
      "explanation": "..."
    }
  ]
}
Generate exactly the number of questions requested. Base every question strictly on \
the provided context — do not invent facts not present in it.
"""


class QuizGenerator:
    def __init__(self, store: VectorStore, api_key: str | None = None, model: str = DEFAULT_MODEL):
        self.store = store
        self.client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model

    def generate(self, topic: str, num_questions: int = 5, top_k: int = 8) -> List[QuizQuestion]:
        hits = self.store.query(topic, top_k=top_k)
        if not hits:
            return []

        context = "\n\n---\n\n".join(h["text"] for h in hits)
        user_message = (
            f"Course material excerpts:\n\n{context}\n\n"
            f"Generate {num_questions} multiple-choice questions on the topic '{topic}'."
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1500,
            system=QUIZ_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        raw_text = "".join(block.text for block in response.content if block.type == "text")

        try:
            parsed = json.loads(raw_text)
            return parsed.get("questions", [])
        except json.JSONDecodeError:
            print("[quiz_generator] Model did not return valid JSON:\n", raw_text)
            return []
