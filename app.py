"""
app.py

Streamlit front-end for the Student Helping Agent.

Run with:
    streamlit run app.py

Three modes:
  1. Ask a Question  - RAG Q&A grounded in your uploaded materials, with citations
  2. Summarize Topic  - condenses everything your notes say about a topic
  3. Practice Quiz    - generates a multiple-choice quiz from your notes
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.document_processor import process_directory
from src.quiz_generator import QuizGenerator
from src.rag_chain import RAGChain
from src.vector_store import VectorStore

load_dotenv()

st.set_page_config(page_title="Student Helping Agent", page_icon="📚", layout="wide")


@st.cache_resource
def get_store() -> VectorStore:
    return VectorStore()


def ingest_uploaded_files(uploaded_files) -> int:
    """Save uploaded files to a temp dir, process, and add to the vector store."""
    store = get_store()
    total_added = 0

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        for f in uploaded_files:
            (tmp_path / f.name).write_bytes(f.read())

        chunks = process_directory(tmp_path)
        total_added += store.add_chunks(chunks)

    return total_added


def main() -> None:
    st.title("📚 Student Helping Agent")
    st.caption("A RAG-powered study assistant grounded in your own course materials.")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        st.warning(
            "No ANTHROPIC_API_KEY found. Set it in a `.env` file (see `.env.example`) "
            "before asking questions or generating quizzes."
        )

    store = get_store()

    with st.sidebar:
        st.header("1. Upload materials")
        uploaded = st.file_uploader(
            "PDF, TXT, or Markdown files",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
        )
        if uploaded and st.button("Ingest files"):
            with st.spinner("Processing and embedding documents..."):
                added = ingest_uploaded_files(uploaded)
            st.success(f"Added {added} chunks to your knowledge base.")

        st.divider()
        st.metric("Chunks stored", store.count())

        if st.button("Clear knowledge base", type="secondary"):
            store.reset()
            st.success("Knowledge base cleared.")
            st.rerun()

    mode = st.radio(
        "What do you want to do?",
        ["Ask a Question", "Summarize a Topic", "Practice Quiz"],
        horizontal=True,
    )

    if store.count() == 0:
        st.info("Upload some course material in the sidebar to get started.")
        return

    chain = RAGChain(store)

    if mode == "Ask a Question":
        question = st.text_input("Your question")
        if question and st.button("Get answer"):
            with st.spinner("Thinking..."):
                result = chain.answer(question)
            st.markdown(result["answer"])
            with st.expander("Sources used"):
                for s in result["sources"]:
                    st.write(f"- **{s['source']}**, page {s['page']}")

    elif mode == "Summarize a Topic":
        topic = st.text_input("Topic to summarize")
        if topic and st.button("Summarize"):
            with st.spinner("Summarizing..."):
                summary = chain.summarize_topic(topic)
            st.markdown(summary)

    elif mode == "Practice Quiz":
        topic = st.text_input("Quiz topic")
        num_q = st.slider("Number of questions", 3, 10, 5)
        if topic and st.button("Generate quiz"):
            with st.spinner("Writing quiz questions..."):
                quiz_gen = QuizGenerator(store)
                questions = quiz_gen.generate(topic, num_questions=num_q)

            if not questions:
                st.error("Couldn't generate a quiz — try a different topic or upload more material.")
            else:
                for i, q in enumerate(questions, start=1):
                    st.subheader(f"Q{i}. {q['question']}")
                    choice = st.radio(
                        "Select an answer",
                        q["options"],
                        key=f"q_{i}",
                        index=None,
                    )
                    if choice is not None:
                        correct = q["options"][q["correct_index"]]
                        if choice == correct:
                            st.success("Correct! " + q["explanation"])
                        else:
                            st.error(f"Not quite. Correct answer: {correct}. {q['explanation']}")


if __name__ == "__main__":
    main()
