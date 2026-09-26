# 📚 Student Helping Agent

A Retrieval-Augmented Generation (RAG) study assistant that answers questions,
summarizes topics, and generates practice quizzes — all grounded in **your own**
course materials (PDFs, notes, markdown files), not the model's general training
data.

Built to demonstrate a practical, end-to-end RAG pipeline: document ingestion,
chunking, local embeddings, vector search, and grounded LLM generation with
source citations.

---

## Features

- **Document ingestion** — upload PDFs / TXT / Markdown files; they're chunked
  and embedded automatically.
- **Grounded Q&A** — answers cite the exact source file and page they came from,
  and the model is instructed to say "I don't know" rather than hallucinate
  when the answer isn't in your materials.
- **Topic summarization** — turns scattered notes into organized study notes.
- **Auto-generated practice quizzes** — multiple-choice questions with
  explanations, generated only from your own uploaded content.
- **Local embeddings** — uses `sentence-transformers` so ingestion is free and
  fast; only the final answer generation calls the Claude API.

## Architecture

```
                ┌─────────────────┐
   PDF/TXT/MD → │ document_        │  chunks text with overlap,
   files        │ processor.py     │  keeps source + page metadata
                └────────┬─────────┘
                         │ Chunk objects
                         ▼
                ┌─────────────────┐
                │ vector_store.py  │  embeds locally (MiniLM),
                │ (ChromaDB)       │  stores + queries vectors
                └────────┬─────────┘
                         │ top-k relevant chunks
                         ▼
        ┌────────────────┴────────────────┐
        ▼                                  ▼
┌───────────────┐                 ┌──────────────────┐
│ rag_chain.py   │                 │ quiz_generator.py │
│ Q&A + summary  │                 │ MCQ generation    │
│ via Claude API │                 │ via Claude API    │
└───────────────┘                 └──────────────────┘
        │                                  │
        └────────────────┬─────────────────┘
                          ▼
                  ┌───────────────┐
                  │   app.py       │  Streamlit UI
                  └───────────────┘
```

## Setup

```bash
# 1. Clone / unzip the project, then:
cd student-helping-agent
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your Anthropic API key
cp .env.example .env
# then edit .env and paste your key

# 4. Run the app
streamlit run app.py
```

The first run will download the local embedding model (`all-MiniLM-L6-v2`),
which takes a minute but only happens once.

## Running tests

```bash
pip install pytest
pytest tests/
```

## Project structure

```
student-helping-agent/
├── app.py                  # Streamlit UI — entry point
├── requirements.txt
├── .env.example
├── src/
│   ├── document_processor.py   # loading + chunking
│   ├── vector_store.py         # ChromaDB wrapper
│   ├── rag_chain.py            # Q&A + summarization
│   └── quiz_generator.py       # quiz generation
├── data/                    # (optional) drop files here for CLI ingestion
└── tests/
    └── test_basic.py
```

## Possible extensions (good "future work" talking points)

- Swap `all-MiniLM-L6-v2` for a larger embedding model and benchmark retrieval quality.
- Add a re-ranking step (e.g. cross-encoder) before passing chunks to the LLM.
- Support OCR for scanned/image-based PDFs.
- Add conversation memory so follow-up questions retain context.
- Deploy behind a lightweight FastAPI backend instead of Streamlit for a
  production-style API.

## Why this project is a good resume piece

It touches most of the skills recruiters look for in an "AI engineering" or
"applied ML" role without needing to train a model from scratch:

- Practical **RAG pipeline design** (chunking strategy, overlap, embeddings, retrieval)
- **Vector database** usage (ChromaDB)
- **LLM API integration** and prompt engineering (grounding, citation, JSON-mode output)
- **Working UI** a non-technical person (e.g. an interviewer) can actually click through
- Basic **testing** discipline
