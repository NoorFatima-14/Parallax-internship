# Parallax Labs AI/ML Internship

This is the main repository for the whole 6 week internship. Each week lives in its own folder with its own README, code, tests and dependencies.

## Weekly Progress

| Week | Folder | Topic | Status |
|------|--------|-------|--------|
| 1 | [week-1](week-1/) | Environment setup, dataset, text cleaning pipeline | Done |
| 2 | [week-2](week-2/) | Chunking, embeddings, ChromaDB, semantic search, retrieval benchmark | Done |
| 3 | [week-3](week-3/) | LLM Integration & Prompt Engineering | Done |
| 4 | [week-4](week-4/) | NLP Analysis | Done |
| 5 | [week-5](week-5/) | FastAPI Service and Evaluation |  Done |
| 6 | week-6 | | Not started |

## Repository Layout

```
.
├── README.md        # this file: overview and weekly progress
├── .gitignore       # shared rules (env files, data, caches, vector database)
├── week-1/          # see week-1/README.md
├── week-2/          # see week-2/README.md
└── week-3/ ...      # added each week
```

## How to Run a Week

Open that week's folder and follow its README. Example for Week 1:

```bash
cd week-1
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

All commands in a week's README are run from inside that week's folder. Week 2 uses the Week 1 virtual environment and the Week 1 clean corpus, so complete Week 1 first.

## Week 1 Summary

A script that checks the GPU and the library imports (pandas, sentence-transformers, ChromaDB, spaCy), a 5,000+ document arXiv dataset, a modular text cleaning pipeline with unit tests, and a validated clean corpus. My laptop has no GPU, so the GPU check is explained in the Week 1 README. Details, setup and results: [week-1/README.md](week-1/README.md).

## Week 2 Summary

A recursive chunking strategy with unit tests, sentence-transformers embeddings with timing logs, a ChromaDB vector store with documented edge-case handling, semantic search, and a script that benchmarks retrieval latency across different queries. The full corpus (6,563 abstracts) became 27,329 chunks in the database. Details, setup and results: [week-2/README.md](week-2/README.md).

## Week 3 Summary

A retrieval-augmented generation system on top of the Week 2 database: an LLM API client (Groq, OpenRouter) with rate-limit, timeout and token-limit handling and model fallbacks, a prompt with context injection, out-of-domain refusals, answer checks against hallucination, and end-to-end latency logging. Details: [week-3/README.md](week-3/README.md).

## Week 4 Summary

Implemented topic modeling, dataset extraction, and NLP-based metadata enrichment for the vector database. Integrated metadata-based filters into semantic search and RAG, evaluated retrieval performance. Details: [week-4/README.md](week-4/README.md).

## Week 5 Summary

A FastAPI service around the Week 3 RAG system and the Week 4 metadata filters (`/ask`, `/search`, `/topics`, `/datasets`, `/health`, `/ready`), with one JSON error format for every failure (400, 422, 500, and 503 or 504 when the language model service is down), and one structured JSON log line per request with a request id. Two automatic evaluations: retrieval (Precision@K, Recall@K, HitRate@K, MRR and nDCG@K on a test set of automatic and hand-judged queries) and end to end (40 real HTTP requests that check answer behaviour, answer quality signals and latency). 181 pytest tests that need no API key or internet. Details: [week-5/README.md](week-5/README.md).
