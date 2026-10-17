# Parallax Labs AI/ML Internship

This is the main repository for the whole 6 week internship. Each week lives in its own folder with its own README, code, tests and dependencies.

## Weekly Progress

| Week | Folder | Topic | Status |
|------|--------|-------|--------|
| 1 | [week-1](week-1/) | Environment setup, dataset, text cleaning pipeline | Done |
| 2 | [week-2](week-2/) | Chunking, embeddings, ChromaDB, semantic search, retrieval benchmark | Done |
| 3 | week-3 | | Not started |
| 4 | week-4 | | Not started |
| 5 | week-5 | | Not started |
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
