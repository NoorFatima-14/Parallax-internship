# Parallax Labs AI/ML Internship

This is the main repository for the whole 6 week internship. Each week lives in its own folder with its own README, code, tests and dependencies.

## Weekly Progress

| Week | Folder | Topic | Status |
|------|--------|-------|--------|
| 1 | [week-1](week-1/) | Environment setup, dataset, text cleaning pipeline | Done |
| 2 | week-2 | | Not started |
| 3 | week-3 | | Not started |
| 4 | week-4 | | Not started |
| 5 | week-5 | | Not started |
| 6 | week-6 | | Not started |

## Repository Layout

```
.
├── README.md        # this file: overview and weekly progress
├── .gitignore       
├── week-1/         
└── week-2/ ...      
```

## How to Run a Week

Open that week's folder and follow its README. Example for Week 1:

```bash
cd week-1
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

All commands in a week's README are run from inside that week's folder.

## Week 1 Summary

Verified environment (CUDA, pandas, sentence-transformers, ChromaDB, spaCy), a 5,000+ document arXiv dataset, a modular text cleaning pipeline with unit tests, and a validated clean corpus. Details, setup and results: [week-1/README.md](week-1/README.md).
