# Week 1: Environment, Data and Text Cleaning

Part of the Parallax Labs AI/ML internship repository. Run all commands below from inside this `week-1/` folder.

## Folder Structure

```
week-1/
├── data/
│   ├── raw/                 # raw downloads (git-ignored)
│   └── processed/           # cleaned corpus (git-ignored)
├── scripts/
│   ├── verify_env.py        # GPU + import checks
│   ├── download_data.py     # fetch arXiv abstracts
│   └── build_clean_corpus.py# raw -> clean parquet
├── src/
│   └── preprocessing.py     # text cleaning functions
├── tests/
│   └── test_preprocessing.py
├── requirements.txt
└── README.md
```

## Progress

- [x] Git repo, virtual environment, `.gitignore`, locked `requirements.txt`
- [x] `scripts/verify_env.py` (CUDA check and imports for pandas, sentence-transformers, ChromaDB, spaCy)
- [x] Dataset of 5,000+ arXiv abstracts saved to `data/raw/`
- [x] Cleaning functions in `src/preprocessing.py` (HTML stripping, Unicode normalization, whitespace cleanup, language filtering)
- [x] Unit tests in `tests/test_preprocessing.py`
- [x] Clean dataset at `data/processed/clean_corpus.parquet`

## Setup

Requires Python 3.10+ and an NVIDIA GPU with a working driver.

```bash
git clone <repo-url>
cd <repo-name>/week-1

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

If `torch` installs without CUDA, install the CUDA build from https://pytorch.org/get-started/locally/ first, then run the command above.

## Usage

```bash
# 1. Verify environment
python scripts/verify_env.py

# 2. Download raw data (about 7,500 abstracts, takes a few minutes)
python scripts/download_data.py --per-category 1500

# 3. Clean and validate
python scripts/build_clean_corpus.py

# 4. Run tests
pytest -v
```

## Dataset

- **Source:** arXiv API, abstracts from cs.CL, cs.LG, cs.CV, cs.AI, stat.ML
- **Raw file:** `data/raw/arxiv_abstracts.jsonl`
- **Output:** `data/processed/clean_corpus.parquet` with columns `id, title, text, published, category`

Cleaning steps, in order: HTML stripping, Unicode NFKC normalization, whitespace cleanup, minimum length (20 words), English-only filter (`langdetect`), exact-duplicate removal. `build_clean_corpus.py` asserts that no HTML tags or repeated whitespace remain and that ids are unique.

**Final counts:** _fill in from the output of `build_clean_corpus.py`_

## Dependencies

See `requirements.txt` (versions locked with `pip freeze`). Main libraries: pandas, pyarrow, requests, langdetect, sentence-transformers, chromadb, spaCy, PyTorch, pytest.

## Notes

- Data and `.env` files are git-ignored. Re-create the dataset with the commands above.
- Language detection is seeded for repeatable results.
