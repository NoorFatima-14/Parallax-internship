# Week 1: Environment, Data and Text Cleaning

Week 1 of the Parallax Labs AI/ML internship. Run all commands below from inside the `week-1/` folder.

## What I Built

- A clean Python environment with a locked `requirements.txt`
- `scripts/verify_env.py`: checks for a CUDA GPU and tests that pandas, sentence-transformers, ChromaDB and spaCy import and work
- `scripts/download_data.py`: downloads paper abstracts from the arXiv API into `data/raw/`
- `src/preprocessing.py`: small, separate text cleaning functions (HTML stripping, Unicode normalization, whitespace cleanup, language filtering)
- `tests/test_preprocessing.py`: unit tests for every cleaning function
- `scripts/build_clean_corpus.py`: cleans the raw data, checks the result, and saves `data/processed/clean_corpus.parquet`

## Approach

- **Dataset:** I used arXiv abstracts because the API is free, needs no login, and gives real text. I downloaded from five categories (cs.CL, cs.LG, cs.CV, cs.AI, stat.ML) so the corpus is not about one topic only.
- **Cleaning:** Each step is its own function, so it is easy to test and reuse. `clean_text` runs them in order: HTML, then Unicode, then whitespace. `clean_corpus` also drops short documents, non-English documents and exact duplicates.
- **HTML stripping:** Only real HTML tags (like `<p>` or `<a href=...>`) are removed. arXiv abstracts contain maths such as `x < 0.5 and y > 2`, and a general tag remover can delete text like that. I found this when my output check failed on the first run, so I made the tag pattern stricter and added tests for it.
- **Language filter:** I used `langdetect` with a fixed seed so results are the same on every run.
- **Validation:** `build_clean_corpus.py` stops with an error if any HTML tag or repeated whitespace is left, if text is empty, or if ids are not unique.
- **No local GPU:** My laptop has no GPU. Everything except the CUDA check runs on CPU, so I did it locally. The CUDA check is run on Google Colab.

## Folder Structure

```
week-1/
├── data/
│   ├── raw/                  # raw downloads (git-ignored)
│   └── processed/            # cleaned corpus (git-ignored)
├── scripts/
│   ├── verify_env.py         # GPU + import checks
│   ├── download_data.py      # fetch arXiv abstracts
│   └── build_clean_corpus.py # raw -> clean parquet
├── src/
│   └── preprocessing.py      # text cleaning functions
├── tests/
│   └── test_preprocessing.py
├── requirements.txt
└── README.md
```

## Progress

- [x] Git repo, virtual environment, `.gitignore` (data files and `.env`), locked `requirements.txt`
- [x] `scripts/verify_env.py` written; import checks pass
- [x] CUDA check run on a GPU (Google Colab)
- [x] Dataset of 5,000+ arXiv abstracts saved to `data/raw/`
- [x] Cleaning functions in `src/preprocessing.py`
- [x] Unit tests in `tests/test_preprocessing.py`
- [x] Clean dataset at `data/processed/clean_corpus.parquet`

## Setup

Requires Python 3.10 or newer. A GPU is only needed for the CUDA check. Without one, everything else runs on CPU.

```bash
git clone https://github.com/NoorFatima-14/Parallax-internship.git
cd Parallax-internship/week-1

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Usage

```bash
# 1. Verify the environment
python scripts/verify_env.py              # needs a CUDA GPU

# 2. Download raw data (about 6,500 abstracts, takes a few minutes)
python scripts/download_data.py --per-category 1500

# 3. Clean and validate
python scripts/build_clean_corpus.py

# 4. Run the tests
pytest -v
```

The raw and processed data are git-ignored, so run steps 2 and 3 to recreate them.

## Running the GPU Check on Google Colab

My laptop has no GPU, so the CUDA check is run on a free Colab T4 GPU.

1. Open a new notebook at colab.research.google.com.
2. Go to Runtime, then Change runtime type, and choose T4 GPU.
3. Run these cells:

```
!git clone https://github.com/NoorFatima-14/Parallax-internship.git
%cd Parallax-internship/week-1
!pip install -q sentence-transformers chromadb spacy pandas pyarrow requests langdetect pytest
!python -m spacy download en_core_web_sm
!python scripts/verify_env.py
```

## Dataset

- **Source:** arXiv API, abstracts from cs.CL, cs.LG, cs.CV, cs.AI, stat.ML
- **Raw file:** `data/raw/arxiv_abstracts.jsonl`
- **Output:** `data/processed/clean_corpus.parquet` with columns `id, title, text, published, category`

Cleaning steps, in order: HTML stripping, Unicode NFKC normalization, whitespace cleanup, minimum length (20 words), English-only filter, exact-duplicate removal.

**Final counts**

| Step | Documents |
|------|-----------|
| Raw download | 6,563 |
| After 20 word minimum | 6,563 |
| After English only filter | 6,563 |
| After de duplication | 6,563 |

The filters removed no documents because arXiv abstracts in these categories were already English, long enough and unique. The filters are covered by unit tests.

## Dependencies

See `requirements.txt`. 
Main libraries: pandas, pyarrow, requests, langdetect, sentence-transformers, chromadb, spaCy, PyTorch, pytest.

## Notes

- The development laptop has no GPU.
