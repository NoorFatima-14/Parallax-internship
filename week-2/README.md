# Week 2: Chunking, Embeddings and Semantic Search

Week 2 of the Parallax Labs AI/ML internship. It uses the clean corpus from Week 1 . Run all commands below from inside the `week-2/` folder.

## What I Built

- `src/chunking.py`: a recursive text splitter (written from scratch) and a helper that chunks whole documents
- `src/embedding.py`: loads a sentence-transformers model, encodes text, and measures how long it takes
- `src/vector_store.py`: a ChromaDB wrapper that handles the edge cases (see the table below)
- `src/search.py`: semantic search that also reports latency
- `scripts/ingest.py`: chunk, embed and store the whole corpus, with timings
- `scripts/search.py`: search from the command line
- `scripts/benchmark_retrieval.py`: tests retrieval and logs latency for different queries
- `tests/`: unit tests for chunking, the vector store, embedding timing and search

## Approach

- **Chunking:** arXiv abstracts are about 190 words long, and embedding models work best on focused text. I split each abstract into chunks of about 500 characters with a small overlap, so one search result is a short, focused passage.
- **Model:** `all-MiniLM-L6-v2` (384 dimensions). It is small and fast on CPU, and my laptop has no GPU. The model name can be changed with `--model`.
- **Vectors:** embeddings are normalized and ChromaDB uses cosine distance. The score shown in results is `1 - distance`, so higher is more similar.
- **Own embeddings:** I pass my own vectors to ChromaDB (`embedding_function=None`). This keeps the embedding step separate, so I can time it, and ChromaDB never downloads a second model.
- **Safe re-runs:** chunks are stored with `upsert` and ids like `<arxiv id>::<chunk number>`, so running ingestion twice does not create duplicates.
- **Testing without a model:** the unit tests use a small fake embedding model that hashes words into vectors. This tests the code logic quickly and does not download anything. Retrieval quality of the real model is measured by the benchmark script, not by the unit tests.

## Chunking Strategy

**Method:** recursive character splitting.

1. If the text is already within `chunk_size`, keep it as one chunk.
2. Otherwise split at the biggest natural break that exists in the text, trying these in order: paragraph (`\n\n`), line (`\n`), sentence end (`. `, `? `, `! `), semicolon, comma, space.
3. Pack the pieces into chunks of at most `chunk_size` characters. A piece that is still too big is split again with the next separator.
4. If there is no separator at all (for example one very long word), cut at `chunk_size`.
5. Each new chunk starts with the last `chunk_overlap` characters of the previous chunk, cut at a word boundary, so an idea on the border is in both chunks.

**Settings (defaults):** `chunk_size = 500` characters (about 75 words), `chunk_overlap = 50` characters.

**Why these values:** the Week 1 abstracts average about 194 words, so each abstract becomes about 4 chunks on average (4.16 in the full run). The model's default input limit is 256 tokens, so 500-character chunks are never cut off. These are sensible starting values. I did not tune them.

**Known limits:**

- Sizes are in characters, not tokens.
- The sentence separator `. ` can split after abbreviations such as "et al." or "e.g.". Numbers like `0.5` are safe because there is no space after the dot.
- Overlap starts at a word boundary, so a chunk can begin in the middle of a sentence.
- When a piece had to be split again, the point where that happens has no overlap.

## ChromaDB Edge Cases

All of these are handled in `src/vector_store.py` and have a unit test in `tests/test_vector_store.py`.

| Case | What happens |
|------|--------------|
| Empty list of chunks | Nothing is added, no error |
| Empty or whitespace-only text | Chunk skipped and counted |
| Missing chunk id | Chunk skipped and counted |
| Duplicate ids in one batch | First kept, others skipped and counted (ChromaDB would raise an error) |
| Ingesting the same data twice | `upsert` replaces, so no duplicates |
| NaN or infinite values in a vector | Chunk skipped and counted (ChromaDB would raise an error) |
| Wrong number of embedding rows | Clear `ValueError` |
| Embedding size differs from the collection | Clear `ValueError` that mentions the model |
| Metadata value is `None` or a list | `None` dropped, other types turned into text |
| No metadata at all | A default `n_chars` field is added, because ChromaDB rejects empty metadata |
| Batch larger than ChromaDB's limit | Split into smaller batches automatically |
| Search on an empty collection | Returns an empty list |
| `top_k` larger than the collection | Capped to the collection size |
| `top_k` below 1 or not an integer | `ValueError` |
| Empty query, or not a string | `ValueError` |
| Very long query | Cut at 2,000 characters |
| Filter that matches nothing | Returns an empty list |
| Query vector of the wrong size, or with NaN | Clear `ValueError` |
| Collection built with a different model | `ValueError` when opening it |
| Invalid collection name | `ValueError` with the naming rules |
| Data after closing and reopening | Persisted on disk |

## Folder Structure

```
week-2/
├── src/
│   ├── chunking.py
│   ├── embedding.py
│   ├── vector_store.py
│   └── search.py
├── scripts/
│   ├── ingest.py
│   ├── search.py
│   └── benchmark_retrieval.py
├── tests/
│   ├── fakes.py                 # fake embedding model for tests
│   ├── test_chunking.py
│   ├── test_vector_store.py
│   └── test_embedding_search.py
├── logs/                        # timing and benchmark results
├── chroma_db/                   # vector database (git-ignored, created by ingest.py)
├── requirements.txt
└── README.md
```

## Progress

- [x] Recursive chunking with unit tests
- [x] Embedding with timing logs
- [x] ChromaDB ingestion and semantic search with unit tests
- [x] Retrieval benchmark script
- [x] Chunking strategy and ChromaDB edge cases documented
- [x] Ingestion run on the full corpus (`logs/ingest_log.json`)
- [x] Benchmark run (`logs/benchmark_summary.md`)

## Setup

Use the same virtual environment as Week 1. It already has all the libraries.

```bash
cd week-2
# activate the environment, for example on Windows:
..\week-1\.venv\Scripts\activate
```

The Week 1 clean corpus must exist. If it does not, build it first:

```bash
cd ../week-1
python scripts/download_data.py --per-category 1500
python scripts/build_clean_corpus.py
cd ../week-2
```

## Usage

```bash
# 1. Run the unit tests
pytest -v

# 2. Quick trial on 200 documents
python scripts/ingest.py --limit 200 --reset

# 3. Full run (chunk, embed, store). Add --reset to start from an empty collection.
python scripts/ingest.py --reset

# 4. Search
python scripts/search.py "object detection with neural networks" --top-k 5
python scripts/search.py "language models" --category cs.CL

# 5. Benchmark retrieval and latency
python scripts/benchmark_retrieval.py
```

`ingest.py` options: `--chunk-size`, `--chunk-overlap`, `--batch-size`, `--model`, `--device`. The first run downloads the model, so it needs an internet connection.

## Results

All numbers come from the log files written by the scripts (CPU run, Windows laptop, no GPU).

**Ingestion** (`logs/ingest_log.json`)

| Step | Result |
|------|--------|
| Documents | 6,563 |
| Chunks | 27,329 (4.16 per document, mean 378 characters, max 500) |
| Chunking time | 0.15 s |
| Model load time | about 5 s |
| Embedding time (CPU) | 1,500 s (25 minutes), 18.2 chunks per second, 54.9 ms per chunk |
| ChromaDB insert time | 47.5 s |

**Retrieval** (`logs/benchmark_summary.md` has the full tables)

- Search without a filter took about 25 to 50 ms per query on average. Most of that time is spent embedding the query (about 15 to 35 ms). The ChromaDB search itself took about 10 to 13 ms.
- The query length matters: the long query (313 characters) was the slowest because embedding it takes longer.
- Searches with a category filter took about 100 ms on average. The ChromaDB step was about 8 times slower than without a filter. I did not investigate why.
- Category match was 1.0 for most topic queries, and lower for image generation, reinforcement learning and the spelling-mistake query. Abstracts can belong to several categories, so this is a rough check only.
- Known-item search has three versions (see the table in `logs/benchmark_summary.md`). The **full chunk** version checks that the index works: a chunk used as its own query should come back first. The two **first sentence** versions are harder, because the opening sentences of different papers are often similar. They are split into the first chunk of a paper and later chunks, because a later chunk starts with text copied from the previous chunk (the overlap), so its first sentence is not a clean start.

What the benchmark measures:

- **Latency:** 13 queries (one word, a question, a long paragraph, spelling mistakes, an off-topic query, and queries with a category filter), each run 10 times. It reports mean, median (p50), p95, and how the time splits between embedding the query and searching ChromaDB.
- **Category match:** the share of results from a category we would expect.
- **Known-item search:** 100 first chunks and 100 later chunks (first-sentence queries) and 200 chunks (full-chunk queries), with recall@1, recall@5 and MRR.

Timings change a little from run to run and depend on the machine. The device and environment are saved with every log entry.


## Dependencies

See `requirements.txt` (locked with `pip freeze`). Main libraries: sentence-transformers, chromadb, PyTorch, pandas, pyarrow, numpy, pytest.

## Notes

- `chroma_db/` and data files are git-ignored. Run `scripts/ingest.py` to recreate the database. The `logs/` folder is committed as a record of the runs.
- Timings depend on the machine. The device and environment are saved with every log entry.
- Run `pytest` from inside `week-2/`, not from the repository root.
