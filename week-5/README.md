# Week 5: FastAPI Service, Request Logging, and Automated Evaluation

Week 5 of the Parallax Labs AI/ML internship. It wraps the Week 3 RAG system and the Week 4 metadata filters in a FastAPI service, and adds two automatic evaluations: one for retrieval (Precision@K, Recall@K) and one end to end (answer quality and latency). Run all commands below from inside the `week-5/` folder.

## What I Built

- **API** (`api/`): FastAPI app with `/ask`, `/search`, `/topics`, `/datasets`, `/health` and `/ready`
- **Error handling** (`api/errors.py`): one JSON error format for every failure, with the HTTP codes 400, 422, 500 (and 503, 504 for the language model service)
- **Request logging** (`api/middleware.py`, `api/logging_setup.py`): one JSON line per request, with a request id, timings and the pipeline result
- **Retrieval evaluation** (`evals/retrieval_eval.py`, `scripts/evaluate_retrieval.py`): Precision@K, Recall@K, HitRate@K, MRR and nDCG@K on a test set
- **Test set builder** (`evals/testset_builder.py`, `scripts/make_retrieval_testset.py`): automatic known-item queries, plus topic queries judged by hand
- **End-to-end evaluation** (`evals/e2e_eval.py`, `scripts/evaluate_e2e.py`): sends 40 real HTTP requests to the running API and reports behaviour, answer quality checks and latency
- **Tests** (`tests/`): 181 pytest tests. They need no API key, no internet and no model download.

## How the API Works

The API does not contain new RAG logic. `api/service.py` calls the Week 3 `RAGPipeline` and turns the Week 4 filter options into a ChromaDB `where` clause. The embedding model, the database and the LLM client are loaded once, when the server starts.

If loading fails (no database, no API key), the server still starts. `/health` works and the other endpoints answer `503` with the reason, instead of the process crashing.

| Endpoint | What it does |
|----------|--------------|
| `POST /ask` | Retrieves passages, calls the LLM, checks the answer against the passages, returns the answer and its sources |
| `POST /search` | Returns the closest passages only (no LLM, no key needed) |
| `GET /topics`, `GET /datasets` | The names that can be used in filters (from Week 4) |
| `GET /health` | Is the process alive? |
| `GET /ready` | Are the database, the embedding model and the LLM client loaded? Shows the settings |
| `GET /docs` | Interactive documentation made by FastAPI |

Both `/ask` and `/search` accept the Week 4 filters: `topics`, `categories`, `datasets` (with `datasets_mode` any or all), `year_from`, `year_to`, `core_only` and `exclude_short`.

```bash
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" \
  -d '{"question": "How is uncertainty estimated?", "filters": {"topics": ["uncertainty"], "core_only": true}}'
```

An answer looks like this (shortened):

```json
{"request_id": "8576dbb3...", "status": "answered",
 "answer": "Convolutional neural networks detect objects in images [1].",
 "sources": [{"n": 1, "doc_id": "2601.00001v1", "title": "Pedestrian detection", "category": "cs.CV", "score": 0.4835}],
 "grounding": {"level": "grounded", "grounded_fraction": 1.0, "citations": [1], "invalid_citations": []},
 "timings_ms": {"embed_ms": 0.1, "search_ms": 3.7, "retrieval_ms": 3.8, "prompt_ms": 0.2, "llm_ms": 0.0, "verify_ms": 0.6, "total_ms": 4.8}}
```

**Refusals are not errors.** If the question is off-topic, the model says "I don't know", or the hallucination checks block the answer, the API returns HTTP 200 and the `status` field tells what happened (`refused_out_of_domain`, `refused_by_model`, `blocked_ungrounded`). The request itself was fine. This is the same set of statuses as in Week 3.

## HTTP Errors

Every error has the same body, and the same `request_id` as the `X-Request-ID` response header:

```json
{"error": {"code": "invalid_question", "message": "...", "request_id": "0fd423d3..."}}
```

Only a 422 (and an LLM failure) adds `details`: the list of wrong fields, for example `{"field": "top_k", "message": "Input should be greater than or equal to 1", "type": "greater_than_equal"}`.

| HTTP | Code | When |
|------|------|------|
| **400** | `invalid_json` | The body is not valid JSON |
| **400** | `invalid_question` | The question or query is blank, or only unprintable characters |
| **400** | `invalid_filter` | Unknown topic or dataset name, or `year_from` after `year_to` |
| **400** | `filter_not_available` | A topic or dataset filter was used, but the Week 4 files were not loaded |
| **422** | `validation_error` | Valid JSON that does not match the schema: missing field, wrong type, `top_k` outside 1 to 20, text longer than 4,000 characters, unknown field. `details` lists each field. |
| **404**, **405** | `not_found`, `method_not_allowed` | Wrong path or method |
| **500** | `internal_error` | Something broke inside the service. The client gets a generic message and the request id. The traceback is only in the log. |
| **500** | `llm_misconfigured` | The API key was rejected, or the model name does not exist |
| **503** | `service_not_ready` | The database or key could not be loaded at start-up |
| **503** | `llm_rate_limited`, `llm_unreachable`, `llm_unavailable` | The language model service is limited, offline or failing. A `Retry-After` header is sent for rate limits. |
| **504** | `llm_timeout` | The language model service did not answer in time |

**Why 400 and 422 are separate.** 422 means the request has the wrong shape, and FastAPI finds that before my code runs. 400 means the shape is right but the content cannot be used (blank text, a filter name that does not exist). A client can fix a 422 by reading `details`, and a 400 by reading `message`.

**Why 503 and 504 when the task names 400, 422 and 500.** If the LLM provider is rate limited, that is not a bug in my service, and the caller should retry. A 500 would hide that. When the search worked but the LLM failed, the error `details` include the top passages, so the caller still gets something useful (same idea as the Week 3 fallback).

**Long text.** A question between 1,000 and 4,000 characters is cut to 1,000 (the Week 3 limit) and the response has the warning `question_truncated`. Above 4,000 it is rejected with 422.

## Request Logging

Every request writes one JSON line to the console and to `logs/api_requests.jsonl` (rotating file, 5 MB, 3 backups). Example (an answered question, shortened):

```json
{"ts": "2026-11-08T16:01:28.444+00:00", "level": "INFO", "logger": "api.request", "request_id": "8576dbb380d347e292b32f8e3a7dad6e",
 "method": "POST", "path": "/ask", "status_code": 200, "duration_ms": 14.0, "client": "testclient", "endpoint": "ask",
 "filters_used": true, "question_chars": 54, "question_preview": "Convolutional neural networks detect ...",
 "pipeline_status": "answered", "top_score": 0.4835, "n_sources": 1, "filter_applied": {"category": {"$in": ["cs.CV"]}},
 "timings_ms": {"embed_ms": 0.1, "search_ms": 3.7, "retrieval_ms": 3.8, "prompt_ms": 0.2, "llm_ms": 0.0, "verify_ms": 0.6, "total_ms": 4.8},
 "model": "fake-model", "llm_retries": 0, "llm_fallbacks": [], "total_tokens": 42, "grounding_level": "grounded", "grounded_fraction": 1.0}
```

This line was captured from the test setup (a 4-paper database and a fake model), which is why the model is `fake-model` and the times are tiny. With the real database the same fields are filled with real values.

- **Request id.** Each request gets a new id (or keeps a safe `X-Request-ID` sent by the caller). It is in the response header, in the error body and in every log line of that request, so one failing request can be found in the log.
- **Log level follows the result.** 2xx is INFO, 4xx is WARNING, 5xx is ERROR. A failed request has `error_code` and `error_message`.
- **Unexpected errors.** The middleware catches them, writes the traceback to the log, and returns the generic 500. The exception text never reaches the client.
- **Privacy.** The log keeps the first 100 characters of each question. Set `API_LOG_QUESTIONS=false` to log only the length. The API key is never logged.
- Start uvicorn with `--no-access-log`, so its own lines do not repeat what the JSON line already says.

## How to Run

```bash
cd week-5
..\week-1\.venv\Scripts\activate          # Windows
pip install fastapi uvicorn httpx
pytest -v                                  # 181 tests, no key or data needed

uvicorn api.main:create_app --factory --port 8000 --no-access-log
# open http://127.0.0.1:8000/docs
```

The server needs the Week 2 `chroma_db/`, and for `/ask` the API key in `week-3/.env`. For topic and dataset filters it also needs the Week 4 enrichment (`week-4/data/processed/topics.json` and `datasets_index.json`).

Settings come from environment variables (all optional): `RAG_DB_PATH`, `RAG_COLLECTION`, `RAG_TOPICS_PATH`, `RAG_DATASETS_PATH`, `LLM_PROVIDER`, `LLM_MODELS`, `RAG_DEVICE`, `RAG_TOP_K`, `RAG_MIN_SCORE`, `RAG_STRICT`, `API_LOG_FILE`, `API_LOG_LEVEL`, `API_LOG_QUESTIONS`. A `.env` file in `week-5/` or `week-3/` is read too.

## Retrieval Evaluation

**What is measured.** For each test query the system returns ranked chunks. Relevance is judged per paper, so a chunk is relevant when its paper is.

| Metric | Meaning |
|--------|---------|
| Precision@K | relevant results in the top K, divided by K |
| Recall@K | different relevant papers in the top K, divided by all relevant papers |
| HitRate@K | 1 if at least one relevant paper is in the top K |
| MRR | 1 divided by the rank of the first relevant paper |
| nDCG@K | like Precision, but a relevant paper higher in the list counts more |

Each is reported at two levels. **Document level** reduces the chunk list to one entry per paper first (so 5 chunks of the same paper count once). **Chunk level** scores the top K chunks as they come, which is what the RAG system really receives. Values are means over queries, with a 95% bootstrap interval.

**The test set** has two kinds of query:

1. **Known-item (automatic).** For 100 random papers (seed 42): the paper's title, and one sentence from its abstract. The one relevant paper is the paper the query came from. The title is not part of the chunk text, so the title query is a fair test.
2. **Topical (judged by hand).** 20 topic queries (`eval_data/topical_queries.json`). The system's top papers for each are written to a sheet **in random order, without rank or score**, and I mark each `y` or `n` by reading the abstract. This is called pooling.

```bash
python scripts/make_retrieval_testset.py known-item      # 200 automatic queries -> eval_data/retrieval_testset.json
python scripts/make_retrieval_testset.py pool-sheet      # writes labels/retrieval_judgments.csv
#   fill the 'relevant' column with y or n (about 30 to 45 minutes)
python scripts/make_retrieval_testset.py merge-labels    # adds the judged topical queries
python scripts/evaluate_retrieval.py                     # writes logs/retrieval_eval.md and .json
python scripts/evaluate_retrieval.py --limit 20          # quick trial
```

No API key or internet is needed. The report is split by query type, so the easy known-item numbers are not mixed with the harder topical ones.

**Why Precision@K looks low for known-item queries, and is not a bug.** A known-item query has one relevant paper. Even a perfect system cannot get Precision@5 above 1/5 = 0.2 at document level. So the report always shows the **ideal Precision@K** next to the real one, and for known-item queries Recall@K and MRR are the main numbers.

## End-to-End Evaluation

`scripts/evaluate_e2e.py` sends the 40 cases in `eval_data/e2e_cases.json` to the running server over HTTP, so it also measures the HTTP layer, the logging middleware and the JSON conversion.

| Group | Cases | Expected |
|-------|-------|----------|
| `in_domain` | 10 | A grounded answer (the Week 3 questions) |
| `filtered` | 4 | A result, and every source inside the requested category |
| `off_topic` | 8 | A refusal |
| `near_domain` | 3 | A refusal (related to ML, but the abstracts do not contain the answer) |
| `edge` | 5 | No server error: gibberish, prompt injection, very long text, Spanish |
| `http_contract` | 10 | The documented 400, 404, 405 or 422 error body, with a matching request id |

```bash
uvicorn api.main:create_app --factory --port 8000 --no-access-log     # terminal 1
python scripts/evaluate_e2e.py                                        # terminal 2 (about 5 minutes)
python scripts/evaluate_e2e.py --limit 2                              # quick trial: 2 cases per group
python scripts/evaluate_e2e.py --groups in_domain off_topic
python scripts/evaluate_e2e.py --delay 15 --strict                    # slower pace, exit code 1 on any failure
```

**Generation quality, with no human rating and no second LLM call:**

- **Answer rate** for in-domain questions, and **refusal rate** for off-topic and near-domain ones. An off-topic question that gets an answer is listed by id as "answered by mistake". This is the number to watch for hallucination.
- **Supported sentences**: mean share of answer sentences supported by the passages (the Week 3 check, returned in `grounding`).
- **Citations**: share of answers with at least one valid `[n]` and no invalid one.
- **Topic word check**: does the answer contain a word that any good answer would use (for example "detect" for the object detection question). This is rough and only a signal.
- **Filters respected**: every source is inside the filter.
- **Error contract**: every 4xx and 5xx body has `code`, `message`, and a `request_id` equal to the `X-Request-ID` header.

**Latency**, in milliseconds: the round trip seen by the client (mean, p50, p95, max) for answered questions, refusals after the LLM call, refusals by the retrieval gate, and search. The server's own stage times (embedding, search, prompt, LLM, verification) come from `timings_ms`, and the **HTTP overhead** is client time minus server total. Short "I don't know" replies are faster than real answers, so the "answered" line is the one to quote for answer latency.

**Two kinds of failure are told apart.** If the LLM provider is rate limited, those cases fail with `llm_rate_limited`, and the report says that the problem is the provider and not the retrieval or checking logic. Run again with a bigger `--delay`. Real server bugs (5xx not caused by the LLM service) are counted separately.

The script waits `--delay` seconds (default 10) after each request that called the LLM, because free Groq plans allow about 30 requests and 8,000 tokens per minute.

## Tests

```bash
pytest -v            # run from inside week-5/
```

| File | What it checks |
|------|----------------|
| `test_ask_and_search.py` | The endpoints with a real tiny ChromaDB, a fake embedding model and a scripted fake LLM: answers, refusals, filters, sources, timings |
| `test_errors.py` | The error codes in the table above, including LLM failures (rate limit, timeout, bad key), a broken database and an unexpected exception. A 422 does not echo the submitted text, and a 500 is logged with its traceback but the client does not see it. |
| `test_logging.py` | One JSON line per request with the basic fields, the pipeline details for `/ask`, request ids that match the response, log level by status code, the question text switched off, the log file, and no duplicate handlers |
| `test_system_endpoints.py` | `/health`, `/ready`, `/topics`, `/datasets`, 404 and 405, a server that failed to start |
| `test_metrics.py`, `test_retrieval_eval.py` | The metrics against values worked out by hand, the evaluation, the test set builder and the judged sheet |
| `test_e2e_eval.py` | The case file, the pass or fail rules, the summary, the report, and the script run against a real local uvicorn server |

The tests use a real ChromaDB (4 papers) with a fake embedding model, and a fake LLM that returns scripted answers or errors. They test the service logic. Behaviour with the real model and the live LLM is what the two evaluation scripts are for.

## Folder Structure

```
week-5/
├── api/                 # main.py, routes.py, schemas.py, service.py, errors.py, middleware.py, logging_setup.py, settings.py
├── evals/               # metrics.py, retrieval_eval.py, e2e_eval.py, testset.py, testset_builder.py
├── scripts/             # evaluate_retrieval.py, evaluate_e2e.py, make_retrieval_testset.py
├── eval_data/           # e2e_cases.json, topical_queries.json, retrieval_testset.json (made by the script)
├── labels/              # retrieval_judgments.csv (my hand judgments, kept in git)
├── logs/                # api_requests.jsonl, retrieval_eval.md/.json, e2e_eval.md/.json
├── tests/
└── requirements.txt
```

## Results

All numbers are written by the scripts. Nothing is entered by hand. After running the steps above, the results are in:

- `logs/retrieval_eval.md`: Precision@K, Recall@K, HitRate@K, MRR and nDCG@K by query type, and the queries that found no relevant paper
- `logs/e2e_eval.md`: pass rate by group, answer and refusal rates, supported sentences, citations, latency and the cases that failed
- `logs/api_requests.jsonl`: the request log of the server

## Limits

- **Known-item queries are an easy test.** They check that a paper can be found from its own title or text, not that a user's real question gets the best papers. The topical queries are closer to real use but there are only 20.
- **Pooling.** Topical relevance is judged only among the papers the system itself returned. Relevant papers it never returned were never judged, so topical Recall@K is probably too high. One person (me) judged everything, so there is no agreement score.
- **"Passed" does not mean "correct".** The end-to-end checks show that the system answered when it should, refused when it should, and that the answer is supported by its passages. They cannot tell a reversed claim from a correct one (the Week 3 limit). Only a person reading the answers can rate correctness.
- **The topic word check is rough.** An answer can contain the word and still be poor, or be good without it.
- **Small samples.** 40 cases, 10 in-domain questions. The Wilson intervals in the report are wide for that reason.
- **Latency depends on the machine and the provider.** It was measured on one laptop with one free-tier provider, and includes provider waiting time.
- **No authentication, rate limiting or HTTPS.** The service is meant for local use. Before putting it on the internet it needs an API key check, request limits and a reverse proxy.
- **One request at a time is the tested case.** The endpoints are plain `def` functions, so FastAPI runs them in a thread pool, but I did not load test concurrent requests. The embedding model runs on CPU and would slow down.
- **Free tiers change.** Model names and limits may be different later (see Week 3).

## Notes

- `chroma_db/`, `data/` and `.env` are git-ignored. `logs/` and the judged sheet in `labels/` are committed as the evidence for the numbers. `logs/api_requests.jsonl` is ignored by the root `.gitignore`.
- Run `pytest` from inside `week-5/`, not from the repository root.
