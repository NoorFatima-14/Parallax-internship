# RAG Evaluation (latest run)

- Provider / models: groq / openai/gpt-oss-120b, openai/gpt-oss-20b
- Embedding model: `all-MiniLM-L6-v2`, chunks in database: 27329
- Settings: top_k 5, min_score 0.3, strict True

This is an automatic check. It does not replace reading the answers.

## In-domain questions

Expected: a grounded answer

- Met expectation: 100% (10/10)
- Outcomes: answered 10

- Blocked by the hallucination checks: 0

## Off-topic questions

Expected: a refusal

- Met expectation: 100% (8/8)
- Outcomes: refused_by_model 4, refused_out_of_domain 4

- Answered although it should refuse: 0

## Related but not in the corpus

Expected: a refusal (a model may know the answer from training, which is exactly what should not be used)

- Met expectation: 100% (3/3)
- Outcomes: refused_by_model 3

- Answered although it should refuse: 0

## Edge cases

Expected: handled without a crash

- Met expectation: 100% (7/7)
- Outcomes: answered 1, invalid_query 2, refused_by_model 4

## Retrieval scores and the out-of-domain threshold

- Lowest top score among in-domain questions: 0.473
- Highest top score among off-topic questions: 0.429
- Threshold in use (min_score): 0.3

A good threshold sits between these two numbers.

## Latency (milliseconds)

Requests that called the LLM:

| Stage | n | Mean | p50 | p95 | Max |
|---|---|---|---|---|---|
| embed | 22 | 50.5 | 49.6 | 69.7 | 107.2 |
| search | 22 | 29.5 | 26.6 | 47.6 | 49.4 |
| retrieval | 22 | 80.0 | 78.3 | 107.3 | 120.0 |
| llm | 22 | 705.3 | 672.0 | 1010.0 | 1391.0 |
| verify | 11 | 253.5 | 268.2 | 325.6 | 356.0 |
| total | 22 | 913.3 | 793.0 | 1379.4 | 1761.0 |

Requests refused before the LLM (retrieval only):

| Stage | n | Mean | p50 | p95 | Max |
|---|---|---|---|---|---|
| embed | 4 | 34.0 | 33.3 | 45.7 | 46.3 |
| search | 4 | 24.7 | 25.6 | 32.6 | 32.7 |
| total | 4 | 59.1 | 59.8 | 78.5 | 79.0 |

## API behaviour

- LLM calls: 22, models used: {'openai/gpt-oss-120b': 22}
- Retries: 0 (waited 0.0 s), fallback models used: 0
- Errors: none, truncated answers: 0, total tokens: 19085
- Mean share of answer sentences supported by the passages: 1.0

