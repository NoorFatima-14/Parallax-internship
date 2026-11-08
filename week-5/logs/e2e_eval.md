# End-to-End Evaluation (latest run)

- API: `http://127.0.0.1:8000`, cases: `e2e_cases.json` (40 requests, 24 called the LLM)
- Server: 27329 chunks, embedding `all-MiniLM-L6-v2`, LLM groq openai/gpt-oss-120b, openai/gpt-oss-20b, top_k 5, min_score 0.3, strict True
- Rates are shown with a 95% Wilson interval and the number of cases.

## Summary

- Cases passed: **40 of 40** (100% (91-100%, n=40))
- In-domain questions answered: 100% (76-100%, n=12)
- Off-topic and near-domain questions refused: 100% (77-100%, n=13)
- Sentences supported by the passages (mean over answers): 1.000
- Answers with a valid citation: 100% (77-100%, n=13)
- Answers containing an expected topic word: 100% (76-100%, n=12) (a rough relevance check)
- Filters respected: 100% (51-100%, n=4). Error bodies in the documented shape: 100% (74-100%, n=11)
- Answers with a warning: 0. Mean answer length: 76 words
- Server errors (5xx): 0. No response: 0

## Behaviour by group

| Group | Passed | Pass rate | What is expected |
|---|---|---|---|
| in_domain | 10/10 | 100% (72-100%, n=10) | Questions the corpus can answer. Expected: a grounded answer. |
| filtered | 4/4 | 100% (51-100%, n=4) | Requests with metadata filters. Expected: a result, and every source inside the filter. |
| off_topic | 8/8 | 100% (68-100%, n=8) | Questions unrelated to the papers. Expected: a refusal. |
| near_domain | 3/3 | 100% (44-100%, n=3) | Related to machine learning but not answerable from the abstracts. Expected: a refusal. |
| edge | 5/5 | 100% (57-100%, n=5) | Gibberish, prompt injection, very long text, another language. Expected: no server error. |
| http_contract | 10/10 | 100% (72-100%, n=10) | Bad requests. Expected: the documented 400, 404, 405 or 422 error body. |

## Latency

Client time is the full round trip seen by the caller (HTTP included). Server stages come from the `timings_ms` field of the response. All values in milliseconds.

| Requests | n | Client mean | p50 | p95 | max | HTTP overhead (mean) |
|---|---|---|---|---|---|---|
| ask, answered | 13 | 1257 | 1202 | 1744 | 1968 | 11 |
| ask, refused after the LLM call (model refusal or blocked) | 11 | 651 | 649 | 852 | 957 | 13 |
| ask, refused by the retrieval gate (no LLM call) | 4 | 91 | 91 | 117 | 120 | 13 |
| search | 1 | 394 | 394 | 394 | 394 | 11 |
| ask, all | 28 | 852 | 768 | 1538 | 1968 | 12 |

Server stages for answered questions:

| Stage | mean | p50 | p95 | max |
|---|---|---|---|---|
| embed_ms | 43 | 44 | 53 | 55 |
| search_ms | 73 | 48 | 209 | 324 |
| retrieval_ms | 117 | 90 | 254 | 375 |
| prompt_ms | 0 | 0 | 1 | 2 |
| llm_ms | 869 | 853 | 1238 | 1560 |
| verify_ms | 259 | 253 | 335 | 394 |
| total_ms | 1246 | 1195 | 1729 | 1953 |

Short refusals are faster than real answers, so the 'ask, all' line understates the time of a full answer. Use the 'answered' line for that.

## Cases that did not pass (0 of 40)

None.
