# Evaluation and Benchmarks

All numbers on this page are copied from the log files that the scripts wrote (the source file is named in each section). Nothing here was estimated by hand. A script (`docs/check_numbers.py`) re-reads the logs and checks the headline numbers against this page.

**Where the results came from:** Windows 10, Python 3.11.5, PyTorch 2.14.1 (CPU only, no GPU), 27,329 chunks in ChromaDB, embedding model `all-MiniLM-L6-v2`. Times on another computer will be different. Quality numbers (precision, recall, refusals) will only be the same if you use the same corpus (see "Reproducibility notes" at the end).

## 1. Summary

| Area | Main result | Source |
|---|---|---|
| Corpus | 6,563 arXiv abstracts, 27,329 chunks (500 characters, 50 overlap) | week-2/logs/ingest_log.json |
| Retrieval, known paper (200 queries) | Recall@1 0.825, Recall@5 0.905, MRR 0.863 | week-5/logs/retrieval_eval.md |
| Retrieval, topical queries (20, hand judged) | Precision@5 0.870, MRR 0.950 | week-5/logs/retrieval_eval.md |
| Search speed | mean 56 ms, p95 74 ms (CPU, includes embedding the query) | week-5/logs/retrieval_eval.md |
| RAG answers (Week 3, 28 questions) | 100% met expectation in every group | week-3/logs/rag_eval_summary.md |
| End to end over HTTP (Week 5, 40 requests) | 40 of 40 passed, 0 server errors | week-5/logs/e2e_eval.md |
| Answer time | mean 1257 ms, p95 1744 ms for answered questions | week-5/logs/e2e_eval.md |
| Topics | BERTopic: 31 topics, NPMI 0.214; you rated 27 good, 4 mixed, 0 poor | week-4/logs |
| Dataset mentions | best system: precision 0.56, recall 0.45, F1 0.50 (50 documents) | week-4/logs/nlp_metadata_evaluation.md |
| Hallucination checks | 3 kinds of wrong answer blocked 8/8; "answer to another question" missed 1 of 8; mixed true+invented flagged 8/8 | week-3/logs/hallucination_check_summary.md |
| Tests | 186 pytest tests in Week 5 (fake model and fake LLM, no network) | week-5/tests |

## 2. Corpus and indexing (Week 2)

| Item | Value |
|---|---|
| Documents | 6,563 (0 skipped) |
| Chunks | 27,329, mean 378.2 characters |
| Chunking time | 0.15 s |
| ChromaDB ingest time | 47.5 s |
| Embedding dimension | 384 |

Time to rebuild everything, from the logs: embedding about 1,500 s, BERTopic 107 s, NER 606 s, metadata enrichment about 18 s.

## 3. Retrieval speed (Week 2 benchmark)

Top 5, 10 repeats per query, CPU. Source: `week-2/logs/benchmark_summary.md`.

| Query type | Mean ms | p95 ms | Top score |
|---|---|---|---|
| one keyword | 27.97 | 33.69 | 0.5471 |
| question | 27.08 | 35.27 | 0.7535 |
| computer vision | 36.00 | 52.60 | 0.5787 |
| long query (about 60 words) | 46.50 | 58.82 | 0.6420 |
| spelling mistakes | 33.65 | 50.04 | 0.3967 |
| off-topic ("best recipe for chocolate cake") | 28.23 | 35.58 | 0.2673 |
| filter: cs.CV | 101.55 | 182.50 | 0.5787 |
| filter: cs.CL | 105.40 | 186.18 | 0.6771 |

What it shows: a normal query takes about 25 to 45 ms. Most of that is embedding the query (14 to 35 ms). A category filter makes search about 3 to 4 times slower (search part 84 to 90 ms against about 11 ms). The off-topic query has the lowest top score (0.267), and this is the idea behind the out-of-domain gate.

## 4. Retrieval quality (Week 5 evaluation)

Source: `week-5/logs/retrieval_eval.md`. 220 queries. Values are means with a 95% bootstrap interval in the log. Document level means one entry per paper; chunk level scores the chunks exactly as the RAG system receives them.

| Query set | Queries | Recall@1 | Recall@5 | Recall@10 | Precision@5 | MRR |
|---|---|---|---|---|---|---|
| all | 220 | 0.760 | 0.872 | 0.927 | 0.244 | 0.871 |
| known item (all) | 200 | 0.825 | 0.905 | 0.920 | 0.181 | 0.863 |
| known item, title as query | 100 | 0.790 | 0.900 | 0.930 | 0.180 | 0.847 |
| known item, one abstract sentence as query | 100 | 0.860 | 0.910 | 0.910 | 0.182 | 0.879 |
| topical (hand judged) | 20 | 0.114 | 0.546 | 1.000 | 0.870 | 0.950 |

How to read it:

- For known-item queries there is exactly one right paper, so Precision@K can never be higher than 1/K. Use Recall@K and MRR for them. Precision@5 0.181 is close to the best possible 0.200.
- For topical queries, Precision@K is the useful number (0.90 at K=3, 0.87 at K=5, 0.81 at K=10). Recall@K is low at small K simply because a topic has about 8 relevant papers.
- Search time during this evaluation: mean 56 ms, p50 55 ms, p95 74 ms.
- 16 of 220 queries had no right paper in the top 10. They are listed in the log. Many are very generic sentences ("Extensive simulations ... demonstrate the effectiveness of the proposed methods.") that fit thousands of papers.

Why Week 2 shows 0.615 and Week 4/5 show higher values for "first sentence" queries: Week 2 picks random chunks from all 27,329 (any part of an abstract, many are middle or end chunks) and needs the exact chunk back. The Week 4 filter benchmark uses the first sentence of a paper's opening chunk, and Week 5 counts a hit when the right paper is found. These are different tests and should not be compared directly.

## 5. RAG answer quality (Week 3)

Source: `week-3/logs/rag_eval_summary.md`. Settings: top_k 5, min_score 0.3, strict mode on, LLM `openai/gpt-oss-120b` on Groq.

| Group | Questions | Met expectation | Outcomes |
|---|---|---|---|
| In domain (should answer) | 10 | 100% | answered 10 |
| Off topic (should refuse) | 8 | 100% | refused_by_model 4, refused_out_of_domain 4 |
| Related but not in corpus (should refuse) | 3 | 100% | refused_by_model 3 |
| Edge cases (no crash) | 7 | 100% | answered 1, invalid_query 2, refused_by_model 4 |

Score gap used for the threshold: lowest top score of an in-domain question 0.473, highest top score of an off-topic question 0.429, threshold 0.30. The threshold is therefore lower than both, which is safe but means the model, not the gate, refuses some off-topic questions. Answered sentences supported by the passages: 100% (mean 1.0). LLM calls: 22, no retries, no fallback, 19,085 tokens in total.

Latency (ms) for requests that called the LLM (n = 22): retrieval mean 80.0, LLM mean 705.3 (p95 1010), verification mean 253.5, total mean 913.3 (p95 1379).

### Hallucination checks

Source: `week-3/logs/hallucination_check_summary.md`. 8 questions, real passages, wrong answers written on purpose, no LLM.

| Test | Result |
|---|---|
| True statement (control) | 8/8 clean |
| Invented number | 8/8 blocked |
| Invalid citation number | 8/8 blocked |
| Unrelated facts | 8/8 blocked |
| Answer to another question | 7/8 blocked, **1 missed** |
| True sentence plus an invented one | 8/8 flagged (warned) |

The one miss: a true-sounding sentence about uncertainty with a valid citation passed because its words overlap with the cited passage. The checks look at citations, numbers and word support; they cannot fully judge meaning. This is a known limit.

## 6. End-to-end API evaluation (Week 5)

Source: `week-5/logs/e2e_eval.md`. 40 real HTTP requests against the running API, 24 of them called the LLM. Intervals are 95% Wilson.

| Group | Passed | What is expected |
|---|---|---|
| in_domain | 10/10 | grounded answer |
| filtered | 4/4 | result, every source inside the filter |
| off_topic | 8/8 | refusal |
| near_domain | 3/3 | refusal |
| edge | 5/5 | no server error (gibberish, prompt injection, very long text, another language) |
| http_contract | 10/10 | documented 400/404/405/422 error body |
| **Total** | **40/40** | 100% (91 to 100%) |

Other measures: sentences supported by the passages 1.000; answers with a valid citation 100% (n=13); answers with warnings 0; mean answer length 76 words; server errors 0.

Client latency in ms (includes HTTP):

| Request type | n | Mean | p50 | p95 | Max |
|---|---|---|---|---|---|
| ask, answered | 13 | 1257 | 1202 | 1744 | 1968 |
| ask, refused after the LLM call | 11 | 651 | 649 | 852 | 957 |
| ask, refused by the gate (no LLM) | 4 | 91 | 91 | 117 | 120 |
| ask, all | 28 | 852 | 768 | 1538 | 1968 |

Server stages for answered questions (mean ms): embed 43, search 73, LLM 869, verify 259, total 1246. The LLM is about 70% of the time; the HTTP layer adds about 12 ms.

## 7. NLP metadata (Week 4)

### Topic models, source `week-4/logs/topic_model_comparison.md`

| Method | Topics | Core share | NPMI coherence | Diversity | Seconds |
|---|---|---|---|---|---|
| BERTopic | 31 | 0.634 | 0.2141 | 0.8258 | 107.0 |
| LDA | 25 | 1.000 | 0.0833 | 0.9080 | 59.7 |

BERTopic is clearly more coherent (0.214 against 0.083). Only 63.4% of documents sit in a real cluster; the others are assigned to the nearest topic afterwards and marked `topic_core = False`. Largest topic: "learning, data, bound, error" (1,543 documents).

Human ratings (`nlp_metadata_evaluation.md`): 27 of 31 topics good (87%, interval 71 to 95%), 4 mixed, 0 poor. Mean NPMI of good topics 0.232 against 0.090 for mixed, so the automatic score agrees with the human view. Topic assignment of single documents (60 checked): fully correct 58% (46 to 70%), correct or partly 83% (72 to 91%).

### Dataset mentions (NER)

Seed list found a known dataset in 517 documents (7.9%); with patterns, 1,087 (16.6%). Checked by hand on 50 documents (33 mentions):

| System | Precision | Recall | F1 |
|---|---|---|---|
| spaCy baseline | 0.09 | 0.48 | 0.16 |
| seed list | 0.57 | 0.12 | 0.20 |
| seed list + patterns | 0.56 | 0.45 | 0.50 |

Seed list + patterns is the best. On the 30 random documents alone it is weak (F1 0.14), so treat dataset filters as approximate. The sample is small and the intervals are wide.

### Metadata filters, source `week-4/logs/filter_benchmark.md`

200 queries, top 5. Every filter had 0 violations (no result outside the filter).

| Filter | Recall@5 | MRR | Mean ms |
|---|---|---|---|
| none | 0.980 | 0.953 | 35.8 |
| category | 0.990 | 0.963 | 87.3 |
| topic | 0.995 | 0.972 | 68.1 |
| year | 0.980 | 0.953 | 259.3 |
| topic + year | 0.995 | 0.972 | 154.0 |
| dataset (25 queries) | 1.000 | 1.000 | 150.4 |

Filtering does not hurt quality and can help a little, but a year filter is slow (about 259 ms) because it scans all chunks.

## 8. Limits to keep in mind

- Small samples: 20 hand-judged topical queries, 50 NER documents, 60 topic checks, 28 and 40 RAG cases. Read the intervals, not only the averages.
- Topical recall is probably too high because only papers returned by the system were judged (pooling).
- Known-item queries made from the paper's own abstract are easier than real user questions.
- The LLM is an external service; its answers and speed can change between runs. Free tiers can return 429/503/504, which the API reports as `llm_*` errors and which are not counted as bugs.
- The hallucination checks are rule based and missed 1 of 8 in one test.
- Corpus is abstracts only (no full text), 5 arXiv categories.

## 9. Reproducibility notes

- `data/` and `chroma_db/` are not in git. Running Week 1 again downloads whatever arXiv returns on that day, so the papers, and therefore the exact numbers, will differ. Keep a copy of your corpus file if you want identical results.
- Random seeds are fixed (42) for sampling in the benchmarks, so the same corpus gives the same samples.
- Versions are pinned in `requirements.txt`. Environment details are stored inside each JSON log (`environment` key).
