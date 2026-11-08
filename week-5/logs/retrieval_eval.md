# Retrieval Evaluation (latest run)

- Test set: `retrieval_testset.json`, 220 queries
- Collection: 27329 chunks, embedding model `all-MiniLM-L6-v2`
- Search depth: 40 chunks per query (K up to 10)
- Metric values are means over queries, with a 95% bootstrap interval in brackets.

How to read it: **document level** reduces the results to one entry per paper and scores the top K papers. **Chunk level** scores the top K chunks as the RAG system receives them (a chunk counts as relevant when its paper is).

## all (220 queries, 1.6 relevant papers on average)

All queries together.

MRR: 0.871 (0.83-0.91). Search latency: mean 56 ms, p50 55 ms, p95 74 ms.

| K | Precision@K (doc) | ideal Precision@K | Recall@K (doc) | HitRate@K (doc) | nDCG@K (doc) | Precision@K (chunk) | Recall@K (chunk) |
|---|---|---|---|---|---|---|---|
| 1 | 0.832 (0.78-0.88) | 1.000 | 0.760 (0.70-0.81) | 0.832 (0.78-0.88) | 0.832 (0.78-0.88) | 0.832 (0.78-0.88) | 0.760 (0.70-0.81) |
| 3 | 0.355 (0.33-0.38) | 0.394 | 0.849 (0.80-0.89) | 0.909 (0.87-0.95) | 0.871 (0.83-0.91) | 0.561 (0.52-0.60) | 0.843 (0.80-0.89) |
| 5 | 0.244 (0.22-0.27) | 0.271 | 0.872 (0.83-0.91) | 0.914 (0.87-0.95) | 0.872 (0.83-0.91) | 0.425 (0.39-0.46) | 0.871 (0.83-0.91) |
| 10 | 0.157 (0.13-0.19) | 0.165 | 0.927 (0.89-0.96) | 0.927 (0.89-0.96) | 0.882 (0.84-0.92) | 0.268 (0.24-0.30) | 0.923 (0.89-0.96) |

## known_item (200 queries, 1.0 relevant papers on average)

One relevant paper per query (the paper the query was made from). Precision@K cannot exceed 1/K here, so read it together with the ideal value, and use Recall@K and MRR as the main numbers.

MRR: 0.863 (0.82-0.91). Search latency: mean 56 ms, p50 55 ms, p95 73 ms.

| K | Precision@K (doc) | ideal Precision@K | Recall@K (doc) | HitRate@K (doc) | nDCG@K (doc) | Precision@K (chunk) | Recall@K (chunk) |
|---|---|---|---|---|---|---|---|
| 1 | 0.825 (0.77-0.88) | 1.000 | 0.825 (0.77-0.88) | 0.825 (0.77-0.88) | 0.825 (0.77-0.88) | 0.825 (0.77-0.88) | 0.825 (0.77-0.88) |
| 3 | 0.300 (0.28-0.31) | 0.333 | 0.900 (0.85-0.94) | 0.900 (0.85-0.94) | 0.868 (0.82-0.91) | 0.527 (0.48-0.57) | 0.895 (0.85-0.94) |
| 5 | 0.181 (0.17-0.19) | 0.200 | 0.905 (0.86-0.94) | 0.905 (0.86-0.94) | 0.871 (0.83-0.91) | 0.380 (0.35-0.41) | 0.905 (0.86-0.94) |
| 10 | 0.092 (0.09-0.10) | 0.100 | 0.920 (0.88-0.95) | 0.920 (0.88-0.95) | 0.876 (0.83-0.92) | 0.213 (0.20-0.23) | 0.920 (0.88-0.95) |

## known_item/title (100 queries, 1.0 relevant papers on average)

The query is the paper title. Not copied from the abstract text, so this is the realistic known-item case.

MRR: 0.847 (0.79-0.91). Search latency: mean 54 ms, p50 53 ms, p95 68 ms.

| K | Precision@K (doc) | ideal Precision@K | Recall@K (doc) | HitRate@K (doc) | nDCG@K (doc) | Precision@K (chunk) | Recall@K (chunk) |
|---|---|---|---|---|---|---|---|
| 1 | 0.790 (0.71-0.87) | 1.000 | 0.790 (0.71-0.87) | 0.790 (0.71-0.87) | 0.790 (0.71-0.87) | 0.790 (0.71-0.87) | 0.790 (0.71-0.87) |
| 3 | 0.297 (0.28-0.32) | 0.333 | 0.890 (0.83-0.95) | 0.890 (0.83-0.95) | 0.850 (0.79-0.91) | 0.557 (0.49-0.62) | 0.890 (0.83-0.95) |
| 5 | 0.180 (0.17-0.19) | 0.200 | 0.900 (0.84-0.95) | 0.900 (0.84-0.95) | 0.855 (0.79-0.92) | 0.408 (0.36-0.46) | 0.900 (0.84-0.95) |
| 10 | 0.093 (0.09-0.10) | 0.100 | 0.930 (0.88-0.98) | 0.930 (0.88-0.98) | 0.865 (0.81-0.92) | 0.230 (0.20-0.26) | 0.930 (0.88-0.98) |

## known_item/sentence (100 queries, 1.0 relevant papers on average)

The query is one sentence copied from the abstract. This is an easy check and should be close to 1.

MRR: 0.879 (0.81-0.94). Search latency: mean 58 ms, p50 57 ms, p95 76 ms.

| K | Precision@K (doc) | ideal Precision@K | Recall@K (doc) | HitRate@K (doc) | nDCG@K (doc) | Precision@K (chunk) | Recall@K (chunk) |
|---|---|---|---|---|---|---|---|
| 1 | 0.860 (0.79-0.93) | 1.000 | 0.860 (0.79-0.93) | 0.860 (0.79-0.93) | 0.860 (0.79-0.93) | 0.860 (0.79-0.93) | 0.860 (0.79-0.93) |
| 3 | 0.303 (0.28-0.32) | 0.333 | 0.910 (0.85-0.96) | 0.910 (0.85-0.96) | 0.886 (0.82-0.94) | 0.497 (0.44-0.55) | 0.900 (0.84-0.95) |
| 5 | 0.182 (0.17-0.19) | 0.200 | 0.910 (0.85-0.96) | 0.910 (0.85-0.96) | 0.886 (0.82-0.94) | 0.352 (0.31-0.39) | 0.910 (0.85-0.96) |
| 10 | 0.091 (0.08-0.10) | 0.100 | 0.910 (0.85-0.96) | 0.910 (0.85-0.96) | 0.886 (0.82-0.94) | 0.196 (0.17-0.22) | 0.910 (0.85-0.96) |

## topical (20 queries, 8.1 relevant papers on average)

Papers judged relevant by hand. Judged only among the papers that the system itself returned (pooling), so Recall@K is probably too high, because relevant papers the system never returns were never judged.

MRR: 0.950 (0.88-1.00). Search latency: mean 53 ms, p50 49 ms, p95 80 ms.

| K | Precision@K (doc) | ideal Precision@K | Recall@K (doc) | HitRate@K (doc) | nDCG@K (doc) | Precision@K (chunk) | Recall@K (chunk) |
|---|---|---|---|---|---|---|---|
| 1 | 0.900 (0.75-1.00) | 1.000 | 0.114 (0.09-0.14) | 0.900 (0.75-1.00) | 0.900 (0.75-1.00) | 0.900 (0.75-1.00) | 0.114 (0.09-0.14) |
| 3 | 0.900 (0.82-0.97) | 1.000 | 0.339 (0.32-0.36) | 1.000 | 0.900 (0.81-0.97) | 0.900 (0.82-0.97) | 0.328 (0.30-0.36) |
| 5 | 0.870 (0.78-0.94) | 0.980 | 0.546 (0.52-0.58) | 1.000 | 0.886 (0.81-0.95) | 0.880 (0.79-0.95) | 0.531 (0.49-0.57) |
| 10 | 0.810 (0.72-0.89) | 0.810 | 1.000 | 1.000 | 0.946 (0.90-0.98) | 0.815 (0.72-0.89) | 0.952 (0.92-0.98) |

## Queries with no relevant paper in the top 10 (16 of 220)

Read these first when looking for what to improve.

- `ki-title-0006` (known_item): Oracle-Efficient Online Classification with Stochastic Inputs and Adversarial Outputs
- `ki-title-0018` (known_item): A Statistical Framework for Data-Driven Discovery of Differential Performance in Clinical Risk Prediction Mode
- `ki-sentence-0029` (known_item): IV shows that many prescribed interventions provide little or no marginal benefit.
- `ki-title-0042` (known_item): Generative Residual Factorization
- `ki-title-0044` (known_item): On the Behavioral Traits of LLM Agents
- `ki-title-0058` (known_item): Smoother Flow Matching via Contrastive Trajectory Repulsion
- `ki-sentence-0058` (known_item): Second, they also imply drastic velocity variations, leading to severe numerical integration errors in few-ste
- `ki-sentence-0059` (known_item): Extensive simulations and real-data analyses demonstrate the effectiveness of the proposed methods.
- `ki-title-0060` (known_item): E2E-SWE: Benchmarking LLMs on Building Working Codebases from Scratch
- `ki-sentence-0062` (known_item): The study evaluates key factors such as interpretability, scalability, robustness, and real-world applicabilit
- `ki-sentence-0073` (known_item): Making these systems inspectable requires exposing interpretable evidence while retaining the richer informati
- `ki-sentence-0074` (known_item): We instantiate it as MS$^3$, a Material-Sensor-Signal-System schema for conductive-fiber flexible sensors, ove
- `ki-sentence-0081` (known_item): (3) The intuition is not maximally condensed, and can be further compressed into a knoledge base.
- `ki-title-0088` (known_item): Benchmarking Psychological Dynamics in Generative Agents
- `ki-sentence-0090` (known_item): These errors propagate to the final answer in at least 82.0% of affected tasks on each benchmark.
- ... and 1 more (see the JSON log)
