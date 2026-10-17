# Retrieval Benchmark (latest run)

- Model: `all-MiniLM-L6-v2` on `cpu`
- Chunks in database: 27329
- top_k: 5, repeats per query: 10

## Fixed queries

| Query type | Chars | Mean ms | p50 ms | p95 ms | Embed ms | Search ms | Top score | Category match |
|---|---|---|---|---|---|---|---|---|
| one keyword | 12 | 27.97 | 28.77 | 33.69 | 16.71 | 11.26 | 0.5471 | - |
| question | 57 | 27.08 | 26.94 | 35.27 | 15.74 | 11.33 | 0.7535 | 1.0 |
| computer vision | 51 | 36.0 | 33.17 | 52.6 | 23.54 | 12.46 | 0.5787 | 1.0 |
| learning theory | 46 | 24.13 | 23.08 | 29.63 | 14.17 | 9.96 | 0.6558 | 1.0 |
| reinforcement learning | 51 | 25.87 | 26.45 | 32.66 | 15.65 | 10.22 | 0.6798 | 0.8 |
| image generation | 39 | 31.65 | 32.72 | 36.62 | 19.02 | 12.63 | 0.7934 | 0.6 |
| statistics | 48 | 31.4 | 32.19 | 42.32 | 19.53 | 11.87 | 0.6583 | 1.0 |
| translation | 47 | 27.43 | 27.82 | 36.44 | 16.39 | 11.04 | 0.6034 | 1.0 |
| long query (about 60 words) | 313 | 46.5 | 46.73 | 58.82 | 34.52 | 11.97 | 0.642 | 1.0 |
| spelling mistakes | 51 | 33.65 | 31.4 | 50.04 | 20.96 | 12.7 | 0.3967 | 0.8 |
| off-topic | 30 | 28.23 | 27.96 | 35.58 | 17.13 | 11.11 | 0.2673 | - |
| filter: cs.CV | 51 | 101.55 | 84.72 | 182.5 | 17.11 | 84.44 | 0.5787 | 1.0 |
| filter: cs.CL | 15 | 105.4 | 91.11 | 186.18 | 15.08 | 90.32 | 0.6771 | 1.0 |

Category match is the share of results whose arXiv category is one we would expect for the topic. It is a rough check only, because papers can have several categories.

## Known-item search

200 random chunks. The query is either the first sentence of the chunk or the full chunk text.

| Metric | First sentence | Full chunk |
|---|---|---|
| Recall@1 | 0.495 | 1.0 |
| Recall@5 | 0.615 | 1.0 |
| MRR | 0.541 | 1.0 |
| Paper recall@5 | 0.675 | 1.0 |
| Latency mean / p50 / p95 (ms) | 34.77 / 33.25 / 55.99 | 47.55 / 44.29 / 70.71 |
