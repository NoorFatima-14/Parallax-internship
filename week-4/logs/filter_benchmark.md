# Filter Benchmark (latest run)

200 random chunks, first sentence as the query, top 5, model on cpu.

| Filter | Queries | Violations | Recall@1 | Recall@5 | MRR | Mean ms | p50 ms | p95 ms | Chunks passing the filter (mean) |
|---|---|---|---|---|---|---|---|---|---|
| no filter | 200 | 0 | 0.935 | 0.980 | 0.953 | 35.8 | 33.6 | 50.7 | 27329 |
| category | 200 | 0 | 0.940 | 0.990 | 0.963 | 87.3 | 85.3 | 118.2 | 5693 |
| topic | 200 | 0 | 0.955 | 0.995 | 0.972 | 68.1 | 63.8 | 94.2 | 2380 |
| year | 200 | 0 | 0.935 | 0.980 | 0.953 | 259.3 | 253.9 | 299.1 | 27329 |
| topic + year | 200 | 0 | 0.955 | 0.995 | 0.972 | 154.0 | 149.0 | 202.5 | 2380 |
| dataset | 25 | 0 | 1.000 | 1.000 | 1.000 | 150.4 | 142.7 | 181.8 | 125 |
