"""Measure what the metadata filters do to search: correctness, speed and recall.

For random chunks, the first sentence is used as the query (as in Week 2). Each query is
run with no filter and with filters built from that chunk's own metadata. Reported:
- violations: results that break the filter (must be 0)
- recall@5 and MRR: is the original chunk found again?
- latency: mean, p50, p95
- candidates: how many chunks pass the filter on average

A filter built from the chunk's own topic always contains that chunk, so a higher recall
here shows the effect of a smaller search space, assuming the right topic is chosen.
It does not show how well users can pick the right topic.

Run:  python scripts/benchmark_filters.py
"""

import argparse
import random
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nlpmeta import cli, paths  # noqa: E402
from nlpmeta.enrich import DATASET_PREFIX  # noqa: E402
from nlpmeta.filters import Filters, build_where, matches  # noqa: E402
from nlpmeta.search import count_matching, filtered_search  # noqa: E402


def first_sentence(text: str) -> str:
    sentence = re.split(r"(?<=[.!?])\s", text, maxsplit=1)[0]
    return (sentence if len(sentence) >= 30 else text)[:250]


def conditions(meta: dict) -> dict:
    """Filters for one chunk, built from its own metadata. Conditions that do not apply are left out."""
    out = {"no filter": Filters(), "category": Filters(categories=[meta["category"]])}
    if "topic_id" in meta:
        out["topic"] = Filters(topics=[meta["topic_id"]])
    if "published_year" in meta:
        out["year"] = Filters(year_from=meta["published_year"], year_to=meta["published_year"])
    if "topic_id" in meta and "published_year" in meta:
        out["topic + year"] = Filters(topics=[meta["topic_id"]], year_from=meta["published_year"],
                                      year_to=meta["published_year"])
    flags = sorted(k for k, v in meta.items() if k.startswith(DATASET_PREFIX) and v)
    if flags:
        out["dataset"] = Filters(datasets=[flags[0]])
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    cli.add_path_args(p)
    p.add_argument("--samples", type=int, default=200)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    args = p.parse_args()
    P = cli.resolve(args)

    store = cli.open_store(args)
    w2 = paths.week2()
    model, info = w2.load_model(w2.DEFAULT_MODEL, args.device)
    data = store.collection.get(include=["documents", "metadatas"])
    pool = [(i, d, m) for i, d, m in zip(data["ids"], data["documents"], data["metadatas"])
            if m.get("chunk_index") == 0 and len(d) >= 100 and "topic_id" in m]
    if not pool:
        sys.exit("No enriched chunks found. Run scripts/enrich_vector_db.py first.")
    picked = random.Random(args.seed).sample(pool, min(args.samples, len(pool)))
    for _ in range(3):  # warm-up
        filtered_search("warm up query", model, store, Filters(), args.top_k)

    rows = {}
    for chunk_id, text, meta in picked:
        query = first_sentence(text)
        for name, f in conditions(meta).items():
            results, t, where = filtered_search(query, model, store, f, args.top_k)
            ids = [r["id"] for r in results]
            row = rows.setdefault(name, {"rank": [], "ms": [], "viol": 0, "cand": [], "empty": 0})
            row["rank"].append(ids.index(chunk_id) + 1 if chunk_id in ids else None)
            row["ms"].append(t["total_ms"])
            row["viol"] += sum(not matches(r["metadata"], f) for r in results)
            row["empty"] += not results
            row["cand"].append(count_matching(store, where))

    lines = ["# Filter Benchmark (latest run)", "",
             f"{len(picked)} random chunks, first sentence as the query, top {args.top_k}, model on {info['device']}.", "",
             "| Filter | Queries | Violations | Recall@1 | Recall@" + str(args.top_k) + " | MRR | Mean ms | p50 ms | p95 ms | Chunks passing the filter (mean) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    summary = {}
    for name, r in rows.items():
        n = len(r["rank"])
        hit1 = sum(x == 1 for x in r["rank"]) / n
        hitk = sum(x is not None for x in r["rank"]) / n
        mrr = sum(1 / x for x in r["rank"] if x) / n
        ms = np.array(r["ms"])
        summary[name] = {"n": n, "violations": r["viol"], "recall_1": hit1, "recall_k": hitk, "mrr": mrr,
                         "mean_ms": float(ms.mean()), "p50_ms": float(np.percentile(ms, 50)),
                         "p95_ms": float(np.percentile(ms, 95)), "mean_candidates": float(np.mean(r["cand"]))}
        lines.append(f"| {name} | {n} | {r['viol']} | {hit1:.3f} | {hitk:.3f} | {mrr:.3f} | {ms.mean():.1f} | "
                     f"{np.percentile(ms, 50):.1f} | {np.percentile(ms, 95):.1f} | {np.mean(r['cand']):.0f} |")
    text = "\n".join(lines) + "\n"
    (P.logs / "filter_benchmark.md").write_text(text, encoding="utf-8")
    w2.append_json_log(P.logs / "filter_benchmark.json", {"samples": len(picked), "top_k": args.top_k,
                                                          "device": info["device"], "results": summary,
                                                          "environment": w2.environment_info()})
    print(text)
    if any(v["violations"] for v in summary.values()):
        print("WARNING: some results broke the filter. This is a bug. Do not rely on the filters.")


if __name__ == "__main__":
    main()
