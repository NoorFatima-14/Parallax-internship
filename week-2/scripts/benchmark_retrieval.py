"""Test retrieval and log search latency.

Two tests:
1. A fixed set of queries (short, long, typos, off-topic, filtered). Each is run
   several times to measure latency. A rough relevance check compares the arXiv
   category of the results with the categories we would expect.
2. Known-item search: take random chunks from the database and use them as queries,
   then check whether that chunk (or its paper) comes back. Two versions run:
   - first sentence of the chunk (a short query, harder)
   - the full chunk text (the chunk should find itself, so this checks the index)

Run:  python scripts/benchmark_retrieval.py
Results: logs/retrieval_benchmark.json (every run) and logs/benchmark_summary.md (latest run).
"""

import argparse
import random
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embedding import (  # noqa: E402
    DEFAULT_MODEL, append_json_log, environment_info, load_model,
)
from src.search import timed_search  # noqa: E402
from src.vector_store import DEFAULT_COLLECTION, VectorStore  # noqa: E402

LONG_QUERY = (
    "We study how neural networks trained on large collections of text and images can "
    "be adapted to new tasks with very little labelled data, and we compare fine-tuning, "
    "prompting and lightweight adapter methods in terms of accuracy, training cost and "
    "robustness when the test data is different from the training data."
)

# 'expected' lists arXiv categories that would be reasonable for the topic.
# Papers can belong to several categories, so this is only a rough check.
QUERIES = [
    {"label": "one keyword", "query": "transformers", "expected": None},
    {"label": "question", "query": "How do large language models handle long context windows?",
     "expected": ["cs.CL", "cs.LG", "cs.AI"]},
    {"label": "computer vision", "query": "object detection with convolutional neural networks",
     "expected": ["cs.CV"]},
    {"label": "learning theory", "query": "generalization bounds for deep neural networks",
     "expected": ["cs.LG", "stat.ML"]},
    {"label": "reinforcement learning", "query": "reinforcement learning with policy gradient methods",
     "expected": ["cs.LG", "cs.AI"]},
    {"label": "image generation", "query": "image generation using diffusion models",
     "expected": ["cs.CV", "cs.LG"]},
    {"label": "statistics", "query": "Bayesian inference and variational approximation",
     "expected": ["stat.ML", "cs.LG"]},
    {"label": "translation", "query": "machine translation with neural sequence models",
     "expected": ["cs.CL"]},
    {"label": "long query (about 60 words)", "query": LONG_QUERY,
     "expected": ["cs.CL", "cs.LG", "cs.AI", "cs.CV"]},
    {"label": "spelling mistakes", "query": "transfomer atention mechanizm for sequense modeling",
     "expected": ["cs.CL", "cs.LG", "cs.CV"]},
    {"label": "off-topic", "query": "best recipe for chocolate cake", "expected": None},
    {"label": "filter: cs.CV", "query": "object detection with convolutional neural networks",
     "expected": ["cs.CV"], "where": {"category": "cs.CV"}},
    {"label": "filter: cs.CL", "query": "language models", "expected": ["cs.CL"],
     "where": {"category": "cs.CL"}},
]


def stats_ms(values) -> dict:
    a = np.asarray(values, dtype=float)
    return {"mean": round(float(a.mean()), 2), "p50": round(float(np.percentile(a, 50)), 2),
            "p95": round(float(np.percentile(a, 95)), 2), "min": round(float(a.min()), 2),
            "max": round(float(a.max()), 2)}


def run_fixed_queries(model, store, top_k: int, repeats: int) -> list[dict]:
    rows = []
    for q in QUERIES:
        where = q.get("where")
        totals, embeds, searches = [], [], []
        results = []
        for _ in range(repeats):
            results, t = timed_search(q["query"], model, store, top_k, where)
            totals.append(t["total_ms"])
            embeds.append(t["embed_ms"])
            searches.append(t["query_ms"])
        match = None
        if q["expected"] and results:
            match = round(sum(r["metadata"].get("category") in q["expected"] for r in results)
                          / len(results), 2)
        rows.append({
            "label": q["label"], "query_chars": len(q["query"]), "results": len(results),
            "top_score": results[0]["score"] if results else None,
            "category_match": match, "total_ms": stats_ms(totals),
            "embed_ms_mean": round(float(np.mean(embeds)), 2),
            "search_ms_mean": round(float(np.mean(searches)), 2),
        })
    return rows


def first_sentence(text: str) -> str:
    sentence = re.split(r"(?<=[.!?])\s", text, maxsplit=1)[0]
    return (sentence if len(sentence) >= 30 else text)[:250]


def run_known_item(model, store, top_k: int, samples: int, seed: int, mode: str) -> dict:
    """mode is 'first-sentence' or 'full-chunk'. The same chunks are used for both."""
    data = store.collection.get(include=["documents", "metadatas"])
    pool = [(i, d, m) for i, d, m in zip(data["ids"], data["documents"], data["metadatas"])
            if len(d) >= 100]
    rng = random.Random(seed)
    picked = rng.sample(pool, min(samples, len(pool)))

    ranks, doc_hits, totals = [], 0, []
    for chunk_id, text, meta in picked:
        query = first_sentence(text) if mode == "first-sentence" else text
        results, t = timed_search(query, model, store, top_k)
        totals.append(t["total_ms"])
        ids = [r["id"] for r in results]
        ranks.append(ids.index(chunk_id) + 1 if chunk_id in ids else None)
        doc_hits += any(r["metadata"].get("doc_id") == meta.get("doc_id") for r in results)

    n = len(picked)
    return {
        "mode": mode, "samples": n, "top_k": top_k,
        "recall_at_1": round(sum(r == 1 for r in ranks) / n, 3),
        f"recall_at_{top_k}": round(sum(r is not None for r in ranks) / n, 3),
        "mrr": round(sum(1 / r for r in ranks if r) / n, 3),
        f"paper_recall_at_{top_k}": round(doc_hits / n, 3),
        "total_ms": stats_ms(totals),
    }


def write_summary(path, header: dict, rows: list[dict], known: dict) -> str:
    lines = [
        "# Retrieval Benchmark (latest run)", "",
        f"- Model: `{header['model']}` on `{header['device']}`",
        f"- Chunks in database: {header['chunks']}",
        f"- top_k: {header['top_k']}, repeats per query: {header['repeats']}", "",
        "## Fixed queries", "",
        "| Query type | Chars | Mean ms | p50 ms | p95 ms | Embed ms | Search ms | Top score | Category match |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        match = "-" if r["category_match"] is None else r["category_match"]
        score = "-" if r["top_score"] is None else r["top_score"]
        lines.append(
            f"| {r['label']} | {r['query_chars']} | {r['total_ms']['mean']} | {r['total_ms']['p50']} "
            f"| {r['total_ms']['p95']} | {r['embed_ms_mean']} | {r['search_ms_mean']} | {score} | {match} |"
        )
    k = header["top_k"]
    a, b = known["first_sentence"], known["full_chunk"]
    lines += [
        "", "Category match is the share of results whose arXiv category is one we would "
        "expect for the topic. It is a rough check only, because papers can have several categories.",
        "", "## Known-item search", "",
        f"{a['samples']} random chunks. The query is either the first sentence of the chunk "
        "or the full chunk text.", "",
        "| Metric | First sentence | Full chunk |", "|---|---|---|",
        f"| Recall@1 | {a['recall_at_1']} | {b['recall_at_1']} |",
        f"| Recall@{k} | {a[f'recall_at_{k}']} | {b[f'recall_at_{k}']} |",
        f"| MRR | {a['mrr']} | {b['mrr']} |",
        f"| Paper recall@{k} | {a[f'paper_recall_at_{k}']} | {b[f'paper_recall_at_{k}']} |",
        f"| Latency mean / p50 / p95 (ms) | {a['total_ms']['mean']} / {a['total_ms']['p50']} / "
        f"{a['total_ms']['p95']} | {b['total_ms']['mean']} / {b['total_ms']['p50']} / "
        f"{b['total_ms']['p95']} |",
    ]
    text = "\n".join(lines) + "\n"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")
    return text


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--db-path", default="chroma_db")
    p.add_argument("--collection", default=DEFAULT_COLLECTION)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--device", default=None)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--known-item-samples", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log", default="logs/retrieval_benchmark.json")
    p.add_argument("--summary", default="logs/benchmark_summary.md")
    args = p.parse_args()

    store = VectorStore(args.db_path, args.collection, args.model)
    if store.count() == 0:
        sys.exit("The collection is empty. Run scripts/ingest.py first.")
    model, info = load_model(args.model, args.device)
    print(f"{store.count()} chunks | model on {info['device']}")

    for _ in range(3):  # warm-up searches, not timed
        timed_search("warm up query", model, store, args.top_k)

    print("Running fixed queries...")
    rows = run_fixed_queries(model, store, args.top_k, args.repeats)
    print("Running known-item search...")
    known = {
        "first_sentence": run_known_item(model, store, args.top_k, args.known_item_samples,
                                         args.seed, "first-sentence"),
        "full_chunk": run_known_item(model, store, args.top_k, args.known_item_samples,
                                     args.seed, "full-chunk"),
    }

    header = {"model": args.model, "device": info["device"], "chunks": store.count(),
              "top_k": args.top_k, "repeats": args.repeats}
    summary = write_summary(args.summary, header, rows, known)
    append_json_log(args.log, {**header, "fixed_queries": rows, "known_item": known,
                               "environment": environment_info()})
    print("\n" + summary)
    print(f"Saved {args.log} and {args.summary}")


if __name__ == "__main__":
    main()
