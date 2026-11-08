"""Retrieval evaluation: Precision@K, Recall@K and related metrics on a test set.

`evaluate` takes any `search_fn(query, n_chunks) -> list of hits`, so the same code runs on the
real database and, in the tests, on a tiny one.

Two views of the same retrieval run are scored:
- document level: the chunk list is reduced to one entry per paper (first occurrence), then the top K papers are scored.
  This matches how relevance is judged (per paper).
- chunk level: the top K chunks are scored as they come. A chunk is relevant if its paper is. This matches what
  the RAG system really receives (the top chunks), including several chunks from one paper.
"""

import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from evals.metrics import (dedupe, hit_at_k, ideal_precision_at_k, mean_ci, ndcg_at_k, normalize_doc_id,
                           precision_at_k, recall_at_k, reciprocal_rank)

DEFAULT_KS = (1, 3, 5, 10)
DEFAULT_OVERFETCH = 4  # chunks fetched per paper wanted, so that K different papers can usually be found


def doc_id_of(hit: dict) -> str:
    meta = hit.get("metadata") or {}
    return normalize_doc_id(meta.get("doc_id") or hit.get("id"))


def evaluate(search_fn, queries: list[dict], ks=DEFAULT_KS, overfetch: int = DEFAULT_OVERFETCH) -> dict:
    ks = sorted(set(ks))
    if not ks or ks[0] < 1:
        raise ValueError("ks must be positive integers")
    n_fetch = max(ks) * max(1, overfetch)
    rows = []
    for q in queries:
        relevant = set(q["relevant"])
        start = time.perf_counter()
        hits = search_fn(q["query"], n_fetch)
        latency_ms = 1000 * (time.perf_counter() - start)
        chunk_docs = [doc_id_of(h) for h in hits]
        docs = dedupe(chunk_docs)
        row = {"id": q["id"], "query": q["query"], "type": q["type"], "subtype": q.get("subtype", ""),
               "n_relevant": len(relevant), "latency_ms": latency_ms,
               "mrr": reciprocal_rank(docs, relevant), "top_docs": docs[:max(ks)],
               "top_score": hits[0]["score"] if hits else None,
               "doc": {}, "chunk": {}}
        for k in ks:
            row["doc"][k] = {"precision": precision_at_k(docs, relevant, k), "recall": recall_at_k(docs, relevant, k),
                             "hit": hit_at_k(docs, relevant, k), "ndcg": ndcg_at_k(docs, relevant, k)}
            row["chunk"][k] = {"precision": precision_at_k(chunk_docs, relevant, k),
                               "recall": recall_at_k(chunk_docs, relevant, k), "hit": hit_at_k(chunk_docs, relevant, k)}
        row["ideal_precision"] = {k: ideal_precision_at_k(len(relevant), k) for k in ks}
        rows.append(row)
    return {"ks": ks, "overfetch": overfetch, "n_queries": len(rows),
            "groups": summarize(rows, ks), "failures": [r for r in rows if r["doc"][max(ks)]["recall"] == 0.0],
            "rows": rows}


def _group_keys(row: dict) -> list[str]:
    keys = ["all", row["type"]]
    if row["subtype"]:
        keys.append(f"{row['type']}/{row['subtype']}")
    return keys


def summarize(rows: list[dict], ks: list[int]) -> dict:
    by_group = defaultdict(list)
    for row in rows:
        for key in _group_keys(row):
            by_group[key].append(row)
    out = {}
    for key, members in by_group.items():
        latency = np.array([r["latency_ms"] for r in members])
        out[key] = {
            "n": len(members),
            "mean_relevant_papers": float(np.mean([r["n_relevant"] for r in members])),
            "mrr": mean_ci([r["mrr"] for r in members]),
            "ideal_precision": {k: float(np.mean([r["ideal_precision"][k] for r in members])) for k in ks},
            "doc": {k: {m: mean_ci([r["doc"][k][m] for r in members]) for m in ("precision", "recall", "hit", "ndcg")}
                    for k in ks},
            "chunk": {k: {m: mean_ci([r["chunk"][k][m] for r in members]) for m in ("precision", "recall", "hit")}
                      for k in ks},
            "latency_ms": {"mean": float(latency.mean()), "p50": float(np.percentile(latency, 50)),
                           "p95": float(np.percentile(latency, 95))},
        }
    return out


# ------------------------------------------------------------------ report
def _fmt(stat: dict) -> str:
    if stat["mean"] is None:
        return "-"
    if stat["n"] > 1 and stat["lo"] != stat["hi"]:
        return f"{stat['mean']:.3f} ({stat['lo']:.2f}-{stat['hi']:.2f})"
    return f"{stat['mean']:.3f}"


GROUP_NOTES = {
    "all": "All queries together.",
    "known_item": "One relevant paper per query (the paper the query was made from). Precision@K cannot exceed 1/K "
                  "here, so read it together with the ideal value, and use Recall@K and MRR as the main numbers.",
    "known_item/title": "The query is the paper title. Not copied from the abstract text, so this is the realistic known-item case.",
    "known_item/sentence": "The query is one sentence copied from the abstract. This is an easy check and should be close to 1.",
    "topical": "Papers judged relevant by hand. Judged only among the papers that the system itself returned "
               "(pooling), so Recall@K is probably too high, because relevant papers the system never returns were never judged.",
}


def write_report(path, summary: dict, header: dict) -> str:
    ks = summary["ks"]
    L = ["# Retrieval Evaluation (latest run)", "",
         f"- Test set: `{header['testset']}`, {summary['n_queries']} queries",
         f"- Collection: {header['chunks']} chunks, embedding model `{header['embed_model']}`",
         f"- Search depth: {max(ks) * summary['overfetch']} chunks per query (K up to {max(ks)})",
         f"- Metric values are means over queries, with a 95% bootstrap interval in brackets.", "",
         "How to read it: **document level** reduces the results to one entry per paper and scores the top K papers. "
         "**Chunk level** scores the top K chunks as the RAG system receives them (a chunk counts as relevant when its paper is).", ""]
    order = ["all", "known_item", "known_item/title", "known_item/sentence", "topical"]
    for key in order + sorted(set(summary["groups"]) - set(order)):
        g = summary["groups"].get(key)
        if not g:
            continue
        L += [f"## {key} ({g['n']} queries, {g['mean_relevant_papers']:.1f} relevant papers on average)", "",
              GROUP_NOTES.get(key, ""), "",
              f"MRR: {_fmt(g['mrr'])}. Search latency: mean {g['latency_ms']['mean']:.0f} ms, "
              f"p50 {g['latency_ms']['p50']:.0f} ms, p95 {g['latency_ms']['p95']:.0f} ms.", "",
              "| K | Precision@K (doc) | ideal Precision@K | Recall@K (doc) | HitRate@K (doc) | nDCG@K (doc) | Precision@K (chunk) | Recall@K (chunk) |",
              "|---|---|---|---|---|---|---|---|"]
        for k in ks:
            d, c = g["doc"][k], g["chunk"][k]
            L.append(f"| {k} | {_fmt(d['precision'])} | {g['ideal_precision'][k]:.3f} | {_fmt(d['recall'])} | "
                     f"{_fmt(d['hit'])} | {_fmt(d['ndcg'])} | {_fmt(c['precision'])} | {_fmt(c['recall'])} |")
        L.append("")
    failures = summary["failures"]
    L += [f"## Queries with no relevant paper in the top {max(ks)} ({len(failures)} of {summary['n_queries']})", ""]
    if failures:
        L += ["Read these first when looking for what to improve.", ""]
        L += [f"- `{r['id']}` ({r['type']}): {r['query'][:110]}" for r in failures[:15]]
        if len(failures) > 15:
            L.append(f"- ... and {len(failures) - 15} more (see the JSON log)")
    else:
        L.append("None.")
    text = "\n".join(L) + "\n"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")
    return text


def json_summary(summary: dict) -> dict:
    """The summary without the bulky per-query rows (those are saved with only the fields that matter)."""
    rows = [{"id": r["id"], "type": r["type"], "subtype": r["subtype"], "n_relevant": r["n_relevant"],
             "mrr": r["mrr"], "latency_ms": round(r["latency_ms"], 1),
             "recall_doc": {k: v["recall"] for k, v in r["doc"].items()},
             "precision_doc": {k: v["precision"] for k, v in r["doc"].items()}} for r in summary["rows"]]
    return {"ks": summary["ks"], "overfetch": summary["overfetch"], "n_queries": summary["n_queries"],
            "groups": summary["groups"], "failed_query_ids": [r["id"] for r in summary["failures"]], "rows": rows}
