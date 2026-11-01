"""Discover the themes of the corpus with BERTopic and compare with an LDA baseline.

Run:  python scripts/run_topic_model.py
Uses the chunk embeddings already stored in the Week 2 database (nothing is embedded again).
Writes data/processed/doc_topics.parquet, topics.json and logs/topic_model_comparison.md.
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nlpmeta import cli, paths  # noqa: E402
from nlpmeta.topics import align_embeddings, build_topics, load_doc_embeddings, quality_metrics  # noqa: E402


def summarize(info, topic_table, metrics) -> dict:
    sizes = [t["size"] for t in topic_table]
    return {"method": info["method"], "topics": info["n_topics"],
            "core_share": round(info["core_share"], 3), "median_size": int(np.median(sizes)) if sizes else 0,
            "npmi": metrics["npmi"], "diversity": metrics["diversity"], "seconds": info["seconds"]}


def main() -> None:
    p = argparse.ArgumentParser()
    cli.add_path_args(p)
    p.add_argument("--method", choices=["bertopic", "both"], default="both")
    p.add_argument("--min-topic-size", type=int, default=25)
    p.add_argument("--nr-topics", type=int, default=None, help="merge BERTopic topics down to this number")
    p.add_argument("--lda-topics", type=int, default=None, help="default: same number as BERTopic")
    p.add_argument("--short-percentile", type=float, default=10, help="documents shorter than this percentile count as short")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    P = cli.resolve(args)

    corpus = pd.read_parquet(cli.need(P.corpus, "Build the Week 1 clean corpus first."))
    store = cli.open_store(args)
    doc_ids, matrix = load_doc_embeddings(store.collection)
    emb, have = align_embeddings(corpus["id"].tolist(), doc_ids, matrix)
    if not have.all():
        print(f"Warning: {int((~have).sum())} documents have no vectors in the database and are skipped.")
        corpus, emb = corpus[have].reset_index(drop=True), emb[have]
    print(f"{len(corpus)} documents, vector size {emb.shape[1]}")

    print("Fitting BERTopic...")
    doc_table, topic_table, info, words, texts = build_topics(
        corpus, emb, "bertopic", args.min_topic_size, args.nr_topics, args.short_percentile, seed=args.seed)
    metrics, per_topic = quality_metrics(words, texts)
    for row in topic_table:
        score = per_topic.get(row["topic_id"])
        row["npmi"] = None if score is None or math.isnan(score) else round(score, 3)
    rows = [summarize(info, topic_table, metrics)]
    doc_table.to_parquet(P.doc_topics, index=False)
    cli.save_json(P.topics, {"info": info, "metrics": metrics, "topics": topic_table})

    if args.method == "both":
        k = args.lda_topics or info["n_topics"]
        print(f"Fitting LDA with {k} topics...")
        lda_table, lda_topics, lda_info, lda_words, _ = build_topics(corpus, emb, "lda", n_lda_topics=k, seed=args.seed)
        lda_metrics, _ = quality_metrics(lda_words, texts)
        rows.append(summarize(lda_info, lda_topics, lda_metrics))
        lda_table.to_parquet(P.doc_topics_lda, index=False)

    lines = ["# Topic Model Comparison (latest run)", "",
             f"- Documents: {info['n_docs']}. Short documents (under {info['short_threshold_words']:.0f} words): "
             f"{info['short_docs']}. They are not used to fit BERTopic and are assigned to the nearest topic afterwards.",
             "- NPMI coherence: -1 to 1, higher is better. Diversity: share of distinct words among all topics' top words.",
             "- Core share: share of documents that belong to a cluster found by the model. The rest "
             "(outliers and short documents) are assigned to the nearest topic by their vectors and marked `topic_core = False`.", "",
             "| Method | Topics | Core share | Median topic size | NPMI | Diversity | Seconds |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['method']} | {r['topics']} | {r['core_share']} | {r['median_size']} | "
                     f"{r['npmi']} | {r['diversity']} | {r['seconds']} |")
    lines += ["", "## Topics (BERTopic, largest first)", "", "| Id | Docs | Label | NPMI |", "|---|---|---|---|"]
    for t in topic_table[:40]:
        lines.append(f"| {t['topic_id']} | {t['size']} | {t['label']} | {t['npmi']} |")
    text = "\n".join(lines) + "\n"
    (P.logs / "topic_model_comparison.md").write_text(text, encoding="utf-8")
    paths.week2().append_json_log(P.logs / "topic_model_runs.json", {
        "settings": {k: v for k, v in vars(args).items() if k in ("method", "min_topic_size", "nr_topics", "seed")},
        "results": rows})
    print("\n" + text)


if __name__ == "__main__":
    main()
