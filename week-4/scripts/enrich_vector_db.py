"""Write the NLP metadata into the ChromaDB chunks, so searches can be filtered.

Run:  python scripts/enrich_vector_db.py
Adds to every chunk: topic_id, topic_label, topic_core, topic_sim, short_doc, doc_words,
has_dataset, datasets, published_year, published_ym, and a ds_<name> flag per frequent dataset.
Safe to run again: values are replaced, old dataset flags are removed.
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nlpmeta import cli, paths  # noqa: E402
from nlpmeta.enrich import apply_to_chroma, build_doc_metadata  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    cli.add_path_args(p)
    p.add_argument("--min-dataset-docs", type=int, default=3,
                   help="a dataset gets its own filter flag when it appears in at least this many documents")
    args = p.parse_args()
    P = cli.resolve(args)

    corpus = pd.read_parquet(cli.need(P.corpus, "Build the Week 1 clean corpus first."))
    doc_topics = pd.read_parquet(cli.need(P.doc_topics, "Run scripts/run_topic_model.py first."))
    entities = pd.read_parquet(cli.need(P.entities, "Run scripts/run_ner.py first."))
    store = cli.open_store(args)

    meta, index = build_doc_metadata(doc_topics, entities, corpus, args.min_dataset_docs)
    start = time.perf_counter()
    stats = apply_to_chroma(store.collection, meta)
    seconds = time.perf_counter() - start
    cli.save_json(P.datasets_index, index)

    col = store.collection
    with_topic = len(col.get(where={"topic_id": {"$gte": 0}}, include=[])["ids"])
    with_dataset = len(col.get(where={"has_dataset": True}, include=[])["ids"])
    core = len(col.get(where={"topic_core": True}, include=[])["ids"])
    print(f"Updated {stats['updated']} of {stats['chunks']} chunks in {seconds:.1f} s "
          f"({stats['no_nlp_data']} chunks had no NLP data, {stats['stale_flags_removed']} old flags removed)")
    print(f"Chunks with a topic: {with_topic} | in a core cluster: {core} | with a dataset mention: {with_dataset}")
    print(f"Dataset filter flags: {len(index)} (datasets in at least {args.min_dataset_docs} documents)")
    paths.week2().append_json_log(P.logs / "enrichment_log.json", {
        **stats, "seconds": round(seconds, 2), "chunks_with_topic": with_topic, "chunks_in_core_cluster": core,
        "chunks_with_dataset": with_dataset, "dataset_flags": len(index), "min_dataset_docs": args.min_dataset_docs})
    sample = col.get(limit=3, include=["metadatas"])
    for cid, m in zip(sample["ids"], sample["metadatas"]):
        print(f"\nExample chunk {cid}:\n  " + ", ".join(f"{k}={v}" for k, v in sorted(m.items()) if k != "title"))


if __name__ == "__main__":
    main()
