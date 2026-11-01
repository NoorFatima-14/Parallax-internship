"""Semantic search with filters on the NLP metadata.

Run:  python scripts/search_filtered.py "uncertainty estimation" --topic uncertainty
      python scripts/search_filtered.py "object detection" --dataset COCO --year-from 2026
      python scripts/search_filtered.py "language models" --category cs.CL --exclude-short
      python scripts/search_filtered.py --list-topics        (show topics)
      python scripts/search_filtered.py --list-datasets      (show datasets you can filter on)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nlpmeta import cli, paths  # noqa: E402
from nlpmeta.filters import Filters, build_where, read_datasets_index, read_topics_table, resolve_dataset, resolve_topics  # noqa: E402
from nlpmeta.search import count_matching, filtered_search  # noqa: E402


def add_filter_args(p) -> None:
    p.add_argument("--topic", action="append", default=[], help="topic id or a word from its label (repeatable)")
    p.add_argument("--category", action="append", default=[], help="arXiv category, e.g. cs.CV (repeatable)")
    p.add_argument("--dataset", action="append", default=[], help="dataset name, e.g. COCO (repeatable)")
    p.add_argument("--dataset-mode", choices=["any", "all"], default="any")
    p.add_argument("--year-from", type=int)
    p.add_argument("--year-to", type=int)
    p.add_argument("--core-only", action="store_true", help="only papers that belong to a core topic cluster")
    p.add_argument("--exclude-short", action="store_true", help="leave out the shortest abstracts")


def build_filters(args, P) -> Filters:
    topics, flags = [], []
    if args.topic:
        table = read_topics_table(cli.need(P.topics, "Run scripts/run_topic_model.py first."))
        for t in args.topic:
            topics += resolve_topics(t, table)
    if args.dataset:
        index = read_datasets_index(cli.need(P.datasets_index, "Run scripts/enrich_vector_db.py first."))
        flags = [resolve_dataset(d, index) for d in args.dataset]
    return Filters(topics=sorted(set(topics)), categories=args.category, datasets=flags,
                   datasets_mode=args.dataset_mode, year_from=args.year_from, year_to=args.year_to,
                   core_only=args.core_only, exclude_short=args.exclude_short)


def main() -> None:
    p = argparse.ArgumentParser()
    cli.add_path_args(p)
    p.add_argument("query", nargs="?")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--device", default=None)
    p.add_argument("--list-topics", action="store_true")
    p.add_argument("--list-datasets", action="store_true")
    add_filter_args(p)
    args = p.parse_args()
    P = cli.resolve(args)

    if args.list_topics:
        for t in read_topics_table(cli.need(P.topics, "Run scripts/run_topic_model.py first.")):
            print(f"{t['topic_id']:>3}  {t['size']:>5} docs  {t['label']}")
        return
    if args.list_datasets:
        for key, v in read_datasets_index(cli.need(P.datasets_index, "Run scripts/enrich_vector_db.py first.")).items():
            print(f"{v['name']:<28} {v['docs']:>5} docs")
        return
    if not args.query:
        p.error("a query is required (or use --list-topics / --list-datasets)")

    try:
        filters = build_filters(args, P)
    except ValueError as exc:
        sys.exit(f"Filter problem: {exc}")
    store = cli.open_store(args)
    w2 = paths.week2()
    model, _ = w2.load_model(w2.DEFAULT_MODEL, args.device)
    matching = count_matching(store, build_where(filters))
    try:
        results, timings, where = filtered_search(args.query, model, store, filters, args.top_k)
    except ValueError as exc:
        sys.exit(f"Search problem: {exc}")

    print(f"Filter: {where}\nChunks that pass the filter: {matching} of {store.count()}")
    if not results:
        print("No results.")
    for rank, r in enumerate(results, 1):
        m = r["metadata"]
        print(f"\n#{rank}  score {r['score']:.3f}  [{m.get('category', '?')}]  topic {m.get('topic_id', '?')}: "
              f"{m.get('topic_label', '')}  | {m.get('published_year', '?')}")
        print(f"    {m.get('title', '')[:100]}")
        if m.get("datasets"):
            print(f"    datasets: {m['datasets']}")
        print(f"    {r['text'][:200]}...")
    print(f"\nLatency: embed {timings['embed_ms']:.1f} ms + search {timings['query_ms']:.1f} ms")


if __name__ == "__main__":
    main()
