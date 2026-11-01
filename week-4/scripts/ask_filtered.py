"""Ask the Week 3 RAG system a question, restricted to papers selected by the NLP metadata.

Run:  python scripts/ask_filtered.py "How is uncertainty estimated?" --topic uncertainty
Needs the Groq key in week-3/.env (see week-3/README.md).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nlpmeta import cli, paths  # noqa: E402
from nlpmeta.filters import build_where  # noqa: E402
from search_filtered import add_filter_args, build_filters  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    cli.add_path_args(p)
    p.add_argument("question")
    p.add_argument("--provider", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--no-strict", action="store_true", help="show answers even if the checks fail")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--min-score", type=float, default=None, help="out-of-domain threshold (default: Week 3 setting)")
    add_filter_args(p)
    args = p.parse_args()
    P = cli.resolve(args)

    try:
        where = build_where(build_filters(args, P))
    except ValueError as exc:
        sys.exit(f"Filter problem: {exc}")
    paths.add_week3()
    from rag.app import build_pipeline
    from rag.config import ConfigError, RAGConfig

    config = RAGConfig(top_k=args.top_k, strict=not args.no_strict)
    if args.min_score is not None:
        config.min_score = args.min_score
    try:
        pipeline = build_pipeline(args.provider, args.model, str(P.db), args.collection, None, None, config, None)
    except (ConfigError, FileNotFoundError, ValueError) as exc:
        sys.exit(f"Setup problem: {exc}")

    result = pipeline.ask(args.question, where=where)
    print(f"Filter: {where}\n\n{result['answer']}")
    for s in result["sources"]:
        print(f"  [{s['n']}] {s['title'][:80]} (arXiv {s['doc_id']}, {s['category']}, score {s['score']})")
    t = result["timings_ms"]
    print(f"\n  status: {result['status']} | retrieval {t.get('retrieval_ms', '-')} ms | "
          f"llm {t.get('llm_ms', '-')} ms | total {t['total_ms']} ms")


if __name__ == "__main__":
    main()
