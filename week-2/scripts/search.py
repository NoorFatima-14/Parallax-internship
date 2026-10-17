
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embedding import DEFAULT_MODEL, load_model  # noqa: E402
from src.search import timed_search  # noqa: E402
from src.vector_store import DEFAULT_COLLECTION, VectorStore  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("query")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--category", default=None, help="only search one arXiv category, e.g. cs.CV")
    p.add_argument("--db-path", default="chroma_db")
    p.add_argument("--collection", default=DEFAULT_COLLECTION)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    store = VectorStore(args.db_path, args.collection, args.model)
    if store.count() == 0:
        sys.exit("The collection is empty. Run scripts/ingest.py first.")
    model, _ = load_model(args.model, args.device)

    where = {"category": args.category} if args.category else None
    try:
        results, timings = timed_search(args.query, model, store, args.top_k, where)
    except ValueError as exc:
        sys.exit(f"Search error: {exc}")

    if not results:
        print("No results.")
    for rank, r in enumerate(results, 1):
        meta = r["metadata"]
        print(f"\n#{rank}  score {r['score']:.3f}  [{meta.get('category', '?')}]  {meta.get('title', '')}")
        print(f"    {r['id']}")
        print(f"    {r['text'][:250]}{'...' if len(r['text']) > 250 else ''}")
    print(f"\nLatency: embed {timings['embed_ms']:.1f} ms + search {timings['query_ms']:.1f} ms "
          f"= {timings['total_ms']:.1f} ms")


if __name__ == "__main__":
    main()
