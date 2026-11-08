"""Evaluate retrieval: Precision@K, Recall@K, HitRate@K, MRR and nDCG@K on the test set.

    python scripts/evaluate_retrieval.py
    python scripts/evaluate_retrieval.py --ks 1 3 5 10 --limit 20      (quick trial: 20 queries)

Needs the Week 2 database and eval_data/retrieval_testset.json (see scripts/make_retrieval_testset.py).
No API key or internet is needed. Writes logs/retrieval_eval.md (latest run) and appends to logs/retrieval_eval.json.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from api.paths import add_other_weeks  # noqa: E402

add_other_weeks()
from evals.retrieval_eval import DEFAULT_KS, DEFAULT_OVERFETCH, evaluate, json_summary, write_report  # noqa: E402
from evals.testset import load_testset  # noqa: E402
from rag.retriever import Retriever, week2  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--testset", default=str(ROOT / "eval_data" / "retrieval_testset.json"))
    p.add_argument("--ks", type=int, nargs="+", default=list(DEFAULT_KS))
    p.add_argument("--overfetch", type=int, default=DEFAULT_OVERFETCH,
                   help="chunks fetched per paper wanted (so that K different papers can be found)")
    p.add_argument("--limit", type=int, default=None, help="use only the first N queries of each type")
    p.add_argument("--db-path", default=str(ROOT.parent / "week-2" / "chroma_db"))
    p.add_argument("--collection", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--report", default=str(ROOT / "logs" / "retrieval_eval.md"))
    p.add_argument("--log", default=str(ROOT / "logs" / "retrieval_eval.json"))
    args = p.parse_args()

    try:
        queries = load_testset(args.testset)
        retriever = Retriever.from_disk(args.db_path, args.collection, None, args.device)
    except (FileNotFoundError, ValueError) as exc:
        sys.exit(f"Setup problem: {exc}")
    if args.limit:
        seen, kept = {}, []
        for q in queries:
            key = (q["type"], q["subtype"])
            if seen.get(key, 0) < args.limit:
                seen[key] = seen.get(key, 0) + 1
                kept.append(q)
        queries = kept

    def search(query: str, n: int):
        return retriever.search(query, n)[0]

    for _ in range(3):  # warm-up, so the first query does not include one-time start-up cost
        search("warm up query", 5)
    print(f"Evaluating {len(queries)} queries, K = {args.ks} ...")
    summary = evaluate(search, queries, args.ks, args.overfetch)

    header = {"testset": Path(args.testset).name, "chunks": retriever.store.count(),
              "embed_model": retriever.store.model_name}
    text = write_report(args.report, summary, header)
    w2 = week2()
    w2.append_json_log(args.log, {**header, **json_summary(summary), "environment": w2.environment_info()})
    print("\n" + text)
    print(f"Saved {args.report} and {args.log}")


if __name__ == "__main__":
    main()
