"""Build the retrieval test set in three steps.

    python scripts/make_retrieval_testset.py known-item          # automatic: title and sentence queries
    python scripts/make_retrieval_testset.py pool-sheet          # writes labels/retrieval_judgments.csv
    (fill the 'relevant' column with y or n, about 30 to 45 minutes for 20 queries x 10 papers)
    python scripts/make_retrieval_testset.py merge-labels        # adds the judged topical queries

The test set is saved in eval_data/retrieval_testset.json. Needs the Week 2 database.
Run all commands from inside week-5/.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from api.paths import add_other_weeks  # noqa: E402

add_other_weeks()
from evals.testset import load_testset, save_testset  # noqa: E402
from evals.testset_builder import (build_known_item, group_by_doc, merge_queries, pool_candidates,  # noqa: E402
                                   read_judgments, write_sheet)

DEFAULT_TESTSET = ROOT / "eval_data" / "retrieval_testset.json"
DEFAULT_SHEET = ROOT / "labels" / "retrieval_judgments.csv"
DEFAULT_QUERIES = ROOT / "eval_data" / "topical_queries.json"


def existing_queries(path: Path) -> list:
    return load_testset(path) if path.is_file() else []


def sheet_has_labels(path: Path) -> bool:
    if not path.is_file():
        return False
    with path.open(newline="", encoding="utf-8") as f:
        return any((row.get("relevant") or "").strip() for row in csv.DictReader(f))


def cmd_known_item(args) -> None:
    from rag.retriever import week2

    db = Path(args.db_path)
    if not db.exists():
        sys.exit(f"Vector database not found at {db}. Run week-2/scripts/ingest.py first.")
    w2 = week2()
    store = w2.VectorStore(db, args.collection or w2.DEFAULT_COLLECTION, w2.DEFAULT_MODEL)
    if store.count() == 0:
        sys.exit("The vector database is empty.")
    data = store.collection.get(include=["documents", "metadatas"])
    docs = group_by_doc(data["ids"], data["documents"], data["metadatas"])
    new = build_known_item(docs, args.docs, args.seed, tuple(args.types))
    if not new:
        sys.exit("No usable papers found (titles are needed for known-item queries).")
    out = Path(args.out)
    kept_topical = [q for q in existing_queries(out) if q["type"] == "topical"]
    save_testset(out, new + kept_topical,
                 notes=f"known-item queries from {len(docs)} papers, seed {args.seed}; {len(kept_topical)} topical queries kept")
    print(f"Saved {len(new)} known-item queries ({len(docs)} papers in the database) and kept "
          f"{len(kept_topical)} topical queries in {out}")


def cmd_pool_sheet(args) -> None:
    from rag.retriever import Retriever

    sheet = Path(args.sheet)
    if sheet_has_labels(sheet) and not args.force:
        sys.exit(f"{sheet} already has judgments. Re-creating it would erase them. Use --force if you really want that.")
    queries = json.loads(Path(args.queries).read_text(encoding="utf-8"))["queries"]
    retriever = Retriever.from_disk(args.db_path, args.collection, None, args.device)
    rows = pool_candidates(lambda q, n: retriever.search(q, n)[0], queries, args.pool_k, seed=args.seed)
    write_sheet(sheet, rows)
    print(f"Wrote {len(rows)} rows ({len(queries)} queries x up to {args.pool_k} papers) to {sheet}\n"
          "Open it in a spreadsheet and put y (the paper is relevant to the query) or n in the 'relevant' column.\n"
          "A paper is relevant if its abstract is clearly about the query topic. Judge the abstract, not the title.")


def cmd_merge_labels(args) -> None:
    topical, report = read_judgments(args.sheet, args.allow_unjudged)
    out = Path(args.out)
    queries = merge_queries(existing_queries(out), topical)
    if not queries:
        sys.exit("Nothing to save: no known-item queries and no relevant papers were marked.")
    save_testset(out, queries, notes=f"{sum(q['type'] == 'known_item' for q in queries)} known-item and "
                                     f"{len(topical)} topical queries; topical relevance judged by hand")
    print(f"Saved {len(queries)} queries to {out}: {report['kept']} topical queries kept of {report['queries_in_sheet']} in the sheet.")
    if report["dropped"]:
        print(f"Dropped (no relevant paper marked): {', '.join(report['dropped'])}")
    if report["unjudged_rows"]:
        print(f"Note: {report['unjudged_rows']} unjudged rows were counted as not relevant.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp, db=True):
        sp.add_argument("--out", default=str(DEFAULT_TESTSET), help="test set file")
        if db:
            sp.add_argument("--db-path", default=str(ROOT.parent / "week-2" / "chroma_db"))
            sp.add_argument("--collection", default=None)
            sp.add_argument("--device", default=None)

    k = sub.add_parser("known-item", help="automatic title and sentence queries")
    common(k)
    k.add_argument("--docs", type=int, default=100, help="papers to use (one title and one sentence query each)")
    k.add_argument("--types", nargs="+", choices=["title", "sentence"], default=["title", "sentence"])
    k.add_argument("--seed", type=int, default=42)
    k.set_defaults(func=cmd_known_item)

    s = sub.add_parser("pool-sheet", help="write the sheet to judge by hand")
    common(s, db=True)
    s.add_argument("--queries", default=str(DEFAULT_QUERIES))
    s.add_argument("--sheet", default=str(DEFAULT_SHEET))
    s.add_argument("--pool-k", type=int, default=10, help="papers per query to judge")
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--force", action="store_true", help="overwrite a sheet that already has judgments")
    s.set_defaults(func=cmd_pool_sheet)

    m = sub.add_parser("merge-labels", help="read the judged sheet and add the topical queries")
    common(m, db=False)
    m.add_argument("--sheet", default=str(DEFAULT_SHEET))
    m.add_argument("--allow-unjudged", action="store_true", help="count blank cells as not relevant")
    m.set_defaults(func=cmd_merge_labels)

    args = p.parse_args()
    try:
        args.func(args)
    except (FileNotFoundError, ValueError) as exc:
        sys.exit(f"Problem: {exc}")


if __name__ == "__main__":
    main()
