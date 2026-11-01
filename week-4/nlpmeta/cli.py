"""Command-line options shared by the Week 4 scripts."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

from nlpmeta import paths


def add_path_args(parser) -> None:
    parser.add_argument("--corpus", default=str(paths.CORPUS), help="Week 1 clean corpus (parquet)")
    parser.add_argument("--db-path", default=str(paths.DB_PATH), help="Week 2 ChromaDB folder")
    parser.add_argument("--collection", default=None)
    parser.add_argument("--data-dir", default=str(paths.DATA))
    parser.add_argument("--labels-dir", default=str(paths.LABELS))
    parser.add_argument("--logs-dir", default=str(paths.LOGS))


def resolve(args) -> SimpleNamespace:
    data, labels, logs = Path(args.data_dir), Path(args.labels_dir), Path(args.logs_dir)
    for folder in (data, labels, logs):
        folder.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        corpus=Path(args.corpus), db=Path(args.db_path), data=data, labels=labels, logs=logs,
        doc_topics=data / "doc_topics.parquet", doc_topics_lda=data / "doc_topics_lda.parquet",
        topics=data / "topics.json", entities=data / "doc_entities.parquet",
        datasets_index=data / "datasets_index.json",
    )


def need(path: Path, hint: str) -> Path:
    if not Path(path).exists():
        sys.exit(f"Missing file: {path}\n{hint}")
    return Path(path)


def open_store(args, collection=None):
    """Open the Week 2 vector store, with a clear message if it is missing or empty."""
    w2 = paths.week2()
    if not Path(args.db_path).exists():
        sys.exit(f"Vector database not found at {args.db_path}. Run week-2/scripts/ingest.py first.")
    store = w2.VectorStore(args.db_path, collection or args.collection or w2.DEFAULT_COLLECTION, w2.DEFAULT_MODEL)
    if store.count() == 0:
        sys.exit("The vector database is empty. Run week-2/scripts/ingest.py first.")
    return store


def save_json(path, data) -> None:
    Path(path).write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
