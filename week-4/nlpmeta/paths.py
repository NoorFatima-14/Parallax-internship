"""Locations of the other weeks' code and data, and access to the Week 2 modules."""

import sys
from pathlib import Path
from types import SimpleNamespace

WEEK4 = Path(__file__).resolve().parents[1]
REPO = WEEK4.parent
WEEK1, WEEK2, WEEK3 = REPO / "week-1", REPO / "week-2", REPO / "week-3"

CORPUS = WEEK1 / "data" / "processed" / "clean_corpus.parquet"
DB_PATH = WEEK2 / "chroma_db"
DATA = WEEK4 / "data" / "processed"
LABELS = WEEK4 / "labels"
LOGS = WEEK4 / "logs"

DOC_TOPICS = DATA / "doc_topics.parquet"
TOPICS_TABLE = DATA / "topics.json"
DOC_ENTITIES = DATA / "doc_entities.parquet"
DATASETS_INDEX = DATA / "datasets_index.json"


def _add(path: Path) -> None:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def week2() -> SimpleNamespace:
    """Import the Week 2 modules (they live in ../week-2/src)."""
    if not (WEEK2 / "src").is_dir():
        raise FileNotFoundError(f"Week 2 code not found at {WEEK2}.")
    _add(WEEK2)
    from src.embedding import DEFAULT_MODEL, append_json_log, environment_info, load_model
    from src.search import timed_search
    from src.vector_store import DEFAULT_COLLECTION, VectorStore

    return SimpleNamespace(
        DEFAULT_MODEL=DEFAULT_MODEL, DEFAULT_COLLECTION=DEFAULT_COLLECTION, VectorStore=VectorStore,
        append_json_log=append_json_log, environment_info=environment_info,
        load_model=load_model, timed_search=timed_search,
    )


def add_week3() -> None:
    """Make the Week 3 package (rag) importable."""
    if not (WEEK3 / "rag").is_dir():
        raise FileNotFoundError(f"Week 3 code not found at {WEEK3}.")
    _add(WEEK3)
