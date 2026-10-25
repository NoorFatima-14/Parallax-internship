"""Retrieval: reuses the Week 2 vector store, embedding and search code."""

import sys
from pathlib import Path
from types import SimpleNamespace

from rag.config import REPO_DIR

WEEK2_DIR = REPO_DIR / "week-2"
DEFAULT_DB_PATH = WEEK2_DIR / "chroma_db"


def week2() -> SimpleNamespace:
    """Import the Week 2 modules (they live in ../week-2/src)."""
    if not (WEEK2_DIR / "src").is_dir():
        raise FileNotFoundError(f"Week 2 code not found at {WEEK2_DIR}. Week 3 needs the week-2 folder.")
    if str(WEEK2_DIR) not in sys.path:
        sys.path.insert(0, str(WEEK2_DIR))
    from src.embedding import DEFAULT_MODEL, append_json_log, environment_info, load_model
    from src.search import timed_search
    from src.vector_store import DEFAULT_COLLECTION, VectorStore

    return SimpleNamespace(
        DEFAULT_MODEL=DEFAULT_MODEL, append_json_log=append_json_log,
        environment_info=environment_info, load_model=load_model,
        timed_search=timed_search, DEFAULT_COLLECTION=DEFAULT_COLLECTION, VectorStore=VectorStore,
    )


class Retriever:
    """Finds the passages closest to a question."""

    def __init__(self, store, model, top_k: int = 5):
        self.store, self.model, self.top_k = store, model, top_k

    def search(self, query: str, top_k: int = None, where: dict = None):
        """Returns (results, timings). Each result has id, text, metadata, score."""
        return week2().timed_search(query, self.model, self.store, top_k or self.top_k, where)

    def encode(self, texts):
        """Normalized embeddings, used by the hallucination check."""
        return self.model.encode(
            list(texts), normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )

    @classmethod
    def from_disk(cls, db_path=None, collection=None, model_name=None, device=None, top_k=5):
        w2 = week2()
        db_path = Path(db_path or DEFAULT_DB_PATH)
        if not db_path.exists():
            raise FileNotFoundError(
                f"Vector database not found at {db_path}. Run week-2/scripts/ingest.py first."
            )
        model_name = model_name or w2.DEFAULT_MODEL
        store = w2.VectorStore(db_path, collection or w2.DEFAULT_COLLECTION, model_name)
        if store.count() == 0:
            raise ValueError("The vector database is empty. Run week-2/scripts/ingest.py first.")
        model, _ = w2.load_model(model_name, device)
        return cls(store, model, top_k)
