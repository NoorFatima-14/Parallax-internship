"""Semantic search: embed the query, then look it up in ChromaDB."""

import time

MAX_QUERY_CHARS = 2000  # the model ignores text past its token limit anyway


def _prepare_query(query) -> str:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query is empty.")
    return " ".join(query.split())[:MAX_QUERY_CHARS]


def timed_search(query, model, store, top_k: int = 5, where: dict | None = None):
    """Run one search. Returns (results, timings in milliseconds)."""
    query = _prepare_query(query)

    t0 = time.perf_counter()
    vector = model.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
    )[0]
    t1 = time.perf_counter()
    results = store.query(vector, top_k=top_k, where=where)
    t2 = time.perf_counter()

    timings = {
        "embed_ms": 1000 * (t1 - t0),
        "query_ms": 1000 * (t2 - t1),
        "total_ms": 1000 * (t2 - t0),
    }
    return results, timings


def semantic_search(query, model, store, top_k: int = 5, where: dict | None = None):
    """Return the top_k most similar chunks for a text query."""
    return timed_search(query, model, store, top_k, where)[0]
