"""Semantic search with metadata filters."""

from nlpmeta.filters import Filters, build_where
from nlpmeta.paths import week2


def count_matching(store, where) -> int:
    """How many chunks pass the filter (without running a vector search)."""
    if where is None:
        return store.count()
    return len(store.collection.get(where=where, include=[])["ids"])


def filtered_search(query: str, model, store, filters: Filters = None, top_k: int = 5):
    """Returns (results, timings, where). Results are the same shape as Week 2 search results."""
    where = build_where(filters) if filters else None
    results, timings = week2().timed_search(query, model, store, top_k, where)
    return results, timings, where
