"""Ranking metrics.

Every function takes a ranked list of document ids (best first) and the set of relevant ids.
The list may contain the same id more than once (when it is a list of chunks); then
`precision_at_k` counts every chunk, and `recall_at_k` counts each relevant document once.
For one-result-per-document metrics, pass `dedupe(ranked)`.

    Precision@K = relevant items in the top K / K
    Recall@K    = distinct relevant documents in the top K / all relevant documents
    HitRate@K   = 1 if at least one relevant document is in the top K, else 0
    MRR         = 1 / rank of the first relevant document (0 if none)
    nDCG@K      = discounted gain of the ranking / the gain of a perfect ranking (binary relevance)

Precision@K divides by K even when fewer than K results exist (a filter may leave few chunks).
"""

import math
import re

import numpy as np


def normalize_doc_id(value) -> str:
    """'http://arxiv.org/abs/2601.00001v1::3' -> '2601.00001v1'. Works for ids, doc ids and plain ids."""
    text = re.sub(r"^https?://arxiv\.org/abs/", "", str(value or "").strip())
    return text.split("::")[0]


def dedupe(ranked: list) -> list:
    """Keep the first occurrence of each id, in order."""
    seen, out = set(), []
    for item in ranked:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _check_k(k: int) -> None:
    if not isinstance(k, int) or k < 1:
        raise ValueError("k must be an integer of at least 1")


def _check_relevant(relevant) -> None:
    if not relevant:
        raise ValueError("The query has no relevant documents, so recall is undefined. Remove it from the test set.")


def precision_at_k(ranked: list, relevant: set, k: int) -> float:
    _check_k(k)
    return sum(item in relevant for item in ranked[:k]) / k


def recall_at_k(ranked: list, relevant: set, k: int) -> float:
    _check_k(k)
    _check_relevant(relevant)
    return len(set(ranked[:k]) & set(relevant)) / len(relevant)


def hit_at_k(ranked: list, relevant: set, k: int) -> float:
    _check_k(k)
    return float(any(item in relevant for item in ranked[:k]))


def reciprocal_rank(ranked: list, relevant: set, k: int | None = None) -> float:
    for rank, item in enumerate(ranked[:k] if k else ranked, 1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: list, relevant: set, k: int) -> float:
    """Binary-relevance nDCG. `ranked` should be one entry per document (use dedupe first)."""
    _check_k(k)
    _check_relevant(relevant)
    dcg = sum(1.0 / math.log2(rank + 1) for rank, item in enumerate(ranked[:k], 1) if item in relevant)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(k, len(relevant)) + 1))
    return dcg / ideal


def ideal_precision_at_k(n_relevant: int, k: int) -> float:
    """The best Precision@K a perfect system can reach: a query with one relevant paper can never exceed 1/K."""
    _check_k(k)
    return min(k, n_relevant) / k


def mean_ci(values, level: float = 0.95, n_boot: int = 2000, seed: int = 0) -> dict:
    """Mean with a bootstrap percentile interval. Returns {'mean', 'lo', 'hi', 'n'}."""
    a = np.asarray(list(values), dtype=float)
    if a.size == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    mean = float(a.mean())
    if a.size == 1 or float(a.std()) == 0.0:
        return {"mean": mean, "lo": mean, "hi": mean, "n": int(a.size)}
    rng = np.random.default_rng(seed)
    means = rng.choice(a, size=(n_boot, a.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [50 * (1 - level), 100 - 50 * (1 - level)])
    return {"mean": mean, "lo": float(lo), "hi": float(hi), "n": int(a.size)}


def wilson_interval(successes: int, n: int, z: float = 1.96) -> dict:
    """95% Wilson interval for a proportion (better than the normal one for small n or rates near 0 or 1).

    Returns {'p', 'lo', 'hi', 'n'}. With n = 0 everything is None.
    """
    if n <= 0:
        return {"p": None, "lo": None, "hi": None, "n": 0}
    if not 0 <= successes <= n:
        raise ValueError("successes must be between 0 and n")
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return {"p": p, "lo": max(0.0, centre - half), "hi": min(1.0, centre + half), "n": n}
