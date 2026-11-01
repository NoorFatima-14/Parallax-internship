"""Small, dependency-free evaluation helpers: topic quality and classification scores."""

import math

import numpy as np


def wilson_interval(successes: int, n: int, z: float = 1.96):
    """95% confidence interval for a proportion. Returns (low, high). Safe for small n."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def prf(tp: int, fp: int, fn: int) -> dict:
    """Precision, recall and F1 from counts (0.0 when a denominator is 0)."""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def topic_diversity(topic_words: dict, top_n: int = 10) -> float:
    """Share of distinct words among all topics' top words (1.0 = no word is repeated)."""
    words = [w for ws in topic_words.values() for w in ws[:top_n]]
    return len(set(words)) / len(words) if words else 0.0


def npmi_coherence(topic_words: dict, binary_matrix, vocabulary: dict, top_n: int = 10):
    """NPMI coherence of topics against document co-occurrence (range -1 to 1, higher is better).

    binary_matrix: documents x terms (scipy sparse, 0/1) from a reference corpus.
    vocabulary: term -> column. Words that are not in the vocabulary are skipped.
    Returns (mean over topics, {topic_id: score}).
    """
    n_docs = binary_matrix.shape[0]
    scores = {}
    for tid, words in topic_words.items():
        cols = [vocabulary[w] for w in words[:top_n] if w in vocabulary]
        if len(cols) < 2:
            continue
        sub = binary_matrix[:, cols].toarray().astype(float)
        df = sub.sum(axis=0)
        pair_scores = []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                p_i, p_j = df[i] / n_docs, df[j] / n_docs
                p_ij = float((sub[:, i] * sub[:, j]).sum()) / n_docs
                if p_ij == 0:
                    pair_scores.append(-1.0)
                elif p_ij >= 1:
                    pair_scores.append(1.0)
                else:
                    pair_scores.append(math.log(p_ij / (p_i * p_j)) / -math.log(p_ij))
        scores[tid] = float(np.mean(pair_scores))
    mean = float(np.mean(list(scores.values()))) if scores else float("nan")
    return mean, scores
