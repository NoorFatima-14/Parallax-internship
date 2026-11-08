"""Ranking metrics, checked against values worked out by hand."""

import math

import pytest

from evals.metrics import (dedupe, hit_at_k, ideal_precision_at_k, mean_ci, ndcg_at_k, normalize_doc_id,
                           precision_at_k, recall_at_k, reciprocal_rank)

RANKED = ["a", "b", "c", "d", "e"]
RELEVANT = {"b", "e", "z"}  # "z" is never retrieved


def test_precision_at_k_by_hand():
    assert precision_at_k(RANKED, RELEVANT, 1) == 0.0
    assert precision_at_k(RANKED, RELEVANT, 2) == 0.5
    assert precision_at_k(RANKED, RELEVANT, 3) == pytest.approx(1 / 3)
    assert precision_at_k(RANKED, RELEVANT, 5) == 0.4


def test_recall_at_k_by_hand():
    assert recall_at_k(RANKED, RELEVANT, 1) == 0.0
    assert recall_at_k(RANKED, RELEVANT, 2) == pytest.approx(1 / 3)
    assert recall_at_k(RANKED, RELEVANT, 5) == pytest.approx(2 / 3)  # "z" can never be found


def test_hit_rate_and_reciprocal_rank():
    assert hit_at_k(RANKED, RELEVANT, 1) == 0.0 and hit_at_k(RANKED, RELEVANT, 2) == 1.0
    assert reciprocal_rank(RANKED, RELEVANT) == 0.5
    assert reciprocal_rank(RANKED, {"e"}) == 0.2
    assert reciprocal_rank(RANKED, {"e"}, k=3) == 0.0   # first relevant is beyond the cutoff
    assert reciprocal_rank(RANKED, {"nothing"}) == 0.0


def test_ndcg_by_hand():
    dcg = 1 / math.log2(3) + 1 / math.log2(6)                    # relevant at rank 2 and rank 5
    ideal = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)  # 3 relevant papers, perfect order
    assert ndcg_at_k(RANKED, RELEVANT, 5) == pytest.approx(dcg / ideal)
    assert ndcg_at_k(["b", "e", "z"], RELEVANT, 3) == pytest.approx(1.0)  # a perfect ranking scores 1
    assert ndcg_at_k(["x", "y"], RELEVANT, 2) == 0.0


def test_perfect_and_worst_rankings():
    assert precision_at_k(["b"], {"b"}, 1) == 1.0 and recall_at_k(["b"], {"b"}, 1) == 1.0
    assert precision_at_k(["x"], {"b"}, 1) == 0.0 and recall_at_k(["x"], {"b"}, 1) == 0.0


def test_precision_divides_by_k_even_when_fewer_results_exist():
    assert precision_at_k(["a"], {"a"}, 5) == 0.2   # one result, one relevant, K = 5
    assert precision_at_k([], {"a"}, 5) == 0.0
    assert recall_at_k([], {"a"}, 5) == 0.0


def test_ideal_precision_is_capped_by_the_number_of_relevant_papers():
    assert ideal_precision_at_k(1, 5) == 0.2    # one relevant paper: P@5 can never exceed 0.2
    assert ideal_precision_at_k(3, 5) == 0.6
    assert ideal_precision_at_k(10, 5) == 1.0


def test_dedupe_keeps_first_occurrence_order():
    assert dedupe(["a", "a", "b", "a", "c", "b"]) == ["a", "b", "c"]
    assert dedupe([]) == []


def test_chunk_level_lists_count_every_chunk_but_each_paper_once_for_recall():
    chunks = ["a", "a", "b"]                        # two chunks of paper a, one of b
    assert precision_at_k(chunks, {"a"}, 3) == pytest.approx(2 / 3)
    assert recall_at_k(chunks, {"a"}, 3) == 1.0     # not 2.0
    assert precision_at_k(dedupe(chunks), {"a"}, 2) == 0.5


@pytest.mark.parametrize("k", [0, -1, 1.5, "3", None])
def test_invalid_k_is_rejected(k):
    for fn in (precision_at_k, recall_at_k, hit_at_k, ndcg_at_k):
        with pytest.raises(ValueError):
            fn(RANKED, RELEVANT, k)


def test_recall_without_relevant_documents_is_an_error_not_a_silent_zero():
    with pytest.raises(ValueError, match="no relevant"):
        recall_at_k(RANKED, set(), 3)
    with pytest.raises(ValueError):
        ndcg_at_k(RANKED, set(), 3)
    assert precision_at_k(RANKED, set(), 3) == 0.0


@pytest.mark.parametrize("raw,expected", [
    ("http://arxiv.org/abs/2601.00001v1::3", "2601.00001v1"),
    ("https://arxiv.org/abs/2601.00001v1", "2601.00001v1"),
    ("2601.00001v1::0", "2601.00001v1"),
    ("  2601.00001v1 ", "2601.00001v1"),
    ("cs/0501001v2", "cs/0501001v2"),
    (None, ""),
])
def test_normalize_doc_id(raw, expected):
    assert normalize_doc_id(raw) == expected


# ------------------------------------------------------------------ mean_ci
def test_mean_ci_basic_properties():
    stat = mean_ci([0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    assert stat["n"] == 8 and stat["mean"] == 0.625
    assert 0.0 <= stat["lo"] <= stat["mean"] <= stat["hi"] <= 1.0


def test_mean_ci_is_repeatable_and_handles_tiny_inputs():
    values = [0.1, 0.5, 0.9, 0.3]
    assert mean_ci(values) == mean_ci(values)
    assert mean_ci([]) == {"mean": None, "lo": None, "hi": None, "n": 0}
    assert mean_ci([0.7]) == {"mean": 0.7, "lo": 0.7, "hi": 0.7, "n": 1}
    constant = mean_ci([1.0, 1.0, 1.0])
    assert constant["lo"] == constant["hi"] == 1.0


def test_mean_ci_gets_narrower_with_more_data():
    small = mean_ci([0, 1] * 5)
    large = mean_ci([0, 1] * 200)
    assert (large["hi"] - large["lo"]) < (small["hi"] - small["lo"])


# ------------------------------------------------------------------ Wilson interval (used for pass rates in the e2e report)
def test_wilson_interval_matches_known_values():
    from evals.metrics import wilson_interval

    all_ok = wilson_interval(10, 10)
    assert all_ok["p"] == 1.0 and all_ok["hi"] == 1.0 and abs(all_ok["lo"] - 0.7225) < 1e-3
    none_ok = wilson_interval(0, 10)
    assert none_ok["p"] == 0.0 and none_ok["lo"] == 0.0 and abs(none_ok["hi"] - 0.2775) < 1e-3
    half = wilson_interval(5, 10)
    assert abs(half["lo"] - 0.2366) < 1e-3 and abs(half["hi"] - 0.7634) < 1e-3
    assert wilson_interval(0, 0) == {"p": None, "lo": None, "hi": None, "n": 0}


def test_wilson_interval_rejects_impossible_counts():
    from evals.metrics import wilson_interval

    with pytest.raises(ValueError):
        wilson_interval(11, 10)
    with pytest.raises(ValueError):
        wilson_interval(-1, 10)
