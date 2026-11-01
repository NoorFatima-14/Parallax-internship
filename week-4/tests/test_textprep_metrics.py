import math

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from nlpmeta.metrics import npmi_coherence, prf, topic_diversity, wilson_interval
from nlpmeta.textprep import clean_for_topics, make_vectorizer, word_count


# --- text preparation ---
def test_clean_removes_math_urls_and_latex_but_keeps_acronyms():
    text = "We study $\\alpha$-divergence in GPT-4 at https://x.org with \\textbf{bold} BERT and 3D-CNN."
    clean = clean_for_topics(text)
    assert "alpha" not in clean and "https" not in clean and "textbf" not in clean
    assert "GPT-4" in clean and "BERT" in clean and "3D-CNN" in clean


@pytest.mark.parametrize("value", [None, 5, ""])
def test_clean_handles_non_text(value):
    assert clean_for_topics(value) == ""


def test_word_count():
    assert word_count("one two  three") == 3 and word_count(None) == 0


def test_vectorizer_keeps_jargon_and_drops_boilerplate():
    v = make_vectorizer(ngram_range=(1, 1), min_df=1, max_df=1.0)
    v.fit(["We propose a novel Vision-Language model with GPT-4 on 3D data and the results"])
    vocab = set(v.vocabulary_)
    assert {"vision-language", "gpt-4", "3d", "data"} <= vocab
    assert not ({"propose", "novel", "results", "the", "and"} & vocab)


# --- confidence interval and scores ---
def test_wilson_interval_edges_and_order():
    assert wilson_interval(0, 0) == (0.0, 0.0)
    low, high = wilson_interval(10, 10)
    assert high == 1.0 and 0.6 < low < 1.0
    low2, high2 = wilson_interval(50, 100)
    assert low2 < 0.5 < high2 and (high2 - low2) < (wilson_interval(5, 10)[1] - wilson_interval(5, 10)[0])


def test_prf_counts():
    r = prf(8, 2, 4)
    assert r["precision"] == 0.8 and r["recall"] == pytest.approx(8 / 12)
    assert r["f1"] == pytest.approx(2 * 0.8 * (8 / 12) / (0.8 + 8 / 12))


def test_prf_zero_denominators_do_not_crash():
    assert prf(0, 0, 0)["f1"] == 0.0 and prf(0, 3, 0)["precision"] == 0.0 and prf(0, 0, 3)["recall"] == 0.0


def test_topic_diversity():
    assert topic_diversity({0: ["a", "b"], 1: ["c", "d"]}) == 1.0
    assert topic_diversity({0: ["a", "b"], 1: ["a", "b"]}) == 0.5
    assert topic_diversity({}) == 0.0


# --- NPMI coherence ---
def _matrix():
    # terms: 0=a 1=b 2=c 3=d. a,b always together; c,d always together; the pairs never overlap.
    rows = [[1, 1, 0, 0]] * 10 + [[0, 0, 1, 1]] * 10
    return csr_matrix(np.array(rows)), {"a": 0, "b": 1, "c": 2, "d": 3}


def test_npmi_is_high_for_words_that_appear_together_and_low_for_words_that_do_not():
    m, vocab = _matrix()
    together, _ = npmi_coherence({0: ["a", "b"]}, m, vocab)
    apart, _ = npmi_coherence({0: ["a", "c"]}, m, vocab)
    assert together == pytest.approx(1.0) and apart == -1.0
    mixed, per = npmi_coherence({0: ["a", "b"], 1: ["a", "c"]}, m, vocab)
    assert mixed == pytest.approx(0.0) and per[0] > per[1]


def test_npmi_skips_unknown_words_and_tiny_topics():
    m, vocab = _matrix()
    mean, per = npmi_coherence({0: ["a", "b", "zzz"], 1: ["a", "zzz"]}, m, vocab)
    assert list(per) == [0] and mean == pytest.approx(1.0)
    assert math.isnan(npmi_coherence({0: ["zzz"]}, m, vocab)[0])
