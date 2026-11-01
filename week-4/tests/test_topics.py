import numpy as np
import pytest

from nlpmeta.topics import (
    align_embeddings, assign_by_centroid, build_topics, load_doc_embeddings, quality_metrics, renumber_by_size,
)
from tests.synth import make_corpus, make_store


def purity(labels, truth):
    """Share of documents whose topic is the majority topic of their true group."""
    return sum(np.bincount(labels[truth == t]).max() for t in set(truth)) / len(truth)


# --- embeddings ---
def test_document_vectors_are_the_normalized_average_of_chunk_vectors(tmp_path):
    corpus, emb, _ = make_corpus(n_per_topic=5)
    store = make_store(tmp_path / "db", corpus, emb)
    ids, matrix = load_doc_embeddings(store.collection, batch=7)  # small batch tests paging
    assert len(ids) == len(corpus) and set(ids) == set(corpus["id"])
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-5)
    row = ids.index(corpus["id"][0])
    assert float(matrix[row] @ emb[0]) > 0.95  # close to the original document vector


def test_align_marks_documents_without_vectors():
    matrix = np.eye(2, dtype=np.float32)
    out, have = align_embeddings(["b", "x", "a"], ["a", "b"], matrix)
    assert have.tolist() == [True, False, True] and out[0].tolist() == [0, 1] and out[1].tolist() == [0, 0]


def test_empty_collection_without_doc_ids_is_reported(tmp_path):
    from nlpmeta.paths import week2

    store = week2().VectorStore(tmp_path / "db", "nodocs", "fake")
    store.add_chunks([{"id": "a", "text": "no doc id here"}], np.ones((1, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="doc_id"):
        load_doc_embeddings(store.collection)


# --- assignment helpers ---
def test_outliers_are_assigned_to_the_nearest_centroid():
    emb = np.array([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]], dtype=np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    labels = np.array([0, 0, 1, 1, -1, -1])
    topic, sim = assign_by_centroid(emb, labels, labels >= 0)
    assert topic.tolist() == [0, 0, 1, 1, 0, 1] and (sim > 0.9).all()


def test_topics_are_renumbered_biggest_first():
    assert renumber_by_size([5, 5, 5, 2, 2, 9]) == {5: 0, 2: 1, 9: 2}


# --- full pipeline on made-up data ---
def test_lda_pipeline_finds_the_three_topics():
    corpus, emb, truth = make_corpus(n_per_topic=60)
    table, topics, info, words, texts = build_topics(corpus, emb, "lda", n_lda_topics=3)
    assert info["n_topics"] <= 3 and table["topic_core"].all()
    assert purity(table["topic_id"].values, truth) > 0.9
    assert sorted(table["topic_id"].unique()) == list(range(info["n_topics"]))
    assert all(t["label"] for t in topics) and sum(t["size"] for t in topics) == len(corpus)


def test_lda_needs_a_topic_count():
    corpus, emb, _ = make_corpus(n_per_topic=10)
    with pytest.raises(ValueError, match="n_lda_topics"):
        build_topics(corpus, emb, "lda")


def test_unknown_method_is_rejected():
    corpus, emb, _ = make_corpus(n_per_topic=10)
    with pytest.raises(ValueError, match="Unknown method"):
        build_topics(corpus, emb, "nope")


def test_quality_metrics_prefer_real_topics_over_shuffled_words():
    corpus, emb, _ = make_corpus(n_per_topic=60)
    _, _, _, words, texts = build_topics(corpus, emb, "lda", n_lda_topics=3)
    real, _ = quality_metrics(words, texts)
    flat = [w for ws in words.values() for w in ws]
    rng = np.random.default_rng(0)
    shuffled = {t: list(rng.choice(flat, 10, replace=False)) for t in words}
    fake, _ = quality_metrics(shuffled, texts)
    assert real["npmi"] > fake["npmi"] and 0 < real["diversity"] <= 1


def test_bertopic_pipeline_finds_topics_and_handles_short_documents():
    pytest.importorskip("bertopic")
    corpus, emb, truth = make_corpus(n_per_topic=100, n_short=30)
    table, topics, info, words, _ = build_topics(corpus, emb, "bertopic", min_topic_size=20)
    assert info["n_topics"] == 3
    short = table["short_doc"].values
    assert short.sum() >= 1 and not table.loc[short, "topic_core"].any()  # short docs are never core
    assert (table.loc[short, "topic_raw"] == -1).all() and (table["topic_id"] >= 0).all()
    assert purity(table["topic_id"].values, truth) > 0.9
    assert purity(table.loc[short, "topic_id"].values, truth[short]) > 0.8  # short docs still land well
    for t in topics:
        assert len(t["examples"]) == 3 and t["core_size"] <= t["size"]


def test_hyphenated_jargon_survives_in_topic_words():
    """BERTopic's own cleaning used to delete hyphens, which broke the word list on real data."""
    pytest.importorskip("bertopic")
    corpus, emb, _ = make_corpus(n_per_topic=100)
    _, topics, _, words, _ = build_topics(corpus, emb, "bertopic", min_topic_size=20)
    all_words = {w for ws in words.values() for w in ws}
    assert all_words & {"vision-language", "multi-scale", "3d-cnn", "cross-lingual", "encoder-decoder", "state-space"}


def test_messy_text_and_many_outliers_do_not_crash():
    """Real abstracts have odd characters, empty-after-cleaning texts and many HDBSCAN outliers."""
    pytest.importorskip("bertopic")
    corpus, emb, _ = make_corpus(n_per_topic=100, noise=0.9, n_short=20)
    corpus.loc[0, "text"] = "$x^2 + \\alpha$ \\textbf{only math}"          # nothing left after cleaning
    corpus.loc[1, "text"] = "Caf\u00e9 na\u00efve non\u2011breaking hyphen CNN/DailyMail O'Neil 3D 5G (ISO/IEC) \u201cquoted\u201d"
    corpus.loc[2, "title"] = None
    table, topics, info, words, texts = build_topics(corpus, emb, "bertopic", min_topic_size=20)
    assert len(table) == len(corpus) and (table["topic_id"] >= 0).all() and info["n_topics"] >= 1
    assert table["topic_label"].map(len).min() > 0 and not table["assign_sim"].isna().any()
    metrics, _ = quality_metrics(words, texts)
    assert -1 <= metrics["npmi"] <= 1


def test_no_topics_gives_a_clear_error():
    pytest.importorskip("bertopic")
    corpus, emb, _ = make_corpus(n_per_topic=60, noise=8.0)
    with pytest.raises(ValueError, match="min-topic-size"):
        build_topics(corpus, emb, "bertopic", min_topic_size=60)
