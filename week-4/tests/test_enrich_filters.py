import random

import numpy as np
import pandas as pd
import pytest

from nlpmeta.enrich import apply_to_chroma, build_doc_metadata, dataset_slug, parse_published
from nlpmeta.filters import Filters, build_where, matches, resolve_dataset, resolve_topics
from nlpmeta.ner import extract_corpus
from nlpmeta.search import count_matching, filtered_search
from nlpmeta.topics import build_topics
from tests.fakes import HashingModel
from tests.synth import make_corpus, make_store


@pytest.fixture(scope="module")
def pipeline():
    """Made-up corpus -> topics -> dataset mentions -> metadata (no database needed)."""
    corpus, emb, _ = make_corpus(n_per_topic=40)
    doc_topics, topics, *_ = build_topics(corpus, emb, "lda", n_lda_topics=3)
    entities = extract_corpus(corpus, nlp=None)
    meta, index = build_doc_metadata(doc_topics, entities, corpus, min_docs=3)
    return corpus, emb, doc_topics, topics, entities, meta, index


@pytest.fixture
def enriched_store(tmp_path, pipeline):
    corpus, emb, _, _, _, meta, _ = pipeline
    store = make_store(tmp_path / "db", corpus, emb)
    apply_to_chroma(store.collection, meta)
    return store


# --- building metadata ---
@pytest.mark.parametrize("value,expected", [("2026-10-05T12:00:00Z", (2026, 202610)), ("2025-01-31", (2025, 202501)),
                                            ("garbage", (None, None)), (None, (None, None)), ("", (None, None))])
def test_parse_published(value, expected):
    assert parse_published(value) == expected


def test_document_metadata_fields(pipeline):
    corpus, _, doc_topics, _, entities, meta, index = pipeline
    first = corpus["id"][0]
    m = meta[first]
    assert {"topic_id", "topic_label", "topic_core", "topic_sim", "short_doc", "doc_words", "has_dataset",
            "datasets", "published_year", "published_ym"} <= set(m)
    assert m["published_ym"] // 100 == m["published_year"]
    assert all(v is not None for v in m.values())  # ChromaDB rejects None
    assert len(meta) == len(corpus)


def test_dataset_flags_only_for_frequent_datasets(pipeline):
    corpus, _, _, _, entities, meta, index = pipeline
    assert dataset_slug("ImageNet") in index and index[dataset_slug("ImageNet")]["docs"] >= 3
    flagged_docs = [m for m in meta.values() if m.get(dataset_slug("ImageNet"))]
    assert flagged_docs and all(m["has_dataset"] and "ImageNet" in m["datasets"] for m in flagged_docs)
    _, strict_index = build_doc_metadata(pipeline[2], entities, corpus, min_docs=10_000)
    assert strict_index == {}


def test_missing_published_column_is_handled(pipeline):
    corpus, _, doc_topics, _, entities, _, _ = pipeline
    meta, _ = build_doc_metadata(doc_topics, entities, corpus.drop(columns=["published"]))
    assert all("published_year" not in m for m in meta.values())


# --- writing to ChromaDB ---
def test_existing_metadata_is_kept_and_new_metadata_added(enriched_store, pipeline):
    got = enriched_store.collection.get(limit=5, include=["metadatas"])["metadatas"]
    assert all({"category", "title", "doc_id", "chunk_index"} <= set(m) and "topic_id" in m for m in got)


def test_enrichment_is_repeatable(tmp_path, pipeline):
    corpus, emb, _, _, _, meta, _ = pipeline
    store = make_store(tmp_path / "db", corpus, emb)
    first = apply_to_chroma(store.collection, meta)
    snapshot = store.collection.get(include=["metadatas"])["metadatas"]
    second = apply_to_chroma(store.collection, meta)
    assert first["updated"] == second["updated"] == store.count() and first["no_nlp_data"] == 0
    assert store.collection.get(include=["metadatas"])["metadatas"] == snapshot


def test_old_dataset_flags_are_removed_on_a_rerun(tmp_path, pipeline):
    corpus, emb, doc_topics, _, entities, meta, index = pipeline
    store = make_store(tmp_path / "db", corpus, emb)
    apply_to_chroma(store.collection, meta)
    key = dataset_slug("ImageNet")
    assert count_matching(store, {key: True}) > 0
    no_flags, _ = build_doc_metadata(doc_topics, entities, corpus, min_docs=10_000)
    stats = apply_to_chroma(store.collection, no_flags)
    assert stats["stale_flags_removed"] > 0 and count_matching(store, {key: True}) == 0
    assert count_matching(store, {"topic_id": {"$gte": 0}}) == store.count()  # topics are still there


def test_chunks_of_unknown_documents_are_counted_and_left_alone(tmp_path, pipeline):
    corpus, emb, _, _, _, meta, _ = pipeline
    store = make_store(tmp_path / "db", corpus, emb)
    partial = {k: v for k, v in list(meta.items())[:10]}
    stats = apply_to_chroma(store.collection, partial)
    assert stats["updated"] == 20 and stats["no_nlp_data"] == store.count() - 20


# --- building filters ---
def test_no_filter_means_no_where_clause():
    assert build_where(Filters()) is None and Filters().is_empty()


def test_single_and_combined_clauses():
    assert build_where(Filters(topics=[3])) == {"topic_id": {"$in": [3]}}
    where = build_where(Filters(topics=[3], year_from=2025, year_to=2026, core_only=True))
    assert where["$and"] == [{"topic_id": {"$in": [3]}}, {"published_year": {"$gte": 2025}},
                             {"published_year": {"$lte": 2026}}, {"topic_core": {"$eq": True}}]


def test_dataset_modes():
    f = Filters(datasets=["ds_a", "ds_b"], datasets_mode="all")
    assert build_where(f) == {"$and": [{"ds_a": {"$eq": True}}, {"ds_b": {"$eq": True}}]}
    assert "$or" in build_where(Filters(datasets=["ds_a", "ds_b"]))


@pytest.mark.parametrize("bad", [Filters(datasets_mode="some"), Filters(year_from=2027, year_to=2025)])
def test_invalid_filters_raise(bad):
    with pytest.raises(ValueError):
        build_where(bad)


TOPICS_TABLE = [{"topic_id": 0, "label": "object detection, vision", "words": ["object detection", "pixels"]},
                {"topic_id": 1, "label": "language models, translation", "words": ["language", "translation"]}]


def test_topic_names_are_resolved():
    assert resolve_topics("1", TOPICS_TABLE) == [1]
    assert resolve_topics("Detection", TOPICS_TABLE) == [0]
    assert resolve_topics("language", TOPICS_TABLE) == [1]


def test_unknown_topics_give_helpful_errors():
    with pytest.raises(ValueError, match="no topic with id 9"):
        resolve_topics("9", TOPICS_TABLE)
    with pytest.raises(ValueError, match="Close labels"):
        resolve_topics("translations", TOPICS_TABLE)


def test_dataset_names_are_resolved_through_aliases():
    index = {"ds_coco": {"name": "COCO", "docs": 9}}
    assert resolve_dataset("MS COCO", index) == "ds_coco" == resolve_dataset("coco", index)
    with pytest.raises(ValueError, match="COCO"):
        resolve_dataset("CoCo2", index)


# --- the filter really filters ---
def test_database_filter_agrees_with_the_python_rules_for_many_random_filters(enriched_store, pipeline):
    """build_where run by ChromaDB must select exactly the chunks that matches() accepts."""
    everything = enriched_store.collection.get(include=["metadatas"])
    rng = random.Random(0)
    checked = 0
    for _ in range(60):
        f = Filters(
            topics=rng.sample([0, 1, 2], rng.randint(0, 2)),
            categories=rng.sample(["cs.CV", "cs.CL", "stat.ML"], rng.randint(0, 2)),
            datasets=rng.sample([dataset_slug("ImageNet"), dataset_slug("COCO")], rng.randint(0, 2)),
            datasets_mode=rng.choice(["any", "all"]),
            year_from=rng.choice([None, 2025, 2026]), year_to=rng.choice([None, 2025, 2026]),
            core_only=rng.random() < 0.3, exclude_short=rng.random() < 0.3)
        if f.year_from and f.year_to and f.year_from > f.year_to:
            continue
        where = build_where(f)
        in_db = set(enriched_store.collection.get(where=where, include=[])["ids"]) if where else set(everything["ids"])
        in_python = {cid for cid, m in zip(everything["ids"], everything["metadatas"]) if matches(m, f)}
        assert in_db == in_python, f"mismatch for {f}"
        assert count_matching(enriched_store, where) == len(in_python)
        checked += 1
    assert checked > 40


def test_filtered_search_only_returns_chunks_that_pass_the_filter(tmp_path, pipeline):
    corpus, emb, _, _, _, meta, _ = pipeline
    model = HashingModel(dim=32)  # same size as the made-up document vectors
    store = make_store(tmp_path / "db", corpus, emb)
    texts = store.collection.get(include=["documents"])
    store.collection.update(ids=texts["ids"], embeddings=model.encode(texts["documents"], normalize_embeddings=True))
    apply_to_chroma(store.collection, meta)
    f = Filters(categories=["cs.CV"], year_from=2026)
    results, timings, where = filtered_search("image detection camera pixels", model, store, f, top_k=8)
    assert results and all(matches(r["metadata"], f) for r in results)
    assert all(r["metadata"]["category"] == "cs.CV" and r["metadata"]["published_year"] >= 2026 for r in results)
    assert set(timings) == {"embed_ms", "query_ms", "total_ms"}
    none, _, _ = filtered_search("image detection", model, store, Filters(topics=[99]))
    assert none == []
    unfiltered, _, w = filtered_search("image detection", model, store, None, top_k=3)
    assert w is None and len(unfiltered) == 3
