import numpy as np
import pytest

from src.search import semantic_search
from src.vector_store import VectorStore
from tests.fakes import HashingModel

TEXTS = {
    "d1::0": "convolutional neural networks for image classification and object detection",
    "d2::0": "transformer language models for machine translation and text generation",
    "d3::0": "bayesian inference with variational methods and gaussian processes",
}


def make_chunks(extra=None):
    chunks = [
        {"id": cid, "text": text, "doc_id": cid.split("::")[0], "category": cat}
        for (cid, text), cat in zip(TEXTS.items(), ["cs.CV", "cs.CL", "stat.ML"])
    ]
    return chunks + (extra or [])


@pytest.fixture
def model():
    return HashingModel(dim=64)


@pytest.fixture
def store(tmp_path):
    return VectorStore(path=tmp_path / "db", collection_name="test_chunks")


@pytest.fixture
def filled(store, model):
    chunks = make_chunks()
    emb = model.encode([c["text"] for c in chunks], normalize_embeddings=True)
    store.add_chunks(chunks, emb)
    return store


# --- basic behaviour ---
def test_add_and_count(store, model):
    chunks = make_chunks()
    emb = model.encode([c["text"] for c in chunks], normalize_embeddings=True)
    stats = store.add_chunks(chunks, emb)
    assert stats["added"] == 3 and store.count() == 3


def test_semantic_search_finds_the_right_chunk(filled, model):
    results = semantic_search("image classification with convolutional networks", model, filled, top_k=3)
    assert results[0]["id"] == "d1::0"
    assert results[0]["score"] >= results[1]["score"] >= results[2]["score"]
    assert results[0]["metadata"]["category"] == "cs.CV"


def test_results_contain_text_and_metadata(filled, model):
    top = semantic_search("bayesian inference", model, filled, top_k=1)[0]
    assert top["id"] == "d3::0" and top["text"] == TEXTS["d3::0"]
    assert "id" not in top["metadata"] and "text" not in top["metadata"]


def test_metadata_filter(filled, model):
    results = semantic_search("neural networks", model, filled, top_k=3, where={"category": "cs.CL"})
    assert [r["id"] for r in results] == ["d2::0"]


def test_filter_matching_nothing_returns_empty_list(filled, model):
    assert semantic_search("anything", model, filled, where={"category": "nope"}) == []


# --- edge cases: input ---
def test_empty_input_is_a_noop(store):
    stats = store.add_chunks([], np.zeros((0, 64)))
    assert stats["added"] == 0 and store.count() == 0


def test_empty_text_and_missing_id_are_skipped(store, model):
    chunks = make_chunks([{"id": "e::0", "text": "   "}, {"id": "", "text": "no id"}, {"text": "no id key"}])
    emb = model.encode([c["text"] for c in chunks], normalize_embeddings=True)
    stats = store.add_chunks(chunks, emb)
    assert stats["added"] == 3
    assert stats["skipped_empty_text"] == 1 and stats["skipped_missing_id"] == 2


def test_duplicate_ids_in_one_batch_keep_the_first(store, model):
    chunks = make_chunks([{"id": "d1::0", "text": "a second copy with the same id"}])
    emb = model.encode([c["text"] for c in chunks], normalize_embeddings=True)
    stats = store.add_chunks(chunks, emb)
    assert stats["skipped_duplicate_id"] == 1 and store.count() == 3


def test_running_ingestion_twice_does_not_duplicate(filled, model):
    chunks = make_chunks()
    emb = model.encode([c["text"] for c in chunks], normalize_embeddings=True)
    filled.add_chunks(chunks, emb)
    assert filled.count() == 3


def test_nan_embedding_is_skipped(store, model):
    chunks = make_chunks()
    emb = model.encode([c["text"] for c in chunks], normalize_embeddings=True)
    emb[1, 0] = np.nan
    stats = store.add_chunks(chunks, emb)
    assert stats["skipped_bad_embedding"] == 1 and store.count() == 2


def test_none_and_odd_metadata_values_are_cleaned(store, model):
    chunks = [{"id": "m::0", "text": "some text here", "title": None, "tags": ["a", "b"], "year": 2026}]
    emb = model.encode(["some text here"], normalize_embeddings=True)
    assert store.add_chunks(chunks, emb)["added"] == 1
    meta = store.collection.get(ids=["m::0"])["metadatas"][0]
    assert "title" not in meta and meta["year"] == 2026 and meta["tags"] == "['a', 'b']"


def test_chunk_with_no_metadata_is_accepted(store, model):
    emb = model.encode(["plain text only"], normalize_embeddings=True)
    assert store.add_chunks([{"id": "p::0", "text": "plain text only"}], emb)["added"] == 1


def test_row_count_mismatch_raises(store, model):
    chunks = make_chunks()
    with pytest.raises(ValueError):
        store.add_chunks(chunks, model.encode(["only one"], normalize_embeddings=True))


def test_wrong_embedding_dimension_raises(filled):
    chunks = [{"id": "x::0", "text": "new text"}]
    with pytest.raises(ValueError, match="does not match"):
        filled.add_chunks(chunks, np.ones((1, 32), dtype=np.float32))


def test_batches_larger_than_batch_size_are_split(store, model):
    chunks = [{"id": f"b{i}::0", "text": f"text number {i}"} for i in range(25)]
    emb = model.encode([c["text"] for c in chunks], normalize_embeddings=True)
    assert store.add_chunks(chunks, emb, batch_size=7)["added"] == 25
    assert store.count() == 25


# --- edge cases: querying ---
def test_query_on_empty_collection_returns_empty_list(store, model):
    assert semantic_search("anything at all", model, store) == []


def test_top_k_larger_than_collection_is_capped(filled, model):
    assert len(semantic_search("neural networks", model, filled, top_k=50)) == 3


@pytest.mark.parametrize("bad", [0, -1, 2.5, None])
def test_invalid_top_k_raises(filled, model, bad):
    with pytest.raises(ValueError):
        semantic_search("neural networks", model, filled, top_k=bad)


@pytest.mark.parametrize("bad", ["", "   ", None, 123])
def test_empty_or_non_string_query_raises(filled, model, bad):
    with pytest.raises(ValueError):
        semantic_search(bad, model, filled)


def test_very_long_query_still_works(filled, model):
    assert len(semantic_search("neural " * 5000, model, filled, top_k=1)) == 1


def test_query_with_wrong_dimension_raises(filled):
    with pytest.raises(ValueError, match="does not match"):
        filled.query(np.ones(10, dtype=np.float32))


def test_query_with_nan_raises(filled):
    with pytest.raises(ValueError):
        filled.query(np.full(64, np.nan, dtype=np.float32))


# --- edge cases: collection setup ---
@pytest.mark.parametrize("name", ["", "ab", "has space", "-start", "end-", "a/b"])
def test_invalid_collection_name_raises(tmp_path, name):
    with pytest.raises(ValueError, match="Invalid collection name"):
        VectorStore(path=tmp_path / "db", collection_name=name)


def test_data_persists_after_reopening(tmp_path, model):
    path = tmp_path / "db"
    first = VectorStore(path=path, collection_name="persist_test")
    chunks = make_chunks()
    first.add_chunks(chunks, model.encode([c["text"] for c in chunks], normalize_embeddings=True))
    second = VectorStore(path=path, collection_name="persist_test")
    assert second.count() == 3


def test_opening_with_a_different_model_raises(tmp_path):
    VectorStore(path=tmp_path / "db", collection_name="model_check", model_name="model-a")
    with pytest.raises(ValueError, match="model-a"):
        VectorStore(path=tmp_path / "db", collection_name="model_check", model_name="model-b")


def test_reset_empties_the_collection(filled):
    filled.reset()
    assert filled.count() == 0
