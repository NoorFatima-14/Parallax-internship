import json

import numpy as np

from src.embedding import append_json_log, timed_encode
from src.search import timed_search
from src.vector_store import VectorStore
from tests.fakes import HashingModel


def test_timed_encode_returns_vectors_and_stats():
    model = HashingModel(dim=32)
    emb, stats = timed_encode(model, ["one text", "another text", "third"], batch_size=2,
                              show_progress_bar=False)
    assert emb.shape == (3, 32) and emb.dtype == np.float32
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-5)
    assert stats["n_texts"] == 3 and stats["embedding_dim"] == 32
    assert stats["total_seconds"] >= 0 and stats["ms_per_text"] >= 0


def test_timed_encode_does_warmup_before_timing():
    model = HashingModel()
    timed_encode(model, ["a", "b"], show_progress_bar=False)
    assert model.calls == 2  # one warm-up call + one timed call


def test_timed_encode_with_no_texts():
    emb, stats = timed_encode(HashingModel(), [], show_progress_bar=False)
    assert emb.shape[0] == 0 and stats["n_texts"] == 0


def test_append_json_log_creates_and_appends(tmp_path):
    path = tmp_path / "logs" / "log.json"
    append_json_log(path, {"run": 1})
    append_json_log(path, {"run": 2})
    records = json.loads(path.read_text())
    assert [r["run"] for r in records] == [1, 2]
    assert "timestamp" in records[0]


def test_append_json_log_keeps_a_broken_file(tmp_path):
    path = tmp_path / "log.json"
    path.write_text("{not valid json")
    append_json_log(path, {"run": 1})
    assert json.loads(path.read_text())[0]["run"] == 1
    assert (tmp_path / "log.broken.json").exists()


def test_timed_search_reports_timings(tmp_path):
    model = HashingModel()
    store = VectorStore(path=tmp_path / "db", collection_name="timing_test")
    chunks = [{"id": "a::0", "text": "graph neural networks"}, {"id": "b::0", "text": "bayesian methods"}]
    store.add_chunks(chunks, model.encode([c["text"] for c in chunks], normalize_embeddings=True))
    results, timings = timed_search("graph networks", model, store, top_k=1)
    assert results[0]["id"] == "a::0"
    assert set(timings) == {"embed_ms", "query_ms", "total_ms"}
    assert abs(timings["total_ms"] - timings["embed_ms"] - timings["query_ms"]) < 5
