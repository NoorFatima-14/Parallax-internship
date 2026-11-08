"""Stand-ins for the slow or external parts, used only in tests.

- HashingModel: a fake embedding model (words are hashed into a vector), so no model download is needed
- FakeLLM: returns scripted answers or raises scripted errors, so no API key or internet is needed
- build_service: a real RAGService over a tiny real ChromaDB, with the fake model and the fake LLM
"""

import zlib

import numpy as np

from api.service import RAGService
from rag.config import RAGConfig
from rag.llm_client import LLMResponse
from rag.pipeline import RAGPipeline
from rag.retriever import Retriever, week2


class HashingModel:
    def __init__(self, dim: int = 512):
        self.dim = dim

    def encode(self, texts, batch_size=32, normalize_embeddings=False, convert_to_numpy=True, show_progress_bar=False):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in str(text).lower().split():
                out[row, zlib.crc32(word.encode()) % self.dim] += 1.0
        if normalize_embeddings:
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            out = out / norms
        return out


GOOD_ANSWER = "Convolutional neural networks detect objects in images [1]."


class FakeLLM:
    """Pops one scripted item per call. An item is a text, an LLMResponse or an exception to raise.
    When the script is empty, `default` is returned (so a test does not have to count calls)."""

    provider = "fake"
    models = ["fake-model"]

    def __init__(self, script=None, default=GOOD_ANSWER):
        self.script, self.default, self.calls = list(script or []), default, []

    def complete(self, messages, max_tokens=700, temperature=0.0, models=None):
        self.calls.append(messages)
        item = self.script.pop(0) if self.script else self.default
        if isinstance(item, Exception):
            raise item
        if isinstance(item, LLMResponse):
            return item
        return LLMResponse(text=item, model="fake-model", usage={"total_tokens": 42})


# id, category, title, year, topic id, uses COCO, text
DOCS = [
    ("2601.00001v1", "cs.CV", "Pedestrian detection", 2026, 0, True,
     "Convolutional neural networks detect objects in images. Pedestrian detection under heavy occlusion "
     "reached 92.5 percent accuracy on 1200 test images."),
    ("2601.00002v1", "cs.CL", "Attention for translation", 2025, 1, False,
     "Transformers use attention for machine translation and language modelling of long text sequences."),
    ("2601.00003v1", "stat.ML", "Variational inference", 2026, 2, False,
     "Bayesian inference with variational methods estimates posterior distributions for gaussian processes."),
    ("2601.00004v1", "cs.CV", "Aerial object detection", 2025, 0, False,
     "Object detection with anchors and region proposals finds vehicles in aerial images."),
]
TOPICS = [
    {"topic_id": 0, "label": "0_image_detection_object", "words": ["image", "detection", "object"], "size": 2},
    {"topic_id": 1, "label": "1_translation_language_text", "words": ["translation", "language", "text"], "size": 1},
    {"topic_id": 2, "label": "2_bayesian_inference_posterior", "words": ["bayesian", "inference", "posterior"], "size": 1},
]
DATASETS_INDEX = {"ds_coco": {"name": "COCO", "docs": 1}}


def build_store(path, name="api_test"):
    """A tiny ChromaDB with the DOCS above, including Week 4 style metadata."""
    model = HashingModel()
    store = week2().VectorStore(path, name, "fake")
    chunks = []
    for doc_id, category, title, year, topic, coco, text in DOCS:
        chunk = {"id": f"http://arxiv.org/abs/{doc_id}::0", "doc_id": f"http://arxiv.org/abs/{doc_id}",
                 "chunk_index": 0, "n_chunks": 1, "text": text, "title": title, "category": category,
                 "published_year": year, "topic_id": topic, "topic_label": TOPICS[topic]["label"],
                 "topic_core": True, "short_doc": False, "has_dataset": coco}
        if coco:
            chunk.update(datasets="COCO", ds_coco=True)
        chunks.append(chunk)
    store.add_chunks(chunks, model.encode([c["text"] for c in chunks], normalize_embeddings=True))
    return Retriever(store, model)


def build_service(path, llm=None, filters=True, config=None) -> tuple[RAGService, FakeLLM]:
    llm = llm or FakeLLM()
    pipeline = RAGPipeline(build_store(path), llm, config or RAGConfig())
    service = RAGService(pipeline, TOPICS if filters else None, DATASETS_INDEX if filters else None)
    return service, llm
