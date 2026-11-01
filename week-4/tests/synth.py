"""A small made-up corpus with three clear topics, for tests. Not real data."""

import numpy as np
import pandas as pd

from nlpmeta.paths import week2

TOPICS = {
    0: ("image detection camera vision pixels object segmentation convolutional multi-scale vision-language 3D-CNN", "cs.CV"),
    1: ("language translation sentence token corpus parsing linguistic grammar cross-lingual encoder-decoder", "cs.CL"),
    2: ("bayesian posterior inference prior variational uncertainty sampling gaussian state-space", "stat.ML"),
}
FILLER = "we propose a novel method and show results".split()


def make_corpus(n_per_topic=80, dim=32, seed=0, n_short=0, noise=0.25):
    """Returns (DataFrame with id/title/text/published/category, unit embeddings, true topic per row)."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(len(TOPICS), dim))
    rows, vectors, truth = [], [], []
    for i in range(n_per_topic * len(TOPICS)):
        t = i % len(TOPICS)
        words, category = TOPICS[t]
        vocab = words.split()
        n_words = 12 if i < n_short else 60
        body = " ".join(rng.choice(vocab + FILLER, size=n_words))
        if t == 0 and i % 4 == 0:
            body += " We evaluate on ImageNet and COCO."
        v = centres[t] + noise * rng.normal(size=dim)
        rows.append({"id": f"http://arxiv.org/abs/2601.{i:05d}v1", "title": f"Paper {i} on {vocab[0]}",
                     "text": body, "category": category,
                     "published": f"202{5 + i % 2}-{1 + i % 9:02d}-15T00:00:00Z"})
        vectors.append(v / np.linalg.norm(v))
        truth.append(t)
    return pd.DataFrame(rows), np.array(vectors, dtype=np.float32), np.array(truth)


def make_store(path, corpus, embeddings, name="synth_test", chunks_per_doc=2, seed=1, model_name="fake"):
    """A Week 2 VectorStore with chunks of the made-up corpus (chunk vectors = document vector + noise)."""
    rng = np.random.default_rng(seed)
    store = week2().VectorStore(path, name, model_name)
    chunks, vectors = [], []
    for row, vec in zip(corpus.itertuples(index=False), embeddings):
        for j in range(chunks_per_doc):
            chunks.append({"id": f"{row.id}::{j}", "text": f"{row.text} part {j}", "doc_id": row.id,
                           "chunk_index": j, "n_chunks": chunks_per_doc, "title": row.title,
                           "category": row.category, "published": row.published})
            noisy = vec + 0.05 * rng.normal(size=vec.shape)
            vectors.append(noisy / np.linalg.norm(noisy))
    store.add_chunks(chunks, np.array(vectors, dtype=np.float32))
    return store
