"""A tiny stand-in for a sentence-transformers model, used only in tests.

It hashes words into a fixed-size vector, so texts that share words end up
close together. This lets the tests check search logic without downloading
a real model.
"""

import zlib

import numpy as np


class HashingModel:
    def __init__(self, dim: int = 64):
        self.dim = dim
        self.calls = 0

    def encode(self, texts, batch_size=32, normalize_embeddings=False,
               convert_to_numpy=True, show_progress_bar=False):
        self.calls += 1
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in str(text).lower().split():
                out[row, zlib.crc32(word.encode()) % self.dim] += 1.0
        if normalize_embeddings:
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            out = out / norms
        return out
