"""ChromaDB wrapper with the edge cases handled in one place.

Edge cases covered (each has a unit test):
- empty input list, empty or whitespace-only text
- duplicate ids inside one batch, and re-running ingestion (upsert, not add)
- embeddings with NaN/inf, wrong shape, or a different dimension than the collection
- metadata values that are None or not str/int/float/bool, or no metadata at all
- batches larger than Chroma's maximum batch size
- querying an empty collection, top_k larger than the collection, top_k < 1
- a filter that matches nothing
- opening a collection that was built with a different embedding model
- invalid collection names
"""

import re
from pathlib import Path

import chromadb
import numpy as np

DEFAULT_COLLECTION = "arxiv_chunks"
_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{1,510}[a-zA-Z0-9]$")
_NOT_METADATA = {"id", "text", "embedding"}


def _clean_metadata(chunk: dict) -> dict:
    """Keep only values Chroma accepts (str, int, float, bool). Drop None."""
    meta = {}
    for key, value in chunk.items():
        if key in _NOT_METADATA or value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            meta[key] = value
        else:
            meta[key] = str(value)
    return meta


class VectorStore:
    def __init__(
        self,
        path="chroma_db",
        collection_name: str = DEFAULT_COLLECTION,
        model_name: str = "all-MiniLM-L6-v2",
        space: str = "cosine",
    ):
        if not _NAME_RE.match(collection_name or ""):
            raise ValueError(
                "Invalid collection name. Use 3-512 letters, digits, '.', '_' or '-', "
                "starting and ending with a letter or digit."
            )
        Path(path).mkdir(parents=True, exist_ok=True)
        self.path = str(path)
        self.collection_name = collection_name
        self.model_name = model_name
        self.client = chromadb.PersistentClient(path=self.path)
        # embedding_function=None: we always pass our own vectors, so Chroma
        # never tries to download its default model.
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": space, "embedding_model": model_name},
            embedding_function=None,
        )
        existing = (self.collection.metadata or {}).get("embedding_model")
        if existing and existing != model_name:
            raise ValueError(
                f"Collection '{collection_name}' was built with model '{existing}', "
                f"not '{model_name}'. Use another collection name or reset it."
            )

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        """Delete all data in this collection and start again."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine", "embedding_model": self.model_name},
            embedding_function=None,
        )

    def add_chunks(self, chunks, embeddings, batch_size: int = 2000) -> dict:
        """Insert chunks with their embeddings. Safe to run again (upsert).

        Returns counts of what was added and what was skipped, and why.
        """
        stats = {
            "received": len(chunks), "added": 0, "skipped_missing_id": 0,
            "skipped_empty_text": 0, "skipped_duplicate_id": 0,
            "skipped_bad_embedding": 0,
        }
        if len(chunks) == 0:
            return stats

        emb = np.asarray(embeddings, dtype=np.float32)
        if emb.ndim != 2 or emb.shape[0] != len(chunks):
            raise ValueError(
                f"Expected a 2D array with {len(chunks)} rows, got shape {emb.shape}"
            )
        if self.count() > 0:
            existing = self.collection.peek(1)["embeddings"][0]
            if len(existing) != emb.shape[1]:
                raise ValueError(
                    f"Embedding size {emb.shape[1]} does not match the collection "
                    f"({len(existing)}). Was another model used?"
                )

        ids, docs, vectors, metas, seen = [], [], [], [], set()
        for chunk, vector in zip(chunks, emb):
            chunk_id = str(chunk.get("id") or "").strip()
            text = str(chunk.get("text") or "").strip()
            if not chunk_id:
                stats["skipped_missing_id"] += 1
            elif not text:
                stats["skipped_empty_text"] += 1
            elif chunk_id in seen:
                stats["skipped_duplicate_id"] += 1
            elif not np.isfinite(vector).all():
                stats["skipped_bad_embedding"] += 1
            else:
                seen.add(chunk_id)
                ids.append(chunk_id)
                docs.append(text)
                vectors.append(vector)
                meta = _clean_metadata(chunk)
                meta["n_chars"] = len(text)  # Chroma rejects empty metadata dicts
                metas.append(meta)

        size = max(1, min(batch_size, self.client.get_max_batch_size()))
        for i in range(0, len(ids), size):
            self.collection.upsert(
                ids=ids[i : i + size],
                documents=docs[i : i + size],
                embeddings=np.array(vectors[i : i + size]),
                metadatas=metas[i : i + size],
            )
        stats["added"] = len(ids)
        return stats

    def query(self, query_embedding, top_k: int = 5, where: dict | None = None) -> list[dict]:
        """Return the closest chunks to a query vector, best first."""
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be an integer of at least 1")
        vector = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        if not np.isfinite(vector).all():
            raise ValueError("Query embedding contains NaN or infinity")

        n = self.count()
        if n == 0:
            return []
        try:
            res = self.collection.query(
                query_embeddings=vector,
                n_results=min(top_k, n),
                where=where or None,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            if "dimension" in str(exc).lower():
                raise ValueError(
                    f"Query embedding has size {vector.shape[1]}, which does not match "
                    "the collection. Use the same model that built it."
                ) from exc
            raise

        results = []
        for cid, doc, meta, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            results.append(
                {"id": cid, "text": doc, "metadata": meta or {},
                 "distance": float(dist), "score": round(1.0 - float(dist), 4)}
            )
        return results
