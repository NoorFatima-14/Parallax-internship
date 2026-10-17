"""Embedding helpers: load a sentence-transformers model, encode texts, log timing."""

import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

DEFAULT_MODEL = "all-MiniLM-L6-v2"


def pick_device(requested: str | None = None) -> str:
    """Return 'cuda' if a GPU is available, otherwise 'cpu' (or the requested device)."""
    if requested:
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def load_model(model_name: str = DEFAULT_MODEL, device: str | None = None):
    """Load a sentence-transformers model. Returns (model, info) with load time."""
    from sentence_transformers import SentenceTransformer

    device = pick_device(device)
    start = time.perf_counter()
    model = SentenceTransformer(model_name, device=device)
    info = {
        "model": model_name,
        "device": device,
        "load_seconds": round(time.perf_counter() - start, 3),
        "max_seq_length": getattr(model, "max_seq_length", None),
    }
    return model, info


def timed_encode(model, texts, batch_size: int = 64, show_progress_bar: bool = True):
    """Encode texts into normalized float32 vectors and measure how long it takes.

    A short warm-up call runs first, so one-time start-up cost is not counted.
    Returns (embeddings, stats).
    """
    texts = list(texts)
    if not texts:
        return np.zeros((0, 0), dtype=np.float32), {
            "n_texts": 0, "batch_size": batch_size, "total_seconds": 0.0,
            "texts_per_second": 0.0, "ms_per_text": 0.0,
        }

    model.encode(["warm up"], show_progress_bar=False)  # not timed

    start = time.perf_counter()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=show_progress_bar,
    )
    elapsed = time.perf_counter() - start

    embeddings = np.asarray(embeddings, dtype=np.float32)
    stats = {
        "n_texts": len(texts),
        "batch_size": batch_size,
        "embedding_dim": int(embeddings.shape[1]),
        "total_seconds": round(elapsed, 3),
        "texts_per_second": round(len(texts) / elapsed, 2) if elapsed > 0 else None,
        "ms_per_text": round(1000 * elapsed / len(texts), 3),
    }
    return embeddings, stats


def environment_info() -> dict:
    """Small description of the machine, saved with every log entry."""
    info = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return info


def append_json_log(path, record: dict) -> None:
    """Append one record to a JSON file that holds a list of records."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    if path.exists():
        try:
            records = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(records, list):
                raise ValueError("log file is not a list")
        except (json.JSONDecodeError, ValueError):
            # Keep the unreadable file instead of overwriting it.
            path.rename(path.with_suffix(".broken.json"))
            records = []
    record = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), **record}
    records.append(record)
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")
