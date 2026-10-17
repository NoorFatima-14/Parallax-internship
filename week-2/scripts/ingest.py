
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunking import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, chunk_documents  # noqa: E402
from src.embedding import (  # noqa: E402
    DEFAULT_MODEL, append_json_log, environment_info, load_model, timed_encode,
)
from src.vector_store import DEFAULT_COLLECTION, VectorStore  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="../week-1/data/processed/clean_corpus.parquet")
    p.add_argument("--db-path", default="chroma_db")
    p.add_argument("--collection", default=DEFAULT_COLLECTION)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--device", default=None, help="cpu or cuda (default: auto)")
    p.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    p.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--limit", type=int, default=None, help="only use the first N documents")
    p.add_argument("--reset", action="store_true", help="empty the collection first")
    p.add_argument("--log", default="logs/ingest_log.json")
    args = p.parse_args()

    path = Path(args.input)
    if not path.exists():
        sys.exit(
            f"Input file not found: {path}\n"
            "Build the Week 1 clean corpus first (from the week-1 folder):\n"
            "  python scripts/download_data.py --per-category 1500\n"
            "  python scripts/build_clean_corpus.py"
        )
    df = pd.read_parquet(path)
    missing = {"id", "text"} - set(df.columns)
    if missing:
        sys.exit(f"Input file is missing required columns: {sorted(missing)}")
    if args.limit:
        df = df.head(args.limit)
    docs = df.to_dict("records")

    # 1. Chunking
    start = time.perf_counter()
    chunks, skipped_docs = chunk_documents(docs, args.chunk_size, args.chunk_overlap)
    chunk_seconds = time.perf_counter() - start
    if not chunks:
        sys.exit("No chunks were produced. Check the input file.")
    lengths = [len(c["text"]) for c in chunks]
    print(f"Documents: {len(docs)} (skipped {skipped_docs}) -> chunks: {len(chunks)}")
    print(f"Chunks per document: {len(chunks) / (len(docs) - skipped_docs):.2f} | "
          f"characters per chunk: mean {sum(lengths) / len(lengths):.0f}, max {max(lengths)}")
    print(f"Chunking took {chunk_seconds:.2f} s")

    # 2. Embeddings
    model, model_info = load_model(args.model, args.device)
    print(f"Model {model_info['model']} on {model_info['device']} "
          f"(loaded in {model_info['load_seconds']} s)")
    embeddings, emb_stats = timed_encode(model, [c["text"] for c in chunks], args.batch_size)
    print(f"Embedding {emb_stats['n_texts']} chunks took {emb_stats['total_seconds']} s "
          f"({emb_stats['texts_per_second']} chunks/s, {emb_stats['ms_per_text']} ms/chunk)")

    # 3. ChromaDB
    store = VectorStore(args.db_path, args.collection, args.model)
    if args.reset:
        store.reset()
    start = time.perf_counter()
    add_stats = store.add_chunks(chunks, embeddings)
    ingest_seconds = time.perf_counter() - start
    print(f"Stored {add_stats['added']} chunks in ChromaDB in {ingest_seconds:.2f} s "
          f"(collection now has {store.count()})")
    skipped = {k: v for k, v in add_stats.items() if k.startswith("skipped") and v}
    if skipped:
        print(f"Skipped: {skipped}")

    append_json_log(args.log, {
        "documents": len(docs), "skipped_documents": skipped_docs, "chunks": len(chunks),
        "chunk_size": args.chunk_size, "chunk_overlap": args.chunk_overlap,
        "mean_chunk_chars": round(sum(lengths) / len(lengths), 1),
        "chunking_seconds": round(chunk_seconds, 3),
        "model": model_info, "embedding": emb_stats,
        "chroma_ingest_seconds": round(ingest_seconds, 3),
        "chroma_stats": add_stats, "collection_count": store.count(),
        "environment": environment_info(),
    })
    print(f"Log saved to {args.log}")


if __name__ == "__main__":
    main()
