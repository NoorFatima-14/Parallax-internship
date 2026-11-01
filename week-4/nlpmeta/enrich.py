"""Write NLP metadata (topic, dataset mentions, dates, length) into the ChromaDB chunks."""

import re
from collections import Counter

from nlpmeta.datasets import normalize_key

DATASET_PREFIX = "ds_"


def dataset_slug(name: str) -> str:
    """Metadata key for a dataset flag: 'ImageNet' -> 'ds_imagenet'."""
    return DATASET_PREFIX + normalize_key(name)


def parse_published(value):
    """'2026-10-05T12:00:00Z' -> (2026, 202610). Returns (None, None) if it cannot be read."""
    match = re.match(r"^(\d{4})-(\d{2})", str(value or ""))
    if not match:
        return None, None
    year, month = int(match.group(1)), int(match.group(2))
    return year, year * 100 + month


def build_doc_metadata(doc_topics, entities, corpus, min_docs: int = 3):
    """Metadata per document. Returns ({doc_id: metadata}, datasets index).

    A dataset gets its own True/False flag only when it appears in at least min_docs
    documents, because every flag is one more key in the database.
    The 'datasets' text field lists all mentions for display.
    """
    counts = Counter(name for names in entities["datasets"] for name in set(names))
    flagged = {name for name, c in counts.items() if c >= min_docs}
    index = {dataset_slug(n): {"name": n, "docs": counts[n]} for n in sorted(flagged, key=lambda n: -counts[n])}

    published = dict(zip(corpus["id"], corpus["published"])) if "published" in corpus else {}
    names_by_doc = dict(zip(entities["doc_id"], entities["datasets"]))
    meta = {}
    for row in doc_topics.itertuples(index=False):
        names = list(names_by_doc.get(row.doc_id, []))
        md = {
            "topic_id": int(row.topic_id), "topic_label": str(row.topic_label),
            "topic_core": bool(row.topic_core), "topic_sim": float(row.assign_sim),
            "short_doc": bool(row.short_doc), "doc_words": int(row.words),
            "has_dataset": bool(names), "datasets": ", ".join(names),
        }
        year, ym = parse_published(published.get(row.doc_id))
        if year:
            md["published_year"], md["published_ym"] = year, ym
        for name in names:
            if name in flagged:
                md[dataset_slug(name)] = True
        meta[row.doc_id] = md
    return meta, index


def apply_to_chroma(collection, meta_by_doc: dict, batch: int = 2000) -> dict:
    """Add the metadata to every chunk of each document. Safe to run again.

    Existing keys (category, title, ...) are kept. Dataset flags from an earlier run that no
    longer apply are removed, so a re-run never leaves stale flags behind.
    """
    chunks, offset = [], 0
    while True:  # read everything first, then update, so paging is not disturbed by writes
        res = collection.get(include=["metadatas"], limit=5000, offset=offset)
        if len(res["ids"]) == 0:
            break
        chunks += [(cid, m or {}) for cid, m in zip(res["ids"], res["metadatas"])]
        offset += len(res["ids"])

    stats = {"chunks": len(chunks), "updated": 0, "no_nlp_data": 0, "stale_flags_removed": 0}
    ids, metas = [], []
    for cid, existing in chunks:
        new = meta_by_doc.get(existing.get("doc_id"))
        if new is None:
            stats["no_nlp_data"] += 1
            continue
        update = dict(new)
        for key in existing:
            if key.startswith(DATASET_PREFIX) and key not in new:
                update[key] = None  # None removes the key
                stats["stale_flags_removed"] += 1
        ids.append(cid)
        metas.append(update)
    for i in range(0, len(ids), batch):
        collection.update(ids=ids[i:i + batch], metadatas=metas[i:i + batch])
    stats["updated"] = len(ids)
    return stats
