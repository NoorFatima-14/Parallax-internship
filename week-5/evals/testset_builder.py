"""Building the retrieval test set.

1. Known-item queries (automatic). For random papers: the title, and one sentence from the abstract.
   The relevant paper is the one the query came from.
2. Topical queries (judged by a person). A list of topic queries is run through the system, the papers that come
   back are written to a sheet in random order WITHOUT their rank or score, and the person marks each one
   y (relevant) or n. This is called pooling. The papers are shuffled so the ranking does not influence the judge.
"""

import csv
import random
import re
from pathlib import Path

from evals.metrics import dedupe, normalize_doc_id
from evals.retrieval_eval import doc_id_of

SHEET_COLUMNS = ["query_id", "query", "doc_id", "title", "category", "snippet", "relevant"]
YES, NO = {"y", "yes", "1", "true", "relevant"}, {"n", "no", "0", "false", "not relevant"}


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", " ".join(text.split())) if s.strip()]


# ------------------------------------------------------------------ known-item queries
def group_by_doc(ids, documents, metadatas) -> dict:
    """{doc_id: {'title': str, 'chunks': [(chunk_index, text), ...]}} from the rows of a ChromaDB collection."""
    docs = {}
    for chunk_id, text, meta in zip(ids, documents, metadatas):
        meta = meta or {}
        entry = docs.setdefault(normalize_doc_id(meta.get("doc_id") or chunk_id),
                                {"title": str(meta.get("title") or "").strip(), "chunks": []})
        entry["chunks"].append((int(meta.get("chunk_index") or 0), text))
    for entry in docs.values():
        entry["chunks"].sort()
    return docs


def build_known_item(docs: dict, n_docs: int = 100, seed: int = 42, subtypes=("title", "sentence")) -> list[dict]:
    """Title and/or sentence queries for `n_docs` random papers.

    Papers whose title is missing, very short, or shared with another paper are skipped (their title would
    not point to one paper). Sentences must be 60 to 300 characters long.
    """
    title_count = {}
    for entry in docs.values():
        title_count[entry["title"].lower()] = title_count.get(entry["title"].lower(), 0) + 1
    eligible = sorted(d for d, e in docs.items()
                      if len(e["title"]) >= 15 and len(e["title"].split()) >= 3 and title_count[e["title"].lower()] == 1)
    rng = random.Random(seed)
    picked = rng.sample(eligible, min(n_docs, len(eligible)))
    queries = []
    for i, doc_id in enumerate(picked, 1):
        entry = docs[doc_id]
        if "title" in subtypes:
            queries.append({"id": f"ki-title-{i:04d}", "query": entry["title"], "type": "known_item",
                            "subtype": "title", "relevant": [doc_id]})
        if "sentence" in subtypes:
            sentences = [s for _, text in entry["chunks"] for s in split_sentences(text) if 60 <= len(s) <= 300]
            if sentences:
                queries.append({"id": f"ki-sentence-{i:04d}", "query": rng.choice(sentences), "type": "known_item",
                                "subtype": "sentence", "relevant": [doc_id]})
    return queries


# ------------------------------------------------------------------ topical queries (pooling and judging)
def pool_candidates(search_fn, topical_queries: list[dict], pool_k: int = 10, overfetch: int = 4,
                    seed: int = 42) -> list[dict]:
    """Rows for the judging sheet: the top `pool_k` papers of every query, in random order (no rank, no score)."""
    rng = random.Random(seed)
    rows = []
    for q in topical_queries:
        hits = search_fn(q["query"], pool_k * overfetch)
        first_hit = {}
        for h in hits:
            first_hit.setdefault(doc_id_of(h), h)
        papers = dedupe([doc_id_of(h) for h in hits])[:pool_k]
        rng.shuffle(papers)
        for doc_id in papers:
            h = first_hit[doc_id]
            meta = h.get("metadata") or {}
            rows.append({"query_id": q["id"], "query": q["query"], "doc_id": doc_id,
                         "title": meta.get("title", ""), "category": meta.get("category", ""),
                         "snippet": " ".join(h["text"].split())[:400], "relevant": ""})
    return rows


def write_sheet(path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SHEET_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def read_judgments(path, allow_unjudged: bool = False):
    """Read the judged sheet. Returns (topical_queries, report).

    A query with no relevant paper is dropped (recall is undefined for it) and listed in report['dropped'].
    A blank 'relevant' cell is an error, unless allow_unjudged is true (then it counts as not relevant).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Judging sheet not found: {path}. Create it with the 'pool-sheet' command first.")
    relevant, text_of, unjudged, bad = {}, {}, [], []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = set(SHEET_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"The sheet is missing columns: {sorted(missing)}")
        for line, row in enumerate(reader, start=2):
            qid, value = row["query_id"].strip(), (row["relevant"] or "").strip().lower()
            text_of[qid] = row["query"]
            relevant.setdefault(qid, set())
            if value in YES:
                relevant[qid].add(normalize_doc_id(row["doc_id"]))
            elif value in NO:
                pass
            elif not value:
                unjudged.append(f"line {line} ({qid}, {row['doc_id']})")
            else:
                bad.append(f"line {line}: '{row['relevant']}'")
    if bad:
        raise ValueError("Use y or n in the 'relevant' column. Problems: " + "; ".join(bad[:10]))
    if unjudged and not allow_unjudged:
        raise ValueError(f"{len(unjudged)} rows are not judged yet, for example {unjudged[:3]}. "
                         "Fill in y or n, or use --allow-unjudged to count them as not relevant.")
    queries = [{"id": qid, "query": text_of[qid], "type": "topical", "subtype": "", "relevant": sorted(docs)}
               for qid, docs in relevant.items() if docs]
    report = {"queries_in_sheet": len(relevant), "kept": len(queries), "unjudged_rows": len(unjudged),
              "dropped": sorted(q for q, d in relevant.items() if not d)}
    return queries, report


def merge_queries(existing: list[dict], new_topical: list[dict]) -> list[dict]:
    """Keep the known-item queries, replace all topical queries with the new ones."""
    return [q for q in existing if q["type"] != "topical"] + new_topical
