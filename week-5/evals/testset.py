"""The retrieval test set: queries with the papers that are relevant to each.

File format (JSON):

    {"version": 1, "created": "...", "notes": "...",
     "queries": [{"id": "ki-title-0001", "query": "...", "type": "known_item", "subtype": "title",
                  "relevant": ["2601.00001v1"]}, ...]}

type "known_item": one relevant paper (the one the query was made from), made automatically
type "topical":    one or more relevant papers, judged by a person (see scripts/make_retrieval_testset.py)
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from evals.metrics import normalize_doc_id

QUERY_TYPES = ("known_item", "topical")
VERSION = 1


def validate_queries(queries) -> list[dict]:
    """Check the queries and return clean copies. Raises ValueError with the position of the first problem."""
    if not isinstance(queries, list) or not queries:
        raise ValueError("The test set has no queries.")
    seen, clean = set(), []
    for i, q in enumerate(queries):
        where = f"query #{i + 1}"
        if not isinstance(q, dict):
            raise ValueError(f"{where}: must be an object.")
        qid = q.get("id")
        if not isinstance(qid, str) or not qid.strip():
            raise ValueError(f"{where}: 'id' must be a non-empty string.")
        if qid in seen:
            raise ValueError(f"{where}: duplicate id '{qid}'.")
        seen.add(qid)
        text = q.get("query")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{qid}: 'query' must be a non-empty string.")
        if q.get("type") not in QUERY_TYPES:
            raise ValueError(f"{qid}: 'type' must be one of {QUERY_TYPES}.")
        relevant = q.get("relevant")
        if not isinstance(relevant, list) or not relevant or not all(isinstance(r, str) and r.strip() for r in relevant):
            raise ValueError(f"{qid}: 'relevant' must be a non-empty list of paper ids.")
        clean.append({"id": qid, "query": text.strip(), "type": q["type"], "subtype": str(q.get("subtype") or ""),
                      "relevant": sorted({normalize_doc_id(r) for r in relevant})})
    return clean


def load_testset(path) -> list[dict]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Test set not found: {path}. Create it with scripts/make_retrieval_testset.py.")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    return validate_queries(data.get("queries") if isinstance(data, dict) else data)


def save_testset(path, queries, notes: str = "") -> None:
    queries = validate_queries(queries)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": VERSION, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "notes": notes, "queries": queries}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
