"""Turn friendly filter options into a ChromaDB 'where' clause, and check the names people type."""

import difflib
from dataclasses import dataclass, field

from nlpmeta.datasets import canonicalize
from nlpmeta.enrich import dataset_slug


@dataclass
class Filters:
    topics: list = field(default_factory=list)       # topic ids
    categories: list = field(default_factory=list)   # arXiv categories, e.g. cs.CV
    datasets: list = field(default_factory=list)     # dataset flag keys, e.g. ds_imagenet
    datasets_mode: str = "any"                       # 'any' or 'all'
    year_from: int = None
    year_to: int = None
    core_only: bool = False                          # only documents that belong to a core cluster
    exclude_short: bool = False

    def is_empty(self) -> bool:
        return not (self.topics or self.categories or self.datasets or self.year_from
                    or self.year_to or self.core_only or self.exclude_short)


def build_where(f: Filters):
    """ChromaDB where clause, or None when there is no filter."""
    if f.datasets_mode not in ("any", "all"):
        raise ValueError("datasets_mode must be 'any' or 'all'")
    if f.year_from and f.year_to and f.year_from > f.year_to:
        raise ValueError("year_from is after year_to")
    clauses = []
    if f.topics:
        clauses.append({"topic_id": {"$in": [int(t) for t in f.topics]}})
    if f.categories:
        clauses.append({"category": {"$in": list(f.categories)}})
    if f.datasets:
        flags = [{key: {"$eq": True}} for key in f.datasets]
        if len(flags) == 1:
            clauses.append(flags[0])
        else:
            clauses.append({"$or" if f.datasets_mode == "any" else "$and": flags})
    if f.year_from:
        clauses.append({"published_year": {"$gte": int(f.year_from)}})
    if f.year_to:
        clauses.append({"published_year": {"$lte": int(f.year_to)}})
    if f.core_only:
        clauses.append({"topic_core": {"$eq": True}})
    if f.exclude_short:
        clauses.append({"short_doc": {"$eq": False}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def resolve_topics(value, topics_table: list) -> list[int]:
    """A topic id ('7') or a word from its label ('diffusion') -> list of matching topic ids."""
    text = str(value).strip()
    if text.lstrip("-").isdigit():
        topic_id = int(text)
        if topic_id not in {t["topic_id"] for t in topics_table}:
            raise ValueError(f"There is no topic with id {topic_id} (ids go from 0 to {len(topics_table) - 1}).")
        return [topic_id]
    needle = text.lower()
    hits = [t["topic_id"] for t in topics_table
            if needle in t["label"].lower() or any(needle in w.lower() for w in t["words"])]
    if not hits:
        labels = [t["label"] for t in topics_table]
        close = difflib.get_close_matches(text, labels, n=3, cutoff=0.3)
        raise ValueError(f"No topic matches '{text}'." + (f" Close labels: {close}" if close else ""))
    return hits


def resolve_dataset(name: str, index: dict) -> str:
    """'MS COCO' -> 'ds_coco' (the flag key). Raises ValueError if that dataset has no flag."""
    key = dataset_slug(canonicalize(name))
    if key in index:
        return key
    names = [v["name"] for v in index.values()]
    close = difflib.get_close_matches(canonicalize(name), names, n=3, cutoff=0.4)
    raise ValueError(f"No flag for dataset '{name}' (it may be mentioned in fewer documents than the "
                     f"minimum)." + (f" Did you mean: {close}?" if close else ""))


def read_topics_table(path) -> list:
    import json

    return json.loads(open(path, encoding="utf-8").read())["topics"]


def read_datasets_index(path) -> dict:
    import json

    return json.loads(open(path, encoding="utf-8").read())


def matches(meta: dict, f: Filters) -> bool:
    """The same rules as build_where, applied in Python to one chunk's metadata.

    Used to check that a filtered search never returns a chunk that breaks the filter.
    """
    if f.topics and meta.get("topic_id") not in [int(t) for t in f.topics]:
        return False
    if f.categories and meta.get("category") not in f.categories:
        return False
    if f.datasets:
        hits = [bool(meta.get(key)) for key in f.datasets]
        if not (any(hits) if f.datasets_mode == "any" else all(hits)):
            return False
    year = meta.get("published_year")
    if f.year_from and not (year is not None and year >= f.year_from):
        return False
    if f.year_to and not (year is not None and year <= f.year_to):
        return False
    if f.core_only and meta.get("topic_core") is not True:
        return False
    if f.exclude_short and meta.get("short_doc") is not False:
        return False
    return True
