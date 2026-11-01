"""Label sheets: create them for hand labeling, and read them back safely.

The sheets are CSV files that open in Excel. A person fills one column per sheet.
The NER sheet does not show any system predictions, so labeling is not influenced by them.
"""

import csv
from pathlib import Path

import numpy as np

from nlpmeta.datasets import canonicalize, normalize_key

TOPIC_COLS = ["topic_id", "size", "top_words", "example_titles", "coherence", "human_label", "notes"]
ASSIGN_COLS = ["doc_id", "title", "abstract", "assigned_topic_id", "assigned_topic_words",
               "correct", "notes", "stratum", "words"]
NER_COLS = ["doc_id", "title", "abstract", "datasets", "labeled", "notes", "stratum"]

COHERENCE_VALUES = {"good": "good", "g": "good", "mixed": "mixed", "m": "mixed", "poor": "poor", "p": "poor"}
CORRECT_VALUES = {"y": 1.0, "yes": 1.0, "n": 0.0, "no": 0.0, "p": 0.5, "partly": 0.5, "partial": 0.5}


# ---------------------------------------------------------------- csv io
def write_sheet(path, columns, rows, force: bool = False) -> None:
    path = Path(path)
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists. Use --force to overwrite it (this deletes your labels).")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:  # BOM so Excel reads accents correctly
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_sheet(path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        return [{k.strip(): (v or "").strip() for k, v in row.items() if k} for row in csv.DictReader(f)]


# ---------------------------------------------------------------- sampling
def sample_docs(doc_table, n_random: int, n_short: int, seed: int = 42) -> list[tuple[int, str]]:
    """Row positions to label: random documents plus short documents. Returns (position, stratum)."""
    rng = np.random.default_rng(seed)
    short = np.where(doc_table["short_doc"].values)[0]
    normal = np.where(~doc_table["short_doc"].values)[0]
    picked_short = rng.choice(short, size=min(n_short, len(short)), replace=False) if len(short) else []
    picked_normal = rng.choice(normal, size=min(n_random, len(normal)), replace=False)
    rows = [(int(i), "random") for i in picked_normal] + [(int(i), "short") for i in picked_short]
    rng.shuffle(rows)
    return rows


def sample_ner_docs(entities, n_random: int, n_likely: int, seed: int = 42) -> list[tuple[int, str]]:
    """Random documents plus documents where some system found a dataset (so there is something to score)."""
    rng = np.random.default_rng(seed)
    likely_mask = entities["datasets"].map(len).values > 0
    likely = np.where(likely_mask)[0]
    picked_likely = rng.choice(likely, size=min(n_likely, len(likely)), replace=False) if len(likely) else []
    taken = set(int(i) for i in picked_likely)
    pool = np.array([i for i in range(len(entities)) if i not in taken])
    picked_random = rng.choice(pool, size=min(n_random, len(pool)), replace=False)
    rows = [(int(i), "random") for i in picked_random] + [(int(i), "likely_mention") for i in picked_likely]
    rng.shuffle(rows)
    return rows


# ---------------------------------------------------------------- reading labels
def parse_topic_review(rows):
    """Returns (list of dicts with topic_id, coherence, human_label), list of problems)."""
    done, problems = [], []
    for row in rows:
        value = row.get("coherence", "").lower()
        if not value:
            continue
        if value not in COHERENCE_VALUES:
            problems.append(f"topic {row.get('topic_id')}: coherence '{value}' is not good/mixed/poor")
            continue
        done.append({"topic_id": int(row["topic_id"]), "coherence": COHERENCE_VALUES[value],
                     "human_label": row.get("human_label", "")})
    return done, problems


def parse_assignments(rows):
    """Returns (list of dicts with doc_id, score, stratum), list of problems)."""
    done, problems = [], []
    for row in rows:
        value = row.get("correct", "").lower()
        if not value:
            continue
        if value not in CORRECT_VALUES:
            problems.append(f"{row.get('doc_id')}: correct '{value}' is not y/n/p")
            continue
        done.append({"doc_id": row["doc_id"], "score": CORRECT_VALUES[value],
                     "stratum": row.get("stratum", ""), "topic_id": int(row["assigned_topic_id"])})
    return done, problems


def parse_ner_gold(rows):
    """Returns ({doc_id: set of keys}, {doc_id: stratum}, problems).

    A document counts as labeled only when the 'labeled' column is not empty. This lets an
    empty 'datasets' cell mean 'no dataset is mentioned'.
    """
    gold, strata, problems = {}, {}, []
    unlabeled = 0
    for row in rows:
        if not row.get("labeled", ""):
            unlabeled += 1
            continue
        names = [n.strip() for n in row.get("datasets", "").replace("\n", ";").split(";") if n.strip()]
        gold[row["doc_id"]] = {normalize_key(canonicalize(n)) for n in names}
        strata[row["doc_id"]] = row.get("stratum", "")
    if unlabeled:
        problems.append(f"{unlabeled} document(s) have no mark in the 'labeled' column and were skipped")
    return gold, strata, problems
