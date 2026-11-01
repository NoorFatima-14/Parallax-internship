"""Entity extraction for the corpus, and scoring against a hand-labeled set.

Three systems are compared on dataset/benchmark mentions:
1. spaCy baseline: the general English model, keeping entities it labels ORG, PRODUCT,
   WORK_OF_ART or EVENT (that is where it puts names like ImageNet).
2. seed list: the known datasets from nlpmeta.datasets
3. seed list + patterns
"""

import pandas as pd

from nlpmeta.datasets import canonicalize, extract_datasets, normalize_key
from nlpmeta.metrics import prf

BASELINE_LABELS = {"ORG", "PRODUCT", "WORK_OF_ART", "EVENT"}
SYSTEMS = ("spacy_baseline", "seed_list", "seed_plus_patterns")


def load_spacy(model: str = "en_core_web_sm"):
    import spacy

    try:
        return spacy.load(model, disable=["lemmatizer"])
    except OSError as exc:
        raise SystemExit(f"spaCy model '{model}' is not installed. Run: python -m spacy download {model}") from exc


def spacy_entities(nlp, texts, batch_size: int = 64):
    """List of [(text, label), ...] for each text."""
    return [[(e.text, e.label_) for e in doc.ents] for doc in nlp.pipe(texts, batch_size=batch_size)]


def extract_corpus(df: pd.DataFrame, nlp=None) -> pd.DataFrame:
    """One row per document: dataset mentions from each system, plus all spaCy entities."""
    texts = [f"{t}. {x}" for t, x in zip(df["title"], df["text"])]
    ents = spacy_entities(nlp, texts) if nlp is not None else [[] for _ in texts]
    rows = []
    for doc_id, text, doc_ents in zip(df["id"], texts, ents):
        found = extract_datasets(text)
        baseline = sorted({canonicalize(t) for t, label in doc_ents if label in BASELINE_LABELS})
        rows.append({
            "doc_id": doc_id,
            "spacy_baseline": baseline,
            "datasets_seed": found["seed"],
            "datasets_pattern": found["pattern"],
            "datasets": found["all"],
            "spacy_labels": sorted({label for _, label in doc_ents}),
            "n_spacy_entities": len(doc_ents),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- scoring
def _keys(names) -> set:
    return {normalize_key(canonicalize(n)) for n in names if n and n.strip()}


def system_predictions(entities: pd.DataFrame) -> dict:
    """{system: {doc_id: set of keys}}"""
    out = {s: {} for s in SYSTEMS}
    for row in entities.itertuples(index=False):
        out["spacy_baseline"][row.doc_id] = _keys(row.spacy_baseline)
        out["seed_list"][row.doc_id] = _keys(row.datasets_seed)
        out["seed_plus_patterns"][row.doc_id] = _keys(row.datasets)
    return out


def score_system(gold: dict, predicted: dict, doc_ids=None) -> dict:
    """Micro precision/recall/F1 over (document, dataset) pairs. Also returns the mistakes."""
    tp = fp = fn = 0
    false_pos, false_neg = [], []
    for doc_id in (doc_ids if doc_ids is not None else gold):
        g, p = gold.get(doc_id, set()), predicted.get(doc_id, set())
        tp += len(g & p)
        fp += len(p - g)
        fn += len(g - p)
        false_pos += [(doc_id, k) for k in sorted(p - g)]
        false_neg += [(doc_id, k) for k in sorted(g - p)]
    result = prf(tp, fp, fn)
    result["false_positives"], result["false_negatives"] = false_pos, false_neg
    return result
