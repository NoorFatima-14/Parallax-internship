"""Extract dataset/benchmark mentions (and general entities) from every document.

Run:  python scripts/run_ner.py
Needs the spaCy model:  python -m spacy download en_core_web_sm
Writes data/processed/doc_entities.parquet and logs/ner_summary.md.
"""

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nlpmeta import cli  # noqa: E402
from nlpmeta.ner import extract_corpus, load_spacy  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    cli.add_path_args(p)
    p.add_argument("--no-spacy", action="store_true", help="skip the spaCy baseline (faster)")
    args = p.parse_args()
    P = cli.resolve(args)

    corpus = pd.read_parquet(cli.need(P.corpus, "Build the Week 1 clean corpus first."))
    nlp = None if args.no_spacy else load_spacy()
    start = time.perf_counter()
    entities = extract_corpus(corpus, nlp)
    seconds = time.perf_counter() - start
    entities.to_parquet(P.entities, index=False)

    n = len(entities)
    seed_docs = int((entities["datasets_seed"].map(len) > 0).sum())
    any_docs = int((entities["datasets"].map(len) > 0).sum())
    names = Counter(x for row in entities["datasets_seed"] for x in row)
    discovered = Counter(x for row in entities["datasets_pattern"] for x in row)
    labels = Counter(x for row in entities["spacy_labels"] for x in row)
    lines = ["# NER Summary (latest run)", "",
             f"- Documents: {n}, time: {seconds:.0f} s", f"- Documents with a known dataset (seed list): {seed_docs} ({100 * seed_docs / n:.1f}%)",
             f"- Documents with any dataset (seed + patterns): {any_docs} ({100 * any_docs / n:.1f}%)",
             f"- Distinct known datasets found: {len(names)}. Distinct names found only by patterns: {len(discovered)}", "",
             "## Most common known datasets", "", "| Dataset | Documents |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in names.most_common(25)]
    lines += ["", "## Most common names found only by patterns (check these for mistakes)", "", "| Name | Documents |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in discovered.most_common(25)]
    if nlp is not None:
        lines += ["", "## spaCy entity labels (documents that contain each label)", "", "| Label | Documents |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in labels.most_common(12)]
    text = "\n".join(lines) + "\n"
    (P.logs / "ner_summary.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
