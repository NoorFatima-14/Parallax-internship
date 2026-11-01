"""Measure topic quality, topic assignment accuracy and NER accuracy from your hand labels.

Run:  python scripts/evaluate_labels.py
Writes logs/nlp_metadata_evaluation.md and logs/nlp_metadata_evaluation.json.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nlpmeta import cli  # noqa: E402
from nlpmeta.labeling import parse_assignments, parse_ner_gold, parse_topic_review, read_sheet  # noqa: E402
from nlpmeta.metrics import wilson_interval  # noqa: E402
from nlpmeta.ner import SYSTEMS, score_system, system_predictions  # noqa: E402

MIN_RECOMMENDED = {"topics": 20, "assignments": 40, "ner": 30}


def pct(x) -> str:
    return f"{100 * x:.0f}%"


def interval(k, n) -> str:
    lo, hi = wilson_interval(k, n)
    return f"{pct(lo)} to {pct(hi)}"


def topic_section(P, report, lines, notes):
    path = P.labels / "topic_review.csv"
    if not path.exists():
        return
    review, problems = parse_topic_review(read_sheet(path))
    notes += problems
    topics = {t["topic_id"]: t for t in json.loads(P.topics.read_text(encoding="utf-8"))["topics"]}
    if not review:
        notes.append("topic_review.csv has no ratings yet")
        return
    n = len(review)
    counts = {c: sum(r["coherence"] == c for r in review) for c in ("good", "mixed", "poor")}
    docs = {c: sum(topics[r["topic_id"]]["size"] for r in review if r["coherence"] == c and r["topic_id"] in topics)
            for c in counts}
    covered = sum(docs.values())
    lines += ["## Topic quality (your ratings)", "", f"{n} topics rated (the largest first).", "",
              "| Rating | Topics | Share of rated topics (95% interval) | Documents in these topics |",
              "|---|---|---|---|"]
    for c in ("good", "mixed", "poor"):
        lines.append(f"| {c} | {counts[c]} | {pct(counts[c] / n)} ({interval(counts[c], n)}) | "
                     f"{docs[c]} ({pct(docs[c] / covered) if covered else '-'}) |")
    npmi = {c: [topics[r["topic_id"]]["npmi"] for r in review
                if r["coherence"] == c and topics.get(r["topic_id"], {}).get("npmi") is not None] for c in counts}
    lines += ["", "Does the automatic score (NPMI) agree with your ratings?", "", "| Rating | Mean NPMI |", "|---|---|"]
    for c in ("good", "mixed", "poor"):
        lines.append(f"| {c} | {np.mean(npmi[c]):.3f} |" if npmi[c] else f"| {c} | - |")
    named = [(r["topic_id"], r["human_label"]) for r in review if r["human_label"]]
    if named:
        lines += ["", "Names you gave to topics: " + "; ".join(f"{i}: {name}" for i, name in named[:30])]
    lines.append("")
    report["topics"] = {"rated": n, **counts, "documents": docs, "mean_npmi": {c: (float(np.mean(v)) if v else None) for c, v in npmi.items()}}
    if n < MIN_RECOMMENDED["topics"]:
        notes.append(f"only {n} topics rated (at least {MIN_RECOMMENDED['topics']} recommended)")


def assignment_section(P, report, lines, notes):
    path = P.labels / "assignment_check.csv"
    if not path.exists():
        return
    done, problems = parse_assignments(read_sheet(path))
    notes += problems
    if not done:
        notes.append("assignment_check.csv has no answers yet")
        return
    lines += ["## Topic assignment accuracy (your judgement)", "", "| Group | Documents | Correct (y) | Correct or partly (y or p) |", "|---|---|---|---|"]
    result = {}
    for name, rows in [("all", done)] + [(s, [r for r in done if r["stratum"] == s]) for s in ("random", "short")]:
        if not rows:
            continue
        n = len(rows)
        yes = sum(r["score"] == 1.0 for r in rows)
        ok = sum(r["score"] >= 0.5 for r in rows)
        result[name] = {"n": n, "correct": yes, "correct_or_partly": ok}
        lines.append(f"| {name} | {n} | {pct(yes / n)} ({interval(yes, n)}) | {pct(ok / n)} ({interval(ok, n)}) |")
    lines += ["", "'short' documents are the shortest 10% of the corpus. They were not used to fit the topic model.", ""]
    report["assignment"] = result
    if len(done) < MIN_RECOMMENDED["assignments"]:
        notes.append(f"only {len(done)} assignments labeled (at least {MIN_RECOMMENDED['assignments']} recommended)")


def ner_section(P, report, lines, notes):
    path = P.labels / "ner_gold.csv"
    if not path.exists() or not P.entities.exists():
        return
    gold, strata, problems = parse_ner_gold(read_sheet(path))
    notes += problems
    if not gold:
        notes.append("ner_gold.csv has no labeled rows yet (mark finished rows with x in the 'labeled' column)")
        return
    predictions = system_predictions(pd.read_parquet(P.entities))
    subsets = [("all labeled documents", list(gold)),
               ("random documents only", [d for d in gold if strata.get(d) == "random"])]
    n_gold = sum(len(v) for v in gold.values())
    lines += ["## Dataset-mention extraction (your labels)", "",
              f"{len(gold)} documents labeled, {n_gold} dataset mentions in total, "
              f"{sum(bool(v) for v in gold.values())} documents mention at least one dataset.", ""]
    report["ner"] = {"documents": len(gold), "gold_mentions": n_gold, "systems": {}}
    best = None
    for title, ids in subsets:
        if not ids:
            continue
        lines += [f"**{title}** ({len(ids)} documents)", "",
                  "| System | Precision | Recall | F1 | Correct | Wrong | Missed | Precision interval | Recall interval |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for system in SYSTEMS:
            r = score_system(gold, predictions[system], ids)
            lines.append(f"| {system} | {r['precision']:.2f} | {r['recall']:.2f} | {r['f1']:.2f} | {r['tp']} | {r['fp']} | {r['fn']} | "
                         f"{interval(r['tp'], r['tp'] + r['fp'])} | {interval(r['tp'], r['tp'] + r['fn'])} |")
            report["ner"]["systems"].setdefault(system, {})[title] = {k: r[k] for k in ("tp", "fp", "fn", "precision", "recall", "f1")}
            if title.startswith("all") and (best is None or r["f1"] > best[1]["f1"]):
                best = (system, r)
        lines.append("")
    if best:
        system, r = best
        lines += [f"## Mistakes of the best system ({system})", "", "Dataset names are shown in normalized form (lower case, no spaces or hyphens).", "",
                  "Wrong (the system found it, you did not):", ""]
        lines += [f"- {key}  ({doc})" for doc, key in r["false_positives"][:20]] or ["- none"]
        lines += ["", "Missed (you found it, the system did not):", ""]
        lines += [f"- {key}  ({doc})" for doc, key in r["false_negatives"][:20]] or ["- none"]
        lines.append("")
    if n_gold < MIN_RECOMMENDED["ner"]:
        notes.append(f"only {n_gold} gold mentions (at least {MIN_RECOMMENDED['ner']} recommended): recall and precision are unreliable")


def main() -> None:
    p = argparse.ArgumentParser()
    cli.add_path_args(p)
    args = p.parse_args()
    P = cli.resolve(args)
    cli.need(P.topics, "Run scripts/run_topic_model.py first.")

    lines, report, notes = ["# NLP Metadata Evaluation", "",
                            "All numbers come from labels that a person created (see labels/GUIDELINES.md). "
                            "Intervals are 95% Wilson intervals: with small samples they are wide, and that is the honest picture.", ""], {}, []
    topic_section(P, report, lines, notes)
    assignment_section(P, report, lines, notes)
    ner_section(P, report, lines, notes)
    if not report:
        sys.exit("No labels found yet. Run make_label_sheets.py, fill the sheets in labels/, then run this again.")
    if notes:
        lines += ["## Things to check", ""] + [f"- {n}" for n in notes] + [""]
    text = "\n".join(lines) + "\n"
    (P.logs / "nlp_metadata_evaluation.md").write_text(text, encoding="utf-8")
    cli.save_json(P.logs / "nlp_metadata_evaluation.json", report)
    print(text)


if __name__ == "__main__":
    main()
