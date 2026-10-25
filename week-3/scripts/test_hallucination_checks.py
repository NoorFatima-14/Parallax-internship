"""Test the hallucination checks on purpose: do they catch wrong answers?

In the evaluation, every answer the model wrote was well supported, so the checks never
had to block anything. This script feeds them answers that are WRONG on purpose, written
against real passages from your database. No LLM calls are made, so it is free and fast.

Each answer ends up as one of three outcomes:
  blocked  the answer would be hidden
  warned   the answer would be shown with a warning (for example "partly unverified")
  clean    the answer would be shown with no warning

Run:  python scripts/test_hallucination_checks.py
Writes logs/hallucination_check_summary.md.
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.config import RAGConfig  # noqa: E402
from rag.guardrails import check_grounding, split_sentences  # noqa: E402
from rag.retriever import Retriever  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# case -> (what should happen, description)
#   "blocked": must be blocked   "clean": must pass with no warning
#   "flagged": ideally blocked or warned (reported, not counted as a failure)
#   "info":    only reported
CASES = {
    "true statement (control)": ("clean", "A real sentence from the top passage, correctly cited."),
    "invented number": ("blocked", "A true sentence plus a made-up accuracy and test-set size."),
    "invalid citation": ("blocked", "A true sentence cited to a passage number that does not exist."),
    "unrelated facts": ("blocked", "History and volcano sentences cited as if from the passages."),
    "answer to another question": ("blocked", "A true sentence taken from the passages of a different question."),
    "true + invented sentence": ("flagged", "One true sentence and one on-topic invented claim, no numbers."),
    "generic on-topic text": ("info", "Vague but on-topic sentences that say nothing checkable."),
}


def first_sentence(text: str) -> str:
    for sentence in split_sentences(text):
        if len(sentence.split()) >= 6:
            return sentence
    return text


def outcome(g: dict) -> str:
    return "blocked" if g["blocked"] else ("warned" if g["warnings"] else "clean")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--questions", type=int, default=8)
    p.add_argument("--db-path", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--min-score", type=float, default=RAGConfig.min_score)
    args = p.parse_args()

    cfg = RAGConfig(min_score=args.min_score)
    retriever = Retriever.from_disk(args.db_path, device=args.device)
    questions = json.loads((ROOT / "eval_questions.json").read_text(encoding="utf-8"))["in_domain"][: args.questions]

    sets = []
    for q in questions:
        hits, _ = retriever.search(q, cfg.top_k)
        kept = [h for h in hits if h["score"] >= cfg.min_score]
        if kept:
            sets.append((q, kept))
    if len(sets) < 2:
        sys.exit("Need at least two questions with retrieved passages.")

    results = defaultdict(list)  # case -> list of (outcome, question, answer, reasons)
    for i, (question, kept) in enumerate(sets):
        true_sentence = re.sub(r"\s*\[\d+\]", "", first_sentence(kept[0]["text"]))
        other_sentence = re.sub(r"\s*\[\d+\]", "", first_sentence(sets[(i + 1) % len(sets)][1][0]["text"]))
        answers = {
            "true statement (control)": f"{true_sentence} [1]",
            "invented number": f"{true_sentence} [1] It reached 87.43% accuracy on 12,517 test images [1].",
            "invalid citation": f"{true_sentence} [9]",
            "unrelated facts": "The Roman Empire expanded across the Mediterranean under Augustus [1]. "
                               "Volcanic eruptions can release large amounts of ash into the atmosphere [2].",
            "answer to another question": f"{other_sentence} [1]",
            "true + invented sentence":
                f"{true_sentence} [1] This approach was first proposed by researchers at a large American "
                "university and is now used in most commercial products [1].",
            "generic on-topic text":
                "Researchers study this topic by combining several complementary approaches and comparing "
                "them on public benchmarks [1]. Results depend on the data, the model and the evaluation "
                "setting [1].",
        }
        for case, answer in answers.items():
            g = check_grounding(answer, kept, retriever.encode, cfg)
            results[case].append((outcome(g), question, answer, g["block_reasons"] or g["warnings"]))

    lines = ["# Hallucination Check Test", "",
             f"{len(sets)} questions, real passages from the database, wrong answers written on purpose. "
             "No LLM calls.", "",
             "| Case | Blocked | Warned | Clean | Expected | Result |", "|---|---|---|---|---|---|"]
    problems = []
    for case, (expect, _) in CASES.items():
        counts = Counter(o for o, *_ in results[case])
        n = len(results[case])
        if expect == "blocked":
            bad = [r for r in results[case] if r[0] != "blocked"]
            verdict = "pass" if not bad else f"MISSED {len(bad)}"
        elif expect == "clean":
            bad = [r for r in results[case] if r[0] != "clean"]
            verdict = "pass" if not bad else f"FALSE ALARM {len(bad)}"
        elif expect == "flagged":
            bad = [r for r in results[case] if r[0] == "clean"]
            verdict = f"flagged {n - len(bad)}/{n} (not counted)"
        else:
            bad = []
            verdict = "information only"
        if expect in ("blocked", "clean"):
            problems += [(case, *r) for r in bad]
        label = {"blocked": "blocked", "clean": "clean", "flagged": "blocked or warned", "info": "-"}[expect]
        lines.append(f"| {case} | {counts['blocked']}/{n} | {counts['warned']}/{n} | {counts['clean']}/{n} "
                     f"| {label} | {verdict} |")

    lines += ["", "Cases that failed their expectation:", ""]
    for case, outcome_name, question, answer, reasons in problems:
        lines.append(f"- {case} ({outcome_name}) | {question} | reasons: {reasons}\n  - {answer[:220]}")
    if not problems:
        lines.append("- none")
    text = "\n".join(lines) + "\n"
    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / "logs" / "hallucination_check_summary.md").write_text(text, encoding="utf-8")
    print(text)
    print("Saved logs/hallucination_check_summary.md")


if __name__ == "__main__":
    main()