"""Check that headline numbers in docs/EVALUATION.md match the log files. Run from the repository root:

    python docs/check_numbers.py
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
doc = (ROOT / "docs" / "EVALUATION.md").read_text(encoding="utf-8")


def read(rel: str) -> str:
    """Return the text of a log file given its path from the repository root."""
    return (ROOT / rel).read_text(encoding="utf-8")


CHECKS = [
    ("week-5/logs/retrieval_eval.md", ["MRR: 0.871", "0.825 (0.77-0.88)", "0.905 (0.86-0.94)", "0.950 (0.88-1.00)", "mean 56 ms"]),
    ("week-5/logs/e2e_eval.md", ["**40 of 40**", "| ask, answered | 13 | 1257 | 1202 | 1744 | 1968"]),
    ("week-3/logs/rag_eval_summary.md", ["100% (10/10)", "100% (8/8)", "100% (3/3)", "100% (7/7)", "0.473", "0.429", "19085"]),
    ("week-4/logs/topic_model_comparison.md", ["| bertopic | 31 | 0.634 | 139 | 0.2141 | 0.8258 | 107.0 |", "| lda | 25 | 1.0 | 73 | 0.0833"]),
    ("week-4/logs/nlp_metadata_evaluation.md", ["| good | 27 |", "| seed_plus_patterns | 0.56 | 0.45 | 0.50 |"]),
    ("week-2/logs/benchmark_summary.md", ["| one keyword | 12 | 27.97", "| off-topic | 30 | 28.23"]),
]

bad = 0
for rel, needles in CHECKS:
    text = read(rel)
    for n in needles:
        if n not in text:
            print(f"NOT IN LOG  {rel}: {n}")
            bad += 1

# every number quoted in the doc's headline claims
for claim in ["27,329", "6,563", "1257", "1744", "0.214", "0.871", "0.863"]:
    if claim not in doc:
        print(f"NOT IN DOC  {claim}")
        bad += 1

ingest = json.loads(read("week-2/logs/ingest_log.json"))
last = ingest[-1] if isinstance(ingest, list) else ingest
if (last["chunks"], last["documents"]) != (27329, 6563):
    print("ingest log differs from the documented corpus size")
    bad += 1

print("All checks passed." if not bad else f"{bad} problem(s).")
sys.exit(1 if bad else 0)
