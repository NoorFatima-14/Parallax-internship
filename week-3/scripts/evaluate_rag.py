"""Evaluate the RAG system: answer quality checks, hallucination handling and latency.

Run:  python scripts/evaluate_rag.py                (about 28 questions, takes several minutes)
      python scripts/evaluate_rag.py --limit 2      (quick trial: 2 questions per group)
Free API plans allow only a few requests per minute, so requests that call the LLM are
spaced out with --delay.

Writes logs/rag_eval.json (every run) and logs/rag_eval_summary.md (latest run).
This is an automatic check, not a human rating: it measures whether answers are grounded in
the passages and whether off-topic questions are refused. Read some answers yourself too.
"""

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.app import build_pipeline  # noqa: E402
from rag.config import ConfigError, RAGConfig  # noqa: E402
from rag.retriever import week2  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ANSWERED = {"answered", "answered_with_warning"}
REFUSED = {"refused_out_of_domain", "refused_by_model"}
STAGES = ["embed_ms", "search_ms", "retrieval_ms", "llm_ms", "verify_ms", "total_ms"]


def load_questions(limit=None):
    data = json.loads((ROOT / "eval_questions.json").read_text(encoding="utf-8"))
    rows = []
    for group in ("in_domain", "out_of_domain", "near_domain_not_in_corpus"):
        for q in data[group][:limit]:
            rows.append({"group": group, "label": q[:50], "question": q,
                         "expect": "answer" if group == "in_domain" else "refuse"})
    for item in data["edge"][:limit]:
        question = item["question"] * item.get("repeat", 1)
        rows.append({"group": "edge", "label": item["label"], "question": question, "expect": item["expect"]})
    return rows


def meets_expectation(row, status) -> bool:
    expect = row["expect"]
    if expect == "answer":
        return status in ANSWERED
    if expect == "refuse":
        return status in REFUSED or status == "blocked_ungrounded"
    if expect == "invalid_query":
        return status == "invalid_query"
    return status not in ("internal_error", "retrieval_error")


def stats(values):
    if not values:
        return None
    a = np.asarray(values, dtype=float)
    return {"n": len(a), "mean": round(float(a.mean()), 1), "p50": round(float(np.percentile(a, 50)), 1),
            "p95": round(float(np.percentile(a, 95)), 1), "max": round(float(a.max()), 1)}


def pct(part, whole):
    return f"{100 * part / whole:.0f}% ({part}/{whole})" if whole else "-"


def summarize(results) -> dict:
    out = {}
    for group in ("in_domain", "out_of_domain", "near_domain_not_in_corpus", "edge"):
        rows = [r for r in results if r["group"] == group]
        if rows:
            out[group] = {"n": len(rows), "status": dict(Counter(r["status"] for r in rows)),
                          "meets_expectation": sum(r["ok"] for r in rows)}
    in_scores = [r["top_score"] for r in results if r["group"] == "in_domain" and r["top_score"] is not None]
    off_scores = [r["top_score"] for r in results
                  if r["group"] in ("out_of_domain",) and r["top_score"] is not None]
    out["top_scores"] = {"in_domain": stats(in_scores), "out_of_domain": stats(off_scores),
                         "in_domain_min": round(min(in_scores), 3) if in_scores else None,
                         "out_of_domain_max": round(max(off_scores), 3) if off_scores else None}
    with_llm = [r for r in results if r["llm"]]
    without = [r for r in results if not r["llm"] and r["status"] in REFUSED and "total_ms" in r["timings_ms"]]
    out["latency_with_llm"] = {k: stats([r["timings_ms"][k] for r in with_llm if k in r["timings_ms"]]) for k in STAGES}
    out["latency_retrieval_only"] = {k: stats([r["timings_ms"][k] for r in without if k in r["timings_ms"]])
                                      for k in ("embed_ms", "search_ms", "total_ms")}
    errors = Counter(r["error"]["kind"] for r in results if r["error"])
    out["api"] = {
        "llm_calls": len(with_llm), "errors": dict(errors),
        "retries": sum(r["llm"]["retries"] for r in with_llm),
        "seconds_waiting": round(sum(r["llm"]["wait_seconds"] for r in with_llm), 1),
        "fallbacks_used": sum(len(r["llm"]["fallbacks"]) for r in with_llm),
        "truncated": sum(bool(r["llm"]["truncated"]) for r in with_llm),
        "total_tokens": sum((r["llm"]["usage"] or {}).get("total_tokens", 0) for r in with_llm),
        "models": dict(Counter(r["llm"]["model"] for r in with_llm)),
    }
    answered = [r for r in results if r["grounding"] and r["status"] in ANSWERED]
    out["grounding"] = {"answers_checked": len(answered),
                        "mean_grounded_fraction": round(float(np.mean([r["grounding"]["grounded_fraction"] for r in answered])), 3) if answered else None}
    return out


def write_summary(path, header, results, s) -> str:
    L = ["# RAG Evaluation (latest run)", "",
         f"- Provider / models: {header['provider']} / {', '.join(header['models'])}",
         f"- Embedding model: `{header['embed_model']}`, chunks in database: {header['chunks']}",
         f"- Settings: top_k {header['config']['top_k']}, min_score {header['config']['min_score']}, "
         f"strict {header['config']['strict']}", "",
         "This is an automatic check. It does not replace reading the answers.", ""]
    groups = [("in_domain", "In-domain questions", "Expected: a grounded answer"),
              ("out_of_domain", "Off-topic questions", "Expected: a refusal"),
              ("near_domain_not_in_corpus", "Related but not in the corpus", "Expected: a refusal (a model may know the answer from training, which is exactly what should not be used)"),
              ("edge", "Edge cases", "Expected: handled without a crash")]
    for key, title, note in groups:
        if key not in s:
            continue
        g = s[key]
        L += [f"## {title}", "", note, "", f"- Met expectation: {pct(g['meets_expectation'], g['n'])}",
              "- Outcomes: " + ", ".join(f"{k} {v}" for k, v in sorted(g["status"].items())), ""]
        if key in ("out_of_domain", "near_domain_not_in_corpus"):
            leaked = [r for r in results if r["group"] == key and r["status"] in ANSWERED]
            L += [f"- Answered although it should refuse: {len(leaked)}"] + \
                 [f"  - {r['question']}" for r in leaked] + [""]
        if key == "in_domain":
            blocked = [r for r in results if r["group"] == key and r["status"] == "blocked_ungrounded"]
            L += [f"- Blocked by the hallucination checks: {len(blocked)}"] + \
                 [f"  - {r['question']} ({', '.join(r['grounding']['block_reasons'])})" for r in blocked] + [""]
    t = s["top_scores"]
    L += ["## Retrieval scores and the out-of-domain threshold", "",
          f"- Lowest top score among in-domain questions: {t['in_domain_min']}",
          f"- Highest top score among off-topic questions: {t['out_of_domain_max']}",
          f"- Threshold in use (min_score): {header['config']['min_score']}", "",
          "A good threshold sits between these two numbers.", ""]
    L += ["## Latency (milliseconds)", "", "Requests that called the LLM:", "",
          "| Stage | n | Mean | p50 | p95 | Max |", "|---|---|---|---|---|---|"]
    for k in STAGES:
        v = s["latency_with_llm"][k]
        if v:
            L.append(f"| {k[:-3]} | {v['n']} | {v['mean']} | {v['p50']} | {v['p95']} | {v['max']} |")
    L += ["", "Requests refused before the LLM (retrieval only):", "",
          "| Stage | n | Mean | p50 | p95 | Max |", "|---|---|---|---|---|---|"]
    for k, v in s["latency_retrieval_only"].items():
        if v:
            L.append(f"| {k[:-3]} | {v['n']} | {v['mean']} | {v['p50']} | {v['p95']} | {v['max']} |")
    a = s["api"]
    L += ["", "## API behaviour", "", f"- LLM calls: {a['llm_calls']}, models used: {a['models']}",
          f"- Retries: {a['retries']} (waited {a['seconds_waiting']} s), fallback models used: {a['fallbacks_used']}",
          f"- Errors: {a['errors'] or 'none'}, truncated answers: {a['truncated']}, total tokens: {a['total_tokens']}",
          f"- Mean share of answer sentences supported by the passages: {s['grounding']['mean_grounded_fraction']}", ""]
    text = "\n".join(L) + "\n"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")
    return text


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--provider", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--limit", type=int, default=None, help="questions per group")
    p.add_argument("--delay", type=float, default=10.0, help="seconds to wait after each LLM request")
    p.add_argument("--min-score", type=float, default=RAGConfig.min_score)
    p.add_argument("--no-strict", action="store_true")
    p.add_argument("--db-path", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--log", default=str(ROOT / "logs" / "rag_eval.json"))
    p.add_argument("--summary", default=str(ROOT / "logs" / "rag_eval_summary.md"))
    args = p.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    cfg = RAGConfig(min_score=args.min_score, strict=not args.no_strict)
    try:
        pipeline = build_pipeline(args.provider, args.model, args.db_path, None, None, args.device, cfg)
    except (ConfigError, FileNotFoundError, ValueError) as exc:
        sys.exit(f"Setup problem: {exc}")

    questions = load_questions(args.limit)
    print(f"Running {len(questions)} questions...")
    results = []
    for i, row in enumerate(questions, 1):
        r = pipeline.ask(row["question"])
        ok = meets_expectation(row, r["status"])
        print(f"[{i}/{len(questions)}] {r['status']:<24} {'ok ' if ok else 'CHECK'} {row['label']}")
        results.append({**row, "question": row["question"][:300], "status": r["status"], "ok": ok,
                        "top_score": r["top_score"], "answer": r["answer"][:600],
                        "unverified_answer": (r.get("unverified_answer") or "")[:600] or None,
                        "grounding": r["grounding"], "llm": r["llm"], "error": r["error"],
                        "n_sources": len(r["sources"]), "timings_ms": r["timings_ms"]})
        if r["llm"] and i < len(questions):
            time.sleep(args.delay)

    s = summarize(results)
    header = {"provider": pipeline.llm.provider, "models": pipeline.llm.models,
              "embed_model": pipeline.retriever.store.model_name, "chunks": pipeline.retriever.store.count(),
              "config": {"top_k": cfg.top_k, "min_score": cfg.min_score, "strict": cfg.strict}}
    text = write_summary(args.summary, header, results, s)
    week2().append_json_log(args.log, {**header, "summary": s, "results": results,
                                       "environment": week2().environment_info()})
    print("\n" + text)
    print(f"Saved {args.log} and {args.summary}")


if __name__ == "__main__":
    main()
