"""Evaluate the running API end to end: answer behaviour, answer quality checks and latency.

Start the server first (in another terminal, from inside week-5/):

    uvicorn api.main:create_app --factory --port 8000 --no-access-log

Then:

    python scripts/evaluate_e2e.py                      full run (about 40 requests, 20+ call the LLM)
    python scripts/evaluate_e2e.py --limit 2            quick trial: 2 cases per group
    python scripts/evaluate_e2e.py --groups in_domain off_topic
    python scripts/evaluate_e2e.py --delay 15           wait longer between LLM requests (free-tier limits)
    python scripts/evaluate_e2e.py --strict             exit with code 1 if any case fails (for CI)

Needs the Groq (or OpenRouter) key in week-3/.env on the server side. Writes logs/e2e_eval.md (latest run)
and appends to logs/e2e_eval.json.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402

from api.paths import add_other_weeks  # noqa: E402

add_other_weeks()
from evals.e2e_eval import json_record, limit_per_group, load_cases, run_all, summarize, write_report  # noqa: E402
from rag.retriever import week2  # noqa: E402


def server_description(client) -> str:
    """One line about the server, taken from GET /ready (also proves that it is up and ready)."""
    r = client.get("/ready")
    if r.status_code != 200:
        try:
            message = r.json()["error"]["message"]
        except (ValueError, KeyError, TypeError):
            message = r.text[:200]
        sys.exit(f"The API answered HTTP {r.status_code} on /ready, so it is not ready: {message}")
    b = r.json()
    s = b.get("settings", {})
    return (f"{b['chunks']} chunks, embedding `{b.get('embedding_model')}`, LLM {b.get('llm_provider')} "
            f"{', '.join(b.get('llm_models', []))}, top_k {s.get('top_k')}, min_score {s.get('min_score')}, "
            f"strict {s.get('strict')}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--cases", default=str(ROOT / "eval_data" / "e2e_cases.json"))
    p.add_argument("--groups", nargs="+", default=None, help="run only these groups (for example in_domain off_topic)")
    p.add_argument("--limit", type=int, default=None, help="use only the first N cases of each group")
    p.add_argument("--delay", type=float, default=10.0,
                   help="seconds to wait after each request that called the LLM (free tiers allow about 30 per minute)")
    p.add_argument("--timeout", type=float, default=120.0, help="seconds to wait for one response")
    p.add_argument("--strict", action="store_true", help="exit with code 1 if any case fails")
    p.add_argument("--report", default=str(ROOT / "logs" / "e2e_eval.md"))
    p.add_argument("--log", default=str(ROOT / "logs" / "e2e_eval.json"))
    args = p.parse_args()

    try:
        cases = load_cases(args.cases)
    except (OSError, ValueError) as exc:
        sys.exit(f"Setup problem with the case file: {exc}")
    if args.groups:
        cases = [c for c in cases if c["group"] in args.groups]
    if args.limit:
        cases = limit_per_group(cases, args.limit)
    if not cases:
        sys.exit("No cases left after --groups / --limit.")

    with httpx.Client(base_url=args.base_url, timeout=args.timeout) as client:
        try:
            server = server_description(client)
        except httpx.TransportError as exc:
            sys.exit(f"Cannot reach the API at {args.base_url} ({type(exc).__name__}). Start it with:\n"
                     "  uvicorn api.main:create_app --factory --port 8000 --no-access-log")
        client.post("/search", json={"query": "warm up query", "top_k": 1})  # first request loads caches

        total = len(cases)
        print(f"Server: {server}\nRunning {total} cases against {args.base_url} (delay {args.delay} s after LLM calls)\n")

        def show(r):
            n = len(shown) + 1
            shown.append(r)
            mark = "PASS" if r["passed"] else "FAIL"
            print(f"[{n:>2}/{total}] {r['id']:<10} {r['path']:<8} HTTP {r['http'] or '---'} "
                  f"{(r['status'] or r['error_code'] or ''):<22} {r['client_ms']:>7.0f} ms  {mark}"
                  + ("" if r["passed"] else f"  <- {r['reason']}"), flush=True)

        shown = []
        results = run_all(client, cases, delay=args.delay, progress=show)

    summary = summarize(results)
    header = {"base_url": args.base_url, "cases": Path(args.cases).name, "server": server}
    text = write_report(args.report, summary, header)
    w2 = week2()
    w2.append_json_log(args.log, {**header, **json_record(summary, results), "environment": w2.environment_info()})
    print("\n" + text)
    print(f"Saved {args.report} and {args.log}")
    if args.strict and (summary["failures"] or summary["server_errors"]):
        sys.exit(1)


if __name__ == "__main__":
    main()
