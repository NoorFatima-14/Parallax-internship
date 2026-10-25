"""Ask a question. The answer is generated from the retrieved arXiv passages.

Run:  python scripts/ask.py "How are diffusion models used for image generation?"
      python scripts/ask.py                 (interactive: type questions, empty line to quit)
      python scripts/ask.py "..." --provider openrouter --category cs.CV
Every request is logged with its latency to logs/rag_log.json.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.app import build_pipeline  # noqa: E402
from rag.config import ConfigError, RAGConfig  # noqa: E402


def show(result: dict, show_blocked: bool = False) -> None:
    print(f"\n{result['answer']}")
    if result["status"] in ("answered", "answered_with_warning", "refused_by_model"):
        for s in result["sources"]:
            print(f"  [{s['n']}] {s['title'][:80]} (arXiv {s['doc_id']}, {s['category']}, score {s['score']})")
    if result.get("warnings"):
        print(f"  Warnings: {', '.join(result['warnings'])}")
    if result["status"] == "blocked_ungrounded" and show_blocked:
        print(f"\n  Blocked answer ({', '.join(result['grounding']['block_reasons'])}):\n  {result['unverified_answer']}")
    if result["status"] == "llm_error":
        print(f"  Error: {result['error']['message']}")
        for p in result.get("fallback_passages", []):
            print(f"  - {p['title'][:70]}: {p['text'][:150]}...")
    t = result["timings_ms"]
    parts = [f"{k[:-3]} {t[k]} ms" for k in ("embed_ms", "search_ms", "llm_ms", "verify_ms") if k in t]
    print(f"\n  status: {result['status']} | top score: {result['top_score']} | "
          f"{' + '.join(parts)} | total {t['total_ms']} ms")
    if result["llm"]:
        llm = result["llm"]
        print(f"  model: {llm['model']} | tokens: {llm['usage'].get('total_tokens', '?')} | "
              f"retries: {llm['retries']} (waited {llm['wait_seconds']} s)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("question", nargs="?")
    p.add_argument("--provider", default=None, help="groq or openrouter (default: groq)")
    p.add_argument("--model", default=None, help="model ID, or several separated by commas (fallbacks)")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--min-score", type=float, default=RAGConfig.min_score)
    p.add_argument("--category", default=None, help="only use one arXiv category, e.g. cs.CV")
    p.add_argument("--no-strict", action="store_true", help="show answers even if the checks fail")
    p.add_argument("--show-blocked", action="store_true", help="print answers that the checks blocked")
    p.add_argument("--db-path", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--no-log", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    cfg = RAGConfig(top_k=args.top_k, min_score=args.min_score, strict=not args.no_strict)
    log_path = None if args.no_log else Path(__file__).resolve().parent.parent / "logs" / "rag_log.json"
    try:
        pipeline = build_pipeline(args.provider, args.model, args.db_path, None, None,
                                  args.device, cfg, log_path)
    except (ConfigError, FileNotFoundError, ValueError) as exc:
        sys.exit(f"Setup problem: {exc}")

    where = {"category": args.category} if args.category else None
    if args.question:
        show(pipeline.ask(args.question, where=where), args.show_blocked)
        return
    print("Type a question (empty line to quit).")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break
        show(pipeline.ask(question, where=where), args.show_blocked)


if __name__ == "__main__":
    main()
