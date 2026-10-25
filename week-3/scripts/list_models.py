"""List the models a provider offers right now, and check your configured models against them.

Free model IDs change often. Run this before you change a model name, and whenever you
get a 404 "model not found" error.

Run:  python scripts/list_models.py                    (Groq)
      python scripts/list_models.py --provider openrouter
"""

import argparse
import os
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.config import PROVIDERS, load_env, resolve_provider_and_models  # noqa: E402


def is_free(model: dict) -> bool:
    pricing = model.get("pricing") or {}
    return str(model.get("id", "")).endswith(":free") or (
        str(pricing.get("prompt")) == "0" and str(pricing.get("completion")) == "0")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--provider", default=None)
    p.add_argument("--all", action="store_true", help="OpenRouter: show paid models too")
    args = p.parse_args()

    load_env()
    provider, configured = resolve_provider_and_models(args.provider)
    preset = PROVIDERS[provider]
    headers = {}
    key = os.environ.get(preset["key_env"], "")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    elif provider == "groq":
        sys.exit(f"Groq needs an API key: put {preset['key_env']}=... in week-3/.env")

    try:
        resp = requests.get(f"{preset['base_url']}/models", headers=headers, timeout=(5, 30))
    except requests.exceptions.RequestException as exc:
        sys.exit(f"Could not reach {provider}: {type(exc).__name__}. Check your internet connection.")
    if resp.status_code != 200:
        sys.exit(f"{provider} returned {resp.status_code}: {resp.text[:200]}")

    models = resp.json().get("data", [])
    models = [m for m in models if m.get("active", True)]
    if provider == "openrouter" and not args.all:
        models = [m for m in models if is_free(m)]

    ids = {m["id"] for m in models}
    print(f"{provider}: {len(models)} {'free ' if provider == 'openrouter' and not args.all else ''}models\n")
    for m in sorted(models, key=lambda m: m["id"]):
        context = m.get("context_window") or m.get("context_length") or "?"
        print(f"  {m['id']:<60} context {context}")

    print("\nYour configured models:")
    for model in configured:
        status = "available" if model in ids else "NOT in this list"
        print(f"  {model:<50} {status}")
    if any(model not in ids for model in configured):
        print("\nChange the model with --model on the command line, or LLM_MODELS=... in week-3/.env.")
        if provider == "openrouter":
            print("(openrouter/free is a router, so it may not appear in this list but can still work.)")


if __name__ == "__main__":
    main()
