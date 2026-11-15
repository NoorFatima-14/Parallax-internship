"""Check that the project is set up correctly before you start the API.

Run from inside week-5/:

    python scripts/check_setup.py

It prints one line per check (OK, WARN or FAIL) and exits with 1 if any check FAILS.
It does not call the LLM and does not load the embedding model.
"""

import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from api.paths import WEEK3, add_other_weeks  # noqa: E402
from api.settings import Settings  # noqa: E402

REQUIRED_PACKAGES = ["numpy", "pandas", "sentence_transformers", "chromadb", "fastapi", "uvicorn", "httpx",
                     "pydantic", "requests", "sklearn"]
OPTIONAL_PACKAGES = ["spacy", "bertopic", "umap", "hdbscan", "pytest"]


def check_python() -> tuple[str, str]:
    """Python 3.10 or newer is needed (3.11 was used for the reported results)."""
    v = sys.version_info
    text = f"Python {v.major}.{v.minor}.{v.micro}"
    return ("OK" if v >= (3, 10) else "FAIL", text)


def check_packages() -> list[tuple[str, str]]:
    """Look for each package without importing it (importing torch is slow)."""
    rows = []
    for name in REQUIRED_PACKAGES:
        found = importlib.util.find_spec(name) is not None
        rows.append(("OK" if found else "FAIL", f"package {name}" + ("" if found else "  (pip install -r requirements.txt)")))
    for name in OPTIONAL_PACKAGES:
        found = importlib.util.find_spec(name) is not None
        rows.append(("OK" if found else "WARN", f"package {name}" + ("" if found else "  (needed for Week 4 scripts or tests)")))
    return rows


def check_spacy_model() -> tuple[str, str]:
    """The English spaCy model is only needed for the Week 4 NER baseline."""
    found = importlib.util.find_spec("en_core_web_sm") is not None
    return ("OK" if found else "WARN", "spaCy model en_core_web_sm" + ("" if found else "  (python -m spacy download en_core_web_sm)"))


def check_key() -> tuple[str, str]:
    """An LLM key must be in the environment or in week-3/.env (the value is never printed)."""
    keys = ("GROQ_API_KEY", "OPENROUTER_API_KEY")
    if any(os.environ.get(k) for k in keys):
        return "OK", "LLM key found in the environment"
    env = WEEK3 / ".env"
    if env.is_file():
        for line in env.read_text(encoding="utf-8", errors="ignore").splitlines():
            name, _, value = line.partition("=")
            if name.strip() in keys and value.strip():
                return "OK", "LLM key found in week-3/.env"
    return "FAIL", "no LLM key (copy week-3/.env.example to week-3/.env and add GROQ_API_KEY)"


def check_data(settings: Settings) -> list[tuple[str, str]]:
    """The vector database and the Week 4 metadata files must exist (they are not stored in git)."""
    rows = []
    db = settings.db_path
    if not db.is_dir():
        rows.append(("FAIL", f"vector database missing: {db}  (run week-2/scripts/ingest.py --reset)"))
    else:
        try:
            import chromadb

            client = chromadb.PersistentClient(path=str(db))
            counts = {c.name: client.get_collection(c.name).count() for c in client.list_collections()}
            total = max(counts.values(), default=0)
            rows.append(("OK" if total else "FAIL", f"vector database {db.name}: {counts or 'no collections'}"))
        except Exception as exc:  # any ChromaDB problem is reported, not raised
            rows.append(("FAIL", f"cannot open the vector database: {type(exc).__name__}: {exc}"))
    for label, path in (("topics file", settings.topics_path), ("datasets index", settings.datasets_path)):
        rows.append(("OK" if path.is_file() else "WARN",
                     f"{label} {'found' if path.is_file() else 'missing (run week-4 scripts, then enrich_vector_db.py)'}"))
    return rows


def run_checks() -> list[tuple[str, str]]:
    """Run every check and return (status, message) rows."""
    add_other_weeks()
    rows = [check_python(), *check_packages(), check_spacy_model(), check_key(), *check_data(Settings.from_env())]
    return rows


def main() -> None:
    """Print the checks and exit with 1 if any failed."""
    rows = run_checks()
    for status, text in rows:
        print(f"[{status:<4}] {text}")
    failed = sum(s == "FAIL" for s, _ in rows)
    print("\nSetup looks good." if not failed else f"\n{failed} check(s) failed. Fix them and run this again.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
