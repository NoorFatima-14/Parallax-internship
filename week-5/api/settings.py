"""Service settings. Defaults point at the folders of the other weeks; environment variables override them."""

import os
from dataclasses import dataclass
from pathlib import Path

from api.paths import REPO, WEEK2, WEEK3, WEEK4, WEEK5


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    return default if value is None else value.strip().lower() in ("1", "true", "yes", "on")


def _env_number(name: str, default, cast):
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    try:
        return cast(value)
    except ValueError:
        raise ValueError(f"Environment variable {name} must be a number, got '{value}'") from None


def _load_env_file() -> None:
    """Read KEY=VALUE lines from .env files (existing variables win). Week 5 first, then Week 3."""
    for path in (WEEK5 / ".env", WEEK3 / ".env", REPO / ".env"):
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass
class Settings:
    # data
    db_path: Path = WEEK2 / "chroma_db"
    collection: str | None = None
    topics_path: Path = WEEK4 / "data" / "processed" / "topics.json"
    datasets_path: Path = WEEK4 / "data" / "processed" / "datasets_index.json"
    # LLM (None = use LLM_PROVIDER / LLM_MODELS or the Week 3 defaults)
    provider: str | None = None
    models: str | None = None
    device: str | None = None
    # RAG behaviour
    top_k: int = 5
    min_score: float = 0.30
    strict: bool = True
    # logging
    log_file: Path | None = WEEK5 / "logs" / "api_requests.jsonl"
    log_level: str = "INFO"
    log_questions: bool = True  # put a short preview of each question in the request log

    @classmethod
    def from_env(cls) -> "Settings":
        _load_env_file()
        s = cls()
        env = os.environ.get
        if env("RAG_DB_PATH"):
            s.db_path = Path(env("RAG_DB_PATH"))
        s.collection = env("RAG_COLLECTION") or s.collection
        if env("RAG_TOPICS_PATH"):
            s.topics_path = Path(env("RAG_TOPICS_PATH"))
        if env("RAG_DATASETS_PATH"):
            s.datasets_path = Path(env("RAG_DATASETS_PATH"))
        s.provider = env("LLM_PROVIDER") or s.provider
        s.models = env("LLM_MODELS") or s.models
        s.device = env("RAG_DEVICE") or s.device
        s.top_k = _env_number("RAG_TOP_K", s.top_k, int)
        s.min_score = _env_number("RAG_MIN_SCORE", s.min_score, float)
        s.strict = _env_bool("RAG_STRICT", s.strict)
        if "API_LOG_FILE" in os.environ:
            s.log_file = Path(os.environ["API_LOG_FILE"]) if os.environ["API_LOG_FILE"].strip() else None
        s.log_level = (env("API_LOG_LEVEL") or s.log_level).upper()
        s.log_questions = _env_bool("API_LOG_QUESTIONS", s.log_questions)
        return s
