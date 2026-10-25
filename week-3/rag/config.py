"""Settings: provider presets, pipeline thresholds, and a tiny .env loader."""

import os
from dataclasses import dataclass
from pathlib import Path

WEEK3_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = WEEK3_DIR.parent

# Free model IDs change often. These are starting points taken from the providers'
# documentation in October 2026. Run scripts/list_models.py to see what is live today.
PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "models": ["openai/gpt-oss-120b", "openai/gpt-oss-20b"],
        "max_tokens_param": "max_completion_tokens",
        "headers": {},
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        "models": ["openrouter/free"],  # router that picks a currently available free model
        "max_tokens_param": "max_tokens",
        "headers": {"X-Title": "Parallax Labs RAG"},
    },
}


class ConfigError(Exception):
    """Missing or invalid configuration (for example no API key)."""


def load_env(paths=None) -> None:
    """Read KEY=VALUE lines from .env files into os.environ (existing variables win)."""
    for path in paths or [WEEK3_DIR / ".env", REPO_DIR / ".env"]:
        path = Path(path)
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def default_extra_params(provider: str, model: str) -> dict:
    """Extra request fields for models that need them.

    GPT-OSS models on Groq are reasoning models. Reasoning tokens count against the
    completion limit, so a low effort and hidden reasoning keep answers from being cut off.
    """
    if provider == "groq" and "gpt-oss" in model:
        return {"reasoning_effort": "low", "include_reasoning": False}
    return {}


def resolve_provider_and_models(provider=None, models=None):
    """Combine command-line choices, environment variables and presets."""
    provider = (provider or os.environ.get("LLM_PROVIDER") or "groq").lower()
    if provider not in PROVIDERS:
        raise ConfigError(f"Unknown provider '{provider}'. Choose one of: {', '.join(PROVIDERS)}")
    if isinstance(models, str):
        models = [m.strip() for m in models.split(",") if m.strip()]
    if not models and os.environ.get("LLM_MODELS"):
        models = [m.strip() for m in os.environ["LLM_MODELS"].split(",") if m.strip()]
    return provider, list(models or PROVIDERS[provider]["models"])


@dataclass
class RAGConfig:
    top_k: int = 5
    min_score: float = 0.30          # chunks scoring below this are not used (out-of-domain gate)
    max_query_chars: int = 1000
    max_context_chars: int = 6000
    max_input_tokens: int = 3000     # rough estimate (characters / 4)
    max_answer_tokens: int = 700
    temperature: float = 0.0
    max_context_shrinks: int = 2     # times to drop chunks and retry after a token-limit error

    # hallucination checks
    strict: bool = True              # block answers that fail the checks
    support_threshold: float = 0.40  # sentence-to-passage similarity needed to count as supported
    lexical_threshold: float = 0.80  # or this share of its content words found in the passages
    min_grounded_fraction: float = 0.70
    partial_fraction: float = 0.40
    require_citation: bool = False
