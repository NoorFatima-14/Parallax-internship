"""Builds a ready-to-use pipeline from command-line style options."""

from pathlib import Path

from rag.config import RAGConfig, load_env, resolve_provider_and_models
from rag.llm_client import LLMClient
from rag.pipeline import RAGPipeline
from rag.retriever import Retriever


def build_pipeline(provider: str | None = None, models: str | list[str] | None = None,
                   db_path: str | Path | None = None, collection: str | None = None,
                   embed_model: str | None = None, device: str | None = None,
                   config: RAGConfig | None = None, log_path: str | Path | None = None) -> RAGPipeline:
    """Create the LLM client, load the embedding model and the vector database, and return a RAGPipeline.

    Every argument is optional: missing values come from the environment, the .env file or the defaults.
    Raises ConfigError (for example no API key) before the slow model load, and FileNotFoundError or
    ValueError if the vector database is missing or empty.
    """
    load_env()
    config = config or RAGConfig()
    provider, models = resolve_provider_and_models(provider, models)
    llm = LLMClient(provider, models)  # first, so a missing API key fails before the slow model load
    retriever = Retriever.from_disk(db_path, collection, embed_model, device, top_k=config.top_k)
    return RAGPipeline(retriever, llm, config, log_path)
