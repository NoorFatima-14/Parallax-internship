"""Builds a ready-to-use pipeline from command-line style options."""

from rag.config import RAGConfig, load_env, resolve_provider_and_models
from rag.llm_client import LLMClient
from rag.pipeline import RAGPipeline
from rag.retriever import Retriever


def build_pipeline(provider=None, models=None, db_path=None, collection=None, embed_model=None,
                   device=None, config: RAGConfig = None, log_path=None) -> RAGPipeline:
    load_env()
    config = config or RAGConfig()
    provider, models = resolve_provider_and_models(provider, models)
    llm = LLMClient(provider, models)  # first, so a missing API key fails before the slow model load
    retriever = Retriever.from_disk(db_path, collection, embed_model, device, top_k=config.top_k)
    return RAGPipeline(retriever, llm, config, log_path)
