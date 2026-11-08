"""The service layer: turns API requests into calls to the Week 3 pipeline and the Week 4 filters.

Nothing in here knows about HTTP except the APIError exceptions, so it can be tested without a server.
"""

import json
import logging
from pathlib import Path

from api.errors import GENERIC_500, APIError
from api.paths import add_other_weeks

add_other_weeks()
from nlpmeta.filters import Filters, build_where, resolve_dataset, resolve_topics  # noqa: E402
from rag.guardrails import validate_query  # noqa: E402
from rag.prompts import passage_label  # noqa: E402

log = logging.getLogger("api.service")

# Pipeline statuses that are real answers (HTTP 200). Refusals and blocked answers are valid outcomes.
OK_STATUSES = {"answered", "answered_with_warning", "refused_by_model", "refused_out_of_domain", "blocked_ungrounded"}

# What the LLM failure kinds from Week 3 become: (HTTP status, error code, extra headers, generic message or None)
LLM_FAILURES = {
    "rate_limit": (503, "llm_rate_limited", {"Retry-After": "30"}, None),
    "timeout": (504, "llm_timeout", None, None),
    "network": (503, "llm_unreachable", None, None),
    "auth": (500, "llm_misconfigured", None, "The language model service is not configured correctly."),
    "model_not_found": (500, "llm_misconfigured", None, "The language model service is not configured correctly."),
}
DEFAULT_LLM_FAILURE = (503, "llm_unavailable", None, None)


def _read_json(path, what: str):
    path = Path(path) if path else None
    if not path or not path.is_file():
        log.warning("%s file not found at %s: those filters are not available", what, path)
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        log.warning("Could not read %s file %s: those filters are not available", what, path, exc_info=True)
        return None


class RAGService:
    def __init__(self, pipeline, topics: list | None = None, datasets_index: dict | None = None):
        self.pipeline = pipeline
        self.topics = topics                  # Week 4 topic table (list of dicts) or None
        self.datasets_index = datasets_index  # Week 4 dataset flags (dict) or None

    # ------------------------------------------------------------ construction
    @classmethod
    def from_settings(cls, settings) -> "RAGService":
        """Load the embedding model, open ChromaDB and create the LLM client (slow: seconds)."""
        from rag.app import build_pipeline
        from rag.config import RAGConfig

        cfg = RAGConfig(top_k=settings.top_k, min_score=settings.min_score, strict=settings.strict)
        pipeline = build_pipeline(settings.provider, settings.models, str(settings.db_path), settings.collection,
                                  None, settings.device, cfg, None)
        topics = _read_json(settings.topics_path, "Topics")
        datasets = _read_json(settings.datasets_path, "Datasets")
        return cls(pipeline, topics["topics"] if isinstance(topics, dict) and "topics" in topics else None,
                   datasets if isinstance(datasets, dict) else None)

    @property
    def cfg(self):
        return self.pipeline.cfg

    # ------------------------------------------------------------ listings and status
    def readiness(self) -> dict:
        retriever, llm = self.pipeline.retriever, self.pipeline.llm
        store = retriever.store
        return {
            "ready": True,
            "chunks": store.count(),
            "embedding_model": getattr(store, "model_name", None),
            "llm_provider": getattr(llm, "provider", None),
            "llm_models": list(getattr(llm, "models", None) or []),
            "topics_loaded": self.topics is not None,
            "datasets_loaded": self.datasets_index is not None,
            "settings": {"top_k": self.cfg.top_k, "min_score": self.cfg.min_score, "strict": self.cfg.strict},
        }

    def topic_list(self) -> list[dict]:
        return [{"topic_id": t["topic_id"], "label": t["label"], "size": t.get("size")}
                for t in (self.topics or [])]

    def dataset_list(self) -> list[dict]:
        rows = [{"name": v["name"], "docs": v.get("docs")} for v in (self.datasets_index or {}).values()]
        return sorted(rows, key=lambda r: (-(r["docs"] or 0), r["name"]))

    # ------------------------------------------------------------ filters
    def build_filter(self, params) -> dict | None:
        """FilterParams -> ChromaDB where clause (or None). Bad names become a 400 error."""
        if params is None:
            return None
        try:
            topics, flags = [], []
            if params.topics:
                if self.topics is None:
                    raise APIError(400, "filter_not_available",
                                   "Topic filters are not available: the Week 4 topic table was not loaded. "
                                   "Run week-4/scripts/run_topic_model.py and enrich_vector_db.py.")
                for t in params.topics:
                    topics += resolve_topics(t, self.topics)
            if params.datasets:
                if self.datasets_index is None:
                    raise APIError(400, "filter_not_available",
                                   "Dataset filters are not available: the Week 4 dataset index was not loaded. "
                                   "Run week-4/scripts/enrich_vector_db.py.")
                flags = [resolve_dataset(d, self.datasets_index) for d in params.datasets]
            return build_where(Filters(
                topics=sorted(set(topics)), categories=list(params.categories), datasets=flags,
                datasets_mode=params.datasets_mode, year_from=params.year_from, year_to=params.year_to,
                core_only=params.core_only, exclude_short=params.exclude_short))
        except ValueError as exc:  # unknown topic or dataset, year_from after year_to, ...
            raise APIError(400, "invalid_filter", str(exc)) from exc

    # ------------------------------------------------------------ operations
    def _clean_text(self, text: str, what: str):
        clean, issue = validate_query(text, self.cfg.max_query_chars)
        if clean is None:
            raise APIError(400, "invalid_question", f"The {what} is empty after removing blank and unprintable characters.")
        return clean, (["question_truncated"] if issue == "truncated" else [])

    def search(self, query: str, top_k: int | None = None, params=None) -> dict:
        where = self.build_filter(params)
        clean, warnings = self._clean_text(query, "query")
        hits, t = self.pipeline.retriever.search(clean, top_k or self.cfg.top_k, where)
        return {
            "query": clean, "count": len(hits), "hits": [self._hit(h) for h in hits], "warnings": warnings,
            "timings_ms": {"embed_ms": round(t["embed_ms"], 1), "search_ms": round(t["query_ms"], 1),
                           "total_ms": round(t["total_ms"], 1)},
            "filter_applied": where,
        }

    def ask(self, question: str, top_k: int | None = None, params=None) -> dict:
        """Returns the Week 3 pipeline result, plus filter_applied and warnings for the response."""
        where = self.build_filter(params)
        clean, warnings = self._clean_text(question, "question")
        result = self.pipeline.ask(clean, where=where, top_k=top_k)
        result["filter_applied"] = where
        result["request_warnings"] = warnings
        return result

    # ------------------------------------------------------------ result handling
    @staticmethod
    def _hit(hit: dict) -> dict:
        meta = hit.get("metadata") or {}
        datasets = meta.get("datasets")
        return {
            "id": hit["id"], "doc_id": passage_label(hit)["doc_id"], "title": meta.get("title", ""),
            "category": meta.get("category", ""), "score": hit["score"], "text": hit["text"],
            "topic_id": meta.get("topic_id"), "topic_label": meta.get("topic_label"),
            "published_year": meta.get("published_year"),
            "datasets": str(datasets) if datasets else None,
        }

    @staticmethod
    def failure_for(result: dict) -> APIError | None:
        """None if the pipeline result is a real answer or refusal, otherwise the HTTP error to return."""
        status = result["status"]
        if status in OK_STATUSES:
            return None
        if status == "invalid_query":
            return APIError(400, "invalid_question", "The question is empty or cannot be used.")
        if status == "llm_error":
            kind = (result.get("error") or {}).get("kind", "llm_error")
            http, code, headers, generic = LLM_FAILURES.get(kind, DEFAULT_LLM_FAILURE)
            details = {"kind": kind}
            if http != 500:  # the search worked, so the passages are still useful to the caller
                details["fallback_passages"] = result.get("fallback_passages", [])
            return APIError(http, code, generic or result["answer"], details, headers)
        return APIError(500, "internal_error", GENERIC_500)  # retrieval_error, internal_error, anything unknown
