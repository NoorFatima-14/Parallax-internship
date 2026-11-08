"""The HTTP endpoints."""

import time

from fastapi import APIRouter, Depends, Request

from api import __version__
from api.errors import APIError
from api.schemas import (AskRequest, AskResponse, DatasetsResponse, ErrorResponse, HealthResponse, ReadyResponse,
                         SearchRequest, SearchResponse, TopicsResponse)
from api.service import RAGService

router = APIRouter()

ERRORS = {
    400: {"model": ErrorResponse, "description": "Request understood but cannot be processed (bad JSON, blank text, unknown filter name)"},
    422: {"model": ErrorResponse, "description": "Request does not match the schema (missing field, wrong type, value out of range)"},
    500: {"model": ErrorResponse, "description": "Internal error"},
    503: {"model": ErrorResponse, "description": "Service not ready, or the language model service is unavailable"},
    504: {"model": ErrorResponse, "description": "The language model service timed out"},
}


# ------------------------------------------------------------------ helpers
def request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "-")


def note(request: Request, **fields) -> None:
    """Add fields to the structured request log line."""
    extra = getattr(request.state, "log_extra", None)
    if isinstance(extra, dict):
        extra.update(fields)


def question_fields(request: Request, text: str) -> dict:
    fields = {"question_chars": len(text)}
    if getattr(request.app.state.settings, "log_questions", False):
        fields["question_preview"] = text[:100]
    return fields


def get_service(request: Request) -> RAGService:
    service = request.app.state.service
    if service is None:
        reason = request.app.state.startup_error or "The service is still starting."
        raise APIError(503, "service_not_ready", f"The RAG service is not ready: {reason}")
    return service


# ------------------------------------------------------------------ system
@router.get("/health", response_model=HealthResponse, tags=["system"], summary="Is the process alive?")
def health(request: Request):
    return {"status": "ok", "version": __version__, "uptime_seconds": round(time.time() - request.app.state.started_at, 1)}


@router.get("/ready", response_model=ReadyResponse, tags=["system"], responses={503: ERRORS[503]},
            summary="Are the database, embedding model and LLM client loaded?")
def ready(request: Request, service: RAGService = Depends(get_service)):
    try:
        info = service.readiness()
    except Exception as exc:
        raise APIError(503, "service_not_ready", f"The vector database is not reachable: {type(exc).__name__}") from exc
    return {"request_id": request_id(request), **info}


@router.get("/topics", response_model=TopicsResponse, tags=["filters"], responses={503: ERRORS[503]},
            summary="Topics that can be used in filters (from Week 4)")
def topics(service: RAGService = Depends(get_service)):
    return {"available": service.topics is not None, "topics": service.topic_list()}


@router.get("/datasets", response_model=DatasetsResponse, tags=["filters"], responses={503: ERRORS[503]},
            summary="Dataset names that can be used in filters (from Week 4)")
def datasets(service: RAGService = Depends(get_service)):
    return {"available": service.datasets_index is not None, "datasets": service.dataset_list()}


# ------------------------------------------------------------------ search and ask
@router.post("/search", response_model=SearchResponse, tags=["rag"], responses={k: ERRORS[k] for k in (400, 422, 500, 503)},
             summary="Semantic search (no LLM)")
def search(body: SearchRequest, request: Request, service: RAGService = Depends(get_service)):
    note(request, endpoint="search", top_k=body.top_k, filters_used=body.filters is not None,
         **question_fields(request, body.query))
    result = service.search(body.query, body.top_k, body.filters)
    note(request, n_results=result["count"], top_score=result["hits"][0]["score"] if result["hits"] else None,
         filter_applied=result["filter_applied"], timings_ms=result["timings_ms"])
    return {"request_id": request_id(request), **result}


@router.post("/ask", response_model=AskResponse, tags=["rag"], responses={k: ERRORS[k] for k in (400, 422, 500, 503, 504)},
             summary="Answer a question from the indexed papers")
def ask(body: AskRequest, request: Request, service: RAGService = Depends(get_service)):
    """Retrieves passages, calls the language model, and checks the answer against the passages.

    Refusals ("I don't know", off-topic) and answers blocked by the hallucination checks are
    normal outcomes: they return HTTP 200 and the `status` field says what happened.
    """
    note(request, endpoint="ask", top_k=body.top_k, filters_used=body.filters is not None,
         **question_fields(request, body.question))
    result = service.ask(body.question, body.top_k, body.filters)

    llm, grounding, error = result.get("llm"), result.get("grounding"), result.get("error")
    note(request, pipeline_status=result["status"], top_score=result["top_score"],
         n_sources=len(result["sources"]), filter_applied=result["filter_applied"],
         timings_ms=result["timings_ms"],
         model=llm["model"] if llm else None, llm_retries=llm["retries"] if llm else None,
         llm_fallbacks=llm["fallbacks"] if llm else None,
         total_tokens=((llm or {}).get("usage") or {}).get("total_tokens") if llm else None,
         grounding_level=grounding["level"] if grounding else None,
         grounded_fraction=grounding["grounded_fraction"] if grounding else None,
         **({"pipeline_error": error} if error else {}))

    failure = service.failure_for(result)
    if failure:
        raise failure

    warnings = list(result.get("warnings") or []) + result["request_warnings"]
    summary = None
    if grounding:
        summary = {k: grounding[k] for k in ("level", "grounded_fraction", "sentences_checked", "citations",
                                             "invalid_citations", "block_reasons")}
        summary["unsupported_number_count"] = len(grounding["unsupported_numbers"])
    return {
        "request_id": request_id(request), "status": result["status"], "answer": result["answer"],
        "sources": result["sources"], "warnings": warnings, "grounding": summary,
        "model": llm["model"] if llm else None, "top_score": result["top_score"],
        "timings_ms": result["timings_ms"], "filter_applied": result["filter_applied"],
    }
