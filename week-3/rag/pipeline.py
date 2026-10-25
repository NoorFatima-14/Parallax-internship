"""The RAG pipeline: retrieve, build the prompt, call the LLM, check the answer, log timings."""

import logging
import time

from rag.config import RAGConfig
from rag.guardrails import check_grounding, is_refusal, normalize_citations, validate_query
from rag.llm_client import LLMError, TokenLimitError
from rag.prompts import REFUSAL_SENTENCE, build_messages, passage_label
from rag.retriever import week2

log = logging.getLogger("rag.pipeline")

OUT_OF_DOMAIN_MESSAGE = (
    "I can only answer questions about the indexed arXiv papers (computer science and "
    "machine learning), and I could not find relevant passages for this question."
)
BLOCKED_MESSAGE = (
    "I found related passages, but I could not produce an answer that I could verify against "
    "them, so I am not showing one. Try rephrasing, or ask something more specific."
)
ERROR_MESSAGES = {
    "rate_limit": "The language model service is rate limited right now. Please wait a minute and try again.",
    "timeout": "The language model service did not answer in time. Please try again.",
    "network": "I could not reach the language model service. Check the internet connection.",
    "auth": "The API key was rejected. Check the key in week-3/.env.",
    "model_not_found": "The configured model was not found (free model IDs change often). "
                       "Run scripts/list_models.py to see the current ones.",
    "token_limit": "The question and passages were too long for the model.",
}


class RAGPipeline:
    """ask() never raises: every failure becomes a result with a status and a message."""

    def __init__(self, retriever, llm, config: RAGConfig = None, log_path=None):
        self.retriever, self.llm = retriever, llm
        self.cfg = config or RAGConfig()
        self.log_path = log_path

    # ------------------------------------------------------------
    def ask(self, question, where: dict = None, top_k: int = None) -> dict:
        start = time.perf_counter()
        timings = {}
        result = {"question": question if isinstance(question, str) else repr(question)[:200],
                  "status": None, "answer": "", "sources": [], "top_score": None,
                  "grounding": None, "llm": None, "error": None, "timings_ms": timings,
                  "query_issue": None}
        try:
            self._run(question, where, top_k, result, timings)
        except Exception as exc:  # last safety net: the caller always gets a result
            log.exception("Unexpected error")
            result.update(status="internal_error", answer="Something went wrong while answering.",
                          error={"kind": "internal_error", "message": f"{type(exc).__name__}: {exc}"})
        timings["total_ms"] = round(1000 * (time.perf_counter() - start), 1)
        self._log(result)
        return result

    # ------------------------------------------------------------
    def _run(self, question, where, top_k, result, timings):
        cfg = self.cfg
        query, issue = validate_query(question, cfg.max_query_chars)
        result["query_issue"] = issue
        if query is None:
            result.update(status="invalid_query", answer="Please type a question.")
            return

        # 1. retrieval
        try:
            hits, rt = self.retriever.search(query, top_k or cfg.top_k, where)
        except Exception as exc:
            result.update(status="retrieval_error", answer="The document search failed.",
                          error={"kind": "retrieval", "message": f"{type(exc).__name__}: {exc}"})
            return
        timings["embed_ms"] = round(rt["embed_ms"], 1)
        timings["search_ms"] = round(rt["query_ms"], 1)
        timings["retrieval_ms"] = round(rt["total_ms"], 1)
        result["top_score"] = hits[0]["score"] if hits else None

        # 2. out-of-domain gate: no passage is close enough to the question
        kept = [h for h in hits if h["score"] >= cfg.min_score]
        if not kept:
            result.update(status="refused_out_of_domain", answer=OUT_OF_DOMAIN_MESSAGE)
            return
        result["sources"] = [{"n": i, **passage_label(h)} for i, h in enumerate(kept, 1)]

        # 3. prompt
        t = time.perf_counter()
        try:
            messages, used = build_messages(query, kept, cfg.max_context_chars, cfg.max_input_tokens)
        except TokenLimitError as exc:
            self._llm_failure(result, exc, kept)
            return
        timings["prompt_ms"] = round(1000 * (time.perf_counter() - t), 1)

        # 4. LLM call. On a token-limit error, send fewer passages and try again.
        t = time.perf_counter()
        response = None
        for _ in range(cfg.max_context_shrinks + 1):
            try:
                response = self.llm.complete(messages, cfg.max_answer_tokens, cfg.temperature)
                break
            except TokenLimitError as exc:
                if len(used) <= 1:
                    self._llm_failure(result, exc, kept)
                    break
                used = used[: max(1, len(used) // 2)]
                log.warning("Token limit hit; retrying with %d passage(s)", len(used))
                messages, used = build_messages(query, used, cfg.max_context_chars, cfg.max_input_tokens)
            except LLMError as exc:
                self._llm_failure(result, exc, kept)
                break
        timings["llm_ms"] = round(1000 * (time.perf_counter() - t), 1)
        if response is None:
            if result["status"] is None:  # shrink attempts used up
                self._llm_failure(result, TokenLimitError("Still too long after dropping passages."), kept)
            return
        result["sources"] = [{"n": i, **passage_label(h)} for i, h in enumerate(used, 1)]
        result["llm"] = {
            "model": response.model, "finish_reason": response.finish_reason,
            "usage": response.usage, "attempts": response.attempts, "retries": response.retries,
            "wait_seconds": response.wait_seconds, "http_ms": response.http_ms,
            "truncated": response.truncated, "dropped_params": response.dropped_params,
            "fallbacks": response.fallbacks, "passages_used": len(used),
        }

        answer_text = normalize_citations(response.text)

        # 5. the model says it does not know
        if is_refusal(answer_text):
            result.update(status="refused_by_model", answer=REFUSAL_SENTENCE)
            return

        # 6. hallucination checks
        t = time.perf_counter()
        grounding = check_grounding(answer_text, used, self.retriever.encode, cfg)
        timings["verify_ms"] = round(1000 * (time.perf_counter() - t), 1)
        result["grounding"] = grounding
        if grounding["blocked"]:
            result.update(status="blocked_ungrounded", answer=BLOCKED_MESSAGE,
                          unverified_answer=answer_text)
            return
        warnings = list(grounding["warnings"]) + (["answer_truncated"] if response.truncated else [])
        result.update(status="answered_with_warning" if warnings else "answered",
                      answer=answer_text, warnings=warnings)

    # ------------------------------------------------------------
    def _llm_failure(self, result, exc, passages):
        kind = getattr(exc, "kind", "llm_error")
        result.update(
            status="llm_error",
            answer=ERROR_MESSAGES.get(kind, "The language model service is unavailable right now."),
            error={"kind": kind, "message": str(exc), "fallbacks": getattr(exc, "fallbacks", [])},
            # the search still worked, so show the passages instead of nothing
            fallback_passages=[{"title": passage_label(p)["title"], "text": p["text"][:300]}
                               for p in passages[:3]],
        )

    def _log(self, result):
        if not self.log_path:
            return
        try:
            week2().append_json_log(self.log_path, {
                "question": result["question"][:200], "status": result["status"],
                "top_score": result["top_score"], "timings_ms": result["timings_ms"],
                "llm": result["llm"], "error": result["error"],
                "grounding": None if not result["grounding"] else {
                    k: result["grounding"][k] for k in
                    ("level", "grounded_fraction", "block_reasons", "warnings")},
            })
        except Exception:
            log.warning("Could not write the log file", exc_info=True)
