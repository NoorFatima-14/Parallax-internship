import json

import pytest

from rag.config import RAGConfig
from rag.llm_client import AuthError, LLMResponse, LLMTimeout, ModelNotFoundError, RateLimitError, TokenLimitError
from rag.pipeline import BLOCKED_MESSAGE, OUT_OF_DOMAIN_MESSAGE, RAGPipeline
from rag.retriever import Retriever, week2
from tests.fakes import HashingModel

DOCS = [
    ("2601.00001v1", "cs.CV", "Convolutional neural networks detect objects in images. Pedestrian detection "
     "under heavy occlusion reached 92.5 percent accuracy on 1200 test images."),
    ("2601.00002v1", "cs.CL", "Transformers use attention for machine translation and language modelling "
     "of long text sequences."),
    ("2601.00003v1", "stat.ML", "Bayesian inference with variational methods estimates posterior "
     "distributions for gaussian processes."),
]
GOOD_ANSWER = "Convolutional neural networks detect objects in images [1]."


class FakeLLM:
    def __init__(self, script):
        self.script, self.calls = list(script), []

    def complete(self, messages, max_tokens=700, temperature=0.0, models=None):
        self.calls.append(messages)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, LLMResponse):
            return item
        return LLMResponse(text=item, model="fake-model", usage={"total_tokens": 42})


@pytest.fixture
def retriever(tmp_path):
    w2, model = week2(), HashingModel(dim=512)
    store = w2.VectorStore(tmp_path / "db", "rag_test", "fake")
    chunks = [{"id": f"http://arxiv.org/abs/{d}::0", "doc_id": f"http://arxiv.org/abs/{d}", "text": t,
               "title": f"Paper {d}", "category": c} for d, c, t in DOCS]
    store.add_chunks(chunks, model.encode([c["text"] for c in chunks], normalize_embeddings=True))
    return Retriever(store, model)


def ask(retriever, script, question="Convolutional neural networks detect objects in images", cfg=None, **kw):
    llm = FakeLLM(script)
    return RAGPipeline(retriever, llm, cfg, **kw).ask(question), llm


# --- normal path ---
def test_grounded_answer_is_returned_with_sources_and_timings(retriever):
    result, llm = ask(retriever, [GOOD_ANSWER])
    assert result["status"] == "answered" and result["answer"] == GOOD_ANSWER
    assert result["sources"][0]["doc_id"] == "2601.00001v1"
    assert result["grounding"]["level"] == "grounded"
    assert result["llm"]["model"] == "fake-model"
    t = result["timings_ms"]
    for key in ("embed_ms", "search_ms", "retrieval_ms", "prompt_ms", "llm_ms", "verify_ms", "total_ms"):
        assert key in t and t[key] >= 0
    assert t["total_ms"] >= t["retrieval_ms"] + t["llm_ms"] - 1
    assert len(llm.calls) == 1


def test_prompt_sent_to_the_llm_contains_the_question_and_numbered_passages(retriever):
    _, llm = ask(retriever, [GOOD_ANSWER])
    user = llm.calls[0][1]["content"]
    assert "[1] Title: Paper 2601.00001v1" in user and "Convolutional neural networks detect" in user


def test_metadata_filter_is_passed_to_the_search(retriever):
    llm = FakeLLM(["Transformers use attention for machine translation [1]."])
    result = RAGPipeline(retriever, llm).ask("transformers use attention for machine translation",
                                             where={"category": "cs.CL"})
    assert result["status"] == "answered"
    assert {s["category"] for s in result["sources"]} == {"cs.CL"}


def test_fullwidth_citations_are_normalized_in_the_returned_answer(retriever):
    result, _ = ask(retriever, ["Convolutional neural networks detect objects in images \u30101\u3011."])
    assert result["status"] == "answered" and result["answer"].endswith("images [1].")
    assert result["grounding"]["citations"] == [1]


# --- out-of-domain ---
def test_off_topic_question_is_refused_without_calling_the_llm(retriever):
    result, llm = ask(retriever, [], question="What is the best recipe for chocolate cake?")
    assert result["status"] == "refused_out_of_domain" and result["answer"] == OUT_OF_DOMAIN_MESSAGE
    assert llm.calls == [] and result["llm"] is None and "llm_ms" not in result["timings_ms"]


def test_gate_threshold_is_configurable(retriever):
    result, llm = ask(retriever, [GOOD_ANSWER], cfg=RAGConfig(min_score=0.99))
    assert result["status"] == "refused_out_of_domain" and llm.calls == []


def test_model_saying_it_does_not_know_is_reported_as_a_refusal(retriever):
    result, _ = ask(retriever, ["I don't know based on the provided documents."])
    assert result["status"] == "refused_by_model"
    assert result["answer"] == "I don't know based on the provided documents."
    assert result["grounding"] is None


# --- hallucination checks ---
def test_hallucinated_answer_is_blocked_and_kept_for_inspection(retriever):
    bad = "The moon orbits the earth every month [1]. Volcanoes erupt with molten basalt [1]."
    result, _ = ask(retriever, [bad])
    assert result["status"] == "blocked_ungrounded" and result["answer"] == BLOCKED_MESSAGE
    assert result["unverified_answer"] == bad


def test_invented_number_is_blocked(retriever):
    bad = "Convolutional neural networks detect objects in images with 99.9 percent accuracy [1]."
    result, _ = ask(retriever, [bad])
    assert result["status"] == "blocked_ungrounded"
    assert result["grounding"]["unsupported_numbers"] == ["99.9"]


def test_non_strict_mode_returns_the_answer(retriever):
    bad = "The moon orbits the earth every month [1]. Volcanoes erupt with molten basalt [1]."
    result, _ = ask(retriever, [bad], cfg=RAGConfig(strict=False))
    assert result["status"] == "answered" and result["answer"] == bad


def test_truncated_answer_gets_a_warning(retriever):
    cut = LLMResponse(text=GOOD_ANSWER, model="fake", finish_reason="length", truncated=True)
    result, _ = ask(retriever, [cut])
    assert result["status"] == "answered_with_warning" and "answer_truncated" in result["warnings"]


# --- bad input ---
@pytest.mark.parametrize("question", ["", "   ", None, 42])
def test_invalid_questions_never_reach_the_llm(retriever, question):
    result, llm = ask(retriever, [], question=question)
    assert result["status"] == "invalid_query" and llm.calls == []


def test_very_long_question_is_truncated_and_still_answered(retriever):
    result, _ = ask(retriever, [GOOD_ANSWER], question="convolutional neural networks detect objects in images " * 40)
    assert result["query_issue"] == "truncated" and result["status"] in ("answered", "answered_with_warning")


# --- API failures ---
@pytest.mark.parametrize("error,kind", [
    (RateLimitError("busy", 429), "rate_limit"), (LLMTimeout("slow"), "timeout"),
    (AuthError("bad key", 401), "auth"), (ModelNotFoundError("gone", 404), "model_not_found"),
])
def test_api_errors_become_a_friendly_result_with_the_passages(retriever, error, kind):
    result, _ = ask(retriever, [error])
    assert result["status"] == "llm_error" and result["error"]["kind"] == kind
    assert "Traceback" not in result["answer"] and result["fallback_passages"]
    assert result["sources"] and result["timings_ms"]["retrieval_ms"] > 0


def test_token_limit_error_sends_fewer_passages_and_succeeds(retriever):
    result, llm = ask(retriever, [TokenLimitError("too long", 400), GOOD_ANSWER],
                      question="convolutional neural networks detect objects with attention and bayesian inference",
                      cfg=RAGConfig(min_score=0.05))
    first, second = (c[1]["content"].count("] Title:") for c in llm.calls)
    assert result["status"] == "answered" and second < first
    assert result["llm"]["passages_used"] == second


def test_token_limit_that_never_clears_ends_with_a_clear_error(retriever):
    result, _ = ask(retriever, [TokenLimitError("too long", 400)] * 5,
                    question="convolutional neural networks detect objects with attention and bayesian inference",
                    cfg=RAGConfig(min_score=0.05))
    assert result["status"] == "llm_error" and result["error"]["kind"] == "token_limit"


def test_unexpected_exceptions_do_not_escape(retriever):
    result, _ = ask(retriever, [ValueError("boom")])
    assert result["status"] == "internal_error" and "boom" in result["error"]["message"]


def test_retrieval_failure_is_reported(retriever):
    class Broken:
        def search(self, *a, **k):
            raise RuntimeError("db down")
    result = RAGPipeline(Broken(), FakeLLM([])).ask("hello there")
    assert result["status"] == "retrieval_error" and "db down" in result["error"]["message"]


# --- logging ---
def test_each_request_is_logged_with_latency(retriever, tmp_path):
    log = tmp_path / "logs" / "rag_log.json"
    ask(retriever, [GOOD_ANSWER], log_path=log)
    ask(retriever, [], question="chocolate cake recipe", log_path=log)
    records = json.loads(log.read_text())
    assert [r["status"] for r in records] == ["answered", "refused_out_of_domain"]
    assert "total_ms" in records[0]["timings_ms"] and "timestamp" in records[0]
    assert records[0]["llm"]["model"] == "fake-model"


def test_a_log_write_failure_does_not_break_the_answer(retriever, tmp_path):
    result, _ = ask(retriever, [GOOD_ANSWER], log_path=tmp_path)  # a directory, not a file
    assert result["status"] == "answered"
