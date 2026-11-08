"""HTTP error responses: 400, 422, 500 and the 5xx cases for the language model service."""

import pytest

from rag.llm_client import AuthError, LLMError, LLMTimeout, ModelNotFoundError, NetworkError, RateLimitError
from tests.fakes import FakeLLM

QUESTION = "Convolutional neural networks detect objects in images"


def error_of(response):
    body = response.json()
    assert set(body) == {"error"}, "every error uses the same shape"
    error = body["error"]
    assert error["request_id"] == response.headers["X-Request-ID"]
    return error


# ================================================================== 422: valid JSON, wrong content
@pytest.mark.parametrize("path,body,field", [
    ("/ask", {}, "question"),                                                  # missing field
    ("/ask", {"question": ""}, "question"),                                    # too short
    ("/ask", {"question": 123}, "question"),                                   # wrong type
    ("/ask", {"question": None}, "question"),
    ("/ask", {"question": "x" * 4001}, "question"),                            # too long
    ("/ask", {"question": QUESTION, "top_k": 0}, "top_k"),                     # out of range
    ("/ask", {"question": QUESTION, "top_k": 21}, "top_k"),
    ("/ask", {"question": QUESTION, "top_k": "five"}, "top_k"),
    ("/ask", {"question": QUESTION, "colour": "red"}, "colour"),               # unknown field
    ("/ask", {"question": QUESTION, "filters": {"year_from": 1500}}, "filters.year_from"),
    ("/ask", {"question": QUESTION, "filters": {"datasets_mode": "some"}}, "filters.datasets_mode"),
    ("/ask", {"question": QUESTION, "filters": {"categories": ["not a category!"]}}, "filters.categories.0"),
    ("/ask", {"question": QUESTION, "filters": {"colour": 1}}, "filters.colour"),
    ("/ask", {"question": QUESTION, "filters": "cs.CV"}, "filters"),
    ("/search", {}, "query"),
    ("/search", {"query": ""}, "query"),
    ("/search", {"query": QUESTION, "top_k": -1}, "top_k"),
])
def test_schema_violations_are_422_and_name_the_field(client, path, body, field):
    r = client.post(path, json=body)
    assert r.status_code == 422
    error = error_of(r)
    assert error["code"] == "validation_error"
    assert field in [d["field"] for d in error["details"]]
    assert all({"field", "message", "type"} <= set(d) for d in error["details"])


def test_422_does_not_echo_the_submitted_text(client):
    r = client.post("/ask", json={"question": "x" * 4001})
    assert r.status_code == 422 and "xxxxxxxxxx" not in r.text


def test_a_json_array_instead_of_an_object_is_422(client):
    assert client.post("/ask", json=["what", "is", "this"]).status_code == 422


# ================================================================== 400: cannot be processed
def test_malformed_json_is_400(client):
    r = client.post("/ask", content=b'{"question": "oops', headers={"Content-Type": "application/json"})
    assert r.status_code == 400 and error_of(r)["code"] == "invalid_json"


def test_empty_body_is_422_not_400(client):
    r = client.post("/ask", content=b"", headers={"Content-Type": "application/json"})
    assert r.status_code in (400, 422)  # FastAPI reports a missing body as a validation problem


@pytest.mark.parametrize("path,key", [("/ask", "question"), ("/search", "query")])
@pytest.mark.parametrize("blank", ["   ", "\n\t  \n", "\x00\x01\x02"])
def test_blank_text_is_400(client, path, key, blank):
    r = client.post(path, json={key: blank})
    assert r.status_code == 400 and error_of(r)["code"] == "invalid_question"


def test_unknown_topic_is_400_with_a_hint(client):
    r = client.post("/ask", json={"question": QUESTION, "filters": {"topics": ["underwater basketweaving"]}})
    assert r.status_code == 400
    error = error_of(r)
    assert error["code"] == "invalid_filter" and "No topic matches" in error["message"]


def test_topic_id_that_does_not_exist_is_400(client):
    r = client.post("/search", json={"query": QUESTION, "filters": {"topics": [99]}})
    assert r.status_code == 400 and "no topic with id 99" in error_of(r)["message"]


def test_unknown_dataset_is_400(client):
    r = client.post("/ask", json={"question": QUESTION, "filters": {"datasets": ["NotARealDataset"]}})
    assert r.status_code == 400 and error_of(r)["code"] == "invalid_filter"


def test_year_from_after_year_to_is_400(client):
    r = client.post("/ask", json={"question": QUESTION, "filters": {"year_from": 2026, "year_to": 2020}})
    assert r.status_code == 400
    error = error_of(r)
    assert error["code"] == "invalid_filter" and "year_from is after year_to" in error["message"]


def test_topic_and_dataset_filters_are_400_when_week4_data_is_missing(make_client):
    client, _, llm = make_client(filters=False)
    for filters in ({"topics": ["image"]}, {"datasets": ["COCO"]}):
        r = client.post("/ask", json={"question": QUESTION, "filters": filters})
        assert r.status_code == 400 and error_of(r)["code"] == "filter_not_available"
    # filters that need no Week 4 data still work
    assert client.post("/ask", json={"question": QUESTION, "filters": {"categories": ["cs.CV"]}}).status_code == 200
    assert len(llm.calls) == 1


def test_a_bad_request_never_calls_the_llm(make_client):
    client, _, llm = make_client()
    client.post("/ask", json={"question": "   "})
    client.post("/ask", json={"question": QUESTION, "filters": {"topics": ["zzz"]}})
    client.post("/ask", json={"question": 5})
    assert llm.calls == []


# ================================================================== 500: unexpected failures
class ExplodingService:
    """A service whose methods raise an unexpected error that contains a secret."""

    def __getattr__(self, name):
        def boom(*args, **kwargs):
            raise RuntimeError("database password is hunter2")
        return boom


def test_unexpected_exception_is_a_generic_500_without_the_message(make_client):
    client, _, _ = make_client(service=ExplodingService())
    for method, path, body in [("post", "/ask", {"question": QUESTION}), ("post", "/search", {"query": QUESTION}),
                               ("get", "/topics", None), ("get", "/datasets", None)]:
        r = client.request(method, path, json=body)
        assert r.status_code == 500, path
        error = error_of(r)
        assert error["code"] == "internal_error" and "hunter2" not in r.text and "RuntimeError" not in r.text


def test_ready_is_503_not_500_when_the_database_cannot_be_read(make_client):
    """A readiness check that fails means 'not ready' (try again later), so it is 503 by design."""
    client, _, _ = make_client(service=ExplodingService())
    r = client.get("/ready")
    assert r.status_code == 503
    error = error_of(r)
    assert error["code"] == "service_not_ready" and "hunter2" not in r.text


def test_500_is_logged_with_the_traceback_but_the_client_does_not_see_it(make_client, caplog):
    client, _, _ = make_client(service=ExplodingService())
    with caplog.at_level("INFO", logger="api"):
        r = client.post("/ask", json={"question": QUESTION})
    assert r.status_code == 500
    errors = [rec for rec in caplog.records if rec.levelname == "ERROR" and rec.name == "api.request"]
    assert errors and errors[0].exc_info is not None and "hunter2" in str(errors[0].exc_info[1])


def test_pipeline_internal_error_status_is_500(make_client, monkeypatch):
    client, service, _ = make_client()
    monkeypatch.setattr(service.pipeline, "ask", lambda *a, **k: {
        "status": "internal_error", "answer": "Something went wrong while answering.", "sources": [],
        "top_score": None, "grounding": None, "llm": None, "timings_ms": {"total_ms": 1.0},
        "error": {"kind": "internal_error", "message": "KeyError: 'x'"}})
    r = client.post("/ask", json={"question": QUESTION})
    assert r.status_code == 500 and error_of(r)["code"] == "internal_error" and "KeyError" not in r.text


def test_retrieval_failure_is_500(make_client, monkeypatch):
    client, service, _ = make_client()

    def broken(*args, **kwargs):
        raise OSError("disk read error")
    monkeypatch.setattr(service.pipeline.retriever, "search", broken)
    r = client.post("/ask", json={"question": QUESTION})
    assert r.status_code == 500 and error_of(r)["code"] == "internal_error" and "disk" not in r.text


def test_search_endpoint_failure_is_500(make_client, monkeypatch):
    client, service, _ = make_client()
    monkeypatch.setattr(service.pipeline.retriever, "search", lambda *a, **k: (_ for _ in ()).throw(OSError("x")))
    assert client.post("/search", json={"query": QUESTION}).status_code == 500


# ================================================================== language model service problems
def llm_error_cases():
    return [
        (RateLimitError("429 slow down"), 503, "llm_rate_limited", True),
        (LLMTimeout("timed out"), 504, "llm_timeout", True),
        (NetworkError("no route"), 503, "llm_unreachable", True),
        (LLMError("something odd"), 503, "llm_unavailable", True),
        (AuthError("401 bad key sk-secret-123"), 500, "llm_misconfigured", False),
        (ModelNotFoundError("404 model gone"), 500, "llm_misconfigured", False),
    ]


@pytest.mark.parametrize("exc,status,code,retryable", llm_error_cases())
def test_llm_failures_map_to_http_errors(make_client, exc, status, code, retryable):
    client, _, _ = make_client(llm=FakeLLM(default=exc))
    r = client.post("/ask", json={"question": QUESTION})
    assert r.status_code == status
    error = error_of(r)
    assert error["code"] == code and "sk-secret-123" not in r.text
    if retryable:  # the search worked, so the passages are returned to help the caller
        assert error["details"]["fallback_passages"] and error["details"]["kind"] == exc.kind
    else:
        assert "fallback_passages" not in error["details"]


def test_rate_limit_error_has_a_retry_after_header(make_client):
    client, _, _ = make_client(llm=FakeLLM(default=RateLimitError("429")))
    assert client.post("/ask", json={"question": QUESTION}).headers["Retry-After"] == "30"


def test_llm_failure_is_logged_with_its_kind(make_client, caplog):
    client, _, _ = make_client(llm=FakeLLM(default=LLMTimeout("timed out")))
    with caplog.at_level("INFO", logger="api"):
        client.post("/ask", json={"question": QUESTION})
    fields = [r.fields for r in caplog.records if getattr(r, "fields", {}).get("event") == "request"][0]
    assert fields["status_code"] == 504 and fields["pipeline_status"] == "llm_error"
    assert fields["pipeline_error"]["kind"] == "timeout" and fields["error_code"] == "llm_timeout"
