import pytest
import requests

from rag.config import ConfigError
from rag.llm_client import (
    AuthError, BadResponseError, LLMClient, LLMError, LLMTimeout, ModelNotFoundError,
    NetworkError, RateLimitError, ServerError, TokenLimitError, parse_duration,
)

MESSAGES = [{"role": "user", "content": "hi"}]


class FakeResponse:
    def __init__(self, status=200, body=None, headers=None, text=None, bad_json=False):
        self.status_code, self.headers, self._body, self._bad = status, headers or {}, body, bad_json
        self.text = text if text is not None else str(body)

    def json(self):
        if self._bad:
            raise ValueError("not json")
        return self._body


class FakeSession:
    """Returns scripted responses in order. An Exception in the script is raised."""

    def __init__(self, script):
        self.script, self.calls = list(script), []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def ok(text="Answer [1].", finish="stop"):
    return FakeResponse(200, {"choices": [{"message": {"content": text}, "finish_reason": finish}],
                              "usage": {"prompt_tokens": 10, "completion_tokens": 5}})


def err(status, message, headers=None):
    return FakeResponse(status, {"error": {"message": message}}, headers)


def make(script, **kwargs):
    sleeps = []
    session = FakeSession(script)
    client = LLMClient("groq", models=kwargs.pop("models", ["model-a", "model-b"]), api_key="sk-secret",
                       session=session, sleep=sleeps.append, **kwargs)
    return client, session, sleeps


# --- success ---
def test_success_returns_text_usage_and_model():
    client, session, _ = make([ok("Hello [1].")])
    r = client.complete(MESSAGES)
    assert r.text == "Hello [1]." and r.model == "model-a" and r.usage["prompt_tokens"] == 10
    assert r.attempts == 1 and r.retries == 0 and not r.truncated
    assert session.calls[0]["url"].endswith("/chat/completions")
    assert session.calls[0]["headers"]["Authorization"] == "Bearer sk-secret"


def test_request_has_a_timeout_and_provider_token_parameter():
    client, session, _ = make([ok()])
    client.complete(MESSAGES, max_tokens=123)
    call = session.calls[0]
    assert call["timeout"] == (5, 60)
    assert call["json"]["max_completion_tokens"] == 123


def test_missing_api_key_raises_config_error(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr("rag.llm_client.load_env", lambda: None)
    with pytest.raises(ConfigError, match="GROQ_API_KEY"):
        LLMClient("groq")


def test_unknown_provider_raises():
    with pytest.raises(ConfigError):
        LLMClient("nope", api_key="x")


# --- rate limits ---
def test_rate_limit_waits_for_retry_after_then_succeeds():
    client, session, sleeps = make([err(429, "slow down", {"Retry-After": "7"}), ok()])
    r = client.complete(MESSAGES)
    assert r.text and r.retries == 1 and r.attempts == 2
    assert sleeps and sleeps[0] >= 7
    assert r.wait_seconds >= 7


def test_rate_limit_uses_try_again_hint_in_message():
    client, _, sleeps = make([err(429, "Rate limit reached. Please try again in 3.5s."), ok()])
    client.complete(MESSAGES)
    assert 3.5 <= sleeps[0] < 5


def test_rate_limit_without_hint_backs_off_exponentially():
    client, _, sleeps = make([err(429, "busy"), err(429, "busy"), ok()], backoff_base=1.0)
    client.complete(MESSAGES)
    assert 1.0 <= sleeps[0] < 1.6 and 2.0 <= sleeps[1] < 2.6


def test_retry_after_longer_than_allowed_fails_fast_without_sleeping():
    client, _, sleeps = make([err(429, "busy", {"Retry-After": "600"})] * 2)
    with pytest.raises(RateLimitError):
        client.complete(MESSAGES)
    assert sleeps == []


def test_daily_quota_is_not_retried():
    client, session, sleeps = make([err(429, "Rate limit reached on requests per day (RPD): Limit 1000")] * 2)
    with pytest.raises(RateLimitError):
        client.complete(MESSAGES)
    assert sleeps == [] and len(session.calls) == 2  # one attempt per model, no retries


def test_rate_limit_retries_are_capped():
    client, session, _ = make([err(429, "busy")] * 20, max_retries=2, models=["only"])
    with pytest.raises(RateLimitError):
        client.complete(MESSAGES)
    assert len(session.calls) == 3  # first try + 2 retries


def test_total_wait_budget_is_respected():
    client, _, sleeps = make([err(429, "busy", {"Retry-After": "20"})] * 10,
                             max_total_wait=45, models=["only"])
    with pytest.raises(RateLimitError):
        client.complete(MESSAGES)
    assert sum(sleeps) <= 45


# --- timeouts, network, server errors ---
def test_timeout_then_success():
    client, _, sleeps = make([requests.exceptions.ReadTimeout(), ok()])
    r = client.complete(MESSAGES)
    assert r.retries == 1 and len(sleeps) == 1


def test_timeout_every_time_raises_llmtimeout():
    client, _, _ = make([requests.exceptions.ReadTimeout()] * 20, max_retries=2, models=["only"])
    with pytest.raises(LLMTimeout):
        client.complete(MESSAGES)


def test_connection_error_is_reported_as_network_error():
    client, _, _ = make([requests.exceptions.ConnectionError()] * 20, max_retries=1, models=["only"])
    with pytest.raises(NetworkError):
        client.complete(MESSAGES)


def test_server_error_then_success():
    client, _, _ = make([err(503, "overloaded"), ok()])
    assert client.complete(MESSAGES).retries == 1


def test_invalid_json_is_retried():
    client, _, _ = make([FakeResponse(200, None, bad_json=True, text="<html>"), ok()])
    assert client.complete(MESSAGES).retries == 1


def test_error_inside_a_200_response_is_handled():
    body = {"error": {"code": 429, "message": "Rate limit reached"}}
    client, _, _ = make([FakeResponse(200, body), ok()])
    assert client.complete(MESSAGES).retries == 1


def test_response_without_choices_is_a_bad_response():
    client, _, _ = make([FakeResponse(200, {"choices": []})] * 10, max_retries=1, models=["only"])
    with pytest.raises(BadResponseError):
        client.complete(MESSAGES)


# --- auth, model not found, fallbacks ---
def test_bad_key_fails_at_once_without_retry():
    client, session, sleeps = make([err(401, "Invalid API key")])
    with pytest.raises(AuthError):
        client.complete(MESSAGES)
    assert len(session.calls) == 1 and sleeps == []


def test_model_not_found_falls_back_to_next_model():
    client, session, _ = make([err(404, "model does not exist"), ok("From B [1].")])
    r = client.complete(MESSAGES)
    assert r.model == "model-b" and r.text == "From B [1]."
    assert r.fallbacks[0]["model"] == "model-a" and r.fallbacks[0]["error"] == "model_not_found"
    assert session.calls[1]["json"]["model"] == "model-b"


def test_all_models_failing_raises_with_the_fallback_history():
    client, _, _ = make([err(404, "gone"), err(404, "gone")])
    with pytest.raises(ModelNotFoundError) as info:
        client.complete(MESSAGES)
    assert "list_models.py" in str(info.value)
    assert [f["model"] for f in info.value.fallbacks] == ["model-a", "model-b"]


def test_rate_limited_model_falls_back_to_the_next_model():
    client, _, _ = make([err(429, "per day (RPD) limit"), ok("B [1].")])
    assert client.complete(MESSAGES).model == "model-b"


# --- token limits ---
@pytest.mark.parametrize("status,message", [
    (400, "This model's maximum context length is 8192 tokens"),
    (413, "Request too large for model on tokens per minute (TPM): Limit 6000, Requested 9000"),
    (400, "context_length_exceeded"),
])
def test_token_limit_errors_raise_token_limit_error(status, message):
    client, session, _ = make([err(status, message)])
    with pytest.raises(TokenLimitError):
        client.complete(MESSAGES)
    assert len(session.calls) == 1


def test_other_bad_requests_are_not_retried():
    client, session, _ = make([err(400, "messages must be a list")] * 2)
    with pytest.raises(LLMError):
        client.complete(MESSAGES)
    assert len(session.calls) == 2  # once per model


# --- optional parameters and truncation ---
def test_unsupported_optional_params_are_dropped_and_retried():
    client, session, _ = make([err(400, "Unrecognized request argument: reasoning_effort"), ok()],
                              models=["openai/gpt-oss-20b"])
    r = client.complete(MESSAGES)
    assert "reasoning_effort" in session.calls[0]["json"]
    assert "reasoning_effort" not in session.calls[1]["json"]
    assert "reasoning_effort" in r.dropped_params


def test_reasoning_models_get_low_effort_by_default():
    client, session, _ = make([ok()], models=["openai/gpt-oss-120b"])
    client.complete(MESSAGES)
    assert session.calls[0]["json"]["reasoning_effort"] == "low"
    assert session.calls[0]["json"]["include_reasoning"] is False


def test_empty_answer_cut_by_max_tokens_is_retried_with_more_tokens():
    client, session, _ = make([ok("", finish="length"), ok("Done [1].")])
    r = client.complete(MESSAGES, max_tokens=100)
    assert session.calls[1]["json"]["max_completion_tokens"] == 200
    assert r.text == "Done [1]."


def test_partial_answer_cut_by_max_tokens_is_flagged_truncated():
    client, _, _ = make([ok("Half an answer", finish="length")])
    assert client.complete(MESSAGES).truncated is True


# --- safety ---
def test_api_key_never_appears_in_error_messages():
    client, _, _ = make([err(400, "bad request for key sk-secret")], models=["only"])
    with pytest.raises(LLMError) as info:
        client.complete(MESSAGES)
    assert "sk-secret" not in str(info.value)


@pytest.mark.parametrize("text,expected", [("1m30.5s", 90.5), ("250ms", 0.25), ("7.66s", 7.66),
                                           ("2h", 7200.0), ("", None), ("soon", None)])
def test_parse_duration(text, expected):
    assert parse_duration(text) == expected
