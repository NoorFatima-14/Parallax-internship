"""Chat client for OpenAI-compatible APIs (Groq, OpenRouter) with careful error handling.

Handled cases (each has a unit test):
- 429 rate limits: wait for Retry-After (or the "try again in 7s" hint, or back off), retry
- daily quota exhausted, or a wait that is too long: stop at once with a clear error
- timeouts, connection errors, 5xx errors, invalid JSON, errors hidden in a 200 response: retry
- 401/403 bad key: stop, no retry
- 404 model not found (free model IDs rotate): try the next model in the list
- 400/413 token or context limits: raise TokenLimitError so the caller can send less context
- 400 for an unsupported parameter: drop the optional parameters and retry once
- empty answer because a reasoning model used up max_tokens: retry once with more tokens
"""

import email.utils
import logging
import os
import random
import re
import time
from dataclasses import dataclass, field

import requests

from rag.config import PROVIDERS, ConfigError, default_extra_params, load_env

log = logging.getLogger("rag.llm")


# ---------------------------------------------------------------- errors
class LLMError(Exception):
    kind = "llm_error"
    retryable = False

    def __init__(self, message, status=None, retry_after=None):
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after
        self.fallbacks = []


class AuthError(LLMError):
    kind = "auth"


class ModelNotFoundError(LLMError):
    kind = "model_not_found"


class RateLimitError(LLMError):
    kind = "rate_limit"
    retryable = True


class LLMTimeout(LLMError):
    kind = "timeout"
    retryable = True


class NetworkError(LLMError):
    kind = "network"
    retryable = True


class ServerError(LLMError):
    kind = "server_error"
    retryable = True


class BadResponseError(LLMError):
    kind = "bad_response"
    retryable = True


class TokenLimitError(LLMError):
    kind = "token_limit"


class UnsupportedParamError(LLMError):
    kind = "unsupported_param"


# ---------------------------------------------------------------- helpers
_TOKEN_LIMIT_RE = re.compile(
    r"context length|context_length|maximum context|too many tokens|reduce the length|"
    r"reduce your (?:prompt|message)|input is too long|prompt is too long|request too large",
    re.I,
)
_UNSUPPORTED_RE = re.compile(
    r"unsupported|unrecognized|unknown (?:request )?(?:parameter|field)|additional properties|"
    r"extra inputs are not permitted|invalid parameter",
    re.I,
)
_DAILY_RE = re.compile(r"per day|\(RPD\)|\(TPD\)|daily", re.I)
_DURATION_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(ms|h|m|s)", re.I)


def parse_duration(text: str):
    """'1m30.5s' -> 90.5, '250ms' -> 0.25. Returns None if nothing is found."""
    units = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}
    parts = _DURATION_RE.findall(text or "")
    if not parts:
        return None
    return sum(float(num) * units[unit.lower()] for num, unit in parts)


def _header(headers, name):
    for key, value in (headers or {}).items():
        if key.lower() == name:
            return value
    return None


def _retry_after_seconds(headers, message):
    value = _header(headers, "retry-after")
    if value is not None:
        try:
            return max(0.0, float(value))
        except ValueError:
            try:
                when = email.utils.parsedate_to_datetime(value)
                return max(0.0, when.timestamp() - time.time())
            except (TypeError, ValueError):
                pass
    match = re.search(r"try again in ([0-9hms.\s]+)", message or "", re.I)
    return parse_duration(match.group(1)) if match else None


def _error_message(body, text):
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])
        if isinstance(err, str):
            return err
        if body.get("message"):
            return str(body["message"])
    return (text or "").strip()[:300] or "no details"


@dataclass
class LLMResponse:
    text: str
    model: str
    finish_reason: str = None
    usage: dict = field(default_factory=dict)
    attempts: int = 1          # HTTP requests sent
    retries: int = 0
    wait_seconds: float = 0.0  # time spent waiting between retries
    http_ms: float = 0.0       # duration of the last HTTP request
    truncated: bool = False    # the answer stopped because of max_tokens
    dropped_params: list = field(default_factory=list)
    fallbacks: list = field(default_factory=list)  # models that failed before this one


# ---------------------------------------------------------------- client
class LLMClient:
    def __init__(
        self,
        provider="groq",
        models=None,
        api_key=None,
        base_url=None,
        timeout=(5, 60),
        max_retries=4,
        backoff_base=1.0,
        max_backoff=20.0,
        max_wait_per_retry=30.0,
        max_total_wait=90.0,
        max_tokens_cap=4000,
        session=None,
        sleep=time.sleep,
        rng=None,
    ):
        if provider not in PROVIDERS:
            raise ConfigError(f"Unknown provider '{provider}'")
        preset = PROVIDERS[provider]
        load_env()
        self.provider = provider
        self.models = list(models or preset["models"])
        self.base_url = (base_url or preset["base_url"]).rstrip("/")
        self.max_tokens_param = preset["max_tokens_param"]
        self.api_key = api_key or os.environ.get(preset["key_env"], "")
        if not self.api_key:
            raise ConfigError(
                f"No API key found. Put {preset['key_env']}=... in week-3/.env "
                "(see .env.example) or set it as an environment variable."
            )
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **preset["headers"],
        }
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.max_backoff = max_backoff
        self.max_wait_per_retry = max_wait_per_retry
        self.max_total_wait = max_total_wait
        self.max_tokens_cap = max_tokens_cap
        self.session = session or requests.Session()
        self.sleep = sleep
        self.rng = rng or random.Random()

    # -- public ----------------------------------------------------
    def complete(self, messages, max_tokens=700, temperature=0.0, models=None) -> LLMResponse:
        """Send a chat request. Tries each model in turn. Raises LLMError if all fail."""
        stats = {"attempts": 0, "retries": 0, "wait_seconds": 0.0, "http_ms": 0.0,
                 "dropped_params": []}
        fallbacks, last_error = [], None
        for model in list(models or self.models):
            try:
                text, finish, usage = self._call_model(
                    model, messages, max_tokens, temperature, stats
                )
            except (AuthError, TokenLimitError):
                raise
            except LLMError as err:
                last_error = err
                fallbacks.append({"model": model, "error": err.kind, "message": str(err)})
                log.warning("Model %s failed (%s): %s", model, err.kind, err)
                continue
            return LLMResponse(
                text=text, model=model, finish_reason=finish, usage=usage,
                attempts=stats["attempts"], retries=stats["retries"],
                wait_seconds=round(stats["wait_seconds"], 3), http_ms=round(stats["http_ms"], 1),
                truncated=(finish == "length"), dropped_params=list(stats["dropped_params"]),
                fallbacks=fallbacks,
            )
        last_error.fallbacks = fallbacks
        raise last_error

    # -- one model, with retries ----------------------------------
    def _call_model(self, model, messages, max_tokens, temperature, stats):
        extra = default_extra_params(self.provider, model)
        tokens, retries_here, raised_tokens = max_tokens, 0, False
        while True:
            payload = {"model": model, "messages": messages, "temperature": temperature,
                       self.max_tokens_param: tokens, **extra}
            try:
                data = self._request(payload, stats)
                text, finish, usage = self._extract(data)
                if not text.strip():
                    if finish == "length" and not raised_tokens and tokens * 2 <= self.max_tokens_cap:
                        log.warning("Empty answer cut off by max tokens (%s); retrying with %s",
                                    tokens, tokens * 2)
                        tokens, raised_tokens = tokens * 2, True
                        continue
                    raise BadResponseError("The model returned an empty answer.")
                return text.strip(), finish, usage
            except UnsupportedParamError as err:
                if not extra:
                    raise
                log.warning("Provider rejected optional parameters %s (%s); retrying without them",
                            list(extra), err)
                stats["dropped_params"] = list(extra)
                extra = {}
                continue
            except LLMError as err:
                if not err.retryable or retries_here >= self.max_retries:
                    raise
                delay = min(self.max_backoff, self.backoff_base * 2 ** retries_here)
                delay += self.rng.uniform(0, 0.5)
                if err.retry_after is not None:
                    if err.retry_after > self.max_wait_per_retry:
                        raise  # the provider asks for a long wait: do not block the caller
                    delay = max(delay, err.retry_after + 0.1)
                if stats["wait_seconds"] + delay > self.max_total_wait:
                    raise
                log.warning("%s (%s). Waiting %.1f s, retry %d/%d",
                            err.kind, err, delay, retries_here + 1, self.max_retries)
                self.sleep(delay)
                stats["wait_seconds"] += delay
                stats["retries"] += 1
                retries_here += 1

    # -- one HTTP request -----------------------------------------
    def _request(self, payload, stats):
        url = f"{self.base_url}/chat/completions"
        start = time.perf_counter()
        stats["attempts"] += 1
        try:
            resp = self.session.post(url, headers=self.headers, json=payload, timeout=self.timeout)
        except requests.exceptions.Timeout as exc:
            raise LLMTimeout("The request timed out.") from exc
        except requests.exceptions.ConnectionError as exc:
            raise NetworkError("Could not connect to the provider. Check the internet connection.") from exc
        except requests.exceptions.RequestException as exc:
            raise NetworkError(f"Request failed ({type(exc).__name__}).") from exc
        stats["http_ms"] = 1000 * (time.perf_counter() - start)

        try:
            body = resp.json()
        except ValueError:
            body = None
        if resp.status_code != 200:
            raise self._map_error(resp.status_code, resp.headers, body, resp.text, payload)
        if body is None:
            raise BadResponseError("The provider returned a response that is not valid JSON.")
        if isinstance(body, dict) and body.get("error"):
            # Some providers report errors inside a 200 response.
            code = body["error"].get("code") if isinstance(body["error"], dict) else None
            status = int(code) if str(code).isdigit() else 500
            raise self._map_error(status, resp.headers, body, resp.text, payload)
        return body

    @staticmethod
    def _extract(data):
        choices = data.get("choices") if isinstance(data, dict) else None
        if not choices:
            raise BadResponseError("The response has no choices.")
        choice = choices[0]
        if choice.get("error"):
            raise ServerError(_error_message({"error": choice["error"]}, ""))
        content = (choice.get("message") or {}).get("content") or ""
        if isinstance(content, list):  # some providers return content parts
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        return content, choice.get("finish_reason"), data.get("usage") or {}

    def _map_error(self, status, headers, body, text, payload):
        message = _error_message(body, text).replace(self.api_key, "***")
        retry_after = _retry_after_seconds(headers, message)
        model = payload.get("model")
        if status in (401, 403):
            return AuthError(f"The API key was rejected ({status}). Check the key in your .env file.", status)
        if status == 402:
            return LLMError(f"Payment required (402): {message}. A negative credit balance "
                            "blocks even free models on OpenRouter.", status)
        if status == 404:
            return ModelNotFoundError(
                f"Model '{model}' was not found (404). Free model IDs change often: "
                "run scripts/list_models.py to see the current ones.", status)
        if status in (400, 413, 422):
            if _TOKEN_LIMIT_RE.search(message):
                return TokenLimitError(f"Too many tokens for this model: {message}", status)
            extra_names = [k for k in ("reasoning_effort", "include_reasoning") if k in message]
            if _UNSUPPORTED_RE.search(message) or extra_names:
                return UnsupportedParamError(message, status)
            return LLMError(f"Bad request ({status}): {message}", status)
        if status == 429:
            err = RateLimitError(f"Rate limit reached: {message}", status, retry_after)
            if _DAILY_RE.search(message):
                err.retryable = False  # a daily quota will not clear within a retry
            return err
        if status == 408:
            return LLMTimeout(f"The provider timed out (408): {message}", status)
        if status >= 500:
            return ServerError(f"Provider error ({status}): {message}", status, retry_after)
        return LLMError(f"Unexpected response ({status}): {message}", status)
