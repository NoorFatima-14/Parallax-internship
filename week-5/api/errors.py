"""One error format for every failure, and the handlers that produce it.

Every error response has this shape (and carries the same request id as the X-Request-ID header):

    {"error": {"code": "validation_error", "message": "...", "request_id": "...", "details": ...}}

400  the request was understood but cannot be processed (bad JSON, blank question, unknown filter name)
422  the JSON is valid but does not match the schema (missing field, wrong type, value out of range)
500  something broke inside the service (the client gets a generic message, the log has the traceback)
503/504  the language model service is unavailable or too slow (a retry may work)
"""

import logging
from typing import Any, Sequence

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

log = logging.getLogger("api.errors")

GENERIC_500 = "Something went wrong on the server. Please try again, and quote the request id if it keeps happening."


class APIError(Exception):
    """An error with a status code, a short machine-readable code and a message for the client."""

    def __init__(self, status_code: int, code: str, message: str, details: Any = None, headers: dict | None = None):
        """`details` is extra data for the error body. `headers` are extra HTTP response headers."""
        super().__init__(message)
        self.status_code, self.code, self.message = status_code, code, message
        self.details, self.headers = details, headers


def error_response(request_id: str, status_code: int, code: str, message: str, details: Any = None,
                   headers: dict | None = None) -> JSONResponse:
    """Build the standard error response: {"error": {code, message, request_id, details?}}."""
    body = {"code": code, "message": message, "request_id": request_id}
    if details is not None:
        body["details"] = details
    return JSONResponse({"error": body}, status_code=status_code, headers=headers)


def _request_id(request: Request) -> str:
    """The request id set by the middleware, or "-" if there is none."""
    return getattr(request.state, "request_id", "-")


def _note_error(request: Request, code: str, message: str) -> None:
    """Put the error code in the request log line."""
    extra = getattr(request.state, "log_extra", None)
    if isinstance(extra, dict):
        extra["error_code"], extra["error_message"] = code, message[:300]


def _field_details(errors: Sequence[Any]) -> list[dict]:
    """Turn pydantic validation errors into a short list of {field, message, type}."""
    details = []
    for e in errors:
        loc = [str(part) for part in e.get("loc", ())]
        if loc and loc[0] in ("body", "query", "path", "header"):
            loc = loc[1:]
        details.append({"field": ".".join(loc) or "body", "message": str(e.get("msg", "")),
                        "type": str(e.get("type", ""))})
    return details


def register_error_handlers(app: FastAPI) -> None:
    """Add the handlers that give every error the same JSON shape."""
    @app.exception_handler(APIError)
    async def handle_api_error(request: Request, exc: APIError) -> JSONResponse:
        """Turn an APIError into the standard error response, with its own status code."""
        _note_error(request, exc.code, exc.message)
        return error_response(_request_id(request), exc.status_code, exc.code, exc.message,
                              exc.details, exc.headers)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        """Answer 400 for a body that is not valid JSON, and 422 for any other schema problem."""
        errors = exc.errors()
        if any(e.get("type") == "json_invalid" for e in errors):
            message = "The request body is not valid JSON."
            _note_error(request, "invalid_json", message)
            return error_response(_request_id(request), 400, "invalid_json", message)
        message = "The request does not match the expected format. See details."
        _note_error(request, "validation_error", message)
        return error_response(_request_id(request), 422, "validation_error", message, _field_details(errors))

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        """Turn HTTP errors such as 404 and 405 into the standard error response."""
        code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
        message = exc.detail if isinstance(exc.detail, str) else "The request could not be handled."
        _note_error(request, code, message)
        return error_response(_request_id(request), exc.status_code, code, message,
                              headers=getattr(exc, "headers", None))

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        """Log any other error and answer a generic 500."""
        # The logging middleware normally catches errors first. This is the safety net.
        log.error("Unhandled error", exc_info=(type(exc), exc, exc.__traceback__))
        _note_error(request, "internal_error", f"{type(exc).__name__}")
        return error_response(_request_id(request), 500, "internal_error", GENERIC_500)
