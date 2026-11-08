"""Request middleware: request ids, timing, one structured log line per request, and a 500 safety net."""

import logging
import re
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from api.errors import GENERIC_500, error_response
from api.logging_setup import request_id_var

log = logging.getLogger("api.request")
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._-]{8,64}$")


def pick_request_id(header_value: str | None) -> str:
    """Keep a client-supplied X-Request-ID if it looks safe, otherwise make a new one."""
    if header_value and _REQUEST_ID_RE.match(header_value):
        return header_value
    return uuid.uuid4().hex


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Adds X-Request-ID and X-Process-Time-ms headers, and logs one JSON line per request.

    Handlers can add their own fields to the line through `request.state.log_extra`.
    If a handler raises an unexpected exception, the traceback is logged here and the
    client receives a generic 500 (never the exception text).
    """

    async def dispatch(self, request: Request, call_next):
        request_id = pick_request_id(request.headers.get("x-request-id"))
        request.state.request_id = request_id
        request.state.log_extra = {}
        token = request_id_var.set(request_id)
        start = time.perf_counter()
        try:
            try:
                response = await call_next(request)
            except Exception as exc:
                log.error("Unhandled error", exc_info=(type(exc), exc, exc.__traceback__))
                request.state.log_extra.update(error_code="internal_error", error_type=type(exc).__name__,
                                               error_message=str(exc)[:300])
                response = error_response(request_id, 500, "internal_error", GENERIC_500)

            duration_ms = round(1000 * (time.perf_counter() - start), 1)
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time-ms"] = str(duration_ms)

            fields = {
                "event": "request",
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
                "client": request.client.host if request.client else None,
                **request.state.log_extra,
            }
            level = logging.ERROR if response.status_code >= 500 else \
                logging.WARNING if response.status_code >= 400 else logging.INFO
            log.log(level, "request", extra={"fields": fields})
            return response
        finally:
            request_id_var.reset(token)
