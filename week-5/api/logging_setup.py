"""Structured (JSON lines) logging.

Every log record becomes one JSON object on one line. Extra fields are passed with
`logger.info("request", extra={"fields": {...}})`. The request id of the request that
is being handled is added to every record automatically (through a context variable).
"""

import contextvars
import json
import logging
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

request_id_var: contextvars.ContextVar = contextvars.ContextVar("request_id", default="-")

LOGGER_NAME = "api"
_HANDLER_MARK = "_api_structured_handler"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_var.get(),
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            entry.update(fields)
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str, ensure_ascii=False)


def configure_logging(level: str = "INFO", log_file=None) -> logging.Logger:
    """Send the `api` loggers to the console and (optionally) to a rotating file, as JSON lines.

    Safe to call more than once: handlers added by an earlier call are replaced.
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level.upper() if isinstance(level, str) else level)
    for handler in list(logger.handlers):
        if getattr(handler, _HANDLER_MARK, False):
            logger.removeHandler(handler)
            handler.close()

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(RotatingFileHandler(path, maxBytes=5_000_000, backupCount=3, encoding="utf-8"))
    for handler in handlers:
        handler.setFormatter(JsonFormatter())
        setattr(handler, _HANDLER_MARK, True)
        logger.addHandler(handler)
    return logger
