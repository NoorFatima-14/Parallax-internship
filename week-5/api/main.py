"""The FastAPI application.

Run it (from inside week-5/):

    uvicorn api.main:create_app --factory --port 8000 --no-access-log

`--no-access-log` turns off uvicorn's own request lines, because the app writes a richer
JSON line for every request. Interactive documentation is at http://127.0.0.1:8000/docs.
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api import __version__
from api.errors import register_error_handlers
from api.logging_setup import configure_logging
from api.middleware import RequestLoggingMiddleware
from api.routes import router
from api.service import RAGService
from api.settings import Settings

log = logging.getLogger("api.app")

DESCRIPTION = """
A retrieval-augmented question answering service over arXiv abstracts.

* `POST /ask` answers a question from the retrieved passages and checks the answer against them
* `POST /search` returns the closest passages only (no language model)
* Both accept the Week 4 metadata filters (topic, category, dataset, year)
* Every response carries an `X-Request-ID` header, and every request is logged as one JSON line
"""


def create_app(service: RAGService | None = None, settings: Settings | None = None) -> FastAPI:
    """Build the application.

    service: a ready RAGService (used by the tests). If None, it is built at startup from the settings.
    If building fails (missing database, missing API key...), the app still starts: /health works,
    and the other endpoints answer 503 with the reason, instead of the process crashing.
    """
    settings = settings or Settings.from_env()
    configure_logging(settings.log_level, settings.log_file)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.service is None:
            try:
                start = time.perf_counter()
                app.state.service = RAGService.from_settings(settings)
                log.info("service_ready", extra={"fields": {"event": "startup", "load_seconds": round(time.perf_counter() - start, 1)}})
            except Exception as exc:
                app.state.startup_error = f"{type(exc).__name__}: {exc}"
                log.error("service_failed_to_start", exc_info=True, extra={"fields": {"event": "startup"}})
        yield

    app = FastAPI(title="Parallax Labs RAG API", version=__version__, description=DESCRIPTION, lifespan=lifespan)
    app.state.service = service
    app.state.startup_error = None
    app.state.settings = settings
    app.state.started_at = time.time()

    app.add_middleware(RequestLoggingMiddleware)
    register_error_handlers(app)
    app.include_router(router)
    return app
