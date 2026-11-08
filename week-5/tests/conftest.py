import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from api.settings import Settings
from tests.fakes import build_service


@pytest.fixture
def make_app(tmp_path):
    """make_app(llm=None, filters=True, service=None, **settings) -> (app, service, llm).

    The app writes no log file (log_file=None) so tests never touch the repository.
    """
    counter = {"n": 0}

    def _make(llm=None, filters=True, service=None, **settings):
        counter["n"] += 1
        llm_used = llm
        if service is None:
            service, llm_used = build_service(tmp_path / f"db{counter['n']}", llm, filters)
        settings.setdefault("log_file", None)
        app = create_app(service=service, settings=Settings(**settings))
        return app, service, llm_used

    return _make


@pytest.fixture
def make_client(make_app):
    """make_client(...) -> (TestClient, service, llm). Server errors become 500 responses, as in production."""

    def _make(**kwargs):
        app, service, llm = make_app(**kwargs)
        return TestClient(app, raise_server_exceptions=False), service, llm

    return _make


@pytest.fixture
def client(make_client):
    return make_client()[0]
