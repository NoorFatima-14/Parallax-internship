"""/health, /ready, /topics, /datasets, request ids, and the 404/405 error shape."""

import re

from fastapi.testclient import TestClient

from api import __version__
from api.main import create_app
from api.settings import Settings


def test_health_returns_ok_and_version(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["version"] == __version__ and body["uptime_seconds"] >= 0


def test_ready_reports_database_and_llm_details(client):
    r = client.get("/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["ready"] is True and body["chunks"] == 4
    assert body["embedding_model"] == "fake" and body["llm_provider"] == "fake" and body["llm_models"] == ["fake-model"]
    assert body["topics_loaded"] and body["datasets_loaded"]
    assert body["settings"] == {"top_k": 5, "min_score": 0.3, "strict": True}
    assert body["request_id"] == r.headers["X-Request-ID"]


def test_topics_and_datasets_lists(client):
    topics = client.get("/topics").json()
    assert topics["available"] is True and [t["topic_id"] for t in topics["topics"]] == [0, 1, 2]
    assert topics["topics"][0]["label"] == "0_image_detection_object"
    datasets = client.get("/datasets").json()
    assert datasets == {"available": True, "datasets": [{"name": "COCO", "docs": 1}]}


def test_lists_are_empty_but_not_an_error_when_week4_files_are_missing(make_client):
    client, _, _ = make_client(filters=False)
    assert client.get("/topics").json() == {"available": False, "topics": []}
    assert client.get("/datasets").json() == {"available": False, "datasets": []}


# ------------------------------------------------------------------ request ids
def test_every_response_has_a_request_id_and_timing_header(client):
    r = client.get("/health")
    assert re.fullmatch(r"[0-9a-f]{32}", r.headers["X-Request-ID"])
    assert float(r.headers["X-Process-Time-ms"]) >= 0


def test_a_safe_client_request_id_is_kept(client):
    r = client.get("/health", headers={"X-Request-ID": "my-trace-12345"})
    assert r.headers["X-Request-ID"] == "my-trace-12345"


def test_an_unsafe_client_request_id_is_replaced(client):
    r = client.get("/health", headers={"X-Request-ID": "bad id with spaces\tand tabs"})
    assert re.fullmatch(r"[0-9a-f]{32}", r.headers["X-Request-ID"])


# ------------------------------------------------------------------ 404 / 405 use the common error shape
def test_unknown_path_is_a_404_with_the_common_error_shape(client):
    r = client.get("/nothing-here")
    assert r.status_code == 404
    error = r.json()["error"]
    assert error["code"] == "not_found" and error["request_id"] == r.headers["X-Request-ID"]


def test_wrong_method_is_a_405_with_an_allow_header(client):
    r = client.get("/ask")
    assert r.status_code == 405 and r.json()["error"]["code"] == "method_not_allowed"
    assert "POST" in r.headers["Allow"]


# ------------------------------------------------------------------ service not ready
def test_app_without_a_service_still_answers_health_but_503_elsewhere():
    app = create_app(service=None, settings=Settings(log_file=None))
    app.state.startup_error = "FileNotFoundError: Vector database not found"
    client = TestClient(app, raise_server_exceptions=False)
    assert client.get("/health").status_code == 200
    for method, path, body in [("get", "/ready", None), ("post", "/ask", {"question": "hi there"}),
                               ("post", "/search", {"query": "hi there"}), ("get", "/topics", None)]:
        r = client.request(method, path, json=body)
        assert r.status_code == 503, path
        error = r.json()["error"]
        assert error["code"] == "service_not_ready" and "Vector database not found" in error["message"]


def test_startup_failure_is_recorded_and_the_app_stays_up(tmp_path):
    settings = Settings(db_path=tmp_path / "missing_db", log_file=None)
    app = create_app(settings=settings)
    with TestClient(app, raise_server_exceptions=False) as client:  # runs the startup code
        assert client.get("/health").status_code == 200
        r = client.get("/ready")
    assert r.status_code == 503
    error = r.json()["error"]
    assert error["code"] == "service_not_ready" and error["message"].startswith("The RAG service is not ready:")
    assert app.state.startup_error  # the reason is kept (here: no API key or no database)
