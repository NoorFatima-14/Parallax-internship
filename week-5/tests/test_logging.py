"""Structured request logging."""

import json
import logging

from api.logging_setup import JsonFormatter, configure_logging, request_id_var
from api.main import create_app
from api.settings import Settings
from fastapi.testclient import TestClient
from tests.fakes import build_service

QUESTION = "Convolutional neural networks detect objects in images"


def request_records(caplog):
    return [r.fields for r in caplog.records if getattr(r, "fields", {}).get("event") == "request"]


def test_one_log_line_per_request_with_the_basic_fields(client, caplog):
    with caplog.at_level("INFO", logger="api"):
        r = client.get("/health", headers={"X-Request-ID": "trace-0001-abcd"})
    [fields] = request_records(caplog)
    assert fields["method"] == "GET" and fields["path"] == "/health" and fields["status_code"] == 200
    assert fields["duration_ms"] >= 0 and "client" in fields
    assert r.headers["X-Request-ID"] == "trace-0001-abcd"


def test_ask_log_line_has_pipeline_details(client, caplog):
    with caplog.at_level("INFO", logger="api"):
        client.post("/ask", json={"question": QUESTION, "top_k": 3})
    [fields] = request_records(caplog)
    assert fields["endpoint"] == "ask" and fields["pipeline_status"] == "answered"
    assert fields["question_chars"] == len(QUESTION) and fields["question_preview"] == QUESTION
    assert fields["top_k"] == 3 and fields["n_sources"] >= 1 and fields["model"] == "fake-model"
    assert fields["grounding_level"] == "grounded" and fields["grounded_fraction"] == 1.0
    assert fields["timings_ms"]["total_ms"] >= 0 and fields["top_score"] > 0.3


def test_search_log_line(client, caplog):
    with caplog.at_level("INFO", logger="api"):
        client.post("/search", json={"query": QUESTION, "filters": {"categories": ["cs.CV"]}})
    [fields] = request_records(caplog)
    assert fields["endpoint"] == "search" and fields["filters_used"] is True and fields["n_results"] >= 1


def test_question_text_can_be_left_out_of_the_log(make_client, caplog):
    client, _, _ = make_client(log_questions=False)
    with caplog.at_level("INFO", logger="api"):
        client.post("/ask", json={"question": QUESTION})
    [fields] = request_records(caplog)
    assert "question_preview" not in fields and fields["question_chars"] == len(QUESTION)
    assert QUESTION not in json.dumps(fields)


def test_log_level_follows_the_status_code(make_client, caplog):
    client, _, _ = make_client()
    with caplog.at_level("INFO", logger="api"):
        client.get("/health")                                      # 200
        client.post("/ask", json={"question": "  "})               # 400
        client.post("/ask", json={})                               # 422
        client.get("/nope")                                        # 404
    levels = [(r.fields["status_code"], r.levelname) for r in caplog.records
              if getattr(r, "fields", {}).get("event") == "request"]
    assert levels == [(200, "INFO"), (400, "WARNING"), (422, "WARNING"), (404, "WARNING")]


def test_error_code_is_in_the_log_line_for_client_errors(client, caplog):
    with caplog.at_level("INFO", logger="api"):
        client.post("/ask", json={"question": QUESTION, "filters": {"topics": ["zzz"]}})
        client.post("/ask", json={})
    codes = [f["error_code"] for f in request_records(caplog)]
    assert codes == ["invalid_filter", "validation_error"]


def test_request_ids_are_unique_per_request_and_match_the_formatted_log_lines(client):
    lines = []

    class Collect(logging.Handler):
        def emit(self, record):  # formatted while the request is still being handled
            lines.append(json.loads(JsonFormatter().format(record)))

    handler = Collect()
    logging.getLogger("api").addHandler(handler)
    try:
        ids = [client.get("/health").headers["X-Request-ID"] for _ in range(3)]
    finally:
        logging.getLogger("api").removeHandler(handler)
    assert len(set(ids)) == 3
    assert [line["request_id"] for line in lines if line.get("event") == "request"] == ids


def test_request_id_is_cleared_after_the_request(client):
    client.get("/health")
    assert request_id_var.get() == "-"


# ------------------------------------------------------------------ the JSON formatter and handlers
def test_formatter_writes_one_valid_json_object_with_request_id():
    token = request_id_var.set("abc-123-xyz")
    try:
        record = logging.LogRecord("api.request", logging.INFO, __file__, 1, "request", None, None)
        record.fields = {"status_code": 200, "timings_ms": {"total_ms": 1.5}}
        line = JsonFormatter().format(record)
    finally:
        request_id_var.reset(token)
    data = json.loads(line)
    assert "\n" not in line
    assert data["request_id"] == "abc-123-xyz" and data["level"] == "INFO" and data["logger"] == "api.request"
    assert data["status_code"] == 200 and data["timings_ms"] == {"total_ms": 1.5} and data["ts"].endswith("+00:00")


def test_formatter_includes_the_exception():
    try:
        raise ValueError("bad thing")
    except ValueError:
        import sys
        record = logging.LogRecord("api", logging.ERROR, __file__, 1, "failed", None, sys.exc_info())
    assert "ValueError: bad thing" in json.loads(JsonFormatter().format(record))["exception"]


def test_log_file_receives_json_lines_with_matching_request_ids(tmp_path):
    log_file = tmp_path / "logs" / "requests.jsonl"
    service, _ = build_service(tmp_path / "db")
    app = create_app(service=service, settings=Settings(log_file=log_file))
    client = TestClient(app, raise_server_exceptions=False)
    ids = [client.post("/ask", json={"question": QUESTION}).headers["X-Request-ID"],
           client.post("/ask", json={}).headers["X-Request-ID"]]
    for handler in logging.getLogger("api").handlers:
        handler.flush()
    lines = [json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines()]
    requests = [line for line in lines if line.get("event") == "request"]
    assert [r["request_id"] for r in requests] == ids
    assert [r["status_code"] for r in requests] == [200, 422]
    configure_logging("INFO", None)  # release the file


def test_configure_logging_twice_does_not_duplicate_handlers():
    configure_logging("INFO", None)
    n = len(logging.getLogger("api").handlers)
    configure_logging("INFO", None)
    assert len(logging.getLogger("api").handlers) == n
