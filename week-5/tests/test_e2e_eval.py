"""The end-to-end evaluation: case loading, judging, summary, report, and a full run against a live local server."""

import json
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import pytest
import uvicorn

from api.main import create_app
from api.settings import Settings
from evals.e2e_eval import (build_request, judge, json_record, limit_per_group, load_cases, run_all, run_case,
                            summarize, write_report)
from rag.llm_client import RateLimitError
from tests.fakes import FakeLLM, build_service

ROOT = Path(__file__).resolve().parent.parent
REAL_CASES = ROOT / "eval_data" / "e2e_cases.json"

# Cases that fit the tiny fake database (4 papers) used by the other tests.
SMALL_CASES = [
    {"id": "a1", "group": "in_domain", "path": "/ask", "json": {"question": "Convolutional neural networks detect objects in images"},
     "expect": {"kind": "answer"}, "keywords_any": ["detect"]},
    {"id": "a2", "group": "in_domain", "path": "/ask", "json": {"question": "Convolutional neural networks detect objects in images"},
     "expect": {"kind": "answer"}, "keywords_any": ["zebra"]},                      # keyword check misses
    {"id": "r1", "group": "off_topic", "path": "/ask", "json": {"question": "best recipe for chocolate cake"},
     "expect": {"kind": "refuse"}},
    {"id": "f1", "group": "filtered", "path": "/search", "json": {"query": "object detection", "filters": {"categories": ["cs.CV"]}},
     "expect": {"kind": "answer"}, "expect_categories": ["cs.CV"]},
    {"id": "h1", "group": "http_contract", "path": "/ask", "json": {"question": "   "},
     "expect": {"kind": "http_error", "http": 400, "code": "invalid_question"}},
    {"id": "h2", "group": "http_contract", "path": "/ask", "raw_body": "{\"question\": \"unfinished",
     "expect": {"kind": "http_error", "http": 400, "code": "invalid_json"}},
    {"id": "h3", "group": "http_contract", "path": "/ask", "json": {},
     "expect": {"kind": "http_error", "http": 422, "code": "validation_error"}},
    {"id": "h4", "group": "http_contract", "path": "/nope", "method": "GET",
     "expect": {"kind": "http_error", "http": 404, "code": "not_found"}},
]


def run(case_run=None, **over):
    """A fake HTTP result for judge(). Fields can be overridden."""
    base = {"http": 200, "client_ms": 10.0, "body": {}, "request_id": "rid", "transport_error": None}
    return {**base, **(case_run or {}), **over}


def case(kind="answer", path="/ask", **extra):
    expect = {"kind": kind, **extra.pop("expect", {})}
    return {"id": "c", "group": "g", "path": path, "expect": expect, **extra}


# ------------------------------------------------------------------ cases
def test_the_real_case_file_is_valid_and_covers_every_group():
    cases = load_cases(REAL_CASES)
    assert len(cases) >= 30
    assert {c["group"] for c in cases} == {"in_domain", "filtered", "off_topic", "near_domain", "edge", "http_contract"}
    assert len({c["id"] for c in cases}) == len(cases)


@pytest.mark.parametrize("data, message", [
    ({"cases": []}, "no 'cases'"),
    ({"cases": [{"id": "x", "group": "g", "path": "/ask"}]}, "missing 'expect'"),
    ({"cases": [{"id": "x", "group": "g", "path": "/ask", "expect": {"kind": "maybe"}}]}, "expect.kind"),
    ({"cases": [{"id": "x", "group": "g", "path": "/ask", "expect": {"kind": "http_error"}}]}, "integer 'http'"),
    ({"cases": [{"id": "x", "group": "g", "path": "/a", "expect": {"kind": "any"}},
                {"id": "x", "group": "g", "path": "/a", "expect": {"kind": "any"}}]}, "Duplicate"),
])
def test_bad_case_files_are_rejected_with_a_reason(tmp_path, data, message):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=message):
        load_cases(path)


def test_limit_per_group_keeps_the_first_n_of_each_group():
    kept = limit_per_group(load_cases(REAL_CASES), 2)
    counts = {}
    for c in kept:
        counts[c["group"]] = counts.get(c["group"], 0) + 1
    assert set(counts.values()) == {2}


def test_build_request_variants():
    assert build_request({"path": "/x", "method": "get"}) == {"method": "GET", "path": "/x", "kwargs": {}}
    raw = build_request({"path": "/x", "raw_body": "{bad"})
    assert raw["kwargs"]["content"] == b"{bad" and raw["kwargs"]["headers"]["Content-Type"] == "application/json"
    # an empty JSON object is still sent (the 422 case depends on it)
    assert build_request({"path": "/x", "json": {}})["kwargs"] == {"json": {}}
    repeated = build_request({"path": "/x", "json": {"question": "ab ", "repeat": 3}})
    assert repeated["kwargs"]["json"] == {"question": "ab ab ab "}


# ------------------------------------------------------------------ judge
def test_answer_case_needs_http_200_and_an_answered_status():
    ok = judge(case(), run(body={"status": "answered", "answer": "x"}))
    assert ok["passed"] and ok["reason"] == ""
    refused = judge(case(), run(body={"status": "refused_by_model"}))
    assert not refused["passed"] and "got status 'refused_by_model'" in refused["reason"]
    assert not judge(case(), run(http=503, body={"error": {"code": "llm_timeout", "message": "m", "request_id": "rid"}}))["passed"]


def test_refuse_case_fails_when_the_system_answers():
    assert judge(case("refuse"), run(body={"status": "refused_out_of_domain"}))["passed"]
    assert judge(case("refuse"), run(body={"status": "blocked_ungrounded"}))["passed"]
    wrong = judge(case("refuse"), run(body={"status": "answered"}))
    assert not wrong["passed"] and "the system answered" in wrong["reason"]


def test_http_error_case_checks_status_code_and_error_code():
    err = {"error": {"code": "invalid_json", "message": "m", "request_id": "rid"}}
    exp = {"http": 400, "code": "invalid_json"}
    assert judge(case("http_error", expect=exp), run(http=400, body=err))["passed"]
    assert not judge(case("http_error", expect=exp), run(http=422, body=err))["passed"]
    wrong_code = {"error": {"code": "other", "message": "m", "request_id": "rid"}}
    assert not judge(case("http_error", expect=exp), run(http=400, body=wrong_code))["passed"]


def test_error_body_must_have_the_documented_shape_and_the_matching_request_id():
    exp = {"http": 400, "code": "invalid_json"}
    missing_id = {"error": {"code": "invalid_json", "message": "m"}}
    v = judge(case("http_error", expect=exp), run(http=400, body=missing_id))
    assert v["contract_ok"] is False and not v["passed"]
    other_id = {"error": {"code": "invalid_json", "message": "m", "request_id": "different"}}
    assert judge(case("http_error", expect=exp), run(http=400, body=other_id))["contract_ok"] is False
    assert judge(case("http_error", expect=exp), run(http=400, body={"detail": "plain"}))["contract_ok"] is False


def test_any_case_passes_unless_the_server_fails():
    assert judge(case("any"), run(body={"status": "refused_by_model"}))["passed"]
    assert judge(case("any"), run(http=422, body={"error": {"code": "c", "message": "m", "request_id": "rid"}}))["passed"]
    failed = judge(case("any"), run(http=500, body={"error": {"code": "internal_error", "message": "m", "request_id": "rid"}}))
    assert not failed["passed"] and "server error HTTP 500" in failed["reason"]


def test_a_valid_request_that_gets_a_5xx_fails_even_when_a_refusal_was_expected():
    err = {"error": {"code": "llm_timeout", "message": "m", "request_id": "rid"}}
    assert not judge(case("refuse"), run(http=504, body=err))["passed"]


def test_filter_violation_fails_a_case_that_would_otherwise_pass():
    c = case(expect_categories=["cs.CV"])
    good = judge(c, run(body={"status": "answered", "answer": "x", "sources": [{"category": "cs.CV"}]}))
    assert good["passed"] and good["category_ok"] is True
    bad = judge(c, run(body={"status": "answered", "answer": "x", "sources": [{"category": "cs.CV"}, {"category": "cs.CL"}]}))
    assert not bad["passed"] and bad["category_ok"] is False and "cs.CL" in bad["reason"]


def test_search_case_checks_hits_and_keywords_in_titles_and_text():
    c = case(path="/search", keywords_any=["detect"])
    ok = judge(c, run(body={"count": 1, "hits": [{"title": "Pedestrian detection", "text": "t", "category": "cs.CV"}]}))
    assert ok["passed"] and ok["keyword_ok"] is True
    assert not judge(c, run(body={"count": 0, "hits": []}))["passed"]


def test_keyword_check_is_a_signal_not_a_failure():
    v = judge(case(keywords_any=["zebra"]), run(body={"status": "answered", "answer": "objects in images"}))
    assert v["passed"] and v["keyword_ok"] is False


def test_transport_error_fails_with_the_reason():
    v = judge(case(), run(http=None, transport_error="ConnectError: refused"))
    assert not v["passed"] and "ConnectError" in v["reason"]


# ------------------------------------------------------------------ run and summarize (fake app, no network)
@pytest.fixture
def small_run(make_client):
    client, _, _ = make_client()
    results = run_all(client, SMALL_CASES)
    return {r["id"]: r for r in results}, results


def test_small_run_judges_each_case_as_expected(small_run):
    by_id, _ = small_run
    assert {k: v["passed"] for k, v in by_id.items()} == {k: True for k in by_id}, \
        {k: v["reason"] for k, v in by_id.items() if not v["passed"]}
    assert by_id["a1"]["status"] == "answered" and by_id["a1"]["keyword_ok"] is True
    assert by_id["a2"]["keyword_ok"] is False
    assert by_id["r1"]["status"] == "refused_out_of_domain" and by_id["r1"]["llm_called"] is False
    assert by_id["h2"]["error_code"] == "invalid_json" and by_id["h2"]["contract_ok"] is True
    assert by_id["f1"]["category_ok"] is True
    assert by_id["a1"]["request_id"]


def test_summary_numbers(small_run):
    _, results = small_run
    s = summarize(results)
    assert s["n_cases"] == 8 and s["passed"] == 8 and s["server_errors"] == [] and s["failures"] == []
    assert s["groups"]["http_contract"]["passed"] == 4 and s["llm_calls"] == 2
    q = s["quality"]
    assert q["answer_rate"]["p"] == 1.0 and q["refusal_rate"]["p"] == 1.0 and q["wrongly_answered"] == []
    assert q["keyword_rate"]["n"] == 2 and q["keyword_rate"]["p"] == 0.5   # a1 has its keyword, a2 does not
    assert q["mean_grounded_fraction"] == 1.0 and q["citation_rate"]["p"] == 1.0
    assert q["error_contract_rate"]["n"] == 4 and q["error_contract_rate"]["p"] == 1.0
    assert q["filter_respected_rate"]["p"] == 1.0


def test_latency_summary_separates_answers_gate_refusals_and_search(small_run):
    _, results = small_run
    lat = summarize(results)["latency"]
    assert set(lat) == {"ask, answered", "ask, refused by the retrieval gate (no LLM call)", "search", "ask, all"}
    answered = lat["ask, answered"]
    assert answered["client_ms"]["n"] == 2 and {"retrieval_ms", "llm_ms", "verify_ms", "total_ms"} <= set(answered["stages_ms"])
    assert "llm_ms" not in lat["ask, refused by the retrieval gate (no LLM call)"]["stages_ms"]
    # the client sees the whole round trip, so it cannot be shorter than the server's own total
    assert answered["overhead_ms"]["mean"] >= 0
    assert answered["client_ms"]["p50"] <= answered["client_ms"]["p95"] <= answered["client_ms"]["max"]


def test_failures_are_listed_and_llm_outages_are_told_apart_from_logic_errors(make_client, monkeypatch):
    client, service, _ = make_client(llm=FakeLLM(default=RateLimitError("429")))
    results = run_all(client, SMALL_CASES[:3])
    s = summarize(results)
    # the 503 is the language model service, so it is not counted as a server bug, but the cases still fail
    assert s["llm_unavailable"] == ["a1", "a2"] and s["server_errors"] == []
    assert [f["id"] for f in s["failures"]] == ["a1", "a2"]
    assert "llm_rate_limited" in s["failures"][0]["reason"]


def test_server_errors_are_counted_and_fail_the_case(make_client, monkeypatch):
    client, service, _ = make_client()
    monkeypatch.setattr(service.pipeline.retriever, "search", lambda *a, **k: (_ for _ in ()).throw(OSError("db gone")))
    results = run_all(client, [SMALL_CASES[0], dict(SMALL_CASES[2], expect={"kind": "any"})])
    s = summarize(results)
    assert s["server_errors"] == ["a1", "r1"] and s["passed"] == 0
    assert all(r["http"] == 500 and r["contract_ok"] for r in results)  # the 500 body is still well formed


def test_a_refusal_case_that_gets_an_answer_is_reported_as_wrongly_answered(make_client):
    client, _, _ = make_client()
    wrong = dict(SMALL_CASES[0], id="w1", group="off_topic", expect={"kind": "refuse"})
    s = summarize(run_all(client, [wrong]))
    assert s["quality"]["wrongly_answered"] == ["w1"] and s["quality"]["refusal_rate"]["p"] == 0.0


def test_delay_is_applied_only_after_requests_that_called_the_llm(make_client):
    client, _, _ = make_client()
    slept = []
    run_all(client, SMALL_CASES, delay=7.5, sleep=slept.append)
    assert slept == [7.5, 7.5]   # a1 and a2 called the LLM; the gate refusal, search and errors did not
    slept.clear()
    run_all(client, SMALL_CASES, delay=0, sleep=slept.append)
    assert slept == []


def test_progress_callback_gets_every_result(make_client):
    client, _, _ = make_client()
    seen = []
    run_all(client, SMALL_CASES[:3], progress=seen.append)
    assert [r["id"] for r in seen] == ["a1", "a2", "r1"]


def test_connection_problem_becomes_a_result_not_an_exception():
    class Down:
        def request(self, *a, **k):
            raise httpx.ConnectError("refused")

    r = run_case(Down(), SMALL_CASES[0])
    assert r["http"] is None and "ConnectError" in r["transport_error"]
    s = summarize(run_all(Down(), SMALL_CASES[:2]))
    assert s["transport_errors"] == ["a1", "a2"] and s["passed"] == 0 and s["latency"] == {}


def test_non_json_response_does_not_crash_the_run():
    class Html:
        def request(self, *a, **k):
            return httpx.Response(502, text="<html>bad gateway</html>")

    r = run_case(Html(), SMALL_CASES[0])
    assert r["http"] == 502 and r["body"] is None
    assert not judge(SMALL_CASES[0], r)["passed"]


# ------------------------------------------------------------------ report and log
def test_report_and_json_record(small_run, tmp_path):
    _, results = small_run
    summary = summarize(results)
    text = write_report(tmp_path / "r" / "e2e.md", summary, {"base_url": "http://x", "cases": "c.json", "server": "4 chunks"})
    assert (tmp_path / "r" / "e2e.md").read_text(encoding="utf-8") == text
    for heading in ("# End-to-End Evaluation", "## Summary", "## Behaviour by group", "## Latency", "## Cases that did not pass (0 of 8)"):
        assert heading in text
    assert "8 of 8" in text and "None." in text
    record = json_record(summary, results)
    json.dumps(record)  # must be serializable
    assert len(record["rows"]) == 8 and "body" not in record["rows"][0]


def test_report_names_failures_and_flags_llm_outages_and_wrong_answers(make_client, tmp_path):
    client, _, _ = make_client(llm=FakeLLM(default=RateLimitError("429")))
    results = run_all(client, SMALL_CASES[:3])
    text = write_report(tmp_path / "r.md", summarize(results), {"base_url": "u", "cases": "c", "server": "s"})
    assert "language model service was unavailable" in text and "`a1`" in text
    client2, _, _ = make_client()
    wrong = dict(SMALL_CASES[0], id="w1", expect={"kind": "refuse"}, group="off_topic")
    text2 = write_report(tmp_path / "r2.md", summarize(run_all(client2, [wrong])), {"base_url": "u", "cases": "c", "server": "s"})
    assert "Answered by mistake: w1" in text2


# ------------------------------------------------------------------ the script, against a real local server
def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def live_server(tmp_path):
    """The real API (fake database and LLM) served by uvicorn on a local port, in a thread."""
    service, _ = build_service(tmp_path / "live_db")
    app = create_app(service=service, settings=Settings(log_file=None))
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", access_log=False))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    assert server.started, "the test server did not start"
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=5)


def run_script(*args):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "evaluate_e2e.py"), *args], cwd=ROOT,
                          capture_output=True, text=True, timeout=120)


def test_script_runs_end_to_end_over_http_and_writes_the_report_and_log(live_server, tmp_path):
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps({"cases": SMALL_CASES}))
    report, log = tmp_path / "e2e.md", tmp_path / "e2e.json"
    done = run_script("--base-url", live_server, "--cases", str(cases), "--delay", "0",
                      "--report", str(report), "--log", str(log), "--strict")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "8 of 8" in report.read_text(encoding="utf-8")
    records = json.loads(log.read_text(encoding="utf-8"))
    assert len(records) == 1 and records[0]["n_cases"] == 8 and records[0]["passed"] == 8 and "environment" in records[0]
    assert "PASS" in done.stdout and "FAIL" not in done.stdout


def test_script_strict_mode_exits_with_1_when_a_case_fails(live_server, tmp_path):
    bad = dict(SMALL_CASES[0], id="w1", group="off_topic", expect={"kind": "refuse"})   # the system will answer it
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps({"cases": [bad]}))
    args = ["--base-url", live_server, "--cases", str(cases), "--delay", "0",
            "--report", str(tmp_path / "r.md"), "--log", str(tmp_path / "l.json")]
    assert run_script(*args).returncode == 0                      # without --strict the exit code stays 0
    done = run_script(*args, "--strict")
    assert done.returncode == 1 and "FAIL" in done.stdout and "the system answered" in done.stdout


def test_script_gives_a_clear_message_when_the_server_is_not_running(tmp_path):
    done = run_script("--base-url", f"http://127.0.0.1:{free_port()}", "--report", str(tmp_path / "r.md"),
                      "--log", str(tmp_path / "l.json"))
    assert done.returncode != 0 and "Cannot reach the API" in (done.stdout + done.stderr)
    assert "uvicorn api.main:create_app" in (done.stdout + done.stderr)
    assert "Traceback" not in done.stderr


def test_script_reports_a_server_that_is_not_ready(tmp_path):
    app = create_app(service=None, settings=Settings(log_file=None))
    app.state.startup_error = "FileNotFoundError: Vector database not found"
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="critical", lifespan="off"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    try:
        done = run_script("--base-url", f"http://127.0.0.1:{port}", "--report", str(tmp_path / "r.md"),
                          "--log", str(tmp_path / "l.json"))
    finally:
        server.should_exit = True
        thread.join(timeout=5)
    assert done.returncode != 0 and "not ready" in (done.stdout + done.stderr)
    assert "Vector database not found" in (done.stdout + done.stderr)


def test_script_rejects_a_broken_case_file_and_empty_selection(live_server, tmp_path):
    broken = tmp_path / "broken.json"
    broken.write_text("{\"cases\": []}")
    assert "Setup problem" in run_script("--base-url", live_server, "--cases", str(broken)).stderr
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps({"cases": SMALL_CASES}))
    done = run_script("--base-url", live_server, "--cases", str(cases), "--groups", "no_such_group")
    assert done.returncode != 0 and "No cases left" in done.stderr
