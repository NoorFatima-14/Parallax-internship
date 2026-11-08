"""End-to-end evaluation of the running API: answer behaviour, answer quality checks and latency.

Every case in `eval_data/e2e_cases.json` is sent to the API as a real HTTP request. The cases cover
in-domain questions, off-topic and near-domain questions, filtered requests, edge cases and the HTTP error contract.
`run_all` takes any client with a `.request(method, path, **kwargs)` method (an `httpx.Client` for the real
service, or a FastAPI `TestClient` in the tests).

What is checked, with no human rating and no extra LLM call:
- behaviour: answer when the question is in the corpus, refuse when it is not, return the documented HTTP error
  for bad requests, never return a 5xx for a valid request, and respect the metadata filters
- quality signals of the answers: share of sentences supported by the passages (the Week 3 hallucination
  check), valid citations, and a rough relevance check (does the answer contain a word that any good answer would use)
- latency: round trip time seen by the client, and the server's own stage timings

A passed case means "the system did the right thing", not "the answer is correct". Correctness would need a person.
"""

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from evals.metrics import wilson_interval

ANSWERED = {"answered", "answered_with_warning"}
REFUSED = {"refused_by_model", "refused_out_of_domain", "blocked_ungrounded"}
KINDS = {"answer", "refuse", "http_error", "any"}
STAGES = ("embed_ms", "search_ms", "retrieval_ms", "prompt_ms", "llm_ms", "verify_ms", "total_ms")
GROUP_ORDER = ["in_domain", "filtered", "off_topic", "near_domain", "edge", "http_contract"]


# ------------------------------------------------------------------ cases
def load_cases(path) -> list[dict]:
    """Read and check the case file. Raises ValueError with the reason if something is wrong."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = data.get("cases") if isinstance(data, dict) else data
    if not isinstance(cases, list) or not cases:
        raise ValueError("The case file has no 'cases' list.")
    seen = set()
    for c in cases:
        for key in ("id", "group", "path", "expect"):
            if key not in c:
                raise ValueError(f"Case {c.get('id', '?')} is missing '{key}'.")
        if c["id"] in seen:
            raise ValueError(f"Duplicate case id '{c['id']}'.")
        seen.add(c["id"])
        kind = c["expect"].get("kind")
        if kind not in KINDS:
            raise ValueError(f"Case {c['id']}: expect.kind must be one of {sorted(KINDS)}, got {kind!r}.")
        if kind == "http_error" and not isinstance(c["expect"].get("http"), int):
            raise ValueError(f"Case {c['id']}: an http_error case needs an integer 'http'.")
    return cases


def limit_per_group(cases: list[dict], n: int) -> list[dict]:
    """Keep the first n cases of every group (for a quick trial run)."""
    counts, kept = defaultdict(int), []
    for c in cases:
        if counts[c["group"]] < n:
            counts[c["group"]] += 1
            kept.append(c)
    return kept


def build_request(case: dict) -> dict:
    """The HTTP request for a case. `repeat` inside `json` repeats the question text (for long inputs)."""
    method = case.get("method", "POST").upper()
    kwargs = {}
    if "raw_body" in case:  # sent exactly as written, so malformed JSON can be tested
        kwargs = {"content": case["raw_body"].encode("utf-8"), "headers": {"Content-Type": "application/json"}}
    elif "json" in case:
        body = dict(case["json"])
        repeat = body.pop("repeat", None)
        if repeat:
            for key in ("question", "query"):
                if isinstance(body.get(key), str):
                    body[key] = body[key] * int(repeat)
        kwargs = {"json": body}
    return {"method": method, "path": case["path"], "kwargs": kwargs}


# ------------------------------------------------------------------ running
def run_case(client, case: dict) -> dict:
    """Send one request. Never raises: a connection problem becomes `transport_error`."""
    req = build_request(case)
    start = time.perf_counter()
    try:
        response = client.request(req["method"], req["path"], **req["kwargs"])
    except Exception as exc:  # connection refused, timeout, ...
        return {"http": None, "client_ms": 1000 * (time.perf_counter() - start), "body": None,
                "request_id": None, "transport_error": f"{type(exc).__name__}: {exc}"}
    client_ms = 1000 * (time.perf_counter() - start)
    try:
        body = response.json()
    except ValueError:
        body = None
    return {"http": response.status_code, "client_ms": client_ms, "body": body,
            "request_id": response.headers.get("X-Request-ID"), "transport_error": None}


def _error_of(body):
    return body.get("error") if isinstance(body, dict) and isinstance(body.get("error"), dict) else None


def _hit_categories(body) -> list[str]:
    items = body.get("sources") or body.get("hits") or [] if isinstance(body, dict) else []
    return [i.get("category", "") for i in items]


def _answer_text(case: dict, body: dict) -> str:
    if case["path"] == "/search":
        return " ".join(f"{h.get('title', '')} {h.get('text', '')}" for h in body.get("hits", []))
    return body.get("answer", "")


def judge(case: dict, run: dict) -> dict:
    """Compare a response with what the case expects. Returns passed, a reason, and the quality signals."""
    out = {"passed": False, "reason": "", "keyword_ok": None, "category_ok": None, "contract_ok": None}
    if run["transport_error"]:
        out["reason"] = f"no response ({run['transport_error']})"
        return out
    http, body, expect = run["http"], run["body"], case["expect"]
    kind = expect["kind"]
    status = body.get("status") if isinstance(body, dict) else None
    error = _error_of(body)

    # Every error response must follow the documented shape and match the X-Request-ID header.
    if http >= 400:
        out["contract_ok"] = bool(error and error.get("code") and error.get("message")
                                  and error.get("request_id") and error["request_id"] == run["request_id"])

    if kind == "http_error":
        ok = http == expect["http"] and (("code" not in expect) or (error or {}).get("code") == expect["code"])
        reason = "" if ok else f"expected HTTP {expect['http']} {expect.get('code', '')}, got HTTP {http} " \
                               f"{(error or {}).get('code', '')}".strip()
    elif http >= 500:
        ok, reason = False, f"server error HTTP {http} {(error or {}).get('code', '')}".strip()
    elif kind == "any":
        ok, reason = http < 500, ""
    elif http != 200:
        ok, reason = False, f"expected HTTP 200, got HTTP {http} {(error or {}).get('code', '')}".strip()
    elif kind == "answer":
        if case["path"] == "/search":
            ok = isinstance(body, dict) and body.get("count", 0) > 0
            reason = "" if ok else "the search returned no results"
        else:
            ok = status in ANSWERED
            reason = "" if ok else f"expected an answer, got status '{status}'"
    else:  # refuse
        ok = status in REFUSED
        reason = "" if ok else f"expected a refusal, got status '{status}' (the system answered)"

    if ok and out["contract_ok"] is False:
        ok, reason = False, "the error body does not have the documented shape or request id"

    allowed = case.get("expect_categories")
    if allowed and isinstance(body, dict) and http == 200:
        bad = sorted({c for c in _hit_categories(body) if c not in allowed})
        out["category_ok"] = not bad
        if bad and ok:
            ok, reason = False, f"filter not respected: got categories {bad}, allowed {allowed}"

    keywords = case.get("keywords_any")
    if keywords and ok and isinstance(body, dict) and (status in ANSWERED or case["path"] == "/search"):
        text = _answer_text(case, body).lower()
        out["keyword_ok"] = any(k.lower() in text for k in keywords)

    out["passed"], out["reason"] = bool(ok), reason
    return out


def run_all(client, cases: list[dict], delay: float = 0.0, sleep=time.sleep, progress=None) -> list[dict]:
    """Run every case in order. `delay` seconds are waited after each request that called the LLM,
    to stay under the free-tier rate limit (Week 3 used 10 s)."""
    results = []
    for case in cases:
        run = run_case(client, case)
        verdict = judge(case, run)
        body = run["body"] if isinstance(run["body"], dict) else {}
        timings = body.get("timings_ms") or {}
        grounding = body.get("grounding") or {}
        error = _error_of(run["body"]) or {}
        result = {
            "id": case["id"], "group": case["group"], "path": case["path"], "expect": case["expect"]["kind"],
            "http": run["http"], "status": body.get("status"), "client_ms": run["client_ms"],
            "request_id": run["request_id"], "timings_ms": timings, "error_code": error.get("code"),
            "grounded_fraction": grounding.get("grounded_fraction"),
            "cited": (bool(grounding.get("citations")) and not grounding.get("invalid_citations"))
                     if grounding else None,
            "answer_words": len(body.get("answer", "").split()) if body.get("status") in ANSWERED else None,
            "n_warnings": len(body.get("warnings") or []),
            "llm_called": "llm_ms" in timings,
            "question": (case.get("json") or {}).get("question") or (case.get("json") or {}).get("query") or "",
            **verdict,
        }
        results.append(result)
        if progress:
            progress(result)
        if delay and result["llm_called"]:
            sleep(delay)
    return results


# ------------------------------------------------------------------ summary
def _stats(values) -> dict | None:
    a = np.asarray([v for v in values if v is not None], dtype=float)
    if a.size == 0:
        return None
    return {"n": int(a.size), "mean": float(a.mean()), "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)), "max": float(a.max())}


def _rate(flags: list[bool]) -> dict:
    return wilson_interval(sum(flags), len(flags))


def summarize(results: list[dict]) -> dict:
    ok_http = [r for r in results if r["http"] == 200]
    asks = [r for r in ok_http if r["path"] == "/ask"]
    answered = [r for r in asks if r["status"] in ANSWERED]

    groups = {}
    for g in sorted({r["group"] for r in results}, key=lambda g: GROUP_ORDER.index(g) if g in GROUP_ORDER else 99):
        members = [r for r in results if r["group"] == g]
        groups[g] = {"n": len(members), "passed": sum(r["passed"] for r in members),
                     "pass_rate": _rate([r["passed"] for r in members])}

    expect_answer = [r for r in results if r["expect"] == "answer" and r["path"] == "/ask"]
    expect_refuse = [r for r in results if r["expect"] == "refuse"]
    keyword = [r["keyword_ok"] for r in results if r["keyword_ok"] is not None]
    contract = [r["contract_ok"] for r in results if r["contract_ok"] is not None]
    quality = {
        "answer_rate": _rate([r["status"] in ANSWERED for r in expect_answer]),
        "refusal_rate": _rate([r["status"] in REFUSED for r in expect_refuse]),
        "wrongly_answered": [r["id"] for r in expect_refuse if r["status"] in ANSWERED],
        "blocked_in_answer_cases": [r["id"] for r in expect_answer if r["status"] == "blocked_ungrounded"],
        "keyword_rate": _rate(keyword),
        "mean_grounded_fraction": float(np.mean([r["grounded_fraction"] for r in answered
                                                 if r["grounded_fraction"] is not None]))
                                  if any(r["grounded_fraction"] is not None for r in answered) else None,
        "citation_rate": _rate([r["cited"] for r in answered if r["cited"] is not None]),
        "answers_with_warning": sum(r["status"] == "answered_with_warning" for r in answered),
        "mean_answer_words": float(np.mean([r["answer_words"] for r in answered])) if answered else None,
        "error_contract_rate": _rate(contract),
        "filter_respected_rate": _rate([r["category_ok"] for r in results if r["category_ok"] is not None]),
    }

    latency_groups = {
        "ask, answered": [r for r in asks if r["status"] in ANSWERED],
        "ask, refused after the LLM call (model refusal or blocked)":
            [r for r in asks if r["status"] in ("refused_by_model", "blocked_ungrounded")],
        "ask, refused by the retrieval gate (no LLM call)": [r for r in asks if r["status"] == "refused_out_of_domain"],
        "search": [r for r in ok_http if r["path"] == "/search"],
        "ask, all": asks,
    }
    latency = {}
    for name, members in latency_groups.items():
        if not members:
            continue
        with_total = [r for r in members if "total_ms" in r["timings_ms"]]
        latency[name] = {
            "client_ms": _stats([r["client_ms"] for r in members]),
            "stages_ms": {s: _stats([r["timings_ms"].get(s) for r in members]) for s in STAGES
                          if any(s in r["timings_ms"] for r in members)},
            "overhead_ms": _stats([r["client_ms"] - r["timings_ms"]["total_ms"] for r in with_total]),
        }

    return {
        "n_cases": len(results), "passed": sum(r["passed"] for r in results),
        "pass_rate": _rate([r["passed"] for r in results]),
        "groups": groups, "quality": quality, "latency": latency,
        # 5xx answers that are not caused by the language model service (those are listed in llm_unavailable)
        "server_errors": [r["id"] for r in results if r["http"] is not None and r["http"] >= 500
                          and not (r["error_code"] or "").startswith("llm_")],
        "transport_errors": [r["id"] for r in results if r["http"] is None],
        "llm_unavailable": [r["id"] for r in results if (r["error_code"] or "").startswith("llm_")],
        "failures": [{k: r[k] for k in ("id", "group", "question", "reason", "http", "status")}
                     for r in results if not r["passed"]],
        "llm_calls": sum(r["llm_called"] for r in results),
    }


# ------------------------------------------------------------------ report
def _pct(stat: dict) -> str:
    if stat["p"] is None:
        return "n/a"
    return f"{100 * stat['p']:.0f}% ({100 * stat['lo']:.0f}-{100 * stat['hi']:.0f}%, n={stat['n']})"


def _ms(stat: dict | None) -> str:
    return "-" if not stat else f"{stat['mean']:.0f} | {stat['p50']:.0f} | {stat['p95']:.0f} | {stat['max']:.0f}"


GROUP_NOTES = {
    "in_domain": "Questions the corpus can answer. Expected: a grounded answer.",
    "filtered": "Requests with metadata filters. Expected: a result, and every source inside the filter.",
    "off_topic": "Questions unrelated to the papers. Expected: a refusal.",
    "near_domain": "Related to machine learning but not answerable from the abstracts. Expected: a refusal.",
    "edge": "Gibberish, prompt injection, very long text, another language. Expected: no server error.",
    "http_contract": "Bad requests. Expected: the documented 400, 404, 405 or 422 error body.",
}


def write_report(path, summary: dict, header: dict) -> str:
    q, s = summary["quality"], summary
    L = ["# End-to-End Evaluation (latest run)", "",
         f"- API: `{header['base_url']}`, cases: `{header['cases']}` ({s['n_cases']} requests, {s['llm_calls']} called the LLM)",
         f"- Server: {header.get('server', 'unknown')}",
         f"- Rates are shown with a 95% Wilson interval and the number of cases.", "",
         "## Summary", "",
         f"- Cases passed: **{s['passed']} of {s['n_cases']}** ({_pct(s['pass_rate'])})",
         f"- In-domain questions answered: {_pct(q['answer_rate'])}",
         f"- Off-topic and near-domain questions refused: {_pct(q['refusal_rate'])}"
         + (f". **Answered by mistake: {', '.join(q['wrongly_answered'])}**" if q["wrongly_answered"] else ""),
         f"- Sentences supported by the passages (mean over answers): "
         + (f"{q['mean_grounded_fraction']:.3f}" if q["mean_grounded_fraction"] is not None else "n/a"),
         f"- Answers with a valid citation: {_pct(q['citation_rate'])}",
         f"- Answers containing an expected topic word: {_pct(q['keyword_rate'])} (a rough relevance check)",
         f"- Filters respected: {_pct(q['filter_respected_rate'])}. Error bodies in the documented shape: {_pct(q['error_contract_rate'])}",
         f"- Answers with a warning: {q['answers_with_warning']}. Mean answer length: "
         + (f"{q['mean_answer_words']:.0f} words" if q["mean_answer_words"] is not None else "n/a"),
         f"- Server errors (5xx): {len(s['server_errors'])}. No response: {len(s['transport_errors'])}", ""]
    if s["llm_unavailable"]:
        L += [f"> {len(s['llm_unavailable'])} request(s) failed because the language model service was unavailable "
              f"({', '.join(s['llm_unavailable'][:8])}). That is a rate limit, quota or network problem, not an error of "
              "the retrieval or checking logic. Run again with a larger `--delay`.", ""]
    if q["blocked_in_answer_cases"]:
        L += [f"> Answers blocked by the hallucination checks in questions that should be answerable: "
              f"{', '.join(q['blocked_in_answer_cases'])}. Read them: the check may be too strict.", ""]

    L += ["## Behaviour by group", "", "| Group | Passed | Pass rate | What is expected |", "|---|---|---|---|"]
    for g, v in summary["groups"].items():
        L.append(f"| {g} | {v['passed']}/{v['n']} | {_pct(v['pass_rate'])} | {GROUP_NOTES.get(g, '')} |")
    L.append("")

    L += ["## Latency", "",
          "Client time is the full round trip seen by the caller (HTTP included). Server stages come from the "
          "`timings_ms` field of the response. All values in milliseconds.", "",
          "| Requests | n | Client mean | p50 | p95 | max | HTTP overhead (mean) |", "|---|---|---|---|---|---|---|"]
    for name, v in summary["latency"].items():
        c, o = v["client_ms"], v["overhead_ms"]
        L.append(f"| {name} | {c['n']} | {c['mean']:.0f} | {c['p50']:.0f} | {c['p95']:.0f} | {c['max']:.0f} | "
                 f"{o['mean']:.0f} |" if o else f"| {name} | {c['n']} | {c['mean']:.0f} | {c['p50']:.0f} | "
                                               f"{c['p95']:.0f} | {c['max']:.0f} | - |")
    L.append("")
    answered = summary["latency"].get("ask, answered")
    if answered and answered["stages_ms"]:
        L += ["Server stages for answered questions:", "", "| Stage | mean | p50 | p95 | max |", "|---|---|---|---|---|"]
        for stage, st in answered["stages_ms"].items():
            L.append(f"| {stage} | {_ms(st)} |")
        L.append("")
        L += ["Short refusals are faster than real answers, so the 'ask, all' line understates the time of a full "
              "answer. Use the 'answered' line for that.", ""]

    L += [f"## Cases that did not pass ({len(summary['failures'])} of {s['n_cases']})", ""]
    if summary["failures"]:
        for f in summary["failures"][:25]:
            text = f" \"{f['question'][:80]}\"" if f["question"] else ""
            L.append(f"- `{f['id']}` ({f['group']}){text}: {f['reason']}")
        if len(summary["failures"]) > 25:
            L.append(f"- ... and {len(summary['failures']) - 25} more (see the JSON log)")
    else:
        L.append("None.")
    text = "\n".join(L) + "\n"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")
    return text


def json_record(summary: dict, results: list[dict]) -> dict:
    """What is saved in the JSON log: the summary and one light row per request."""
    rows = [{k: (round(r[k], 1) if k == "client_ms" else r[k]) for k in
             ("id", "group", "path", "http", "status", "passed", "reason", "client_ms", "request_id",
              "error_code", "grounded_fraction", "keyword_ok", "timings_ms")} for r in results]
    return {**summary, "rows": rows}
