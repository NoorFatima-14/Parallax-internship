"""The two main endpoints: normal behaviour, refusals, filters and response shape."""

from rag.llm_client import LLMResponse

from tests.fakes import GOOD_ANSWER, FakeLLM

QUESTION = "Convolutional neural networks detect objects in images"


# ------------------------------------------------------------------ /ask: normal answers
def test_ask_returns_a_grounded_answer_with_sources_and_timings(make_client):
    client, _, llm = make_client()
    r = client.post("/ask", json={"question": QUESTION})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "answered" and body["answer"] == GOOD_ANSWER
    assert body["request_id"] == r.headers["X-Request-ID"]
    assert body["sources"][0]["n"] == 1 and body["sources"][0]["doc_id"] == "2601.00001v1"
    assert body["sources"][0]["category"] == "cs.CV" and body["model"] == "fake-model"
    assert body["grounding"]["level"] == "grounded" and body["grounding"]["citations"] == [1]
    assert body["warnings"] == [] and body["filter_applied"] is None
    for stage in ("embed_ms", "search_ms", "retrieval_ms", "llm_ms", "verify_ms", "total_ms"):
        assert body["timings_ms"][stage] >= 0
    assert len(llm.calls) == 1


def test_ask_uses_the_question_as_typed_after_cleaning_spaces(make_client):
    client, _, llm = make_client()
    client.post("/ask", json={"question": f"  {QUESTION}\n\n  "})
    assert QUESTION in llm.calls[0][1]["content"]


def test_top_k_limits_the_number_of_sources(make_client):
    client, _, _ = make_client()
    body = client.post("/ask", json={"question": QUESTION, "top_k": 1}).json()
    assert len(body["sources"]) == 1


def test_answered_with_warning_includes_the_warning(make_client):
    client, _, _ = make_client(llm=FakeLLM(default="Convolutional neural networks detect objects in images."))
    body = client.post("/ask", json={"question": QUESTION}).json()
    assert body["status"] == "answered_with_warning" and "no_citation" in body["warnings"]


# ------------------------------------------------------------------ /ask: refusals are normal outcomes (200)
def test_off_topic_question_is_refused_with_200_and_no_llm_call(make_client):
    client, _, llm = make_client()
    r = client.post("/ask", json={"question": "What is the best recipe for chocolate cake?"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "refused_out_of_domain" and body["sources"] == [] and body["model"] is None
    assert "llm_ms" not in body["timings_ms"] and llm.calls == []


def test_model_refusal_is_200(make_client):
    client, _, _ = make_client(llm=FakeLLM(default="I don't know based on the provided documents."))
    r = client.post("/ask", json={"question": QUESTION})
    assert r.status_code == 200 and r.json()["status"] == "refused_by_model"


def test_ungrounded_answer_is_blocked_and_the_text_is_not_returned(make_client):
    invented = "Pedestrian detection reached 99.9 percent accuracy on 5000 images [1]."
    client, _, _ = make_client(llm=FakeLLM(default=invented))
    r = client.post("/ask", json={"question": QUESTION})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "blocked_ungrounded" and "99.9" not in body["answer"]
    assert "unsupported_number" in body["grounding"]["block_reasons"]
    assert body["grounding"]["unsupported_number_count"] == 2
    assert "99.9" not in r.text and "5000" not in r.text  # nothing from the blocked answer is returned


def test_a_question_longer_than_the_limit_is_cut_and_a_warning_is_returned(make_client):
    client, _, _ = make_client()
    long_question = QUESTION + " " + "images " * 200  # about 1,400 characters: allowed, but cut to 1,000
    body = client.post("/ask", json={"question": long_question}).json()
    assert "question_truncated" in body["warnings"]


# ------------------------------------------------------------------ /ask: filters
def test_category_filter_limits_the_sources(make_client):
    client, _, _ = make_client(llm=FakeLLM(default="Transformers use attention for machine translation [1]."))
    body = client.post("/ask", json={"question": "transformers use attention for machine translation",
                                     "filters": {"categories": ["cs.CL"]}}).json()
    assert body["status"] == "answered"
    assert {s["category"] for s in body["sources"]} == {"cs.CL"}
    assert body["filter_applied"] == {"category": {"$in": ["cs.CL"]}}


def test_topic_can_be_chosen_by_a_word_from_its_label(make_client):
    client, _, _ = make_client()
    body = client.post("/ask", json={"question": QUESTION, "filters": {"topics": ["detection"]}}).json()
    assert body["filter_applied"] == {"topic_id": {"$in": [0]}}
    assert {s["doc_id"] for s in body["sources"]} <= {"2601.00001v1", "2601.00004v1"}


def test_topic_id_can_be_a_number_and_dataset_and_year_filters_combine(make_client):
    client, _, _ = make_client()
    body = client.post("/ask", json={"question": QUESTION, "filters": {
        "topics": [0], "datasets": ["MS COCO"], "year_from": 2026}}).json()
    assert body["filter_applied"] == {"$and": [{"topic_id": {"$in": [0]}}, {"ds_coco": {"$eq": True}},
                                               {"published_year": {"$gte": 2026}}]}
    assert [s["doc_id"] for s in body["sources"]] == ["2601.00001v1"]


def test_a_filter_that_matches_nothing_is_a_refusal_not_an_error(make_client):
    client, _, llm = make_client()
    r = client.post("/ask", json={"question": QUESTION, "filters": {"categories": ["cs.RO"]}})
    assert r.status_code == 200 and r.json()["status"] == "refused_out_of_domain" and llm.calls == []


# ------------------------------------------------------------------ /search
def test_search_returns_ranked_hits_with_metadata(make_client):
    client, _, llm = make_client()
    r = client.post("/search", json={"query": QUESTION, "top_k": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 3 and len(body["hits"]) == 3 and body["query"] == QUESTION
    first = body["hits"][0]
    assert first["doc_id"] == "2601.00001v1" and first["category"] == "cs.CV" and first["title"] == "Pedestrian detection"
    assert first["topic_id"] == 0 and first["published_year"] == 2026 and first["datasets"] == "COCO"
    scores = [h["score"] for h in body["hits"]]
    assert scores == sorted(scores, reverse=True)
    assert set(body["timings_ms"]) == {"embed_ms", "search_ms", "total_ms"}
    assert llm.calls == []  # search never calls the language model


def test_search_with_filters(make_client):
    client, _, _ = make_client()
    body = client.post("/search", json={"query": QUESTION, "top_k": 5, "filters": {"categories": ["cs.CL"]}}).json()
    assert body["count"] == 1 and body["hits"][0]["category"] == "cs.CL"


def test_search_with_no_matches_returns_an_empty_list(make_client):
    client, _, _ = make_client()
    r = client.post("/search", json={"query": QUESTION, "filters": {"year_from": 2099}})
    assert r.status_code == 200 and r.json()["hits"] == [] and r.json()["count"] == 0


# ------------------------------------------------------------------ a successful LLM response with details
def test_llm_details_are_returned_as_the_model_name(make_client):
    response = LLMResponse(text=GOOD_ANSWER, model="other-model", usage={"total_tokens": 9})
    client, _, _ = make_client(llm=FakeLLM([response]))
    assert client.post("/ask", json={"question": QUESTION}).json()["model"] == "other-model"
