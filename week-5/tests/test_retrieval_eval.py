"""The retrieval evaluation, the test set format and the test set builder."""

import csv
import json

import pytest

from evals.retrieval_eval import evaluate, json_summary, write_report
from evals.testset import load_testset, save_testset, validate_queries
from evals.testset_builder import (SHEET_COLUMNS, build_known_item, group_by_doc, merge_queries, pool_candidates,
                                   read_judgments, split_sentences, write_sheet)
from rag.retriever import Retriever, week2
from tests.fakes import DOCS, HashingModel, build_store


def search_fn_for(retriever):
    return lambda query, n: retriever.search(query, n)[0]


def known(qid, query, doc, subtype="title"):
    return {"id": qid, "query": query, "type": "known_item", "subtype": subtype, "relevant": [doc]}


# ------------------------------------------------------------------ evaluate()
def test_exact_text_query_finds_its_paper_first(tmp_path):
    retriever = build_store(tmp_path / "db")
    text = DOCS[0][6]
    result = evaluate(search_fn_for(retriever), [known("q1", text, "2601.00001v1", "sentence")], ks=[1, 3])
    row = result["rows"][0]
    assert row["mrr"] == 1.0
    assert row["doc"][1] == {"precision": 1.0, "recall": 1.0, "hit": 1.0, "ndcg": 1.0}
    assert row["ideal_precision"][3] == pytest.approx(1 / 3)
    assert row["doc"][3]["precision"] == pytest.approx(1 / 3)  # one relevant paper among three slots
    assert result["failures"] == []


def test_searching_every_paper_gives_full_recall_and_precision_equal_to_share_relevant(tmp_path):
    retriever = build_store(tmp_path / "db")
    topical = {"id": "t1", "query": "object detection in images", "type": "topical", "subtype": "",
               "relevant": ["2601.00001v1", "2601.00004v1"]}
    row = evaluate(search_fn_for(retriever), [topical], ks=[4])["rows"][0]
    assert row["doc"][4]["recall"] == 1.0 and row["doc"][4]["precision"] == 0.5 and row["doc"][4]["hit"] == 1.0


def test_a_query_whose_paper_is_not_in_the_database_is_listed_as_a_failure(tmp_path):
    retriever = build_store(tmp_path / "db")
    queries = [known("ok", DOCS[1][6], "2601.00002v1", "sentence"), known("lost", "anything at all", "9999.99999v9")]
    result = evaluate(search_fn_for(retriever), queries, ks=[1, 5])
    assert [r["id"] for r in result["failures"]] == ["lost"]
    assert result["groups"]["all"]["doc"][5]["recall"]["mean"] == 0.5


def test_groups_cover_all_types_and_subtypes(tmp_path):
    retriever = build_store(tmp_path / "db")
    queries = [known("a", DOCS[0][2], "2601.00001v1", "title"), known("b", DOCS[1][6], "2601.00002v1", "sentence"),
               {"id": "c", "query": "inference", "type": "topical", "subtype": "", "relevant": ["2601.00003v1"]}]
    groups = evaluate(search_fn_for(retriever), queries, ks=[1, 5])["groups"]
    assert set(groups) == {"all", "known_item", "known_item/title", "known_item/sentence", "topical"}
    assert [groups[g]["n"] for g in ("all", "known_item", "topical")] == [3, 2, 1]


def test_doc_level_and_chunk_level_differ_when_a_paper_has_several_chunks(tmp_path):
    w2, model = week2(), HashingModel()
    store = w2.VectorStore(tmp_path / "db", "multi", "fake")
    chunks = [{"id": "d1::0", "doc_id": "d1", "text": "cats purr softly at home"},
              {"id": "d1::1", "doc_id": "d1", "text": "cats purr and cats sleep all day"},
              {"id": "d2::0", "doc_id": "d2", "text": "dogs bark loudly at night"}]
    store.add_chunks(chunks, model.encode([c["text"] for c in chunks], normalize_embeddings=True))
    retriever = Retriever(store, model)
    query = {"id": "q", "query": "cats purr", "type": "topical", "subtype": "", "relevant": ["d1"]}
    row = evaluate(search_fn_for(retriever), [query], ks=[2])["rows"][0]
    assert row["chunk"][2]["precision"] == 1.0     # the top 2 chunks are both from d1
    assert row["doc"][2]["precision"] == 0.5       # the top 2 papers are d1 and d2
    assert row["chunk"][2]["recall"] == row["doc"][2]["recall"] == 1.0


def test_ids_with_arxiv_prefix_in_the_test_set_match_the_database(tmp_path):
    retriever = build_store(tmp_path / "db")
    queries = validate_queries([known("q", DOCS[0][6], "http://arxiv.org/abs/2601.00001v1")])
    assert evaluate(search_fn_for(retriever), queries, ks=[1])["rows"][0]["mrr"] == 1.0


def test_invalid_ks_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        evaluate(lambda q, n: [], [known("q", "x", "d")], ks=[0, 3])
    with pytest.raises(ValueError):
        evaluate(lambda q, n: [], [known("q", "x", "d")], ks=[])


def test_search_function_is_asked_for_overfetch_times_max_k_chunks():
    asked = []
    evaluate(lambda q, n: asked.append(n) or [], [known("q", "x", "d")], ks=[1, 5], overfetch=3)
    assert asked == [15]


def test_report_and_json_summary(tmp_path):
    retriever = build_store(tmp_path / "db")
    queries = [known("a", DOCS[0][2], "2601.00001v1"), known("lost", "nothing", "9999.99999v9")]
    result = evaluate(search_fn_for(retriever), queries, ks=[1, 5])
    text = write_report(tmp_path / "out" / "report.md", result, {"testset": "t.json", "chunks": 4, "embed_model": "fake"})
    assert (tmp_path / "out" / "report.md").read_text(encoding="utf-8") == text
    for expected in ("# Retrieval Evaluation", "Precision@K (doc)", "Recall@K (doc)", "known_item/title",
                     "ideal Precision@K", "`lost`"):
        assert expected in text
    summary = json_summary(result)
    json.dumps(summary)  # must be JSON-serializable
    assert summary["failed_query_ids"] == ["lost"] and len(summary["rows"]) == 2


# ------------------------------------------------------------------ test set file
def test_validate_queries_cleans_and_normalizes():
    out = validate_queries([{"id": " q1", "query": "  hello ", "type": "known_item",
                             "relevant": ["http://arxiv.org/abs/2601.1v1", "2601.1v1"]}])
    assert out == [{"id": " q1", "query": "hello", "type": "known_item", "subtype": "", "relevant": ["2601.1v1"]}]


@pytest.mark.parametrize("bad,message", [
    ([], "no queries"),
    ("text", "no queries"),
    (["x"], "must be an object"),
    ([{"query": "a", "type": "topical", "relevant": ["d"]}], "'id'"),
    ([{"id": "a", "query": " ", "type": "topical", "relevant": ["d"]}], "'query'"),
    ([{"id": "a", "query": "q", "type": "other", "relevant": ["d"]}], "'type'"),
    ([{"id": "a", "query": "q", "type": "topical", "relevant": []}], "'relevant'"),
    ([{"id": "a", "query": "q", "type": "topical", "relevant": [""]}], "'relevant'"),
    ([{"id": "a", "query": "q", "type": "topical", "relevant": "d"}], "'relevant'"),
])
def test_validate_queries_rejects_bad_input(bad, message):
    with pytest.raises(ValueError, match=message):
        validate_queries(bad)


def test_duplicate_ids_are_rejected():
    q = known("same", "x", "d")
    with pytest.raises(ValueError, match="duplicate"):
        validate_queries([q, dict(q)])


def test_save_and_load_round_trip(tmp_path):
    path = tmp_path / "deep" / "set.json"
    save_testset(path, [known("q1", "hello", "d1")], notes="n")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["version"] == 1 and data["notes"] == "n" and data["created"]
    assert load_testset(path) == [known("q1", "hello", "d1")]


def test_load_testset_gives_clear_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="make_retrieval_testset"):
        load_testset(tmp_path / "missing.json")
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_testset(tmp_path / "bad.json")


# ------------------------------------------------------------------ known-item builder
def make_docs():
    long_sentence = "This sentence is long enough to be used as a query because it has many words in it."
    return {
        "d1": {"title": "A study of pedestrian detection", "chunks": [(0, "Short one. " + long_sentence)]},
        "d2": {"title": "Another study of language models", "chunks": [(0, long_sentence + " Tiny.")]},
        "d3": {"title": "Short", "chunks": [(0, long_sentence)]},                                # title too short
        "d4": {"title": "Same shared paper title here", "chunks": [(0, long_sentence)]},         # duplicate title
        "d5": {"title": "same shared paper title here", "chunks": [(0, long_sentence)]},
        "d6": {"title": "No usable sentence in this paper", "chunks": [(0, "Too short. Also short.")]},
        "d7": {"title": "", "chunks": [(0, long_sentence)]},                                      # no title
    }


def test_known_item_skips_papers_without_a_unique_usable_title():
    queries = build_known_item(make_docs(), n_docs=100, seed=1, subtypes=("title",))
    assert sorted(q["relevant"][0] for q in queries) == ["d1", "d2", "d6"]
    assert all(q["type"] == "known_item" and q["subtype"] == "title" for q in queries)


def test_known_item_sentences_have_a_usable_length_and_come_from_the_paper():
    docs = make_docs()
    queries = [q for q in build_known_item(docs, 100, 1) if q["subtype"] == "sentence"]
    assert sorted(q["relevant"][0] for q in queries) == ["d1", "d2"]   # d6 has no sentence of 60+ characters
    for q in queries:
        assert 60 <= len(q["query"]) <= 300 and q["query"] in docs[q["relevant"][0]]["chunks"][0][1]


def test_known_item_is_repeatable_and_respects_the_limit():
    docs = make_docs()
    assert build_known_item(docs, 2, seed=7) == build_known_item(docs, 2, seed=7)
    assert len({q["relevant"][0] for q in build_known_item(docs, 2, seed=7)}) == 2
    ids = [q["id"] for q in build_known_item(docs, 100, 1)]
    assert len(ids) == len(set(ids))


def test_group_by_doc_orders_chunks_and_normalizes_ids():
    docs = group_by_doc(["http://arxiv.org/abs/x1::1", "http://arxiv.org/abs/x1::0"], ["second", "first"],
                        [{"doc_id": "http://arxiv.org/abs/x1", "title": " T ", "chunk_index": 1},
                         {"doc_id": "http://arxiv.org/abs/x1", "title": " T ", "chunk_index": 0}])
    assert docs == {"x1": {"title": "T", "chunks": [(0, "first"), (1, "second")]}}


def test_split_sentences():
    assert split_sentences("One here.  Two there?\nThree!") == ["One here.", "Two there?", "Three!"]


# ------------------------------------------------------------------ pooling and judging
def test_sheet_hides_rank_and_score_and_covers_each_paper_once(tmp_path):
    retriever = build_store(tmp_path / "db")
    queries = [{"id": "tp-01", "query": "object detection images"}, {"id": "tp-02", "query": "bayesian inference"}]
    rows = pool_candidates(search_fn_for(retriever), queries, pool_k=3)
    assert set(rows[0]) == set(SHEET_COLUMNS) and "score" not in rows[0] and "rank" not in rows[0]
    for qid in ("tp-01", "tp-02"):
        docs = [r["doc_id"] for r in rows if r["query_id"] == qid]
        assert len(docs) == 3 and len(set(docs)) == 3
    assert all(r["relevant"] == "" and r["title"] and r["snippet"] for r in rows)


def test_sheet_order_is_shuffled_but_repeatable(tmp_path):
    retriever = build_store(tmp_path / "db")
    queries = [{"id": "tp-01", "query": "object detection images"}]
    first = [r["doc_id"] for r in pool_candidates(search_fn_for(retriever), queries, 4, seed=1)]
    again = [r["doc_id"] for r in pool_candidates(search_fn_for(retriever), queries, 4, seed=1)]
    assert first == again and sorted(first) == sorted(d[0] for d in DOCS)


def fill_sheet(path, answers):
    """answers: {(query_id, doc_id): 'y'/'n'/...}. Everything else stays blank."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["relevant"] = answers.get((row["query_id"], row["doc_id"]), "")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SHEET_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def sheet_rows():
    return [{"query_id": q, "query": f"query {q}", "doc_id": d, "title": "t", "category": "cs.CV", "snippet": "s",
             "relevant": ""} for q, d in [("q1", "a"), ("q1", "b"), ("q2", "c"), ("q2", "d"), ("q3", "e")]]


def test_judged_sheet_becomes_topical_queries(tmp_path):
    sheet = tmp_path / "sheet.csv"
    write_sheet(sheet, sheet_rows())
    fill_sheet(sheet, {("q1", "a"): "Y", ("q1", "b"): "yes", ("q2", "c"): "n", ("q2", "d"): "no", ("q3", "e"): "y"})
    queries, report = read_judgments(sheet)
    assert [(q["id"], q["relevant"]) for q in queries] == [("q1", ["a", "b"]), ("q3", ["e"])]
    assert all(q["type"] == "topical" for q in queries)
    assert report["dropped"] == ["q2"] and report["kept"] == 2 and report["queries_in_sheet"] == 3
    validate_queries(queries)


def test_blank_judgments_are_an_error_unless_allowed(tmp_path):
    sheet = tmp_path / "sheet.csv"
    write_sheet(sheet, sheet_rows())
    fill_sheet(sheet, {("q1", "a"): "y"})
    with pytest.raises(ValueError, match="not judged"):
        read_judgments(sheet)
    queries, report = read_judgments(sheet, allow_unjudged=True)
    assert [q["id"] for q in queries] == ["q1"] and report["unjudged_rows"] == 4


def test_unclear_judgments_are_an_error(tmp_path):
    sheet = tmp_path / "sheet.csv"
    write_sheet(sheet, sheet_rows())
    fill_sheet(sheet, {("q1", "a"): "maybe"})
    with pytest.raises(ValueError, match="y or n"):
        read_judgments(sheet, allow_unjudged=True)


def test_missing_sheet_and_missing_columns_give_clear_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="pool-sheet"):
        read_judgments(tmp_path / "none.csv")
    (tmp_path / "bad.csv").write_text("query_id,query\nq1,hello\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        read_judgments(tmp_path / "bad.csv")


def test_merge_keeps_known_item_and_replaces_topical():
    old = [known("k1", "x", "d1"), {"id": "t-old", "query": "q", "type": "topical", "subtype": "", "relevant": ["d9"]}]
    new = [{"id": "t-new", "query": "q", "type": "topical", "subtype": "", "relevant": ["d2"]}]
    assert [q["id"] for q in merge_queries(old, new)] == ["k1", "t-new"]
