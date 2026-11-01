import pandas as pd
import pytest

from nlpmeta.datasets import normalize_key
from nlpmeta.labeling import (
    NER_COLS, parse_assignments, parse_ner_gold, parse_topic_review, read_sheet, sample_docs,
    sample_ner_docs, write_sheet,
)


def test_sheet_roundtrip_keeps_accents_and_commas(tmp_path):
    path = tmp_path / "s.csv"
    write_sheet(path, NER_COLS, [{"doc_id": "a", "title": "Café, résumé", "abstract": "line one\nline two"}])
    row = read_sheet(path)[0]
    assert row["title"] == "Café, résumé" and "line two" in row["abstract"] and row["datasets"] == ""


def test_existing_sheet_is_not_overwritten_without_force(tmp_path):
    path = tmp_path / "s.csv"
    write_sheet(path, NER_COLS, [{"doc_id": "a"}])
    with pytest.raises(FileExistsError, match="--force"):
        write_sheet(path, NER_COLS, [{"doc_id": "b"}])
    write_sheet(path, NER_COLS, [{"doc_id": "b"}], force=True)
    assert read_sheet(path)[0]["doc_id"] == "b"


# --- reading labels ---
def test_topic_review_accepts_words_and_letters_and_reports_bad_values():
    rows = [{"topic_id": "0", "coherence": "Good", "human_label": "detection"}, {"topic_id": "1", "coherence": "m"},
            {"topic_id": "2", "coherence": ""}, {"topic_id": "3", "coherence": "great"}]
    done, problems = parse_topic_review(rows)
    assert [(d["topic_id"], d["coherence"]) for d in done] == [(0, "good"), (1, "mixed")]
    assert len(problems) == 1 and "topic 3" in problems[0]


def test_assignments_scores_and_problems():
    rows = [{"doc_id": "a", "correct": "Y", "assigned_topic_id": "1", "stratum": "random"},
            {"doc_id": "b", "correct": "partly", "assigned_topic_id": "2", "stratum": "short"},
            {"doc_id": "c", "correct": "no", "assigned_topic_id": "3", "stratum": "random"},
            {"doc_id": "d", "correct": "", "assigned_topic_id": "3", "stratum": "random"},
            {"doc_id": "e", "correct": "maybe", "assigned_topic_id": "3", "stratum": "random"}]
    done, problems = parse_assignments(rows)
    assert [d["score"] for d in done] == [1.0, 0.5, 0.0] and len(problems) == 1


def test_ner_gold_only_counts_rows_marked_as_labeled():
    rows = [{"doc_id": "a", "datasets": "MS COCO; ImageNet", "labeled": "x", "stratum": "random"},
            {"doc_id": "b", "datasets": "", "labeled": "x", "stratum": "likely_mention"},
            {"doc_id": "c", "datasets": "GLUE", "labeled": "", "stratum": "random"}]
    gold, strata, problems = parse_ner_gold(rows)
    assert gold["a"] == {normalize_key("COCO"), normalize_key("ImageNet")}  # 'MS COCO' becomes COCO
    assert gold["b"] == set() and "c" not in gold  # empty + marked = no dataset; unmarked = not labeled
    assert strata["b"] == "likely_mention" and "1 document" in problems[0]


def test_ner_gold_accepts_new_lines_between_names():
    gold, _, _ = parse_ner_gold([{"doc_id": "a", "datasets": "GLUE\nSQuAD", "labeled": "x"}])
    assert gold["a"] == {normalize_key("GLUE"), normalize_key("SQuAD")}


# --- sampling ---
def _doc_table(n=200, n_short=20):
    return pd.DataFrame({"doc_id": [f"d{i}" for i in range(n)], "short_doc": [i < n_short for i in range(n)]})


def test_sample_docs_is_repeatable_and_has_the_requested_groups():
    table = _doc_table()
    a, b = sample_docs(table, 45, 15, seed=1), sample_docs(table, 45, 15, seed=1)
    assert a == b and sample_docs(table, 45, 15, seed=2) != a
    assert sum(s == "random" for _, s in a) == 45 and sum(s == "short" for _, s in a) == 15
    assert len({pos for pos, _ in a}) == 60
    assert all(table.iloc[pos]["short_doc"] == (s == "short") for pos, s in a)


def test_sample_docs_does_not_fail_when_there_are_few_short_documents():
    picks = sample_docs(_doc_table(100, 3), 10, 15)
    assert sum(s == "short" for _, s in picks) == 3


def test_sample_ner_docs_separates_likely_and_random_documents():
    ent = pd.DataFrame({"doc_id": [f"d{i}" for i in range(100)],
                        "datasets": [["ImageNet"] if i < 30 else [] for i in range(100)]})
    picks = sample_ner_docs(ent, 30, 20)
    assert sum(s == "likely_mention" for _, s in picks) == 20 and sum(s == "random" for _, s in picks) == 30
    assert len({pos for pos, _ in picks}) == 50
    assert all(len(ent.iloc[p]["datasets"]) > 0 for p, s in picks if s == "likely_mention")


def test_sample_ner_docs_when_nothing_was_found():
    ent = pd.DataFrame({"doc_id": ["a", "b", "c"], "datasets": [[], [], []]})
    assert all(s == "random" for _, s in sample_ner_docs(ent, 2, 5))
