import pytest

from rag.config import RAGConfig
from rag.guardrails import (
    check_grounding, extract_citations, normalize_citations, extract_numbers, is_refusal, split_sentences,
    strip_citations, validate_query,
)
from tests.fakes import HashingModel

MODEL = HashingModel(dim=512)


def encode(texts):
    return MODEL.encode(list(texts), normalize_embeddings=True)


CHUNKS = [
    {"text": "Convolutional neural networks detect objects in images. The model reached 92.5 percent "
             "accuracy on 1200 test images in benchmark experiments."},
    {"text": "Transformers use attention to model long range dependencies in language translation tasks."},
]
GOOD = "Convolutional neural networks detect objects in images [1]. Transformers use attention for language translation [2]."


# --- query validation ---
@pytest.mark.parametrize("query,expected", [("  hello   world ", "hello world"), ("a\x00b\x07c", "abc")])
def test_validate_query_cleans_text(query, expected):
    assert validate_query(query) == (expected, None)


@pytest.mark.parametrize("query,issue", [("", "empty"), ("   \n\t ", "empty"), (None, "not_text"),
                                         (123, "not_text"), (["x"], "not_text")])
def test_validate_query_rejects_unusable_input(query, issue):
    assert validate_query(query) == (None, issue)


def test_validate_query_truncates_long_input():
    clean, issue = validate_query("word " * 1000, max_chars=100)
    assert len(clean) == 100 and issue == "truncated"


# --- refusal detection ---
@pytest.mark.parametrize("text", [
    "I don't know based on the provided documents.",
    "I don\u2019t know based on the provided documents.",
    "  i don't know based on the provided documents",
])
def test_refusals_are_detected(text):
    assert is_refusal(text)


def test_normal_answers_are_not_refusals():
    assert not is_refusal("CNNs detect objects [1]. I don't know the exact figure.")


# --- text helpers ---
def test_citation_helpers():
    assert extract_citations("A [1] and B [2][3] again [1].") == [1, 2, 3, 1]
    assert strip_citations("Text here [1][2]. More [3].") == "Text here. More."


def test_numbers_skip_small_counts_and_list_numbers_and_citations():
    text = "1. First point.\n2. Second.\nIt has 3 parts [2], reached 92.5% on 1,200 images in 2024."
    assert extract_numbers(text) == ["92.5", "1200", "2024"]


def test_split_sentences_handles_bullets():
    assert split_sentences("- One is here. Two is here!\n1. Three is here?") == [
        "One is here.", "Two is here!", "Three is here?"]


# --- grounding ---
def test_grounded_answer_passes():
    g = check_grounding(GOOD, CHUNKS, encode)
    assert g["level"] == "grounded" and not g["blocked"] and g["citations"] == [1, 2]
    assert g["unsupported_sentences"] == [] and g["block_reasons"] == []


def test_invented_sentence_is_flagged_and_answer_blocked_when_mostly_unsupported():
    answer = "The moon orbits the earth every month [1]. Volcanoes erupt with molten basalt flows [2]."
    g = check_grounding(answer, CHUNKS, encode)
    assert g["level"] == "ungrounded" and g["blocked"]
    assert len(g["unsupported_sentences"]) == 2


def test_one_unsupported_sentence_out_of_two_is_partial_with_a_warning():
    answer = "Convolutional neural networks detect objects in images [1]. The moon orbits the earth every month [2]."
    g = check_grounding(answer, CHUNKS, encode)
    assert g["level"] == "partial" and "partly_unverified" in g["warnings"] and not g["blocked"]


def test_number_not_in_the_passages_blocks_the_answer():
    g = check_grounding("Convolutional neural networks detect objects in images with 99.9% accuracy [1].",
                        CHUNKS, encode)
    assert g["unsupported_numbers"] == ["99.9"] and g["blocked"]
    assert "unsupported_number" in g["block_reasons"]


def test_number_that_is_in_the_passages_is_accepted():
    g = check_grounding("Convolutional neural networks detect objects in images and reached 92.5 percent "
                        "accuracy on 1,200 test images [1].", CHUNKS, encode)
    assert g["unsupported_numbers"] == [] and not g["blocked"]


def test_citation_to_a_missing_passage_blocks_the_answer():
    g = check_grounding("Convolutional neural networks detect objects in images [5].", CHUNKS, encode)
    assert g["invalid_citations"] == [5] and g["blocked"] and "invalid_citation" in g["block_reasons"]


def test_missing_citations_warn_by_default_and_block_when_required():
    answer = "Convolutional neural networks detect objects in images."
    assert "no_citation" in check_grounding(answer, CHUNKS, encode)["warnings"]
    g = check_grounding(answer, CHUNKS, encode, RAGConfig(require_citation=True))
    assert g["blocked"] and "no_citation" in g["block_reasons"]


def test_non_strict_mode_reports_problems_but_does_not_block():
    answer = "The moon orbits the earth every month [1]. Volcanoes erupt with molten basalt flows [2]."
    g = check_grounding(answer, CHUNKS, encode, RAGConfig(strict=False))
    assert g["level"] == "ungrounded" and g["block_reasons"] and not g["blocked"]


def test_word_overlap_alone_can_support_a_sentence_without_embeddings():
    g = check_grounding("Convolutional neural networks detect objects in images [1].", CHUNKS, encode=None)
    assert g["level"] == "grounded"


def test_empty_or_tiny_answers_have_nothing_to_check_and_are_blocked():
    assert check_grounding("", CHUNKS, encode)["level"] == "no_checkable_content"
    assert check_grounding("Yes [1].", CHUNKS, encode)["blocked"]


# --- other citation styles (a model answered with full-width brackets in the live run) ---
@pytest.mark.parametrize("text,expected", [
    ("Detects objects \u30101\u3011.", "Detects objects [1]."),
    ("Detects objects \u30101\u2020source\u3011.", "Detects objects [1]."),
    ("Both agree \u30101\u3011\u30104\u3011.", "Both agree [1][4]."),
    ("Both agree [1, 2].", "Both agree [1][2]."),
    ("Both agree \u30101\uff0c2\u3011.", "Both agree [1][2]."),
    ("Plain [3] stays.", "Plain [3] stays."),
    ("A list [12 items] is not a citation.", "A list [12 items] is not a citation."),
])
def test_normalize_citations(text, expected):
    assert normalize_citations(text) == expected


def test_fullwidth_citations_count_as_citations():
    g = check_grounding("Convolutional neural networks detect objects in images \u30101\u3011.", CHUNKS, encode)
    assert g["citations"] == [1] and "no_citation" not in g["warnings"] and not g["blocked"]


def test_invalid_fullwidth_citation_is_still_caught():
    g = check_grounding("Convolutional neural networks detect objects in images \u30109\u3011.", CHUNKS, encode)
    assert g["invalid_citations"] == [9] and g["blocked"]
