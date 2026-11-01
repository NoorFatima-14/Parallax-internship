import pandas as pd
import pytest

from nlpmeta.datasets import canonicalize, extract_datasets, find_patterns, find_seed, normalize_key
from nlpmeta.ner import SYSTEMS, extract_corpus, score_system, system_predictions


# --- seed list ---
def test_known_datasets_are_found_with_canonical_names():
    assert find_seed("We evaluate on ImageNet, MS COCO and CIFAR10.") == ["ImageNet", "COCO", "CIFAR-10"]


def test_longest_name_wins_and_both_are_kept_when_both_appear():
    assert find_seed("Results on MMLU-Pro.") == ["MMLU-Pro"]
    assert find_seed("MMLU and MMLU-Pro are used.") == ["MMLU", "MMLU-Pro"]


def test_names_inside_other_words_are_not_matched():
    assert find_seed("Cocoa and COCOA and Scoco are not datasets.") == []
    assert find_seed("coco is lower case") == []


def test_canonicalize_maps_aliases_and_keeps_unknown_names():
    assert canonicalize("MS-COCO") == canonicalize("ms coco") == "COCO"
    assert canonicalize("Unknown-Set") == "Unknown-Set"
    assert normalize_key("MS-COCO") == "mscoco"


def test_no_duplicates_when_a_dataset_is_mentioned_twice():
    assert find_seed("ImageNet is large. ImageNet-1K is a subset of ImageNet.") == ["ImageNet"]


# --- patterns ---
def test_pattern_finds_names_before_dataset_words():
    assert "Foo-Bench" in find_patterns("We introduce the Foo-Bench benchmark for agents.")
    assert "Stanford Cars" in find_patterns("Results on the Stanford Cars dataset are strong.")


def test_pattern_finds_lists_after_evaluation_words():
    names = find_patterns("The model is evaluated on Foo-Bench, Bar-QA and Baz-Set 2 with gains.")
    assert {"Foo-Bench", "Bar-QA"} <= set(names)


@pytest.mark.parametrize("text", [
    "This dataset is small.", "We use a new dataset of images.", "Tested on CPU and GPU hardware.",
    "Experiments on LLMs show gains.", "The proposed benchmark is useful.",
])
def test_pattern_ignores_ordinary_words_and_acronyms(text):
    assert find_patterns(text) == []


def test_pattern_does_not_return_pieces_of_known_names():
    result = extract_datasets("We evaluate on ImageNet, MS COCO and Foo-Bench.")
    assert result["seed"] == ["ImageNet", "COCO"]
    assert result["pattern"] == ["Foo-Bench"]
    assert "MS" not in result["all"] and set(result["seed"]).isdisjoint(result["pattern"])


@pytest.mark.parametrize("value", ["", None])
def test_empty_text(value):
    assert extract_datasets(value) == {"seed": [], "pattern": [], "all": []}


# --- corpus extraction and scoring ---
DF = pd.DataFrame({"id": ["a", "b", "c"], "title": ["ImageNet-X study", "Plain", "Other"],
                   "text": ["We evaluate on ImageNet and MS COCO.", "No names here.", "Results on the Foo-Bench benchmark."]})


def test_extract_corpus_without_spacy_has_all_columns():
    ent = extract_corpus(DF, nlp=None)
    assert list(ent["doc_id"]) == ["a", "b", "c"]
    assert ent.loc[0, "datasets_seed"] == ["ImageNet", "COCO"]
    assert ent.loc[2, "datasets_pattern"] == ["Foo-Bench"] and ent.loc[2, "datasets_seed"] == []
    assert ent.loc[1, "datasets"] == [] and ent.loc[0, "spacy_baseline"] == []


def test_scoring_counts_are_exact():
    ent = extract_corpus(DF, nlp=None)
    pred = system_predictions(ent)
    gold = {"a": {normalize_key("ImageNet"), normalize_key("COCO"), normalize_key("CIFAR-10")}, "b": set(),
            "c": {normalize_key("Foo-Bench")}}
    seed = score_system(gold, pred["seed_list"])
    assert (seed["tp"], seed["fp"], seed["fn"]) == (2, 0, 2)  # missed CIFAR-10 and Foo-Bench
    both = score_system(gold, pred["seed_plus_patterns"])
    assert (both["tp"], both["fp"], both["fn"]) == (3, 0, 1)
    assert both["false_negatives"] == [("a", "cifar10")] and both["recall"] > seed["recall"]


def test_scoring_uses_aliases_so_ms_coco_matches_coco():
    pred = {"a": {normalize_key(canonicalize("COCO"))}}
    gold = {"a": {normalize_key(canonicalize("MS COCO"))}}
    assert score_system(gold, pred)["tp"] == 1


def test_scoring_can_be_limited_to_some_documents():
    ent = extract_corpus(DF, nlp=None)
    gold = {"a": {normalize_key("ImageNet")}, "c": {normalize_key("Foo-Bench")}}
    r = score_system(gold, system_predictions(ent)["seed_plus_patterns"], doc_ids=["c"])
    assert r["tp"] == 1 and r["fp"] == 0 and r["fn"] == 0


def test_all_three_systems_are_scored():
    assert set(system_predictions(extract_corpus(DF, nlp=None))) == set(SYSTEMS)


spacy = pytest.importorskip("spacy")


@pytest.mark.skipif(not spacy.util.is_package("en_core_web_sm"), reason="spaCy model not installed")
def test_spacy_baseline_runs_and_returns_lists():
    from nlpmeta.ner import load_spacy

    ent = extract_corpus(DF, load_spacy())
    assert all(isinstance(x, list) for x in ent["spacy_baseline"]) and (ent["n_spacy_entities"] >= 0).all()
