import pytest

from src.chunking import chunk_documents, split_text

# Every sentence is different, so the tests can tell exactly where a chunk starts.
LONG = " ".join(f"Sentence {i} discusses model number {i * 7} in some detail." for i in range(40))


def _check_covers_text(text, chunks):
    """Every chunk is a slice of the text, in order, and only whitespace is left out."""
    pos, prev_end = 0, 0
    for chunk in chunks:
        start = text.find(chunk, pos)
        assert start != -1, "chunk is not a slice of the original text"
        assert text[prev_end:start].strip() == "", "text was lost between chunks"
        prev_end = start + len(chunk)
        pos = start + 1
    assert text.find(chunks[0]) == 0
    assert prev_end == len(text)


def test_empty_and_whitespace_return_no_chunks():
    assert split_text("") == []
    assert split_text("   \n\t ") == []
    assert split_text(None) == []


def test_short_text_is_one_chunk():
    assert split_text("A short sentence.", chunk_size=100) == ["A short sentence."]


def test_chunks_never_exceed_chunk_size():
    chunks = split_text(LONG, chunk_size=200, chunk_overlap=40)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)


def test_no_text_is_lost():
    chunks = split_text(LONG, chunk_size=200, chunk_overlap=40)
    _check_covers_text(LONG, chunks)


def test_consecutive_chunks_overlap():
    chunks = split_text(LONG, chunk_size=200, chunk_overlap=40)
    for prev, nxt in zip(chunks, chunks[1:]):
        m = next(
            (k for k in range(min(len(prev), len(nxt)), 0, -1) if nxt.startswith(prev[-k:])),
            0,
        )
        assert 0 < m <= 40


def test_zero_overlap_gives_no_shared_text():
    chunks = split_text(LONG, chunk_size=200, chunk_overlap=0)
    _check_covers_text(LONG, chunks)
    assert "".join(c + " " for c in chunks).split() == LONG.split()


def test_prefers_sentence_boundaries():
    chunks = split_text(LONG, chunk_size=200, chunk_overlap=0)
    assert all(c.endswith(".") for c in chunks)


def test_paragraphs_split_first():
    text = ("A" * 30 + "\n\n" + "B" * 30 + "\n\n" + "C" * 30)
    assert split_text(text, chunk_size=40, chunk_overlap=0) == ["A" * 30, "B" * 30, "C" * 30]


def test_long_word_without_separators_is_hard_split():
    chunks = split_text("x" * 250, chunk_size=100, chunk_overlap=10)
    assert all(len(c) <= 100 for c in chunks)
    assert chunks[0] == "x" * 100
    assert len(chunks) == 3  # 0-100, 90-190, 180-250


def test_unicode_text_is_kept():
    text = ("Ünïcödé テキスト 文本 " * 40).strip()
    chunks = split_text(text, chunk_size=80, chunk_overlap=10)
    assert all(len(c) <= 80 for c in chunks)
    assert "テキスト" in " ".join(chunks)


def test_math_and_symbols_are_not_changed():
    text = "We show that x < 0.5 and y > 2 holds. " * 20
    joined = " ".join(split_text(text, chunk_size=100, chunk_overlap=10))
    assert "x < 0.5 and y > 2" in joined


@pytest.mark.parametrize("size,overlap", [(0, 0), (-5, 0), (100, -1), (100, 100), (100, 150)])
def test_invalid_settings_raise(size, overlap):
    with pytest.raises(ValueError):
        split_text("some text", chunk_size=size, chunk_overlap=overlap)


def test_same_input_gives_same_output():
    assert split_text(LONG, 200, 40) == split_text(LONG, 200, 40)


def test_chunk_documents_ids_and_metadata():
    docs = [{"id": "doc1", "text": LONG, "title": "T", "category": "cs.CL", "published": "2026"}]
    chunks, skipped = chunk_documents(docs, chunk_size=200, chunk_overlap=40)
    assert skipped == 0
    assert chunks[0]["id"] == "doc1::0"
    assert [c["chunk_index"] for c in chunks] == list(range(len(chunks)))
    assert all(c["n_chunks"] == len(chunks) for c in chunks)
    assert all(c["doc_id"] == "doc1" and c["category"] == "cs.CL" for c in chunks)
    assert len({c["id"] for c in chunks}) == len(chunks)


def test_chunk_documents_skips_missing_id_or_text():
    docs = [
        {"id": "a", "text": ""},
        {"id": None, "text": "has text but no id"},
        {"text": "no id key"},
        {"id": "ok", "text": "fine text"},
    ]
    chunks, skipped = chunk_documents(docs)
    assert skipped == 3
    assert [c["doc_id"] for c in chunks] == ["ok"]
