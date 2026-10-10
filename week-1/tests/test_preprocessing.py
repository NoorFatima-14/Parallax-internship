from src.preprocessing import (
    clean_corpus,
    clean_text,
    is_english,
    normalize_unicode,
    normalize_whitespace,
    strip_html,
)

EN = (
    "Transformers have become the standard approach for many natural language "
    "processing tasks because they model long range dependencies well."
)
FR = (
    "Les modèles de langage sont devenus la méthode standard pour de nombreuses "
    "tâches de traitement automatique des langues naturelles."
)


# --- strip_html ---
def test_strip_html_removes_tags():
    assert strip_html("<p>Hello <b>world</b></p>").split() == ["Hello", "world"]


def test_strip_html_removes_script_and_style():
    html = "<style>p{color:red}</style>Text<script>alert(1)</script>"
    assert strip_html(html).strip() == "Text"


def test_strip_html_decodes_entities():
    assert strip_html("Tom &amp; Jerry &lt;3") == "Tom & Jerry <3"


def test_strip_html_handles_empty_and_none():
    assert strip_html("") == ""
    assert strip_html(None) == ""


def test_strip_html_plain_text_unchanged():
    assert strip_html("no tags here") == "no tags here"


def test_strip_html_keeps_math_comparisons():
    text = "We show that if x < 0.5 and y > 2 then a<b and c>d holds."
    assert strip_html(text) == text


def test_strip_html_removes_comments_and_attributes():
    html = '<!-- note --><a href="http://x.com" class="l">link</a> text'
    assert strip_html(html).split() == ["link", "text"]


# --- normalize_unicode ---
def test_unicode_nfkc_converts_compat_chars():
    assert normalize_unicode("ﬁne \uff21") == "fine A"


def test_unicode_removes_zero_width_chars():
    assert normalize_unicode("a\u200bb\ufeffc") == "abc"


def test_unicode_removes_control_chars_keeps_newline():
    assert normalize_unicode("a\x00b\x07c\nd") == "abc\nd"


def test_unicode_empty():
    assert normalize_unicode("") == ""


# --- normalize_whitespace ---
def test_whitespace_collapses_and_trims():
    assert normalize_whitespace("  a \t b\n\n c  ") == "a b c"


def test_whitespace_empty():
    assert normalize_whitespace("") == ""


# --- is_english ---
def test_is_english_true_for_english():
    assert is_english(EN)


def test_is_english_false_for_french():
    assert not is_english(FR)


def test_is_english_false_for_short_or_empty():
    assert not is_english("hi")
    assert not is_english("")


# --- clean_text ---
def test_clean_text_full_chain():
    raw = "<div>Hello&nbsp;\u200bworld</div>\n\n<p>Second   line</p>"
    assert clean_text(raw) == "Hello world Second line"


# --- clean_corpus ---
def test_clean_corpus_filters_short_foreign_and_duplicates():
    texts = [
        f"<p>{EN}</p>",  # kept
        "Too short.",  # too short
        FR,  # not English
        EN,  # duplicate of the first after cleaning
    ]
    cleaned, kept = clean_corpus(texts, min_words=10)
    assert kept == [0]
    assert cleaned == [EN]