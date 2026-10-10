import re
import unicodedata
from html import unescape

from langdetect import DetectorFactory, LangDetectException, detect

DetectorFactory.seed = 0  # make language detection repeatable

# Zero-width and byte-order-mark characters that add no visible text.
_INVISIBLE = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff"), None)

# Only real HTML tags are removed. Text such as "x < 0.5 and y > 2" (common in
# math and science abstracts) must stay untouched.
_BLOCK = r"p|br|div|li|ul|ol|tr|table|h[1-6]|blockquote|hr"
_INLINE = (
    r"span|a|b|i|u|s|em|strong|sub|sup|code|pre|font|img|td|th|"
    r"html|head|body|title|meta|link|small|big|center|abbr|cite|mark"
)
_SCRIPT_STYLE_RE = re.compile(
    r"<(script|style)\b[^>]*>.*?</\1\s*>", re.IGNORECASE | re.DOTALL
)
# A tag is bare ("<b>", "</b>", "<br/>") or has attributes ("<a href=...>").
# "a<b and c>d" is not matched because "b and c" has no attribute "=".
_TAIL = r"(?:\s+[^<>]*?=[^<>]*)?\s*/?>"
_BLOCK_TAG_RE = re.compile(rf"</?(?:{_BLOCK}){_TAIL}", re.IGNORECASE)
_INLINE_TAG_RE = re.compile(rf"</?(?:{_INLINE}){_TAIL}", re.IGNORECASE)
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)

# Used by the build script to check that no HTML tags remain.
HTML_TAG_RE = re.compile(rf"</?(?:{_BLOCK}|{_INLINE}){_TAIL}", re.IGNORECASE)


def strip_html(text: str) -> str:
    """Remove HTML tags, script/style content, and decode HTML entities."""
    if not text:
        return ""
    text = _COMMENT_RE.sub(" ", text)
    text = _SCRIPT_STYLE_RE.sub(" ", text)
    text = _BLOCK_TAG_RE.sub(" ", text)  # block tags become a space
    text = _INLINE_TAG_RE.sub("", text)  # inline tags just disappear
    return unescape(text)


def normalize_unicode(text: str) -> str:
    """Apply NFKC normalization and drop invisible and control characters."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_INVISIBLE)
    # Drop control characters (category Cc) except tab and newline.
    return "".join(
        ch for ch in text if ch in "\t\n" or unicodedata.category(ch) != "Cc"
    )


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace into single spaces and trim the ends."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def is_english(text: str, min_chars: int = 20) -> bool:
    """Return True if the text is detected as English.

    Text shorter than `min_chars` is rejected because detection is unreliable.
    """
    if not text or len(text) < min_chars:
        return False
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False


def clean_text(text: str) -> str:
    """Run the full cleaning chain: HTML -> Unicode -> whitespace."""
    return normalize_whitespace(normalize_unicode(strip_html(text)))


def clean_corpus(texts, min_words: int = 20, language: str = "en"):
    """Clean a list of texts and return (cleaned_texts, kept_indices).

    Drops documents that are too short, not in the target language, or
    exact duplicates of an earlier document.
    """
    cleaned, kept, seen = [], [], set()
    for i, raw in enumerate(texts):
        text = clean_text(raw)
        if len(text.split()) < min_words:
            continue
        if language == "en" and not is_english(text):
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(text)
        kept.append(i)
    return cleaned, kept