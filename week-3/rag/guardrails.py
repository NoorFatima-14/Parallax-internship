"""Query checks, out-of-domain handling and hallucination checks.

The hallucination checks run after the model answers and use no extra API calls:
1. every [n] citation must point to a passage that was really in the context
2. every number in the answer must appear in the passages
3. every sentence must be supported by at least one passage (embedding similarity,
   or most of its content words found in the passages)
"""

import re

import numpy as np

from rag.prompts import REFUSAL_SENTENCE

_CITATION_RE = re.compile(r"\[(\d+)\]")
_NUMBER_RE = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)*%?")
_WORD_RE = re.compile(r"[a-z][a-z\-]{3,}")
_STOPWORDS = {
    "this", "that", "with", "from", "have", "been", "were", "also", "which", "their", "these",
    "those", "such", "into", "than", "then", "they", "them", "while", "about", "using", "used",
    "based", "more", "most", "some", "other", "both", "each", "only", "over", "between",
    "provided", "documents", "context", "passage", "passages", "paper", "papers", "study",
    "according", "describes", "described", "approach", "approaches", "method", "methods",
}


# ---------------------------------------------------------------- query
def validate_query(query, max_chars=1000):
    """Return (clean_query, issue). clean_query is None when the query cannot be used."""
    if not isinstance(query, str):
        return None, "not_text"
    clean = "".join(ch for ch in query if ch.isprintable() or ch in "\n\t")
    clean = " ".join(clean.split())
    if not clean:
        return None, "empty"
    if len(clean) > max_chars:
        return clean[:max_chars], "truncated"
    return clean, None


# ---------------------------------------------------------------- model refusal
def is_refusal(text: str) -> bool:
    """True if the model said it does not know (the exact sentence from the system prompt)."""
    normal = re.sub(r"\s+", " ", text.replace("\u2019", "'")).strip().lower()
    return normal.startswith(REFUSAL_SENTENCE.lower()[:-1]) or normal.startswith("i don't know based on")


# ---------------------------------------------------------------- text helpers
_ANY_CITATION_RE = re.compile(r"[\[\u3010]\s*(\d+(?:\s*[,\uff0c\u3001]\s*\d+)*)\s*(?:\u2020[^\]\u3011]*)?[\]\u3011]")


def normalize_citations(text: str) -> str:
    """Turn other citation styles into [n]: full-width 【1】, 【1†source】 and [1, 2] -> [1][2].

    Some models use these styles. Without this, real citations would not be recognised.
    """
    return _ANY_CITATION_RE.sub(lambda m: "".join(f"[{n}]" for n in re.findall(r"\d+", m.group(1))), text)


def strip_citations(text: str) -> str:
    return re.sub(r"\s*\[\d+\]", "", text)


def extract_citations(text: str) -> list[int]:
    return [int(n) for n in _CITATION_RE.findall(text)]


def split_sentences(text: str) -> list[str]:
    sentences = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:[-*\u2022]|\d+[.)])\s+", "", line).strip()
        if line:
            sentences.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+", line) if s.strip())
    return sentences


def _normalize_number(token: str) -> str:
    return token.replace(",", "").rstrip("%")


def extract_numbers(text: str) -> list[str]:
    """Numbers worth checking. Small whole numbers (0-10) are skipped: they are mostly counts."""
    text = strip_citations(text)
    text = re.sub(r"^\s*(?:\d+[.)])\s+", "", text, flags=re.M)  # list numbering
    found = []
    for token in _NUMBER_RE.findall(text):
        value = _normalize_number(token)
        if value.isdigit() and int(value) <= 10 and not token.endswith("%"):
            continue
        found.append(value)
    return found


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS}


# ---------------------------------------------------------------- grounding
def check_grounding(answer: str, chunks, encode=None, cfg=None) -> dict:
    """Check an answer against the passages it was given.

    encode: function(list[str]) -> normalized embeddings (optional; without it only the
    word-overlap test is used). Returns a dict with a verdict and the reasons.
    """
    from rag.config import RAGConfig

    cfg = cfg or RAGConfig()
    answer = normalize_citations(answer)
    texts = [c["text"] for c in chunks]
    context_text = " ".join(texts)
    context_numbers = {_normalize_number(t) for t in _NUMBER_RE.findall(context_text)}
    context_words = _content_words(context_text)

    # 1. citations
    cited = extract_citations(answer)
    valid = sorted({n for n in cited if 1 <= n <= len(chunks)})
    invalid = sorted({n for n in cited if not 1 <= n <= len(chunks)})

    # 2. numbers
    unsupported_numbers = sorted({n for n in extract_numbers(answer) if n not in context_numbers})

    # 3. sentence support
    sentences = [strip_citations(s) for s in split_sentences(answer)]
    sentences = [s for s in sentences if len(s.split()) >= 4]
    sims = np.zeros(len(sentences))
    if sentences and encode is not None and texts:
        sims = (np.asarray(encode(sentences)) @ np.asarray(encode(texts)).T).max(axis=1)

    details, supported = [], 0
    for sentence, sim in zip(sentences, sims):
        words = _content_words(sentence)
        lexical = len(words & context_words) / len(words) if words else 1.0
        ok = sim >= cfg.support_threshold or lexical >= cfg.lexical_threshold
        supported += ok
        details.append({"sentence": sentence, "similarity": round(float(sim), 3),
                        "word_overlap": round(lexical, 3), "supported": bool(ok)})

    fraction = supported / len(sentences) if sentences else 0.0
    if not sentences:
        level = "no_checkable_content"
    elif fraction >= cfg.min_grounded_fraction:
        level = "grounded"
    elif fraction >= cfg.partial_fraction:
        level = "partial"
    else:
        level = "ungrounded"

    block_reasons, warnings = [], []
    if level in ("ungrounded", "no_checkable_content"):
        block_reasons.append(level)
    if unsupported_numbers:
        block_reasons.append("unsupported_number")
    if invalid:
        block_reasons.append("invalid_citation")
    if level == "partial":
        warnings.append("partly_unverified")
    if not valid:
        (block_reasons if cfg.require_citation else warnings).append("no_citation")

    return {
        "level": level,
        "grounded_fraction": round(fraction, 3),
        "sentences_checked": len(sentences),
        "min_similarity": round(float(sims.min()), 3) if len(sentences) else None,
        "citations": valid,
        "invalid_citations": invalid,
        "unsupported_numbers": unsupported_numbers,
        "unsupported_sentences": [d for d in details if not d["supported"]],
        "block_reasons": block_reasons,
        "warnings": warnings,
        "blocked": bool(cfg.strict and block_reasons),
    }
