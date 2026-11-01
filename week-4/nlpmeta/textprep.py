"""Text preparation for topic modeling: remove math and markup, keep jargon and acronyms."""

import re
import warnings

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

_URL = re.compile(r"https?://\S+|www\.\S+")
_MATH = re.compile(r"\$[^$]{0,300}\$")
_LATEX = re.compile(r"\\[A-Za-z]+\*?(?:\{[^}]{0,60}\})?")
_SPACE = re.compile(r"\s+")

# Words that appear in almost every abstract and say nothing about the topic.
ACADEMIC_STOPWORDS = {
    "paper", "papers", "propose", "proposed", "proposes", "present", "presents", "presented",
    "novel", "new", "show", "shows", "shown", "demonstrate", "demonstrates", "demonstrated",
    "result", "results", "approach", "approaches", "method", "methods", "work", "works",
    "study", "studies", "introduce", "introduces", "introduced", "address", "addresses",
    "achieve", "achieves", "achieved", "existing", "recent", "recently", "however", "significantly",
    "significant", "extensive", "experiments", "experimental", "experiment", "performance",
    "state", "art", "outperforms", "outperform", "improve", "improves", "improved", "improvement",
    "effective", "effectively", "efficient", "framework", "based", "using", "use", "used",
    "can", "also", "well", "various", "different", "key", "main", "provide", "provides",
    "enable", "enables", "leverage", "leverages", "specifically", "furthermore", "moreover",
    "particular", "finally", "first", "two", "three", "one", "high", "large", "better", "task",
    "tasks", "problem", "problems", "analysis", "evaluation", "evaluate", "evaluated", "across",
    "code", "available", "github", "https", "arxiv", "dataset", "datasets", "benchmark", "benchmarks",
}
STOPWORDS = sorted(set(ENGLISH_STOP_WORDS) | ACADEMIC_STOPWORDS)

# Words of 3+ characters that start with a letter, or like "3D" (digits then letters). Hyphens and
# digits are kept inside a word, so "Vision-Language", "GPT-4" and "3D-CNN" are not broken apart.
# Plain numbers are not words.
TOKEN_PATTERN = r"(?u)\b(?:[A-Za-z][A-Za-z0-9\-]{2,}|[0-9]+[A-Za-z][A-Za-z0-9\-]*)\b"


def clean_for_topics(text: str) -> str:
    """Remove URLs, inline math and LaTeX commands. Acronyms and technical terms stay."""
    if not isinstance(text, str):
        return ""
    text = _URL.sub(" ", text)
    text = _MATH.sub(" ", text)
    text = _LATEX.sub(" ", text)
    return _SPACE.sub(" ", text).strip()


def word_count(text: str) -> int:
    return len(text.split()) if isinstance(text, str) else 0


def make_vectorizer(ngram_range=(1, 2), min_df=5, max_df=0.6, binary=False, stop=True, vocabulary=None) -> CountVectorizer:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return CountVectorizer(
            stop_words=STOPWORDS if stop else None, ngram_range=ngram_range, min_df=min_df,
            max_df=max_df, token_pattern=TOKEN_PATTERN, binary=binary, lowercase=True,
            vocabulary=vocabulary,
        )
