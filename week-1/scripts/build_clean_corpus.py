
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Allow `python scripts/build_clean_corpus.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing import HTML_TAG_RE, clean_text, is_english

REQUIRED_COLUMNS = ["id", "title", "text", "published", "category"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/arxiv_abstracts.jsonl")
    parser.add_argument("--output", default="data/processed/clean_corpus.parquet")
    parser.add_argument("--min-words", type=int, default=20)
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        df = pd.DataFrame(json.loads(line) for line in f)
    n_raw = len(df)

    df["text"] = df["text"].map(clean_text)
    df["title"] = df["title"].map(clean_text)

    df = df[df["text"].str.split().str.len() >= args.min_words]
    n_after_length = len(df)

    df = df[df["text"].map(is_english)]
    n_after_lang = len(df)

    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    n_final = len(df)

    # Validation: fail loudly if the output is not what we expect.
    assert list(df.columns) == REQUIRED_COLUMNS, f"unexpected columns: {df.columns}"
    assert not df["text"].isna().any(), "null text found"
    assert not df["text"].str.contains(HTML_TAG_RE).any(), "HTML tags left in text"
    assert not df["text"].str.contains(r"\s{2,}").any(), "extra whitespace left"
    assert df["id"].is_unique, "duplicate ids"

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)

    print(f"raw documents:            {n_raw}")
    print(f"after min-length filter:  {n_after_length}")
    print(f"after language filter:    {n_after_lang}")
    print(f"after deduplication:      {n_final}")
    print(f"mean words per document:  {df['text'].str.split().str.len().mean():.1f}")
    print(f"saved to {args.output}")
    if n_final < 5000:
        print("WARNING: fewer than 5,000 documents. Download more raw data.")


if __name__ == "__main__":
    main()