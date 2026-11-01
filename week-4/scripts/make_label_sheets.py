"""Create the sheets you label by hand. Run once, after run_topic_model.py and run_ner.py.

Run:  python scripts/make_label_sheets.py
Creates in labels/:
  topic_review.csv      rate each topic (good / mixed / poor) and give it a name
  assignment_check.csv  say whether a document's assigned topic is right (y / n / p)
  ner_gold.csv          type the dataset names you find in each abstract (no predictions are shown)
Open them in Excel, fill the empty column(s), save as CSV (keep the same name). Read labels/GUIDELINES.md first.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nlpmeta import cli  # noqa: E402
from nlpmeta.labeling import (  # noqa: E402
    ASSIGN_COLS, NER_COLS, TOPIC_COLS, sample_docs, sample_ner_docs, write_sheet,
)


def main() -> None:
    p = argparse.ArgumentParser()
    cli.add_path_args(p)
    p.add_argument("--max-topics", type=int, default=60, help="topics to review (largest first)")
    p.add_argument("--assign-random", type=int, default=45)
    p.add_argument("--assign-short", type=int, default=15)
    p.add_argument("--ner-random", type=int, default=30)
    p.add_argument("--ner-likely", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true", help="overwrite existing sheets (deletes your labels!)")
    args = p.parse_args()
    P = cli.resolve(args)

    corpus = pd.read_parquet(cli.need(P.corpus, "Build the Week 1 clean corpus first.")).set_index("id")
    doc_topics = pd.read_parquet(cli.need(P.doc_topics, "Run scripts/run_topic_model.py first."))
    entities = pd.read_parquet(cli.need(P.entities, "Run scripts/run_ner.py first."))
    topics = json.loads(cli.need(P.topics, "Run scripts/run_topic_model.py first.").read_text(encoding="utf-8"))["topics"]
    topic_words = {t["topic_id"]: t["label"] for t in topics}

    topic_rows = [{"topic_id": t["topic_id"], "size": t["size"], "top_words": "; ".join(t["words"][:10]),
                   "example_titles": " || ".join(t["examples"])} for t in topics[: args.max_topics]]
    write_sheet(P.labels / "topic_review.csv", TOPIC_COLS, topic_rows, args.force)

    assign_rows = []
    for pos, stratum in sample_docs(doc_topics, args.assign_random, args.assign_short, args.seed):
        d = doc_topics.iloc[pos]
        doc = corpus.loc[d["doc_id"]]
        assign_rows.append({"doc_id": d["doc_id"], "title": doc["title"], "abstract": doc["text"],
                            "assigned_topic_id": int(d["topic_id"]),
                            "assigned_topic_words": topic_words.get(int(d["topic_id"]), ""),
                            "stratum": stratum, "words": int(d["words"])})
    write_sheet(P.labels / "assignment_check.csv", ASSIGN_COLS, assign_rows, args.force)

    ner_rows = []
    for pos, stratum in sample_ner_docs(entities, args.ner_random, args.ner_likely, args.seed):
        doc_id = entities.iloc[pos]["doc_id"]
        doc = corpus.loc[doc_id]
        ner_rows.append({"doc_id": doc_id, "title": doc["title"], "abstract": doc["text"], "stratum": stratum})
    write_sheet(P.labels / "ner_gold.csv", NER_COLS, ner_rows, args.force)

    print(f"Created 3 sheets in {P.labels}:")
    print(f"  topic_review.csv      {len(topic_rows)} topics   (about 15 minutes)")
    print(f"  assignment_check.csv  {len(assign_rows)} documents (about 20 minutes)")
    print(f"  ner_gold.csv          {len(ner_rows)} documents (about 30 minutes)")
    print("Read labels/GUIDELINES.md, then fill the sheets. Do the NER sheet before looking at any NER output.")


if __name__ == "__main__":
    main()
