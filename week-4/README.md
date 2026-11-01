# Week 4: NLP Metadata and Filtered Retrieval

Week 4 of the Parallax Labs AI/ML internship. It adds NLP metadata to the Week 2 vector database (27,329 chunks of 6,563 arXiv abstracts) and uses it to filter search and RAG answers. The Week 1 corpus and the Week 2 database must exist. Run all commands below from inside the `week-4/` folder.

## What I Built

- **Topic modeling** (`nlpmeta/topics.py`): BERTopic on the document vectors that are already in the database, with an LDA baseline for comparison
- **Dataset-mention extraction** (`nlpmeta/datasets.py`, `nlpmeta/ner.py`): finds named datasets and benchmarks (ImageNet, GLUE, MMLU...) with a seed list plus patterns, and compares it with spaCy's general NER
- **Label sheets and scoring** (`nlpmeta/labeling.py`, `scripts/make_label_sheets.py`, `scripts/evaluate_labels.py`): sheets for hand labeling, and accuracy measured against those labels
- **Enrichment** (`nlpmeta/enrich.py`, `scripts/enrich_vector_db.py`): writes topic, dataset, date and length metadata into every chunk
- **Filtered retrieval** (`nlpmeta/filters.py`, `scripts/search_filtered.py`, `scripts/ask_filtered.py`): filter by topic, category, dataset, year and more, in search and in the Week 3 RAG system
- **Filter benchmark** (`scripts/benchmark_filters.py`): checks that filters never return a wrong chunk, and measures their speed and effect on recall
- **Tests** (`tests/`): unit tests, including a check that the database filter gives exactly the same chunks as an independent Python implementation of the rules

## Why This Metadata

Searching 27,000 chunks by meaning alone cannot answer questions like "only papers about uncertainty", "papers that use the COCO benchmark" or "only 2026 papers". With topics, dataset mentions and dates stored next to each chunk, search can be narrowed before the vector comparison, and the RAG system can answer from a chosen part of the corpus.

## Approach

**Topics.** Each document's vector is the average of its chunk vectors from the Week 2 database, so nothing is embedded again. BERTopic (UMAP, HDBSCAN, c-TF-IDF) finds the topics. An LDA model with the same number of topics is the baseline. They are compared with NPMI coherence, topic diversity, and my own ratings of the topics.

**Why dataset mentions and not sentiment.** The task allows either. I chose NER for dataset and benchmark names because they are useful to filter on. I did not measure sentiment on these abstracts.

**Three systems for dataset mentions**, all scored against the same hand-labeled set:
1. spaCy `en_core_web_sm`, keeping entities it labels ORG, PRODUCT, WORK_OF_ART or EVENT (it labels names like ImageNet as ORG)
2. a seed list of about 140 well-known datasets and benchmarks, with aliases (for example "MS COCO" and "COCO" are the same dataset)
3. the seed list plus patterns that find unknown names ("the Foo-Bench benchmark", "evaluated on A, B and C")

**Edge cases.**
- *Short documents.* The shortest 10% of abstracts (at most 100 words) are not used to fit BERTopic, because their vectors are unstable. They are assigned afterwards to the nearest topic and marked `short_doc`. They are also labeled separately in the manual check, so their accuracy is reported on its own.
- *Outliers.* HDBSCAN leaves some documents outside every cluster. They are assigned to the nearest topic too, and marked `topic_core = false`, so filters can use `--core-only` to keep only well-clustered papers.
- *Jargon.* Hyphens, digits and acronyms stay inside words (`vision-language`, `GPT-4`, `3D-CNN`). Inline math, LaTeX commands and URLs are removed. A list of words that appear in nearly every abstract ("propose", "novel", "results") is ignored.
- *A bug found by testing.* BERTopic builds its word list from one text per topic, so a normal `min_df` rule would remove the words that only one topic uses. The word list is therefore chosen on documents (in at least 5 documents and at most half of them) before BERTopic sees it.

## Metadata Added to Every Chunk

| Key | Meaning |
|-----|---------|
| `topic_id`, `topic_label` | The topic of the paper and its top words |
| `topic_core` | `true` if the paper is inside a found cluster, `false` if it was assigned afterwards |
| `topic_sim` | Cosine similarity between the paper's vector and its topic centre |
| `short_doc`, `doc_words` | Whether the abstract is short, and its length in words |
| `has_dataset`, `datasets` | Whether a dataset is mentioned, and the names (text, for display) |
| `ds_<name>` | `true` for papers that mention that dataset (only for datasets in at least 3 papers), for example `ds_imagenet` |
| `published_year`, `published_ym` | Numbers (`2026`, `202610`) for date range filters |

ChromaDB metadata can only hold single values, not lists. That is why each frequent dataset has its own true/false key. The existing keys from Week 2 (category, title, doc_id...) are kept.

## Filters

Different kinds of filters are combined with AND. Several values of the same kind are combined with OR (`--dataset-mode all` requires all datasets).

```bash
python scripts/search_filtered.py --list-topics
python scripts/search_filtered.py --list-datasets
python scripts/search_filtered.py "how is uncertainty estimated" --topic uncertainty
python scripts/search_filtered.py "object detection" --dataset COCO --dataset ImageNet --dataset-mode all
python scripts/search_filtered.py "language models" --category cs.CL --year-from 2026 --exclude-short
python scripts/ask_filtered.py "What methods estimate uncertainty?" --topic uncertainty --core-only
```

`--topic` accepts a topic id or a word from its label. Wrong names give an error with suggestions. A filter that matches nothing returns an empty result, not an error.

## How to Run

```bash
cd week-4
..\week-1\.venv\Scripts\activate         # Windows
pip install bertopic umap-learn hdbscan scikit-learn
pytest -v                                # unit tests, no data needed

python scripts/run_topic_model.py        # topics (BERTopic and LDA), a few minutes
python scripts/run_ner.py                # dataset mentions and spaCy entities
python scripts/make_label_sheets.py      # creates the sheets in labels/
```

Then label the three sheets by hand (about one hour, see `labels/GUIDELINES.md`) and continue:

```bash
python scripts/evaluate_labels.py        # accuracy from your labels
python scripts/enrich_vector_db.py       # writes the metadata into ChromaDB
python scripts/benchmark_filters.py      # filter correctness, speed and recall
```

Labeling is done by me (the author) and not by the system. The NER sheet shows no predictions, so labeling is not influenced by them.

## How the Accuracy Is Measured

| What | How |
|------|-----|
| Topic quality | NPMI coherence and diversity, and my rating (good, mixed, poor) of the 60 largest topics. The report also shows whether the automatic score agrees with my ratings. |
| Topic assignment | 45 random papers and 15 short papers. For each I say whether the assigned topic fits (y, partly, n). Reported with 95% Wilson intervals, for all papers and for short papers on their own. |
| Dataset mentions | 30 random papers and 20 papers where some system found a dataset, labeled without seeing predictions. Micro precision, recall and F1 for the three systems, with intervals, for all labeled papers and for the random ones only. Only the random papers give an estimate for the whole corpus. |
| Filters | Violations (results that break the filter, must be 0), recall@5 and MRR using the first sentence of random chunks as queries, latency, and the number of chunks that pass each filter |

## Results

All numbers are written by the scripts. Nothing is entered by hand.

- `logs/topic_model_comparison.md`: BERTopic and LDA side by side (topics, share of papers in found clusters, NPMI, diversity, time) and the topic list
- `logs/ner_summary.md`: how many papers mention datasets, the most common datasets, and names found only by patterns
- `logs/nlp_metadata_evaluation.md`: topic ratings, assignment accuracy, precision, recall and F1 of the three dataset systems, and the mistakes of the best one
- `logs/enrichment_log.json`: how many chunks were updated and how many have each kind of metadata
- `logs/filter_benchmark.md`: correctness, latency and recall for each kind of filter
- `labels/*.csv`: the hand labels, kept in the repository as the evidence for the accuracy numbers

## Limits

- **One person labeled everything**, so there is no agreement score between labelers, and the labels may contain mistakes.
- **Small samples** (60 topics, 60 assignments, 50 abstracts for dataset names). The 95% intervals are wide, and they are shown for that reason.
- **The filter benchmark assumes the right topic is chosen.** A filter built from a chunk's own topic always contains that chunk, so a higher recall shows the effect of a smaller search space. It does not show how well a user finds the right topic.
- **The seed list is written by hand**, covers mostly computer vision and NLP, and is English only. The patterns are heuristics and can return wrong names.
- **Document vectors are averages of chunk vectors**, not an embedding of the whole abstract.
- **Topic settings were not tuned.** The number of topics depends on `--min-topic-size` (default 25).
- **The metadata becomes outdated** if the corpus or the database changes. Run the topic, NER and enrichment scripts again after re-ingesting.
- Sentiment was not measured.

## Notes

- `data/processed/` (topic and entity tables) and `chroma_db/` are git-ignored and are recreated by the scripts. The `labels/` sheets and the `logs/` are committed.
- Run `pytest` from inside `week-4/`, not from the repository root.
