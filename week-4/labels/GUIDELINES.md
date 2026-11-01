# Labeling Guidelines

You create the labels. They are the "ground truth" that the accuracy numbers are measured against, so label honestly and without looking at what the system predicted for the NER sheet. About one hour in total. Open the `.csv` files in Excel, fill the empty column, and save as **CSV (comma delimited)** with the same file name.

## 1. `topic_review.csv` (about 15 minutes)

Each row is one topic, shown by its top words and three example titles.

- **coherence** (required): 
  - `good` – the words and titles are clearly about one theme you could name.
  - `mixed` – there is a main theme, but some words or titles do not fit.
  - `poor` – no clear theme, or several unrelated themes.
- **human_label** (optional but useful): a short name you would give the topic, such as "pedestrian detection".
- **notes** (optional).

## 2. `assignment_check.csv` (about 20 minutes)

Each row is one paper with the topic the system assigned to it.

- **correct** (required): 
  - `y` – the topic words describe what this paper is about.
  - `p` – partly: the topic is related, but not the best description.
  - `n` – the topic does not describe the paper.
- Short abstracts are mixed in on purpose. Judge them the same way.

## 3. `ner_gold.csv` (about 30 minutes)

Each row is a title and abstract. The system's predictions are **not** shown, on purpose.

- **datasets**: the names of datasets or benchmarks that are named in the title or abstract, separated by `;` (for example `ImageNet; COCO`).
  - Include: named datasets, corpora and benchmarks (ImageNet, GLUE, MMLU, Cityscapes, a new dataset that has a name such as "Foo-Bench").
  - Do not include: generic phrases ("a new dataset", "large-scale data"), models (BERT, ResNet), metrics (BLEU, accuracy), organizations, or tools.
  - Write the name as the paper does. Variants such as "MS COCO" and "COCO" are matched automatically for known datasets.
- **labeled** (required for every row you finish): type `x`. This lets an empty `datasets` cell mean "this abstract names no dataset". Rows without an `x` are ignored.
- **notes** (optional).

Rules for both sets: if you are unsure, make a note and decide the same way every time. Do not edit the other columns. Do not leave the sheets half done, because partial sets give misleading numbers.
