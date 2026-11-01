"""Topic modeling: BERTopic (main method) and LDA (baseline), with short-document handling.

Document vectors are the average of each document's chunk vectors from the Week 2
database, so nothing is embedded again. Documents that are too short for a stable vector
are not used to fit the model. They are assigned afterwards to the nearest topic.
"""

import time

import numpy as np
import pandas as pd

from nlpmeta.metrics import npmi_coherence, topic_diversity
from nlpmeta.textprep import clean_for_topics, make_vectorizer, word_count


# ---------------------------------------------------------------- embeddings
def load_doc_embeddings(collection, batch: int = 5000):
    """Average the chunk embeddings of each document. Returns (doc_ids, matrix of unit vectors)."""
    sums, counts, offset = {}, {}, 0
    while True:
        res = collection.get(include=["embeddings", "metadatas"], limit=batch, offset=offset)
        if len(res["ids"]) == 0:
            break
        for vec, meta in zip(np.asarray(res["embeddings"]), res["metadatas"]):
            doc_id = (meta or {}).get("doc_id")
            if not doc_id:
                continue
            sums[doc_id] = sums.get(doc_id, 0) + vec
            counts[doc_id] = counts.get(doc_id, 0) + 1
        if len(res["ids"]) < batch:
            break
        offset += batch
    doc_ids = list(sums)
    if not doc_ids:
        raise ValueError("No chunk has a doc_id. Was the collection built by Week 2's ingest.py?")
    matrix = np.vstack([sums[d] / counts[d] for d in doc_ids]).astype(np.float32)
    matrix /= np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-9, None)
    return doc_ids, matrix


def align_embeddings(corpus_ids, doc_ids, matrix):
    """Reorder embeddings to the corpus order. Returns (matrix, mask of documents that have a vector)."""
    index = {d: i for i, d in enumerate(doc_ids)}
    out = np.zeros((len(corpus_ids), matrix.shape[1]), dtype=np.float32)
    have = np.zeros(len(corpus_ids), dtype=bool)
    for row, d in enumerate(corpus_ids):
        if d in index:
            out[row], have[row] = matrix[index[d]], True
    return out, have


# ---------------------------------------------------------------- models
def fit_bertopic(texts, embeddings, min_topic_size=25, nr_topics=None, seed=42, top_n=10):
    """BERTopic on precomputed embeddings. Returns (labels per text, {topic: words})."""
    from bertopic import BERTopic
    from umap import UMAP

    try:
        from hdbscan import HDBSCAN

        clusterer = HDBSCAN(min_cluster_size=min_topic_size, metric="euclidean",
                            cluster_selection_method="eom")
    except ImportError:  # scikit-learn 1.3+ has its own HDBSCAN
        from sklearn.cluster import HDBSCAN

        clusterer = HDBSCAN(min_cluster_size=min_topic_size)
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=seed)
    # BERTopic fits its vectorizer on one text per topic, so min_df/max_df would count topics, not
    # documents, and would remove the words that only one topic uses. The word list is therefore chosen
    # here, on documents: words in at least 5 documents and in at most half of them.
    vocabulary = make_vectorizer(ngram_range=(1, 2), min_df=5, max_df=0.5).fit(texts).vocabulary_
    vectorizer = make_vectorizer(ngram_range=(1, 2), min_df=1, max_df=1.0, vocabulary=vocabulary)
    # language="multilingual" switches off BERTopic's English cleaning, which deletes every character
    # except letters, digits and spaces. That would turn "vision-language" into "visionlanguage",
    # which is no longer in the word list above, so its count would be zero.
    model = BERTopic(language="multilingual", embedding_model=None, umap_model=umap_model,
                     hdbscan_model=clusterer, vectorizer_model=vectorizer, nr_topics=nr_topics,
                     top_n_words=top_n, calculate_probabilities=False, verbose=False)
    labels, _ = model.fit_transform(texts, embeddings=embeddings)
    if not np.isfinite(model.c_tf_idf_.data).all():
        raise RuntimeError("BERTopic produced infinite word scores: some words in the word list do not "
                           "appear in the topic texts. Check the text cleaning in textprep.py.")
    labels = np.asarray(labels)
    words = {int(t): [w for w, _ in model.get_topic(t)][:top_n] for t in sorted(set(labels)) if t != -1}
    return labels, words


def fit_lda(texts, n_topics, seed=42, top_n=10):
    """Classic LDA baseline. Returns (labels per text, {topic: words}, max topic probability per text)."""
    from sklearn.decomposition import LatentDirichletAllocation

    vectorizer = make_vectorizer(ngram_range=(1, 1), min_df=5, max_df=0.5)
    counts = vectorizer.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=n_topics, learning_method="online",
                                    max_iter=15, random_state=seed)
    doc_topic = lda.fit_transform(counts)
    vocab = np.array(vectorizer.get_feature_names_out())
    words = {t: list(vocab[np.argsort(row)[::-1][:top_n]]) for t, row in enumerate(lda.components_)}
    return doc_topic.argmax(axis=1), words, doc_topic.max(axis=1)


# ---------------------------------------------------------------- assignment
def assign_by_centroid(embeddings, labels, fit_mask):
    """Assign every document to the nearest topic centroid. Returns (topic per doc, similarity)."""
    topics = sorted(t for t in set(labels[fit_mask]) if t >= 0)
    centroids = []
    for t in topics:
        c = embeddings[fit_mask & (labels == t)].mean(axis=0)
        centroids.append(c / max(np.linalg.norm(c), 1e-9))
    sims = embeddings @ np.vstack(centroids).T
    best = sims.argmax(axis=1)
    return np.array(topics)[best], sims.max(axis=1)


def renumber_by_size(topic_ids):
    """Map topic ids to 0..K-1, the biggest topic first. Returns {old: new}."""
    counts = pd.Series(topic_ids).value_counts()
    return {int(old): new for new, old in enumerate(counts.index)}


# ---------------------------------------------------------------- pipeline
def build_topics(df, embeddings, method="bertopic", min_topic_size=25, nr_topics=None,
                 short_percentile=10, short_cap=100, n_lda_topics=None, seed=42):
    """Run one topic model.

    df needs columns id, title, text. Returns (doc_table, topic_table, info).
    doc_table: doc_id, topic_id, topic_raw (-1 = not part of a core cluster), topic_core,
    assign_sim, short_doc, topic_label, words.
    """
    start = time.perf_counter()
    texts = [clean_for_topics(f"{t}. {x}") for t, x in zip(df["title"], df["text"])]
    words_per_doc = np.array([word_count(x) for x in df["text"]])
    short_threshold = float(min(np.percentile(words_per_doc, short_percentile), short_cap))
    short = words_per_doc < short_threshold
    fit_mask = ~short

    if method == "bertopic":
        fit_texts = [t for t, keep in zip(texts, fit_mask) if keep]
        fitted, topic_words = fit_bertopic(fit_texts, embeddings[fit_mask], min_topic_size, nr_topics, seed)
        if not topic_words:
            raise ValueError(f"BERTopic found no topics with min_topic_size={min_topic_size}. "
                             "Try a smaller --min-topic-size.")
        raw = np.full(len(df), -1)
        raw[fit_mask] = fitted
        assigned, sim = assign_by_centroid(embeddings, raw, raw >= 0)
        final = np.where(raw >= 0, raw, assigned)
        core = raw >= 0
    elif method == "lda":
        if not n_lda_topics:
            raise ValueError("n_lda_topics is required for LDA")
        raw, topic_words, _ = fit_lda(texts, n_lda_topics, seed)
        final, core = raw.copy(), np.ones(len(df), dtype=bool)
        sim = np.zeros(len(df))
        for t in set(raw):
            members = raw == t
            c = embeddings[members].mean(axis=0)
            sim[members] = embeddings[members] @ (c / max(np.linalg.norm(c), 1e-9))
    else:
        raise ValueError(f"Unknown method '{method}'")

    mapping = renumber_by_size(final)
    final = np.array([mapping[int(t)] for t in final])
    raw_mapped = np.array([mapping[int(t)] if t in mapping and t >= 0 and core[i] else -1
                           for i, t in enumerate(raw)])
    topic_words = {mapping[int(t)]: w for t, w in topic_words.items() if int(t) in mapping}
    labels = {t: ", ".join(w[:4]) for t, w in topic_words.items()}

    doc_table = pd.DataFrame({
        "doc_id": df["id"].values, "topic_id": final, "topic_raw": raw_mapped, "topic_core": core,
        "assign_sim": np.round(sim, 4), "short_doc": short, "words": words_per_doc,
        "topic_label": [labels.get(int(t), "") for t in final],
    })
    topic_table = describe_topics(doc_table, topic_words, df, embeddings)
    info = {"method": method, "n_topics": len(topic_words), "n_docs": len(df),
            "core_share": float(core.mean()), "short_docs": int(short.sum()),
            "short_threshold_words": short_threshold, "seconds": round(time.perf_counter() - start, 1)}
    return doc_table, topic_table, info, topic_words, texts


def describe_topics(doc_table, topic_words, df, embeddings, n_examples=3):
    """One row per topic: size, top words, and the documents closest to the topic centre."""
    titles = df["title"].values
    rows = []
    for t, words in sorted(topic_words.items()):
        members = np.where((doc_table["topic_id"] == t) & doc_table["topic_core"])[0]
        if len(members) == 0:
            members = np.where(doc_table["topic_id"] == t)[0]
        centre = embeddings[members].mean(axis=0)
        centre /= max(np.linalg.norm(centre), 1e-9)
        order = members[np.argsort(embeddings[members] @ centre)[::-1][:n_examples]]
        rows.append({
            "topic_id": int(t), "label": ", ".join(words[:4]), "words": words,
            "size": int((doc_table["topic_id"] == t).sum()), "core_size": int(len(members)),
            "examples": [str(titles[i]) for i in order],
        })
    return rows


def quality_metrics(topic_words, texts, top_n=10):
    """NPMI coherence and diversity, measured against the same corpus."""
    ref = make_vectorizer(ngram_range=(1, 2), min_df=3, max_df=1.0, binary=True)
    matrix = ref.fit_transform(texts)
    mean, per_topic = npmi_coherence(topic_words, matrix, ref.vocabulary_, top_n)
    return {"npmi": round(mean, 4), "diversity": round(topic_diversity(topic_words, top_n), 4)}, per_topic
