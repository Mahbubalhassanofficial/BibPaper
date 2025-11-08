# ===============================================================
# modules/ai_topics.py
# AI Topic Modeling: LDA (lightweight) + BERTopic (advanced)
# Exports: topics table, topic prevalence, 600-DPI figures
# ===============================================================

import re
from io import BytesIO
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Text / ML ---
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# NLTK (stopwords optional)
import nltk
from nltk.corpus import stopwords

# Try BERTopic (optional)
try:
    from bertopic import BERTopic
    _HAS_BERTOPIC = True
except Exception:
    _HAS_BERTOPIC = False

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "savefig.dpi": 600,
    "figure.dpi": 600
})
sns.set_style("whitegrid")


# ------------------------ Utilities ------------------------

def _ensure_stopwords():
    """Download NLTK stopwords if not available."""
    try:
        _ = stopwords.words("english")
    except LookupError:
        nltk.download("stopwords")


def fig_to_bytes(fig, fmt="png", dpi=600):
    buf = BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    return buf


def build_corpus(df: pd.DataFrame,
                 abstract_col: str = "Abstract",
                 author_kw_col: str = "Author_Keywords",
                 index_kw_col: str = "Index_Keywords") -> list[str]:
    """
    Build a simple corpus by concatenating Abstract + Keywords.
    Clean punctuation and lowercase.
    """
    texts = []
    for _, row in df.iterrows():
        parts = []
        if abstract_col in df.columns and pd.notna(row.get(abstract_col)):
            parts.append(str(row[abstract_col]))
        # merge author + index keywords
        kws = []
        if author_kw_col in df.columns and pd.notna(row.get(author_kw_col)):
            kws.append(str(row[author_kw_col]))
        if index_kw_col in df.columns and pd.notna(row.get(index_kw_col)):
            kws.append(str(row[index_kw_col]))
        if kws:
            parts.append("; ".join(kws))

        txt = " ".join(parts).lower()
        # light cleaning
        txt = re.sub(r"[^a-z0-9\s\-]", " ", txt)
        txt = re.sub(r"\s+", " ", txt).strip()
        texts.append(txt)
    return texts


# ------------------------ LDA Pipeline ------------------------

def lda_fit(texts: list[str], n_topics: int = 8, max_features: int = 5000,
            min_df: int = 2, max_df: float = 0.95, n_top_words: int = 12):
    """
    Fit an LDA model and return (topics_table, doc_topics, vectorizer, lda_model).
    topics_table columns: Topic, Top_Words
    doc_topics: topic index (argmax) per document
    """
    _ensure_stopwords()
    sw = set(stopwords.words("english"))

    vectorizer = CountVectorizer(stop_words=sw,
                                 max_features=max_features,
                                 min_df=min_df, max_df=max_df,
                                 token_pattern=r"(?u)\b\w[\w\-]+\b")
    X = vectorizer.fit_transform(texts)

    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("Insufficient text after vectorization. Check your inputs.")

    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    W = lda.fit_transform(X)  # doc-topic
    H = lda.components_         # topic-word

    terms = np.array(vectorizer.get_feature_names_out())
    rows = []
    for k in range(n_topics):
        top_idx = H[k].argsort()[::-1][:n_top_words]
        top_terms = terms[top_idx]
        rows.append({"Topic": k, "Top_Words": ", ".join(top_terms)})
    topics_table = pd.DataFrame(rows)

    doc_topics = W.argmax(axis=1)  # most likely topic per doc

    return topics_table, doc_topics, vectorizer, lda


def plot_topic_wordbars(topics_table: pd.DataFrame, n_cols: int = 2):
    """
    Build a single figure with small multiples (bar-like labels) listing top words per topic.
    """
    n_topics = topics_table.shape[0]
    n_rows = int(np.ceil(n_topics / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11, 1.8 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for i, (_, row) in enumerate(topics_table.iterrows()):
        ax = axes[i]
        ax.axis("off")
        ax.text(0, 0.7, f"Topic {row['Topic']}", fontsize=12, fontweight="bold", color="#004C97")
        ax.text(0, 0.35, row["Top_Words"], fontsize=10)
    # turn off any extra axes
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.suptitle("LDA Topics — Top Words", fontweight="bold")
    fig.tight_layout()
    return fig


def compute_topic_prevalence_over_time(doc_topics: np.ndarray, years: pd.Series):
    """
    Return a dataframe with columns: Year, Topic, Count
    """
    y = pd.to_numeric(years, errors="coerce")
    valid_idx = y.notna()
    y = y[valid_idx].astype(int)
    t = pd.Series(doc_topics)[valid_idx].astype(int)

    df = pd.DataFrame({"Year": y.values, "Topic": t.values})
    counts = df.groupby(["Year", "Topic"]).size().reset_index(name="Count")
    return counts


def plot_topic_prevalence(counts_df: pd.DataFrame):
    """
    Produce a line plot (one series per topic) showing prevalence per year.
    """
    if counts_df.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "No topic prevalence data.", ha="center", va="center")
        ax.axis("off")
        return fig

    pivoted = counts_df.pivot(index="Year", columns="Topic", values="Count").fillna(0)
    fig, ax = plt.subplots(figsize=(8, 5))
    for topic in pivoted.columns:
        ax.plot(pivoted.index, pivoted[topic], marker="o", label=f"Topic {topic}")
    ax.set_title("Topic Prevalence over Time", fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Count")
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    return fig


# ------------------------ BERTopic Pipeline (Optional) ------------------------

def bertopic_fit(texts: list[str], n_topics: int | None = None):
    """
    Fit BERTopic if installed. Returns (topics_table, doc_topics, model).
    Topics table columns: Topic, Top_Words
    """
    if not _HAS_BERTOPIC:
        raise ImportError("BERTopic not installed. Remove it from UI or install dependencies.")

    topic_model = BERTopic(top_n_words=10, n_episodes=None, verbose=False)
    topics, _ = topic_model.fit_transform(texts)
    # Build topics table
    info = topic_model.get_topic_info()  # columns: Topic, Count, Name
    rows = []
    for k in info["Topic"].tolist():
        if k == -1:
            continue
        words = [w for w, _ in topic_model.get_topic(k)]
        rows.append({"Topic": k, "Top_Words": ", ".join(words)})
    topics_table = pd.DataFrame(rows).sort_values("Topic").reset_index(drop=True)
    doc_topics = np.array(topics)
    return topics_table, doc_topics, topic_model
