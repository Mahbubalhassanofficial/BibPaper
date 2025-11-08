# ===============================================================
# modules/thematic_structure.py
# Thematic & Conceptual Structure Analysis
# ===============================================================

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from io import BytesIO

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "savefig.dpi": 600,
    "figure.dpi": 600
})
sns.set_style("whitegrid")


# ----------- Helpers -----------
def _clean_keywords(series):
    if series is None:
        return []
    s = (series.dropna()
               .astype(str)
               .str.replace(r"\s*[,;]\s*", ";", regex=True)
               .str.split(";")
               .explode()
               .str.strip()
               .str.lower())
    s = s[s != ""]
    return s.tolist()


def fig_to_bytes(fig, fmt="png", dpi=600):
    buf = BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    return buf


# ---------------------- 1. Keyword Co-occurrence Network ----------------------

def plot_keyword_network(df, author_kw_col="Author_Keywords",
                         index_kw_col="Index_Keywords", top_n=30):
    """Network of top-N co-occurring keywords."""
    kws = _clean_keywords(df.get(author_kw_col)) + _clean_keywords(df.get(index_kw_col))
    if not kws:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No keywords available", ha="center", va="center")
        ax.axis("off")
        return fig

    # co-occurrence matrix using CountVectorizer
    docs = df[author_kw_col].fillna("") + ";" + df.get(index_kw_col, "")
    vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w[\w\-]+\b", lowercase=True)
    X = vectorizer.fit_transform(docs)
    vocab = np.array(vectorizer.get_feature_names_out())

    # co-occurrence
    co_matrix = (X.T @ X).toarray()
    np.fill_diagonal(co_matrix, 0)

    # build graph
    G = nx.Graph()
    for i in range(len(vocab)):
        for j in range(i + 1, len(vocab)):
            if co_matrix[i, j] > 0:
                G.add_edge(vocab[i], vocab[j], weight=int(co_matrix[i, j]))

    # keep top edges by weight
    edges_sorted = sorted(G.edges(data=True), key=lambda x: x[2]["weight"], reverse=True)[: top_n * 3]
    G_sub = nx.Graph()
    for u, v, w in edges_sorted:
        G_sub.add_edge(u, v, weight=w["weight"])

    pos = nx.spring_layout(G_sub, k=0.5, seed=42)
    weights = np.array([w["weight"] for _, _, w in G_sub.edges(data=True)])
    weights = 0.5 + 0.8 * np.log1p(weights)
    fig, ax = plt.subplots(figsize=(8, 8))
    nx.draw_networkx_edges(G_sub, pos, width=weights, edge_color="#C0C0C0", alpha=0.7)
    nx.draw_networkx_nodes(G_sub, pos, node_color="#E4007C", alpha=0.85,
                           node_size=[max(40, 10 * np.log1p(G_sub.degree(n))) for n in G_sub.nodes()])
    lbls = dict(sorted(G_sub.degree(), key=lambda x: x[1], reverse=True)[: top_n])
    nx.draw_networkx_labels(G_sub, pos, labels={n: n for n in lbls}, font_size=8, font_color="black")
    ax.set_title(f"Keyword Co-occurrence Network (Top {top_n})", fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    return fig


# ---------------------- 2. Thematic Map (Density × Centrality) ----------------------

def plot_thematic_map(df, author_kw_col="Author_Keywords", index_kw_col="Index_Keywords"):
    """Simulated density–centrality map (keyword clusters)."""
    kws = _clean_keywords(df.get(author_kw_col)) + _clean_keywords(df.get(index_kw_col))
    if not kws:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No keywords available", ha="center", va="center")
        ax.axis("off")
        return fig

    # fake density and centrality for demonstration (replace with cluster stats later)
    keywords = pd.Series(kws).value_counts().head(40)
    np.random.seed(42)
    density = np.random.uniform(0, 1, len(keywords))
    centrality = np.random.uniform(0, 1, len(keywords))
    data = pd.DataFrame({"Keyword": keywords.index, "Density": density, "Centrality": centrality,
                         "Frequency": keywords.values})

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(data=data, x="Centrality", y="Density", size="Frequency", hue="Frequency",
                    sizes=(40, 400), palette="plasma", legend=False, ax=ax)
    for _, r in data.iterrows():
        ax.text(r["Centrality"], r["Density"], r["Keyword"], fontsize=7)
    ax.set_title("Thematic Map (Density × Centrality)", fontweight="bold")
    ax.set_xlabel("Centrality → Relevance")
    ax.set_ylabel("Density → Development")
    fig.tight_layout()
    return fig


# ---------------------- 3. Conceptual Structure (PCA / MDS) ----------------------

def plot_conceptual_structure(df, author_kw_col="Author_Keywords", top_n=40):
    """Simple PCA 2-D embedding of keyword co-occurrence."""
    kws = _clean_keywords(df.get(author_kw_col))
    if not kws:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No keywords found", ha="center", va="center")
        ax.axis("off")
        return fig

    # build term–document matrix
    docs = df[author_kw_col].fillna("")
    vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w[\w\-]+\b", lowercase=True)
    X = vectorizer.fit_transform(docs)
    vocab = np.array(vectorizer.get_feature_names_out())
    if len(vocab) > top_n:
        vocab = vocab[:top_n]
        X = X[:, :top_n]

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X.toarray().T)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(coords[:, 0], coords[:, 1], s=40, c="#004C97")
    for i, word in enumerate(vocab):
        ax.text(coords[i, 0], coords[i, 1], word, fontsize=8)
    ax.set_title("Conceptual Structure (PCA Keyword Clusters)", fontweight="bold")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    fig.tight_layout()
    return fig
