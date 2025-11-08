# ===============================================================
# modules/visualizer.py
# High-quality, 600-DPI figures for bibliometric analyses
# ===============================================================

from io import BytesIO
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

# ---- Global Matplotlib styling for Q1 aesthetics ----
plt.rcParams.update({
    "figure.dpi": 600,
    "savefig.dpi": 600,
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "axes.grid": False
})
sns.set_style("whitegrid")


# ----------------------------- Utilities -----------------------------

def _clean_keywords(series: pd.Series) -> pd.Series:
    """Split and tidy author keywords or index keywords."""
    if series is None:
        return pd.Series(dtype=str)
    s = (series.dropna()
               .astype(str)
               .str.replace(r"\s*;\s*|\s*,\s*", ";", regex=True)  # unify delimiters to ;
               .str.split(";")
               .explode()
               .str.strip()
               .str.lower())
    s = s[s != ""]
    return s


def _fig_to_bytes(fig, fmt="png", dpi=600):
    """Convert a Matplotlib figure to bytes for Streamlit download."""
    buf = BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    return buf


# ---------------------- 1) Publication Trend -------------------------

def plot_publication_trend(df: pd.DataFrame, year_col: str = "Year"):
    """
    Annual publication count line plot.
    Assumes df[year_col] is numeric or convertible to numeric.
    """
    tmp = df.copy()
    tmp[year_col] = pd.to_numeric(tmp[year_col], errors="coerce")
    trend = (tmp.dropna(subset=[year_col])
                .groupby(year_col)
                .size()
                .reset_index(name="Publications")
                .sort_values(year_col))

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=trend, x=year_col, y="Publications", marker="o", ax=ax)
    ax.set_title("Annual Publication Trend", fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Publications")
    ax.margins(x=0.02, y=0.10)
    fig.tight_layout()
    return fig


# ---------------------- 2) Keyword Frequency -------------------------

def plot_top_keywords(df: pd.DataFrame,
                      author_kw_col: str = "Author_Keywords",
                      index_kw_col: str = "Index_Keywords",
                      top_n: int = 20):
    """
    Top-N keyword frequency bar chart (author + index keywords combined).
    """
    a = _clean_keywords(df.get(author_kw_col))
    b = _clean_keywords(df.get(index_kw_col))
    keywords = pd.concat([a, b], ignore_index=True)

    if keywords.empty:
        # create empty placeholder plot
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.text(0.5, 0.5, "No keywords found.", ha="center", va="center")
        ax.axis("off")
        return fig

    freq = (keywords.value_counts()
                     .head(top_n)
                     .sort_values(ascending=True))  # for horizontal bars

    fig, ax = plt.subplots(figsize=(7.5, 6))
    sns.barplot(x=freq.values, y=freq.index, ax=ax, palette="husl")
    ax.set_title(f"Top {min(top_n, len(freq))} Keywords", fontweight="bold")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("")
    fig.tight_layout()
    return fig


# ---------------------- 3) Co-authorship Network ---------------------

def plot_coauthorship_network(df: pd.DataFrame,
                              authors_col: str = "Authors",
                              max_nodes: int = 150,
                              min_degree: int = 1):
    """
    Builds an undirected co-authorship graph from the Authors column.
    - Splits authors by comma/semicolon.
    - Limits to max_nodes (highest-degree) for clarity.
    - Filters nodes by degree >= min_degree.
    """
    G = nx.Graph()

    # robust split for authors (handles "A; B" or "A, B")
    for raw in df.get(authors_col, pd.Series(dtype=str)).dropna().astype(str):
        parts = re.split(r"\s*[,;]\s*", raw)
        parts = [p for p in (p.strip() for p in parts) if p]
        # add edges for all coauthor pairs
        for i in range(len(parts)):
            for j in range(i + 1, len(parts)):
                a, b = parts[i], parts[j]
                if a != b:
                    if G.has_edge(a, b):
                        G[a][b]["weight"] += 1
                    else:
                        G.add_edge(a, b, weight=1)

    if G.number_of_nodes() == 0:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.text(0.5, 0.5, "No co-authorship data available.", ha="center", va="center")
        ax.axis("off")
        return fig

    # filter by degree
    if min_degree > 1:
        keep = [n for n, d in G.degree() if d >= min_degree]
        G = G.subgraph(keep).copy()

    # restrict to top-degree nodes for readability
    if G.number_of_nodes() > max_nodes:
        deg = dict(G.degree())
        top = sorted(deg, key=deg.get, reverse=True)[:max_nodes]
        G = G.subgraph(top).copy()

    # layout and drawing
    pos = nx.spring_layout(G, k=0.35, seed=42)  # stable layout
    degrees = dict(G.degree())
    node_sizes = [max(30, degrees[n] * 20) for n in G.nodes()]
    edge_weights = [0.5 + 0.8 * np.log1p(w["weight"]) for _, _, w in G.edges(data=True)]

    fig, ax = plt.subplots(figsize=(8, 8))
    nx.draw_networkx_edges(G, pos, width=edge_weights, edge_color="#A0A0A0", alpha=0.6, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="#004C97", alpha=0.85, ax=ax)
    # Labels only for highest-degree subset (readability)
    highest = sorted(degrees, key=degrees.get, reverse=True)[: max(10, int(0.05 * G.number_of_nodes()))]
    lbls = {n: n for n in highest}
    nx.draw_networkx_labels(G, pos, labels=lbls, font_size=8, font_color="black", ax=ax)

    ax.set_title("Co-authorship Network", fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    return fig


# ---------------------- Export helpers for Streamlit ------------------

def figure_bytes(fig, fmt="png", dpi=600):
    """Return figure as bytes buffer for Streamlit download buttons."""
    return _fig_to_bytes(fig, fmt=fmt, dpi=dpi)
