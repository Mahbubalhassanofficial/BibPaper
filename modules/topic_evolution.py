# ===============================================================
# modules/topic_evolution.py
# Topic Evolution & Interactive Dashboard
# ===============================================================

import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from collections import Counter
import re

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "savefig.dpi": 600,
    "figure.dpi": 600
})
sns.set_style("whitegrid")

# ----------------------------- Helpers -----------------------------

def _extract_keywords(text):
    """Split and normalize keyword strings."""
    if pd.isna(text):
        return []
    parts = re.split(r"[;,]", str(text))
    return [p.strip().lower() for p in parts if p.strip()]


def fig_to_bytes(fig, fmt="png", dpi=600):
    buf = BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    return buf


# ---------------------- 1) Keyword Trend Over Time ----------------------

def plot_keyword_trend(df, year_col="Year", kw_col="Author_Keywords", top_n=10):
    """
    Plot keyword frequency trends over time.
    """
    df = df.copy()
    df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
    df = df.dropna(subset=[year_col, kw_col])

    # explode keywords by year
    data = []
    for _, row in df.iterrows():
        year = int(row[year_col])
        for kw in _extract_keywords(row[kw_col]):
            data.append((year, kw))
    trend_df = pd.DataFrame(data, columns=["Year", "Keyword"])

    # filter top keywords overall
    top_kw = [kw for kw, _ in Counter(trend_df["Keyword"]).most_common(top_n)]
    trend_df = trend_df[trend_df["Keyword"].isin(top_kw)]

    fig = px.line(
        trend_df.groupby(["Year", "Keyword"]).size().reset_index(name="Count"),
        x="Year", y="Count", color="Keyword",
        markers=True,
        title=f"Keyword Evolution Over Time (Top {top_n})",
        color_discrete_sequence=px.colors.qualitative.Vivid
    )
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", font_family="Times New Roman")
    return fig


# ---------------------- 2) Interactive Summary Dashboard ----------------------

def generate_dashboard(df):
    """
    Returns multiple Plotly figures summarizing key metrics.
    """
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    yearly = df.groupby("Year").size().reset_index(name="Publications")
    cite = df.groupby("Year")["Citations"].sum().reset_index(name="Citations")

    f1 = px.bar(yearly, x="Year", y="Publications", title="Publications per Year",
                color_discrete_sequence=["#004C97"])
    f2 = px.line(cite, x="Year", y="Citations", markers=True, title="Citations per Year",
                 color_discrete_sequence=["#E4007C"])
    f1.update_layout(font_family="Times New Roman", plot_bgcolor="white")
    f2.update_layout(font_family="Times New Roman", plot_bgcolor="white")
    return f1, f2
