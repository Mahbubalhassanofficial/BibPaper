# ===============================================================
# modules/performance_plus_geo.py
# Extended performance & geographical analysis
# ===============================================================

import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "savefig.dpi": 600,
    "figure.dpi": 600
})

# ---------- Helper for export ----------
def fig_to_bytes(fig, fmt="png", dpi=600):
    buf = BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    return buf


# ---------- Citation Trend ----------
def plot_citation_trend(df):
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    trend = (df.groupby("Year")["Citations"]
               .sum()
               .reset_index()
               .sort_values("Year"))
    fig, ax = plt.subplots(figsize=(8,5))
    sns.lineplot(data=trend, x="Year", y="Citations", marker="o", ax=ax)
    ax.set_title("Total Citations per Year", fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Citations")
    fig.tight_layout()
    return fig


# ---------- Top Cited Documents ----------
def top_cited_table(df, n=10):
    if "Citations" not in df.columns:
        return pd.DataFrame()
    df_sorted = (df[["Title","Authors","Year","Citations","Source"]]
                 .dropna(subset=["Citations"])
                 .sort_values("Citations", ascending=False)
                 .head(n))
    return df_sorted


# ---------- Country Extraction ----------
def extract_country(text):
    if pd.isna(text):
        return None
    # Naïve country extraction (last comma element)
    parts = str(text).split(",")
    return parts[-1].strip().title() if len(parts) > 1 else None


# ---------- Country Productivity (Barplot) ----------
def plot_country_productivity(df):
    df["Country"] = df["Affiliations"].apply(extract_country)
    counts = (df["Country"].dropna()
              .value_counts()
              .head(20)
              .sort_values(ascending=True))
    fig, ax = plt.subplots(figsize=(8,6))
    sns.barplot(x=counts.values, y=counts.index, palette="deep", ax=ax)
    ax.set_title("Top 20 Productive Countries", fontweight="bold")
    ax.set_xlabel("Publications")
    ax.set_ylabel("")
    fig.tight_layout()
    return fig


# ---------- World Map ----------
def plot_world_map(df):
    df["Country"] = df["Affiliations"].apply(extract_country)
    country_df = (df["Country"].value_counts().reset_index())
    country_df.columns = ["Country","Publications"]
    fig = px.choropleth(
        country_df,
        locations="Country",
        locationmode="country names",
        color="Publications",
        color_continuous_scale="YlGnBu",
        title="Global Publication Distribution"
    )
    return fig
