# ===============================================================
# 📘 BIBLIOMETRIC DATA HARMONIZATION TOOL
# Developed by: Mahbub Hassan (Chulalongkorn University)
# Phase 1: Upload, Detect, Harmonize, Merge, and Download
# ===============================================================

import streamlit as st
import pandas as pd
import numpy as np

# ---------------------------------------------------------------
# 1. PAGE CONFIGURATION
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Bibliometric Harmonization Studio",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------
# 2. CUSTOM PAGE STYLE (ELEGANT Q1 COLOR THEME)
# ---------------------------------------------------------------
st.markdown("""
<style>
    :root {
        --chula-pink: #E4007C;
        --chula-gold: #B9975B;
        --chula-navy: #004C97;
        --light-bg: #F7F8FA;
    }
    .stApp {
        background-color: var(--light-bg);
        font-family: 'Times New Roman', serif;
    }
    .stButton>button {
        background-color: var(--chula-pink);
        color: white;
        border-radius: 8px;
        padding: 0.5em 1em;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: var(--chula-gold);
        color: black;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# 3. HELPER FUNCTIONS
# ---------------------------------------------------------------

def detect_source(df):
    """Detects if uploaded file is Scopus or WoS based on columns."""
    cols = [c.lower() for c in df.columns]
    if "source title" in cols or "cited by" in cols:
        return "Scopus"
    elif "so" in cols or "py" in cols or "au" in cols:
        return "WoS"
    else:
        return "Unknown"

def harmonize_scopus(df):
    """Rename Scopus fields to unified schema."""
    mapping = {
        "Authors": "Authors",
        "Title": "Title",
        "Year": "Year",
        "Source title": "Source",
        "Cited by": "Citations",
        "DOI": "DOI",
        "Author Keywords": "Author_Keywords",
        "Index Keywords": "Index_Keywords",
        "Abstract": "Abstract",
        "Affiliations": "Affiliations"
    }
    df = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})
    df["Source_Type"] = "Scopus"
    return df

def harmonize_wos(df):
    """Rename WoS fields to unified schema."""
    mapping = {
        "AU": "Authors",
        "TI": "Title",
        "PY": "Year",
        "SO": "Source",
        "TC": "Citations",
        "DI": "DOI",
        "DE": "Author_Keywords",
        "ID": "Index_Keywords",
        "AB": "Abstract",
        "C1": "Affiliations"
    }
    df = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})
    df["Source_Type"] = "WoS"
    return df

def clean_and_merge(df_list):
    """Combine datasets, clean duplicates, standardize fields."""
    combined = pd.concat(df_list, ignore_index=True)
    combined = combined.replace(r'^\s*$', np.nan, regex=True)
    combined.dropna(subset=["Title"], inplace=True)
    combined["Title"] = combined["Title"].str.strip()
    combined["Year"] = pd.to_numeric(combined["Year"], errors="coerce").astype("Int64")

    # Drop duplicates by DOI and Title
    if "DOI" in combined.columns:
        combined.drop_duplicates(subset=["DOI"], keep="first", inplace=True)
    combined.drop_duplicates(subset=["Title"], keep="first", inplace=True)

    combined.reset_index(drop=True, inplace=True)
    return combined

def download_dataframe(df, filename="Harmonized_Bibliometric_Data.csv"):
    """Creates a download button for CSV."""
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Harmonized File",
        data=csv,
        file_name=filename,
        mime="text/csv"
    )

# ---------------------------------------------------------------
# 4. STREAMLIT INTERFACE
# ---------------------------------------------------------------
st.title("📘 Bibliometric Harmonization Studio")
st.write("Upload Scopus and/or Web of Science files to harmonize them into a unified dataset.")

mode = st.radio("Select Upload Mode:", ["Scopus Only", "WoS Only", "Merge (Scopus + WoS)"])

# --- SCOPUS ONLY ---
if mode == "Scopus Only":
    file = st.file_uploader("📄 Upload Scopus CSV/XLSX", type=["csv", "xlsx"])
    if file:
        try:
            df = pd.read_csv(file)
        except:
            df = pd.read_excel(file)

        st.success(f"✅ File uploaded successfully with {df.shape[0]} records.")
        src = detect_source(df)
        st.info(f"Detected Source: **{src}**")

        df_h = harmonize_scopus(df)
        # --- Fix duplicate column names (important for Streamlit) ---
df_h = df_h.loc[:, ~df_h.columns.duplicated()].copy()
        st.dataframe(df_h.head())
        download_dataframe(df_h)

# --- WOS ONLY ---
elif mode == "WoS Only":
    file = st.file_uploader("📄 Upload Web of Science CSV/XLSX", type=["csv", "xlsx"])
    if file:
        try:
            df = pd.read_csv(file)
        except:
            df = pd.read_excel(file)

        st.success(f"✅ File uploaded successfully with {df.shape[0]} records.")
        src = detect_source(df)
        st.info(f"Detected Source: **{src}**")

        df_h = harmonize_wos(df)
        st.dataframe(df_h.head())
        download_dataframe(df_h)

# --- MERGE MODE ---
else:
    st.subheader("🔄 Merge Scopus and Web of Science Files")
    col1, col2 = st.columns(2)
    with col1:
        scopus_file = st.file_uploader("📘 Upload Scopus File", type=["csv", "xlsx"], key="scopus")
    with col2:
        wos_file = st.file_uploader("📙 Upload Web of Science File", type=["csv", "xlsx"], key="wos")

    if st.button("⚙️ Harmonize & Merge"):
        if not scopus_file or not wos_file:
            st.error("Please upload both files first.")
        else:
            try:
                scopus_df = pd.read_csv(scopus_file)
            except:
                scopus_df = pd.read_excel(scopus_file)
            try:
                wos_df = pd.read_csv(wos_file)
            except:
                wos_df = pd.read_excel(wos_file)

            scopus_h = harmonize_scopus(scopus_df)
            wos_h = harmonize_wos(wos_df)
            merged_df = clean_and_merge([scopus_h, wos_h])

            st.success(f"✅ Harmonization complete: {merged_df.shape[0]} total records.")
            st.dataframe(merged_df.head(10))
            download_dataframe(merged_df)

# ===============================================================
# PHASE 2 — ANALYSIS & FIGURES
# ===============================================================
st.markdown("## 📈 Analysis & Figures (Phase 2)")

st.info("Upload a **harmonized CSV** exported from Phase 1 to generate figures.")

analysis_file = st.file_uploader("Upload Harmonized CSV (from Phase 1)", type=["csv"], key="analysis_csv")

if analysis_file:
    df_ana = pd.read_csv(analysis_file)

    # lazy import to keep app start fast
    from modules.visualizer import (
        plot_publication_trend, plot_top_keywords, plot_coauthorship_network, figure_bytes
    )

    st.markdown("### Choose Analysis")
    choice = st.selectbox(
        "Select a figure to generate",
        [
            "Publication Trend",
            "Top Keywords",
            "Co-authorship Network"
        ]
    )

    if choice == "Publication Trend":
        fig = plot_publication_trend(df_ana, year_col="Year")

    elif choice == "Top Keywords":
        colA, colB = st.columns(2)
        with colA:
            top_n = st.number_input("Top N keywords", min_value=5, max_value=100, value=20, step=1)
        fig = plot_top_keywords(df_ana, top_n=top_n)

    else:  # Co-authorship Network
        colA, colB = st.columns(2)
        with colA:
            max_nodes = st.number_input("Max nodes", min_value=30, max_value=1000, value=150, step=10)
        with colB:
            min_degree = st.number_input("Minimum degree", min_value=1, max_value=10, value=1, step=1)
        fig = plot_coauthorship_network(df_ana, max_nodes=int(max_nodes), min_degree=int(min_degree))

    st.pyplot(fig, clear_figure=False)

    # ---- Download buttons (PNG / JPEG / PDF — all 600 DPI) ----
    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button("Download PNG (600 DPI)", data=figure_bytes(fig, fmt="png"),
                           file_name="figure.png", mime="image/png")
    with col2:
        st.download_button("Download JPEG (600 DPI)", data=figure_bytes(fig, fmt="jpeg"),
                           file_name="figure.jpeg", mime="image/jpeg")
    with col3:
        st.download_button("Download PDF (vector)", data=figure_bytes(fig, fmt="pdf"),
                           file_name="figure.pdf", mime="application/pdf")

# ===============================================================
# PHASE 3 — Extended Analysis and Report
# ===============================================================
st.markdown("## 🌍 Extended Analysis (Phase 3)")

analysis_file2 = st.file_uploader("Upload Harmonized CSV for Extended Analysis", type=["csv"], key="analysis_csv2")
if analysis_file2:
    df_ext = pd.read_csv(analysis_file2)
    from modules.performance_plus_geo import (
        plot_citation_trend, top_cited_table, plot_country_productivity, plot_world_map, fig_to_bytes
    )
    from modules.report_generator import create_report

    choice2 = st.selectbox(
        "Select Extended Analysis",
        ["Citation Trend", "Top Cited Papers", "Country Productivity", "World Map", "Generate Full Report"]
    )

    if choice2 == "Citation Trend":
        fig = plot_citation_trend(df_ext)
        st.pyplot(fig)
        st.download_button("Download PNG", data=fig_to_bytes(fig), file_name="citations.png")

    elif choice2 == "Top Cited Papers":
        table = top_cited_table(df_ext, n=15)
        st.dataframe(table)

    elif choice2 == "Country Productivity":
        fig = plot_country_productivity(df_ext)
        st.pyplot(fig)
        st.download_button("Download PNG", data=fig_to_bytes(fig), file_name="countries.png")

    elif choice2 == "World Map":
        fig = plot_world_map(df_ext)
        st.plotly_chart(fig, use_container_width=True)

    else:  # Generate Full Report
        st.info("Generating multi-page PDF summary…")
        figs = {
            "Citation Trend": plot_citation_trend(df_ext),
            "Country Productivity": plot_country_productivity(df_ext)
        }
        pdf_path = create_report(figs)
        with open(pdf_path, "rb") as f:
            st.download_button("⬇️ Download Bibliometric Report", data=f, file_name="Bibliometric_Report.pdf")
# ===============================================================
# PHASE 4 — Thematic & Conceptual Structure
# ===============================================================
st.markdown("## 🧠 Thematic & Conceptual Structure (Phase 4)")

analysis_file3 = st.file_uploader("Upload Harmonized CSV for Thematic Analysis", type=["csv"], key="analysis_csv3")
if analysis_file3:
    df_them = pd.read_csv(analysis_file3)
    from modules.thematic_structure import (
        plot_keyword_network, plot_thematic_map, plot_conceptual_structure, fig_to_bytes
    )

    choice3 = st.selectbox(
        "Select Analysis",
        ["Keyword Co-occurrence Network", "Thematic Map (Density × Centrality)", "Conceptual Structure (PCA)"]
    )

    if choice3 == "Keyword Co-occurrence Network":
        top_n = st.slider("Number of Top Keywords", 10, 80, 30, step=5)
        fig = plot_keyword_network(df_them, top_n=top_n)

    elif choice3 == "Thematic Map (Density × Centrality)":
        fig = plot_thematic_map(df_them)

    else:
        top_n = st.slider("Top Keywords for PCA", 10, 100, 40, step=10)
        fig = plot_conceptual_structure(df_them, top_n=top_n)

    st.pyplot(fig, clear_figure=False)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button("Download PNG (600 DPI)", data=fig_to_bytes(fig, "png"),
                           file_name="thematic.png", mime="image/png")
    with col2:
        st.download_button("Download JPEG (600 DPI)", data=fig_to_bytes(fig, "jpeg"),
                           file_name="thematic.jpeg", mime="image/jpeg")
    with col3:
        st.download_button("Download PDF (Vector)", data=fig_to_bytes(fig, "pdf"),
                           file_name="thematic.pdf", mime="application/pdf")
# ===============================================================
# PHASE 5 — Topic Evolution & Interactive Dashboard
# ===============================================================
st.markdown("## 🔭 Topic Evolution & Interactive Dashboard (Phase 5)")

analysis_file4 = st.file_uploader("Upload Harmonized CSV for Topic Evolution", type=["csv"], key="analysis_csv4")
if analysis_file4:
    df_topic = pd.read_csv(analysis_file4)
    from modules.topic_evolution import plot_keyword_trend, generate_dashboard, fig_to_bytes

    tab1, tab2 = st.tabs(["📈 Topic Evolution", "📊 Dashboard Summary"])

    with tab1:
        st.write("Analyze how top research topics evolved over time.")
        top_n = st.slider("Top Keywords", 5, 30, 10, step=1)
        fig = plot_keyword_trend(df_topic, top_n=top_n)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.write("Interactive overview of performance metrics.")
        f1, f2 = generate_dashboard(df_topic)
        st.plotly_chart(f1, use_container_width=True)
        st.plotly_chart(f2, use_container_width=True)

    # Download PNG export
    st.download_button("⬇️ Download Topic Evolution PNG (600 DPI)",
                       data=fig_to_bytes(plt.figure(figsize=(8,5))), file_name="topic_evolution.png")
# ===============================================================
# PHASE 6 — AI Topic Modeling (LDA / BERTopic) + Report Hooks
# ===============================================================
st.markdown("## 🧪 AI Topic Modeling (Phase 6)")

analysis_file5 = st.file_uploader("Upload Harmonized CSV for Topic Modeling", type=["csv"], key="analysis_csv5")
if analysis_file5:
    df_tm = pd.read_csv(analysis_file5)

    from modules.ai_topics import (
        build_corpus,
        lda_fit, plot_topic_wordbars, compute_topic_prevalence_over_time, plot_topic_prevalence, fig_to_bytes,
        bertopic_fit
    )
    from modules.report_generator import create_report, plotly_fig_to_png_bytes

    # --- Parameters & Engine selection ---
    engine = st.radio("Select Topic Model", ["LDA (lightweight)", "BERTopic (advanced)"])
    n_topics = st.slider("Number of Topics (LDA only)", min_value=4, max_value=20, value=8, step=1)
    n_top_words = st.slider("Top words per topic (LDA)", min_value=6, max_value=20, value=12, step=1)

    # --- Build corpus from Abstract + Keywords ---
    st.info("Corpus = Abstract + Author/Index Keywords (cleaned).")
    texts = build_corpus(df_tm)

    # --- Run model ---
    topics_table = None
    doc_topics = None
    topics_fig = None
    prevalence_fig = None

    try:
        if engine.startswith("BERTopic"):
            topics_table, doc_topics, _ = bertopic_fit(texts)
            st.success("BERTopic trained successfully.")
            # Simple topics table preview
            st.dataframe(topics_table.head(15))
            # Prevalence (using argmax topic per doc)
            counts_df = compute_topic_prevalence_over_time(doc_topics, df_tm["Year"])
            prevalence_fig = plot_topic_prevalence(counts_df)
            st.pyplot(prevalence_fig)

            # Download topics table
            st.download_button(
                "⬇️ Download Topics Table (CSV)",
                data=topics_table.to_csv(index=False).encode("utf-8"),
                file_name="topics_table_bertopic.csv",
                mime="text/csv"
            )

        else:
            topics_table, doc_topics, _, _ = lda_fit(texts, n_topics=n_topics, n_top_words=n_top_words)
            st.success("LDA trained successfully.")
            st.dataframe(topics_table.head(15))
            topics_fig = plot_topic_wordbars(topics_table)
            st.pyplot(topics_fig)

            counts_df = compute_topic_prevalence_over_time(doc_topics, df_tm["Year"])
            prevalence_fig = plot_topic_prevalence(counts_df)
            st.pyplot(prevalence_fig)

            st.download_button(
                "⬇️ Download Topics Table (CSV)",
                data=topics_table.to_csv(index=False).encode("utf-8"),
                file_name="topics_table_lda.csv",
                mime="text/csv"
            )

        # --- Figure downloads (PNG/JPEG/PDF) ---
        if topics_fig is not None:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.download_button("Download Topics PNG (600 DPI)",
                                   data=fig_to_bytes(topics_fig, "png"),
                                   file_name="topics.png", mime="image/png")
            with c2:
                st.download_button("Download Topics JPEG (600 DPI)",
                                   data=fig_to_bytes(topics_fig, "jpeg"),
                                   file_name="topics.jpeg", mime="image/jpeg")
            with c3:
                st.download_button("Download Topics PDF (Vector)",
                                   data=fig_to_bytes(topics_fig, "pdf"),
                                   file_name="topics.pdf", mime="application/pdf")

        if prevalence_fig is not None:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.download_button("Download Prevalence PNG (600 DPI)",
                                   data=fig_to_bytes(prevalence_fig, "png"),
                                   file_name="prevalence.png", mime="image/png")
            with c2:
                st.download_button("Download Prevalence JPEG (600 DPI)",
                                   data=fig_to_bytes(prevalence_fig, "jpeg"),
                                   file_name="prevalence.jpeg", mime="image/jpeg")
            with c3:
                st.download_button("Download Prevalence PDF (Vector)",
                                   data=fig_to_bytes(prevalence_fig, "pdf"),
                                   file_name="prevalence.pdf", mime="application/pdf")

        # --- Optional: Add both topic figures to a multi-page report ---
        if st.checkbox("Add topic figures to a multi-page PDF report"):
            figs = {}
            if topics_fig is not None:
                figs["Topic Words (LDA)"] = topics_fig
            if prevalence_fig is not None:
                figs["Topic Prevalence Over Time"] = prevalence_fig

            pdf_path = "outputs/reports/Bibliometric_Topic_Report.pdf"
            path_out = create_report(figs, output_path=pdf_path)
            with open(path_out, "rb") as f:
                st.download_button("⬇️ Download Topic Report (PDF)", data=f, file_name="Bibliometric_Topic_Report.pdf")

    except Exception as e:
        st.error(f"Modeling error: {e}")
# ===============================================================
# PHASE 7 — Session State + One-Click Full Report + Theme
# ===============================================================
st.markdown("## 🧾 One-Click Full Report & Theme Editor (Phase 7)")

from modules.session_manager import store_dataframe, get_dataframe
from modules.full_report import generate_full_report

# Theme editor
st.sidebar.subheader("🎨 Theme Editor")
theme = st.sidebar.selectbox("Choose Color Theme", ["Chulalongkorn", "IEEE Blue", "Nature Green"])
font = st.sidebar.selectbox("Font Family", ["Times New Roman", "Serif", "Helvetica"])

# Optional style configuration (just stored)
st.session_state["theme"] = theme
st.session_state["font"] = font

# Session loader
st.info("If you have already uploaded a harmonized file in previous phases, click below to load it automatically.")
if st.button("🔄 Load Harmonized Dataset from Session"):
    df_session = get_dataframe()
    if df_session is not None:
        st.success(f"Loaded dataset with {df_session.shape[0]} records from session.")
    else:
        st.error("No dataset found in session. Please upload in Phase 1 first.")

# Manual upload fallback
uploaded_file = st.file_uploader("Upload Harmonized CSV (optional, if not using session)", type=["csv"], key="phase7")
if uploaded_file:
    df_session = pd.read_csv(uploaded_file)
    store_dataframe(df_session)

# Report generation section
if "harmonized_df" in st.session_state:
    df_final = get_dataframe()
    st.success(f"Dataset active: {df_final.shape[0]} records.")
    st.markdown("Generate a **complete bibliometric report** combining figures from multiple phases.")
    if st.button("📘 Generate Full Report"):
        from modules.visualizer import plot_publication_trend, plot_top_keywords
        from modules.performance_plus_geo import plot_country_productivity, plot_citation_trend

        figs = {
            "Publication Trend": plot_publication_trend(df_final),
            "Top Keywords": plot_top_keywords(df_final, top_n=20),
            "Citation Trend": plot_citation_trend(df_final),
            "Country Productivity": plot_country_productivity(df_final)
        }

        pdf_path = generate_full_report(df_final, figs)
        with open(pdf_path, "rb") as f:
            st.download_button("⬇️ Download Full Report (PDF)", data=f, file_name="Bibliometric_Full_Report.pdf")
else:
    st.warning("⚠️ No harmonized data loaded yet. Please complete Phase 1 first.")

st.markdown("---")
st.caption("Developed by **Mahbub Hassan**, Chulalongkorn University © 2025")

