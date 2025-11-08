# ===============================================================
# modules/session_manager.py
# Handles session state for multi-phase data persistence
# ===============================================================

import streamlit as st
import pandas as pd

def store_dataframe(df: pd.DataFrame, key: str = "harmonized_df"):
    """Stores a DataFrame persistently in session state."""
    st.session_state[key] = df

def get_dataframe(key: str = "harmonized_df") -> pd.DataFrame | None:
    """Retrieves stored DataFrame from session state."""
    return st.session_state.get(key, None)
