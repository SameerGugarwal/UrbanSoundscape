import os
import glob
import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Urban Sound Dashboard", layout="wide")

st.title("🌆 Urban Soundscape Dashboard (Phase 5.6)")
st.write("Spark + ML analytics + charts generated from SONYC-UST predictions")

# -------- Paths (you already created these) ----------
CHART_TOP_LABELS = "results/chart_top_labels.png"
CHART_COUNT_DIST = "results/chart_predicted_count_distribution.png"
CHART_TOP_PAIRS  = "results/chart_top_pairs.png"

CSV_TOP_LABELS = "results/dashboard_top_labels.csv"
CSV_COUNT_DIST = "results/dashboard_predicted_count_distribution.csv"
CSV_TOP_PAIRS  = "results/dashboard_top_pairs.csv"

# -------- Helpers ----------
def safe_read_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

def show_chart(path, title):
    st.subheader(title)
    if os.path.exists(path):
        st.image(path, use_container_width=True)
    else:
        st.warning(f"Chart not found: {path}")

# -------- Layout ----------
col1, col2 = st.columns(2)

with col1:
    show_chart(CHART_TOP_LABELS, "Top Predicted Labels")
with col2:
    show_chart(CHART_COUNT_DIST, "Predicted Label Count Distribution")

show_chart(CHART_TOP_PAIRS, "Top Co-occurring Label Pairs")

st.divider()
st.header("📊 Analytics Tables (from Spark outputs)")

tab1, tab2, tab3 = st.tabs(["Top Labels", "Count Distribution", "Top Pairs"])

with tab1:
    df = safe_read_csv(CSV_TOP_LABELS)
    if df is not None:
        st.dataframe(df)
    else:
        st.warning(f"Missing: {CSV_TOP_LABELS}")

with tab2:
    df = safe_read_csv(CSV_COUNT_DIST)
    if df is not None:
        st.dataframe(df)
    else:
        st.warning(f"Missing: {CSV_COUNT_DIST}")

with tab3:
    df = safe_read_csv(CSV_TOP_PAIRS)
    if df is not None:
        st.dataframe(df)
    else:
        st.warning(f"Missing: {CSV_TOP_PAIRS}")

st.divider()
st.caption("✅ Phase 5.6 Dashboard: Spark analytics + visualization ready for report/demo.")
