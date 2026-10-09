"""Main entry point for the multi-page purchase-intent dashboard."""

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(
    page_title="Commerce intelligence",
    page_icon=":material/query_stats:",
    layout="wide",
    initial_sidebar_state="auto",
)

st.logo(str(Path(__file__).resolve().parent / "assets" / "mark.svg"))

pages = [
    st.Page(PROJECT_ROOT / "dashboard/pages/overview.py", title="Overview", icon=":material/dashboard:", default=True),
    st.Page(PROJECT_ROOT / "dashboard/pages/scoring.py", title="Live scoring", icon=":material/bolt:"),
    st.Page(PROJECT_ROOT / "dashboard/pages/analytics.py", title="Funnel & cohorts", icon=":material/analytics:"),
    st.Page(PROJECT_ROOT / "dashboard/pages/models.py", title="Model comparison", icon=":material/monitoring:"),
    st.Page(PROJECT_ROOT / "dashboard/pages/explainability.py", title="Explainability", icon=":material/psychology:"),
    st.Page(PROJECT_ROOT / "dashboard/pages/system.py", title="System health", icon=":material/settings:"),
]

with st.sidebar:
    st.markdown("### Commerce intelligence")
    st.caption("Purchase intent · XAI workspace")
    st.badge("Late-session model", color="blue")
    st.caption("Page values are included. Score when the full session signal is available.")

navigation = st.navigation(pages, position="sidebar")
navigation.run()
