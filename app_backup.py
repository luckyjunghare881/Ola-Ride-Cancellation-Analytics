"""
OLA Ride Cancellation Analytics — Streamlit Dashboard
Main entry point. Multi-page app with sidebar navigation.
"""

import streamlit as st
import pandas as pd
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.data_processing import load_csv, preprocess, compute_kpis
from src.generate_data import generate_dataset, save_dataset

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OLA Cancellation Analytics",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1B9E3E;
        text-align: center;
        padding: 0.5rem 0 1rem 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #888;
        text-align: center;
        margin-top: -1rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #1B9E3E;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1B9E3E;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #aaa;
        margin-top: 0.3rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0E1117 0%, #1a1a2e 100%);
    }
</style>
""", unsafe_allow_html=True)


# ── Session state init ─────────────────────────────────────────────────────
def init_session_state():
    defaults = {
        "df": None,
        "df_clean": None,
        "kpis": None,
        "cleaning_report": None,
        "data_loaded": False,
        "ml_result": None,
        "user_role": "Analyst",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚗 OLA Analytics")
    st.markdown("---")

    # User role
    st.session_state.user_role = st.radio(
        "👤 User Role",
        ["Analyst", "Business User"],
        help="Analyst: Full access. Business User: Pre-loaded demo data with recommendations.",
    )

    st.markdown("---")

    page = st.radio(
        "📊 Navigation",
        [
            "🏠 Home & Upload",
            "📈 EDA Insights",
            "🗃️ SQL Queries",
            "🤖 ML Predictions",
            "💡 Recommendations",
        ],
        index=0,
    )

    st.markdown("---")

    # Quick data load
    if not st.session_state.data_loaded:
        if st.button("🎲 Load Demo Data", use_container_width=True):
            with st.spinner("Generating demo dataset..."):
                demo_df = generate_dataset(num_rows=12000)
                demo_df_clean, report = preprocess(demo_df)
                kpis = compute_kpis(demo_df_clean)

                st.session_state.df = demo_df
                st.session_state.df_clean = demo_df_clean
                st.session_state.kpis = kpis
                st.session_state.cleaning_report = report
                st.session_state.data_loaded = True
            st.success("✅ Demo data loaded!")
            st.rerun()
    else:
        st.success(f"✅ Data loaded: {len(st.session_state.df_clean):,} rows")
        if st.button("🔄 Reset Data", use_container_width=True):
            for key in ["df", "df_clean", "kpis", "cleaning_report", "ml_result"]:
                st.session_state[key] = None
            st.session_state.data_loaded = False
            st.rerun()

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#666; font-size:0.75rem;'>"
        "Built with Streamlit + Python<br>© 2026 OLA Analytics"
        "</div>",
        unsafe_allow_html=True,
    )


# ── Page routing ───────────────────────────────────────────────────────────
if page == "🏠 Home & Upload":
    from pages import home
    home.render()
elif page == "📈 EDA Insights":
    from pages import eda
    eda.render()
elif page == "🗃️ SQL Queries":
    from pages import sql_page
    sql_page.render()
elif page == "🤖 ML Predictions":
    from pages import ml_page
    ml_page.render()
elif page == "💡 Recommendations":
    from pages import reco_page
    reco_page.render()
