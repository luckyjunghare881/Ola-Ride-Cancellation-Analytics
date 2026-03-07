"""
OLA Ride Cancellation Analytics - Streamlit Dashboard
Main entry point. Multi-page app with modern dark sidebar navigation.
"""

import streamlit as st
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.data_processing import load_csv, preprocess, compute_kpis
from src.generate_data import generate_dataset, save_dataset

st.set_page_config(
    page_title="OLA Cancellation Analytics",
    page_icon="https://img.icons8.com/color/48/ola-.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Master CSS - Dark Theme
st.markdown("""
<style>
    :root {
        --primary: #00C853; --primary-dark: #009624; --accent: #FF6B35;
        --warning: #FFB300; --danger: #F44336; --success: #4CAF50;
        --bg-dark: #0D0D1A; --bg-card: #16213E; --bg-card2: #1A1A2E;
        --text-primary: #FFFFFF; --text-secondary: #A0AEC0;
        --border: rgba(255,255,255,0.08);
    }
    .stApp { background: #0D0D1A !important; color: #FFFFFF !important; }
    header[data-testid="stHeader"] { background: transparent !important; }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #16213E 0%, #0D0D1A 100%) !important;
        border-right: 1px solid rgba(255,255,255,0.08) !important;
    }
    section[data-testid="stSidebar"] .stMarkdown { color: #A0AEC0 !important; }
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0D0D1A; }
    ::-webkit-scrollbar-thumb { background: #00C853; border-radius: 3px; }
    div[data-testid="stMetric"] {
        background: #16213E !important; border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 16px !important; padding: 18px 22px !important;
        transition: transform 0.2s, box-shadow 0.2s !important;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-3px) !important; box-shadow: 0 8px 30px rgba(0,0,0,0.3) !important;
    }
    div[data-testid="stMetric"] label {
        color: #A0AEC0 !important; font-size: 0.75rem !important;
        text-transform: uppercase !important; letter-spacing: 0.5px !important; font-weight: 500 !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #FFFFFF !important; font-size: 1.8rem !important; font-weight: 800 !important;
    }
    details {
        background: #16213E !important; border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 12px !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        background: #0D0D1A !important; border-radius: 12px !important; padding: 4px !important;
        gap: 4px !important; border: 1px solid rgba(255,255,255,0.08);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px !important; font-weight: 600 !important; color: #A0AEC0 !important;
        background: transparent !important; border: none !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: #16213E !important; color: #00C853 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3) !important;
    }
    .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none !important; }
    .stButton > button {
        background: #00C853 !important; color: white !important; border: none !important;
        border-radius: 10px !important; padding: 9px 18px !important; font-weight: 600 !important;
    }
    .stButton > button:hover { background: #009624 !important; box-shadow: 0 0 20px rgba(0,200,83,0.3) !important; }
    .stDownloadButton > button {
        background: transparent !important; border: 1px solid rgba(255,255,255,0.08) !important;
        color: #A0AEC0 !important; border-radius: 10px !important;
    }
    .stDownloadButton > button:hover { border-color: #00C853 !important; color: #00C853 !important; }
    .stSelectbox [data-baseweb="select"] > div {
        background: #0D0D1A !important; border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 10px !important; color: #FFFFFF !important;
    }
    .stTextInput input, .stNumberInput input, .stTextArea textarea {
        background: #0D0D1A !important; border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 10px !important; color: #FFFFFF !important;
    }
    .stMultiSelect [data-baseweb="select"] > div {
        background: #0D0D1A !important; border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 10px !important;
    }
    .stFileUploader { border-radius: 12px !important; }
    @keyframes pulse-border { 0%,100%{border-color:rgba(244,67,54,0.3)} 50%{border-color:rgba(244,67,54,0.7)} }
    @keyframes pulse-spot { 0%,100%{transform:scale(1);opacity:0.8} 50%{transform:scale(1.15);opacity:1} }
    @keyframes fadeUp { from{opacity:0;transform:translateY(10px)} to{opacity:1;transform:translateY(0)} }
    .fade-up { animation: fadeUp 0.5s ease forwards; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    .stRadio > div { gap: 0 !important; }
    .stPlotlyChart { border-radius: 12px; overflow: hidden; }
    div[data-testid="stRadio"] label {
        padding: 8px 14px !important; border-radius: 10px !important; transition: all 0.2s ease !important;
    }
    div[data-testid="stRadio"] label:hover { background: rgba(0,200,83,0.08) !important; }
    .stSlider [data-baseweb="slider"] div div div { background: #00C853 !important; }
</style>
""", unsafe_allow_html=True)

def init_session_state():
    defaults = {
        "df": None, "df_clean": None, "kpis": None,
        "cleaning_report": None, "data_loaded": False,
        "ml_result": None, "user_role": "Analyst",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()

with st.sidebar:
    st.markdown("""
    <div style="padding:20px 16px;border-bottom:1px solid rgba(255,255,255,0.08);display:flex;align-items:center;gap:12px;margin-bottom:10px;">
        <div style="width:42px;height:42px;background:linear-gradient(135deg,#00C853,#009624);border-radius:12px;display:flex;align-items:center;justify-content:center;box-shadow:0 0 20px rgba(0,200,83,0.3);">
            <svg viewBox="0 0 24 24" fill="white" width="24" height="24"><path d="M18.92 6.01C18.72 5.42 18.16 5 17.5 5h-11c-.66 0-1.21.42-1.42 1.01L3 12v8c0 .55.45 1 1 1h1c.55 0 1-.45 1-1v-1h12v1c0 .55.45 1 1 1h1c.55 0 1-.45 1-1v-8l-2.08-5.99zM6.5 16c-.83 0-1.5-.67-1.5-1.5S5.67 13 6.5 13s1.5.67 1.5 1.5S7.33 16 6.5 16zm11 0c-.83 0-1.5-.67-1.5-1.5s.67-1.5 1.5-1.5 1.5.67 1.5 1.5-.67 1.5-1.5 1.5zM5 11l1.5-4.5h11L19 11H5z"/></svg>
        </div>
        <div>
            <div style="font-size:18px;font-weight:700;color:#FFFFFF;">OLA Analytics</div>
            <div style="font-size:11px;color:#00C853;font-weight:500;letter-spacing:1px;text-transform:uppercase;">Cancellation Suite</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="font-size:10px;color:#A0AEC0;text-transform:uppercase;letter-spacing:1.5px;padding:0 8px;margin:16px 0 8px;font-weight:600;">MAIN</div>', unsafe_allow_html=True)

    page = st.radio("Navigation",
        ["Overview Dashboard", "EDA Insights", "SQL Queries", "ML Predictions", "Recommendations"],
        index=0, label_visibility="collapsed")

    st.markdown("---")

    if not st.session_state.data_loaded:
        st.markdown('<div style="font-size:10px;color:#A0AEC0;text-transform:uppercase;letter-spacing:1.5px;padding:0 8px;margin:8px 0;font-weight:600;">DATA</div>', unsafe_allow_html=True)
        if st.button("Load Demo Data", use_container_width=True):
            with st.spinner("Generating demo dataset..."):
                demo_df = generate_dataset(num_rows=12000)
                demo_df_clean, report = preprocess(demo_df)
                kpis = compute_kpis(demo_df_clean)
                st.session_state.df = demo_df
                st.session_state.df_clean = demo_df_clean
                st.session_state.kpis = kpis
                st.session_state.cleaning_report = report
                st.session_state.data_loaded = True
            st.rerun()
    else:
        n = len(st.session_state.df_clean)
        st.markdown(f"""
        <div style="background:rgba(0,200,83,0.1);border:1px solid rgba(0,200,83,0.2);border-radius:10px;padding:10px 14px;margin-bottom:10px;">
            <div style="font-size:12px;color:#00C853;font-weight:600;">
                <svg style="width:14px;height:14px;vertical-align:middle;margin-right:4px" viewBox="0 0 24 24" fill="#00C853"><path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/></svg>DATA LOADED
            </div>
            <div style="font-size:11px;color:#A0AEC0;margin-top:2px;">{n:,} rows processed</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Reset Data", use_container_width=True):
            for key in ["df", "df_clean", "kpis", "cleaning_report", "ml_result"]:
                st.session_state[key] = None
            st.session_state.data_loaded = False
            st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;padding:10px 12px;border-radius:10px;background:rgba(255,255,255,0.04);margin-top:10px;">
        <div style="width:36px;height:36px;background:linear-gradient(135deg,#667eea,#764ba2);border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:14px;font-weight:700;color:white;">LJ</div>
        <div>
            <div style="font-size:13px;font-weight:600;color:#FFFFFF;">Lucky Junghare</div>
            <div style="font-size:11px;color:#A0AEC0;">Data Analyst</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div style="text-align:center;color:#555;font-size:0.7rem;margin-top:14px;">Built with Streamlit + Python<br>\u00a9 2026 OLA Analytics</div>', unsafe_allow_html=True)

if page == "Overview Dashboard":
    from pages import home
    home.render()
elif page == "EDA Insights":
    from pages import eda
    eda.render()
elif page == "SQL Queries":
    from pages import sql_page
    sql_page.render()
elif page == "ML Predictions":
    from pages import ml_page
    ml_page.render()
elif page == "Recommendations":
    from pages import reco_page
    reco_page.render()
