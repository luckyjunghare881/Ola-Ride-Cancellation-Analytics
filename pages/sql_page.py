"""
SQL Queries page - Run pre-built or custom SQL queries on ride data.
Dark theme styled.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from src.sql_engine import SQLEngine
from src.utils import df_to_csv_download


def _apply_dark(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#A0AEC0"), title_font=dict(color="#FFFFFF"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.05)"),
        legend=dict(font=dict(color="#A0AEC0")),
    )
    return fig


def render():
    st.markdown("""
    <div style="background:#16213E;border:1px solid rgba(255,255,255,0.08);border-radius:12px;padding:14px 28px;margin-bottom:20px;display:flex;align-items:center;gap:12px;">
        <svg viewBox="0 0 24 24" fill="#2196F3" width="24" height="24"><path d="M20 4H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V6c0-1.1-.9-2-2-2zm0 14H4V8h16v10z"/><path d="M6 10h2v2H6zm0 4h8v2H6zm10 0h2v2h-2zm-6-4h8v2h-8z"/></svg>
        <div>
            <div style="font-size:20px;font-weight:700;color:#FFF;">SQL Query Engine</div>
            <div style="font-size:12px;color:#A0AEC0;">Query ride data using SQL - SELECT queries only</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.data_loaded:
        st.warning("Please load data first from the Overview Dashboard or click 'Load Demo Data' in the sidebar.")
        return

    df = st.session_state.df_clean

    if "sql_engine" not in st.session_state or st.session_state.sql_engine is None:
        engine = SQLEngine()
        engine.load_data(df)
        st.session_state.sql_engine = engine
    else:
        engine = st.session_state.sql_engine

    st.markdown("""
    <div style="background:rgba(33,150,243,0.1);border:1px solid rgba(33,150,243,0.3);border-radius:10px;padding:10px 18px;margin-bottom:16px;display:flex;align-items:center;gap:8px;">
        <svg viewBox="0 0 24 24" fill="#2196F3" width="18" height="18"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z"/></svg>
        <span style="color:#A0AEC0;font-size:13px;">Query the ride data using SQL. Only <strong style='color:#FFF;'>SELECT</strong> queries are allowed for safety.</span>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Pre-built Queries", "Custom Query"])

    # Tab 1: Pre-built
    with tab1:
        queries = engine.get_prebuilt_queries()
        query_names = list(queries.keys())
        selected = st.selectbox("Select a query:", query_names)

        if selected:
            query_info = queries[selected]
            st.markdown(f'<div style="color:#A0AEC0;font-size:13px;margin-bottom:8px;"><strong style="color:#FFF;">Description:</strong> {query_info["description"]}</div>', unsafe_allow_html=True)

            with st.expander("View SQL", expanded=False):
                st.code(query_info["sql"], language="sql")

            if st.button("Run Query", key="run_prebuilt", use_container_width=True):
                with st.spinner("Executing query..."):
                    result, error = engine.execute_query(query_info["sql"])

                if error:
                    st.error(error)
                elif result.empty:
                    st.info("Query returned no results.")
                else:
                    st.markdown(f"""
                    <div style="background:rgba(0,200,83,0.1);border:1px solid rgba(0,200,83,0.3);border-radius:8px;padding:8px 14px;margin-bottom:10px;display:flex;align-items:center;gap:8px;">
                        <svg viewBox="0 0 24 24" fill="#00C853" width="16" height="16"><path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/></svg>
                        <span style="color:#00C853;font-size:13px;font-weight:600;">{len(result)} rows returned</span>
                    </div>
                    """, unsafe_allow_html=True)
                    st.dataframe(result, use_container_width=True, height=350)
                    _auto_chart(result, selected)
                    csv_bytes = df_to_csv_download(result, f"query_{selected}.csv")
                    st.download_button("Download Results", data=csv_bytes,
                        file_name="query_result.csv", mime="text/csv", use_container_width=True)

    # Tab 2: Custom
    with tab2:
        st.markdown('<div style="font-size:15px;font-weight:600;color:#FFF;margin-bottom:10px;">Write a Custom SQL Query</div>', unsafe_allow_html=True)
        st.markdown('<div style="color:#A0AEC0;font-size:13px;margin-bottom:8px;">Table name: <code style="background:#16213E;padding:2px 8px;border-radius:4px;color:#00C853;">rides</code></div>', unsafe_allow_html=True)

        with st.expander("Available Columns", expanded=False):
            col_info = pd.DataFrame({
                "Column": df.columns.tolist(),
                "Type": [str(df[c].dtype) for c in df.columns],
                "Non-Null": [df[c].notna().sum() for c in df.columns],
                "Sample": [str(df[c].iloc[0]) if len(df) > 0 else "" for c in df.columns],
            })
            st.dataframe(col_info, use_container_width=True, height=300)

        st.markdown('<div style="color:#A0AEC0;font-size:13px;font-weight:600;margin-bottom:6px;">Quick Templates:</div>', unsafe_allow_html=True)
        templates = {
            "All canceled rides": "SELECT * FROM rides WHERE IsCanceled = 1 LIMIT 100;",
            "Cancel rate by location": "SELECT PickupLocation, COUNT(*) as Total, SUM(IsCanceled) as Canceled, ROUND(SUM(IsCanceled)*100.0/COUNT(*),2) as CancelRate FROM rides GROUP BY PickupLocation ORDER BY CancelRate DESC;",
            "High-value canceled rides": "SELECT BookingID, PickupLocation, DropLocation, BookingValue_INR, CancelReason FROM rides WHERE IsCanceled = 1 AND BookingValue_INR > 200 ORDER BY BookingValue_INR DESC LIMIT 50;",
            "Avg ETA by status": "SELECT RideStatus, ROUND(AVG(ETA_Pickup_min),2) as AvgETA, COUNT(*) as Count FROM rides GROUP BY RideStatus;",
        }

        template_choice = st.selectbox("Load template:", ["(custom)"] + list(templates.keys()))
        default_sql = templates.get(template_choice, "SELECT * FROM rides LIMIT 10;")

        custom_sql = st.text_area("SQL Query", value=default_sql, height=120,
            help="Write SELECT queries only. Table name: rides")

        if st.button("Execute", key="run_custom", use_container_width=True):
            if not custom_sql.strip():
                st.warning("Please enter a SQL query.")
            else:
                with st.spinner("Executing..."):
                    result, error = engine.execute_query(custom_sql)

                if error:
                    st.error(error)
                elif result.empty:
                    st.info("Query returned no results.")
                else:
                    st.markdown(f"""
                    <div style="background:rgba(0,200,83,0.1);border:1px solid rgba(0,200,83,0.3);border-radius:8px;padding:8px 14px;margin-bottom:10px;display:flex;align-items:center;gap:8px;">
                        <svg viewBox="0 0 24 24" fill="#00C853" width="16" height="16"><path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/></svg>
                        <span style="color:#00C853;font-size:13px;font-weight:600;">{len(result)} rows returned</span>
                    </div>
                    """, unsafe_allow_html=True)
                    st.dataframe(result, use_container_width=True, height=400)
                    csv_bytes = df_to_csv_download(result)
                    st.download_button("Download Results", data=csv_bytes,
                        file_name="custom_query_result.csv", mime="text/csv", use_container_width=True)


def _auto_chart(result: pd.DataFrame, query_name: str):
    """Auto-generate a dark-themed chart based on query results."""
    if len(result) < 2:
        return
    cols = result.columns.tolist()
    numeric_cols = result.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in cols if c not in numeric_cols]

    try:
        if "CancelRate" in cols and cat_cols:
            fig = px.bar(result, x=cat_cols[0], y="CancelRate", title=f"{query_name}",
                color="CancelRate", color_continuous_scale=["#00C853","#FFB300","#F44336"],
                text="CancelRate")
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            _apply_dark(fig)
            st.plotly_chart(fig, use_container_width=True)
        elif "Occurrences" in cols and cat_cols:
            fig = px.bar(result, y=cat_cols[0], x="Occurrences", title=f"{query_name}",
                orientation="h", color="Occurrences", color_continuous_scale=[[0,"#0D0D1A"],[1,"#F44336"]])
            _apply_dark(fig)
            st.plotly_chart(fig, use_container_width=True)
        elif len(numeric_cols) >= 1 and cat_cols:
            fig = px.bar(result, x=cat_cols[0], y=numeric_cols[0], title=f"{query_name}",
                color=cat_cols[0])
            _apply_dark(fig)
            st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass
