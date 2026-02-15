"""
SQL Queries page — Run pre-built or custom SQL queries on ride data.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from src.sql_engine import SQLEngine
from src.utils import df_to_csv_download


def render():
    st.markdown("## 🗃️ SQL Query Engine")

    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first from the Home page or click 'Load Demo Data' in the sidebar.")
        return

    df = st.session_state.df_clean

    # Initialize SQL engine
    if "sql_engine" not in st.session_state or st.session_state.sql_engine is None:
        engine = SQLEngine()
        engine.load_data(df)
        st.session_state.sql_engine = engine
    else:
        engine = st.session_state.sql_engine

    st.info("💡 Query the ride data using SQL. Only **SELECT** queries are allowed for safety.")

    tab1, tab2 = st.tabs(["📋 Pre-built Queries", "✏️ Custom Query"])

    # ── Tab 1: Pre-built queries ──────────────────────────────────────────
    with tab1:
        queries = engine.get_prebuilt_queries()
        query_names = list(queries.keys())

        selected = st.selectbox("Select a query:", query_names)

        if selected:
            query_info = queries[selected]
            st.markdown(f"**Description:** {query_info['description']}")

            with st.expander("📄 View SQL", expanded=False):
                st.code(query_info["sql"], language="sql")

            if st.button("▶️ Run Query", key="run_prebuilt"):
                with st.spinner("Executing query..."):
                    result, error = engine.execute_query(query_info["sql"])

                if error:
                    st.error(f"❌ {error}")
                elif result.empty:
                    st.info("Query returned no results.")
                else:
                    st.success(f"✅ {len(result)} rows returned")

                    # Show results
                    st.dataframe(result, use_container_width=True, height=350)

                    # Auto-visualize
                    _auto_chart(result, selected)

                    # Download
                    csv_bytes = df_to_csv_download(result, f"query_{selected}.csv")
                    st.download_button(
                        "📥 Download Results",
                        data=csv_bytes,
                        file_name=f"query_result.csv",
                        mime="text/csv",
                    )

    # ── Tab 2: Custom query ───────────────────────────────────────────────
    with tab2:
        st.markdown("#### Write a Custom SQL Query")
        st.markdown("Table name: **`rides`** | [View columns below]")

        with st.expander("📋 Available Columns", expanded=False):
            col_info = pd.DataFrame({
                "Column": df.columns.tolist(),
                "Type": [str(df[c].dtype) for c in df.columns],
                "Non-Null": [df[c].notna().sum() for c in df.columns],
                "Sample": [str(df[c].iloc[0]) if len(df) > 0 else "" for c in df.columns],
            })
            st.dataframe(col_info, use_container_width=True, height=300)

        # Query templates
        st.markdown("**Quick templates:**")
        templates = {
            "All canceled rides": "SELECT * FROM rides WHERE IsCanceled = 1 LIMIT 100;",
            "Cancel rate by location": "SELECT PickupLocation, COUNT(*) as Total, SUM(IsCanceled) as Canceled, ROUND(SUM(IsCanceled)*100.0/COUNT(*),2) as CancelRate FROM rides GROUP BY PickupLocation ORDER BY CancelRate DESC;",
            "High-value canceled rides": "SELECT BookingID, PickupLocation, DropLocation, BookingValue_INR, CancelReason FROM rides WHERE IsCanceled = 1 AND BookingValue_INR > 200 ORDER BY BookingValue_INR DESC LIMIT 50;",
            "Avg ETA by status": "SELECT RideStatus, ROUND(AVG(ETA_Pickup_min),2) as AvgETA, COUNT(*) as Count FROM rides GROUP BY RideStatus;",
        }

        template_choice = st.selectbox("Load template:", ["(custom)"] + list(templates.keys()))
        default_sql = templates.get(template_choice, "SELECT * FROM rides LIMIT 10;")

        custom_sql = st.text_area(
            "SQL Query",
            value=default_sql,
            height=120,
            help="Write SELECT queries only. Table name: rides",
        )

        if st.button("▶️ Execute", key="run_custom"):
            if not custom_sql.strip():
                st.warning("Please enter a SQL query.")
            else:
                with st.spinner("Executing..."):
                    result, error = engine.execute_query(custom_sql)

                if error:
                    st.error(f"❌ {error}")
                elif result.empty:
                    st.info("Query returned no results.")
                else:
                    st.success(f"✅ {len(result)} rows returned")
                    st.dataframe(result, use_container_width=True, height=400)

                    csv_bytes = df_to_csv_download(result)
                    st.download_button(
                        "📥 Download Results",
                        data=csv_bytes,
                        file_name="custom_query_result.csv",
                        mime="text/csv",
                    )


def _auto_chart(result: pd.DataFrame, query_name: str):
    """Auto-generate a chart based on query results."""
    if len(result) < 2:
        return

    cols = result.columns.tolist()
    numeric_cols = result.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in cols if c not in numeric_cols]

    try:
        if "CancelRate" in cols and cat_cols:
            fig = px.bar(
                result, x=cat_cols[0], y="CancelRate",
                title=f"{query_name} — Visualization",
                color="CancelRate",
                color_continuous_scale=["#1B9E3E", "#ffa502", "#ff6b6b"],
                text="CancelRate",
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            st.plotly_chart(fig, use_container_width=True)
        elif "Occurrences" in cols and cat_cols:
            fig = px.bar(
                result, y=cat_cols[0], x="Occurrences",
                title=f"{query_name} — Visualization",
                orientation="h",
                color="Occurrences",
                color_continuous_scale="Reds",
            )
            st.plotly_chart(fig, use_container_width=True)
        elif len(numeric_cols) >= 1 and cat_cols:
            fig = px.bar(
                result, x=cat_cols[0], y=numeric_cols[0],
                title=f"{query_name} — Visualization",
                color=cat_cols[0],
            )
            st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass  # Skip chart if auto-detection fails
