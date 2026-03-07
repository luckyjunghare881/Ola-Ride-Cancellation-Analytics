"""
Home & Upload page — CSV upload, data preview, KPI cards.
"""

import streamlit as st
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from src.data_processing import load_csv, preprocess, compute_kpis, get_summary_stats
from src.generate_data import generate_dataset
from src.utils import df_to_csv_download


def render():
    st.markdown('<div class="main-header">🚗 OLA Ride Cancellation Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload data • Explore patterns • Reduce cancellations</div>', unsafe_allow_html=True)

    # ── Upload section ────────────────────────────────────────────────────
    if not st.session_state.data_loaded:
        st.markdown("### 📂 Upload Your Data")

        col1, col2 = st.columns([2, 1])

        with col1:
            uploaded_file = st.file_uploader(
                "Upload OLA ride data (CSV)",
                type=["csv"],
                help="Upload a CSV file with columns: BookingID, BookingDate, RideStatus, etc.",
            )

            if uploaded_file is not None:
                with st.spinner("Processing uploaded file..."):
                    df, warnings = load_csv(uploaded_file)
                    if warnings:
                        for w in warnings:
                            st.error(w)
                        return

                    df_clean, report = preprocess(df)
                    kpis = compute_kpis(df_clean)

                    st.session_state.df = df
                    st.session_state.df_clean = df_clean
                    st.session_state.kpis = kpis
                    st.session_state.cleaning_report = report
                    st.session_state.data_loaded = True
                    st.success(f"✅ Data loaded! {len(df_clean):,} rows processed.")
                    st.rerun()

        with col2:
            st.markdown("#### Quick Start")
            st.info(
                "**No data?** Click **'Load Demo Data'** in the sidebar "
                "to explore with a pre-generated 12,000-row dataset."
            )
            st.markdown("**Expected columns:**")
            st.code(
                "BookingID, BookingDate, RideStatus,\n"
                "VehicleType, PickupLocation, DropLocation,\n"
                "PaymentMode, RideDistance_km, BookingValue_INR,\n"
                "DriverRating, CustomerRating, ETA_Pickup_min,\n"
                "RideDuration_min, CancelReason",
                language=None,
            )
        return

    # ── KPI Dashboard ─────────────────────────────────────────────────────
    df = st.session_state.df_clean
    kpis = st.session_state.kpis
    report = st.session_state.cleaning_report

    st.markdown("### 📊 KPI Overview")

    # Row 1: Main metrics
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.metric("Total Bookings", f"{kpis['total_bookings']:,}")
    with c2:
        st.metric("Completed", f"{kpis['completed_rides']:,}")
    with c3:
        st.metric("Canceled", f"{kpis['canceled_rides']:,}")
    with c4:
        st.metric("Cancel Rate", f"{kpis['cancellation_rate']}%")
    with c5:
        st.metric("Driver Cancel", f"{kpis['driver_cancel_rate']}%")
    with c6:
        st.metric("Customer Cancel", f"{kpis['customer_cancel_rate']}%")

    # Row 2: Secondary metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Avg Fare", f"₹{kpis['avg_booking_value']:,.0f}")
    with c2:
        st.metric("Avg Distance", f"{kpis['avg_ride_distance']:.1f} km")
    with c3:
        st.metric("Avg Driver Rating", f"{kpis['avg_driver_rating']:.1f} / 5")
    with c4:
        st.metric("Revenue Loss (Est.)", f"₹{kpis['revenue_loss_est']:,.0f}")

    st.markdown("---")

    # ── Data cleaning report ──────────────────────────────────────────────
    with st.expander("🧹 Data Cleaning Report", expanded=False):
        if report:
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Original rows:** {report.get('original_rows', 'N/A'):,}")
                st.write(f"**Final rows:** {report.get('final_rows', 'N/A'):,}")
                st.write(f"**Duplicates removed:** {report.get('duplicates_removed', 0):,}")
            with col2:
                if report.get("missing_filled"):
                    st.write("**Missing values handled:**")
                    for col, desc in report["missing_filled"].items():
                        st.write(f"  - `{col}`: {desc}")
                if report.get("columns_added"):
                    st.write(f"**Features added:** {', '.join(report['columns_added'])}")

    # ── Data preview ──────────────────────────────────────────────────────
    st.markdown("### 📋 Data Preview")

    tab1, tab2, tab3 = st.tabs(["📄 Raw Data", "📊 Summary Statistics", "📥 Export"])

    with tab1:
        # Pagination
        page_size = st.selectbox("Rows per page", [25, 50, 100, 250], index=0)
        total_pages = max(1, (len(df) - 1) // page_size + 1)
        page_num = st.number_input("Page", min_value=1, max_value=total_pages, value=1)

        start_idx = (page_num - 1) * page_size
        end_idx = min(start_idx + page_size, len(df))
        st.dataframe(df.iloc[start_idx:end_idx], use_container_width=True, height=400)
        st.caption(f"Showing rows {start_idx + 1} – {end_idx} of {len(df):,}")

    with tab2:
        stats = get_summary_stats(df)
        if not stats.empty:
            st.dataframe(stats, use_container_width=True)
        else:
            st.info("No numeric columns found.")

    with tab3:
        st.markdown("#### Download Data")
        col1, col2 = st.columns(2)
        with col1:
            csv_bytes = df_to_csv_download(df, "ola_rides_cleaned.csv")
            st.download_button(
                "📥 Download Cleaned CSV",
                data=csv_bytes,
                file_name="ola_rides_cleaned.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col2:
            if "IsCanceled" in df.columns:
                canceled_only = df[df["IsCanceled"] == 1]
                csv_cancel = df_to_csv_download(canceled_only, "canceled_rides.csv")
                st.download_button(
                    "📥 Download Canceled Rides Only",
                    data=csv_cancel,
                    file_name="canceled_rides.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
