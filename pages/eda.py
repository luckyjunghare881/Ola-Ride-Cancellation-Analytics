"""
EDA Insights page - Interactive exploratory data analysis with Plotly charts.
Dark theme styled to match the dashboard design.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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
        <svg viewBox="0 0 24 24" fill="#00C853" width="24" height="24"><path d="M16 6l2.29 2.29-4.88 4.88-4-4L2 16.59 3.41 18l6-6 4 4 6.3-6.29L22 12V6z"/></svg>
        <div>
            <div style="font-size:20px;font-weight:700;color:#FFF;">Exploratory Data Analysis</div>
            <div style="font-size:12px;color:#A0AEC0;">Interactive charts and deep-dive insights</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.data_loaded:
        st.warning("Please load data first from the Overview Dashboard or click 'Load Demo Data' in the sidebar.")
        return

    df = st.session_state.df_clean

    # Filters
    with st.expander("Filters", expanded=True):
        fc1, fc2, fc3, fc4 = st.columns(4)
        with fc1:
            ride_filter = st.multiselect("Ride Status", options=df["RideStatus"].unique().tolist(),
                                         default=df["RideStatus"].unique().tolist())
        with fc2:
            vehicle_filter = st.multiselect("Vehicle Type",
                options=df["VehicleType"].unique().tolist() if "VehicleType" in df.columns else [],
                default=df["VehicleType"].unique().tolist() if "VehicleType" in df.columns else [])
        with fc3:
            payment_filter = st.multiselect("Payment Mode",
                options=df["PaymentMode"].unique().tolist() if "PaymentMode" in df.columns else [],
                default=df["PaymentMode"].unique().tolist() if "PaymentMode" in df.columns else [])
        with fc4:
            if "PickupLocation" in df.columns:
                locations = sorted(df["PickupLocation"].unique().tolist())
                loc_filter = st.multiselect("Pickup Location", options=locations, default=locations)
            else:
                loc_filter = []

    mask = df["RideStatus"].isin(ride_filter)
    if "VehicleType" in df.columns and vehicle_filter:
        mask &= df["VehicleType"].isin(vehicle_filter)
    if "PaymentMode" in df.columns and payment_filter:
        mask &= df["PaymentMode"].isin(payment_filter)
    if "PickupLocation" in df.columns and loc_filter:
        mask &= df["PickupLocation"].isin(loc_filter)
    filtered = df[mask]

    st.markdown(f'<div style="font-size:12px;color:#A0AEC0;margin-bottom:16px;">Filtered: {len(filtered):,} / {len(df):,} rows</div>', unsafe_allow_html=True)

    if filtered.empty:
        st.warning("No data matches the current filters.")
        return

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Time Analysis", "Location Analysis", "Payment & Vehicle", "Heatmaps"])

    # Tab 1: Overview
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            status_counts = filtered["RideStatus"].value_counts().reset_index()
            status_counts.columns = ["RideStatus", "Count"]
            fig = px.pie(status_counts, values="Count", names="RideStatus", title="Ride Status Distribution",
                         color_discrete_sequence=["#4CAF50", "#F44336", "#FF9800", "#FFB300"], hole=0.4)
            fig.update_traces(textinfo="percent+label", textfont=dict(color="#FFF"))
            _apply_dark(fig)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            cancel_df = filtered[filtered.get("IsCanceled", pd.Series()) == 1]
            if not cancel_df.empty:
                cancel_type = cancel_df["RideStatus"].value_counts().reset_index()
                cancel_type.columns = ["Type", "Count"]
                fig = px.bar(cancel_type, x="Type", y="Count", title="Cancellation Breakdown",
                             color="Type", color_discrete_sequence=["#F44336", "#FF9800"])
                _apply_dark(fig)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No cancellations in filtered data.")

        if "BookingValue_INR" in filtered.columns:
            fig = px.histogram(filtered, x="BookingValue_INR", color="RideStatus",
                               title="Booking Value Distribution by Status", nbins=50,
                               barmode="overlay", opacity=0.7,
                               color_discrete_sequence=["#00C853", "#F44336", "#FF9800"])
            fig.update_layout(xaxis_title="Booking Value (INR)", yaxis_title="Count")
            _apply_dark(fig)
            st.plotly_chart(fig, use_container_width=True)

    # Tab 2: Time Analysis
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            if "BookingHour" in filtered.columns:
                hourly = filtered.groupby("BookingHour").agg(
                    total=("IsCanceled", "count"), canceled=("IsCanceled", "sum")).reset_index()
                hourly["cancel_rate"] = (hourly["canceled"] / hourly["total"] * 100).round(2)
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Bar(x=hourly["BookingHour"], y=hourly["total"],
                    name="Total Bookings", marker_color="#00C853", opacity=0.6), secondary_y=False)
                fig.add_trace(go.Scatter(x=hourly["BookingHour"], y=hourly["cancel_rate"],
                    name="Cancel Rate %", line=dict(color="#F44336", width=3), mode="lines+markers"), secondary_y=True)
                fig.update_layout(title="Hourly Booking Volume & Cancellation Rate", xaxis_title="Hour of Day")
                fig.update_yaxes(title_text="Bookings", secondary_y=False)
                fig.update_yaxes(title_text="Cancel Rate %", secondary_y=True)
                _apply_dark(fig)
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            if "DayOfWeek" in filtered.columns:
                day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
                daily = filtered.groupby("DayOfWeek").agg(
                    total=("IsCanceled","count"), canceled=("IsCanceled","sum")).reindex(day_order).reset_index()
                daily["cancel_rate"] = (daily["canceled"] / daily["total"] * 100).round(2)
                fig = px.bar(daily, x="DayOfWeek", y="cancel_rate", title="Cancellation Rate by Day of Week",
                             color="cancel_rate", color_continuous_scale=["#00C853","#FFB300","#F44336"])
                _apply_dark(fig)
                st.plotly_chart(fig, use_container_width=True)

        if "Month" in filtered.columns and "BookingDate" in filtered.columns:
            monthly = filtered.groupby(filtered["BookingDate"].dt.to_period("M")).agg(
                total=("IsCanceled","count"), canceled=("IsCanceled","sum")).reset_index()
            monthly["BookingDate"] = monthly["BookingDate"].astype(str)
            monthly["cancel_rate"] = (monthly["canceled"] / monthly["total"] * 100).round(2)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=monthly["BookingDate"], y=monthly["total"],
                name="Total Bookings", fill="tozeroy", line=dict(color="#00C853")))
            fig.add_trace(go.Scatter(x=monthly["BookingDate"], y=monthly["canceled"],
                name="Canceled", fill="tozeroy", line=dict(color="#F44336")))
            fig.update_layout(title="Monthly Booking & Cancellation Trend", xaxis_title="Month", yaxis_title="Count")
            _apply_dark(fig)
            st.plotly_chart(fig, use_container_width=True)

        if "TimePeriod" in filtered.columns:
            tp = filtered.groupby("TimePeriod").agg(
                total=("IsCanceled","count"), canceled=("IsCanceled","sum")).reset_index()
            tp["cancel_rate"] = (tp["canceled"] / tp["total"] * 100).round(2)
            fig = px.bar(tp, x="TimePeriod", y=["total","canceled"], title="Bookings by Time Period",
                         barmode="group", color_discrete_sequence=["#00C853","#F44336"])
            _apply_dark(fig)
            st.plotly_chart(fig, use_container_width=True)

    # Tab 3: Location
    with tab3:
        if "PickupLocation" in filtered.columns:
            loc_stats = filtered.groupby("PickupLocation").agg(
                total=("IsCanceled","count"), canceled=("IsCanceled","sum")).reset_index()
            loc_stats["cancel_rate"] = (loc_stats["canceled"] / loc_stats["total"] * 100).round(2)
            loc_stats = loc_stats.sort_values("cancel_rate", ascending=True)

            col1, col2 = st.columns(2)
            with col1:
                fig = px.bar(loc_stats, y="PickupLocation", x="cancel_rate",
                    title="Cancellation Rate by Pickup Location", orientation="h",
                    color="cancel_rate", color_continuous_scale=["#00C853","#FFB300","#F44336"])
                fig.update_layout(yaxis_title="", xaxis_title="Cancel Rate %", height=600)
                _apply_dark(fig)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = px.bar(loc_stats.sort_values("total", ascending=True),
                    y="PickupLocation", x="total", title="Total Bookings by Location",
                    orientation="h", color="total", color_continuous_scale="Greens")
                fig.update_layout(yaxis_title="", xaxis_title="Total Bookings", height=600)
                _apply_dark(fig)
                st.plotly_chart(fig, use_container_width=True)

            if "DropLocation" in filtered.columns:
                routes = filtered.groupby(["PickupLocation","DropLocation"]).agg(
                    total=("IsCanceled","count"), canceled=("IsCanceled","sum")).reset_index()
                routes["cancel_rate"] = (routes["canceled"] / routes["total"] * 100).round(2)
                top_routes = routes.nlargest(15, "cancel_rate")
                fig = px.scatter(top_routes, x="total", y="cancel_rate", size="canceled",
                    color="PickupLocation", hover_data=["DropLocation"],
                    title="Top 15 High-Cancellation Routes (Pickup to Drop)")
                fig.update_layout(xaxis_title="Total Rides", yaxis_title="Cancel Rate %")
                _apply_dark(fig)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No location data available.")

    # Tab 4: Payment & Vehicle
    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            if "PaymentMode" in filtered.columns:
                pay_stats = filtered.groupby("PaymentMode").agg(
                    total=("IsCanceled","count"), canceled=("IsCanceled","sum")).reset_index()
                pay_stats["cancel_rate"] = (pay_stats["canceled"] / pay_stats["total"] * 100).round(2)
                fig = px.bar(pay_stats, x="PaymentMode", y="cancel_rate",
                    title="Cancellation Rate by Payment Mode", color="cancel_rate",
                    color_continuous_scale=["#00C853","#FFB300","#F44336"], text="cancel_rate")
                fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                _apply_dark(fig)
                st.plotly_chart(fig, use_container_width=True)

                fig2 = px.pie(pay_stats, values="total", names="PaymentMode",
                    title="Booking Share by Payment Mode",
                    color_discrete_sequence=["#00C853","#4FC3F7","#FF9800","#9C27B0","#FFB300"])
                fig2.update_traces(textfont=dict(color="#FFF"))
                _apply_dark(fig2)
                st.plotly_chart(fig2, use_container_width=True)

        with col2:
            if "VehicleType" in filtered.columns:
                veh_stats = filtered.groupby("VehicleType").agg(
                    total=("IsCanceled","count"), canceled=("IsCanceled","sum"),
                    avg_fare=("BookingValue_INR","mean")).reset_index()
                veh_stats["cancel_rate"] = (veh_stats["canceled"] / veh_stats["total"] * 100).round(2)
                fig = px.bar(veh_stats, x="VehicleType", y="cancel_rate",
                    title="Cancellation Rate by Vehicle Type", color="cancel_rate",
                    color_continuous_scale=["#00C853","#FFB300","#F44336"], text="cancel_rate")
                fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                _apply_dark(fig)
                st.plotly_chart(fig, use_container_width=True)

                fig2 = px.bar(veh_stats, x="VehicleType", y="avg_fare",
                    title="Average Fare by Vehicle Type", color="VehicleType",
                    color_discrete_sequence=["#00C853","#4FC3F7","#FF9800","#9C27B0","#FFB300"],
                    text=veh_stats["avg_fare"].round(0))
                fig2.update_traces(texttemplate="\u20b9%{text:,.0f}", textposition="outside")
                _apply_dark(fig2)
                st.plotly_chart(fig2, use_container_width=True)

        if "ETA_Pickup_min" in filtered.columns:
            fig = px.box(filtered, x="RideStatus", y="ETA_Pickup_min",
                title="ETA Distribution by Ride Status", color="RideStatus",
                color_discrete_sequence=["#00C853","#F44336","#FF9800"])
            fig.update_layout(xaxis_title="", yaxis_title="ETA to Pickup (min)")
            _apply_dark(fig)
            st.plotly_chart(fig, use_container_width=True)

    # Tab 5: Heatmaps
    with tab5:
        if "BookingHour" in filtered.columns and "DayOfWeek" in filtered.columns and "IsCanceled" in filtered.columns:
            day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            hm = filtered.groupby(["DayOfWeek","BookingHour"])["IsCanceled"].mean().reset_index()
            hm["IsCanceled"] = (hm["IsCanceled"] * 100).round(2)
            hm_pivot = hm.pivot(index="DayOfWeek", columns="BookingHour", values="IsCanceled").reindex(day_order)
            fig = px.imshow(hm_pivot, title="Cancellation Rate Heatmap - Day x Hour",
                labels=dict(x="Hour of Day", y="Day of Week", color="Cancel Rate %"),
                color_continuous_scale=[[0,"rgba(0,200,83,0.2)"],[0.5,"rgba(255,179,0,0.6)"],[1,"rgba(244,67,54,0.95)"]],
                aspect="auto")
            fig.update_layout(height=450)
            _apply_dark(fig)
            st.plotly_chart(fig, use_container_width=True)

        if "PickupLocation" in filtered.columns and "VehicleType" in filtered.columns and "IsCanceled" in filtered.columns:
            loc_veh = filtered.groupby(["PickupLocation","VehicleType"])["IsCanceled"].mean().reset_index()
            loc_veh["IsCanceled"] = (loc_veh["IsCanceled"] * 100).round(2)
            loc_veh_pivot = loc_veh.pivot(index="PickupLocation", columns="VehicleType", values="IsCanceled")
            fig = px.imshow(loc_veh_pivot, title="Cancellation Rate - Location x Vehicle Type",
                labels=dict(x="Vehicle Type", y="Pickup Location", color="Cancel Rate %"),
                color_continuous_scale=[[0,"rgba(0,200,83,0.2)"],[0.5,"rgba(255,179,0,0.6)"],[1,"rgba(244,67,54,0.95)"]],
                aspect="auto")
            fig.update_layout(height=650)
            _apply_dark(fig)
            st.plotly_chart(fig, use_container_width=True)

        if "CancelReason" in filtered.columns:
            reasons = filtered[filtered["CancelReason"].notna()]["CancelReason"].value_counts().head(10).reset_index()
            reasons.columns = ["Reason", "Count"]
            fig = px.bar(reasons, y="Reason", x="Count", title="Top 10 Cancellation Reasons",
                orientation="h", color="Count", color_continuous_scale=["#FF9800","#F44336"])
            fig.update_layout(yaxis_title="", height=400)
            _apply_dark(fig)
            st.plotly_chart(fig, use_container_width=True)
