"""
EDA Insights page — Interactive exploratory data analysis with Plotly charts.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def render():
    st.markdown("## 📈 Exploratory Data Analysis")

    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first from the Home page or click 'Load Demo Data' in the sidebar.")
        return

    df = st.session_state.df_clean

    # ── Filters ───────────────────────────────────────────────────────────
    with st.expander("🔍 Filters", expanded=True):
        fc1, fc2, fc3, fc4 = st.columns(4)
        with fc1:
            ride_filter = st.multiselect(
                "Ride Status",
                options=df["RideStatus"].unique().tolist(),
                default=df["RideStatus"].unique().tolist(),
            )
        with fc2:
            vehicle_filter = st.multiselect(
                "Vehicle Type",
                options=df["VehicleType"].unique().tolist() if "VehicleType" in df.columns else [],
                default=df["VehicleType"].unique().tolist() if "VehicleType" in df.columns else [],
            )
        with fc3:
            payment_filter = st.multiselect(
                "Payment Mode",
                options=df["PaymentMode"].unique().tolist() if "PaymentMode" in df.columns else [],
                default=df["PaymentMode"].unique().tolist() if "PaymentMode" in df.columns else [],
            )
        with fc4:
            if "PickupLocation" in df.columns:
                locations = sorted(df["PickupLocation"].unique().tolist())
                loc_filter = st.multiselect("Pickup Location", options=locations, default=locations)
            else:
                loc_filter = []

    # Apply filters
    mask = df["RideStatus"].isin(ride_filter)
    if "VehicleType" in df.columns and vehicle_filter:
        mask &= df["VehicleType"].isin(vehicle_filter)
    if "PaymentMode" in df.columns and payment_filter:
        mask &= df["PaymentMode"].isin(payment_filter)
    if "PickupLocation" in df.columns and loc_filter:
        mask &= df["PickupLocation"].isin(loc_filter)

    filtered = df[mask]
    st.caption(f"Filtered: {len(filtered):,} / {len(df):,} rows")

    if filtered.empty:
        st.warning("No data matches the current filters.")
        return

    # ── Tabs ──────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", "⏰ Time Analysis", "📍 Location Analysis",
        "💳 Payment & Vehicle", "🔥 Heatmaps"
    ])

    # ── Tab 1: Overview ───────────────────────────────────────────────────
    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            # Ride status distribution
            status_counts = filtered["RideStatus"].value_counts().reset_index()
            status_counts.columns = ["RideStatus", "Count"]
            fig = px.pie(
                status_counts, values="Count", names="RideStatus",
                title="Ride Status Distribution",
                color_discrete_sequence=px.colors.qualitative.Set2,
                hole=0.4,
            )
            fig.update_traces(textinfo="percent+label")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Cancellation type breakdown
            cancel_df = filtered[filtered.get("IsCanceled", pd.Series()) == 1]
            if not cancel_df.empty:
                cancel_type = cancel_df["RideStatus"].value_counts().reset_index()
                cancel_type.columns = ["Type", "Count"]
                fig = px.bar(
                    cancel_type, x="Type", y="Count",
                    title="Cancellation Breakdown",
                    color="Type",
                    color_discrete_sequence=["#ff6b6b", "#ffa502"],
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No cancellations in filtered data.")

        # Booking value distribution
        if "BookingValue_INR" in filtered.columns:
            fig = px.histogram(
                filtered, x="BookingValue_INR", color="RideStatus",
                title="Booking Value Distribution by Status",
                nbins=50, barmode="overlay", opacity=0.7,
                color_discrete_sequence=["#1B9E3E", "#ff6b6b", "#ffa502"],
            )
            fig.update_layout(xaxis_title="Booking Value (₹)", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)

    # ── Tab 2: Time Analysis ──────────────────────────────────────────────
    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            if "BookingHour" in filtered.columns:
                hourly = filtered.groupby("BookingHour").agg(
                    total=("IsCanceled", "count"),
                    canceled=("IsCanceled", "sum"),
                ).reset_index()
                hourly["cancel_rate"] = (hourly["canceled"] / hourly["total"] * 100).round(2)

                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(
                    go.Bar(x=hourly["BookingHour"], y=hourly["total"],
                           name="Total Bookings", marker_color="#1B9E3E", opacity=0.6),
                    secondary_y=False,
                )
                fig.add_trace(
                    go.Scatter(x=hourly["BookingHour"], y=hourly["cancel_rate"],
                               name="Cancel Rate %", line=dict(color="#ff6b6b", width=3),
                               mode="lines+markers"),
                    secondary_y=True,
                )
                fig.update_layout(title="Hourly Booking Volume & Cancellation Rate",
                                  xaxis_title="Hour of Day")
                fig.update_yaxes(title_text="Bookings", secondary_y=False)
                fig.update_yaxes(title_text="Cancel Rate %", secondary_y=True)
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            if "DayOfWeek" in filtered.columns:
                day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                daily = filtered.groupby("DayOfWeek").agg(
                    total=("IsCanceled", "count"),
                    canceled=("IsCanceled", "sum"),
                ).reindex(day_order).reset_index()
                daily["cancel_rate"] = (daily["canceled"] / daily["total"] * 100).round(2)

                fig = px.bar(
                    daily, x="DayOfWeek", y="cancel_rate",
                    title="Cancellation Rate by Day of Week",
                    color="cancel_rate",
                    color_continuous_scale=["#1B9E3E", "#ffa502", "#ff6b6b"],
                )
                fig.update_layout(xaxis_title="Day", yaxis_title="Cancel Rate %")
                st.plotly_chart(fig, use_container_width=True)

        # Monthly trend
        if "Month" in filtered.columns and "BookingDate" in filtered.columns:
            monthly = filtered.groupby(filtered["BookingDate"].dt.to_period("M")).agg(
                total=("IsCanceled", "count"),
                canceled=("IsCanceled", "sum"),
            ).reset_index()
            monthly["BookingDate"] = monthly["BookingDate"].astype(str)
            monthly["cancel_rate"] = (monthly["canceled"] / monthly["total"] * 100).round(2)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=monthly["BookingDate"], y=monthly["total"],
                name="Total Bookings", fill="tozeroy",
                line=dict(color="#1B9E3E"),
            ))
            fig.add_trace(go.Scatter(
                x=monthly["BookingDate"], y=monthly["canceled"],
                name="Canceled", fill="tozeroy",
                line=dict(color="#ff6b6b"),
            ))
            fig.update_layout(title="Monthly Booking & Cancellation Trend",
                              xaxis_title="Month", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)

        # Time period analysis
        if "TimePeriod" in filtered.columns:
            tp = filtered.groupby("TimePeriod").agg(
                total=("IsCanceled", "count"),
                canceled=("IsCanceled", "sum"),
            ).reset_index()
            tp["cancel_rate"] = (tp["canceled"] / tp["total"] * 100).round(2)

            fig = px.bar(
                tp, x="TimePeriod", y=["total", "canceled"],
                title="Bookings by Time Period",
                barmode="group",
                color_discrete_sequence=["#1B9E3E", "#ff6b6b"],
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Tab 3: Location Analysis ──────────────────────────────────────────
    with tab3:
        if "PickupLocation" in filtered.columns:
            loc_stats = filtered.groupby("PickupLocation").agg(
                total=("IsCanceled", "count"),
                canceled=("IsCanceled", "sum"),
            ).reset_index()
            loc_stats["cancel_rate"] = (loc_stats["canceled"] / loc_stats["total"] * 100).round(2)
            loc_stats = loc_stats.sort_values("cancel_rate", ascending=True)

            col1, col2 = st.columns(2)
            with col1:
                fig = px.bar(
                    loc_stats, y="PickupLocation", x="cancel_rate",
                    title="Cancellation Rate by Pickup Location",
                    orientation="h",
                    color="cancel_rate",
                    color_continuous_scale=["#1B9E3E", "#ffa502", "#ff6b6b"],
                )
                fig.update_layout(yaxis_title="", xaxis_title="Cancel Rate %", height=600)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = px.bar(
                    loc_stats.sort_values("total", ascending=True),
                    y="PickupLocation", x="total",
                    title="Total Bookings by Location",
                    orientation="h",
                    color="total",
                    color_continuous_scale="Greens",
                )
                fig.update_layout(yaxis_title="", xaxis_title="Total Bookings", height=600)
                st.plotly_chart(fig, use_container_width=True)

            # Top routes
            if "DropLocation" in filtered.columns:
                routes = filtered.groupby(["PickupLocation", "DropLocation"]).agg(
                    total=("IsCanceled", "count"),
                    canceled=("IsCanceled", "sum"),
                ).reset_index()
                routes["cancel_rate"] = (routes["canceled"] / routes["total"] * 100).round(2)
                top_cancel_routes = routes.nlargest(15, "cancel_rate")

                fig = px.scatter(
                    top_cancel_routes, x="total", y="cancel_rate",
                    size="canceled", color="PickupLocation",
                    hover_data=["DropLocation"],
                    title="Top 15 High-Cancellation Routes (Pickup → Drop)",
                )
                fig.update_layout(xaxis_title="Total Rides", yaxis_title="Cancel Rate %")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No location data available.")

    # ── Tab 4: Payment & Vehicle ──────────────────────────────────────────
    with tab4:
        col1, col2 = st.columns(2)

        with col1:
            if "PaymentMode" in filtered.columns:
                pay_stats = filtered.groupby("PaymentMode").agg(
                    total=("IsCanceled", "count"),
                    canceled=("IsCanceled", "sum"),
                ).reset_index()
                pay_stats["cancel_rate"] = (pay_stats["canceled"] / pay_stats["total"] * 100).round(2)

                fig = px.bar(
                    pay_stats, x="PaymentMode", y="cancel_rate",
                    title="Cancellation Rate by Payment Mode",
                    color="cancel_rate",
                    color_continuous_scale=["#1B9E3E", "#ffa502", "#ff6b6b"],
                    text="cancel_rate",
                )
                fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                st.plotly_chart(fig, use_container_width=True)

                # Payment mode pie
                fig2 = px.pie(
                    pay_stats, values="total", names="PaymentMode",
                    title="Booking Share by Payment Mode",
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                )
                st.plotly_chart(fig2, use_container_width=True)

        with col2:
            if "VehicleType" in filtered.columns:
                veh_stats = filtered.groupby("VehicleType").agg(
                    total=("IsCanceled", "count"),
                    canceled=("IsCanceled", "sum"),
                    avg_fare=("BookingValue_INR", "mean"),
                ).reset_index()
                veh_stats["cancel_rate"] = (veh_stats["canceled"] / veh_stats["total"] * 100).round(2)

                fig = px.bar(
                    veh_stats, x="VehicleType", y="cancel_rate",
                    title="Cancellation Rate by Vehicle Type",
                    color="cancel_rate",
                    color_continuous_scale=["#1B9E3E", "#ffa502", "#ff6b6b"],
                    text="cancel_rate",
                )
                fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                st.plotly_chart(fig, use_container_width=True)

                # Avg fare by vehicle
                fig2 = px.bar(
                    veh_stats, x="VehicleType", y="avg_fare",
                    title="Average Fare by Vehicle Type",
                    color="VehicleType",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                    text=veh_stats["avg_fare"].round(0),
                )
                fig2.update_traces(texttemplate="₹%{text:,.0f}", textposition="outside")
                st.plotly_chart(fig2, use_container_width=True)

        # ETA vs Cancellation
        if "ETA_Pickup_min" in filtered.columns:
            fig = px.box(
                filtered, x="RideStatus", y="ETA_Pickup_min",
                title="ETA Distribution by Ride Status",
                color="RideStatus",
                color_discrete_sequence=["#1B9E3E", "#ff6b6b", "#ffa502"],
            )
            fig.update_layout(xaxis_title="", yaxis_title="ETA to Pickup (min)")
            st.plotly_chart(fig, use_container_width=True)

    # ── Tab 5: Heatmaps ──────────────────────────────────────────────────
    with tab5:
        if "BookingHour" in filtered.columns and "DayOfWeek" in filtered.columns and "IsCanceled" in filtered.columns:
            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            heatmap_data = filtered.groupby(["DayOfWeek", "BookingHour"])["IsCanceled"].mean().reset_index()
            heatmap_data["IsCanceled"] = (heatmap_data["IsCanceled"] * 100).round(2)
            heatmap_pivot = heatmap_data.pivot(index="DayOfWeek", columns="BookingHour", values="IsCanceled")
            heatmap_pivot = heatmap_pivot.reindex(day_order)

            fig = px.imshow(
                heatmap_pivot,
                title="Cancellation Rate Heatmap — Day of Week × Hour",
                labels=dict(x="Hour of Day", y="Day of Week", color="Cancel Rate %"),
                color_continuous_scale="RdYlGn_r",
                aspect="auto",
            )
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)

        # Location × Vehicle Type heatmap
        if "PickupLocation" in filtered.columns and "VehicleType" in filtered.columns and "IsCanceled" in filtered.columns:
            loc_veh = filtered.groupby(["PickupLocation", "VehicleType"])["IsCanceled"].mean().reset_index()
            loc_veh["IsCanceled"] = (loc_veh["IsCanceled"] * 100).round(2)
            loc_veh_pivot = loc_veh.pivot(index="PickupLocation", columns="VehicleType", values="IsCanceled")

            fig = px.imshow(
                loc_veh_pivot,
                title="Cancellation Rate — Location × Vehicle Type",
                labels=dict(x="Vehicle Type", y="Pickup Location", color="Cancel Rate %"),
                color_continuous_scale="RdYlGn_r",
                aspect="auto",
            )
            fig.update_layout(height=650)
            st.plotly_chart(fig, use_container_width=True)

        # Cancel reason breakdown
        if "CancelReason" in filtered.columns:
            reasons = filtered[filtered["CancelReason"].notna()]["CancelReason"].value_counts().head(10).reset_index()
            reasons.columns = ["Reason", "Count"]
            fig = px.bar(
                reasons, y="Reason", x="Count",
                title="Top 10 Cancellation Reasons",
                orientation="h",
                color="Count",
                color_continuous_scale="Reds",
            )
            fig.update_layout(yaxis_title="", height=400)
            st.plotly_chart(fig, use_container_width=True)
