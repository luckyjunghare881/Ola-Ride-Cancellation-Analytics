"""
Home & Overview Dashboard - KPI cards, charts, heatmaps, insights, map, table.
Fully functional with real data from the loaded dataset.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
from src.data_processing import load_csv, preprocess, compute_kpis, get_summary_stats
from src.generate_data import generate_dataset
from src.utils import df_to_csv_download

DARK_TEMPLATE = dict(
    layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#A0AEC0", size=12),
        title_font=dict(color="#FFFFFF", size=15),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.05)"),
        legend=dict(font=dict(color="#A0AEC0")),
        margin=dict(l=40, r=20, t=50, b=40),
    )
)


def _apply_dark(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#A0AEC0"),
        title_font=dict(color="#FFFFFF"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.05)"),
        legend=dict(font=dict(color="#A0AEC0")),
    )
    return fig


def render():
    # -------- TOPBAR --------
    st.markdown("""
    <div style="background:#16213E;border:1px solid rgba(255,255,255,0.08);border-radius:12px;padding:14px 28px;margin-bottom:20px;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;">
        <div>
            <div style="font-size:20px;font-weight:700;color:#FFF;">Cancellation Analytics</div>
            <div style="font-size:12px;color:#A0AEC0;">
                <svg style="width:12px;height:12px;vertical-align:middle;margin-right:3px" viewBox="0 0 24 24" fill="#A0AEC0"><path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7zm0 9.5c-1.38 0-2.5-1.12-2.5-2.5s1.12-2.5 2.5-2.5 2.5 1.12 2.5 2.5-1.12 2.5-2.5 2.5z"/></svg>
                Bangalore Metro Region &middot; Analytics Dashboard
            </div>
        </div>
        <div style="display:flex;gap:8px;align-items:center;">
            <div style="display:inline-flex;background:#0D0D1A;border:1px solid rgba(255,255,255,0.08);border-radius:10px;overflow:hidden;">
                <span style="padding:6px 14px;font-size:12px;color:#A0AEC0;">1D</span>
                <span style="padding:6px 14px;font-size:12px;color:#A0AEC0;">7D</span>
                <span style="padding:6px 14px;font-size:12px;background:#00C853;color:white;">30D</span>
                <span style="padding:6px 14px;font-size:12px;color:#A0AEC0;">90D</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # -------- UPLOAD / LOAD DATA --------
    if not st.session_state.data_loaded:
        st.markdown("""
        <div style="background:#16213E;border:1px solid rgba(255,255,255,0.08);border-radius:16px;padding:40px;text-align:center;">
            <div style="font-size:48px;margin-bottom:16px;">
                <svg viewBox="0 0 24 24" fill="#00C853" width="48" height="48"><path d="M19.35 10.04C18.67 6.59 15.64 4 12 4 9.11 4 6.6 5.64 5.35 8.04 2.34 8.36 0 10.91 0 14c0 3.31 2.69 6 6 6h13c2.76 0 5-2.24 5-5 0-2.64-2.05-4.78-4.65-4.96zM14 13v4h-4v-4H7l5-5 5 5h-3z"/></svg>
            </div>
            <div style="font-size:20px;font-weight:700;color:#FFF;margin-bottom:8px;">Upload Your Data</div>
            <div style="font-size:13px;color:#A0AEC0;margin-bottom:20px;">Upload a CSV file or click "Load Demo Data" in the sidebar to get started</div>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Upload OLA ride data (CSV)", type=["csv"], label_visibility="collapsed")
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
                st.rerun()

        st.markdown("""
        <div style="margin-top:20px;background:#16213E;border:1px solid rgba(255,255,255,0.08);border-radius:12px;padding:16px 20px;">
            <div style="font-size:14px;font-weight:600;color:#FFF;margin-bottom:8px;">
                <svg style="width:16px;height:16px;vertical-align:middle;margin-right:6px" viewBox="0 0 24 24" fill="#00C853"><path d="M11 7h2v2h-2zm0 4h2v6h-2zm1-9C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8z"/></svg>
                Expected Column Format
            </div>
            <div style="font-size:12px;color:#A0AEC0;font-family:monospace;line-height:1.8;">
                BookingID, BookingDate, RideStatus, VehicleType, PickupLocation,<br>
                DropLocation, PaymentMode, RideDistance_km, BookingValue_INR,<br>
                DriverRating, CustomerRating, ETA_Pickup_min, RideDuration_min, CancelReason
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # ============================================================
    # DATA IS LOADED - RENDER FULL DASHBOARD
    # ============================================================
    df = st.session_state.df_clean
    kpis = st.session_state.kpis
    report = st.session_state.cleaning_report

    # -------- ALERT BANNER --------
    # Find if there's a high-cancel zone
    alert_zone = ""
    alert_rate = 0
    if "PickupLocation" in df.columns and "IsCanceled" in df.columns:
        loc_rates = df.groupby("PickupLocation")["IsCanceled"].mean() * 100
        if len(loc_rates) > 0:
            alert_zone = loc_rates.idxmax()
            alert_rate = loc_rates.max()

    if alert_rate > 30:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,rgba(244,67,54,0.15),rgba(244,67,54,0.05));border:1px solid rgba(244,67,54,0.3);border-radius:12px;padding:14px 20px;margin-bottom:20px;display:flex;align-items:center;gap:12px;animation:pulse-border 2s infinite;">
            <div>
                <svg viewBox="0 0 24 24" fill="#FF7070" width="24" height="24"><path d="M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z"/></svg>
            </div>
            <div>
                <div style="font-size:14px;font-weight:600;color:#FF7070;">High Cancellation Detected &mdash; {alert_zone}</div>
                <div style="font-size:12px;color:#A0AEC0;margin-top:2px;">Cancellation rate at {alert_rate:.1f}% &mdash; significantly above average</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # -------- TABS --------
    tab_summary, tab_driver, tab_data, tab_export = st.tabs([
        "Summary", "Driver & Rider Analysis", "Data Preview", "Export"
    ])

    with tab_summary:
        _render_summary(df, kpis, report)

    with tab_driver:
        _render_driver_rider(df, kpis)

    with tab_data:
        _render_data_preview(df, kpis, report)

    with tab_export:
        _render_export(df, kpis)


def _kpi_card_html(label, value, icon_svg, color, change_text, change_dir="up"):
    change_color = "#F44336" if change_dir == "up" else "#4CAF50"
    arrow = "&#x2191;" if change_dir == "up" else "&#x2193;"
    gradient_map = {
        "red": "linear-gradient(90deg, #F44336, #FF7043)",
        "green": "linear-gradient(90deg, #00C853, #69F0AE)",
        "orange": "linear-gradient(90deg, #FF6B35, #FFB300)",
        "blue": "linear-gradient(90deg, #4FC3F7, #1976D2)",
    }
    icon_bg_map = {
        "red": "rgba(244,67,54,0.15)",
        "green": "rgba(0,200,83,0.15)",
        "orange": "rgba(255,107,53,0.15)",
        "blue": "rgba(79,195,247,0.15)",
    }
    value_color_map = {
        "red": "#FF7070",
        "green": "#00C853",
        "orange": "#FF8C61",
        "blue": "#4FC3F7",
    }
    return f"""
    <div style="background:#16213E;border:1px solid rgba(255,255,255,0.08);border-radius:16px;padding:22px;position:relative;overflow:hidden;transition:transform 0.2s,box-shadow 0.2s;cursor:default;" onmouseover="this.style.transform='translateY(-3px)';this.style.boxShadow='0 8px 30px rgba(0,0,0,0.3)'" onmouseout="this.style.transform='none';this.style.boxShadow='none'">
        <div style="position:absolute;top:0;left:0;right:0;height:3px;border-radius:16px 16px 0 0;background:{gradient_map.get(color, gradient_map['blue'])}"></div>
        <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:14px;">
            <div style="font-size:12px;color:#A0AEC0;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;">{label}</div>
            <div style="width:40px;height:40px;border-radius:10px;background:{icon_bg_map.get(color, icon_bg_map['blue'])};display:flex;align-items:center;justify-content:center;">
                {icon_svg}
            </div>
        </div>
        <div style="font-size:32px;font-weight:800;letter-spacing:-1px;margin-bottom:8px;color:{value_color_map.get(color, '#4FC3F7')};">{value}</div>
        <div style="display:flex;align-items:center;gap:6px;font-size:12px;font-weight:600;">
            <span style="color:{change_color};">{arrow} {change_text}</span>
            <span style="color:#A0AEC0;font-weight:400;">vs last period</span>
        </div>
    </div>
    """


def _render_summary(df, kpis, report):
    # -------- KPI CARDS --------
    cancel_icon = '<svg viewBox="0 0 24 24" fill="#FF7070" width="20" height="20"><path d="M12 2C6.47 2 2 6.47 2 12s4.47 10 10 10 10-4.47 10-10S17.53 2 12 2zm5 13.59L15.59 17 12 13.41 8.41 17 7 15.59 10.59 12 7 8.41 8.41 7 12 10.59 15.59 7 17 8.41 13.41 12 17 15.59z"/></svg>'
    rate_icon = '<svg viewBox="0 0 24 24" fill="#FF8C61" width="20" height="20"><path d="M16 6l2.29 2.29-4.88 4.88-4-4L2 16.59 3.41 18l6-6 4 4 6.3-6.29L22 12V6z"/></svg>'
    revenue_icon = '<svg viewBox="0 0 24 24" fill="#4FC3F7" width="20" height="20"><path d="M11.8 10.9c-2.27-.59-3-1.2-3-2.15 0-1.09 1.01-1.85 2.7-1.85 1.78 0 2.44.85 2.5 2.1h2.21c-.07-1.72-1.12-3.3-3.21-3.81V3h-3v2.16c-1.94.42-3.5 1.68-3.5 3.61 0 2.31 1.91 3.46 4.7 4.13 2.5.6 3 1.48 3 2.41 0 .69-.49 1.79-2.7 1.79-2.06 0-2.87-.92-2.98-2.1h-2.2c.12 2.19 1.76 3.42 3.68 3.83V21h3v-2.15c1.95-.37 3.5-1.5 3.5-3.55 0-2.84-2.43-3.81-4.7-4.4z"/></svg>'
    complete_icon = '<svg viewBox="0 0 24 24" fill="#00C853" width="20" height="20"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/></svg>'

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(_kpi_card_html(
            "Total Cancellations", f"{kpis['canceled_rides']:,}",
            cancel_icon, "red", f"{kpis['cancellation_rate']:.1f}%", "up"
        ), unsafe_allow_html=True)
    with c2:
        st.markdown(_kpi_card_html(
            "Cancellation Rate", f"{kpis['cancellation_rate']}%",
            rate_icon, "orange", f"{kpis['driver_cancel_rate']:.1f}% driver", "up"
        ), unsafe_allow_html=True)
    with c3:
        st.markdown(_kpi_card_html(
            "Revenue Lost (INR)", f"\u20b9{kpis['revenue_loss_est']:,.0f}",
            revenue_icon, "blue", "estimated loss", "up"
        ), unsafe_allow_html=True)
    with c4:
        completion = 100 - kpis['cancellation_rate']
        st.markdown(_kpi_card_html(
            "Completion Rate", f"{completion:.1f}%",
            complete_icon, "green", f"{kpis['completed_rides']:,} rides", "down"
        ), unsafe_allow_html=True)

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    # -------- CHARTS ROW 1: Bar + Donut --------
    col_bar, col_donut = st.columns([2, 1])

    with col_bar:
        st.markdown('<div style="background:#16213E;border:1px solid rgba(255,255,255,0.08);border-radius:16px;padding:22px;">', unsafe_allow_html=True)
        st.markdown("""
            <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px;">
                <div>
                    <div style="font-size:15px;font-weight:700;color:#FFF;">Daily Cancellation vs Completion Trend</div>
                    <div style="font-size:12px;color:#A0AEC0;margin-top:3px;">Aggregated by date</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        if "BookingDate" in df.columns:
            daily = df.groupby(df["BookingDate"].dt.date).agg(
                total=("IsCanceled", "count"),
                canceled=("IsCanceled", "sum"),
            ).reset_index()
            daily.columns = ["Date", "Total", "Canceled"]
            daily["Completed"] = daily["Total"] - daily["Canceled"]

            fig = go.Figure()
            fig.add_trace(go.Bar(x=daily["Date"], y=daily["Canceled"], name="Cancellations",
                                 marker_color="#F44336", opacity=0.85))
            fig.add_trace(go.Bar(x=daily["Date"], y=daily["Completed"], name="Completions",
                                 marker_color="#00C853", opacity=0.85))
            fig.update_layout(barmode="group", height=300,
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font=dict(color="#A0AEC0"), margin=dict(l=40,r=20,t=10,b=40),
                              legend=dict(orientation="h", y=-0.15, font=dict(color="#A0AEC0")),
                              xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                              yaxis=dict(gridcolor="rgba(255,255,255,0.05)"))
            st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_donut:
        st.markdown('<div style="background:#16213E;border:1px solid rgba(255,255,255,0.08);border-radius:16px;padding:22px;">', unsafe_allow_html=True)
        st.markdown("""
            <div style="margin-bottom:8px;">
                <div style="font-size:15px;font-weight:700;color:#FFF;">Cancellation By</div>
                <div style="font-size:12px;color:#A0AEC0;margin-top:3px;">Who cancelled the ride?</div>
            </div>
        """, unsafe_allow_html=True)

        if "RideStatus" in df.columns:
            status_counts = df["RideStatus"].value_counts().reset_index()
            status_counts.columns = ["Status", "Count"]
            colors = ["#4CAF50", "#F44336", "#FF9800", "#FFB300", "#7B1FA2"]
            fig = go.Figure(data=[go.Pie(
                labels=status_counts["Status"], values=status_counts["Count"],
                hole=0.55, marker_colors=colors[:len(status_counts)],
                textinfo="percent+label", textfont=dict(size=11, color="#FFF"),
            )])
            fig.update_layout(height=300,
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font=dict(color="#A0AEC0"), margin=dict(l=10,r=10,t=10,b=10),
                              showlegend=False,
                              annotations=[dict(text=f'{kpis["cancellation_rate"]}%', x=0.5, y=0.55,
                                                font_size=22, font_color="#FF7070", showarrow=False),
                                           dict(text="Cancel Rate", x=0.5, y=0.42,
                                                font_size=11, font_color="#A0AEC0", showarrow=False)])
            st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    # -------- CHARTS ROW 2: Reasons + Heatmap + Trend --------
    col_reasons, col_heat, col_trend = st.columns(3)

    with col_reasons:
        st.markdown('<div style="background:#16213E;border:1px solid rgba(255,255,255,0.08);border-radius:16px;padding:22px;">', unsafe_allow_html=True)
        st.markdown("""
            <div style="font-size:15px;font-weight:700;color:#FFF;margin-bottom:4px;">Top Cancellation Reasons</div>
            <div style="font-size:12px;color:#A0AEC0;margin-bottom:14px;">By frequency</div>
        """, unsafe_allow_html=True)

        if "CancelReason" in df.columns:
            reasons = df[df["CancelReason"].notna()]["CancelReason"].value_counts().head(5)
            total_reasons = reasons.sum()
            bar_colors = ["#F44336", "#FF6B35", "#FFB300", "#4FC3F7", "#9C27B0"]
            for i, (reason, count) in enumerate(reasons.items()):
                pct = count / total_reasons * 100
                color = bar_colors[i % len(bar_colors)]
                st.markdown(f"""
                <div style="margin-bottom:14px;">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                        <div style="font-size:12px;color:#A0AEC0;">{reason[:30]}</div>
                        <div style="font-size:13px;font-weight:700;color:{color};">{pct:.0f}%</div>
                    </div>
                    <div style="height:8px;background:rgba(255,255,255,0.06);border-radius:4px;overflow:hidden;">
                        <div style="height:100%;width:{pct}%;border-radius:4px;background:{color};"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown('<div style="color:#A0AEC0;font-size:12px;">No cancellation reason data</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_heat:
        st.markdown('<div style="background:#16213E;border:1px solid rgba(255,255,255,0.08);border-radius:16px;padding:22px;">', unsafe_allow_html=True)
        st.markdown("""
            <div style="font-size:15px;font-weight:700;color:#FFF;margin-bottom:4px;">Cancellation Heatmap</div>
            <div style="font-size:12px;color:#A0AEC0;margin-bottom:10px;">Hour x Day of Week</div>
        """, unsafe_allow_html=True)

        if "BookingHour" in df.columns and "DayOfWeek" in df.columns and "IsCanceled" in df.columns:
            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            hm = df.groupby(["DayOfWeek", "BookingHour"])["IsCanceled"].mean().reset_index()
            hm["IsCanceled"] = (hm["IsCanceled"] * 100).round(1)
            hm_pivot = hm.pivot(index="DayOfWeek", columns="BookingHour", values="IsCanceled")
            hm_pivot = hm_pivot.reindex(day_order)

            fig = px.imshow(hm_pivot, color_continuous_scale=[[0, "rgba(244,67,54,0.1)"], [0.5, "rgba(244,67,54,0.5)"], [1, "rgba(244,67,54,0.95)"]],
                            labels=dict(x="Hour", y="Day", color="Rate %"), aspect="auto")
            fig.update_layout(height=260,
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font=dict(color="#A0AEC0", size=10), margin=dict(l=70,r=10,t=10,b=30),
                              coloraxis_colorbar=dict(len=0.8, thickness=10))
            st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_trend:
        st.markdown('<div style="background:#16213E;border:1px solid rgba(255,255,255,0.08);border-radius:16px;padding:22px;">', unsafe_allow_html=True)
        st.markdown("""
            <div style="font-size:15px;font-weight:700;color:#FFF;margin-bottom:4px;">Weekly Cancel Rate Trend</div>
            <div style="font-size:12px;color:#A0AEC0;margin-bottom:10px;">Rolling average</div>
        """, unsafe_allow_html=True)

        if "BookingDate" in df.columns and "IsCanceled" in df.columns:
            weekly = df.set_index("BookingDate").resample("W")["IsCanceled"].mean().reset_index()
            weekly["Rate"] = (weekly["IsCanceled"] * 100).round(2)
            weekly["Week"] = range(1, len(weekly) + 1)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=weekly["Week"], y=weekly["Rate"], mode="lines+markers",
                line=dict(color="#F44336", width=2.5), marker=dict(size=4, color="#FF7070"),
                fill="tozeroy", fillcolor="rgba(244,67,54,0.1)", name="Cancel Rate %"
            ))
            fig.update_layout(height=200,
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font=dict(color="#A0AEC0"), margin=dict(l=40,r=10,t=10,b=30),
                              xaxis=dict(title="Week", gridcolor="rgba(255,255,255,0.05)"),
                              yaxis=dict(title="Rate %", gridcolor="rgba(255,255,255,0.05)"),
                              showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            peak_week = weekly.loc[weekly["Rate"].idxmax()]
            st.markdown(f"""
            <div style="padding:10px;background:rgba(244,67,54,0.05);border-radius:8px;border:1px solid rgba(244,67,54,0.1);">
                <div style="font-size:12px;color:#A0AEC0;">
                    <svg style="width:12px;height:12px;vertical-align:middle;margin-right:3px" viewBox="0 0 24 24" fill="#FF7070"><path d="M16 6l2.29 2.29-4.88 4.88-4-4L2 16.59 3.41 18l6-6 4 4 6.3-6.29L22 12V6z"/></svg>
                    Peak week detected
                </div>
                <div style="font-size:13px;font-weight:700;color:#FF7070;margin-top:4px;">Week {int(peak_week['Week'])} &mdash; {peak_week['Rate']:.1f}% cancel rate</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    # -------- Compute alert_zone for Key Insights --------
    alert_zone = ""
    alert_rate = 0.0
    if "PickupLocation" in df.columns and "IsCanceled" in df.columns:
        loc_rates = df.groupby("PickupLocation")["IsCanceled"].mean() * 100
        if len(loc_rates) > 0:
            alert_zone = loc_rates.idxmax()
            alert_rate = loc_rates.max()

    # -------- KEY INSIGHTS --------
    st.markdown("""
        <div style="font-size:15px;font-weight:700;color:#FFF;margin-bottom:14px;">
            <svg style="width:16px;height:16px;vertical-align:middle;margin-right:6px" viewBox="0 0 24 24" fill="#FFB300"><path d="M9 21c0 .55.45 1 1 1h4c.55 0 1-.45 1-1v-1H9v1zm3-19C8.14 2 5 5.14 5 9c0 2.38 1.19 4.47 3 5.74V17c0 .55.45 1 1 1h6c.55 0 1-.45 1-1v-2.26c1.81-1.27 3-3.36 3-5.74 0-3.86-3.14-7-7-7z"/></svg>
            Key Insights
        </div>
    """, unsafe_allow_html=True)

    ic1, ic2, ic3 = st.columns(3)

    # Insight 1: Peak cancel window
    peak_window = "N/A"
    if "BookingHour" in df.columns and "IsCanceled" in df.columns:
        hourly_cancel = df.groupby("BookingHour")["IsCanceled"].mean() * 100
        peak_hour = hourly_cancel.idxmax()
        peak_window = f"{peak_hour}:00 - {(peak_hour + 3) % 24}:00"
        peak_pct = hourly_cancel.max()

    with ic1:
        st.markdown(f"""
        <div style="background:#16213E;border:1px solid rgba(255,255,255,0.08);border-radius:16px;padding:20px;position:relative;overflow:hidden;">
            <div style="position:absolute;top:-30px;right:-30px;width:100px;height:100px;border-radius:50%;background:#F44336;opacity:0.05;"></div>
            <div style="font-size:28px;margin-bottom:10px;">
                <svg viewBox="0 0 24 24" fill="#FF7070" width="28" height="28"><path d="M11.99 2C6.47 2 2 6.48 2 12s4.47 10 9.99 10C17.52 22 22 17.52 22 12S17.52 2 11.99 2zM12 20c-4.42 0-8-3.58-8-8s3.58-8 8-8 8 3.58 8 8-3.58 8-8 8zm.5-13H11v6l5.25 3.15.75-1.23-4.5-2.67z"/></svg>
            </div>
            <div style="font-size:13px;color:#A0AEC0;margin-bottom:6px;">Peak Cancel Window</div>
            <div style="font-size:22px;font-weight:800;color:#FF7070;margin-bottom:8px;">{peak_window}</div>
            <div style="font-size:12px;color:#A0AEC0;line-height:1.5;">Peak hours account for highest cancellation rates in the dataset.</div>
            <div style="display:inline-block;margin-top:10px;padding:4px 10px;border-radius:20px;font-size:11px;font-weight:600;background:rgba(244,67,54,0.15);color:#FF7070;">
                <svg style="width:10px;height:10px;vertical-align:middle;margin-right:3px" viewBox="0 0 24 24" fill="#FF7070"><path d="M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z"/></svg>
                High Risk
            </div>
        </div>
        """, unsafe_allow_html=True)

    with ic2:
        if alert_zone:
            st.markdown(f"""
            <div style="background:#16213E;border:1px solid rgba(255,255,255,0.08);border-radius:16px;padding:20px;position:relative;overflow:hidden;">
                <div style="position:absolute;top:-30px;right:-30px;width:100px;height:100px;border-radius:50%;background:#FFB300;opacity:0.05;"></div>
                <div style="font-size:28px;margin-bottom:10px;">
                    <svg viewBox="0 0 24 24" fill="#FFB300" width="28" height="28"><path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7zm0 9.5c-1.38 0-2.5-1.12-2.5-2.5s1.12-2.5 2.5-2.5 2.5 1.12 2.5 2.5-1.12 2.5-2.5 2.5z"/></svg>
                </div>
                <div style="font-size:13px;color:#A0AEC0;margin-bottom:6px;">Highest Cancel Zone</div>
                <div style="font-size:22px;font-weight:800;color:#FFB300;margin-bottom:8px;">{alert_zone}</div>
                <div style="font-size:12px;color:#A0AEC0;line-height:1.5;">This zone shows {alert_rate:.1f}% cancellation rate.</div>
                <div style="display:inline-block;margin-top:10px;padding:4px 10px;border-radius:20px;font-size:11px;font-weight:600;background:rgba(255,179,0,0.15);color:#FFB300;">
                    <svg style="width:10px;height:10px;vertical-align:middle;margin-right:3px" viewBox="0 0 24 24" fill="#FFB300"><path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7z"/></svg>
                    Zone Alert
                </div>
            </div>
            """, unsafe_allow_html=True)

    with ic3:
        st.markdown(f"""
        <div style="background:#16213E;border:1px solid rgba(255,255,255,0.08);border-radius:16px;padding:20px;position:relative;overflow:hidden;">
            <div style="position:absolute;top:-30px;right:-30px;width:100px;height:100px;border-radius:50%;background:#00C853;opacity:0.05;"></div>
            <div style="font-size:28px;margin-bottom:10px;">
                <svg viewBox="0 0 24 24" fill="#00C853" width="28" height="28"><path d="M9 21c0 .55.45 1 1 1h4c.55 0 1-.45 1-1v-1H9v1zm3-19C8.14 2 5 5.14 5 9c0 2.38 1.19 4.47 3 5.74V17c0 .55.45 1 1 1h6c.55 0 1-.45 1-1v-2.26c1.81-1.27 3-3.36 3-5.74 0-3.86-3.14-7-7-7z"/></svg>
            </div>
            <div style="font-size:13px;color:#A0AEC0;margin-bottom:6px;">Top Recommendation</div>
            <div style="font-size:22px;font-weight:800;color:#00C853;margin-bottom:8px;">Surge Incentive</div>
            <div style="font-size:12px;color:#A0AEC0;line-height:1.5;">Implementing driver surge bonus during peak hours can reduce cancellations.</div>
            <div style="display:inline-block;margin-top:10px;padding:4px 10px;border-radius:20px;font-size:11px;font-weight:600;background:rgba(0,200,83,0.15);color:#00C853;">
                <svg style="width:10px;height:10px;vertical-align:middle;margin-right:3px" viewBox="0 0 24 24" fill="#00C853"><path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/></svg>
                Action Ready
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    # -------- GEOGRAPHIC HOTSPOT MAP --------
    _render_hotspot_map(df)

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    # -------- RECENT CANCELLATIONS TABLE --------
    _render_recent_table(df)


def _render_hotspot_map(df):
    """Render a geographic hotspot visualization using location data."""
    if "PickupLocation" not in df.columns or "IsCanceled" not in df.columns:
        return

    loc_stats = df.groupby("PickupLocation").agg(
        total=("IsCanceled", "count"),
        canceled=("IsCanceled", "sum"),
    ).reset_index()
    loc_stats["rate"] = (loc_stats["canceled"] / loc_stats["total"] * 100).round(1)
    loc_stats = loc_stats.sort_values("rate", ascending=False).head(10)

    st.markdown("""
    <div style="background:#16213E;border:1px solid rgba(255,255,255,0.08);border-radius:16px;padding:22px;">
        <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:14px;">
            <div>
                <div style="font-size:15px;font-weight:700;color:#FFF;">
                    <svg style="width:16px;height:16px;vertical-align:middle;margin-right:6px" viewBox="0 0 24 24" fill="#4FC3F7"><path d="M20.5 3l-.16.03L15 5.1 9 3 3.36 4.9c-.21.07-.36.25-.36.48V20.5c0 .28.22.5.5.5l.16-.03L9 18.9l6 2.1 5.64-1.9c.21-.07.36-.25.36-.48V3.5c0-.28-.22-.5-.5-.5zM15 19l-6-2.11V5l6 2.11V19z"/></svg>
                    Cancellation Hotspot Map &mdash; Locations
                </div>
                <div style="font-size:12px;color:#A0AEC0;margin-top:3px;">Bubble size = cancellation volume &middot; Color = cancellation rate</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    fig = px.scatter(
        loc_stats, x="total", y="rate", size="canceled",
        color="rate", color_continuous_scale=[[0, "#00C853"], [0.5, "#FFB300"], [1, "#F44336"]],
        hover_name="PickupLocation", hover_data={"total": True, "canceled": True, "rate": ":.1f"},
        labels={"total": "Total Rides", "rate": "Cancel Rate %", "canceled": "Cancellations"},
        size_max=40,
    )
    for _, row in loc_stats.iterrows():
        fig.add_annotation(x=row["total"], y=row["rate"], text=row["PickupLocation"],
                           showarrow=False, font=dict(size=9, color="#FFF"), yshift=15)
    fig.update_layout(height=300,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#A0AEC0"), margin=dict(l=50,r=20,t=10,b=40),
                      xaxis=dict(title="Total Rides", gridcolor="rgba(255,255,255,0.05)"),
                      yaxis=dict(title="Cancel Rate %", gridcolor="rgba(255,255,255,0.05)"),
                      showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


def _render_recent_table(df):
    """Render recent cancellation events table."""
    if "IsCanceled" not in df.columns:
        return

    canceled_df = df[df["IsCanceled"] == 1].tail(10).iloc[::-1]
    if canceled_df.empty:
        return

    total_canceled = df["IsCanceled"].sum()

    st.markdown(f"""
    <div style="background:#16213E;border:1px solid rgba(255,255,255,0.08);border-radius:16px;padding:22px;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:18px;flex-wrap:wrap;gap:10px;">
            <div>
                <div style="font-size:15px;font-weight:700;color:#FFF;">Recent Cancellation Events</div>
                <div style="font-size:12px;color:#A0AEC0;margin-top:3px;">Showing latest 10 of {total_canceled:,} cancelled records</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    display_cols = []
    for c in ["BookingID", "RideStatus", "PickupLocation", "VehicleType", "PaymentMode",
              "BookingValue_INR", "ETA_Pickup_min", "CancelReason"]:
        if c in canceled_df.columns:
            display_cols.append(c)

    if display_cols:
        table_html = '<table style="width:100%;border-collapse:collapse;">'
        table_html += '<thead><tr>'
        for col in display_cols:
            table_html += f'<th style="font-size:11px;font-weight:600;color:#A0AEC0;text-transform:uppercase;letter-spacing:0.7px;padding:10px 14px;text-align:left;border-bottom:1px solid rgba(255,255,255,0.08);">{col}</th>'
        table_html += '</tr></thead><tbody>'

        for _, row in canceled_df.iterrows():
            table_html += '<tr style="border-bottom:1px solid rgba(255,255,255,0.04);">'
            for col in display_cols:
                val = row[col]
                style = "padding:12px 14px;font-size:13px;color:#A0AEC0;"
                if col == "RideStatus":
                    if "Driver" in str(val):
                        style += "color:#FF7070;"
                    elif "Customer" in str(val):
                        style += "color:#FFB74D;"
                    val = f'<span style="background:rgba(244,67,54,0.15);color:#FF7070;padding:4px 10px;border-radius:20px;font-size:11px;font-weight:600;">{val}</span>'
                elif col == "BookingValue_INR":
                    val = f"\u20b9{val:,.0f}" if pd.notna(val) else "-"
                    style += "font-weight:600;color:#FFF;"
                elif col == "CancelReason":
                    val = f'<span style="background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.08);padding:3px 10px;border-radius:20px;font-size:11px;">{val}</span>' if pd.notna(val) else "-"
                elif col == "ETA_Pickup_min":
                    if pd.notna(val) and float(val) > 10:
                        style += "color:#FF7070;font-weight:600;"
                    val = f"{val:.0f} min" if pd.notna(val) else "-"
                elif col == "BookingID":
                    style += "font-family:monospace;font-size:12px;"
                table_html += f'<td style="{style}">{val}</td>'
            table_html += '</tr>'
        table_html += '</tbody></table>'
        st.markdown(table_html, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


def _render_driver_rider(df, kpis):
    """Driver & Rider analysis tab."""
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div style="background:#16213E;border:1px solid rgba(255,255,255,0.08);border-radius:16px;padding:22px;margin-bottom:18px;">', unsafe_allow_html=True)
        st.markdown('<div style="font-size:15px;font-weight:700;color:#FFF;margin-bottom:14px;">Driver Cancellation Analysis</div>', unsafe_allow_html=True)

        driver_df = df[df["RideStatus"].str.contains("Driver", case=False, na=False)] if "RideStatus" in df.columns else pd.DataFrame()

        if not driver_df.empty and "PickupLocation" in driver_df.columns:
            loc_counts = driver_df["PickupLocation"].value_counts().head(8).reset_index()
            loc_counts.columns = ["Location", "Count"]
            fig = px.bar(loc_counts, y="Location", x="Count", orientation="h",
                         color="Count", color_continuous_scale=["#F44336", "#FF7043"],
                         title="Driver Cancellations by Location")
            _apply_dark(fig)
            fig.update_layout(height=300, margin=dict(l=100,r=20,t=40,b=30), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No driver cancellation data available.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div style="background:#16213E;border:1px solid rgba(255,255,255,0.08);border-radius:16px;padding:22px;margin-bottom:18px;">', unsafe_allow_html=True)
        st.markdown('<div style="font-size:15px;font-weight:700;color:#FFF;margin-bottom:14px;">Customer Cancellation Analysis</div>', unsafe_allow_html=True)

        customer_df = df[df["RideStatus"].str.contains("Customer", case=False, na=False)] if "RideStatus" in df.columns else pd.DataFrame()

        if not customer_df.empty and "PickupLocation" in customer_df.columns:
            loc_counts = customer_df["PickupLocation"].value_counts().head(8).reset_index()
            loc_counts.columns = ["Location", "Count"]
            fig = px.bar(loc_counts, y="Location", x="Count", orientation="h",
                         color="Count", color_continuous_scale=["#FF9800", "#FFB74D"],
                         title="Customer Cancellations by Location")
            _apply_dark(fig)
            fig.update_layout(height=300, margin=dict(l=100,r=20,t=40,b=30), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No customer cancellation data available.")
        st.markdown('</div>', unsafe_allow_html=True)

    # Rating distribution
    if "DriverRating" in df.columns:
        st.markdown('<div style="background:#16213E;border:1px solid rgba(255,255,255,0.08);border-radius:16px;padding:22px;margin-bottom:18px;">', unsafe_allow_html=True)
        st.markdown('<div style="font-size:15px;font-weight:700;color:#FFF;margin-bottom:14px;">Rating Distribution by Ride Status</div>', unsafe_allow_html=True)

        fig = px.box(df, x="RideStatus", y="DriverRating", color="RideStatus",
                     color_discrete_sequence=["#4CAF50", "#F44336", "#FF9800"],
                     title="Driver Rating Distribution")
        _apply_dark(fig)
        fig.update_layout(height=300, margin=dict(l=40,r=20,t=40,b=30))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


def _render_data_preview(df, kpis, report):
    """Data preview and cleaning report."""
    # Cleaning report
    with st.expander("Data Cleaning Report", expanded=False):
        if report:
            c1, c2 = st.columns(2)
            with c1:
                st.write(f"**Original rows:** {report.get('original_rows', 'N/A'):,}")
                st.write(f"**Final rows:** {report.get('final_rows', 'N/A'):,}")
                st.write(f"**Duplicates removed:** {report.get('duplicates_removed', 0):,}")
            with c2:
                if report.get("missing_filled"):
                    st.write("**Missing values handled:**")
                    for col, desc in report["missing_filled"].items():
                        st.write(f"  - `{col}`: {desc}")
                if report.get("columns_added"):
                    st.write(f"**Features added:** {', '.join(report['columns_added'])}")

    # Data table
    page_size = st.selectbox("Rows per page", [25, 50, 100, 250], index=0)
    total_pages = max(1, (len(df) - 1) // page_size + 1)
    page_num = st.number_input("Page", min_value=1, max_value=total_pages, value=1)

    start_idx = (page_num - 1) * page_size
    end_idx = min(start_idx + page_size, len(df))
    st.dataframe(df.iloc[start_idx:end_idx], use_container_width=True, height=400)
    st.caption(f"Showing rows {start_idx + 1} - {end_idx} of {len(df):,}")

    # Summary stats
    with st.expander("Summary Statistics", expanded=False):
        stats = get_summary_stats(df)
        if not stats.empty:
            st.dataframe(stats, use_container_width=True)


def _render_export(df, kpis):
    """Export section."""
    st.markdown("""
    <div style="background:#16213E;border:1px solid rgba(255,255,255,0.08);border-radius:16px;padding:22px;">
        <div style="font-size:15px;font-weight:700;color:#FFF;margin-bottom:14px;">
            <svg style="width:16px;height:16px;vertical-align:middle;margin-right:6px" viewBox="0 0 24 24" fill="#4FC3F7"><path d="M19 9h-4V3H9v6H5l7 7 7-7zM5 18v2h14v-2H5z"/></svg>
            Export Data
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        csv_bytes = df_to_csv_download(df, "ola_rides_cleaned.csv")
        st.download_button("Download Cleaned CSV", data=csv_bytes,
                           file_name="ola_rides_cleaned.csv", mime="text/csv",
                           use_container_width=True)
    with col2:
        if "IsCanceled" in df.columns:
            canceled_only = df[df["IsCanceled"] == 1]
            csv_cancel = df_to_csv_download(canceled_only, "canceled_rides.csv")
            st.download_button("Download Canceled Rides Only", data=csv_cancel,
                               file_name="canceled_rides.csv", mime="text/csv",
                               use_container_width=True)
