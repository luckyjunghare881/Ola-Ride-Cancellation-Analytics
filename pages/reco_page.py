"""
Recommendations page - AI-generated insights and business recommendations.
Dark theme styled.
"""

import streamlit as st
import pandas as pd
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from src.recommendations import generate_recommendations, generate_executive_summary
from src.utils import generate_pdf_report


def render():
    st.markdown("""
    <div style="background:#16213E;border:1px solid rgba(255,255,255,0.08);border-radius:12px;padding:14px 28px;margin-bottom:20px;display:flex;align-items:center;gap:12px;">
        <svg viewBox="0 0 24 24" fill="#FFB300" width="24" height="24"><path d="M9 21c0 .55.45 1 1 1h4c.55 0 1-.45 1-1v-1H9v1zm3-19C8.14 2 5 5.14 5 9c0 2.38 1.19 4.47 3 5.74V17c0 .55.45 1 1 1h6c.55 0 1-.45 1-1v-2.26c1.81-1.27 3-3.36 3-5.74 0-3.86-3.14-7-7-7z"/></svg>
        <div>
            <div style="font-size:20px;font-weight:700;color:#FFF;">AI-Powered Recommendations</div>
            <div style="font-size:12px;color:#A0AEC0;">Business insights and actionable recommendations</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.data_loaded:
        st.warning("Please load data first from the Overview Dashboard or click 'Load Demo Data' in the sidebar.")
        return

    df = st.session_state.df_clean
    kpis = st.session_state.kpis

    recommendations = generate_recommendations(df, kpis)
    summary = generate_executive_summary(kpis, recommendations)

    # Executive Summary
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,rgba(0,200,83,0.1),rgba(33,150,243,0.1));
        border:1px solid rgba(0,200,83,0.3);border-radius:16px;padding:22px 28px;margin-bottom:20px;">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;">
            <svg viewBox="0 0 24 24" fill="#00C853" width="20" height="20"><path d="M14 2H6c-1.1 0-2 .9-2 2v16c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V8l-6-6zm2 16H8v-2h8v2zm0-4H8v-2h8v2zm-3-5V3.5L18.5 9H13z"/></svg>
            <span style="font-size:16px;font-weight:700;color:#FFF;">Executive Summary</span>
        </div>
        <div style="color:#A0AEC0;font-size:13px;line-height:1.8;">{summary.replace(chr(10), '<br>')}</div>
    </div>
    """, unsafe_allow_html=True)

    # Priority filter
    priorities = list(set(r["priority"] for r in recommendations))
    selected_priorities = st.multiselect("Filter by Priority", options=sorted(priorities), default=sorted(priorities))
    filtered_recs = [r for r in recommendations if r["priority"] in selected_priorities]

    # Summary badges
    crit = sum(1 for r in filtered_recs if r["priority"]=="Critical")
    high = sum(1 for r in filtered_recs if r["priority"]=="High")
    med = sum(1 for r in filtered_recs if r["priority"]=="Medium")
    low = sum(1 for r in filtered_recs if r["priority"]=="Low")
    st.markdown(f"""
    <div style="display:flex;gap:12px;margin-bottom:18px;flex-wrap:wrap;">
        <span style="background:rgba(244,67,54,0.15);color:#F44336;padding:4px 14px;border-radius:20px;font-size:13px;font-weight:600;">Critical: {crit}</span>
        <span style="background:rgba(255,179,0,0.15);color:#FFB300;padding:4px 14px;border-radius:20px;font-size:13px;font-weight:600;">High: {high}</span>
        <span style="background:rgba(0,200,83,0.15);color:#00C853;padding:4px 14px;border-radius:20px;font-size:13px;font-weight:600;">Medium: {med}</span>
        <span style="background:rgba(33,150,243,0.15);color:#2196F3;padding:4px 14px;border-radius:20px;font-size:13px;font-weight:600;">Low: {low}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f'<div style="font-size:15px;font-weight:600;color:#FFF;margin-bottom:10px;">Action Items ({len(filtered_recs)} recommendations)</div>', unsafe_allow_html=True)

    for i, rec in enumerate(filtered_recs, 1):
        priority = rec["priority"]
        if priority == "Critical":
            border_color, bg_color, icon_path = "#F44336", "rgba(244,67,54,0.06)", "M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"
        elif priority == "High":
            border_color, bg_color, icon_path = "#FFB300", "rgba(255,179,0,0.06)", "M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z"
        elif priority == "Medium":
            border_color, bg_color, icon_path = "#00C853", "rgba(0,200,83,0.06)", "M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"
        else:
            border_color, bg_color, icon_path = "#2196F3", "rgba(33,150,243,0.06)", "M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z"

        st.markdown(f"""
        <div style="border-left:4px solid {border_color};background:{bg_color};padding:16px 20px;
            border-radius:0 12px 12px 0;margin-bottom:10px;">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <div style="display:flex;align-items:center;gap:8px;">
                    <svg viewBox="0 0 24 24" fill="{border_color}" width="18" height="18"><path d="{icon_path}"/></svg>
                    <span style="font-size:15px;font-weight:600;color:#FFF;">{rec['category']}</span>
                </div>
                <span style="background:{border_color};color:#FFF;padding:2px 14px;border-radius:12px;font-size:12px;font-weight:600;">{priority}</span>
            </div>
            <div style="margin-top:8px;color:#A0AEC0;font-size:13px;">
                <svg viewBox="0 0 24 24" fill="#A0AEC0" width="14" height="14" style="vertical-align:middle;margin-right:4px;"><path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm-5 14H7v-2h7v2zm3-4H7v-2h10v2zm0-4H7V7h10v2z"/></svg>
                <strong>Insight:</strong> {rec['insight']}
            </div>
            <div style="margin-top:6px;color:#E0E0E0;font-size:13px;">
                <svg viewBox="0 0 24 24" fill="#00C853" width="14" height="14" style="vertical-align:middle;margin-right:4px;"><path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/></svg>
                <strong>Action:</strong> {rec['recommendation']}
            </div>
            <div style="margin-top:6px;color:#00C853;font-size:13px;font-weight:500;">
                <svg viewBox="0 0 24 24" fill="#00C853" width="14" height="14" style="vertical-align:middle;margin-right:4px;"><path d="M16 6l2.29 2.29-4.88 4.88-4-4L2 16.59 3.41 18l6-6 4 4 6.3-6.29L22 12V6z"/></svg>
                <strong>Expected Impact:</strong> {rec['impact']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Export section
    st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;">
        <svg viewBox="0 0 24 24" fill="#2196F3" width="20" height="20"><path d="M19 9h-4V3H9v6H5l7 7 7-7zM5 18v2h14v-2H5z"/></svg>
        <span style="font-size:15px;font-weight:600;color:#FFF;">Export Report</span>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        report_bytes = generate_pdf_report(kpis, recommendations, summary)
        st.download_button("Download Full Report (TXT)", data=report_bytes,
            file_name="ola_cancellation_report.txt", mime="text/plain", use_container_width=True)
    with col2:
        rec_df = pd.DataFrame(recommendations)
        csv_bytes = rec_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Recommendations (CSV)", data=csv_bytes,
            file_name="recommendations.csv", mime="text/csv", use_container_width=True)

    # 30-Day Action Plan
    st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#16213E;border:1px solid rgba(255,255,255,0.08);border-radius:16px;padding:22px 28px;">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:16px;">
            <svg viewBox="0 0 24 24" fill="#FFB300" width="20" height="20"><path d="M19 3h-1V1h-2v2H8V1H6v2H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm0 16H5V8h14v11z"/></svg>
            <span style="font-size:16px;font-weight:700;color:#FFF;">30-Day Action Plan</span>
        </div>
    """, unsafe_allow_html=True)

    critical_recs = [r for r in recommendations if r["priority"] in ("Critical","High")]

    if critical_recs:
        st.markdown('<div style="color:#FFB300;font-weight:600;font-size:14px;margin-bottom:8px;">Week 1-2: Immediate Actions</div>', unsafe_allow_html=True)
        for i, rec in enumerate(critical_recs[:3], 1):
            st.markdown(f'<div style="color:#A0AEC0;font-size:13px;margin-left:12px;margin-bottom:4px;">{i}. {rec["recommendation"][:150]}</div>', unsafe_allow_html=True)

        st.markdown('<div style="color:#2196F3;font-weight:600;font-size:14px;margin-top:12px;margin-bottom:8px;">Week 3-4: Monitor & Optimize</div>', unsafe_allow_html=True)
        for i, rec in enumerate(critical_recs[3:6], 4):
            st.markdown(f'<div style="color:#A0AEC0;font-size:13px;margin-left:12px;margin-bottom:4px;">{i}. {rec["recommendation"][:150]}</div>', unsafe_allow_html=True)

        st.markdown("""
        <div style="color:#00C853;font-weight:600;font-size:14px;margin-top:12px;margin-bottom:8px;">Ongoing: Track KPIs</div>
        <div style="color:#A0AEC0;font-size:13px;margin-left:12px;line-height:1.8;">
            - Monitor daily cancellation rate (target: &lt;20%)<br>
            - Track driver response times and ETA accuracy<br>
            - Review weekly customer feedback on cancellations<br>
            - A/B test interventions before full rollout
        </div>
        """, unsafe_allow_html=True)
    else:
        st.success("No critical issues found. Continue monitoring KPIs.")

    st.markdown('</div>', unsafe_allow_html=True)
