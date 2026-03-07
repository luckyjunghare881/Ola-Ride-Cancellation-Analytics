"""
Recommendations page — AI-generated insights and business recommendations.
"""

import streamlit as st
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from src.recommendations import generate_recommendations, generate_executive_summary
from src.utils import generate_pdf_report


def render():
    st.markdown("## 💡 AI-Powered Recommendations")

    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first from the Home page or click 'Load Demo Data' in the sidebar.")
        return

    df = st.session_state.df_clean
    kpis = st.session_state.kpis

    # Generate recommendations
    recommendations = generate_recommendations(df, kpis)
    summary = generate_executive_summary(kpis, recommendations)

    # ── Executive Summary ─────────────────────────────────────────────────
    with st.expander("📄 Executive Summary", expanded=True):
        st.markdown(summary)

    st.markdown("---")

    # ── Priority filter ───────────────────────────────────────────────────
    priorities = list(set(r["priority"] for r in recommendations))
    selected_priorities = st.multiselect(
        "Filter by Priority",
        options=priorities,
        default=priorities,
    )

    filtered_recs = [r for r in recommendations if r["priority"] in selected_priorities]

    # ── Recommendation cards ──────────────────────────────────────────────
    st.markdown(f"### Action Items ({len(filtered_recs)} recommendations)")

    for i, rec in enumerate(filtered_recs, 1):
        priority = rec["priority"]
        if priority == "Critical":
            border_color = "#ff4444"
            bg_color = "rgba(255,68,68,0.08)"
        elif priority == "High":
            border_color = "#ffa502"
            bg_color = "rgba(255,165,2,0.08)"
        elif priority == "Medium":
            border_color = "#1B9E3E"
            bg_color = "rgba(27,158,62,0.08)"
        else:
            border_color = "#4a90d9"
            bg_color = "rgba(74,144,217,0.08)"

        st.markdown(
            f"""
            <div style="
                border-left: 4px solid {border_color};
                background: {bg_color};
                padding: 1.2rem 1.5rem;
                border-radius: 0 10px 10px 0;
                margin-bottom: 1rem;
            ">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-size:1.15rem; font-weight:600;">{rec['category']}</span>
                    <span style="
                        background:{border_color};
                        color:white;
                        padding:2px 12px;
                        border-radius:12px;
                        font-size:0.8rem;
                        font-weight:600;
                    ">{priority}</span>
                </div>
                <div style="margin-top:0.6rem; color:#ccc;">
                    <strong>📊 Insight:</strong> {rec['insight']}
                </div>
                <div style="margin-top:0.5rem; color:#eee;">
                    <strong>✅ Action:</strong> {rec['recommendation']}
                </div>
                <div style="margin-top:0.5rem; color:#1B9E3E; font-weight:500;">
                    <strong>📈 Expected Impact:</strong> {rec['impact']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Export section ────────────────────────────────────────────────────
    st.markdown("### 📥 Export Report")

    col1, col2 = st.columns(2)
    with col1:
        report_bytes = generate_pdf_report(kpis, recommendations, summary)
        st.download_button(
            "📥 Download Full Report (TXT)",
            data=report_bytes,
            file_name="ola_cancellation_report.txt",
            mime="text/plain",
            use_container_width=True,
        )
    with col2:
        # Recommendations as CSV
        rec_df = pd.DataFrame(recommendations)
        csv_bytes = rec_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Recommendations (CSV)",
            data=csv_bytes,
            file_name="recommendations.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # ── Quick action plan ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🎯 30-Day Action Plan")

    critical_recs = [r for r in recommendations if r["priority"] in ("Critical", "High")]

    if critical_recs:
        st.markdown("**Week 1-2: Immediate Actions**")
        for i, rec in enumerate(critical_recs[:3], 1):
            st.markdown(f"  {i}. {rec['recommendation'][:150]}")

        st.markdown("\n**Week 3-4: Monitor & Optimize**")
        for i, rec in enumerate(critical_recs[3:6], 4):
            st.markdown(f"  {i}. {rec['recommendation'][:150]}")

        st.markdown("\n**Ongoing: Track KPIs**")
        st.markdown("  - Monitor daily cancellation rate (target: <20%)")
        st.markdown("  - Track driver response times and ETA accuracy")
        st.markdown("  - Review weekly customer feedback on cancellations")
        st.markdown("  - A/B test interventions before full rollout")
    else:
        st.success("No critical issues found. Continue monitoring KPIs.")
