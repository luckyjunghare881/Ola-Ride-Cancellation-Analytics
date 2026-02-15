"""
Utility functions: PDF/CSV export, helpers, and shared components.
"""

import pandas as pd
import io
import base64
from typing import Optional


def df_to_csv_download(df: pd.DataFrame, filename: str = "export.csv") -> bytes:
    """Convert DataFrame to CSV bytes for download."""
    return df.to_csv(index=False).encode("utf-8")


def generate_pdf_report(kpis: dict, recommendations: list, summary: str) -> bytes:
    """
    Generate a simple text-based report as downloadable content.
    Uses plain text format (no heavy PDF library dependency).
    """
    lines = []
    lines.append("=" * 70)
    lines.append("OLA RIDE CANCELLATION ANALYTICS — REPORT")
    lines.append("=" * 70)
    lines.append("")

    # KPIs
    lines.append("KEY PERFORMANCE INDICATORS")
    lines.append("-" * 40)
    for key, val in kpis.items():
        label = key.replace("_", " ").title()
        if isinstance(val, float):
            lines.append(f"  {label}: {val:,.2f}")
        else:
            lines.append(f"  {label}: {val:,}")
    lines.append("")

    # Recommendations
    lines.append("RECOMMENDATIONS")
    lines.append("-" * 40)
    for i, rec in enumerate(recommendations, 1):
        lines.append(f"\n  [{rec['priority']}] {rec['category']}")
        lines.append(f"  Insight: {rec['insight']}")
        lines.append(f"  Action: {rec['recommendation']}")
        lines.append(f"  Impact: {rec['impact']}")
    lines.append("")

    # Summary
    lines.append("EXECUTIVE SUMMARY")
    lines.append("-" * 40)
    # Strip markdown formatting for plain text
    clean_summary = summary.replace("**", "").replace("##", "").replace("---", "")
    lines.append(clean_summary)

    lines.append("")
    lines.append("=" * 70)
    lines.append("End of Report")

    return "\n".join(lines).encode("utf-8")


def format_metric(value, prefix: str = "", suffix: str = "", decimals: int = 1) -> str:
    """Format a metric value for display."""
    if isinstance(value, (int, np.integer)):
        return f"{prefix}{value:,}{suffix}"
    elif isinstance(value, float):
        return f"{prefix}{value:,.{decimals}f}{suffix}"
    return f"{prefix}{value}{suffix}"


# Need numpy for format_metric
import numpy as np
