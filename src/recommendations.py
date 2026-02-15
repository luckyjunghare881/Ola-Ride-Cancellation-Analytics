"""
AI Recommendations engine.
Generates natural-language business insights from KPIs and data patterns.
No external LLM dependency — uses rule-based + template system.
"""

import pandas as pd
import numpy as np
from typing import Dict, List


def generate_recommendations(df: pd.DataFrame, kpis: Dict) -> List[Dict]:
    """
    Analyze data patterns and generate actionable recommendations.
    Returns list of {category, priority, insight, recommendation, impact}.
    """
    recommendations = []

    if df.empty:
        return [{"category": "Data", "priority": "High",
                 "insight": "No data available for analysis.",
                 "recommendation": "Upload a valid CSV dataset to begin.",
                 "impact": "N/A"}]

    # ── 1. Overall cancellation rate ──────────────────────────────────────
    cancel_rate = kpis.get("cancellation_rate", 0)
    if cancel_rate > 35:
        recommendations.append({
            "category": "🚨 Critical Alert",
            "priority": "Critical",
            "insight": f"Cancellation rate is critically high at {cancel_rate}% — well above the industry benchmark of 15-20%.",
            "recommendation": "Immediately investigate top cancellation reasons. Implement driver incentive programs and customer loyalty rewards to reduce cancellations by at least 10%.",
            "impact": f"Potential revenue recovery: ₹{kpis.get('revenue_loss_est', 0):,.0f} estimated lost revenue.",
        })
    elif cancel_rate > 25:
        recommendations.append({
            "category": "⚠️ High Cancellation Rate",
            "priority": "High",
            "insight": f"Cancellation rate of {cancel_rate}% exceeds acceptable threshold.",
            "recommendation": "Focus on the top 3 cancellation reasons and implement targeted interventions. Consider dynamic pricing adjustments during high-cancellation periods.",
            "impact": f"Reducing cancellation by 5% could recover ~₹{kpis.get('revenue_loss_est', 0) * 0.15:,.0f} in revenue.",
        })

    # ── 2. Driver vs Customer cancellations ───────────────────────────────
    driver_rate = kpis.get("driver_cancel_rate", 0)
    customer_rate = kpis.get("customer_cancel_rate", 0)

    if driver_rate > customer_rate * 1.5:
        recommendations.append({
            "category": "🚗 Driver Behavior",
            "priority": "High",
            "insight": f"Driver cancellations ({driver_rate}%) significantly exceed customer cancellations ({customer_rate}%). Drivers are the primary source of ride failures.",
            "recommendation": "Implement driver accountability metrics: penalize frequent cancellers, reward consistent drivers with bonuses and priority ride allocation. Consider mandatory acceptance windows.",
            "impact": "Targeting driver cancellations can reduce overall cancelation rate by 8-12%.",
        })
    elif customer_rate > driver_rate * 1.3:
        recommendations.append({
            "category": "👤 Customer Retention",
            "priority": "High",
            "insight": f"Customer cancellations ({customer_rate}%) are notably higher than driver cancellations ({driver_rate}%).",
            "recommendation": "Improve ETA accuracy, offer real-time driver tracking, and implement cancellation fee waivers for first-time cancellations. Add incentives (discount codes) for completing rides.",
            "impact": "Better customer experience can reduce customer cancellations by 15-20%.",
        })

    # ── 3. Peak hour analysis ─────────────────────────────────────────────
    if "BookingHour" in df.columns and "IsCanceled" in df.columns:
        hourly = df.groupby("BookingHour")["IsCanceled"].mean() * 100
        peak_hours = hourly[hourly > hourly.mean() + hourly.std()].index.tolist()
        if peak_hours:
            peak_str = ", ".join([f"{h}:00" for h in sorted(peak_hours)])
            max_rate = hourly[peak_hours].max()
            recommendations.append({
                "category": "⏰ Peak Hour Strategy",
                "priority": "High",
                "insight": f"Cancellation spikes at {peak_str} (up to {max_rate:.1f}%). These are likely rush-hour periods with supply-demand mismatch.",
                "recommendation": f"Deploy surge-pricing buffers and increase driver availability during {peak_str}. Offer guaranteed ride incentives (₹20-50 discount) to customers who wait during peak hours.",
                "impact": "Peak-hour optimization can reduce 20-30% of all cancellations.",
            })

    # ── 4. Location hotspots ──────────────────────────────────────────────
    if "PickupLocation" in df.columns and "IsCanceled" in df.columns:
        loc_cancel = df.groupby("PickupLocation").agg(
            total=("IsCanceled", "count"),
            canceled=("IsCanceled", "sum"),
        )
        loc_cancel["rate"] = loc_cancel["canceled"] / loc_cancel["total"] * 100
        hotspots = loc_cancel[loc_cancel["rate"] > loc_cancel["rate"].mean() + loc_cancel["rate"].std()]
        if not hotspots.empty:
            top_3 = hotspots.nlargest(3, "rate")
            loc_str = ", ".join(top_3.index.tolist())
            recommendations.append({
                "category": "📍 Location Hotspots",
                "priority": "Medium",
                "insight": f"High cancellation zones: {loc_str} (rates: {', '.join([f'{r:.1f}%' for r in top_3['rate']])})",
                "recommendation": f"Position more drivers near {loc_str} during peak hours. Investigate infrastructure issues (poor GPS, difficult pickup points) at these locations. Consider designated pickup spots.",
                "impact": "Location-targeted interventions typically reduce local cancellations by 25-35%.",
            })

    # ── 5. Payment mode insights ──────────────────────────────────────────
    if "PaymentMode" in df.columns and "IsCanceled" in df.columns:
        pay_cancel = df.groupby("PaymentMode")["IsCanceled"].mean() * 100
        worst_payment = pay_cancel.idxmax()
        best_payment = pay_cancel.idxmin()
        if pay_cancel.max() - pay_cancel.min() > 5:
            recommendations.append({
                "category": "💳 Payment Optimization",
                "priority": "Medium",
                "insight": f"'{worst_payment}' payments have highest cancellation rate ({pay_cancel.max():.1f}%), while '{best_payment}' has lowest ({pay_cancel.min():.1f}%).",
                "recommendation": f"Promote {best_payment} payment with cashback offers. Investigate friction in {worst_payment} payment flow. Consider pre-authorization for high-cancel payment modes.",
                "impact": "Payment method optimization can reduce cancellations by 3-5%.",
            })

    # ── 6. ETA impact ─────────────────────────────────────────────────────
    if "ETA_Pickup_min" in df.columns and "IsCanceled" in df.columns:
        high_eta = df[df["ETA_Pickup_min"] > 15]
        low_eta = df[df["ETA_Pickup_min"] <= 5]
        if len(high_eta) > 0 and len(low_eta) > 0:
            high_cancel_rate = high_eta["IsCanceled"].mean() * 100
            low_cancel_rate = low_eta["IsCanceled"].mean() * 100
            if high_cancel_rate > low_cancel_rate * 1.3:
                recommendations.append({
                    "category": "🕐 ETA Optimization",
                    "priority": "High",
                    "insight": f"High ETA (>15 min) rides cancel at {high_cancel_rate:.1f}% vs {low_cancel_rate:.1f}% for low ETA (≤5 min). Long wait times drive cancellations.",
                    "recommendation": "Optimize driver positioning algorithms to reduce ETAs. Show accurate real-time ETAs. Offer wait-time compensation (₹10-20) for ETAs exceeding 10 minutes.",
                    "impact": "Reducing average ETA by 3 minutes could lower cancellations by 8-12%.",
                })

    # ── 7. Vehicle type analysis ──────────────────────────────────────────
    if "VehicleType" in df.columns and "IsCanceled" in df.columns:
        veh_cancel = df.groupby("VehicleType")["IsCanceled"].mean() * 100
        if veh_cancel.max() - veh_cancel.min() > 5:
            worst_veh = veh_cancel.idxmax()
            recommendations.append({
                "category": "🚘 Vehicle Strategy",
                "priority": "Medium",
                "insight": f"'{worst_veh}' rides have highest cancellation rate ({veh_cancel.max():.1f}%). Supply-demand mismatch likely.",
                "recommendation": f"Increase {worst_veh} vehicle supply or offer alternatives. Implement automatic vehicle-type upgrades when {worst_veh} availability is low.",
                "impact": "Vehicle rebalancing can improve completion rates by 5-8%.",
            })

    # ── 8. Weekend vs Weekday ─────────────────────────────────────────────
    if "IsWeekend" in df.columns and "IsCanceled" in df.columns:
        weekend_rate = df[df["IsWeekend"] == True]["IsCanceled"].mean() * 100
        weekday_rate = df[df["IsWeekend"] == False]["IsCanceled"].mean() * 100
        if abs(weekend_rate - weekday_rate) > 3:
            higher = "weekends" if weekend_rate > weekday_rate else "weekdays"
            recommendations.append({
                "category": "📅 Day Pattern",
                "priority": "Low",
                "insight": f"Cancellations are higher on {higher} ({max(weekend_rate, weekday_rate):.1f}% vs {min(weekend_rate, weekday_rate):.1f}%).",
                "recommendation": f"Adjust driver scheduling to ensure adequate coverage on {higher}. Run targeted promotions to balance demand.",
                "impact": "Day-adjusted supply planning can reduce cancellations by 3-5%.",
            })

    # ── 9. Revenue impact summary ─────────────────────────────────────────
    rev_loss = kpis.get("revenue_loss_est", 0)
    if rev_loss > 0:
        recommendations.append({
            "category": "💰 Revenue Recovery",
            "priority": "Critical",
            "insight": f"Estimated revenue loss from cancellations: ₹{rev_loss:,.0f}. This represents significant unrealized income.",
            "recommendation": "Implement a phased cancellation reduction program targeting 20% improvement over 3 months. Focus on top-3 cancellation reasons, peak-hour optimization, and driver incentives.",
            "impact": f"A 20% reduction in cancellations could recover ~₹{rev_loss * 0.20:,.0f} in revenue.",
        })

    # ── 10. General best practices ────────────────────────────────────────
    recommendations.append({
        "category": "📊 Continuous Monitoring",
        "priority": "Ongoing",
        "insight": "Regular monitoring of cancellation KPIs is essential for sustained improvement.",
        "recommendation": "Set up daily/weekly cancellation dashboards. Track cancellation rate, driver response time, and customer satisfaction scores. A/B test interventions before full rollout.",
        "impact": "Data-driven decision making typically improves operational metrics by 10-15% annually.",
    })

    return recommendations


def generate_executive_summary(kpis: Dict, recommendations: List[Dict]) -> str:
    """Generate an executive summary paragraph from KPIs and recommendations."""
    critical = [r for r in recommendations if r["priority"] in ("Critical", "High")]
    n_critical = len(critical)

    summary = f"""
## Executive Summary — OLA Ride Cancellation Analytics

**Period Analyzed:** Full dataset | **Total Bookings:** {kpis.get('total_bookings', 0):,}

The overall cancellation rate stands at **{kpis.get('cancellation_rate', 0)}%**, with 
**{kpis.get('canceled_rides', 0):,}** rides canceled out of {kpis.get('total_bookings', 0):,} bookings. 
Driver-initiated cancellations account for **{kpis.get('driver_cancel_rate', 0)}%** and customer-initiated 
for **{kpis.get('customer_cancel_rate', 0)}%** of all bookings.

**Estimated Revenue Impact:** ₹{kpis.get('revenue_loss_est', 0):,.0f} in potential revenue lost to cancellations.

**{n_critical} high-priority action items** have been identified. Key focus areas include:
"""
    for i, rec in enumerate(critical[:5], 1):
        summary += f"\n{i}. **{rec['category']}** — {rec['recommendation'][:120]}..."

    summary += f"""

**Average Metrics:** Booking Value ₹{kpis.get('avg_booking_value', 0):,.0f} | 
Ride Distance {kpis.get('avg_ride_distance', 0):.1f} km | Driver Rating {kpis.get('avg_driver_rating', 0):.1f}/5 | 
ETA to Pickup {kpis.get('avg_eta_pickup', 0):.1f} min

---
*Report generated by OLA Ride Cancellation Analytics Dashboard*
"""
    return summary
