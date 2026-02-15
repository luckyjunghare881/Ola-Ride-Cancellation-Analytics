"""
Pre-compute all analytics data and save as JSON files.
Run this script before deploying to Vercel.
"""
import json
import os
import sys
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing import preprocess, compute_kpis
from src.generate_data import generate_dataset
from src.sql_engine import SQLEngine
from src.ml_model import train_model, predict_single, cluster_cancellation_hotspots
from src.recommendations import generate_recommendations, generate_executive_summary

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUT_DIR, exist_ok=True)


def save_json(data, filename):
    path = os.path.join(OUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"  Saved {filename} ({os.path.getsize(path) / 1024:.1f} KB)")


def main():
    print("Generating dataset...")
    df = generate_dataset(10000)
    df_clean, report = preprocess(df)
    kpis = compute_kpis(df_clean)

    print("Computing KPIs...")
    save_json(kpis, "kpis.json")

    print("Computing EDA data...")
    save_json(df_clean["RideStatus"].value_counts().to_dict(), "status_dist.json")

    hourly = df_clean.groupby("BookingHour").agg(
        total=("IsCanceled", "count"), canceled=("IsCanceled", "sum"),
    ).reset_index()
    hourly["cancel_rate"] = (hourly["canceled"] / hourly["total"] * 100).round(2)
    save_json(json.loads(hourly.to_json(orient="records")), "hourly.json")

    loc = df_clean.groupby("PickupLocation").agg(
        total=("IsCanceled", "count"), canceled=("IsCanceled", "sum"),
    ).reset_index()
    loc["cancel_rate"] = (loc["canceled"] / loc["total"] * 100).round(2)
    save_json(json.loads(loc.sort_values("cancel_rate", ascending=False).to_json(orient="records")), "location.json")

    pay = df_clean.groupby("PaymentMode").agg(
        total=("IsCanceled", "count"), canceled=("IsCanceled", "sum"),
    ).reset_index()
    pay["cancel_rate"] = (pay["canceled"] / pay["total"] * 100).round(2)
    save_json(json.loads(pay.to_json(orient="records")), "payment.json")

    veh = df_clean.groupby("VehicleType").agg(
        total=("IsCanceled", "count"), canceled=("IsCanceled", "sum"),
        avg_fare=("BookingValue_INR", "mean"),
    ).reset_index()
    veh["cancel_rate"] = (veh["canceled"] / veh["total"] * 100).round(2)
    veh["avg_fare"] = veh["avg_fare"].round(2)
    save_json(json.loads(veh.to_json(orient="records")), "vehicle.json")

    daily = df_clean.groupby("DayOfWeek").agg(
        total=("IsCanceled", "count"), canceled=("IsCanceled", "sum"),
    ).reset_index()
    daily["cancel_rate"] = (daily["canceled"] / daily["total"] * 100).round(2)
    save_json(json.loads(daily.to_json(orient="records")), "daily.json")

    reasons = df_clean[df_clean["CancelReason"].notna()]["CancelReason"].value_counts().head(10)
    save_json({r: int(c) for r, c in reasons.items()}, "reasons.json")

    save_json(json.loads(df_clean.head(50).to_json(orient="records", date_format="iso")), "sample.json")

    print("Computing SQL query results...")
    engine = SQLEngine()
    engine.load_data(df_clean)
    queries = engine.get_prebuilt_queries()
    sql_queries = {n: info["description"] for n, info in queries.items()}
    save_json(sql_queries, "sql_queries.json")

    sql_results = {}
    for name, info in queries.items():
        result, error = engine.execute_query(info["sql"])
        if not error and not result.empty:
            sql_results[name] = {
                "rows": len(result),
                "columns": result.columns.tolist(),
                "data": json.loads(result.to_json(orient="records")),
            }
    save_json(sql_results, "sql_results.json")

    print("Training ML model...")
    ml_result = train_model(df_clean, model_type="random_forest")
    save_json({
        "metrics": ml_result["metrics"],
        "feature_importance": json.loads(ml_result["feature_importance"].to_json(orient="records")),
    }, "ml_metrics.json")

    # Save model as pickle for predictions
    model_data = {
        "model": ml_result["model"],
        "encoders": ml_result["encoders"],
        "features": ml_result["features"],
    }
    model_path = os.path.join(OUT_DIR, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)
    print(f"  Saved model.pkl ({os.path.getsize(model_path) / 1024:.1f} KB)")

    clusters = cluster_cancellation_hotspots(df_clean, n_clusters=5)
    save_json(json.loads(clusters.to_json(orient="records")) if not clusters.empty else [], "clusters.json")

    print("Generating recommendations...")
    recs = generate_recommendations(df_clean, kpis)
    summary = generate_executive_summary(kpis, recs)
    save_json({
        "recommendations": recs,
        "executive_summary": summary,
        "kpis": kpis,
    }, "recommendations.json")

    print("\n✅ All data pre-computed successfully!")
    print(f"   Files in {OUT_DIR}:")
    for f in sorted(os.listdir(OUT_DIR)):
        size = os.path.getsize(os.path.join(OUT_DIR, f))
        print(f"   - {f} ({size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
