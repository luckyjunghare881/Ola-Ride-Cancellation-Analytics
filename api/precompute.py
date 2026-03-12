"""
Pre-compute all analytics data and save as JSON files.
Run this script before deploying to Vercel.
"""
import json
import os
import sys
import pickle
import pandas as pd

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


def _serialize_tree(tree):
    """Serialize a single sklearn DecisionTree to a JSON-safe nested dict."""
    t = tree.tree_
    def _node(i):
        if t.children_left[i] == -1:  # leaf
            counts = t.value[i][0].tolist()
            total = sum(counts)
            return {"leaf": True, "proba": [c / total for c in counts]}
        return {
            "leaf": False,
            "feature": int(t.feature[i]),
            "threshold": float(t.threshold[i]),
            "left": _node(int(t.children_left[i])),
            "right": _node(int(t.children_right[i])),
        }
    return _node(0)


def _serialize_rf_model(model, encoders, features):
    """Serialize RandomForest model + encoders to a JSON-safe dict."""
    trees = [_serialize_tree(est) for est in model.estimators_]
    enc_map = {}
    for src, le in encoders.items():
        enc_map[src] = {cls: int(idx) for idx, cls in enumerate(le.classes_)}
    return {
        "features": features,
        "n_classes": int(model.n_classes_),
        "trees": trees,
        "encoders": enc_map,
    }


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

    # Full dataset (compact) for client-side time filtering
    print("Saving compact full dataset for time filtering...")
    compact_cols = ["BookingDate","RideStatus","VehicleType","PickupLocation","PaymentMode",
                    "RideDistance_km","BookingValue_INR","DriverRating","CustomerRating",
                    "BookingHour","DayOfWeek","Month","IsCanceled","CancelReason"]
    full_compact = df_clean[compact_cols].copy()
    full_compact["BookingDate"] = pd.to_datetime(full_compact["BookingDate"]).dt.strftime("%Y-%m-%d")
    save_json(json.loads(full_compact.to_json(orient="records")), "full_data.json")

    # Heatmap: Hour x DayOfWeek cancel rates
    print("Computing heatmap data...")
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    hm = df_clean.groupby(["DayOfWeek", "BookingHour"])["IsCanceled"].mean().reset_index()
    hm["IsCanceled"] = (hm["IsCanceled"] * 100).round(1)
    hm_out = {}
    for day in day_order:
        day_data = hm[hm["DayOfWeek"] == day].sort_values("BookingHour")
        hm_out[day] = {int(r["BookingHour"]): r["IsCanceled"] for _, r in day_data.iterrows()}
    save_json(hm_out, "heatmap.json")

    # Monthly trend
    print("Computing monthly trend...")
    month_order = ["January","February","March","April","May","June","July","August","September","October","November","December"]
    monthly = df_clean.groupby("Month").agg(total=("IsCanceled","count"), canceled=("IsCanceled","sum")).reset_index()
    monthly["cancel_rate"] = (monthly["canceled"]/monthly["total"]*100).round(2)
    monthly["Month"] = pd.Categorical(monthly["Month"], categories=month_order, ordered=True)
    monthly = monthly.sort_values("Month")
    save_json(json.loads(monthly.to_json(orient="records")), "monthly.json")

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

    # Save model as pickle for local usage
    model_data = {
        "model": ml_result["model"],
        "encoders": ml_result["encoders"],
        "features": ml_result["features"],
    }
    model_path = os.path.join(OUT_DIR, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)
    print(f"  Saved model.pkl ({os.path.getsize(model_path) / 1024:.1f} KB)")

    # Export model as JSON for lightweight Vercel deployment (no sklearn needed)
    model_json = _serialize_rf_model(
        ml_result["model"], ml_result["encoders"], ml_result["features"]
    )
    save_json(model_json, "model_light.json")
    print(f"  Saved model_light.json for Vercel (pure-Python inference)")

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
