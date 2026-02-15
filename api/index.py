"""
Vercel Serverless API — OLA Ride Cancellation Analytics
FastAPI backend serving data, KPIs, SQL queries, ML predictions, and recommendations.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing import preprocess, compute_kpis, get_summary_stats
from src.generate_data import generate_dataset
from src.ml_model import train_model, predict_single, cluster_cancellation_hotspots
from src.sql_engine import SQLEngine
from src.recommendations import generate_recommendations, generate_executive_summary

app = FastAPI(
    title="OLA Ride Cancellation Analytics API",
    description="API for ride cancellation analytics, ML predictions, and recommendations",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Cached data (generated once per cold start) ─────────────────────────
_cache = {}


def _get_data():
    """Generate and cache the demo dataset."""
    if "df" not in _cache:
        df = generate_dataset(12000)
        df_clean, report = preprocess(df)
        kpis = compute_kpis(df_clean)
        _cache["df"] = df_clean
        _cache["report"] = report
        _cache["kpis"] = kpis
    return _cache["df"], _cache.get("report", {}), _cache.get("kpis", {})


def _get_sql_engine():
    """Get or create SQL engine."""
    if "sql_engine" not in _cache:
        df, _, _ = _get_data()
        engine = SQLEngine()
        engine.load_data(df)
        _cache["sql_engine"] = engine
    return _cache["sql_engine"]


def _get_ml_result():
    """Get or train ML model."""
    if "ml_result" not in _cache:
        df, _, _ = _get_data()
        _cache["ml_result"] = train_model(df)
    return _cache["ml_result"]


# ── Routes ───────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    """Serve the dashboard HTML page."""
    html_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "public", "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return JSONResponse({
        "app": "OLA Ride Cancellation Analytics",
        "status": "running",
        "endpoints": ["/api/kpis", "/api/data", "/api/sql", "/api/predict", "/api/recommendations"],
    })


@app.get("/api/health")
async def health():
    return {"status": "ok", "app": "OLA Ride Cancellation Analytics"}


@app.get("/api/kpis")
async def get_kpis():
    """Return all computed KPIs."""
    _, _, kpis = _get_data()
    return kpis


@app.get("/api/data/summary")
async def data_summary():
    """Return dataset summary statistics."""
    df, report, kpis = _get_data()
    stats = get_summary_stats(df)
    return {
        "shape": {"rows": len(df), "columns": len(df.columns)},
        "columns": df.columns.tolist(),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "cleaning_report": report,
        "kpis": kpis,
        "summary_stats": stats.to_dict() if not stats.empty else {},
    }


@app.get("/api/data/sample")
async def data_sample(limit: int = Query(default=50, le=500)):
    """Return a sample of the dataset."""
    df, _, _ = _get_data()
    sample = df.head(limit)
    return json.loads(sample.to_json(orient="records", date_format="iso"))


@app.get("/api/eda/status-distribution")
async def eda_status_distribution():
    """Ride status distribution."""
    df, _, _ = _get_data()
    dist = df["RideStatus"].value_counts().to_dict()
    return {"status_distribution": dist}


@app.get("/api/eda/hourly")
async def eda_hourly():
    """Hourly booking and cancellation pattern."""
    df, _, _ = _get_data()
    hourly = df.groupby("BookingHour").agg(
        total=("IsCanceled", "count"),
        canceled=("IsCanceled", "sum"),
    ).reset_index()
    hourly["cancel_rate"] = (hourly["canceled"] / hourly["total"] * 100).round(2)
    return json.loads(hourly.to_json(orient="records"))


@app.get("/api/eda/location")
async def eda_location():
    """Cancellation rate by pickup location."""
    df, _, _ = _get_data()
    loc = df.groupby("PickupLocation").agg(
        total=("IsCanceled", "count"),
        canceled=("IsCanceled", "sum"),
    ).reset_index()
    loc["cancel_rate"] = (loc["canceled"] / loc["total"] * 100).round(2)
    return json.loads(loc.sort_values("cancel_rate", ascending=False).to_json(orient="records"))


@app.get("/api/eda/payment")
async def eda_payment():
    """Cancellation rate by payment mode."""
    df, _, _ = _get_data()
    pay = df.groupby("PaymentMode").agg(
        total=("IsCanceled", "count"),
        canceled=("IsCanceled", "sum"),
    ).reset_index()
    pay["cancel_rate"] = (pay["canceled"] / pay["total"] * 100).round(2)
    return json.loads(pay.to_json(orient="records"))


@app.get("/api/eda/vehicle")
async def eda_vehicle():
    """Cancellation rate by vehicle type."""
    df, _, _ = _get_data()
    veh = df.groupby("VehicleType").agg(
        total=("IsCanceled", "count"),
        canceled=("IsCanceled", "sum"),
        avg_fare=("BookingValue_INR", "mean"),
    ).reset_index()
    veh["cancel_rate"] = (veh["canceled"] / veh["total"] * 100).round(2)
    veh["avg_fare"] = veh["avg_fare"].round(2)
    return json.loads(veh.to_json(orient="records"))


@app.get("/api/eda/daily")
async def eda_daily():
    """Cancellation rate by day of week."""
    df, _, _ = _get_data()
    daily = df.groupby("DayOfWeek").agg(
        total=("IsCanceled", "count"),
        canceled=("IsCanceled", "sum"),
    ).reset_index()
    daily["cancel_rate"] = (daily["canceled"] / daily["total"] * 100).round(2)
    return json.loads(daily.to_json(orient="records"))


@app.get("/api/eda/cancel-reasons")
async def eda_cancel_reasons():
    """Top cancellation reasons."""
    df, _, _ = _get_data()
    reasons = df[df["CancelReason"].notna()]["CancelReason"].value_counts().head(10)
    return {r: int(c) for r, c in reasons.items()}


@app.get("/api/sql/prebuilt")
async def sql_prebuilt_list():
    """List all pre-built SQL queries."""
    engine = _get_sql_engine()
    queries = engine.get_prebuilt_queries()
    return {name: info["description"] for name, info in queries.items()}


@app.get("/api/sql/run")
async def sql_run(query_name: str = Query(default=None), custom: str = Query(default=None)):
    """Run a pre-built or custom SQL query."""
    engine = _get_sql_engine()

    if query_name:
        queries = engine.get_prebuilt_queries()
        if query_name not in queries:
            raise HTTPException(status_code=404, detail=f"Query '{query_name}' not found")
        sql = queries[query_name]["sql"]
    elif custom:
        sql = custom
    else:
        raise HTTPException(status_code=400, detail="Provide query_name or custom SQL")

    result, error = engine.execute_query(sql)
    if error:
        raise HTTPException(status_code=400, detail=error)

    return {
        "rows": len(result),
        "columns": result.columns.tolist(),
        "data": json.loads(result.to_json(orient="records")),
    }


@app.get("/api/ml/metrics")
async def ml_metrics():
    """Return ML model training metrics."""
    result = _get_ml_result()
    return {
        "metrics": result["metrics"],
        "feature_importance": json.loads(result["feature_importance"].to_json(orient="records")),
    }


@app.get("/api/ml/predict")
async def ml_predict(
    booking_hour: int = Query(default=18, ge=0, le=23),
    ride_distance: float = Query(default=8.0, gt=0),
    booking_value: float = Query(default=150.0, gt=0),
    driver_rating: float = Query(default=4.0, ge=1, le=5),
    customer_rating: float = Query(default=4.0, ge=1, le=5),
    eta_pickup: float = Query(default=8.0, gt=0),
    is_weekend: bool = Query(default=False),
    vehicle_type: str = Query(default="Mini"),
    payment_mode: str = Query(default="UPI"),
    pickup_location: str = Query(default="Koramangala"),
):
    """Predict cancellation probability for a single booking."""
    result = _get_ml_result()
    prediction = predict_single(
        result["model"], result["encoders"], result["features"],
        booking_hour, ride_distance, booking_value,
        driver_rating, customer_rating, eta_pickup,
        is_weekend, vehicle_type, payment_mode, pickup_location,
    )
    return prediction


@app.get("/api/ml/clusters")
async def ml_clusters(n_clusters: int = Query(default=5, ge=2, le=10)):
    """Run KMeans clustering on cancellation hotspots."""
    df, _, _ = _get_data()
    clusters = cluster_cancellation_hotspots(df, n_clusters=n_clusters)
    if clusters.empty:
        return {"clusters": []}
    return {"clusters": json.loads(clusters.to_json(orient="records"))}


@app.get("/api/recommendations")
async def get_recommendations():
    """Generate AI-powered recommendations."""
    df, _, kpis = _get_data()
    recs = generate_recommendations(df, kpis)
    summary = generate_executive_summary(kpis, recs)
    return {
        "recommendations": recs,
        "executive_summary": summary,
        "kpis": kpis,
    }
