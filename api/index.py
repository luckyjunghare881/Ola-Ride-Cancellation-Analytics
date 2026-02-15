"""
Vercel Serverless API — OLA Ride Cancellation Analytics
Serves pre-computed JSON data. No heavy ML/data libraries needed at runtime.
Only requires: fastapi (lightweight).
Prediction uses pickle-loaded sklearn model (lazy-loaded on demand).
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

app = FastAPI(title="OLA Ride Cancellation Analytics API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_json_cache = {}


def _load_json(filename):
    """Load a pre-computed JSON file (cached in memory)."""
    if filename not in _json_cache:
        path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(path):
            raise HTTPException(status_code=500, detail=f"Data file {filename} not found. Run api/precompute.py first.")
        with open(path, "r", encoding="utf-8") as f:
            _json_cache[filename] = json.load(f)
    return _json_cache[filename]


@app.get("/")
async def root():
    html_path = os.path.join(ROOT, "public", "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return JSONResponse({"app": "OLA Analytics", "status": "running"})


@app.get("/api/health")
async def health():
    return {"status": "ok", "app": "OLA Ride Cancellation Analytics"}


@app.get("/api/kpis")
async def get_kpis():
    return _load_json("kpis.json")


@app.get("/api/data/sample")
async def data_sample():
    return _load_json("sample.json")


@app.get("/api/eda/status-distribution")
async def eda_status():
    return {"status_distribution": _load_json("status_dist.json")}


@app.get("/api/eda/hourly")
async def eda_hourly():
    return _load_json("hourly.json")


@app.get("/api/eda/location")
async def eda_location():
    return _load_json("location.json")


@app.get("/api/eda/payment")
async def eda_payment():
    return _load_json("payment.json")


@app.get("/api/eda/vehicle")
async def eda_vehicle():
    return _load_json("vehicle.json")


@app.get("/api/eda/daily")
async def eda_daily():
    return _load_json("daily.json")


@app.get("/api/eda/cancel-reasons")
async def eda_reasons():
    return _load_json("reasons.json")


@app.get("/api/sql/prebuilt")
async def sql_list():
    return _load_json("sql_queries.json")


@app.get("/api/sql/run")
async def sql_run(query_name: str = Query(default=None)):
    if not query_name:
        raise HTTPException(status_code=400, detail="Provide query_name parameter")
    results = _load_json("sql_results.json")
    if query_name not in results:
        raise HTTPException(status_code=404, detail=f"Query '{query_name}' not found")
    return results[query_name]


@app.get("/api/ml/metrics")
async def ml_metrics():
    return _load_json("ml_metrics.json")


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
    """Predict cancellation using pre-trained model."""
    import pickle
    model_path = os.path.join(DATA_DIR, "model.pkl")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=500, detail="Model not found. Run api/precompute.py first.")

    if "model_data" not in _json_cache:
        with open(model_path, "rb") as f:
            _json_cache["model_data"] = pickle.load(f)

    md = _json_cache["model_data"]
    model = md["model"]
    encoders = md["encoders"]
    features = md["features"]

    # Build feature row
    row = {
        "BookingHour": booking_hour,
        "RideDistance_km": ride_distance,
        "BookingValue_INR": booking_value,
        "DriverRating": driver_rating,
        "CustomerRating": customer_rating,
        "ETA_Pickup_min": eta_pickup,
        "IsWeekend_enc": int(is_weekend),
    }

    for src, enc_name in [("VehicleType", "VehicleType_enc"),
                          ("PaymentMode", "PaymentMode_enc"),
                          ("PickupLocation", "PickupLocation_enc")]:
        if src in encoders:
            le = encoders[src]
            val = vehicle_type if src == "VehicleType" else (payment_mode if src == "PaymentMode" else pickup_location)
            row[enc_name] = int(le.transform([val])[0]) if val in le.classes_ else 0
        else:
            row[enc_name] = 0

    # Predict
    import numpy as np
    X = np.array([[row.get(f, 0) for f in features]])
    proba = float(model.predict_proba(X)[0][1])
    prediction = int(model.predict(X)[0])
    risk_level = "Low" if proba < 0.3 else ("Medium" if proba < 0.6 else "High")

    return {
        "cancellation_probability": round(proba, 4),
        "prediction": prediction,
        "risk_level": risk_level,
    }


@app.get("/api/ml/clusters")
async def ml_clusters():
    return {"clusters": _load_json("clusters.json")}


@app.get("/api/recommendations")
async def get_recommendations():
    return _load_json("recommendations.json")
