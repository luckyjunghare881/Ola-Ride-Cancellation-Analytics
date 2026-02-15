"""
ML module: Train cancellation prediction model, predict, explain.
Uses RandomForest + optional XGBoost. Persists with joblib.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
from sklearn.cluster import KMeans
import joblib
import os
from typing import Dict, Tuple, Optional, List


MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

# Features used for prediction
FEATURE_COLS = [
    "BookingHour", "RideDistance_km", "BookingValue_INR",
    "DriverRating", "CustomerRating", "ETA_Pickup_min",
    "IsWeekend_enc",
    "VehicleType_enc", "PaymentMode_enc", "PickupLocation_enc",
]

LABEL_COL = "IsCanceled"


def _encode_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """Encode categorical features for ML."""
    df = df.copy()
    encoders = {}

    cat_cols = {
        "VehicleType": "VehicleType_enc",
        "PaymentMode": "PaymentMode_enc",
        "PickupLocation": "PickupLocation_enc",
    }

    for src, dest in cat_cols.items():
        if src in df.columns:
            le = LabelEncoder()
            df[dest] = le.fit_transform(df[src].astype(str))
            encoders[src] = le
        else:
            df[dest] = 0

    # Boolean encoding
    if "IsWeekend" in df.columns:
        df["IsWeekend_enc"] = df["IsWeekend"].astype(int)
    else:
        df["IsWeekend_enc"] = 0

    return df, encoders


def prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, Dict]:
    """Prepare features and labels for training."""
    df_enc, encoders = _encode_features(df)

    available_features = [c for c in FEATURE_COLS if c in df_enc.columns]

    X = df_enc[available_features].fillna(0)
    y = df_enc[LABEL_COL] if LABEL_COL in df_enc.columns else pd.Series([0] * len(df_enc))

    return X, y, {"encoders": encoders, "features": available_features}


def train_model(
    df: pd.DataFrame,
    model_type: str = "random_forest",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict:
    """
    Train cancellation prediction model.
    Returns dict with model, metrics, feature importance, etc.
    """
    X, y, meta = prepare_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if model_type == "gradient_boosting":
        model = GradientBoostingClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.1,
            random_state=random_state, subsample=0.8,
        )
    else:
        model = RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=random_state,
            n_jobs=-1, class_weight="balanced",
        )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }

    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="f1")
    metrics["cv_f1_mean"] = round(cv_scores.mean(), 4)
    metrics["cv_f1_std"] = round(cv_scores.std(), 4)

    # Feature importance
    feat_importance = pd.DataFrame({
        "Feature": meta["features"],
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    return {
        "model": model,
        "metrics": metrics,
        "feature_importance": feat_importance,
        "confusion_matrix": cm,
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "encoders": meta["encoders"],
        "features": meta["features"],
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


def predict_single(
    model, encoders: Dict, features: List[str],
    booking_hour: int, ride_distance: float, booking_value: float,
    driver_rating: float, customer_rating: float, eta_pickup: float,
    is_weekend: bool, vehicle_type: str, payment_mode: str,
    pickup_location: str,
) -> Dict:
    """Predict cancellation probability for a single booking."""
    row = {
        "BookingHour": booking_hour,
        "RideDistance_km": ride_distance,
        "BookingValue_INR": booking_value,
        "DriverRating": driver_rating,
        "CustomerRating": customer_rating,
        "ETA_Pickup_min": eta_pickup,
        "IsWeekend_enc": int(is_weekend),
    }

    # Encode categoricals
    for src, enc_name in [("VehicleType", "VehicleType_enc"),
                          ("PaymentMode", "PaymentMode_enc"),
                          ("PickupLocation", "PickupLocation_enc")]:
        if src in encoders:
            le = encoders[src]
            val = vehicle_type if src == "VehicleType" else (payment_mode if src == "PaymentMode" else pickup_location)
            if val in le.classes_:
                row[enc_name] = le.transform([val])[0]
            else:
                row[enc_name] = 0
        else:
            row[enc_name] = 0

    X_single = pd.DataFrame([row])[features]
    proba = model.predict_proba(X_single)[0][1]
    prediction = model.predict(X_single)[0]

    risk_level = "Low" if proba < 0.3 else ("Medium" if proba < 0.6 else "High")

    return {
        "cancellation_probability": round(float(proba), 4),
        "prediction": int(prediction),
        "risk_level": risk_level,
    }


def cluster_cancellation_hotspots(df: pd.DataFrame, n_clusters: int = 5) -> pd.DataFrame:
    """
    KMeans clustering on cancellation hotspots by location and time.
    Returns cluster assignments with aggregated stats.
    """
    canceled_df = df[df.get("IsCanceled", pd.Series()) == 1].copy()
    if len(canceled_df) < n_clusters:
        return pd.DataFrame()

    # Aggregate by location
    agg = canceled_df.groupby("PickupLocation").agg(
        CancelCount=("IsCanceled", "sum"),
        AvgHour=("BookingHour", "mean"),
        AvgETA=("ETA_Pickup_min", "mean"),
        AvgDistance=("RideDistance_km", "mean"),
        AvgFare=("BookingValue_INR", "mean"),
    ).reset_index()

    if len(agg) < n_clusters:
        n_clusters = max(2, len(agg))

    features = agg[["CancelCount", "AvgHour", "AvgETA", "AvgDistance", "AvgFare"]].fillna(0)

    # Normalize
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    agg["Cluster"] = kmeans.fit_predict(features_scaled)

    return agg.sort_values("CancelCount", ascending=False).reset_index(drop=True)


def save_model(model_result: Dict, filename: str = "cancel_predictor.joblib"):
    """Save trained model and encoders to disk."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, filename)
    joblib.dump({
        "model": model_result["model"],
        "encoders": model_result["encoders"],
        "features": model_result["features"],
        "metrics": model_result["metrics"],
    }, path)
    return path


def load_model(filename: str = "cancel_predictor.joblib") -> Optional[Dict]:
    """Load saved model from disk."""
    path = os.path.join(MODEL_DIR, filename)
    if os.path.exists(path):
        return joblib.load(path)
    return None
