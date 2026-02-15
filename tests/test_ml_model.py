"""
Unit tests for the ML model module.
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__))))

from src.data_processing import preprocess
from src.ml_model import (
    _encode_features, prepare_data, train_model,
    predict_single, cluster_cancellation_hotspots,
)


@pytest.fixture
def training_df():
    """Create a preprocessed DataFrame for ML testing."""
    np.random.seed(42)
    n = 500
    df = pd.DataFrame({
        "BookingID": [f"BK{str(i).zfill(5)}" for i in range(n)],
        "BookingDate": pd.date_range("2025-06-01", periods=n, freq="h"),
        "RideStatus": np.random.choice(
            ["Completed", "Canceled by Driver", "Canceled by Customer"],
            size=n, p=[0.6, 0.25, 0.15],
        ),
        "VehicleType": np.random.choice(["Mini", "Sedan", "SUV", "Auto", "Bike"], size=n),
        "PickupLocation": np.random.choice(
            ["Koramangala", "Indiranagar", "Whitefield", "HSR Layout", "MG Road"], size=n
        ),
        "DropLocation": np.random.choice(["BTM Layout", "JP Nagar", "Hebbal"], size=n),
        "PaymentMode": np.random.choice(["Cash", "UPI", "Credit Card"], size=n),
        "RideDistance_km": np.round(np.random.uniform(2, 30, n), 1),
        "BookingValue_INR": np.round(np.random.uniform(50, 500, n), 0),
        "DriverRating": np.round(np.random.uniform(1, 5, n), 1),
        "CustomerRating": np.round(np.random.uniform(1, 5, n), 1),
        "ETA_Pickup_min": np.round(np.random.uniform(2, 25, n), 1),
    })
    df_clean, _ = preprocess(df)
    return df_clean


# ── Tests: encode_features ───────────────────────────────────────────────

class TestEncodeFeatures:
    def test_encoding_creates_columns(self, training_df):
        df_enc, encoders = _encode_features(training_df)
        assert "VehicleType_enc" in df_enc.columns
        assert "PaymentMode_enc" in df_enc.columns
        assert "PickupLocation_enc" in df_enc.columns
        assert "IsWeekend_enc" in df_enc.columns

    def test_encoders_are_returned(self, training_df):
        _, encoders = _encode_features(training_df)
        assert "VehicleType" in encoders
        assert "PaymentMode" in encoders
        assert "PickupLocation" in encoders


# ── Tests: prepare_data ─────────────────────────────────────────────────

class TestPrepareData:
    def test_returns_X_y_meta(self, training_df):
        X, y, meta = prepare_data(training_df)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert "encoders" in meta
        assert "features" in meta
        assert len(X) == len(y)

    def test_no_nans_in_X(self, training_df):
        X, _, _ = prepare_data(training_df)
        assert X.isna().sum().sum() == 0


# ── Tests: train_model ──────────────────────────────────────────────────

class TestTrainModel:
    def test_random_forest_training(self, training_df):
        result = train_model(training_df, model_type="random_forest")
        assert "model" in result
        assert "metrics" in result
        assert "feature_importance" in result
        assert "confusion_matrix" in result

    def test_gradient_boosting_training(self, training_df):
        result = train_model(training_df, model_type="gradient_boosting")
        assert result["metrics"]["accuracy"] > 0

    def test_metrics_are_valid(self, training_df):
        result = train_model(training_df)
        metrics = result["metrics"]
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["f1_score"] <= 1
        assert 0 <= metrics["roc_auc"] <= 1

    def test_feature_importance_sums_to_one(self, training_df):
        result = train_model(training_df)
        total_imp = result["feature_importance"]["Importance"].sum()
        assert abs(total_imp - 1.0) < 0.01


# ── Tests: predict_single ───────────────────────────────────────────────

class TestPredictSingle:
    def test_prediction_output(self, training_df):
        result = train_model(training_df)
        pred = predict_single(
            result["model"], result["encoders"], result["features"],
            booking_hour=18, ride_distance=10.0, booking_value=200.0,
            driver_rating=3.5, customer_rating=4.0, eta_pickup=15.0,
            is_weekend=False, vehicle_type="Mini",
            payment_mode="Cash", pickup_location="Koramangala",
        )
        assert "cancellation_probability" in pred
        assert "prediction" in pred
        assert "risk_level" in pred
        assert 0 <= pred["cancellation_probability"] <= 1
        assert pred["risk_level"] in ("Low", "Medium", "High")

    def test_unknown_category_handling(self, training_df):
        result = train_model(training_df)
        pred = predict_single(
            result["model"], result["encoders"], result["features"],
            booking_hour=12, ride_distance=5.0, booking_value=100.0,
            driver_rating=4.0, customer_rating=4.0, eta_pickup=5.0,
            is_weekend=True, vehicle_type="UNKNOWN_VEHICLE",
            payment_mode="UNKNOWN_PAY", pickup_location="UNKNOWN_LOC",
        )
        assert 0 <= pred["cancellation_probability"] <= 1


# ── Tests: cluster_cancellation_hotspots ─────────────────────────────────

class TestClustering:
    def test_clustering_returns_results(self, training_df):
        clusters = cluster_cancellation_hotspots(training_df, n_clusters=3)
        assert isinstance(clusters, pd.DataFrame)
        if not clusters.empty:
            assert "Cluster" in clusters.columns
            assert "PickupLocation" in clusters.columns

    def test_empty_df_returns_empty(self):
        empty = pd.DataFrame(columns=["IsCanceled", "PickupLocation", "BookingHour",
                                       "ETA_Pickup_min", "RideDistance_km", "BookingValue_INR"])
        clusters = cluster_cancellation_hotspots(empty)
        assert clusters.empty
