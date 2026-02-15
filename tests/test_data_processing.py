"""
Unit tests for the data processing module.
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__))))

from src.data_processing import load_csv, preprocess, compute_kpis, validate_csv, get_summary_stats


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Create a minimal valid DataFrame for testing."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "BookingID": [f"BK{str(i).zfill(5)}" for i in range(n)],
        "BookingDate": pd.date_range("2025-06-01", periods=n, freq="h"),
        "RideStatus": np.random.choice(
            ["Completed", "Canceled by Driver", "Canceled by Customer"],
            size=n, p=[0.6, 0.25, 0.15],
        ),
        "VehicleType": np.random.choice(["Mini", "Sedan", "SUV", "Auto", "Bike"], size=n),
        "PickupLocation": np.random.choice(["Koramangala", "Indiranagar", "Whitefield"], size=n),
        "DropLocation": np.random.choice(["HSR Layout", "MG Road", "BTM Layout"], size=n),
        "PaymentMode": np.random.choice(["Cash", "UPI", "Credit Card"], size=n),
        "RideDistance_km": np.round(np.random.uniform(2, 30, n), 1),
        "BookingValue_INR": np.round(np.random.uniform(50, 500, n), 0),
        "DriverRating": np.round(np.random.uniform(1, 5, n), 1),
        "CustomerRating": np.round(np.random.uniform(1, 5, n), 1),
        "ETA_Pickup_min": np.round(np.random.uniform(2, 25, n), 1),
        "RideDuration_min": np.where(
            np.random.choice([True, False], n, p=[0.6, 0.4]),
            np.round(np.random.uniform(5, 60, n), 1),
            None,
        ),
        "CancelReason": np.where(
            np.random.choice([True, False], n, p=[0.4, 0.6]),
            "Driver too far away",
            None,
        ),
    })


@pytest.fixture
def sample_csv(sample_df, tmp_path):
    """Save sample DataFrame to a temporary CSV file."""
    path = tmp_path / "test_rides.csv"
    sample_df.to_csv(path, index=False)
    return str(path)


# ── Tests: validate_csv ──────────────────────────────────────────────────

class TestValidateCSV:
    def test_valid_df(self, sample_df):
        valid, errors = validate_csv(sample_df)
        assert valid is True
        assert len(errors) == 0

    def test_empty_df(self):
        valid, errors = validate_csv(pd.DataFrame())
        assert valid is False
        assert any("empty" in e.lower() for e in errors)

    def test_missing_required_columns(self):
        df = pd.DataFrame({"SomeColumn": [1, 2, 3]})
        valid, errors = validate_csv(df)
        assert valid is False
        assert any("missing" in e.lower() for e in errors)


# ── Tests: load_csv ──────────────────────────────────────────────────────

class TestLoadCSV:
    def test_load_from_path(self, sample_csv):
        df, warnings = load_csv(sample_csv)
        assert not df.empty
        assert len(warnings) == 0
        assert "BookingID" in df.columns

    def test_load_invalid_path(self):
        df, warnings = load_csv("nonexistent_file.csv")
        assert df.empty
        assert len(warnings) > 0

    def test_load_invalid_csv_content(self, tmp_path):
        path = tmp_path / "bad.csv"
        path.write_text("col1,col2\n1,2\n3,4")
        df, warnings = load_csv(str(path))
        assert df.empty  # Missing required columns
        assert len(warnings) > 0


# ── Tests: preprocess ────────────────────────────────────────────────────

class TestPreprocess:
    def test_returns_cleaned_df_and_report(self, sample_df):
        df_clean, report = preprocess(sample_df)
        assert isinstance(df_clean, pd.DataFrame)
        assert isinstance(report, dict)
        assert len(df_clean) > 0

    def test_adds_feature_columns(self, sample_df):
        df_clean, report = preprocess(sample_df)
        expected_cols = ["BookingHour", "DayOfWeek", "Month", "IsWeekend", "TimePeriod", "IsCanceled"]
        for col in expected_cols:
            assert col in df_clean.columns, f"Missing column: {col}"

    def test_removes_duplicates(self, sample_df):
        df_with_dups = pd.concat([sample_df, sample_df.iloc[:5]], ignore_index=True)
        df_clean, report = preprocess(df_with_dups)
        assert report["duplicates_removed"] >= 5
        assert len(df_clean) <= len(df_with_dups)

    def test_standardizes_ride_status(self, sample_df):
        sample_df.loc[0, "RideStatus"] = "  completed  "
        df_clean, _ = preprocess(sample_df)
        assert df_clean["RideStatus"].iloc[0] == "Completed"

    def test_is_canceled_flag(self, sample_df):
        df_clean, _ = preprocess(sample_df)
        completed_mask = df_clean["RideStatus"] == "Completed"
        assert (df_clean.loc[completed_mask, "IsCanceled"] == 0).all()
        canceled_mask = df_clean["RideStatus"] != "Completed"
        assert (df_clean.loc[canceled_mask, "IsCanceled"] == 1).all()

    def test_report_tracks_original_and_final_rows(self, sample_df):
        df_clean, report = preprocess(sample_df)
        assert "original_rows" in report
        assert "final_rows" in report
        assert report["final_rows"] == len(df_clean)


# ── Tests: compute_kpis ─────────────────────────────────────────────────

class TestComputeKPIs:
    def test_kpis_structure(self, sample_df):
        df_clean, _ = preprocess(sample_df)
        kpis = compute_kpis(df_clean)
        expected_keys = [
            "total_bookings", "completed_rides", "canceled_rides",
            "cancellation_rate", "driver_cancel_rate", "customer_cancel_rate",
            "avg_booking_value", "avg_ride_distance", "avg_driver_rating",
            "avg_eta_pickup", "revenue_loss_est",
        ]
        for key in expected_keys:
            assert key in kpis, f"Missing KPI: {key}"

    def test_kpis_values_make_sense(self, sample_df):
        df_clean, _ = preprocess(sample_df)
        kpis = compute_kpis(df_clean)
        assert kpis["total_bookings"] == len(df_clean)
        assert kpis["completed_rides"] + kpis["canceled_rides"] == kpis["total_bookings"]
        assert 0 <= kpis["cancellation_rate"] <= 100
        assert kpis["avg_booking_value"] > 0

    def test_empty_df_kpis(self):
        kpis = compute_kpis(pd.DataFrame(columns=["IsCanceled", "RideStatus"]))
        assert kpis["total_bookings"] == 0
        assert kpis["cancellation_rate"] == 0.0


# ── Tests: get_summary_stats ─────────────────────────────────────────────

class TestSummaryStats:
    def test_returns_dataframe(self, sample_df):
        stats = get_summary_stats(sample_df)
        assert isinstance(stats, pd.DataFrame)
        assert not stats.empty

    def test_empty_df(self):
        stats = get_summary_stats(pd.DataFrame({"text_col": ["a", "b"]}))
        assert stats.empty
