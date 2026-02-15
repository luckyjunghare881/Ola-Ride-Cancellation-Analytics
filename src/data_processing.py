"""
Data preprocessing and validation module.
Handles CSV loading, cleaning, validation, and feature engineering.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
import io


# ── Expected schema ────────────────────────────────────────────────────────
REQUIRED_COLUMNS = ["BookingID", "BookingDate", "RideStatus"]
EXPECTED_COLUMNS = [
    "BookingID", "BookingDate", "RideStatus", "VehicleType",
    "PickupLocation", "DropLocation", "PaymentMode",
    "RideDistance_km", "BookingValue_INR", "DriverRating",
    "CustomerRating", "ETA_Pickup_min", "RideDuration_min",
    "CancelReason",
]

MAX_ROWS = 200_000  # Safety limit


def validate_csv(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate uploaded CSV against expected schema."""
    errors = []
    if df.empty:
        errors.append("Dataset is empty (0 rows).")
        return False, errors

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}")
        return False, errors

    if len(df) > MAX_ROWS:
        errors.append(f"Dataset too large ({len(df):,} rows). Max: {MAX_ROWS:,}.")
        return False, errors

    return True, []


def load_csv(file_or_path, max_rows: int = MAX_ROWS) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load CSV from file path or uploaded file object.
    Returns (DataFrame, list_of_warnings).
    """
    warnings = []
    try:
        if isinstance(file_or_path, str):
            df = pd.read_csv(file_or_path, nrows=max_rows)
        else:
            df = pd.read_csv(file_or_path, nrows=max_rows)
    except Exception as e:
        return pd.DataFrame(), [f"Failed to parse CSV: {str(e)}"]

    valid, errors = validate_csv(df)
    if not valid:
        return pd.DataFrame(), errors

    return df, warnings


def preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Clean and feature-engineer the DataFrame.
    Returns (cleaned_df, cleaning_report).
    """
    report = {
        "original_rows": len(df),
        "duplicates_removed": 0,
        "missing_filled": {},
        "columns_added": [],
    }

    df = df.copy()

    # ── Remove exact duplicates ───────────────────────────────────────────
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        report["duplicates_removed"] = int(dup_count)

    # ── Parse dates ───────────────────────────────────────────────────────
    if "BookingDate" in df.columns:
        df["BookingDate"] = pd.to_datetime(df["BookingDate"], errors="coerce")
        null_dates = df["BookingDate"].isna().sum()
        if null_dates > 0:
            df = df.dropna(subset=["BookingDate"]).reset_index(drop=True)
            report["missing_filled"]["BookingDate"] = f"{null_dates} unparseable rows dropped"

    # ── Standardize RideStatus ────────────────────────────────────────────
    if "RideStatus" in df.columns:
        df["RideStatus"] = df["RideStatus"].str.strip().str.title()

    # ── Fill numeric NaNs with median ─────────────────────────────────────
    numeric_cols = ["RideDistance_km", "BookingValue_INR", "DriverRating",
                    "CustomerRating", "ETA_Pickup_min"]
    for col in numeric_cols:
        if col in df.columns:
            na_count = df[col].isna().sum()
            if na_count > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                report["missing_filled"][col] = f"{na_count} filled with median ({median_val:.1f})"

    # ── Feature engineering ───────────────────────────────────────────────
    if "BookingDate" in df.columns:
        if "BookingHour" not in df.columns:
            df["BookingHour"] = df["BookingDate"].dt.hour
            report["columns_added"].append("BookingHour")
        if "DayOfWeek" not in df.columns:
            df["DayOfWeek"] = df["BookingDate"].dt.day_name()
            report["columns_added"].append("DayOfWeek")
        if "Month" not in df.columns:
            df["Month"] = df["BookingDate"].dt.month_name()
            report["columns_added"].append("Month")
        if "IsWeekend" not in df.columns:
            df["IsWeekend"] = df["BookingDate"].dt.dayofweek >= 5
            report["columns_added"].append("IsWeekend")
        if "TimePeriod" not in df.columns:
            df["TimePeriod"] = pd.cut(
                df["BookingHour"],
                bins=[-1, 5, 11, 16, 20, 24],
                labels=["Late Night", "Morning", "Afternoon", "Evening", "Night"],
            )
            report["columns_added"].append("TimePeriod")

    # ── IsCanceled flag ───────────────────────────────────────────────────
    if "RideStatus" in df.columns and "IsCanceled" not in df.columns:
        df["IsCanceled"] = (~df["RideStatus"].str.contains("Completed", case=False, na=False)).astype(int)
        report["columns_added"].append("IsCanceled")

    report["final_rows"] = len(df)
    return df, report


def compute_kpis(df: pd.DataFrame) -> Dict:
    """Compute key performance indicators from the dataset."""
    total = len(df)
    if total == 0:
        return {
            "total_bookings": 0,
            "completed_rides": 0,
            "canceled_rides": 0,
            "cancellation_rate": 0.0,
            "driver_cancel_rate": 0.0,
            "customer_cancel_rate": 0.0,
            "avg_booking_value": 0.0,
            "avg_ride_distance": 0.0,
            "avg_driver_rating": 0.0,
            "avg_eta_pickup": 0.0,
            "revenue_loss_est": 0.0,
        }

    canceled = df["IsCanceled"].sum() if "IsCanceled" in df.columns else 0
    completed = total - canceled

    driver_cancel = len(df[df["RideStatus"].str.contains("Driver", case=False, na=False)]) if "RideStatus" in df.columns else 0
    customer_cancel = len(df[df["RideStatus"].str.contains("Customer", case=False, na=False)]) if "RideStatus" in df.columns else 0

    avg_value = df["BookingValue_INR"].mean() if "BookingValue_INR" in df.columns else 0
    avg_dist = df["RideDistance_km"].mean() if "RideDistance_km" in df.columns else 0
    avg_rating = df["DriverRating"].mean() if "DriverRating" in df.columns else 0
    avg_eta = df["ETA_Pickup_min"].mean() if "ETA_Pickup_min" in df.columns else 0

    # Revenue loss = canceled rides × avg fare
    canceled_df = df[df.get("IsCanceled", pd.Series([0]*total)) == 1] if "IsCanceled" in df.columns else pd.DataFrame()
    rev_loss = canceled_df["BookingValue_INR"].sum() if "BookingValue_INR" in canceled_df.columns else 0

    return {
        "total_bookings": int(total),
        "completed_rides": int(completed),
        "canceled_rides": int(canceled),
        "cancellation_rate": round(canceled / total * 100, 2),
        "driver_cancel_rate": round(driver_cancel / total * 100, 2),
        "customer_cancel_rate": round(customer_cancel / total * 100, 2),
        "avg_booking_value": round(float(avg_value), 2),
        "avg_ride_distance": round(float(avg_dist), 2),
        "avg_driver_rating": round(float(avg_rating), 2),
        "avg_eta_pickup": round(float(avg_eta), 2),
        "revenue_loss_est": round(float(rev_loss), 2),
    }


def get_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics for numeric columns."""
    num_df = df.select_dtypes(include=[np.number])
    if num_df.empty:
        return pd.DataFrame()
    return num_df.describe().round(2).T
