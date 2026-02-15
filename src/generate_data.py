"""
Generate synthetic OLA ride booking dataset (10,000+ rows).
Schema mirrors real OLA ride data with realistic distributions.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import random

# ── Configuration ──────────────────────────────────────────────────────────
NUM_ROWS = 12_000
SEED = 42
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sample_ola_rides.csv")

# ── Reference data ─────────────────────────────────────────────────────────
PICKUP_LOCATIONS = [
    "Koramangala", "Indiranagar", "Whitefield", "Electronic City",
    "MG Road", "HSR Layout", "Marathahalli", "Jayanagar",
    "BTM Layout", "Yelahanka", "Hebbal", "Banashankari",
    "JP Nagar", "Rajajinagar", "Malleswaram", "Bellandur",
    "Sarjapur Road", "KR Puram", "Bannerghatta Road", "Silk Board"
]

DROP_LOCATIONS = PICKUP_LOCATIONS.copy()

PAYMENT_MODES = ["Cash", "UPI", "Credit Card", "Debit Card", "Wallet"]
PAYMENT_WEIGHTS = [0.25, 0.35, 0.15, 0.10, 0.15]

RIDE_STATUSES = ["Completed", "Canceled by Driver", "Canceled by Customer"]
STATUS_WEIGHTS = [0.62, 0.22, 0.16]

VEHICLE_TYPES = ["Mini", "Sedan", "SUV", "Auto", "Bike"]
VEHICLE_WEIGHTS = [0.30, 0.25, 0.15, 0.20, 0.10]

CANCEL_REASONS_DRIVER = [
    "Customer was not available",
    "Wrong address / pickup mismatch",
    "Heavy traffic / long route",
    "Personal emergency",
    "Vehicle issue",
    "Customer asked to cancel",
]

CANCEL_REASONS_CUSTOMER = [
    "Driver too far away",
    "Change of plans",
    "Found alternative transport",
    "Long ETA",
    "Driver not moving",
    "Price too high",
    "Booked by mistake",
]


def generate_dataset(num_rows: int = NUM_ROWS, seed: int = SEED) -> pd.DataFrame:
    """Generate a synthetic OLA rides DataFrame."""
    np.random.seed(seed)
    random.seed(seed)

    # ── Booking dates: last 12 months ──────────────────────────────────────
    end_date = datetime(2026, 1, 31)
    start_date = end_date - timedelta(days=365)
    booking_dates = [
        start_date + timedelta(
            seconds=np.random.randint(0, int((end_date - start_date).total_seconds()))
        )
        for _ in range(num_rows)
    ]
    booking_dates.sort()

    # ── Core columns ───────────────────────────────────────────────────────
    booking_ids = [f"BK{str(i+1).zfill(6)}" for i in range(num_rows)]

    ride_statuses = np.random.choice(RIDE_STATUSES, size=num_rows, p=STATUS_WEIGHTS)

    payment_modes = np.random.choice(PAYMENT_MODES, size=num_rows, p=PAYMENT_WEIGHTS)

    pickup_locations = np.random.choice(PICKUP_LOCATIONS, size=num_rows)

    # Ensure drop ≠ pickup
    drop_locations = []
    for pu in pickup_locations:
        dl = np.random.choice([loc for loc in DROP_LOCATIONS if loc != pu])
        drop_locations.append(dl)

    vehicle_types = np.random.choice(VEHICLE_TYPES, size=num_rows, p=VEHICLE_WEIGHTS)

    # ── Numeric features ───────────────────────────────────────────────────
    ride_distances = np.round(np.random.exponential(scale=8, size=num_rows) + 1.5, 1)
    ride_distances = np.clip(ride_distances, 1.0, 45.0)

    # Fare based on distance + vehicle multiplier
    vehicle_multiplier = {"Mini": 8, "Sedan": 11, "SUV": 15, "Auto": 7, "Bike": 5}
    base_fare = 30
    fares = []
    for i in range(num_rows):
        fare = base_fare + ride_distances[i] * vehicle_multiplier[vehicle_types[i]]
        fare *= np.random.uniform(0.85, 1.25)  # surge variation
        fares.append(round(fare, 0))
    fares = np.array(fares)

    # Driver ratings (1-5)
    driver_ratings = np.round(np.random.normal(4.1, 0.6, size=num_rows), 1)
    driver_ratings = np.clip(driver_ratings, 1.0, 5.0)

    # Customer ratings
    customer_ratings = np.round(np.random.normal(4.3, 0.5, size=num_rows), 1)
    customer_ratings = np.clip(customer_ratings, 1.0, 5.0)

    # ETA to pickup (minutes)
    eta_pickup = np.round(np.random.exponential(scale=7, size=num_rows) + 2, 1)
    eta_pickup = np.clip(eta_pickup, 1.0, 40.0)

    # ── Cancellation reasons ──────────────────────────────────────────────
    cancel_reasons = []
    for status in ride_statuses:
        if status == "Canceled by Driver":
            cancel_reasons.append(random.choice(CANCEL_REASONS_DRIVER))
        elif status == "Canceled by Customer":
            cancel_reasons.append(random.choice(CANCEL_REASONS_CUSTOMER))
        else:
            cancel_reasons.append(None)

    # ── Ride duration (only for completed) ────────────────────────────────
    ride_durations = []
    for i in range(num_rows):
        if ride_statuses[i] == "Completed":
            dur = ride_distances[i] * np.random.uniform(2.5, 4.5)  # mins
            ride_durations.append(round(dur, 1))
        else:
            ride_durations.append(None)

    # ── Build DataFrame ───────────────────────────────────────────────────
    df = pd.DataFrame({
        "BookingID": booking_ids,
        "BookingDate": booking_dates,
        "RideStatus": ride_statuses,
        "VehicleType": vehicle_types,
        "PickupLocation": pickup_locations,
        "DropLocation": drop_locations,
        "PaymentMode": payment_modes,
        "RideDistance_km": ride_distances,
        "BookingValue_INR": fares,
        "DriverRating": driver_ratings,
        "CustomerRating": customer_ratings,
        "ETA_Pickup_min": eta_pickup,
        "RideDuration_min": ride_durations,
        "CancelReason": cancel_reasons,
    })

    # ── Add time-derived columns ──────────────────────────────────────────
    df["BookingDate"] = pd.to_datetime(df["BookingDate"])
    df["BookingHour"] = df["BookingDate"].dt.hour
    df["DayOfWeek"] = df["BookingDate"].dt.day_name()
    df["Month"] = df["BookingDate"].dt.month_name()

    # ── Inject realistic cancellation bias ────────────────────────────────
    # Higher cancellation at peak hours
    peak_mask = df["BookingHour"].isin([8, 9, 17, 18, 19])
    extra_cancel_idx = df[peak_mask & (df["RideStatus"] == "Completed")].sample(
        frac=0.12, random_state=seed
    ).index
    df.loc[extra_cancel_idx, "RideStatus"] = np.random.choice(
        ["Canceled by Driver", "Canceled by Customer"],
        size=len(extra_cancel_idx),
        p=[0.6, 0.4],
    )
    for idx in extra_cancel_idx:
        if df.loc[idx, "RideStatus"] == "Canceled by Driver":
            df.loc[idx, "CancelReason"] = random.choice(CANCEL_REASONS_DRIVER)
        else:
            df.loc[idx, "CancelReason"] = random.choice(CANCEL_REASONS_CUSTOMER)
        df.loc[idx, "RideDuration_min"] = None

    return df


def save_dataset(df: pd.DataFrame, path: str = OUTPUT_PATH) -> str:
    """Save DataFrame to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    print("Generating synthetic OLA ride dataset...")
    df = generate_dataset()
    path = save_dataset(df)
    print(f"✅ Dataset saved to {path}")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    print(f"\nRide Status Distribution:")
    print(df["RideStatus"].value_counts().to_string())
