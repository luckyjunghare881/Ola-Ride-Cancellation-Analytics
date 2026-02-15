"""
Unit tests for the SQL engine module.
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__))))

from src.data_processing import preprocess
from src.sql_engine import SQLEngine


@pytest.fixture
def sql_engine():
    """Create a SQL engine loaded with sample data."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "BookingID": [f"BK{str(i).zfill(5)}" for i in range(n)],
        "BookingDate": pd.date_range("2025-06-01", periods=n, freq="h"),
        "RideStatus": np.random.choice(
            ["Completed", "Canceled by Driver", "Canceled by Customer"],
            size=n, p=[0.6, 0.25, 0.15],
        ),
        "VehicleType": np.random.choice(["Mini", "Sedan"], size=n),
        "PickupLocation": np.random.choice(["Koramangala", "Indiranagar"], size=n),
        "DropLocation": np.random.choice(["HSR Layout", "MG Road"], size=n),
        "PaymentMode": np.random.choice(["Cash", "UPI"], size=n),
        "RideDistance_km": np.round(np.random.uniform(2, 30, n), 1),
        "BookingValue_INR": np.round(np.random.uniform(50, 500, n), 0),
        "DriverRating": np.round(np.random.uniform(1, 5, n), 1),
        "CustomerRating": np.round(np.random.uniform(1, 5, n), 1),
        "ETA_Pickup_min": np.round(np.random.uniform(2, 25, n), 1),
    })
    df_clean, _ = preprocess(df)

    engine = SQLEngine()
    engine.load_data(df_clean)
    yield engine
    engine.close()


class TestSQLEngine:
    def test_basic_select(self, sql_engine):
        result, error = sql_engine.execute_query("SELECT COUNT(*) AS cnt FROM rides")
        assert error is None
        assert not result.empty
        assert result["cnt"].iloc[0] > 0

    def test_filter_query(self, sql_engine):
        result, error = sql_engine.execute_query(
            "SELECT * FROM rides WHERE IsCanceled = 1 LIMIT 10"
        )
        assert error is None
        assert len(result) <= 10

    def test_group_by_query(self, sql_engine):
        result, error = sql_engine.execute_query(
            "SELECT RideStatus, COUNT(*) AS cnt FROM rides GROUP BY RideStatus"
        )
        assert error is None
        assert len(result) >= 1

    def test_destructive_query_blocked(self, sql_engine):
        result, error = sql_engine.execute_query("DROP TABLE rides")
        assert error is not None
        assert "SELECT" in error

    def test_invalid_sql(self, sql_engine):
        result, error = sql_engine.execute_query("SELECT * FROM nonexistent_table")
        assert error is not None

    def test_no_data_loaded(self):
        engine = SQLEngine()
        result, error = engine.execute_query("SELECT 1")
        assert error is not None
        assert "no data" in error.lower()

    def test_prebuilt_queries_exist(self, sql_engine):
        queries = sql_engine.get_prebuilt_queries()
        assert len(queries) > 0
        for name, info in queries.items():
            assert "sql" in info
            assert "description" in info

    def test_prebuilt_query_execution(self, sql_engine):
        queries = sql_engine.get_prebuilt_queries()
        # Test first 3 prebuilt queries
        for name in list(queries.keys())[:3]:
            result, error = sql_engine.execute_query(queries[name]["sql"])
            assert error is None, f"Query '{name}' failed: {error}"
            assert isinstance(result, pd.DataFrame)
