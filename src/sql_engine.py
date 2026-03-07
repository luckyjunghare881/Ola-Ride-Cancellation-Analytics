"""
SQL Query Engine using SQLite for in-memory querying of ride data.
Provides pre-built analytics queries and custom query execution.
"""

import pandas as pd
import sqlite3
from typing import Tuple, List, Dict, Optional


# ── Pre-built analytical queries ──────────────────────────────────────────
PREBUILT_QUERIES: Dict[str, Dict] = {
    "Overall Cancellation Summary": {
        "sql": """
            SELECT RideStatus,
                   COUNT(*) AS TotalRides,
                   ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM rides), 2) AS Percentage
            FROM rides
            GROUP BY RideStatus
            ORDER BY TotalRides DESC;
        """,
        "description": "Breakdown of ride statuses with percentages.",
    },
    "Cancellations by Hour": {
        "sql": """
            SELECT BookingHour,
                   COUNT(*) AS TotalBookings,
                   SUM(CASE WHEN IsCanceled = 1 THEN 1 ELSE 0 END) AS Canceled,
                   ROUND(SUM(CASE WHEN IsCanceled = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS CancelRate
            FROM rides
            GROUP BY BookingHour
            ORDER BY BookingHour;
        """,
        "description": "Hourly cancellation patterns — identifies peak cancellation hours.",
    },
    "Cancellations by Day of Week": {
        "sql": """
            SELECT DayOfWeek,
                   COUNT(*) AS TotalBookings,
                   SUM(CASE WHEN IsCanceled = 1 THEN 1 ELSE 0 END) AS Canceled,
                   ROUND(SUM(CASE WHEN IsCanceled = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS CancelRate
            FROM rides
            GROUP BY DayOfWeek
            ORDER BY CancelRate DESC;
        """,
        "description": "Cancellation rate by day of week.",
    },
    "Top 10 Cancellation Hotspot Locations": {
        "sql": """
            SELECT PickupLocation,
                   COUNT(*) AS TotalBookings,
                   SUM(CASE WHEN IsCanceled = 1 THEN 1 ELSE 0 END) AS Canceled,
                   ROUND(SUM(CASE WHEN IsCanceled = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS CancelRate
            FROM rides
            GROUP BY PickupLocation
            HAVING TotalBookings >= 10
            ORDER BY CancelRate DESC
            LIMIT 10;
        """,
        "description": "Pickup locations with highest cancellation rates.",
    },
    "Cancellations by Payment Mode": {
        "sql": """
            SELECT PaymentMode,
                   COUNT(*) AS TotalBookings,
                   SUM(CASE WHEN IsCanceled = 1 THEN 1 ELSE 0 END) AS Canceled,
                   ROUND(SUM(CASE WHEN IsCanceled = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS CancelRate
            FROM rides
            GROUP BY PaymentMode
            ORDER BY CancelRate DESC;
        """,
        "description": "Which payment modes have higher cancellation rates?",
    },
    "Cancellations by Vehicle Type": {
        "sql": """
            SELECT VehicleType,
                   COUNT(*) AS TotalBookings,
                   SUM(CASE WHEN IsCanceled = 1 THEN 1 ELSE 0 END) AS Canceled,
                   ROUND(SUM(CASE WHEN IsCanceled = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS CancelRate
            FROM rides
            GROUP BY VehicleType
            ORDER BY CancelRate DESC;
        """,
        "description": "Cancellation patterns across different vehicle types.",
    },
    "Top Cancel Reasons (Driver)": {
        "sql": """
            SELECT CancelReason,
                   COUNT(*) AS Occurrences
            FROM rides
            WHERE RideStatus LIKE '%Driver%'
              AND CancelReason IS NOT NULL
            GROUP BY CancelReason
            ORDER BY Occurrences DESC;
        """,
        "description": "Most common reasons drivers cancel rides.",
    },
    "Top Cancel Reasons (Customer)": {
        "sql": """
            SELECT CancelReason,
                   COUNT(*) AS Occurrences
            FROM rides
            WHERE RideStatus LIKE '%Customer%'
              AND CancelReason IS NOT NULL
            GROUP BY CancelReason
            ORDER BY Occurrences DESC;
        """,
        "description": "Most common reasons customers cancel rides.",
    },
    "Monthly Cancellation Trend": {
        "sql": """
            SELECT Month,
                   COUNT(*) AS TotalBookings,
                   SUM(CASE WHEN IsCanceled = 1 THEN 1 ELSE 0 END) AS Canceled,
                   ROUND(SUM(CASE WHEN IsCanceled = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS CancelRate
            FROM rides
            GROUP BY Month
            ORDER BY CancelRate DESC;
        """,
        "description": "Cancellation trends across months.",
    },
    "Revenue Loss from Cancellations": {
        "sql": """
            SELECT RideStatus,
                   COUNT(*) AS Rides,
                   ROUND(SUM(BookingValue_INR), 2) AS TotalValue,
                   ROUND(AVG(BookingValue_INR), 2) AS AvgValue
            FROM rides
            GROUP BY RideStatus
            ORDER BY TotalValue DESC;
        """,
        "description": "Estimated revenue impact of cancellations.",
    },
    "High ETA vs Cancellation": {
        "sql": """
            SELECT
                CASE
                    WHEN ETA_Pickup_min <= 5 THEN '0-5 min'
                    WHEN ETA_Pickup_min <= 10 THEN '5-10 min'
                    WHEN ETA_Pickup_min <= 15 THEN '10-15 min'
                    WHEN ETA_Pickup_min <= 20 THEN '15-20 min'
                    ELSE '20+ min'
                END AS ETA_Bucket,
                COUNT(*) AS TotalBookings,
                SUM(CASE WHEN IsCanceled = 1 THEN 1 ELSE 0 END) AS Canceled,
                ROUND(SUM(CASE WHEN IsCanceled = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS CancelRate
            FROM rides
            GROUP BY ETA_Bucket
            ORDER BY CancelRate DESC;
        """,
        "description": "How ETA to pickup affects cancellation probability.",
    },
}


class SQLEngine:
    """In-memory SQLite engine for ride data analytics."""

    def __init__(self):
        self.conn: Optional[sqlite3.Connection] = None

    def load_data(self, df: pd.DataFrame, table_name: str = "rides"):
        """Load DataFrame into in-memory SQLite database."""
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        df.to_sql(table_name, self.conn, index=False, if_exists="replace")

    def execute_query(self, sql: str) -> Tuple[pd.DataFrame, Optional[str]]:
        """Execute SQL query and return results or error."""
        if self.conn is None:
            return pd.DataFrame(), "No data loaded. Please load data first."
        try:
            # Safety: block destructive operations
            sql_upper = sql.strip().upper()
            if any(kw in sql_upper for kw in ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE"]):
                return pd.DataFrame(), "Only SELECT queries are allowed."
            result = pd.read_sql_query(sql, self.conn)
            return result, None
        except Exception as e:
            return pd.DataFrame(), f"SQL Error: {str(e)}"

    def get_prebuilt_queries(self) -> Dict[str, Dict]:
        """Return available pre-built queries."""
        return PREBUILT_QUERIES

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
