#!/usr/bin/env python3
"""
Refresh all materialized views for the Swift Live Tracking Dashboard.
Run this script every 5-10 minutes to keep the dashboard data fresh.

Usage:
    python3 refresh_views.py

You can set this up as a cron job:
    */5 * * * * cd /path/to/dashboard && python3 refresh_views.py >> refresh.log 2>&1
"""

import os
from dotenv import load_dotenv
import psycopg2
import time
from datetime import datetime

load_dotenv()

def refresh_views():
    conn = psycopg2.connect(
        host=os.getenv("Host"),
        user=os.getenv("UserName"),
        password=os.getenv("Password"),
        database=os.getenv("database_name"),
        port=int(os.getenv("Port", 5432)),
        connect_timeout=30
    )
    conn.autocommit = True
    cur = conn.cursor()

    # Views with unique index (can use CONCURRENTLY)
    concurrent_views = [
        "mv_latest_vehicle_positions",
        "mv_last_movement"
    ]

    # Views without unique index (use regular refresh)
    regular_views = [
        "mv_overspeed_24h",
        "mv_overspeed_monthly",
        "mv_night_driving",
        "mv_night_driving_monthly"
    ]

    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting refresh...")

    for view in concurrent_views:
        try:
            start = time.time()
            cur.execute(f"REFRESH MATERIALIZED VIEW CONCURRENTLY {view}")
            elapsed = time.time() - start
            print(f"  {view}: {elapsed:.1f}s (concurrent)")
        except Exception as e:
            print(f"  {view}: ERROR - {e}")

    for view in regular_views:
        try:
            start = time.time()
            cur.execute(f"REFRESH MATERIALIZED VIEW {view}")
            elapsed = time.time() - start
            print(f"  {view}: {elapsed:.1f}s")
        except Exception as e:
            print(f"  {view}: ERROR - {e}")

    print("Refresh complete!")
    conn.close()

if __name__ == "__main__":
    refresh_views()
