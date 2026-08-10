"""
db/logger.py

Phase E — minimal SQLite logging for demo runs. No shared team logger
to hook into (solo build), so this is self-contained: one table,
flattened summary columns for easy querying/writeup, plus a raw JSON
blob column per row so nothing from a given run is ever lost even if
the flattened schema turns out to be missing something later.

Logging is silent by design (no UI view) - call log_run() right after
run_scenario_comparison() returns, nothing else needed.
"""

import json
import os
import sqlite3
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(__file__), "urbanflow_runs.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    n INTEGER NOT NULL,
    s INTEGER NOT NULL,
    e INTEGER NOT NULL,
    w INTEGER NOT NULL,
    total_vehicles INTEGER NOT NULL,
    fixed_clearance_time REAL,
    fixed_cleared INTEGER NOT NULL,
    rl_clearance_time REAL,
    rl_cleared INTEGER NOT NULL,
    forced_switches INTEGER,
    pct_improvement REAL,
    raw_json TEXT NOT NULL
);
"""


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(SCHEMA)
    return conn


def log_run(result: dict) -> int:
    """
    Inserts one row for a completed run_scenario_comparison() result.
    Returns the inserted row's id.

    Expects the exact dict shape returned by run_scenario_comparison():
        {
          "input_counts": {"N":.., "S":.., "E":.., "W":..},
          "total_vehicles": int,
          "fixed_timer": {"clearance_time":.., "steps":.., "cleared":bool},
          "rl_agent": {"clearance_time":.., "steps":.., "cleared":bool, "forced_switches":int},
          "clearance_time_diff_s": float | None,
          "pct_improvement": float | None,
        }
    """
    counts = result["input_counts"]
    fixed = result["fixed_timer"]
    rl = result["rl_agent"]

    conn = _get_conn()
    try:
        cur = conn.execute(
            """
            INSERT INTO runs (
                timestamp, n, s, e, w, total_vehicles,
                fixed_clearance_time, fixed_cleared,
                rl_clearance_time, rl_cleared, forced_switches,
                pct_improvement, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                counts["N"], counts["S"], counts["E"], counts["W"],
                result["total_vehicles"],
                fixed.get("clearance_time"),
                int(bool(fixed.get("cleared"))),
                rl.get("clearance_time"),
                int(bool(rl.get("cleared"))),
                rl.get("forced_switches"),
                result.get("pct_improvement"),
                json.dumps(result),
            ),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


if __name__ == "__main__":
    # Quick manual smoke test with a fake result shaped like the real thing.
    fake_result = {
        "input_counts": {"N": 10, "S": 10, "E": 5, "W": 5},
        "total_vehicles": 30,
        "fixed_timer": {"clearance_time": 60.25, "steps": 1205, "cleared": True},
        "rl_agent": {"clearance_time": 56.0, "steps": 7, "cleared": True, "forced_switches": 0},
        "clearance_time_diff_s": 4.25,
        "pct_improvement": 7.05,
    }
    row_id = log_run(fake_result)
    print(f"Inserted row id={row_id} into {DB_PATH}")

    conn = _get_conn()
    row = conn.execute("SELECT * FROM runs WHERE id = ?", (row_id,)).fetchone()
    print(row)
    conn.close()