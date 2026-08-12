"""
Watchlist Builder - Phase 2
============================
Runs once per day (as part of the same daily job as recovery_radar_phase1.py,
right after candle ingestion). Identifies which currently-listed coins are
RIGHT NOW in a genuine "quiet compression" phase, based on the pattern
validated against the 40-coin explosion history:
    - 27/29 testable historical explosions (93%) showed a real quiet period
      before breaking out, with a median max quiet-streak of 38 days.
    - "Quiet" is defined RELATIVE to the market's own typical volatility
      (full-market median 30-day range was measured at ~145%), not an
      arbitrary fixed number.

This script does NOT predict anything by itself - it narrows ~490 coins down
to a short "Watchlist" of genuine compression candidates and records each
one's current range ceiling/floor. The separate, frequently-running
check_ignition.py then watches ONLY this short list for a live breakout.

Usage: python update_watchlist.py
Reads/writes: recovery_radar.db (must already exist and be freshly ingested)
"""

from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timezone

DB_PATH = "recovery_radar.db"

ROLLING_WINDOW = 30          # days, for the "current range" and "ceiling/floor"
QUIET_LOOKBACK = 90          # days, for measuring %-time-quiet
QUIET_THRESHOLD_FACTOR = 0.5 # "quiet" = current 30d range < 0.5 * market median
MIN_QUIET_PCT = 50.0         # must have spent at least this % of the lookback quiet
MIN_HISTORY_DAYS = ROLLING_WINDOW + QUIET_LOOKBACK  # 120 days needed

# Coins to NEVER treat as compression candidates - pegged/stable assets are
# trivially "tight" by design and would otherwise dominate the watchlist.
# Verified against this database on 2026-08-13: UUSDT and KGSTUSDT are real
# stablecoins (fiat-pegged), not data errors - caught by the 'USD' substring
# rule and this explicit set respectively.
STABLE_PEGGED_EXCLUDE = {
    "EUR", "EURI", "AEUR", "U", "KGST", "PAXG", "XAUT", "WBTC", "WBETH",
}


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS watchlist (
            symbol TEXT PRIMARY KEY,
            computed_date TEXT NOT NULL,
            in_compression INTEGER NOT NULL,
            quiet_pct_90d REAL,
            current_30d_range_pct REAL,
            range_ceiling REAL,     -- highest High in the last 30 days (breakout level)
            range_floor REAL,       -- lowest Low in the last 30 days
            market_baseline_used REAL,
            alerted INTEGER NOT NULL DEFAULT 0,
            alerted_at TEXT
        );
        """
    )
    conn.commit()
    return conn


def is_excluded(base_asset: str) -> bool:
    if base_asset in STABLE_PEGGED_EXCLUDE:
        return True
    if "USD" in base_asset.upper():
        return True
    return False


def compute_market_baseline(conn: sqlite3.Connection) -> float:
    """Median 30-day range% across all eligible (non-excluded) symbols today -
    this is what 'quiet relative to the market' is measured against."""
    cur = conn.execute("SELECT symbol, base_asset FROM universe")
    symbols = [(s, b) for s, b in cur.fetchall() if not is_excluded(b)]

    values = []
    for symbol, _ in symbols:
        cur2 = conn.execute(
            """SELECT high, low FROM candles_daily
               WHERE symbol=? ORDER BY open_time_utc DESC LIMIT ?""",
            (symbol, ROLLING_WINDOW),
        )
        rows = cur2.fetchall()
        if len(rows) < ROLLING_WINDOW:
            continue
        highs = [r[0] for r in rows]
        lows = [r[1] for r in rows]
        if min(lows) <= 0:
            continue
        values.append((max(highs) - min(lows)) / min(lows) * 100)

    if not values:
        return 145.0  # fallback to the last known measured baseline
    values.sort()
    n = len(values)
    return values[n // 2] if n % 2 else (values[n // 2 - 1] + values[n // 2]) / 2


def rolling_range_series(candles: list[tuple], window: int) -> list[float]:
    """candles must be chronological [(day, close, high, low), ...]. Returns
    range% at each point once enough history exists."""
    out = []
    for i in range(window, len(candles) + 1):
        chunk = candles[i - window : i]
        highs = [c[2] for c in chunk]
        lows = [c[3] for c in chunk]
        if min(lows) <= 0:
            out.append(None)
            continue
        out.append((max(highs) - min(lows)) / min(lows) * 100)
    return out


def evaluate_symbol(conn: sqlite3.Connection, symbol: str, market_baseline: float) -> dict | None:
    cur = conn.execute(
        """SELECT open_time_utc, close, high, low FROM candles_daily
           WHERE symbol=? ORDER BY open_time_utc""",
        (symbol,),
    )
    candles = cur.fetchall()
    if len(candles) < MIN_HISTORY_DAYS:
        return None

    threshold = market_baseline * QUIET_THRESHOLD_FACTOR
    rr = rolling_range_series(candles, ROLLING_WINDOW)
    recent = rr[-QUIET_LOOKBACK:]
    recent_valid = [v for v in recent if v is not None]
    if not recent_valid:
        return None

    current_range_pct = recent_valid[-1]
    quiet_pct = sum(1 for v in recent_valid if v < threshold) / len(recent_valid) * 100
    in_compression = current_range_pct < threshold and quiet_pct >= MIN_QUIET_PCT

    last30 = candles[-ROLLING_WINDOW:]
    ceiling = max(c[2] for c in last30)  # highest High
    floor = min(c[3] for c in last30)    # lowest Low

    return {
        "in_compression": in_compression,
        "quiet_pct_90d": round(quiet_pct, 1),
        "current_30d_range_pct": round(current_range_pct, 1),
        "range_ceiling": ceiling,
        "range_floor": floor,
    }


def run() -> None:
    conn = get_conn()
    print("[1/2] Computing today's market baseline (median 30-day range across eligible coins)...")
    baseline = compute_market_baseline(conn)
    threshold = baseline * QUIET_THRESHOLD_FACTOR
    print(f"      Market baseline: {baseline:.1f}%   Quiet threshold: {threshold:.1f}%")

    cur = conn.execute("SELECT symbol, base_asset FROM universe")
    symbols = [(s, b) for s, b in cur.fetchall() if not is_excluded(b)]
    print(f"[2/2] Evaluating {len(symbols)} eligible symbols (stablecoins/pegged assets excluded)...")

    today = datetime.now(timezone.utc).date().isoformat()
    in_compression_count = 0
    for symbol, _ in symbols:
        result = evaluate_symbol(conn, symbol, baseline)
        if result is None:
            continue

        # check if the ceiling changed since last time - if so, this is a NEW
        # compression cycle and any previous alert should be cleared
        cur2 = conn.execute("SELECT range_ceiling, alerted FROM watchlist WHERE symbol=?", (symbol,))
        prev = cur2.fetchone()
        alerted = 0
        alerted_at = None
        if prev and prev[0] is not None and abs(prev[0] - result["range_ceiling"]) < 1e-12:
            alerted = prev[1]  # ceiling unchanged, keep existing alert state

        conn.execute(
            """
            INSERT INTO watchlist
                (symbol, computed_date, in_compression, quiet_pct_90d,
                 current_30d_range_pct, range_ceiling, range_floor,
                 market_baseline_used, alerted, alerted_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol) DO UPDATE SET
                computed_date=excluded.computed_date,
                in_compression=excluded.in_compression,
                quiet_pct_90d=excluded.quiet_pct_90d,
                current_30d_range_pct=excluded.current_30d_range_pct,
                range_ceiling=excluded.range_ceiling,
                range_floor=excluded.range_floor,
                market_baseline_used=excluded.market_baseline_used,
                alerted=excluded.alerted,
                alerted_at=CASE WHEN excluded.alerted=0 THEN NULL ELSE watchlist.alerted_at END
            """,
            (
                symbol, today, int(result["in_compression"]), result["quiet_pct_90d"],
                result["current_30d_range_pct"], result["range_ceiling"], result["range_floor"],
                baseline, alerted, alerted_at,
            ),
        )
        if result["in_compression"]:
            in_compression_count += 1

    conn.commit()
    conn.close()
    print(f"Done. {in_compression_count} symbols currently in compression (on the watchlist).")


if __name__ == "__main__":
    run()
