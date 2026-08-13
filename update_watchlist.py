"""
Watchlist Builder - Phase 2 (v2: self-relative percentile compression)
========================================================================
Runs once per day (as part of the same daily job as recovery_radar_phase1.py,
right after candle ingestion). Identifies which currently-listed coins are
RIGHT NOW in a genuine "quiet compression" phase, based on the pattern
validated against the 40-coin explosion history (27/29 testable explosions
showed a real quiet period first, median max quiet-streak 38 days).

DESIGN NOTE - why this isn't "range < X% of market average":
A shared market-wide threshold and a fixed calendar window (e.g. "180 days")
both failed a real backtest against TUT/USDT's actual pre-explosion history -
TUT's own quietest point (~21.7% 30-day range) never dropped as low as a
generic market-wide threshold demanded, simply because TUT is naturally a
much more volatile coin than the market average. A quiet coin and a wild
coin can't fairly share one bar.

Instead, each coin is compared ONLY to ITS OWN historical range distribution
(self-relative percentile rank) - no fixed day-count, no shared market
number. A coin qualifies when its current 30-day rolling range sits in the
bottom PERCENTILE_THRESHOLD% of its own full available history, and that
tightness has persisted for a meaningful chunk of the recent period (not
just a single lucky day).

Usage: python update_watchlist.py
Reads/writes: recovery_radar.db (must already exist and be freshly ingested)
"""

from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timezone

DB_PATH = "recovery_radar.db"

ROLLING_WINDOW = 30           # days, for the "current range" and ceiling/floor
PERCENTILE_THRESHOLD = 15.0   # must be in the tightest 15% of the coin's OWN history
PERSISTENCE_LOOKBACK = 60     # days, how far back to check persistence
MIN_PERSISTENCE_PCT = 30.0    # at least this % of the last 60 days must also be this tight
MIN_HISTORY_FOR_PERCENTILE = 150  # need a reasonably large sample to trust a percentile rank

# Coins to NEVER treat as compression candidates - pegged/stable assets are
# trivially "tight" by design and would otherwise dominate the watchlist.
STABLE_PEGGED_EXCLUDE = {
    "EUR", "EURI", "AEUR", "U", "KGST", "PAXG", "XAUT", "WBTC", "WBETH",
}


_NEW_WATCHLIST_COLUMNS = [
    ("current_percentile", "REAL"),
    ("persistence_pct", "REAL"),
    ("history_days", "INTEGER"),
    ("current_price", "REAL"),
    ("proximity_pct", "REAL"),      # 0=at 30d floor, 100=at 30d ceiling
    ("explosion_score", "REAL"),    # composite readiness score, 0-100, higher=closer to a breakout
]


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS watchlist (
            symbol TEXT PRIMARY KEY,
            computed_date TEXT NOT NULL,
            in_compression INTEGER NOT NULL,
            current_percentile REAL,       -- where today's range sits in the coin's OWN history (0-100, lower=tighter)
            persistence_pct REAL,          -- % of last 60 days also this tight
            current_30d_range_pct REAL,
            range_ceiling REAL,     -- highest High in the last 30 days (breakout level)
            range_floor REAL,       -- lowest Low in the last 30 days
            history_days INTEGER,
            current_price REAL,
            proximity_pct REAL,
            explosion_score REAL,
            alerted INTEGER NOT NULL DEFAULT 0,
            alerted_at TEXT
        );
        """
    )
    conn.commit()
    # the table may already exist on disk from an earlier schema version
    # (quiet_pct_90d/market_baseline_used) - add any missing new columns
    existing = {row[1] for row in conn.execute("PRAGMA table_info(watchlist)").fetchall()}
    for col_name, col_type in _NEW_WATCHLIST_COLUMNS:
        if col_name not in existing:
            conn.execute(f"ALTER TABLE watchlist ADD COLUMN {col_name} {col_type}")
    conn.commit()
    return conn


def is_excluded(base_asset: str) -> bool:
    if base_asset in STABLE_PEGGED_EXCLUDE:
        return True
    if "USD" in base_asset.upper():
        return True
    return False


def compute_range_series(candles: list[tuple], window: int) -> list[float]:
    """candles must be chronological [(day, high, low), ...]. Returns range%
    at each point once enough history exists (skips windows with a zero low)."""
    out = []
    for i in range(window, len(candles) + 1):
        chunk = candles[i - window : i]
        highs = [c[1] for c in chunk]
        lows = [c[2] for c in chunk]
        if min(lows) <= 0:
            continue
        out.append((max(highs) - min(lows)) / min(lows) * 100)
    return out


def percentile_rank(series: list[float], value: float) -> float:
    """% of the series that is <= value (0 = tightest ever, 100 = widest ever)."""
    if not series:
        return 100.0
    below_or_equal = sum(1 for v in series if v <= value)
    return below_or_equal / len(series) * 100


def evaluate_symbol(conn: sqlite3.Connection, symbol: str) -> dict | None:
    cur = conn.execute(
        """SELECT open_time_utc, high, low FROM candles_daily
           WHERE symbol=? ORDER BY open_time_utc""",
        (symbol,),
    )
    candles = cur.fetchall()
    if len(candles) < MIN_HISTORY_FOR_PERCENTILE + ROLLING_WINDOW:
        return None

    range_series = compute_range_series(candles, ROLLING_WINDOW)
    if len(range_series) < MIN_HISTORY_FOR_PERCENTILE:
        return None

    current_range = range_series[-1]
    current_percentile = percentile_rank(range_series, current_range)

    # persistence: what fraction of the recent lookback was ALSO this tight,
    # using the value that marks the percentile cutoff (self-relative, not a
    # separate arbitrary number)
    sorted_series = sorted(range_series)
    cutoff_idx = max(0, int(len(sorted_series) * PERCENTILE_THRESHOLD / 100) - 1)
    cutoff_value = sorted_series[cutoff_idx]
    recent = range_series[-PERSISTENCE_LOOKBACK:] if len(range_series) >= PERSISTENCE_LOOKBACK else range_series
    persistence_pct = sum(1 for v in recent if v <= cutoff_value) / len(recent) * 100

    in_compression = current_percentile <= PERCENTILE_THRESHOLD and persistence_pct >= MIN_PERSISTENCE_PCT

    last_window = candles[-ROLLING_WINDOW:]
    ceiling = max(c[1] for c in last_window)  # highest High
    floor = min(c[2] for c in last_window)    # lowest Low
    current_price = candles[-1][1]  # use the latest stored High as a close proxy (cheap, no extra API call)

    proximity_pct = 50.0
    if ceiling > floor:
        proximity_pct = max(0.0, min(100.0, (current_price - floor) / (ceiling - floor) * 100))

    tightness_score = max(0.0, 100.0 - current_percentile)
    explosion_score = 0.4 * tightness_score + 0.3 * persistence_pct + 0.3 * proximity_pct

    return {
        "in_compression": in_compression,
        "current_percentile": round(current_percentile, 1),
        "persistence_pct": round(persistence_pct, 1),
        "current_30d_range_pct": round(current_range, 1),
        "range_ceiling": ceiling,
        "range_floor": floor,
        "history_days": len(candles),
        "current_price": current_price,
        "proximity_pct": round(proximity_pct, 1),
        "explosion_score": round(explosion_score, 1),
    }


def run() -> None:
    conn = get_conn()
    cur = conn.execute("SELECT symbol, base_asset FROM universe")
    symbols = [(s, b) for s, b in cur.fetchall() if not is_excluded(b)]
    print(f"[1/1] Evaluating {len(symbols)} eligible symbols against their OWN historical range distribution...")
    print(f"      (stablecoins/pegged assets excluded; need {MIN_HISTORY_FOR_PERCENTILE}+ days of history to qualify)")

    today = datetime.now(timezone.utc).date().isoformat()
    in_compression_count = 0
    skipped_short_history = 0
    for symbol, _ in symbols:
        result = evaluate_symbol(conn, symbol)
        if result is None:
            skipped_short_history += 1
            continue

        cur2 = conn.execute("SELECT range_ceiling, alerted FROM watchlist WHERE symbol=?", (symbol,))
        prev = cur2.fetchone()
        alerted = 0
        alerted_at = None
        if prev and prev[0] is not None and abs(prev[0] - result["range_ceiling"]) < 1e-12:
            alerted = prev[1]  # ceiling unchanged, keep existing alert state

        conn.execute(
            """
            INSERT INTO watchlist
                (symbol, computed_date, in_compression, current_percentile,
                 persistence_pct, current_30d_range_pct, range_ceiling, range_floor,
                 history_days, current_price, proximity_pct, explosion_score,
                 alerted, alerted_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol) DO UPDATE SET
                computed_date=excluded.computed_date,
                in_compression=excluded.in_compression,
                current_percentile=excluded.current_percentile,
                persistence_pct=excluded.persistence_pct,
                current_30d_range_pct=excluded.current_30d_range_pct,
                range_ceiling=excluded.range_ceiling,
                range_floor=excluded.range_floor,
                history_days=excluded.history_days,
                current_price=excluded.current_price,
                proximity_pct=excluded.proximity_pct,
                explosion_score=excluded.explosion_score,
                alerted=excluded.alerted,
                alerted_at=CASE WHEN excluded.alerted=0 THEN NULL ELSE watchlist.alerted_at END
            """,
            (
                symbol, today, int(result["in_compression"]), result["current_percentile"],
                result["persistence_pct"], result["current_30d_range_pct"],
                result["range_ceiling"], result["range_floor"], result["history_days"],
                result["current_price"], result["proximity_pct"], result["explosion_score"],
                alerted, alerted_at,
            ),
        )
        if result["in_compression"]:
            in_compression_count += 1

    conn.commit()
    conn.close()
    print(f"Done. {in_compression_count} symbols currently in compression (self-relative percentile).")
    print(f"({skipped_short_history} symbols skipped for insufficient history - need {MIN_HISTORY_FOR_PERCENTILE}+ days)")


if __name__ == "__main__":
    run()
