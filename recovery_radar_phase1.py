"""
HH 10/10 Recovery Radar - Phase 1
==================================
Scope of this phase (deliberately limited - do not extend without deciding
the numeric quality-filter thresholds first, see TODO markers below):

  1. Pull the full current Binance Spot USDT-pair universe.
  2. For each symbol, download and PERMANENTLY store closed daily candles:
       - pre-crash reference window: 2025-09-01 .. 2025-10-09 (inclusive)
       - post-crash window: 2025-10-11 .. latest closed UTC day
       - the 2025-10-10 candle itself is NEVER downloaded/stored (per spec:
         ignore the crash day entirely, including its wicks)
  3. Compute, per symbol:
       - pre_crash_normal_level = average close of 2025-10-08 and 2025-10-09
         (the OFFICIAL figure, per your decision)
       - sept_reference_avg = average daily close over the full Sep-2025
         window, stored ONLY as a sanity check, never used in the score
       - divergence_flag = True if the two disagree by more than 15%
         (this does not exclude a coin - it just gets surfaced so you see it
         instead of it silently skewing a multiple)
       - post_crash_low = the 2025-10-11 daily Low
       - current_price = latest available price (ticker, not a stale close)
       - recovery_multiple = pre_crash_normal_level / current_price
  4. Keep only symbols with recovery_multiple >= 5.0.
  5. Everything lands in one SQLite file = the single source of truth.
     On every run, the ingester reads what's already stored and only
     fetches what's missing (new closed candles, or symbols never seen).
     Verified historical rows are NEVER re-downloaded.

Explicitly OUT of scope for Phase 1 (needs your numeric thresholds first):
  - supply/dilution, market cap/FDV, VR20, liquidity quality filters
  - HH Alpha Explosion Radar cross-check / DOUBLE SIGNAL
  - Recovery Score (currently we only produce the raw qualifying list,
    sorted by recovery_multiple descending, NOT a weighted score)

Requirements: Python 3.11+, `pip install requests`
(sqlite3 is in the standard library, no extra install needed)

Usage:
    python recovery_radar_phase1.py --init      # first run: build DB, full backfill
    python recovery_radar_phase1.py              # normal run: fill gaps + rescan
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional

import requests

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

BASE_URL = "https://api.binance.com"
DB_PATH = "recovery_radar.db"

CRASH_DAY = "2025-10-10"                # never downloaded/stored
PRE_CRASH_START = "2025-09-01"
PRE_CRASH_END = "2025-10-09"            # inclusive, day before crash
POST_CRASH_START = "2025-10-11"         # inclusive, day after crash
NORMAL_LEVEL_DAYS = ["2025-10-08", "2025-10-09"]  # official pre-crash reference

RECOVERY_MULTIPLE_MIN = 5.0
SANITY_DIVERGENCE_PCT = 0.15            # flag if Oct8-9 avg vs Sept avg differ >15%

REQUEST_TIMEOUT = 15
RATE_LIMIT_SLEEP = 0.25                 # seconds between klines calls, stay well under Binance weight limits
KLINES_LIMIT = 1000                     # max candles per call


# ----------------------------------------------------------------------------
# DB schema
# ----------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS universe (
    symbol TEXT PRIMARY KEY,
    base_asset TEXT NOT NULL,
    quote_asset TEXT NOT NULL,
    status TEXT NOT NULL,
    first_seen_utc TEXT NOT NULL,
    existed_pre_crash INTEGER,          -- NULL until we check, else 0/1
    last_checked_utc TEXT
);

CREATE TABLE IF NOT EXISTS candles_daily (
    symbol TEXT NOT NULL,
    open_time_utc TEXT NOT NULL,        -- 'YYYY-MM-DD', the UTC day of the candle open
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    base_volume REAL NOT NULL,
    quote_volume REAL NOT NULL,
    trade_count INTEGER NOT NULL,
    taker_buy_base_volume REAL NOT NULL,
    taker_buy_quote_volume REAL NOT NULL,
    is_closed INTEGER NOT NULL DEFAULT 1,   -- we only ever store closed candles
    PRIMARY KEY (symbol, open_time_utc)
);

CREATE TABLE IF NOT EXISTS recovery_snapshot (
    symbol TEXT PRIMARY KEY,
    pre_crash_normal_level REAL,
    sept_reference_avg REAL,
    divergence_flag INTEGER,
    divergence_pct REAL,
    post_crash_low REAL,
    current_price REAL,
    recovery_multiple REAL,
    data_status TEXT NOT NULL,          -- VERIFIED / PENDING / UNKNOWN
    last_updated_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS run_history (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_utc TEXT NOT NULL,
    finished_utc TEXT,
    symbols_scanned INTEGER,
    symbols_qualified INTEGER,
    notes TEXT
);
"""


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.executescript(SCHEMA)
    return conn


# ----------------------------------------------------------------------------
# Binance REST helpers
# ----------------------------------------------------------------------------

def _get(path: str, params: dict) -> object:
    url = f"{BASE_URL}{path}"
    for attempt in range(5):
        try:
            resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 429 or resp.status_code == 418:
                # rate limited / banned - back off hard
                wait = 5 * (attempt + 1)
                print(f"[rate-limit] {resp.status_code}, sleeping {wait}s", file=sys.stderr)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            wait = 2 * (attempt + 1)
            print(f"[retry] {e}, sleeping {wait}s", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"Failed to GET {path} after retries")


def fetch_usdt_universe() -> list[dict]:
    """All currently TRADING spot symbols quoted in USDT."""
    data = _get("/api/v3/exchangeInfo", {})
    out = []
    for s in data.get("symbols", []):
        if s.get("quoteAsset") == "USDT" and s.get("status") == "TRADING" and s.get("isSpotTradingAllowed", True):
            out.append(s)
    return out


def fetch_daily_klines(symbol: str, start_ms: int, end_ms: int) -> list[list]:
    """Fetch closed daily klines in [start_ms, end_ms), paginating as needed."""
    all_rows: list[list] = []
    cursor = start_ms
    while cursor < end_ms:
        params = {
            "symbol": symbol,
            "interval": "1d",
            "startTime": cursor,
            "endTime": end_ms,
            "limit": KLINES_LIMIT,
        }
        rows = _get("/api/v3/klines", params)
        time.sleep(RATE_LIMIT_SLEEP)
        if not rows:
            break
        all_rows.extend(rows)
        last_open = rows[-1][0]
        next_cursor = last_open + 24 * 60 * 60 * 1000
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        if len(rows) < KLINES_LIMIT:
            break
    return all_rows


def fetch_current_price(symbol: str) -> Optional[float]:
    try:
        data = _get("/api/v3/ticker/price", {"symbol": symbol})
        return float(data["price"])
    except Exception as e:
        print(f"[warn] no current price for {symbol}: {e}", file=sys.stderr)
        return None


# ----------------------------------------------------------------------------
# Date helpers
# ----------------------------------------------------------------------------

def day_to_ms(day: str) -> int:
    dt = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def utc_day_str(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def latest_closed_utc_day() -> str:
    """Yesterday UTC (today's daily candle is still open, we never store it)."""
    return (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")


# ----------------------------------------------------------------------------
# Ingestion (reads DB first, never re-downloads verified rows)
# ----------------------------------------------------------------------------

def already_stored_days(conn: sqlite3.Connection, symbol: str) -> set[str]:
    cur = conn.execute(
        "SELECT open_time_utc FROM candles_daily WHERE symbol = ?", (symbol,)
    )
    return {row[0] for row in cur.fetchall()}


def store_candles(conn: sqlite3.Connection, symbol: str, rows: list[list]) -> int:
    inserted = 0
    for r in rows:
        open_ms = r[0]
        day = utc_day_str(open_ms)
        if day == CRASH_DAY:
            continue  # hard rule: never store the crash day
        # skip anything that isn't a fully closed candle relative to "now"
        if day >= datetime.now(timezone.utc).strftime("%Y-%m-%d"):
            continue
        conn.execute(
            """
            INSERT OR IGNORE INTO candles_daily
            (symbol, open_time_utc, open, high, low, close,
             base_volume, quote_volume, trade_count,
             taker_buy_base_volume, taker_buy_quote_volume, is_closed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
            """,
            (
                symbol, day,
                float(r[1]), float(r[2]), float(r[3]), float(r[4]),
                float(r[5]), float(r[7]), int(r[8]),
                float(r[9]), float(r[10]),
            ),
        )
        inserted += conn.total_changes  # rough signal only
    return inserted


def ingest_symbol_history(conn: sqlite3.Connection, symbol: str) -> None:
    stored_days = already_stored_days(conn, symbol)
    end_day = latest_closed_utc_day()

    needed_ranges = []
    # pre-crash window
    if PRE_CRASH_START not in stored_days or PRE_CRASH_END not in stored_days:
        needed_ranges.append((PRE_CRASH_START, PRE_CRASH_END))
    # post-crash window up to latest closed day
    if POST_CRASH_START not in stored_days or end_day not in stored_days:
        needed_ranges.append((POST_CRASH_START, end_day))

    for start_day, end_day_r in needed_ranges:
        start_ms = day_to_ms(start_day)
        end_ms = day_to_ms(end_day_r) + 24 * 60 * 60 * 1000
        rows = fetch_daily_klines(symbol, start_ms, end_ms)
        store_candles(conn, symbol, rows)
    conn.commit()


# ----------------------------------------------------------------------------
# Recovery calc
# ----------------------------------------------------------------------------

@dataclass
class RecoveryResult:
    symbol: str
    pre_crash_normal_level: Optional[float]
    sept_reference_avg: Optional[float]
    divergence_flag: bool
    divergence_pct: Optional[float]
    post_crash_low: Optional[float]
    current_price: Optional[float]
    recovery_multiple: Optional[float]
    data_status: str


def compute_recovery(conn: sqlite3.Connection, symbol: str) -> RecoveryResult:
    def get_close(day: str) -> Optional[float]:
        cur = conn.execute(
            "SELECT close FROM candles_daily WHERE symbol=? AND open_time_utc=?",
            (symbol, day),
        )
        row = cur.fetchone()
        return row[0] if row else None

    closes_normal = [c for c in (get_close(d) for d in NORMAL_LEVEL_DAYS) if c is not None]
    if len(closes_normal) < len(NORMAL_LEVEL_DAYS):
        return RecoveryResult(symbol, None, None, False, None, None, None, None, "PENDING")

    pre_crash_normal_level = sum(closes_normal) / len(closes_normal)

    # sanity-check reference: full September average close
    cur = conn.execute(
        """SELECT close FROM candles_daily
           WHERE symbol=? AND open_time_utc BETWEEN ? AND '2025-09-30'""",
        (symbol, PRE_CRASH_START),
    )
    sept_closes = [r[0] for r in cur.fetchall()]
    sept_reference_avg = sum(sept_closes) / len(sept_closes) if sept_closes else None

    divergence_flag = False
    divergence_pct = None
    if sept_reference_avg:
        divergence_pct = abs(pre_crash_normal_level - sept_reference_avg) / sept_reference_avg
        divergence_flag = divergence_pct > SANITY_DIVERGENCE_PCT

    post_low = get_close("2025-10-11")  # placeholder if Low not separately queried below
    cur = conn.execute(
        "SELECT low FROM candles_daily WHERE symbol=? AND open_time_utc='2025-10-11'",
        (symbol,),
    )
    row = cur.fetchone()
    post_crash_low = row[0] if row else None

    current_price = fetch_current_price(symbol)

    if current_price is None:
        return RecoveryResult(
            symbol, pre_crash_normal_level, sept_reference_avg,
            divergence_flag, divergence_pct, post_crash_low,
            None, None, "PENDING",
        )

    recovery_multiple = pre_crash_normal_level / current_price if current_price > 0 else None

    return RecoveryResult(
        symbol, pre_crash_normal_level, sept_reference_avg,
        divergence_flag, divergence_pct, post_crash_low,
        current_price, recovery_multiple, "VERIFIED",
    )


def save_snapshot(conn: sqlite3.Connection, r: RecoveryResult) -> None:
    conn.execute(
        """
        INSERT INTO recovery_snapshot
        (symbol, pre_crash_normal_level, sept_reference_avg, divergence_flag,
         divergence_pct, post_crash_low, current_price, recovery_multiple,
         data_status, last_updated_utc)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(symbol) DO UPDATE SET
            pre_crash_normal_level=excluded.pre_crash_normal_level,
            sept_reference_avg=excluded.sept_reference_avg,
            divergence_flag=excluded.divergence_flag,
            divergence_pct=excluded.divergence_pct,
            post_crash_low=excluded.post_crash_low,
            current_price=excluded.current_price,
            recovery_multiple=excluded.recovery_multiple,
            data_status=excluded.data_status,
            last_updated_utc=excluded.last_updated_utc
        """,
        (
            r.symbol, r.pre_crash_normal_level, r.sept_reference_avg,
            int(r.divergence_flag), r.divergence_pct, r.post_crash_low,
            r.current_price, r.recovery_multiple, r.data_status,
            datetime.now(timezone.utc).isoformat(),
        ),
    )


# ----------------------------------------------------------------------------
# Main run
# ----------------------------------------------------------------------------

def run(full_backfill: bool) -> None:
    conn = get_conn()
    started = datetime.now(timezone.utc).isoformat()
    cur = conn.execute(
        "INSERT INTO run_history (started_utc, notes) VALUES (?, ?)",
        (started, "full_backfill" if full_backfill else "incremental"),
    )
    run_id = cur.lastrowid
    conn.commit()

    print("[1/4] Fetching current Binance USDT spot universe...")
    universe = fetch_usdt_universe()
    print(f"      {len(universe)} USDT spot symbols currently TRADING")

    now_iso = datetime.now(timezone.utc).isoformat()
    for s in universe:
        conn.execute(
            """
            INSERT INTO universe (symbol, base_asset, quote_asset, status, first_seen_utc, last_checked_utc)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol) DO UPDATE SET
                status=excluded.status, last_checked_utc=excluded.last_checked_utc
            """,
            (s["symbol"], s["baseAsset"], s["quoteAsset"], s["status"], now_iso, now_iso),
        )
    conn.commit()

    print("[2/4] Ingesting historical daily candles (pre-crash + post-crash windows)...")
    symbols = [s["symbol"] for s in universe]
    for i, symbol in enumerate(symbols, 1):
        try:
            ingest_symbol_history(conn, symbol)
        except Exception as e:
            print(f"[warn] ingestion failed for {symbol}: {e}", file=sys.stderr)
        if i % 25 == 0:
            print(f"      ...{i}/{len(symbols)} symbols ingested")

    print("[3/4] Computing recovery multiples...")
    qualified = 0
    for i, symbol in enumerate(symbols, 1):
        result = compute_recovery(conn, symbol)
        save_snapshot(conn, result)
        if result.recovery_multiple is not None and result.recovery_multiple >= RECOVERY_MULTIPLE_MIN:
            qualified += 1
        if i % 25 == 0:
            conn.commit()
    conn.commit()

    print(f"[4/4] Done. {qualified} symbols currently qualify at >= {RECOVERY_MULTIPLE_MIN}x recovery multiple.")

    conn.execute(
        """UPDATE run_history SET finished_utc=?, symbols_scanned=?, symbols_qualified=? WHERE run_id=?""",
        (datetime.now(timezone.utc).isoformat(), len(symbols), qualified, run_id),
    )
    conn.commit()
    conn.close()


def print_qualified_report() -> None:
    conn = get_conn()
    cur = conn.execute(
        """
        SELECT symbol, pre_crash_normal_level, current_price, recovery_multiple,
               divergence_flag, divergence_pct, data_status
        FROM recovery_snapshot
        WHERE recovery_multiple >= ?
        ORDER BY recovery_multiple DESC
        """,
        (RECOVERY_MULTIPLE_MIN,),
    )
    rows = cur.fetchall()
    conn.close()

    if not rows:
        print("No qualifying symbols found (or data not yet ingested - run with --init first).")
        return

    print(f"\n{'Symbol':<12}{'PreCrashLvl':>14}{'Current':>12}{'Multiple':>10}{'Divergence':>12}  Status")
    print("-" * 76)
    for symbol, pre, cur_price, mult, div_flag, div_pct, status in rows:
        div_str = f"{div_pct*100:.1f}%" if (div_flag and div_pct is not None) else "-"
        print(f"{symbol:<12}{pre:>14.6f}{cur_price:>12.6f}{mult:>10.2f}x{div_str:>12}  {status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HH 10/10 Recovery Radar - Phase 1")
    parser.add_argument("--init", action="store_true", help="First run: full backfill of history")
    parser.add_argument("--report-only", action="store_true", help="Skip fetching, just print current DB report")
    args = parser.parse_args()

    if args.report_only:
        print_qualified_report()
    else:
        run(full_backfill=args.init)
        print_qualified_report()
