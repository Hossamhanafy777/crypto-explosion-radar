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
       - post_crash_low_alltime = the lowest daily Low reached at ANY point
         from 2025-10-11 to now (a running all-time-low, moves down only)
       - ever_touched_5x = walking day by day from 2025-10-11 forward, was
         that day's High ever >= 5x the running low established up to and
         including that day? Checked over the WHOLE history, not just now.
         A coin that touched 5x even once (wick or close, doesn't matter)
         and fell back still counts as "touched" - permanent, one-way flag.
       - current_multiple_from_low = current_price / post_crash_low_alltime
         (also checked live against the 5x threshold, in case today's
         still-open candle would trip it before its High is ever stored)
       - pre_crash_normal_level = average close of 2025-10-08 and 2025-10-09
         (the OFFICIAL figure, per your decision) - NO LONGER a gate, now a
         SCORING INPUT: pre_crash_normal_level / current_price estimates
         theoretical upside if the coin ever reclaimed its old level
       - sept_reference_avg = average daily close over the full Sep-2025
         window, stored ONLY as a sanity check on pre_crash_normal_level
       - divergence_flag = True if the two disagree by more than 15%
         (surfaced, not excluding - so a distorted 2-day reference doesn't
         silently skew the theoretical-upside score)
  4. NO FILTERING/GATING happens in this phase. Every currently-listed USDT
     symbol (the full Binance Spot universe, not just visibly-crashed coins)
     gets a row in recovery_snapshot with all of the above computed and
     stored. ever_touched_5x is stored as an INFORMATIONAL flag only - it is
     NOT used to exclude anything here. Strategy/scoring/filtering is a
     separate layer applied on top of this data later, and can be changed
     freely without re-collecting anything.
  5. ath_price / ath_date = the absolute all-time-high ever recorded for the
     symbol on Binance (lifetime, back to its listing date if needed),
     combined from a one-time historical scan (pre-Sept-2025, price+date
     only - old candles themselves are NOT stored) and the max High already
     present in candles_daily (Sept 2025 onward, which we do store in full).
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

BASE_URL = "https://data-api.binance.vision"
DB_PATH = "recovery_radar.db"

CRASH_DAY = "2025-10-10"                # never downloaded/stored
PRE_CRASH_START = "2025-09-01"
PRE_CRASH_END = "2025-10-09"            # inclusive, day before crash
POST_CRASH_START = "2025-10-11"         # inclusive, day after crash
NORMAL_LEVEL_DAYS = ["2025-10-08", "2025-10-09"]  # official pre-crash reference

EXPLOSION_MULTIPLE_THRESHOLD = 5.0      # GATE: exclude forever once High >= 5x the running post-crash low
UPSIDE_SCORE_REFERENCE = 5.0            # informational only, not a gate (kept for reference/back-compat)
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
    last_checked_utc TEXT,
    ath_pre_sept_price REAL,            -- ATH found in full lifetime history UP TO 2025-09-01, fetched once
    ath_pre_sept_date TEXT,
    ath_pre_sept_fetched INTEGER DEFAULT 0   -- 1 once the one-time lifetime scan is done (never re-fetched)
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
    post_crash_low REAL,                -- kept for back-compat (2025-10-11 Low only)
    current_price REAL,
    recovery_multiple REAL,             -- kept for back-compat = pre_crash_normal_level / current_price
    post_crash_low_alltime REAL,        -- NEW: lowest Low from 2025-10-11 to now, running
    low_reached_date TEXT,              -- NEW: which day set that low
    ever_touched_5x INTEGER,            -- NEW: 1 = permanently excluded, already moved 5x off its low
    touched_5x_date TEXT,               -- NEW: which day first tripped it (NULL if never)
    current_multiple_from_low REAL,     -- NEW: current_price / post_crash_low_alltime
    theoretical_upside REAL,            -- NEW: pre_crash_normal_level / current_price (scoring input, not gate)
    ath_price REAL,                     -- NEW: absolute all-time high on Binance (pre-Sept-2025 scan combined with stored candles)
    ath_date TEXT,                      -- NEW: date that ATH was set
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


_NEW_SNAPSHOT_COLUMNS = [
    ("post_crash_low_alltime", "REAL"),
    ("low_reached_date", "TEXT"),
    ("ever_touched_5x", "INTEGER"),
    ("touched_5x_date", "TEXT"),
    ("current_multiple_from_low", "REAL"),
    ("theoretical_upside", "REAL"),
    ("ath_price", "REAL"),
    ("ath_date", "TEXT"),
]

_NEW_UNIVERSE_COLUMNS = [
    ("ath_pre_sept_price", "REAL"),
    ("ath_pre_sept_date", "TEXT"),
    ("ath_pre_sept_fetched", "INTEGER DEFAULT 0"),
]


def _migrate(conn: sqlite3.Connection) -> None:
    """Add any new columns to an existing DB file committed under an older schema."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(recovery_snapshot)").fetchall()}
    for col_name, col_type in _NEW_SNAPSHOT_COLUMNS:
        if col_name not in existing:
            conn.execute(f"ALTER TABLE recovery_snapshot ADD COLUMN {col_name} {col_type}")

    existing_universe = {row[1] for row in conn.execute("PRAGMA table_info(universe)").fetchall()}
    for col_name, col_type in _NEW_UNIVERSE_COLUMNS:
        if col_name not in existing_universe:
            conn.execute(f"ALTER TABLE universe ADD COLUMN {col_name} {col_type}")
    conn.commit()


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.executescript(SCHEMA)
    _migrate(conn)
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


def fetch_lifetime_ath_pre_sept(symbol: str) -> tuple[Optional[float], Optional[str]]:
    """
    One-time scan: paginate the symbol's FULL daily history from listing up to
    (but not including) 2025-09-01, tracking the max High and its date in
    memory only - individual candles from this scan are never written to
    candles_daily (per your decision: store the ATH scalar, not the old bars).
    """
    end_ms = day_to_ms(PRE_CRASH_START)  # exclusive upper bound
    cursor = 0  # startTime=0 lets Binance return from the symbol's actual listing date
    ath_price: Optional[float] = None
    ath_date: Optional[str] = None

    while True:
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
        for r in rows:
            high = float(r[2])
            if ath_price is None or high > ath_price:
                ath_price = high
                ath_date = utc_day_str(r[0])
        last_open = rows[-1][0]
        next_cursor = last_open + 24 * 60 * 60 * 1000
        if next_cursor <= cursor or next_cursor >= end_ms or len(rows) < KLINES_LIMIT:
            break
        cursor = next_cursor

    return ath_price, ath_date


def ensure_ath(conn: sqlite3.Connection, symbol: str) -> None:
    """Fetch the one-time lifetime-pre-Sept ATH for a symbol if not already done."""
    cur = conn.execute(
        "SELECT ath_pre_sept_fetched FROM universe WHERE symbol=?", (symbol,)
    )
    row = cur.fetchone()
    if row and row[0]:
        return  # already fetched once, never re-fetch (per spec: verified history isn't re-downloaded)
    try:
        ath_price, ath_date = fetch_lifetime_ath_pre_sept(symbol)
    except Exception as e:
        print(f"[warn] ATH scan failed for {symbol}: {e}", file=sys.stderr)
        return
    conn.execute(
        """UPDATE universe SET ath_pre_sept_price=?, ath_pre_sept_date=?, ath_pre_sept_fetched=1
           WHERE symbol=?""",
        (ath_price, ath_date, symbol),
    )
    conn.commit()


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
    post_crash_low: Optional[float]              # back-compat: 2025-10-11 Low only
    current_price: Optional[float]
    recovery_multiple: Optional[float]            # back-compat: old formula
    post_crash_low_alltime: Optional[float]
    low_reached_date: Optional[str]
    ever_touched_5x: Optional[bool]
    touched_5x_date: Optional[str]
    current_multiple_from_low: Optional[float]
    theoretical_upside: Optional[float]
    ath_price: Optional[float]
    ath_date: Optional[str]
    data_status: str


def compute_low_and_explosion(
    conn: sqlite3.Connection, symbol: str, current_price: Optional[float]
) -> tuple[Optional[float], Optional[str], bool, Optional[str], Optional[float]]:
    """
    Walk the full post-crash daily history in chronological order, tracking a
    running all-time-low. On each day, check whether that day's High ever
    reached >= EXPLOSION_MULTIPLE_THRESHOLD x the running low up to and
    including that day. Once true, ever_touched_5x is permanently True
    (a wick counts - we don't care if it closed back down).

    Also checks the live current_price against the final running low, in
    case today's still-open candle would trip the threshold before its High
    is ever stored as a closed candle.
    """
    cur = conn.execute(
        """SELECT open_time_utc, high, low FROM candles_daily
           WHERE symbol=? AND open_time_utc >= ?
           ORDER BY open_time_utc ASC""",
        (symbol, POST_CRASH_START),
    )
    rows = cur.fetchall()
    if not rows:
        return None, None, False, None, None

    running_low = None
    low_reached_date = None
    ever_touched_5x = False
    touched_5x_date = None

    for day, high, low in rows:
        if running_low is None or low < running_low:
            running_low = low
            low_reached_date = day
        if not ever_touched_5x and running_low and high >= EXPLOSION_MULTIPLE_THRESHOLD * running_low:
            ever_touched_5x = True
            touched_5x_date = day

    current_multiple_from_low = None
    if running_low and current_price:
        current_multiple_from_low = current_price / running_low
        # live check: today's still-open candle might already exceed the threshold
        if not ever_touched_5x and current_multiple_from_low >= EXPLOSION_MULTIPLE_THRESHOLD:
            ever_touched_5x = True
            touched_5x_date = "live (current price)"

    return running_low, low_reached_date, ever_touched_5x, touched_5x_date, current_multiple_from_low


def compute_ath(conn: sqlite3.Connection, symbol: str) -> tuple[Optional[float], Optional[str]]:
    """
    Combine the one-time pre-Sept-2025 lifetime scan (universe table) with the
    max High among the daily candles we already store (Sept 2025 onward,
    which covers the crash and everything after) to get the true absolute
    all-time-high on Binance for this symbol.
    """
    cur = conn.execute(
        "SELECT ath_pre_sept_price, ath_pre_sept_date FROM universe WHERE symbol=?", (symbol,)
    )
    row = cur.fetchone()
    pre_sept_price, pre_sept_date = (row[0], row[1]) if row else (None, None)

    cur = conn.execute(
        """SELECT open_time_utc, high FROM candles_daily
           WHERE symbol=? ORDER BY high DESC LIMIT 1""",
        (symbol,),
    )
    row2 = cur.fetchone()
    stored_max_date, stored_max_high = (row2[0], row2[1]) if row2 else (None, None)

    candidates = []
    if pre_sept_price is not None:
        candidates.append((pre_sept_price, pre_sept_date))
    if stored_max_high is not None:
        candidates.append((stored_max_high, stored_max_date))
    if not candidates:
        return None, None
    return max(candidates, key=lambda t: t[0])


def compute_recovery(conn: sqlite3.Connection, symbol: str) -> RecoveryResult:
    def get_close(day: str) -> Optional[float]:
        cur = conn.execute(
            "SELECT close FROM candles_daily WHERE symbol=? AND open_time_utc=?",
            (symbol, day),
        )
        row = cur.fetchone()
        return row[0] if row else None

    closes_normal = [c for c in (get_close(d) for d in NORMAL_LEVEL_DAYS) if c is not None]
    current_price = fetch_current_price(symbol)

    post_crash_low_alltime, low_reached_date, ever_touched_5x, touched_5x_date, current_multiple_from_low = (
        compute_low_and_explosion(conn, symbol, current_price)
    )
    ath_price, ath_date = compute_ath(conn, symbol)

    if len(closes_normal) < len(NORMAL_LEVEL_DAYS) or post_crash_low_alltime is None:
        # not enough history to compute the theoretical-upside score yet,
        # but we may still have enough for the gate - report PENDING either way
        # so a human checks it rather than silently dropping it
        return RecoveryResult(
            symbol, None, None, False, None, None, current_price, None,
            post_crash_low_alltime, low_reached_date, ever_touched_5x, touched_5x_date,
            current_multiple_from_low, None, ath_price, ath_date, "PENDING",
        )

    pre_crash_normal_level = sum(closes_normal) / len(closes_normal)

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

    cur = conn.execute(
        "SELECT low FROM candles_daily WHERE symbol=? AND open_time_utc='2025-10-11'",
        (symbol,),
    )
    row = cur.fetchone()
    post_crash_low = row[0] if row else None

    if current_price is None:
        return RecoveryResult(
            symbol, pre_crash_normal_level, sept_reference_avg,
            divergence_flag, divergence_pct, post_crash_low,
            None, None,
            post_crash_low_alltime, low_reached_date, ever_touched_5x, touched_5x_date,
            current_multiple_from_low, None, ath_price, ath_date, "PENDING",
        )

    recovery_multiple = pre_crash_normal_level / current_price if current_price > 0 else None
    theoretical_upside = recovery_multiple  # same formula, renamed to reflect its new scoring role

    return RecoveryResult(
        symbol, pre_crash_normal_level, sept_reference_avg,
        divergence_flag, divergence_pct, post_crash_low,
        current_price, recovery_multiple,
        post_crash_low_alltime, low_reached_date, ever_touched_5x, touched_5x_date,
        current_multiple_from_low, theoretical_upside, ath_price, ath_date, "VERIFIED",
    )


def save_snapshot(conn: sqlite3.Connection, r: RecoveryResult) -> None:
    conn.execute(
        """
        INSERT INTO recovery_snapshot
        (symbol, pre_crash_normal_level, sept_reference_avg, divergence_flag,
         divergence_pct, post_crash_low, current_price, recovery_multiple,
         post_crash_low_alltime, low_reached_date, ever_touched_5x, touched_5x_date,
         current_multiple_from_low, theoretical_upside, ath_price, ath_date,
         data_status, last_updated_utc)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(symbol) DO UPDATE SET
            pre_crash_normal_level=excluded.pre_crash_normal_level,
            sept_reference_avg=excluded.sept_reference_avg,
            divergence_flag=excluded.divergence_flag,
            divergence_pct=excluded.divergence_pct,
            post_crash_low=excluded.post_crash_low,
            current_price=excluded.current_price,
            recovery_multiple=excluded.recovery_multiple,
            post_crash_low_alltime=excluded.post_crash_low_alltime,
            low_reached_date=excluded.low_reached_date,
            ever_touched_5x=excluded.ever_touched_5x,
            touched_5x_date=excluded.touched_5x_date,
            current_multiple_from_low=excluded.current_multiple_from_low,
            theoretical_upside=excluded.theoretical_upside,
            ath_price=excluded.ath_price,
            ath_date=excluded.ath_date,
            data_status=excluded.data_status,
            last_updated_utc=excluded.last_updated_utc
        """,
        (
            r.symbol, r.pre_crash_normal_level, r.sept_reference_avg,
            int(r.divergence_flag), r.divergence_pct, r.post_crash_low,
            r.current_price, r.recovery_multiple,
            r.post_crash_low_alltime, r.low_reached_date,
            None if r.ever_touched_5x is None else int(r.ever_touched_5x),
            r.touched_5x_date, r.current_multiple_from_low, r.theoretical_upside,
            r.ath_price, r.ath_date,
            r.data_status, datetime.now(timezone.utc).isoformat(),
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
        try:
            ensure_ath(conn, symbol)  # one-time lifetime ATH scan, skipped if already cached
        except Exception as e:
            print(f"[warn] ATH ensure failed for {symbol}: {e}", file=sys.stderr)
        if i % 25 == 0:
            print(f"      ...{i}/{len(symbols)} symbols ingested")

    print("[3/4] Computing post-crash lows, ATH and explosion-flag (informational, no filtering)...")
    touched_count = 0
    for i, symbol in enumerate(symbols, 1):
        result = compute_recovery(conn, symbol)
        save_snapshot(conn, result)
        if result.ever_touched_5x:
            touched_count += 1
        if i % 25 == 0:
            conn.commit()
    conn.commit()

    print(
        f"[4/4] Done. Data stored for {len(symbols)} symbols. "
        f"{touched_count} already touched {EXPLOSION_MULTIPLE_THRESHOLD}x off their post-crash low "
        f"(flagged, not excluded - strategy/filtering is applied separately on top of this data)."
    )

    conn.execute(
        """UPDATE run_history SET finished_utc=?, symbols_scanned=?, symbols_qualified=? WHERE run_id=?""",
        (datetime.now(timezone.utc).isoformat(), len(symbols), touched_count, run_id),
    )
    conn.commit()
    conn.close()


def print_qualified_report() -> None:
    conn = get_conn()
    cur = conn.execute(
        """
        SELECT symbol, post_crash_low_alltime, current_price, current_multiple_from_low,
               ath_price, ath_date, ever_touched_5x, theoretical_upside, data_status
        FROM recovery_snapshot
        ORDER BY symbol ASC
        """,
    )
    rows = cur.fetchall()
    conn.close()

    if not rows:
        print("No data found yet - run with --init first.")
        return

    print(
        f"\n{'Symbol':<12}{'PostCrashLow':>14}{'Current':>12}"
        f"{'FromLow':>10}{'ATH':>14}{'ATH Date':>12}{'Touched5x':>10}{'TheoUpside':>12}  Status"
    )
    print("-" * 110)
    for symbol, low, cur_price, from_low, ath, ath_date, touched, upside, status in rows:
        low_str = f"{low:.6f}" if low is not None else "-"
        cur_str = f"{cur_price:.6f}" if cur_price is not None else "-"
        from_low_str = f"{from_low:.2f}x" if from_low is not None else "-"
        ath_str = f"{ath:.6f}" if ath is not None else "-"
        ath_date_str = ath_date or "-"
        touched_str = "YES" if touched else ("no" if touched is not None else "-")
        upside_str = f"{upside:.2f}x" if upside is not None else "-"
        print(
            f"{symbol:<12}{low_str:>14}{cur_str:>12}"
            f"{from_low_str:>10}{ath_str:>14}{ath_date_str:>12}{touched_str:>10}{upside_str:>12}  {status}"
        )


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
