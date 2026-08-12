"""
Backtest: Do Precursor Signals Really Appear Before 5x Explosions?
===================================================================
Tests two hypotheses against data ALREADY stored in recovery_radar.db
(no new Binance calls needed - everything here is daily data we already
downloaded and kept permanently):

  H1 (Taker Buy Ratio): the share of aggressive/taker buying
     (taker_buy_quote_volume / quote_volume) rises in the days just before
     a coin touches 5x off its post-crash low - "quiet accumulation" while
     price itself hasn't moved yet.

  H2 (Volatility Squeeze): Bollinger Band width (20-day) compresses in the
     days just before the explosion - "coiled spring" pattern.

METHOD - this is the part that keeps us honest:
  EVENT group   = every symbol that actually touched 5x (ever_touched_5x=1),
                  measured in the real days leading up to touched_5x_date.
  CONTROL group = symbols that never touched 5x, measured around a RANDOM
                  date in their own history (same calendar era, same
                  measurement, but no explosion followed).

  The 30 days before the event/pseudo-event are split into THREE phases,
  not two - a flat "before vs after" comparison accidentally blends the
  explosion's own early days into the "before" bucket:
      baseline      = day -30 .. -15  (16 days) - normal reference behavior
      accumulation  = day -14 .. -8   (7 days)  - the "quiet before" phase
      ignition      = day -7  .. -1   (7 days)  - the move may already be starting

  For each symbol+date we compute four scores (per metric, comparing
  accumulation-vs-baseline and ignition-vs-accumulation separately), then
  compare EVENT vs CONTROL distributions with a two-sample t-test for each.
  The interesting result is COMPRESSION during accumulation followed by
  EXPANSION during ignition - a real two-phase pattern, not just "volatility
  rises as price rises" (trivially true, proves nothing predictive).

Requirements: pip install scipy
Usage: python backtest_precursor_signals.py
Reads: recovery_radar.db (must already exist - built by recovery_radar_phase1.py)
"""

from __future__ import annotations

import random
import sqlite3
import sys
from dataclasses import dataclass
from typing import Optional

try:
    from scipy import stats
except ImportError:
    print("Missing dependency. Run: pip install scipy", file=sys.stderr)
    raise

DB_PATH = "recovery_radar.db"

PRE_WINDOW_DAYS = 7          # kept for backward reference only
IGNITION_DAYS = 7            # day -7 to -1: the move may already be starting here
ACCUMULATION_DAYS = 7        # day -14 to -8: the "quiet before" window we actually care about for H2
BASELINE_DAYS = 16           # day -30 to -15: normal/reference behavior
TOTAL_LOOKBACK_DAYS = IGNITION_DAYS + ACCUMULATION_DAYS + BASELINE_DAYS  # 30
BB_PERIOD = 20                # Bollinger Band lookback period
MIN_HISTORY_NEEDED = TOTAL_LOOKBACK_DAYS + BB_PERIOD  # 50 days, so the earliest
                                                        # baseline day still has
                                                        # a full 20-day BB lookback

RANDOM_SEED = 42              # reproducible control-group sampling
MAX_CONTROL_SAMPLES = 300     # cap so the control group doesn't dwarf the event group unnecessarily


@dataclass
class DayRow:
    day: str
    close: float
    quote_volume: float
    taker_buy_quote_volume: float


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def get_event_symbols(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    """Symbols that actually touched 5x, with a real calendar date (excludes the
    'live (current price)' pseudo-date, which isn't a specific day we can look back from)."""
    cur = conn.execute(
        """SELECT symbol, touched_5x_date FROM recovery_snapshot
           WHERE ever_touched_5x = 1 AND touched_5x_date IS NOT NULL
             AND touched_5x_date != 'live (current price)'"""
    )
    return cur.fetchall()


def get_non_event_symbols(conn: sqlite3.Connection) -> list[str]:
    cur = conn.execute(
        """SELECT symbol FROM recovery_snapshot
           WHERE ever_touched_5x = 0 OR ever_touched_5x IS NULL"""
    )
    return [r[0] for r in cur.fetchall()]


def get_available_dates(conn: sqlite3.Connection, symbol: str) -> list[str]:
    cur = conn.execute(
        "SELECT open_time_utc FROM candles_daily WHERE symbol=? ORDER BY open_time_utc ASC",
        (symbol,),
    )
    return [r[0] for r in cur.fetchall()]


def get_candles_before(conn: sqlite3.Connection, symbol: str, before_day: str, limit: int) -> list[DayRow]:
    cur = conn.execute(
        """SELECT open_time_utc, close, quote_volume, taker_buy_quote_volume
           FROM candles_daily
           WHERE symbol=? AND open_time_utc < ?
           ORDER BY open_time_utc DESC LIMIT ?""",
        (symbol, before_day, limit),
    )
    rows = cur.fetchall()
    rows.reverse()  # chronological order
    return [DayRow(*r) for r in rows]


def taker_buy_ratio(row: DayRow) -> Optional[float]:
    if row.quote_volume and row.quote_volume > 0:
        return row.taker_buy_quote_volume / row.quote_volume
    return None


def bollinger_width_series(closes: list[float], period: int) -> list[Optional[float]]:
    out: list[Optional[float]] = []
    for i in range(len(closes)):
        if i + 1 < period:
            out.append(None)
            continue
        window = closes[i + 1 - period : i + 1]
        mean = sum(window) / period
        if mean == 0:
            out.append(None)
            continue
        var = sum((c - mean) ** 2 for c in window) / period
        std = var ** 0.5
        out.append((4 * std) / mean)  # (upper - lower) / mean, upper/lower = mean +/- 2*std
    return out


def compute_signal_scores(candles: list[DayRow]) -> Optional[dict]:
    """
    Splits the 30 days before the event into three phases:
        baseline      = day -30 .. -15  (16 days) - normal reference behavior
        accumulation  = day -14 .. -8   (7 days)  - the "quiet before" phase we care about
        ignition      = day -7  .. -1   (7 days)  - the move may already be starting here

    Returns a dict of scores, or None if there isn't enough history:
        taker_accum_score = mean(taker, accumulation) - mean(taker, baseline)
        taker_ignite_score = mean(taker, ignition) - mean(taker, accumulation)
        bb_accum_score = mean(bb, baseline) - mean(bb, accumulation)   [positive = compression]
        bb_ignite_score = mean(bb, ignition) - mean(bb, accumulation)  [positive = expansion/breakout]
    """
    if len(candles) < MIN_HISTORY_NEEDED:
        return None

    window = candles[-TOTAL_LOOKBACK_DAYS:]
    closes_full = [c.close for c in candles]
    bb_full = bollinger_width_series(closes_full, BB_PERIOD)
    bb_window = bb_full[-TOTAL_LOOKBACK_DAYS:]
    taker_window = [taker_buy_ratio(c) for c in window]

    baseline_taker = [v for v in taker_window[:BASELINE_DAYS] if v is not None]
    accum_taker = [v for v in taker_window[BASELINE_DAYS:BASELINE_DAYS + ACCUMULATION_DAYS] if v is not None]
    ignite_taker = [v for v in taker_window[BASELINE_DAYS + ACCUMULATION_DAYS:] if v is not None]

    baseline_bb = [v for v in bb_window[:BASELINE_DAYS] if v is not None]
    accum_bb = [v for v in bb_window[BASELINE_DAYS:BASELINE_DAYS + ACCUMULATION_DAYS] if v is not None]
    ignite_bb = [v for v in bb_window[BASELINE_DAYS + ACCUMULATION_DAYS:] if v is not None]

    groups = [baseline_taker, accum_taker, ignite_taker, baseline_bb, accum_bb, ignite_bb]
    if any(len(g) == 0 for g in groups):
        return None

    def avg(xs: list[float]) -> float:
        return sum(xs) / len(xs)

    return {
        "taker_accum_score": avg(accum_taker) - avg(baseline_taker),
        "taker_ignite_score": avg(ignite_taker) - avg(accum_taker),
        "bb_accum_score": avg(baseline_bb) - avg(accum_bb),   # positive = compression vs baseline
        "bb_ignite_score": avg(ignite_bb) - avg(accum_bb),    # positive = expansion vs accumulation
    }


def run_backtest() -> None:
    random.seed(RANDOM_SEED)
    conn = get_conn()

    print("[1/3] Loading event group (symbols that actually touched 5x)...")
    event_symbols = get_event_symbols(conn)
    print(f"      {len(event_symbols)} candidate event symbols")

    event_scores: list[dict] = []
    event_used = 0
    excluded_events: list[tuple[str, str, int]] = []
    for symbol, touched_date in event_symbols:
        candles = get_candles_before(conn, symbol, touched_date, MIN_HISTORY_NEEDED)
        scores = compute_signal_scores(candles)
        if scores is None:
            excluded_events.append((symbol, touched_date, len(candles)))
            continue
        event_scores.append(scores)
        event_used += 1
    print(f"      {event_used} event symbols had enough history ({MIN_HISTORY_NEEDED}+ days) to score")
    if excluded_events:
        print(f"      {len(excluded_events)} event symbols EXCLUDED for insufficient prior history:")
        for sym, date, n in sorted(excluded_events, key=lambda x: x[1]):
            print(f"        {sym:<14} touched_5x_date={date}  only {n}/{MIN_HISTORY_NEEDED} prior days available")

    print("[2/3] Building control group (random dates, symbols that never touched 5x)...")
    non_event_symbols = get_non_event_symbols(conn)
    random.shuffle(non_event_symbols)

    control_scores: list[dict] = []
    control_used = 0
    for symbol in non_event_symbols:
        if control_used >= MAX_CONTROL_SAMPLES:
            break
        dates = get_available_dates(conn, symbol)
        if len(dates) < MIN_HISTORY_NEEDED + 1:
            continue
        # pick a random pseudo-event date that still has enough history before it
        pseudo_idx = random.randint(MIN_HISTORY_NEEDED, len(dates) - 1)
        pseudo_date = dates[pseudo_idx]
        candles = get_candles_before(conn, symbol, pseudo_date, MIN_HISTORY_NEEDED)
        scores = compute_signal_scores(candles)
        if scores is None:
            continue
        control_scores.append(scores)
        control_used += 1
    print(f"      {control_used} control samples collected")

    conn.close()

    if event_used < 5 or control_used < 5:
        print(
            "\n[STOP] Not enough usable history yet to run a meaningful test "
            f"(need several weeks of accumulated daily candles per symbol; "
            f"currently only {event_used} event / {control_used} control samples qualify). "
            "Re-run this backtest again after the database has accumulated more days."
        )
        return

    print("\n[3/3] Statistical comparison (event group vs control group)\n")

    def report(name: str, key: str) -> None:
        event_vals = [s[key] for s in event_scores]
        control_vals = [s[key] for s in control_scores]
        t_stat, p_value = stats.ttest_ind(event_vals, control_vals, equal_var=False)
        event_mean = sum(event_vals) / len(event_vals)
        control_mean = sum(control_vals) / len(control_vals)
        verdict = (
            "LIKELY REAL SIGNAL (p < 0.05)" if p_value < 0.05
            else "NOT STATISTICALLY DISTINGUISHABLE FROM RANDOM (p >= 0.05)"
        )
        print(f"--- {name} ---")
        print(f"Event group   mean score: {event_mean:+.5f}  (n={len(event_vals)})")
        print(f"Control group mean score: {control_mean:+.5f}  (n={len(control_vals)})")
        print(f"t-statistic: {t_stat:.3f}   p-value: {p_value:.4f}")
        print(f"Verdict: {verdict}\n")

    report("H1a: Taker Buy Ratio rises during ACCUMULATION (day -14..-8) vs baseline", "taker_accum_score")
    report("H1b: Taker Buy Ratio rises further during IGNITION (day -7..-1) vs accumulation", "taker_ignite_score")
    report("H2a: Volatility COMPRESSES during ACCUMULATION (day -14..-8) vs baseline", "bb_accum_score")
    report("H2b: Volatility EXPANDS during IGNITION (day -7..-1) vs accumulation (the breakout itself)", "bb_ignite_score")

    print(
        "Reminder: statistical significance here means the pattern is unlikely to be pure\n"
        "chance GIVEN THIS DATA - it does not by itself mean the pattern is strong enough,\n"
        "reliable enough, or early enough to trade on profitably. Treat a positive result as\n"
        "'worth building a live monitor for', not 'confirmed trading edge'.\n"
        "The interesting case is H2a true + H2b true together: quiet compression BEFORE the\n"
        "move, followed by real expansion once it starts - a genuine two-phase pattern,\n"
        "not just 'volatility went up right as the price went up' (which is trivially true)."
    )


if __name__ == "__main__":
    run_backtest()
