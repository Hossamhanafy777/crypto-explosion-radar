"""
Ignition Checker - Phase 2 (runs every 15 minutes)
====================================================
Lightweight by design - only checks the SHORT watchlist (coins already
flagged 'in_compression' by update_watchlist.py, not all 490 symbols), so it
stays cheap enough to run every 15 minutes on GitHub Actions' free tier.

QUALIFYING GATE for a breakout (all must pass):
    1. Live price > stored range_ceiling by at least MIN_BREAKOUT_MARGIN_PCT
       (not just any tiny amount above - a razor-thin break is noise)
    2. Live 24h quote volume >= VOLUME_CONFIRM_MULTIPLIER x the 20-day average
    3. At least MIN_FACTORS_REQUIRED of 6 price/volume factors confirm
       (see below - based on real price-action/volume-analysis research,
       not off-the-shelf oscillators like RSI)
    4. Holds for CONFIRMATION_MINUTES before the final alert fires

THE 6 CONFIRMATION FACTORS (computed from stored daily candles, no extra
API calls beyond what's already fetched):
    PRICE-BASED:
    1. Rising Floor      - in the last 5 days, did the daily Low rise/hold
                            at least 3 times? (buyers defending higher ground)
    2. Close Strength     - over the last 3 days, did price close in the top
                            40%+ of its daily range on average? (real control,
                            not a fading wick)
    3. NR7 (Narrow Range 7) - was the most recent day's range the narrowest
                            of the last 7 days? (maximum coil right before
                            the break - a classic price-action trader concept)
    4. Ceiling Tests      - did price test within 2% of the ceiling at least
                            twice in the last 30 days without breaking?
                            (a level earns credibility by being defended)
    VOLUME-BASED:
    5. Volume Dry-Up (VDU) - was the average volume of the 10 days right
                            before the breakout LOWER than the 20 days before
                            that? (widely cited as one of the most reliable
                            pre-breakout signals - quiet accumulation, not
                            an already-noisy base)
    6. OBV Confirmation   - does On-Balance Volume (a running total: add
                            volume on up-days, subtract on down-days) rise
                            together with price over the last 15 days? If
                            price rises but OBV doesn't, that's a classic
                            divergence warning of unsupported buying.

Does NOT modify recovery_radar.db (avoids committing a 19MB+ binary file
every 15 minutes, which would bloat the repo badly). Alert state is tracked
in a small separate alerted_state.json file instead - cheap to commit.

Requirements: pip install requests
Env vars required (set as GitHub Secrets): TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
Usage: python check_ignition.py
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE_URL = "https://data-api.binance.vision"
DB_PATH = "recovery_radar.db"
STATE_PATH = Path("alerted_state.json")

VOLUME_CONFIRM_MULTIPLIER = 3.0   # raised from 2x - stronger confirmation required
MIN_BREAKOUT_MARGIN_PCT = 5.0     # must clear the ceiling by at least this much, not just any amount
MIN_FACTORS_REQUIRED = 4          # out of the 6 price/volume factors below
CONFIRMATION_MINUTES = 30  # breakout must persist for at least this long before alerting
WEAKNESS_DRAWDOWN_PCT = 15.0   # send a weakness update if price pulls back this much from its post-alert peak
TRACKING_WINDOW_HOURS = 48    # weakness-tracking window (separate from the longer milestone window below)
MILESTONE_MULTIPLE = 10.0     # send a milestone alert if price reaches this multiple of the alert price
MAX_TRACKING_DAYS = 30        # stop tracking a confirmed alert entirely after this long (bounds milestone checks too)
REQUEST_TIMEOUT = 15


def load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except Exception:
            return {}
    return {}


def save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2))


def get_watchlist() -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute(
        """SELECT symbol, range_ceiling, alerted FROM watchlist
           WHERE in_compression = 1"""
    )
    rows = cur.fetchall()
    conn.close()
    return [{"symbol": r[0], "ceiling": r[1], "alerted": r[2]} for r in rows]


def get_recent_avg_quote_volume(symbol: str, days: int = 20) -> float | None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute(
        """SELECT quote_volume FROM candles_daily
           WHERE symbol=? ORDER BY open_time_utc DESC LIMIT ?""",
        (symbol, days),
    )
    rows = [r[0] for r in cur.fetchall()]
    conn.close()
    if not rows:
        return None
    return sum(rows) / len(rows)


def get_recent_candles(symbol: str, days: int) -> list[tuple]:
    """Chronological (day, open, high, low, close, quote_volume) for the last
    `days` CLOSED daily candles - does not include today's still-open candle."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute(
        """SELECT open_time_utc, open, high, low, close, quote_volume
           FROM candles_daily WHERE symbol=? ORDER BY open_time_utc DESC LIMIT ?""",
        (symbol, days),
    )
    rows = cur.fetchall()
    conn.close()
    rows.reverse()
    return rows


def check_rising_floor(candles: list[tuple]) -> bool:
    """Last 5 days: did the daily Low rise or hold at least 3 times?"""
    if len(candles) < 5:
        return False
    last5 = candles[-5:]
    rising = sum(1 for i in range(1, len(last5)) if last5[i][3] >= last5[i - 1][3])
    return rising >= 3


def check_close_strength(candles: list[tuple]) -> bool:
    """Last 3 days: average close position within the day's range >= 60%."""
    if len(candles) < 3:
        return False
    last3 = candles[-3:]
    strengths = []
    for _, _, h, l, c, _ in last3:
        strengths.append((c - l) / (h - l) if h > l else 0.5)
    return (sum(strengths) / len(strengths)) >= 0.60


def check_nr7(candles: list[tuple]) -> bool:
    """Was the most recent closed day's range the narrowest of the last 7 days?"""
    if len(candles) < 7:
        return False
    last7 = candles[-7:]
    ranges = [h - l for _, _, h, l, _, _ in last7]
    return ranges[-1] == min(ranges)


def check_ceiling_tests(candles: list[tuple], ceiling: float) -> bool:
    """Last 30 days: did price test within 2% of the ceiling at least twice
    without closing above it? (a level earns credibility by being defended)"""
    window = candles[-30:] if len(candles) >= 30 else candles
    if not window:
        return False
    tests = sum(1 for _, _, h, _, c, _ in window if h >= 0.98 * ceiling and c < ceiling)
    return tests >= 2


def check_volume_dry_up(candles: list[tuple]) -> bool:
    """Was the average volume of the 10 days right before the breakout LOWER
    than the 20 days before that? (quiet accumulation, not an already-noisy base)"""
    if len(candles) < 30:
        return False
    recent10 = candles[-10:]
    baseline20 = candles[-30:-10]
    recent_avg = sum(c[5] for c in recent10) / len(recent10)
    baseline_avg = sum(c[5] for c in baseline20) / len(baseline20)
    if baseline_avg <= 0:
        return False
    return recent_avg <= 0.8 * baseline_avg


def check_obv_confirmation(candles: list[tuple]) -> bool:
    """Does On-Balance Volume rise together with price over the last 15 days?
    (price up + OBV flat/down = divergence = unsupported buying)"""
    if len(candles) < 15:
        return False
    last15 = candles[-15:]
    obv = [0.0]
    for i in range(1, len(last15)):
        prev_close, close, vol = last15[i - 1][4], last15[i][4], last15[i][5]
        if close > prev_close:
            obv.append(obv[-1] + vol)
        elif close < prev_close:
            obv.append(obv[-1] - vol)
        else:
            obv.append(obv[-1])
    price_rising = last15[-1][4] > last15[0][4]
    obv_rising = obv[-1] > obv[0]
    return price_rising and obv_rising


def evaluate_confirmation_factors(symbol: str, ceiling: float) -> tuple[int, dict]:
    """Returns (score out of 6, {factor_name: bool}) using stored daily candles."""
    candles = get_recent_candles(symbol, 35)
    results = {
        "rising_floor": check_rising_floor(candles),
        "close_strength": check_close_strength(candles),
        "nr7": check_nr7(candles),
        "ceiling_tests": check_ceiling_tests(candles, ceiling),
        "volume_dry_up": check_volume_dry_up(candles),
        "obv_confirmation": check_obv_confirmation(candles),
    }
    score = sum(1 for v in results.values() if v)
    return score, results


def fetch_ticker_24h(symbol: str) -> dict | None:
    try:
        resp = requests.get(
            f"{BASE_URL}/api/v3/ticker/24hr",
            params={"symbol": symbol},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[warn] ticker fetch failed for {symbol}: {e}", file=sys.stderr)
        return None


def send_telegram_alert(message: str) -> bool:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("[error] TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set", file=sys.stderr)
        return False
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": message},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return True
    except Exception as e:
        print(f"[error] Telegram send failed: {e}", file=sys.stderr)
        return False


def track_confirmed_alerts(state: dict) -> int:
    """For every already-confirmed alert not yet closed, check for three
    outcomes and send a follow-up Telegram update:
        - price fell back below ceiling -> FALSE BREAKOUT, close tracking
          permanently (this also permanently blocks any future milestone alert)
        - price still above ceiling but pulled back >= WEAKNESS_DRAWDOWN_PCT
          from its post-alert peak -> WEAKNESS update (sent once, only within
          the shorter TRACKING_WINDOW_HOURS)
        - price reached MILESTONE_MULTIPLE x the alert price, and no false
          breakout was ever declared for it -> MILESTONE update (sent once,
          tracked independently up to the longer MAX_TRACKING_DAYS)
    Returns the number of follow-up messages sent.
    """
    updates_sent = 0
    now = datetime.now(timezone.utc)

    for state_key, entry in list(state.items()):
        if not isinstance(entry, dict) or entry.get("status") != "confirmed" or entry.get("closed"):
            continue

        symbol = entry.get("symbol")
        ceiling = entry.get("ceiling")
        alert_price = entry.get("alert_price")
        if not symbol or ceiling is None or alert_price is None:
            continue

        alert_time = datetime.fromisoformat(entry["alert_time"])
        hours_since_alert = (now - alert_time).total_seconds() / 3600
        days_since_alert = hours_since_alert / 24

        if days_since_alert >= MAX_TRACKING_DAYS:
            entry["closed"] = True
            print(f"[tracking ended] {symbol} - {MAX_TRACKING_DAYS}-day max tracking window elapsed")
            continue

        ticker = fetch_ticker_24h(symbol)
        if ticker is None:
            continue
        try:
            live_price = float(ticker["lastPrice"])
        except (KeyError, ValueError):
            continue

        if live_price < ceiling:
            message = (
                f"Update: {symbol} - FALSE BREAKOUT confirmed\n\n"
                f"Price fell back below the {ceiling:.8f} ceiling "
                f"(now {live_price:.8f}), reversing the earlier ignition alert.\n"
                f"Peak reached after the alert: {entry.get('peak_price', entry.get('alert_price')):.8f}"
            )
            if send_telegram_alert(message):
                entry["closed"] = True  # permanent - also blocks any future milestone alert
                updates_sent += 1
                print(f"[false breakout update sent] {symbol}")
            continue

        peak_price = max(entry.get("peak_price", live_price), live_price)
        entry["peak_price"] = peak_price

        # weakness check - only within the shorter tracking window
        if hours_since_alert < TRACKING_WINDOW_HOURS and not entry.get("weakness_alert_sent"):
            drawdown_pct = (peak_price - live_price) / peak_price * 100 if peak_price > 0 else 0
            if drawdown_pct >= WEAKNESS_DRAWDOWN_PCT:
                message = (
                    f"Update: {symbol} - weakness signs after breakout\n\n"
                    f"Still holding above the {ceiling:.8f} ceiling, but has pulled back "
                    f"{drawdown_pct:.1f}% from its post-alert peak of {peak_price:.8f}.\n"
                    f"Current price: {live_price:.8f}\n\n"
                    f"Momentum may be fading - not necessarily a failed breakout, "
                    f"but worth re-checking before adding to a position."
                )
                if send_telegram_alert(message):
                    entry["weakness_alert_sent"] = True
                    updates_sent += 1
                    print(f"[weakness update sent] {symbol}")

        # milestone check - independent of the weakness window, runs up to
        # MAX_TRACKING_DAYS, but only as long as no false breakout was ever declared
        if not entry.get("milestone_10x_sent"):
            multiple = live_price / alert_price if alert_price > 0 else 0
            if multiple >= MILESTONE_MULTIPLE:
                message = (
                    f"MILESTONE: {symbol} reached {MILESTONE_MULTIPLE:.0f}x since its ignition alert!\n\n"
                    f"Alert price: {alert_price:.8f}\n"
                    f"Current price: {live_price:.8f}\n"
                    f"Multiple: {multiple:.1f}x\n\n"
                    f"No false-breakout was ever declared for this alert."
                )
                if send_telegram_alert(message):
                    entry["milestone_10x_sent"] = True
                    updates_sent += 1
                    print(f"[milestone {MILESTONE_MULTIPLE:.0f}x sent] {symbol}")

    return updates_sent


def run() -> None:
    watchlist = get_watchlist()
    print(f"Checking {len(watchlist)} watchlist symbols for live breakout...")

    state = load_state()
    alerts_sent = 0

    for item in watchlist:
        symbol = item["symbol"]
        ceiling = item["ceiling"]
        if ceiling is None:
            continue

        state_key = f"{symbol}:{ceiling:.10f}"
        entry = state.get(state_key)

        # already confirmed and alerted for this exact ceiling - nothing more to do
        if isinstance(entry, dict) and entry.get("status") == "confirmed":
            continue

        ticker = fetch_ticker_24h(symbol)
        if ticker is None:
            continue
        try:
            live_price = float(ticker["lastPrice"])
            live_quote_volume_24h = float(ticker["quoteVolume"])
        except (KeyError, ValueError):
            continue

        if live_price <= ceiling:
            # breakout failed or reversed - clear any pending state so a future
            # real breakout can start a fresh confirmation cycle
            if state_key in state:
                del state[state_key]
            continue

        pct_above = (live_price - ceiling) / ceiling * 100
        if pct_above < MIN_BREAKOUT_MARGIN_PCT:
            continue  # too marginal above the ceiling - likely noise, not a real break

        avg_volume = get_recent_avg_quote_volume(symbol)
        if avg_volume is None or avg_volume <= 0:
            continue

        if live_quote_volume_24h < VOLUME_CONFIRM_MULTIPLIER * avg_volume:
            continue  # breakout without strong volume confirmation - likely a fakeout, skip

        factor_score, factor_results = evaluate_confirmation_factors(symbol, ceiling)
        if factor_score < MIN_FACTORS_REQUIRED:
            continue  # not enough price/volume factors confirm - skip, don't even go pending

        now = datetime.now(timezone.utc)

        if not isinstance(entry, dict) or entry.get("status") != "pending":
            # first time we've seen this breakout - record when, wait for
            # CONFIRMATION_MINUTES of it still holding before alerting
            state[state_key] = {
                "status": "pending", "first_seen": now.isoformat(), "first_price": live_price,
                "factor_score": factor_score,
            }
            print(f"[pending] {symbol} broke ceiling ({factor_score}/6 factors), starting {CONFIRMATION_MINUTES}-min confirmation window")

            passed_factors = ", ".join(k for k, v in factor_results.items() if v)
            pending_message = (
                f"Pending: {symbol} just broke its ceiling\n\n"
                f"Ceiling: {ceiling:.8f}\n"
                f"Live price: {live_price:.8f} (+{pct_above:.1f}% above ceiling)\n"
                f"24h volume: {live_quote_volume_24h:,.0f} USDT ({live_quote_volume_24h/avg_volume:.1f}x avg)\n"
                f"Confirmation factors: {factor_score}/6 ({passed_factors})\n\n"
                f"Watching for {CONFIRMATION_MINUTES} minutes to confirm it holds "
                f"before the full Ignition Alert. This is NOT confirmed yet."
            )
            send_telegram_alert(pending_message)
            continue

        first_seen = datetime.fromisoformat(entry["first_seen"])
        elapsed_minutes = (now - first_seen).total_seconds() / 60

        if elapsed_minutes < CONFIRMATION_MINUTES:
            print(f"[pending] {symbol} still breaking out - {elapsed_minutes:.0f}/{CONFIRMATION_MINUTES} min confirmed")
            continue

        # held above ceiling with volume for the full confirmation window - alert
        pct_above = (live_price - ceiling) / ceiling * 100
        factor_score = entry.get("factor_score", "?")
        message = (
            f"Ignition Alert: {symbol}\n"
            f"Broke above its 30-day range ceiling with volume + {factor_score}/6 price-action "
            f"factor confirmation (held for {CONFIRMATION_MINUTES}+ minutes).\n\n"
            f"Ceiling: {ceiling:.8f}\n"
            f"Live price: {live_price:.8f} (+{pct_above:.1f}% above ceiling)\n"
            f"24h volume: {live_quote_volume_24h:,.0f} USDT "
            f"({live_quote_volume_24h/avg_volume:.1f}x the 20-day average)\n\n"
            f"Reminder: this narrows the field, it is not a buy signal by itself. "
            f"Verify manually before acting."
        )
        if send_telegram_alert(message):
            state[state_key] = {
                "status": "confirmed",
                "symbol": symbol,
                "ceiling": ceiling,
                "alert_price": live_price,
                "alert_time": now.isoformat(),
                "peak_price": live_price,
                "weakness_alert_sent": False,
                "closed": False,
            }
            alerts_sent += 1
            print(f"[alert sent] {symbol}")

    tracking_updates = track_confirmed_alerts(state)

    save_state(state)
    print(f"Done. {alerts_sent} new alert(s), {tracking_updates} follow-up update(s) sent this run.")


if __name__ == "__main__":
    run()
