"""
Ignition Checker - Phase 2 (runs every 15 minutes)
====================================================
Lightweight by design - only checks the SHORT watchlist (coins already
flagged 'in_compression' by update_watchlist.py, not all 490 symbols), so it
stays cheap enough to run every 15 minutes on GitHub Actions' free tier.

For each watchlist symbol:
    1. Fetch the LIVE current price.
    2. Fetch 24h ticker stats (for a live volume read).
    3. If live price > stored range_ceiling (the 30-day breakout level)
       AND live 24h quote volume > 2x the recent daily average volume
       (confirmation - a breakout without volume is often a fakeout)
       AND we haven't already alerted for this exact ceiling
    -> send a Telegram alert and record it.

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
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE_URL = "https://data-api.binance.vision"
DB_PATH = "recovery_radar.db"
STATE_PATH = Path("alerted_state.json")

VOLUME_CONFIRM_MULTIPLIER = 2.0
CONFIRMATION_MINUTES = 30  # breakout must persist for at least this long before alerting
WEAKNESS_DRAWDOWN_PCT = 15.0   # send a weakness update if price pulls back this much from its post-alert peak
TRACKING_WINDOW_HOURS = 48    # stop tracking a confirmed alert after this long, regardless of outcome
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
    import sqlite3

    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute(
        """SELECT symbol, range_ceiling, alerted FROM watchlist
           WHERE in_compression = 1"""
    )
    rows = cur.fetchall()
    conn.close()
    return [{"symbol": r[0], "ceiling": r[1], "alerted": r[2]} for r in rows]


def get_recent_avg_quote_volume(symbol: str, days: int = 20) -> float | None:
    import sqlite3

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
    """For every already-confirmed alert not yet closed, check for two
    outcomes and send a follow-up Telegram update:
        - price fell back below ceiling -> FALSE BREAKOUT, close tracking
        - price still above ceiling but pulled back >= WEAKNESS_DRAWDOWN_PCT
          from its post-alert peak -> WEAKNESS update (sent once)
    Tracking stops automatically after TRACKING_WINDOW_HOURS regardless.
    Returns the number of follow-up messages sent.
    """
    updates_sent = 0
    now = datetime.now(timezone.utc)

    for state_key, entry in list(state.items()):
        if not isinstance(entry, dict) or entry.get("status") != "confirmed" or entry.get("closed"):
            continue

        symbol = entry.get("symbol")
        ceiling = entry.get("ceiling")
        if not symbol or ceiling is None:
            continue

        alert_time = datetime.fromisoformat(entry["alert_time"])
        hours_since_alert = (now - alert_time).total_seconds() / 3600
        if hours_since_alert >= TRACKING_WINDOW_HOURS:
            entry["closed"] = True
            print(f"[tracking ended] {symbol} - {TRACKING_WINDOW_HOURS}h tracking window elapsed")
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
                entry["closed"] = True
                updates_sent += 1
                print(f"[false breakout update sent] {symbol}")
            continue

        peak_price = max(entry.get("peak_price", live_price), live_price)
        entry["peak_price"] = peak_price

        drawdown_pct = (peak_price - live_price) / peak_price * 100 if peak_price > 0 else 0

        if drawdown_pct >= WEAKNESS_DRAWDOWN_PCT and not entry.get("weakness_alert_sent"):
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

        avg_volume = get_recent_avg_quote_volume(symbol)
        if avg_volume is None or avg_volume <= 0:
            continue

        if live_quote_volume_24h < VOLUME_CONFIRM_MULTIPLIER * avg_volume:
            continue  # breakout without volume confirmation - likely a fakeout, skip

        now = datetime.now(timezone.utc)

        if not isinstance(entry, dict) or entry.get("status") != "pending":
            # first time we've seen this breakout - record when, wait for
            # CONFIRMATION_MINUTES of it still holding before alerting
            state[state_key] = {"status": "pending", "first_seen": now.isoformat(), "first_price": live_price}
            print(f"[pending] {symbol} broke ceiling, starting {CONFIRMATION_MINUTES}-min confirmation window")
            continue

        first_seen = datetime.fromisoformat(entry["first_seen"])
        elapsed_minutes = (now - first_seen).total_seconds() / 60

        if elapsed_minutes < CONFIRMATION_MINUTES:
            print(f"[pending] {symbol} still breaking out - {elapsed_minutes:.0f}/{CONFIRMATION_MINUTES} min confirmed")
            continue

        # held above ceiling with volume for the full confirmation window - alert
        pct_above = (live_price - ceiling) / ceiling * 100
        message = (
            f"Ignition Alert: {symbol}\n"
            f"Broke above its 30-day range ceiling with volume confirmation "
            f"(held for {CONFIRMATION_MINUTES}+ minutes).\n\n"
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
