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
from pathlib import Path

import requests

BASE_URL = "https://data-api.binance.vision"
DB_PATH = "recovery_radar.db"
STATE_PATH = Path("alerted_state.json")

VOLUME_CONFIRM_MULTIPLIER = 2.0
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

        # already alerted for this exact ceiling? skip (state key includes ceiling
        # so a NEW compression cycle with a new ceiling can alert again)
        state_key = f"{symbol}:{ceiling:.10f}"
        if state.get(state_key):
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
            continue  # no breakout yet

        avg_volume = get_recent_avg_quote_volume(symbol)
        if avg_volume is None or avg_volume <= 0:
            continue

        # 24h volume vs recent daily average - rough but cheap confirmation
        if live_quote_volume_24h < VOLUME_CONFIRM_MULTIPLIER * avg_volume:
            continue  # breakout without volume confirmation - likely a fakeout, skip

        pct_above = (live_price - ceiling) / ceiling * 100
        message = (
            f"Ignition Alert: {symbol}\n"
            f"Broke above its 30-day range ceiling with volume confirmation.\n\n"
            f"Ceiling: {ceiling:.8f}\n"
            f"Live price: {live_price:.8f} (+{pct_above:.1f}% above ceiling)\n"
            f"24h volume: {live_quote_volume_24h:,.0f} USDT "
            f"({live_quote_volume_24h/avg_volume:.1f}x the 20-day average)\n\n"
            f"Reminder: this narrows the field, it is not a buy signal by itself. "
            f"Verify manually before acting."
        )
        if send_telegram_alert(message):
            state[state_key] = True
            alerts_sent += 1
            print(f"[alert sent] {symbol}")

    save_state(state)
    print(f"Done. {alerts_sent} new alert(s) sent this run.")


if __name__ == "__main__":
    run()
