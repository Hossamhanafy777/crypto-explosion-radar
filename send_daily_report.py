"""
Daily Watchlist Report
========================
Sends one Telegram message per day listing every symbol currently flagged
in_compression=1 in the watchlist table - even if nothing changed since
yesterday. This is separate from check_ignition.py's alerts (which only
fire on an actual breakout); this is just a standing daily status report.

Runs as part of the daily job, right after update_watchlist.py.

Requirements: pip install requests
Env vars required (GitHub Secrets): TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
Usage: python send_daily_report.py
"""

from __future__ import annotations

import os
import sqlite3
import sys
from datetime import datetime, timezone

import requests

DB_PATH = "recovery_radar.db"


def get_watchlist() -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute(
        """SELECT symbol, quiet_pct_90d, current_30d_range_pct, range_ceiling, range_floor
           FROM watchlist
           WHERE in_compression = 1
           ORDER BY quiet_pct_90d DESC"""
    )
    rows = cur.fetchall()
    conn.close()
    return [
        {
            "symbol": r[0], "quiet_pct": r[1], "range_pct": r[2],
            "ceiling": r[3], "floor": r[4],
        }
        for r in rows
    ]


def build_message(watchlist: list[dict]) -> str:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if not watchlist:
        return f"*Daily Watchlist Report - {today}*\n\nNo symbols currently in compression."

    lines = [f"*Daily Watchlist Report - {today}*", f"{len(watchlist)} symbol(s) currently in compression:\n"]
    for w in watchlist:
        lines.append(
            f"`{w['symbol']}`  quiet={w['quiet_pct']:.0f}%  "
            f"range={w['range_pct']:.0f}%  ceiling={w['ceiling']:.8f}"
        )
    lines.append("\nReminder: being on this list is not a signal by itself - it only means the coin is currently quiet relative to the market.")
    return "\n".join(lines)


def send_telegram(message: str) -> bool:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("[error] TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set", file=sys.stderr)
        return False
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"},
            timeout=15,
        )
        resp.raise_for_status()
        return True
    except Exception as e:
        print(f"[error] Telegram send failed: {e}", file=sys.stderr)
        return False


def run() -> None:
    watchlist = get_watchlist()
    message = build_message(watchlist)
    if send_telegram(message):
        print(f"Daily report sent - {len(watchlist)} symbol(s) on the watchlist.")
    else:
        print("Failed to send daily report.", file=sys.stderr)


if __name__ == "__main__":
    run()
