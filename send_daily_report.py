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
        """SELECT symbol, current_percentile, persistence_pct, current_30d_range_pct, range_ceiling, range_floor
           FROM watchlist
           WHERE in_compression = 1
           ORDER BY current_percentile ASC"""
    )
    rows = cur.fetchall()
    conn.close()
    return [
        {
            "symbol": r[0], "percentile": r[1], "persistence": r[2],
            "range_pct": r[3], "ceiling": r[4], "floor": r[5],
        }
        for r in rows
    ]


def build_message(watchlist: list[dict]) -> str:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if not watchlist:
        return f"Daily Watchlist Report - {today}\n\nNo symbols currently in compression."

    lines = [f"Daily Watchlist Report - {today}", f"{len(watchlist)} symbol(s) currently in compression:\n"]
    for w in watchlist:
        pct = w["percentile"] if w["percentile"] is not None else 0.0
        persist = w["persistence"] if w["persistence"] is not None else 0.0
        rng = w["range_pct"] if w["range_pct"] is not None else 0.0
        ceiling = w["ceiling"] if w["ceiling"] is not None else 0.0
        lines.append(
            f"{w['symbol']}  tightest={pct:.1f}pct-of-own-history  "
            f"persistence={persist:.0f}%  range={rng:.0f}%  ceiling={ceiling:.8f}"
        )
    lines.append("\nReminder: being on this list is not a signal by itself - it only means the coin is currently unusually quiet compared to its OWN history.")
    return "\n".join(lines)


def send_telegram(message: str) -> bool:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("[error] TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set", file=sys.stderr)
        return False
    print(f"[debug] chat_id being used: '{chat_id}' (length={len(chat_id)})", file=sys.stderr)
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": message},
            timeout=15,
        )
        if resp.status_code != 200:
            print(f"[error] Telegram returned {resp.status_code}: {resp.text}", file=sys.stderr)
            return False
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
