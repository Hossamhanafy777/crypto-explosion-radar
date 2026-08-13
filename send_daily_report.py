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
import time
from datetime import datetime, timezone

import requests

DB_PATH = "recovery_radar.db"


def get_watchlist() -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute(
        """SELECT symbol, current_percentile, persistence_pct, current_30d_range_pct,
                  range_ceiling, range_floor, proximity_pct, explosion_score
           FROM watchlist
           WHERE in_compression = 1
           ORDER BY explosion_score DESC"""
    )
    rows = cur.fetchall()
    conn.close()
    return [
        {
            "symbol": r[0], "percentile": r[1], "persistence": r[2],
            "range_pct": r[3], "ceiling": r[4], "floor": r[5],
            "proximity": r[6], "score": r[7],
        }
        for r in rows
    ]


MAX_MESSAGE_CHARS = 3500  # stay comfortably under Telegram's 4096 limit


def build_messages(watchlist: list[dict]) -> list[str]:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if not watchlist:
        return [f"Daily Watchlist Report - {today}\n\nNo symbols currently in compression."]

    header = f"Daily Watchlist Report - {today}\n{len(watchlist)} symbol(s) currently in compression, ranked by explosion_score (0-100, higher = closer to a breakout):\n"
    footer = "\nReminder: this ranks readiness, not certainty - it is not a signal by itself."

    entry_lines = []
    for i, w in enumerate(watchlist, 1):
        score = w["score"] if w["score"] is not None else 0.0
        pct = w["percentile"] if w["percentile"] is not None else 0.0
        persist = w["persistence"] if w["persistence"] is not None else 0.0
        prox = w["proximity"] if w["proximity"] is not None else 0.0
        entry_lines.append(
            f"{i}. {w['symbol']}  score={score:.0f}  "
            f"tight={pct:.0f}pct  persist={persist:.0f}%  near_ceiling={prox:.0f}%"
        )

    # pack lines into chunks that stay under the char limit
    messages = []
    current_lines = [header]
    current_len = len(header)
    for line in entry_lines:
        if current_len + len(line) + 1 > MAX_MESSAGE_CHARS:
            messages.append("\n".join(current_lines))
            current_lines = []
            current_len = 0
        current_lines.append(line)
        current_len += len(line) + 1
    current_lines.append(footer)
    messages.append("\n".join(current_lines))

    # number the parts if there's more than one message
    if len(messages) > 1:
        messages = [f"[{i}/{len(messages)}]\n{m}" for i, m in enumerate(messages, 1)]

    return messages


def send_telegram(message: str) -> bool:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("[error] TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set", file=sys.stderr)
        return False
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
    messages = build_messages(watchlist)
    print(f"Sending daily report in {len(messages)} message(s) ({len(watchlist)} symbols)...")

    failures = 0
    for i, msg in enumerate(messages, 1):
        if send_telegram(msg):
            print(f"  part {i}/{len(messages)} sent ({len(msg)} chars)")
        else:
            print(f"  part {i}/{len(messages)} FAILED", file=sys.stderr)
            failures += 1
        time.sleep(1)  # small gap between messages to avoid Telegram rate limits

    if failures:
        print(f"Failed to send {failures}/{len(messages)} message part(s).", file=sys.stderr)
        sys.exit(1)
    print(f"Daily report sent successfully - {len(watchlist)} symbol(s) on the watchlist.")


if __name__ == "__main__":
    run()
