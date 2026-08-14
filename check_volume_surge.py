"""
Volume Surge Detector
======================
Independent from the Watchlist/Ignition pipeline entirely - this watches
ALL currently-listed coins (not just the ~101 in compression) for abnormal
VOLUME behavior, on the theory that quiet accumulation can show up in volume
before it shows up in price range (e.g. MOVEUSDT: volume up sharply, price
range still "normal" width, so it never appeared on the compression watchlist
at all).

GATE (per your decision): excludes any coin with ever_touched_5x=1 in
recovery_snapshot (already made its move since the Oct-11 crash low) and the
same stablecoin/pegged exclusion list used elsewhere.

FOUR-FACTOR COMPOSITE SCORE (0-100), weights as agreed:
    35%  Volume surge strength   - today's live quote volume vs its own 20-day average
    30%  Taker buy ratio         - aggressive buying vs total volume, today so far
    20%  Trade count density     - number of trades today vs recent daily average
                                    (many small trades = organic accumulation;
                                    one or two huge trades = a single whale, weaker signal)
    15%  Relative strength vs BTC - today's % change vs BTC's % change (decoupling)

TWO-STAGE DESIGN (cheap screen, then targeted deep check) - keeps this
affordable to run every 15 minutes across the whole universe:
    Stage 1: ONE bulk API call (/api/v3/ticker/24hr, no symbol param) returns
             live price + volume for all ~490 symbols at once. Compare each to
             its stored 20-day average (already in our DB, no extra calls) to
             build a shortlist of symbols with volume >= SURGE_MULTIPLIER x normal.
    Stage 2: For the shortlist only (capped), fetch today's still-open daily
             candle (klines limit=1) to get taker-buy-ratio and trade count so
             far today. BTC's % change is read for free from the Stage-1 bulk
             response - no extra call needed.

STABILITY / CONFIRMATION (per your decision): once a candidate first crosses
the surge threshold, it enters PENDING and its price is recorded at every
15-min check. It only gets the final confirmed alert once CONFIRMATION_MINUTES
have passed AND the current price is still at or above the RUNNING AVERAGE of
all prices recorded since detection (not strictly above the very first price -
this tolerates minor pullbacks while still requiring the overall trend to hold).
If price drops below that running average at any check, the candidate resets.

Requirements: pip install requests
Env vars required (GitHub Secrets): TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
Usage: python check_volume_surge.py
Reads: recovery_radar.db (for universe, stablecoin exclusion, ever_touched_5x,
       and stored 20-day volume/trade-count averages)
Writes: volume_surge_state.json (separate from ignition's alerted_state.json)
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE_URL = "https://data-api.binance.vision"
DB_PATH = "recovery_radar.db"
STATE_PATH = Path("volume_surge_state.json")

SURGE_MULTIPLIER = 3.0        # Stage 1 shortlist threshold: live volume vs 20-day avg
STAGE2_MAX_CANDIDATES = 30    # cap deep-check API calls per run
CONFIRMATION_MINUTES = 30
REQUEST_TIMEOUT = 20

WEIGHT_VOLUME = 0.35
WEIGHT_TAKER = 0.30
WEIGHT_TRADES = 0.20
WEIGHT_BTC = 0.15

STABLE_PEGGED_EXCLUDE = {
    "EUR", "EURI", "AEUR", "U", "KGST", "PAXG", "XAUT", "WBTC", "WBETH",
}


def is_excluded(base_asset: str) -> bool:
    if base_asset in STABLE_PEGGED_EXCLUDE:
        return True
    if "USD" in base_asset.upper():
        return True
    return False


def load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except Exception:
            return {}
    return {}


def save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2))


def get_eligible_symbols(conn: sqlite3.Connection) -> list[str]:
    """Universe minus stablecoins/pegged assets minus anything that already
    touched 5x since the crash low (per your gate)."""
    cur = conn.execute(
        """SELECT u.symbol, u.base_asset,
                  COALESCE(r.ever_touched_5x, 0) as touched
           FROM universe u
           LEFT JOIN recovery_snapshot r ON u.symbol = r.symbol"""
    )
    out = []
    for symbol, base_asset, touched in cur.fetchall():
        if is_excluded(base_asset):
            continue
        if touched:
            continue
        out.append(symbol)
    return out


def get_avg_quote_volume(conn: sqlite3.Connection, symbol: str, days: int = 20) -> float | None:
    cur = conn.execute(
        """SELECT quote_volume FROM candles_daily
           WHERE symbol=? ORDER BY open_time_utc DESC LIMIT ?""",
        (symbol, days),
    )
    rows = [r[0] for r in cur.fetchall()]
    if not rows:
        return None
    return sum(rows) / len(rows)


def get_avg_trade_count(conn: sqlite3.Connection, symbol: str, days: int = 20) -> float | None:
    cur = conn.execute(
        """SELECT trade_count FROM candles_daily
           WHERE symbol=? ORDER BY open_time_utc DESC LIMIT ?""",
        (symbol, days),
    )
    rows = [r[0] for r in cur.fetchall()]
    if not rows:
        return None
    return sum(rows) / len(rows)


def fetch_all_tickers() -> list[dict]:
    resp = requests.get(f"{BASE_URL}/api/v3/ticker/24hr", timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def fetch_today_kline(symbol: str) -> dict | None:
    """The current, still-open daily candle - gives accumulated taker-buy
    volume and trade count so far today."""
    try:
        resp = requests.get(
            f"{BASE_URL}/api/v3/klines",
            params={"symbol": symbol, "interval": "1d", "limit": 1},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        rows = resp.json()
        if not rows:
            return None
        r = rows[-1]
        return {
            "quote_volume": float(r[7]),
            "trade_count": int(r[8]),
            "taker_buy_quote_volume": float(r[10]),
        }
    except Exception as e:
        print(f"[warn] kline fetch failed for {symbol}: {e}", file=sys.stderr)
        return None


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


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def run() -> None:
    conn = sqlite3.connect(DB_PATH)
    state = load_state()

    print("[1/4] Loading eligible universe (excludes stablecoins + already-5x'd coins)...")
    eligible = set(get_eligible_symbols(conn))
    print(f"      {len(eligible)} eligible symbols")

    print("[2/4] Stage 1: bulk ticker fetch (1 API call for all symbols)...")
    all_tickers = fetch_all_tickers()
    ticker_by_symbol = {t["symbol"]: t for t in all_tickers}

    btc_ticker = ticker_by_symbol.get("BTCUSDT")
    btc_change_pct = float(btc_ticker["priceChangePercent"]) if btc_ticker else 0.0

    shortlist = []
    for symbol in eligible:
        ticker = ticker_by_symbol.get(symbol)
        if not ticker:
            continue
        try:
            live_quote_volume = float(ticker["quoteVolume"])
            live_price = float(ticker["lastPrice"])
            price_change_pct = float(ticker["priceChangePercent"])
        except (KeyError, ValueError):
            continue

        avg_volume = get_avg_quote_volume(conn, symbol)
        if not avg_volume or avg_volume <= 0:
            continue

        volume_ratio = live_quote_volume / avg_volume
        if volume_ratio >= SURGE_MULTIPLIER:
            shortlist.append((symbol, volume_ratio, live_price, price_change_pct))

    shortlist.sort(key=lambda x: -x[1])
    shortlist = shortlist[:STAGE2_MAX_CANDIDATES]
    print(f"      {len(shortlist)} candidates cleared the {SURGE_MULTIPLIER}x volume threshold (capped at {STAGE2_MAX_CANDIDATES})")

    print("[3/4] Stage 2: deep check (taker ratio, trade count) for shortlist...")
    now = datetime.now(timezone.utc)
    confirmed_count = 0

    for symbol, volume_ratio, live_price, price_change_pct in shortlist:
        today = fetch_today_kline(symbol)
        if today is None or today["quote_volume"] <= 0:
            continue

        taker_ratio = today["taker_buy_quote_volume"] / today["quote_volume"]
        avg_trades = get_avg_trade_count(conn, symbol) or 1
        trade_ratio = today["trade_count"] / avg_trades

        volume_score = clamp((volume_ratio / 10) * 100, 0, 100)
        taker_score = clamp(taker_ratio * 100, 0, 100)
        trade_score = clamp((trade_ratio / 5) * 100, 0, 100)
        relative_strength = price_change_pct - btc_change_pct
        btc_score = clamp((relative_strength + 20) / 40 * 100, 0, 100)

        final_score = (
            WEIGHT_VOLUME * volume_score
            + WEIGHT_TAKER * taker_score
            + WEIGHT_TRADES * trade_score
            + WEIGHT_BTC * btc_score
        )

        state_key = f"vs:{symbol}"
        entry = state.get(state_key)

        if not isinstance(entry, dict) or entry.get("status") != "pending":
            state[state_key] = {
                "status": "pending",
                "first_seen": now.isoformat(),
                "prices": [live_price],
                "score": round(final_score, 1),
            }
            message = (
                f"Volume Surge (pending): {symbol}\n\n"
                f"Volume: {volume_ratio:.1f}x its 20-day average\n"
                f"Taker buy ratio (today): {taker_ratio*100:.0f}%\n"
                f"Trade count vs average: {trade_ratio:.1f}x\n"
                f"vs BTC today: {relative_strength:+.1f}%\n"
                f"Score: {final_score:.0f}/100\n\n"
                f"Watching for {CONFIRMATION_MINUTES} minutes to confirm it holds "
                f"before a final alert. Not confirmed yet."
            )
            send_telegram(message)
            print(f"[pending] {symbol} score={final_score:.0f}")
            continue

        # already pending - update tracking
        entry["prices"].append(live_price)
        entry["score"] = round(final_score, 1)
        first_seen = datetime.fromisoformat(entry["first_seen"])
        elapsed_minutes = (now - first_seen).total_seconds() / 60
        running_avg_price = sum(entry["prices"]) / len(entry["prices"])

        if live_price < running_avg_price:
            print(f"[reset] {symbol} fell below its running average - clearing pending state")
            del state[state_key]
            continue

        if elapsed_minutes < CONFIRMATION_MINUTES:
            print(f"[pending] {symbol} still holding - {elapsed_minutes:.0f}/{CONFIRMATION_MINUTES} min")
            continue

        message = (
            f"Volume Surge CONFIRMED: {symbol}\n\n"
            f"Held above its running average price for {CONFIRMATION_MINUTES}+ minutes.\n"
            f"Volume: {volume_ratio:.1f}x its 20-day average\n"
            f"Taker buy ratio (today): {taker_ratio*100:.0f}%\n"
            f"Trade count vs average: {trade_ratio:.1f}x\n"
            f"vs BTC today: {relative_strength:+.1f}%\n"
            f"Final Score: {final_score:.0f}/100\n\n"
            f"Has NOT touched 5x off its post-crash low yet (per the exclusion gate).\n"
            f"This is a volume/accumulation signal, not a price-range breakout - "
            f"verify manually before acting."
        )
        if send_telegram(message):
            state[state_key] = {"status": "confirmed", "confirmed_at": now.isoformat()}
            confirmed_count += 1
            print(f"[confirmed] {symbol} score={final_score:.0f}")

    print(f"[4/4] Done. {confirmed_count} confirmed volume surge(s) this run.")
    save_state(state)
    conn.close()


if __name__ == "__main__":
    run()
