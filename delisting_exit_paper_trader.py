#!/usr/bin/env python3
"""
Delisting-exit paper trader -- forward proof-of-edge engine (narrow scope).

Backtest (news_sentiment_backtest.py, see the delisting-notice-exit-edge-
candidate memory) found: a Binance delisting announcement predicts a real
price drop in the affected symbol over the following hours/days, robust
even under a 4x-stressed cost assumption (all of 1h/4h/24h/72h horizons
positive out-of-sample). BUT ~41% of that sample was already thin (<$50k/hr
pre-announcement volume) and only 9% had a clear futures market to short
through -- so an ACTIVE short-selling version of this carries real,
unresolved execution risk.

Scope, deliberately narrowed (explicit user decision, 2026-07-15): this
daemon ONLY acts on delisting notices for symbols the account CURRENTLY
HOLDS -- "sell what you already own the moment a delisting notice drops,"
not "open a new short on a coin you don't hold." This sidesteps the
thin-market-shorting risk entirely: it's accelerating an exit you'd need to
make anyway before the pair closes, not entering a new risky position.

Runs against the REAL account's real spot balances (read-only in paper/
dryrun mode) -- the same account main.py trades -- so this is a faithful
forward test of what THIS account's actual future holdings would benefit
from, not a synthetic watchlist. The account is presently cash-heavy, so
this daemon may see few or no trigger events for a while -- an honest
scope consequence, not a bug. Do not widen to a broader symbol watchlist
without another explicit decision, since that reintroduces the
thin-market/no-futures-market risk this narrowing was meant to avoid.

Usage:
  python3 delisting_exit_paper_trader.py            # run the paper daemon
  python3 delisting_exit_paper_trader.py report      # print the live edge verdict
Env: DELIST_EXIT_MODE (paper|dryrun|live, default paper)
     DELIST_EXIT_POLL_MIN (10) DELIST_EXIT_HORIZONS_H ("1,4,24,72")
     DELIST_EXIT_LIVE_ARM_FILE (logs/delisting_exit_live_armed) -- 2nd gate
     DELIST_EXIT_KILL_FILE (logs/delisting_exit.stop)
     DELIST_EXIT_MIN_TRADES (30) -- forward-proof gate, matches carry's convention
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

load_dotenv()

STATE = "logs/delisting_exit_state.json"
LEDGER = "logs/delisting_exit_ledger.jsonl"

BAPI_URL = "https://www.binance.com/bapi/composite/v1/public/cms/article/list/query"
CATALOG_DELISTING = 161

MODE = os.getenv("DELIST_EXIT_MODE", "paper").lower()
POLL_MIN = float(os.getenv("DELIST_EXIT_POLL_MIN", "10"))
HORIZONS_H = [float(x) for x in os.getenv("DELIST_EXIT_HORIZONS_H", "1,4,24,72").split(",")]
LIVE_ARM_FILE = os.getenv("DELIST_EXIT_LIVE_ARM_FILE", "logs/delisting_exit_live_armed")
KILL_FILE = os.getenv("DELIST_EXIT_KILL_FILE", "logs/delisting_exit.stop")
MIN_TRADES = int(os.getenv("DELIST_EXIT_MIN_TRADES", "30"))

STABLE_ASSETS = {"USDT", "USDC", "BUSD", "FDUSD", "DAI", "TUSD", "BNB"}
DUST_THRESHOLD_USD = 5.0  # ignore holdings too small to matter

TICKER_STOPWORDS = {
    "USD", "USDT", "BUSD", "THE", "AND", "FOR", "NEW", "API", "ATH", "APR",
    "ADD", "ADDS", "ADDING", "WILL", "NOTICE", "REMOVAL", "SPOT", "MARGIN",
    "LOAN", "ON", "OF", "TO", "IS", "BINANCE", "FUTURES", "EXCHANGE",
    "LAUNCH", "LAUNCHES", "TRADING", "PAIR", "PAIRS", "PAIR(S)", "LISTING",
    "DELIST", "DELISTING", "PERPETUAL", "CONTRACT", "CONTRACTS", "UPDATE",
    "UPDATES", "UPDATED",
}
TICKER_RE = re.compile(r"\b[A-Z][A-Z0-9]{1,9}\b")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
}


def extract_tickers(title: str) -> list[str]:
    tokens = TICKER_RE.findall(title)
    out: list[str] = []
    for t in tokens:
        base = t[:-4] if t.endswith("USDT") else t
        if base in TICKER_STOPWORDS or len(base) < 2:
            continue
        if base not in out:
            out.append(base)
    return out


def _killed() -> bool:
    return os.path.exists(KILL_FILE)


def _load(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _save(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def _log_trade(rec: dict):
    with open(LEDGER, "a") as f:
        f.write(json.dumps(rec) + "\n")


def fetch_recent_delisting_articles(page_size: int = 50) -> list[dict]:
    """Just the most recent notices -- state's seen_articles set is what
    prevents reprocessing, so deep pagination isn't needed on every poll."""
    try:
        resp = requests.get(
            BAPI_URL,
            params={"type": 1, "catalogId": CATALOG_DELISTING, "pageNo": 1, "pageSize": page_size},
            headers=HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()["data"]["catalogs"][0]["articles"]
    except Exception as e:
        print(f"[delist-exit] delisting fetch failed: {str(e)[:100]}")
        return []


async def get_real_holdings(client) -> dict[str, float]:
    """{asset: free_qty} for non-stable, non-dust holdings in the real spot account."""
    try:
        acct = await client.get_account()
    except Exception as e:
        print(f"[delist-exit] account fetch failed: {str(e)[:100]}")
        return {}
    out = {}
    for b in acct.get("balances", []):
        asset = b.get("asset", "")
        if asset in STABLE_ASSETS:
            continue
        free = float(b.get("free", 0.0) or 0.0)
        if free <= 0:
            continue
        out[asset] = free
    return out


async def _price(client, symbol: str) -> float | None:
    try:
        t = await client.get_symbol_ticker(symbol=f"{symbol}USDT")
        return float(t["price"])
    except Exception:
        return None


async def execute_exit(client, symbol: str, qty: float) -> tuple[bool, str]:
    """Returns (acted, description). paper: log only. dryrun: log the intended
    order. live: place a REAL market sell for the held quantity -- DOUBLE-
    GATED (MODE=live AND arm file). NOTE: quantity precision (LOT_SIZE step)
    is NOT hardened here -- this needs testnet validation before real arming,
    same discipline carry_paper_trader.py's own docstring calls for."""
    desc = f"SELL {qty} {symbol} (existing holding, delisting notice)"
    if MODE == "paper":
        return True, f"[PAPER] {desc}"
    if MODE == "dryrun":
        return True, f"[DRYRUN] {desc}"
    if MODE == "live":
        if not os.path.exists(LIVE_ARM_FILE):
            return False, f"[LIVE-BLOCKED] {symbol}: arm file '{LIVE_ARM_FILE}' missing -- no order sent"
        try:
            await client.create_order(symbol=f"{symbol}USDT", side="SELL", type="MARKET", quantity=qty)
            return True, f"[LIVE] {desc}"
        except Exception as e:
            return False, f"[LIVE-ERROR] {symbol}: {str(e)[:100]}"
    return False, "unknown mode"


async def forward_price(client, symbol: str, ts_ms: int, horizon_h: float) -> float | None:
    try:
        target = ts_ms + int(horizon_h * 3600_000)
        klines = await client.get_historical_klines(
            f"{symbol}USDT", "1h", start_str=str(target - 3600_000), end_str=str(target + 3600_000)
        )
    except Exception:
        return None
    if not klines:
        return None
    return float(klines[-1][4])


def _report():
    trades = []
    for line in open(LEDGER) if os.path.exists(LEDGER) else []:
        line = line.strip()
        if line:
            try:
                trades.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    if not trades:
        print("No closed delisting-exit trades yet.")
        return
    print("=" * 64)
    print(f"DELISTING-EXIT PAPER — FORWARD TRACK RECORD ({len(trades)} closed)")
    print("=" * 64)
    for h in HORIZONS_H:
        vals = [t["avoided_loss_pct"][str(h)] for t in trades if str(h) in t.get("avoided_loss_pct", {})]
        if not vals:
            continue
        n = len(vals)
        wins = sum(1 for v in vals if v > 0)
        avg = sum(vals) / n
        print(f"  {h:>5.1f}h   n={n:<4} avg_avoided_loss={avg:+.3f}%   win-rate={wins/n*100:.0f}%")
    primary = [t["avoided_loss_pct"].get("72.0") for t in trades if "72.0" in t.get("avoided_loss_pct", {})]
    n = len(primary)
    print("-" * 64)
    if n < MIN_TRADES:
        print(f"VERDICT: ⏳ INCONCLUSIVE — {n}/{MIN_TRADES} trades (72h horizon). Keep running.")
    elif sum(primary) / n > 0:
        print(f"VERDICT: ✅ FORWARD EDGE CONFIRMED (72h) — avg avoided loss {sum(primary)/n:+.3f}%.")
    else:
        print(f"VERDICT: ❌ NO FORWARD EDGE (72h) — avg avoided loss {sum(primary)/n:+.3f}%.")
    print("=" * 64)


async def run():
    from binance import AsyncClient

    testnet = os.getenv("DELIST_EXIT_TESTNET", "false").lower() in ("1", "true", "yes")
    if testnet:
        client = await AsyncClient.create(
            os.getenv("BINANCE_TESTNET_API_KEY") or "x",
            os.getenv("BINANCE_TESTNET_API_SECRET_HMAC") or "x",
            testnet=True,
        )
    else:
        client = await AsyncClient.create(
            os.getenv("BINANCE_API_KEY") or "x", os.getenv("BINANCE_API_SECRET") or "x"
        )

    state = _load(STATE, {"pending": {}, "seen_articles": []})
    seen = set(state.get("seen_articles", []))
    max_h = max(HORIZONS_H)

    armed = MODE == "live" and os.path.exists(LIVE_ARM_FILE)
    banner = {"paper": "📝 PAPER (no orders)", "dryrun": "🧪 DRY-RUN (logs orders, sends none)",
              "live": ("🔴 LIVE-ARMED (REAL ORDERS)" if armed else "🔴 live (BLOCKED — arm file missing)")}
    print(f"[delist-exit] start — MODE={MODE} → {banner.get(MODE, MODE)}")
    print(f"[delist-exit] scope — existing holdings only (no new shorts) · "
          f"horizons={HORIZONS_H} · poll={POLL_MIN}m · kill_file={KILL_FILE}")

    try:
        while True:
            now_ms = int(time.time() * 1000)

            if not _killed():
                holdings = await get_real_holdings(client)
                articles = fetch_recent_delisting_articles()
                for art in articles:
                    aid = art.get("id")
                    if aid is None or aid in seen:
                        continue
                    seen.add(aid)
                    title = art.get("title", "")
                    tickers = extract_tickers(title)
                    for sym in tickers:
                        if sym not in holdings or sym in state["pending"]:
                            continue
                        price = await _price(client, sym)
                        if price is None:
                            continue
                        acted, desc = await execute_exit(client, sym, holdings[sym])
                        print(f"[delist-exit] TRIGGER: {desc} (notice: {title[:70]})")
                        state["pending"][sym] = {
                            "notice_ts": now_ms, "article_id": aid,
                            "exit_price": price, "acted": acted,
                        }
                # Persist every cycle regardless of whether any article matched a
                # held asset -- otherwise a restart would re-scan already-seen
                # (but irrelevant) articles indefinitely.
                state["seen_articles"] = list(seen)[-500:]  # bounded memory

            # Finalize pending entries once the longest horizon has elapsed.
            for sym in list(state["pending"].keys()):
                pos = state["pending"][sym]
                held_h = (now_ms - pos["notice_ts"]) / 3600_000
                if held_h < max_h:
                    continue
                avoided: dict[str, float] = {}
                for h in HORIZONS_H:
                    fwd = await forward_price(client, sym, pos["notice_ts"], h)
                    if fwd is not None and pos["exit_price"] > 0:
                        avoided[str(h)] = (pos["exit_price"] - fwd) / pos["exit_price"] * 100.0
                if avoided:
                    _log_trade({
                        "ts": datetime.now(timezone.utc).isoformat(), "symbol": sym,
                        "notice_ts": pos["notice_ts"], "exit_price": pos["exit_price"],
                        "avoided_loss_pct": avoided, "mode": MODE,
                    })
                del state["pending"][sym]

            _save(STATE, state)
            ts = datetime.now(timezone.utc).strftime("%H:%M")
            print(f"[delist-exit {ts}] pending={len(state['pending'])} seen={len(seen)}")
            await asyncio.sleep(POLL_MIN * 60)
    finally:
        await client.close_connection()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        _report()
    else:
        asyncio.run(run())
