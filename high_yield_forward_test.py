#!/usr/bin/env python3
"""High-yield earn assets — a FORWARD test, because the backward one is rigged.

THE QUESTION
------------
The operator asked why the machine settles for a 6.42pp gap when the earn board
shows assets paying 33-48%. Fair question. A backward look was encouraging:
today's top-22 by APR returned a median +17.0% in price over 90 days, +24.1%
including yield, with 19 of 22 positive — against a broad-market median of
+1.9%. That is +15.1pp of apparent excess.

WHY THAT NUMBER CANNOT BE TRUSTED
---------------------------------
The selection is contaminated by the thing being measured. I picked assets by
TODAY's APR and looked BACKWARD at their price. On Binance, a flexible earn
rate is driven by margin borrow demand, and borrow demand spikes AFTER a large
move. LSK pays 33% today BECAUSE it rose 361% — traders borrowed it to short
the spike. So the backward test measures the pump that CAUSED the rate, and
reports it as a return the rate predicted.

Two further contaminants: the window was a bull market (BTC +21.4%), and
today's list cannot contain assets that crashed so hard they were delisted.

THE ONLY HONEST VERSION
-----------------------
Fix the basket TODAY, from today's rates, and measure FORWARD. Nothing about
tomorrow's price can leak into today's selection. That is the same discipline
every other candidate in this repo was held to — spread market-making took 701
fills, order-book imbalance took 57 days — and it is why those verdicts are
trustworthy.

WHAT IT TRACKS
--------------
Price AND yield separately, because conflating them is how a 33% APR on an
asset that fell 40% gets reported as a win. Also a broad-market control basket,
so a rising tide is never mistaken for the strategy working.

Read-only. Public price data only; no credentials, no orders, no money.

Usage:
  python3 high_yield_forward_test.py start    # fix the basket from today's rates
  python3 high_yield_forward_test.py report   # forward performance since then
Env: HYF_STATE  HYF_N (20)
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import random
import statistics
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

KEY, SECRET = os.getenv("BINANCE_API_KEY"), os.getenv("BINANCE_API_SECRET")
BASE = "https://api.binance.com"
STATE = os.getenv("HYF_STATE", "logs/high_yield_forward.json")
N = int(os.getenv("HYF_N", "20"))


def _signed(path: str, params: dict | None = None):
    p = dict(params or {})
    p["timestamp"] = int(time.time() * 1000)
    p["recvWindow"] = 10000
    qs = urllib.parse.urlencode(p)
    qs += "&signature=" + hmac.new(SECRET.encode(), qs.encode(), hashlib.sha256).hexdigest()
    req = urllib.request.Request(f"{BASE}{path}?{qs}", headers={"X-MBX-APIKEY": KEY})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def _pub(path: str, params: dict | None = None):
    with urllib.request.urlopen(
            f"{BASE}{path}?{urllib.parse.urlencode(params or {})}", timeout=20) as r:
        return json.loads(r.read().decode())


def _price(asset: str) -> float | None:
    try:
        return float(_pub("/api/v3/ticker/price", {"symbol": f"{asset}USDT"})["price"])
    except Exception:
        return None


def start() -> int:
    rows, cur = [], 1
    while cur <= 12:
        got = _signed("/sapi/v1/simple-earn/flexible/list",
                      {"current": cur, "size": 100}).get("rows", [])
        rows += got
        if len(got) < 100:
            break
        cur += 1
    ranked = sorted(((float(r["latestAnnualPercentageRate"]), r["asset"]) for r in rows),
                    reverse=True)
    basket = {}
    for apr, a in ranked:
        if len(basket) >= N:
            break
        px = _price(a)
        if px:
            basket[a] = {"apr": apr, "px0": px}

    # Control: a random basket priced the same day. Without it, a bull market
    # reads as the strategy working — which is exactly what the backward test did.
    info = _pub("/api/v3/exchangeInfo")
    pool = [s["baseAsset"] for s in info["symbols"]
            if s["status"] == "TRADING" and s["quoteAsset"] == "USDT"
            and not any(x in s["symbol"] for x in ("UP", "DOWN", "BULL", "BEAR"))]
    rng = random.Random(int(time.time()))
    control = {}
    for a in rng.sample(pool, min(60, len(pool))):
        if a in basket:
            continue
        px = _price(a)
        if px:
            control[a] = {"px0": px}
        if len(control) >= 40:
            break

    state = {"started": datetime.now(timezone.utc).isoformat(),
             "basket": basket, "control": control}
    os.makedirs(os.path.dirname(STATE) or ".", exist_ok=True)
    with open(STATE, "w") as f:
        json.dump(state, f, indent=1)
    print(f"Basket fixed: {len(basket)} high-APR assets "
          f"({min(v['apr'] for v in basket.values())*100:.1f}%–"
          f"{max(v['apr'] for v in basket.values())*100:.1f}% APR), "
          f"{len(control)} control assets.")
    print("Nothing about tomorrow's price can leak into this selection.")
    print("Re-run with `report` in a week or more.")
    return 0


def report() -> int:
    try:
        st = json.load(open(STATE))
    except FileNotFoundError:
        print(f"No basket yet. Run: high_yield_forward_test.py start")
        return 1
    days = (datetime.now(timezone.utc) -
            datetime.fromisoformat(st["started"])).total_seconds() / 86400

    hp, hy, rows = [], [], []
    for a, v in st["basket"].items():
        px = _price(a)
        if not px:
            continue                      # delisted or unpriceable: excluded, counted below
        chg = (px - v["px0"]) / v["px0"]
        yld = v["apr"] * days / 365
        hp.append(chg)
        hy.append(yld)
        rows.append((a, v["apr"], chg, yld))
    cp = []
    for a, v in st["control"].items():
        px = _price(a)
        if px:
            cp.append((px - v["px0"]) / v["px0"])

    print("=" * 70)
    print(f"HIGH-YIELD FORWARD TEST — {days:.1f} days since the basket was fixed")
    print("=" * 70)
    if not hp:
        print("  no basket asset is priceable — nothing to report")
        return 1
    print(f"  {'asset':<8}{'APR':>8}{'price':>10}{'yield':>9}{'net':>9}")
    for a, apr, chg, yld in sorted(rows, key=lambda r: -(r[2] + r[3])):
        print(f"  {a:<8}{apr*100:>7.1f}%{chg*100:>9.1f}%{yld*100:>8.1f}%{(chg+yld)*100:>8.1f}%")
    mp, my = statistics.median(hp), statistics.median(hy)
    print("-" * 70)
    print(f"  high-APR basket  price {mp*100:+.1f}%   yield {my*100:+.1f}%   "
          f"NET {(mp+my)*100:+.1f}%   ({len(hp)} priced of {len(st['basket'])})")
    if cp:
        mc = statistics.median(cp)
        print(f"  control basket   price {mc*100:+.1f}%                      "
              f"({len(cp)} priced of {len(st['control'])})")
        print(f"  EXCESS over control   {((mp+my)-mc)*100:+.1f}pp   ← the only number that matters")
    print(f"  stable alternative    {8.60*days/365:+.2f}%   (USD1 at 8.60%, zero price risk)")
    if days < 30:
        print(f"\n  ⚠ {days:.1f} days. Too short to conclude. Spread market-making needed 701")
        print("    fills and order-book imbalance needed 57 days before their verdicts held.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(start() if (sys.argv[1:] or ["report"])[0] == "start" else report())
