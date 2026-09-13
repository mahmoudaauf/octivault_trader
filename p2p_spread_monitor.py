#!/usr/bin/env python3
"""USDT/EGP P2P spread — does the 1.4% persist, or was it a lucky snapshot?

WHY THIS EXISTS
---------------
On 2026-09-13 a single reading of Binance P2P showed USDT/EGP buying at 51.70
and selling at 52.43 — a 1.41% round-trip spread. That is the only mechanism
found in this project whose arithmetic reaches $1/hour without six figures:
profit is capital x spread x cycles, so ~$400 turned over a few times a day
clears the target where $192,700 of yield capital would be needed instead.

It is also ONE SNAPSHOT, and this repo has an expensive lesson about those.
delta_neutral_yield_scan.py ranked candidates on a snapshot and 11 of 25 were
NEGATIVE on 90-day history; the scan now ranks on the mean and says so in its
own docstring. A spread is not an edge until it persists.

So this records the book on a schedule and reports the DISTRIBUTION, not the
latest number. What matters for a turnover business is not the best spread ever
seen but the spread available at a typical moment — specifically the lower
quantiles, because those are the ones you will actually trade through.

WHAT IT DOES NOT TELL YOU
-------------------------
Persistence of the QUOTED spread is necessary, not sufficient. It cannot
measure:
  - whether a counterparty will actually fill you at the top-of-book price
  - payment-rail mismatch: the best buy ad may be InstaPay while the best sell
    ad is Vodafone Cash, and you need both rails to capture the pair
  - fraud, chargebacks and reversed transfers
  - the regulatory and bank-account risk of high-volume crypto-linked transfers
  - that capturing the spread means being the MAKER; as a taker you PAY it
Those are operator questions. This answers exactly one: is the spread real and
stable enough to be worth the rest of the diligence?

Read-only. No credentials, no orders, no money.

Usage:
  python3 p2p_spread_monitor.py sample     # record one observation
  python3 p2p_spread_monitor.py report     # distribution so far
  python3 p2p_spread_monitor.py watch      # sample every SAMPLE_MIN minutes
Env: P2P_FIAT (EGP)  P2P_ASSET (USDT)  P2P_SAMPLE_MIN (15)  P2P_STATE
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
import urllib.request
from datetime import datetime, timezone

FIAT = os.getenv("P2P_FIAT", "EGP")
ASSET = os.getenv("P2P_ASSET", "USDT")
STATE = os.getenv("P2P_STATE", "logs/p2p_spread_history.jsonl")
SAMPLE_MIN = float(os.getenv("P2P_SAMPLE_MIN", "15"))
URL = "https://p2p.binance.com/bapi/c2c/v2/friendly/c2c/adv/search"


def _ads(trade_type: str, rows: int = 20) -> list[dict]:
    body = json.dumps({"page": 1, "rows": rows, "payTypes": [], "asset": ASSET,
                       "tradeType": trade_type, "fiat": FIAT,
                       "publisherType": None}).encode()
    req = urllib.request.Request(URL, data=body, headers={
        "Content-Type": "application/json", "User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=25) as r:
        return json.loads(r.read().decode()).get("data") or []


def sample() -> dict | None:
    """One observation of both sides of the book.

    Records the MINIMUM ORDER SIZE alongside the price, because a spread you
    cannot reach is not a spread: at $60 this account sits below the smallest
    ad on the book and can place no trade at all.
    """
    try:
        buys, sells = _ads("BUY"), _ads("SELL")
    except Exception as e:
        print(f"[p2p] fetch failed: {str(e)[:90]}")
        return None
    if not buys or not sells:
        print("[p2p] empty book — not recording")
        return None

    def px(a):
        return float(a["adv"]["price"])

    def minamt(a):
        return float(a["adv"]["minSingleTransAmount"])

    best_buy = min(buys, key=px)      # cheapest price we can BUY at
    best_sell = max(sells, key=px)    # highest price we can SELL at
    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "buy": px(best_buy), "sell": px(best_sell),
        "buy_min_fiat": minamt(best_buy), "sell_min_fiat": minamt(best_sell),
        "spread_pct": round((px(best_sell) - px(best_buy)) / px(best_buy) * 100, 4),
        # Medians are the honest middle of the book; top-of-book can be one
        # outlier ad with an unreachable minimum.
        "buy_median": round(statistics.median(px(a) for a in buys), 4),
        "sell_median": round(statistics.median(px(a) for a in sells), 4),
        "n_buy": len(buys), "n_sell": len(sells),
    }
    row["spread_median_pct"] = round(
        (row["sell_median"] - row["buy_median"]) / row["buy_median"] * 100, 4)
    os.makedirs(os.path.dirname(STATE) or ".", exist_ok=True)
    with open(STATE, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"[p2p {row['ts'][11:16]}] buy {row['buy']:.2f} sell {row['sell']:.2f} "
          f"top {row['spread_pct']:+.3f}%  median {row['spread_median_pct']:+.3f}%  "
          f"min order {row['buy_min_fiat']:,.0f} {FIAT}")
    return row


def report() -> int:
    try:
        rows = [json.loads(l) for l in open(STATE) if l.strip()]
    except FileNotFoundError:
        print(f"No history yet at {STATE}. Run: p2p_spread_monitor.py sample")
        return 1
    if not rows:
        print("No observations recorded yet.")
        return 1
    tops = [r["spread_pct"] for r in rows]
    meds = [r.get("spread_median_pct", r["spread_pct"]) for r in rows]
    span_h = (datetime.fromisoformat(rows[-1]["ts"]) -
              datetime.fromisoformat(rows[0]["ts"])).total_seconds() / 3600

    def q(xs, p):
        s = sorted(xs)
        return s[min(int(len(s) * p), len(s) - 1)]

    print("=" * 70)
    print(f"{ASSET}/{FIAT} P2P SPREAD — {len(rows)} observations over {span_h:.1f}h")
    print("=" * 70)
    for label, xs in (("top-of-book", tops), ("median-of-book", meds)):
        print(f"  {label:<16} mean {statistics.mean(xs):+.3f}%   "
              f"p10 {q(xs,0.10):+.3f}%   p50 {q(xs,0.50):+.3f}%   "
              f"p90 {q(xs,0.90):+.3f}%   min {min(xs):+.3f}%")
    # What the operator actually earns depends on the TYPICAL spread, not the best.
    typical = q(meds, 0.50)
    print()
    print(f"  A turnover business earns the TYPICAL spread, not the best one.")
    print(f"  At the median-of-book p50 of {typical:+.3f}%, $1/hour needs "
          f"${100/typical if typical>0 else float('inf'):,.0f} of turnover per hour.")
    reach = statistics.median(r["buy_min_fiat"] for r in rows)
    print(f"  Median minimum order on the buy side: {reach:,.0f} {FIAT}"
          f"  (~${reach/statistics.median(r['buy'] for r in rows):,.0f})")
    if span_h < 48:
        print(f"\n  ⚠️  {span_h:.1f}h of data. This repo's own history says a snapshot")
        print("      misleads: 11 of 25 delta-neutral candidates looked positive on a")
        print("      snapshot and were negative on 90-day history. Not a verdict yet.")
    print("=" * 70)
    return 0


def main() -> int:
    cmd = (sys.argv[1:] or ["sample"])[0]
    if cmd == "report":
        return report()
    if cmd == "watch":
        print(f"[p2p] sampling {ASSET}/{FIAT} every {SAMPLE_MIN:.0f} min — Ctrl-C to stop")
        while True:
            sample()
            time.sleep(SAMPLE_MIN * 60)
    sample()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
