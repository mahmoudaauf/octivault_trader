#!/usr/bin/env python3
"""
Negative-funding carry backtest — WITH REAL BORROW COSTS.

WHY
---
docs/carry_frontier_findings.md established that positive-only carry (what
carry_paper_trader.py runs) is falsified: funding extremes are 98.4% NEGATIVE,
so v1 fishes in 1.6% of its own pond. The +0.93%/trade headline is real but
~98% of it lives in negative funding, which needs long-perp + short-spot —
i.e. BORROWING the spot asset on margin.

That headline never subtracted borrow interest. This script does.

Negative-funding carry mechanics:
  funding < 0  ->  shorts pay longs  ->  be LONG perp (receive |funding|)
  hedge by SHORTING spot             ->  borrow the asset, pay interest hourly
  net per 8h window = |funding| - borrow_rate_per_8h

Real Binance cross-margin borrow rates (2026-07-17) run ~0.042%-0.183% per 8h
on the alts where the opportunity concentrates. The entry threshold is
0.060%/8h. So AT THE THRESHOLD borrow roughly cancels funding — the edge, if it
exists, lives entirely in the extreme tail. This measures whether that tail is
fat enough to pay for the marginal trades.

Usage: python3 carry_negative_funding_backtest.py
Env:   FUNDING_ENTRY_BPS (6)  FUNDING_EXIT_BPS (1)
       CARRY_ROUND_TRIP_PCT (0.24)  CARRY_MAX_WINDOWS (45)
       NEG_SWEEP_ENTRIES ("6,10,20,50,100" — bps, to find where the tail pays)
       CARRY_BORROW_CACHE (data/carry_borrow_rates.json)
"""
from __future__ import annotations

import asyncio
import json
import os

from dotenv import load_dotenv

load_dotenv()

FUNDING_CACHE = "data/carry_funding_cache.json"
BORROW_CACHE = os.getenv("CARRY_BORROW_CACHE", "data/carry_borrow_rates.json")
EXIT = float(os.getenv("FUNDING_EXIT_BPS", "1")) / 10000.0
RT_COST = float(os.getenv("CARRY_ROUND_TRIP_PCT", "0.24")) / 100.0
MAX_WINDOWS = int(os.getenv("CARRY_MAX_WINDOWS", "45"))
ENTRIES_BPS = [float(x) for x in os.getenv("NEG_SWEEP_ENTRIES", "6,10,20,50,100").split(",")]


def span_days(f):
    return (f[-1][0] - f[0][0]) / 86_400_000.0 if len(f) > 1 else 0.0


async def fetch_borrow_rates(symbols) -> dict:
    """{asset: daily_interest_rate}. Cached — 361 calls is slow and the rates
    move slowly. Absent asset => not borrowable => not tradeable this way."""
    try:
        with open(BORROW_CACHE) as f:
            cached = json.load(f)
        print(f"Using cached borrow rates ({len(cached)} assets). Delete {BORROW_CACHE} to refresh.")
        return {k: float(v) for k, v in cached.items()}
    except (OSError, ValueError):
        pass

    from binance import AsyncClient

    c = await AsyncClient.create(os.getenv("BINANCE_API_KEY"), os.getenv("BINANCE_API_SECRET"))
    out = {}
    try:
        print(f"Fetching cross-margin borrow rates for {len(symbols)} assets (once, cached)...")
        for i, sym in enumerate(symbols, 1):
            asset = sym[:-4]
            try:
                r = await c.margin_interest_rate_history(asset=asset, limit=1)
                if r:
                    out[asset] = float(r[0]["dailyInterestRate"])
            except Exception:
                pass  # not borrowable / not on margin -> excluded downstream
            if i % 50 == 0:
                print(f"  ...{i}/{len(symbols)}, {len(out)} borrowable")
            await asyncio.sleep(0.12)
    finally:
        await c.close_connection()
    os.makedirs(os.path.dirname(BORROW_CACHE) or ".", exist_ok=True)
    with open(BORROW_CACHE, "w") as f:
        json.dump(out, f)
    print(f"  {len(out)}/{len(symbols)} assets are borrowable on cross-margin\n")
    return out


def simulate_negative(funding, borrow_daily: float, entry: float) -> list[float]:
    """Long perp + short spot, entered on NEGATIVE funding.

    Per 8h window: receive |funding| (shorts pay longs), pay borrow interest on
    the borrowed spot leg. Exits on |funding| < EXIT (funding normalised) or max
    hold — mirroring carry_paper_trader.py's own exit rule.
    """
    borrow_8h = borrow_daily / 3.0
    trades = []
    i, n = 0, len(funding)
    while i < n:
        _, fr = funding[i]
        if fr > -entry:  # only NEGATIVE funding beyond the threshold
            i += 1
            continue
        collected = 0.0
        held = 0
        j = i
        while j < n and held < MAX_WINDOWS:
            _, frj = funding[j]
            if held > 0 and abs(frj) < EXIT:
                break
            # Long perp receives when funding is negative, PAYS if it flips
            # positive mid-hold. Signed, committed-direction accounting.
            collected += (-frj) - borrow_8h
            held += 1
            j += 1
        trades.append((collected - RT_COST) * 100.0)
        i = j
    return trades


async def main():
    d = json.load(open(FUNDING_CACHE))
    fb = {k: [(int(t), float(r)) for t, r in v] for k, v in d["funding"].items()}
    vol = {k: float(v) for k, v in d["vol"].items()}

    borrow = await fetch_borrow_rates(sorted(fb.keys()))

    tradeable = [s for s in fb if s[:-4] in borrow]
    print("=" * 86)
    print("NEGATIVE-FUNDING CARRY (long perp + short spot) — WITH REAL BORROW COSTS")
    print("=" * 86)
    print(f"Universe: {len(fb)} spot-hedgeable perps -> {len(tradeable)} also BORROWABLE on margin")
    print(f"Round-trip cost {RT_COST*100:.2f}% | exit<{EXIT*100:.3f}% | max hold {MAX_WINDOWS}w")
    print()
    print(f"{'entry':>7} {'minvol':>7} {'trades':>7} {'trd/day':>8} {'avg%/trade':>11} {'win%':>6}"
          f"  {'net total%':>11}")
    print("-" * 86)

    for bps in ENTRIES_BPS:
        entry = bps / 10000.0
        for thr_m in [0.0, 5.0]:
            subset = [s for s in tradeable if vol.get(s, 0.0) >= thr_m * 1e6]
            trades, rate = [], 0.0
            for s in subset:
                t = simulate_negative(fb[s], borrow[s[:-4]], entry)
                trades.extend(t)
                sd = span_days(fb[s])
                if sd > 0:
                    rate += len(t) / sd
            n = len(trades)
            avg = sum(trades) / n if n else 0.0
            wr = sum(1 for x in trades if x > 0) / n * 100 if n else 0.0
            print(f"{bps:>6.0f}b {thr_m:>6.0f}M {n:>7} {rate:>8.3f} {avg:>+11.4f} {wr:>5.0f}%"
                  f"  {sum(trades):>+11.2f}")
        print("-" * 86)

    print()
    print("Compare to POSITIVE-only (falsified): +0.0697%/trade, 45% win at 6bps/$0M.")
    print("A viable point needs avg%/trade comfortably > 0 AND win% > 50 AND enough n.")
    print("NOTE: borrow rates are TODAY's snapshot applied to ~166d of history, and")
    print("      rates rise exactly when a coin is in demand to short — i.e. during")
    print("      the very negative-funding spikes this trades. Real cost is likely")
    print("      HIGHER than modelled here, so treat these numbers as optimistic.")


asyncio.run(main())
