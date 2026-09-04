#!/usr/bin/env python3
"""
Delta-neutral staking yield — systematic scan, and an HONEST persistence test.

THE IDEA
--------
Binance Simple Earn pays a staking APR on ~400 assets, some above 40%. That
yield is not free: holding the token carries its price risk, and these tokens
fall hard. The structure that removes the price risk is:

    long spot (subscribed to Simple Earn, collecting the APR)
  + short perpetual future of the same asset (hedging the price)
  = delta neutral, earning  staking_APR  ±  funding

WHY MOST OF IT IS A TRAP (measured 2026-09-04)
----------------------------------------------
The highest APRs are the worst trades, because the same scarcity that pays the
staking yield makes the hedge expensive:

    ACE   staking +40.63%   funding to a short -310.62%   net -270%
    ONG   staking +41.65%   funding to a short -111.07%   net  -69%

A scan that ranked on APR alone would pick exactly these. The market has
already priced them. Only the assets where a high-ish APR coincides with
NEUTRAL funding are candidates at all:

    ONT   +20.01%  +5.48%  ->  +25.5%
    SYN   +18.43%  +5.48%  ->  +23.9%
    MOVE  +16.81%  +6.36%  ->  +23.2%

WHY THIS IS A MONITOR AND NOT A TRADER
--------------------------------------
Those numbers are ONE SNAPSHOT, and a snapshot is not an edge. Both legs move:
  - staking APRs are promotional and can collapse overnight
  - funding reprices every 8 hours, and this project has already been burned
    by exactly that. Delta-neutral funding farming on 12 majors looked fine and
    came out at roughly -2%/yr over 166 days.
So this file MEASURES, appending every scan to a JSONL log, and `report` shows
how stable each candidate has actually been. A candidate earns real money only
after its net rate holds up across many scans, not because it looked good once.

COSTS AND FRICTIONS THIS ACCOUNTS FOR
-------------------------------------
  - round-trip execution on both legs (COST_PCT, default 0.4% all-in)
  - capital split: Simple Earn balance CANNOT double as futures margin, so
    roughly half the capital sits as collateral earning nothing. The reported
    `net_on_capital` halves the yield accordingly rather than quoting the
    headline on money that is only half deployed.
Not modelled, and they matter: liquidation risk on the short, borrow/quota
limits, min notionals, and APR changing while you hold.

Usage:
  python3 delta_neutral_yield_scan.py          # one scan, append to the log
  python3 delta_neutral_yield_scan.py report   # persistence across all scans
"""
from __future__ import annotations

import asyncio
import json
import os
import statistics
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

LOG = os.getenv("DNY_LOG", "logs/delta_neutral_scans.jsonl")
# All-in round-trip cost estimate across both legs (spot + perp, in and out).
COST_PCT = float(os.getenv("DNY_COST_PCT", "0.4"))
# Minimum staking APR worth the operational effort at all.
MIN_APR = float(os.getenv("DNY_MIN_APR", "0.08"))
# Assume half the capital is tied up as futures margin, earning nothing.
DEPLOYED_FRACTION = float(os.getenv("DNY_DEPLOYED_FRACTION", "0.5"))
TOP_N = int(os.getenv("DNY_TOP_N", "15"))


async def _flexible_products(client) -> dict:
    """{asset: apr} for every purchasable flexible product, paged."""
    out, page = {}, 1
    while True:
        r = await client.get_simple_earn_flexible_product_list(current=page, size=100)
        rows = r.get("rows") or []
        for p in rows:
            if p.get("canPurchase") is False or p.get("isSoldOut") is True:
                continue
            try:
                apr = float(p.get("latestAnnualPercentageRate", 0) or 0)
            except (TypeError, ValueError):
                continue
            asset = str(p.get("asset", "")).upper()
            if asset:
                out[asset] = max(out.get(asset, 0.0), apr)
        if len(rows) < 100:
            return out
        page += 1


async def _perp_funding(client) -> dict:
    """{symbol: annualised funding rate paid TO a short}.

    Positive means a short RECEIVES. Annualised from the last 8h rate, which is
    a snapshot and is treated as such everywhere downstream.
    """
    info = await client.futures_exchange_info()
    perps = {s["symbol"] for s in info["symbols"]
             if s.get("contractType") == "PERPETUAL" and s.get("status") == "TRADING"}
    out = {}
    for row in await client.futures_mark_price():
        sym = row.get("symbol")
        if sym in perps:
            try:
                out[sym] = float(row.get("lastFundingRate", 0) or 0) * 3 * 365
            except (TypeError, ValueError):
                continue
    return out


async def scan() -> dict:
    from binance import AsyncClient
    from exchange_resilience import create_client_with_retry
    client = await create_client_with_retry(AsyncClient)
    try:
        aprs = await _flexible_products(client)
        funding = await _perp_funding(client)
    finally:
        await client.close_connection()

    rows = []
    for asset, apr in aprs.items():
        if apr < MIN_APR:
            continue
        sym = f"{asset}USDT"
        if sym not in funding:
            continue                      # unhedgeable: price risk cannot be removed
        f = funding[sym]
        gross = apr + f                   # a short RECEIVES positive funding
        rows.append({
            "asset": asset, "staking_apr": round(apr, 6),
            "funding_to_short": round(f, 6),
            "gross": round(gross, 6),
            # Halved because Simple Earn balance cannot also serve as margin.
            "net_on_capital": round(gross * DEPLOYED_FRACTION - COST_PCT / 100.0, 6),
        })
    rows.sort(key=lambda r: -r["net_on_capital"])
    rec = {"ts": datetime.now(timezone.utc).isoformat(),
           "scanned": len(aprs), "hedgeable": len(rows), "rows": rows[:60]}
    os.makedirs(os.path.dirname(LOG) or ".", exist_ok=True)
    with open(LOG, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    return rec


def _print_scan(rec: dict) -> None:
    print("=" * 78)
    print(f"DELTA-NEUTRAL YIELD SCAN — {rec['ts'][:16]}   "
          f"{rec['scanned']} products, {rec['hedgeable']} hedgeable")
    print("=" * 78)
    print(f"  {'asset':<8}{'staking':>10}{'funding':>11}{'gross':>10}{'net/capital':>13}")
    for r in rec["rows"][:TOP_N]:
        print(f"  {r['asset']:<8}{r['staking_apr']*100:>9.2f}%{r['funding_to_short']*100:>10.2f}%"
              f"{r['gross']*100:>9.2f}%{r['net_on_capital']*100:>12.2f}%")
    print(f"\n  net/capital assumes {DEPLOYED_FRACTION:.0%} deployed (rest is futures")
    print(f"  margin earning nothing) and {COST_PCT}% round-trip cost.")
    print("  ONE SNAPSHOT. Not an edge until `report` shows it persists.")


def report() -> None:
    if not os.path.exists(LOG):
        print("No scans yet. Run the scan a few times a day for a week first.")
        return
    scans = []
    with open(LOG) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    scans.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    if not scans:
        print("No readable scans yet.")
        return
    series: dict[str, list] = {}
    for s in scans:
        for r in s["rows"]:
            series.setdefault(r["asset"], []).append(r["net_on_capital"])
    print("=" * 78)
    print(f"PERSISTENCE — {len(scans)} scans, {scans[0]['ts'][:16]} -> {scans[-1]['ts'][:16]}")
    print("=" * 78)
    if len(scans) < 5:
        print(f"  ⚠️  only {len(scans)} scans. A rate that looked good once means")
        print("      nothing; this needs days of scans before it says anything.")
    print(f"  {'asset':<8}{'n':>4}{'mean':>9}{'worst':>9}{'best':>9}{'stdev':>9}  verdict")
    ranked = sorted(series.items(),
                    key=lambda kv: -(statistics.mean(kv[1]) if kv[1] else 0))
    for asset, vals in ranked[:TOP_N]:
        if not vals:
            continue
        mean = statistics.mean(vals)
        sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        # Present in every scan AND never negative is the only interesting shape.
        verdict = ("candidate" if len(vals) == len(scans) and min(vals) > 0
                   else "intermittent" if len(vals) < len(scans)
                   else "goes negative")
        print(f"  {asset:<8}{len(vals):>4}{mean*100:>8.1f}%{min(vals)*100:>8.1f}%"
              f"{max(vals)*100:>8.1f}%{sd*100:>8.1f}%  {verdict}")
    print("\n  'candidate' = positive in EVERY scan so far. Still not proof:")
    print("  funding-farming on majors also looked fine before it returned -2%/yr.")


if __name__ == "__main__":
    if (sys.argv[1:] or [""])[0] == "report":
        report()
    else:
        _print_scan(asyncio.run(scan()))
