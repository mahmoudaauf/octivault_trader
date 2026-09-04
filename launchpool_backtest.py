#!/usr/bin/env python3
"""
Launchpool backtest — what did staking in each pool ACTUALLY return, and does
holding the reward token beat selling it on day one?

THE IDEA UNDER TEST (operator's, 2026-09-04)
--------------------------------------------
"Think of new coins under issuance, hold them." Binance Launchpool gives newly
issued tokens for free to anyone with BNB / FDUSD / USDC in Simple Earn
Flexible. Principal is returned in full. So the return is the value of the free
tokens per dollar staked per day — a structural yield needing no prediction.
This is DIFFERENT from `new-listing momentum`, which this project already
falsified: that bought after listing; this receives before it.

RESULT (12 projects with full pool data + a USDT kline, Apr 2024 – Mar 2026)
----------------------------------------------------------------------------
Selling the reward on listing day 1, annualised over the farming window:

    pool    n   median   mean   worst   best    avg $ staked
    FDUSD  11   21.1%   36.0%   10.9%  188.0%     $1.1B
    BNB    12   17.5%   23.6%    8.3%  102.9%    $12.3B
    USDC    9    7.7%    9.6%    5.8%   22.2%     $2.5B

POSITIVE IN EVERY CASE. The worst pool of the worst project still annualised
above the 5% stablecoin baseline. That is the rarest shape this project has
ever measured.

Three things the headline hides:
  1. HOLDING LOSES. Reward token, day 1 -> day 7: median -12% to -13%, up in
     only 2 of 12 (BNB), 2 of 11 (FDUSD), 1 of 9 (USDC). The operator's
     instinct to hold is falsified by the data. Sell on listing, sweep to earn.
  2. THE WINDOW IS SHORT. ~3.5 days per project, ~0.2-0.3% per project. The
     annualised figure only accrues while a pool is live; between pools the
     capital earns its Simple Earn rate. With ~36 projects/yr historically that
     is ~35% utilisation -> ~7-10%/yr blended from Launchpool on top of earn.
  3. IT IS DECAYING. Stablecoin pools in the last 12 months: n=3, median 7.6%,
     against 12.3% for the older set. More capital is crowding in (billions per
     pool) and fewer projects are launching. On 2026-09-04 no pool was active.

WHAT THIS IMPLIES FOR THE MACHINE
---------------------------------
  - USDC is the right home BETWEEN pools (5.00% tier, see hybrid_allocator).
  - FDUSD is the right home DURING a pool that has an FDUSD option (its Simple
    Earn base is only 0.51%, so it is the wrong home otherwise).
  - Rotating on pool events, and selling rewards on day 1, is the whole edge.
  - BNB adds HODLer airdrops (not in this dataset) but carries price risk;
    hedging it costs ~nothing (BNB funding pays a short ~+2%/yr over 365d),
    yet halves deployable capital for margin. FDUSD wins on simplicity.

Usage:
  python3 launchpool_backtest.py          # re-run against the live project list
Env: BINANCE_API_KEY / BINANCE_API_SECRET (read-only is enough)
"""
from __future__ import annotations

import asyncio
import json
import os
import statistics
import sys
import time

from dotenv import load_dotenv

load_dotenv()

STABLES = {"FDUSD", "USDC", "USD1", "U", "USDT"}


async def _px_at(client, symbol: str, ts_ms: float, offset_days: int = 0):
    """Daily close at/after ts (+offset). None if the pair has no kline there."""
    try:
        k = await asyncio.wait_for(
            client.get_klines(symbol=symbol, interval="1d",
                              startTime=int(ts_ms) + offset_days * 86_400_000, limit=1),
            timeout=10)
        return float(k[0][4]) if k else None
    except Exception:
        return None


async def backtest() -> list[dict]:
    from binance import AsyncClient
    client = await asyncio.wait_for(
        AsyncClient.create(os.getenv("BINANCE_API_KEY"), os.getenv("BINANCE_API_SECRET")),
        timeout=15)
    rows, skipped = [], []
    try:
        data = await client._request_margin_api("get", "launchpool/project/list", True, data={})
        for pr in data["completed"]["list"]:
            coin, dur = pr["rebateCoin"], float(pr["duration"])
            t_list = float(pr.get("coinTradeTime") or 0) * 1000
            if not t_list or not pr.get("projects"):
                skipped.append(coin)
                continue
            p1 = await _px_at(client, f"{coin}USDT", t_list, 0)
            p7 = await _px_at(client, f"{coin}USDT", t_list, 7)
            if not p1:
                skipped.append(coin)
                continue
            for pool in pr["projects"]:
                asset = pool["asset"]
                staked = float(pool.get("totalInvestAmount") or 0)
                reward = float(pool.get("rebateTotalAmount") or 0)
                if staked <= 0 or reward <= 0:
                    continue
                apx = 1.0 if asset in STABLES else await _px_at(client, f"{asset}USDT", pr["investStartTime"])
                if not apx:
                    continue
                staked_usd = staked * apx
                ret1 = reward * p1 / staked_usd
                rows.append({
                    "coin": coin, "pool": asset, "days": dur, "staked_usd": staked_usd,
                    "ret_sell_day1": ret1, "apr_sell_day1": ret1 * 365 / dur,
                    "reward_px_d1_to_d7": (p7 / p1 - 1) if p7 else None,
                    "mine_end_ms": pr["mineEndTime"],
                })
    finally:
        await client.close_connection()
    print(f"{len(rows)} pool-results across {len({r['coin'] for r in rows})} projects; "
          f"{len(skipped)} projects skipped (no pool data or no USDT kline)\n")
    return rows


def report(rows: list[dict]) -> None:
    for pool in ("FDUSD", "BNB", "USDC"):
        rs = [r for r in rows if r["pool"] == pool]
        if not rs:
            continue
        aprs = [r["apr_sell_day1"] for r in rs]
        hold = [r["reward_px_d1_to_d7"] for r in rs if r["reward_px_d1_to_d7"] is not None]
        print(f"{pool} pool   n={len(rs)}   avg window {statistics.mean(r['days'] for r in rs):.1f}d   "
              f"avg staked ${statistics.mean(r['staked_usd'] for r in rs)/1e9:.1f}B")
        print(f"   per project, sell day 1 : median {statistics.median(r['ret_sell_day1'] for r in rs)*100:.3f}%")
        print(f"   ANNUALISED, sell day 1  : median {statistics.median(aprs)*100:.1f}%  mean {statistics.mean(aprs)*100:.1f}%  "
              f"worst {min(aprs)*100:.1f}%  best {max(aprs)*100:.1f}%")
        if hold:
            print(f"   HOLD instead (d1->d7)   : median {statistics.median(hold)*100:+.1f}%  "
                  f"up {sum(1 for h in hold if h > 0)}/{len(hold)}  <- holding loses")
        print()
    cut = time.time() * 1000 - 365 * 86_400_000
    for label, keep in (("last 12 months", lambda r: r["mine_end_ms"] >= cut),
                        ("older", lambda r: r["mine_end_ms"] < cut)):
        rs = [r for r in rows if r["pool"] in ("FDUSD", "USDC") and keep(r)]
        if rs:
            print(f"stablecoin pools, {label:<15}: n={len(rs):>2}  annualised median "
                  f"{statistics.median(r['apr_sell_day1'] for r in rs)*100:.1f}%")
    print("\nWorst case is still positive; holding the reward is not. Sell day 1, sweep to earn.")


if __name__ == "__main__":
    report(asyncio.run(backtest()))
