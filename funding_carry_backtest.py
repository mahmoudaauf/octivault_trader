#!/usr/bin/env python3
"""
Funding-rate carry backtest (delta-neutral).

Unlike directional prediction, this needs NO forecast — funding rates are a
historical FACT, so we can compute exactly what a delta-neutral carry would have
earned, fee-accurate, with zero look-ahead.

Strategy modelled:
  - Perp funding is paid every 8h between longs/shorts.
  - When |funding| >= ENTRY_THRESHOLD, enter delta-neutral (long spot + short perp
    if funding>0, the reverse if <0). Price moves cancel → we just collect funding.
  - Each held 8h window collects |funding| on the notional.
  - Exit when |funding| < EXIT_THRESHOLD (normalised) or after MAX_WINDOWS.
  - Net trade P&L = sum(|funding| collected) - ROUND_TRIP_COST (4 fills: 2 legs x in+out).

Criteria (fixed in advance — don't move the goalposts):
  - Net-positive total after fees
  - Positive average net PER TRADE
  - >= MIN_TRADES sample

Usage:  python3 funding_carry_backtest.py
Env:    FUNDING_ENTRY_BPS (default 3 = 0.03%/8h), FUNDING_EXIT_BPS (1),
        CARRY_ROUND_TRIP_PCT (0.24 = 0.24% all-in), CARRY_MAX_WINDOWS (45)
"""
from __future__ import annotations

import asyncio
import os
import sys

PERPS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT",
    "AVAXUSDT", "LINKUSDT", "MATICUSDT", "ARBUSDT", "APTUSDT", "UNIUSDT", "LTCUSDT",
    "NEARUSDT", "SUIUSDT", "OPUSDT", "INJUSDT", "TIAUSDT", "SEIUSDT",
]

ENTRY = float(os.getenv("FUNDING_ENTRY_BPS", "3")) / 10000.0   # |funding|/8h to enter
EXIT = float(os.getenv("FUNDING_EXIT_BPS", "1")) / 10000.0     # |funding|/8h to exit
RT_COST = float(os.getenv("CARRY_ROUND_TRIP_PCT", "0.24")) / 100.0  # all-in 4-fill cost
MAX_WINDOWS = int(os.getenv("CARRY_MAX_WINDOWS", "45"))        # ~15 days max hold
MIN_TRADES = 30


async def fetch_funding(client, symbol: str) -> list[tuple[int, float]]:
    """Historical funding: list of (fundingTime_ms, fundingRate). ~1000 windows ≈ 333d."""
    out = []
    try:
        rows = await client.futures_funding_rate(symbol=symbol, limit=1000)
        for r in rows:
            out.append((int(r["fundingTime"]), float(r["fundingRate"])))
    except Exception as e:
        print(f"  {symbol}: funding fetch failed ({str(e)[:50]})")
    return out


def simulate(funding: list[tuple[int, float]]) -> list[float]:
    """Walk funding windows; return list of net %-returns per completed carry trade."""
    trades = []
    i, n = 0, len(funding)
    while i < n:
        _, fr = funding[i]
        if abs(fr) < ENTRY:
            i += 1
            continue
        # Enter; collect |funding| each window until it normalises or max hold.
        collected = 0.0
        held = 0
        j = i
        while j < n and held < MAX_WINDOWS:
            _, frj = funding[j]
            if held > 0 and abs(frj) < EXIT:
                break
            collected += abs(frj)
            held += 1
            j += 1
        net = collected - RT_COST
        trades.append(net * 100.0)  # percent
        i = j  # resume after the trade
    return trades


async def main() -> None:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from dotenv import load_dotenv

    load_dotenv()
    from binance import AsyncClient

    key = os.getenv("BINANCE_API_KEY") or "x"
    sec = os.getenv("BINANCE_API_SECRET") or "x"
    client = await AsyncClient.create(key, sec)

    universe = PERPS
    if str(os.getenv("CARRY_ALL_PERPS", "false")).lower() in ("1", "true", "yes"):
        try:
            info = await client.futures_exchange_info()
            universe = [
                s["symbol"] for s in info["symbols"]
                if s.get("quoteAsset") == "USDT" and s.get("contractType") == "PERPETUAL"
                and s.get("status") == "TRADING"
            ]
            print(f"Universe: ALL {len(universe)} USDT perps")
            # Delta-neutral REQUIRES a spot leg to hedge. Without spot you'd be
            # directionally exposed — so the realistically-tradeable carry universe is
            # only perps that ALSO have a Binance spot market.
            if str(os.getenv("CARRY_REQUIRE_SPOT", "false")).lower() in ("1", "true", "yes"):
                spot = await client.get_exchange_info()
                spot_syms = {
                    s["symbol"] for s in spot["symbols"]
                    if s.get("status") == "TRADING" and s.get("quoteAsset") == "USDT"
                }
                universe = [s for s in universe if s in spot_syms]
                print(f"Filtered to {len(universe)} perps that ALSO have a spot market "
                      f"(delta-neutral actually possible)")
            # Liquidity filter — you can only run carry at size on perps with real
            # depth. The headline winners are tiny illiquid names where slippage would
            # eat the edge; restrict to names with >= CARRY_MIN_VOLUME_USD/24h.
            min_vol = float(os.getenv("CARRY_MIN_VOLUME_USD", "0") or 0)
            if min_vol > 0:
                tick = await client.futures_ticker()
                vol = {t["symbol"]: float(t.get("quoteVolume", 0) or 0) for t in tick}
                universe = [s for s in universe if vol.get(s, 0) >= min_vol]
                print(f"Filtered to {len(universe)} LIQUID perps "
                      f"(>=${min_vol/1e6:.0f}M/24h — tradeable at size)")
        except Exception as e:
            print(f"perp-list fetch failed ({str(e)[:50]}); using default {len(PERPS)}")

    print(f"Funding-carry backtest — entry>={ENTRY*100:.3f}%/8h  exit<{EXIT*100:.3f}%  "
          f"cost={RT_COST*100:.2f}%  max_hold={MAX_WINDOWS}w (~{MAX_WINDOWS/3:.0f}d)")
    all_trades: list[float] = []
    per_sym = {}
    try:
        for sym in universe:
            fund = await fetch_funding(client, sym)
            if len(fund) < 50:
                continue
            t = simulate(fund)
            per_sym[sym] = t
            all_trades.extend(t)
    finally:
        await client.close_connection()

    if not all_trades:
        print("No trades / no data.")
        return

    n = len(all_trades)
    wins = [x for x in all_trades if x > 0]
    total = sum(all_trades)
    avg = total / n
    wr = len(wins) / n
    # Rough annualisation: avg hold ~ (collected windows); approximate by trades/yr.
    # Each symbol has ~1000 windows = ~333 days of history.
    days_hist = 1000 / 3.0
    syms = len([s for s, t in per_sym.items() if t])
    trades_per_yr = (n / syms) / (days_hist / 365.0) if syms else 0

    print("\n" + "=" * 64)
    print(f"FUNDING CARRY BACKTEST — {n} trades across {syms} perps (~{days_hist:.0f}d each)")
    print("=" * 64)
    print(f"  Net total:          {total:+.2f}%   (sum of all trades' net carry)")
    print(f"  Avg net/trade:      {avg:+.4f}%   ← the number that matters")
    print(f"  Win rate:           {wr*100:.0f}%")
    print(f"  ~Trades/yr/symbol:  {trades_per_yr:.0f}   → est. ~{avg*trades_per_yr:+.1f}%/yr/symbol")
    print("\n  Per-symbol net carry (top/bottom):")
    ranked = sorted(((s, sum(t), len(t)) for s, t in per_sym.items() if t), key=lambda x: -x[1])
    for s, tot, cnt in ranked[:6]:
        print(f"    {s:10} {cnt:>3} trades  net {tot:+.2f}%")
    if len(ranked) > 6:
        for s, tot, cnt in ranked[-3:]:
            print(f"    {s:10} {cnt:>3} trades  net {tot:+.2f}%")

    print("\n" + "-" * 64)
    if n < MIN_TRADES:
        print(f"VERDICT: ⏳ INCONCLUSIVE — {n}/{MIN_TRADES} trades.")
    elif avg > 0 and total > 0:
        print(f"VERDICT: ✅ POSITIVE CARRY — avg {avg:+.4f}%/trade, {wr*100:.0f}% win. "
              f"Edge candidate — validate execution + basis risk next.")
    else:
        print(f"VERDICT: ❌ NO NET CARRY — avg {avg:+.4f}%/trade after {RT_COST*100:.2f}% costs. "
              f"Funding doesn't clear costs at these thresholds.")
    print("-" * 64)


if __name__ == "__main__":
    asyncio.run(main())
