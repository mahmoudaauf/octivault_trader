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

# ── Model-fidelity flags — BOTH default OFF ────────────────────────────────────
# Default-off keeps the default path byte-for-byte identical to the methodology
# that produced this project's recorded headline (361 spot-hedgeable perps, 944
# trades, +0.9434%/trade, 60% win, at FUNDING_ENTRY_BPS=6). Turning either flag
# ON makes the sim MORE faithful to what carry_paper_trader.py can actually do
# live — and both corrections are expected to reduce the headline number.
#
# CARRY_SIM_POSITIVE_ONLY: enter only on POSITIVE funding, matching the live
#   daemon's POSITIVE_ONLY=true v1 restriction (short perp + long spot; negative
#   funding would need spot-margin shorting, which v1 does not implement). The
#   default abs(fr) >= ENTRY counts negative-funding entries the daemon
#   STRUCTURALLY CANNOT EXECUTE, so the default headline is not a valid
#   predictor of live positive-only performance.
SIM_POSITIVE_ONLY = os.getenv("CARRY_SIM_POSITIVE_ONLY", "false").lower() in ("1", "true", "yes")
# CARRY_SIM_SIGNED_COLLECT: collect DIRECTION-ADJUSTED funding per window rather
#   than abs(). A real position commits to a direction at entry: a short perp
#   receives funding while fr > 0 but PAYS when fr flips negative mid-hold.
#   abs() models "always receive, whatever the sign" — optimistic. Signed
#   collection prices a mid-hold sign flip as the real cost it is.
SIM_SIGNED_COLLECT = os.getenv("CARRY_SIM_SIGNED_COLLECT", "false").lower() in ("1", "true", "yes")


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


def simulate(
    funding: list[tuple[int, float]],
    *,
    positive_only: bool | None = None,
    signed_collect: bool | None = None,
    exit_on_flip: bool = False,
    entry: float | None = None,
) -> list[float]:
    """Walk funding windows; return list of net %-returns per completed carry trade.

    Behavior defaults to the SIM_POSITIVE_ONLY / SIM_SIGNED_COLLECT / ENTRY
    module globals (both flags default OFF -> identical to the original
    validated methodology). See their definitions above for why each default is
    optimistic relative to live. The keyword args exist so a caller
    (carry_frontier_sweep.py) can vary the model in-process across many configs
    without re-importing or re-fetching funding data.
    """
    positive_only = SIM_POSITIVE_ONLY if positive_only is None else positive_only
    signed_collect = SIM_SIGNED_COLLECT if signed_collect is None else signed_collect
    entry = ENTRY if entry is None else entry

    trades = []
    i, n = 0, len(funding)
    while i < n:
        _, fr = funding[i]
        entered = (fr >= entry) if positive_only else (abs(fr) >= entry)
        if not entered:
            i += 1
            continue
        # A real position commits to a direction at entry: +1 = short perp
        # (receives funding while fr > 0), -1 = long perp (receives while
        # fr < 0). Under SIM_SIGNED_COLLECT the per-window income is
        # direction * frj, so a mid-hold sign flip correctly becomes a cost.
        direction = 1.0 if fr > 0 else -1.0
        collected = 0.0
        held = 0
        j = i
        while j < n and held < MAX_WINDOWS:
            _, frj = funding[j]
            if held > 0 and abs(frj) < EXIT:
                break
            # exit_on_flip models a PROPOSED daemon fix, not current behavior.
            # carry_paper_trader.py exits only on `abs(fr) < EXIT`, so when
            # funding flips against the committed direction the position stays
            # open and BLEEDS until the rate decays back toward zero. Exiting the
            # moment funding turns against us stops that bleed.
            if held > 0 and exit_on_flip and (direction * frj) <= 0:
                break
            collected += (direction * frj) if signed_collect else abs(frj)
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
    # Make the methodology self-documenting in the output. The prior headline
    # number was recorded in prose without its flags, which made it
    # unverifiable after the fact -- don't repeat that.
    _model = (
        f"positive_only={SIM_POSITIVE_ONLY} signed_collect={SIM_SIGNED_COLLECT}"
    )
    print(f"Model fidelity — {_model}"
          f"{'  [DEFAULT: matches the original validated methodology]' if not (SIM_POSITIVE_ONLY or SIM_SIGNED_COLLECT) else '  [live-faithful corrections ON]'}")
    all_trades: list[float] = []
    per_sym = {}
    per_sym_span_days: dict[str, float] = {}
    try:
        for sym in universe:
            fund = await fetch_funding(client, sym)
            if len(fund) < 50:
                continue
            t = simulate(fund)
            per_sym[sym] = t
            # Measure each symbol's REAL history span from its own timestamps
            # rather than assuming a fixed window (see the annualisation note
            # below for why the old fixed 333d assumption was wrong).
            per_sym_span_days[sym] = (fund[-1][0] - fund[0][0]) / 86_400_000.0
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
    # Annualisation, measured — NOT assumed.
    # This previously hardcoded `days_hist = 1000 / 3.0` (=333d) on the premise
    # that limit=1000 returns 1000 funding windows. It does not: Binance's
    # /fapi/v1/fundingRate caps at 500 rows, so a full-history symbol spans only
    # ~166d, and newer listings far less (e.g. HOMEUSDT ≈ 38d). Every
    # trades/yr and %/yr figure this script printed under that assumption was
    # therefore ~2x too low. Use each symbol's real measured span instead.
    _spans = [d for s, d in per_sym_span_days.items() if per_sym.get(s) and d > 0]
    days_hist = (sum(_spans) / len(_spans)) if _spans else 0.0  # mean real span
    syms = len([s for s, t in per_sym.items() if t])
    # Per-symbol trade rate summed over symbols, each against its OWN span.
    trades_per_yr = (
        sum(len(t) / per_sym_span_days[s] for s, t in per_sym.items()
            if t and per_sym_span_days.get(s, 0) > 0) / syms * 365.0
    ) if syms else 0

    print("\n" + "=" * 64)
    print(f"FUNDING CARRY BACKTEST — {n} trades across {syms} perps "
          f"(~{days_hist:.0f}d avg real span each)")
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
