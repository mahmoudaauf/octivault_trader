#!/usr/bin/env python3
"""
Carry liquidity-frontier sweep — the (frequency vs edge) curve.

WHY THIS EXISTS
---------------
The live daemon (carry_paper_trader.py) fires roughly once per 9 days. Two
restrictions the validated backtest never had explain ~26x of that:

  1. A $50M/24h liquidity filter cuts the universe 361 -> 28 (~13x fewer
     opportunities).  [CARRY_MIN_VOL_USD, carry_paper_trader.py:42]
  2. positive_only=true — the backtest's simulate() used abs(fr), counting
     negative-funding entries the daemon structurally cannot execute (~2x).

Whether widening the liquidity filter is safe rests entirely on one claim,
recorded in project memory as prose: "liquid ($50M/24h) + spot-hedgeable +
0.24% cost -> +0.90%/trade, 73% win, 26 trades." A full search of the repo,
git history, docs/ and logs/ found ZERO evidence for it — the original run
printed to stdout and was never kept, and it does not record its entry
threshold. This script replaces that unverifiable number with a logged,
reproducible table.

METHOD
------
Fetches funding history ONCE for the full spot-hedgeable universe (~361
symbols), then evaluates every (liquidity threshold x model) combination
in-process. That's ~361 API calls total rather than ~4,300 for the naive
re-run-per-config approach.

Two models are reported side by side:
  * validated  (positive_only=False, signed_collect=False) — reproduces the
    original +0.9434%/trade methodology; the number to sanity-check against.
  * live-faithful (positive_only=True, signed_collect=True) — what
    carry_paper_trader.py can ACTUALLY execute today.

The gap between those two columns is the honest cost of v1's positive-only
restriction, and has never been measured before.

Usage:  python3 carry_frontier_sweep.py 2>&1 | tee logs/carry_frontier_sweep.log
Env:    FUNDING_ENTRY_BPS (6 — matches the live daemon, NOT the script default of 3)
        CARRY_SWEEP_THRESHOLDS ("0,1,5,10,25,50" — in $M/24h)
        CARRY_ROUND_TRIP_PCT (0.24)
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from funding_carry_backtest import fetch_funding, simulate  # noqa: E402

THRESHOLDS_M = [float(x) for x in os.getenv("CARRY_SWEEP_THRESHOLDS", "0,1,5,10,25,50").split(",")]
ENTRY_BPS = float(os.getenv("FUNDING_ENTRY_BPS", "6"))
ENTRY = ENTRY_BPS / 10000.0

CACHE_PATH = os.getenv("CARRY_SWEEP_CACHE", "data/carry_funding_cache.json")

# Ordered to DECOMPOSE the gap between the validated headline and live reality.
# Reading down the rows at a fixed threshold isolates one variable at a time:
#   1 -> 2 : cost of positive_only alone (entry filter; fewer trades, same accounting)
#   2 -> 3 : cost of committed-direction reality (mid-hold sign flips become
#            real payments instead of phantom income)
#   3 -> 4 : how much of that a sign-flip exit would WIN BACK (a proposed
#            carry_paper_trader.py fix — it currently exits only on
#            abs(fr) < EXIT, so it holds through flips and bleeds)
MODELS = [
    ("1 validated (abs, both signs)", dict(positive_only=False, signed_collect=False)),
    ("2 pos-only, abs collect", dict(positive_only=True, signed_collect=False)),
    ("3 pos-only, signed (= LIVE TODAY)", dict(positive_only=True, signed_collect=True)),
    ("4 pos-only, signed, exit-on-flip", dict(positive_only=True, signed_collect=True, exit_on_flip=True)),
]


def span_days(funding: list[tuple[int, float]]) -> float:
    """Real calendar span of a symbol's funding history, measured from its own
    timestamps.

    Do NOT assume a fixed window. funding_carry_backtest.py hardcodes
    `days_hist = 1000 / 3.0` (=333d) on the assumption that limit=1000 returns
    1000 windows — but Binance's /fapi/v1/fundingRate caps at 500 rows, so a
    full-history symbol actually spans ~166d, and every per-day/per-year figure
    computed off 333 is 2x too low. Newer listings are shorter still (e.g.
    HOMEUSDT ≈ 38d), so a single global divisor is wrong even in principle.
    """
    if len(funding) < 2:
        return 0.0
    return (funding[-1][0] - funding[0][0]) / 86_400_000.0


def _stats(trades: list[float], rate_per_day: float) -> tuple[int, float, float, float]:
    n = len(trades)
    if n == 0:
        return 0, 0.0, 0.0, rate_per_day
    wins = sum(1 for t in trades if t > 0)
    avg = sum(trades) / n
    return n, avg, wins / n * 100.0, rate_per_day


def _load_cache() -> tuple[dict, dict] | None:
    """(funding_by_sym, vol) from a previous run, or None. Funding history is
    the expensive part (~361 REST calls); the analysis on top of it is free, so
    cache it to make model iteration instant."""
    try:
        with open(CACHE_PATH) as f:
            payload = json.load(f)
        fb = {k: [(int(t), float(r)) for t, r in v] for k, v in payload["funding"].items()}
        return fb, {k: float(v) for k, v in payload["vol"].items()}
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _save_cache(funding_by_sym: dict, vol: dict) -> None:
    try:
        os.makedirs(os.path.dirname(CACHE_PATH) or ".", exist_ok=True)
        with open(CACHE_PATH, "w") as f:
            json.dump({"funding": funding_by_sym, "vol": vol}, f)
        print(f"  cached funding -> {CACHE_PATH} (re-runs are instant; delete to refresh)")
    except OSError as e:
        print(f"  cache write failed (non-fatal): {str(e)[:80]}")


async def _fetch_all() -> tuple[dict, dict]:
    from dotenv import load_dotenv

    load_dotenv()
    from binance import AsyncClient

    client = await AsyncClient.create(
        os.getenv("BINANCE_API_KEY") or "x", os.getenv("BINANCE_API_SECRET") or "x"
    )
    try:
        # ── Universe: USDT perps that also have a spot market (hedgeable) ──
        info = await client.futures_exchange_info()
        perps = [
            s["symbol"] for s in info["symbols"]
            if s.get("quoteAsset") == "USDT" and s.get("contractType") == "PERPETUAL"
            and s.get("status") == "TRADING"
        ]
        spot = await client.get_exchange_info()
        spot_syms = {
            s["symbol"] for s in spot["symbols"]
            if s.get("status") == "TRADING" and s.get("quoteAsset") == "USDT"
        }
        universe = [s for s in perps if s in spot_syms]
        print(f"Universe: {len(perps)} USDT perps -> {len(universe)} spot-hedgeable")

        # ── Volume per symbol (one call) ──
        tick = await client.futures_ticker()
        vol = {t["symbol"]: float(t.get("quoteVolume", 0) or 0) for t in tick}

        print(f"\nFetching funding history for {len(universe)} symbols (once; reused for all configs)...")
        funding_by_sym: dict[str, list[tuple[int, float]]] = {}
        for i, sym in enumerate(universe, 1):
            f = await fetch_funding(client, sym)
            if len(f) >= 50:
                funding_by_sym[sym] = f
            if i % 50 == 0:
                print(f"  ...{i}/{len(universe)} fetched, {len(funding_by_sym)} usable")
            await asyncio.sleep(0.12)  # polite pacing
        _save_cache(funding_by_sym, vol)
        return funding_by_sym, vol
    finally:
        await client.close_connection()


async def main() -> None:
    print("=" * 84)
    print("CARRY LIQUIDITY-FRONTIER SWEEP")
    print("=" * 84)
    print(f"entry>={ENTRY_BPS:.0f}bps (={ENTRY*100:.3f}%/8h, matches live daemon)")

    cached = _load_cache()
    if cached:
        funding_by_sym, vol = cached
        print(f"Using cached funding for {len(funding_by_sym)} symbols ({CACHE_PATH}).")
    else:
        funding_by_sym, vol = await _fetch_all()

    spans = sorted(span_days(f) for f in funding_by_sym.values())
    if spans:
        _med = spans[len(spans) // 2]
        print(f"  real history span: median={_med:.0f}d  min={spans[0]:.0f}d  max={spans[-1]:.0f}d")
        print(f"  (note: Binance caps /fapi/v1/fundingRate at 500 rows -> ~166d max, "
              f"NOT the 333d funding_carry_backtest.py assumes)\n")

    # ── Sweep ──
    print("=" * 84)
    print(f"{'min_vol':>8} {'symbols':>8} | {'model':<36} {'trades':>7} {'trd/day':>8} "
          f"{'avg%/trade':>11} {'win%':>6}")
    print("-" * 84)

    results = []
    for thr_m in THRESHOLDS_M:
        thr = thr_m * 1e6
        subset = [s for s in funding_by_sym if vol.get(s, 0.0) >= thr]
        for label, kw in MODELS:
            trades: list[float] = []
            # Market-wide rate = sum of each symbol's OWN rate (trades / its own
            # span). Symbols have different history lengths, so a single global
            # divisor would be wrong; watching N symbols simultaneously gives the
            # sum of their individual per-day rates.
            rate_per_day = 0.0
            for sym in subset:
                f = funding_by_sym[sym]
                t = simulate(f, entry=ENTRY, **kw)
                trades.extend(t)
                sd = span_days(f)
                if sd > 0:
                    rate_per_day += len(t) / sd
            n, avg, wr, per_day = _stats(trades, rate_per_day)
            results.append((thr_m, len(subset), label, n, per_day, avg, wr))
            print(f"{thr_m:>7.0f}M {len(subset):>8} | {label:<36} {n:>7} {per_day:>8.3f} "
                  f"{avg:>+11.4f} {wr:>5.0f}%")
        print("-" * 78)

    # ── Verdict ──
    print("\n" + "=" * 78)
    print("READING THIS TABLE")
    print("=" * 84)
    print("* 'trd/day' is MARKET-WIDE across that subset — the live daemon's realised")
    print("  rate is further limited by MAX_POS=5 and its $10/leg sizing.")
    print("* Row 3 at $50M is WHAT THE DAEMON RUNS TODAY. Row 1 is the number this")
    print("  project has been treating as its validated edge.")
    print("* Read DOWN the rows at one threshold to decompose the gap:")
    print("    1->2  cost of positive_only (entry filter)")
    print("    2->3  cost of committed-direction reality (mid-hold sign flips are")
    print("          real payments; row 1/2's abs() counted them as income)")
    print("    3->4  how much a sign-flip exit would win back (a PROPOSED daemon fix)")
    print("* A threshold is only worth widening to if avg%/trade stays comfortably")
    print("  above the round-trip cost AND win-rate holds up — remember thinner names")
    print("  carry real slippage this sim does NOT model (it assumes clean fills).")

    def _pick(thr, prefix):
        m = [r for r in results if r[0] == thr and r[2].startswith(prefix)]
        return m[0] if m else None

    print("\n" + "=" * 78)
    print("DECOMPOSITION (at the widest universe, $0M — most data)")
    print("=" * 84)
    for prefix in ("1 ", "2 ", "3 ", "4 "):
        r = _pick(0.0, prefix)
        if r:
            print(f"  {r[2]:<36} n={r[3]:<5} {r[4]:>6.3f} trd/day  {r[5]:+.4f}%/trade  {r[6]:.0f}% win")

    live = _pick(50.0, "3 ")
    fixed = _pick(50.0, "4 ")
    if live and fixed:
        print("\n" + "=" * 78)
        print("AT THE DAEMON'S ACTUAL CONFIG ($50M filter)")
        print("=" * 84)
        print(f"  today          : n={live[3]:<4} {live[4]:.3f} trd/day  {live[5]:+.4f}%/trade  {live[6]:.0f}% win")
        print(f"  with flip-exit : n={fixed[3]:<4} {fixed[4]:.3f} trd/day  {fixed[5]:+.4f}%/trade  {fixed[6]:.0f}% win")


if __name__ == "__main__":
    asyncio.run(main())
