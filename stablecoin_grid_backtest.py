#!/usr/bin/env python3
"""Stablecoin grid backtest — the one spot strategy where fees are actually zero.

WHY THIS IS WORTH TESTING WHEN EVERYTHING ELSE FAILED
-----------------------------------------------------
Every spot strategy in this repo died the same death: the edge was smaller than
the cost of acting on it. Spread market-making lost 0.59% per fill across 701
real fills; order-book imbalance lost almost exactly the 0.12% round-trip on
every horizon tested. Fees and adverse selection ate everything.

USDCUSDT, USD1USDT, USD1USDC, FDUSDUSDT and TUSDUSDT are ZERO FEE on this
account — verified against /sapi/v1/asset/tradeFee, maker AND taker at 0.0000%.
That removes the exact thing that killed the others, and it removes directional
risk too: both legs are dollars, so a grid that never fills simply holds
dollars rather than bleeding.

So the question is narrow and fair: does the pair oscillate enough to pay for
the SPREAD, which is now the only cost?

HOW IT IS TESTED
----------------
A symmetric grid around the running mean. Buy a rung when price crosses below
it, sell that inventory when price crosses back above the rung above. Every
buy pays the ASK and every sell receives the BID — crossing the spread on both
sides, which is the honest way to model a taker grid and the assumption most
grid backtests quietly skip.

Chronological 70/30 in-sample / holdout, because a grid has parameters (rung
spacing) and tuning them on all the data is how a flat line becomes a
backtest that "works".

Usage:  python3 stablecoin_grid_backtest.py [SYMBOL] [DAYS]
"""
from __future__ import annotations

import json
import statistics
import sys
import urllib.parse
import urllib.request

BASE = "https://api.binance.com"


def klines(symbol: str, interval: str = "1m", days: int = 30) -> list[tuple]:
    """(open_time, high, low, close) at 1m, paged back `days`."""
    out, end = [], None
    need = days * 24 * 60
    while len(out) < need:
        p = {"symbol": symbol, "interval": interval, "limit": 1000}
        if end:
            p["endTime"] = end
        try:
            rows = json.loads(urllib.request.urlopen(
                f"{BASE}/api/v3/klines?{urllib.parse.urlencode(p)}", timeout=25).read())
        except Exception:
            break
        if not rows:
            break
        out = [(r[0], float(r[2]), float(r[3]), float(r[4])) for r in rows] + out
        end = rows[0][0] - 1
        if len(rows) < 1000:
            break
    return out


def spread_pct(symbol: str) -> float:
    """Live top-of-book spread — the only cost left once fees are zero."""
    try:
        t = json.loads(urllib.request.urlopen(
            f"{BASE}/api/v3/ticker/bookTicker?symbol={symbol}", timeout=15).read())
        b, a = float(t["bidPrice"]), float(t["askPrice"])
        return (a - b) / b
    except Exception:
        return 0.0001


def run_grid(bars: list[tuple], rung_pct: float, spread: float,
             capital: float = 60.0, rungs: int = 5) -> dict:
    """Symmetric grid around a rolling mean. Buys pay the ask, sells hit the bid.

    Inventory is capped at `rungs` units so a one-way drift cannot silently
    become an unbounded position — the failure mode that makes naive grid
    backtests look profitable right up until they blow up.
    """
    if len(bars) < 240:
        return {"trades": 0, "net": 0.0, "note": "insufficient data"}
    unit = capital / rungs
    held: list[float] = []          # entry prices of open rungs
    realised, trades = 0.0, 0
    window = [b[3] for b in bars[:120]]
    half = spread / 2.0

    for _, hi, lo, close in bars[120:]:
        mean = statistics.fmean(window)
        window.append(close)
        if len(window) > 120:
            window.pop(0)
        # Buy rung: the low of the bar reached a level below the mean.
        buy_at = mean * (1 - rung_pct)
        if lo <= buy_at and len(held) < rungs:
            held.append(buy_at * (1 + half))        # pay the ask
            trades += 1
        # Sell rung: the high reached a level above the mean, close oldest lot.
        sell_at = mean * (1 + rung_pct)
        if hi >= sell_at and held:
            entry = held.pop(0)
            exit_px = sell_at * (1 - half)          # receive the bid
            realised += (exit_px - entry) / entry * unit
            trades += 1

    # Mark any open inventory at the last close, so a "profit" that is really
    # an unsold bag cannot hide.
    last = bars[-1][3]
    unreal = sum((last * (1 - half) - e) / e * unit for e in held)
    return {"trades": trades, "net": realised + unreal,
            "realised": realised, "open": len(held), "unrealised": unreal}


def main() -> int:
    symbol = (sys.argv[1:] or ["USDCUSDT"])[0]
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    bars = klines(symbol, days=days)
    if len(bars) < 500:
        print(f"Not enough data for {symbol} ({len(bars)} bars).")
        return 1
    sp = spread_pct(symbol)
    closes = [b[3] for b in bars]
    rng = (max(closes) - min(closes)) / statistics.fmean(closes)

    print("=" * 70)
    print(f"STABLECOIN GRID — {symbol}, {len(bars):,} 1m bars ({len(bars)/1440:.1f} days)")
    print("=" * 70)
    print(f"  price range   {min(closes):.5f} – {max(closes):.5f}  ({rng*100:.3f}% of mean)")
    print(f"  live spread   {sp*100:.4f}%   ← the only cost; fees are 0.0000%")
    print(f"  a round trip must clear {sp*100:.4f}% just to break even\n")

    cut = int(len(bars) * 0.7)
    ins, oos = bars[:cut], bars[cut:]
    print(f"  {'rung':>8}{'IS trades':>11}{'IS net':>11}{'OOS trades':>12}{'OOS net':>11}")
    best = None
    # Rungs must be scaled to the pair's actual range. USDCUSDT moves 0.148%
    # in a MONTH, so a 0.2% rung never fires — the first pass tested rungs
    # wider than the entire price range and reported a flat zero, which is
    # "untested", not "no edge".
    for rung in (0.00005, 0.0001, 0.00015, 0.0002, 0.0003, 0.0005):
        a = run_grid(ins, rung, sp)
        b = run_grid(oos, rung, sp)
        print(f"  {rung*100:>7.2f}%{a['trades']:>11,}{a['net']:>+11.4f}"
              f"{b['trades']:>12,}{b['net']:>+11.4f}")
        if best is None or a["net"] > best[1]["net"]:
            best = (rung, a, b)

    rung, a, b = best
    print(f"\n  Best IN-SAMPLE rung {rung*100:.2f}% → ${a['net']:+.4f}")
    print(f"  Same rung OUT-OF-SAMPLE            → ${b['net']:+.4f} "
          f"over {b['trades']:,} fills, {b['open']} lots left open")
    verdict = "EDGE" if b["net"] > 0 and b["trades"] > 20 else "NO EDGE"
    print(f"\n  VERDICT (holdout): {verdict}")
    if b["trades"] > 0:
        print(f"  per fill: ${b['net']/b['trades']:+.6f} on ${60/5:.0f} units")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# ── VERDICT, 2026-09-13 ───────────────────────────────────────────────────────
#
# Two pairs showed POSITIVE out-of-sample — USD1USDT +$0.0256 and FDUSDUSDT
# +$0.0382 over a 9.2-day holdout. That is the first spot strategy in this
# repo to survive a holdout at all, and it still LOSES, for a reason the
# backtest above cannot see:
#
#   A grid needs its capital sitting in SPOT, quoting. Spot earns 0.00%.
#   The same capital in Simple Earn earns 8.59%.
#
#       best grid, taken at face value   +$1.52/yr
#       doing nothing (Simple Earn)      +$5.18/yr
#       running the grid COSTS           -$3.66/yr
#
# The opportunity cost of leaving yield is roughly 3x the entire edge. Any
# strategy that parks capital in spot on this account starts 8.59% behind, and
# nothing found here clears that hurdle.
#
# Three further reasons not to trust the positive numbers, in case a future
# session is tempted:
#   1. 50-fill holdout samples. The market-making study needed 701 fills to
#      resolve, and resolved NEGATIVE.
#   2. The fill model assumes "the bar's low touched the level, therefore we
#      bought there". On 1m bars that is generous; a real limit order at a
#      three-tick-wide zero-fee pair sits behind every market maker on Binance.
#   3. The first run tested rungs WIDER than the pair's entire monthly range
#      (USDCUSDT moves 0.148% in a month) and printed a flat zero. That is
#      "untested", not "no edge" — rungs must be scaled to the pair's range, or
#      the backtest reports nothing and looks like a result.
#
# Keep this file. It is the honest test of the last zero-fee corner of spot,
# and its answer is that the hurdle is not fees — it is the 8.59% you give up
# to play.
