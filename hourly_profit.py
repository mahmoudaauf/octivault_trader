#!/usr/bin/env python3
"""Hourly compounding profit — the goal, measured against what actually arrives.

THE GOAL
--------
Profit that compounds every hour, tracked hourly, with an explicit dollar
target per hour (HOURLY_TARGET_USD, default $1.00).

WHAT SETTING THIS GOAL DOES AND DOES NOT DO
-------------------------------------------
It makes the objective measurable and puts every hour on the record. It does
not raise the return, and no code can: the money already accrues continuously
(per second, not per hour), and Binance credits it on its own schedule. Moving
a credit from daily to hourly is worth $0.0004 a year on this balance — payment
frequency is a calendar, not a revenue source. What decides the hourly number is
the rate (7.12%, already the maximum this account can get, verified against real
payments) and the balance. So this tool measures the goal honestly rather than
implying the goal has been met.

THE DECOMPOSITION THAT MATTERS
------------------------------
`growth` in the NAV ledger is NAV minus contributions, so it already excludes
deposits. But it still mixes two very different things, and reporting them
together would flatter or panic at random:

    yield  — earn rewards. Real, banked, cannot go negative.
    mark   — price moves on the leftover dust holdings (BNB, INJ, ADA, ...).
             Noise on ~$0.31 of coins, and it can swamp the yield in any given
             hour. It is not profit until sold, and nothing is selling it.

An hour that shows +$0.004 because BNB ticked up is not an hour that earned
anything. So the two are always reported apart, and progress toward the target
is judged on YIELD only.

Mark is computed as a PRICE-ONLY revaluation -- qty_before x (price_now -
price_before), summed over assets held in both snapshots -- not as the change in
holdings_usd. The difference is not academic: holdings_usd also falls when an
asset is SOLD, and selling converts holdings into cash without changing NAV.
Differencing it charged the whole sale to mark and handed the offset to yield,
which reported +$12.05/hr of "yield" across the window containing the BTC sale,
against a true figure near half a cent. Quantity changes must land in neither
column, and with a price-only mark they do not.

MEASUREMENT FLOOR -- READ BEFORE TRUSTING AN HOUR
-------------------------------------------------
NAV is recorded to four decimal places, a $0.0001 quantum, while one hour of
yield on this balance is $0.000493. The rounding is therefore 20% of the signal
and a SINGLE HOUR CANNOT BE MEASURED FROM NAV AT ALL. Only multi-hour averages
carry information, and even those drift: a 72-hour window spanning the BTC sale
and the options-wallet round trip read 17.8%/yr, while a clean 24-hour window
read 6.02% against a true 7.12%. The ground truth for yield is the reward
ledger (`simple-earn/flexible/history/rewardsRecord`), which this tool does not
replace. Treat the hourly column as a progress log, not as an accounting
statement.

Usage:
  python3 hourly_profit.py                # last 24 complete hours
  python3 hourly_profit.py --hours 72
  python3 hourly_profit.py --target 0.01  # a target that is actually in reach
Env: HOURLY_TARGET_USD
"""
from __future__ import annotations

import argparse
import json
import os
from collections import OrderedDict
from datetime import datetime, timezone

NAV_FILE = os.getenv("HYBRID_NAV_FILE", "logs/nav_history.jsonl")
TARGET = float(os.getenv("HOURLY_TARGET_USD", "1.00"))


def _rows(path: str) -> list[dict]:
    out = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue          # a torn last line must not kill the report
                if isinstance(r, dict) and "ts" in r and "growth" in r:
                    out.append(r)
    except FileNotFoundError:
        return []
    out.sort(key=lambda r: r["ts"])
    return out


def _price_move(before: dict, after: dict) -> float:
    """Value change from PRICE alone, holding quantity fixed at its opening level.

    Only assets present in both snapshots count, and only their price is allowed
    to vary. An asset that was bought, sold or swept during the hour contributes
    nothing here, which is correct: moving value between holdings and cash does
    not change NAV, so it must not appear as either mark or yield.
    """
    total = 0.0
    for asset, prev in before.items():
        cur = after.get(asset)
        if not isinstance(prev, dict) or not isinstance(cur, dict):
            continue
        try:
            qty = float(prev.get("qty", 0.0) or 0.0)
            p0 = float(prev.get("price", 0.0) or 0.0)
            p1 = float(cur.get("price", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if qty and p0 and p1:
            total += qty * (p1 - p0)
    return total


def _hourly(rows: list[dict]) -> "OrderedDict[str, dict]":
    """Last observation of each calendar hour, then hour-over-hour deltas.

    The last observation is used rather than the first because it is the hour's
    closing state; differencing closes gives exactly the profit attributable to
    that hour, with no double counting at the boundary.
    """
    closes: "OrderedDict[str, dict]" = OrderedDict()
    for r in rows:
        try:
            t = datetime.fromisoformat(r["ts"]).astimezone(timezone.utc)
        except (ValueError, TypeError):
            continue
        closes[t.strftime("%Y-%m-%d %H")] = r

    keys = list(closes)
    out: "OrderedDict[str, dict]" = OrderedDict()
    for prev, cur in zip(keys, keys[1:]):
        a, b = closes[prev], closes[cur]
        total = b.get("growth", 0.0) - a.get("growth", 0.0)
        mark = _price_move(a.get("holdings") or {}, b.get("holdings") or {})
        out[cur] = {"total": total, "mark": mark, "yield": total - mark,
                    "nav": b.get("nav", 0.0)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=24)
    ap.add_argument("--target", type=float, default=TARGET)
    a = ap.parse_args()

    rows = _rows(NAV_FILE)
    if len(rows) < 2:
        print(f"Not enough NAV history in {NAV_FILE} to measure an hour yet.")
        return 1
    hours = _hourly(rows)
    if not hours:
        print("No complete hour boundaries recorded yet.")
        return 1

    shown = list(hours.items())[-a.hours:]
    print("=" * 68)
    print(f"HOURLY COMPOUNDING PROFIT — goal ${a.target:.4f}/hour")
    print("=" * 68)
    print(f"  {'hour (UTC)':<16}{'yield':>11}{'mark':>11}{'total':>11}{'NAV':>10}")
    for k, v in shown:
        print(f"  {k:<16}{v['yield']:>+11.6f}{v['mark']:>+11.6f}"
              f"{v['total']:>+11.6f}{v['nav']:>10.4f}")

    ys = [v["yield"] for _, v in shown]
    avg = sum(ys) / len(ys)
    nav = shown[-1][1]["nav"]
    print("-" * 68)
    print(f"  hours measured        : {len(ys)}")
    print(f"  yield banked          : ${sum(ys):+.6f}")
    print(f"  average YIELD per hour: ${avg:+.6f}")
    print(f"  mark-to-market noise  : ${sum(v['mark'] for _, v in shown):+.6f} "
          f"(dust price moves, not earned)")
    print()
    if avg > 0:
        pct = avg / nav * 100.0
        print(f"  implied compounding   : {pct:.6f}%/hour  = {((1+avg/nav)**8760-1)*100:.2f}%/yr")
    print(f"  progress to goal      : {avg/a.target*100:.4f}% of ${a.target:.4f}/hour")
    # State the floor next to the number, so no one reads one hour as fact.
    quantum = 0.0001
    print(f"  measurement floor     : NAV rounds to ${quantum:.4f}, "
          f"{quantum/max(avg,1e-9)*100:.0f}% of an hour's yield — "
          f"single hours are noise")
    gap = a.target / avg if avg > 0 else float("inf")
    if gap != float("inf"):
        print(f"  gap                   : {gap:,.0f}x")
        # The rate is already maxed, so the only free variable is the balance.
        print(f"  balance that would pay ${a.target:.4f}/hr at this rate: "
              f"${nav*gap:,.0f}")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
