#!/usr/bin/env python3
"""Track the first N profit-generating round-trips after a baseline epoch.

FIFO-matches BUY/SELL fills per symbol from the trade journal and reports net
P&L after Binance round-trip fees (0.1% per side). Only counts round-trips that
*close* at or after BASELINE_EPOCH so we measure trades made under the
stale-price fix (real fill prices), not earlier history.
"""
from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict, deque

FEE_PER_SIDE = 0.001  # 0.1% Binance taker, each leg
TARGET = 5

# Baseline = bot restart when freshness/fill-price fixes went live (18:42 UTC).
def _arg_epoch(default: float) -> float:
    for a in sys.argv[1:]:
        try:
            return float(a)
        except ValueError:
            continue
    return default


def _pinned_baseline(default: float) -> float:
    """Read the pinned post-fix baseline epoch shared with edge_report.py."""
    try:
        with open("logs/postfix_baseline.epoch") as f:
            return float(f.read().strip())
    except (FileNotFoundError, ValueError):
        return default


BASELINE_EPOCH = _arg_epoch(_pinned_baseline(1781116920.0))


def load_fills() -> list[dict]:
    rows: list[dict] = []
    for path in sorted(glob.glob("logs/trade_journal_*.jsonl")):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    rows.sort(key=lambda r: r.get("epoch", 0.0))
    return rows


def match_round_trips(rows: list[dict]) -> list[dict]:
    """FIFO-match sells against buys per symbol; return completed round-trips."""
    lots: dict[str, deque] = defaultdict(deque)  # symbol -> deque of open buy lots
    trips: list[dict] = []

    for r in rows:
        sym = r.get("symbol", "")
        ev = r.get("event", "")
        qty = float(r.get("qty", 0.0) or 0.0)
        price = float(r.get("price", 0.0) or 0.0)
        epoch = float(r.get("epoch", 0.0) or 0.0)
        if qty <= 0 or price <= 0:
            continue

        if ev == "BUY_FILLED":
            lots[sym].append({"qty": qty, "price": price, "epoch": epoch})
        elif ev == "SELL_FILLED":
            remaining = qty
            cost = 0.0
            matched = 0.0
            entry_epoch = epoch
            while remaining > 1e-12 and lots[sym]:
                lot = lots[sym][0]
                take = min(remaining, lot["qty"])
                cost += take * lot["price"]
                matched += take
                entry_epoch = min(entry_epoch, lot["epoch"])
                lot["qty"] -= take
                remaining -= take
                if lot["qty"] <= 1e-12:
                    lots[sym].popleft()
            if matched <= 0:
                continue
            revenue = matched * price
            fees = (cost + revenue) * FEE_PER_SIDE
            net = revenue - cost - fees
            trips.append({
                "symbol": sym,
                "entry_epoch": entry_epoch,
                "exit_epoch": epoch,
                "qty": matched,
                "buy_avg": cost / matched,
                "sell": price,
                "cost": cost,
                "revenue": revenue,
                "net": net,
                "pct": (net / cost * 100.0) if cost else 0.0,
            })
    return trips


def main() -> None:
    import datetime as dt

    rows = load_fills()
    trips = match_round_trips(rows)
    after = [t for t in trips if t["exit_epoch"] >= BASELINE_EPOCH]

    def ts(e: float) -> str:
        return dt.datetime.utcfromtimestamp(e).strftime("%H:%M:%S")

    profitable = [t for t in after if t["net"] > 0]

    print(f"Baseline: {dt.datetime.utcfromtimestamp(BASELINE_EPOCH)} UTC")
    print(f"Round-trips closed since baseline: {len(after)}  |  profitable: {len(profitable)}\n")

    print("All round-trips since baseline (chrono):")
    for t in after:
        mark = "✅" if t["net"] > 0 else "❌"
        held = (t["exit_epoch"] - t["entry_epoch"]) / 60.0
        print(f"  {mark} {ts(t['exit_epoch'])} {t['symbol']:11} "
              f"buy={t['buy_avg']:.6g} sell={t['sell']:.6g} held={held:4.0f}m "
              f"net={t['net']:+.4f} ({t['pct']:+.2f}%)")

    print(f"\n── First {TARGET} PROFIT-generating trades ──")
    if not profitable:
        print("  (none yet — waiting)")
    for i, t in enumerate(profitable[:TARGET], 1):
        print(f"  #{i} {ts(t['exit_epoch'])} {t['symbol']:11} net=+{t['net']:.4f} ({t['pct']:+.2f}%)")

    got = min(len(profitable), TARGET)
    print(f"\nProgress: {got}/{TARGET} profitable"
          + ("  🎯 TARGET REACHED" if got >= TARGET else f"  ({TARGET-got} to go)"))

    # open positions
    print("\nOpen (unmatched) buys:")
    lots: dict[str, float] = defaultdict(float)
    for t in []:
        pass
    # recompute opens
    open_lots: dict[str, list] = defaultdict(list)
    for r in rows:
        sym, ev = r.get("symbol",""), r.get("event","")
        q, p = float(r.get("qty",0) or 0), float(r.get("price",0) or 0)
        if ev == "BUY_FILLED" and q>0:
            open_lots[sym].append([q,p])
        elif ev == "SELL_FILLED" and q>0:
            rem=q
            while rem>1e-12 and open_lots[sym]:
                if open_lots[sym][0][0]<=rem+1e-12:
                    rem-=open_lots[sym][0][0]; open_lots[sym].pop(0)
                else:
                    open_lots[sym][0][0]-=rem; rem=0
    any_open=False
    for sym,ls in open_lots.items():
        tot=sum(l[0] for l in ls)
        if tot>1e-9:
            any_open=True
            costv=sum(l[0]*l[1] for l in ls)
            print(f"  {sym}: qty={tot:.6g} avg={costv/tot:.6g} (~${costv:.2f})")
    if not any_open:
        print("  (none)")


def watch() -> None:
    """Poll the journal until TARGET profitable round-trips close since baseline."""
    import datetime as dt
    import time

    status_path = "logs/profit_tracker_status.txt"
    seen_exits: set = set()
    profitable_count = 0
    start = time.time()

    def log(msg: str) -> None:
        line = f"{dt.datetime.now().strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        with open(status_path, "a") as f:
            f.write(line + "\n")

    log(f"WATCH START — tracking first {TARGET} profitable trades since "
        f"{dt.datetime.utcfromtimestamp(BASELINE_EPOCH)} UTC")

    while profitable_count < TARGET:
        rows = load_fills()
        trips = [t for t in match_round_trips(rows) if t["exit_epoch"] >= BASELINE_EPOCH]
        for t in trips:
            key = (t["symbol"], round(t["exit_epoch"], 3))
            if key in seen_exits:
                continue
            seen_exits.add(key)
            mark = "✅ PROFIT" if t["net"] > 0 else "❌ loss"
            if t["net"] > 0:
                profitable_count += 1
                log(f"{mark} #{profitable_count}/{TARGET} {t['symbol']} "
                    f"net=+{t['net']:.4f} ({t['pct']:+.2f}%) "
                    f"buy={t['buy_avg']:.6g} sell={t['sell']:.6g}")
            else:
                log(f"{mark} {t['symbol']} net={t['net']:+.4f} ({t['pct']:+.2f}%) (not counted)")
        if profitable_count >= TARGET:
            break
        time.sleep(90)

    log(f"🎯 DONE — first {TARGET} profitable trades captured "
        f"in {(time.time()-start)/60:.0f} min")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "watch":
        # Optional 2nd arg overrides baseline
        if len(sys.argv) > 2:
            BASELINE_EPOCH = float(sys.argv[2])
        watch()
    else:
        main()
