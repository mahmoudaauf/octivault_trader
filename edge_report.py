#!/usr/bin/env python3
"""
Proof-of-edge measurement layer.

Turns "I hope it works" into "here is whether it works." Reads the trade journal
and decision logs, FIFO-matches round-trips, attributes each to its entry regime /
confidence band / exit reason, and reports expectancy after fees with a PASS/FAIL
readout against criteria defined IN ADVANCE (below) so we don't fool ourselves.

Usage:
    python3 edge_report.py                 # full history
    python3 edge_report.py 1781116920      # only round-trips closing at/after epoch
"""
from __future__ import annotations

import ast
import datetime as dt
import glob
import json
import re
import statistics
import sys
from collections import defaultdict, deque

# ── Success criteria — SET IN ADVANCE, do not move the goalposts ───────────────
FEE_PER_SIDE = 0.001          # 0.1% Binance taker per leg (round-trip 0.2%)
MIN_SAMPLE = 50               # need this many closed round-trips to judge
PASS_EXPECTANCY = 0.0         # net expectancy/trade after fees must exceed this
QUALITY_WIN_PCT = 0.005       # a "quality" win nets >= 0.5% after fees (clears costs+)
_BASELINE_FILE = "logs/postfix_baseline.epoch"


def _default_baseline() -> float:
    """Post-fix baseline epoch (pinned when all fixes went live). 0 = full history."""
    try:
        with open(_BASELINE_FILE) as f:
            return float(f.read().strip())
    except (FileNotFoundError, ValueError):
        return 0.0


# arg overrides; "all" forces full history; default = pinned post-fix baseline
if len(sys.argv) > 1 and sys.argv[1].replace(".", "").isdigit():
    BASELINE = float(sys.argv[1])
elif len(sys.argv) > 1 and sys.argv[1].lower() == "all":
    BASELINE = 0.0
else:
    BASELINE = _default_baseline()


def load_journal() -> list[dict]:
    rows: list[dict] = []
    for path in sorted(glob.glob("logs/trade_journal_*.jsonl")):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    rows.sort(key=lambda r: r.get("epoch", 0.0))
    return rows


def load_decisions() -> list[dict]:
    """Parse QUANT_LOOP_SUMMARY dicts (carry regime, confidence, edge per executed BUY)."""
    out: list[dict] = []
    for path in sorted(glob.glob("logs/run_latest.log")) + sorted(glob.glob("logs/bot_*.log")):
        try:
            with open(path, errors="ignore") as f:
                for line in f:
                    i = line.find("QUANT_LOOP_SUMMARY ")
                    if i == -1:
                        continue
                    try:
                        d = ast.literal_eval(line[i + len("QUANT_LOOP_SUMMARY "):].strip())
                        if isinstance(d, dict) and d.get("symbol"):
                            out.append(d)
                    except (ValueError, SyntaxError):
                        pass
        except FileNotFoundError:
            pass
    out.sort(key=lambda d: float(d.get("timestamp", 0.0) or 0.0))
    return out


def load_exit_reasons() -> list[tuple]:
    """Scan logs for TP/SL/time exit markers → (epoch_approx, symbol, reason). Best-effort."""
    reasons: list[tuple] = []
    pat = re.compile(r"\[main:TPSL\] ([A-Z0-9]+USDT) trigger=([A-Z_]+)")
    # log timestamps are local 'YYYY-MM-DD HH:MM:SS,mmm'
    tspat = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
    for path in sorted(glob.glob("logs/run_latest.log")) + sorted(glob.glob("logs/bot_*.log")):
        try:
            with open(path, errors="ignore") as f:
                for line in f:
                    m = pat.search(line)
                    if not m:
                        continue
                    tm = tspat.match(line)
                    ep = 0.0
                    if tm:
                        try:
                            ep = dt.datetime.strptime(tm.group(1), "%Y-%m-%d %H:%M:%S").timestamp()
                        except ValueError:
                            ep = 0.0
                    reasons.append((ep, m.group(1), m.group(2)))
        except FileNotFoundError:
            pass
    return reasons


def attr_regime_conf(decisions: list[dict], symbol: str, entry_epoch: float) -> tuple:
    """Nearest executed BUY decision for symbol at/just-before entry → (regime, confidence)."""
    best = None
    for d in decisions:
        if d.get("symbol") != symbol or str(d.get("action", "")).upper() != "BUY":
            continue
        ts = float(d.get("timestamp", 0.0) or 0.0)
        if ts <= entry_epoch + 5:  # allow tiny clock skew
            if best is None or ts > best[0]:
                best = (ts, str(d.get("market_regime", "?")), float(d.get("confidence", 0.0) or 0.0))
    if best:
        return best[1], best[2]
    return "?", 0.0


def attr_exit(reasons: list[tuple], symbol: str, exit_epoch: float) -> str:
    best, bestdt = "signal/other", 1e9
    for ep, sym, rs in reasons:
        if sym != symbol:
            continue
        delta = abs(ep - exit_epoch)
        if delta < bestdt and delta < 120:  # within 2 min of the fill
            bestdt, best = delta, rs
    return best


def match_round_trips(rows: list[dict]) -> list[dict]:
    lots: dict[str, deque] = defaultdict(deque)
    trips: list[dict] = []
    for r in rows:
        sym, ev = r.get("symbol", ""), r.get("event", "")
        qty = float(r.get("qty", 0.0) or 0.0)
        price = float(r.get("price", 0.0) or 0.0)
        epoch = float(r.get("epoch", 0.0) or 0.0)
        if qty <= 0 or price <= 0:
            continue
        if ev == "BUY_FILLED":
            lots[sym].append({"qty": qty, "price": price, "epoch": epoch})
        elif ev == "SELL_FILLED":
            remaining, cost, matched, entry_epoch = qty, 0.0, 0.0, epoch
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
            net = revenue - cost - (revenue + cost) * FEE_PER_SIDE
            trips.append({
                "symbol": sym, "entry_epoch": entry_epoch, "exit_epoch": epoch,
                "cost": cost, "net": net, "pct": (net / cost) if cost else 0.0,
                "hold_min": (epoch - entry_epoch) / 60.0,
            })
    return trips


def expectancy(trips: list[dict]) -> dict:
    if not trips:
        return {"n": 0}
    nets = [t["net"] for t in trips]
    wins = [n for n in nets if n > 0]
    losses = [n for n in nets if n <= 0]
    return {
        "n": len(trips),
        "total": sum(nets),
        "expectancy": statistics.mean(nets),
        "win_rate": len(wins) / len(trips),
        "avg_win": statistics.mean(wins) if wins else 0.0,
        "avg_loss": statistics.mean(losses) if losses else 0.0,
        "quality_wins": sum(1 for t in trips if t["pct"] >= QUALITY_WIN_PCT),
        "best_streak": _best_streak(nets),
    }


def _best_streak(nets: list[float]) -> int:
    best = cur = 0
    for n in nets:
        cur = cur + 1 if n > 0 else 0
        best = max(best, cur)
    return best


def _bd(title: str, groups: dict):
    print(f"\n── {title} ──")
    print(f"  {'group':<14} {'n':>3} {'win%':>5} {'expR$':>8} {'total$':>9}")

    def _tot(k):
        return sum(t["net"] for t in groups[k])

    for k in sorted(groups, key=_tot):
        e = expectancy(groups[k])
        if e["n"] == 0:
            continue
        print(f"  {str(k):<14} {e['n']:>3} {e['win_rate']*100:>4.0f}% "
              f"{e['expectancy']:>+8.4f} {e['total']:>+9.4f}")


def main():
    rows = load_journal()
    decisions = load_decisions()
    reasons = load_exit_reasons()
    trips = [t for t in match_round_trips(rows) if t["exit_epoch"] >= BASELINE]

    for t in trips:
        t["regime"], t["entry_conf"] = attr_regime_conf(decisions, t["symbol"], t["entry_epoch"])
        t["exit_reason"] = attr_exit(reasons, t["symbol"], t["exit_epoch"])
        t["conf_band"] = (f"{int(t['entry_conf']*10)/10:.1f}-{int(t['entry_conf']*10)/10+0.1:.1f}"
                          if t["entry_conf"] > 0 else "unknown")

    e = expectancy(trips)
    print("=" * 60)
    print("PROOF-OF-EDGE REPORT")
    print("=" * 60)
    _win = ("FULL HISTORY (pre+post fix)" if BASELINE <= 0
            else f"POST-FIX only (since {dt.datetime.utcfromtimestamp(BASELINE):%Y-%m-%d %H:%M} UTC)")
    print(f"Window: {_win}")
    print(f"Criteria (fixed in advance): sample>={MIN_SAMPLE}, expectancy>{PASS_EXPECTANCY:+.4f}/trade "
          f"after {FEE_PER_SIDE*200:.1f}% round-trip fees")
    if e["n"] == 0:
        print("\nNo closed round-trips yet. Nothing to judge.")
        return

    print(f"\nClosed round-trips: {e['n']}")
    print(f"  Net total:        {e['total']:+.4f} USDT")
    print(f"  Expectancy/trade: {e['expectancy']:+.4f} USDT   ← the number that matters")
    print(f"  Win rate:         {e['win_rate']*100:.0f}%   (avg win {e['avg_win']:+.4f}, avg loss {e['avg_loss']:+.4f})")
    print(f"  Quality wins (≥{QUALITY_WIN_PCT*100:.1f}%): {e['quality_wins']}/{e['n']}")
    print(f"  Best win streak:  {e['best_streak']}")

    # payoff ratio
    if e["avg_loss"] != 0:
        pr = abs(e["avg_win"] / e["avg_loss"])
        be = 1.0 / (1.0 + pr)  # break-even win rate for this payoff
        print(f"  Payoff ratio:     {pr:.2f}  → break-even win rate {be*100:.0f}% "
              f"(you have {e['win_rate']*100:.0f}%)")

    by_regime = defaultdict(list); by_sym = defaultdict(list)
    by_exit = defaultdict(list); by_conf = defaultdict(list)
    for t in trips:
        by_regime[t["regime"]].append(t)
        by_sym[t["symbol"]].append(t)
        by_exit[t["exit_reason"]].append(t)
        by_conf[t["conf_band"]].append(t)
    _bd("By entry regime", by_regime)
    _bd("By exit reason", by_exit)
    _bd("By entry confidence band", by_conf)
    _bd("By symbol", by_sym)

    print("\n" + "=" * 60)
    if e["n"] < MIN_SAMPLE:
        print(f"VERDICT: ⏳ INCONCLUSIVE — {e['n']}/{MIN_SAMPLE} trades. Keep running, do not scale.")
    elif e["expectancy"] > PASS_EXPECTANCY:
        print(f"VERDICT: ✅ EDGE PRESENT — expectancy {e['expectancy']:+.4f}/trade over {e['n']} trades.")
        print("         Next: optimize cost (maker fees), trade only winning regimes, then scale slowly.")
    else:
        print(f"VERDICT: ❌ NO EDGE — expectancy {e['expectancy']:+.4f}/trade over {e['n']} trades.")
        print("         The thesis (not the code) needs to change. Do not add capital.")
    print("=" * 60)


if __name__ == "__main__":
    main()
