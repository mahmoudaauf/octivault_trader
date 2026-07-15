#!/usr/bin/env python3
"""
Order-book imbalance momentum backtest -- PRELIMINARY (thin data, said upfront).

Hypothesis: extreme order-book imbalance (heavy bid depth vs. heavy ask depth)
predicts short-term price direction. This is the one genuinely untested lever
flagged in the edge-verdict-no-edge memory -- everything else (ML model,
stat-arb, lead-lag, funding-contrarian, cross-exchange, new-listing momentum,
general news sentiment) has already been tested and mostly falsified.

Data: logs/orderbook_history_*.jsonl -- real live snapshots (symbol, bid,
bid_qty, ask, ask_qty, spread_pct, imbalance, ts), persisted since this
session enabled the hook. As of this run: ~22.7 HOURS of history across 25
symbols -- a small fraction of what every other backtest in this project
used (months to years). Treat any result here as preliminary, not a verdict
on the same footing as funding_carry_backtest.py or news_sentiment_backtest.py.
Re-run this again once more data has accumulated before trusting it further.

imbalance = bid_qty / (bid_qty + ask_qty). 1.0 = all bid depth (buy
pressure), 0.0 = all ask depth (sell pressure), 0.5 = balanced.

Usage: python3 orderbook_imbalance_backtest.py
Env:   OB_IMBALANCE_HI (0.85) OB_IMBALANCE_LO (0.15)
       OB_HORIZONS_MIN ("1,5,15,60") OB_COOLDOWN_MIN (5)
       OB_MIN_TRADES (30) OB_COST_PCT (0.12) -- round-trip cost assumption
"""
from __future__ import annotations

import glob
import json
import os

IMBALANCE_HI = float(os.getenv("OB_IMBALANCE_HI", "0.85"))
IMBALANCE_LO = float(os.getenv("OB_IMBALANCE_LO", "0.15"))
HORIZONS_MIN = [float(x) for x in os.getenv("OB_HORIZONS_MIN", "1,5,15,60").split(",")]
COOLDOWN_MIN = float(os.getenv("OB_COOLDOWN_MIN", "5"))
MIN_TRADES = int(os.getenv("OB_MIN_TRADES", "30"))
COST_PCT = float(os.getenv("OB_COST_PCT", "0.12"))


def load_rows() -> dict[str, list[dict]]:
    by_symbol: dict[str, list[dict]] = {}
    for fname in sorted(glob.glob("logs/orderbook_history_*.jsonl")):
        with open(fname) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "symbol" not in r or "ts" not in r:
                    continue
                r["mid"] = (r["bid"] + r["ask"]) / 2.0
                by_symbol.setdefault(r["symbol"], []).append(r)
    for sym in by_symbol:
        by_symbol[sym].sort(key=lambda r: r["ts"])
    return by_symbol


def forward_mid(rows: list[dict], idx: int, horizon_s: float, tolerance_s: float) -> float | None:
    target_ts = rows[idx]["ts"] + horizon_s
    for j in range(idx + 1, len(rows)):
        if rows[j]["ts"] >= target_ts:
            if rows[j]["ts"] <= target_ts + tolerance_s:
                return rows[j]["mid"]
            return None  # nearest snapshot is too far past the target -- a data gap
    return None


def build_events(by_symbol: dict[str, list[dict]]) -> list[dict]:
    events = []
    for sym, rows in by_symbol.items():
        last_signal_ts: dict[int, float] = {}  # direction -> last signal ts, for cooldown
        for i, r in enumerate(rows):
            imb = r.get("imbalance")
            if imb is None:
                continue
            if imb >= IMBALANCE_HI:
                direction = 1
            elif imb <= IMBALANCE_LO:
                direction = -1
            else:
                continue
            if r["ts"] - last_signal_ts.get(direction, -1e18) < COOLDOWN_MIN * 60:
                continue  # still in cooldown -- same "event", don't double count
            last_signal_ts[direction] = r["ts"]

            returns = {}
            for h in HORIZONS_MIN:
                fwd = forward_mid(rows, i, h * 60, tolerance_s=h * 60 * 0.5)
                if fwd is not None and r["mid"] > 0:
                    returns[h] = (fwd - r["mid"]) / r["mid"] * 100.0
            if returns:
                events.append({"symbol": sym, "ts": r["ts"], "direction": direction,
                                "imbalance": imb, "returns": returns})
    return events


def split(events: list[dict]) -> tuple[list[dict], list[dict]]:
    ordered = sorted(events, key=lambda e: e["ts"])
    cut = int(len(ordered) * 0.7)
    return ordered[:cut], ordered[cut:]


def report(events: list[dict], horizon_min: float) -> None:
    print("\n" + "=" * 72)
    print(f"ORDER-BOOK IMBALANCE -> forward return — horizon {horizon_min:.0f}min "
          f"(cost model: {COST_PCT:.2f}% round-trip)")
    print("=" * 72)

    def stats(rows: list[dict]) -> tuple[int, float, float]:
        rets = [r["returns"][horizon_min] * r["direction"] - COST_PCT
                for r in rows if horizon_min in r["returns"]]
        n = len(rets)
        if n == 0:
            return 0, 0.0, 0.0
        wins = sum(1 for x in rets if x > 0)
        return n, sum(rets) / n, wins / n * 100.0

    in_sample, out_sample = split(events)
    n_in, avg_in, wr_in = stats(in_sample)
    n_out, avg_out, wr_out = stats(out_sample)
    n_all, avg_all, wr_all = stats(events)

    print(f"  All:          n={n_all:<5} avg_net_ret={avg_all:+.4f}%  win-rate={wr_all:.0f}%")
    print(f"  In-sample:    n={n_in:<5} avg={avg_in:+.4f}%  win-rate={wr_in:.0f}%  (first 70% chronologically)")
    print(f"  Out-of-sample n={n_out:<5} avg={avg_out:+.4f}%  win-rate={wr_out:.0f}%  (last 30%, holdout)")
    print("-" * 72)
    if n_out < MIN_TRADES:
        print(f"  VERDICT: ⏳ INCONCLUSIVE — only {n_out}/{MIN_TRADES} out-of-sample events. "
              f"PRELIMINARY DATA (~1 day) — treat any number here as noise-prone regardless of sign.")
    elif avg_out > 0 and wr_out > 50:
        print(f"  VERDICT: ~POSSIBLE SIGNAL~ (out-of-sample) — avg {avg_out:+.4f}%, {wr_out:.0f}% win. "
              f"Still PRELIMINARY given ~1 day of data — do NOT treat as validated. Re-run with more history.")
    else:
        print(f"  VERDICT: ❌ NO SIGNAL (out-of-sample) — avg {avg_out:+.4f}%, {wr_out:.0f}% win.")
    print("-" * 72)


def main() -> None:
    by_symbol = load_rows()
    total_rows = sum(len(v) for v in by_symbol.values())
    if not by_symbol:
        print("No order-book history data found in logs/orderbook_history_*.jsonl")
        return
    span_h = (max(r["ts"] for rows in by_symbol.values() for r in rows)
              - min(r["ts"] for rows in by_symbol.values() for r in rows)) / 3600.0
    print(f"Loaded {total_rows} snapshots across {len(by_symbol)} symbols, spanning {span_h:.1f} hours.")
    print(f"⚠️  This is a FRACTION of the data used by every other backtest in this project "
          f"(months to years). Results below are preliminary, not a verdict.\n")
    print(f"Signal: imbalance >= {IMBALANCE_HI} (buy) or <= {IMBALANCE_LO} (sell), "
          f"{COOLDOWN_MIN}min cooldown per symbol/direction.")

    events = build_events(by_symbol)
    print(f"Built {len(events)} signal events after cooldown dedup.")

    for h in HORIZONS_MIN:
        report(events, h)


if __name__ == "__main__":
    main()
