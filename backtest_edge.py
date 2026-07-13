#!/usr/bin/env python3
"""
Standalone edge backtest.

Reuses MLForecaster.backtest_confidence_buckets() — the SAME model + feature +
fee-adjusted-outcome pipeline the live bot uses — to replay the trained models over
historical klines and answer the only question that matters:

    Does higher model confidence produce higher NET (after-fee) P&L?

If yes (win-rate / net-pnl rises with confidence), the model discriminates → edge.
If not, the confidence is noise → no edge, and no execution/gate tweak can fix it.

Usage:  python3 backtest_edge.py [max_samples_per_symbol]
"""
from __future__ import annotations

import asyncio
import glob
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone


async def main() -> None:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from dotenv import load_dotenv

    load_dotenv()  # same env (API keys, etc.) the live bot uses
    from agents.ml_forecaster import MLForecaster
    from core_engine.native.bootstrap import BootstrapConfig, _default_exchange_factory
    from core_engine.native.market_data import NativeMarketData
    from core_engine.native.model_manager import ModelManager
    from core_engine.native.shared_state import NativeSharedState

    max_per_sym = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    tf = os.getenv("BACKTEST_TIMEFRAME", "5m")

    cfg = BootstrapConfig.from_env()
    exchange_client = _default_exchange_factory(cfg)
    shared_state = NativeSharedState()
    model_manager = ModelManager(config=cfg)
    market_data = NativeMarketData(exchange_client, symbols=[], shared_state=shared_state)

    # Every symbol with a trained model.
    syms = sorted(
        os.path.basename(f).split("_")[1]
        for f in glob.glob(f"models/mlforecaster_*_{tf}.keras")
        if ".backup" not in f
    )
    print(f"Backtesting {len(syms)} models, up to {max_per_sym} samples/symbol...")

    fc = MLForecaster(
        shared_state=shared_state,
        execution_manager=None,
        config=cfg,
        symbols=syms,
        timeframe=tf,
        market_data_feed=market_data,
        exchange_client=exchange_client,
        model_manager=model_manager,
    )

    try:
        res = await fc.backtest_confidence_buckets(
            symbols=syms,
            max_samples_per_symbol=max_per_sym,
            source="edge_proof",
            include_records=True,
        )
    finally:
        close = getattr(exchange_client, "close_connection", None) or getattr(
            exchange_client, "close", None
        )
        if callable(close):
            await close()

    buckets = res.get("rows") or []
    total_n = int(res.get("records", 0) or 0)
    if not buckets:
        print("No buckets. Result:", {k: v for k, v in res.items() if k != "rows"})
        return

    def _f(rec, *keys):
        for k in keys:
            if k in rec and rec[k] is not None:
                return float(rec[k])
        return 0.0

    # Each bucket: mean_conf, samples, wins, win_rate, avg_net_pnl_pct (a fraction).
    bk = []
    for r in buckets:
        bk.append(
            {
                "conf": _f(r, "mean_conf"),
                "n": int(_f(r, "samples")),
                "wins": int(_f(r, "wins")),
                "net": _f(r, "avg_net_pnl_pct") * 100.0,  # → percent
            }
        )
    bk.sort(key=lambda x: x["conf"])

    tot_n = sum(b["n"] for b in bk) or 1
    tot_wins = sum(b["wins"] for b in bk)
    overall_wr = tot_wins / tot_n
    overall_exp = sum(b["net"] * b["n"] for b in bk) / tot_n

    print("\n" + "=" * 64)
    print(f"EDGE BACKTEST — {total_n} samples across {len(syms)} symbols")
    print("=" * 64)
    print(f"Overall: net expectancy = {overall_exp:+.4f}%/trade   win-rate = {overall_wr*100:.1f}%")
    print("\nNet P&L by confidence bucket (THE discrimination test):")
    print(f"  {'mean_conf':>9} {'n':>6} {'win%':>6} {'net_pnl%':>10}")
    for b in bk:
        wr = (b["wins"] / b["n"] * 100) if b["n"] else 0
        print(f"  {b['conf']:>9.3f} {b['n']:>6} {wr:>5.0f}% {b['net']:>+9.4f}")

    # Discrimination: sample-weighted net P&L of the low-confidence half vs high half.
    half = tot_n / 2.0
    cum = 0
    low, high = [], []
    for b in bk:
        (low if cum < half else high).append(b)
        cum += b["n"]
    q1_exp = sum(b["net"] * b["n"] for b in low) / max(1, sum(b["n"] for b in low))
    q4_exp = sum(b["net"] * b["n"] for b in high) / max(1, sum(b["n"] for b in high))
    discriminates = q4_exp > q1_exp
    print(f"\n  low-conf half: {q1_exp:+.4f}%   high-conf half: {q4_exp:+.4f}%")

    samples = res.get("samples") or []
    by_symbol = defaultdict(list)
    by_regime = defaultdict(list)
    daily_wins = defaultdict(int)
    sample_days = []
    for sample in samples:
        net_pct = float(sample.get("net_pnl_pct", 0.0) or 0.0) * 100.0
        by_symbol[str(sample.get("symbol", "UNKNOWN"))].append(net_pct)
        by_regime[str(sample.get("regime", "unknown"))].append(net_pct)
        ts = float(sample.get("entry_ts", 0.0) or 0.0)
        if ts > 0:
            day = datetime.fromtimestamp(ts, tz=timezone.utc).date()
            sample_days.append(day)
            if net_pct > 0:
                daily_wins[day] += 1

    if by_symbol:
        ranked = []
        for symbol, pnl_values in by_symbol.items():
            wins = sum(value > 0 for value in pnl_values)
            ranked.append(
                (sum(pnl_values) / len(pnl_values), wins / len(pnl_values), len(pnl_values), symbol)
            )
        ranked.sort(reverse=True)
        print("\nPer-symbol net edge (minimum 20 samples):")
        eligible = [row for row in ranked if row[2] >= 20]
        for expectancy, win_rate, count, symbol in eligible[:10]:
            print(
                f"  {symbol:12} n={count:>4} win={win_rate*100:>5.1f}% "
                f"net={expectancy:>+8.4f}%"
            )
        positive = [row for row in eligible if row[0] > 0]
        print(f"  positive symbols: {len(positive)}/{len(eligible)} with >=20 samples")

    if by_regime:
        print("\nPer-regime net edge:")
        for regime, pnl_values in sorted(by_regime.items()):
            wins = sum(value > 0 for value in pnl_values)
            print(
                f"  {regime:12} n={len(pnl_values):>4} "
                f"win={wins/len(pnl_values)*100:>5.1f}% "
                f"net={sum(pnl_values)/len(pnl_values):>+8.4f}%"
            )

    if sample_days:
        start_day, end_day = min(sample_days), max(sample_days)
        calendar_days = (end_day - start_day).days + 1
        raw_winners_per_day = sum(daily_wins.values()) / calendar_days
        print(
            "\nRaw opportunity diagnostic (overlapping samples; NOT executable portfolio proof):"
        )
        print(
            f"  {start_day}..{end_day} ({calendar_days} UTC days): "
            f"{sum(daily_wins.values())} net-positive samples = "
            f"{raw_winners_per_day:.2f}/day"
        )
    print("\n" + "-" * 64)
    if overall_exp > 0 and discriminates:
        print(f"VERDICT: ✅ EDGE SIGNAL — net+ ({overall_exp:+.4f}%) and high-conf "
              f"({q4_exp:+.4f}%) beats low-conf ({q1_exp:+.4f}%). Forward-confirm in paper.")
    elif discriminates:
        print(f"VERDICT: ⚠️ DISCRIMINATES but net-negative — high-conf ({q4_exp:+.4f}%) "
              f"beats low-conf ({q1_exp:+.4f}%) but overall {overall_exp:+.4f}%. Costs eat the edge.")
    else:
        print(f"VERDICT: ❌ NO EDGE — confidence does not predict net P&L "
              f"(high-conf {q4_exp:+.4f}% vs low-conf {q1_exp:+.4f}%, overall {overall_exp:+.4f}%).")
    print("-" * 64)


if __name__ == "__main__":
    asyncio.run(main())
