#!/usr/bin/env python3
"""
Stablecoin depeg reversion backtest.

Hypothesis: when a USD-pegged stablecoin trades away from $1 on Binance by
more than a threshold, it reverts back toward $1 over the following
minutes/hours -- a mechanical mispricing (arbitrageurs/redemption mechanisms
pull it back to par), not a price PREDICTION. Same "structural mechanism,
not a guess" family as funding-carry and delisting-exit -- the two
strategies that have actually shown real edge in this project.

Universe: every USD-pegged stablecoin Binance lists against USDT (TUSD,
USDC, USDP, FDUSD, XUSD, USD1, USDE, FRAX, RLUSD -- EUR-pegged EURUSDT/
AEURUSDT excluded, wrong peg target). Real klines, as much history as each
symbol has (several are recent listings).

Signal: |price - 1.0| >= DEPEG_THRESHOLD_PCT. direction = +1 (expect reversion
UP) if price < 1 - threshold, direction = -1 (expect reversion DOWN) if price
> 1 + threshold. Cooldown avoids counting one depeg event's many bars as
independent "events."

Usage: python3 stablecoin_depeg_backtest.py
Env:   SD_INTERVAL (5m) SD_LOOKBACK_DAYS (730)
       SD_DEPEG_THRESHOLD_PCT ("0.15,0.30,0.50") -- tested independently
       SD_HORIZONS_MIN ("15,60,240,1440") SD_COOLDOWN_MIN (60)
       SD_MIN_TRADES (30) SD_COST_PCT (0.10) -- tighter than other scripts;
       a near-arb strategy should be executed with maker orders/at size,
       not a retail-taker assumption, but 0.10% is still conservative-ish.
"""
from __future__ import annotations

import asyncio
import os
import sys

INTERVAL = os.getenv("SD_INTERVAL", "5m")
LOOKBACK_DAYS = int(os.getenv("SD_LOOKBACK_DAYS", "730"))
THRESHOLDS_PCT = [float(x) for x in os.getenv("SD_DEPEG_THRESHOLD_PCT", "0.15,0.30,0.50").split(",")]
HORIZONS_MIN = [float(x) for x in os.getenv("SD_HORIZONS_MIN", "15,60,240,1440").split(",")]
COOLDOWN_MIN = float(os.getenv("SD_COOLDOWN_MIN", "60"))
MIN_TRADES = int(os.getenv("SD_MIN_TRADES", "30"))
COST_PCT = float(os.getenv("SD_COST_PCT", "0.10"))

STABLE_SYMBOLS = [
    "TUSDUSDT", "USDCUSDT", "USDPUSDT", "FDUSDUSDT",
    "XUSDUSDT", "USD1USDT", "USDEUSDT", "FRAXUSDT", "RLUSDUSDT",
]

_INTERVAL_MIN = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240}


async def fetch_klines(client, symbol: str) -> list[list] | None:
    try:
        return await client.get_historical_klines(
            symbol, INTERVAL, start_str=f"{LOOKBACK_DAYS} days ago UTC"
        )
    except Exception as e:
        print(f"  {symbol}: fetch failed ({str(e)[:80]})")
        return None


def find_signals(symbol: str, klines: list[list], threshold_pct: float) -> list[dict]:
    bar_min = _INTERVAL_MIN[INTERVAL]
    cooldown_bars = max(1, int(COOLDOWN_MIN / bar_min))
    threshold = threshold_pct / 100.0

    closes = [float(k[4]) for k in klines]
    times = [int(k[0]) for k in klines]

    signals = []
    last_signal_idx = -10**9
    for i in range(len(klines) - 1):
        if i - last_signal_idx < cooldown_bars:
            continue
        price = closes[i]
        if price <= 0:
            continue
        dev = price - 1.0
        if dev <= -threshold:
            direction = 1   # below peg -> expect reversion UP
        elif dev >= threshold:
            direction = -1  # above peg -> expect reversion DOWN
        else:
            continue

        returns = {}
        for h in HORIZONS_MIN:
            fwd_idx = i + max(1, int(round(h / bar_min)))
            if fwd_idx >= len(klines):
                continue
            fwd = closes[fwd_idx]
            returns[h] = (fwd - price) / price * 100.0
        if returns:
            signals.append({
                "symbol": symbol, "ts": times[i], "direction": direction,
                "entry_price": price, "returns": returns,
            })
            last_signal_idx = i
    return signals


def split(signals: list[dict]) -> tuple[list[dict], list[dict]]:
    ordered = sorted(signals, key=lambda s: s["ts"])
    cut = int(len(ordered) * 0.7)
    return ordered[:cut], ordered[cut:]


def report(signals: list[dict], threshold_pct: float, horizon_min: float) -> None:
    print("\n" + "=" * 72)
    print(f"STABLECOIN DEPEG (threshold {threshold_pct:.2f}%) -> forward return "
          f"— horizon {horizon_min:.0f}min (cost model: {COST_PCT:.2f}% round-trip)")
    print("=" * 72)
    if not signals:
        print("  No signals.")
        return

    def stats(rows: list[dict]) -> tuple[int, float, float]:
        rets = [r["returns"][horizon_min] * r["direction"] - COST_PCT
                for r in rows if horizon_min in r["returns"]]
        n = len(rets)
        if n == 0:
            return 0, 0.0, 0.0
        wins = sum(1 for x in rets if x > 0)
        return n, sum(rets) / n, wins / n * 100.0

    in_sample, out_sample = split(signals)
    n_in, avg_in, wr_in = stats(in_sample)
    n_out, avg_out, wr_out = stats(out_sample)
    n_all, avg_all, wr_all = stats(signals)

    print(f"  All:          n={n_all:<5} avg_net_ret={avg_all:+.4f}%  win-rate={wr_all:.0f}%")
    print(f"  In-sample:    n={n_in:<5} avg={avg_in:+.4f}%  win-rate={wr_in:.0f}%  (first 70% chronologically)")
    print(f"  Out-of-sample n={n_out:<5} avg={avg_out:+.4f}%  win-rate={wr_out:.0f}%  (last 30%, holdout)")
    print("-" * 72)
    if n_out < MIN_TRADES:
        print(f"  VERDICT: ⏳ INCONCLUSIVE — only {n_out}/{MIN_TRADES} out-of-sample signals.")
    elif avg_out > 0 and wr_out > 50:
        print(f"  VERDICT: ✅ EDGE CANDIDATE (out-of-sample) — avg {avg_out:+.4f}%, {wr_out:.0f}% win.")
    else:
        print(f"  VERDICT: ❌ NO EDGE (out-of-sample) — avg {avg_out:+.4f}%, {wr_out:.0f}% win.")
    print("-" * 72)


async def main() -> None:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from dotenv import load_dotenv

    load_dotenv()
    from binance import AsyncClient

    client = await AsyncClient.create(
        os.getenv("BINANCE_API_KEY") or "x", os.getenv("BINANCE_API_SECRET") or "x"
    )
    all_klines: dict[str, list[list]] = {}
    try:
        info = await client.get_exchange_info()
        listed = {s["symbol"] for s in info["symbols"] if s.get("status") == "TRADING"}
        universe = [s for s in STABLE_SYMBOLS if s in listed]
        print(f"Universe: {universe} ({len(universe)}/{len(STABLE_SYMBOLS)} currently trading)\n")

        for sym in universe:
            klines = await fetch_klines(client, sym)
            if klines:
                all_klines[sym] = klines
                span_days = (int(klines[-1][0]) - int(klines[0][0])) / 86_400_000
                print(f"  {sym}: {len(klines)} bars, {span_days:.0f}d span")
            await asyncio.sleep(0.2)
    finally:
        await client.close_connection()

    for threshold_pct in THRESHOLDS_PCT:
        all_signals = []
        for sym, klines in all_klines.items():
            all_signals.extend(find_signals(sym, klines, threshold_pct))
        print(f"\n{'#' * 72}\nTHRESHOLD {threshold_pct:.2f}% — {len(all_signals)} total signals "
              f"across {len(all_klines)} symbols\n{'#' * 72}")
        for h in HORIZONS_MIN:
            report(all_signals, threshold_pct, h)


if __name__ == "__main__":
    asyncio.run(main())
