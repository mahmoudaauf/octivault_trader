#!/usr/bin/env python3
"""
Volatility-breakout momentum backtest.

Hypothesis: a price breakout beyond its recent N-bar range, confirmed by
above-average volume, predicts continued directional momentum. This is a
genuinely different methodology from everything else tested in this
project -- not the GRU/OHLCV-feature ML model (backtest_edge.py, proven no
edge), not pairs mean-reversion (statarb_discover.py, proven no edge), not
a cash-flow strategy (funding_carry_backtest.py). It's a classic rules-based
Donchian-channel breakout + volume-confirmation system, tested here with the
same real-out-of-sample discipline as every other backtest in this project.

Universe: top-N USDT spot pairs by 24h quote volume (liquid only -- illiquid
breakouts are the least tradeable kind). History: real klines, multi-year,
via python-binance's paginating get_historical_klines (same call
news_sentiment_backtest.py uses).

Signal: close breaks above the highest high (long) or below the lowest low
(short) of the last BREAKOUT_LOOKBACK bars, AND that bar's volume is >=
VOLUME_MULT x the average volume of the same lookback window. A cooldown
after each signal avoids counting a single breakout's continuation bars as
many independent "events."

Usage: python3 volatility_breakout_backtest.py
Env:   VB_UNIVERSE_SIZE (40) VB_LOOKBACK_DAYS (730) VB_INTERVAL (1h)
       VB_BREAKOUT_LOOKBACK (20) VB_VOLUME_MULT (1.5)
       VB_HORIZONS_H ("4,24,72") VB_COOLDOWN_H (72)
       VB_MIN_TRADES (30) VB_COST_PCT (0.24)
"""
from __future__ import annotations

import asyncio
import os
import sys

UNIVERSE_SIZE = int(os.getenv("VB_UNIVERSE_SIZE", "40"))
LOOKBACK_DAYS = int(os.getenv("VB_LOOKBACK_DAYS", "730"))
INTERVAL = os.getenv("VB_INTERVAL", "1h")
BREAKOUT_LOOKBACK = int(os.getenv("VB_BREAKOUT_LOOKBACK", "20"))
VOLUME_MULT = float(os.getenv("VB_VOLUME_MULT", "1.5"))
HORIZONS_H = [float(x) for x in os.getenv("VB_HORIZONS_H", "4,24,72").split(",")]
COOLDOWN_H = float(os.getenv("VB_COOLDOWN_H", "72"))
MIN_TRADES = int(os.getenv("VB_MIN_TRADES", "30"))
COST_PCT = float(os.getenv("VB_COST_PCT", "0.24"))

_INTERVAL_HOURS = {"5m": 5 / 60, "15m": 0.25, "1h": 1.0, "4h": 4.0, "1d": 24.0}


async def fetch_klines(client, symbol: str) -> list[list] | None:
    try:
        return await client.get_historical_klines(
            symbol, INTERVAL, start_str=f"{LOOKBACK_DAYS} days ago UTC"
        )
    except Exception as e:
        print(f"  {symbol}: fetch failed ({str(e)[:80]})")
        return None


def find_signals(symbol: str, klines: list[list]) -> list[dict]:
    """Returns [{symbol, ts, direction, entry, returns:{h: pct}}]."""
    bar_h = _INTERVAL_HOURS[INTERVAL]
    n = BREAKOUT_LOOKBACK
    cooldown_bars = max(1, int(COOLDOWN_H / bar_h))
    max_h_bars = max(1, int(max(HORIZONS_H) / bar_h))

    closes = [float(k[4]) for k in klines]
    highs = [float(k[2]) for k in klines]
    lows = [float(k[3]) for k in klines]
    vols = [float(k[5]) for k in klines]
    times = [int(k[0]) for k in klines]

    signals = []
    last_signal_idx = -10**9
    for i in range(n, len(klines) - 1):
        if i - last_signal_idx < cooldown_bars:
            continue
        window_high = max(highs[i - n:i])
        window_low = min(lows[i - n:i])
        avg_vol = sum(vols[i - n:i]) / n
        if avg_vol <= 0:
            continue

        direction = 0
        if closes[i] > window_high and vols[i] >= VOLUME_MULT * avg_vol:
            direction = 1
        elif closes[i] < window_low and vols[i] >= VOLUME_MULT * avg_vol:
            direction = -1
        if direction == 0:
            continue

        entry = closes[i]
        if entry <= 0 or i + 1 >= len(klines):
            continue
        returns = {}
        for h in HORIZONS_H:
            fwd_idx = i + max(1, int(round(h / bar_h)))
            if fwd_idx >= len(klines):
                continue
            fwd = closes[fwd_idx]
            returns[h] = (fwd - entry) / entry * 100.0
        if returns:
            signals.append({"symbol": symbol, "ts": times[i], "direction": direction, "returns": returns})
            last_signal_idx = i
    return signals


def split(signals: list[dict]) -> tuple[list[dict], list[dict]]:
    ordered = sorted(signals, key=lambda s: s["ts"])
    cut = int(len(ordered) * 0.7)
    return ordered[:cut], ordered[cut:]


def report(signals: list[dict], horizon_h: float) -> None:
    print("\n" + "=" * 72)
    print(f"VOLATILITY BREAKOUT -> forward return — horizon {horizon_h:.0f}h "
          f"(cost model: {COST_PCT:.2f}% round-trip)")
    print("=" * 72)
    if not signals:
        print("  No signals.")
        return

    def stats(rows: list[dict]) -> tuple[int, float, float]:
        rets = [r["returns"][horizon_h] * r["direction"] - COST_PCT
                for r in rows if horizon_h in r["returns"]]
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
    try:
        print("Selecting universe: top USDT pairs by 24h quote volume...")
        info = await client.get_exchange_info()
        usdt_symbols = {
            s["symbol"] for s in info["symbols"]
            if s.get("quoteAsset") == "USDT" and s.get("status") == "TRADING"
            and not any(x in s["symbol"] for x in ("UP", "DOWN", "BEAR", "BULL"))  # skip leveraged tokens
        }
        tickers = await client.get_ticker()
        vol_by_symbol = {t["symbol"]: float(t.get("quoteVolume", 0) or 0) for t in tickers}
        universe = sorted(
            (s for s in usdt_symbols if s in vol_by_symbol),
            key=lambda s: -vol_by_symbol[s],
        )[:UNIVERSE_SIZE]
        print(f"  Universe: {len(universe)} symbols "
              f"(top by volume, min ${vol_by_symbol.get(universe[-1], 0):,.0f}/24h)")

        print(f"\nFetching {LOOKBACK_DAYS}d of {INTERVAL} klines per symbol "
              f"(breakout_lookback={BREAKOUT_LOOKBACK} bars, volume_mult={VOLUME_MULT}x, "
              f"cooldown={COOLDOWN_H:.0f}h)...")
        all_signals = []
        for i, sym in enumerate(universe, 1):
            klines = await fetch_klines(client, sym)
            if not klines or len(klines) < BREAKOUT_LOOKBACK * 2:
                continue
            sigs = find_signals(sym, klines)
            all_signals.extend(sigs)
            if i % 10 == 0:
                print(f"  ...{i}/{len(universe)} symbols processed, {len(all_signals)} signals so far")
            await asyncio.sleep(0.2)  # light courtesy pacing on Binance REST
    finally:
        await client.close_connection()

    print(f"\nTotal signals: {len(all_signals)} across {len(universe)} symbols, {LOOKBACK_DAYS}d history each")
    for h in HORIZONS_H:
        report(all_signals, h)


if __name__ == "__main__":
    asyncio.run(main())
