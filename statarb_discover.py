#!/usr/bin/env python3
"""
Statistical-arbitrage discovery + OUT-OF-SAMPLE falsification.

Edge source: mean-reversion of the spread between cointegrated assets — STRUCTURE,
not price prediction. The fatal trap in pairs trading is overfitting: you can ALWAYS
find pairs that "worked" in-sample. So we SELECT pairs on the first half of history
(correlation + spread mean-reversion / half-life) and TEST the strategy ONLY on the
held-out second half — using the in-sample hedge ratio and z-score stats (no
look-ahead). Net-of-fee, criteria fixed in advance.

Usage: python3 statarb_discover.py
"""
from __future__ import annotations

import asyncio
import itertools
import os
import sys

import numpy as np

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT",
    "AVAXUSDT", "LINKUSDT", "MATICUSDT", "DOTUSDT", "LTCUSDT", "BCHUSDT", "ATOMUSDT",
    "UNIUSDT", "APTUSDT", "ARBUSDT", "OPUSDT", "INJUSDT", "NEARUSDT", "FILUSDT",
    "AAVEUSDT", "ETCUSDT", "XLMUSDT", "ALGOUSDT", "ICPUSDT", "SUIUSDT", "SEIUSDT",
    "RUNEUSDT", "GALAUSDT",
]
INTERVAL = os.getenv("SA_INTERVAL", "1h")
BARS = int(os.getenv("SA_BARS", "1000"))
ENTRY_Z = float(os.getenv("SA_ENTRY_Z", "2.0"))
EXIT_Z = float(os.getenv("SA_EXIT_Z", "0.5"))
STOP_Z = float(os.getenv("SA_STOP_Z", "4.0"))
FEE_RT = float(os.getenv("SA_FEE_RT_PCT", "0.4")) / 100.0  # 4 legs round-trip
CORR_MIN = float(os.getenv("SA_CORR_MIN", "0.8"))
MAX_HALFLIFE = float(os.getenv("SA_MAX_HALFLIFE", "40"))   # bars
MIN_TRADES = 30


async def fetch(client, sym):
    try:
        kl = await client.get_klines(symbol=sym, interval=INTERVAL, limit=BARS)
        return np.array([float(k[4]) for k in kl], dtype=np.float64)
    except Exception:
        return None


def half_life(spread: np.ndarray) -> float:
    """OU mean-reversion half-life (bars) via AR(1): ds = b*s_lag. b<0 = reverting."""
    s = spread[:-1]
    ds = np.diff(spread)
    if len(s) < 20 or np.std(s) == 0:
        return 1e9
    b = np.polyfit(s, ds, 1)[0]
    if b >= 0:
        return 1e9
    return float(-np.log(2) / b)


ZWINDOW = int(os.getenv("SA_ZWINDOW", "48"))  # trailing window for rolling z-score


def backtest_pair(spread_full: np.ndarray, split: int) -> list[float]:
    """Trade the spread with a TRAILING rolling z-score (standard pairs approach,
    adapts to drift, no look-ahead). Pairs were selected in-sample; we trade only the
    OOS portion (t >= split). Returns net %-returns per completed trade."""
    n = len(spread_full)
    w = ZWINDOW
    # Trailing rolling mean/std (uses only past data → no look-ahead).
    z = np.zeros(n)
    for t in range(n):
        lo = max(0, t - w)
        seg = spread_full[lo:t]
        if len(seg) >= 10 and np.std(seg) > 0:
            z[t] = (spread_full[t] - np.mean(seg)) / np.std(seg)
    spread_oos = spread_full
    trades, pos, entry = [], 0, 0.0
    for t in range(split, n):
        if pos == 0:
            if z[t] >= ENTRY_Z:
                pos, entry = -1, spread_oos[t]   # short spread (expect fall)
            elif z[t] <= -ENTRY_Z:
                pos, entry = 1, spread_oos[t]     # long spread (expect rise)
        else:
            if abs(z[t]) <= EXIT_Z or abs(z[t]) >= STOP_Z or t == n - 1:
                pnl = (entry - spread_oos[t]) if pos == -1 else (spread_oos[t] - entry)
                trades.append(pnl * 100.0 - FEE_RT * 100.0)
                pos = 0
    return trades


async def main() -> None:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from dotenv import load_dotenv

    load_dotenv()
    from binance import AsyncClient

    client = await AsyncClient.create(
        os.getenv("BINANCE_API_KEY") or "x", os.getenv("BINANCE_API_SECRET") or "x"
    )

    print(f"Stat-arb discovery — {len(SYMBOLS)} symbols, {INTERVAL}x{BARS} bars, "
          f"entry|z|>={ENTRY_Z} exit<{EXIT_Z} fee_rt={FEE_RT*100:.2f}%")
    closes = {}
    try:
        for s in SYMBOLS:
            c = await fetch(client, s)
            if c is not None and len(c) >= BARS * 0.9:
                closes[s] = c
    finally:
        await client.close_connection()

    # Align lengths.
    L = min(len(c) for c in closes.values())
    logp = {s: np.log(c[-L:]) for s, c in closes.items()}
    split = L // 2  # first half = select, second half = test (out-of-sample)
    print(f"Loaded {len(logp)} symbols, {L} bars each → in-sample {split}, OOS {L-split}")

    selected, all_trades = [], []
    for a, b in itertools.combinations(sorted(logp), 2):
        la, lb = logp[a], logp[b]
        la_is, lb_is = la[:split], lb[:split]
        # Return correlation pre-filter (in-sample only).
        ra, rb = np.diff(la_is), np.diff(lb_is)
        if np.std(ra) == 0 or np.std(rb) == 0:
            continue
        corr = float(np.corrcoef(ra, rb)[0, 1])
        if corr < CORR_MIN:
            continue
        # Hedge ratio (OLS slope of la on lb), in-sample.
        beta = float(np.cov(la_is, lb_is)[0, 1] / np.var(lb_is))
        spread_is = la_is - beta * lb_is
        hl = half_life(spread_is)
        if not (1.0 < hl < MAX_HALFLIFE):
            continue
        selected.append((a, b, corr, beta, hl))
        # OUT-OF-SAMPLE test: full spread (in-sample beta), trade only OOS with a
        # trailing rolling z-score (no look-ahead, adapts to drift).
        spread_full = la - beta * lb
        t = backtest_pair(spread_full, split)
        all_trades.extend(t)

    print(f"\nSelected {len(selected)} cointegrated pairs (corr>={CORR_MIN}, "
          f"half-life<{MAX_HALFLIFE} bars)")
    if not all_trades:
        print("No OOS trades generated.")
        return

    n = len(all_trades)
    wins = [x for x in all_trades if x > 0]
    total = sum(all_trades)
    avg = total / n
    wr = len(wins) / n

    print("\n" + "=" * 64)
    print(f"STAT-ARB OUT-OF-SAMPLE — {n} trades, {len(selected)} pairs")
    print("=" * 64)
    print(f"  Net total:       {total:+.2f}%")
    print(f"  Avg net/trade:   {avg:+.4f}%   ← the number that matters")
    print(f"  Win rate:        {wr*100:.0f}%")
    print("\n  Sample pairs (corr, half-life bars):")
    for a, b, corr, beta, hl in sorted(selected, key=lambda x: x[4])[:8]:
        print(f"    {a:9}/{b:9}  corr={corr:.2f}  half-life={hl:.0f}b")

    print("\n" + "-" * 64)
    if n < MIN_TRADES:
        print(f"VERDICT: ⏳ INCONCLUSIVE — {n}/{MIN_TRADES} OOS trades.")
    elif avg > 0 and total > 0:
        print(f"VERDICT: ✅ OOS EDGE — avg {avg:+.4f}%/trade, {wr*100:.0f}% win on "
              f"held-out data. Real candidate — validate execution + capacity next.")
    else:
        print(f"VERDICT: ❌ NO OOS EDGE — avg {avg:+.4f}%/trade after {FEE_RT*100:.2f}% "
              f"fees. Pairs that cohered in-sample didn't pay out-of-sample (overfit).")
    print("-" * 64)


if __name__ == "__main__":
    asyncio.run(main())
