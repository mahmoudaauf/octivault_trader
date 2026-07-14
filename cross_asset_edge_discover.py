#!/usr/bin/env python3
"""
Cross-asset / funding-sentiment signal discovery — OUT-OF-SAMPLE falsification.

Context: the live ML forecaster (GRU + OHLCV-derived technical features, e.g.
RSI/MACD/EMA/ATR/volume z-scores) has a proven negative edge, and a from-scratch
stat-arb pairs test (statarb_discover.py, same OOS methodology) also failed.
Both prior tests only ever showed the model/strategy information ALREADY
derivable from a single symbol's own OHLCV history. This script tests two
DIFFERENT categories of information the prior approaches never used at all:

  1. BTC lead-lag momentum spillover — does BTC's most recent move predict
     an altcoin's NEXT move? (cross-asset, not single-symbol technical).
  2. Funding-rate extremes as a CONTRARIAN sentiment signal — does an extreme
     funding print (crowded long/short positioning) predict a subsequent
     price reversal? (This is a different use of funding data than the
     funding-carry strategy, which profits from collecting funding regardless
     of direction; here funding is used purely as a directional signal.)

Same discipline as statarb_discover.py: select on the FIRST half of history
(in-sample), test ONLY on the held-out second half (out-of-sample), with
thresholds and parameters FIXED IN ADVANCE (not tuned per-symbol beyond a
simple in-sample correlation pre-filter, to avoid the "always find something
in-sample" trap). Net-of-fee, criteria fixed in advance, MIN_TRADES=30.

Usage: python3 cross_asset_edge_discover.py
"""
from __future__ import annotations

import asyncio
import os
import sys

import numpy as np

ALTS = [
    "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT",
    "AVAXUSDT", "LINKUSDT", "MATICUSDT", "DOTUSDT", "LTCUSDT", "BCHUSDT",
    "ATOMUSDT", "UNIUSDT", "APTUSDT", "ARBUSDT", "OPUSDT", "INJUSDT",
    "NEARUSDT", "FILUSDT", "AAVEUSDT", "ETCUSDT", "XLMUSDT", "ALGOUSDT",
    "ICPUSDT", "SUIUSDT", "SEIUSDT", "RUNEUSDT", "GALAUSDT",
]
BTC = "BTCUSDT"

# ── Hypothesis 1: BTC lead-lag momentum spillover ───────────────────────────
H1_INTERVAL = os.getenv("XA_H1_INTERVAL", "1h")
H1_BARS = int(os.getenv("XA_H1_BARS", "1000"))
H1_LAG = 1          # bars — BTC's most recently completed return (fixed, not tuned)
H1_HORIZON = 4      # bars — alt's forward return window (fixed, not tuned)
H1_BTC_ENTRY_PCT = float(os.getenv("XA_H1_BTC_ENTRY_PCT", "0.5")) / 100.0  # |BTC 1h return| trigger
H1_CORR_MIN = float(os.getenv("XA_H1_CORR_MIN", "0.15"))  # in-sample pre-filter threshold
H1_FEE_RT = float(os.getenv("XA_H1_FEE_RT_PCT", "0.15")) / 100.0  # spot round-trip cost

# ── Hypothesis 2: funding-rate extreme mean-reversion ───────────────────────
H2_INTERVAL = "8h"  # matches funding settlement cadence — trivial time alignment
H2_BARS = int(os.getenv("XA_H2_BARS", "500"))  # Binance futures_funding_rate caps at ~500 rows regardless of limit requested (confirmed empirically, ~166 days)
H2_HORIZON = 2      # windows forward (16h)
H2_PCTILE = float(os.getenv("XA_H2_PCTILE", "20"))  # top/bottom Nth percentile of funding = "extreme"
H2_CORR_MAX = float(os.getenv("XA_H2_CORR_MAX", "-0.15"))  # in-sample pre-filter: funding vs fwd return must be negatively correlated (contrarian)
H2_FEE_RT = float(os.getenv("XA_H2_FEE_RT_PCT", "0.15")) / 100.0

MIN_TRADES = 30


async def fetch_closes(client, symbol: str, interval: str, bars: int) -> np.ndarray | None:
    try:
        kl = await client.get_klines(symbol=symbol, interval=interval, limit=bars)
        return np.array([float(k[4]) for k in kl], dtype=np.float64)
    except Exception as e:
        print(f"  {symbol}: klines fetch failed ({str(e)[:50]})")
        return None


async def fetch_funding_rates(client, symbol: str, limit: int) -> np.ndarray | None:
    try:
        rows = await client.futures_funding_rate(symbol=symbol, limit=limit)
        return np.array([float(r["fundingRate"]) for r in rows], dtype=np.float64)
    except Exception as e:
        print(f"  {symbol}: funding fetch failed ({str(e)[:50]})")
        return None


def hypothesis_1(btc_closes: np.ndarray, alt_closes: dict[str, np.ndarray]) -> None:
    print("\n" + "=" * 72)
    print("HYPOTHESIS 1 — BTC lead-lag momentum spillover")
    print("=" * 72)
    L = min(len(btc_closes), min(len(c) for c in alt_closes.values()))
    btc = btc_closes[-L:]
    btc_ret = np.diff(np.log(btc))  # 1h log returns, length L-1
    split = (L - 1) // 2

    selected: list[tuple[str, float]] = []
    all_trades: list[float] = []
    for sym, closes in alt_closes.items():
        alt = closes[-L:]
        alt_ret = np.diff(np.log(alt))  # aligned with btc_ret
        n = len(alt_ret)
        # In-sample: correlate BTC's return at t-1 with alt's return at t (1-bar lead).
        btc_lead_is = btc_ret[H1_LAG - 1 : split - 1]
        alt_next_is = alt_ret[H1_LAG : split]
        if len(btc_lead_is) < 30 or np.std(btc_lead_is) == 0 or np.std(alt_next_is) == 0:
            continue
        corr = float(np.corrcoef(btc_lead_is, alt_next_is)[0, 1])
        if corr < H1_CORR_MIN:
            continue
        selected.append((sym, corr))

        # OOS: mechanical rule, fixed thresholds, using in-sample-confirmed
        # direction (positive corr => trade WITH BTC's move).
        t = split
        while t + H1_HORIZON < n:
            btc_move = btc_ret[t - H1_LAG]
            if abs(btc_move) >= H1_BTC_ENTRY_PCT:
                direction = 1.0 if btc_move > 0 else -1.0
                fwd_ret = float(np.log(alt[t + 1 + H1_HORIZON] / alt[t + 1]))
                pnl_pct = direction * fwd_ret * 100.0 - H1_FEE_RT * 100.0
                all_trades.append(pnl_pct)
                t += H1_HORIZON  # no overlapping trades on the same symbol
            else:
                t += 1

    print(f"Selected {len(selected)}/{len(alt_closes)} alts with in-sample BTC-lead corr >= {H1_CORR_MIN}")
    for sym, corr in sorted(selected, key=lambda x: -x[1])[:10]:
        print(f"    {sym:9}  corr={corr:.3f}")

    _print_verdict("H1 BTC-lead-lag", all_trades, H1_FEE_RT)


def hypothesis_2(funding: dict[str, np.ndarray], closes: dict[str, np.ndarray]) -> None:
    print("\n" + "=" * 72)
    print("HYPOTHESIS 2 — funding-rate extremes as contrarian sentiment signal")
    print("=" * 72)

    selected: list[tuple[str, float]] = []
    all_trades: list[float] = []
    for sym in funding:
        fr = funding[sym]
        px = closes.get(sym)
        if px is None:
            continue
        L = min(len(fr), len(px))
        fr, px = fr[-L:], px[-L:]
        split = L // 2
        fwd_ret = np.full(L, np.nan)
        for t in range(L - H2_HORIZON):
            fwd_ret[t] = np.log(px[t + H2_HORIZON] / px[t])

        fr_is, fwd_is = fr[:split], fwd_ret[:split]
        mask_is = ~np.isnan(fwd_is)
        if mask_is.sum() < 30 or np.std(fr_is[mask_is]) == 0:
            continue
        corr = float(np.corrcoef(fr_is[mask_is], fwd_is[mask_is])[0, 1])
        if corr > H2_CORR_MAX:  # need corr <= H2_CORR_MAX (i.e. sufficiently negative)
            continue
        selected.append((sym, corr))

        # Calibrate extreme thresholds from IN-SAMPLE funding distribution only.
        hi_thr = float(np.percentile(fr_is, 100 - H2_PCTILE))
        lo_thr = float(np.percentile(fr_is, H2_PCTILE))

        # OOS: mechanical contrarian rule with in-sample-calibrated thresholds.
        t = split
        while t + H2_HORIZON < L:
            f = fr[t]
            if f >= hi_thr and hi_thr > 0:
                pnl_pct = -1.0 * fwd_ret[t] * 100.0 - H2_FEE_RT * 100.0  # short (fade crowded longs)
                all_trades.append(pnl_pct)
                t += H2_HORIZON
            elif f <= lo_thr and lo_thr < 0:
                pnl_pct = 1.0 * fwd_ret[t] * 100.0 - H2_FEE_RT * 100.0  # long (fade crowded shorts)
                all_trades.append(pnl_pct)
                t += H2_HORIZON
            else:
                t += 1

    print(f"Selected {len(selected)}/{len(funding)} symbols with in-sample funding-vs-fwd-return corr <= {H2_CORR_MAX}")
    for sym, corr in sorted(selected, key=lambda x: x[1])[:10]:
        print(f"    {sym:9}  corr={corr:.3f}")

    _print_verdict("H2 funding-extreme contrarian", all_trades, H2_FEE_RT)


def _print_verdict(label: str, trades: list[float], fee_rt: float) -> None:
    n = len(trades)
    if n == 0:
        print(f"\n{label}: no OOS trades generated.")
        return
    wins = [x for x in trades if x > 0]
    total = sum(trades)
    avg = total / n
    wr = len(wins) / n
    print(f"\n{'-'*72}")
    print(f"{label} — OOS RESULTS: {n} trades")
    print(f"  Net total:       {total:+.2f}%")
    print(f"  Avg net/trade:   {avg:+.4f}%   <- the number that matters")
    print(f"  Win rate:        {wr*100:.0f}%")
    if n < MIN_TRADES:
        print(f"  VERDICT: INCONCLUSIVE — {n}/{MIN_TRADES} OOS trades.")
    elif avg > 0 and total > 0:
        print(f"  VERDICT: OOS EDGE — avg {avg:+.4f}%/trade, {wr*100:.0f}% win on held-out data.")
    else:
        print(f"  VERDICT: NO OOS EDGE — avg {avg:+.4f}%/trade after {fee_rt*100:.2f}% fees.")
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
        print(f"Fetching {H1_INTERVAL} klines ({H1_BARS} bars) for BTC + {len(ALTS)} alts...")
        btc_closes = await fetch_closes(client, BTC, H1_INTERVAL, H1_BARS)
        alt_closes_1h: dict[str, np.ndarray] = {}
        for sym in ALTS:
            c = await fetch_closes(client, sym, H1_INTERVAL, H1_BARS)
            if c is not None and len(c) >= H1_BARS * 0.9:
                alt_closes_1h[sym] = c
        if btc_closes is not None and alt_closes_1h:
            hypothesis_1(btc_closes, alt_closes_1h)
        else:
            print("H1 skipped: insufficient data.")

        print(f"\nFetching funding rate history + {H2_INTERVAL} klines for {len(ALTS)} alts...")
        funding: dict[str, np.ndarray] = {}
        closes_8h: dict[str, np.ndarray] = {}
        for sym in ALTS:
            fr = await fetch_funding_rates(client, sym, H2_BARS)
            if fr is not None and len(fr) >= H2_BARS * 0.9:
                funding[sym] = fr
            c = await fetch_closes(client, sym, H2_INTERVAL, H2_BARS)
            if c is not None and len(c) >= H2_BARS * 0.9:
                closes_8h[sym] = c
        if funding and closes_8h:
            hypothesis_2(funding, closes_8h)
        else:
            print("H2 skipped: insufficient data.")
    finally:
        await client.close_connection()


if __name__ == "__main__":
    asyncio.run(main())
