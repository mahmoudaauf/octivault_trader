#!/usr/bin/env python3
"""
Cross-exchange signal discovery — OUT-OF-SAMPLE falsification.

Context: cross_asset_edge_discover.py already falsified BTC lead-lag momentum
spillover and funding-extreme contrarian sentiment (both -- see
edge-verdict-no-edge memory). Those tests, like the ML forecaster and
stat-arb before them, only ever used information already derivable from
Binance's own OHLCV/funding history. This script tests a genuinely different
information category never used anywhere in this codebase: an INDEPENDENT
exchange's price series (Coinbase) as a signal for Binance-tradeable moves.

Both hypotheses are executed ENTIRELY on Binance -- Coinbase is used only as
an external information source, not a venue to trade. No cross-exchange
arbitrage (buy-here-sell-there) is involved.

  H1. Cross-exchange lead-lag: does Coinbase's most recent 1h return predict
      Binance's NEXT 1h return for the same symbol? (differing liquidity /
      order-flow composition between a US-institutional-hours-heavy venue
      and a 24/7 global-retail-heavy venue could plausibly cause a brief,
      tradeable lag.)
  H2. Cross-exchange spread reversion: when Binance trades at an unusual
      premium/discount to Coinbase for the same symbol, does Binance's OWN
      price subsequently converge back toward Coinbase's (i.e. move in the
      direction that closes the gap)? Tradeable on Binance alone.

Same discipline as the prior two scripts: select on the FIRST half of
history (in-sample correlation pre-filter only -- lag/horizon/thresholds are
FIXED IN ADVANCE, not tuned), test ONLY on the held-out second half
(out-of-sample), net-of-fee, MIN_TRADES=30, honest verdict either way.

Usage: python3 cross_exchange_edge_discover.py
"""
from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np

# Alt universe: same as cross_asset_edge_discover.py's ALTS, minus RUNEUSDT
# (RUNE-USD returns 404 on Coinbase's public API -- confirmed empirically).
ALTS = [
    "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT",
    "AVAXUSDT", "LINKUSDT", "MATICUSDT", "DOTUSDT", "LTCUSDT", "BCHUSDT",
    "ATOMUSDT", "UNIUSDT", "APTUSDT", "ARBUSDT", "OPUSDT", "INJUSDT",
    "NEARUSDT", "FILUSDT", "AAVEUSDT", "ETCUSDT", "XLMUSDT", "ALGOUSDT",
    "ICPUSDT", "SUIUSDT", "SEIUSDT", "GALAUSDT",
]
INTERVAL = "1h"
BARS = int(os.getenv("XE_BARS", "700"))
COINBASE_BASE = "https://api.exchange.coinbase.com"

# ── Hypothesis 1: cross-exchange lead-lag (Coinbase -> Binance) ────────────
H1_LAG = 1            # bars -- Coinbase's most recently completed return (fixed)
H1_HORIZON = 1         # bars -- Binance's forward return window (fixed)
H1_ENTRY_PCT = float(os.getenv("XE_H1_ENTRY_PCT", "0.3")) / 100.0  # |Coinbase 1h return| trigger
H1_CORR_MIN = float(os.getenv("XE_H1_CORR_MIN", "0.15"))  # in-sample pre-filter
H1_FEE_RT = float(os.getenv("XE_FEE_RT_PCT", "0.15")) / 100.0  # spot round-trip cost

# ── Hypothesis 2: Binance-vs-Coinbase spread reversion ──────────────────────
H2_HORIZON = 2         # bars forward
H2_PCTILE = float(os.getenv("XE_H2_PCTILE", "20"))  # top/bottom Nth percentile spread = "extreme"
H2_CORR_MAX = float(os.getenv("XE_H2_CORR_MAX", "-0.15"))  # in-sample pre-filter: must be reversion (negative corr)
H2_FEE_RT = float(os.getenv("XE_FEE_RT_PCT", "0.15")) / 100.0

MIN_TRADES = 30


async def fetch_binance_closes_by_ts(client, symbol: str, interval: str, bars: int) -> dict[int, float] | None:
    """epoch(sec) -> close, keyed by each kline's OPEN time."""
    try:
        kl = await client.get_klines(symbol=symbol, interval=interval, limit=bars)
        return {int(k[0]) // 1000: float(k[4]) for k in kl}
    except Exception as e:
        print(f"  {symbol}: Binance klines fetch failed ({str(e)[:50]})")
        return None


async def fetch_coinbase_closes(session, product: str, bars: int) -> dict[int, float] | None:
    """epoch(sec) -> close, paginated backward in <=300-row chunks (Coinbase's per-request cap)."""
    out: dict[int, float] = {}
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    chunk_hours = 300
    stalls = 0
    while len(out) < bars and stalls < 3:
        start = end - timedelta(hours=chunk_hours)
        url = (
            f"{COINBASE_BASE}/products/{product}/candles"
            f"?granularity=3600&start={start.isoformat()}&end={end.isoformat()}"
        )
        try:
            async with session.get(url, timeout=15) as resp:
                if resp.status != 200:
                    break
                rows = await resp.json()
        except Exception as e:
            print(f"  {product}: Coinbase fetch failed ({str(e)[:50]})")
            break
        if not rows:
            stalls += 1
        else:
            stalls = 0
            for r in rows:
                # row: [time, low, high, open, close, volume]
                out[int(r[0])] = float(r[4])
        end = start
        await asyncio.sleep(0.35)  # polite pacing for the public endpoint
    return out or None


def _align(binance_map: dict[int, float], coinbase_map: dict[int, float]) -> tuple[np.ndarray, np.ndarray] | None:
    """Intersect by exact hour-epoch timestamp -- never assume positional alignment
    across two independently-fetched, independently-gapped series."""
    common = sorted(set(binance_map) & set(coinbase_map))
    if len(common) < 200:
        return None
    b = np.array([binance_map[t] for t in common], dtype=np.float64)
    c = np.array([coinbase_map[t] for t in common], dtype=np.float64)
    return b, c


def hypothesis_1(pairs: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    print("\n" + "=" * 72)
    print("HYPOTHESIS 1 — cross-exchange lead-lag (Coinbase -> Binance)")
    print("=" * 72)

    selected: list[tuple[str, float]] = []
    all_trades: list[float] = []
    for sym, (binance, coinbase) in pairs.items():
        b_ret = np.diff(np.log(binance))
        c_ret = np.diff(np.log(coinbase))
        n = len(b_ret)
        split = n // 2

        c_lead_is = c_ret[H1_LAG - 1 : split - 1]
        b_next_is = b_ret[H1_LAG : split]
        if len(c_lead_is) < 30 or np.std(c_lead_is) == 0 or np.std(b_next_is) == 0:
            continue
        corr = float(np.corrcoef(c_lead_is, b_next_is)[0, 1])
        if corr < H1_CORR_MIN:
            continue
        selected.append((sym, corr))

        # OOS: mechanical rule, fixed thresholds, in-sample-confirmed direction
        # (positive corr => trade Binance WITH Coinbase's prior move).
        t = split
        while t + H1_HORIZON < n:
            cb_move = c_ret[t - H1_LAG]
            if abs(cb_move) >= H1_ENTRY_PCT:
                direction = 1.0 if cb_move > 0 else -1.0
                fwd_ret = float(np.log(binance[t + 1 + H1_HORIZON] / binance[t + 1]))
                pnl_pct = direction * fwd_ret * 100.0 - H1_FEE_RT * 100.0
                all_trades.append(pnl_pct)
                t += H1_HORIZON
            else:
                t += 1

    print(f"Selected {len(selected)}/{len(pairs)} symbols with in-sample Coinbase-lead corr >= {H1_CORR_MIN}")
    for sym, corr in sorted(selected, key=lambda x: -x[1])[:10]:
        print(f"    {sym:9}  corr={corr:.3f}")

    _print_verdict("H1 cross-exchange lead-lag", all_trades, H1_FEE_RT)


def hypothesis_2(pairs: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    print("\n" + "=" * 72)
    print("HYPOTHESIS 2 — Binance-vs-Coinbase spread reversion (Binance-only execution)")
    print("=" * 72)

    selected: list[tuple[str, float]] = []
    all_trades: list[float] = []
    for sym, (binance, coinbase) in pairs.items():
        n = len(binance)
        spread = (binance - coinbase) / coinbase
        fwd_ret = np.full(n, np.nan)
        for t in range(n - H2_HORIZON):
            fwd_ret[t] = np.log(binance[t + H2_HORIZON] / binance[t])

        split = n // 2
        spread_is, fwd_is = spread[:split], fwd_ret[:split]
        mask_is = ~np.isnan(fwd_is)
        if mask_is.sum() < 30 or np.std(spread_is[mask_is]) == 0:
            continue
        corr = float(np.corrcoef(spread_is[mask_is], fwd_is[mask_is])[0, 1])
        if corr > H2_CORR_MAX:  # need corr <= H2_CORR_MAX (sufficiently negative = reversion)
            continue
        selected.append((sym, corr))

        hi_thr = float(np.percentile(spread_is, 100 - H2_PCTILE))
        lo_thr = float(np.percentile(spread_is, H2_PCTILE))

        t = split
        while t + H2_HORIZON < n:
            s = spread[t]
            if s >= hi_thr and hi_thr > 0:
                pnl_pct = -1.0 * fwd_ret[t] * 100.0 - H2_FEE_RT * 100.0  # Binance premium -> short
                all_trades.append(pnl_pct)
                t += H2_HORIZON
            elif s <= lo_thr and lo_thr < 0:
                pnl_pct = 1.0 * fwd_ret[t] * 100.0 - H2_FEE_RT * 100.0  # Binance discount -> long
                all_trades.append(pnl_pct)
                t += H2_HORIZON
            else:
                t += 1

    print(f"Selected {len(selected)}/{len(pairs)} symbols with in-sample spread-vs-fwd-return corr <= {H2_CORR_MAX}")
    for sym, corr in sorted(selected, key=lambda x: x[1])[:10]:
        print(f"    {sym:9}  corr={corr:.3f}")

    _print_verdict("H2 cross-exchange spread reversion", all_trades, H2_FEE_RT)


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
    import aiohttp

    client = await AsyncClient.create(
        os.getenv("BINANCE_API_KEY") or "x", os.getenv("BINANCE_API_SECRET") or "x"
    )

    pairs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    try:
        async with aiohttp.ClientSession() as session:
            print(f"Fetching {INTERVAL} closes ({BARS} bars) for {len(ALTS)} symbols on Binance + Coinbase...")
            for sym in ALTS:
                base = sym[:-4]  # strip "USDT"
                product = f"{base}-USD"
                b_map = await fetch_binance_closes_by_ts(client, sym, INTERVAL, BARS)
                if b_map is None:
                    continue
                c_map = await fetch_coinbase_closes(session, product, BARS)
                if c_map is None:
                    continue
                aligned = _align(b_map, c_map)
                if aligned is None:
                    print(f"  {sym}: insufficient overlapping bars, skipped")
                    continue
                pairs[sym] = aligned
                print(f"  {sym}: {len(aligned[0])} aligned bars")

        if not pairs:
            print("No symbols had usable aligned data. Aborting.")
            return

        hypothesis_1(pairs)
        hypothesis_2(pairs)
    finally:
        await client.close_connection()


if __name__ == "__main__":
    asyncio.run(main())
