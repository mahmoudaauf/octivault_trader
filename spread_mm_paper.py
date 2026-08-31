#!/usr/bin/env python3
"""Paper market-maker — does spread capture survive adverse selection?

PAPER ONLY. There is no live code path in this file: it places no orders, holds
no keys beyond read-only market data, and moves no money. It exists to falsify
(or support) one claim with real numbers, the way every other candidate edge in
this project was tested.

THE QUESTION
------------
Liquid pairs are hopeless: at a 0.100% maker fee a round trip costs 0.200% and
BTCUSDT's spread is 0.0000%. But ~20% of USDT pairs quote wider than that
hurdle. Is that spread real profit, or is it compensation for risk that shows up
the moment you actually hold inventory?

WHY MARK-OUTS ARE THE METRIC
----------------------------
Quoting both sides is economically B-booking: you internalise flow and take the
other side of whoever hits you. Every B-book desk lives or dies on whether that
flow is BENIGN (price mean-reverts after the fill, you keep the spread) or TOXIC
(price keeps running, the spread never covers it). The standard measure is the
MARK-OUT: where mid sits 1m/5m after a fill, signed by trade direction.

  positive mark-out -> the fill was good; you were paid to provide liquidity
  negative mark-out -> you were picked off; the "spread" was an illusion

Naive P&L hides this because unrealised inventory drift masquerades as profit.
Mark-outs separate the two.

FILL MODEL (deliberately conservative)
-------------------------------------
We cannot know real queue position from outside the book, so the default only
fills when the market trades THROUGH our quote — i.e. a print strictly better
than our price, which we'd have been filled by regardless of queue position.
`MM_QUEUE_MODEL=touch` fills at our price too (optimistic; assumes front of
queue). Run both: the truth is bracketed between them.

Usage:
  python3 spread_mm_paper.py                 # run the paper daemon
  python3 spread_mm_paper.py report          # mark-out + P&L analysis
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import statistics
import sys
import time
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

STATE = os.getenv("MM_STATE_FILE", "logs/spread_mm_state.json")
FILLS = os.getenv("MM_FILLS_FILE", "logs/spread_mm_fills.jsonl")

_DEFAULT_SYMS = "BTTCUSDT,SCRTUSDT,MOVEUSDT,COOKIEUSDT"
SYMBOLS = [s.strip().upper() for s in os.getenv("MM_SYMBOLS", _DEFAULT_SYMS).split(",") if s.strip()]
QUOTE_USD = float(os.getenv("MM_QUOTE_USD", "10.0"))      # notional per side
MAKER_FEE_PCT = float(os.getenv("MM_MAKER_FEE_PCT", "0.10"))
# How far inside the touch we quote, as a fraction of the spread. 0 = at the
# touch; 0.25 = a quarter of the way in (more fills, less capture).
INSIDE_FRAC = float(os.getenv("MM_INSIDE_FRAC", "0.0"))
QUEUE_MODEL = os.getenv("MM_QUEUE_MODEL", "through").lower()   # through | touch
MAX_INVENTORY_USD = float(os.getenv("MM_MAX_INVENTORY_USD", "30.0"))
POLL_S = float(os.getenv("MM_POLL_S", "20"))
MARKOUT_S = [float(x) for x in os.getenv("MM_MARKOUT_S", "60,300").split(",")]
KILL_FILE = os.getenv("MM_KILL_FILE", "logs/spread_mm.stop")


def _load(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return dict(default)


def _save(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def _log_fill(rec: dict):
    os.makedirs(os.path.dirname(FILLS) or ".", exist_ok=True)
    with open(FILLS, "a") as f:
        f.write(json.dumps(rec) + "\n")


def compute_quotes(bid: float, ask: float, inside_frac: float = None):
    """Our two quote prices. inside_frac steps us into the spread: 0 quotes at
    the touch (max capture, worst queue position), 0.5 would cross to mid."""
    frac = INSIDE_FRAC if inside_frac is None else inside_frac
    spread = ask - bid
    return bid + spread * frac, ask - spread * frac


def simulate_fills(trades, bid_q: float, ask_q: float, queue_model: str = None):
    """Which of our resting quotes would these prints have filled?

    Binance aggTrade `m` = True means the BUYER was the maker, i.e. the
    aggressor SOLD -> that hits our BID. `m` False -> aggressor BOUGHT -> lifts
    our ASK.

    'through' (default) only fills when the print is strictly better than our
    quote, so the market traded past our level and queue position is irrelevant.
    'touch' also fills at our exact price, assuming we were at the front.
    """
    model = QUEUE_MODEL if queue_model is None else queue_model
    bought = sold = 0.0
    for t in trades:
        px, qty, buyer_is_maker = float(t["p"]), float(t["q"]), bool(t["m"])
        if buyer_is_maker:                      # aggressive SELL -> hits our bid
            if px < bid_q or (model == "touch" and px <= bid_q):
                bought += qty
        else:                                   # aggressive BUY -> lifts our ask
            if px > ask_q or (model == "touch" and px >= ask_q):
                sold += qty
    return bought, sold


def markout_pct(side: str, mid_at_fill: float, later_mid: float) -> float:
    """Signed mark-out: how far MID drifted after the fill, in our favour.

    Measured mid-to-mid, NOT fill-to-mid. Comparing the fill price to mid double
    counts the spread: buying at the bid sits half a spread below mid by
    definition, so a fill-to-mid mark-out reads +half-spread even when the price
    never moves. That produced an apparent edge on the first smoke run where
    every mark-out equalled the half-spread almost exactly -- a false positive,
    the most expensive kind of measurement error.

    Mid-to-mid isolates the only thing mark-outs are for: did the market move
    against us after we were filled?
      positive -> benign flow, we keep the spread
      negative -> toxic flow, we were picked off
    """
    if mid_at_fill <= 0:
        return 0.0
    raw = (later_mid - mid_at_fill) / mid_at_fill * 100.0
    return raw if side == "buy" else -raw


def _report():
    if not os.path.exists(FILLS):
        print("No fills recorded yet.")
        return
    rows = []
    with open(FILLS) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    done = [r for r in rows if r.get("markouts")]
    print("=" * 72)
    print("PAPER MARKET-MAKER — spread capture vs adverse selection")
    print("=" * 72)
    print(f"  fills recorded      : {len(rows)}   (with mark-outs: {len(done)})")
    if not done:
        print("  Not enough history yet — mark-outs need the horizon to elapse.")
        print("=" * 72)
        return
    by_sym: dict[str, list] = {}
    for r in done:
        by_sym.setdefault(r["symbol"], []).append(r)

    hdr = f"{'symbol':12} {'fills':>6} {'gross%':>8} {'fees%':>7}"
    for h in MARKOUT_S:
        hdr += f" {'mo'+str(int(h))+'s%':>9}"
    hdr += f" {'net%':>8}  verdict"
    print(hdr)
    print("-" * 72)
    for sym, rs in sorted(by_sym.items()):
        gross = statistics.mean([r["half_spread_pct"] for r in rs])
        fees = MAKER_FEE_PCT
        line = f"{sym:12} {len(rs):6d} {gross:8.3f} {fees:7.3f}"
        mos = {}
        for h in MARKOUT_S:
            vals = [r["markouts"][str(h)] for r in rs if str(h) in r.get("markouts", {})]
            mo = statistics.mean(vals) if vals else float("nan")
            mos[h] = mo
            line += f" {mo:9.3f}"
        worst = min([m for m in mos.values() if not math.isnan(m)], default=0.0)
        net = gross - fees + worst
        line += f" {net:8.3f}  {'✅ edge' if net > 0 else '❌ no edge'}"
        print(line)
    print("-" * 72)
    allmo = {h: [r["markouts"][str(h)] for r in done if str(h) in r.get("markouts", {})]
             for h in MARKOUT_S}
    for h in MARKOUT_S:
        v = allmo[h]
        if v:
            neg = sum(1 for x in v if x < 0) / len(v) * 100
            print(f"  mark-out @{int(h):>4}s : mean {statistics.mean(v):+.4f}%   "
                  f"negative on {neg:.0f}% of fills")
    print("\n  Interpretation: mean mark-out is what the flow is really worth to you.")
    print("  Negative => the quoted spread was compensation for toxicity, not profit.")
    print(f"  Fill model: {QUEUE_MODEL} (conservative='through'); quote inside={INSIDE_FRAC}")
    print("=" * 72)


async def run():
    from binance import AsyncClient
    from exchange_resilience import create_client_with_retry, resync_clock

    client = await create_client_with_retry(AsyncClient, label="spread-mm")
    state = _load(STATE, {"inventory": {}, "last_trade_id": {}, "pending": []})
    state.setdefault("inventory", {})
    state.setdefault("last_trade_id", {})
    state.setdefault("pending", [])

    print(f"[spread-mm] start — PAPER ONLY (no orders, no keys used for trading)")
    print(f"[spread-mm] symbols={SYMBOLS} quote=${QUOTE_USD}/side inside={INSIDE_FRAC} "
          f"queue={QUEUE_MODEL} fee={MAKER_FEE_PCT}%")
    print(f"[spread-mm] measuring MARK-OUTS at {MARKOUT_S}s — the toxic-flow test")
    try:
        while True:
            await resync_clock(client, "spread-mm")
            if os.path.exists(KILL_FILE):
                print("[spread-mm] kill file present — idling")
                await asyncio.sleep(POLL_S)
                continue
            now = time.time()
            for sym in SYMBOLS:
                try:
                    # CAUSAL ORDER MATTERS. A quote can only be filled by trades
                    # that happen AFTER it is posted. So: match this interval's
                    # new prints against LAST cycle's quote, then post a fresh
                    # quote from the current book for the next interval.
                    #
                    # Getting this wrong is not a small error: aggTrades returns
                    # the last 500 prints (hours, on a thin pair), so matching
                    # them against the current book manufactured a full set of
                    # fake fills on the very first cycle and reported an edge.
                    trades = await client.get_aggregate_trades(symbol=sym, limit=500)
                    if not trades:
                        continue
                    newest = max(t["a"] for t in trades)
                    last_id = state["last_trade_id"].get(sym)
                    prev = state.get("quote", {}).get(sym)

                    if last_id is None:
                        # First sight of this symbol: adopt the watermark and
                        # post a quote. Deliberately no fills -- the backlog
                        # predates any quote of ours.
                        state["last_trade_id"][sym] = newest
                        print(f"  [WARMUP] {sym} watermark set; no fills from backlog")
                    else:
                        fresh = [t for t in trades if t["a"] > last_id]
                        state["last_trade_id"][sym] = newest
                        if fresh and prev:
                            bought, sold = simulate_fills(fresh, prev["bid_q"], prev["ask_q"])
                            inv = float(state["inventory"].get(sym, 0.0))
                            mid_p = prev["mid"]
                            max_qty = QUOTE_USD / mid_p if mid_p > 0 else 0.0
                            for side, qty, px in (("buy", bought, prev["bid_q"]),
                                                  ("sell", sold, prev["ask_q"])):
                                if qty <= 0:
                                    continue
                                qty = min(qty, max_qty)
                                if side == "buy" and (inv + qty) * mid_p > MAX_INVENTORY_USD:
                                    qty = max(0.0, MAX_INVENTORY_USD / mid_p - inv)
                                if side == "sell" and (inv - qty) * mid_p < -MAX_INVENTORY_USD:
                                    qty = max(0.0, inv + MAX_INVENTORY_USD / mid_p)
                                if qty <= 0:
                                    continue
                                inv = inv + qty if side == "buy" else inv - qty
                                rec = {
                                    "ts": datetime.now(timezone.utc).isoformat(), "t": now,
                                    "symbol": sym, "side": side, "price": px, "qty": qty,
                                    "mid_at_fill": mid_p,
                                    "half_spread_pct": prev["half_spread_pct"],
                                    "inventory_after": inv, "markouts": {},
                                }
                                state["pending"].append(rec)
                                print(f"  [FILL] {sym} {side} {qty:.6f} @ {px:.10g} "
                                      f"half-spread={prev['half_spread_pct']:.3f}% "
                                      f"inv=${inv*mid_p:.2f}")
                            state["inventory"][sym] = inv

                    # Post the quote that will be live over the NEXT interval.
                    ob = await client.get_order_book(symbol=sym, limit=5)
                    if not ob.get("bids") or not ob.get("asks"):
                        continue
                    bid, ask = float(ob["bids"][0][0]), float(ob["asks"][0][0])
                    if bid <= 0 or ask <= bid:
                        continue
                    mid = (bid + ask) / 2
                    bid_q, ask_q = compute_quotes(bid, ask)
                    state.setdefault("quote", {})[sym] = {
                        "bid_q": bid_q, "ask_q": ask_q, "mid": mid,
                        "half_spread_pct": (ask_q - bid_q) / 2 / mid * 100, "t": now,
                    }
                except Exception as e:
                    print(f"  [MM-ERROR] {sym}: {str(e)[:80]}")

            # Resolve mark-outs whose horizon has elapsed.
            still = []
            for rec in state["pending"]:
                try:
                    age = now - rec["t"]
                    if age < max(MARKOUT_S):
                        for h in MARKOUT_S:
                            if age >= h and str(h) not in rec["markouts"]:
                                ob = await client.get_order_book(symbol=rec["symbol"], limit=5)
                                m = (float(ob["bids"][0][0]) + float(ob["asks"][0][0])) / 2
                                rec["markouts"][str(h)] = markout_pct(rec["side"], rec["mid_at_fill"], m)
                        still.append(rec)
                    else:
                        for h in MARKOUT_S:
                            if str(h) not in rec["markouts"]:
                                ob = await client.get_order_book(symbol=rec["symbol"], limit=5)
                                m = (float(ob["bids"][0][0]) + float(ob["asks"][0][0])) / 2
                                rec["markouts"][str(h)] = markout_pct(rec["side"], rec["mid_at_fill"], m)
                        _log_fill(rec)
                except Exception as e:
                    print(f"  [MARKOUT-ERROR] {str(e)[:70]}")
                    still.append(rec)
            state["pending"] = still

            _save(STATE, state)
            ts = datetime.now(timezone.utc).strftime("%H:%M")
            invs = ", ".join(f"{s}:{state['inventory'].get(s,0.0):.4g}" for s in SYMBOLS)
            print(f"[spread-mm {ts}] pending={len(state['pending'])} inv[{invs}]")
            await asyncio.sleep(POLL_S)
    finally:
        await client.close_connection()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        _report()
    else:
        asyncio.run(run())
