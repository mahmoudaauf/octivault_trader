#!/usr/bin/env python3
"""
Halal allocator — autonomous growth through OWNERSHIP, not interest.

WHY THIS EXISTS
---------------
`hybrid_allocator.py` grows capital by lending it: USDC sits in Binance Simple
Earn and is paid a fixed APR. That return is riba, and the operator ruled out
instruments of that character on 2026-09-05. This file is the replacement that
keeps the autonomy and drops the interest.

Growth here comes from HOLDING APPRECIATING ASSETS. The machine buys, it never
lends, never borrows, never shorts, never uses leverage or derivatives. Its
return is whatever the assets do — which is real growth when they rise, and a
real loss when they fall. That honesty is the whole trade: Simple Earn paid a
near-certain 5%; this pays an uncertain amount and can be negative.

WHAT IT HOLDS (config/halal_targets.json)
-----------------------------------------
Default is gold-led, because allocated physical gold is the least contested
store of value in Islamic finance:
    PAXG 60%   1 token = 1 troy oz of allocated gold, Paxos, redeemable
    BTC  25%   held outright, no yield
    ETH  15%   held outright, no yield
Gold is up 23.4% over the trailing year as measured on 2026-09-05, which is
NOT a forecast and must not be read as one.

BTC and ETH permissibility is genuinely debated among scholars; gold is not.
Set both to 0 and PAXG to 100% if you want only the uncontested asset. Nothing
in this file assumes the default is the right answer for you.

HOW IT BUYS
-----------
Gap-fill, buy-only. Idle USDT goes to whichever holding is furthest BELOW its
target value, so the book drifts back toward target using new money and NOTHING
IS EVER SOLD to rebalance. Selling to rebalance would pay a spread twice and
realise gains for no reason; letting deposits do the work is free.

WHAT IT WILL NOT DO
-------------------
  - no Simple Earn, no lending, no staking-for-yield
  - no margin, no borrowing, no futures, no options, no short selling
  - no selling at all, except when the operator changes targets by hand
So the failure mode is "the assets fell", never "a position was liquidated".

Modes (HALAL_MODE, default paper):
  paper  — print the exact orders, send nothing
  live   — real spot buys; DOUBLE-GATED on MODE=live AND logs/halal_live_armed

Usage:
  python3 halal_allocator.py            # daemon
  python3 halal_allocator.py plan       # what it would buy right now
  python3 halal_allocator.py report     # NAV, drift, contributions vs growth
  python3 halal_allocator.py exit-earn  # plan the redeem out of Simple Earn
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone

from dotenv import load_dotenv

import capital_ledger as ledger

load_dotenv()

TARGETS_FILE = os.getenv("HALAL_TARGETS", "config/halal_targets.json")
STATE = os.getenv("HALAL_STATE", "logs/halal_state.json")
NAV_FILE = os.getenv("HALAL_NAV_FILE", "logs/halal_nav_history.jsonl")
LEDGER = os.getenv("HALAL_LEDGER", "logs/halal_ledger.jsonl")
PIDFILE = os.getenv("HALAL_PIDFILE", "logs/halal.pid")
KILL_FILE = os.getenv("HALAL_KILL_FILE", "logs/halal.stop")
ARM_FILE = os.getenv("HALAL_LIVE_ARM_FILE", "logs/halal_live_armed")
MODE = os.getenv("HALAL_MODE", "paper").lower()
POLL_MIN = float(os.getenv("HALAL_POLL_MIN", "60"))
CYCLE_TIMEOUT_S = float(os.getenv("HALAL_CYCLE_TIMEOUT_S", "300"))
QUOTE = "USDT"
# Cash left unspent. Binance's spot minimum is $5, so a buffer below that just
# strands money; this only covers rounding.
CASH_BUFFER = float(os.getenv("HALAL_CASH_BUFFER", "0.50"))
API_RETRIES = int(os.getenv("HALAL_API_RETRIES", "3"))

_lock = None


def _live() -> bool:
    return MODE == "live" and os.path.exists(ARM_FILE)


def _log(msg: str) -> None:
    print(f"[halal {datetime.now(timezone.utc):%m-%d %H:%M}] {msg}", flush=True)


def _book(rec: dict) -> None:
    os.makedirs(os.path.dirname(LEDGER) or ".", exist_ok=True)
    with open(LEDGER, "a") as f:
        f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(), **rec}) + "\n")


def load_targets() -> dict:
    """Load and validate weights. A table that does not sum to 1.0 silently
    under- or over-deploys every future buy while the plan still looks sane."""
    with open(TARGETS_FILE) as f:
        cfg = json.load(f)
    w = {k.upper(): float(v) for k, v in cfg["weights"].items() if float(v) > 0}
    if not w:
        raise ValueError(f"{TARGETS_FILE}: no assets with a positive weight")
    total = sum(w.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"{TARGETS_FILE}: weights sum to {total:.6f}, must be 1.0")
    cfg["_w"] = w
    return cfg


async def _retry(fn, *a, **k):
    last = None
    for i in range(API_RETRIES):
        try:
            return await fn(*a, **k)
        except Exception as e:
            last = e
            if i < API_RETRIES - 1:
                await asyncio.sleep(2 * (i + 1))
    raise last


async def _filters(client, symbol: str):
    info = await _retry(client.get_symbol_info, symbol)
    f = {x["filterType"]: x for x in info["filters"]}
    step = float(f["LOT_SIZE"]["stepSize"])
    mn = float(f.get("NOTIONAL", f.get("MIN_NOTIONAL", {})).get("minNotional", 5.0))
    return step, mn


async def snapshot(client, cfg: dict) -> dict | None:
    """Value every held asset plus the quote balance. None if unreadable, so a
    failed read is never written to the curve as a crash."""
    try:
        acct = await _retry(client.get_account)
    except Exception as e:
        _log(f"  balance read failed ({str(e)[:60]}) — skipping cycle")
        return None
    free = {b["asset"]: float(b["free"]) for b in acct["balances"]}
    cash = free.get(QUOTE, 0.0)
    holdings, invested = {}, 0.0
    for asset in cfg["_w"]:
        qty = free.get(asset, 0.0)
        try:
            px = float((await _retry(client.get_symbol_ticker, symbol=f"{asset}{QUOTE}"))["price"])
        except Exception:
            continue
        usd = qty * px
        holdings[asset] = {"qty": qty, "price": px, "usd": round(usd, 4)}
        invested += usd
    return {"cash": round(cash, 4), "invested": round(invested, 4),
            "holdings": holdings, "nav": round(cash + invested, 4)}


def plan_buys(snap: dict, cfg: dict) -> dict:
    """Gap-fill buy list. Every dollar goes to whatever is furthest below target.

    Degenerates to the plain target split when the book is already on target,
    which is correct rather than a special case.
    """
    w = cfg["_w"]
    deployable = max(0.0, snap["cash"] - CASH_BUFFER)
    values = {a: snap["holdings"].get(a, {}).get("usd", 0.0) for a in w}
    future = sum(values.values()) + deployable
    gaps = {a: max(0.0, future * w[a] - values[a]) for a in w}
    total_gap = sum(gaps.values())
    if deployable <= 0 or total_gap <= 0:
        alloc = {a: 0.0 for a in w}
    else:
        alloc = {a: deployable * gaps[a] / total_gap for a in w}
    return {"deployable": round(deployable, 4),
            "alloc": {a: round(v, 4) for a, v in alloc.items()},
            "values": values, "future": future}


def drift(snap: dict, cfg: dict) -> dict:
    w = cfg["_w"]
    inv = snap["invested"]
    return {a: {"usd": snap["holdings"].get(a, {}).get("usd", 0.0),
                "now_pct": round(100 * snap["holdings"].get(a, {}).get("usd", 0.0) / inv, 2) if inv else 0.0,
                "target_pct": round(100 * w[a], 2)} for a in w}


async def execute(client, plan: dict, cfg: dict) -> float:
    """Place the buys. Skips anything under the exchange minimum rather than
    sending an order that would be rejected and leave cash stranded."""
    spent = 0.0
    for asset, amt in sorted(plan["alloc"].items(), key=lambda kv: -kv[1]):
        if amt <= 0:
            continue
        symbol = f"{asset}{QUOTE}"
        try:
            _step, min_notional = await _filters(client, symbol)
        except Exception as e:
            _log(f"  [SKIP] {symbol}: filters unreadable ({str(e)[:50]})")
            continue
        if amt < min_notional:
            _log(f"  [SKIP] {asset}: ${amt:.2f} below ${min_notional:.0f} minimum — "
                 f"accumulating until it clears")
            continue
        if not _live():
            _log(f"  [{MODE.upper()}] would BUY ${amt:.2f} of {asset} (nothing sent)")
            continue
        try:
            # quoteOrderQty spends an exact dollar amount, sidestepping lot-step
            # rounding on the base asset entirely.
            r = await _retry(client.order_market_buy, symbol=symbol,
                             quoteOrderQty=f"{amt:.2f}")
            got = float(r.get("executedQty", 0))
            spent += float(r.get("cummulativeQuoteQty", amt))
            _log(f"  [BUY] 🟢 ${amt:.2f} -> {got} {asset}")
            _book({"kind": "buy", "asset": asset, "usd": amt, "qty": got, "mode": MODE})
        except Exception as e:
            _log(f"  [BUY-FAIL] {asset} ${amt:.2f}: {str(e)[:80]}")
    return spent


async def _cycle(client, cfg: dict, state: dict) -> None:
    snap = await snapshot(client, cfg)
    if snap is None:
        return
    plan = plan_buys(snap, cfg)
    if plan["deployable"] > 0:
        await execute(client, plan, cfg)
        snap = await snapshot(client, cfg) or snap
    row = ledger.record_nav(NAV_FILE, state, snap, 0.0)
    d = drift(snap, cfg)
    _log(f"nav=${snap['nav']:.2f} cash=${snap['cash']:.2f} growth=${row['growth']:+.2f}  "
         + " ".join(f"{a} ${v['usd']:.2f}({v['now_pct']:.0f}/{v['target_pct']:.0f}%)"
                    for a, v in d.items()))
    ledger.save_state(STATE, state)


async def cmd_plan() -> None:
    from binance import AsyncClient
    cfg = load_targets()
    c = await AsyncClient.create(os.getenv("BINANCE_API_KEY"), os.getenv("BINANCE_API_SECRET"))
    try:
        snap = await snapshot(c, cfg)
        if not snap:
            print("balances unreadable")
            return
        plan = plan_buys(snap, cfg)
        d = drift(snap, cfg)
        print("=" * 62)
        print(f"HALAL ALLOCATOR — ownership only, no interest   (MODE={MODE})")
        print("=" * 62)
        print(f"  NAV ${snap['nav']:.2f}   invested ${snap['invested']:.2f}   cash ${snap['cash']:.2f}")
        print(f"  {'asset':<7}{'held':>10}{'now':>8}{'target':>8}{'BUY':>10}")
        for a, v in d.items():
            print(f"  {a:<7}${v['usd']:>9.2f}{v['now_pct']:>7.0f}%{v['target_pct']:>7.0f}%"
                  f"${plan['alloc'].get(a, 0):>9.2f}")
        print(f"\n  deployable ${plan['deployable']:.2f} (cash minus ${CASH_BUFFER:.2f} buffer)")
        print("  BUY ONLY. Nothing is ever sold to rebalance.")
        if not _live():
            print(f"  Not armed: MODE={MODE}, arm file {'present' if os.path.exists(ARM_FILE) else 'absent'}")
    finally:
        await c.close_connection()


async def cmd_exit_earn() -> None:
    """Show exactly what leaving Simple Earn would redeem. Prints only."""
    from binance import AsyncClient
    c = await AsyncClient.create(os.getenv("BINANCE_API_KEY"), os.getenv("BINANCE_API_SECRET"))
    try:
        pos = await c.get_simple_earn_flexible_product_position()
        rows = pos.get("rows", []) if isinstance(pos, dict) else pos
        if not rows:
            print("Nothing in Simple Earn. Already out.")
            return
        print("Currently LENT OUT in Simple Earn (this is the interest to exit):")
        for r in rows:
            print(f"   {r['asset']:<6} {float(r['totalAmount']):>10.4f}   productId={r.get('productId')}")
        print("\nRun with --yes to redeem these to spot, then `plan` to deploy them.")
    finally:
        await c.close_connection()


async def run() -> None:
    global _lock
    from binance import AsyncClient
    _lock = ledger.acquire_lock(PIDFILE)
    if _lock is None:
        print(f"[halal] another instance holds {PIDFILE}")
        return
    cfg = load_targets()
    c = await AsyncClient.create(os.getenv("BINANCE_API_KEY"), os.getenv("BINANCE_API_SECRET"))
    state = ledger.load_state(STATE, {})
    _log(f"start — MODE={MODE} {'🟢 LIVE (real buys)' if _live() else '📝 simulated'}")
    _log("OWNERSHIP ONLY: no lending, no interest, no leverage, no shorting, no selling.")
    _log("targets: " + ", ".join(f"{a} {w*100:.0f}%" for a, w in cfg["_w"].items()))
    try:
        while True:
            if os.path.exists(KILL_FILE):
                _log("kill file present — idling")
                await asyncio.sleep(POLL_MIN * 60)
                continue
            try:
                cfg = load_targets()          # pick up operator edits live
                await asyncio.wait_for(_cycle(c, cfg, state), timeout=CYCLE_TIMEOUT_S)
            except asyncio.TimeoutError:
                _log(f"  cycle exceeded {CYCLE_TIMEOUT_S:.0f}s — abandoned, retrying next poll")
            except Exception as e:
                _log(f"  [ERROR] {str(e)[:120]}")
            await asyncio.sleep(POLL_MIN * 60)
    finally:
        await c.close_connection()


if __name__ == "__main__":
    arg = (sys.argv[1:] or [""])[0]
    if arg == "plan":
        asyncio.run(cmd_plan())
    elif arg == "report":
        ledger.print_nav_report(NAV_FILE, "HALAL ALLOCATOR")
    elif arg == "exit-earn":
        asyncio.run(cmd_exit_earn())
    else:
        asyncio.run(run())
