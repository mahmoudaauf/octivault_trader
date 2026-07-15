#!/usr/bin/env python3
"""
Funding-carry PAPER trader — forward proof-of-edge engine.

Runs the validated delta-neutral funding-carry strategy LIVE on paper so you build a
real forward track record (backtests can lie; forward can't). When |funding| is
extreme it "opens" a delta-neutral position (paper), accrues the REAL settled funding
each 8h window, and "closes" when funding normalises — booking net = funding − fees.
Every closed trade is logged to carry_ledger.jsonl; run `carry_paper_trader.py report`
for the live edge verdict.

When the forward record is conclusively positive, this same logic flips to live by
replacing the paper open/close with real perp-short + spot-long orders.

Usage:
  python3 carry_paper_trader.py            # run the paper trader (daemon)
  python3 carry_paper_trader.py report     # print the live edge verdict and exit
Env: CARRY_ENTRY_BPS(6) CARRY_EXIT_BPS(1) CARRY_FEE_RT_PCT(0.24)
     CARRY_MIN_VOL_USD(50e6) CARRY_MAX_HOLD_H(360) CARRY_POLL_MIN(30)
     CARRY_NOTIONAL (explicit override, $/leg -- if unset, sizing is NAV-aware:
     see CARRY_NAV_FRACTION/CARRY_MIN_NOTIONAL/CARRY_MAX_NOTIONAL below)
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

STATE = "logs/carry_state.json"
LEDGER = "logs/carry_ledger.jsonl"

ENTRY = float(os.getenv("CARRY_ENTRY_BPS", "6")) / 10000.0
EXIT = float(os.getenv("CARRY_EXIT_BPS", "1")) / 10000.0
FEE_RT = float(os.getenv("CARRY_FEE_RT_PCT", "0.24")) / 100.0
MIN_VOL = float(os.getenv("CARRY_MIN_VOL_USD", "50000000"))
MAX_HOLD_H = float(os.getenv("CARRY_MAX_HOLD_H", "360"))      # 15 days
POLL_MIN = float(os.getenv("CARRY_POLL_MIN", "30"))
# NOTIONAL ($/leg): a flat, NAV-independent constant was unrealistic for a real
# account (e.g. $1000/leg on a $58 account is ~17x the whole account). If
# CARRY_NOTIONAL is explicitly set, that override wins as-is (unchanged
# behavior). Otherwise sizing is NAV-aware -- see _resolve_notional() -- based
# on the account's real free USDT balance, fetched once at daemon startup.
_NOTIONAL_OVERRIDE = os.getenv("CARRY_NOTIONAL")
NOTIONAL = float(_NOTIONAL_OVERRIDE) if _NOTIONAL_OVERRIDE else 1000.0  # $ per leg (placeholder until resolved)
CARRY_NAV_FRACTION = float(os.getenv("CARRY_NAV_FRACTION", "0.02"))  # 2% of free USDT per leg
CARRY_MIN_NOTIONAL = float(os.getenv("CARRY_MIN_NOTIONAL", "10"))    # exchange-realistic floor
CARRY_MAX_NOTIONAL = float(os.getenv("CARRY_MAX_NOTIONAL", "1000"))  # upper cap (old default's value)
MIN_TRADES = 30

# ── Execution mode + safety ────────────────────────────────────────────────────
# paper  : simulate only (default) — builds the forward proof, zero orders.
# dryrun : compute + LOG the exact real orders it WOULD place (sizing, legs, safety),
#          but send nothing. Fully testable with no risk.
# live   : place REAL perp+spot orders — DOUBLE-GATED: requires MODE=live AND the arm
#          file to exist. v1 trades POSITIVE funding only (short perp + long spot — no
#          spot-margin shorting). NOTE: the live path must be validated on TESTNET
#          before real capital; until then keep MODE in {paper,dryrun}.
MODE = os.getenv("CARRY_MODE", "paper").lower()
POSITIVE_ONLY = os.getenv("CARRY_POSITIVE_ONLY", "true").lower() == "true"
MAX_POS = int(os.getenv("CARRY_MAX_POSITIONS", "5"))           # concurrency cap
MAX_TOTAL_USD = float(os.getenv("CARRY_MAX_TOTAL_USD", "5000"))  # total deployed cap
LEVERAGE = int(os.getenv("CARRY_LEVERAGE", "2"))              # low perp leverage
KILL_FILE = os.getenv("CARRY_KILL_FILE", "logs/carry.stop")   # touch to halt new opens
LIVE_ARM_FILE = os.getenv("CARRY_LIVE_ARM_FILE", "logs/carry_live_armed")  # 2nd gate
MAX_DD_PCT = float(os.getenv("CARRY_MAX_DD_PCT", "5.0"))      # auto-halt at this drawdown
LIQ_BUFFER_PCT = float(os.getenv("CARRY_LIQ_BUFFER_PCT", "15.0"))  # close if this close to liq


def _killed() -> bool:
    return os.path.exists(KILL_FILE)


def _current_drawdown_pct() -> float:
    """Current drawdown (%) below the peak of the cumulative net-P&L equity curve."""
    if not os.path.exists(LEDGER):
        return 0.0
    cum = peak = 0.0
    for line in open(LEDGER):
        line = line.strip()
        if not line:
            continue
        try:
            cum += float(json.loads(line).get("net_pct", 0.0))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        peak = max(peak, cum)
    return max(0.0, peak - cum)


async def safety_checks(client, state: dict) -> None:
    """Drawdown auto-halt (all modes) + liquidation-buffer guard (live only)."""
    # Drawdown auto-halt → engage the kill-switch (which closes all + stops new opens).
    dd = _current_drawdown_pct()
    if dd >= MAX_DD_PCT and not _killed():
        open(KILL_FILE, "w").close()
        print(f"[carry] 🛑 DRAWDOWN HALT: {dd:.2f}% ≥ {MAX_DD_PCT}% — kill-switch engaged, "
              f"closing all and stopping new entries.")
    # Liquidation-buffer guard (live perp legs only).
    if MODE == "live":
        for sym in list(state.get("open", {})):
            try:
                pos = await client.futures_position_information(symbol=sym)
                p = next((x for x in pos if x["symbol"] == sym), None)
                if not p or float(p.get("positionAmt", 0) or 0) == 0:
                    continue
                liq = float(p.get("liquidationPrice", 0) or 0)
                mark = float(p.get("markPrice", 0) or 0)
                if liq > 0 and mark > 0:
                    buf = abs(mark - liq) / mark * 100.0
                    if buf < LIQ_BUFFER_PCT:
                        print(f"[carry] ⚠️  {sym} within {buf:.1f}% of liquidation "
                              f"(< {LIQ_BUFFER_PCT}%) — force-closing leg.")
                        await execute_legs(client, sym,
                                           state["open"][sym].get("entry_funding", 0.0), "close")
                        state["open"].pop(sym, None)
            except Exception:
                pass


async def execute_legs(client, symbol: str, funding: float, action: str) -> bool:
    """Place/sim the delta-neutral leg pair. action ∈ {open, close}.
    Returns True if the (paper/dryrun/live) action is considered done.

    Positive funding (longs pay shorts) → SHORT perp + LONG spot (collect funding).
    v1 skips negative funding in dryrun/live (needs spot-margin shorting).
    """
    if action == "open" and POSITIVE_ONLY and funding <= 0:
        return False  # skip — negative-funding carry needs margin shorting (v1: off)

    if MODE == "paper":
        return True

    perp_side = ("SELL" if funding > 0 else "BUY") if action == "open" else \
                ("BUY" if funding > 0 else "SELL")
    spot_side = ("BUY") if action == "open" else ("SELL")
    desc = (f"{action.upper()} {symbol}: perp {perp_side} + spot {spot_side} "
            f"(~${NOTIONAL:.0f}/leg, {LEVERAGE}x)")

    if MODE == "dryrun":
        print(f"  [DRYRUN] {desc}  funding={funding*100:.3f}%")
        return True

    if MODE == "live":
        if not os.path.exists(LIVE_ARM_FILE):
            print(f"  [LIVE-BLOCKED] {symbol}: arm file '{LIVE_ARM_FILE}' missing — no order sent")
            return False
        # Real orders — gated. (Validate on TESTNET before real capital.)
        try:
            if action == "open":
                await client.futures_change_leverage(symbol=symbol, leverage=LEVERAGE)
            mark = await client.futures_mark_price(symbol=symbol)
            px = float(mark["markPrice"])
            qty = round(NOTIONAL / px, 6)
            await client.futures_create_order(
                symbol=symbol, side=perp_side, type="MARKET", quantity=qty)
            await client.create_order(
                symbol=symbol, side=spot_side, type="MARKET", quantity=qty)
            print(f"  [LIVE] {desc} @ ~{px}")
            return True
        except Exception as e:
            print(f"  [LIVE-ERROR] {symbol} {action}: {str(e)[:80]}")
            return False
    return False


def _load(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _save(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def _log_trade(rec: dict):
    with open(LEDGER, "a") as f:
        f.write(json.dumps(rec) + "\n")


async def _resolve_notional(client) -> tuple[float, str]:
    """Resolve $/leg sizing. Explicit CARRY_NOTIONAL always wins (unchanged
    behavior). Otherwise, size off the account's real free USDT balance --
    this is a paper daemon so no capital actually moves, but the forward-proof
    run should reflect what a real deployment would look like at THIS
    account's actual size, not an arbitrary flat dollar figure. Falls back to
    CARRY_MAX_NOTIONAL (the old flat default) if the balance fetch fails, so a
    transient API error can never silently zero out sizing."""
    if _NOTIONAL_OVERRIDE:
        return float(_NOTIONAL_OVERRIDE), "explicit CARRY_NOTIONAL override"
    try:
        bal = await client.get_asset_balance(asset="USDT")
        free_usdt = float((bal or {}).get("free", 0.0) or 0.0)
    except Exception as e:
        print(f"[carry] ⚠️  free-USDT fetch failed ({str(e)[:60]}) — using flat ${CARRY_MAX_NOTIONAL:.0f}/leg fallback")
        return CARRY_MAX_NOTIONAL, "fallback (balance fetch failed)"
    notional = max(CARRY_MIN_NOTIONAL, min(free_usdt * CARRY_NAV_FRACTION, CARRY_MAX_NOTIONAL))
    return notional, f"NAV-aware ({CARRY_NAV_FRACTION*100:.1f}% of ${free_usdt:.2f} free USDT)"


async def build_universe(client) -> set[str]:
    """USDT perps that ALSO have spot (hedgeable) AND >= MIN_VOL/24h (tradeable)."""
    info = await client.futures_exchange_info()
    perps = {
        s["symbol"] for s in info["symbols"]
        if s.get("quoteAsset") == "USDT" and s.get("contractType") == "PERPETUAL"
        and s.get("status") == "TRADING"
    }
    spot = await client.get_exchange_info()
    spot_syms = {s["symbol"] for s in spot["symbols"] if s.get("status") == "TRADING"}
    tick = await client.futures_ticker()
    vol = {t["symbol"]: float(t.get("quoteVolume", 0) or 0) for t in tick}
    return {s for s in perps if s in spot_syms and vol.get(s, 0) >= MIN_VOL}


async def current_funding(client, symbols: set[str]) -> dict[str, float]:
    """Latest funding rate per symbol from futures premium index (one call)."""
    out = {}
    try:
        prem = await client.futures_mark_price()  # list, all symbols
        for p in prem:
            s = p.get("symbol")
            if s in symbols:
                out[s] = float(p.get("lastFundingRate", 0) or 0)
    except Exception as e:
        print(f"funding fetch failed: {str(e)[:60]}")
    return out


async def settled_funding(client, symbol: str, since_ms: int) -> float:
    """Sum of |funding rates| actually settled for symbol since entry (real cash flow)."""
    try:
        rows = await client.futures_funding_rate(symbol=symbol, startTime=since_ms, limit=1000)
        return sum(abs(float(r["fundingRate"])) for r in rows)
    except Exception:
        return 0.0


def _report():
    trades = []
    for line in open(LEDGER) if os.path.exists(LEDGER) else []:
        line = line.strip()
        if line:
            try:
                trades.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    if not trades:
        print("No closed carry trades yet.")
        return
    nets = [t["net_pct"] for t in trades]
    n = len(nets)
    wins = sum(1 for x in nets if x > 0)
    avg = sum(nets) / n
    print("=" * 60)
    print(f"CARRY PAPER — FORWARD TRACK RECORD ({n} closed trades)")
    print("=" * 60)
    print(f"  Avg net/trade: {avg:+.4f}%   win-rate: {wins/n*100:.0f}%   "
          f"cum: {sum(nets):+.2f}%")
    if n < MIN_TRADES:
        print(f"  VERDICT: ⏳ INCONCLUSIVE — {n}/{MIN_TRADES} trades. Keep running.")
    elif avg > 0:
        print(f"  VERDICT: ✅ FORWARD EDGE CONFIRMED — deploy-capital candidate.")
    else:
        print(f"  VERDICT: ❌ NO FORWARD EDGE — do NOT deploy capital.")
    print("=" * 60)


async def run():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from binance import AsyncClient

    # Testnet validation: point the live order path at Binance FUTURES testnet (fake
    # money, real order mechanics) to prove execution before any real capital. Needs
    # separate testnet keys (create at testnet.binancefuture.com). NOTE: Binance spot &
    # futures testnets are separate systems — this validates the FUTURES/perp leg (the
    # risky one: leverage/liquidation/funding). The spot leg is low-risk market buys.
    testnet = os.getenv("CARRY_TESTNET", "false").lower() in ("1", "true", "yes")
    if testnet:
        key = os.getenv("CARRY_TESTNET_KEY") or ""
        sec = os.getenv("CARRY_TESTNET_SECRET") or ""
        if not key or not sec:
            print("[carry] ⚠️  CARRY_TESTNET=true but CARRY_TESTNET_KEY/SECRET not set — "
                  "create keys at testnet.binancefuture.com. Aborting.")
            return
        client = await AsyncClient.create(key, sec, testnet=True)
    else:
        client = await AsyncClient.create(
            os.getenv("BINANCE_API_KEY") or "x", os.getenv("BINANCE_API_SECRET") or "x"
        )
    global NOTIONAL
    state = _load(STATE, {"open": {}})
    armed = MODE == "live" and os.path.exists(LIVE_ARM_FILE)
    banner = {"paper": "📝 PAPER (no orders)", "dryrun": "🧪 DRY-RUN (logs orders, sends none)",
              "live": ("🔴 LIVE-ARMED (REAL ORDERS)" if armed else "🔴 live (BLOCKED — arm file missing)")}
    print(f"[carry] start — MODE={MODE} → {banner.get(MODE, MODE)}")
    print(f"[carry] safety — positive_only={POSITIVE_ONLY} max_pos={MAX_POS} "
          f"max_total=${MAX_TOTAL_USD:.0f} leverage={LEVERAGE}x")
    print(f"[carry] guards — auto_halt_drawdown={MAX_DD_PCT}% liq_buffer={LIQ_BUFFER_PCT}% "
          f"kill_file={KILL_FILE}")
    NOTIONAL, _notional_reason = await _resolve_notional(client)
    print(f"[carry] params — entry|f|>={ENTRY*100:.3f}% exit<{EXIT*100:.3f}% "
          f"fee={FEE_RT*100:.2f}% notional=${NOTIONAL:.2f}/leg ({_notional_reason}) poll={POLL_MIN}m")
    if MODE == "live" and not armed:
        print(f"[carry] ⚠️  MODE=live but '{LIVE_ARM_FILE}' absent — will NOT send orders.")
    try:
        universe = await build_universe(client)
        print(f"[carry] universe: {len(universe)} liquid spot-hedgeable perps")
        while True:
            now = time.time()
            funding = await current_funding(client, universe)
            opened = closed = 0
            # Manage open positions (exit + close legs).
            for sym, pos in list(state["open"].items()):
                fr = funding.get(sym, 0.0)
                held_h = (now - pos["entry_ts"]) / 3600.0
                if abs(fr) < EXIT or held_h >= MAX_HOLD_H or _killed():
                    await execute_legs(client, sym, pos.get("entry_funding", fr), "close")
                    accrued = await settled_funding(client, sym, int(pos["entry_ts"] * 1000))
                    net = (accrued - FEE_RT) * 100.0
                    _log_trade({
                        "ts": datetime.now(timezone.utc).isoformat(), "symbol": sym,
                        "held_h": round(held_h, 1), "accrued_funding_pct": round(accrued * 100, 4),
                        "net_pct": round(net, 4), "exit_funding": fr, "mode": MODE,
                    })
                    del state["open"][sym]
                    closed += 1
            # Safety: drawdown auto-halt + liquidation-buffer guard.
            await safety_checks(client, state)
            _save(STATE, state)
            # Open new positions on extreme funding — subject to safety gates.
            if not _killed():
                for sym, fr in funding.items():
                    if sym in state["open"] or abs(fr) < ENTRY:
                        continue
                    if len(state["open"]) >= MAX_POS:
                        break  # concurrency cap
                    if (len(state["open"]) + 1) * NOTIONAL > MAX_TOTAL_USD:
                        break  # total-deployed cap
                    if not await execute_legs(client, sym, fr, "open"):
                        continue  # skipped (negative funding in v1, or live-blocked)
                    state["open"][sym] = {
                        "entry_ts": now, "entry_funding": fr,
                        "dir": "short_perp" if fr > 0 else "long_perp",
                    }
                    opened += 1
            _save(STATE, state)
            ts = datetime.now(timezone.utc).strftime("%H:%M")
            print(f"[carry {ts}] open={len(state['open'])} opened={opened} closed={closed} "
                  f"| max_funding={max([abs(v) for v in funding.values()] or [0])*100:.3f}%")
            await asyncio.sleep(POLL_MIN * 60)
    finally:
        await client.close_connection()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        _report()
    else:
        asyncio.run(run())
