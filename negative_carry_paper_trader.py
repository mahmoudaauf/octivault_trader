#!/usr/bin/env python3
"""
Negative-funding carry PAPER trader — forward proof-of-edge engine, step 1 of the
validation ladder (backtest -> paper -> testnet -> armed) for the ONLY slice of
funding-carry that carries real edge: negative funding.

WHY THIS EXISTS
---------------
carry_paper_trader.py / carry_negative_funding_backtest.py established:
  - Positive-only carry (the live daemon) is FALSIFIED at every threshold —
    see docs/carry_frontier_findings.md. It fishes in the 1.6% of funding
    extremes that are positive; ~98% of the real opportunity is negative
    funding, which needs LONG PERP + SHORT SPOT (borrow the asset, sell it).
  - Real Binance cross-margin borrow costs run ~0.04-0.18%/8h on the alts
    where this concentrates — at the 6bps entry threshold that's enough to
    flip the win rate to 37%. The edge only survives at a HIGHER threshold
    (measured: ~20bps + $5M liquidity -> +1.90%/trade, 82% win, n=109 over
    ~166d) where the funding collected clearly outruns the borrow cost.

This script is PAPER ONLY. It does not borrow anything or place any order —
it books what a real position WOULD have earned, using REAL settled funding
and REAL, freshly-fetched borrow rates (not a static backtest snapshot), so
the forward record is honest about rates moving. A borrow-rate spike during
the exact panic this strategy trades is the single biggest risk the backtest
could not model; this daemon measures it in real time instead of assuming it
away.

UPDATE 2026-07-20 — a second, bigger real-world gap found and fixed: every
symbol this daemon had ever "opened" (both closed trades, all open positions)
had ZERO real supply in Binance's cross-margin lending pool when checked live
via get_max_margin_loan(). isBorrowable=True is a static asset property, not
a live inventory check — the pool empties out exactly where this strategy's
edge concentrates (illiquid coins everyone else also wants to short during a
panic). Entries now check real live borrow supply before "opening" a paper
position (_borrow_supply_available()), so the forward record only counts
trades that were genuinely executable at that moment.

Mechanics (mirrors carry_paper_trader.py's structure, opposite direction):
  funding very negative -> shorts pay longs -> be LONG perp (receive funding)
  hedge by SHORTING spot -> borrow the asset, sell it, repay + buy back to close
  net per trade = signed_funding_collected - borrow_interest_accrued - fees

Usage:
  python3 negative_carry_paper_trader.py            # run the paper daemon
  python3 negative_carry_paper_trader.py report      # print the live edge verdict

Env: NEG_CARRY_ENTRY_BPS(20) NEG_CARRY_EXIT_BPS(1) NEG_CARRY_FEE_RT_PCT(0.24)
     NEG_CARRY_MIN_VOL_USD(5e6) NEG_CARRY_MAX_HOLD_H(360) NEG_CARRY_POLL_MIN(30)
     NEG_CARRY_NAV_FRACTION(0.02) NEG_CARRY_MIN_NOTIONAL(10) NEG_CARRY_MAX_NOTIONAL(1000)
     NEG_CARRY_MAX_POSITIONS(5) NEG_CARRY_MAX_TOTAL_USD(5000)
     NEG_CARRY_MAX_DD_PCT(5.0) NEG_CARRY_KILL_FILE(logs/neg_carry.stop)
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

STATE = "logs/neg_carry_state.json"
LEDGER = "logs/neg_carry_ledger.jsonl"

# Entry defaults to 20bps, NOT 6bps like the positive-only daemon. Measured
# (carry_negative_funding_backtest.py): at 6bps, real borrow cost cancels the
# funding collected at the margin (37% win). The edge only clears above
# ~10-20bps where the tail is fat enough to outrun borrow cost.
ENTRY = float(os.getenv("NEG_CARRY_ENTRY_BPS", "20")) / 10000.0
EXIT = float(os.getenv("NEG_CARRY_EXIT_BPS", "1")) / 10000.0
FEE_RT = float(os.getenv("NEG_CARRY_FEE_RT_PCT", "0.24")) / 100.0
MIN_VOL = float(os.getenv("NEG_CARRY_MIN_VOL_USD", "5000000"))
MAX_HOLD_H = float(os.getenv("NEG_CARRY_MAX_HOLD_H", "360"))
POLL_MIN = float(os.getenv("NEG_CARRY_POLL_MIN", "30"))

_NOTIONAL_OVERRIDE = os.getenv("NEG_CARRY_NOTIONAL")
NOTIONAL = float(_NOTIONAL_OVERRIDE) if _NOTIONAL_OVERRIDE else 1000.0
NAV_FRACTION = float(os.getenv("NEG_CARRY_NAV_FRACTION", "0.02"))
MIN_NOTIONAL = float(os.getenv("NEG_CARRY_MIN_NOTIONAL", "10"))
MAX_NOTIONAL = float(os.getenv("NEG_CARRY_MAX_NOTIONAL", "1000"))
MIN_TRADES = 30

# Paper only. The live execution path (borrow -> short spot -> repay) has not
# been built or testnet-validated — that is the NEXT rung of the ladder, only
# after this forward record proves out. Do not add a live path here without
# testnet validation first, matching the discipline used for positive-only
# carry (which WAS testnet-validated before any live gate existed).
MODE = "paper"
MAX_POS = int(os.getenv("NEG_CARRY_MAX_POSITIONS", "5"))
MAX_TOTAL_USD = float(os.getenv("NEG_CARRY_MAX_TOTAL_USD", "5000"))
KILL_FILE = os.getenv("NEG_CARRY_KILL_FILE", "logs/neg_carry.stop")
MAX_DD_PCT = float(os.getenv("NEG_CARRY_MAX_DD_PCT", "5.0"))

# Borrow rates move; re-fetch per symbol at most this often instead of once
# at startup (which is what the backtest did, and flagged as its single most
# optimistic assumption — rates rise exactly when a coin is in demand to
# short, i.e. during the negative-funding spike this strategy trades).
_BORROW_RATE_TTL_S = 900.0
_borrow_cache: dict[str, tuple[float, float]] = {}  # asset -> (hourly_rate, fetched_ts)


def _killed() -> bool:
    return os.path.exists(KILL_FILE)


def _current_drawdown_pct() -> float:
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
    if _NOTIONAL_OVERRIDE:
        return float(_NOTIONAL_OVERRIDE), "explicit NEG_CARRY_NOTIONAL override"
    try:
        bal = await client.get_asset_balance(asset="USDT")
        free_usdt = float((bal or {}).get("free", 0.0) or 0.0)
    except Exception as e:
        print(f"[neg-carry] free-USDT fetch failed ({str(e)[:60]}) — using flat ${MAX_NOTIONAL:.0f}/leg")
        return MAX_NOTIONAL, "fallback (balance fetch failed)"
    notional = max(MIN_NOTIONAL, min(free_usdt * NAV_FRACTION, MAX_NOTIONAL))
    return notional, f"NAV-aware ({NAV_FRACTION*100:.1f}% of ${free_usdt:.2f} free USDT)"


async def build_universe(client) -> set[str]:
    """USDT perps that ALSO have spot (hedgeable), are margin-BORROWABLE (the
    part that's actually new here), and clear the liquidity floor."""
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
    candidates = {s for s in perps if s in spot_syms and vol.get(s, 0) >= MIN_VOL}

    borrowable = set()
    for sym in candidates:
        asset = sym[:-4]  # strip USDT
        try:
            info_a = await client.get_margin_asset(asset=asset)
            if str(info_a.get("isBorrowable", False)).lower() == "true":
                borrowable.add(sym)
        except Exception:
            pass  # not on cross-margin -> not tradeable this way
        await asyncio.sleep(0.08)
    return borrowable


async def current_funding(client, symbols: set[str]) -> dict[str, float]:
    out = {}
    try:
        prem = await client.futures_mark_price()
        for p in prem:
            s = p.get("symbol")
            if s in symbols:
                out[s] = float(p.get("lastFundingRate", 0) or 0)
    except Exception as e:
        print(f"[neg-carry] funding fetch failed: {str(e)[:60]}")
    return out


async def signed_settled_funding_long(client, symbol: str, since_ms: int) -> float:
    """Net funding P&L for a LONG PERP holder since entry — signed, so a
    mid-hold flip to positive funding correctly shows as a cost, not ignored.
    (Contrast carry_paper_trader.py's settled_funding(), which sums abs() —
    correct only for a position that never flips sign against it.)"""
    try:
        rows = await client.futures_funding_rate(symbol=symbol, startTime=since_ms, limit=1000)
        return sum(-float(r["fundingRate"]) for r in rows)
    except Exception:
        return 0.0


async def _hourly_borrow_rate(client, asset: str) -> float:
    """Real, currently-forecast hourly interest rate to borrow `asset` on
    cross-margin. Cached for _BORROW_RATE_TTL_S so a live panic (rate spikes
    exactly when everyone wants to short) shows up within ~15 minutes instead
    of never, unlike a startup-only snapshot."""
    now = time.time()
    cached = _borrow_cache.get(asset)
    if cached and (now - cached[1]) < _BORROW_RATE_TTL_S:
        return cached[0]
    try:
        r = await client.margin_next_hourly_interest_rate(assets=asset, isIsolated=False)
        rate = float(r[0]["nextHourlyInterestRate"]) if r else 0.0
    except Exception:
        rate = cached[0] if cached else 0.0  # stale-but-real beats silently zero
    _borrow_cache[asset] = (rate, now)
    return rate


_supply_cache: dict[str, tuple[bool, float]] = {}  # asset -> (has_supply, checked_ts)
_SUPPLY_CACHE_TTL_S = 300.0  # supply can empty out fast during a panic; check often


async def _borrow_supply_available(client, asset: str, needed_qty: float) -> bool:
    """Real, live check: is there actually enough of `asset` left in
    Binance's cross-margin lending pool to borrow `needed_qty` right now?

    Found empirically (2026-07-20): every symbol this daemon has ever
    "opened" a paper position in had ZERO available supply when checked live
    — isBorrowable=True is a static asset property, not a live inventory
    check. The lending pool empties out exactly where this strategy's edge
    concentrates (illiquid coins everyone else also wants to short during a
    panic), so a paper record that assumes infinite supply silently overstates
    how often this strategy is actually executable. This check makes the
    forward record honest: an entry only "opens" here if it could have
    genuinely been borrowed at that moment."""
    now = time.time()
    cached = _supply_cache.get(asset)
    if cached and (now - cached[1]) < _SUPPLY_CACHE_TTL_S:
        return cached[0]
    has_supply = False
    try:
        maxb = await client.get_max_margin_loan(asset=asset)
        has_supply = float(maxb.get("amount", 0.0) or 0.0) >= needed_qty
    except Exception:
        has_supply = False  # not borrowable / not on margin / pool empty -> can't trade it
    _supply_cache[asset] = (has_supply, now)
    return has_supply


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
        print("No closed negative-funding carry trades yet.")
        return
    nets = [t["net_pct"] for t in trades]
    n = len(nets)
    wins = sum(1 for x in nets if x > 0)
    avg = sum(nets) / n
    print("=" * 68)
    print(f"NEGATIVE-FUNDING CARRY PAPER — FORWARD TRACK RECORD ({n} closed trades)")
    print("=" * 68)
    print(f"  Avg net/trade: {avg:+.4f}%   win-rate: {wins/n*100:.0f}%   "
          f"cum: {sum(nets):+.2f}%")
    if n < MIN_TRADES:
        print(f"  VERDICT: ⏳ INCONCLUSIVE — {n}/{MIN_TRADES} trades. Keep running.")
    elif avg > 0:
        print(f"  VERDICT: ✅ FORWARD EDGE CONFIRMED — testnet-validate the borrow/short/repay "
              f"cycle next, do NOT skip to live capital.")
    else:
        print(f"  VERDICT: ❌ NO FORWARD EDGE — do NOT deploy capital.")
    print("=" * 68)


async def run():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from binance import AsyncClient

    client = await AsyncClient.create(
        os.getenv("BINANCE_API_KEY") or "x", os.getenv("BINANCE_API_SECRET") or "x"
    )
    global NOTIONAL
    state = _load(STATE, {"open": {}})
    print(f"[neg-carry] start — MODE=paper (borrow/short/repay NOT yet built — "
          f"this is step 1 of the ladder: forward proof-of-edge only)")
    print(f"[neg-carry] safety — max_pos={MAX_POS} max_total=${MAX_TOTAL_USD:.0f} "
          f"auto_halt_drawdown={MAX_DD_PCT}% kill_file={KILL_FILE}")
    NOTIONAL, _notional_reason = await _resolve_notional(client)
    print(f"[neg-carry] params — entry<=-{ENTRY*100:.3f}% exit<{EXIT*100:.3f}% "
          f"fee={FEE_RT*100:.2f}% notional=${NOTIONAL:.2f}/leg ({_notional_reason}) "
          f"poll={POLL_MIN}m borrow_rate_ttl={_BORROW_RATE_TTL_S/60:.0f}m")
    try:
        universe = await build_universe(client)
        print(f"[neg-carry] universe: {len(universe)} liquid, spot-hedgeable, "
              f"margin-BORROWABLE perps")
        while True:
            now = time.time()
            funding = await current_funding(client, universe)
            opened = closed = 0

            for sym, pos in list(state["open"].items()):
                fr = funding.get(sym, 0.0)
                held_h = (now - pos["entry_ts"]) / 3600.0
                asset = sym[:-4]
                hourly_rate = await _hourly_borrow_rate(client, asset)
                # Accrue borrow interest for the elapsed interval at the
                # CURRENT rate (real-time, not a startup snapshot).
                last_check = pos.get("last_check_ts", pos["entry_ts"])
                elapsed_h = max(0.0, (now - last_check) / 3600.0)
                pos["accrued_borrow_pct"] = pos.get("accrued_borrow_pct", 0.0) + hourly_rate * elapsed_h
                pos["last_check_ts"] = now

                if abs(fr) < EXIT or held_h >= MAX_HOLD_H or _killed():
                    signed_funding = await signed_settled_funding_long(
                        client, sym, int(pos["entry_ts"] * 1000))
                    net = (signed_funding - pos["accrued_borrow_pct"] - FEE_RT) * 100.0
                    _log_trade({
                        "ts": datetime.now(timezone.utc).isoformat(), "symbol": sym,
                        "held_h": round(held_h, 1),
                        "signed_funding_pct": round(signed_funding * 100, 4),
                        "borrow_cost_pct": round(pos["accrued_borrow_pct"] * 100, 4),
                        "net_pct": round(net, 4), "entry_funding": pos["entry_funding"],
                        "exit_funding": fr, "mode": "paper",
                    })
                    del state["open"][sym]
                    closed += 1

            dd = _current_drawdown_pct()
            if dd >= MAX_DD_PCT and not _killed():
                open(KILL_FILE, "w").close()
                print(f"[neg-carry] 🛑 DRAWDOWN HALT: {dd:.2f}% >= {MAX_DD_PCT}% — "
                      f"kill-switch engaged, no new entries.")
            _save(STATE, state)

            if not _killed():
                for sym, fr in funding.items():
                    if sym in state["open"] or fr > -ENTRY:  # negative-funding ONLY
                        continue
                    if len(state["open"]) >= MAX_POS:
                        break
                    if (len(state["open"]) + 1) * NOTIONAL > MAX_TOTAL_USD:
                        break
                    asset = sym[:-4]
                    try:
                        mark = await client.futures_mark_price(symbol=sym)
                        qty_needed = NOTIONAL / float(mark["markPrice"])
                    except Exception:
                        continue  # can't size it -> can't check supply -> skip
                    if not await _borrow_supply_available(client, asset, qty_needed):
                        print(f"  [SKIP-NO-SUPPLY] {sym}: funding {fr*100:+.3f}% qualifies but "
                              f"the {asset} lending pool has no supply to borrow right now")
                        continue
                    state["open"][sym] = {
                        "entry_ts": now, "entry_funding": fr,
                        "last_check_ts": now, "accrued_borrow_pct": 0.0,
                    }
                    opened += 1
            _save(STATE, state)

            ts = datetime.now(timezone.utc).strftime("%H:%M")
            _vals = list(funding.values()) or [0.0]
            _best_neg = min(_vals) if _vals else 0.0
            print(f"[neg-carry {ts}] open={len(state['open'])} opened={opened} closed={closed} "
                  f"| most_negative={_best_neg*100:.3f}% (entry<=-{ENTRY*100:.3f}%)")
            await asyncio.sleep(POLL_MIN * 60)
    finally:
        await client.close_connection()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        _report()
    else:
        asyncio.run(run())
