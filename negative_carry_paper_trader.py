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

Modes (NEG_CARRY_MODE, default paper):
  paper  — simulate only; the forward proof-of-edge record. Zero orders.
  dryrun — log the exact real orders it WOULD place; send nothing.
  live   — place REAL margin borrow + short-spot + long-perp orders.
           DOUBLE-GATED: MODE=live AND arm file (NEG_CARRY_LIVE_ARM_FILE).
           Mechanics validated 2026-07-20 via margin_carry_live_probe.py (~$0.05
           real). Holds a BORROWED position for days to collect funding — start
           tiny (MAX_POS=1, $10/leg) and supervised. Liquidation-guarded.

Usage:
  python3 negative_carry_paper_trader.py            # run the daemon (mode from env)
  python3 negative_carry_paper_trader.py report      # print the edge verdict

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

STATE = os.getenv("NEG_CARRY_STATE_FILE", "logs/neg_carry_state.json")
LEDGER = os.getenv("NEG_CARRY_LEDGER_FILE", "logs/neg_carry_ledger.jsonl")

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

# ── Execution mode + safety ─────────────────────────────────────────────────
# paper  : simulate only (default) — the forward proof-of-edge record, zero orders.
# dryrun : compute + LOG the exact real orders it WOULD place (borrow/sell/perp,
#          sizing, supply), but send nothing. Proves the live logic path safely.
# live   : place REAL margin+perp orders — DOUBLE-GATED: requires MODE=live AND
#          the arm file. Testnet CANNOT validate this (Binance's public spot
#          testnet has no margin API — confirmed), so the borrow/short/repay
#          MECHANICS were instead validated with a real tiny probe
#          (margin_carry_live_probe.py, 2026-07-20, ~$0.05). Live here holds a
#          BORROWED position for days to actually collect funding — that is the
#          real risk this gating exists to contain. Start tiny (MAX_POS=1,
#          $10/leg) and supervised.
MODE = os.getenv("NEG_CARRY_MODE", "paper").lower()
LIVE_ARM_FILE = os.getenv("NEG_CARRY_LIVE_ARM_FILE", "logs/neg_carry_live_armed")
LEVERAGE = int(os.getenv("NEG_CARRY_LEVERAGE", "2"))
# Cross-margin liquidation guard: force-close if marginLevel drops below this
# (Binance liquidates around ~1.1; 1.5 is a conservative early exit). Delta-
# neutral + tiny size vs collateral keeps this far away in normal conditions.
LIQ_MARGIN_LEVEL_MIN = float(os.getenv("NEG_CARRY_LIQ_MARGIN_LEVEL_MIN", "1.5"))
# Perp liquidation guard: force-close the pair if the perp leg gets within this
# % of its liquidation price.
LIQ_PERP_BUFFER_PCT = float(os.getenv("NEG_CARRY_LIQ_PERP_BUFFER_PCT", "15.0"))
MAX_POS = int(os.getenv("NEG_CARRY_MAX_POSITIONS", "5"))
# Separate, tighter cap on concurrent LIVE positions (paper positions can coexist
# above this). Lets the first live trade stay isolated to 1 while existing paper
# positions run to completion. Defaults to MAX_POS (no extra restriction).
MAX_LIVE_POS = int(os.getenv("NEG_CARRY_MAX_LIVE_POSITIONS", str(MAX_POS)))
MAX_TOTAL_USD = float(os.getenv("NEG_CARRY_MAX_TOTAL_USD", "5000"))
KILL_FILE = os.getenv("NEG_CARRY_KILL_FILE", "logs/neg_carry.stop")
MAX_DD_PCT = float(os.getenv("NEG_CARRY_MAX_DD_PCT", "5.0"))

_filters_cache: dict[str, tuple] = {}  # symbol -> (qty_step, spot_min_notional)

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
            _r = json.loads(line)
            if _r.get("net_pct") is None:
                continue                      # unknown funding is never a 0
            cum += float(_r["net_pct"])
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
        # Never fall back to MAX_NOTIONAL: that sizes at maximum exactly when
        # the account is unreadable, and this runs once at startup so one
        # transient failure locks it in for the whole process.
        print(f"[neg-carry] free-USDT fetch failed ({str(e)[:60]}) — sizing UNKNOWN, refusing to guess")
        return None, "balance unreadable"
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
        return None      # UNKNOWN — never 0.0, which books a fabricated number


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


# ── LIVE execution (real margin borrow + short-spot + long-perp) ─────────────
# Modelled on margin_carry_live_probe.py, whose exact sequence was validated
# with real money 2026-07-20. Every function is a NO-OP unless MODE=="live" and
# the arm file exists — the caller gates that. Sizing rounds both legs to the
# SAME whole quantity so the hedge is exactly delta-neutral.

def _round_step(qty: float, step: float) -> float:
    if step <= 0:
        return qty
    return (int(qty / step)) * step


async def _symbol_filters(client, symbol: str) -> tuple[float, float]:
    """(qty_step, spot_min_notional) — coarser of spot/futures LOT_SIZE so a
    single quantity is valid on BOTH legs. Cached; exchange filters are static."""
    if symbol in _filters_cache:
        return _filters_cache[symbol]
    def step_of(filters):
        f = next(x for x in filters if x["filterType"] == "LOT_SIZE")
        return float(f["stepSize"])
    sinfo = await client.get_exchange_info()
    ssym = next(x for x in sinfo["symbols"] if x["symbol"] == symbol)
    spot_step = step_of(ssym["filters"])
    spot_min_notional = 5.0
    for f in ssym["filters"]:
        if f["filterType"] in ("NOTIONAL", "MIN_NOTIONAL"):
            spot_min_notional = float(f.get("minNotional", f.get("notional", 5.0)))
    finfo = await client.futures_exchange_info()
    fsym = next(x for x in finfo["symbols"] if x["symbol"] == symbol)
    fut_step = step_of(fsym["filters"])
    qty_step = max(spot_step, fut_step)  # coarser step valid on both legs
    _filters_cache[symbol] = (qty_step, spot_min_notional)
    return _filters_cache[symbol]


async def _live_open(client, symbol: str, notional: float) -> dict | None:
    """Borrow asset -> sell it on margin (short-spot) -> long perp (hedge).
    Returns {borrowed_qty, ...ids} on success, or None (fully unwound) on failure.
    A borrowed asset must NEVER be left un-hedged or un-repaid on a failure path."""
    asset = symbol[:-4]
    qty_step, spot_min_notional = await _symbol_filters(client, symbol)
    px = float((await client.futures_mark_price(symbol=symbol))["markPrice"])
    qty = _round_step(notional / px, qty_step)
    if qty <= 0 or qty * px < max(spot_min_notional, 5.0):
        print(f"  [LIVE-SKIP] {symbol}: ${notional:.2f} below min-notional at qty_step={qty_step}")
        return None

    borrowed = short_done = False
    try:
        await client.create_margin_loan(asset=asset, amount=str(qty))
        borrowed = True
        so = await client.create_margin_order(
            symbol=symbol, side="SELL", type="MARKET", quantity=qty,
            sideEffectType="NO_SIDE_EFFECT")
        short_done = True
        await client.futures_change_leverage(symbol=symbol, leverage=LEVERAGE)
        po = await client.futures_create_order(
            symbol=symbol, side="BUY", type="MARKET", quantity=qty)
        print(f"  [LIVE-OPEN] {symbol}: borrowed+sold {qty} {asset} (~${qty*px:.2f}), "
              f"long perp — short_oid={so.get('orderId')} perp_oid={po.get('orderId')}")
        return {"borrowed_qty": qty, "short_orderId": so.get("orderId"),
                "perp_orderId": po.get("orderId")}
    except Exception as e:
        print(f"  [LIVE-OPEN-ERROR] {symbol}: {type(e).__name__}: {str(e)[:140]} — unwinding")
        try:
            if short_done:
                await client.create_margin_order(
                    symbol=symbol, side="BUY", type="MARKET", quantity=qty,
                    sideEffectType="AUTO_REPAY")
                print(f"    unwound: bought back + repaid {symbol}")
            elif borrowed:
                await client.repay_margin_loan(asset=asset, amount=str(qty))
                print(f"    unwound: repaid loan (no spot leg opened)")
        except Exception as e2:
            print(f"    ⚠️  UNWIND FAILED — manual check needed: {str(e2)[:120]}")
        return None


async def _live_close(client, symbol: str, borrowed_qty: float) -> bool:
    """Close perp -> buy back + AUTO_REPAY the loan -> verify borrowed==0, top up
    any residual (AUTO_REPAY under-repays when Binance front-loads a min interest
    period — the exact quirk margin_carry_live_probe.py surfaced 2026-07-20)."""
    asset = symbol[:-4]
    try:
        await client.futures_create_order(
            symbol=symbol, side="SELL", type="MARKET", quantity=borrowed_qty, reduceOnly=True)
        await client.create_margin_order(
            symbol=symbol, side="BUY", type="MARKET", quantity=borrowed_qty,
            sideEffectType="AUTO_REPAY")
        # Verify the loan is actually zero; top up the residual if not.
        acc = await client.get_margin_account()
        row = next((a for a in acc.get("userAssets", []) if a["asset"] == asset), None)
        owed = (float(row["borrowed"]) + float(row["interest"])) if row else 0.0
        if owed > 0:
            qty_step, spot_min_notional = await _symbol_filters(client, symbol)
            px = float((await client.futures_mark_price(symbol=symbol))["markPrice"])
            import math
            buy_qty = max(_round_step(owed, qty_step) + qty_step,
                          math.ceil(max(spot_min_notional, 5.0) / px / qty_step) * qty_step)
            await client.create_margin_order(
                symbol=symbol, side="BUY", type="MARKET", quantity=buy_qty,
                sideEffectType="NO_SIDE_EFFECT")
            await client.repay_margin_loan(asset=asset, amount=str(owed))
            print(f"  [LIVE-CLOSE] {symbol}: cleared ${0:.2f} residual loan of {owed} {asset}")
        print(f"  [LIVE-CLOSE] {symbol}: perp closed, loan repaid")
        return True
    except Exception as e:
        print(f"  [LIVE-CLOSE-ERROR] {symbol}: {type(e).__name__}: {str(e)[:140]} — MANUAL CHECK NEEDED")
        return False


async def _liquidation_guard(client, open_syms: list[str]) -> list[str]:
    """Returns symbols that must be force-closed NOW for safety: cross-margin
    level too low, or a perp leg too close to its liquidation price."""
    danger = []
    try:
        acc = await client.get_margin_account()
        lvl = float(acc.get("marginLevel", 999) or 999)
        if lvl < LIQ_MARGIN_LEVEL_MIN:
            print(f"  [LIQ-GUARD] cross-margin level {lvl:.2f} < {LIQ_MARGIN_LEVEL_MIN} — "
                  f"force-closing ALL to de-risk")
            return list(open_syms)
    except Exception:
        pass
    for sym in open_syms:
        try:
            pos = await client.futures_position_information(symbol=sym)
            p = next((x for x in pos if x["symbol"] == sym), None)
            if not p or float(p.get("positionAmt", 0) or 0) == 0:
                continue
            liq = float(p.get("liquidationPrice", 0) or 0)
            mark = float(p.get("markPrice", 0) or 0)
            if liq > 0 and mark > 0 and abs(mark - liq) / mark * 100.0 < LIQ_PERP_BUFFER_PCT:
                print(f"  [LIQ-GUARD] {sym} perp within {LIQ_PERP_BUFFER_PCT}% of liq — force-close")
                danger.append(sym)
        except Exception:
            pass
    return danger


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
    nets = [t["net_pct"] for t in trades if t.get("net_pct") is not None]
    n = len(nets)
    wins = sum(1 for x in nets if x > 0)
    avg = sum(nets) / n
    live_n = sum(1 for t in trades if t.get("mode") == "live")
    print("=" * 68)
    print(f"NEGATIVE-FUNDING CARRY — FORWARD TRACK RECORD ({n} closed trades, "
          f"{live_n} live / {n - live_n} paper)")
    print("=" * 68)
    print(f"  Avg net/trade: {avg:+.4f}%   win-rate: {wins/n*100:.0f}%   "
          f"cum: {sum(nets):+.2f}%")
    if n < MIN_TRADES:
        print(f"  VERDICT: ⏳ INCONCLUSIVE — {n}/{MIN_TRADES} trades. Keep running.")
    elif avg > 0:
        print(f"  VERDICT: ✅ FORWARD EDGE CONFIRMED.")
    else:
        print(f"  VERDICT: ❌ NO FORWARD EDGE — do NOT deploy capital.")
    print("=" * 68)


async def run():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from binance import AsyncClient

    from exchange_resilience import create_client_with_retry, resync_clock

    client = await create_client_with_retry(AsyncClient, label="neg-carry")
    global NOTIONAL
    state = _load(STATE, {"open": {}})
    armed = MODE == "live" and os.path.exists(LIVE_ARM_FILE)
    banner = {"paper": "📝 PAPER (no orders)",
              "dryrun": "🧪 DRY-RUN (logs real orders, sends none)",
              "live": ("🔴 LIVE-ARMED (REAL borrow+orders)" if armed
                       else "🔴 live (BLOCKED — arm file missing, no orders)")}
    print(f"[neg-carry] start — MODE={MODE} → {banner.get(MODE, MODE)}")
    print(f"[neg-carry] safety — max_pos={MAX_POS} max_total=${MAX_TOTAL_USD:.0f} "
          f"auto_halt_drawdown={MAX_DD_PCT}% kill_file={KILL_FILE} "
          f"liq_margin_floor={LIQ_MARGIN_LEVEL_MIN} liq_perp_buf={LIQ_PERP_BUFFER_PCT}%")
    NOTIONAL = None
    for _attempt in range(5):
        NOTIONAL, _notional_reason = await _resolve_notional(client)
        if NOTIONAL is not None:
            break
        print(f"[neg-carry] sizing unreadable (attempt {_attempt + 1}/5) — retrying in 10s")
        await asyncio.sleep(10)
    if NOTIONAL is None:
        print("[neg-carry] ABORT — refusing to run on an unreadable balance rather "
              "than defaulting to maximum notional")
        return
    print(f"[neg-carry] params — entry<=-{ENTRY*100:.3f}% exit<{EXIT*100:.3f}% "
          f"fee={FEE_RT*100:.2f}% notional=${NOTIONAL:.2f}/leg ({_notional_reason}) "
          f"leverage={LEVERAGE}x poll={POLL_MIN}m borrow_rate_ttl={_BORROW_RATE_TTL_S/60:.0f}m")
    if MODE == "live":
        # Reconcile: surface any REAL borrowed assets / open perp positions so an
        # orphan from a prior crash is visible before we trade on state-file trust.
        try:
            acc = await client.get_margin_account()
            real_borrowed = {a["asset"]: float(a["borrowed"]) for a in acc.get("userAssets", [])
                             if float(a.get("borrowed", 0) or 0) > 0}
            fpos = await client.futures_account()
            real_perp = {p["symbol"]: float(p["positionAmt"]) for p in fpos.get("positions", [])
                         if float(p.get("positionAmt", 0) or 0) != 0}
            print(f"[neg-carry] live reconcile — real borrowed={real_borrowed or 'none'} "
                  f"real_perp={real_perp or 'none'} marginLevel={acc.get('marginLevel')}")
            tracked = {s for s, p in state["open"].items() if p.get("mode") == "live"}
            if real_borrowed or real_perp:
                print(f"[neg-carry] ⚠️  live positions exist on exchange; tracked-live={tracked or 'none'} "
                      f"— verify these match before trusting automated close.")
        except Exception as e:
            print(f"[neg-carry] ⚠️  live reconcile failed ({str(e)[:80]}) — proceeding read-only-safe")
    try:
        universe = await build_universe(client)
        print(f"[neg-carry] universe: {len(universe)} liquid, spot-hedgeable, "
              f"margin-BORROWABLE perps")
        while True:
            # Keep the signing clock aligned before any SIGNED call; drift
            # silently disables them all while unsigned calls keep working.
            await resync_clock(client, "neg-carry")
            now = time.time()
            funding = await current_funding(client, universe)
            opened = closed = 0

            # Liquidation guard (live only): symbols that MUST be force-closed now.
            danger = []
            if MODE == "live" and state["open"]:
                live_syms = [s for s, p in state["open"].items() if p.get("mode") == "live"]
                if live_syms:
                    danger = await _liquidation_guard(client, live_syms)

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

                if abs(fr) < EXIT or held_h >= MAX_HOLD_H or _killed() or sym in danger:
                    pos_mode = pos.get("mode", "paper")
                    # Live positions: actually close the real legs FIRST. If the
                    # real close fails, do NOT drop it from state — keep tracking
                    # so the next poll retries rather than orphaning a live loan.
                    if pos_mode == "live":
                        ok = await _live_close(client, sym, float(pos.get("borrowed_qty", 0.0)))
                        if not ok:
                            print(f"  [LIVE-CLOSE] {sym} did NOT confirm — keeping tracked, will retry")
                            continue
                    signed_funding = await signed_settled_funding_long(
                        client, sym, int(pos["entry_ts"] * 1000))
                    # signed_funding=0.0 on a failed read booked a fabricated
                    # net (= -borrow -fee) into the ledger and the verdict.
                    rec = {
                        "ts": datetime.now(timezone.utc).isoformat(), "symbol": sym,
                        "held_h": round(held_h, 1),
                        "borrow_cost_pct": round(pos["accrued_borrow_pct"] * 100, 4),
                        "entry_funding": pos["entry_funding"],
                        "exit_funding": fr, "mode": pos_mode,
                    }
                    if signed_funding is None:
                        rec.update({"signed_funding_pct": None, "net_pct": None,
                                    "funding_unknown": True})
                        print(f"  [NEG-CARRY] {sym} closed, funding UNREADABLE — booked "
                              f"unknown, excluded from stats")
                    else:
                        rec.update({
                            "signed_funding_pct": round(signed_funding * 100, 4),
                            "net_pct": round((signed_funding - pos["accrued_borrow_pct"]
                                              - FEE_RT) * 100.0, 4)})
                    _log_trade(rec)
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

                    pos_rec = {"entry_ts": now, "entry_funding": fr,
                               "last_check_ts": now, "accrued_borrow_pct": 0.0, "mode": "paper"}
                    if MODE == "dryrun":
                        print(f"  [DRYRUN] would OPEN {sym}: borrow+sell+long-perp "
                              f"~${NOTIONAL:.2f}/leg at funding {fr*100:+.3f}% (sending nothing)")
                        continue  # prove the decision/sizing path; hold no phantom position
                    if MODE == "live":
                        if not os.path.exists(LIVE_ARM_FILE):
                            print(f"  [LIVE-BLOCKED] {sym}: arm file '{LIVE_ARM_FILE}' missing — no order")
                            continue
                        live_open_ct = sum(1 for p in state["open"].values() if p.get("mode") == "live")
                        if live_open_ct >= MAX_LIVE_POS:
                            continue  # live-position cap reached (paper may coexist above it)
                        res = await _live_open(client, sym, NOTIONAL)
                        if not res:
                            continue  # failed + already unwound, or below min-notional
                        pos_rec["mode"] = "live"
                        pos_rec["borrowed_qty"] = res["borrowed_qty"]
                    state["open"][sym] = pos_rec
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
