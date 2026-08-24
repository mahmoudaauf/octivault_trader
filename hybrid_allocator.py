#!/usr/bin/env python3
"""
Hybrid capital allocator — safe yield CORE + capped auto-speculative SATELLITE.

WHAT THIS IS (and honestly is not)
----------------------------------
A barbell / core-satellite structure, operator-chosen with full disclosure:
  • CORE (default 80%) stays in Binance Simple Earn flexible USDT — safe yield,
    the account's only real (slow) growth. Never traded, never at risk here.
  • SATELLITE (default 20%) runs a simple AUTOMATED speculative trader.

The satellite has NO EDGE. It is negative expected value. Its entry rule is a
Donchian breakout — one of the rules this very project already FALSIFIED
(volatility_breakout_backtest.py). It is used precisely BECAUSE any rule here
loses on average, and breakout is simple and transparent. Do not expect profit.

THE POINT OF THIS FILE IS THE GUARDRAILS, NOT THE TRADING:
  1. THE WALL — the satellite can lose its whole allocation but can NEVER pull
     from the core. `redeem_simple_earn_flexible_product` is called ONLY in the
     one-time live setup, NEVER in the trading loop. Money only flows
     satellite -> core (win-sweep), never core -> satellite.
  2. WIN-SWEEP — satellite gains above target are swept back into the safe core.
  3. FLOOR — below the floor the satellite stops (can't meet min-notional).
  4. Double-gated live (MODE=live AND arm file), kill-switch, drawdown-halt.

Production hardening (2026-08-17, gap audit G1-G8):
  G1 EXCHANGE-SIDE STOP — on every live open an OCO (take-profit + stop-loss)
     rests on Binance, so the stop is enforced even if this process is dead or
     the machine is asleep. The poll loop is only a backup + the time-stop.
  G2 RESTART RECONCILIATION — on live startup, verify the tracked position
     against real balances; adopt/alert on stray holdings; protect an
     unprotected adopted position.
  G3 SINGLE-INSTANCE LOCK — an flock pidfile prevents two daemons trading one
     account.
  G5 API RETRY — order/price calls retry with backoff; a filled-but-errored buy
     is reconciled instead of leaving an untracked position.
  G6 paper cash now applies fees (was inconsistent with the ledger).
  G7 drawdown-halt counts LIVE trades only (paper history no longer trips live).
  G8 live sizing is capped to the satellite budget, not all spot USDT.

Modes (HYBRID_MODE, default paper):
  paper  — simulate the satellite against a NOTIONAL balance; move NO real money.
  dryrun — log the exact real orders it would place; send nothing.
  live   — real spot orders; DOUBLE-GATED on MODE=live AND logs/hybrid_live_armed.

Usage:
  python3 hybrid_allocator.py          # run the daemon (mode from env)
  python3 hybrid_allocator.py report    # print the satellite track record
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

STATE = os.getenv("HYBRID_STATE_FILE", "logs/hybrid_state.json")
LEDGER = os.getenv("HYBRID_LEDGER_FILE", "logs/hybrid_ledger.jsonl")

# ── Allocation ───────────────────────────────────────────────────────────────
CORE_FRACTION = float(os.getenv("HYBRID_CORE_FRACTION", "0.80"))   # 80% safe
SWEEP_THRESHOLD_USD = float(os.getenv("HYBRID_SWEEP_THRESHOLD_USD", "0.5"))
SATELLITE_FLOOR_USD = float(os.getenv("HYBRID_SATELLITE_FLOOR_USD", "5.0"))
TRADE_FRACTION = float(os.getenv("HYBRID_TRADE_FRACTION", "0.95"))  # of sat budget

# ── Satellite trading ────────────────────────────────────────────────────────
_DEFAULT_WATCH = "SOLUSDT,AVAXUSDT,LINKUSDT,DOGEUSDT,ADAUSDT,SUIUSDT,APTUSDT,ARBUSDT,INJUSDT,NEARUSDT"
# Assets the satellite must NEVER touch — long-term buy-and-hold buckets that
# live in the same spot wallet. Without this, `_reconcile`'s stray-holding
# adoption would grab the BTC hold and put a -2% stop on a position meant to be
# held for years. Filtered out of the watchlist, so it guards entry AND adoption.
PROTECTED_ASSETS = {s.strip().upper() for s in os.getenv("HYBRID_PROTECTED_ASSETS", "BTC").split(",") if s.strip()}
WATCHLIST = [s.strip().upper() for s in os.getenv("HYBRID_WATCHLIST", _DEFAULT_WATCH).split(",") if s.strip()]
WATCHLIST = [s for s in WATCHLIST if s[:-4] not in PROTECTED_ASSETS]
INTERVAL = os.getenv("HYBRID_INTERVAL", "1h")
BREAKOUT_LOOKBACK = int(os.getenv("HYBRID_BREAKOUT_LOOKBACK", "20"))
VOLUME_MULT = float(os.getenv("HYBRID_VOLUME_MULT", "2.0"))
TP_PCT = float(os.getenv("HYBRID_TP_PCT", "4.0"))   # take-profit %
SL_PCT = float(os.getenv("HYBRID_SL_PCT", "2.0"))   # stop-loss %
# OCO stop-limit sits this % below the stop TRIGGER, so a triggered stop still
# fills through a fast move instead of resting as an unfilled limit.
STOP_LIMIT_OFFSET_PCT = float(os.getenv("HYBRID_STOP_LIMIT_OFFSET_PCT", "0.8"))
MAX_HOLD_H = float(os.getenv("HYBRID_MAX_HOLD_H", "48"))
FEE_RT_PCT = float(os.getenv("HYBRID_FEE_RT_PCT", "0.2"))  # round-trip spot taker
COOLDOWN_H = float(os.getenv("HYBRID_COOLDOWN_H", "1"))
POLL_MIN = float(os.getenv("HYBRID_POLL_MIN", "15"))

_INTERVAL_HOURS = {"5m": 5 / 60, "15m": 0.25, "30m": 0.5, "1h": 1.0, "2h": 2.0, "4h": 4.0, "1d": 24.0}

# ── Execution mode + safety ──────────────────────────────────────────────────
MODE = os.getenv("HYBRID_MODE", "paper").lower()
LIVE_ARM_FILE = os.getenv("HYBRID_LIVE_ARM_FILE", "logs/hybrid_live_armed")
KILL_FILE = os.getenv("HYBRID_KILL_FILE", "logs/hybrid.stop")
PIDFILE = os.getenv("HYBRID_PIDFILE", "logs/hybrid.pid")          # G3
MAX_DD_PCT = float(os.getenv("HYBRID_MAX_DD_PCT", "100.0"))
API_RETRIES = int(os.getenv("HYBRID_API_RETRIES", "3"))          # G5
API_RETRY_DELAY_S = float(os.getenv("HYBRID_API_RETRY_DELAY_S", "2.0"))

_EARN_PRODUCT_ID = None  # resolved once at startup (USDT flexible, e.g. "USDT001")
_filters_cache: dict[str, tuple] = {}  # symbol -> (qty_step, min_notional, tick)
_lock_handle = None      # keep the flock fd alive for the process lifetime


def _is_live() -> bool:
    return MODE == "live" and os.path.exists(LIVE_ARM_FILE)


def _killed() -> bool:
    return os.path.exists(KILL_FILE)


# ── G3: single-instance lock ─────────────────────────────────────────────────
def _acquire_lock() -> bool:
    """Exclusive flock on a pidfile. Prevents two daemons trading one account
    (double orders, racing sweeps, state clobbering). Returns False if held."""
    global _lock_handle
    import fcntl
    os.makedirs(os.path.dirname(PIDFILE) or ".", exist_ok=True)
    fh = open(PIDFILE, "w")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return False
    fh.seek(0)
    fh.write(str(os.getpid()))
    fh.truncate()
    fh.flush()
    _lock_handle = fh  # held until process exit
    return True


def _current_drawdown_pct(live_only: bool = True) -> float:
    """Peak-to-current drawdown over summed net_pct. G7: by default counts only
    LIVE trades, so a long PAPER loss history can't trip the live drawdown-halt
    the instant you arm live."""
    if not os.path.exists(LEDGER):
        return 0.0
    cum = peak = 0.0
    with open(LEDGER) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if live_only and rec.get("mode") != "live":
                continue
            try:
                cum += float(rec.get("net_pct", 0.0))
            except (TypeError, ValueError):
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


def _round_step(qty: float, step: float) -> float:
    if step <= 0:
        return qty
    return (int(qty / step)) * step


def _round_price(price: float, tick: float) -> float:
    if tick <= 0:
        return price
    return round(int(price / tick) * tick, 8)


# ── G5: retry wrapper for flaky API calls ────────────────────────────────────
async def _retry(fn, *args, **kwargs):
    """Await fn(*args, **kwargs) with bounded retry/backoff. Re-raises the last
    error so callers still see a genuine failure."""
    last = None
    for attempt in range(API_RETRIES):
        try:
            return await fn(*args, **kwargs)
        except Exception as e:
            last = e
            if attempt < API_RETRIES - 1:
                await asyncio.sleep(API_RETRY_DELAY_S * (attempt + 1))
    raise last


async def _symbol_filters(client, symbol: str) -> tuple[float, float, float]:
    """(qty_step, min_notional, price_tick) for a SPOT symbol. Cached."""
    if symbol in _filters_cache:
        return _filters_cache[symbol]
    sinfo = await client.get_exchange_info()
    ssym = next(x for x in sinfo["symbols"] if x["symbol"] == symbol)
    step = next(float(f["stepSize"]) for f in ssym["filters"] if f["filterType"] == "LOT_SIZE")
    tick = 0.0
    min_notional = 5.0
    for f in ssym["filters"]:
        if f["filterType"] == "PRICE_FILTER":
            tick = float(f["tickSize"])
        if f["filterType"] in ("NOTIONAL", "MIN_NOTIONAL"):
            min_notional = float(f.get("minNotional", f.get("notional", 5.0)))
    _filters_cache[symbol] = (step, min_notional, tick)
    return _filters_cache[symbol]


# ── Earn (core) helpers — READ + one-directional writes ──────────────────────
async def _resolve_product_id(client) -> str | None:
    try:
        lst = await client.get_simple_earn_flexible_product_list(asset="USDT")
        rows = lst.get("rows", []) if isinstance(lst, dict) else lst
        for p in rows:
            if str(p.get("asset")) == "USDT":
                return p.get("productId")
    except Exception as e:
        print(f"[hybrid] earn product lookup failed: {str(e)[:70]}")
    return None


async def _earn_usdt(client) -> float:
    try:
        pos = await client.get_simple_earn_flexible_product_position(asset="USDT")
        rows = pos.get("rows", []) if isinstance(pos, dict) else pos
        return sum(float(p.get("totalAmount", 0.0) or 0.0) for p in rows)
    except Exception:
        return 0.0


async def _resync_clock(client) -> bool:
    """Re-derive the client's Binance timestamp offset.

    python-binance computes `timestamp_offset` ONCE inside AsyncClient.create()
    and never refreshes it. Local clock drift — which macOS accrues freely across
    sleep/wake — eventually pushes every SIGNED request's timestamp outside
    recvWindow, so they all fail -1021 permanently while UNSIGNED calls (klines,
    tickers) keep working. Observed live 2026-08-23: after ~1h the daemon read
    its balance as $0.00 and idled the sleeve for 12h with $9.62 sitting in spot.
    """
    try:
        res = await client.get_server_time()
        before = client.timestamp_offset
        client.timestamp_offset = res["serverTime"] - int(time.time() * 1000)
        if abs(client.timestamp_offset - before) > 500:
            print(f"  [CLOCK-RESYNC] timestamp offset {before}ms → {client.timestamp_offset}ms")
        return True
    except Exception as e:
        print(f"  [CLOCK-RESYNC] failed: {str(e)[:80]}")
        return False


async def _read_balance(client, asset: str):
    """Raw balance dict for an asset, or None if it could NOT be read.

    None means UNKNOWN and must never be coerced to zero. Swallowing read errors
    into 0.0 caused two real failures: (1) the sleeve sat idle for 12h believing
    it had $0.00 while $9.62 was in spot; (2) far worse, `_detect_exchange_close`
    reads this to decide whether the coin is gone — a failed read looked like
    "OCO fired", which would book a FABRICATED close and abandon a live position.
    """
    try:
        bal = await _retry(client.get_asset_balance, asset=asset)
    except Exception as e:
        # A signature/timestamp rejection is self-healable — resync and retry
        # once before giving up, so drift costs one read instead of a whole day.
        if "-1021" in str(e) or "-1022" in str(e) or "recvWindow" in str(e):
            print(f"  [BAL-READ-FAIL] {asset}: {str(e)[:70]} — resyncing clock")
            if await _resync_clock(client):
                try:
                    bal = await _retry(client.get_asset_balance, asset=asset)
                    print(f"  [BAL-READ-OK] {asset}: recovered after clock resync")
                    return bal if isinstance(bal, dict) and "free" in bal else None
                except Exception as e2:
                    e = e2
        print(f"  [BAL-READ-FAIL] {asset}: {str(e)[:90]} — UNKNOWN (not zero)")
        return None
    if not isinstance(bal, dict) or "free" not in bal:
        print(f"  [BAL-READ-FAIL] {asset}: unexpected response {str(bal)[:60]} — UNKNOWN (not zero)")
        return None
    return bal


async def _spot_free_usdt(client):
    """Free spot USDT, or None if unreadable."""
    return await _asset_free(client, "USDT")


async def _asset_qty(client, asset: str):
    """Free + locked balance (locked because an OCO holds it), or None if unreadable."""
    bal = await _read_balance(client, asset)
    if bal is None:
        return None
    try:
        return float(bal.get("free") or 0.0) + float(bal.get("locked") or 0.0)
    except (TypeError, ValueError):
        return None


async def _asset_free(client, asset: str):
    """FREE balance only — what can actually be sold right now — or None if
    unreadable. A resting OCO locks its quantity, so free < total until that OCO
    is cancelled or fills."""
    bal = await _read_balance(client, asset)
    if bal is None:
        return None
    try:
        return float(bal.get("free") or 0.0)
    except (TypeError, ValueError):
        return None


# ── Breakout signal ──────────────────────────────────────────────────────────
async def _fetch_klines(client, symbol: str):
    try:
        lookback_days = max(2, int((BREAKOUT_LOOKBACK + 4) * _INTERVAL_HOURS[INTERVAL] / 24) + 1)
        return await client.get_historical_klines(
            symbol, INTERVAL, start_str=f"{lookback_days} days ago UTC")
    except Exception:
        return None


def _is_breakout(klines) -> bool:
    """LONG breakout on the last CLOSED bar: close > prior N-bar high AND volume
    >= VOLUME_MULT x average. (klines[-1] may be the still-forming bar → use -2.)"""
    n = BREAKOUT_LOOKBACK
    if not klines or len(klines) < n + 2:
        return False
    highs = [float(k[2]) for k in klines]
    closes = [float(k[4]) for k in klines]
    vols = [float(k[5]) for k in klines]
    i = len(klines) - 2  # last closed bar
    window_high = max(highs[i - n:i])
    avg_vol = sum(vols[i - n:i]) / n
    if avg_vol <= 0:
        return False
    return closes[i] > window_high and vols[i] >= VOLUME_MULT * avg_vol


async def _price(client, symbol: str) -> float:
    t = await _retry(client.get_symbol_ticker, symbol=symbol)
    return float(t["price"])


# ── Satellite budget ─────────────────────────────────────────────────────────
async def _satellite_cash(client, state) -> float:
    """Investable satellite USDT. G8: live is capped to the satellite target
    (never sizes off unrelated USDT parked in spot). Paper: simulated cash."""
    if _is_live():
        spot = await _spot_free_usdt(client)
        if spot is None:
            return None                      # UNKNOWN — caller must skip, not trade
        target = float(state.get("satellite_target", 0.0)) or spot
        return min(spot, target)
    return float(state.get("satellite_cash", 0.0))


# ── Core allocation: one-time setup + win-sweep (the WALL lives here) ─────────
async def _setup_allocation(client, state):
    """Establish the 80/20 split. In LIVE this redeems earn->spot ONCE. This is
    the ONLY place redeem is called — the loop never redeems, so satellite
    losses can never touch the core."""
    spot = await _spot_free_usdt(client)
    earn = await _earn_usdt(client)
    if spot is None:
        # The one-time split (and its redeem) must never be sized off a failed
        # read; retry on the next start rather than move the wrong amount.
        print("[hybrid] setup deferred — spot balance unreadable this cycle")
        return
    total = spot + earn
    if "satellite_target" not in state:
        state["satellite_target"] = round(total * (1.0 - CORE_FRACTION), 2)
    sat_target = state["satellite_target"]

    if not _is_live():
        if "satellite_cash" not in state:
            state["satellite_cash"] = round(sat_target, 2)
        print(f"[hybrid] paper split — total=${total:.2f} core={total*CORE_FRACTION:.2f} "
              f"satellite=${state['satellite_cash']:.2f} (simulated, no real money moved)")
        return

    if state.get("setup_done"):
        print(f"[hybrid] live split already established — satellite_target=${sat_target:.2f}, spot=${spot:.2f}")
        return
    deficit = sat_target - spot
    if deficit > 0.01 and _EARN_PRODUCT_ID:
        try:
            await client.redeem_simple_earn_flexible_product(
                productId=_EARN_PRODUCT_ID, amount=str(round(deficit, 2)))
            print(f"[hybrid] live setup — redeemed ${deficit:.2f} earn->spot; core stays ~${total*CORE_FRACTION:.2f}")
        except Exception as e:
            print(f"[hybrid] live setup redeem failed: {str(e)[:90]} — satellite underfunded, continuing")
    state["setup_done"] = True


async def _win_sweep(client, state, sat_cash: float):
    """Ratchet: sweep satellite cash above target back into the safe core.
    Money only ever flows satellite -> core here — never the reverse."""
    target = float(state.get("satellite_target", 0.0))
    if target <= 0:
        return sat_cash
    excess = sat_cash - target - SWEEP_THRESHOLD_USD
    if excess < 0.5:
        return sat_cash
    if _is_live():
        if _EARN_PRODUCT_ID:
            try:
                await client.subscribe_simple_earn_flexible_product(
                    productId=_EARN_PRODUCT_ID, amount=str(round(excess, 2)))
                print(f"[hybrid] 🟢 WIN-SWEEP ${excess:.2f} satellite->core (banked to safe yield)")
            except Exception as e:
                print(f"[hybrid] win-sweep subscribe failed: {str(e)[:80]}")
                return sat_cash
    else:
        state["satellite_cash"] = round(sat_cash - excess, 2)
        print(f"[hybrid] 🟢 WIN-SWEEP ${excess:.2f} satellite->core (simulated)")
    return sat_cash - excess


# ── G1: exchange-side protective OCO (take-profit + stop-loss) ────────────────
async def _place_protective_oco(client, symbol, qty, tp_price, sl_price):
    """Rest an OCO SELL on Binance so the stop/target is enforced even if this
    process dies or the machine sleeps. Returns orderListId, or None on failure
    (the software poll loop then remains the only stop — logged loudly)."""
    step, _mn, tick = await _symbol_filters(client, symbol)
    q = _round_step(qty, step)
    tp = _round_price(tp_price, tick)
    stop_trig = _round_price(sl_price, tick)
    stop_limit = _round_price(sl_price * (1 - STOP_LIMIT_OFFSET_PCT / 100.0), tick)
    if q <= 0:
        return None
    def _fmt(p):
        return f"{p:.8f}".rstrip("0").rstrip(".")
    try:
        # Binance's current OCO API uses the above/below-order form (the old
        # price/stopPrice/stopLimitPrice form is rejected with -1102). For a SELL
        # OCO: ABOVE = take-profit (LIMIT_MAKER, price > last), BELOW = stop-loss
        # (STOP_LOSS_LIMIT, stopPrice < last).
        o = await _retry(
            client.create_oco_order, symbol=symbol, side="SELL", quantity=q,
            aboveType="LIMIT_MAKER", abovePrice=_fmt(tp),
            belowType="STOP_LOSS_LIMIT", belowStopPrice=_fmt(stop_trig),
            belowPrice=_fmt(stop_limit), belowTimeInForce="GTC")
        oid = o.get("orderListId")
        print(f"  [OCO] {symbol} protective stop+target resting on exchange (id={oid}) "
              f"tp={tp} stop={stop_trig}")
        return oid
    except Exception as e:
        print(f"  ⚠️  [OCO-FAIL] {symbol}: {str(e)[:120]} — SOFTWARE STOP ONLY until next open")
        return None


async def _cancel_oco(client, symbol, oco_id) -> bool:
    """Cancel a resting OCO. Returns True if the list is gone (cancelled now, or
    already filled/cancelled), False if it may STILL be resting — in which case
    the base asset is still locked and a market sell would fail.

    Must hit DELETE /api/v3/orderList (`v3_delete_order_list`), NOT
    DELETE /api/v3/order (`cancel_order`) — the latter cancels a single order by
    orderId and rejects `orderListId`, so it never cancelled the OCO at all.
    """
    if oco_id is None:
        return True
    canceller = getattr(client, "v3_delete_order_list", None)
    if canceller is None:  # very old python-binance — no spot order-list cancel
        print(f"  ⚠️  [OCO-CANCEL] {symbol}: client lacks v3_delete_order_list — cannot cancel list {oco_id}")
        return False
    try:
        await _retry(canceller, symbol=symbol, orderListId=oco_id)
        return True
    except Exception as e:
        # Already filled/cancelled → the list is gone, which is what we wanted.
        if "-2011" in str(e) or "Unknown order" in str(e) or "-2013" in str(e):
            return True
        print(f"  ⚠️  [OCO-CANCEL] {symbol} id={oco_id}: {str(e)[:80]} — qty may still be LOCKED")
        return False


# ── Satellite trade execution ────────────────────────────────────────────────
async def _open_position(client, state, symbol: str, sat_cash: float):
    step, min_notional, _tick = await _symbol_filters(client, symbol)
    price = await _price(client, symbol)
    notional = min(sat_cash * TRADE_FRACTION, sat_cash - 0.05)
    if notional < max(min_notional, SATELLITE_FLOOR_USD):
        return False
    qty = _round_step(notional / price, step)
    if qty <= 0 or qty * price < max(min_notional, 5.0):
        return False

    if MODE == "dryrun":
        print(f"  [DRYRUN] would BUY {qty} {symbol} @ ~{price:.6f} (~${qty*price:.2f}) "
              f"tp=+{TP_PCT}% sl=-{SL_PCT}% + resting OCO — sending nothing")
        return False

    fill_price = price
    oco_id = None
    if _is_live():
        try:
            o = await _retry(client.order_market_buy, symbol=symbol, quantity=qty)
            fills = o.get("fills", [])
            if fills:
                spent = sum(float(f["price"]) * float(f["qty"]) for f in fills)
                got = sum(float(f["qty"]) for f in fills)
                fill_price = spent / got if got else price
                qty = got
        except Exception as e:
            # G5: a buy can fill AND raise (timeout after match). Reconcile the
            # real balance before assuming no fill, so we never leave an
            # untracked, unstopped position.
            print(f"  [SAT-OPEN-ERROR] {symbol}: {str(e)[:90]} — checking for a stray fill")
            held = await _asset_qty(client, symbol[:-4])
            if held is None:
                print(f"  [SAT-OPEN-ERROR] {symbol}: balance unreadable — cannot confirm fill; "
                      f"NOT tracking. Reconcile will adopt it on the next start if it filled.")
                return False
            if held * price >= max(min_notional, 5.0):
                qty = _round_step(held, step)
                print(f"  [SAT-OPEN-RECOVER] {symbol}: buy DID fill ({held}) — adopting + protecting")
            else:
                return False
        # G1: rest the protective OCO on the exchange (uses actual held qty).
        held = await _asset_qty(client, symbol[:-4])
        if held is None:
            held = qty          # unreadable → protect the qty we believe we bought
        tp_price = fill_price * (1 + TP_PCT / 100.0)
        sl_price = fill_price * (1 - SL_PCT / 100.0)
        oco_id = await _place_protective_oco(client, symbol, min(qty, held), tp_price, sl_price)
    else:
        state["satellite_cash"] = round(sat_cash - qty * fill_price * (1 + FEE_RT_PCT / 200.0), 2)

    state["position"] = {
        "symbol": symbol, "entry_ts": time.time(), "entry_price": fill_price,
        "qty": qty, "tp_price": fill_price * (1 + TP_PCT / 100.0),
        "sl_price": fill_price * (1 - SL_PCT / 100.0), "mode": MODE, "oco_id": oco_id,
    }
    state["last_entry_ts"] = time.time()
    print(f"  [SAT-OPEN] {symbol} {qty} @ {fill_price:.6f} (~${qty*fill_price:.2f}) "
          f"tp={state['position']['tp_price']:.6f} sl={state['position']['sl_price']:.6f} [{MODE}]")
    return True


def _book_close(state, symbol, entry, exit_price, reason, qty):
    """Write the ledger record and clear the position (shared by all close paths)."""
    gross_pct = (exit_price - entry) / entry * 100.0
    net_pct = gross_pct - FEE_RT_PCT
    pos = state.get("position") or {}
    held_h = (time.time() - pos.get("entry_ts", time.time())) / 3600.0
    _log_trade({
        "ts": datetime.now(timezone.utc).isoformat(), "symbol": symbol,
        "held_h": round(held_h, 2), "entry_price": entry, "exit_price": round(exit_price, 8),
        "qty": qty, "gross_pct": round(gross_pct, 4), "net_pct": round(net_pct, 4),
        "reason": reason, "mode": pos.get("mode", MODE),
    })
    print(f"  [SAT-CLOSE] {symbol} @ {exit_price:.6f} net={net_pct:+.2f}% ({reason}) [{pos.get('mode', MODE)}]")
    state["position"] = None


async def _actual_exit_vwap(client, symbol: str, entry_ts: float):
    """Volume-weighted price of the SELL fills that closed this position, read
    from real trade history. Returns None if history is unavailable, so callers
    can fall back rather than book a fabricated price."""
    try:
        trades = await _retry(client.get_my_trades, symbol=symbol, limit=50)
        since_ms = int(entry_ts * 1000)
        sells = [t for t in (trades or [])
                 if not t.get("isBuyer") and int(t.get("time", 0) or 0) >= since_ms]
        total_qty = sum(float(t["qty"]) for t in sells)
        if total_qty <= 0:
            return None
        return sum(float(t["price"]) * float(t["qty"]) for t in sells) / total_qty
    except Exception as e:
        # Any shape surprise from the API must fall back, never crash the poll.
        print(f"  [EXIT-FILL] {symbol}: trade history unusable ({str(e)[:60]})")
        return None


async def _detect_exchange_close(client, state) -> bool:
    """G1: if the resting OCO already executed (stop or target hit) — possibly
    while this process was asleep or between polls — the coin balance is gone.
    Detect it and book the trade at the OCO price, so the exchange stop and our
    records stay consistent. Returns True if it closed the position."""
    pos = state.get("position")
    if not pos or pos.get("mode") != "live":
        return False
    symbol = pos["symbol"]
    held = await _asset_qty(client, symbol[:-4])
    if held is None:
        # Balance UNKNOWN. Never infer "the OCO fired" from a failed read — that
        # would book a fabricated close and abandon a live position.
        print(f"  [OCO-CHECK] {symbol}: balance unreadable — assuming still held, retrying next poll")
        return False
    # Position still meaningfully held → OCO hasn't fired.
    if held >= pos["qty"] * 0.5:
        return False
    # Coin is gone → the OCO closed it. Book the REAL fill, not the intended
    # trigger price: a STOP_LOSS_LIMIT fills near sl*(1-STOP_LIMIT_OFFSET_PCT),
    # not at sl, and labelling by the *current* price (up to POLL_MIN stale)
    # could book a rebounded stop-out as a take-profit.
    actual = await _actual_exit_vwap(client, symbol, pos["entry_ts"])
    if actual is not None:
        exit_price = actual
        reason = "take-profit-oco" if actual >= pos["entry_price"] else "stop-loss-oco"
    else:
        # Trade history unavailable — fall back to the intended prices, but mark
        # the record as an estimate so the ledger never claims false precision.
        price = await _price(client, symbol)
        if price >= pos["tp_price"] * 0.999:
            exit_price, reason = pos["tp_price"], "take-profit-oco-est"
        else:
            exit_price, reason = pos["sl_price"], "stop-loss-oco-est"
    print(f"  [OCO-FILLED] {symbol} closed on exchange (held={held}) → booking {reason} @ {exit_price:.6f}")
    _book_close(state, symbol, pos["entry_price"], exit_price, reason, pos["qty"])
    return True


async def _close_position(client, state, reason: str):
    """Software-initiated close (time-stop / kill-switch / paper). Cancels the
    resting OCO first so the market sell isn't blocked by the locked balance."""
    pos = state["position"]
    symbol, qty, entry = pos["symbol"], pos["qty"], pos["entry_price"]
    exit_price = await _price(client, symbol)
    if _is_live():
        try:
            cancelled = await _cancel_oco(client, symbol, pos.get("oco_id"))
            step, _mn, _tick = await _symbol_filters(client, symbol)
            asset = symbol[:-4]
            total = await _asset_qty(client, asset)      # free + locked
            free = await _asset_free(client, asset)      # what can actually be sold
            if total is None or free is None:
                print(f"  [SAT-CLOSE-BLOCKED] {symbol}: balance unreadable — keeping tracked, retrying next poll")
                return False
            if total <= 0:
                # Coin vanished between the pre-check and here — the OCO fired in
                # that window. BOOK it at the real fill; silently dropping the
                # position would lose the trade from the ledger entirely.
                actual = await _actual_exit_vwap(client, symbol, pos["entry_ts"])
                if actual is not None:
                    exit_price = actual
                    reason = "take-profit-oco" if actual >= entry else "stop-loss-oco"
                print(f"  [SAT-CLOSE] {symbol} nothing held — OCO closed it mid-close; "
                      f"booking {reason} @ {exit_price:.6f}")
                _book_close(state, symbol, entry, exit_price, reason, qty)
                return True
            if free <= 0:
                # Held but fully LOCKED — an OCO is still resting. Dropping the
                # position here would orphan a real holding, so keep tracking it.
                print(f"  [SAT-CLOSE-BLOCKED] {symbol} {total} held but LOCKED by OCO "
                      f"{pos.get('oco_id')} (cancel_ok={cancelled}) — keeping tracked, retrying next poll")
                return False
            sell_qty = _round_step(min(qty, free), step)
            if sell_qty <= 0:
                print(f"  [SAT-CLOSE] {symbol} free {free} below lot step {step} — keeping tracked")
                return False
            o = await _retry(client.order_market_sell, symbol=symbol, quantity=sell_qty)
            fills = o.get("fills", [])
            if fills:
                got = sum(float(f["price"]) * float(f["qty"]) for f in fills)
                sold = sum(float(f["qty"]) for f in fills)
                exit_price = got / sold if sold else exit_price
        except Exception as e:
            print(f"  [SAT-CLOSE-ERROR] {symbol}: {str(e)[:100]} — keeping tracked, will retry")
            return False
    else:
        state["satellite_cash"] = round(
            float(state.get("satellite_cash", 0.0)) + qty * exit_price * (1 - FEE_RT_PCT / 200.0), 2)
    _book_close(state, symbol, entry, exit_price, reason, qty)
    return True


# ── G2: restart / crash reconciliation ───────────────────────────────────────
async def _reconcile(client, state):
    """On live startup, make the tracked state agree with the exchange:
      • tracked position but the coin is GONE → it closed while we were down; book it.
      • tracked position still held but NO resting OCO → protect it now.
      • state is FLAT but a stray coin balance ≥ min-notional exists → adopt +
        protect it (a buy whose state-save crashed) instead of leaving it
        unmanaged and unstopped.
    """
    if not _is_live():
        return
    pos = state.get("position")
    if pos and pos.get("mode") == "live":
        symbol = pos["symbol"]
        held = await _asset_qty(client, symbol[:-4])
        if held is None:
            print(f"[hybrid] reconcile — {symbol} balance unreadable; leaving tracked state untouched")
            return
        if held < pos["qty"] * 0.5:
            print(f"[hybrid] reconcile — tracked {symbol} no longer held; booking its exchange close")
            await _detect_exchange_close(client, state)
        elif not pos.get("oco_id"):
            print(f"[hybrid] reconcile — tracked {symbol} has NO resting stop; placing one now")
            pos["oco_id"] = await _place_protective_oco(
                client, symbol, min(pos["qty"], held), pos["tp_price"], pos["sl_price"])
        return
    # Flat: look for a stray holding to adopt.
    for symbol in WATCHLIST:
        try:
            held = await _asset_qty(client, symbol[:-4])
            price = await _price(client, symbol)
            if held is None:
                continue
            if held * price >= max(SATELLITE_FLOOR_USD, 5.0):
                print(f"[hybrid] reconcile — stray {symbol} holding ${held*price:.2f} found; adopting + protecting")
                tp, sl = price * (1 + TP_PCT / 100.0), price * (1 - SL_PCT / 100.0)
                oco = await _place_protective_oco(client, symbol, held, tp, sl)
                state["position"] = {
                    "symbol": symbol, "entry_ts": time.time(), "entry_price": price,
                    "qty": held, "tp_price": tp, "sl_price": sl, "mode": "live", "oco_id": oco,
                }
                return
        except Exception:
            continue


def _report():
    trades = []
    if os.path.exists(LEDGER):
        with open(LEDGER) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        trades.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    if not trades:
        print("No closed satellite trades yet.")
        return
    nets = [t["net_pct"] for t in trades]
    n = len(nets)
    wins = sum(1 for x in nets if x > 0)
    print("=" * 66)
    print(f"HYBRID SATELLITE — TRACK RECORD ({n} closed trades)")
    print("=" * 66)
    print(f"  Avg net/trade: {sum(nets)/n:+.3f}%   win-rate: {wins/n*100:.0f}%   cum: {sum(nets):+.2f}%")
    print("  NOTE: satellite is negative-EV by construction — capped 'action',")
    print("        not a growth engine. The core (80% in yield) is where growth lives.")
    print("=" * 66)


async def run():
    global _EARN_PRODUCT_ID
    from binance import AsyncClient

    if not _acquire_lock():                                   # G3
        print(f"[hybrid] another instance holds {PIDFILE} — refusing to start (single-instance lock)")
        return

    client = await AsyncClient.create(
        os.getenv("BINANCE_API_KEY") or "x", os.getenv("BINANCE_API_SECRET") or "x")
    state = _load(STATE, {"position": None})
    armed = _is_live()
    banner = {"paper": "📝 PAPER (no real money)",
              "dryrun": "🧪 DRY-RUN (logs orders, sends none)",
              "live": ("🔴 LIVE-ARMED (real spot orders + exchange stops)" if armed
                       else "🔴 live (BLOCKED — arm file missing, no orders)")}
    print(f"[hybrid] start — MODE={MODE} → {banner.get(MODE, MODE)}")
    print(f"[hybrid] barbell — core={CORE_FRACTION*100:.0f}% (yield, untouchable) "
          f"satellite={(1-CORE_FRACTION)*100:.0f}% (auto-speculative, NEGATIVE-EV, capped)")
    print(f"[hybrid] satellite — breakout({BREAKOUT_LOOKBACK}/{INTERVAL}) tp=+{TP_PCT}% sl=-{SL_PCT}% "
          f"floor=${SATELLITE_FLOOR_USD} sweep>+${SWEEP_THRESHOLD_USD} kill={KILL_FILE}")
    try:
        _EARN_PRODUCT_ID = await _resolve_product_id(client)
        await _setup_allocation(client, state)
        await _reconcile(client, state)                       # G2
        _save(STATE, state)

        while True:
            now = time.time()
            # Keep the signing clock aligned BEFORE any signed call this cycle.
            # Cheap (one unsigned request) and prevents drift from silently
            # disabling every signed request until a restart.
            await _resync_clock(client)

            # ── Manage an open position ──
            if state.get("position"):
                try:
                    # G1: did the exchange OCO already close it (asleep/between polls)?
                    if not await _detect_exchange_close(client, state):
                        pos = state["position"]
                        price = await _price(client, pos["symbol"])
                        held_h = (now - pos["entry_ts"]) / 3600.0
                        reason = None
                        # Software TP/SL remain as a BACKUP to the exchange OCO
                        # (and are the only stop in paper mode).
                        if price >= pos["tp_price"]:
                            reason = "take-profit"
                        elif price <= pos["sl_price"]:
                            reason = "stop-loss"
                        elif held_h >= MAX_HOLD_H:
                            reason = "time-stop"
                        elif _killed():
                            reason = "kill-switch"
                        if reason:
                            await _close_position(client, state, reason)
                    _save(STATE, state)
                except Exception as e:
                    print(f"  [SAT-MANAGE-ERROR] {str(e)[:90]}")

            # ── Drawdown auto-halt (G7: live trades only) ──
            dd = _current_drawdown_pct(live_only=True)
            if dd >= MAX_DD_PCT and not _killed():
                open(KILL_FILE, "w").close()
                print(f"[hybrid] 🛑 DRAWDOWN HALT {dd:.1f}% >= {MAX_DD_PCT}% — kill-switch engaged")

            sat_cash = await _satellite_cash(client, state)
            if sat_cash is None:
                # Balance unreadable — do NOT report $0.00 and do NOT trade off a
                # guess. Skip the cycle; the next poll re-reads.
                print(f"[hybrid {datetime.now().strftime('%H:%M')}] mode={MODE} "
                      f"sat_cash=UNKNOWN (balance read failed) — skipping this cycle")
                await asyncio.sleep(POLL_MIN * 60)
                continue

            # ── Win-sweep (only when flat) ──
            if not state.get("position"):
                sat_cash = await _win_sweep(client, state, sat_cash)
                _save(STATE, state)

            # ── Entry ──
            opened = False
            floor = max(SATELLITE_FLOOR_USD, 5.0)
            if (not state.get("position") and not _killed()
                    and sat_cash >= floor / TRADE_FRACTION   # G-low: no dead-zone busy-loop
                    and (now - float(state.get("last_entry_ts", 0))) >= COOLDOWN_H * 3600):
                if MODE == "live" and not _is_live():
                    print("  [LIVE-BLOCKED] arm file missing — no order")
                else:
                    for symbol in WATCHLIST:
                        try:
                            klines = await _fetch_klines(client, symbol)
                            if klines and _is_breakout(klines):
                                if await _open_position(client, state, symbol, sat_cash):
                                    opened = True
                                    _save(STATE, state)
                                    break
                        except Exception as e:
                            print(f"  [SCAN-ERROR] {symbol}: {str(e)[:70]}")
            elif sat_cash < floor and not state.get("position"):
                print(f"  [SATELLITE-FLOOR] cash ${sat_cash:.2f} < ${floor:.2f} — sleeve idle "
                      f"(never topped up from core)")

            ts = datetime.now(timezone.utc).strftime("%H:%M")
            held = "flat" if not state.get("position") else state["position"]["symbol"]
            print(f"[hybrid {ts}] mode={MODE} sat_cash=${sat_cash:.2f} pos={held} opened={int(opened)}")
            _save(STATE, state)
            await asyncio.sleep(POLL_MIN * 60)
    finally:
        await client.close_connection()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        _report()
    else:
        asyncio.run(run())
