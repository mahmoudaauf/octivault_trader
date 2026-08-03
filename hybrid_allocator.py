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
(volatility_breakout_backtest.py: 5,552 signals, negative net at every horizon,
35-42% win). It is used precisely BECAUSE any rule here loses on average, and
breakout is simple and transparent. On a ~$7.50 sleeve with a $5 min-notional
and ~0.2% round-trip fees, the most likely outcome is a slow bleed toward the
$5 floor, then idle. Do not expect profit.

THE POINT OF THIS FILE IS THE GUARDRAILS, NOT THE TRADING:
  1. THE WALL — the satellite can lose its whole allocation but can NEVER pull
     from the core. `redeem_simple_earn_flexible_product` is called ONLY in the
     one-time live setup, NEVER in the trading loop. Money only ever flows
     satellite -> core (win-sweep), never core -> satellite.
  2. WIN-SWEEP — satellite gains above target are swept back into the safe core,
     so a lucky run is banked instead of compounding into a bigger bet.
  3. FLOOR — below $5 the satellite stops (can't meet min-notional anyway).
  4. Double-gated live (MODE=live AND arm file), kill-switch, drawdown-halt.

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
# Buffer above target before a win is banked to the core. Kept SMALL relative to
# the tiny satellite base ($0.50, not $2) so ordinary gains actually get swept —
# a $2 buffer on a ~$7.50 sleeve would need a ~33% run before banking anything,
# defeating the ratchet. With the $0.50 min-sweep guard in _win_sweep this means
# a win is banked once the sleeve clears ~$1 over target.
SWEEP_THRESHOLD_USD = float(os.getenv("HYBRID_SWEEP_THRESHOLD_USD", "0.5"))
SATELLITE_FLOOR_USD = float(os.getenv("HYBRID_SATELLITE_FLOOR_USD", "5.0"))
TRADE_FRACTION = float(os.getenv("HYBRID_TRADE_FRACTION", "0.95"))  # of sat cash

# ── Satellite trading ────────────────────────────────────────────────────────
_DEFAULT_WATCH = "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,DOGEUSDT,AVAXUSDT,LINKUSDT,ADAUSDT,SUIUSDT"
WATCHLIST = [s.strip().upper() for s in os.getenv("HYBRID_WATCHLIST", _DEFAULT_WATCH).split(",") if s.strip()]
INTERVAL = os.getenv("HYBRID_INTERVAL", "1h")
BREAKOUT_LOOKBACK = int(os.getenv("HYBRID_BREAKOUT_LOOKBACK", "20"))
VOLUME_MULT = float(os.getenv("HYBRID_VOLUME_MULT", "1.5"))
TP_PCT = float(os.getenv("HYBRID_TP_PCT", "4.0"))   # take-profit %
SL_PCT = float(os.getenv("HYBRID_SL_PCT", "2.0"))   # stop-loss %
MAX_HOLD_H = float(os.getenv("HYBRID_MAX_HOLD_H", "48"))
FEE_RT_PCT = float(os.getenv("HYBRID_FEE_RT_PCT", "0.2"))  # round-trip spot taker
COOLDOWN_H = float(os.getenv("HYBRID_COOLDOWN_H", "1"))
POLL_MIN = float(os.getenv("HYBRID_POLL_MIN", "15"))

_INTERVAL_HOURS = {"5m": 5 / 60, "15m": 0.25, "30m": 0.5, "1h": 1.0, "2h": 2.0, "4h": 4.0, "1d": 24.0}

# ── Execution mode + safety (double-gated live, cloned from carry daemon) ─────
MODE = os.getenv("HYBRID_MODE", "paper").lower()
LIVE_ARM_FILE = os.getenv("HYBRID_LIVE_ARM_FILE", "logs/hybrid_live_armed")
KILL_FILE = os.getenv("HYBRID_KILL_FILE", "logs/hybrid.stop")
MAX_DD_PCT = float(os.getenv("HYBRID_MAX_DD_PCT", "100.0"))  # sat may fully bleed by design; floor stops it

_EARN_PRODUCT_ID = None  # resolved once at startup (USDT flexible, e.g. "USDT001")
_filters_cache: dict[str, tuple] = {}  # symbol -> (qty_step, min_notional)


def _is_live() -> bool:
    return MODE == "live" and os.path.exists(LIVE_ARM_FILE)


def _killed() -> bool:
    return os.path.exists(KILL_FILE)


def _current_drawdown_pct() -> float:
    """Peak-to-current drawdown over the satellite ledger's summed net_pct."""
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


def _round_step(qty: float, step: float) -> float:
    if step <= 0:
        return qty
    return (int(qty / step)) * step


async def _symbol_filters(client, symbol: str) -> tuple[float, float]:
    """(qty_step, min_notional) for a SPOT symbol. Cached; filters are static."""
    if symbol in _filters_cache:
        return _filters_cache[symbol]
    sinfo = await client.get_exchange_info()
    ssym = next(x for x in sinfo["symbols"] if x["symbol"] == symbol)
    step = next(float(f["stepSize"]) for f in ssym["filters"] if f["filterType"] == "LOT_SIZE")
    min_notional = 5.0
    for f in ssym["filters"]:
        if f["filterType"] in ("NOTIONAL", "MIN_NOTIONAL"):
            min_notional = float(f.get("minNotional", f.get("notional", 5.0)))
    _filters_cache[symbol] = (step, min_notional)
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


async def _spot_free_usdt(client) -> float:
    try:
        bal = await client.get_asset_balance(asset="USDT")
        return float((bal or {}).get("free", 0.0) or 0.0)
    except Exception:
        return 0.0


# ── Breakout signal (adapted from volatility_breakout_backtest.find_signals) ──
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
    t = await client.get_symbol_ticker(symbol=symbol)
    return float(t["price"])


# ── Satellite cash / value ───────────────────────────────────────────────────
async def _satellite_cash(client, state) -> float:
    """Uninvested satellite USDT. Live: real spot free. Paper: simulated cash."""
    if _is_live():
        return await _spot_free_usdt(client)
    return float(state.get("satellite_cash", 0.0))


# ── Core allocation: one-time setup + win-sweep (the WALL lives here) ─────────
async def _setup_allocation(client, state):
    """Establish the 80/20 split. In LIVE this redeems earn->spot ONCE to fund
    the satellite. This is the ONLY place redeem is ever called — the trading
    loop never redeems, so satellite losses can never touch the core."""
    spot = await _spot_free_usdt(client)
    earn = await _earn_usdt(client)
    total = spot + earn
    # Set the satellite target ONCE at first setup and PERSIST it — never
    # recompute on restart. While a position is open its value is held as the
    # traded coin (not in spot or earn), so `total` understates the account and
    # a recompute would shrink the target, making the win-sweep bank the
    # satellite's own returning capital as if it were a gain.
    if "satellite_target" not in state:
        state["satellite_target"] = round(total * (1.0 - CORE_FRACTION), 2)
    sat_target = state["satellite_target"]

    if not _is_live():
        # Paper: seed a simulated satellite balance; move no real money.
        if "satellite_cash" not in state:
            state["satellite_cash"] = round(sat_target, 2)
        print(f"[hybrid] paper split — total=${total:.2f} core(80%)=${total*CORE_FRACTION:.2f} "
              f"satellite(20%)=${state['satellite_cash']:.2f} (simulated, no real money moved)")
        return

    # Live: fund the satellite to its target from earn, ONE TIME only.
    if state.get("setup_done"):
        print(f"[hybrid] live split already established — satellite_target=${sat_target:.2f}, "
              f"spot=${spot:.2f}")
        return
    deficit = sat_target - spot
    if deficit > 0.01 and _EARN_PRODUCT_ID:
        try:
            await client.redeem_simple_earn_flexible_product(
                productId=_EARN_PRODUCT_ID, amount=str(round(deficit, 2)))
            print(f"[hybrid] live setup — redeemed ${deficit:.2f} earn->spot to fund satellite; "
                  f"core stays ~${total*CORE_FRACTION:.2f}")
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
    if excess < 0.5:  # not enough to bother
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


# ── Satellite trade execution ────────────────────────────────────────────────
async def _open_position(client, state, symbol: str, sat_cash: float):
    step, min_notional = await _symbol_filters(client, symbol)
    price = await _price(client, symbol)
    notional = min(sat_cash * TRADE_FRACTION, sat_cash - 0.05)
    if notional < max(min_notional, SATELLITE_FLOOR_USD):
        return False
    qty = _round_step(notional / price, step)
    if qty <= 0 or qty * price < max(min_notional, 5.0):
        return False

    if MODE == "dryrun":
        print(f"  [DRYRUN] would BUY {qty} {symbol} @ ~{price:.6f} (~${qty*price:.2f}) "
              f"tp=+{TP_PCT}% sl=-{SL_PCT}% — sending nothing")
        return False

    fill_price = price
    if _is_live():
        try:
            o = await client.order_market_buy(symbol=symbol, quantity=qty)
            fills = o.get("fills", [])
            if fills:
                spent = sum(float(f["price"]) * float(f["qty"]) for f in fills)
                got = sum(float(f["qty"]) for f in fills)
                fill_price = spent / got if got else price
                qty = got
        except Exception as e:
            print(f"  [SAT-OPEN-ERROR] {symbol}: {str(e)[:100]}")
            return False
    else:
        state["satellite_cash"] = round(sat_cash - qty * fill_price, 2)

    state["position"] = {
        "symbol": symbol, "entry_ts": time.time(), "entry_price": fill_price,
        "qty": qty, "tp_price": fill_price * (1 + TP_PCT / 100.0),
        "sl_price": fill_price * (1 - SL_PCT / 100.0), "mode": MODE,
    }
    state["last_entry_ts"] = time.time()
    print(f"  [SAT-OPEN] {symbol} {qty} @ {fill_price:.6f} (~${qty*fill_price:.2f}) "
          f"tp={state['position']['tp_price']:.6f} sl={state['position']['sl_price']:.6f} [{MODE}]")
    return True


async def _close_position(client, state, reason: str):
    pos = state["position"]
    symbol, qty, entry = pos["symbol"], pos["qty"], pos["entry_price"]
    price = await _price(client, symbol)
    exit_price = price
    if _is_live():
        try:
            step, _ = await _symbol_filters(client, symbol)
            # Sell the LESSER of the recorded (bought) qty and the ACTUAL free
            # balance, rounded down. If Binance took the buy fee in the base
            # asset, the real balance is a hair below the recorded qty, and
            # selling the recorded amount would fail "insufficient balance".
            asset = symbol[:-4]
            try:
                bal = await client.get_asset_balance(asset=asset)
                actual = float((bal or {}).get("free", 0.0) or 0.0)
            except Exception:
                actual = qty
            sell_qty = _round_step(min(qty, actual), step)
            if sell_qty <= 0:
                print(f"  [SAT-CLOSE] {symbol} nothing sellable (balance {actual}) — treating as closed")
                state["position"] = None
                return True
            o = await client.order_market_sell(symbol=symbol, quantity=sell_qty)
            fills = o.get("fills", [])
            if fills:
                got = sum(float(f["price"]) * float(f["qty"]) for f in fills)
                sold = sum(float(f["qty"]) for f in fills)
                exit_price = got / sold if sold else price
        except Exception as e:
            print(f"  [SAT-CLOSE-ERROR] {symbol}: {str(e)[:100]} — keeping tracked, will retry")
            return False
    else:
        state["satellite_cash"] = round(float(state.get("satellite_cash", 0.0)) + qty * exit_price, 2)

    gross_pct = (exit_price - entry) / entry * 100.0
    net_pct = gross_pct - FEE_RT_PCT
    held_h = (time.time() - pos["entry_ts"]) / 3600.0
    _log_trade({
        "ts": datetime.now(timezone.utc).isoformat(), "symbol": symbol,
        "held_h": round(held_h, 2), "entry_price": entry, "exit_price": round(exit_price, 8),
        "qty": qty, "gross_pct": round(gross_pct, 4), "net_pct": round(net_pct, 4),
        "reason": reason, "mode": pos.get("mode", MODE),
    })
    print(f"  [SAT-CLOSE] {symbol} @ {exit_price:.6f} net={net_pct:+.2f}% ({reason}) [{MODE}]")
    state["position"] = None
    return True


def _report():
    trades = []
    for line in (open(LEDGER) if os.path.exists(LEDGER) else []):
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
    print("  NOTE: satellite is negative-EV by construction — this is capped 'action',")
    print("        not a growth engine. The core (80% in yield) is where growth lives.")
    print("=" * 66)


async def run():
    global _EARN_PRODUCT_ID
    from binance import AsyncClient

    client = await AsyncClient.create(
        os.getenv("BINANCE_API_KEY") or "x", os.getenv("BINANCE_API_SECRET") or "x")
    state = _load(STATE, {"position": None})
    armed = _is_live()
    banner = {"paper": "📝 PAPER (no real money)",
              "dryrun": "🧪 DRY-RUN (logs orders, sends none)",
              "live": ("🔴 LIVE-ARMED (real spot orders)" if armed
                       else "🔴 live (BLOCKED — arm file missing, no orders)")}
    print(f"[hybrid] start — MODE={MODE} → {banner.get(MODE, MODE)}")
    print(f"[hybrid] barbell — core={CORE_FRACTION*100:.0f}% (yield, untouchable) "
          f"satellite={(1-CORE_FRACTION)*100:.0f}% (auto-speculative, NEGATIVE-EV, capped)")
    print(f"[hybrid] satellite — breakout({BREAKOUT_LOOKBACK}/{INTERVAL}) tp=+{TP_PCT}% sl=-{SL_PCT}% "
          f"floor=${SATELLITE_FLOOR_USD} sweep>+${SWEEP_THRESHOLD_USD} kill={KILL_FILE}")
    try:
        _EARN_PRODUCT_ID = await _resolve_product_id(client)
        await _setup_allocation(client, state)
        _save(STATE, state)

        while True:
            now = time.time()

            # ── Manage an open satellite position (TP / SL / time-stop) ──
            if state.get("position"):
                pos = state["position"]
                try:
                    price = await _price(client, pos["symbol"])
                    held_h = (now - pos["entry_ts"]) / 3600.0
                    reason = None
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

            # ── Drawdown auto-halt (backstop; floor is the primary stop) ──
            dd = _current_drawdown_pct()
            if dd >= MAX_DD_PCT and not _killed():
                open(KILL_FILE, "w").close()
                print(f"[hybrid] 🛑 DRAWDOWN HALT {dd:.1f}% >= {MAX_DD_PCT}% — kill-switch engaged")

            sat_cash = await _satellite_cash(client, state)

            # ── Win-sweep: bank satellite gains into the core (only when flat) ──
            if not state.get("position"):
                sat_cash = await _win_sweep(client, state, sat_cash)
                _save(STATE, state)

            # ── Entry: one position at a time, breakout, floor + cooldown gated ──
            opened = False
            if (not state.get("position") and not _killed()
                    and sat_cash >= max(SATELLITE_FLOOR_USD, 5.0)
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
            elif sat_cash < max(SATELLITE_FLOOR_USD, 5.0) and not state.get("position"):
                print(f"  [SATELLITE-FLOOR] cash ${sat_cash:.2f} < ${max(SATELLITE_FLOOR_USD,5.0):.2f} "
                      f"— sleeve idle (never topped up from core)")

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
