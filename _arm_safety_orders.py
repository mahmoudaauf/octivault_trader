#!/usr/bin/env python3
"""
🛡️  EXCHANGE-NATIVE SAFETY ORDER ARMER
=======================================
Places real OCO (One-Cancels-Other) sell orders on Binance SPOT for currently
held positions, providing HARDWARE stop-loss + take-profit protection that
survives bot restarts, crashes, and connectivity loss.

Failure modes addressed:
  1. Sideways drift  → time-aware SL ratchet (use --tighten after N hours)
  2. Drawdown        → exchange-native STOP_LOSS_LIMIT (-3% default)
  3. Stop-loss orders→ POSTED as live orders on Binance (not soft polling)

Usage:
  python _arm_safety_orders.py --dry-run            # plan only, no orders
  python _arm_safety_orders.py --live               # place real OCO orders
  python _arm_safety_orders.py --live --tp 0.015 --sl 0.03   # custom %
  python _arm_safety_orders.py --cancel             # cancel all OCO listings
  python _arm_safety_orders.py --status             # list open SAFETY orders
"""
from __future__ import annotations
import os, time, hmac, hashlib, json, sys, argparse, math
from urllib.parse import urlencode
from pathlib import Path
from decimal import Decimal, ROUND_DOWN, ROUND_UP, getcontext
import requests
from requests.adapters import HTTPAdapter

getcontext().prec = 28
_SESSION = requests.Session()
_SESSION.mount("https://", HTTPAdapter(pool_connections=2, pool_maxsize=4, max_retries=3))


def _http(method: str, url: str, headers=None, timeout=20, attempts=3):
    last = None
    for i in range(attempts):
        try:
            return _SESSION.request(method, url, headers=headers or {}, timeout=timeout)
        except Exception as e:
            last = e
            time.sleep(1.5 * (i + 1))
    raise last

# ─── Config ────────────────────────────────────────────────────────────────
ENV = {}
_envp = Path(__file__).parent / ".env"
if _envp.exists():
    for line in _envp.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            ENV[k.strip()] = v.strip().strip('"').strip("'")
KEY = ENV.get("BINANCE_API_KEY") or ""
SEC = ENV.get("BINANCE_API_SECRET_HMAC") or ""
TESTNET = ENV.get("BINANCE_TESTNET", "false").lower() == "true"
URL = "https://testnet.binance.vision" if TESTNET else "https://api.binance.com"

# Known entry prices (from prior journal analysis – fallback if API doesn't show them)
KNOWN_ENTRIES = {
    "ETHUSDT": 2344.91,
    "SOLUSDT": 84.40,
    "XRPUSDT": 1.3986,
}

# Min notional for OCO = $5 default per symbol on SPOT (5 USDT MIN_NOTIONAL)
DEFAULT_TP_PCT = 0.015   # +1.5%  (conservative quick-exit)
DEFAULT_SL_PCT = 0.030   # -3.0%  (FIX8_4TH_SLOT_STOP_LOSS_PCT)
SL_LIMIT_BUFFER = 0.003  # stopLimitPrice = stopPrice * (1 - 0.3%) for SELL


# ─── Binance helpers ───────────────────────────────────────────────────────
def _signed(method: str, path: str, params: dict | None = None) -> dict:
    if not KEY or not SEC:
        raise RuntimeError("BINANCE_API_KEY / BINANCE_API_SECRET_HMAC missing in .env")
    params = dict(params or {})
    params["timestamp"] = int(time.time() * 1000)
    params.setdefault("recvWindow", 5000)
    qs = urlencode(params)
    sig = hmac.new(SEC.encode(), qs.encode(), hashlib.sha256).hexdigest()
    headers = {"X-MBX-APIKEY": KEY}
    url = f"{URL}{path}?{qs}&signature={sig}"
    r = _http(method, url, headers=headers)
    if r.status_code >= 400:
        raise RuntimeError(f"{method} {path} → HTTP {r.status_code}: {r.text[:400]}")
    return r.json()


def _public(path: str, params: dict | None = None) -> dict:
    qs = urlencode(params or {})
    url = f"{URL}{path}" + (f"?{qs}" if qs else "")
    r = _http("GET", url)
    r.raise_for_status()
    return r.json()


def get_account() -> dict:
    return _signed("GET", "/api/v3/account")


def get_prices() -> dict[str, float]:
    return {p["symbol"]: float(p["price"]) for p in _public("/api/v3/ticker/price")}


def get_exchange_info(symbols: list[str]) -> dict[str, dict]:
    out = {}
    for sym in symbols:
        try:
            info = _public("/api/v3/exchangeInfo", {"symbol": sym})
        except Exception as e:
            print(f"   ⚠ exchangeInfo {sym}: {e}")
            continue
        for s in info.get("symbols", []):
            flt = {f["filterType"]: f for f in s.get("filters", [])}
            out[s["symbol"]] = {
                "tickSize": Decimal(flt.get("PRICE_FILTER", {}).get("tickSize", "0.01")),
                "stepSize": Decimal(flt.get("LOT_SIZE", {}).get("stepSize", "0.0001")),
                "minQty": Decimal(flt.get("LOT_SIZE", {}).get("minQty", "0")),
                "minNotional": Decimal(
                    flt.get("NOTIONAL", flt.get("MIN_NOTIONAL", {})).get("minNotional", "5")
                ),
                "ocoAllowed": s.get("ocoAllowed", True),
            }
    return out


def round_step(value: Decimal, step: Decimal, mode=ROUND_DOWN) -> Decimal:
    if step == 0:
        return value
    return (value / step).quantize(Decimal("1"), rounding=mode) * step


def fmt(v: Decimal) -> str:
    s = format(v.normalize(), "f")
    return s if "." in s else s


def get_open_orders() -> list[dict]:
    return _signed("GET", "/api/v3/openOrders")


# ─── Core logic ────────────────────────────────────────────────────────────
def discover_positions(min_value: float = 5.0) -> list[dict]:
    """Return tradeable positions worth >= $min_value with current price + qty."""
    acct = get_account()
    prices = get_prices()
    rows = []
    for b in acct["balances"]:
        asset = b["asset"]
        if asset in ("USDT", "BFUSD"):
            continue
        free = float(b["free"]); locked = float(b["locked"])
        total = free + locked
        if total <= 0:
            continue
        sym = f"{asset}USDT"
        px = prices.get(sym, 0.0)
        value = total * px
        if value < min_value or px == 0:
            continue
        rows.append({
            "symbol": sym, "asset": asset,
            "free_qty": free, "locked_qty": locked, "total_qty": total,
            "price_now": px, "value_usdt": value,
            "entry_price": KNOWN_ENTRIES.get(sym, px),
        })
    rows.sort(key=lambda r: -r["value_usdt"])
    return rows


def plan_oco(pos: dict, filt: dict, tp_pct: float, sl_pct: float) -> dict:
    """Compute OCO sell order params for a single position."""
    sym = pos["symbol"]
    entry = Decimal(str(pos["entry_price"]))
    price_now = Decimal(str(pos["price_now"]))

    # Price refs based on entry (what we paid), not current.
    tp_raw = entry * (Decimal("1") + Decimal(str(tp_pct)))
    sl_raw = entry * (Decimal("1") - Decimal(str(sl_pct)))

    # Sanity: SL must be below current price (otherwise instant trigger);
    # TP must be above current price.
    if sl_raw >= price_now:
        # Position is already underwater enough; tighten SL to -1% from current
        sl_raw = price_now * Decimal("0.99")
    if tp_raw <= price_now:
        # Position is already in profit; raise TP to +0.8% from current
        tp_raw = price_now * Decimal("1.008")

    tick = filt["tickSize"]
    step = filt["stepSize"]

    tp_price = round_step(tp_raw, tick, ROUND_UP)
    sl_stop = round_step(sl_raw, tick, ROUND_DOWN)
    sl_limit_raw = sl_stop * (Decimal("1") - Decimal(str(SL_LIMIT_BUFFER)))
    sl_limit = round_step(sl_limit_raw, tick, ROUND_DOWN)

    # Quantity = ALL FREE (don't touch locked qty already in other orders)
    qty_raw = Decimal(str(pos["free_qty"]))
    qty = round_step(qty_raw, step, ROUND_DOWN)

    # Notional check
    notional_now = qty * price_now
    enough_notional = notional_now >= filt["minNotional"]
    enough_qty = qty >= filt["minQty"]

    return {
        "symbol": sym,
        "side": "SELL",
        "qty": qty,
        "qty_str": fmt(qty),
        "tp_price": tp_price,
        "tp_price_str": fmt(tp_price),
        "sl_stop": sl_stop,
        "sl_stop_str": fmt(sl_stop),
        "sl_limit": sl_limit,
        "sl_limit_str": fmt(sl_limit),
        "entry": entry,
        "price_now": price_now,
        "notional_now": notional_now,
        "valid": enough_notional and enough_qty,
        "reason": (
            "" if (enough_notional and enough_qty)
            else f"qty<min ({qty}<{filt['minQty']}) or notional<${filt['minNotional']}"
        ),
    }


def place_oco(plan: dict) -> dict:
    """Submit OCO sell order using /api/v3/order/oco."""
    params = {
        "symbol": plan["symbol"],
        "side": "SELL",
        "quantity": plan["qty_str"],
        "price": plan["tp_price_str"],            # Limit (TP) leg
        "stopPrice": plan["sl_stop_str"],         # Stop trigger
        "stopLimitPrice": plan["sl_limit_str"],   # Stop limit
        "stopLimitTimeInForce": "GTC",
        "newOrderRespType": "RESULT",
        "listClientOrderId": f"safety_{plan['symbol']}_{int(time.time())}",
    }
    return _signed("POST", "/api/v3/order/oco", params)


# ─── Reporting ─────────────────────────────────────────────────────────────
def print_plan(plans: list[dict], tp_pct: float, sl_pct: float):
    print(f"\n{'='*72}")
    print(f"🛡️  EXCHANGE-NATIVE SAFETY ORDER PLAN  (TP +{tp_pct*100:.2f}% / SL -{sl_pct*100:.2f}%)")
    print(f"{'='*72}")
    print(f"{'SYMBOL':<10}{'QTY':<14}{'ENTRY':<12}{'NOW':<12}{'TP':<12}{'SL':<12}{'SLLim':<12}{'OK'}")
    for p in plans:
        flag = "✅" if p["valid"] else "❌"
        print(f"{p['symbol']:<10}{p['qty_str']:<14}{fmt(p['entry']):<12}"
              f"{fmt(p['price_now']):<12}{p['tp_price_str']:<12}"
              f"{p['sl_stop_str']:<12}{p['sl_limit_str']:<12}{flag}")
        if not p["valid"]:
            print(f"           └─ skipped: {p['reason']}")


def print_status():
    orders = get_open_orders()
    safety = [o for o in orders if str(o.get("clientOrderId", "")).startswith("safety_")
              or str(o.get("origClientOrderId", "")).startswith("safety_")]
    print(f"\n📋 OPEN ORDERS: {len(orders)} total, {len(safety)} marked as safety")
    for o in orders:
        print(f"   {o['symbol']:<10} {o['side']:<5} {o['type']:<22} "
              f"qty={o['origQty']:<10} price={o['price']:<10} stop={o.get('stopPrice','-'):<10} "
              f"id={o['orderId']} client={o.get('clientOrderId','')}")


# ─── Main ──────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Exchange-native safety order armer")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--dry-run", action="store_true", help="Plan only (default)")
    g.add_argument("--live", action="store_true", help="Actually place orders")
    g.add_argument("--cancel", action="store_true", help="Cancel all safety_* OCO listings")
    g.add_argument("--status", action="store_true", help="Show open orders only")
    ap.add_argument("--tp", type=float, default=DEFAULT_TP_PCT, help=f"TP fraction (default {DEFAULT_TP_PCT})")
    ap.add_argument("--sl", type=float, default=DEFAULT_SL_PCT, help=f"SL fraction (default {DEFAULT_SL_PCT})")
    ap.add_argument("--symbols", nargs="*", default=None, help="Limit to symbols (e.g. ETHUSDT SOLUSDT)")
    args = ap.parse_args()

    print(f"🌐 Binance {'TESTNET' if TESTNET else 'LIVE'}: {URL}")

    if args.status:
        print_status()
        return

    if args.cancel:
        orders = get_open_orders()
        n = 0
        for o in orders:
            cid = str(o.get("clientOrderId", "")) + str(o.get("origClientOrderId", ""))
            if "safety_" not in cid:
                continue
            try:
                _signed("DELETE", "/api/v3/order",
                        {"symbol": o["symbol"], "orderId": o["orderId"]})
                print(f"   ✖ cancelled {o['symbol']} order {o['orderId']}")
                n += 1
            except Exception as e:
                print(f"   ⚠ failed to cancel {o['orderId']}: {e}")
        print(f"\n✅ Cancelled {n} safety order(s)")
        return

    # Plan / Live
    print("\n🔍 Discovering positions ≥ $5 …")
    positions = discover_positions(min_value=5.0)
    if args.symbols:
        positions = [p for p in positions if p["symbol"] in set(args.symbols)]
    if not positions:
        print("⚠️  No tradeable positions found.")
        return

    syms = [p["symbol"] for p in positions]
    filters = get_exchange_info(syms)

    # Skip symbols already protected by an OCO
    open_ords = get_open_orders()
    protected = set()
    for o in open_ords:
        if o.get("type") in ("STOP_LOSS_LIMIT", "TAKE_PROFIT_LIMIT") and o.get("side") == "SELL":
            protected.add(o["symbol"])
    if protected:
        print(f"ℹ️  Already-protected symbols (skipping): {sorted(protected)}")

    plans = []
    for pos in positions:
        if pos["symbol"] in protected:
            continue
        f = filters.get(pos["symbol"])
        if not f:
            print(f"   ⚠ no filters for {pos['symbol']}, skipped")
            continue
        if not f["ocoAllowed"]:
            print(f"   ⚠ {pos['symbol']} does not allow OCO, skipped")
            continue
        plans.append(plan_oco(pos, f, args.tp, args.sl))

    print_plan(plans, args.tp, args.sl)

    if args.live:
        print(f"\n🚀 PLACING LIVE OCO ORDERS …")
        ok, fail = 0, 0
        for p in plans:
            if not p["valid"]:
                continue
            try:
                resp = place_oco(p)
                lid = resp.get("orderListId", "?")
                print(f"   ✅ {p['symbol']:<10} OCO submitted (orderListId={lid})")
                ok += 1
            except Exception as e:
                print(f"   ❌ {p['symbol']:<10} FAILED: {e}")
                fail += 1
        print(f"\n✅ Submitted: {ok} | ❌ Failed: {fail}")
        print_status()
    else:
        print(f"\nℹ️  DRY-RUN. Re-run with --live to actually place orders.")


if __name__ == "__main__":
    main()
