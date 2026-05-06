#!/usr/bin/env python3
"""
🧪 FAILURE-MODE TEST HARNESS
============================
Validates the 3 active failure modes against real positions WITHOUT touching
real funds. Combines:

  Mode #1  Sideways drift   → time-based forced-exit logic (simulator)
  Mode #2  Drawdown         → -3% software stop check (simulator)
  Mode #3  Stop-loss orders → exchange-native order presence (live read)

Usage:
  python _test_failure_modes.py              # run all 3 tests
  python _test_failure_modes.py --mode 3     # only mode N
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import sys
import time
from pathlib import Path
from urllib.parse import urlencode

import requests

# Reuse env loader pattern
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


def _http(method, url, **kw):
    kw.setdefault("timeout", 20)
    h = kw.pop("headers", {}) or {}
    h["Connection"] = "close"
    last = None
    for i in range(3):
        try:
            return requests.request(method, url, headers=h, **kw)
        except Exception as e:
            last = e
            time.sleep(1.5 * (i + 1))
    raise last


def _signed(method, path, params=None):
    params = dict(params or {})
    params["timestamp"] = int(time.time() * 1000)
    params.setdefault("recvWindow", 5000)
    qs = urlencode(params)
    sig = hmac.new(SEC.encode(), qs.encode(), hashlib.sha256).hexdigest()
    r = _http(method, f"{URL}{path}?{qs}&signature={sig}", headers={"X-MBX-APIKEY": KEY})
    if r.status_code >= 400:
        raise RuntimeError(f"{method} {path} → HTTP {r.status_code}: {r.text[:300]}")
    return r.json()


# ─── Test fixtures ─────────────────────────────────────────────────────────
POSITIONS = [
    {"symbol": "ETHUSDT", "qty": 0.0109, "entry": 2344.91},
    {"symbol": "SOLUSDT", "qty": 0.296, "entry": 84.40},
    {"symbol": "XRPUSDT", "qty": 17.8, "entry": 1.3986},
]

# Config thresholds (mirrors src/l0_core/config.py)
STOP_LOSS_PCT = -0.03  # -3%
TAKE_PROFIT_PCT = +0.10  # +10%
PARTIAL_TP_PCT = +0.005  # +0.5%
PARTIAL_TP_AGE_S = 30
SIDEWAYS_TP_PCT = +0.003  # tiny edge after long stagnation
SIDEWAYS_MAX_AGE_H = 24  # force-exit after 24h flat


# ─── Mode #1: Sideways drift simulator ─────────────────────────────────────
def should_force_exit_sideways(entry: float, current: float, age_s: float) -> tuple[bool, str]:
    """Replicates fourth_slot_tracker MAX_DURATION_REACHED + tiny-profit logic."""
    age_h = age_s / 3600.0
    pct = (current - entry) / entry
    if age_h >= SIDEWAYS_MAX_AGE_H and abs(pct) < 0.005:
        return True, f"MAX_DURATION_REACHED age={age_h:.1f}h pct={pct*100:+.2f}%"
    # Tiny-profit force-exit only for stagnant winners (< 1.5%) — let real winners run
    if age_h >= 12 and SIDEWAYS_TP_PCT <= pct < 0.015:
        return True, f"SIDEWAYS_TINY_TP age={age_h:.1f}h pct={pct*100:+.2f}%"
    return False, f"hold age={age_h:.1f}h pct={pct*100:+.2f}%"


def test_mode_1_sideways():
    print("\n🧪 MODE #1: SIDEWAYS DRIFT ─ time-based forced exit")
    print("─" * 72)
    cases = [
        # (label, age_seconds, current_price_factor, expect_exit)
        ("fresh +0.1% (1h)", 3600, 1.001, False),
        ("flat 12h", 12 * 3600, 1.000, False),
        ("flat 24h drift -0.2%", 24 * 3600, 0.998, True),
        ("flat 24h drift +0.4%", 24 * 3600, 1.004, True),
        ("12h tiny-profit +0.4%", 12 * 3600, 1.004, True),
        ("48h drift +0.1%", 48 * 3600, 1.001, True),
        ("48h profit +5%", 48 * 3600, 1.05, False),  # let winner run
    ]
    p = 100.0
    pass_n = fail_n = 0
    for label, age, fac, expect in cases:
        cur = p * fac
        ex, why = should_force_exit_sideways(p, cur, age)
        ok = ex == expect
        flag = "✅" if ok else "❌"
        print(f"  {flag} {label:<28} → exit={ex}  ({why})")
        pass_n += ok
        fail_n += not ok
    print(f"  Result: {pass_n}/{pass_n+fail_n} passed")
    return fail_n == 0


# ─── Mode #2: Drawdown simulator ───────────────────────────────────────────
def should_stop_loss(entry: float, current: float) -> tuple[bool, str]:
    pct = (current - entry) / entry
    if pct <= STOP_LOSS_PCT:
        return True, f"SL_HIT pct={pct*100:+.2f}% ≤ {STOP_LOSS_PCT*100:.1f}%"
    return False, f"SL_OK pct={pct*100:+.2f}%"


def should_take_profit(entry: float, current: float) -> tuple[bool, str]:
    pct = (current - entry) / entry
    if pct >= TAKE_PROFIT_PCT:
        return True, f"TP_HIT pct={pct*100:+.2f}%"
    return False, f"TP_OK pct={pct*100:+.2f}%"


def test_mode_2_drawdown():
    print("\n🧪 MODE #2: DRAWDOWN ─ software stop-loss / take-profit logic")
    print("─" * 72)
    cases = [
        ("flat", 1.000, False, False),
        ("-1% dip", 0.990, False, False),
        ("-2.99% edge", 0.9701, False, False),
        ("-3% trigger", 0.970, True, False),
        ("-5% loss", 0.950, True, False),
        ("+5% partial", 1.050, False, False),
        ("+10% TP", 1.100, False, True),
        ("+15% jackpot", 1.150, False, True),
    ]
    p = 100.0
    pass_n = fail_n = 0
    for label, fac, exp_sl, exp_tp in cases:
        cur = p * fac
        sl, why_sl = should_stop_loss(p, cur)
        tp, why_tp = should_take_profit(p, cur)
        ok = (sl == exp_sl) and (tp == exp_tp)
        flag = "✅" if ok else "❌"
        print(f"  {flag} {label:<14} SL={sl} TP={tp}  ({why_sl}; {why_tp})")
        pass_n += ok
        fail_n += not ok
    print(f"  Result: {pass_n}/{pass_n+fail_n} passed")
    return fail_n == 0


# ─── Mode #3: Exchange-native order presence (LIVE read-only) ──────────────
def test_mode_3_exchange_orders():
    print("\n🧪 MODE #3: EXCHANGE-NATIVE STOP-LOSS PRESENCE (live read)")
    print("─" * 72)
    if not KEY or not SEC:
        print("  ⚠ no API keys — skipping live check")
        return False
    try:
        orders = _signed("GET", "/api/v3/openOrders")
    except Exception as e:
        print(f"  ❌ failed to fetch open orders: {e}")
        return False

    print(f"  Open orders on Binance: {len(orders)}")
    by_sym = {}
    for o in orders:
        by_sym.setdefault(o["symbol"], []).append(o)

    pass_n = fail_n = 0
    for pos in POSITIONS:
        sym = pos["symbol"]
        sym_orders = by_sym.get(sym, [])
        has_sl = any(
            o.get("type") in ("STOP_LOSS_LIMIT", "STOP_LOSS") and o.get("side") == "SELL"
            for o in sym_orders
        )
        has_tp = any(
            o.get("type") in ("TAKE_PROFIT_LIMIT", "LIMIT_MAKER", "LIMIT")
            and o.get("side") == "SELL"
            for o in sym_orders
        )
        protected = has_sl  # SL is the must-have
        flag = "✅" if protected else "❌"
        print(f"  {flag} {sym:<10} orders={len(sym_orders)}  SL={has_sl}  TP={has_tp}")
        for o in sym_orders:
            print(
                f"        ↳ {o['type']:<22} {o['side']} qty={o['origQty']} "
                f"price={o['price']} stop={o.get('stopPrice','-')}"
            )
        pass_n += protected
        fail_n += not protected

    if fail_n:
        print(f"\n  ⚠️  {fail_n} position(s) UNPROTECTED on exchange.")
        print("      Run: python _arm_safety_orders.py --live")
    print(f"  Result: {pass_n}/{pass_n+fail_n} positions protected")
    return fail_n == 0


# ─── Main ──────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=int, choices=[1, 2, 3], help="run only one mode")
    args = ap.parse_args()

    print(f"🌐 Binance {'TESTNET' if TESTNET else 'LIVE'}: {URL}")
    print(f"📊 Testing against {len(POSITIONS)} known positions")

    results = {}
    if not args.mode or args.mode == 1:
        results["sideways"] = test_mode_1_sideways()
    if not args.mode or args.mode == 2:
        results["drawdown"] = test_mode_2_drawdown()
    if not args.mode or args.mode == 3:
        results["exchange_sl"] = test_mode_3_exchange_orders()

    print("\n" + "═" * 72)
    print("📋 FINAL SUMMARY")
    print("═" * 72)
    for k, v in results.items():
        flag = "✅ PASS" if v else "❌ FAIL"
        print(f"  {flag}  {k}")
    overall = all(results.values())
    print(f"\n  Overall: {'✅ ALL PASS' if overall else '❌ ACTION REQUIRED'}")
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
