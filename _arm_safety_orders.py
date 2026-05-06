#!/usr/bin/env python3
"""
🛡️  SAFETY ORDER CLI WRAPPER
============================
Thin CLI that delegates to the canonical L4 component
`src.l4_execution.safety_order_manager.SafetyOrderManager`.

The orchestrator already wires SafetyOrderManager into the live runtime, so
under normal operation no manual invocation is required. This script is for
ad-hoc / out-of-band use and for verification.

Usage:
  python _arm_safety_orders.py --dry-run            # plan only
  python _arm_safety_orders.py --live               # arm OCO orders
  python _arm_safety_orders.py --cancel             # cancel safety_* orders
  python _arm_safety_orders.py --status             # list open orders
  python _arm_safety_orders.py --live --tp 0.015 --sl 0.03
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import hmac
import logging
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlencode

import requests

# Load .env
ENV: dict = {}
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

# Make repo importable
sys.path.insert(0, str(Path(__file__).parent))
from src.l4_execution.safety_order_manager import CLIENT_ID_PREFIX, SafetyOrderManager


# ─── HTTP helpers ──────────────────────────────────────────────────────────
def _http(method, url, headers=None, **kw):
    kw.setdefault("timeout", 20)
    h = dict(headers or {})
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


# ─── Stub clients (subset of real interfaces used by SafetyOrderManager) ───
class _StubExchangeClient:
    async def get_account_balances(self) -> dict:
        d = _signed("GET", "/api/v3/account")
        return {
            b["asset"]: {"free": float(b["free"]), "locked": float(b["locked"])}
            for b in d.get("balances", [])
            if float(b["free"]) + float(b["locked"]) > 0
        }

    async def get_open_orders(self, symbol=None) -> list:
        params = {"symbol": symbol} if symbol else None
        return _signed("GET", "/api/v3/openOrders", params)

    async def get_price(self, symbol) -> float:
        r = _http("GET", f"{URL}/api/v3/ticker/price?symbol={symbol}")
        r.raise_for_status()
        return float(r.json()["price"])

    async def get_symbol_info(self, symbol):
        r = _http("GET", f"{URL}/api/v3/exchangeInfo?symbol={symbol}")
        r.raise_for_status()
        for s in r.json().get("symbols", []):
            if s["symbol"] == symbol:
                return s
        return None

    async def _request(self, method, path, params=None, signed=False, api="spot_api"):
        if signed:
            return _signed(method, path, params)
        qs = urlencode(params or {})
        url = f"{URL}{path}" + (f"?{qs}" if qs else "")
        r = _http(method, url)
        r.raise_for_status()
        return r.json()


class _StubSharedState:
    def __init__(self):
        self.positions: dict = {}
        self.component_statuses: dict = {}
        self.component_last_seen: dict = {}


# ─── Output helpers ────────────────────────────────────────────────────────
def print_status():
    orders = _signed("GET", "/api/v3/openOrders")
    safety = [
        o
        for o in orders
        if str(o.get("clientOrderId", "")).startswith(CLIENT_ID_PREFIX)
        or str(o.get("origClientOrderId", "")).startswith(CLIENT_ID_PREFIX)
    ]
    print(f"\n📋 OPEN ORDERS: {len(orders)} total, {len(safety)} are safety_*")
    for o in orders:
        print(
            f"   {o['symbol']:<10} {o['side']:<5} {o['type']:<22} "
            f"qty={o['origQty']:<10} price={o['price']:<10} "
            f"stop={o.get('stopPrice','-'):<10} id={o['orderId']} "
            f"client={o.get('clientOrderId','')}"
        )


def make_config(args) -> SimpleNamespace:
    return SimpleNamespace(
        SAFETY_ORDERS_ENABLED=True,
        SAFETY_ORDER_TP_PCT=args.tp,
        SAFETY_ORDER_SL_PCT=args.sl,
        SAFETY_ORDER_SL_LIMIT_BUFFER=0.003,
        SAFETY_ORDER_MIN_NOTIONAL_USDT=5.0,
        SAFETY_ORDER_RECHECK_INTERVAL=300,
        SAFETY_ORDER_AUTO_ARM_ON_STARTUP=False,  # CLI is explicit
        SAFETY_ORDER_DRY_RUN=args.dry_run,
    )


# ─── Main ──────────────────────────────────────────────────────────────────
async def amain():
    ap = argparse.ArgumentParser(
        description="Safety order CLI (delegates to L4 SafetyOrderManager)"
    )
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--dry-run", action="store_true", help="Plan only (no orders)")
    g.add_argument("--live", action="store_true", help="Place real OCO orders")
    g.add_argument("--cancel", action="store_true", help="Cancel safety_* orders")
    g.add_argument("--status", action="store_true", help="List open orders")
    ap.add_argument("--tp", type=float, default=0.015)
    ap.add_argument("--sl", type=float, default=0.030)
    args = ap.parse_args()

    if not (args.dry_run or args.live or args.cancel or args.status):
        args.dry_run = True

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    print(f"🌐 Binance {'TESTNET' if TESTNET else 'LIVE'}: {URL}")

    if args.status:
        print_status()
        return

    cfg = make_config(args)
    mgr = SafetyOrderManager(
        shared_state=_StubSharedState(),
        config=cfg,
        exchange_client=_StubExchangeClient(),
    )

    if args.cancel:
        n = await mgr.cancel_all_safety_orders()
        print(f"✅ Cancelled {n} safety_* order(s)")
        print_status()
        return

    armed, skipped = await mgr.arm_all_positions()
    label = "DRY-RUN" if args.dry_run else "LIVE"
    print(f"\n{label}: armed={armed} skipped={skipped}")
    if not args.dry_run:
        print_status()


if __name__ == "__main__":
    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        sys.exit(130)
