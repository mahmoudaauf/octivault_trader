#!/usr/bin/env python3
"""
Unit test for src.l4_execution.safety_order_manager.SafetyOrderManager
=====================================================================
Pure-mock test (no network). Validates:
 1. Plan generation (round_step + filters)
 2. arm_all_positions skips already-protected symbols
 3. cancel_all_safety_orders matches by clientOrderId prefix
 4. dry_run mode does NOT call exchange POST
"""
from __future__ import annotations
import asyncio
import sys
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent))
from src.l4_execution.safety_order_manager import (
    SafetyOrderManager,
    CLIENT_ID_PREFIX,
)


class MockExchange:
    def __init__(self, balances, prices, symbol_info, open_orders=None):
        self.balances = balances
        self.prices = prices
        self.symbol_info = symbol_info
        self.open_orders = open_orders or []
        self.posted: list[dict] = []
        self.cancelled: list[dict] = []

    async def get_account_balances(self):
        return self.balances

    async def get_open_orders(self, symbol=None):
        if symbol:
            return [o for o in self.open_orders if o["symbol"] == symbol]
        return list(self.open_orders)

    async def get_price(self, symbol):
        return float(self.prices.get(symbol, 0))

    async def get_symbol_info(self, symbol):
        return self.symbol_info.get(symbol)

    async def _request(self, method, path, params=None, signed=False, api=None):
        if method == "POST" and path == "/api/v3/order/oco":
            self.posted.append(dict(params))
            return {"orderListId": 12345 + len(self.posted)}
        if method == "DELETE" and path == "/api/v3/order":
            self.cancelled.append(dict(params))
            return {"status": "CANCELED"}
        return {}


class MockSharedState:
    def __init__(self, positions=None):
        self.positions = positions or {}
        self.component_statuses: dict = {}
        self.component_last_seen: dict = {}


def _symbol_info(symbol, tick="0.01", step="0.0001", min_qty="0.0001", min_notional="5"):
    return {
        "symbol": symbol,
        "ocoAllowed": True,
        "filters": [
            {"filterType": "PRICE_FILTER", "tickSize": tick},
            {"filterType": "LOT_SIZE", "stepSize": step, "minQty": min_qty},
            {"filterType": "NOTIONAL", "minNotional": min_notional},
        ],
    }


def cfg(**over):
    base = dict(
        SAFETY_ORDERS_ENABLED=True,
        SAFETY_ORDER_TP_PCT=0.015,
        SAFETY_ORDER_SL_PCT=0.030,
        SAFETY_ORDER_SL_LIMIT_BUFFER=0.003,
        SAFETY_ORDER_MIN_NOTIONAL_USDT=5.0,
        SAFETY_ORDER_RECHECK_INTERVAL=300,
        SAFETY_ORDER_AUTO_ARM_ON_STARTUP=False,
        SAFETY_ORDER_DRY_RUN=False,
    )
    base.update(over)
    return SimpleNamespace(**base)


# ─── Tests ─────────────────────────────────────────────────────────────────
async def test_arms_unprotected_positions():
    ex = MockExchange(
        balances={
            "SOL": {"free": 0.296, "locked": 0.0},
            "DOGE": {"free": 150.0, "locked": 0.0},
            "USDT": {"free": 26.59, "locked": 0.0},
        },
        prices={"SOLUSDT": 84.65, "DOGEUSDT": 0.166},
        symbol_info={
            "SOLUSDT": _symbol_info("SOLUSDT", tick="0.01", step="0.001"),
            "DOGEUSDT": _symbol_info("DOGEUSDT", tick="0.00001", step="1"),
        },
        open_orders=[],
    )
    ss = MockSharedState(positions={
        "SOLUSDT": {"avg_price": 84.40},
        "DOGEUSDT": {"avg_price": 0.165},
    })
    mgr = SafetyOrderManager(shared_state=ss, config=cfg(), exchange_client=ex)
    armed, skipped = await mgr.arm_all_positions()
    assert armed == 2, f"expected 2 armed, got {armed}"
    assert len(ex.posted) == 2
    for p in ex.posted:
        assert p["side"] == "SELL"
        assert "stopPrice" in p and "stopLimitPrice" in p and "price" in p
        assert p["listClientOrderId"].startswith(CLIENT_ID_PREFIX)
    print(f"  ✅ arms_unprotected: armed={armed} posted={len(ex.posted)}")


async def test_skips_already_protected():
    ex = MockExchange(
        balances={"SOL": {"free": 0.296, "locked": 0.0}},
        prices={"SOLUSDT": 84.65},
        symbol_info={"SOLUSDT": _symbol_info("SOLUSDT", tick="0.01", step="0.001")},
        open_orders=[
            {"symbol": "SOLUSDT", "side": "SELL", "type": "STOP_LOSS_LIMIT",
             "orderId": 1, "origQty": "0.296", "price": "82",
             "stopPrice": "82", "clientOrderId": "safety_SOLUSDT_123"},
        ],
    )
    ss = MockSharedState(positions={"SOLUSDT": {"avg_price": 84.40}})
    mgr = SafetyOrderManager(shared_state=ss, config=cfg(), exchange_client=ex)
    armed, skipped = await mgr.arm_all_positions()
    assert armed == 0, f"should skip protected, got armed={armed}"
    assert skipped == 1
    assert len(ex.posted) == 0
    print(f"  ✅ skips_protected: armed={armed} skipped={skipped} posted={len(ex.posted)}")


async def test_dry_run_does_not_post():
    ex = MockExchange(
        balances={"SOL": {"free": 0.296, "locked": 0.0}},
        prices={"SOLUSDT": 84.65},
        symbol_info={"SOLUSDT": _symbol_info("SOLUSDT", tick="0.01", step="0.001")},
    )
    ss = MockSharedState(positions={"SOLUSDT": {"avg_price": 84.40}})
    mgr = SafetyOrderManager(
        shared_state=ss, config=cfg(SAFETY_ORDER_DRY_RUN=True), exchange_client=ex
    )
    armed, skipped = await mgr.arm_all_positions()
    assert armed == 1
    assert len(ex.posted) == 0, "dry-run must not POST"
    print(f"  ✅ dry_run: armed={armed} posted={len(ex.posted)} (none)")


async def test_cancel_only_safety_orders():
    ex = MockExchange(
        balances={},
        prices={},
        symbol_info={},
        open_orders=[
            {"symbol": "SOLUSDT", "orderId": 1, "clientOrderId": "safety_SOLUSDT_1"},
            {"symbol": "DOGEUSDT", "orderId": 2, "clientOrderId": "user_manual_xyz"},
            {"symbol": "ETHUSDT", "orderId": 3, "clientOrderId": "safety_ETHUSDT_2"},
        ],
    )
    ss = MockSharedState()
    mgr = SafetyOrderManager(shared_state=ss, config=cfg(), exchange_client=ex)
    n = await mgr.cancel_all_safety_orders()
    assert n == 2, f"expected to cancel 2, got {n}"
    cancelled_ids = {c["orderId"] for c in ex.cancelled}
    assert cancelled_ids == {1, 3}
    print(f"  ✅ cancel: {n} safety orders, untouched user order")


async def test_filters_min_notional():
    """Position too small (< minNotional) must be skipped."""
    ex = MockExchange(
        balances={"PEPE": {"free": 100, "locked": 0}},   # value ≈ $1
        prices={"PEPEUSDT": 0.01},
        symbol_info={"PEPEUSDT": _symbol_info("PEPEUSDT", min_notional="5")},
    )
    ss = MockSharedState()
    mgr = SafetyOrderManager(shared_state=ss, config=cfg(), exchange_client=ex)
    armed, skipped = await mgr.arm_all_positions()
    assert armed == 0
    assert len(ex.posted) == 0
    print(f"  ✅ min_notional: tiny position correctly skipped")


async def main():
    print("🧪 SafetyOrderManager unit tests")
    print("─" * 60)
    tests = [
        test_arms_unprotected_positions,
        test_skips_already_protected,
        test_dry_run_does_not_post,
        test_cancel_only_safety_orders,
        test_filters_min_notional,
    ]
    fails = 0
    for t in tests:
        try:
            await t()
        except AssertionError as e:
            print(f"  ❌ {t.__name__}: {e}")
            fails += 1
        except Exception as e:
            print(f"  💥 {t.__name__}: {e!r}")
            fails += 1
    print("─" * 60)
    print(f"  Result: {len(tests) - fails}/{len(tests)} passed")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
