"""
Remediation item #17: NativeCapitalAllocator must read
shared_state.nav_protection_state["protection_floor_usdt"] and pass it into
DailyCompoundingPolicy.sizing_nav() so the floor actually constrains position
sizing, not just gets computed and logged.
"""
from __future__ import annotations

import pytest

from core_engine.native.capital_allocator import NativeCapitalAllocator
from core_engine.native.shared_state import NativeSharedState


class _PM:
    def __init__(self, nav: float) -> None:
        self._nav = nav

    async def get_nav(self) -> float:
        return self._nav


@pytest.mark.asyncio
async def test_allocate_for_buy_respects_nav_protection_floor() -> None:
    ss = NativeSharedState()
    ss.free_balance_usdt = 100.0
    ss.balance = {"USDT": 100.0}
    # Simulate what main.py's 60s evaluate_nav_protection() call writes.
    ss.nav_protection_state = {"protection_floor_usdt": 98.0}

    alloc = NativeCapitalAllocator(
        shared_state=ss, portfolio_manager=_PM(100.0), allocation_pct=100.0,
    )
    quote = await alloc.allocate_for_buy("BTCUSDT")

    # Only $2 of the $100 NAV is above the protection floor -- allocation must
    # not exceed that, regardless of allocation_pct=100.
    assert quote <= 2.0 + 1e-6, (
        f"allocated {quote:.2f} USDT but NAV protection floor left only $2.00 risk-eligible"
    )


@pytest.mark.asyncio
async def test_allocate_for_buy_unaffected_when_no_nav_protection_state() -> None:
    """No nav_protection_state yet (e.g. before main.py's first 60s evaluation)
    must behave exactly as before this change — no floor applied."""
    ss = NativeSharedState()
    ss.free_balance_usdt = 100.0
    ss.balance = {"USDT": 100.0}
    assert ss.nav_protection_state == {}

    alloc = NativeCapitalAllocator(
        shared_state=ss, portfolio_manager=_PM(100.0), allocation_pct=5.0,
    )
    quote = await alloc.allocate_for_buy("BTCUSDT")
    assert quote > 0.0
