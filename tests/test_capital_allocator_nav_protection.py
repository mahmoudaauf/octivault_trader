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


class _StubPerfTracker:
    def __init__(self, multiplier: float) -> None:
        self._multiplier = multiplier

    def get_size_multiplier(self, symbol: str) -> float:
        return self._multiplier


@pytest.mark.asyncio
async def test_allocate_for_buy_scales_with_symbol_size_multiplier() -> None:
    """The gap this closes (2026-07-16): SymbolPerformanceTracker.get_size_multiplier()
    -- up to 1.25x for a symbol on a proven >=80% win-rate streak -- was computed
    correctly in arbitration_engine.py but never actually consumed by
    allocate_for_buy(); every buy sized identically regardless of that symbol's
    real track record. nav=90 (small-account <$100 path, the actual live path for
    this account) keeps both allocations comfortably between the exchange-min
    floor and the 20%-of-NAV cap, so the multiplier's effect isn't clipped away
    by either bound -- the resulting quantities should differ by exactly the
    ratio of the two multipliers.
    """
    ss_baseline = NativeSharedState()
    ss_baseline.free_balance_usdt = 90.0
    ss_baseline.balance = {"USDT": 90.0}
    alloc_baseline = NativeCapitalAllocator(
        shared_state=ss_baseline, portfolio_manager=_PM(90.0), allocation_pct=15.0,
        perf_tracker=_StubPerfTracker(1.0),
    )
    qty_baseline = await alloc_baseline.allocate_for_buy("BTCUSDT")

    ss_hot = NativeSharedState()
    ss_hot.free_balance_usdt = 90.0
    ss_hot.balance = {"USDT": 90.0}
    alloc_hot = NativeCapitalAllocator(
        shared_state=ss_hot, portfolio_manager=_PM(90.0), allocation_pct=15.0,
        perf_tracker=_StubPerfTracker(1.25),
    )
    qty_hot = await alloc_hot.allocate_for_buy("BTCUSDT")

    assert qty_baseline > 0.0
    assert qty_hot > qty_baseline, (
        f"expected a proven-winner symbol (1.25x) to size larger than baseline "
        f"(1.0x), got hot={qty_hot} vs baseline={qty_baseline} -- the size "
        f"multiplier likely isn't reaching allocate_for_buy() at all"
    )
    ratio = qty_hot / qty_baseline
    assert ratio == pytest.approx(1.25, abs=0.01), (
        f"expected the 1.25x symbol multiplier to scale the allocation by "
        f"exactly that ratio, got {ratio:.4f}"
    )
