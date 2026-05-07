"""
Tests for NativePortfolioManager (Phase 8.3.7).

Covers the API contract consumed by SituationEngineImpl and meta_controller:
- get_nav (canonical, derived fallback, zero state)
- get_positions, .positions property
- get_pnl (sum across positions; sign of P&L)
- get_capital_allocated (canonical + sum-of-positions fallback)
- get_capital_available (canonical + balance fallback)
- get_dust_state / get_dust_record (unknown symbol, dust threshold)
- bootstrap wiring: app_ctx["portfolio_manager"] is the real impl
- compat stub does NOT overwrite the native portfolio_manager
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from core_engine.native.app_context import build_native_app_ctx
from core_engine.native.bootstrap import BootstrapConfig, build_components
from core_engine.native.portfolio_manager import NativePortfolioManager
from core_engine.native.shared_state import NativeSharedState, Position


# ----------------------------------------------------------------------
# Stub plumbing (mirrors test_native_bootstrap fixtures)
# ----------------------------------------------------------------------
class _StubExchangeClient:
    def __init__(self, *_: Any, **__: Any) -> None:
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1

    async def get_account(self) -> dict[str, Any]:
        return {"balances": []}

    async def get_ticker_prices(self) -> dict[str, float]:
        return {}

    async def get_klines(self, *a: Any, **kw: Any) -> list[Any]:
        return []


class _StubBalanceSync:
    """Minimal NativeBalanceSync stand-in for unit tests."""

    def __init__(self, balances: dict[str, float] | None = None) -> None:
        self._balances = dict(balances or {})

    def get_balance(self) -> dict[str, float]:
        return dict(self._balances)


def _min_cfg(**overrides: Any) -> BootstrapConfig:
    base: dict[str, Any] = {
        "api_key": "k",
        "api_secret": "s",
        "testnet": True,
        "symbols": ["BTCUSDT"],
    }
    base.update(overrides)
    return BootstrapConfig(**base)


# ----------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------
def test_rejects_non_positive_min_order_usdt() -> None:
    with pytest.raises(ValueError):
        NativePortfolioManager(
            shared_state=NativeSharedState(),
            balance_sync=_StubBalanceSync(),
            min_order_usdt=0.0,
        )
    with pytest.raises(ValueError):
        NativePortfolioManager(
            shared_state=NativeSharedState(),
            balance_sync=_StubBalanceSync(),
            min_order_usdt=-5.0,
        )


# ----------------------------------------------------------------------
# get_nav
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_get_nav_uses_canonical_when_set() -> None:
    state = NativeSharedState()
    state.nav_usdt = 1234.56
    pm = NativePortfolioManager(state, _StubBalanceSync({"USDT": 999.0}))
    assert await pm.get_nav() == 1234.56


@pytest.mark.asyncio
async def test_get_nav_falls_back_to_balance_plus_positions() -> None:
    state = NativeSharedState()
    # nav_usdt left at 0 → trigger fallback
    state.positions = {
        "BTCUSDT": Position(symbol="BTCUSDT", qty=0.1, entry_price=50_000.0, mark_price=60_000.0),
    }
    pm = NativePortfolioManager(state, _StubBalanceSync({"USDT": 1_000.0}))
    # 1000 free + (0.1 * 60000) = 7000
    assert await pm.get_nav() == 7_000.0


@pytest.mark.asyncio
async def test_get_nav_zero_when_state_empty() -> None:
    pm = NativePortfolioManager(NativeSharedState(), _StubBalanceSync())
    assert await pm.get_nav() == 0.0


# ----------------------------------------------------------------------
# capital available / allocated
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_get_capital_available_prefers_shared_state() -> None:
    state = NativeSharedState()
    state.free_balance_usdt = 500.0
    pm = NativePortfolioManager(state, _StubBalanceSync({"USDT": 999.0}))
    assert await pm.get_capital_available() == 450.0


@pytest.mark.asyncio
async def test_get_capital_available_subtracts_active_reservations() -> None:
    state = NativeSharedState()
    state.free_balance_usdt = 500.0
    state.reserve_quote("USDT", 100.0, ttl_sec=30.0, reservation_id="r1", created_at=time.time())
    pm = NativePortfolioManager(state, _StubBalanceSync({"USDT": 999.0}))
    assert await pm.get_capital_available() == 350.0


@pytest.mark.asyncio
async def test_get_capital_available_prunes_expired_reservations() -> None:
    state = NativeSharedState()
    state.free_balance_usdt = 500.0
    state.reserve_quote("USDT", 100.0, ttl_sec=1.0, reservation_id="stale", created_at=1.0)
    state.prune_quote_reservations("USDT", now_ts=1000.0)
    pm = NativePortfolioManager(state, _StubBalanceSync({"USDT": 999.0}))
    assert await pm.get_capital_available() == 450.0


@pytest.mark.asyncio
async def test_get_capital_available_falls_back_to_balance_sync() -> None:
    pm = NativePortfolioManager(NativeSharedState(), _StubBalanceSync({"USDT": 250.0}))
    assert await pm.get_capital_available() == 225.0


@pytest.mark.asyncio
async def test_get_capital_available_falls_back_to_shared_state_balance_without_sync() -> None:
    state = NativeSharedState()
    state.balance = {"USDT": 100.0}
    pm = NativePortfolioManager(state, None)
    assert await pm.get_capital_available() == 90.0


@pytest.mark.asyncio
async def test_get_capital_allocated_prefers_canonical() -> None:
    state = NativeSharedState()
    state.invested_capital_usdt = 4_000.0
    state.positions = {
        "BTCUSDT": Position("BTCUSDT", qty=1.0, entry_price=10_000, mark_price=11_000)
    }
    pm = NativePortfolioManager(state, _StubBalanceSync())
    # canonical wins; not 11_000
    assert await pm.get_capital_allocated() == 4_000.0


@pytest.mark.asyncio
async def test_get_capital_allocated_sums_positions_when_canonical_zero() -> None:
    state = NativeSharedState()
    state.positions = {
        "BTCUSDT": Position("BTCUSDT", qty=1.0, entry_price=10_000, mark_price=11_000),
        "ETHUSDT": Position("ETHUSDT", qty=2.0, entry_price=2_000, mark_price=2_100),
    }
    pm = NativePortfolioManager(state, _StubBalanceSync())
    # 1*11000 + 2*2100 = 15200
    assert await pm.get_capital_allocated() == 15_200.0


# ----------------------------------------------------------------------
# positions
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_get_positions_returns_symbol_to_qty() -> None:
    state = NativeSharedState()
    state.positions = {
        "BTCUSDT": Position("BTCUSDT", qty=0.5, entry_price=50_000, mark_price=51_000),
        "ETHUSDT": Position("ETHUSDT", qty=10.0, entry_price=2_000, mark_price=1_950),
    }
    pm = NativePortfolioManager(state, _StubBalanceSync())
    result = await pm.get_positions()
    assert result == {"BTCUSDT": 0.5, "ETHUSDT": 10.0}


@pytest.mark.asyncio
async def test_get_positions_skips_zero_qty() -> None:
    state = NativeSharedState()
    state.positions = {
        "BTCUSDT": Position("BTCUSDT", qty=0.0, entry_price=50_000, mark_price=50_000),
        "ETHUSDT": Position("ETHUSDT", qty=1.5, entry_price=2_000, mark_price=2_100),
    }
    pm = NativePortfolioManager(state, _StubBalanceSync())
    assert await pm.get_positions() == {"ETHUSDT": 1.5}


def test_positions_property_returns_snapshot_copy() -> None:
    state = NativeSharedState()
    pos = Position("BTCUSDT", qty=1.0, entry_price=50_000, mark_price=51_000)
    state.positions = {"BTCUSDT": pos}
    pm = NativePortfolioManager(state, _StubBalanceSync())

    snap = pm.positions
    assert snap == {"BTCUSDT": pos}
    # External mutation must not corrupt L0 state.
    snap.clear()
    assert state.positions == {"BTCUSDT": pos}


# ----------------------------------------------------------------------
# P&L
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_get_pnl_positive() -> None:
    state = NativeSharedState()
    state.positions = {
        # +20% on 1 BTC at 10k entry, 12k mark = +2000
        "BTCUSDT": Position("BTCUSDT", qty=1.0, entry_price=10_000.0, mark_price=12_000.0),
    }
    pm = NativePortfolioManager(state, _StubBalanceSync())
    assert await pm.get_pnl() == 2_000.0


@pytest.mark.asyncio
async def test_get_pnl_negative_and_summed() -> None:
    state = NativeSharedState()
    state.positions = {
        # +500
        "BTCUSDT": Position("BTCUSDT", qty=1.0, entry_price=10_000.0, mark_price=10_500.0),
        # -200
        "ETHUSDT": Position("ETHUSDT", qty=2.0, entry_price=2_000.0, mark_price=1_900.0),
    }
    pm = NativePortfolioManager(state, _StubBalanceSync())
    assert await pm.get_pnl() == 300.0


@pytest.mark.asyncio
async def test_get_pnl_skips_invalid_positions() -> None:
    state = NativeSharedState()
    state.positions = {
        # zero qty
        "AAA": Position("AAA", qty=0.0, entry_price=100.0, mark_price=200.0),
        # zero entry
        "BBB": Position("BBB", qty=1.0, entry_price=0.0, mark_price=200.0),
        # zero mark
        "CCC": Position("CCC", qty=1.0, entry_price=100.0, mark_price=0.0),
        # valid
        "DDD": Position("DDD", qty=1.0, entry_price=100.0, mark_price=110.0),
    }
    pm = NativePortfolioManager(state, _StubBalanceSync())
    assert await pm.get_pnl() == 10.0


# ----------------------------------------------------------------------
# Dust
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_get_dust_state_unknown_symbol() -> None:
    pm = NativePortfolioManager(NativeSharedState(), _StubBalanceSync(), min_order_usdt=10.0)
    state = await pm.get_dust_state("UNKNOWN")
    assert state == {
        "symbol": "UNKNOWN",
        "is_dust": False,
        "qty": 0.0,
        "value_usdt": 0.0,
        "threshold_usdt": 10.0,
    }


@pytest.mark.asyncio
async def test_get_dust_state_below_threshold_is_dust() -> None:
    state = NativeSharedState()
    # 0.001 BTC * 5000 = 5 USDT < 10 → dust
    state.positions = {
        "BTCUSDT": Position("BTCUSDT", qty=0.001, entry_price=5_000.0, mark_price=5_000.0),
    }
    pm = NativePortfolioManager(state, _StubBalanceSync(), min_order_usdt=10.0)
    result = await pm.get_dust_state("BTCUSDT")
    assert result["is_dust"] is True
    assert result["value_usdt"] == 5.0
    assert result["threshold_usdt"] == 10.0


@pytest.mark.asyncio
async def test_get_dust_state_above_threshold_is_not_dust() -> None:
    state = NativeSharedState()
    state.positions = {
        "BTCUSDT": Position("BTCUSDT", qty=1.0, entry_price=50_000.0, mark_price=50_000.0),
    }
    pm = NativePortfolioManager(state, _StubBalanceSync(), min_order_usdt=10.0)
    result = await pm.get_dust_state("BTCUSDT")
    assert result["is_dust"] is False


@pytest.mark.asyncio
async def test_get_dust_record_aliases_get_dust_state() -> None:
    state = NativeSharedState()
    state.positions = {
        "ETHUSDT": Position("ETHUSDT", qty=0.001, entry_price=2_000.0, mark_price=2_000.0),
    }
    pm = NativePortfolioManager(state, _StubBalanceSync(), min_order_usdt=10.0)
    record = await pm.get_dust_record("ETHUSDT")
    state_payload = await pm.get_dust_state("ETHUSDT")
    assert record == state_payload


# ----------------------------------------------------------------------
# Bootstrap wiring
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_bootstrap_attaches_native_portfolio_manager() -> None:
    components = await build_components(
        _min_cfg(), exchange_client_factory=lambda _c: _StubExchangeClient()
    )
    try:
        assert isinstance(components.portfolio_manager, NativePortfolioManager)
    finally:
        from core_engine.native.bootstrap import shutdown_components

        await shutdown_components(components)


@pytest.mark.asyncio
async def test_native_portfolio_manager_visible_in_app_ctx() -> None:
    components = await build_components(
        _min_cfg(), exchange_client_factory=lambda _c: _StubExchangeClient()
    )
    try:
        app_ctx, _orch = build_native_app_ctx(components)
        assert isinstance(app_ctx["portfolio_manager"], NativePortfolioManager)
    finally:
        from core_engine.native.bootstrap import shutdown_components

        await shutdown_components(components)


@pytest.mark.asyncio
async def test_compat_stub_does_not_overwrite_native_portfolio_manager() -> None:
    """
    register_compat_stubs uses dict.setdefault, so the real impl must
    win when compat=True is requested.
    """
    components = await build_components(
        _min_cfg(), exchange_client_factory=lambda _c: _StubExchangeClient()
    )
    try:
        app_ctx, _orch = build_native_app_ctx(components, compat=True)
        assert isinstance(app_ctx["portfolio_manager"], NativePortfolioManager)
    finally:
        from core_engine.native.bootstrap import shutdown_components

        await shutdown_components(components)
