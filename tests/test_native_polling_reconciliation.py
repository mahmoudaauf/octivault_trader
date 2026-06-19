from __future__ import annotations

import pytest

from core_engine.native.polling_coordinator import NativePollingConfig, NativePollingCoordinator
from core_engine.native.shared_state import NativeSharedState, Position


class _StubExchange:
    def __init__(self, balances: dict[str, float], prices: dict[str, float] | None = None) -> None:
        self._balances = dict(balances)
        self._prices = prices or {}

    async def get_balance(self) -> dict[str, float]:
        return dict(self._balances)

    async def get_prices(self, symbols: list[str] | None = None) -> dict[str, float]:
        if symbols is None:
            return dict(self._prices)
        return {s: self._prices[s] for s in symbols if s in self._prices}

    def is_throttled(self) -> bool:
        return False

    def last_error(self) -> str:
        return ""


@pytest.mark.asyncio
async def test_balance_sync_reconciles_wallet_holdings_into_positions() -> None:
    state = NativeSharedState()
    state.prices = {"XRPUSDT": 0.55}
    exchange = _StubExchange({"USDT": 20.0, "XRP": 100.0})
    pc = NativePollingCoordinator(
        shared_state=state,
        exchange_client=exchange,
        config=NativePollingConfig(enable_active_trades_gate=False),
    )

    await pc._fetch_and_sync_balance()

    assert state.free_balance_usdt == 20.0
    assert "XRPUSDT" in state.positions
    assert state.positions["XRPUSDT"]["qty"] == 100.0
    assert state.positions["XRPUSDT"]["_mirrored"] is True


@pytest.mark.asyncio
async def test_position_price_refresh_updates_price_cache() -> None:
    """Batch price fetch updates price_cache for all held symbols."""
    state = NativeSharedState()
    # Seed positions as dict (hydration-style) with stale entry prices
    state.positions["BIOUSDT"] = {
        "symbol": "BIOUSDT", "qty": 10.0, "entry_price": 0.50,
        "mark_price": 0.50, "current_price": 0.50,
    }
    state.positions["STOUSDT"] = {
        "symbol": "STOUSDT", "qty": 5.0, "entry_price": 1.20,
        "mark_price": 1.20, "current_price": 1.20,
    }
    # Exchange returns current (higher) prices
    exchange = _StubExchange(
        balances={"USDT": 10.0},
        prices={"BIOUSDT": 0.65, "STOUSDT": 1.45},
    )
    pc = NativePollingCoordinator(
        shared_state=state,
        exchange_client=exchange,
        config=NativePollingConfig(enable_active_trades_gate=False),
    )

    await pc._fetch_and_refresh_position_prices()

    assert state.price_cache["BIOUSDT"] == 0.65
    assert state.price_cache["STOUSDT"] == 1.45


@pytest.mark.asyncio
async def test_position_price_refresh_nav_accuracy() -> None:
    """NAV computed after price refresh matches real market value."""
    state = NativeSharedState()
    state.free_balance_usdt = 3.48
    state.positions["BIOUSDT"] = {
        "symbol": "BIOUSDT", "qty": 100.0, "entry_price": 0.50,
        "mark_price": 0.50, "current_price": 0.50,
    }
    state.positions["ETHUSDT"] = {
        "symbol": "ETHUSDT", "qty": 0.01, "entry_price": 1600.0,
        "mark_price": 1600.0, "current_price": 1600.0,
    }
    exchange = _StubExchange(
        balances={"USDT": 3.48},
        prices={"BIOUSDT": 0.62, "ETHUSDT": 1685.0},
    )
    pc = NativePollingCoordinator(
        shared_state=state,
        exchange_client=exchange,
        config=NativePollingConfig(enable_active_trades_gate=False),
    )

    # Before refresh: stale entry prices used
    nav_before = state.free_balance_usdt + state.get_portfolio_value()
    assert abs(nav_before - (3.48 + 100 * 0.50 + 0.01 * 1600.0)) < 0.01

    # After refresh: live prices
    await pc._fetch_and_refresh_position_prices()
    nav_after = state.free_balance_usdt + state.get_portfolio_value()
    expected = 3.48 + 100 * 0.62 + 0.01 * 1685.0
    assert abs(nav_after - expected) < 0.01


@pytest.mark.asyncio
async def test_position_price_refresh_skips_when_no_positions() -> None:
    """No REST call made when positions dict is empty."""
    state = NativeSharedState()
    call_count = 0

    class _CountingExchange(_StubExchange):
        async def get_prices(self, symbols=None):
            nonlocal call_count
            call_count += 1
            return {}

    exchange = _CountingExchange(balances={"USDT": 10.0})
    pc = NativePollingCoordinator(
        shared_state=state,
        exchange_client=exchange,
        config=NativePollingConfig(enable_active_trades_gate=False),
    )

    await pc._fetch_and_refresh_position_prices()
    assert call_count == 0


@pytest.mark.asyncio
async def test_position_price_refresh_dict_positions_not_wiped() -> None:
    """Dict-based positions survive the 25s position sync loop (regression for NAV wipe bug)."""
    state = NativeSharedState()
    state.positions["WLFIUSDT"] = {
        "symbol": "WLFIUSDT", "qty": 500.0, "entry_price": 0.08,
        "mark_price": 0.08, "current_price": 0.08,
    }
    state.price_cache["WLFIUSDT"] = 0.09

    exchange = _StubExchange(balances={"USDT": 5.0})
    pc = NativePollingCoordinator(
        shared_state=state,
        exchange_client=exchange,
        config=NativePollingConfig(enable_active_trades_gate=False),
    )

    await pc._fetch_and_sync_positions()

    # Position must still exist and have non-zero qty
    assert "WLFIUSDT" in state.positions
    qty = state.positions["WLFIUSDT"]
    if isinstance(qty, dict):
        assert qty["qty"] > 0
    else:
        assert qty.qty > 0
