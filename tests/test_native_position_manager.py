"""
Tests for NativePositionManager (Phase 8.3.8).

Covers the API contract consumed by SituationEngine.analyze_position
and DecisionEngine.make_sell_decision:
- get_position (live, missing, zero-qty)
- analyze_position (full schema, dust detection, risk classification)
- bootstrap wiring: app_ctx["position_manager"] is the real impl
- compat stub does NOT overwrite the native position_manager
"""

from __future__ import annotations

from typing import Any

import pytest

from core_engine.native.app_context import build_native_app_ctx
from core_engine.native.bootstrap import BootstrapConfig, build_components, shutdown_components
from core_engine.native.position_manager import NativePositionManager
from core_engine.native.shared_state import NativeSharedState, Position


# ----------------------------------------------------------------------
# Stub plumbing
# ----------------------------------------------------------------------
class _StubExchangeClient:
    async def close(self) -> None:
        return None

    async def get_account(self) -> dict[str, Any]:
        return {"balances": []}

    async def get_ticker_prices(self) -> dict[str, float]:
        return {}

    async def get_klines(self, *a: Any, **kw: Any) -> list[Any]:
        return []


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
        NativePositionManager(NativeSharedState(), min_order_usdt=0.0)


def test_rejects_inverted_risk_thresholds() -> None:
    with pytest.raises(ValueError):
        NativePositionManager(
            NativeSharedState(),
            risk_high_pct=2.0,
            risk_med_pct=5.0,
        )


def test_rejects_negative_risk_med() -> None:
    with pytest.raises(ValueError):
        NativePositionManager(
            NativeSharedState(),
            risk_high_pct=10.0,
            risk_med_pct=-1.0,
        )


# ----------------------------------------------------------------------
# get_position
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_get_position_returns_live_position() -> None:
    state = NativeSharedState()
    pos = Position("BTCUSDT", qty=0.5, entry_price=50_000.0, mark_price=51_000.0)
    state.positions = {"BTCUSDT": pos}
    pm = NativePositionManager(state)
    assert await pm.get_position("BTCUSDT") is pos


@pytest.mark.asyncio
async def test_get_position_returns_none_for_unknown_symbol() -> None:
    pm = NativePositionManager(NativeSharedState())
    assert await pm.get_position("UNKNOWN") is None


@pytest.mark.asyncio
async def test_get_position_returns_none_for_zero_qty() -> None:
    state = NativeSharedState()
    state.positions = {
        "BTCUSDT": Position("BTCUSDT", qty=0.0, entry_price=50_000.0, mark_price=51_000.0),
    }
    pm = NativePositionManager(state)
    assert await pm.get_position("BTCUSDT") is None


@pytest.mark.asyncio
async def test_get_position_supports_dict_backed_hydrated_positions() -> None:
    state = NativeSharedState()
    state.positions = {
        "BNBUSDT": {"qty": 0.25, "entry_price": 600.0, "mark_price": 650.0},
    }
    pm = NativePositionManager(state)
    pos = await pm.get_position("BNBUSDT")
    assert isinstance(pos, dict)
    assert pos["qty"] == 0.25


# ----------------------------------------------------------------------
# analyze_position — schema + numerics
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_analyze_position_unknown_returns_empty_schema() -> None:
    pm = NativePositionManager(NativeSharedState())
    result = await pm.analyze_position("UNKNOWN")
    assert result == {
        "quantity": 0.0,
        "entry_price": 0.0,
        "current_price": 0.0,
        "p_and_l": 0.0,
        "p_and_l_pct": 0.0,
        "status": "ACTIVE",
        "risk_level": "LOW",
    }


@pytest.mark.asyncio
async def test_analyze_position_computes_pnl_correctly() -> None:
    state = NativeSharedState()
    state.positions = {
        # 1 BTC entry 10k, mark 12k → +2000 USDT, +20%
        "BTCUSDT": Position("BTCUSDT", qty=1.0, entry_price=10_000.0, mark_price=12_000.0),
    }
    pm = NativePositionManager(state, risk_high_pct=10.0, risk_med_pct=3.0)
    result = await pm.analyze_position("BTCUSDT")
    assert result["quantity"] == 1.0
    assert result["entry_price"] == 10_000.0
    assert result["current_price"] == 12_000.0
    assert result["p_and_l"] == 2_000.0
    assert result["p_and_l_pct"] == pytest.approx(20.0)


@pytest.mark.asyncio
async def test_analyze_position_supports_dict_backed_hydrated_positions() -> None:
    state = NativeSharedState()
    state.positions = {
        "BNBUSDT": {"qty": 0.5, "entry_price": 600.0, "mark_price": 660.0},
    }
    pm = NativePositionManager(state, risk_high_pct=10.0, risk_med_pct=3.0)
    result = await pm.analyze_position("BNBUSDT")
    assert result["quantity"] == 0.5
    assert result["entry_price"] == 600.0
    assert result["current_price"] == 660.0
    assert result["p_and_l"] == pytest.approx(30.0)
    assert result["p_and_l_pct"] == pytest.approx(10.0)


# ----------------------------------------------------------------------
# Risk classification
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_risk_low_when_pnl_under_med_threshold() -> None:
    state = NativeSharedState()
    # +1% (under 3% MED threshold)
    state.positions = {
        "BTCUSDT": Position("BTCUSDT", qty=1.0, entry_price=10_000.0, mark_price=10_100.0),
    }
    pm = NativePositionManager(state, risk_high_pct=10.0, risk_med_pct=3.0)
    assert (await pm.analyze_position("BTCUSDT"))["risk_level"] == "LOW"


@pytest.mark.asyncio
async def test_risk_medium_at_med_threshold() -> None:
    state = NativeSharedState()
    state.positions = {
        # +3% exactly
        "BTCUSDT": Position("BTCUSDT", qty=1.0, entry_price=10_000.0, mark_price=10_300.0),
    }
    pm = NativePositionManager(state, risk_high_pct=10.0, risk_med_pct=3.0)
    assert (await pm.analyze_position("BTCUSDT"))["risk_level"] == "MEDIUM"


@pytest.mark.asyncio
async def test_risk_high_at_high_threshold() -> None:
    state = NativeSharedState()
    state.positions = {
        # +10%
        "BTCUSDT": Position("BTCUSDT", qty=1.0, entry_price=10_000.0, mark_price=11_000.0),
    }
    pm = NativePositionManager(state, risk_high_pct=10.0, risk_med_pct=3.0)
    assert (await pm.analyze_position("BTCUSDT"))["risk_level"] == "HIGH"


@pytest.mark.asyncio
async def test_risk_uses_absolute_value_for_losses() -> None:
    state = NativeSharedState()
    # -15% loss → still HIGH risk
    state.positions = {
        "BTCUSDT": Position("BTCUSDT", qty=1.0, entry_price=10_000.0, mark_price=8_500.0),
    }
    pm = NativePositionManager(state, risk_high_pct=10.0, risk_med_pct=3.0)
    result = await pm.analyze_position("BTCUSDT")
    assert result["risk_level"] == "HIGH"
    assert result["p_and_l_pct"] == pytest.approx(-15.0)


# ----------------------------------------------------------------------
# Status classification
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_status_active_for_normal_size_position() -> None:
    state = NativeSharedState()
    state.positions = {
        "BTCUSDT": Position("BTCUSDT", qty=1.0, entry_price=50_000.0, mark_price=50_000.0),
    }
    pm = NativePositionManager(state, min_order_usdt=10.0)
    assert (await pm.analyze_position("BTCUSDT"))["status"] == "ACTIVE"


@pytest.mark.asyncio
async def test_status_dust_locked_below_min_order_threshold() -> None:
    state = NativeSharedState()
    # 0.001 BTC * 5000 = 5 USDT < 10 → dust
    state.positions = {
        "BTCUSDT": Position("BTCUSDT", qty=0.001, entry_price=5_000.0, mark_price=5_000.0),
    }
    pm = NativePositionManager(state, min_order_usdt=10.0)
    assert (await pm.analyze_position("BTCUSDT"))["status"] == "DUST_LOCKED"


# ----------------------------------------------------------------------
# Bootstrap wiring
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_bootstrap_attaches_native_position_manager() -> None:
    components = await build_components(
        _min_cfg(), exchange_client_factory=lambda _c: _StubExchangeClient()
    )
    try:
        assert isinstance(components.position_manager, NativePositionManager)
    finally:
        await shutdown_components(components)


@pytest.mark.asyncio
async def test_native_position_manager_visible_in_app_ctx() -> None:
    components = await build_components(
        _min_cfg(), exchange_client_factory=lambda _c: _StubExchangeClient()
    )
    try:
        app_ctx, _orch = build_native_app_ctx(components)
        assert isinstance(app_ctx["position_manager"], NativePositionManager)
    finally:
        await shutdown_components(components)


@pytest.mark.asyncio
async def test_compat_stub_does_not_overwrite_native_position_manager() -> None:
    components = await build_components(
        _min_cfg(), exchange_client_factory=lambda _c: _StubExchangeClient()
    )
    try:
        app_ctx, _orch = build_native_app_ctx(components, compat=True)
        assert isinstance(app_ctx["position_manager"], NativePositionManager)
    finally:
        await shutdown_components(components)
