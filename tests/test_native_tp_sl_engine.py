"""
Tests for NativeTPSLEngine (Phase 8.3.9).

Covers the API contract consumed by:
- DecisionEngineImpl.evaluate_exit_signals (check_exit_levels)
- meta_controller / execution_manager legacy callers
  (calculate_tp_sl, set_initial_tp_sl, health)

Test groups
-----------
- Construction validation (3)
- calculate_tp_sl pure math (3)
- set_initial_tp_sl + get_targets (3)
- check_exit_levels crossing detection (5)
- tier overrides (2)
- clear / health (3)
- Bootstrap wiring + compat-stub-doesn't-overwrite (3)
"""

from __future__ import annotations

from typing import Any

import pytest

from core_engine.native.app_context import build_native_app_ctx
from core_engine.native.bootstrap import BootstrapConfig, build_components, shutdown_components
from core_engine.native.shared_state import NativeSharedState, Position
from core_engine.native.tp_sl_engine import NativeTPSLEngine


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
def test_rejects_non_positive_tp_pct() -> None:
    with pytest.raises(ValueError):
        NativeTPSLEngine(NativeSharedState(), tp_pct=0.0)


def test_rejects_non_positive_sl_pct() -> None:
    with pytest.raises(ValueError):
        NativeTPSLEngine(NativeSharedState(), sl_pct=-0.01)


def test_default_construction_succeeds() -> None:
    eng = NativeTPSLEngine(NativeSharedState())
    h = eng.health()
    assert h["ok"] is True
    assert h["tracked_symbols"] == 0


# ----------------------------------------------------------------------
# calculate_tp_sl — pure math
# ----------------------------------------------------------------------
def test_calculate_tp_sl_basic() -> None:
    eng = NativeTPSLEngine(NativeSharedState(), tp_pct=0.03, sl_pct=0.02)
    tp, sl = eng.calculate_tp_sl("BTCUSDT", 100.0)
    assert tp == pytest.approx(103.0)
    assert sl == pytest.approx(98.0)


def test_calculate_tp_sl_zero_price_returns_zeros() -> None:
    eng = NativeTPSLEngine(NativeSharedState())
    assert eng.calculate_tp_sl("BTCUSDT", 0.0) == (0.0, 0.0)
    assert eng.calculate_tp_sl("BTCUSDT", -10.0) == (0.0, 0.0)


def test_calculate_tp_sl_does_not_mutate_targets() -> None:
    eng = NativeTPSLEngine(NativeSharedState())
    eng.calculate_tp_sl("BTCUSDT", 50_000.0)
    assert eng.get_targets("BTCUSDT") is None


# ----------------------------------------------------------------------
# set_initial_tp_sl + get_targets
# ----------------------------------------------------------------------
def test_set_initial_tp_sl_persists_targets() -> None:
    eng = NativeTPSLEngine(NativeSharedState(), tp_pct=0.05, sl_pct=0.03)
    eng.set_initial_tp_sl("BTCUSDT", entry_price=10_000.0, qty=0.5)
    t = eng.get_targets("BTCUSDT")
    assert t is not None
    assert t["entry_price"] == 10_000.0
    assert t["qty"] == 0.5
    assert t["tp_price"] == pytest.approx(10_500.0)
    assert t["sl_price"] == pytest.approx(9_700.0)
    assert t["tier"] is None


def test_set_initial_tp_sl_overwrites_prior_entry() -> None:
    eng = NativeTPSLEngine(NativeSharedState())
    eng.set_initial_tp_sl("BTCUSDT", 10_000.0, 1.0)
    eng.set_initial_tp_sl("BTCUSDT", 12_000.0, 2.0)
    t = eng.get_targets("BTCUSDT")
    assert t is not None
    assert t["entry_price"] == 12_000.0
    assert t["qty"] == 2.0


def test_set_initial_tp_sl_ignores_invalid_inputs() -> None:
    eng = NativeTPSLEngine(NativeSharedState())
    eng.set_initial_tp_sl("BTCUSDT", entry_price=0.0, qty=1.0)
    eng.set_initial_tp_sl("BTCUSDT", entry_price=10_000.0, qty=0.0)
    eng.set_initial_tp_sl("BTCUSDT", entry_price=-10.0, qty=1.0)
    assert eng.get_targets("BTCUSDT") is None


# ----------------------------------------------------------------------
# check_exit_levels — crossing detection
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_check_exit_levels_no_targets_returns_none() -> None:
    eng = NativeTPSLEngine(NativeSharedState())
    assert await eng.check_exit_levels("BTCUSDT") is None


@pytest.mark.asyncio
async def test_check_exit_levels_within_band_returns_none() -> None:
    state = NativeSharedState()
    state.positions = {
        "BTCUSDT": Position("BTCUSDT", qty=1.0, entry_price=10_000.0, mark_price=10_100.0),
    }
    eng = NativeTPSLEngine(state, tp_pct=0.03, sl_pct=0.02)
    eng.set_initial_tp_sl("BTCUSDT", entry_price=10_000.0, qty=1.0)
    # mark 10_100 vs tp 10_300 / sl 9_800 → no crossing
    assert await eng.check_exit_levels("BTCUSDT") is None


@pytest.mark.asyncio
async def test_check_exit_levels_tp_hit() -> None:
    state = NativeSharedState()
    state.positions = {
        "BTCUSDT": Position("BTCUSDT", qty=1.0, entry_price=10_000.0, mark_price=10_500.0),
    }
    eng = NativeTPSLEngine(state, tp_pct=0.03, sl_pct=0.02)
    eng.set_initial_tp_sl("BTCUSDT", entry_price=10_000.0, qty=1.0)
    # mark 10_500 >= tp 10_300
    assert await eng.check_exit_levels("BTCUSDT") == "TP_HIT"
    assert eng.health()["tp_hits"] == 1


@pytest.mark.asyncio
async def test_check_exit_levels_sl_hit() -> None:
    state = NativeSharedState()
    state.positions = {
        "BTCUSDT": Position("BTCUSDT", qty=1.0, entry_price=10_000.0, mark_price=9_500.0),
    }
    eng = NativeTPSLEngine(state, tp_pct=0.03, sl_pct=0.02)
    eng.set_initial_tp_sl("BTCUSDT", entry_price=10_000.0, qty=1.0)
    # mark 9_500 <= sl 9_800
    assert await eng.check_exit_levels("BTCUSDT") == "SL_HIT"
    assert eng.health()["sl_hits"] == 1


@pytest.mark.asyncio
async def test_check_exit_levels_falls_back_to_price_cache() -> None:
    """If position has zero mark_price, fall back to shared_state.price_cache."""
    state = NativeSharedState()
    state.price_cache = {"BTCUSDT": 10_500.0}
    # No position registered with mark_price
    eng = NativeTPSLEngine(state, tp_pct=0.03, sl_pct=0.02)
    eng.set_initial_tp_sl("BTCUSDT", entry_price=10_000.0, qty=1.0)
    assert await eng.check_exit_levels("BTCUSDT") == "TP_HIT"


# ----------------------------------------------------------------------
# Tier overrides
# ----------------------------------------------------------------------
def test_tier_override_used_in_calculation() -> None:
    eng = NativeTPSLEngine(
        NativeSharedState(),
        tp_pct=0.03,
        sl_pct=0.02,
        tier_overrides={"swing": (0.10, 0.05)},
    )
    tp, sl = eng.calculate_tp_sl("BTCUSDT", 100.0, tier="swing")
    assert tp == pytest.approx(110.0)
    assert sl == pytest.approx(95.0)


def test_unknown_tier_falls_back_to_default() -> None:
    eng = NativeTPSLEngine(
        NativeSharedState(),
        tp_pct=0.03,
        sl_pct=0.02,
        tier_overrides={"swing": (0.10, 0.05)},
    )
    tp, sl = eng.calculate_tp_sl("BTCUSDT", 100.0, tier="nonexistent")
    assert tp == pytest.approx(103.0)
    assert sl == pytest.approx(98.0)


# ----------------------------------------------------------------------
# clear / health
# ----------------------------------------------------------------------
def test_clear_drops_targets() -> None:
    eng = NativeTPSLEngine(NativeSharedState())
    eng.set_initial_tp_sl("BTCUSDT", 10_000.0, 1.0)
    assert eng.get_targets("BTCUSDT") is not None
    eng.clear("BTCUSDT")
    assert eng.get_targets("BTCUSDT") is None


def test_clear_is_idempotent_for_unknown_symbol() -> None:
    eng = NativeTPSLEngine(NativeSharedState())
    eng.clear("UNKNOWN")  # must not raise


@pytest.mark.asyncio
async def test_health_counts_checks_and_hits() -> None:
    state = NativeSharedState()
    state.positions = {
        "BTCUSDT": Position("BTCUSDT", qty=1.0, entry_price=10_000.0, mark_price=10_100.0),
    }
    eng = NativeTPSLEngine(state, tp_pct=0.03, sl_pct=0.02)
    eng.set_initial_tp_sl("BTCUSDT", entry_price=10_000.0, qty=1.0)
    await eng.check_exit_levels("BTCUSDT")
    await eng.check_exit_levels("BTCUSDT")
    h = eng.health()
    assert h["checks"] == 2
    assert h["tracked_symbols"] == 1
    assert h["tp_pct"] == 0.03
    assert h["sl_pct"] == 0.02


# ----------------------------------------------------------------------
# Bootstrap wiring
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_bootstrap_attaches_native_tp_sl_engine() -> None:
    components = await build_components(
        _min_cfg(), exchange_client_factory=lambda _c: _StubExchangeClient()
    )
    try:
        assert isinstance(components.tp_sl_engine, NativeTPSLEngine)
    finally:
        await shutdown_components(components)


@pytest.mark.asyncio
async def test_native_tp_sl_engine_visible_in_app_ctx() -> None:
    components = await build_components(
        _min_cfg(), exchange_client_factory=lambda _c: _StubExchangeClient()
    )
    try:
        app_ctx, _orch = build_native_app_ctx(components)
        assert isinstance(app_ctx["tp_sl_engine"], NativeTPSLEngine)
    finally:
        await shutdown_components(components)


@pytest.mark.asyncio
async def test_compat_stub_does_not_overwrite_native_tp_sl_engine() -> None:
    components = await build_components(
        _min_cfg(), exchange_client_factory=lambda _c: _StubExchangeClient()
    )
    try:
        app_ctx, _orch = build_native_app_ctx(components, compat=True)
        assert isinstance(app_ctx["tp_sl_engine"], NativeTPSLEngine)
    finally:
        await shutdown_components(components)


@pytest.mark.asyncio
async def test_bootstrap_uses_tp_sl_pct_from_config() -> None:
    components = await build_components(
        _min_cfg(tp_pct=0.07, sl_pct=0.04),
        exchange_client_factory=lambda _c: _StubExchangeClient(),
    )
    try:
        h = components.tp_sl_engine.health()
        assert h["tp_pct"] == 0.07
        assert h["sl_pct"] == 0.04
    finally:
        await shutdown_components(components)
