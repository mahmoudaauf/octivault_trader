"""
Tests for position hydration and startup state machine integration (Phase 8.4).

Verifies:
1. Hydration engine reconstructs positions from fills
2. Startup state machine enforces correct progression
3. BUY decisions blocked until READY state
4. Integration with bootstrap, orchestrator, decisions
"""

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from core_engine.native.position_hydration_engine import (
    HydratedPosition,
    HydrationState,
    NativePositionHydrationEngine,
)
from core_engine.native.shared_state import NativeSharedState
from core_engine.native.startup_state_machine import (
    NativeStartupStateMachine,
    StartupState,
)

# ──────────────────────────────────────────────────────────────────────────────
# Test Position Hydration Engine
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_hydration_engine_instantiates():
    """Test that hydration engine can be instantiated."""
    ss = NativeSharedState()
    engine = NativePositionHydrationEngine(
        shared_state=ss,
        trade_journal=None,
        exchange_client=None,
    )
    assert engine is not None


@pytest.mark.asyncio
async def test_hydration_engine_reconstructs_positions_from_fills(tmp_path):
    """Test reconstructing positions from fills."""
    ss = NativeSharedState()
    engine = NativePositionHydrationEngine(
        shared_state=ss,
        trade_journal=None,
        exchange_client=None,
        journal_dir=str(tmp_path),
    )

    # Create a mock trade journal file
    journal_file = tmp_path / "trade_journal_20260507.jsonl"
    fills = [
        {
            "epoch": 1620000000.0,
            "event": "ORDER_FILLED",
            "symbol": "AVAXUSDT",
            "side": "BUY",
            "qty": 0.01,
            "price": 100.00,
            "fee": 0.002,
        },
        {
            "epoch": 1620000100.0,
            "event": "ORDER_FILLED",
            "symbol": "AVAXUSDT",
            "side": "BUY",
            "qty": 0.005,
            "price": 101.00,
            "fee": 0.001,
        },
    ]

    with open(journal_file, "w") as f:
        for fill in fills:
            f.write(json.dumps(fill) + "\n")

    # Mock exchange client
    engine._exchange = AsyncMock()
    engine._exchange.get_account = AsyncMock(
        return_value={
            "balances": [
                {"asset": "USDT", "free": "99.00", "locked": "0.0"},
                {"asset": "AVAX", "free": "0.015", "locked": "0.0"},
            ]
        }
    )

    # Run hydration
    state = await engine.hydrate()

    # Verify results
    assert state.success
    assert "AVAXUSDT" in state.positions
    pos = state.positions["AVAXUSDT"]
    assert pos.qty == 0.015
    assert pos.avg_entry_price == pytest.approx(100.33, rel=0.01)  # Weighted average
    assert pos.realized_pnl == 0.0  # No sells
    assert state.total_balance_usdt == pytest.approx(99.0)


@pytest.mark.asyncio
async def test_hydration_engine_applies_to_shared_state():
    """Test applying hydrated positions to shared state."""
    ss = NativeSharedState()

    # Initialize shared state positions dict
    ss.positions = {}

    engine = NativePositionHydrationEngine(
        shared_state=ss,
        trade_journal=None,
        exchange_client=None,
    )

    # Create a hydration state
    hydrated_pos = HydratedPosition(
        symbol="AVAXUSDT",
        qty=0.01,
        avg_entry_price=100.0,
        current_price=101.0,
        unrealized_pnl=0.01,
        realized_pnl=0.0,
        tp_price=101.5,
        sl_price=99.0,
    )

    state = HydrationState(
        success=True,
        message="test",
        positions={"AVAXUSDT": hydrated_pos},
    )

    # Apply to shared state
    await engine.apply_to_shared_state(state)

    # Verify shared state was updated
    assert "AVAXUSDT" in ss.positions
    assert ss.positions["AVAXUSDT"]["qty"] == 0.01
    assert ss.positions["AVAXUSDT"]["entry_price"] == 100.0


# ──────────────────────────────────────────────────────────────────────────────
# Test Startup State Machine
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_startup_state_machine_instantiates():
    """Test that state machine can be instantiated."""
    sm = NativeStartupStateMachine()
    assert sm.current_state() == StartupState.BOOTING
    assert not sm.is_ready()


@pytest.mark.asyncio
async def test_startup_state_machine_can_buy_gating():
    """Test that can_buy() returns False until READY."""
    sm = NativeStartupStateMachine()

    # BOOTING → can't buy
    assert not sm.can_buy()

    # HYDRATING → can't buy
    sm._state = StartupState.HYDRATING
    assert not sm.can_buy()

    # RECONCILING → can't buy
    sm._state = StartupState.RECONCILING
    assert not sm.can_buy()

    # VALIDATING → can't buy
    sm._state = StartupState.VALIDATING
    assert not sm.can_buy()

    # READY → can buy
    sm._state = StartupState.READY
    assert sm.can_buy()

    # FAILED → can't buy
    sm._state = StartupState.FAILED
    assert not sm.can_buy()


@pytest.mark.asyncio
async def test_startup_state_machine_progression():
    """Test full state machine progression."""
    sm = NativeStartupStateMachine()

    # Register simple callbacks
    callback_order = []

    async def hydrate_cb():
        callback_order.append("hydrate")
        return True

    async def reconcile_cb():
        callback_order.append("reconcile")
        return True

    async def validate_cb():
        callback_order.append("validate")
        return True

    sm.set_callback(StartupState.HYDRATING, hydrate_cb)
    sm.set_callback(StartupState.RECONCILING, reconcile_cb)
    sm.set_callback(StartupState.VALIDATING, validate_cb)

    # Run startup
    success = await sm.run_startup(timeout_sec=10.0)

    # Verify progression
    assert success
    assert sm.is_ready()
    assert sm.current_state() == StartupState.READY
    assert callback_order == ["hydrate", "reconcile", "validate"]


@pytest.mark.asyncio
async def test_startup_state_machine_failure_handling():
    """Test failure handling in state machine."""
    sm = NativeStartupStateMachine()

    async def failing_callback():
        raise RuntimeError("Test failure")

    sm.set_callback(StartupState.HYDRATING, failing_callback)

    # Run startup
    success = await sm.run_startup(timeout_sec=10.0)

    # Verify failure state
    assert not success
    assert sm.current_state() == StartupState.FAILED
    assert not sm.is_ready()


@pytest.mark.asyncio
async def test_startup_state_machine_timeout():
    """Test timeout handling."""
    sm = NativeStartupStateMachine()

    async def slow_callback():
        await asyncio.sleep(10.0)  # Longer than timeout

    sm.set_callback(StartupState.HYDRATING, slow_callback)

    # Run startup with short timeout
    success = await sm.run_startup(timeout_sec=0.1)

    # Verify timeout handling
    assert not success
    assert sm.current_state() == StartupState.FAILED


@pytest.mark.asyncio
async def test_startup_state_machine_transition_history():
    """Test transition history tracking."""
    sm = NativeStartupStateMachine()

    # Make some transitions
    await sm._transition_to(StartupState.HYDRATING, reason="test1")
    await sm._transition_to(StartupState.RECONCILING, reason="test2")

    # Check history
    history = sm.get_transition_history()
    assert len(history) == 2
    assert history[0].from_state == StartupState.BOOTING
    assert history[0].to_state == StartupState.HYDRATING
    assert history[0].reason == "test1"


# ──────────────────────────────────────────────────────────────────────────────
# Integration Tests
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_hydration_and_state_machine_integration():
    """Test hydration engine + state machine together."""
    ss = NativeSharedState()
    ss.positions = {}

    # Create both components
    hydration_engine = NativePositionHydrationEngine(
        shared_state=ss,
        trade_journal=None,
        exchange_client=None,
    )

    state_machine = NativeStartupStateMachine()

    # Create a mock hydration callback
    async def hydrate_callback():
        # Simulate successful hydration
        hydrated_pos = HydratedPosition(
            symbol="AVAXUSDT",
            qty=0.01,
            avg_entry_price=100.0,
            current_price=101.0,
            unrealized_pnl=0.01,
            realized_pnl=0.0,
        )
        state = HydrationState(
            success=True,
            message="test",
            positions={"AVAXUSDT": hydrated_pos},
        )
        await hydration_engine.apply_to_shared_state(state)
        return True

    state_machine.set_callback(StartupState.HYDRATING, hydrate_callback)

    # Run startup
    success = await state_machine.run_startup()

    # Verify integration
    assert success
    assert state_machine.is_ready()
    assert "AVAXUSDT" in ss.positions


@pytest.mark.asyncio
async def test_buy_gating_with_startup_state():
    """Test that BUY decisions are blocked during startup."""
    ss = NativeSharedState()
    state_machine = NativeStartupStateMachine()

    # Simulate phase_decide logic
    async def simulate_phase_decide(state_machine):
        """Simulates the orchestrator's _phase_decide gating logic."""
        signals = {"AVAXUSDT": {"direction": "BUY"}}

        # This is the gate from orchestrator.py
        if state_machine is not None and not state_machine.can_buy():
            # Filter to SELL only
            signals = {
                sym: sig for sym, sig in signals.items() if str(sig.get("direction")) == "SELL"
            }

        return signals

    # During BOOTING (can't buy) → signals filtered
    state_machine._state = StartupState.BOOTING
    filtered_signals = await simulate_phase_decide(state_machine)
    assert len(filtered_signals) == 0  # BUY signal removed

    # During READY (can buy) → signals pass through
    state_machine._state = StartupState.READY
    filtered_signals = await simulate_phase_decide(state_machine)
    assert len(filtered_signals) == 1  # BUY signal kept
    assert filtered_signals["AVAXUSDT"]["direction"] == "BUY"


# ──────────────────────────────────────────────────────────────────────────────
# Regression Tests
# ──────────────────────────────────────────────────────────────────────────────


def test_position_hydrated_position_pnl_calculation():
    """Test PnL calculations in HydratedPosition."""
    pos = HydratedPosition(
        symbol="AVAXUSDT",
        qty=0.01,
        avg_entry_price=100.0,
        current_price=110.0,
        unrealized_pnl=0.1,
        realized_pnl=0.05,
    )

    assert pos.total_pnl == pytest.approx(0.15)
    assert pos.position_value == pytest.approx(1.1)
    assert pos.is_profitable()
    assert not pos.is_losing()


def test_hydrated_position_dust_and_stale():
    """Test dust and stale position classification."""
    # Dust position
    dust_pos = HydratedPosition(
        symbol="DUSTUSDT",
        qty=0.00001,
        avg_entry_price=1.0,
        current_price=1.0,
    )
    assert dust_pos.is_dust(dust_threshold_usdt=1.0)

    # Stale position
    stale_pos = HydratedPosition(
        symbol="STALEUDT",
        qty=0.01,
        avg_entry_price=100.0,
        current_price=100.0,
        last_update=0.0,  # Very old
    )
    assert stale_pos.is_stale(stale_age_sec=1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
