"""
Regression tests for a real bug found 2026-07-14: when the native startup
state machine failed (timeout or hydration failure), NativeOrchestrator.start()
logged "Startup failed; trading will be blocked" and fell straight through --
nothing downstream actually blocked anything. session_anchor_nav stayed stuck
at 0 (or a stale persisted value), bypassing every NAV-protection
session-scoped guard, and OperationsEngineImpl.startup_system() returned True
regardless (only an exception from native_orch.start() makes it return False),
so main.py's Engines.initialize() (which discarded the return value anyway)
sailed straight into the trading loop on a broken foundation.

Fix: NativeOrchestrator.start() now raises when run_startup() fails, so
startup_system()'s except-block correctly returns False, and
Engines.initialize() now checks that return value and refuses to proceed.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from core_engine.implementations import OperationsEngineImpl
from core_engine.native.orchestrator import NativeOrchestrator


def _make_orchestrator(*, startup_succeeds: bool) -> NativeOrchestrator:
    market_data = MagicMock()
    market_data.start = AsyncMock()
    startup_state_machine = MagicMock()
    startup_state_machine.set_callback = MagicMock()
    startup_state_machine.run_startup = AsyncMock(return_value=startup_succeeds)

    orch = NativeOrchestrator(
        market_data=market_data,
        signal_engine=MagicMock(),
        decision_engine=MagicMock(),
        executor=MagicMock(),
        shared_state=MagicMock(),
        startup_state_machine=startup_state_machine,
    )
    # Avoid needing a real _wait_for_initial_data implementation.
    orch._wait_for_initial_data = AsyncMock()  # type: ignore[method-assign]
    return orch


@pytest.mark.asyncio
async def test_orchestrator_start_raises_when_startup_state_machine_fails():
    orch = _make_orchestrator(startup_succeeds=False)
    with pytest.raises(RuntimeError, match="startup state machine failed"):
        await orch.start()


@pytest.mark.asyncio
async def test_orchestrator_start_succeeds_when_startup_state_machine_succeeds():
    orch = _make_orchestrator(startup_succeeds=True)
    # Should not raise. (session_anchor assignment inside sleeps 15s+2s in the
    # real code -- patch it out by giving shared_state a MagicMock that makes
    # nav_usdt <= 0 so that branch is skipped quickly.)
    orch._shared_state.nav_usdt = 0.0
    orch._shared_state.balance = {}
    import asyncio as _asyncio
    real_sleep = _asyncio.sleep
    async def _fast_sleep(_):
        return None
    _asyncio.sleep = _fast_sleep  # type: ignore[assignment]
    try:
        await orch.start()
    finally:
        _asyncio.sleep = real_sleep  # type: ignore[assignment]


@pytest.mark.asyncio
async def test_startup_system_returns_false_when_native_orchestrator_raises():
    """OperationsEngineImpl.startup_system()'s except-block must catch the
    raised RuntimeError and correctly return False (this was already the
    mechanism -- confirming it works now that start() actually raises)."""
    failing_orch = MagicMock()
    failing_orch.start = AsyncMock(side_effect=RuntimeError("native startup state machine failed"))
    app_ctx = {"_native_orchestrator": failing_orch}

    result = await OperationsEngineImpl.startup_system(app_ctx)

    assert result is False


@pytest.mark.asyncio
async def test_engines_initialize_raises_when_startup_system_returns_false(monkeypatch):
    """main.py's Engines.initialize() must refuse to proceed (raise) rather
    than silently continue into market/situation/decision/execution init and
    the trading loop when startup_system() reports failure."""
    import main as main_module

    app_ctx: dict = {}
    engines = main_module.Engines(app_ctx)
    engines.operations.startup_system = AsyncMock(return_value=False)  # type: ignore[method-assign]
    engines.market.initialize = AsyncMock()  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="startup_system"):
        await engines.initialize()

    engines.market.initialize.assert_not_called()
