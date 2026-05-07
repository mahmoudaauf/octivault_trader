"""
Tests for Native L8 (Phase 8.2.9) - NativeOrchestrator.

End-to-end cycle testing with mocked L0-L7 layers.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from core_engine.native import (
    CycleMetrics,
    NativeOrchestrator,
)


# ─────────────────────────────────────────────────────────────────────
# Stubs
# ─────────────────────────────────────────────────────────────────────
class _StubMarketData:
    def __init__(self) -> None:
        self.prices = {"BTCUSDT": 50000.0, "ETHUSDT": 3000.0}
        self.kline_data = [[1, 2, 3, 4, 50000.0, 6] for _ in range(100)]
        self.calls = 0

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    def get_prices(self) -> dict[str, float]:
        return dict(self.prices)

    async def get_klines(
        self, symbol: str, interval: str = "1m", limit: int = 100
    ) -> list[list[Any]]:
        self.calls += 1
        return self.kline_data


class _StubSignalEngine:
    def evaluate(self, symbol: str, klines: list[list[Any]]) -> Any:
        if symbol == "BTCUSDT":
            return type(
                "Signal",
                (),
                {
                    "direction": "BUY",
                    "score": 0.8,
                    "contributions": [],
                },
            )()
        return None


class _StubDecisionEngine:
    def decide(self, signals: dict[str, Any], portfolio: Any, balance_usdt: float) -> list[Any]:
        return [
            type(
                "Decision",
                (),
                {
                    "symbol": "BTCUSDT",
                    "action": "OPEN",
                    "quantity": 0.1,
                    "decision_id": "dec-1",
                },
            )()
        ]


class _StubExecutor:
    def __init__(self) -> None:
        self.executions = []

    async def execute(self, decisions: list[Any]) -> list[Any]:
        self.executions.extend(decisions)
        return [
            type(
                "ExecutionResult",
                (),
                {
                    "symbol": "BTCUSDT",
                    "status": type("Status", (), {"value": "SUCCESS"})(),
                    "decision_id": "dec-1",
                },
            )()
        ]


class _StubBalanceSync:
    def __init__(self) -> None:
        self.balance_data = {"USDT": 10000.0, "BTC": 0.1}

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    def get_balance(self) -> dict[str, float]:
        return dict(self.balance_data)


class _StubSharedState:
    def __init__(self) -> None:
        self.nav = 10000.0
        self.metrics = {"peak_nav": 10000.0, "trades_in_window": 0}
        self.exchange_throttled = False
        self.exchange_throttle_until_ts = 0.0
        self.trading_halted = False
        self.current_mode = ""
        self.current_mode_reason = ""


class _StubModeManager:
    def evaluate(self, *, nav: float, metrics: dict[str, Any], state: Any | None = None) -> Any:
        return type("Mode", (), {"name": "NORMAL"})()


# ─────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────
class TestNativeOrchestrator:
    @pytest.mark.asyncio
    async def test_run_single_cycle(self) -> None:
        md = _StubMarketData()
        sig = _StubSignalEngine()
        dec = _StubDecisionEngine()
        exe = _StubExecutor()
        bal = _StubBalanceSync()
        state = _StubSharedState()

        def portfolio_accessor() -> Any:
            return type(
                "Portfolio",
                (),
                {
                    "nav": 10000.0,
                    "nav_peak": 10000.0,
                    "balance": {"USDT": 10000.0},
                    "positions": {},
                    "open_orders": {},
                },
            )()

        orch = NativeOrchestrator(
            market_data=md,
            signal_engine=sig,
            decision_engine=dec,
            executor=exe,
            balance_sync=bal,
            shared_state=state,
            mode_manager=_StubModeManager(),
            portfolio_accessor=portfolio_accessor,
        )
        await orch.start()
        m = await orch.run_cycle()
        await orch.stop()

        assert isinstance(m, CycleMetrics)
        assert m.cycle_num == 1
        assert m.duration_ms > 0
        assert m.nav == 10000.0
        assert m.signals_count >= 0
        assert m.decisions_count >= 0
        assert m.executions_count >= 0
        assert state.current_mode == "NORMAL"

    @pytest.mark.asyncio
    async def test_cycle_metrics_include_phase_times(self) -> None:
        md = _StubMarketData()
        sig = _StubSignalEngine()
        dec = _StubDecisionEngine()
        exe = _StubExecutor()
        bal = _StubBalanceSync()
        state = _StubSharedState()

        orch = NativeOrchestrator(
            market_data=md,
            signal_engine=sig,
            decision_engine=dec,
            executor=exe,
            balance_sync=bal,
            shared_state=state,
        )
        await orch.start()
        m = await orch.run_cycle()
        await orch.stop()

        assert "read" in m.phase_times
        assert "understand" in m.phase_times
        assert "decide" in m.phase_times
        assert "execute" in m.phase_times
        assert "recover" in m.phase_times
        assert all(t >= 0 for t in m.phase_times.values())

    @pytest.mark.asyncio
    async def test_run_loop_with_duration(self) -> None:
        md = _StubMarketData()
        sig = _StubSignalEngine()
        dec = _StubDecisionEngine()
        exe = _StubExecutor()
        bal = _StubBalanceSync()
        state = _StubSharedState()

        orch = NativeOrchestrator(
            market_data=md,
            signal_engine=sig,
            decision_engine=dec,
            executor=exe,
            balance_sync=bal,
            shared_state=state,
        )
        metrics = await orch.run_loop(duration_sec=0.1)
        assert len(metrics) >= 1

    @pytest.mark.asyncio
    async def test_start_does_not_wait_full_timeout_when_throttled_with_prices(self) -> None:
        md = _StubMarketData()
        sig = _StubSignalEngine()
        dec = _StubDecisionEngine()
        exe = _StubExecutor()
        bal = _StubBalanceSync()
        bal.balance_data = {}
        state = _StubSharedState()
        state.exchange_throttled = True
        state.exchange_throttle_until_ts = time.time() + 60.0

        orch = NativeOrchestrator(
            market_data=md,
            signal_engine=sig,
            decision_engine=dec,
            executor=exe,
            balance_sync=bal,
            shared_state=state,
        )

        start = asyncio.get_event_loop().time()
        await orch.start()
        elapsed = asyncio.get_event_loop().time() - start
        await orch.stop()
        assert elapsed < 5.0

    @pytest.mark.asyncio
    async def test_run_loop_with_max_cycles(self) -> None:
        md = _StubMarketData()
        sig = _StubSignalEngine()
        dec = _StubDecisionEngine()
        exe = _StubExecutor()
        bal = _StubBalanceSync()
        state = _StubSharedState()

        orch = NativeOrchestrator(
            market_data=md,
            signal_engine=sig,
            decision_engine=dec,
            executor=exe,
            balance_sync=bal,
            shared_state=state,
        )
        metrics = await orch.run_loop(max_cycles=3)
        assert len(metrics) == 3
        assert metrics[0].cycle_num == 1
        assert metrics[1].cycle_num == 2
        assert metrics[2].cycle_num == 3

    @pytest.mark.asyncio
    async def test_cycle_tracking(self) -> None:
        md = _StubMarketData()
        sig = _StubSignalEngine()
        dec = _StubDecisionEngine()
        exe = _StubExecutor()
        bal = _StubBalanceSync()
        state = _StubSharedState()

        orch = NativeOrchestrator(
            market_data=md,
            signal_engine=sig,
            decision_engine=dec,
            executor=exe,
            balance_sync=bal,
            shared_state=state,
        )
        await orch.start()
        m1 = await orch.run_cycle()
        m2 = await orch.run_cycle()
        m3 = await orch.run_cycle()
        await orch.stop()

        assert m1.cycle_num == 1
        assert m2.cycle_num == 2
        assert m3.cycle_num == 3

    @pytest.mark.asyncio
    async def test_execution_results_counted(self) -> None:
        md = _StubMarketData()
        sig = _StubSignalEngine()
        dec = _StubDecisionEngine()

        exe = _StubExecutor()
        bal = _StubBalanceSync()
        state = _StubSharedState()

        orch = NativeOrchestrator(
            market_data=md,
            signal_engine=sig,
            decision_engine=dec,
            executor=exe,
            balance_sync=bal,
            shared_state=state,
        )
        await orch.start()
        m = await orch.run_cycle()
        await orch.stop()

        assert m.executions_count >= 0
        assert m.execution_successes >= 0
        assert m.execution_failures >= 0

    @pytest.mark.asyncio
    async def test_graceful_stop_from_loop(self) -> None:
        md = _StubMarketData()
        sig = _StubSignalEngine()
        dec = _StubDecisionEngine()
        exe = _StubExecutor()
        bal = _StubBalanceSync()
        state = _StubSharedState()

        orch = NativeOrchestrator(
            market_data=md,
            signal_engine=sig,
            decision_engine=dec,
            executor=exe,
            balance_sync=bal,
            shared_state=state,
        )

        async def run_and_stop() -> None:
            await asyncio.sleep(0.05)
            await orch.stop()

        task = asyncio.create_task(orch.run_loop(duration_sec=10.0))
        await asyncio.sleep(0.01)
        await run_and_stop()
        metrics = await task
        assert len(metrics) >= 1  # at least one cycle completed before stop

    @pytest.mark.asyncio
    async def test_per_symbol_signal_errors_are_swallowed(self) -> None:
        """Signal-eval errors are caught per-symbol and logged; cycle continues."""
        md = _StubMarketData()
        sig = MagicMock()
        sig.evaluate.side_effect = RuntimeError("signal error")
        dec = _StubDecisionEngine()
        exe = _StubExecutor()
        bal = _StubBalanceSync()
        state = _StubSharedState()

        orch = NativeOrchestrator(
            market_data=md,
            signal_engine=sig,
            decision_engine=dec,
            executor=exe,
            balance_sync=bal,
            shared_state=state,
        )
        await orch.start()
        m = await orch.run_cycle()
        await orch.stop()

        # Per-symbol exceptions are swallowed -> 0 signals, no top-level errors.
        assert m.signals_count == 0
        assert m.errors == []

    @pytest.mark.asyncio
    async def test_top_level_error_recorded_in_metrics(self) -> None:
        """Errors outside the per-symbol try are captured in CycleMetrics.errors."""
        md = _StubMarketData()
        sig = _StubSignalEngine()
        dec = _StubDecisionEngine()
        exe = MagicMock()
        exe.execute = AsyncMock(side_effect=RuntimeError("execute boom"))
        bal = _StubBalanceSync()
        state = _StubSharedState()

        def portfolio_accessor() -> Any:
            return type(
                "Portfolio",
                (),
                {
                    "nav": 10000.0,
                    "balance": {"USDT": 10000.0},
                    "positions": {},
                },
            )()

        orch = NativeOrchestrator(
            market_data=md,
            signal_engine=sig,
            decision_engine=dec,
            executor=exe,
            balance_sync=bal,
            shared_state=state,
            portfolio_accessor=portfolio_accessor,
        )
        await orch.start()
        m = await orch.run_cycle()
        await orch.stop()

        assert len(m.errors) > 0
        assert "RuntimeError" in m.errors[0]

    @pytest.mark.asyncio
    async def test_nav_captured_in_metrics(self) -> None:
        md = _StubMarketData()
        sig = _StubSignalEngine()
        dec = _StubDecisionEngine()
        exe = _StubExecutor()
        bal = _StubBalanceSync()
        state = _StubSharedState()
        state.nav = 12345.67

        orch = NativeOrchestrator(
            market_data=md,
            signal_engine=sig,
            decision_engine=dec,
            executor=exe,
            balance_sync=bal,
            shared_state=state,
        )
        await orch.start()
        m = await orch.run_cycle()
        await orch.stop()

        assert m.nav == 12345.67


class TestRealSharedStateNavRegression:
    """
    Regression: orchestrator must read NAV from the *real*
    ``NativeSharedState.nav_usdt`` field, not a stub-only ``.nav``.

    Discovered by ``scripts/native_smoke.py`` (Phase 8.2.8 step 5):
    every cycle was raising ``AttributeError: 'NativeSharedState' object
    has no attribute 'nav'`` because every L8 stub carries ``.nav`` but
    the production class exposes ``nav_usdt``.
    """

    @pytest.mark.asyncio
    async def test_run_cycle_reads_nav_usdt_from_real_shared_state(self) -> None:
        from core_engine.native.shared_state import NativeSharedState

        state = NativeSharedState()
        state.nav_usdt = 9_999.99

        md = _StubMarketData()
        sig = _StubSignalEngine()
        dec = _StubDecisionEngine()
        exe = _StubExecutor()
        bal = _StubBalanceSync()

        orch = NativeOrchestrator(
            market_data=md,
            signal_engine=sig,
            decision_engine=dec,
            executor=exe,
            balance_sync=bal,
            shared_state=state,
        )
        await orch.start()
        m = await orch.run_cycle()
        await orch.stop()

        assert m.errors == [], f"unexpected cycle errors: {m.errors}"
        assert m.nav == 9_999.99


# ─────────────────────────────────────────────────────────────────────
# L7 watchdog wiring (Phase 8.3.12 follow-up)
# ─────────────────────────────────────────────────────────────────────
class _RecordingWatchdog:
    """Captures every record_heartbeat call for assertion."""

    def __init__(self) -> None:
        self.calls: list[bool] = []

    def record_heartbeat(self, *, ok: bool = True) -> None:
        self.calls.append(ok)


def _orch_with_watchdog(wd: Any | None) -> NativeOrchestrator:
    return NativeOrchestrator(
        market_data=_StubMarketData(),
        signal_engine=_StubSignalEngine(),
        decision_engine=_StubDecisionEngine(),
        executor=_StubExecutor(),
        balance_sync=_StubBalanceSync(),
        shared_state=_StubSharedState(),
        watchdog=wd,
    )


class TestOrchestratorWatchdogWiring:
    @pytest.mark.asyncio
    async def test_run_cycle_records_heartbeat_ok_when_no_errors(self) -> None:
        wd = _RecordingWatchdog()
        orch = _orch_with_watchdog(wd)
        await orch.start()
        m = await orch.run_cycle()
        await orch.stop()

        assert m.errors == []
        assert wd.calls == [True]

    @pytest.mark.asyncio
    async def test_run_loop_records_one_heartbeat_per_cycle(self) -> None:
        wd = _RecordingWatchdog()
        orch = _orch_with_watchdog(wd)
        await orch.run_loop(max_cycles=3)
        assert wd.calls == [True, True, True]

    @pytest.mark.asyncio
    async def test_run_cycle_records_heartbeat_failed_when_errors_present(self) -> None:
        """A cycle whose inner phases raise still emits a heartbeat with ok=False."""
        wd = _RecordingWatchdog()

        class _BoomMD(_StubMarketData):
            def get_prices(self) -> dict[str, float]:
                raise RuntimeError("boom")

        orch = NativeOrchestrator(
            market_data=_BoomMD(),
            signal_engine=_StubSignalEngine(),
            decision_engine=_StubDecisionEngine(),
            executor=_StubExecutor(),
            balance_sync=_StubBalanceSync(),
            shared_state=_StubSharedState(),
            watchdog=wd,
        )
        await orch.start()
        m = await orch.run_cycle()
        await orch.stop()

        assert m.errors  # cycle did fail
        assert wd.calls == [False]

    @pytest.mark.asyncio
    async def test_no_watchdog_is_zero_overhead(self) -> None:
        """Orchestrator without a watchdog runs cleanly (no AttributeError)."""
        orch = _orch_with_watchdog(None)
        await orch.start()
        m = await orch.run_cycle()
        await orch.stop()
        assert m.cycle_num == 1
        assert m.errors == []

    @pytest.mark.asyncio
    async def test_watchdog_exception_does_not_break_cycle(self) -> None:
        """A misbehaving watchdog never poisons a cycle."""

        class _BoomWatchdog:
            def record_heartbeat(self, *, ok: bool = True) -> None:
                raise RuntimeError("watchdog exploded")

        orch = _orch_with_watchdog(_BoomWatchdog())
        await orch.start()
        m = await orch.run_cycle()  # must NOT raise
        await orch.stop()
        assert m.cycle_num == 1
        # Cycle itself succeeded; watchdog crash is logged, not propagated.
        assert m.errors == []
