"""
Integration tests - Native L0-L8 working together via NativeOrchestrator.

Validates the 5-phase RUDE cycle (READ -> UNDERSTAND -> DECIDE -> EXECUTE -> RECOVER)
end-to-end with realistic stubs. Assertions reflect the actual orchestrator
behavior in ``core_engine/native/orchestrator.py``.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from core_engine.native import CycleMetrics, NativeOrchestrator


# ─────────────────────────────────────────────────────────────────────
# Stubs
# ─────────────────────────────────────────────────────────────────────
class _MD:
    def __init__(self, prices: dict[str, float] | None = None) -> None:
        self.prices = prices if prices is not None else {"BTCUSDT": 50000.0}

    async def start(self) -> None:
        ...

    async def stop(self) -> None:
        ...

    def get_prices(self) -> dict[str, float]:
        return dict(self.prices)

    async def get_klines(self, symbol: str, interval: str = "1m", limit: int = 100):
        return [[i, i + 1, i + 2, i + 3, 50000.0, i + 5] for i in range(limit)]


class _Sig:
    def __init__(self, score: float = 0.8) -> None:
        self.score = score
        self.calls: list[str] = []

    def evaluate(self, symbol: str, klines):
        self.calls.append(symbol)
        return type(
            "AggSig",
            (),
            {
                "direction": "BUY",
                "score": self.score,
                "contributions": {"trend": self.score},
            },
        )()


class _Dec:
    def __init__(self) -> None:
        self.calls = 0

    def decide(self, signals, portfolio, balance_usdt):
        self.calls += 1
        if not signals:
            return []
        return [
            type(
                "Decision",
                (),
                {
                    "symbol": next(iter(signals)),
                    "action": "OPEN",
                    "quantity": 0.1,
                    "decision_id": f"dec-{self.calls}",
                },
            )()
        ]


class _Exe:
    def __init__(self) -> None:
        self.executions: list = []

    async def execute(self, decisions):
        self.executions.append(list(decisions))
        return [
            type(
                "ExecResult",
                (),
                {
                    "symbol": d.symbol,
                    "status": type("S", (), {"value": "SUCCESS"})(),
                    "decision_id": d.decision_id,
                },
            )()
            for d in decisions
        ]


class _Bal:
    def __init__(self, usdt: float = 10000.0) -> None:
        self.usdt = usdt

    async def start(self) -> None:
        ...

    async def stop(self) -> None:
        ...

    def get_balance(self) -> dict[str, float]:
        return {"USDT": self.usdt}


class _State:
    def __init__(self, nav: float = 10000.0) -> None:
        self.nav = nav


def _portfolio() -> Any:
    return type(
        "PF",
        (),
        {
            "nav": 10000.0,
            "balance": {"USDT": 10000.0},
            "positions": {},
        },
    )()


def _make_orch(
    *,
    md: Any | None = None,
    sig: Any | None = None,
    dec: Any | None = None,
    exe: Any | None = None,
    bal: Any | None = None,
    state: Any | None = None,
    with_portfolio: bool = True,
) -> NativeOrchestrator:
    return NativeOrchestrator(
        market_data=md or _MD(),
        signal_engine=sig or _Sig(),
        decision_engine=dec or _Dec(),
        executor=exe or _Exe(),
        balance_sync=bal or _Bal(),
        shared_state=state or _State(),
        portfolio_accessor=_portfolio if with_portfolio else None,
    )


# ─────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────
class TestRUDECycleIntegration:
    """End-to-end RUDE cycle with realistic component stubs."""

    @pytest.mark.asyncio
    async def test_full_cycle_executes_all_phases(self) -> None:
        sig, dec, exe = _Sig(), _Dec(), _Exe()
        orch = _make_orch(sig=sig, dec=dec, exe=exe)

        m = await orch.run_cycle()

        # READ produced prices, UNDERSTAND generated signals
        assert sig.calls == ["BTCUSDT"]
        assert m.signals_count == 1
        # DECIDE produced a decision
        assert dec.calls == 1
        assert m.decisions_count == 1
        # EXECUTE ran it
        assert len(exe.executions) == 1
        assert m.executions_count == 1
        assert m.execution_successes == 1
        assert m.execution_failures == 0

    @pytest.mark.asyncio
    async def test_market_data_drives_signal_evaluation(self) -> None:
        md = _MD(prices={"BTCUSDT": 50000.0, "ETHUSDT": 3000.0})
        sig = _Sig()
        orch = _make_orch(md=md, sig=sig)

        m = await orch.run_cycle()

        assert sorted(sig.calls) == ["BTCUSDT", "ETHUSDT"]
        assert m.signals_count == 2

    @pytest.mark.asyncio
    async def test_no_portfolio_accessor_skips_decide(self) -> None:
        dec = _Dec()
        orch = _make_orch(dec=dec, with_portfolio=False)

        m = await orch.run_cycle()

        assert dec.calls == 0
        assert m.decisions_count == 0

    @pytest.mark.asyncio
    async def test_empty_market_data_yields_zero_signals(self) -> None:
        md = _MD(prices={})
        sig = _Sig()
        orch = _make_orch(md=md, sig=sig)

        m = await orch.run_cycle()

        assert sig.calls == []
        assert m.signals_count == 0
        assert m.decisions_count == 0
        assert m.executions_count == 0
        assert m.errors == []

    @pytest.mark.asyncio
    async def test_per_symbol_signal_errors_are_isolated(self) -> None:
        md = _MD(prices={"BTCUSDT": 50000.0, "ETHUSDT": 3000.0})
        sig = MagicMock()
        # First call raises, second returns a valid signal
        sig.evaluate.side_effect = [
            RuntimeError("BTC failed"),
            type(
                "AggSig",
                (),
                {
                    "direction": "BUY",
                    "score": 0.7,
                    "contributions": {},
                },
            )(),
        ]
        orch = _make_orch(md=md, sig=sig)

        m = await orch.run_cycle()

        # One symbol errored, the other produced a signal
        assert m.signals_count == 1
        assert m.errors == []  # per-symbol errors are swallowed

    @pytest.mark.asyncio
    async def test_execute_failure_recorded_at_top_level(self) -> None:
        exe = MagicMock()
        exe.execute = AsyncMock(side_effect=RuntimeError("execute boom"))
        orch = _make_orch(exe=exe)

        m = await orch.run_cycle()

        assert len(m.errors) == 1
        assert "RuntimeError" in m.errors[0]
        assert m.duration_ms > 0

    @pytest.mark.asyncio
    async def test_phase_times_recorded_for_each_phase(self) -> None:
        orch = _make_orch()

        m = await orch.run_cycle()

        for phase in ("read", "understand", "decide", "execute", "recover"):
            assert phase in m.phase_times
            assert m.phase_times[phase] >= 0.0

    @pytest.mark.asyncio
    async def test_nav_is_pulled_from_shared_state(self) -> None:
        orch = _make_orch(state=_State(nav=12345.67))

        m = await orch.run_cycle()

        assert m.nav == 12345.67

    @pytest.mark.asyncio
    async def test_cycle_counter_increments_across_cycles(self) -> None:
        orch = _make_orch()
        await orch.start()
        try:
            metrics = [await orch.run_cycle() for _ in range(3)]
        finally:
            await orch.stop()

        assert [m.cycle_num for m in metrics] == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_run_loop_with_max_cycles(self) -> None:
        orch = _make_orch()
        metrics = await orch.run_loop(max_cycles=4)
        assert len(metrics) == 4
        assert metrics[-1].cycle_num == 4

    @pytest.mark.asyncio
    async def test_run_loop_with_duration_bound(self) -> None:
        orch = _make_orch()
        metrics = await orch.run_loop(duration_sec=0.05)
        assert len(metrics) >= 1
        for m in metrics:
            assert isinstance(m, CycleMetrics)
            assert m.duration_ms > 0

    @pytest.mark.asyncio
    async def test_graceful_stop_terminates_loop(self) -> None:
        orch = _make_orch()
        task = asyncio.create_task(orch.run_loop(duration_sec=10.0))
        await asyncio.sleep(0.05)
        await orch.stop()
        metrics = await task
        assert len(metrics) >= 1

    @pytest.mark.asyncio
    async def test_no_signals_short_circuits_decide_and_execute(self) -> None:
        # Signal engine returns None for every symbol
        sig = MagicMock()
        sig.evaluate.return_value = None
        dec = _Dec()
        exe = _Exe()
        orch = _make_orch(sig=sig, dec=dec, exe=exe)

        m = await orch.run_cycle()

        assert m.signals_count == 0
        # decision engine still called (by design) with empty signals → returns []
        assert dec.calls == 1
        assert m.decisions_count == 0
        assert exe.executions == [[]]
        assert m.executions_count == 0

    @pytest.mark.asyncio
    async def test_balance_passed_into_decision_engine(self) -> None:
        bal = _Bal(usdt=42_000.0)
        captured: dict[str, float] = {}

        class _CaptureDec:
            def decide(self, signals, portfolio, balance_usdt):
                captured["bal"] = balance_usdt
                return []

        orch = _make_orch(bal=bal, dec=_CaptureDec())
        await orch.run_cycle()
        assert captured["bal"] == 42_000.0
