"""
Tests for Native L5 (Phase 8.2.6) — NativeExecutor.

Mocks NativeOrderExecution. Tests dedup, error classification, per-symbol sequencing.
"""

from __future__ import annotations

from typing import Any

import pytest

from core_engine.native import (
    ExecutionStatus,
    NativeBalanceValidator,
    NativeExecutor,
    NativeSharedState,
)
from core_engine.native.decisions import Action, Decision


# ─────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────
class _StubOrderExecution:
    """Minimal NativeOrderExecution stub."""

    def __init__(self) -> None:
        self.placed_calls: list[dict[str, Any]] = []
        self.sold_calls: list[dict[str, Any]] = []
        self.next_buy_result: dict[str, Any] = {"success": True, "orderId": 1, "quantity": 0.1}
        self.next_sell_result: dict[str, Any] = {"success": True, "orderId": 2, "quantity": 0.1}

    async def place_market_buy(self, symbol: str, quantity: float, **kwargs: Any) -> Any:
        self.placed_calls.append({"symbol": symbol, "quantity": quantity, **kwargs})
        return type(
            "OrderResult",
            (),
            {
                "success": self.next_buy_result["success"],
                "exchange_order_id": self.next_buy_result.get("orderId"),
                "quantity": self.next_buy_result.get("quantity", quantity),
                "raw": self.next_buy_result,
                "error": self.next_buy_result.get("error"),
            },
        )()

    async def place_market_sell(self, symbol: str, quantity: float, **kwargs: Any) -> Any:
        self.sold_calls.append({"symbol": symbol, "quantity": quantity, **kwargs})
        return type(
            "OrderResult",
            (),
            {
                "success": self.next_sell_result["success"],
                "exchange_order_id": self.next_sell_result.get("orderId"),
                "quantity": self.next_sell_result.get("quantity", quantity),
                "raw": self.next_sell_result,
                "error": self.next_sell_result.get("error"),
            },
        )()


# ─────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────
class TestNativeExecutor:
    @pytest.mark.asyncio
    async def test_execute_open_decision(self) -> None:
        stub = _StubOrderExecution()
        executor = NativeExecutor(stub)  # type: ignore[arg-type]
        dec = Decision("BTCUSDT", Action.OPEN, 0.1, "test", 0.7)
        results = await executor.execute([dec])
        assert len(results) == 1
        assert results[0].status == ExecutionStatus.SUCCESS
        assert results[0].symbol == "BTCUSDT"
        assert len(stub.placed_calls) == 1
        assert stub.placed_calls[0]["symbol"] == "BTCUSDT"

    @pytest.mark.asyncio
    async def test_execute_close_decision(self) -> None:
        stub = _StubOrderExecution()
        executor = NativeExecutor(stub)  # type: ignore[arg-type]
        dec = Decision("BTCUSDT", Action.CLOSE, 0.1, "test", 0.7)
        results = await executor.execute([dec])
        assert len(results) == 1
        assert results[0].status == ExecutionStatus.SUCCESS
        assert results[0].symbol == "BTCUSDT"
        assert len(stub.sold_calls) == 1

    @pytest.mark.asyncio
    async def test_dedup_prevents_reexecution(self) -> None:
        stub = _StubOrderExecution()
        executor = NativeExecutor(stub)  # type: ignore[arg-type]
        dec = Decision("BTCUSDT", Action.OPEN, 0.1, "test", 0.7, decision_id="id-1")
        # First execution
        results1 = await executor.execute([dec])
        assert len(results1) == 1
        assert len(stub.placed_calls) == 1
        # Second execution with same decision_id
        results2 = await executor.execute([dec])
        assert len(results2) == 0  # deduped; no new results
        assert len(stub.placed_calls) == 1  # still 1; no new call

    @pytest.mark.asyncio
    async def test_multiple_decisions_sequential(self) -> None:
        stub = _StubOrderExecution()
        executor = NativeExecutor(stub)  # type: ignore[arg-type]
        decs = [
            Decision("BTCUSDT", Action.OPEN, 0.1, "test", 0.7),
            Decision("ETHUSDT", Action.OPEN, 0.5, "test", 0.8),
        ]
        results = await executor.execute(decs)
        assert len(results) == 2
        assert results[0].symbol == "BTCUSDT"
        assert results[1].symbol == "ETHUSDT"
        assert len(stub.placed_calls) == 2

    @pytest.mark.asyncio
    async def test_exchange_order_id_captured(self) -> None:
        stub = _StubOrderExecution()
        stub.next_buy_result["orderId"] = 999
        executor = NativeExecutor(stub)  # type: ignore[arg-type]
        dec = Decision("BTCUSDT", Action.OPEN, 0.1, "test", 0.7)
        results = await executor.execute([dec])
        assert results[0].exchange_order_id == 999

    @pytest.mark.asyncio
    async def test_order_failure_retryable(self) -> None:
        stub = _StubOrderExecution()
        stub.next_buy_result = {
            "success": False,
            "error": "429 rate limited",
            "orderId": None,
        }
        executor = NativeExecutor(stub)  # type: ignore[arg-type]
        dec = Decision("BTCUSDT", Action.OPEN, 0.1, "test", 0.7)
        results = await executor.execute([dec])
        assert len(results) == 1
        assert results[0].status == ExecutionStatus.RETRYABLE
        assert "429" in (results[0].error or "")

    @pytest.mark.asyncio
    async def test_order_failure_terminal_insufficient_balance(self) -> None:
        stub = _StubOrderExecution()
        stub.next_buy_result = {
            "success": False,
            "error": "insufficient balance",
            "orderId": None,
        }
        executor = NativeExecutor(stub)  # type: ignore[arg-type]
        dec = Decision("BTCUSDT", Action.OPEN, 0.1, "test", 0.7)
        results = await executor.execute([dec])
        assert len(results) == 1
        assert results[0].status == ExecutionStatus.TERMINAL
        assert "insufficient" in (results[0].error or "").lower()

    @pytest.mark.asyncio
    async def test_decision_id_used_as_client_order_id(self) -> None:
        stub = _StubOrderExecution()
        executor = NativeExecutor(stub)  # type: ignore[arg-type]
        dec = Decision("BTCUSDT", Action.OPEN, 0.1, "test", 0.7, decision_id="my-id-42")
        await executor.execute([dec])
        # Verify client_order_id was passed through
        assert stub.placed_calls[0]["client_order_id"] == "my-id-42"

    @pytest.mark.asyncio
    async def test_error_classification(self) -> None:
        assert NativeExecutor._classify_error("429 too many requests") == ExecutionStatus.RETRYABLE
        assert (
            NativeExecutor._classify_error("503 service unavailable") == ExecutionStatus.RETRYABLE
        )
        assert NativeExecutor._classify_error("timeout") == ExecutionStatus.RETRYABLE
        assert NativeExecutor._classify_error("insufficient balance") == ExecutionStatus.TERMINAL
        assert NativeExecutor._classify_error("invalid quantity") == ExecutionStatus.TERMINAL
        assert NativeExecutor._classify_error("rejected order") == ExecutionStatus.TERMINAL
        assert NativeExecutor._classify_error("unknown error") == ExecutionStatus.RETRYABLE

    @pytest.mark.asyncio
    async def test_reset_dedup_state(self) -> None:
        stub = _StubOrderExecution()
        executor = NativeExecutor(stub)  # type: ignore[arg-type]
        dec = Decision("BTCUSDT", Action.OPEN, 0.1, "test", 0.7, decision_id="id-1")
        # Execute once
        await executor.execute([dec])
        assert len(stub.placed_calls) == 1
        # Reset dedup state
        executor.reset_dedup_state()
        # Execute same decision again — should work now
        results = await executor.execute([dec])
        assert len(results) == 1
        assert len(stub.placed_calls) == 2

    @pytest.mark.asyncio
    async def test_execution_result_has_metadata(self) -> None:
        stub = _StubOrderExecution()
        executor = NativeExecutor(stub)  # type: ignore[arg-type]
        dec = Decision("BTCUSDT", Action.OPEN, 0.1, "test", 0.7)
        results = await executor.execute([dec])
        res = results[0]
        assert res.decision_id == dec.decision_id
        assert res.symbol == "BTCUSDT"
        assert res.quantity_requested == 0.1
        assert res.ts > 0
        assert res.raw is not None

    @pytest.mark.asyncio
    async def test_mixed_buy_sell(self) -> None:
        stub = _StubOrderExecution()
        executor = NativeExecutor(stub)  # type: ignore[arg-type]
        decs = [
            Decision("BTCUSDT", Action.OPEN, 0.1, "buy", 0.8),
            Decision("ETHUSDT", Action.CLOSE, 0.5, "sell", 0.7),
        ]
        results = await executor.execute(decs)
        assert len(results) == 2
        assert results[0].status == ExecutionStatus.SUCCESS
        assert results[1].status == ExecutionStatus.SUCCESS
        assert len(stub.placed_calls) == 1
        assert len(stub.sold_calls) == 1

    @pytest.mark.asyncio
    async def test_hold_action_rejected(self) -> None:
        stub = _StubOrderExecution()
        executor = NativeExecutor(stub)  # type: ignore[arg-type]
        dec = Decision("BTCUSDT", Action.HOLD, 0.0, "test", 0.0)
        results = await executor.execute([dec])
        assert len(results) == 1
        assert results[0].status == ExecutionStatus.TERMINAL
        assert "unknown action" in (results[0].error or "").lower()

    @pytest.mark.asyncio
    async def test_balance_validator_blocks_over_allocation(self) -> None:
        stub = _StubOrderExecution()
        state = NativeSharedState()
        state.free_balance_usdt = 50.0
        state.balance = {"USDT": 50.0}
        executor = NativeExecutor(
            stub, shared_state=state, balance_validator=NativeBalanceValidator()
        )  # type: ignore[arg-type]
        dec = Decision("BTCUSDT", Action.OPEN, 100.0, "test", 0.7)
        results = await executor.execute([dec])
        assert len(results) == 1
        assert results[0].status == ExecutionStatus.TERMINAL
        assert "balance validation failed" in (results[0].error or "")
        assert len(stub.placed_calls) == 0

    @pytest.mark.asyncio
    async def test_successful_buy_commits_allocation_ledger(self) -> None:
        stub = _StubOrderExecution()
        state = NativeSharedState()
        state.free_balance_usdt = 500.0
        state.balance = {"USDT": 500.0}

        class _MD:
            def get_price(self, _symbol: str) -> float:
                return 100.0

        validator = NativeBalanceValidator()
        executor = NativeExecutor(
            stub, market_data=_MD(), shared_state=state, balance_validator=validator
        )  # type: ignore[arg-type]
        dec = Decision("BTCUSDT", Action.OPEN, 100.0, "test", 0.7)
        results = await executor.execute([dec])
        assert results[0].status == ExecutionStatus.SUCCESS
        assert validator.allocated_balance == 100.0
        assert validator.recent_entries(1)[0].status == "committed"
        assert state.reserved_quote_total("USDT") == 100.0
