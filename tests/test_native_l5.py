"""
Tests for Native L5 (Phase 8.2.6) — NativeExecutor.

Mocks NativeOrderExecution. Tests dedup, error classification, per-symbol sequencing.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from core_engine.native import (
    ExecutionStatus,
    NativeBalanceValidator,
    NativeExecutor,
    NativeSharedState,
)
from core_engine.native.decisions import Action, Decision
from core_engine.native.executor import commission_quote_from_fills, compute_net_trade_pnl


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
                "executed_qty": self.next_buy_result.get("executed_qty"),
                "avg_price": self.next_buy_result.get("avg_price"),
                "price": self.next_buy_result.get("price"),
                "order_type": self.next_buy_result.get("order_type", "MARKET"),
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
                "executed_qty": self.next_sell_result.get("executed_qty"),
                "avg_price": self.next_sell_result.get("avg_price"),
                "price": self.next_sell_result.get("price"),
                "order_type": self.next_sell_result.get("order_type", "MARKET"),
                "raw": self.next_sell_result,
                "error": self.next_sell_result.get("error"),
            },
        )()


class _MD:
    """Minimal market_data stub — executor.py needs a price to convert a
    decision's USD quantity into a base-asset order size (see executor.py
    "BUY rejected: price unavailable ... cannot size order")."""

    def get_price(self, _symbol: str) -> float:
        return 100.0


# ─────────────────────────────────────────────────────────────────────
# compute_net_trade_pnl / commission_quote_from_fills (Priority 2 item #4)
# ─────────────────────────────────────────────────────────────────────
class TestCanonicalNetPnl:
    def test_net_pnl_subtracts_both_fees(self) -> None:
        net = compute_net_trade_pnl(
            entry_price=100.0, exit_price=110.0, qty=1.0,
            entry_commission_quote=0.1, exit_commission_quote=0.11,
        )
        # gross = 10.0, minus 0.21 in fees
        assert net == pytest.approx(9.79)

    def test_net_pnl_negative_commission_ignored(self) -> None:
        # Defensive: a negative fee value must never inflate profit.
        net = compute_net_trade_pnl(
            entry_price=100.0, exit_price=110.0, qty=1.0,
            entry_commission_quote=-5.0, exit_commission_quote=0.0,
        )
        assert net == pytest.approx(10.0)

    def test_commission_quote_from_fills_quote_asset_passthrough(self) -> None:
        fills = [{"price": "100.0", "qty": "1.0", "commission": "0.1", "commissionAsset": "USDT"}]
        assert commission_quote_from_fills(fills, symbol="BTCUSDT") == pytest.approx(0.1)

    def test_commission_quote_from_fills_base_asset_converted(self) -> None:
        fills = [{"price": "100.0", "qty": "1.0", "commission": "0.001", "commissionAsset": "BTC"}]
        # 0.001 BTC * 100.0 price = 0.1 USDT
        assert commission_quote_from_fills(fills, symbol="BTCUSDT") == pytest.approx(0.1)

    def test_commission_quote_from_fills_unconvertible_asset_contributes_zero(self) -> None:
        fills = [{"price": "100.0", "qty": "1.0", "commission": "1.0", "commissionAsset": "BNB"}]
        assert commission_quote_from_fills(fills, symbol="BTCUSDT", price_cache={}) == 0.0

    def test_commission_quote_from_fills_bnb_converted_via_price_cache(self) -> None:
        fills = [{"price": "100.0", "qty": "1.0", "commission": "1.0", "commissionAsset": "BNB"}]
        out = commission_quote_from_fills(
            fills, symbol="BTCUSDT", price_cache={"BNBUSDT": 600.0}
        )
        assert out == pytest.approx(600.0)

    def test_commission_quote_from_fills_multiple_fills_summed(self) -> None:
        fills = [
            {"price": "100.0", "qty": "0.5", "commission": "0.05", "commissionAsset": "USDT"},
            {"price": "101.0", "qty": "0.5", "commission": "0.05", "commissionAsset": "USDT"},
        ]
        assert commission_quote_from_fills(fills, symbol="BTCUSDT") == pytest.approx(0.1)

    def test_commission_quote_from_fills_empty_or_none(self) -> None:
        assert commission_quote_from_fills(None, symbol="BTCUSDT") == 0.0
        assert commission_quote_from_fills([], symbol="BTCUSDT") == 0.0


# ─────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────
class TestNativeExecutor:
    @pytest.fixture(autouse=True)
    def _disable_maker_first_buy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """executor.py's maker-first BUY path (MAKER_ENTRY_ENABLED, default on)
        places a LIMIT order via place_limit_buy()/refresh_status()/cancel() and
        waits MAKER_GRACE_S (2s default) before falling back to a market order.
        These tests exercise dedup/error-classification/basic execution, not the
        maker-entry feature itself, and _StubOrderExecution doesn't implement the
        limit-order interface — force the "no usable maker window" branch so
        execute() takes the plain market-buy path these tests were written
        against (see executor.py::_maker_first_buy, `grace_s <= 0` branch).
        """
        monkeypatch.setenv("MAKER_GRACE_S", "0")

    def test_execution_quality_tracks_adverse_cost_and_maker_rate(self) -> None:
        state = NativeSharedState()
        executor = NativeExecutor(_StubOrderExecution(), shared_state=state)  # type: ignore[arg-type]

        buy = executor._record_execution_quality(
            side="BUY", reference_price=100.0, fill_price=99.99, is_maker=True
        )
        sell = executor._record_execution_quality(
            side="SELL", reference_price=100.0, fill_price=99.90, is_maker=False
        )

        assert buy["adverse_slippage_bps"] == 0.0
        assert buy["price_improvement_bps"] == pytest.approx(1.0)
        assert sell["adverse_slippage_bps"] == pytest.approx(10.01001)
        assert state.metrics["execution_quality_samples"] == 2
        assert state.metrics["avg_slippage_bps"] == pytest.approx(5.005005)
        assert state.metrics["maker_fill_rate"] == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_sell_uses_exchange_fill_price_for_slippage_and_pnl(self) -> None:
        stub = _StubOrderExecution()
        stub.next_sell_result.update(
            {"quantity": 1.0, "executed_qty": 1.0, "avg_price": 100.5}
        )
        state = NativeSharedState()
        state.update_position("BTCUSDT", 1.0, 100.0, 101.0)

        class _MD:
            def get_price(self, _symbol: str) -> float:
                return 101.0

        executor = NativeExecutor(stub, market_data=_MD(), shared_state=state)  # type: ignore[arg-type]
        result = (await executor.execute([
            Decision("BTCUSDT", Action.CLOSE, 1.0, "test", 0.7)
        ]))[0]

        quality = result.raw["_execution_quality"]
        assert quality["fill_price"] == 100.5
        assert quality["reference_price"] == 101.0
        assert quality["adverse_slippage_bps"] == pytest.approx(
            (101.0 / 100.5 - 1.0) * 10_000.0
        )
        assert state.get_position("BTCUSDT") is None

    @pytest.mark.asyncio
    async def test_execute_open_decision(self) -> None:
        stub = _StubOrderExecution()
        executor = NativeExecutor(stub, market_data=_MD())  # type: ignore[arg-type]
        dec = Decision("BTCUSDT", Action.OPEN, 20.0, "test", 0.7)
        results = await executor.execute([dec])
        assert len(results) == 1
        assert results[0].status == ExecutionStatus.SUCCESS
        assert results[0].symbol == "BTCUSDT"
        assert len(stub.placed_calls) == 1
        assert stub.placed_calls[0]["symbol"] == "BTCUSDT"

    @pytest.mark.asyncio
    async def test_execute_close_decision(self) -> None:
        stub = _StubOrderExecution()
        executor = NativeExecutor(stub, market_data=_MD())  # type: ignore[arg-type]
        dec = Decision("BTCUSDT", Action.CLOSE, 0.1, "test", 0.7)
        results = await executor.execute([dec])
        assert len(results) == 1
        assert results[0].status == ExecutionStatus.SUCCESS
        assert results[0].symbol == "BTCUSDT"
        assert len(stub.sold_calls) == 1

    @pytest.mark.asyncio
    async def test_dedup_prevents_reexecution(self) -> None:
        stub = _StubOrderExecution()
        executor = NativeExecutor(stub, market_data=_MD())  # type: ignore[arg-type]
        dec = Decision("BTCUSDT", Action.OPEN, 20.0, "test", 0.7, decision_id="id-1")
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
        executor = NativeExecutor(stub, market_data=_MD())  # type: ignore[arg-type]
        decs = [
            Decision("BTCUSDT", Action.OPEN, 20.0, "test", 0.7),
            Decision("ETHUSDT", Action.OPEN, 20.0, "test", 0.8),
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
        executor = NativeExecutor(stub, market_data=_MD())  # type: ignore[arg-type]
        dec = Decision("BTCUSDT", Action.OPEN, 20.0, "test", 0.7)
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
        executor = NativeExecutor(stub, market_data=_MD())  # type: ignore[arg-type]
        dec = Decision("BTCUSDT", Action.OPEN, 20.0, "test", 0.7)
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
        executor = NativeExecutor(stub, market_data=_MD())  # type: ignore[arg-type]
        dec = Decision("BTCUSDT", Action.OPEN, 20.0, "test", 0.7)
        results = await executor.execute([dec])
        assert len(results) == 1
        assert results[0].status == ExecutionStatus.TERMINAL
        assert "insufficient" in (results[0].error or "").lower()

    @pytest.mark.asyncio
    async def test_decision_id_used_as_client_order_id(self) -> None:
        stub = _StubOrderExecution()
        executor = NativeExecutor(stub, market_data=_MD())  # type: ignore[arg-type]
        dec = Decision("BTCUSDT", Action.OPEN, 20.0, "test", 0.7, decision_id="my-id-42")
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
        executor = NativeExecutor(stub, market_data=_MD())  # type: ignore[arg-type]
        dec = Decision("BTCUSDT", Action.OPEN, 20.0, "test", 0.7, decision_id="id-1")
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
        executor = NativeExecutor(stub, market_data=_MD())  # type: ignore[arg-type]
        decs = [
            Decision("BTCUSDT", Action.OPEN, 20.0, "buy", 0.8),
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

    @pytest.mark.asyncio
    async def test_close_position_records_net_of_fee_realized_pnl(self) -> None:
        """BUY then CLOSE must persist metrics['realized_pnl'] net of both
        entry and exit commission, not the gross-only figure that used to
        only appear in a log line (Priority 2 item #4)."""
        stub = _StubOrderExecution()
        stub.next_buy_result.update({
            "quantity": 1.0, "executed_qty": 1.0, "avg_price": 100.0, "price": 100.0,
            "fills": [{"price": "100.0", "qty": "1.0", "commission": "0.1", "commissionAsset": "USDT"}],
        })
        stub.next_sell_result.update({
            "quantity": 1.0, "executed_qty": 1.0, "avg_price": 110.0, "price": 110.0,
            "fills": [{"price": "110.0", "qty": "1.0", "commission": "0.11", "commissionAsset": "USDT"}],
        })
        state = NativeSharedState()

        class _MD:
            def get_price(self, _symbol: str) -> float:
                return 110.0

        executor = NativeExecutor(stub, market_data=_MD(), shared_state=state)  # type: ignore[arg-type]

        buy_result = (await executor.execute([Decision("BTCUSDT", Action.OPEN, 100.0, "test", 0.7)]))[0]
        assert buy_result.status == ExecutionStatus.SUCCESS
        assert executor._entry_commission_quote["BTCUSDT"] == pytest.approx(0.1)

        await asyncio.sleep(0.15)  # clear the per-symbol sequential-execution lock (0.1s)
        sell_result = (await executor.execute([Decision("BTCUSDT", Action.CLOSE, 1.0, "test", 0.7)]))[0]
        assert sell_result.status == ExecutionStatus.SUCCESS

        # gross = (110-100)*1.0 = 10.0; net = 10.0 - 0.1 - 0.11 = 9.79
        assert state.metrics["realized_pnl"] == pytest.approx(9.79)
        assert state.metrics["trades_in_window"] == 1
        # Entry fee must be cleared after being consumed, not double-counted on a future close.
        assert "BTCUSDT" not in executor._entry_commission_quote

    @pytest.mark.asyncio
    async def test_daily_target_monitor_records_order_fill_and_trade_closed(self) -> None:
        """Remediation item #18: BUY submission/fill and the final net-of-fee
        close must all reach an injected daily_target_monitor."""
        from core_engine.native.daily_target_monitor import NativeDailyTargetMonitor

        stub = _StubOrderExecution()
        stub.next_buy_result.update({
            "quantity": 1.0, "executed_qty": 1.0, "avg_price": 100.0, "price": 100.0,
            "fills": [{"price": "100.0", "qty": "1.0", "commission": "0.0", "commissionAsset": "USDT"}],
        })
        stub.next_sell_result.update({
            "quantity": 1.0, "executed_qty": 1.0, "avg_price": 110.0, "price": 110.0,
            "fills": [{"price": "110.0", "qty": "1.0", "commission": "0.0", "commissionAsset": "USDT"}],
        })
        state = NativeSharedState()
        monitor = NativeDailyTargetMonitor()

        executor = NativeExecutor(
            stub, market_data=_MD(), shared_state=state, daily_target_monitor=monitor,
        )  # type: ignore[arg-type]

        buy_result = (await executor.execute([Decision("BTCUSDT", Action.OPEN, 100.0, "test", 0.7)]))[0]
        assert buy_result.status == ExecutionStatus.SUCCESS
        assert monitor.state.orders_submitted == 1
        assert monitor.state.entries_filled == 1

        await asyncio.sleep(0.15)
        sell_result = (await executor.execute([Decision("BTCUSDT", Action.CLOSE, 1.0, "test", 0.7)]))[0]
        assert sell_result.status == ExecutionStatus.SUCCESS

        assert monitor.state.trades_closed == 1
        assert monitor.state.net_profitable_trades == 1
        assert monitor.state.compoundable_trades == 1
        assert monitor.state.net_pnl_usdt == pytest.approx(10.0)
