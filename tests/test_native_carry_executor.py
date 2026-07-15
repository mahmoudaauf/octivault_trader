"""
Tests for CarryLegExecutor (Phase 3 of the funding-carry native-wiring plan).

Covers: happy-path open/close, the new "confirm perp filled before firing
spot" safety net, and the leg-mismatch alarm (naked-leg detection + kill
file) for both open and close paths.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from core_engine.native.carry.executor import CarryLegExecutor
from core_engine.native.carry.state import CarrySharedState


def _carry_state(tmp_path) -> CarrySharedState:
    return CarrySharedState(
        state_path=str(tmp_path / "carry_state.json"),
        ledger_path=str(tmp_path / "carry_ledger.jsonl"),
    )


def _filled(**overrides):
    base = {"status": "FILLED", "symbol": "BTCUSDT"}
    base.update(overrides)
    return base


def _not_filled(**overrides):
    base = {"status": "NEW", "symbol": "BTCUSDT"}
    base.update(overrides)
    return base


def _executor(tmp_path, *, futures=None, spot=None, **overrides) -> CarryLegExecutor:
    cs = _carry_state(tmp_path)
    kwargs = dict(
        futures_client=futures or AsyncMock(),
        spot_client=spot or AsyncMock(),
        carry_state=cs,
        leverage=2,
        mismatch_kill_file=str(tmp_path / "carry.stop"),
    )
    kwargs.update(overrides)
    return CarryLegExecutor(**kwargs)


class TestOpenHedge:
    @pytest.mark.asyncio
    async def test_happy_path_both_legs_filled(self, tmp_path) -> None:
        futures = AsyncMock()
        futures.futures_mark_price.return_value = {"symbol": "BTCUSDT", "markPrice": "50000.0"}
        futures.futures_change_leverage.return_value = {"leverage": 2}
        futures.futures_create_order.return_value = _filled(side="SELL")
        spot = AsyncMock()
        spot.place_order.return_value = _filled(side="BUY")

        ex = _executor(tmp_path, futures=futures, spot=spot)
        result = await ex.open_hedge("BTCUSDT", 0.0007, 500.0)

        assert result.success is True
        assert result.position is not None
        assert result.position.symbol == "BTCUSDT"
        assert result.position.notional_usd == pytest.approx(500.0)
        assert ex._carry_state.get_open_hedge("BTCUSDT") is not None
        futures.futures_change_leverage.assert_awaited_once_with("BTCUSDT", 2)
        spot.place_order.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_positive_funding_shorts_perp_and_longs_spot(self, tmp_path) -> None:
        futures = AsyncMock()
        futures.futures_mark_price.return_value = {"markPrice": "50000.0"}
        futures.futures_create_order.return_value = _filled()
        spot = AsyncMock()
        spot.place_order.return_value = _filled()

        ex = _executor(tmp_path, futures=futures, spot=spot)
        await ex.open_hedge("BTCUSDT", 0.0007, 500.0)

        perp_args = futures.futures_create_order.call_args
        assert perp_args[0][1] == "SELL"  # symbol, side, qty positional
        spot_args = spot.place_order.call_args
        assert spot_args[0][1] == "BUY"

    @pytest.mark.asyncio
    async def test_perp_not_filled_never_fires_spot_leg(self, tmp_path) -> None:
        """The new safety net: if the perp leg isn't confirmed FILLED, the
        spot leg must never be sent -- no naked position, no mismatch alarm."""
        futures = AsyncMock()
        futures.futures_mark_price.return_value = {"markPrice": "50000.0"}
        futures.futures_create_order.return_value = _not_filled()
        spot = AsyncMock()

        ex = _executor(tmp_path, futures=futures, spot=spot)
        result = await ex.open_hedge("BTCUSDT", 0.0007, 500.0)

        assert result.success is False
        assert "perp_leg_not_filled" in result.reason
        assert result.naked_leg is None
        spot.place_order.assert_not_awaited()
        assert ex._carry_state.get_open_hedge("BTCUSDT") is None
        assert not ex._carry_state.state_path.exists() or ex._carry_state.open_count() == 0

    @pytest.mark.asyncio
    async def test_perp_leg_exception_never_fires_spot_leg(self, tmp_path) -> None:
        futures = AsyncMock()
        futures.futures_mark_price.return_value = {"markPrice": "50000.0"}
        futures.futures_create_order.side_effect = RuntimeError("network error")
        spot = AsyncMock()

        ex = _executor(tmp_path, futures=futures, spot=spot)
        result = await ex.open_hedge("BTCUSDT", 0.0007, 500.0)

        assert result.success is False
        assert "perp_leg_failed" in result.reason
        spot.place_order.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_spot_leg_failure_after_perp_filled_raises_mismatch_alarm(self, tmp_path) -> None:
        futures = AsyncMock()
        futures.futures_mark_price.return_value = {"markPrice": "50000.0"}
        futures.futures_create_order.return_value = _filled()
        spot = AsyncMock()
        spot.place_order.side_effect = RuntimeError("insufficient balance")

        kill_file = tmp_path / "carry.stop"
        ex = _executor(tmp_path, futures=futures, spot=spot, mismatch_kill_file=str(kill_file))
        result = await ex.open_hedge("BTCUSDT", 0.0007, 500.0)

        assert result.success is False
        assert result.naked_leg == "perp"
        assert kill_file.exists(), "leg mismatch must touch the kill file to halt new opens"
        # No hedge should be recorded -- the spot leg never actually filled.
        assert ex._carry_state.get_open_hedge("BTCUSDT") is None

    @pytest.mark.asyncio
    async def test_spot_leg_not_filled_after_perp_filled_raises_mismatch_alarm(self, tmp_path) -> None:
        futures = AsyncMock()
        futures.futures_mark_price.return_value = {"markPrice": "50000.0"}
        futures.futures_create_order.return_value = _filled()
        spot = AsyncMock()
        spot.place_order.return_value = _not_filled()

        kill_file = tmp_path / "carry.stop"
        ex = _executor(tmp_path, futures=futures, spot=spot, mismatch_kill_file=str(kill_file))
        result = await ex.open_hedge("BTCUSDT", 0.0007, 500.0)

        assert result.success is False
        assert result.naked_leg == "perp"
        assert kill_file.exists()

    @pytest.mark.asyncio
    async def test_invalid_mark_price_aborts_before_any_orders(self, tmp_path) -> None:
        futures = AsyncMock()
        futures.futures_mark_price.return_value = {"markPrice": "0"}
        spot = AsyncMock()

        ex = _executor(tmp_path, futures=futures, spot=spot)
        result = await ex.open_hedge("BTCUSDT", 0.0007, 500.0)

        assert result.success is False
        assert result.reason == "invalid_mark_price"
        futures.futures_create_order.assert_not_awaited()
        spot.place_order.assert_not_awaited()


class TestCloseHedge:
    @pytest.mark.asyncio
    async def test_happy_path_both_legs_close_filled(self, tmp_path) -> None:
        futures = AsyncMock()
        futures.futures_create_order.return_value = _filled()
        spot = AsyncMock()
        spot.place_order.return_value = _filled()

        ex = _executor(tmp_path, futures=futures, spot=spot)
        ex._carry_state.open_hedge(
            "BTCUSDT", entry_funding=0.0007, perp_qty=0.01, spot_qty=0.01, notional_usd=500.0
        )
        result = await ex.close_hedge("BTCUSDT")

        assert result.success is True
        assert ex._carry_state.get_open_hedge("BTCUSDT") is None
        # closing a short_perp position must BUY to cover, reduce_only=True
        _, kwargs = futures.futures_create_order.call_args
        assert kwargs.get("reduce_only") is True
        perp_call_args = futures.futures_create_order.call_args[0]
        assert perp_call_args[1] == "BUY"

    @pytest.mark.asyncio
    async def test_close_not_open_symbol(self, tmp_path) -> None:
        ex = _executor(tmp_path)
        result = await ex.close_hedge("BTCUSDT")
        assert result.success is False
        assert result.reason == "not_open"

    @pytest.mark.asyncio
    async def test_close_perp_leg_not_filled_never_fires_spot_leg(self, tmp_path) -> None:
        futures = AsyncMock()
        futures.futures_create_order.return_value = _not_filled()
        spot = AsyncMock()

        ex = _executor(tmp_path, futures=futures, spot=spot)
        ex._carry_state.open_hedge(
            "BTCUSDT", entry_funding=0.0007, perp_qty=0.01, spot_qty=0.01, notional_usd=500.0
        )
        result = await ex.close_hedge("BTCUSDT")

        assert result.success is False
        spot.place_order.assert_not_awaited()
        # Position must remain open -- close didn't actually happen.
        assert ex._carry_state.get_open_hedge("BTCUSDT") is not None

    @pytest.mark.asyncio
    async def test_close_spot_leg_failure_raises_mismatch_alarm(self, tmp_path) -> None:
        futures = AsyncMock()
        futures.futures_create_order.return_value = _filled()
        spot = AsyncMock()
        spot.place_order.side_effect = RuntimeError("network error")

        kill_file = tmp_path / "carry.stop"
        ex = _executor(tmp_path, futures=futures, spot=spot, mismatch_kill_file=str(kill_file))
        ex._carry_state.open_hedge(
            "BTCUSDT", entry_funding=0.0007, perp_qty=0.01, spot_qty=0.01, notional_usd=500.0
        )
        result = await ex.close_hedge("BTCUSDT")

        assert result.success is False
        assert result.naked_leg == "spot"
        assert kill_file.exists()
        # The perp side is now closed but the hedge is still tracked as
        # "open" in state (since we don't know the true position without a
        # reconciliation pass) -- confirms we don't silently drop tracking.
        assert ex._carry_state.get_open_hedge("BTCUSDT") is not None
