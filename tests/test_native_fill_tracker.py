"""
Tests for NativeFillTracker: fill detection and position updates.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from core_engine.native.fill_tracker import Fill, NativeFillTracker
from core_engine.native.shared_state import NativeSharedState


class TestNativeFillTracker:
    """Test suite for fill tracker."""

    @pytest.mark.asyncio
    async def test_fill_tracker_initialization(self):
        """Test FillTracker initializes with correct settings."""
        exchange = AsyncMock()
        shared_state = NativeSharedState()
        tracker = NativeFillTracker(
            exchange_client=exchange,
            shared_state=shared_state,
            poll_interval_sec=5.0,
        )

        assert tracker._exchange is exchange
        assert tracker._shared_state is shared_state
        assert tracker._poll_interval == 5.0
        assert tracker._running is False

    def test_fill_initialization(self):
        """Test Fill dataclass is properly constructed."""
        fill = Fill(
            exchange_trade_id=12345,
            symbol="BTCUSDT",
            side="BUY",
            quantity=0.5,
            price=65000.0,
            fee_qty=0.001,
            fee_asset="BTC",
            timestamp_ms=1609459200000,
            commission=32.5,
        )

        assert fill.exchange_trade_id == 12345
        assert fill.symbol == "BTCUSDT"
        assert fill.side == "BUY"
        assert fill.quantity == 0.5
        assert fill.price == 65000.0
        assert fill.commission == 32.5

    def test_fee_metric_converts_assets_and_uses_true_average(self):
        exchange = AsyncMock()
        shared_state = NativeSharedState()
        shared_state.price_cache["BNBUSDT"] = 500.0
        tracker = NativeFillTracker(exchange_client=exchange, shared_state=shared_state)

        base_fee = Fill(
            exchange_trade_id=1, symbol="BTCUSDT", side="BUY", quantity=1.0,
            price=100.0, fee_qty=0.001, fee_asset="BTC",
            timestamp_ms=1609459200000, commission=0.001,
        )
        bnb_fee = Fill(
            exchange_trade_id=2, symbol="BTCUSDT", side="BUY", quantity=1.0,
            price=100.0, fee_qty=0.004, fee_asset="BNB",
            timestamp_ms=1609459200001, commission=0.004,
        )

        first = tracker._record_fee_metric(
            quote_value=100.0, commission_quote=tracker._commission_quote(base_fee)
        )
        second = tracker._record_fee_metric(
            quote_value=100.0, commission_quote=tracker._commission_quote(bnb_fee)
        )

        assert first == pytest.approx(10.0)
        assert second == pytest.approx(200.0)
        assert shared_state.metrics["fee_samples"] == 2
        assert shared_state.metrics["avg_fee_bps"] == pytest.approx(105.0)

    @pytest.mark.asyncio
    async def test_process_buy_fill_creates_new_position(self):
        """Test BUY fill creates a new position."""
        exchange = AsyncMock()
        shared_state = NativeSharedState()
        tracker = NativeFillTracker(
            exchange_client=exchange,
            shared_state=shared_state,
            poll_interval_sec=5.0,
        )

        fill = Fill(
            exchange_trade_id=1,
            symbol="BTCUSDT",
            side="BUY",
            quantity=1.0,
            price=65000.0,
            fee_qty=0.001,
            fee_asset="BTC",
            timestamp_ms=1609459200000,
            commission=32.5,
        )

        await tracker._process_buy_fill(fill)

        # Position should be created
        pos = shared_state.get_position("BTCUSDT")
        assert pos is not None
        assert pos.symbol == "BTCUSDT"
        assert pos.qty == 1.0
        assert pos.entry_price == 65000.0

    @pytest.mark.asyncio
    async def test_process_buy_fill_adds_to_existing_position(self):
        """Test BUY fill adds to existing position (averaging in)."""
        exchange = AsyncMock()
        shared_state = NativeSharedState()
        tracker = NativeFillTracker(
            exchange_client=exchange,
            shared_state=shared_state,
            poll_interval_sec=5.0,
        )

        # Create initial position
        shared_state.update_position("BTCUSDT", 1.0, 65000.0, 65000.0)

        # Add to position with BUY fill at different price
        fill = Fill(
            exchange_trade_id=2,
            symbol="BTCUSDT",
            side="BUY",
            quantity=1.0,
            price=64000.0,
            fee_qty=0.001,
            fee_asset="BTC",
            timestamp_ms=1609459200000,
            commission=32.0,
        )

        await tracker._process_buy_fill(fill)

        pos = shared_state.get_position("BTCUSDT")
        assert pos.qty == 2.0
        # Average entry: (1.0 * 65000 + 1.0 * 64000) / 2.0 = 64500
        assert abs(pos.entry_price - 64500.0) < 1.0

    @pytest.mark.asyncio
    async def test_process_sell_fill_closes_full_position(self):
        """Test SELL fill closes position when qty matches."""
        exchange = AsyncMock()
        shared_state = NativeSharedState()
        tracker = NativeFillTracker(
            exchange_client=exchange,
            shared_state=shared_state,
            poll_interval_sec=5.0,
        )

        # Create position
        shared_state.update_position("BTCUSDT", 1.0, 65000.0, 66000.0)

        # Sell all
        fill = Fill(
            exchange_trade_id=3,
            symbol="BTCUSDT",
            side="SELL",
            quantity=1.0,
            price=66000.0,
            fee_qty=0.001,
            fee_asset="BTC",
            timestamp_ms=1609459200000,
            commission=33.0,
        )

        await tracker._process_sell_fill(fill)

        # Position should be closed
        pos = shared_state.get_position("BTCUSDT")
        assert pos is None

    @pytest.mark.asyncio
    async def test_process_sell_fill_partial_close(self):
        """Test SELL fill reduces position when qty < position."""
        exchange = AsyncMock()
        shared_state = NativeSharedState()
        tracker = NativeFillTracker(
            exchange_client=exchange,
            shared_state=shared_state,
            poll_interval_sec=5.0,
        )

        # Create position
        shared_state.update_position("BTCUSDT", 2.0, 65000.0, 66000.0)

        # Sell partial
        fill = Fill(
            exchange_trade_id=4,
            symbol="BTCUSDT",
            side="SELL",
            quantity=1.0,
            price=66000.0,
            fee_qty=0.0005,
            fee_asset="BTC",
            timestamp_ms=1609459200000,
            commission=33.0,
        )

        await tracker._process_sell_fill(fill)

        # Position should be reduced
        pos = shared_state.get_position("BTCUSDT")
        assert pos is not None
        assert pos.qty == 1.0

    @pytest.mark.asyncio
    async def test_process_fill_routes_to_buy_or_sell(self):
        """Test process_fill routes correctly based on side."""
        exchange = AsyncMock()
        shared_state = NativeSharedState()
        tracker = NativeFillTracker(
            exchange_client=exchange,
            shared_state=shared_state,
            poll_interval_sec=5.0,
        )

        # Test BUY
        buy_fill = Fill(
            exchange_trade_id=5,
            symbol="ETHUSDT",
            side="BUY",
            quantity=10.0,
            price=4000.0,
            fee_qty=0.01,
            fee_asset="ETH",
            timestamp_ms=1609459200000,
            commission=40.0,
        )

        await tracker._process_fill(buy_fill)

        pos = shared_state.get_position("ETHUSDT")
        assert pos is not None
        assert pos.qty == 10.0

    @pytest.mark.asyncio
    async def test_dedup_tracking(self):
        """Test FillTracker deduplicates fills by exchange_trade_id."""
        exchange = AsyncMock()
        shared_state = NativeSharedState()
        tracker = NativeFillTracker(
            exchange_client=exchange,
            shared_state=shared_state,
            poll_interval_sec=5.0,
        )

        assert 123 not in tracker._processed_trades
        tracker._processed_trades.add(123)
        assert 123 in tracker._processed_trades

    @pytest.mark.asyncio
    async def test_start_sets_running_flag(self):
        """Test start() sets _running flag."""
        exchange = AsyncMock()
        shared_state = NativeSharedState()
        tracker = NativeFillTracker(
            exchange_client=exchange,
            shared_state=shared_state,
            poll_interval_sec=5.0,
        )

        assert tracker._running is False
        await tracker.start()
        # Running should be True after start
        assert tracker._running is True

        # Cleanup
        await tracker.stop()

    @pytest.mark.asyncio
    async def test_stop_clears_running_flag(self):
        """Test stop() clears _running flag."""
        exchange = AsyncMock()
        shared_state = NativeSharedState()
        tracker = NativeFillTracker(
            exchange_client=exchange,
            shared_state=shared_state,
            poll_interval_sec=5.0,
        )

        await tracker.start()
        assert tracker._running is True

        await tracker.stop()
        assert tracker._running is False

    @pytest.mark.asyncio
    async def test_fetch_recent_trades_without_method(self):
        """Test fetch_recent_trades returns empty list if exchange doesn't have method."""
        exchange = MagicMock()
        shared_state = NativeSharedState()
        tracker = NativeFillTracker(
            exchange_client=exchange,
            shared_state=shared_state,
            poll_interval_sec=5.0,
        )

        # Remove the method
        delattr(exchange, "get_my_trades")

        fills = await tracker._fetch_recent_trades("BTCUSDT")
        assert fills == []

    @pytest.mark.asyncio
    async def test_emit_fill_event(self):
        """Test fill events are emitted correctly."""
        exchange = AsyncMock()
        shared_state = NativeSharedState()
        tracker = NativeFillTracker(
            exchange_client=exchange,
            shared_state=shared_state,
            poll_interval_sec=5.0,
        )

        # Mock emit_event
        shared_state.emit_event = AsyncMock()

        await tracker._emit_fill_event(
            fill_type="BUY_FILLED",
            symbol="BTCUSDT",
            qty=1.0,
            price=65000.0,
            commission=32.5,
        )

        # Verify event was emitted
        shared_state.emit_event.assert_called_once()
        call_args = shared_state.emit_event.call_args
        assert call_args[0][0] == "BUY_FILLED"
        payload = call_args[0][1]
        assert payload["symbol"] == "BTCUSDT"
        assert payload["qty"] == 1.0
