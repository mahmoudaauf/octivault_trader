"""
Unit tests for Native L0 components (Phase 8.2.1)

Tests:
    - NativeSharedState: Position tracking, NAV management, balance
    - NativeTimeUtils: Timestamps, candle alignment, formatting
    - ConfigLoader: Config loading and retrieval
    - NativeRetryManager: Retries, exponential backoff, jitter
"""


import pytest

from core_engine.native import (
    ConfigLoader,
    NativeRetryManager,
    NativeSharedState,
    NativeTimeUtils,
)

# ==================== NativeSharedState Tests ====================


class TestNativeSharedState:
    """Test NativeSharedState component"""

    def test_nav_update_and_get(self):
        """Test NAV update and retrieval"""
        state = NativeSharedState()
        assert state.get_nav() == 0.0

        state.update_nav(100.0)
        assert state.get_nav() == 100.0

        state.update_nav(86.99)
        assert state.get_nav() == 86.99

    def test_nav_never_negative(self):
        """Test NAV cannot go negative"""
        state = NativeSharedState()
        state.update_nav(-50.0)
        assert state.get_nav() == 0.0

    def test_balance_update(self):
        """Test balance update"""
        state = NativeSharedState()
        state.update_balance(50.0, 30.0)

        assert state.free_balance_usdt == 50.0
        assert state.invested_capital_usdt == 30.0
        assert state.get_nav() == 80.0  # Auto-calculated

    def test_position_tracking(self):
        """Test position management"""
        state = NativeSharedState()

        # Add position
        state.update_position("ETHUSDT", qty=1.0, entry=2000.0, current=2360.0)
        assert len(state.get_all_positions()) == 1

        # Retrieve position
        pos = state.get_position("ETHUSDT")
        assert pos.qty == 1.0
        assert pos.entry_price == 2000.0
        assert pos.mark_price == 2360.0
        assert pos.unrealized_pnl_pct > 0

    def test_position_closes_at_dust(self):
        """Test position is removed when qty too small"""
        state = NativeSharedState()

        state.update_position("BTCUSDT", qty=1e-9, entry=80000, current=90000)
        assert len(state.get_all_positions()) == 0

    def test_portfolio_value_calculation(self):
        """Test portfolio value"""
        state = NativeSharedState()

        state.update_position("ETHUSDT", qty=1.0, entry=2000, current=2360)
        state.update_position("BNBUSDT", qty=0.1, entry=600, current=630)

        portfolio_value = state.get_portfolio_value()
        expected = 1.0 * 2360 + 0.1 * 630
        assert abs(portfolio_value - expected) < 0.01

    def test_dict_backed_positions_are_counted_in_portfolio_value_and_qty(self):
        state = NativeSharedState()
        state.positions = {
            "BNBUSDT": {"qty": 0.5, "entry_price": 600.0, "mark_price": 650.0},
        }

        assert state.get_position_qty("BNBUSDT") == 0.5
        assert abs(state.get_portfolio_value() - 325.0) < 0.01

    def test_symbol_management(self):
        """Test symbol tracking"""
        state = NativeSharedState()

        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
        state.set_accepted_symbols(symbols)

        assert state.get_accepted_symbols() == set(symbols)

        state.add_accepted_symbol("SOLUSDT")
        assert "SOLUSDT" in state.get_accepted_symbols()

    def test_dust_tracking(self):
        """Test dust symbol tracking"""
        state = NativeSharedState()

        state.mark_dust("PEPEUSDT")
        assert state.is_dust("PEPEUSDT")
        assert not state.is_dust("ETHUSDT")

    def test_order_tracking(self):
        """Test order management"""
        state = NativeSharedState()

        state.add_order("12345", "ETHUSDT", "BUY", 1.0, 2360.0)
        assert state.get_open_order_count() == 1

        state.mark_order_filled("12345")
        assert state.open_orders["12345"].status == "FILLED"

    def test_price_update_marks_tick_freshness(self):
        state = NativeSharedState()
        state.update_price("btcusdt", 65000.0)
        assert state.get_price("BTCUSDT") == 65000.0
        assert "BTCUSDT" in state._last_tick_timestamps

    @pytest.mark.asyncio
    async def test_hydration_ready_event(self):
        """Test hydration ready event"""
        state = NativeSharedState()
        assert not state.is_ready()

        state.mark_ready()
        assert state.is_ready()


# ==================== Order-Book History Persistence Tests ====================
# 2026-07-14: update_book() previously only kept the latest in-memory snapshot,
# so no order-book history was ever available for backtesting. These tests
# cover the new throttled JSONL persistence hook.


class TestOrderBookHistoryPersistence:
    def _make_state(self, tmp_path, **overrides):
        state = NativeSharedState()
        state._orderbook_history_dir = str(tmp_path)
        state._orderbook_history_min_interval_s = overrides.get("min_interval_s", 60.0)
        state._orderbook_history_enabled = overrides.get("enabled", True)
        return state

    def _read_jsonl(self, path):
        import json

        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]

    def test_update_book_writes_one_record_on_first_tick(self, tmp_path):
        state = self._make_state(tmp_path)
        state.update_book("btcusdt", bid=100.0, bid_qty=1.0, ask=100.2, ask_qty=2.0)

        files = list(tmp_path.glob("orderbook_history_*.jsonl"))
        assert len(files) == 1
        records = self._read_jsonl(files[0])
        assert len(records) == 1
        rec = records[0]
        assert rec["symbol"] == "BTCUSDT"
        assert rec["bid"] == 100.0
        assert rec["ask"] == 100.2
        assert "spread_pct" in rec and "imbalance" in rec and "ts" in rec

    def test_rapid_ticks_within_min_interval_are_throttled(self, tmp_path):
        state = self._make_state(tmp_path, min_interval_s=60.0)
        for _ in range(5):
            state.update_book("BTCUSDT", bid=100.0, bid_qty=1.0, ask=100.2, ask_qty=2.0)

        files = list(tmp_path.glob("orderbook_history_*.jsonl"))
        records = self._read_jsonl(files[0])
        # Only the first tick should have been persisted -- the rest fall
        # inside the same throttle window (all calls happen well under 60s apart).
        assert len(records) == 1

    def test_throttle_is_independent_per_symbol(self, tmp_path):
        state = self._make_state(tmp_path, min_interval_s=60.0)
        state.update_book("BTCUSDT", bid=100.0, bid_qty=1.0, ask=100.2, ask_qty=2.0)
        state.update_book("ETHUSDT", bid=50.0, bid_qty=1.0, ask=50.1, ask_qty=2.0)

        files = list(tmp_path.glob("orderbook_history_*.jsonl"))
        records = self._read_jsonl(files[0])
        symbols = {r["symbol"] for r in records}
        assert symbols == {"BTCUSDT", "ETHUSDT"}

    def test_disabled_flag_skips_persistence_entirely(self, tmp_path):
        state = self._make_state(tmp_path, enabled=False)
        state.update_book("BTCUSDT", bid=100.0, bid_qty=1.0, ask=100.2, ask_qty=2.0)

        assert list(tmp_path.glob("orderbook_history_*.jsonl")) == []
        # In-memory snapshot must still work regardless of persistence toggle.
        assert state.get_book("BTCUSDT")["bid"] == 100.0

    def test_invalid_book_tick_is_never_persisted(self, tmp_path):
        state = self._make_state(tmp_path)
        state.update_book("BTCUSDT", bid=0.0, bid_qty=1.0, ask=100.2, ask_qty=2.0)

        assert list(tmp_path.glob("orderbook_history_*.jsonl")) == []

    def test_persist_failure_does_not_raise_or_break_in_memory_update(
        self, tmp_path, monkeypatch
    ):
        state = self._make_state(tmp_path)

        def _boom(*_args, **_kwargs):
            raise OSError("disk full")

        monkeypatch.setattr("core_engine.native.shared_state.os.makedirs", _boom)

        # Must not raise despite the persistence path failing internally.
        state.update_book("BTCUSDT", bid=100.0, bid_qty=1.0, ask=100.2, ask_qty=2.0)
        assert state.get_book("BTCUSDT")["bid"] == 100.0


# ==================== NativeTimeUtils Tests ====================


class TestNativeTimeUtils:
    """Test NativeTimeUtils component"""

    def test_unix_now_ms(self):
        """Test Unix timestamp in milliseconds"""
        now_ms = NativeTimeUtils.unix_now_ms()
        assert isinstance(now_ms, int)
        assert now_ms > 1700000000000  # After 2023-11-01

    def test_unix_now_s(self):
        """Test Unix timestamp in seconds"""
        now_s = NativeTimeUtils.unix_now_s()
        assert isinstance(now_s, float)
        assert now_s > 1700000000  # After 2023-11-01

    def test_iso_now(self):
        """Test ISO8601 timestamp"""
        iso = NativeTimeUtils.iso_now()
        assert isinstance(iso, str)
        assert "+" in iso or "Z" in iso  # Has timezone
        assert "T" in iso  # ISO format

    def test_candle_alignment_1m(self):
        """Test 1-minute candle alignment"""
        # Use a known timestamp
        unix_ms = 1000000  # arbitrary
        aligned = NativeTimeUtils.align_candle_time(unix_ms, 60)  # 1-minute

        # Should align to minute boundary (60000 ms)
        assert aligned % 60000 == 0

    def test_candle_alignment_5m(self):
        """Test 5-minute candle alignment"""
        unix_ms = 1000000
        aligned = NativeTimeUtils.align_candle_time(unix_ms, 300)  # 5-minute

        # Should align to 5-minute boundary (300000 ms)
        assert aligned % 300000 == 0

    def test_seconds_until_next_candle(self):
        """Test seconds until next candle"""
        seconds = NativeTimeUtils.seconds_until_next_candle(60)

        # Should be between 0 and 60
        assert 0 <= seconds <= 60

    def test_format_duration_seconds(self):
        """Test duration formatting"""
        # 1 hour 30 minutes 45 seconds
        formatted = NativeTimeUtils.format_duration_sec(5445)
        assert "1h" in formatted
        assert "30m" in formatted
        assert "45s" in formatted

    def test_is_market_hours(self):
        """Test market hours (crypto is 24/7)"""
        assert NativeTimeUtils.is_market_hours(1000000) is True


# ==================== ConfigLoader Tests ====================


class TestConfigLoader:
    """Test ConfigLoader component"""

    def test_config_initialization(self):
        """Test config loads without errors"""
        config = ConfigLoader()
        assert config is not None

    def test_symbols_config(self):
        """Test symbols configuration"""
        config = ConfigLoader()
        symbols = config.get("SYMBOLS", "symbols")

        assert isinstance(symbols, list)
        assert len(symbols) > 0
        assert "BTCUSDT" in symbols

    def test_capital_config(self):
        """Test capital configuration"""
        config = ConfigLoader()
        reserve_pct = config.get("CAPITAL", "reserve_pct")

        assert isinstance(reserve_pct, float)
        assert 0 < reserve_pct <= 1

    def test_get_group(self):
        """Test getting entire config group"""
        config = ConfigLoader()
        capital_group = config.get_group("CAPITAL")

        assert isinstance(capital_group, dict)
        assert len(capital_group) > 0
        assert "reserve_pct" in capital_group

    def test_get_all(self):
        """Test getting all config"""
        config = ConfigLoader()
        all_config = config.get_all()

        assert isinstance(all_config, dict)
        assert len(all_config) > 3  # At least 3 groups

    def test_default_values(self):
        """Test default values for missing keys"""
        config = ConfigLoader()
        value = config.get("NONEXISTENT", "key", default="fallback")

        assert value == "fallback"


# ==================== NativeRetryManager Tests ====================


class TestNativeRetryManager:
    """Test NativeRetryManager component"""

    @pytest.mark.asyncio
    async def test_successful_call_first_try(self):
        """Test successful call on first attempt"""

        async def success_func():
            return "success"

        manager = NativeRetryManager()
        result = await manager.call(success_func)

        assert result == "success"

    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        """Test retry after failure"""
        attempts = [0]

        async def fail_then_succeed():
            attempts[0] += 1
            if attempts[0] < 2:
                raise ValueError("First attempt fails")
            return "success"

        manager = NativeRetryManager(max_attempts=3, base_delay_sec=0.01)
        result = await manager.call(fail_then_succeed)

        assert result == "success"
        assert attempts[0] == 2

    @pytest.mark.asyncio
    async def test_max_attempts_exceeded(self):
        """Test exception after max attempts"""

        async def always_fails():
            raise ValueError("Always fails")

        manager = NativeRetryManager(max_attempts=2, base_delay_sec=0.01)

        with pytest.raises(ValueError):
            await manager.call(always_fails)

    @pytest.mark.asyncio
    async def test_call_with_fallback(self):
        """Test fallback value on all failures"""

        async def always_fails():
            raise ValueError("Always fails")

        manager = NativeRetryManager(max_attempts=2, base_delay_sec=0.01)
        result = await manager.call_with_fallback(always_fails, "fallback_value")

        assert result == "fallback_value"

    def test_delay_calculation(self):
        """Test exponential backoff calculation"""
        manager = NativeRetryManager(base_delay_sec=0.1, max_delay_sec=10.0, jitter=False)

        delay_1 = manager._calculate_delay(1)
        delay_2 = manager._calculate_delay(2)
        delay_3 = manager._calculate_delay(3)

        # Should be exponential
        assert delay_1 == 0.1
        assert delay_2 == 0.2
        assert delay_3 == 0.4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
