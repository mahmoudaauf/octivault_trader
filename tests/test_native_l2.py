"""
Tests for Native L2 (Phase 8.2.3) — NativeMarketData.

Mocks NativeExchangeClient with a stub. No network.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

import pytest

from core_engine.native import NativeMarketData
from core_engine.native.exchange_client import ExchangeClientError
from core_engine.native.market_data_websocket import NativeMarketDataWebSocket


# ─────────────────────────────────────────────────────────────────────
# Stub
# ─────────────────────────────────────────────────────────────────────
class _StubClient:
    def __init__(self) -> None:
        self.prices = {"BTCUSDT": 50000.0, "ETHUSDT": 3000.0, "XRPUSDT": 0.55}
        self.price_calls = 0
        self.last_price_filter: Optional[list[str]] = None
        self.kline_calls = 0
        self.kline_response: list[list[Any]] = [[1, 2, 3, 4, 5]]
        self.fail_next: int = 0

    async def get_prices(self, symbols: Optional[list[str]] = None) -> dict[str, float]:
        self.price_calls += 1
        self.last_price_filter = list(symbols) if symbols else None
        if self.fail_next > 0:
            self.fail_next -= 1
            raise ExchangeClientError("simulated outage")
        if symbols:
            return {s: self.prices[s] for s in symbols if s in self.prices}
        return dict(self.prices)

    async def get_klines(
        self, symbol: str, interval: str = "1m", limit: int = 100
    ) -> list[list[Any]]:
        self.kline_calls += 1
        return [list(row) for row in self.kline_response]


# ─────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────
class TestNativeMarketDataLifecycle:
    @pytest.mark.asyncio
    async def test_start_primes_then_stops(self) -> None:
        stub = _StubClient()
        md = NativeMarketData(stub, poll_interval_sec=0.5)  # type: ignore[arg-type]
        await md.start()
        assert md.is_running
        assert md.get_price("BTCUSDT") == 50000.0
        assert md.get_price("ETHUSDT") == 3000.0
        await md.stop()
        assert not md.is_running
        assert stub.price_calls >= 1

    @pytest.mark.asyncio
    async def test_idempotent_start_stop(self) -> None:
        stub = _StubClient()
        md = NativeMarketData(stub, poll_interval_sec=10.0)  # type: ignore[arg-type]
        await md.start()
        await md.start()  # second call is no-op
        await md.stop()
        await md.stop()  # second call is no-op
        assert not md.is_running

    @pytest.mark.asyncio
    async def test_loop_polls_multiple_times(self) -> None:
        stub = _StubClient()
        md = NativeMarketData(stub, poll_interval_sec=0.4)  # type: ignore[arg-type]
        await md.start()
        await asyncio.sleep(0.6)
        await md.stop()
        assert stub.price_calls >= 2

    @pytest.mark.asyncio
    async def test_start_can_skip_initial_rest_prime(self) -> None:
        stub = _StubClient()
        md = NativeMarketData(
            stub,
            poll_interval_sec=10.0,
            prime_on_start=False,
        )  # type: ignore[arg-type]
        await md.start()
        await md.stop()
        assert stub.price_calls == 0


class TestNativeMarketDataPrices:
    @pytest.mark.asyncio
    async def test_symbol_filter_passed_to_client(self) -> None:
        stub = _StubClient()
        md = NativeMarketData(  # type: ignore[arg-type]
            stub, poll_interval_sec=10.0, symbols=["BTCUSDT"]
        )
        await md.start()
        await md.stop()
        assert stub.last_price_filter == ["BTCUSDT"]
        # Only BTC should be cached
        assert md.get_price("BTCUSDT") == 50000.0
        assert md.get_price("ETHUSDT") is None

    @pytest.mark.asyncio
    async def test_get_prices_returns_copy(self) -> None:
        stub = _StubClient()
        md = NativeMarketData(stub, poll_interval_sec=10.0)  # type: ignore[arg-type]
        await md.start()
        snap = md.get_prices()
        snap["BTCUSDT"] = 1.0  # mutate copy
        assert md.get_price("BTCUSDT") == 50000.0  # original unchanged
        await md.stop()

    @pytest.mark.asyncio
    async def test_set_symbols_updates_filter(self) -> None:
        stub = _StubClient()
        md = NativeMarketData(stub, poll_interval_sec=10.0)  # type: ignore[arg-type]
        md.set_symbols(["ETHUSDT"])
        assert md.symbols() == ["ETHUSDT"]
        await md.prime()
        assert stub.last_price_filter == ["ETHUSDT"]

    @pytest.mark.asyncio
    async def test_prime_forces_refresh(self) -> None:
        stub = _StubClient()
        md = NativeMarketData(stub, poll_interval_sec=10.0)  # type: ignore[arg-type]
        await md.prime()
        n1 = stub.price_calls
        await md.prime()
        assert stub.price_calls == n1 + 1


class TestNativeMarketDataStaleness:
    @pytest.mark.asyncio
    async def test_unknown_symbol_is_stale(self) -> None:
        stub = _StubClient()
        md = NativeMarketData(stub, poll_interval_sec=10.0)  # type: ignore[arg-type]
        assert md.is_stale("NOPEUSDT") is True
        assert md.price_age("NOPEUSDT") is None

    @pytest.mark.asyncio
    async def test_fresh_quote_not_stale(self) -> None:
        stub = _StubClient()
        md = NativeMarketData(  # type: ignore[arg-type]
            stub, poll_interval_sec=10.0, stale_threshold_sec=5.0
        )
        await md.prime()
        assert md.is_stale("BTCUSDT") is False
        assert (md.price_age("BTCUSDT") or 0.0) < 1.0

    @pytest.mark.asyncio
    async def test_stale_threshold_flags_old_quote(self) -> None:
        stub = _StubClient()
        md = NativeMarketData(  # type: ignore[arg-type]
            stub, poll_interval_sec=10.0, stale_threshold_sec=0.05
        )
        await md.prime()
        await asyncio.sleep(0.1)
        assert md.is_stale("BTCUSDT") is True
        assert "BTCUSDT" in md.stale_symbols()

    @pytest.mark.asyncio
    async def test_refresh_failure_keeps_old_data(self) -> None:
        stub = _StubClient()
        md = NativeMarketData(stub, poll_interval_sec=10.0)  # type: ignore[arg-type]
        await md.prime()
        old_price = md.get_price("BTCUSDT")
        stub.fail_next = 1
        # _refresh_prices propagates the error from prime() (no swallow)
        with pytest.raises(ExchangeClientError):
            await md.prime()
        # Cache from previous successful refresh is preserved
        assert md.get_price("BTCUSDT") == old_price


class TestNativeMarketDataKlines:
    @pytest.mark.asyncio
    async def test_klines_fetch_and_cache_hit(self) -> None:
        stub = _StubClient()
        md = NativeMarketData(stub, poll_interval_sec=10.0)  # type: ignore[arg-type]
        k1 = await md.get_klines("BTCUSDT", "1m", 10, max_age_sec=10.0)
        k2 = await md.get_klines("BTCUSDT", "1m", 10, max_age_sec=10.0)
        assert k1 == k2
        assert stub.kline_calls == 1  # second call was a cache hit

    @pytest.mark.asyncio
    async def test_klines_cache_expiry(self) -> None:
        stub = _StubClient()
        md = NativeMarketData(stub, poll_interval_sec=10.0)  # type: ignore[arg-type]
        await md.get_klines("BTCUSDT", "1m", 10, max_age_sec=0.05)
        await asyncio.sleep(0.1)
        await md.get_klines("BTCUSDT", "1m", 10, max_age_sec=0.05)
        assert stub.kline_calls == 2

    @pytest.mark.asyncio
    async def test_klines_lru_eviction(self) -> None:
        stub = _StubClient()
        md = NativeMarketData(  # type: ignore[arg-type]
            stub, poll_interval_sec=10.0, klines_cache_size=2
        )
        await md.get_klines("BTCUSDT", "1m", 10)
        await md.get_klines("ETHUSDT", "1m", 10)
        await md.get_klines("XRPUSDT", "1m", 10)
        # BTCUSDT should have been evicted (oldest); refetch hits client
        n_before = stub.kline_calls
        await md.get_klines("BTCUSDT", "1m", 10)
        assert stub.kline_calls == n_before + 1

    @pytest.mark.asyncio
    async def test_klines_distinct_keys(self) -> None:
        stub = _StubClient()
        md = NativeMarketData(stub, poll_interval_sec=10.0)  # type: ignore[arg-type]
        await md.get_klines("BTCUSDT", "1m", 10)
        await md.get_klines("BTCUSDT", "5m", 10)
        await md.get_klines("BTCUSDT", "1m", 50)
        # 3 distinct (sym, interval, limit) tuples ⇒ 3 client calls
        assert stub.kline_calls == 3


class _WsState:
    def __init__(self) -> None:
        self.price_cache: dict[str, float] = {}
        self.prices: dict[str, float] = {}
        self.market_data: dict[tuple[str, str], list[dict[str, Any]]] = {}
        self.market_data_ready = False


class TestNativeMarketDataWebSocket:
    @pytest.mark.asyncio
    async def test_handle_message_unwraps_multiplex_ticker_payload(self) -> None:
        state = _WsState()
        ws = NativeMarketDataWebSocket(exchange_client=object(), shared_state=state, symbols=[])
        await ws._handle_message(
            {
                "stream": "btcusdt@ticker",
                "data": {"e": "24hrTicker", "s": "BTCUSDT", "c": "65000.12"},
            }
        )
        assert state.prices["BTCUSDT"] == 65000.12
        assert state.price_cache["BTCUSDT"] == 65000.12
        assert state.market_data_ready is True

    @pytest.mark.asyncio
    async def test_handle_message_unwraps_multiplex_kline_payload(self) -> None:
        state = _WsState()
        ws = NativeMarketDataWebSocket(exchange_client=object(), shared_state=state, symbols=[])
        await ws._handle_message(
            {
                "stream": "btcusdt@kline_1m",
                "data": {
                    "e": "kline",
                    "k": {
                        "s": "BTCUSDT",
                        "i": "1m",
                        "x": True,
                        "t": 1_700_000_000_000,
                        "o": "1.0",
                        "h": "2.0",
                        "l": "0.5",
                        "c": "1.5",
                        "v": "10.0",
                    },
                },
            }
        )
        assert ("BTCUSDT", "1m") in state.market_data
        candle = state.market_data[("BTCUSDT", "1m")][0]
        assert candle["close"] == 1.5
        assert state.market_data_ready is True


# ─────────────────────────────────────────────────────────────────────
# 2026-07-14: bookTicker queue-overflow storm fix regression tests.
# @bookTicker previously shared one multiplexed connection with @ticker/
# @kline, so a bookTicker-side queue overflow tore down price/candle
# delivery too. Fixed by: (1) routing @bookTicker to its own isolated
# connection, (2) a configurable BinanceSocketManager queue size (was an
# unconfigured, too-small library default of 100), (3) only resetting the
# reconnect/backoff counters once a message is actually received, not on
# bare connect (previously let a connect->instant-overflow->disconnect loop
# report "attempt 1/N" forever and never honor max_reconnect_attempts).
# ─────────────────────────────────────────────────────────────────────
class TestBookTickerConnectionIsolation:
    def test_primary_streams_never_include_bookticker(self, monkeypatch) -> None:
        monkeypatch.setenv("WS_ENABLE_BOOKTICKER", "true")
        state = _WsState()
        ws = NativeMarketDataWebSocket(
            exchange_client=object(), shared_state=state, symbols=["BTCUSDT", "ETHUSDT"]
        )
        streams = ws._build_primary_streams()
        assert not any("bookTicker" in s for s in streams)
        assert "btcusdt@ticker" in streams
        assert "btcusdt@kline_1m" in streams

    def test_bookticker_streams_empty_when_disabled(self, monkeypatch) -> None:
        monkeypatch.setenv("WS_ENABLE_BOOKTICKER", "false")
        state = _WsState()
        ws = NativeMarketDataWebSocket(
            exchange_client=object(), shared_state=state, symbols=["BTCUSDT"]
        )
        assert ws._build_bookticker_streams() == []

    def test_bookticker_streams_populated_when_enabled(self, monkeypatch) -> None:
        monkeypatch.setenv("WS_ENABLE_BOOKTICKER", "true")
        state = _WsState()
        ws = NativeMarketDataWebSocket(
            exchange_client=object(), shared_state=state, symbols=["BTCUSDT", "ETHUSDT"]
        )
        streams = ws._build_bookticker_streams()
        assert streams == ["btcusdt@bookTicker", "ethusdt@bookTicker"]

    def test_queue_max_size_defaults_and_env_override(self, monkeypatch) -> None:
        state = _WsState()
        default_ws = NativeMarketDataWebSocket(exchange_client=object(), shared_state=state)
        assert default_ws._queue_max_size == 2000

        monkeypatch.setenv("WS_QUEUE_MAX_SIZE", "5000")
        env_ws = NativeMarketDataWebSocket(exchange_client=object(), shared_state=state)
        assert env_ws._queue_max_size == 5000

        explicit_ws = NativeMarketDataWebSocket(
            exchange_client=object(), shared_state=state, queue_max_size=42
        )
        assert explicit_ws._queue_max_size == 42

    @pytest.mark.asyncio
    async def test_start_launches_two_independent_connection_tasks(self) -> None:
        state = _WsState()
        ws = NativeMarketDataWebSocket(exchange_client=object(), shared_state=state, symbols=[])
        started = []

        async def _fake_run_connection(streams_fn, conn_label, *, critical):
            started.append((conn_label, critical))
            await asyncio.sleep(3600)  # stay "running" until cancelled

        ws._run_connection = _fake_run_connection
        await ws.start()
        try:
            await asyncio.sleep(0)  # let both tasks start
            assert ("primary", True) in started
            assert ("bookTicker", False) in started
            assert ws._ws_task is not None and ws._bookticker_task is not None
        finally:
            await ws.stop()
            assert ws._ws_task.cancelled() or ws._ws_task.done()
            assert ws._bookticker_task.cancelled() or ws._bookticker_task.done()

    @pytest.mark.asyncio
    async def test_reconnect_counter_does_not_reset_until_a_message_is_received(
        self, monkeypatch
    ) -> None:
        """The exact bug scenario: every connect immediately overflows before any
        message arrives. Must count toward max_reconnect_attempts and give up --
        not loop forever reporting 'attempt 1/N'."""
        import binance

        connect_attempts = {"n": 0}

        class _FakeStreamCtx:
            async def __aenter__(self):
                connect_attempts["n"] += 1
                return self

            async def __aexit__(self, *_a):
                return False

            async def recv(self):
                raise RuntimeError("simulated BinanceWebsocketQueueOverflow before any message")

        class _FakeSocketManager:
            def __init__(self, client, max_queue_size=100):
                self.max_queue_size = max_queue_size

            def multiplex_socket(self, streams):
                return _FakeStreamCtx()

        class _FakeAsyncClient:
            def __init__(self, *_a, **_kw):
                pass

            async def close_connection(self):
                pass

        monkeypatch.setattr(binance, "AsyncClient", _FakeAsyncClient)
        monkeypatch.setattr(binance, "BinanceSocketManager", _FakeSocketManager)

        state = _WsState()
        ws = NativeMarketDataWebSocket(
            exchange_client=type("_EC", (), {"api_key": "k", "api_secret": "s"})(),
            shared_state=state,
            symbols=["BTCUSDT"],
            max_reconnect_attempts=3,
            initial_backoff_sec=0.0,
            max_backoff_sec=0.0,
        )
        ws._running = True

        await ws._run_connection(ws._build_primary_streams, "primary", critical=True)

        assert connect_attempts["n"] == 3, (
            "each overflow-before-any-message cycle must count toward the "
            f"reconnect cap; got {connect_attempts['n']} attempts"
        )
        assert ws._running is False, "critical connection exhausting reconnects must stop the feed"

    @pytest.mark.asyncio
    async def test_noncritical_connection_exhausting_reconnects_does_not_stop_feed(
        self, monkeypatch
    ) -> None:
        """bookTicker (critical=False) permanently failing must NOT flip
        self._running -- @ticker/@kline delivery must be unaffected."""
        import binance

        class _FakeStreamCtx:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_a):
                return False

            async def recv(self):
                raise RuntimeError("simulated overflow")

        class _FakeSocketManager:
            def __init__(self, client, max_queue_size=100):
                pass

            def multiplex_socket(self, streams):
                return _FakeStreamCtx()

        class _FakeAsyncClient:
            def __init__(self, *_a, **_kw):
                pass

            async def close_connection(self):
                pass

        monkeypatch.setattr(binance, "AsyncClient", _FakeAsyncClient)
        monkeypatch.setattr(binance, "BinanceSocketManager", _FakeSocketManager)
        monkeypatch.setenv("WS_ENABLE_BOOKTICKER", "true")

        state = _WsState()
        ws = NativeMarketDataWebSocket(
            exchange_client=type("_EC", (), {"api_key": "k", "api_secret": "s"})(),
            shared_state=state,
            symbols=["BTCUSDT"],
            max_reconnect_attempts=2,
            initial_backoff_sec=0.0,
            max_backoff_sec=0.0,
        )
        ws._running = True

        await ws._run_connection(ws._build_bookticker_streams, "bookTicker", critical=False)

        assert ws._running is True
