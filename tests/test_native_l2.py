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
