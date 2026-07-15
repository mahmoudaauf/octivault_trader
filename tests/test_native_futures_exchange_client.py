"""
Tests for NativeFuturesExchangeClient (Phase 1 of the funding-carry
native-wiring plan — see the engineering-study plan doc).

Mirrors tests/test_native_l1.py's TestNativeExchangeClient conventions:
signing, base URLs, paper-mode gating on every signed/mutating method,
and market-data parsing.
"""

from __future__ import annotations

import hashlib
import hmac
from unittest.mock import AsyncMock

import pytest

from core_engine.native.futures_exchange_client import (
    FuturesExchangeClientError,
    NativeFuturesExchangeClient,
)
from core_engine.native.retry_manager import NativeRetryManager


class TestNativeFuturesExchangeClient:
    def _client(self) -> NativeFuturesExchangeClient:
        retry = NativeRetryManager(max_attempts=1, base_delay_sec=0.0, jitter=False)
        return NativeFuturesExchangeClient("KEY", "SECRET", retry=retry)

    def _paper_client(self) -> NativeFuturesExchangeClient:
        retry = NativeRetryManager(max_attempts=1, base_delay_sec=0.0, jitter=False)
        return NativeFuturesExchangeClient("paper_key", "paper_secret", retry=retry)

    # ── signing / base URL ──────────────────────────────────────────────
    def test_sign_includes_signature_timestamp_recvwindow(self) -> None:
        c = self._client()
        params = c._sign({"symbol": "BTCUSDT"})
        assert "signature" in params
        assert "timestamp" in params
        assert "recvWindow" in params
        assert params["symbol"] == "BTCUSDT"
        assert len(params["signature"]) == 64
        int(params["signature"], 16)

    def test_sign_signature_matches_hmac(self) -> None:
        c = self._client()
        params = c._sign({"symbol": "BTCUSDT", "timestamp": 1, "recvWindow": 5000})
        from urllib.parse import urlencode

        sig = params.pop("signature")
        expected = hmac.new(
            b"SECRET", urlencode(params, doseq=True).encode(), hashlib.sha256
        ).hexdigest()
        assert sig == expected

    def test_default_base_url(self) -> None:
        c = NativeFuturesExchangeClient("k", "s")
        assert c.base_url == NativeFuturesExchangeClient.DEFAULT_BASE_URL
        assert "fapi.binance.com" in c.base_url

    def test_testnet_base_url(self) -> None:
        c = NativeFuturesExchangeClient("k", "s", testnet=True)
        assert c.base_url == NativeFuturesExchangeClient.DEFAULT_TESTNET_URL
        assert "testnet" in c.base_url

    def test_base_url_distinct_from_spot(self) -> None:
        """Sanity: the futures base URL must never collide with the spot
        client's — this is a genuinely separate account/venue."""
        from core_engine.native.exchange_client import NativeExchangeClient

        assert (
            NativeFuturesExchangeClient.DEFAULT_BASE_URL
            != NativeExchangeClient.DEFAULT_BASE_URL
        )

    # ── market data parsing ─────────────────────────────────────────────
    @pytest.mark.asyncio
    async def test_futures_mark_price_bulk_call_has_no_symbol_param(self) -> None:
        c = self._client()
        c._request = AsyncMock(return_value=[{"symbol": "BTCUSDT", "lastFundingRate": "0.0001"}])
        await c.futures_mark_price()
        c._request.assert_awaited_once_with("GET", c.EP_PREMIUM_INDEX, params=None)

    @pytest.mark.asyncio
    async def test_futures_mark_price_single_symbol(self) -> None:
        c = self._client()
        c._request = AsyncMock(return_value={"symbol": "BTCUSDT", "markPrice": "50000.0"})
        result = await c.futures_mark_price("BTCUSDT")
        c._request.assert_awaited_once_with("GET", c.EP_PREMIUM_INDEX, params={"symbol": "BTCUSDT"})
        assert result["markPrice"] == "50000.0"

    @pytest.mark.asyncio
    async def test_futures_funding_rate_pagination_params(self) -> None:
        c = self._client()
        c._request = AsyncMock(return_value=[{"fundingRate": "0.0002"}])
        rows = await c.futures_funding_rate("BTCUSDT", start_time=123456, limit=500)
        c._request.assert_awaited_once_with(
            "GET", c.EP_FUNDING_RATE, params={"symbol": "BTCUSDT", "limit": 500, "startTime": 123456}
        )
        assert rows == [{"fundingRate": "0.0002"}]

    @pytest.mark.asyncio
    async def test_futures_funding_rate_non_list_response_returns_empty(self) -> None:
        c = self._client()
        c._request = AsyncMock(return_value={"unexpected": "shape"})
        rows = await c.futures_funding_rate("BTCUSDT")
        assert rows == []

    # ── paper-mode gating: every signed/mutating call must never hit the network ──
    @pytest.mark.asyncio
    async def test_futures_balance_paper_mode_never_calls_request(self) -> None:
        c = self._paper_client()
        c._request = AsyncMock()
        bal = await c.futures_balance()
        c._request.assert_not_awaited()
        assert bal == {"USDT": 1000.0}

    @pytest.mark.asyncio
    async def test_futures_position_information_paper_mode_returns_empty(self) -> None:
        c = self._paper_client()
        c._request = AsyncMock()
        positions = await c.futures_position_information()
        c._request.assert_not_awaited()
        assert positions == []

    @pytest.mark.asyncio
    async def test_futures_change_leverage_paper_mode_never_calls_request(self) -> None:
        c = self._paper_client()
        c._request = AsyncMock()
        result = await c.futures_change_leverage("BTCUSDT", 2)
        c._request.assert_not_awaited()
        assert result["leverage"] == 2

    @pytest.mark.asyncio
    async def test_futures_create_order_paper_mode_never_calls_request(self) -> None:
        c = self._paper_client()
        c._request = AsyncMock()
        c.futures_mark_price = AsyncMock(return_value={"symbol": "BTCUSDT", "markPrice": "50000.0"})
        order = await c.futures_create_order("BTCUSDT", "SELL", 0.001)
        c._request.assert_not_awaited()
        assert order["status"] == "FILLED"
        assert order["side"] == "SELL"

    @pytest.mark.asyncio
    async def test_futures_create_order_paper_mode_prices_off_mark_not_last_traded(self) -> None:
        """The paper simulator must use futures_mark_price (mark price), never
        a last-traded-price source — the strategy's own funding math is keyed
        off mark/premium-index data, so the fill price must be consistent."""
        c = self._paper_client()
        c.futures_mark_price = AsyncMock(return_value={"symbol": "BTCUSDT", "markPrice": "61234.5"})
        order = await c.futures_create_order("BTCUSDT", "SELL", 0.01)
        c.futures_mark_price.assert_awaited_once_with("BTCUSDT")
        assert float(order["avgPrice"]) == pytest.approx(61234.5)

    @pytest.mark.asyncio
    async def test_futures_create_order_invalid_side(self) -> None:
        c = self._client()
        with pytest.raises(ValueError):
            await c.futures_create_order("BTCUSDT", "HOLD", 0.001)

    @pytest.mark.asyncio
    async def test_futures_create_order_reduce_only_param(self) -> None:
        c = self._client()
        c._request = AsyncMock(return_value={"status": "FILLED"})
        await c.futures_create_order("BTCUSDT", "BUY", 0.001, reduce_only=True)
        _, kwargs = c._request.call_args
        assert kwargs["params"]["reduceOnly"] == "true"

    @pytest.mark.asyncio
    async def test_futures_create_order_limit_requires_price(self) -> None:
        c = self._client()
        with pytest.raises(ValueError):
            await c.futures_create_order("BTCUSDT", "BUY", 0.001, order_type="LIMIT")

    # ── throttle / budget (mirrors spot client's already-proven behavior) ──
    def test_record_error_sets_throttle_window(self) -> None:
        c = self._client()
        c._record_error(429, "rate limited")
        assert c.is_throttled()

    def test_restore_throttle_state_rehydrates_client(self) -> None:
        c = self._client()
        import time as _time

        future_ts = _time.time() + 100
        c.restore_throttle_state(until_ts=future_ts, reason="persisted ban")
        assert c.is_throttled()
        assert c.last_error() == "persisted ban"

    @pytest.mark.asyncio
    async def test_request_short_circuits_while_throttled(self) -> None:
        c = self._client()
        c._throttled_until_ts = 9_999_999_999.0
        with pytest.raises(FuturesExchangeClientError):
            await c._request("GET", c.EP_EXCHANGE_INFO)
