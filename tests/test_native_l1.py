"""
Tests for Native L1 (Phase 8.2.2).

Covers:
* NativeExchangeClient: signing, request plumbing (mocked), parsers.
* NativeBalanceSync: lifecycle, cache, callback (sync + async).
* NativeOrderExecution: place/cancel/refresh, OrderResult mapping.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
from typing import Any, Optional
from unittest.mock import AsyncMock

import pytest

from core_engine.native import (
    ExchangeClientError,
    NativeBalanceSync,
    NativeExchangeClient,
    NativeOrderExecution,
    NativeRetryManager,
    OrderResult,
)


# ─────────────────────────────────────────────────────────────────────
# NativeExchangeClient
# ─────────────────────────────────────────────────────────────────────
class TestNativeExchangeClient:
    def _client(self) -> NativeExchangeClient:
        # Tight retry so tests don't sleep.
        retry = NativeRetryManager(max_attempts=1, base_delay_sec=0.0, jitter=False)
        return NativeExchangeClient("KEY", "SECRET", retry=retry)

    def test_sign_includes_signature_timestamp_recvwindow(self) -> None:
        c = self._client()
        params = c._sign({"symbol": "BTCUSDT"})
        assert "signature" in params
        assert "timestamp" in params
        assert "recvWindow" in params
        assert params["symbol"] == "BTCUSDT"
        # signature is hex sha256
        assert len(params["signature"]) == 64
        int(params["signature"], 16)  # raises if not hex

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
        c = NativeExchangeClient("k", "s")
        assert c.base_url == NativeExchangeClient.DEFAULT_BASE_URL

    def test_testnet_base_url(self) -> None:
        c = NativeExchangeClient("k", "s", testnet=True)
        assert c.base_url == NativeExchangeClient.DEFAULT_TESTNET_URL

    @pytest.mark.asyncio
    async def test_get_balance_filters_zeros(self) -> None:
        c = self._client()
        c._request = AsyncMock(  # type: ignore[assignment]
            return_value={
                "balances": [
                    {"asset": "BTC", "free": "0.5", "locked": "0"},
                    {"asset": "ETH", "free": "0", "locked": "0"},
                    {"asset": "USDT", "free": "100.25", "locked": "0"},
                ]
            }
        )
        bal = await c.get_balance()
        assert bal == {"BTC": 0.5, "USDT": 100.25}

    @pytest.mark.asyncio
    async def test_get_prices_multi(self) -> None:
        c = self._client()
        c._request = AsyncMock(  # type: ignore[assignment]
            return_value=[
                {"symbol": "BTCUSDT", "price": "50000.00"},
                {"symbol": "ETHUSDT", "price": "3000.00"},
                {"symbol": "XRPUSDT", "price": "0.55"},
            ]
        )
        prices = await c.get_prices(["BTCUSDT", "ETHUSDT"])
        assert prices == {"BTCUSDT": 50000.0, "ETHUSDT": 3000.0}

    @pytest.mark.asyncio
    async def test_get_prices_single_dict_response(self) -> None:
        c = self._client()
        c._request = AsyncMock(  # type: ignore[assignment]
            return_value={"symbol": "BTCUSDT", "price": "50000.0"}
        )
        prices = await c.get_prices(["BTCUSDT"])
        assert prices == {"BTCUSDT": 50000.0}

    @pytest.mark.asyncio
    async def test_place_order_market_params(self) -> None:
        c = self._client()
        captured: dict[str, Any] = {}

        async def fake_request(method: str, endpoint: str, **kw: Any) -> dict[str, Any]:
            captured["method"] = method
            captured["endpoint"] = endpoint
            captured["params"] = kw.get("params")
            captured["signed"] = kw.get("signed")
            return {"orderId": 42, "status": "FILLED"}

        c._request = fake_request  # type: ignore[assignment]
        out = await c.place_order("BTCUSDT", "BUY", 0.001, order_type="MARKET")
        assert out["orderId"] == 42
        assert captured["method"] == "POST"
        assert captured["signed"] is True
        params = captured["params"]
        assert params["symbol"] == "BTCUSDT"
        assert params["side"] == "BUY"
        assert params["type"] == "MARKET"
        assert "price" not in params
        assert "timeInForce" not in params

    @pytest.mark.asyncio
    async def test_place_order_limit_requires_price(self) -> None:
        c = self._client()
        with pytest.raises(ValueError):
            await c.place_order("BTCUSDT", "BUY", 0.001, order_type="LIMIT")

    @pytest.mark.asyncio
    async def test_place_order_invalid_side(self) -> None:
        c = self._client()
        with pytest.raises(ValueError):
            await c.place_order("BTCUSDT", "HOLD", 0.001)

    @pytest.mark.asyncio
    async def test_cancel_requires_id(self) -> None:
        c = self._client()
        with pytest.raises(ValueError):
            await c.cancel_order("BTCUSDT")


# ─────────────────────────────────────────────────────────────────────
# NativeBalanceSync
# ─────────────────────────────────────────────────────────────────────
class _StubClient:
    """Minimal stand-in for NativeExchangeClient used by BalanceSync tests."""

    def __init__(self, balances: dict[str, float]) -> None:
        self.balances = balances
        self.calls = 0
        self.fail_n: int = 0

    async def get_balance(self) -> dict[str, float]:
        self.calls += 1
        if self.fail_n > 0:
            self.fail_n -= 1
            raise ExchangeClientError("boom")
        return dict(self.balances)


class TestNativeBalanceSync:
    @pytest.mark.asyncio
    async def test_start_primes_cache_and_stops_clean(self) -> None:
        stub = _StubClient({"USDT": 100.0, "BTC": 0.1})
        bs = NativeBalanceSync(stub, poll_interval_sec=0.5)  # type: ignore[arg-type]
        await bs.start()
        assert bs.is_running
        assert bs.get_balance() == {"USDT": 100.0, "BTC": 0.1}
        assert bs.get_asset("USDT") == 100.0
        assert bs.get_asset("DOGE") == 0.0
        assert bs.last_update_ts > 0
        await bs.stop()
        assert not bs.is_running

    @pytest.mark.asyncio
    async def test_sync_callback_invoked(self) -> None:
        stub = _StubClient({"USDT": 1.0})
        seen: list[dict[str, float]] = []
        bs = NativeBalanceSync(
            stub,  # type: ignore[arg-type]
            poll_interval_sec=10.0,
            on_update=lambda b: seen.append(b),
        )
        await bs.start()
        await bs.stop()
        assert seen and seen[0] == {"USDT": 1.0}

    @pytest.mark.asyncio
    async def test_async_callback_invoked(self) -> None:
        stub = _StubClient({"USDT": 2.0})
        seen: list[dict[str, float]] = []

        async def cb(b: dict[str, float]) -> None:
            seen.append(b)

        bs = NativeBalanceSync(
            stub,
            poll_interval_sec=10.0,
            on_update=cb,  # type: ignore[arg-type]
        )
        await bs.start()
        await bs.stop()
        assert seen == [{"USDT": 2.0}]

    @pytest.mark.asyncio
    async def test_loop_polls_multiple_times(self) -> None:
        stub = _StubClient({"USDT": 1.0})
        bs = NativeBalanceSync(stub, poll_interval_sec=0.5)  # type: ignore[arg-type]
        await bs.start()
        # prime = 1 call. Wait long enough for ≥1 more.
        await asyncio.sleep(0.7)
        await bs.stop()
        assert stub.calls >= 2


# ─────────────────────────────────────────────────────────────────────
# NativeOrderExecution
# ─────────────────────────────────────────────────────────────────────
class _OrderStubClient:
    """Stand-in for NativeExchangeClient with order-related methods."""

    def __init__(self) -> None:
        self.placed: list[dict[str, Any]] = []
        self.canceled: list[dict[str, Any]] = []
        self.next_response: dict[str, Any] = {"orderId": 1, "status": "NEW"}
        self.next_get: dict[str, Any] = {
            "orderId": 1,
            "status": "FILLED",
            "side": "BUY",
            "origQty": "0.001",
            "type": "MARKET",
            "price": "0.00",
        }
        self.raise_on_place: Optional[Exception] = None

    async def place_order(self, **kwargs: Any) -> dict[str, Any]:
        if self.raise_on_place is not None:
            raise self.raise_on_place
        self.placed.append(kwargs)
        return dict(self.next_response)

    async def cancel_order(self, symbol: str, **kwargs: Any) -> dict[str, Any]:
        self.canceled.append({"symbol": symbol, **kwargs})
        return {"status": "CANCELED"}

    async def get_order(self, symbol: str, **kwargs: Any) -> dict[str, Any]:
        return dict(self.next_get)


class TestNativeOrderExecution:
    @pytest.mark.asyncio
    async def test_market_buy_returns_success_result(self) -> None:
        stub = _OrderStubClient()
        ex = NativeOrderExecution(stub)  # type: ignore[arg-type]
        res = await ex.place_market_buy("BTCUSDT", 0.001)
        assert isinstance(res, OrderResult)
        assert res.success is True
        assert res.symbol == "BTCUSDT"
        assert res.side == "BUY"
        assert res.order_type == "MARKET"
        assert res.exchange_order_id == 1
        assert stub.placed[0]["symbol"] == "BTCUSDT"
        assert stub.placed[0]["side"] == "BUY"

    @pytest.mark.asyncio
    async def test_limit_sell_passes_price(self) -> None:
        stub = _OrderStubClient()
        ex = NativeOrderExecution(stub)  # type: ignore[arg-type]
        res = await ex.place_limit_sell("BTCUSDT", 0.001, 60000.0)
        assert res.success is True
        assert stub.placed[0]["order_type"] == "LIMIT"
        assert stub.placed[0]["price"] == 60000.0

    @pytest.mark.asyncio
    async def test_place_failure_returns_unsuccessful_result(self) -> None:
        stub = _OrderStubClient()
        stub.raise_on_place = ExchangeClientError("rejected")
        ex = NativeOrderExecution(stub)  # type: ignore[arg-type]
        res = await ex.place_market_buy("BTCUSDT", 0.001)
        assert res.success is False
        assert res.status == "ERROR"
        assert "rejected" in (res.error or "")
        # Not tracked as open
        assert ex.open_orders() == []

    @pytest.mark.asyncio
    async def test_open_order_is_tracked_until_terminal(self) -> None:
        stub = _OrderStubClient()
        stub.next_response = {"orderId": 7, "status": "NEW"}
        ex = NativeOrderExecution(stub)  # type: ignore[arg-type]
        res = await ex.place_limit_buy("BTCUSDT", 0.001, 50000.0)
        assert len(ex.open_orders()) == 1

        # Refresh → exchange now reports FILLED → drop from open list.
        stub.next_get = {
            "orderId": 7,
            "status": "FILLED",
            "side": "BUY",
            "origQty": "0.001",
            "type": "LIMIT",
            "price": "50000.00",
        }
        refreshed = await ex.refresh_status("BTCUSDT", res.client_order_id)
        assert refreshed.status == "FILLED"
        assert refreshed.price == 50000.0
        assert ex.open_orders() == []

    @pytest.mark.asyncio
    async def test_cancel_removes_from_open_orders(self) -> None:
        stub = _OrderStubClient()
        stub.next_response = {"orderId": 9, "status": "NEW"}
        ex = NativeOrderExecution(stub)  # type: ignore[arg-type]
        res = await ex.place_limit_buy("BTCUSDT", 0.001, 50000.0)
        ok = await ex.cancel("BTCUSDT", res.client_order_id)
        assert ok is True
        assert ex.open_orders() == []
        assert stub.canceled[0]["client_order_id"] == res.client_order_id
