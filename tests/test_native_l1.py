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
import time
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
from core_engine.native.symbol_discovery import NativeSymbolDiscovery


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
    async def test_get_book_ticker_parses_bid_ask(self) -> None:
        c = self._client()
        c._request = AsyncMock(  # type: ignore[assignment]
            return_value={
                "symbol": "BTCUSDT",
                "bidPrice": "49999.5", "bidQty": "1.2",
                "askPrice": "50000.5", "askQty": "0.8",
            }
        )
        book = await c.get_book_ticker("BTCUSDT")
        assert book == {"bid": 49999.5, "bid_qty": 1.2, "ask": 50000.5, "ask_qty": 0.8}
        c._request.assert_awaited_once_with(
            "GET", c.EP_BOOK_TICKER, params={"symbol": "BTCUSDT"}
        )

    @pytest.mark.asyncio
    async def test_get_book_ticker_returns_none_on_request_failure(self) -> None:
        c = self._client()
        c._request = AsyncMock(side_effect=ExchangeClientError("boom"))  # type: ignore[assignment]
        assert await c.get_book_ticker("BTCUSDT") is None

    @pytest.mark.asyncio
    async def test_get_book_ticker_returns_none_on_malformed_response(self) -> None:
        c = self._client()
        c._request = AsyncMock(return_value={"symbol": "BTCUSDT"})  # type: ignore[assignment]
        assert await c.get_book_ticker("BTCUSDT") is None

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

    def _paper_client(self) -> NativeExchangeClient:
        retry = NativeRetryManager(max_attempts=1, base_delay_sec=0.0, jitter=False)
        return NativeExchangeClient("paper_key", "paper_secret", retry=retry)

    @pytest.mark.asyncio
    async def test_place_order_paper_mode_never_calls_request(self) -> None:
        c = self._paper_client()
        c._request = AsyncMock(side_effect=AssertionError("paper mode must not call _request"))  # type: ignore[assignment]
        c.get_prices = AsyncMock(return_value={"BTCUSDT": 50000.0})  # type: ignore[assignment]
        out = await c.place_order("BTCUSDT", "BUY", 0.001, order_type="MARKET")
        assert out["status"] == "FILLED"
        assert out["symbol"] == "BTCUSDT"
        assert float(out["price"]) == 50000.0
        assert float(out["executedQty"]) == 0.001
        assert out["fills"][0]["price"] == out["price"]
        c._request.assert_not_called()

    @pytest.mark.asyncio
    async def test_place_order_paper_mode_no_api_secret_also_gated(self) -> None:
        retry = NativeRetryManager(max_attempts=1, base_delay_sec=0.0, jitter=False)
        c = NativeExchangeClient("anything", "", retry=retry)
        c._request = AsyncMock(side_effect=AssertionError("must not call _request"))  # type: ignore[assignment]
        c.get_prices = AsyncMock(return_value={"ETHUSDT": 3000.0})  # type: ignore[assignment]
        out = await c.place_order("ETHUSDT", "SELL", 1.0, order_type="MARKET")
        assert out["status"] == "FILLED"
        c._request.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_order_paper_mode_returns_real_placed_fill_not_data_less_stub(self) -> None:
        """Regression: get_order() after a paper place_order() must return the
        REAL fill (price/qty/status), not a generic zero-value stub — a
        data-less stub previously corrupted executor.py's maker-first-buy
        refresh_status() poll, silently zeroing out entry price/qty/fees."""
        c = self._paper_client()
        c._request = AsyncMock(side_effect=AssertionError("paper mode must not call _request"))  # type: ignore[assignment]
        c.get_prices = AsyncMock(return_value={"BTCUSDT": 100.0})  # type: ignore[assignment]
        placed = await c.place_order("BTCUSDT", "BUY", 5.0, order_type="LIMIT", price=99.99)

        fetched_by_id = await c.get_order("BTCUSDT", order_id=placed["orderId"])
        assert fetched_by_id["price"] == placed["price"]
        assert fetched_by_id["origQty"] == placed["origQty"]
        assert fetched_by_id["status"] == "FILLED"

        fetched_by_coid = await c.get_order("BTCUSDT", client_order_id=placed["clientOrderId"])
        assert fetched_by_coid["price"] == placed["price"]

    @pytest.mark.asyncio
    async def test_get_order_paper_mode_unknown_order_reports_not_filled(self) -> None:
        """An order id never placed this process must NOT be claimed FILLED —
        that false-positive is exactly what caused the original bug."""
        c = self._paper_client()
        out = await c.get_order("BTCUSDT", order_id=999999999)
        assert out["status"] != "FILLED"

    @pytest.mark.asyncio
    async def test_cancel_order_paper_mode_returns_real_placed_order_as_canceled(self) -> None:
        c = self._paper_client()
        c.get_prices = AsyncMock(return_value={"BTCUSDT": 100.0})  # type: ignore[assignment]
        placed = await c.place_order("BTCUSDT", "BUY", 5.0, order_type="LIMIT", price=99.99)
        out = await c.cancel_order("BTCUSDT", order_id=placed["orderId"])
        assert out["status"] == "CANCELED"
        assert out["price"] == placed["price"]  # real data preserved, not wiped

    @pytest.mark.asyncio
    async def test_cancel_order_paper_mode_never_calls_request(self) -> None:
        c = self._paper_client()
        c._request = AsyncMock(side_effect=AssertionError("paper mode must not call _request"))  # type: ignore[assignment]
        out = await c.cancel_order("BTCUSDT", order_id=7)
        assert out["status"] == "CANCELED"
        c._request.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_order_paper_mode_never_calls_request(self) -> None:
        c = self._paper_client()
        c._request = AsyncMock(side_effect=AssertionError("paper mode must not call _request"))  # type: ignore[assignment]
        # order_id=7 was never placed via place_order() this process, so the
        # paper ledger has no record of it — status must NOT be reported as
        # FILLED (see test_get_order_paper_mode_unknown_order_reports_not_filled
        # for why: a false FILLED here previously corrupted maker-first-buy's
        # fill-price extraction with zero-value data).
        out = await c.get_order("BTCUSDT", order_id=7)
        assert out["status"] != "FILLED"
        c._request.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_my_trades_paper_mode_returns_empty_never_calls_request(self) -> None:
        c = self._paper_client()
        c._request = AsyncMock(side_effect=AssertionError("paper mode must not call _request"))  # type: ignore[assignment]
        out = await c.get_my_trades("BTCUSDT")
        assert out == []
        c._request.assert_not_called()

    @pytest.mark.asyncio
    async def test_place_order_live_mode_still_calls_request(self) -> None:
        # Regression guard: real KEY/SECRET must still hit _request as before.
        c = self._client()
        c._request = AsyncMock(return_value={"orderId": 1, "status": "FILLED"})  # type: ignore[assignment]
        out = await c.place_order("BTCUSDT", "BUY", 0.001, order_type="MARKET")
        assert out["orderId"] == 1
        c._request.assert_called_once()

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

    def test_record_error_sets_throttle_window(self) -> None:
        c = self._client()
        until_ms = int((time.time() + 120.0) * 1000)
        c._record_error(418, f"Way too much request weight used; IP banned until {until_ms}.")
        assert c.is_throttled() is True
        assert c.throttled_until_ts() > 0
        assert "418" in c.last_error()

    def test_restore_throttle_state_rehydrates_client(self) -> None:
        c = self._client()
        until_ts = time.time() + 120.0
        c.restore_throttle_state(until_ts=until_ts, reason="persisted 418")
        assert c.is_throttled() is True
        assert c.throttled_until_ts() >= until_ts - 0.1
        assert c.last_error() == "persisted 418"

    @pytest.mark.asyncio
    async def test_request_short_circuits_while_throttled(self) -> None:
        c = self._client()
        c._record_error(429, "too many requests")
        with pytest.raises(ExchangeClientError, match="exchange throttled"):
            await c._request("GET", c.EP_TIME)

    def test_local_budget_snapshot_updates_after_request_record(self) -> None:
        c = self._client()
        c._record_request(weight=5, signed=False)
        snap = c.request_budget_snapshot()
        assert snap["used_weight"] >= 5
        assert snap["remaining_weight"] <= snap["soft_limit"]

    @pytest.mark.asyncio
    async def test_local_signed_request_cooldown_blocks_repeat_account_call(self) -> None:
        c = NativeExchangeClient(
            "KEY",
            "SECRET",
            retry=NativeRetryManager(max_attempts=1, base_delay_sec=0.0, jitter=False),
            signed_request_cooldown_sec=30.0,
        )
        c._record_request(weight=20, signed=True)
        with pytest.raises(ExchangeClientError, match="signed-request cooldown"):
            await c._request("GET", c.EP_ACCOUNT, signed=True)

    @pytest.mark.asyncio
    async def test_local_budget_blocks_request_before_network(self) -> None:
        c = NativeExchangeClient(
            "KEY",
            "SECRET",
            retry=NativeRetryManager(max_attempts=1, base_delay_sec=0.0, jitter=False),
            request_budget_soft_limit=3,
        )
        c._record_request(weight=2, signed=False)
        with pytest.raises(ExchangeClientError, match="local request budget exceeded"):
            await c._request("GET", c.EP_TICKER_PRICE)


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
        bs = NativeBalanceSync(stub, poll_interval_sec=0.5, min_refresh_interval_sec=0.0)  # type: ignore[arg-type]
        await bs.start()
        # prime = 1 call. Wait long enough for ≥1 more.
        await asyncio.sleep(0.7)
        await bs.stop()
        assert stub.calls >= 2

    @pytest.mark.asyncio
    async def test_min_refresh_interval_defers_extra_fetches(self) -> None:
        stub = _StubClient({"USDT": 1.0})
        bs = NativeBalanceSync(
            stub,  # type: ignore[arg-type]
            poll_interval_sec=0.5,
            min_refresh_interval_sec=5.0,
        )
        await bs.start()
        await asyncio.sleep(0.7)
        await bs.stop()
        assert stub.calls == 1


class _DiscoveryClient:
    def __init__(self, balances: list[dict[str, float]]) -> None:
        self._balances = list(balances)
        self.calls = 0

    def is_throttled(self) -> bool:
        return False

    async def get_balance(self) -> dict[str, float]:
        self.calls += 1
        if self._balances:
            return self._balances.pop(0)
        return {}


class TestNativeSymbolDiscovery:
    @pytest.mark.asyncio
    async def test_discovery_uses_cache_between_scans(self) -> None:
        client = _DiscoveryClient([{"BTC": 0.1, "USDT": 50.0}])
        discovery = NativeSymbolDiscovery(
            client,
            min_scan_interval_sec=600.0,
            empty_scan_retry_sec=300.0,
        )
        first = await discovery.discover()
        second = await discovery.discover()
        assert first == ["BTCUSDT"]
        assert second == ["BTCUSDT"]
        assert client.calls == 1

    @pytest.mark.asyncio
    async def test_discovery_defers_retry_after_empty_scan(self) -> None:
        client = _DiscoveryClient([{}, {"ETH": 1.0}])
        discovery = NativeSymbolDiscovery(
            client,
            min_scan_interval_sec=600.0,
            empty_scan_retry_sec=600.0,
        )
        first = await discovery.discover()
        second = await discovery.discover()
        assert first == []
        assert second == []
        assert client.calls == 1

    @pytest.mark.asyncio
    async def test_discovery_uses_shared_state_symbols_when_throttled(self) -> None:
        client = _DiscoveryClient([{"BTC": 0.1, "USDT": 50.0}])
        state = type(
            "State",
            (),
            {
                "exchange_throttle_until_ts": time.time() + 60.0,
                "accepted_symbols": {"ETHUSDT"},
                "positions": {"SOLUSDT": object()},
                "balance": {"XRP": 10.0, "USDT": 1.0},
            },
        )()
        discovery = NativeSymbolDiscovery(client, shared_state=state)
        symbols = await discovery.discover()
        assert symbols == ["ETHUSDT", "SOLUSDT", "XRPUSDT"]
        assert client.calls == 0


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
