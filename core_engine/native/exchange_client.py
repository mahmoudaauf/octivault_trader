"""
Native L1: Exchange Client (Phase 8.2.2)

Lightweight REST API wrapper for Binance Spot. Replaces ~800-line legacy
ExchangeClient with a focused ~400-line implementation.

Design choices
--------------
* REST only (no WebSocket). Polling is good enough for our cycle cadence.
* aiohttp for async I/O. Single shared session.
* HMAC-SHA256 signing per Binance spec.
* All retries delegated to NativeRetryManager (L0).
* Pure data in / pure data out — no internal cache, no callbacks.

Public surface (8 methods)
--------------------------
* get_server_time()         → int (ms)
* get_exchange_info()       → dict
* get_balance()             → dict[asset, free]
* get_account()             → dict (raw account payload)
* get_prices(symbols?)      → dict[symbol, price]
* get_klines(sym, iv, lim)  → list[list]
* place_order(...)          → dict
* cancel_order(sym, oid)    → dict
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import time
from typing import Any, Optional
from urllib.parse import urlencode

try:
    import aiohttp  # type: ignore
except ImportError:  # graceful fallback for environments without aiohttp
    aiohttp = None  # type: ignore

from core_engine.native.retry_manager import RETRY_STANDARD, NativeRetryManager

logger = logging.getLogger(__name__)


class ExchangeClientError(Exception):
    """Raised when an exchange call fails after retries."""


class NativeExchangeClient:
    """
    Minimal Binance Spot REST client.

    Thread/async-safety: instantiate once per process; all methods are
    coroutines. The underlying aiohttp.ClientSession is created lazily on
    first use and closed via :py:meth:`close`.
    """

    DEFAULT_BASE_URL = "https://api.binance.com"
    DEFAULT_TESTNET_URL = "https://testnet.binance.vision"

    # Public endpoints (no signature)
    EP_PING = "/api/v3/ping"
    EP_TIME = "/api/v3/time"
    EP_EXCHANGE_INFO = "/api/v3/exchangeInfo"
    EP_TICKER_PRICE = "/api/v3/ticker/price"
    EP_KLINES = "/api/v3/klines"

    # Signed endpoints (HMAC required)
    EP_ACCOUNT = "/api/v3/account"
    EP_ORDER = "/api/v3/order"

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        *,
        base_url: Optional[str] = None,
        testnet: bool = False,
        recv_window_ms: int = 5_000,
        request_timeout_sec: float = 10.0,
        retry: Optional[NativeRetryManager] = None,
    ) -> None:
        self.api_key = api_key
        self.api_secret = api_secret.encode() if api_secret else b""
        self.base_url = base_url or (self.DEFAULT_TESTNET_URL if testnet else self.DEFAULT_BASE_URL)
        self.recv_window_ms = recv_window_ms
        self.request_timeout_sec = request_timeout_sec
        self.retry = retry or RETRY_STANDARD
        self._session: Optional[aiohttp.ClientSession] = None  # type: ignore

    # ──────────────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────────────
    async def _get_session(self) -> aiohttp.ClientSession:  # type: ignore
        if aiohttp is None:
            raise ExchangeClientError(
                "aiohttp is not installed; install it or inject a stub session"
            )
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.request_timeout_sec)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self) -> None:
        """Close the underlying HTTP session. Safe to call multiple times."""
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    async def __aenter__(self) -> NativeExchangeClient:
        await self._get_session()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    # ──────────────────────────────────────────────────────────────────
    # Signing
    # ──────────────────────────────────────────────────────────────────
    def _sign(self, params: dict[str, Any]) -> dict[str, Any]:
        """Return params dict with ``timestamp``, ``recvWindow`` and ``signature``."""
        payload = dict(params)
        payload.setdefault("timestamp", int(time.time() * 1000))
        payload.setdefault("recvWindow", self.recv_window_ms)
        qs = urlencode(payload, doseq=True)
        signature = hmac.new(self.api_secret, qs.encode(), hashlib.sha256).hexdigest()
        payload["signature"] = signature
        return payload

    # ──────────────────────────────────────────────────────────────────
    # Low-level request
    # ──────────────────────────────────────────────────────────────────
    async def _raw_request(
        self,
        method: str,
        endpoint: str,
        *,
        params: Optional[dict[str, Any]] = None,
        signed: bool = False,
    ) -> Any:
        if signed:
            params = self._sign(params or {})

        session = await self._get_session()
        url = f"{self.base_url}{endpoint}"
        headers = {"X-MBX-APIKEY": self.api_key} if signed or self.api_key else {}

        async with session.request(method, url, params=params, headers=headers) as resp:
            text = await resp.text()
            if resp.status >= 400:
                raise ExchangeClientError(f"{method} {endpoint} → {resp.status}: {text[:200]}")
            try:
                return await resp.json(content_type=None)
            except Exception as e:  # pragma: no cover — defensive
                raise ExchangeClientError(f"non-JSON response from {endpoint}: {text[:200]}") from e

    async def _request(
        self,
        method: str,
        endpoint: str,
        *,
        params: Optional[dict[str, Any]] = None,
        signed: bool = False,
    ) -> Any:
        """Retrying wrapper around :py:meth:`_raw_request`."""
        return await self.retry.call(
            self._raw_request, method, endpoint, params=params, signed=signed
        )

    # ──────────────────────────────────────────────────────────────────
    # Public market data
    # ──────────────────────────────────────────────────────────────────
    async def ping(self) -> bool:
        await self._request("GET", self.EP_PING)
        return True

    async def get_server_time(self) -> int:
        data = await self._request("GET", self.EP_TIME)
        return int(data["serverTime"])

    async def get_exchange_info(self, symbol: Optional[str] = None) -> dict[str, Any]:
        params = {"symbol": symbol} if symbol else None
        return await self._request("GET", self.EP_EXCHANGE_INFO, params=params)

    async def get_prices(self, symbols: Optional[list[str]] = None) -> dict[str, float]:
        """
        Return ``{symbol: price}``. If ``symbols`` is ``None`` returns all symbols.
        """
        params: Optional[dict[str, Any]] = None
        if symbols and len(symbols) == 1:
            params = {"symbol": symbols[0]}
        data = await self._request("GET", self.EP_TICKER_PRICE, params=params)

        if isinstance(data, dict):  # single-symbol response
            data = [data]

        out: dict[str, float] = {}
        wanted = set(symbols) if symbols else None
        for row in data:
            sym = row.get("symbol")
            if sym is None:
                continue
            if wanted is not None and sym not in wanted:
                continue
            try:
                out[sym] = float(row["price"])
            except (KeyError, TypeError, ValueError):
                continue
        return out

    async def get_klines(
        self, symbol: str, interval: str = "1m", limit: int = 100
    ) -> list[list[Any]]:
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        return await self._request("GET", self.EP_KLINES, params=params)

    # ──────────────────────────────────────────────────────────────────
    # Signed account / trading
    # ──────────────────────────────────────────────────────────────────
    async def get_account(self) -> dict[str, Any]:
        return await self._request("GET", self.EP_ACCOUNT, signed=True)

    async def get_balance(self) -> dict[str, float]:
        """
        Return free balances as ``{asset: free_amount}``.

        Zero balances are filtered out to keep the dict small.
        """
        account = await self.get_account()
        balances: dict[str, float] = {}
        for entry in account.get("balances", []):
            try:
                free = float(entry.get("free", 0.0))
            except (TypeError, ValueError):
                free = 0.0
            if free > 0:
                balances[entry["asset"]] = free
        return balances

    async def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        *,
        order_type: str = "MARKET",
        price: Optional[float] = None,
        time_in_force: str = "GTC",
        client_order_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Place a spot order. ``side`` ∈ {BUY, SELL}; ``order_type`` ∈
        {MARKET, LIMIT}. For LIMIT orders ``price`` is required.
        """
        side_u = side.upper()
        if side_u not in ("BUY", "SELL"):
            raise ValueError(f"invalid side: {side!r}")
        type_u = order_type.upper()

        params: dict[str, Any] = {
            "symbol": symbol,
            "side": side_u,
            "type": type_u,
            "quantity": f"{quantity:.8f}".rstrip("0").rstrip("."),
        }
        if type_u == "LIMIT":
            if price is None:
                raise ValueError("price is required for LIMIT orders")
            params["price"] = f"{price:.8f}".rstrip("0").rstrip(".")
            params["timeInForce"] = time_in_force
        if client_order_id:
            params["newClientOrderId"] = client_order_id

        return await self._request("POST", self.EP_ORDER, params=params, signed=True)

    async def cancel_order(
        self,
        symbol: str,
        *,
        order_id: Optional[int] = None,
        client_order_id: Optional[str] = None,
    ) -> dict[str, Any]:
        if order_id is None and client_order_id is None:
            raise ValueError("order_id or client_order_id is required")
        params: dict[str, Any] = {"symbol": symbol}
        if order_id is not None:
            params["orderId"] = order_id
        if client_order_id is not None:
            params["origClientOrderId"] = client_order_id
        return await self._request("DELETE", self.EP_ORDER, params=params, signed=True)

    async def get_order(
        self,
        symbol: str,
        *,
        order_id: Optional[int] = None,
        client_order_id: Optional[str] = None,
    ) -> dict[str, Any]:
        if order_id is None and client_order_id is None:
            raise ValueError("order_id or client_order_id is required")
        params: dict[str, Any] = {"symbol": symbol}
        if order_id is not None:
            params["orderId"] = order_id
        if client_order_id is not None:
            params["origClientOrderId"] = client_order_id
        return await self._request("GET", self.EP_ORDER, params=params, signed=True)
