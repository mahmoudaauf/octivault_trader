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

Public surface (9 methods)
--------------------------
* get_server_time()         → int (ms)
* get_exchange_info()       → dict
* get_balance()             → dict[asset, free]
* get_account()             → dict (raw account payload)
* get_prices(symbols?)      → dict[symbol, price]
* get_klines(sym, iv, lim)  → list[list]
* place_order(...)          → dict
* cancel_order(sym, oid)    → dict
* get_my_trades(sym, limit) → list[dict] (executed trades for one symbol)
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import re
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
    EP_MY_TRADES = "/api/v3/myTrades"

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
        request_budget_window_sec: float = 60.0,
        request_budget_soft_limit: int = 1200,
        signed_request_cooldown_sec: float = 15.0,
    ) -> None:
        self.api_key = api_key
        self.api_secret = api_secret.encode() if api_secret else b""
        self.base_url = base_url or (self.DEFAULT_TESTNET_URL if testnet else self.DEFAULT_BASE_URL)
        self.recv_window_ms = recv_window_ms
        self.request_timeout_sec = request_timeout_sec
        self.retry = retry or RETRY_STANDARD
        self._session: Optional[aiohttp.ClientSession] = None  # type: ignore
        self._throttled_until_ts: float = 0.0
        self._last_error: str = ""
        self._request_budget_window_sec = max(1.0, float(request_budget_window_sec))
        self._request_budget_soft_limit = max(1, int(request_budget_soft_limit))
        self._signed_request_cooldown_sec = max(0.0, float(signed_request_cooldown_sec))
        self._request_events: list[tuple[float, int]] = []
        self._last_signed_request_ts: float = 0.0
        # Paper-mode order ledger: place_order() results, keyed both ways so
        # get_order()/cancel_order() can return the REAL simulated fill instead
        # of a data-less generic stub (a data-less stub previously corrupted
        # maker-first-buy's refresh_status() poll — see _simulate_place_order).
        self._paper_orders_by_id: dict[str, dict[str, Any]] = {}
        self._paper_orders_by_coid: dict[str, dict[str, Any]] = {}

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
                self._record_error(resp.status, text)
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
        if self.is_throttled():
            raise ExchangeClientError(
                f"exchange throttled until {self._throttled_until_ts:.0f}: {self._last_error}"
            )
        self._enforce_local_budget(endpoint=endpoint, params=params, signed=signed)
        weight = self._estimate_request_weight(endpoint=endpoint, params=params, signed=signed)
        out = await self.retry.call(
            self._raw_request, method, endpoint, params=params, signed=signed
        )
        self._record_request(weight=weight, signed=signed)
        return out

    def is_throttled(self) -> bool:
        return self._throttled_until_ts > time.time()

    def throttled_until_ts(self) -> float:
        return self._throttled_until_ts

    def last_error(self) -> str:
        return self._last_error

    def restore_throttle_state(self, *, until_ts: float = 0.0, reason: str = "") -> None:
        """
        Apply a previously persisted throttle window to this live client.

        This lets startup honor a known ban/cooldown period before any new
        signed or market-data REST calls are attempted.
        """
        until_ts = max(0.0, float(until_ts or 0.0))
        if until_ts <= time.time():
            return
        self._throttled_until_ts = max(self._throttled_until_ts, until_ts)
        if reason:
            self._last_error = str(reason)

    def _record_error(self, status: int, text: str) -> None:
        self._last_error = f"{status}: {text[:200]}"
        if status not in (418, 429):
            return
        until_ts = time.time() + (60.0 if status == 429 else 300.0)
        match = re.search(r"until\s+(\d{10,13})", text)
        if match:
            raw = int(match.group(1))
            until_ts = raw / 1000.0 if raw > 10_000_000_000 else float(raw)
        self._throttled_until_ts = max(self._throttled_until_ts, until_ts)

    def request_budget_snapshot(self) -> dict[str, float]:
        now = time.time()
        self._prune_request_events(now)
        used = sum(weight for _ts, weight in self._request_events)
        return {
            "window_sec": self._request_budget_window_sec,
            "soft_limit": float(self._request_budget_soft_limit),
            "used_weight": float(used),
            "remaining_weight": float(max(0, self._request_budget_soft_limit - used)),
            "last_signed_request_ts": float(self._last_signed_request_ts),
        }

    def _enforce_local_budget(
        self,
        *,
        endpoint: str,
        params: Optional[dict[str, Any]],
        signed: bool,
    ) -> None:
        now = time.time()
        self._prune_request_events(now)
        if signed and self._signed_request_cooldown_sec > 0:
            since_last_signed = now - self._last_signed_request_ts
            if 0 < since_last_signed < self._signed_request_cooldown_sec:
                remaining = self._signed_request_cooldown_sec - since_last_signed
                self._last_error = f"local signed-request cooldown active for {remaining:.1f}s"
                raise ExchangeClientError(self._last_error)

        weight = self._estimate_request_weight(endpoint=endpoint, params=params, signed=signed)
        used = sum(w for _ts, w in self._request_events)
        if used + weight > self._request_budget_soft_limit:
            cooldown = min(self._request_budget_window_sec, 60.0)
            self._throttled_until_ts = max(self._throttled_until_ts, now + cooldown)
            self._last_error = (
                f"local request budget exceeded: used={used} next={weight} "
                f"limit={self._request_budget_soft_limit}"
            )
            raise ExchangeClientError(self._last_error)

    def _record_request(self, *, weight: int, signed: bool) -> None:
        now = time.time()
        self._request_events.append((now, max(1, int(weight))))
        self._prune_request_events(now)
        if signed:
            self._last_signed_request_ts = now

    def _prune_request_events(self, now: float) -> None:
        cutoff = now - self._request_budget_window_sec
        self._request_events = [(ts, w) for ts, w in self._request_events if ts >= cutoff]

    def _estimate_request_weight(
        self,
        *,
        endpoint: str,
        params: Optional[dict[str, Any]],
        signed: bool,
    ) -> int:
        del signed
        if endpoint == self.EP_ACCOUNT:
            return 20
        if endpoint == self.EP_EXCHANGE_INFO:
            return 10
        if endpoint == self.EP_KLINES:
            return 2
        if endpoint == self.EP_TICKER_PRICE:
            return 1 if params and params.get("symbol") else 2
        if endpoint == self.EP_ORDER:
            return 4
        if endpoint == self.EP_TIME:
            return 1
        return 1

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

    async def get_ohlcv(
        self,
        symbol: str,
        interval: str = "5m",
        timeframe: str = None,
        tf: str = None,
        limit: int = 1000,
        end_time: int = None,
        endTime: int = None,
    ) -> list[list[Any]]:
        """Alias for get_klines — supports kwargs used by MLForecaster._fetch_exchange_ohlcv_batch."""
        tf_resolved = interval or timeframe or tf or "5m"
        end_ms = end_time or endTime
        params: dict[str, Any] = {"symbol": symbol, "interval": tf_resolved, "limit": min(limit, 1000)}
        if end_ms is not None:
            params["endTime"] = int(end_ms)
        return await self._request("GET", self.EP_KLINES, params=params)

    # ──────────────────────────────────────────────────────────────────
    # Signed account / trading
    # ──────────────────────────────────────────────────────────────────
    def _is_paper(self) -> bool:
        """True when this client is running with sentinel/absent credentials.

        Every signed endpoint that can mutate exchange state or read real trade
        history MUST check this before touching the network — sentinel keys are
        not valid Binance credentials, so without this guard a paper-mode run
        would still fire a real signed HTTP request at the production API.
        """
        return self.api_key == "paper_key" or not self.api_secret

    async def get_account(self) -> dict[str, Any]:
        # Paper mode: return simulated account
        if self._is_paper():
            return {
                "balances": [{"asset": "USDT", "free": "1000.0", "locked": "0.0"}],
                "canTrade": True,
                "canDeposit": True,
                "canWithdraw": True,
            }
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
                locked = float(entry.get("locked", 0.0))
            except (TypeError, ValueError):
                free = locked = 0.0
            total = free + locked
            if total > 0:
                balances[entry["asset"]] = total
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

        if self._is_paper():
            return await self._simulate_place_order(
                symbol, side_u, quantity, order_type=type_u, price=price,
                client_order_id=client_order_id,
            )

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

    async def _simulate_place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        *,
        order_type: str,
        price: Optional[float],
        client_order_id: Optional[str],
    ) -> dict[str, Any]:
        """Fabricate an instantly-filled order response for paper mode.

        Never touches the network for the order itself — ``get_prices`` is a
        public, unsigned, read-only endpoint so it's safe to call for a
        realistic fill price. Response shape matches what order_execution.py /
        executor.py / safety_order_manager.py already read off a real Binance
        place_order response (orderId, status, executedQty, cummulativeQuoteQty,
        price, fills[].{price,qty,commission,commissionAsset}).
        """
        fill_price = price
        if fill_price is None:
            prices = await self.get_prices([symbol])
            fill_price = prices.get(symbol, 0.0)
        quote_qty = quantity * fill_price
        order_id = int(time.time() * 1000)
        qty_s = f"{quantity:.8f}"
        price_s = f"{fill_price:.8f}"
        coid = client_order_id or f"paper-{order_id}"
        resp = {
            "symbol": symbol,
            "orderId": order_id,
            "clientOrderId": coid,
            "transactTime": int(time.time() * 1000),
            "price": price_s,
            "origQty": qty_s,
            "executedQty": qty_s,
            "cummulativeQuoteQty": f"{quote_qty:.8f}",
            "status": "FILLED",
            "timeInForce": "GTC",
            "type": order_type,
            "side": side,
            "fills": [
                {"price": price_s, "qty": qty_s, "commission": "0.0", "commissionAsset": "USDT"}
            ],
        }
        # Register so a later get_order()/cancel_order() poll (e.g. executor.py's
        # maker-first-buy refresh_status() after its grace window) returns this
        # exact fill instead of a data-less guess — this order is instantly
        # FILLED, so order_execution.py's _open cache never holds it (only
        # non-terminal orders are cached there), making this ledger the only
        # place a later status poll can find real data.
        self._paper_orders_by_id[str(order_id)] = resp
        self._paper_orders_by_coid[coid] = resp
        return resp

    async def cancel_order(
        self,
        symbol: str,
        *,
        order_id: Optional[int] = None,
        client_order_id: Optional[str] = None,
    ) -> dict[str, Any]:
        if order_id is None and client_order_id is None:
            raise ValueError("order_id or client_order_id is required")
        if self._is_paper():
            stored = (
                self._paper_orders_by_id.get(str(order_id)) if order_id is not None else None
            ) or (
                self._paper_orders_by_coid.get(client_order_id) if client_order_id is not None else None
            )
            if stored is not None:
                stored = dict(stored)
                stored["status"] = "CANCELED"
                return stored
            return {
                "symbol": symbol,
                "orderId": order_id,
                "clientOrderId": client_order_id,
                "status": "CANCELED",
            }
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
        if self._is_paper():
            stored = (
                self._paper_orders_by_id.get(str(order_id)) if order_id is not None else None
            ) or (
                self._paper_orders_by_coid.get(client_order_id) if client_order_id is not None else None
            )
            if stored is not None:
                return dict(stored)
            # Genuinely unknown order (e.g. a lookup for an id never placed this
            # process) — report a terminal-but-NOT-filled status. Claiming FILLED
            # here for an order we have zero data on is exactly the bug this
            # ledger fixes: it silently fed data-less zero qty/price into
            # executor.py's maker-first-buy fill-price extraction as if a real
            # fill had occurred.
            return {
                "symbol": symbol,
                "orderId": order_id,
                "clientOrderId": client_order_id,
                "side": "BUY",
                "origQty": "0.0",
                "price": "0.0",
                "status": "REJECTED",
            }
        params: dict[str, Any] = {"symbol": symbol}
        if order_id is not None:
            params["orderId"] = order_id
        if client_order_id is not None:
            params["origClientOrderId"] = client_order_id
        return await self._request("GET", self.EP_ORDER, params=params, signed=True)

    async def get_my_trades(
        self,
        symbol: str,
        *,
        limit: int = 500,
        from_id: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Fetch this account's executed trades for ``symbol`` (GET /api/v3/myTrades).

        Binance requires a single ``symbol`` per call — there is no
        all-symbols trade-history endpoint. Results are ordered oldest-first
        by trade id. Pass ``from_id`` (a trade id, exclusive lower bound minus
        one — i.e. pass ``last_seen_id + 1``) to page through accounts with
        more than ``limit`` historical trades; see position_hydration_engine's
        pagination loop.
        """
        if self._is_paper():
            # Sentinel keys have no real trade history — always report empty
            # rather than guessing, consistent with position_hydration_engine's
            # existing "assume fresh account" fallback.
            return []
        params: dict[str, Any] = {"symbol": symbol, "limit": limit}
        if from_id is not None:
            params["fromId"] = int(from_id)
        result = await self._request("GET", self.EP_MY_TRADES, params=params, signed=True)
        return result if isinstance(result, list) else []
