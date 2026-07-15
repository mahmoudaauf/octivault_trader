"""
Native L1: Futures Exchange Client — funding-rate carry strategy support.

Sibling to NativeExchangeClient (exchange_client.py), not a shared base class.
The endpoint surface, response shapes, and paper-simulation semantics diverge
enough (funding rates, position risk, leverage, mark price vs. last-traded
price) that forcing inheritance would mostly produce `if futures:` branches
in already-working, live spot-client code. Duplicating the small amount of
shared plumbing (signing, request/retry wrapper) is deliberately cheaper and
safer than sharing it, given this is a new, separately-risk-profiled surface
touching a different (perpetual-futures) account than the live spot client.

Design choices (mirrors exchange_client.py exactly where the concepts
overlap):
* REST only. aiohttp for async I/O. Single shared session.
* HMAC-SHA256 signing — identical scheme to spot, same as Binance uses
  account-wide (not endpoint-specific).
* All retries delegated to NativeRetryManager (L0) — reuses the exact same
  `self.retry.call(self._raw_request, ...)` pattern as exchange_client.py.
* Pure data in / pure data out — no internal cache, no callbacks.

Public surface (funding-carry strategy needs, per carry_paper_trader.py):
* futures_exchange_info()                  -> dict
* futures_ticker(symbol?)                  -> list[dict] | dict
* futures_mark_price(symbol?)              -> list[dict] | dict (incl. lastFundingRate)
* futures_funding_rate(symbol, start, lim) -> list[dict]
* futures_position_information(symbol?)    -> list[dict]
* futures_change_leverage(symbol, lev)     -> dict
* futures_create_order(...)                -> dict
* futures_balance()                        -> dict[asset, float]  (for NAV reconciliation)

Explicitly NOT in this class: no spot-margin borrow/repay endpoints
(negative-funding carry stays out of scope — v1 is POSITIVE_ONLY, matching
carry_paper_trader.py's own restriction).
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


class FuturesExchangeClientError(Exception):
    """Raised when a futures exchange call fails after retries."""


class NativeFuturesExchangeClient:
    """
    Minimal Binance USDT-M Futures REST client, for the funding-rate carry
    strategy's short-perp leg.

    Thread/async-safety: instantiate once per process; all methods are
    coroutines. The underlying aiohttp.ClientSession is created lazily on
    first use and closed via :py:meth:`close`.
    """

    DEFAULT_BASE_URL = "https://fapi.binance.com"
    DEFAULT_TESTNET_URL = "https://testnet.binancefuture.com"

    # Public endpoints (no signature)
    EP_EXCHANGE_INFO = "/fapi/v1/exchangeInfo"
    EP_TICKER_24HR = "/fapi/v1/ticker/24hr"
    EP_PREMIUM_INDEX = "/fapi/v1/premiumIndex"   # mark price + lastFundingRate
    EP_FUNDING_RATE = "/fapi/v1/fundingRate"     # settled funding-rate history

    # Signed endpoints (HMAC required)
    EP_POSITION_RISK = "/fapi/v2/positionRisk"
    EP_LEVERAGE = "/fapi/v1/leverage"
    EP_ORDER = "/fapi/v1/order"
    EP_BALANCE = "/fapi/v2/balance"

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
        # Paper-mode order ledger, mirroring exchange_client.py's rationale:
        # a later get_order()-style poll must see the real simulated fill,
        # not a data-less stub.
        self._paper_orders_by_id: dict[str, dict[str, Any]] = {}

    # ──────────────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────────────
    async def _get_session(self) -> aiohttp.ClientSession:  # type: ignore
        if aiohttp is None:
            raise FuturesExchangeClientError(
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

    async def __aenter__(self) -> NativeFuturesExchangeClient:
        await self._get_session()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    # ──────────────────────────────────────────────────────────────────
    # Signing (identical HMAC-SHA256 scheme to spot — same account secret)
    # ──────────────────────────────────────────────────────────────────
    def _sign(self, params: dict[str, Any]) -> dict[str, Any]:
        payload = dict(params)
        payload.setdefault("timestamp", int(time.time() * 1000))
        payload.setdefault("recvWindow", self.recv_window_ms)
        qs = urlencode(payload, doseq=True)
        signature = hmac.new(self.api_secret, qs.encode(), hashlib.sha256).hexdigest()
        payload["signature"] = signature
        return payload

    # ──────────────────────────────────────────────────────────────────
    # Low-level request (mirrors exchange_client.py's _raw_request/_request)
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
                raise FuturesExchangeClientError(f"{method} {endpoint} → {resp.status}: {text[:200]}")
            try:
                return await resp.json(content_type=None)
            except Exception as e:  # pragma: no cover — defensive
                raise FuturesExchangeClientError(f"non-JSON response from {endpoint}: {text[:200]}") from e

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
            raise FuturesExchangeClientError(
                f"futures exchange throttled until {self._throttled_until_ts:.0f}: {self._last_error}"
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
        """Apply a previously persisted throttle window to this live client
        (same rationale as exchange_client.py's identically-named method)."""
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
                raise FuturesExchangeClientError(self._last_error)

        weight = self._estimate_request_weight(endpoint=endpoint, params=params, signed=signed)
        used = sum(w for _ts, w in self._request_events)
        if used + weight > self._request_budget_soft_limit:
            cooldown = min(self._request_budget_window_sec, 60.0)
            self._throttled_until_ts = max(self._throttled_until_ts, now + cooldown)
            self._last_error = (
                f"local request budget exceeded: used={used} next={weight} "
                f"limit={self._request_budget_soft_limit}"
            )
            raise FuturesExchangeClientError(self._last_error)

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
        if endpoint == self.EP_EXCHANGE_INFO:
            return 1
        if endpoint == self.EP_TICKER_24HR:
            return 1 if params and params.get("symbol") else 40
        if endpoint == self.EP_PREMIUM_INDEX:
            return 1 if params and params.get("symbol") else 10
        if endpoint == self.EP_FUNDING_RATE:
            return 1
        if endpoint == self.EP_POSITION_RISK:
            return 5
        if endpoint == self.EP_BALANCE:
            return 5
        if endpoint == self.EP_ORDER:
            return 1
        if endpoint == self.EP_LEVERAGE:
            return 1
        return 1

    # ──────────────────────────────────────────────────────────────────
    # Public market data
    # ──────────────────────────────────────────────────────────────────
    def _is_paper(self) -> bool:
        """True when this client is running with sentinel/absent credentials —
        same guard rationale as exchange_client.py's identically-named method."""
        return self.api_key == "paper_key" or not self.api_secret

    async def futures_exchange_info(self) -> dict[str, Any]:
        return await self._request("GET", self.EP_EXCHANGE_INFO)

    async def futures_ticker(self, symbol: Optional[str] = None) -> Any:
        """24hr ticker stats (incl. quoteVolume — used for the liquidity filter)."""
        params = {"symbol": symbol} if symbol else None
        return await self._request("GET", self.EP_TICKER_24HR, params=params)

    async def futures_mark_price(self, symbol: Optional[str] = None) -> Any:
        """Mark price + lastFundingRate. Without ``symbol`` returns all symbols
        in one call — this is the bulk funding-rate poll carry_paper_trader.py
        uses every cycle via current_funding()."""
        params = {"symbol": symbol} if symbol else None
        return await self._request("GET", self.EP_PREMIUM_INDEX, params=params)

    async def futures_funding_rate(
        self, symbol: str, *, start_time: Optional[int] = None, limit: int = 1000
    ) -> list[dict[str, Any]]:
        """Settled funding-rate history for one symbol — the real-cash-flow
        proxy used to compute accrued funding at close time. Binance caps this
        at ~500 rows regardless of the requested limit (confirmed empirically
        elsewhere in this codebase — see cross_asset_edge_discover.py)."""
        params: dict[str, Any] = {"symbol": symbol, "limit": limit}
        if start_time is not None:
            params["startTime"] = int(start_time)
        result = await self._request("GET", self.EP_FUNDING_RATE, params=params)
        return result if isinstance(result, list) else []

    # ──────────────────────────────────────────────────────────────────
    # Signed account / trading
    # ──────────────────────────────────────────────────────────────────
    async def futures_balance(self) -> dict[str, float]:
        """Futures wallet balances as {asset: balance}. Paper mode returns a
        simulated USDT balance (mirrors exchange_client.py's get_account())."""
        if self._is_paper():
            return {"USDT": 1000.0}
        result = await self._request("GET", self.EP_BALANCE, signed=True)
        out: dict[str, float] = {}
        for entry in result if isinstance(result, list) else []:
            try:
                bal = float(entry.get("balance", 0.0) or 0.0)
            except (TypeError, ValueError):
                bal = 0.0
            if bal != 0.0:
                out[entry.get("asset", "")] = bal
        return out

    async def futures_position_information(self, symbol: Optional[str] = None) -> list[dict[str, Any]]:
        """Live position risk (positionAmt, entryPrice, markPrice,
        liquidationPrice) — used for the liquidation-buffer guard. Paper mode
        returns an empty list (no real positions exist)."""
        if self._is_paper():
            return []
        params = {"symbol": symbol} if symbol else None
        result = await self._request("GET", self.EP_POSITION_RISK, params=params, signed=True)
        return result if isinstance(result, list) else []

    async def futures_change_leverage(self, symbol: str, leverage: int) -> dict[str, Any]:
        if self._is_paper():
            return {"symbol": symbol, "leverage": leverage, "maxNotionalValue": "0"}
        params = {"symbol": symbol, "leverage": int(leverage)}
        return await self._request("POST", self.EP_LEVERAGE, params=params, signed=True)

    async def futures_create_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        *,
        order_type: str = "MARKET",
        price: Optional[float] = None,
        time_in_force: str = "GTC",
        client_order_id: Optional[str] = None,
        reduce_only: bool = False,
    ) -> dict[str, Any]:
        """Place a USDT-M futures order. ``side`` in {BUY, SELL}; ``order_type``
        in {MARKET, LIMIT}. ``reduce_only=True`` for closing an existing
        position (prevents accidentally flipping/increasing exposure)."""
        side_u = side.upper()
        if side_u not in ("BUY", "SELL"):
            raise ValueError(f"invalid side: {side!r}")
        type_u = order_type.upper()

        if self._is_paper():
            return await self._simulate_futures_order(
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
        if reduce_only:
            params["reduceOnly"] = "true"

        return await self._request("POST", self.EP_ORDER, params=params, signed=True)

    async def _simulate_futures_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        *,
        order_type: str,
        price: Optional[float],
        client_order_id: Optional[str],
    ) -> dict[str, Any]:
        """Fabricate an instantly-filled futures order response for paper mode.

        Unlike the spot simulator (which uses last-traded price via
        get_prices()), this MUST use mark price (futures_mark_price()) — the
        strategy's own funding math (entry_funding, settled_funding) is keyed
        off mark/premium-index data, so a paper fill priced off a different
        source would be subtly inconsistent with the real accrual logic it's
        meant to be validating."""
        fill_price = price
        if fill_price is None:
            mark_data = await self.futures_mark_price(symbol)
            if isinstance(mark_data, list):
                mark_data = next((m for m in mark_data if m.get("symbol") == symbol), {})
            fill_price = float((mark_data or {}).get("markPrice", 0.0) or 0.0)
        order_id = int(time.time() * 1000)
        qty_s = f"{quantity:.8f}"
        price_s = f"{fill_price:.8f}"
        coid = client_order_id or f"paper-futures-{order_id}"
        resp = {
            "symbol": symbol,
            "orderId": order_id,
            "clientOrderId": coid,
            "updateTime": int(time.time() * 1000),
            "avgPrice": price_s,
            "origQty": qty_s,
            "executedQty": qty_s,
            "cumQuote": f"{quantity * fill_price:.8f}",
            "status": "FILLED",
            "timeInForce": "GTC",
            "type": order_type,
            "side": side,
        }
        self._paper_orders_by_id[str(order_id)] = resp
        return resp


__all__ = ["NativeFuturesExchangeClient", "FuturesExchangeClientError"]
