"""
Native L2: Market Data (Phase 8.2.3 + WebSocket)

In-memory price + klines cache with WebSocket primary source.

Design choices
--------------
* WebSocket primary (real-time prices + klines) — ZERO API rate limits
* REST fallback (on-demand klines, bootstrap) — only when needed
* Stale-quote detection: any price whose age exceeds stale_threshold_sec
* Pure read API after start — no callbacks, no locks (single-threaded asyncio)
* Auto-fallback to REST if WebSocket unavailable
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from typing import Any, Optional

from core_engine.native.exchange_client import (
    ExchangeClientError,
    NativeExchangeClient,
)

logger = logging.getLogger(__name__)


class NativeMarketData:
    """
    Lightweight market-data cache.

    Usage::

        client = NativeExchangeClient(key, secret)
        md = NativeMarketData(client, poll_interval_sec=2.0,
                              symbols=["BTCUSDT", "ETHUSDT"])
        await md.start()
        ...
        price = md.get_price("BTCUSDT")        # cached float or None
        kl = await md.get_klines("BTCUSDT", "1m", 100)
        await md.stop()
    """

    def __init__(
        self,
        client: NativeExchangeClient,
        *,
        poll_interval_sec: float = 2.0,
        symbols: Optional[list[str]] = None,
        stale_threshold_sec: float = 30.0,
        klines_cache_size: int = 64,
        shared_state: Optional[Any] = None,
    ) -> None:
        self._client = client
        self._poll_interval = max(0.25, float(poll_interval_sec))
        self._symbols: Optional[list[str]] = list(symbols) if symbols else None
        self._stale_threshold = float(stale_threshold_sec)
        self._klines_cache_size = max(1, int(klines_cache_size))
        self._shared_state = shared_state  # Optional reference to shared_state for WebSocket data

        self._prices: dict[str, float] = {}
        self._price_ts: dict[str, float] = {}
        self._klines: OrderedDict[
            tuple[str, str, int], tuple[float, list[list[Any]]]
        ] = OrderedDict()

        self._task: Optional[asyncio.Task[None]] = None
        self._stopped: Optional[asyncio.Event] = None  # Lazy initialization

    # ──────────────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────────────
    async def start(self) -> None:
        """Start background poller. Idempotent. Primes cache before loop."""
        if self._task and not self._task.done():
            return
        # Lazy-create event when we're in async context
        if self._stopped is None:
            self._stopped = asyncio.Event()
        self._stopped.clear()
        try:
            await self._refresh_prices()
        except Exception as e:  # pragma: no cover — best-effort prime
            logger.warning("initial market-data refresh failed: %r", e)
        self._task = asyncio.create_task(self._run(), name="native-market-data")

    async def stop(self) -> None:
        """Stop background poller. Idempotent."""
        # Ensure event is created before setting it
        if self._stopped is None:
            self._stopped = asyncio.Event()
        self._stopped.set()
        task = self._task
        self._task = None
        if task and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    # ──────────────────────────────────────────────────────────────────
    # Symbol management
    # ──────────────────────────────────────────────────────────────────
    def set_symbols(self, symbols: Optional[list[str]]) -> None:
        """Restrict polling to ``symbols`` (None = all symbols)."""
        self._symbols = list(symbols) if symbols else None

    def symbols(self) -> Optional[list[str]]:
        return list(self._symbols) if self._symbols else None

    # ──────────────────────────────────────────────────────────────────
    # Price accessors
    # ──────────────────────────────────────────────────────────────────
    def get_price(self, symbol: str) -> Optional[float]:
        # Try shared_state first (WebSocket source), fall back to REST cache
        if self._shared_state and hasattr(self._shared_state, "prices"):
            price = self._shared_state.prices.get(symbol)
            if price is not None:
                return price
        return self._prices.get(symbol)

    def get_prices(self) -> dict[str, float]:
        # Return WebSocket data if available, fall back to REST cache
        if self._shared_state and hasattr(self._shared_state, "prices"):
            return dict(self._shared_state.prices)
        return dict(self._prices)

    def price_age(self, symbol: str) -> Optional[float]:
        ts = self._price_ts.get(symbol)
        if ts is None:
            return None
        return time.time() - ts

    def is_stale(self, symbol: str) -> bool:
        age = self.price_age(symbol)
        if age is None:
            return True  # never seen ⇒ stale
        return age > self._stale_threshold

    def stale_symbols(self) -> list[str]:
        now = time.time()
        return [s for s, ts in self._price_ts.items() if (now - ts) > self._stale_threshold]

    async def prime(self, symbols: Optional[list[str]] = None) -> None:
        """Force a single price refresh (optionally narrowed to ``symbols``)."""
        if symbols is not None:
            self._symbols = list(symbols)
        await self._refresh_prices()

    # ──────────────────────────────────────────────────────────────────
    # Klines (LRU)
    # ──────────────────────────────────────────────────────────────────
    async def get_klines(
        self,
        symbol: str,
        interval: str = "1m",
        limit: int = 100,
        *,
        max_age_sec: float = 60.0,
    ) -> list[list[Any]]:
        """
        Fetch klines with on-demand caching. Returns cached value if fresh
        (age ≤ ``max_age_sec``); otherwise refetches.
        """
        key = (symbol, interval, int(limit))
        now = time.time()
        cached = self._klines.get(key)
        if cached is not None:
            ts, data = cached
            if (now - ts) <= max_age_sec:
                # bump LRU
                self._klines.move_to_end(key)
                return data

        data = await self._client.get_klines(symbol, interval=interval, limit=limit)
        self._klines[key] = (now, data)
        self._klines.move_to_end(key)
        # bound cache
        while len(self._klines) > self._klines_cache_size:
            self._klines.popitem(last=False)
        return data

    # ──────────────────────────────────────────────────────────────────
    # Polling internals
    # ──────────────────────────────────────────────────────────────────
    async def _run(self) -> None:
        while not self._stopped.is_set():
            try:
                await asyncio.wait_for(self._stopped.wait(), timeout=self._poll_interval)
                continue
            except asyncio.TimeoutError:
                pass

            try:
                await self._refresh_prices()
                await self._refresh_klines()
            except ExchangeClientError as e:
                logger.warning("market-data refresh failed (exchange): %s", e)
            except asyncio.CancelledError:
                break
            except Exception as e:  # pragma: no cover
                logger.exception("market-data refresh failed (unexpected): %s", e)

    async def _refresh_prices(self) -> None:
        # Skip REST polling if no symbols configured (WebSocket will handle it)
        if not self._symbols:
            return
        prices = await self._client.get_prices(self._symbols)
        now = time.time()
        # Replace dict atomically; preserve previously-seen symbols' last
        # known price if not in the new payload (stale flagging will catch it).
        for sym, px in prices.items():
            self._prices[sym] = px
            self._price_ts[sym] = now

    async def _refresh_klines(self) -> None:
        if not self._symbols:
            return
        now = time.time()
        interval = "1m"
        limit = 100
        for sym in self._symbols:
            try:
                klines = await self._client.get_klines(sym, interval, limit)
                if klines:
                    key = (sym, interval, limit)
                    self._klines[key] = (now, klines)
                    # Enforce LRU: remove oldest if cache exceeds size
                    while len(self._klines) > self._klines_cache_size:
                        self._klines.popitem(last=False)
            except ExchangeClientError as e:
                logger.debug("klines fetch failed for %s: %s", sym, e)
