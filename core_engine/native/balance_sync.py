"""
Native L1: Balance Sync (Phase 8.2.2)

Lightweight polling-based balance cache. Replaces ~300-line legacy
BalanceSync with a focused ~150-line implementation.

Design choices
--------------
* Periodic polling via asyncio.create_task — no callbacks, no WS.
* Single in-memory dict; zero locks (asyncio is single-threaded per loop).
* Optional update callbacks (sync or async).
* Clean start/stop lifecycle.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Optional, Union

from core_engine.native.exchange_client import (
    ExchangeClientError,
    NativeExchangeClient,
)

logger = logging.getLogger(__name__)

UpdateCallback = Callable[[dict[str, float]], Union[None, Awaitable[None]]]


class NativeBalanceSync:
    """
    Periodically fetch balances from the exchange and cache them in memory.

    Usage::

        client = NativeExchangeClient(key, secret)
        bs = NativeBalanceSync(client, poll_interval_sec=5.0)
        await bs.start()
        ...
        balances = bs.get_balance()      # cached dict
        free_btc = bs.get_asset("BTC")   # single value
        await bs.stop()
    """

    def __init__(
        self,
        client: NativeExchangeClient,
        *,
        poll_interval_sec: float = 5.0,
        on_update: Optional[UpdateCallback] = None,
    ) -> None:
        self._client = client
        self._poll_interval = max(0.5, float(poll_interval_sec))
        self._on_update = on_update

        self._balances: dict[str, float] = {}
        self._last_update_ts: float = 0.0
        self._task: Optional[asyncio.Task[None]] = None
        self._stopped: Optional[asyncio.Event] = None  # Lazy initialization

    # ──────────────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────────────
    async def start(self) -> None:
        """Start the background poller. Idempotent."""
        if self._task and not self._task.done():
            return
        # Lazy-create event when we're in async context
        if self._stopped is None:
            self._stopped = asyncio.Event()
        self._stopped.clear()
        # Prime the cache once before the loop so callers see data fast.
        try:
            await self._refresh_once()
        except Exception as e:  # pragma: no cover — best-effort prime
            logger.warning("initial balance refresh failed: %r", e)
        self._task = asyncio.create_task(self._run(), name="native-balance-sync")

    async def stop(self) -> None:
        """Stop the background poller. Idempotent."""
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

    # ──────────────────────────────────────────────────────────────────
    # Read accessors
    # ──────────────────────────────────────────────────────────────────
    def get_balance(self) -> dict[str, float]:
        """Return a copy of the cached balance dict."""
        return dict(self._balances)

    def get_asset(self, asset: str) -> float:
        """Free amount for a single asset, or 0.0 if absent."""
        return float(self._balances.get(asset, 0.0))

    @property
    def last_update_ts(self) -> float:
        return self._last_update_ts

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    # ──────────────────────────────────────────────────────────────────
    # Polling internals
    # ──────────────────────────────────────────────────────────────────
    async def _run(self) -> None:
        """Background loop. Cancels cleanly on stop."""
        while not self._stopped.is_set():
            try:
                await asyncio.wait_for(self._stopped.wait(), timeout=self._poll_interval)
                # _stopped was set → loop exits next check
                continue
            except asyncio.TimeoutError:
                pass

            try:
                await self._refresh_once()
            except ExchangeClientError as e:
                logger.warning("balance refresh failed (exchange): %s", e)
            except asyncio.CancelledError:
                break
            except Exception as e:  # pragma: no cover — defensive
                logger.exception("balance refresh failed (unexpected): %s", e)

    async def _refresh_once(self) -> None:
        """Single fetch + cache update + callback."""
        new_balances = await self._client.get_balance()
        self._balances = new_balances
        self._last_update_ts = time.time()
        if self._on_update is not None:
            try:
                result = self._on_update(new_balances)
                if inspect.isawaitable(result):
                    await result
            except Exception as e:  # pragma: no cover — never crash poller
                logger.exception("on_update callback failed: %s", e)
