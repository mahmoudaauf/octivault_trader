"""
Native Polling Coordinator — Staggered polling with active-trades gate

Ports legacy PollingCoordinator from src/l1_exchange/polling_coordinator.py
to native stack. Uses wider intervals (25-40 seconds) instead of aggressive
2-5s REST polling. Only polls when active trades exist (efficiency gate).

Reduces API weight from ~1800/min to ~200/min:
  - Open orders: 25s (1200 weight/10min → 120 weight → 2.4 calls/min)
  - Balance: 40s (1200 weight/10min → 75 weight → 1.5 calls/min)
  - Positions: 25s (1200 weight/10min → 120 weight → 2.4 calls/min)
  - Only when trades exist (default gate: True)
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


class NativePollingConfig:
    """Configuration for staggered polling intervals."""

    def __init__(
        self,
        open_orders_interval_sec: float = 25.0,
        balance_interval_sec: float = 40.0,
        position_interval_sec: float = 25.0,
        enable_active_trades_gate: bool = True,
        health_cadence_sec: float = 5.0,
    ):
        """
        Initialize polling configuration.

        Args:
            open_orders_interval_sec: How often to poll open orders (default 25s)
            balance_interval_sec: How often to poll balance (default 40s)
            position_interval_sec: How often to poll positions (default 25s)
            enable_active_trades_gate: Skip polling if no active trades (default True)
            health_cadence_sec: How often to emit health/status (default 5s)
        """
        self.open_orders_interval_sec = float(open_orders_interval_sec)
        self.balance_interval_sec = float(balance_interval_sec)
        self.position_interval_sec = float(position_interval_sec)
        self.enable_active_trades_gate = bool(enable_active_trades_gate)
        self.health_cadence_sec = float(health_cadence_sec)


class NativePollingCoordinator:
    """
    Manages staggered polling of open orders, balance, and positions.

    Responsibilities:
      1. Coordinate three independent polling loops (orders, balance, positions)
      2. Gate polling on presence of active trades (if enabled)
      3. Integrate with SharedState for data syncing
      4. Reduce API weight via wide intervals + active-trades gate

    This replaces aggressive polling (2s market data, 5s balance, 5s fills)
    with legacy-style staggered approach (25s orders, 40s balance, 25s positions).
    """

    def __init__(
        self,
        shared_state: Any,
        exchange_client: Any,
        config: Optional[NativePollingConfig] = None,
        logger_: Optional[logging.Logger] = None,
    ):
        """
        Initialize NativePollingCoordinator.

        Args:
            shared_state: NativeSharedState instance
            exchange_client: NativeExchangeClient instance
            config: NativePollingConfig with interval settings
            logger_: Optional logger instance
        """
        self.shared_state = shared_state
        self.exchange_client = exchange_client
        self.config = config or NativePollingConfig()
        self.logger = logger_ or logging.getLogger("NativePollingCoordinator")

        # Lifecycle
        self._running = False
        self._stop_event = asyncio.Event()

        # Background tasks
        self._open_orders_task: Optional[asyncio.Task] = None
        self._balance_task: Optional[asyncio.Task] = None
        self._position_task: Optional[asyncio.Task] = None

        # Timing trackers
        self._last_orders_poll: float = 0.0
        self._last_balance_poll: float = 0.0
        self._last_position_poll: float = 0.0
        self._poll_error_count: dict[str, int] = {
            "orders": 0,
            "balance": 0,
            "positions": 0,
        }

        self.logger.info(
            "[PollingCoordinator] Initialized (orders=%.0fs, balance=%.0fs, positions=%.0fs, gate=%s)",
            self.config.open_orders_interval_sec,
            self.config.balance_interval_sec,
            self.config.position_interval_sec,
            "enabled" if self.config.enable_active_trades_gate else "disabled",
        )

    # ====================================================================
    # Lifecycle
    # ====================================================================

    async def start(self) -> None:
        """Start all polling loops."""
        if self._running:
            self.logger.warning("[PollingCoordinator] Already running")
            return

        self._running = True
        self._stop_event.clear()

        self.logger.info("[PollingCoordinator] Starting polling loops...")

        try:
            self._open_orders_task = asyncio.create_task(self._poll_open_orders_loop())
            self._balance_task = asyncio.create_task(self._poll_balance_loop())
            self._position_task = asyncio.create_task(self._poll_positions_loop())

            self.logger.info("[PollingCoordinator] All polling loops started")
        except Exception as e:
            self._running = False
            self.logger.error("[PollingCoordinator] Failed to start: %s", e, exc_info=True)
            raise

    async def stop(self) -> None:
        """Stop all polling loops gracefully."""
        if not self._running:
            return

        self.logger.info("[PollingCoordinator] Stopping polling loops...")
        self._running = False
        self._stop_event.set()

        tasks = [t for t in [self._open_orders_task, self._balance_task, self._position_task] if t]

        if tasks:
            try:
                await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=10.0)
            except asyncio.TimeoutError:
                self.logger.warning("[PollingCoordinator] Tasks did not finish within timeout")
                for t in tasks:
                    if t and not t.done():
                        t.cancel()

        self.logger.info("[PollingCoordinator] Polling loops stopped")

    # ====================================================================
    # Private: Gate & checks
    # ====================================================================

    async def _should_poll(self) -> bool:
        """
        Determine if polling should proceed.

        Returns False if:
          - enable_active_trades_gate is True AND
          - there are no active trades/positions

        Otherwise returns True.
        """
        if not self.config.enable_active_trades_gate:
            return True

        try:
            return await self._check_active_trades()
        except Exception as e:
            self.logger.debug("[PollingCoordinator] Error checking active trades: %s", e)
            return True  # Default to polling on error

    async def _check_active_trades(self) -> bool:
        """
        Check if there are active trades in SharedState.

        Returns:
            True if any active positions exist; False otherwise
        """
        try:
            if hasattr(self.shared_state, "get_all_positions"):
                positions = self.shared_state.get_all_positions()
                if positions:
                    return True
        except Exception:
            pass

        return False

    # ====================================================================
    # Private: Polling loops
    # ====================================================================

    async def _poll_open_orders_loop(self) -> None:
        """Poll open orders at OPEN_ORDERS_INTERVAL_SEC intervals."""
        self.logger.info(
            "[PollingCoordinator] Open orders loop starting (interval=%.0fs)",
            self.config.open_orders_interval_sec,
        )

        try:
            while self._running:
                try:
                    if not await self._should_poll():
                        await asyncio.sleep(1.0)
                        continue

                    await self._fetch_and_sync_open_orders()
                    self._last_orders_poll = time.time()
                    self._poll_error_count["orders"] = 0

                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    self.logger.debug("[PollingCoordinator] Open orders poll error: %s", e)
                    self._poll_error_count["orders"] += 1

                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(), timeout=self.config.open_orders_interval_sec
                    )
                    break
                except asyncio.TimeoutError:
                    pass

        except asyncio.CancelledError:
            pass
        finally:
            self.logger.debug("[PollingCoordinator] Open orders loop finished")

    async def _poll_balance_loop(self) -> None:
        """Poll balance at BALANCE_INTERVAL_SEC intervals."""
        self.logger.info(
            "[PollingCoordinator] Balance loop starting (interval=%.0fs)",
            self.config.balance_interval_sec,
        )

        try:
            while self._running:
                try:
                    if not await self._should_poll():
                        await asyncio.sleep(1.0)
                        continue

                    await self._fetch_and_sync_balance()
                    self._last_balance_poll = time.time()
                    self._poll_error_count["balance"] = 0

                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    self.logger.debug("[PollingCoordinator] Balance poll error: %s", e)
                    self._poll_error_count["balance"] += 1

                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(), timeout=self.config.balance_interval_sec
                    )
                    break
                except asyncio.TimeoutError:
                    pass

        except asyncio.CancelledError:
            pass
        finally:
            self.logger.debug("[PollingCoordinator] Balance loop finished")

    async def _poll_positions_loop(self) -> None:
        """Poll positions at POSITION_INTERVAL_SEC intervals."""
        self.logger.info(
            "[PollingCoordinator] Position loop starting (interval=%.0fs)",
            self.config.position_interval_sec,
        )

        try:
            while self._running:
                try:
                    if not await self._should_poll():
                        await asyncio.sleep(1.0)
                        continue

                    await self._fetch_and_sync_positions()
                    self._last_position_poll = time.time()
                    self._poll_error_count["positions"] = 0

                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    self.logger.debug("[PollingCoordinator] Position poll error: %s", e)
                    self._poll_error_count["positions"] += 1

                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(), timeout=self.config.position_interval_sec
                    )
                    break
                except asyncio.TimeoutError:
                    pass

        except asyncio.CancelledError:
            pass
        finally:
            self.logger.debug("[PollingCoordinator] Position loop finished")

    # ====================================================================
    # Private: Sync operations
    # ====================================================================

    async def _fetch_and_sync_open_orders(self) -> None:
        """Fetch open orders and sync to SharedState."""
        if not hasattr(self.exchange_client, "get_open_orders"):
            return

        try:
            orders = await self.exchange_client.get_open_orders()
            if orders and hasattr(self.shared_state, "open_orders"):
                self.shared_state.open_orders = orders
                self.logger.debug("[PollingCoordinator] Synced %d open orders", len(orders))
        except Exception as e:
            self.logger.debug("[PollingCoordinator] Failed to fetch open orders: %s", e)
            raise

    async def _fetch_and_sync_balance(self) -> None:
        """Fetch balance and sync to SharedState."""
        if not hasattr(self.exchange_client, "get_balances"):
            return

        try:
            balances = await self.exchange_client.get_balances()
            if balances and hasattr(self.shared_state, "update_balances"):
                res = self.shared_state.update_balances(balances)
                if asyncio.iscoroutine(res):
                    await res
                self.logger.debug("[PollingCoordinator] Synced balance from exchange")
        except Exception as e:
            self.logger.debug("[PollingCoordinator] Failed to fetch balance: %s", e)
            raise

    async def _fetch_and_sync_positions(self) -> None:
        """Fetch positions and sync to SharedState."""
        if not hasattr(self.exchange_client, "get_positions"):
            return

        try:
            positions = await self.exchange_client.get_positions()
            if positions and hasattr(self.shared_state, "update_positions"):
                res = self.shared_state.update_positions(positions)
                if asyncio.iscoroutine(res):
                    await res
                self.logger.debug("[PollingCoordinator] Synced %d positions", len(positions))
        except Exception as e:
            self.logger.debug("[PollingCoordinator] Failed to fetch positions: %s", e)
            raise

    # ====================================================================
    # Public: Query interface
    # ====================================================================

    def get_last_poll_times(self) -> dict[str, float]:
        """Get timestamps of last successful polls."""
        return {
            "open_orders": self._last_orders_poll,
            "balance": self._last_balance_poll,
            "positions": self._last_position_poll,
        }

    def get_error_counts(self) -> dict[str, int]:
        """Get error counts for each poll type."""
        return dict(self._poll_error_count)

    def is_running(self) -> bool:
        """Check if coordinator is actively polling."""
        return self._running
