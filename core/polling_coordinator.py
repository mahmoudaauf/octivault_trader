# core/polling_coordinator.py
# ============================================================================
# Polling Coordinator — Staggered order/balance/position polling with
# SharedState + ExecutionManager integration
#
# Architecture:
#   - Open Orders polling:   25s (track live execution state)
#   - Balance polling:       40s (update spendable capital)
#   - Position polling:      25s (sync with exchange ground truth)
#   - Only when active trades exist (efficiency)
# ============================================================================

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, Optional, Callable
from contextlib import asynccontextmanager

logger = logging.getLogger("PollingCoordinator")


class PollingConfig:
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


class PollingCoordinator:
    """
    Manages staggered polling of open orders, balance, and positions.
    
    Responsibilities:
      1. Coordinate three independent polling loops (orders, balance, positions)
      2. Gate polling on presence of active trades (if enabled)
      3. Emit health/readiness events to SharedState
      4. Integrate with ExecutionManager for trade tracking
      5. Provide context for AppContext P4+ initialization
    """
    
    def __init__(
        self,
        shared_state: Any,
        exchange_client: Any,
        config: Optional[PollingConfig] = None,
        logger_: Optional[logging.Logger] = None,
    ):
        """
        Initialize PollingCoordinator.
        
        Args:
            shared_state: SharedState instance for syncing data
            exchange_client: ExchangeClient for API calls
            config: PollingConfig with interval settings
            logger_: Optional logger instance
        """
        self.shared_state = shared_state
        self.exchange_client = exchange_client
        self.config = config or PollingConfig()
        self.logger = logger_ or logging.getLogger("PollingCoordinator")
        
        # Lifecycle
        self._running = False
        self._stop_event = asyncio.Event()
        self._lock = asyncio.Lock()
        
        # Background tasks
        self._open_orders_task: Optional[asyncio.Task] = None
        self._balance_task: Optional[asyncio.Task] = None
        self._position_task: Optional[asyncio.Task] = None
        self._health_task: Optional[asyncio.Task] = None
        
        # Timing trackers (for health reporting)
        self._last_orders_poll: float = 0.0
        self._last_balance_poll: float = 0.0
        self._last_position_poll: float = 0.0
        self._poll_error_count: Dict[str, int] = {
            "orders": 0,
            "balance": 0,
            "positions": 0,
        }
        
        self.logger.info(
            "[PollingCoordinator] Initialized (orders=%.0fs, balance=%.0fs, position=%.0fs)",
            self.config.open_orders_interval_sec,
            self.config.balance_interval_sec,
            self.config.position_interval_sec,
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
            # Launch all three independent polling loops + health emitter
            self._open_orders_task = asyncio.create_task(self._poll_open_orders_loop())
            self._balance_task = asyncio.create_task(self._poll_balance_loop())
            self._position_task = asyncio.create_task(self._poll_positions_loop())
            self._health_task = asyncio.create_task(self._emit_health_loop())
            
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
        
        # Wait for all tasks to finish
        tasks = [t for t in [
            self._open_orders_task,
            self._balance_task,
            self._position_task,
            self._health_task,
        ] if t is not None]
        
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
    # Private: Polling loops
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
        
        # Check if there are active trades
        try:
            has_active = await self._check_active_trades()
            return has_active
        except Exception as e:
            self.logger.debug("[PollingCoordinator] Error checking active trades: %s", e)
            # On error, assume we should poll to be safe
            return True
    
    async def _check_active_trades(self) -> bool:
        """
        Check if there are active trades in SharedState or ExecutionManager.
        
        Returns:
            True if any active trades exist; False otherwise
        """
        # Check SharedState positions
        try:
            if hasattr(self.shared_state, "get_positions"):
                positions = self.shared_state.get_positions()
                if positions:
                    return True
        except Exception:
            pass
        
        # Check SharedState open trades
        try:
            if hasattr(self.shared_state, "open_trades"):
                open_trades = self.shared_state.open_trades
                if isinstance(open_trades, dict) and len(open_trades) > 0:
                    return True
        except Exception:
            pass
        
        return False
    
    async def _poll_open_orders_loop(self) -> None:
        """Poll open orders at OPEN_ORDERS_INTERVAL_SEC intervals."""
        self.logger.info(
            "[PollingCoordinator] Open orders loop starting (interval=%.0fs)",
            self.config.open_orders_interval_sec,
        )
        
        try:
            while self._running:
                try:
                    # Check if we should poll
                    if not await self._should_poll():
                        await asyncio.sleep(1.0)  # Short sleep before re-checking gate
                        continue
                    
                    # Perform poll
                    await self._fetch_and_sync_open_orders()
                    self._last_orders_poll = time.time()
                    self._poll_error_count["orders"] = 0
                    
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    self.logger.error("[PollingCoordinator] Open orders poll error: %s", e, exc_info=True)
                    self._poll_error_count["orders"] += 1
                    await self._emit_health("orders", "error", str(e))
                
                # Sleep for configured interval (or until stop event)
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=self.config.open_orders_interval_sec
                    )
                    # Stop event was set; exit loop
                    break
                except asyncio.TimeoutError:
                    # Timeout = interval elapsed, continue looping
                    pass
        
        except asyncio.CancelledError:
            pass
        finally:
            self.logger.info("[PollingCoordinator] Open orders loop finished")
    
    async def _poll_balance_loop(self) -> None:
        """Poll balance at BALANCE_INTERVAL_SEC intervals."""
        self.logger.info(
            "[PollingCoordinator] Balance loop starting (interval=%.0fs)",
            self.config.balance_interval_sec,
        )
        
        try:
            while self._running:
                try:
                    # Check if we should poll
                    if not await self._should_poll():
                        await asyncio.sleep(1.0)
                        continue
                    
                    # Perform poll
                    await self._fetch_and_sync_balance()
                    self._last_balance_poll = time.time()
                    self._poll_error_count["balance"] = 0
                    
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    self.logger.error("[PollingCoordinator] Balance poll error: %s", e, exc_info=True)
                    self._poll_error_count["balance"] += 1
                    await self._emit_health("balance", "error", str(e))
                
                # Sleep for configured interval
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=self.config.balance_interval_sec
                    )
                    break
                except asyncio.TimeoutError:
                    pass
        
        except asyncio.CancelledError:
            pass
        finally:
            self.logger.info("[PollingCoordinator] Balance loop finished")
    
    async def _poll_positions_loop(self) -> None:
        """Poll positions at POSITION_INTERVAL_SEC intervals."""
        self.logger.info(
            "[PollingCoordinator] Position loop starting (interval=%.0fs)",
            self.config.position_interval_sec,
        )
        
        try:
            while self._running:
                try:
                    # Check if we should poll
                    if not await self._should_poll():
                        await asyncio.sleep(1.0)
                        continue
                    
                    # Perform poll
                    await self._fetch_and_sync_positions()
                    self._last_position_poll = time.time()
                    self._poll_error_count["positions"] = 0
                    
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    self.logger.error("[PollingCoordinator] Position poll error: %s", e, exc_info=True)
                    self._poll_error_count["positions"] += 1
                    await self._emit_health("positions", "error", str(e))
                
                # Sleep for configured interval
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=self.config.position_interval_sec
                    )
                    break
                except asyncio.TimeoutError:
                    pass
        
        except asyncio.CancelledError:
            pass
        finally:
            self.logger.info("[PollingCoordinator] Position loop finished")
    
    async def _emit_health_loop(self) -> None:
        """Periodically emit health/status events."""
        try:
            while self._running:
                try:
                    await self._emit_periodic_health()
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    self.logger.debug("[PollingCoordinator] Health emit error: %s", e)
                
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=self.config.health_cadence_sec
                    )
                    break
                except asyncio.TimeoutError:
                    pass
        
        except asyncio.CancelledError:
            pass
        finally:
            self.logger.debug("[PollingCoordinator] Health emit loop finished")
    
    # ====================================================================
    # Private: Sync operations
    # ====================================================================
    
    async def _fetch_and_sync_open_orders(self) -> None:
        """Fetch open orders and sync to SharedState."""
        if not hasattr(self.exchange_client, "get_open_orders"):
            return
        
        try:
            orders = await self.exchange_client.get_open_orders()
            if orders:
                # Sync to SharedState
                if hasattr(self.shared_state, "open_orders"):
                    self.shared_state.open_orders = orders
                
                self.logger.debug("[PollingCoordinator] Synced %d open orders", len(orders))
        except Exception as e:
            self.logger.error("[PollingCoordinator] Failed to fetch open orders: %s", e)
            raise
    
    async def _fetch_and_sync_balance(self) -> None:
        """Fetch balance and sync to SharedState."""
        if not hasattr(self.exchange_client, "get_balances"):
            return
        
        try:
            balances = await self.exchange_client.get_balances()
            if balances:
                # Sync to SharedState
                if hasattr(self.shared_state, "update_balances"):
                    res = self.shared_state.update_balances(balances)
                    if asyncio.iscoroutine(res):
                        await res
                elif hasattr(self.shared_state, "balances"):
                    self.shared_state.balances = balances
                
                self.logger.debug("[PollingCoordinator] Synced balance from exchange")
        except Exception as e:
            self.logger.error("[PollingCoordinator] Failed to fetch balance: %s", e)
            raise
    
    async def _fetch_and_sync_positions(self) -> None:
        """Fetch positions and sync to SharedState."""
        if not hasattr(self.exchange_client, "get_positions"):
            return
        
        try:
            positions = await self.exchange_client.get_positions()
            if positions:
                # Sync to SharedState
                if hasattr(self.shared_state, "update_positions"):
                    res = self.shared_state.update_positions(positions)
                    if asyncio.iscoroutine(res):
                        await res
                elif hasattr(self.shared_state, "positions"):
                    self.shared_state.positions = positions
                
                self.logger.debug("[PollingCoordinator] Synced %d positions", len(positions))
        except Exception as e:
            self.logger.error("[PollingCoordinator] Failed to fetch positions: %s", e)
            raise
    
    # ====================================================================
    # Private: Health & status
    # ====================================================================
    
    async def _emit_health(self, poll_type: str, status: str, detail: str = "") -> None:
        """Emit health event for a poll type."""
        if not hasattr(self.shared_state, "update_component_status"):
            return
        
        try:
            res = self.shared_state.update_component_status(
                component=f"PollingCoordinator[{poll_type}]",
                status=status,
                detail=detail,
            )
            if asyncio.iscoroutine(res):
                await res
        except Exception:
            pass
    
    async def _emit_periodic_health(self) -> None:
        """Emit a consolidated health status."""
        if not hasattr(self.shared_state, "update_component_status"):
            return
        
        now = time.time()
        orders_age = now - self._last_orders_poll if self._last_orders_poll > 0 else -1.0
        balance_age = now - self._last_balance_poll if self._last_balance_poll > 0 else -1.0
        position_age = now - self._last_position_poll if self._last_position_poll > 0 else -1.0
        
        detail = (
            f"orders_age={orders_age:.1f}s errors={self._poll_error_count['orders']} | "
            f"balance_age={balance_age:.1f}s errors={self._poll_error_count['balance']} | "
            f"position_age={position_age:.1f}s errors={self._poll_error_count['positions']}"
        )
        
        try:
            res = self.shared_state.update_component_status(
                component="PollingCoordinator",
                status="ok" if all(e < 3 for e in self._poll_error_count.values()) else "warn",
                detail=detail,
            )
            if asyncio.iscoroutine(res):
                await res
        except Exception:
            pass
    
    # ====================================================================
    # Public: Query interface
    # ====================================================================
    
    def get_last_poll_times(self) -> Dict[str, float]:
        """Get timestamps of last successful polls."""
        return {
            "open_orders": self._last_orders_poll,
            "balance": self._last_balance_poll,
            "positions": self._last_position_poll,
        }
    
    def get_error_counts(self) -> Dict[str, int]:
        """Get error counts for each poll type."""
        return dict(self._poll_error_count)
    
    def is_running(self) -> bool:
        """Check if coordinator is actively polling."""
        return self._running


# ============================================================================
# Integration helpers for AppContext
# ============================================================================

@asynccontextmanager
async def polling_coordinator_context(
    shared_state: Any,
    exchange_client: Any,
    config: Optional[PollingConfig] = None,
):
    """
    Context manager for PollingCoordinator lifecycle.
    
    Usage:
        async with polling_coordinator_context(shared_state, exchange_client) as pc:
            # PollingCoordinator is running
            await asyncio.sleep(60)
        # PollingCoordinator has been stopped
    """
    pc = PollingCoordinator(shared_state, exchange_client, config)
    await pc.start()
    try:
        yield pc
    finally:
        await pc.stop()
