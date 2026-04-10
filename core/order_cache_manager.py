# core/order_cache_manager.py
# ============================================================================
# Order Cache Manager — Efficient order tracking via local caching
#
# Philosophy:
#   - MAIN SOURCE: Local cache updated on order placement/fill events
#   - SAFETY NET:  Periodic polling for reconciliation (detects exchange fills)
#   - POLLING ONLY: When cache is empty (no active trades)
#
# Pattern:
#   1. place_order() → store locally immediately
#   2. on_fill() → update quantity
#   3. _reconcile() → poll exchange only if active_orders > 0
# ============================================================================

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, Optional, List, Set, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from contextlib import asynccontextmanager

logger = logging.getLogger("OrderCacheManager")


class OrderStatus(Enum):
    """Order lifecycle states."""
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    PENDING_CANCEL = "PENDING_CANCEL"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


@dataclass
class CachedOrder:
    """
    In-memory order representation.
    
    Attributes:
        order_id: Unique order ID from exchange
        client_order_id: Client-assigned order ID (optional)
        symbol: Trading pair (e.g., "BTCUSDT")
        side: "BUY" or "SELL"
        quantity: Original order quantity
        executed_quantity: How much has filled
        price: Limit price (for limit orders)
        status: OrderStatus enum
        placed_at: Unix timestamp when created locally
        last_update: Unix timestamp of last state change
        tags: Dict of metadata (strategy_id, decision_id, etc.)
    """
    order_id: str
    client_order_id: str
    symbol: str
    side: str  # "BUY" or "SELL"
    quantity: float
    executed_quantity: float
    price: float
    status: OrderStatus
    placed_at: float
    last_update: float
    tags: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def remaining_quantity(self) -> float:
        """Unfilled portion of order."""
        return max(0.0, self.quantity - self.executed_quantity)
    
    @property
    def is_open(self) -> bool:
        """True if order can still be filled."""
        return self.status in (OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED)
    
    @property
    def age_sec(self) -> float:
        """How many seconds since this order was placed."""
        return time.time() - self.placed_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        d = asdict(self)
        d["status"] = self.status.value
        return d
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> CachedOrder:
        """Deserialize from dictionary."""
        d_copy = dict(d)
        status_val = d_copy.pop("status", "NEW")
        status = OrderStatus(status_val) if isinstance(status_val, str) else status_val
        return CachedOrder(status=status, **d_copy)


class CacheMissReason(Enum):
    """Why an order is missing from cache after poll."""
    ORDER_FILLED = "filled"
    ORDER_CANCELLED = "cancelled"
    ORDER_EXPIRED = "expired"
    CACHE_CLEARED = "cache_cleared"
    UNKNOWN = "unknown"


@dataclass
class CacheMissEvent:
    """Notification when a cached order is no longer on exchange."""
    order_id: str
    symbol: str
    reason: CacheMissReason
    timestamp: float = field(default_factory=time.time)


class OrderCacheManager:
    """
    Maintains local cache of placed orders.
    
    Responsibilities:
      1. Accept order placements from ExecutionManager
      2. Track order fills/cancellations from events
      3. Reconcile with exchange periodically (only if cache not empty)
      4. Emit events for fills and stale orders
      5. Provide query interface for strategy/execution decisions
    """
    
    def __init__(
        self,
        shared_state: Any,
        exchange_client: Any,
        config: Optional[Dict[str, Any]] = None,
        logger_: Optional[logging.Logger] = None,
    ):
        """
        Initialize OrderCacheManager.
        
        Args:
            shared_state: SharedState for syncing
            exchange_client: ExchangeClient for API calls
            config: Configuration dict with keys:
                - reconciliation_interval_sec: How often to poll (default 30.0)
                - stale_order_timeout_sec: Age before marking stale (default 300.0)
                - cache_size_limit: Max orders to track (default 1000)
            logger_: Optional logger
        """
        self.shared_state = shared_state
        self.exchange_client = exchange_client
        self.config = config or {}
        self.logger = logger_ or logging.getLogger("OrderCacheManager")
        
        # Configuration
        self.reconciliation_interval_sec = float(
            self.config.get("reconciliation_interval_sec", 30.0)
        )
        self.stale_order_timeout_sec = float(
            self.config.get("stale_order_timeout_sec", 300.0)
        )
        self.cache_size_limit = int(self.config.get("cache_size_limit", 1000))
        
        # Local cache: symbol -> {order_id -> CachedOrder}
        self._cache: Dict[str, Dict[str, CachedOrder]] = {}
        self._cache_lock = asyncio.Lock()
        
        # Lifecycle
        self._running = False
        self._stop_event = asyncio.Event()
        self._reconcile_task: Optional[asyncio.Task] = None
        
        # Metrics
        self._total_orders_placed = 0
        self._total_fills = 0
        self._total_cache_misses = 0
        self._last_reconcile_time = 0.0
        self._cache_miss_events: List[CacheMissEvent] = []
        
        self.logger.info(
            "[OrderCacheManager] Initialized (reconcile_interval=%.1fs, stale_timeout=%.1fs)",
            self.reconciliation_interval_sec,
            self.stale_order_timeout_sec,
        )
    
    # ====================================================================
    # Lifecycle
    # ====================================================================
    
    async def start(self) -> None:
        """Start reconciliation loop."""
        if self._running:
            self.logger.warning("[OrderCacheManager] Already running")
            return
        
        self._running = True
        self._stop_event.clear()
        self.logger.info("[OrderCacheManager] Starting")
        
        try:
            self._reconcile_task = asyncio.create_task(self._reconcile_loop())
            self.logger.info("[OrderCacheManager] Reconciliation loop started")
        except Exception as e:
            self._running = False
            self.logger.error("[OrderCacheManager] Failed to start: %s", e, exc_info=True)
            raise
    
    async def stop(self) -> None:
        """Stop reconciliation loop."""
        if not self._running:
            return
        
        self.logger.info("[OrderCacheManager] Stopping")
        self._running = False
        self._stop_event.set()
        
        if self._reconcile_task:
            try:
                await asyncio.wait_for(self._reconcile_task, timeout=5.0)
            except asyncio.TimeoutError:
                self.logger.warning("[OrderCacheManager] Reconcile task timeout")
                self._reconcile_task.cancel()
        
        self.logger.info("[OrderCacheManager] Stopped")
    
    # ====================================================================
    # Public API: Order Registration
    # ====================================================================
    
    async def register_placed_order(
        self,
        order_id: str,
        client_order_id: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float = 0.0,
        tags: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Cache an order that was just placed.
        
        Called IMMEDIATELY after ExchangeClient.place_*_order() succeeds.
        
        Args:
            order_id: Exchange order ID
            client_order_id: Client-assigned ID
            symbol: Trading pair
            side: "BUY" or "SELL"
            quantity: Order size
            price: Limit price (0 for market orders)
            tags: Optional metadata dict
        """
        async with self._cache_lock:
            if symbol not in self._cache:
                self._cache[symbol] = {}
            
            now = time.time()
            order = CachedOrder(
                order_id=str(order_id),
                client_order_id=str(client_order_id),
                symbol=symbol,
                side=side.upper(),
                quantity=float(quantity),
                executed_quantity=0.0,
                price=float(price),
                status=OrderStatus.NEW,
                placed_at=now,
                last_update=now,
                tags=tags or {},
            )
            
            self._cache[symbol][order.order_id] = order
            self._total_orders_placed += 1
            
            self.logger.info(
                "[OrderCacheManager] Registered: %s %s %.8f %s @ $%.2f",
                symbol, side.upper(), quantity, order.order_id, price,
            )
    
    async def register_fill(
        self,
        order_id: str,
        symbol: str,
        filled_qty: float,
    ) -> bool:
        """
        Update cached order with a fill event.
        
        Called when PositionManager receives executionReport with FILLED/PARTIALLY_FILLED.
        
        Args:
            order_id: Order to update
            symbol: Symbol for context
            filled_qty: How much filled in THIS event
        
        Returns:
            True if order was found and updated; False if cache miss
        """
        async with self._cache_lock:
            if symbol not in self._cache:
                self._cache[symbol] = {}
            
            if order_id not in self._cache[symbol]:
                self.logger.debug(
                    "[OrderCacheManager] Fill received for unknown order %s (cache miss)", order_id
                )
                self._total_cache_misses += 1
                return False
            
            order = self._cache[symbol][order_id]
            old_executed = order.executed_quantity
            order.executed_quantity = min(
                order.quantity,
                order.executed_quantity + filled_qty,
            )
            order.last_update = time.time()
            
            # Update status
            if order.executed_quantity >= order.quantity:
                order.status = OrderStatus.FILLED
            else:
                order.status = OrderStatus.PARTIALLY_FILLED
            
            self._total_fills += 1
            
            self.logger.info(
                "[OrderCacheManager] Fill: %s executed +%.8f (total=%.8f/%.8f)",
                order_id, filled_qty, order.executed_quantity, order.quantity,
            )
            
            return True
    
    async def register_cancellation(self, order_id: str, symbol: str) -> bool:
        """
        Mark a cached order as cancelled.
        
        Args:
            order_id: Order to cancel
            symbol: Symbol for context
        
        Returns:
            True if order was found and cancelled; False otherwise
        """
        async with self._cache_lock:
            if symbol not in self._cache or order_id not in self._cache[symbol]:
                return False
            
            order = self._cache[symbol][order_id]
            order.status = OrderStatus.CANCELED
            order.last_update = time.time()
            
            self.logger.info(
                "[OrderCacheManager] Cancelled: %s (filled=%.8f/%.8f)",
                order_id, order.executed_quantity, order.quantity,
            )
            
            return True
    
    # ====================================================================
    # Public API: Queries
    # ====================================================================
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> Dict[str, List[CachedOrder]]:
        """
        Get all open orders (status=NEW or PARTIALLY_FILLED).
        
        Args:
            symbol: If provided, return only orders for this symbol
        
        Returns:
            Dict[symbol, List[CachedOrder]]
        """
        async with self._cache_lock:
            result = {}
            
            symbols_to_check = [symbol] if symbol else list(self._cache.keys())
            
            for sym in symbols_to_check:
                if sym not in self._cache:
                    continue
                
                open_orders = [
                    o for o in self._cache[sym].values()
                    if o.is_open
                ]
                if open_orders:
                    result[sym] = open_orders
            
            return result
    
    async def get_order(self, order_id: str, symbol: str) -> Optional[CachedOrder]:
        """
        Fetch a specific order from cache.
        
        Returns:
            CachedOrder if found; None otherwise
        """
        async with self._cache_lock:
            if symbol not in self._cache:
                return None
            return self._cache[symbol].get(order_id)
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics for health reporting.
        
        Returns:
            Dict with metrics
        """
        async with self._cache_lock:
            total_cached = sum(len(orders) for orders in self._cache.values())
            open_cached = sum(
                sum(1 for o in orders.values() if o.is_open)
                for orders in self._cache.values()
            )
        
        return {
            "total_cached_orders": total_cached,
            "open_cached_orders": open_cached,
            "total_orders_placed": self._total_orders_placed,
            "total_fills": self._total_fills,
            "total_cache_misses": self._total_cache_misses,
            "last_reconcile_time": self._last_reconcile_time,
            "recent_cache_misses": len(self._cache_miss_events[-10:]),  # Last 10
        }
    
    # ====================================================================
    # Private: Reconciliation Loop
    # ====================================================================
    
    async def _reconcile_loop(self) -> None:
        """
        Periodically reconcile cache with exchange.
        
        Only polls when there are open orders (efficiency gate).
        """
        self.logger.info(
            "[OrderCacheManager] Reconciliation loop starting (interval=%.1fs)",
            self.reconciliation_interval_sec,
        )
        
        try:
            while self._running:
                try:
                    # Check if cache is empty
                    async with self._cache_lock:
                        has_open = any(
                            any(o.is_open for o in orders.values())
                            for orders in self._cache.values()
                        )
                    
                    if has_open:
                        # Reconcile with exchange
                        await self._reconcile_with_exchange()
                        self._last_reconcile_time = time.time()
                    else:
                        self.logger.debug("[OrderCacheManager] Cache empty; skipping reconcile")
                    
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    self.logger.error("[OrderCacheManager] Reconciliation error: %s", e, exc_info=True)
                
                # Sleep or wait for stop
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=self.reconciliation_interval_sec,
                    )
                    break
                except asyncio.TimeoutError:
                    continue
        
        except asyncio.CancelledError:
            self.logger.debug("[OrderCacheManager] Reconcile loop cancelled")
            raise
    
    async def _reconcile_with_exchange(self) -> None:
        """
        Fetch open orders from exchange and detect cache misses.
        
        Algorithm:
          1. Get all cached symbols with open orders
          2. For each symbol: fetch_open_orders() from exchange
          3. For each cached order: check if it's still on exchange
          4. If missing: emit cache_miss event
          5. If new fills detected: update cache
        """
        try:
            async with self._cache_lock:
                # Get set of symbols with open orders
                symbols_to_reconcile = [
                    sym for sym, orders in self._cache.items()
                    if any(o.is_open for o in orders.values())
                ]
            
            if not symbols_to_reconcile:
                return
            
            self.logger.debug(
                "[OrderCacheManager] Reconciling %d symbols with exchange",
                len(symbols_to_reconcile),
            )
            
            for symbol in symbols_to_reconcile:
                try:
                    # Fetch exchange orders for this symbol
                    exchange_orders = await self.exchange_client.get_open_orders(symbol)
                    exchange_order_ids = {
                        str(o.get("orderId", o.get("order_id", "")))
                        for o in (exchange_orders or [])
                        if isinstance(o, dict)
                    }
                    
                    # Detect cache misses
                    async with self._cache_lock:
                        if symbol not in self._cache:
                            continue
                        
                        cached_open = {
                            oid: order
                            for oid, order in self._cache[symbol].items()
                            if order.is_open
                        }
                        
                        for oid, cached_order in cached_open.items():
                            if oid not in exchange_order_ids:
                                # Order disappeared from exchange
                                reason = CacheMissReason.ORDER_FILLED
                                self._handle_cache_miss(
                                    oid, symbol, cached_order, reason
                                )
                    
                    # Detect new fills from exchange data
                    await self._detect_fills_from_exchange(symbol, exchange_orders)
                
                except Exception as e:
                    self.logger.error(
                        "[OrderCacheManager] Reconcile failed for %s: %s", symbol, e
                    )
        
        except Exception as e:
            self.logger.error("[OrderCacheManager] Exchange reconciliation failed: %s", e)
    
    def _handle_cache_miss(
        self,
        order_id: str,
        symbol: str,
        order: CachedOrder,
        reason: CacheMissReason,
    ) -> None:
        """
        Process a cached order that's no longer on exchange.
        
        Updates cache and emits event.
        """
        # Mark order as complete
        if reason == CacheMissReason.ORDER_FILLED:
            order.status = OrderStatus.FILLED
            order.executed_quantity = order.quantity
        elif reason == CacheMissReason.ORDER_CANCELLED:
            order.status = OrderStatus.CANCELED
        
        order.last_update = time.time()
        
        # Record miss event
        event = CacheMissEvent(order_id=order_id, symbol=symbol, reason=reason)
        self._cache_miss_events.append(event)
        self._total_cache_misses += 1
        
        self.logger.info(
            "[OrderCacheManager] Cache miss: %s %s reason=%s (filled=%.8f/%.8f)",
            order_id, symbol, reason.value,
            order.executed_quantity, order.quantity,
        )
    
    async def _detect_fills_from_exchange(
        self,
        symbol: str,
        exchange_orders: Optional[List[Dict[str, Any]]],
    ) -> None:
        """
        Check exchange order data for fills we might have missed.
        
        Updates cache with executedQty from exchange.
        """
        if not exchange_orders:
            return
        
        try:
            async with self._cache_lock:
                if symbol not in self._cache:
                    return
                
                for ex_order in exchange_orders:
                    if not isinstance(ex_order, dict):
                        continue
                    
                    oid = str(ex_order.get("orderId", ex_order.get("order_id", "")))
                    if not oid or oid not in self._cache[symbol]:
                        continue
                    
                    cached = self._cache[symbol][oid]
                    ex_executed = float(ex_order.get("executedQty", cached.executed_quantity))
                    
                    if ex_executed > cached.executed_quantity:
                        # Found a fill we missed
                        fill_qty = ex_executed - cached.executed_quantity
                        cached.executed_quantity = ex_executed
                        cached.last_update = time.time()
                        
                        if ex_executed >= cached.quantity:
                            cached.status = OrderStatus.FILLED
                        else:
                            cached.status = OrderStatus.PARTIALLY_FILLED
                        
                        self.logger.debug(
                            "[OrderCacheManager] Fill detected from exchange: %s +%.8f",
                            oid, fill_qty,
                        )
        
        except Exception as e:
            self.logger.debug("[OrderCacheManager] Fill detection error: %s", e)
    
    # ====================================================================
    # Cleanup & Maintenance
    # ====================================================================
    
    async def cleanup_stale_orders(self) -> int:
        """
        Remove filled/cancelled orders older than stale_order_timeout_sec.
        
        Returns:
            Number of orders removed
        """
        removed = 0
        now = time.time()
        
        async with self._cache_lock:
            for symbol in list(self._cache.keys()):
                orders_by_id = self._cache[symbol]
                stale_ids = [
                    oid for oid, order in orders_by_id.items()
                    if not order.is_open and (now - order.last_update) > self.stale_order_timeout_sec
                ]
                
                for oid in stale_ids:
                    del orders_by_id[oid]
                    removed += 1
                
                # Clean up empty symbol dicts
                if not orders_by_id:
                    del self._cache[symbol]
        
        if removed > 0:
            self.logger.info("[OrderCacheManager] Cleaned up %d stale orders", removed)
        
        return removed
    
    async def clear_cache(self) -> None:
        """Clear all cached orders (use cautiously)."""
        async with self._cache_lock:
            self._cache.clear()
        self.logger.warning("[OrderCacheManager] Cache cleared")
    
    # ====================================================================
    # Context Manager
    # ====================================================================
    
    @asynccontextmanager
    async def order_cache_context(self):
        """Context manager for order cache lifecycle."""
        try:
            await self.start()
            yield self
        finally:
            await self.stop()
