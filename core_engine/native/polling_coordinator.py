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
        position_price_refresh_interval_sec: float = 60.0,
    ):
        """
        Initialize polling configuration.

        Args:
            open_orders_interval_sec: How often to poll open orders (default 25s)
            balance_interval_sec: How often to poll balance (default 40s)
            position_interval_sec: How often to poll positions (default 25s)
            enable_active_trades_gate: Skip polling if no active trades (default True)
            health_cadence_sec: How often to emit health/status (default 5s)
            position_price_refresh_interval_sec: Batch REST price fetch for all held symbols (default 60s)
        """
        self.fills_reconcile_interval_sec: float = 60.0  # cross-check fill prices vs Binance
        self.open_orders_interval_sec = float(open_orders_interval_sec)
        self.balance_interval_sec = float(balance_interval_sec)
        self.position_interval_sec = float(position_interval_sec)
        self.enable_active_trades_gate = bool(enable_active_trades_gate)
        self.health_cadence_sec = float(health_cadence_sec)
        self.position_price_refresh_interval_sec = float(position_price_refresh_interval_sec)


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
        self._position_price_refresh_task: Optional[asyncio.Task] = None
        self._fills_task: Optional[asyncio.Task] = None

        # Timing trackers
        self._startup_ts: float = (
            time.time()
        )  # For startup grace period (allow polling without trades)
        self._last_orders_poll: float = 0.0
        self._last_balance_poll: float = 0.0
        self._last_position_poll: float = 0.0
        self._last_price_refresh_poll: float = 0.0
        self._fills_processed: set[int] = set()  # dedup by Binance trade ID (int)
        self._last_fills_poll: float = 0.0
        self._poll_error_count: dict[str, int] = {
            "orders": 0,
            "balance": 0,
            "positions": 0,
            "price_refresh": 0,
            "fills": 0,
        }

        self.logger.info(
            "[PollingCoordinator] Initialized (orders=%.0fs, balance=%.0fs, positions=%.0fs, price_refresh=%.0fs, gate=%s)",
            self.config.open_orders_interval_sec,
            self.config.balance_interval_sec,
            self.config.position_interval_sec,
            self.config.position_price_refresh_interval_sec,
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
            self._position_price_refresh_task = asyncio.create_task(self._poll_position_prices_loop())
            self._fills_task = asyncio.create_task(self._poll_fills_loop())

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

        tasks = [t for t in [self._open_orders_task, self._balance_task, self._position_task, self._position_price_refresh_task, self._fills_task] if t]

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
        throttled_until_ts = float(
            getattr(self.shared_state, "exchange_throttle_until_ts", 0.0) or 0.0
        )
        if throttled_until_ts > time.time():
            return False
        if hasattr(self.exchange_client, "is_throttled") and self.exchange_client.is_throttled():
            self._mark_throttle_state(RuntimeError("exchange throttled"))
            return False
        if not self.config.enable_active_trades_gate:
            return True

        # Allow polling during startup grace period (first 60 seconds)
        if time.time() - self._startup_ts < 60.0:
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
        """Poll balance at BALANCE_INTERVAL_SEC intervals.

        Note: Balance polling is NOT gated by active trades (unlike orders/positions).
        Balance is critical for NAV tracking and capital allocation, so we poll
        continuously on schedule regardless of whether trades are open.
        """
        self.logger.info(
            "[PollingCoordinator] Balance loop starting (interval=%.0fs)",
            self.config.balance_interval_sec,
        )

        try:
            while self._running:
                try:
                    # Balance polling is always allowed (not gated by active trades)
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
            self._mark_throttle_state(e)
            self.logger.debug("[PollingCoordinator] Failed to fetch open orders: %s", e)
            raise

    async def _fetch_and_sync_balance(self) -> None:
        """Fetch balance and sync to SharedState."""
        if not hasattr(self.exchange_client, "get_balance"):
            return

        try:
            if self._should_defer_balance_sync():
                self.logger.debug("[PollingCoordinator] Deferring balance sync this cycle")
                return
            balances = await self.exchange_client.get_balance()
            if balances and hasattr(self.shared_state, "update_balance_map"):
                self.shared_state.update_balance_map(balances)
                # Also update free_balance_usdt directly for decision engine
                usdt_balance = float(balances.get("USDT", 0.0))
                if usdt_balance > 0 and hasattr(self.shared_state, "free_balance_usdt"):
                    # update_balance_map above already sets free_balance_usdt + NAV correctly
                    # Always reset session anchor on first balance sync of this process.
                    # This ensures drawdown is measured from today's starting NAV,
                    # not a stale value persisted from a previous session.
                    if not getattr(self, "_session_anchor_set", False):
                        self._session_anchor_set = True
                        # Anchor is set by orchestrator after startup completes — skip here
                # Keep OFC telemetry alive — it gates on last_update_ts and elapsed_h freshness
                if hasattr(self.shared_state, "metrics") and isinstance(self.shared_state.metrics, dict):
                    _now = time.time()
                    self.shared_state.metrics["last_update_ts"] = _now
                    _start = float(getattr(self.shared_state, "_session_start_ts", _now) or _now)
                    self.shared_state.metrics["session_elapsed_h"] = (_now - _start) / 3600.0
                self.logger.debug(
                    "[PollingCoordinator] Synced balance from exchange: %.2f USDT", usdt_balance
                )
        except Exception as e:
            self._mark_throttle_state(e)
            self.logger.debug("[PollingCoordinator] Failed to fetch balance: %s", e)
            raise

    def _should_defer_balance_sync(self) -> bool:
        if bool(getattr(self.shared_state, "exchange_throttled", False)):
            return True
        active_orders = len(getattr(self.shared_state, "open_orders", {}) or {})
        active_positions = len(getattr(self.shared_state, "positions", {}) or {})
        # Always poll on every cycle if we have active trades
        if active_orders > 0 or active_positions > 0:
            return False
        # No active trades: during startup (2 min grace), always poll every 40s
        if (time.time() - self._startup_ts) < 120.0:
            return False
        # After 2 minutes of idle (no trades): sparse polling every 30 minutes
        # Check if we've ever polled before and how recently
        last_balance_poll = float(self._last_balance_poll or 0.0)
        if last_balance_poll <= 0:
            return False  # Never polled: do it now
        # Polled before: only skip if within 30-minute window
        return (time.time() - last_balance_poll) < 1800.0

    async def _fetch_and_sync_positions(self) -> None:
        """Sync position marks from shared-state prices.

        Native spot stack tracks fills/positions locally; there is no separate
        exchange positions endpoint. This loop refreshes mark prices so NAV/PnL
        stay current even when balance polling is sparse.
        """
        try:
            if not hasattr(self.shared_state, "get_all_positions"):
                return
            positions = self.shared_state.get_all_positions()
            price_cache = getattr(self.shared_state, "price_cache", {}) or {}
            updated = 0
            for sym, pos in positions.items():
                price = float(
                    price_cache.get(str(sym).upper())
                    or price_cache.get(sym)
                    or 0.0
                )
                if price > 0:
                    # Only update mark price — never re-write qty from stale state.
                    # qty is authoritative from exchange balance reconciliation.
                    if isinstance(pos, dict):
                        pos["mark_price"] = price
                    elif hasattr(pos, "mark_price"):
                        pos.mark_price = price
                    updated += 1
            if updated:
                self.logger.debug("[PollingCoordinator] Refreshed %d position marks", updated)
        except Exception as e:
            self._mark_throttle_state(e)
            self.logger.debug("[PollingCoordinator] Failed to fetch positions: %s", e)
            raise

    async def _poll_position_prices_loop(self) -> None:
        """Batch-fetch live prices for all held position symbols every 60s via REST.

        Fills the gap for symbols not covered by WebSocket subscriptions so NAV
        stays accurate for the full wallet (not just the 10 actively traded symbols).
        One REST call fetches all symbols at once — minimal API weight.
        """
        self.logger.info(
            "[PollingCoordinator] Position price refresh loop starting (interval=%.0fs)",
            self.config.position_price_refresh_interval_sec,
        )
        try:
            while self._running:
                try:
                    await self._fetch_and_refresh_position_prices()
                    self._last_price_refresh_poll = time.time()
                    self._poll_error_count["price_refresh"] = 0
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    self.logger.debug("[PollingCoordinator] Position price refresh error: %s", e)
                    self._poll_error_count["price_refresh"] += 1

                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=self.config.position_price_refresh_interval_sec,
                    )
                    break
                except asyncio.TimeoutError:
                    pass
        except asyncio.CancelledError:
            pass
        finally:
            self.logger.debug("[PollingCoordinator] Position price refresh loop finished")

    async def _fetch_and_refresh_position_prices(self) -> None:
        """Fetch live prices for all held positions in one batch REST call."""
        if not hasattr(self.exchange_client, "get_prices"):
            return
        positions = getattr(self.shared_state, "positions", {}) or {}
        if not positions:
            return

        symbols = [str(sym).upper() for sym in positions.keys()]
        try:
            prices = await self.exchange_client.get_prices(symbols=symbols)
        except Exception as e:
            self._mark_throttle_state(e)
            self.logger.debug("[PollingCoordinator] Batch price fetch failed: %s", e)
            raise

        if not prices:
            return

        # Update price_cache with fresh prices
        price_cache = getattr(self.shared_state, "price_cache", None)
        if price_cache is not None:
            price_cache.update(prices)

        # Also call update_price if available (updates prices + price_timestamps)
        if hasattr(self.shared_state, "update_price"):
            for sym, price in prices.items():
                self.shared_state.update_price(sym, price)

        self.logger.info(
            "[PollingCoordinator] Refreshed live prices for %d position symbols (nav accuracy)",
            len(prices),
        )

    async def _poll_fills_loop(self) -> None:
        """Cross-check fill prices against Binance trade history every 60s.

        Authoritative source: Binance get_my_trades() returns the actual
        cummulativeQuoteQty/executedQty for each fill — the only reliable avg
        fill price for market orders (order response price field is "0").

        Only updates entry_price when:
          - Position entry_price is 0 (never recorded), OR
          - New fills detected since last poll (trade ID not in dedup set)
        Never overwrites a valid entry with a worse estimate.
        """
        self.logger.info("[PollingCoordinator] Fills reconciliation loop starting (interval=60s)")
        try:
            while self._running:
                try:
                    if not await self._should_poll():
                        await asyncio.sleep(1.0)
                        continue
                    await self._fetch_and_reconcile_fills()
                    self._last_fills_poll = time.time()
                    self._poll_error_count["fills"] = 0
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    self.logger.debug("[PollingCoordinator] Fills reconciliation error: %s", e)
                    self._poll_error_count["fills"] += 1

                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=self.config.fills_reconcile_interval_sec,
                    )
                    break
                except asyncio.TimeoutError:
                    pass
        except asyncio.CancelledError:
            pass
        finally:
            self.logger.debug("[PollingCoordinator] Fills reconciliation loop finished")

    async def _fetch_and_reconcile_fills(self) -> None:
        """Fetch recent trades from Binance and reconcile entry prices in shared_state."""
        if not hasattr(self.exchange_client, "get_account_trades"):
            return
        positions = getattr(self.shared_state, "positions", {}) or {}
        if not positions:
            return

        for sym in list(positions.keys()):
            pos = positions.get(sym)
            if pos is None:
                continue
            current_entry = float(
                pos.get("entry_price", 0.0) if isinstance(pos, dict)
                else getattr(pos, "entry_price", 0.0)
            )
            try:
                trades = await self.exchange_client.get_account_trades(symbol=str(sym).upper(), limit=20)
            except Exception as e:
                self.logger.debug("[PollingCoordinator] get_account_trades(%s) failed: %s", sym, e)
                continue

            if not trades:
                continue

            # Find new BUY fills not yet processed
            new_buys = [
                t for t in trades
                if int(t.get("id") or 0) not in self._fills_processed
                and str(t.get("isBuyer") or t.get("side") or "").upper() in ("TRUE", "BUY", "1")
            ]

            if not new_buys and current_entry > 0:
                # Mark all as seen, nothing to update
                for t in trades:
                    self._fills_processed.add(int(t.get("id") or 0))
                continue

            # Compute weighted average fill price from ALL buy trades for this symbol
            all_buys = [
                t for t in trades
                if str(t.get("isBuyer") or t.get("side") or "").upper() in ("TRUE", "BUY", "1")
            ]
            if not all_buys:
                continue

            total_qty = sum(float(t.get("qty") or 0) for t in all_buys)
            total_cost = sum(float(t.get("qty") or 0) * float(t.get("price") or 0) for t in all_buys)
            avg_fill_price = total_cost / total_qty if total_qty > 0 else 0.0

            if avg_fill_price <= 0:
                continue

            # Only update if entry is missing OR a new fill changed the average materially (>0.1%)
            needs_update = current_entry <= 0 or (
                new_buys and abs(avg_fill_price - current_entry) / current_entry > 0.001
            )

            if needs_update:
                current_qty = float(
                    pos.get("qty", 0.0) if isinstance(pos, dict)
                    else getattr(pos, "qty", 0.0)
                )
                current_mark = float(
                    pos.get("mark_price", avg_fill_price) if isinstance(pos, dict)
                    else getattr(pos, "mark_price", avg_fill_price)
                )
                if hasattr(self.shared_state, "update_position"):
                    self.shared_state.update_position(
                        symbol=str(sym).upper(),
                        qty=current_qty,
                        entry=avg_fill_price,
                        current=current_mark,
                    )
                    self.logger.info(
                        "[PollingCoordinator:FILLS] %s entry reconciled: %.6f → %.6f (Binance avg fill, %d trades)",
                        sym, current_entry, avg_fill_price, len(all_buys),
                    )

            # Mark all fetched trades as processed
            for t in trades:
                self._fills_processed.add(int(t.get("id") or 0))

        # Trim dedup set to avoid unbounded growth (keep last 10k IDs)
        if len(self._fills_processed) > 10000:
            self._fills_processed = set(sorted(self._fills_processed)[-5000:])

    def _mark_throttle_state(self, err: Exception) -> None:
        if not hasattr(self.shared_state, "set_exchange_throttle"):
            return
        self.shared_state.set_exchange_throttle(
            bool(getattr(self.exchange_client, "is_throttled", lambda: False)()),
            reason=str(getattr(self.exchange_client, "last_error", lambda: "")() or err),
            until_ts=float(
                getattr(self.exchange_client, "throttled_until_ts", lambda: 0.0)() or 0.0
            ),
        )

    # ====================================================================
    # Public: Query interface
    # ====================================================================

    def get_last_poll_times(self) -> dict[str, float]:
        """Get timestamps of last successful polls."""
        return {
            "open_orders": self._last_orders_poll,
            "balance": self._last_balance_poll,
            "positions": self._last_position_poll,
            "price_refresh": self._last_price_refresh_poll,
            "fills": self._last_fills_poll,
        }

    def get_error_counts(self) -> dict[str, int]:
        """Get error counts for each poll type."""
        return dict(self._poll_error_count)

    def is_running(self) -> bool:
        """Check if coordinator is actively polling."""
        return self._running
