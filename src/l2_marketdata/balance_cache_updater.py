"""
balance_cache_updater.py - Real-Time Balance Cache Synchronization

Immediately updates the cached balance in SharedState when new capital arrives
on Binance, enabling the bot to trade without waiting for Binance API polls.

Architecture:
  • Monitors account balance changes in real-time
  • Updates SharedState.nav immediately when balance increases
  • Triggers balance-dependent gates (affordability checks, etc.)
  • Logs all cache updates with timestamp and source
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger("BalanceCacheUpdater")


class BalanceCacheUpdater:
    """
    Real-time cache updater for account balance.

    When capital is deposited to Binance account, immediately updates
    the in-memory cached balance so trading can begin without delay.
    """

    def __init__(self, shared_state=None, exchange_client=None, config=None):
        """
        Initialize balance cache updater.

        Args:
            shared_state: SharedState instance to update
            exchange_client: Exchange client for fetching fresh balance
            config: Config instance for parameters
        """
        self.shared_state = shared_state
        self.exchange_client = exchange_client
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Track balance state
        self.last_known_balance = 0.0
        self.last_update_time = 0.0
        self.update_count = 0
        self.balance_increase_detected = False

        # Configuration
        self.poll_interval_seconds = 5.0  # Check balance every 5 seconds
        self.min_balance_increase_threshold = 0.01  # Minimum $0.01 change to trigger update
        self.enable_logging = True

        # Load config if provided
        if self.config:
            self.poll_interval_seconds = float(
                getattr(config, "BALANCE_CACHE_POLL_INTERVAL_SEC", 5.0)
            )
            self.min_balance_increase_threshold = float(
                getattr(config, "BALANCE_CACHE_MIN_INCREASE_THRESHOLD", 0.01)
            )
            self.enable_logging = bool(getattr(config, "BALANCE_CACHE_LOGGING_ENABLED", True))

    async def start_monitoring(self):
        """
        Start real-time balance monitoring loop.

        This should be called as a background task during bot startup.
        Continuously monitors for balance changes and updates cache immediately.
        """
        self.logger.info(
            f"[BalanceCacheUpdater] Starting real-time balance monitoring "
            f"(poll interval: {self.poll_interval_seconds}s)"
        )

        try:
            while True:
                try:
                    # Poll fresh balance from Binance
                    fresh_balance = await self._fetch_fresh_balance()

                    if fresh_balance is not None:
                        # Check if balance increased
                        balance_increase = fresh_balance - self.last_known_balance

                        if abs(balance_increase) >= self.min_balance_increase_threshold:
                            # Significant change detected
                            await self._update_cache(fresh_balance, balance_increase)

                        self.last_known_balance = fresh_balance

                    # Wait before next poll
                    await asyncio.sleep(self.poll_interval_seconds)

                except Exception as e:
                    self.logger.error(
                        f"[BalanceCacheUpdater] Error in monitoring loop: {e}", exc_info=True
                    )
                    # Continue monitoring even if there's an error
                    await asyncio.sleep(self.poll_interval_seconds)

        except asyncio.CancelledError:
            self.logger.info("[BalanceCacheUpdater] Monitoring cancelled")
            raise

    async def _fetch_fresh_balance(self) -> Optional[float]:
        """
        Fetch fresh balance from Binance API.

        Returns:
            Balance in USDT or None if fetch failed
        """
        try:
            if not self.exchange_client:
                return None

            # Get account balance
            balance_info = await self.exchange_client.get_account_balance("USDT")

            if isinstance(balance_info, dict):
                free_balance = float(balance_info.get("free", 0.0))
                return free_balance
            elif isinstance(balance_info, (int, float)):
                return float(balance_info)

            return None

        except Exception as e:
            self.logger.warning(f"[BalanceCacheUpdater] Failed to fetch fresh balance: {e}")
            return None

    async def _update_cache(self, new_balance: float, balance_change: float):
        """
        Update SharedState cache with new balance immediately.

        Args:
            new_balance: New USDT balance
            balance_change: Change from last known balance
        """
        try:
            if not self.shared_state:
                return

            # Update the main NAV in metrics
            self.shared_state.metrics["nav"] = new_balance
            self.shared_state.nav = new_balance

            # Also update portfolio_nav mirror
            self.shared_state.portfolio_nav = new_balance

            # Update total_value
            self.shared_state.total_value = new_balance

            # Update update timestamp
            self.shared_state.metrics["last_balance_update_ts"] = time.time()
            self.shared_state.metrics["last_balance_update_dt"] = datetime.utcnow().isoformat()

            # Mark as ready for trading
            if new_balance >= 10.0:
                self.shared_state.metrics["nav_ready"] = True
                if hasattr(self.shared_state, "nav_ready_event"):
                    self.shared_state.nav_ready_event.set()

            # Increment update counter
            self.update_count += 1

            # Log the update
            if self.enable_logging:
                log_message = (
                    f"[BalanceCacheUpdater:LIVE_UPDATE] #{self.update_count} "
                    f"Balance: ${new_balance:.2f} "
                )

                if balance_change > 0:
                    log_message += f"(+${balance_change:.2f} DEPOSITED ✅)"
                    self.balance_increase_detected = True
                elif balance_change < 0:
                    log_message += f"(-${abs(balance_change):.2f})"

                self.logger.info(log_message)

            return True

        except Exception as e:
            self.logger.error(f"[BalanceCacheUpdater] Failed to update cache: {e}", exc_info=True)
            return False

    async def force_balance_update(self, new_balance: float):
        """
        Force an immediate balance update (for testing or manual adjustment).

        Args:
            new_balance: New balance to set
        """
        self.logger.info(
            f"[BalanceCacheUpdater:FORCE_UPDATE] " f"Forcing balance to ${new_balance:.2f}"
        )

        balance_change = new_balance - self.last_known_balance
        await self._update_cache(new_balance, balance_change)
        self.last_known_balance = new_balance

    def get_status(self) -> dict[str, Any]:
        """
        Get current updater status.

        Returns:
            Status dictionary
        """
        return {
            "last_known_balance": self.last_known_balance,
            "last_update_time": self.last_update_time,
            "update_count": self.update_count,
            "balance_increase_detected": self.balance_increase_detected,
            "poll_interval_seconds": self.poll_interval_seconds,
            "enabled": True,
        }


# Convenience function to create and start the updater
async def create_and_start_balance_updater(
    shared_state,
    exchange_client,
    config=None,
) -> BalanceCacheUpdater:
    """
    Create and start a balance cache updater.

    Args:
        shared_state: SharedState instance
        exchange_client: Exchange client
        config: Config instance (optional)

    Returns:
        Started BalanceCacheUpdater instance
    """
    updater = BalanceCacheUpdater(
        shared_state=shared_state,
        exchange_client=exchange_client,
        config=config,
    )

    # Start monitoring in background
    asyncio.create_task(updater.start_monitoring(), name="BalanceCacheUpdater")

    return updater
