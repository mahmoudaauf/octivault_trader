"""
balance_sync.py - Real-time Balance Synchronization

Continuously syncs account balance from Binance to SharedState.nav
Ensures the bot always operates with live, accurate balance information.

The problem this solves:
- Balance was fetched once at startup and never updated
- Trading decisions used stale NAV
- Actual account balance ($103.91) vs bot's tracked balance ($100.57) diverged
- No real-time awareness of account growth

Solution:
- Continuously fetch account balance from Binance every 2-3 seconds
- Update shared_state.nav with live authoritative balance
- Enables real-time balance monitoring and growth verification
"""

import asyncio
import logging
import time
from typing import Any, Optional

logger = logging.getLogger("BalanceSync")


class BalanceSync:
    """Real-time balance synchronization from Binance to SharedState"""

    def __init__(
        self,
        shared_state: Any,
        exchange_client: Any,
        update_interval_sec: float = 3.0,
        logger_obj: Optional[logging.Logger] = None,
    ):
        """
        Initialize balance sync

        Args:
            shared_state: SharedState instance
            exchange_client: ExchangeClient instance
            update_interval_sec: How often to fetch balance (default 3 seconds)
            logger_obj: Optional logger override
        """
        self.shared_state = shared_state
        self.exchange_client = exchange_client
        self.update_interval_sec = float(update_interval_sec)
        self.logger = logger_obj or logger
        self.running = False
        self._task: Optional[asyncio.Task] = None
        self.last_updated_ts = 0.0
        self.last_nav = 0.0
        self.updates_count = 0
        self.failures_count = 0

    async def start(self) -> None:
        """Start balance sync background task"""
        if self.running:
            self.logger.warning("[BalanceSync] Already running")
            return
        self.running = True
        self._task = asyncio.create_task(self._sync_loop())
        self.logger.info("[BalanceSync] Started (interval=%.1fs)", self.update_interval_sec)

    async def stop(self) -> None:
        """Stop balance sync gracefully"""
        self.running = False
        if self._task:
            try:
                self._task.cancel()
                await self._task
            except asyncio.CancelledError:
                pass
        self.logger.info(
            "[BalanceSync] Stopped (updates=%d, failures=%d)",
            self.updates_count,
            self.failures_count,
        )

    async def _sync_loop(self) -> None:
        """Main balance sync loop"""
        while self.running:
            try:
                await self._fetch_and_update()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.failures_count += 1
                self.logger.warning("[BalanceSync] Sync error: %s", e, exc_info=True)

            await asyncio.sleep(self.update_interval_sec)

    async def _fetch_and_update(self) -> None:
        """Fetch live balance from Binance and update SharedState"""
        if not self.exchange_client:
            self.logger.warning("[BalanceSync] No exchange client available")
            return

        try:
            # Use get_authoritative_nav if available (primary source of truth)
            auth_nav_fn = getattr(self.shared_state, "get_authoritative_nav", None)
            if callable(auth_nav_fn):
                # Call with max_age_sec=0 to force fresh fetch (bypass cache)
                nav = await self._maybe_await(auth_nav_fn(max_age_sec=0.0))
                if nav > 0:
                    self._update_nav(nav, source="authoritative")
                    return

            # Fallback: fetch account balances manually
            if not hasattr(self.exchange_client, "get_account_balances"):
                self.logger.debug("[BalanceSync] ExchangeClient missing get_account_balances")
                return

            balances = await self._maybe_await(self.exchange_client.get_account_balances())
            if not isinstance(balances, dict) or not balances:
                self.logger.debug("[BalanceSync] No balances returned from exchange")
                return

            # Compute NAV from balances
            nav = await self._compute_nav_from_balances(balances)
            if nav > 0:
                self._update_nav(nav, source="computed")

        except Exception as e:
            self.failures_count += 1
            self.logger.debug("[BalanceSync] Failed to fetch balance: %s", e)

    async def _compute_nav_from_balances(self, balances: dict[str, Any]) -> float:
        """
        Compute NAV from account balances

        Args:
            balances: Dict from get_account_balances()
                      {asset: {free: X, locked: Y}, ...}

        Returns:
            Total NAV in USDT
        """
        try:
            quote_assets = ["USDT"]
            if hasattr(self.shared_state, "quote_assets"):
                qa = getattr(self.shared_state, "quote_assets", None)
                if isinstance(qa, list):
                    quote_assets = qa
                elif qa:
                    quote_assets = [str(qa).upper()]

            nav = 0.0

            # Direct USDT balance
            if "USDT" in balances:
                bal = balances["USDT"]
                if isinstance(bal, dict):
                    usdt_qty = float(bal.get("free", 0) or 0) + float(bal.get("locked", 0) or 0)
                    nav += usdt_qty

            # Price other assets at market
            for asset, bal in balances.items():
                if asset.upper() in quote_assets or asset == "USDT":
                    continue
                if not isinstance(bal, dict):
                    continue

                qty = float(bal.get("free", 0) or 0) + float(bal.get("locked", 0) or 0)
                if qty <= 0:
                    continue

                # Get market price
                try:
                    symbol = f"{asset.upper()}USDT"
                    # Try WebSocket cache first
                    price = None
                    cache = getattr(self.shared_state, "_price_cache", None)
                    if isinstance(cache, dict) and symbol in cache:
                        price_tuple = cache.get(symbol)
                        if isinstance(price_tuple, (tuple, list)) and len(price_tuple) >= 1:
                            price = float(price_tuple[0])

                    # Fallback to REST API
                    if not price and hasattr(self.exchange_client, "get_current_price"):
                        price = await self._maybe_await(
                            self.exchange_client.get_current_price(symbol)
                        )

                    if price and float(price) > 0:
                        nav += qty * float(price)
                except Exception:
                    # Skip unavailable prices
                    pass

            return nav

        except Exception as e:
            self.logger.debug("[BalanceSync] NAV computation error: %s", e)
            return 0.0

    def _update_nav(self, nav: float, source: str = "unknown") -> None:
        """Update shared_state.nav with new balance"""
        try:
            now = time.time()
            old_nav = float(getattr(self.shared_state, "nav", 0.0) or 0.0)
            delta = nav - old_nav
            delta_pct = (delta / old_nav * 100.0) if old_nav > 0 else 0.0

            # Update nav attribute
            self.shared_state.nav = nav

            # Cache timestamps for expiry validation
            self.shared_state._nav_sync_ts = now
            self.shared_state._nav_sync_source = source

            self.last_updated_ts = now
            self.last_nav = nav
            self.updates_count += 1

            # Log with proper color coding for growth/decay
            if delta > 0:
                status = "📈 GROWING"
            elif delta < 0:
                status = "📉 DECAYING"
            else:
                status = "➡️ STABLE"

            log_level = logging.INFO if self.updates_count <= 5 else logging.DEBUG

            if log_level == logging.INFO or self.updates_count % 10 == 0:
                self.logger.log(
                    log_level,
                    "[BalanceSync] 💰 NAV updated: $%.2f %s (delta=$%.2f/%+.2f%%) [source=%s]",
                    nav,
                    status,
                    delta,
                    delta_pct,
                    source,
                )
            else:
                self.logger.debug(
                    "[BalanceSync] NAV=$%.2f %s delta=$%.2f/%+.2f%% [%s]",
                    nav,
                    status,
                    delta,
                    delta_pct,
                    source,
                )

        except Exception as e:
            self.logger.warning("[BalanceSync] Failed to update nav: %s", e)

    async def _maybe_await(self, value: Any) -> Any:
        """Await if coroutine, otherwise return value"""
        if hasattr(value, "__await__"):
            return await value
        return value

    def get_status(self) -> dict[str, Any]:
        """Get balance sync status"""
        now = time.time()
        age_sec = now - self.last_updated_ts if self.last_updated_ts > 0 else None
        return {
            "running": self.running,
            "last_nav": self.last_nav,
            "last_updated_ts": self.last_updated_ts,
            "age_sec": age_sec,
            "updates_count": self.updates_count,
            "failures_count": self.failures_count,
            "update_interval_sec": self.update_interval_sec,
        }
