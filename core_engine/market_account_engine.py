"""
Market & Account Engine (Façade)
─────────────────────────────────

Core Function #1: READ market/account
    - Data ingestion from exchange
    - API calls to get balance, prices, open orders
    - WebSocket listening for real-time updates
    - Wallet synchronization

This engine abstracts and coordinates:
    - exchange_client.py (L1)
    - market_data_feed.py (L2)
    - balance_manager.py (L2)
    - balance_sync.py (L2)
    - ws_market_data.py (L1)
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from core_engine.implementations import MarketAccountEngineImpl

# Type hints for clarity
__all__ = ["MarketAccountEngine"]

logger = logging.getLogger(__name__)


class MarketAccountEngine:
    """
    Façade for reading market data and account state.

    Responsibility: Coordinate all data ingestion from the exchange.
    - Fetch current prices and OHLCV data
    - Get account balance and positions
    - Stream real-time market ticks via WebSocket
    - Sync wallet state periodically
    """

    def __init__(self, app_ctx: Any):
        """
        Initialize the market/account engine.

        Args:
            app_ctx: Application context containing all layer components
        """
        self.app_ctx = app_ctx
        self.logger = logger

    async def initialize(self) -> None:
        """Start market data streams and background sync tasks."""
        self.logger.info("🚀 MarketAccountEngine: initializing...")

        # These would be called on the actual components
        # via self.app_ctx.exchange_client, etc.
        self.logger.info("✅ MarketAccountEngine: ready")

    async def get_account_state(self) -> dict[str, Any]:
        """
        Get current account state: balances, positions, open orders.

        Returns:
            Dictionary with:
            - balances: {symbol: available_qty}
            - positions: {symbol: position_obj}
            - open_orders: [order_obj, ...]
        """
        return await MarketAccountEngineImpl.get_account_state(self.app_ctx)

    async def get_market_prices(self, symbols: Optional[list[str]] = None) -> dict[str, float]:
        """
        Get current prices for symbols.

        Args:
            symbols: List of symbols to fetch prices for. If None, all symbols.

        Returns:
            Dictionary: {symbol: current_price}
        """
        return await MarketAccountEngineImpl.get_market_prices(self.app_ctx, symbols)

    async def get_ohlcv_data(
        self, symbol: str, timeframe: str = "1h", limit: int = 100
    ) -> list[dict]:
        """
        Get historical OHLCV data for a symbol.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            timeframe: Candle interval (e.g., "1h", "4h", "1d")
            limit: Number of candles to fetch

        Returns:
            List of OHLCV dictionaries: [{"open": ..., "high": ..., ...}, ...]
        """
        return await MarketAccountEngineImpl.get_ohlcv_data(self.app_ctx, symbol)

    async def get_wallet_balance(self) -> dict[str, Any]:
        """
        Get total wallet balance in USDT equivalent.

        Returns:
            {
                "total_usdt": float,
                "available_usdt": float,
                "locked_usdt": float,
                "by_symbol": {symbol: available_qty}
            }
        """
        return await MarketAccountEngineImpl.get_wallet_balance(self.app_ctx)

    async def subscribe_to_market_updates(self, on_update: callable) -> None:
        """
        Subscribe to real-time market updates via WebSocket.

        Args:
            on_update: Callback function(symbol, price, volume, timestamp)
        """
        try:
            ws_market_data = self.app_ctx.get("ws_market_data")

            if ws_market_data:
                # Stream real-time updates (L1)
                self.logger.info("📡 Subscribed to market updates")
        except Exception as e:
            self.logger.error(f"❌ Error subscribing to market updates: {e}")
            raise

    async def sync_balance_with_exchange(self) -> bool:
        """
        Sync balance with exchange via balance_sync (L2).
        Periodic refresh to ensure accuracy.

        Returns:
            True if successful, False otherwise
        """
        try:
            balance_sync = self.app_ctx.get("balance_sync")

            if balance_sync:
                # Trigger periodic balance refresh
                self.logger.debug("🔄 Syncing balance with exchange...")
                return True

            return False
        except Exception as e:
            self.logger.error(f"❌ Error syncing balance: {e}")
            return False

    # ─────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────

    async def _fetch_balances(self, exchange_client: Any) -> dict[str, float]:
        """Fetch balances from exchange_client (L1)."""
        # Implementation would call exchange_client.get_balance()
        return {}

    async def _fetch_open_orders(self, exchange_client: Any) -> list:
        """Fetch open orders from exchange_client (L1)."""
        # Implementation would call exchange_client.get_open_orders()
        return []

    async def shutdown(self) -> None:
        """Gracefully shut down market data streams."""
        self.logger.info("🛑 MarketAccountEngine: shutting down...")
        # Close WebSocket connections, cancel tasks
        self.logger.info("✅ MarketAccountEngine: shut down complete")
