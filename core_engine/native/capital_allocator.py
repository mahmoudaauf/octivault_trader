"""
Native L6: Simple Capital Allocator (Phase 8.2.8)

Allocates trading capital for buy decisions based on available balance.
Designed for MVP — allocates fixed percentage per symbol (default 5% of USDT).

Design choices
--------------
* Pure allocation logic: no I/O, no state mutation beyond return value
* Percentage-based: allocate X% of available USDT per signal
* Symbol-agnostic: same allocation policy for all symbols
* Stateless: no tracking of cumulative allocation across signals
  (caller aggregates quantity across multiple signals)
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class NativeCapitalAllocator:
    """
    Simple capital allocator for trading decisions.

    Usage::

        allocator = NativeCapitalAllocator(
            portfolio_manager=pm,
            allocation_pct=5.0,  # Allocate 5% per symbol
        )
        quantity = await allocator.allocate_for_buy(symbol="BTCUSDT")
        # ⇒ float (quantity to buy)
    """

    def __init__(
        self,
        *,
        portfolio_manager: Any,
        market_data: Any | None = None,
        allocation_pct: float = 5.0,
    ) -> None:
        """
        Initialize capital allocator.

        Args:
            portfolio_manager: NativePortfolioManager with get_nav() method
            market_data: NativeMarketData with get_price() method (optional; if not provided, tries portfolio_manager)
            allocation_pct: Percentage of available USDT to allocate per buy signal (default 5%)
        """
        self._pm = portfolio_manager
        self._md = market_data
        self._allocation_pct = max(0.1, min(100.0, float(allocation_pct)))

    async def allocate_for_buy(self, symbol: str) -> float:
        """
        Allocate quantity for a buy signal.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")

        Returns:
            Quantity to buy (0.0 if allocation fails or insufficient capital)
        """
        if not self._pm:
            logger.debug("Portfolio manager unavailable; no allocation")
            return 0.0

        try:
            # Get current NAV
            nav = await self._pm.get_nav()
            if not nav or nav <= 0:
                logger.debug("NAV %s too low to allocate", nav)
                return 0.0

            # Get price from market_data if available, otherwise skip allocation
            price = None
            if self._md and hasattr(self._md, "get_price"):
                price = self._md.get_price(symbol)

            if not price or price <= 0:
                logger.debug("Price for %s unavailable or zero", symbol)
                return 0.0

            # Allocate percentage of NAV
            allocation_usdt = nav * (self._allocation_pct / 100.0)
            quantity = allocation_usdt / price

            logger.debug(
                "Allocate for %s: nav=%.2f pct=%.1f usdt=%.2f price=%.2f qty=%.6f",
                symbol,
                nav,
                self._allocation_pct,
                allocation_usdt,
                price,
                quantity,
            )

            return float(max(0.0, quantity))

        except Exception as e:
            logger.warning("Capital allocation failed for %s: %s", symbol, e)
            return 0.0

    async def allocate_for_sell(self, symbol: str, position_quantity: float) -> float:
        """
        Allocate quantity for a sell signal.

        Args:
            symbol: Trading pair
            position_quantity: Current position size

        Returns:
            Quantity to sell (clamped to position_quantity, default all)
        """
        # For MVP: sell all (100% of position)
        return float(max(0.0, position_quantity))
