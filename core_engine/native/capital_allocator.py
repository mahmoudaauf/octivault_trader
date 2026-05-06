"""
Native L6: Adaptive Capital Allocator (Phase 8.2.8 + Feedback)

Allocates trading capital for buy decisions based on available balance,
with optional feedback from AdaptiveCapitalEngine and runtime overrides
from ObjectiveFeedbackController.

Design choices
--------------
* Adaptive allocation: uses ACE to compute dynamic risk_fraction per trade
* Feedback-aware: reads SIZE_MULTIPLIER from OFC runtime_overrides
* Market-driven: incorporates volatility, drawdown, win rate, fee efficiency
* Graceful degradation: falls back to flat % if ACE unavailable
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class NativeCapitalAllocator:
    """
    Adaptive capital allocator for trading decisions.

    Integrates AdaptiveCapitalEngine (ACE) for dynamic sizing based on
    performance history, and reads SIZE_MULTIPLIER from ObjectiveFeedbackController.

    Usage::

        allocator = NativeCapitalAllocator(
            portfolio_manager=pm,
            market_data=md,
            allocation_pct=5.0,
            adaptive_engine=ace,       # NativeAdaptiveCapitalEngine
            shared_state=ss,           # NativeSharedState
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
        adaptive_engine: Any | None = None,
        shared_state: Any | None = None,
    ) -> None:
        """
        Initialize adaptive capital allocator.

        Args:
            portfolio_manager: NativePortfolioManager with get_nav() method
            market_data: NativeMarketData with get_price() method (optional)
            allocation_pct: Base percentage of available USDT per buy signal (default 5%)
            adaptive_engine: NativeAdaptiveCapitalEngine for dynamic sizing (optional)
            shared_state: NativeSharedState for feedback loop data (optional)
        """
        self._pm = portfolio_manager
        self._md = market_data
        self._allocation_pct = max(0.1, min(100.0, float(allocation_pct)))
        self._ace = adaptive_engine
        self._ss = shared_state

    async def allocate_for_buy(self, symbol: str) -> float:
        """
        Allocate quantity for a buy signal.

        First attempts to use AdaptiveCapitalEngine if available, then applies
        SIZE_MULTIPLIER from OFC runtime_overrides. Falls back to flat percentage
        if ACE is unavailable.

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

            # Apply runtime_overrides from ObjectiveFeedbackController if present
            overrides = getattr(self._ss, "runtime_overrides", {}) if self._ss else {}
            size_mult = float(overrides.get("SIZE_MULTIPLIER", 1.0))

            # Compute risk_fraction: either from ACE or flat percentage
            if self._ace and getattr(self._ace, "enabled", False):
                # Build ACE inputs from SharedState
                try:
                    trade_history = self._ss.trade_history.get(symbol, []) if self._ss else []
                    positions = self._ss.get_all_positions() if self._ss else {}
                    free_capital = self._ss.free_balance_usdt if self._ss else nav
                    drawdown_pct = self._compute_drawdown_pct()
                    volatility_pct = self._compute_volatility_pct(symbol)

                    decision = self._ace.evaluate(
                        symbol=symbol,
                        nav=nav,
                        free_capital=free_capital,
                        base_risk_fraction=self._allocation_pct / 100.0,
                        volatility_pct=volatility_pct,
                        drawdown_pct=drawdown_pct,
                        fee_bps=float(self._ss.metrics.get("avg_fee_bps", 10.0))
                        if self._ss
                        else 10.0,
                        slippage_bps=float(self._ss.metrics.get("avg_slippage_bps", 5.0))
                        if self._ss
                        else 5.0,
                        min_notional=10.0,
                        slot_utilization=len(positions) / max(1, 5),
                        throughput_per_hour=float(
                            overrides.get("TARGET_THROUGHPUT_PER_HOUR", 10.0)
                        ),
                        target_throughput_per_hour=10.0,
                        trade_history=trade_history,
                    )
                    risk_fraction = decision.risk_fraction * size_mult

                    logger.debug(
                        "ACE allocation for %s: risk=%.3f mult=%.2f reason=%s",
                        symbol,
                        decision.risk_fraction,
                        size_mult,
                        " | ".join(decision.reasons[:2]),
                    )
                except Exception as e:
                    logger.warning("ACE evaluation failed; falling back to flat: %s", e)
                    risk_fraction = (self._allocation_pct / 100.0) * size_mult
            else:
                # Flat percentage allocation with OFC multiplier
                risk_fraction = (self._allocation_pct / 100.0) * size_mult

            # Convert risk fraction to USDT allocation
            allocation_usdt = nav * risk_fraction
            quantity = allocation_usdt / price

            logger.debug(
                "Allocate for %s: nav=%.2f risk=%.3f mult=%.2f usdt=%.2f " "price=%.2f qty=%.6f",
                symbol,
                nav,
                risk_fraction,
                size_mult,
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

    def _compute_drawdown_pct(self) -> float:
        """Compute current drawdown percentage from peak NAV."""
        if not self._ss:
            return 0.0
        peak = self._ss.metrics.get("peak_nav", self._ss.nav_usdt)
        if not peak or peak <= 0:
            return 0.0
        return float(max(0.0, (peak - self._ss.nav_usdt) / peak * 100.0))

    def _compute_volatility_pct(self, symbol: str) -> float:
        """Compute volatility percentage for symbol.

        Placeholder: returns 0.008 (0.8% — mid-range volatility)
        until NativeMarketData provides rolling volatility estimates."""
        # TODO: Implement rolling volatility from klines
        return 0.008
