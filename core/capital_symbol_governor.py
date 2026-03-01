"""
Capital & Symbol Governor

Dynamically caps the number of active trading symbols based on:
  • Available capital
  • Risk constraints
  • System health metrics

This ensures bootstrap trades don't overextend a small account.

Architecture:
  Placement: Between SymbolManager → Set Accepted Symbols
  Integration: SymbolManager calls governor BEFORE finalizing accepted symbols

Rules:
  1. Capital Floor: Maps equity to symbol cap (e.g., <250 USDT → max 2 symbols)
  2. API Health: Reduces cap if rate limit detected
  3. Retrain Stability: Reduces cap if retrain skipped (depth not loading)
  4. Drawdown Guard: Drops to 1 symbol if drawdown > 8%
"""

import asyncio
import logging
from typing import Any, Optional


class CapitalSymbolGovernor:
    """
    Governor that dynamically limits active symbols based on capital and system health.
    
    Usage:
      governor = CapitalSymbolGovernor(shared_state, config)
      symbol_cap = await governor.compute_symbol_cap()
      accepted_symbols = accepted_symbols[:symbol_cap]
    """

    def __init__(
        self,
        shared_state: Optional[Any] = None,
        config: Optional[Any] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.shared_state = shared_state
        self.config = config
        self.logger = logger or logging.getLogger("CapitalSymbolGovernor")

        # Config parameters
        self.exposure_ratio = float(
            getattr(config, "MAX_EXPOSURE_RATIO", 0.6) if config else 0.6
        )
        self.min_trade_size_usdt = float(
            getattr(config, "MIN_ECONOMIC_TRADE_USDT", 30) if config else 30
        )
        self.max_drawdown_guard = float(
            getattr(config, "MAX_DRAWDOWN_PCT", 8.0) if config else 8.0
        )
        self.max_retrain_skips = int(
            getattr(config, "MAX_RETRAIN_SKIPS", 2) if config else 2
        )

        # Health state tracking
        self._api_rate_limited = False
        self._retrain_skipped_count = 0
        self._last_check_time = 0.0

    async def compute_symbol_cap(self) -> int:
        """
        Compute the dynamic symbol cap based on capital and health rules.
        
        Returns:
          int: Number of symbols to trade (minimum 1, typically 2-4)
        """
        # Rule 1: Capital Floor mapping
        equity = await self._get_equity()
        cap = self._capital_floor_cap(equity)
        
        self.logger.info(
            f"🎛️  Capital Floor: equity={equity:.2f} USDT → cap={cap} symbols"
        )

        # Rule 2: API Health Guard
        if self._api_rate_limited:
            cap = max(1, cap - 1)
            self.logger.warning(
                f"⚠️  API Rate Limited detected → reduce cap to {cap}"
            )

        # Rule 3: Retrain Stability Guard
        if self._retrain_skipped_count > self.max_retrain_skips:
            cap = max(1, cap - 1)
            self.logger.warning(
                f"⚠️  Retrain skipped {self._retrain_skipped_count} times → reduce cap to {cap}"
            )

        # Rule 4: Drawdown Guard
        drawdown = await self._get_drawdown_pct()
        if drawdown is not None and drawdown > self.max_drawdown_guard:
            cap = 1
            self.logger.warning(
                f"🛡️  Drawdown {drawdown:.2f}% > {self.max_drawdown_guard}% → DEFENSIVE (cap=1)"
            )

        return cap

    def _capital_floor_cap(self, equity: float) -> int:
        """
        Map equity to symbol cap using static tiers.
        
        Equity Range    | Cap
        ───────────────────────
        < 250           | 2
        250–800         | 3
        800–2000        | 4
        2000+           | dynamic
        """
        if equity < 250:
            return 2
        elif equity < 800:
            return 3
        elif equity < 2000:
            return 4
        else:
            # Dynamic: cap = max(2, floor(usable_capital / min_trade_size))
            usable = equity * self.exposure_ratio
            raw_cap = int(usable // self.min_trade_size_usdt)
            return max(2, raw_cap)

    async def _get_equity(self) -> float:
        """
        Fetch total USDT equity from SharedState.balances.
        
        Returns:
          float: Total USDT equity (free + locked)
        """
        if not self.shared_state:
            self.logger.warning("❌ SharedState not available, using 0 equity")
            return 0.0

        try:
            balances = self.shared_state.balances
            if not balances:
                return 0.0

            usdt_balance = balances.get("USDT", {})
            free = float(usdt_balance.get("free", 0.0))
            locked = float(usdt_balance.get("locked", 0.0))
            return free + locked

        except Exception as e:
            self.logger.error(f"❌ Error fetching equity: {e}")
            return 0.0

    async def _get_drawdown_pct(self) -> Optional[float]:
        """
        Fetch current drawdown percentage from SharedState.
        
        Returns:
          float or None: Current drawdown as percentage (e.g., 5.2 for 5.2%)
        """
        if not self.shared_state:
            return None

        try:
            # Try to get drawdown from shared_state (various possible attributes)
            if hasattr(self.shared_state, "current_drawdown"):
                return float(self.shared_state.current_drawdown)
            
            if hasattr(self.shared_state, "metrics"):
                metrics = self.shared_state.metrics
                if isinstance(metrics, dict) and "drawdown_pct" in metrics:
                    return float(metrics["drawdown_pct"])
            
            # Fallback: compute from peak equity if available
            if hasattr(self.shared_state, "peak_equity") and hasattr(self.shared_state, "current_equity"):
                peak = float(self.shared_state.peak_equity)
                current = float(self.shared_state.current_equity)
                if peak > 0:
                    return ((peak - current) / peak) * 100.0
            
            return None

        except Exception as e:
            self.logger.debug(f"Could not fetch drawdown: {e}")
            return None

    def mark_api_rate_limited(self):
        """
        Called by MarketDataFeed when rate limit detected.
        Triggers Rule 2 on next compute_symbol_cap().
        """
        self._api_rate_limited = True
        self.logger.info("📊 API rate limit flagged for governor")

    def clear_api_rate_limit(self):
        """Clear the rate limit flag after cooldown."""
        self._api_rate_limited = False
        self.logger.info("✅ API rate limit cleared")

    def record_retrain_skip(self):
        """
        Called when ML retrain is skipped (e.g., depth not loading).
        Triggers Rule 3 if count exceeds max_retrain_skips.
        """
        self._retrain_skipped_count += 1
        self.logger.debug(
            f"📊 Retrain skip recorded ({self._retrain_skipped_count} total)"
        )

    def reset_retrain_skips(self):
        """Reset the retrain skip counter after successful retrain."""
        self._retrain_skipped_count = 0
        self.logger.debug("✅ Retrain skip counter reset")
