"""
Portfolio Authority (Layer 3) - P9 Canonical Design
Provides higher-level governance for capital utilization, profit recycling, and target velocity.
"""

import time
import logging
from typing import Dict, Any, List, Optional, Tuple


def _dynamic_exposure_cap(nav: float) -> float:
    """Return max single-symbol exposure fraction based on NAV bracket.

    Micro accounts are allowed to be concentrated (only one position possible).
    As NAV grows, diversification rules progressively tighten.
    """
    if nav < 200.0:
        return 0.90
    elif nav < 500.0:
        return 0.70
    elif nav < 2000.0:
        return 0.50
    else:
        return 0.30


class PortfolioAuthority:
    def __init__(self, logger: logging.Logger, config: Any, shared_state: Any):
        self.logger = logger
        self.config = config
        self.ss = shared_state
        
        # Thresholds
        self.min_utilization = float(getattr(config, "PORTFOLIO_MIN_UTILIZATION_PCT", 0.5)) # 50%
        self.target_velocity_ratio = float(getattr(config, "TARGET_PROFIT_RATIO_PER_HOUR", 0.001))
        self.max_symbol_concentration = float(getattr(config, "MAX_SYMBOL_CONCENTRATION_PCT", 0.3)) # 30%

    def _is_permanent_dust_position(self, symbol: str, pos: Dict[str, Any]) -> bool:
        """Permanent dust is invisible to portfolio governance."""
        sym = str(symbol or "").upper()
        try:
            if hasattr(self.ss, "is_permanent_dust") and self.ss.is_permanent_dust(sym):
                return True
        except Exception:
            pass
        threshold = float(getattr(self.config, "PERMANENT_DUST_USDT_THRESHOLD", 1.0) or 1.0)
        try:
            value = float((pos or {}).get("value_usdt", 0.0) or 0.0)
            if value <= 0:
                qty = float((pos or {}).get("quantity", 0.0) or (pos or {}).get("qty", 0.0) or 0.0)
                px = 0.0
                with_ohlc = getattr(self.ss, "latest_prices", {}) or {}
                px = float(with_ohlc.get(sym, 0.0) or 0.0) if isinstance(with_ohlc, dict) else 0.0
                if qty > 0 and px > 0:
                    value = qty * px
            return bool(value > 0 and value < threshold)
        except Exception:
            return False

    def authorize_velocity_exit(self, owned_positions: Dict[str, Any], current_metrics: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Layer 3: Evaluate if we need to force an exit because target velocity is not met.
        If profit/hr is low, we may exit low-conv positions to recycle capital.
        """
        run_rate = float(current_metrics.get("run_rate", 0.0))
        nav = float(current_metrics.get("nav_quote", 0.0) or getattr(self.ss, "_total_value", 0.0) or 0.0)
        target_velocity = max(0.0, nav * self.target_velocity_ratio)
        if run_rate >= target_velocity:
            return None # Velocity target met
            
        # If we are underperforming velocity, find the lowest-ALPHA position to recycle
        if not owned_positions:
            return None
            
        # Find position with lowest unrealized PnL or highest age with low profit
        candidates = []
        now = time.time()
        
        for sym, pos in owned_positions.items():
            if self._is_permanent_dust_position(sym, pos):
                continue
            if pos.get("state") == "EXITING":
                continue
                
            pnl = float(pos.get("unrealized_pnl_pct", 0.0) or 0.0)
            entry_ts = float(pos.get("entry_time") or pos.get("opened_at") or now)
            age_hr = (now - entry_ts) / 3600.0
            
            # Recyclability score: higher = better for recycling
            # (Young positions or high-PnL winners are NOT candidates)
            if age_hr > 0.5: # At least 30 mins hold
                recycle_score = (1.0 - pnl) * (age_hr / 1.0)
                candidates.append((sym, recycle_score))
                
        if not candidates:
            return None
            
        worst_sym, highest_score = max(candidates, key=lambda x: x[1])
        
        # Authority check: only recycle if score is high enough
        if highest_score > 1.2: # Tuneable threshold
            self.logger.warning(
                "[PortfolioAuth:Velocity] 🔄 RECYCLING CAPITAL: %s (score=%.2f) - Below target velocity ($%.2f/hr)",
                worst_sym, highest_score, target_velocity
            )
            return {
                "symbol": worst_sym,
                "action": "SELL",
                "confidence": 1.0,
                "agent": "PortfolioAuthority",
                "reason": "VELOCITY_RECYCLING",
                "_forced_exit": True,
                "_is_recycling": True
            }
            
        return None

    def authorize_rebalance_exit(self, owned_positions: Dict[str, Any], nav: float) -> Optional[Dict[str, Any]]:
        """
        Layer 3: Authorize exits for portfolio rebalancing (e.g. over-concentration).
        """
        if nav <= 0: return None

        cap = _dynamic_exposure_cap(nav)
        self.logger.debug("[PortfolioAuth:Rebalance] DynamicExposure NAV=%.2f → cap=%.0f%%", nav, cap * 100)

        for sym, pos in owned_positions.items():
            if self._is_permanent_dust_position(sym, pos):
                continue
            val = float(pos.get("value_usdt", 0.0))
            concentration = val / nav

            if concentration > cap:
                self.logger.warning(
                    "[PortfolioAuth:Rebalance] ⚖️ CONCENTRATION ALERT: %s at %.1f%% (>%.0f%% cap for NAV=%.2f). Authorizing partial exit.",
                    sym, concentration * 100, cap * 100, nav
                )
                return {
                    "symbol": sym,
                    "action": "SELL",
                    "confidence": 1.0,
                    "agent": "PortfolioAuthority",
                    "reason": "CONCENTRATION_REBALANCE",
                    "_forced_exit": True,
                    "allow_partial": True,
                    "target_fraction": 0.5 # Sell half to rebalance
                }
        return None

    def authorize_profit_recycling(self, owned_positions: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Layer 3: Force exit of winners to recycle profit into new opportunities.
        Prevents "profit stagnation" where we sit on a winner for too long without compounding.
        """
        recycle_pnl_threshold = float(getattr(self.config, "RECYCLE_PNL_PCT", 0.02)) # 2% profit
        recycle_min_age_hr = float(getattr(self.config, "RECYCLE_MIN_AGE_HR", 1.0)) # 1 hour
        
        now = time.time()
        for sym, pos in owned_positions.items():
            if self._is_permanent_dust_position(sym, pos):
                continue
            if pos.get("state") == "EXITING":
                continue
                
            pnl = float(pos.get("unrealized_pnl_pct", 0.0) or 0.0)
            entry_ts = float(pos.get("entry_time") or pos.get("opened_at") or now)
            age_hr = (now - entry_ts) / 3600.0
            
            # If we have a decent profit and have held long enough, recycle it
            if pnl >= recycle_pnl_threshold and age_hr >= recycle_min_age_hr:
                self.logger.warning(
                    "[PortfolioAuth:Recycle] ♻️ PROFIT RECYCLING: %s at %.2f%% profit after %.1fh. "
                    "Locking in for rotation.",
                    sym, pnl * 100, age_hr
                )
                return {
                    "symbol": sym,
                    "action": "SELL",
                    "confidence": 1.0,
                    "agent": "PortfolioAuthority",
                    "reason": "PROFIT_RECYCLING",
                    "_forced_exit": True,
                    "_is_recycling": True
                }
        return None
