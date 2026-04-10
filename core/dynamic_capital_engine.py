"""
Dynamic Capital Management Engine — Professional-grade position sizing & capital allocation.

Implements best-practice formulas for modern trading systems:
  • Dynamic Capital Floor (volatility-scaled liquidity reserve)
  • Risk-Based Position Sizing (capital × risk / stop_loss_distance)
  • ML Confidence Position Scaling (0.5x—2.0x multiplier)
  • Dynamic Maximum Positions (capital-aware concurrency)
  • Dynamic Exposure Control (volatility-adjusted)
  • Capital Velocity Targets (profit per unit capital per hour)
  • Dynamic Profit Targets (ATR-based take-profit)

Wired into: SharedState, CapitalAllocator, ScalingManager
Constructor: AppContext with config, logger, shared_state, risk_manager

All thresholds overridable via .env (DYNAMIC_CAPITAL_*)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class DynamicCapitalConfig:
    """Runtime configuration for dynamic capital engine (production-grade)."""
    absolute_min_reserve: float = 10.0  # Absolute minimum liquidity (USDT)
    base_liquidity_ratio: float = 0.12  # 12% of NAV as base liquidity (hard-capped max)
    min_liquidity_ratio: float = 0.05  # Minimum 5% in high volatility (production best-practice)
    max_liquidity_ratio: float = 0.12  # Maximum 12% (hard cap, never exceeds this)
    low_volatility_threshold: float = 0.015  # ≤1.5% ATR = 12% reserve (very low vol)
    mid_volatility_threshold: float = 0.03  # ≤3.0% ATR = 8% reserve (low-mid vol)
    high_volatility_threshold: float = 0.05  # >5% ATR = 5% reserve (high vol) [deprecated, for reference]
    base_risk_per_trade: float = 0.01  # 1% of NAV per trade
    confidence_threshold: float = 0.60  # ML confidence threshold
    position_size_min_multiplier: float = 0.5  # 50% of base
    position_size_max_multiplier: float = 2.0  # 200% of base
    base_exposure_ratio: float = 0.20  # 20% NAV exposure target
    max_concurrent_positions: int = 5  # Maximum open positions
    concurrency_buffer: float = 1.5  # 50% buffer for slippage
    capital_velocity_target_hourly: float = 0.02  # 2% per hour target
    atr_take_profit_multiplier: float = 2.5  # 2.5x ATR for TP
    velocity_measurement_window_hours: float = 4.0  # Rolling window


class DynamicCapitalEngine:
    """
    Professional dynamic capital manager.
    
    Integrates with:
    - SharedState: NAV, volatility, position metrics
    - CapitalAllocator: Capital pool sizing
    - ScalingManager: Position quote calculation
    - RiskManager: Stop-loss and exposure validation
    """

    component_name = "DynamicCapitalEngine"

    def __init__(
        self,
        config: Any = None,
        shared_state: Any = None,
        risk_manager: Any = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.config = config
        self.shared_state = shared_state
        self.risk_manager = risk_manager
        self.logger = logger or logging.getLogger(self.component_name)

        # Load configuration from environment / config object
        self.cfg = self._load_config()

        # Metrics tracking
        self._capital_floor_cache: float = 0.0
        self._last_capital_floor_update: float = 0.0
        self._velocity_metrics: List[Dict[str, float]] = []
        self._last_velocity_calc: float = 0.0

        self.logger.info(
            "[DCE] Initialized — base_liquidity=%.1f%%, base_risk=%.2f%%, "
            "base_exposure=%.1f%%, cv_target=%.2f%%/hr",
            self.cfg.base_liquidity_ratio * 100,
            self.cfg.base_risk_per_trade * 100,
            self.cfg.base_exposure_ratio * 100,
            self.cfg.capital_velocity_target_hourly * 100,
        )

    async def start(self) -> None:
        """Start the Dynamic Capital Engine (P8 phase startup)."""
        self.logger.info("[DCE] Starting DynamicCapitalEngine")
        self._last_capital_floor_update = time.time()
        self._last_velocity_calc = time.time()
        self.logger.info("[DCE] DynamicCapitalEngine started successfully")

    # ═══════════════════════════════════════════════════════════════════════
    # PUBLIC API — Dynamic Capital Floor (Volatility-Scaled Liquidity Reserve)
    # ═══════════════════════════════════════════════════════════════════════

    async def compute_capital_floor(self) -> float:
        """
        Dynamic liquidity reserve: scales with volatility and position count.

        Formula:
            capital_floor = max(
                absolute_min,
                NAV × base_liquidity_ratio × volatility_multiplier,
                trade_size × concurrency_buffer
            )

        Returns:
            float: Minimum reserved liquidity (USDT)
        """
        try:
            now = time.time()
            # Recompute every 60 seconds or on explicit request
            if now - self._last_capital_floor_update < 60:
                return self._capital_floor_cache

            nav = await self._get_nav()
            if nav <= 0:
                return self.cfg.absolute_min_reserve

            volatility = await self._get_volatility_multiplier()
            active_positions = await self._get_active_position_count()
            avg_trade_size = await self._get_average_trade_size()

            # Component 1: Base liquidity (volatile-scaled)
            base_floor = nav * self.cfg.base_liquidity_ratio * volatility

            # Component 2: Concurrency buffer
            concurrency_floor = (
                avg_trade_size * self.cfg.concurrency_buffer * active_positions
            )

            # Component 3: Absolute minimum
            floor = max(
                self.cfg.absolute_min_reserve,
                base_floor,
                concurrency_floor,
            )

            self._capital_floor_cache = floor
            self._last_capital_floor_update = now

            self.logger.debug(
                "[DCE:Floor] NAV=$%.2f → vol=%.2fx, pos=%d, avg_trade=$%.2f → "
                "floor=$%.2f (base=$%.2f, concurrency=$%.2f)",
                nav,
                volatility,
                active_positions,
                avg_trade_size,
                floor,
                base_floor,
                concurrency_floor,
            )

            return floor

        except Exception as e:
            self.logger.warning("[DCE:Floor] Computation failed: %s — return minimum", e)
            return self.cfg.absolute_min_reserve

    # ═══════════════════════════════════════════════════════════════════════
    # PUBLIC API — Risk-Based Position Sizing
    # ═══════════════════════════════════════════════════════════════════════

    async def compute_position_size(
        self,
        symbol: str,
        stop_loss_distance_pct: float,
        nav: Optional[float] = None,
        risk_per_trade_override: Optional[float] = None,
    ) -> float:
        """
        Calculate position size based on risk allocation.

        Formula:
            position_size = (NAV × risk_per_trade) / stop_loss_distance

        Args:
            symbol: Trading symbol (BTCUSDT, ETHUSDT, etc.)
            stop_loss_distance_pct: Distance to SL as % (0.02 = 2%)
            nav: Override NAV (uses shared_state if None)
            risk_per_trade_override: Override risk fraction (uses config if None)

        Returns:
            float: Position size in USDT
        """
        try:
            if nav is None:
                nav = await self._get_nav()
            if nav <= 0:
                return 0.0

            if stop_loss_distance_pct <= 0:
                self.logger.warning(
                    "[DCE:PosSizing] Invalid SL distance %.2f%% for %s",
                    stop_loss_distance_pct * 100,
                    symbol,
                )
                return 0.0

            risk_fraction = risk_per_trade_override or self.cfg.base_risk_per_trade
            risk_budget = nav * risk_fraction
            position_size = risk_budget / stop_loss_distance_pct

            self.logger.debug(
                "[DCE:PosSizing] %s: NAV=$%.2f, risk=%.2f%%, SL=%.2f%% → size=$%.2f",
                symbol,
                nav,
                risk_fraction * 100,
                stop_loss_distance_pct * 100,
                position_size,
            )

            return position_size

        except Exception as e:
            self.logger.warning("[DCE:PosSizing] Sizing failed for %s: %s", symbol, e)
            return 0.0

    # ═══════════════════════════════════════════════════════════════════════
    # PUBLIC API — ML Confidence Position Scaling
    # ═══════════════════════════════════════════════════════════════════════

    def compute_confidence_multiplier(
        self, ml_confidence: float, confidence_threshold: Optional[float] = None
    ) -> float:
        """
        Scale position size by ML model confidence.

        Formula:
            multiplier = min(2.0, max(0.5, confidence / threshold))

        Args:
            ml_confidence: Model confidence [0.0, 1.0]
            confidence_threshold: Scaling threshold (uses config if None)

        Returns:
            float: Multiplier [0.5, 2.0]
        """
        try:
            threshold = confidence_threshold or self.cfg.confidence_threshold
            if threshold <= 0:
                return 1.0

            multiplier = ml_confidence / threshold

            # Clamp to [0.5, 2.0]
            multiplier = max(
                self.cfg.position_size_min_multiplier,
                min(self.cfg.position_size_max_multiplier, multiplier),
            )

            self.logger.debug(
                "[DCE:Confidence] confidence=%.2f%%, threshold=%.2f%% → multiplier=%.2fx",
                ml_confidence * 100,
                threshold * 100,
                multiplier,
            )

            return multiplier

        except Exception as e:
            self.logger.warning("[DCE:Confidence] Calculation failed: %s", e)
            return 1.0

    # ═══════════════════════════════════════════════════════════════════════
    # PUBLIC API — Dynamic Maximum Positions
    # ═══════════════════════════════════════════════════════════════════════

    async def compute_max_positions(self, nav: Optional[float] = None) -> int:
        """
        Calculate maximum concurrent positions scaled by capital.

        Formula:
            max_positions = floor((NAV × exposure_ratio) / position_size)

        With professional constraint: capital-dependent concurrency limits.

        Args:
            nav: Override NAV (uses shared_state if None)

        Returns:
            int: Maximum concurrent open positions
        """
        try:
            if nav is None:
                nav = await self._get_nav()
            if nav <= 0:
                return 1

            # Professional bracket-based limits
            if nav < 500:
                max_positions = 1  # Micro
            elif nav < 2000:
                max_positions = 2  # Small
            elif nav < 10000:
                max_positions = 3  # Medium
            else:
                max_positions = self.cfg.max_concurrent_positions  # Large

            self.logger.debug(
                "[DCE:MaxPos] NAV=$%.2f → max_concurrent=%d",
                nav,
                max_positions,
            )

            return max_positions

        except Exception as e:
            self.logger.warning("[DCE:MaxPos] Calculation failed: %s", e)
            return 1

    # ═══════════════════════════════════════════════════════════════════════
    # PUBLIC API — Dynamic Exposure Control (Volatility-Adjusted)
    # ═══════════════════════════════════════════════════════════════════════

    async def compute_exposure_ratio(self) -> float:
        """
        Adjust target exposure based on volatility regime.

        Formula:
            exposure_ratio = base_exposure × volatility_adjustment

        Where:
            - Low volatility (VIX < 15):   1.2x increase (aggressive)
            - Normal volatility (15-25):   1.0x baseline
            - High volatility (> 25):      0.7x reduction (defensive)

        Returns:
            float: Target exposure as fraction of NAV
        """
        try:
            volatility_mult = await self._get_volatility_multiplier()
            base = self.cfg.base_exposure_ratio

            # Interpret volatility as market regime
            if volatility_mult < 0.9:
                # Low volatility: more aggressive
                adjustment = 1.2
            elif volatility_mult > 1.1:
                # High volatility: more defensive
                adjustment = 0.7
            else:
                # Normal: baseline
                adjustment = 1.0

            exposure = base * adjustment

            self.logger.debug(
                "[DCE:Exposure] vol_mult=%.2fx, adjustment=%.2fx → exposure=%.1f%%",
                volatility_mult,
                adjustment,
                exposure * 100,
            )

            return exposure

        except Exception as e:
            self.logger.warning("[DCE:Exposure] Calculation failed: %s — using base", e)
            return self.cfg.base_exposure_ratio

    # ═══════════════════════════════════════════════════════════════════════
    # PUBLIC API — Capital Velocity Target
    # ═══════════════════════════════════════════════════════════════════════

    async def get_capital_velocity_target(self) -> Dict[str, float]:
        """
        Get current capital velocity and target.

        Formula:
            capital_velocity = profit / (capital × time)

        Returns:
            Dict with:
            {
                "current_velocity": float,      # % return per hour
                "target_velocity": float,       # % return per hour
                "efficiency_ratio": float,      # current / target
                "measurement_period_hours": float,
            }
        """
        try:
            nav = await self._get_nav()
            if nav <= 0:
                return {
                    "current_velocity": 0.0,
                    "target_velocity": self.cfg.capital_velocity_target_hourly,
                    "efficiency_ratio": 0.0,
                    "measurement_period_hours": self.cfg.velocity_measurement_window_hours,
                }

            realized_pnl = await self._get_realized_pnl()
            elapsed_hours = self.cfg.velocity_measurement_window_hours
            current_velocity = (realized_pnl / nav) / elapsed_hours if elapsed_hours > 0 else 0.0

            target_velocity = self.cfg.capital_velocity_target_hourly
            efficiency_ratio = (
                current_velocity / target_velocity
                if target_velocity > 0
                else 1.0
            )

            self.logger.debug(
                "[DCE:Velocity] NAV=$%.2f, PnL=$%.2f over %.1fh → "
                "current=%.2f%%/h, target=%.2f%%/h, efficiency=%.1f%%",
                nav,
                realized_pnl,
                elapsed_hours,
                current_velocity * 100,
                target_velocity * 100,
                efficiency_ratio * 100,
            )

            return {
                "current_velocity": current_velocity,
                "target_velocity": target_velocity,
                "efficiency_ratio": efficiency_ratio,
                "measurement_period_hours": elapsed_hours,
            }

        except Exception as e:
            self.logger.warning("[DCE:Velocity] Calculation failed: %s", e)
            return {
                "current_velocity": 0.0,
                "target_velocity": self.cfg.capital_velocity_target_hourly,
                "efficiency_ratio": 0.0,
                "measurement_period_hours": self.cfg.velocity_measurement_window_hours,
            }

    # ═══════════════════════════════════════════════════════════════════════
    # PUBLIC API — Dynamic Profit Targets (ATR-Based)
    # ═══════════════════════════════════════════════════════════════════════

    async def compute_take_profit_target(
        self,
        symbol: str,
        entry_price: float,
        atr_pct: float,
    ) -> float:
        """
        Calculate take-profit price using ATR-based scaling.

        Formula:
            take_profit = entry_price + (ATR × multiplier)

        Args:
            symbol: Trading symbol
            entry_price: Entry price (USDT)
            atr_pct: ATR as % of price (0.03 = 3%)

        Returns:
            float: Take-profit price (USDT)
        """
        try:
            if entry_price <= 0 or atr_pct <= 0:
                self.logger.warning(
                    "[DCE:TP] Invalid input: price=$%.2f, ATR=%.2f%%",
                    entry_price,
                    atr_pct * 100,
                )
                return 0.0

            volatility_mult = await self._get_volatility_multiplier()
            effective_multiplier = self.cfg.atr_take_profit_multiplier * volatility_mult
            atr_amount = entry_price * atr_pct
            take_profit = entry_price + (atr_amount * effective_multiplier)

            self.logger.debug(
                "[DCE:TP] %s: entry=$%.2f, ATR=%.2f%%, vol_mult=%.2fx → TP=$%.2f (%.2f%%)",
                symbol,
                entry_price,
                atr_pct * 100,
                volatility_mult,
                take_profit,
                ((take_profit - entry_price) / entry_price) * 100,
            )

            return take_profit

        except Exception as e:
            self.logger.warning("[DCE:TP] Calculation failed for %s: %s", symbol, e)
            return 0.0

    # ═══════════════════════════════════════════════════════════════════════
    # INTEGRATED BEST-PRACTICE SYSTEM
    # ═══════════════════════════════════════════════════════════════════════

    async def compute_integrated_position_plan(
        self,
        symbol: str,
        stop_loss_distance_pct: float,
        entry_price: float,
        atr_pct: float,
        ml_confidence: float,
        nav: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Compute complete position plan using all best-practice formulas.

        Returns integrated plan:
        {
            "symbol": str,
            "position_size_usd": float,
            "confidence_multiplier": float,
            "final_position_size": float,
            "take_profit_price": float,
            "max_concurrent_positions": int,
            "capital_floor_reserve": float,
            "exposure_ratio": float,
            "capital_velocity_info": dict,
            "timestamp": float,
        }
        """
        try:
            if nav is None:
                nav = await self._get_nav()
            if nav <= 0:
                return {"error": "NAV <= 0", "symbol": symbol}

            # 1. Risk-based position size
            base_position_size = await self.compute_position_size(
                symbol, stop_loss_distance_pct, nav
            )

            # 2. ML confidence scaling
            confidence_mult = self.compute_confidence_multiplier(ml_confidence)

            # 3. Final position size
            final_position_size = base_position_size * confidence_mult

            # 4. Dynamic profit target
            take_profit = await self.compute_take_profit_target(
                symbol, entry_price, atr_pct
            )

            # 5. Dynamic position limits
            max_positions = await self.compute_max_positions(nav)

            # 6. Dynamic capital floor
            capital_floor = await self.compute_capital_floor()

            # 7. Dynamic exposure
            exposure = await self.compute_exposure_ratio()

            # 8. Capital velocity
            velocity = await self.get_capital_velocity_target()

            plan = {
                "symbol": symbol,
                "position_size_usd": base_position_size,
                "confidence_multiplier": confidence_mult,
                "final_position_size": final_position_size,
                "take_profit_price": take_profit,
                "max_concurrent_positions": max_positions,
                "capital_floor_reserve": capital_floor,
                "exposure_ratio": exposure,
                "capital_velocity_info": velocity,
                "timestamp": time.time(),
            }

            self.logger.info(
                "[DCE:Plan] %s: size=$%.2f × %.2f = $%.2f, TP=$%.2f, "
                "max_pos=%d, floor=$%.2f, exposure=%.1f%%",
                symbol,
                base_position_size,
                confidence_mult,
                final_position_size,
                take_profit,
                max_positions,
                capital_floor,
                exposure * 100,
            )

            return plan

        except Exception as e:
            self.logger.error("[DCE:Plan] Plan computation failed for %s: %s", symbol, e)
            return {"error": str(e), "symbol": symbol}

    # ═══════════════════════════════════════════════════════════════════════
    # DYNAMIC LIQUIDITY RATIO (Volatility-Based)
    # ═══════════════════════════════════════════════════════════════════════

    async def get_dynamic_liquidity_ratio(self) -> float:
        """
        Calculate dynamic liquidity ratio based on current market volatility.
        
        Production-grade implementation with hard caps (professional trading engine best practice).
        
        Logic:
        - Very low volatility (≤1.5% ATR):  12% reserve (growth-oriented)
        - Low-mid volatility (≤3.0% ATR):   8% reserve (balanced)
        - High volatility (>3.0% ATR):      5% reserve (agility)
        - HARD CAP: ratio = min(calculated_ratio, 0.12) to prevent over-reservation
        
        This matches production trading engine configurations that never allow
        volatility classifiers to push reserves above ~15% automatically.
        
        Returns:
            float: Dynamic liquidity ratio (0.05 to 0.12, with hard cap at 0.12)
        """
        try:
            volatility = await self._get_volatility_atr()
            
            # Calculate ratio based on volatility tiers
            if volatility <= 0.015:  # ≤1.5% ATR
                ratio = 0.12
                reason = "very low volatility"
            elif volatility <= 0.03:  # ≤3.0% ATR
                ratio = 0.08
                reason = "low-mid volatility"
            else:  # >3.0% ATR
                ratio = 0.05
                reason = "high volatility"
            
            # HARD CAP: Never exceed 12% reserve (professional best practice)
            # This prevents volatility classifier from being too conservative
            ratio = min(ratio, 0.12)
            
            self.logger.debug(
                "[DCE:Ratio] Volatility=%.3f%% (%.2f ATR) → ratio=%.2f%% (%s, capped)",
                volatility * 100,
                volatility,
                ratio * 100,
                reason,
            )
            
            return ratio
            
        except Exception as e:
            self.logger.warning("[DCE:Ratio] Calculation failed: %s — using baseline 0.12", e)
            return 0.12  # Production-grade fallback: 12% (hard-capped baseline)

    # ═══════════════════════════════════════════════════════════════════════
    # INTERNAL HELPERS
    # ═══════════════════════════════════════════════════════════════════════

    def _load_config(self) -> DynamicCapitalConfig:
        """Load configuration from environment."""
        cfg = DynamicCapitalConfig()

        # Override with config object if present
        if self.config:
            cfg.absolute_min_reserve = float(
                getattr(self.config, "DYNAMIC_CAPITAL_ABSOLUTE_MIN_RESERVE", cfg.absolute_min_reserve) or cfg.absolute_min_reserve
            )
            cfg.base_liquidity_ratio = float(
                getattr(self.config, "DYNAMIC_CAPITAL_BASE_LIQUIDITY_RATIO", cfg.base_liquidity_ratio) or cfg.base_liquidity_ratio
            )
            cfg.base_risk_per_trade = float(
                getattr(self.config, "DYNAMIC_CAPITAL_BASE_RISK_PER_TRADE", cfg.base_risk_per_trade) or cfg.base_risk_per_trade
            )
            cfg.confidence_threshold = float(
                getattr(self.config, "DYNAMIC_CAPITAL_CONFIDENCE_THRESHOLD", cfg.confidence_threshold) or cfg.confidence_threshold
            )
            cfg.capital_velocity_target_hourly = float(
                getattr(self.config, "DYNAMIC_CAPITAL_VELOCITY_TARGET_HOURLY", cfg.capital_velocity_target_hourly) or cfg.capital_velocity_target_hourly
            )
            cfg.atr_take_profit_multiplier = float(
                getattr(self.config, "DYNAMIC_CAPITAL_ATR_TP_MULTIPLIER", cfg.atr_take_profit_multiplier) or cfg.atr_take_profit_multiplier
            )

        return cfg

    async def _get_nav(self) -> float:
        """Get NAV from shared_state."""
        try:
            if hasattr(self.shared_state, "get_nav"):
                val = self.shared_state.get_nav()
                if hasattr(val, "__await__"):
                    val = await val
                return float(val or 0.0)
        except Exception:
            pass
        return 0.0

    async def _get_volatility_multiplier(self) -> float:
        """
        Get volatility adjustment multiplier.
        
        Returns:
            float: 1.0 = normal, <1.0 = low vol (aggressive), >1.0 = high vol (defensive)
        """
        try:
            # Try to get ATR or volatility from shared_state
            if hasattr(self.shared_state, "metrics"):
                metrics = self.shared_state.metrics or {}
                volatility = float(metrics.get("volatility_pct", 0.03) or 0.03)
                baseline = self.cfg.base_volatility_threshold

                # Volatility multiplier: higher vol = higher multiplier
                multiplier = volatility / baseline if baseline > 0 else 1.0
                return max(0.5, min(2.0, multiplier))  # Clamp [0.5, 2.0]
        except Exception:
            pass
        return 1.0

    async def _get_active_position_count(self) -> int:
        """Get number of active open positions."""
        try:
            if hasattr(self.shared_state, "get_active_positions"):
                val = self.shared_state.get_active_positions()
                if hasattr(val, "__await__"):
                    val = await val
                return int(val or 0)
            
            # Fallback: check portfolio
            if hasattr(self.shared_state, "portfolio"):
                portfolio = self.shared_state.portfolio or {}
                return len(portfolio.get("positions", {}))
        except Exception:
            pass
        return 0

    async def _get_average_trade_size(self) -> float:
        """Get average position size from recent trades."""
        try:
            if hasattr(self.shared_state, "metrics"):
                metrics = self.shared_state.metrics or {}
                avg_size = float(metrics.get("avg_position_size", 50.0) or 50.0)
                return avg_size
        except Exception:
            pass
        return 50.0  # Default fallback

    async def _get_realized_pnl(self) -> float:
        """Get realized P&L."""
        try:
            if hasattr(self.shared_state, "metrics"):
                metrics = self.shared_state.metrics or {}
                pnl = float(metrics.get("realized_pnl", 0.0) or 0.0)
                return pnl
        except Exception:
            pass
        return 0.0

    async def _get_volatility_atr(self) -> float:
        """
        Get current volatility as ATR percentage.
        
        Returns:
            float: Volatility as percentage (0.03 = 3%)
        """
        try:
            if hasattr(self.shared_state, "metrics"):
                metrics = self.shared_state.metrics or {}
                volatility = float(metrics.get("volatility_pct", self.cfg.base_volatility_threshold) or self.cfg.base_volatility_threshold)
                return volatility
            
            # Fallback to base threshold
            return self.cfg.base_volatility_threshold
        except Exception:
            pass
        return self.cfg.base_volatility_threshold
