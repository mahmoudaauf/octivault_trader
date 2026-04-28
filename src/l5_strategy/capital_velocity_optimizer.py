# -*- coding: utf-8 -*-
"""
Capital Velocity Optimizer - Institutional Capital Velocity Coordination

Bridges the gap between velocity governance (PortfolioAuthority, RotationAuthority)
and velocity optimization (forward-looking allocation planning).

This module does NOT implement new exit logic. Instead, it:
1. MEASURES real-time position velocity
2. ESTIMATES opportunity velocity (from ML signals + metrics)
3. OPTIMIZES capital allocation across positions
4. COORDINATES rotation recommendations

Think of it as the conductor that orchestrates existing velocity instruments.

Core Philosophy:
  - Reactive velocity control already exists (PortfolioAuthority, RotationAuthority)
  - This optimizer adds PROACTIVE capital allocation
  - All decisions feed back to MetaController via recommendations
  - Leverages existing modules; adds no direct exit authority

Architecture:
  Position Velocity (realized)  ←── PortfolioAuthority
  Opportunity Velocity (ML forecast)  ←── MLForecaster signals
           ↓
    Capital Velocity Optimizer
           ↓
  Rotation Recommendations  ──→ MetaController
"""

import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class PositionVelocityMetric:
    """Real-time velocity snapshot for a position."""
    symbol: str
    pnl_pct: float              # Unrealized P&L %
    entry_time: float           # Unix timestamp
    age_hours: float            # How long held
    pnl_per_hour: float         # P&L % / hours held
    holding_cost_bps: float     # Fees as basis points per hour
    net_velocity: float         # P&L per hour minus holding cost
    is_recyclable: bool         # Can be exited for better opportunity


@dataclass
class OpportunityVelocityMetric:
    """Estimated velocity for a candidate symbol."""
    symbol: str
    expected_return_pct: float   # ML-based edge estimate
    expected_move_pct: float     # Volatility/movement estimate from ML
    ml_confidence: float         # MLForecaster confidence (0-1)
    estimated_velocity_pct: float  # Expected return per hour
    liquidity_score: float       # How easy to enter/exit (0-1)
    time_to_breakeven_hours: float  # How long to recover fees


@dataclass
class VelocityOptimizationPlan:
    """Output: structured recommendation for capital reallocation."""
    timestamp: float
    portfolio_velocity_pct_per_hour: float   # Current weighted average
    opportunity_velocity_pct_per_hour: float # Best available opportunity
    velocity_gap: float                       # Opportunity - Portfolio
    rotations_recommended: List[Dict[str, Any]]  # [{"exit_symbol": X, "reason": Y, ...}]
    hold_positions: List[str]                 # Symbols to keep
    analysis: Dict[str, Any]                 # Debug metrics


class CapitalVelocityOptimizer:
    """
    Institutional capital velocity coordinator.
    
    Responsibilities:
    1. Measure position velocity (realized P&L per unit time)
    2. Estimate opportunity velocity (forward-looking from ML + market metrics)
    3. Identify recyclable positions (low velocity relative to opportunities)
    4. Recommend optimal rotation timing
    
    Design:
    - Reads from existing authorities (PortfolioAuthority, MLForecaster, performance metrics)
    - Outputs structured recommendations to MetaController
    - Does NOT execute; purely advisory
    """

    def __init__(self, config: Any, shared_state: Any, logger: Optional[logging.Logger] = None):
        """
        Args:
            config: Configuration object
            shared_state: SharedState for position, signal, and metric access
            logger: Optional logger instance
        """
        self.config = config
        self.ss = shared_state
        self.logger = logger or logging.getLogger("CapitalVelocityOptimizer")

        # Tuning parameters
        self.velocity_gap_threshold_pct = float(
            getattr(config, "VELOCITY_GAP_THRESHOLD_PCT", 0.5)  # 0.5% per hour
        )
        self.min_position_age_hours = float(
            getattr(config, "VELOCITY_MIN_POSITION_AGE_HOURS", 0.25)  # 15 minutes
        )
        self.holding_cost_fee_bps = float(
            getattr(config, "VELOCITY_HOLDING_COST_FEE_BPS", 10.0)  # 10 bps per trade
        )
        self.velocity_confidence_min = float(
            getattr(config, "VELOCITY_CONFIDENCE_MIN", 0.55)
        )
        self.enable_velocity_optimization = bool(
            getattr(config, "ENABLE_CAPITAL_VELOCITY_OPTIMIZATION", True)
        )

    # ════════════════════════════════════════════════════════════════════════════════
    # SECTION A: POSITION VELOCITY MEASUREMENT
    # ════════════════════════════════════════════════════════════════════════════════

    def evaluate_position_velocity(self, symbol: str, position: Dict[str, Any], now: float) -> PositionVelocityMetric:
        """
        Measure real-time velocity of an open position.
        
        Args:
            symbol: Position symbol
            position: Position dict from SharedState
            now: Current timestamp (unix)
            
        Returns:
            PositionVelocityMetric with velocity components
        """
        pnl_pct = float((position or {}).get("unrealized_pnl_pct", 0.0) or 0.0)
        entry_time = float((position or {}).get("entry_time", 0.0) or (position or {}).get("opened_at", 0.0) or now)
        age_hours = max(0.001, (now - entry_time) / 3600.0)

        # P&L per hour (annualized velocity)
        pnl_per_hour = pnl_pct / age_hours if age_hours > 0 else 0.0

        # Estimate holding cost: round-trip fees (~20 bps, amortized over hold time)
        # For a 15-min hold: ~20bps / 4 = 5bps/hour
        # For a 2-hour hold: ~20bps / 2 = 10bps/hour
        holding_cost_bps = (self.holding_cost_fee_bps * 100.0) / max(1.0, age_hours)

        # Net velocity: P&L rate minus holding cost
        net_velocity = pnl_per_hour - (holding_cost_bps / 10000.0)

        # Recyclability: can exit if negative net velocity
        is_recyclable = net_velocity < 0 and age_hours >= self.min_position_age_hours

        return PositionVelocityMetric(
            symbol=symbol,
            pnl_pct=pnl_pct,
            entry_time=entry_time,
            age_hours=age_hours,
            pnl_per_hour=pnl_per_hour,
            holding_cost_bps=holding_cost_bps,
            net_velocity=net_velocity,
            is_recyclable=is_recyclable,
        )

    async def measure_portfolio_velocity(
        self, owned_positions: Dict[str, Any]
    ) -> Tuple[float, Dict[str, PositionVelocityMetric]]:
        """
        Measure aggregate portfolio velocity.
        
        Returns:
            (weighted_avg_velocity_pct_per_hour, position_metrics_dict)
        """
        if not owned_positions:
            return 0.0, {}

        now = time.time()
        position_metrics = {}
        total_exposure = 0.0
        weighted_velocity = 0.0

        for symbol, pos in owned_positions.items():
            metric = self.evaluate_position_velocity(symbol, pos, now)
            position_metrics[symbol] = metric

            # Weight by position value if available
            exposure = float((pos or {}).get("value_usdt", 0.0) or 0.0)
            if exposure > 0:
                total_exposure += exposure
                weighted_velocity += metric.net_velocity * exposure

        if total_exposure > 0:
            avg_velocity = weighted_velocity / total_exposure
        else:
            avg_velocity = 0.0

        return avg_velocity, position_metrics

    # ════════════════════════════════════════════════════════════════════════════════
    # SECTION B: OPPORTUNITY VELOCITY ESTIMATION
    # ════════════════════════════════════════════════════════════════════════════════

    def estimate_opportunity_velocity(
        self, symbol: str, ml_signal: Dict[str, Any]
    ) -> Optional[OpportunityVelocityMetric]:
        """
        Estimate velocity potential of a candidate symbol using ML signal data.
        
        Uses MLForecaster outputs:
        - confidence: probability of direction correctness
        - _expected_move_pct: volatility / movement estimate
        - action: BUY/SELL direction
        
        Formula:
          expected_return = confidence * expected_move_pct
          estimated_velocity = expected_return / time_to_achieve
          
        Args:
            symbol: Candidate symbol
            ml_signal: Signal dict from MLForecaster (must contain confidence, _expected_move_pct)
            
        Returns:
            OpportunityVelocityMetric or None if signal too weak
        """
        if not ml_signal:
            return None

        ml_confidence = float((ml_signal or {}).get("confidence", 0.0) or 0.0)
        expected_move_pct = float((ml_signal or {}).get("_expected_move_pct", 0.0) or 0.0)

        # Filter weak signals
        if ml_confidence < self.velocity_confidence_min:
            self.logger.debug(
                f"[OpportVelocity] {symbol} confidence {ml_confidence:.2f} < {self.velocity_confidence_min:.2f}; skipping"
            )
            return None

        if expected_move_pct <= 0:
            return None

        # Expected return = confidence * expected_move
        # This is institutional forecast methodology
        expected_return_pct = ml_confidence * expected_move_pct

        # Time to achieve expected move (estimate from recent volatility)
        # Typical regime: 15min - 2hour target
        # Assume 1 hour on average as planning horizon
        time_to_achieve_hours = 1.0

        # Estimated velocity: return per hour
        estimated_velocity_pct = expected_return_pct / time_to_achieve_hours

        # Liquidity score (placeholder; could integrate from market_data or config)
        liquidity_score = 0.85  # Default assumption

        # Time to breakeven (fees recovered)
        holding_cost_pct = self.holding_cost_fee_bps / 10000.0
        if expected_return_pct > 0:
            time_to_breakeven = holding_cost_pct / expected_return_pct
        else:
            time_to_breakeven = float("inf")

        return OpportunityVelocityMetric(
            symbol=symbol,
            expected_return_pct=expected_return_pct,
            expected_move_pct=expected_move_pct,
            ml_confidence=ml_confidence,
            estimated_velocity_pct=estimated_velocity_pct,
            liquidity_score=liquidity_score,
            time_to_breakeven_hours=time_to_breakeven,
        )

    async def estimate_universe_opportunity(
        self, candidate_symbols: List[str]
    ) -> Dict[str, OpportunityVelocityMetric]:
        """
        Estimate velocity for all candidate symbols using latest ML signals.
        
        Args:
            candidate_symbols: List of symbols to evaluate
            
        Returns:
            Dict[symbol] -> OpportunityVelocityMetric (only for viable candidates)
        """
        opportunities = {}

        # Fetch latest ML signals from SharedState
        signals = {}
        if hasattr(self.ss, "latest_ml_signals"):
            signals = getattr(self.ss, "latest_ml_signals", {}) or {}
        elif hasattr(self.ss, "strategy_signals"):
            signals = getattr(self.ss, "strategy_signals", {}) or {}

        for symbol in candidate_symbols:
            # Get latest BUY signal for this symbol (preference for entry opportunities)
            signal = signals.get(symbol) or {}
            if not signal:
                self.logger.debug(f"[OpportVelocity] No signal for {symbol}; skipping estimate")
                continue

            # Only consider BUY signals for this analysis
            if str((signal or {}).get("action", "")).upper() != "BUY":
                continue

            metric = self.estimate_opportunity_velocity(symbol, signal)
            if metric:
                opportunities[symbol] = metric

        return opportunities

    # ════════════════════════════════════════════════════════════════════════════════
    # SECTION C: OPTIMAL HOLD TIME ESTIMATION
    # ════════════════════════════════════════════════════════════════════════════════

    def estimate_optimal_exit_time(
        self, position_metric: PositionVelocityMetric
    ) -> Dict[str, Any]:
        """
        Estimate when to exit a position based on velocity and aging.
        
        Simple heuristic:
        - If velocity is positive and recent (< 1 hour): hold
        - If velocity turns negative: consider exit
        - If velocity stagnates for extended period: rotate
        
        Returns dict with recommendation and rationale.
        """
        hold_recommendation = {
            "action": "HOLD",
            "reason": "positive_velocity",
            "confidence": 0.0,
            "estimated_hold_hours": 2.0,
        }

        # Rule 1: Negative velocity → exit soon
        if position_metric.net_velocity < 0:
            return {
                "action": "EXIT",
                "reason": "negative_velocity",
                "confidence": 0.8,
                "estimated_hold_hours": 0.25,  # 15 minutes
                "net_velocity": position_metric.net_velocity,
            }

        # Rule 2: Positive but decelerating velocity → watch
        if 0 < position_metric.net_velocity < 0.001:  # < 0.1% per hour
            return {
                "action": "WATCH",
                "reason": "low_velocity",
                "confidence": 0.6,
                "estimated_hold_hours": 1.0,
                "net_velocity": position_metric.net_velocity,
            }

        # Rule 3: Healthy velocity → hold
        return {
            "action": "HOLD",
            "reason": "healthy_velocity",
            "confidence": 0.9,
            "estimated_hold_hours": 2.0,
            "net_velocity": position_metric.net_velocity,
        }

    # ════════════════════════════════════════════════════════════════════════════════
    # SECTION D: PORTFOLIO CAPITAL BALANCING & RECOMMENDATIONS
    # ════════════════════════════════════════════════════════════════════════════════

    def recommend_rotation(
        self,
        portfolio_metrics: Dict[str, PositionVelocityMetric],
        opportunity_metrics: Dict[str, OpportunityVelocityMetric],
        portfolio_velocity_avg: float,
    ) -> List[Dict[str, Any]]:
        """
        Generate rotation recommendations by comparing position vs opportunity velocity.
        
        Logic:
        1. Identify low-velocity positions (recyclable)
        2. Compare vs best opportunities
        3. If velocity_gap > threshold, recommend rotation
        
        Args:
            portfolio_metrics: Position velocity metrics
            opportunity_metrics: Candidate velocity estimates
            portfolio_velocity_avg: Weighted portfolio velocity
            
        Returns:
            List of rotation recommendations [{exit_symbol, opportunity_symbol, gap, confidence}, ...]
        """
        recommendations = []

        if not opportunity_metrics:
            return recommendations

        # Sort opportunities by velocity (descending)
        sorted_opportunities = sorted(
            opportunity_metrics.items(),
            key=lambda x: x[1].estimated_velocity_pct,
            reverse=True,
        )

        best_opportunity = sorted_opportunities[0] if sorted_opportunities else None
        if not best_opportunity:
            return recommendations

        best_opp_symbol, best_opp_metric = best_opportunity

        # Find recyclable positions
        recyclable = [
            (sym, metric)
            for sym, metric in portfolio_metrics.items()
            if metric.is_recyclable
        ]

        if not recyclable:
            self.logger.debug("[VelocityOptimizer] No recyclable positions for rotation")
            return recommendations

        # Sort by velocity (ascending) → worst first
        recyclable.sort(key=lambda x: x[1].net_velocity)

        for position_sym, position_metric in recyclable:
            velocity_gap = best_opp_metric.estimated_velocity_pct - position_metric.net_velocity

            if velocity_gap > (self.velocity_gap_threshold_pct / 100.0):
                recommendations.append(
                    {
                        "exit_symbol": position_sym,
                        "opportunity_symbol": best_opp_symbol,
                        "velocity_gap_pct_per_hour": velocity_gap * 100.0,
                        "current_velocity_pct": position_metric.net_velocity * 100.0,
                        "opportunity_velocity_pct": best_opp_metric.estimated_velocity_pct * 100.0,
                        "reason": "VELOCITY_OPTIMIZATION_GAP",
                        "confidence": best_opp_metric.ml_confidence,
                        "position_age_hours": position_metric.age_hours,
                    }
                )

        return recommendations

    # ════════════════════════════════════════════════════════════════════════════════
    # SECTION E: MAIN COORDINATION INTERFACE
    # ════════════════════════════════════════════════════════════════════════════════

    async def optimize_capital_velocity(
        self,
        owned_positions: Dict[str, Any],
        candidate_symbols: List[str],
    ) -> VelocityOptimizationPlan:
        """
        Main entry point: coordinate all velocity measurements and generate optimization plan.
        
        Args:
            owned_positions: Dict of current positions from SharedState
            candidate_symbols: List of symbols being considered for entry
            
        Returns:
            VelocityOptimizationPlan with recommendations
        """
        if not self.enable_velocity_optimization:
            return VelocityOptimizationPlan(
                timestamp=time.time(),
                portfolio_velocity_pct_per_hour=0.0,
                opportunity_velocity_pct_per_hour=0.0,
                velocity_gap=0.0,
                rotations_recommended=[],
                hold_positions=[],
                analysis={"disabled": True},
            )

        try:
            # Step 1: Measure current portfolio velocity
            portfolio_velocity_avg, position_metrics = await self.measure_portfolio_velocity(owned_positions)

            # Step 2: Estimate opportunity velocity
            opportunity_metrics = await self.estimate_universe_opportunity(candidate_symbols)

            # Step 3: Generate recommendations
            rotations = self.recommend_rotation(
                position_metrics, opportunity_metrics, portfolio_velocity_avg
            )

            # Step 4: Identify holds
            held_symbols = list(owned_positions.keys())

            # Step 5: Build analysis
            best_opp_velocity = (
                max((m.estimated_velocity_pct for m in opportunity_metrics.values()), default=0.0)
                if opportunity_metrics
                else 0.0
            )

            plan = VelocityOptimizationPlan(
                timestamp=time.time(),
                portfolio_velocity_pct_per_hour=portfolio_velocity_avg * 100.0,
                opportunity_velocity_pct_per_hour=best_opp_velocity * 100.0,
                velocity_gap=(best_opp_velocity - portfolio_velocity_avg) * 100.0,
                rotations_recommended=rotations,
                hold_positions=held_symbols,
                analysis={
                    "position_count": len(owned_positions),
                    "candidate_count": len(candidate_symbols),
                    "recyclable_count": sum(1 for m in position_metrics.values() if m.is_recyclable),
                    "opportunity_count": len(opportunity_metrics),
                    "position_metrics": {
                        sym: {
                            "age_hours": m.age_hours,
                            "pnl_pct": m.pnl_pct,
                            "net_velocity_pct": m.net_velocity * 100.0,
                            "recyclable": m.is_recyclable,
                        }
                        for sym, m in position_metrics.items()
                    },
                    "opportunity_metrics": {
                        sym: {
                            "expected_return_pct": m.expected_return_pct * 100.0,
                            "ml_confidence": m.ml_confidence,
                            "estimated_velocity_pct": m.estimated_velocity_pct * 100.0,
                        }
                        for sym, m in opportunity_metrics.items()
                    },
                },
            )

            self.logger.info(
                "[VelocityOptimizer] Portfolio: %.2f%% | Opportunity: %.2f%% | Gap: %.2f%% | Rotations: %d",
                plan.portfolio_velocity_pct_per_hour,
                plan.opportunity_velocity_pct_per_hour,
                plan.velocity_gap,
                len(rotations),
            )

            return plan

        except Exception as e:
            self.logger.error("[VelocityOptimizer] Exception in optimize_capital_velocity", exc_info=True)
            return VelocityOptimizationPlan(
                timestamp=time.time(),
                portfolio_velocity_pct_per_hour=0.0,
                opportunity_velocity_pct_per_hour=0.0,
                velocity_gap=0.0,
                rotations_recommended=[],
                hold_positions=list(owned_positions.keys()),
                analysis={"error": str(e)},
            )
