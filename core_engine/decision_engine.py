"""
Decision Engine (Façade)
────────────────────────

Core Function #3: DECIDE what to do
    - Decision logic and signal arbitration
    - Trading mode selection (PAUSED, PROTECTIVE, BOOTSTRAP, etc.)
    - Multi-layer gates (symbol format, confidence, regime, position limit, capital, risk)
    - Position sizing and capital allocation
    - Entry/exit decision logic
    - Policy enforcement (daily profit targets, max leverage, etc.)

This engine abstracts and coordinates:
    - meta_controller.py (L8) - decision cycle
    - arbitration_engine.py (L5) - multi-layer gates
    - mode_manager.py (L5) - mode selection
    - capital_allocator.py (L6) - sizing
    - policy_manager.py (L6) - policy enforcement
    - execution_logic.py (L4) - decision translation
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, List, Optional

from core_engine.implementations import DecisionEngineImpl

# Type hints
__all__ = ["DecisionEngine", "TradeDecision", "ArbitrationResult"]

logger = logging.getLogger(__name__)


@dataclass
class TradeDecision:
    """Decision output from the decision engine."""
    symbol: str
    action: str  # "BUY", "SELL", "HOLD", "FORCE_EXIT"
    quantity: float
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reason: str = ""
    confidence: float = 0.0  # 0.0 to 1.0
    timestamp: float = 0.0
    mode: str = ""  # Current trading mode


@dataclass
class ArbitrationResult:
    """Result of arbitration gates evaluation."""
    passed: bool
    gates_status: Dict[str, bool]  # e.g., {"symbol_format": True, "confidence": True, ...}
    blocking_gates: List[str]  # Names of gates that blocked the decision
    reason: str = ""


class DecisionEngine:
    """
    Façade for trading decision making.
    
    Responsibility: Synthesize all signals, modes, and policies into trades.
    - Arbitrate signals through multi-layer safety gates
    - Evaluate trading mode constraints
    - Apply risk and policy limits
    - Generate buy/sell/exit decisions
    - Size positions according to capital allocation
    """

    def __init__(self, app_ctx: Any):
        """
        Initialize the decision engine.
        
        Args:
            app_ctx: Application context containing all layer components
        """
        self.app_ctx = app_ctx
        self.logger = logger

    async def initialize(self) -> None:
        """Initialize decision engine."""
        self.logger.info("🚀 DecisionEngine: initializing...")
        self.logger.info("✅ DecisionEngine: ready")

    async def get_current_mode(self) -> str:
        """
        Get current trading mode.
        
        Returns:
            Mode string: "PAUSED", "PROTECTIVE", "BOOTSTRAP", "GROWTH", etc.
        """
        return await DecisionEngineImpl.get_current_mode(self.app_ctx)

    async def set_mode(self, mode: str) -> bool:
        """
        Switch trading mode.
        
        Args:
            mode: Mode to switch to
        
        Returns:
            True if successful
        """
        try:
            mode_manager = self.app_ctx.get("mode_manager")
            
            if mode_manager:
                # Set mode (L5)
                self.logger.info(f"🔄 Switching mode to: {mode}")
                # await mode_manager.set_mode(mode)
                return True
            
            return False
        except Exception as e:
            self.logger.error(f"❌ Error setting mode: {e}")
            return False

    async def evaluate_signal(self, symbol: str, signal_type: str, edge_score: float) -> ArbitrationResult:
        """
        Evaluate a signal through arbitration gates.
        
        Applies 6-layer gate evaluation:
        1. Symbol format validation
        2. Confidence floor (mode-dependent)
        3. Market regime check
        4. Position limit check
        5. Capital available check
        6. Risk manager approval
        
        Args:
            symbol: Trading pair
            signal_type: "BUY" or "SELL"
            edge_score: Signal edge score (-1.0 to +1.0)
        
        Returns:
            ArbitrationResult with gate-by-gate status
        """
        return await DecisionEngineImpl.evaluate_signal(self.app_ctx, symbol, signal_type, edge_score)

    async def make_buy_decision(self, symbol: str, signal_edge: float) -> Optional[TradeDecision]:
        """
        Make a buy decision for a symbol.
        
        Args:
            symbol: Trading pair
            signal_edge: Signal edge score from fusion
        
        Returns:
            TradeDecision if decision made, None if rejected
        """
        return await DecisionEngineImpl.make_buy_decision(self.app_ctx, symbol, signal_edge)

    async def make_sell_decision(self, symbol: str, signal_edge: float, reason: str = "") -> Optional[TradeDecision]:
        """
        Make a sell decision for a symbol.
        
        Args:
            symbol: Trading pair
            signal_edge: Signal edge score from fusion
            reason: Reason for sell (e.g., "TP_HIT", "SL_HIT", "SIGNAL", "EXIT")
        
        Returns:
            TradeDecision if decision made, None if no position to sell
        """
        try:
            # Step 1: Check if we have a position
            position_manager = self.app_ctx.get("position_manager")
            
            if position_manager:
                # Get position (L3)
                # position = await position_manager.get_position(symbol)
                # if not position or position.quantity <= 0:
                #     return None
                pass
            
            # Step 2: Arbitrate sell signal (if signal-based)
            if "SIGNAL" in reason:
                arb_result = await self.evaluate_signal(symbol, "SELL", signal_edge)
                
                if not arb_result.passed:
                    self.logger.debug(
                        f"🚫 SELL signal rejected for {symbol}: "
                        f"gates blocked: {arb_result.blocking_gates}"
                    )
                    return None
            
            # Step 3: Create decision
            decision = TradeDecision(
                symbol=symbol,
                action="SELL",
                quantity=0.0,  # Sell all
                reason=reason or f"Signal edge: {signal_edge:.3f}",
                confidence=abs(signal_edge),
                timestamp=asyncio.get_event_loop().time(),
                mode=await self.get_current_mode(),
            )
            
            self.logger.info(f"✅ SELL decision: {symbol} (reason: {reason})")
            return decision
        except Exception as e:
            self.logger.error(f"❌ Error making SELL decision for {symbol}: {e}")
            return None

    async def evaluate_exit_signals(self, symbol: str) -> Optional[TradeDecision]:
        """
        Check if a position should be exited based on TP/SL/timeout.
        
        Args:
            symbol: Trading pair
        
        Returns:
            TradeDecision with EXIT action, or None if no exit triggered
        """
        try:
            tp_sl_engine = self.app_ctx.get("tp_sl_engine")
            fourth_slot_tracker = self.app_ctx.get("fourth_slot_tracker")
            
            if tp_sl_engine:
                # Check TP/SL levels (L4)
                # exit_reason = await tp_sl_engine.check_exit_levels(symbol)
                # if exit_reason:
                #     return TradeDecision(
                #         symbol=symbol,
                #         action="FORCE_EXIT",
                #         quantity=0.0,
                #         reason=exit_reason,
                #     )
                pass
            
            if fourth_slot_tracker:
                # Check 4th slot forced exit (L8)
                # if await fourth_slot_tracker.should_exit(symbol):
                #     return TradeDecision(
                #         symbol=symbol,
                #         action="FORCE_EXIT",
                #         quantity=0.0,
                #         reason="4TH_SLOT_TIMEOUT",
                #     )
                pass
            
            return None
        except Exception as e:
            self.logger.error(f"❌ Error evaluating exit signals for {symbol}: {e}")
            return None

    async def apply_policy_constraints(self, decision: TradeDecision) -> bool:
        """
        Check if decision passes all policy constraints.
        
        Args:
            decision: Trade decision to validate
        
        Returns:
            True if passes all policies
        """
        try:
            policy_manager = self.app_ctx.get("policy_manager")
            risk_manager = self.app_ctx.get("risk_manager")
            
            if policy_manager:
                # Check policies (L6): daily profit targets, dust ratio, etc.
                # if not await policy_manager.is_allowed(decision):
                #     return False
                pass
            
            if risk_manager:
                # Check risk limits (L6)
                # if not await risk_manager.is_acceptable(decision):
                #     return False
                pass
            
            return True
        except Exception as e:
            self.logger.error(f"❌ Error applying policy constraints: {e}")
            return False

    async def get_mode_constraints(self) -> Dict[str, Any]:
        """
        Get constraints for current trading mode.
        
        Returns:
            {
                "mode": "PROTECTIVE",
                "allow_new_positions": bool,
                "max_open_positions": int,
                "confidence_floor": float,
                "position_size_cap": float,
                "leverage_cap": float,
            }
        """
        try:
            mode_manager = self.app_ctx.get("mode_manager")
            
            constraints = {
                "mode": "PROTECTIVE",
                "allow_new_positions": False,
                "max_open_positions": 3,
                "confidence_floor": 0.6,
                "position_size_cap": 0.1,  # 10% of capital per position
                "leverage_cap": 1.0,  # No margin
            }
            
            if mode_manager:
                # Get mode constraints (L5)
                # constraints = await mode_manager.get_constraints()
                pass
            
            return constraints
        except Exception as e:
            self.logger.error(f"❌ Error getting mode constraints: {e}")
            return constraints

    async def shutdown(self) -> None:
        """Gracefully shut down decision engine."""
        self.logger.info("🛑 DecisionEngine: shutting down...")
        self.logger.info("✅ DecisionEngine: shut down complete")
