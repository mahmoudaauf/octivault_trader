"""
ExternalAdoptionEngine: Intelligent Handling of Pre-Existing Assets

Converts external (pre-existing) positions into strategy-managed positions
with intelligent decision logic based on risk, exposure, and classification.

Decision Logic:
    if value < dust_threshold:
        → LIQUIDATE (clean up)
    elif symbol in active_universe:
        → ADOPT (convert to strategy position)
    elif exposure > risk_limit:
        → HEDGE (reduce gradually)
    else:
        → IGNORE (leave as-is)

Modes:
    1. IGNORE - Do nothing (hold manually)
    2. LIQUIDATE - Sell immediately (dust cleanup)
    3. ADOPT - Convert to strategy position (assign TP/SL)
    4. HEDGE - Reduce exposure gradually (sell proportionally)
"""

import asyncio
import logging
from typing import Dict, Optional, List, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from decimal import Decimal

logger = logging.getLogger(__name__)


class AdoptionMode(Enum):
    """How to handle external positions"""
    IGNORE = "ignore"           # Do nothing
    LIQUIDATE = "liquidate"     # Sell immediately
    ADOPT = "adopt"             # Convert to strategy position
    HEDGE = "hedge"             # Reduce gradually


class AdoptionDecision(Enum):
    """Decision made for external position"""
    IGNORED = "ignored"
    LIQUIDATED = "liquidated"
    ADOPTED = "adopted"
    HEDGING = "hedging"
    ERROR = "error"


@dataclass
class ExternalPosition:
    """Represents pre-existing external position"""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    value_usdt: float
    _mirrored: bool = True  # Wallet is authoritative


@dataclass
class AdoptionResult:
    """Result of adoption decision"""
    symbol: str
    decision: AdoptionDecision
    mode: AdoptionMode
    reason: str
    value_usdt: float
    action_taken: Optional[str] = None  # What we did
    hedge_percentage: Optional[float] = None  # If hedging
    estimated_proceeds: Optional[float] = None  # If liquidating
    timestamp: float = field(default_factory=lambda: datetime.utcnow().timestamp())


class ExternalAdoptionEngine:
    """
    Intelligent adoption logic for external/pre-existing positions

    Responsibilities:
    1. Classify external positions
    2. Evaluate each position against strategy parameters
    3. Make adoption decision (IGNORE/LIQUIDATE/ADOPT/HEDGE)
    4. Execute decision (if applicable)
    5. Track and log all decisions

    Design:
    - Operator has final say (all decisions logged for review)
    - Conservative defaults (prefer IGNORE unless clear reason to act)
    - Multi-level validation
    - Complete audit trail
    """

    def __init__(
        self,
        shared_state=None,
        risk_manager=None,
        execution_manager=None,
        config=None,
    ):
        self.shared_state = shared_state
        self.risk_manager = risk_manager
        self.execution_manager = execution_manager
        self.config = config or {}

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Configuration thresholds
        self.dust_threshold_usdt = float(self.config.get("EXTERNAL_DUST_THRESHOLD_USDT", 10.0))
        self.adoption_value_min_usdt = float(self.config.get("EXTERNAL_ADOPTION_MIN_USDT", 100.0))
        self.exposure_limit_pct = float(self.config.get("EXTERNAL_EXPOSURE_LIMIT_PCT", 0.40))
        self.hedge_speed_pct_per_day = float(self.config.get("EXTERNAL_HEDGE_SPEED_PCT", 5.0))

        # State
        self._adoption_history: List[AdoptionResult] = []
        self._adopted_positions: Dict[str, ExternalPosition] = {}
        self._hedging_positions: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    # ═══════════════════════════════════════════════════════════════════
    # Public API
    # ═══════════════════════════════════════════════════════════════════

    async def evaluate_external_positions(
        self,
        external_positions: Dict[str, ExternalPosition],
    ) -> List[AdoptionResult]:
        """
        Evaluate all external positions and make decisions

        Args:
            external_positions: Dict of {symbol: ExternalPosition}

        Returns:
            List of AdoptionResult decisions
        """
        results = []

        for symbol, position in external_positions.items():
            result = await self.evaluate_position(symbol, position)
            results.append(result)

        return results

    async def evaluate_position(
        self,
        symbol: str,
        position: ExternalPosition,
    ) -> AdoptionResult:
        """
        Evaluate single external position

        Decision tree:
        1. Is it dust? → LIQUIDATE
        2. Is symbol in active universe? → ADOPT
        3. Is exposure over limit? → HEDGE
        4. Otherwise → IGNORE
        """
        async with self._lock:
            try:
                # Gather decision factors
                nav = await self._get_nav()
                symbol_exposure_pct = position.value_usdt / nav if nav > 0 else 0.0
                active_universe = await self._get_active_universe()
                unrealized_pnl_pct = self._calculate_pnl_pct(position)

                # Decision tree
                decision = self._make_adoption_decision(
                    symbol=symbol,
                    position=position,
                    nav=nav,
                    exposure_pct=symbol_exposure_pct,
                    in_universe=symbol in active_universe,
                    unrealized_pnl_pct=unrealized_pnl_pct,
                )

                # Execute decision (if applicable)
                action_taken = None
                if decision.mode == AdoptionMode.LIQUIDATE:
                    action_taken = await self._execute_liquidate(symbol, position)
                elif decision.mode == AdoptionMode.ADOPT:
                    action_taken = await self._execute_adopt(symbol, position)
                elif decision.mode == AdoptionMode.HEDGE:
                    action_taken = await self._execute_hedge(symbol, position)

                # Store result
                result = AdoptionResult(
                    symbol=symbol,
                    decision=decision["enum"],
                    mode=decision["mode"],
                    reason=decision["reason"],
                    value_usdt=position.value_usdt,
                    action_taken=action_taken,
                    hedge_percentage=decision.get("hedge_pct"),
                    estimated_proceeds=decision.get("estimated_proceeds"),
                )

                self._adoption_history.append(result)
                self._log_adoption_decision(result)

                return result

            except Exception as e:
                self.logger.error(f"[ExternalAdoption] Evaluation error for {symbol}: {e}", exc_info=True)
                return AdoptionResult(
                    symbol=symbol,
                    decision=AdoptionDecision.ERROR,
                    mode=AdoptionMode.IGNORE,
                    reason=f"Evaluation error: {str(e)}",
                    value_usdt=position.value_usdt,
                )

    async def mark_position_adopted(
        self,
        symbol: str,
        position: ExternalPosition,
        tp_price: Optional[float] = None,
        sl_price: Optional[float] = None,
    ) -> bool:
        """
        Mark position as manually adopted

        Operator can manually override adoption logic
        """
        async with self._lock:
            try:
                self._adopted_positions[symbol] = position

                self.logger.info(
                    "[EXTERNAL_ADOPTION] Manual adoption: symbol=%s value=%.2f tp=%.2f sl=%.2f",
                    symbol,
                    position.value_usdt,
                    tp_price or 0.0,
                    sl_price or 0.0,
                )

                # Could integrate with TPSLEngine to set TP/SL
                if self.execution_manager and tp_price:
                    # Set TP/SL via TPSLEngine if available
                    try:
                        # Check if execution_manager has TPSLEngine
                        if hasattr(self.execution_manager, 'tpsl_engine') and self.execution_manager.tpsl_engine:
                            await self.execution_manager.tpsl_engine.set_take_profit(
                                symbol=symbol,
                                tp_price=tp_price,
                                tp_percent=None
                            )
                            if sl_price:
                                await self.execution_manager.tpsl_engine.set_stop_loss(
                                    symbol=symbol,
                                    sl_price=sl_price,
                                    sl_percent=None
                                )
                            self.logger.debug(f"[ExternalAdoption] TP/SL set for {symbol}: TP={tp_price}, SL={sl_price}")
                    except Exception as e:
                        self.logger.warning(f"[ExternalAdoption] Could not set TP/SL via TPSLEngine: {e}")

                return True
            except Exception as e:
                self.logger.error(f"[ExternalAdoption] Adoption error: {e}", exc_info=True)
                return False

    async def reject_adoption(self, symbol: str) -> bool:
        """
        Reject adoption - position stays external/manual

        Operator can manually reject adoption suggestions
        """
        async with self._lock:
            self._adopted_positions.pop(symbol, None)
            self._hedging_positions.pop(symbol, None)
            self.logger.info("[EXTERNAL_ADOPTION] Adoption rejected for %s", symbol)
            return True

    def get_adoption_status(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get adoption status"""
        if symbol:
            return {
                "adopted": symbol in self._adopted_positions,
                "hedging": symbol in self._hedging_positions,
                "history": [
                    {
                        "decision": r.decision.value,
                        "mode": r.mode.value,
                        "reason": r.reason,
                        "timestamp": r.timestamp,
                    }
                    for r in self._adoption_history if r.symbol == symbol
                ],
            }

        return {
            "adopted_count": len(self._adopted_positions),
            "hedging_count": len(self._hedging_positions),
            "total_decisions": len(self._adoption_history),
            "adopted_symbols": list(self._adopted_positions.keys()),
            "hedging_symbols": list(self._hedging_positions.keys()),
        }

    # ═══════════════════════════════════════════════════════════════════
    # Internal Decision Logic
    # ═══════════════════════════════════════════════════════════════════

    def _make_adoption_decision(
        self,
        symbol: str,
        position: ExternalPosition,
        nav: float,
        exposure_pct: float,
        in_universe: bool,
        unrealized_pnl_pct: float,
    ) -> Dict[str, Any]:
        """
        Decision tree for adoption

        Returns:
            {
                "enum": AdoptionDecision,
                "mode": AdoptionMode,
                "reason": str,
                "hedge_pct": Optional[float],  # For HEDGE
                "estimated_proceeds": Optional[float],  # For LIQUIDATE
            }
        """

        # Decision 1: Is it dust?
        if position.value_usdt < self.dust_threshold_usdt:
            return {
                "enum": AdoptionDecision.LIQUIDATED,
                "mode": AdoptionMode.LIQUIDATE,
                "reason": f"Below dust threshold: {position.value_usdt:.2f} < {self.dust_threshold_usdt}",
                "estimated_proceeds": position.value_usdt * 0.99,  # Assume 1% fee
            }

        # Decision 2: Is symbol in active universe?
        # (Don't adopt if value is too small relative to strategy capital)
        if in_universe and position.value_usdt >= self.adoption_value_min_usdt:
            # Check if adoption would violate concentration
            if exposure_pct <= self.exposure_limit_pct:
                return {
                    "enum": AdoptionDecision.ADOPTED,
                    "mode": AdoptionMode.ADOPT,
                    "reason": f"Symbol in active universe, value {position.value_usdt:.2f}, exposure {exposure_pct*100:.1f}%",
                }

        # Decision 3: Is exposure over limit?
        if exposure_pct > self.exposure_limit_pct:
            # Calculate hedge amount (reduce to limit)
            target_value = nav * self.exposure_limit_pct
            hedge_qty = position.quantity * (1.0 - (target_value / position.value_usdt))
            hedge_pct = (hedge_qty / position.quantity) * 100 if position.quantity > 0 else 0

            return {
                "enum": AdoptionDecision.HEDGING,
                "mode": AdoptionMode.HEDGE,
                "reason": f"Exposure {exposure_pct*100:.1f}% > limit {self.exposure_limit_pct*100:.1f}%",
                "hedge_pct": hedge_pct,
            }

        # Decision 4: Otherwise ignore
        return {
            "enum": AdoptionDecision.IGNORED,
            "mode": AdoptionMode.IGNORE,
            "reason": f"Value {position.value_usdt:.2f}, exposure {exposure_pct*100:.1f}%, in_universe={in_universe}",
        }

    def _calculate_pnl_pct(self, position: ExternalPosition) -> float:
        """Calculate unrealized P&L as percentage"""
        if position.entry_price <= 0:
            return 0.0
        return ((position.current_price - position.entry_price) / position.entry_price) * 100

    # ═══════════════════════════════════════════════════════════════════
    # Execution Methods
    # ═══════════════════════════════════════════════════════════════════

    async def _execute_liquidate(self, symbol: str, position: ExternalPosition) -> str:
        """Execute liquidation of external position"""
        try:
            if self.execution_manager:
                result = await self.execution_manager.execute_trade(
                    symbol=symbol,
                    side="SELL",
                    quantity=position.quantity,
                    reason="external_dust_liquidation",
                )
                if result.get("ok"):
                    self.logger.info(
                        "[EXTERNAL_ADOPTION] Liquidated dust: %s qty=%.8f value=%.2f",
                        symbol,
                        position.quantity,
                        position.value_usdt,
                    )
                    return "liquidated"
        except Exception as e:
            self.logger.error(f"[ExternalAdoption] Liquidation error for {symbol}: {e}", exc_info=True)
        return "liquidation_attempted"

    async def _execute_adopt(self, symbol: str, position: ExternalPosition) -> str:
        """Execute adoption: Convert to strategy position"""
        try:
            await self.mark_position_adopted(symbol, position)
            self.logger.info(
                "[EXTERNAL_ADOPTION] Adopted: %s qty=%.8f value=%.2f entry=%.2f",
                symbol,
                position.quantity,
                position.value_usdt,
                position.entry_price,
            )
            return "adopted"
        except Exception as e:
            self.logger.error(f"[ExternalAdoption] Adoption error for {symbol}: {e}", exc_info=True)
        return "adoption_attempted"

    async def _execute_hedge(self, symbol: str, position: ExternalPosition) -> str:
        """Execute hedge: Reduce exposure gradually"""
        try:
            # Start gradual hedging process
            self._hedging_positions[symbol] = {
                "position": position,
                "started_at": datetime.utcnow().timestamp(),
                "daily_pct": self.hedge_speed_pct_per_day,
                "completed_pct": 0.0,
            }
            self.logger.info(
                "[EXTERNAL_ADOPTION] Hedging started: %s value=%.2f hedge_speed=%.1f%% per day",
                symbol,
                position.value_usdt,
                self.hedge_speed_pct_per_day,
            )
            return "hedging_started"
        except Exception as e:
            self.logger.error(f"[ExternalAdoption] Hedge error for {symbol}: {e}", exc_info=True)
        return "hedge_attempted"

    # ═══════════════════════════════════════════════════════════════════
    # Data Retrieval
    # ═══════════════════════════════════════════════════════════════════

    async def _get_nav(self) -> float:
        """Get current NAV"""
        if not self.shared_state:
            return 1000.0  # Default fallback
        try:
            nav = await self.shared_state.get_nav()
            return float(nav or 1000.0)
        except Exception:
            return 1000.0

    async def _get_active_universe(self) -> set:
        """Get set of symbols in active trading universe"""
        if not self.shared_state:
            return set()
        try:
            # Try to get from shared state or config
            return set(self.config.get("SYMBOLS", []))
        except Exception:
            return set()

    def _log_adoption_decision(self, result: AdoptionResult):
        """Log adoption decision"""
        self.logger.info(
            "[EXTERNAL_ADOPTION] symbol=%s action=%s mode=%s reason=%s value=%.2f",
            result.symbol,
            result.decision.value,
            result.mode.value,
            result.reason,
            result.value_usdt,
        )

    # ═══════════════════════════════════════════════════════════════════
    # Diagnostic API
    # ═══════════════════════════════════════════════════════════════════

    async def get_status(self) -> Dict[str, Any]:
        """Get full adoption engine status"""
        return {
            "status": "operational",
            "adopted_positions": list(self._adopted_positions.keys()),
            "hedging_positions": list(self._hedging_positions.keys()),
            "total_decisions": len(self._adoption_history),
            "adoption_history": [
                {
                    "symbol": r.symbol,
                    "decision": r.decision.value,
                    "mode": r.mode.value,
                    "value_usdt": r.value_usdt,
                    "timestamp": r.timestamp,
                }
                for r in self._adoption_history[-20:]
            ],
            "thresholds": {
                "dust_threshold_usdt": self.dust_threshold_usdt,
                "adoption_value_min_usdt": self.adoption_value_min_usdt,
                "exposure_limit_pct": self.exposure_limit_pct,
                "hedge_speed_pct_per_day": self.hedge_speed_pct_per_day,
            },
        }
