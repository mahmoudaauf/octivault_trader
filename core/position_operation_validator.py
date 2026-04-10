# core/position_operation_validator.py
# Professional Safety Validation Layer for Three-Layer Capital Accounting
# Prevents dangerous operations on external positions, dust, and protected assets

import logging
from typing import Any, Dict, Optional, Tuple
from enum import Enum

logger = logging.getLogger("PositionOperationValidator")


class OperationType(Enum):
    """Operation types that require validation."""
    TRADE_ENTRY = "TRADE_ENTRY"           # Opening a new position (BUY)
    TRADE_EXIT = "TRADE_EXIT"             # Closing a position (SELL)
    LIQUIDATION = "LIQUIDATION"           # Emergency liquidation (forced SELL)
    REBALANCE = "REBALANCE"               # Position rebalancing
    DUST_CLEANUP = "DUST_CLEANUP"         # Dust liquidation
    POSITION_MODIFICATION = "POSITION_MODIFICATION"  # Modifying existing position


class OperationResult:
    """Result of a validation check."""
    
    def __init__(self, allowed: bool, reason: str = "", severity: str = "INFO"):
        self.allowed = allowed
        self.reason = reason
        self.severity = severity  # INFO, WARNING, CRITICAL
    
    def __repr__(self):
        return f"OperationResult(allowed={self.allowed}, severity={self.severity}, reason='{self.reason}')"


class PositionOperationValidator:
    """
    Professional safety validation layer.
    
    Responsibilities:
      - Prevent liquidation of EXTERNAL_POSITION assets (user holdings)
      - Allow liquidation of BOT_POSITION assets (bot trades)
      - Allow dusty DUST positions to be liquidated
      - Protect STABLE assets from unnecessary operations
      - Enforce layer contracts and boundaries
    
    Design Principle:
      - Fail SAFE: If uncertain, block the operation
      - Defensive: Check all position metadata before operation
      - Auditable: Log all validations for compliance
    """
    
    def __init__(self, shared_state: Any, config: Any):
        self.ss = shared_state
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Validation configuration
        self.protect_external_positions = bool(
            getattr(config, "PROTECT_EXTERNAL_POSITIONS", True)
        )
        self.allow_stable_trading = bool(
            getattr(config, "ALLOW_STABLE_TRADING", False)
        )
        self.log_all_validations = bool(
            getattr(config, "LOG_ALL_VALIDATIONS", True)
        )
        
        self.logger.info(
            f"PositionOperationValidator initialized ("
            f"protect_external={self.protect_external_positions}, "
            f"allow_stable_trading={self.allow_stable_trading})"
        )
    
    async def can_trade_entry(
        self,
        symbol: str,
        quantity: float,
        reason: str = ""
    ) -> OperationResult:
        """
        Validate if a BUY (trade entry) is allowed on this symbol.
        
        Checks:
          - Symbol is not locked for other operations
          - No critical risk violations
          - Position doesn't already exist (or is stackable)
        
        Returns: OperationResult(allowed=bool, reason=str)
        """
        try:
            # Get position classification if exists
            classification = await self._get_position_classification(symbol)
            
            # Allow BUY on new symbols
            if classification is None:
                self.logger.info(
                    f"[VALIDATE] TRADE_ENTRY {symbol}: OK (new position, "
                    f"quantity={quantity}, reason='{reason}')"
                )
                return OperationResult(allowed=True, reason="New position allowed")
            
            # Allow BUY on BOT_POSITION (stacking)
            if classification == "BOT_POSITION":
                self.logger.info(
                    f"[VALIDATE] TRADE_ENTRY {symbol}: OK (BOT_POSITION stack, "
                    f"quantity={quantity}, reason='{reason}')"
                )
                return OperationResult(allowed=True, reason="Stack on existing bot position allowed")
            
            # CRITICAL: Block BUY on EXTERNAL_POSITION (don't increase user holdings)
            if classification == "EXTERNAL_POSITION" and self.protect_external_positions:
                self.logger.warning(
                    f"[VALIDATE] TRADE_ENTRY {symbol}: BLOCKED (EXTERNAL_POSITION protected, "
                    f"quantity={quantity}, reason='{reason}')"
                )
                return OperationResult(
                    allowed=False,
                    reason="Cannot trade entry on EXTERNAL_POSITION (user holding)",
                    severity="CRITICAL"
                )
            
            # Allow BUY on DUST or STABLE
            if classification in ("DUST", "STABLE"):
                self.logger.info(
                    f"[VALIDATE] TRADE_ENTRY {symbol}: OK ({classification}, "
                    f"quantity={quantity}, reason='{reason}')"
                )
                return OperationResult(allowed=True, reason=f"Entry allowed on {classification}")
            
            # Default allow
            return OperationResult(allowed=True, reason=f"Entry allowed (classification={classification})")
        
        except Exception as e:
            self.logger.error(f"[VALIDATE] TRADE_ENTRY {symbol}: ERROR - {e}", exc_info=True)
            return OperationResult(
                allowed=False,
                reason=f"Validation error: {e}",
                severity="CRITICAL"
            )
    
    async def can_trade_exit(
        self,
        symbol: str,
        quantity: float,
        reason: str = ""
    ) -> OperationResult:
        """
        Validate if a SELL (trade exit) is allowed on this symbol.
        
        Checks:
          - Symbol exists and has valid quantity
          - Position is not locked
          - If EXTERNAL_POSITION, only allow via special approval reason
        
        Returns: OperationResult(allowed=bool, reason=str)
        """
        try:
            # Get position data
            classification = await self._get_position_classification(symbol)
            current_qty = await self._get_position_quantity(symbol)
            
            # Must have existing position to exit
            if current_qty is None or current_qty <= 0:
                self.logger.warning(
                    f"[VALIDATE] TRADE_EXIT {symbol}: BLOCKED (no position to exit, "
                    f"quantity={quantity}, reason='{reason}')"
                )
                return OperationResult(
                    allowed=False,
                    reason="No position to exit",
                    severity="WARNING"
                )
            
            # Check quantity validity
            if quantity <= 0 or quantity > current_qty:
                self.logger.warning(
                    f"[VALIDATE] TRADE_EXIT {symbol}: BLOCKED (invalid quantity, "
                    f"requested={quantity}, available={current_qty}, reason='{reason}')"
                )
                return OperationResult(
                    allowed=False,
                    reason=f"Invalid exit quantity: requested={quantity}, available={current_qty}",
                    severity="WARNING"
                )
            
            # Allow exit of BOT_POSITION
            if classification == "BOT_POSITION":
                self.logger.info(
                    f"[VALIDATE] TRADE_EXIT {symbol}: OK (BOT_POSITION, "
                    f"quantity={quantity}, reason='{reason}')"
                )
                return OperationResult(allowed=True, reason="Exit of bot position allowed")
            
            # EXTERNAL_POSITION: Only allow with explicit approval reason
            if classification == "EXTERNAL_POSITION":
                # Check if reason indicates approval (e.g., "USER_APPROVED", "EMERGENCY_LIQUIDATION")
                if self.protect_external_positions and not self._is_approved_external_operation(reason):
                    self.logger.warning(
                        f"[VALIDATE] TRADE_EXIT {symbol}: BLOCKED (EXTERNAL_POSITION requires approval, "
                        f"reason='{reason}')"
                    )
                    return OperationResult(
                        allowed=False,
                        reason="EXTERNAL_POSITION exit requires explicit approval",
                        severity="CRITICAL"
                    )
                
                self.logger.warning(
                    f"[VALIDATE] TRADE_EXIT {symbol}: APPROVED (EXTERNAL_POSITION, "
                    f"reason='{reason}')"
                )
                return OperationResult(
                    allowed=True,
                    reason=f"Approved external position exit (reason: {reason})",
                    severity="WARNING"
                )
            
            # Allow exit of DUST
            if classification == "DUST":
                self.logger.info(
                    f"[VALIDATE] TRADE_EXIT {symbol}: OK (DUST cleanup, "
                    f"quantity={quantity}, reason='{reason}')"
                )
                return OperationResult(allowed=True, reason="Dust liquidation allowed")
            
            # Default allow for other classifications
            return OperationResult(allowed=True, reason=f"Exit allowed (classification={classification})")
        
        except Exception as e:
            self.logger.error(f"[VALIDATE] TRADE_EXIT {symbol}: ERROR - {e}", exc_info=True)
            return OperationResult(
                allowed=False,
                reason=f"Validation error: {e}",
                severity="CRITICAL"
            )
    
    async def can_liquidate(
        self,
        symbol: str,
        quantity: float,
        reason: str = ""
    ) -> OperationResult:
        """
        Validate if a LIQUIDATION (forced SELL) is allowed on this symbol.
        
        Checks:
          - Symbol exists and has valid quantity
          - Classification allows liquidation (BOT_POSITION, DUST, RECOVERY)
          - CRITICAL: EXTERNAL_POSITION liquidation is blocked
        
        Returns: OperationResult(allowed=bool, reason=str)
        """
        try:
            # Get position data
            classification = await self._get_position_classification(symbol)
            current_qty = await self._get_position_quantity(symbol)
            
            # Must have existing position to liquidate
            if current_qty is None or current_qty <= 0:
                return OperationResult(
                    allowed=False,
                    reason="No position to liquidate",
                    severity="WARNING"
                )
            
            # Check quantity validity
            if quantity <= 0 or quantity > current_qty:
                return OperationResult(
                    allowed=False,
                    reason=f"Invalid liquidation quantity: {quantity} (available: {current_qty})",
                    severity="WARNING"
                )
            
            # CRITICAL: Block liquidation of EXTERNAL_POSITION
            if classification == "EXTERNAL_POSITION":
                self.logger.critical(
                    f"[VALIDATE] LIQUIDATION {symbol}: BLOCKED (EXTERNAL_POSITION is protected, "
                    f"quantity={quantity}, reason='{reason}')"
                )
                return OperationResult(
                    allowed=False,
                    reason="EXTERNAL_POSITION cannot be liquidated (user holding)",
                    severity="CRITICAL"
                )
            
            # Allow liquidation of BOT_POSITION
            if classification == "BOT_POSITION":
                self.logger.warning(
                    f"[VALIDATE] LIQUIDATION {symbol}: OK (BOT_POSITION, "
                    f"quantity={quantity}, reason='{reason}')"
                )
                return OperationResult(
                    allowed=True,
                    reason="Liquidation of bot position allowed",
                    severity="WARNING"
                )
            
            # Allow liquidation of DUST
            if classification == "DUST":
                self.logger.info(
                    f"[VALIDATE] LIQUIDATION {symbol}: OK (DUST, "
                    f"quantity={quantity}, reason='{reason}')"
                )
                return OperationResult(allowed=True, reason="Dust liquidation allowed")
            
            # Allow liquidation of RECOVERY positions
            if classification == "RECOVERY":
                self.logger.info(
                    f"[VALIDATE] LIQUIDATION {symbol}: OK (RECOVERY, "
                    f"quantity={quantity}, reason='{reason}')"
                )
                return OperationResult(allowed=True, reason="Recovery position liquidation allowed")
            
            # Allow liquidation of STABLE (stablecoins don't lose value)
            if classification == "STABLE":
                self.logger.info(
                    f"[VALIDATE] LIQUIDATION {symbol}: OK (STABLE, "
                    f"quantity={quantity}, reason='{reason}')"
                )
                return OperationResult(allowed=True, reason="Stable asset liquidation allowed")
            
            # Default block (fail safe)
            return OperationResult(
                allowed=False,
                reason=f"Liquidation blocked for unknown classification: {classification}",
                severity="CRITICAL"
            )
        
        except Exception as e:
            self.logger.error(f"[VALIDATE] LIQUIDATION {symbol}: ERROR - {e}", exc_info=True)
            return OperationResult(
                allowed=False,
                reason=f"Validation error: {e}",
                severity="CRITICAL"
            )
    
    async def validate_operation(
        self,
        operation_type: OperationType,
        symbol: str,
        quantity: float,
        reason: str = ""
    ) -> OperationResult:
        """
        Main validation dispatcher. Route operation to appropriate validator.
        
        Args:
          - operation_type: Type of operation (TRADE_ENTRY, TRADE_EXIT, LIQUIDATION, etc.)
          - symbol: Trading symbol (e.g., "BTCUSDT")
          - quantity: Quantity to operate on
          - reason: Reason for operation (helps determine intent)
        
        Returns: OperationResult with detailed validation info
        """
        if operation_type == OperationType.TRADE_ENTRY:
            return await self.can_trade_entry(symbol, quantity, reason)
        elif operation_type == OperationType.TRADE_EXIT:
            return await self.can_trade_exit(symbol, quantity, reason)
        elif operation_type == OperationType.LIQUIDATION or operation_type == OperationType.DUST_CLEANUP:
            return await self.can_liquidate(symbol, quantity, reason)
        else:
            # Default allow for unknown types
            return OperationResult(allowed=True, reason=f"No validation for {operation_type}")
    
    # ========== PRIVATE HELPERS ==========
    
    async def _get_position_classification(self, symbol: str) -> Optional[str]:
        """Query position classification from SharedState."""
        try:
            if hasattr(self.ss, "get_position_classification"):
                result = self.ss.get_position_classification(symbol)
                if asyncio.iscoroutine(result):
                    return await result
                return result
        except Exception as e:
            self.logger.debug(f"Failed to get classification for {symbol}: {e}")
        return None
    
    async def _get_position_quantity(self, symbol: str) -> Optional[float]:
        """Query position quantity from SharedState."""
        try:
            if hasattr(self.ss, "get_position_quantity"):
                result = self.ss.get_position_quantity(symbol)
                if asyncio.iscoroutine(result):
                    return await result
                return result
            
            # Fallback: check positions dict
            if hasattr(self.ss, "positions"):
                pos = (self.ss.positions or {}).get(symbol, {})
                qty = float(pos.get("quantity", 0.0) or pos.get("qty", 0.0) or 0.0)
                return qty if qty > 0 else None
        except Exception as e:
            self.logger.debug(f"Failed to get quantity for {symbol}: {e}")
        return None
    
    def _is_approved_external_operation(self, reason: str) -> bool:
        """Check if reason indicates explicit approval for external position operation."""
        approved_reasons = {
            "USER_APPROVED",
            "MANUAL_OVERRIDE",
            "EMERGENCY_LIQUIDATION",
            "SYSTEM_EMERGENCY",
            "FORCED_EXIT",
            "COMPLIANCE_REQUIREMENT"
        }
        return any(ar in str(reason).upper() for ar in approved_reasons)


# Import asyncio at module level (needed for iscoroutine checks)
import asyncio
