# -*- coding: utf-8 -*-
"""
leverage_manager.py - P9-aligned Leverage Validation System

Implements leverage checks to prevent over-leveraging:
  • Calculates current leverage ratio
  • Enforces max leverage limit
  • Rejects positions exceeding limit
  • Monitors leverage trends

Architecture:
  • LeverageValidator: Position-level leverage checks
  • LeverageMonitor: Portfolio-level tracking
"""

import logging
from typing import Dict, Optional, Tuple
from datetime import datetime
from enum import Enum

logger = logging.getLogger("LeverageManager")


class LeverageStatus(str, Enum):
    """Leverage validation status"""
    ACCEPTED = "accepted"
    REJECTED_EXCEEDS_MAX = "rejected_exceeds_max"
    REJECTED_INVALID = "rejected_invalid"
    FAILED = "failed"


class LeverageValidator:
    """
    Position-level leverage validation.
    
    Prevents individual positions from exceeding max leverage.
    """
    
    def __init__(self, max_leverage: float = 1.0):
        """
        Initialize leverage validator.
        
        Args:
            max_leverage: Maximum allowed leverage (1.0 = no leverage)
        """
        if max_leverage < 1.0 or max_leverage > 125.0:
            raise ValueError("Max leverage must be between 1.0 and 125.0")
        
        self.max_leverage = max_leverage
        self.logger = logging.getLogger(__name__)
        self.validation_log = []
        
        self.logger.info(f"✅ Leverage validator initialized: max_leverage={max_leverage}x")
    
    async def validate_position_leverage(
        self,
        symbol: str,
        quantity: float,
        entry_price: float,
        account_balance: float,
    ) -> Tuple[bool, LeverageStatus, str, float]:
        """
        Validate position leverage.
        
        Args:
            symbol: Trading symbol
            quantity: Position quantity
            entry_price: Entry price per unit
            account_balance: Total account balance
        
        Returns:
            (is_valid, status, reason, calculated_leverage)
        """
        try:
            # Calculate position value
            position_value = quantity * entry_price
            
            # Calculate leverage
            if account_balance <= 0:
                return False, LeverageStatus.REJECTED_INVALID, \
                       "Invalid account balance", 0.0
            
            calculated_leverage = position_value / account_balance
            
            # Validation check
            if calculated_leverage > self.max_leverage:
                self.logger.warning(
                    f"❌ Position rejected - exceeds max leverage: "
                    f"{symbol} {quantity} @ {entry_price} = {calculated_leverage:.2f}x "
                    f"(max: {self.max_leverage}x)"
                )
                self.validation_log.append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "symbol": symbol,
                    "quantity": quantity,
                    "entry_price": entry_price,
                    "calculated_leverage": calculated_leverage,
                    "status": "rejected_exceeds_max",
                })
                return False, LeverageStatus.REJECTED_EXCEEDS_MAX, \
                       f"Position leverage {calculated_leverage:.2f}x exceeds max {self.max_leverage}x", \
                       calculated_leverage
            
            # Position accepted
            self.logger.info(
                f"✅ Position accepted: {symbol} {quantity} @ {entry_price} = {calculated_leverage:.2f}x "
                f"(max: {self.max_leverage}x)"
            )
            self.validation_log.append({
                "timestamp": datetime.utcnow().isoformat(),
                "symbol": symbol,
                "quantity": quantity,
                "entry_price": entry_price,
                "calculated_leverage": calculated_leverage,
                "status": "accepted",
            })
            return True, LeverageStatus.ACCEPTED, \
                   f"Position leverage {calculated_leverage:.2f}x is acceptable", \
                   calculated_leverage
        
        except Exception as e:
            self.logger.error(f"❌ Leverage validation failed: {e}", exc_info=True)
            return False, LeverageStatus.FAILED, str(e), 0.0
    
    def get_validation_log(self, limit: int = 50) -> list:
        """Get recent validation log"""
        return self.validation_log[-limit:]


class LeverageMonitor:
    """
    Portfolio-level leverage monitoring.
    
    Tracks overall portfolio leverage across all positions.
    """
    
    def __init__(self, max_portfolio_leverage: float = 2.0):
        """
        Initialize leverage monitor.
        
        Args:
            max_portfolio_leverage: Maximum portfolio-wide leverage
        """
        if max_portfolio_leverage < 1.0:
            raise ValueError("Portfolio leverage must be >= 1.0")
        
        self.max_portfolio_leverage = max_portfolio_leverage
        self.logger = logging.getLogger(__name__)
        self.positions: Dict = {}
        self.leverage_history = []
    
    def add_position(
        self,
        symbol: str,
        quantity: float,
        entry_price: float,
        order_id: Optional[str] = None,
    ) -> None:
        """Add position to portfolio"""
        self.positions[symbol] = {
            "quantity": quantity,
            "entry_price": entry_price,
            "order_id": order_id,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self.logger.info(f"✅ Position added: {symbol} {quantity} @ {entry_price}")
    
    def remove_position(self, symbol: str) -> None:
        """Remove position from portfolio"""
        if symbol in self.positions:
            del self.positions[symbol]
            self.logger.info(f"✅ Position removed: {symbol}")
    
    def calculate_portfolio_leverage(self, account_balance: float) -> float:
        """
        Calculate total portfolio leverage.
        
        Args:
            account_balance: Total account balance
        
        Returns:
            Portfolio leverage ratio
        """
        if account_balance <= 0:
            return 0.0
        
        total_value = sum(
            pos["quantity"] * pos["entry_price"]
            for pos in self.positions.values()
        )
        
        leverage = total_value / account_balance
        
        self.leverage_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "portfolio_leverage": leverage,
            "account_balance": account_balance,
            "total_position_value": total_value,
            "position_count": len(self.positions),
        })
        
        return leverage
    
    async def validate_portfolio_leverage(
        self,
        new_symbol: str,
        new_quantity: float,
        new_entry_price: float,
        account_balance: float,
    ) -> Tuple[bool, str, float]:
        """
        Validate portfolio leverage with new position.
        
        Args:
            new_symbol: New position symbol
            new_quantity: New position quantity
            new_entry_price: New position entry price
            account_balance: Account balance
        
        Returns:
            (is_valid, reason, new_portfolio_leverage)
        """
        try:
            # Calculate current portfolio leverage
            current_leverage = self.calculate_portfolio_leverage(account_balance)
            
            # Calculate leverage with new position
            new_position_value = new_quantity * new_entry_price
            total_value = sum(
                pos["quantity"] * pos["entry_price"]
                for pos in self.positions.values()
            ) + new_position_value
            
            new_leverage = total_value / account_balance
            
            # Check if exceeds max
            if new_leverage > self.max_portfolio_leverage:
                self.logger.warning(
                    f"❌ Portfolio leverage would exceed max: "
                    f"{new_leverage:.2f}x (max: {self.max_portfolio_leverage}x)"
                )
                return False, \
                       f"Portfolio leverage would be {new_leverage:.2f}x (max: {self.max_portfolio_leverage}x)", \
                       new_leverage
            
            self.logger.info(
                f"✅ Portfolio leverage acceptable: {new_leverage:.2f}x "
                f"(max: {self.max_portfolio_leverage}x)"
            )
            return True, f"Portfolio leverage {new_leverage:.2f}x is acceptable", new_leverage
        
        except Exception as e:
            self.logger.error(f"❌ Portfolio leverage validation failed: {e}", exc_info=True)
            return False, str(e), 0.0
    
    def get_portfolio_status(self) -> Dict:
        """Get current portfolio status"""
        return {
            "position_count": len(self.positions),
            "positions": self.positions,
            "max_portfolio_leverage": self.max_portfolio_leverage,
            "recent_leverage": self.leverage_history[-1] if self.leverage_history else None,
        }


# Global instances
_leverage_validator: Optional[LeverageValidator] = None
_leverage_monitor: Optional[LeverageMonitor] = None


def get_leverage_validator(max_leverage: float = 1.0) -> LeverageValidator:
    """Get or create global leverage validator"""
    global _leverage_validator
    if _leverage_validator is None:
        _leverage_validator = LeverageValidator(max_leverage=max_leverage)
    return _leverage_validator


def get_leverage_monitor(max_portfolio_leverage: float = 2.0) -> LeverageMonitor:
    """Get or create global leverage monitor"""
    global _leverage_monitor
    if _leverage_monitor is None:
        _leverage_monitor = LeverageMonitor(max_portfolio_leverage=max_portfolio_leverage)
    return _leverage_monitor


def initialize_leverage_management(
    max_leverage: float = 1.0,
    max_portfolio_leverage: float = 2.0,
) -> Tuple[LeverageValidator, LeverageMonitor]:
    """Initialize leverage management system"""
    global _leverage_validator, _leverage_monitor
    _leverage_validator = LeverageValidator(max_leverage=max_leverage)
    _leverage_monitor = LeverageMonitor(max_portfolio_leverage=max_portfolio_leverage)
    logger.info(
        f"✅ Leverage management initialized: "
        f"max_leverage={max_leverage}x, max_portfolio_leverage={max_portfolio_leverage}x"
    )
    return _leverage_validator, _leverage_monitor
