# -*- coding: utf-8 -*-
"""
balance_manager.py - P9-aligned Balance Reconciliation System

Implements critical pre-allocation balance checks to prevent:
  • Capital over-allocation (prevents margin calls)
  • Double-spending of allocated capital
  • Untracked capital drift

Architecture:
  • BalanceValidator: Pre-flight balance checks
  • BalanceLedger: Immutable audit trail
  • CircuitBreaker: Fail-safe on allocation errors
"""

import logging
from typing import Dict, Optional, Tuple
from datetime import datetime
from enum import Enum
import asyncio

logger = logging.getLogger("BalanceManager")


class AllocationStatus(str, Enum):
    """Allocation status values"""
    SUCCESS = "success"
    INSUFFICIENT_BALANCE = "insufficient_balance"
    CIRCUIT_BREAKER_OPEN = "circuit_breaker_open"
    INVALID_AMOUNT = "invalid_amount"
    FAILED = "failed"


class BalanceValidator:
    """
    Pre-flight balance validation before trade allocation.
    
    Prevents over-allocation and capital loss through validation gates.
    """
    
    def __init__(self):
        """Initialize balance validator"""
        self.logger = logging.getLogger(__name__)
        self.total_balance = 0.0
        self.allocated_balance = 0.0
        self.reserved_balance = 0.0
        self.circuit_breaker_open = False
        self.failed_allocations = 0
        self.max_failed_before_circuit = 5
        self.allocation_ledger: list = []
    
    def set_total_balance(self, balance: float) -> None:
        """Set current total balance from Binance"""
        if balance < 0:
            raise ValueError("Total balance cannot be negative")
        self.total_balance = balance
        self.logger.info(f"Total balance updated: ${balance:.2f}")
    
    def get_available_balance(self) -> float:
        """Get available balance for new allocations"""
        return self.total_balance - self.allocated_balance - self.reserved_balance
    
    async def validate_allocation(
        self,
        amount: float,
        symbol: str,
        side: str,
        order_id: Optional[str] = None,
    ) -> Tuple[bool, AllocationStatus, str]:
        """
        Pre-flight validation before allocation.
        
        Args:
            amount: Amount to allocate (USDT)
            symbol: Trading symbol
            side: Order side (BUY/SELL)
            order_id: Optional order ID for tracking
        
        Returns:
            (is_valid, status, reason)
        """
        try:
            # Check 1: Circuit breaker
            if self.circuit_breaker_open:
                self.logger.error(
                    "❌ Allocation rejected: Circuit breaker open (too many failures)"
                )
                return False, AllocationStatus.CIRCUIT_BREAKER_OPEN, \
                       "Circuit breaker open due to repeated failures"
            
            # Check 2: Valid amount
            if amount <= 0:
                self.logger.error(f"❌ Invalid allocation amount: {amount}")
                return False, AllocationStatus.INVALID_AMOUNT, \
                       f"Amount must be positive, got {amount}"
            
            # Check 3: Sufficient balance
            available = self.get_available_balance()
            if amount > available:
                self.logger.error(
                    f"❌ Insufficient balance: need ${amount:.2f}, have ${available:.2f}"
                )
                return False, AllocationStatus.INSUFFICIENT_BALANCE, \
                       f"Insufficient balance: need ${amount:.2f}, available ${available:.2f}"
            
            # Check 4: Reserved balance protection (for fees, etc)
            reserved_after = self.allocated_balance + amount + self.reserved_balance
            if reserved_after > self.total_balance * 0.98:  # Max 98% deployment
                self.logger.error(
                    f"❌ Over-deployment protection: would deploy {reserved_after / self.total_balance * 100:.1f}%"
                )
                return False, AllocationStatus.INSUFFICIENT_BALANCE, \
                       "Allocation would exceed maximum deployment ratio"
            
            # All checks passed
            self.logger.info(
                f"✅ Allocation validated: ${amount:.2f} {side} {symbol}"
            )
            return True, AllocationStatus.SUCCESS, "Validation passed"
        
        except Exception as e:
            self.logger.error(f"❌ Allocation validation failed: {e}", exc_info=True)
            return False, AllocationStatus.FAILED, str(e)
    
    async def commit_allocation(
        self,
        amount: float,
        symbol: str,
        side: str,
        order_id: str,
    ) -> bool:
        """
        Commit allocation to ledger (after trade executes).
        
        Args:
            amount: Amount allocated
            symbol: Trading symbol
            side: Order side
            order_id: Order ID
        
        Returns:
            True if committed successfully
        """
        try:
            # Verify amount is still valid
            available = self.get_available_balance()
            if amount > available:
                self.failed_allocations += 1
                self._check_circuit_breaker()
                self.logger.error(
                    f"❌ Cannot commit allocation: insufficient balance"
                )
                return False
            
            # Commit to ledger
            self.allocated_balance += amount
            self.allocation_ledger.append({
                "timestamp": datetime.utcnow().isoformat(),
                "symbol": symbol,
                "side": side,
                "amount": amount,
                "order_id": order_id,
                "status": "committed",
                "total_balance": self.total_balance,
                "allocated_after": self.allocated_balance,
            })
            
            self.logger.info(
                f"✅ Allocation committed: ${amount:.2f} {side} {symbol} "
                f"(Allocated: ${self.allocated_balance:.2f})"
            )
            return True
        
        except Exception as e:
            self.failed_allocations += 1
            self._check_circuit_breaker()
            self.logger.error(f"❌ Failed to commit allocation: {e}", exc_info=True)
            return False
    
    async def release_allocation(
        self,
        amount: float,
        symbol: str,
        order_id: str,
        reason: str = "position_closed",
    ) -> bool:
        """
        Release allocation when trade closes.
        
        Args:
            amount: Amount to release
            symbol: Trading symbol
            order_id: Order ID
            reason: Reason for release (position_closed, error, etc)
        
        Returns:
            True if released successfully
        """
        try:
            if amount > self.allocated_balance:
                self.logger.error(
                    f"❌ Cannot release ${amount:.2f}: only ${self.allocated_balance:.2f} allocated"
                )
                return False
            
            self.allocated_balance -= amount
            self.allocation_ledger.append({
                "timestamp": datetime.utcnow().isoformat(),
                "symbol": symbol,
                "order_id": order_id,
                "amount": amount,
                "status": f"released_{reason}",
                "allocated_after": self.allocated_balance,
            })
            
            self.logger.info(
                f"✅ Allocation released: ${amount:.2f} {symbol} ({reason}) "
                f"(Allocated: ${self.allocated_balance:.2f})"
            )
            
            # Reset failures on successful release
            if self.failed_allocations > 0:
                self.failed_allocations = 0
            
            return True
        
        except Exception as e:
            self.logger.error(f"❌ Failed to release allocation: {e}", exc_info=True)
            return False
    
    def _check_circuit_breaker(self) -> None:
        """Check if circuit breaker should open"""
        if self.failed_allocations >= self.max_failed_before_circuit:
            self.circuit_breaker_open = True
            self.logger.error(
                f"🚨 CIRCUIT BREAKER OPENED: {self.failed_allocations} "
                f"allocation failures. Disabling new allocations."
            )
    
    def reset_circuit_breaker(self) -> None:
        """Reset circuit breaker after issue is resolved"""
        self.circuit_breaker_open = False
        self.failed_allocations = 0
        self.logger.info("✅ Circuit breaker reset")
    
    def get_audit_trail(self, limit: int = 100) -> list:
        """Get recent allocation audit trail"""
        return self.allocation_ledger[-limit:]
    
    def get_status(self) -> Dict:
        """Get current balance manager status"""
        return {
            "total_balance": self.total_balance,
            "allocated_balance": self.allocated_balance,
            "available_balance": self.get_available_balance(),
            "reserved_balance": self.reserved_balance,
            "allocation_ratio": (self.allocated_balance / self.total_balance * 100) 
                              if self.total_balance > 0 else 0,
            "circuit_breaker_open": self.circuit_breaker_open,
            "failed_allocations": self.failed_allocations,
            "audit_trail_count": len(self.allocation_ledger),
        }


# Global balance manager instance
_balance_validator: Optional[BalanceValidator] = None


def get_balance_validator() -> BalanceValidator:
    """Get or create global balance validator"""
    global _balance_validator
    if _balance_validator is None:
        _balance_validator = BalanceValidator()
    return _balance_validator


def initialize_balance_manager() -> BalanceValidator:
    """Initialize balance manager"""
    global _balance_validator
    _balance_validator = BalanceValidator()
    logger.info("✅ Balance manager initialized")
    return _balance_validator
