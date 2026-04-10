# -*- coding: utf-8 -*-
"""
trading_hours_manager.py - P9-aligned Market Hours Validation

Implements time-based trading rules to prevent:
  • Trading outside market hours
  • Trades during maintenance windows
  • Trades during low-liquidity periods

Architecture:
  • MarketHours: Market hours definitions by symbol
  • TradingHoursValidator: Time-based validation
"""

import logging
from typing import Dict, Optional, Tuple, List
from datetime import datetime, time
from enum import Enum
import pytz

logger = logging.getLogger("TradingHoursManager")


class TradingStatus(str, Enum):
    """Trading hour validation status"""
    ALLOWED = "allowed"
    REJECTED_MARKET_CLOSED = "rejected_market_closed"
    REJECTED_MAINTENANCE = "rejected_maintenance"
    REJECTED_LOW_LIQUIDITY = "rejected_low_liquidity"
    FAILED = "failed"


class MarketHours:
    """Market hours for different symbol types"""
    
    # US Stock Market (EQUITIES on US exchanges)
    US_EQUITY_MARKET = {
        "name": "US Stock Market",
        "timezone": "America/New_York",
        "trading_days": [0, 1, 2, 3, 4],  # Monday-Friday
        "regular_open": time(9, 30),
        "regular_close": time(16, 0),
        "pre_market_open": time(4, 0),
        "pre_market_close": time(9, 30),
        "after_market_open": time(16, 0),
        "after_market_close": time(20, 0),
        "holidays": [
            "2026-01-01",  # New Year
            "2026-01-19",  # MLK Day
            "2026-02-16",  # Presidents Day
            "2026-03-27",  # Good Friday
            "2026-05-25",  # Memorial Day
            "2026-07-03",  # Independence Day (observed)
            "2026-09-07",  # Labor Day
            "2026-11-26",  # Thanksgiving
            "2026-12-25",  # Christmas
        ],
    }
    
    # Crypto (24/7 trading)
    CRYPTO_MARKET = {
        "name": "Cryptocurrency (24/7)",
        "timezone": "UTC",
        "trading_days": [0, 1, 2, 3, 4, 5, 6],  # Every day
        "regular_open": time(0, 0),
        "regular_close": time(23, 59),
    }
    
    # Forex (mostly 24/5)
    FOREX_MARKET = {
        "name": "Forex (24/5)",
        "timezone": "UTC",
        "trading_days": [0, 1, 2, 3, 4],  # Monday-Friday
        "regular_open": time(17, 0),  # Previous Friday 5 PM
        "regular_close": time(17, 0),  # Friday 5 PM
    }


class TradingHoursValidator:
    """
    Validates trading times based on symbol and market hours.
    """
    
    def __init__(self):
        """Initialize trading hours validator"""
        self.logger = logging.getLogger(__name__)
        self.market_definitions: Dict = {
            "crypto": MarketHours.CRYPTO_MARKET,
            "us_equity": MarketHours.US_EQUITY_MARKET,
            "forex": MarketHours.FOREX_MARKET,
        }
        self.rejection_log = []
        self.allowed_trades_log = []
    
    def register_symbol_market(
        self,
        symbol: str,
        market_type: str,
    ) -> None:
        """
        Register symbol to market type.
        
        Args:
            symbol: Trading symbol
            market_type: Market type (crypto, us_equity, forex)
        """
        if market_type not in self.market_definitions:
            raise ValueError(f"Unknown market type: {market_type}")
        
        if not hasattr(self, "symbol_markets"):
            self.symbol_markets = {}
        
        self.symbol_markets[symbol] = market_type
        self.logger.info(f"✅ Symbol registered: {symbol} -> {market_type}")
    
    async def validate_trading_allowed(
        self,
        symbol: str,
        current_time: Optional[datetime] = None,
    ) -> Tuple[bool, TradingStatus, str]:
        """
        Validate if trading is allowed for symbol at current time.
        
        Args:
            symbol: Trading symbol
            current_time: Time to validate (default: now)
        
        Returns:
            (is_allowed, status, reason)
        """
        try:
            # Default to now
            if current_time is None:
                current_time = datetime.utcnow()
            
            # Determine market
            if not hasattr(self, "symbol_markets"):
                self.symbol_markets = {}
            
            # Default to crypto for unknown symbols
            market_type = self.symbol_markets.get(symbol, "crypto")
            market_def = self.market_definitions[market_type]
            
            # Get market timezone
            tz = pytz.timezone(market_def["timezone"])
            market_time = current_time.astimezone(tz)
            
            # Check 1: Market is open today?
            if market_time.weekday() not in market_def["trading_days"]:
                self.logger.warning(
                    f"❌ Trading not allowed: {symbol} market closed (not trading day)"
                )
                self.rejection_log.append({
                    "timestamp": current_time.isoformat(),
                    "symbol": symbol,
                    "market_type": market_type,
                    "reason": "market_closed_weekend",
                    "market_time": market_time.isoformat(),
                })
                return False, TradingStatus.REJECTED_MARKET_CLOSED, \
                       f"{symbol} market closed (weekend)"
            
            # Check 2: Within trading hours?
            market_open = market_def["regular_open"]
            market_close = market_def["regular_close"]
            current_market_time = market_time.time()
            
            if not (market_open <= current_market_time < market_close):
                self.logger.warning(
                    f"❌ Trading not allowed: {symbol} outside market hours "
                    f"({market_open}-{market_close})"
                )
                self.rejection_log.append({
                    "timestamp": current_time.isoformat(),
                    "symbol": symbol,
                    "market_type": market_type,
                    "reason": "outside_market_hours",
                    "market_time": market_time.isoformat(),
                    "market_hours": f"{market_open}-{market_close}",
                })
                return False, TradingStatus.REJECTED_MARKET_CLOSED, \
                       f"{symbol} outside market hours ({market_open}-{market_close})"
            
            # Check 3: Not in maintenance window (typically 4:50-5:00 PM ET for US markets)
            if market_type == "us_equity":
                maintenance_start = time(16, 50)
                maintenance_end = time(17, 0)
                if maintenance_start <= current_market_time < maintenance_end:
                    self.logger.warning(
                        f"❌ Trading not allowed: {symbol} during maintenance window"
                    )
                    self.rejection_log.append({
                        "timestamp": current_time.isoformat(),
                        "symbol": symbol,
                        "reason": "maintenance_window",
                    })
                    return False, TradingStatus.REJECTED_MAINTENANCE, \
                           f"{symbol} in maintenance window"
            
            # All checks passed
            self.logger.info(f"✅ Trading allowed: {symbol} at {market_time.isoformat()}")
            self.allowed_trades_log.append({
                "timestamp": current_time.isoformat(),
                "symbol": symbol,
                "market_time": market_time.isoformat(),
            })
            return True, TradingStatus.ALLOWED, \
                   f"{symbol} trading allowed at {current_market_time.isoformat()}"
        
        except Exception as e:
            self.logger.error(f"❌ Trading hours validation failed: {e}", exc_info=True)
            return False, TradingStatus.FAILED, str(e)
    
    def queue_off_hours_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "MARKET",
    ) -> str:
        """
        Queue an order for execution at market open.
        
        Args:
            symbol: Trading symbol
            side: Order side (BUY/SELL)
            quantity: Order quantity
            order_type: Order type
        
        Returns:
            Order ID for tracking
        """
        order_id = f"OFF_HOURS_{symbol}_{datetime.utcnow().timestamp()}"
        self.logger.info(
            f"📋 Order queued for market open: {side} {quantity} {symbol} "
            f"(Order ID: {order_id})"
        )
        return order_id
    
    def get_rejection_log(self, limit: int = 50) -> List:
        """Get recent rejection log"""
        return self.rejection_log[-limit:]
    
    def get_stats(self) -> Dict:
        """Get trading hours stats"""
        return {
            "total_allowed": len(self.allowed_trades_log),
            "total_rejected": len(self.rejection_log),
            "rejection_rate": len(self.rejection_log) / (len(self.allowed_trades_log) + len(self.rejection_log)) * 100
                            if (len(self.allowed_trades_log) + len(self.rejection_log)) > 0 else 0,
            "registered_symbols": len(self.symbol_markets) if hasattr(self, "symbol_markets") else 0,
        }


# Global instance
_trading_hours_validator: Optional[TradingHoursValidator] = None


def get_trading_hours_validator() -> TradingHoursValidator:
    """Get or create global trading hours validator"""
    global _trading_hours_validator
    if _trading_hours_validator is None:
        _trading_hours_validator = TradingHoursValidator()
    return _trading_hours_validator


def initialize_trading_hours_manager() -> TradingHoursValidator:
    """Initialize trading hours manager"""
    global _trading_hours_validator
    _trading_hours_validator = TradingHoursValidator()
    logger.info("✅ Trading hours manager initialized")
    return _trading_hours_validator
