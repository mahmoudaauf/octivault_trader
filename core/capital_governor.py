# -*- coding: utf-8 -*-
"""
Capital Governor - Best Practice Decision Tree for Symbol Rotation & Position Sizing

This module implements the Best Practice Decision Tree for capital-aware trading:

╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                  BEST PRACTICE DECISION TREE (SMALL CAPITAL)                ║
║                                                                              ║
║  If equity < $500:                                                          ║
║    ├─ Fix 1–2 core pairs (no rotation)                                      ║
║    ├─ Allow 1 rotating slot max                                            ║
║    └─ Maximize learning, minimize capital bleed                             ║
║                                                                              ║
║  Else (equity >= $500):                                                     ║
║    ├─ Allow 5–10 rotating symbols                                          ║
║    └─ Institutionalize: strict gates, profit lock, diversification          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Usage:
    from core.capital_governor import CapitalGovernor
    
    governor = CapitalGovernor(config)
    
    # Check position limits for NAV
    nav = 350.0
    limits = governor.get_position_limits(nav)
    # => {"max_active_symbols": 2, "max_rotating_slots": 1, "core_pairs": 2, ...}
    
    # Get position sizing (defaults to appropriate profile)
    sizing = governor.get_position_sizing(nav, symbol)
    # => {"quote_per_position": 12.0, "max_per_symbol": 24.0, ...}

Architecture Notes:
    - Decision tree is applied once per major phase update (e.g., P7, P8)
    - Position limits cached in SharedState for fast access
    - Changes to position allocation handled by PositionManager integration
    - Symbol rotation respects the rotating_slots allocation
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class CapitalBracket(Enum):
    """Capital bracket classification for position management."""
    MICRO = "micro"          # < $500 (learning phase)
    SMALL = "small"          # $500-$2000 (growth phase)
    MEDIUM = "medium"        # $2000-$10000 (scaling phase)
    LARGE = "large"          # >= $10000 (institutional phase)


class CapitalGovernor:
    """
    Implements Best Practice Decision Tree for capital-aware trading.
    
    Automatically adjusts position limits, symbol rotation allowances,
    and sizing based on account equity bracket.
    
    CRITICAL: Always syncs authoritative balance before reading NAV to ensure
    fresh, accurate account data.
    """
    
    def __init__(self, config, shared_state=None):
        """
        Initialize Capital Governor.
        
        Args:
            config: Configuration object with capital thresholds
            shared_state: Optional SharedState reference for direct NAV access
        """
        self.config = config
        self.shared_state = shared_state  # For sync_authoritative_balance()
        
        # Capital bracket thresholds
        self.micro_threshold = 500.0      # < $500: micro bracket
        self.small_threshold = 2000.0     # $500-$2000: small bracket
        self.medium_threshold = 10000.0   # $2000-$10000: medium bracket
        # >= $10000: large bracket
        
        logger.info(
            "[CapitalGovernor] Initialized with brackets: "
            "MICRO=<$%.0f, SMALL=$%.0f-$%.0f, MEDIUM=$%.0f-$%.0f, LARGE>=$%.0f",
            self.micro_threshold,
            self.micro_threshold,
            self.small_threshold,
            self.small_threshold,
            self.medium_threshold,
            self.medium_threshold
        )
    
    async def get_fresh_nav(self) -> float:
        """
        Get fresh NAV by syncing authoritative balance first.
        
        CRITICAL: This ensures NAV is accurate and not stale.
        
        Returns:
            Fresh NAV value from account
        """
        nav = 0.0
        
        try:
            # Step 1: Sync authoritative balance if available
            if self.shared_state and hasattr(self.shared_state, "sync_authoritative_balance"):
                try:
                    await self.shared_state.sync_authoritative_balance(force=True)
                    logger.debug("[CapitalGovernor] Synced authoritative balance for NAV")
                except Exception as e:
                    logger.warning("[CapitalGovernor] Failed to sync balance: %s", e)
            
            # Step 2: Read fresh NAV from shared_state
            if self.shared_state:
                # Try multiple NAV sources in order of preference
                nav = float(getattr(self.shared_state, "nav", None) or 
                           getattr(self.shared_state, "total_value", None) or 
                           getattr(self.shared_state, "total_balance", None) or 0.0)
            
            if nav <= 0:
                logger.warning("[CapitalGovernor] Fresh NAV is invalid: %.2f", nav)
                return 0.0
            
            logger.debug("[CapitalGovernor] Fresh NAV obtained: $%.2f", nav)
            return nav
            
        except Exception as e:
            logger.error("[CapitalGovernor:FreshNAV] Failed to get fresh NAV: %s", e)
            return 0.0
    
    def get_nav_sync_required(self, nav_source: str = "parameter") -> Tuple[bool, str]:
        """
        Check if NAV sync is required before using Governor.
        
        Args:
            nav_source: Source of NAV ("parameter", "cached", "shared_state")
            
        Returns:
            Tuple[bool, str] = (sync_required, reason)
        """
        if nav_source == "parameter":
            return False, "NAV passed as parameter (assumed fresh)"
        elif nav_source == "cached":
            return True, "NAV is cached (may be stale)"
        elif nav_source == "shared_state":
            return True, "NAV from shared_state (sync recommended)"
        else:
            return True, "Unknown NAV source (sync recommended)"
    
    def get_bracket(self, nav: float) -> CapitalBracket:
        """
        Classify equity into capital bracket.
        
        Args:
            nav: Net Asset Value (current account equity)
            
        Returns:
            CapitalBracket enum indicating current bracket
        """
        if nav < self.micro_threshold:
            return CapitalBracket.MICRO
        elif nav < self.small_threshold:
            return CapitalBracket.SMALL
        elif nav < self.medium_threshold:
            return CapitalBracket.MEDIUM
        else:
            return CapitalBracket.LARGE
    
    def get_position_limits(self, nav: float) -> Dict[str, Any]:
        """
        Get position limits based on capital bracket (BEST PRACTICE DECISION TREE).
        
        Args:
            nav: Net Asset Value (current account equity)
            
        Returns:
            Dict with position limits:
            {
                "bracket": CapitalBracket,
                "max_active_symbols": int,           # Total symbols to track
                "max_rotating_slots": int,           # Symbols eligible for rotation
                "core_pairs": int,                   # Fixed core symbols (no rotation)
                "max_concurrent_positions": int,     # Max open at same time
                "allow_rotation": bool,              # Can rotate symbols
                "symbol_replacement_multiplier": float,  # Quality bar for rotation
                "soft_lock_duration_sec": int,       # How long to lock after trade
            }
        """
        bracket = self.get_bracket(nav)
        
        if bracket == CapitalBracket.MICRO:
            # MICRO BRACKET: < $500
            # Best Practice: Fix 1-2 core pairs, allow 1 rotating slot max
            limits = {
                "bracket": bracket.value,
                "max_active_symbols": 2,              # Only 2 symbols total
                "core_pairs": 2,                      # Both are core (no rotation)
                "max_rotating_slots": 0,              # No rotation allowed!
                "max_concurrent_positions": 1,        # Only 1 position at a time
                "allow_rotation": False,              # Disable rotation
                "symbol_replacement_multiplier": 2.0, # Require 100% improvement (unreachable)
                "soft_lock_duration_sec": 86400,      # Lock for 24 hours (1 day)
                "rotation_mode": "NONE",
                "reason": "MICRO_BRACKET: Focus on 2 core pairs for learning"
            }
            
        elif bracket == CapitalBracket.SMALL:
            # SMALL BRACKET: $500-$2000
            # Best Practice: Allow 1-2 rotating slots, keep 2-3 core
            limits = {
                "bracket": bracket.value,
                "max_active_symbols": 5,              # Up to 5 symbols
                "core_pairs": 2,                      # 2 core (no rotation)
                "max_rotating_slots": 1,              # 1 slot for rotation (+ 2 core = 3 total)
                "max_concurrent_positions": 2,        # Up to 2 positions
                "allow_rotation": True,
                "symbol_replacement_multiplier": 1.50,  # Require 50% improvement
                "soft_lock_duration_sec": 3600,       # Lock for 1 hour
                "rotation_mode": "CONSERVATIVE",
                "reason": "SMALL_BRACKET: 2 core + 1 rotating = stable growth"
            }
            
        elif bracket == CapitalBracket.MEDIUM:
            # MEDIUM BRACKET: $2000-$10000
            # Best Practice: Scale to 3-5 rotating slots
            limits = {
                "bracket": bracket.value,
                "max_active_symbols": 10,             # Up to 10 symbols
                "core_pairs": 3,                      # 3 core (no rotation)
                "max_rotating_slots": 5,              # 5 slots for rotation
                "max_concurrent_positions": 3,        # Up to 3 positions
                "allow_rotation": True,
                "symbol_replacement_multiplier": 1.25,  # Require 25% improvement
                "soft_lock_duration_sec": 1800,       # Lock for 30 minutes
                "rotation_mode": "MODERATE",
                "reason": "MEDIUM_BRACKET: 3 core + 5 rotating = scaling phase"
            }
            
        else:  # LARGE bracket (>= $10000)
            # LARGE BRACKET: >= $10000
            # Best Practice: Full institutional rotation (5-10 rotating)
            limits = {
                "bracket": bracket.value,
                "max_active_symbols": 20,             # Up to 20 symbols
                "core_pairs": 5,                      # 5 core
                "max_rotating_slots": 10,             # 10 slots for rotation
                "max_concurrent_positions": 5,        # Up to 5 positions
                "allow_rotation": True,
                "symbol_replacement_multiplier": 1.10,  # Require 10% improvement
                "soft_lock_duration_sec": 300,        # Lock for 5 minutes
                "rotation_mode": "AGGRESSIVE",
                "reason": "LARGE_BRACKET: 5 core + 10 rotating = institutional diversification"
            }
        
        logger.info(
            "[CapitalGovernor:PositionLimits] NAV=$%.2f → %s bracket: "
            "%d active symbols (%d core + %d rotating), %d max positions, rotation=%s",
            nav,
            limits["bracket"],
            limits["max_active_symbols"],
            limits["core_pairs"],
            limits["max_rotating_slots"],
            limits["max_concurrent_positions"],
            limits["allow_rotation"]
        )
        
        return limits
    
    def get_position_sizing(self, nav: float, symbol: str = "") -> Dict[str, float]:
        """
        Get position sizing based on capital bracket.
        
        Args:
            nav: Net Asset Value (current account equity)
            symbol: Symbol being sized (optional, for logging)
            
        Returns:
            Dict with sizing parameters:
            {
                "quote_per_position": float,     # USDT per position
                "max_per_symbol": float,         # Max total USDT per symbol
                "portfolio_allocation_pct": float,  # % of capital per position
                "min_order_usdt": float,         # Minimum order size
                "enable_profit_lock": bool,      # Enable profit lock
                "ev_multiplier": float,          # EV gate multiplier
            }
        """
        bracket = self.get_bracket(nav)
        
        if bracket == CapitalBracket.MICRO:
            sizing = {
                "quote_per_position": 12.0,       # Very small orders
                "max_per_symbol": 24.0,           # Max 2x position per symbol
                "portfolio_allocation_pct": 5.0,  # 5% of capital per position
                "min_order_usdt": 12.0,
                "enable_profit_lock": False,      # Learning phase
                "ev_multiplier": 1.4,             # Permissive gate
            }
            
        elif bracket == CapitalBracket.SMALL:
            sizing = {
                "quote_per_position": 15.0,
                "max_per_symbol": 30.0,
                "portfolio_allocation_pct": 3.0,  # 3% per position
                "min_order_usdt": 15.0,
                "enable_profit_lock": False,
                "ev_multiplier": 1.6,
            }
            
        elif bracket == CapitalBracket.MEDIUM:
            sizing = {
                "quote_per_position": 25.0,
                "max_per_symbol": 75.0,
                "portfolio_allocation_pct": 2.0,  # 2% per position
                "min_order_usdt": 20.0,
                "enable_profit_lock": True,       # Start locking profits
                "ev_multiplier": 1.8,
            }
            
        else:  # LARGE bracket
            sizing = {
                "quote_per_position": 50.0,
                "max_per_symbol": 150.0,
                "portfolio_allocation_pct": 1.0,  # 1% per position
                "min_order_usdt": 30.0,
                "enable_profit_lock": True,
                "ev_multiplier": 2.0,             # Strict gate
            }
        
        if symbol:
            logger.debug(
                "[CapitalGovernor:Sizing] NAV=$%.2f → %s: "
                "$%.1f per position, EV×%.1f, profit_lock=%s",
                nav,
                symbol,
                sizing["quote_per_position"],
                sizing["ev_multiplier"],
                sizing["enable_profit_lock"]
            )
        
        return sizing
    
    def should_restrict_rotation(self, nav: float) -> bool:
        """
        Check if rotation should be restricted (for MICRO bracket).
        
        Args:
            nav: Net Asset Value
            
        Returns:
            True if rotation is disabled for this bracket
        """
        bracket = self.get_bracket(nav)
        return bracket == CapitalBracket.MICRO
    
    def get_recommended_core_pairs(self, nav: float, available_symbols: Optional[List[str]] = None) -> List[str]:
        """
        Get list of recommended core pairs for this bracket.
        
        Args:
            nav: Net Asset Value
            available_symbols: List of available symbols to choose from
            
        Returns:
            List of recommended core symbols
        """
        limits = self.get_position_limits(nav)
        core_count = limits["core_pairs"]
        
        if not available_symbols:
            logger.warning("[CapitalGovernor] No available symbols provided")
            return []
        
        # Recommendation: Choose highest liquidity / most stable
        # For now, return first N symbols (caller should sort by liquidity first)
        core_pairs = available_symbols[:core_count]
        
        logger.info(
            "[CapitalGovernor:CorePairs] NAV=$%.2f → Recommended %d core pairs: %s",
            nav,
            core_count,
            ", ".join(core_pairs)
        )
        
        return core_pairs
    
    def validate_symbol_for_bracket(self, nav: float, symbol: str, is_core: bool = False) -> bool:
        """
        Check if a symbol can be traded in this bracket.
        
        Args:
            nav: Net Asset Value
            symbol: Symbol to validate
            is_core: Is this a core pair (always allowed)?
            
        Returns:
            True if symbol is allowed in this bracket
        """
        bracket = self.get_bracket(nav)
        
        if is_core:
            return True  # Core pairs always allowed
        
        # Non-core symbols (rotating) check
        if bracket == CapitalBracket.MICRO:
            # MICRO: No rotation at all
            return False
        
        # All other brackets allow rotation
        return True
    
    def format_limits_for_display(self, nav: float) -> str:
        """
        Format position limits as human-readable string.
        
        Args:
            nav: Net Asset Value
            
        Returns:
            Formatted string for logging
        """
        limits = self.get_position_limits(nav)
        sizing = self.get_position_sizing(nav)
        
        return (
            f"[CapitalGovernor Report]\n"
            f"  Equity: ${nav:.2f}\n"
            f"  Bracket: {limits['bracket'].upper()}\n"
            f"  Position Limits:\n"
            f"    - Max Active Symbols: {limits['max_active_symbols']}\n"
            f"    - Core Pairs: {limits['core_pairs']}\n"
            f"    - Rotating Slots: {limits['max_rotating_slots']}\n"
            f"    - Max Concurrent Positions: {limits['max_concurrent_positions']}\n"
            f"    - Rotation Allowed: {limits['allow_rotation']}\n"
            f"  Position Sizing:\n"
            f"    - Per Position: ${sizing['quote_per_position']:.2f}\n"
            f"    - Portfolio Allocation: {sizing['portfolio_allocation_pct']:.1f}%\n"
            f"    - EV Multiplier: {sizing['ev_multiplier']:.1f}x\n"
            f"    - Profit Lock: {sizing['enable_profit_lock']}\n"
            f"  Reason: {limits['reason']}"
        )


def initialize_capital_governor(config) -> CapitalGovernor:
    """
    Factory function to create and initialize CapitalGovernor.
    
    Args:
        config: Configuration object
        
    Returns:
        Initialized CapitalGovernor instance
    """
    governor = CapitalGovernor(config)
    logger.info("[CapitalGovernor] Factory initialized")
    return governor
