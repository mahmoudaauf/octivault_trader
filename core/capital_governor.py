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
import time
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
        
        # Capital bracket thresholds — configurable via CAPITAL_*_THRESHOLD, fall back to defaults.
        # Allows threshold adjustment without code changes; changes take effect on next init.
        self.micro_threshold = float(getattr(config, "CAPITAL_MICRO_THRESHOLD", 500.0) or 500.0)
        self.small_threshold = float(getattr(config, "CAPITAL_SMALL_THRESHOLD", 2000.0) or 2000.0)
        self.medium_threshold = float(getattr(config, "CAPITAL_MEDIUM_THRESHOLD", 10000.0) or 10000.0)
        # >= medium_threshold: LARGE bracket

        # MICRO bracket policy knobs (env-configurable through Config)
        self.micro_max_active_symbols = max(
            1, int(getattr(config, "CAPITAL_MICRO_MAX_ACTIVE_SYMBOLS", 3) or 3)
        )
        self.micro_core_pairs = max(
            1, int(getattr(config, "CAPITAL_MICRO_CORE_PAIRS", 2) or 2)
        )
        self.micro_rotating_slots = max(
            0, int(getattr(config, "CAPITAL_MICRO_MAX_ROTATING_SLOTS", 1) or 1)
        )
        self.micro_max_concurrent_positions = max(
            1, int(getattr(config, "CAPITAL_MICRO_MAX_CONCURRENT_POSITIONS", 2) or 2)
        )
        self.micro_allow_rotation = bool(
            getattr(config, "CAPITAL_MICRO_ALLOW_ROTATION", True)
        )
        self.micro_replacement_multiplier = float(
            getattr(config, "CAPITAL_MICRO_SYMBOL_REPLACEMENT_MULTIPLIER", 1.35) or 1.35
        )
        self.micro_soft_lock_sec = max(
            60, int(getattr(config, "CAPITAL_MICRO_SOFT_LOCK_DURATION_SEC", 3600) or 3600)
        )
        self.micro_enforce_liveness = bool(
            getattr(config, "CAPITAL_MICRO_ENFORCE_LIVENESS", True)
        )

        # Adaptive micro-capacity escape hatch: respond to POSITION_ALREADY_OPEN pressure.
        self.micro_adaptive_enabled = bool(
            getattr(config, "CAPITAL_MICRO_ADAPTIVE_CAPACITY_ENABLED", True)
        )
        self.micro_adaptive_trigger = max(
            1, int(getattr(config, "CAPITAL_MICRO_ADAPTIVE_PRESSURE_TRIGGER", 6) or 6)
        )
        self.micro_adaptive_window_sec = max(
            30.0, float(getattr(config, "CAPITAL_MICRO_ADAPTIVE_WINDOW_SEC", 300.0) or 300.0)
        )
        self.micro_adaptive_max_positions = max(
            self.micro_max_concurrent_positions,
            int(getattr(config, "CAPITAL_MICRO_ADAPTIVE_MAX_CONCURRENT_POSITIONS", 2) or 2),
        )
        self.micro_adaptive_rotating_slots = max(
            self.micro_rotating_slots,
            int(getattr(config, "CAPITAL_MICRO_ADAPTIVE_MAX_ROTATING_SLOTS", 1) or 1),
        )
        self.micro_adaptive_soft_lock_sec = max(
            60, int(getattr(config, "CAPITAL_MICRO_ADAPTIVE_SOFT_LOCK_DURATION_SEC", 900) or 900)
        )
        self.micro_adaptive_replacement_multiplier = float(
            getattr(config, "CAPITAL_MICRO_ADAPTIVE_SYMBOL_REPLACEMENT_MULTIPLIER", 1.35) or 1.35
        )
        self.micro_adaptive_sell_deadlock_enabled = bool(
            getattr(config, "CAPITAL_MICRO_ADAPTIVE_SELL_DEADLOCK_ENABLED", True)
        )
        self.micro_adaptive_sell_deadlock_trigger = max(
            1, int(getattr(config, "CAPITAL_MICRO_ADAPTIVE_SELL_DEADLOCK_TRIGGER", 5) or 5)
        )
        self.micro_adaptive_sell_deadlock_window_sec = max(
            30.0,
            float(
                getattr(
                    config,
                    "CAPITAL_MICRO_ADAPTIVE_SELL_DEADLOCK_WINDOW_SEC",
                    self.micro_adaptive_window_sec,
                )
                or self.micro_adaptive_window_sec
            ),
        )
        sell_deadlock_reasons_csv = str(
            getattr(
                config,
                "CAPITAL_MICRO_ADAPTIVE_SELL_DEADLOCK_REASONS",
                "PORTFOLIO_PNL_IMPROVEMENT,SELL_DYNAMIC_EDGE_MIN,CLOSE_NOT_SUBMITTED,SELL_NET_PNL_MIN",
            )
            or ""
        )
        sell_deadlock_reasons = [
            token.strip().upper()
            for token in sell_deadlock_reasons_csv.split(",")
            if str(token).strip()
        ]
        self.micro_adaptive_sell_deadlock_reasons = tuple(sell_deadlock_reasons)

        # Prevent deadlock-prone micro setup (single slot + no rotation) unless explicitly disabled.
        if self.micro_enforce_liveness:
            self._apply_micro_liveness_guard()
        
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

    def _apply_micro_liveness_guard(self) -> None:
        """
        Ensure micro profile can execute at least one new opportunity while holding one position.

        This preserves conservative risk while preventing perpetual decision=NONE loops
        caused by (max_positions=1, rotation=False).
        """
        deadlock_prone = (
            int(self.micro_max_concurrent_positions) <= 1
            and int(self.micro_rotating_slots) <= 0
            and not bool(self.micro_allow_rotation)
        )
        if not deadlock_prone:
            return

        self.micro_max_concurrent_positions = max(2, int(self.micro_max_concurrent_positions))
        self.micro_rotating_slots = max(1, int(self.micro_rotating_slots))
        self.micro_allow_rotation = True
        self.micro_max_active_symbols = max(
            int(self.micro_max_active_symbols),
            int(self.micro_core_pairs) + int(self.micro_rotating_slots),
        )
        self.micro_replacement_multiplier = min(float(self.micro_replacement_multiplier), 1.35)
        self.micro_soft_lock_sec = min(int(self.micro_soft_lock_sec), 3600)
        logger.warning(
            "[CapitalGovernor] Applied MICRO liveness guard: active_symbols=%d core_pairs=%d "
            "rotating_slots=%d max_positions=%d rotation=%s replacement_multiplier=%.2f soft_lock=%ss",
            self.micro_max_active_symbols,
            self.micro_core_pairs,
            self.micro_rotating_slots,
            self.micro_max_concurrent_positions,
            self.micro_allow_rotation,
            self.micro_replacement_multiplier,
            self.micro_soft_lock_sec,
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

    def _get_position_open_pressure(self) -> int:
        """
        Estimate recent POSITION_ALREADY_OPEN BUY rejection pressure.

        Uses SharedState rejection_history when available and falls back to
        rejection_counters. This keeps pressure local in time and avoids
        permanently widening limits from stale historical rejections.
        """
        if not self.shared_state:
            return 0

        now_ts = time.time()
        window_sec = float(self.micro_adaptive_window_sec)
        pressure = 0

        try:
            history = list(getattr(self.shared_state, "rejection_history", []) or [])
            for item in history:
                if not isinstance(item, dict):
                    continue
                if str(item.get("side", "")).upper() != "BUY":
                    continue
                if str(item.get("reason", "")).upper() != "POSITION_ALREADY_OPEN":
                    continue
                ts = float(item.get("ts", 0.0) or 0.0)
                if ts <= 0 or (now_ts - ts) > window_sec:
                    continue
                pressure += 1
        except Exception:
            pressure = 0

        if pressure > 0:
            return int(pressure)

        # Fallback: aggregate from counters when history is unavailable.
        try:
            counters = getattr(self.shared_state, "rejection_counters", {}) or {}
            timestamps = getattr(self.shared_state, "rejection_timestamps", {}) or {}
            for key, count in counters.items():
                if not isinstance(key, tuple) or len(key) < 3:
                    continue
                sym, side, reason = key[0], key[1], key[2]
                if str(side).upper() != "BUY":
                    continue
                if str(reason).upper() != "POSITION_ALREADY_OPEN":
                    continue
                ts = float(timestamps.get(key, 0.0) or 0.0)
                if ts > 0 and (now_ts - ts) <= window_sec:
                    pressure += int(count or 0)
        except Exception:
            return 0

        return max(0, int(pressure))

    def _get_sell_deadlock_pressure(self) -> int:
        """
        Estimate recent SELL deadlock pressure in micro mode.

        Captures repeated SELL rejections that block capacity recycling
        (for example portfolio improvement or dynamic edge guards), then
        allows the governor to temporarily widen micro limits.
        """
        if not self.shared_state or not self.micro_adaptive_sell_deadlock_enabled:
            return 0

        reasons = set(self.micro_adaptive_sell_deadlock_reasons)
        if not reasons:
            return 0

        now_ts = time.time()
        window_sec = float(self.micro_adaptive_sell_deadlock_window_sec)
        pressure = 0

        try:
            history = list(getattr(self.shared_state, "rejection_history", []) or [])
            for item in history:
                if not isinstance(item, dict):
                    continue
                if str(item.get("side", "")).upper() != "SELL":
                    continue
                if str(item.get("reason", "")).upper() not in reasons:
                    continue
                ts = float(item.get("ts", 0.0) or 0.0)
                if ts <= 0 or (now_ts - ts) > window_sec:
                    continue
                pressure += 1
        except Exception:
            pressure = 0

        if pressure > 0:
            return int(pressure)

        # Fallback to counters when detailed history is unavailable.
        try:
            counters = getattr(self.shared_state, "rejection_counters", {}) or {}
            timestamps = getattr(self.shared_state, "rejection_timestamps", {}) or {}
            for key, count in counters.items():
                if not isinstance(key, tuple) or len(key) < 3:
                    continue
                _, side, reason = key[0], key[1], key[2]
                if str(side).upper() != "SELL":
                    continue
                if str(reason).upper() not in reasons:
                    continue
                ts = float(timestamps.get(key, 0.0) or 0.0)
                if ts > 0 and (now_ts - ts) <= window_sec:
                    pressure += int(count or 0)
        except Exception:
            return 0

        return max(0, int(pressure))
    
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
                "max_active_symbols": self.micro_max_active_symbols,
                "core_pairs": self.micro_core_pairs,
                "max_rotating_slots": self.micro_rotating_slots,
                "max_concurrent_positions": self.micro_max_concurrent_positions,
                "allow_rotation": self.micro_allow_rotation,
                "symbol_replacement_multiplier": self.micro_replacement_multiplier,
                "soft_lock_duration_sec": self.micro_soft_lock_sec,
                "rotation_mode": "NONE",
                "reason": "MICRO_BRACKET: Focus on 2 core pairs for learning"
            }

            # Adaptive unlock for micro deadlocks:
            # when BUYs repeatedly fail with POSITION_ALREADY_OPEN, temporarily allow
            # one extra slot + limited rotation to restore throughput.
            pressure = self._get_position_open_pressure()
            sell_deadlock_pressure = self._get_sell_deadlock_pressure()
            adaptive_open_pressure = pressure >= self.micro_adaptive_trigger
            adaptive_sell_deadlock = (
                self.micro_adaptive_sell_deadlock_enabled
                and sell_deadlock_pressure >= self.micro_adaptive_sell_deadlock_trigger
            )
            if self.micro_adaptive_enabled and (adaptive_open_pressure or adaptive_sell_deadlock):
                limits["max_concurrent_positions"] = int(self.micro_adaptive_max_positions)
                limits["max_rotating_slots"] = int(self.micro_adaptive_rotating_slots)
                limits["allow_rotation"] = bool(limits["max_rotating_slots"] > 0)
                limits["max_active_symbols"] = max(
                    int(limits["max_active_symbols"]),
                    int(limits["core_pairs"]) + int(limits["max_rotating_slots"]),
                )
                limits["soft_lock_duration_sec"] = int(
                    min(limits["soft_lock_duration_sec"], self.micro_adaptive_soft_lock_sec)
                )
                limits["symbol_replacement_multiplier"] = float(
                    min(
                        float(limits["symbol_replacement_multiplier"]),
                        float(self.micro_adaptive_replacement_multiplier),
                    )
                )
                limits["rotation_mode"] = "MICRO_ADAPTIVE"
                adaptive_reasons = []
                if adaptive_open_pressure:
                    adaptive_reasons.append(
                        f"POSITION_ALREADY_OPEN pressure ({pressure} rejects/{int(self.micro_adaptive_window_sec)}s)"
                    )
                if adaptive_sell_deadlock:
                    adaptive_reasons.append(
                        f"SELL deadlock pressure ({sell_deadlock_pressure} rejects/{int(self.micro_adaptive_sell_deadlock_window_sec)}s)"
                    )
                if not adaptive_reasons:
                    adaptive_reasons.append("pressure trigger")
                limits["reason"] = (
                    "MICRO_ADAPTIVE: unlocked extra slot/rotation due to "
                    + " + ".join(adaptive_reasons)
                )
            
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
    
    def get_position_sizing(self, nav: float, symbol: str = "", current_position_value: float = 0.0) -> Dict[str, float]:
        """
        Get position sizing based on capital bracket WITH CONCENTRATION GATING.
        
        ===== PHASE 5: PRE-TRADE RISK GATE (Concentration-Aware Sizing) =====
        
        CRITICAL FIX: This implements risk enforcement BEFORE execution, not after.
        
        Instead of:  Signal → BUY huge position → System tries to fix
        Now we do:  Signal → Check concentration → Return adjusted size
        
        Args:
            nav: Net Asset Value (current account equity)
            symbol: Symbol being sized (optional, for logging)
            current_position_value: Current market value of position in this symbol (USDT)
            
        Returns:
            Dict with sizing parameters:
            {
                "quote_per_position": float,     # USDT per position (CONCENTRATION-CAPPED)
                "max_per_symbol": float,         # Max total USDT per symbol
                "max_position_pct": float,       # Max % of NAV for single position
                "portfolio_allocation_pct": float,  # % of capital per position
                "min_order_usdt": float,         # Minimum order size
                "enable_profit_lock": bool,      # Enable profit lock
                "ev_multiplier": float,          # EV gate multiplier
                "concentration_headroom": float, # Remaining headroom (USDT) before limit
            }
        """
        bracket = self.get_bracket(nav)
        
        # Step 1: Get base sizing for bracket
        if bracket == CapitalBracket.MICRO:
            sizing = {
                "quote_per_position": 12.0,       # Very small orders
                "max_per_symbol": 24.0,           # Max 2x position per symbol
                "max_position_pct": 0.50,         # ← NEW: Max 50% of NAV per position (MICRO)
                "portfolio_allocation_pct": 5.0,  # 5% of capital per position
                "min_order_usdt": 12.0,
                "enable_profit_lock": False,      # Learning phase
                "ev_multiplier": 1.4,             # Permissive gate
            }
            
        elif bracket == CapitalBracket.SMALL:
            sizing = {
                "quote_per_position": 15.0,
                "max_per_symbol": 30.0,
                "max_position_pct": 0.35,         # ← NEW: Max 35% of NAV per position (SMALL)
                "portfolio_allocation_pct": 3.0,  # 3% per position
                "min_order_usdt": 15.0,
                "enable_profit_lock": False,
                "ev_multiplier": 1.6,
            }
            
        elif bracket == CapitalBracket.MEDIUM:
            sizing = {
                "quote_per_position": 25.0,
                "max_per_symbol": 75.0,
                "max_position_pct": 0.25,         # ← NEW: Max 25% of NAV per position (MEDIUM)
                "portfolio_allocation_pct": 2.0,  # 2% per position
                "min_order_usdt": 20.0,
                "enable_profit_lock": True,       # Start locking profits
                "ev_multiplier": 1.8,
            }
            
        else:  # LARGE bracket
            sizing = {
                "quote_per_position": 50.0,
                "max_per_symbol": 150.0,
                "max_position_pct": 0.20,         # ← NEW: Max 20% of NAV per position (LARGE)
                "portfolio_allocation_pct": 1.0,  # 1% per position
                "min_order_usdt": 30.0,
                "enable_profit_lock": True,
                "ev_multiplier": 2.0,             # Strict gate
            }
        
        # ===== PHASE 5: CONCENTRATION GATING (NEW) =====
        # Calculate max allowed position size based on NAV
        if nav > 0:
            max_position = nav * sizing["max_position_pct"]
            
            # Calculate remaining headroom (how much we can add to this symbol)
            current_value = float(current_position_value or 0.0)
            headroom = max(0.0, max_position - current_value)
            
            # Cap the quote to not exceed headroom
            original_quote = sizing["quote_per_position"]
            adjusted_quote = min(original_quote, headroom)
            
            sizing["concentration_headroom"] = headroom
            sizing["quote_per_position"] = adjusted_quote
            
            # Log concentration gating if applied
            if adjusted_quote < original_quote:
                logger.warning(
                    "[CapitalGovernor:ConcentrationGate] %s CAPPED: "
                    "max_position=%.2f%% (%.0f USDT), current=%.0f, "
                    "headroom=%.0f → quote adjusted %.0f → %.0f USDT",
                    symbol or "symbol",
                    sizing["max_position_pct"] * 100,
                    max_position,
                    current_value,
                    headroom,
                    original_quote,
                    adjusted_quote
                )
        else:
            sizing["concentration_headroom"] = 0.0
        
        if symbol:
            logger.debug(
                "[CapitalGovernor:Sizing] NAV=$%.2f → %s: "
                "$%.1f per position (max_pct=%.0f%%), EV×%.1f, profit_lock=%s",
                nav,
                symbol,
                sizing["quote_per_position"],
                sizing["max_position_pct"] * 100,
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
        limits = self.get_position_limits(nav)
        return not bool(limits.get("allow_rotation", False))
    
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
        if is_core:
            return True  # Core pairs always allowed

        limits = self.get_position_limits(nav)
        return bool(limits.get("allow_rotation", False))
    
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
