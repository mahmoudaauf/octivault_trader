"""
Phase 1: Safe Upgrade - Symbol Rotation Manager
Implements soft bootstrap lock and symbol rotation eligibility checking.

This module provides:
1. Soft bootstrap lock (duration-based, can be overridden)
2. Replacement multiplier checking
3. Universe size enforcement
4. Integration with symbol screener
"""

import time
import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


class SymbolRotationManager:
    """
    Manages soft bootstrap lock and symbol rotation eligibility.
    
    Replaces hard bootstrap lock with duration-based soft lock that:
    - Prevents rotation for 1 hour after a trade
    - Can be disabled via configuration
    - Allows rotation if replacement candidate exceeds threshold
    """
    
    def __init__(self, config):
        """
        Initialize rotation manager with configuration.
        
        Args:
            config: Configuration object with settings:
                - BOOTSTRAP_SOFT_LOCK_ENABLED: Enable soft lock (default: True)
                - BOOTSTRAP_SOFT_LOCK_DURATION_SEC: Lock duration in seconds (default: 3600 = 1 hour)
                - SYMBOL_REPLACEMENT_MULTIPLIER: Score multiplier for rotation (default: 1.10)
                - MAX_ACTIVE_SYMBOLS: Maximum active symbols (default: 5)
                - MIN_ACTIVE_SYMBOLS: Minimum active symbols (default: 3)
        """
        self.soft_lock_enabled = getattr(config, 'BOOTSTRAP_SOFT_LOCK_ENABLED', True)
        self.soft_lock_duration = getattr(config, 'BOOTSTRAP_SOFT_LOCK_DURATION_SEC', 3600)
        self.replacement_multiplier = getattr(config, 'SYMBOL_REPLACEMENT_MULTIPLIER', 1.10)
        self.max_active_symbols = getattr(config, 'MAX_ACTIVE_SYMBOLS', 5)
        self.min_active_symbols = getattr(config, 'MIN_ACTIVE_SYMBOLS', 3)
        
        # Timestamp of last rotation (for soft lock)
        self.last_rotation_ts = 0.0
        
        # Track active symbols for universe enforcement
        self.active_symbols: List[str] = []
        
        logger.info(
            "[SymbolRotation] Initialized: soft_lock=%s duration=%d multiplier=%.2f "
            "universe=[%d-%d]",
            self.soft_lock_enabled,
            self.soft_lock_duration,
            self.replacement_multiplier,
            self.min_active_symbols,
            self.max_active_symbols
        )
    
    def is_locked(self) -> bool:
        """
        Check if bootstrap soft lock is still active.
        
        Returns:
            True if soft lock is engaged and duration hasn't elapsed
        """
        if not self.soft_lock_enabled:
            return False  # Soft lock disabled, allow immediate rotation
        
        elapsed = time.time() - self.last_rotation_ts
        is_locked = elapsed < self.soft_lock_duration
        
        if is_locked:
            remaining = self.soft_lock_duration - elapsed
            logger.debug(
                "[SymbolRotation:SoftLock] Locked for %.1f more seconds",
                remaining
            )
        
        return is_locked
    
    def lock(self):
        """
        Activate soft lock after a trade execution.
        Called after successful trade to prevent immediate rotation.
        """
        self.last_rotation_ts = time.time()
        logger.info(
            "[SymbolRotation:SoftLock] Engaged: next rotation allowed in %.0f seconds",
            self.soft_lock_duration
        )
    
    def can_rotate_to_score(self, current_score: float, candidate_score: float) -> bool:
        """
        Check if candidate score exceeds replacement threshold.
        
        Multiplier logic:
        - If candidate_score > (current_score * multiplier), rotation is allowed
        - Default multiplier: 1.10 (10% improvement needed)
        - Prevents frivolous rotations
        
        Args:
            current_score: Current symbol's performance score
            candidate_score: Candidate symbol's score
        
        Returns:
            True if candidate is sufficiently better than current
        """
        threshold = current_score * self.replacement_multiplier
        can_rotate = candidate_score > threshold
        
        if can_rotate:
            improvement = ((candidate_score - current_score) / current_score * 100) if current_score > 0 else 0
            logger.info(
                "[SymbolRotation:Multiplier] ✅ Can rotate: %.2f > %.2f threshold "
                "(improvement: %.1f%%, multiplier: %.2f)",
                candidate_score,
                threshold,
                improvement,
                self.replacement_multiplier
            )
        else:
            improvement = ((candidate_score - current_score) / current_score * 100) if current_score > 0 else 0
            logger.debug(
                "[SymbolRotation:Multiplier] ❌ Cannot rotate: %.2f <= %.2f threshold "
                "(improvement: %.1f%%, need: %.2f)",
                candidate_score,
                threshold,
                improvement,
                self.replacement_multiplier
            )
        
        return can_rotate
    
    def can_rotate_symbol(self, current_symbol: str, candidate_symbol: str, 
                         current_score: float, candidate_score: float) -> bool:
        """
        Check if rotation from current to candidate is allowed.
        
        Checks both soft lock and replacement multiplier thresholds.
        
        Args:
            current_symbol: Symbol to rotate away from
            candidate_symbol: Symbol to rotate to
            current_score: Current symbol's score
            candidate_score: Candidate symbol's score
        
        Returns:
            True if rotation is allowed
        """
        # Check soft lock first
        if self.is_locked():
            logger.info(
                "[SymbolRotation:Eligibility] Cannot rotate %s -> %s: soft lock engaged",
                current_symbol,
                candidate_symbol
            )
            return False
        
        # Check replacement multiplier
        if not self.can_rotate_to_score(current_score, candidate_score):
            logger.info(
                "[SymbolRotation:Eligibility] Cannot rotate %s -> %s: score improvement "
                "insufficient (%.2f -> %.2f, need %.2f)",
                current_symbol,
                candidate_symbol,
                current_score,
                candidate_score,
                current_score * self.replacement_multiplier
            )
            return False
        
        logger.info(
            "[SymbolRotation:Eligibility] ✅ Can rotate %s -> %s "
            "(score: %.2f -> %.2f, multiplier: %.2f)",
            current_symbol,
            candidate_symbol,
            current_score,
            candidate_score,
            self.replacement_multiplier
        )
        return True
    
    def update_active_symbols(self, symbols: List[str]):
        """
        Update list of active trading symbols.
        
        Args:
            symbols: List of currently active symbols
        """
        self.active_symbols = symbols
        logger.debug(
            "[SymbolRotation:Universe] Updated active symbols: %d/%d [%s]",
            len(symbols),
            self.max_active_symbols,
            ", ".join(symbols) if symbols else "empty"
        )
    
    def enforce_universe_size(self, current_active: List[str], 
                             all_candidates: List[str]) -> Dict[str, Any]:
        """
        Enforce min/max active symbols constraints.
        
        Returns dict with:
        - action: 'none' | 'add' | 'remove'
        - affected_symbols: List of symbols to add/remove
        - reason: Explanation
        
        Args:
            current_active: List of currently active symbols
            all_candidates: List of all candidate symbols from screener
        
        Returns:
            Dict with enforcement action details
        """
        current_count = len(current_active)
        
        # Too many symbols?
        if current_count > self.max_active_symbols:
            excess = current_count - self.max_active_symbols
            logger.warning(
                "[SymbolRotation:Universe] Universe oversized: %d > %d (max). "
                "Need to remove %d symbols",
                current_count,
                self.max_active_symbols,
                excess
            )
            return {
                'action': 'remove',
                'count': excess,
                'affected_symbols': current_active[-excess:],  # Remove worst performers
                'reason': f'Universe oversized: {current_count} > {self.max_active_symbols}'
            }
        
        # Too few symbols?
        elif current_count < self.min_active_symbols:
            deficit = self.min_active_symbols - current_count
            candidates = [s for s in all_candidates if s not in current_active]
            
            if len(candidates) >= deficit:
                logger.warning(
                    "[SymbolRotation:Universe] Universe undersized: %d < %d (min). "
                    "Need to add %d symbols from %d candidates",
                    current_count,
                    self.min_active_symbols,
                    deficit,
                    len(candidates)
                )
                return {
                    'action': 'add',
                    'count': deficit,
                    'affected_symbols': candidates[:deficit],  # Add top candidates
                    'reason': f'Universe undersized: {current_count} < {self.min_active_symbols}'
                }
            else:
                logger.warning(
                    "[SymbolRotation:Universe] Universe undersized but insufficient candidates: "
                    "%d < %d (min), only %d candidates available",
                    current_count,
                    self.min_active_symbols,
                    len(candidates)
                )
                return {
                    'action': 'none',
                    'affected_symbols': [],
                    'reason': f'Universe undersized but insufficient candidates ({len(candidates)})'
                }
        
        # Universe is correctly sized
        logger.debug(
            "[SymbolRotation:Universe] Universe correctly sized: %d/%d symbols",
            current_count,
            self.max_active_symbols
        )
        return {
            'action': 'none',
            'affected_symbols': [],
            'reason': 'Universe correctly sized'
        }
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current rotation manager status.
        
        Returns:
            Dict with current lock state, active symbols, configuration
        """
        is_locked = self.is_locked()
        elapsed = time.time() - self.last_rotation_ts
        remaining = max(0, self.soft_lock_duration - elapsed)
        
        return {
            'soft_lock_enabled': self.soft_lock_enabled,
            'is_locked': is_locked,
            'lock_remaining_sec': remaining if is_locked else 0.0,
            'replacement_multiplier': self.replacement_multiplier,
            'active_symbols': self.active_symbols,
            'active_count': len(self.active_symbols),
            'max_active': self.max_active_symbols,
            'min_active': self.min_active_symbols,
            'last_rotation_ts': self.last_rotation_ts,
            'elapsed_since_rotation': elapsed
        }
