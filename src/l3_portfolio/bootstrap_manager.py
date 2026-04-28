# -*- coding: utf-8 -*-
"""
BootstrapManager - Extracted from MetaController

Responsibility:
- Bootstrap mode lifecycle and logic
- Dust bypass management
- Bootstrap override handling
- Transitions in/out of bootstrap mode

This module extracts bootstrap-related logic from MetaController
to reduce coupling and improve testability.
"""

import logging
import time
from typing import Dict, Optional, Any
from collections import defaultdict
from enum import Enum


class DustState(Enum):
    """Dust accumulation state machine."""
    EMPTY = "empty"
    DUST_ACCUMULATING = "dust_accumulating"
    DUST_MATURED = "dust_matured"
    TRADABLE = "tradable"


class BootstrapDustBypassManager:
    """Manages bootstrap-mode dust bypass allowances."""
    
    def __init__(self):
        self._bootstrap_dust_bypass_symbols = set()
        self._bootstrap_dust_bypass_ts = {}
        self._bootstrap_dust_bypass_budget = 1  # One bypass per cycle
        self._bootstrap_cycle_start = None
    
    def reset_cycle(self):
        """Reset bootstrap cycle and bypass budget."""
        self._bootstrap_dust_bypass_symbols.clear()
        self._bootstrap_dust_bypass_budget = 1
        self._bootstrap_cycle_start = time.time()
    
    def can_use(self, symbol: str) -> bool:
        """Check if this symbol can bypass dust checks this bootstrap cycle."""
        return symbol not in self._bootstrap_dust_bypass_symbols
    
    def mark_used(self, symbol: str):
        """Mark symbol as used for bootstrap dust bypass."""
        if symbol not in self._bootstrap_dust_bypass_symbols:
            self._bootstrap_dust_bypass_symbols.add(symbol)
            self._bootstrap_dust_bypass_ts[symbol] = time.time()
    
    def get_status(self, symbol: str) -> Dict[str, Any]:
        """Get bootstrap dust bypass status for symbol."""
        return {
            'has_bypass': symbol in self._bootstrap_dust_bypass_symbols,
            'used_at': self._bootstrap_dust_bypass_ts.get(symbol),
            'budget_remaining': max(0, self._bootstrap_dust_bypass_budget - len(self._bootstrap_dust_bypass_symbols)),
        }
    
    def reset_symbol(self, symbol: str):
        """Reset bypass status for symbol (for manual override)."""
        self._bootstrap_dust_bypass_symbols.discard(symbol)
        self._bootstrap_dust_bypass_ts.pop(symbol, None)


class BootstrapOrchestrator:
    """Orchestrates bootstrap mode operations."""
    
    def __init__(self, shared_state, meta_controller):
        """Initialize bootstrap orchestrator."""
        self.shared_state = shared_state
        self.meta_controller = meta_controller
        self.bypass_manager = BootstrapDustBypassManager()
        self.logger = logging.getLogger("BootstrapOrchestrator")
        
        # Track bootstrap mode state
        self._bootstrap_mode_active = False
        self._bootstrap_start_time = None
        self._bootstrap_nav_at_start = None
    
    def is_active(self) -> bool:
        """Check if bootstrap mode is currently active."""
        return self._bootstrap_mode_active
    
    def enter_bootstrap_mode(self):
        """Enter bootstrap mode."""
        if not self._bootstrap_mode_active:
            self._bootstrap_mode_active = True
            self._bootstrap_start_time = time.time()
            self._bootstrap_nav_at_start = self.shared_state.nav
            
            self.logger.info(
                f"🟢 Entered bootstrap mode "
                f"(NAV: {self._bootstrap_nav_at_start:.2f})"
            )
    
    def exit_bootstrap_mode(self):
        """Exit bootstrap mode."""
        if self._bootstrap_mode_active:
            elapsed = time.time() - self._bootstrap_start_time
            nav_gain = self.shared_state.nav - self._bootstrap_nav_at_start
            
            self.logger.info(
                f"🔴 Exiting bootstrap mode "
                f"(Duration: {elapsed:.0f}s, NAV gain: {nav_gain:.2f})"
            )
            
            self._bootstrap_mode_active = False
            self.bypass_manager.reset_cycle()
    
    async def apply_bootstrap_logic(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Apply bootstrap mode logic to signal."""
        symbol = signal.get('symbol')
        
        # In bootstrap mode, apply special rules
        if self.is_active():
            # Allow dust bypass for one symbol per cycle
            can_bypass_dust = self.bypass_manager.can_use(symbol)
            
            if can_bypass_dust:
                self.logger.debug(
                    f"Bootstrap bypass enabled for {symbol}"
                )
                signal['bootstrap_bypass_applied'] = True
            
            return await self._execute_bootstrap_signal(signal)
        
        return {'status': 'bootstrap_inactive', 'signal_ignored': True}
    
    async def _execute_bootstrap_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Execute signal with bootstrap mode active."""
        # Implementation would call back to MetaController for execution
        # This is a placeholder
        return {
            'status': 'executed_in_bootstrap',
            'symbol': signal.get('symbol'),
            'bootstrap_mode_active': True,
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current bootstrap mode status."""
        return {
            'active': self._bootstrap_mode_active,
            'started_at': self._bootstrap_start_time,
            'duration_seconds': (
                time.time() - self._bootstrap_start_time
                if self._bootstrap_mode_active else None
            ),
            'initial_nav': self._bootstrap_nav_at_start,
            'current_nav': self.shared_state.nav,
            'bypass_manager_status': {
                'symbols_with_bypass': list(
                    self.bypass_manager._bootstrap_dust_bypass_symbols
                ),
                'budget_remaining': max(
                    0,
                    self.bypass_manager._bootstrap_dust_bypass_budget - len(
                        self.bypass_manager._bootstrap_dust_bypass_symbols
                    ),
                ),
            },
        }
