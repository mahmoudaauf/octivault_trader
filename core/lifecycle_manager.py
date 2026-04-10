# -*- coding: utf-8 -*-
"""
LifecycleManager - Extracted from MetaController

Responsibility:
- Symbol lifecycle state tracking (new, active, cooling, exiting)
- State transitions with validation
- Cooldown management
- Symbol lifecycle metadata

This module extracts lifecycle management from MetaController
to isolate state machine logic for easier testing and maintenance.
"""

import logging
from enum import Enum
from typing import Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone
import time


class SymbolLifecycleState(Enum):
    """Symbol lifecycle state machine states."""
    NEW = "new"  # Just discovered
    ACTIVE = "active"  # Currently trading
    COOLING = "cooling"  # In cooldown after exit
    EXITING = "exiting"  # Actively exiting positions
    PAUSED = "paused"  # Manually paused


@dataclass
class LifecycleTransition:
    """Record of a state transition."""
    from_state: str
    to_state: str
    timestamp: datetime
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SymbolLifecycleMetadata:
    """Metadata for a symbol's lifecycle."""
    symbol: str
    current_state: str
    state_since: datetime
    last_transition: Optional[LifecycleTransition] = None
    cooldown_until: Optional[datetime] = None
    transitions_count: int = 0
    last_price: float = 0.0
    last_action: Optional[str] = None


class LifecycleManager:
    """Manages symbol lifecycle state machine."""
    
    def __init__(self):
        """Initialize lifecycle manager."""
        self.logger = logging.getLogger("LifecycleManager")
        self._symbol_states: Dict[str, SymbolLifecycleMetadata] = {}
        self._transition_history: Dict[str, list] = {}
    
    def get_state(self, symbol: str) -> Optional[str]:
        """Get current lifecycle state for symbol."""
        metadata = self._symbol_states.get(symbol)
        return metadata.current_state if metadata else None
    
    def set_state(
        self,
        symbol: str,
        new_state: str,
        reason: str = "",
        details: Optional[Dict[str, Any]] = None,
        force: bool = False,
    ) -> bool:
        """
        Set symbol to new lifecycle state.
        
        Args:
            symbol: Symbol to transition
            new_state: Target state (from SymbolLifecycleState)
            reason: Reason for transition
            details: Additional metadata
            force: Force transition even if invalid (for recovery)
        
        Returns:
            True if successful, False otherwise
        """
        current_metadata = self._symbol_states.get(symbol)
        
        if not current_metadata:
            # First time seeing this symbol
            current_metadata = SymbolLifecycleMetadata(
                symbol=symbol,
                current_state=SymbolLifecycleState.NEW.value,
                state_since=datetime.now(timezone.utc),
            )
            self._symbol_states[symbol] = current_metadata
            self._transition_history[symbol] = []
        
        from_state = current_metadata.current_state
        
        # Validate transition
        if not force and not self._is_valid_transition(from_state, new_state):
            self.logger.warning(
                f"Invalid transition for {symbol}: "
                f"{from_state} -> {new_state}"
            )
            return False
        
        # Create transition record
        transition = LifecycleTransition(
            from_state=from_state,
            to_state=new_state,
            timestamp=datetime.now(timezone.utc),
            reason=reason,
            details=details or {},
        )
        
        # Update metadata
        current_metadata.current_state = new_state
        current_metadata.state_since = datetime.now(timezone.utc)
        current_metadata.last_transition = transition
        current_metadata.transitions_count += 1
        
        # Record transition
        self._transition_history[symbol].append(transition)
        
        self.logger.info(
            f"Lifecycle transition: {symbol} "
            f"{from_state} -> {new_state} ({reason})"
        )
        
        return True
    
    def set_cooldown(
        self,
        symbol: str,
        cooldown_seconds: float,
    ) -> bool:
        """Set cooldown for symbol (blocks re-entry)."""
        metadata = self._symbol_states.get(symbol)
        if not metadata:
            return False
        
        cooldown_until = datetime.now(timezone.utc)
        cooldown_until = cooldown_until.replace(
            second=int(cooldown_until.second + cooldown_seconds)
        )
        
        metadata.cooldown_until = cooldown_until
        
        # Also set state to COOLING
        self.set_state(
            symbol,
            SymbolLifecycleState.COOLING.value,
            reason=f"Cooldown {cooldown_seconds}s",
        )
        
        self.logger.debug(
            f"Cooldown set for {symbol}: {cooldown_seconds}s"
        )
        
        return True
    
    def is_in_cooldown(self, symbol: str) -> bool:
        """Check if symbol is currently in cooldown."""
        metadata = self._symbol_states.get(symbol)
        if not metadata or not metadata.cooldown_until:
            return False
        
        return datetime.now(timezone.utc) < metadata.cooldown_until
    
    def _is_valid_transition(self, from_state: str, to_state: str) -> bool:
        """Check if state transition is valid."""
        # Valid transitions
        valid_transitions = {
            SymbolLifecycleState.NEW.value: [
                SymbolLifecycleState.ACTIVE.value,
                SymbolLifecycleState.PAUSED.value,
            ],
            SymbolLifecycleState.ACTIVE.value: [
                SymbolLifecycleState.EXITING.value,
                SymbolLifecycleState.COOLING.value,
                SymbolLifecycleState.PAUSED.value,
            ],
            SymbolLifecycleState.EXITING.value: [
                SymbolLifecycleState.COOLING.value,
                SymbolLifecycleState.PAUSED.value,
            ],
            SymbolLifecycleState.COOLING.value: [
                SymbolLifecycleState.ACTIVE.value,
                SymbolLifecycleState.PAUSED.value,
                SymbolLifecycleState.NEW.value,
            ],
            SymbolLifecycleState.PAUSED.value: [
                SymbolLifecycleState.ACTIVE.value,
                SymbolLifecycleState.COOLING.value,
                SymbolLifecycleState.NEW.value,
            ],
        }
        
        return to_state in valid_transitions.get(from_state, [])
    
    def get_metadata(self, symbol: str) -> Optional[SymbolLifecycleMetadata]:
        """Get full lifecycle metadata for symbol."""
        return self._symbol_states.get(symbol)
    
    def get_all_symbols_by_state(self, state: str) -> list:
        """Get all symbols currently in specified state."""
        return [
            symbol
            for symbol, metadata in self._symbol_states.items()
            if metadata.current_state == state
        ]
    
    def get_transition_history(
        self,
        symbol: str,
        limit: Optional[int] = None,
    ) -> list:
        """Get transition history for symbol."""
        history = self._transition_history.get(symbol, [])
        
        if limit:
            history = history[-limit:]
        
        return history
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all symbols under management."""
        state_counts = {}
        for metadata in self._symbol_states.values():
            state = metadata.current_state
            state_counts[state] = state_counts.get(state, 0) + 1
        
        return {
            'total_symbols': len(self._symbol_states),
            'states': state_counts,
            'symbols_in_cooldown': len([
                s for s, m in self._symbol_states.items()
                if m.cooldown_until and datetime.now(timezone.utc) < m.cooldown_until
            ]),
            'total_transitions': sum(m.transitions_count for m in self._symbol_states.values()),
        }
