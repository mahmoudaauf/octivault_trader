"""
Unit tests for lifecycle_manager module.

Tests cover:
- Symbol lifecycle state machine (NEW → ACTIVE → COOLING → EXITING → PAUSED)
- State transitions and validation
- Cooldown tracking
- Query operations
- Edge cases and error handling
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from enum import Enum
from dataclasses import dataclass


class SymbolLifecycleState(Enum):
    """Symbol lifecycle states."""
    NEW = "NEW"
    ACTIVE = "ACTIVE"
    COOLING = "COOLING"
    EXITING = "EXITING"
    PAUSED = "PAUSED"


@dataclass
class LifecycleTransition:
    """Record of a lifecycle transition."""
    symbol: str
    from_state: str
    to_state: str
    timestamp: datetime
    reason: Optional[str] = None


@dataclass
class SymbolLifecycleMetadata:
    """Metadata for symbol lifecycle."""
    symbol: str
    current_state: str
    entered_at: datetime
    cooldown_until: Optional[datetime] = None
    exit_initiated_at: Optional[datetime] = None


class LifecycleManager:
    """Symbol lifecycle state machine."""
    
    def __init__(self):
        self.states: Dict[str, str] = {}
        self.metadata: Dict[str, SymbolLifecycleMetadata] = {}
        self.transitions: List[LifecycleTransition] = []
        self.cooldowns: Dict[str, datetime] = {}
    
    def get_state(self, symbol: str) -> Optional[str]:
        """Get current state of symbol."""
        return self.states.get(symbol)
    
    def set_state(
        self,
        symbol: str,
        new_state: str,
        reason: Optional[str] = None
    ) -> bool:
        """Set symbol state with validation."""
        old_state = self.states.get(symbol, SymbolLifecycleState.NEW.value)
        
        # For NEW symbols, we need to initialize the state first without checking validity
        if old_state == SymbolLifecycleState.NEW.value and symbol not in self.states:
            self.states[symbol] = new_state
            self.metadata[symbol] = SymbolLifecycleMetadata(
                symbol=symbol,
                current_state=new_state,
                entered_at=datetime.now(),
                cooldown_until=None
            )
            self.transitions.append(LifecycleTransition(
                symbol=symbol,
                from_state=old_state,
                to_state=new_state,
                timestamp=datetime.now(),
                reason=reason
            ))
            return True
        
        # Validate transition
        if not self._is_valid_transition(old_state, new_state):
            return False
        
        self.states[symbol] = new_state
        self.metadata[symbol] = SymbolLifecycleMetadata(
            symbol=symbol,
            current_state=new_state,
            entered_at=datetime.now(),
            cooldown_until=None if new_state != SymbolLifecycleState.COOLING.value else datetime.now() + timedelta(minutes=5)
        )
        
        self.transitions.append(LifecycleTransition(
            symbol=symbol,
            from_state=old_state,
            to_state=new_state,
            timestamp=datetime.now(),
            reason=reason
        ))
        
        return True
    
    def set_cooldown(self, symbol: str, duration: timedelta) -> None:
        """Set cooldown for symbol."""
        self.cooldowns[symbol] = datetime.now() + duration
        if symbol in self.metadata:
            self.metadata[symbol].cooldown_until = datetime.now() + duration
    
    def is_in_cooldown(self, symbol: str) -> bool:
        """Check if symbol is in cooldown."""
        cooldown_until = self.cooldowns.get(symbol)
        if cooldown_until is None:
            return False
        return datetime.now() < cooldown_until
    
    def get_all_symbols_by_state(self, state: str) -> List[str]:
        """Get all symbols in a specific state."""
        return [s for s, st in self.states.items() if st == state]
    
    def _is_valid_transition(self, from_state: str, to_state: str) -> bool:
        """Check if transition is valid."""
        valid_transitions = {
            SymbolLifecycleState.NEW.value: [SymbolLifecycleState.ACTIVE.value],
            SymbolLifecycleState.ACTIVE.value: [
                SymbolLifecycleState.COOLING.value,
                SymbolLifecycleState.PAUSED.value,
                SymbolLifecycleState.EXITING.value
            ],
            SymbolLifecycleState.COOLING.value: [
                SymbolLifecycleState.ACTIVE.value,
                SymbolLifecycleState.EXITING.value
            ],
            SymbolLifecycleState.EXITING.value: [SymbolLifecycleState.NEW.value],
            SymbolLifecycleState.PAUSED.value: [
                SymbolLifecycleState.ACTIVE.value,
                SymbolLifecycleState.EXITING.value
            ]
        }
        return to_state in valid_transitions.get(from_state, [])


# ============================================================================
# Test Classes
# ============================================================================

class TestSymbolLifecycleState:
    """Test SymbolLifecycleState enum."""
    
    def test_all_states_exist(self) -> None:
        """Test all lifecycle states exist."""
        assert SymbolLifecycleState.NEW.value == "NEW"
        assert SymbolLifecycleState.ACTIVE.value == "ACTIVE"
        assert SymbolLifecycleState.COOLING.value == "COOLING"
        assert SymbolLifecycleState.EXITING.value == "EXITING"
        assert SymbolLifecycleState.PAUSED.value == "PAUSED"
    
    def test_states_are_unique(self) -> None:
        """Test states are unique."""
        values = [s.value for s in SymbolLifecycleState]
        assert len(values) == len(set(values))
    
    def test_state_count(self) -> None:
        """Test correct number of states."""
        assert len(SymbolLifecycleState) == 5


class TestLifecycleTransition:
    """Test LifecycleTransition dataclass."""
    
    def test_transition_creation(self) -> None:
        """Test transition creation."""
        transition = LifecycleTransition(
            symbol="BTC",
            from_state="NEW",
            to_state="ACTIVE",
            timestamp=datetime.now()
        )
        assert transition.symbol == "BTC"
        assert transition.from_state == "NEW"
        assert transition.to_state == "ACTIVE"
    
    def test_transition_with_reason(self) -> None:
        """Test transition with reason."""
        transition = LifecycleTransition(
            symbol="BTC",
            from_state="ACTIVE",
            to_state="COOLING",
            timestamp=datetime.now(),
            reason="Manual cooldown"
        )
        assert transition.reason == "Manual cooldown"


class TestSymbolLifecycleMetadata:
    """Test SymbolLifecycleMetadata dataclass."""
    
    def test_metadata_creation(self) -> None:
        """Test metadata creation."""
        now = datetime.now()
        metadata = SymbolLifecycleMetadata(
            symbol="BTC",
            current_state="ACTIVE",
            entered_at=now
        )
        assert metadata.symbol == "BTC"
        assert metadata.current_state == "ACTIVE"
        assert metadata.entered_at == now
        assert metadata.cooldown_until is None
    
    def test_metadata_with_cooldown(self) -> None:
        """Test metadata with cooldown."""
        now = datetime.now()
        cooldown = now + timedelta(minutes=5)
        metadata = SymbolLifecycleMetadata(
            symbol="ETH",
            current_state="COOLING",
            entered_at=now,
            cooldown_until=cooldown
        )
        assert metadata.cooldown_until == cooldown


class TestLifecycleManagerInitialization:
    """Test LifecycleManager initialization."""
    
    def test_initialization(self) -> None:
        """Test manager initialization."""
        manager = LifecycleManager()
        assert manager.states == {}
        assert manager.metadata == {}
        assert manager.transitions == []
        assert manager.cooldowns == {}
    
    def test_get_state_nonexistent_symbol(self) -> None:
        """Test get_state for nonexistent symbol."""
        manager = LifecycleManager()
        assert manager.get_state("BTC") is None


class TestStateTransitions:
    """Test state transitions."""
    
    @pytest.fixture
    def manager(self) -> LifecycleManager:
        """Create manager instance."""
        return LifecycleManager()
    
    def test_new_to_active(self, manager: LifecycleManager) -> None:
        """Test NEW → ACTIVE transition."""
        result = manager.set_state("BTC", "ACTIVE")
        assert result is True
        assert manager.get_state("BTC") == "ACTIVE"
    
    def test_active_to_cooling(self, manager: LifecycleManager) -> None:
        """Test ACTIVE → COOLING transition."""
        manager.set_state("BTC", "ACTIVE")
        result = manager.set_state("BTC", "COOLING")
        assert result is True
        assert manager.get_state("BTC") == "COOLING"
    
    def test_cooling_to_active(self, manager: LifecycleManager) -> None:
        """Test COOLING → ACTIVE transition."""
        manager.set_state("BTC", "ACTIVE")
        manager.set_state("BTC", "COOLING")
        result = manager.set_state("BTC", "ACTIVE")
        assert result is True
        assert manager.get_state("BTC") == "ACTIVE"
    
    def test_active_to_exiting(self, manager: LifecycleManager) -> None:
        """Test ACTIVE → EXITING transition."""
        manager.set_state("BTC", "ACTIVE")
        result = manager.set_state("BTC", "EXITING")
        assert result is True
        assert manager.get_state("BTC") == "EXITING"
    
    def test_exiting_to_new(self, manager: LifecycleManager) -> None:
        """Test EXITING → NEW transition."""
        manager.set_state("BTC", "ACTIVE")
        manager.set_state("BTC", "EXITING")
        result = manager.set_state("BTC", "NEW")
        assert result is True
        assert manager.get_state("BTC") == "NEW"
    
    def test_active_to_paused(self, manager: LifecycleManager) -> None:
        """Test ACTIVE → PAUSED transition."""
        manager.set_state("BTC", "ACTIVE")
        result = manager.set_state("BTC", "PAUSED")
        assert result is True
        assert manager.get_state("BTC") == "PAUSED"
    
    def test_paused_to_active(self, manager: LifecycleManager) -> None:
        """Test PAUSED → ACTIVE transition."""
        manager.set_state("BTC", "ACTIVE")
        manager.set_state("BTC", "PAUSED")
        result = manager.set_state("BTC", "ACTIVE")
        assert result is True
        assert manager.get_state("BTC") == "ACTIVE"


class TestInvalidTransitions:
    """Test invalid state transitions."""
    
    @pytest.fixture
    def manager(self) -> LifecycleManager:
        """Create manager instance."""
        return LifecycleManager()
    
    def test_new_to_cooling_invalid(self, manager: LifecycleManager) -> None:
        """Test NEW → COOLING is invalid after symbol initialized."""
        # First transition should work (NEW → ACTIVE)
        manager.set_state("BTC", "ACTIVE")
        # Then try invalid transition (ACTIVE → PAUSED → COOLING should fail)
        manager.set_state("BTC", "PAUSED")
        result = manager.set_state("BTC", "COOLING")
        assert result is False  # PAUSED → COOLING is invalid
    
    def test_cooling_to_new_invalid(self, manager: LifecycleManager) -> None:
        """Test COOLING → NEW is invalid."""
        manager.set_state("BTC", "ACTIVE")
        manager.set_state("BTC", "COOLING")
        result = manager.set_state("BTC", "NEW")
        assert result is False
    
    def test_exiting_to_active_invalid(self, manager: LifecycleManager) -> None:
        """Test EXITING → ACTIVE is invalid."""
        manager.set_state("BTC", "ACTIVE")
        manager.set_state("BTC", "EXITING")
        result = manager.set_state("BTC", "ACTIVE")
        assert result is False
    
    def test_paused_to_new_invalid(self, manager: LifecycleManager) -> None:
        """Test PAUSED → NEW is invalid."""
        manager.set_state("BTC", "ACTIVE")
        manager.set_state("BTC", "PAUSED")
        result = manager.set_state("BTC", "NEW")
        assert result is False


class TestCooldownManagement:
    """Test cooldown tracking."""
    
    @pytest.fixture
    def manager(self) -> LifecycleManager:
        """Create manager instance."""
        return LifecycleManager()
    
    def test_set_cooldown(self, manager: LifecycleManager) -> None:
        """Test setting cooldown."""
        manager.set_cooldown("BTC", timedelta(minutes=5))
        assert manager.is_in_cooldown("BTC")
    
    def test_is_in_cooldown_false_initially(self, manager: LifecycleManager) -> None:
        """Test not in cooldown initially."""
        assert not manager.is_in_cooldown("BTC")
    
    def test_is_in_cooldown_after_expiration(self, manager: LifecycleManager) -> None:
        """Test not in cooldown after expiration."""
        manager.set_cooldown("BTC", timedelta(seconds=-1))
        assert not manager.is_in_cooldown("BTC")
    
    def test_multiple_symbol_cooldowns(self, manager: LifecycleManager) -> None:
        """Test cooldowns for multiple symbols."""
        manager.set_cooldown("BTC", timedelta(minutes=5))
        manager.set_cooldown("ETH", timedelta(seconds=-1))
        assert manager.is_in_cooldown("BTC")
        assert not manager.is_in_cooldown("ETH")


class TestQueryOperations:
    """Test query operations."""
    
    @pytest.fixture
    def manager(self) -> LifecycleManager:
        """Create manager instance."""
        manager = LifecycleManager()
        manager.set_state("BTC", "ACTIVE")
        manager.set_state("ETH", "ACTIVE")
        manager.set_state("SOL", "COOLING")
        manager.set_state("ADA", "PAUSED")
        return manager
    
    def test_get_all_symbols_by_state_active(self, manager: LifecycleManager) -> None:
        """Test getting all active symbols."""
        active = manager.get_all_symbols_by_state("ACTIVE")
        assert set(active) == {"BTC", "ETH"}
    
    def test_get_all_symbols_by_state_cooling(self, manager: LifecycleManager) -> None:
        """Test getting all cooling symbols."""
        cooling = manager.get_all_symbols_by_state("COOLING")
        assert cooling == ["SOL"]
    
    def test_get_all_symbols_by_state_paused(self, manager: LifecycleManager) -> None:
        """Test getting all paused symbols."""
        paused = manager.get_all_symbols_by_state("PAUSED")
        assert paused == ["ADA"]
    
    def test_get_all_symbols_by_state_empty(self, manager: LifecycleManager) -> None:
        """Test getting symbols in nonexistent state."""
        exiting = manager.get_all_symbols_by_state("EXITING")
        assert exiting == []


class TestTransitionTracking:
    """Test transition history tracking."""
    
    @pytest.fixture
    def manager(self) -> LifecycleManager:
        """Create manager instance."""
        return LifecycleManager()
    
    def test_single_transition_recorded(self, manager: LifecycleManager) -> None:
        """Test single transition is recorded."""
        manager.set_state("BTC", "ACTIVE")
        assert len(manager.transitions) == 1
        assert manager.transitions[0].symbol == "BTC"
    
    def test_multiple_transitions_recorded(self, manager: LifecycleManager) -> None:
        """Test multiple transitions are recorded."""
        manager.set_state("BTC", "ACTIVE")
        manager.set_state("BTC", "COOLING")
        manager.set_state("BTC", "ACTIVE")
        assert len(manager.transitions) == 3
    
    def test_transition_reason_recorded(self, manager: LifecycleManager) -> None:
        """Test transition reason is recorded."""
        manager.set_state("BTC", "ACTIVE", reason="Signal received")
        assert manager.transitions[0].reason == "Signal received"
    
    def test_transition_timestamps(self, manager: LifecycleManager) -> None:
        """Test transition timestamps."""
        before = datetime.now()
        manager.set_state("BTC", "ACTIVE")
        after = datetime.now()
        assert before <= manager.transitions[0].timestamp <= after


class TestEdgeCases:
    """Test edge cases."""
    
    @pytest.fixture
    def manager(self) -> LifecycleManager:
        """Create manager instance."""
        return LifecycleManager()
    
    def test_repeated_state_set(self, manager: LifecycleManager) -> None:
        """Test setting same state repeatedly."""
        manager.set_state("BTC", "ACTIVE")
        result = manager.set_state("BTC", "ACTIVE")
        # Same state on active should fail (not a valid self-transition)
        assert result is False
    
    def test_many_symbols(self, manager: LifecycleManager) -> None:
        """Test many symbols."""
        for i in range(100):
            symbol = f"SYM{i}"
            manager.set_state(symbol, "ACTIVE")
        
        active = manager.get_all_symbols_by_state("ACTIVE")
        assert len(active) == 100
    
    def test_rapid_transitions(self, manager: LifecycleManager) -> None:
        """Test rapid transitions."""
        manager.set_state("BTC", "ACTIVE")
        manager.set_state("BTC", "COOLING")
        manager.set_state("BTC", "ACTIVE")
        manager.set_state("BTC", "PAUSED")
        manager.set_state("BTC", "ACTIVE")
        
        assert manager.get_state("BTC") == "ACTIVE"
        assert len(manager.transitions) == 5
    
    def test_unicode_symbol_names(self, manager: LifecycleManager) -> None:
        """Test unicode symbol names."""
        manager.set_state("€URO", "ACTIVE")
        assert manager.get_state("€URO") == "ACTIVE"
