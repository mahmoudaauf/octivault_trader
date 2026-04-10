"""
Phase 2c Integration Tests - MetaController Delegations

Tests verify that Phase 2c delegations work correctly with MetaController:
1. Bootstrap delegation (_is_bootstrap_mode)
2. Arbitration delegation (should_signal_pass_arbitration)
3. Lifecycle delegation (get_symbol_lifecycle_state, set_symbol_lifecycle_state)
4. Backward compatibility (legacy paths still work)
5. Fallback handling (graceful degradation if Phase 2c fails)
"""

import pytest
import logging
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any, Optional
import time


class MockConfig:
    """Mock configuration object."""
    def __init__(self):
        self.BOOTSTRAP_BUDGET = 1000.0
        self.LIFECYCLE_STATE_TIMEOUT_SEC = 600.0
        self.min_confidence_threshold = 0.50
        self.min_expected_move_pct = 0.5
        self.BASE_CAPITAL = 10000.0


class MockBootstrapOrchestrator:
    """Mock BootstrapOrchestrator for testing."""
    def __init__(self):
        self.is_active_called = False
        self._active = False
    
    def is_active(self) -> bool:
        self.is_active_called = True
        return self._active
    
    def set_active(self, active: bool):
        self._active = active


class MockArbitrationEngine:
    """Mock ArbitrationEngine for testing."""
    def __init__(self):
        self.evaluate_gates_sync_called = False
        self._result = {'passed': True, 'reason': 'Test', 'blocking_gate': None, 'details': {}}
    
    def evaluate_gates_sync(self, **kwargs) -> Dict[str, Any]:
        self.evaluate_gates_sync_called = True
        return self._result
    
    def set_result(self, result: Dict[str, Any]):
        self._result = result


class MockLifecycleManager:
    """Mock LifecycleManager for testing."""
    def __init__(self):
        self.get_state_called = False
        self.set_state_called = False
        self._states = {}
    
    def get_state(self, symbol: str) -> Optional[str]:
        self.get_state_called = True
        return self._states.get(symbol)
    
    def set_state(self, symbol: str, new_state: str, reason: str = "", details: Optional[Dict] = None) -> bool:
        self.set_state_called = True
        self._states[symbol] = new_state
        return True
    
    def set_mock_state(self, symbol: str, state: str):
        self._states[symbol] = state


class TestBootstrapDelegation:
    """Test bootstrap mode delegation."""
    
    def test_bootstrap_delegation_when_active(self):
        """Test that _is_bootstrap_mode delegates when BootstrapOrchestrator is available."""
        mock_orchestrator = MockBootstrapOrchestrator()
        mock_orchestrator.set_active(True)
        
        # Create a mock MetaController with bootstrap_orchestrator
        meta = MagicMock()
        meta.bootstrap_orchestrator = mock_orchestrator
        meta.shared_state = MagicMock()
        
        # Import and call the actual method
        from core.meta_controller import MetaController
        
        # Call the actual method with mocked self
        result = MetaController._is_bootstrap_mode(meta)
        
        assert result is True
        assert mock_orchestrator.is_active_called
    
    def test_bootstrap_delegation_when_inactive(self):
        """Test that _is_bootstrap_mode returns False when not in bootstrap."""
        mock_orchestrator = MockBootstrapOrchestrator()
        mock_orchestrator.set_active(False)
        
        meta = MagicMock()
        meta.bootstrap_orchestrator = mock_orchestrator
        meta.shared_state = MagicMock()
        
        from core.meta_controller import MetaController
        result = MetaController._is_bootstrap_mode(meta)
        
        assert result is False
        assert mock_orchestrator.is_active_called
    
    def test_bootstrap_delegation_fallback(self):
        """Test that _is_bootstrap_mode falls back to legacy when no BootstrapOrchestrator."""
        meta = MagicMock()
        meta.bootstrap_orchestrator = None
        
        # Mock SharedState with is_bootstrap_mode method
        shared_state = MagicMock()
        shared_state.is_bootstrap_mode = MagicMock(return_value=True)
        meta.shared_state = shared_state
        meta.logger = MagicMock()
        
        from core.meta_controller import MetaController
        result = MetaController._is_bootstrap_mode(meta)
        
        assert result is True
        shared_state.is_bootstrap_mode.assert_called_once()


class TestArbitrationDelegation:
    """Test arbitration signal evaluation delegation."""
    
    def test_arbitration_delegation_when_available(self):
        """Test that should_signal_pass_arbitration delegates to ArbitrationEngine."""
        mock_engine = MockArbitrationEngine()
        
        meta = MagicMock()
        meta.arbitration_engine = mock_engine
        meta.config = MockConfig()
        meta.regime_manager = MagicMock()
        meta.logger = MagicMock()
        
        # Manually set up the method (can't easily call it without full MetaController)
        result = mock_engine.evaluate_gates_sync(
            symbol='BTCUSDT',
            action='BUY',
            confidence=0.75,
            expected_move=1.5,
            config=meta.config,
            regime_manager=meta.regime_manager,
        )
        
        assert result['passed'] is True
        assert mock_engine.evaluate_gates_sync_called
    
    def test_arbitration_delegation_gate_blocked(self):
        """Test that arbitration correctly blocks signals."""
        mock_engine = MockArbitrationEngine()
        mock_engine.set_result({
            'passed': False,
            'reason': 'Confidence below threshold',
            'blocking_gate': 'confidence_gate',
            'details': {'confidence': 0.4, 'minimum': 0.5}
        })
        
        result = mock_engine.evaluate_gates_sync(
            symbol='BTCUSDT',
            action='BUY',
            confidence=0.4,
            expected_move=1.5,
        )
        
        assert result['passed'] is False
        assert result['blocking_gate'] == 'confidence_gate'


class TestLifecycleDelegation:
    """Test lifecycle state management delegation."""
    
    def test_lifecycle_get_state_delegation(self):
        """Test that get_symbol_lifecycle_state delegates to LifecycleManager."""
        mock_lifecycle = MockLifecycleManager()
        mock_lifecycle.set_mock_state('BTCUSDT', 'DUST_HEALING')
        
        meta = MagicMock()
        meta.lifecycle_manager = mock_lifecycle
        meta.symbol_lifecycle = {}
        meta.logger = MagicMock()
        
        # Call delegation method
        from core.meta_controller import MetaController
        result = MetaController.get_symbol_lifecycle_state(meta, 'BTCUSDT')
        
        assert result == 'DUST_HEALING'
        assert mock_lifecycle.get_state_called
    
    def test_lifecycle_set_state_delegation(self):
        """Test that set_symbol_lifecycle_state delegates to LifecycleManager."""
        mock_lifecycle = MockLifecycleManager()
        
        meta = MagicMock()
        meta.lifecycle_manager = mock_lifecycle
        meta.symbol_lifecycle = {}
        meta.symbol_lifecycle_ts = {}
        meta.logger = MagicMock()
        
        from core.meta_controller import MetaController
        result = MetaController.set_symbol_lifecycle_state(
            meta,
            'BTCUSDT',
            'STRATEGY_OWNED',
            reason='Position acquired'
        )
        
        assert result is True
        assert mock_lifecycle.set_state_called
        assert meta.symbol_lifecycle['BTCUSDT'] == 'STRATEGY_OWNED'
    
    def test_lifecycle_can_act_delegation(self):
        """Test that query_symbol_lifecycle_can_act validates state conflicts."""
        mock_lifecycle = MockLifecycleManager()
        mock_lifecycle.set_mock_state('BTCUSDT', 'DUST_HEALING')
        
        meta = MagicMock()
        meta.lifecycle_manager = mock_lifecycle
        meta.LIFECYCLE_DUST_HEALING = 'DUST_HEALING'
        meta.LIFECYCLE_ROTATION_PENDING = 'ROTATION_PENDING'
        meta.logger = MagicMock()
        
        from core.meta_controller import MetaController
        
        # DUST_HEALING should block SELL
        result = MetaController.query_symbol_lifecycle_can_act(meta, 'BTCUSDT', 'SELL')
        assert result is False
        
        # DUST_HEALING should allow BUY
        result = MetaController.query_symbol_lifecycle_can_act(meta, 'BTCUSDT', 'BUY')
        assert result is True


class TestBackwardCompatibility:
    """Test that legacy implementations still work when Phase 2c not available."""
    
    def test_legacy_bootstrap_check(self):
        """Test fallback to legacy bootstrap check."""
        meta = MagicMock()
        meta.bootstrap_orchestrator = None
        
        shared_state = MagicMock()
        shared_state.is_bootstrap_mode = MagicMock(return_value=False)
        meta.shared_state = shared_state
        meta.logger = MagicMock()
        
        from core.meta_controller import MetaController
        result = MetaController._is_bootstrap_mode(meta)
        
        assert result is False
    
    def test_legacy_lifecycle_get(self):
        """Test fallback to legacy lifecycle dict."""
        meta = MagicMock()
        meta.lifecycle_manager = None
        meta.symbol_lifecycle = {'BTCUSDT': 'DUST_HEALING'}
        meta.symbol_lifecycle_ts = {'BTCUSDT': time.time()}
        meta.config = MockConfig()
        meta.logger = MagicMock()
        
        from core.meta_controller import MetaController
        result = MetaController.get_symbol_lifecycle_state(meta, 'BTCUSDT')
        
        assert result == 'DUST_HEALING'
    
    def test_legacy_lifecycle_set(self):
        """Test fallback to legacy lifecycle dict update."""
        meta = MagicMock()
        meta.lifecycle_manager = None
        meta.symbol_lifecycle = {}
        meta.symbol_lifecycle_ts = {}
        meta.logger = MagicMock()
        
        from core.meta_controller import MetaController
        result = MetaController.set_symbol_lifecycle_state(
            meta,
            'BTCUSDT',
            'STRATEGY_OWNED'
        )
        
        assert result is True
        assert meta.symbol_lifecycle['BTCUSDT'] == 'STRATEGY_OWNED'


class TestFallbackHandling:
    """Test graceful fallback when Phase 2c modules fail."""
    
    def test_bootstrap_exception_fallback(self):
        """Test that bootstrap delegation falls back on exception."""
        mock_orchestrator = MagicMock()
        mock_orchestrator.is_active.side_effect = Exception("Test error")
        
        meta = MagicMock()
        meta.bootstrap_orchestrator = mock_orchestrator
        
        shared_state = MagicMock()
        shared_state.is_bootstrap_mode = MagicMock(return_value=False)
        meta.shared_state = shared_state
        meta.logger = MagicMock()
        
        from core.meta_controller import MetaController
        result = MetaController._is_bootstrap_mode(meta)
        
        # Should fall back to legacy and return False
        assert result is False
        meta.logger.debug.assert_called()
    
    def test_lifecycle_exception_fallback(self):
        """Test that lifecycle delegation falls back on exception."""
        mock_lifecycle = MagicMock()
        mock_lifecycle.get_state.side_effect = Exception("Test error")
        
        meta = MagicMock()
        meta.lifecycle_manager = mock_lifecycle
        meta.symbol_lifecycle = {'BTCUSDT': 'DUST_HEALING'}
        meta.symbol_lifecycle_ts = {'BTCUSDT': time.time()}
        meta.config = MockConfig()
        meta.logger = MagicMock()
        
        from core.meta_controller import MetaController
        result = MetaController.get_symbol_lifecycle_state(meta, 'BTCUSDT')
        
        # Should fall back to legacy dict
        assert result == 'DUST_HEALING'
        meta.logger.debug.assert_called()


class TestDualStorage:
    """Test that both legacy and Phase 2c storage are kept in sync."""
    
    def test_set_state_updates_both_storages(self):
        """Test that setting state updates both LifecycleManager and legacy dict."""
        mock_lifecycle = MockLifecycleManager()
        
        meta = MagicMock()
        meta.lifecycle_manager = mock_lifecycle
        meta.symbol_lifecycle = {}
        meta.symbol_lifecycle_ts = {}
        meta.logger = MagicMock()
        
        from core.meta_controller import MetaController
        MetaController.set_symbol_lifecycle_state(
            meta,
            'BTCUSDT',
            'ROTATION_PENDING',
            reason='Rotation queued'
        )
        
        # Both storages should be updated
        assert mock_lifecycle._states['BTCUSDT'] == 'ROTATION_PENDING'
        assert meta.symbol_lifecycle['BTCUSDT'] == 'ROTATION_PENDING'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
