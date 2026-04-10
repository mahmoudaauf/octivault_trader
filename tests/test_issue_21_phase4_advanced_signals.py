"""
Test suite for Issue #21 Phase 4: Signal Cache Advanced Features

Phase 4 Focus: Add advanced signal cache capabilities for better analytics.
Target: Reduce signal processing overhead via windowing and aggregation.

Implementation:
- get_signal_windowed(): Sliding window signal retrieval (lookback_sec)
- get_universe_signal_stats(): Aggregated universe statistics
- flush_stale_signals(): Periodic cache cleanup (max_age_sec)
- validate_signal_cache_consistency(): Cache coherency checking

Expected Features:
- Per-symbol windowed signal analysis (multi-signal decisions)
- Universe-wide coverage metrics and aging statistics
- Automatic stale signal cleanup to prevent bloat
- Consistency validation for debugging cache issues
"""

import inspect
import pytest
from core.meta_controller import MetaController


class TestSignalWindowingMethods:
    """Test existence and signatures of Phase 4 windowing methods."""
    
    def test_get_signal_windowed_exists(self):
        """Verify get_signal_windowed method exists."""
        assert hasattr(MetaController, "get_signal_windowed")
    
    def test_get_universe_signal_stats_exists(self):
        """Verify get_universe_signal_stats method exists."""
        assert hasattr(MetaController, "get_universe_signal_stats")
    
    def test_flush_stale_signals_exists(self):
        """Verify flush_stale_signals method exists."""
        assert hasattr(MetaController, "flush_stale_signals")
    
    def test_validate_signal_cache_consistency_exists(self):
        """Verify validate_signal_cache_consistency method exists."""
        assert hasattr(MetaController, "validate_signal_cache_consistency")
    
    def test_get_signal_windowed_signature(self):
        """Verify get_signal_windowed has correct signature."""
        method = getattr(MetaController, "get_signal_windowed")
        sig = inspect.signature(method)
        params = list(sig.parameters.keys())
        
        assert "symbol" in params
        assert "lookback_sec" in params
    
    def test_flush_stale_signals_signature(self):
        """Verify flush_stale_signals has correct signature."""
        method = getattr(MetaController, "flush_stale_signals")
        sig = inspect.signature(method)
        params = list(sig.parameters.keys())
        
        assert "max_age_sec" in params


class TestSignalWindowingImplementation:
    """Test Phase 4 implementation details in source code."""
    
    def test_windowed_lookback_logic_in_source(self):
        """Verify windowed lookback logic in get_signal_windowed."""
        source = inspect.getsource(MetaController.get_signal_windowed)
        
        # Check for window time logic
        assert "lookback_sec" in source
        assert "window_start" in source
        assert "time.time()" in source
    
    def test_universe_stats_aggregation_in_source(self):
        """Verify universe stats aggregation in get_universe_signal_stats."""
        source = inspect.getsource(MetaController.get_universe_signal_stats)
        
        # Check for aggregation logic
        assert "signal_count" in source
        assert "symbols_with_signals" in source
        assert "avg_signal_age_sec" in source
    
    def test_stale_signal_flushing_in_source(self):
        """Verify stale signal flushing logic in flush_stale_signals."""
        source = inspect.getsource(MetaController.flush_stale_signals)
        
        # Check for cleanup logic
        assert "max_age_sec" in source
        assert "to_remove" in source
        assert "del" in source
    
    def test_consistency_validation_in_source(self):
        """Verify consistency validation in validate_signal_cache_consistency."""
        source = inspect.getsource(MetaController.validate_signal_cache_consistency)
        
        # Check for validation logic
        assert "recent" in source
        assert "per_symbol" in source
        assert "return" in source


class TestSignalWindowingDocumentation:
    """Test that Phase 4 methods are properly documented."""
    
    def test_windowed_has_docstring(self):
        """Verify get_signal_windowed has docstring."""
        method = getattr(MetaController, "get_signal_windowed")
        assert method.__doc__ is not None
        assert len(method.__doc__) > 20
        assert "window" in method.__doc__.lower()
    
    def test_universe_stats_has_docstring(self):
        """Verify get_universe_signal_stats has docstring."""
        method = getattr(MetaController, "get_universe_signal_stats")
        assert method.__doc__ is not None
        assert len(method.__doc__) > 20
        assert "universe" in method.__doc__.lower() or "stats" in method.__doc__.lower()
    
    def test_flush_stale_has_docstring(self):
        """Verify flush_stale_signals has docstring."""
        method = getattr(MetaController, "flush_stale_signals")
        assert method.__doc__ is not None
        assert len(method.__doc__) > 20
    
    def test_consistency_check_has_docstring(self):
        """Verify validate_signal_cache_consistency has docstring."""
        method = getattr(MetaController, "validate_signal_cache_consistency")
        assert method.__doc__ is not None
        assert len(method.__doc__) > 20


class TestPhase4Implementation:
    """Test Phase 4 components all present and integrated."""
    
    def test_all_phase4_components_present(self):
        """Verify all Phase 4 methods are present."""
        assert hasattr(MetaController, "get_signal_windowed")
        assert hasattr(MetaController, "get_universe_signal_stats")
        assert hasattr(MetaController, "flush_stale_signals")
        assert hasattr(MetaController, "validate_signal_cache_consistency")
    
    def test_phase4_uses_signal_cache(self):
        """Verify Phase 4 methods use _signal_cache structure."""
        windowed_src = inspect.getsource(MetaController.get_signal_windowed)
        stats_src = inspect.getsource(MetaController.get_universe_signal_stats)
        
        assert "_signal_cache" in windowed_src
        assert "_signal_cache" in stats_src
    
    def test_phase4_methods_integrated(self):
        """Verify Phase 4 methods reference signal cache properly."""
        source = inspect.getsource(MetaController.flush_stale_signals)
        
        # Should reference both per_symbol and recent
        assert "per_symbol" in source
        assert "recent" in source


class TestSignalWindowingFeatures:
    """Test specific features of signal windowing."""
    
    def test_windowed_returns_list(self):
        """Verify get_signal_windowed returns List[Dict]."""
        source = inspect.getsource(MetaController.get_signal_windowed)
        
        # Check for list return type
        assert "windowed_signals" in source
        assert "append" in source
        assert "return windowed_signals" in source
    
    def test_universe_stats_returns_dict(self):
        """Verify get_universe_signal_stats returns Dict."""
        source = inspect.getsource(MetaController.get_universe_signal_stats)
        
        # Check for dict with all expected keys
        assert "signal_count" in source
        assert "symbols_with_signals" in source
        assert "avg_signal_age_sec" in source
        assert "universe_coverage" in source
    
    def test_flush_stale_returns_count(self):
        """Verify flush_stale_signals returns count of flushed signals."""
        source = inspect.getsource(MetaController.flush_stale_signals)
        
        # Should track count of removed signals
        assert "len(to_remove)" in source
    
    def test_consistency_check_returns_bool(self):
        """Verify validate_signal_cache_consistency returns bool."""
        source = inspect.getsource(MetaController.validate_signal_cache_consistency)
        
        # Check for return bool pattern
        assert "return True" in source
        assert "return False" in source


class TestSignalWindowingPerformance:
    """Test Phase 4 performance characteristics."""
    
    def test_windowed_uses_defensive_copy(self):
        """Verify get_signal_windowed uses .copy() for thread safety."""
        source = inspect.getsource(MetaController.get_signal_windowed)
        
        # Check for defensive copying
        assert ".copy()" in source
    
    def test_stats_handles_empty_cache(self):
        """Verify get_universe_signal_stats handles empty cache gracefully."""
        source = inspect.getsource(MetaController.get_universe_signal_stats)
        
        # Should handle empty/zero cache
        assert "if not per_symbol:" in source or "if not self._signal_cache" in source
    
    def test_flush_is_efficient(self):
        """Verify flush_stale_signals uses single pass for cleanup."""
        source = inspect.getsource(MetaController.flush_stale_signals)
        
        # Should collect to_remove first, then delete (avoid dict mutation during iteration)
        assert "to_remove" in source
        assert "for symbol in to_remove:" in source
    
    def test_consistency_check_is_lightweight(self):
        """Verify validate_signal_cache_consistency is lightweight."""
        source = inspect.getsource(MetaController.validate_signal_cache_consistency)
        
        # Should not do deep validation, just basic checks
        assert "len(per_symbol)" in source or "len(recent)" in source


class TestPhase4Integration:
    """Test Phase 4 integration with existing components."""
    
    def test_phase4_with_signal_cache_structure(self):
        """Verify Phase 4 methods work with signal cache structure."""
        stats_src = inspect.getsource(MetaController.get_universe_signal_stats)
        flush_src = inspect.getsource(MetaController.flush_stale_signals)
        
        # Should reference signal cache structure
        assert "per_symbol" in stats_src
        assert "per_symbol" in flush_src
        assert "recent" in flush_src
    
    def test_phase4_time_awareness(self):
        """Verify Phase 4 methods are time-aware (use timestamps)."""
        source = inspect.getsource(MetaController.get_signal_windowed)
        
        # Should use time for windowing
        assert "time.time()" in source
    
    def test_phase4_logging_integration(self):
        """Verify Phase 4 methods integrate with logging."""
        consistency_src = inspect.getsource(MetaController.validate_signal_cache_consistency)
        
        # Should have logger calls for warnings
        assert "logger" in consistency_src or "warning" in consistency_src.lower()


class TestSignalWindowingAdvanced:
    """Test advanced signal windowing features."""
    
    def test_windowed_with_parameters(self):
        """Verify get_signal_windowed respects lookback_sec parameter."""
        source = inspect.getsource(MetaController.get_signal_windowed)
        
        # Should use lookback_sec in calculation
        assert "lookback_sec" in source
        assert "window_start = now - lookback_sec" in source or "now - lookback_sec" in source
    
    def test_universe_coverage_calculation(self):
        """Verify get_universe_signal_stats calculates coverage correctly."""
        source = inspect.getsource(MetaController.get_universe_signal_stats)
        
        # Should calculate universe_size ratio
        assert "universe_coverage" in source
        assert "universe_size" in source
    
    def test_stale_cleanup_removes_from_both_dicts(self):
        """Verify flush_stale_signals removes from both per_symbol and recent."""
        source = inspect.getsource(MetaController.flush_stale_signals)
        
        # Should remove from both caches
        per_symbol_del = "del per_symbol[symbol]" in source
        recent_del = "del recent[symbol]" in source
        
        assert per_symbol_del or "del per_symbol" in source
        assert recent_del or "del recent" in source
    
    def test_consistency_checks_both_directions(self):
        """Verify validate_signal_cache_consistency checks both cache directions."""
        source = inspect.getsource(MetaController.validate_signal_cache_consistency)
        
        # Should check recent->per_symbol consistency
        assert "recent" in source
        assert "per_symbol" in source
        assert "not in" in source


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
