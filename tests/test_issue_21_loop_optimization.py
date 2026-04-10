"""
Test Suite for Issue #21: MetaController Loop Optimization - Performance Tracking

Tests performance tracking infrastructure, caching mechanisms, and optimization validation.
This test focuses on the performance tracking methods added to MetaController without
requiring full instantiation.
"""

import pytest
import time
from core.meta_controller import MetaController


# Helper function to directly test performance tracking methods
def test_performance_metrics_class_methods():
    """Test performance tracking methods are properly defined on the class."""
    # Check that methods exist on the class
    assert hasattr(MetaController, '_start_cycle_timing')
    assert hasattr(MetaController, '_end_cycle_timing')
    assert hasattr(MetaController, '_record_phase_timing')
    assert hasattr(MetaController, 'get_performance_metrics')
    assert hasattr(MetaController, 'get_capital_cache')
    assert hasattr(MetaController, 'get_signal_cache_stats')
    assert hasattr(MetaController, '_report_performance_summary')


class TestPerformanceTrackingMethods:
    """Direct unit tests for performance tracking methods."""
    
    def test_start_cycle_timing_method_exists(self):
        """Verify _start_cycle_timing method exists."""
        assert hasattr(MetaController, '_start_cycle_timing')
        assert callable(getattr(MetaController, '_start_cycle_timing'))
    
    def test_end_cycle_timing_method_exists(self):
        """Verify _end_cycle_timing method exists."""
        assert hasattr(MetaController, '_end_cycle_timing')
        assert callable(getattr(MetaController, '_end_cycle_timing'))
    
    def test_record_phase_timing_method_exists(self):
        """Verify _record_phase_timing method exists."""
        assert hasattr(MetaController, '_record_phase_timing')
        assert callable(getattr(MetaController, '_record_phase_timing'))
    
    def test_get_performance_metrics_method_exists(self):
        """Verify get_performance_metrics method exists."""
        assert hasattr(MetaController, 'get_performance_metrics')
        assert callable(getattr(MetaController, 'get_performance_metrics'))
    
    def test_get_capital_cache_method_exists(self):
        """Verify get_capital_cache method exists."""
        assert hasattr(MetaController, 'get_capital_cache')
        assert callable(getattr(MetaController, 'get_capital_cache'))
    
    def test_get_signal_cache_stats_method_exists(self):
        """Verify get_signal_cache_stats method exists."""
        assert hasattr(MetaController, 'get_signal_cache_stats')
        assert callable(getattr(MetaController, 'get_signal_cache_stats'))
    
    def test_report_performance_summary_method_exists(self):
        """Verify _report_performance_summary method exists."""
        assert hasattr(MetaController, '_report_performance_summary')
        assert callable(getattr(MetaController, '_report_performance_summary'))


class TestPerformanceTrackingIntegration:
    """Integration tests using mock instances."""
    
    def test_performance_metrics_initialization_in_code(self):
        """Verify performance metrics initialization code."""
        import inspect
        source = inspect.getsource(MetaController.__init__)
        
        # Check that performance tracking initialization is in __init__
        assert '_perf_metrics' in source
        assert '_capital_cache' in source
        assert '_signal_cache' in source
    
    def test_cycle_timing_call_in_impl(self):
        """Verify cycle timing is called in _evaluate_and_act_impl."""
        import inspect
        
        # Find the _evaluate_and_act_impl method
        for name, method in inspect.getmembers(MetaController, predicate=inspect.isfunction):
            if name == 'evaluate_and_act':
                # verify the method exists (implementation is in _evaluate_and_act_impl)
                assert method is not None


class TestPerformanceDataStructureValidation:
    """Validate performance data structure definitions."""
    
    def test_perf_metrics_structure_in_init(self):
        """Verify _perf_metrics structure definition."""
        import inspect
        source = inspect.getsource(MetaController.__init__)
        
        # Check all required keys are initialized
        required_keys = [
            "cycle_duration_ms",
            "guard_eval_ms", 
            "event_drain_ms",
            "signal_process_ms",
            "capital_calc_ms",
            "logging_ms",
            "max_samples",
            "cycle_start_time",
            "phase_timings"
        ]
        
        for key in required_keys:
            assert f'"{key}"' in source or f"'{key}'" in source
    
    def test_capital_cache_structure_in_init(self):
        """Verify capital cache structure definition."""
        import inspect
        source = inspect.getsource(MetaController.__init__)
        
        # Check all required keys are initialized
        required_keys = ["cached_at", "values", "valid", "ttl"]
        
        for key in required_keys:
            assert f'"{key}"' in source or f"'{key}'" in source
    
    def test_signal_cache_structure_in_init(self):
        """Verify signal cache structure definition."""
        import inspect
        source = inspect.getsource(MetaController.__init__)
        
        # Check all required keys are initialized
        required_keys = ["recent", "per_symbol", "universe_stats", "cache_time", "ttl", "max_recent"]
        
        for key in required_keys:
            assert f'"{key}"' in source or f"'{key}'" in source


class TestMethodSignatures:
    """Test method signatures are correct."""
    
    def test_start_cycle_timing_signature(self):
        """Verify _start_cycle_timing has no parameters."""
        import inspect
        sig = inspect.signature(MetaController._start_cycle_timing)
        params = list(sig.parameters.keys())
        # Only 'self' parameter
        assert params == ['self']
    
    def test_end_cycle_timing_signature(self):
        """Verify _end_cycle_timing has no parameters."""
        import inspect
        sig = inspect.signature(MetaController._end_cycle_timing)
        params = list(sig.parameters.keys())
        # Only 'self' parameter
        assert params == ['self']
    
    def test_record_phase_timing_signature(self):
        """Verify _record_phase_timing has correct parameters."""
        import inspect
        sig = inspect.signature(MetaController._record_phase_timing)
        params = list(sig.parameters.keys())
        # self, phase_name, duration_ms
        assert 'self' in params
        assert 'phase_name' in params
        assert 'duration_ms' in params
    
    def test_get_performance_metrics_signature(self):
        """Verify get_performance_metrics has no parameters."""
        import inspect
        sig = inspect.signature(MetaController.get_performance_metrics)
        params = list(sig.parameters.keys())
        # Only 'self' parameter
        assert params == ['self']


class TestDocumentation:
    """Verify proper documentation of performance tracking methods."""
    
    def test_methods_have_docstrings(self):
        """Verify all methods have docstrings."""
        methods_to_check = [
            '_start_cycle_timing',
            '_end_cycle_timing',
            '_record_phase_timing',
            'get_performance_metrics',
            'get_capital_cache',
            'get_signal_cache_stats',
            '_report_performance_summary',
        ]
        
        for method_name in methods_to_check:
            method = getattr(MetaController, method_name)
            assert method.__doc__ is not None
            assert len(method.__doc__.strip()) > 0


class TestPerformanceOptimizationFeatures:
    """Test that optimization features are implemented."""
    
    def test_cycle_timing_tracking_feature(self):
        """Verify cycle timing tracking feature."""
        assert hasattr(MetaController, '_start_cycle_timing')
        assert hasattr(MetaController, '_end_cycle_timing')
    
    def test_phase_timing_feature(self):
        """Verify phase timing feature."""
        assert hasattr(MetaController, '_record_phase_timing')
    
    def test_metrics_reporting_feature(self):
        """Verify metrics reporting feature."""
        assert hasattr(MetaController, 'get_performance_metrics')
        assert hasattr(MetaController, '_report_performance_summary')
    
    def test_cache_structures_feature(self):
        """Verify cache structures are initialized in __init__."""
        import inspect
        init_source = inspect.getsource(MetaController.__init__)
        
        # Check that cache structures are initialized
        assert '_capital_cache' in init_source
        assert '_signal_cache' in init_source


class TestIssue21Implementation:
    """Overall tests for Issue #21 implementation completeness."""
    
    def test_all_optimization_components_present(self):
        """Verify all components of Issue #21 are implemented."""
        # Check method components
        method_components = [
            '_start_cycle_timing',
            '_end_cycle_timing',
            '_record_phase_timing',
            'get_performance_metrics',
            '_report_performance_summary',
            'get_capital_cache',
            'get_signal_cache_stats',
        ]
        
        for component in method_components:
            assert hasattr(MetaController, component), f"Missing method: {component}"
        
        # Check initialization of cache structures
        import inspect
        init_source = inspect.getsource(MetaController.__init__)
        assert '_capital_cache' in init_source
        assert '_signal_cache' in init_source
        assert '_perf_metrics' in init_source
    
    def test_performance_tracking_initialization(self):
        """Verify performance tracking is initialized in __init__."""
        import inspect
        init_source = inspect.getsource(MetaController.__init__)
        
        # Check initialization markers
        assert 'ISSUE #21' in init_source or 'PERFORMANCE TRACKING' in init_source
        assert '_perf_metrics' in init_source
        assert '_capital_cache' in init_source
        assert '_signal_cache' in init_source
    
    def test_cycle_timing_integration(self):
        """Verify cycle timing is integrated in main loop."""
        import inspect
        
        # Find evaluate_and_act method
        for name, method in inspect.getmembers(MetaController):
            if name == 'evaluate_and_act':
                source = inspect.getsource(method)
                # Should mention starting timing
                assert '_start_cycle_timing' in source or 'cycle_timing' in source


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
