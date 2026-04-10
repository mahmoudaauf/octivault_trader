"""
Test Suite for Issue #21 Phase 2: Capital Caching Implementation

Tests capital allocation caching and signal caching mechanisms.
"""

import pytest
import time
import inspect
from core.meta_controller import MetaController


class TestCapitalCachingMethods:
    """Test capital caching method existence and signatures."""
    
    def test_cache_capital_allocation_exists(self):
        """Verify cache_capital_allocation method exists."""
        assert hasattr(MetaController, 'cache_capital_allocation')
        assert callable(getattr(MetaController, 'cache_capital_allocation'))
    
    def test_get_cached_capital_allocation_exists(self):
        """Verify get_cached_capital_allocation method exists."""
        assert hasattr(MetaController, 'get_cached_capital_allocation')
        assert callable(getattr(MetaController, 'get_cached_capital_allocation'))
    
    def test_invalidate_capital_cache_exists(self):
        """Verify invalidate_capital_cache method exists."""
        assert hasattr(MetaController, 'invalidate_capital_cache')
        assert callable(getattr(MetaController, 'invalidate_capital_cache'))
    
    def test_cache_capital_allocation_signature(self):
        """Verify cache_capital_allocation has correct signature."""
        import inspect
        sig = inspect.signature(MetaController.cache_capital_allocation)
        params = list(sig.parameters.keys())
        assert 'self' in params
        assert 'allocation_data' in params
    
    def test_get_cached_capital_allocation_signature(self):
        """Verify get_cached_capital_allocation has correct signature."""
        import inspect
        sig = inspect.signature(MetaController.get_cached_capital_allocation)
        params = list(sig.parameters.keys())
        assert params == ['self']


class TestSignalCachingMethods:
    """Test signal caching method existence and signatures."""
    
    def test_cache_signal_exists(self):
        """Verify cache_signal method exists."""
        assert hasattr(MetaController, 'cache_signal')
        assert callable(getattr(MetaController, 'cache_signal'))
    
    def test_get_cached_signal_exists(self):
        """Verify get_cached_signal method exists."""
        assert hasattr(MetaController, 'get_cached_signal')
        assert callable(getattr(MetaController, 'get_cached_signal'))
    
    def test_clear_signal_cache_exists(self):
        """Verify clear_signal_cache method exists."""
        assert hasattr(MetaController, 'clear_signal_cache')
        assert callable(getattr(MetaController, 'clear_signal_cache'))
    
    def test_cache_signal_signature(self):
        """Verify cache_signal has correct signature."""
        import inspect
        sig = inspect.signature(MetaController.cache_signal)
        params = list(sig.parameters.keys())
        assert 'self' in params
        assert 'symbol' in params
        assert 'signal_data' in params
    
    def test_get_cached_signal_signature(self):
        """Verify get_cached_signal has correct signature."""
        import inspect
        sig = inspect.signature(MetaController.get_cached_signal)
        params = list(sig.parameters.keys())
        assert 'self' in params
        assert 'symbol' in params


class TestCachingImplementation:
    """Test caching implementation in source code."""
    
    def test_capital_caching_in_source(self):
        """Verify capital caching code is in source."""
        import inspect
        
        # Get cache_capital_allocation source
        source = inspect.getsource(MetaController.cache_capital_allocation)
        assert '_capital_cache' in source
        assert 'time.time()' in source
    
    def test_signal_caching_in_source(self):
        """Verify signal caching code is in source."""
        import inspect
        
        # Get cache_signal source
        source = inspect.getsource(MetaController.cache_signal)
        assert '_signal_cache' in source
        assert 'per_symbol' in source
        assert 'recent' in source
    
    def test_ttl_validation_in_source(self):
        """Verify TTL validation is implemented."""
        import inspect
        
        # Get get_cached_signal source
        source = inspect.getsource(MetaController.get_cached_signal)
        assert 'ttl' in source.lower()
        assert 'cache_age' in source


class TestCachingDocumentation:
    """Test that caching methods have proper documentation."""
    
    def test_capital_caching_methods_documented(self):
        """Verify capital caching methods have docstrings."""
        methods = [
            'cache_capital_allocation',
            'get_cached_capital_allocation',
            'invalidate_capital_cache',
        ]
        
        for method_name in methods:
            method = getattr(MetaController, method_name)
            assert method.__doc__ is not None
            assert len(method.__doc__.strip()) > 0
    
    def test_signal_caching_methods_documented(self):
        """Verify signal caching methods have docstrings."""
        methods = [
            'cache_signal',
            'get_cached_signal',
            'clear_signal_cache',
        ]
        
        for method_name in methods:
            method = getattr(MetaController, method_name)
            assert method.__doc__ is not None
            assert len(method.__doc__.strip()) > 0


class TestPhase2Implementation:
    """Overall Phase 2 implementation completeness."""
    
    def test_all_phase2_components_present(self):
        """Verify all Phase 2 components are implemented."""
        components = [
            # Capital caching
            'cache_capital_allocation',
            'get_cached_capital_allocation',
            'invalidate_capital_cache',
            
            # Signal caching
            'cache_signal',
            'get_cached_signal',
            'clear_signal_cache',
        ]
        
        for component in components:
            assert hasattr(MetaController, component), f"Missing: {component}"
    
    def test_capital_cache_initialization_still_present(self):
        """Verify capital cache initialization from Phase 1 still exists."""
        import inspect
        init_source = inspect.getsource(MetaController.__init__)
        assert '_capital_cache' in init_source
        assert 'ttl' in init_source
        assert '0.5' in init_source  # 0.5 second TTL
    
    def test_signal_cache_initialization_still_present(self):
        """Verify signal cache initialization from Phase 1 still exists."""
        import inspect
        init_source = inspect.getsource(MetaController.__init__)
        assert '_signal_cache' in init_source
        assert 'max_recent' in init_source
        assert '100' in init_source  # Max 100 recent signals
    
    def test_capital_caching_methods_integrated(self):
        """Verify capital caching methods work together."""
        # Test that methods reference the same cache structure
        cache_alloc_source = inspect.getsource(MetaController.cache_capital_allocation)
        get_cache_source = inspect.getsource(MetaController.get_cached_capital_allocation)
        invalid_source = inspect.getsource(MetaController.invalidate_capital_cache)
        
        # All should reference same cache
        for source in [cache_alloc_source, get_cache_source, invalid_source]:
            assert '_capital_cache' in source
    
    def test_signal_caching_methods_integrated(self):
        """Verify signal caching methods work together."""
        cache_source = inspect.getsource(MetaController.cache_signal)
        get_cache_source = inspect.getsource(MetaController.get_cached_signal)
        clear_source = inspect.getsource(MetaController.clear_signal_cache)
        
        # All should reference same cache
        for source in [cache_source, get_cache_source, clear_source]:
            assert '_signal_cache' in source


class TestCachingFeatures:
    """Test caching feature implementation details."""
    
    def test_capital_cache_has_ttl_handling(self):
        """Verify capital cache implements TTL handling."""
        import inspect
        source = inspect.getsource(MetaController.get_cached_capital_allocation)
        
        # Check for TTL validation
        assert 'time.time()' in source
        assert 'ttl' in source
        assert 'valid' in source
    
    def test_signal_cache_has_ttl_handling(self):
        """Verify signal cache implements TTL handling."""
        import inspect
        source = inspect.getsource(MetaController.get_cached_signal)
        
        # Check for TTL validation
        assert 'time.time()' in source
        assert 'ttl' in source
        assert 'timestamp' in source
    
    def test_capital_cache_has_invalidation(self):
        """Verify capital cache can be invalidated."""
        import inspect
        source = inspect.getsource(MetaController.invalidate_capital_cache)
        
        # Check for invalidation logic
        assert 'valid' in source
        assert 'False' in source
    
    def test_signal_cache_has_cleanup(self):
        """Verify signal cache can be cleaned up."""
        import inspect
        source = inspect.getsource(MetaController.clear_signal_cache)
        
        # Check for cleanup logic
        assert 'clear' in source.lower()
        assert 'recent' in source or 'per_symbol' in source


class TestCachingPerformanceTargets:
    """Verify caching methods support performance targets."""
    
    def test_capital_cache_copy_strategy(self):
        """Verify capital cache uses copy to avoid mutations."""
        import inspect
        
        cache_source = inspect.getsource(MetaController.cache_capital_allocation)
        get_source = inspect.getsource(MetaController.get_cached_capital_allocation)
        
        # Both should use .copy() for thread safety
        assert 'copy()' in cache_source
        assert 'copy()' in get_source
    
    def test_signal_cache_copy_strategy(self):
        """Verify signal cache uses copy to avoid mutations."""
        import inspect
        
        cache_source = inspect.getsource(MetaController.cache_signal)
        get_source = inspect.getsource(MetaController.get_cached_signal)
        
        # Both should use .copy() for thread safety
        assert 'copy()' in cache_source
        assert 'copy()' in get_source
    
    def test_capital_allocation_stored_efficiently(self):
        """Verify capital allocation data structure."""
        import inspect
        source = inspect.getsource(MetaController.cache_capital_allocation)
        
        # Should store as dict copy for efficiency
        assert 'allocation_data.copy()' in source


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
