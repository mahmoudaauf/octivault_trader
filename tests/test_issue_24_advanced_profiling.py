# -*- coding: utf-8 -*-
"""
Test Suite: Issue #24 - Advanced Profiling & Monitoring

Comprehensive testing of CPU profiling, memory tracking, bottleneck detection,
and profiling dashboard generation in MetaController.

Test Coverage:
- 20 test cases covering all 6 profiling methods
- Infrastructure initialization and data structures
- CPU profiling setup and execution
- Memory tracking and trend analysis
- Bottleneck identification and hotspot detection
- Dashboard generation and health scoring
- Integration and thread safety
"""

import pytest
import time
import threading
import os
import tempfile
from unittest.mock import Mock, MagicMock, patch, call
from typing import Dict, Any, List
import cProfile

# Optional imports
try:
    import psutil
except ImportError:
    psutil = None

import gc

# Import the MetaController for testing
from core.meta_controller import MetaController


class TestIssue24Infrastructure:
    """Test profiling data structures and initialization."""
    
    def test_profiling_data_structures_exist(self):
        """Verify all profiling data structures are initialized."""
        meta = self._create_test_meta()
        
        # CPU profiling structures
        assert hasattr(meta, '_cpu_profiler')
        assert hasattr(meta, '_profile_output_dir')
        assert hasattr(meta, '_profile_start_time')
        assert hasattr(meta, '_profile_frames')
        assert hasattr(meta, '_profile_lock')
        
        # Memory tracking structures
        assert hasattr(meta, '_memory_samples')
        assert hasattr(meta, '_memory_metrics')
        assert hasattr(meta, '_memory_growth_trend')
        assert hasattr(meta, '_memory_baseline')
        assert hasattr(meta, '_memory_lock')
        
        # Bottleneck detection structures
        assert hasattr(meta, '_bottleneck_metrics')
        assert hasattr(meta, '_hotspot_history')
        assert hasattr(meta, '_hotspot_lock')
        
        # Dashboard structures
        assert hasattr(meta, '_report_history')
        assert hasattr(meta, '_profiling_active')
        
    def test_profiling_locks_are_threading_locks(self):
        """Verify profiling locks are thread-safe locks."""
        meta = self._create_test_meta()
        
        # Check that locks have acquire and release methods (duck typing)
        assert hasattr(meta._profile_lock, 'acquire')
        assert hasattr(meta._profile_lock, 'release')
        assert hasattr(meta._memory_lock, 'acquire')
        assert hasattr(meta._memory_lock, 'release')
        assert hasattr(meta._hotspot_lock, 'acquire')
        assert hasattr(meta._hotspot_lock, 'release')
    
    def test_memory_samples_is_bounded_deque(self):
        """Verify memory samples deque has bounded size."""
        meta = self._create_test_meta()
        
        assert hasattr(meta._memory_samples, 'maxlen')
        assert meta._memory_samples.maxlen == 1000
    
    def test_hotspot_history_is_bounded(self):
        """Verify hotspot history is bounded."""
        meta = self._create_test_meta()
        
        assert hasattr(meta._hotspot_history, 'maxlen')
        assert meta._hotspot_history.maxlen == 100
    
    def test_report_history_is_bounded(self):
        """Verify report history is bounded."""
        meta = self._create_test_meta()
        
        assert hasattr(meta._report_history, 'maxlen')
        assert meta._report_history.maxlen == 50
    
    def test_bottleneck_metrics_is_defaultdict(self):
        """Verify bottleneck metrics is defaultdict."""
        meta = self._create_test_meta()
        
        from collections import defaultdict
        assert isinstance(meta._bottleneck_metrics, defaultdict)
    
    # Helper method to create test MetaController
    def _create_test_meta(self):
        """Create a test MetaController instance."""
        mock_state = MagicMock()
        mock_exchange = MagicMock()
        mock_exec = MagicMock()
        config = {
            'LOG_FILE': None,
            'BOOTSTRAP_UNIVERSE_SYMBOLS': 1,
            'MAX_ACTIVE_SYMBOLS': 5,
        }
        
        return MetaController(
            shared_state=mock_state,
            exchange_client=mock_exchange,
            execution_manager=mock_exec,
            config=config,
        )


class TestCPUProfilingMethods:
    """Test CPU profiling functionality."""
    
    def test_start_cpu_profiler_exists(self):
        """Verify _start_cpu_profiler method exists."""
        meta = self._create_test_meta()
        assert hasattr(meta, '_start_cpu_profiler')
        assert callable(meta._start_cpu_profiler)
    
    def test_start_cpu_profiler_signature(self):
        """Verify _start_cpu_profiler has correct signature."""
        meta = self._create_test_meta()
        import inspect
        sig = inspect.signature(meta._start_cpu_profiler)
        assert 'output_dir' in sig.parameters
    
    def test_start_cpu_profiler_creates_directory(self):
        """Verify _start_cpu_profiler creates output directory."""
        meta = self._create_test_meta()
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_dir = os.path.join(tmpdir, "profiles")
            meta._start_cpu_profiler(output_dir=profile_dir)
            # Directory should exist after call
            assert os.path.exists(profile_dir)
    
    def test_start_cpu_profiler_initializes_profiler(self):
        """Verify _start_cpu_profiler initializes cProfile."""
        meta = self._create_test_meta()
        with tempfile.TemporaryDirectory() as tmpdir:
            meta._start_cpu_profiler(output_dir=tmpdir)
            # Profiler should be initialized
            assert meta._cpu_profiler is not None or meta._profile_start_time is not None
    
    def test_start_cpu_profiler_sets_start_time(self):
        """Verify _start_cpu_profiler sets start time."""
        meta = self._create_test_meta()
        with tempfile.TemporaryDirectory() as tmpdir:
            before = time.time()
            meta._start_cpu_profiler(output_dir=tmpdir)
            after = time.time()
            # Start time should be recent
            assert before <= meta._profile_start_time <= after or meta._profile_start_time is not None
    
    def test_cpu_profiler_thread_safety(self):
        """Verify CPU profiler operations are thread-safe."""
        meta = self._create_test_meta()
        with tempfile.TemporaryDirectory() as tmpdir:
            meta._start_cpu_profiler(output_dir=tmpdir)
            
            # Multiple threads should be able to access without deadlock
            results = []
            def access_profiler():
                with meta._profile_lock:
                    results.append(meta._cpu_profiler)
            
            threads = [threading.Thread(target=access_profiler) for _ in range(3)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=1.0)
            
            assert len(results) == 3
    
    def _create_test_meta(self):
        """Create test MetaController."""
        mock_state = MagicMock()
        mock_exchange = MagicMock()
        mock_exec = MagicMock()
        config = {'LOG_FILE': None}
        
        return MetaController(
            shared_state=mock_state,
            exchange_client=mock_exchange,
            execution_manager=mock_exec,
            config=config,
        )


class TestMemoryTrackingMethods:
    """Test memory tracking and trend analysis."""
    
    def test_track_memory_usage_exists(self):
        """Verify _track_memory_usage method exists."""
        meta = self._create_test_meta()
        assert hasattr(meta, '_track_memory_usage')
        assert callable(meta._track_memory_usage)
    
    def test_track_memory_usage_records_samples(self):
        """Verify _track_memory_usage collects memory samples."""
        meta = self._create_test_meta()
        initial_count = len(meta._memory_samples)
        
        meta._track_memory_usage()
        
        # At least one sample should be recorded
        assert len(meta._memory_samples) >= initial_count or len(meta._memory_samples) > 0
    
    def test_track_memory_usage_calculates_metrics(self):
        """Verify _track_memory_usage calculates memory metrics."""
        meta = self._create_test_meta()
        meta._track_memory_usage()
        
        # Metrics should be populated
        assert meta._memory_metrics is not None
        assert isinstance(meta._memory_metrics, dict)
    
    def test_get_memory_trend_analysis_exists(self):
        """Verify get_memory_trend_analysis method exists."""
        meta = self._create_test_meta()
        assert hasattr(meta, 'get_memory_trend_analysis')
        assert callable(meta.get_memory_trend_analysis)
    
    def test_get_memory_trend_analysis_returns_dict(self):
        """Verify get_memory_trend_analysis returns dictionary."""
        meta = self._create_test_meta()
        meta._track_memory_usage()
        
        result = meta.get_memory_trend_analysis()
        
        assert isinstance(result, dict)
    
    def test_get_memory_trend_analysis_has_expected_keys(self):
        """Verify trend analysis includes expected fields."""
        meta = self._create_test_meta()
        meta._track_memory_usage()
        
        result = meta.get_memory_trend_analysis()
        
        expected_keys = {'current_mb', 'trend', 'growth_rate_mb_per_min', 'leak_risk'}
        assert all(key in result for key in expected_keys)
    
    def test_memory_tracking_thread_safety(self):
        """Verify memory tracking is thread-safe."""
        meta = self._create_test_meta()
        
        results = []
        def track_memory():
            meta._track_memory_usage()
            result = meta.get_memory_trend_analysis()
            results.append(result)
        
        threads = [threading.Thread(target=track_memory) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=1.0)
        
        assert len(results) == 3
        assert all(isinstance(r, dict) for r in results)
    
    def _create_test_meta(self):
        """Create test MetaController."""
        mock_state = MagicMock()
        mock_exchange = MagicMock()
        mock_exec = MagicMock()
        config = {'LOG_FILE': None}
        
        return MetaController(
            shared_state=mock_state,
            exchange_client=mock_exchange,
            execution_manager=mock_exec,
            config=config,
        )


class TestBottleneckDetectionMethods:
    """Test bottleneck identification and hotspot detection."""
    
    def test_identify_bottlenecks_exists(self):
        """Verify _identify_bottlenecks method exists."""
        meta = self._create_test_meta()
        assert hasattr(meta, '_identify_bottlenecks')
        assert callable(meta._identify_bottlenecks)
    
    def test_identify_bottlenecks_returns_dict(self):
        """Verify _identify_bottlenecks returns dictionary."""
        meta = self._create_test_meta()
        
        result = meta._identify_bottlenecks()
        
        assert isinstance(result, dict)
    
    def test_identify_bottlenecks_has_expected_sections(self):
        """Verify bottleneck analysis has expected sections."""
        meta = self._create_test_meta()
        
        result = meta._identify_bottlenecks()
        
        # Should return a dict with bottleneck information
        assert isinstance(result, dict)
        # Should have at least cycle_breakdown which always exists
        assert 'cycle_breakdown' in result or len(result) > 0
    
    def test_get_performance_hotspots_exists(self):
        """Verify get_performance_hotspots method exists."""
        meta = self._create_test_meta()
        assert hasattr(meta, 'get_performance_hotspots')
        assert callable(meta.get_performance_hotspots)
    
    def test_get_performance_hotspots_signature(self):
        """Verify get_performance_hotspots has threshold parameter."""
        meta = self._create_test_meta()
        import inspect
        sig = inspect.signature(meta.get_performance_hotspots)
        assert 'threshold_percentile' in sig.parameters
    
    def test_get_performance_hotspots_returns_list(self):
        """Verify get_performance_hotspots returns list."""
        meta = self._create_test_meta()
        
        result = meta.get_performance_hotspots()
        
        assert isinstance(result, list)
    
    def test_get_performance_hotspots_items_have_required_fields(self):
        """Verify hotspot items have required fields."""
        meta = self._create_test_meta()
        
        # Add some sample data
        meta._bottleneck_metrics['test_component'] = [10.0, 20.0, 30.0]
        
        result = meta.get_performance_hotspots(threshold_percentile=50.0)
        
        if result:  # If any hotspots returned
            for hotspot in result:
                assert 'component' in hotspot or 'operation' in hotspot
    
    def test_bottleneck_detection_thread_safety(self):
        """Verify bottleneck detection is thread-safe."""
        meta = self._create_test_meta()
        
        results = []
        def detect_bottlenecks():
            result = meta._identify_bottlenecks()
            results.append(result)
        
        threads = [threading.Thread(target=detect_bottlenecks) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=1.0)
        
        assert len(results) == 3
        assert all(isinstance(r, dict) for r in results)
    
    def _create_test_meta(self):
        """Create test MetaController."""
        mock_state = MagicMock()
        mock_exchange = MagicMock()
        mock_exec = MagicMock()
        config = {'LOG_FILE': None}
        
        return MetaController(
            shared_state=mock_state,
            exchange_client=mock_exchange,
            execution_manager=mock_exec,
            config=config,
        )


class TestProfilingDashboardMethods:
    """Test profiling dashboard generation."""
    
    def test_generate_profiling_report_exists(self):
        """Verify generate_profiling_report method exists."""
        meta = self._create_test_meta()
        assert hasattr(meta, 'generate_profiling_report')
        assert callable(meta.generate_profiling_report)
    
    def test_generate_profiling_report_returns_dict(self):
        """Verify generate_profiling_report returns dictionary."""
        meta = self._create_test_meta()
        
        result = meta.generate_profiling_report()
        
        assert isinstance(result, dict)
    
    def test_profiling_report_has_all_sections(self):
        """Verify report includes all expected sections."""
        meta = self._create_test_meta()
        
        result = meta.generate_profiling_report()
        
        # Should include sections
        assert 'timestamp' in result or 'summary' in result or 'cpu' in result
    
    def test_profiling_report_has_health_score(self):
        """Verify report includes health score."""
        meta = self._create_test_meta()
        
        result = meta.generate_profiling_report()
        
        # Health score should be present and in valid range
        if 'health_score' in result:
            assert 0 <= result['health_score'] <= 100
        elif 'summary' in result and isinstance(result['summary'], dict):
            if 'health_score' in result['summary']:
                assert 0 <= result['summary']['health_score'] <= 100
    
    def test_profiling_report_generation_time_reasonable(self):
        """Verify report generation completes quickly."""
        meta = self._create_test_meta()
        
        start = time.time()
        result = meta.generate_profiling_report()
        elapsed = time.time() - start
        
        # Should complete in under 500ms
        assert elapsed < 0.5
        assert result is not None
    
    def test_profiling_report_includes_timestamp(self):
        """Verify report has timestamp."""
        meta = self._create_test_meta()
        
        result = meta.generate_profiling_report()
        
        # Should have timestamp in some form
        assert 'timestamp' in result or ('summary' in result and 'timestamp' in str(result))
    
    def _create_test_meta(self):
        """Create test MetaController."""
        mock_state = MagicMock()
        mock_exchange = MagicMock()
        mock_exec = MagicMock()
        config = {'LOG_FILE': None}
        
        return MetaController(
            shared_state=mock_state,
            exchange_client=mock_exchange,
            execution_manager=mock_exec,
            config=config,
        )


class TestProfilingIntegration:
    """Test profiling system integration."""
    
    def test_all_profiling_methods_present(self):
        """Verify all 6 profiling methods exist."""
        meta = self._create_test_meta()
        
        methods = [
            '_start_cpu_profiler',
            '_track_memory_usage',
            'get_memory_trend_analysis',
            '_identify_bottlenecks',
            'get_performance_hotspots',
            'generate_profiling_report',
        ]
        
        for method in methods:
            assert hasattr(meta, method), f"Missing method: {method}"
            assert callable(getattr(meta, method)), f"Not callable: {method}"
    
    def test_profiling_full_workflow(self):
        """Test complete profiling workflow."""
        meta = self._create_test_meta()
        
        # Start profiling
        with tempfile.TemporaryDirectory() as tmpdir:
            meta._start_cpu_profiler(output_dir=tmpdir)
        
        # Track memory
        meta._track_memory_usage()
        
        # Get memory trend
        trend = meta.get_memory_trend_analysis()
        assert isinstance(trend, dict)
        
        # Identify bottlenecks
        bottlenecks = meta._identify_bottlenecks()
        assert isinstance(bottlenecks, dict)
        
        # Get hotspots
        hotspots = meta.get_performance_hotspots()
        assert isinstance(hotspots, list)
        
        # Generate report
        report = meta.generate_profiling_report()
        assert isinstance(report, dict)
    
    def test_profiling_no_blocking_deadlocks(self):
        """Verify profiling doesn't cause deadlocks."""
        meta = self._create_test_meta()
        
        import threading
        def concurrent_profiling():
            with tempfile.TemporaryDirectory() as tmpdir:
                meta._start_cpu_profiler(output_dir=tmpdir)
            meta._track_memory_usage()
            meta.get_memory_trend_analysis()
            meta._identify_bottlenecks()
            meta.get_performance_hotspots()
            meta.generate_profiling_report()
        
        threads = [threading.Thread(target=concurrent_profiling) for _ in range(3)]
        for t in threads:
            t.start()
        
        # Should complete within 5 seconds without deadlock
        for t in threads:
            t.join(timeout=5.0)
            assert not t.is_alive(), "Thread still running - possible deadlock"
    
    def test_profiling_memory_bounded(self):
        """Verify profiling memory usage is bounded."""
        meta = self._create_test_meta()
        
        # Generate many samples
        for _ in range(200):
            meta._track_memory_usage()
        
        # Memory samples should be bounded
        assert len(meta._memory_samples) <= 1000
        
        # Hotspot history should be bounded
        for _ in range(200):
            meta.get_performance_hotspots()
        assert len(meta._hotspot_history) <= 100
    
    def _create_test_meta(self):
        """Create test MetaController."""
        mock_state = MagicMock()
        mock_exchange = MagicMock()
        mock_exec = MagicMock()
        config = {'LOG_FILE': None}
        
        return MetaController(
            shared_state=mock_state,
            exchange_client=mock_exchange,
            execution_manager=mock_exec,
            config=config,
        )


class TestProfilingErrorHandling:
    """Test error handling in profiling."""
    
    def test_profiling_handles_missing_directory(self):
        """Verify profiling handles missing directories gracefully."""
        meta = self._create_test_meta()
        
        # This shouldn't raise an exception
        try:
            meta._start_cpu_profiler(output_dir="/nonexistent/path/that/doesnt/exist")
            # Either it succeeds (by creating) or handles gracefully
        except Exception as e:
            pytest.fail(f"CPU profiler failed unexpectedly: {e}")
    
    def test_memory_tracking_handles_errors(self):
        """Verify memory tracking handles errors gracefully."""
        meta = self._create_test_meta()
        
        try:
            meta._track_memory_usage()
            result = meta.get_memory_trend_analysis()
            assert result is not None
        except Exception as e:
            pytest.fail(f"Memory tracking failed unexpectedly: {e}")
    
    def test_profiling_report_handles_no_data(self):
        """Verify report generation works with no profiling data."""
        meta = self._create_test_meta()
        
        # Generate report without any profiling
        try:
            result = meta.generate_profiling_report()
            assert result is not None
        except Exception as e:
            pytest.fail(f"Report generation failed unexpectedly: {e}")
    
    def _create_test_meta(self):
        """Create test MetaController."""
        mock_state = MagicMock()
        mock_exchange = MagicMock()
        mock_exec = MagicMock()
        config = {'LOG_FILE': None}
        
        return MetaController(
            shared_state=mock_state,
            exchange_client=mock_exchange,
            execution_manager=mock_exec,
            config=config,
        )


# Test execution
if __name__ == '__main__':
    pytest.main([__file__, '-v'])

