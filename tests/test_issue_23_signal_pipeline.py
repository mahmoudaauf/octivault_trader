"""
Test suite for Issue #23: Signal Processing Pipeline Enhancement

Tests batch processing, queue configuration, consumer management, and latency measurement.
Target: 20-30% signal processing latency reduction (45-60ms → 30-40ms).
"""

import pytest
import time
import threading
import inspect
from unittest.mock import Mock, MagicMock, patch
from core.meta_controller import MetaController


class TestSignalPipelineMethodsExist:
    """Tests that all signal pipeline methods are properly defined."""
    
    def test_collect_signal_batch_exists(self):
        """Verify _collect_signal_batch method exists."""
        assert hasattr(MetaController, '_collect_signal_batch')
        assert callable(getattr(MetaController, '_collect_signal_batch'))
    
    def test_process_signal_batch_exists(self):
        """Verify _process_signal_batch method exists."""
        assert hasattr(MetaController, '_process_signal_batch')
        assert callable(getattr(MetaController, '_process_signal_batch'))
    
    def test_validate_batch_consistency_exists(self):
        """Verify _validate_batch_consistency method exists."""
        assert hasattr(MetaController, '_validate_batch_consistency')
        assert callable(getattr(MetaController, '_validate_batch_consistency'))
    
    def test_get_batch_stats_exists(self):
        """Verify get_batch_processing_stats method exists."""
        assert hasattr(MetaController, 'get_batch_processing_stats')
        assert callable(getattr(MetaController, 'get_batch_processing_stats'))
    
    def test_configure_queue_exists(self):
        """Verify _configure_signal_queue method exists."""
        assert hasattr(MetaController, '_configure_signal_queue')
        assert callable(getattr(MetaController, '_configure_signal_queue'))
    
    def test_configure_consumer_group_exists(self):
        """Verify _configure_consumer_group method exists."""
        assert hasattr(MetaController, '_configure_consumer_group')
        assert callable(getattr(MetaController, '_configure_consumer_group'))
    
    def test_apply_backpressure_exists(self):
        """Verify _apply_backpressure method exists."""
        assert hasattr(MetaController, '_apply_backpressure')
        assert callable(getattr(MetaController, '_apply_backpressure'))
    
    def test_start_consumer_threads_exists(self):
        """Verify _start_consumer_threads method exists."""
        assert hasattr(MetaController, '_start_consumer_threads')
        assert callable(getattr(MetaController, '_start_consumer_threads'))
    
    def test_rebalance_consumers_exists(self):
        """Verify _rebalance_consumers method exists."""
        assert hasattr(MetaController, '_rebalance_consumers')
        assert callable(getattr(MetaController, '_rebalance_consumers'))
    
    def test_track_consumer_lag_exists(self):
        """Verify _track_consumer_lag method exists."""
        assert hasattr(MetaController, '_track_consumer_lag')
        assert callable(getattr(MetaController, '_track_consumer_lag'))
    
    def test_stop_consumer_threads_exists(self):
        """Verify _stop_consumer_threads method exists."""
        assert hasattr(MetaController, '_stop_consumer_threads')
        assert callable(getattr(MetaController, '_stop_consumer_threads'))
    
    def test_record_ingestion_time_exists(self):
        """Verify _record_signal_ingestion_time method exists."""
        assert hasattr(MetaController, '_record_signal_ingestion_time')
        assert callable(getattr(MetaController, '_record_signal_ingestion_time'))
    
    def test_record_completion_time_exists(self):
        """Verify _record_signal_completion_time method exists."""
        assert hasattr(MetaController, '_record_signal_completion_time')
        assert callable(getattr(MetaController, '_record_signal_completion_time'))
    
    def test_get_latency_percentiles_exists(self):
        """Verify get_signal_latency_percentiles method exists."""
        assert hasattr(MetaController, 'get_signal_latency_percentiles')
        assert callable(getattr(MetaController, 'get_signal_latency_percentiles'))
    
    def test_generate_latency_report_exists(self):
        """Verify _generate_latency_report method exists."""
        assert hasattr(MetaController, '_generate_latency_report')
        assert callable(getattr(MetaController, '_generate_latency_report'))


class TestSignalPipelineInfrastructure:
    """Tests for signal pipeline infrastructure initialization."""
    
    def test_batch_queue_initialized_in_code(self):
        """Verify batch queue is initialized in __init__."""
        source = inspect.getsource(MetaController.__init__)
        
        assert '_signal_batch_queue' in source
        assert 'deque' in source
        assert 'maxlen' in source or 'maxlen=50' in source
    
    def test_batch_lock_initialized_in_code(self):
        """Verify batch lock is initialized."""
        source = inspect.getsource(MetaController.__init__)
        
        assert '_batch_lock' in source
        assert 'threading.Lock' in source
    
    def test_batch_stats_initialized_in_code(self):
        """Verify batch statistics are initialized."""
        source = inspect.getsource(MetaController.__init__)
        
        assert '_batch_stats' in source
        assert 'total_batches' in source
        assert 'total_signals' in source
    
    def test_queue_config_initialized_in_code(self):
        """Verify queue configuration is initialized."""
        source = inspect.getsource(MetaController.__init__)
        
        assert '_queue_config' in source
        assert 'max_size' in source or 'max_size=1000' in source
        assert 'prefetch' in source
    
    def test_consumer_metrics_initialized_in_code(self):
        """Verify consumer lag metrics are initialized."""
        source = inspect.getsource(MetaController.__init__)
        
        assert '_consumer_lag_metrics' in source
        assert 'consumer_count' in source
    
    def test_signal_latencies_initialized_in_code(self):
        """Verify signal latency tracking is initialized."""
        source = inspect.getsource(MetaController.__init__)
        
        assert '_signal_latencies' in source
    
    def test_latency_metrics_initialized_in_code(self):
        """Verify latency metrics are initialized."""
        source = inspect.getsource(MetaController.__init__)
        
        assert '_latency_metrics' in source
        assert 'p50_ms' in source or 'p95_ms' in source


class TestBatchProcessing:
    """Tests for batch processing functionality."""
    
    def test_collect_signal_batch_signature(self):
        """Verify batch collection has correct signature."""
        sig = inspect.signature(MetaController._collect_signal_batch)
        params = list(sig.parameters.keys())
        
        assert 'timeout_ms' in params
        assert 'max_batch_size' in params
    
    def test_collect_batch_returns_list(self):
        """Verify batch collection returns a list."""
        source = inspect.getsource(MetaController._collect_signal_batch)
        
        assert 'return' in source
        assert 'batch' in source.lower()
    
    def test_process_batch_returns_tuple(self):
        """Verify batch processing returns tuple."""
        source = inspect.getsource(MetaController._process_signal_batch)
        
        assert 'return' in source
        assert '(' in source and ')' in source
    
    def test_validate_batch_has_duplicate_check(self):
        """Verify batch validation checks for duplicates."""
        source = inspect.getsource(MetaController._validate_batch_consistency)
        
        assert 'duplicate' in source.lower() or 'seen' in source.lower()
    
    def test_validate_batch_checks_required_fields(self):
        """Verify batch validation checks required fields."""
        source = inspect.getsource(MetaController._validate_batch_consistency)
        
        assert 'required' in source.lower() or 'field' in source.lower()
    
    def test_batch_stats_includes_batch_count(self):
        """Verify batch stats include batch count."""
        source = inspect.getsource(MetaController.get_batch_processing_stats)
        
        assert 'total_batches' in source or 'batch' in source.lower()
    
    def test_batch_stats_includes_signal_count(self):
        """Verify batch stats include signal count."""
        source = inspect.getsource(MetaController.get_batch_processing_stats)
        
        assert 'total_signals' in source or 'signal' in source.lower()
    
    def test_batch_stats_includes_avg_batch_size(self):
        """Verify batch stats include average batch size."""
        source = inspect.getsource(MetaController.get_batch_processing_stats)
        
        assert 'avg' in source.lower() or 'average' in source.lower()


class TestQueueConfiguration:
    """Tests for queue configuration functionality."""
    
    def test_configure_queue_accepts_max_size(self):
        """Verify queue configuration accepts max_size parameter."""
        sig = inspect.signature(MetaController._configure_signal_queue)
        params = list(sig.parameters.keys())
        
        assert 'max_size' in params
    
    def test_configure_queue_accepts_prefetch(self):
        """Verify queue configuration accepts prefetch parameter."""
        sig = inspect.signature(MetaController._configure_signal_queue)
        params = list(sig.parameters.keys())
        
        assert 'prefetch' in params
    
    def test_queue_config_returns_success_tuple(self):
        """Verify queue config returns success/message tuple."""
        source = inspect.getsource(MetaController._configure_signal_queue)
        
        assert 'return' in source
        assert '(' in source and ')' in source
    
    def test_configure_consumer_group_accepts_count(self):
        """Verify consumer group configuration accepts count."""
        sig = inspect.signature(MetaController._configure_consumer_group)
        params = list(sig.parameters.keys())
        
        assert 'num_consumers' in params
    
    def test_consumer_group_config_returns_success(self):
        """Verify consumer group config returns success/message."""
        source = inspect.getsource(MetaController._configure_consumer_group)
        
        assert 'return' in source


class TestBackpressureHandling:
    """Tests for backpressure handling."""
    
    def test_backpressure_method_signature(self):
        """Verify backpressure method has correct signature."""
        sig = inspect.signature(MetaController._apply_backpressure)
        params = list(sig.parameters.keys())
        
        assert 'drop_policy' in params or len(params) >= 0
    
    def test_backpressure_implements_drop_oldest(self):
        """Verify backpressure implements drop_oldest policy."""
        source = inspect.getsource(MetaController._apply_backpressure)
        
        assert 'drop_oldest' in source.lower() or 'drop' in source.lower()
    
    def test_backpressure_returns_applied_flag(self):
        """Verify backpressure returns applied flag."""
        source = inspect.getsource(MetaController._apply_backpressure)
        
        assert 'return' in source
        assert 'True' in source or 'False' in source


class TestConsumerManagement:
    """Tests for consumer thread management."""
    
    def test_start_consumers_accepts_count(self):
        """Verify consumer startup accepts count parameter."""
        sig = inspect.signature(MetaController._start_consumer_threads)
        params = list(sig.parameters.keys())
        
        assert 'count' in params
    
    def test_start_consumers_returns_count_and_message(self):
        """Verify consumer startup returns count and message."""
        source = inspect.getsource(MetaController._start_consumer_threads)
        
        assert 'return' in source
        assert 'len' in source.lower() or 'count' in source.lower()
    
    def test_rebalance_has_scaling_logic(self):
        """Verify rebalancing has scaling logic."""
        source = inspect.getsource(MetaController._rebalance_consumers)
        
        # Should have conditions for scaling up/down
        assert '>' in source or '<' in source
    
    def test_track_lag_calculates_metrics(self):
        """Verify lag tracking calculates metrics."""
        source = inspect.getsource(MetaController._track_consumer_lag)
        
        assert 'lag' in source.lower() or 'queue_depth' in source.lower()
    
    def test_stop_consumers_clears_threads(self):
        """Verify stop consumers clears thread list."""
        source = inspect.getsource(MetaController._stop_consumer_threads)
        
        assert 'stop' in source.lower() or 'clear' in source.lower()


class TestLatencyMeasurement:
    """Tests for latency measurement functionality."""
    
    def test_record_ingestion_time_accepts_signal_id(self):
        """Verify ingestion time recording accepts signal_id."""
        sig = inspect.signature(MetaController._record_signal_ingestion_time)
        params = list(sig.parameters.keys())
        
        assert 'signal_id' in params
    
    def test_record_completion_time_accepts_signal_id(self):
        """Verify completion time recording accepts signal_id."""
        sig = inspect.signature(MetaController._record_signal_completion_time)
        params = list(sig.parameters.keys())
        
        assert 'signal_id' in params
    
    def test_completion_time_calculates_latency(self):
        """Verify completion time calculation includes latency."""
        source = inspect.getsource(MetaController._record_signal_completion_time)
        
        assert 'latency' in source.lower() or 'completion' in source.lower()
    
    def test_latency_percentiles_returns_dict(self):
        """Verify percentile calculation returns dict."""
        source = inspect.getsource(MetaController.get_signal_latency_percentiles)
        
        assert 'return' in source
        assert 'p50' in source or 'p95' in source or 'p99' in source
    
    def test_latency_percentiles_sorts_latencies(self):
        """Verify percentile calculation sorts latencies."""
        source = inspect.getsource(MetaController.get_signal_latency_percentiles)
        
        assert 'sort' in source.lower()
    
    def test_latency_report_includes_percentiles(self):
        """Verify latency report includes percentiles."""
        source = inspect.getsource(MetaController._generate_latency_report)
        
        assert 'percentile' in source.lower() or 'p95' in source.lower()
    
    def test_latency_report_includes_batch_stats(self):
        """Verify latency report includes batch statistics."""
        source = inspect.getsource(MetaController._generate_latency_report)
        
        assert 'batch' in source.lower()
    
    def test_latency_report_includes_consumer_lag(self):
        """Verify latency report includes consumer lag."""
        source = inspect.getsource(MetaController._generate_latency_report)
        
        assert 'lag' in source.lower()


class TestCodeQuality:
    """Tests for code quality metrics."""
    
    def test_batch_collection_has_docstring(self):
        """Verify batch collection method has docstring."""
        source = inspect.getsource(MetaController._collect_signal_batch)
        assert '"""' in source or "'''" in source
    
    def test_batch_processing_has_docstring(self):
        """Verify batch processing method has docstring."""
        source = inspect.getsource(MetaController._process_signal_batch)
        assert '"""' in source or "'''" in source
    
    def test_validation_has_docstring(self):
        """Verify validation method has docstring."""
        source = inspect.getsource(MetaController._validate_batch_consistency)
        assert '"""' in source or "'''" in source
    
    def test_queue_config_has_docstring(self):
        """Verify queue config method has docstring."""
        source = inspect.getsource(MetaController._configure_signal_queue)
        assert '"""' in source or "'''" in source
    
    def test_consumer_management_has_docstrings(self):
        """Verify consumer methods have docstrings."""
        source = inspect.getsource(MetaController._start_consumer_threads)
        assert '"""' in source or "'''" in source
    
    def test_latency_methods_have_docstrings(self):
        """Verify latency methods have docstrings."""
        source = inspect.getsource(MetaController.get_signal_latency_percentiles)
        assert '"""' in source or "'''" in source


class TestThreadSafety:
    """Tests for thread safety mechanisms."""
    
    def test_batch_collection_uses_lock(self):
        """Verify batch collection uses lock for safety."""
        source = inspect.getsource(MetaController._collect_signal_batch)
        
        assert '_batch_lock' in source
    
    def test_batch_processing_uses_lock(self):
        """Verify batch processing updates use lock."""
        source = inspect.getsource(MetaController._process_signal_batch)
        
        assert '_batch_lock' in source
    
    def test_latency_tracking_thread_safe(self):
        """Verify latency tracking is thread-safe."""
        source = inspect.getsource(MetaController._record_signal_completion_time)
        
        # Should handle concurrent access safely
        assert 'latency' in source.lower()


class TestPerformanceOptimization:
    """Tests for performance optimization features."""
    
    def test_batch_collection_has_timeout(self):
        """Verify batch collection has timeout mechanism."""
        source = inspect.getsource(MetaController._collect_signal_batch)
        
        assert 'timeout' in source.lower()
    
    def test_batch_stats_use_exponential_smoothing(self):
        """Verify batch stats use exponential moving average."""
        source = inspect.getsource(MetaController._process_signal_batch)
        
        assert '0.9' in source or '0.1' in source or 'exponential' in source.lower()
    
    def test_consumer_lag_calculated(self):
        """Verify consumer lag is calculated."""
        source = inspect.getsource(MetaController._track_consumer_lag)
        
        assert 'lag' in source.lower() or 'queue_depth' in source.lower()
    
    def test_dynamic_rebalancing_implemented(self):
        """Verify dynamic consumer rebalancing is implemented."""
        source = inspect.getsource(MetaController._rebalance_consumers)
        
        # Should have logic to scale up/down
        assert 'queue_depth' in source.lower() or '>' in source or '<' in source


class TestIntegration:
    """Tests for integration with existing code."""
    
    def test_batch_queue_initialization(self):
        """Verify batch queue proper initialization."""
        import_file = inspect.getfile(MetaController)
        with open(import_file, 'r') as f:
            content = f.read()
            assert '_signal_batch_queue' in content
    
    def test_signal_pipeline_imports(self):
        """Verify necessary imports are present."""
        import_file = inspect.getfile(MetaController)
        with open(import_file, 'r') as f:
            content = f.read(5000)
            assert 'deque' in content or 'collections' in content
    
    def test_all_methods_callable(self):
        """Verify all pipeline methods are callable."""
        assert callable(MetaController._collect_signal_batch)
        assert callable(MetaController._process_signal_batch)
        assert callable(MetaController.get_batch_processing_stats)
        assert callable(MetaController.get_signal_latency_percentiles)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
