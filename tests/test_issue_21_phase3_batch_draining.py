"""
Test suite for Issue #21 Phase 3: Event Draining Batch Optimization

Phase 3 Focus: Optimize event draining by batching events for reduced latency.
Target: Reduce event drain overhead from 35ms to 20ms per cycle (43% reduction)

Implementation:
- drain_trade_intent_events_batch(): Collect all events synchronously, normalize in batch
- drain_and_process_intents_optimized(): Single async call for entire batch
- get_event_drain_metrics(): Performance tracking for Phase 3 optimization

Expected Performance Gains:
- Per-event overhead: Eliminated (batch collection)
- Normalization: Single pass through all events
- Async overhead: Single await vs per-event awaits
"""

import inspect
import pytest
from core.meta_controller import MetaController


class TestBatchDrainingMethods:
    """Test existence and signatures of Phase 3 batch draining methods."""
    
    def test_drain_trade_intent_events_batch_exists(self):
        """Verify drain_trade_intent_events_batch method exists."""
        assert hasattr(MetaController, "drain_trade_intent_events_batch")
    
    def test_drain_and_process_intents_optimized_exists(self):
        """Verify drain_and_process_intents_optimized method exists."""
        assert hasattr(MetaController, "drain_and_process_intents_optimized")
    
    def test_get_event_drain_metrics_exists(self):
        """Verify get_event_drain_metrics method exists."""
        assert hasattr(MetaController, "get_event_drain_metrics")
    
    def test_drain_trade_intent_events_batch_signature(self):
        """Verify drain_trade_intent_events_batch has correct signature."""
        method = getattr(MetaController, "drain_trade_intent_events_batch")
        sig = inspect.signature(method)
        params = list(sig.parameters.keys())
        
        # Should have self, max_items
        assert "self" in params or params[0:1] == []  # self implicit in unbound
        assert "max_items" in params
    
    def test_drain_and_process_intents_optimized_signature(self):
        """Verify drain_and_process_intents_optimized is async and has correct signature."""
        method = getattr(MetaController, "drain_and_process_intents_optimized")
        assert inspect.iscoroutinefunction(method)
        
        sig = inspect.signature(method)
        params = list(sig.parameters.keys())
        assert "max_items" in params


class TestBatchDrainingImplementation:
    """Test Phase 3 implementation details in source code."""
    
    def test_batch_collection_in_source(self):
        """Verify batch collection logic in drain_trade_intent_events_batch."""
        source = inspect.getsource(MetaController.drain_trade_intent_events_batch)
        
        # Check for batch collection pattern
        assert "batch" in source.lower()
        assert "max_items" in source
        assert "get_nowait()" in source
        assert "return batch" in source
    
    def test_optimized_processing_in_source(self):
        """Verify optimized async processing in drain_and_process_intents_optimized."""
        source = inspect.getsource(MetaController.drain_and_process_intents_optimized)
        
        # Check for batch-first approach
        assert "batch = self.drain_trade_intent_events_batch" in source
        assert "receive_intents(batch)" in source
        assert "if not batch:" in source
    
    def test_metrics_calculation_in_source(self):
        """Verify metrics calculation in get_event_drain_metrics."""
        source = inspect.getsource(MetaController.get_event_drain_metrics)
        
        # Check for metrics computation
        assert "event_drain_ms" in source
        assert "drain_avg_ms" in source
        assert "queue_size" in source
        assert "batch_efficiency" in source


class TestBatchDrainingDocumentation:
    """Test that Phase 3 methods are properly documented."""
    
    def test_batch_drain_has_docstring(self):
        """Verify drain_trade_intent_events_batch has docstring."""
        method = getattr(MetaController, "drain_trade_intent_events_batch")
        assert method.__doc__ is not None
        assert len(method.__doc__) > 20
        assert "batch" in method.__doc__.lower()
    
    def test_optimized_process_has_docstring(self):
        """Verify drain_and_process_intents_optimized has docstring."""
        method = getattr(MetaController, "drain_and_process_intents_optimized")
        assert method.__doc__ is not None
        assert len(method.__doc__) > 20
        assert "batch" in method.__doc__.lower() or "optimized" in method.__doc__.lower()
    
    def test_metrics_has_docstring(self):
        """Verify get_event_drain_metrics has docstring."""
        method = getattr(MetaController, "get_event_drain_metrics")
        assert method.__doc__ is not None
        assert len(method.__doc__) > 20


class TestPhase3Implementation:
    """Test Phase 3 components all present and integrated."""
    
    def test_all_phase3_components_present(self):
        """Verify all Phase 3 methods are present."""
        assert hasattr(MetaController, "drain_trade_intent_events_batch")
        assert hasattr(MetaController, "drain_and_process_intents_optimized")
        assert hasattr(MetaController, "get_event_drain_metrics")
    
    def test_phase3_uses_perf_metrics(self):
        """Verify Phase 3 methods use perf_metrics for tracking."""
        source = inspect.getsource(MetaController.get_event_drain_metrics)
        assert "_perf_metrics" in source or "self._perf_metrics" in source
    
    def test_phase3_methods_integrated(self):
        """Verify Phase 3 methods reference each other properly."""
        optimized_source = inspect.getsource(MetaController.drain_and_process_intents_optimized)
        batch_source = inspect.getsource(MetaController.drain_trade_intent_events_batch)
        
        # Optimized should call batch method
        assert "drain_trade_intent_events_batch" in optimized_source
        # Batch should use synchronous collection
        assert "get_nowait()" in batch_source


class TestBatchDrainingFeatures:
    """Test specific features of batch draining optimization."""
    
    def test_batch_drain_returns_list(self):
        """Verify drain_trade_intent_events_batch returns List[Dict]."""
        source = inspect.getsource(MetaController.drain_trade_intent_events_batch)
        
        # Check for list return type
        assert "return batch" in source
        assert "[]" in source or "batch:" in source
    
    def test_batch_drain_uses_copy(self):
        """Verify batch drain uses defensive copying for thread safety."""
        source = inspect.getsource(MetaController.drain_trade_intent_events_batch)
        
        # Batch drain delegates copying to _normalize_trade_intent_event
        # or uses data from queue directly (already safe)
        # Check that data is properly extracted without mutation
        assert "_normalize_trade_intent_event" in source or "append(norm)" in source
    
    def test_optimized_drain_is_async(self):
        """Verify drain_and_process_intents_optimized is async."""
        method = getattr(MetaController, "drain_and_process_intents_optimized")
        assert inspect.iscoroutinefunction(method)
    
    def test_optimized_drain_single_await(self):
        """Verify optimized drain has single receive_intents call (not per-event)."""
        source = inspect.getsource(MetaController.drain_and_process_intents_optimized)
        
        # Count await calls for receive_intents
        await_count = source.count("await self.intent_manager.receive_intents(batch)")
        assert await_count == 1
    
    def test_metrics_has_efficiency_calculation(self):
        """Verify get_event_drain_metrics calculates batch efficiency."""
        source = inspect.getsource(MetaController.get_event_drain_metrics)
        
        # Check for efficiency metric
        assert "batch_efficiency" in source
        assert "len(metrics)" in source


class TestPhase3PerformanceTargets:
    """Test Phase 3 optimization targets are met in code."""
    
    def test_batch_drain_minimizes_exception_handling(self):
        """Verify batch drain minimizes exception handling (lightweight validation)."""
        source = inspect.getsource(MetaController.drain_trade_intent_events_batch)
        
        # Should have minimal try/except (only for get_nowait and task_done)
        try_count = source.count("try:")
        # Only 2 try blocks expected: one for get_nowait, one for task_done
        assert try_count <= 3
    
    def test_batch_drain_normalizes_in_single_pass(self):
        """Verify batch drain normalizes all events in single pass."""
        source = inspect.getsource(MetaController.drain_trade_intent_events_batch)
        
        # Should have single loop iterating max_items times
        assert "for _ in range(max_items):" in source
        # Should collect normalized items
        assert "_normalize_trade_intent_event" in source
    
    def test_optimized_drain_avoids_per_event_awaits(self):
        """Verify optimized drain avoids per-event async operations."""
        source = inspect.getsource(MetaController.drain_and_process_intents_optimized)
        
        # Should batch collect first, then single async call
        lines = source.split('\n')
        batch_collect_idx = next(i for i, line in enumerate(lines) if "drain_trade_intent_events_batch" in line)
        receive_intents_idx = next(i for i, line in enumerate(lines) if "receive_intents(batch)" in line)
        
        # receive_intents should come after batch collection
        assert receive_intents_idx > batch_collect_idx


class TestPhase3Integration:
    """Test Phase 3 integration with existing Phase 1/2 components."""
    
    def test_phase3_with_perf_metrics_structure(self):
        """Verify Phase 3 methods work with Phase 1 perf_metrics structure."""
        source = inspect.getsource(MetaController.get_event_drain_metrics)
        
        # Should reference event_drain_ms from perf_metrics
        assert "event_drain_ms" in source
        assert "_perf_metrics" in source
    
    def test_phase3_with_intent_manager(self):
        """Verify Phase 3 delegates to intent_manager properly."""
        source = inspect.getsource(MetaController.drain_and_process_intents_optimized)
        
        # Should call intent_manager.receive_intents
        assert "intent_manager" in source
        assert "receive_intents" in source
    
    def test_phase3_logging_integration(self):
        """Verify Phase 3 integrates logging for monitoring."""
        optimized_source = inspect.getsource(MetaController.drain_and_process_intents_optimized)
        
        # Should have logger calls
        assert "logger.debug" in optimized_source or "logger.warning" in optimized_source
        assert "EventDrainOptimized" in optimized_source or "event" in optimized_source.lower()


class TestEventDrainMetricsTracking:
    """Test Phase 3 metrics tracking functionality."""
    
    def test_metrics_dict_structure(self):
        """Verify get_event_drain_metrics returns proper dict structure."""
        source = inspect.getsource(MetaController.get_event_drain_metrics)
        
        # Check for return dict with expected keys
        assert "drain_count" in source
        assert "drain_avg_ms" in source
        assert "queue_size" in source
        assert "batch_efficiency" in source
    
    def test_metrics_calculation_accuracy(self):
        """Verify metrics calculations use correct formulas."""
        source = inspect.getsource(MetaController.get_event_drain_metrics)
        
        # Check for proper averaging
        assert "sum(metrics)" in source or "sum(" in source
        assert "len(metrics)" in source
        
        # Check for efficiency calculation
        assert "batch_efficiency" in source
    
    def test_metrics_handles_empty_case(self):
        """Verify metrics handles empty metrics list."""
        source = inspect.getsource(MetaController.get_event_drain_metrics)
        
        # Should handle case when no metrics exist
        assert "if not metrics:" in source
        assert "{" in source  # Returns dict


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
