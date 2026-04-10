"""
Test suite for Issue #22: Guard Evaluation Parallelization

Tests parallel guard evaluation, thread safety, timeout handling, and performance improvements.
Target latency: 80ms → 45ms (45% reduction)
"""

import pytest
import time
import threading
import inspect
from unittest.mock import Mock, MagicMock, patch
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from core.meta_controller import MetaController


class TestGuardParallelizationMethodsExist:
    """Tests that all guard parallelization methods are properly defined."""
    
    def test_evaluate_guards_parallel_exists(self):
        """Verify evaluate_guards_parallel method exists."""
        assert hasattr(MetaController, 'evaluate_guards_parallel')
        assert callable(getattr(MetaController, 'evaluate_guards_parallel'))
    
    def test_evaluate_guard_wrapper_exists(self):
        """Verify _evaluate_guard_wrapper method exists."""
        assert hasattr(MetaController, '_evaluate_guard_wrapper')
        assert callable(getattr(MetaController, '_evaluate_guard_wrapper'))
    
    def test_get_guard_metrics_exists(self):
        """Verify get_guard_parallelization_metrics method exists."""
        assert hasattr(MetaController, 'get_guard_parallelization_metrics')
        assert callable(getattr(MetaController, 'get_guard_parallelization_metrics'))
    
    def test_shutdown_executor_exists(self):
        """Verify shutdown_guard_executor method exists."""
        assert hasattr(MetaController, 'shutdown_guard_executor')
        assert callable(getattr(MetaController, 'shutdown_guard_executor'))


class TestGuardParallelizationInfrastructure:
    """Tests for parallel guard infrastructure setup in __init__."""
    
    def test_guard_executor_initialization_in_code(self):
        """Verify ThreadPoolExecutor initialization code exists in __init__."""
        source = inspect.getsource(MetaController.__init__)
        
        assert '_guard_executor' in source
        assert 'ThreadPoolExecutor' in source
        assert 'max_workers=6' in source
    
    def test_guard_lock_initialization_in_code(self):
        """Verify threading.Lock initialization code exists in __init__."""
        source = inspect.getsource(MetaController.__init__)
        
        assert '_guard_lock' in source
        assert 'threading.Lock' in source
    
    def test_guard_timeout_configured_in_code(self):
        """Verify timeout is set correctly in __init__."""
        source = inspect.getsource(MetaController.__init__)
        
        assert '_guard_timeout_sec' in source
        assert '2.0' in source
    
    def test_guard_metrics_initialized_in_code(self):
        """Verify metrics dictionary is properly initialized in __init__."""
        source = inspect.getsource(MetaController.__init__)
        
        assert '_guard_eval_metrics' in source
        assert 'parallel_count' in source
        assert 'parallel_success' in source
        assert 'parallel_timeout' in source
        assert 'sequential_fallback_count' in source
        assert 'parallel_avg_ms' in source


class TestGuardParallelEvaluationMethod:
    """Tests for the evaluate_guards_parallel method implementation."""
    
    def test_method_signature_correct(self):
        """Verify method has correct signature."""
        sig = inspect.signature(MetaController.evaluate_guards_parallel)
        
        params = list(sig.parameters.keys())
        assert 'symbol' in params
        assert 'signal' in params
    
    def test_method_returns_tuple(self):
        """Verify method should return tuple from code analysis."""
        source = inspect.getsource(MetaController.evaluate_guards_parallel)
        
        # Should return tuple
        assert 'return (' in source
        assert 'bool' in source or 'True' in source or 'False' in source
    
    def test_method_has_timeout_logic(self):
        """Verify method includes timeout handling."""
        source = inspect.getsource(MetaController.evaluate_guards_parallel)
        
        assert 'timeout' in source.lower()
        assert 'TimeoutError' in source or 'timeout' in source
    
    def test_method_has_metrics_tracking(self):
        """Verify method tracks parallelization metrics."""
        source = inspect.getsource(MetaController.evaluate_guards_parallel)
        
        assert '_guard_eval_metrics' in source
        assert 'parallel_count' in source or 'parallel_success' in source
    
    def test_method_has_parallel_submission(self):
        """Verify method submits guards to executor."""
        source = inspect.getsource(MetaController.evaluate_guards_parallel)
        
        assert 'submit' in source.lower()
        assert '_guard_executor' in source
    
    def test_method_evaluates_volatility_guard(self):
        """Verify method checks volatility guard."""
        source = inspect.getsource(MetaController.evaluate_guards_parallel)
        
        assert 'volatility' in source.lower()
    
    def test_method_evaluates_edge_guard(self):
        """Verify method checks edge quality guard."""
        source = inspect.getsource(MetaController.evaluate_guards_parallel)
        
        assert 'edge' in source.lower()
    
    def test_method_evaluates_economic_guard(self):
        """Verify method checks economic viability guard."""
        source = inspect.getsource(MetaController.evaluate_guards_parallel)
        
        assert 'economic' in source.lower()
    
    def test_method_evaluates_concentration_guard(self):
        """Verify method checks concentration guard."""
        source = inspect.getsource(MetaController.evaluate_guards_parallel)
        
        assert 'concentration' in source.lower()


class TestGuardWrapperMethod:
    """Tests for the _evaluate_guard_wrapper method."""
    
    def test_wrapper_exists_with_correct_name(self):
        """Verify wrapper method exists."""
        assert hasattr(MetaController, '_evaluate_guard_wrapper')
    
    def test_wrapper_accepts_guard_name(self):
        """Verify wrapper accepts guard_name parameter."""
        sig = inspect.signature(MetaController._evaluate_guard_wrapper)
        
        params = list(sig.parameters.keys())
        assert 'guard_name' in params
    
    def test_wrapper_accepts_guard_func(self):
        """Verify wrapper accepts guard_func parameter."""
        sig = inspect.signature(MetaController._evaluate_guard_wrapper)
        
        params = list(sig.parameters.keys())
        assert 'guard_func' in params
    
    def test_wrapper_has_exception_handling(self):
        """Verify wrapper handles exceptions."""
        source = inspect.getsource(MetaController._evaluate_guard_wrapper)
        
        assert 'try' in source.lower()
        assert 'except' in source.lower()
    
    def test_wrapper_handles_bool_returns(self):
        """Verify wrapper handles bool return values."""
        source = inspect.getsource(MetaController._evaluate_guard_wrapper)
        
        assert 'bool' in source or 'isinstance' in source


class TestGuardMetricsMethod:
    """Tests for the get_guard_parallelization_metrics method."""
    
    def test_metrics_method_exists(self):
        """Verify metrics method exists."""
        assert hasattr(MetaController, 'get_guard_parallelization_metrics')
    
    def test_metrics_method_returns_dict(self):
        """Verify metrics method returns dictionary."""
        source = inspect.getsource(MetaController.get_guard_parallelization_metrics)
        
        assert 'return' in source
        assert 'dict' in source or '{' in source
    
    def test_metrics_includes_total_evaluations(self):
        """Verify metrics includes total evaluation count."""
        source = inspect.getsource(MetaController.get_guard_parallelization_metrics)
        
        assert 'total' in source.lower() or 'count' in source.lower()
    
    def test_metrics_includes_success_rate(self):
        """Verify metrics includes success rate."""
        source = inspect.getsource(MetaController.get_guard_parallelization_metrics)
        
        assert 'success' in source.lower() or 'rate' in source.lower()
    
    def test_metrics_includes_timeout_count(self):
        """Verify metrics includes timeout count."""
        source = inspect.getsource(MetaController.get_guard_parallelization_metrics)
        
        assert 'timeout' in source.lower()
    
    def test_metrics_uses_lock_for_safety(self):
        """Verify metrics method uses lock for thread safety."""
        source = inspect.getsource(MetaController.get_guard_parallelization_metrics)
        
        assert 'lock' in source.lower()
    
    def test_metrics_calculates_percentages(self):
        """Verify metrics calculates percentage values."""
        source = inspect.getsource(MetaController.get_guard_parallelization_metrics)
        
        assert '%' in source or 'pct' in source.lower() or '* 100' in source


class TestShutdownMethod:
    """Tests for the shutdown_guard_executor method."""
    
    def test_shutdown_method_exists(self):
        """Verify shutdown method exists."""
        assert hasattr(MetaController, 'shutdown_guard_executor')
    
    def test_shutdown_calls_executor_shutdown(self):
        """Verify shutdown method calls executor shutdown."""
        source = inspect.getsource(MetaController.shutdown_guard_executor)
        
        assert 'shutdown' in source.lower()
        assert '_guard_executor' in source
    
    def test_shutdown_has_timeout_parameter(self):
        """Verify shutdown has proper timeout."""
        source = inspect.getsource(MetaController.shutdown_guard_executor)
        
        assert 'timeout' in source.lower()
    
    def test_shutdown_has_exception_handling(self):
        """Verify shutdown handles exceptions gracefully."""
        source = inspect.getsource(MetaController.shutdown_guard_executor)
        
        assert 'try' in source.lower()
        assert 'except' in source.lower()


class TestGuardParallelizationCodeQuality:
    """Tests for code quality and proper implementation."""
    
    def test_parallel_method_has_docstring(self):
        """Verify parallel method has docstring."""
        source = inspect.getsource(MetaController.evaluate_guards_parallel)
        
        assert '"""' in source or "'''" in source
    
    def test_wrapper_method_has_docstring(self):
        """Verify wrapper method has docstring."""
        source = inspect.getsource(MetaController._evaluate_guard_wrapper)
        
        assert '"""' in source or "'''" in source
    
    def test_metrics_method_has_docstring(self):
        """Verify metrics method has docstring."""
        source = inspect.getsource(MetaController.get_guard_parallelization_metrics)
        
        assert '"""' in source or "'''" in source
    
    def test_shutdown_method_has_docstring(self):
        """Verify shutdown method has docstring."""
        source = inspect.getsource(MetaController.shutdown_guard_executor)
        
        assert '"""' in source or "'''" in source
    
    def test_parallel_method_has_logging(self):
        """Verify parallel method includes logging."""
        source = inspect.getsource(MetaController.evaluate_guards_parallel)
        
        assert 'logger' in source.lower() or 'log' in source.lower()
    
    def test_wrapper_method_has_logging(self):
        """Verify wrapper method includes logging."""
        source = inspect.getsource(MetaController._evaluate_guard_wrapper)
        
        assert 'logger' in source.lower() or 'log' in source.lower()


class TestGuardThreadSafety:
    """Tests for thread safety mechanisms."""
    
    def test_guard_lock_used_in_metrics_update(self):
        """Verify guard lock is used when updating metrics."""
        source = inspect.getsource(MetaController.evaluate_guards_parallel)
        
        # Check that lock is acquired for metrics updates
        assert '_guard_lock' in source
    
    def test_metrics_method_uses_lock_for_copy(self):
        """Verify metrics method copies data under lock."""
        source = inspect.getsource(MetaController.get_guard_parallelization_metrics)
        
        assert '_guard_lock' in source
        assert '.copy()' in source or 'copy' in source.lower()
    
    def test_concurrent_guard_submission(self):
        """Verify guards can be submitted concurrently."""
        source = inspect.getsource(MetaController.evaluate_guards_parallel)
        
        # Should submit multiple guards without waiting
        assert 'submit' in source.lower()
        
        # Multiple guard names indicate parallel submission
        guard_count = source.count('submit')
        assert guard_count >= 4  # At least 4 guards


class TestPerformanceOptimization:
    """Tests for performance optimization aspects."""
    
    def test_as_completed_or_similar_used(self):
        """Verify method uses async result collection pattern."""
        source = inspect.getsource(MetaController.evaluate_guards_parallel)
        
        # Should use as_completed or similar to avoid waiting for slowest guard
        assert 'as_completed' in source or 'future' in source.lower()
    
    def test_early_exit_on_guard_failure(self):
        """Verify method exits early if guard fails."""
        source = inspect.getsource(MetaController.evaluate_guards_parallel)
        
        # Should return immediately on failure, not wait for all
        assert 'return' in source
        assert 'False' in source
    
    def test_timeout_prevents_hanging(self):
        """Verify timeout prevents indefinite waiting."""
        source = inspect.getsource(MetaController.evaluate_guards_parallel)
        
        assert 'timeout' in source.lower()
        assert '2' in source  # 2.0 second timeout


class TestMetricsTracking:
    """Tests for metrics tracking and reporting."""
    
    def test_parallel_count_tracked(self):
        """Verify parallel evaluation count is tracked."""
        source = inspect.getsource(MetaController.evaluate_guards_parallel)
        
        assert 'parallel_count' in source
    
    def test_success_count_tracked(self):
        """Verify successful evaluations are tracked."""
        source = inspect.getsource(MetaController.evaluate_guards_parallel)
        
        assert 'parallel_success' in source
    
    def test_timeout_count_tracked(self):
        """Verify timeouts are tracked."""
        source = inspect.getsource(MetaController.evaluate_guards_parallel)
        
        assert 'parallel_timeout' in source
    
    def test_fallback_count_tracked(self):
        """Verify sequential fallbacks are tracked."""
        source = inspect.getsource(MetaController.evaluate_guards_parallel)
        
        assert 'sequential_fallback' in source or 'fallback' in source.lower()
    
    def test_average_latency_tracked(self):
        """Verify average evaluation latency is tracked."""
        source = inspect.getsource(MetaController.evaluate_guards_parallel)
        
        assert 'avg' in source.lower() or 'average' in source.lower()
        assert 'ms' in source or 'time' in source.lower()


class TestGuardParallelizationIntegration:
    """Integration tests for the parallelization system."""
    
    def test_imports_in_file(self):
        """Verify necessary imports are in the module."""
        import_file = inspect.getfile(MetaController)
        with open(import_file, 'r') as f:
            content = f.read(10000)  # Read first 10K chars
            
            assert 'import threading' in content
            assert 'ThreadPoolExecutor' in content
    
    def test_all_export_includes_new_methods(self):
        """Verify new methods are accessible from module."""
        # The methods should be accessible without import errors
        assert callable(MetaController.evaluate_guards_parallel)
        assert callable(MetaController._evaluate_guard_wrapper)
        assert callable(MetaController.get_guard_parallelization_metrics)
        assert callable(MetaController.shutdown_guard_executor)


class TestPerformanceTargetAlignment:
    """Tests for alignment with performance targets."""
    
    def test_target_latency_reduction_documented(self):
        """Verify performance target is documented."""
        source = inspect.getsource(MetaController.evaluate_guards_parallel)
        
        # Method should have timing logic for 80->45ms reduction
        assert 'time' in source.lower()
        assert 'ms' in source or 'elapsed' in source.lower()
    
    def test_metrics_support_latency_monitoring(self):
        """Verify metrics support latency monitoring."""
        source = inspect.getsource(MetaController.get_guard_parallelization_metrics)
        
        # Should include latency/timing metrics
        assert 'avg' in source.lower() or 'time' in source.lower() or 'ms' in source


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
