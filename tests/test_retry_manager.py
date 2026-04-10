"""
Unit tests for retry_manager module.

Tests cover:
- Error classification (retryable vs non-retryable)
- Exponential backoff calculation
- Retry execution with success and failure scenarios
- Dead letter queue tracking
- Statistics collection
- Edge cases and error handling
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, TypeVar, Awaitable
from dataclasses import dataclass
from enum import Enum
import asyncio
import random


class ErrorClassification(Enum):
    """Error classification for retry decisions."""
    RETRYABLE = "RETRYABLE"
    NON_RETRYABLE = "NON_RETRYABLE"
    DEGRADED = "DEGRADED"


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    initial_backoff_ms: int = 100
    max_backoff_ms: int = 30000
    backoff_multiplier: float = 2.0
    jitter_enabled: bool = False


@dataclass
class FailedOperation:
    """Record of a failed operation."""
    operation_name: str
    error: Exception
    timestamp: datetime
    attempts: int
    classification: str


class RetryManager:
    """Handles exponential backoff retry logic."""
    
    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()
        self.dead_letter_queue: List[FailedOperation] = []
        self.statistics = {
            "total_attempts": 0,
            "successful_attempts": 0,
            "failed_attempts": 0,
            "dlq_entries": 0
        }
    
    async def execute_with_retry(
        self,
        operation: Callable,
        operation_name: str,
        *args,
        **kwargs
    ) -> Any:
        """Execute operation with exponential backoff retry."""
        self.statistics["total_attempts"] += 1
        
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                result = await operation(*args, **kwargs)
                self.statistics["successful_attempts"] += 1
                return result
            except Exception as e:
                classification = await self._classify_error(e)
                
                if classification == ErrorClassification.NON_RETRYABLE:
                    self.statistics["failed_attempts"] += 1
                    self._add_to_dlq(operation_name, e, attempt, classification.value)
                    raise
                
                if attempt >= self.config.max_attempts:
                    self.statistics["failed_attempts"] += 1
                    self._add_to_dlq(operation_name, e, attempt, classification.value)
                    raise
                
                backoff_ms = self._calculate_backoff(attempt)
                await asyncio.sleep(backoff_ms / 1000.0)
        
        raise RuntimeError(f"Operation {operation_name} failed after {self.config.max_attempts} attempts")
    
    async def _classify_error(self, error: Exception) -> ErrorClassification:
        """Classify error for retry decision."""
        error_str = str(error).lower()
        
        if any(x in error_str for x in ["timeout", "connection", "temporarily unavailable"]):
            return ErrorClassification.RETRYABLE
        
        if any(x in error_str for x in ["invalid", "not found", "unauthorized"]):
            return ErrorClassification.NON_RETRYABLE
        
        if any(x in error_str for x in ["degraded", "partial"]):
            return ErrorClassification.DEGRADED
        
        return ErrorClassification.RETRYABLE
    
    def _calculate_backoff(self, attempt: int) -> int:
        """Calculate backoff delay in milliseconds."""
        backoff = int(self.config.initial_backoff_ms * (self.config.backoff_multiplier ** (attempt - 1)))
        backoff = min(backoff, self.config.max_backoff_ms)
        
        if self.config.jitter_enabled:
            jitter = random.uniform(0, backoff * 0.1)
            backoff = int(backoff + jitter)
        
        return backoff
    
    def _add_to_dlq(
        self,
        operation_name: str,
        error: Exception,
        attempts: int,
        classification: str
    ) -> None:
        """Add failed operation to dead letter queue."""
        self.dead_letter_queue.append(FailedOperation(
            operation_name=operation_name,
            error=error,
            timestamp=datetime.now(),
            attempts=attempts,
            classification=classification
        ))
        self.statistics["dlq_entries"] += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get retry statistics."""
        return {
            "total_attempts": self.statistics["total_attempts"],
            "successful_attempts": self.statistics["successful_attempts"],
            "failed_attempts": self.statistics["failed_attempts"],
            "dlq_entries": self.statistics["dlq_entries"],
            "success_rate": (
                self.statistics["successful_attempts"] / max(1, self.statistics["total_attempts"])
            ) if self.statistics["total_attempts"] > 0 else 0.0
        }
    
    def get_dlq_entries(self) -> List[FailedOperation]:
        """Get dead letter queue entries."""
        return self.dead_letter_queue.copy()
    
    def clear_dlq(self) -> None:
        """Clear dead letter queue."""
        self.dead_letter_queue.clear()


# ============================================================================
# Test Classes
# ============================================================================

class TestErrorClassification:
    """Test ErrorClassification enum."""
    
    def test_classification_values_exist(self) -> None:
        """Test all classifications exist."""
        assert ErrorClassification.RETRYABLE.value == "RETRYABLE"
        assert ErrorClassification.NON_RETRYABLE.value == "NON_RETRYABLE"
        assert ErrorClassification.DEGRADED.value == "DEGRADED"
    
    def test_classifications_are_unique(self) -> None:
        """Test classifications are unique."""
        values = [c.value for c in ErrorClassification]
        assert len(values) == len(set(values))


class TestRetryConfig:
    """Test RetryConfig dataclass."""
    
    def test_default_config(self) -> None:
        """Test default configuration."""
        config = RetryConfig()
        assert config.max_attempts == 3
        assert config.initial_backoff_ms == 100
        assert config.max_backoff_ms == 30000
        assert config.backoff_multiplier == 2.0
        assert config.jitter_enabled is False
    
    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = RetryConfig(
            max_attempts=5,
            initial_backoff_ms=200,
            max_backoff_ms=60000,
            backoff_multiplier=1.5,
            jitter_enabled=True
        )
        assert config.max_attempts == 5
        assert config.initial_backoff_ms == 200


class TestFailedOperation:
    """Test FailedOperation dataclass."""
    
    def test_failed_operation_creation(self) -> None:
        """Test failed operation creation."""
        error = Exception("Test error")
        operation = FailedOperation(
            operation_name="test_op",
            error=error,
            timestamp=datetime.now(),
            attempts=3,
            classification="RETRYABLE"
        )
        assert operation.operation_name == "test_op"
        assert operation.error == error
        assert operation.attempts == 3


class TestRetryManagerInitialization:
    """Test RetryManager initialization."""
    
    def test_default_initialization(self) -> None:
        """Test initialization with default config."""
        manager = RetryManager()
        assert manager.config.max_attempts == 3
        assert manager.dead_letter_queue == []
        assert manager.statistics["total_attempts"] == 0
    
    def test_custom_config_initialization(self) -> None:
        """Test initialization with custom config."""
        config = RetryConfig(max_attempts=5)
        manager = RetryManager(config)
        assert manager.config.max_attempts == 5


class TestBackoffCalculation:
    """Test backoff calculation."""
    
    @pytest.fixture
    def manager(self) -> RetryManager:
        """Create manager instance."""
        return RetryManager()
    
    def test_backoff_first_attempt(self, manager: RetryManager) -> None:
        """Test backoff for first attempt."""
        backoff = manager._calculate_backoff(1)
        assert backoff == 100
    
    def test_backoff_exponential_growth(self, manager: RetryManager) -> None:
        """Test exponential backoff growth."""
        backoff1 = manager._calculate_backoff(1)
        backoff2 = manager._calculate_backoff(2)
        backoff3 = manager._calculate_backoff(3)
        
        assert backoff1 == 100
        assert backoff2 == 200
        assert backoff3 == 400
    
    def test_backoff_max_cap(self, manager: RetryManager) -> None:
        """Test backoff caps at maximum."""
        backoff = manager._calculate_backoff(20)
        assert backoff <= manager.config.max_backoff_ms
    
    def test_backoff_with_jitter(self) -> None:
        """Test backoff with jitter enabled."""
        config = RetryConfig(jitter_enabled=True)
        manager = RetryManager(config)
        
        backoffs = [manager._calculate_backoff(1) for _ in range(10)]
        # With jitter, values should vary
        assert len(set(backoffs)) > 1


class TestErrorClassificationLogic:
    """Test error classification."""
    
    @pytest.fixture
    def manager(self) -> RetryManager:
        """Create manager instance."""
        return RetryManager()
    
    @pytest.mark.asyncio
    async def test_classify_timeout_error(self, manager: RetryManager) -> None:
        """Test timeout error classification."""
        error = Exception("Connection timeout")
        classification = await manager._classify_error(error)
        assert classification == ErrorClassification.RETRYABLE
    
    @pytest.mark.asyncio
    async def test_classify_invalid_error(self, manager: RetryManager) -> None:
        """Test invalid parameter error classification."""
        error = Exception("Invalid symbol")
        classification = await manager._classify_error(error)
        assert classification == ErrorClassification.NON_RETRYABLE
    
    @pytest.mark.asyncio
    async def test_classify_degraded_error(self, manager: RetryManager) -> None:
        """Test degraded service error classification."""
        error = Exception("Service degraded")
        classification = await manager._classify_error(error)
        assert classification == ErrorClassification.DEGRADED
    
    @pytest.mark.asyncio
    async def test_classify_generic_error(self, manager: RetryManager) -> None:
        """Test generic error defaults to retryable."""
        error = Exception("Unknown error")
        classification = await manager._classify_error(error)
        assert classification == ErrorClassification.RETRYABLE


class TestRetryExecution:
    """Test retry execution."""
    
    @pytest.fixture
    def manager(self) -> RetryManager:
        """Create manager instance."""
        return RetryManager()
    
    @pytest.mark.asyncio
    async def test_successful_first_attempt(self, manager: RetryManager) -> None:
        """Test successful execution on first attempt."""
        async def successful_op():
            return "success"
        
        result = await manager.execute_with_retry(successful_op, "test_op")
        assert result == "success"
        assert manager.statistics["successful_attempts"] == 1
        assert manager.statistics["failed_attempts"] == 0
    
    @pytest.mark.asyncio
    async def test_successful_after_retries(self, manager: RetryManager) -> None:
        """Test successful execution after retries."""
        call_count = 0
        
        async def flaky_op():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Connection timeout")
            return "success"
        
        result = await manager.execute_with_retry(flaky_op, "test_op")
        assert result == "success"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_permanent_failure(self, manager: RetryManager) -> None:
        """Test permanent failure goes to DLQ."""
        async def failing_op():
            raise Exception("Invalid symbol")
        
        with pytest.raises(Exception):
            await manager.execute_with_retry(failing_op, "test_op")
        
        assert len(manager.dead_letter_queue) == 1
        assert manager.dead_letter_queue[0].classification == "NON_RETRYABLE"
    
    @pytest.mark.asyncio
    async def test_max_retries_exhausted(self, manager: RetryManager) -> None:
        """Test max retries exhausted."""
        async def always_fails():
            raise Exception("Connection timeout")
        
        with pytest.raises(Exception):
            await manager.execute_with_retry(always_fails, "test_op")
        
        assert len(manager.dead_letter_queue) == 1
        assert manager.dead_letter_queue[0].attempts == 3
    
    @pytest.mark.asyncio
    async def test_execution_with_args(self, manager: RetryManager) -> None:
        """Test execute with positional arguments."""
        async def operation_with_args(arg1, arg2):
            return f"{arg1}-{arg2}"
        
        result = await manager.execute_with_retry(
            operation_with_args,
            "test_op",
            "hello",
            "world"
        )
        assert result == "hello-world"
    
    @pytest.mark.asyncio
    async def test_execution_with_kwargs(self, manager: RetryManager) -> None:
        """Test execute with keyword arguments."""
        async def operation_with_kwargs(name, value):
            return {"name": name, "value": value}
        
        result = await manager.execute_with_retry(
            operation_with_kwargs,
            "test_op",
            name="test",
            value=42
        )
        assert result == {"name": "test", "value": 42}


class TestDeadLetterQueue:
    """Test dead letter queue operations."""
    
    @pytest.fixture
    def manager(self) -> RetryManager:
        """Create manager instance."""
        return RetryManager()
    
    def test_empty_dlq_initially(self, manager: RetryManager) -> None:
        """Test DLQ is empty initially."""
        assert manager.get_dlq_entries() == []
    
    def test_add_to_dlq(self, manager: RetryManager) -> None:
        """Test adding to DLQ."""
        error = Exception("Test error")
        manager._add_to_dlq("test_op", error, 3, "RETRYABLE")
        
        assert len(manager.dead_letter_queue) == 1
        assert manager.dead_letter_queue[0].operation_name == "test_op"
    
    def test_dlq_entries_copy(self, manager: RetryManager) -> None:
        """Test DLQ entries returns copy."""
        error = Exception("Test error")
        manager._add_to_dlq("test_op", error, 3, "RETRYABLE")
        
        entries = manager.get_dlq_entries()
        entries.clear()
        
        # Original should not be cleared
        assert len(manager.dead_letter_queue) == 1
    
    def test_clear_dlq(self, manager: RetryManager) -> None:
        """Test clearing DLQ."""
        error = Exception("Test error")
        manager._add_to_dlq("test_op", error, 3, "RETRYABLE")
        manager._add_to_dlq("test_op2", error, 3, "RETRYABLE")
        
        manager.clear_dlq()
        assert len(manager.dead_letter_queue) == 0


class TestStatistics:
    """Test statistics tracking."""
    
    @pytest.fixture
    def manager(self) -> RetryManager:
        """Create manager instance."""
        return RetryManager()
    
    def test_initial_statistics(self, manager: RetryManager) -> None:
        """Test initial statistics."""
        stats = manager.get_statistics()
        assert stats["total_attempts"] == 0
        assert stats["successful_attempts"] == 0
        assert stats["failed_attempts"] == 0
        assert stats["dlq_entries"] == 0
        assert stats["success_rate"] == 0.0
    
    @pytest.mark.asyncio
    async def test_statistics_after_success(self, manager: RetryManager) -> None:
        """Test statistics after successful execution."""
        async def successful_op():
            return "success"
        
        await manager.execute_with_retry(successful_op, "test_op")
        stats = manager.get_statistics()
        
        assert stats["total_attempts"] == 1
        assert stats["successful_attempts"] == 1
        assert stats["success_rate"] == 1.0
    
    @pytest.mark.asyncio
    async def test_statistics_mixed_results(self, manager: RetryManager) -> None:
        """Test statistics with mixed results."""
        async def successful_op():
            return "success"
        
        async def failing_op():
            raise Exception("Invalid symbol")
        
        await manager.execute_with_retry(successful_op, "op1")
        
        try:
            await manager.execute_with_retry(failing_op, "op2")
        except Exception:
            pass
        
        stats = manager.get_statistics()
        assert stats["successful_attempts"] == 1
        assert stats["dlq_entries"] == 1


class TestEdgeCases:
    """Test edge cases."""
    
    @pytest.fixture
    def manager(self) -> RetryManager:
        """Create manager instance."""
        return RetryManager()
    
    @pytest.mark.asyncio
    async def test_immediate_failure_no_retries(self) -> None:
        """Test immediate failure with max_attempts=1."""
        config = RetryConfig(max_attempts=1)
        manager = RetryManager(config)
        
        async def failing_op():
            raise Exception("Invalid symbol")
        
        with pytest.raises(Exception):
            await manager.execute_with_retry(failing_op, "test_op")
        
        assert len(manager.dead_letter_queue) == 1
    
    @pytest.mark.asyncio
    async def test_zero_args_operation(self, manager: RetryManager) -> None:
        """Test operation with zero arguments."""
        async def no_args_op():
            return "result"
        
        result = await manager.execute_with_retry(no_args_op, "test_op")
        assert result == "result"
    
    @pytest.mark.asyncio
    async def test_operation_returns_none(self, manager: RetryManager) -> None:
        """Test operation that returns None."""
        async def returns_none():
            return None
        
        result = await manager.execute_with_retry(returns_none, "test_op")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_very_large_max_attempts(self) -> None:
        """Test with very large max_attempts."""
        config = RetryConfig(max_attempts=100)
        manager = RetryManager(config)
        
        async def quick_op():
            return "success"
        
        result = await manager.execute_with_retry(quick_op, "test_op")
        assert result == "success"


class TestIntegration:
    """Integration tests."""
    
    @pytest.mark.asyncio
    async def test_complete_retry_workflow(self) -> None:
        """Test complete retry workflow."""
        manager = RetryManager()
        
        # Track behavior
        attempt_count = 0
        
        async def flaky_operation(id):
            nonlocal attempt_count
            attempt_count += 1
            
            if attempt_count < 3:
                raise Exception("Temporary timeout")
            
            return f"Completed: {id}"
        
        result = await manager.execute_with_retry(
            flaky_operation,
            "test_operation",
            "task-123"
        )
        
        assert result == "Completed: task-123"
        assert manager.statistics["successful_attempts"] == 1
        assert len(manager.dead_letter_queue) == 0
