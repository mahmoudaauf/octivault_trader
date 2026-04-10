# -*- coding: utf-8 -*-
"""
RetryManager - Exponential Backoff Retry Logic

Responsibility:
- Classify errors as retryable vs non-retryable
- Implement exponential backoff with jitter
- Track failed operations (dead letter queue)
- Provide retry metrics and statistics

This module implements the error recovery strategy to address
the "Limited Error Recovery" issue identified in Phase 2 review.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Callable, Awaitable
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime, timezone
from enum import Enum
import random


class ErrorClassification(Enum):
    """Classification of error types."""
    RETRYABLE = "retryable"  # Transient, should retry
    NON_RETRYABLE = "non_retryable"  # Permanent, don't retry
    DEGRADED = "degraded"  # Partial success, may retry with caution


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    # Retry strategy parameters
    max_attempts: int = 3
    initial_delay_ms: float = 100.0
    max_delay_ms: float = 30000.0  # 30 seconds
    backoff_multiplier: float = 2.0
    jitter_enabled: bool = True
    
    # Error classification
    retryable_errors: List[str] = field(default_factory=lambda: [
        "EXTERNAL_API_ERROR",
        "NETWORK_ERROR",
        "TIMEOUT_ERROR",
        "RATE_LIMIT",
        "TEMPORARY_SERVICE_ERROR",
    ])
    
    non_retryable_errors: List[str] = field(default_factory=lambda: [
        "INVALID_PARAMETERS",
        "INSUFFICIENT_BALANCE",
        "INVALID_ORDER",
        "AUTHENTICATION_ERROR",
        "PERMISSION_ERROR",
    ])
    
    degraded_errors: List[str] = field(default_factory=lambda: [
        "PARTIAL_FILL",
        "QUANTITY_REDUCED",
        "PRICE_SLIPPAGE",
    ])


@dataclass
class FailedOperation:
    """Record of a failed operation."""
    operation_name: str
    error_type: str
    error_message: str
    attempts: int
    last_attempt_time: datetime
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class RetryManager:
    """Manages retry logic for operations."""
    
    def __init__(self, config: Optional[RetryConfig] = None):
        """
        Initialize retry manager.
        
        Args:
            config: RetryConfig instance or None for defaults
        """
        self.config = config or RetryConfig()
        self.logger = logging.getLogger("RetryManager")
        
        # Dead letter queue for failed operations
        self.failed_queue: deque = deque()
        self.max_queue_size = 1000
        
        # Statistics
        self.stats = {
            'total_attempts': 0,
            'successful_retries': 0,
            'failed_operations': 0,
            'operations_retried': {},  # operation_name -> count
        }
    
    async def execute_with_retry(
        self,
        operation: Callable[..., Awaitable],
        operation_name: str,
        *args,
        **kwargs,
    ) -> Any:
        """
        Execute operation with automatic retry on failure.
        
        Args:
            operation: Async function to execute
            operation_name: Name for logging and tracking
            *args: Positional arguments for operation
            **kwargs: Keyword arguments for operation
        
        Returns:
            Result from successful operation execution
        
        Raises:
            Last ExecutionError if all retries exhausted
        """
        attempt = 0
        last_error = None
        
        while attempt < self.config.max_attempts:
            try:
                attempt += 1
                self.stats['total_attempts'] += 1
                
                result = await operation(*args, **kwargs)
                
                # Success!
                if attempt > 1:
                    self.stats['successful_retries'] += 1
                    self.logger.info(
                        f"✅ {operation_name} succeeded on attempt {attempt}"
                    )
                
                return result
                
            except Exception as e:
                last_error = e
                error_type = self._classify_error(e)
                
                # Check if error is retryable
                if error_type == ErrorClassification.NON_RETRYABLE:
                    self.logger.warning(
                        f"❌ Non-retryable error in {operation_name}: "
                        f"{e.__class__.__name__}: {str(e)}"
                    )
                    raise
                
                # Check if this is last attempt
                if attempt >= self.config.max_attempts:
                    self.logger.error(
                        f"❌ All {self.config.max_attempts} retries exhausted "
                        f"for {operation_name}"
                    )
                    break
                
                # Calculate backoff delay
                delay_ms = self._calculate_backoff(attempt - 1)
                
                self.logger.info(
                    f"⏳ Retry {attempt}/{self.config.max_attempts} "
                    f"for {operation_name} in {delay_ms:.0f}ms "
                    f"({error_type.value})"
                )
                
                # Wait before retry
                await asyncio.sleep(delay_ms / 1000.0)
        
        # All retries exhausted
        self.stats['failed_operations'] += 1
        self.stats['operations_retried'][operation_name] = (
            self.stats['operations_retried'].get(operation_name, 0) + 1
        )
        
        # Queue for dead letter processing
        failed_op = FailedOperation(
            operation_name=operation_name,
            error_type=self._classify_error(last_error).value if last_error else 'unknown',
            error_message=str(last_error) if last_error else 'unknown error',
            attempts=attempt,
            last_attempt_time=datetime.now(timezone.utc),
            args=args,
            kwargs=kwargs,
        )
        
        self._queue_failed_operation(failed_op)
        
        raise last_error
    
    def _classify_error(self, error: Exception) -> ErrorClassification:
        """
        Classify error type for retry decision.
        
        Args:
            error: Exception to classify
        
        Returns:
            ErrorClassification enum
        """
        error_str = error.__class__.__name__
        error_msg = str(error).upper()
        
        # Check against configured lists
        if error_str in self.config.retryable_errors or any(
            e in error_msg for e in self.config.retryable_errors
        ):
            return ErrorClassification.RETRYABLE
        
        if error_str in self.config.non_retryable_errors or any(
            e in error_msg for e in self.config.non_retryable_errors
        ):
            return ErrorClassification.NON_RETRYABLE
        
        if error_str in self.config.degraded_errors or any(
            e in error_msg for e in self.config.degraded_errors
        ):
            return ErrorClassification.DEGRADED
        
        # Default to retryable for unknown errors
        return ErrorClassification.RETRYABLE
    
    def _calculate_backoff(self, retry_count: int) -> float:
        """
        Calculate backoff delay with exponential growth and optional jitter.
        
        Args:
            retry_count: Retry number (0-based)
        
        Returns:
            Delay in milliseconds
        """
        # Exponential backoff: initial_delay * multiplier^retry_count
        delay = self.config.initial_delay_ms * (
            self.config.backoff_multiplier ** retry_count
        )
        
        # Cap at maximum
        delay = min(delay, self.config.max_delay_ms)
        
        # Add jitter (±10%)
        if self.config.jitter_enabled:
            jitter = delay * 0.1 * (random.random() - 0.5)
            delay = max(delay + jitter, self.config.initial_delay_ms)
        
        return delay
    
    def _queue_failed_operation(self, failed_op: FailedOperation):
        """Queue failed operation for dead letter processing."""
        self.failed_queue.append(failed_op)
        
        # Maintain size limit
        while len(self.failed_queue) > self.max_queue_size:
            self.failed_queue.popleft()
        
        self.logger.error(
            f"Failed operation queued: {failed_op.operation_name} "
            f"({failed_op.attempts} attempts)"
        )
    
    def get_failed_operations(
        self,
        limit: Optional[int] = None,
    ) -> List[FailedOperation]:
        """
        Get list of failed operations from dead letter queue.
        
        Args:
            limit: Maximum number to return (None = all)
        
        Returns:
            List of failed operations
        """
        failed_list = list(self.failed_queue)
        
        if limit:
            failed_list = failed_list[-limit:]
        
        return failed_list
    
    def clear_failed_operations(self):
        """Clear failed operations queue."""
        count = len(self.failed_queue)
        self.failed_queue.clear()
        self.logger.info(f"Cleared {count} failed operations from queue")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get retry statistics for monitoring."""
        return {
            'total_attempts': self.stats['total_attempts'],
            'successful_retries': self.stats['successful_retries'],
            'failed_operations': self.stats['failed_operations'],
            'failed_queue_size': len(self.failed_queue),
            'operations_failed': self.stats['operations_retried'],
            'config': {
                'max_attempts': self.config.max_attempts,
                'backoff_multiplier': self.config.backoff_multiplier,
                'max_delay_ms': self.config.max_delay_ms,
            },
        }
    
    def reset_statistics(self):
        """Reset statistics counters."""
        self.stats = {
            'total_attempts': 0,
            'successful_retries': 0,
            'failed_operations': 0,
            'operations_retried': {},
        }
        self.logger.info("Retry statistics reset")
