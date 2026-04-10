"""
Test Suite for Error Handler Module

Tests error classification, structured logging, and recovery decisions.
"""

import pytest
import logging
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from core.error_types import (
    ErrorSeverity,
    ErrorCategory,
    ErrorRecovery,
    create_error_context,
    TraderException,
    BootstrapError,
    ExchangeError,
    ExchangeRateLimitError,
    StateError,
    StateCorruptionError,
    NetworkError,
    ConnectionTimeoutError,
    ValidationError,
    InvalidParameterError,
    ExecutionError,
    OrderPlacementError,
)
from core.error_handler import (
    ErrorClassification,
    ErrorClassifier,
    StructuredErrorLogger,
    RecoveryDecisionEngine,
    ErrorHandler,
    get_error_handler,
    reset_error_handler,
)


# ============================================================================
# ERROR CLASSIFICATION TESTS
# ============================================================================

class TestErrorClassification:
    """Test ErrorClassification class."""
    
    def test_create_classification(self):
        """Create ErrorClassification instance."""
        classification = ErrorClassification(
            error_type=ExchangeError,
            category=ErrorCategory.EXCHANGE,
            severity=ErrorSeverity.WARNING,
            is_retryable=True,
            is_critical=False,
            recovery_action=ErrorRecovery.RETRY,
            suggested_delay=2.0,
            max_retries=5,
        )
        assert classification.error_type == ExchangeError
        assert classification.is_retryable
        assert not classification.is_critical
        assert classification.suggested_delay == 2.0
    
    def test_should_retry_true(self):
        """Test should_retry() returns True when appropriate."""
        classification = ErrorClassification(
            error_type=ExchangeError,
            category=ErrorCategory.EXCHANGE,
            severity=ErrorSeverity.WARNING,
            is_retryable=True,
            is_critical=False,
            recovery_action=ErrorRecovery.RETRY,
        )
        assert classification.should_retry()
    
    def test_should_retry_false_not_retryable(self):
        """Test should_retry() returns False when not retryable."""
        classification = ErrorClassification(
            error_type=ValidationError,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.ERROR,
            is_retryable=False,
            is_critical=False,
            recovery_action=ErrorRecovery.NONE,
        )
        assert not classification.should_retry()
    
    def test_should_retry_false_critical(self):
        """Test should_retry() returns False for critical errors."""
        classification = ErrorClassification(
            error_type=StateError,
            category=ErrorCategory.STATE,
            severity=ErrorSeverity.CRITICAL,
            is_retryable=True,
            is_critical=True,
            recovery_action=ErrorRecovery.ESCALATE,
        )
        assert not classification.should_retry()
    
    def test_should_circuit_break_true(self):
        """Test should_circuit_break() returns True."""
        classification = ErrorClassification(
            error_type=StateError,
            category=ErrorCategory.STATE,
            severity=ErrorSeverity.CRITICAL,
            is_retryable=False,
            is_critical=True,
            recovery_action=ErrorRecovery.CIRCUIT_BREAK,
        )
        assert classification.should_circuit_break()
    
    def test_should_escalate_true(self):
        """Test should_escalate() returns True."""
        classification = ErrorClassification(
            error_type=StateError,
            category=ErrorCategory.STATE,
            severity=ErrorSeverity.CRITICAL,
            is_retryable=False,
            is_critical=True,
            recovery_action=ErrorRecovery.ESCALATE,
        )
        assert classification.should_escalate()


# ============================================================================
# ERROR CLASSIFIER TESTS
# ============================================================================

class TestErrorClassifier:
    """Test ErrorClassifier class."""
    
    def test_classify_exchange_error(self):
        """Classify an exchange error."""
        context = create_error_context(
            category=ErrorCategory.EXCHANGE,
            severity=ErrorSeverity.WARNING,
            error_code="EXCHANGE_RATE_LIMIT",
            message="Rate limited",
            recovery_strategy=ErrorRecovery.RETRY,
        )
        exc = ExchangeRateLimitError(context)
        
        classification = ErrorClassifier.classify(exc)
        
        assert classification is not None
        assert classification.is_retryable
        assert not classification.is_critical
        assert classification.recovery_action == ErrorRecovery.RETRY
    
    def test_classify_state_error(self):
        """Classify a state error."""
        context = create_error_context(
            category=ErrorCategory.STATE,
            severity=ErrorSeverity.CRITICAL,
            error_code="STATE_CORRUPTION",
            message="Corruption detected",
            recovery_strategy=ErrorRecovery.ESCALATE,
        )
        exc = StateCorruptionError(context)
        
        classification = ErrorClassifier.classify(exc)
        
        assert classification is not None
        assert not classification.is_retryable
        assert classification.is_critical
        assert classification.recovery_action == ErrorRecovery.ESCALATE
    
    def test_classify_network_error(self):
        """Classify a network error."""
        context = create_error_context(
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.WARNING,
            error_code="NETWORK_TIMEOUT",
            message="Connection timeout",
            recovery_strategy=ErrorRecovery.RETRY,
        )
        exc = ConnectionTimeoutError(context)
        
        classification = ErrorClassifier.classify(exc)
        
        assert classification is not None
        assert classification.is_retryable
        assert classification.suggested_delay > 0
    
    def test_classify_validation_error(self):
        """Classify a validation error."""
        context = create_error_context(
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.WARNING,
            error_code="INVALID_PARAMETER",
            message="Invalid value",
        )
        exc = InvalidParameterError(context)
        
        classification = ErrorClassifier.classify(exc)
        
        assert classification is not None
        assert not classification.is_retryable
        assert not classification.is_critical
    
    def test_classify_non_trader_exception(self):
        """Classify non-TraderException returns None."""
        regular_exception = ValueError("Some error")
        
        classification = ErrorClassifier.classify(regular_exception)
        
        assert classification is None
    
    def test_classify_bootstrap_error(self):
        """Classify a bootstrap error."""
        context = create_error_context(
            category=ErrorCategory.BOOTSTRAP,
            severity=ErrorSeverity.ERROR,
            error_code="BOOTSTRAP_TIMEOUT",
            message="Bootstrap timed out",
            recovery_strategy=ErrorRecovery.ESCALATE,
        )
        exc = BootstrapError(context)
        
        classification = ErrorClassifier.classify(exc)
        
        assert classification is not None
        assert not classification.is_retryable
        assert classification.recovery_action == ErrorRecovery.ESCALATE


# ============================================================================
# STRUCTURED ERROR LOGGER TESTS
# ============================================================================

class TestStructuredErrorLogger:
    """Test StructuredErrorLogger class."""
    
    def test_log_trader_exception(self, caplog):
        """Log a TraderException."""
        context = create_error_context(
            category=ErrorCategory.EXCHANGE,
            severity=ErrorSeverity.ERROR,
            error_code="EXCHANGE_API",
            message="API call failed",
            operation="get_balance",
            component="exchange_client",
            symbol="BTC/USDT",
        )
        exc = ExchangeError(context)
        
        with caplog.at_level(logging.ERROR):
            StructuredErrorLogger.log_exception(exc)
        
        assert "[EXCHANGE_API]" in caplog.text
        assert "API call failed" in caplog.text
    
    def test_log_non_trader_exception(self, caplog):
        """Log a non-TraderException."""
        exc = ValueError("Some error")
        
        with caplog.at_level(logging.ERROR):
            StructuredErrorLogger.log_exception(exc)
        
        assert "Unexpected exception" in caplog.text
    
    def test_log_with_classification(self, caplog):
        """Log exception with classification."""
        context = create_error_context(
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.WARNING,
            error_code="NETWORK_TIMEOUT",
            message="Timeout",
        )
        exc = ConnectionTimeoutError(context)
        classification = ErrorClassifier.classify(exc)
        
        with caplog.at_level(logging.WARNING):
            StructuredErrorLogger.log_exception(
                exc,
                classification=classification
            )
        
        assert "[NETWORK_TIMEOUT]" in caplog.text
    
    def test_log_with_additional_context(self, caplog):
        """Log exception with additional context."""
        context = create_error_context(
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.WARNING,
            error_code="INVALID_PARAMETER",
            message="Invalid",
        )
        exc = InvalidParameterError(context)
        
        additional = {"parameter_name": "amount", "value": -100}
        
        with caplog.at_level(logging.WARNING):
            StructuredErrorLogger.log_exception(
                exc,
                additional_context=additional
            )
        
        assert "[INVALID_PARAMETER]" in caplog.text
    
    def test_log_recovery_attempt(self, caplog):
        """Log a recovery attempt."""
        with caplog.at_level(logging.INFO):
            StructuredErrorLogger.log_recovery_attempt(
                error_code="EXCHANGE_API",
                attempt=1,
                recovery_action=ErrorRecovery.RETRY,
                delay=2.5,
            )
        
        assert "[EXCHANGE_API]" in caplog.text
        assert "attempt 1" in caplog.text.lower()
        assert "retry" in caplog.text.lower()


# ============================================================================
# RECOVERY DECISION ENGINE TESTS
# ============================================================================

class TestRecoveryDecisionEngine:
    """Test RecoveryDecisionEngine class."""
    
    def test_should_retry_true(self):
        """Test should_retry() returns True."""
        engine = RecoveryDecisionEngine()
        
        classification = ErrorClassification(
            error_type=ExchangeError,
            category=ErrorCategory.EXCHANGE,
            severity=ErrorSeverity.WARNING,
            is_retryable=True,
            is_critical=False,
            recovery_action=ErrorRecovery.RETRY,
            max_retries=3,
        )
        
        assert engine.should_retry(classification, "EXCHANGE_API")
    
    def test_should_retry_false_max_retries_exceeded(self):
        """Test should_retry() returns False when max retries exceeded."""
        engine = RecoveryDecisionEngine()
        
        classification = ErrorClassification(
            error_type=ExchangeError,
            category=ErrorCategory.EXCHANGE,
            severity=ErrorSeverity.WARNING,
            is_retryable=True,
            is_critical=False,
            recovery_action=ErrorRecovery.RETRY,
            max_retries=3,
        )
        
        # Record max retries
        for _ in range(3):
            engine.record_retry_attempt("EXCHANGE_API")
        
        assert not engine.should_retry(classification, "EXCHANGE_API")
    
    def test_record_retry_attempt(self):
        """Test recording retry attempts."""
        engine = RecoveryDecisionEngine()
        
        engine.record_retry_attempt("TEST_ERROR")
        engine.record_retry_attempt("TEST_ERROR")
        
        assert len(engine.retry_history["TEST_ERROR"]) == 2
    
    def test_reset_retry_history(self):
        """Test resetting retry history."""
        engine = RecoveryDecisionEngine()
        
        engine.record_retry_attempt("TEST_ERROR")
        engine.reset_retry_history("TEST_ERROR")
        
        assert len(engine.retry_history["TEST_ERROR"]) == 0
    
    def test_activate_circuit_breaker(self):
        """Test activating circuit breaker."""
        engine = RecoveryDecisionEngine()
        
        engine.activate_circuit_breaker("TEST_ERROR", duration=10.0)
        
        assert engine.is_circuit_broken("TEST_ERROR")
    
    def test_circuit_breaker_expires(self):
        """Test circuit breaker expires."""
        engine = RecoveryDecisionEngine()
        
        # Activate with past expiration time
        engine.circuit_breaker_status["TEST_ERROR"] = (
            True,
            datetime.now() - timedelta(seconds=1)
        )
        
        # Should deactivate automatically
        assert not engine.is_circuit_broken("TEST_ERROR")
    
    def test_deactivate_circuit_breaker(self):
        """Test deactivating circuit breaker."""
        engine = RecoveryDecisionEngine()
        
        engine.activate_circuit_breaker("TEST_ERROR")
        engine.deactivate_circuit_breaker("TEST_ERROR")
        
        assert not engine.is_circuit_broken("TEST_ERROR")
    
    def test_get_next_retry_delay(self):
        """Test calculating retry delay."""
        engine = RecoveryDecisionEngine()
        
        classification = ErrorClassification(
            error_type=ExchangeError,
            category=ErrorCategory.EXCHANGE,
            severity=ErrorSeverity.WARNING,
            is_retryable=True,
            is_critical=False,
            recovery_action=ErrorRecovery.RETRY,
            suggested_delay=1.0,
        )
        
        # First retry should be relatively short
        delay1 = engine.get_next_retry_delay(classification, 1)
        assert 0.9 < delay1 < 1.2  # With jitter
        
        # Second retry should be longer (exponential backoff)
        delay2 = engine.get_next_retry_delay(classification, 2)
        assert delay2 > delay1


# ============================================================================
# ERROR HANDLER TESTS
# ============================================================================

class TestErrorHandler:
    """Test ErrorHandler facade class."""
    
    def test_create_handler(self):
        """Create ErrorHandler instance."""
        handler = ErrorHandler()
        
        assert handler.classifier is not None
        assert handler.logger is not None
        assert handler.recovery_engine is not None
    
    def test_handle_trader_exception(self):
        """Handle a TraderException."""
        handler = ErrorHandler()
        
        context = create_error_context(
            category=ErrorCategory.EXCHANGE,
            severity=ErrorSeverity.WARNING,
            error_code="EXCHANGE_API",
            message="API error",
            recovery_strategy=ErrorRecovery.RETRY,
        )
        exc = ExchangeError(context)
        
        classification = handler.handle_exception(exc)
        
        assert classification is not None
        assert classification.is_retryable
    
    def test_handle_non_trader_exception(self):
        """Handle a non-TraderException."""
        handler = ErrorHandler()
        
        exc = ValueError("Some error")
        
        classification = handler.handle_exception(exc)
        
        # Non-trader exceptions return None classification
        assert classification is None
    
    def test_should_handle_recovery_true(self):
        """Test should_handle_recovery() returns True."""
        handler = ErrorHandler()
        
        context = create_error_context(
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.WARNING,
            error_code="NETWORK_TIMEOUT",
            message="Timeout",
            recovery_strategy=ErrorRecovery.RETRY,
        )
        exc = ConnectionTimeoutError(context)
        
        should_recover, classification = handler.should_handle_recovery(exc)
        
        assert should_recover
        assert classification is not None
    
    def test_should_handle_recovery_false(self):
        """Test should_handle_recovery() returns False."""
        handler = ErrorHandler()
        
        context = create_error_context(
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.WARNING,
            error_code="INVALID_PARAMETER",
            message="Invalid",
        )
        exc = InvalidParameterError(context)
        
        should_recover, classification = handler.should_handle_recovery(exc)
        
        assert not should_recover
        assert classification is not None
    
    def test_record_recovery_attempt(self):
        """Test recording recovery attempt."""
        handler = ErrorHandler()
        
        context = create_error_context(
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.WARNING,
            error_code="NETWORK_TIMEOUT",
            message="Timeout",
            recovery_strategy=ErrorRecovery.RETRY,
        )
        exc = ConnectionTimeoutError(context)
        
        delay = handler.record_recovery_attempt(
            exc,
            ErrorRecovery.RETRY
        )
        
        assert delay is not None
        assert delay > 0
    
    def test_record_recovery_non_retryable(self):
        """Test recording recovery for non-retryable error."""
        handler = ErrorHandler()
        
        context = create_error_context(
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.WARNING,
            error_code="INVALID_PARAMETER",
            message="Invalid",
            recovery_strategy=ErrorRecovery.NONE,
        )
        exc = InvalidParameterError(context)
        
        # Try to record recovery with RETRY, but error says NO recovery
        delay = handler.record_recovery_attempt(
            exc,
            ErrorRecovery.RETRY
        )
        
        # Non-retryable errors (recovery_strategy=NONE) shouldn't return a delay
        assert delay is None


# ============================================================================
# SINGLETON TESTS
# ============================================================================

class TestErrorHandlerSingleton:
    """Test error handler singleton."""
    
    def test_get_error_handler_creates_instance(self):
        """Test get_error_handler() creates instance."""
        reset_error_handler()
        
        handler1 = get_error_handler()
        assert handler1 is not None
    
    def test_get_error_handler_returns_same_instance(self):
        """Test get_error_handler() returns same instance."""
        reset_error_handler()
        
        handler1 = get_error_handler()
        handler2 = get_error_handler()
        
        assert handler1 is handler2
    
    def test_reset_error_handler(self):
        """Test reset_error_handler()."""
        reset_error_handler()
        
        handler1 = get_error_handler()
        reset_error_handler()
        handler2 = get_error_handler()
        
        assert handler1 is not handler2


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestErrorHandlingWorkflow:
    """Integration tests for error handling workflow."""
    
    def test_complete_error_workflow_retryable(self):
        """Test complete workflow for retryable error."""
        handler = ErrorHandler()
        
        # Create a retryable error
        context = create_error_context(
            category=ErrorCategory.EXCHANGE,
            severity=ErrorSeverity.WARNING,
            error_code="EXCHANGE_RATE_LIMIT",
            message="Rate limited",
            operation="get_ticker",
            recovery_strategy=ErrorRecovery.RETRY,
        )
        exc = ExchangeRateLimitError(context)
        
        # Step 1: Handle exception
        classification = handler.handle_exception(exc)
        assert classification is not None
        
        # Step 2: Check if should recover
        should_recover, _ = handler.should_handle_recovery(exc, attempt=1)
        assert should_recover
        
        # Step 3: Record recovery attempt
        delay = handler.record_recovery_attempt(exc, ErrorRecovery.RETRY)
        assert delay > 0
    
    def test_complete_error_workflow_critical(self):
        """Test complete workflow for critical error."""
        handler = ErrorHandler()
        
        # Create a critical error
        context = create_error_context(
            category=ErrorCategory.STATE,
            severity=ErrorSeverity.CRITICAL,
            error_code="STATE_CORRUPTION",
            message="State corrupted",
        )
        exc = StateCorruptionError(context)
        
        # Step 1: Handle exception
        classification = handler.handle_exception(exc)
        assert classification is not None
        
        # Step 2: Should not recover
        should_recover, _ = handler.should_handle_recovery(exc)
        assert not should_recover
        
        # Step 3: Should escalate
        assert classification.should_escalate()
    
    def test_multiple_retry_attempts(self):
        """Test multiple retry attempts."""
        handler = ErrorHandler()
        
        context = create_error_context(
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.WARNING,
            error_code="CONNECTION_TIMEOUT",
            message="Timeout",
            recovery_strategy=ErrorRecovery.RETRY,
        )
        exc = ConnectionTimeoutError(context)
        
        # Simulate multiple retry attempts
        delays = []
        for attempt in range(3):
            should_recover, _ = handler.should_handle_recovery(exc, attempt)
            if should_recover:
                delay = handler.record_recovery_attempt(exc, ErrorRecovery.RETRY)
                if delay:
                    delays.append(delay)
        
        # Each delay should be longer than the last (exponential backoff)
        assert len(delays) == 3
        assert delays[1] > delays[0]
        assert delays[2] > delays[1]


# ============================================================================
# CLASSIFICATION RULES TESTS
# ============================================================================

class TestClassificationRules:
    """Test that classification rules are correct."""
    
    def test_all_rules_defined(self):
        """Verify all error categories have rules."""
        from core.error_types import (
            BootstrapError, ArbitrationError, LifecycleError,
            ExecutionError, ExchangeError, StateError,
            NetworkError, ValidationError, ConfigurationError, ResourceError
        )
        
        error_types = [
            BootstrapError, ArbitrationError, LifecycleError,
            ExecutionError, ExchangeError, StateError,
            NetworkError, ValidationError, ConfigurationError, ResourceError
        ]
        
        for error_type in error_types:
            assert error_type in ErrorClassifier.CLASSIFICATION_RULES
            rules = ErrorClassifier.CLASSIFICATION_RULES[error_type]
            assert "severity" in rules
            assert "is_retryable" in rules
            assert "recovery" in rules


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
