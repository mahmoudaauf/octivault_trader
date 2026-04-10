"""
Error Handler Module

Provides error classification, structured logging, and recovery decision logic.
Bridges typed exceptions with recovery strategies.

Philosophy:
  • Classify errors to enable recovery decisions
  • Log with rich context for debugging
  • Automate retry/fallback decisions
  • Track error patterns for monitoring
"""

import logging
from typing import Optional, Dict, Type, Callable, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum

from core.error_types import (
    TraderException,
    ErrorCategory,
    ErrorSeverity,
    ErrorRecovery,
    ErrorContext,
    # Error types
    BootstrapError,
    ArbitrationError,
    LifecycleError,
    ExecutionError,
    ExchangeError,
    StateError,
    NetworkError,
    ValidationError,
    ConfigurationError,
    ResourceError,
)


logger = logging.getLogger(__name__)


# ============================================================================
# ERROR CLASSIFICATION
# ============================================================================

class ErrorClassification:
    """Classification result for an exception."""
    
    def __init__(
        self,
        error_type: Type[TraderException],
        category: ErrorCategory,
        severity: ErrorSeverity,
        is_retryable: bool,
        is_critical: bool,
        recovery_action: ErrorRecovery,
        suggested_delay: Optional[float] = None,
        max_retries: int = 3,
    ):
        """
        Initialize error classification.
        
        Args:
            error_type: Type of the error
            category: Error category
            severity: Error severity
            is_retryable: Whether operation can be retried
            is_critical: Whether error is critical
            recovery_action: Recommended recovery action
            suggested_delay: Suggested delay before retry (seconds)
            max_retries: Maximum retry attempts
        """
        self.error_type = error_type
        self.category = category
        self.severity = severity
        self.is_retryable = is_retryable
        self.is_critical = is_critical
        self.recovery_action = recovery_action
        self.suggested_delay = suggested_delay or 1.0
        self.max_retries = max_retries
    
    def should_retry(self) -> bool:
        """Determine if operation should be retried."""
        return self.is_retryable and not self.is_critical
    
    def should_circuit_break(self) -> bool:
        """Determine if circuit breaker should be activated."""
        return (self.recovery_action == ErrorRecovery.CIRCUIT_BREAK or
                self.is_critical)
    
    def should_escalate(self) -> bool:
        """Determine if error should be escalated."""
        return (self.recovery_action == ErrorRecovery.ESCALATE or
                self.is_critical)


# ============================================================================
# ERROR CLASSIFIER
# ============================================================================

class ErrorClassifier:
    """Classifies exceptions to enable recovery decisions."""
    
    # Classification rules: Exception type → Classification parameters
    CLASSIFICATION_RULES: Dict[Type[TraderException], Dict[str, Any]] = {
        # Bootstrap errors - Generally not retryable
        BootstrapError: {
            "severity": ErrorSeverity.ERROR,
            "is_retryable": False,
            "recovery": ErrorRecovery.ESCALATE,
        },
        
        # Arbitration errors - Usually not retryable
        ArbitrationError: {
            "severity": ErrorSeverity.WARNING,
            "is_retryable": False,
            "recovery": ErrorRecovery.SKIP,
        },
        
        # Lifecycle errors - May be retryable
        LifecycleError: {
            "severity": ErrorSeverity.ERROR,
            "is_retryable": False,
            "recovery": ErrorRecovery.RESET,
        },
        
        # Execution errors - Not retryable
        ExecutionError: {
            "severity": ErrorSeverity.ERROR,
            "is_retryable": False,
            "recovery": ErrorRecovery.NONE,
        },
        
        # Exchange errors - Often retryable
        ExchangeError: {
            "severity": ErrorSeverity.WARNING,
            "is_retryable": True,
            "recovery": ErrorRecovery.RETRY,
            "delay": 2.0,
            "max_retries": 5,
        },
        
        # State errors - Require escalation
        StateError: {
            "severity": ErrorSeverity.CRITICAL,
            "is_retryable": False,
            "recovery": ErrorRecovery.ESCALATE,
        },
        
        # Network errors - Retryable
        NetworkError: {
            "severity": ErrorSeverity.WARNING,
            "is_retryable": True,
            "recovery": ErrorRecovery.RETRY,
            "delay": 5.0,
            "max_retries": 3,
        },
        
        # Validation errors - Not retryable
        ValidationError: {
            "severity": ErrorSeverity.WARNING,
            "is_retryable": False,
            "recovery": ErrorRecovery.NONE,
        },
        
        # Configuration errors - Not retryable
        ConfigurationError: {
            "severity": ErrorSeverity.CRITICAL,
            "is_retryable": False,
            "recovery": ErrorRecovery.ESCALATE,
        },
        
        # Resource errors - Might be retryable
        ResourceError: {
            "severity": ErrorSeverity.ERROR,
            "is_retryable": True,
            "recovery": ErrorRecovery.RETRY,
            "delay": 10.0,
            "max_retries": 2,
        },
    }
    
    @classmethod
    def classify(
        cls,
        exception: Exception,
        attempt: int = 1,
    ) -> Optional[ErrorClassification]:
        """
        Classify an exception to determine recovery strategy.
        
        Args:
            exception: Exception to classify
            attempt: Current attempt number (for retry logic)
        
        Returns:
            ErrorClassification if exception is a TraderException, None otherwise
        """
        if not isinstance(exception, TraderException):
            return None
        
        error_type = type(exception)
        context = exception.context
        
        # Get base classification rules
        rules = cls._get_rules_for_exception(exception)
        if not rules:
            # Fallback for unknown exception types
            rules = {
                "severity": ErrorSeverity.ERROR,
                "is_retryable": False,
                "recovery": ErrorRecovery.ESCALATE,
            }
        
        # Extract parameters from rules (these are the defaults)
        severity = rules.get("severity", ErrorSeverity.ERROR)
        is_retryable = rules.get("is_retryable", False)
        recovery = rules.get("recovery", ErrorRecovery.ESCALATE)
        suggested_delay = rules.get("delay", 1.0)
        max_retries = rules.get("max_retries", 3)
        
        # Use context values as they are always set (don't override)
        severity = context.severity
        recovery = context.recovery_strategy
        
        # Override retryability based on recovery strategy
        is_retryable = recovery in (ErrorRecovery.RETRY, ErrorRecovery.RESET)
        
        is_critical = severity == ErrorSeverity.CRITICAL
        
        return ErrorClassification(
            error_type=error_type,
            category=context.category,
            severity=severity,
            is_retryable=is_retryable,
            is_critical=is_critical,
            recovery_action=recovery,
            suggested_delay=suggested_delay,
            max_retries=max_retries,
        )
    
    @classmethod
    def _get_rules_for_exception(
        cls,
        exception: TraderException,
    ) -> Optional[Dict[str, Any]]:
        """Get classification rules for exception."""
        exception_type = type(exception)
        
        # Check exact type first
        if exception_type in cls.CLASSIFICATION_RULES:
            return cls.CLASSIFICATION_RULES[exception_type]
        
        # Check base classes
        for base in exception_type.__mro__[1:]:
            if base in cls.CLASSIFICATION_RULES:
                return cls.CLASSIFICATION_RULES[base]
        
        return None


# ============================================================================
# ERROR LOGGING
# ============================================================================

class StructuredErrorLogger:
    """Logs errors with rich context for debugging and monitoring."""
    
    # Severity → logging level mapping
    SEVERITY_TO_LOG_LEVEL = {
        ErrorSeverity.DEBUG: logging.DEBUG,
        ErrorSeverity.INFO: logging.INFO,
        ErrorSeverity.WARNING: logging.WARNING,
        ErrorSeverity.ERROR: logging.ERROR,
        ErrorSeverity.CRITICAL: logging.CRITICAL,
    }
    
    @classmethod
    def log_exception(
        cls,
        exception: Exception,
        classification: Optional[ErrorClassification] = None,
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log an exception with structured context.
        
        Args:
            exception: Exception to log
            classification: Error classification (if available)
            additional_context: Additional context to include
        """
        if not isinstance(exception, TraderException):
            # Log non-trader exceptions differently
            logger.exception("Unexpected exception", exc_info=exception)
            return
        
        context = exception.context
        log_level = cls.SEVERITY_TO_LOG_LEVEL.get(
            context.severity,
            logging.ERROR
        )
        
        # Build log message
        parts = [
            f"[{context.error_code}]",
            f"({context.severity.value.upper()})",
        ]
        
        if context.operation:
            parts.append(f"| Op: {context.operation}")
        
        if context.component:
            parts.append(f"| Component: {context.component}")
        
        if context.symbol:
            parts.append(f"| Symbol: {context.symbol}")
        
        message = " ".join(parts)
        
        # Build structured log data
        log_data = {
            "error_code": context.error_code,
            "severity": context.severity.value,
            "category": context.category.value,
            "message": context.message,
            "operation": context.operation,
            "component": context.component,
            "symbol": context.symbol,
            "timestamp": context.timestamp.isoformat(),
        }
        
        if classification:
            log_data.update({
                "is_retryable": classification.is_retryable,
                "is_critical": classification.is_critical,
                "recovery_action": classification.recovery_action.value,
                "suggested_delay": classification.suggested_delay,
            })
        
        if additional_context:
            log_data["additional"] = additional_context
        
        if context.metadata:
            log_data["metadata"] = context.metadata
        
        if exception.cause:
            log_data["cause"] = str(exception.cause)
        
        # Log with structured data
        logger.log(
            log_level,
            message,
            extra={"structured": log_data},
            exc_info=exception,
        )
    
    @classmethod
    def log_recovery_attempt(
        cls,
        error_code: str,
        attempt: int,
        recovery_action: ErrorRecovery,
        delay: Optional[float] = None,
    ) -> None:
        """
        Log a recovery attempt.
        
        Args:
            error_code: Error code being recovered from
            attempt: Attempt number
            recovery_action: Recovery action being taken
            delay: Delay before retry (if applicable)
        """
        parts = [
            f"[{error_code}]",
            f"Recovery attempt {attempt}",
            f"| Action: {recovery_action.value}",
        ]
        
        if delay is not None:
            parts.append(f"| Delay: {delay}s")
        
        message = " ".join(parts)
        
        log_data = {
            "error_code": error_code,
            "attempt": attempt,
            "recovery_action": recovery_action.value,
        }
        
        if delay is not None:
            log_data["delay_seconds"] = delay
        
        logger.info(message, extra={"structured": log_data})


# ============================================================================
# RECOVERY DECISION ENGINE
# ============================================================================

class RecoveryDecisionEngine:
    """Determines recovery actions for classified errors."""
    
    def __init__(self):
        """Initialize recovery decision engine."""
        self.retry_history: Dict[str, list] = {}
        self.circuit_breaker_status: Dict[str, Tuple[bool, datetime]] = {}
    
    def should_retry(
        self,
        classification: ErrorClassification,
        error_code: str,
    ) -> bool:
        """
        Determine if operation should be retried.
        
        Args:
            classification: Error classification
            error_code: Error code
        
        Returns:
            True if should retry, False otherwise
        """
        if not classification.should_retry():
            return False
        
        # Check if max retries exceeded
        history = self.retry_history.get(error_code, [])
        if len(history) >= classification.max_retries:
            return False
        
        # Check if circuit breaker is active
        if self.is_circuit_broken(error_code):
            return False
        
        return True
    
    def record_retry_attempt(
        self,
        error_code: str,
    ) -> None:
        """
        Record a retry attempt.
        
        Args:
            error_code: Error code
        """
        if error_code not in self.retry_history:
            self.retry_history[error_code] = []
        
        self.retry_history[error_code].append(datetime.now())
    
    def reset_retry_history(self, error_code: str) -> None:
        """
        Reset retry history for an error code.
        
        Args:
            error_code: Error code
        """
        self.retry_history[error_code] = []
    
    def activate_circuit_breaker(
        self,
        error_code: str,
        duration: float = 60.0,
    ) -> None:
        """
        Activate circuit breaker for an error code.
        
        Args:
            error_code: Error code
            duration: Duration to keep circuit open (seconds)
        """
        reset_time = datetime.now() + timedelta(seconds=duration)
        self.circuit_breaker_status[error_code] = (True, reset_time)
        
        logger.warning(
            f"Circuit breaker activated for {error_code}",
            extra={
                "structured": {
                    "error_code": error_code,
                    "reset_time": reset_time.isoformat(),
                    "duration_seconds": duration,
                }
            }
        )
    
    def deactivate_circuit_breaker(self, error_code: str) -> None:
        """
        Deactivate circuit breaker for an error code.
        
        Args:
            error_code: Error code
        """
        if error_code in self.circuit_breaker_status:
            del self.circuit_breaker_status[error_code]
            logger.info(
                f"Circuit breaker deactivated for {error_code}",
                extra={"structured": {"error_code": error_code}}
            )
    
    def is_circuit_broken(self, error_code: str) -> bool:
        """
        Check if circuit breaker is active.
        
        Args:
            error_code: Error code
        
        Returns:
            True if circuit is broken (open), False otherwise
        """
        if error_code not in self.circuit_breaker_status:
            return False
        
        is_broken, reset_time = self.circuit_breaker_status[error_code]
        
        # Check if reset time has passed
        if datetime.now() >= reset_time:
            self.deactivate_circuit_breaker(error_code)
            return False
        
        return is_broken
    
    def get_next_retry_delay(
        self,
        classification: ErrorClassification,
        attempt: int,
        base_delay: Optional[float] = None,
    ) -> float:
        """
        Calculate delay for next retry attempt.
        
        Args:
            classification: Error classification
            attempt: Current attempt number
            base_delay: Override base delay (seconds)
        
        Returns:
            Delay in seconds before retry
        """
        delay = base_delay or classification.suggested_delay
        
        # Exponential backoff with jitter
        # delay = base_delay * (2 ^ (attempt - 1)) + jitter
        import random
        
        exponential_delay = delay * (2 ** (attempt - 1))
        jitter = random.uniform(0, exponential_delay * 0.1)  # 10% jitter
        
        return exponential_delay + jitter


# ============================================================================
# ERROR HANDLER FACADE
# ============================================================================

class ErrorHandler:
    """
    High-level error handler combining classification, logging, and recovery.
    
    Provides convenient interface for handling exceptions throughout application.
    """
    
    def __init__(self):
        """Initialize error handler."""
        self.classifier = ErrorClassifier()
        self.logger = StructuredErrorLogger()
        self.recovery_engine = RecoveryDecisionEngine()
    
    def handle_exception(
        self,
        exception: Exception,
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> ErrorClassification:
        """
        Handle an exception with full processing pipeline.
        
        Args:
            exception: Exception to handle
            additional_context: Additional context to include in logs
        
        Returns:
            ErrorClassification for the exception
        """
        classification = self.classifier.classify(exception)
        
        if classification:
            self.logger.log_exception(
                exception,
                classification=classification,
                additional_context=additional_context,
            )
        else:
            # Non-trader exception
            self.logger.log_exception(exception)
        
        return classification
    
    def should_handle_recovery(
        self,
        exception: Exception,
        attempt: int = 1,
    ) -> Tuple[bool, Optional[ErrorClassification]]:
        """
        Determine if exception should trigger recovery.
        
        Args:
            exception: Exception to evaluate
            attempt: Current attempt number
        
        Returns:
            Tuple of (should_recover, classification)
        """
        classification = self.classifier.classify(exception)
        
        if not classification:
            return False, None
        
        if not classification.should_retry():
            return False, classification
        
        if isinstance(exception, TraderException):
            error_code = exception.context.error_code
            should_retry = self.recovery_engine.should_retry(
                classification,
                error_code,
            )
            return should_retry, classification
        
        return False, classification
    
    def record_recovery_attempt(
        self,
        exception: Exception,
        recovery_action: ErrorRecovery,
    ) -> Optional[float]:
        """
        Record a recovery attempt and calculate retry delay.
        
        Args:
            exception: Exception being recovered
            recovery_action: Recovery action being taken
        
        Returns:
            Delay before retry (if applicable), None otherwise
        """
        if not isinstance(exception, TraderException):
            return None
        
        error_code = exception.context.error_code
        classification = self.classifier.classify(exception)
        
        # Only record retry if:
        # 1. We have a classification
        # 2. Recovery action is RETRY
        # 3. The error is actually retryable
        if not classification or recovery_action != ErrorRecovery.RETRY:
            return None
        
        if not classification.is_retryable:
            return None
        
        self.recovery_engine.record_retry_attempt(error_code)
        
        history = self.recovery_engine.retry_history.get(error_code, [])
        attempt = len(history)
        
        delay = self.recovery_engine.get_next_retry_delay(
            classification,
            attempt,
        )
        
        self.logger.log_recovery_attempt(
            error_code,
            attempt,
            recovery_action,
            delay=delay,
        )
        
        return delay


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_error_handler: Optional[ErrorHandler] = None


def get_error_handler() -> ErrorHandler:
    """Get or create global error handler instance."""
    global _error_handler
    
    if _error_handler is None:
        _error_handler = ErrorHandler()
    
    return _error_handler


def reset_error_handler() -> None:
    """Reset global error handler (for testing)."""
    global _error_handler
    _error_handler = None
