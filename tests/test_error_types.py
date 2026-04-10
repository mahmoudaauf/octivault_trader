"""
Test Suite for Error Types Module

Tests exception hierarchy, error contexts, and error code registry.
Validates all error types instantiate correctly and carry proper context.
"""

import pytest
from datetime import datetime
from typing import Dict, Any

from core.error_types import (
    ErrorSeverity,
    ErrorCategory,
    ErrorRecovery,
    ErrorContext,
    TraderException,
    # Bootstrap errors
    BootstrapError,
    BootstrapTimeoutError,
    BootstrapValidationError,
    BootstrapResourceError,
    # Arbitration errors
    ArbitrationError,
    GateValidationError,
    SignalValidationError,
    ConfidenceThresholdError,
    # Lifecycle errors
    LifecycleError,
    InvalidStateTransitionError,
    SymbolNotReadyError,
    SymbolLockError,
    # Execution errors
    ExecutionError,
    OrderPlacementError,
    InsufficientBalanceError,
    MinNotionalViolationError,
    OrderValidationError,
    DuplicateOrderError,
    # Exchange errors
    ExchangeError,
    ExchangeAPIError,
    ExchangeRateLimitError,
    ExchangeAuthenticationError,
    ExchangeInvalidPairError,
    ExchangeInsufficientLiquidityError,
    # State errors
    StateError,
    StateSyncError,
    StateLockError,
    StateCorruptionError,
    StateConsistencyError,
    # Network errors
    NetworkError,
    ConnectionTimeoutError,
    ConnectionRefusedError,
    ConnectionResetError,
    DNSResolutionError,
    # Validation errors
    ValidationError,
    InvalidParameterError,
    MissingParameterError,
    TypeMismatchError,
    RangeError,
    # Configuration errors
    ConfigurationError,
    InvalidConfigurationError,
    MissingConfigurationError,
    ConfigurationValidationError,
    # Resource errors
    ResourceError,
    InsufficientMemoryError,
    ResourceLimitExceededError,
    ResourceUnavailableError,
    # Utilities
    ERROR_CODES,
    create_error_context,
)


# ============================================================================
# ENUM TESTS
# ============================================================================

class TestErrorSeverity:
    """Test ErrorSeverity enum."""
    
    def test_all_severity_levels_exist(self):
        """Verify all severity levels are defined."""
        levels = {e.value for e in ErrorSeverity}
        assert len(levels) == 5
        assert "debug" in levels
        assert "info" in levels
        assert "warning" in levels
        assert "error" in levels
        assert "critical" in levels
    
    def test_severity_comparison(self):
        """Verify severity enum values."""
        assert ErrorSeverity.DEBUG.value == "debug"
        assert ErrorSeverity.CRITICAL.value == "critical"


class TestErrorCategory:
    """Test ErrorCategory enum."""
    
    def test_all_categories_exist(self):
        """Verify all categories are defined."""
        categories = {e.value for e in ErrorCategory}
        assert len(categories) == 10
        assert "bootstrap" in categories
        assert "arbitration" in categories
        assert "lifecycle" in categories
        assert "execution" in categories
        assert "exchange" in categories
        assert "state" in categories
        assert "network" in categories
        assert "validation" in categories
        assert "configuration" in categories
        assert "resource" in categories


class TestErrorRecovery:
    """Test ErrorRecovery enum."""
    
    def test_all_recovery_strategies_exist(self):
        """Verify all recovery strategies are defined."""
        strategies = {e.value for e in ErrorRecovery}
        assert len(strategies) == 7
        assert "none" in strategies
        assert "retry" in strategies
        assert "fallback" in strategies
        assert "skip" in strategies
        assert "reset" in strategies
        assert "circuit_break" in strategies
        assert "escalate" in strategies


# ============================================================================
# ERROR CONTEXT TESTS
# ============================================================================

class TestErrorContext:
    """Test ErrorContext class."""
    
    def test_create_minimal_context(self):
        """Create context with required fields only."""
        context = ErrorContext(
            category=ErrorCategory.BOOTSTRAP,
            severity=ErrorSeverity.ERROR,
            recovery_strategy=ErrorRecovery.RETRY,
            error_code="BOOTSTRAP_TIMEOUT",
            message="Bootstrap timed out",
            timestamp=datetime.now(),
        )
        assert context.category == ErrorCategory.BOOTSTRAP
        assert context.severity == ErrorSeverity.ERROR
        assert context.recovery_strategy == ErrorRecovery.RETRY
        assert context.error_code == "BOOTSTRAP_TIMEOUT"
        assert context.metadata == {}
    
    def test_create_full_context(self):
        """Create context with all fields."""
        now = datetime.now()
        metadata = {"attempts": 3, "delay": 5}
        context = ErrorContext(
            category=ErrorCategory.EXECUTION,
            severity=ErrorSeverity.CRITICAL,
            recovery_strategy=ErrorRecovery.ESCALATE,
            error_code="ORDER_PLACEMENT",
            message="Order placement failed",
            timestamp=now,
            operation="place_order",
            component="execution_manager",
            symbol="BTC/USDT",
            metadata=metadata,
        )
        assert context.operation == "place_order"
        assert context.component == "execution_manager"
        assert context.symbol == "BTC/USDT"
        assert context.metadata["attempts"] == 3
    
    def test_context_to_dict(self):
        """Convert context to dictionary."""
        context = create_error_context(
            category=ErrorCategory.EXCHANGE,
            severity=ErrorSeverity.WARNING,
            error_code="EXCHANGE_RATE_LIMIT",
            message="Rate limited",
            recovery_strategy=ErrorRecovery.RETRY,
            symbol="ETH/USDT",
            metadata={"retry_after": 60},
        )
        d = context.to_dict()
        assert d["category"] == "exchange"
        assert d["severity"] == "warning"
        assert d["recovery"] == "retry"
        assert d["code"] == "EXCHANGE_RATE_LIMIT"
        assert d["symbol"] == "ETH/USDT"
        assert d["metadata"]["retry_after"] == 60
        assert "timestamp" in d
    
    def test_context_initialization_with_none_metadata(self):
        """Verify metadata defaults to empty dict."""
        context = ErrorContext(
            category=ErrorCategory.STATE,
            severity=ErrorSeverity.ERROR,
            recovery_strategy=ErrorRecovery.RESET,
            error_code="STATE_SYNC",
            message="State sync failed",
            timestamp=datetime.now(),
            metadata=None,
        )
        assert context.metadata == {}
        assert isinstance(context.metadata, dict)


# ============================================================================
# TRADER EXCEPTION TESTS
# ============================================================================

class TestTraderException:
    """Test TraderException base class."""
    
    def test_create_exception(self):
        """Create a TraderException instance."""
        context = create_error_context(
            category=ErrorCategory.BOOTSTRAP,
            severity=ErrorSeverity.ERROR,
            error_code="BOOTSTRAP_TIMEOUT",
            message="Bootstrap timed out",
        )
        exc = TraderException(context)
        assert exc.context == context
        assert exc.cause is None
        assert "[BOOTSTRAP_TIMEOUT]" in str(exc)
    
    def test_exception_with_cause(self):
        """Create exception with original cause."""
        context = create_error_context(
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.ERROR,
            error_code="NETWORK_TIMEOUT",
            message="Network timeout",
        )
        original_error = ConnectionError("Connection failed")
        exc = TraderException(context, cause=original_error)
        assert exc.cause == original_error
    
    def test_exception_message_formatting_minimal(self):
        """Verify exception message format without optional fields."""
        context = create_error_context(
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.WARNING,
            error_code="INVALID_PARAMETER",
            message="Invalid value",
        )
        exc = TraderException(context)
        msg = str(exc)
        assert "[INVALID_PARAMETER]" in msg
        assert "(WARNING)" in msg
        assert "Invalid value" in msg
    
    def test_exception_message_formatting_full(self):
        """Verify exception message format with all fields."""
        context = create_error_context(
            category=ErrorCategory.EXECUTION,
            severity=ErrorSeverity.CRITICAL,
            error_code="ORDER_PLACEMENT",
            message="Order failed",
            operation="place_order",
            component="execution_manager",
            symbol="BTC/USDT",
        )
        exc = TraderException(context)
        msg = str(exc)
        assert "[ORDER_PLACEMENT]" in msg
        assert "(CRITICAL)" in msg
        assert "Order failed" in msg
        assert "Op: place_order" in msg
        assert "Component: execution_manager" in msg
        assert "Symbol: BTC/USDT" in msg
    
    def test_is_retryable_true(self):
        """Test is_retryable() returns True for retryable errors."""
        context = create_error_context(
            category=ErrorCategory.EXCHANGE,
            severity=ErrorSeverity.ERROR,
            error_code="EXCHANGE_API",
            message="API error",
            recovery_strategy=ErrorRecovery.RETRY,
        )
        exc = TraderException(context)
        assert exc.is_retryable()
    
    def test_is_retryable_false(self):
        """Test is_retryable() returns False for non-retryable errors."""
        context = create_error_context(
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.ERROR,
            error_code="INVALID_PARAMETER",
            message="Invalid value",
            recovery_strategy=ErrorRecovery.NONE,
        )
        exc = TraderException(context)
        assert not exc.is_retryable()
    
    def test_is_critical_true(self):
        """Test is_critical() returns True for critical errors."""
        context = create_error_context(
            category=ErrorCategory.STATE,
            severity=ErrorSeverity.CRITICAL,
            error_code="STATE_CORRUPTION",
            message="State corrupted",
        )
        exc = TraderException(context)
        assert exc.is_critical()
    
    def test_is_critical_false(self):
        """Test is_critical() returns False for non-critical errors."""
        context = create_error_context(
            category=ErrorCategory.EXCHANGE,
            severity=ErrorSeverity.WARNING,
            error_code="EXCHANGE_RATE_LIMIT",
            message="Rate limited",
        )
        exc = TraderException(context)
        assert not exc.is_critical()


# ============================================================================
# EXCEPTION HIERARCHY TESTS
# ============================================================================

class TestBootstrapErrorHierarchy:
    """Test Bootstrap error classes."""
    
    def test_bootstrap_error_instantiation(self):
        """Create BootstrapError."""
        context = create_error_context(
            category=ErrorCategory.BOOTSTRAP,
            severity=ErrorSeverity.ERROR,
            error_code="BOOTSTRAP_TIMEOUT",
            message="Bootstrap timed out",
        )
        exc = BootstrapError(context)
        assert isinstance(exc, TraderException)
        assert isinstance(exc, Exception)
    
    def test_bootstrap_timeout_error(self):
        """Create BootstrapTimeoutError."""
        context = create_error_context(
            category=ErrorCategory.BOOTSTRAP,
            severity=ErrorSeverity.CRITICAL,
            error_code="BOOTSTRAP_TIMEOUT",
            message="Bootstrap timeout",
        )
        exc = BootstrapTimeoutError(context)
        assert isinstance(exc, BootstrapError)
    
    def test_bootstrap_validation_error(self):
        """Create BootstrapValidationError."""
        context = create_error_context(
            category=ErrorCategory.BOOTSTRAP,
            severity=ErrorSeverity.ERROR,
            error_code="BOOTSTRAP_VALIDATION",
            message="Validation failed",
        )
        exc = BootstrapValidationError(context)
        assert isinstance(exc, BootstrapError)
    
    def test_bootstrap_resource_error(self):
        """Create BootstrapResourceError."""
        context = create_error_context(
            category=ErrorCategory.BOOTSTRAP,
            severity=ErrorSeverity.CRITICAL,
            error_code="BOOTSTRAP_RESOURCE",
            message="Resource allocation failed",
        )
        exc = BootstrapResourceError(context)
        assert isinstance(exc, BootstrapError)


class TestArbitrationErrorHierarchy:
    """Test Arbitration error classes."""
    
    def test_gate_validation_error(self):
        """Create GateValidationError."""
        context = create_error_context(
            category=ErrorCategory.ARBITRATION,
            severity=ErrorSeverity.WARNING,
            error_code="GATE_VALIDATION",
            message="Gate check failed",
        )
        exc = GateValidationError(context)
        assert isinstance(exc, ArbitrationError)
    
    def test_signal_validation_error(self):
        """Create SignalValidationError."""
        context = create_error_context(
            category=ErrorCategory.ARBITRATION,
            severity=ErrorSeverity.ERROR,
            error_code="SIGNAL_VALIDATION",
            message="Signal validation failed",
        )
        exc = SignalValidationError(context)
        assert isinstance(exc, ArbitrationError)
    
    def test_confidence_threshold_error(self):
        """Create ConfidenceThresholdError."""
        context = create_error_context(
            category=ErrorCategory.ARBITRATION,
            severity=ErrorSeverity.INFO,
            error_code="CONFIDENCE_THRESHOLD",
            message="Confidence too low",
        )
        exc = ConfidenceThresholdError(context)
        assert isinstance(exc, ArbitrationError)


class TestLifecycleErrorHierarchy:
    """Test Lifecycle error classes."""
    
    def test_invalid_state_transition_error(self):
        """Create InvalidStateTransitionError."""
        context = create_error_context(
            category=ErrorCategory.LIFECYCLE,
            severity=ErrorSeverity.ERROR,
            error_code="INVALID_STATE_TRANSITION",
            message="Invalid transition",
            symbol="BTC/USDT",
        )
        exc = InvalidStateTransitionError(context)
        assert isinstance(exc, LifecycleError)
    
    def test_symbol_not_ready_error(self):
        """Create SymbolNotReadyError."""
        context = create_error_context(
            category=ErrorCategory.LIFECYCLE,
            severity=ErrorSeverity.WARNING,
            error_code="SYMBOL_NOT_READY",
            message="Symbol not ready",
            symbol="ETH/USDT",
        )
        exc = SymbolNotReadyError(context)
        assert isinstance(exc, LifecycleError)
    
    def test_symbol_lock_error(self):
        """Create SymbolLockError."""
        context = create_error_context(
            category=ErrorCategory.LIFECYCLE,
            severity=ErrorSeverity.ERROR,
            error_code="SYMBOL_LOCK",
            message="Lock timeout",
            symbol="ADA/USDT",
        )
        exc = SymbolLockError(context)
        assert isinstance(exc, LifecycleError)


class TestExecutionErrorHierarchy:
    """Test Execution error classes."""
    
    def test_order_placement_error(self):
        """Create OrderPlacementError."""
        context = create_error_context(
            category=ErrorCategory.EXECUTION,
            severity=ErrorSeverity.ERROR,
            error_code="ORDER_PLACEMENT",
            message="Placement failed",
            symbol="BTC/USDT",
        )
        exc = OrderPlacementError(context)
        assert isinstance(exc, ExecutionError)
    
    def test_insufficient_balance_error(self):
        """Create InsufficientBalanceError."""
        context = create_error_context(
            category=ErrorCategory.EXECUTION,
            severity=ErrorSeverity.WARNING,
            error_code="INSUFFICIENT_BALANCE",
            message="Not enough balance",
            metadata={"required": 100, "available": 50},
        )
        exc = InsufficientBalanceError(context)
        assert isinstance(exc, ExecutionError)
    
    def test_min_notional_violation_error(self):
        """Create MinNotionalViolationError."""
        context = create_error_context(
            category=ErrorCategory.EXECUTION,
            severity=ErrorSeverity.WARNING,
            error_code="MIN_NOTIONAL_VIOLATION",
            message="Below minimum notional",
            symbol="BTC/USDT",
        )
        exc = MinNotionalViolationError(context)
        assert isinstance(exc, ExecutionError)
    
    def test_order_validation_error(self):
        """Create OrderValidationError."""
        context = create_error_context(
            category=ErrorCategory.EXECUTION,
            severity=ErrorSeverity.ERROR,
            error_code="ORDER_VALIDATION",
            message="Validation failed",
        )
        exc = OrderValidationError(context)
        assert isinstance(exc, ExecutionError)
    
    def test_duplicate_order_error(self):
        """Create DuplicateOrderError."""
        context = create_error_context(
            category=ErrorCategory.EXECUTION,
            severity=ErrorSeverity.WARNING,
            error_code="DUPLICATE_ORDER",
            message="Duplicate detected",
        )
        exc = DuplicateOrderError(context)
        assert isinstance(exc, ExecutionError)


class TestExchangeErrorHierarchy:
    """Test Exchange error classes."""
    
    def test_exchange_api_error(self):
        """Create ExchangeAPIError."""
        context = create_error_context(
            category=ErrorCategory.EXCHANGE,
            severity=ErrorSeverity.ERROR,
            error_code="EXCHANGE_API",
            message="API error",
            recovery_strategy=ErrorRecovery.RETRY,
        )
        exc = ExchangeAPIError(context)
        assert isinstance(exc, ExchangeError)
    
    def test_exchange_rate_limit_error(self):
        """Create ExchangeRateLimitError."""
        context = create_error_context(
            category=ErrorCategory.EXCHANGE,
            severity=ErrorSeverity.WARNING,
            error_code="EXCHANGE_RATE_LIMIT",
            message="Rate limited",
            recovery_strategy=ErrorRecovery.RETRY,
            metadata={"retry_after": 60},
        )
        exc = ExchangeRateLimitError(context)
        assert isinstance(exc, ExchangeError)
        assert exc.is_retryable()
    
    def test_exchange_authentication_error(self):
        """Create ExchangeAuthenticationError."""
        context = create_error_context(
            category=ErrorCategory.EXCHANGE,
            severity=ErrorSeverity.CRITICAL,
            error_code="EXCHANGE_AUTH",
            message="Authentication failed",
        )
        exc = ExchangeAuthenticationError(context)
        assert isinstance(exc, ExchangeError)
    
    def test_exchange_invalid_pair_error(self):
        """Create ExchangeInvalidPairError."""
        context = create_error_context(
            category=ErrorCategory.EXCHANGE,
            severity=ErrorSeverity.ERROR,
            error_code="EXCHANGE_INVALID_PAIR",
            message="Invalid pair",
            symbol="XYZ/ABC",
        )
        exc = ExchangeInvalidPairError(context)
        assert isinstance(exc, ExchangeError)
    
    def test_exchange_insufficient_liquidity_error(self):
        """Create ExchangeInsufficientLiquidityError."""
        context = create_error_context(
            category=ErrorCategory.EXCHANGE,
            severity=ErrorSeverity.WARNING,
            error_code="EXCHANGE_LIQUIDITY",
            message="Low liquidity",
        )
        exc = ExchangeInsufficientLiquidityError(context)
        assert isinstance(exc, ExchangeError)


class TestStateErrorHierarchy:
    """Test State error classes."""
    
    def test_state_sync_error(self):
        """Create StateSyncError."""
        context = create_error_context(
            category=ErrorCategory.STATE,
            severity=ErrorSeverity.CRITICAL,
            error_code="STATE_SYNC",
            message="Sync failed",
        )
        exc = StateSyncError(context)
        assert isinstance(exc, StateError)
    
    def test_state_lock_error(self):
        """Create StateLockError."""
        context = create_error_context(
            category=ErrorCategory.STATE,
            severity=ErrorSeverity.ERROR,
            error_code="STATE_LOCK",
            message="Lock failed",
        )
        exc = StateLockError(context)
        assert isinstance(exc, StateError)
    
    def test_state_corruption_error(self):
        """Create StateCorruptionError."""
        context = create_error_context(
            category=ErrorCategory.STATE,
            severity=ErrorSeverity.CRITICAL,
            error_code="STATE_CORRUPTION",
            message="Corruption detected",
        )
        exc = StateCorruptionError(context)
        assert isinstance(exc, StateError)
        assert exc.is_critical()
    
    def test_state_consistency_error(self):
        """Create StateConsistencyError."""
        context = create_error_context(
            category=ErrorCategory.STATE,
            severity=ErrorSeverity.ERROR,
            error_code="STATE_CONSISTENCY",
            message="Consistency check failed",
        )
        exc = StateConsistencyError(context)
        assert isinstance(exc, StateError)


class TestNetworkErrorHierarchy:
    """Test Network error classes."""
    
    def test_connection_timeout_error(self):
        """Create ConnectionTimeoutError."""
        context = create_error_context(
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.WARNING,
            error_code="NETWORK_TIMEOUT",
            message="Connection timeout",
            recovery_strategy=ErrorRecovery.RETRY,
        )
        exc = ConnectionTimeoutError(context)
        assert isinstance(exc, NetworkError)
        assert exc.is_retryable()
    
    def test_connection_refused_error(self):
        """Create ConnectionRefusedError."""
        context = create_error_context(
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.ERROR,
            error_code="CONNECTION_REFUSED",
            message="Connection refused",
        )
        exc = ConnectionRefusedError(context)
        assert isinstance(exc, NetworkError)
    
    def test_connection_reset_error(self):
        """Create ConnectionResetError."""
        context = create_error_context(
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.WARNING,
            error_code="CONNECTION_RESET",
            message="Connection reset",
        )
        exc = ConnectionResetError(context)
        assert isinstance(exc, NetworkError)
    
    def test_dns_resolution_error(self):
        """Create DNSResolutionError."""
        context = create_error_context(
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.ERROR,
            error_code="DNS_RESOLUTION",
            message="DNS resolution failed",
        )
        exc = DNSResolutionError(context)
        assert isinstance(exc, NetworkError)


class TestValidationErrorHierarchy:
    """Test Validation error classes."""
    
    def test_invalid_parameter_error(self):
        """Create InvalidParameterError."""
        context = create_error_context(
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.WARNING,
            error_code="INVALID_PARAMETER",
            message="Invalid value",
        )
        exc = InvalidParameterError(context)
        assert isinstance(exc, ValidationError)
    
    def test_missing_parameter_error(self):
        """Create MissingParameterError."""
        context = create_error_context(
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.ERROR,
            error_code="MISSING_PARAMETER",
            message="Parameter missing",
        )
        exc = MissingParameterError(context)
        assert isinstance(exc, ValidationError)
    
    def test_type_mismatch_error(self):
        """Create TypeMismatchError."""
        context = create_error_context(
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.ERROR,
            error_code="TYPE_MISMATCH",
            message="Type mismatch",
        )
        exc = TypeMismatchError(context)
        assert isinstance(exc, ValidationError)
    
    def test_range_error(self):
        """Create RangeError."""
        context = create_error_context(
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.WARNING,
            error_code="RANGE_ERROR",
            message="Out of range",
        )
        exc = RangeError(context)
        assert isinstance(exc, ValidationError)


class TestConfigurationErrorHierarchy:
    """Test Configuration error classes."""
    
    def test_invalid_configuration_error(self):
        """Create InvalidConfigurationError."""
        context = create_error_context(
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.CRITICAL,
            error_code="INVALID_CONFIG",
            message="Invalid config",
        )
        exc = InvalidConfigurationError(context)
        assert isinstance(exc, ConfigurationError)
    
    def test_missing_configuration_error(self):
        """Create MissingConfigurationError."""
        context = create_error_context(
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.CRITICAL,
            error_code="MISSING_CONFIG",
            message="Missing config",
        )
        exc = MissingConfigurationError(context)
        assert isinstance(exc, ConfigurationError)
    
    def test_configuration_validation_error(self):
        """Create ConfigurationValidationError."""
        context = create_error_context(
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.ERROR,
            error_code="CONFIG_VALIDATION",
            message="Validation failed",
        )
        exc = ConfigurationValidationError(context)
        assert isinstance(exc, ConfigurationError)


class TestResourceErrorHierarchy:
    """Test Resource error classes."""
    
    def test_insufficient_memory_error(self):
        """Create InsufficientMemoryError."""
        context = create_error_context(
            category=ErrorCategory.RESOURCE,
            severity=ErrorSeverity.CRITICAL,
            error_code="INSUFFICIENT_MEMORY",
            message="Not enough memory",
        )
        exc = InsufficientMemoryError(context)
        assert isinstance(exc, ResourceError)
    
    def test_resource_limit_exceeded_error(self):
        """Create ResourceLimitExceededError."""
        context = create_error_context(
            category=ErrorCategory.RESOURCE,
            severity=ErrorSeverity.ERROR,
            error_code="RESOURCE_LIMIT",
            message="Limit exceeded",
        )
        exc = ResourceLimitExceededError(context)
        assert isinstance(exc, ResourceError)
    
    def test_resource_unavailable_error(self):
        """Create ResourceUnavailableError."""
        context = create_error_context(
            category=ErrorCategory.RESOURCE,
            severity=ErrorSeverity.WARNING,
            error_code="RESOURCE_UNAVAILABLE",
            message="Resource unavailable",
        )
        exc = ResourceUnavailableError(context)
        assert isinstance(exc, ResourceError)


# ============================================================================
# ERROR CODE REGISTRY TESTS
# ============================================================================

class TestErrorCodeRegistry:
    """Test ERROR_CODES registry."""
    
    def test_error_codes_not_empty(self):
        """Verify ERROR_CODES registry is not empty."""
        assert len(ERROR_CODES) > 0
    
    def test_error_codes_all_strings(self):
        """Verify all error codes are strings."""
        for code, description in ERROR_CODES.items():
            assert isinstance(code, str)
            assert isinstance(description, str)
            assert len(code) > 0
            assert len(description) > 0
    
    def test_error_codes_unique_descriptions(self):
        """Verify error code descriptions are unique or reasonable."""
        descriptions = list(ERROR_CODES.values())
        # Allow some duplicates due to similar errors, but check basic distribution
        assert len(set(descriptions)) > len(descriptions) * 0.7
    
    def test_bootstrap_error_codes(self):
        """Verify bootstrap error codes are registered."""
        assert "BOOTSTRAP_TIMEOUT" in ERROR_CODES
        assert "BOOTSTRAP_VALIDATION" in ERROR_CODES
        assert "BOOTSTRAP_RESOURCE" in ERROR_CODES
    
    def test_exchange_error_codes(self):
        """Verify exchange error codes are registered."""
        assert "EXCHANGE_API" in ERROR_CODES
        assert "EXCHANGE_RATE_LIMIT" in ERROR_CODES
        assert "EXCHANGE_AUTH" in ERROR_CODES


# ============================================================================
# UTILITY FUNCTION TESTS
# ============================================================================

class TestCreateErrorContext:
    """Test create_error_context utility function."""
    
    def test_create_minimal_context(self):
        """Create context with required parameters."""
        ctx = create_error_context(
            category=ErrorCategory.BOOTSTRAP,
            severity=ErrorSeverity.ERROR,
            error_code="BOOTSTRAP_TIMEOUT",
            message="Timed out",
        )
        assert ctx.category == ErrorCategory.BOOTSTRAP
        assert ctx.severity == ErrorSeverity.ERROR
        assert ctx.error_code == "BOOTSTRAP_TIMEOUT"
        assert ctx.recovery_strategy == ErrorRecovery.NONE
    
    def test_create_full_context(self):
        """Create context with all parameters."""
        metadata = {"key": "value"}
        ctx = create_error_context(
            category=ErrorCategory.EXECUTION,
            severity=ErrorSeverity.CRITICAL,
            error_code="ORDER_PLACEMENT",
            message="Failed",
            recovery_strategy=ErrorRecovery.RETRY,
            operation="place_order",
            component="executor",
            symbol="BTC/USDT",
            metadata=metadata,
        )
        assert ctx.operation == "place_order"
        assert ctx.component == "executor"
        assert ctx.symbol == "BTC/USDT"
        assert ctx.metadata["key"] == "value"
    
    def test_create_context_timestamp(self):
        """Verify timestamp is set automatically."""
        before = datetime.now()
        ctx = create_error_context(
            category=ErrorCategory.STATE,
            severity=ErrorSeverity.ERROR,
            error_code="STATE_SYNC",
            message="Sync failed",
        )
        after = datetime.now()
        assert before <= ctx.timestamp <= after


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestErrorIntegration:
    """Integration tests for error system."""
    
    def test_exception_inheritance_chain(self):
        """Verify exception hierarchy is correct."""
        context = create_error_context(
            category=ErrorCategory.EXCHANGE,
            severity=ErrorSeverity.ERROR,
            error_code="EXCHANGE_RATE_LIMIT",
            message="Rate limited",
        )
        exc = ExchangeRateLimitError(context)
        
        # Check inheritance chain
        assert isinstance(exc, ExchangeRateLimitError)
        assert isinstance(exc, ExchangeError)
        assert isinstance(exc, TraderException)
        assert isinstance(exc, Exception)
    
    def test_catch_by_base_type(self):
        """Verify can catch specific error by base class."""
        context = create_error_context(
            category=ErrorCategory.EXECUTION,
            severity=ErrorSeverity.ERROR,
            error_code="ORDER_PLACEMENT",
            message="Failed",
        )
        exc = OrderPlacementError(context)
        
        # Should be catchable by parent class
        try:
            raise exc
        except ExecutionError:
            pass  # Expected
        except Exception:
            pytest.fail("Should be caught as ExecutionError")
    
    def test_error_recovery_workflow(self):
        """Test error recovery decision workflow."""
        # Retryable error
        ctx_retry = create_error_context(
            category=ErrorCategory.EXCHANGE,
            severity=ErrorSeverity.WARNING,
            error_code="EXCHANGE_RATE_LIMIT",
            message="Rate limited",
            recovery_strategy=ErrorRecovery.RETRY,
        )
        exc_retry = ExchangeRateLimitError(ctx_retry)
        assert exc_retry.is_retryable()
        
        # Non-retryable error
        ctx_no_retry = create_error_context(
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.ERROR,
            error_code="INVALID_PARAMETER",
            message="Invalid",
            recovery_strategy=ErrorRecovery.NONE,
        )
        exc_no_retry = InvalidParameterError(ctx_no_retry)
        assert not exc_no_retry.is_retryable()
    
    def test_error_severity_escalation(self):
        """Test error severity levels."""
        warnings = [ErrorSeverity.DEBUG, ErrorSeverity.INFO, ErrorSeverity.WARNING]
        errors = [ErrorSeverity.ERROR, ErrorSeverity.CRITICAL]
        
        for severity in warnings:
            ctx = create_error_context(
                category=ErrorCategory.BOOTSTRAP,
                severity=severity,
                error_code="TEST",
                message="Test",
            )
            assert not TraderException(ctx).is_critical()
        
        for severity in errors:
            ctx = create_error_context(
                category=ErrorCategory.BOOTSTRAP,
                severity=severity,
                error_code="TEST",
                message="Test",
            )
            # Only CRITICAL is critical
            if severity == ErrorSeverity.CRITICAL:
                assert TraderException(ctx).is_critical()


# ============================================================================
# DOCUMENTATION TESTS
# ============================================================================

class TestErrorDocumentation:
    """Tests verifying error types are properly documented."""
    
    def test_error_severity_has_docstrings(self):
        """Verify ErrorSeverity members have documentation."""
        for member in ErrorSeverity:
            assert member.__doc__ is not None
            assert len(member.__doc__) > 0
    
    def test_error_category_has_docstrings(self):
        """Verify ErrorCategory members have documentation."""
        for member in ErrorCategory:
            assert member.__doc__ is not None
            assert len(member.__doc__) > 0
    
    def test_error_recovery_has_docstrings(self):
        """Verify ErrorRecovery members have documentation."""
        for member in ErrorRecovery:
            assert member.__doc__ is not None
            assert len(member.__doc__) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
