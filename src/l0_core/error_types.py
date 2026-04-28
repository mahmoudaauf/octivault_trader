"""
Error Types and Exception Hierarchy Module

Defines all custom exception types used throughout the application.
Replaces broad Exception handlers with specific, typed exceptions.

Philosophy:
  • Every error has a specific type
  • Errors include severity levels
  • Errors carry context (what, why, recovery suggestion)
  • Type-safe exception handling
  • Enables predictable error recovery
"""

from enum import Enum
from typing import Optional, Dict, Any, Final
from dataclasses import dataclass
from datetime import datetime


class ErrorSeverity(Enum):
    """Severity levels for all errors."""
    
    DEBUG = "debug"
    """Diagnostic information - doesn't require action"""
    
    INFO = "info"
    """Informational - expected condition, no action needed"""
    
    WARNING = "warning"
    """Warning - unusual condition, may need monitoring"""
    
    ERROR = "error"
    """Error - operation failed, recovery attempted"""
    
    CRITICAL = "critical"
    """Critical - system integrity at risk, immediate action needed"""


class ErrorCategory(Enum):
    """Categories of errors by origin."""
    
    BOOTSTRAP = "bootstrap"
    """Bootstrap/initialization errors"""
    
    ARBITRATION = "arbitration"
    """Signal arbitration/validation errors"""
    
    LIFECYCLE = "lifecycle"
    """Symbol lifecycle state machine errors"""
    
    EXECUTION = "execution"
    """Trade execution errors"""
    
    EXCHANGE = "exchange"
    """Exchange API errors"""
    
    STATE = "state"
    """State management errors"""
    
    NETWORK = "network"
    """Network/connectivity errors"""
    
    VALIDATION = "validation"
    """Input validation errors"""
    
    CONFIGURATION = "configuration"
    """Configuration/setup errors"""
    
    RESOURCE = "resource"
    """Resource allocation errors (memory, handles, etc)"""


class ErrorRecovery(Enum):
    """Recovery strategies for different error types."""
    
    NONE = "none"
    """No recovery - error must be escalated"""
    
    RETRY = "retry"
    """Retry the operation after delay"""
    
    FALLBACK = "fallback"
    """Use fallback/default behavior"""
    
    SKIP = "skip"
    """Skip this operation and continue"""
    
    RESET = "reset"
    """Reset state and retry"""
    
    CIRCUIT_BREAK = "circuit_break"
    """Stop operations temporarily (circuit breaker)"""
    
    ESCALATE = "escalate"
    """Escalate to higher-level handler"""


@dataclass
class ErrorContext:
    """Context information for all errors."""
    
    category: ErrorCategory
    """What category of error this is"""
    
    severity: ErrorSeverity
    """How severe this error is"""
    
    recovery_strategy: ErrorRecovery
    """Recommended recovery action"""
    
    error_code: str
    """Unique error code for tracking"""
    
    message: str
    """Human-readable error message"""
    
    timestamp: datetime
    """When the error occurred"""
    
    operation: Optional[str] = None
    """What operation was being performed"""
    
    component: Optional[str] = None
    """Which component produced the error"""
    
    symbol: Optional[str] = None
    """Trading symbol (if applicable)"""
    
    metadata: Dict[str, Any] = None
    """Additional context data"""
    
    def __post_init__(self):
        """Initialize metadata dict."""
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for logging."""
        return {
            "category": self.category.value,
            "severity": self.severity.value,
            "recovery": self.recovery_strategy.value,
            "code": self.error_code,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "operation": self.operation,
            "component": self.component,
            "symbol": self.symbol,
            "metadata": self.metadata,
        }


# ============================================================================
# BASE EXCEPTION CLASS
# ============================================================================

class TraderException(Exception):
    """
    Base exception class for all application errors.
    
    All custom exceptions should inherit from this class.
    Carries error context for debugging and recovery.
    """
    
    def __init__(
        self,
        context: ErrorContext,
        cause: Optional[Exception] = None,
    ):
        """
        Initialize exception with context.
        
        Args:
            context: ErrorContext with error details
            cause: Original exception that caused this (if any)
        """
        self.context = context
        self.cause = cause
        super().__init__(self._format_message())
    
    def _format_message(self) -> str:
        """Format exception message with context."""
        parts = [
            f"[{self.context.error_code}]",
            f"({self.context.severity.value.upper()})",
            self.context.message,
        ]
        
        if self.context.operation:
            parts.append(f"| Op: {self.context.operation}")
        
        if self.context.component:
            parts.append(f"| Component: {self.context.component}")
        
        if self.context.symbol:
            parts.append(f"| Symbol: {self.context.symbol}")
        
        return " ".join(parts)
    
    def is_retryable(self) -> bool:
        """Check if this error can be retried."""
        return self.context.recovery_strategy == ErrorRecovery.RETRY
    
    def is_critical(self) -> bool:
        """Check if this error is critical."""
        return self.context.severity == ErrorSeverity.CRITICAL


# ============================================================================
# BOOTSTRAP ERRORS
# ============================================================================

class BootstrapError(TraderException):
    """Errors during bootstrap/initialization phase."""
    pass


class BootstrapTimeoutError(BootstrapError):
    """Bootstrap operation exceeded timeout."""
    pass


class BootstrapValidationError(BootstrapError):
    """Bootstrap validation checks failed."""
    pass


class BootstrapResourceError(BootstrapError):
    """Bootstrap resource allocation failed."""
    pass


# ============================================================================
# ARBITRATION ERRORS
# ============================================================================

class ArbitrationError(TraderException):
    """Errors during signal arbitration/validation."""
    pass


class GateValidationError(ArbitrationError):
    """Gate validation check failed."""
    pass


class SignalValidationError(ArbitrationError):
    """Signal validation failed."""
    pass


class ConfidenceThresholdError(ArbitrationError):
    """Signal confidence below threshold."""
    pass


# ============================================================================
# LIFECYCLE ERRORS
# ============================================================================

class LifecycleError(TraderException):
    """Errors in symbol lifecycle state machine."""
    pass


class InvalidStateTransitionError(LifecycleError):
    """Attempted invalid state transition."""
    pass


class SymbolNotReadyError(LifecycleError):
    """Symbol in wrong state for operation."""
    pass


class SymbolLockError(LifecycleError):
    """Could not acquire symbol lock."""
    pass


# ============================================================================
# EXECUTION ERRORS
# ============================================================================

class ExecutionError(TraderException):
    """Errors during trade execution."""
    pass


class OrderPlacementError(ExecutionError):
    """Failed to place order."""
    pass


class InsufficientBalanceError(ExecutionError):
    """Insufficient balance for operation."""
    pass


class MinNotionalViolationError(ExecutionError):
    """Order violates minimum notional value."""
    pass


class OrderValidationError(ExecutionError):
    """Order validation checks failed."""
    pass


class DuplicateOrderError(ExecutionError):
    """Attempted to place duplicate order."""
    pass


# ============================================================================
# EXCHANGE ERRORS
# ============================================================================

class ExchangeError(TraderException):
    """Errors from exchange API."""
    pass


class ExchangeAPIError(ExchangeError):
    """Exchange API call failed."""
    pass


class ExchangeRateLimitError(ExchangeError):
    """Exchange rate limit exceeded."""
    pass


class ExchangeAuthenticationError(ExchangeError):
    """Exchange authentication failed."""
    pass


class ExchangeInvalidPairError(ExchangeError):
    """Invalid trading pair for exchange."""
    pass


class ExchangeInsufficientLiquidityError(ExchangeError):
    """Insufficient liquidity for trade."""
    pass


# ============================================================================
# STATE ERRORS
# ============================================================================

class StateError(TraderException):
    """Errors in state management."""
    pass


class StateSyncError(StateError):
    """State synchronization failed."""
    pass


class StateLockError(StateError):
    """Could not acquire state lock."""
    pass


class StateCorruptionError(StateError):
    """State corruption detected."""
    pass


class StateConsistencyError(StateError):
    """State consistency check failed."""
    pass


# ============================================================================
# NETWORK ERRORS
# ============================================================================

class NetworkError(TraderException):
    """Network/connectivity errors."""
    pass


class ConnectionTimeoutError(NetworkError):
    """Network connection timeout."""
    pass


class ConnectionRefusedError(NetworkError):
    """Network connection refused."""
    pass


class ConnectionResetError(NetworkError):
    """Network connection reset."""
    pass


class DNSResolutionError(NetworkError):
    """DNS resolution failed."""
    pass


# ============================================================================
# VALIDATION ERRORS
# ============================================================================

class ValidationError(TraderException):
    """Input/parameter validation errors."""
    pass


class InvalidParameterError(ValidationError):
    """Invalid parameter value."""
    pass


class MissingParameterError(ValidationError):
    """Required parameter missing."""
    pass


class TypeMismatchError(ValidationError):
    """Type mismatch in parameter."""
    pass


class RangeError(ValidationError):
    """Value outside valid range."""
    pass


# ============================================================================
# CONFIGURATION ERRORS
# ============================================================================

class ConfigurationError(TraderException):
    """Configuration/setup errors."""
    pass


class InvalidConfigurationError(ConfigurationError):
    """Configuration is invalid."""
    pass


class MissingConfigurationError(ConfigurationError):
    """Required configuration missing."""
    pass


class ConfigurationValidationError(ConfigurationError):
    """Configuration validation failed."""
    pass


# ============================================================================
# RESOURCE ERRORS
# ============================================================================

class ResourceError(TraderException):
    """Resource allocation errors."""
    pass


class InsufficientMemoryError(ResourceError):
    """Insufficient memory available."""
    pass


class ResourceLimitExceededError(ResourceError):
    """Resource limit exceeded."""
    pass


class ResourceUnavailableError(ResourceError):
    """Required resource unavailable."""
    pass


# ============================================================================
# ERROR CODE REGISTRY
# ============================================================================

ERROR_CODES: Final[Dict[str, str]] = {
    # Bootstrap errors
    "BOOTSTRAP_TIMEOUT": "Bootstrap operation timeout",
    "BOOTSTRAP_VALIDATION": "Bootstrap validation failed",
    "BOOTSTRAP_RESOURCE": "Bootstrap resource error",
    
    # Arbitration errors
    "GATE_VALIDATION": "Gate validation failed",
    "SIGNAL_VALIDATION": "Signal validation failed",
    "CONFIDENCE_THRESHOLD": "Confidence below threshold",
    
    # Lifecycle errors
    "INVALID_STATE_TRANSITION": "Invalid state transition",
    "SYMBOL_NOT_READY": "Symbol not in ready state",
    "SYMBOL_LOCK": "Could not acquire symbol lock",
    
    # Execution errors
    "ORDER_PLACEMENT": "Order placement failed",
    "INSUFFICIENT_BALANCE": "Insufficient balance",
    "MIN_NOTIONAL_VIOLATION": "Minimum notional violated",
    "ORDER_VALIDATION": "Order validation failed",
    "DUPLICATE_ORDER": "Duplicate order detected",
    
    # Exchange errors
    "EXCHANGE_API": "Exchange API error",
    "EXCHANGE_RATE_LIMIT": "Exchange rate limited",
    "EXCHANGE_AUTH": "Exchange authentication failed",
    "EXCHANGE_INVALID_PAIR": "Invalid trading pair",
    "EXCHANGE_LIQUIDITY": "Insufficient liquidity",
    
    # State errors
    "STATE_SYNC": "State sync failed",
    "STATE_LOCK": "State lock failed",
    "STATE_CORRUPTION": "State corruption detected",
    "STATE_CONSISTENCY": "State consistency failed",
    
    # Network errors
    "NETWORK_TIMEOUT": "Network timeout",
    "CONNECTION_REFUSED": "Connection refused",
    "CONNECTION_RESET": "Connection reset",
    "DNS_RESOLUTION": "DNS resolution failed",
    
    # Validation errors
    "INVALID_PARAMETER": "Invalid parameter",
    "MISSING_PARAMETER": "Missing parameter",
    "TYPE_MISMATCH": "Type mismatch",
    "RANGE_ERROR": "Value out of range",
    
    # Configuration errors
    "INVALID_CONFIG": "Invalid configuration",
    "MISSING_CONFIG": "Missing configuration",
    "CONFIG_VALIDATION": "Configuration validation failed",
    
    # Resource errors
    "INSUFFICIENT_MEMORY": "Insufficient memory",
    "RESOURCE_LIMIT": "Resource limit exceeded",
    "RESOURCE_UNAVAILABLE": "Resource unavailable",
}


def create_error_context(
    category: ErrorCategory,
    severity: ErrorSeverity,
    error_code: str,
    message: str,
    recovery_strategy: ErrorRecovery = ErrorRecovery.NONE,
    operation: Optional[str] = None,
    component: Optional[str] = None,
    symbol: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> ErrorContext:
    """
    Create an ErrorContext instance.
    
    Args:
        category: Error category
        severity: Error severity
        error_code: Unique error code
        message: Error message
        recovery_strategy: How to recover
        operation: What operation was being performed
        component: Which component produced the error
        symbol: Trading symbol (if applicable)
        metadata: Additional context
    
    Returns:
        Configured ErrorContext instance
    """
    return ErrorContext(
        category=category,
        severity=severity,
        recovery_strategy=recovery_strategy,
        error_code=error_code,
        message=message,
        timestamp=datetime.now(),
        operation=operation,
        component=component,
        symbol=symbol,
        metadata=metadata or {},
    )
