"""
Native L0: Error Types and Exception Hierarchy

Defines all custom exception types with:
- Severity levels (DEBUG → CRITICAL)
- Categories (BOOTSTRAP, EXCHANGE, EXECUTION, etc.)
- Recovery strategies (NONE, RETRY, FALLBACK, ESCALATE)
- Contextual information (what went wrong, suggestions)

Design: Type-safe, predictable error handling enables automatic recovery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class ErrorSeverity(Enum):
    """Severity levels for all errors."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Categories of errors by origin."""

    BOOTSTRAP = "bootstrap"
    EXCHANGE = "exchange"
    EXECUTION = "execution"
    SIGNAL = "signal"
    PORTFOLIO = "portfolio"
    VALIDATION = "validation"
    NETWORK = "network"
    STATE = "state"
    CONFIGURATION = "configuration"
    RESOURCE = "resource"


class ErrorRecovery(Enum):
    """Recovery strategies for different error types."""

    NONE = "none"  # Error must be escalated
    RETRY = "retry"  # Retry the operation
    FALLBACK = "fallback"  # Use fallback logic
    SKIP = "skip"  # Skip this operation, continue
    ESCALATE = "escalate"  # Escalate to higher level


@dataclass
class OctiError(Exception):
    """Base exception class with context and recovery hints."""

    message: str
    category: ErrorCategory
    severity: ErrorSeverity = ErrorSeverity.ERROR
    recovery: ErrorRecovery = ErrorRecovery.ESCALATE
    context: dict[str, Any] = field(default_factory=dict)
    suggestion: Optional[str] = None

    def __str__(self) -> str:
        """Human-readable error message."""
        parts = [
            f"[{self.category.value.upper()}] {self.message}",
            f"Severity: {self.severity.value}",
            f"Recovery: {self.recovery.value}",
        ]
        if self.context:
            parts.append(f"Context: {self.context}")
        if self.suggestion:
            parts.append(f"Suggestion: {self.suggestion}")
        return " | ".join(parts)

    def __repr__(self) -> str:
        return f"OctiError({self.message!r})"


# ─────────────────────────────────────────────────────────────────────
# Specific Exception Types
# ─────────────────────────────────────────────────────────────────────


class BootstrapError(OctiError):
    """Initialization/bootstrap failures."""

    def __init__(
        self,
        message: str,
        context: dict[str, Any] | None = None,
        suggestion: str | None = None,
    ):
        super().__init__(
            message=message,
            category=ErrorCategory.BOOTSTRAP,
            severity=ErrorSeverity.CRITICAL,
            recovery=ErrorRecovery.NONE,
            context=context or {},
            suggestion=suggestion,
        )


class ExchangeError(OctiError):
    """Exchange API failures."""

    def __init__(
        self,
        message: str,
        context: dict[str, Any] | None = None,
        recovery: ErrorRecovery = ErrorRecovery.RETRY,
        suggestion: str | None = None,
    ):
        super().__init__(
            message=message,
            category=ErrorCategory.EXCHANGE,
            severity=ErrorSeverity.ERROR,
            recovery=recovery,
            context=context or {},
            suggestion=suggestion,
        )


class ExecutionError(OctiError):
    """Order execution failures."""

    def __init__(
        self,
        message: str,
        context: dict[str, Any] | None = None,
        recovery: ErrorRecovery = ErrorRecovery.ESCALATE,
        suggestion: str | None = None,
    ):
        super().__init__(
            message=message,
            category=ErrorCategory.EXECUTION,
            severity=ErrorSeverity.ERROR,
            recovery=recovery,
            context=context or {},
            suggestion=suggestion,
        )


class SignalError(OctiError):
    """Signal generation/validation failures."""

    def __init__(
        self,
        message: str,
        context: dict[str, Any] | None = None,
        recovery: ErrorRecovery = ErrorRecovery.SKIP,
        suggestion: str | None = None,
    ):
        super().__init__(
            message=message,
            category=ErrorCategory.SIGNAL,
            severity=ErrorSeverity.WARNING,
            recovery=recovery,
            context=context or {},
            suggestion=suggestion,
        )


class PortfolioError(OctiError):
    """Portfolio management failures."""

    def __init__(
        self,
        message: str,
        context: dict[str, Any] | None = None,
        recovery: ErrorRecovery = ErrorRecovery.ESCALATE,
        suggestion: str | None = None,
    ):
        super().__init__(
            message=message,
            category=ErrorCategory.PORTFOLIO,
            severity=ErrorSeverity.ERROR,
            recovery=recovery,
            context=context or {},
            suggestion=suggestion,
        )


class ValidationError(OctiError):
    """Input validation failures."""

    def __init__(
        self,
        message: str,
        context: dict[str, Any] | None = None,
        suggestion: str | None = None,
    ):
        super().__init__(
            message=message,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.WARNING,
            recovery=ErrorRecovery.NONE,
            context=context or {},
            suggestion=suggestion,
        )


class NetworkError(OctiError):
    """Network/connectivity failures."""

    def __init__(
        self,
        message: str,
        context: dict[str, Any] | None = None,
        recovery: ErrorRecovery = ErrorRecovery.RETRY,
        suggestion: str | None = None,
    ):
        super().__init__(
            message=message,
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.WARNING,
            recovery=recovery,
            context=context or {},
            suggestion=suggestion,
        )


__all__ = [
    "OctiError",
    "ErrorSeverity",
    "ErrorCategory",
    "ErrorRecovery",
    "BootstrapError",
    "ExchangeError",
    "ExecutionError",
    "SignalError",
    "PortfolioError",
    "ValidationError",
    "NetworkError",
]
