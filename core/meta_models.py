# -*- coding: utf-8 -*-
"""
MetaController Data Models & Utilities

Extracted from meta_controller.py to reduce monolith size.
Contains shared data models, caches, and utility functions used by
MetaController and its dependent modules.
"""

from __future__ import annotations

import time
from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass


class ExecutionError(Exception):
    """Canonical P9 execution error with proper classification."""

    class Type:
        """Error type constants for backward compatibility (string-based comparison)."""
        EXTERNAL_API_ERROR = "EXTERNAL_API_ERROR"
        MIN_NOTIONAL_VIOLATION = "MIN_NOTIONAL_VIOLATION"
        FEE_SAFETY_VIOLATION = "FEE_SAFETY_VIOLATION"
        RISK_CAP_EXCEEDED = "RISK_CAP_EXCEEDED"
        INSUFFICIENT_BALANCE = "INSUFFICIENT_BALANCE"
        INTEGRITY_ERROR = "INTEGRITY_ERROR"

    def __init__(
        self,
        error_type: str,
        message: str,
        symbol: str = "",
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.error_type = error_type
        self.message = message
        self.symbol = symbol
        self.context = context or {}


class DustState(Enum):
    """Dust Accumulation State Machine."""
    EMPTY = "empty"
    DUST_ACCUMULATING = "dust_accumulating"
    DUST_MATURED = "dust_matured"
    TRADABLE = "tradable"


@dataclass
class LiquidityPlan:
    """Enhanced liquidation plan with P9 compliance."""
    status: str
    exits: List[Dict[str, Any]]
    freed_quote: float
    trace_id: str
    risk_assessment: Optional[Dict[str, Any]] = None


class BoundedCache:
    """Thread-safe bounded cache with TTL support."""

    def __init__(self, max_size: int = 1000, default_ttl: float = 300.0):
        self._cache: Dict[str, tuple] = {}
        self._max_size = max_size
        self._default_ttl = default_ttl

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache if not expired."""
        if key not in self._cache:
            return default
        value, expires_at = self._cache[key]
        now = time.time()
        if now > expires_at:
            del self._cache[key]
            return default
        return value

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set value in cache with TTL."""
        now = time.time()
        expires_at = now + (ttl or self._default_ttl)
        self._cache[key] = (value, expires_at)

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Alias for set() for compatibility."""
        self.set(key, value, ttl)

    def list_all(self) -> List[Any]:
        """Return all non-expired values."""
        now = time.time()
        results = []
        expired_keys = []
        for k, (val, exp) in list(self._cache.items()):
            if now <= exp:
                results.append(val)
            else:
                expired_keys.append(k)
        for k in expired_keys:
            self._cache.pop(k, None)
        return results

    def cleanup_expired(self) -> int:
        """Remove expired entries from cache. Returns count of removed items."""
        now = time.time()
        expired_keys = [k for k, (_, exp) in self._cache.items() if now > exp]
        for k in expired_keys:
            del self._cache[k]
        return len(expired_keys)


class ThreadSafeIntentSink:
    """Thread-safe collection for trade intents."""

    def __init__(self, max_size: int = 1000):
        self._intents = []
        self._max_size = max_size

    def add(self, intent: Any) -> None:
        """Add intent to sink."""
        self._intents.append(intent)
        if len(self._intents) > self._max_size:
            self._intents.pop(0)

    def get_all(self) -> List[Any]:
        """Get all intents and clear."""
        result = list(self._intents)
        self._intents.clear()
        return result


def parse_timestamp(val: Any, default_ts: float = 0.0) -> float:
    """Coerce timestamps into epoch seconds."""
    if val is None:
        return default_ts
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        try:
            if val.endswith('Z'):
                val = val[:-1] + '+00:00'
            from datetime import datetime
            dt = datetime.fromisoformat(val.replace('Z', '+00:00'))
            return dt.timestamp()
        except Exception:
            return default_ts
    return default_ts


def classify_execution_error(err: Any, symbol: str = "") -> ExecutionError:
    """Classify execution error type and return ExecutionError object."""
    if isinstance(err, ExecutionError):
        if symbol and not err.symbol:
            err.symbol = symbol
        return err
    if hasattr(err, 'error_type'):
        return ExecutionError(err.error_type, str(err), symbol)
    err_str = str(err).lower()
    if 'notional' in err_str:
        return ExecutionError(ExecutionError.Type.MIN_NOTIONAL_VIOLATION, str(err), symbol)
    if 'balance' in err_str or 'insufficient' in err_str:
        return ExecutionError(ExecutionError.Type.INSUFFICIENT_BALANCE, str(err), symbol)
    if 'fee' in err_str:
        return ExecutionError(ExecutionError.Type.FEE_SAFETY_VIOLATION, str(err), symbol)
    return ExecutionError("UNKNOWN", str(err), symbol)
