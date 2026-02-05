"""
Cache utilities for MetaController.

This module provides bounded caches with TTL support and intent collection.
"""
from __future__ import annotations

import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from core.stubs import TradeIntent


class BoundedCache:
    """
    Thread-safe bounded cache with TTL support.

    Note: While called "thread-safe", this relies on asyncio's single-threaded
    execution model. Methods are atomic only if they don't contain await points.
    For true concurrent access, consider adding asyncio.Lock.
    """

    def __init__(self, max_size: int = 1000, default_ttl: float = 300.0):
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._access_order: deque = deque(maxlen=max_size)
        self._max_size = max_size
        self._default_ttl = default_ttl

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get value from cache if not expired.

        Args:
            key: Cache key
            default: Default value if key not found or expired

        Returns:
            Cached value or default
        """
        if key not in self._cache:
            return default
        value, expires_at = self._cache[key]
        now = time.time()
        if now > expires_at:
            # expire and remove from access order
            del self._cache[key]
            if key in self._access_order:
                self._access_order.remove(key)
            return default
        # mark as recently used
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
        return value

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Set value in cache with TTL.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (uses default_ttl if None)
        """
        now = time.time()
        expires_at = now + (ttl or self._default_ttl)

        # Evict if at capacity
        while len(self._cache) >= self._max_size and self._access_order:
            oldest_key = self._access_order.popleft()
            self._cache.pop(oldest_key, None)

        self._cache[key] = (value, expires_at)
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

    def list_all(self) -> List[Any]:
        """
        Return all non-expired values in the cache.

        Returns:
            List of all non-expired cached values
        """
        now = time.time()
        results = []
        expired_keys = []
        # Iterate once, collect both results and expired keys
        for k, (val, exp) in list(self._cache.items()):
            if now <= exp:
                results.append(val)
            else:
                expired_keys.append(k)

        # Cleanup expired entries
        for k in expired_keys:
            self._cache.pop(k, None)
            if k in self._access_order:
                self._access_order.remove(k)

        return results

    def cleanup_expired(self) -> int:
        """
        Remove expired entries and return count cleaned.

        Returns:
            Number of expired entries removed
        """
        now = time.time()
        expired_keys = [k for k, (_, exp) in self._cache.items() if now > exp]
        for key in expired_keys:
            self._cache.pop(key, None)
            # Use try/except to avoid expensive list removal failures
            try:
                self._access_order.remove(key)
            except ValueError:
                pass
        return len(expired_keys)


class ThreadSafeIntentSink:
    """
    Thread-safe intent collection with bounded storage.

    Note: Safe for asyncio single-threaded concurrency model.
    """

    def __init__(self, max_size: int = 500):
        self._intents: deque = deque(maxlen=max_size)

    def append(self, intent: "TradeIntent") -> None:
        """Append a single intent."""
        self._intents.append(intent)

    def extend(self, intents: List["TradeIntent"]) -> None:
        """Append multiple intents."""
        self._intents.extend(intents)

    def drain(self) -> List["TradeIntent"]:
        """
        Extract all intents and clear the sink.

        Returns:
            List of all collected intents
        """
        intents = list(self._intents)
        self._intents.clear()
        return intents
