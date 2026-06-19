"""Bounded cache with TTL — pure utility, no external deps.

Lives at L0 so any layer can use it without violating the dependency
graph. Originally extracted from src/l8_lifecycle/meta_controller.py
to fix an L5→L8 leak in src/l5_strategy/signal_manager.py.
"""

from __future__ import annotations

import time
from typing import Any, Optional


class BoundedCache:
    """Thread-safe bounded cache with TTL support.

    Stores values up to ``max_size``. When at capacity, evicts the
    soonest-expiring entry. Expired entries are evicted lazily on
    access and proactively via :meth:`cleanup_expired`.
    """

    def __init__(self, max_size: int = 1000, default_ttl: float = 300.0):
        self._cache: dict[str, tuple[Any, float]] = {}
        self._max_size = max_size
        self._default_ttl = default_ttl

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache if not expired."""
        if key not in self._cache:
            return default
        value, expires_at = self._cache[key]
        if time.time() > expires_at:
            del self._cache[key]
            return default
        return value

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set value with TTL. Evicts soonest-expiring entry when at capacity."""
        expires_at = time.time() + (ttl if ttl is not None else self._default_ttl)
        if key not in self._cache and len(self._cache) >= self._max_size:
            try:
                oldest_key = min(self._cache, key=lambda k: self._cache[k][1])
                del self._cache[oldest_key]
            except (ValueError, KeyError):
                pass
        self._cache[key] = (value, expires_at)

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Alias for :meth:`set` for compatibility."""
        self.set(key, value, ttl)

    def list_all(self) -> list[Any]:
        """Return all non-expired values."""
        now = time.time()
        results: list[Any] = []
        expired_keys: list[str] = []
        for k, (val, exp) in list(self._cache.items()):
            if now <= exp:
                results.append(val)
            else:
                expired_keys.append(k)
        for k in expired_keys:
            self._cache.pop(k, None)
        return results

    def cleanup_expired(self) -> int:
        """Remove expired entries. Returns count of removed items."""
        now = time.time()
        expired_keys = [k for k, (_, exp) in self._cache.items() if now > exp]
        for k in expired_keys:
            del self._cache[k]
        return len(expired_keys)

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, key: str) -> bool:
        if key not in self._cache:
            return False
        if time.time() > self._cache[key][1]:
            del self._cache[key]
            return False
        return True


__all__ = ["BoundedCache"]
