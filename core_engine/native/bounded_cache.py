from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any


class NativeBoundedCache:
    """Small async-compatible TTL cache for execution idempotency guards."""

    def __init__(self, max_entries: int = 10000) -> None:
        self._max_entries = max(1, int(max_entries or 10000))
        self._store: OrderedDict[str, tuple[float, Any]] = OrderedDict()

    def _prune(self) -> None:
        now = time.time()
        expired = [k for k, (exp, _) in self._store.items() if exp > 0 and exp <= now]
        for key in expired:
            self._store.pop(key, None)
        while len(self._store) > self._max_entries:
            self._store.popitem(last=False)

    async def get(self, key: str, default: Any = None) -> Any:
        self._prune()
        item = self._store.get(str(key))
        if item is None:
            return default
        exp, value = item
        if exp > 0 and exp <= time.time():
            self._store.pop(str(key), None)
            return default
        self._store.move_to_end(str(key))
        return value

    async def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        ttl_sec = max(0.0, float(ttl or 0.0))
        exp = time.time() + ttl_sec if ttl_sec > 0 else 0.0
        self._store[str(key)] = (exp, value)
        self._store.move_to_end(str(key))
        self._prune()
