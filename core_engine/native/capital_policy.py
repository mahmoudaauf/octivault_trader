"""
Native capital policy helpers.

Shared reserve / spendable-capital logic lives here so the decision engine,
allocator, and portfolio facades all reason about the same executable cash.
"""

from __future__ import annotations

import time
from typing import Any


def compute_spendable_quote(
    free_quote: float,
    *,
    reserve_ratio: float = 0.10,
    min_reserve: float = 0.0,
    reserved_quote: float = 0.0,
) -> float:
    """Return executable quote after reserve policy.

    Small-account policy intentionally relaxes reserves to avoid capital
    starvation while keeping a minimal buffer for fees and drift.
    """
    available = max(0.0, float(free_quote or 0.0))
    reserve = max(available * max(0.0, float(reserve_ratio)), max(0.0, float(min_reserve)))

    if 2.0 < available < 50.0:
        if available < 25.0:
            reserve = 1.0
        else:
            reserve = max(available * 0.05, 0.50)

    return max(0.0, available - max(0.0, float(reserved_quote or 0.0)) - reserve)


def prune_reservations(
    reservations: list[dict[str, Any]] | None,
    *,
    now_ts: float | None = None,
    max_age_sec: float = 90.0,
    default_ttl_sec: float = 30.0,
) -> tuple[list[dict[str, Any]], float]:
    """
    Return (valid_reservations, freed_quote) after pruning stale entries.
    """
    now = float(now_ts or time.time())
    kept: list[dict[str, Any]] = []
    freed = 0.0
    for reservation in reservations or []:
        amount = float(reservation.get("amount", 0.0) or 0.0)
        expires_at = float(reservation.get("expires_at", 0.0) or 0.0)
        created_at = float(reservation.get("created_at", 0.0) or 0.0)

        if expires_at <= 0:
            freed += amount
            continue
        if expires_at <= now:
            freed += amount
            continue
        if created_at <= 0:
            created_at = expires_at - float(default_ttl_sec or 30.0)
        if (now - created_at) > max_age_sec:
            freed += amount
            continue
        kept.append(dict(reservation))
    return kept, freed
