"""
Sanity tests for the InsufficientBalance circuit breaker (run-#5 fix).

Validates that:
  T1  After N consecutive InsufficientBalance rejections on the same (symbol,side),
      the breaker trips, applies a cooldown, and forces a wallet re-hydration.
  T2  is_insuff_bal_cooling() returns True during the cooldown window.
  T3  A non-insufficient outcome resets the streak before threshold.
  T4  Different (symbol, side) keys track independently.
"""

from __future__ import annotations

import asyncio
import time
import types

import pytest

from src.l4_execution.execution_manager import ExecutionManager


class _StubSharedState:
    def __init__(self) -> None:
        self.hydrate_calls = 0
        self.prune_calls = 0

    async def hydrate_positions_from_balances(self):
        self.hydrate_calls += 1

    async def prune_reservations(self):
        self.prune_calls += 1

    async def get_free_usdt(self):
        return 100.0


def _make_em(monkeypatch) -> ExecutionManager:
    """Build a bare ExecutionManager without invoking __init__."""
    em = ExecutionManager.__new__(ExecutionManager)
    em.shared_state = _StubSharedState()
    import logging
    em.logger = logging.getLogger("test_em")
    return em


def test_breaker_trips_after_threshold(monkeypatch):
    monkeypatch.setenv("INSUFF_BAL_BREAKER_THRESHOLD", "5")
    monkeypatch.setenv("INSUFF_BAL_BREAKER_COOLDOWN_S", "120")
    em = _make_em(monkeypatch)

    # 4 rejections — below threshold, no cooldown yet
    for _ in range(4):
        asyncio.run(em._on_order_failed("ETHUSDT", "SELL", "InsufficientBalance", quote=25.0))
    assert em.is_insuff_bal_cooling("ETHUSDT", "SELL") is False
    assert em.shared_state.hydrate_calls == 0

    # 5th rejection trips the breaker
    asyncio.run(em._on_order_failed("ETHUSDT", "SELL", "InsufficientBalance", quote=25.0))
    assert em.is_insuff_bal_cooling("ETHUSDT", "SELL") is True
    assert em.shared_state.hydrate_calls == 1


def test_cooldown_expires(monkeypatch):
    monkeypatch.setenv("INSUFF_BAL_BREAKER_THRESHOLD", "2")
    monkeypatch.setenv("INSUFF_BAL_BREAKER_COOLDOWN_S", "0.1")  # 100ms for test
    em = _make_em(monkeypatch)

    asyncio.run(em._on_order_failed("ETHUSDT", "SELL", "InsufficientBalance"))
    asyncio.run(em._on_order_failed("ETHUSDT", "SELL", "InsufficientBalance"))
    assert em.is_insuff_bal_cooling("ETHUSDT", "SELL") is True

    time.sleep(0.15)
    assert em.is_insuff_bal_cooling("ETHUSDT", "SELL") is False


def test_non_insufficient_resets_streak(monkeypatch):
    monkeypatch.setenv("INSUFF_BAL_BREAKER_THRESHOLD", "3")
    em = _make_em(monkeypatch)

    asyncio.run(em._on_order_failed("ETHUSDT", "SELL", "InsufficientBalance"))
    asyncio.run(em._on_order_failed("ETHUSDT", "SELL", "InsufficientBalance"))
    # A different reason resets the streak
    asyncio.run(em._on_order_failed("ETHUSDT", "SELL", "MinNotionalViolation"))
    asyncio.run(em._on_order_failed("ETHUSDT", "SELL", "InsufficientBalance"))
    asyncio.run(em._on_order_failed("ETHUSDT", "SELL", "InsufficientBalance"))

    # Streak is now 2, threshold is 3 → still NOT tripped
    assert em.is_insuff_bal_cooling("ETHUSDT", "SELL") is False
    assert em.shared_state.hydrate_calls == 0


def test_different_keys_track_independently(monkeypatch):
    monkeypatch.setenv("INSUFF_BAL_BREAKER_THRESHOLD", "2")
    em = _make_em(monkeypatch)

    asyncio.run(em._on_order_failed("ETHUSDT", "SELL", "InsufficientBalance"))
    asyncio.run(em._on_order_failed("BTCUSDT", "SELL", "InsufficientBalance"))
    assert em.is_insuff_bal_cooling("ETHUSDT", "SELL") is False
    assert em.is_insuff_bal_cooling("BTCUSDT", "SELL") is False

    asyncio.run(em._on_order_failed("ETHUSDT", "SELL", "InsufficientBalance"))
    # Only ETH tripped
    assert em.is_insuff_bal_cooling("ETHUSDT", "SELL") is True
    assert em.is_insuff_bal_cooling("BTCUSDT", "SELL") is False
    # And SELL ≠ BUY
    assert em.is_insuff_bal_cooling("ETHUSDT", "BUY") is False
