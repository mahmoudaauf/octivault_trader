"""
Sanity tests for Part B: strict cap counting.

Validates that when STRICT_CAP_COUNT_TRADABLE=1, the
MetaController._count_significant_positions() helper promotes tradable-dust
(value ≥ STRICT_CAP_MIN_VALUE_USDT) into the significant count. This is the
fix for the DOGE-bypass: 33 wallet positions individually below the $25
significant floor must collectively count toward the position cap.

These tests do not boot the full MetaController. They invoke the real method
on a minimally-stubbed instance, exercising only the lines we changed.
"""

from __future__ import annotations

import asyncio

from src.l8_lifecycle.meta_controller import MetaController


class _StubSharedState:
    """Just enough of SharedState to drive _count_significant_positions."""

    def __init__(self, positions: dict, classification: dict) -> None:
        self._positions = positions
        self._classification = classification

    async def classify_positions_by_size(self):
        return dict(self._classification)

    def get_positions_snapshot(self, **_kw):
        return dict(self._positions)


def _make_meta_with(positions: dict, classification: dict) -> MetaController:
    """Build a bare MetaController without invoking __init__."""
    mc = MetaController.__new__(MetaController)
    mc.shared_state = _StubSharedState(positions, classification)
    # _count_significant_positions only uses self.logger and self.shared_state
    import logging

    mc.logger = logging.getLogger("test")
    return mc


def _pos(qty: float, price: float) -> dict:
    return {
        "quantity": qty,
        "mark_price": price,
        "value_usdt": qty * price,
    }


def test_strict_cap_disabled_dust_does_not_count(monkeypatch):
    """Without STRICT_CAP_COUNT_TRADABLE set, dust stays dust."""
    monkeypatch.delenv("STRICT_CAP_COUNT_TRADABLE", raising=False)

    positions = {
        "AAA/USDT": _pos(1, 100),  # significant
        "BBB/USDT": _pos(1, 10),  # tradable-dust ($10)
        "CCC/USDT": _pos(1, 8),  # tradable-dust ($8)
    }
    classification = {
        "significant": ["AAA/USDT"],
        "dust": ["BBB/USDT", "CCC/USDT"],
        "permanent_dust": [],
    }
    mc = _make_meta_with(positions, classification)

    total, sig, dust = asyncio.run(mc._count_significant_positions())

    assert total == 3
    assert sig == 1  # ← only the truly significant one
    assert dust == 2


def test_strict_cap_enabled_promotes_tradable_dust(monkeypatch):
    """With STRICT_CAP_COUNT_TRADABLE=1, dust ≥ $5 is promoted to sig."""
    monkeypatch.setenv("STRICT_CAP_COUNT_TRADABLE", "1")
    monkeypatch.setenv("STRICT_CAP_MIN_VALUE_USDT", "5")

    positions = {
        "AAA/USDT": _pos(1, 100),  # significant — already in sig
        "BBB/USDT": _pos(1, 10),  # dust $10 — promoted
        "CCC/USDT": _pos(1, 8),  # dust $8  — promoted
        "DDD/USDT": _pos(1, 3),  # dust $3  — NOT promoted
    }
    classification = {
        "significant": ["AAA/USDT"],
        "dust": ["BBB/USDT", "CCC/USDT", "DDD/USDT"],
        "permanent_dust": [],
    }
    mc = _make_meta_with(positions, classification)

    total, sig, dust = asyncio.run(mc._count_significant_positions())

    assert total == 4
    assert sig == 3  # AAA + BBB + CCC promoted
    assert dust == 1  # only DDD remains dust


def test_strict_cap_doge_scenario(monkeypatch):
    """The actual DOGE-bypass scenario: 33 wallet positions just below floor.

    Without strict cap → sig=0, max_pos=2 not blocked.
    With strict cap    → sig=33, far above max_pos, BUY blocked.
    """
    monkeypatch.setenv("STRICT_CAP_COUNT_TRADABLE", "1")
    monkeypatch.setenv("STRICT_CAP_MIN_VALUE_USDT", "5")

    # 33 positions, each $7 (above $5 floor, below $25 significant floor)
    positions = {f"S{i}/USDT": _pos(1.0, 7.0) for i in range(33)}
    dust_list = list(positions.keys())
    classification = {
        "significant": [],
        "dust": dust_list,
        "permanent_dust": [],
    }
    mc = _make_meta_with(positions, classification)

    total, sig, dust = asyncio.run(mc._count_significant_positions())

    assert total == 33
    assert sig == 33  # all promoted
    assert dust == 0
    # In _is_portfolio_full(total=33, sig=33, dust=0, max_pos=2),
    # CAPACITY_COUNT_SIGNIFICANT_ONLY mode → sig (33) >= max_pos (2) → FULL ✅


def test_strict_cap_min_value_floor_excludes_micro_dust(monkeypatch):
    """Positions below STRICT_CAP_MIN_VALUE_USDT are NOT promoted."""
    monkeypatch.setenv("STRICT_CAP_COUNT_TRADABLE", "1")
    monkeypatch.setenv("STRICT_CAP_MIN_VALUE_USDT", "10")  # higher floor

    positions = {
        "OK/USDT": _pos(1, 15),  # ≥ $10 → promoted
        "MICRO/USDT": _pos(1, 5),  # <  $10 → stays dust
    }
    classification = {
        "significant": [],
        "dust": ["OK/USDT", "MICRO/USDT"],
        "permanent_dust": [],
    }
    mc = _make_meta_with(positions, classification)

    total, sig, dust = asyncio.run(mc._count_significant_positions())

    assert total == 2
    assert sig == 1
    assert dust == 1
