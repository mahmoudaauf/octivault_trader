"""
Native L3: Position manager (Phase 8.3.8)

Read-only per-symbol position accessor over ``NativeSharedState``.
Replaces the ``compat.py`` null stub for the ``position_manager``
app_ctx key.

API contract (consumed by core_engine.situation_engine.analyze_position
and core_engine.decision_engine.make_sell_decision):

* ``get_position(symbol) -> Position | None``   — live position or None
* ``analyze_position(symbol) -> dict``          — risk-tagged snapshot

Schema returned by ``analyze_position`` (from situation_engine docstring):
    {
        "quantity": float,
        "entry_price": float,
        "current_price": float,
        "p_and_l": float,
        "p_and_l_pct": float,
        "status": "ACTIVE" | "DUST_LOCKED",
        "risk_level": "LOW" | "MEDIUM" | "HIGH",
    }

Design choices
--------------
* Pure read-side. No mutation of shared state, no exchange I/O,
  no background tasks → no shutdown wiring needed.
* Risk classification is derived from absolute unrealized P&L percent:
  HIGH ≥ ``risk_high_pct`` (default 10%), MEDIUM ≥ ``risk_med_pct``
  (default 3%), else LOW. Both thresholds are constructor-tunable.
* DUST_LOCKED status is detected when qty > 0 but position value
  is below ``min_order_usdt``. ACTIVE otherwise.
* The legacy "LIQUIDATING" status is *not* emitted because L3 has no
  source of truth for in-flight liquidation orders; that signal lives
  on L4. Callers should treat its absence as "not liquidating".
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .shared_state import NativeSharedState, Position

logger = logging.getLogger(__name__)


class NativePositionManager:
    """Read-only per-symbol position accessor over native L0 state."""

    def __init__(
        self,
        shared_state: NativeSharedState,
        *,
        min_order_usdt: float = 10.0,
        risk_high_pct: float = 10.0,
        risk_med_pct: float = 3.0,
    ) -> None:
        if min_order_usdt <= 0:
            raise ValueError(f"min_order_usdt must be > 0, got {min_order_usdt}")
        if risk_high_pct <= risk_med_pct:
            raise ValueError(
                f"risk_high_pct ({risk_high_pct}) must be > risk_med_pct ({risk_med_pct})"
            )
        if risk_med_pct < 0:
            raise ValueError(f"risk_med_pct must be >= 0, got {risk_med_pct}")
        self._state = shared_state
        self._min_order_usdt = float(min_order_usdt)
        self._risk_high_pct = float(risk_high_pct)
        self._risk_med_pct = float(risk_med_pct)

    # ------------------------------------------------------------------
    # Single-position accessors
    # ------------------------------------------------------------------
    async def get_position(self, symbol: str) -> Position | None:
        """
        Return the live ``Position`` for *symbol* or ``None``.

        Returns ``None`` for unknown symbols and for tracked positions
        whose quantity is zero, so callers can use a single
        ``if position is None`` guard.
        """
        positions = getattr(self._state, "positions", None) or {}
        pos = positions.get(symbol)
        if pos is None:
            return None
        qty = float(getattr(pos, "qty", 0.0) or 0.0)
        if qty == 0.0:
            return None
        return pos

    async def analyze_position(self, symbol: str) -> dict[str, Any]:
        """
        Return a stable risk-tagged snapshot for *symbol*.

        For unknown / zero-qty positions the schema is still returned
        with zeroed numerics, ``status="ACTIVE"``, ``risk_level="LOW"``,
        so callers can avoid ``KeyError`` guards.
        """
        empty = self._empty_analysis()
        pos = await self.get_position(symbol)
        if pos is None:
            return empty

        qty = float(getattr(pos, "qty", 0.0) or 0.0)
        entry = float(getattr(pos, "entry_price", 0.0) or 0.0)
        mark = float(getattr(pos, "mark_price", 0.0) or 0.0)
        value = qty * mark
        pnl = (mark - entry) * qty if (entry > 0 and mark > 0) else 0.0
        pnl_pct = ((mark - entry) / entry * 100.0) if entry > 0 else 0.0

        return {
            "quantity": qty,
            "entry_price": entry,
            "current_price": mark,
            "p_and_l": pnl,
            "p_and_l_pct": pnl_pct,
            "status": self._classify_status(qty, value),
            "risk_level": self._classify_risk(pnl_pct),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _empty_analysis() -> dict[str, Any]:
        return {
            "quantity": 0.0,
            "entry_price": 0.0,
            "current_price": 0.0,
            "p_and_l": 0.0,
            "p_and_l_pct": 0.0,
            "status": "ACTIVE",
            "risk_level": "LOW",
        }

    def _classify_status(self, qty: float, value_usdt: float) -> str:
        if qty > 0 and 0.0 < value_usdt < self._min_order_usdt:
            return "DUST_LOCKED"
        return "ACTIVE"

    def _classify_risk(self, pnl_pct: float) -> str:
        mag = abs(pnl_pct)
        if mag >= self._risk_high_pct:
            return "HIGH"
        if mag >= self._risk_med_pct:
            return "MEDIUM"
        return "LOW"


__all__ = ["NativePositionManager"]
