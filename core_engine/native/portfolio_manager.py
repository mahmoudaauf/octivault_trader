"""
Native L3: Portfolio manager (Phase 8.3.7)

Read-only portfolio aggregator over ``NativeSharedState`` and
``NativeBalanceSync``. Replaces the ``compat.py`` null stub for the
``portfolio_manager`` app_ctx key with a real implementation.

API contract (consumed by core_engine.implementations.SituationEngineImpl
and core_engine.situation_engine):

* ``get_nav() -> float``                 — total account value in USDT
* ``get_positions() -> dict[str,float]`` — symbol -> qty (open positions)
* ``get_pnl() -> float``                 — unrealized P&L in USDT
* ``get_capital_allocated() -> float``   — invested capital (locked)
* ``get_capital_available() -> float``   — free USDT balance
* ``positions`` (attr)                   — live dict[symbol, Position]
  exposed for ``meta_controller.py`` legacy consumers.
* ``get_dust_state(symbol) -> dict``     — minimal dust tracking
* ``get_dust_record(symbol) -> dict``    — alias for legacy callers

Design choices
--------------
* Pure read-side. No mutation of shared state, no exchange I/O.
* All accessors are async to match the legacy contract that the façade
  engines call with ``await``. They never block the event loop.
* NAV computation has the same fallback ladder as
  ``bootstrap._make_portfolio_accessor``: prefer ``shared_state.nav_usdt``
  → fallback to ``USDT free balance + Σ position_value``.
* P&L is the sum of ``Position.unrealized_pnl_pct`` weighted by
  position value (USDT-denominated unrealized gain/loss).
* Dust = position with ``position_value < min_order_usdt``. The legacy
  "state/record" distinction is preserved as separate accessors that
  return identical payloads.

This module is **not** wired into the orchestrator or executor — it
is a passive accessor over state already maintained by L0/L2.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .balance_sync import NativeBalanceSync
    from .shared_state import NativeSharedState

logger = logging.getLogger(__name__)


class NativePortfolioManager:
    """Read-only portfolio aggregator over native L0/L2 state."""

    def __init__(
        self,
        shared_state: NativeSharedState,
        balance_sync: NativeBalanceSync,
        *,
        min_order_usdt: float = 10.0,
    ) -> None:
        if min_order_usdt <= 0:
            raise ValueError(f"min_order_usdt must be > 0, got {min_order_usdt}")
        self._state = shared_state
        self._balance_sync = balance_sync
        self._min_order_usdt = float(min_order_usdt)

    # ------------------------------------------------------------------
    # NAV / capital
    # ------------------------------------------------------------------
    async def get_nav(self) -> float:
        """Total account value in USDT (canonical or derived)."""
        nav = float(getattr(self._state, "nav_usdt", 0.0) or 0.0)
        if nav > 0.0:
            return nav
        # Fallback: USDT free + sum of position values at mark price.
        return self._derived_nav()

    async def get_capital_available(self) -> float:
        """Free USDT balance (uninvested cash)."""
        # Prefer the canonical shared_state field if the writer has set
        # it; fall back to the L1 balance poller.
        free = float(getattr(self._state, "free_balance_usdt", 0.0) or 0.0)
        if free > 0.0:
            return free
        balances = self._balance_sync.get_balance()
        return float(balances.get("USDT", 0.0))

    async def get_capital_allocated(self) -> float:
        """Capital currently locked in open positions (USDT-denominated)."""
        invested = float(getattr(self._state, "invested_capital_usdt", 0.0) or 0.0)
        if invested > 0.0:
            return invested
        return self._sum_position_values()

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------
    async def get_positions(self) -> dict[str, float]:
        """Symbol → qty for every position with non-zero quantity."""
        out: dict[str, float] = {}
        for sym, pos in self._raw_positions().items():
            qty = self._extract_qty(pos)
            if qty != 0.0:
                out[sym] = qty
        return out

    @property
    def positions(self) -> dict[str, Any]:
        """
        Live view of ``shared_state.positions`` for legacy callers in
        ``src/l8_lifecycle/meta_controller.py`` that read
        ``portfolio_manager.positions`` directly. Returned dict is a
        snapshot copy so external mutation cannot corrupt L0 state.
        """
        return dict(self._raw_positions())

    # ------------------------------------------------------------------
    # P&L
    # ------------------------------------------------------------------
    async def get_pnl(self) -> float:
        """Unrealized P&L across open positions (USDT)."""
        pnl = 0.0
        for pos in self._raw_positions().values():
            qty = self._extract_qty(pos)
            entry = float(getattr(pos, "entry_price", 0.0) or 0.0)
            mark = float(getattr(pos, "mark_price", 0.0) or 0.0)
            if qty == 0.0 or entry <= 0.0 or mark <= 0.0:
                continue
            pnl += (mark - entry) * qty
        return pnl

    # ------------------------------------------------------------------
    # Dust (legacy contract)
    # ------------------------------------------------------------------
    async def get_dust_state(self, symbol: str) -> dict[str, Any]:
        """
        Dust = position whose USDT value is below the minimum order size.

        Returns a stable schema even when the symbol is unknown or
        the position is non-dust, so callers can branch on
        ``state["is_dust"]`` without ``KeyError`` guards.
        """
        positions = self._raw_positions()
        pos = positions.get(symbol)
        if pos is None:
            return {
                "symbol": symbol,
                "is_dust": False,
                "qty": 0.0,
                "value_usdt": 0.0,
                "threshold_usdt": self._min_order_usdt,
            }
        qty = self._extract_qty(pos)
        mark = float(getattr(pos, "mark_price", 0.0) or 0.0)
        value = qty * mark
        return {
            "symbol": symbol,
            "is_dust": 0.0 < value < self._min_order_usdt,
            "qty": qty,
            "value_usdt": value,
            "threshold_usdt": self._min_order_usdt,
        }

    async def get_dust_record(self, symbol: str) -> dict[str, Any]:
        """Legacy alias for ``get_dust_state``."""
        return await self.get_dust_state(symbol)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _raw_positions(self) -> dict[str, Any]:
        return getattr(self._state, "positions", {}) or {}

    @staticmethod
    def _extract_qty(pos: Any) -> float:
        """Pull qty from ``Position`` dataclass or scalar fallback."""
        qty = getattr(pos, "qty", None)
        if qty is None and isinstance(pos, (int, float)):
            return float(pos)
        try:
            return float(qty or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _sum_position_values(self) -> float:
        total = 0.0
        for pos in self._raw_positions().values():
            qty = self._extract_qty(pos)
            mark = float(getattr(pos, "mark_price", 0.0) or 0.0)
            total += qty * mark
        return total

    def _derived_nav(self) -> float:
        """USDT free balance + Σ position_value (mark-to-market)."""
        balances = self._balance_sync.get_balance()
        free = float(balances.get("USDT", 0.0))
        return free + self._sum_position_values()


__all__ = ["NativePortfolioManager"]
