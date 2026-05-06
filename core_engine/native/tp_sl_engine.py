"""
Native L4: Take-profit / stop-loss engine (Phase 8.3.9).

Per-symbol TP/SL target store with threshold-crossing detection.
Replaces the ``compat.py`` null stub for the ``tp_sl_engine`` app_ctx
key, satisfying the contract consumed by
``DecisionEngineImpl.evaluate_exit_signals`` and the legacy
``meta_controller`` / ``execution_manager`` callers.

API contract
------------
* ``calculate_tp_sl(symbol, current_price) -> (tp, sl)`` — pure math
* ``set_initial_tp_sl(symbol, entry_price, qty, *, tier=None) -> None``
  — store/refresh per-symbol exit targets
* ``async check_exit_levels(symbol) -> str | None`` — returns
  ``"TP_HIT"`` / ``"SL_HIT"`` / ``None``; reads mark from
  ``NativeSharedState.positions[symbol].mark_price``
* ``clear(symbol)``      — drop targets after position closed
* ``get_targets(symbol)``— inspect current targets (testing/observability)
* ``health() -> dict``    — basic liveness counters

Design choices
--------------
* TP/SL percentages come from ``BootstrapConfig`` (``tp_pct``, ``sl_pct``)
  so the same engine can be tuned per-deployment without code changes.
  Both are stored as positive fractions (0.03 = 3%); SL is subtracted,
  TP is added.
* Per-tier overrides live in ``tier_overrides: dict[str, tuple[float, float]]``
  optional constructor kwarg. ``tier`` is a free-form string passed by
  the caller (``conservative``, ``swing``, etc.).
* Thread-safety: the target store is a plain dict; all mutations happen
  on the asyncio event loop, no locks needed.
* The engine is **read-only** w.r.t. exchange — it never places orders.
  It reports threshold crossings; the caller (DecisionEngine) translates
  those into ``FORCE_EXIT`` decisions.
* No background tasks → no shutdown wiring needed.

Returned tuple ordering for ``calculate_tp_sl`` is ``(tp, sl)`` to match
the legacy ``meta_controller`` unpacking ``tp, sl = engine.calculate_tp_sl(...)``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .shared_state import NativeSharedState

logger = logging.getLogger(__name__)


@dataclass
class _Targets:
    """Per-symbol TP/SL state."""

    entry_price: float
    qty: float
    tp_price: float
    sl_price: float
    tier: str | None = None


class NativeTPSLEngine:
    """Per-symbol TP/SL target store with threshold-crossing detection."""

    def __init__(
        self,
        shared_state: NativeSharedState,
        *,
        tp_pct: float = 0.03,
        sl_pct: float = 0.02,
        tier_overrides: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        if tp_pct <= 0:
            raise ValueError(f"tp_pct must be > 0, got {tp_pct}")
        if sl_pct <= 0:
            raise ValueError(f"sl_pct must be > 0, got {sl_pct}")
        self._state = shared_state
        self._tp_pct = float(tp_pct)
        self._sl_pct = float(sl_pct)
        self._tier_overrides: dict[str, tuple[float, float]] = dict(tier_overrides or {})
        self._targets: dict[str, _Targets] = {}

        # Health counters
        self._tp_hits = 0
        self._sl_hits = 0
        self._checks = 0

    # ------------------------------------------------------------------
    # Pure math
    # ------------------------------------------------------------------
    def calculate_tp_sl(
        self, symbol: str, current_price: float, *, tier: str | None = None
    ) -> tuple[float, float]:
        """
        Compute (tp, sl) absolute prices from a reference *current_price*.

        Pure function — does not mutate stored targets. Use
        ``set_initial_tp_sl`` to persist.
        """
        if current_price <= 0:
            return (0.0, 0.0)
        tp_pct, sl_pct = self._pcts_for(tier)
        return (current_price * (1.0 + tp_pct), current_price * (1.0 - sl_pct))

    # ------------------------------------------------------------------
    # Target store
    # ------------------------------------------------------------------
    def set_initial_tp_sl(
        self,
        symbol: str,
        entry_price: float,
        qty: float,
        *,
        tier: str | None = None,
    ) -> None:
        """
        Persist TP/SL targets for *symbol*. Overwrites any prior entry.

        Silently no-ops on non-positive ``entry_price`` or ``qty`` so that
        spurious fills/zero quantities don't poison the store.
        """
        if entry_price <= 0 or qty <= 0:
            return
        tp, sl = self.calculate_tp_sl(symbol, entry_price, tier=tier)
        self._targets[symbol] = _Targets(
            entry_price=float(entry_price),
            qty=float(qty),
            tp_price=tp,
            sl_price=sl,
            tier=tier,
        )

    def clear(self, symbol: str) -> None:
        """Drop targets for *symbol* (call after the position closes)."""
        self._targets.pop(symbol, None)

    def get_targets(self, symbol: str) -> dict[str, Any] | None:
        """
        Return a serialisable snapshot of stored targets, or ``None``
        if the symbol has no targets. Useful for tests and telemetry.
        """
        t = self._targets.get(symbol)
        if t is None:
            return None
        return {
            "entry_price": t.entry_price,
            "qty": t.qty,
            "tp_price": t.tp_price,
            "sl_price": t.sl_price,
            "tier": t.tier,
        }

    # ------------------------------------------------------------------
    # Crossing detection
    # ------------------------------------------------------------------
    async def check_exit_levels(self, symbol: str) -> str | None:
        """
        Inspect current mark price for *symbol* and report a crossing.

        Returns ``"TP_HIT"`` if mark >= tp_price, ``"SL_HIT"`` if
        mark <= sl_price, else ``None``. Returns ``None`` for symbols
        with no targets registered, an unknown position, or a
        non-positive mark price.

        TP is checked **before** SL in the rare case both have crossed
        in the same tick (e.g., gap-up then gap-down between polls);
        this preserves the legacy ordering used by ``meta_controller``.
        """
        self._checks += 1
        targets = self._targets.get(symbol)
        if targets is None:
            return None

        mark = self._mark_for(symbol)
        if mark <= 0:
            return None

        if mark >= targets.tp_price:
            self._tp_hits += 1
            return "TP_HIT"
        if mark <= targets.sl_price:
            self._sl_hits += 1
            return "SL_HIT"
        return None

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------
    def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "tracked_symbols": len(self._targets),
            "tp_hits": self._tp_hits,
            "sl_hits": self._sl_hits,
            "checks": self._checks,
            "tp_pct": self._tp_pct,
            "sl_pct": self._sl_pct,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _pcts_for(self, tier: str | None) -> tuple[float, float]:
        if tier and tier in self._tier_overrides:
            return self._tier_overrides[tier]
        return (self._tp_pct, self._sl_pct)

    def _mark_for(self, symbol: str) -> float:
        positions = getattr(self._state, "positions", None) or {}
        pos = positions.get(symbol)
        if pos is not None:
            mark = float(getattr(pos, "mark_price", 0.0) or 0.0)
            if mark > 0:
                return mark
        # Fallback: shared price cache (kept fresh by NativeMarketData)
        cache = getattr(self._state, "price_cache", None) or {}
        return float(cache.get(symbol, 0.0) or 0.0)


__all__ = ["NativeTPSLEngine"]
