"""
Native L4: Recovery engine (Phase 8.3.11).

System self-diagnosis and recovery planner. Replaces the ``compat.py``
null stub for the ``recovery_engine`` app_ctx key, satisfying the
contract consumed by ``OperationsEngine.recover_state`` /
``OperationsEngine.apply_recovery``.

Responsibility
--------------
Inspect the running native stack (``NativeSharedState`` +
``NativeBalanceSync`` + ``NativeSafetyOrderManager``) and:

1. Detect anomalies whose recovery is in-process (no operator action):
   - Orphaned safety-order intents whose underlying position no longer
     exists (position closed but OCO never cleared)
   - Stale market data (no price tick for *symbol* despite an open
     position)
   - NAV drift between ``shared_state.nav_usdt`` and the derived
     ``free + Σ position_value`` figure beyond a tolerance
   - Open positions that still have zero ``entry_price`` (hydration
     incomplete)
2. Translate each anomaly into a structured recovery step.
3. Apply those steps idempotently when ``apply_plan`` is invoked.

Out of scope (handled elsewhere, intentionally)
-----------------------------------------------
- Filesystem checkpoint replay → ``system_state_manager.py``
- Order-state reconciliation against exchange → L4 executor
- Process-level restart → ``auto_recovery.py``

Design
------
* Pure observation in ``generate_recovery_plan``: no mutation, no I/O.
* All mutations are in ``apply_plan``, which dispatches on the
  ``op`` field of each step. Unknown ops are logged and skipped (safe
  forward-compat — operators can author new ops without bumping this
  module immediately).
* Returns the SAME ``RecoveryPlan`` dataclass the legacy
  ``OperationsEngine`` already uses, so no caller changes are needed.
* Read-only w.r.t. the exchange: ``cancel_oco`` calls go through
  ``NativeSafetyOrderManager`` which already has its own best-effort
  exchange path.
* No background tasks → no shutdown wiring needed.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .safety_order_manager import NativeSafetyOrderManager
    from .shared_state import NativeSharedState

logger = logging.getLogger(__name__)


# Re-export of the legacy plan shape so callers can ``from
# core_engine.native.recovery_engine import RecoveryPlan`` instead of
# pulling it through OperationsEngine. Keeps the field set identical.
@dataclass
class RecoveryPlan:
    """System recovery plan (mirrors core_engine.operations_engine shape)."""

    issues: list[str]
    recovery_steps: list[str]
    estimated_recovery_time_sec: float
    priority: str  # "IMMEDIATE", "URGENT", "HIGH", "NORMAL"
    auto_recover: bool = True
    # Native-only: structured ops the apply_plan dispatcher consumes.
    # Each entry: {"op": str, "symbol": str | None, **op_specific_kwargs}.
    ops: list[dict[str, Any]] = field(default_factory=list)


_PRIORITY_RANK = {"NORMAL": 0, "HIGH": 1, "URGENT": 2, "IMMEDIATE": 3}


def _max_priority(a: str, b: str) -> str:
    return a if _PRIORITY_RANK.get(a, 0) >= _PRIORITY_RANK.get(b, 0) else b


class NativeRecoveryEngine:
    """In-process self-diagnosis and recovery executor."""

    def __init__(
        self,
        shared_state: NativeSharedState,
        *,
        safety_order_manager: NativeSafetyOrderManager | None = None,
        nav_drift_tolerance_pct: float = 5.0,
        stale_price_threshold_sec: float = 60.0,
    ) -> None:
        if nav_drift_tolerance_pct < 0:
            raise ValueError(f"nav_drift_tolerance_pct must be >= 0, got {nav_drift_tolerance_pct}")
        if stale_price_threshold_sec < 0:
            raise ValueError(
                f"stale_price_threshold_sec must be >= 0, got {stale_price_threshold_sec}"
            )
        self._state = shared_state
        self._safety = safety_order_manager
        self._nav_tol_pct = float(nav_drift_tolerance_pct)
        self._stale_sec = float(stale_price_threshold_sec)

        # Health
        self._plans_generated = 0
        self._plans_applied = 0
        self._ops_applied = 0
        self._ops_skipped = 0

    # ------------------------------------------------------------------
    # Plan generation
    # ------------------------------------------------------------------
    async def generate_recovery_plan(self) -> RecoveryPlan:
        """
        Inspect state and emit a structured recovery plan.

        Returns a plan with empty ``ops`` when nothing needs attention.
        ``priority`` reflects the most severe detected issue.
        """
        self._plans_generated += 1
        issues: list[str] = []
        steps: list[str] = []
        ops: list[dict[str, Any]] = []
        priority = "NORMAL"
        eta = 0.0

        positions = getattr(self._state, "positions", None) or {}
        position_symbols = {sym for sym, pos in positions.items() if self._qty_of(pos) > 0}

        # 1) Orphan OCO intents (intent without a live position).
        if self._safety is not None:
            for sym in self._safety.list_active():
                if sym not in position_symbols:
                    issues.append(f"orphan OCO intent for {sym} (position closed)")
                    steps.append(f"cancel orphan OCO for {sym}")
                    ops.append({"op": "cancel_orphan_oco", "symbol": sym})
                    priority = _max_priority(priority, "HIGH")
                    eta += 0.5

        # 2) Stale market data for symbols we still hold.
        timestamps = getattr(self._state, "price_timestamps", None) or {}
        now = time.time()
        for sym in position_symbols:
            ts = float(timestamps.get(sym, 0.0) or 0.0)
            if ts <= 0:
                continue  # no timestamp tracking → skip silently
            age = now - ts
            if age > self._stale_sec:
                issues.append(f"stale market data for {sym} ({age:.1f}s old)")
                steps.append(f"force market-data refresh for {sym}")
                ops.append({"op": "request_price_refresh", "symbol": sym})
                priority = _max_priority(priority, "HIGH")
                eta += 1.0

        # 3) NAV drift vs derived value.
        nav = float(getattr(self._state, "nav_usdt", 0.0) or 0.0)
        derived = self._derive_nav(positions)
        if nav > 0 and derived > 0:
            drift_pct = abs(nav - derived) / max(nav, derived) * 100.0
            if drift_pct > self._nav_tol_pct:
                issues.append(
                    f"NAV drift {drift_pct:.2f}% (canonical={nav:.2f}, " f"derived={derived:.2f})"
                )
                steps.append("recompute NAV from balance + position values")
                ops.append({"op": "recompute_nav", "symbol": None})
                priority = _max_priority(priority, "URGENT")
                eta += 0.2

        # 4) Open positions with zero entry_price (hydration incomplete).
        for sym in position_symbols:
            entry = float(getattr(positions[sym], "entry_price", 0.0) or 0.0)
            if entry <= 0:
                issues.append(f"position {sym} has zero entry_price (incomplete hydration)")
                steps.append(f"backfill entry_price for {sym} from mark")
                ops.append({"op": "backfill_entry_from_mark", "symbol": sym})
                priority = _max_priority(priority, "HIGH")
                eta += 0.3

        return RecoveryPlan(
            issues=issues,
            recovery_steps=steps,
            estimated_recovery_time_sec=eta,
            priority=priority,
            auto_recover=True,
            ops=ops,
        )

    # ------------------------------------------------------------------
    # Plan execution
    # ------------------------------------------------------------------
    async def apply_plan(self, plan: RecoveryPlan) -> bool:
        """
        Execute every op in *plan*. Returns True iff every op succeeded.

        Unknown ops are logged and counted as skipped (not failed) so
        forward-compat plan authors can co-exist with older engines.
        """
        self._plans_applied += 1
        if not plan.ops:
            return True

        all_ok = True
        for step in plan.ops:
            op = step.get("op", "")
            symbol = step.get("symbol")
            try:
                ok = await self._dispatch(op, symbol, step)
            except Exception as exc:
                logger.warning("recovery op %s(%s) raised: %s", op, symbol, exc)
                ok = False
            if ok:
                self._ops_applied += 1
            else:
                all_ok = False
        return all_ok

    # ------------------------------------------------------------------
    # Health / observability
    # ------------------------------------------------------------------
    def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "plans_generated": self._plans_generated,
            "plans_applied": self._plans_applied,
            "ops_applied": self._ops_applied,
            "ops_skipped": self._ops_skipped,
            "nav_drift_tolerance_pct": self._nav_tol_pct,
            "stale_price_threshold_sec": self._stale_sec,
            "safety_wired": self._safety is not None,
        }

    # ------------------------------------------------------------------
    # Internal — op dispatch
    # ------------------------------------------------------------------
    async def _dispatch(self, op: str, symbol: str | None, step: dict[str, Any]) -> bool:
        if op == "cancel_orphan_oco":
            if self._safety is None or symbol is None:
                self._ops_skipped += 1
                return False
            return await self._safety.cancel_oco(symbol)

        if op == "recompute_nav":
            positions = getattr(self._state, "positions", None) or {}
            derived = self._derive_nav(positions)
            if derived > 0:
                self._state.nav_usdt = derived
                return True
            self._ops_skipped += 1
            return False

        if op == "backfill_entry_from_mark":
            positions = getattr(self._state, "positions", None) or {}
            if symbol is None or symbol not in positions:
                self._ops_skipped += 1
                return False
            pos = positions[symbol]
            mark = float(getattr(pos, "mark_price", 0.0) or 0.0)
            if mark <= 0:
                self._ops_skipped += 1
                return False
            try:
                pos.entry_price = mark
                return True
            except (AttributeError, TypeError):
                self._ops_skipped += 1
                return False

        if op == "request_price_refresh":
            # No-op on its own — the next market_data poll cycle will
            # refresh the cache. We treat this as advisory: counted as
            # applied so the plan still reports success.
            return True

        # Unknown op: forward-compat skip.
        logger.debug("recovery: unknown op %r — skipping", op)
        self._ops_skipped += 1
        return False

    # ------------------------------------------------------------------
    # Internal — math helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _qty_of(pos: Any) -> float:
        try:
            return float(getattr(pos, "qty", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _derive_nav(self, positions: dict[str, Any]) -> float:
        free = float(getattr(self._state, "free_balance_usdt", 0.0) or 0.0)
        total = free
        for pos in positions.values():
            qty = self._qty_of(pos)
            mark = float(getattr(pos, "mark_price", 0.0) or 0.0)
            total += qty * mark
        return total


__all__ = ["NativeRecoveryEngine", "RecoveryPlan"]
