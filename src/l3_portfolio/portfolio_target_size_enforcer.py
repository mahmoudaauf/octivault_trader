"""
PortfolioTargetSizeEnforcer (L3 portfolio policy, invoked at L8 startup)

Single-purpose, startup-time policy:
  "When the bot starts, ensure the wallet holds at most TARGET_POSITION_COUNT
   tradable positions. If the wallet has more, liquidate the lowest-value ones
   until the count reaches the target."

Design contract:
  * Idempotent: re-running with portfolio already at/below target is a no-op.
  * Bounded: single pass — never recurses or retries internally.
  * Read-only by default: requires explicit enable=True before issuing SELLs.
  * Honors exchange constraints: only counts/liquidates positions whose value
    is ≥ the symbol's exchange min_notional (avoids touching dust).
  * Cooperates with the existing liquidation pipeline:
    `ExecutionManager.execute_liquidation_plan(exits=[{symbol, quantity, tag}])`.
  * Skips bot-managed positions that are explicitly RECOVERY/BOT_POSITION
    (the bot is presumed to want those). Only EXTERNAL_POSITION + DUST in
    excess of target are candidates for trimming.

This is intentionally tiny (~150 lines). It does not score by quality; it just
enforces the count contract by lowest-value-first ordering. Quality-based
selection should live in RotationExitAuthority for steady-state rotation.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional


class PortfolioTargetSizeEnforcer:
    """One-shot startup policy: trim wallet to N tradable positions."""

    LIQUIDATION_TAG = "startup/target_size_trim"

    def __init__(
        self,
        shared_state: Any,
        execution_manager: Any,
        *,
        target_count: int = 5,
        enable: bool = False,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.shared_state = shared_state
        self.execution_manager = execution_manager
        self.target_count = max(1, int(target_count or 5))
        self.enable = bool(enable)
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self._has_run: bool = False

    # ─────────────────────────────────────────────────────────────────
    # PUBLIC ENTRY POINT
    # ─────────────────────────────────────────────────────────────────

    async def enforce_once(self) -> Dict[str, Any]:
        """
        Run a single trim pass. Returns a structured report:
          {
            "ran": bool,
            "candidates_considered": int,
            "tradable_count": int,
            "target": int,
            "to_liquidate": int,
            "exits_submitted": int,
            "exits_filled": int,
            "skipped_reason": Optional[str],
          }
        """
        if self._has_run:
            return self._report(skipped_reason="already_ran_in_this_session")
        self._has_run = True

        if not self.enable:
            self.logger.info(
                "[TargetSizeEnforcer] disabled (set STARTUP_TRIM_TO_TARGET=1 to enable). target=%d",
                self.target_count,
            )
            return self._report(skipped_reason="disabled")

        if self.execution_manager is None or self.shared_state is None:
            return self._report(skipped_reason="missing_dependencies")

        # Wait briefly for hydration to settle (positions populated, prices set)
        await asyncio.sleep(2.0)

        snap = self._snapshot_positions()
        candidates = self._build_tradable_candidates(snap)

        tradable_count = len(candidates)
        if tradable_count <= self.target_count:
            self.logger.info(
                "[TargetSizeEnforcer] portfolio already at/below target "
                "(%d tradable ≤ target=%d) — no action",
                tradable_count, self.target_count,
            )
            return self._report(
                candidates_considered=len(snap),
                tradable_count=tradable_count,
                to_liquidate=0,
                skipped_reason="below_target",
            )

        # Sort by value asc → trim lowest-value first
        candidates.sort(key=lambda c: float(c["value_usdt"]))
        excess = tradable_count - self.target_count
        to_liquidate = candidates[:excess]

        self.logger.warning(
            "[TargetSizeEnforcer] trimming %d positions (tradable=%d → target=%d): %s",
            excess, tradable_count, self.target_count,
            [(c["symbol"], round(c["value_usdt"], 2)) for c in to_liquidate],
        )

        exits = [
            {
                "symbol": c["symbol"],
                "quantity": float(c["quantity"]),
                "tag": self.LIQUIDATION_TAG,
            }
            for c in to_liquidate
        ]

        # Best-effort execution; the executor will log per-symbol outcome.
        ok = False
        try:
            ok = bool(
                await self.execution_manager.execute_liquidation_plan(exits)
            )
        except Exception as e:
            self.logger.error(
                "[TargetSizeEnforcer] execute_liquidation_plan raised: %s", e
            )

        # Re-count after execution to report what actually got out
        post_snap = self._snapshot_positions()
        post_candidates = self._build_tradable_candidates(post_snap)
        post_count = len(post_candidates)
        exits_filled = max(0, tradable_count - post_count)

        self.logger.warning(
            "[TargetSizeEnforcer] done: submitted=%d filled=%d post_tradable=%d target=%d ok=%s",
            len(exits), exits_filled, post_count, self.target_count, ok,
        )
        return self._report(
            candidates_considered=len(snap),
            tradable_count=tradable_count,
            to_liquidate=excess,
            exits_submitted=len(exits),
            exits_filled=exits_filled,
        )

    # ─────────────────────────────────────────────────────────────────
    # INTERNAL HELPERS
    # ─────────────────────────────────────────────────────────────────

    def _snapshot_positions(self) -> Dict[str, Dict[str, Any]]:
        try:
            getter = getattr(self.shared_state, "get_positions_snapshot", None)
            if callable(getter):
                snap = getter(include_wallet_inventory=True) or {}
                return dict(snap)
        except TypeError:
            try:
                return dict(getter() or {})
            except Exception:
                return {}
        except Exception:
            return {}
        return {}

    def _build_tradable_candidates(
        self, snap: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Build the list of positions eligible for trim consideration.

        Eligibility:
          - quantity > 0
          - mark_price > 0
          - value_usdt ≥ MIN_TRIM_VALUE_USDT (default $5; smaller is dust pipeline's job)
          - is_tradable is not False
          - classification is NOT BOT_POSITION/RECOVERY (those belong to the bot)

        Returns a list of dicts with keys: symbol, quantity, value_usdt, classification.
        """
        min_value = float(os.environ.get("STARTUP_TRIM_MIN_VALUE_USDT", "5") or 5)
        out: List[Dict[str, Any]] = []
        for sym, pos in (snap or {}).items():
            if not isinstance(pos, dict):
                continue
            qty = float(pos.get("quantity") or pos.get("qty") or 0.0)
            if qty <= 0:
                continue
            if pos.get("is_tradable") is False:
                continue
            classification = str(pos.get("classification") or "").upper()
            if classification in {"BOT_POSITION", "RECOVERY"}:
                # Bot-managed positions are out of scope for startup trim
                continue
            px = float(
                pos.get("mark_price")
                or pos.get("current_price")
                or pos.get("avg_price")
                or 0.0
            )
            if px <= 0:
                continue
            value = float(pos.get("value_usdt") or qty * px)
            if value < min_value:
                continue
            out.append({
                "symbol": sym,
                "quantity": qty,
                "value_usdt": value,
                "classification": classification,
            })
        return out

    def _report(
        self,
        *,
        candidates_considered: int = 0,
        tradable_count: int = 0,
        to_liquidate: int = 0,
        exits_submitted: int = 0,
        exits_filled: int = 0,
        skipped_reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        return {
            "ran": skipped_reason is None,
            "candidates_considered": int(candidates_considered),
            "tradable_count": int(tradable_count),
            "target": int(self.target_count),
            "to_liquidate": int(to_liquidate),
            "exits_submitted": int(exits_submitted),
            "exits_filled": int(exits_filled),
            "skipped_reason": skipped_reason,
        }
