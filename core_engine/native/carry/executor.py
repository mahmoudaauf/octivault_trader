"""
Native carry executor (Phase 3 of the funding-carry native-wiring plan).

Carries over carry_paper_trader.py's execute_legs() sequential-open,
no-atomic-rollback behavior as-is, rather than building true
compensating-transaction execution: NativeExecutor (executor.py) has zero
existing atomic/paired-execution infrastructure anywhere to build on
(confirmed via broad grep during the engineering-study research pass), and
building it from scratch (idempotency keys, retry-with-backoff specifically
on an unwind order, partial-fill reconciliation on the unwind) is a
meaningfully-sized new subsystem not justified before the strategy itself
has a live track record. This carries over an existing, already-accepted
risk (a possible naked leg if the second order fails after the first
succeeds) rather than introducing a new one -- the standalone script has
run this exact way since inception.

One new safety net beyond the standalone script: confirm the perp leg is
actually FILLED (checking the order response's own status field) before
firing the spot leg. carry_paper_trader.py's execute_legs() fires both
orders back-to-back with no check in between; this closes the "perp leg
silently rejected, spot leg fires anyway into a naked position" gap for the
common case (MARKET orders on Binance normally report fill status in the
synchronous create-order response).

On a leg mismatch (perp filled, spot leg raised/failed): log CRITICAL and
halt new carry opens via the shared kill-file mechanism (CarryGateEngine
already respects it) rather than attempting an automated unwind -- that's a
Phase 3.5 addition once there's a live track record, not part of this cut.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Optional

from .state import CarrySharedState, HedgePosition

logger = logging.getLogger(__name__)


@dataclass
class LegExecutionResult:
    success: bool
    reason: str = ""
    position: Optional[HedgePosition] = None
    # Populated only on a leg-mismatch failure (perp filled, spot did not) --
    # the caller / on-call runbook needs this to know a naked leg may exist.
    naked_leg: Optional[str] = None


class CarryLegExecutor:
    """Places the two legs of a delta-neutral carry hedge. Requires BOTH a
    futures client (the new NativeFuturesExchangeClient, Phase 1) and the
    EXISTING NativeExchangeClient (spot leg) -- reused as-is, not duplicated
    or modified, matching the plan's "parallel, additive" design."""

    def __init__(
        self,
        *,
        futures_client: Any,
        spot_client: Any,
        carry_state: CarrySharedState,
        leverage: int = 2,
        mismatch_kill_file: str = "logs/native_carry.stop",
    ) -> None:
        self._futures = futures_client
        self._spot = spot_client
        self._carry_state = carry_state
        self.leverage = int(leverage)
        self._mismatch_kill_file = mismatch_kill_file

    @staticmethod
    def _is_filled(order_response: dict[str, Any]) -> bool:
        return str(order_response.get("status", "")).upper() == "FILLED"

    def _raise_mismatch_alarm(self, symbol: str, *, leg: str, detail: str) -> None:
        logger.critical(
            "[carry] LEG MISMATCH on %s: %s leg failed after the other leg filled (%s). "
            "Halting new carry opens via kill file -- manual review required.",
            symbol, leg, detail,
        )
        try:
            os.makedirs(os.path.dirname(self._mismatch_kill_file) or ".", exist_ok=True)
            open(self._mismatch_kill_file, "w").close()
        except OSError:
            logger.exception("[carry] failed to write mismatch kill file %s", self._mismatch_kill_file)

    # ──────────────────────────────────────────────────────────────────
    # Open
    # ──────────────────────────────────────────────────────────────────
    async def open_hedge(self, symbol: str, funding_rate: float, notional_usd: float) -> LegExecutionResult:
        """funding_rate > 0 -> short perp + long spot (v1's only supported
        direction, matching CarryGateEngine's POSITIVE_ONLY restriction)."""
        perp_side = "SELL" if funding_rate > 0 else "BUY"
        spot_side = "BUY"  # v1: spot leg is always a long buy, matching carry_paper_trader.py
        # (a negative-funding "long_perp" hedge would need a spot-SHORT leg,
        # which requires spot-margin borrowing -- explicitly out of scope,
        # see HedgePosition's docstring and the plan's "not now" list.)

        try:
            mark_data = await self._futures.futures_mark_price(symbol)
            if isinstance(mark_data, list):
                mark_data = next((m for m in mark_data if m.get("symbol") == symbol), {})
            mark_price = float((mark_data or {}).get("markPrice", 0.0) or 0.0)
        except Exception as e:
            return LegExecutionResult(False, f"mark_price_fetch_failed: {str(e)[:100]}")

        if mark_price <= 0:
            return LegExecutionResult(False, "invalid_mark_price")

        qty = round(notional_usd / mark_price, 6)
        if qty <= 0:
            return LegExecutionResult(False, "computed_qty_non_positive")

        try:
            await self._futures.futures_change_leverage(symbol, self.leverage)
        except Exception as e:
            return LegExecutionResult(False, f"set_leverage_failed: {str(e)[:100]}")

        # Leg 1: perp.
        try:
            perp_order = await self._futures.futures_create_order(symbol, perp_side, qty)
        except Exception as e:
            return LegExecutionResult(False, f"perp_leg_failed: {str(e)[:100]}")

        if not self._is_filled(perp_order):
            # Confirmed NOT filled -- safe to stop here, no spot leg fired,
            # no naked position (this is the new safety net over the
            # standalone script's fire-both-unconditionally behavior).
            return LegExecutionResult(
                False, f"perp_leg_not_filled: status={perp_order.get('status')}"
            )

        # Leg 2: spot. Perp leg is confirmed filled -- if this fails now,
        # we have a naked perp position (the accepted, carried-over risk).
        try:
            spot_order = await self._spot.place_order(symbol, spot_side, qty)
        except Exception as e:
            self._raise_mismatch_alarm(symbol, leg="spot", detail=str(e)[:200])
            return LegExecutionResult(False, f"spot_leg_failed_after_perp_filled: {str(e)[:100]}", naked_leg="perp")

        if not self._is_filled(spot_order):
            self._raise_mismatch_alarm(
                symbol, leg="spot", detail=f"status={spot_order.get('status')}"
            )
            return LegExecutionResult(
                False, f"spot_leg_not_filled_after_perp_filled: status={spot_order.get('status')}",
                naked_leg="perp",
            )

        pos = self._carry_state.open_hedge(
            symbol,
            entry_funding=funding_rate,
            perp_qty=qty,
            spot_qty=qty,
            notional_usd=notional_usd,
        )
        return LegExecutionResult(True, "ok", position=pos)

    # ──────────────────────────────────────────────────────────────────
    # Close
    # ──────────────────────────────────────────────────────────────────
    async def close_hedge(self, symbol: str) -> LegExecutionResult:
        pos = self._carry_state.get_open_hedge(symbol)
        if pos is None:
            return LegExecutionResult(False, "not_open")

        perp_side = "BUY" if pos.direction == "short_perp" else "SELL"
        spot_side = "SELL"

        try:
            perp_order = await self._futures.futures_create_order(
                symbol, perp_side, pos.perp_qty, reduce_only=True
            )
        except Exception as e:
            return LegExecutionResult(False, f"perp_close_failed: {str(e)[:100]}")

        if not self._is_filled(perp_order):
            return LegExecutionResult(
                False, f"perp_close_not_filled: status={perp_order.get('status')}"
            )

        try:
            spot_order = await self._spot.place_order(symbol, spot_side, pos.spot_qty)
        except Exception as e:
            self._raise_mismatch_alarm(symbol, leg="spot_close", detail=str(e)[:200])
            return LegExecutionResult(False, f"spot_close_failed_after_perp_closed: {str(e)[:100]}", naked_leg="spot")

        if not self._is_filled(spot_order):
            self._raise_mismatch_alarm(
                symbol, leg="spot_close", detail=f"status={spot_order.get('status')}"
            )
            return LegExecutionResult(
                False, f"spot_close_not_filled: status={spot_order.get('status')}", naked_leg="spot",
            )

        closed = self._carry_state.close_hedge(symbol)
        return LegExecutionResult(True, "ok", position=closed)


__all__ = ["CarryLegExecutor", "LegExecutionResult"]
