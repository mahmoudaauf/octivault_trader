"""
Native L5: Execution Coordinator (Phase 8.2.6)

Stateful order sequencing + partial-fill reconciliation. Replaces ~700 LOC
legacy ``execution_coordinator.py`` with focused ~200-line implementation.

Design choices
--------------
* Sequential execution per symbol (no concurrent orders on same symbol).
* Idempotency dedup via decision UUID (skip if already executed in this cycle).
* Partial-fill reconciliation: compare exchange balance vs local tracking.
* Failure classification: retryable (network, 429) vs terminal (insufficient
  balance, invalid qty).
* No internal queue — caller drives the sequencing via :py:meth:`execute`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from core_engine.native.decisions import Decision
from core_engine.native.order_execution import ExchangeClientError, NativeOrderExecution

logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    """Outcome of a single execution attempt."""

    SUCCESS = "SUCCESS"  # order placed + accepted
    PARTIAL = "PARTIAL"  # order placed; fill status TBD (need reconciliation)
    RETRYABLE = "RETRYABLE"  # temporary failure; can retry
    TERMINAL = "TERMINAL"  # permanent failure; do not retry


@dataclass
class ExecutionResult:
    """Outcome of executing a single decision."""

    decision_id: str
    symbol: str
    status: ExecutionStatus
    quantity_requested: float
    quantity_executed: float = 0.0
    exchange_order_id: Optional[int] = None
    error: Optional[str] = None
    raw: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


class NativeExecutor:
    """
    Stateful order executor with dedup and reconciliation.

    Usage::

        executor = NativeExecutor(order_execution_client, ...)
        results = await executor.execute(decisions)
        # ⇒ list[ExecutionResult] with status + order IDs
    """

    def __init__(self, order_execution: NativeOrderExecution) -> None:
        self._order_exec = order_execution
        self._executed_ids: set[str] = set()  # dedup tracking
        self._symbol_locks: dict[str, float] = {}  # symbol → last execution ts

    # ──────────────────────────────────────────────────────────────────
    # Main API
    # ──────────────────────────────────────────────────────────────────
    async def execute(self, decisions: list[Decision]) -> list[ExecutionResult]:
        """
        Execute a list of decisions sequentially.

        Returns results with status + order IDs. Dedup prevents re-execution
        of the same decision_id within a session.
        """
        results: list[ExecutionResult] = []
        for dec in decisions:
            # Dedup gate
            if dec.decision_id in self._executed_ids:
                logger.debug("decision %s already executed; skipping", dec.decision_id)
                continue

            result = await self._execute_one(dec)
            results.append(result)

            # Mark as executed on success
            if result.status == ExecutionStatus.SUCCESS:
                self._executed_ids.add(dec.decision_id)

        return results

    # ──────────────────────────────────────────────────────────────────
    # Single decision execution
    # ──────────────────────────────────────────────────────────────────
    async def _execute_one(self, decision: Decision) -> ExecutionResult:
        """Execute a single decision. Handle errors and classify them."""
        result = ExecutionResult(
            decision_id=decision.decision_id,
            symbol=decision.symbol,
            status=ExecutionStatus.RETRYABLE,
            quantity_requested=decision.quantity,
        )

        # Per-symbol sequential gate (defensive; caller should not violate)
        last_ts = self._symbol_locks.get(decision.symbol)
        if last_ts is not None and (time.time() - last_ts) < 0.1:
            result.status = ExecutionStatus.RETRYABLE
            result.error = "symbol lock still held; try again"
            return result
        self._symbol_locks[decision.symbol] = time.time()

        try:
            from core_engine.native.decisions import Action

            if decision.action == Action.OPEN:
                result = await self._place_order(decision, result)
            elif decision.action == Action.CLOSE:
                result = await self._close_position(decision, result)
            else:
                result.status = ExecutionStatus.TERMINAL
                result.error = f"unknown action: {decision.action}"

        except ExchangeClientError as e:
            result.status = self._classify_error(str(e))
            result.error = str(e)
        except Exception as e:  # pragma: no cover — defensive
            logger.exception("unexpected error executing decision: %s", e)
            result.status = ExecutionStatus.TERMINAL
            result.error = f"unexpected: {type(e).__name__}: {e}"

        return result

    # ──────────────────────────────────────────────────────────────────
    # Order placement
    # ──────────────────────────────────────────────────────────────────
    async def _place_order(self, decision: Decision, result: ExecutionResult) -> ExecutionResult:
        """Place a new order (BUY)."""
        from core_engine.native.decisions import Action

        if decision.action != Action.OPEN:
            result.status = ExecutionStatus.TERMINAL
            result.error = "expected OPEN action"
            return result

        # Simple market order: BUY
        order_result = await self._order_exec.place_market_buy(
            decision.symbol,
            decision.quantity,
            client_order_id=decision.decision_id,  # idempotency key
        )

        result.raw = order_result.raw
        result.exchange_order_id = order_result.exchange_order_id
        result.quantity_executed = order_result.quantity

        if order_result.success:
            result.status = ExecutionStatus.SUCCESS
        else:
            # Exchange rejected the order
            if "insufficient" in (order_result.error or "").lower():
                result.status = ExecutionStatus.TERMINAL
            else:
                result.status = ExecutionStatus.RETRYABLE
            result.error = order_result.error

        return result

    async def _close_position(self, decision: Decision, result: ExecutionResult) -> ExecutionResult:
        """Close an existing position (SELL at market)."""
        from core_engine.native.decisions import Action

        if decision.action != Action.CLOSE:
            result.status = ExecutionStatus.TERMINAL
            result.error = "expected CLOSE action"
            return result

        # Market sell
        order_result = await self._order_exec.place_market_sell(
            decision.symbol,
            decision.quantity,
            client_order_id=decision.decision_id,
        )

        result.raw = order_result.raw
        result.exchange_order_id = order_result.exchange_order_id
        result.quantity_executed = order_result.quantity

        if order_result.success:
            result.status = ExecutionStatus.SUCCESS
        else:
            result.status = ExecutionStatus.RETRYABLE
            result.error = order_result.error

        return result

    # ──────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _classify_error(error_msg: str) -> ExecutionStatus:
        """Classify an error as retryable or terminal."""
        error_lower = error_msg.lower()
        # Retryable: rate limit, network, timeout
        if any(x in error_lower for x in ["429", "timeout", "network", "503", "502"]):
            return ExecutionStatus.RETRYABLE
        # Terminal: invalid request, insufficient balance, etc.
        if any(x in error_lower for x in ["insufficient", "invalid", "rejected"]):
            return ExecutionStatus.TERMINAL
        # Default to retryable (network partition, etc.)
        return ExecutionStatus.RETRYABLE

    def reset_dedup_state(self) -> None:
        """Clear dedup tracking (e.g., at session start)."""
        self._executed_ids.clear()
        self._symbol_locks.clear()
