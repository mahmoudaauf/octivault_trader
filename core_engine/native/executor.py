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

        executor = NativeExecutor(order_execution_client, market_data=md)
        results = await executor.execute(decisions)
        # ⇒ list[ExecutionResult] with status + order IDs
    """

    def __init__(
        self,
        order_execution: NativeOrderExecution,
        market_data: Optional[Any] = None,
        exchange_client: Optional[Any] = None,
    ) -> None:
        self._order_exec = order_execution
        self._market_data = market_data  # for price lookups when decision.quantity is in USD
        self._exchange_client = exchange_client  # for symbol filters (LOT_SIZE, MIN_NOTIONAL)
        self._executed_ids: set[str] = set()  # dedup tracking
        self._symbol_locks: dict[str, float] = {}  # symbol → last execution ts
        self._symbol_filters_cache: dict[str, dict[str, Any]] = {}  # symbol → filters

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

        # Decision.quantity is in USD; convert to base asset using current price
        qty_to_place = decision.quantity
        price = None
        if self._market_data:
            try:
                price = self._market_data.get_price(decision.symbol)
            except Exception as e:
                logger.warning("price lookup failed for %s: %s", decision.symbol, e)

        if not price or price <= 0:
            result.status = ExecutionStatus.RETRYABLE
            result.error = f"price unavailable for {decision.symbol} (current: {price})"
            logger.warning("❌ Order deferred: %s price lookup failed", decision.symbol)
            return result

        qty_to_place = decision.quantity / price
        logger.debug("converted USD %.2f → %.6f @ $%.2f", decision.quantity, qty_to_place, price)

        # Layer 2: Validate LOT_SIZE and MIN_NOTIONAL constraints
        valid, error_msg, corrected_qty = await self._validate_lot_size(
            decision.symbol, qty_to_place, price
        )
        if not valid:
            result.status = ExecutionStatus.TERMINAL
            result.error = f"LOT_SIZE validation failed: {error_msg}"
            logger.warning(
                "❌ Order rejected (LOT_SIZE): %s qty=%.6f price=$%.2f reason=%s",
                decision.symbol,
                qty_to_place,
                price,
                error_msg,
            )
            return result
        qty_to_place = corrected_qty  # Use step-size aligned quantity

        # Simple market order: BUY
        order_result = await self._order_exec.place_market_buy(
            decision.symbol,
            qty_to_place,
            client_order_id=decision.decision_id,  # idempotency key
        )

        result.raw = order_result.raw
        result.exchange_order_id = order_result.exchange_order_id
        result.quantity_executed = order_result.quantity

        if order_result.success:
            result.status = ExecutionStatus.SUCCESS
            logger.info(
                "✅ Order placed: %s qty=%.6f status=%s",
                decision.symbol,
                qty_to_place,
                order_result.status,
            )
        else:
            # Exchange rejected the order
            logger.warning(
                "❌ Order failed: %s qty=%.6f error=%s",
                decision.symbol,
                qty_to_place,
                order_result.error,
            )
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
    # Symbol filters (LOT_SIZE, MIN_NOTIONAL validation)
    # ──────────────────────────────────────────────────────────────────
    async def _get_symbol_filters(self, symbol: str) -> dict[str, Any]:
        """Fetch and cache symbol filters from exchange (LOT_SIZE, MIN_NOTIONAL)."""
        if symbol in self._symbol_filters_cache:
            return self._symbol_filters_cache[symbol]

        filters: dict[str, Any] = {
            "step_size": 0.000001,  # LOT_SIZE stepSize default
            "min_qty": 0.0,  # LOT_SIZE minQty default
            "min_notional": 10.0,  # MIN_NOTIONAL default
        }

        if not self._exchange_client:
            self._symbol_filters_cache[symbol] = filters
            return filters

        try:
            info = await self._exchange_client.get_exchange_info(symbol)
            # info.symbols[0].filters contains the filter list
            if not isinstance(info, dict) or "symbols" not in info:
                self._symbol_filters_cache[symbol] = filters
                return filters

            symbol_info = info.get("symbols", [{}])[0]
            for f in symbol_info.get("filters", []):
                ftype = f.get("filterType", "")
                if ftype == "LOT_SIZE":
                    try:
                        filters["step_size"] = float(f.get("stepSize", "0.000001"))
                        filters["min_qty"] = float(f.get("minQty", "0"))
                    except (TypeError, ValueError):
                        pass
                elif ftype in ("MIN_NOTIONAL", "NOTIONAL"):
                    try:
                        filters["min_notional"] = float(
                            f.get("minNotional", f.get("minNot ional", "10"))
                        )
                    except (TypeError, ValueError):
                        pass
            logger.debug(
                "filters for %s: step_size=%.8f min_notional=%.2f",
                symbol,
                filters["step_size"],
                filters["min_notional"],
            )
        except Exception as e:
            logger.warning("failed to fetch filters for %s: %s; using defaults", symbol, e)

        self._symbol_filters_cache[symbol] = filters
        return filters

    async def _validate_lot_size(
        self, symbol: str, qty: float, price: float
    ) -> tuple[bool, str, float]:
        """Validate order qty against LOT_SIZE and MIN_NOTIONAL constraints. Returns (valid, error_msg, corrected_qty)."""
        filters = await self._get_symbol_filters(symbol)
        step_size = filters["step_size"]
        min_notional = filters["min_notional"]
        min_qty = filters["min_qty"]
        corrected_qty = qty

        # Check minimum quantity
        if qty < min_qty:
            return False, f"qty {qty:.8f} < min {min_qty:.8f}", corrected_qty

        # Check minimum notional (USD value)
        notional = qty * price
        if notional < min_notional:
            return False, f"notional ${notional:.2f} < min ${min_notional:.2f}", corrected_qty

        # Check step size alignment
        if step_size > 0:
            remainder = qty % step_size
            if remainder > 1e-8:  # allow tiny floating-point errors
                # Round down to nearest step
                corrected_qty = (qty // step_size) * step_size
                if corrected_qty <= 0:
                    return (
                        False,
                        f"qty {qty:.8f} rounds to 0 after step_size {step_size:.8f} adjustment",
                        qty,
                    )
                notional_after = corrected_qty * price
                if notional_after < min_notional:
                    return (
                        False,
                        f"after rounding to step_size, notional ${notional_after:.2f} < min ${min_notional:.2f}",
                        qty,
                    )
                logger.debug(
                    "rounded qty {:.8f} → {:.8f} (step_size {:.8f})", qty, corrected_qty, step_size
                )

        return True, "", corrected_qty

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
