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

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from core_engine.native.decisions import Decision
from core_engine.native.order_execution import ExchangeClientError, NativeOrderExecution

from .balance_validator import NativeBalanceValidator

logger = logging.getLogger(__name__)


def commission_quote_from_fills(
    fills: Any, *, symbol: str, fallback_price: float = 0.0, price_cache: Optional[dict[str, float]] = None,
) -> float:
    """Sum a Binance order response's ``fills[].commission`` in quote-currency terms.

    ``fills`` entries carry ``commission``/``commissionAsset`` independently
    per fill (Binance can split one order across multiple price levels, each
    with its own commission entry) — quote-asset commissions pass through
    as-is, base-asset commissions convert via the fill's own price, and any
    other asset (e.g. BNB fee discount) converts via ``price_cache`` if a
    ``<asset><quote>`` rate is available, else contributes 0 rather than guess.
    """
    if not isinstance(fills, list) or not fills:
        return 0.0
    symbol_u = symbol.upper()
    quote_asset = "USDT" if symbol_u.endswith("USDT") else ""
    base_asset = symbol_u[: -len(quote_asset)] if quote_asset else ""
    price_cache = price_cache or {}
    total = 0.0
    for f in fills:
        try:
            commission = max(0.0, float(f.get("commission") or 0.0))
        except (TypeError, ValueError, AttributeError):
            continue
        if commission <= 0:
            continue
        fee_asset = str(f.get("commissionAsset") or "").upper()
        if fee_asset == quote_asset:
            total += commission
        elif fee_asset == base_asset:
            fill_price = float(f.get("price") or fallback_price or 0.0)
            total += commission * fill_price
        else:
            conversion = float(price_cache.get(f"{fee_asset}{quote_asset}", 0.0) or 0.0)
            if conversion > 0:
                total += commission * conversion
    return total


def compute_net_trade_pnl(
    *,
    entry_price: float,
    exit_price: float,
    qty: float,
    entry_commission_quote: float = 0.0,
    exit_commission_quote: float = 0.0,
) -> float:
    """Canonical net-of-fee realized P&L for one closed trade cycle.

    ``entry_price``/``exit_price`` are actual fill prices (not pre-trade
    reference prices), so adverse slippage is already reflected in the gross
    term — this is deliberately the ONE place in the codebase that defines
    "net realized profit" so a "net-profitable trade" claim doesn't require
    manual reconciliation against Binance's own trade history.
    """
    gross_pnl = (exit_price - entry_price) * qty
    return gross_pnl - max(0.0, entry_commission_quote) - max(0.0, exit_commission_quote)


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
        shared_state: Optional[Any] = None,
        balance_validator: Optional[NativeBalanceValidator] = None,
        trade_journal: Optional[Any] = None,
        tp_sl_engine: Optional[Any] = None,
        daily_target_monitor: Optional[Any] = None,
    ) -> None:
        self._order_exec = order_execution
        self._market_data = market_data  # for price lookups when decision.quantity is in USD
        self._exchange_client = exchange_client  # for symbol filters (LOT_SIZE, MIN_NOTIONAL)
        self._shared_state = shared_state
        self._balance_validator = balance_validator
        self._trade_journal = trade_journal
        self._tp_sl_engine = tp_sl_engine
        self._daily_target_monitor = daily_target_monitor
        self._executed_ids: set[str] = set()  # dedup tracking
        self._symbol_locks: dict[str, float] = {}  # symbol → last execution ts
        self._entry_commission_quote: dict[str, float] = {}  # symbol → accumulated entry-side fee (quote)
        self._symbol_filters_cache: dict[str, dict[str, Any]] = {}  # symbol → filters
        # Reject a BUY when the cached price is older than this (seconds). A stale
        # price corrupts both order sizing and the SL anchor → instant phantom stop-outs.
        import os as _os

        self._price_max_age_s: float = float(_os.getenv("PRICE_MAX_AGE_S") or 30.0)
        # If the cached price deviates from a fresh REST quote by more than this
        # fraction, treat it as stale and use the REST price for sizing/entry.
        self._price_max_deviation: float = float(_os.getenv("PRICE_MAX_DEVIATION") or 0.015)

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

    async def place_order(
        self,
        *,
        symbol: str,
        quantity: float,
        action: str,
        price: float | None = None,
        order_type: str = "MARKET",
    ) -> dict[str, Any]:
        """
        Lightweight façade adapter used by SafeExecutionEngine.

        Quantity is interpreted the same way as the native decision path:
        BUY quantities are quote-USDT intent and are converted downstream using
        current price when available; SELL quantities are base-asset quantity.
        """
        from core_engine.native.decisions import Action, Decision

        decision = Decision(
            symbol=symbol,
            action=Action.OPEN if str(action).upper() == "BUY" else Action.CLOSE,
            quantity=float(quantity or 0.0),
            reason=f"facade_{str(action).lower()}",
            risk_score=0.5,
        )

        results = await self.execute([decision])
        if not results:
            return {"status": "FAILED", "error": "no execution result"}

        result = results[0]
        status = result.status.value if hasattr(result.status, "value") else str(result.status)
        return {
            "orderId": result.exchange_order_id,
            "executedQty": result.quantity_executed,
            "price": float(price or 0.0),
            "status": status,
            "error": result.error,
        }

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
    @staticmethod
    def _raw_executed_qty(order: Any) -> float:
        """Authoritative filled base qty. refresh_status updates `raw` but not
        `executed_qty`, so read the fresh raw first, then fall back to the field."""
        try:
            rq = float((getattr(order, "raw", {}) or {}).get("executedQty", 0.0) or 0.0)
        except (TypeError, ValueError):
            rq = 0.0
        eq = float(getattr(order, "executed_qty", None) or 0.0)
        return max(rq, eq)

    def _record_execution_quality(
        self,
        *,
        side: str,
        reference_price: float,
        fill_price: float,
        is_maker: bool,
    ) -> dict[str, Any]:
        """Persist observed fill quality for cost calibration.

        ``avg_slippage_bps`` is deliberately an adverse-cost measure because it is
        consumed by entry and sizing gates. Favorable fills are retained separately
        as price improvement instead of being allowed to hide adverse fills.
        """
        side_upper = str(side).upper()
        if reference_price <= 0 or fill_price <= 0 or side_upper not in {"BUY", "SELL"}:
            return {}
        signed_adverse_bps = (
            (fill_price / reference_price - 1.0) * 10_000.0
            if side_upper == "BUY"
            else (reference_price / fill_price - 1.0) * 10_000.0
        )
        adverse_bps = max(0.0, signed_adverse_bps)
        improvement_bps = max(0.0, -signed_adverse_bps)
        quality = {
            "side": side_upper,
            "reference_price": float(reference_price),
            "fill_price": float(fill_price),
            "adverse_slippage_bps": float(adverse_bps),
            "price_improvement_bps": float(improvement_bps),
            "liquidity_role": "maker" if is_maker else "taker",
        }
        metrics = getattr(self._shared_state, "metrics", None)
        if isinstance(metrics, dict):
            count = int(metrics.get("execution_quality_samples", 0) or 0)
            next_count = count + 1
            for key, value in (
                ("avg_slippage_bps", adverse_bps),
                ("avg_price_improvement_bps", improvement_bps),
            ):
                previous = float(metrics.get(key, 0.0) or 0.0)
                metrics[key] = ((previous * count) + float(value)) / next_count
            side_key = f"avg_{side_upper.lower()}_slippage_bps"
            side_count_key = f"{side_upper.lower()}_execution_samples"
            side_count = int(metrics.get(side_count_key, 0) or 0)
            previous_side = float(metrics.get(side_key, 0.0) or 0.0)
            metrics[side_key] = ((previous_side * side_count) + adverse_bps) / (side_count + 1)
            metrics[side_count_key] = side_count + 1
            metrics["execution_quality_samples"] = next_count
            maker_fills = int(metrics.get("maker_fills", 0) or 0) + (1 if is_maker else 0)
            metrics["maker_fills"] = maker_fills
            metrics["taker_fills"] = next_count - maker_fills
            metrics["maker_fill_rate"] = maker_fills / next_count
            metrics["last_execution_quality"] = quality
            metrics["last_update_ts"] = time.time()
        logger.info(
            "[Executor:Quality] %s %s ref=%.8f fill=%.8f slip=%.2fbps improve=%.2fbps",
            side_upper,
            quality["liquidity_role"],
            reference_price,
            fill_price,
            adverse_bps,
            improvement_bps,
        )
        return quality

    async def _maker_first_buy(
        self, symbol: str, qty: float, ref_price: float, coid: str
    ) -> Any:
        """Try a maker (LIMIT) BUY just below ``ref_price`` for a short grace window to
        save the taker fee, then fall back to MARKET for fill certainty.

        Safety: never double-buys. On a *partial* maker fill we keep the partial (the
        downstream registers the real executed qty); only on a *zero* fill do we
        market-buy the full qty. A cancel/fill race is resolved by re-reading the
        executed qty after cancel. Returns an OrderResult-compatible object.
        """
        enabled = str(os.getenv("MAKER_ENTRY_ENABLED", "true")).lower() in (
            "1", "true", "yes", "on",
        )
        grace_s = float(os.getenv("MAKER_GRACE_S", "2.0") or 2.0)
        offset_bps = float(os.getenv("MAKER_OFFSET_BPS", "1.0") or 1.0)
        # Disabled / no usable price → plain market order (today's behaviour).
        if not enabled or grace_s <= 0 or ref_price <= 0 or qty <= 0:
            return await self._order_exec.place_market_buy(symbol, qty, client_order_id=coid)

        mk_coid = (coid + "-mk")[:36]
        limit_px = ref_price * (1.0 - offset_bps / 10000.0)
        lr = await self._order_exec.place_limit_buy(symbol, qty, limit_px, client_order_id=coid)
        if not lr.success:
            logger.info("[Executor] maker limit place failed for %s — market fallback", symbol)
            return await self._order_exec.place_market_buy(symbol, qty, client_order_id=mk_coid)

        # Give the resting limit a brief window to fill as maker.
        await asyncio.sleep(grace_s)
        try:
            st = await self._order_exec.refresh_status(symbol, lr.client_order_id)
        except Exception as e:
            logger.debug("[Executor] maker refresh_status failed for %s: %s", symbol, e)
            st = lr
        executed = self._raw_executed_qty(st)
        if executed >= qty * 0.999 or str(getattr(st, "status", "")).upper() == "FILLED":
            logger.info(
                "[Executor] ✅ maker fill %s qty=%.6f @ ~%.6f (saved taker fee)",
                symbol, executed or qty, limit_px,
            )
            return st

        # Not fully filled within grace → cancel the resting order, then resolve the
        # final executed qty (cancel may race a fill).
        try:
            await self._order_exec.cancel(symbol, lr.client_order_id)
        except Exception as e:
            logger.debug("[Executor] maker cancel failed for %s: %s", symbol, e)
        final_exec = executed
        try:
            fs = await self._order_exec.refresh_status(symbol, lr.client_order_id)
            final_exec = max(final_exec, self._raw_executed_qty(fs))
        except Exception:
            fs = st
        if final_exec >= qty * 0.999:
            logger.info("[Executor] maker filled during cancel race %s qty=%.6f", symbol, final_exec)
            return fs
        if final_exec > 0:
            # Partial maker fill — keep it; do NOT market the remainder (avoids double-buy).
            logger.info(
                "[Executor] partial maker fill %s: kept %.6f of %.6f (no taker remainder)",
                symbol, final_exec, qty,
            )
            return fs
        # Zero fill → market-buy the full qty (taker fallback).
        logger.info(
            "[Executor] maker unfilled %s after %.1fs — market fallback (taker)", symbol, grace_s
        )
        return await self._order_exec.place_market_buy(symbol, qty, client_order_id=mk_coid)

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

        # ── Price-freshness guard ──────────────────────────────────────────
        # A stale cached price corrupts order sizing AND the SL anchor, causing
        # instant phantom stop-outs against the real market. If the cached price
        # is too old, refresh it via REST; reject the BUY if we still can't get a
        # fresh, agreeing price.
        price = await self._ensure_fresh_price(decision.symbol, price, result)
        if result.status == ExecutionStatus.TERMINAL:
            return result

        self._sync_balance_validator()
        if self._balance_validator is not None:
            ok, _, reason = self._balance_validator.validate_allocation(
                amount=float(decision.quantity or 0.0),
                symbol=decision.symbol,
                side="BUY",
                order_id=decision.decision_id,
            )
            if not ok:
                result.status = ExecutionStatus.TERMINAL
                result.error = f"balance validation failed: {reason}"
                return result

        if price and price > 0:
            qty_to_place = decision.quantity / price
            logger.debug(
                "converted USD %.2f → %.6f @ $%.2f", decision.quantity, qty_to_place, price
            )
        else:
            # decision.quantity is a USD quote intent; without a price we cannot
            # convert it to a base quantity safely. Treating the USD figure as base
            # qty would place a wildly mis-sized order (e.g. $6 → 6 base units), so
            # reject and let the next cycle retry once a price is available.
            result.status = ExecutionStatus.TERMINAL
            result.error = f"BUY rejected: price unavailable for {decision.symbol}, cannot size order"
            logger.warning(
                "❌ BUY rejected (no price): %s quote=$%.2f — cannot convert to base qty",
                decision.symbol,
                float(decision.quantity or 0.0),
            )
            return result

        # Layer 2: Validate LOT_SIZE and MIN_NOTIONAL constraints
        if price and price > 0:
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

        # Maker-first BUY: try a resting LIMIT at/just-below price to save the taker
        # fee, with a MARKET fallback for fill certainty. Falls back to market if maker
        # is disabled or unavailable.
        order_result = await self._maker_first_buy(
            decision.symbol,
            qty_to_place,
            float(price or 0.0),
            decision.decision_id,  # idempotency key for the limit leg
        )

        result.raw = dict(order_result.raw or {})
        result.exchange_order_id = order_result.exchange_order_id
        result.quantity_executed = float(
            getattr(order_result, "executed_qty", None) or order_result.quantity
        )

        if self._daily_target_monitor is not None:
            try:
                self._daily_target_monitor.record_order_submitted(decision.symbol)
            except Exception:
                pass

        if order_result.success:
            result.status = ExecutionStatus.SUCCESS
            if self._daily_target_monitor is not None:
                try:
                    self._daily_target_monitor.record_entry_filled(decision.symbol)
                except Exception:
                    pass
            if self._balance_validator is not None:
                self._balance_validator.commit_allocation(
                    amount=float(decision.quantity or 0.0),
                    symbol=decision.symbol,
                    side="BUY",
                    order_id=decision.decision_id,
                )
            if self._shared_state is not None and hasattr(self._shared_state, "reserve_quote"):
                self._shared_state.reserve_quote(
                    "USDT",
                    float(decision.quantity or 0.0),
                    reservation_id=decision.decision_id,
                )
            logger.info(
                "✅ Order placed: %s qty=%.6f status=%s",
                decision.symbol,
                qty_to_place,
                getattr(order_result, "status", "accepted"),
            )
            # Real volume-weighted fill price from the exchange response — never the
            # pre-trade cached price (which may be stale and misanchor the SL).
            fill_price = float(
                getattr(order_result, "avg_price", None)
                or getattr(order_result, "price", None)
                or price
                or 0.0
            )
            if fill_price <= 0:
                fill_price = float(price or 0.0)
            quality = self._record_execution_quality(
                side="BUY",
                reference_price=float(price or 0.0),
                fill_price=fill_price,
                is_maker=str(getattr(order_result, "order_type", "")).upper() == "LIMIT",
            )
            if quality:
                result.raw["_execution_quality"] = quality
            # Use the ACTUAL executed quantity (partial fills leave us holding less than
            # requested). Registering the requested qty over-states the position and
            # triggers -2010 on the eventual sell. Fall back to requested if unavailable.
            filled_qty = float(getattr(order_result, "executed_qty", None) or qty_to_place)
            if abs(filled_qty - qty_to_place) > qty_to_place * 0.001:
                logger.info(
                    "[Executor] %s partial/adjusted fill: requested %.6f, executed %.6f",
                    decision.symbol, qty_to_place, filled_qty,
                )
            # Track entry-side commission so _close_position can compute a true
            # net-of-fee realized P&L instead of a gross-only, never-persisted number.
            entry_fee_quote = commission_quote_from_fills(
                order_result.raw.get("fills") if order_result.raw else None,
                symbol=decision.symbol,
                fallback_price=fill_price,
                price_cache=getattr(self._shared_state, "price_cache", None),
            )
            if entry_fee_quote > 0:
                self._entry_commission_quote[decision.symbol] = (
                    self._entry_commission_quote.get(decision.symbol, 0.0) + entry_fee_quote
                )
            # Register position immediately — fill_tracker is None in polling mode,
            # so executor is the only place that can write this synchronously.
            if self._shared_state is not None and hasattr(self._shared_state, "update_position"):
                current_pos = self._shared_state.get_position(decision.symbol) if hasattr(self._shared_state, "get_position") else None
                if current_pos is not None:
                    old_qty = float(getattr(current_pos, "qty", 0.0) or 0.0)
                    old_entry = float(getattr(current_pos, "entry_price", 0.0) or 0.0)
                    new_qty = old_qty + filled_qty
                    avg_entry = ((old_qty * old_entry) + (filled_qty * fill_price)) / new_qty if new_qty > 0 else fill_price
                    self._shared_state.update_position(decision.symbol, new_qty, avg_entry, fill_price)
                else:
                    self._shared_state.update_position(decision.symbol, filled_qty, fill_price, fill_price)
                logger.info(
                    "[Executor] Position registered: %s qty=%.2f entry=%.8f",
                    decision.symbol, filled_qty, fill_price,
                )
                # Arm TP/SL immediately so time-based and trailing logic starts from fill
                if self._tp_sl_engine is not None and hasattr(self._tp_sl_engine, "arm_position"):
                    try:
                        import time as _time
                        self._tp_sl_engine.arm_position(decision.symbol, fill_price, entry_ts=_time.time())
                    except Exception as _tpsl_err:
                        logger.debug("[Executor] TP/SL arm failed (non-fatal): %s", _tpsl_err)
            if self._trade_journal is not None:
                import asyncio as _asyncio
                _asyncio.ensure_future(self._trade_journal.record("BUY_FILLED", {
                    "symbol": decision.symbol,
                    "qty": filled_qty,
                    "price": fill_price or price or 0.0,
                    "quote_qty": decision.quantity,
                    "order_id": result.exchange_order_id,
                    "reason": getattr(decision, "reason", ""),
                    "execution_quality": quality,
                }))
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

        # Round quantity to step size before sending to exchange
        qty_to_sell = decision.quantity
        # Full-balance sell: prefer the REAL exchange-held base balance over the tracked
        # decision.quantity so we don't orphan a dust remainder from tracked-vs-real drift
        # (e.g. ADA sold 58.1 of a tracked 58.2, leaving 0.1 dust). Only override when the
        # real balance is positive and within 5% of the tracked qty — a larger gap means a
        # stale balance poll, so we trust the tracked qty instead of selling a wrong amount.
        try:
            _real_bal = self._real_base_balance(decision.symbol)
            if _real_bal > 0 and qty_to_sell > 0:
                _drift = abs(_real_bal - qty_to_sell) / qty_to_sell
                if _drift <= 0.05:
                    if _real_bal != qty_to_sell:
                        logger.info(
                            "[Executor] full-balance sell %s: tracked=%.8f → real=%.8f (drift=%.2f%%)",
                            decision.symbol, qty_to_sell, _real_bal, _drift * 100.0,
                        )
                    qty_to_sell = _real_bal
        except Exception as _e:
            logger.debug("[Executor] full-balance sell lookup failed for %s: %s", decision.symbol, _e)
        sell_price = 0.0
        if self._market_data is not None:
            try:
                sell_price = float(self._market_data.get_price(decision.symbol) or 0.0)
            except Exception:
                pass
        if sell_price <= 0 and self._shared_state is not None:
            try:
                sell_price = float(
                    getattr(self._shared_state, "price_cache", {}).get(decision.symbol, 0.0) or 0.0
                )
            except Exception:
                pass
        if sell_price <= 0:
            result.status = ExecutionStatus.TERMINAL
            result.error = f"SELL rejected: price unavailable for {decision.symbol}, cannot verify MIN_NOTIONAL"
            logger.warning("❌ SELL rejected (no price): %s qty=%.8f", decision.symbol, qty_to_sell)
            return result
        valid, error_msg, corrected_qty = await self._validate_lot_size(
            decision.symbol, qty_to_sell, price=sell_price
        )
        if not valid:
            result.status = ExecutionStatus.TERMINAL
            result.error = f"LOT_SIZE validation failed (SELL): {error_msg}"
            logger.warning(
                "❌ SELL rejected (LOT_SIZE): %s qty=%.8f reason=%s",
                decision.symbol, decision.quantity, error_msg,
            )
            return result
        if corrected_qty > 0:
            qty_to_sell = corrected_qty

        # Market sell
        order_result = await self._order_exec.place_market_sell(
            decision.symbol,
            qty_to_sell,
            client_order_id=decision.decision_id,
        )

        result.raw = dict(order_result.raw or {})
        result.exchange_order_id = order_result.exchange_order_id
        result.quantity_executed = float(
            getattr(order_result, "executed_qty", None) or order_result.quantity
        )

        if order_result.success:
            result.status = ExecutionStatus.SUCCESS
            fill_price = float(
                getattr(order_result, "avg_price", None)
                or getattr(order_result, "price", None)
                or sell_price
                or 0.0
            )
            quality = self._record_execution_quality(
                side="SELL",
                reference_price=sell_price,
                fill_price=fill_price,
                is_maker=str(getattr(order_result, "order_type", "")).upper() == "LIMIT",
            )
            if quality:
                result.raw["_execution_quality"] = quality
            self._sync_balance_validator()
            if self._balance_validator is not None and self._market_data is not None:
                price = 0.0
                try:
                    price = float(self._market_data.get_price(decision.symbol) or 0.0)
                except Exception:
                    price = 0.0
                release_amount = float(decision.quantity or 0.0) * max(0.0, price)
                if release_amount > 0:
                    self._balance_validator.release_allocation(
                        amount=release_amount,
                        symbol=decision.symbol,
                        order_id=decision.decision_id,
                    )
            if self._shared_state is not None:
                if hasattr(self._shared_state, "release_quote_reservation"):
                    self._shared_state.release_quote_reservation("USDT", decision.decision_id)
                # Clear position from state — balance polling will re-hydrate if any remainder
                if hasattr(self._shared_state, "close_position"):
                    # Capture entry price BEFORE clearing so we can log realized P&L.
                    _entry_px = 0.0
                    try:
                        _cur_pos = (
                            self._shared_state.get_position(decision.symbol)
                            if hasattr(self._shared_state, "get_position")
                            else None
                        )
                        _entry_px = float(
                            getattr(_cur_pos, "entry_price", None)
                            or (_cur_pos.get("entry_price", 0) if isinstance(_cur_pos, dict) else 0)
                            or 0.0
                        )
                    except Exception:
                        _entry_px = 0.0
                    self._shared_state.close_position(decision.symbol)
                    if _entry_px > 0 and fill_price > 0:
                        _qty = float(qty_to_sell or 0.0)
                        _gross_pnl = (fill_price - _entry_px) * _qty
                        _pnl_pct = (fill_price / _entry_px - 1.0) * 100.0
                        _entry_fee = self._entry_commission_quote.pop(decision.symbol, 0.0)
                        _exit_fee = commission_quote_from_fills(
                            result.raw.get("fills") if result.raw else None,
                            symbol=decision.symbol,
                            fallback_price=fill_price,
                            price_cache=getattr(self._shared_state, "price_cache", None),
                        )
                        _net_pnl = compute_net_trade_pnl(
                            entry_price=_entry_px, exit_price=fill_price, qty=_qty,
                            entry_commission_quote=_entry_fee, exit_commission_quote=_exit_fee,
                        )
                        _outcome = "WIN" if _net_pnl > 0 else ("LOSS" if _net_pnl < 0 else "FLAT")
                        logger.info(
                            "[Executor] TRADE_CLOSED %s %s entry=%.8f exit=%.8f qty=%.8f "
                            "gross_pnl=%.4f net_pnl=%.4f USDT pnl_pct=%.3f%% "
                            "entry_fee=%.6f exit_fee=%.6f reason=%s",
                            decision.symbol, _outcome, _entry_px, fill_price, _qty,
                            _gross_pnl, _net_pnl, _pnl_pct, _entry_fee, _exit_fee,
                            getattr(decision, "reason", ""),
                        )
                        metrics = getattr(self._shared_state, "metrics", None)
                        if isinstance(metrics, dict):
                            metrics["realized_pnl"] = metrics.get("realized_pnl", 0.0) + _net_pnl
                            metrics["trades_in_window"] = metrics.get("trades_in_window", 0) + 1
                            metrics["last_update_ts"] = time.time()
                            prev_wr = metrics.get("win_rate_window", 0.5)
                            metrics["win_rate_window"] = prev_wr * 0.9 + (1.0 if _net_pnl > 0 else 0.0) * 0.1
                        if self._daily_target_monitor is not None:
                            try:
                                self._daily_target_monitor.record_trade_closed(
                                    decision.symbol,
                                    gross_pnl=_gross_pnl,
                                    net_pnl=_net_pnl,
                                    fees=_entry_fee + _exit_fee,
                                    compoundable=_net_pnl > 0,
                                )
                            except Exception as _dtm_err:
                                logger.debug(
                                    "[Executor] daily_target_monitor record failed for %s: %s",
                                    decision.symbol, _dtm_err,
                                )
                    else:
                        logger.info(
                            "[Executor] SELL %s filled — position cleared from state "
                            "(realized P&L unavailable: entry=%.8f exit=%.8f)",
                            decision.symbol, _entry_px, fill_price,
                        )
            # P1: Atomic TP/SL cleanup — clear registries immediately on SELL fill,
            # don't wait for fill_tracker (5s lag can allow ghost to be written to disk).
            if self._tp_sl_engine is not None:
                try:
                    self._tp_sl_engine.close_position_tracking(decision.symbol)
                except Exception as _e:
                    logger.debug("[Executor] close_position_tracking failed for %s: %s", decision.symbol, _e)
            if self._trade_journal is not None:
                import asyncio as _asyncio
                _asyncio.ensure_future(self._trade_journal.record("SELL_FILLED", {
                    "symbol": decision.symbol,
                    "qty": qty_to_sell,
                    "price": fill_price,
                    "reference_price": sell_price,
                    "order_id": result.exchange_order_id,
                    "reason": getattr(decision, "reason", ""),
                    "execution_quality": quality,
                }))
        else:
            error_str = str(order_result.error or "")
            # -2010 = insufficient balance. This does NOT always mean the position is
            # gone — it also fires when our tracked qty is slightly stale (rounding,
            # locked balance, dust). Blindly removing the position orphans a real
            # holding: it then has no TP/SL and vanishes from NAV. So re-check the
            # actual exchange balance before deciding.
            if "-2010" in error_str or "insufficient balance" in error_str.lower():
                result.status = ExecutionStatus.TERMINAL
                real_qty = self._real_base_balance(decision.symbol)
                sell_px = 0.0
                if self._market_data is not None:
                    try:
                        sell_px = float(self._market_data.get_price(decision.symbol) or 0.0)
                    except Exception:
                        sell_px = 0.0
                real_notional = real_qty * sell_px if sell_px > 0 else real_qty
                # Treat as truly closed only if the asset is effectively gone (dust).
                gone = (real_qty <= 1e-8) or (sell_px > 0 and real_notional < 1.0)
                if gone:
                    if self._shared_state is not None and hasattr(self._shared_state, "close_position"):
                        self._shared_state.close_position(decision.symbol)
                        logger.warning(
                            "[Executor] SELL %s -2010 and balance≈0 (qty=%.8f) — confirmed cleared, removing",
                            decision.symbol, real_qty,
                        )
                elif self._shared_state is not None and hasattr(self._shared_state, "update_position"):
                    # Still held — correct the tracked qty to the real balance and KEEP it
                    # managed so TP/SL keeps watching it (preserve the original entry price).
                    cur = self._shared_state.get_position(decision.symbol) if hasattr(self._shared_state, "get_position") else None
                    entry = float(getattr(cur, "entry_price", None) or (cur.get("entry_price", 0) if isinstance(cur, dict) else 0) or sell_px or 0.0)
                    self._shared_state.update_position(decision.symbol, real_qty, entry or sell_px, sell_px or entry)
                    if self._tp_sl_engine is not None and hasattr(self._tp_sl_engine, "arm_position") and (entry or sell_px) > 0:
                        try:
                            import time as _time
                            self._tp_sl_engine.arm_position(decision.symbol, entry or sell_px, entry_ts=_time.time())
                        except Exception:
                            pass
                    logger.warning(
                        "[Executor] SELL %s -2010 but still held (qty=%.8f ~$%.2f) — qty re-synced, kept managed",
                        decision.symbol, real_qty, real_notional,
                    )
            else:
                result.status = ExecutionStatus.RETRYABLE
            result.error = error_str
            logger.warning(
                "[Executor] SELL failed for %s qty=%.6f: %s",
                decision.symbol, decision.quantity, error_str,
            )

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
                    "rounded qty %.8f → %.8f (step_size %.8f)", qty, corrected_qty, step_size
                )

        return True, "", corrected_qty

    def _classify_error(error_msg: str) -> ExecutionStatus:
        """Classify an error as retryable or terminal."""
        error_lower = error_msg.lower()
        # Retryable: rate limit, network, timeout, cooldown
        if any(x in error_lower for x in ["429", "timeout", "network", "503", "502", "cooldown"]):
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

    async def _ensure_fresh_price(
        self, symbol: str, cached_price: Optional[float], result: ExecutionResult
    ) -> Optional[float]:
        """Return a trustworthy price for sizing/entry, or mark result TERMINAL.

        Refreshes via REST when the cached tick is stale or missing. If a fresh
        REST quote disagrees with the cached price beyond the deviation tolerance,
        the REST price wins (it reflects the market the order will actually fill at).
        """
        age = float("inf")
        if self._shared_state is not None and hasattr(self._shared_state, "get_price_age"):
            try:
                age = self._shared_state.get_price_age(symbol)
            except Exception:
                age = float("inf")

        stale = (not cached_price or cached_price <= 0) or (age > self._price_max_age_s)

        # Always cross-check against a fresh REST quote when cached looks stale.
        if stale and self._exchange_client is not None:
            try:
                quotes = await self._exchange_client.get_prices([symbol])
                rest_price = float(quotes.get(symbol, 0.0) or 0.0)
            except Exception as e:
                rest_price = 0.0
                logger.warning("[Executor] REST price refresh failed for %s: %s", symbol, e)

            if rest_price > 0:
                # Push the fresh price back into shared_state so the SL anchor and
                # downstream consumers use it too.
                if self._shared_state is not None and hasattr(self._shared_state, "update_price"):
                    try:
                        self._shared_state.update_price(symbol, rest_price)
                    except Exception:
                        pass
                if cached_price and cached_price > 0:
                    dev = abs(rest_price - cached_price) / cached_price
                    if dev > self._price_max_deviation:
                        logger.warning(
                            "[Executor] %s cached price %.8f stale (age=%.0fs, dev=%.2f%% vs REST %.8f) — using REST",
                            symbol, cached_price, age, dev * 100.0, rest_price,
                        )
                return rest_price

            # No fresh price available and cached is stale → refuse to trade blind.
            if stale and (not cached_price or cached_price <= 0 or age > self._price_max_age_s):
                result.status = ExecutionStatus.TERMINAL
                result.error = (
                    f"stale_price: {symbol} age={age:.0f}s > {self._price_max_age_s:.0f}s "
                    f"and REST refresh unavailable"
                )
                logger.warning(
                    "❌ BUY rejected (stale price): %s age=%.0fs cached=%s — no fresh quote",
                    symbol, age, cached_price,
                )
                return cached_price

        return cached_price

    def _real_base_balance(self, symbol: str) -> float:
        """Actual exchange-held quantity of the base asset (e.g. STRK for STRKUSDT).

        Reads the live balance map (populated by the balance poller, same source
        gate_4 uses). Used to decide whether a -2010 sell failure means the
        position is truly gone vs merely a stale tracked quantity.
        """
        if self._shared_state is None:
            return 0.0
        base = str(symbol or "").upper()
        if base.endswith("USDT"):
            base = base[:-4]
        bal = getattr(self._shared_state, "balance", {}) or {}
        return float(bal.get(base, 0.0) or 0.0)

    def _sync_balance_validator(self) -> None:
        if self._balance_validator is None or self._shared_state is None:
            return
        free_usdt = float(getattr(self._shared_state, "free_balance_usdt", 0.0) or 0.0)
        if free_usdt <= 0:
            balance = getattr(self._shared_state, "balance", {}) or {}
            free_usdt = float(balance.get("USDT", 0.0) or 0.0)
        self._balance_validator.set_total_balance(
            free_usdt + float(self._balance_validator.allocated_balance or 0.0)
        )
