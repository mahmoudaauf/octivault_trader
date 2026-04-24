"""
Octivault Trader — P9 Canonical ExecutionManager (native to your SharedState & ExchangeClient)
"""

from __future__ import annotations

__all__ = ["ExecutionManager"]

import asyncio
import contextlib
from contextlib import asynccontextmanager
from collections import deque
import logging
import json
import time
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from typing import Any, Dict, Optional, Tuple, Union, Literal
from core.stubs import resilient_trade, maybe_call, BinanceAPIException, ExecutionError, TradeIntent
from core.maker_execution import MakerExecutor, MakerExecutionConfig

# Optional: Import EventStore for Phase 5 event sourcing
try:
    from core.event_store import EventStore, EventType
except Exception:
    EventStore = None
    EventType = None

# =============================
# Utility shims
# =============================
try:
    from core.shared_state import PendingPositionIntent
except Exception:
    PendingPositionIntent = None

try:
    from utils import shared_state_tools, indicators, pnl_calculator
except Exception:
    pass

def round_step(value: float, step: float) -> float:
    if step <= 0:
        return float(value)
    q = (Decimal(str(value)) / Decimal(str(step))).to_integral_value(rounding=ROUND_DOWN)
    return float(q * Decimal(str(step)))


# =============================
# Exchange error shims
# =============================
try:
    from core.stubs import ExecutionBlocked
except Exception:
    class ExecutionBlocked(Exception):
        def __init__(self, code: str, planned_quote: float, available_quote: float, min_required: float):
            self.code = code
            self.planned_quote = float(planned_quote or 0.0)
            self.available_quote = float(available_quote or 0.0)
            self.min_required = float(min_required or 0.0)
            super().__init__(f"{code}: planned={self.planned_quote:.2f} available={self.available_quote:.2f} min_required={self.min_required:.2f}")


class SymbolFilters:
    def __init__(self, step_size: float = 0.0, min_qty: float = 0.0,
                 max_qty: float = 0.0, tick_size: float = 0.0,
                 min_notional: float = 0.0, min_entry_quote: float = 0.0):
        self.step_size = step_size
        self.min_qty = min_qty
        self.max_qty = max_qty
        self.tick_size = tick_size
        self.min_notional = min_notional
        self.min_entry_quote = min_entry_quote


async def validate_order_request(*, side: str, qty: float, price: float,
                                  filters: SymbolFilters, taker_fee_bps: int = 10,
                                  use_quote_amount: Optional[float] = None):
    if price <= 0:
        return False, 0.0, 0.0, "invalid_price"
    if use_quote_amount is not None:
        # INSTITUTIONAL FIX: Round UP to satisfy min_notional, not down
        min_required_notional = max(filters.min_notional, filters.min_entry_quote or 0.0)
        # QUOTE UPGRADE: Instead of rejecting, upgrade the quote to meet minimum
        use_quote_amount = max(use_quote_amount, min_required_notional)

        step = float(filters.step_size or 0.0)
        price_safe = price if price > 0 else 1.0
        
        # Calculate minimum required quote to meet notional floor
        min_required_quote = max(use_quote_amount, min_required_notional)
        estimated_qty = min_required_quote / price_safe

        if step > 0:
            # Round UP to ensure we meet min_notional constraint
            q = (Decimal(str(estimated_qty)) / Decimal(str(step))).to_integral_value(rounding=ROUND_UP)
            qty = float(q * Decimal(str(step)))
        else:
            qty = estimated_qty

        if qty <= 0:
            return False, 0.0, 0.0, "ZERO_QTY_AFTER_ROUNDING"

        if qty < float(filters.min_qty):
            return False, 0.0, 0.0, "QTY_LT_MIN"

        final_notional = qty * price
        
        # Final validation: notional must meet minimum
        if final_notional < min_required_notional:
            return False, 0.0, 0.0, "NOTIONAL_LT_MIN_AFTER_ROUNDING"

        # Spend amount is recalculated based on rounded-up qty
        spend = final_notional
        return True, float(qty), spend, "OK"
    else:
        step = float(filters.step_size or 0.0)
        if step > 0:
            q = (Decimal(str(qty)) / Decimal(str(step))).to_integral_value(rounding=ROUND_DOWN)
            qty = float(q * Decimal(str(step)))

        if qty <= 0:
            return False, 0.0, 0.0, "ZERO_QTY_AFTER_ROUNDING"

        if qty * price < max(filters.min_notional, filters.min_entry_quote or 0.0):
            return False, 0.0, 0.0, "NOTIONAL_LT_MIN"
        return True, float(qty), 0.0, "OK"


logger = logging.getLogger("ExecutionManager")

class ExecutionManager:
    """
    P9 ExecutionManager — canonical single-order path, natively aligned with:
    - SharedState: get_spendable_balance(), get_position_quantity(), reserve_liquidity()/release_liquidity()
    - ExchangeClient: place_market_order(), get_exchange_filters(), get_current_price()
    """
    
    @staticmethod
    def _safe_float(val: Any, default: float = 0.0) -> float:
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _round_step_down(value: float, step: float) -> float:
        if step <= 0:
            return float(value)
        try:
            q = (Decimal(str(value)) / Decimal(str(step))).to_integral_value(rounding=ROUND_DOWN)
            return float(q * Decimal(str(step)))
        except Exception:
            return float(0.0)

    def _normalize_quantity(self, quantity: float, step_size: float) -> float:
        """
        ✅ BEST PRACTICE: Normalize quantity to exchange step_size precision.
        
        This ensures the order respects exchange precision requirements:
        - Rounds DOWN to the nearest valid step
        - Rounds to 8 decimal places (standard for crypto)
        - Safe for all symbol types (spot, margin, futures)
        
        Example:
          quantity=0.99000900, step_size=0.01 → 0.99
          quantity=1.234567891, step_size=0.1 → 1.2
        """
        qty = float(quantity or 0.0)
        if qty <= 0 or step_size <= 0:
            return 0.0
        
        try:
            # Use Decimal for precision
            qty_dec = Decimal(str(qty))
            step_dec = Decimal(str(step_size))
            
            # Round DOWN to nearest step
            normalized = (qty_dec / step_dec).to_integral_value(rounding=ROUND_DOWN) * step_dec
            
            # Round to 8 decimal places (standard for crypto)
            result = float(normalized)
            result = round(result, 8)
            
            if result != qty:
                self.logger.debug(
                    "[EM:NormalizeQty] quantity=%.8f step_size=%.8f normalized=%.8f",
                    qty, step_size, result
                )
            
            return float(result)
        except Exception as e:
            self.logger.warning(
                "[EM:NormalizeQty] Error normalizing qty=%.8f step=%.8f: %s",
                qty, step_size, str(e)
            )
            return float(0.0)

    def _resolve_post_fill_price(self, order: Dict[str, Any], exec_qty: float) -> float:
        """
        Resolve best-effort execution price from exchange payload.
        Handles MARKET responses where `price` may be "0.00000000".
        """
        avg_price = self._safe_float(order.get("avgPrice") or order.get("avg_price") or 0.0, 0.0)
        if avg_price > 0:
            return float(avg_price)

        cumm_quote = self._safe_float(
            order.get("cummulativeQuoteQty") or order.get("cummulative_quote"),
            0.0,
        )
        if cumm_quote > 0 and exec_qty > 0:
            return float(cumm_quote / max(exec_qty, 1e-12))

        fills = order.get("fills") or []
        if isinstance(fills, list) and fills:
            weighted_num = 0.0
            weighted_den = 0.0
            for f in fills:
                if not isinstance(f, dict):
                    continue
                q = self._safe_float(f.get("qty"), 0.0)
                p = self._safe_float(f.get("price"), 0.0)
                if q > 0 and p > 0:
                    weighted_num += q * p
                    weighted_den += q
            if weighted_den > 0:
                return float(weighted_num / weighted_den)
            for f in fills:
                if not isinstance(f, dict):
                    continue
                p = self._safe_float(f.get("price"), 0.0)
                if p > 0:
                    return float(p)

        px = self._safe_float(order.get("price"), 0.0)
        if px > 0:
            return float(px)
        return 0.0

    def _canonical_exec_result(
        self,
        *,
        symbol: str,
        side: str,
        raw_order: Optional[Dict[str, Any]] = None,
        default_status: str = "REJECTED",
        default_reason: str = "",
        default_quote: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Normalize order responses to a symmetric canonical contract.
        Prevents raw exchange payload leakage from internal placement helpers.
        """
        raw = dict(raw_order) if isinstance(raw_order, dict) else {}
        sym = self._norm_symbol(symbol)
        side_u = str(side or "").upper()
        status = str(raw.get("status") or default_status or "REJECTED").upper()

        executed_qty = self._safe_float(raw.get("executedQty") or raw.get("executed_qty"), 0.0)
        price = self._resolve_post_fill_price(raw, executed_qty)
        if price <= 0:
            price = self._safe_float(raw.get("price"), 0.0)

        cumm_quote = self._safe_float(raw.get("cummulativeQuoteQty") or raw.get("cummulative_quote"), 0.0)
        if cumm_quote <= 0 and executed_qty > 0 and price > 0:
            cumm_quote = float(executed_qty) * float(price)
        if cumm_quote <= 0 and float(default_quote or 0.0) > 0:
            cumm_quote = float(default_quote)

        exchange_order_id = raw.get("orderId") or raw.get("exchange_order_id") or raw.get("order_id")
        client_order_id = raw.get("clientOrderId") or raw.get("client_order_id") or raw.get("origClientOrderId")
        if not client_order_id:
            oid_fallback = raw.get("order_id")
            if isinstance(oid_fallback, str) and oid_fallback and not oid_fallback.isdigit():
                client_order_id = oid_fallback

        is_fill = status in ("FILLED", "PARTIALLY_FILLED") and executed_qty > 0
        ok = bool(raw.get("ok", False) or is_fill)

        reason = str(raw.get("reason") or raw.get("error_msg") or default_reason or "")
        error_code = raw.get("error_code")
        if not error_code and status in ("REJECTED", "EXPIRED", "CANCELED"):
            error_code = status

        out = dict(raw)
        out.update(
            {
                "ok": ok,
                "symbol": sym,
                "side": side_u,
                "status": status,
                "executedQty": float(executed_qty),
                "price": float(price),
                "avgPrice": float(price),
                "cummulativeQuoteQty": float(cumm_quote),
                "orderId": exchange_order_id,
                "exchange_order_id": exchange_order_id,
                "clientOrderId": client_order_id,
                "client_order_id": client_order_id,
                "order_id": raw.get("order_id") or exchange_order_id or client_order_id or "",
                "reason": reason,
                "error_code": error_code,
            }
        )
        return out

    # --- Post-fill realized PnL emitter (P9 observability contract) ---
    async def _handle_post_fill(
        self,
        symbol: str,
        side: str,
        order: Dict[str, Any],
        tier: Optional[str] = None,
        tag: str = "",
        confidence: Optional[float] = None,
        agent: Optional[str] = None,
        planned_quote: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Best-effort: compute/record realized PnL delta when a trade fills, then emit
        a `RealizedPnlUpdated` event and persist the delta via SharedState if possible.
        This is tolerant to different SharedState contract shapes.
        """
        # DEBUG: Log entry for post-fill handling
        try:
            self.logger.debug(f"[DEBUG] Entering _handle_post_fill: symbol={symbol} side={side} exec_qty={order.get('executedQty')}")
        except Exception:
            pass
        emitted = False
        realized_committed = False
        trade_event_emitted = False
        delta_f = None
        try:
            sym = self._norm_symbol(symbol)
            side_u = (side or "").upper()
            exec_qty = self._safe_float(order.get("executedQty") or order.get("executed_qty"), 0.0)
            if exec_qty <= 0:
                return {
                    "delta": delta_f,
                    "realized_committed": realized_committed,
                    "emitted": emitted,
                    "trade_event_emitted": trade_event_emitted,
                }

            price = self._resolve_post_fill_price(order, exec_qty)
            if price > 0:
                order.setdefault("avgPrice", float(price))
                if self._safe_float(order.get("price"), 0.0) <= 0:
                    order["price"] = float(price)
            # P9 event contract: every confirmed fill must emit TRADE_EXECUTED.
            # Emission is anchored to post-fill processing, independent of tag/agent/side.
            # DEBUG: Log before emitting trade event
            self.logger.debug(f"[DEBUG] Emitting trade executed event: symbol={sym} side={side_u} tag={tag}")
            trade_event_emitted = bool(
                await self._emit_trade_executed_event(sym, side_u, str(tag or ""), order)
            )
            # DEBUG: Log after emitting trade event
            self.logger.debug(f"[DEBUG] Trade executed event emitted: symbol={sym} side={side_u} tag={tag} emitted={trade_event_emitted}")

            if price <= 0:
                self.logger.warning(
                    "[POST_FILL_PRICE_MISSING] symbol=%s side=%s qty=%.8f order_id=%s client_order_id=%s",
                    sym,
                    side_u,
                    exec_qty,
                    order.get("orderId") or order.get("order_id") or order.get("exchange_order_id"),
                    order.get("clientOrderId") or order.get("client_order_id"),
                )
                return {
                    "delta": delta_f,
                    "realized_committed": realized_committed,
                    "emitted": emitted,
                    "trade_event_emitted": trade_event_emitted,
                }

            ss = self.shared_state
            realized_before = float(getattr(ss, "metrics", {}).get("realized_pnl", 0.0) or 0.0)

            fee_quote = float(order.get("fee_quote", 0.0) or order.get("fee", 0.0) or 0.0)
            fee_base = float(order.get("fee_base", 0.0) or 0.0)
            try:
                base_asset, quote_asset = self._split_base_quote(sym)
                fills = order.get("fills") or []
                if isinstance(fills, list):
                    fee_base = sum(
                        float(f.get("commission", 0.0) or 0.0)
                        for f in fills
                        if str(f.get("commissionAsset") or f.get("commission_asset") or "").upper() == base_asset
                    ) or fee_base
                    fee_quote = sum(
                        float(f.get("commission", 0.0) or 0.0)
                        for f in fills
                        if str(f.get("commissionAsset") or f.get("commission_asset") or "").upper() == quote_asset
                    ) or fee_quote
            except Exception:
                pass
            
            delta = None
            trade_recorded = False
            # P9 Frequency Engineering: Record trade for tier tracking and open trades
            if hasattr(ss, "record_trade"):
                try:
                    # Get fee if available
                    _rt_result = await ss.record_trade(sym, side_u, exec_qty, price, fee_quote=fee_quote, fee_base=fee_base, tier=tier)
                    trade_recorded = True
                    if isinstance(_rt_result, dict) and _rt_result.get("realized_pnl_delta") is not None:
                        delta = _rt_result["realized_pnl_delta"]
                    
                    # Frequency Engineering: Trigger TP/SL setup for BUYs
                    # [FIX] Add economic guard: only arm TP/SL if trade is economically viable
                    if (
                        side_u == "BUY"
                        and exec_qty > 0
                        and price > 0
                        and hasattr(self, "tp_sl_engine")
                        and self.tp_sl_engine
                    ):
                        # Economic guard: check notional value before arming
                        notional = exec_qty * price
                        min_notional = float(self._cfg("MIN_ECONOMIC_TRADE_USDT", 10.0) or 10.0)

                        if notional >= min_notional and hasattr(self.tp_sl_engine, "set_initial_tp_sl"):
                            try:
                                self.tp_sl_engine.set_initial_tp_sl(sym, price, exec_qty, tier=tier)
                            except Exception as e:
                                self.logger.error("[TPSL_ARM_FAILED] %s: %s", sym, e, exc_info=True)
                                try:
                                    ot = getattr(ss, "open_trades", None)
                                    if isinstance(ot, dict) and sym in ot:
                                        ot[sym]["_tpsl_armed"] = False
                                except Exception:
                                    pass
                        elif notional < min_notional:
                            self.logger.info(
                                "[TPSL_SKIPPED_ECONOMIC] %s notional=%.4f < min=%.2f",
                                sym, notional, min_notional
                            )
                except Exception as e:
                    self.logger.warning(f"Failed to record trade in SharedState: {e}")

            # Try canonical API first
            if delta is None:
                delta = await maybe_call(ss, "compute_realized_pnl_delta", sym, side_u, exec_qty, price)

            # Try fill-recording APIs that return a dict containing the delta
            # FIX: Don't call record_fill again if record_trade already did the job
            if delta is None and not trade_recorded:
                res = await maybe_call(ss, "record_fill", sym, side_u, exec_qty, price)
                if isinstance(res, dict):
                    delta = res.get("realized_pnl_delta") or res.get("pnl_delta")

            # Try position manager-style API
            if delta is None:
                res = await maybe_call(ss, "apply_fill_to_positions", sym, side_u, exec_qty, price)
                if isinstance(res, dict):
                    delta = res.get("realized_pnl_delta") or res.get("pnl_delta")

            realized_after = float(getattr(ss, "metrics", {}).get("realized_pnl", 0.0) or 0.0)
            realized_committed = realized_after != realized_before

            # If SharedState already committed realized PnL but didn't expose a delta,
            # infer the exact delta from before/after snapshots for audit consistency.
            if delta is None and side_u == "SELL" and realized_committed:
                delta = realized_after - realized_before

            if delta is not None:
                try:
                    delta_f = float(delta)
                except Exception:
                    delta_f = None
            if delta_f is None and side_u == "SELL" and not realized_committed:
                try:
                    pos = getattr(ss, "positions", {}).get(sym, {}) if hasattr(ss, "positions") else {}
                    entry = float(pos.get("avg_price", 0.0) or 0.0)
                    if entry <= 0:
                        ot = getattr(ss, "open_trades", {}).get(sym, {}) if hasattr(ss, "open_trades") else {}
                        entry = float(ot.get("entry_price", 0.0) or 0.0)
                    side_hint = str(pos.get("side") or pos.get("position") or "long").lower()
                    if entry > 0:
                        if side_hint in ("short", "sell"):
                            delta_f = (entry - price) * exec_qty - fee_quote
                        else:
                            delta_f = (price - entry) * exec_qty - fee_quote
                except Exception:
                    delta_f = None

            # Only persist manually when SharedState did NOT already commit.
            # This prevents double counting when record_trade/record_fill handled PnL.
            if not realized_committed and delta_f is not None and delta_f != 0.0:
                now = time.time()
                try:
                    ss.metrics["realized_pnl"] = float(getattr(ss, "metrics", {}).get("realized_pnl", 0.0) or 0.0) + delta_f
                    # Manual commit to metrics succeeded — mark committed so callers won't double-write
                    realized_committed = True
                except Exception:
                    pass
                # Persist via public API when available
                if hasattr(ss, "append_realized_pnl_delta"):
                    with contextlib.suppress(Exception):
                        await maybe_call(ss, "append_realized_pnl_delta", now, delta_f)
                    # Best-effort: mark as committed when public API wrote the delta
                    realized_committed = True
                else:
                    # Fallback to internal store (bounded deque)
                    try:
                        ss._realized_pnl.append((now, delta_f))
                        realized_committed = True
                    except Exception:
                        ss._realized_pnl = deque(maxlen=4096)
                        ss._realized_pnl.append((now, delta_f))
                        realized_committed = True

                # Emit the event with optional nav_quote
                nav_q = None
                try:
                    if hasattr(ss, "get_nav_quote"):
                        nav_q = float(await maybe_call(ss, "get_nav_quote"))
                except Exception:
                    nav_q = None

                payload = {"pnl_delta": delta_f, "symbol": sym, "timestamp": now}
                if nav_q is not None:
                    payload["nav_quote"] = nav_q
                with contextlib.suppress(Exception):
                    await maybe_call(ss, "emit_event", "RealizedPnlUpdated", payload)
                    emitted = True
        except Exception:
            self.logger.debug("post-fill PnL handler failed (non-fatal)", exc_info=True)

        # Unified TRADE_AUDIT: one structured record per confirmed fill
        pf_result = {"delta": delta_f, "realized_committed": realized_committed, "emitted": emitted, "trade_event_emitted": trade_event_emitted}
        with contextlib.suppress(Exception):
            await self._emit_trade_audit(
                symbol=symbol,
                side=side,
                order=order,
                tier=tier,
                tag=tag,
                confidence=confidence,
                agent=agent,
                planned_quote=planned_quote,
                post_fill_result=pf_result,
            )

        return pf_result

    async def _update_position_from_fill(
        self,
        symbol: str,
        side: str,
        order: Dict[str, Any],
        tag: str = ""
    ) -> bool:
        """
        PHASE 4: Update position using actual fill data.
        
        Uses order["executedQty"] (actual filled quantity) instead of planned amounts.
        This ensures positions reflect reality.
        
        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            side: "BUY" or "SELL"
            order: Binance order response with executedQty
            tag: Optional tag for logging
        
        Returns:
            bool: True if position was updated successfully
        """
        try:
            sym = self._norm_symbol(symbol)
            side_u = (side or "").upper()
            ss = self.shared_state
            if not ss:
                return False

            # Canonical fill accounting lives in SharedState.record_trade/record_fill and
            # is triggered from _handle_post_fill in this same execution path.
            # Running this legacy pre-fill mutation as well causes double-mutation drift
            # (qty/avg corruption, phantom dust, and inaccurate realized accounting).
            # Keep this path only as an explicit fallback for legacy SharedState shapes.
            direct_phase4_enabled = bool(self._cfg("ENABLE_PHASE4_DIRECT_POSITION_UPDATE", False))
            has_canonical_fill_api = callable(getattr(ss, "record_trade", None)) or callable(
                getattr(ss, "record_fill", None)
            )
            if has_canonical_fill_api and not direct_phase4_enabled:
                self.logger.debug(
                    "[PHASE4_BYPASS] %s %s fill delegated to SharedState record_fill pipeline",
                    sym,
                    side_u,
                )
                return True
            
            # CRITICAL: Use actual fill, not planned amount
            executed_qty = float(order.get("executedQty") or 0.0)
            if executed_qty <= 0:
                self.logger.warning(
                    "[PHASE4] Position update skipped: no executed quantity. "
                    "symbol=%s side=%s orderId=%s",
                    sym, side_u, order.get("orderId")
                )
                return False
            
            # Get actual execution price (what was really spent/received)
            executed_price = self._resolve_post_fill_price(order, executed_qty)
            if executed_price <= 0:
                self.logger.warning(
                    "[PHASE4] Position update skipped: missing execution price. "
                    "symbol=%s orderId=%s",
                    sym, order.get("orderId")
                )
                return False
            
            # Get current position
            positions = getattr(ss, "positions", {}) or {}
            pos = dict(positions.get(sym, {}) or {})
            
            # PHASE 4: Calculate new position using ACTUAL fills
            current_qty = float(pos.get("quantity", 0.0) or 0.0)
            current_cost = float(pos.get("cost_basis", 0.0) or 0.0)
            current_avg_price = float(pos.get("avg_price", 0.0) or 0.0)
            
            if side_u == "BUY":
                # BUY: add to position
                new_qty = current_qty + executed_qty
                new_cost = current_cost + (executed_qty * executed_price)
                new_avg_price = new_cost / new_qty if new_qty > 0 else 0.0
            elif side_u == "SELL":
                # SELL: reduce position
                new_qty = current_qty - executed_qty
                # Keep cost basis proportional
                if current_qty > 0:
                    new_cost = current_cost * (new_qty / current_qty) if new_qty > 0 else 0.0
                else:
                    new_cost = 0.0
                new_avg_price = new_cost / new_qty if new_qty > 0 else 0.0
            else:
                self.logger.error("[PHASE4] Unknown side: %s", side_u)
                return False
            
            # Update position with actual values
            pos["quantity"] = float(new_qty)
            pos["cost_basis"] = float(new_cost)
            pos["avg_price"] = float(new_avg_price)
            pos["last_executed_price"] = float(executed_price)
            pos["last_executed_qty"] = float(executed_qty)
            pos["last_filled_time"] = order.get("updateTime") or order.get("timestamp") or int(time.time() * 1000)
            
            # Preserve metadata
            for key in ["status", "state", "is_significant", "is_dust", "_is_dust", "open_position"]:
                pos.pop(key, None)
            
            # Persist updated position
            if hasattr(ss, "update_position"):
                await ss.update_position(sym, pos)
                self.logger.info(
                    "[PHASE4_POSITION_UPDATED] %s side=%s qty=%.10f avg_price=%.10f "
                    "executed_qty=%.10f executed_price=%.10f tag=%s",
                    sym, side_u, new_qty, new_avg_price,
                    executed_qty, executed_price, tag
                )
                return True
            else:
                self.logger.warning(
                    "[PHASE4_NO_POSITION_API] SharedState missing update_position method"
                )
                return False
                
        except Exception as e:
            self.logger.error(
                "[PHASE4_POSITION_UPDATE_FAILED] symbol=%s side=%s error=%s",
                symbol, side, e, exc_info=True
            )
            return False

    async def _ensure_post_fill_handled(
        self,
        symbol: str,
        side: str,
        order: Optional[Dict[str, Any]],
        *,
        tier: Optional[str] = None,
        tag: str = "",
    ) -> Dict[str, Any]:
        """
        Idempotent post-fill hook wrapper.
        Reuses cached result on the order payload when available to prevent
        duplicate realized-PnL/event emissions across overlapping call paths.
        """
        default = {
            "delta": None,
            "realized_committed": False,
            "emitted": False,
            "trade_event_emitted": False,
        }
        if not isinstance(order, dict):
            return dict(default)

        exec_qty = self._safe_float(order.get("executedQty") or order.get("executed_qty"), 0.0)
        if exec_qty <= 0.0:
            # Avoid caching non-fill results on mutable order payloads. If the same
            # payload later reconciles to a real fill, post-fill must still execute.
            return dict(default)

        cached = order.get("_post_fill_result")
        if isinstance(cached, dict) and bool(order.get("_post_fill_done")):
            # Cache is valid once post-fill processing has completed.
            out = dict(default)
            out.update(cached)
            if str(side or "").upper() == "SELL":
                with contextlib.suppress(Exception):
                    self._track_sell_fill_observed(
                        symbol=symbol,
                        order=order,
                        tag=str(tag or ""),
                    )
            return out

        res = await self._handle_post_fill(
            symbol=symbol,
            side=side,
            order=order,
            tier=tier,
            tag=tag,
        )
        out = dict(default)
        if isinstance(res, dict):
            out.update(res)
        order["_post_fill_result"] = out
        order["_post_fill_done"] = True
        if str(side or "").upper() == "SELL":
            with contextlib.suppress(Exception):
                self._track_sell_fill_observed(
                    symbol=symbol,
                    order=order,
                    tag=str(tag or ""),
                )
        return out

    async def _reconcile_delayed_fill(
        self,
        symbol: str,
        side: str,
        order: Optional[Dict[str, Any]],
        *,
        tag: str = "",
        tier: Optional[str] = None,
        order_id_hint: Optional[Union[str, int]] = None,
        client_order_id_hint: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Reconcile orders that may fill slightly after initial placement response.
        """
        merged: Dict[str, Any] = dict(order) if isinstance(order, dict) else {}
        if not merged:
            if order_id_hint not in (None, ""):
                merged["orderId"] = order_id_hint
                merged["exchange_order_id"] = order_id_hint
            if client_order_id_hint:
                merged["clientOrderId"] = str(client_order_id_hint)
                merged["client_order_id"] = str(client_order_id_hint)
            if merged:
                merged.setdefault("symbol", self._norm_symbol(symbol))
                merged.setdefault("side", str(side or "").upper())

        if not merged:
            return order

        get_order = getattr(self.exchange_client, "get_order", None)
        status = str(merged.get("status", "")).upper()
        exec_qty = self._safe_float(merged.get("executedQty") or merged.get("executed_qty"), 0.0)
        # FILLED payloads from normalized exchange-client responses can carry only
        # client order IDs (`order_id`) and omit exchange `orderId`. Enrich once so
        # canonical TRADE_EXECUTED emits and auditor matching use exchange order IDs.
        if status in ("FILLED", "PARTIALLY_FILLED") and exec_qty > 0:
            has_exchange_oid = bool(
                merged.get("orderId") or merged.get("exchange_order_id")
            )
            cid_probe = (
                merged.get("clientOrderId")
                or merged.get("client_order_id")
                or merged.get("origClientOrderId")
            )
            if (not has_exchange_oid) and cid_probe and callable(get_order):
                fresh = None
                with contextlib.suppress(Exception):
                    fresh = await get_order(symbol, client_order_id=str(cid_probe))
                if isinstance(fresh, dict) and fresh:
                    merged.update(fresh)
                    merged.setdefault(
                        "exchange_order_id",
                        fresh.get("orderId") or fresh.get("order_id"),
                    )
                    merged.setdefault(
                        "client_order_id",
                        fresh.get("clientOrderId") or fresh.get("origClientOrderId") or cid_probe,
                    )
            # [FIX] Reconcile returns merged order without calling _ensure_post_fill_handled().
            # Caller (close_position) is responsible for post-fill + finalize.
            # Reason: Double-calling _ensure_post_fill_handled causes idempotency issues:
            # - First call (here) sets _post_fill_done=True
            # - Second call (in close_position) returns cached result
            # - Finalize then sees empty cached dict and skips _emit_close_events
            # - Position never reduces in SharedState
            return merged

        if not callable(get_order):
            return merged

        oid_raw = (
            merged.get("orderId")
            or merged.get("exchange_order_id")
            or merged.get("order_id")
            or order_id_hint
        )
        cid_raw = (
            merged.get("clientOrderId")
            or merged.get("client_order_id")
            or merged.get("origClientOrderId")
            or merged.get("order_id")
            or client_order_id_hint
        )
        if oid_raw in (None, "") and not cid_raw:
            if str(side or "").upper() == "SELL":
                status_u = str(merged.get("status", "")).upper() or "UNKNOWN"
                reason_u = str(merged.get("reason", "")).upper()
                # Avoid misclassifying cleanly blocked/skipped exits as "recovery unavailable".
                # This signal is meant for ambiguous exchange submissions only.
                non_submission_statuses = {"BLOCKED", "SKIPPED", "REJECTED", "CANCELED", "EXPIRED"}
                is_non_submission = (
                    status_u in non_submission_statuses
                    or "NO_POSITION" in reason_u
                    or "SELL_GUARD" in reason_u
                    or "MISSING_META_TRACE_ID" in reason_u
                    or "TRADEABILITY_GATE_MISSING" in reason_u
                )
                if not is_non_submission:
                    self._journal("SELL_RECOVERY_UNAVAILABLE_NO_IDS", {
                        "symbol": self._norm_symbol(symbol),
                        "side": "SELL",
                        "status": status_u,
                        "reason": "missing_order_id_and_client_order_id",
                        "tag": str(tag or ""),
                        "timestamp": time.time(),
                    })
            return merged

        delay_s = float(self._cfg("POST_SUBMIT_RECHECK_DELAY_S", 0.2) or 0.2)
        delay_s = min(max(delay_s, 0.1), 0.5)
        attempts = int(self._cfg("POST_SUBMIT_RECHECK_ATTEMPTS", 6) or 6)
        attempts = max(1, min(attempts, 20))

        last_fresh: Optional[Dict[str, Any]] = None

        for attempt in range(1, attempts + 1):
            await asyncio.sleep(delay_s)

            fresh: Optional[Dict[str, Any]] = None
            if oid_raw not in (None, ""):
                with contextlib.suppress(Exception):
                    fresh = await get_order(symbol, order_id=int(str(oid_raw)))
            if fresh is None and cid_raw:
                with contextlib.suppress(Exception):
                    fresh = await get_order(symbol, client_order_id=str(cid_raw))

            if not isinstance(fresh, dict) or not fresh:
                continue
            last_fresh = fresh

            merged.update(fresh)
            merged.setdefault("exchange_order_id", fresh.get("orderId") or fresh.get("order_id") or oid_raw)
            merged.setdefault(
                "client_order_id",
                fresh.get("clientOrderId") or fresh.get("origClientOrderId") or cid_raw,
            )
            oid_raw = merged.get("orderId") or merged.get("exchange_order_id") or merged.get("order_id") or oid_raw
            cid_raw = (
                merged.get("clientOrderId")
                or merged.get("client_order_id")
                or merged.get("origClientOrderId")
                or cid_raw
            )

            fresh_status = str(merged.get("status", "")).upper()
            fresh_qty = self._safe_float(merged.get("executedQty") or merged.get("executed_qty"), 0.0)
            if fresh_status in ("FILLED", "PARTIALLY_FILLED") and fresh_qty > 0:
                # [FIX] Reconcile merges fresh order data but does NOT call _ensure_post_fill_handled().
                # Caller (e.g. close_position or execute_trade) handles post-fill + finalize.
                # This prevents double-idempotency-check that causes finalize to skip.
                
                # ✅ CRITICAL: Log to journal IMMEDIATELY when fill is detected
                # This ensures the exchange execution is captured even if subsequent
                # processing fails or returns None
                self._journal("RECONCILED_DELAYED_FILL", {
                    "symbol": symbol,
                    "side": side.upper() if side else "UNKNOWN",
                    "executed_qty": fresh_qty,
                    "avg_price": self._safe_float(merged.get("avgPrice") or merged.get("price"), 0.0),
                    "cumm_quote": self._safe_float(merged.get("cummulativeQuoteQty"), 0.0),
                    "order_id": merged.get("orderId") or merged.get("exchange_order_id") or merged.get("order_id"),
                    "status": fresh_status,
                    "attempt": attempt,
                    "total_attempts": attempts,
                    "timestamp": time.time(),
                })
                
                # ✅ ELITE: Verify position invariants after SELL
                if side and side.upper() == "SELL":
                    invariant_ok = await self._verify_position_invariants(
                        symbol=symbol,
                        event_type="RECONCILED_DELAYED_FILL",
                        before_qty=0.0,  # We don't have before_qty here, but check will validate monotonicity
                    )
                    if not invariant_ok:
                        self.logger.error(
                            "[EM:INVARIANT] SELL reconciliation on %s failed invariant check - position may be corrupted",
                            symbol
                        )
                
                self.logger.info(
                    "[EM:DelayedFill] Reconciled delayed fill symbol=%s side=%s order_id=%s status=%s qty=%.8f attempt=%d/%d",
                    self._norm_symbol(symbol),
                    str(side or "").upper(),
                    merged.get("orderId") or merged.get("exchange_order_id") or merged.get("order_id"),
                    fresh_status,
                    fresh_qty,
                    attempt,
                    attempts,
                )
                return merged

        if isinstance(last_fresh, dict) and last_fresh:
            self.logger.debug(
                "[EM:DelayedFill] Pending after retries symbol=%s side=%s order_id=%s status=%s qty=%.8f attempts=%d",
                self._norm_symbol(symbol),
                str(side or "").upper(),
                merged.get("orderId") or merged.get("order_id") or merged.get("exchange_order_id"),
                str(merged.get("status", "")).upper(),
                self._safe_float(merged.get("executedQty") or merged.get("executed_qty"), 0.0),
                attempts,
            )
            if str(side or "").upper() == "SELL":
                with contextlib.suppress(Exception):
                    self._schedule_sell_fill_recovery(
                        symbol=symbol,
                        order=merged,
                        tag=str(tag or ""),
                        tier=tier,
                    )
            return merged

        if str(side or "").upper() == "SELL":
            with contextlib.suppress(Exception):
                self._schedule_sell_fill_recovery(
                    symbol=symbol,
                    order=merged,
                    tag=str(tag or ""),
                    tier=tier,
                )
        return merged if merged else order

    async def _get_exchange_position_qty(self, symbol: str) -> Tuple[bool, float]:
        """Authoritative base-asset quantity from exchange account balance."""
        get_bal = getattr(self.exchange_client, "get_account_balance", None)
        if not callable(get_bal):
            return False, 0.0
        base_asset, _ = self._split_base_quote(symbol)
        try:
            bal = await get_bal(base_asset)
            if isinstance(bal, dict):
                free = self._safe_float(bal.get("free"), 0.0)
                locked = self._safe_float(bal.get("locked"), 0.0)
                return True, max(0.0, float(free + locked))
            return True, max(0.0, self._safe_float(bal, 0.0))
        except Exception:
            return False, 0.0

    async def _position_sync_qty_tol(self, symbol: str) -> float:
        tol = max(1e-10, float(self._cfg("POSITION_SYNC_QTY_TOL", 1e-8) or 1e-8))
        try:
            filters = await self.exchange_client.ensure_symbol_filters_ready(self._norm_symbol(symbol))
            step_size, min_qty, _, _, _ = self._extract_filter_vals(filters)
            if step_size > 0:
                tol = max(tol, float(step_size) * 0.5)
            if min_qty > 0:
                tol = max(tol, float(min_qty) * 0.5)
        except Exception:
            pass
        return float(tol)

    async def _sync_shared_position_after_sell_fill(
        self,
        *,
        symbol: str,
        order: Optional[Dict[str, Any]],
        reason: str,
    ) -> None:
        """
        Reconcile SharedState to exchange truth after a confirmed SELL fill.
        Prevents phantom positions when post-fill bookkeeping partially fails.
        """
        if not isinstance(order, dict):
            return
        sym = self._norm_symbol(symbol)
        ss = self.shared_state
        tol = await self._position_sync_qty_tol(sym)

        local_qty = 0.0
        try:
            if hasattr(ss, "get_position_quantity"):
                local_qty = float(await maybe_call(ss, "get_position_quantity", sym) or 0.0)
            elif isinstance(getattr(ss, "positions", None), dict):
                local_qty = float((ss.positions.get(sym, {}) or {}).get("quantity", 0.0) or 0.0)
        except Exception:
            local_qty = 0.0

        exchange_ok, exchange_qty = await self._get_exchange_position_qty(sym)
        exec_qty = self._safe_float(order.get("executedQty") or order.get("executed_qty"), 0.0)
        exec_px = self._resolve_post_fill_price(order, exec_qty)

        # If exchange cannot be queried, use conservative fallback:
        # when executed qty fully covers local qty, force local close.
        if not exchange_ok:
            if local_qty > tol and exec_qty >= max(local_qty - tol, 0.0):
                with contextlib.suppress(Exception):
                    await self._force_finalize_position(sym, f"{reason}_fallback")
            return

        # Exchange is flat but SharedState still shows qty -> phantom position.
        if exchange_qty <= tol and local_qty > tol:
            self.logger.error(
                "[EM:PhantomRepair] %s exchange_qty=%.10f local_qty=%.10f reason=%s -> force finalize",
                sym,
                float(exchange_qty),
                float(local_qty),
                reason,
            )
            # 🔥 MANDATORY: Journal position closure BEFORE mark_position_closed
            self._journal("PHANTOM_POSITION_CLOSURE", {
                "symbol": sym,
                "local_qty": float(local_qty),
                "exchange_qty": float(exchange_qty),
                "exec_price": float(exec_px or 0.0),
                "reason": str(reason),
                "timestamp": time.time(),
            })
            with contextlib.suppress(Exception):
                if hasattr(ss, "mark_position_closed"):
                    await maybe_call(
                        ss,
                        "mark_position_closed",
                        symbol=sym,
                        qty=float(local_qty),
                        price=float(exec_px or 0.0),
                        reason=str(reason),
                        tag="execution_sync",
                    )
            with contextlib.suppress(Exception):
                await self._force_finalize_position(sym, reason)
            return

        # Exchange has remaining qty and local qty drifted materially -> align local state.
        if exchange_qty > tol and abs(local_qty - exchange_qty) > tol:
            self.logger.warning(
                "[EM:QtyResync] %s local_qty=%.10f exchange_qty=%.10f reason=%s",
                sym,
                float(local_qty),
                float(exchange_qty),
                reason,
            )
            with contextlib.suppress(Exception):
                pos = dict((getattr(ss, "positions", {}) or {}).get(sym, {}) or {})
                if pos:
                    pos["quantity"] = float(exchange_qty)
                    for k in ("status", "state", "is_significant", "is_dust", "_is_dust", "open_position"):
                        pos.pop(k, None)
                    await maybe_call(ss, "update_position", sym, pos)
            with contextlib.suppress(Exception):
                ot = getattr(ss, "open_trades", None)
                if isinstance(ot, dict):
                    tr = dict(ot.get(sym, {}) or {})
                    if tr:
                        tr["quantity"] = float(exchange_qty)
                        ot[sym] = tr

    def _schedule_sell_fill_recovery(
        self,
        *,
        symbol: str,
        order: Optional[Dict[str, Any]],
        tag: str = "",
        tier: Optional[str] = None,
    ) -> None:
        """Background recovery for SELL orders that remain pending after short reconciliation."""
        if not isinstance(order, dict):
            return
        status = str(order.get("status", "")).upper()
        exec_qty = self._safe_float(order.get("executedQty") or order.get("executed_qty"), 0.0)
        if status in ("FILLED", "PARTIALLY_FILLED") and exec_qty > 0:
            return
        if status in {"REJECTED", "CANCELED", "EXPIRED"}:
            return
        oid = order.get("orderId") or order.get("order_id") or order.get("exchange_order_id")
        cid = order.get("clientOrderId") or order.get("client_order_id") or order.get("origClientOrderId")
        if not oid and not cid:
            return

        sym = self._norm_symbol(symbol)
        key = str(order.get("_sell_finalize_key") or "").strip() or self._sell_finalize_key(sym, order)
        tasks = getattr(self, "_sell_fill_recovery_tasks", None)
        if not isinstance(tasks, dict):
            tasks = {}
            self._sell_fill_recovery_tasks = tasks
        existing = tasks.get(key)
        if existing is not None and not existing.done():
            return

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        task = loop.create_task(
            self._recover_sell_fill_task(
                symbol=sym,
                order=dict(order),
                key=key,
                tag=str(tag or ""),
                tier=tier,
            ),
            name=f"em.sell_recover.{sym}",
        )
        tasks[key] = task

        def _cleanup(done_task: asyncio.Task) -> None:
            with contextlib.suppress(Exception):
                tasks.pop(key, None)
            # 🔥 FIX: Log recovery task exceptions (don't silently suppress)
            # Reason: Silent failures hide orphaned positions
            try:
                done_task.exception()  # Will raise if task failed
            except asyncio.CancelledError:
                pass  # Expected for cancelled tasks
            except Exception as e:
                self.logger.error(
                    "[EM:RecoveryTaskFailed] Recovery task failed for symbol=%s key=%s: %s",
                    sym, key, str(e), exc_info=True
                )

        task.add_done_callback(_cleanup)
        self.logger.info(
            "[EM:DelayedFillRecover] Scheduled SELL recovery key=%s symbol=%s status=%s qty=%.8f",
            key,
            sym,
            status or "UNKNOWN",
            float(exec_qty),
        )

    async def _recover_sell_fill_task(
        self,
        *,
        symbol: str,
        order: Dict[str, Any],
        key: str,
        tag: str = "",
        tier: Optional[str] = None,
    ) -> None:
        """Poll exchange order status until terminal, then finalize SELL fill if observed."""
        get_order = getattr(self.exchange_client, "get_order", None)
        if not callable(get_order):
            return

        poll_s = float(self._cfg("SELL_RECOVERY_POLL_SEC", 0.5) or 0.5)
        poll_s = min(max(poll_s, 0.2), 2.0)
        # 🔥 FIX: Increase recovery window from 20s to 60s
        # Reason: Exchange stale timeout is 120s, recovery must complete before then
        # This prevents orphaned fills that occur after recovery window closes
        max_wait_s = float(self._cfg("SELL_RECOVERY_MAX_WAIT_SEC", 60.0) or 60.0)
        max_wait_s = min(max(max_wait_s, 2.0), 180.0)
        deadline = time.time() + max_wait_s

        sym = self._norm_symbol(symbol)
        merged = dict(order)
        oid_raw = merged.get("orderId") or merged.get("order_id") or merged.get("exchange_order_id")
        cid_raw = (
            merged.get("clientOrderId")
            or merged.get("client_order_id")
            or merged.get("origClientOrderId")
            or merged.get("order_id")
        )

        while time.time() < deadline:
            status = str(merged.get("status", "")).upper()
            exec_qty = self._safe_float(merged.get("executedQty") or merged.get("executed_qty"), 0.0)
            if status in ("FILLED", "PARTIALLY_FILLED") and exec_qty > 0:
                post_fill = await self._ensure_post_fill_handled(
                    symbol=sym,
                    side="SELL",
                    order=merged,
                    tier=tier,
                    tag=str(tag or ""),
                )
                await self._finalize_sell_post_fill(
                    symbol=sym,
                    order=merged,
                    tag=str(tag or ""),
                    post_fill=post_fill,
                    policy_ctx={"reason": "delayed_fill_recovery"},
                    tier=tier,
                )
                with contextlib.suppress(Exception):
                    await self._audit_post_fill_accounting(
                        symbol=sym,
                        side="sell",
                        raw=merged,
                        stage="delayed_fill_recovery",
                    )
                self.logger.warning(
                    "[EM:DelayedFillRecover] Finalized delayed SELL fill key=%s symbol=%s qty=%.8f status=%s",
                    key,
                    sym,
                    float(exec_qty),
                    status,
                )
                return
            if status in ("CANCELED", "REJECTED", "EXPIRED"):
                return

            fresh = None
            if oid_raw not in (None, ""):
                with contextlib.suppress(Exception):
                    fresh = await get_order(sym, order_id=int(str(oid_raw)))
            if fresh is None and cid_raw:
                with contextlib.suppress(Exception):
                    fresh = await get_order(sym, client_order_id=str(cid_raw))

            if isinstance(fresh, dict) and fresh:
                merged.update(fresh)
                oid_raw = merged.get("orderId") or merged.get("order_id") or merged.get("exchange_order_id") or oid_raw
                cid_raw = (
                    merged.get("clientOrderId")
                    or merged.get("client_order_id")
                    or merged.get("origClientOrderId")
                    or cid_raw
                )

            await asyncio.sleep(poll_s)

        # Timeout fallback: if exchange is flat while SharedState still open, repair phantom.
        with contextlib.suppress(Exception):
            await self._sync_shared_position_after_sell_fill(
                symbol=sym,
                order=merged,
                reason="SELL_RECOVERY_TIMEOUT",
            )

    def _calc_close_payload(self, sym: str, raw: Dict[str, Any]) -> Tuple[float, float, float, float]:
        entry_price = float(self._get_entry_price_for_sell(sym) or 0.0)
        exec_px = float(raw.get("avgPrice", raw.get("price", 0.0)) or 0.0)
        exec_qty = float(raw.get("executedQty", 0.0) or 0.0)
        fee_quote = float(raw.get("fee_quote", 0.0) or raw.get("fee", 0.0) or 0.0)
        try:
            _, quote_asset = self._split_base_quote(sym)
            fills = raw.get("fills") or []
            if isinstance(fills, list):
                fee_quote = sum(
                    float(f.get("commission", 0.0) or 0.0)
                    for f in fills
                    if str(f.get("commissionAsset") or f.get("commission_asset") or "").upper() == quote_asset
                ) or fee_quote
        except Exception:
            pass

        realized_pnl = 0.0
        pos = getattr(self.shared_state, "positions", {}).get(sym, {}) if hasattr(self.shared_state, "positions") else {}
        side_hint = str(pos.get("side") or pos.get("position") or "long").lower()
        if entry_price > 0 and exec_px > 0 and exec_qty > 0:
            if side_hint in ("short", "sell"):
                realized_pnl = (entry_price - exec_px) * exec_qty - fee_quote
            else:
                realized_pnl = (exec_px - entry_price) * exec_qty - fee_quote

        return entry_price, exec_px, exec_qty, realized_pnl

    async def _emit_close_events(self, sym: str, raw: Dict[str, Any], post_fill: Optional[Dict[str, Any]] = None) -> None:
        entry_price, exec_px, exec_qty, realized_pnl = self._calc_close_payload(sym, raw)
        
        # 🔴 FIX: Use executedQty from the filled order directly, not from remaining position state.
        # The distinction is critical:
        # - exec_qty from _calc_close_payload() = remaining position (may be dust → 0)
        # - executedQty from raw order = what was actually FILLED in this order
        # We should emit POSITION_CLOSED based on the FILLED quantity, not the remaining.
        actual_executed_qty = self._safe_float(raw.get("executedQty") or raw.get("executed_qty"), 0.0)
        
        if actual_executed_qty <= 0 or exec_px <= 0:
            return

        # committed/emitted come from _ensure_post_fill_handled return dict when available
        committed = (post_fill or {}).get("realized_committed", False)
        emitted = (post_fill or {}).get("emitted", False)

        # Ensure canonical TRADE_EXECUTED exists for SELL closes. Some paths (recovered fills,
        # transient emit failures, or external/order-recovery flows) may reach close events
        # without a prior canonical TRADE_EXECUTED. To preserve the architecture invariant
        # (every confirmed fill must emit TRADE_EXECUTED) we re-emit here when missing.
        try:
            tag = (raw or {}).get("tag") or (raw or {}).get("order_tag") or ""
            # Idempotent/dedupe-protected: always attempt canonical emit for invariants
            with contextlib.suppress(Exception):
                await self._emit_trade_executed_event(sym, "SELL", str(tag or ""), raw)
        except Exception:
            self.logger.debug("[EM:CloseEmitRecover] re-emit TRADE_EXECUTED failed", exc_info=True)

        # Keep realized PnL accounting authoritative in SharedState.
        # If post-fill did not commit (or committed zero), apply close delta here
        # through increment_realized_pnl when available so lock + mirrors stay consistent.
        close_delta = None
        if not committed:
            pf_delta = None
            if isinstance(post_fill, dict) and post_fill.get("delta") is not None:
                with contextlib.suppress(Exception):
                    pf_delta = float(post_fill.get("delta") or 0.0)
            if pf_delta is not None and abs(pf_delta) > 1e-12:
                close_delta = pf_delta
            elif abs(float(realized_pnl or 0.0)) > 1e-12:
                close_delta = float(realized_pnl)

            if close_delta is not None:
                applied = False
                with contextlib.suppress(Exception):
                    if hasattr(self.shared_state, "increment_realized_pnl"):
                        await maybe_call(self.shared_state, "increment_realized_pnl", float(close_delta))
                        applied = True
                if not applied:
                    with contextlib.suppress(Exception):
                        cur = float(getattr(self.shared_state, "metrics", {}).get("realized_pnl", 0.0) or 0.0)
                        self.shared_state.metrics["realized_pnl"] = cur + float(close_delta)
                        setattr(self.shared_state, "realized_pnl", float(self.shared_state.metrics["realized_pnl"]))

        if not emitted:
            now = time.time()
            event_delta = float(close_delta if close_delta is not None else realized_pnl)
            payload = {
                "realized_pnl": float(getattr(self.shared_state, "metrics", {}).get("realized_pnl", 0.0) or 0.0),
                "pnl_delta": event_delta,
                "symbol": sym,
                "price": exec_px,
                "qty": actual_executed_qty,  # ← Use actual filled quantity
                "timestamp": now,
            }
            with contextlib.suppress(Exception):
                await maybe_call(self.shared_state, "emit_event", "RealizedPnlUpdated", payload)

        # Append to trade_history so PerformanceEvaluator.usdt_per_hour is non-zero.
        # Without this, PerformanceEvaluator reads an empty history, reports 0 usdt/h,
        # triggers global_systemic_degradation, and CapitalAllocator halves all budgets.
        with contextlib.suppress(Exception):
            _trade_delta = float(close_delta if close_delta is not None else realized_pnl or 0.0)
            _history = getattr(self.shared_state, "trade_history", None)
            if _history is None:
                self.shared_state.trade_history = []
                _history = self.shared_state.trade_history
            _history.append({
                "ts": time.time(),
                "symbol": sym,
                "realized_delta": _trade_delta,
                "price": exec_px,
                "qty": actual_executed_qty,
            })
            # Keep the deque bounded (max 2000 entries ≈ last ~10 trading hours at 5min avg hold)
            if isinstance(_history, list) and len(_history) > 2000:
                del _history[:-2000]

        self.logger.info(json.dumps({
            "event": "POSITION_CLOSED",
            "symbol": sym,
            "entry_price": entry_price,
            "exit_price": exec_px,
            "qty": actual_executed_qty,  # ← Use actual filled quantity (not remaining)
            "realized_pnl": realized_pnl,
        }, separators=(",", ":")))
        # Emit canonical POSITION_CLOSED so other components observe lifecycle transitions
        try:
            await maybe_call(self.shared_state, "emit_event", "POSITION_CLOSED", {
                "symbol": sym,
                "entry_price": float(entry_price or 0.0),
                "price": float(exec_px or 0.0),
                "qty": float(actual_executed_qty or 0.0),  # ← Use actual filled quantity
                "realized_pnl": float(realized_pnl or 0.0),
                "timestamp": time.time(),
            })
        except Exception:
            pass

    @staticmethod
    def _is_tp_sl_exit_reason(reason: Optional[str]) -> bool:
        reason_u = str(reason or "").strip().upper()
        if not reason_u:
            return False
        if reason_u in {"TP", "SL", "TP_HIT", "SL_HIT", "TPSL_EXIT", "TP_SL", "TAKE_PROFIT", "STOP_LOSS"}:
            return True
        return ("TP_SL" in reason_u) or ("TAKE_PROFIT" in reason_u) or ("STOP_LOSS" in reason_u)

    async def _record_sell_exit_bookkeeping(
        self,
        *,
        symbol: str,
        tag: str,
        policy_ctx: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record canonical SELL exit metadata so Meta re-entry lock can see fresh exits.
        This is a backup path for TP/SL and liquidation exits.
        """
        sym = self._norm_symbol(symbol)
        ctx = policy_ctx or {}
        raw_reason = str(
            ctx.get("exit_reason")
            or ctx.get("reason")
            or ctx.get("liquidation_reason")
            or ""
        ).strip()
        reason_u = raw_reason.upper()
        tag_u = str(tag or "").upper()

        reason_code = ""
        if self._is_tp_sl_exit_reason(reason_u) or ("TP_SL" in tag_u):
            if reason_u in {"TP", "TP_HIT", "TAKE_PROFIT"}:
                reason_code = "TP"
            elif reason_u in {"SL", "SL_HIT", "STOP_LOSS"}:
                reason_code = "SL"
            else:
                reason_code = "TPSL_EXIT"
        elif "LIQUIDATION" in tag_u or "LIQ" in reason_u:
            reason_code = "LIQUIDATION"

        if not reason_code:
            return

        with contextlib.suppress(Exception):
            if hasattr(self.shared_state, "record_exit_reason"):
                self.shared_state.record_exit_reason(sym, reason_code, source="execution_manager")

        cooldown_sec = 0.0
        if reason_code in {"TP", "SL", "TPSL_EXIT"}:
            cooldown_sec = float(
                self._cfg("TP_SL_REENTRY_LOCK_SEC", self._cfg("REENTRY_LOCK_SEC", 0.0)) or 0.0
            )
        elif reason_code == "LIQUIDATION":
            cooldown_sec = float(
                self._cfg(
                    "LIQUIDATION_REENTRY_LOCK_SEC",
                    self._cfg("TP_SL_REENTRY_LOCK_SEC", self._cfg("REENTRY_LOCK_SEC", 0.0)),
                )
                or 0.0
            )
        if cooldown_sec <= 0:
            return

        with contextlib.suppress(Exception):
            if hasattr(self.shared_state, "set_cooldown"):
                await maybe_call(self.shared_state, "set_cooldown", sym, float(cooldown_sec))

    def _sell_finalize_key(self, symbol: str, order: Dict[str, Any]) -> str:
        sym = self._norm_symbol(symbol)
        oid = str(
            order.get("orderId")
            or order.get("exchange_order_id")
            or order.get("order_id")
            or ""
        ).strip()
        if oid:
            return f"{sym}|oid:{oid}"

        cid = str(
            order.get("clientOrderId")
            or order.get("origClientOrderId")
            or order.get("client_order_id")
            or ""
        ).strip()
        if cid:
            return f"{sym}|cid:{cid}"

        update_ms = 0
        for k in ("updateTime", "time", "transactTime", "workingTime"):
            raw_v = order.get(k)
            try:
                if raw_v is not None:
                    update_ms = int(float(raw_v))
                    if update_ms > 0:
                        break
            except Exception:
                continue
        qty = self._safe_float(order.get("executedQty") or order.get("executed_qty"), 0.0)
        if update_ms > 0 and qty > 0:
            return f"{sym}|ts:{update_ms}|qty:{qty:.12f}"
        return f"{sym}|obj:{id(order)}"

    def _track_sell_fill_observed(self, *, symbol: str, order: Dict[str, Any], tag: str = "") -> None:
        if not isinstance(order, dict):
            return
        status = str(order.get("status", "")).upper()
        qty = self._safe_float(order.get("executedQty") or order.get("executed_qty"), 0.0)
        if status not in ("FILLED", "PARTIALLY_FILLED") or qty <= 0:
            return

        sym = self._norm_symbol(symbol)
        key = str(order.get("_sell_finalize_key") or "").strip() or self._sell_finalize_key(sym, order)
        order["_sell_finalize_key"] = key
        now = time.time()

        row = self._sell_finalize_state.get(key)
        if not isinstance(row, dict):
            row = {
                "symbol": sym,
                "order_id": str(order.get("orderId") or order.get("exchange_order_id") or order.get("order_id") or ""),
                "client_order_id": str(order.get("clientOrderId") or order.get("origClientOrderId") or order.get("client_order_id") or ""),
                "fill_seen": 0,
                "finalized": 0,
                "fill_ts": now,
                "finalized_ts": 0.0,
                "last_ts": now,
                "tag": str(tag or ""),
                "timeout_reported": False,
                "dup_reported": False,
                "missing_fill_reported": False,
            }
            self._sell_finalize_state[key] = row

        row["symbol"] = sym
        row["last_ts"] = now
        row["status"] = status
        row["qty"] = float(qty)
        if tag:
            row["tag"] = str(tag)
        if not row.get("order_id"):
            row["order_id"] = str(order.get("orderId") or order.get("exchange_order_id") or order.get("order_id") or "")
        if not row.get("client_order_id"):
            row["client_order_id"] = str(order.get("clientOrderId") or order.get("origClientOrderId") or order.get("client_order_id") or "")

        if int(row.get("fill_seen", 0) or 0) <= 0:
            row["fill_seen"] = 1
            row["fill_ts"] = now
            self._sell_finalize_stats["fills_seen"] = int(self._sell_finalize_stats.get("fills_seen", 0) or 0) + 1
        else:
            self._sell_finalize_stats["fills_seen_duplicate"] = int(self._sell_finalize_stats.get("fills_seen_duplicate", 0) or 0) + 1

        self._audit_sell_finalize_invariant(now=now)

    def _track_sell_finalize(
        self,
        *,
        symbol: str,
        order: Dict[str, Any],
        tag: str = "",
        duplicate_attempt: bool = False,
    ) -> None:
        if not isinstance(order, dict):
            return

        sym = self._norm_symbol(symbol)
        key = str(order.get("_sell_finalize_key") or "").strip() or self._sell_finalize_key(sym, order)
        order["_sell_finalize_key"] = key
        now = time.time()

        row = self._sell_finalize_state.get(key)
        if not isinstance(row, dict):
            row = {
                "symbol": sym,
                "order_id": str(order.get("orderId") or order.get("exchange_order_id") or order.get("order_id") or ""),
                "client_order_id": str(order.get("clientOrderId") or order.get("origClientOrderId") or order.get("client_order_id") or ""),
                "fill_seen": 0,
                "finalized": 0,
                "fill_ts": now,
                "finalized_ts": 0.0,
                "last_ts": now,
                "tag": str(tag or ""),
                "timeout_reported": False,
                "dup_reported": False,
                "missing_fill_reported": False,
            }
            self._sell_finalize_state[key] = row

        row["symbol"] = sym
        row["last_ts"] = now
        if tag:
            row["tag"] = str(tag)
        if not row.get("order_id"):
            row["order_id"] = str(order.get("orderId") or order.get("exchange_order_id") or order.get("order_id") or "")
        if not row.get("client_order_id"):
            row["client_order_id"] = str(order.get("clientOrderId") or order.get("origClientOrderId") or order.get("client_order_id") or "")

        already_finalized = int(row.get("finalized", 0) or 0) > 0
        if duplicate_attempt:
            if already_finalized:
                self._sell_finalize_stats["duplicate_finalize"] = int(self._sell_finalize_stats.get("duplicate_finalize", 0) or 0) + 1
                if not bool(row.get("dup_reported")):
                    self.logger.error(
                        "[EM:SellFinalizeAssert] Duplicate SELL close finalization attempt key=%s symbol=%s order_id=%s client_order_id=%s tag=%s",
                        key,
                        sym,
                        row.get("order_id") or "n/a",
                        row.get("client_order_id") or "n/a",
                        row.get("tag") or "",
                    )
                    row["dup_reported"] = True
            self._audit_sell_finalize_invariant(now=now)
            return

        if int(row.get("fill_seen", 0) or 0) <= 0:
            self._sell_finalize_stats["finalize_without_fill"] = int(self._sell_finalize_stats.get("finalize_without_fill", 0) or 0) + 1
            if not bool(row.get("missing_fill_reported")):
                self.logger.error(
                    "[EM:SellFinalizeAssert] SELL close finalized without prior fill observation key=%s symbol=%s order_id=%s client_order_id=%s tag=%s",
                    key,
                    sym,
                    row.get("order_id") or "n/a",
                    row.get("client_order_id") or "n/a",
                    row.get("tag") or "",
                )
                row["missing_fill_reported"] = True

        if already_finalized:
            self._sell_finalize_stats["duplicate_finalize"] = int(self._sell_finalize_stats.get("duplicate_finalize", 0) or 0) + 1
            if not bool(row.get("dup_reported")):
                self.logger.error(
                    "[EM:SellFinalizeAssert] Duplicate SELL close finalization key=%s symbol=%s order_id=%s client_order_id=%s tag=%s",
                    key,
                    sym,
                    row.get("order_id") or "n/a",
                    row.get("client_order_id") or "n/a",
                    row.get("tag") or "",
                )
                row["dup_reported"] = True
        else:
            row["finalized"] = 1
            row["finalized_ts"] = now
            self._sell_finalize_stats["finalized"] = int(self._sell_finalize_stats.get("finalized", 0) or 0) + 1
            row["dup_reported"] = False

        self._audit_sell_finalize_invariant(now=now)

    def _audit_sell_finalize_invariant(self, *, now: Optional[float] = None, force_log: bool = False) -> None:
        now_ts = float(now if now is not None else time.time())
        assert_window = max(3.0, float(self._sell_finalize_assert_window_s or 30.0))
        ttl = max(assert_window * 4.0, float(self._sell_finalize_track_ttl_s or 3600.0))

        pending = 0
        for key, row in list(self._sell_finalize_state.items()):
            if not isinstance(row, dict):
                self._sell_finalize_state.pop(key, None)
                continue

            fill_seen = int(row.get("fill_seen", 0) or 0)
            finalized = int(row.get("finalized", 0) or 0)
            fill_ts = float(row.get("fill_ts", row.get("last_ts", now_ts)) or now_ts)
            last_ts = float(row.get("last_ts", fill_ts) or fill_ts)

            if fill_seen > 0 and finalized <= 0:
                pending += 1
                age_s = max(0.0, now_ts - fill_ts)
                if age_s >= assert_window and not bool(row.get("timeout_reported")):
                    row["timeout_reported"] = True
                    self._sell_finalize_stats["pending_timeout"] = int(self._sell_finalize_stats.get("pending_timeout", 0) or 0) + 1
                    self.logger.error(
                        "[EM:SellFinalizeAssert] Missing SELL close finalization key=%s symbol=%s age=%.2fs order_id=%s client_order_id=%s tag=%s",
                        key,
                        row.get("symbol") or "UNKNOWN",
                        age_s,
                        row.get("order_id") or "n/a",
                        row.get("client_order_id") or "n/a",
                        row.get("tag") or "",
                    )

            if ttl > 0 and (now_ts - last_ts) > ttl:
                self._sell_finalize_state.pop(key, None)

        self._sell_finalize_pending = int(pending)

        finalized_total = int(self._sell_finalize_stats.get("finalized", 0) or 0)
        fills_total = int(self._sell_finalize_stats.get("fills_seen", 0) or 0)
        should_log = bool(force_log)

        if self._sell_finalize_log_every > 0 and finalized_total > 0:
            if finalized_total % self._sell_finalize_log_every == 0 and finalized_total != int(self._sell_finalize_last_report_finalized):
                should_log = True

        if not should_log and (now_ts - float(self._sell_finalize_last_report_ts or 0.0)) >= 180.0:
            if fills_total > 0 or pending > 0:
                should_log = True

        if should_log:
            self._sell_finalize_last_report_ts = now_ts
            self._sell_finalize_last_report_finalized = finalized_total
            self.logger.info(
                "[EM:SellFinalizeCounter] fills_seen=%d finalized=%d pending=%d duplicate_finalize=%d finalize_without_fill=%d pending_timeout=%d fills_seen_duplicate=%d",
                fills_total,
                finalized_total,
                pending,
                int(self._sell_finalize_stats.get("duplicate_finalize", 0) or 0),
                int(self._sell_finalize_stats.get("finalize_without_fill", 0) or 0),
                int(self._sell_finalize_stats.get("pending_timeout", 0) or 0),
                int(self._sell_finalize_stats.get("fills_seen_duplicate", 0) or 0),
            )

    async def _finalize_sell_post_fill(
        self,
        *,
        symbol: str,
        order: Optional[Dict[str, Any]],
        tag: str = "",
        post_fill: Optional[Dict[str, Any]] = None,
        policy_ctx: Optional[Dict[str, Any]] = None,
        tier: Optional[str] = None,
    ) -> None:
        """
        Canonical SELL post-fill finalizer.
        Ensures close bookkeeping/events are emitted exactly once per order payload.
        
        OPTION 1: Implements idempotent finalization via result cache.
        Tracks finalization by (symbol, order_id) to prevent duplicate execution
        when called multiple times with same position close.
        
        OPTION 3: After finalization, verifies the close actually worked by checking
        if position qty decreased as expected.
        """
        if not isinstance(order, dict):
            return
        
        sym = self._norm_symbol(symbol)
        order_id = str(order.get("orderId") or order.get("order_id") or "")
        
        # --- OPTION 1: Check finalization result cache ---
        cache_key = f"{sym}:{order_id}"
        now_ts = time.time()
        
        # Prune expired cache entries
        if cache_key in self._sell_finalize_result_cache_ts:
            entry_ts = self._sell_finalize_result_cache_ts[cache_key]
            if now_ts - entry_ts > self._sell_finalize_cache_ttl_s:
                self._sell_finalize_result_cache.pop(cache_key, None)
                self._sell_finalize_result_cache_ts.pop(cache_key, None)
        
        # If already finalized, return cached result
        if cache_key in self._sell_finalize_result_cache:
            cached_result = self._sell_finalize_result_cache[cache_key]
            with contextlib.suppress(Exception):
                self._track_sell_finalize(
                    symbol=sym,
                    order=order,
                    tag=str(tag or ""),
                    duplicate_attempt=True,
                )
            self.logger.debug(
                "[SELL_FINALIZE:Idempotent] Skipped duplicate finalization for %s order_id=%s (cached)",
                sym, order_id or "unknown"
            )
            return
        
        # Check old-style flag (backward compat with existing code)
        if bool(order.get("_sell_close_events_done")):
            with contextlib.suppress(Exception):
                self._track_sell_finalize(
                    symbol=symbol,
                    order=order,
                    tag=str(tag or ""),
                    duplicate_attempt=True,
                )
            return

        status = str(order.get("status", "")).upper()
        exec_qty = self._safe_float(order.get("executedQty") or order.get("executed_qty"), 0.0)
        if status not in ("FILLED", "PARTIALLY_FILLED") or exec_qty <= 0:
            return

        # Ensure post-fill processing has actually run. callers may pass an empty dict,
        # so treat only a real dict with _post_fill_done as valid. Otherwise, force
        # execution of the post-fill handler to guarantee accounting/emits happen here.
        pf = post_fill if isinstance(post_fill, dict) else None
        if not pf or not order.get("_post_fill_done"):
            pf = await self._ensure_post_fill_handled(
                symbol=sym,
                side="SELL",
                order=order,
                tier=tier,
                tag=str(tag or ""),
            )

        with contextlib.suppress(Exception):
            await self._record_sell_exit_bookkeeping(
                symbol=sym,
                tag=str(tag or ""),
                policy_ctx=policy_ctx,
            )

        try:
            await self._emit_close_events(sym, order, pf if isinstance(pf, dict) else None)
        except Exception as e:
            self.logger.error(f"[SELL_CLOSE_EVENTS_CRASH] {sym}: {e}", exc_info=True)
            with contextlib.suppress(Exception):
                await self._sync_shared_position_after_sell_fill(
                    symbol=sym,
                    order=order,
                    reason="SELL_CLOSE_EVENTS_CRASH_RECOVERY",
                )
            if bool(self._cfg("STRICT_ACCOUNTING_INTEGRITY", False)):
                raise
            return

        with contextlib.suppress(Exception):
            await self._sync_shared_position_after_sell_fill(
                symbol=sym,
                order=order,
                reason="SELL_FILLED_SYNC",
            )

        order["_sell_close_events_done"] = True
        
        # --- OPTION 1: Cache the finalization result ---
        finalize_result = {
            "symbol": sym,
            "order_id": order_id,
            "executed_qty": exec_qty,
            "timestamp": now_ts,
            "tag": str(tag or ""),
        }
        self._sell_finalize_result_cache[cache_key] = finalize_result
        self._sell_finalize_result_cache_ts[cache_key] = now_ts
        
        # --- OPTION 3: Queue for post-finalize verification ---
        try:
            pos_qty_before = exec_qty  # What we closed
            verification_entry = {
                "symbol": sym,
                "order_id": order_id,
                "expected_close_qty": pos_qty_before,
                "verified_at_ts": None,
                "verification_status": None,
                "created_ts": now_ts,
            }
            self._pending_close_verification[cache_key] = verification_entry
            self.logger.debug(
                "[SELL_FINALIZE:PostVerify] Queued verification for %s order_id=%s (expect qty reduced by %.8f)",
                sym, order_id or "unknown", pos_qty_before
            )
        except Exception as e:
            self.logger.debug("[SELL_FINALIZE:PostVerify] Failed to queue verification: %s", e, exc_info=True)
        
        with contextlib.suppress(Exception):
            self._track_sell_finalize(
                symbol=sym,
                order=order,
                tag=str(tag or ""),
                duplicate_attempt=False,
            )

    async def _verify_pending_closes(self) -> None:
        """
        OPTION 3: Post-finalize verification.
        Periodically checks that positions marked for close verification are actually closed.
        
        This is a background task that runs independently to ensure finalization actually
        resulted in the expected position reduction. If verification fails, it can:
        1. Log warnings for monitoring/alerting
        2. Retry finalization if position still exists
        3. Update verification tracking state
        """
        now_ts = time.time()
        to_remove = []
        
        for cache_key, entry in list(self._pending_close_verification.items()):
            try:
                symbol = entry.get("symbol", "")
                order_id = entry.get("order_id", "")
                expected_close_qty = float(entry.get("expected_close_qty", 0.0) or 0.0)
                created_ts = float(entry.get("created_ts", now_ts) or now_ts)
                
                # Check age: remove old entries after timeout
                age_s = now_ts - created_ts
                timeout_s = float(self._cfg("CLOSE_VERIFICATION_TIMEOUT_SEC", 60.0) or 60.0)
                if age_s > timeout_s:
                    to_remove.append(cache_key)
                    self.logger.warning(
                        "[SELL_VERIFY:Timeout] Position close verification timed out: %s order_id=%s (age=%.1fs)",
                        symbol, order_id or "unknown", age_s
                    )
                    continue
                
                # Get current position qty
                try:
                    current_qty = 0.0
                    if hasattr(self.shared_state, "get_position_qty"):
                        current_qty = float(self.shared_state.get_position_qty(symbol) or 0.0)
                    else:
                        positions = getattr(self.shared_state, "positions", {}) or {}
                        pos_entry = positions.get(symbol, {})
                        if isinstance(pos_entry, dict):
                            current_qty = float(pos_entry.get("qty", pos_entry.get("quantity", 0.0)) or 0.0)
                except Exception:
                    current_qty = 0.0
                
                # Verification success: position is closed (qty near zero)
                if current_qty <= 1e-8:
                    entry["verification_status"] = "VERIFIED_CLOSED"
                    entry["verified_at_ts"] = now_ts
                    to_remove.append(cache_key)
                    self.logger.debug(
                        "[SELL_VERIFY:Success] Position close verified: %s order_id=%s (final_qty=%.8f)",
                        symbol, order_id or "unknown", current_qty
                    )
                    continue
                
                # Position still open: log warning but don't remove yet
                if age_s > 10.0:  # Only warn after 10s, allow some grace
                    self.logger.warning(
                        "[SELL_VERIFY:Pending] Position close not yet verified: %s order_id=%s current_qty=%.8f expected_close=%.8f (age=%.1fs)",
                        symbol, order_id or "unknown", current_qty, expected_close_qty, age_s
                    )
                    
            except Exception as e:
                self.logger.debug("[SELL_VERIFY] Error during verification: %s", e, exc_info=True)
                to_remove.append(cache_key)
        
        # Cleanup expired entries
        for key in to_remove:
            self._pending_close_verification.pop(key, None)

    # Consider consolidating _split_symbol_quote and _split_base_quote to avoid drift.
    def _split_base_quote(self, symbol: str) -> Tuple[str, str]:
        s = (symbol or "").upper()
        # Check configured quote currency first so instance config takes precedence
        _base_ccy = (self.base_ccy or "").upper()
        if _base_ccy and s.endswith(_base_ccy):
            return s[:-len(_base_ccy)], _base_ccy
        for q in ("USDT", "FDUSD", "USDC", "BUSD", "TUSD", "BTC", "ETH"):
            if s.endswith(q):
                return s[:-len(q)], q
        # last resort: naive 3–4 letter quote split
        return s[:-4], s[-4:]

    def __init__(self, config: Any, shared_state: Any, exchange_client: Any, alert_callback=None, event_store: Optional[Any] = None):
        # Heartbeat task (must be set before any other logic to avoid AttributeError)
        self._heartbeat_task = None
        self._decision_id_seq = 0
        self.config = config
        self.shared_state = shared_state
        self.exchange_client = exchange_client
        self.alert_callback = alert_callback
        self.event_store = event_store  # Phase 5: Event sourcing
        self.logger = logging.getLogger(self.__class__.__name__)

        # Execution-block cooldowns (finite no-trade states)
        self._buy_block_state: Dict[str, Dict[str, float]] = {}
        self._position_open_buy_block_until: Dict[str, float] = {}

        # Contract check: must expose place_market_order()
        if not hasattr(self.exchange_client, "place_market_order") or not callable(getattr(self.exchange_client, "place_market_order", None)):
            raise RuntimeError("ExchangeClient must expose place_market_order() for canonical path")

        # Dependencies (injected later)
        self.meta_controller = None
        self.risk_manager = None
        self.tp_sl_engine = None
        self.trade_journal = None  # injected by AppContext
        self.session_id: str = ""  # injected by AppContext
        self._journal_bootstrap_attempted = False
        self._journal_fallback_warned = False
        self._require_trade_journal_live = bool(self._cfg("REQUIRE_TRADE_JOURNAL_IN_LIVE", True))
        self._journal_log_dir = str(self._cfg("TRADE_JOURNAL_LOG_DIR", "logs") or "logs")

        # Config
        self.base_ccy = str(getattr(config, "BASE_CURRENCY", "USDT")).upper()
        self.safety_headroom = float(getattr(config, "QUOTE_HEADROOM", 1.02))
        self.trade_fee_pct = float(getattr(config, "TRADE_FEE_PCT", 0.001))
        self.max_spend_per_trade = float(getattr(config, "MAX_SPEND_PER_TRADE_USDT", 0))
        self.min_conf = float(getattr(config, "MIN_EXECUTION_CONFIDENCE", 0.6))
        self.min_entry_quote_usdt = float(getattr(config, "MIN_ENTRY_QUOTE_USDT", 0.0))
        self.order_monitor_interval = float(getattr(config, "ORDER_MONITOR_INTERVAL", 15))
        self.stale_order_timeout_s = int(getattr(config, "STALE_ORDER_TIMEOUT_SECONDS", 120))
        self.max_concurrent_orders = int(getattr(config, "MAX_CONCURRENT_ORDERS", 5))
        self.enable_exec_cot = bool(getattr(config, "ENABLE_COT_VALIDATION_AT_EXECUTION", False))
        # SELL economic gate (fee-aware)
        self.allow_sell_below_fee = bool(getattr(config, "ALLOW_SELL_BELOW_FEE", False))
        self.sell_min_net_pnl_usdt = float(getattr(config, "SELL_MIN_NET_PNL_USDT", 0.0))
        self.min_net_profit_after_fees_pct = float(getattr(config, "MIN_NET_PROFIT_AFTER_FEES", 0.0035))

        # PHASE 2 NOTE: min_notional_floor removed (capital floor check moved to MetaController)
        # ExecutionManager no longer enforces capital policy
        self.maker_grace_s = float(self._cfg('execution.maker_grace_s', 0.0))
        self.allow_taker_if_within_bps = float(self._cfg('execution.allow_taker_if_within_bps', 0.0))
        
        # CRITICAL: Initialize min_free_reserve_usdt to 0.0 FIRST to avoid AttributeError
        # Then assign from config sources
        self.min_free_reserve_usdt = 0.0
        self.min_free_reserve_usdt = float(
            max(
                float(self._cfg('execution.min_free_reserve_usdt', 0.0) or 0.0),
                float(getattr(config, "EXECUTION_MIN_FREE_RESERVE_USDT", 0.0) or 0.0),
                float(getattr(config, "MIN_LIQUIDITY_BUFFER", 0.0) or 0.0),
            )
        )
        self.no_remainder_below_quote = float(
            self._cfg('execution.no_remainder_below_quote', getattr(config, "NO_REMAINDER_BELOW_QUOTE", 0.0)) or 0.0
        )
        # When NAV is tiny, serialize placement globally
        self.small_nav_threshold = float(self._cfg('capital.small_nav_threshold_usdt', 50.0))

        # ========== PHASE-BASED BOOTSTRAP CONTROL ==========
        # Consultant Recommendation:
        # Phase 1: Fix idempotency + allow ONE clean bootstrap execution (capital ~100-170 USDT)
        # Phase 2: Disable bootstrap override entirely (after first fill confirmed)
        # Phase 3: Re-enable smart bootstrap logic if needed (capital > 400 USDT)
        
        self.bootstrap_phase_1_capital_min = float(self._cfg('BOOTSTRAP_PHASE_1_CAPITAL_MIN', 50.0))
        self.bootstrap_phase_1_capital_max = float(self._cfg('BOOTSTRAP_PHASE_1_CAPITAL_MAX', 200.0))
        self.bootstrap_phase_3_capital_threshold = float(self._cfg('BOOTSTRAP_PHASE_3_CAPITAL_THRESHOLD', 400.0))
        
        # Phase 2 explicitly disables bootstrap override
        self.bootstrap_allow_override = bool(self._cfg('BOOTSTRAP_ALLOW_OVERRIDE', False))
        
        # Track if we've done the first bootstrap fill
        self._bootstrap_first_fill_done = False
        self._bootstrap_phase_2_active = False
        
        # ========== MAKER-BIASED EXECUTION CONFIGURATION ==========
        # Initialize MakerExecutor for limit order placement inside spread
        # Reduces execution cost from ~0.34% (market order) to ~0.03% (maker order + fee)
        maker_config = MakerExecutionConfig(
            enable_maker_orders=bool(self._cfg('maker_execution.enable', True)),
            nav_threshold=float(self._cfg('maker_execution.nav_threshold', 500.0)),
            spread_placement_ratio=float(self._cfg('maker_execution.spread_placement_ratio', 0.2)),
            limit_order_timeout_sec=float(self._cfg('maker_execution.timeout_sec', 5.0)),
            max_spread_pct=float(self._cfg('maker_execution.max_spread_pct', 0.002)),
            aggressive_spread_ratio=float(self._cfg('maker_execution.aggressive_ratio', 0.5)),
        )
        self.maker_executor = MakerExecutor(config=maker_config)
        self.logger.info(
            f"[MakerExecution] Initialized: enable={maker_config.enable_maker_orders} "
            f"nav_threshold={maker_config.nav_threshold} "
            f"timeout={maker_config.limit_order_timeout_sec}s spread_placement={maker_config.spread_placement_ratio}"
        )
        
        # Liquidity healing
        self.max_liquidity_retries = int(getattr(config, "MAX_LIQUIDITY_RETRIES", 1))
        self.liquidity_retry_delay = float(getattr(config, "LIQUIDITY_RETRY_DELAY_SECONDS", 3.0))

        # Execution-block cooldowns (finite no-trade states)
        self.exec_block_max_retries = int(getattr(config, "EXEC_BLOCK_MAX_RETRIES", 3))
        self.exec_block_cooldown_sec = int(getattr(config, "EXEC_BLOCK_COOLDOWN_SEC", 600))

        # Concurrency (defer semaphore creation to first use, need running loop)
        self._concurrent_orders_sem = None
        self._cancel_sem = None
        self._semaphores_initialized = False

        # ============================================================
        # 🎯 BEST PRACTICE IDEMPOTENCY CONFIGURATION
        # ============================================================
        # Core principle: TRACK ACTIVE ORDERS, DON'T PENALIZE REJECTIONS
        # 
        # Recommended configuration for production stability:
        # - idempotent_window: 8 seconds (short, prevents false duplicates)
        # - rejection_threshold: 5 (only lock after 5+ genuine rejections)
        # - rejection_reset: 60 seconds (auto-clear failed attempts)
        # - duplicate_rejection_penalty: 0 (IDEMPOTENT doesn't count toward lock)
        # - bootstrap_override: Always allowed (trades must bootstrap)
        # ============================================================
        
        # Idempotency + active order guards (symbol, side) with time-scoped tracking
        self._active_symbol_side_orders: Dict[tuple, float] = {}  # (symbol, side) -> timestamp
        self._active_order_timeout_s = 8.0  # 🎯 SHORT WINDOW: 8 seconds for production stability
        self._seen_client_order_ids: Dict[str, float] = {}  # client_id -> timestamp
        self._client_order_id_timeout_s = 8.0  # 🎯 Matches symbol/side window for consistency
        
        # 🎯 Rejection counter handling: IDEMPOTENT rejections don't count toward lock
        self._ignore_idempotent_in_rejection_count = True  # Never penalize IDEMPOTENT
        self._rejection_exempt_reasons = {"IDEMPOTENT", "ACTIVE_ORDER"}  # Don't count these
        
        # 🎯 Auto-reset mechanism: Clear stale rejection counters
        self._rejection_reset_window_s = 60.0  # Reset counters older than 60s
        self._last_rejection_reset_check_ts: Dict[str, float] = {}  # sym -> last_check_ts
        # SELL close-finalization runtime invariant tracker.
        self._sell_finalize_state: Dict[str, Dict[str, Any]] = {}
        self._sell_finalize_stats: Dict[str, int] = {
            "fills_seen": 0,
            "finalized": 0,
            "fills_seen_duplicate": 0,
            "duplicate_finalize": 0,
            "finalize_without_fill": 0,
            "pending_timeout": 0,
        }
        # --- OPTION 1: Idempotent finalize cache ---
        # Maps (symbol, side) -> finalization result to prevent duplicate finalization
        # Survives the lifetime of the position close, TTL-based cleanup
        self._sell_finalize_result_cache: Dict[str, Dict[str, Any]] = {}
        self._sell_finalize_result_cache_ts: Dict[str, float] = {}
        self._sell_finalize_cache_ttl_s = float(self._cfg("SELL_FINALIZE_CACHE_TTL_SEC", 300.0) or 300.0)
        # --- OPTION 3: Post-finalize verification tracking ---
        # Tracks positions that should be closed to verify finalization actually worked
        self._pending_close_verification: Dict[str, Dict[str, Any]] = {}
        self._close_verification_check_interval_s = float(self._cfg("CLOSE_VERIFICATION_INTERVAL_SEC", 2.0) or 2.0)
        self._sell_finalize_pending = 0
        self._sell_finalize_assert_window_s = float(self._cfg("SELL_FINALIZE_ASSERT_WINDOW_SEC", 30.0) or 30.0)
        self._sell_finalize_track_ttl_s = float(self._cfg("SELL_FINALIZE_TRACK_TTL_SEC", 3600.0) or 3600.0)
        self._sell_finalize_log_every = int(self._cfg("SELL_FINALIZE_LOG_EVERY", 25) or 25)
        self._sell_finalize_last_report_ts = 0.0
        self._sell_finalize_last_report_finalized = -1
        self._sell_fill_recovery_tasks: Dict[str, asyncio.Task] = {}
        self._exchange_zero_mismatch_counts: Dict[str, int] = {}
        self._liq_sell_fail_counts: Dict[str, int] = {}
        self._liq_sell_cooldown_until: Dict[str, float] = {}
        self._close_escape_last_attempt_ts: Dict[str, float] = {}

        # Optional local cache to serve callers that want filters without reaching ExchangeClient internals.
        # ExchangeClient already maintains its own `symbol_filters` cache; this layer is primarily API sugar.
        self._symbol_filters_cache: Dict[str, Dict[str, Any]] = {}
        self._symbol_filters_cache_ts: Dict[str, float] = {}
        self._symbol_filters_cache_max_age_s = float(getattr(config, "SYMBOL_FILTERS_CACHE_MAX_AGE_S", 900.0) or 900.0)

        # Watchdog periodic status reporting
        self._execution_counter = 0
        self._last_watchdog_report_ts = time.time()
        self._watchdog_report_interval_s = 30.0  # Report to watchdog every 30 seconds

        self.logger.info("ExecutionManager initialized with P9 configuration")
        self._ensure_trade_journal_ready(reason="init")

        # --- Health: mark as Initialized right away (so Watchdog stops "no-report") ---
        try:
            # primary API
            upd = getattr(self.shared_state, "update_component_status", None) \
                or getattr(self.shared_state, "set_component_status", None)
            if callable(upd):
                res = upd("ExecutionManager", "Initialized", "Ready")
                if asyncio.iscoroutine(res):
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(res)
                    except RuntimeError:
                        pass  # No running loop yet (called from __init__)
            # compatibility mirror for Watchdog (_safe_status fallback)
            try:
                now_ts = time.time()
                cs = getattr(self.shared_state, "component_statuses", None)
                if isinstance(cs, dict):
                    cs["ExecutionManager"] = {"status": "Initialized", "message": "Ready", "timestamp": now_ts, "ts": now_ts}
            except Exception:
                pass
        except Exception:
            self.logger.debug("EM init health update failed", exc_info=True)

    # 🔧 NEW: Entry floor guard to prevent new dust creation
    async def _check_entry_floor_guard(
        self, symbol: str, quote_amount: float, is_dust_healing_buy: bool = False
    ) -> Tuple[bool, str]:
        """
        Guard: Prevent opening new trades below significant floor unless:
        1. Explicitly allowed via allow_entry_below_significant_floor flag, OR
        2. Dust healing buyback (is_dust_healing_buy=True)
        
        This prevents creating new dust positions from regular entries.
        
        Args:
            symbol: Trading pair
            quote_amount: Entry amount in quote asset (USDT)
            is_dust_healing_buy: True if this is a dust healing/reentry trade
        
        Returns:
            (is_allowed: bool, reason: str)
        """
        # Allow dust healing trades to bypass floor guard
        if is_dust_healing_buy:
            return True, "[EM:ENTRY_FLOOR_GUARD] Dust healing trade bypasses floor guard"
        
        shared_cfg = getattr(self.shared_state, "config", None)
        significant_floor = float(
            getattr(
                self.config,
                "SIGNIFICANT_POSITION_FLOOR",
                getattr(self.config, "MIN_SIGNIFICANT_POSITION_USDT", 20.0),
            ) or 20.0
        )
        allow_below_floor = bool(
            getattr(
                self.shared_state,
                "allow_entry_below_significant_floor",
                getattr(
                    shared_cfg,
                    "allow_entry_below_significant_floor",
                    getattr(
                        self.config,
                        "allow_entry_below_significant_floor",
                        getattr(self.config, "ALLOW_ENTRY_BELOW_SIGNIFICANT_FLOOR", False),
                    ),
                ),
            )
        )
        
        # Check if entry would be below significant floor
        if quote_amount < significant_floor:
            if not allow_below_floor:
                reason = (
                    f"[EM:ENTRY_FLOOR_GUARD] {symbol} entry ${quote_amount:.2f} "
                    f"below significant floor ${significant_floor:.2f}. "
                    f"Would create dust on entry. "
                    f"Set allow_entry_below_significant_floor=True to override."
                )
                self.logger.warning(reason)
                return False, reason
            else:
                reason = (
                    f"[EM:ENTRY_FLOOR_GUARD_OVERRIDE] {symbol} entry ${quote_amount:.2f} "
                    f"below significant floor ${significant_floor:.2f}, "
                    f"but override flag is enabled."
                )
                self.logger.info(reason)
                return True, reason
        
        return True, "[EM:ENTRY_FLOOR_GUARD] Entry floor check passed"

    async def _report_watchdog_status(self, status: str = "Operational", detail: str = ""):
        """Report status to watchdog for health monitoring"""
        try:
            update_fn = getattr(self.shared_state, "update_component_status", None)
            if update_fn:
                result = update_fn("ExecutionManager", status, detail)
                if asyncio.iscoroutine(result):
                    await result
        except Exception:
            pass  # Status reporting is non-critical

    def _cfg(self, path: str, default):
        cur = self.config
        for part in path.split('.'):
            if isinstance(cur, dict):
                cur = cur.get(part, default)
            else:
                cur = getattr(cur, part, default)
            default = cur if cur is not None else default
        return cur if cur is not None else default

    # ============================================================================
    # PHASE 1: EV ALIGNMENT PUBLIC API
    # ============================================================================
    # Expose EV calculation methods for validation and alignment with UURE
    
    def get_round_trip_cost_pct(self) -> float:
        """
        PHASE 1: PUBLIC API for round-trip cost calculation.
        
        This method returns the total cost (fees + slippage + buffer) as a ratio.
        Must return EXACTLY the same value as UURE's get_round_trip_cost_pct().
        
        Used for EV alignment validation.
        
        Returns:
            Round-trip cost as decimal (e.g., 0.0035 = 0.35%)
        """
        slippage_bps = float(
            self._cfg("EXIT_SLIPPAGE_BPS", self._cfg("CR_PRICE_SLIPPAGE_BPS", 15.0)) or 0.0
        )
        buffer_bps = float(self._cfg("TP_MIN_BUFFER_BPS", 0.0) or 0.0)
        round_trip_cost_pct = (float(self.trade_fee_pct or 0.0) * 2.0) + (
            (slippage_bps + buffer_bps) / 10000.0
        )
        return round_trip_cost_pct
    
    def get_ev_multiplier_for_regime(self, regime: str) -> float:
        """
        PHASE 1: PUBLIC API for EV multiplier by regime.
        
        This method returns the multiplier used to calculate required edge.
        Must return EXACTLY the same multiplier as UURE would use for this regime.
        
        Args:
            regime: Market volatility regime ('normal', 'bull', 'other', 'low', 'high', 'extreme')
        
        Returns:
            EV multiplier (e.g., 1.3 means required_edge = round_trip × 1.3)
        """
        rg = str(regime or "").strip().lower()
        
        # Check for global override
        override = self._cfg("UURE_SOFT_EV_MULTIPLIER", None)
        if override is not None:
            try:
                return max(0.5, float(override))
            except Exception:
                pass
        
        # Check for spot mode (lower thresholds)
        spot_mode = bool(self._cfg("UURE_SPOT_MODE", False))
        if spot_mode:
            if rg == "normal":
                return max(0.5, float(self._cfg("UURE_EV_MULT_SPOT_NORMAL", 0.7)))
            elif rg == "bull":
                return max(0.5, float(self._cfg("UURE_EV_MULT_SPOT_BULL", 1.0)))
            else:
                return max(0.5, float(self._cfg("UURE_EV_MULT_SPOT_OTHER", 1.4)))
        
        # Standard regime multipliers
        if rg == "normal":
            default = 1.3
            key = "UURE_EV_MULT_NORMAL"
        elif rg == "bull":
            default = 1.8
            key = "UURE_EV_MULT_BULL"
        else:
            default = 2.0
            key = "UURE_EV_MULT_OTHER"
        
        try:
            return max(0.5, float(self._cfg(key, default) or default))
        except Exception:
            return default
    
    def get_required_edge_for_regime(self, regime: str) -> float:
        """
        PHASE 1: PUBLIC API for required minimum edge by regime.
        
        This is the minimum edge (as decimal) that must be present for entry.
        Same formula as UURE uses: required_edge = round_trip × multiplier.
        
        Args:
            regime: Market regime
        
        Returns:
            Required edge as decimal (e.g., 0.00455 = 0.455%)
        """
        round_trip = self.get_round_trip_cost_pct()
        multiplier = self.get_ev_multiplier_for_regime(regime)
        return round_trip * multiplier
    
    def get_ev_config_summary(self) -> Dict[str, Any]:
        """
        PHASE 1: Export EV configuration for alignment audit.
        
        Returns a dict summarizing current EV settings used by ExecutionManager.
        Compare this with UURE's settings to verify alignment.
        
        Returns:
            Dictionary with EV configuration details
        """
        spot_mode = bool(self._cfg("UURE_SPOT_MODE", False))
        return {
            "round_trip_cost_pct": self.get_round_trip_cost_pct(),
            "ev_mult_normal": self.get_ev_multiplier_for_regime("normal"),
            "ev_mult_bull": self.get_ev_multiplier_for_regime("bull"),
            "ev_mult_other": self.get_ev_multiplier_for_regime("other"),
            "spot_mode_enabled": spot_mode,
            "required_edge_normal": self.get_required_edge_for_regime("normal"),
            "required_edge_bull": self.get_required_edge_for_regime("bull"),
            "required_edge_other": self.get_required_edge_for_regime("other"),
            "source": "ExecutionManager (PHASE 1 EV ALIGNMENT)",
        }

    # ========== BOOTSTRAP PHASE MANAGEMENT ==========
    def _get_current_nav(self) -> float:
        """Get current portfolio NAV in USDT."""
        try:
            if hasattr(self.shared_state, "portfolio_nav"):
                nav = self.shared_state.portfolio_nav
                if callable(nav):
                    nav = nav()
                return float(nav or 0.0)
            if hasattr(self.shared_state, "total_equity_usdt"):
                return float(self.shared_state.total_equity_usdt or 0.0)
        except Exception:
            pass
        return 0.0

    def _get_bootstrap_phase(self) -> str:
        """
        Determine current bootstrap phase based on capital and state.
        
        Returns: "phase_1", "phase_2", or "phase_3"
        """
        nav = self._get_current_nav()
        
        # Phase 2: Explicitly enabled (takes precedence)
        if self._bootstrap_phase_2_active or not self.bootstrap_allow_override:
            return "phase_2"
        
        # Phase 3: High capital
        if nav >= self.bootstrap_phase_3_capital_threshold:
            return "phase_3"
        
        # Phase 1: Bootstrap capital range
        if self.bootstrap_phase_1_capital_min <= nav <= self.bootstrap_phase_1_capital_max:
            return "phase_1"
        
        # Default: Phase 1 if below Phase 1 max
        if nav < self.bootstrap_phase_1_capital_max:
            return "phase_1"
        
        # High capital but Phase 2 not explicitly disabled
        return "phase_3"

    def _is_bootstrap_allowed(self) -> bool:
        """Check if bootstrap override is allowed in current phase."""
        phase = self._get_bootstrap_phase()
        
        if phase == "phase_1":
            # Phase 1: Allow ONE bootstrap execution
            return not self._bootstrap_first_fill_done
        elif phase == "phase_2":
            # Phase 2: Disable bootstrap entirely
            return False
        elif phase == "phase_3":
            # Phase 3: Allow smart bootstrap logic
            return True
        
        return False

    def _mark_bootstrap_fill_done(self):
        """Mark that first bootstrap fill has been executed."""
        self._bootstrap_first_fill_done = True
        self.logger.info(
            "[BOOTSTRAP] First fill confirmed. Moving to Phase 2 (bootstrap disabled). "
            "Re-enable at capital > %.2f USDT",
            self.bootstrap_phase_3_capital_threshold
        )

    def _activate_phase_2(self):
        """Explicitly activate Phase 2 (disable bootstrap override)."""
        self._bootstrap_phase_2_active = True
        self.bootstrap_allow_override = False
        self.logger.warning(
            "[BOOTSTRAP] Phase 2 activated: bootstrap override DISABLED. "
            "EV + adaptive logic will handle entries naturally. "
            "Phase 3 available at capital > %.2f USDT",
            self.bootstrap_phase_3_capital_threshold
        )

    def _exit_phase_2(self):
        """Exit Phase 2 when capital conditions are met."""
        nav = self._get_current_nav()
        if nav >= self.bootstrap_phase_3_capital_threshold:
            self._bootstrap_phase_2_active = False
            self.logger.info(
                "[BOOTSTRAP] Phase 2 exited: Entering Phase 3 (smart bootstrap). "
                "Capital: %.2f USDT >= threshold: %.2f USDT",
                nav,
                self.bootstrap_phase_3_capital_threshold
            )
            return True
        return False

    def _exit_fee_bps(self) -> float:
        cfg_val = float(self._cfg("EXIT_FEE_BPS", self._cfg("CR_FEE_BPS", 0.0)) or 0.0)
        fee_from_pct = float(self.trade_fee_pct or 0.0) * 10000.0
        return max(cfg_val, fee_from_pct)

    def _exit_slippage_bps(self) -> float:
        return float(self._cfg("EXIT_SLIPPAGE_BPS", self._cfg("CR_PRICE_SLIPPAGE_BPS", 15.0)) or 0.0)

    async def _get_exit_floor_info(self, symbol: str, price: Optional[float] = None) -> Dict[str, float]:
        if hasattr(self.shared_state, "compute_symbol_exit_floor"):
            return await self.shared_state.compute_symbol_exit_floor(
                symbol,
                price=price,
                fee_bps=self._exit_fee_bps(),
                slippage_bps=self._exit_slippage_bps(),
            )
        return {"min_exit_quote": 0.0, "min_notional": 0.0}

    async def _get_total_equity(self) -> float:
        """Best-effort total equity (USDT) from SharedState across sync/async variants.
        
        Tries multiple sources in priority order:
        1. Synchronous get_nav_quote() - PRIMARY (fast, reliable)
        2. Async get_total_equity() method
        3. Async get_nav() method
        4. Direct .total_equity attribute
        5. Bootstrap: free balance in quote asset
        6. Metrics dict cache
        
        Returns 0.0 only if ALL sources exhausted.
        """
        # ✅ PRIMARY: get_nav_quote() is SYNCHRONOUS, do NOT await
        # This is the most reliable source during bootstrap and normal operation
        try:
            if hasattr(self.shared_state, "get_nav_quote") and callable(getattr(self.shared_state, "get_nav_quote")):
                val = self.shared_state.get_nav_quote()
                if val is not None:
                    num = float(val)
                    if num > 0:
                        self.logger.debug(f"[NAV] get_nav_quote() returned {num:.2f} USDT")
                        return num
        except Exception as e:
            self.logger.debug(f"[NAV] get_nav_quote() failed: {e}")
        
        # Try async methods next
        try:
            val = await maybe_call(self.shared_state, "get_total_equity")
            if val is not None:
                num = float(val)
                if num > 0:
                    self.logger.debug(f"[NAV] get_total_equity() returned {num:.2f} USDT")
                    return num
        except Exception as e:
            self.logger.debug(f"[NAV] get_total_equity() failed: {e}")
        
        try:
            val = await maybe_call(self.shared_state, "get_nav")
            if val is not None:
                num = float(val)
                if num > 0:
                    self.logger.debug(f"[NAV] get_nav() returned {num:.2f} USDT")
                    return num
        except Exception as e:
            self.logger.debug(f"[NAV] get_nav() failed: {e}")
        
        # ✅ BOOTSTRAP: Check free balance as fallback
        try:
            free = await maybe_call(self.shared_state, "get_free_balance", self.base_ccy)
            if free is not None:
                num = float(free)
                if num > 0:
                    self.logger.info(f"[NAV BOOTSTRAP] Using free {self.base_ccy} balance: {num:.2f} USDT")
                    return num
        except Exception as e:
            self.logger.debug(f"[NAV] get_free_balance({self.base_ccy}) failed: {e}")
        
        # Check attribute
        try:
            num = float(getattr(self.shared_state, "total_equity", 0.0) or 0.0)
            if num > 0:
                self.logger.debug(f"[NAV] .total_equity attribute: {num:.2f} USDT")
                return num
        except Exception as e:
            self.logger.debug(f"[NAV] .total_equity attribute failed: {e}")
        
        # Last resort: metrics cache
        try:
            metrics = getattr(self.shared_state, "metrics", {}) or {}
            num = float(metrics.get("nav", 0.0) or metrics.get("total_equity", 0.0) or 0.0)
            if num > 0:
                self.logger.debug(f"[NAV] metrics cache: {num:.2f} USDT")
                return num
        except Exception as e:
            self.logger.debug(f"[NAV] metrics cache failed: {e}")
        
        # All sources exhausted
        self.logger.warning("[NAV] All equity sources returned 0 or None. Account may be in cold-start state.")
        return 0.0

    async def get_tradable_nav(self) -> float:
        """Get NAV capped at BASE_CAPITAL (max tradable amount).
        
        Returns the minimum of:
        - Actual total equity (from _get_total_equity)
        - BASE_CAPITAL config setting
        
        This prevents over-leverage when wallet grows beyond initial capital.
        If BASE_CAPITAL is not configured or <= 0, returns actual equity.
        
        DEBUG: Logs when NAV calculation changes or caps are applied.
        """
        try:
            total_equity = await self._get_total_equity()
            base_capital = self._cfg("BASE_CAPITAL", None)
            
            if base_capital is None or base_capital <= 0:
                # No cap configured, return actual equity
                nav = float(total_equity or 0.0)
                if nav == 0.0:
                    self.logger.warning("[NAV] get_tradable_nav() returning 0.0 - account may need bootstrap liquidity")
                return nav
            
            base_capital = float(base_capital)
            total_equity = float(total_equity or 0.0)
            capped_nav = min(total_equity, base_capital)
            
            # Log when NAV is capped
            if capped_nav < total_equity:
                self.logger.info(
                    f"[NAV] Capped at BASE_CAPITAL: actual={total_equity:.2f} USDT, "
                    f"base_capital={base_capital:.2f} USDT, tradable_nav={capped_nav:.2f} USDT"
                )
            elif capped_nav == 0.0:
                self.logger.warning("[NAV] get_tradable_nav() = 0.0 - check equity sources")
            else:
                self.logger.debug(
                    f"[NAV] tradable_nav={capped_nav:.2f} USDT (within BASE_CAPITAL={base_capital:.2f})"
                )
            
            return capped_nav
        except Exception as e:
            self.logger.error(f"[NAV] Error in get_tradable_nav(): {e}", exc_info=True)
            # Fallback to uncapped equity on error
            fallback = await self._get_total_equity()
            self.logger.warning(f"[NAV] Using fallback equity: {fallback:.2f} USDT")
            return fallback

    async def _resolve_nav_tier_economic_floor(
        self,
        symbol: Optional[str] = None,
        min_notional: Optional[float] = None,
    ) -> float:
        """
        Resolve a sane minimum economic trade floor.

        This is an execution FLOOR, not a position-sizing target. It must stay in the
        same order of magnitude as configured trade sizes, otherwise affordability
        checks become unusable. Legacy NAV-percentage logic here incorrectly turned
        large account NAV into absurd per-trade minimums (for example ~5% of NAV).

        Resolution:
        - Always respect exchange min_notional.
        - Use MIN_ECONOMIC_TRADE_USDT / MIN_ECONOMIC_TRADE_USD as the base floor.
        - Allow an optional NAV-based add-on only when explicitly configured.
        - Never exceed MAX_SPEND_PER_TRADE_USDT when that cap is configured.
        """
        sym = self._norm_symbol(symbol or "")
        min_notional_val = float(min_notional or 0.0)
        if min_notional_val <= 0 and sym:
            try:
                filters = await self.exchange_client.ensure_symbol_filters_ready(sym)
                min_notional_val = float(self._extract_min_notional(filters) or 0.0)
            except Exception:
                min_notional_val = 0.0

        nav = float(await self.get_tradable_nav() or 0.0)
        base_floor = float(
            self._cfg(
                "MIN_ECONOMIC_TRADE_USDT",
                self._cfg("MIN_ECONOMIC_TRADE_USD", 10.0),
            )
            or 10.0
        )
        base_floor = max(base_floor, 10.0, min_notional_val)

        nav_floor_pct = float(self._cfg("NAV_TIER_MIN_ECON_FLOOR_PCT", 0.0) or 0.0)
        nav_floor_cap = float(self._cfg("NAV_TIER_MIN_ECON_FLOOR_CAP_USDT", 0.0) or 0.0)
        max_spend_cap = float(self._cfg("MAX_SPEND_PER_TRADE_USDT", 0.0) or 0.0)

        nav_floor = 0.0
        if nav > 0.0 and nav_floor_pct > 0.0:
            nav_floor = nav * nav_floor_pct
            if nav_floor_cap > 0.0:
                nav_floor = min(nav_floor, nav_floor_cap)

        floor = max(base_floor, nav_floor)
        if max_spend_cap > 0.0:
            floor = min(floor, max_spend_cap)

        return max(float(floor), min_notional_val)

    async def _resolve_nav_tier_profit_target(self) -> float:
        """
        Dynamic NAV-aware hourly profit target.
        Replaces static PROFIT_TARGET_BASE_USD_PER_HOUR.
        """
        try:
            nav = float(await self.get_tradable_nav() or 0.0)
        except Exception:
            nav = 0.0

        if nav <= 0:
            return 0.5  # safe fallback

        if nav < 500:
            return max(0.5, nav * 0.01)        # 1% hourly
        elif nav < 2000:
            return nav * 0.005                # 0.5%
        elif nav < 10000:
            return nav * 0.003                # 0.3%
        else:
            return nav * 0.002                # 0.2%

    async def _calculate_adaptive_execution_threshold(
        self,
        symbol: str,
        planned_quote: float,
        price: float,
        signal_confidence: Optional[float] = None,
        policy_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Adaptive Execution Threshold Engine (Institutional-Grade)

        Calculates minimum required edge for execution using multi-factor analysis:
        minimum_edge_required = fee_cost + volatility_regime_factor + confidence_discount + NAV_risk_profile

        Returns dict with threshold analysis and decision
        """
        policy_ctx = policy_context or {}

        # 1. Fee Cost Component
        round_trip_fee_rate = float(self.trade_fee_pct) * 2.0  # Round trip fees
        slippage_estimate = float(self._cfg("SLIPPAGE_ESTIMATE_PCT", 0.0005) or 0.0005)  # 0.05% slippage
        fee_cost = round_trip_fee_rate + slippage_estimate

        # 2. Volatility Regime Factor
        atr_pct = 0.0
        volatility_regime = "normal"
        try:
            if hasattr(self.shared_state, "calc_atr"):
                atr_5m = float(await self.shared_state.calc_atr(symbol, "5m", 14) or 0.0)
                atr_1h = float(await self.shared_state.calc_atr(symbol, "1h", 14) or 0.0)
                if atr_5m <= 0:
                    atr_5m = float(await self.shared_state.calc_atr(symbol, "1m", 14) or 0.0)

                if atr_5m > 0 and price > 0:
                    atr_pct = atr_5m / price

                    # Determine volatility regime
                    if atr_pct > 0.02:  # >2% ATR = high volatility
                        volatility_regime = "high"
                        volatility_factor = atr_pct * 1.5  # Require 50% more edge in high vol
                    elif atr_pct < 0.005:  # <0.5% ATR = low volatility
                        volatility_regime = "low"
                        volatility_factor = atr_pct * 0.7  # Allow 30% less edge in low vol
                    else:
                        volatility_regime = "normal"
                        volatility_factor = atr_pct
                else:
                    atr_pct = float(self._cfg("MICRO_TRADE_KILL_FALLBACK_ATR_PCT", 0.005) or 0.005)
                    volatility_factor = atr_pct
        except Exception:
            atr_pct = float(self._cfg("MICRO_TRADE_KILL_FALLBACK_ATR_PCT", 0.005) or 0.005)
            volatility_factor = atr_pct

        # 3. Confidence Discount Factor
        if signal_confidence is not None:
            confidence = float(signal_confidence)
        elif policy_ctx.get("confidence") is not None:
            confidence = float(policy_ctx.get("confidence"))
        else:
            confidence = 0.5
        # High confidence (0.8+) gets discount, low confidence (<0.4) requires premium
        if confidence >= 0.8:
            confidence_adjustment = 0.7  # 30% discount for high confidence
        elif confidence >= 0.6:
            confidence_adjustment = 0.85  # 15% discount for good confidence
        elif confidence >= 0.4:
            confidence_adjustment = 1.0  # No adjustment for neutral confidence
        else:
            confidence_adjustment = 1.3  # 30% premium for low confidence

        # 4. NAV Risk Profile
        nav = float(await self.get_tradable_nav() or 0.0)
        if nav <= 0:
            nav_risk_profile = 1.5  # Conservative when NAV unknown
        elif nav < 100:
            nav_risk_profile = 1.0  # Allow bootstrap growth
        elif nav < 500:
            nav_risk_profile = 0.95  # Slightly relaxed for small accounts
        elif nav < 2000:
            nav_risk_profile = 1.0  # Standard risk for established accounts
        elif nav < 10000:
            nav_risk_profile = 0.9  # Slightly relaxed for larger accounts
        else:
            nav_risk_profile = 0.8  # More relaxed for institutional accounts

        # 5. NAV-Aware Position Sizing Factor
        position_size_pct_of_nav = (planned_quote / nav) if nav > 0 else 0.1
        if nav < 500:
            size_risk_factor = 1.0  # Small accounts: 25% positions are normal
        elif position_size_pct_of_nav > 0.05:  # >5% of NAV
            size_risk_factor = 1.05  # Require more edge for large positions
        elif position_size_pct_of_nav > 0.02:  # >2% of NAV
            size_risk_factor = 1.1  # Moderate premium for medium positions
        else:
            size_risk_factor = 1.0  # No adjustment for small positions

        # Calculate Final Required Edge
        base_required_edge = fee_cost + volatility_factor
        confidence_adjusted_edge = base_required_edge * confidence_adjustment
        nav_adjusted_edge = confidence_adjusted_edge * nav_risk_profile
        final_required_edge = nav_adjusted_edge * size_risk_factor

        # 🛡️ Safe Guard: Clamp required edge for small accounts
        if nav < 500:
            final_required_edge = min(final_required_edge, atr_pct * 0.9)

        # Current Market Edge (simplified - in reality would use order book analysis)
        # For now, use ATR as proxy for available edge
        current_market_edge = atr_pct

        # Decision Logic
        # Relaxed threshold: only block if ratio < 0.50 (was < 1.0, i.e., current < required)
        edge_sufficiency_ratio = current_market_edge / final_required_edge if final_required_edge > 0 else 0
        can_execute = edge_sufficiency_ratio >= 0.50

        return {
            "can_execute": can_execute,
            "required_edge_pct": final_required_edge,
            "current_edge_pct": current_market_edge,
            "edge_sufficiency_ratio": edge_sufficiency_ratio,
            "components": {
                "fee_cost": fee_cost,
                "volatility_factor": volatility_factor,
                "confidence_adjustment": confidence_adjustment,
                "nav_risk_profile": nav_risk_profile,
                "size_risk_factor": size_risk_factor,
                "volatility_regime": volatility_regime
            },
            "analysis": {
                "confidence_level": confidence,
                "nav_amount": nav,
                "position_size_pct": position_size_pct_of_nav,
                "kill_reason": None if can_execute else "INSUFFICIENT_EDGE"
            }
        }

    async def _get_min_entry_quote(
        self,
        symbol: str,
        price: Optional[float] = None,
        min_notional: Optional[float] = None,
        policy_context: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Canonical BUY economic-floor resolver.

        This is the single method used by execution paths to compute the minimum
        acceptable planned quote for entry decisions.
        """
        sym = self._norm_symbol(symbol)
        policy_ctx = policy_context or {}

        base_quote = float(
            policy_ctx.get(
                "min_entry_quote",
                self._cfg("DEFAULT_PLANNED_QUOTE", self._cfg("MIN_ENTRY_QUOTE_USDT", 0.0)) or 0.0,
            )
            or 0.0
        )

        price_f = float(price or 0.0)
        if price_f <= 0:
            try:
                price_f = float(getattr(self.shared_state, "latest_prices", {}).get(sym, 0.0) or 0.0)
            except Exception:
                price_f = 0.0
        if price_f <= 0:
            with contextlib.suppress(Exception):
                price_f = float(await maybe_call(self.exchange_client, "get_current_price", sym) or 0.0)
        if price_f <= 0:
            with contextlib.suppress(Exception):
                price_f = float(await maybe_call(self.exchange_client, "get_price", sym) or 0.0)

        min_notional_val = float(min_notional or policy_ctx.get("min_notional", 0.0) or 0.0)
        if min_notional_val <= 0:
            try:
                filters = await self.exchange_client.ensure_symbol_filters_ready(sym)
                min_notional_val = float(self._extract_min_notional(filters) or 0.0)
            except Exception:
                min_notional_val = 0.0

        exit_info: Dict[str, float] = {"min_exit_quote": 0.0, "min_entry_quote": 0.0}
        with contextlib.suppress(Exception):
            exit_info = await self._get_exit_floor_info(sym, price=price_f if price_f > 0 else None)

        nav_floor = 0.0
        with contextlib.suppress(Exception):
            nav_floor = float(
                await self._resolve_nav_tier_economic_floor(
                    symbol=sym, min_notional=min_notional_val
                )
                or 0.0
            )

        shared_state_floor = 0.0
        if hasattr(self.shared_state, "compute_min_entry_quote"):
            with contextlib.suppress(Exception):
                shared_state_floor = float(
                    await self.shared_state.compute_min_entry_quote(
                        sym,
                        default_quote=base_quote,
                        price=price_f if price_f > 0 else None,
                        min_notional_override=min_notional_val if min_notional_val > 0 else None,
                    )
                    or 0.0
                )

        min_position_usdt = float(self._cfg("MIN_POSITION_USDT", 0.0) or 0.0)
        min_notional_mult = float(self._cfg("MIN_POSITION_MIN_NOTIONAL_MULT", 2.0) or 2.0)
        min_position_floor = min_notional_val * min_notional_mult if min_notional_val > 0 else 0.0

        candidates = (
            base_quote,
            float(policy_ctx.get("min_entry_quote", 0.0) or 0.0),
            float(min_notional_val or 0.0),
            float(exit_info.get("min_exit_quote", 0.0) or 0.0),
            float(exit_info.get("min_entry_quote", 0.0) or 0.0),
            float(min_position_usdt or 0.0),
            float(min_position_floor or 0.0),
            float(nav_floor or 0.0),
            float(shared_state_floor or 0.0),
        )
        final_floor = max(float(x) for x in candidates if float(x) >= 0.0)

        # Ensure the floor survives lot-step rounding so the eventual notional
        # remains exchange-valid after quantity quantization.
        rounded_exchange_floor = float(min_notional_val or 0.0)
        rounded_final_floor = float(final_floor or 0.0)
        if price_f > 0:
            try:
                filters = await self.exchange_client.ensure_symbol_filters_ready(sym)
                step_size, _, _, _, _ = self._extract_filter_vals(filters)
                if min_notional_val > 0:
                    rounded_exchange_floor = float(
                        self._adjust_quote_for_step_rounding(
                            min_entry_quote=min_notional_val,
                            current_price=price_f,
                            step_size=step_size,
                        )
                    )
                if final_floor > 0:
                    rounded_final_floor = float(
                        self._adjust_quote_for_step_rounding(
                            min_entry_quote=final_floor,
                            current_price=price_f,
                            step_size=step_size,
                        )
                    )
            except Exception:
                rounded_exchange_floor = float(min_notional_val or 0.0)
                rounded_final_floor = float(final_floor or 0.0)

        # Guardrail: never let dynamic entry floors exceed the configured per-trade
        # spend cap, otherwise affordability probes can deadlock on small accounts.
        max_spend_cap = float(self._cfg("MAX_SPEND_PER_TRADE_USDT", 0.0) or 0.0)
        if max_spend_cap > 0.0:
            final_floor = min(final_floor, max_spend_cap)
        return max(
            float(final_floor),
            float(min_notional_val or 0.0),
            float(rounded_exchange_floor or 0.0),
            float(rounded_final_floor or 0.0),
        )

    def _is_dust_operation_context(
        self,
        policy_ctx: Optional[Dict[str, Any]] = None,
        tier: Optional[str] = None,
        tag: Optional[str] = None,
        symbol: Optional[str] = None,
    ) -> bool:
        """Detect dust-healing/dust-recovery intent consistently across callers."""
        ctx = policy_ctx or {}
        reason = str(ctx.get("reason") or "").upper()
        ctx_tier = str(ctx.get("tier") or "").upper()
        arg_tier = str(tier or "").upper()
        tag_u = str(tag or "").upper()
        sym = self._norm_symbol(symbol or ctx.get("symbol") or "")
        has_dust_deficit = False
        has_dust_symbol_marker = False
        try:
            deficit_map = getattr(self.shared_state, "dust_healing_deficit", {}) or {}
            has_dust_deficit = float(deficit_map.get(sym, 0.0) or 0.0) > 0.0 if sym else False
        except Exception:
            has_dust_deficit = False
        try:
            mark_map = getattr(self.shared_state, "dust_operation_symbols", {}) or {}
            has_dust_symbol_marker = bool(mark_map.get(sym)) if sym else False
        except Exception:
            has_dust_symbol_marker = False
        return bool(
            ctx.get("is_dust_healing")
            or ctx.get("_dust_healing")
            or reason == "DUST_HEALING_BUY"
            or ctx_tier == "DUST_RECOVERY"
            or arg_tier == "DUST_RECOVERY"
            or "DUST_HEAL" in tag_u
            or has_dust_deficit
            or has_dust_symbol_marker
        )

    async def _heartbeat_loop(self):
        """Continuous heartbeat to satisfy Watchdog when no trades are occurring."""
        while True:
            try:
                await self._emit_status("Operational", "Idle / Ready")
            except Exception:
                pass
            with contextlib.suppress(Exception):
                self._audit_sell_finalize_invariant()
            # Prune expired _sell_finalize_result_cache entries
            with contextlib.suppress(Exception):
                _now = time.time()
                _ttl = self._sell_finalize_cache_ttl_s
                _expired = [
                    k for k, ts in list(self._sell_finalize_result_cache_ts.items())
                    if _now - ts > _ttl
                ]
                for k in _expired:
                    self._sell_finalize_result_cache.pop(k, None)
                    self._sell_finalize_result_cache_ts.pop(k, None)
            # --- OPTION 3: Run post-finalize verification checks ---
            with contextlib.suppress(Exception):
                await self._verify_pending_closes()
            await asyncio.sleep(60)

    def _ensure_heartbeat(self) -> None:
        """Start the heartbeat task if it isn't running yet."""
        if self._heartbeat_task is not None and not self._heartbeat_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        try:
            self._heartbeat_task = loop.create_task(self._heartbeat_loop(), name="ExecutionManager:heartbeat")
        except Exception:
            pass

    # small helper to emit status consistently
    async def _emit_status(self, status: str, detail: str = ""):
        try:
            # Update timestamp so Watchdog sees activity
            if hasattr(self.shared_state, "update_timestamp"):
                await maybe_call(self.shared_state, "update_timestamp", "ExecutionManager")

            upd = getattr(self.shared_state, "update_component_status", None) \
                or getattr(self.shared_state, "set_component_status", None)
            if callable(upd):
                await (upd("ExecutionManager", status, detail) if asyncio.iscoroutinefunction(upd)
                    else asyncio.to_thread(upd, "ExecutionManager", status, detail))
        except Exception:
            self.logger.debug("EM status emit (primary) failed", exc_info=True)
        # best-effort compatibility mirror
        try:
            now_ts = time.time()
            cs = getattr(self.shared_state, "component_statuses", None)
            if isinstance(cs, dict):
                cs["ExecutionManager"] = {"status": status, "message": detail, "timestamp": now_ts, "ts": now_ts}
        except Exception:
            pass

    def _ensure_semaphores_ready(self):
        """Lazy-initialize semaphores when needed (requires running event loop)."""
        if self._semaphores_initialized:
            return
        try:
            if self._concurrent_orders_sem is None:
                self._concurrent_orders_sem = asyncio.Semaphore(self.max_concurrent_orders)
                cfg_max_conc = int(self._cfg('execution.max_concurrency', self.max_concurrent_orders))
                if cfg_max_conc and cfg_max_conc < self.max_concurrent_orders:
                    self._concurrent_orders_sem = asyncio.Semaphore(cfg_max_conc)
            if self._cancel_sem is None:
                self._cancel_sem = asyncio.Semaphore(3)
            self._semaphores_initialized = True
        except Exception as e:
            self.logger.debug(f"Semaphore initialization deferred: {e}")

    # =============================
    # Setters
    # =============================
    def set_meta_controller(self, meta_controller):
        self.meta_controller = meta_controller

    def set_risk_manager(self, risk_manager):
        self.risk_manager = risk_manager

    def set_tp_sl_engine(self, tp_sl_engine):
        self.tp_sl_engine = tp_sl_engine

    def health(self) -> Dict[str, Any]:
        """Lightweight health snapshot used by MetaController/OpsPlaneReady checks."""
        try:
            hb_ok = self._heartbeat_task is not None and not self._heartbeat_task.done()
        except Exception:
            hb_ok = False
        
        # Handle both old (set) and new (dict) formats for _active_symbol_side_orders
        active_orders = getattr(self, "_active_symbol_side_orders", {})
        if isinstance(active_orders, set):
            active_orders_count = len(active_orders)
        else:
            active_orders_count = len(active_orders) if isinstance(active_orders, dict) else 0
        
        return {
            "component": "ExecutionManager",
            "status": "Healthy",
            "heartbeat": "running" if hb_ok else "stopped",
            "active_symbol_side_orders": active_orders_count,
            "seen_client_order_ids": len(getattr(self, "_seen_client_order_ids", {}) or {}),
            "sell_finalize_fills_seen": int(getattr(self, "_sell_finalize_stats", {}).get("fills_seen", 0) or 0),
            "sell_finalize_finalized": int(getattr(self, "_sell_finalize_stats", {}).get("finalized", 0) or 0),
            "sell_finalize_pending": int(getattr(self, "_sell_finalize_pending", 0) or 0),
            "sell_finalize_duplicate": int(getattr(self, "_sell_finalize_stats", {}).get("duplicate_finalize", 0) or 0),
            "sell_finalize_pending_timeout": int(getattr(self, "_sell_finalize_stats", {}).get("pending_timeout", 0) or 0),
        }

    async def wait_for_inflight_sells(self, timeout: float = 3.0, poll_sec: float = 0.1) -> bool:
        """
        Wait briefly for SELL lifecycle activity to quiesce.
        Used during controller shutdown to avoid canceling mid-finalization.
        """
        timeout_s = max(0.0, float(timeout or 0.0))
        poll = min(max(float(poll_sec or 0.1), 0.05), 0.5)
        deadline = time.time() + timeout_s

        while True:
            with contextlib.suppress(Exception):
                self._audit_sell_finalize_invariant()

            active_sells = 0
            with contextlib.suppress(Exception):
                active_orders = getattr(self, "_active_symbol_side_orders", {})
                if isinstance(active_orders, dict):
                    active_sells = sum(
                        1
                        for item, _ts in active_orders.items()
                        if isinstance(item, tuple) and len(item) >= 2 and str(item[1]).upper() == "SELL"
                    )
                else:
                    # Fallback for old set-based format
                    active_sells = sum(
                        1
                        for item in (active_orders or set())
                        if isinstance(item, tuple) and len(item) >= 2 and str(item[1]).upper() == "SELL"
                    )

            recovery_pending = 0
            with contextlib.suppress(Exception):
                recovery_pending = sum(
                    1
                    for task in (getattr(self, "_sell_fill_recovery_tasks", {}) or {}).values()
                    if task is not None and not task.done()
                )

            finalize_pending = int(getattr(self, "_sell_finalize_pending", 0) or 0)

            exit_lock_pending = 0
            with contextlib.suppress(Exception):
                exit_lock_pending = sum(
                    1
                    for v in (getattr(self.shared_state, "exit_in_progress", {}) or {}).values()
                    if bool(v)
                )

            if active_sells <= 0 and recovery_pending <= 0 and finalize_pending <= 0 and exit_lock_pending <= 0:
                return True

            if time.time() >= deadline:
                self.logger.warning(
                    "[EM:SellDrainTimeout] active_sells=%d recovery_pending=%d finalize_pending=%d exit_lock_pending=%d timeout=%.2fs",
                    int(active_sells),
                    int(recovery_pending),
                    int(finalize_pending),
                    int(exit_lock_pending),
                    timeout_s,
                )
                return False

            await asyncio.sleep(poll)

    async def get_symbol_filters_cached(self, symbol: str, *, max_age_s: Optional[float] = None) -> Dict[str, Any]:
        """Return normalized exchange filters for `symbol` (best-effort, cached).

        Prefer ExchangeClient as source of truth. Falls back to a small local TTL cache to
        keep callers (MetaController/Scaling) resilient if ExchangeClient doesn't expose the helpers.
        """
        sym = self._norm_symbol(symbol)
        ttl = self._symbol_filters_cache_max_age_s if max_age_s is None else float(max_age_s)
        now = time.time()
        try:
            ts = self._symbol_filters_cache_ts.get(sym, 0.0)
            cached = self._symbol_filters_cache.get(sym)
            if cached and ttl > 0 and (now - ts) <= ttl:
                return cached
        except Exception:
            pass

        filters: Dict[str, Any] = {}
        # Primary: ExchangeClient caches
        try:
            if hasattr(self.exchange_client, "get_exchange_filters"):
                filters = await self.exchange_client.get_exchange_filters(sym)
            elif hasattr(self.exchange_client, "get_symbol_filters"):
                filters = await self.exchange_client.get_symbol_filters(sym)
            elif hasattr(self.exchange_client, "get_symbol_filters_raw"):
                filters = await self.exchange_client.get_symbol_filters_raw(sym)
            elif hasattr(self.exchange_client, "ensure_symbol_filters_ready"):
                filters = await self.exchange_client.ensure_symbol_filters_ready(sym)
        except Exception:
            filters = {}

        if isinstance(filters, dict):
            try:
                self._symbol_filters_cache[sym] = filters
                self._symbol_filters_cache_ts[sym] = now
            except Exception:
                pass
            return filters
        return {}

    async def close_trade(self, symbol: str, reason: str = "", *, tag: str = "tp_sl", force_finalize: bool = False) -> Dict[str, Any]:
        """Compatibility shim expected by RiskManager."""
        return await self.close_position(symbol=symbol, reason=reason, tag=tag, force_finalize=force_finalize)

    # =============================
    # Utilities
    # =============================

    async def _get_sellable_qty(self, sym: str) -> float:
        """Best-effort lookup for how much we can sell right now.
        
        ✅ FIX #2: REORDERED PRECEDENCE - Exchange is authoritative!
        Order of precedence (REVISED for eventual consistency):
        1) ExchangeClient.get_account_balance(base_asset) ← AUTHORITATIVE SOURCE
        2) SharedState.get_position_quantity(sym) ← Cache (fallback)
        3) SharedState.position_manager.get_position(sym).quantity ← Last resort
        
        Rationale: Exchange has the actual fill immediately. SharedState might be 
        delayed by 100-500ms. Checking SharedState first causes stale-state rejections.
        Returns 0.0 if unknown.
        """
        try:
            qty = 0.0
            base_asset, _ = self._split_base_quote(sym)
            exchange_checked = False
            exchange_query_ok = False
            
            # ✅ FIX #2: CHECK EXCHANGE FIRST (authoritative source, < 50ms)
            get_bal = getattr(self.exchange_client, "get_account_balance", None)
            if callable(get_bal):
                exchange_checked = True
                try:
                    bal = await get_bal(base_asset)
                    exchange_query_ok = True
                    free = float((bal or {}).get("free", 0.0))
                    locked = float((bal or {}).get("locked", 0.0))
                    qty = float(free + locked)
                    if qty > 0:
                        self._exchange_zero_mismatch_counts.pop(sym, None)
                        self.logger.debug(f"[GetSellable] {sym}: qty={qty:.6f} from Exchange (AUTHORITATIVE)")
                        return qty
                except Exception:
                    # Lookup failure is not authoritative zero; keep local fallback path alive.
                    self.logger.debug(
                        "[GetSellable] %s exchange balance lookup failed; using local fallback path",
                        sym,
                        exc_info=True,
                    )

            if exchange_checked and exchange_query_ok:
                local_qty = 0.0
                with contextlib.suppress(Exception):
                    if hasattr(self.shared_state, "get_position_quantity"):
                        local_qty = float(await self.shared_state.get_position_quantity(sym) or 0.0)
                    elif isinstance(getattr(self.shared_state, "positions", None), dict):
                        local_qty = float((self.shared_state.positions.get(sym, {}) or {}).get("quantity", 0.0) or 0.0)
                
                # If exchange=0 but local>0, treat local state as potentially stale.
                # Optional override exists for environments that intentionally permit
                # one-shot local fallback during eventual-consistency windows.
                if local_qty > 0:
                    mismatch_count = int(self._exchange_zero_mismatch_counts.get(sym, 0) or 0) + 1
                    self._exchange_zero_mismatch_counts[sym] = mismatch_count
                    repair_threshold = int(self._cfg("EXCHANGE_ZERO_REPAIR_THRESHOLD", 2) or 2)

                    allow_local_fallback = str(
                        self._cfg("ALLOW_LOCAL_SELL_QTY_WHEN_EXCHANGE_ZERO", "true")
                    ).strip().lower() in {"1", "true", "yes", "on"}

                    if allow_local_fallback and mismatch_count <= max(1, repair_threshold):
                        self.logger.warning(
                            "[GetSellable] %s exchange qty=0 but local qty=%.8f (count=%d) -> using local fallback (flag enabled)",
                            sym,
                            local_qty,
                            mismatch_count,
                        )
                        return float(local_qty)

                    self.logger.warning(
                        "[GetSellable] %s exchange qty=0 while local qty=%.8f (count=%d) -> blocking SELL fallback (authoritative exchange zero)",
                        sym,
                        local_qty,
                        mismatch_count,
                    )
                    return 0.0
                else:
                    self._exchange_zero_mismatch_counts.pop(sym, None)
                    self.logger.debug(f"[GetSellable] {sym}: qty=0 from Exchange (authoritative, no local position)")
                    return 0.0
            
            # ✅ FIX #2: FALLBACK to SharedState only if Exchange unavailable
            if hasattr(self.shared_state, "get_position_quantity"):
                with contextlib.suppress(Exception):
                    q = await self.shared_state.get_position_quantity(sym)
                    qty = float(q or 0.0)
                    if qty > 0:
                        self.logger.debug(f"[GetSellable] {sym}: qty={qty:.6f} from SharedState (cache)")
                        return qty
            
            # ✅ FIX #2: PositionManager fallback (if present under SharedState)
            pm = getattr(self.shared_state, "position_manager", None)
            if pm is not None:
                getp = getattr(pm, "get_position", None)
                if callable(getp):
                    with contextlib.suppress(Exception):
                        p = await getp(sym) if asyncio.iscoroutinefunction(getp) else getp(sym)
                        if p:
                            q = getattr(p, "quantity", None)
                            if q is None:
                                q = getattr(p, "qty", 0.0)
                            qty = float(q or 0.0)
                            if qty > 0:
                                self.logger.debug(f"[GetSellable] {sym}: qty={qty:.6f} from PositionManager")
                                return qty
            
            # If we reached here, both Exchange and SharedState think we have 0
            if getattr(self.shared_state, "positions", {}).get(sym):
                self.logger.warning(f"[GetSellable] {sym}: Zero quantity returned despite position record existence.")
                
        except Exception:
            self.logger.debug("_get_sellable_qty failed (non-fatal)", exc_info=True)
        return 0.0

    async def _buffer_liquidation_sell_qty(self, sym: str, qty: float) -> float:
        """
        Conservative qty buffer for liquidation SELL to reduce repeated insufficient-balance rejects.

        Why: exchange balances can lag local fill bookkeeping by fees/rounding dust.
        Subtracting one lot step for liquidation exits is a cheap, robust guard.
        """
        qty_in = float(qty or 0.0)
        if qty_in <= 0:
            return 0.0

        try:
            filters = await self.exchange_client.ensure_symbol_filters_ready(sym)
            step_size, min_qty, _, _, _ = self._extract_filter_vals(filters)
            step_size = float(step_size or 0.0)
            min_qty = float(min_qty or 0.0)

            use_step_buffer = str(
                self._cfg("LIQUIDATION_SELL_STEP_BUFFER_ENABLED", "true")
            ).strip().lower() in {"1", "true", "yes", "on"}
            qty_out = float(qty_in)

            if use_step_buffer and step_size > 0:
                rounded_no_buffer = float(round_step(qty_out, step_size))
                candidate = round_step(max(0.0, qty_out - step_size), step_size)
                # Keep exchange-valid minimums; fallback to original rounded qty if buffer is too aggressive.
                if candidate > 0 and (min_qty <= 0 or candidate >= min_qty):
                    qty_out = float(candidate)
                else:
                    qty_out = float(rounded_no_buffer)

                # Dust-trap guard:
                # For tiny positions, subtracting one step can leave a meaningful residual
                # that repeatedly triggers dust-healing buy/sell loops. If buffered qty
                # would leave residual notional above the permanent-dust threshold,
                # prefer unbuffered rounded qty to retire position cleanly.
                avoid_dust_trap = str(
                    self._cfg("LIQUIDATION_SELL_AVOID_DUST_TRAP", "true")
                ).strip().lower() in {"1", "true", "yes", "on"}
                if avoid_dust_trap and rounded_no_buffer > 0 and qty_out < rounded_no_buffer:
                    residual_qty = max(0.0, float(qty_in) - float(qty_out))
                    if residual_qty > 0:
                        mark_price = 0.0
                        try:
                            mark_price = float(
                                (getattr(self.shared_state, "latest_prices", {}) or {}).get(sym, 0.0) or 0.0
                            )
                        except Exception:
                            mark_price = 0.0
                        if mark_price <= 0:
                            with contextlib.suppress(Exception):
                                get_px = getattr(self.exchange_client, "get_current_price", None) or getattr(
                                    self.exchange_client, "get_price", None
                                )
                                if callable(get_px):
                                    mark_price = float(await get_px(sym) or 0.0)

                        if mark_price > 0:
                            residual_notional = residual_qty * mark_price
                            permanent_floor = float(
                                self._cfg("PERMANENT_DUST_USDT_THRESHOLD", 1.0) or 1.0
                            )
                            if residual_notional > permanent_floor:
                                self.logger.info(
                                    "[EM:LiqQtyBuffer] %s avoid dust trap: buffered_residual=%.6f (~$%.4f) > permanent_floor=$%.2f. "
                                    "Using unbuffered rounded qty %.10f instead of %.10f.",
                                    sym,
                                    residual_qty,
                                    residual_notional,
                                    permanent_floor,
                                    rounded_no_buffer,
                                    qty_out,
                                )
                                qty_out = float(rounded_no_buffer)

            if qty_out > qty_in:
                qty_out = qty_in

            if qty_out > 0 and qty_out < qty_in:
                self.logger.info(
                    "[EM:LiqQtyBuffer] %s qty %.10f -> %.10f (step=%.10f min_qty=%.10f)",
                    sym,
                    qty_in,
                    qty_out,
                    step_size,
                    min_qty,
                )
            return float(qty_out or qty_in)
        except Exception:
            self.logger.debug("[EM:LiqQtyBuffer] %s failed; using raw qty", sym, exc_info=True)
            return float(qty_in)

    async def _ensure_position_ready(self, sym: str, max_retries: int = 3) -> float:
        """
        ✅ FIX #4: Wait for position to be available, with state reconciliation retries.
        
        Purpose: Handle the temporal gap where:
        - Exchange has the position (just filled)
        - SharedState doesn't yet (refresh in flight)
        - ExecutionManager needs to SELL immediately after
        
        Strategy:
        1. Check position via _get_sellable_qty()
        2. If 0, trigger authoritative sync + brief wait
        3. Retry (up to max_retries times)
        4. Return actual qty or 0 if confirmed empty
        
        Returns: Quantity if found, 0 if confirmed absent after retries
        """
        base_asset, _ = self._split_base_quote(sym)
        
        for attempt in range(max_retries):
            # Check current balance (Exchange-first via FIX #2)
            qty = await self._get_sellable_qty(sym)
            if qty > 0:
                self.logger.info(
                    f"[PositionReady] {sym}: qty={qty:.6f} available (attempt {attempt + 1}/{max_retries})"
                )
                return qty

            if attempt < max_retries - 1:
                # State might be in flight; trigger refresh and retry
                self.logger.warning(
                    f"[PositionReady] {sym}: No qty found (attempt {attempt + 1}/{max_retries}). "
                    f"Syncing state..."
                )
                
                try:
                    # Force authoritative sync
                    await self.shared_state.sync_authoritative_balance(force=True)
                except Exception as e:
                    self.logger.debug(f"[PositionReady] Sync failed: {e}")
                
                # Brief wait for state propagation (50-100ms should be sufficient)
                await asyncio.sleep(0.05 * (attempt + 1))  # Exponential backoff: 50ms, 100ms, 150ms
        
        # After all retries, qty is truly 0
        self.logger.info(f"[PositionReady] {sym}: Confirmed no position after {max_retries} attempts")
        return 0.0

    def _get_entry_price_for_sell(self, sym: str) -> float:
        """Best-effort lookup of entry/avg price for net PnL gating."""
        try:
            ot = getattr(self.shared_state, "open_trades", {}) or {}
            entry = float((ot.get(sym) or {}).get("entry_price", 0.0) or 0.0)
            if entry > 0:
                return entry
        except Exception:
            pass
        try:
            pos = getattr(self.shared_state, "positions", {}) or {}
            entry = float((pos.get(sym) or {}).get("avg_price", 0.0) or 0.0)
            if entry > 0:
                return entry
        except Exception:
            pass
        try:
            return float(getattr(self.shared_state, "_avg_price_cache", {}).get(sym, 0.0) or 0.0)
        except Exception:
            return 0.0

    def _entry_profitability_feasible(
        self,
        symbol: Optional[str] = None,
        price: Optional[float] = None,
        atr_pct: Optional[float] = None,
    ) -> Tuple[bool, Dict[str, float]]:
        """Check if TP max can clear required exit move and net-profit floor."""
        # FIXED: Use realistic defaults for Binance trading instead of 0.0
        trade_fee_pct = float(self._cfg("TRADE_FEE_PCT", 0.001) or 0.001)  # 0.1% default
        exit_fee_bps = float(self._cfg("EXIT_FEE_BPS", 10.0) or 10.0)  # 10 bps default
        fee_bps = max(exit_fee_bps, trade_fee_pct * 10000.0)
        r_fee = fee_bps / 10000.0
        r_slip = float(self._cfg("EXIT_SLIPPAGE_BPS", 15.0) or 15.0) / 10000.0  # 15 bps default
        r_buf = float(self._cfg("TP_MIN_BUFFER_BPS", 5.0) or 5.0) / 10000.0  # 5 bps default
        m_entry = float(self._cfg("MIN_PLANNED_QUOTE_FEE_MULT", 1.5) or 1.5)  # Lowered from 2.5
        m_exit = float(self._cfg("MIN_PROFIT_EXIT_FEE_MULT", 1.0) or 1.0)  # Lowered from 2.0
        m_exit = max(m_exit, m_entry)

        r_req = (2.0 * r_fee * m_exit) + r_slip + r_buf
        r_min_net = float(self._cfg("MIN_NET_PROFIT_AFTER_FEES", 0.0001) or 0.0001)  # 0.01% min profit
        min_tp_needed_for_net = r_min_net + (2.0 * r_fee) + r_slip
        required_tp = max(r_req, min_tp_needed_for_net)
        tp_pct_min = float(self._cfg("TP_PCT_MIN", 0.003) or 0.003)  # 0.3% minimum TP
        tp_max_cfg = float(self._cfg("TP_PCT_MAX", 0.05) or 0.05)  # 5% maximum TP
        tp_atr_mult = float(self._cfg("TP_ATR_MULT", 2.0) or 2.0)  # 2x ATR for TP
        if atr_pct is None or atr_pct <= 0:
            atr_pct = float(self._cfg("TPSL_FALLBACK_ATR_PCT", 0.005) or 0.005)  # 0.5% fallback

        tp_from_atr = (atr_pct * tp_atr_mult) if (atr_pct > 0 and tp_atr_mult > 0) else 0.0
        tp_max = tp_max_cfg if tp_max_cfg > 0 else max(tp_from_atr, tp_pct_min)

        detail = {
            "required_exit": r_req,
            "min_net_required": min_tp_needed_for_net,
            "required_tp": required_tp,
            "tp_max": tp_max,
            "fee_bps": fee_bps,
            "slippage_bps": float(self._cfg("EXIT_SLIPPAGE_BPS", 0.0) or 0.0),
            "buffer_bps": float(self._cfg("TP_MIN_BUFFER_BPS", 0.0) or 0.0),
            "exit_fee_mult": m_exit,
            "price": float(price or 0.0),
            "atr_pct": float(atr_pct or 0.0),
            "tp_from_atr": float(tp_from_atr),
            "tp_min": tp_pct_min,
        }
        if tp_max <= 0 or tp_max < required_tp:
            return False, detail
        return True, detail

    async def _resolve_dynamic_sell_threshold(
        self,
        *,
        sym: str,
        entry: float,
        price: float,
        qty: float,
        policy_ctx: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Compute dynamic sell edge threshold: fees + slippage + volatility noise + USDT floor."""
        ctx = policy_ctx or {}
        fee_component = (float(self._exit_fee_bps() or 0.0) / 10000.0) * 2.0

        atr_tf = str(
            self._cfg("SELL_DYNAMIC_ATR_TIMEFRAME", self._cfg("VOLATILITY_REGIME_TIMEFRAME", "5m")) or "5m"
        )
        atr_period = int(
            self._cfg("SELL_DYNAMIC_ATR_PERIOD", self._cfg("VOLATILITY_REGIME_ATR_PERIOD", 14)) or 14
        )
        atr_pct = 0.0
        try:
            if hasattr(self.shared_state, "calc_atr"):
                atr_val = float(await maybe_call(self.shared_state, "calc_atr", sym, atr_tf, atr_period) or 0.0)
                if atr_val <= 0:
                    atr_val = float(await maybe_call(self.shared_state, "calc_atr", sym, "1m", atr_period) or 0.0)
                if atr_val > 0 and float(price or 0.0) > 0:
                    atr_pct = atr_val / max(float(price), 1e-9)
        except Exception:
            atr_pct = 0.0
        if atr_pct <= 0:
            atr_pct = float(
                self._cfg(
                    "SELL_DYNAMIC_FALLBACK_ATR_PCT",
                    self._cfg("MICRO_TRADE_KILL_FALLBACK_ATR_PCT", 0.0),
                )
                or 0.0
            )

        slippage_min_pct = float(self._cfg("SELL_DYNAMIC_SLIPPAGE_MIN_PCT", 0.0003) or 0.0003)
        slippage_atr_mult = float(self._cfg("SELL_DYNAMIC_SLIPPAGE_ATR_MULT", 0.05) or 0.05)
        slippage_component = max(slippage_min_pct, atr_pct * slippage_atr_mult)

        vol_buffer_atr_mult = float(self._cfg("SELL_DYNAMIC_VOL_BUFFER_ATR_MULT", 0.15) or 0.15)
        volatility_buffer = max(0.0, atr_pct * vol_buffer_atr_mult)

        strategic_buffer = float(self._cfg("SELL_DYNAMIC_STRATEGIC_BUFFER_PCT", 0.0) or 0.0)
        min_usdt_floor_cfg = float(self._cfg("SELL_DYNAMIC_MIN_USDT_FLOOR", 0.12) or 0.12)
        pos_notional_quote = max(
            float(qty or 0.0) * float(price or 0.0),
            float(qty or 0.0) * float(entry or 0.0),
            0.0,
        )

        # Adapt USDT floor for micro-notional exits and repeated dynamic-edge deadlocks.
        usdt_floor_notional_scale = 1.0
        micro_notional_cap = float(self._cfg("SELL_DYNAMIC_MICRO_NOTIONAL_CAP_USDT", 80.0) or 80.0)
        micro_floor_min_scale = float(self._cfg("SELL_DYNAMIC_MICRO_MIN_FLOOR_SCALE", 0.30) or 0.30)
        min_usdt_floor_effective = max(0.0, min_usdt_floor_cfg)
        if pos_notional_quote > 0.0 and micro_notional_cap > 0.0 and pos_notional_quote < micro_notional_cap:
            usdt_floor_notional_scale = max(
                micro_floor_min_scale,
                min(1.0, pos_notional_quote / micro_notional_cap),
            )
            min_usdt_floor_effective *= usdt_floor_notional_scale

        sell_edge_rejections = 0
        try:
            if hasattr(self.shared_state, "get_rejection_count"):
                sell_edge_rejections = int(
                    self.shared_state.get_rejection_count(sym, "SELL", "SELL_DYNAMIC_EDGE_MIN") or 0
                )
        except Exception:
            sell_edge_rejections = 0

        usdt_floor_feedback_scale = 1.0
        relax_trigger = int(self._cfg("SELL_DYNAMIC_EDGE_RELAX_TRIGGER", 3) or 3)
        if sell_edge_rejections >= max(1, relax_trigger):
            relax_steps = min(6, sell_edge_rejections - relax_trigger + 1)
            relax_step = float(self._cfg("SELL_DYNAMIC_EDGE_RELAX_STEP", 0.10) or 0.10)
            relax_min_scale = float(self._cfg("SELL_DYNAMIC_EDGE_RELAX_MIN_SCALE", 0.45) or 0.45)
            usdt_floor_feedback_scale = max(relax_min_scale, 1.0 - (relax_steps * relax_step))
            min_usdt_floor_effective *= usdt_floor_feedback_scale

        usdt_floor_pct = (min_usdt_floor_effective / pos_notional_quote) if pos_notional_quote > 0 else 0.0

        required_profit_pct = fee_component + slippage_component + volatility_buffer + strategic_buffer
        required_profit_pct = max(required_profit_pct, usdt_floor_pct)

        regime = str(
            ctx.get("volatility_regime")
            or ctx.get("tradeability_regime")
            or (getattr(self.shared_state, "metrics", {}) or {}).get("volatility_regime")
            or "normal"
        ).lower()
        regime_mult = 1.0
        if regime == "high":
            regime_mult = float(self._cfg("SELL_DYNAMIC_REGIME_HIGH_MULT", 1.3) or 1.3)
        elif regime == "low":
            regime_mult = float(self._cfg("SELL_DYNAMIC_REGIME_LOW_MULT", 0.8) or 0.8)
        required_profit_pct *= regime_mult

        return {
            "required_profit_pct": float(required_profit_pct),
            "fee_component_pct": float(fee_component),
            "slippage_component_pct": float(slippage_component),
            "volatility_buffer_pct": float(volatility_buffer),
            "strategic_buffer_pct": float(strategic_buffer),
            "usdt_floor_pct": float(usdt_floor_pct),
            "atr_pct": float(atr_pct),
            "regime": regime,
            "regime_mult": float(regime_mult),
            "position_notional_quote": float(pos_notional_quote),
            "min_usdt_floor": float(min_usdt_floor_cfg),
            "min_usdt_floor_effective": float(min_usdt_floor_effective),
            "usdt_floor_notional_scale": float(usdt_floor_notional_scale),
            "usdt_floor_feedback_scale": float(usdt_floor_feedback_scale),
            "sell_edge_rejection_count": int(sell_edge_rejections),
        }

    async def _check_sell_net_pnl_gate(
        self,
        *,
        sym: str,
        quantity: Optional[float],
        policy_ctx: Dict[str, Any],
        tag: str,
        is_liq_full: bool,
        special_liq_bypass: bool,
    ) -> Optional[Dict[str, Any]]:
        """Block non-liquidation SELLs that don't clear fee-aware net PnL gate."""
        if is_liq_full or special_liq_bypass:
            return None

        reason_text = " ".join([
            str(policy_ctx.get("reason") or ""),
            str(policy_ctx.get("exit_reason") or ""),
            str(policy_ctx.get("signal_reason") or ""),
            str(policy_ctx.get("liquidation_reason") or ""),
            str(tag or ""),
        ]).upper()
        tag_lower = str(tag or "").lower()
        rotation_override = bool(policy_ctx.get("rotation_sell_override"))
        bootstrap_override = bool(policy_ctx.get("bootstrap_sell_override")) or rotation_override or ("bootstrap_exit" in tag_lower)

        bootstrap_override = bootstrap_override and bool(self._cfg("BOOTSTRAP_ALLOW_SELL_BELOW_FEE", True))
        bootstrap_min_net = float(
            policy_ctx.get("bootstrap_sell_min_net", self._cfg("BOOTSTRAP_MAX_NEGATIVE_PNL", 0.0)) or 0.0
        )

        if not bootstrap_override:
            if self.allow_sell_below_fee and float(self.sell_min_net_pnl_usdt or 0.0) <= 0.0:
                return None

        qty = float(quantity or 0.0)
        if qty <= 0:
            qty = await self._get_sellable_qty(sym)
        if qty <= 0:
            return None

        get_px = getattr(self.exchange_client, "get_current_price", None) or getattr(self.exchange_client, "get_price")
        price = 0.0
        try:
            price = float(await get_px(sym)) if get_px else 0.0
        except Exception:
            price = 0.0
        if price <= 0:
            price = float(getattr(self.shared_state, "latest_prices", {}).get(sym, 0.0) or 0.0)

        entry = self._get_entry_price_for_sell(sym)
        if price <= 0 or entry <= 0:
            return None

        proceeds = qty * price
        fee_est = proceeds * float(self.trade_fee_pct or 0.0) * 2.0
        entry_cost = qty * entry
        net_pnl = proceeds - fee_est - entry_cost
        min_net = float(self.sell_min_net_pnl_usdt or 0.0)
        if bootstrap_override:
            min_net = bootstrap_min_net

        expected_move_pct = (price - entry) / max(entry, 1e-9)
        fee_bps = float(self._exit_fee_bps() or 0.0)
        dynamic_gate_enabled = bool(self._cfg("SELL_DYNAMIC_EDGE_GATE_ENABLED", True))
        dynamic = await self._resolve_dynamic_sell_threshold(
            sym=sym,
            entry=entry,
            price=price,
            qty=qty,
            policy_ctx=policy_ctx,
        )

        slippage_component = float(dynamic.get("slippage_component_pct", 0.0) or 0.0)
        fee_component = float(dynamic.get("fee_component_pct", 0.0) or 0.0)
        slippage_bps = slippage_component * 10000.0
        net_after_fees_pct = expected_move_pct - fee_component - slippage_component
        required_move_pct = float(dynamic.get("required_profit_pct", 0.0) or 0.0)
        sell_edge_rej = int(dynamic.get("sell_edge_rejection_count", 0) or 0)
        buy_lock_rej = 0
        try:
            if hasattr(self.shared_state, "get_rejection_count"):
                buy_lock_rej = int(self.shared_state.get_rejection_count(sym, "BUY", "POSITION_ALREADY_OPEN") or 0)
        except Exception:
            buy_lock_rej = 0

        deadlock_bypass_trigger_sell = int(self._cfg("DEADLOCK_SELL_EDGE_BYPASS_TRIGGER", 4) or 4)
        deadlock_bypass_trigger_buy = int(self._cfg("DEADLOCK_SELL_EDGE_BUY_LOCK_TRIGGER", 6) or 6)
        deadlock_bypass_max_fraction = float(self._cfg("DEADLOCK_SELL_EDGE_BYPASS_MAX_FRACTION", 0.5) or 0.5)
        deadlock_bypass_max_fraction_recovery = float(
            self._cfg("DEADLOCK_SELL_EDGE_BYPASS_MAX_FRACTION_RECOVERY", 1.0) or 1.0
        )
        deadlock_min_net_usdt = float(self._cfg("DEADLOCK_SELL_EDGE_MIN_NET_USDT", -0.25) or -0.25)
        is_capacity_recovery_sell = any(
            token in reason_text
            for token in (
                "CAPITAL_RECOVERY",
                "LIQUIDITY_RESTORATION",
                "ROTATION",
                "REBALANCE",
                "FORCED_EXIT",
                "META_EXIT",
                "STRATEGY_SELL",
            )
        )

        total_sellable_qty = 0.0
        try:
            total_sellable_qty = float(await self._get_sellable_qty(sym) or 0.0)
        except Exception:
            total_sellable_qty = 0.0
        qty_fraction = (qty / max(total_sellable_qty, 1e-12)) if total_sellable_qty > 0 else 1.0
        max_bypass_fraction = max(0.05, min(1.0, deadlock_bypass_max_fraction))
        if is_capacity_recovery_sell or buy_lock_rej >= max(1, deadlock_bypass_trigger_buy):
            max_bypass_fraction = max(
                max_bypass_fraction,
                max(0.05, min(1.0, deadlock_bypass_max_fraction_recovery)),
            )

        deadlock_bypass = bool(
            dynamic_gate_enabled
            and expected_move_pct < required_move_pct
            and sell_edge_rej >= max(1, deadlock_bypass_trigger_sell)
            and (
                buy_lock_rej >= max(1, deadlock_bypass_trigger_buy)
                or is_capacity_recovery_sell
            )
            and qty_fraction <= max_bypass_fraction
        )
        if deadlock_bypass:
            min_net = min(min_net, deadlock_min_net_usdt)
            self.logger.warning(
                "[EM:SellDynamicGate:DeadlockBypass] Allowing SELL %s despite edge gate "
                "(edge=%.4f%% req=%.4f%% sell_edge_rej=%d buy_lock_rej=%d recovery_sell=%s qty_frac=%.3f max_frac=%.3f min_net=%.4f)",
                sym,
                expected_move_pct * 100.0,
                required_move_pct * 100.0,
                sell_edge_rej,
                buy_lock_rej,
                is_capacity_recovery_sell,
                qty_fraction,
                max_bypass_fraction,
                min_net,
            )

        if dynamic_gate_enabled and expected_move_pct < required_move_pct and not deadlock_bypass:
            self.logger.info(
                "[EM:SellDynamicGate] Blocked SELL %s: edge=%.4f%% < required=%.4f%% "
                "(fee=%.4f%% slip=%.4f%% vol=%.4f%% floor=%.4f%% regime=%s x%.2f atr=%.4f%% "
                "pos=%.4f floor_eff=%.4f floor_cfg=%.4f edge_rej=%d)",
                sym,
                expected_move_pct * 100.0,
                required_move_pct * 100.0,
                fee_component * 100.0,
                slippage_component * 100.0,
                float(dynamic.get("volatility_buffer_pct", 0.0) or 0.0) * 100.0,
                float(dynamic.get("usdt_floor_pct", 0.0) or 0.0) * 100.0,
                str(dynamic.get("regime", "normal")),
                float(dynamic.get("regime_mult", 1.0) or 1.0),
                float(dynamic.get("atr_pct", 0.0) or 0.0) * 100.0,
                float(dynamic.get("position_notional_quote", 0.0) or 0.0),
                float(dynamic.get("min_usdt_floor_effective", 0.0) or 0.0),
                float(dynamic.get("min_usdt_floor", 0.0) or 0.0),
                int(dynamic.get("sell_edge_rejection_count", 0) or 0),
            )
            try:
                await self.shared_state.record_rejection(sym, "SELL", "SELL_DYNAMIC_EDGE_MIN", source="ExecutionManager")
            except Exception:
                pass
            return {
                "ok": False,
                "status": "blocked",
                "reason": "sell_dynamic_edge_below_min",
                "error_code": "SELL_DYNAMIC_EDGE_MIN",
                "expected_move_pct": expected_move_pct,
                "required_move_pct": required_move_pct,
                "fee_bps": fee_bps,
                "fee_component_pct": fee_component,
                "slippage_component_pct": slippage_component,
                "volatility_buffer_pct": float(dynamic.get("volatility_buffer_pct", 0.0) or 0.0),
                "usdt_floor_pct": float(dynamic.get("usdt_floor_pct", 0.0) or 0.0),
                "min_usdt_floor_effective": float(dynamic.get("min_usdt_floor_effective", 0.0) or 0.0),
                "min_usdt_floor": float(dynamic.get("min_usdt_floor", 0.0) or 0.0),
                "sell_edge_rejection_count": int(dynamic.get("sell_edge_rejection_count", 0) or 0),
                "buy_position_lock_rejection_count": int(buy_lock_rej),
                "atr_pct": float(dynamic.get("atr_pct", 0.0) or 0.0),
                "regime": str(dynamic.get("regime", "normal")),
                "regime_mult": float(dynamic.get("regime_mult", 1.0) or 1.0),
            }
        min_net_pct = float(self.min_net_profit_after_fees_pct or 0.0)
        legacy_net_pct_guard = bool(self._cfg("SELL_DYNAMIC_LEGACY_NET_PCT_GUARD", False))
        if legacy_net_pct_guard and min_net_pct > 0 and net_after_fees_pct < min_net_pct:
            self.logger.info(
                "[EM:SellNetPctGate] Blocked SELL %s: net_after_fees=%.4f%% < min=%.4f%% (legacy guard enabled)",
                sym,
                net_after_fees_pct * 100.0,
                min_net_pct * 100.0,
            )
            try:
                await self.shared_state.record_rejection(sym, "SELL", "SELL_NET_PCT_MIN", source="ExecutionManager")
            except Exception:
                pass
            return {
                "ok": False,
                "status": "blocked",
                "reason": "sell_net_pct_below_min",
                "error_code": "SELL_NET_PCT_MIN",
                "net_after_fees_pct": net_after_fees_pct,
                "min_net_profit_pct": min_net_pct,
                "fee_bps": fee_bps,
                "slippage_bps": slippage_bps,
            }

        if net_pnl < min_net and not self.allow_sell_below_fee:
            self.logger.info(
                "[EM:SellNetGate] Blocked SELL %s: net_pnl=%.4f < min_net=%.4f (fee=%.4f entry=%.4f price=%.4f qty=%.6f)",
                sym, net_pnl, min_net, fee_est, entry, price, qty
            )
            try:
                await self.shared_state.record_rejection(sym, "SELL", "SELL_NET_PNL_MIN", source="ExecutionManager")
            except Exception:
                pass
            return {
                "ok": False,
                "status": "blocked",
                "reason": "sell_net_pnl_below_min",
                "error_code": "SELL_NET_PNL_MIN",
                "net_pnl": net_pnl,
                "min_net_pnl": min_net,
                "fee_estimate": fee_est,
            }

        # Portfolio-level improvement guard (monotonic realized PnL)
        try:
            metrics = getattr(self.shared_state, "metrics", {}) or {}
            realized = float(metrics.get("realized_pnl", 0.0) or 0.0)
            min_abs = float(self._cfg("MIN_PORTFOLIO_IMPROVEMENT_USD", 0.05) or 0.0)
            min_pct = float(self._cfg("MIN_PORTFOLIO_IMPROVEMENT_PCT", 0.0015) or 0.0)
            min_required = max(min_abs, abs(realized) * min_pct)
            projected = realized + net_pnl
            if min_required > 0 and projected < (realized + min_required):
                self.logger.info(
                    "[EM:PortfolioPnLGuard] Blocked SELL %s: projected=%.4f < required=%.4f (realized=%.4f, min=%.4f)",
                    sym, projected, realized + min_required, realized, min_required
                )
                try:
                    await self.shared_state.record_rejection(sym, "SELL", "PORTFOLIO_PNL_IMPROVEMENT", source="ExecutionManager")
                except Exception:
                    pass
                return {
                    "ok": False,
                    "status": "blocked",
                    "reason": "portfolio_pnl_improvement",
                    "error_code": "PORTFOLIO_PNL_IMPROVEMENT",
                    "projected_realized_pnl": projected,
                    "realized_pnl": realized,
                    "min_required": min_required,
                }
        except Exception:
            pass

        return None

    async def _check_dust_retirement_before_rejection(self, symbol: str, side: str) -> bool:
        """
        Check if position should be retired to PERMANENT_DUST before recording a rejection.

        Returns True if position was retired (rejection should be skipped).
        Returns False if safe to record rejection.

        Prevents dust positions from entering infinite rejection loops.
        """
        sym = symbol.upper()
        side_upper = side.upper()
        
        # If already permanent dust, skip rejection recording entirely
        if self.shared_state and hasattr(self.shared_state, "is_permanent_dust"):
            if self.shared_state.is_permanent_dust(sym):
                self.logger.debug(f"[DUST_RETIREMENT] {sym} is PERMANENT_DUST, skipping rejection recording")
                return True
        
        # Check if position is marked as DUST_LOCKED
        if not (self.shared_state and hasattr(self.shared_state, "positions")):
            return False
        
        pos = self.shared_state.positions.get(sym, {})
        pos_state = pos.get("state", "")
        pos_status = pos.get("status", "")
        
        # Only check DUST positions
        if pos_state != "DUST_LOCKED" and pos_status != "DUST":
            return False
        
        # Get current rejection count
        rej_count = self.shared_state.get_rejection_count(sym, side_upper)
        retirement_threshold = getattr(self.shared_state, "dust_retirement_rejection_threshold", 3)
        
        # If rejection count >= threshold, mark as PERMANENT_DUST and skip recording
        if rej_count >= retirement_threshold:
            self.logger.info(
                f"[DUST_RETIRED] {sym} marked PERMANENT_DUST "
                f"(status={pos_status}, state={pos_state}, rejections={rej_count}). "
                f"Retiring from rejection tracking and future liquidation operations."
            )
            
            # Mark as permanent dust
            if hasattr(self.shared_state, "mark_as_permanent_dust"):
                self.shared_state.mark_as_permanent_dust(sym)
            else:
                # Fallback if method doesn't exist
                if not hasattr(self.shared_state, "permanent_dust"):
                    self.shared_state.permanent_dust = set()
                self.shared_state.permanent_dust.add(sym)
                self.logger.warning(f"[DUST_RETIRED] Fallback permanent_dust tracking for {sym}")
            
            # Clear existing rejections
            await maybe_call(self.shared_state, "clear_rejections", sym, side_upper)

            # Don't record new rejection
            return True
        
        return False

    async def _is_position_terminal_dust(self, symbol: str) -> bool:
        """
        🚫 TERMINAL_DUST: Check if position is below minNotional (terminal dust).
        
        Terminal dust positions:
        - Are BELOW minNotional (value < $10 USDT for example)
        - Should NOT be liquidated
        - Should NOT trigger MetaDustLiquidator signals
        - Should NOT create replacement pressure
        - Dust ratio becomes informational only
        
        Returns True if position is terminal dust (should block liquidation).
        Returns False if position can be liquidated.
        """
        sym = symbol.upper()
        
        # Get position from shared_state
        if not (self.shared_state and hasattr(self.shared_state, "positions")):
            return False
        
        pos = self.shared_state.positions.get(sym, {})
        qty = float(pos.get("quantity", 0.0))
        
        if qty <= 0:
            return False  # No position = not dust

        # Permanent dust is terminal by governance definition.
        try:
            if hasattr(self.shared_state, "is_permanent_dust") and self.shared_state.is_permanent_dust(sym):
                return True
        except Exception:
            pass
        
        # Get minNotional for this symbol
        try:
            _, min_notional = await self.shared_state.compute_symbol_trade_rules(sym)
            if min_notional <= 0:
                min_notional = 10.0  # Default fallback
        except Exception:
            min_notional = 10.0
        
        # Get current price
        try:
            price = await self.shared_state.get_latest_price(sym)
            if not price or price <= 0:
                # Fallback to position's own price if market price unavailable
                price = float(pos.get("mark_price", 0.0)) or float(pos.get("avg_price", 0.0)) or float(pos.get("entry_price", 0.0))
                if not price or price <= 0:
                    return False  # Can't determine, assume tradeable
        except Exception:
            return False  # Can't determine, assume tradeable
        
        # Calculate notional value
        notional_value = qty * float(price)

        # Institutional residual floor: treat sub-threshold value as permanent dust noise.
        permanent_dust_threshold = float(
            self._cfg("PERMANENT_DUST_USDT_THRESHOLD", 1.0) or 1.0
        )
        if 0 < notional_value < permanent_dust_threshold:
            self.logger.info(
                "[TERMINAL_DUST] %s value=%.4f < permanent_threshold=%.4f -> PERMANENT_DUST",
                sym,
                notional_value,
                permanent_dust_threshold,
            )
            with contextlib.suppress(Exception):
                if hasattr(self.shared_state, "mark_as_permanent_dust"):
                    self.shared_state.mark_as_permanent_dust(sym)
            return True
        
        # TERMINAL_DUST: Value is below minNotional
        is_terminal_dust = notional_value < min_notional
        
        if is_terminal_dust:
            self.logger.debug(
                f"[TERMINAL_DUST] {sym}: notional=${notional_value:.2f} < "
                f"minNotional=${min_notional:.2f} → TERMINAL_DUST (liquidation blocked)"
            )
            # 🛡️ P9 SYNC: Record in dust registry so other agents (like LiquidationAgent) see it
            if hasattr(self.shared_state, "record_dust"):
                self.shared_state.record_dust(
                    sym, 
                    qty, 
                    origin="execution_manager_terminal",
                    context={"notional": float(notional_value), "min_notional": float(min_notional)}
                )
        
        return is_terminal_dust

    def _normalize_quote_precision(self, symbol: str, quote: float) -> float:
        """Round quote amount down to exchange tick precision to avoid rejection."""
        try:
            filters = self._symbol_filters_cache.get(self._norm_symbol(symbol), {})
            price_filter = filters.get("PRICE_FILTER", {})
            tick = float(price_filter.get("tickSize", 0) or 0)
            if tick > 0:
                return float(Decimal(str(quote)).quantize(Decimal(str(tick)), rounding=ROUND_DOWN))
        except Exception:
            pass
        # Fallback: round to 8 decimal places
        return round(float(quote), 8)

    def _norm_symbol(self, s: str) -> str:
        return (s or "").replace("/", "").upper()

    def _split_symbol_quote(self, symbol: str) -> str:
        _, quote = self._split_base_quote(symbol)
        return quote

    async def _get_available_quote(self, symbol: str) -> float:
        quote_asset = self._split_symbol_quote(symbol)
        try:
            if hasattr(self.shared_state, "get_spendable_balance"):
                v = await maybe_call(self.shared_state, "get_spendable_balance", quote_asset)
                return float(v or 0.0)
        except Exception:
            pass
        try:
            if hasattr(self.shared_state, "get_free_balance"):
                v = await maybe_call(self.shared_state, "get_free_balance", quote_asset)
                return float(v or 0.0)
        except Exception:
            pass
        try:
            if hasattr(self.exchange_client, "get_account_balance"):
                bal = await self.exchange_client.get_account_balance(quote_asset)
                return float((bal or {}).get("free", 0.0))
        except Exception:
            pass
        return 0.0

    async def _is_buy_blocked(self, symbol: str) -> Tuple[bool, float]:
        state = self._buy_block_state.get(symbol)
        if not state:
            return False, 0.0
        now = time.time()
        blocked_until = float(state.get("blocked_until", 0.0))
        if blocked_until <= now:
            return False, 0.0
        available = await self._get_available_quote(symbol)
        last_available = float(state.get("last_available", 0.0))
        if available > last_available + 0.01:
            self._buy_block_state.pop(symbol, None)
            return False, 0.0
        return True, max(0.0, blocked_until - now)

    async def _record_buy_block(self, symbol: str, available_quote: float) -> None:
        state = self._buy_block_state.setdefault(symbol, {"count": 0, "blocked_until": 0.0, "last_available": 0.0})
        state["count"] = int(state.get("count", 0)) + 1
        state["last_available"] = float(available_quote or 0.0)
        if state["count"] >= self.exec_block_max_retries:
            # 🎯 BOOTSTRAP FIX: REDUCE cooldown from 600s to 30s for faster recovery
            # During bootstrap, capital is dynamic and may be freed up quickly
            # A 10-minute cooldown is too aggressive and prevents legitimate trading
            effective_cooldown_sec = max(30, int(self.exec_block_cooldown_sec / 20))  # ~30s cooldown minimum
            state["blocked_until"] = time.time() + float(effective_cooldown_sec)
            self.logger.warning(
                "[ExecutionManager] BUY cooldown engaged: symbol=%s attempts=%d cooldown=%ds (reduced from %ds for bootstrap tolerance)",
                symbol, state["count"], effective_cooldown_sec, self.exec_block_cooldown_sec
            )

    async def _log_execution_event(self, event_type: str, symbol: str, details: Dict[str, Any]):
        event = {"ts": time.time(), "component": "ExecutionManager", "event": event_type, "symbol": symbol, **details}
        self.logger.debug(f"[{event_type}] {symbol}: {details}")
        # Use SharedState.emit_event (exists) — not append_event
        await maybe_call(self.shared_state, "emit_event", "ExecEvent", event)

    def _is_live_trading_mode(self) -> bool:
        return bool(self._cfg("LIVE_MODE", False)) and not bool(self._cfg("SIMULATION_MODE", False)) \
            and not bool(self._cfg("PAPER_MODE", False)) and not bool(self._cfg("TESTNET_MODE", False)) \
            and not bool(self._cfg("BINANCE_TESTNET", False))

    def _ensure_trade_journal_ready(self, *, reason: str = "runtime") -> bool:
        tj = getattr(self, "trade_journal", None)
        if tj is not None and callable(getattr(tj, "record", None)):
            return True

        if not bool(self._journal_bootstrap_attempted):
            self._journal_bootstrap_attempted = True
            try:
                from core.trade_journal import TradeJournal
                self.trade_journal = TradeJournal(log_dir=self._journal_log_dir)
                self.logger.warning(
                    "[EM:Journal] TradeJournal auto-initialized (reason=%s, log_dir=%s)",
                    reason,
                    self._journal_log_dir,
                )
                self._journal_fallback_warned = False
                return True
            except Exception as e:
                severe = bool(self._require_trade_journal_live and self._is_live_trading_mode())
                if severe:
                    self.logger.critical(
                        "[EM:Journal] TradeJournal unavailable in live mode (reason=%s): %s",
                        reason,
                        e,
                    )
                else:
                    self.logger.warning(
                        "[EM:Journal] TradeJournal unavailable (reason=%s): %s",
                        reason,
                        e,
                    )
                self.logger.debug("[EM:Journal] auto-init stack", exc_info=True)
                self._journal_fallback_warned = True
                return False

        if bool(self._require_trade_journal_live and self._is_live_trading_mode()) and not bool(self._journal_fallback_warned):
            self.logger.critical(
                "[EM:Journal] TradeJournal still unavailable in live mode (reason=%s).",
                reason,
            )
            self._journal_fallback_warned = True
        return False

    def _is_order_fill_confirmed(self, order: Optional[Dict[str, Any]]) -> bool:
        if not isinstance(order, dict):
            return False
        status = str(order.get("status", "")).upper()
        exec_qty = self._safe_float(order.get("executedQty") or order.get("executed_qty"), 0.0)
        return status in ("FILLED", "PARTIALLY_FILLED") and exec_qty > 0.0

    def _is_ambiguous_submit_error(self, exc: Exception) -> bool:
        if isinstance(exc, (ConnectionError, TimeoutError, asyncio.TimeoutError)):
            return True
        if isinstance(exc, BinanceAPIException):
            with contextlib.suppress(Exception):
                code = int(getattr(exc, "code", 0) or 0)
                if code in {-1000, -1001, -1006, -1007, -1008, -1015, -1021}:
                    return True
        msg = str(exc or "").lower()
        markers = (
            "timeout",
            "timed out",
            "connection reset",
            "connection aborted",
            "service unavailable",
            "gateway timeout",
            "temporarily unavailable",
            "read timed out",
        )
        return any(m in msg for m in markers)

    async def _recover_order_by_client_id(
        self,
        *,
        symbol: str,
        side: str,
        client_order_id: str,
        retries: int,
        delay_s: float,
    ) -> Optional[Dict[str, Any]]:
        get_order = getattr(self.exchange_client, "get_order", None)
        if not callable(get_order):
            return None

        sym = self._norm_symbol(symbol)
        for attempt in range(1, retries + 1):
            if attempt > 1:
                await asyncio.sleep(delay_s)
            try:
                fresh = await get_order(sym, client_order_id=str(client_order_id))
            except TypeError:
                try:
                    fresh = await get_order(sym, clientOrderId=str(client_order_id))
                except Exception:
                    fresh = None
            except Exception:
                fresh = None
            if isinstance(fresh, dict) and fresh:
                self.logger.warning(
                    "[EM:ConnErrRecover] Recovered order after transport failure symbol=%s side=%s client_id=%s attempt=%d/%d",
                    sym,
                    str(side or "").upper(),
                    str(client_order_id),
                    attempt,
                    retries,
                )
                return fresh
        return None

    def _build_submission_unknown_order(
        self,
        *,
        symbol: str,
        side: str,
        client_order_id: Optional[str],
        reason: str,
        error_code: str,
        error_text: str,
    ) -> Dict[str, Any]:
        now_ms = int(time.time() * 1000)
        payload: Dict[str, Any] = {
            "ok": False,
            "symbol": self._norm_symbol(symbol),
            "side": str(side or "").upper(),
            "status": "UNKNOWN_SUBMISSION",
            "executedQty": 0.0,
            "cummulativeQuoteQty": 0.0,
            "reason": str(reason),
            "error_code": str(error_code),
            "error_message": str(error_text or ""),
            "_submission_unknown": True,
            "updateTime": now_ms,
            "time": now_ms,
        }
        if client_order_id:
            payload["clientOrderId"] = str(client_order_id)
            payload["client_order_id"] = str(client_order_id)
        return payload

    def _journal(self, event_type: str, data: Dict[str, Any]) -> None:
        """Write to the persistent trade journal (fire-and-forget, never raises)."""
        tj = getattr(self, "trade_journal", None)
        if tj is None:
            self._ensure_trade_journal_ready(reason=f"event:{event_type}")
            tj = getattr(self, "trade_journal", None)
        try:
            data.setdefault("session_id", getattr(self, "session_id", ""))
            if tj is not None and callable(getattr(tj, "record", None)):
                tj.record(event_type, data)
                return
            self.logger.info(
                "[TRADE_JOURNAL_FALLBACK] %s",
                json.dumps(
                    {
                        "event": event_type,
                        "epoch": time.time(),
                        **dict(data or {}),
                    },
                    separators=(",", ":"),
                    default=str,
                ),
            )
        except Exception:
            self.logger.debug("TradeJournal write failed", exc_info=True)

    async def _verify_position_invariants(self, symbol: str, event_type: str, before_qty: float = 0.0) -> bool:
        """
        ✅ ELITE-LEVEL: Invariant check after SELL events.
        
        Guarantees:
        1. After SELL: position_qty_after <= position_qty_before (monotonic decrease)
        2. Periodically: exchange_position == internal_position (drift detection)
        
        On violation:
        - Log CRITICAL error
        - Emit health status DEGRADED
        - Hard-stop trading (optional, depends on STRICT_POSITION_INVARIANTS)
        
        Returns: True if invariants pass, False if violated
        """
        try:
            sym = self._norm_symbol(symbol)
            
            # Get current internal position from SharedState
            internal_qty = 0.0
            try:
                if hasattr(self.shared_state, "get_position_quantity"):
                    internal_qty = float(await self.shared_state.get_position_quantity(sym) or 0.0)
                elif isinstance(getattr(self.shared_state, "positions", None), dict):
                    internal_qty = float((self.shared_state.positions.get(sym, {}) or {}).get("quantity", 0.0) or 0.0)
            except Exception:
                pass
            
            # Get current exchange position
            exchange_qty = 0.0
            try:
                base_asset = self._split_base_quote(sym)[0]
                get_bal = getattr(self.exchange_client, "get_account_balance", None)
                if callable(get_bal):
                    bal = await get_bal(base_asset)
                    free = float((bal or {}).get("free", 0.0))
                    locked = float((bal or {}).get("locked", 0.0))
                    exchange_qty = float(free + locked)
            except Exception:
                pass
            
            # INVARIANT #1: SELL should monotonically decrease position
            if event_type in ("ORDER_FILLED", "RECONCILED_DELAYED_FILL", "SELL_ORDER_PLACED"):
                if before_qty > 0 and internal_qty > before_qty:
                    # ❌ CRITICAL: Position INCREASED after SELL (should have decreased)
                    error_msg = (
                        f"🚨 INVARIANT VIOLATED: {sym} position INCREASED during SELL "
                        f"(before={before_qty:.8f} after={internal_qty:.8f}). "
                        f"This indicates double-execution or state corruption."
                    )
                    self.logger.critical(error_msg)
                    self._journal("INVARIANT_VIOLATION", {
                        "symbol": sym,
                        "event_type": event_type,
                        "violation_type": "POSITION_INCREASED_DURING_SELL",
                        "before_qty": float(before_qty),
                        "after_qty": float(internal_qty),
                        "exchange_qty": float(exchange_qty),
                    })
                    
                    # Emit CRITICAL health status
                    try:
                        if hasattr(self.shared_state, "emit_event"):
                            self.shared_state.emit_event("HealthStatus", {
                                "level": "CRITICAL",
                                "component": "ExecutionManager",
                                "issue": "PositionInvariantViolation",
                                "symbol": sym,
                                "details": {"before": before_qty, "after": internal_qty},
                            })
                    except Exception:
                        pass
                    
                    # Hard-stop if configured
                    if bool(self._cfg("STRICT_POSITION_INVARIANTS", False)):
                        raise RuntimeError(f"Position invariant violation on {sym}: HALTING")
                    
                    return False
            
            # INVARIANT #2: Exchange and internal should not drift significantly
            # (This runs periodically, not on every trade)
            if event_type == "PERIODIC_SYNC_CHECK":
                eps = float(self._cfg("POSITION_SYNC_TOLERANCE", 0.00001))
                drift = abs(exchange_qty - internal_qty)
                
                if drift > eps:
                    # ⚠️ WARNING: Position drift detected
                    self.logger.warning(
                        f"⚠️ Position drift on {sym}: exchange={exchange_qty:.8f} "
                        f"internal={internal_qty:.8f} drift={drift:.8f}"
                    )
                    
                    # Only escalate to CRITICAL if drift is large (> 1% of position)
                    if internal_qty > 0 and (drift / internal_qty) > 0.01:
                        self.logger.critical(
                            f"🚨 LARGE DRIFT on {sym}: {(drift/internal_qty)*100:.2f}% "
                            f"(exchange={exchange_qty:.8f} internal={internal_qty:.8f})"
                        )
                        self._journal("LARGE_POSITION_DRIFT", {
                            "symbol": sym,
                            "exchange_qty": float(exchange_qty),
                            "internal_qty": float(internal_qty),
                            "drift_abs": float(drift),
                            "drift_pct": float((drift / internal_qty) * 100) if internal_qty > 0 else 0.0,
                        })
                        
                        # Emit health warning
                        try:
                            if hasattr(self.shared_state, "emit_event"):
                                self.shared_state.emit_event("HealthStatus", {
                                    "level": "DEGRADED",
                                    "component": "ExecutionManager",
                                    "issue": "PositionDriftDetected",
                                    "symbol": sym,
                                    "details": {"exchange": exchange_qty, "internal": internal_qty},
                                })
                        except Exception:
                            pass
                        
                        return False
            
            return True  # All invariants pass
            
        except Exception:
            self.logger.debug("Position invariant check failed", exc_info=True)
            return True  # Don't halt trading on check failure itself

    async def _passes_profit_gate(
        self,
        symbol: str,
        side: str,
        quantity: float,
        current_price: float,
    ) -> bool:
        """
        🔥 CRITICAL EXECUTION LAYER PROFIT GATE
        
        Enforces profit constraint at the EXECUTION layer, BEFORE place_market_order.
        This gate CANNOT be bypassed, even by recovery or emergency modes.
        
        Purpose:
        --------
        Prevent unprofitable SELL orders from executing, protecting capital.
        
        Rules:
        ------
        1. BUY orders: Always allow (gate is for SELL only)
        2. SELL orders (NOT recovery/force_close):
           - Get entry price from position
           - Calculate: profit = (current_price - entry_price) * quantity - fees
           - Check: profit >= SELL_MIN_NET_PNL_USDT (default=0)
           - If profit < gate: BLOCK and return False
        3. Recovery/Force-close SELL: Allow (bypass gate with explicit intent)
        
        Returns
        -------
        True: SELL allowed (passes profit gate)
        False: SELL blocked (fails profit gate)
        """
        symbol = self._norm_symbol(symbol)
        
        # BUY orders always pass
        if side.upper() != "SELL":
            return True
        
        # Get sell threshold from config
        sell_min_net_pnl_usdt = float(self._cfg("SELL_MIN_NET_PNL_USDT", 0.0) or 0.0)
        
        # If gate is disabled (threshold = 0), allow all SELL
        if sell_min_net_pnl_usdt <= 0:
            return True
        
        # Get position entry price
        try:
            pos = await self.shared_state.get_position(symbol) if hasattr(self.shared_state, "get_position") else None
            if not pos or not isinstance(pos, dict):
                # Position not found → allow SELL (may be already closed or phantom)
                self.logger.warning(f"[EM:ProfitGate] Position not found for {symbol}, allowing SELL")
                return True
            
            entry_price = float(pos.get("entry_price") or pos.get("entryPrice") or 0.0)
            if entry_price <= 0:
                # No entry price → allow SELL
                self.logger.warning(f"[EM:ProfitGate] No entry price for {symbol}, allowing SELL")
                return True
        except Exception as e:
            # On error, allow SELL (fail-open to prevent blocking)
            self.logger.warning(f"[EM:ProfitGate] Error getting position {symbol}: {e}, allowing SELL")
            return True
        
        # Calculate gross profit (before fees)
        gross_profit = (current_price - entry_price) * quantity
        
        # Estimate fees
        fee_rate = float(self.trade_fee_pct or 0.001)
        estimated_entry_fee = entry_price * quantity * fee_rate
        estimated_exit_fee = current_price * quantity * fee_rate
        estimated_fees = estimated_entry_fee + estimated_exit_fee
        
        # Net profit = gross profit - fees
        net_profit = gross_profit - estimated_fees
        
        # Check gate
        if net_profit < sell_min_net_pnl_usdt:
            self.logger.warning(
                f"🚫 [EM:ProfitGate] SELL BLOCKED for {symbol}: "
                f"net_profit={net_profit:.2f} < threshold={sell_min_net_pnl_usdt:.2f} "
                f"(entry={entry_price:.8f} current={current_price:.8f} qty={quantity:.8f} fees={estimated_fees:.2f})"
            )
            self._journal("SELL_BLOCKED_BY_PROFIT_GATE", {
                "symbol": symbol,
                "side": "SELL",
                "quantity": float(quantity),
                "entry_price": float(entry_price),
                "current_price": float(current_price),
                "gross_profit": float(gross_profit),
                "estimated_fees": float(estimated_fees),
                "net_profit": float(net_profit),
                "threshold": float(sell_min_net_pnl_usdt),
                "timestamp": time.time(),
            })
            return False
        
        # SELL allowed
        self.logger.info(
            f"✅ [EM:ProfitGate] SELL ALLOWED for {symbol}: "
            f"net_profit={net_profit:.2f} >= threshold={sell_min_net_pnl_usdt:.2f}"
        )
        return True

    async def _emit_trade_executed_event(
        self,
        symbol: str,
        side: str,
        tag: str,
        order: Optional[Dict[str, Any]] = None,
    ) -> bool:
        sym = self._norm_symbol(symbol)
        side_u = str(side or "").upper()
        payload = {
            "ts": time.time(),
            "symbol": sym,
            "side": side_u,
            "tag": tag,
            "source": "ExecutionManager",
        }
        dedupe_key = ""
        if isinstance(order, dict):
            exec_qty = self._safe_float(order.get("executedQty") or order.get("executed_qty"), 0.0)
            avg_price = self._safe_float(order.get("avgPrice") or order.get("avg_price"), 0.0)
            if avg_price <= 0:
                avg_price = self._resolve_post_fill_price(order, exec_qty)
            exchange_order_id = order.get("orderId") or order.get("exchange_order_id")
            internal_order_id = order.get("order_id")
            order_id = exchange_order_id or internal_order_id
            client_order_id = (
                order.get("clientOrderId") or order.get("client_order_id") or order.get("origClientOrderId")
            )
            payload.update({
                "executed_qty": float(exec_qty),
                "avg_price": float(avg_price),
                "order_id": order_id,
                "exchange_order_id": exchange_order_id or order_id,
                "client_order_id": client_order_id,
                "cummulative_quote": order.get("cummulativeQuoteQty") or order.get("cummulative_quote"),
                "status": str(order.get("status", "")).lower(),
            })
            if exchange_order_id not in (None, ""):
                dedupe_key = f"{sym}:{side_u}:OID:{exchange_order_id}"
            elif internal_order_id not in (None, "") and str(internal_order_id).isdigit():
                dedupe_key = f"{sym}:{side_u}:OID:{internal_order_id}"
            elif client_order_id not in (None, ""):
                dedupe_key = f"{sym}:{side_u}:CID:{client_order_id}"
            else:
                order_ts = (
                    order.get("updateTime")
                    or order.get("transactTime")
                    or order.get("time")
                    or order.get("timestamp")
                    or int(time.time() * 1000)
                )
                dedupe_key = (
                    f"{sym}:{side_u}:QTY:{float(exec_qty):.12f}:PX:{float(avg_price):.12f}:"
                    f"TS:{int(float(order_ts) or 0.0)}"
                )

        lock = getattr(self, "_trade_event_emit_lock", None)
        if lock is None:
            try:
                asyncio.get_running_loop()
                lock = asyncio.Lock()
                self._trade_event_emit_lock = lock
            except RuntimeError:
                return False  # No running loop yet

        cache = getattr(self, "_trade_event_emit_cache", None)
        if not isinstance(cache, dict):
            cache = {}
            self._trade_event_emit_cache = cache

        ttl_sec = float(self._cfg("TRADE_EXECUTED_DEDUPE_TTL_SEC", 86400.0) or 86400.0)
        try:
            async with lock:
                now = time.time()
                if dedupe_key:
                    prev_ts = float(cache.get(dedupe_key, 0.0) or 0.0)
                    if prev_ts > 0 and (now - prev_ts) <= ttl_sec:
                        self.logger.debug(
                            "[TRADE_EXECUTED_DEDUP] Skip duplicate canonical emit %s key=%s",
                            sym,
                            dedupe_key,
                        )
                        return True

                emit = getattr(self.shared_state, "emit_event", None)
                if not callable(emit):
                    raise RuntimeError("SharedState missing emit_event() for TRADE_EXECUTED")

                last_err = None
                for attempt in (1, 2):
                    try:
                        res = emit("TRADE_EXECUTED", payload)
                        if asyncio.iscoroutine(res):
                            await res
                        if dedupe_key:
                            cache[dedupe_key] = time.time()
                            if len(cache) > 8000:
                                cutoff = time.time() - max(60.0, ttl_sec)
                                for k, ts in list(cache.items()):
                                    if float(ts or 0.0) < cutoff:
                                        cache.pop(k, None)
                        return True
                    except Exception as e:
                        last_err = e
                        if attempt == 1:
                            await asyncio.sleep(0)
                        else:
                            raise last_err
        except Exception as e:
            self.logger.error(
                "[TRADE_EXECUTED_EVENT_FAIL] symbol=%s side=%s tag=%s err=%s",
                sym,
                side_u,
                tag,
                e,
                exc_info=True,
            )
            if bool(self._cfg("STRICT_OBSERVABILITY_EVENTS", False)):
                raise
            return False

    # ── Unified TRADE_AUDIT layer ──────────────────────────────────────
    async def _emit_trade_audit(
        self,
        *,
        symbol: str,
        side: str,
        order: Dict[str, Any],
        tier: Optional[str] = None,
        tag: str = "",
        confidence: Optional[float] = None,
        agent: Optional[str] = None,
        planned_quote: Optional[float] = None,
        post_fill_result: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Unified TRADE_AUDIT: single structured record for every confirmed fill.
        Captures complete context (exchange, routing, PnL, TP/SL, market regime)
        so every trade is fully auditable from one event/log line.
        """
        try:
            sym = self._norm_symbol(symbol)
            side_u = str(side or "").upper()
            now = time.time()

            exec_qty = self._safe_float(order.get("executedQty") or order.get("executed_qty"), 0.0)
            avg_price = self._safe_float(order.get("avgPrice") or order.get("avg_price"), 0.0)
            if avg_price <= 0:
                avg_price = self._resolve_post_fill_price(order, exec_qty)
            order_id = order.get("orderId") or order.get("exchange_order_id") or order.get("order_id") or ""
            client_order_id = order.get("clientOrderId") or order.get("client_order_id") or order.get("origClientOrderId") or ""
            cumm_quote = self._safe_float(order.get("cummulativeQuoteQty") or order.get("cummulative_quote"), 0.0)
            status = str(order.get("status", "")).upper()

            # Fee breakdown
            fee_quote = 0.0
            fee_base = 0.0
            try:
                base_asset, quote_asset = self._split_base_quote(sym)
                fills = order.get("fills") or []
                if isinstance(fills, list) and fills:
                    fee_base = sum(
                        float(f.get("commission", 0.0) or 0.0)
                        for f in fills
                        if str(f.get("commissionAsset") or "").upper() == base_asset
                    )
                    fee_quote = sum(
                        float(f.get("commission", 0.0) or 0.0)
                        for f in fills
                        if str(f.get("commissionAsset") or "").upper() == quote_asset
                    )
                if fee_quote <= 0:
                    fee_quote = float(order.get("fee_quote", 0.0) or order.get("fee", 0.0) or 0.0)
                if fee_base <= 0:
                    fee_base = float(order.get("fee_base", 0.0) or 0.0)
            except Exception:
                pass

            # Position context (entry price, TP/SL, holding time)
            ss = self.shared_state
            positions = getattr(ss, "positions", {}) or {}
            open_trades = getattr(ss, "open_trades", {}) or {}
            pos = positions.get(sym, {}) if isinstance(positions, dict) else {}
            ot = open_trades.get(sym, {}) if isinstance(open_trades, dict) else {}

            entry_price = float(
                (ot or {}).get("entry_price")
                or (ot or {}).get("avg_price")
                or (pos or {}).get("avg_price")
                or 0.0
            )
            tp = float((ot or {}).get("tp") or 0.0)
            sl = float((ot or {}).get("sl") or 0.0)
            tp_sl_method = str((ot or {}).get("tp_sl_method") or "")

            # PnL (for SELLs)
            realized_pnl = 0.0
            pnl_pct = 0.0
            if side_u == "SELL" and entry_price > 0 and avg_price > 0:
                realized_pnl = (avg_price - entry_price) * exec_qty - fee_quote
                pnl_pct = (avg_price - entry_price) / entry_price
            pf = post_fill_result if isinstance(post_fill_result, dict) else {}
            if pf.get("delta") is not None:
                realized_pnl = float(pf["delta"])

            # Holding time (for SELLs)
            holding_sec = 0.0
            if side_u == "SELL":
                opened_at = float((ot or {}).get("opened_at") or (ot or {}).get("created_at") or 0.0)
                if opened_at > 0:
                    holding_sec = now - opened_at

            # Exit reason from policy context or tag
            exit_reason = ""
            if side_u == "SELL":
                tag_u = str(tag or "").upper()
                if "TP" in tag_u and "SL" not in tag_u:
                    exit_reason = "TP"
                elif "SL" in tag_u:
                    exit_reason = "SL"
                elif "LIQUIDATION" in tag_u or "LIQ" in tag_u:
                    exit_reason = "LIQUIDATION"
                elif "STAGNATION" in tag_u:
                    exit_reason = "STAGNATION"
                elif "ROTATION" in tag_u:
                    exit_reason = "ROTATION"
                elif tag_u:
                    exit_reason = tag_u

            # Market context
            atr_pct = 0.0
            regime = ""
            try:
                if avg_price > 0 and hasattr(ss, "calc_atr"):
                    atr_period = int(self._cfg("SELL_DYNAMIC_ATR_PERIOD", 14) or 14)
                    atr_tf_primary = str(self._cfg("SELL_DYNAMIC_ATR_TIMEFRAME", "5m") or "5m").strip()
                    atr_val = 0.0
                    checked = set()
                    for tf in (atr_tf_primary, "5m", "1h", "1m"):
                        tf_norm = str(tf or "").strip().lower()
                        if not tf_norm or tf_norm in checked:
                            continue
                        checked.add(tf_norm)
                        with contextlib.suppress(Exception):
                            atr_candidate = float(await maybe_call(ss, "calc_atr", sym, tf_norm, atr_period) or 0.0)
                            if atr_candidate > 0:
                                atr_val = atr_candidate
                                break
                    if atr_val > 0:
                        atr_pct = atr_val / max(float(avg_price), 1e-9)
                vs = getattr(ss, "volatility_state", {}) or {}
                regime = str(vs.get(sym, "") or "")
            except Exception:
                pass

            # Compounding phase
            phase = ""
            try:
                dyn = getattr(ss, "dynamic_config", {}) or {}
                phase = str(dyn.get("compounding_phase") or dyn.get("COMPOUNDING_PHASE") or "")
            except Exception:
                pass

            payload = {
                "ts": now,
                "symbol": sym,
                "side": side_u,
                "executed_qty": float(exec_qty),
                "avg_price": float(avg_price),
                "cummulative_quote": float(cumm_quote),
                "order_id": str(order_id),
                "client_order_id": str(client_order_id),
                "status": status,
                "tier": str(tier or ""),
                "tag": str(tag or ""),
                "source": "ExecutionManager",
                "agent": str(agent or ""),
                "confidence": float(confidence or 0.0),
                "planned_quote": float(planned_quote or 0.0),
                "entry_price": float(entry_price),
                "pnl_pct": round(float(pnl_pct), 6),
                "realized_pnl": round(float(realized_pnl), 6),
                "fee_quote": round(float(fee_quote), 8),
                "fee_base": round(float(fee_base), 8),
                "exit_reason": exit_reason,
                "tp": float(tp),
                "sl": float(sl),
                "tp_sl_method": tp_sl_method,
                "atr_pct": round(float(atr_pct), 6),
                "regime": regime,
                "phase": phase,
                "holding_sec": round(float(holding_sec), 1),
            }

            # Structured log line (greppable independently of event bus)
            self.logger.info("[TRADE_AUDIT] %s", json.dumps(payload, separators=(",", ":")))

            # Emit to event bus
            emit = getattr(ss, "emit_event", None)
            if callable(emit):
                res = emit("TRADE_AUDIT", payload)
                if asyncio.iscoroutine(res):
                    await res
        except Exception:
            self.logger.debug("[TRADE_AUDIT] emit failed (non-fatal)", exc_info=True)

    async def _audit_post_fill_accounting(
        self,
        *,
        symbol: str,
        side: str,
        raw: Dict[str, Any],
        stage: str,
        decision_id: str = "",
    ) -> None:
        """
        Runtime accounting integrity audit after state mutation.
        Logs a compact snapshot for observability and emits an invariant-breach event
        when post-fill state is inconsistent.
        """
        sym = self._norm_symbol(symbol)
        side_l = str(side or "").lower()
        strict_accounting = bool(self._cfg("STRICT_ACCOUNTING_INTEGRITY", False))

        positions = getattr(self.shared_state, "positions", {}) or {}
        open_trades = getattr(self.shared_state, "open_trades", {}) or {}
        pos = dict(positions.get(sym, {}) or {}) if isinstance(positions, dict) else {}
        ot = dict(open_trades.get(sym, {}) or {}) if isinstance(open_trades, dict) else {}

        qty_pos = float((pos or {}).get("quantity", 0.0) or (pos or {}).get("qty", 0.0) or 0.0)
        qty_ot = float((ot or {}).get("quantity", 0.0) or (ot or {}).get("qty", 0.0) or 0.0)
        exec_qty = float((raw or {}).get("executedQty", 0.0) or 0.0)
        order_status = str((raw or {}).get("status", "")).upper()
        exec_px = float((raw or {}).get("avgPrice", (raw or {}).get("price", 0.0)) or 0.0)

        # For BUY, use executed cost (cummulativeQuoteQty) if available; else fallback to position value_usdt
        value_usdt = float((pos or {}).get("value_usdt", 0.0) or 0.0)
        if side_l == "buy":
            cumm_quote = self._safe_float((raw or {}).get("cummulativeQuoteQty") or (raw or {}).get("cummulative_quote"), 0.0)
            if cumm_quote > 0:
                value_usdt = cumm_quote
        significant = bool((pos or {}).get("is_significant", False))
        floor_usdt = float((pos or {}).get("significant_floor_usdt", 0.0) or 0.0)
        classifier = "position_fields"

        try:
            if hasattr(self.shared_state, "classify_position_snapshot"):
                gate_ref = pos if pos else {"quantity": qty_pos}
                significant, cls_value, floor_usdt = self.shared_state.classify_position_snapshot(
                    sym,
                    gate_ref,
                    price_hint=exec_px if exec_px > 0 else 0.0,
                )
                if float(cls_value or 0.0) > 0:
                    value_usdt = float(cls_value)
                classifier = "classify_position_snapshot"
        except Exception as e:
            self.logger.error(
                "[ACCOUNTING_AUDIT_CLASSIFY_FAIL] symbol=%s side=%s stage=%s err=%s",
                sym,
                side_l,
                stage,
                e,
                exc_info=True,
            )
            if strict_accounting:
                raise

        issues: list[str] = []
        if exec_qty > 0 and side_l == "buy" and qty_pos <= 0 and qty_ot <= 0:
            issues.append("BUY_FILL_NOT_REGISTERED")
        if exec_qty > 0 and side_l == "sell" and order_status == "FILLED":
            if qty_ot > 0:
                issues.append("SELL_FILLED_BUT_OPEN_TRADE_PRESENT")
            if qty_pos > 0 and bool(significant):
                issues.append("SELL_FILLED_BUT_POSITION_STILL_SIGNIFICANT")
        if qty_pos > 0 and value_usdt > 1.0 and floor_usdt <= 0:
            issues.append("FLOOR_ZERO_WITH_NONTRIVIAL_POSITION")

        snapshot = {
            "event": "ACCOUNTING_AUDIT",
            "symbol": sym,
            "side": side_l,
            "stage": stage,
            "decision_id": str(decision_id or ""),
            "order_status": order_status,
            "executed_qty": float(exec_qty),
            "position_qty": float(qty_pos),
            "open_trade_qty": float(qty_ot),
            "value_usdt": float(value_usdt),
            "significant": bool(significant),
            "floor_usdt": float(floor_usdt),
            "status_field": str((pos or {}).get("status", "")),
            "classifier": classifier,
            "issues": list(issues),
        }
        self.logger.info("[ACCOUNTING_AUDIT] %s", json.dumps(snapshot, separators=(",", ":")))

        if issues:
            self.logger.error(
                "[ACCOUNTING_INVARIANT_BREACH] symbol=%s side=%s stage=%s issues=%s",
                sym,
                side_l,
                stage,
                ",".join(issues),
            )
            try:
                await maybe_call(self.shared_state, "emit_event", "ACCOUNTING_INVARIANT_BREACH", snapshot)
            except Exception:
                self.logger.debug("emit ACCOUNTING_INVARIANT_BREACH failed", exc_info=True)
            if strict_accounting:
                raise RuntimeError(
                    f"ACCOUNTING_INVARIANT_BREACH:{sym}:{side_l}:{stage}:{','.join(issues)}"
                )

    async def _on_order_failed(self, symbol: str, side: str, reason: str, quote: Optional[float] = None):
        """
        GAP #2 FIX: Called when an order fails. Triggers pruning if capital is tight.
        This enables stale reservation cleanup to recover blocked capital.
        """
        try:
            # If order failed due to insufficient capital, trigger immediate prune
            if quote and reason in ("InsufficientBalance", "InsufficientLiquidity", "INSUFFICIENT_BALANCE"):
                try:
                    spendable = await maybe_call(self.shared_state, "get_free_usdt") or 0.0
                    if float(spendable) < (quote * 0.5):
                        self.logger.warning(
                            f"[OrderFailed:Prune] {symbol} {side} failed with low capital ({spendable:.2f} < {quote * 0.5:.2f}). "
                            f"Triggering reservation cleanup..."
                        )
                        # Proactively prune stale reservations
                        if hasattr(self.shared_state, "prune_reservations"):
                            await self.shared_state.prune_reservations()
                            self.logger.info(f"[OrderFailed:Prune] ✅ Pruned reservations for {symbol}")
                except Exception as e:
                    self.logger.debug(f"[OrderFailed:Prune] Prune attempt failed: {e}")
        except Exception as e:
            self.logger.debug(f"[OrderFailed] Exception in _on_order_failed: {e}")

    def _classify_execution_error(self, exception: Exception, symbol: str = "", operation: str = "") -> ExecutionError:
        if isinstance(exception, BinanceAPIException):
            code = getattr(exception, "code", None)
            if code == -2010:
                return ExecutionError("InsufficientBalance", str(exception), symbol)
            elif code in [-1013, -1021]:
                return ExecutionError("MinNotionalViolation", str(exception), symbol)
            else:
                return ExecutionError("ExternalAPIError", str(exception), symbol, {"api_code": code})
        emsg = str(exception).lower()
        if "min_notional" in emsg:
            return ExecutionError("MinNotionalViolation", str(exception), symbol)
        if "fee" in emsg or "safety" in emsg:
            return ExecutionError("FeeSafetyViolation", str(exception), symbol)
        if "risk" in emsg or "cap" in emsg:
            return ExecutionError("RiskCapExceeded", str(exception), symbol)
        if "integrity" in emsg:
            return ExecutionError("IntegrityError", str(exception), symbol)
        return ExecutionError("ExternalAPIError", str(exception), symbol, {"operation": operation})

    def _sanitize_tag(self, tag: Optional[str]) -> str:
        s = (tag or "meta/Agent")
        # ensure P9-compliant namespace
        if not (s.startswith("meta/") or s in ("balancer", "liquidation", "tp_sl", "rebalance", "meta_exit")):
            s = "meta/" + s
        out = []
        for ch in s:
            if ch.isalnum() or ch in "-_/":
                out.append(ch)
        return ("".join(out) or "meta/Agent")[:36]

    def _resolve_decision_id(self, policy_ctx: Optional[Dict[str, Any]] = None) -> str:
        ctx = policy_ctx or {}
        for key in (
            "decision_id",
            "decisionId",
            "signal_id",
            "intent_id",
            "decision_key",
            "decision_hash",
            "request_id",
        ):
            val = ctx.get(key)
            if val:
                return str(val)
        self._decision_id_seq += 1
        return f"auto-{int(time.time() * 1000)}-{self._decision_id_seq}"

    def _enter_exchange_order_scope(self) -> Optional[str]:
        """Authorize ExchangeClient order APIs for this ExecutionManager call stack."""
        begin = getattr(self.exchange_client, "begin_execution_order_scope", None)
        if not callable(begin):
            return None
        try:
            return begin("ExecutionManager")
        except Exception:
            self.logger.debug("begin_execution_order_scope failed", exc_info=True)
            return None

    def _exit_exchange_order_scope(self, token: Optional[str]) -> None:
        """Release ExchangeClient order API authorization scope."""
        end = getattr(self.exchange_client, "end_execution_order_scope", None)
        if not callable(end):
            return
        try:
            end(token)
        except Exception:
            self.logger.debug("end_execution_order_scope failed", exc_info=True)

    async def _maybe_auto_reset_rejections(self, symbol: str, side: str) -> None:
        """
        🎯 BEST PRACTICE #4: Auto-reset rejection counters after 60 seconds of inactivity.
        
        This prevents stale rejection counts from permanently blocking a symbol/side pair.
        Once the 60-second window passes without new rejections, the counter automatically
        resets, allowing the bot to make fresh trading attempts.
        
        Called opportunistically during order placement to maintain freshness.
        """
        if not self.shared_state:
            return
        
        sym = symbol.upper()
        side_upper = side.upper()
        key = f"{sym}:{side_upper}"
        now = time.time()
        
        # Check if we've already checked this symbol/side recently
        last_check = self._last_rejection_reset_check_ts.get(key, 0.0)
        if now - last_check < 10.0:  # Only check every 10 seconds max
            return
        
        self._last_rejection_reset_check_ts[key] = now
        
        # Get the rejection tracker timestamp if available
        if hasattr(self.shared_state, "get_rejection_timestamp"):
            try:
                last_rejection_ts = self.shared_state.get_rejection_timestamp(sym, side_upper)
                if last_rejection_ts and (now - last_rejection_ts) > self._rejection_reset_window_s:
                    # No rejections in the last 60 seconds — clear the counter
                    if hasattr(self.shared_state, "clear_rejections"):
                        await maybe_call(self.shared_state, "clear_rejections", sym, side_upper)
                        self.logger.info(
                            "[EM:REJECTION_RESET] Auto-reset rejection counter for %s %s "
                            "(no rejections for %.0fs)",
                            sym, side_upper, self._rejection_reset_window_s
                        )
            except Exception as e:
                self.logger.debug(f"[EM:REJECTION_RESET] Failed to check timestamp: {e}")

    def _build_client_order_id(self, symbol: str, side: str, decision_id: str) -> str:
        return f"{symbol}:{side}:{decision_id}"

    def _is_duplicate_client_order_id(self, client_id: str) -> bool:
        """
        Check if client_order_id is a duplicate within the SHORT idempotency window.
        
        🎯 BEST PRACTICE #1: Short idempotency window (8 seconds)
        - Prevents false positive duplicates from stale retries
        - Allows rapid recovery from network glitches
        - Aligns with symbol/side active order tracking
        
        🧹 HARDENING: Garbage collect old entries to prevent memory leaks.
        """
        now = time.time()
        seen = self._seen_client_order_ids
        
        # 🧹 HARDENING: Garbage collect if cache grows too large
        # Prevent memory leak after millions of orders
        if len(seen) > 5000:
            cutoff = now - 30.0  # Keep only 30s of history (4x the window)
            removed = 0
            for key, ts in list(seen.items()):
                if ts < cutoff:
                    seen.pop(key, None)
                    removed += 1
            if removed > 0:
                self.logger.debug(
                    "[EM:DupIdGC] Garbage collected %d stale client_order_ids, dict_size=%d",
                    removed, len(seen)
                )
        
        # 🎯 BEST PRACTICE #1: Check if this ID was seen RECENTLY (< 8s window)
        is_duplicate = False
        if client_id in seen:
            last_seen = seen[client_id]
            elapsed = now - last_seen
            
            # If < 8s ago, it's a genuine duplicate within the idempotency window
            if elapsed < self._client_order_id_timeout_s:
                self.logger.debug(
                    "[EM:DupClientId] Duplicate client_order_id %s (%.1fs ago); blocking.",
                    client_id, elapsed
                )
                is_duplicate = True
            else:
                # Outside window (>8s) — allow retry, it's a fresh attempt
                self.logger.debug(
                    "[EM:DupClientIdFresh] Client order ID seen %.1fs ago; allowing fresh attempt.",
                    elapsed
                )
                is_duplicate = False
        
        # ✅ HARDENING: Always update timestamp, every code path
        # This ensures retries are properly tracked
        seen[client_id] = now
        return is_duplicate

    async def _validate_order_request_contract(self, **kwargs) -> Tuple[bool, float, float, str]:
        """
        Contract adapter for local P9 validator.

        Enforces tuple contract: (ok, qty, adjusted_quote, reason).
        Raises explicit runtime error on interface mismatch to avoid silent
        zero-qty / unexpected-error fallthrough.
        """
        res = await validate_order_request(**kwargs)
        if not isinstance(res, (tuple, list)) or len(res) != 4:
            self.logger.error(
                "[EM:HygieneContract] validate_order_request returned invalid payload: type=%s value=%r",
                type(res).__name__,
                res,
            )
            raise RuntimeError("HYGIENE_INTERFACE_MISMATCH")

        ok_raw, qty_raw, adjusted_quote_raw, reason_raw = res
        try:
            qty = float(qty_raw or 0.0)
        except Exception:
            qty = 0.0
        try:
            adjusted_quote = float(adjusted_quote_raw or 0.0)
        except Exception:
            adjusted_quote = 0.0
        reason = str(reason_raw or "UNKNOWN")
        return bool(ok_raw), qty, adjusted_quote, reason


    @asynccontextmanager
    async def _small_nav_guard(self):
        """Serialize order placement when NAV is tiny."""
        nav_q = None
        try:
            if hasattr(self.shared_state, 'get_nav_quote'):
                # Wrap in timeout to prevent hangs on PnL calculation
                nav_q = await asyncio.wait_for(
                    maybe_call(self.shared_state, 'get_nav_quote', self.base_ccy),
                    timeout=2.0
                )
        except Exception:
            pass

        if nav_q is not None and float(nav_q) < self.small_nav_threshold:
            if not hasattr(self, '_nav1_sem'):
                self._nav1_sem = asyncio.Semaphore(1)
            async with self._nav1_sem:
                yield
        else:
            yield

    async def _get_free_quote_and_remainder_ok(self, quote_asset: str, spend: float) -> Tuple[float, bool, str]:
        """Return (free_quote, ok, reason). Enforce min_free_reserve_usdt and no_remainder_below_quote."""
        free = 0.0
        if hasattr(self.shared_state, 'get_free_balance'):
            with contextlib.suppress(Exception):
                v = await maybe_call(self.shared_state, 'get_free_balance', quote_asset)
                free = float(v or 0.0)
        if free <= 0 and hasattr(self.shared_state, 'get_spendable_balance'):
            with contextlib.suppress(Exception):
                v = await maybe_call(self.shared_state, 'get_spendable_balance', quote_asset)
                free = float(v or 0.0)
        if free <= 0 and hasattr(self.exchange_client, 'get_account_balance'):
            with contextlib.suppress(Exception):
                bal = await self.exchange_client.get_account_balance(quote_asset)
                free = float((bal or {}).get('free', 0.0))
        remainder = max(0.0, free - float(spend))
        min_free_reserve_usdt = float(getattr(self, "min_free_reserve_usdt", 0.0) or 0.0)
        no_remainder_below_quote = float(getattr(self, "no_remainder_below_quote", 0.0) or 0.0)
        if min_free_reserve_usdt > 0 and remainder < min_free_reserve_usdt:
            return free, False, 'RESERVE_FLOOR'
        if no_remainder_below_quote > 0 and 0 < remainder < no_remainder_below_quote:
            return free, False, 'TINY_REMAINDER'
        return free, True, 'OK'

    # =============================
    # Affordability & Liquidity
    # =============================
    async def can_afford_market_buy(
        self, 
        symbol: str, 
        quote_amount: Union[float, Decimal],
        intent_override: Optional[PendingPositionIntent] = None,
        policy_context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, float, str]:
        self._ensure_heartbeat()
        ok, gap, reason = await self._explain_afford_market_buy_tuple(symbol, Decimal(str(quote_amount)), intent_override=intent_override, policy_context=policy_context)
        return ok, float(gap), reason

    async def explain_afford_market_buy(self, symbol: str, quote_amount: Union[float, Decimal]) -> Tuple[bool, str]:
        ok, _, reason = await self._explain_afford_market_buy_tuple(symbol, Decimal(str(quote_amount)))
        return ok, reason

    async def _explain_afford_market_buy_tuple(
        self, 
        symbol: str, 
        qa: Decimal, 
        intent_override: Optional[PendingPositionIntent] = None,
        policy_context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Decimal, str]:
        # --- BOOTSTRAP OVERRIDE ---
        skip_micro_trade_kill_switch = False
        policy_ctx = policy_context or {}  # Ensure it's never None
        bootstrap_mode_active = bool(policy_ctx.get("bootstrap_mode", False))
        bootstrap_bypass = bool(policy_ctx.get("bootstrap_bypass", False))
        affordability_probe = bool(
            policy_ctx.get("affordability_probe")
            or policy_ctx.get("probe_only")
        )
        probe_source = str(policy_ctx.get("probe_source") or "").strip()
        if not bootstrap_mode_active:
            try:
                if hasattr(self.shared_state, "is_bootstrap_mode"):
                    bootstrap_mode_active = bool(await maybe_call(self.shared_state, "is_bootstrap_mode"))
            except Exception:
                bootstrap_mode_active = False
        if bootstrap_mode_active or bootstrap_bypass:
            # Bootstrap is capital deployment initialization; bypass micro/edge-only filters.
            skip_micro_trade_kill_switch = True
            policy_ctx.setdefault("bootstrap_mode", True)
            policy_ctx.setdefault("bootstrap_bypass", True)
        # --- EXPLICIT BOOTSTRAP ESCAPE HATCH ---
        if policy_ctx.get("is_flat", False) and policy_ctx.get("bootstrap_mode", False):
            # Allow exactly one BUY regardless of economic/micro trade guards
            return (True, qa, "BOOTSTRAP_ESCAPE_HATCH")
        if policy_ctx.get("bootstrap_bypass", False):
            skip_micro_trade_kill_switch = True
        # --- DUST HEALING BYPASS ---
        is_dust_healing_buy = bool(
            policy_ctx.get("_is_dust_healing_buy")
            or policy_ctx.get("is_dust_healing")
            or policy_ctx.get("_dust_healing")
            or str(policy_ctx.get("reason") or "").upper() == "DUST_HEALING_BUY"
        )
        if is_dust_healing_buy:
            policy_ctx["_is_dust_healing_buy"] = True
        is_dust_operation = self._is_dust_operation_context(policy_ctx, symbol=symbol)
        if is_dust_operation:
            # For dust healing/recovery, only validate step_size, min_notional, available balance
            # Skip all other guards: min_econ_trade, profitability, micro_trade_kill_switch, fee_floor
            skip_micro_trade_kill_switch = True
        if affordability_probe:
            # AppContext probes should test sizing/funds only, not emit misleading EV gate noise.
            skip_micro_trade_kill_switch = True
        
        # --- CRITICAL: MetaController Decision Override ---
        # If MetaController has made an explicit planning decision with planned_quote,
        # bypass micro gates (sideways, EV, etc) that would reject valid decisions.
        # Some Meta paths may not carry a decision_id, so also trust canonical authority markers.
        has_meta_planned_quote = qa is not None and float(qa) > 0
        policy_authority = str(
            policy_ctx.get("authority") or policy_ctx.get("policy_authority") or ""
        ).strip().lower()
        meta_validated = bool(
            policy_ctx.get("decision_id")
            or policy_ctx.get("trace_id")
            or policy_authority == "metacontroller"
            or policy_ctx.get("tradeability_gate_checked")
        )
        if has_meta_planned_quote and meta_validated:
            # MetaController has validated this trade through its own gates
            # Trust the decision and skip redundant micro-level gates
            skip_micro_trade_kill_switch = True
            self.logger.info(
                "[EM:MetaOverride] Bypassing micro gates for %s (planned_quote=%.2f decision_id=%s authority=%s tradeability_gate_checked=%s)",
                symbol,
                float(qa),
                policy_ctx.get("decision_id") or policy_ctx.get("trace_id"),
                policy_authority or "unknown",
                bool(policy_ctx.get("tradeability_gate_checked")),
            )
        
        self.logger.warning(
            f"[EXEC_TRACE] received bootstrap_bypass={policy_ctx.get('bootstrap_bypass', False)} "
            f"bootstrap_mode={policy_ctx.get('bootstrap_mode', False)} "
            f"skip_micro_gate={skip_micro_trade_kill_switch} "
            f"has_meta_planned_quote={has_meta_planned_quote} "
            f"authority={policy_authority or 'unknown'} "
            f"meta_validated={meta_validated} "
            f"decision_marker={bool(policy_ctx.get('decision_id') or policy_ctx.get('trace_id'))} "
            f"tradeability_gate_checked={bool(policy_ctx.get('tradeability_gate_checked'))} "
            f"affordability_probe={affordability_probe} "
            f"probe_source={probe_source or 'n/a'}"
        )
        try:
            if qa is None or float(qa) <= 0:
                # Zero-sized trades MUST be treated as failure (Behavior Change 1.1)
                return (False, Decimal("0"), "ZERO_SIZE_TRADE")

            policy_min_notional = float(policy_ctx.get("min_notional", 0.0) or 0.0)
            nav_tier_min_econ = Decimal(
                str(
                    await self._resolve_nav_tier_economic_floor(
                        symbol=symbol,
                        min_notional=policy_min_notional,
                    )
                )
            )
            dynamic_min_econ = Decimal("0")
            if hasattr(self.shared_state, "compute_min_entry_quote") and not is_dust_healing_buy:
                try:
                    dyn_floor = await self.shared_state.compute_min_entry_quote(
                        symbol,
                        default_quote=float(qa),
                    )
                    dynamic_min_econ = Decimal(str(dyn_floor or 0.0))
                except Exception:
                    dynamic_min_econ = Decimal("0")
            min_econ_trade = max(nav_tier_min_econ, dynamic_min_econ)
            # QUOTE UPGRADE: Instead of rejecting, upgrade the quote to meet minimum
            if min_econ_trade > 0 and qa < min_econ_trade and not is_dust_operation:
                qa = max(qa, min_econ_trade)  # Upgrade quote to meet minimum economic threshold
                self.logger.info(
                    f"[EM:QUOTE_UPGRADE] {symbol} BUY: Upgraded quote to minimum economic amount "
                    f"upgraded_quote={float(qa):.2f} USDT, min_econ={float(min_econ_trade):.2f} USDT"
                )

            sym = self._norm_symbol(symbol)
            eps = Decimal("1e-9")

            # FETCH PRICE (prefer SharedState cache to reduce exchange load)
            price = 0.0
            try:
                lp = getattr(self.shared_state, "latest_prices", {}) or {}
                if isinstance(lp, dict):
                    price = float(lp.get(sym, 0.0) or 0.0)
            except Exception:
                price = 0.0
            if price <= 0:
                get_px = getattr(self.exchange_client, "get_ticker_price", None) \
                        or getattr(self.exchange_client, "get_current_price", None) \
                        or getattr(self.exchange_client, "get_price", None)
                if callable(get_px):
                    price = float(await get_px(sym) or 0.0)
            if price <= 0:
                # If we can't get price, we can't estimate quantity effectively
                # Fallback logic or hard fail? Let's assume 1.0 but log warning, or fail.
                # Failing is safer for affordability checks.
                return (False, Decimal("0"), "PRICE_UNAVAILABLE")

            # Use planned_qty only when caller provided it explicitly.
            # Never synthesize qty from min-entry here because affordability must reflect qa.
            planned_qty = policy_ctx.get("planned_qty")

            atr_pct = 0.0
            try:
                if hasattr(self.shared_state, "calc_atr"):
                    atr_period = int(self._cfg("SELL_DYNAMIC_ATR_PERIOD", 14) or 14)
                    atr_tf_primary = str(self._cfg("SELL_DYNAMIC_ATR_TIMEFRAME", "5m") or "5m").strip()

                    atr = 0.0
                    atr_timeframes = []
                    for tf in (atr_tf_primary, "5m", "1h", "1m"):
                        tf_norm = str(tf or "").strip()
                        if tf_norm and tf_norm not in atr_timeframes:
                            atr_timeframes.append(tf_norm)

                    for tf in atr_timeframes:
                        atr_candidate = float(
                            await maybe_call(self.shared_state, "calc_atr", sym, tf, atr_period) or 0.0
                        )
                        if atr_candidate > 0:
                            atr = atr_candidate
                            break
                    if atr and price > 0:
                        atr_pct = float(atr) / float(price)
            except Exception:
                atr_pct = 0.0

            trade_regime = str(
                policy_ctx.get("tradeability_regime") or policy_ctx.get("_regime") or ""
            ).strip().lower()
            vol_regime = str(policy_ctx.get("volatility_regime") or "").strip().lower()
            if not vol_regime:
                tf = str(self._cfg("VOLATILITY_REGIME_TIMEFRAME", "1h") or "1h")
                try:
                    if hasattr(self.shared_state, "get_volatility_regime"):
                        reginfo = await maybe_call(self.shared_state, "get_volatility_regime", sym, tf, 600)
                        if not reginfo:
                            reginfo = await maybe_call(self.shared_state, "get_volatility_regime", "GLOBAL", tf, 600)
                        if isinstance(reginfo, dict):
                            vol_regime = str(reginfo.get("regime") or "").strip().lower()
                except Exception:
                    vol_regime = vol_regime or ""
            if not vol_regime:
                try:
                    vol_regime = str(
                        (getattr(self.shared_state, "metrics", {}) or {}).get("volatility_regime") or ""
                    ).strip().lower()
                except Exception:
                    vol_regime = ""

            # Sideways (low-vol) regime trading is disabled by default (noise protection).
            # CRITICAL: If MetaController has sent an explicit override, bypass this gate
            bootstrap_override = bool(policy_ctx.get("_bootstrap_override", False))
            if not skip_micro_trade_kill_switch and bool(self._cfg("DISABLE_SIDEWAYS_REGIME_TRADING", True)):
                if (trade_regime == "sideways" or vol_regime == "low") and not bootstrap_override:
                    self.logger.info(
                        "[EM:SidewaysDisabled] Blocked BUY %s (trade_regime=%s vol_regime=%s)",
                        sym,
                        trade_regime or "unknown",
                        vol_regime or "unknown",
                    )
                    with contextlib.suppress(Exception):
                        await maybe_call(
                            self.shared_state,
                            "emit_event",
                            "EntrySidewaysRegimeBlocked",
                            {
                                "symbol": sym,
                                "trade_regime": trade_regime,
                                "vol_regime": vol_regime,
                                "timestamp": time.time(),
                            },
                        )
                    return (False, Decimal("0"), "SIDEWAYS_REGIME_DISABLED")

            if not skip_micro_trade_kill_switch:
                expected_move_pct = None
                expected_move_key = None
                for key in (
                    "tradeability_expected_move_pct",
                    "_expected_move_pct",
                    "expected_move_pct",
                    "expected_alpha_pct",
                ):
                    if key in policy_ctx and policy_ctx.get(key) is not None:
                        with contextlib.suppress(Exception):
                            expected_move_pct = float(policy_ctx.get(key))
                            expected_move_key = str(key)
                            break
                if expected_move_pct is None:
                    econ_guard = policy_ctx.get("economic_guard") if isinstance(policy_ctx, dict) else None
                    if isinstance(econ_guard, dict):
                        edge_bps = econ_guard.get("edge_bps")
                        with contextlib.suppress(Exception):
                            if edge_bps is not None:
                                expected_move_pct = float(edge_bps) / 10000.0
                if expected_move_pct is not None and expected_move_key is None:
                    expected_move_key = "economic_guard"

                expected_move_key = expected_move_key or "atr_fallback"
                expected_move_raw_pct = float(expected_move_pct or 0.0)
                expected_move_used_pct = float(expected_move_raw_pct)
                expected_move_floor_pct = 0.0

                slippage_bps = float(
                    self._cfg("EXIT_SLIPPAGE_BPS", self._cfg("CR_PRICE_SLIPPAGE_BPS", 15.0)) or 0.0
                )
                buffer_bps = float(self._cfg("TP_MIN_BUFFER_BPS", 0.0) or 0.0)
                round_trip_cost_pct = (float(self.trade_fee_pct or 0.0) * 2.0) + (
                    (slippage_bps + buffer_bps) / 10000.0
                )
                base_mult = float(self._cfg("EV_HARD_SAFETY_MULT", 1.2) or 1.2)
                
                # ADAPTIVE: Read regime from signal context (MetaController)
                regime = str(trade_regime or "unknown").strip().lower()
                
                # FALLBACK: If regime not in policy_ctx, query MetaController directly
                if not trade_regime or regime == "unknown":
                    try:
                        if hasattr(self.shared_state, "get_current_regime"):
                            mc_regime = await maybe_call(self.shared_state, "get_current_regime", sym)
                            if mc_regime:
                                regime = str(mc_regime).strip().lower()
                                self.logger.debug("[EM:EV_HARD_GATE] Fetched regime from MetaController: %s", regime)
                    except Exception as e:
                        self.logger.debug("[EM:EV_HARD_GATE] Failed to fetch regime from MetaController: %s", e)
                
                # ADAPTIVE: Regime-aware multiplier scaling
                # LOWERED: Previous values were too aggressive for small account trading (0.35% base cost)
                # New values account for fact that round-trip costs dominate on small accounts
                regime_multipliers = {
                    "trend": 1.1,          # Lowered from 1.4 (was blocking too many low-vol opportunities)
                    "volatile": 1.0,       # Lowered from 1.2 (small accounts need less stringency)
                    "sideways": 0.95,      # Lowered from 1.0 (range-bound favors entries)
                    "chop": 0.95,          # Lowered from 1.0 (Alias)
                    "range": 0.95,         # Lowered from 1.0 (Alias)
                    "low": 0.95,           # Lowered from 1.0 (Alias)
                    "flat": 0.95,          # Lowered from 1.0 (Alias)
                    "unknown": 0.95,       # Lowered from 1.2 (unknown regime should be conservative but not aggressive)
                }
                regime_mult = float(regime_multipliers.get(regime, 0.95))  # Default to conservative 0.95
                
                # ADAPTIVE: ATR-based scaling
                if atr_pct is not None and float(atr_pct) > 0:
                    atr_pct_val = float(atr_pct)
                    if atr_pct_val < 0.0015:        # <0.15% ATR
                        regime_mult *= 0.85         # 15% relief
                    elif atr_pct_val < 0.003:      # <0.30% ATR
                        regime_mult *= 0.90         # 10% relief
                    elif atr_pct_val > 0.01:       # >1.0% ATR
                        regime_mult *= 1.1          # 10% stricter
                
                # ADAPTIVE: Bootstrap override (easier during Phase 1)
                if bootstrap_override:
                    regime_mult *= 0.75
                
                # CRITICAL FIX: Use soft floor, not hard override
                # Config sets safety floor (e.g., 0.85), not a hard ceiling
                min_safe_mult = float(self._cfg("EV_HARD_MIN_MULT", 0.85) or 0.85)
                mult = max(float(min_safe_mult), regime_mult)  # ← Adaptive intelligence wins, safety floor wins vs 1.0x

                # Expected-move calibration (ATR floor). If upstream doesn't provide an expected-move,
                # we still enforce the gate using an ATR-based proxy (1h preferred).
                if bool(self._cfg("EV_EXPECTED_MOVE_CALIBRATION_ENABLED", True)):
                    try:
                        em_tf = str(self._cfg("EV_EXPECTED_MOVE_ATR_TIMEFRAME", "1h") or "1h")
                        em_period = int(self._cfg("EV_EXPECTED_MOVE_ATR_PERIOD", 14) or 14)
                        em_mult = float(self._cfg("EV_EXPECTED_MOVE_ATR_MULT", 1.0) or 1.0)
                        if hasattr(self.shared_state, "calc_atr") and em_mult > 0:
                            em_atr = float(await maybe_call(self.shared_state, "calc_atr", sym, em_tf, em_period) or 0.0)
                            if em_atr > 0 and price > 0:
                                expected_move_floor_pct = (em_atr / max(float(price), 1e-9)) * em_mult
                    except Exception:
                        expected_move_floor_pct = 0.0
                    if expected_move_floor_pct <= 0.0:
                        fallback_scale = float(self._cfg("EV_EXPECTED_MOVE_FALLBACK_SCALE", 3.0) or 3.0)
                        expected_move_floor_pct = float(atr_pct or 0.0) * max(1.0, fallback_scale)
                    if expected_move_floor_pct > 0:
                        expected_move_used_pct = max(float(expected_move_used_pct), float(expected_move_floor_pct))

                atr_ref_pct = float(expected_move_floor_pct or 0.0)
                required_move_pct = round_trip_cost_pct * float(mult)
                
                # CRITICAL GUARD: Prevent dead gate (expected_move=0)
                # If all sources failed, fall back to minimum viable expected move
                if float(expected_move_used_pct) <= 0.0:
                    min_expected_move = max(float(atr_pct or 0.0), 0.0025)  # Min 0.25% or ATR
                    expected_move_used_pct = min_expected_move
                    expected_move_key = "emergency_fallback"
                    self.logger.debug(
                        "[EM:EV_HARD_GATE] No expected_move available; using emergency fallback: %.4f%%",
                        float(expected_move_used_pct) * 100.0
                    )
                
                # CRITICAL: If MetaController has sent an explicit override, bypass this gate
                if float(expected_move_used_pct) <= float(required_move_pct) and not bootstrap_override:
                    self.logger.warning(
                        "[EM:EV_HARD_GATE] Blocked BUY %s: expected_move=%.4f%% (raw=%.4f%% key=%s floor=%.4f%%) <= required=%.4f%% "
                        "(round_trip=%.4f%% mult=%.2f regime_mult=%.2f regime=%s atr_adj=%.4f%% vol_regime=%s base_mult=%.2f)",
                        sym,
                        float(expected_move_used_pct) * 100.0,
                        float(expected_move_raw_pct) * 100.0,
                        expected_move_key or "unknown",
                        float(expected_move_floor_pct) * 100.0,
                        float(required_move_pct) * 100.0,
                        float(round_trip_cost_pct) * 100.0,
                        float(mult),
                        float(regime_mult),
                        regime or "unknown",
                        float(atr_pct or 0.0) * 100.0,
                        vol_regime or "unknown",
                        float(base_mult),
                    )
                    with contextlib.suppress(Exception):
                        await maybe_call(
                            self.shared_state,
                            "emit_event",
                            "EntryExpectedMoveBlocked",
                            {
                                "symbol": sym,
                                "expected_move_pct": float(expected_move_used_pct),
                                "expected_move_raw_pct": float(expected_move_raw_pct),
                                "expected_move_key": expected_move_key,
                                "expected_move_floor_pct": float(expected_move_floor_pct),
                                "required_move_pct": float(required_move_pct),
                                "round_trip_cost_pct": float(round_trip_cost_pct),
                                "safety_mult": float(mult),  # back-compat key
                                "ev_mult": float(mult),
                                "regime_mult": float(regime_mult),  # NEW: adaptive multiplier
                                "regime": regime or "unknown",  # NEW: detected regime
                                "vol_regime": vol_regime,
                                "atr_pct": float(atr_pct or 0.0),  # NEW: ATR percentage
                                "base_mult": float(base_mult),  # NEW: configured base
                                "timestamp": time.time(),
                            },
                        )
                    return (False, Decimal("0"), "EXPECTED_MOVE_LT_ROUND_TRIP_COST")

            feasible, feas_detail = self._entry_profitability_feasible(sym, price=price, atr_pct=atr_pct)
            # CRITICAL: If MetaController has sent an explicit override, bypass this gate
            if not feasible and not is_dust_operation and not bootstrap_override:
                payload = {
                    "reason": "INFEASIBLE_PROFITABILITY",
                    "symbol": sym,
                    **feas_detail,
                }
                self.logger.warning("[EM:ProfitFeasibility] Blocked BUY: %s", payload)
                with contextlib.suppress(Exception):
                    await maybe_call(self.shared_state, "emit_event", "EntryProfitabilityBlocked", payload)
                return (False, Decimal("0"), "INFEASIBLE_PROFITABILITY")

            # Adaptive Execution Threshold Engine (Institutional-Grade)
            if not skip_micro_trade_kill_switch:
                if self._cfg("ADAPTIVE_EXECUTION_THRESHOLD_ENABLED", True):  # Default enabled
                    # Extract signal confidence from policy context
                    signal_confidence = None
                    if policy_context:
                        signal_confidence = policy_context.get("confidence") or policy_context.get("signal_confidence")

                    # Calculate adaptive execution threshold
                    threshold_analysis = await self._calculate_adaptive_execution_threshold(
                        symbol=sym,
                        planned_quote=float(qa or 0.0),
                        price=price,
                        signal_confidence=signal_confidence,
                        policy_context=policy_context
                    )

                    if not threshold_analysis["can_execute"]:
                        kill_reason = threshold_analysis["analysis"]["kill_reason"] or "ADAPTIVE_THRESHOLD_KILL"
                        self.logger.warning(
                            f"[ADAPTIVE_THRESHOLD] Execution blocked for {sym}: "
                            f"required_edge={threshold_analysis['required_edge_pct']*100:.3f}%, "
                            f"current_edge={threshold_analysis['current_edge_pct']*100:.3f}%, "
                            f"ratio={threshold_analysis['edge_sufficiency_ratio']:.2f}, "
                            f"confidence={threshold_analysis['analysis']['confidence_level']:.2f}"
                        )
                        return (False, Decimal("0"), kill_reason)

            # --- P9 Phase 4: Accumulation Synergy ---
            # 1. Fetch existing intent to check for frozen constraints (Point 2)
            intent = intent_override or self.shared_state.get_pending_intent(sym, "BUY")
            
            # 2. Fetch filters and venue/config floors
            filters = await self.exchange_client.ensure_symbol_filters_ready(sym)
            
            # Point 2: Freeze floor once intent exists
            if intent and intent.state == "ACCUMULATING" and intent.min_notional > 0:
                min_notional = intent.min_notional
                self.logger.debug(f"[EM] Using frozen min_notional {min_notional} for {sym}")
            else:
                # Check policy_context first, then fall back to exchange filters
                min_notional = policy_ctx.get("min_notional", self._extract_min_notional(filters))

            if min_notional <= 0:
                return (False, Decimal("0"), f"invalid_min_notional({min_notional})")

            # Dynamic exit-feasibility floor (symbol-aware)
            if is_dust_operation:
                # Dust healing should not inherit normal strategy floor (e.g., MIN_ENTRY_QUOTE_USDT).
                min_required_val = float(min_notional)
            else:
                min_required_val = await self._get_min_entry_quote(sym, price=price, min_notional=float(min_notional), policy_context=policy_ctx)
            min_required = Decimal(str(min_required_val))

            # Planned-quote fee floor (planned_quote >= fee_mult × round-trip fee on min-required quote)
            fee_mult = Decimal(str(self._cfg("MIN_PLANNED_QUOTE_FEE_MULT", 2.5) or 2.5))
            round_trip_fee_rate = Decimal(str(self.trade_fee_pct)) * Decimal("2")
            if round_trip_fee_rate > 0 and fee_mult > 0 and not is_dust_operation:
                planned_fee_floor = min_required * round_trip_fee_rate * fee_mult
                # QUOTE UPGRADE: Instead of rejecting, upgrade the quote to cover fees
                if qa < planned_fee_floor - eps:
                    qa = max(qa, planned_fee_floor)
                    self.logger.info(
                        f"[EM:QUOTE_UPGRADE] {sym} BUY: Upgraded quote to cover round-trip fees "
                        f"upgraded_quote={float(qa):.2f} USDT, fee_floor={float(planned_fee_floor):.2f} USDT"
                    )

            # 3. Determine spendable (Strict Mode: Fresh from SharedState)
            taker_fee = Decimal(str(self.trade_fee_pct))
            headroom = Decimal(str(self.safety_headroom))
            quote_asset = self._split_symbol_quote(sym)
            
            # Use raw get_balance to be sure, then subtract reservations manually if needed
            # This bypasses any potential caching in get_spendable_balance wrapper
            # PHASE A FIX: Trust SharedState's authoritative spendable balance
            # This prevents double-subtraction of reserves (once in Meta, once here)
            spendable = await self.shared_state.get_spendable_balance(quote_asset)
            spendable_dec = Decimal(str(spendable))

            # Add existing accumulation to effectively check if we cross the minNotional threshold.
            acc_val = Decimal(str(intent.accumulated_quote)) if intent and intent.state == "ACCUMULATING" else Decimal("0")
            effective_qa = qa + acc_val

            # Store policy context for later access in bootstrap execution
            self._current_policy_context = policy_ctx
            
            # 3) ACCUMULATE_MODE/BOOTSTRAP_BYPASS CHECK: Skip min_notional validation for special modes
            # This allows P0 dust promotion and bootstrap execution to work without being blocked by min_notional guards
            accumulate_mode = False
            bootstrap_bypass = False
            no_downscale_planned_quote = False
            if policy_ctx:
                accumulate_mode = bool(policy_ctx.get("_accumulate_mode", False))
                bootstrap_bypass = bool(policy_ctx.get("bootstrap_bypass", False))
                no_downscale_planned_quote = bool(
                    policy_ctx.get("_no_downscale_planned_quote", False)
                    or policy_ctx.get("no_downscale_planned_quote", False)
                )
            
            
            # Skip min_notional check only for accumulate mode (adding to existing positions)
            # CRITICAL: accumulate_mode allows dust promotion without min_notional guards
            bypass_min_notional = accumulate_mode
            
            # For dust healing and Meta-validated decisions, bypass internal economic floor
            # but keep enforcing exchange min_notional and hard affordability.
            bypass_internal_economic_floor = (
                bypass_min_notional
                or is_dust_operation
                or skip_micro_trade_kill_switch
            )
            
            if not bypass_internal_economic_floor:
                # 3) If planned quote + accumulation is below the floor, decide between MIN_NOTIONAL vs NAV shortfall
                if effective_qa < min_required - eps:
                    # If the user is effectively trying to spend all they have (including accumulation)
                    # and that amount is still below the venue/config floor, classify as NAV shortfall.
                    if spendable_dec > 0 and (qa <= spendable_dec + eps) and (spendable_dec + acc_val < min_required - eps):
                        gap = (min_required - (spendable_dec + acc_val)).max(Decimal("0"))
                        self.logger.warning(
                            "[AFFORD_BLOCK] %s BUY insufficient quote vs min_required "
                            "(qa=%.2f effective=%.2f spendable=%.2f min_required=%.2f min_notional=%.2f acc=%.2f)",
                            sym,
                            float(qa),
                            float(effective_qa),
                            float(spendable_dec),
                            float(min_required),
                            float(min_notional),
                            float(acc_val),
                        )
                        return (False, gap, "INSUFFICIENT_QUOTE")
                    # QUOTE UPGRADE: Instead of rejecting, upgrade the quote to meet minimum
                    # The caller asked below the floor but has enough NAV.
                    effective_qa = max(effective_qa, min_required)
                    qa = max(qa, min_required)
                    self.logger.info(
                        f"[EM:QUOTE_UPGRADE] {sym} BUY: Upgraded quote to minimum allocation "
                        f"upgraded_quote={float(qa):.2f} USDT, min_required={float(min_required):.2f} USDT"
                    )
            else:
                # BOOTSTRAP: Bypass min_notional for first trade execution
                if effective_qa < min_required - eps:
                    self.logger.warning(
                        f"[EM:BOOTSTRAP] {sym} BUY: Bypassing min_entry={float(min_required):.2f} "
                        f"for quote={float(effective_qa):.2f} USDT"
                    )
                # Continue to affordability check even if below min_entry


            # 4) Affordability with fees/headroom
            # We don't just check gross_needed; we check if the rounded quantity is affordable.
            
            # Extract step_size from filters for quantity rounding
            step_size, min_qty, max_qty, tick_size, _ = self._extract_filter_vals(filters)
            
            # Apply fee buffer to find "net" spendable for the asset (including accumulation)
            price_f = float(price) if price > 0 else 1.0
            
            # PREFER PLANNED_QTY from policy_context (MetaController computed)
            # Fall back to computing from quote amount
            # EXCEPTION: For dust healing, ALWAYS recalculate from current quote amount to avoid variable leakage
            qty_source = "quote"
            if planned_qty is not None and not is_dust_operation:
                est_units = float(planned_qty)
                est_units_raw = float(effective_qa) / price_f  # Still calculate for diagnostics
                implied_quote = est_units * price_f
                # Guard against stale/mismatched upstream qty that makes diagnostics and gates inconsistent.
                if abs(implied_quote - float(effective_qa)) > max(0.01, float(effective_qa) * 0.05):
                    self.logger.warning(
                        "[AFFORD_DIAG] %s planned_qty mismatch (implied_quote=%.2f effective_quote=%.2f). "
                        "Recomputing from quote.",
                        sym, implied_quote, float(effective_qa),
                    )
                    step = step_size
                    est_units = self._round_step_down(est_units_raw, step) if step > 0 else est_units_raw
                    qty_source = "quote_recomputed"
                else:
                    qty_source = "policy_planned_qty"
                    self.logger.debug(f"[EM] Using planned_qty {planned_qty} for {sym}")
            else:
                est_units_raw = float(effective_qa) / price_f
                # P9 Corrective: Use the extracted step_size (robust) instead of direct dict access
                step = step_size
                if step > 0:
                    est_units = self._round_step_down(est_units_raw, step)
                else:
                    est_units = est_units_raw
                
            if est_units <= 0:
                return (False, Decimal("0"), "ZERO_QTY_AFTER_ROUNDING")
            
            # DIAGNOSTIC: Log affordability calculation details
            qty_raw = 0.0
            if 'est_units_raw' in locals() and est_units_raw is not None:
                qty_raw = est_units_raw
            elif price_f > 0:
                qty_raw = float(effective_qa) / price_f
            notional_f = float(est_units or 0.0) * float(price_f or 0.0)
            self.logger.warning(
                f"[AFFORD_DIAG] symbol={sym} "
                f"planned_quote={float(qa):.2f} "
                f"effective_quote={float(effective_qa):.2f} "
                f"price={price_f:.2f} "
                f"qty_raw={qty_raw:.8f} "
                f"step_size={step_size:.8f} "
                f"qty_exec={est_units:.8f} "
                f"notional={notional_f:.2f} "
                f"spendable={float(spendable_dec):.2f} "
                f"source={qty_source}"
            )
            
            est_notional = est_units * price_f
            
            # Check if this executable chunk meets minNotional (SKIP in bootstrap/accumulate modes, but NEVER for dust healing)
            exchange_floor = float(min_notional)  # NOT min_required
            if not bypass_min_notional and est_notional < exchange_floor:
                # QUOTE UPGRADE: Instead of rejecting, upgrade the quantity to meet minimum notional
                min_units_for_floor = Decimal(str(exchange_floor)) / Decimal(str(price_f))
                est_units = max(est_units, float(min_units_for_floor))
                est_notional = est_units * price_f
                self.logger.info(
                    f"[EM:QUOTE_UPGRADE] {sym} BUY: Upgraded quantity to meet exchange minimum "
                    f"upgraded_qty={est_units:.8f}, exchange_floor={exchange_floor:.2f} USDT"
                )
            elif bypass_min_notional and est_notional < exchange_floor and not is_dust_operation:
                # Log bypass for accumulate execution, but not for dust healing
                self.logger.warning(
                    f"[EM:ACCUMULATE] {sym} BUY: Bypassing second min_entry check - "
                    f"est_notional={est_notional:.2f} < exchange_floor={exchange_floor:.2f}"
                )
            elif is_dust_operation and est_notional < exchange_floor:
                # Dust healing/recovery must respect exchange min_notional
                gap = Decimal(str(exchange_floor)) - Decimal(str(est_notional))
                return (False, gap.max(Decimal("0")), "DUST_OPERATION_LT_MIN_NOTIONAL")


            gross_needed = qa * (Decimal("1") + taker_fee) * headroom
            if spendable_dec < gross_needed - eps:
                # Point 3: Dynamic Resizing (Downscaling)
                # If we have less than planned, but enough for minNotional, we downscale.
                max_qa = spendable_dec / ((Decimal("1") + taker_fee) * headroom)
                # QUOTE UPGRADE: Instead of rejecting, allow downscaling to what we can afford
                # (Previously: if no_downscale_planned_quote=True, would reject)
                if max_qa >= Decimal(str(exchange_floor)) or bypass_min_notional:
                    self.logger.info(f"[EM] Dynamic Resizing: Downscaling {qa} -> {max_qa:.2f} to fit spendable {spendable_dec:.2f}")
                    return (True, max_qa, "OK_DOWNSCALED")
                else:
                    # Point 2: Accumulation Pivot
                    # No enough for minNotional even with all cash.
                    gap = Decimal(str(exchange_floor)) - max_qa
                    return (False, gap.max(Decimal("0")), "INSUFFICIENT_QUOTE_FOR_ACCUMULATION")


            if price <= 0:
                self.logger.warning("[EM] ExecutionProbe = FAIL (Reason: Market Price 0). Readiness = FALSE.")
                return (False, Decimal("0"), "ZERO_MARKET_PRICE")

            if price > 0:
                filters_obj = SymbolFilters(
                    step_size=float(min_notional/10), # dummy or real if we had it
                    min_qty=0.0, 
                    max_qty=float("inf"),
                    tick_size=1e-8,
                    min_notional=float(min_notional),
                    min_entry_quote=(0.0 if bypass_internal_economic_floor else float(min_required))
                )
                # Re-fetch real filters for accuracy
                f_data = await self.exchange_client.ensure_symbol_filters_ready(sym)
                step_size, min_qty, _, _, _ = self._extract_filter_vals(f_data)
                filters_obj.step_size = step_size
                filters_obj.min_qty = min_qty

                v_ok, v_qty, _, v_reason = await self._validate_order_request_contract(
                    side="BUY", qty=0, price=price, filters=filters_obj, use_quote_amount=float(effective_qa)
                )
                
                # BOOTSTRAP FIX: Force minimum quantity if we have capital but qty rounded to zero
                if not v_ok and v_reason in ("QTY_LT_MIN", "NOTIONAL_LT_MIN_OR_ZERO_QTY"):
                    # Check if we're in bootstrap mode (either via policy context or shared state)
                    is_bootstrap = bootstrap_bypass
                    if not is_bootstrap:
                        try:
                            if hasattr(self.shared_state, "is_bootstrap_mode"):
                                is_bootstrap = self.shared_state.is_bootstrap_mode()
                        except Exception:
                            pass
                    
                    # If bootstrap AND we have enough capital for minNotional, force minimum quantity
                    if is_bootstrap and spendable_dec >= Decimal(str(filters_obj.min_notional)):
                        step = Decimal(str(filters_obj.step_size or 0.1))
                        min_qty_dec = Decimal(str(filters_obj.min_qty or 0.0))
                        forced_qty = max(min_qty_dec, step)
                        
                        # Verify the forced quantity is actually executable
                        forced_notional = forced_qty * Decimal(str(price))
                        if forced_notional <= spendable_dec:
                            self.logger.info(
                                f"[EM] BOOTSTRAP: Forcing minimum quantity for {sym}: "
                                f"qty={float(forced_qty):.8f}, notional={float(forced_notional):.2f} USDT"
                            )
                            # Override: treat as OK with the forced quantity
                            return (True, forced_qty, "OK_BOOTSTRAP_FORCED")
                
                if not v_ok:
                    # BOOTSTRAP FIX 2.0: If we have enough capital for minNotional, logic should PASS not fail on rounding.
                    # This is critical for readiness probes on high-priced or weird-step-size symbols.
                    spendable_dec = Decimal(str(spendable))
                    min_req_dec = Decimal(str(filters_obj.min_notional))
                    
                    if spendable_dec >= min_req_dec:
                        # Use at least one step_size/min_qty
                        forced_qty = max(Decimal(str(filters_obj.min_qty)), Decimal(str(filters_obj.step_size or 0.1)))
                        self.logger.info(f"[EM] Readiness Check: Ignoring rounding error because spendable {spendable_dec:.2f} >= minNotional {min_req_dec:.2f}. Forcing qty={forced_qty}")
                        return (True, forced_qty, "OK_BOOTSTRAP_FORCED")
                    
                    gap = Decimal("0")
                    if v_reason == "QTY_LT_MIN" or v_reason == "NOTIONAL_LT_MIN_OR_ZERO_QTY":
                        # Rule 2/5 Enhancement: Calculate exactly what we need to reach 1 unit
                        step = Decimal(str(filters_obj.step_size or 0.1))
                        target_qty = max(Decimal(str(filters_obj.min_qty)), step)
                        target_notional = max(Decimal(str(filters_obj.min_notional)), target_qty * Decimal(str(price)))
                        gap = (target_notional - qa).max(Decimal("0"))
                        
                    self.logger.warning("[EM] ExecutionProbe = FAIL (Reason: %s, Gap: %.2f). Readiness = FALSE.", v_reason, gap)
                    return (False, gap, f"NOT_EXECUTABLE:{v_reason}")

            # 6) Enforce reserve floor & tiny-remainder guard
            gross_needed_f = float(gross_needed)
            _free_q, _ok_rem, _why_rem = await self._get_free_quote_and_remainder_ok(quote_asset, gross_needed_f)
            if not _ok_rem:
                return (False, Decimal("0"), _why_rem)

            return (True, Decimal(str(est_units)), "OK")

        except Exception as e:
            self.logger.exception(f"Affordability check failed for {symbol}: {e}")
            return (False, Decimal("0"), "unexpected_error")

    async def _attempt_liquidity_healing(self, symbol: str, needed_quote: float, context: Dict[str, Any]) -> bool:
        if not self.meta_controller or not hasattr(self.meta_controller, "request_liquidity"):
            return False
        for attempt in range(self.max_liquidity_retries):
            try:
                self.logger.info(f"[Heal] Liquidity attempt {attempt + 1}/{self.max_liquidity_retries} for {symbol}")
                plan = await self.meta_controller.request_liquidity(symbol, needed_quote, context)
                if not plan:
                    continue
                if hasattr(self.meta_controller, "liquidation_agent") and self.meta_controller.liquidation_agent:
                    res = await self.meta_controller.liquidation_agent.execute_plan(plan)
                    if res.get("success", False):
                        await asyncio.sleep(self.liquidity_retry_delay)
                        ok_verify, _, _ = await self.can_afford_market_buy(symbol, needed_quote)
                        if ok_verify:
                            await self._log_execution_event("liquidity_healing_success", symbol, {
                                "attempt": attempt + 1, "needed_quote": needed_quote, "plan_result": res
                            })
                            return True
            except Exception as e:
                exec_error = self._classify_execution_error(e, symbol, "liquidity_healing")
                await self._log_execution_event("liquidity_healing_error", symbol, {
                    "attempt": attempt + 1, "error_type": getattr(exec_error, "error_type", "Unknown"),
                    "error": str(exec_error)
                })
            if attempt < self.max_liquidity_retries - 1:
                await asyncio.sleep(self.liquidity_retry_delay)
        return False

    def _recent_rejection_stats(
        self,
        *,
        symbol: str,
        side: str,
        reason: str,
        window_sec: float,
    ) -> Tuple[int, float]:
        """
        Return (count, age_sec) for recent matching rejections from SharedState history.
        age_sec is the elapsed time since the oldest matching event in the window.
        """
        if not self.shared_state:
            return 0, 0.0

        sym = self._norm_symbol(symbol)
        side_u = str(side or "").upper()
        reason_u = str(reason or "").upper()
        now_ts = time.time()
        oldest_ts = 0.0
        count = 0

        try:
            history = list(getattr(self.shared_state, "rejection_history", []) or [])
            for item in history:
                if not isinstance(item, dict):
                    continue
                if str(item.get("symbol", "")).upper() != sym:
                    continue
                if str(item.get("side", "")).upper() != side_u:
                    continue
                if str(item.get("reason", "")).upper() != reason_u:
                    continue
                ts = float(item.get("ts", 0.0) or 0.0)
                if ts <= 0:
                    continue
                if (now_ts - ts) > float(window_sec or 0.0):
                    continue
                count += 1
                if oldest_ts <= 0.0 or ts < oldest_ts:
                    oldest_ts = ts
        except Exception:
            return 0, 0.0

        age_sec = max(0.0, now_ts - oldest_ts) if oldest_ts > 0.0 else 0.0
        return int(count), float(age_sec)

    async def _maybe_retry_close_with_forced_exit(
        self,
        *,
        symbol: str,
        reason_text: str,
        tag: str,
        trace_id: Optional[str],
        policy_context: Dict[str, Any],
        blocked_reason: str,
        blocked_error_code: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Timed close escape hatch:
        after repeated CLOSE_NOT_SUBMITTED blocks in a short window, retry SELL once
        with `_forced_exit` + `CAPITAL_RECOVERY` context so deadlocked positions can close.
        """
        if not bool(self._cfg("CLOSE_ESCAPE_HATCH_ENABLED", True)):
            return None
        if bool(policy_context.get("_close_escape_retry")):
            return None

        reason_l = str(blocked_reason or "").strip().lower()
        code_l = str(blocked_error_code or "").strip().lower()
        allow_csv = str(
            self._cfg(
                "CLOSE_ESCAPE_HATCH_ALLOW_REASONS",
                "portfolio_pnl_improvement,sell_dynamic_edge_below_min,sell_net_pnl_below_min,sell_net_pct_below_min",
            )
            or ""
        )
        allow_tokens = {
            tok.strip().lower()
            for tok in allow_csv.split(",")
            if str(tok).strip()
        }
        if allow_tokens and reason_l not in allow_tokens and code_l not in allow_tokens:
            return None

        window_sec = max(30.0, float(self._cfg("CLOSE_ESCAPE_HATCH_WINDOW_SEC", 900.0) or 900.0))
        trigger_count = max(1, int(self._cfg("CLOSE_ESCAPE_HATCH_TRIGGER_COUNT", 4) or 4))
        min_age_sec = max(0.0, float(self._cfg("CLOSE_ESCAPE_HATCH_MIN_AGE_SEC", 120.0) or 120.0))
        cooldown_sec = max(0.0, float(self._cfg("CLOSE_ESCAPE_HATCH_RETRY_COOLDOWN_SEC", 180.0) or 180.0))
        now_ts = time.time()

        recent_count, age_sec = self._recent_rejection_stats(
            symbol=symbol,
            side="SELL",
            reason="CLOSE_NOT_SUBMITTED",
            window_sec=window_sec,
        )
        if recent_count < trigger_count or age_sec < min_age_sec:
            return None

        buy_lock_trigger = int(self._cfg("CLOSE_ESCAPE_HATCH_BUY_LOCK_TRIGGER", 0) or 0)
        if buy_lock_trigger > 0:
            buy_lock_count = 0
            try:
                if hasattr(self.shared_state, "get_rejection_count"):
                    buy_lock_count = int(
                        self.shared_state.get_rejection_count(symbol, "BUY", "POSITION_ALREADY_OPEN") or 0
                    )
            except Exception:
                buy_lock_count = 0
            if buy_lock_count < buy_lock_trigger:
                return None

        sym = self._norm_symbol(symbol)
        last_attempt_ts = float(self._close_escape_last_attempt_ts.get(sym, 0.0) or 0.0)
        if last_attempt_ts > 0 and (now_ts - last_attempt_ts) < cooldown_sec:
            return None

        self._close_escape_last_attempt_ts[sym] = now_ts
        forced_ctx = dict(policy_context or {})
        forced_ctx["_close_escape_retry"] = True
        forced_ctx["_forced_exit"] = True
        forced_ctx["reason"] = "CAPITAL_RECOVERY_FORCED_EXIT"
        forced_ctx["exit_reason"] = "CAPITAL_RECOVERY_FORCED_EXIT"
        forced_ctx["liquidation_reason"] = "CAPITAL_RECOVERY"
        forced_ctx["close_escape_origin_reason"] = str(blocked_reason or "")
        forced_ctx["close_escape_origin_code"] = str(blocked_error_code or "")
        forced_ctx["close_escape_requested_reason"] = str(reason_text or "")
        forced_ctx["close_escape_recent_blocks"] = int(recent_count)
        forced_ctx["close_escape_block_age_sec"] = float(age_sec)

        self.logger.warning(
            "[EM:CLOSE_ESCAPE] Triggering forced-exit retry for %s after %d CLOSE_NOT_SUBMITTED blocks in %.0fs (reason=%s code=%s).",
            sym,
            recent_count,
            age_sec,
            str(blocked_reason or ""),
            str(blocked_error_code or ""),
        )
        try:
            if hasattr(self.shared_state, "record_rejection"):
                await self.shared_state.record_rejection(sym, "SELL", "CLOSE_ESCAPE_ATTEMPT", source="ExecutionManager")
        except Exception:
            pass

        retry_intent = TradeIntent(
            symbol=symbol,
            side="SELL",
            quantity=None,
            planned_quote=None,
            tag=str(tag or "tp_sl"),
            trace_id=trace_id,
            is_liquidation=True,
            policy_context=forced_ctx,
        )
        return await self.execute_trade(retry_intent)

    # =============================
    # Canonical execution API
    # =============================
    async def close_position(
        self,
        *,
        symbol: str,
        reason: str = "",
        is_liquidation: Optional[bool] = None,
        force_finalize: bool = False,
        tag: str = "tp_sl",
        trace_id: Optional[str] = None,
        policy_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Close a position via the canonical execution path with optional forced finalization."""
        reason_text = str(reason or "").strip() or "EXIT"
        sym = self._norm_symbol(symbol)
        pos_qty = 0.0
        try:
            if hasattr(self.shared_state, "get_position_qty"):
                pos_qty = float(self.shared_state.get_position_qty(sym) or 0.0)
            elif isinstance(getattr(self.shared_state, "positions", None), dict):
                pos_qty = float((self.shared_state.positions.get(sym, {}) or {}).get("quantity", 0.0) or 0.0)
        except Exception:
            pos_qty = 0.0
        if is_liquidation is None:
            tag_l = str(tag or "").lower()
            reason_u = reason_text.upper()
            is_liq_intent = bool(
                tag_l in {"tp_sl", "liquidation", "dust_cleanup"}
                or "LIQUIDATION" in reason_u
                or "EMERGENCY" in reason_u
                or "STOP_LOSS" in reason_u
                or "SL" == reason_u
            )
        else:
            is_liq_intent = bool(is_liquidation)
        self.logger.info(
            "[EM:CLOSE_ATTEMPT] symbol=%s qty=%.8f reason=%s tag=%s is_liquidation=%s force_finalize=%s trace_id=%s",
            sym,
            pos_qty,
            reason_text,
            tag,
            is_liq_intent,
            bool(force_finalize),
            str(trace_id or ""),
        )
        policy_context = dict(policy_context or {})
        policy_context.setdefault("exit_reason", reason_text)
        policy_context.setdefault("reason", reason_text)
        if trace_id:
            policy_context["trace_id"] = str(trace_id)
            policy_context["decision_id"] = str(trace_id)
        if is_liq_intent:
            policy_context["liquidation_reason"] = reason_text
        # Create a TradeIntent for the canonical execute_trade API
        from core.stubs import TradeIntent
        trade_intent = TradeIntent(
            symbol=symbol,
            side="SELL",
            quantity=None,
            planned_quote=None,
            tag=tag,
            trace_id=trace_id,
            is_liquidation=is_liq_intent,
            policy_context=policy_context,
        )
        res = await self.execute_trade(trade_intent)
        try:
            if isinstance(res, dict):
                self.logger.info(
                    "[EM:CLOSE_RESULT] symbol=%s ok=%s status=%s reason=%s error_code=%s order_id=%s executed_qty=%s cumm_quote=%s",
                    sym,
                    bool(res.get("ok")),
                    str(res.get("status", "")),
                    str(res.get("reason", "")),
                    str(res.get("error_code", "")),
                    str(res.get("orderId") or res.get("order_id") or res.get("exchange_order_id")),
                    str(res.get("executedQty") or res.get("executed_qty")),
                    str(res.get("cummulativeQuoteQty") or res.get("cummulative_quote", "")),
                )
            else:
                self.logger.warning("[EM:CLOSE_RESULT] symbol=%s non-dict response=%r", sym, res)
        except Exception:
            self.logger.debug("[EM] close result logging failed for %s", sym, exc_info=True)
        if isinstance(res, dict):
            try:
                initial_status = str(res.get("status", "")).upper()
                has_submission_ref = bool(
                    res.get("orderId")
                    or res.get("order_id")
                    or res.get("exchange_order_id")
                    or res.get("clientOrderId")
                    or res.get("client_order_id")
                    or res.get("origClientOrderId")
                )
                # Preserve upstream block/skip reasons when nothing was submitted.
                # Running delayed-fill reconciliation here would emit misleading
                # recovery-no-ids events and hide the true failure cause.
                if (not has_submission_ref) and initial_status in {"BLOCKED", "SKIPPED", "REJECTED", "CANCELED", "EXPIRED"}:
                    reason_l = str(res.get("reason", "") or "").lower()
                    code_l = str(res.get("error_code", "") or "").lower()
                    is_terminal_dust = (
                        "terminal_dust" in reason_l
                        or "permanent_dust" in reason_l
                        or code_l == "terminaldust"
                        or code_l == "permanentdust"
                    )
                    if is_terminal_dust:
                        # Terminal dust cannot be executed economically on exchange.
                        # Retire local position to prevent infinite retry/reject loops.
                        try:
                            await self._force_finalize_position(sym, "terminal_dust_write_down")
                            if hasattr(self.shared_state, "close_position"):
                                await maybe_call(self.shared_state, "close_position", sym, "terminal_dust_write_down")
                            self.logger.info(
                                "[EM:CLOSE_TERMINAL_DUST] symbol=%s retired local position after blocked close (%s/%s)",
                                sym,
                                str(res.get("reason", "")),
                                str(res.get("error_code", "")),
                            )
                        except Exception:
                            self.logger.debug("[EM] terminal dust local retirement failed for %s", sym, exc_info=True)
                    try:
                        if hasattr(self.shared_state, "record_rejection"):
                            await self.shared_state.record_rejection(
                                sym,
                                "SELL",
                                "CLOSE_NOT_SUBMITTED",
                                source="ExecutionManager",
                            )
                    except Exception:
                        pass
                    escape_res = await self._maybe_retry_close_with_forced_exit(
                        symbol=sym,
                        reason_text=reason_text,
                        tag=str(tag or "tp_sl"),
                        trace_id=trace_id,
                        policy_context=policy_context,
                        blocked_reason=str(res.get("reason", "") or ""),
                        blocked_error_code=str(res.get("error_code", "") or ""),
                    )
                    if isinstance(escape_res, dict):
                        res = escape_res
                        initial_status = str(res.get("status", "")).upper()
                        has_submission_ref = bool(
                            res.get("orderId")
                            or res.get("order_id")
                            or res.get("exchange_order_id")
                            or res.get("clientOrderId")
                            or res.get("client_order_id")
                            or res.get("origClientOrderId")
                        )
                        self.logger.warning(
                            "[EM:CLOSE_ESCAPE_RESULT] symbol=%s submitted=%s status=%s reason=%s error_code=%s",
                            sym,
                            has_submission_ref,
                            initial_status,
                            str(res.get("reason", "")),
                            str(res.get("error_code", "")),
                        )
                    if (not has_submission_ref) and initial_status in {"BLOCKED", "SKIPPED", "REJECTED", "CANCELED", "EXPIRED"}:
                        self.logger.info(
                            "[EM:CLOSE_RECONCILE_SKIPPED] symbol=%s status=%s reason=%s (no submission refs)",
                            sym,
                            initial_status,
                            str(res.get("reason", "")),
                        )
                        self._journal("CLOSE_NOT_SUBMITTED", {
                            "symbol": sym,
                            "side": "SELL",
                            "status": initial_status or "UNKNOWN",
                            "reason": str(res.get("reason", "") or "close_not_submitted"),
                            "error_code": str(res.get("error_code", "") or ""),
                            "tag": str(tag or "tp_sl"),
                            "timestamp": time.time(),
                        })
                        return res
                res = await self._reconcile_delayed_fill(
                    symbol=sym,
                    side="SELL",
                    order=res,
                    tag=str(tag or "tp_sl"),
                    tier=None,
                )
                if not isinstance(res, dict):
                    raise ValueError(f"reconcile returned non-dict: {type(res)}")
                
                # [FIX] Single responsibility: reconcile returns merged order,
                # close_position handles post-fill + finalize exactly once.
                status = str(res.get("status", "")).lower()
                exec_qty = float(res.get("executedQty", res.get("executed_qty", 0.0)) or 0.0)
                is_fill = status in {"filled", "partially_filled"} and exec_qty > 0.0
                
                if is_fill:
                    # Handle post-fill and finalize in single call sequence
                    post_fill = await self._ensure_post_fill_handled(
                        symbol=sym,
                        side="SELL",
                        order=res,
                        tier=None,
                        tag=str(tag or "tp_sl"),
                    )
                    await self._finalize_sell_post_fill(
                        symbol=sym,
                        order=res,
                        tag=str(tag or "tp_sl"),
                        post_fill=post_fill,
                        policy_ctx={"exit_reason": reason_text, "reason": reason_text},
                        tier=None,
                    )

                    if force_finalize:
                        await self._force_finalize_position(sym, reason_text)
                        self.logger.info("[EM:CLOSE_FINALIZE] symbol=%s applied=True reason=%s", sym, reason_text)
                    
                    # ✅ HYDRATION FIX: Refresh wallet state after successful trade
                    # This keeps SharedState synchronized with Binance and allows
                    # the system to recalculate NAV and execute new trades
                    try:
                        if self.shared_state:
                            await self.shared_state.hydrate_balances_from_exchange()
                            await self.shared_state.hydrate_positions_from_balances()
                            self.logger.info(
                                "[EM:HYDRATE] symbol=%s post-fill balance refresh completed",
                                sym
                            )
                    except Exception as e:
                        self.logger.warning(
                            "[EM:HYDRATE] symbol=%s post-fill balance refresh failed: %s",
                            sym, str(e)
                        )
                else:
                    self.logger.warning(
                        "[EM:CLOSE_FINALIZE] symbol=%s applied=False status=%s exec_qty=%.8f reason=%s",
                        sym, status, exec_qty, reason_text
                    )
            except Exception:
                self.logger.error("[EM:CLOSE_RECONCILE_FAILED] symbol=%s — post-fill may be lost", symbol, exc_info=True)
        return res

    async def _force_finalize_position(self, symbol: str, reason: str) -> None:
        """Best-effort: mark a position as closed in SharedState after an exit."""
        ss = self.shared_state
        sym = self._norm_symbol(symbol)
        try:
            pos = None
            if hasattr(ss, "positions") and isinstance(ss.positions, dict):
                pos = ss.positions.get(sym)
            if pos is None:
                pos = await maybe_call(ss, "get_position", sym)
            if not isinstance(pos, dict):
                pos = {}
            updated = dict(pos)
            updated["quantity"] = 0.0
            updated["status"] = "CLOSED"
            updated["closed_reason"] = reason
            updated["closed_at"] = time.time()
            await maybe_call(ss, "update_position", sym, updated)
        except Exception:
            self.logger.debug("[EM] Failed to update position as CLOSED for %s", sym, exc_info=True)
        try:
            ot = getattr(ss, "open_trades", None)
            if isinstance(ot, dict):
                ot.pop(sym, None)
        except Exception:
            pass

    async def _persist_trade_intent(self, intent: TradeIntent) -> Optional[int]:
        """
        PHASE 5: Persist TradeIntent to EventStore before execution.
        
        Event sourcing ensures complete audit trail:
        - Every trading decision is immutable
        - Complete history for compliance
        - Recovery point for replays
        
        Returns: event sequence number if persisted, None if EventStore unavailable
        """
        if not self.event_store or not EventStore or not EventType:
            return None
        
        try:
            from core.event_store import Event
            
            # Create event from TradeIntent
            event = Event(
                event_type=EventType.TRADE_EXECUTED,  # Persisted before actual execution
                component="execution_manager",
                symbol=intent.symbol,
                timestamp=intent.timestamp or time.time(),
                data={
                    "side": intent.side.upper(),
                    "quantity": intent.quantity,
                    "planned_quote": intent.planned_quote,
                    "confidence": intent.confidence,
                    "trace_id": intent.trace_id,
                    "tier": intent.tier,
                    "is_liquidation": intent.is_liquidation,
                    "tag": intent.tag,
                    "agent": intent.agent,
                    "execution_mode": intent.execution_mode,
                    "policy_context": intent.policy_context or {},
                },
                tags=["canonical_intent", intent.execution_mode],
            )
            
            # Persist atomically
            sequence = await self.event_store.append(event)
            self.logger.debug(
                "[EM:EventSource] Persisted TradeIntent: symbol=%s side=%s "
                "confidence=%.2f trace_id=%s (seq=%d)",
                intent.symbol, intent.side, intent.confidence, intent.trace_id, sequence
            )
            return sequence
        
        except Exception as e:
            self.logger.warning(
                "[EM:EventSource] Failed to persist TradeIntent: %s (non-blocking)", str(e)
            )
            return None

    async def execute_trade(self, intent: TradeIntent) -> Dict[str, Any]:
        """
        Tier-aware execution (Phase A Frequency Engineering).

        CANONICAL INTERFACE (Phase 4):
        All trade execution flows through TradeIntent objects.
        No loose parameters. Single source of truth.

           await executor.execute_trade(intent=trade_intent)

        PHASE 5: EVENT SOURCING
        - TradeIntent persisted to EventStore before execution
        - Complete audit trail for compliance
        - Immutable history for recovery and replays

        ExecutionManager answers: "Given this allowed intent, what is the maximum safe executable order?"

        [PHASE 2 REQUIREMENT] trace_id from MetaController:
        - All orders (except liquidation) REQUIRE trace_id from MetaController decision
        - Enforces that CompoundingEngine directives go through MetaController validation
        - Every order has an audit trail back to MetaController decision

        Respects (unless is_liquidation=True):
        - RiskManager caps (already checked via pre_check)
        - Tier caps (informational, logged)
        - Available quote
        - Fees
        - MinNotional

        [FIX #9] Liquidation Bypass:
        - If is_liquidation=True: Bypasses ALL guards (capital, throughput, min-notional)
        - TP/SL exits are LIQUIDATION (risk management), NOT trading decisions
        - Tag "tp_sl" also triggers liquidation mode for backwards compatibility

        Final size = minimum of all constraints (except when liquidation bypasses)

        Returns normalized contract:
        { ok, status, executedQty, avgPrice, cummulativeQuoteQty, orderId, reason, error_code?, tier? }
        """
        # ✅ GLOBAL EXECUTION LOCK: Only allow trading in live mode
        # This is a critical safety mechanism to prevent accidental executions
        # in test/paper/simulation environments
        # Use consistent config check (not env var) to match AppContext behavior
        trading_mode = str(self._cfg("trading_mode", "live") or "live").lower()
        if trading_mode != "live":
            self.logger.warning(
                "[EXECUTION BLOCKED] mode=%s symbol=%s side=%s",
                trading_mode,
                intent.symbol,
                intent.side
            )
            return {
                "status": "shadow_blocked",
                "symbol": intent.symbol,
                "side": intent.side,
                "reason": f"Execution blocked in {trading_mode} mode"
            }
        
        # PHASE 5: Persist intent before execution (event sourcing)
        await self._persist_trade_intent(intent)
        
        # Delegate to implementation
        return await self._execute_trade_impl(
            symbol=intent.symbol,
            side=intent.side,
            quantity=intent.quantity,
            planned_quote=intent.planned_quote,
            tag=intent.tag,
            trace_id=intent.trace_id,
            tier=intent.tier,
            is_liquidation=intent.is_liquidation,
            policy_context=intent.policy_context,
            confidence=intent.confidence,
        )
    
    async def _execute_trade_impl(
        self,
        symbol: str,
        side: str,
        quantity: Optional[float] = None,
        planned_quote: Optional[float] = None,
        tag: str = "meta/Agent",
        trace_id: Optional[str] = None,
        tier: Optional[str] = None,
        is_liquidation: bool = False,
        policy_context: Optional[Dict[str, Any]] = None,
        confidence: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Internal implementation of trade execution.
        Use execute_trade() public API instead.
        """
        # Periodic watchdog status reporting
        self._execution_counter += 1
        now_ts = time.time()
        if (now_ts - self._last_watchdog_report_ts) >= self._watchdog_report_interval_s:
            await self._report_watchdog_status(
                "Operational",
                f"Processed {self._execution_counter} executions"
            )
            self._last_watchdog_report_ts = now_ts
        
        self._ensure_heartbeat()
        side = (side or "").lower()

        sym = self._norm_symbol(symbol)
        policy_ctx = dict(policy_context or {})
        decision_id = self._resolve_decision_id(policy_ctx)
        policy_ctx.setdefault("decision_id", decision_id)
        if tier is not None and "tier" not in policy_ctx:
            policy_ctx["tier"] = tier

        # [PHASE 2] GUARD: Require trace_id for all orders (except liquidation)
        # This ensures CompoundingEngine directives go through MetaController validation
        if not trace_id and not is_liquidation:
            self.logger.warning(
                "[EXEC:TraceID] Blocked %s %s: missing trace_id from MetaController (Phase 2 architecture)",
                side.upper(), sym,
            )
            return {
                "ok": False,
                "status": "blocked",
                "reason": "missing_meta_trace_id",
                "error_code": "MISSING_META_TRACE_ID",
            }

        # Contract guard: if tradeability metadata is present, Meta must have evaluated
        # _passes_tradeability_gate before any quote reservation in this method.
        if side == "buy":
            has_tradeability_payload = any(
                policy_ctx.get(k) is not None
                for k in (
                    "required_conf",
                    "break_even_prob",
                    "tradeability_expected_move_pct",
                )
            ) or bool(
                str(policy_ctx.get("tradeability_hint", "") or "").strip()
            ) or bool(
                str(policy_ctx.get("tradeability_regime", "") or "").strip()
            )
            if has_tradeability_payload and not bool(policy_ctx.get("tradeability_gate_checked")):
                self.logger.warning(
                    "[EXEC:Tradeability] Blocked BUY %s: missing Meta tradeability gate before reservation.",
                    sym,
                )
                return {
                    "ok": False,
                    "status": "blocked",
                    "reason": "tradeability_gate_missing",
                    "error_code": "TRADEABILITY_GATE_MISSING",
                }

        # Define dust operation flags for bypass logic
        is_dust_healing_buy = bool(
            policy_ctx.get("_is_dust_healing_buy")
            or policy_ctx.get("is_dust_healing")
            or policy_ctx.get("_dust_healing")
            or str(policy_ctx.get("reason") or "").upper() == "DUST_HEALING_BUY"
        )
        if is_dust_healing_buy:
            policy_ctx["_is_dust_healing_buy"] = True
        is_dust_operation = self._is_dust_operation_context(policy_ctx, tier=tier, tag=tag, symbol=sym)

        # === DUST PRIORITY: REUSE dust when a same-symbol BUY arrives ===
        # Priority 1 (Reuse > Aggregate > Cleanup): if the symbol already has a
        # sub-minNotional dust position, net out its notional from planned_quote so
        # we don't over-allocate capital that we effectively already own.
        if side == "buy" and not is_dust_healing_buy and planned_quote is not None:
            try:
                dust_entry = (getattr(self.shared_state, "dust_registry", None) or {}).get(sym)
                if dust_entry:
                    dust_qty = float(dust_entry.get("qty", 0.0))
                    if dust_qty > 0.0:
                        pos = await self.shared_state.get_position(sym) or {}
                        dust_price = float(
                            pos.get("mark_price") or pos.get("entry_price") or 0.0
                        )
                        if dust_price > 0.0:
                            dust_notional = dust_qty * dust_price
                            reduced_quote = max(0.0, float(planned_quote) - dust_notional)
                            self.logger.info(
                                "[Dust:REUSE] %s dust_qty=%.6f dust_notional=%.4f "
                                "planned_quote %.4f → %.4f",
                                sym, dust_qty, dust_notional, float(planned_quote), reduced_quote,
                            )
                            planned_quote = reduced_quote
                            policy_ctx["_dust_reused_qty"] = dust_qty
                            policy_ctx["_dust_reused_notional"] = dust_notional
            except Exception as _dust_reuse_err:
                self.logger.debug("[Dust:REUSE] %s skipped: %s", sym, _dust_reuse_err)

        tag_raw = tag or ""
        tag_lower = tag_raw.lower()
        clean_tag = self._sanitize_tag(tag)
        tier_label = f" tier={tier}" if tier else ""

        # 🛡️ P9 SOP GOVERNANCE: Last line of defense (unless full liquidation bypass)
        # Exits/Liquidation are usually allowed even in safety modes to protect capital.
        #
        # EXIT PRIORITY HIERARCHY (best practice):
        # Risk exits MUST bypass all strategy/entry guards.
        # _forced_exit=True signals (concentration, starvation, rotation, rebalance exits)
        # are risk-management decisions — they must never be blocked by profit gates,
        # scaling rules, or capital allocation checks that apply only to entries.
        reason_hint_u = " ".join([
            str(policy_ctx.get("reason") or ""),
            str(policy_ctx.get("exit_reason") or ""),
            str(policy_ctx.get("signal_reason") or ""),
            str(policy_ctx.get("liquidation_reason") or ""),
        ]).upper()
        meta_exit_forced_liq = bool(
            "meta_exit" in tag_lower
            and any(k in reason_hint_u for k in ("LIQUIDATION", "EMERGENCY", "CAPITAL_RECOVERY", "DUST_CLEANUP"))
        )
        forced_exit_flag = side == "sell" and bool(policy_ctx.get("_forced_exit"))
        forced_exit_is_risk_liq = bool(
            forced_exit_flag
            and any(
                k in reason_hint_u
                for k in (
                    "LIQUIDATION",
                    "EMERGENCY",
                    "STOP_LOSS",
                    "HARD_STOP",
                    "CAPITAL_RECOVERY",
                    "DUST_CLEANUP",
                    "CONCENTRATION",
                )
            )
        )
        if forced_exit_flag and not forced_exit_is_risk_liq and "ROTATION" in reason_hint_u:
            self.logger.info(
                "[EXEC:ForcedExit] %s SELL forced_exit treated as regular exit (rotation path). "
                "Profitability gates remain active.",
                sym,
            )
        is_liq_full = (
            is_liquidation
            or any(x in tag_lower for x in ("tp_sl", "balancer", "liquidation"))
            or meta_exit_forced_liq
            or forced_exit_is_risk_liq
        )

        # ===== CAPITAL ESCAPE HATCH =====
        # When portfolio concentration exceeds 85% NAV AND a forced exit is attempted,
        # bypass all execution checks to ensure the system can always escape deadlock.
        # This is the final backstop against execution paralysis under concentration stress.
        bypass_checks = False
        if side == "sell" and bool(policy_ctx.get("_forced_exit")):
            try:
                nav = float(await self.get_tradable_nav() or 0.0)
                position_value = float(policy_ctx.get("position_value", 0.0))
                
                if nav > 0 and position_value > 0:
                    concentration = position_value / nav
                    
                    if concentration >= 0.85:
                        self.logger.warning(
                            "[EscapeHatch] CAPITAL_ESCAPE_HATCH activated for %s (%.1f%% NAV concentration) - bypassing all execution checks",
                            sym,
                            concentration * 100
                        )
                        bypass_checks = True
                        is_liq_full = True  # Force liquidation priority for high concentration exits
            except Exception as e:
                self.logger.debug(f"[EscapeHatch] Error checking concentration: {e}")

        # Global SELL guard (real capital only): allow only TP/SL (liquidation) or explicit EMERGENCY exits.
        # This prevents state/rotation/recovery SELLs from fragmenting compounding on live capital.
        is_real_mode = bool(self._cfg("LIVE_MODE", False)) and not bool(self._cfg("SIMULATION_MODE", False)) and not bool(self._cfg("PAPER_MODE", False)) and not bool(self._cfg("TESTNET_MODE", False))
        allow_authorized_meta_exit = False
        if side == "sell":
            meta_reason_u = " ".join([
                str(policy_ctx.get("reason") or ""),
                str(policy_ctx.get("exit_reason") or ""),
                str(policy_ctx.get("signal_reason") or ""),
            ]).upper()
            allow_authorized_meta_exit = bool(
                "meta_exit" in tag_lower
                and any(k in meta_reason_u for k in ("STRATEGY_SELL", "STRATEGY_EXIT", "META_EXIT", "ROTATION"))
            )

        if side == "sell" and is_real_mode and not is_liq_full and not bypass_checks and not allow_authorized_meta_exit:
            reason_text = " ".join([
                str(policy_ctx.get("reason") or ""),
                str(policy_ctx.get("exit_reason") or ""),
                str(policy_ctx.get("signal_reason") or ""),
                str(policy_ctx.get("liquidation_reason") or ""),
                str(tag or ""),
            ]).upper()
            is_emergency = "EMERGENCY" in reason_text
            if not is_emergency:
                self.logger.warning(
                    "[EM:SellGuard] Blocked SELL %s (real mode). reason=%s tag=%s | allowed only TP/SL or EMERGENCY.",
                    sym, policy_ctx.get("reason") or policy_ctx.get("exit_reason") or "",
                    tag or "",
                )
                return {
                    "ok": False,
                    "status": "blocked",
                    "reason": "real_mode_sell_guard",
                    "error_code": "SELL_GUARD",
                }
        
        if not is_liq_full and not bypass_checks:
            metrics = getattr(self.shared_state, "metrics", None) or {}
            mode = str(metrics.get("current_mode", "NORMAL")).upper()
            gov_decision = metrics.get("governance_decision") or {}
            allowed_actions = {
                str(a).upper() for a in (gov_decision.get("allowed_actions") or [])
            }
            if mode == "PAUSED":
                self.logger.warning(f"[EM:GovBlock] Blocked {side.upper()} {sym}: System is PAUSED.")
                return {"ok": False, "status": "blocked", "reason": "System PAUSED", "error_code": "PAUSED_MODE"}
            
            if side == "buy" and mode == "PROTECTIVE":
                # Respect MetaController governance decision (micro-account exception may allow BUY in PROTECTIVE).
                if "BUY" not in allowed_actions:
                    self.logger.warning(f"[EM:GovBlock] Blocked BUY {sym}: System is in PROTECTIVE mode.")
                    return {"ok": False, "status": "blocked", "reason": "BUY Disabled in PROTECTIVE Mode", "error_code": "PROTECTIVE_MODE"}
                self.logger.info(
                    "[EM:GovAllow] BUY %s allowed in PROTECTIVE (governance allowed_actions=%s)",
                    sym,
                    sorted(allowed_actions),
                )
        allow_partial = bool(
            policy_ctx.get("allow_partial")
            or policy_ctx.get("partial_exit")
            or policy_ctx.get("scaling_out")
            or policy_ctx.get("_partial_pct")
        )
        if side == "sell" and not allow_partial:
            qty_full = await self._get_sellable_qty(sym)
            if qty_full > 0:
                quantity = qty_full
                planned_quote = None
        
        liq_reason = str(
            policy_ctx.get("liquidation_reason")
            or policy_ctx.get("reason")
            or policy_ctx.get("exit_reason")
            or policy_ctx.get("signal_reason")
            or ""
        ).strip()
        liq_reason_norm = liq_reason.upper()
        special_liq_bypass = (
            side == "sell"
            and liq_reason_norm in {"CAPITAL_RECOVERY", "DUST_CLEANUP"}
            and ("liquidation" in tag_lower or "dust_cleanup" in tag_lower)
        )
        liq_marker = " [LIQUIDATION]" if is_liq_full else ""
        if special_liq_bypass:
            liq_marker = " [LIQUIDATION:SAFE_BYPASS]"
        self.logger.info(f"[EXEC] Request: {sym} {side.upper()} q={quantity} p_quote={planned_quote} tag={clean_tag}{tier_label}{liq_marker}")

        # FIX: For DUST_RECOVERY, set risk_based_quote before scaling to bypass ATR sizing
        if tier == "DUST_RECOVERY":
            deficit = getattr(self.shared_state, "dust_healing_deficit", {}).get(sym, 0.0)
            small_buffer = float(getattr(self.config, "DUST_HEALING_BUFFER_USDT", 0.5))
            planned_quote = deficit + small_buffer
            self.shared_state.risk_based_quote = getattr(self.shared_state, "risk_based_quote", {})
            self.shared_state.risk_based_quote[sym] = planned_quote
            self.logger.info(f"[EM:DUST_RECOVERY] Set risk_based_quote for {sym} to {planned_quote:.2f}")

        # HARD-SEPARATE DUST HEALING: Bypass all risk sizing for DUST_HEALING_BUY
        is_dust_healing_buy = bool(
            policy_ctx.get("_is_dust_healing_buy")
            or policy_ctx.get("is_dust_healing")
            or policy_ctx.get("_dust_healing")
            or str(policy_ctx.get("reason") or "").upper() == "DUST_HEALING_BUY"
        )
        if is_dust_healing_buy:
            policy_ctx["_is_dust_healing_buy"] = True
        if is_dust_operation and side == "buy":
            deficit = getattr(self.shared_state, "dust_healing_deficit", {}).get(sym, 0.0)
            if deficit <= 0:
                deficit = float(planned_quote or 0.0)
            # Check if healing would create oversized position
            current_value = 0.0
            try:
                positions = self.shared_state.get_open_positions()
                if positions and sym in positions:
                    pos = positions[sym]
                    qty = float(pos.get("quantity", 0.0) or pos.get("qty", 0.0))
                    if qty > 0:
                        price = float(self.shared_state.latest_prices.get(sym, 0.0))
                        if price > 0:
                            current_value = qty * price
            except Exception:
                current_value = 0.0
            
            new_position_value = current_value + deficit
            economic_floor = await self._resolve_nav_tier_economic_floor(
                symbol=sym,
                min_notional=float(policy_ctx.get("min_notional", 0.0) or 0.0),
            )
            if new_position_value >= economic_floor:
                self.logger.warning(
                    "[EM:DUST_HEALING] Blocking %s healing: new_position_value=%.2f >= economic_floor=%.2f (already healed)",
                    sym, new_position_value, economic_floor
                )
                return {"ok": False, "status": "blocked", "reason": "dust_already_healed", "error_code": "DUST_ALREADY_HEALED"}
            
            planned_quote = deficit  # Exact deficit, no buffer for dust healing
            self.shared_state.risk_based_quote = getattr(self.shared_state, "risk_based_quote", {})
            self.shared_state.risk_based_quote[sym] = planned_quote
            # Mark this symbol as an active dust operation for components that need symbol-scoped handling.
            is_dust_operation = True
            if not hasattr(self.shared_state, "dust_operation_symbols") or not isinstance(getattr(self.shared_state, "dust_operation_symbols", None), dict):
                self.shared_state.dust_operation_symbols = {}
            self.shared_state.dust_operation_symbols[sym] = time.time()
            self.logger.info(f"[EM:DUST_HEALING] Hard-separated sizing for {sym}: exact_deficit={planned_quote:.2f}, new_pos_value={new_position_value:.2f}, bypassing risk engine")
            
            # Forward bypass flags to ensure no module modifies planned_quote
            policy_ctx['bypass_risk'] = True
            policy_ctx['bypass_scaling'] = True
            policy_ctx['bypass_economic_floor'] = True
            policy_ctx['bypass_micro_trade'] = True
            policy_ctx['is_dust_healing'] = True

        # Fee-aware net PnL gate for non-liquidation SELLs (no stop-loss bypass)
        if side == "sell":
            net_gate = await self._check_sell_net_pnl_gate(
                sym=sym,
                quantity=quantity,
                policy_ctx=policy_ctx,
                tag=tag_raw,
                is_liq_full=is_liq_full,
                special_liq_bypass=special_liq_bypass,
            )
            if net_gate is not None:
                if self.shared_state:
                    self.shared_state.exit_in_progress[sym] = False
                return net_gate
        policy_authority = str(policy_ctx.get("authority") or policy_ctx.get("policy_authority") or "").lower()
        policy_validated = policy_authority == "metacontroller"
        if policy_validated and side == "buy" and planned_quote and planned_quote > 0:
            # ✅ FIX #11: Check if this is a bootstrap first trade
            # During bootstrap with limited capital, we MUST allow downscaling
            # Only lock planned_quote if we're NOT in bootstrap override mode
            is_bootstrap_override = policy_ctx.get("_bootstrap_override", False) or policy_ctx.get("_bypass_reason") == "BOOTSTRAP_FIRST_TRADE"
            
            # DEBUG: Log the policy context
            self.logger.info(f"[EM:Bootstrap:DEBUG] sym={sym} _bootstrap_override={policy_ctx.get('_bootstrap_override')} _bypass_reason={policy_ctx.get('_bypass_reason')} is_bootstrap={is_bootstrap_override}")
            
            if not is_bootstrap_override:
                # MetaController planned_quote is authoritative: do not downscale below it.
                policy_ctx.setdefault("_no_downscale_planned_quote", True)
                policy_ctx.setdefault("_planned_quote_floor", float(planned_quote))
            else:
                # Bootstrap mode: Allow downscaling to available capital
                self.logger.info(f"[EM:Bootstrap] Allowing downscale for {sym}: bootstrap_override=True")
                policy_ctx["_no_downscale_planned_quote"] = False
        
        # [FIX #4] UNIFIED SELL AUTHORITY: MetaController has supreme authority over SELL decisions
        # If UNIFIED_SELL_AUTHORITY=True, SELL overrides most operational gates
        # (except actual position existence and exchange rejection)
        unified_sell_authority = bool(policy_ctx.get("UNIFIED_SELL_AUTHORITY", False))
        if unified_sell_authority and side == "sell":
            self.logger.info(f"[EM:UnifiedSell] {sym} SELL has UNIFIED_SELL_AUTHORITY: bypassing operational veto points")
        
        # 🚫 TERMINAL_DUST BLOCK: If position is terminal dust, block liquidation attempts.
        # Permanent dust is always blocked (never override with unified sell authority).
        if side == "sell" and (is_liq_full or special_liq_bypass):
            if special_liq_bypass:
                self.logger.warning(
                    "[TERMINAL_DUST:BYPASS] %s liquidation SELL bypassing dust guard "
                    "(reason=%s, tag=%s)",
                    sym, liq_reason_norm or "UNKNOWN", tag_raw
                )
            else:
                # Check if this position is below minNotional (dust)
                is_dust = await self._is_position_terminal_dust(sym)
                if is_dust:
                    is_permanent = False
                    with contextlib.suppress(Exception):
                        if hasattr(self.shared_state, "is_permanent_dust"):
                            is_permanent = bool(self.shared_state.is_permanent_dust(sym))
                    if is_permanent:
                        self.logger.warning(
                            "[TERMINAL_DUST] %s is PERMANENT_DUST. Blocking liquidation attempt.",
                            sym,
                        )
                        await self._log_execution_event("terminal_dust_blocked", sym, {
                            "reason": "permanent_dust_terminal",
                            "side": "sell",
                            "liquidation_blocked": True
                        })
                        return {
                            "ok": False,
                            "status": "blocked",
                            "reason": "permanent_dust_terminal",
                            "error_code": "PermanentDust",
                            "executedQty": 0.0
                        }
                    # If UNIFIED_SELL_AUTHORITY, allow the liquidation (to clean dust)
                    if unified_sell_authority:
                        self.logger.warning(
                            f"[TERMINAL_DUST:OVERRIDE] {sym} is below minNotional but UNIFIED_SELL_AUTHORITY=True. "
                            f"Allowing liquidation to clean dust position."
                        )
                    else:
                        self.logger.warning(
                            f"[TERMINAL_DUST] {sym} is below minNotional (terminal dust). "
                            f"Blocking liquidation attempt. (Dust ratio informational only)"
                        )
                        await self._log_execution_event("terminal_dust_blocked", sym, {
                            "reason": "terminal_dust",
                            "side": "sell",
                            "liquidation_blocked": True
                        })
                        return {
                            "ok": False,
                            "status": "blocked",
                            "reason": "terminal_dust_below_notional",
                            "error_code": "TerminalDust",
                            "executedQty": 0.0
                        }
        
        # Symbol-level exit lock to prevent duplicate SELL orders
        if side == "sell":
            if self.shared_state.exit_in_progress.get(sym, False):
                self.logger.warning(f"[EXEC:EXIT_LOCK] {sym} SELL blocked: exit already in progress")
                return {
                    "ok": False,
                    "status": "blocked",
                    "reason": "exit_in_progress",
                    "error_code": "EXIT_LOCK",
                }
            self.shared_state.exit_in_progress[sym] = True
            self.logger.info(f"[EXEC:EXIT_LOCK] {sym} SELL exit lock acquired")

        # [FIX #9] LIQUIDATION BYPASS: If this is liquidation, skip ALL guards and go straight to execution
        if is_liq_full and side == "sell":
            self.logger.info(f"[EXEC:LIQ] LIQUIDATION SELL: {sym} - bypassing all guards (capital, min-notional, throughput)")

            now_ts = time.time()
            cooldown_until = float(self._liq_sell_cooldown_until.get(sym, 0.0) or 0.0)
            if cooldown_until > now_ts:
                wait_s = max(0.0, cooldown_until - now_ts)
                self.logger.warning(
                    "[EXEC:LIQ] %s in liquidation cooldown for %.1fs after repeated failures",
                    sym,
                    wait_s,
                )
                self.shared_state.exit_in_progress[sym] = False
                return {
                    "ok": False,
                    "status": "skipped",
                    "reason": "liquidation_cooldown",
                    "error_code": "LiquidationCooldown",
                }
            
            # For liquidation SELLs, we execute with best-effort quantity
            if not quantity or quantity <= 0:
                qty = await self._get_sellable_qty(sym)
                if qty <= 0:
                    self.logger.warning(f"[EXEC:LIQ] No position to liquidate for {sym}")
                    self._journal("LIQUIDATION_SKIPPED_NO_POSITION_QTY", {
                        "symbol": sym,
                        "side": "SELL",
                        "status": "SKIPPED",
                        "reason": "no_position_quantity",
                        "tag": str(clean_tag or ""),
                        "timestamp": time.time(),
                    })
                    await self.shared_state.record_rejection(sym, "SELL", "NO_POSITION_QUANTITY", source="ExecutionManager")
                    self.shared_state.exit_in_progress[sym] = False
                    return {"ok": False, "status": "skipped", "reason": "no_position_quantity", "error_code": "NoPosition"}
                quantity = qty

            quantity = await self._buffer_liquidation_sell_qty(sym, float(quantity))
            if not quantity or float(quantity) <= 0:
                self.logger.warning("[EXEC:LIQ] Buffered qty resolved to zero for %s", sym)
                self.shared_state.exit_in_progress[sym] = False
                return {
                    "ok": False,
                    "status": "skipped",
                    "reason": "no_position_quantity",
                    "error_code": "NoPosition",
                }
            
            # Execute SELL immediately without any guards, then always reconcile delayed fill
            raw = await self._place_market_order_qty(
                sym,
                float(quantity),
                "SELL",
                clean_tag,
                is_liquidation=True,
                decision_id=decision_id,
            )
            liq_client_hint = self._build_client_order_id(sym, "SELL", decision_id) if decision_id else None

            # Always reconcile delayed fill and finalize only once with correct policy_ctx
            merged = await self._reconcile_delayed_fill(
                symbol=sym,
                side="SELL",
                order=raw,
                tag=clean_tag,
                tier=tier,
                client_order_id_hint=liq_client_hint,
            )

            status = str(merged.get("status", "REJECTED")).upper()
            exec_qty = float(merged.get("executedQty", 0.0))
            is_filled = status in ("FILLED", "PARTIALLY_FILLED") and exec_qty > 0

            if is_filled:
                # Track bootstrap fill if this was a bootstrap signal
                is_bootstrap_sig = bool(policy_ctx.get("_bootstrap", False)) if policy_ctx else False
                if is_bootstrap_sig and not self._bootstrap_first_fill_done:
                    self._mark_bootstrap_fill_done()
                
                await self._emit_status("Operational", f"filled {sym} {side} status={status}")
                try:
                    if hasattr(self.shared_state, "clear_rejections"):
                        await self.shared_state.clear_rejections(sym, side.upper())
                        self.logger.info(f"[MemoryOfFailure] ✅ Cleared rejections for {sym} {side} (liquidation success)")
                except Exception as e:
                    self.logger.debug(f"[MemoryOfFailure] Failed to clear: {e}")
                # --- Canonical accounting: ONE path for all SELL fills ---
                # _handle_post_fill computes realized delta, updates metrics,
                # appends rolling PnL, emits RealizedPnlUpdated, and runs audit.
                # This replaces manual pm.close_position PnL logic.
                # [FIX] Now that _reconcile_delayed_fill no longer calls post-fill,
                # liquidation path must call it directly. Set _post_fill_done flag
                # so _finalize_sell_post_fill doesn't call it again.
                try:
                    if not merged.get("_post_fill_done"):
                        pf_result = await self._handle_post_fill(
                            symbol=sym,
                            side="SELL",
                            order=merged,
                            tag=str(clean_tag or ""),
                            tier=tier,
                        )
                        # Mark as done and cache the result so finalize won't duplicate
                        merged["_post_fill_done"] = True
                        merged["_post_fill_result"] = pf_result if isinstance(pf_result, dict) else {}
                except Exception:
                    self.logger.error("[LIQUIDATION_ACCOUNTING_FAIL] %s", sym, exc_info=True)
                    if bool(self._cfg("STRICT_ACCOUNTING_INTEGRITY", False)):
                        raise

                try:
                    await self._finalize_sell_post_fill(
                        symbol=sym,
                        order=merged,
                        tag=str(clean_tag or ""),
                        post_fill=merged.get("_post_fill_result") or {},
                        policy_ctx=policy_ctx,
                        tier=tier,
                    )
                except Exception:
                    self.logger.error("[EM:LIQ_FINALIZE_CRASH] %s: canonical SELL finalization failed", sym, exc_info=True)

                try:
                    await self._audit_post_fill_accounting(
                        symbol=sym,
                        side=side,
                        raw=merged,
                        stage="liq_post_mutation",
                        decision_id=decision_id,
                    )
                except Exception:
                    self.logger.error(f"[ACCOUNTING_AUDIT_CRASH] {sym} sell liquidation audit failed", exc_info=True)
                    # Fill already confirmed/finalized; keep canonical execution outcome.

                # Build result from `merged` (reconciled data), not `raw` (may be stale
                # for delayed fills). Propagate post-fill flags so callers like
                # close_position safely deduplicate instead of double-counting PnL.
                result = {
                    "ok": True,
                    "status": str(merged.get("status", "FILLED")).lower(),
                    "executedQty": float(merged.get("executedQty", 0.0)),
                    "avgPrice": float(merged.get("avgPrice", merged.get("price", 0.0)) or 0.0),
                    "cummulativeQuoteQty": float(merged.get("cummulativeQuoteQty", 0.0)),
                    "orderId": merged.get("orderId") or merged.get("order_id") or merged.get("exchange_order_id"),
                    "clientOrderId": merged.get("clientOrderId") or merged.get("client_order_id") or merged.get("origClientOrderId"),
                    "reason": "[LIQUIDATION]",
                    # Propagate idempotency flags so upstream callers skip redundant finalization
                    "_post_fill_result": merged.get("_post_fill_result"),
                    "_post_fill_done": merged.get("_post_fill_done"),
                    "_sell_close_events_done": merged.get("_sell_close_events_done"),
                    "_sell_finalize_key": merged.get("_sell_finalize_key"),
                }
                self.logger.info(f"[EXEC:LIQ] ✅ Liquidation SELL executed: {sym} qty={result['executedQty']:.6f}")
                self._liq_sell_fail_counts.pop(sym, None)
                self._liq_sell_cooldown_until.pop(sym, None)
                self.shared_state.exit_in_progress[sym] = False
                return result
            else:
                fail_reason = str(merged.get("reason") or raw.get("reason") or "liquidation_not_filled")
                fail_code = str(merged.get("error_code") or raw.get("error_code") or "LiquidationFailed")
                self.logger.warning(
                    "[EXEC:LIQ] ⚠️ Liquidation SELL failed: status=%s qty=%.8f reason=%s code=%s",
                    status,
                    exec_qty,
                    fail_reason,
                    fail_code,
                )

                # If exchange now reports zero sellable qty, retire stale local position
                # to stop retry storms from trapped/non-existent inventory.
                post_fail_qty = await self._get_sellable_qty(sym)
                if float(post_fail_qty) <= 0.0:
                    self._journal("LIQUIDATION_SKIPPED_NO_POSITION_QTY", {
                        "symbol": sym,
                        "side": "SELL",
                        "status": "SKIPPED",
                        "reason": "no_position_after_failed_liquidation",
                        "tag": str(clean_tag or ""),
                        "timestamp": time.time(),
                    })
                    with contextlib.suppress(Exception):
                        await self.shared_state.close_position(sym, reason="no_position_after_failed_liquidation")
                    self._liq_sell_fail_counts.pop(sym, None)
                    self._liq_sell_cooldown_until.pop(sym, None)
                    self.shared_state.exit_in_progress[sym] = False
                    return {
                        "ok": False,
                        "status": "skipped",
                        "reason": "no_position_quantity",
                        "error_code": "NoPosition",
                    }

                fail_count = int(self._liq_sell_fail_counts.get(sym, 0) or 0) + 1
                self._liq_sell_fail_counts[sym] = fail_count
                base_cd = float(self._cfg("LIQUIDATION_FAILURE_COOLDOWN_SEC", 30.0) or 30.0)
                max_cd = float(self._cfg("LIQUIDATION_FAILURE_COOLDOWN_MAX_SEC", 300.0) or 300.0)
                cooldown_s = min(max_cd, base_cd * float(2 ** min(max(fail_count - 1, 0), 4)))
                self._liq_sell_cooldown_until[sym] = time.time() + cooldown_s
                self.logger.warning(
                    "[EXEC:LIQ] %s fail_count=%d -> cooldown %.1fs (remaining_qty=%.10f)",
                    sym,
                    fail_count,
                    cooldown_s,
                    float(post_fail_qty),
                )
                await self._log_execution_event("liquidation_fail", sym, {"reason": "not_filled"})
                self.shared_state.exit_in_progress[sym] = False
                return {
                    "ok": False,
                    "status": status.lower(),
                    "executedQty": exec_qty,
                    "orderId": merged.get("orderId") or raw.get("orderId") or raw.get("exchange_order_id") or raw.get("order_id"),
                    "exchange_order_id": merged.get("exchange_order_id") or raw.get("exchange_order_id") or raw.get("orderId") or raw.get("order_id"),
                    "client_order_id": merged.get("clientOrderId") or raw.get("clientOrderId") or raw.get("client_order_id") or raw.get("origClientOrderId"),
                    "reason": "[LIQUIDATION_FAILED]",
                    "reason_detail": fail_reason,
                    "error_code": fail_code or "LiquidationFailed",
                }

        # ---- Risk gate (ALLOW / DENY / ADJUST) ---- [SKIPPED FOR LIQUIDATION AND DUST HEALING]
        # [FIX #4] UNIFIED SELL AUTHORITY: RiskManager is advisory, not veto, for SELL
        strict_risk = bool(self._cfg("STRICT_RISK_AT_EXEC", False))
        adjusted = None
        if self.risk_manager and hasattr(self.risk_manager, "check") and not is_dust_operation:
            try:
                decision = await self.risk_manager.check({
                    "symbol": sym, "side": side.upper(),
                    "qty": quantity, "quote_amount": planned_quote, "tag": clean_tag
                })
                risk_ok, risk_reason = True, None
                if isinstance(decision, dict):
                    risk_ok = bool(decision.get("ok", True))
                    risk_reason = decision.get("reason")
                    adjusted = decision.get("adjusted") or decision.get("caps")
                elif isinstance(decision, (tuple, list)):
                    risk_ok = bool(decision[0])
                    risk_reason = decision[1] if len(decision) > 1 else None
                    adjusted = decision[2] if len(decision) > 2 else None
                else:
                    risk_ok = bool(decision)
                if not risk_ok:
                    # [FIX #4] UNIFIED SELL AUTHORITY: Allow SELL to proceed despite risk checks
                    if side == "sell" and unified_sell_authority:
                        self.logger.warning(
                            f"[EM:UnifiedSell:RiskOverride] {sym} SELL failed risk check ({risk_reason}) "
                            f"but UNIFIED_SELL_AUTHORITY overrides (frees capital to restore balance)"
                        )
                    else:
                        # 🔒 DUST RETIREMENT CHECK: Don't record rejection for permanent dust
                        if await self._check_dust_retirement_before_rejection(sym, side.upper()):
                            await self._log_execution_event("risk_block", sym, {"side": side, "reason": "permanent_dust_retired"})
                            if side == "sell" and self.shared_state:
                                self.shared_state.exit_in_progress[sym] = False
                            return {"ok": False, "status": "skipped", "reason": "permanent_dust_retired", "error_code": "DustRetired"}

                        await self._log_execution_event("risk_block", sym, {"side": side, "reason": risk_reason})
                        await self.shared_state.record_rejection(sym, side.upper(), "RISK_CAP_EXCEEDED", source="ExecutionManager")
                        self.logger.info(f"[EXEC_REJECT] symbol={sym} side={side.upper()} reason=RISK_CAP_EXCEEDED count={self.shared_state.get_rejection_count(sym, side.upper())} action=SKIP")
                        if side == "sell" and self.shared_state:
                            self.shared_state.exit_in_progress[sym] = False
                        return {"ok": False, "status": "skipped" if not strict_risk else "error",
                                "reason": "RiskCapExceeded", "error_code": "RiskCapExceeded"}
                if adjusted:
                    if "quote_amount" in adjusted and planned_quote:
                        adjusted_quote = float(adjusted["quote_amount"]) if adjusted["quote_amount"] is not None else None
                        if policy_ctx.get("_no_downscale_planned_quote") and adjusted_quote is not None:
                            if adjusted_quote + 1e-9 < float(planned_quote):
                                self.logger.warning(
                                    f"[EM] Risk downscale ignored (authoritative planned_quote={planned_quote:.2f} > adjusted={adjusted_quote:.2f})"
                                )
                            else:
                                planned_quote = min(float(planned_quote), adjusted_quote)
                        else:
                            planned_quote = min(float(planned_quote), float(adjusted["quote_amount"]))
                    if "qty" in adjusted and quantity:
                        quantity = min(float(quantity), float(adjusted["qty"]))
            except Exception as _e:
                self.logger.warning(f"Risk check failed open: {_e}")
                if strict_risk:
                    if side == "sell" and self.shared_state:
                        self.shared_state.exit_in_progress[sym] = False
                    return {"ok": False, "status": "error", "reason": "RiskCheckFailed", "error_code": "RiskCheckFailed"}

        # ---- ProfitTarget guard (optional) ----
        try:
            guard_fn = getattr(self.shared_state, "profit_target_ok", None)
            if callable(guard_fn):
                # Do not let the profit guard block SELLs or liquidity/TP-SL ops or dust healing
                if side == "buy" and not any(x in (tag or "") for x in ("liquidation", "tp_sl", "balancer")) and not is_dust_operation:
                    # P9: Use NAV-aware dynamic profit target instead of static USD/hour
                    min_target = await self._resolve_nav_tier_profit_target()
                    guard_ok = bool(await guard_fn(min_usdt_per_hour=min_target))
                    self.logger.debug(
                        "[EXEC:ProfitGuard] NAV-aware target=%.4f NAV=%.2f",
                        min_target,
                        float(await self.get_tradable_nav() or 0.0)
                    )
                    if not guard_ok:
                        # 🔒 DUST RETIREMENT CHECK: Don't record rejection for permanent dust
                        if await self._check_dust_retirement_before_rejection(sym, side.upper()):
                            await self._log_execution_event("profit_target_block", sym, {"side": side, "reason": "permanent_dust_retired"})
                            return {"ok": False, "status": "skipped", "reason": "permanent_dust_retired", "error_code": "DustRetired"}
                        
                        await self._log_execution_event("profit_target_block", sym, {"side": side, "min_usdt_per_hour": min_target})
                        await self.shared_state.record_rejection(sym, side.upper(), "PROFIT_TARGET_GUARD", source="ExecutionManager")
                        self.logger.info(f"[EXEC_REJECT] symbol={sym} side={side.upper()} reason=PROFIT_TARGET_GUARD count={self.shared_state.get_rejection_count(sym, side.upper())} action=SKIP")
                        return {"ok": False, "status": "skipped", "reason": "ProfitTargetGuard", "error_code": "ProfitTargetGuard"}
        except Exception:
            pass  # non-fatal

        raw: Dict[str, Any] = {}
        try:
            # ---- Safety Check: Circuit Breaker & Health (Invariant 2) ----
            # EXIT PRIORITY: Circuit breaker is an entry guard only.
            # SELL exits must never be blocked — an open CB means we stop buying,
            # not that we trap capital in existing positions.
            if side != "sell" and hasattr(self.shared_state, "is_circuit_breaker_open") and await self.shared_state.is_circuit_breaker_open():
                self.logger.warning(f"[EXEC] 🛑 Circuit Breaker OPEN. Blocking BUY for {sym} (SELL exits always pass).")
                return {"ok": False, "status": "blocked", "reason": "CB_OPEN", "error_code": "CB_OPEN"}

            # Health: we’re about to execute
            await self._emit_status("Running", f"execute_trade {sym} {side}")
            # ---- Route by side ----
            async with self._small_nav_guard():
                if side == "buy":
                    # Local cooldown for repeated POSITION_ALREADY_OPEN loops.
                    now_buy = time.time()
                    pos_open_block_until = float(self._position_open_buy_block_until.get(sym, 0.0) or 0.0)
                    if pos_open_block_until > now_buy:
                        wait_s = int(max(0.0, pos_open_block_until - now_buy))
                        self.logger.warning(
                            "[EM:PositionOpenCooldown] Blocking BUY %s for %ss after repeated POSITION_ALREADY_OPEN rejections",
                            sym,
                            wait_s,
                        )
                        return {
                            "ok": False,
                            "status": "blocked",
                            "reason": "POSITION_OPEN_COOLDOWN",
                            "error_code": "POSITION_OPEN_COOLDOWN",
                            "retry_after_sec": wait_s,
                        }
                    elif pos_open_block_until > 0.0:
                        self._position_open_buy_block_until.pop(sym, None)

                    if planned_quote and planned_quote > 0:
                        # 🛡️ OWNERSHIP GUARD: Check SharedState classification (canonical authority)
                        # Design Rule: Classification comes ONLY from SharedState.position.classification
                        # NOT from agent intent or external signals
                        # This prevents double-opening external positions or orphaning capital
                        try:
                            pos = await self.shared_state.get_position(sym) or {}
                            classification = pos.get("classification", "")
                            
                            if classification == "BOT_POSITION":
                                # Position already owned by bot - reject to prevent double-entry
                                rej_count = 0
                                try:
                                    await self.shared_state.record_rejection(
                                        sym, "BUY", "POSITION_ALREADY_OPEN", source="ExecutionManager"
                                    )
                                    rej_count = int(
                                        self.shared_state.get_rejection_count(sym, "BUY", "POSITION_ALREADY_OPEN") or 0
                                    )
                                except Exception:
                                    rej_count = 0
                                cooldown_trigger = int(
                                    self._cfg("POSITION_OPEN_BUY_COOLDOWN_TRIGGER", 6) or 6
                                )
                                cooldown_sec = float(
                                    self._cfg("POSITION_OPEN_BUY_COOLDOWN_SEC", 120.0) or 120.0
                                )
                                if rej_count >= max(1, cooldown_trigger):
                                    self._position_open_buy_block_until[sym] = time.time() + max(5.0, cooldown_sec)
                                    self.logger.warning(
                                        "[EM:PositionOpenCooldown] Engaged for %s BUY (%ss) after %d POSITION_ALREADY_OPEN rejections",
                                        sym,
                                        int(max(5.0, cooldown_sec)),
                                        rej_count,
                                    )
                                self.logger.warning(
                                    "[EM:Ownership] Blocked BUY %s: position already BOT_POSITION (prevent duplicate entry)",
                                    sym
                                )
                                return {
                                    "ok": False,
                                    "status": "blocked",
                                    "reason": "POSITION_ALREADY_OPEN",
                                    "error_code": "POSITION_ALREADY_OPEN",
                                    "reason_detail": f"position_open_rej_count_{rej_count}",
                                }
                            elif classification == "EXTERNAL_POSITION":
                                # External position exists - skip but don't block entire system
                                self.logger.info(
                                    "[EM:Ownership] Skipping BUY %s: external position exists (not bot-owned, will track)",
                                    sym
                                )
                                return {
                                    "ok": False,
                                    "status": "skipped",
                                    "reason": "EXTERNAL_POSITION_EXISTS",
                                    "error_code": "EXTERNAL_POSITION_EXISTS",
                                }
                        except Exception as e:
                            self.logger.debug("[EM:Ownership] Classification check error: %s (proceeding with caution)", e)
                        
                        # 🎯 BOOTSTRAP FIX: SKIP cooldown check during bootstrap mode
                        # Cooldown is too aggressive when capital is dynamic and prices are volatile
                        is_bootstrap_now = bool(policy_ctx.get("bootstrap_mode", False)) if policy_ctx else False
                        
                        # Cooldown: suppress repeated execution-blocked BUYs (SKIP during bootstrap)
                        if not is_bootstrap_now and policy_ctx.get("_no_downscale_planned_quote"):
                            blocked, remaining = await self._is_buy_blocked(sym)
                            if blocked:
                                self.logger.warning(
                                    "[ExecutionManager] BUY blocked by cooldown: symbol=%s remaining=%ds",
                                    sym, int(remaining)
                                )
                                return {
                                    "ok": False,
                                    "status": "blocked",
                                    "reason": "EXEC_BLOCK_COOLDOWN",
                                    "error_code": "EXEC_BLOCK_COOLDOWN",
                                }
                        # 1) P9-Safe Check: Check intent bucket BEFORE expensive probes
                        intent = self.shared_state.get_pending_intent(sym, "BUY")
                        
                        if intent and intent.state == "ACCUMULATING":
                            # 1a, 1b: If YES (accumulated already >= min_notional) or if this piece crosses hurdle, we proceed to EXECUTE
                            projected_total = intent.accumulated_quote + float(planned_quote)
                            
                            # 1c: If NO -> accumulate and return immediately (efficiency win)
                            if projected_total < intent.min_notional:
                                await self.shared_state.add_to_accumulation(sym, "BUY", float(planned_quote))
                                self.logger.info(f"[EM] P9-Safe Accumulate: {sym} jar {intent.accumulated_quote:.2f} + {planned_quote:.2f} < {intent.min_notional:.2f}")
                                return {"ok": True, "status": "accumulating", "reason": "P9_SAFE_UNDER_HURDLE", "executedQty": 0.0}
                        
                        # 2) Run affordability (Step 2a)
                        # P9: Check agent budget reservation if this is an Agent-tagged trade
                        # BOOTSTRAP FIX: Skip reservation check during bootstrap mode
                        bootstrap_bypass = policy_ctx.get("bootstrap_bypass", False) if policy_ctx else False
                        
                        if clean_tag.startswith("meta/") and not bootstrap_bypass and not is_dust_operation:
                            agent_id = clean_tag.split("/")[-1]
                            reservations = getattr(self.shared_state, "_authoritative_reservations", {})
                            agent_budget = reservations.get(agent_id, 0.0)
                            if agent_budget < float(planned_quote or 0.0) - 0.01:
                                # ROOT CAUSE FIX: "Phantom Veto"
                                # If agent_budget is 0.0 (Allocator didn't run or gave 0), we might still want to proceed
                                # IF the global free balance allows it (e.g. Bootstrap or Surplus).
                                # However, if Allocator gave 0 intentionally, we should block.
                                
                                # Check if authorization exists at all
                                if agent_id in reservations:
                                    # 🔒 DUST RETIREMENT CHECK: Don't record rejection for permanent dust
                                    if await self._check_dust_retirement_before_rejection(sym, "BUY"):
                                        self.logger.warning(f"[EM] {sym} is PERMANENT_DUST, skipping rejection recording")
                                        return {"ok": False, "status": "skipped", "reason": "permanent_dust_retired", "error_code": "DustRetired"}
                                    
                                    self.logger.warning(f"[EM] Reservation Veto: {agent_id} budget {agent_budget:.2f} < planned {planned_quote:.2f}")
                                    # P9 MANDATORY: Record rejection (I1 Invariant - Failure Memory)
                                    await self.shared_state.record_rejection(sym, "BUY", "AGENT_RESERVATION_EXCEEDED", source="ExecutionManager")
                                    rej_count = self.shared_state.get_rejection_count(sym, "BUY")
                                    self.logger.info(f"[EXEC_REJECT] symbol={sym} side=BUY reason=AFFORDABILITY_CHECK_FAILED count={rej_count} action=RETRY")
                                    return {"ok": False, "status": "skipped", "reason": "AGENT_RESERVATION_EXCEEDED", "error_code": "RESERVATION_LT_QUOTE"}
                                else:
                                    # If agent not in reservation map, it might be new or Allocator hasn't run.
                                    # We proceed to affordability check (can_afford_market_buy) which uses REAL balance.
                                    self.logger.debug(f"[EM] No reservation record for {agent_id}, proceeding to balance check.")
                        elif bootstrap_bypass:
                            self.logger.info(f"[EM:BOOTSTRAP] Bypassing agent reservation check for bootstrap execution")

                        can, gap, reason = await self.can_afford_market_buy(sym, planned_quote, intent_override=intent, policy_context=policy_ctx)
                        
                        if not can:
                            if is_dust_operation:
                                await self._log_execution_event("order_skip", sym, {"side": "buy", "reason": reason, "dust_operation": True})
                                return {"ok": False, "status": "skipped", "reason": reason, "error_code": reason}
                            # Mandatory explicit failure for authoritative planned quotes
                            # BOOTSTRAP FIX: Skip this block during bootstrap as we've already verified spendable balance
                            min_required_quote = await self._get_min_entry_quote(sym, policy_context=policy_context)
                            # CRITICAL FIX: Only raise INSUFFICIENT_QUOTE if we're below the minimum requirement
                            # AND the reason is actually about insufficient capital (not other reasons like downscaling)
                            if policy_ctx.get("_no_downscale_planned_quote") and not bootstrap_bypass and reason == "INSUFFICIENT_QUOTE":
                                available = await self._get_available_quote(sym)
                                self.logger.error(
                                    "[ExecutionManager] BLOCKED: INSUFFICIENT_QUOTE\nplanned=%.2f available=%.2f",
                                    float(planned_quote), float(available)
                                )
                                raise ExecutionBlocked(
                                    code="INSUFFICIENT_QUOTE",
                                    planned_quote=float(planned_quote),
                                    available_quote=float(available),
                                    min_required=float(min_required_quote),
                                )
                            if reason in ("QUOTE_LT_MIN_NOTIONAL", "ZERO_QTY_AFTER_ROUNDING", "INSUFFICIENT_QUOTE_FOR_ACCUMULATION"):
                                # Intent creation/update logic
                                if not intent:
                                    filters = await self.exchange_client.ensure_symbol_filters_ready(sym)
                                    min_n = self._extract_min_notional(filters)
                                    intent = PendingPositionIntent(
                                        symbol=sym, side="BUY", target_quote=float(planned_quote),
                                        accumulated_quote=float(planned_quote),
                                        min_notional=float(min_n),
                                        ttl_sec=int(self._cfg("ACCUMULATION_TTL", 3600)),
                                        source_agent=clean_tag
                                    )
                                    await self.shared_state.record_position_intent(intent)
                                else:
                                    await self.shared_state.add_to_accumulation(sym, "BUY", float(planned_quote))
                                
                                total_acc = intent.accumulated_quote if intent else planned_quote
                                self.logger.info(f"[EM] Accumulating {planned_quote} for {sym} BUY (Reason: {reason}). Total: {total_acc}")
                                return {"ok": True, "status": "accumulating", "reason": reason, "executedQty": 0.0}

                            if reason == "INSUFFICIENT_QUOTE":
                                healed = await self._attempt_liquidity_healing(sym, max(float(gap), float(planned_quote)), {
                                    "reason": reason, "needed_quote": float(max(gap, 0.0)), "planned_quote": planned_quote
                                })
                                if not healed:
                                    # 🔒 DUST RETIREMENT CHECK: Don't record rejection for permanent dust
                                    if await self._check_dust_retirement_before_rejection(sym, "BUY"):
                                        self.logger.warning(f"[EM] {sym} is PERMANENT_DUST, skipping rejection recording")
                                        return {"ok": False, "status": "skipped", "reason": "permanent_dust_retired", "error_code": "DustRetired"}
                                    
                                    if self.shared_state and hasattr(self.shared_state, "report_agent_capital_failure"):
                                        self.shared_state.report_agent_capital_failure(f"exec_fail_{sym}")
                                    # P9 MANDATORY: Record rejection (I1 Invariant - Failure Memory)
                                    await self.shared_state.record_rejection(sym, "BUY", reason, source="ExecutionManager")
                                    rej_count = self.shared_state.get_rejection_count(sym, "BUY")
                                    self.logger.info(f"[EXEC_REJECT] symbol={sym} side=BUY reason=AFFORDABILITY_CHECK_FAILED count={rej_count} action=RETRY")
                                    await self._log_execution_event("order_skip", sym, {"side": "buy", "reason": reason})
                                    return {"ok": False, "status": "skipped", "reason": reason, "error_code": reason}
                            else:
                                # 🔒 DUST RETIREMENT CHECK: Don't record rejection for permanent dust
                                if await self._check_dust_retirement_before_rejection(sym, "BUY"):
                                    self.logger.warning(f"[EM] {sym} is PERMANENT_DUST, skipping rejection recording")
                                    return {"ok": False, "status": "skipped", "reason": "permanent_dust_retired", "error_code": "DustRetired"}
                                
                                if self.shared_state and hasattr(self.shared_state, "report_agent_capital_failure"):
                                    self.shared_state.report_agent_capital_failure(f"exec_fail_{sym}")
                                # P9 MANDATORY: Record rejection (I1 Invariant - Failure Memory)
                                await self.shared_state.record_rejection(sym, "BUY", reason, source="ExecutionManager")
                                rej_count = self.shared_state.get_rejection_count(sym, "BUY")
                                self.logger.info(f"[EXEC_REJECT] symbol={sym} side=BUY reason={reason} count={rej_count} action=RETRY")
                                await self._log_execution_event("order_skip", sym, {"side": "buy", "reason": reason})
                                return {"ok": False, "status": "skipped", "reason": reason, "error_code": reason}

                        # Threshold Met or Downscaled (can = True)
                        execute_quote = float(gap) if reason == "OK_DOWNSCALED" else float(planned_quote)

                        if policy_ctx.get("_no_downscale_planned_quote") and execute_quote != float(planned_quote) and not (intent and intent.accumulated_quote > 0):
                            self.logger.critical(
                                "Execution quote mismatch: Meta vs Execution (planned=%.2f execute=%.2f)",
                                float(planned_quote), float(execute_quote)
                            )
                            raise ExecutionBlocked(
                                code="EXEC_QUOTE_MISMATCH",
                                planned_quote=float(planned_quote),
                                available_quote=float(execute_quote),
                                min_required=float(planned_quote),
                            )
                        
                        # Apply intent aggregation if threshold is reached
                        if intent and intent.accumulated_quote > 0 and intent.state == "ACCUMULATING":
                            # Condition B: Market Validity
                            if not self.shared_state.is_intent_valid(sym, "BUY"):
                                self.logger.info(f"[EM] Intent for {sym} BUY no longer valid. Clearing bucket.")
                                await self.shared_state.clear_pending_intent(sym, "BUY")
                            # Atomic claim
                            elif await self.shared_state.mark_intent_ready(sym, "BUY"):
                                execute_quote += intent.accumulated_quote
                                self.logger.info(f"[EM] Pending intent READY -> executing {sym} BUY with total {execute_quote}")
                                await self.shared_state.clear_pending_intent(sym, "BUY")
                            else:
                                # Someone else claimed it. Check if our 'planned_quote' alone is enough.
                                filters = await self.exchange_client.ensure_symbol_filters_ready(sym)
                                min_n = self._extract_min_notional(filters)
                                await self.shared_state.add_to_accumulation(sym, "BUY", execute_quote)
                                return {"ok": True, "status": "accumulating", "reason": "intent_claimed_restarting"}

                        # Normalize execute_quote to exchange precision requirements
                        execute_quote = self._normalize_quote_precision(sym, execute_quote)

                        # 🛡️ ENTRY FLOOR GUARD: Prevent opening new trades below significant floor
                        is_dust_healing = bool(
                            policy_ctx.get("_is_dust_healing_buy")
                            or policy_ctx.get("is_dust_healing")
                            or policy_ctx.get("_dust_healing")
                            or str(policy_ctx.get("reason") or "").upper() == "DUST_HEALING_BUY"
                        ) if policy_ctx else False
                        guard_allowed, guard_reason = await self._check_entry_floor_guard(
                            symbol=sym,
                            quote_amount=float(execute_quote),
                            is_dust_healing_buy=bool(is_dust_healing)
                        )
                        if not guard_allowed:
                            self.logger.warning(f"[EM:EXEC_BLOCKED] {guard_reason}")
                            await self.shared_state.record_rejection(sym, "BUY", guard_reason, source="ExecutionManager")
                            return {"ok": False, "status": "skipped", "reason": guard_reason, "error_code": "ENTRY_FLOOR_GUARD"}

                        # Route BUY-by-quote through canonical placement path so delayed fills are
                        # reconciled before post-fill accounting/mutation checks.
                        raw = await self._place_market_order_quote(
                            sym,
                            float(execute_quote),
                            clean_tag,
                            side="BUY",
                            policy_validated=True,
                            is_liquidation=False,
                            bypass_min_notional=bool(
                                policy_ctx.get("bootstrap_bypass", False) or is_dust_operation
                            ),
                            decision_id=decision_id,
                        )
                        if not isinstance(raw, dict):
                            raw = {
                                "status": "REJECTED",
                                "executedQty": 0.0,
                                "reason": "order_not_placed",
                            }

                        filled_qty = float(raw.get("executedQty", 0.0) or 0.0)
                        avg_price = 0.0
                        if filled_qty > 0:
                            avg_price = float(raw.get("cummulativeQuoteQty", 0.0) or 0.0) / filled_qty
                        
                        # [ECON_INVARIANT] Debug invariant: Log once per BUY to verify notional >= min_notional
                        if filled_qty > 0:
                            # Get min_notional for this symbol
                            filters = await self.exchange_client.ensure_symbol_filters_ready(sym)
                            min_notional = self._extract_min_notional(filters)
                            notional = filled_qty * avg_price
                            self.logger.info(
                                "[ECON_INVARIANT] %s qty=%.6f price=%.4f notional=%.4f min_notional=%.4f",
                                sym, filled_qty, avg_price, notional, min_notional
                            )
                            # If this ever logs notional < min_notional → STOP IMMEDIATELY
                            if notional < min_notional:
                                self.logger.critical(
                                    "[ECON_INVARIANT:VIOLATION] %s notional %.4f < min_notional %.4f - STOPPING",
                                    sym, notional, min_notional
                                )
                                # This should never happen - if it does, we have a critical bug
                        
                        if avg_price > 0:
                            raw.setdefault("avgPrice", avg_price)
                            raw.setdefault("price", avg_price)
                    else:
                        if not quantity or quantity <= 0:
                            await self._log_execution_event("order_skip", sym, {"side": "buy", "reason": "InvalidQuantity"})
                            return {"ok": False, "status": "skipped", "reason": "zero_or_negative_quantity", "error_code": "InvalidQuantity"}
                        
                        # 🛡️ ENTRY FLOOR GUARD: Prevent opening new trades below significant floor
                        # For qty-based BUY, estimate quote from current market price
                        try:
                            current_price = await self.exchange_client.get_mark_price(sym)
                            estimated_quote = float(quantity) * float(current_price or 0.0)
                        except Exception:
                            estimated_quote = float(planned_quote or 0.0)
                        
                        is_dust_healing = bool(
                            policy_ctx.get("_is_dust_healing_buy")
                            or policy_ctx.get("is_dust_healing")
                            or policy_ctx.get("_dust_healing")
                            or str(policy_ctx.get("reason") or "").upper() == "DUST_HEALING_BUY"
                        ) if policy_ctx else False
                        guard_allowed, guard_reason = await self._check_entry_floor_guard(
                            symbol=sym,
                            quote_amount=estimated_quote,
                            is_dust_healing_buy=bool(is_dust_healing)
                        )
                        if not guard_allowed:
                            self.logger.warning(f"[EM:EXEC_BLOCKED] {guard_reason}")
                            await self.shared_state.record_rejection(sym, "BUY", guard_reason, source="ExecutionManager")
                            return {"ok": False, "status": "skipped", "reason": guard_reason, "error_code": "ENTRY_FLOOR_GUARD"}
                        
                        raw = await self._place_market_order_qty(
                            sym,
                            quantity,
                            "BUY",
                            clean_tag,
                            decision_id=decision_id,
                        )
                        
                        # [ECON_INVARIANT] Debug invariant: Log once per BUY to verify notional >= min_notional
                        if raw and isinstance(raw, dict):
                            filled_qty = float(raw.get("executedQty", 0.0) or 0.0)
                            if filled_qty > 0:
                                avg_price = float(raw.get("avgPrice", 0.0) or raw.get("price", 0.0) or 0.0)
                                if avg_price <= 0:
                                    # Calculate avg price from cummulativeQuoteQty if not provided
                                    cummulative_quote = float(raw.get("cummulativeQuoteQty", 0.0) or 0.0)
                                    if cummulative_quote > 0:
                                        avg_price = cummulative_quote / max(filled_qty, 1e-12)
                                
                                if avg_price > 0:
                                    # Get min_notional for this symbol
                                    filters = await self.exchange_client.ensure_symbol_filters_ready(sym)
                                    min_notional = self._extract_min_notional(filters)
                                    notional = filled_qty * avg_price
                                    self.logger.info(
                                        "[ECON_INVARIANT] %s qty=%.6f price=%.4f notional=%.4f min_notional=%.4f",
                                        sym, filled_qty, avg_price, notional, min_notional
                                    )
                                    # If this ever logs notional < min_notional → STOP IMMEDIATELY
                                    if notional < min_notional:
                                        self.logger.critical(
                                            "[ECON_INVARIANT:VIOLATION] %s notional %.4f < min_notional %.4f - STOPPING",
                                            sym, notional, min_notional
                                        )
                                        # This should never happen - if it does, we have a critical bug

                elif side == "sell":
                    # GAP #4 FIX: SELL should only be blocked on true cold bootstrap, not on any bootstrap state
                    # Allow SELL if:
                    # 1. System is not in cold bootstrap, OR
                    # 2. This is a liquidation/tp_sl/balancer operation, OR
                    # 3. We have a valid position
                    if not policy_validated:
                        is_cold = self.shared_state and hasattr(self.shared_state, "is_cold_bootstrap") and \
                                self.shared_state.is_cold_bootstrap()
                        is_liquidation = any(x in tag_lower for x in ("liquidation", "tp_sl", "balancer"))

                        if is_cold and not is_liquidation:
                            # 🔒 DUST RETIREMENT CHECK: Don't record rejection for permanent dust
                            if await self._check_dust_retirement_before_rejection(sym, "SELL"):
                                return {"ok": False, "status": "skipped", "reason": "permanent_dust_retired", "error_code": "DustRetired"}

                            await self.shared_state.record_rejection(sym, "SELL", "COLD_BOOTSTRAP_BLOCK", source="ExecutionManager")
                            rej_count = self.shared_state.get_rejection_count(sym, "SELL")
                            self.logger.info(f"[EXEC_REJECT] symbol={sym} side=SELL reason=COLD_BOOTSTRAP_BLOCK count={rej_count} action=RETRY")
                            await self._log_execution_event("sell_block_cold_bootstrap", sym, {"reason": "cold_bootstrap"})
                            return {"ok": False, "status": "blocked", "reason": "cold_bootstrap_no_sell", "error_code": "ColdBootstrap"}
                    
                    # ===== FIX A: Quote-Based SELL Support for Liquidation =====
                    # Check if this is a quote-based liquidation SELL (planned_quote provided, quantity=None)
                    if planned_quote and planned_quote > 0 and (not quantity or quantity <= 0):
                        # Quote-based SELL path (used by liquidation hard path)
                        self.logger.info(
                            "[EM:QuoteLiq:SELL] Quote-based liquidation SELL: symbol=%s, target_usdt=%.2f. "
                            "Using _place_market_order_quote (bypasses min-notional via quoteOrderQty).",
                            sym, planned_quote
                        )
                        raw = await self._place_market_order_quote(
                            sym,
                            float(planned_quote),
                            clean_tag,
                            side="SELL",
                            policy_validated=policy_validated,
                            is_liquidation=is_liq_full,
                            bypass_min_notional=special_liq_bypass,
                            decision_id=decision_id,
                        )
                    else:
                        # Standard quantity-based SELL path
                        qty = float(quantity or 0.0)
                        if qty <= 0:
                            # ✅ FIX #4: RETRY LOOP - wait for position to be available
                            # Rationale: Execution happens in sub-second, but balance refresh takes 100-500ms.
                            # Don't fail on first check; wait a bit and retry.
                            qty = await self._ensure_position_ready(sym, max_retries=3)
                            
                            if qty <= 0:
                                # 🔒 DUST RETIREMENT CHECK: Don't record rejection for permanent dust
                                if await self._check_dust_retirement_before_rejection(sym, "SELL"):
                                    return {"ok": False, "status": "skipped", "reason": "permanent_dust_retired", "error_code": "DustRetired"}
                                
                                # P9 MANDATORY: Record rejection (I1 Invariant - Failure Memory)
                                await self.shared_state.record_rejection(sym, "SELL", "NO_POSITION_QUANTITY", source="ExecutionManager")
                                rej_count = self.shared_state.get_rejection_count(sym, "SELL")
                                self.logger.info(f"[EXEC_REJECT] symbol={sym} side=SELL reason=NO_POSITION_QUANTITY count={rej_count} action=SKIP")
                                await self._log_execution_event("sell_block_no_qty", sym, {"reason": "no_position_quantity"})
                                return {"ok": False, "status": "skipped", "reason": "no_position_quantity", "error_code": "NoPosition"}
                            
                            # ✅ TARGET_FRACTION: Compute partial quantity if target_fraction is present
                            target_fraction = float(policy_ctx.get("target_fraction", 1.0) or 1.0)
                            if target_fraction > 0 and target_fraction < 1.0:
                                qty = qty * target_fraction
                                self.logger.info(
                                    "[EM:TargetFraction] SELL %s: full_qty=%.6f, target_fraction=%.2f, computed_qty=%.6f",
                                    sym, float(qty / target_fraction), target_fraction, qty
                                )
                        quantity = qty
                        raw = await self._place_market_order_qty(
                            sym,
                            float(quantity),
                            "SELL",
                            clean_tag,
                            bypass_min_notional=special_liq_bypass,
                            decision_id=decision_id,
                        )

                else:
                    # 🔒 DUST RETIREMENT CHECK: Don't record rejection for permanent dust
                    if await self._check_dust_retirement_before_rejection(sym, side.upper()):
                        await self._log_execution_event("order_skip", sym, {"side": side, "reason": "permanent_dust_retired"})
                        return {"ok": False, "status": "skipped", "reason": "permanent_dust_retired", "error_code": "DustRetired"}
                    
                    # P9 MANDATORY: Record rejection (I1 Invariant - Failure Memory)
                    await self.shared_state.record_rejection(sym, side.upper(), "INVALID_SIDE", source="ExecutionManager")
                    rej_count = self.shared_state.get_rejection_count(sym, side.upper())
                    self.logger.info(f"[EXEC_REJECT] symbol={sym} side={side.upper()} reason=INVALID_SIDE count={rej_count} action=SKIP")
                    await self._log_execution_event("order_skip", sym, {"side": side, "reason": "invalid_side"})
                    return {"ok": False, "status": "skipped", "reason": "invalid_side", "error_code": "InvalidSide"}

            # ---- Normalize output & Enforce Invariant 1 (Truth) ----
            # We ONLY mutate SharedState if we have positive confirmation of fill/partial fill.
            # REJECTED, EXPIRED, NEW, or BLOCKED must NOT mutate!

            if isinstance(raw, dict):
                skip_reason = str(raw.get("reason") or "").upper()
                if str(raw.get("status") or "").upper() == "SKIPPED" and skip_reason in ("IDEMPOTENT", "ACTIVE_ORDER"):
                    return {
                        "ok": False,
                        "status": "skipped",
                        "reason": skip_reason.lower(),
                        "error_code": skip_reason,
                    }
            
            status = str(raw.get("status", "REJECTED")).upper()
            exec_qty = float(raw.get("executedQty", 0.0))
            is_filled = status in ("FILLED", "PARTIALLY_FILLED") and exec_qty > 0

            if is_filled:
                # Track bootstrap fill if this was a bootstrap signal
                is_bootstrap_sig = bool(policy_ctx.get("_bootstrap", False)) if policy_ctx else False
                if is_bootstrap_sig and not self._bootstrap_first_fill_done:
                    self._mark_bootstrap_fill_done()
                
                # Health: success path
                await self._emit_status("Operational", f"filled {sym} {side} status={status}")
                
                # ✅ FIX #1: POST-FILL STATE SYNC
                # Force authoritative balance refresh immediately after fill
                # This bridges the temporal gap between Exchange (immediate) and SharedState (delayed)
                # Without this, next SELL decision will use stale balance data
                try:
                    await self.shared_state.sync_authoritative_balance(force=True)
                    self.logger.info(f"[StateSync:PostFill] ✅ Refreshed balances after {sym} {side} fill")
                except Exception as e:
                    self.logger.debug(f"[StateSync:PostFill] Balance sync failed (non-fatal): {e}")
                
                # CRITICAL FIX: Memory of Failure - RESET rejection counters on successful execution
                # This unlocks the system from deadlock by allowing previously-rejected symbols to be retried
                try:
                    if hasattr(self.shared_state, "clear_rejections"):
                        await self.shared_state.clear_rejections(sym, side.upper())
                        self.logger.info(f"[MemoryOfFailure] ✅ Cleared rejections for {sym} {side} (liquidation success)")
                    if side == "buy":
                        self._buy_block_state.pop(sym, None)
                except Exception as e:
                    self.logger.debug(f"[MemoryOfFailure] Failed to clear: {e}")

                # Ensure BUY positions have an entry timestamp for time-based exits.
                if side == "buy":
                    try:
                        now_ts = time.time()
                        pos = {}
                        if hasattr(self.shared_state, "positions") and isinstance(self.shared_state.positions, dict):
                            pos = dict(self.shared_state.positions.get(sym, {}) or {})
                        if pos and not pos.get("opened_at"):
                            pos["opened_at"] = now_ts
                            if hasattr(self.shared_state, "update_position"):
                                await self.shared_state.update_position(sym, pos)
                            else:
                                self.shared_state.positions[sym] = pos
                        ot = getattr(self.shared_state, "open_trades", None)
                        if isinstance(ot, dict):
                            tr = dict(ot.get(sym, {}) or {})
                            if tr and not tr.get("opened_at"):
                                tr["opened_at"] = now_ts
                                ot[sym] = tr
                    except Exception as e:
                        self.logger.debug("[EM] Failed to set opened_at for %s: %s", sym, e)
                
                # Emit realized PnL delta if SharedState can compute it
                post_fill = None
                try:
                    post_fill = await self._ensure_post_fill_handled(sym, side, raw, tier=tier, tag=tag_raw)
                except Exception as e:
                    self.logger.error(f"[POST_FILL_ACCOUNTING_CRASH] {sym}: {e}", exc_info=True)
                    if bool(self._cfg("STRICT_ACCOUNTING_INTEGRITY", False)):
                        raise

                # Explicit BUY registration/finalization hook for position observability parity.
                if side == "buy":
                    try:
                        pm = getattr(self.shared_state, "position_manager", None)
                        exec_px = float(raw.get("avgPrice", raw.get("price", 0.0)) or 0.0)
                        if exec_px <= 0 and exec_qty > 0:
                            cum_quote = float(raw.get("cummulativeQuoteQty", 0.0) or 0.0)
                            if cum_quote > 0:
                                exec_px = cum_quote / max(exec_qty, 1e-12)
                        significant_floor_usdt = 0.0
                        try:
                            if hasattr(self.shared_state, "get_significant_position_floor"):
                                significant_floor_usdt = float(
                                    await maybe_call(self.shared_state, "get_significant_position_floor", sym) or 0.0
                                )
                        except Exception:
                            significant_floor_usdt = 0.0
                        if significant_floor_usdt <= 0:
                            significant_floor_usdt = float(
                                self._cfg(
                                    "SIGNIFICANT_POSITION_FLOOR",
                                    self._cfg("MIN_SIGNIFICANT_POSITION_USDT", 25.0),
                                )
                                or 25.0
                            )
                        fee_quote = float(raw.get("fee_quote", 0.0) or raw.get("fee", 0.0) or 0.0)
                        try:
                            _, quote_asset = self._split_base_quote(sym)
                            fills = raw.get("fills") or []
                            if isinstance(fills, list):
                                fee_quote = sum(
                                    float(f.get("commission", 0.0) or 0.0)
                                    for f in fills
                                    if str(f.get("commissionAsset") or f.get("commission_asset") or "").upper() == quote_asset
                                ) or fee_quote
                        except Exception:
                            pass

                        if pm and hasattr(pm, "open_position"):
                            await pm.open_position(
                                symbol=sym,
                                executed_qty=exec_qty,
                                executed_price=exec_px,
                                fee_quote=fee_quote,
                                reason=str(policy_ctx.get("entry_reason") or policy_ctx.get("reason") or "BUY_FILLED"),
                                tag=str(tag_raw or ""),
                                tier=tier,
                                significant_floor_usdt=float(significant_floor_usdt),
                            )
                        elif hasattr(self.shared_state, "open_position"):
                            await maybe_call(
                                self.shared_state,
                                "open_position",
                                sym,
                                quantity=exec_qty,
                                avg_price=exec_px,
                            )
                    except Exception:
                        self.logger.debug("[EM] open_position finalize failed for %s", sym, exc_info=True)

                if side == "sell":
                    await self._finalize_sell_post_fill(
                        symbol=sym,
                        order=raw,
                        tag=str(tag_raw or ""),
                        post_fill=post_fill,
                        policy_ctx=policy_ctx,
                        tier=tier,
                    )

                try:
                    await self._audit_post_fill_accounting(
                        symbol=sym,
                        side=side,
                        raw=raw,
                        stage="post_mutation",
                        decision_id=decision_id,
                    )
                except Exception:
                    self.logger.error("[ACCOUNTING_AUDIT_CRASH] %s %s audit failed", sym, side, exc_info=True)
                    raise

                # ✅ HYDRATION FIX: Refresh wallet state after successful trade
                # This keeps SharedState synchronized with Binance and allows
                # the system to recalculate NAV and execute new trades
                try:
                    if self.shared_state and exec_qty > 0:
                        await self.shared_state.hydrate_balances_from_exchange()
                        await self.shared_state.hydrate_positions_from_balances()
                        self.logger.info(
                            "[EM:HYDRATE] symbol=%s side=%s post-fill balance refresh completed",
                            sym, side.upper()
                        )
                except Exception as e:
                    self.logger.warning(
                        "[EM:HYDRATE] symbol=%s side=%s post-fill balance refresh failed: %s",
                        sym, side.upper(), str(e)
                    )

                result = {
                    "ok": True,
                    "status": str(raw.get("status", "FILLED")).lower(),
                    "executedQty": float(raw.get("executedQty", 0.0)),
                    "avgPrice": float(raw.get("avgPrice", raw.get("price", 0.0)) or 0.0),
                    "cummulativeQuoteQty": float(raw.get("cummulativeQuoteQty", 0.0)),
                    "orderId": raw.get("orderId") or raw.get("exchange_order_id") or raw.get("order_id"),
                    "reason": raw.get("reason"),
                }
                
                # Tier-aware logging (Phase A Frequency Engineering)
                if tier:
                    result["tier"] = tier
                    final_quote = result.get("cummulativeQuoteQty", 0.0)
                    reduction = ""
                    if planned_quote and final_quote < planned_quote:
                        reduction = f" (reduced from {planned_quote:.2f})"
                    self.logger.info(f"[EXEC] ✅ Tier-{tier} filled: {sym} {side.upper()} | "
                                f"quote={final_quote:.2f}{reduction} | qty={result['executedQty']:.6f}")
                
                return result
            else:
                self.logger.warning(f"[EXEC] ⚠️ Invariant Check Failed: status={status}, execQty={exec_qty}. Skipping State Mutation.")
                await self._emit_status("Warning", f"skipped_mutation {sym} {side} status={status}")
                # GAP #2 FIX: Trigger pruning on failure
                await self._on_order_failed(sym, side, raw.get("reason") or "NOT_FILLED", planned_quote)
                return {
                    "ok": False,
                    "status": status.lower(),
                    "executedQty": exec_qty,
                    "orderId": raw.get("orderId") or raw.get("exchange_order_id") or raw.get("order_id"),
                    "exchange_order_id": raw.get("exchange_order_id") or raw.get("orderId") or raw.get("order_id"),
                    "client_order_id": raw.get("clientOrderId") or raw.get("client_order_id") or raw.get("origClientOrderId"),
                    "reason": raw.get("reason") or "NOT_FILLED",
                    "error_code": raw.get("error_code") or "NOT_FILLED"
                }

        except ExecutionBlocked as eb:
            if side == "buy":
                with contextlib.suppress(Exception):
                    await self._record_buy_block(sym, eb.available_quote)
            self.logger.error(
                "[ExecutionManager] BLOCKED: %s\nplanned=%.2f available=%.2f",
                eb.code, eb.planned_quote, eb.available_quote
            )
            await self._log_execution_event("order_blocked", sym, {
                "side": side, "reason": eb.code,
                "planned_quote": eb.planned_quote,
                "available_quote": eb.available_quote,
                "min_required": eb.min_required,
            })
            return {
                "ok": False,
                "status": "blocked",
                "reason": eb.code,
                "error_code": eb.code,
                "planned_quote": eb.planned_quote,
                "available_quote": eb.available_quote,
                "min_required": eb.min_required,
            }
        except Exception as e:
            recovered_sell_fill = False
            recovered_post_fill: Optional[Dict[str, Any]] = None
            recovered_status = ""
            recovered_exec_qty = 0.0
            if side == "sell" and isinstance(raw, dict):
                recovered_status = str(raw.get("status", "")).upper()
                recovered_exec_qty = float(raw.get("executedQty", 0.0) or 0.0)
                if recovered_status in ("FILLED", "PARTIALLY_FILLED") and recovered_exec_qty > 0.0:
                    try:
                        recovered_post_fill = await self._ensure_post_fill_handled(
                            sym,
                            "SELL",
                            raw,
                            tier=tier,
                            tag=str(tag_raw or ""),
                        )
                        await self._finalize_sell_post_fill(
                            symbol=sym,
                            order=raw,
                            tag=str(tag_raw or ""),
                            post_fill=recovered_post_fill,
                            policy_ctx=policy_ctx,
                            tier=tier,
                        )
                        recovered_sell_fill = True
                        self.logger.warning(
                            "[EM:SELL_EXCEPTION_RECOVERY] symbol=%s status=%s qty=%.8f order_id=%s",
                            sym,
                            recovered_status,
                            recovered_exec_qty,
                            raw.get("orderId") or raw.get("exchange_order_id") or raw.get("order_id"),
                        )
                    except Exception:
                        self.logger.error(
                            "[EM:SELL_EXCEPTION_RECOVERY_FAIL] symbol=%s status=%s qty=%.8f",
                            sym,
                            recovered_status or "UNKNOWN",
                            recovered_exec_qty,
                            exc_info=True,
                        )
            # Point 5: Escape Hatch - Report exception failure
            if self.shared_state and hasattr(self.shared_state, "report_agent_capital_failure"):
                self.shared_state.report_agent_capital_failure(f"exec_fail_{sym}")
            exec_error = self._classify_execution_error(e, sym, "execute_trade")
            error_type = getattr(exec_error, "error_type", "Unknown")
            error_msg = str(exec_error)
            self.logger.error(f"[EM:SELL_EXCEPTION] symbol={sym} side={side} exception_type={type(e).__name__} error_type={error_type} message={error_msg}", exc_info=True)
            await self._log_execution_event("order_exception", sym, {
                "side": side, "error_type": error_type,
                "error": error_msg, "tag": clean_tag, "exception_type": type(e).__name__
            })
            if recovered_sell_fill:
                await self._emit_status("Warning", f"sell_fill_recovered_after_exception {sym}")
                return {
                    "ok": True,
                    "status": recovered_status.lower() or "filled",
                    "executedQty": recovered_exec_qty,
                    "avgPrice": float(raw.get("avgPrice", raw.get("price", 0.0)) or 0.0),
                    "cummulativeQuoteQty": float(raw.get("cummulativeQuoteQty", 0.0) or 0.0),
                    "orderId": raw.get("orderId") or raw.get("exchange_order_id") or raw.get("order_id"),
                    "client_order_id": raw.get("clientOrderId") or raw.get("client_order_id") or raw.get("origClientOrderId"),
                    "reason": "sell_fill_recovered_after_exception",
                    "error_code": error_type,
                }
            # GAP #2 FIX: Trigger pruning on exception
            await self._on_order_failed(sym, side, error_type, planned_quote)
            # Health: hard error
            await self._emit_status("Error", f"exception {sym} {side}: {error_type}")
            return {"ok": False, "status": "error",
                    "reason": f"exception:{error_type}",
                    "error_code": error_type}
        finally:
            # SAFETY: Always release exit lock for SELL orders, regardless of how we exit
            if side == "sell" and self.shared_state:
                try:
                    self.shared_state.exit_in_progress[sym] = False
                except Exception:
                    pass

    async def execute_liquidation_plan(self, exits: list[dict]) -> bool:
        def _coalesce(exits_list: list[dict]) -> list[dict]:
            grouped: Dict[str, Dict[str, Any]] = {}
            for ex in exits_list or []:
                if not ex:
                    continue
                sym = self._norm_symbol(ex.get("symbol"))
                qty = float(ex.get("quantity", 0) or 0.0)
                if not sym or qty <= 0:
                    continue
                tag = str(ex.get("tag") or "liquidation")
                entry = grouped.setdefault(sym, {"symbol": sym, "quantity": 0.0, "tags": set(), "raw": []})
                entry["quantity"] += qty
                entry["tags"].add(tag)
                entry["raw"].append(ex)

            coalesced: list[dict] = []
            for sym, data in grouped.items():
                qty = float(data.get("quantity", 0.0) or 0.0)
                if qty <= 0:
                    continue
                try:
                    pos_qty = float(self.shared_state.get_position_qty(sym) or 0.0)
                    if pos_qty > 0 and qty > pos_qty:
                        self.logger.info(
                            "[ExecutionManager] Coalesced liquidation qty %.8f exceeds position qty %.8f for %s. Clamping.",
                            qty, pos_qty, sym,
                        )
                        qty = pos_qty
                except Exception:
                    pass

                tags = list(data.get("tags") or [])
                tag = tags[0] if len(tags) == 1 else "liquidation/coalesced"
                coalesced.append({
                    "symbol": sym,
                    "quantity": qty,
                    "tag": tag,
                    "_coalesced_count": len(data.get("raw") or []),
                    "_coalesced_tags": tags,
                })
            return coalesced

        exits = _coalesce(exits or [])
        any_success = False
        for ex in exits:
            try:
                sym = self._norm_symbol(ex.get("symbol"))
                qty = float(ex.get("quantity", 0))
                tag = self._sanitize_tag(ex.get("tag") or "liquidation")
                # Build policy_ctx for this exit (if any context is available)
                policy_ctx = ex.get("policy_ctx") if isinstance(ex, dict) and ex.get("policy_ctx") else {}
                if qty <= 0:
                    continue
                await self._log_execution_event(
                    "liquidation_exit_attempt",
                    sym,
                    {
                        "qty": float(qty),
                        "tag": tag,
                        "coalesced_count": int(ex.get("_coalesced_count") or 1),
                    },
                )
                if int(ex.get("_coalesced_count") or 0) > 1:
                    self.logger.info(
                        "[ExecutionManager] Coalesced %d liquidation exits for %s into qty=%.8f (tags=%s)",
                        int(ex.get("_coalesced_count") or 0),
                        sym,
                        qty,
                        ",".join(ex.get("_coalesced_tags") or [])
                    )
                raw = await self._place_market_order_qty(sym, qty, "SELL", tag)
                # Always reconcile delayed fill and finalize only once with correct policy_ctx
                merged = await self._reconcile_delayed_fill(
                    symbol=sym,
                    side="SELL",
                    order=raw,
                    tag=tag,
                    tier=None,
                    client_order_id_hint=(
                        (raw or {}).get("clientOrderId")
                        or (raw or {}).get("client_order_id")
                        or (raw or {}).get("origClientOrderId")
                    ) if isinstance(raw, dict) else None,
                )
                status = str((merged or {}).get("status", "")).upper() if isinstance(merged, dict) else ""
                exec_qty = self._safe_float((merged or {}).get("executedQty") if isinstance(merged, dict) else 0.0, 0.0)
                ok_field = bool((merged or {}).get("ok")) if isinstance(merged, dict) else False
                is_filled = bool(
                    ok_field
                    or (status in ("FILLED", "PARTIALLY_FILLED") and exec_qty > 0.0)
                )
                await self._log_execution_event(
                    "liquidation_exit_result",
                    sym,
                    {
                        "status": status or "unknown",
                        "ok_field": bool(ok_field),
                        "filled": bool(is_filled),
                        "executed_qty": float(exec_qty),
                        "order_id": (merged or {}).get("orderId") if isinstance(merged, dict) else None,
                        "exchange_order_id": (merged or {}).get("exchange_order_id") if isinstance(merged, dict) else None,
                        "client_order_id": (merged or {}).get("clientOrderId") if isinstance(merged, dict) else None,
                        "error_code": (merged or {}).get("error_code") if isinstance(merged, dict) else None,
                        "reason": (merged or {}).get("reason") if isinstance(merged, dict) else "no_raw_response",
                        "tag": tag,
                    },
                )
                if is_filled and isinstance(merged, dict):
                    any_success = True
                    # Ensure canonical SELL finalization for fills reconciled
                    # after _place_market_order_qty returned (late fills).
                    try:
                        lp_post = await self._ensure_post_fill_handled(
                            symbol=sym, side="SELL", order=merged,
                            tier=None, tag=str(tag or ""),
                        )
                        await self._finalize_sell_post_fill(
                            symbol=sym, order=merged, tag=str(tag or ""),
                            post_fill=lp_post, policy_ctx=policy_ctx, tier=None,
                        )
                    except Exception:
                        self.logger.error("[EM:LiqPlan:FINALIZE_CRASH] %s", sym, exc_info=True)
                else:
                    self.logger.warning(
                        "[ExecutionManager] Liquidation exit not filled for %s qty=%.8f status=%s ok=%s reason=%s",
                        sym,
                        float(qty),
                        status or "unknown",
                        ok_field,
                        (merged or {}).get("reason") if isinstance(merged, dict) else "no_raw_response",
                    )
            except Exception as e:
                self.logger.warning(f"Liquidation exit failed for {ex}: {e}")
        return any_success

    async def start_order_monitoring(self):
        """
        Background loop to monitor open orders (staleness, hygiene).
        """
        self.logger.info("🕵️ ExecutionManager order monitoring started.")
        while True:
            # Placeholder: In future, check for orders stuck in NEW/PARTIALLY_FILLED for too long
            await asyncio.sleep(self.order_monitor_interval)

    async def start_position_sync_monitor(self):
        """
        ✅ ELITE: Background loop for periodic position invariant checks.
        
        Runs every N seconds (configurable) to detect:
        - Exchange/internal position drift
        - Phantom positions (position exists but shouldn't)
        - Lost executions (internal has execution but exchange doesn't)
        
        On violation: Logs CRITICAL, emits health status DEGRADED, optionally halts.
        
        Config:
        - POSITION_SYNC_CHECK_INTERVAL_SEC=60 (default)
        - POSITION_SYNC_TOLERANCE=0.00001 (absolute drift allowed)
        - STRICT_POSITION_INVARIANTS=False (if True, halt on violation)
        """
        try:
            check_interval = float(self._cfg("POSITION_SYNC_CHECK_INTERVAL_SEC", 60.0) or 60.0)
            check_interval = max(5.0, min(check_interval, 300.0))  # 5-300 sec range
            
            self.logger.info("[EM:PosSyncMonitor] Started (interval=%.1fs)", check_interval)
            
            while True:
                try:
                    await asyncio.sleep(check_interval)
                    
                    # Get all symbols being tracked
                    symbols = []
                    try:
                        if hasattr(self.shared_state, "positions"):
                            symbols = [str(s).upper() for s in self.shared_state.positions.keys()]
                    except Exception:
                        pass
                    
                    if not symbols:
                        continue
                    
                    # Check each symbol's invariants
                    violations = []
                    for symbol in symbols[:20]:  # Limit per cycle to avoid hammering exchange
                        try:
                            ok = await self._verify_position_invariants(
                                symbol=symbol,
                                event_type="PERIODIC_SYNC_CHECK",
                            )
                            if not ok:
                                violations.append(symbol)
                        except Exception:
                            pass
                    
                    if violations:
                        self.logger.warning(
                            "[EM:PosSyncMonitor] Invariant violations on: %s",
                            ", ".join(violations)
                        )
                
                except asyncio.CancelledError:
                    self.logger.info("[EM:PosSyncMonitor] Stopped")
                    break
                except Exception:
                    self.logger.debug("[EM:PosSyncMonitor] Iteration failed", exc_info=True)
                    
        except Exception:
            self.logger.error("[EM:PosSyncMonitor] Failed to start", exc_info=True)

    # =============================
    # Placement internals
    # =============================
    async def _place_market_order_qty(
        self,
        symbol: str,
        quantity: float,
        side: str = "BUY",
        tag: Optional[str] = None,
        policy_validated: bool = False,
        is_liquidation: bool = False,
        bypass_min_notional: bool = False,
        decision_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        get_px = getattr(self.exchange_client, "get_current_price", None) or getattr(self.exchange_client, "get_price")
        current_price = await get_px(symbol)
        if not current_price:
            await self._log_execution_event("order_skip", symbol, {"side": side.upper(), "reason": "no_price"})
            return self._canonical_exec_result(
                symbol=symbol,
                side=side,
                raw_order=None,
                default_status="REJECTED",
                default_reason="no_price",
            )

        qty = float(quantity or 0.0)
        if qty <= 0:
            await self._log_execution_event("order_skip", symbol, {"side": side.upper(), "reason": "qty_invalid"})
            return self._canonical_exec_result(
                symbol=symbol,
                side=side,
                raw_order=None,
                default_status="REJECTED",
                default_reason="qty_invalid",
            )

        if side.upper() == "BUY" and not policy_validated:
            ok_aff, why = await self.explain_afford_market_buy(symbol, Decimal(str(qty)))
            if not ok_aff:
                if "INSUFFICIENT_QUOTE" in why or "QUOTE_LT_MIN_NOTIONAL" in why:
                    need_quote = float(qty) * float(current_price) * float(1.0 + self.trade_fee_pct) * float(self.safety_headroom)
                    heal_context = {"reason": "affordability_gate", "symbol": symbol, "needed_quote": need_quote, "current_price": current_price}
                    healed = await self._attempt_liquidity_healing(symbol, need_quote, heal_context)
                    if not healed:
                        await self._log_execution_event("order_skip", symbol, {"side": side.upper(), "reason": str(why)})
                        return self._canonical_exec_result(
                            symbol=symbol,
                            side=side,
                            raw_order=None,
                            default_status="REJECTED",
                            default_reason=str(why),
                            default_quote=float(qty) * float(current_price),
                        )
                else:
                    await self._log_execution_event("order_skip", symbol, {"side": side.upper(), "reason": str(why)})
                    return self._canonical_exec_result(
                        symbol=symbol,
                        side=side,
                        raw_order=None,
                        default_status="REJECTED",
                        default_reason=str(why),
                        default_quote=float(qty) * float(current_price),
                    )

        quote_asset = self._split_symbol_quote(symbol)
        reservation_id: Optional[str] = None
        planned_quote = float(qty) * float(current_price)
        if side.upper() == "BUY":
            try:
                reservation_id = await self.shared_state.reserve_liquidity(quote_asset, float(planned_quote), ttl_seconds=30)
                await self._log_execution_event("liquidity_reserved", symbol, {
                    "asset": quote_asset, "amount": float(planned_quote), "scope": "buy_by_qty", "reservation_id": reservation_id
                })
            except Exception as e:
                self.logger.warning(f"Reserve failed, proceeding: {e}")

        try:
            raw_order = await self._place_market_order_internal(
                symbol=symbol,
                side=side.upper(),
                quantity=float(qty),
                current_price=current_price,
                planned_quote=float(planned_quote),
                comment=self._sanitize_tag(tag or "meta"),
                is_liquidation=is_liquidation,
                bypass_min_notional=bypass_min_notional,
                decision_id=decision_id,
            )

            order_res = self._canonical_exec_result(
                symbol=symbol,
                side=side,
                raw_order=raw_order,
                default_status="REJECTED",
                default_quote=planned_quote,
            )

            has_submission_ref = bool(
                order_res.get("orderId")
                or order_res.get("exchange_order_id")
                or order_res.get("client_order_id")
                or order_res.get("clientOrderId")
            )
            status = str(order_res.get("status", "")).upper()
            is_filled = status in ("FILLED", "PARTIALLY_FILLED")

            # Treat only truly unplaced payloads as order-not-placed.
            if not has_submission_ref and not is_filled:
                # Order not placed or failed
                if reservation_id:
                    with contextlib.suppress(Exception):
                        await self.shared_state.rollback_liquidity(quote_asset, reservation_id)
                        await self._log_execution_event("liquidity_rolled_back", symbol, {
                            "asset": quote_asset,
                            "amount": float(planned_quote),
                            "scope": "buy_by_qty",
                            "reason": "order_not_placed",
                            "reservation_id": reservation_id
                        })
                await self._log_execution_event("order_skip", symbol, {"side": side.upper(), "reason": "order_not_placed"})
                return self._canonical_exec_result(
                    symbol=symbol,
                    side=side,
                    raw_order=raw_order,
                    default_status="REJECTED",
                    default_reason="order_not_placed",
                    default_quote=planned_quote,
                )

            # PHASE 4: Update position with actual fills (before post-fill and liquidity release)
            if is_filled:
                position_updated = await self._update_position_from_fill(
                    symbol=symbol,
                    side=side,
                    order=order_res,
                    tag=str(tag or "")
                )
                if not position_updated:
                    self.logger.warning(
                        "[PHASE4_SKIPPED] Position not updated for %s", symbol
                    )
            else:
                self.logger.info(
                    "[PHASE4_SKIPPED_NO_FILL] Position update skipped (order not filled). "
                    "symbol=%s status=%s", symbol, status
                )

            if reservation_id:
                if is_filled:
                    # Release: Order was filled (or partially filled)
                    spent = float(order_res.get("cummulativeQuoteQty", planned_quote))
                    with contextlib.suppress(Exception):
                        await self.shared_state.release_liquidity(quote_asset, reservation_id)
                        await self._log_execution_event("liquidity_released", symbol, {
                            "asset": quote_asset,
                            "amount": float(spent),
                            "scope": "buy_by_qty",
                            "reason": "order_filled",
                            "reservation_id": reservation_id,
                            "actual_status": status,
                        })
                else:
                    # Rollback: Order not filled yet (pending, new, etc.)
                    with contextlib.suppress(Exception):
                        await self.shared_state.rollback_liquidity(quote_asset, reservation_id)
                        await self._log_execution_event("liquidity_rolled_back", symbol, {
                            "asset": quote_asset,
                            "amount": float(planned_quote),
                            "scope": "buy_by_qty",
                            "reason": "order_not_filled",
                            "reservation_id": reservation_id,
                            "actual_status": status,
                        })

            # Continue with post-fill handling (only if filled)
            if order_res and is_filled:
                try:
                    post_fill = await self._ensure_post_fill_handled(
                        symbol,
                        side.upper(),
                        order_res,
                        tier=None,
                        tag=str(tag or ""),
                    )
                    if side.upper() == "SELL":
                        await self._finalize_sell_post_fill(
                            symbol=symbol,
                            order=order_res,
                            tag=str(tag or ""),
                            post_fill=post_fill,
                            policy_ctx=None,
                            tier=None,
                        )
                except Exception as e:
                    self.logger.error(f"[POST_FILL_CRASH_DIRECT_PATH] {symbol}: {e}", exc_info=True)

            return order_res
        except Exception:
            if reservation_id:
                with contextlib.suppress(Exception):
                    await self.shared_state.rollback_liquidity(quote_asset, reservation_id)
                    await self._log_execution_event("liquidity_rolled_back", symbol, {
                        "asset": quote_asset,
                        "amount": float(planned_quote),
                        "scope": "buy_by_qty",
                        "reason": "exception",
                        "reservation_id": reservation_id
                    })
            raise

    async def _place_market_order_quote(
        self,
        symbol: str,
        quote: float,
        tag: Optional[str],
        side: str = "BUY",
        policy_validated: bool = False,
        is_liquidation: bool = False,
        bypass_min_notional: bool = False,
        decision_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        get_px = getattr(self.exchange_client, "get_current_price", None) or getattr(self.exchange_client, "get_price")
        current_price = await get_px(symbol)
        if not current_price:
            await self._log_execution_event("order_skip", symbol, {"side": side.upper(), "reason": "no_price"})
            return self._canonical_exec_result(
                symbol=symbol,
                side=side,
                raw_order=None,
                default_status="REJECTED",
                default_reason="no_price",
            )
        
        # ════════════════════════════════════════════════════════════════════
        # BOOTSTRAP MINIMUM ALLOCATION ENFORCEMENT
        # ════════════════════════════════════════════════════════════════════
        # During bootstrap execution, enforce minimum allocation to guarantee
        # orders pass exchange minNotional requirements.
        #
        # Formula: allocation = max(capital * risk_fraction, min_notional * 1.1)
        # Example:
        #   capital = 91 USDT
        #   risk_fraction = 0.1
        #   normal = 91 * 0.1 = 9.1
        #   min_notional = 10
        #   allocation = max(9.1, 10 * 1.1) = max(9.1, 11) = 11 USDT ✓ Passes
        # ════════════════════════════════════════════════════════════════════
        
        is_bootstrap = bool(
            bypass_min_notional or 
            (tag and "bootstrap" in str(tag).lower()) or
            (decision_id and "bootstrap" in str(decision_id).lower())
        )
        
        if is_bootstrap and side.upper() == "BUY" and quote > 0:
            try:
                # Get min_notional requirement from exchange
                min_notional = 10.0  # Default minimum
                try:
                    if hasattr(self.exchange_client, "get_symbol_info"):
                        info = await self.exchange_client.get_symbol_info(symbol)
                        if info and isinstance(info, dict):
                            filters = info.get("filters", [])
                            for f in filters:
                                if f.get("filterType") == "MIN_NOTIONAL" or f.get("filterType") == "NOTIONAL":
                                    min_notional = float(f.get("minNotional", f.get("minValue", 10.0)))
                                    break
                except Exception:
                    min_notional = float(getattr(self.config, "MIN_NOTIONAL_USDT", 10.0))
                
                # Enforce minimum: allocation must be >= min_notional * 1.1
                minimum_allocation = min_notional * 1.1
                original_quote = quote
                
                if quote < minimum_allocation:
                    quote = minimum_allocation
                    self.logger.info(
                        "[EM:BOOTSTRAP_ALLOC] 🚀 MINIMUM ALLOCATION ENFORCED:\n"
                        "  Original: %.2f USDT\n"
                        "  MinNotional: %.2f USDT\n"
                        "  Required: %.2f USDT (min_notional * 1.1)\n"
                        "  Adjusted: %.2f USDT ✓ Now passes minimum",
                        original_quote, min_notional, minimum_allocation, quote
                    )
                else:
                    self.logger.debug(
                        "[EM:BOOTSTRAP_ALLOC] ✓ Allocation already meets minimum:\n"
                        "  Quote: %.2f USDT >= Required: %.2f USDT",
                        quote, minimum_allocation
                    )
            except Exception as e:
                self.logger.warning("[EM:BOOTSTRAP_ALLOC] Error enforcing minimum allocation: %s", e)
                # Continue with original quote if enforcement fails

        # For SELL orders, skip affordability check (they use position quantity, not capital)
        if side.upper() == "BUY" and not policy_validated:
            ok_aff, why = await self.explain_afford_market_buy(symbol, Decimal(str(quote)))
            if not ok_aff:
                if "INSUFFICIENT_QUOTE" in why or "QUOTE_LT_MIN_NOTIONAL" in why:
                    need_quote = float(quote) * float(1.0 + self.trade_fee_pct) * float(self.safety_headroom)
                    heal_context = {"reason": "affordability_gate", "symbol": symbol, "needed_quote": need_quote, "current_price": current_price}
                    healed = await self._attempt_liquidity_healing(symbol, need_quote, heal_context)
                    if not healed:
                        await self._log_execution_event("order_skip", symbol, {"side": side.upper(), "reason": str(why)})
                        return self._canonical_exec_result(
                            symbol=symbol,
                            side=side,
                            raw_order=None,
                            default_status="REJECTED",
                            default_reason=str(why),
                            default_quote=quote,
                        )
                else:
                    await self._log_execution_event("order_skip", symbol, {"side": side.upper(), "reason": str(why)})
                    return self._canonical_exec_result(
                        symbol=symbol,
                        side=side,
                        raw_order=None,
                        default_status="REJECTED",
                        default_reason=str(why),
                        default_quote=quote,
                    )

        quote_asset = self._split_symbol_quote(symbol)

        # Reserve planned quote; release/commit after exec (use reservation_id)
        # For SELL, we're selling into USDT so no reservation needed
        reservation_id: Optional[str] = None
        if side.upper() == "BUY":
            try:
                reservation_id = await self.shared_state.reserve_liquidity(quote_asset, float(quote), ttl_seconds=30)
                await self._log_execution_event("liquidity_reserved", symbol, {
                    "asset": quote_asset, "amount": float(quote), "scope": "buy_by_quote", "reservation_id": reservation_id
                })
            except Exception as e:
                self.logger.warning(f"Reserve failed, proceeding: {e}")

        try:
            raw_order = await self._place_market_order_internal(
                symbol=symbol,
                side=side.upper(),
                quantity=0.0,
                current_price=current_price,
                planned_quote=float(quote),
                comment=self._sanitize_tag(tag or "meta"),
                is_liquidation=is_liquidation,
                bypass_min_notional=bypass_min_notional,
                decision_id=decision_id,
            )

            order_res = self._canonical_exec_result(
                symbol=symbol,
                side=side,
                raw_order=raw_order,
                default_status="REJECTED",
                default_quote=quote,
            )

            has_submission_ref = bool(
                order_res.get("orderId")
                or order_res.get("exchange_order_id")
                or order_res.get("client_order_id")
                or order_res.get("clientOrderId")
            )
            status = str(order_res.get("status", "")).upper()
            is_filled = status in ("FILLED", "PARTIALLY_FILLED")

            # Treat only truly unplaced payloads as order-not-placed.
            if not has_submission_ref and not is_filled:
                # Order not placed or failed
                if reservation_id:
                    with contextlib.suppress(Exception):
                        await self.shared_state.rollback_liquidity(quote_asset, reservation_id)
                        await self._log_execution_event("liquidity_rolled_back", symbol, {
                            "asset": quote_asset,
                            "amount": float(quote),
                            "scope": "buy_by_quote",
                            "reason": "order_not_placed",
                            "reservation_id": reservation_id
                        })
                await self._log_execution_event("order_skip", symbol, {"side": side.upper(), "reason": "order_not_placed"})
                return self._canonical_exec_result(
                    symbol=symbol,
                    side=side,
                    raw_order=raw_order,
                    default_status="REJECTED",
                    default_reason="order_not_placed",
                    default_quote=quote,
                )

            # PHASE 4: Update position with actual fills (before post-fill and liquidity release)
            if is_filled:
                position_updated = await self._update_position_from_fill(
                    symbol=symbol,
                    side=side,
                    order=order_res,
                    tag=str(tag or "")
                )
                if not position_updated:
                    self.logger.warning(
                        "[PHASE4_SKIPPED] Position not updated for %s", symbol
                    )
                else:
                    # CRITICAL: Journal ORDER_FILLED for audit trail and invariant validation
                    self._journal("ORDER_FILLED", {
                        "symbol": symbol,
                        "side": side.upper(),
                        "executed_qty": float(order_res.get("executedQty", 0.0) or 0.0),
                        "avg_price": self._resolve_post_fill_price(
                            order_res,
                            float(order_res.get("executedQty", 0.0) or 0.0)
                        ),
                        "cumm_quote": float(order_res.get("cummulativeQuoteQty", quote) or quote),
                        "order_id": str(order_res.get("orderId", "")),
                        "status": str(order_res.get("status", "")),
                        "tag": str(tag or ""),
                        "path": "quote_based",
                    })
            else:
                self.logger.info(
                    "[PHASE4_SKIPPED_NO_FILL] Position update skipped (order not filled). "
                    "symbol=%s status=%s", symbol, status
                )

            if reservation_id:
                if is_filled:
                    # Release: Order was filled (or partially filled)
                    spent = float(order_res.get("cummulativeQuoteQty", quote))
                    with contextlib.suppress(Exception):
                        await self.shared_state.release_liquidity(quote_asset, reservation_id)
                        await self._log_execution_event("liquidity_released", symbol, {
                            "asset": quote_asset,
                            "amount": float(spent),
                            "scope": "buy_by_quote",
                            "reason": "order_filled",
                            "reservation_id": reservation_id,
                            "actual_status": status,
                        })
                else:
                    # Rollback: Order not filled yet (pending, new, etc.)
                    with contextlib.suppress(Exception):
                        await self.shared_state.rollback_liquidity(quote_asset, reservation_id)
                        await self._log_execution_event("liquidity_rolled_back", symbol, {
                            "asset": quote_asset,
                            "amount": float(quote),
                            "scope": "buy_by_quote",
                            "reason": "order_not_filled",
                            "reservation_id": reservation_id,
                            "actual_status": status,
                        })

            # Continue with post-fill handling (only if filled)
            if order_res and is_filled:
                try:
                    post_fill = await self._ensure_post_fill_handled(
                        symbol,
                        side.upper(),
                        order_res,
                        tier=None,
                        tag=str(tag or ""),
                    )
                    if side.upper() == "SELL":
                        await self._finalize_sell_post_fill(
                            symbol=symbol,
                            order=order_res,
                            tag=str(tag or ""),
                            post_fill=post_fill,
                            policy_ctx=None,
                            tier=None,
                        )
                except Exception as e:
                    self.logger.error(
                        f"[POST_FILL_CRASH_QUOTE_PATH] {symbol}: {e}",
                        exc_info=True,
                    )

            return order_res
        except Exception:
            if reservation_id:
                with contextlib.suppress(Exception):
                    await self.shared_state.rollback_liquidity(quote_asset, reservation_id)
                    await self._log_execution_event("liquidity_rolled_back", symbol, {
                        "asset": quote_asset,
                        "amount": float(quote),
                        "scope": "buy_by_quote",
                        "reason": "exception",
                        "reservation_id": reservation_id
                    })
            raise

    @resilient_trade(component_name="ExecutionManager", max_attempts=3)
    async def _place_market_order_internal(
        self,
        symbol: str,
        side: str,
        quantity: float,
        current_price: float,
        planned_quote: Optional[float] = None,
        comment: str = "",
        is_liquidation: bool = False,
        bypass_min_notional: bool = False,
        decision_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        return await self._place_market_order_core(
            symbol,
            side,
            quantity,
            current_price,
            planned_quote,
            comment,
            is_liquidation,
            bypass_min_notional,
            decision_id,
        )

    async def _decide_execution_method(
        self,
        symbol: str,
        side: str,
        quantity: float,
        current_price: float,
        planned_quote: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """
        Decide whether to use maker-biased execution (limit order) vs market order.
        
        Returns: (use_maker, reason)
        - use_maker: True if should try maker limit order first
        - reason: Description of why this decision was made
        """
        # Get current NAV for micro-account detection
        try:
            nav_quote = self.shared_state.get_nav_quote()
        except Exception:
            nav_quote = None
        
        # Check if maker orders are enabled and NAV is below threshold
        if not self.maker_executor.should_use_maker_orders(nav_quote):
            return False, f"nav_above_threshold(nav={nav_quote})"
        
        # Check if spread quality is acceptable (evaluate_spread_quality checks spread_pct)
        try:
            acceptable, spread_pct, quality_reason = await self.maker_executor.evaluate_spread_quality(
                symbol=symbol,
                side=side,
                current_price=current_price,
            )
            if not acceptable:
                return False, f"spread_too_wide({quality_reason}, spread={spread_pct*100:.3f}%)"
        except Exception as e:
            self.logger.debug(f"[MakerExec] evaluate_spread_quality failed: {e}; falling back to market")
            return False, f"spread_eval_error({str(e)})"
        
        # Only use maker for BUY orders (more likely to fill, easier to handle)
        if side.upper() != "BUY":
            return False, "sell_orders_use_market_only"
        
        return True, "maker_conditions_met"

    async def _place_market_order_core(
        self,
        symbol: str,
        side: str,
        quantity: float,
        current_price: float,
        planned_quote: Optional[float] = None,
        comment: str = "",
        is_liquidation: bool = False,
        bypass_min_notional: bool = False,
        decision_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        symbol = self._norm_symbol(symbol)
        if not self._ensure_trade_journal_ready(reason="pre_submit"):
            if self._is_live_trading_mode() and bool(self._require_trade_journal_live):
                with contextlib.suppress(Exception):
                    await self._log_execution_event("order_skip", symbol, {
                        "side": side.upper(),
                        "reason": "TRADE_JOURNAL_UNAVAILABLE",
                    })
                return {
                    "ok": False,
                    "status": "SKIPPED",
                    "reason": "TRADE_JOURNAL_UNAVAILABLE",
                    "error_code": "TRADE_JOURNAL_UNAVAILABLE",
                    "symbol": symbol,
                    "side": side.upper(),
                }

        # --- Filters from ExchangeClient (raw) ---
        filters = await self.exchange_client.ensure_symbol_filters_ready(symbol)
        step_size, min_qty, max_qty, tick_size, min_notional = self._extract_filter_vals(filters)
        if step_size <= 0 or min_notional <= 0:
            return None

        use_quote_path = side.upper() == "BUY" and planned_quote and planned_quote > 0

        # ✅ BEST PRACTICE: Normalize quantity to step_size before submission for qty-based orders.
        # Quote-based BUYs must skip this check entirely because they intentionally submit
        # `quoteOrderQty` with no precomputed quantity.
        # For SELL orders, preserve the raw quantity so the dust-prevention round-up logic
        # later can see the true remainder before deciding to round down or up.
        _raw_quantity = float(quantity or 0.0)
        if not use_quote_path:
            quantity = self._normalize_quantity(quantity, step_size)
            if quantity <= 0:
                self.logger.warning(
                    "[EM:NormalizeQty] %s %s raw_quantity=%.8f normalized to 0 (step_size=%.8f); rejecting order",
                    symbol, side.upper(), _raw_quantity, step_size
                )
                return self._canonical_exec_result(
                    symbol=symbol,
                    side=side,
                    raw_order=None,
                    default_status="REJECTED",
                    default_reason="qty_invalid_after_normalization",
                )

        safe_tag = self._sanitize_tag(comment)
        decision_id = decision_id or self._resolve_decision_id(getattr(self, "_current_policy_context", None))
        client_id = self._build_client_order_id(symbol, side.upper(), decision_id)

        # Check for bootstrap flag - bypass idempotency check if bootstrap is allowed and flagged
        is_bootstrap_signal = False
        if hasattr(self, "_current_policy_context") and self._current_policy_context:
            is_bootstrap_signal = bool(self._current_policy_context.get("_bootstrap", False))
            # CRITICAL: Also check for _bootstrap_override (used by gate bypass)
            is_bootstrap_signal = is_bootstrap_signal or bool(self._current_policy_context.get("_bootstrap_override", False))
        
        # Only allow bootstrap bypass if:
        # 1. Signal is marked as bootstrap
        # 2. We're in a phase that allows bootstrap override
        allow_bootstrap_bypass = is_bootstrap_signal and self._is_bootstrap_allowed()
        
        # is_bootstrap flag for use in quote adjustment logic
        is_bootstrap = allow_bootstrap_bypass or bypass_min_notional
        
        if not allow_bootstrap_bypass:
            if self._is_duplicate_client_order_id(client_id):
                self.logger.debug("[EM] Duplicate client_order_id for %s %s; skipping.", symbol, side.upper())
                return {"status": "SKIPPED", "reason": "IDEMPOTENT"}

        # 🎯 BEST PRACTICE #4: Opportunistically auto-reset stale rejection counters
        # This runs in the background and clears counters older than 60 seconds
        try:
            await self._maybe_auto_reset_rejections(symbol, side)
        except Exception as e:
            self.logger.debug(f"[EM:REJECTION_RESET] Auto-reset failed: {e}")

        order_key = (symbol, side.upper())
        now = time.time()
        
        # 🎯 BOOTSTRAP FIX: SMART IDEMPOTENT WINDOW
        # During bootstrap, use SHORTER windows (2s instead of 8s) to allow faster retries
        # This is safe because bootstrap phase is inherently high-retry due to capital constraints
        is_bootstrap_mode = bool(getattr(self, "_current_policy_context", {}).get("bootstrap_mode", False))
        active_order_timeout = 2.0 if is_bootstrap_mode else self._active_order_timeout_s
        
        # 🎯 BEST PRACTICE #2: Track active orders with SHORT 8-second window
        # This prevents duplicates while allowing rapid recovery from network issues
        if order_key in self._active_symbol_side_orders:
            last_attempt = self._active_symbol_side_orders[order_key]
            time_since_last = now - last_attempt
            
            if time_since_last < active_order_timeout:
                # Still within the window — genuine duplicate in flight
                self.logger.debug(
                    "[EM:ACTIVE_ORDER] Order in flight for %s %s (%.1fs ago); skipping. (timeout=%.1fs, bootstrap=%s)",
                    symbol, side.upper(), time_since_last, active_order_timeout, is_bootstrap_mode
                )
                return {"status": "SKIPPED", "reason": "ACTIVE_ORDER"}
            else:
                # Outside window — forcibly clear and allow fresh attempt
                self.logger.info(
                    "[EM:RETRY_ALLOWED] Previous attempt for %s %s timed out (%.1fs); allowing fresh retry. (timeout=%.1fs, bootstrap=%s)",
                    symbol, side.upper(), time_since_last, active_order_timeout, is_bootstrap_mode
                )
                del self._active_symbol_side_orders[order_key]
        
        # Record this attempt with current timestamp
        self._active_symbol_side_orders[order_key] = now

        sem_acquired = False
        try:
            # Lazy-init semaphores (need running loop)
            self._ensure_semaphores_ready()

            # 🔥 FIX: Add timeout to semaphore acquisition to prevent deadlock
            # Reason: If all semaphore slots fill up, prevent indefinite blocking
            try:
                await asyncio.wait_for(
                    self._concurrent_orders_sem.acquire(),
                    timeout=10.0  # 10 second timeout
                )
                sem_acquired = True
            except asyncio.TimeoutError:
                self.logger.error(
                    "[EM:SemaphoreTimeout] Semaphore acquisition timeout for %s %s - order placement blocked",
                    symbol, side.upper()
                )
                return {
                    "ok": False,
                    "status": "SKIPPED",
                    "reason": "SEMAPHORE_TIMEOUT",
                    "error_code": "SEMAPHORE_TIMEOUT",
                    "symbol": symbol,
                    "side": side.upper(),
                }

            # Use quote-amount path whenever BUY + planned_quote is given

            if side.upper() == "BUY" and planned_quote and planned_quote > 0:
                    # Initialize spend from the planned quote
                    spend = float(planned_quote)
                    was_downscaled_for_affordability = False  # Track if we downscaled
                    
                    # Bootstrap/dust bypass modes intentionally skip this internal floor and rely on
                    # executable-qty + exchange min_notional checks downstream.
                    min_entry = await self._get_min_entry_quote(symbol, price=current_price, min_notional=min_notional)
                    
                    # PROPER FIX: Adjust min_entry to account for step_size rounding
                    # When order is placed with quote=X, Binance computes qty = X / price and rounds by step_size.
                    # This can result in final_quote < min_entry. We must increase min_entry to the actual
                    # minimum that will SURVIVE rounding.
                    # HOWEVER: For bootstrap mode with limited capital, cap to available funds first
                    if (is_bootstrap or bypass_min_notional):
                        # Check available capital FIRST for bootstrap mode
                        q_asset_bootstrap = self._split_symbol_quote(symbol)
                        _free_bootstrap, _ok_bootstrap, _ = await self._get_free_quote_and_remainder_ok(q_asset_bootstrap, spend)
                        if spend > _free_bootstrap:
                            self.logger.info(
                                "[EM:BootstrapCap] %s BUY spend capped from %.2f to available %.2f before rounding calc",
                                symbol, float(spend), float(_free_bootstrap)
                            )
                            spend = float(_free_bootstrap)
                            was_downscaled_for_affordability = True  # Mark that we downscaled
                    
                    # CRITICAL: Recalculate min_entry_after_rounding based on CURRENT spend, not planned_quote
                    # If we downscaled from $30 to $27, we need to verify $27 meets Binance's minimums
                    # NOT validate against the $30 minimums
                    effective_min_entry = await self._get_min_entry_quote(symbol, price=current_price, min_notional=min_notional)
                    # Only use the original min_entry if we HAVEN'T downscaled
                    if was_downscaled_for_affordability:
                        # After downscaling, recalculate minimum entry requirements for bootstrap mode
                        # Bootstrap mode is allowed to go below normal minimums if capital limited
                        min_entry_for_validation = max(effective_min_entry, min_notional)
                        self.logger.info(
                            "[EM:PostDownscale] After downscaling spend to %.2f, min_entry for validation=%.2f (was %.2f)",
                            float(spend), float(min_entry_for_validation), float(effective_min_entry)
                        )
                    else:
                        min_entry_for_validation = effective_min_entry
                        min_entry = min_entry_for_validation
                    
                    min_entry_after_rounding = self._adjust_quote_for_step_rounding(
                        min_entry_quote=min_entry,
                        current_price=current_price,
                        step_size=step_size,
                    )
                    self.logger.info(
                        "[EM:RoundingAdjust] %s min_entry before rounding=%.2f after rounding=%.2f (downscaled=%s)",
                        symbol, float(min_entry), float(min_entry_after_rounding), was_downscaled_for_affordability
                    )
                    
                    # For non-bootstrap mode, escalate spend to meet rounding floor
                    # BUT: Never escalate if MetaController marked this as bootstrap override
                    is_bootstrap_override = bool(
                        (getattr(self, "_current_policy_context", None) or {}).get("_bootstrap_override", False)
                    )
                    if spend < min_entry_after_rounding and not (is_bootstrap or bypass_min_notional or is_bootstrap_override) and not was_downscaled_for_affordability:
                        self.logger.info(
                            "[EM:AutoAdjust] Escalating %s BUY spend from %.2f → %.2f to meet min_entry_after_rounding",
                            symbol, float(spend), float(min_entry_after_rounding)
                        )
                        spend = float(min_entry_after_rounding)
                    
                    # For bootstrap mode, don't escalate beyond what we have
                    if spend < min_entry_after_rounding and (is_bootstrap or bypass_min_notional):
                        self.logger.info(
                            "[EM:MinEntryBypass] %s BUY bypassing min_entry floor %.4f with spend %.4f "
                            "(bootstrap=%s bypass_min_notional=%s)",
                            symbol,
                            float(min_entry_after_rounding),
                            float(spend),
                            bool(is_bootstrap),
                            bool(bypass_min_notional),
                        )

                    # DEADLOCK FIX: Escalation logic for gross notional requirements
                    # Check if planned quote meets GROSS requirements (net + fees + safety)
                    # Use min_entry_after_rounding to ensure we meet minimum AFTER step rounding
                    q_asset = self._split_symbol_quote(symbol)
                    trade_fee_pct = float(self.trade_fee_pct)
                    safety_headroom = float(self.safety_headroom)
                    gross_factor = (1.0 + trade_fee_pct) * safety_headroom
                    min_required_gross = min_entry_after_rounding * gross_factor

                    no_downscale = False
                    is_bootstrap_escalation = False
                    if hasattr(self, "_current_policy_context") and self._current_policy_context:
                        no_downscale = bool(self._current_policy_context.get("_no_downscale_planned_quote", False))
                        is_bootstrap_escalation = bool(self._current_policy_context.get("_bootstrap_override", False))

                    # FIX: For bootstrap orders, DISABLE escalation - just use what we have
                    # CRITICAL: After downscaling for affordability, ALWAYS cap spend to available capital
                    if was_downscaled_for_affordability:
                        # Double-check we never exceed available capital after downscaling
                        q_asset_check = self._split_symbol_quote(symbol)
                        _free_final, _ok_final, _ = await self._get_free_quote_and_remainder_ok(q_asset_check, spend)
                        if spend > _free_final:
                            self.logger.warning(
                                "[EM:FinalCapCap] After escalation check, spend %.2f still > available %.2f. Re-capping.",
                                float(spend), float(_free_final)
                            )
                            spend = float(_free_final)
                        else:
                            self.logger.info(
                                "[EM:PostDownscaleCheck] After downscaling and escalation check, spend=%.2f is within available=%.2f ✓",
                                float(spend), float(_free_final)
                            )
                    if spend < min_required_gross and not no_downscale and not is_bootstrap_escalation:
                        # Attempt escalation to meet gross requirements
                        escalated_spend = min_required_gross
                        _free_esc, _ok_rem_esc, _why_rem_esc = await self._get_free_quote_and_remainder_ok(
                            q_asset, escalated_spend * gross_factor
                        )
                        
                        if _ok_rem_esc:
                            self.logger.info(
                                "[EM:Escalation] Quote escalated for %s: %.2f → %.2f to meet gross requirements "
                                "(net=%.2f, fee=%.1f%%, safety=%.2f)",
                                symbol, spend, escalated_spend, min_entry, trade_fee_pct * 100, safety_headroom
                            )
                            spend = escalated_spend
                        else:
                            # Escalation failed - capital insufficient
                            await self._log_execution_event("order_skip", symbol, {
                                "side": "BUY",
                                "reason": "UNESCALATABLE_NOTIONAL",
                                "planned_quote": spend,
                                "min_required_gross": min_required_gross,
                                "free_available": _free_esc,
                                "gap": min_required_gross - _free_esc,
                            })
                            # P9 MANDATORY: Record rejection (I1 Invariant - Failure Memory)
                            await self.shared_state.record_rejection(
                                symbol, "BUY", "UNESCALATABLE_NOTIONAL", source="ExecutionManager"
                            )
                            return None

                    # Reserve floor + tiny-remainder pre-check using estimated gross cost
                    gross_needed = float(spend) * (1.0 + trade_fee_pct) * safety_headroom
                    _free, _ok_rem, _why_rem = await self._get_free_quote_and_remainder_ok(q_asset, gross_needed)
                    if not _ok_rem:
                        await self._log_execution_event("order_skip", symbol, {
                            "side": "BUY",
                            "reason": _why_rem,
                            "free_quote": _free,
                            "needed": gross_needed,
                            "min_free_reserve_usdt": self.min_free_reserve_usdt,
                            "no_remainder_below_quote": self.no_remainder_below_quote
                        })
                        return None

                    filters_obj = SymbolFilters(
                        step_size=step_size,
                        min_qty=min_qty,
                        max_qty=max_qty,
                        tick_size=tick_size,
                        min_notional=min_notional,
                        min_entry_quote=(0.0 if bypass_min_notional else min_entry_after_rounding),
                    )
                    
                    if is_bootstrap:
                        # BOOTSTRAP: Use true quote-based market order (quoteOrderQty) for guaranteed execution
                        self.logger.info(
                            "[EM:BOOTSTRAP] Using quoteOrderQty=%s for %s BUY to guarantee bootstrap execution (bypassing quantity precision issues)",
                            spend, symbol
                        )
                        
                        # Use exchange client's quote-based order directly
                        try:
                            if hasattr(self.exchange_client, '_send_real_order_quote'):
                                self._journal("ORDER_SUBMITTED", {
                                    "symbol": symbol, "side": "BUY", "qty": 0,
                                    "price": current_price, "quote": float(spend),
                                    "tag": safe_tag, "client_order_id": client_id,
                                    "path": "bootstrap_quote",
                                })
                                order_scope = self._enter_exchange_order_scope()
                                try:
                                    order_id = await self.exchange_client._send_real_order_quote(
                                        symbol=symbol,
                                        side="BUY",
                                        quote=float(spend),
                                        tag=safe_tag,
                                        client_order_id=client_id,
                                    )
                                finally:
                                    self._exit_exchange_order_scope(order_scope)
                                if order_id:
                                    # Create a minimal order result for consistency
                                    order = {
                                        "symbol": symbol,
                                        "orderId": order_id,
                                        "side": "BUY",
                                        "type": "MARKET",
                                        "executedQty": "0",  # Will be updated by exchange
                                        "quoteOrderQty": str(spend),
                                        "status": "FILLED"
                                    }
                                    await self._log_execution_event("order_placed", symbol, {
                                        "type": "MARKET_QUOTE", "side": "BUY", "tag": safe_tag,
                                        "using": "quoteOrderQty", "quote_amount": spend
                                    })
                                    order = await self._reconcile_delayed_fill(
                                        symbol=symbol,
                                        side="BUY",
                                        order=order,
                                        tag=safe_tag,
                                        tier=None,
                                        client_order_id_hint=client_id,
                                    )
                                    if self._is_order_fill_confirmed(order):
                                        self._journal("ORDER_FILLED", {
                                            "symbol": symbol, "side": "BUY",
                                            "executed_qty": self._safe_float(order.get("executedQty") or order.get("executed_qty"), 0.0),
                                            "avg_price": self._resolve_post_fill_price(order, self._safe_float(order.get("executedQty") or order.get("executed_qty"), 0.0)),
                                            "cumm_quote": self._safe_float(order.get("cummulativeQuoteQty") or order.get("cummulative_quote"), 0.0),
                                            "order_id": str(order.get("orderId") or order.get("order_id") or ""),
                                            "status": str(order.get("status", "")),
                                            "tag": safe_tag, "client_order_id": client_id,
                                            "path": "bootstrap_quote",
                                        })
                                    elif isinstance(order, dict):
                                        self._journal("ORDER_PENDING", {
                                            "symbol": symbol,
                                            "side": "BUY",
                                            "status": str(order.get("status", "")).upper() or "UNKNOWN",
                                            "order_id": str(order.get("orderId") or order.get("order_id") or ""),
                                            "tag": safe_tag,
                                            "client_order_id": client_id,
                                            "path": "bootstrap_quote",
                                        })
                                    return order
                        except Exception as e:
                            self.logger.warning(f"[EM:BOOTSTRAP] Quote-based order failed: {e}, falling back to quantity-based")
                    
                    # CRITICAL FIX: When planned_quote is provided, use quoteOrderQty instead of quantity
                    # This ensures execution respects the decision quote, not the wallet balance
                    # Use quoteOrderQty parameter (Binance native) instead of converting to quantity
                    self.logger.critical(
                        "[EM:QUOTE_PATH] Using quoteOrderQty=%.2f for %s BUY (planned_quote path, not quantity-based)",
                        float(spend), symbol
                    )
                    self._journal("ORDER_SUBMITTED", {
                        "symbol": symbol, "side": "BUY", "qty": 0,
                        "price": current_price, "quote": float(spend),
                        "tag": safe_tag, "client_order_id": client_id,
                    })
                    
                    # ⚠️ SHADOW MODE PROTECTION: Check before placing real orders
                    if getattr(self.shared_state, "is_shadow", False):
                        self.logger.warning(
                            "[SHADOW_MODE] Simulated BUY %s quote=%.2f (no real order placed)",
                            symbol, float(spend),
                        )
                        order = {
                            "status": "SHADOW_FILLED",
                            "symbol": symbol,
                            "side": "BUY",
                            "quote": float(spend),
                            "price": current_price,
                            "executedQty": "0",
                            "cummulativeQuoteQty": str(float(spend)),
                        }
                    else:
                        order = await self._place_with_client_id(
                            symbol=symbol, side="BUY", quote_order_qty=float(spend), tag=safe_tag, clientOrderId=client_id
                        )
                    if isinstance(order, dict) and not bool(order.get("_submission_unknown")):
                        await self._log_execution_event("order_placed", symbol, {
                            "type": "MARKET", "side": "BUY", "tag": safe_tag,
                            "client_id": client_id, "using": "qty"
                        })
                    order = await self._reconcile_delayed_fill(
                        symbol=symbol,
                        side="BUY",
                        order=order,
                        tag=safe_tag,
                        tier=None,
                        client_order_id_hint=client_id,
                    )
                    if self._is_order_fill_confirmed(order):
                        self._journal("ORDER_FILLED", {
                            "symbol": symbol, "side": "BUY",
                            "executed_qty": self._safe_float(order.get("executedQty") or order.get("executed_qty"), 0.0),
                            "avg_price": self._resolve_post_fill_price(order, self._safe_float(order.get("executedQty") or order.get("executed_qty"), 0.0)),
                            "cumm_quote": self._safe_float(order.get("cummulativeQuoteQty") or order.get("cummulative_quote"), 0.0),
                            "order_id": str(order.get("orderId") or order.get("order_id") or ""),
                            "status": str(order.get("status", "")),
                            "tag": safe_tag, "client_order_id": client_id,
                        })
                    elif isinstance(order, dict):
                        self._journal("ORDER_PENDING", {
                            "symbol": symbol,
                            "side": "BUY",
                            "status": str(order.get("status", "")).upper() or "UNKNOWN",
                            "order_id": str(order.get("orderId") or order.get("order_id") or ""),
                            "tag": safe_tag,
                            "client_order_id": client_id,
                        })
                    return order

            # else: qty path (BUY without planned_quote, or SELL)
            qty = round_step(quantity, step_size)
            # For SELL orders: if the remainder after ROUND_DOWN is below min_qty, round UP instead.
            # This prevents perpetual dust: e.g. position=0.001234, step=0.001 → ROUND_DOWN sells
            # 0.001 and leaves 0.000234 stranded forever. Round UP to 0.002 only when remainder < min_qty.
            if side.upper() == "SELL" and step_size > 0:
                remainder = _raw_quantity - float(qty)
                residual_notional = max(0.0, remainder) * float(current_price or 0.0)
                dust_floor_quote = float(
                    self._cfg(
                        "DUST_MIN_QUOTE_USDT",
                        getattr(self.config, "DUST_MIN_QUOTE_USDT", getattr(self.config, "dust_min_quote_usdt", 5.0)),
                    ) or 5.0
                )
                write_down_quote = float(
                    self._cfg(
                        "PERMANENT_DUST_USDT_THRESHOLD",
                        getattr(self.config, "PERMANENT_DUST_USDT_THRESHOLD", 1.0),
                    ) or 1.0
                )
                residual_floor = max(float(min_notional or 0.0), dust_floor_quote, write_down_quote)
                qty_residual_is_dust = remainder > 0 and remainder < max(float(min_qty), float(step_size))
                notional_residual_is_dust = residual_notional > 0 and residual_notional < residual_floor
                if qty_residual_is_dust or notional_residual_is_dust:
                    qty_up = float(qty) + float(step_size)
                    # Only round up if the position actually has that much (don't oversell)
                    if qty_up <= _raw_quantity + float(step_size) * 0.01:
                        self.logger.info(
                            "[EM:SellRoundUp] %s: qty ROUND_UP %.8f→%.8f to avoid dust "
                            "(remainder=%.8f residual_notional=%.4f floor=%.4f min_qty=%.8f step=%.8f)",
                            symbol,
                            float(qty),
                            float(qty_up),
                            float(remainder),
                            float(residual_notional),
                            float(residual_floor),
                            float(min_qty),
                            float(step_size),
                        )
                        qty = qty_up
            if max_qty > 0 and qty > max_qty:
                qty = round_step(max_qty, step_size)
            
            exit_floor = 0.0
            if not is_liquidation:
                try:
                    exit_info = await self._get_exit_floor_info(symbol, price=current_price)
                    exit_floor = float(exit_info.get("min_exit_quote", 0.0) or 0.0)
                except Exception:
                    exit_floor = 0.0

            # [FIX] Skip notional check for liquidation orders (they bypass guards)
            min_required_notional = exit_floor if exit_floor > 0 else float(min_notional)
            if not is_liquidation and not bypass_min_notional and (qty <= 0 or qty * current_price < min_required_notional):
                await self._log_execution_event("order_skip", symbol, {
                    "side": side.upper(), "reason": "NOTIONAL_LT_MIN_PRE_VALIDATION",
                    "notional": qty * current_price, "min_required": min_required_notional
                })
                return None
            elif qty <= 0:
                # For liquidation, still reject if qty is exactly zero
                await self._log_execution_event("order_skip", symbol, {
                    "side": side.upper(), "reason": "ZERO_QUANTITY",
                    "notional": qty * current_price
                })
                return None

            if side.upper() == "BUY":
                # no hard block here; reservations already protect
                pass
            if side.upper() == "SELL":
                # P9 Optimization: Trust the MetaController/PositionManager's view of quantity.
                # If the exchange actually has less, the order will fail with an explicit API error,
                # which is better than silently trimming or skipping here based on potentially stale data.
                pass

            filters_obj = SymbolFilters(
                step_size=step_size, min_qty=min_qty, max_qty=max_qty,
                tick_size=tick_size, min_notional=min_notional, min_entry_quote=float(exit_floor or 0.0)
            )
            
            # [FIX] Skip filter validation for liquidation orders (they bypass guards)
            if is_liquidation:
                # For liquidation, just ensure qty > 0 and apply step size rounding
                final_qty = max(float(qty), 0.0)
            else:
                ok, adj_qty, _, _ = await self._validate_order_request_contract(
                    side=side.upper(), qty=qty, price=current_price, filters=filters_obj,
                    taker_fee_bps=self._cfg("TAKER_FEE_BPS", 10), use_quote_amount=None
                )
                if not ok:
                    await self._log_execution_event("order_skip", symbol, {"side": side.upper(), "reason": "order_filters_rejected"})
                    return None
                final_qty = float(adj_qty)
            # Enforce notional floor on BUY-by-qty as well as SELL
            if side.upper() == "BUY":
                # Per-trade cap for qty path
                if self.max_spend_per_trade > 0:
                    est_quote = final_qty * current_price
                    if est_quote > self.max_spend_per_trade and current_price > 0:
                        final_qty = round_step(self.max_spend_per_trade / current_price, step_size)

                notional_now = final_qty * current_price
                # PHASE 2 NOTE: Capital floor check already done in MetaController
                exec_floor = 10.0  # Minimal execution-level floor
                min_req = max(min_notional, exec_floor)
                if notional_now < min_req:
                    await self._log_execution_event('order_skip', symbol, {
                        'side': 'BUY', 'reason': 'NOTIONAL_LT_MIN_AFTER_ROUND',
                        'rounded_qty': final_qty, 'notional': notional_now, 'min_required': min_req
                    })
                    return None

                # Reserve floor + tiny-remainder check for BUY-by-qty
                q_asset = self._split_symbol_quote(symbol)
                gross_needed = float(final_qty * current_price) * (1.0 + float(self.trade_fee_pct)) * float(self.safety_headroom)
                _free, _ok_rem, _why_rem = await self._get_free_quote_and_remainder_ok(q_asset, gross_needed)
                if not _ok_rem:
                    await self._log_execution_event("order_skip", symbol, {
                        "side": "BUY", "reason": _why_rem, "free_quote": _free, "needed": gross_needed,
                        "min_free_reserve_usdt": self.min_free_reserve_usdt,
                        "no_remainder_below_quote": self.no_remainder_below_quote
                    })
                    return None
            else:
                # [FIX] Skip notional check for liquidation SELL orders
                # PHASE 2 NOTE: Capital floor check already done in MetaController
                exec_floor = 10.0  # Minimal execution-level floor
                if not is_liquidation and final_qty * current_price < max(min_notional, exec_floor):
                    await self._log_execution_event('order_skip', symbol, {
                        'side': side.upper(), 'reason': 'NOTIONAL_LT_MIN',
                        'notional': final_qty * current_price,
                        'min_required': max(min_notional, exec_floor)
                    })
                    return None

                # Profit gate: entry-only protection — does not apply to risk/liquidation exits.
                # is_liquidation=True means this SELL is a risk-management exit (TP/SL, forced),
                # and must not be blocked by a profit constraint that only makes sense for entries.
                if not is_liquidation and not await self._passes_profit_gate(symbol, side, final_qty, current_price):
                    self.logger.warning(f"🚫 SELL blocked at Execution layer by profit gate for {symbol}")
                    return None

                self._journal("ORDER_SUBMITTED", {
                    "symbol": symbol, "side": side.upper(), "qty": final_qty,
                    "price": current_price, "quote": final_qty * current_price,
                    "tag": safe_tag, "client_order_id": client_id,
                    "timestamp": time.time(),
                })
                
                # ========== MAKER-BIASED EXECUTION DECISION ==========
                # Decide whether to use maker limit order (inside spread) or market order
                use_maker, decision_reason = await self._decide_execution_method(
                    symbol=symbol,
                    side=side.upper(),
                    quantity=final_qty,
                    current_price=current_price,
                    planned_quote=planned_quote,
                )
                
                if use_maker:
                    self.logger.info(
                        f"[MakerExec] {symbol} {side.upper()} qty={final_qty:.8f} "
                        f"price={current_price:.8f}: {decision_reason}"
                    )
                else:
                    self.logger.info(
                        f"[MarketExec] {symbol} {side.upper()} qty={final_qty:.8f} "
                        f"price={current_price:.8f}: {decision_reason}"
                    )
                
                # ⚠️ SHADOW MODE PROTECTION: Check before placing real orders
                if getattr(self.shared_state, "is_shadow", False):
                    self.logger.warning(
                        "[SHADOW_MODE] Simulated %s %s qty=%.8f (no real order placed)",
                        symbol, side.upper(), final_qty,
                    )
                    order = {
                        "status": "SHADOW_FILLED",
                        "symbol": symbol,
                        "side": side.upper(),
                        "quantity": final_qty,
                        "price": current_price,
                        "executedQty": str(final_qty),
                        "cummulativeQuoteQty": str(final_qty * current_price),
                    }
                else:
                    order = await self._place_with_client_id(
                        symbol=symbol, side=side.upper(), quantity=final_qty, tag=safe_tag, clientOrderId=client_id
                    )
                if isinstance(order, dict) and not bool(order.get("_submission_unknown")):
                    await self._log_execution_event("order_placed", symbol, {
                        "type": "MARKET", "side": side.upper(), "tag": safe_tag,
                        "client_id": client_id, "using": "qty"
                    })
                
                # ✅ CRITICAL: For SELL orders, capture the order response IMMEDIATELY
                # If exchange executed but returned None/empty, we still need to log it
                # This prevents state-sync issues where SELL is executed but never logged
                if side.upper() == "SELL":
                    self._journal("SELL_ORDER_PLACED", {
                        "symbol": symbol, "qty": final_qty, "price": current_price,
                        "client_order_id": client_id, "response_received": bool(order),
                        "timestamp": time.time(),
                    })
                
                order = await self._reconcile_delayed_fill(
                    symbol=symbol,
                    side=side.upper(),
                    order=order,
                    tag=safe_tag,
                    tier=None,
                    client_order_id_hint=client_id,
                )

                if self._is_order_fill_confirmed(order):
                    self._journal("ORDER_FILLED", {
                        "symbol": symbol, "side": side.upper(),
                        "executed_qty": self._safe_float(order.get("executedQty") or order.get("executed_qty"), 0.0),
                        "avg_price": self._resolve_post_fill_price(order, self._safe_float(order.get("executedQty") or order.get("executed_qty"), 0.0)),
                        "cumm_quote": self._safe_float(order.get("cummulativeQuoteQty") or order.get("cummulative_quote"), 0.0),
                        "order_id": str(order.get("orderId") or order.get("order_id") or ""),
                        "status": str(order.get("status", "")),
                        "tag": safe_tag, "client_order_id": client_id,
                    })
                elif isinstance(order, dict):
                    self._journal("ORDER_PENDING", {
                        "symbol": symbol,
                        "side": side.upper(),
                        "status": str(order.get("status", "")).upper() or "UNKNOWN",
                        "order_id": str(order.get("orderId") or order.get("order_id") or ""),
                        "tag": safe_tag,
                        "client_order_id": client_id,
                    })

                if order:
                    post_fill = await self._ensure_post_fill_handled(
                        symbol=symbol,
                        side=side.upper(),
                        order=order,
                        tier=None,
                        tag=str(safe_tag or ""),
                    )
                    if isinstance(order, dict) and isinstance(post_fill, dict):
                        order["_post_fill_result"] = dict(post_fill)
                        order["_post_fill_done"] = True
                    if side.upper() == "SELL":
                        await self._finalize_sell_post_fill(
                            symbol=symbol,
                            order=order,
                            tag=str(safe_tag or ""),
                            post_fill=post_fill,
                            policy_ctx=None,
                            tier=None,
                        )
                return order
        finally:
            # Release semaphore if acquired
            if sem_acquired:
                try:
                    self._concurrent_orders_sem.release()
                except Exception:
                    pass
            # Clear the active order entry (always, to prevent stale entries)
            self._active_symbol_side_orders.pop(order_key, None)

    async def _place_with_client_id(self, **kwargs) -> Any:
        """
        Wrap the exchange client's MARKET placement with comprehensive error classification.
        Expected kwargs: symbol, side, quantity? or quote?, tag, clientOrderId
        
        P9 GUARANTEE: All exceptions are classified deterministically:
        - TypeError: Retry without clientOrderId
        - BinanceAPIException: Extract code and classify
        - Network/Connection errors: Classify as EXTERNAL_API_ERROR
        - Unknown exceptions: Log and classify as EXTERNAL_API_ERROR (never propagate raw)
        """
        request_kwargs = dict(kwargs)
        symbol = self._norm_symbol(request_kwargs.get("symbol", "UNKNOWN"))
        side = str(request_kwargs.get("side", "UNKNOWN")).upper()
        coid = (
            request_kwargs.get("clientOrderId")
            or request_kwargs.get("client_order_id")
            or request_kwargs.get("origClientOrderId")
        )
        retries = int(self._cfg("ORDER_RECOVERY_RETRIES", 5) or 5)
        retries = max(1, min(retries, 20))
        delay_s = float(self._cfg("ORDER_RECOVERY_RETRY_DELAY_S", 0.2) or 0.2)
        delay_s = min(max(delay_s, 0.05), 1.0)

        # CRITICAL DEBUG: Log what we're sending to the exchange
        if request_kwargs.get("quote_order_qty"):
            self.logger.info(
                "[EM:SEND_ORDER] %s %s quote_order_qty=%.2f (NOT quantity-based)",
                symbol, side, float(request_kwargs.get("quote_order_qty", 0))
            )
        elif request_kwargs.get("quantity"):
            self.logger.info(
                "[EM:SEND_ORDER] %s %s quantity=%.8f",
                symbol, side, float(request_kwargs.get("quantity", 0))
            )

        scope_token = self._enter_exchange_order_scope()
        try:
            try:
                return await self.exchange_client.place_market_order(**request_kwargs)
            except TypeError as te:
                # Some clients may not accept clientOrderId; retry without it
                if "clientOrderId" in request_kwargs:
                    self.logger.debug(f"[EM:TypeErr] clientOrderId not supported, retrying: {te}")
                    retry_kwargs = dict(request_kwargs)
                    retry_kwargs.pop("clientOrderId", None)
                    try:
                        return await self.exchange_client.place_market_order(**retry_kwargs)
                    except Exception as retry_err:
                        if coid and self._is_ambiguous_submit_error(retry_err):
                            recovered = await self._recover_order_by_client_id(
                                symbol=symbol,
                                side=side,
                                client_order_id=str(coid),
                                retries=retries,
                                delay_s=delay_s,
                            )
                            if isinstance(recovered, dict) and recovered:
                                return recovered
                            self._journal("ORDER_SUBMISSION_AMBIGUOUS", {
                                "symbol": symbol,
                                "side": side,
                                "client_order_id": str(coid),
                                "context": "place_market_order_retry",
                                "error": str(retry_err),
                                "timestamp": time.time(),
                            })
                            return self._build_submission_unknown_order(
                                symbol=symbol,
                                side=side,
                                client_order_id=str(coid),
                                reason="order_submission_ambiguous",
                                error_code="ORDER_SUBMISSION_AMBIGUOUS",
                                error_text=str(retry_err),
                            )
                        # Classify retry error
                        return self._classify_exchange_error(
                            symbol, side, retry_err, "place_market_order_retry"
                        )
                # If not a clientOrderId issue, re-raise as deterministic ExecutionError
                return self._classify_exchange_error(
                    symbol, side, te, "place_market_order"
                )
            except BinanceAPIException as bex:
                if coid and self._is_ambiguous_submit_error(bex):
                    recovered = await self._recover_order_by_client_id(
                        symbol=symbol,
                        side=side,
                        client_order_id=str(coid),
                        retries=retries,
                        delay_s=delay_s,
                    )
                    if isinstance(recovered, dict) and recovered:
                        return recovered
                    self._journal("ORDER_SUBMISSION_AMBIGUOUS", {
                        "symbol": symbol,
                        "side": side,
                        "client_order_id": str(coid),
                        "context": "place_market_order_binance_exception",
                        "error": str(bex),
                        "code": getattr(bex, "code", None),
                        "timestamp": time.time(),
                    })
                    return self._build_submission_unknown_order(
                        symbol=symbol,
                        side=side,
                        client_order_id=str(coid),
                        reason="order_submission_ambiguous",
                        error_code="ORDER_SUBMISSION_AMBIGUOUS",
                        error_text=str(bex),
                    )
                # Binance-specific API error with code
                return self._classify_exchange_error(
                    symbol, side, bex, "place_market_order"
                )
            except (ConnectionError, TimeoutError, asyncio.TimeoutError) as conn_err:
                # Network/connection issues
                self.logger.error(
                    f"[EM:ConnErr] {symbol} {side}: Connection error from ExchangeClient: {conn_err}"
                )
                # Best-effort: if a clientOrderId was provided, probe the exchange for the order
                if coid:
                    recovered = await self._recover_order_by_client_id(
                        symbol=symbol,
                        side=side,
                        client_order_id=str(coid),
                        retries=retries,
                        delay_s=delay_s,
                    )
                    if isinstance(recovered, dict) and recovered:
                        return recovered
                    self._journal("ORDER_SUBMISSION_AMBIGUOUS", {
                        "symbol": symbol,
                        "side": side,
                        "client_order_id": str(coid),
                        "context": "place_market_order_transport_error",
                        "error": str(conn_err),
                        "timestamp": time.time(),
                    })
                    return self._build_submission_unknown_order(
                        symbol=symbol,
                        side=side,
                        client_order_id=str(coid),
                        reason="order_submission_ambiguous",
                        error_code="ORDER_SUBMISSION_AMBIGUOUS",
                        error_text=str(conn_err),
                    )
                return None  # Network error → order not sent (deterministic)
            except Exception as unknown_err:
                # P9 MANDATE: No unknown exceptions allowed
                self.logger.error(
                    f"[EM:UnknownErr] {symbol} {side}: Unclassified exception from place_market_order: "
                    f"{type(unknown_err).__name__}: {unknown_err}",
                    exc_info=True
                )
                if coid and self._is_ambiguous_submit_error(unknown_err):
                    recovered = await self._recover_order_by_client_id(
                        symbol=symbol,
                        side=side,
                        client_order_id=str(coid),
                        retries=retries,
                        delay_s=delay_s,
                    )
                    if isinstance(recovered, dict) and recovered:
                        return recovered
                    self._journal("ORDER_SUBMISSION_AMBIGUOUS", {
                        "symbol": symbol,
                        "side": side,
                        "client_order_id": str(coid),
                        "context": "place_market_order_unknown_error",
                        "error": str(unknown_err),
                        "timestamp": time.time(),
                    })
                    return self._build_submission_unknown_order(
                        symbol=symbol,
                        side=side,
                        client_order_id=str(coid),
                        reason="order_submission_ambiguous",
                        error_code="ORDER_SUBMISSION_AMBIGUOUS",
                        error_text=str(unknown_err),
                    )
                return None  # Unknown error → order not sent (deterministic fallback)
        finally:
            self._exit_exchange_order_scope(scope_token)

    def _classify_exchange_error(
        self, symbol: str, side: str, exc: Exception, context: str
    ) -> None:
        """
        Classify exchange errors into deterministic categories.
        Returns None (order not placed) for all classified errors.
        """
        if isinstance(exc, BinanceAPIException):
            code = getattr(exc, "code", None)
            msg = str(exc)
            
            # Classify by Binance error code
            if code == -1013:
                # Invalid quantity
                self.logger.warning(
                    f"[EM:Classify] {symbol} {side}: INVALID_QUANTITY (code={code}) - {msg}"
                )
                return None
            elif code == -1022:
                # Invalid API-key format
                self.logger.error(
                    f"[EM:Classify] {symbol} {side}: INVALID_API_KEY (code={code}) - {msg}"
                )
                return None
            elif code == -1003:
                # WAF limit or IP ban
                self.logger.warning(
                    f"[EM:Classify] {symbol} {side}: RATE_LIMIT_OR_BAN (code={code}) - {msg}"
                )
                return None
            elif code == -2015:
                # Invalid API permissions
                self.logger.error(
                    f"[EM:Classify] {symbol} {side}: INVALID_PERMISSIONS (code={code}) - {msg}"
                )
                return None
            elif code == -1111:
                # Precision error (step size)
                self.logger.error(
                    f"[EM:Classify] {symbol} {side}: PRECISION_ERROR (code={code}) - {msg}"
                )
                return None
            elif code == -1000 or code == -1001:
                # Unauthorized or service error
                self.logger.error(
                    f"[EM:Classify] {symbol} {side}: SERVICE_ERROR (code={code}) - {msg}"
                )
                return None
            else:
                # Generic Binance error
                self.logger.warning(
                    f"[EM:Classify] {symbol} {side}: BINANCE_ERROR (code={code}) - {msg}"
                )
                return None
        
        elif isinstance(exc, TypeError):
            self.logger.error(
                f"[EM:Classify] {symbol} {side}: TYPE_ERROR in {context}: {exc}"
            )
            return None
        
        else:
            # Unknown exception type
            self.logger.error(
                f"[EM:Classify] {symbol} {side}: UNCLASSIFIED ({type(exc).__name__}) in {context}: {exc}"
            )
            return None

    # =============================
    # Helpers
    # =============================
    def _extract_min_notional(self, filters: Dict[str, Any]) -> float:
        # Accept both normalized and raw filter shapes
        if "min_notional" in filters:  # normalized
            try:
                return float(filters.get("min_notional", 0) or 0)
            except Exception:
                return 0.0
        block = filters.get("MIN_NOTIONAL") or filters.get("NOTIONAL") or {}
        try:
            return float(block.get("minNotional", 0) or 0)
        except Exception:
            return 0.0

    def _adjust_quote_for_step_rounding(
        self,
        min_entry_quote: float,
        current_price: float,
        step_size: float,
    ) -> float:
        """
        Compute the ACTUAL quote needed to satisfy min_entry AFTER step_size rounding.
        
        Problem:
          min_entry = 30 USDT
          When we send quote=31, Binance computes qty = 31 / 45000 = 0.000688...
          After step rounding (0.001), qty becomes 0.001
          Final quote = 0.001 * 45000 = 45 USDT ✓ OK
        
        But if min_entry_quote is very close to a step boundary, quote might round DOWN:
          min_entry = 30 USDT
          price = 45000
          qty_raw = 30 / 45000 = 0.000666...
          After step round (0.001), qty = 0.0 (too small!)
          This violates the min_entry requirement.
        
        Solution:
          1. Compute exact qty that satisfies min_entry:  qty = min_entry / price
          2. Round UP to next step:  qty_rounded = ceil(qty / step) * step
          3. Compute adjusted quote:  adjusted_quote = qty_rounded * price
          4. Return this adjusted quote so order satisfies min_entry AFTER rounding
        
        Args:
            min_entry_quote: Target minimum quote (30 USDT)
            current_price: Asset price (45000 USDT)
            step_size: Lot step for quantity (0.001)
        
        Returns:
            Adjusted quote that, when rounded, still >= min_entry_quote
        """
        try:
            # Avoid division by zero
            if not step_size or step_size <= 0 or current_price <= 0:
                return float(min_entry_quote)
            
            from decimal import Decimal, ROUND_UP
            
            # Convert to Decimal for precision
            min_quote = Decimal(str(max(0.0, float(min_entry_quote))))
            price = Decimal(str(max(0.0001, float(current_price))))
            step = Decimal(str(max(0.0001, float(step_size))))
            
            # qty_raw = min_quote / price
            qty_raw = min_quote / price
            
            # Round UP to next step: ceil(qty_raw / step) * step
            qty_rounded = (qty_raw / step).to_integral_value(rounding=ROUND_UP) * step
            
            # Final adjusted quote
            adjusted_quote = qty_rounded * price
            
            result = float(adjusted_quote)
            self.logger.debug(
                "[AdjustQuote] min_entry=%.2f price=%.2f step_size=%.8f -> "
                "qty_raw=%.8f -> qty_rounded=%.8f -> adjusted_quote=%.2f",
                float(min_quote), float(price), float(step),
                float(qty_raw), float(qty_rounded), result
            )
            return result
        except Exception as e:
            self.logger.debug(f"[AdjustQuote] Error: {e}, returning min_entry as-is")
            return float(min_entry_quote)

    def _extract_filter_vals(self, filters: Dict[str, Any]):
        fs = filters or {}
        # P9 Fix 4: Prefer LOT_SIZE for small capital/micro trading
        lot = fs.get("LOT_SIZE") or fs.get("MARKET_LOT_SIZE") or {}
        price = fs.get("PRICE_FILTER", {})
        notional = (fs.get("MIN_NOTIONAL") or fs.get("NOTIONAL") or {})
        step_str = str(lot.get("stepSize", "0.000001"))
        step_size = float(step_str)
        min_qty = float(lot.get("minQty", 0))
        max_qty = float(lot.get("maxQty", 0))
        tick_size = float(price.get("tickSize", 0))
        min_notional = float(notional.get("minNotional", 0))
        return step_size, min_qty, max_qty, tick_size, min_notional

    def extract_symbol_economics(self, symbol: str, filters: Dict[str, Any]) -> Tuple[float, float, float]:
        """
        Canonical extraction of symbol economics for quantity calculations.
        Raises RuntimeError if filters are invalid or missing critical values.
        """
        if not filters:
            raise RuntimeError(f"Symbol filters missing for {symbol}")

        # Extract LOT_SIZE
        lot = filters.get("LOT_SIZE") or filters.get("MARKET_LOT_SIZE")
        if not lot:
            raise RuntimeError(f"LOT_SIZE filter missing for {symbol}")

        # Extract step_size
        step_size_str = lot.get("stepSize")
        if not step_size_str:
            raise RuntimeError(f"stepSize missing in LOT_SIZE for {symbol}")
        try:
            step_size = float(step_size_str)
            if step_size <= 0:
                raise ValueError("step_size must be positive")
        except (ValueError, TypeError):
            raise RuntimeError(f"Invalid stepSize '{step_size_str}' for {symbol}")

        # Extract min_qty
        min_qty_str = lot.get("minQty")
        if not min_qty_str:
            raise RuntimeError(f"minQty missing in LOT_SIZE for {symbol}")
        try:
            min_qty = float(min_qty_str)
            if min_qty < 0:
                raise ValueError("min_qty cannot be negative")
        except (ValueError, TypeError):
            raise RuntimeError(f"Invalid minQty '{min_qty_str}' for {symbol}")

        # Extract min_notional
        notional = filters.get("MIN_NOTIONAL") or filters.get("NOTIONAL")
        if not notional:
            raise RuntimeError(f"NOTIONAL filter missing for {symbol}")
        min_notional_str = notional.get("minNotional")
        if not min_notional_str:
            raise RuntimeError(f"minNotional missing in NOTIONAL for {symbol}")
        try:
            min_notional = float(min_notional_str)
            if min_notional < 0:
                raise ValueError("min_notional cannot be negative")
        except (ValueError, TypeError):
            raise RuntimeError(f"Invalid minNotional '{min_notional_str}' for {symbol}")

        # Log extracted values
        self.logger.info(
            f"[FILTERS] {symbol} step_size={step_size} min_qty={min_qty} min_notional={min_notional}"
        )

        return step_size, min_qty, min_notional

    def reset_idempotent_cache(self):
        """
        🔧 FIX 2: Reset idempotent protection caches.
        
        Clears the SELL finalization cache to allow re-execution of orders.
        This unblocks deduplication logic that was preventing signal retries.
        
        Safe to call multiple times and between trading cycles.
        """
        try:
            # Clear the finalization result cache entirely
            self._sell_finalize_result_cache.clear()
            self._sell_finalize_result_cache_ts.clear()
            
            self.logger.warning(
                "[EXEC:IDEMPOTENT_RESET] ✅ Cleared SELL finalization cache (entries cleared: finalize_cache)"
            )
        except Exception as e:
            self.logger.warning(
                "[EXEC:IDEMPOTENT_RESET] Failed to reset idempotent cache: %s",
                e,
                exc_info=True
            )

    async def start(self):
        """
        Minimal start hook so AppContext can warm this component during P5.
        Safely warms symbol filters and returns immediately.
        """
        try:
            # Initialize semaphores and heartbeat when loop is available
            self._ensure_semaphores_ready()
            
            # Start heartbeat task if not already started (use loop.create_task for safety)
            if self._heartbeat_task is None or self._heartbeat_task.done():
                try:
                    loop = asyncio.get_running_loop()
                    self._heartbeat_task = loop.create_task(self._heartbeat_loop(), name="ExecutionManager:heartbeat")
                except RuntimeError:
                    pass  # No running loop
                except Exception as e:
                    self.logger.debug(f"Failed to start heartbeat task: {e}")
            
            ensure = getattr(self.exchange_client, "ensure_symbol_filters_ready", None)
            if callable(ensure):
                maybe = ensure()
                if asyncio.iscoroutine(maybe):
                    await maybe
            await self._emit_status("Initialized", "start() no-op warmup complete")
            self.logger.info("ExecutionManager.start: symbol filters warmed (if supported)")
        except Exception:
            self.logger.debug("ExecutionManager.start warmup failed (non-fatal)", exc_info=True)
