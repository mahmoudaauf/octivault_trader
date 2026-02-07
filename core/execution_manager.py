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
    from decimal import Decimal, ROUND_DOWN
    # uuid import removed (unused)
    from typing import Any, Dict, Optional, Tuple, Union, Literal
    from core.stubs import resilient_trade, maybe_call
    from core.shared_state import PendingPositionIntent

    # =============================
    # Utility shims (maybe_call, round_step, resilient_trade)
    # =============================
    try:
        from utils import shared_state_tools, indicators, pnl_calculator
    except Exception:
        import asyncio as _asyncio
        from functools import wraps as _wraps

    def round_step(value: float, step: float) -> float:
        if step <= 0: return float(value)
        q = (Decimal(str(value)) / Decimal(str(step))).to_integral_value(rounding=ROUND_DOWN)
        return float(q * Decimal(str(step)))




    # =============================
    # Exchange error shims (BinanceAPIException, ExecutionError)
    # =============================
    try:
        from core.stubs import BinanceAPIException, ExecutionError
    # type: ignore
    except Exception:
        class BinanceAPIException(Exception):
            def __init__(self, code: int | None = None, message: str = ""):
                self.code = code
                super().__init__(message or f"BinanceAPIException({code})")

        class ExecutionError(Exception):
            def __init__(self, error_type: str, message: str = "", symbol: str = "", meta: dict | None = None):
                self.error_type = error_type
                self.symbol = symbol
                self.meta = meta or {}
                super().__init__(message or error_type)

    class ExecutionBlocked(Exception):
        def __init__(self, code: str, planned_quote: float, available_quote: float, min_required: float):
            self.code = code
            self.planned_quote = float(planned_quote or 0.0)
            self.available_quote = float(available_quote or 0.0)
            self.min_required = float(min_required or 0.0)
            super().__init__(f"{code}: planned={self.planned_quote:.2f} available={self.available_quote:.2f} min_required={self.min_required:.2f}")

    # =============================
    # Order filter shims (SymbolFilters, validate_order_request)
    # =============================
    from dataclasses import dataclass
    @dataclass
    class SymbolFilters:
        step_size: float
        min_qty: float
        max_qty: float
        tick_size: float
        min_notional: float
        min_entry_quote: float = 0.0

    async def validate_order_request(*, side: str, qty: float, price: float,
                                        filters: SymbolFilters, taker_fee_bps: int = 10,
                                        use_quote_amount: Optional[float] = None):
        if price <= 0:
            return False, 0.0, 0.0, "invalid_price"
        if use_quote_amount is not None:
            spend = float(use_quote_amount)
            # 1. Nominal Floor Check (Spec Requirement 1.1)
            if spend < max(filters.min_notional, filters.min_entry_quote or 0.0):
                return False, 0.0, 0.0, "QUOTE_LT_MIN_NOTIONAL"
            
            # 2. Executable Quantity Check (Spec Requirement 1.2)
            # Affordability = "The trade produces a non-zero executable quantity"
            from decimal import Decimal, ROUND_DOWN
            step = float(filters.step_size or 0.0) # BOOTSTRAP FIX: Handle 0.0 step
            price_safe = price if price > 0 else 1.0 # Guard zero div
            estimated_qty = spend / price_safe
            
            if step > 0:
                q = (Decimal(str(estimated_qty)) / Decimal(str(step))).to_integral_value(rounding=ROUND_DOWN)
                qty = float(q * Decimal(str(step)))
            else:
                qty = estimated_qty
                
            # STRICT CHECK: If rounding kills the quantity, we CANNOT execute.
            if qty <= 0:
                return False, 0.0, 0.0, "ZERO_QTY_AFTER_ROUNDING"

            if qty < float(filters.min_qty):
                return False, 0.0, 0.0, "QTY_LT_MIN"
                
            return True, float(qty), spend, "OK"
        else:
            from decimal import Decimal, ROUND_DOWN
            step = float(filters.step_size or 0.0)
            if step > 0:
                q = (Decimal(str(qty)) / Decimal(str(step))).to_integral_value(rounding=ROUND_DOWN)
                qty = float(q * Decimal(str(step)))
            
            if qty <= 0:
                return False, 0.0, 0.0, "ZERO_QTY_AFTER_ROUNDING"
                
            if qty * price < max(filters.min_notional, filters.min_entry_quote or 0.0):
                # Strict Rule 2.1: amount == 0 -> ExecutionProbe = FAIL
                return False, 0.0, 0.0, "NOTIONAL_LT_MIN"
            return True, float(qty), 0.0, "OK"

    # =============================
    # ExecutionManager
    # =============================

    class ExecutionManager:
        """
        P9 ExecutionManager — canonical single-order path, natively aligned with:
        - SharedState: get_spendable_balance(), get_position_quantity(), reserve_liquidity()/release_liquidity()
        - ExchangeClient: place_market_order(), get_exchange_filters(), get_current_price()
        """
        
        @staticmethod
        def _round_step_down(value: float, step: float) -> float:
            """Safe step rounding (floor)."""
            if step <= 0: return value
            import math
            # Avoid float precision garbage
            steps = math.floor(value / step)
            return steps * step

        # --- Post-fill realized PnL emitter (P9 observability contract) ---
        async def _handle_post_fill(self, symbol: str, side: str, order: Dict[str, Any], tier: Optional[str] = None) -> Dict[str, Any]:
            """
            Best-effort: compute/record realized PnL delta when a trade fills, then emit
            a `RealizedPnlUpdated` event and persist the delta via SharedState if possible.
            This is tolerant to different SharedState contract shapes.
            """
            emitted = False
            realized_committed = False
            delta_f = None
            try:
                sym = self._norm_symbol(symbol)
                side_u = (side or "").upper()
                exec_qty = float(order.get("executedQty") or order.get("executed_qty") or 0.0)
                price = float(order.get("avgPrice") or order.get("price") or 0.0)
                if exec_qty <= 0 or price <= 0:
                    return

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
                
                trade_recorded = False
                # P9 Frequency Engineering: Record trade for tier tracking and open trades
                if hasattr(ss, "record_trade"):
                    try:
                        # Get fee if available
                        await ss.record_trade(sym, side_u, exec_qty, price, fee_quote=fee_quote, fee_base=fee_base, tier=tier)
                        trade_recorded = True
                        
                        # Frequency Engineering: Trigger TP/SL setup for BUYs
                        if side_u == "BUY" and hasattr(self, "tp_sl_engine") and self.tp_sl_engine:
                            if hasattr(self.tp_sl_engine, "set_initial_tp_sl"):
                                try:
                                    self.tp_sl_engine.set_initial_tp_sl(sym, price, exec_qty, tier=tier)
                                except Exception as e:
                                    self.logger.warning(f"Failed to set initial TP/SL: {e}")
                    except Exception as e:
                        self.logger.warning(f"Failed to record trade in SharedState: {e}")

                delta = None

                # Try canonical API first
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

                # If we have a delta or no commit happened, persist and emit
                if delta is not None or (side_u == "SELL" and not realized_committed):
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

                    if delta_f is not None and delta_f != 0.0:
                        now = time.time()
                        try:
                            ss.metrics["realized_pnl"] = float(getattr(ss, "metrics", {}).get("realized_pnl", 0.0) or 0.0) + delta_f
                        except Exception:
                            pass
                        # Persist via public API when available
                        if hasattr(ss, "append_realized_pnl_delta"):
                            with contextlib.suppress(Exception):
                                await maybe_call(ss, "append_realized_pnl_delta", now, delta_f)
                        else:
                            # Fallback to internal store (bounded deque)
                            try:
                                ss._realized_pnl.append((now, delta_f))
                            except Exception:
                                ss._realized_pnl = deque(maxlen=4096)
                                ss._realized_pnl.append((now, delta_f))

                        # Emit the event with optional nav_quote
                        nav_q = None
                        try:
                            if hasattr(ss, "get_nav_quote"):
                                nav_q = float(await maybe_call(ss, "get_nav_quote", sym))
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
            return {
                "delta": delta_f,
                "realized_committed": realized_committed,
                "emitted": emitted,
            }

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
            if exec_qty <= 0 or exec_px <= 0:
                return

            committed = bool(post_fill or {}).get("realized_committed", False)
            emitted = bool(post_fill or {}).get("emitted", False)

            if not committed:
                try:
                    cur = float(getattr(self.shared_state, "metrics", {}).get("realized_pnl", 0.0) or 0.0)
                    self.shared_state.metrics["realized_pnl"] = cur + float(realized_pnl)
                except Exception:
                    pass

            if not emitted:
                now = time.time()
                payload = {
                    "realized_pnl": float(getattr(self.shared_state, "metrics", {}).get("realized_pnl", 0.0) or 0.0),
                    "pnl_delta": float(realized_pnl),
                    "symbol": sym,
                    "price": exec_px,
                    "qty": exec_qty,
                    "timestamp": now,
                }
                with contextlib.suppress(Exception):
                    await maybe_call(self.shared_state, "emit_event", "RealizedPnlUpdated", payload)

            self.logger.info(json.dumps({
                "event": "POSITION_CLOSED",
                "symbol": sym,
                "entry_price": entry_price,
                "exit_price": exec_px,
                "qty": exec_qty,
                "realized_pnl": realized_pnl,
            }, separators=(",", ":")))

        # Consider consolidating _split_symbol_quote and _split_base_quote to avoid drift.
        def _split_base_quote(self, symbol: str) -> Tuple[str, str]:
            s = (symbol or "").upper()
            for q in ("USDT", "FDUSD", "USDC", "BUSD", "TUSD", "BTC", "ETH"):
                if s.endswith(q):
                    return s[:-len(q)], q
            # fallback: treat configured base_ccy as quote
            if s.endswith(self.base_ccy):
                return s[:-len(self.base_ccy)], self.base_ccy
            # last resort: naive 3–4 letter quote split
            return s[:-4], s[-4:]

        def __init__(self, config: Any, shared_state: Any, exchange_client: Any, alert_callback=None):
            self.config = config
            self.shared_state = shared_state
            self.exchange_client = exchange_client
            self.alert_callback = alert_callback
            self.logger = logging.getLogger(self.__class__.__name__)

            # Contract check: must expose place_market_order()
            if not hasattr(self.exchange_client, "place_market_order") or not callable(getattr(self.exchange_client, "place_market_order", None)):
                raise RuntimeError("ExchangeClient must expose place_market_order() for canonical path")

            # Dependencies (injected later)
            self.meta_controller = None
            self.risk_manager = None

            # Config
            self.base_ccy = str(getattr(config, "BASE_CURRENCY", "USDT")).upper()
            self.safety_headroom = float(getattr(config, "QUOTE_HEADROOM", 1.02))
            self.trade_fee_pct = float(getattr(config, "TRADE_FEE_PCT", 0.001))
            self.max_spend_per_trade = float(getattr(config, "MAX_SPEND_PER_TRADE_USDT", 0))
            self.min_conf = float(getattr(config, "MIN_CONFIDENCE", 0.6))
            self.min_entry_quote_usdt = float(getattr(config, "MIN_ENTRY_QUOTE_USDT", 0.0))
            self.order_monitor_interval = float(getattr(config, "ORDER_MONITOR_INTERVAL", 15))
            self.stale_order_timeout_s = int(getattr(config, "STALE_ORDER_TIMEOUT_SECONDS", 120))
            self.max_concurrent_orders = int(getattr(config, "MAX_CONCURRENT_ORDERS", 5))
            self.enable_exec_cot = bool(getattr(config, "ENABLE_COT_VALIDATION_AT_EXECUTION", False))
            # SELL economic gate (fee-aware)
            self.allow_sell_below_fee = bool(getattr(config, "ALLOW_SELL_BELOW_FEE", False))
            self.sell_min_net_pnl_usdt = float(getattr(config, "SELL_MIN_NET_PNL_USDT", 0.0))
            self.min_net_profit_after_fees_pct = float(getattr(config, "MIN_NET_PROFIT_AFTER_FEES", 0.0035))
            # --- Hard-guard & sizing configuration (P9) ---
            def _cfg(path: str, default):
                cur = self.config
                for part in path.split('.'):
                    if isinstance(cur, dict):
                        cur = cur.get(part, default)
                    else:
                        cur = getattr(cur, part, default)
                    default = cur if cur is not None else default
                return cur if cur is not None else default

            # PHASE 2 NOTE: min_notional_floor removed (capital floor check moved to MetaController)
            # ExecutionManager no longer enforces capital policy
            self.maker_grace_s = float(_cfg('execution.maker_grace_s', 0.0))
            self.allow_taker_if_within_bps = float(_cfg('execution.allow_taker_if_within_bps', 0.0))
            self.min_free_reserve_usdt = float(_cfg('execution.min_free_reserve_usdt', 0.0))
            self.no_remainder_below_quote = float(_cfg('execution.no_remainder_below_quote', 0.0))
            # When NAV is tiny, serialize placement globally
            self.small_nav_threshold = float(_cfg('capital.small_nav_threshold_usdt', 50.0))

            # Liquidity healing
            self.max_liquidity_retries = int(getattr(config, "MAX_LIQUIDITY_RETRIES", 1))
            self.liquidity_retry_delay = float(getattr(config, "LIQUIDITY_RETRY_DELAY_SECONDS", 3.0))

            # Execution-block cooldowns (finite no-trade states)
            self.exec_block_max_retries = int(getattr(config, "EXEC_BLOCK_MAX_RETRIES", 3))
            self.exec_block_cooldown_sec = int(getattr(config, "EXEC_BLOCK_COOLDOWN_SEC", 600))
            self._buy_block_state: Dict[str, Dict[str, float]] = {}

            # Concurrency (defer semaphore creation to first use, need running loop)
            self._concurrent_orders_sem = None
            self._cancel_sem = None
            self._semaphores_initialized = False

            # Idempotency + active order guards (symbol, side)
            self._active_symbol_side_orders = set()
            self._seen_client_order_ids: Dict[str, float] = {}
            self._decision_id_seq = 0

            # Heartbeat task (will be created on first async operation)
            self._heartbeat_task = None

            self.logger.info("ExecutionManager initialized with P9 configuration")

            # --- Health: mark as Initialized right away (so Watchdog stops "no-report") ---
            try:
                # primary API
                upd = getattr(self.shared_state, "update_component_status", None) \
                    or getattr(self.shared_state, "set_component_status", None)
                if callable(upd):
                    res = upd("ExecutionManager", "Initialized", "Ready")
                    if asyncio.iscoroutine(res):
                        asyncio.create_task(res)
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

        def _exit_fee_bps(self) -> float:
            cfg_val = float(getattr(self.config, "EXIT_FEE_BPS", getattr(self.config, "CR_FEE_BPS", 0.0)) or 0.0)
            fee_from_pct = float(self.trade_fee_pct or 0.0) * 10000.0
            return max(cfg_val, fee_from_pct)

        def _exit_slippage_bps(self) -> float:
            return float(getattr(self.config, "EXIT_SLIPPAGE_BPS", getattr(self.config, "CR_PRICE_SLIPPAGE_BPS", 15.0)) or 0.0)

        async def _get_exit_floor_info(self, symbol: str, price: Optional[float] = None) -> Dict[str, float]:
            if hasattr(self.shared_state, "compute_symbol_exit_floor"):
                return await self.shared_state.compute_symbol_exit_floor(
                    symbol,
                    price=price,
                    fee_bps=self._exit_fee_bps(),
                    slippage_bps=self._exit_slippage_bps(),
                )
            return {"min_exit_quote": 0.0, "min_notional": 0.0}

        async def _get_min_entry_quote(self, symbol: str, price: Optional[float] = None, min_notional: Optional[float] = None) -> float:
            base_quote = float(getattr(self.config, "DEFAULT_PLANNED_QUOTE", getattr(self.config, "MIN_ENTRY_QUOTE_USDT", 0.0)) or 0.0)
            exit_info = await self._get_exit_floor_info(symbol, price=price)

            min_position_usdt = float(getattr(self.config, "MIN_POSITION_USDT", 0.0) or 0.0)
            min_notional_mult = float(getattr(self.config, "MIN_POSITION_MIN_NOTIONAL_MULT", 2.0) or 2.0)
            min_notional_val = float(min_notional or 0.0)
            if min_notional_val <= 0:
                try:
                    filters = await self.exchange_client.ensure_symbol_filters_ready(symbol)
                    min_notional_val = float(self._extract_min_notional(filters) or 0.0)
                except Exception:
                    min_notional_val = 0.0
            min_position_floor = min_notional_val * min_notional_mult if min_notional_val > 0 else 0.0
            return max(float(exit_info.get("min_exit_quote", 0.0)), float(base_quote), min_position_usdt, min_position_floor)

        async def _heartbeat_loop(self):
            """Continuous heartbeat to satisfy Watchdog when no trades are occurring."""
            while True:
                try:
                    await self._emit_status("Operational", "Idle / Ready")
                except Exception:
                    pass
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
            self._ensure_heartbeat()
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
                    cfg_max_conc = int(getattr(self.config, 'execution.max_concurrency', self.max_concurrent_orders))
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
                
                # ✅ FIX #2: CHECK EXCHANGE FIRST (authoritative source, < 50ms)
                get_bal = getattr(self.exchange_client, "get_account_balance", None)
                if callable(get_bal):
                    with contextlib.suppress(Exception):
                        bal = await get_bal(base_asset)
                        free = float((bal or {}).get("free", 0.0))
                        locked = float((bal or {}).get("locked", 0.0))
                        qty = float(free + locked)
                        if qty > 0:
                            self.logger.debug(f"[GetSellable] {sym}: qty={qty:.6f} from Exchange (AUTHORITATIVE)")
                            return qty
                
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
                
                # If we found it, return early
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
            trade_fee_pct = float(getattr(self.config, "TRADE_FEE_PCT", 0.0) or 0.0)
            exit_fee_bps = float(getattr(self.config, "EXIT_FEE_BPS", 0.0) or 0.0)
            fee_bps = max(exit_fee_bps, trade_fee_pct * 10000.0)
            r_fee = fee_bps / 10000.0
            r_slip = float(getattr(self.config, "EXIT_SLIPPAGE_BPS", 0.0) or 0.0) / 10000.0
            r_buf = float(getattr(self.config, "TP_MIN_BUFFER_BPS", 0.0) or 0.0) / 10000.0
            m_entry = float(getattr(self.config, "MIN_PLANNED_QUOTE_FEE_MULT", 2.5) or 2.5)
            m_exit = float(getattr(self.config, "MIN_PROFIT_EXIT_FEE_MULT", 2.0) or 2.0)
            m_exit = max(m_exit, m_entry)

            r_req = (2.0 * r_fee * m_exit) + r_slip + r_buf
            r_min_net = float(getattr(self.config, "MIN_NET_PROFIT_AFTER_FEES", 0.0) or 0.0)
            min_tp_needed_for_net = r_min_net + (2.0 * r_fee) + r_slip
            required_tp = max(r_req, min_tp_needed_for_net)
            tp_pct_min = float(getattr(self.config, "TP_PCT_MIN", 0.0) or 0.0)
            tp_max_cfg = float(getattr(self.config, "TP_PCT_MAX", 0.0) or 0.0)
            tp_atr_mult = float(getattr(self.config, "TP_ATR_MULT", 0.0) or 0.0)
            if atr_pct is None or atr_pct <= 0:
                atr_pct = float(getattr(self.config, "TPSL_FALLBACK_ATR_PCT", 0.0) or 0.0)

            tp_from_atr = (atr_pct * tp_atr_mult) if (atr_pct > 0 and tp_atr_mult > 0) else 0.0
            tp_max = tp_max_cfg if tp_max_cfg > 0 else max(tp_from_atr, tp_pct_min)

            detail = {
                "required_exit": r_req,
                "min_net_required": min_tp_needed_for_net,
                "required_tp": required_tp,
                "tp_max": tp_max,
                "fee_bps": fee_bps,
                "slippage_bps": float(getattr(self.config, "EXIT_SLIPPAGE_BPS", 0.0) or 0.0),
                "buffer_bps": float(getattr(self.config, "TP_MIN_BUFFER_BPS", 0.0) or 0.0),
                "exit_fee_mult": m_exit,
                "price": float(price or 0.0),
                "atr_pct": float(atr_pct or 0.0),
                "tp_from_atr": float(tp_from_atr),
                "tp_min": tp_pct_min,
            }
            if tp_max <= 0 or tp_max < required_tp:
                return False, detail
            return True, detail

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
            bootstrap_override = bootstrap_override and bool(getattr(self.config, "BOOTSTRAP_ALLOW_SELL_BELOW_FEE", True))
            bootstrap_min_net = float(
                policy_ctx.get("bootstrap_sell_min_net", getattr(self.config, "BOOTSTRAP_MAX_NEGATIVE_PNL", 0.0)) or 0.0
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
            slippage_bps = float(self._exit_slippage_bps() or 0.0)
            fee_pct_total = (fee_bps / 10000.0) * 2.0
            slippage_pct = slippage_bps / 10000.0
            buffer_pct = float(getattr(self.config, "TP_MIN_BUFFER_BPS", 0.0) or 0.0) / 10000.0
            entry_fee_mult = float(getattr(self.config, "MIN_PLANNED_QUOTE_FEE_MULT", 2.5) or 2.5)
            exit_fee_mult = float(getattr(self.config, "MIN_PROFIT_EXIT_FEE_MULT", 2.0) or 2.0)
            exit_fee_mult = max(exit_fee_mult, entry_fee_mult)
            net_after_fees_pct = expected_move_pct - fee_pct_total - slippage_pct
            min_net_pct = float(self.min_net_profit_after_fees_pct or 0.0)
            required_move_pct = (fee_pct_total * exit_fee_mult) + slippage_pct + buffer_pct
            if expected_move_pct < required_move_pct:
                self.logger.info(
                    "[EM:SellNetPctGate] Blocked SELL %s: move=%.4f%% < required=%.4f%% (fee_mult=%.2f fees=%.4f%% slip=%.4f%% buffer=%.4f%%)",
                    sym,
                    expected_move_pct * 100.0,
                    required_move_pct * 100.0,
                    exit_fee_mult,
                    fee_pct_total * 100.0,
                    slippage_pct * 100.0,
                    buffer_pct * 100.0,
                )
                try:
                    await self.shared_state.record_rejection(sym, "SELL", "SELL_BELOW_FEES", source="ExecutionManager")
                except Exception:
                    pass
                return {
                    "ok": False,
                    "status": "blocked",
                    "reason": "sell_below_fees",
                    "error_code": "SELL_BELOW_FEES",
                    "expected_move_pct": expected_move_pct,
                    "required_move_pct": required_move_pct,
                    "fee_mult": exit_fee_mult,
                    "fee_bps": fee_bps,
                    "slippage_bps": slippage_bps,
                    "buffer_bps": float(getattr(self.config, "TP_MIN_BUFFER_BPS", 0.0) or 0.0),
                }
            if min_net_pct > 0 and net_after_fees_pct < min_net_pct:
                self.logger.info(
                    "[EM:SellNetPctGate] Blocked SELL %s: net_after_fees=%.4f%% < min=%.4f%% (move=%.4f%% fees=%.4f%% slip=%.4f%%)",
                    sym,
                    net_after_fees_pct * 100.0,
                    min_net_pct * 100.0,
                    expected_move_pct * 100.0,
                    fee_pct_total * 100.0,
                    slippage_pct * 100.0,
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
                min_abs = float(getattr(self.config, "MIN_PORTFOLIO_IMPROVEMENT_USD", 0.05) or 0.0)
                min_pct = float(getattr(self.config, "MIN_PORTFOLIO_IMPROVEMENT_PCT", 0.0015) or 0.0)
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
            🔒 DUST RETIREMENT RULE: Check if position should be retired to PERMANENT_DUST.
            
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
                if hasattr(self.shared_state, "clear_rejections"):
                    await self.shared_state.clear_rejections(sym, side_upper)
                
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

        def _norm_symbol(self, s: str) -> str:
            return (s or "").replace("/", "").upper()

        def _split_symbol_quote(self, symbol: str) -> str:
            s = (symbol or "").upper()
            for q in ("USDT", "FDUSD", "USDC", "BUSD", "TUSD", "BTC", "ETH"):
                if s.endswith(q):
                    return q
            return self.base_ccy

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
                state["blocked_until"] = time.time() + float(self.exec_block_cooldown_sec)
                self.logger.warning(
                    "[ExecutionManager] BUY cooldown engaged: symbol=%s attempts=%d cooldown=%ds",
                    symbol, state["count"], self.exec_block_cooldown_sec
                )

        async def _log_execution_event(self, event_type: str, symbol: str, details: Dict[str, Any]):
            event = {"ts": time.time(), "component": "ExecutionManager", "event": event_type, "symbol": symbol, **details}
            self.logger.info(f"[{event_type}] {symbol}: {details}")
            # Use SharedState.emit_event (exists) — not append_event
            await maybe_call(self.shared_state, "emit_event", "ExecEvent", event)

        async def _emit_trade_executed_event(
            self,
            symbol: str,
            side: str,
            tag: str,
            order: Optional[Dict[str, Any]] = None,
        ) -> None:
            tag_lower = str(tag or "").lower()
            if side.lower() != "sell" or "tp_sl" not in tag_lower:
                return
            payload = {
                "ts": time.time(),
                "symbol": symbol,
                "side": side.upper(),
                "tag": tag,
                "source": "ExecutionManager",
            }
            if isinstance(order, dict):
                payload.update({
                    "executed_qty": float(order.get("executedQty", 0.0) or 0.0),
                    "avg_price": float(order.get("avgPrice", order.get("price", 0.0)) or 0.0),
                    "order_id": order.get("orderId") or order.get("order_id") or order.get("exchange_order_id"),
                    "status": str(order.get("status", "")).lower(),
                })
            try:
                await maybe_call(self.shared_state, "emit_event", "TRADE_EXECUTED", payload)
            except Exception:
                pass

        async def _on_order_failed(self, symbol: str, side: str, reason: str, quote: Optional[float] = None):
            """
            GAP #2 FIX: Called when an order fails. Triggers pruning if capital is tight.
            This enables stale reservation cleanup to recover blocked capital.
            """
            try:
                # If order failed due to insufficient capital, trigger immediate prune
                if quote and reason in ("InsufficientBalance", "InsufficientLiquidity", "INSUFFICIENT_BALANCE"):
                    try:
                        spendable = await maybe_call(self.shared_state, "get_free_usdt", return_default=0.0)
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

        def _build_client_order_id(self, symbol: str, side: str, decision_id: str) -> str:
            return f"{symbol}:{side}:{decision_id}"

        def _is_duplicate_client_order_id(self, client_id: str) -> bool:
            now = time.time()
            seen = self._seen_client_order_ids
            if client_id in seen:
                return True
            seen[client_id] = now
            if len(seen) > 5000:
                cutoff = now - 86400
                for key, ts in list(seen.items()):
                    if ts < cutoff:
                        seen.pop(key, None)
            return False


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
            if self.min_free_reserve_usdt > 0 and remainder < self.min_free_reserve_usdt:
                return free, False, 'RESERVE_FLOOR'
            if self.no_remainder_below_quote > 0 and 0 < remainder < self.no_remainder_below_quote:
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
            try:
                if qa is None or float(qa) <= 0:
                    # Zero-sized trades MUST be treated as failure (Behavior Change 1.1)
                    return (False, Decimal("0"), "ZERO_SIZE_TRADE")

                min_econ_trade_cfg = Decimal(str(getattr(self.config, "MIN_ECONOMIC_TRADE_USDT", 0.0) or 0.0))
                dynamic_min_econ = Decimal("0")
                if hasattr(self.shared_state, "compute_min_entry_quote"):
                    try:
                        dyn_floor = await self.shared_state.compute_min_entry_quote(
                            symbol,
                            default_quote=float(qa),
                        )
                        dynamic_min_econ = Decimal(str(dyn_floor or 0.0))
                    except Exception:
                        dynamic_min_econ = Decimal("0")
                min_econ_trade = max(min_econ_trade_cfg, dynamic_min_econ)
                if min_econ_trade > 0 and qa < min_econ_trade:
                    gap = (min_econ_trade - qa).max(Decimal("0"))
                    return (False, gap, "QUOTE_LT_MIN_ECONOMIC")

                sym = self._norm_symbol(symbol)
                eps = Decimal("1e-9")

                # FETCH PRICE (Fix for NameError) -- Use exchange_client
                price = await self.exchange_client.get_ticker_price(sym)
                if price <= 0:
                    # If we can't get price, we can't estimate quantity effectively
                    # Fallback logic or hard fail? Let's assume 1.0 but log warning, or fail.
                    # Failing is safer for affordability checks.
                    return (False, Decimal("0"), "PRICE_UNAVAILABLE")

                atr_pct = 0.0
                try:
                    if hasattr(self.shared_state, "calc_atr"):
                        atr = float(await self.shared_state.calc_atr(sym, "5m", 14) or 0.0)
                        if atr <= 0:
                            atr = float(await self.shared_state.calc_atr(sym, "1m", 14) or 0.0)
                        if atr and price > 0:
                            atr_pct = float(atr) / float(price)
                except Exception:
                    atr_pct = 0.0

                feasible, feas_detail = self._entry_profitability_feasible(sym, price=price, atr_pct=atr_pct)
                if not feasible:
                    payload = {
                        "reason": "INFEASIBLE_PROFITABILITY",
                        "symbol": sym,
                        **feas_detail,
                    }
                    self.logger.warning("[EM:ProfitFeasibility] Block BUY: %s", payload)
                    with contextlib.suppress(Exception):
                        await maybe_call(self.shared_state, "emit_event", "EntryProfitabilityBlocked", payload)
                    return (False, Decimal("0"), "INFEASIBLE_PROFITABILITY")

                # Micro-trade kill switch: low equity + low volatility
                if getattr(self.config, "MICRO_TRADE_KILL_SWITCH_ENABLED", False):
                    nav = await self.shared_state.get_nav()
                    nav_max = float(getattr(self.config, "MICRO_TRADE_KILL_EQUITY_MAX", 0.0) or 0.0)
                    if nav_max > 0 and nav > 0 and nav < nav_max:
                        atr_pct = 0.0
                        try:
                            if hasattr(self.shared_state, "calc_atr"):
                                atr = float(await self.shared_state.calc_atr(sym, "5m", 14) or 0.0)
                                if atr <= 0:
                                    atr = float(await self.shared_state.calc_atr(sym, "1m", 14) or 0.0)
                                if atr and price > 0:
                                    atr_pct = float(atr) / float(price)
                        except Exception:
                            atr_pct = 0.0
                        if atr_pct <= 0:
                            atr_pct = float(getattr(self.config, "MICRO_TRADE_KILL_FALLBACK_ATR_PCT", 0.0) or 0.0)
                        round_trip_fee_rate = float(self.trade_fee_pct) * 2.0
                        fee_mult = float(getattr(self.config, "MICRO_TRADE_KILL_ATR_FEE_MULT", 1.0) or 1.0)
                        fee_threshold = round_trip_fee_rate * fee_mult
                        if fee_threshold > 0 and atr_pct < fee_threshold:
                            return (False, Decimal("0"), "MICRO_TRADE_KILL_SWITCH")

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
                    min_notional = self._extract_min_notional(filters)

                if min_notional <= 0:
                    return (False, Decimal("0"), f"invalid_min_notional({min_notional})")

                # Dynamic exit-feasibility floor (symbol-aware)
                min_required_val = await self._get_min_entry_quote(sym, price=price, min_notional=float(min_notional))
                min_required = Decimal(str(min_required_val))

                # Planned-quote fee floor (planned_quote >= fee_mult × round-trip fee on min-required quote)
                fee_mult = Decimal(str(getattr(self.config, "MIN_PLANNED_QUOTE_FEE_MULT", 2.5) or 2.5))
                round_trip_fee_rate = Decimal(str(self.trade_fee_pct)) * Decimal("2")
                if round_trip_fee_rate > 0 and fee_mult > 0:
                    planned_fee_floor = min_required * round_trip_fee_rate * fee_mult
                    if qa < planned_fee_floor - eps:
                        gap = (planned_fee_floor - qa).max(Decimal("0"))
                        return (False, gap, "QUOTE_LT_FEE_FLOOR")

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
                self._current_policy_context = policy_context
                
                # 3) ACCUMULATE_MODE/BOOTSTRAP_BYPASS CHECK: Skip min_notional validation for special modes
                # This allows P0 dust promotion and bootstrap execution to work without being blocked by min_notional guards
                accumulate_mode = False
                bootstrap_bypass = False
                no_downscale_planned_quote = False
                if policy_context:
                    accumulate_mode = bool(policy_context.get("_accumulate_mode", False))
                    bootstrap_bypass = bool(policy_context.get("bootstrap_bypass", False))
                    no_downscale_planned_quote = bool(
                        policy_context.get("_no_downscale_planned_quote", False)
                        or policy_context.get("no_downscale_planned_quote", False)
                    )
                
                
                # Skip min_notional check only for bootstrap_bypass mode
                # CRITICAL: bootstrap_bypass allows first trade on flat portfolio to execute
                bypass_min_notional = bootstrap_bypass
                
                if not bypass_min_notional:
                    # 3) If planned quote + accumulation is below the floor, decide between MIN_NOTIONAL vs NAV shortfall
                    if effective_qa < min_required - eps:
                        # If the user is effectively trying to spend all they have (including accumulation)
                        # and that amount is still below the venue/config floor, classify as NAV shortfall.
                        if spendable_dec > 0 and (qa <= spendable_dec + eps) and (spendable_dec + acc_val < min_required - eps):
                            gap = (min_required - (spendable_dec + acc_val)).max(Decimal("0"))
                            return (False, gap, "INSUFFICIENT_QUOTE")
                        # Otherwise, the caller asked below the floor while having enough NAV.
                        gap = (min_required - effective_qa).max(Decimal("0"))
                        return (False, gap, "QUOTE_LT_MIN_NOTIONAL")
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
                est_units_raw = float(effective_qa) / price_f
                
                # P9 Corrective: Use the extracted step_size (robust) instead of direct dict access
                step = step_size
                if step > 0:
                    est_units = self._round_step_down(est_units_raw, step)
                else:
                    est_units = est_units_raw
                    
                if est_units <= 0:
                    return (False, Decimal("0"), "ZERO_QTY_AFTER_ROUNDING")
                
                est_notional = est_units * price_f
                
                # Check if this executable chunk meets minNotional (SKIP in bootstrap/accumulate modes)
                if not bypass_min_notional and est_notional < float(min_required):
                    gap = Decimal(str(min_required)) - Decimal(str(est_notional))
                    return (False, gap.max(Decimal("0")), "QUOTE_LT_MIN_NOTIONAL")
                elif bypass_min_notional and est_notional < float(min_required):
                    # Log bypass for accumulate execution
                    self.logger.warning(
                        f"[EM:ACCUMULATE] {sym} BUY: Bypassing second min_entry check - "
                        f"est_notional={est_notional:.2f} < min_required={float(min_required):.2f}"
                    )

                gross_needed = qa * (Decimal("1") + taker_fee) * headroom
                # CRITICAL FIX: Use EPSILON tolerance to avoid float/timing false negatives
                # Problem: spendable_dec < gross_needed can fail due to tiny float differences
                # Solution: Allow small tolerance (EPSILON = 1e-6) for capital availability
                EPSILON = Decimal("1e-6")
                if spendable_dec < gross_needed - EPSILON:
                    # Point 3: Dynamic Resizing (Downscaling)
                    # If we have less than planned, but enough for minNotional, we downscale.
                    max_qa = spendable_dec / ((Decimal("1") + taker_fee) * headroom)
                    if no_downscale_planned_quote:
                        gap = (qa - max_qa).max(Decimal("0"))
                        return (False, gap, "INSUFFICIENT_QUOTE")
                    if max_qa >= Decimal(str(min_required)) or bypass_min_notional:
                        self.logger.info(f"[EM] Dynamic Resizing: Downscaling {qa} -> {max_qa:.2f} to fit spendable {spendable_dec:.2f}")
                        return (True, max_qa, "OK_DOWNSCALED")
                    else:
                        # Point 2: Accumulation Pivot
                        # No enough for minNotional even with all cash.
                        gap = Decimal(str(min_required)) - max_qa
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
                        min_entry_quote=float(min_required)
                    )
                    # Re-fetch real filters for accuracy
                    f_data = await self.exchange_client.ensure_symbol_filters_ready(sym)
                    step_size, min_qty, _, _, _ = self._extract_filter_vals(f_data)
                    filters_obj.step_size = step_size
                    filters_obj.min_qty = min_qty

                    v_ok, v_qty, _, v_reason = await validate_order_request(
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

        # =============================
        # Canonical execution API
        # =============================
        async def close_position(
            self,
            *,
            symbol: str,
            reason: str = "",
            force_finalize: bool = False,
            tag: str = "tp_sl",
        ) -> Dict[str, Any]:
            """Close a position via the canonical execution path with optional forced finalization."""
            reason_text = str(reason or "").strip() or "EXIT"
            policy_context = {
                "exit_reason": reason_text,
                "reason": reason_text,
                "liquidation_reason": reason_text,
            }
            res = await self.execute_trade(
                symbol=symbol,
                side="sell",
                quantity=None,
                planned_quote=None,
                tag=tag,
                is_liquidation=True,
                policy_context=policy_context,
            )
            if force_finalize:
                try:
                    status = str(res.get("status", "")).lower() if isinstance(res, dict) else ""
                    ok = bool(res.get("ok")) if isinstance(res, dict) else False
                    if ok or status in {"placed", "executed", "filled", "partially_filled"}:
                        await self._force_finalize_position(symbol, reason_text)
                except Exception:
                    self.logger.debug("[EM] force_finalize_position failed for %s", symbol, exc_info=True)
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

        async def execute_trade(
            self,
            *,
            symbol: str,
            side: str,
            quantity: Optional[float] = None,
            planned_quote: Optional[float] = None,
            tag: str = "meta/Agent",
            tier: Optional[str] = None,
            is_liquidation: bool = False,
            policy_context: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            """
            Tier-aware execution (Phase A Frequency Engineering).

            ExecutionManager answers: "Given this allowed intent, what is the maximum safe executable order?"

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
            self._ensure_heartbeat()
            side = (side or "").lower()
            sym = self._norm_symbol(symbol)
            policy_ctx = dict(policy_context or {})
            decision_id = self._resolve_decision_id(policy_ctx)
            policy_ctx.setdefault("decision_id", decision_id)

            # 🛡️ P9 SOP GOVERNANCE: Last line of defense (unless full liquidation bypass)
            # Exits/Liquidation are usually allowed even in safety modes to protect capital.
            is_liq_full = is_liquidation or any(x in (tag or "").lower() for x in ("tp_sl", "balancer"))

            # Global SELL guard (real capital only): allow only TP/SL (liquidation) or explicit EMERGENCY exits.
            # This prevents state/rotation/recovery SELLs from fragmenting compounding on live capital.
            is_real_mode = bool(getattr(self.config, "LIVE_MODE", False)) and not bool(getattr(self.config, "SIMULATION_MODE", False)) and not bool(getattr(self.config, "PAPER_MODE", False)) and not bool(getattr(self.config, "TESTNET_MODE", False))
            if side == "sell" and is_real_mode and not is_liq_full:
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
            
            if not is_liq_full:
                mode = str(self.shared_state.metrics.get("current_mode", "NORMAL")).upper()
                if mode == "PAUSED":
                    self.logger.warning(f"[EM:GovBlock] Blocked {side.upper()} {sym}: System is PAUSED.")
                    return {"ok": False, "status": "blocked", "reason": "System PAUSED", "error_code": "PAUSED_MODE"}
                
                if side == "buy" and mode == "PROTECTIVE":
                    self.logger.warning(f"[EM:GovBlock] Blocked BUY {sym}: System is in PROTECTIVE mode.")
                    return {"ok": False, "status": "blocked", "reason": "BUY Disabled in PROTECTIVE Mode", "error_code": "PROTECTIVE_MODE"}
            tag_raw = tag or ""
            tag_lower = tag_raw.lower()
            clean_tag = self._sanitize_tag(tag)
            tier_label = f" tier={tier}" if tier else ""

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
            
            # [FIX #9] Detect liquidation: explicit flag OR tag contains tp_sl/balancer
            # NOTE: tag 'liquidation' alone no longer implies full bypass.
            is_liq_full = is_liquidation or any(x in tag_lower for x in ("tp_sl", "balancer"))
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
                    return net_gate
            policy_authority = str(policy_ctx.get("authority") or policy_ctx.get("policy_authority") or "").lower()
            policy_validated = policy_authority == "metacontroller"
            if policy_validated and side == "buy" and planned_quote and planned_quote > 0:
                # MetaController planned_quote is authoritative: do not downscale below it.
                policy_ctx.setdefault("_no_downscale_planned_quote", True)
                policy_ctx.setdefault("_planned_quote_floor", float(planned_quote))
            
            # [FIX #4] UNIFIED SELL AUTHORITY: MetaController has supreme authority over SELL decisions
            # If UNIFIED_SELL_AUTHORITY=True, SELL overrides most operational gates
            # (except actual position existence and exchange rejection)
            unified_sell_authority = bool(policy_ctx.get("UNIFIED_SELL_AUTHORITY", False))
            if unified_sell_authority and side == "sell":
                self.logger.info(f"[EM:UnifiedSell] {sym} SELL has UNIFIED_SELL_AUTHORITY: bypassing operational veto points")
            
            # 🚫 TERMINAL_DUST BLOCK: If position < minNotional, block all liquidation attempts
            # MODIFIED [FIX #4]: Allow terminal dust block to be overridden by unified sell authority
            # This prevents endless liquidation loops UNLESS MetaController explicitly overrides
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
            
            # [FIX #9] LIQUIDATION BYPASS: If this is liquidation, skip ALL guards and go straight to execution
            if is_liq_full and side == "sell":
                self.logger.info(f"[EXEC:LIQ] LIQUIDATION SELL: {sym} - bypassing all guards (capital, min-notional, throughput)")
                
                # For liquidation SELLs, we execute with best-effort quantity
                if not quantity or quantity <= 0:
                    qty = await self._get_sellable_qty(sym)
                    if qty <= 0:
                        self.logger.warning(f"[EXEC:LIQ] No position to liquidate for {sym}")
                        await self.shared_state.record_rejection(sym, "SELL", "NO_POSITION_QUANTITY", source="ExecutionManager")
                        return {"ok": False, "status": "skipped", "reason": "no_position_quantity", "error_code": "NoPosition"}
                    quantity = qty
                
                # Execute SELL immediately without any guards
                raw = await self._place_market_order_qty(
                    sym,
                    float(quantity),
                    "SELL",
                    clean_tag,
                    is_liquidation=True,
                    decision_id=decision_id,
                )
                
                # Normalize output
                status = str(raw.get("status", "REJECTED")).upper()
                exec_qty = float(raw.get("executedQty", 0.0))
                is_filled = status in ("FILLED", "PARTIALLY_FILLED") and exec_qty > 0
                
                if is_filled:
                    await self._emit_status("Operational", f"filled {sym} {side} status={status}")
                    try:
                        if hasattr(self.shared_state, "clear_rejections"):
                            await self.shared_state.clear_rejections(sym, side.upper())
                            self.logger.info(f"[MemoryOfFailure] ✅ Cleared rejections for {sym} {side} (liquidation success)")
                    except Exception as e:
                        self.logger.debug(f"[MemoryOfFailure] Failed to clear: {e}")
                    
                    post_fill = None
                    with contextlib.suppress(Exception):
                        post_fill = await self._handle_post_fill(sym, side, raw, tier=tier)

                    with contextlib.suppress(Exception):
                        await self._emit_close_events(sym, raw, post_fill)

                    with contextlib.suppress(Exception):
                        await self._emit_trade_executed_event(sym, side, tag_raw, raw)

                    # Close position explicitly on liquidation SELL fills
                    try:
                        pm = getattr(self.shared_state, "position_manager", None)
                        exec_qty = float(raw.get("executedQty", 0.0))
                        exec_px = float(raw.get("avgPrice", raw.get("price", 0.0)) or 0.0)
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
                        if pm and hasattr(pm, "close_position"):
                            await pm.close_position(
                                symbol=sym,
                                executed_qty=exec_qty,
                                executed_price=exec_px,
                                fee_quote=fee_quote,
                                reason=str(policy_ctx.get("exit_reason") or policy_ctx.get("reason") or "SELL_FILLED"),
                            )
                        elif pm and hasattr(pm, "finalize_position"):
                            await pm.finalize_position(
                                symbol=sym,
                                executed_qty=exec_qty,
                                executed_price=exec_px,
                                reason=str(policy_ctx.get("exit_reason") or policy_ctx.get("reason") or "SELL_FILLED"),
                            )
                        elif hasattr(self.shared_state, "close_position"):
                            await self.shared_state.close_position(
                                sym,
                                reason=str(policy_ctx.get("exit_reason") or policy_ctx.get("reason") or "SELL_FILLED"),
                            )
                    except Exception:
                        self.logger.debug("[EM] finalize_position failed for %s", sym, exc_info=True)
                    
                    result = {
                        "ok": True,
                        "status": str(raw.get("status", "FILLED")).lower(),
                        "executedQty": float(raw.get("executedQty", 0.0)),
                        "avgPrice": float(raw.get("avgPrice", raw.get("price", 0.0)) or 0.0),
                        "cummulativeQuoteQty": float(raw.get("cummulativeQuoteQty", 0.0)),
                        "orderId": raw.get("orderId") or raw.get("order_id") or raw.get("exchange_order_id"),
                        "reason": "[LIQUIDATION]",
                    }
                    self.logger.info(f"[EXEC:LIQ] ✅ Liquidation SELL executed: {sym} qty={result['executedQty']:.6f}")
                    return result
                else:
                    self.logger.warning(f"[EXEC:LIQ] ⚠️ Liquidation SELL failed: status={status}, qty={exec_qty}")
                    await self._log_execution_event("liquidation_fail", sym, {"reason": "not_filled"})
                    return {
                        "ok": False,
                        "status": status.lower(),
                        "executedQty": exec_qty,
                        "reason": "[LIQUIDATION_FAILED]",
                        "error_code": "LiquidationFailed"
                    }

            # ---- Risk gate (ALLOW / DENY / ADJUST) ---- [SKIPPED FOR LIQUIDATION]
            # [FIX #4] UNIFIED SELL AUTHORITY: RiskManager is advisory, not veto, for SELL
            strict_risk = bool(getattr(self.config, "STRICT_RISK_AT_EXEC", False))
            adjusted = None
            if self.risk_manager and hasattr(self.risk_manager, "check"):
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
                                return {"ok": False, "status": "skipped", "reason": "permanent_dust_retired", "error_code": "DustRetired"}
                            
                            await self._log_execution_event("risk_block", sym, {"side": side, "reason": risk_reason})
                            await self.shared_state.record_rejection(sym, side.upper(), "RISK_CAP_EXCEEDED", source="ExecutionManager")
                            self.logger.info(f"[EXEC_REJECT] symbol={sym} side={side.upper()} reason=RISK_CAP_EXCEEDED count={self.shared_state.get_rejection_count(sym, side.upper())} action=SKIP")
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
                        return {"ok": False, "status": "error", "reason": "RiskCheckFailed", "error_code": "RiskCheckFailed"}

            # ---- ProfitTarget guard (optional) ----
            try:
                guard_fn = getattr(self.shared_state, "profit_target_ok", None)
                if callable(guard_fn):
                    # Do not let the profit guard block SELLs or liquidity/TP-SL ops
                    if side == "buy" and not any(x in (tag or "") for x in ("liquidation", "tp_sl", "balancer")):
                        # P9: Use dynamic profit target from shared_state/config
                        min_target = float(getattr(self.config, "PROFIT_TARGET_BASE_USD_PER_HOUR", 20.0))
                        guard_ok = bool(await guard_fn(min_usdt_per_hour=min_target))
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

            try:
                # ---- Safety Check: Circuit Breaker & Health (Invariant 2) ----
                if hasattr(self.shared_state, "is_circuit_breaker_open") and await self.shared_state.is_circuit_breaker_open():
                    self.logger.warning(f"[EXEC] 🛑 Circuit Breaker OPEN. Blocking trade for {sym}.")
                    return {"ok": False, "status": "blocked", "reason": "CB_OPEN", "error_code": "CB_OPEN"}

                # Health: we’re about to execute
                await self._emit_status("Running", f"execute_trade {sym} {side}")
                # ---- Route by side ----
                async with self._small_nav_guard():
                    if side == "buy":
                        if planned_quote and planned_quote > 0:
                            # Cooldown: suppress repeated execution-blocked BUYs
                            if policy_ctx.get("_no_downscale_planned_quote"):
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
                            
                            if clean_tag.startswith("meta/") and not bootstrap_bypass:
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
                                # Mandatory explicit failure for authoritative planned quotes
                                # BOOTSTRAP FIX: Skip this block during bootstrap as we've already verified spendable balance
                                min_required_quote = await self._get_min_entry_quote(sym)
                                if policy_ctx.get("_no_downscale_planned_quote") and planned_quote >= min_required_quote and not bootstrap_bypass:
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
                                            ttl_sec=int(getattr(self.config, "ACCUMULATION_TTL", 3600)),
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
                                            if self.shared_state and hasattr(self.shared_state, "report_agent_capital_failure"):
                                                self.shared_state.report_agent_capital_failure(f"exec_fail_{sym}")
                                            await self._log_execution_event("order_skip", sym, {"side": "buy", "reason": "permanent_dust_retired"})
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
                                        if self.shared_state and hasattr(self.shared_state, "report_agent_capital_failure"):
                                            self.shared_state.report_agent_capital_failure(f"exec_fail_{sym}")
                                        await self._log_execution_event("order_skip", sym, {"side": "buy", "reason": "permanent_dust_retired"})
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

                            order = await self.exchange_client.market_buy(sym, execute_quote, tag=clean_tag)
                            filled_qty = float(order.get("executedQty", 0.0) or 0.0)
                            avg_price = 0.0
                            if filled_qty > 0:
                                avg_price = float(order.get("cummulativeQuoteQty", 0.0) or 0.0) / filled_qty
                            if avg_price > 0:
                                order.setdefault("avgPrice", avg_price)
                                order.setdefault("price", avg_price)
                            raw = order
                        else:
                            if not quantity or quantity <= 0:
                                await self._log_execution_event("order_skip", sym, {"side": "buy", "reason": "InvalidQuantity"})
                                return {"ok": False, "status": "skipped", "reason": "zero_or_negative_quantity", "error_code": "InvalidQuantity"}
                            raw = await self._place_market_order_qty(
                                sym,
                                quantity,
                                "BUY",
                                clean_tag,
                                decision_id=decision_id,
                            )

                    elif side == "sell":
                        # GAP #4 FIX: SELL should only be blocked on true cold bootstrap, not on any bootstrap state
                        # Allow SELL if:
                        # 1. System is not in cold bootstrap, OR
                        # 2. This is a liquidation/tp_sl/balancer operation, OR
                        # 3. We have a valid position
                        if not policy_validated:
                            is_cold = self.shared_state and hasattr(self.shared_state, "is_cold_bootstrap") and \
                                    self.shared_state.is_cold_bootstrap()
                            is_liquidation = any(x in (tag or "") for x in ("liquidation", "tp_sl", "balancer"))

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
                            if not quantity or quantity <= 0:
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
                            self.logger.info(f"[MemoryOfFailure] ✅ Cleared rejections for {sym} {side} (successful execution)")
                        if side == "buy":
                            self._buy_block_state.pop(sym, None)
                    except Exception as e:
                        self.logger.debug(f"[MemoryOfFailure] Failed to clear rejections: {e}")

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
                    with contextlib.suppress(Exception):
                        post_fill = await self._handle_post_fill(sym, side, raw, tier=tier)

                    if side == "sell":
                        with contextlib.suppress(Exception):
                            await self._emit_close_events(sym, raw, post_fill)

                    with contextlib.suppress(Exception):
                        await self._emit_trade_executed_event(sym, side, tag_raw, raw)

                    # Finalize position on SELL fills
                    if side == "sell":
                        try:
                            pm = getattr(self.shared_state, "position_manager", None)
                            exec_qty = float(raw.get("executedQty", 0.0))
                            exec_px = float(raw.get("avgPrice", raw.get("price", 0.0)) or 0.0)
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
                            if pm and hasattr(pm, "close_position"):
                                await pm.close_position(
                                    symbol=sym,
                                    executed_qty=exec_qty,
                                    executed_price=exec_px,
                                    fee_quote=fee_quote,
                                    reason=str(policy_ctx.get("exit_reason") or policy_ctx.get("reason") or "SELL_FILLED"),
                                )
                            elif pm and hasattr(pm, "finalize_position"):
                                await pm.finalize_position(
                                    symbol=sym,
                                    executed_qty=exec_qty,
                                    executed_price=exec_px,
                                    reason=str(policy_ctx.get("exit_reason") or policy_ctx.get("reason") or "SELL_FILLED"),
                                )
                            elif hasattr(self.shared_state, "close_position"):
                                await self.shared_state.close_position(
                                    sym,
                                    reason=str(policy_ctx.get("exit_reason") or policy_ctx.get("reason") or "SELL_FILLED"),
                                )
                            if hasattr(self.shared_state, "mark_position_closed"):
                                await self.shared_state.mark_position_closed(
                                    symbol=sym,
                                    qty=exec_qty,
                                    price=exec_px,
                                    reason=str(policy_ctx.get("exit_reason") or policy_ctx.get("reason") or "SELL_FILLED"),
                                    tag=str(tag_raw or ""),
                                )
                        except Exception:
                            self.logger.debug("[EM] finalize_position failed for %s", sym, exc_info=True)

                    result = {
                        "ok": True,
                        "status": str(raw.get("status", "FILLED")).lower(),
                        "executedQty": float(raw.get("executedQty", 0.0)),
                        "avgPrice": float(raw.get("avgPrice", raw.get("price", 0.0)) or 0.0),
                        "cummulativeQuoteQty": float(raw.get("cummulativeQuoteQty", 0.0)),
                        "orderId": raw.get("orderId") or raw.get("order_id") or raw.get("exchange_order_id"),
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
                # GAP #2 FIX: Trigger pruning on exception
                await self._on_order_failed(sym, side, error_type, planned_quote)
                # Health: hard error
                await self._emit_status("Error", f"exception {sym} {side}: {error_type}")
                return {"ok": False, "status": "error",
                        "reason": f"exception:{error_type}",
                        "error_code": error_type}

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
                    if qty <= 0:
                        continue
                    if int(ex.get("_coalesced_count") or 0) > 1:
                        self.logger.info(
                            "[ExecutionManager] Coalesced %d liquidation exits for %s into qty=%.8f (tags=%s)",
                            int(ex.get("_coalesced_count") or 0),
                            sym,
                            qty,
                            ",".join(ex.get("_coalesced_tags") or [])
                        )
                    raw = await self._place_market_order_qty(sym, qty, "SELL", tag)
                    if raw and raw.get("ok"):
                        any_success = True
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
                return {"ok": False, "reason": "no_price"}

            qty = float(quantity or 0.0)
            if qty <= 0:
                await self._log_execution_event("order_skip", symbol, {"side": side.upper(), "reason": "qty_invalid"})
                return {"ok": False, "reason": "qty_invalid"}

            if side.upper() == "BUY" and not policy_validated:
                ok_aff, why = await self.explain_afford_market_buy(symbol, Decimal(str(qty)))
                if not ok_aff:
                    if "INSUFFICIENT_QUOTE" in why or "QUOTE_LT_MIN_NOTIONAL" in why:
                        need_quote = float(qty) * float(current_price) * float(1.0 + self.trade_fee_pct) * float(self.safety_headroom)
                        heal_context = {"reason": "affordability_gate", "symbol": symbol, "needed_quote": need_quote, "current_price": current_price}
                        healed = await self._attempt_liquidity_healing(symbol, need_quote, heal_context)
                        if not healed:
                            await self._log_execution_event("order_skip", symbol, {"side": side.upper(), "reason": str(why)})
                            return {"ok": False, "reason": why}
                    else:
                        await self._log_execution_event("order_skip", symbol, {"side": side.upper(), "reason": str(why)})
                        return {"ok": False, "reason": why}

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
                if not raw_order:
                    if reservation_id:
                        with contextlib.suppress(Exception):
                            await self.shared_state.release_liquidity(quote_asset, reservation_id)
                            await self._log_execution_event("liquidity_released", symbol, {
                                "asset": quote_asset, "amount": float(planned_quote), "scope": "buy_by_qty",
                                "reason": "order_not_placed", "reservation_id": reservation_id
                            })
                    await self._log_execution_event("order_skip", symbol, {"side": side.upper(), "reason": "order_not_placed"})
                    return {"ok": False, "reason": "order_not_placed"}

                if reservation_id:
                    spent = float(raw_order.get("cummulativeQuoteQty") or planned_quote)
                    with contextlib.suppress(Exception):
                        await self.shared_state.release_liquidity(quote_asset, reservation_id)
                        await self._log_execution_event("liquidity_released", symbol, {
                            "asset": quote_asset, "amount": float(spent), "scope": "buy_by_qty",
                            "reason": "order_filled", "reservation_id": reservation_id
                        })

                return raw_order
            except Exception:
                if reservation_id:
                    with contextlib.suppress(Exception):
                        await self.shared_state.release_liquidity(quote_asset, reservation_id)
                        await self._log_execution_event("liquidity_released", symbol, {
                            "asset": quote_asset, "amount": float(planned_quote), "scope": "buy_by_qty",
                            "reason": "exception", "reservation_id": reservation_id
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
                return {"ok": False, "reason": "no_price"}

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
                            return {"ok": False, "reason": why}
                    else:
                        await self._log_execution_event("order_skip", symbol, {"side": side.upper(), "reason": str(why)})
                        return {"ok": False, "reason": why}

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
                if not raw_order:
                    if reservation_id:
                        with contextlib.suppress(Exception):
                            await self.shared_state.release_liquidity(quote_asset, reservation_id)
                            await self._log_execution_event("liquidity_released", symbol, {
                                "asset": quote_asset, "amount": float(quote), "scope": "buy_by_quote",
                                "reason": "order_not_placed", "reservation_id": reservation_id
                            })
                    await self._log_execution_event("order_skip", symbol, {"side": side, "reason": "order_not_placed"})
                    return {"ok": False, "reason": "order_not_placed"}

                if reservation_id:
                    spent = float(raw_order.get("cummulativeQuoteQty") or quote)
                    with contextlib.suppress(Exception):
                        await self.shared_state.release_liquidity(quote_asset, reservation_id)
                        await self._log_execution_event("liquidity_released", symbol, {
                            "asset": quote_asset, "amount": float(spent), "scope": "buy_by_qty",
                            "reason": "order_filled", "reservation_id": reservation_id
                        })

                return raw_order
            except Exception:
                if reservation_id:
                    with contextlib.suppress(Exception):
                        await self.shared_state.release_liquidity(quote_asset, reservation_id)
                        await self._log_execution_event("liquidity_released", symbol, {
                            "asset": quote_asset, "amount": float(quote), "scope": "buy_by_quote",
                            "reason": "exception", "reservation_id": reservation_id
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

            # --- Filters from ExchangeClient (raw) ---
            filters = await self.exchange_client.ensure_symbol_filters_ready(symbol)
            step_size, min_qty, max_qty, tick_size, min_notional = self._extract_filter_vals(filters)
            if step_size <= 0 or min_notional <= 0:
                return None

            safe_tag = self._sanitize_tag(comment)
            decision_id = decision_id or self._resolve_decision_id(getattr(self, "_current_policy_context", None))
            client_id = self._build_client_order_id(symbol, side.upper(), decision_id)

            if self._is_duplicate_client_order_id(client_id):
                self.logger.debug("[EM] Duplicate client_order_id for %s %s; skipping.", symbol, side.upper())
                return {"status": "SKIPPED", "reason": "IDEMPOTENT"}

            order_key = (symbol, side.upper())
            if order_key in self._active_symbol_side_orders:
                self.logger.debug("[EM] Active order exists for %s %s; skipping.", symbol, side.upper())
                return {"status": "SKIPPED", "reason": "ACTIVE_ORDER"}
            self._active_symbol_side_orders.add(order_key)

            try:
                # Lazy-init semaphores (need running loop)
                self._ensure_semaphores_ready()

                async with self._concurrent_orders_sem:
                    # Use quote-amount path whenever BUY + planned_quote is given

                    if side.upper() == "BUY" and planned_quote and planned_quote > 0:

                        spend = float(planned_quote)

                        # Enforce both venue min_notional (with buffer) and minimal execution floor
                        # PHASE 2 NOTE: Capital floor check already done in MetaController
                        min_entry = await self._get_min_entry_quote(symbol, price=current_price, min_notional=min_notional)
                        if spend < min_entry:
                            await self._log_execution_event("order_skip", symbol, {"side": "BUY", "reason": "NOTIONAL_LT_MIN", "min_required": min_entry})
                            return None

                        # DEADLOCK FIX: Escalation logic for gross notional requirements
                        # Check if planned quote meets GROSS requirements (net + fees + safety)
                        q_asset = self._split_symbol_quote(symbol)
                        trade_fee_pct = float(self.trade_fee_pct)
                        safety_headroom = float(self.safety_headroom)
                        gross_factor = (1.0 + trade_fee_pct) * safety_headroom
                        min_required_gross = min_entry * gross_factor

                        no_downscale = False
                        if hasattr(self, "_current_policy_context") and self._current_policy_context:
                            no_downscale = bool(self._current_policy_context.get("_no_downscale_planned_quote", False))

                        if spend < min_required_gross and not no_downscale:
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
                            step_size=step_size, min_qty=min_qty, max_qty=max_qty,
                            tick_size=tick_size, min_notional=min_notional, min_entry_quote=min_entry
                        )
                        
                        # BOOTSTRAP FIX: Check if this is a bootstrap bypass scenario
                        is_bootstrap = False
                        try:
                            # Check if we're in bootstrap mode (from policy context or shared state)
                            if hasattr(self, '_current_policy_context') and self._current_policy_context:
                                is_bootstrap = bool(self._current_policy_context.get("bootstrap_bypass", False))
                            if not is_bootstrap and hasattr(self.shared_state, "is_bootstrap_mode"):
                                is_bootstrap = self.shared_state.is_bootstrap_mode()
                        except Exception:
                            pass
                        
                        if is_bootstrap:
                            # BOOTSTRAP: Use true quote-based market order (quoteOrderQty) for guaranteed execution
                            self.logger.info(
                                "[EM:BOOTSTRAP] Using quoteOrderQty=%s for %s BUY to guarantee bootstrap execution (bypassing quantity precision issues)",
                                spend, symbol
                            )
                            
                            # Use exchange client's quote-based order directly
                            try:
                                if hasattr(self.exchange_client, '_send_real_order_quote'):
                                    order_id = await self.exchange_client._send_real_order_quote(
                                        symbol=symbol, 
                                        side="BUY", 
                                        quote=float(spend), 
                                        tag=safe_tag,
                                        client_order_id=client_id,
                                    )
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
                                        return order
                            except Exception as e:
                                self.logger.warning(f"[EM:BOOTSTRAP] Quote-based order failed: {e}, falling back to quantity-based")
                        
                        # Standard quantity-based execution path
                        ok, qty, adjusted_quote, _ = await validate_order_request(
                            side="BUY", qty=0, price=current_price, filters=filters_obj,
                            taker_fee_bps=getattr(self.config, "TAKER_FEE_BPS", 10), use_quote_amount=spend
                        )
                        if not ok:
                            return None
                        
                        # Use the quantity calculated by validate_order_request to ensure 
                        # quantity-based execution (avoiding Binance quoteOrderQty rejections).
                        final_qty = float(qty)

                        if self.max_spend_per_trade > 0:
                            est_spend = final_qty * current_price
                            if est_spend > self.max_spend_per_trade:
                                final_qty = round_step(self.max_spend_per_trade / current_price, step_size)

                        # --- Maker-first path (if supported and configured) ---
                        if self.maker_grace_s > 0 and hasattr(self.exchange_client, 'place_limit_post_only'):
                            try:
                                px = float(current_price) * (1.0 - 0.0001)  # ~1bp better for BUY
                                lim = await self.exchange_client.place_limit_post_only(
                                    symbol=symbol, side='BUY', quantity=final_qty, price=px, tag=safe_tag, clientOrderId=client_id
                                )
                                await self._log_execution_event('maker_order_placed', symbol, {
                                    'side': 'BUY', 'price': px, 'client_id': client_id, 'grace_s': self.maker_grace_s
                                })
                                await asyncio.sleep(self.maker_grace_s)
                                status = None
                                if hasattr(self.exchange_client, 'get_order_status'):
                                    with contextlib.suppress(Exception):
                                        status = await self.exchange_client.get_order_status(symbol=symbol, clientOrderId=client_id)
                                filled = bool(status and str(status.get('status', '')).upper() in ('FILLED', 'PARTIALLY_FILLED'))
                                if status is None:
                                    filled = False  # unknown → treat as not filled
                                if filled:
                                    await self._log_execution_event("order_placed", symbol, {
                                        "type": "LIMIT_POST_ONLY", "side": "BUY", "tag": safe_tag,
                                        "client_id": client_id, "using": "qty"
                                    })
                                    return lim
                                # cancel and consider taker fallback
                                with contextlib.suppress(Exception):
                                    if hasattr(self.exchange_client, 'cancel_order'):
                                        await self.exchange_client.cancel_order(symbol=symbol, clientOrderId=client_id)
                                cur_px = float(current_price)
                                bps = abs(cur_px - px) / max(1e-9, cur_px) * 10000.0
                                if self.allow_taker_if_within_bps > 0 and bps <= self.allow_taker_if_within_bps:
                                    await self._log_execution_event('maker_fallback_to_taker', symbol, {'bps': bps, 'limit_px': px, 'cur_px': cur_px})
                                else:
                                    await self._log_execution_event('order_skip', symbol, {'side': 'BUY', 'reason': 'maker_not_filled_and_bps_too_wide', 'bps': bps})
                                    return None
                            except Exception:
                                # fall back to market path if maker path errors
                                pass

                        order = await self._place_with_client_id(
                            symbol=symbol, side="BUY", quantity=final_qty, tag=safe_tag, clientOrderId=client_id
                        )
                        if order:
                            await self._log_execution_event("order_placed", symbol, {
                                "type": "MARKET", "side": "BUY", "tag": safe_tag,
                                "client_id": client_id, "using": "qty"
                            })
                        return order

                # else: qty path (BUY without planned_quote, or SELL)
                qty = round_step(quantity, step_size)
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
                    ok, adj_qty, _, _ = await validate_order_request(
                        side=side.upper(), qty=qty, price=current_price, filters=filters_obj,
                        taker_fee_bps=getattr(self.config, "TAKER_FEE_BPS", 10), use_quote_amount=None
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
                        if est_quote > self.max_spend_per_trade:
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

                    order = await self._place_with_client_id(
                        symbol=symbol, side=side.upper(), quantity=final_qty, tag=safe_tag, clientOrderId=client_id
                    )
                    if order:
                        await self._log_execution_event("order_placed", symbol, {
                            "type": "MARKET", "side": side.upper(), "tag": safe_tag,
                            "client_id": client_id, "using": "qty"
                        })
                    return order
            finally:
                self._active_symbol_side_orders.discard(order_key)

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
            symbol = kwargs.get("symbol", "UNKNOWN")
            side = kwargs.get("side", "UNKNOWN")
            
            try:
                return await self.exchange_client.place_market_order(**kwargs)
            except TypeError as te:
                # Some clients may not accept clientOrderId; retry without it
                if "clientOrderId" in kwargs:
                    self.logger.debug(f"[EM:TypeErr] clientOrderId not supported, retrying: {te}")
                    kwargs.pop("clientOrderId", None)
                    try:
                        return await self.exchange_client.place_market_order(**kwargs)
                    except Exception as retry_err:
                        # Classify retry error
                        return await self._classify_exchange_error(
                            symbol, side, retry_err, "place_market_order_retry"
                        )
                # If not a clientOrderId issue, re-raise as deterministic ExecutionError
                return await self._classify_exchange_error(
                    symbol, side, te, "place_market_order"
                )
            except BinanceAPIException as bex:
                # Binance-specific API error with code
                return await self._classify_exchange_error(
                    symbol, side, bex, "place_market_order"
                )
            except (ConnectionError, TimeoutError, asyncio.TimeoutError) as conn_err:
                # Network/connection issues
                self.logger.error(
                    f"[EM:ConnErr] {symbol} {side}: Connection error from ExchangeClient: {conn_err}"
                )
                return None  # Network error → order not sent (deterministic)
            except Exception as unknown_err:
                # P9 MANDATE: No unknown exceptions allowed
                self.logger.error(
                    f"[EM:UnknownErr] {symbol} {side}: Unclassified exception from place_market_order: "
                    f"{type(unknown_err).__name__}: {unknown_err}",
                    exc_info=True
                )
                return None  # Unknown error → order not sent (deterministic fallback)

        async def _classify_exchange_error(
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
                    self.logger.warning(
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
        async def start(self):
            """
            Minimal start hook so AppContext can warm this component during P5.
            Safely warms symbol filters and returns immediately.
            """
            try:
                # Initialize semaphores and heartbeat when loop is available
                self._ensure_semaphores_ready()
                
                # Start heartbeat task if not already started
                if self._heartbeat_task is None or self._heartbeat_task.done():
                    try:
                        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop(), name="ExecutionManager:heartbeat")
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
