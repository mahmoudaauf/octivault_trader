import logging
import asyncio
import time
import os
from inspect import iscoroutine
from typing import Optional, List, Dict, Any, Tuple
from math import fsum
from utils.shared_state_tools import fee_bps
from core.exit_utils import post_exit_bookkeeping

class TPSLEngine:
    """
    Dynamic TP/SL engine:
      • ATR(14) from recent candles (prefers 5m, falls back to 1m)
      • Sentiment + regime aware TP/SL multipliers
      • Stores initial TP/SL on fill (EM will call set_initial_tp_sl)
      • Monotonic, drift-free scheduler
      • Concurrent, rate-limited exits with debounce
    """

    def __init__(self, shared_state, config, execution_manager, **kwargs):
        self.shared_state = shared_state
        self.config = config
        self.execution_manager = execution_manager
        self.logger = logging.getLogger("TPSLEngine")
        self._stop_event = asyncio.Event()
        self._task = None
        self._heartbeat_task = None

        # Defensive defaults on shared_state attributes we use
        if not hasattr(self.shared_state, "open_trades"): self.shared_state.open_trades = {}
        if not hasattr(self.shared_state, "latest_prices"): self.shared_state.latest_prices = {}
        if not hasattr(self.shared_state, "sentiment_score"): self.shared_state.sentiment_score = {}
        if not hasattr(self.shared_state, "volatility_state"): self.shared_state.volatility_state = {}

        # Config snapshot (hot-path)
        self._interval = float(getattr(self.config, "TPSL_CHECK_INTERVAL", 10) or 10)
        self._close_concurrency = int(getattr(self.config, "MAX_CONCURRENT_CLOSES", 5) or 5)
        self._fallback_atr_pct = float(getattr(self.config, "TPSL_FALLBACK_ATR_PCT", 0.01) or 0.01)
        self._debounce_close_sec = float(getattr(self.config, "TPSL_DEBOUNCE_CLOSE_SEC", 5.0) or 5.0)

        # Time helpers & close debounce state
        self._mono = time.monotonic
        self._session_start = time.monotonic()  # Track when this session started
        self._last_close_attempt: Dict[str, float] = {}
        self._tp_floor_hit: Dict[str, bool] = {}
        self._last_tp_sl_method: Dict[str, str] = {}
        self._last_profit_audit_ts: Dict[str, float] = {}
        self._price_stale_sec = float(getattr(self.config, "TPSL_PRICE_STALE_SEC", 120.0) or 120.0)
        self._ohlcv_stale_sec = float(getattr(self.config, "TPSL_OHLCV_STALE_SEC", 300.0) or 300.0)
        self._min_notional_safety = float(getattr(self.config, "TPSL_MIN_NOTIONAL_SAFETY", 1.0) or 1.0)
        # Default strategy contract: hybrid ATR targets + time-based exits
        self._tp_sl_strategy = str(getattr(self.config, "TPSL_STRATEGY", "hybrid_atr_time") or "hybrid_atr_time")
        self._tp_sl_calc_model = str(getattr(self.config, "TPSL_CALC_MODEL", "atr_pct") or "atr_pct")
        self._status_timeout_sec = float(getattr(self.config, "TPSL_STATUS_TIMEOUT_SEC", 1.0) or 1.0)
        self._profit_audit_enabled = bool(getattr(self.config, "TPSL_PROFIT_AUDIT", False))
        self._profit_audit_sec = float(getattr(self.config, "TPSL_PROFIT_AUDIT_SEC", 300.0) or 300.0)
        self._time_exit_enabled = bool(getattr(self.config, "TPSL_TIME_EXIT_ENABLED", False))
    async def start(self):
        """
        P9 contract: start() creates the monitoring task and emits an initial status.
        Uses the existing `run()` loop.
        """
        if getattr(self, "_task", None) and not self._task.done():
            return
        self._stop_event.clear()
        
        # Start heartbeat first to satisfy Watchdog during startup gates
        if not self._heartbeat_task or self._heartbeat_task.done():
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop(), name="TPSLEngine:heartbeat")

        # Auto-arm TP/SL for existing open positions (startup safety)
        try:
            if bool(getattr(self.config, "TPSL_AUTO_ARM_ON_STARTUP", True)):
                await self._auto_arm_existing_trades()
        except Exception:
            self.logger.debug("TPSL auto-arm on startup failed (non-fatal)", exc_info=True)
            
        self._task = asyncio.create_task(self.run())
        await self._safe_status_update("Initialized", "Ready")

    async def _auto_arm_existing_trades(self) -> None:
        """Ensure TP/SL is set for existing open positions at startup."""
        open_trades = dict(getattr(self.shared_state, "open_trades", {}) or {})
        positions = getattr(self.shared_state, "positions", {}) or {}
        symbols = set(open_trades.keys()) | set(positions.keys())

        for symbol in symbols:
            tr = open_trades.get(symbol, {}) if isinstance(open_trades, dict) else {}
            if not isinstance(tr, dict):
                tr = {}
            if tr.get("tp") is not None and tr.get("sl") is not None:
                continue

            entry_price = float(tr.get("entry_price") or tr.get("avg_price") or tr.get("entry") or 0.0)
            qty = float(tr.get("quantity") or tr.get("qty") or 0.0)

            if (entry_price <= 0 or qty <= 0) and isinstance(positions, dict):
                pos = positions.get(symbol, {}) or {}
                if entry_price <= 0:
                    entry_price = float(
                        pos.get("avg_price")
                        or pos.get("entry_price")
                        or pos.get("mark_price")
                        or 0.0
                    )
                if qty <= 0:
                    qty = float(pos.get("quantity") or pos.get("qty") or 0.0)

            if entry_price > 0 and qty > 0:
                tier = tr.get("tier")
                tp, sl = self.set_initial_tp_sl(symbol, entry_price, qty, tier=tier)
                self.logger.info(
                    "[TPSL:auto_arm] %s tp=%.6f sl=%.6f entry=%.6f qty=%.6f",
                    symbol, float(tp or 0.0), float(sl or 0.0), entry_price, qty
                )
            else:
                self.logger.warning(
                    "[TPSL:auto_arm] %s missing entry/qty; TP/SL not armed (entry=%.6f qty=%.6f)",
                    symbol, entry_price, qty
                )

    async def stop(self):
        """
        P9 contract: stop() cancels the monitoring task and emits a final status.
        """
        self._stop_event.set()
        
        # Cancel heartbeat
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            self._heartbeat_task = None

        t = getattr(self, "_task", None)
        self._task = None
        if t:
            try:
                t.cancel()
                try:
                    await asyncio.wait_for(t, timeout=float(getattr(self.config, "STOP_JOIN_TIMEOUT_S", 5.0)))
                except asyncio.CancelledError:
                    pass
            except Exception:
                self.logger.debug("TPSLEngine stop wait failed", exc_info=True)
        try:
            await self.shared_state.update_component_status("TPSLEngine", "Stopped", "Stopped by request")
        except Exception:
            self.logger.debug("TPSLEngine final health update failed", exc_info=True)

    async def _heartbeat_loop(self):
        """Continuous heartbeat to satisfy Watchdog when no trades exist or loop is gated."""
        while True:
            try:
                await self._safe_status_update("Operational", "Heartbeat: Active / Monitoring")
                _emit_health(self.shared_state, "Operational", "Heartbeat: Active / Monitoring")
            except Exception:
                pass
            await asyncio.sleep(60)

    async def _safe_status_update(self, status: str, message: str) -> None:
        ts = time.time()
        try:
            if hasattr(self.shared_state, "update_timestamp"):
                res = self.shared_state.update_timestamp("TPSLEngine")
                if iscoroutine(res):
                    await res
                res = self.shared_state.update_timestamp("TP_SLEngine")
                if iscoroutine(res):
                    await res
        except Exception:
            pass
        try:
            statuses = getattr(self.shared_state, "component_statuses", None)
            if isinstance(statuses, dict):
                statuses["TPSLEngine"] = {
                    "status": status,
                    "message": message,
                    "timestamp": ts,
                    "ts": ts,
                }
                statuses["TP_SLEngine"] = {
                    "status": status,
                    "message": message,
                    "timestamp": ts,
                    "ts": ts,
                }
            last_seen = getattr(self.shared_state, "component_last_seen", None)
            if isinstance(last_seen, dict):
                last_seen["TPSLEngine"] = ts
                last_seen["TP_SLEngine"] = ts
        except Exception:
            pass
        try:
            uh = getattr(self.shared_state, "update_system_health", None)
            if callable(uh):
                res = uh(component="TPSLEngine", status=status, message=message)
                if iscoroutine(res):
                    await asyncio.wait_for(res, timeout=self._status_timeout_sec)
        except Exception:
            pass
        try:
            cs = getattr(self.shared_state, "update_component_status", None)
            if callable(cs):
                res = cs("TPSLEngine", status, message)
                if iscoroutine(res):
                    await asyncio.wait_for(res, timeout=self._status_timeout_sec)
                res = cs("TP_SLEngine", status, message)
                if iscoroutine(res):
                    await asyncio.wait_for(res, timeout=self._status_timeout_sec)
        except Exception:
            pass

    # ---------- small helpers ----------

    def _safe_set_cot(self, symbol: str, src: str, msg: str) -> None:
        """Fire-and-forget CoT explanation; uses record_cot_explanations if available."""
        setter = getattr(self.shared_state, "record_cot_explanations", None)
        if not setter:
            return
        try:
            # record_cot_explanations(symbol, agent, text)
            res = setter(symbol, src, msg)
            if iscoroutine(res):
                try:
                    asyncio.get_running_loop().create_task(res)
                except RuntimeError:
                    # no running loop; ignore (non-critical)
                    pass
        except Exception:
            # very chatty when exc_info=True; keep quiet on TPSL hot path
            self.logger.debug("set_cot_explanation failed (non-fatal)")

    def _maybe_log_profit_audit(self, symbol: str, tp_pct: float, sl_pct: float) -> None:
        if not self._profit_audit_enabled:
            return
        now = time.time()
        last = self._last_profit_audit_ts.get(symbol, 0.0)
        if (now - last) < self._profit_audit_sec:
            return
        self._last_profit_audit_ts[symbol] = now

        taker_bps = float(fee_bps(self.shared_state, "taker") or 10.0)
        fee_pct = (taker_bps * 2.0) / 10000.0
        slippage_pct = float(getattr(self.config, "EXIT_SLIPPAGE_BPS", getattr(self.config, "CR_PRICE_SLIPPAGE_BPS", 0.0)) or 0.0) / 10000.0
        buffer_pct = float(getattr(self.config, "TP_MIN_BUFFER_BPS", 0.0) or 0.0) / 10000.0
        net_tp_pct = tp_pct - (fee_pct + slippage_pct)
        net_sl_pct = sl_pct + (fee_pct + slippage_pct)
        denom = net_tp_pct + net_sl_pct
        win_rate_be = (net_sl_pct / denom) if denom > 0 else 1.0

        self.logger.info(
            "[TPSL:profit_audit] %s tp=%.3f%% sl=%.3f%% net_tp=%.3f%% net_sl=%.3f%% fee=%.3f%% slip=%.3f%% buffer=%.3f%% breakeven=%.1f%%",
            symbol,
            tp_pct * 100.0,
            sl_pct * 100.0,
            net_tp_pct * 100.0,
            net_sl_pct * 100.0,
            fee_pct * 100.0,
            slippage_pct * 100.0,
            buffer_pct * 100.0,
            win_rate_be * 100.0,
        )

    def _pick_candles(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Prefer shared_state.market_data[symbol]['5m']['ohlcv']; fallback to '1m'.
        Normalizes rows to dicts: {'timestamp','open','high','low','close'}.
        """
        md = getattr(self.shared_state, "market_data", {}) or {}
        getter = getattr(self.shared_state, "get_market_data_for_symbol", None)
        if callable(getter):
            tf_map = getter(symbol) or {}
        else:
            tf_map = md.get(symbol) or {}
        ohlcv = None
        rec_5 = tf_map.get("5m") if isinstance(tf_map, dict) else None
        if isinstance(rec_5, dict):
            ohlcv = rec_5.get("ohlcv")
        if not ohlcv:
            rec_1 = tf_map.get("1m") if isinstance(tf_map, dict) else None
            if isinstance(rec_1, dict):
                ohlcv = rec_1.get("ohlcv")
        if not ohlcv:
            return []

        out = []
        ap = out.append
        for row in ohlcv:
            if isinstance(row, dict):
                ap({
                    "timestamp": row.get("timestamp"),
                    "open": float(row.get("open", 0.0)),
                    "high": float(row.get("high", 0.0)),
                    "low": float(row.get("low", 0.0)),
                    "close": float(row.get("close", 0.0)),
                })
            elif isinstance(row, (list, tuple)) and len(row) >= 5:
                ap({
                    "timestamp": row[0],
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                })
        return out

    def _compute_atr(self, symbol: str, lookback: int = 14) -> Optional[float]:
        """
        ATR(lookback) using the last lookback+1 candles.
        TR = max(H-L, |H-prevC|, |L-prevC|)
        """
        try:
            candles = self._pick_candles(symbol)
            n = len(candles)
            if n < lookback + 1:
                return None
            window = candles[-(lookback + 1):]
            trs: List[float] = []
            prev_c = window[0]["close"]
            for i in range(1, len(window)):
                h = window[i]["high"]; l = window[i]["low"]
                tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
                trs.append(tr)
                prev_c = window[i]["close"]
            if len(trs) < lookback:
                return None
            atr = fsum(trs[-lookback:]) / float(lookback)
            return atr if atr > 0 else None
        except Exception:
            return None

    def _get_min_notional_sync(self, symbol: str) -> float:
        sym = (symbol or "").upper().replace("/", "")
        try:
            filters = getattr(self.shared_state, "symbol_filters", {}).get(sym, {}) or {}
        except Exception:
            filters = {}
        min_notional = float(
            filters.get("MIN_NOTIONAL", {}).get("minNotional")
            or filters.get("minNotional")
            or filters.get("_normalized", {}).get("min_notional", 0.0)
            or 0.0
        )
        return min_notional

    def _pre_activation_guard(self, symbol: str, entry_price: float, qty: float) -> Tuple[bool, str]:
        if entry_price <= 0:
            return False, "missing_entry_or_qty"
        if qty <= 0:
            try:
                pos = getattr(self.shared_state, "positions", {}).get(symbol, {})
                if isinstance(pos, dict):
                    qty = float(pos.get("quantity", 0.0) or 0.0)
            except Exception:
                qty = qty
        if qty <= 0:
            return False, "missing_entry_or_qty"
        notional = float(entry_price) * float(qty)
        min_notional = self._get_min_notional_sync(symbol)
        safety = float(self._min_notional_safety or 1.0)
        if min_notional > 0 and notional < (min_notional * safety):
            return False, "below_min_notional"
        return True, "ok"

    def _is_price_stale(self, symbol: str) -> bool:
        if self._price_stale_sec <= 0:
            return False
        ts = None
        for key in ("latest_price_ts", "latest_prices_ts", "price_timestamps"):
            try:
                mp = getattr(self.shared_state, key, None)
                if isinstance(mp, dict) and symbol in mp:
                    ts = mp.get(symbol)
                    break
            except Exception:
                pass
        if ts is None:
            return False
        try:
            ts_val = float(ts or 0.0)
        except Exception:
            return False
        if ts_val > 1e12:
            ts_val /= 1000.0
        return (time.time() - ts_val) > self._price_stale_sec

    def _is_ohlcv_stale(self, symbol: str) -> bool:
        if self._ohlcv_stale_sec <= 0:
            return False
        candles = self._pick_candles(symbol)
        if not candles:
            return True
        last_ts = candles[-1].get("timestamp")
        if not last_ts:
            return False
        try:
            ts_val = float(last_ts)
        except Exception:
            return False
        if ts_val > 1e12:
            ts_val /= 1000.0
        return (time.time() - ts_val) > self._ohlcv_stale_sec

    def _plan_reason_code(self, reason: str) -> str:
        ru = str(reason or "").upper()
        if "TP" in ru:
            return "TP_HIT"
        if "SL" in ru:
            return "SL_HIT"
        if "MAX-HOLD" in ru or "MAX_HOLD" in ru:
            return "MAX_HOLD"
        if "CAPITAL RECOVERY" in ru:
            return "CAPITAL_RECOVERY"
        if "STAGNATION" in ru:
            return "STAGNATION_PURGE"
        if "BREAKEVEN" in ru:
            return "BREAKEVEN_EXIT"
        if "MICRO-CYCLE" in ru or "MICRO CYCLE" in ru:
            return "MICRO_CYCLE"
        if "TTL" in ru:
            return "TIER_TTL"
        return "TPSL_EXIT"

    # ---------- public API used by EM ----------

    def set_initial_tp_sl(self, symbol: str, entry_price: float, quantity: float, tier: Optional[str] = None) -> Tuple[Optional[float], Optional[float]]:
        """
        Called by ExecutionManager right after a fill.
        Computes and stores TP/SL on the open_trades record (if present).
        
        Tier-aware (Phase A Frequency Engineering):
        - Tier B: Tighter TP, slightly wider SL, shorter TTL for fast capital velocity
        - Tier A: Normal TP/SL behavior
        
        Returns (tp, sl).
        """
        ok, reason = self._pre_activation_guard(symbol, entry_price, quantity)
        if not ok:
            self.logger.info("[TPSL:PreGuard] %s skip set_initial_tp_sl: %s", symbol, reason)
            return None, None
        tp, sl = self.calculate_tp_sl(symbol, entry_price, tier=tier)
        try:
            ot = self.shared_state.open_trades.get(symbol)
            if not isinstance(ot, dict):
                ot = {
                    "entry_price": entry_price,
                    "quantity": quantity,
                    "position": "long",
                }
            ot["tp"] = tp
            ot["sl"] = sl
            ot["tier"] = tier  # Store tier for monitoring
            ot["tp_sl_method"] = self._last_tp_sl_method.get(symbol)
            ot.setdefault("entry_price", entry_price)
            ot.setdefault("quantity", quantity)

            # Tier B: Set shorter TTL for fast exits
            if tier == "B":
                ot["ttl_sec"] = int(getattr(self.config, "TIER_B_TTL_SEC", 300))  # 5 minutes default
                ot["created_at"] = time.time()

            self.shared_state.open_trades[symbol] = ot
        except Exception:
            pass
        return tp, sl

    # ---------- core logic ----------

    def calculate_tp_sl(self, symbol: str, entry_price: float, tier: Optional[str] = None) -> Tuple[Optional[float], Optional[float]]:
        """
        Compute TP/SL given entry, ATR, sentiment, regime, and tier.
        
        Tier-aware (Phase A Frequency Engineering):
        - Tier B: Tighter TP (0.8x), slightly wider SL (1.1x) for fast capital velocity
        - Tier A: Normal multipliers
        """
        try:
            md = getattr(self.shared_state, "market_data", {}) or {}
            atr_cached = 0.0
            sym_md = md.get(symbol)
            if isinstance(sym_md, dict):
                with_5m = sym_md.get("5m") or {}
                atr_cached = float(sym_md.get("atr") or with_5m.get("atr") or 0.0)

            atr_live = self._compute_atr(symbol, lookback=14)
            atr = atr_live or atr_cached
            sentiment = float(self.shared_state.sentiment_score.get(symbol, 0.0) or 0.0)
            regime = (self.shared_state.volatility_state.get(symbol, "sideways") or "sideways").lower()

            # Base multipliers from Phase A Config
            tp_mul = float(getattr(self.config, "TP_ATR_MULT", 1.2))
            sl_mul = float(getattr(self.config, "SL_ATR_MULT", 0.8))
            
            # Tier-specific adjustments (Phase A Frequency Engineering)
            if tier == "B":
                # Tier B: Tighter TP for quick profit-taking, slightly wider SL for breathing room
                tp_mul *= 0.8  # Reduce TP distance (faster exits)
                sl_mul *= 1.1  # Increase SL distance (more tolerance for micro trades)
                tier_label = " [Tier-B: tight TP, wide SL]"
            else:
                tier_label = ""
            
            # Regime adjustments
            if regime in ("trend", "uptrend", "downtrend"):
                tp_mul += 0.5
            elif regime in ("high_vol", "high"):
                sl_mul += 0.5

            # Sentiment adjustments
            if sentiment > 0.5:
                tp_mul += 0.3
            elif sentiment < -0.5:
                sl_mul += 0.3

            # [FIX #6] Add multiplier bounds to prevent extreme TP/SL values
            tp_mul = max(0.5, min(3.0, tp_mul))  # 0.5x to 3.0x range
            sl_mul = max(0.3, min(2.0, sl_mul))  # 0.3x to 2.0x range

            if not atr or atr <= 0:
                self._safe_set_cot(symbol, "TPSLEngine", f"Veto: ATR too small/zero ({atr})")
                atr = entry_price * self._fallback_atr_pct  # fallback ATR
                self.logger.debug("[%s] ATR too small; fallback=%.6f", symbol, atr)

            method = "atr" if atr_live else ("atr_cached" if atr_cached > 0 else "atr_fallback")
            if tier == "B":
                method += "+tier_b"
            if sentiment != 0.0 or regime not in ("sideways", ""):
                method += "+context"
            self._last_tp_sl_method[symbol] = method

            tp_dist = atr * tp_mul
            sl_dist = atr * sl_mul

            # Optional percent clamps to keep TP/SL in tight target ranges
            tp_pct_min = float(getattr(self.config, "TP_PCT_MIN", 0.003) or 0.003)
            tp_pct_max = float(getattr(self.config, "TP_PCT_MAX", 0.006) or 0.006)
            sl_pct_min = float(getattr(self.config, "SL_PCT_MIN", 0.005) or 0.005)
            sl_pct_max = float(getattr(self.config, "SL_PCT_MAX", 0.008) or 0.008)

            # Fee-aware TP floor: 2*fee + slippage + buffer (with fee multiplier)
            taker_bps = float(fee_bps(self.shared_state, "taker") or 10.0)
            slippage_bps = float(getattr(self.config, "EXIT_SLIPPAGE_BPS", getattr(self.config, "CR_PRICE_SLIPPAGE_BPS", 0.0)) or 0.0)
            buffer_bps = float(getattr(self.config, "TP_MIN_BUFFER_BPS", 0.0) or 0.0)
            entry_fee_mult = float(getattr(self.config, "MIN_PLANNED_QUOTE_FEE_MULT", 2.5) or 2.5)
            fee_mult = float(getattr(self.config, "MIN_PROFIT_EXIT_FEE_MULT", 2.0) or 2.0)
            fee_mult = max(fee_mult, entry_fee_mult)
            base_floor = (taker_bps * 2.0 + slippage_bps + buffer_bps) / 10000.0
            tp_floor = max(base_floor, (taker_bps * 2.0 * fee_mult + slippage_bps + buffer_bps) / 10000.0)

            # Dynamic TP floor for micro trades: add extra bps when notional is small
            micro_notional = float(getattr(self.config, "TP_MICRO_NOTIONAL_USDT", 25.0) or 25.0)
            micro_extra_bps = float(getattr(self.config, "TP_MICRO_EXTRA_BPS", 0.0) or 0.0)
            try:
                qty = 0.0
                ot = getattr(self.shared_state, "open_trades", {}) or {}
                if isinstance(ot, dict):
                    qty = float((ot.get(symbol) or {}).get("quantity", 0.0) or 0.0)
                if qty <= 0 and hasattr(self.shared_state, "get_position_qty"):
                    qty = float(self.shared_state.get_position_qty(symbol) or 0.0)
                trade_notional = float(entry_price) * float(qty)
            except Exception:
                trade_notional = 0.0
            if micro_extra_bps > 0 and micro_notional > 0 and trade_notional > 0 and trade_notional < micro_notional:
                scale = max(0.0, 1.0 - (trade_notional / micro_notional))
                tp_floor += (micro_extra_bps * scale) / 10000.0

            raw_tp_pct = (tp_dist / entry_price) if entry_price > 0 else 0.0
            self._tp_floor_hit[symbol] = bool(raw_tp_pct > 0 and raw_tp_pct < tp_floor)

            tp_pct_min = max(tp_pct_min, tp_floor)  # Clamp target floor

            if entry_price > 0:
                tp_pct = tp_dist / entry_price
                sl_pct = sl_dist / entry_price
                
                # Apply tiered caps for frequency
                if tier == "B":
                    # Tier B: Aggressive frequency (cap at min_tp)
                    tp_pct = tp_pct_min
                else:
                    tp_pct = max(tp_pct_min, min(tp_pct_max, tp_pct))
                    
                sl_pct = max(sl_pct_min, min(sl_pct_max, sl_pct))
                tp_dist = entry_price * tp_pct
                sl_dist = entry_price * sl_pct
                self._maybe_log_profit_audit(symbol, tp_pct, sl_pct)

            tp_price = round(entry_price + tp_dist, 4)
            sl_price = round(entry_price - sl_dist, 4)

            self.logger.debug("📏 TP/SL %s: TP=%.4f SL=%.4f (ATR=%.6f, Sent=%.2f, Reg=%s)%s",
                              symbol, tp_price, sl_price, atr, sentiment, regime, tier_label)
            return tp_price, sl_price
        except Exception as e:
            self.logger.error("TP/SL calc failed for %s: %s", symbol, e)
            return None, None

    async def _current_price(self, symbol: str) -> Optional[float]:
        """
        Prefer SharedState.safe_price() (handles cache/callback/exchange). Fallback to latest_prices.
        """
        try:
            safe_price = getattr(self.shared_state, "safe_price", None)
            if callable(safe_price):
                px = await safe_price(symbol, default=0.0)
                return float(px) if px and px > 0 else None
        except Exception:
            pass
        try:
            p = self.shared_state.latest_prices.get(symbol)
            return float(p) if p and p > 0 else None
        except Exception:
            return None

    def _passes_profit_gate(self, pnl_pct: float, reason: str) -> bool:
        """Return True if a SELL reason is allowed by fee-aware profit gate."""
        reason_u = str(reason or "").upper()
        if "SL" in reason_u or "STOP_LOSS" in reason_u:
            return True

        entry_fee_mult = float(getattr(self.config, "MIN_PLANNED_QUOTE_FEE_MULT", 2.5) or 2.5)
        fee_mult = float(getattr(self.config, "MIN_PROFIT_EXIT_FEE_MULT", 2.0) or 2.0)
        fee_mult = max(fee_mult, entry_fee_mult)
        rt_fee_pct = ((fee_bps(self.shared_state, "taker") or 10.0) * 2.0) / 10000.0
        slippage_pct = float(getattr(self.config, "EXIT_SLIPPAGE_BPS", getattr(self.config, "CR_PRICE_SLIPPAGE_BPS", 0.0)) or 0.0) / 10000.0
        buffer_pct = float(getattr(self.config, "TP_MIN_BUFFER_BPS", 0.0) or 0.0) / 10000.0
        min_net_pct = float(getattr(self.config, "MIN_NET_PROFIT_AFTER_FEES", 0.0035) or 0.0035)
        fee_floor = (rt_fee_pct * fee_mult) + slippage_pct + buffer_pct
        min_profit = max(min_net_pct, fee_floor)
        return pnl_pct >= min_profit

    def _passes_net_exit_gate(self, pnl_pct: float) -> bool:
        """Exit gate: realized pnl must clear exit-side costs (fee + slippage + buffer)."""
        taker_bps = float(fee_bps(self.shared_state, "taker") or 10.0)
        fee_pct = taker_bps / 10000.0
        slippage_pct = float(getattr(self.config, "EXIT_SLIPPAGE_BPS", getattr(self.config, "CR_PRICE_SLIPPAGE_BPS", 0.0)) or 0.0) / 10000.0
        buffer_pct = float(getattr(self.config, "TP_MIN_BUFFER_BPS", 0.0) or 0.0) / 10000.0
        min_exit = fee_pct + slippage_pct + buffer_pct
        if pnl_pct < min_exit:
            self.logger.info(
                "[TPSL:ExitGate] blocked pnl=%.4f%% < min_exit=%.4f%%",
                pnl_pct * 100.0,
                min_exit * 100.0,
            )
            return False
        return True

    async def _passes_excursion_gate(self, symbol: str, entry_price: float, exit_price: float, atr: Optional[float], reason: str) -> bool:
        """Minimum price excursion gate for non-SL TP exits."""
        reason_u = str(reason or "").upper()
        if "SL" in reason_u or "STOP_LOSS" in reason_u or "EMERGENCY" in reason_u:
            return True

        if entry_price <= 0 or exit_price <= 0:
            self.logger.info("[TPSL:ExcursionGate] %s blocked (missing entry/exit price).", symbol)
            return False

        sym = (symbol or "").upper().replace("/", "")
        filters = {}
        try:
            getter = getattr(self.shared_state, "get_symbol_filters_cached", None)
            if callable(getter):
                res = getter(sym)
                if iscoroutine(res):
                    res = await res
                if isinstance(res, dict):
                    filters = res
        except Exception:
            filters = {}
        if not filters:
            filters = getattr(self.shared_state, "symbol_filters", {}).get(sym, {}) or {}

        tick_size = float(
            filters.get("PRICE_FILTER", {}).get("tickSize")
            or filters.get("tickSize")
            or filters.get("_normalized", {}).get("tick_size", 0.0)
            or 0.0
        )

        spread = 0.0
        try:
            ex = getattr(self.shared_state, "exchange_client", None)
            if ex and hasattr(ex, "get_best_bid_ask"):
                bid, ask = await ex.get_best_bid_ask(symbol)
                if bid and ask and ask > 0 and bid > 0:
                    spread = abs(float(ask) - float(bid))
        except Exception:
            spread = 0.0

        atr_val = float(atr or 0.0)
        if atr_val <= 0:
            atr_val = entry_price * float(getattr(self.config, "TPSL_FALLBACK_ATR_PCT", 0.01) or 0.01)

        tick_mult = float(getattr(self.config, "EXIT_EXCURSION_TICK_MULT", 2.0) or 2.0)
        atr_mult = float(getattr(self.config, "EXIT_EXCURSION_ATR_MULT", 0.35) or 0.35)
        spread_mult = float(getattr(self.config, "EXIT_EXCURSION_SPREAD_MULT", 3.0) or 3.0)
        min_tick_move = tick_size * tick_mult if tick_size > 0 else 0.0
        threshold = max(min_tick_move, atr_mult * atr_val, spread_mult * spread)
        if threshold <= 0:
            return True

        excursion = abs(exit_price - entry_price)
        if excursion < threshold:
            self.logger.info(
                "[TPSL:ExcursionGate] %s blocked excursion=%.6f < threshold=%.6f (tick=%.6f atr=%.6f spread=%.6f)",
                symbol, excursion, threshold, min_tick_move, atr_val, spread,
            )
            return False

        return True

    async def _get_min_notional(self, symbol: str) -> float:
        sym = (symbol or "").upper().replace("/", "")
        filters = {}
        try:
            getter = getattr(self.shared_state, "get_symbol_filters_cached", None)
            if callable(getter):
                res = getter(sym)
                if iscoroutine(res):
                    res = await res
                if isinstance(res, dict):
                    filters = res
        except Exception:
            filters = {}
        if not filters:
            filters = getattr(self.shared_state, "symbol_filters", {}).get(sym, {}) or {}

        min_notional = float(
            filters.get("MIN_NOTIONAL", {}).get("minNotional")
            or filters.get("minNotional")
            or filters.get("_normalized", {}).get("min_notional", 0.0)
            or 0.0
        )
        return min_notional

    async def _force_close_all_open_lots(self, symbol: str, reason: str) -> None:
        force_fn = getattr(self.shared_state, "force_close_all_open_lots", None)
        if callable(force_fn):
            res = force_fn(symbol, reason=reason)
            if iscoroutine(res):
                await res
            return
        try:
            ot = getattr(self.shared_state, "open_trades", None)
            if isinstance(ot, dict):
                ot.pop(symbol, None)
        except Exception:
            pass

    async def _emit_tp_sl_plan(
        self,
        symbol: str,
        reason: str,
        position: str,
        position_qty: Optional[float],
        entry_price: Optional[float],
        current_price: Optional[float],
        tp: Optional[float],
        sl: Optional[float],
        pnl_pct: Optional[float],
        tier: Optional[str],
        method: Optional[str],
    ) -> None:
        emitter = getattr(self.shared_state, "emit_event", None)
        if not callable(emitter):
            return
        payload = {
            "symbol": symbol,
            "reason": reason,
            "reason_code": self._plan_reason_code(reason),
            "position": position,
            "side": "SELL" if position == "long" else "BUY",
            "position_qty": position_qty,
            "entry_price": entry_price,
            "current_price": current_price,
            "tp": tp,
            "sl": sl,
            "pnl_pct": pnl_pct,
            "tier": tier,
            "method": method,
            "strategy": self._tp_sl_strategy,
            "calc_model": self._tp_sl_calc_model,
            "tag": "tp_sl",
            "ts": time.time(),
        }
        try:
            res = emitter("TP_SL_PLAN", payload)
            if iscoroutine(res):
                await res
        except Exception:
            pass

    async def check_orders(self):
        """
        Evaluate open trades; if TP/SL is hit → close via EM (bounded concurrency).
        Debounces repeated closes per symbol.
        """
        def _log_skip(symbol: str, qty: float, price: Optional[float], tp: Optional[float], sl: Optional[float], reason: str) -> None:
            self.logger.info(
                "[TPSL:skip] %s qty=%.8f price=%s tp=%s sl=%s reason=%s",
                symbol,
                float(qty or 0.0),
                f"{price:.8f}" if price is not None else "None",
                f"{tp:.8f}" if tp is not None else "None",
                f"{sl:.8f}" if sl is not None else "None",
                reason,
            )
        try:
            open_trades = dict(self.shared_state.open_trades or {})  # snapshot
            prices = self.shared_state.latest_prices or {}
            self.logger.info("[TPSL:check] open_trades=%d prices=%d", len(open_trades), len(prices))
            self.logger.info("[TPSL:check] checking orders")
            to_close: List[Tuple[str, str]] = []

            for symbol, tr in open_trades.items():
                position = tr.get("position")
                entry_price = tr.get("entry_price")
                if not position or entry_price is None:
                    self.logger.debug(
                        "[%s] TPSL skip: missing position or entry_price (position=%s entry_price=%s)",
                        symbol, position, entry_price
                    )
                    _log_skip(symbol, 0.0, None, None, None, "missing_position_or_entry")
                    continue

                qty = float(tr.get("quantity") or 0.0)
                ok, pre_reason = self._pre_activation_guard(symbol, float(entry_price), qty)
                if not ok:
                    self.logger.info("[TPSL:PreGuard] %s skip evaluate: %s", symbol, pre_reason)
                    _log_skip(symbol, qty, None, None, None, pre_reason)
                    continue

                if self._is_ohlcv_stale(symbol):
                    self.logger.info("[TPSL:StaleOHLCV] %s market data stale; deferring", symbol)
                    _log_skip(symbol, qty, None, None, None, "stale_ohlcv")
                    continue

                if self._is_price_stale(symbol):
                    self.logger.info("[TPSL:StalePrice] %s price stale; deferring", symbol)
                    _log_skip(symbol, qty, None, None, None, "stale_price")
                    continue

                # [FIX #4] Compute ATR per symbol for trailing SL support
                atr = self._compute_atr(symbol, lookback=14)
                if atr is None:
                    # Fallback: use cached or computed fallback ATR
                    atr_cached = 0.0
                    md = getattr(self.shared_state, "market_data", {}) or {}
                    sym_md = md.get(symbol)
                    if isinstance(sym_md, dict):
                        with_5m = sym_md.get("5m") or {}
                        atr_cached = float(sym_md.get("atr") or with_5m.get("atr") or 0.0)
                    atr = atr_cached or (float(entry_price) * self._fallback_atr_pct)

                now = self._mono()
                last = self._last_close_attempt.get(symbol, 0.0)
                
                # [FIX #8] Debounce logic: if symbol not in dict yet (first call), allow it.
                # Otherwise, require debounce_close_sec to have passed since last attempt.
                if symbol in self._last_close_attempt:
                    debounce_ok = now - last >= self._debounce_close_sec
                else:
                    debounce_ok = True  # First call for this symbol, no debounce
                
                if not debounce_ok:
                    continue  # debounce

                # Calculate age once (Phase A optimization)
                positions = getattr(self.shared_state, "positions", {}) or {}
                pos = positions.get(symbol, {}) if isinstance(positions, dict) else {}
                created_at = float(
                    tr.get("created_at")
                    or tr.get("opened_at")
                    or pos.get("entry_time")
                    or pos.get("opened_at")
                    or 0.0
                )
                age_sec = time.time() - created_at if created_at > 0 else 0.0

                if self._time_exit_enabled:
                    # 1. Hard max-hold time exit (no exceptions)
                    max_hold_sec = float(getattr(self.config, "MAX_HOLD_TIME_SEC", 1800.0) or 1800.0)
                    if max_hold_sec > 0 and age_sec >= max_hold_sec:
                        self.logger.info(
                            "[TPSL:MAX_HOLD_EXIT] %s age=%.0fs >= max_hold=%.0fs",
                            symbol, age_sec, max_hold_sec
                        )
                        to_close.append((
                            symbol,
                            f"Max-hold exit: age={int(age_sec)}s >= {int(max_hold_sec)}s"
                        ))
                        continue

                # 2. Capital recovery mode: force small-profit or time-based exit
                try:
                    cap_rec = getattr(self.shared_state, "capital_recovery_mode", {}) or {}
                except Exception:
                    cap_rec = {}

                cap_rec_active = isinstance(cap_rec, dict) and cap_rec.get("active")
                # Best-practice: do not let recovery exits fire when TP/SL is armed
                try:
                    if cap_rec_active and bool(getattr(self.config, "CAPITAL_RECOVERY_TPSL_GUARD", True)):
                        if tr.get("tp") is not None and tr.get("sl") is not None:
                            cap_rec_active = False
                except Exception:
                    pass

                if self._time_exit_enabled and cap_rec_active:
                    min_pnl_rec = float(cap_rec.get("min_pnl_pct", 0.0002) or 0.0002)
                    max_age_sec_rec = float(cap_rec.get("max_age_sec", 0.0) or 0.0)
                    
                    # Mandatory time-only exit once max age is reached
                    if max_age_sec_rec > 0 and age_sec >= max_age_sec_rec:
                        to_close.append((
                            symbol,
                            f"Capital recovery exit: age={int(age_sec)}s (max={int(max_age_sec_rec)}s)"
                        ))
                        continue

                current_price = prices.get(symbol)
                if current_price is None:
                    current_price = await self._current_price(symbol)
                if current_price is None:
                    self.logger.debug(
                        "[%s] TPSL skip: missing current price (cannot evaluate TP/SL or pnl-based exits).",
                        symbol
                    )
                    _log_skip(symbol, qty, None, None, None, "missing_current_price")
                    continue

                # Prefer stored TP/SL, compute if missing
                tp = tr.get("tp"); sl = tr.get("sl")
                if tp is None or sl is None:
                    tp, sl = self.calculate_tp_sl(symbol, float(entry_price))
                    if tp is None or sl is None:
                        _log_skip(symbol, qty, float(current_price) if current_price is not None else None, tp, sl, "missing_tp_sl")
                        continue
                    try:
                        tr["tp"] = tp; tr["sl"] = sl  # cache for next pass
                        self.shared_state.open_trades[symbol] = tr
                    except Exception:
                        pass

                cp = float(current_price)
                
                # Tier B TTL check (Phase A Frequency Engineering)
                tier = tr.get("tier")
                pnl_pct = (cp - entry_price) / entry_price if position == "long" else (entry_price - cp) / entry_price

                # Dust exit rule: if value is tiny and held too long, force exit to free capital.
                if self._time_exit_enabled:
                    dust_usdt_threshold = float(getattr(self.config, "TPSL_DUST_EXIT_USDT", 4.0) or 4.0)
                    max_hold_min = float(getattr(self.config, "TPSL_DUST_MAX_HOLD_MINUTES", 15.0) or 15.0)
                    if dust_usdt_threshold > 0 and max_hold_min > 0:
                        position_value = float(qty) * cp
                        if position_value > 0 and position_value < dust_usdt_threshold and age_sec >= (max_hold_min * 60.0):
                            self.logger.warning(
                                "[TPSL:DustExit] %s value=%.2f < %.2f and age=%.0fs >= %.0fs. Forcing exit.",
                                symbol,
                                position_value,
                                dust_usdt_threshold,
                                age_sec,
                                max_hold_min * 60.0,
                            )
                            to_close.append((
                                symbol,
                                f"Dust exit: value={position_value:.2f} < {dust_usdt_threshold:.2f} age={int(age_sec)}s"
                            ))
                            continue

                # --- Fee-aware Stagnation Purge (Phase A Frequency Engineering) ---
                # Rule: If PnL < (2 * fees) AND age > 45m → Exit to recycle capital.
                # This prevents "limbo" trades where capital is trapped in slow-moving assets.
                stagnation_min = float(getattr(self.config, "STAGNATION_EXIT_MINUTES", 45.0))
                rt_fee_pct = ((fee_bps(self.shared_state, "taker") or 10.0) * 2.0) / 10000.0
                
                if self._time_exit_enabled and age_sec >= (stagnation_min * 60.0):
                    stagnation_pnl_threshold = rt_fee_pct * 2.0
                    if pnl_pct < stagnation_pnl_threshold:
                         self.logger.warning(
                             "[TPSL:StagnationPurge] %s age=%dm >= %dm, pnl=%.2f%% < threshold=%.2f%%. Purging.",
                             symbol, int(age_sec/60), int(stagnation_min), pnl_pct*100, stagnation_pnl_threshold*100
                         )
                         reason = f"Stagnation purge: age={int(age_sec/60)}m pnl={pnl_pct*100:.2f}%"
                         if self._passes_profit_gate(pnl_pct, reason):
                             to_close.append((symbol, reason))
                         else:
                             self.logger.info("[%s] Profit gate blocked SELL (%s).", symbol, reason)
                         continue

                # 🔑 Rule 1: Fee-Adjusted Breakeven Exit
                # If PnL <= fees AND age >= 20m -> EXIT IMMEDIATELY
                # This prevents fee-dominated positions and capital deadlocks.
                min_hold_breakeven = float(getattr(self.config, "BREAKEVEN_EXIT_MINUTES", 20.0))
                if self._time_exit_enabled and age_sec >= (min_hold_breakeven * 60.0):
                    if pnl_pct <= rt_fee_pct:
                         self.logger.warning(
                             "[TPSL:Breakeven] %s age=%dm >= %dm, pnl=%.2f%% <= fees=%.2f%%. Breakeven exit.",
                             symbol, int(age_sec/60), int(min_hold_breakeven), pnl_pct*100, rt_fee_pct*100
                         )
                         reason = f"Fee-adjusted breakeven: age={int(age_sec/60)}m pnl={pnl_pct*100:.2f}%"
                         if self._passes_profit_gate(pnl_pct, reason):
                             to_close.append((symbol, reason))
                         else:
                             self.logger.info("[%s] Profit gate blocked SELL (%s).", symbol, reason)
                         continue

                # 3. Capital recovery mode: optional early exit optimizer (small profit)
                if cap_rec_active and min_pnl_rec is not None:
                    if pnl_pct >= min_pnl_rec:
                        reason = f"Capital recovery exit: pnl={pnl_pct*100:.2f}% age={int(age_sec or 0)}s"
                        if self._passes_profit_gate(pnl_pct, reason):
                            to_close.append((symbol, reason))
                        else:
                            self.logger.info("[%s] Profit/excursion gate blocked SELL (%s).", symbol, reason)
                        continue

                # 4. Micro-profit cycle: time exit after first trade succeeds
                try:
                    metrics = getattr(self.shared_state, "metrics", {}) or {}
                    trades_executed = int(metrics.get("total_trades_executed", 0) or 0)
                except Exception:
                    trades_executed = 0

                if self._time_exit_enabled and trades_executed >= 1 and bool(getattr(self.config, "MICRO_PROFIT_CYCLE_ENABLED", False)):
                    min_age_h = float(getattr(self.config, "MICRO_PROFIT_CYCLE_MIN_AGE_HOURS", 2.0) or 2.0)
                    min_pnl_cycle = float(getattr(self.config, "MICRO_PROFIT_CYCLE_MIN_PNL_PCT", 0.001) or 0.001)
                    if age_sec >= (min_age_h * 3600) and pnl_pct >= min_pnl_cycle:
                        reason = f"Micro-cycle time exit: age={int(age_sec)}s pnl={pnl_pct*100:.2f}%"
                        if self._passes_profit_gate(pnl_pct, reason):
                            to_close.append((symbol, reason))
                        else:
                            self.logger.info("[%s] Profit/excursion gate blocked SELL (%s).", symbol, reason)
                        continue

                if self._time_exit_enabled and tier == "B":
                    ttl_sec = tr.get("ttl_sec", 300)  # Default 5 minutes
                    if age_sec > ttl_sec:
                            # Grace period: only for winners (PnL > 1%)
                            if pnl_pct > 0.01 and age_sec < (ttl_sec + 120):
                                self.logger.info("[%s] Tier-B TTL GRACE: PnL=%.2f%%, age=%ds, extending", 
                                               symbol, pnl_pct*100, int(age_sec))
                                continue  # Skip closing - let position run
                            else:
                                # TTL expired and no grace → force close
                                to_close.append((symbol, f"Tier-B FORCED TTL: age={age_sec:.0f}s > {ttl_sec}s"))
                                continue  # Skip TP/SL check, TTL takes priority
                
                # --- Wealth Guard: Support for Trailing SL and Ratcheting ---
                use_trailing = tr.get("_use_trailing", False)

                self.logger.info(
                    "[TPSL:eval] %s price=%.4f tp=%.4f sl=%.4f",
                    symbol,
                    cp,
                    float(tp),
                    float(sl),
                )
                
                if position == "long":
                    # Update Trailing SL if price moves up
                    if use_trailing and atr and atr > 0:
                        # Trail at 1.5x ATR distance
                        trail_dist = atr * float(getattr(self.config, "TRAILING_ATR_MULT", 1.5))
                        new_sl = round(cp - trail_dist, 4)
                        if new_sl > float(sl):
                             self.logger.info("[%s] Trailing SL up: %.4f -> %.4f", symbol, float(sl), new_sl)
                             tr["sl"] = new_sl
                             sl = new_sl
                             self.shared_state.open_trades[symbol] = tr

                    if cp >= float(tp):
                        reason = "TP Hit"
                        if (
                            self._passes_net_exit_gate(pnl_pct)
                            and self._passes_profit_gate(pnl_pct, reason)
                            and await self._passes_excursion_gate(symbol, entry_price, cp, atr, reason)
                        ):
                            to_close.append((symbol, reason))
                        else:
                            self.logger.info("[%s] Profit gate blocked SELL (%s).", symbol, reason)
                    elif cp <= float(sl):
                        to_close.append((symbol, "SL Hit"))
                elif position == "short":
                    # Update Trailing SL if price moves down
                    if use_trailing and atr and atr > 0:
                        trail_dist = atr * float(getattr(self.config, "TRAILING_ATR_MULT", 1.5))
                        new_sl = round(cp + trail_dist, 4)
                        if new_sl < float(sl):
                             self.logger.info("[%s] Trailing SL down: %.4f -> %.4f", symbol, float(sl), new_sl)
                             tr["sl"] = new_sl
                             sl = new_sl
                             self.shared_state.open_trades[symbol] = tr

                    if cp <= float(tp):
                        reason = "TP Hit (short)"
                        if (
                            self._passes_net_exit_gate(pnl_pct)
                            and self._passes_profit_gate(pnl_pct, reason)
                            and await self._passes_excursion_gate(symbol, entry_price, cp, atr, reason)
                        ):
                            to_close.append((symbol, reason))
                        else:
                            self.logger.info("[%s] Profit gate blocked SELL (%s).", symbol, reason)
                    elif cp >= float(sl):
                        to_close.append((symbol, "SL Hit (short)"))

            if to_close:
                sem = asyncio.Semaphore(self._close_concurrency)

                async def _close(sym: str, reason: str):
                    async with sem:
                        self._last_close_attempt[sym] = self._mono()
                        self.logger.info("[%s] Closing due to %s", sym, reason)
                        # [FIX #9] CRITICAL: Mark as liquidation to bypass ALL guards
                        # TP/SL exits are liquidation (risk management), NOT trading decisions.
                        # They must execute regardless of: capital, throughput, min-notional, etc.
                        reason_u = str(reason).upper()
                        if "TP" in reason_u:
                            close_reason = "TP_HIT"
                        elif "SL" in reason_u:
                            close_reason = "SL_HIT"
                        else:
                            close_reason = "TPSL_EXIT"

                        try:
                            total_position_qty = 0.0
                            pos = getattr(self.shared_state, "positions", {}).get(sym, {}) if hasattr(self.shared_state, "positions") else {}
                            if isinstance(pos, dict):
                                total_position_qty = float(pos.get("quantity", 0.0) or 0.0)
                            if total_position_qty <= 0 and hasattr(self.shared_state, "get_position_qty"):
                                total_position_qty = float(self.shared_state.get_position_qty(sym) or 0.0)
                            min_notional = await self._get_min_notional(sym)
                            if min_notional > 0 and total_position_qty < min_notional:
                                self.logger.warning(
                                    "[TPSL] Proceeding with close %s even though qty=%.8f < min_notional=%.8f",
                                    sym,
                                    total_position_qty,
                                    min_notional,
                                )
                        except Exception:
                            pass
                        try:
                            ot = getattr(self.shared_state, "open_trades", {}) or {}
                            tr = ot.get(sym, {}) if isinstance(ot, dict) else {}
                            entry_price = float(tr.get("entry_price", 0.0) or 0.0) or None
                            position_qty = float(tr.get("quantity", 0.0) or 0.0) or None
                            tp = float(tr.get("tp", 0.0) or 0.0) or None
                            sl = float(tr.get("sl", 0.0) or 0.0) or None
                            tier = tr.get("tier")
                            method = tr.get("tp_sl_method") or self._last_tp_sl_method.get(sym)
                            position = tr.get("position") or "long"
                            current_price = None
                            try:
                                p = self.shared_state.latest_prices.get(sym)
                                if p:
                                    current_price = float(p)
                            except Exception:
                                current_price = None
                            if current_price is None:
                                current_price = await self._current_price(sym)
                            pnl_pct = None
                            if entry_price and current_price:
                                if position == "long":
                                    pnl_pct = (current_price - entry_price) / entry_price
                                else:
                                    pnl_pct = (entry_price - current_price) / entry_price
                            await self._emit_tp_sl_plan(
                                sym,
                                reason,
                                position,
                                position_qty,
                                entry_price,
                                current_price,
                                tp,
                                sl,
                                pnl_pct,
                                tier,
                                method,
                            )
                        except Exception:
                            pass
                        res = await self.execution_manager.close_position(
                            symbol=sym,
                            reason=close_reason,
                            force_finalize=True,
                            tag="tp_sl",
                        )
                        try:
                            status = str(res.get("status", "")).lower() if isinstance(res, dict) else ""
                            ok = bool(res.get("ok")) if isinstance(res, dict) else False
                            if ok or status in {"placed", "executed", "filled", "partially_filled"}:
                                if close_reason == "TP_HIT":
                                    code = "TP"
                                elif close_reason == "SL_HIT":
                                    code = "SL"
                                else:
                                    code = "TPSL_EXIT"
                                await post_exit_bookkeeping(
                                    self.shared_state,
                                    self.config,
                                    self.logger,
                                    sym,
                                    code,
                                    "tp_sl",
                                )
                                await self._force_close_all_open_lots(sym, close_reason)
                        except Exception:
                            pass

                # [FIX #10] Timeout protection: prevent TPSL from stalling on ExchangeClient hangs
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*(_close(s, r) for s, r in to_close)),
                        timeout=15  # Max 15 seconds for all closes to complete
                    )
                except asyncio.TimeoutError:
                    self.logger.error("[TIMEOUT] TP/SL close operations exceeded 15s timeout")
                    await self.shared_state.update_system_health(
                        component="TPSLEngine",
                        status="Warning",
                        message=f"TP/SL close timeout: {len(to_close)} positions may have been skipped"
                    )

            uh = getattr(self.shared_state, "update_system_health", None)
            if callable(uh):
                res = uh(
                    component="TPSLEngine",
                    status="Operational",
                    message=f"TP/SL evaluation OK. Closed={len(to_close)}"
                )
                if iscoroutine(res):
                    await res
            # P9 HealthStatus event
            _emit_health(self.shared_state, "Running", f"Evaluation OK. Closed={len(to_close)}")

        except Exception as e:
            self.logger.error("TPSLEngine.check_orders failed: %s", e)
            uh = getattr(self.shared_state, "update_system_health", None)
            if callable(uh):
                res = uh(
                    component="TPSLEngine",
                    status="Error",
                    message=f"Exception during TP/SL evaluation: {e}"
                )
                if iscoroutine(res):
                    await res
            _emit_health(self.shared_state, "Error", f"Evaluation failed: {e}")

    async def run(self):
        """
        Continuous, drift-free monitoring loop.
        """
        # Wait gates using events if present (non-fatal if missing)
        async def _wait_gate(ev, name: str, timeout: float) -> None:
            try:
                await asyncio.wait_for(ev.wait(), timeout=timeout)
                self.logger.info("[TPSL] Gate open: %s", name)
            except asyncio.TimeoutError:
                self.logger.warning("[TPSL] Gate timeout: %s — proceeding anyway", name)
            except Exception:
                self.logger.debug("[TPSL] Gate error: %s", name, exc_info=True)

        gate_timeout = float(getattr(self.config, "TPSL_GATE_TIMEOUT_SEC", 20.0) or 20.0)
        self.logger.info("[TPSL] Waiting for startup gates")
        try:
            ev_acc = getattr(self.shared_state, "get_accepted_symbols_ready_event", None)
            if callable(ev_acc):
                await _wait_gate(ev_acc(), "AcceptedSymbolsReady", gate_timeout)
        except Exception:
            pass
        try:
            ev_md = getattr(self.shared_state, "get_market_data_ready_event", None)
            if callable(ev_md):
                await _wait_gate(ev_md(), "MarketDataReady", gate_timeout)
        except Exception:
            pass
        self.logger.info("[TPSL] Startup gates cleared")
        self.logger.error("[TPSL:DIAG] reached pre-loop section")

        await self._safe_status_update("Healthy", "TPSLEngine loop initialized.")
        self.logger.info("🚀 TPSLEngine started TP/SL monitoring loop.")
        # Mirror to component-status mirror (optional)
        try:
            await self._safe_status_update("OK", "Running normally.")
        except Exception:
            pass

        interval = max(0.5, float(self._interval))
        next_tick = self._mono()

        try:
            while True:
                self.logger.error("[TPSL:DIAG] entered monitoring loop")
                self.logger.info("[TPSL:tick] alive")
                self.status = "RUNNING"
                self.last_tick = time.time()
                try:
                    await self.shared_state.update_timestamp("TPSLEngine")
                except Exception:
                    pass

                await self.check_orders()

                try:
                    await self._safe_status_update("OK", "Running normally.")
                    _emit_health(self.shared_state, "Running", "Monitoring active")
                except Exception:
                    pass

                now = self._mono()
                if now < next_tick:
                    await asyncio.sleep(next_tick - now)
                next_tick += interval
        except asyncio.CancelledError:
            self.logger.info("TPSLEngine loop cancelled.")
            raise



# ===== P9 Spec Helpers (added) =====
import asyncio, time
from datetime import datetime

def _iso_now():
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def _emit_health(ss, status: str, message: str):
    try:
        if ss and hasattr(ss, "emit_event"):
            payload = {
                "component": "TPSLEngine",
                "status": status,
                "message": message,
                "timestamp": _iso_now()
            }
            # Fire and forget safely
            try:
                loop = asyncio.get_running_loop()
                if loop.is_running():
                    # If we are in an async spot, schedule it
                    res = ss.emit_event("HealthStatus", payload)
                    if asyncio.iscoroutine(res):
                        loop.create_task(res)
            except RuntimeError:
                # Fallback for sync or no-loop spots
                pass
    except Exception:
        pass

def _norm_exec_order_tp_sl(x):
    def g(obj, k, d=None):
        return getattr(obj, k, None) if hasattr(obj, k) else (obj.get(k, d) if isinstance(obj, dict) else d)
    symbol = g(x, "symbol")
    side = g(x, "side")
    if not symbol or not side:
        return None
    quantity = g(x, "quantity") or g(x, "qty") or g(x, "qty_hint")
    quote_hint = g(x, "quote_hint")
    eo = {"symbol": symbol, "side": side, "source": "tp_sl", "tag": "tp_sl"}
    if quantity is not None:
        eo["quantity"] = quantity
    if quote_hint is not None:
        eo["quote_hint"] = quote_hint
    return eo



# ===== P9 TP/SL Normalization Wrappers (added) =====
def _wrap_tp_sl_outputs(cls):
    producer_names = ["produce_orders", "collect_orders", "check_triggers", "generate_closures"]
    for name in producer_names:
        if hasattr(cls, name) and not hasattr(cls, f"_{name}_raw"):
            setattr(cls, f"_{name}_raw", getattr(cls, name))
            def _make_wrapper(mname):
                async def _wrapped(self, *a, **kw):
                    res = getattr(cls, f"_{mname}_raw")(self, *a, **kw)
                    out = await res if asyncio.iscoroutine(res) else res
                    arr = out if isinstance(out, (list, tuple)) else ([out] if out is not None else [])
                    norm = []
                    for x in arr:
                        eo = _norm_exec_order_tp_sl(x)
                        if eo:
                            norm.append(eo)
                    _emit_health(getattr(self, "shared_state", None), "Running", f"tp_sl orders={len(norm)}")
                    return norm
                return _wrapped
            setattr(cls, name, _make_wrapper(name))
    if not hasattr(cls, "warmup"):
        async def warmup(self):
            _emit_health(getattr(self, "shared_state", None), "Running", "warmup ok")
        cls.warmup = warmup
    if not hasattr(cls, "health"):
        def health(self):
            return {"component": "TPSLEngine", "status": "Running", "timestamp": _iso_now()}
        cls.health = health
    return cls


# Apply P9 TP/SL wrapper
TPSLEngine = _wrap_tp_sl_outputs(TPSLEngine)
