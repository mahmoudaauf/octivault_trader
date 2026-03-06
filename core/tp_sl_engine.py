import logging
import asyncio
import time
import os
from datetime import datetime
from inspect import iscoroutine
from typing import Optional, List, Dict, Any, Tuple
from math import fsum, sqrt
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
        self.trade_journal = None  # injected by AppContext
        self.session_id: str = ""  # injected by AppContext

        # RACE CONDITION FIX #4: Per-symbol locking to prevent concurrent closes
        self._symbol_close_locks: Dict[str, asyncio.Lock] = {}
        self._symbol_close_locks_lock = asyncio.Lock()

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
        self._rv_lookback = int(getattr(self.config, "TPSL_RV_LOOKBACK", 20) or 20)
        self._vol_low_pct = float(getattr(self.config, "TPSL_VOL_LOW_ATR_PCT", 0.0045) or 0.0045)
        self._vol_high_pct = float(getattr(self.config, "TPSL_VOL_HIGH_ATR_PCT", 0.0150) or 0.0150)
        self._vol_target_atr_pct = float(getattr(self.config, "TPSL_VOL_TARGET_ATR_PCT", 0.0090) or 0.0090)
        self._rr_min = float(getattr(self.config, "TPSL_DYNAMIC_RR_MIN", 1.35) or 1.35)
        self._rr_max = float(getattr(self.config, "TPSL_DYNAMIC_RR_MAX", 2.60) or 2.60)
        self._spread_adaptive_enabled = bool(
            getattr(self.config, "TPSL_SPREAD_ADAPTIVE_ENABLED", True)
        )
        self._spread_tight_bps = float(getattr(self.config, "TPSL_SPREAD_TIGHT_BPS", 6.0) or 6.0)
        self._spread_high_bps = float(getattr(self.config, "TPSL_SPREAD_HIGH_BPS", 18.0) or 18.0)
        self._spread_extreme_bps = float(getattr(self.config, "TPSL_SPREAD_EXTREME_BPS", 45.0) or 45.0)
        self._spread_rr_bonus_max = float(getattr(self.config, "TPSL_SPREAD_RR_BONUS_MAX", 0.18) or 0.18)
        self._spread_rr_discount_max = float(getattr(self.config, "TPSL_SPREAD_RR_DISCOUNT_MAX", 0.06) or 0.06)
        self._spread_tp_floor_mult = float(getattr(self.config, "TPSL_SPREAD_TP_FLOOR_MULT", 2.0) or 2.0)
        self._asymmetric_tp_enabled = bool(getattr(self.config, "TPSL_ASYMMETRIC_TP_ENABLED", True))
        self._asymmetric_tp_trend_bonus = float(getattr(self.config, "TPSL_ASYM_TP_TREND_BONUS", 0.12) or 0.12)
        self._asymmetric_tp_high_vol_bonus = float(getattr(self.config, "TPSL_ASYM_TP_HIGH_VOL_BONUS", 0.08) or 0.08)
        self._asymmetric_tp_chop_discount = float(getattr(self.config, "TPSL_ASYM_TP_CHOP_DISCOUNT", 0.08) or 0.08)
        self._asymmetric_tp_sentiment_weight = float(getattr(self.config, "TPSL_ASYM_TP_SENTIMENT_WEIGHT", 0.06) or 0.06)
        self._asymmetric_tp_phase_gap_weight = float(getattr(self.config, "TPSL_ASYM_TP_PHASE_GAP_WEIGHT", 0.20) or 0.20)
        self._asymmetric_tp_phase_bonus_cap = float(getattr(self.config, "TPSL_ASYM_TP_PHASE_BONUS_CAP", 0.12) or 0.12)
        self._asymmetric_tp_min_bias = float(getattr(self.config, "TPSL_ASYM_TP_MIN_BIAS", 0.92) or 0.92)
        self._asymmetric_tp_max_bias = float(getattr(self.config, "TPSL_ASYM_TP_MAX_BIAS", 1.35) or 1.35)
        self._asymmetric_tp_tier_b_cap = float(getattr(self.config, "TPSL_ASYM_TP_TIER_B_CAP", 1.08) or 1.08)
        self._enforce_execution_manager_only = bool(
            getattr(self.config, "TPSL_ENFORCE_EXECUTION_MANAGER_ONLY", True)
        )
        self._dynamic_trailing_mult: Dict[str, float] = {}
        self._snowball_asymmetry_enabled = bool(
            getattr(self.config, "TPSL_SNOWBALL_ASYMMETRY_ENABLED", True)
        )
        self._snowball_phase_profiles = getattr(
            self.config,
            "COMPOUNDING_TPSL_PHASE_PROFILES",
            {
                "PHASE_1_SEED": {"tp_mult": 1.20, "sl_mult": 1.00, "rr_bonus": 0.04},
                "PHASE_2_TRACTION": {"tp_mult": 1.40, "sl_mult": 0.95, "rr_bonus": 0.12},
                "PHASE_3_ACCELERATE": {"tp_mult": 1.60, "sl_mult": 0.90, "rr_bonus": 0.22},
                # Phase 4 shifts to capital defense: reduced TP aggression, tighter SL.
                "PHASE_4_SNOWBALL": {"tp_mult": 1.30, "sl_mult": 0.75, "rr_bonus": 0.10},
            },
        ) or {}
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
            self.logger.error(
                "[TPSLEngine] Auto-arm failed on startup — open positions may have no stop-loss",
                exc_info=True,
            )
            
        self._task = asyncio.create_task(self.run())
        await self._safe_status_update("Initialized", "Ready")

    async def _auto_arm_existing_trades(self) -> None:
        """Ensure TP/SL is set for existing open positions at startup."""
        # Canonical architecture: positions are the only source of truth
        if getattr(self.shared_state, "trading_mode", "") == "shadow":
            positions = getattr(self.shared_state, "virtual_positions", {}) or {}
        else:
            positions = getattr(self.shared_state, "positions", {}) or {}

        # Derive open trades from positions
        open_trades = {
            sym: pos
            for sym, pos in positions.items()
            if float(pos.get("position_qty", 0) or 0) > 0
        }
        symbols = set(open_trades.keys()) | set(positions.keys())

        for symbol in symbols:
            tr = open_trades.get(symbol, {}) if isinstance(open_trades, dict) else {}
            if not isinstance(tr, dict):
                tr = {}
            pos = positions.get(symbol, {}) if isinstance(positions, dict) else {}
            gate_ref = tr if tr else pos
            try:
                is_open_sig, value_usdt, floor_usdt = self.shared_state.classify_position_snapshot(symbol, gate_ref)
            except Exception:
                is_open_sig, value_usdt, floor_usdt = True, 0.0, 0.0
            if not is_open_sig:
                try:
                    self.shared_state.open_trades.pop(symbol, None)
                except Exception:
                    pass
                self.logger.info(
                    "[TPSL:auto_arm] %s skipped below significant floor (value=%.6f floor=%.6f)",
                    symbol,
                    float(value_usdt or 0.0),
                    float(floor_usdt or 0.0),
                )
                continue
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
                # Temporary debug: surface final TP/SL values for troubleshooting
                tp_price = float(tp or 0.0)
                sl_price = float(sl or 0.0)
                self.logger.debug(
                    "[TPSL:ARM_DEBUG] %s FINAL TP=%.6f SL=%.6f entry=%.6f",
                    symbol, tp_price, sl_price, entry_price
                )
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
            last_seen = getattr(self.shared_state, "component_last_seen", None)
            if isinstance(last_seen, dict):
                last_seen["TPSLEngine"] = ts
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

    def _compute_realized_vol_pct(self, symbol: str, lookback: Optional[int] = None) -> float:
        """
        Close-to-close realized volatility as std-dev of returns.
        Returns volatility as a percentage ratio (e.g., 0.01 == 1%).
        """
        try:
            lb = int(lookback or self._rv_lookback or 20)
            candles = self._pick_candles(symbol)
            if len(candles) < (lb + 1):
                return 0.0
            closes = [float(c.get("close", 0.0) or 0.0) for c in candles[-(lb + 1):]]
            if any(c <= 0 for c in closes):
                return 0.0
            returns = []
            for i in range(1, len(closes)):
                prev_c = closes[i - 1]
                curr_c = closes[i]
                returns.append((curr_c / prev_c) - 1.0)
            if len(returns) < 2:
                return 0.0
            mean_r = fsum(returns) / float(len(returns))
            var = fsum((r - mean_r) ** 2 for r in returns) / float(max(1, len(returns) - 1))
            rv = sqrt(max(var, 0.0))
            return float(rv if rv > 0 else 0.0)
        except Exception:
            return 0.0

    def _build_volatility_profile(self, symbol: str, entry_price: float, atr: float, tier: Optional[str] = None) -> Dict[str, Any]:
        """
        Build a normalized volatility profile used by TP/SL and trailing logic.
        """
        atr_pct = (float(atr) / float(entry_price)) if entry_price > 0 else 0.0
        rv_pct = self._compute_realized_vol_pct(symbol)

        low = max(1e-6, float(self._vol_low_pct))
        high = max(low + 1e-6, float(self._vol_high_pct))
        target = max(low, float(self._vol_target_atr_pct))

        # volatility_score in [-1, +1]
        midpoint = (low + high) * 0.5
        half_band = max((high - low) * 0.5, 1e-6)
        vol_score = max(-1.0, min(1.0, (atr_pct - midpoint) / half_band))

        # volatility pressure around target ATR%
        vol_pressure = max(-0.6, min(1.2, (atr_pct / max(target, 1e-6)) - 1.0))

        # Optional external regime from shared_state; fallback to inferred regime.
        ext_regime = str((getattr(self.shared_state, "volatility_state", {}) or {}).get(symbol, "") or "").lower()
        if ext_regime in {"trend", "uptrend", "downtrend", "high_vol", "high", "sideways", "chop"}:
            regime = ext_regime
        else:
            if atr_pct >= high:
                regime = "high_vol"
            elif atr_pct <= low:
                regime = "sideways"
            else:
                regime = "trend"

        # Tier-B is designed for faster turns; keep profile but expose tighter behaviour hint.
        tier_b = str(tier or "").upper() == "B"

        return {
            "atr_pct": float(atr_pct),
            "rv_pct": float(rv_pct),
            "vol_score": float(vol_score),
            "vol_pressure": float(vol_pressure),
            "regime": regime,
            "tier_b": tier_b,
        }

    def _estimate_spread(self, symbol: str, ref_price: float) -> Tuple[float, float]:
        """
        Estimate current spread using best in-memory sources.
        Returns (spread_abs, spread_pct_of_price).
        """
        bid = 0.0
        ask = 0.0
        sym = str(symbol or "").upper().replace("/", "")

        try:
            ba_map = getattr(self.shared_state, "best_bid_ask", None)
            if isinstance(ba_map, dict):
                rec = ba_map.get(sym) or ba_map.get(symbol)
                if isinstance(rec, dict):
                    bid = float(rec.get("bid", 0.0) or rec.get("best_bid", 0.0) or 0.0)
                    ask = float(rec.get("ask", 0.0) or rec.get("best_ask", 0.0) or 0.0)
                elif isinstance(rec, (list, tuple)) and len(rec) >= 2:
                    bid = float(rec[0] or 0.0)
                    ask = float(rec[1] or 0.0)
        except Exception:
            pass

        if bid <= 0 or ask <= 0:
            try:
                md = getattr(self.shared_state, "market_data", {}) or {}
                sym_md = md.get(sym, {}) if isinstance(md, dict) else {}
                if isinstance(sym_md, dict):
                    bid = float(sym_md.get("bid", 0.0) or sym_md.get("best_bid", 0.0) or bid)
                    ask = float(sym_md.get("ask", 0.0) or sym_md.get("best_ask", 0.0) or ask)
            except Exception:
                pass

        spread_abs = 0.0
        if ask > 0 and bid > 0 and ask >= bid:
            spread_abs = abs(ask - bid)

        px = float(ref_price or 0.0)
        if px <= 0 and ask > 0 and bid > 0:
            px = (ask + bid) * 0.5

        spread_pct = (spread_abs / px) if (spread_abs > 0 and px > 0) else 0.0
        return float(spread_abs), float(spread_pct)

    def _resolve_adaptive_rr(self, base_rr: float, profile: Dict[str, Any], sentiment: float) -> float:
        rr = float(base_rr)
        regime = str(profile.get("regime", "sideways")).lower()
        vol_score = float(profile.get("vol_score", 0.0))

        if regime in {"trend", "uptrend", "downtrend"}:
            rr += 0.15
        elif regime in {"high_vol", "high"}:
            rr += 0.20
        elif regime in {"sideways", "chop"}:
            rr -= 0.08

        # Positive sentiment supports wider TP, negative sentiment biases capital defense.
        rr += max(-0.15, min(0.15, sentiment * 0.15))
        # High volatility should expand RR (with bounds), low volatility should compress RR.
        rr += max(-0.10, min(0.18, vol_score * 0.12))
        rr = max(float(self._rr_min), min(float(self._rr_max), rr))
        return rr

    def _resolve_asymmetric_tp_bias(
        self,
        profile: Dict[str, Any],
        sentiment: float,
        phase_profile: Dict[str, float],
        tier: Optional[str] = None,
    ) -> float:
        """
        Explicit asymmetric TP layer:
        - Expand TP in trend/high-vol regimes with constructive sentiment.
        - Tighten TP in chop while preserving downside defense via unchanged SL.
        """
        if not self._asymmetric_tp_enabled:
            return 1.0

        regime = str(profile.get("regime", "sideways")).lower()
        vol_score = float(profile.get("vol_score", 0.0) or 0.0)
        bias = 1.0

        if regime in {"trend", "uptrend", "downtrend"}:
            bias += max(0.0, float(self._asymmetric_tp_trend_bonus)) * max(0.0, 1.0 + (vol_score * 0.25))
        elif regime in {"high_vol", "high"}:
            bias += max(0.0, float(self._asymmetric_tp_high_vol_bonus)) * max(0.6, 1.0 + (vol_score * 0.30))
        elif regime in {"sideways", "chop"}:
            bias -= max(0.0, float(self._asymmetric_tp_chop_discount)) * max(0.25, 1.0 - max(0.0, vol_score))

        sent_w = max(0.0, float(self._asymmetric_tp_sentiment_weight))
        bias += max(-sent_w, min(sent_w, float(sentiment or 0.0) * sent_w))

        phase_tp = float(phase_profile.get("tp_mult", 1.0) or 1.0)
        phase_sl = float(phase_profile.get("sl_mult", 1.0) or 1.0)
        phase_gap = max(0.0, phase_tp - phase_sl)
        phase_bonus = min(
            max(0.0, float(self._asymmetric_tp_phase_bonus_cap)),
            phase_gap * max(0.0, float(self._asymmetric_tp_phase_gap_weight)),
        )
        bias += phase_bonus

        if str(tier or "").upper() == "B":
            bias = min(bias, float(self._asymmetric_tp_tier_b_cap))

        low = min(float(self._asymmetric_tp_min_bias), float(self._asymmetric_tp_max_bias))
        high = max(float(self._asymmetric_tp_min_bias), float(self._asymmetric_tp_max_bias))
        return max(low, min(high, float(bias)))

    def _get_compounding_phase_profile(self) -> Dict[str, float]:
        phase_name = "PHASE_1_SEED"
        active = True
        try:
            dyn = getattr(self.shared_state, "dynamic_config", {}) or {}
            phase_name = str(
                dyn.get("compounding_phase", dyn.get("COMPOUNDING_PHASE", phase_name)) or phase_name
            ).upper()
            active = bool(dyn.get("COMPOUNDING_GROWTH_ACTIVE", True))
        except Exception:
            phase_name = "PHASE_1_SEED"
            active = True

        profiles = self._snowball_phase_profiles if isinstance(self._snowball_phase_profiles, dict) else {}
        profile = dict(profiles.get(phase_name) or profiles.get("PHASE_1_SEED") or {})

        # Optional explicit per-phase TP/SL maps from config.
        try:
            tp_map = getattr(self.config, "TP_PHASE_MULTIPLIERS", {}) or {}
            if isinstance(tp_map, dict) and phase_name in tp_map:
                profile["tp_mult"] = float(tp_map.get(phase_name))
        except Exception:
            pass
        try:
            sl_map = getattr(self.config, "SL_PHASE_MULTIPLIERS", {}) or {}
            if isinstance(sl_map, dict) and phase_name in sl_map:
                profile["sl_mult"] = float(sl_map.get(phase_name))
        except Exception:
            pass

        if not active:
            profile = {"tp_mult": 1.0, "sl_mult": 1.0, "rr_bonus": 0.0}
            phase_name = "PHASE_1_SEED"
        return {
            "name": phase_name,
            "tp_mult": float(profile.get("tp_mult", 1.0) or 1.0),
            "sl_mult": float(profile.get("sl_mult", 1.0) or 1.0),
            "rr_bonus": float(profile.get("rr_bonus", 0.0) or 0.0),
        }

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
            return False  # No data yet (startup warmup) — not stale, just not ready
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

        NEW: Risk-based position sizing integrated with TP/SL calculation
        - Uses SL distance to determine optimal position size
        - Ensures consistent risk across all trades

        Returns (tp, sl).
        """
        ok, reason = self._pre_activation_guard(symbol, entry_price, quantity)
        if not ok:
            self.logger.info("[TPSL:PreGuard] %s skip set_initial_tp_sl: %s", symbol, reason)
            return None, None

        tp, sl = self.calculate_tp_sl(symbol, entry_price, tier=tier)

        # NEW: Calculate risk-based position size using SL distance
        if tier == "DUST_RECOVERY":
            # For dust healing, use deficit + small buffer, skip risk sizing
            deficit = getattr(self.shared_state, "dust_healing_deficit", {}).get(symbol, 0.0)
            small_buffer = float(getattr(self.config, "DUST_HEALING_BUFFER_USDT", 0.5))
            planned_quote = deficit + small_buffer
            self.shared_state.risk_based_quote = getattr(self.shared_state, "risk_based_quote", {})
            self.shared_state.risk_based_quote[symbol] = planned_quote
        elif bool(getattr(self.shared_state, "dust_operation_symbols", {}).get(symbol)):
            # If this is a dust healing operation, preserve the exact deficit sizing.
            # Do not apply risk-based sizing that would override the dust healing amount.
            self.logger.info("[TPSL:DustOperation] %s is dust healing operation, preserving exact deficit sizing", symbol)
        elif sl is not None:
            risk_based_quote = self.calculate_risk_based_position_size(symbol, entry_price, sl, tier=tier)
            # Store for ExecutionManager to use
            self.shared_state.risk_based_quote = getattr(self.shared_state, "risk_based_quote", {})
            self.shared_state.risk_based_quote[symbol] = risk_based_quote

        try:
            gate_ref = {
                "quantity": float(quantity or 0.0),
                "entry_price": float(entry_price or 0.0),
                "avg_price": float(entry_price or 0.0),
                "value_usdt": float(quantity or 0.0) * float(entry_price or 0.0),
            }
            is_open_sig, value_usdt, floor_usdt = self.shared_state.classify_position_snapshot(
                symbol,
                gate_ref,
                price_hint=float(entry_price or 0.0),
            )
            if not is_open_sig:
                self.shared_state.open_trades.pop(symbol, None)
                self.logger.info(
                    "[TPSL] %s TP/SL not armed below significant floor (value=%.6f floor=%.6f)",
                    symbol,
                    float(value_usdt or 0.0),
                    float(floor_usdt or 0.0),
                )
                return tp, sl

            ot = self.shared_state.open_trades.get(symbol)
            if not isinstance(ot, dict):
                ot = {
                    "entry_price": entry_price,
                    "quantity": quantity,
                    "position": "long",
                }
            now_ts = time.time()
            ot["tp"] = tp
            ot["sl"] = sl
            if tp is not None:
                ot.setdefault("initial_tp", tp)
            if sl is not None:
                ot.setdefault("initial_sl", sl)
            ot["tier"] = tier  # Store tier for monitoring
            ot["tp_sl_method"] = self._last_tp_sl_method.get(symbol)
            # Refresh core entry metadata on every new fill to avoid stale-age churn loops.
            ot["entry_price"] = float(entry_price or 0.0)
            ot["quantity"] = float(quantity or 0.0)
            ot["opened_at"] = now_ts
            ot["created_at"] = now_ts
            if symbol in self._dynamic_trailing_mult:
                ot["trailing_atr_mult"] = float(self._dynamic_trailing_mult.get(symbol, 0.0) or 0.0)

            # Tier B: Set shorter TTL for fast exits
            if tier == "B":
                ot["ttl_sec"] = int(getattr(self.config, "TIER_B_TTL_SEC", 300))  # 5 minutes default
                ot["created_at"] = now_ts

            self.shared_state.open_trades[symbol] = ot
        except Exception as persist_err:
            self.logger.error(
                "[TPSL_PERSIST_FAILED] %s: TP/SL computed (tp=%s sl=%s) but not stored: %s",
                symbol, tp, sl, persist_err, exc_info=True,
            )
            # Mark as unarmed so check_orders will retry on next cycle
            try:
                ot_fallback = getattr(self.shared_state, "open_trades", None)
                if isinstance(ot_fallback, dict) and symbol in ot_fallback:
                    ot_fallback[symbol]["_tpsl_armed"] = False
            except Exception:
                pass
        return tp, sl

    # ---------- core logic ----------

    def calculate_tp_sl(self, symbol: str, entry_price: float, tier: Optional[str] = None) -> Tuple[Optional[float], Optional[float]]:
        """
        Volatility-adaptive TP/SL model.
        - Uses ATR + realized volatility profile for dynamic SL/TP distances
        - Preserves fee-clearance and RR guardrails
        - Produces adaptive trailing ATR multiple per symbol
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

            # ATR zero-collapse protection: hard floor at 0.1% of entry price
            min_atr_pct = 0.001
            atr = max(atr, entry_price * min_atr_pct)
            # Note: guard below is unreachable after max() — kept as tombstone comment only.
            # atr is guaranteed > 0 since entry_price > 0 and min_atr_pct > 0.

            sentiment = float(self.shared_state.sentiment_score.get(symbol, 0.0) or 0.0)
            profile = self._build_volatility_profile(symbol, float(entry_price), float(atr), tier=tier)
            regime = str(profile.get("regime", "sideways")).lower()
            vol_score = float(profile.get("vol_score", 0.0))
            vol_pressure = float(profile.get("vol_pressure", 0.0))

            base_tp_atr_mult = float(getattr(self.config, "TP_ATR_MULT", 1.5) or 1.5)
            base_sl_atr_mult = float(getattr(self.config, "SL_ATR_MULT", 1.0) or 1.0)
            phase_profile = self._get_compounding_phase_profile()
            if self._snowball_asymmetry_enabled:
                base_tp_atr_mult *= float(phase_profile.get("tp_mult", 1.0) or 1.0)
                base_sl_atr_mult *= float(phase_profile.get("sl_mult", 1.0) or 1.0)

            # First cycle: prioritize getting the system live with a cleaner edge.
            try:
                total_exec = int(getattr(self.shared_state, "metrics", {}).get("total_trades_executed", 0) or 0)
            except Exception:
                total_exec = 0
            if total_exec < 1:
                base_tp_atr_mult *= float(getattr(self.config, "FIRST_CYCLE_TP_BOOST_MULT", 1.15) or 1.15)
                base_sl_atr_mult *= float(getattr(self.config, "FIRST_CYCLE_SL_TIGHTEN_MULT", 0.90) or 0.90)

            # Adaptive multipliers
            tp_atr_mult = base_tp_atr_mult
            sl_atr_mult = base_sl_atr_mult

            # Volatility response: high vol widens SL more than TP; low vol gently tightens both.
            sl_atr_mult *= (1.0 + max(-0.25, min(0.55, vol_pressure * 0.35)))
            tp_atr_mult *= (1.0 + max(-0.20, min(0.40, vol_pressure * 0.22)))

            # Market structure regime response.
            if regime in ("trend", "uptrend", "downtrend"):
                tp_atr_mult *= 1.15
                sl_atr_mult *= 0.95
            elif regime in ("high_vol", "high"):
                # High-vol regime: widen both, and widen TP more than SL to preserve expectancy.
                tp_atr_mult *= 1.20
                sl_atr_mult *= 1.10
            elif regime in ("sideways", "chop"):
                tp_atr_mult *= 0.88
                sl_atr_mult *= 0.90

            # Sentiment bias.
            if sentiment > 0.5:
                tp_atr_mult *= 1.08
            elif sentiment < -0.5:
                sl_atr_mult *= 1.08

            # Adaptive-capital feedback loop may request TP tightening/loosening.
            try:
                adaptive_tp_bias = float(
                    (getattr(self.shared_state, "dynamic_config", {}) or {}).get(
                        "ADAPTIVE_TP_BIAS_MULT",
                        1.0,
                    )
                    or 1.0
                )
                tp_atr_mult *= max(0.80, min(1.20, adaptive_tp_bias))
            except Exception:
                pass

            # Tier behavior.
            if tier == "B":
                tp_atr_mult *= 0.82
                sl_atr_mult *= 1.10

            sl_atr_mult = max(0.55, min(2.40, sl_atr_mult))
            tp_atr_mult = max(0.60, min(3.20, tp_atr_mult))

            sl_dist_raw = atr * sl_atr_mult
            tp_dist_raw = atr * tp_atr_mult
            sl_pct_raw = sl_dist_raw / entry_price if entry_price > 0 else 0.0
            tp_pct_raw = tp_dist_raw / entry_price if entry_price > 0 else 0.0

            tp_pct_min = float(getattr(self.config, "TP_PCT_MIN", 0.003) or 0.003)
            tp_pct_max = float(getattr(self.config, "TP_PCT_MAX", 0.015) or 0.015)
            sl_pct_min = float(getattr(self.config, "SL_PCT_MIN", 0.003) or 0.003)
            sl_pct_max = float(getattr(self.config, "SL_PCT_MAX", 0.008) or 0.008)

            tp_pct_clamped = max(tp_pct_min, min(tp_pct_max, tp_pct_raw))
            sl_pct_clamped = max(sl_pct_min, min(sl_pct_max, sl_pct_raw))
            tp_dist_clamped = entry_price * tp_pct_clamped
            sl_dist = entry_price * sl_pct_clamped

            phase_rr_bonus = float(phase_profile.get("rr_bonus", 0.0) or 0.0) if self._snowball_asymmetry_enabled else 0.0
            base_rr = float(getattr(self.config, "TARGET_RR_RATIO", 1.8) or 1.8) + phase_rr_bonus
            if tier == "B":
                base_rr = min(base_rr, 1.55)
            target_rr = self._resolve_adaptive_rr(base_rr, profile, sentiment)
            tp_asymmetry_bias = self._resolve_asymmetric_tp_bias(profile, sentiment, phase_profile, tier=tier)
            spread_abs, spread_pct = self._estimate_spread(symbol, float(entry_price))
            spread_rr_adjust = 0.0
            if self._spread_adaptive_enabled and spread_pct > 0:
                tight_pct = max(0.0, float(self._spread_tight_bps)) / 10000.0
                high_pct = max(0.0, float(self._spread_high_bps)) / 10000.0
                extreme_pct = max(high_pct + 1e-6, float(self._spread_extreme_bps) / 10000.0)
                spread_norm = max(
                    0.0,
                    min(1.0, (spread_pct - high_pct) / max(extreme_pct - high_pct, 1e-9)),
                )
                tight_norm = 0.0
                if tight_pct > 0:
                    tight_norm = max(0.0, min(1.0, (tight_pct - spread_pct) / tight_pct))
                spread_rr_adjust = (
                    spread_norm * max(0.0, float(self._spread_rr_bonus_max))
                    - tight_norm * max(0.0, float(self._spread_rr_discount_max))
                )
                target_rr *= (1.0 + spread_rr_adjust)
                target_rr = max(float(self._rr_min), min(float(self._rr_max), target_rr))
            if self._snowball_asymmetry_enabled:
                # Ensure TP phase bias still has effect when RR branch dominates TP distance.
                tp_bias = max(1.0, float(phase_profile.get("tp_mult", 1.0) or 1.0))
                rr_tp_lift = 1.0 + min(0.20, max(0.0, (tp_bias - 1.0) * 0.35))
                target_rr *= rr_tp_lift
                target_rr = max(float(getattr(self.config, "TP_SL_MIN_RR", 1.2) or 1.2), target_rr)
                target_rr = min(float(self._rr_max), target_rr)

            tp_dist = max(target_rr * sl_dist, tp_dist_clamped)
            tp_dist *= float(tp_asymmetry_bias)

            taker_bps = float(fee_bps(self.shared_state, "taker") or 10.0)
            slippage_bps = float(getattr(self.config, "EXIT_SLIPPAGE_BPS", 0.0) or 0.0)
            buffer_bps = float(getattr(self.config, "TP_MIN_BUFFER_BPS", 5.0) or 5.0)
            fee_clearance_bps = (taker_bps * 2.0) + slippage_bps + buffer_bps
            fee_clearance_pct = fee_clearance_bps / 10000.0

            min_tp_dist = entry_price * fee_clearance_pct
            if self._spread_adaptive_enabled and spread_abs > 0:
                min_tp_dist = max(min_tp_dist, spread_abs * max(1.0, float(self._spread_tp_floor_mult)))
            # Mark floor-hit only when TP was actually below the floor before clamping.
            tp_floor_hit = bool(tp_dist < (min_tp_dist - 1e-12))
            tp_dist = max(tp_dist, min_tp_dist)
            self._tp_floor_hit[symbol] = tp_floor_hit

            min_rr = float(getattr(self.config, "TP_SL_MIN_RR", 1.4) or 1.4)
            rr_realized = tp_dist / max(sl_dist, 1e-12)
            if rr_realized < min_rr:
                tp_dist = min_rr * sl_dist
                rr_realized = tp_dist / max(sl_dist, 1e-12)

            # Respect TP clamp ceiling unless fee floor or RR guard exceeds it.
            tp_cap_mult = 1.0
            if self._snowball_asymmetry_enabled:
                tp_cap_mult += min(
                    0.50,
                    max(0.0, float(phase_profile.get("tp_mult", 1.0) or 1.0) - 1.0) * 0.50,
                )
            tp_dist_max = entry_price * tp_pct_max * tp_cap_mult
            if tp_dist > tp_dist_max and tp_dist_max > min_tp_dist:
                tp_dist = tp_dist_max
                rr_realized = tp_dist / max(sl_dist, 1e-12)

            method = f"vol_adaptive_rr{target_rr:.2f}_{regime}"
            if atr_live:
                method += "_live"
            elif atr_cached > 0:
                method += "_cached"
            else:
                method += "_fallback"
            if tier == "B":
                method += "_tier_b"
            if spread_pct > 0:
                method += f"_spr{int(round(spread_pct * 10000.0))}bps"
            if abs(float(tp_asymmetry_bias) - 1.0) > 1e-9:
                method += f"_asym{int(round((float(tp_asymmetry_bias) - 1.0) * 100.0))}"

            self._last_tp_sl_method[symbol] = method
            final_tp_pct = tp_dist / entry_price if entry_price > 0 else 0.0
            final_sl_pct = sl_dist / entry_price if entry_price > 0 else 0.0

            # Symbol-level adaptive trailing multiplier for exit loop.
            trailing_base = float(getattr(self.config, "TRAILING_ATR_MULT", 1.5) or 1.5)
            trailing = trailing_base * (1.0 + max(-0.25, min(0.45, vol_score * 0.25)))
            if regime in {"trend", "uptrend", "downtrend"}:
                trailing *= 0.92
            elif regime in {"high_vol", "high"}:
                trailing *= 1.18
            if tier == "B":
                trailing *= 0.95
            trailing = max(0.85, min(2.50, trailing))
            self._dynamic_trailing_mult[symbol] = trailing

            self._maybe_log_profit_audit(symbol, final_tp_pct, final_sl_pct)

            tp_price = self._round_price_sync(symbol, entry_price + tp_dist)
            sl_price = self._round_price_sync(symbol, entry_price - sl_dist)

            # Calculate risk-based position size for logging (defensive)
            try:
                risk_based_quote = self.calculate_risk_based_position_size(symbol, entry_price, sl_price, tier=tier)
            except Exception:
                risk_based_quote = 0.0

            # Get equity for logging (defensive)
            equity = float(getattr(self.shared_state, "total_equity", 0.0) or 0.0)

            self.logger.info(
                "VOL-ADAPTIVE TP/SL %s: TP=%.4f SL=%.4f | ATR=%.6f ATR%%=%.2f RV%%=%.2f regime=%s vol=%.2f "
                "| RR target=%.2f final=%.2f rr_spread_adj=%.2f%% spread=%.2fbps tp_asym_bias=%.3f | SL%%=%.2f TP%%=%.2f fee_clear=%.2f%% trail_atr=%.2f "
                "| phase=%s tp_asym=%.2f sl_asym=%.2f rr_bonus=%.2f | RiskSize=%.2f RiskUSD=%.2f",
                symbol,
                tp_price,
                sl_price,
                atr,
                profile.get("atr_pct", 0.0) * 100.0,
                profile.get("rv_pct", 0.0) * 100.0,
                regime,
                vol_score,
                target_rr,
                rr_realized,
                spread_rr_adjust * 100.0,
                spread_pct * 10000.0,
                float(tp_asymmetry_bias),
                final_sl_pct * 100.0,
                final_tp_pct * 100.0,
                fee_clearance_pct * 100.0,
                trailing,
                str(phase_profile.get("name", "PHASE_1_SEED")),
                float(phase_profile.get("tp_mult", 1.0) or 1.0),
                float(phase_profile.get("sl_mult", 1.0) or 1.0),
                float(phase_profile.get("rr_bonus", 0.0) or 0.0),
                risk_based_quote,
                equity * float(getattr(self.config, "RISK_PCT_PER_TRADE", 0.01)),
            )
            return tp_price, sl_price

        except Exception as e:
            self.logger.error("VOL-ADAPTIVE TP/SL calc failed for %s: %s", symbol, e)
            return None, None

    def calculate_risk_based_position_size(self, symbol: str, entry_price: float, sl_price: float, tier: Optional[str] = None, context: Optional[Dict[str, Any]] = None) -> float:
        """
        RISK-BASED POSITION SIZING

        Converts fixed quote amounts to volatility-adjusted risk management:
        - Risk per trade: % of available equity
        - Position size: Risk Amount / SL Distance
        - Ensures consistent risk across all trades

        Formula: Position_USD = (Equity × Risk%) / SL_Distance_USD
        """
        # Guard for dust healing: skip risk-based sizing
        if context and context.get("is_dust_healing", False):
            self.logger.info("[TPSLEngine] Skipping risk-based sizing for dust healing operation on %s", symbol)
            return 0.0  # Return 0 to indicate no risk-based adjustment
            
        try:
            # Get available equity
            equity = float(getattr(self.shared_state, "total_equity", 0.0) or 0.0)
            if equity <= 0:
                self.logger.warning("[%s] No equity available for risk sizing", symbol)
                return float(getattr(self.config, "DEFAULT_PLANNED_QUOTE", 80.0))

            # Risk per trade based on tier
            risk_pct_per_trade = float(getattr(self.config, "RISK_PCT_PER_TRADE", 0.01))  # Default 1%
            if tier == "B":
                risk_pct_per_trade = float(getattr(self.config, "TIER_B_RISK_PCT", 0.005))  # 0.5% for micro trades

            # Calculate risk amount in USD
            risk_amount_usd = equity * risk_pct_per_trade

            # Calculate SL distance in USD and as a fraction of entry for logging
            sl_distance_usd = abs(sl_price - entry_price)
            sl_distance_pct = sl_distance_usd / entry_price if entry_price else 0.0

            if sl_distance_usd <= 0:
                self.logger.warning("[%s] Invalid SL distance for risk sizing: %.6f", symbol, sl_distance_usd)
                return float(getattr(self.config, "DEFAULT_PLANNED_QUOTE", 80.0))

            # Calculate position size: Risk Amount / SL Distance
            position_size_usd = risk_amount_usd / sl_distance_usd

            # Apply bounds
            min_size = float(getattr(self.config, "MIN_TRADE_QUOTE", 50.0))
            max_size = float(getattr(self.config, "MAX_TRADE_QUOTE", 250.0))
            position_size_usd = max(min_size, min(max_size, position_size_usd))

            # Tier-specific adjustments
            if tier == "B":
                position_size_usd = min(position_size_usd, float(getattr(self.config, "TIER_B_MAX_QUOTE", 40.0)))

            self.logger.info("🎯 Risk-based sizing %s: $%.2f (Risk=%.1f%% of $%.2f, SL=%.1f%%)",
                            symbol, position_size_usd, risk_pct_per_trade*100, equity, sl_distance_pct*100)

            return position_size_usd

        except Exception as e:
            self.logger.error("Risk-based sizing failed for %s: %s", symbol, e)
            return float(getattr(self.config, "DEFAULT_PLANNED_QUOTE", 80.0))

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

    def _round_trip_cost_pct(self) -> float:
        taker_bps = float(fee_bps(self.shared_state, "taker") or 10.0)
        slippage_bps = float(
            getattr(
                self.config,
                "EXIT_SLIPPAGE_BPS",
                getattr(self.config, "CR_PRICE_SLIPPAGE_BPS", 0.0),
            )
            or 0.0
        )
        fee_pct = (taker_bps * 2.0) / 10000.0
        slip_pct = (slippage_bps * 2.0) / 10000.0
        return max(0.0, fee_pct + slip_pct)

    def _passes_profit_gate(self, pnl_pct: float, reason: str) -> bool:
        """Return True if a SELL reason is allowed by fee-aware profit gate."""
        reason_u = str(reason or "").upper()
        if "SL" in reason_u or "STOP_LOSS" in reason_u:
            return True

        entry_fee_mult = float(getattr(self.config, "MIN_PLANNED_QUOTE_FEE_MULT", 2.5) or 2.5)
        fee_mult = float(getattr(self.config, "MIN_PROFIT_EXIT_FEE_MULT", 2.0) or 2.0)
        fee_mult = max(1.0, fee_mult, entry_fee_mult)
        buffer_pct = float(getattr(self.config, "TP_MIN_BUFFER_BPS", 0.0) or 0.0) / 10000.0
        min_exit = (self._round_trip_cost_pct() * fee_mult) + buffer_pct
        if pnl_pct < min_exit:
            self.logger.info(
                "[TPSL:ExitGate] blocked pnl=%.4f%% < min_exit=%.4f%%",
                pnl_pct * 100.0,
                min_exit * 100.0,
            )
            return False
        return True

    def _passes_net_exit_gate(self, pnl_pct: float) -> bool:
        """
        Hard EV gate for TP exits.
        Require expected gain to clear round-trip cost with a safety multiplier.
        """
        safety_mult = float(getattr(self.config, "TPSL_NET_EXIT_SAFETY_MULT", 2.0) or 2.0)
        min_net = self._round_trip_cost_pct() * max(1.0, safety_mult)
        if pnl_pct <= min_net:
            self.logger.info(
                "[TPSL:NetExitGate] blocked pnl=%.4f%% <= min_net=%.4f%% (safety_mult=%.2f)",
                pnl_pct * 100.0,
                min_net * 100.0,
                max(1.0, safety_mult),
            )
            return False
        return True

    def _passes_tp_distance_gate(self, symbol: str, entry_price: float, tp_price: float) -> bool:
        """Reject TP targets that are too close to cover round-trip costs safely."""
        if entry_price <= 0 or tp_price <= 0:
            return False
        tp_move_pct = abs(float(tp_price) - float(entry_price)) / max(float(entry_price), 1e-12)
        dist_mult = float(getattr(self.config, "TPSL_MIN_TP_DISTANCE_RT_MULT", 2.0) or 2.0)
        min_tp_pct = self._round_trip_cost_pct() * max(1.0, dist_mult)
        if tp_move_pct < min_tp_pct:
            self.logger.warning(
                "[TPSL:TPDistanceGate] %s blocked tp_move=%.4f%% < min_tp=%.4f%%",
                symbol,
                tp_move_pct * 100.0,
                min_tp_pct * 100.0,
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

        filters = await self._get_symbol_filters(symbol)
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

    async def _get_symbol_filters(self, symbol: str) -> dict:
        """Fetch exchange symbol filters, preferring async cached getter over shared_state fallback."""
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
        return filters

    def _get_symbol_filters_sync(self, symbol: str) -> dict:
        """Synchronous fallback: reads only from shared_state.symbol_filters cache."""
        sym = (symbol or "").upper().replace("/", "")
        return getattr(self.shared_state, "symbol_filters", {}).get(sym, {}) or {}

    def _round_price_sync(self, symbol: str, price: float) -> float:
        """Round price to the exchange tick size; falls back to 4 d.p. if unavailable."""
        try:
            filters = self._get_symbol_filters_sync(symbol)
            tick_size = float(
                filters.get("PRICE_FILTER", {}).get("tickSize")
                or filters.get("tickSize")
                or filters.get("_normalized", {}).get("tick_size", 0.0)
                or 0.0
            )
            if tick_size > 0:
                return round(round(price / tick_size) * tick_size, 8)
        except Exception:
            pass
        return round(price, 4)

    async def _get_min_notional(self, symbol: str) -> float:
        filters = await self._get_symbol_filters(symbol)
        return float(
            filters.get("MIN_NOTIONAL", {}).get("minNotional")
            or filters.get("minNotional")
            or filters.get("_normalized", {}).get("min_notional", 0.0)
            or 0.0
        )

    async def _get_close_lock(self, symbol: str) -> asyncio.Lock:
        """
        Get or create an asyncio.Lock for close operations on this symbol.
        
        RACE CONDITION FIX #4: Prevents concurrent _close() calls for same symbol.
        
        THREAD-SAFE: Uses double-check locking pattern.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC')
            
        Returns:
            asyncio.Lock for close operations on this symbol
        """
        sym = str(symbol or "").upper()
        
        # Fast path: lock already exists
        if sym in self._symbol_close_locks:
            return self._symbol_close_locks[sym]
        
        # Slow path: create new lock under synchronization
        async with self._symbol_close_locks_lock:
            # Double-check after acquiring lock
            if sym not in self._symbol_close_locks:
                self._symbol_close_locks[sym] = asyncio.Lock()
                self.logger.debug(f"[TPSL:Race:Lock] Created close lock for {sym}")
            return self._symbol_close_locks[sym]

    async def _close_via_execution_manager(
        self,
        symbol: str,
        close_reason: str,
        *,
        tag: str = "tp_sl",
        force_finalize: bool = True,
    ) -> Dict[str, Any]:
        """
        Canonical close path for TP/SL.
        Contract: TP/SL must never execute orders via ExchangeClient directly.
        """
        em = getattr(self, "execution_manager", None)
        close_fn = getattr(em, "close_position", None)
        if self._enforce_execution_manager_only and not callable(close_fn):
            raise RuntimeError(
                "TPSLEngine requires ExecutionManager.close_position(); direct exchange execution is forbidden"
            )
        if not callable(close_fn):
            raise RuntimeError("TPSLEngine close routing unavailable: ExecutionManager.close_position missing")
        res = close_fn(
            symbol=symbol,
            reason=close_reason,
            force_finalize=bool(force_finalize),
            tag=str(tag or "tp_sl"),
        )
        if iscoroutine(res):
            res = await res
        if not isinstance(res, dict):
            return {"ok": False, "status": "error", "reason": "invalid_close_result"}
        return res

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
            # Canonical architecture: positions are the only source of truth
            if getattr(self.shared_state, "trading_mode", "") == "shadow":
                positions = getattr(self.shared_state, "virtual_positions", {}) or {}
            else:
                positions = getattr(self.shared_state, "positions", {}) or {}

            # Derive open trades from positions
            open_trades = {
                sym: pos
                for sym, pos in positions.items()
                if float(pos.get("position_qty", 0) or 0) > 0
            }
            
            prices = self.shared_state.latest_prices or {}
            self.logger.info("[TPSL:check] open_trades=%d prices=%d", len(open_trades), len(prices))
            to_close: List[Tuple[str, str]] = []

            for symbol, tr in open_trades.items():
                pos = positions.get(symbol, {}) if isinstance(positions, dict) else {}
                gate_ref = pos if isinstance(pos, dict) and pos else tr
                try:
                    is_open_sig, value_usdt, floor_usdt = self.shared_state.classify_position_snapshot(symbol, gate_ref)
                except Exception:
                    is_open_sig, value_usdt, floor_usdt = True, 0.0, 0.0
                if not is_open_sig:
                    try:
                        self.shared_state.open_trades.pop(symbol, None)
                    except Exception:
                        pass
                    _log_skip(
                        symbol,
                        float(tr.get("quantity") or 0.0),
                        None,
                        None,
                        None,
                        f"below_significant_floor value={value_usdt:.6f} floor={floor_usdt:.6f}",
                    )
                    continue
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

                # Prefer stored TP/SL, compute if missing or if arming failed
                tp = tr.get("tp"); sl = tr.get("sl")
                if tr.get("_tpsl_armed") is False:
                    # EM flagged that set_initial_tp_sl failed — force re-arm
                    self.logger.warning("[TPSL:RE_ARM] %s retrying TP/SL arm (_tpsl_armed=False)", symbol)
                    tp, sl = None, None  # force recalculation below
                if tp is None or sl is None:
                    tp, sl = self.calculate_tp_sl(symbol, float(entry_price))
                    if tp is None or sl is None:
                        _log_skip(symbol, qty, float(current_price) if current_price is not None else None, tp, sl, "missing_tp_sl")
                        continue
                    try:
                        tr["tp"] = tp; tr["sl"] = sl  # cache for next pass
                        tr.setdefault("initial_tp", tp)
                        tr.setdefault("initial_sl", sl)
                        tr.pop("_tpsl_armed", None)  # clear failed-arm flag on success
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
                # Rule: If PnL < (2 × round-trip fees = 4 × taker) AND age > 45m → Exit to recycle capital.
                # rt_fee_pct = 2 × taker; threshold = 2 × rt_fee_pct = 4 × taker.
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
                        activate_r = float(getattr(self.config, "TRAILING_ACTIVATE_R_MULT", 0.60) or 0.60)
                        initial_sl = float(tr.get("initial_sl") or tr.get("sl") or 0.0)
                        base_r = max(float(entry_price) - initial_sl, 0.0)
                        mfe = max(cp - float(entry_price), 0.0)
                        if base_r > 0 and mfe < (activate_r * base_r):
                            self.logger.debug(
                                "[%s] Trailing not armed yet: mfe=%.6f < %.2fR (R=%.6f)",
                                symbol, mfe, activate_r, base_r
                            )
                        else:
                            trailing_mult = float(
                                tr.get("trailing_atr_mult")
                                or self._dynamic_trailing_mult.get(symbol)
                                or getattr(self.config, "TRAILING_ATR_MULT", 1.5)
                                or 1.5
                            )
                            trail_dist = atr * trailing_mult
                            new_sl = self._round_price_sync(symbol, cp - trail_dist)
                            if new_sl > float(sl):
                                 self.logger.info("[%s] Trailing SL up: %.4f -> %.4f (atr_mult=%.2f)", symbol, float(sl), new_sl, trailing_mult)
                                 tr["sl"] = new_sl
                                 sl = new_sl
                                 self.shared_state.open_trades[symbol] = tr

                    if cp >= float(tp):
                        reason = "TP Hit"
                        if (
                            self._passes_tp_distance_gate(symbol, float(entry_price), float(tp))
                            and
                            self._passes_net_exit_gate(pnl_pct)
                            and self._passes_profit_gate(pnl_pct, reason)
                            and await self._passes_excursion_gate(symbol, entry_price, cp, atr, reason)
                        ):
                            to_close.append((symbol, reason))
                        else:
                            self.logger.info("[%s] Profit gate blocked SELL (%s).", symbol, reason)
                    elif cp <= float(sl):
                        self.logger.warning(
                            "[TPSL:SL_TRIGGER] %s cp=%.4f <= sl=%.4f entry=%.4f pnl=%.4f%%",
                            symbol,
                            cp,
                            float(sl),
                            entry_price,
                            pnl_pct * 100 if pnl_pct is not None else -999,
                        )
                        to_close.append((symbol, "SL Hit"))
                elif position == "short":
                    # Update Trailing SL if price moves down
                    if use_trailing and atr and atr > 0:
                        activate_r = float(getattr(self.config, "TRAILING_ACTIVATE_R_MULT", 0.60) or 0.60)
                        initial_sl = float(tr.get("initial_sl") or tr.get("sl") or 0.0)
                        base_r = max(initial_sl - float(entry_price), 0.0)
                        mfe = max(float(entry_price) - cp, 0.0)
                        if base_r > 0 and mfe < (activate_r * base_r):
                            self.logger.debug(
                                "[%s] Trailing not armed yet (short): mfe=%.6f < %.2fR (R=%.6f)",
                                symbol, mfe, activate_r, base_r
                            )
                        else:
                            trailing_mult = float(
                                tr.get("trailing_atr_mult")
                                or self._dynamic_trailing_mult.get(symbol)
                                or getattr(self.config, "TRAILING_ATR_MULT", 1.5)
                                or 1.5
                            )
                            trail_dist = atr * trailing_mult
                            new_sl = self._round_price_sync(symbol, cp + trail_dist)
                            if new_sl < float(sl):
                                 self.logger.info("[%s] Trailing SL down: %.4f -> %.4f (atr_mult=%.2f)", symbol, float(sl), new_sl, trailing_mult)
                                 tr["sl"] = new_sl
                                 sl = new_sl
                                 self.shared_state.open_trades[symbol] = tr

                    if cp <= float(tp):
                        reason = "TP Hit (short)"
                        if (
                            self._passes_tp_distance_gate(symbol, float(entry_price), float(tp))
                            and
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
                    # RACE CONDITION FIX #4: Hold per-symbol lock to prevent concurrent closes
                    lock = await self._get_close_lock(sym)
                    async with lock:
                        async with sem:
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
                                if min_notional > 0 and total_position_qty > 0:
                                    try:
                                        cp_now = float((self.shared_state.latest_prices or {}).get(sym) or 0.0)
                                        value_usdt = total_position_qty * cp_now if cp_now > 0 else 0.0
                                        if value_usdt > 0 and value_usdt < min_notional:
                                            self.logger.warning(
                                                "[TPSL] Proceeding with close %s: notional=%.4f < min_notional=%.4f",
                                                sym,
                                                value_usdt,
                                                min_notional,
                                            )
                                    except Exception:
                                        pass
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
                        try:
                            tj = getattr(self, "trade_journal", None)
                            if tj:
                                try:
                                    tj.record("TPSL_TRIGGER", {
                                        "symbol": sym,
                                        "reason": close_reason,
                                        "trigger": reason,
                                        "entry_price": entry_price,
                                        "current_price": current_price,
                                        "tp": tp,
                                        "sl": sl,
                                        "pnl_pct": round(pnl_pct * 100, 4) if pnl_pct is not None else None,
                                        "qty": position_qty,
                                        "position": position,
                                        "tier": tier,
                                        "session_id": getattr(self, "session_id", ""),
                                    })
                                except Exception:
                                    pass
                            res = await self._close_via_execution_manager(
                                sym,
                                close_reason,
                                force_finalize=True,
                                tag="tp_sl",
                            )
                        except Exception as route_err:
                            self.logger.error(
                                "[TPSL:ROUTING_GUARD] %s close blocked: %s",
                                sym,
                                route_err,
                                exc_info=True,
                            )
                            res = {"ok": False, "status": "error", "reason": str(route_err)}
                        try:
                            if isinstance(res, dict):
                                self.logger.info(
                                    "[TPSL:CLOSE_RESULT] %s trigger=%s mapped_reason=%s ok=%s status=%s order_id=%s executed_qty=%s",
                                    sym,
                                    reason,
                                    close_reason,
                                    bool(res.get("ok")),
                                    str(res.get("status", "")),
                                    str(res.get("orderId", res.get("order_id", ""))),
                                    str(res.get("executedQty", res.get("executed_qty", ""))),
                                )
                            else:
                                self.logger.warning("[TPSL:CLOSE_RESULT] %s trigger=%s non-dict response=%r", sym, reason, res)
                        except Exception:
                            self.logger.debug("[TPSL] close result logging failed for %s", sym, exc_info=True)
                        try:
                            status = str(res.get("status", "")).lower() if isinstance(res, dict) else ""
                            executed_qty = 0.0
                            if isinstance(res, dict):
                                try:
                                    executed_qty = float(res.get("executedQty", res.get("executed_qty", 0.0)) or 0.0)
                                except Exception:
                                    executed_qty = 0.0
                            is_fill = status in {"filled", "partially_filled"} and executed_qty > 0.0
                            # Always stamp the attempt so failed closes are debounced too.
                            self._last_close_attempt[sym] = self._mono()
                            if is_fill:
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
                            elif status and status not in {"error", "rejected", "cancelled", "canceled"}:
                                self.logger.warning(
                                    "[TPSL:PENDING_FILL] %s close submitted but not filled yet status=%s qty=%.8f — EM reconciliation will handle",
                                    sym, status, executed_qty,
                                )
                        except Exception as bk_err:
                            self.logger.error("[TPSL:BOOKKEEPING_FAILED] %s: %s", sym, bk_err, exc_info=True)

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
        self.logger.debug("[TPSL:DIAG] reached pre-loop section")

        await self._safe_status_update("OK", "TPSLEngine loop initialized.")
        self.logger.info("TPSLEngine started TP/SL monitoring loop.")

        interval = max(0.5, float(self._interval))
        next_tick = self._mono()

        try:
            while True:
                self.logger.debug("[TPSL:DIAG] entered monitoring loop")
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



# ===== P9 Spec Helpers =====

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



# ===== P9 TP/SL Normalization Wrappers =====
def _wrap_tp_sl_outputs(cls):
    # Add required P9 lifecycle stubs if not already defined.
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
