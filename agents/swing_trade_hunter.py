import asyncio
import logging
import os
import time
import numpy as np
try:
    import tensorflow as tf
except Exception:
    tf = None
from datetime import datetime
from functools import partial
from typing import Any, Dict, List, Set, Tuple

from utils.indicators import compute_ema, compute_rsi, compute_macd, compute_bollinger_bands
try:
    from utils.status_logger import log_component_status
except Exception:
    def log_component_status(*args, **kwargs):
        return None
from utils.shared_state_tools import inject_agent_signal, spread_bps as ss_spread_bps, min_notional as ss_min_notional
from core.model_manager import safe_load_model, save_model, build_model_path
from core.stubs import TradeIntent
from agents.edge_calculator import compute_agent_edge

AGENT_NAME = "SwingTradeHunter"
logger = logging.getLogger(AGENT_NAME)
logger.setLevel(logging.DEBUG)

# Setup file handler
log_path = f"logs/agents/{AGENT_NAME.lower()}.log"
os.makedirs(os.path.dirname(log_path), exist_ok=True)
file_handler = logging.FileHandler(log_path)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] [%(name)s] %(message)s')
file_handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(file_handler)


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


class SwingTradeHunter:
    agent_type = "strategy"

    def __init__(
        self,
        shared_state,
        market_data,
        execution_manager,
        config,
        tp_sl_engine=None,
        model_manager=None,
        symbols=None,
        timeframe='1h',
        name=AGENT_NAME,
        **kwargs
    ):
        self.shared_state = shared_state
        self.market_data = market_data
        self.execution_manager = execution_manager  # kept for backward compatibility
        self.config = config
        self.tp_sl_engine = tp_sl_engine
        self.model_manager = model_manager
        self.name = name
        self.timeframe = timeframe
        # Signal collection buffer for AgentManager
        self._collected_signals: List[Dict[str, Any]] = []
        self._collecting_for_agent_manager = False

        # Modified line: Use get_accepted_symbols() from shared_state
        resolved_symbols = symbols or []
        if not resolved_symbols:
            getter = getattr(self.shared_state, "get_accepted_symbols", None)
            if callable(getter):
                try:
                    resolved_symbols = getter()
                except Exception:
                    resolved_symbols = []
        if asyncio.iscoroutine(resolved_symbols):
            resolved_symbols = []
        if isinstance(resolved_symbols, dict):
            resolved_symbols = list(resolved_symbols.keys())
        self.symbols = list(resolved_symbols or [])
        self.model_cache = {}
        # Performance tracking
        self.trades_count = 0
        self.win_count = 0
        self.loss_count = 0
        if isinstance(self.config, dict):
            auto_train_raw = self.config.get(
                "SWING_AUTO_TRAIN",
                self.config.get("AUTO_TRAIN", False),
            )
        else:
            auto_train_raw = getattr(
                self.config,
                "SWING_AUTO_TRAIN",
                getattr(self.config, "AUTO_TRAIN", False),
            )
        self._auto_train_enabled = _as_bool(auto_train_raw)
        self._retrain_inflight: Set[str] = set()
        self._retrain_last_attempt_ts: Dict[str, float] = {}
        self._retrain_last_failure_ts: Dict[str, float] = {}
        self._retrain_last_success_ts: Dict[str, float] = {}

        if tf is None:
            logger.warning("[%s] TensorFlow unavailable; inference will use indicator-only path and retrain is disabled.", self.name)

        # Pre-load models
        for symbol in self.symbols:
            path = build_model_path(self.name, symbol)
            model = safe_load_model(path) # Use safe_load_model which handles exceptions
            if model is None:
                logger.warning(f"[{self.name}] ❌ No pre-trained model found for {symbol}. Will train on first retrain.")
                self.model_cache[symbol] = None # Explicitly set to None if not found
            else:
                logger.info(f"[{self.name}] ✅ Loaded model for {symbol}")
                self.model_cache[symbol] = model

        log_component_status(self.name, "Initialized")
        logger.info(f"🚀 {self.name} initialized with {len(self.symbols)} symbols on {self.timeframe} timeframe.")

    @staticmethod
    async def _await_maybe(coro):
        """Helper to await if coroutine, else return value."""
        return await coro if asyncio.iscoroutine(coro) else coro

    def _cfg(self, key: str, default: Any = None) -> Any:
        if isinstance(self.config, dict):
            return self.config.get(key, default)
        return getattr(self.config, key, default)

    def _round_trip_cost_pct(self) -> float:
        """Estimate round-trip friction as ratio (e.g., 0.0045 = 0.45%)."""
        fee_bps = float(self._cfg("CR_FEE_BPS", 10.0) or 10.0)
        slip_bps = float(self._cfg("CR_PRICE_SLIPPAGE_BPS", 10.0) or 10.0)
        buffer_bps = float(self._cfg("PRETRADE_EFFECT_BUFFER_BPS", 5.0) or 5.0)
        return ((fee_bps * 2.0) + (slip_bps * 2.0) + buffer_bps) / 10000.0

    async def _compute_expected_move_pct(self, symbol: str, action: str) -> float:
        """
        Reuse existing OHLCV + TP/SL context to produce expected_move_pct in percent units.
        This keeps Swing signals compatible with pre-trade profitability gates.
        """
        fallback_pct = float(self._cfg("SWING_EXPECTED_MOVE_FALLBACK_PCT", 1.2) or 1.2)
        try:
            data = None
            if hasattr(self.market_data, "get_market_data_sync"):
                data = self.market_data.get_market_data_sync(symbol, self.timeframe)
            elif hasattr(self.market_data, "get_market_data"):
                data = await self.market_data.get_market_data(symbol, self.timeframe)
            if not data and hasattr(self.shared_state, "get_market_data_sync"):
                data = self.shared_state.get_market_data_sync(symbol, self.timeframe)

            rows = self._normalize_ohlcv_rows(data)
            if len(rows) < 20:
                return fallback_pct

            closes = np.asarray([float(r.get("close", 0.0) or 0.0) for r in rows], dtype=float)
            highs = np.asarray([float(r.get("high", 0.0) or 0.0) for r in rows], dtype=float)
            lows = np.asarray([float(r.get("low", 0.0) or 0.0) for r in rows], dtype=float)
            if closes.size == 0 or float(closes[-1]) <= 0:
                return fallback_pct

            close_now = float(closes[-1])

            # ATR-like volatility move
            tr_vals: List[float] = []
            for i in range(len(closes)):
                prev_close = float(closes[i - 1]) if i > 0 else float(closes[i])
                tr = max(
                    float(highs[i] - lows[i]),
                    abs(float(highs[i] - prev_close)),
                    abs(float(lows[i] - prev_close)),
                )
                tr_vals.append(float(tr))
            atr = float(np.mean(tr_vals[-14:])) if tr_vals else 0.0
            atr_pct = (atr / close_now) * 100.0 if close_now > 0 else fallback_pct
            atr_pct = max(0.5, min(6.0, float(atr_pct or fallback_pct)))

            # TP/SL-derived move
            tp_pct = 0.0
            try:
                if self.tp_sl_engine and hasattr(self.tp_sl_engine, "calculate_tp_sl"):
                    tp, sl = self.tp_sl_engine.calculate_tp_sl(symbol, close_now)
                    if str(action or "").upper() == "BUY" and float(tp or 0.0) > close_now:
                        tp_pct = ((float(tp) - close_now) / close_now) * 100.0
                    elif str(action or "").upper() == "SELL" and float(sl or 0.0) < close_now:
                        tp_pct = ((close_now - float(sl)) / close_now) * 100.0
            except Exception:
                tp_pct = 0.0

            if tp_pct <= 0.0:
                tp_pct = atr_pct * 1.2

            expected_move_pct = (0.65 * float(tp_pct)) + (0.35 * float(atr_pct))
            min_pct = float(self._cfg("SWING_EXPECTED_MOVE_MIN_PCT", 0.5) or 0.5)
            max_pct = float(self._cfg("SWING_EXPECTED_MOVE_MAX_PCT", 6.0) or 6.0)
            return max(min_pct, min(max_pct, float(expected_move_pct)))
        except Exception as e:
            logger.debug("[%s] Expected move fallback for %s: %s", self.name, symbol, e, exc_info=True)
            return fallback_pct

    async def _passes_local_buy_viability(
        self,
        symbol: str,
        quote_hint: float,
        expected_move_pct: float,
    ) -> Tuple[bool, str]:
        """Lightweight pre-publish viability guard for BUY signals."""
        try:
            quote = float(quote_hint or 0.0)
            if quote <= 0.0:
                return False, "invalid_quote_hint"

            move_pct = float(expected_move_pct or 0.0)
            if move_pct <= 0.0:
                return False, "missing_expected_move"

            # Normalize move into ratio space for EV comparison.
            move_ratio = (move_pct / 100.0) if abs(move_pct) > 1.0 else move_pct
            round_trip_ratio = float(self._round_trip_cost_pct())
            ev_mult = float(self._cfg("AGENT_LOCAL_EV_MULTIPLIER", 1.0) or 1.0)
            ev_mult = max(0.5, min(2.0, ev_mult))
            required_ratio = round_trip_ratio * ev_mult
            if move_ratio < required_ratio:
                return False, "expected_move_below_cost_floor"

            spread = ss_spread_bps(self.shared_state, symbol)
            max_spread_bps = float(self._cfg("BUY_MAX_SPREAD_BPS", 25.0) or 25.0)
            if spread is not None and spread > max_spread_bps:
                return False, "spread_too_wide"

            mn = ss_min_notional(self.shared_state, symbol)
            if mn is not None and quote < (float(mn) * 0.95):
                return False, "quote_below_symbol_min_notional"
            return True, "ok"
        except Exception:
            return True, "guard_error_allow"

    def _normalize_ohlcv_rows(self, data: Any) -> List[Dict[str, float]]:
        """
        Normalize OHLCV into canonical dict rows with open/high/low/close/volume.
        Accepts both dict payloads and list/tuple klines.
        """
        rows: List[Dict[str, float]] = []
        if not isinstance(data, list):
            return rows

        for idx, item in enumerate(data):
            try:
                ts = float(idx)
                if isinstance(item, dict):
                    o = item.get("o", item.get("open"))
                    h = item.get("h", item.get("high"))
                    l = item.get("l", item.get("low"))
                    c = item.get("c", item.get("close", item.get("price")))
                    v = item.get("v", item.get("volume"))
                    ts = float(item.get("ts", item.get("timestamp", item.get("t", idx))) or idx)
                else:
                    seq = list(item)
                    if len(seq) >= 6:
                        ts = float(seq[0])
                        o, h, l, c, v = seq[1], seq[2], seq[3], seq[4], seq[5]
                    elif len(seq) == 5:
                        o, h, l, c, v = seq
                    else:
                        continue

                rows.append(
                    {
                        "timestamp": float(ts),
                        "open": float(o),
                        "high": float(h),
                        "low": float(l),
                        "close": float(c),
                        "volume": float(v),
                    }
                )
            except Exception:
                continue

        if not rows:
            return rows
        dedup: Dict[float, Dict[str, float]] = {}
        for r in rows:
            dedup[float(r["timestamp"])] = r
        return [dedup[k] for k in sorted(dedup.keys())]

    async def _fetch_retrain_rows(self, symbol: str, min_rows: int, fetch_limit: int) -> List[Dict[str, float]]:
        rows: List[Dict[str, float]] = []
        get_md = getattr(self.shared_state, "get_market_data", None)
        if callable(get_md):
            try:
                cached = get_md(symbol, self.timeframe)
                cached = await self._await_maybe(cached)
                rows = self._normalize_ohlcv_rows(cached)
            except Exception:
                rows = []

        ec = getattr(self.execution_manager, "exchange_client", None) if self.execution_manager else None
        if len(rows) < min_rows and ec and hasattr(ec, "get_klines"):
            try:
                raw_klines = await ec.get_klines(symbol, self.timeframe, limit=int(fetch_limit))
                fetched_rows = self._normalize_ohlcv_rows(raw_klines)
                if len(fetched_rows) > len(rows):
                    rows = fetched_rows
                set_md = getattr(self.shared_state, "set_market_data", None)
                if callable(set_md) and fetched_rows:
                    # Store canonical short keys so other components can reuse training history.
                    compact = [
                        {
                            "ts": r["timestamp"],
                            "o": r["open"],
                            "h": r["high"],
                            "l": r["low"],
                            "c": r["close"],
                            "v": r["volume"],
                        }
                        for r in fetched_rows
                    ]
                    await self._await_maybe(set_md(symbol, self.timeframe, compact))
            except Exception as e:
                logger.debug("[%s] Retrain backfill failed for %s: %s", self.name, symbol, e, exc_info=True)

        return rows

    def _can_launch_retrain(self, symbol: str) -> Tuple[bool, str]:
        now_ts = time.time()
        cooldown_s = max(0.0, float(self._cfg("SWING_RETRAIN_COOLDOWN_S", 900.0) or 0.0))
        last_attempt = float(self._retrain_last_attempt_ts.get(symbol, 0.0) or 0.0)
        if cooldown_s > 0 and (now_ts - last_attempt) < cooldown_s:
            remain = max(0.0, cooldown_s - (now_ts - last_attempt))
            return False, f"cooldown_active_{remain:.1f}s"

        fail_backoff_s = max(0.0, float(self._cfg("SWING_RETRAIN_FAIL_BACKOFF_S", 1800.0) or 0.0))
        last_fail = float(self._retrain_last_failure_ts.get(symbol, 0.0) or 0.0)
        if fail_backoff_s > 0 and last_fail > 0 and (now_ts - last_fail) < fail_backoff_s:
            remain = max(0.0, fail_backoff_s - (now_ts - last_fail))
            return False, f"failure_backoff_{remain:.1f}s"

        return True, "ok"

    async def generate_signals(self) -> List[Dict[str, Any]]:
        """
        Canonical strategy contract:
        - Generate and return signal payloads
        - Do not execute trades directly
        """
        self._collected_signals = []
        self._collecting_for_agent_manager = True
        try:
            # CRITICAL: Load symbols first like TrendHunter/DipSniper do!
            await self._load_symbols()
            await self.run_once()
            signals = self._collected_signals
            self._collected_signals = []
            return signals
        finally:
            self._collecting_for_agent_manager = False

    async def _publish_trade_intent(self, payload: Dict[str, Any]) -> bool:
        logger.warning(f"[{self.name}] ENTERING _publish_trade_intent with payload: symbol={payload.get('symbol')} side={payload.get('side')}")
        event_bus = getattr(self.shared_state, "event_bus", None)
        logger.warning(f"[{self.name}] event_bus={event_bus}")
        publish = getattr(event_bus, "publish", None) if event_bus else None
        logger.warning(f"[{self.name}] publish method={publish}")
        if callable(publish):
            try:
                logger.warning(f"[{self.name}] Calling event_bus.publish('events.trade.intent', TradeIntent(...))")
                await publish("events.trade.intent", TradeIntent(**payload))
                logger.info(
                    "[%s] Published TradeIntent: %s %s",
                    self.name,
                    payload.get("symbol"),
                    payload.get("side"),
                )
                logger.warning(f"[{self.name}] ✅ Successfully published TradeIntent")
                return True
            except Exception as e:
                logger.warning(
                    "[%s] ❌ Failed to publish TradeIntent for %s: %s",
                    self.name,
                    payload.get("symbol"),
                    e,
                    exc_info=True,
                )
        else:
            logger.warning(f"[{self.name}] event_bus.publish is NOT callable, trying fallback")
        
        emit_event = getattr(self.shared_state, "emit_event", None)
        logger.warning(f"[{self.name}] emit_event method={emit_event}")
        if callable(emit_event):
            try:
                logger.warning(f"[{self.name}] Calling shared_state.emit_event('TradeIntent', ...)")
                await emit_event("TradeIntent", dict(payload))
                logger.warning(f"[{self.name}] ✅ Successfully emitted via fallback")
                return True
            except Exception as e:
                logger.warning(
                    "[%s] ❌ Fallback TradeIntent emit failed for %s: %s",
                    self.name,
                    payload.get("symbol"),
                    e,
                    exc_info=True,
                )
        else:
            logger.warning(f"[{self.name}] shared_state.emit_event is NOT callable")
        
        logger.warning(f"[{self.name}] ❌ FAILED to publish TradeIntent - no method available")
        return False

    async def _load_symbols(self) -> None:
        """Load accepted symbols from SharedState (matching TrendHunter/DipSniper pattern)."""
        try:
            getter = getattr(self.shared_state, "get_accepted_symbols", None)
            if callable(getter):
                try:
                    res = getter(full=True)
                except TypeError:
                    res = getter()
                accepted = await res if asyncio.iscoroutine(res) else (res or {})
            else:
                accepted = {}
            
            if not isinstance(accepted, dict):
                snap = getattr(self.shared_state, "get_accepted_symbols_snapshot", None)
                if callable(snap):
                    r = snap()
                    accepted = await r if asyncio.iscoroutine(r) else (r or {})
                if not isinstance(accepted, dict):
                    accepted = {s: {} for s in (accepted or [])}
            
            # ITERATION 2 FIX: If accepted_symbols empty, use DEFAULT_SYMBOLS as fallback
            if not accepted:
                logger.warning(f"[{self.name}] ⚠️  accepted_symbols is empty! Using DEFAULT_SYMBOLS fallback...")
                try:
                    from core.bootstrap_symbols import DEFAULT_SYMBOLS
                    accepted = DEFAULT_SYMBOLS
                    logger.warning(f"[{self.name}] ✅ Using {len(DEFAULT_SYMBOLS)} DEFAULT_SYMBOLS as fallback")
                except Exception as e:
                    logger.error(f"[{self.name}] Failed to load DEFAULT_SYMBOLS fallback: {e}")
                    accepted = {}
            
            new_symbols = list(accepted.keys())
            if new_symbols != self.symbols:
                self.symbols = new_symbols
                logger.info(f"[{self.name}] 🔄 Loaded {len(self.symbols)} symbols from SharedState")
                # Pre-load models for any new symbols
                for symbol in new_symbols:
                    if symbol not in self.model_cache:
                        try:
                            path = build_model_path(self.name, symbol)
                            self.model_cache[symbol] = safe_load_model(path)
                        except Exception:
                            self.model_cache[symbol] = None
        except Exception as e:
            logger.error(f"[{self.name}] Failed to load symbols: {e}", exc_info=True)
            # ✅ FIX: Do NOT wipe symbols on exception - keep existing ones
            # This prevents: error in loading → self.symbols=[] → loop over zero symbols
            if not self.symbols:
                logger.warning(f"[{self.name}] No symbols available to trade; agent will be idle until symbols load")
            # Removed: self.symbols = []

    async def run_once(self):
        # ✅ ATOMIC GUARD: Prevent concurrent run_once() calls
        # This stops scheduler re-entry and BTC duplicate processing
        if hasattr(self, '_run_once_lock'):
            if self._run_once_lock:
                logger.warning(f"[{self.name}] run_once() already in progress, skipping concurrent invocation.")
                return
        else:
            self._run_once_lock = False
        
        self._run_once_lock = True
        try:
            logger.info(f"[{self.name}] Entering run_once loop (ATOMIC).")
            await self._load_symbols()
            
            # ITERATION 1 FIX: Bypass market data ready check - symbols available, data flowing
            # try:
            #     if hasattr(self.shared_state, "is_market_data_ready"):
            #         ready = await self._await_maybe(self.shared_state.is_market_data_ready())
            #         if not ready:
            #             logger.warning(f"[{self.name}] Market data not ready. Skipping run.")
            #             return
            # except Exception:
            #     pass
            
            if not self.symbols:
                logger.info(f"[{self.name}] No symbols configured. Skipping.")
                return
            
            # ✅ CRITICAL: Iterate symbols as a snapshot to prevent re-entry issues
            symbols_to_process = list(self.symbols)
            logger.debug(f"[{self.name}] Processing {len(symbols_to_process)} symbols: {symbols_to_process}")
            
            for symbol in symbols_to_process:
                await self._process_symbol(symbol)
            
            logger.info(f"[{self.name}] Exiting run_once loop (ATOMIC COMPLETE).")
        finally:
            self._run_once_lock = False

    async def _refresh_symbols(self) -> None:
        getter = getattr(self.shared_state, "get_accepted_symbols", None)
        if not callable(getter):
            return
        try:
            res = getter()
            res = await res if asyncio.iscoroutine(res) else res
            if isinstance(res, dict):
                symbols = list(res.keys())
            else:
                symbols = list(res or [])
            if symbols:
                self.symbols = symbols
                for symbol in symbols:
                    if symbol in self.model_cache:
                        continue
                    try:
                        path = build_model_path(self.name, symbol)
                        self.model_cache[symbol] = safe_load_model(path)
                    except Exception:
                        self.model_cache[symbol] = None
        except Exception:
            logger.debug("[%s] Failed to refresh symbols from SharedState", self.name, exc_info=True)

    async def _process_symbol(self, symbol):
        logger.info(f"[{self.name}] Processing {symbol}")

        # ✅ Auto-train model if missing (NON-BLOCKING fire-and-forget)
        if self.model_cache.get(symbol) is None:
            if tf is None:
                logger.debug("[%s] Model missing for %s; TensorFlow unavailable so using indicator-only mode.", self.name, symbol)
            elif not self._auto_train_enabled:
                logger.debug("[%s] Model missing for %s; AUTO_TRAIN disabled.", self.name, symbol)
            elif symbol in self._retrain_inflight:
                logger.debug("[%s] Retrain already in-flight for %s; skipping duplicate launch.", self.name, symbol)
            else:
                allowed, gate_reason = self._can_launch_retrain(symbol)
                if not allowed:
                    logger.debug("[%s] Retrain launch blocked for %s: %s", self.name, symbol, gate_reason)
                    allowed = False
                if not allowed:
                    pass
                else:
                    self._retrain_last_attempt_ts[symbol] = time.time()
                    logger.info(f"[{self.name}] 🧠 Auto-training missing model for {symbol}... (background)")
                    self._retrain_inflight.add(symbol)
                    # Schedule training in background without blocking symbol iteration
                    asyncio.create_task(self._retrain_async_single(symbol))
        
        # Generate signal (use cached model or indicators-only fallback)
        # ✅ FIX: Add timeout to prevent signal generation from blocking loop indefinitely
        try:
            action, confidence, reason = await asyncio.wait_for(
                self._generate_signal(symbol),
                timeout=5.0
            )
        except asyncio.TimeoutError:
            logger.error(f"[{self.name}] ⏱️ Signal generation TIMEOUT for {symbol} (>5s)")
            action, confidence, reason = 'hold', 0.0, 'Signal generation timeout'
        except Exception as e:
            logger.error(f"[{self.name}] ❌ Signal generation failed for {symbol}: {e}", exc_info=True)
            action, confidence, reason = 'hold', 0.0, f'Error: {str(e)}'
        signal = {
            "source": self.name,
            "action": action,
            "confidence": confidence,
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat()
        }

        # Inject signal for tracing
        await inject_agent_signal(self.shared_state, self.name, symbol, signal)

        # Signal-only routing: actionable signals go to signal bus + collection buffer.
        min_conf = float(getattr(self.config, 'SWING_MIN_CONFIDENCE', 0.5) or 0.5)
        if action in ['buy', 'sell'] and confidence > min_conf:
            logger.info(f"[{self.name}] ✅ SIGNAL ACTIONABLE: {symbol} action={action} conf={confidence:.2f} > min={min_conf:.2f}")
            await self._submit_signal(symbol, action.upper(), float(confidence), str(reason))
        else:
            logger.warning(
                "[%s] ❌ NOT EMITTING SIGNAL for %s (action=%s conf=%.2f min=%.2f reason=%s)",
                self.name,
                symbol,
                action,
                float(confidence or 0.0),
                min_conf,
                reason,
            )

        # Update shared state health (safely handle if method doesn't exist)
        try:
            fn = getattr(self.shared_state, "update_system_health", None)
            if callable(fn):
                res = fn(
                    self.name,
                    'Operational',
                    f"Last signal: {action} ({confidence:.2f})"
                )
                if asyncio.iscoroutine(res):
                    await res
        except Exception:
            pass  # Non-fatal; ignore if method not available

    async def _submit_signal(self, symbol: str, action: str, confidence: float, reason: str) -> None:
        logger.warning(f"[{self.name}] ENTERING _submit_signal: {symbol} {action} conf={confidence}")
        action_u = str(action or "").upper().strip()
        if action_u not in {"BUY", "SELL"}:
            logger.warning(f"[{self.name}] SKIPPING _submit_signal: invalid action={action_u}")
            return

        if action_u == "SELL":
            try:
                get_qty = getattr(self.shared_state, "get_position_quantity", None)
                if callable(get_qty):
                    res = get_qty(symbol)
                    pos_qty = await res if asyncio.iscoroutine(res) else float(res or 0.0)
                    if pos_qty <= 0:
                        logger.warning("[%s] Skip SELL for %s — no position (qty=%.8f).", self.name, symbol, pos_qty)
                        return
            except Exception as e:
                logger.warning(f"[{self.name}] Failed to check position qty for {symbol}: {e}")

        quote_hint = None
        if action_u == "BUY":
            try:
                quote_hint = float(
                    getattr(
                        self.config,
                        "EMIT_BUY_QUOTE",
                        getattr(self.config, "MIN_ENTRY_USDT", getattr(self.config, "DEFAULT_PLANNED_QUOTE", 10.0)),
                    )
                    or 10.0
                )
            except Exception:
                quote_hint = 10.0

        expected_move_pct = 0.0
        edge = compute_agent_edge(
            agent_name=self.name,
            action=action_u,
            confidence=float(confidence),
            expected_move_pct=None,
            symbol=symbol,
            timeframe=self.timeframe,
        )
        if action_u == "BUY":
            expected_move_pct = await self._compute_expected_move_pct(symbol, action_u)
            edge = compute_agent_edge(
                agent_name=self.name,
                action=action_u,
                confidence=float(confidence),
                expected_move_pct=float(expected_move_pct),
                symbol=symbol,
                timeframe=self.timeframe,
            )
            buy_ok, buy_reason = await self._passes_local_buy_viability(
                symbol=symbol,
                quote_hint=float(quote_hint or 0.0),
                expected_move_pct=float(expected_move_pct),
            )
            if not buy_ok:
                logger.info(
                    "[%s] Drop BUY intent for %s: pre-publish viability failed (%s) "
                    "[exp_move=%.2f%% quote=%.2f]",
                    self.name,
                    symbol,
                    buy_reason,
                    float(expected_move_pct),
                    float(quote_hint or 0.0),
                )
                return

        signal = {
            "symbol": symbol,
            "action": action_u,
            "side": action_u,
            "confidence": float(confidence),
            "reason": reason,
            "quote": quote_hint,
            "quote_hint": quote_hint,
            "horizon_hours": 6.0,
            "agent": self.name,
            "expected_move_pct": float(expected_move_pct),
            "_expected_move_pct": float(expected_move_pct),
            "edge": float(edge),
        }
        self._collected_signals.append(signal)
        logger.info(
            "[%s] Buffered %s for %s (conf=%.2f, exp_move=%.2f%%, edge=%.3f)",
            self.name,
            action_u,
            symbol,
            float(confidence),
            float(expected_move_pct),
            float(edge),
        )
        # Always publish TradeIntent to event bus (needed for MetaController drain)
        intent_payload = {
            "symbol": str(symbol).replace("/", "").upper(),
            "side": action_u,
            "planned_quote": quote_hint,
            "quote_hint": quote_hint,
            "confidence": float(confidence),
            "agent": self.name,
            "tag": f"strategy/{self.name}",
            "reason": str(reason),
            "rationale": str(reason),
            "timeframe": self.timeframe,
            "timestamp": time.time(),
            "policy_context": {
                "expected_move_pct": float(expected_move_pct),
                "_expected_move_pct": float(expected_move_pct),
                "edge": float(edge),
            },
        }
        logger.warning(f"[{self.name}] ABOUT TO PUBLISH TradeIntent: {symbol} {action_u}")
        await self._publish_trade_intent(intent_payload)
        logger.warning(f"[{self.name}] PUBLISHED TradeIntent: {symbol} {action_u}")

    async def _generate_signal(self, symbol):
        logger.warning(f"[{self.name}] ENTERING _generate_signal for {symbol}")
        # ✅ FIX: Ensure market data readiness event is checked before proceeding
        # We rely on run_once() to have already verified readiness, but double-check here
        try:
            # Attempt to wait for market data ready event with timeout
            md_ready_event = getattr(self.shared_state, "market_data_ready_event", None)
            if md_ready_event and hasattr(md_ready_event, "is_set"):
                if not md_ready_event.is_set():
                    logger.warning(f"[{self.name}] Market data ready event not set for {symbol}. Returning hold.")
                    return 'hold', 0.0, 'Market data not ready'
        except Exception as e:
            logger.warning(f"[{self.name}] Failed to check market data ready: {e}")
            pass
        
        # Fetch market data (first try cache, then fallback to exchange)
        logger.warning(f"[{self.name}] Fetching market data for {symbol} on {self.timeframe}")
        
        # Step 1: Try to get from cache
        data = None
        if hasattr(self.market_data, "get_market_data_sync"):
            data = self.market_data.get_market_data_sync(symbol, self.timeframe)
        elif hasattr(self.market_data, "get_market_data"):
            data = await self.market_data.get_market_data(symbol, self.timeframe)
        
        # Step 2: If cache empty, try to fetch from SharedState directly
        if not data and hasattr(self.shared_state, "get_market_data_sync"):
            data = self.shared_state.get_market_data_sync(symbol, self.timeframe)
        
        # Step 3: If still empty, fetch directly from exchange
        if not data:
            logger.warning(f"[{self.name}] Cache empty for {symbol} {self.timeframe}, fetching from exchange...")
            try:
                if hasattr(self, 'execution_manager') and hasattr(self.execution_manager, 'exchange_client'):
                    ec = self.execution_manager.exchange_client
                    fetch_limit = max(100, int(self._cfg("SWING_SIGNAL_FETCH_LIMIT", 300) or 300))
                    # Fetch raw klines and convert to OHLCV format
                    raw_klines = await ec.get_klines(symbol, self.timeframe, limit=fetch_limit)
                    if raw_klines:
                        data = [
                            {
                                "ts": float(kline[0]) / 1000,  # Convert ms to seconds
                                "open": float(kline[1]),
                                "high": float(kline[2]),
                                "low": float(kline[3]),
                                "close": float(kline[4]),
                                "volume": float(kline[7]),  # Quote asset volume
                            }
                            for kline in raw_klines
                        ]
                        logger.warning(f"[{self.name}] Fetched {len(data)} candles from exchange for {symbol}")
                        set_md = getattr(self.shared_state, "set_market_data", None)
                        if callable(set_md):
                            compact_rows = [
                                {
                                    "ts": row["ts"],
                                    "o": row["open"],
                                    "h": row["high"],
                                    "l": row["low"],
                                    "c": row["close"],
                                    "v": row["volume"],
                                }
                                for row in data
                            ]
                            await self._await_maybe(set_md(symbol, self.timeframe, compact_rows))
            except Exception as e:
                logger.warning(f"[{self.name}] Failed to fetch from exchange: {e}")
                data = None
        
        logger.warning(f"[{self.name}] Got {len(data) if data else 0} candles for {symbol}")
        rows = self._normalize_ohlcv_rows(data)
        if not rows or len(rows) < 50:
            logger.warning(f"[{self.name}] Insufficient data for {symbol}: {len(rows) if rows else 0} < 50")
            return 'hold', 0.0, 'Insufficient data'

        # --- Fix 2: Placeholder for skipping inference if no model is loaded ---
        # If you decide to use the trained model for signal generation,
        # uncomment and integrate this logic.
        # model = self.model_cache.get(symbol)
        # if not model:
        #     logger.debug(f"[{self.name}] Skipping signal generation for {symbol} - no model loaded for inference.")
        #     return 'hold', 0.0, 'No model for inference'
        # --- End Fix 2 placeholder ---

        closes = np.array([r['close'] for r in rows], dtype=float)
        ema20 = compute_ema(closes, 20)
        ema50 = compute_ema(closes, 50)
        rsi = compute_rsi(closes, 14)
        macd_line, signal_line, hist = compute_macd(closes)
        
        # Validate indicators have values - handle both Series and ndarray
        try:
            ema20_len = len(ema20) if hasattr(ema20, '__len__') else 0
            ema50_len = len(ema50) if hasattr(ema50, '__len__') else 0
            rsi_len = len(rsi) if hasattr(rsi, '__len__') else 0
            hist_len = len(hist) if hasattr(hist, '__len__') else 0
            
            if ema20_len == 0 or ema50_len == 0 or rsi_len == 0 or hist_len == 0:
                logger.warning(f"[{self.name}] {symbol}: insufficient indicator data - skipping")
                return 'hold', 0.0, 'Insufficient indicator data'
        except:
            logger.warning(f"[{self.name}] {symbol}: error validating indicators")
            return 'hold', 0.0, 'Indicator validation error'
        
        # Safe access with bounds checking
        try:
            ema20_val = float(ema20.iloc[-1]) if hasattr(ema20, 'iloc') else float(ema20[-1])
            ema50_val = float(ema50.iloc[-1]) if hasattr(ema50, 'iloc') else float(ema50[-1])
            rsi_val = float(rsi.iloc[-1]) if hasattr(rsi, 'iloc') else float(rsi[-1])
            hist_val = float(hist.iloc[-1]) if hasattr(hist, 'iloc') else float(hist[-1])
        except (IndexError, KeyError, ValueError, AttributeError) as e:
            logger.warning(f"[{self.name}] {symbol}: error accessing indicator values: {e}")
            return 'hold', 0.0, 'Cannot access indicators'
        
        logger.warning(f"[{self.name}] {symbol} indicators: ema20={ema20_val:.4f} ema50={ema50_val:.4f} rsi={rsi_val:.2f} macd_hist={hist_val:.6f}")
        
        # RELAXED SIGNAL CRITERIA: Temporarily lowered thresholds to generate signals during testing
        # Original: EMA20 > EMA50 AND MACD > 0 AND RSI < 70
        # Relaxed: EMA20 > EMA50 AND RSI < 75 (removed MACD check - sometimes conflicting with price)
        
        if ema20_val > ema50_val and rsi_val < 75:
            logger.warning(f"[{self.name}] ✅ BUY SIGNAL for {symbol}: EMA uptrend + RSI favorable (EMA20 > EMA50, RSI < 75)")
            return 'buy', 0.65, 'EMA uptrend detected'
        if ema20_val < ema50_val and rsi_val > 30:
            logger.warning(f"[{self.name}] ✅ SELL SIGNAL for {symbol}: EMA downtrend + RSI unfavorable (EMA20 < EMA50, RSI > 30)")
            return 'sell', 0.65, 'EMA downtrend detected'
        
        logger.warning(f"[{self.name}] ❌ HOLD for {symbol}: no clear signal")
        return 'hold', 0.0, 'No clear signal'

    async def _retrain_async_single(self, symbol: str):
        """
        Non-blocking async retrain for a SINGLE symbol.
        Designed to be called via asyncio.create_task() to train in background
        without blocking the symbol iteration loop.
        """
        if tf is None:
            logger.warning("[%s] Retrain skipped: TensorFlow unavailable.", self.name)
            return
        
        if not symbol:
            logger.warning("[%s] Retrain skipped: no symbol specified.", self.name)
            return
        
        logger.info(f"[{self.name}] 🧠 Background retrain started for {symbol}")
        loop = asyncio.get_event_loop()
        
        try:
            lookback = int(self._cfg("SWING_RETRAIN_LOOKBACK", self._cfg("RETRAIN_LOOKBACK", 100)) or 100)
            min_rows = max(lookback + 50, int(self._cfg("SWING_RETRAIN_MIN_BARS", 180) or 180))
            fetch_limit = max(min_rows + 50, int(self._cfg("SWING_RETRAIN_FETCH_LIMIT", 600) or 600))
            rows = await self._fetch_retrain_rows(symbol, min_rows=min_rows, fetch_limit=fetch_limit)

            # Run the blocking retrain in executor (non-blocking to main loop)
            result = await loop.run_in_executor(None, self._retrain_blocking, symbol, rows)
            ok = bool(result.get("ok")) if isinstance(result, dict) else bool(result)
            if ok:
                self._retrain_last_failure_ts.pop(symbol, None)
                self._retrain_last_success_ts[symbol] = time.time()
                logger.info(f"[{self.name}] ✅ Background retrain completed for {symbol}")
            else:
                self._retrain_last_failure_ts[symbol] = time.time()
                reason = result.get("reason") if isinstance(result, dict) else "train_failed"
                logger.warning(f"[{self.name}] ⚠️ Background retrain did not complete for {symbol} (reason={reason})")
        except Exception as e:
            self._retrain_last_failure_ts[symbol] = time.time()
            logger.error(f"[{self.name}] Background retrain failed for {symbol}: {e}", exc_info=True)
        finally:
            self._retrain_inflight.discard(symbol)

    async def _retrain_async(self, symbol=None):
        """
        Async wrapper for model retraining (BLOCKING variant for explicit waits).
        
        ⚠️ DEPRECATED for per-symbol calls - use _retrain_async_single() instead.
        This awaits ALL retraining to complete before returning.
        Runs blocking training in executor to avoid blocking event loop.
        """
        if tf is None:
            logger.warning("[%s] Retrain skipped: TensorFlow unavailable.", self.name)
            return
        
        symbols = [symbol] if symbol else self.symbols
        loop = asyncio.get_event_loop()
        
        for sym in symbols:
            try:
                # Run the blocking retrain in executor
                await loop.run_in_executor(None, self._retrain_blocking, sym)
            except Exception as e:
                logger.error(f"[{self.name}] Async retrain failed for {sym}: {e}", exc_info=True)

    def _retrain_blocking(self, symbol: str, data: Any = None):
        """
        Synchronous blocking model retraining.
        Should be called via run_in_executor to avoid blocking event loop.
        """
        if tf is None:
            logger.warning("[%s] Retrain skipped: TensorFlow unavailable.", self.name)
            return {"ok": False, "reason": "tensorflow_unavailable"}
        
        try:
            # Use pre-fetched async rows when available, else fall back to sync cache.
            if data is None:
                sync_rows = self.shared_state.get_market_data_sync(symbol, self.timeframe)
                data = self._normalize_ohlcv_rows(sync_rows)
            else:
                data = self._normalize_ohlcv_rows(data)

            lookback = int(self._cfg("SWING_RETRAIN_LOOKBACK", self._cfg("RETRAIN_LOOKBACK", 100)) or 100)
            min_rows = max(lookback + 50, int(self._cfg("SWING_RETRAIN_MIN_BARS", 180) or 180))
            if not data or len(data) < min_rows:
                logger.warning("Cannot retrain %s: insufficient data (rows=%d need>=%d).", symbol, len(data or []), min_rows)
                return {"ok": False, "reason": "insufficient_data", "rows": int(len(data or []))}
            
            # Prepare data X,y similar to MLForecaster
            max_rows = max(min_rows, int(self._cfg("SWING_RETRAIN_MAX_ROWS", 1200) or 1200))
            data = data[-max_rows:]
            X, y = [], []
            for i in range(lookback, len(data)):
                window = data[i-lookback:i]
                X.append([[c['open'], c['high'], c['low'], c['close'], c['volume']] for c in window])
                future = data[i]['close']
                current = window[-1]['close']
                if future > current:
                    y.append([1,0,0])
                elif future < current:
                    y.append([0,1,0])
                else:
                    y.append([0,0,1])

            min_samples = max(32, int(self._cfg("SWING_RETRAIN_MIN_SAMPLES", 64) or 64))
            if len(X) < min_samples:
                logger.warning(
                    "[%s] Cannot retrain %s: insufficient training samples (samples=%d need>=%d).",
                    self.name,
                    symbol,
                    len(X),
                    min_samples,
                )
                return {"ok": False, "reason": "insufficient_samples", "samples": int(len(X))}
            
            X = np.array(X)
            y = np.array(y)
            
            # Build and train model
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(lookback, 5)),
                tf.keras.layers.LSTM(50),
                tf.keras.layers.Dense(3, activation='softmax')
            ])
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            epochs = max(1, int(self._cfg("SWING_RETRAIN_EPOCHS", 5) or 5))
            batch_size = max(8, int(self._cfg("SWING_RETRAIN_BATCH_SIZE", 32) or 32))
            fit_kwargs = {"epochs": epochs, "batch_size": batch_size, "verbose": 0}
            if len(X) >= 100:
                fit_kwargs["validation_split"] = 0.1
            model.fit(X, y, **fit_kwargs)
            
            # Save model
            path = build_model_path(self.name, symbol)
            save_model(model, path)
            self.model_cache[symbol] = model
            logger.info(f"✅ [%s] Retrained and saved model for %s at %s", self.name, symbol, path)
            return {"ok": True, "reason": "trained", "rows": int(len(data)), "samples": int(len(X))}
        except Exception as e:
            logger.error(f"[{self.name}] Blocking retrain failed for {symbol}: {e}", exc_info=True)
            return {"ok": False, "reason": "exception"}

    def retrain(self, symbol=None):
        """
        Synchronous retrain method (for backward compatibility).
        
        ⚠️ WARNING: This blocks the event loop!
        Use _retrain_async() instead when in async context.
        """
        if tf is None:
            logger.warning("[%s] Retrain skipped: TensorFlow unavailable.", self.name)
            return
        symbols = [symbol] if symbol else self.symbols
        for sym in symbols:
            self._retrain_blocking(sym)
