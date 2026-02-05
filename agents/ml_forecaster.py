import os
import asyncio
import logging
import numpy as np
from datetime import datetime
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List, Set
from decimal import Decimal
import inspect  # <-- added

from core.agent_optimizer import load_tuned_params
from core.component_status_logger import log_component_status
from core.stubs import is_fresh  # freshness gate
from core.model_trainer import ModelTrainer

logger = logging.getLogger("MLForecaster")
if not logger.handlers:
    logger.setLevel(logging.INFO)


class MLForecaster:

    async def _get_market_data_safe(self, symbol: str, timeframe: str):
        fn = getattr(self.shared_state, "get_market_data", None)
        if not callable(fn):
            return None
        data = fn(symbol, timeframe)
        if asyncio.iscoroutine(data):
            data = await data
        return data
    """
    ML-based forecaster that PRODUCES signals and delegates execution to MetaController.
    - Emits via meta_controller.receive_signal(name, symbol, payload, confidence)
    - Respects confidence/freshness
    - No direct order placement by default (ALLOW_AGENT_DIRECT_EXECUTION=False)
    """

    agent_type = "strategy"

    def __init__(self, shared_state, execution_manager, config, tp_sl_engine=None, **kwargs):
        self.name = kwargs.get("name", "MLForecaster")
        self.shared_state = shared_state
        self.execution_manager = execution_manager
        self.config = config
        self.tp_sl_engine = tp_sl_engine

        self.logger = logging.getLogger(self.name)
        self.interval = float(getattr(config, "ML_FORECASTER_INTERVAL", 60.0))

        # Freshness & emission controls
        self.fresh_max_age_s = float(getattr(self.config, "FRESH_MAX_AGE_S", 120.0))
        self.allow_sell_without_position = bool(getattr(self.config, "ALLOW_SELL_WITHOUT_POSITION", False))
        # Prevent log spam for "direct exec disabled"
        self._direct_exec_logged: Set[str] = set()

        # Optional deps
        self.market_data_feed = kwargs.get("market_data_feed")
        self.meta_controller = kwargs.get("meta_controller")
        self.model_manager = kwargs.get("model_manager")
        self.symbol_manager = kwargs.get("symbol_manager")
        self.exchange_client = kwargs.get("exchange_client")
        self.database_manager = kwargs.get("database_manager")
        self.agent_schedule = kwargs.get("agent_schedule")

        # Symbols & tf
        self.symbols: List[str] = list(kwargs.get("symbols", []) or [])
        self.timeframe = kwargs.get("timeframe", "5m")

        # Tuned params (safe)
        # Move defining _cfg later or call it via self._cfg
        try:
            tuned_global = load_tuned_params(self.name) or {}
        except Exception:
            tuned_global = {}

        self._tuned_global = tuned_global
        self.window_size = int(tuned_global.get("window_size", self._cfg("WINDOW_SIZE", 60)))

        # Caching / performance
        self.model_cache: Dict[str, Any] = {}
        self._model_mtime: Dict[str, float] = {}
        self._predict_fns: Dict[Tuple[str, int], Any] = {}  # (model_path, lookback) -> tf.function

        # Concurrency & limits
        self.max_concurrency = int(getattr(self.config, "MLF_MAX_CONCURRENCY", 6))
        self.symbol_timeout_s = float(getattr(self.config, "MLF_SYMBOL_TIMEOUT_S", 15.0))
        self.predict_timeout_s = float(getattr(self.config, "MLF_PREDICT_TIMEOUT_S", 5.0))
        self.max_symbols_per_tick = int(getattr(self.config, "MLF_MAX_SYMBOLS_PER_TICK", 50))
        self._sem = asyncio.Semaphore(self.max_concurrency)
        self._stop_event = asyncio.Event()
        
        # ARCHITECTURAL FIX: Signal collection buffer for generate_signals()
        self._collected_signals: List[Dict[str, Any]] = []

    @property
    def min_conf(self) -> float:
        """Dynamic access to minimum signal confidence (Phase A)."""
        return float(self._tuned_global.get("ML_MIN_CONF_EMIT", self._cfg("ML_MIN_CONF_EMIT", 0.55)))

    def _cfg(self, key: str, default: Any = None) -> Any:
        # 1. Check SharedState for live/dynamic overrides
        if hasattr(self.shared_state, "dynamic_config"):
            val = self.shared_state.dynamic_config.get(key)
            if val is not None:
                return val

        # 2. Fallback to static config (env or file)
        if isinstance(self.config, dict):
            return self.config.get(key, default)
        return getattr(self.config, key, default)

    async def _has_position(self, symbol: str) -> bool:
        """Return True if we hold a positive position for symbol. Safe across sync/async branches."""
        qty = 0.0
        try:
            fn = getattr(self.shared_state, "get_position_qty", None) or getattr(self.shared_state, "get_position_quantity", None)
            if callable(fn):
                res = fn(symbol)
                qty = await res if asyncio.iscoroutine(res) else (res or 0.0)
        except Exception:
            qty = 0.0
        try:
            return float(qty) > 0
        except Exception:
            return False

        log_component_status(self.name, "Initialized")
        self.logger.info(
            f"{self.name} initialized (tf={self.timeframe}, interval={self.interval}s, "
            f"conc={self.max_concurrency}, symbol_timeout={self.symbol_timeout_s}s)"
        )

    # ---------------- Lifecycle ----------------

    async def stop(self):
        self.logger.info(f"🚓 {self.name} stopping...")
        self._stop_event.set()

    # ARCHITECTURAL FIX: run_loop() REMOVED
    # Strategy agents MUST NOT self-schedule
    # AgentManager calls generate_signals() on a central tick

    async def generate_signals(self):
        """Generate signals for all symbols. Called by AgentManager."""
        # Collect signals from run_once()
        self._collected_signals = []
        await self.run_once()
        # Return collected signals
        signals = self._collected_signals
        self._collected_signals = []
        return signals

    # ---------------- Core pass ----------------

    async def run_once(self):
        # Early skip if market data is not ready
        self.logger.info(f"[{self.name}] run_once starting. SharedState ID: {id(self.shared_state)}")
        if hasattr(self.shared_state, "is_market_data_ready"):
            try:
                if not self.shared_state.is_market_data_ready():
                    self.logger.warning(f"[{self.name}] Market data not ready; skipping tick.")
                    return
            except Exception:
                pass
        self.logger.debug(f"[{self.name}] Executing run_once.")
        if True:
            syms = await self._safe_get_symbols()
            if set(syms) != set(self.symbols):
                self.symbols = list(syms)
                self.logger.info(f"[{self.name}] Updated symbols: {len(self.symbols)}")
        if not self.symbols:
            self.logger.warning(f"[{self.name}] No symbols to process.")
            return

        # ARCHITECTURAL FIX: Process ALL symbols, not a subset
        # AgentManager contract: generate_signals() must scan ALL symbols per tick
        batch = self.symbols  # Removed slicing: was [:self.max_symbols_per_tick]

        async def _one(sym: str):
            async with self._sem:
                try:
                    return await asyncio.wait_for(self.run(sym), timeout=self.symbol_timeout_s)
                except asyncio.TimeoutError:
                    self.logger.warning(
                        f"[{self.name}] {sym} timed out after {self.symbol_timeout_s}s; dropping to NO_DECISION."
                    )
                    return {"action": "hold", "confidence": 0.0, "reason": "timeout"}
                except Exception:
                    self.logger.exception("[%s] %s processing failed; dropping to NO_DECISION.", self.name, sym)
                    return {"action": "hold", "confidence": 0.0, "reason": "processing_error"}

        # Capture results (currently unused but good practice)
        results = await asyncio.gather(*[_one(s) for s in batch], return_exceptions=False)
        self.logger.debug(f"[{self.name}] Finished run_once (processed={len(batch)} symbols, generated={len(self._collected_signals)} signals).")

    # ---------------- Helpers ----------------

    async def _safe_get_symbols(self) -> List[str]:
        """
        SharedState.get_accepted_symbols may return dict/list and may be sync/async across branches.
        Normalize to List[str].
        """
        try:
            getter = getattr(self.shared_state, "get_accepted_symbols", None)
            if callable(getter):
                res = getter()
                if asyncio.iscoroutine(res):
                    res = await res
            else:
                res = []
            if isinstance(res, dict):
                return list(res.keys())
            return list(res or [])
        except Exception:
            return []

    def _std_row(self, r) -> Optional[List[float]]:
        """
        Accept either short-key (o/h/l/c/v) or long-key (open/high/low/close/volume) dicts,
        or a sequence ending with [open, high, low, close, volume].
        """
        try:
            if isinstance(r, dict):
                d = r.get("k", r)
                o = d.get("o", d.get("open"))
                h = d.get("h", d.get("high"))
                l = d.get("l", d.get("low"))
                c = d.get("c", d.get("close", d.get("last", d.get("price"))))
                v = d.get("v", d.get("volume"))
                if None in (o, h, l, c, v):
                    return None
                return [float(o), float(h), float(l), float(c), float(v)]
            else:
                seq = list(r)
                if len(seq) >= 6:
                    seq = seq[-5:]  # keep last 5 numbers
                if len(seq) == 5:
                    return [float(x) for x in seq]
        except Exception:
            return None
        return None

    async def _collect_signal(self, symbol: str, action: str, confidence: float, reason: str, extras: Optional[Dict[str, Any]] = None):
        """
        ARCHITECTURAL FIX: Build signal dict instead of emitting to Meta.
        AgentManager will collect these and forward them.
        """
        if action.upper() not in ("BUY", "SELL"):
            return
            
        # DYNAMIC THRESHOLD (P9 Profit Feedback)
        # Check system-wide aggression to relax standards if behind profit target
        agg_factor = 1.0
        if hasattr(self.shared_state, "get_dynamic_param"):
            agg_factor = float(self.shared_state.get_dynamic_param("aggression_factor", 1.0))
            
        effective_min_conf = float(self.min_conf)
        if agg_factor > 1.0:
            effective_min_conf = max(0.40, effective_min_conf / agg_factor) # Don't go below 0.40

        # Optional SELL-specific confidence floor (to enable SELL path without loosening BUYs)
        if action.upper() == "SELL":
            effective_min_conf = float(
                self._cfg("ML_MIN_CONF_EMIT_SELL", self._cfg("SELL_MIN_CONF", effective_min_conf))
            )
            
        if float(confidence) < effective_min_conf:
            return
        if not await is_fresh(self.shared_state, symbol, max_age_sec=self.fresh_max_age_s):
            self.logger.debug(f"[{self.name}] Stale; skip {symbol}")
            return

        # Optional SELL guard
        if action.upper() == "SELL" and not self.allow_sell_without_position:
            try:
                if not await self._has_position(symbol):
                    self.logger.info(f"[{self.name}] Skip SELL for {symbol} — no position (signal suppressed).")
                    return
            except Exception:
                pass
                
        # Quote hint
        qh = getattr(
            self.config,
            "EMIT_BUY_QUOTE",
            getattr(self.config, "MIN_ENTRY_USDT", getattr(self.config, "DEFAULT_PLANNED_QUOTE", 25.0)),
        )
        try:
            if isinstance(qh, dict):
                qh = float(qh.get(symbol, qh.get("default", 25.0)))
            else:
                qh = float(qh)
        except Exception:
            qh = 25.0

        # Mandatory P9 Signal Contract: Emit to Signal Bus
        if hasattr(self.shared_state, "add_agent_signal"):
            try:
                # determine tier based on confidence or config
                tier = "A" if confidence >= 0.85 else "B"
                await self.shared_state.add_agent_signal(
                    symbol=symbol,
                    agent=self.name,
                    side=action.upper(),
                    confidence=confidence,
                    ttl_sec=300,
                    tier=tier,
                    rationale=reason
                )
            except Exception as e:
                self.logger.warning(f"[{self.name}] Failed to emit to signal bus: {e}")

        # Build signal dictionary
        signal = {
            "symbol": symbol,
            "action": action.upper(),
            "side": action.upper(),  # Alias for compatibility
            "confidence": max(0.0, min(1.0, float(confidence or 0.0))),
            "reason": reason,
            "quote": qh,
            "quote_hint": qh,  # Alias
            "horizon_hours": float(getattr(self.config, "DEFAULT_SIGNAL_HORIZON_H", 6.0)),
            "agent": self.name,
        }
        if extras:
            signal.update(extras)

        # GAP FIX A: Validate quote against min_notional BEFORE buffering
        # Prevents sub-5 USDT signals from reaching MetaController
        MIN_NOTIONAL_FLOOR = float(getattr(self.config, "MIN_NOTIONAL_FLOOR", 5.0))
        if signal["quote"] < MIN_NOTIONAL_FLOOR * 0.8:  # 80% headroom for fees
            self.logger.warning(
                f"[{self.name}] Signal quote {signal['quote']:.2f} < min_notional {MIN_NOTIONAL_FLOOR:.2f}; filtering out"
            )
            return  # Don't emit sub-minimum signals

        # Add to collection buffer (AgentManager will forward to Meta)
        self._collected_signals.append(signal)
        self.logger.info(f"[{self.name}] SIGNAL: {symbol} {signal['action']} conf={confidence:.2f}")

    # ---------------- Run per symbol ----------------

    async def run(self, symbol: str):
        cur_sym = symbol.upper()
        cur_tf = self.timeframe
        self.logger.debug(f"🚀 {self.name} run for {cur_sym} @ {cur_tf}")

        # Ensure model path and load/train
        model_path = self.model_manager.build_model_path(agent_name=self.name, symbol=cur_sym, version=cur_tf)

        model = None
        if not self.model_manager.model_exists(model_path):
            self.logger.warning(f"[{self.name}] 🚧 Model not found for {cur_sym}. Initiating training...")
            raw_data = await self._get_market_data_safe(cur_sym, cur_tf)
            if not raw_data or len(raw_data) < 100:
                self.logger.error(f"[{self.name}] ❌ Not enough data to train model for {cur_sym}. Skipping.")
                return {"action": "hold", "confidence": 0.0, "reason": "Insufficient data for training"}

            # Training off-loop
            try:
                df = pd.DataFrame(raw_data)
                if {"o","h","l","c","v"}.issubset(df.columns):
                    df = df.rename(columns={"o":"open","h":"high","l":"low","c":"close","v":"volume"})
                required = {"open","high","low","close","volume"}
                if not required.issubset(df.columns):
                    raise ValueError("DataFrame missing required OHLCV columns")
            except Exception as e:
                self.logger.error(f"[{self.name}] ❌ Data formatting error for {cur_sym}: {e}")
                return {"action": "hold", "confidence": 0.0, "reason": "Data formatting error"}

            trainer = ModelTrainer(symbol=cur_sym, timeframe=cur_tf, agent_name=self.name, model_manager=self.model_manager)
            loop = asyncio.get_running_loop()
            ok = await loop.run_in_executor(None, trainer.train_model, df)
            if not ok:
                self.logger.error(f"[{self.name}] ❌ Training failed for {cur_sym}.")
                return {"action": "hold", "confidence": 0.0, "reason": "Model training failed"}

        # Cached load with mtime guard
        try:
            mtime = os.path.getmtime(model_path)
        except Exception:
            mtime = 0.0
        cached = self.model_cache.get(model_path)
        if cached and self._model_mtime.get(model_path) == mtime:
            model = cached
        else:
            model = self.model_manager.safe_load_model(model_path)
            if model is not None:
                self.model_cache[model_path] = model
                self._model_mtime[model_path] = mtime

        # Per-symbol tuned params
        tuned = {}
        try:
            tuned = load_tuned_params(f"{self.name}_{cur_sym}_{cur_tf}") or {}
        except Exception:
            tuned = {}
        lookback = int(tuned.get("lookback", getattr(self.config, "LOOKBACK", 20)))
        confidence_threshold = float(tuned.get("confidence_threshold", getattr(self.config, "CONFIDENCE_THRESHOLD", 0.7)))
        is_active = bool(tuned.get("active", True))
        self.logger.debug(f"[{self.name}] Tuned for {cur_sym}: lookback={lookback}, active={is_active}")

        if not is_active:
            self.logger.info(f"⚠️ {self.name} for {cur_sym}@{cur_tf} is deactivated.")
            return {"action": "hold", "confidence": 0.0, "reason": "Deactivated"}

        # Data fetch
        try:
            ohlcv = await self._get_market_data_safe(cur_sym, cur_tf)
            if ohlcv is None:
                return {"action": "hold", "confidence": 0.0, "reason": "OHLCV None"}
            if not isinstance(ohlcv, list) or len(ohlcv) < lookback:
                return {"action": "hold", "confidence": 0.0, "reason": "Insufficient OHLCV"}
        except Exception as e:
            self.logger.error(f"❌ OHLCV fetch failed for {cur_sym}@{cur_tf}: {e}", exc_info=True)
            return {"action": "hold", "confidence": 0.0, "reason": "Data fetch error"}

        # Build input
        try:
            rows = [self._std_row(c) for c in (ohlcv[-lookback:] or [])]
            rows = [r for r in rows if r is not None]
            if len(rows) < lookback:
                return {"action": "hold", "confidence": 0.0, "reason": "Insufficient OHLCV"}
            X = np.asarray(rows, dtype=np.float32).reshape((1, lookback, 5))
        except Exception as e:
            self.logger.error(f"❌ Input formatting failed for {cur_sym}: {e}", exc_info=True)
            return {"action": "hold", "confidence": 0.0, "reason": "Input formatting error"}

        # Predict
        try:
            if model is None:
                return {"action": "hold", "confidence": 0.0, "reason": "Model not available"}

            import tensorflow as tf  # local import to avoid import-time cost when unused
            key = (model_path, lookback)
            predict_fn = self._predict_fns.get(key)
            if predict_fn is None:
                spec = tf.TensorSpec(shape=[None, lookback, 5], dtype=tf.float32)
                @tf.function(input_signature=[spec], reduce_retracing=True)
                def _predict_fn(x):
                    return model(x, training=False)
                self._predict_fns[key] = _predict_fn
                predict_fn = _predict_fn

            # Safety: ensure cached function was built with same lookback
            try:
                _ = predict_fn.get_concrete_function(tf.TensorSpec(shape=[None, lookback, 5], dtype=tf.float32))
            except Exception:
                # Rebuild if signature mismatched for any reason
                @tf.function(input_signature=[spec], reduce_retracing=True)
                def _predict_fn(x):
                    return model(x, training=False)
                self._predict_fns[key] = _predict_fn
                predict_fn = _predict_fn

            x_tf = tf.convert_to_tensor(X, dtype=tf.float32)
            x_tf = tf.ensure_shape(x_tf, [None, lookback, 5])

            # Small tensors; inline call is fine
            y_raw = predict_fn(x_tf).numpy()[0]
            
            # P9 Normalized Confidence: Apply softmax if output is likely logits
            def softmax(x):
                e_x = np.exp(x - np.max(x))
                return e_x / e_x.sum(axis=0)
            
            y = softmax(y_raw)
            idx = int(np.argmax(y))
            # P9: Explicit clip to [0.0, 1.0] for absolute safety
            confidence = float(np.clip(np.max(y), 0.0, 1.0))
            action = ["buy", "sell", "hold"][idx]
            
            if np.max(y_raw) > 1.0 or np.min(y_raw) < 0.0:
                self.logger.info(f"[{self.name}] {cur_sym}: logits={y_raw} -> action={action}, conf={confidence:.2f} (normalized)")
            else:
                self.logger.info(f"[{self.name}] {cur_sym}: action={action}, conf={confidence:.2f}")
        except Exception as e:
            self.logger.error(f"❌ Prediction failed for {cur_sym}: {e}", exc_info=True)
            return {"action": "hold", "confidence": 0.0, "reason": "Prediction failure"}

        # Filters (sentiment / regime / CoT echo if present)
        try:
            sentiment_val = await self.shared_state.get_sentiment(cur_sym)
        except Exception:
            sentiment_val = None
        sentiment = float(sentiment_val) if sentiment_val is not None else 0.0

        reginfo = None
        try:
            reginfo = await self.shared_state.get_volatility_regime(cur_sym, timeframe=self.timeframe)
        except Exception:
            reginfo = None
        regime = str((reginfo or {}).get("regime", "normal")).lower()

        cot_txt = ""
        try:
            get_cot = getattr(self.shared_state, "get_cot_explanation", None)
            if callable(get_cot):
                cot_txt = (await get_cot(cur_sym, self.name)) or ""
        except Exception:
            cot_txt = ""

        cot_num = 1 if "yes" in cot_txt.lower() else (-1 if "no" in cot_txt.lower() else 0)
        self.logger.info(
            f"[{self.name}] Filters: Sentiment={sentiment:.2f}, VolatilityRegime={regime}, CoT='{cot_txt}', CoT_Numeric={cot_num}"
        )

        if (sentiment < -0.5) or (regime == "high") or (cot_num < 0):
            action = "hold"

        self.logger.info(f"[{self.name}] Final decision for {cur_sym} => Action: {action.upper()}, Confidence: {confidence:.2f}")

        # Store a brief CoT/debug explanation for UI/debuggers
        try:
            await self.shared_state.set_cot_explanation(
                cur_sym,
                text=f"Pred={action} conf={confidence:.2f} on features shape={X.shape}",
                source=self.name,
            )
        except Exception:
            pass

        # --------- Emission (signal-only) ---------
        await self._collect_signal(
            symbol=cur_sym,
            action=action,
            confidence=confidence,
            reason="ML model prediction",
        )

        # Optional direct execution (OFF by default)
        if action in ("buy", "sell") and confidence >= float(confidence_threshold):
            # Do not even consider direct exec if data is stale
            if not await is_fresh(self.shared_state, cur_sym, max_age_sec=self.fresh_max_age_s):
                return {"action": action, "confidence": confidence, "reason": "stale_data_no_exec"}
            if not getattr(self.config, "ALLOW_AGENT_DIRECT_EXECUTION", False):
                if cur_sym not in self._direct_exec_logged:
                    self.logger.info(f"[{self.name}] Direct execution disabled by config; signal-only for {cur_sym}.")
                    self._direct_exec_logged.add(cur_sym)
                return {"action": action, "confidence": confidence, "reason": "Signal only"}
            # If you *really* want direct exec, leave the block below; otherwise it remains unused.
            try:
                # Price fetch fallback chain
                get_price_cb = None
                try:
                    get_price_cb = self.execution_manager.exchange_client.get_current_price
                except Exception:
                    get_price_cb = getattr(self.execution_manager, "get_current_price", None)

                price = await self.shared_state.get_or_fetch_price(cur_sym, get_price_cb)
                if not price or price <= 0:
                    return {"action": "hold", "confidence": 0.0, "reason": "price_unavailable"}

                # Spendable quote after reservations
                if action == "sell":
                    has_pos = await self._has_position(cur_sym)
                    if not has_pos:
                        self.logger.info(f"[{self.name}] Skip SELL for {cur_sym} — no position.")
                        return {"action": "hold", "confidence": 0.0, "reason": "no_position"}
                    # Use latest known position quantity
                    pos_qty = 0.0
                    try:
                        fnq = getattr(self.shared_state, "get_position_qty", None) or getattr(self.shared_state, "get_position_quantity", None)
                        if callable(fnq):
                            resq = fnq(cur_sym)
                            pos_qty = await resq if asyncio.iscoroutine(resq) else (resq or 0.0)
                    except Exception:
                        pos_qty = 0.0
                    qty = Decimal(str(round(float(pos_qty), 6)))
                else:
                    # Compute spendable size in BASE via spendable USDT
                    usdt_free = 0.0
                    try:
                        fnf = getattr(self.shared_state, "get_spendable_quote", None)
                        if callable(fnf):
                            rf = fnf("USDT")
                            usdt_free = await rf if asyncio.iscoroutine(rf) else (rf or 0.0)
                    except Exception:
                        usdt_free = 0.0
                    pct = float(getattr(self.config, "DIRECT_EXEC_BUY_PCT", 0.10))
                    qty_float = round(((usdt_free / price) * pct), 8) if (usdt_free and usdt_free > 10 and price and price > 0) else 0.0
                    qty = Decimal(str(qty_float))

                ok = True
                normalized_qty = float(qty)
                reason_msg = ""
                try:
                    fnv = getattr(self.shared_state, "is_trade_notional_valid", None)
                    if callable(fnv):
                        vr = fnv(cur_sym, qty, action, price, get_price_fallback=get_price_cb)
                        ok, normalized_qty, reason_msg = await vr if asyncio.iscoroutine(vr) else vr
                except Exception as _e:
                    ok, normalized_qty, reason_msg = True, float(qty), "validation_skipped"
                if not ok:
                    self.logger.info(f"⛔ {cur_sym} trade skipped – validation_failed ({reason_msg}).")
                    return {"action": "hold", "confidence": 0.0, "reason": f"validation_failed:{reason_msg}"}

                qty = Decimal(str(normalized_qty))
                if qty <= 0:
                    return {"action": "hold", "confidence": 0.0, "reason": "qty_zero_after_normalization"}

                # Optional TP/SL
                tp, sl = (None, None)
                if self.tp_sl_engine:
                    tp, sl = self.tp_sl_engine.calculate_tp_sl(cur_sym, price)

                trade_result = await self.execution_manager.execute_trade(
                    symbol=cur_sym,
                    side=action,
                    qty=float(qty),
                    mode="market",
                    take_profit=tp,
                    stop_loss=sl,
                    comment=f"{self.name}_strategy",
                )
                self.logger.info(f"[{self.name}] ✅ Executed trade: {trade_result}")
            except Exception as e:
                self.logger.error(f"❌ Trade execution failed for {cur_sym}: {e}", exc_info=True)

        return {"action": action, "confidence": confidence, "reason": "Prediction processed"}

