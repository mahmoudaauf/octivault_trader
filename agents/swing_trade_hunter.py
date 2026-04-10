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
from typing import Any, Dict, List

from utils.indicators import compute_ema, compute_rsi, compute_macd, compute_bollinger_bands
try:
    from utils.status_logger import log_component_status
except Exception:
    def log_component_status(*args, **kwargs):
        return None
from utils.shared_state_tools import inject_agent_signal
from core.model_manager import safe_load_model, save_model, build_model_path
from core.stubs import TradeIntent

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
            
            # ✅ FIX: Use readiness event instead of data existence check
            try:
                if hasattr(self.shared_state, "is_market_data_ready"):
                    ready = await self._await_maybe(self.shared_state.is_market_data_ready())
                    if not ready:
                        logger.warning(f"[{self.name}] Market data not ready. Skipping run.")
                        return
            except Exception:
                pass
            
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
            logger.info(f"[{self.name}] 🧠 Auto-training missing model for {symbol}... (background)")
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

        # Update shared state health
        await self.shared_state.update_system_health(
            self.name,
            'Operational',
            f"Last signal: {action} ({confidence:.2f})"
        )

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
        }
        self._collected_signals.append(signal)
        logger.info("[%s] Buffered %s for %s (conf=%.2f)", self.name, action_u, symbol, float(confidence))
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
        
        # Fetch market data (sync call - get_market_data is just a cache lookup)
        logger.warning(f"[{self.name}] Fetching market data for {symbol} on {self.timeframe}")
        # Use get_market_data_sync since get_market_data() is not truly async (just returns cached data)
        if hasattr(self.market_data, "get_market_data_sync"):
            data = self.market_data.get_market_data_sync(symbol, self.timeframe)
        else:
            # Fallback: await the async method if available
            data = await self.market_data.get_market_data(symbol, self.timeframe) if hasattr(self.market_data, "get_market_data") else None
        logger.warning(f"[{self.name}] Got {len(data) if data else 0} candles for {symbol}")
        if not data or len(data) < 50:
            logger.warning(f"[{self.name}] Insufficient data for {symbol}: {len(data) if data else 0} < 50")
            return 'hold', 0.0, 'Insufficient data'

        # --- Fix 2: Placeholder for skipping inference if no model is loaded ---
        # If you decide to use the trained model for signal generation,
        # uncomment and integrate this logic.
        # model = self.model_cache.get(symbol)
        # if not model:
        #     logger.debug(f"[{self.name}] Skipping signal generation for {symbol} - no model loaded for inference.")
        #     return 'hold', 0.0, 'No model for inference'
        # --- End Fix 2 placeholder ---

        closes = np.array([c['close'] for c in data], dtype=float)
        ema20 = compute_ema(closes, 20)
        ema50 = compute_ema(closes, 50)
        rsi = compute_rsi(closes, 14)
        macd_line, signal_line, hist = compute_macd(closes)
        
        logger.warning(f"[{self.name}] {symbol} indicators: ema20={ema20[-1]:.4f} ema50={ema50[-1]:.4f} rsi={rsi[-1]:.2f} macd_hist={hist[-1]:.6f}")
        
        # Simple logic
        if ema20[-1] > ema50[-1] and hist[-1] > 0 and rsi[-1] < 70:
            logger.warning(f"[{self.name}] ✅ BUY SIGNAL for {symbol}: bullish crossover (EMA20 > EMA50, MACD > 0, RSI < 70)")
            return 'buy', 0.8, 'Bullish crossover'
        if ema20[-1] < ema50[-1] and hist[-1] < 0 and rsi[-1] > 30:
            logger.warning(f"[{self.name}] ✅ SELL SIGNAL for {symbol}: bearish crossover (EMA20 < EMA50, MACD < 0, RSI > 30)")
            return 'sell', 0.8, 'Bearish crossover'
        
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
            # Run the blocking retrain in executor (non-blocking to main loop)
            await loop.run_in_executor(None, self._retrain_blocking, symbol)
            logger.info(f"[{self.name}] ✅ Background retrain completed for {symbol}")
        except Exception as e:
            logger.error(f"[{self.name}] Background retrain failed for {symbol}: {e}", exc_info=True)

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

    def _retrain_blocking(self, symbol: str):
        """
        Synchronous blocking model retraining.
        Should be called via run_in_executor to avoid blocking event loop.
        """
        if tf is None:
            logger.warning("[%s] Retrain skipped: TensorFlow unavailable.", self.name)
            return
        
        try:
            # Use synchronous market data access for blocking context
            data = self.shared_state.get_market_data_sync(symbol, self.timeframe)
            if not data or len(data) < getattr(self.config, 'RETRAIN_LOOKBACK', 100):
                logger.warning(f"Cannot retrain {symbol}: insufficient data.")
                return
            
            # Prepare data X,y similar to MLForecaster
            lookback = getattr(self.config, 'RETRAIN_LOOKBACK', 100)
            X, y = [], []
            for i in range(lookback, len(data)):
                window = data[i-lookback:i]
                X.append([[c['o'], c['h'], c['l'], c['c'], c['v']] for c in window])
                future = data[i]['c']
                current = window[-1]['c']
                if future > current:
                    y.append([1,0,0])
                elif future < current:
                    y.append([0,1,0])
                else:
                    y.append([0,0,1])
            
            X = np.array(X)
            y = np.array(y)
            
            # Build and train model
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(lookback, 5)),
                tf.keras.layers.LSTM(50),
                tf.keras.layers.Dense(3, activation='softmax')
            ])
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            model.fit(X, y, epochs=5, batch_size=32, verbose=0)
            
            # Save model
            path = build_model_path(self.name, symbol)
            save_model(model, path)
            self.model_cache[symbol] = model
            logger.info(f"✅ [%s] Retrained and saved model for %s at %s", self.name, symbol, path)
        except Exception as e:
            logger.error(f"[{self.name}] Blocking retrain failed for {symbol}: {e}", exc_info=True)

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
