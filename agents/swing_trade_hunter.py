import asyncio
import logging
import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from functools import partial
from typing import Any, Dict, List

from utils.indicators import compute_ema, compute_rsi, compute_macd, compute_bollinger_bands
from utils.status_logger import log_component_status
from utils.shared_state_tools import inject_agent_signal
from core.model_manager import safe_load_model, save_model, build_model_path

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
        tp_sl_engine,
        model_manager,
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

    async def generate_signals(self) -> List[Dict[str, Any]]:
        """
        Canonical strategy contract:
        - Generate and return signal payloads
        - Do not execute trades directly
        """
        self._collected_signals = []
        await self.run_once()
        signals = self._collected_signals
        self._collected_signals = []
        return signals

    async def run_once(self):
        logger.info(f"[{self.name}] Entering run_once loop.")
        await self._refresh_symbols()
        if not getattr(self.shared_state, 'initial_market_data_loaded', False):
            logger.warning(f"[{self.name}] Market data not ready. Skipping run.")
            return
        if not self.symbols:
            logger.info(f"[{self.name}] No symbols configured. Skipping.")
            return
        for symbol in self.symbols:
            await self._process_symbol(symbol)
        logger.info(f"[{self.name}] Exiting run_once loop.")

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

        # ✅ Auto-train model if missing
        if self.model_cache.get(symbol) is None:
            logger.info(f"[{self.name}] 🧠 Auto-training missing model for {symbol}...")
            self.retrain(symbol) # Call retrain method

        # Generate signal
        action, confidence, reason = await self._generate_signal(symbol)
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
            await self._submit_signal(symbol, action.upper(), float(confidence), str(reason))
        else:
            logger.debug(
                "[%s] Not emitting actionable signal for %s (action=%s conf=%.2f min=%.2f)",
                self.name,
                symbol,
                action,
                float(confidence or 0.0),
                min_conf,
            )

        # Update shared state health
        await self.shared_state.update_system_health(
            self.name,
            'Operational',
            f"Last signal: {action} ({confidence:.2f})"
        )

    async def _submit_signal(self, symbol: str, action: str, confidence: float, reason: str) -> None:
        action_u = str(action or "").upper().strip()
        if action_u not in {"BUY", "SELL"}:
            return

        if action_u == "SELL":
            try:
                get_qty = getattr(self.shared_state, "get_position_quantity", None)
                if callable(get_qty):
                    res = get_qty(symbol)
                    pos_qty = await res if asyncio.iscoroutine(res) else float(res or 0.0)
                    if pos_qty <= 0:
                        logger.info("[%s] Skip SELL for %s — no position.", self.name, symbol)
                        return
            except Exception:
                pass

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

        if hasattr(self.shared_state, "add_agent_signal"):
            try:
                tier = "A" if float(confidence) >= 0.85 else "B"
                await self.shared_state.add_agent_signal(
                    symbol=symbol,
                    agent=self.name,
                    side=action_u,
                    confidence=float(confidence),
                    ttl_sec=300,
                    tier=tier,
                    rationale=reason,
                )
            except Exception as e:
                logger.warning("[%s] Failed to emit to signal bus for %s: %s", self.name, symbol, e)

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

    async def _generate_signal(self, symbol):
        # Fetch market data
        data = self.market_data.get_market_data(symbol, self.timeframe)
        if not data or len(data) < 50:
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
        # Simple logic
        if ema20[-1] > ema50[-1] and hist[-1] > 0 and rsi[-1] < 70:
            return 'buy', 0.8, 'Bullish crossover'
        if ema20[-1] < ema50[-1] and hist[-1] < 0 and rsi[-1] > 30:
            return 'sell', 0.8, 'Bearish crossover'
        return 'hold', 0.0, 'No clear signal'

    def retrain(self, symbol=None):
        """
        Retrains model on historical market data.
        """
        symbols = [symbol] if symbol else self.symbols
        for sym in symbols:
            data = self.market_data.get_market_data(sym, self.timeframe)
            if len(data) < getattr(self.config, 'RETRAIN_LOOKBACK', 100):
                logger.warning(f"Cannot retrain {sym}: insufficient data.")
                continue
            # prepare data X,y similar to MLForecaster
            lookback = getattr(self.config, 'RETRAIN_LOOKBACK', 100)
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
            X = np.array(X)
            y = np.array(y)
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(lookback, 5)),
                tf.keras.layers.LSTM(50),
                tf.keras.layers.Dense(3, activation='softmax')
            ])
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            model.fit(X, y, epochs=5, batch_size=32, verbose=0)
            # save
            path = build_model_path(self.name, sym)
            save_model(model, path)
            self.model_cache[sym] = model
            logger.info(f"Retrained and saved model for {sym} at {path}")
