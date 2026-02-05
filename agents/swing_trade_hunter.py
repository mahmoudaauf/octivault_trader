import asyncio
import logging
import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from functools import partial

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
        self.execution_manager = execution_manager
        self.config = config
        self.tp_sl_engine = tp_sl_engine
        self.model_manager = model_manager
        self.name = name
        self.timeframe = timeframe
        # Modified line: Use get_accepted_symbols() from shared_state
        self.symbols = symbols or self.shared_state.get_accepted_symbols()
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

    async def run_once(self):
        logger.info(f"[{self.name}] Entering run_once loop.")
        if not getattr(self.shared_state, 'initial_market_data_loaded', False):
            logger.warning(f"[{self.name}] Market data not ready. Skipping run.")
            return
        if not self.symbols:
            logger.info(f"[{self.name}] No symbols configured. Skipping.")
            return
        for symbol in self.symbols:
            await self._process_symbol(symbol)
        logger.info(f"[{self.name}] Exiting run_once loop.")

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

        # Execute trade if actionable
        if action in ['buy', 'sell'] and confidence > getattr(self.config, 'SWING_MIN_CONFIDENCE', 0.5):
            price = await self.shared_state.get_latest_price(symbol)
            balance = self.shared_state.balances.get(self.config.BASE_CURRENCY, {}).get('free', 0)
            qty = round(balance / price * 0.1, 6) if price and balance > 0 else 0
            tp, sl = self.tp_sl_engine.calculate_tp_sl(symbol, price)
            result = await self.execution_manager.execute_trade(
                symbol=symbol,
                side=action,
                qty=qty,
                mode='market',
                take_profit=tp,
                stop_loss=sl,
                comment=f"{self.name}_trade"
            )
            entry = result.get('avg_price_entry', price)
            exit_price = result.get('avg_price_exit', price)
            filled = result.get('filled_qty', 0)
            pnl = filled * (exit_price - entry) if action == 'buy' else filled * (entry - exit_price)
            success = pnl > 0
            self.trades_count += 1
            if success:
                self.win_count += 1
                logger.info(f"[{self.name}] Trade success on {symbol}: PnL={pnl:.4f}")
            else:
                self.loss_count += 1
                logger.info(f"[{self.name}] Trade failure on {symbol}: PnL={pnl:.4f}")

            # Update agent_scores
            stats = self.shared_state.agent_scores.setdefault(self.name, {}).setdefault(symbol, {})
            stats['trades'] = self.trades_count
            stats['wins'] = self.win_count
            stats['losses'] = self.loss_count
            stats['win_rate'] = self.win_count / self.trades_count if self.trades_count else 0

        # Update shared state health
        await self.shared_state.update_system_health(
            self.name,
            'Operational',
            f"Last signal: {action} ({confidence:.2f})"
        )

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
