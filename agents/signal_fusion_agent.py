# agents/signal_fusion_agent.py
import logging
import asyncio
import numpy as np
from collections import deque
import time # Using time.time() for timestamps, could use datetime

from core.shared_state import SharedState
from core.config import Config
from core.exchange_client import ExchangeClient # Using base class, assume specific client is passed

logger = logging.getLogger("SignalFusion") # Updated logger name

class SignalFusion: # Class name changed from SignalFusionAgent to SignalFusion
    def __init__(self, shared_state: SharedState, config: Config, exchange_client: ExchangeClient):
        self.shared_state = shared_state
        self.config = config
        self.exchange_client = exchange_client # Not directly used in this simple example, but common dependency
        self.logger = logging.getLogger("SignalFusion") # Updated logger name
        self.trading_enabled = getattr(config, "SIGNAL_FUSION_AGENT_ENABLED", False) # Default to False if not in config

        # Configuration for SMA strategy (example)
        self.short_sma_period = getattr(config, "SFA_SHORT_SMA_PERIOD", 10) # SFA: Signal Fusion Agent
        self.long_sma_period = getattr(config, "SFA_LONG_SMA_PERIOD", 30)
        self.required_candles = max(self.short_sma_period, self.long_sma_period) # Minimum candles needed for SMA
        self.signal_confidence_threshold = getattr(config, "SFA_SIGNAL_CONFIDENCE_THRESHOLD", 0.6) # Min confidence to send a strong signal

        self.symbol = "N/A"

        self.logger.info("SignalFusion initialized. Enabled: %s", self.trading_enabled) # Updated log message

    async def _generate_signal(self, symbol: str, interval: str) -> dict:
        candles = self.shared_state.get_candlestick_data(symbol, interval)

        if not candles or len(candles) < self.required_candles:
            self.logger.debug(f"Not enough {interval} candles for {symbol} ({len(candles)}/{self.required_candles}). Holding.")
            return {"action": "HOLD", "confidence": 0.0, "timestamp": time.time()}

        close_prices = np.array([c['close'] for c in candles])

        if len(close_prices) < self.long_sma_period:
            self.logger.debug(f"Not enough close prices for SMAs for {symbol}. Holding.")
            return {"action": "HOLD", "confidence": 0.0, "timestamp": time.time()}

        short_sma = np.mean(close_prices[-self.short_sma_period:])
        long_sma = np.mean(close_prices[-self.long_sma_period:])

        if len(close_prices) < self.long_sma_period + 1:
            self.logger.debug(f"Not enough historical data for previous SMA values for {symbol}. Holding.")
            return {"action": "HOLD", "confidence": 0.0, "timestamp": time.time()}

        prev_short_sma = np.mean(close_prices[-(self.short_sma_period + 1):-1])
        prev_long_sma = np.mean(close_prices[-(self.long_sma_period + 1):-1])

        action = "HOLD"
        confidence = 0.0
        reason = "No strong signal"

        if short_sma > long_sma and prev_short_sma <= prev_long_sma:
            action = "BUY"
            confidence = 0.75
            reason = f"Bullish SMA crossover: Short({self.short_sma_period}) {short_sma:.2f} > Long({self.long_sma_period}) {long_sma:.2f}"
            if confidence < self.signal_confidence_threshold and close_prices[-1] > short_sma:
                confidence = min(1.0, confidence + 0.1)
        elif short_sma < long_sma and prev_short_sma >= prev_long_sma:
            action = "SELL"
            confidence = 0.75
            reason = f"Bearish SMA crossover: Short({self.short_sma_period}) {short_sma:.2f} < Long({self.long_sma_period}) {long_sma:.2f}"
            if confidence < self.signal_confidence_threshold and close_prices[-1] < short_sma:
                confidence = min(1.0, confidence + 0.1)

        self.logger.info(f"Generated signal for {symbol} ({interval}): {action} with {confidence:.2f} confidence. Reason: {reason}")
        return {"action": action, "confidence": confidence, "timestamp": time.time(), "reason": reason}

    async def run(self):
        if not self.trading_enabled:
            self.logger.info("SignalFusion is disabled in config. Skipping run loop.") # Updated log message
            return

        self.logger.info("SignalFusion started. Awaiting signals...") # Updated log message
        processing_interval = getattr(self.config, "SIGNAL_FUSION_AGENT_INTERVAL", 30)
        signal_interval = getattr(self.config, "MARKET_DATA_INTERVAL", "1m")

        while True:
            try:
                if not getattr(self.shared_state, 'initial_market_data_loaded', False):
                    self.logger.debug("SignalFusion waiting for initial market data to load...") # Updated log message
                    await asyncio.sleep(5)
                    continue

                active_symbols = self.shared_state.symbols

                if not active_symbols:
                    self.logger.warning("No active symbols found in shared state. SignalFusion skipping cycle.") # Updated log message
                    await asyncio.sleep(processing_interval)
                    continue

                for symbol in active_symbols:
                    self.symbol = symbol
                    self.logger.info(f"[SignalFusion] Running on {self.symbol}") # Updated log message
                    signal = await self._generate_signal(symbol, signal_interval)

                    if signal:
                        if not hasattr(self.shared_state, 'agent_signals'):
                            self.shared_state.agent_signals = {}
                        if "SignalFusionAgent" not in self.shared_state.agent_signals: # Keep this key as it refers to the agent type
                            self.shared_state.agent_signals["SignalFusionAgent"] = {}

                        self.shared_state.agent_signals["SignalFusionAgent"][symbol] = signal # Keep this key as it refers to the agent type
                        self.logger.debug(f"Published signal for {symbol}: {signal['action']} (Confidence: {signal['confidence']:.2f})")
                    else:
                        self.logger.warning(f"Failed to generate signal for {symbol}.")

            except Exception as e:
                self.logger.exception(f"Error in SignalFusion run loop: {e}") # Updated log message

            await asyncio.sleep(processing_interval)

    async def run_once(self):
        if not self.trading_enabled:
            self.logger.info("SignalFusion is disabled in config. Skipping run_once.") # Updated log message
            return

        signal_interval = getattr(self.config, "MARKET_DATA_INTERVAL", "1m")
        active_symbols = self.shared_state.symbols

        if not active_symbols:
            self.logger.warning("No active symbols found in shared state. Skipping run_once.") # Updated log message
            return

        for symbol in active_symbols:
            self.symbol = symbol
            self.logger.info(f"[SignalFusion] run_once on {self.symbol}") # Updated log message
            signal = await self._generate_signal(symbol, signal_interval)

            if signal:
                if not hasattr(self.shared_state, 'agent_signals'):
                    self.shared_state.agent_signals = {}
                if "SignalFusionAgent" not in self.shared_state.agent_signals: # Keep this key as it refers to the agent type
                    self.shared_state.agent_signals["SignalFusionAgent"] = {}

                self.shared_state.agent_signals["SignalFusionAgent"][symbol] = signal # Keep this key as it refers to the agent type
                self.logger.debug(f"Published signal for {symbol}: {signal['action']} ({signal['confidence']:.2f})")
            else:
                self.logger.warning(f"Failed to generate signal for {symbol}.")
