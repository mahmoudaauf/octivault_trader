import asyncio
from typing import Any, Dict, List, Optional
import logging
import os
import pandas as pd
import numpy as np
import time
from functools import partial

from utils.indicators import compute_ema, compute_bollinger_bands, compute_atr
from core.agent_optimizer import load_tuned_params
from core.component_status_logger import log_component_status

AGENT_NAME = "DipSniper"
logger = logging.getLogger(AGENT_NAME)
logger.setLevel(logging.INFO)
logger.propagate = False

_log_path = f"logs/agents/{AGENT_NAME.lower()}.log"
os.makedirs(os.path.dirname(_log_path), exist_ok=True)

file_handler = logging.FileHandler(_log_path)
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] [%(name)s] %(message)s"))
if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
    logger.addHandler(file_handler)


class DipSniper:
    """
    P9-COMPLIANT: Signal-only dip detection agent.
    
    Responsibilities:
    - Analyze symbols for dip conditions
    - Generate BUY signals when conditions met
    - Return signals to AgentManager
    
    NOT responsible for:
    - Executing trades
    - Calculating quantities
    - Managing TP/SL
    - Direct MetaController emission
    """
    
    agent_type = "strategy"
    
    def __init__(self, shared_state, execution_manager, config, tp_sl_engine=None,
                 timeframe='5m', symbols=None, name="DipSniper", **kwargs):
        self.name = self.__class__.__name__
        self.shared_state = shared_state
        # execution_manager and tp_sl_engine kept for compatibility but NOT USED
        self.config = config
        self.timeframe = timeframe
        
        # P9 FIX: Signal collection buffer
        self._collected_signals: List[Dict[str, Any]] = []
        
        # P9 FIX: Use get_accepted_symbols() not direct .symbols access
        self.symbols = symbols if symbols is not None else []
        
        # IMPROVEMENT: Concurrency control (optional but recommended)
        max_concurrency = int(getattr(config, "DIPSNIPER_MAX_CONCURRENCY", 5))
        self._sem = asyncio.Semaphore(max_concurrency)
        
    # P9 Config Tuning (Relaxed Thresholds per User Request)
    # NOTE: Use _cfg-backed properties to avoid clashing with @property names.
        
        log_component_status(self.name, "Initialized")
        logger.info(f"🚀 {self.name} initialized (P9-compliant, signal-only, max_concurrency={max_concurrency})")

    async def generate_signals(self) -> List[Dict[str, Any]]:
        """
        P9 CONTRACT: Generate signals for ALL symbols.
        Called by AgentManager. Returns signal list.
        """
        self._collected_signals = []
        
        # Refresh symbols from SharedState
        symbols = await self._safe_get_symbols()
        if set(symbols) != set(self.symbols):
            self.symbols = list(symbols)
            logger.info(f"[{self.name}] Updated symbols: {len(self.symbols)}")
        
        if not self.symbols:
            logger.warning(f"[{self.name}] No symbols to analyze")
            return []
        
        # Analyze ALL symbols with concurrency control
        async def _one(sym: str):
            async with self._sem:
                await self._analyze_symbol(sym)
        
        await asyncio.gather(*[_one(s) for s in self.symbols], return_exceptions=True)
        
        # Return collected signals
        signals = self._collected_signals
        self._collected_signals = []
        logger.info(f"[{self.name}] Generated {len(signals)} signals across {len(self.symbols)} symbols")
        return signals

    async def _safe_get_symbols(self) -> List[str]:
        """P9 FIX: Use canonical symbol source."""
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

    async def _analyze_symbol(self, symbol: str):
        """Analyze one symbol and collect signal if conditions met."""
        logger.debug(f"[{self.name}] 🔍 Analyzing {symbol}")
        
        try:
            # IMPROVEMENT 1: Freshness check (prevent stale data signals)
            if hasattr(self.shared_state, "is_fresh"):
                try:
                    fresh = self.shared_state.is_fresh(symbol, self.timeframe)
                    if asyncio.iscoroutine(fresh):
                        fresh = await fresh
                    if not fresh:
                        logger.debug(f"[{self.name}] Stale data for {symbol}, skipping")
                        return
                except Exception:
                    pass  # If freshness check fails, continue anyway
            
            candles = await self.shared_state.get_market_data(symbol, self.timeframe)
            
            # Handle dict response
            if isinstance(candles, dict):
                candles = candles.get(symbol, [])
            
            if not isinstance(candles, list) or len(candles) < 50:
                logger.debug(f"[{self.name}] Insufficient data for {symbol}")
                return  # P9 FIX: Don't emit HOLD signals
            
            # Create DataFrame
            if isinstance(candles[0], dict):
                df = await asyncio.to_thread(pd.DataFrame, candles)
            else:
                df = await asyncio.to_thread(
                    pd.DataFrame,
                    candles,
                    columns=["timestamp", "open", "high", "low", "close", "volume"]
                )
            
            # Clean data
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df.dropna(subset=['close', 'volume'], inplace=True)
            
            if len(df) < 50:
                return  # Not enough data - don't emit
            
            # Compute indicators
            def compute_indicators_sync(dataframe):
                df_copy = dataframe.copy()
                df_copy['ema20'] = compute_ema(df_copy['close'], span=20)
                df_copy['bb_upper'], df_copy['bb_middle'], df_copy['bb_lower'] = compute_bollinger_bands(df_copy['close'])
                df_copy['atr'] = compute_atr(df_copy)
                return df_copy
            
            df = await asyncio.to_thread(compute_indicators_sync, df)
            
            if df.empty or len(df) < 2:
                return  # Not enough data after indicators
            
            latest = df.iloc[-1]
            previous = df.iloc[-2]
            
            # Check for NaN indicators
            if pd.isna(latest['bb_lower']) or pd.isna(latest['ema20']) or pd.isna(latest['atr']):
                return  # Invalid indicators - don't emit
            
            if latest['atr'] <= 0:
                return  # Can't compute confidence
            
            # Calculate dip metrics
            dip_percent = (previous['close'] - latest['close']) / previous['close'] * 100 if previous['close'] != 0 else 0.0
            volume_avg = df['volume'].rolling(10).mean().iloc[-1]
            volume_spike = latest['volume'] > volume_avg * self.volume_spike_multiplier
            
            # Check dip conditions
            condition = (
                latest['close'] < latest['bb_lower'] and
                latest['close'] < latest['ema20'] and
                dip_percent > self.dip_threshold_percent and
                volume_spike
            )
            
            # P9 FIX: Only emit BUY signals when conditions met
            if self.enabled and condition:
                confidence = np.clip(abs((latest['close'] - latest['bb_lower']) / latest['atr']), 0, 1)
                
                # IMPROVEMENT 2: Confidence floor (reduce noise)
                base_min_conf = float(self._cfg("MIN_SIGNAL_CONF", 0.55))
                
                # Dynamic Aggression (P9 Loop)
                agg_factor = 1.0
                if hasattr(self.shared_state, "get_dynamic_param"):
                    agg_factor = float(self.shared_state.get_dynamic_param("aggression_factor", 1.0))
                
                min_conf = base_min_conf
                if agg_factor > 1.0:
                    min_conf = max(0.40, base_min_conf / agg_factor)

                if confidence < min_conf:
                    logger.debug(f"[{self.name}] {symbol} confidence {confidence:.2f} below dynamic threshold {min_conf:.2f}")
                    return
                
                # Soft guard: warn if quote hint is below SAFE_ENTRY_USDT; execution layer enforces floors
                quote_hint = float(
                    self._cfg(
                        "EMIT_BUY_QUOTE",
                        self._cfg("MIN_ENTRY_USDT", self._cfg("DEFAULT_PLANNED_QUOTE", 25.0)),
                    )
                )
                min_entry = float(getattr(self.config, "MIN_ENTRY_USDT", getattr(self.config, "SAFE_ENTRY_USDT", 12.0)))
                if quote_hint < min_entry:
                    logger.warning(
                        f"[{self.name}] Signal quote {quote_hint:.2f} < MIN_ENTRY_USDT {min_entry:.2f}; deferring to execution layer"
                    )
                
                # P9 CONTRACT: Build signal dict
                signal = {
                    "symbol": symbol,
                    "side": "BUY",
                    "action": "BUY",  # Kept for compatibility
                    "confidence": float(confidence),
                    "reason": f"DipSniper: dip {dip_percent:.2f}% below BB with volume spike",
                    "agent": self.name,
                    "quote": quote_hint,
                    "horizon_hours": 6.0,
                }
                
                # Mandatory P9 Signal Contract: Emit to Signal Bus
                if hasattr(self.shared_state, "add_agent_signal"):
                    try:
                        await self.shared_state.add_agent_signal(
                            symbol=symbol,
                            agent=self.name,
                            side="BUY",
                            confidence=float(confidence),
                            ttl_sec=300,
                            tier="B",
                            rationale=f"Dip detected: {dip_percent:.2f}%"
                        )
                    except Exception as e:
                        logger.warning(f"[{self.name}] Failed to emit to signal bus: {e}")

                self._collected_signals.append(signal)
                logger.info(f"[{self.name}] 📤 BUY signal: {symbol} conf={confidence:.2f} dip={dip_percent:.2f}%")
            
            # P9 FIX: No HOLD signals - silence is implicit hold
                
        except Exception as e:
            logger.error(f"[{self.name}] ❌ Error analyzing {symbol}: {e}", exc_info=True)
            # Don't emit error signals - let it silently fail

    def _cfg(self, key: str, default: Any = None) -> Any:
        """Get config value with fallback."""
        # Check SharedState for dynamic overrides
        if hasattr(self.shared_state, "dynamic_config"):
            val = self.shared_state.dynamic_config.get(key)
            if val is not None:
                return val
        
        # Fallback to static config
        if isinstance(self.config, dict):
            return self.config.get(key, default)
        return getattr(self.config, key, default)

    @property
    def volume_spike_multiplier(self) -> float:
        return float(self._cfg("VOLUME_SPIKE_MULTIPLIER", 1.5))

    @property
    def dip_threshold_percent(self) -> float:
        return float(self._cfg("DIP_THRESHOLD_PERCENT", 2.0))

    @property
    def enabled(self) -> bool:
        return bool(self._cfg("DIPSNIPER_ENABLED", True))
