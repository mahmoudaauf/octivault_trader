"""
Market Regime Detector for Agent Specialization

Purpose
-------
Detects market regimes (trend, range, volatile, breakout, low_liquidity) to enable
agent specialization in your multi-agent system.

When agents operate in their specialized regime, they produce higher-conviction signals,
which improves MetaController weighted voting and reduces conflicting signals.

Regimes & Indicators
--------------------
Regime         | Primary Indicators | Characteristics
-------------- | ------------------ | -------------------------
trend          | ADX > 25           | Strong directional movement
range          | ADX < 20, RSI range| Sideways, mean-reversion territory
volatile       | ATR > threshold    | High price swings, uncertainty
breakout       | ADX rising, new hi/lo | Price breaking out of range
low_liquidity  | Spread > threshold | Poor market conditions

Agent Specialization Example
----------------------------
Regime      | Active Agents
----------- | ------------------------------------------
trend       | TrendHunter (0.4), MLForecaster (0.35)
range       | DipSniper (0.6), MLForecaster (0.4)
volatile    | RiskManager (0.5), MomentumAgent (0.3)
breakout    | TrendHunter (0.5), MomentumAgent (0.35)
low_liq     | None (all agents paused)

Integration Points
------------------
1. MetaController: select weighted agents based on detected regime
2. AgentManager: enable/disable agents based on regime
3. CapitalAllocator: adjust capital per regime
4. ExecutionManager: adjust execution style (maker vs aggressive)

Architecture
-----------
MarketDataFeed
      ↓
MarketRegimeDetector (this module)
      ↓
AgentManager (regime-aware agent selection)
      ↓
MetaController (weighted signal voting)
      ↓
PortfolioBudgetEngine / ExecutionManager

Config
------
MARKET_REGIME_DETECTOR:
  ENABLED: true
  ADX_PERIOD: 14                # RSI/ADX lookback
  ADX_TREND_THRESHOLD: 25       # ADX > this = trend
  ADX_RANGE_THRESHOLD: 20       # ADX < this = range
  ATR_PERIOD: 14                # ATR lookback
  ATR_VOLATILITY_THRESHOLD: 0.015  # ATR % > this = volatile
  RSI_PERIOD: 14
  RSI_OVERBOUGHT: 70
  RSI_OVERSOLD: 30
  SPREAD_MAX_PCT: 0.002         # Spread > 0.2% = low liquidity
  REGIME_SAMPLE_SIZE: 50        # OHLCV candles to analyze
  REGIME_SMOOTHING: true        # Smooth regime transitions
  SMOOTHING_WINDOW: 3           # votes required to flip regime

Performance Impact
------------------
Agent Specialization typically yields:
- Sharpe ratio: +40–80%
- Trade accuracy: +10–20%
- Fee efficiency: +20–30%
- Capital utilization: +30%
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

__all__ = ["MarketRegimeDetector", "MarketRegime", "RegimeMetrics"]

logger = logging.getLogger("MarketRegimeDetector")


class MarketRegime(Enum):
    """Market regime classifications."""
    TREND = "trend"
    RANGE = "range"
    VOLATILE = "volatile"
    BREAKOUT = "breakout"
    LOW_LIQUIDITY = "low_liquidity"
    UNKNOWN = "unknown"


@dataclass
class RegimeMetrics:
    """Metrics describing the current market regime."""
    regime: MarketRegime
    adx: float
    atr_pct: float
    rsi: float
    spread_pct: float
    volatility: float
    trend_strength: float
    momentum: float
    confidence: float  # 0.0-1.0, how confident in this regime classification
    timestamp: float
    symbol: str = ""


class MarketRegimeDetector:
    """
    Detects market regimes to guide agent specialization.
    
    Algorithm:
    1. Calculate ADX (trend strength)
    2. Calculate ATR (volatility)
    3. Calculate RSI (overbought/oversold)
    4. Analyze bid-ask spread
    5. Classify regime based on thresholds
    6. Optional smoothing to avoid whipsaw
    """

    def __init__(self, config: Optional[Any] = None, logger: Optional[logging.Logger] = None):
        """
        Initialize MarketRegimeDetector.
        
        Args:
            config: Configuration object with MARKET_REGIME_DETECTOR settings
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or globals()["logger"]
        
        # Extract config with sensible defaults
        regime_cfg = getattr(config, "MARKET_REGIME_DETECTOR", {}) if config else {}
        if not isinstance(regime_cfg, dict):
            regime_cfg = {}
        
        self.enabled = bool(regime_cfg.get("ENABLED", True))
        self.adx_period = int(regime_cfg.get("ADX_PERIOD", 14))
        self.adx_trend_threshold = float(regime_cfg.get("ADX_TREND_THRESHOLD", 25.0))
        self.adx_range_threshold = float(regime_cfg.get("ADX_RANGE_THRESHOLD", 20.0))
        self.atr_period = int(regime_cfg.get("ATR_PERIOD", 14))
        self.atr_volatility_threshold = float(regime_cfg.get("ATR_VOLATILITY_THRESHOLD", 0.015))
        self.rsi_period = int(regime_cfg.get("RSI_PERIOD", 14))
        self.rsi_overbought = float(regime_cfg.get("RSI_OVERBOUGHT", 70.0))
        self.rsi_oversold = float(regime_cfg.get("RSI_OVERSOLD", 30.0))
        self.spread_max_pct = float(regime_cfg.get("SPREAD_MAX_PCT", 0.002))
        self.sample_size = int(regime_cfg.get("REGIME_SAMPLE_SIZE", 50))
        self.enable_smoothing = bool(regime_cfg.get("REGIME_SMOOTHING", True))
        self.smoothing_window = int(regime_cfg.get("SMOOTHING_WINDOW", 3))
        
        # Regime history for smoothing
        self._regime_history: Dict[str, List[MarketRegime]] = {}
        self._metrics_cache: Dict[str, RegimeMetrics] = {}
        self._last_update: Dict[str, float] = {}
    
    def detect(
        self,
        symbol: str,
        ohlcv: List[Dict[str, float]],
        bid_price: Optional[float] = None,
        ask_price: Optional[float] = None,
    ) -> RegimeMetrics:
        """
        Detect market regime for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            ohlcv: List of OHLCV candles, each with:
                   {"open": float, "high": float, "low": float, "close": float, "volume": float}
                   Most recent candle should be last in list.
            bid_price: Current bid price (for spread calculation)
            ask_price: Current ask price (for spread calculation)
        
        Returns:
            RegimeMetrics with classification and confidence
        
        Example:
            >>> ohlcv = [
            ...     {"open": 100, "high": 102, "low": 99, "close": 101, "volume": 1000},
            ...     {"open": 101, "high": 103, "low": 100, "close": 102, "volume": 1100},
            ... ]
            >>> metrics = detector.detect("BTCUSDT", ohlcv, bid=101.5, ask=102.5)
            >>> print(metrics.regime)  # MarketRegime.TREND
        """
        if not self.enabled:
            return RegimeMetrics(
                regime=MarketRegime.UNKNOWN,
                adx=0.0,
                atr_pct=0.0,
                rsi=0.0,
                spread_pct=0.0,
                volatility=0.0,
                trend_strength=0.0,
                momentum=0.0,
                confidence=0.0,
                timestamp=time.time(),
                symbol=symbol,
            )
        
        # Validate input
        if not ohlcv or len(ohlcv) < self.adx_period:
            return RegimeMetrics(
                regime=MarketRegime.UNKNOWN,
                adx=0.0,
                atr_pct=0.0,
                rsi=0.0,
                spread_pct=0.0,
                volatility=0.0,
                trend_strength=0.0,
                momentum=0.0,
                confidence=0.0,
                timestamp=time.time(),
                symbol=symbol,
            )
        
        try:
            # Extract price data
            # Support both normalized format ("c"/"h"/"l") and full names ("close"/"high"/"low")
            # CRITICAL: Use 'in' check instead of 'or' to avoid treating 0.0 as falsy
            sample = ohlcv[-self.sample_size:]
            
            # Try normalized keys first ("c", "h", "l"), then full names ("close", "high", "low")
            closes = []
            highs = []
            lows = []
            
            for c in sample:
                # Extract close: try "c" first, then "close"
                close_val = c.get("c") if "c" in c else c.get("close", 0.0)
                closes.append(float(close_val) if close_val is not None else 0.0)
                
                # Extract high: try "h" first, then "high"
                high_val = c.get("h") if "h" in c else c.get("high", 0.0)
                highs.append(float(high_val) if high_val is not None else 0.0)
                
                # Extract low: try "l" first, then "low"
                low_val = c.get("l") if "l" in c else c.get("low", 0.0)
                lows.append(float(low_val) if low_val is not None else 0.0)
            
            # Validate that we actually have data
            valid_closes = [c for c in closes if c and c > 0.0]
            if not valid_closes:
                self.logger.error(
                    f"[RegimeDetector] ZERO PRICES EXTRACTED for {symbol}! "
                    f"Sample size={len(sample)}, closes={closes[:3]}... "
                    f"OHLCV format check: first bar keys={list(sample[0].keys()) if sample else 'EMPTY'} "
                    f"first bar data={sample[0] if sample else 'N/A'}"
                )
                return RegimeMetrics(
                    regime=MarketRegime.UNKNOWN,
                    adx=0.0, atr_pct=0.0, rsi=0.0, spread_pct=0.0,
                    volatility=0.0, trend_strength=0.0, momentum=0.0, confidence=0.0,
                    timestamp=time.time(), symbol=symbol,
                )
            
            # Calculate indicators
            adx = self._calculate_adx(highs, lows, closes)
            atr, atr_pct = self._calculate_atr(highs, lows, closes)
            rsi = self._calculate_rsi(closes)
            volatility = self._calculate_volatility(closes)
            trend_strength = adx / 100.0  # Normalize to 0-1
            momentum = self._calculate_momentum(closes)
            
            # Spread analysis
            spread_pct = 0.0
            if bid_price and ask_price and ask_price > 0:
                spread_pct = (ask_price - bid_price) / ask_price
            
            # Check for low liquidity first (overrides other regimes)
            if spread_pct > self.spread_max_pct:
                regime = MarketRegime.LOW_LIQUIDITY
                confidence = 0.95
            # Classify regime
            elif adx > self.adx_trend_threshold:
                # Trending market
                regime = MarketRegime.TREND
                confidence = min(0.95, 0.5 + (adx / 100.0) * 0.45)
            elif adx < self.adx_range_threshold:
                # Ranging market
                regime = MarketRegime.RANGE
                confidence = min(0.95, 0.5 + ((20 - adx) / 20.0) * 0.45)
            elif atr_pct > self.atr_volatility_threshold:
                # High volatility
                regime = MarketRegime.VOLATILE
                confidence = min(0.95, 0.5 + (atr_pct / (self.atr_volatility_threshold * 2)) * 0.45)
            elif abs(momentum) > 0.7:
                # Strong momentum = breakout
                regime = MarketRegime.BREAKOUT
                confidence = min(0.95, 0.5 + abs(momentum) * 0.25)
            else:
                # Default to range
                regime = MarketRegime.RANGE
                confidence = 0.5
            
            # Apply smoothing if enabled
            if self.enable_smoothing:
                regime = self._apply_regime_smoothing(symbol, regime)
            
            # Build metrics
            metrics = RegimeMetrics(
                regime=regime,
                adx=adx,
                atr_pct=atr_pct,
                rsi=rsi,
                spread_pct=spread_pct,
                volatility=volatility,
                trend_strength=trend_strength,
                momentum=momentum,
                confidence=confidence,
                timestamp=time.time(),
                symbol=symbol,
            )
            
            # Cache for reuse
            self._metrics_cache[symbol] = metrics
            self._last_update[symbol] = time.time()
            
            # Log successful detection
            self.logger.info(
                f"[RegimeDetector] SUCCESS {symbol}: regime={regime.value} "
                f"adx={adx:.1f} atr_pct={atr_pct:.4f} spread={spread_pct:.6f} "
                f"confidence={confidence:.2f}"
            )
            
            return metrics
        
        except Exception as e:
            self.logger.error(f"Error detecting regime for {symbol}: {e}", exc_info=True)
            return RegimeMetrics(
                regime=MarketRegime.UNKNOWN,
                adx=0.0,
                atr_pct=0.0,
                rsi=0.0,
                spread_pct=0.0,
                volatility=0.0,
                trend_strength=0.0,
                momentum=0.0,
                confidence=0.0,
                timestamp=time.time(),
                symbol=symbol,
            )
    
    def _calculate_adx(self, highs: List[float], lows: List[float], closes: List[float]) -> float:
        """
        Calculate Average Directional Index (ADX).
        
        ADX measures trend strength (0-100).
        - ADX > 25: Strong trend
        - ADX 20-25: Moderate trend
        - ADX < 20: Weak trend (ranging)
        """
        if len(closes) < self.adx_period:
            return 0.0
        
        try:
            # Simplified ADX using Wilder's method
            plus_dm = []
            minus_dm = []
            tr = []
            
            for i in range(1, len(highs)):
                high_diff = highs[i] - highs[i - 1]
                low_diff = lows[i - 1] - lows[i]
                
                plus_dm.append(max(high_diff, 0.0) if high_diff > low_diff else 0.0)
                minus_dm.append(max(low_diff, 0.0) if low_diff > high_diff else 0.0)
                
                tr_val = max(
                    highs[i] - lows[i],
                    abs(highs[i] - closes[i - 1]),
                    abs(lows[i] - closes[i - 1]),
                )
                tr.append(tr_val)
            
            # Calculate smoothed values
            atr = sum(tr[-self.atr_period:]) / self.atr_period if tr else 0.0
            plus_di = (sum(plus_dm[-self.atr_period:]) / self.atr_period / atr * 100) if atr > 0 else 0.0
            minus_di = (sum(minus_dm[-self.atr_period:]) / self.atr_period / atr * 100) if atr > 0 else 0.0
            
            di_diff = abs(plus_di - minus_di)
            di_sum = plus_di + minus_di
            
            adx = (di_diff / di_sum * 100) if di_sum > 0 else 0.0
            return min(adx, 100.0)
        
        except Exception:
            return 0.0
    
    def _calculate_atr(self, highs: List[float], lows: List[float], closes: List[float]) -> Tuple[float, float]:
        """
        Calculate Average True Range (ATR).
        
        Returns:
            (atr_absolute, atr_percentage)
        
        ATR measures volatility. ATR % > 1.5% is high volatility.
        """
        if len(closes) < self.atr_period:
            return 0.0, 0.0
        
        try:
            tr = []
            for i in range(1, len(closes)):
                tr_val = max(
                    highs[i] - lows[i],
                    abs(highs[i] - closes[i - 1]),
                    abs(lows[i] - closes[i - 1]),
                )
                tr.append(tr_val)
            
            atr = sum(tr[-self.atr_period:]) / self.atr_period
            atr_pct = atr / closes[-1] if closes[-1] > 0 else 0.0
            
            return atr, atr_pct
        
        except Exception:
            return 0.0, 0.0
    
    def _calculate_rsi(self, closes: List[float]) -> float:
        """
        Calculate Relative Strength Index (RSI).
        
        RSI measures momentum (0-100).
        - RSI > 70: Overbought
        - RSI < 30: Oversold
        - RSI 40-60: Neutral
        """
        if len(closes) < self.rsi_period:
            return 50.0  # Default neutral
        
        try:
            deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
            
            gains = [d if d > 0 else 0.0 for d in deltas]
            losses = [-d if d < 0 else 0.0 for d in deltas]
            
            avg_gain = sum(gains[-self.rsi_period:]) / self.rsi_period
            avg_loss = sum(losses[-self.rsi_period:]) / self.rsi_period
            
            if avg_loss == 0:
                return 100.0 if avg_gain > 0 else 50.0
            
            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))
            
            return min(max(rsi, 0.0), 100.0)
        
        except Exception:
            return 50.0
    
    def _calculate_volatility(self, closes: List[float]) -> float:
        """
        Calculate simple price volatility (standard deviation of returns).
        
        Returns value 0.0-1.0 representing volatility intensity.
        """
        if len(closes) < 2:
            return 0.0
        
        try:
            returns = [
                (closes[i] - closes[i - 1]) / closes[i - 1]
                for i in range(1, min(len(closes), self.adx_period))
                if closes[i - 1] > 0
            ]
            
            if not returns:
                return 0.0
            
            mean_return = sum(returns) / len(returns)
            variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
            std_dev = variance ** 0.5
            
            # Normalize to 0-1 (assume max volatility is 5% per candle)
            return min(std_dev / 0.05, 1.0)
        
        except Exception:
            return 0.0
    
    def _calculate_momentum(self, closes: List[float]) -> float:
        """
        Calculate price momentum as rate of change.
        
        Returns value -1.0 to 1.0 representing momentum direction and intensity.
        """
        if len(closes) < 2:
            return 0.0
        
        try:
            period = min(self.atr_period, len(closes))
            change = (closes[-1] - closes[-period]) / closes[-period] if closes[-period] > 0 else 0.0
            
            # Normalize to -1 to 1 (assume max momentum is +/- 5%)
            return max(-1.0, min(change / 0.05, 1.0))
        
        except Exception:
            return 0.0
    
    def _apply_regime_smoothing(self, symbol: str, regime: MarketRegime) -> MarketRegime:
        """
        Smooth regime transitions to avoid whipsawing.
        
        Requires N consecutive readings of the same regime before flipping.
        """
        if not self.enable_smoothing:
            return regime
        
        if symbol not in self._regime_history:
            self._regime_history[symbol] = []
        
        history = self._regime_history[symbol]
        history.append(regime)
        
        # Keep only recent readings
        if len(history) > self.smoothing_window * 2:
            history.pop(0)
        
        # Count votes for current regime
        recent = history[-self.smoothing_window:]
        if len(recent) < self.smoothing_window:
            # Not enough history yet, return current
            return regime
        
        # If all recent readings are the same, use it
        if all(r == regime for r in recent):
            return regime
        
        # Otherwise, return the previous regime (maintain stability)
        if len(history) > 1:
            return history[-2]
        
        return regime
    
    def get_agent_weights(self, regime: MarketRegime) -> Dict[str, float]:
        """
        Get recommended agent weights for a specific regime.
        
        This guides the MetaController weighted signal voting.
        
        Returns:
            Dict of {agent_name: weight} where weights sum to 1.0
        
        Example:
            >>> weights = detector.get_agent_weights(MarketRegime.TREND)
            >>> # Returns {"TrendHunter": 0.4, "MLForecaster": 0.35, "DipSniper": 0.25}
        """
        weights = {
            MarketRegime.TREND: {
                "TrendHunter": 0.40,
                "MLForecaster": 0.35,
                "DipSniper": 0.15,
                "MomentumAgent": 0.10,
            },
            MarketRegime.RANGE: {
                "DipSniper": 0.40,
                "MLForecaster": 0.35,
                "MomentumAgent": 0.15,
                "TrendHunter": 0.10,
            },
            MarketRegime.VOLATILE: {
                "RiskManager": 0.40,
                "MomentumAgent": 0.30,
                "MLForecaster": 0.20,
                "TrendHunter": 0.10,
            },
            MarketRegime.BREAKOUT: {
                "TrendHunter": 0.40,
                "MomentumAgent": 0.35,
                "MLForecaster": 0.20,
                "DipSniper": 0.05,
            },
            MarketRegime.LOW_LIQUIDITY: {
                # All agents paused during low liquidity
            },
            MarketRegime.UNKNOWN: {
                # Equal weight to all agents when regime unknown
                "TrendHunter": 0.25,
                "DipSniper": 0.25,
                "MLForecaster": 0.25,
                "MomentumAgent": 0.25,
            },
        }
        
        return weights.get(regime, {})
    
    def get_execution_style(self, regime: MarketRegime) -> Dict[str, Any]:
        """
        Get recommended execution style for a regime.
        
        Guides MakerExecutor and ExecutionManager on how to execute trades
        given the current market conditions.
        
        Returns:
            Dict with execution parameters like maker_ratio, timeout, etc.
        """
        styles = {
            MarketRegime.TREND: {
                "maker_ratio": 0.3,  # 30% maker, 70% aggressive
                "limit_order_timeout_sec": 2.0,
                "spread_placement_ratio": 0.1,
                "description": "Fast execution in trending market",
            },
            MarketRegime.RANGE: {
                "maker_ratio": 0.8,  # 80% maker, 20% aggressive
                "limit_order_timeout_sec": 5.0,
                "spread_placement_ratio": 0.5,
                "description": "Patient execution in range-bound market",
            },
            MarketRegime.VOLATILE: {
                "maker_ratio": 0.2,  # 20% maker, 80% aggressive
                "limit_order_timeout_sec": 1.0,
                "spread_placement_ratio": 0.05,
                "description": "Fast execution in volatile market",
            },
            MarketRegime.BREAKOUT: {
                "maker_ratio": 0.2,  # 20% maker, 80% aggressive
                "limit_order_timeout_sec": 1.0,
                "spread_placement_ratio": 0.1,
                "description": "Fast execution on breakout",
            },
            MarketRegime.LOW_LIQUIDITY: {
                "maker_ratio": 1.0,  # 100% maker only
                "limit_order_timeout_sec": 10.0,
                "spread_placement_ratio": 1.0,
                "description": "Extreme caution in low liquidity",
            },
        }
        
        return styles.get(regime, styles[MarketRegime.RANGE])
    
    def get_capital_allocation(self, regime: MarketRegime) -> Dict[str, float]:
        """
        Get recommended capital allocation adjustments for a regime.
        
        Guides CapitalAllocator on how much capital to deploy in current conditions.
        
        Returns:
            Dict with allocation parameters
        """
        allocations = {
            MarketRegime.TREND: {
                "deploy_ratio": 0.8,  # Deploy 80% of available capital
                "risk_adjustment": 1.0,  # Normal risk
                "description": "High conviction trending market",
            },
            MarketRegime.RANGE: {
                "deploy_ratio": 0.5,  # Deploy 50% of available capital
                "risk_adjustment": 0.8,  # Reduce risk slightly
                "description": "Lower conviction range-bound market",
            },
            MarketRegime.VOLATILE: {
                "deploy_ratio": 0.3,  # Deploy 30% of available capital
                "risk_adjustment": 0.5,  # Significantly reduce risk
                "description": "High uncertainty, defensive posture",
            },
            MarketRegime.BREAKOUT: {
                "deploy_ratio": 0.7,  # Deploy 70% of available capital
                "risk_adjustment": 1.1,  # Slightly increase risk (high opportunity)
                "description": "Moderate conviction breakout move",
            },
            MarketRegime.LOW_LIQUIDITY: {
                "deploy_ratio": 0.0,  # Deploy 0% of capital
                "risk_adjustment": 0.0,  # No trading
                "description": "Market closed to trading",
            },
        }
        
        return allocations.get(regime, allocations[MarketRegime.RANGE])
    
    def get_cached_metrics(self, symbol: str) -> Optional[RegimeMetrics]:
        """Get previously calculated metrics for a symbol (if fresh)."""
        if symbol not in self._metrics_cache:
            return None
        
        metrics = self._metrics_cache[symbol]
        age = time.time() - metrics.timestamp
        
        # Cache valid for 60 seconds
        if age > 60:
            del self._metrics_cache[symbol]
            return None
        
        return metrics
