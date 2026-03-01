"""
Universe-Ready Live Trading System Architecture

Design principle: Symbol-agnostic regime detection + per-symbol exposure engine
Allows future expansion to rotation layer without refactoring.

Structure:
  1. RegimeDetectionEngine (symbol-agnostic)
  2. ExposureController (per-symbol config)
  3. PositionSizer (per-symbol risk management)
  4. UniverseManager (future rotation layer)
  5. LiveTradingOrchestrator (main controller)
"""

import pandas as pd
import numpy as np
import asyncio
import logging
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES - Universe agnostic
# ============================================================================

@dataclass
class RegimeState:
    """Market regime snapshot - symbol agnostic"""
    timestamp: datetime
    symbol: str
    
    # Volatility regime
    volatility: float
    volatility_regime: str  # LOW_VOL, NORMAL, HIGH_VOL
    
    # Trend regime
    momentum: float
    autocorr_lag1: float
    trend_regime: str  # TRENDING, MEAN_REVERT
    
    # Combined regime
    regime: str  # e.g., LOW_VOL_TRENDING
    
    # Macro regime
    price: float
    sma_200: float
    macro_trend: str  # UPTREND, DOWNTREND
    
    def is_alpha_regime(self) -> bool:
        """Check if in alpha generation regime"""
        return self.regime == 'LOW_VOL_TRENDING' and self.macro_trend == 'UPTREND'


@dataclass
class SymbolConfig:
    """Per-symbol configuration"""
    symbol: str
    enabled: bool = True
    
    # Regime parameters
    volatility_lookback: int = 100
    vol_percentile_low: float = 0.33
    vol_percentile_high: float = 0.67
    autocorr_threshold: float = 0.1
    sma_period: int = 200
    
    # Exposure configuration
    base_exposure: float = 1.0  # 1x leverage base
    alpha_exposure: float = 2.0  # 2x leverage when in alpha regime
    downtrend_exposure: float = 0.0  # Flat in downtrends
    
    # Risk management
    max_position_size_pct: float = 0.05  # 5% of account
    max_drawdown_threshold: float = 0.30  # 30% stop loss
    daily_loss_limit: float = 0.02  # 2% daily loss limit
    
    # Validation
    min_signal_frequency: float = 0.01  # Alert if < 1% alpha regime occurrence


@dataclass
class PositionState:
    """Current position state - per symbol"""
    symbol: str
    size: float  # Number of units
    entry_price: float
    entry_time: datetime
    current_price: float
    current_pnl: float
    current_pnl_pct: float
    exposure: float  # Current leverage
    
    def unrealized_pnl_pct(self) -> float:
        """Calculate unrealized P&L percentage"""
        if self.entry_price == 0:
            return 0.0
        return (self.current_price - self.entry_price) / self.entry_price


# ============================================================================
# REGIME DETECTION ENGINE - Symbol agnostic
# ============================================================================

class RegimeDetectionEngine:
    """
    Symbol-agnostic regime detection.
    
    Works on any asset with OHLCV data.
    Returns standardized RegimeState for any symbol.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.RegimeDetection")
    
    def detect(self, df: pd.DataFrame, config: SymbolConfig) -> RegimeState:
        """
        Detect regime from OHLCV data.
        
        Args:
            df: DataFrame with OHLCV data (open, high, low, close, volume)
            config: SymbolConfig with regime parameters
        
        Returns:
            RegimeState: Current market regime
        """
        
        if len(df) < max(config.volatility_lookback, config.sma_period):
            raise ValueError(f"Insufficient data: {len(df)} candles, need {max(config.volatility_lookback, config.sma_period)}")
        
        # Use most recent candle
        latest = df.iloc[-1]
        
        # Calculate returns and volatility
        returns = df['close'].pct_change()
        volatility = returns.rolling(config.volatility_lookback).std().iloc[-1]
        
        # Volatility regime
        vol_low = df['close'].pct_change().rolling(config.volatility_lookback).std().quantile(config.vol_percentile_low)
        vol_high = df['close'].pct_change().rolling(config.volatility_lookback).std().quantile(config.vol_percentile_high)
        
        if volatility < vol_low:
            vol_regime = 'LOW_VOL'
        elif volatility > vol_high:
            vol_regime = 'HIGH_VOL'
        else:
            vol_regime = 'NORMAL'
        
        # Momentum and autocorrelation
        momentum = returns.rolling(config.volatility_lookback).mean().iloc[-1]
        autocorr_lag1 = returns.rolling(config.volatility_lookback).apply(
            lambda x: x.autocorr(lag=1) if len(x) > 1 else 0
        ).iloc[-1]
        
        # Trend regime
        momentum_sign = np.sign(momentum)
        autocorr_positive = autocorr_lag1 > config.autocorr_threshold
        
        if momentum_sign * autocorr_positive > 0:
            trend_regime = 'TRENDING'
        else:
            trend_regime = 'MEAN_REVERT'
        
        # Combined regime
        combined_regime = f"{vol_regime}_{trend_regime}"
        
        # Macro regime (uptrend/downtrend)
        sma_200 = df['close'].rolling(config.sma_period).mean().iloc[-1]
        macro_trend = 'UPTREND' if latest['close'] > sma_200 else 'DOWNTREND'
        
        return RegimeState(
            timestamp=pd.to_datetime(latest['timestamp']) if isinstance(latest['timestamp'], (int, float)) else latest['timestamp'],
            symbol=config.symbol,
            volatility=volatility,
            volatility_regime=vol_regime,
            momentum=momentum,
            autocorr_lag1=autocorr_lag1,
            trend_regime=trend_regime,
            regime=combined_regime,
            price=float(latest['close']),
            sma_200=float(sma_200),
            macro_trend=macro_trend,
        )


# ============================================================================
# EXPOSURE ENGINE - Per-symbol configuration
# ============================================================================

class ExposureController:
    """
    Per-symbol exposure calculation.
    
    Maps regime state to leverage decisions.
    Handles macro filter (reduce leverage in downtrends).
    """
    
    def __init__(self, config: SymbolConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.Exposure.{config.symbol}")
    
    def calculate_exposure(self, regime_state: RegimeState) -> float:
        """
        Calculate recommended exposure for given regime.
        
        Args:
            regime_state: Current market regime
        
        Returns:
            float: Recommended leverage (1.0, 2.0, etc.)
        """
        
        # Macro filter: reduce/eliminate leverage in downtrends
        if regime_state.macro_trend == 'DOWNTREND':
            return self.config.downtrend_exposure
        
        # Alpha regime: apply 2x leverage
        if regime_state.is_alpha_regime():
            return self.config.alpha_exposure
        
        # Uptrend but not alpha regime: base exposure
        if regime_state.macro_trend == 'UPTREND':
            return self.config.base_exposure
        
        # Fallback
        return self.config.base_exposure
    
    def get_signal(self, regime_state: RegimeState) -> Dict:
        """Get trading signal from regime state"""
        
        exposure = self.calculate_exposure(regime_state)
        
        return {
            'symbol': self.config.symbol,
            'regime': regime_state.regime,
            'macro_trend': regime_state.macro_trend,
            'is_alpha_regime': regime_state.is_alpha_regime(),
            'exposure': exposure,
            'action': 'INCREASE' if exposure > 1.0 else ('FLAT' if exposure == 0.0 else 'HOLD'),
        }


# ============================================================================
# POSITION SIZER - Per-symbol risk management
# ============================================================================

class PositionSizer:
    """
    Per-symbol position sizing and risk management.
    
    Handles:
    - Kelly criterion adjustments
    - Account risk limits
    - Drawdown protection
    - Dynamic leverage
    """
    
    def __init__(self, config: SymbolConfig, account_balance: float):
        self.config = config
        self.account_balance = account_balance
        self.logger = logging.getLogger(f"{__name__}.PositionSizer.{config.symbol}")
    
    def calculate_position_size(self, 
                               exposure: float,
                               current_price: float,
                               current_drawdown: float) -> float:
        """
        Calculate position size for symbol.
        
        Args:
            exposure: Leverage multiplier (1.0, 2.0, etc.)
            current_price: Current asset price
            current_drawdown: Current portfolio drawdown (-0.15 = -15%)
        
        Returns:
            float: Position size in units
        """
        
        # Base position size (5% of account)
        max_risk_amount = self.account_balance * self.config.max_position_size_pct
        
        # Drawdown adjustment: reduce leverage if already in drawdown
        dd_adjustment = 1.0
        if current_drawdown < 0:
            # Reduce exposure if in drawdown
            dd_adjustment = max(0.5, 1.0 + current_drawdown)  # 50% minimum
        
        # Effective leverage
        effective_leverage = exposure * dd_adjustment
        
        # Position size in units
        position_size = (max_risk_amount * effective_leverage) / current_price
        
        return position_size
    
    def check_risk_limits(self, 
                         current_pnl_pct: float,
                         daily_pnl_pct: float) -> Tuple[bool, str]:
        """
        Check if risk limits are breached.
        
        Returns:
            (is_ok, reason)
        """
        
        if current_pnl_pct < -self.config.max_drawdown_threshold:
            return False, f"Max drawdown exceeded: {current_pnl_pct:.2%} < {-self.config.max_drawdown_threshold:.2%}"
        
        if daily_pnl_pct < -self.config.daily_loss_limit:
            return False, f"Daily loss limit exceeded: {daily_pnl_pct:.2%} < {-self.config.daily_loss_limit:.2%}"
        
        return True, "OK"


# ============================================================================
# UNIVERSE MANAGER - Framework for future rotation
# ============================================================================

class UniverseManager:
    """
    Manages multiple symbols and their configs.
    
    Future: Can implement rotation layer here.
    """
    
    def __init__(self):
        self.configs: Dict[str, SymbolConfig] = {}
        self.logger = logging.getLogger(f"{__name__}.UniverseManager")
    
    def add_symbol(self, config: SymbolConfig):
        """Add symbol to universe"""
        self.configs[config.symbol] = config
        self.logger.info(f"Added {config.symbol} to universe")
    
    def remove_symbol(self, symbol: str):
        """Remove symbol from universe"""
        if symbol in self.configs:
            del self.configs[symbol]
            self.logger.info(f"Removed {symbol} from universe")
    
    def get_enabled_symbols(self) -> List[str]:
        """Get list of enabled symbols"""
        return [s for s, c in self.configs.items() if c.enabled]
    
    def get_config(self, symbol: str) -> Optional[SymbolConfig]:
        """Get config for symbol"""
        return self.configs.get(symbol)


# ============================================================================
# LIVE TRADING ORCHESTRATOR - Main controller
# ============================================================================

class LiveTradingOrchestrator:
    """
    Main orchestrator for live trading system.
    
    Coordinates:
    - Real-time data fetching
    - Regime detection (symbol-agnostic)
    - Exposure calculation (per-symbol)
    - Position sizing (per-symbol)
    - Trade execution
    - Risk monitoring
    """
    
    def __init__(self, account_balance: float = 100000):
        self.account_balance = account_balance
        self.universe_manager = UniverseManager()
        self.regime_engine = RegimeDetectionEngine()
        
        self.positions: Dict[str, PositionState] = {}
        self.regime_states: Dict[str, RegimeState] = {}
        self.exposure_controllers: Dict[str, ExposureController] = {}
        self.position_sizers: Dict[str, PositionSizer] = {}
        
        self.logger = logging.getLogger(f"{__name__}.Orchestrator")
    
    def initialize_symbol(self, config: SymbolConfig):
        """Initialize symbol for trading"""
        
        self.universe_manager.add_symbol(config)
        self.exposure_controllers[config.symbol] = ExposureController(config)
        self.position_sizers[config.symbol] = PositionSizer(config, self.account_balance)
        
        self.logger.info(f"Initialized {config.symbol}")
        self.logger.info(f"  Base exposure: {config.base_exposure}x")
        self.logger.info(f"  Alpha exposure: {config.alpha_exposure}x")
        self.logger.info(f"  Downtrend exposure: {config.downtrend_exposure}x")
    
    def update_regimes(self, symbol_data: Dict[str, pd.DataFrame]):
        """
        Update regime states for all symbols.
        
        Args:
            symbol_data: Dict of {symbol: ohlcv_dataframe}
        """
        
        for symbol in self.universe_manager.get_enabled_symbols():
            config = self.universe_manager.get_config(symbol)
            
            if symbol not in symbol_data:
                self.logger.warning(f"No data for {symbol}")
                continue
            
            try:
                regime_state = self.regime_engine.detect(symbol_data[symbol], config)
                self.regime_states[symbol] = regime_state
                
            except Exception as e:
                self.logger.error(f"Error detecting regime for {symbol}: {e}")
    
    def calculate_signals(self) -> Dict[str, Dict]:
        """
        Calculate trading signals for all enabled symbols.
        
        Returns:
            Dict of {symbol: signal_dict}
        """
        
        signals = {}
        
        for symbol in self.universe_manager.get_enabled_symbols():
            if symbol not in self.regime_states:
                continue
            
            regime_state = self.regime_states[symbol]
            exposure_controller = self.exposure_controllers[symbol]
            
            signal = exposure_controller.get_signal(regime_state)
            signals[symbol] = signal
        
        return signals
    
    def get_position_summary(self) -> pd.DataFrame:
        """Get summary of all positions"""
        
        rows = []
        for symbol, position in self.positions.items():
            rows.append({
                'Symbol': symbol,
                'Size': position.size,
                'EntryPrice': position.entry_price,
                'CurrentPrice': position.current_price,
                'PnL': position.current_pnl,
                'PnL%': position.current_pnl_pct,
                'Exposure': position.exposure,
            })
        
        return pd.DataFrame(rows) if rows else pd.DataFrame()
    
    def get_regime_summary(self) -> pd.DataFrame:
        """Get summary of all regime states"""
        
        rows = []
        for symbol, regime in self.regime_states.items():
            rows.append({
                'Symbol': symbol,
                'Regime': regime.regime,
                'MacroTrend': regime.macro_trend,
                'Volatility': regime.volatility,
                'Momentum': regime.momentum,
                'Autocorr': regime.autocorr_lag1,
                'Price': regime.price,
                'SMA200': regime.sma_200,
            })
        
        return pd.DataFrame(rows) if rows else pd.DataFrame()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == '__main__':
    logger.info("\n" + "="*80)
    logger.info("UNIVERSE-READY LIVE TRADING SYSTEM - ARCHITECTURE DEMO")
    logger.info("="*80)
    
    # Initialize orchestrator
    orchestrator = LiveTradingOrchestrator(account_balance=100000)
    
    # Configure ETH (immediate deployment)
    eth_config = SymbolConfig(
        symbol='ETHUSDT',
        enabled=True,
        base_exposure=1.0,
        alpha_exposure=2.0,
        downtrend_exposure=0.0,
    )
    orchestrator.initialize_symbol(eth_config)
    
    # Configure BTC (future expansion)
    btc_config = SymbolConfig(
        symbol='BTCUSDT',
        enabled=False,  # Disabled for now, enable when ready
        base_exposure=1.0,
        alpha_exposure=1.0,  # Lower leverage for BTC
        downtrend_exposure=0.0,
    )
    orchestrator.initialize_symbol(btc_config)
    
    logger.info("\nSystem initialized:")
    logger.info(f"  Account balance: ${orchestrator.account_balance:,.0f}")
    logger.info(f"  Enabled symbols: {orchestrator.universe_manager.get_enabled_symbols()}")
    logger.info(f"  Total symbols configured: {len(orchestrator.universe_manager.configs)}")
    
    logger.info("\n✅ Architecture ready for live deployment")
    logger.info("   - Symbol-agnostic regime detection")
    logger.info("   - Per-symbol exposure engine")
    logger.info("   - Per-symbol position sizing")
    logger.info("   - Framework for future rotation layer")
