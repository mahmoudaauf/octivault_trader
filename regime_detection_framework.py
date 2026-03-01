"""
Regime Detection + Exposure Control Framework

Core Insight:
Instead of predicting price direction, detect market REGIME and adjust exposure.

Regimes have persistence:
- Trending markets reward trend-following
- Mean-reverting markets reward counter-trend
- Low-vol markets reward aggressive leverage
- High-vol markets reward position sizing down

Edge: Detect regime TRANSITIONS, rebalance exposure accordingly.

Not: "Will price go up?" (hard problem)
But: "Is market trending or mean-reverting?" (easier problem)
Then: Size exposure to match regime characteristics
"""

import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RegimeDetector:
    """Detect market regimes using statistical properties"""
    
    def __init__(self, lookback_periods: int = 100):
        """
        Args:
            lookback_periods: Window for calculating regime statistics
        """
        self.lookback = lookback_periods
    
    def detect_regimes(self, ohlcv_df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect market regimes from OHLCV data.
        
        Regimes:
        - TRENDING: High momentum, autocorrelated returns
        - MEAN_REVERT: Negative autocorrelation, reversals
        - LOW_VOL: Low volatility, tight ranges
        - HIGH_VOL: High volatility, large moves
        - TRANSITION: Regime changing, uncertain
        
        Args:
            ohlcv_df: DataFrame with OHLCV columns
        
        Returns:
            DataFrame with added regime columns
        """
        
        df = ohlcv_df.copy()
        df['return'] = df['close'].pct_change()
        df['abs_return'] = df['return'].abs()
        
        # Rolling statistics
        df['volatility'] = df['return'].rolling(self.lookback).std()
        df['momentum'] = df['return'].rolling(self.lookback).mean()
        
        # Autocorrelation: measure of persistence
        df['autocorr_lag1'] = df['return'].rolling(self.lookback).apply(
            lambda x: x.autocorr(lag=1) if len(x) > 1 else np.nan
        )
        
        # Range efficiency: trending markets have high range efficiency
        df['range'] = df['high'].rolling(self.lookback).max() - df['low'].rolling(self.lookback).min()
        df['atr'] = df['return'].rolling(self.lookback).std() * df['close']
        df['range_efficiency'] = df['range'] / (df['atr'] * self.lookback)
        
        # Volatility regime
        vol_p33 = df['volatility'].quantile(0.33)
        vol_p67 = df['volatility'].quantile(0.67)
        
        df['volatility_regime'] = 'NORMAL'
        df.loc[df['volatility'] < vol_p33, 'volatility_regime'] = 'LOW_VOL'
        df.loc[df['volatility'] > vol_p67, 'volatility_regime'] = 'HIGH_VOL'
        
        # Trend regime: High momentum + High autocorrelation
        momentum_sign = np.sign(df['momentum'])
        autocorr_positive = df['autocorr_lag1'] > 0.1
        
        df['trend_regime'] = 'MEAN_REVERT'
        df.loc[momentum_sign * autocorr_positive > 0, 'trend_regime'] = 'TRENDING'
        
        # Combined regime
        df['regime'] = df['volatility_regime'] + '_' + df['trend_regime']
        
        # Transition detection: when regime changes
        df['regime_changed'] = df['regime'].shift(1) != df['regime']
        df['regime_transition'] = False
        df.loc[df['regime_changed'] == True, 'regime_transition'] = True
        
        return df
    
    def regime_statistics(self, df_with_regimes: pd.DataFrame) -> dict:
        """
        Calculate regime-specific statistics.
        
        Returns:
            Dict with regime stats (win rate, risk, sharpe by regime)
        """
        
        stats_dict = {}
        
        for regime in df_with_regimes['regime'].unique():
            if pd.isna(regime):
                continue
            
            regime_df = df_with_regimes[df_with_regimes['regime'] == regime]
            
            if len(regime_df) < 10:
                continue
            
            returns = regime_df['return'].dropna()
            
            stats_dict[regime] = {
                'count': len(regime_df),
                'mean_return': float(returns.mean()),
                'volatility': float(returns.std()),
                'win_rate': float((returns > 0).sum() / len(returns)) if len(returns) > 0 else 0,
                'positive_autocorr': float((regime_df['autocorr_lag1'] > 0).sum() / len(regime_df)),
                'sharpe': float(returns.mean() / returns.std()) if returns.std() > 0 else 0,
            }
        
        return stats_dict


class ExposureController:
    """Control position sizing based on regime"""
    
    def __init__(self, base_exposure: float = 1.0, max_exposure: float = 2.0):
        """
        Args:
            base_exposure: Default position size (1.0 = 100% normal allocation)
            max_exposure: Maximum allowed exposure
        """
        self.base_exposure = base_exposure
        self.max_exposure = max_exposure
    
    def calculate_exposure(self, regime_row: pd.Series, regime_stats: dict) -> float:
        """
        Calculate position sizing based on current regime.
        
        Logic:
        - TRENDING + LOW_VOL: 2.0x exposure (high confidence trend)
        - TRENDING + HIGH_VOL: 0.8x exposure (trend but risky)
        - MEAN_REVERT + LOW_VOL: 1.2x exposure (mean reversion is safer)
        - MEAN_REVERT + HIGH_VOL: 0.6x exposure (high risk, counter-trend)
        - TRANSITION: 0.5x exposure (uncertain)
        
        Args:
            regime_row: Current row with regime info
            regime_stats: Dict with regime statistics
        
        Returns:
            Exposure multiplier (0.5 - 2.0)
        """
        
        regime = regime_row['regime']
        volatility_regime = regime_row['volatility_regime']
        trend_regime = regime_row['trend_regime']
        is_transition = regime_row['regime_transition']
        
        # Transition = reduce exposure
        if is_transition:
            return 0.5
        
        # Base on trend + volatility
        if trend_regime == 'TRENDING':
            if volatility_regime == 'LOW_VOL':
                return 2.0  # High confidence trend
            elif volatility_regime == 'HIGH_VOL':
                return 0.8  # Risky trend
            else:
                return 1.5  # Normal trend
        
        else:  # MEAN_REVERT
            if volatility_regime == 'LOW_VOL':
                return 1.2  # Safe reversion
            elif volatility_regime == 'HIGH_VOL':
                return 0.6  # Risky reversion
            else:
                return 1.0  # Normal reversion
    
    def exposure_schedule(self, df_with_regimes: pd.DataFrame, regime_stats: dict) -> pd.DataFrame:
        """
        Calculate exposure schedule for all rows.
        
        Args:
            df_with_regimes: DataFrame with regime columns
            regime_stats: Regime statistics
        
        Returns:
            DataFrame with exposure column
        """
        
        df = df_with_regimes.copy()
        df['exposure'] = df.apply(
            lambda row: self.calculate_exposure(row, regime_stats),
            axis=1
        )
        
        # Smooth exposure changes to avoid whipsaws
        df['exposure_smooth'] = df['exposure'].ewm(span=10).mean()
        df['exposure_smooth'] = df['exposure_smooth'].clip(0.5, self.max_exposure)
        
        return df


class RegimeFrameworkValidator:
    """Validate regime detection framework on real data"""
    
    def __init__(self, data_dir: str = "data/historical", output_dir: str = "validation_outputs"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.detector = RegimeDetector(lookback_periods=100)
        self.controller = ExposureController(base_exposure=1.0, max_exposure=2.0)
    
    def run_validation(self, symbols: list = None):
        """Run full validation"""
        
        if symbols is None:
            symbols = ['BTCUSDT', 'ETHUSDT']
        
        logger.info("=" * 80)
        logger.info("REGIME DETECTION + EXPOSURE CONTROL FRAMEWORK")
        logger.info("=" * 80)
        logger.info("")
        logger.info("Core Concept: Not predicting price, detecting regime transitions")
        logger.info("Edge: Adjust exposure based on regime persistence")
        logger.info("")
        
        all_results = {}
        
        for symbol in symbols:
            logger.info("=" * 80)
            logger.info(f"{symbol}")
            logger.info("=" * 80)
            
            # Load data
            csv_file = self.data_dir / f"{symbol}_5m.csv"
            if not csv_file.exists():
                logger.warning(f"Data file not found: {csv_file}")
                continue
            
            df = pd.read_csv(csv_file)
            logger.info(f"Loaded {len(df)} candles")
            
            # Detect regimes
            logger.info("Detecting regimes...")
            df_regimes = self.detector.detect_regimes(df)
            
            # Calculate regime statistics
            regime_stats = self.detector.regime_statistics(df_regimes)
            
            logger.info("")
            logger.info("REGIME CHARACTERISTICS:")
            for regime_name, stats in regime_stats.items():
                logger.info(f"  {regime_name}:")
                logger.info(f"    Count: {stats['count']}")
                logger.info(f"    Mean return: {100*stats['mean_return']:.4f}%")
                logger.info(f"    Volatility: {100*stats['volatility']:.4f}%")
                logger.info(f"    Win rate: {100*stats['win_rate']:.1f}%")
                logger.info(f"    Positive autocorr: {100*stats['positive_autocorr']:.1f}%")
                logger.info(f"    Sharpe: {stats['sharpe']:.4f}")
            
            # Calculate exposure
            logger.info("")
            logger.info("Calculating exposure schedule...")
            df_exposure = self.controller.exposure_schedule(df_regimes, regime_stats)
            
            # Exposure statistics
            logger.info("")
            logger.info("EXPOSURE CONTROL:")
            logger.info(f"  Mean exposure: {df_exposure['exposure'].mean():.2f}x")
            logger.info(f"  Min exposure: {df_exposure['exposure'].min():.2f}x")
            logger.info(f"  Max exposure: {df_exposure['exposure'].max():.2f}x")
            logger.info(f"  Std exposure: {df_exposure['exposure'].std():.2f}x")
            
            # Exposure changes
            n_transitions = df_exposure['regime_transition'].sum()
            logger.info(f"  Regime transitions: {n_transitions} ({100*n_transitions/len(df_exposure):.2f}%)")
            
            # Save results
            output_csv = self.output_dir / f"{symbol}_with_regime_and_exposure.csv"
            df_exposure.to_csv(output_csv, index=False)
            logger.info(f"✅ Saved to {output_csv}")
            
            # Performance by regime
            logger.info("")
            logger.info("RETURNS BY REGIME:")
            for regime in df_exposure['regime'].unique():
                if pd.isna(regime):
                    continue
                regime_df = df_exposure[df_exposure['regime'] == regime]
                regime_returns = regime_df['return'].dropna()
                if len(regime_returns) > 0:
                    logger.info(f"  {regime}: {100*regime_returns.mean():.4f}% ± {100*regime_returns.std():.4f}%")
            
            all_results[symbol] = {
                'status': 'SUCCESS',
                'n_candles': len(df_exposure),
                'regime_stats': regime_stats,
                'exposure_mean': float(df_exposure['exposure'].mean()),
                'exposure_std': float(df_exposure['exposure'].std()),
                'regime_transitions': int(n_transitions),
                'output_file': str(output_csv),
            }
            
            logger.info("")
        
        # Summary
        logger.info("=" * 80)
        logger.info("SUMMARY")
        logger.info("=" * 80)
        logger.info("")
        logger.info("Framework: REGIME DETECTION + EXPOSURE CONTROL")
        logger.info("")
        logger.info("Key Metrics:")
        logger.info("  ✅ Detected market regimes (trending vs mean-revert)")
        logger.info("  ✅ Measured regime persistence (autocorrelation)")
        logger.info("  ✅ Calculated regime-specific win rates")
        logger.info("  ✅ Designed exposure schedule (0.5x - 2.0x)")
        logger.info("")
        logger.info("Next Steps:")
        logger.info("  1. Backtest exposure schedule")
        logger.info("  2. Measure Sharpe improvement from dynamic sizing")
        logger.info("  3. Test on out-of-sample data")
        logger.info("  4. Integrate into trading system")
        logger.info("")
        
        # Save summary
        summary_file = self.output_dir / "regime_framework_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        logger.info(f"✅ Summary saved to {summary_file}")
        
        return all_results


if __name__ == '__main__':
    
    logger.info("")
    logger.info("🎯 REGIME DETECTION + EXPOSURE CONTROL")
    logger.info("Pivot from directional prediction to regime-based sizing")
    logger.info("")
    
    validator = RegimeFrameworkValidator(
        data_dir='data/historical',
        output_dir='validation_outputs'
    )
    
    results = validator.run_validation()
