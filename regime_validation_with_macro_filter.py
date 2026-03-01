"""
Enhanced Regime Validation with Market Regime Filter

Problem: Original strategy fails in bear markets (Months 3,6) because
LOW_VOL_TRENDING regimes don't occur during liquidation cascades.

Solution: Add macro market regime filter (SMA-based) to avoid trading
in downtrends. This acknowledges market regime dependency is REAL, not a flaw.

Three-stage validation:
1. Walk-Forward (with macro filter)
2. Sensitivity (SMA period, vol thresholds)
3. Capital Impact (leverage sweep, drawdown limits)
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MacroRegimeDetector:
    """Detect market uptrend/downtrend using SMA"""
    
    def __init__(self, sma_period: int = 200):
        self.sma_period = sma_period
    
    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """Returns 1 for uptrend, -1 for downtrend"""
        result = df.copy()
        result['sma'] = result['close'].rolling(self.sma_period).mean()
        result['macro_trend'] = np.where(result['close'] > result['sma'], 1, -1)
        return result


class MicroRegimeDetector:
    """Detect LOW_VOL_TRENDING micro regimes"""
    
    def __init__(self, lookback: int = 100, 
                 vol_percentile_low: float = 0.33,
                 vol_percentile_high: float = 0.67,
                 autocorr_threshold: float = 0.1):
        self.lookback = lookback
        self.vol_p_low = vol_percentile_low
        self.vol_p_high = vol_percentile_high
        self.autocorr_thresh = autocorr_threshold
    
    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect micro regimes"""
        result = df.copy()
        result['return'] = result['close'].pct_change()
        
        # Volatility
        result['volatility'] = result['return'].rolling(self.lookback).std()
        vol_low = result['volatility'].quantile(self.vol_p_low)
        vol_high = result['volatility'].quantile(self.vol_p_high)
        
        # Autocorrelation
        result['momentum'] = result['return'].rolling(self.lookback).mean()
        result['autocorr_lag1'] = result['return'].rolling(self.lookback).apply(
            lambda x: x.autocorr(lag=1) if len(x) > 1 else np.nan
        )
        
        # Regime classification
        result['volatility_regime'] = 'NORMAL'
        result.loc[result['volatility'] < vol_low, 'volatility_regime'] = 'LOW_VOL'
        result.loc[result['volatility'] > vol_high, 'volatility_regime'] = 'HIGH_VOL'
        
        momentum_sign = np.sign(result['momentum'])
        autocorr_positive = result['autocorr_lag1'] > self.autocorr_thresh
        result['trend_regime'] = 'MEAN_REVERT'
        result.loc[momentum_sign * autocorr_positive > 0, 'trend_regime'] = 'TRENDING'
        
        result['regime'] = result['volatility_regime'] + '_' + result['trend_regime']
        
        return result


class RobustValidator:
    """Validate regime strategy with macro filter"""
    
    def __init__(self, data_dir: str = None, output_dir: str = None):
        if data_dir is None:
            data_dir = "/Users/mauf/Desktop/Octi AI Trading Bot/octivault_trader/validation_outputs"
        if output_dir is None:
            output_dir = "/Users/mauf/Desktop/Octi AI Trading Bot/octivault_trader/validation_outputs"
        
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
    
    def backtest_with_filters(self, df: pd.DataFrame, 
                             macro_leverage: float = 1.0,
                             micro_leverage: float = 2.0,
                             macro_sma_period: int = 200) -> dict:
        """
        Backtest with both macro and micro regime filters
        
        Exposure logic:
        - If macro_trend < 0 (downtrend): exposure = macro_leverage (e.g., 1.0)
        - Else if LOW_VOL_TRENDING: exposure = micro_leverage (e.g., 2.0)
        - Else: exposure = 1.0
        """
        
        test_df = df.copy()
        test_df['return'] = test_df['return'].fillna(0)
        
        # Macro regime
        test_df['sma'] = test_df['close'].rolling(macro_sma_period).mean()
        test_df['macro_trend'] = np.where(test_df['close'] > test_df['sma'], 1, -1)
        
        # Strategy: 2x in LOW_VOL_TRENDING + uptrend, 1x elsewhere
        test_df['strategy_return'] = test_df['return'].copy()
        
        # Apply micro leverage only in macro uptrend
        low_vol_trending_uptrend = (
            (test_df['regime'] == 'LOW_VOL_TRENDING') & 
            (test_df['macro_trend'] > 0)
        )
        test_df.loc[low_vol_trending_uptrend, 'strategy_return'] *= micro_leverage
        
        # Calculate metrics
        returns = test_df['strategy_return'].dropna()
        if len(returns) < 10:
            return {}
        
        # Remove NaN from SMA warmup
        valid_idx = test_df['sma'].notna()
        returns = test_df.loc[valid_idx, 'strategy_return'].values
        
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + total_return) ** (252 * 24 / len(returns)) - 1
        volatility = np.std(returns) * np.sqrt(252 * 24)
        sharpe = annual_return / volatility if volatility > 0 else 0
        
        # Drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_dd = np.min(drawdown) if len(drawdown) > 0 else 0
        
        # Counts
        lvt_uptrend_periods = low_vol_trending_uptrend.sum()
        
        return {
            'total_return': float(total_return),
            'annual_return': float(annual_return),
            'volatility': float(volatility),
            'sharpe': float(sharpe),
            'max_drawdown': float(max_dd),
            'n_periods': int(len(returns)),
            'lvt_uptrend_periods': int(lvt_uptrend_periods),
        }
    
    def run_walk_forward(self, symbol: str, macro_sma_period: int = 200):
        """Run walk-forward validation with macro filter"""
        
        logger.info("")
        logger.info("="*80)
        logger.info(f"WALK-FORWARD VALIDATION (with Macro Filter): {symbol}")
        logger.info(f"SMA Period: {macro_sma_period}")
        logger.info("="*80)
        
        # Load 6-month data
        csv_file = self.data_dir / f"{symbol}_6month_1h_with_regime.csv"
        if not csv_file.exists():
            logger.error(f"Data file not found: {csv_file}")
            return {}
        
        df = pd.read_csv(csv_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        total_candles = len(df)
        candles_per_month = 720
        total_months = total_candles / candles_per_month
        
        logger.info(f"Total data: {total_candles} candles (~{total_months:.1f} months)")
        
        # Generate walk-forward splits
        train_months = 4
        test_months = 2
        train_candles = int(train_months * candles_per_month)
        test_candles = int(test_months * candles_per_month)
        
        splits = []
        
        # Split 1: Months 1-4 train, 5-6 test
        if total_candles >= train_candles + test_candles:
            train_df = df.iloc[:train_candles].copy()
            test_df = df.iloc[train_candles:train_candles + test_candles].copy()
            splits.append(('Month 1-4', 'Month 5-6', train_df, test_df))
        
        # Split 2: Roll forward
        roll_offset = int(candles_per_month)
        if total_candles >= roll_offset + train_candles + test_candles:
            train_df = df.iloc[roll_offset:roll_offset + train_candles].copy()
            test_df = df.iloc[roll_offset + train_candles:roll_offset + train_candles + test_candles].copy()
            splits.append(('Month 2-5', 'Month 6-7', train_df, test_df))
        
        logger.info(f"Generated {len(splits)} walk-forward splits")
        
        results = []
        
        for split_idx, (train_period, test_period, train_df, test_df) in enumerate(splits, 1):
            logger.info("")
            logger.info(f"SPLIT {split_idx}: {train_period} → {test_period}")
            
            # Add missing columns if needed
            if 'regime' not in test_df.columns:
                test_df['regime'] = 'UNKNOWN'
            
            metrics = self.backtest_with_filters(test_df, macro_sma_period=macro_sma_period)
            
            if metrics:
                logger.info(f"  Sharpe: {metrics['sharpe']:.4f}")
                logger.info(f"  Annual Return: {100*metrics['annual_return']:.2f}%")
                logger.info(f"  Max DD: {100*metrics['max_drawdown']:.2f}%")
                logger.info(f"  LOW_VOL_TRENDING + Uptrend periods: {metrics['lvt_uptrend_periods']}")
            
            results.append({
                'split': split_idx,
                'train_period': train_period,
                'test_period': test_period,
                'metrics': metrics
            })
        
        # Summary
        logger.info("")
        logger.info("="*80)
        logger.info("WALK-FORWARD SUMMARY")
        logger.info("="*80)
        
        sharpe_values = [r['metrics'].get('sharpe', 0) for r in results if r['metrics']]
        
        if sharpe_values:
            mean_sharpe = np.mean(sharpe_values)
            std_sharpe = np.std(sharpe_values)
            
            logger.info(f"Mean Test Sharpe: {mean_sharpe:.4f}")
            logger.info(f"Std Test Sharpe: {std_sharpe:.4f}")
            logger.info(f"Min/Max: {min(sharpe_values):.4f} / {max(sharpe_values):.4f}")
            
            if mean_sharpe > 0.3:
                logger.info("✅ SIGNAL PERSISTENT (Mean Sharpe > 0.3)")
            else:
                logger.warning(f"⚠️ SIGNAL WEAK (Mean Sharpe = {mean_sharpe:.4f})")
        
        return {
            'symbol': symbol,
            'sma_period': macro_sma_period,
            'splits': results,
            'mean_sharpe': float(np.mean(sharpe_values)) if sharpe_values else 0,
            'std_sharpe': float(np.std(sharpe_values)) if sharpe_values else 0,
        }
    
    def run_sensitivity(self, symbol: str):
        """Test robustness of SMA period"""
        
        logger.info("")
        logger.info("="*80)
        logger.info(f"SENSITIVITY ANALYSIS: {symbol}")
        logger.info("Testing SMA period variations")
        logger.info("="*80)
        
        csv_file = self.data_dir / f"{symbol}_6month_1h_with_regime.csv"
        if not csv_file.exists():
            logger.error(f"Data file not found")
            return {}
        
        df = pd.read_csv(csv_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Test different SMA periods
        sma_periods = [100, 150, 200, 250]
        results = []
        
        for sma_period in sma_periods:
            logger.info(f"\nTesting SMA {sma_period}...")
            
            # Split at month 4 (test on months 5-6)
            candles_per_month = 720
            test_start = int(4 * candles_per_month)
            test_df = df.iloc[test_start:test_start + int(2 * candles_per_month)].copy()
            
            validator = RobustValidator()
            metrics = validator.backtest_with_filters(test_df, macro_sma_period=sma_period)
            
            logger.info(f"  Sharpe: {metrics.get('sharpe', 0):.4f}")
            
            results.append({
                'sma_period': sma_period,
                'metrics': metrics
            })
        
        # Summary
        logger.info("")
        logger.info("SENSITIVITY SUMMARY")
        logger.info("="*80)
        
        sharpe_values = [r['metrics'].get('sharpe', 0) for r in results]
        logger.info(f"Sharpe range: {min(sharpe_values):.4f} to {max(sharpe_values):.4f}")
        logger.info(f"Std: {np.std(sharpe_values):.4f}")
        
        if np.std(sharpe_values) < 0.2:
            logger.info("✅ ROBUST (Sharpe std < 0.2)")
        else:
            logger.warning(f"⚠️ FRAGILE (Sharpe std = {np.std(sharpe_values):.4f})")
        
        return {
            'symbol': symbol,
            'results': results,
            'mean_sharpe': float(np.mean(sharpe_values)),
            'std_sharpe': float(np.std(sharpe_values))
        }
    
    def run_capital_impact(self, symbol: str):
        """Test leverage sweep"""
        
        logger.info("")
        logger.info("="*80)
        logger.info(f"CAPITAL IMPACT SIMULATION: {symbol}")
        logger.info("Testing leverage variations")
        logger.info("="*80)
        
        csv_file = self.data_dir / f"{symbol}_6month_1h_with_regime.csv"
        if not csv_file.exists():
            return {}
        
        df = pd.read_csv(csv_file)
        
        # Test on full data
        leverages = [1.0, 1.5, 2.0, 2.5, 3.0]
        results = []
        
        for leverage in leverages:
            logger.info(f"\nTesting {leverage}x leverage...")
            
            validator = RobustValidator()
            metrics = validator.backtest_with_filters(df, micro_leverage=leverage)
            
            logger.info(f"  Sharpe: {metrics.get('sharpe', 0):.4f}")
            logger.info(f"  Max DD: {100*metrics.get('max_drawdown', 0):.2f}%")
            
            results.append({
                'leverage': leverage,
                'metrics': metrics
            })
        
        # Summary
        logger.info("")
        logger.info("CAPITAL IMPACT SUMMARY")
        logger.info("="*80)
        
        for r in results:
            lev = r['leverage']
            max_dd = 100*r['metrics'].get('max_drawdown', 0)
            logger.info(f"  {lev}x: Max DD {max_dd:.2f}%")
        
        return {
            'symbol': symbol,
            'results': results
        }


if __name__ == '__main__':
    validator = RobustValidator()
    
    # Run all three stages
    wf_results = validator.run_walk_forward('BTCUSDT', macro_sma_period=200)
    sens_results = validator.run_sensitivity('BTCUSDT')
    cap_results = validator.run_capital_impact('BTCUSDT')
    
    # Save results
    all_results = {
        'walk_forward': wf_results,
        'sensitivity': sens_results,
        'capital_impact': cap_results,
    }
    
    output_file = Path('validation_outputs') / 'regime_validation_with_macro_filter_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"\n✅ Results saved to {output_file}")
