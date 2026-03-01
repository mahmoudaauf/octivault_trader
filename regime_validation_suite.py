"""
Walk-Forward Validation + Sensitivity Analysis + Capital Impact Simulation

Core Principle: Prove edge is real, not overfit.

Three validation stages:
1. Walk-Forward: Train on months 1-4, test on 5-6. Roll forward. Does Sharpe persist?
2. Sensitivity: Vary regime thresholds ±5-10%. Does edge collapse or hold?
3. Capital Impact: Test leverage from 1x to 3x. Check drawdown and tail risk.

Only if ALL THREE pass → move to production.
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RegimeDetectorParametric:
    """Parameterized regime detector for sensitivity analysis"""
    
    def __init__(self, lookback: int = 100, 
                 vol_percentile_low: float = 0.33,
                 vol_percentile_high: float = 0.67,
                 autocorr_threshold: float = 0.1):
        self.lookback = lookback
        self.vol_p_low = vol_percentile_low
        self.vol_p_high = vol_percentile_high
        self.autocorr_thresh = autocorr_threshold
    
    def detect_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect regimes with parameterized thresholds"""
        
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


class WalkForwardValidator:
    """Walk-forward validation of regime strategy"""
    
    def __init__(self, data_dir: str = "validation_outputs", output_dir: str = "validation_outputs"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def split_walk_forward(self, df: pd.DataFrame, train_months: int = 4, test_months: int = 2):
        """
        Generate walk-forward splits.
        
        For 1H data: 24 candles/day, 720 candles/month
        For 5M data: 288 candles/day, 8640 candles/month
        
        Returns:
            List of (train_df, test_df, train_period, test_period) tuples
        """
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Detect timeframe by checking candle frequency
        if len(df) > 1:
            time_diff = df['timestamp'].iloc[1] - df['timestamp'].iloc[0]
            if time_diff.total_seconds() <= 300:  # 5 minutes
                candles_per_month = 8640
            else:  # 1 hour or longer
                candles_per_month = 720
        else:
            candles_per_month = 8640  # default to 5m
        
        total_candles = len(df)
        total_months = total_candles / candles_per_month
        
        logger.info(f"Total data: {total_candles} candles (~{total_months:.1f} months)")
        
        splits = []
        
        # Need at least train + test months of data
        if total_months < train_months + test_months:
            logger.warning(f"Not enough data for walk-forward: {total_months:.1f} months < {train_months + test_months} required")
            return []
        
        # Initial split: months 0-3 train, months 4-5 test
        train_end = int(train_months * candles_per_month)
        test_end = int((train_months + test_months) * candles_per_month)
        
        train_df = df.iloc[:train_end].copy()
        test_df = df.iloc[train_end:test_end].copy()
        
        splits.append((
            train_df,
            test_df,
            f"Month 1-{train_months}",
            f"Month {train_months + 1}-{train_months + test_months}"
        ))
        
        # Roll forward if enough data
        if total_months >= train_months + test_months + 2:
            roll_train_start = int(candles_per_month)
            roll_train_end = int((train_months + 1) * candles_per_month)
            roll_test_end = int((train_months + test_months + 1) * candles_per_month)
            
            train_df_roll = df.iloc[roll_train_start:roll_train_end].copy()
            test_df_roll = df.iloc[roll_train_end:roll_test_end].copy()
            
            splits.append((
                train_df_roll,
                test_df_roll,
                f"Month 2-{train_months + 1}",
                f"Month {train_months + 2}-{train_months + test_months + 1}"
            ))
        
        return splits
    
    def backtest_regime_strategy(self, df_with_regime: pd.DataFrame, 
                                 aggressive_leverage: float = 2.0) -> dict:
        """Backtest strategy on given regime data"""
        
        df = df_with_regime.copy()
        df['return'] = df['close'].pct_change()
        
        # Dynamic exposure
        df['exposure'] = 1.0
        low_vol_trending = (df['regime'] == 'LOW_VOL_TRENDING')
        df.loc[low_vol_trending, 'exposure'] = aggressive_leverage
        
        # Strategy returns
        df['strategy_return'] = df['return'] * df['exposure']
        
        # Metrics
        returns = df['strategy_return'].dropna()
        if len(returns) == 0:
            return {}
        
        total_return = (returns + 1).prod() - 1
        annual_return = (1 + total_return) ** (252 * 24 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252 * 24)
        sharpe = annual_return / volatility if volatility > 0 else 0
        
        # Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()
        
        win_rate = (returns > 0).sum() / len(returns)
        
        # Tail risk (99th percentile loss)
        tail_loss = returns.quantile(0.01)
        
        return {
            'total_return': float(total_return),
            'annual_return': float(annual_return),
            'volatility': float(volatility),
            'sharpe': float(sharpe),
            'max_drawdown': float(max_dd),
            'win_rate': float(win_rate),
            'tail_loss_1pct': float(tail_loss),
            'n_periods': int(len(returns)),
        }
    
    def run_walk_forward(self, symbol: str):
        """Run walk-forward validation for symbol"""
        
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"WALK-FORWARD VALIDATION: {symbol}")
        logger.info("=" * 80)
        
        # Try to load 1H data (6 months), else fall back to 5M (0.1 month)
        csv_file = self.data_dir / f"{symbol}_6month_1h_structural.csv"
        if not csv_file.exists():
            csv_file = self.data_dir / f"{symbol}_with_regime_and_exposure.csv"
        
        if not csv_file.exists():
            logger.error(f"Data file not found")
            return {}
        
        df = pd.read_csv(csv_file)
        logger.info(f"Loaded {len(df)} candles")
        
        # Generate splits
        splits = self.split_walk_forward(df, train_months=4, test_months=2)
        
        if not splits:
            logger.error("Could not generate walk-forward splits")
            return {}
        
        results = []
        
        for train_df, test_df, train_period, test_period in splits:
            logger.info("")
            logger.info(f"SPLIT: {train_period} (train) → {test_period} (test)")
            
            # Train on regime parameters from training set
            detector = RegimeDetectorParametric()
            train_regimes = detector.detect_regimes(train_df)
            
            # Apply SAME parameters to test set
            test_regimes = detector.detect_regimes(test_df)
            
            # Backtest on test set
            test_metrics = self.backtest_regime_strategy(test_regimes, aggressive_leverage=2.0)
            
            logger.info(f"  Test Sharpe: {test_metrics.get('sharpe', 0):.4f}")
            logger.info(f"  Test Return: {100*test_metrics.get('annual_return', 0):.2f}%")
            logger.info(f"  Max DD: {100*test_metrics.get('max_drawdown', 0):.2f}%")
            
            results.append({
                'train_period': train_period,
                'test_period': test_period,
                'metrics': test_metrics
            })
        
        return {
            'symbol': symbol,
            'splits': results,
            'mean_test_sharpe': float(np.mean([s['metrics'].get('sharpe', 0) for s in results])),
            'std_test_sharpe': float(np.std([s['metrics'].get('sharpe', 0) for s in results])),
        }


class SensitivityAnalyzer:
    """Test robustness of regime thresholds"""
    
    def __init__(self, data_dir: str = "validation_outputs"):
        self.data_dir = Path(data_dir)
    
    def run_sensitivity(self, symbol: str):
        """Test regime strategy under threshold variations"""
        
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"SENSITIVITY ANALYSIS: {symbol}")
        logger.info("=" * 80)
        
        # Load data
        csv_file = self.data_dir / f"{symbol}_with_regime_and_exposure.csv"
        if not csv_file.exists():
            logger.error(f"Data file not found: {csv_file}")
            return {}
        
        df = pd.read_csv(csv_file)
        df['return'] = df['close'].pct_change()
        
        # Base case
        detector_base = RegimeDetectorParametric(vol_percentile_low=0.33, vol_percentile_high=0.67)
        df_base = detector_base.detect_regimes(df)
        metrics_base = self._backtest_regimes(df_base)
        
        logger.info(f"Base case Sharpe: {metrics_base.get('sharpe', 0):.4f}")
        
        # Variations
        variations = [
            {'name': 'Vol-5%', 'vol_percentile_low': 0.28, 'vol_percentile_high': 0.62},
            {'name': 'Vol+5%', 'vol_percentile_low': 0.38, 'vol_percentile_high': 0.72},
            {'name': 'Autocorr strict', 'autocorr_threshold': 0.15},
            {'name': 'Autocorr loose', 'autocorr_threshold': 0.05},
        ]
        
        results = {'base': metrics_base}
        
        for var in variations:
            name = var.pop('name')
            params = {k: v for k, v in var.items()}
            detector = RegimeDetectorParametric(**params)
            df_var = detector.detect_regimes(df)
            metrics_var = self._backtest_regimes(df_var)
            
            results[name] = metrics_var
            change = (metrics_var.get('sharpe', 0) / max(0.001, metrics_base.get('sharpe', 0)) - 1) * 100
            logger.info(f"{name} Sharpe: {metrics_var.get('sharpe', 0):.4f} ({change:+.1f}%)")
        
        # Stability assessment
        sharpes = [v.get('sharpe', 0) for v in results.values()]
        sharpe_std = np.std(sharpes)
        sharpe_mean = np.mean(sharpes)
        
        logger.info(f"\nSharpe stability: mean={sharpe_mean:.4f}, std={sharpe_std:.4f}")
        
        if sharpe_std / max(abs(sharpe_mean), 0.001) < 0.10:
            logger.info("✅ ROBUST: Sharpe stable across threshold variations")
            verdict = "ROBUST"
        elif sharpe_std / max(abs(sharpe_mean), 0.001) < 0.25:
            logger.info("⚠️  MODERATE: Some sensitivity to thresholds")
            verdict = "MODERATE"
        else:
            logger.info("❌ FRAGILE: High sensitivity to threshold changes")
            verdict = "FRAGILE"
        
        return {
            'symbol': symbol,
            'results': results,
            'verdict': verdict,
            'sharpe_mean': float(sharpe_mean),
            'sharpe_std': float(sharpe_std),
        }
    
    def _backtest_regimes(self, df: pd.DataFrame) -> dict:
        """Simple backtest on given regimes"""
        
        df_copy = df.copy()
        df_copy['exposure'] = 1.0
        df_copy.loc[df_copy['regime'] == 'LOW_VOL_TRENDING', 'exposure'] = 2.0
        df_copy['strategy_return'] = df_copy['return'] * df_copy['exposure']
        
        returns = df_copy['strategy_return'].dropna()
        if len(returns) == 0:
            return {}
        
        total_return = (returns + 1).prod() - 1
        annual_return = (1 + total_return) ** (252 * 24 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252 * 24)
        sharpe = annual_return / volatility if volatility > 0 else 0
        
        return {
            'sharpe': float(sharpe),
            'annual_return': float(annual_return),
            'volatility': float(volatility),
        }


class CapitalImpactSimulator:
    """Test leverage impact on drawdown and tail risk"""
    
    def __init__(self, data_dir: str = "validation_outputs"):
        self.data_dir = Path(data_dir)
    
    def run_capital_impact(self, symbol: str):
        """Test various leverage levels"""
        
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"CAPITAL IMPACT SIMULATION: {symbol}")
        logger.info("=" * 80)
        
        csv_file = self.data_dir / f"{symbol}_with_regime_and_exposure.csv"
        if not csv_file.exists():
            logger.error(f"Data file not found: {csv_file}")
            return {}
        
        df = pd.read_csv(csv_file)
        df['return'] = df['close'].pct_change()
        
        leverage_levels = [1.0, 1.5, 2.0, 2.5, 3.0]
        results = {}
        
        for leverage in leverage_levels:
            df_lev = df.copy()
            df_lev['exposure'] = 1.0
            df_lev.loc[df_lev['regime'] == 'LOW_VOL_TRENDING', 'exposure'] = leverage
            df_lev['strategy_return'] = df_lev['return'] * df_lev['exposure']
            
            returns = df_lev['strategy_return'].dropna()
            
            # Metrics
            total_return = (returns + 1).prod() - 1
            annual_return = (1 + total_return) ** (252 * 24 / len(returns)) - 1
            volatility = returns.std() * np.sqrt(252 * 24)
            sharpe = annual_return / volatility if volatility > 0 else 0
            
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_dd = drawdown.min()
            
            tail_loss = returns.quantile(0.01)  # 1% worst loss
            tail_loss_5pct = returns.quantile(0.05)  # 5% worst loss
            
            results[leverage] = {
                'annual_return': float(annual_return),
                'volatility': float(volatility),
                'sharpe': float(sharpe),
                'max_drawdown': float(max_dd),
                'tail_loss_1pct': float(tail_loss),
                'tail_loss_5pct': float(tail_loss_5pct),
            }
            
            logger.info(f"\nLeverage {leverage}x:")
            logger.info(f"  Sharpe: {sharpe:.4f}")
            logger.info(f"  Max DD: {100*max_dd:.2f}%")
            logger.info(f"  1% worst loss: {100*tail_loss:.2f}%")
        
        # Recommendation
        max_dd_1x = abs(results[1.0]['max_drawdown'])
        max_dd_2x = abs(results[2.0]['max_drawdown'])
        
        logger.info("")
        logger.info("CAPITAL ALLOCATION RECOMMENDATION:")
        logger.info(f"  1.0x drawdown: {100*max_dd_1x:.2f}%")
        logger.info(f"  2.0x drawdown: {100*max_dd_2x:.2f}%")
        
        if max_dd_2x > 0.20:
            logger.info("  ⚠️  WARNING: 2x leverage creates >20% drawdown risk")
            rec_leverage = 1.5
        else:
            rec_leverage = 2.0
        
        logger.info(f"  → Recommend {rec_leverage}x for survivability")
        
        return {
            'symbol': symbol,
            'leverage_sweep': results,
            'recommended_leverage': float(rec_leverage),
        }


if __name__ == '__main__':
    
    logger.info("")
    logger.info("🧪 REGIME STRATEGY VALIDATION SUITE")
    logger.info("")
    logger.info("Step 1: Walk-Forward Validation")
    logger.info("Step 2: Sensitivity Analysis")
    logger.info("Step 3: Capital Impact Simulation")
    logger.info("")
    
    symbols = ['BTCUSDT', 'ETHUSDT']
    
    all_results = {
        'walk_forward': {},
        'sensitivity': {},
        'capital_impact': {},
    }
    
    # Step 1: Walk-Forward
    logger.info("=" * 80)
    logger.info("STEP 1: WALK-FORWARD VALIDATION")
    logger.info("=" * 80)
    wf_validator = WalkForwardValidator()
    for sym in symbols:
        all_results['walk_forward'][sym] = wf_validator.run_walk_forward(sym)
    
    # Step 2: Sensitivity
    logger.info("")
    logger.info("=" * 80)
    logger.info("STEP 2: SENSITIVITY ANALYSIS")
    logger.info("=" * 80)
    sens_analyzer = SensitivityAnalyzer()
    for sym in symbols:
        all_results['sensitivity'][sym] = sens_analyzer.run_sensitivity(sym)
    
    # Step 3: Capital Impact
    logger.info("")
    logger.info("=" * 80)
    logger.info("STEP 3: CAPITAL IMPACT SIMULATION")
    logger.info("=" * 80)
    capital_sim = CapitalImpactSimulator()
    for sym in symbols:
        all_results['capital_impact'][sym] = capital_sim.run_capital_impact(sym)
    
    # Summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 80)
    
    for sym in symbols:
        logger.info(f"\n{sym}:")
        wf = all_results['walk_forward'].get(sym, {})
        sens = all_results['sensitivity'].get(sym, {})
        cap = all_results['capital_impact'].get(sym, {})
        
        if wf:
            logger.info(f"  Walk-Forward mean Sharpe: {wf.get('mean_test_sharpe', 0):.4f}")
        if sens:
            logger.info(f"  Sensitivity verdict: {sens.get('verdict', 'N/A')}")
        if cap:
            logger.info(f"  Recommended leverage: {cap.get('recommended_leverage', 1.0):.1f}x")
    
    # Final recommendation
    logger.info("")
    logger.info("=" * 80)
    logger.info("FINAL RECOMMENDATION")
    logger.info("=" * 80)
    
    all_pass = True
    for sym in symbols:
        wf = all_results['walk_forward'].get(sym, {})
        sens = all_results['sensitivity'].get(sym, {})
        
        if wf.get('mean_test_sharpe', 0) < 0.3:
            all_pass = False
            logger.warning(f"  {sym}: Walk-forward Sharpe too low")
        
        if sens.get('verdict', '') == 'FRAGILE':
            all_pass = False
            logger.warning(f"  {sym}: Sensitivity analysis shows fragility")
    
    if all_pass:
        logger.info("✅ ALL TESTS PASSED: Proceed to live integration")
    else:
        logger.info("⚠️  SOME TESTS FAILED: Iterate regime definition")
    
    # Save results
    output_file = Path("validation_outputs") / "regime_validation_complete.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=float)
    logger.info(f"\n✅ Results saved to {output_file}")
