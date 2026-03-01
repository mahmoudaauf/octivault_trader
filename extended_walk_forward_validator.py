"""
Extended Walk-Forward Validation - 24 Months

Validates regime strategy on full 24-month dataset with:
- 12-month rolling training window
- 6-month rolling test window
- Multiple folds for statistical confidence
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ExtendedWalkForwardValidator:
    """Walk-forward validation on 24-month dataset"""
    
    def __init__(self, data_dir: str = "validation_outputs"):
        self.data_dir = Path(data_dir)
        self.output_dir = self.data_dir
    
    def backtest_regime_strategy(self, df: pd.DataFrame) -> dict:
        """
        Backtest strategy: 2x leverage in LOW_VOL_TRENDING + UPTREND
        """
        
        test_df = df.copy()
        
        # Fill missing returns
        test_df['return'] = test_df['return'].fillna(0)
        
        # Strategy: 2x in LOW_VOL_TRENDING + UPTREND, 1x elsewhere
        test_df['exposure'] = 1.0
        
        low_vol_trending_uptrend = (
            (test_df['regime'] == 'LOW_VOL_TRENDING') &
            (test_df['macro_trend'] == 'UPTREND')
        )
        test_df.loc[low_vol_trending_uptrend, 'exposure'] = 2.0
        
        test_df['strategy_return'] = test_df['return'] * test_df['exposure']
        
        # Calculate metrics
        returns = test_df['strategy_return'].dropna()
        if len(returns) < 10:
            return {}
        
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + total_return) ** (252 * 24 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252 * 24)
        sharpe = annual_return / volatility if volatility > 0 else 0
        
        # Drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_dd = np.min(drawdown) if len(drawdown) > 0 else 0
        
        # Win rate
        win_rate = (returns > 0).sum() / len(returns)
        
        # Tail risk (1% VaR)
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
    
    def run_extended_walk_forward(self, symbol: str):
        """Run walk-forward validation with 12m train, 6m test windows"""
        
        logger.info("")
        logger.info("="*80)
        logger.info(f"EXTENDED WALK-FORWARD VALIDATION: {symbol}")
        logger.info(f"Setup: 12-month train, 6-month test, rolling forward")
        logger.info("="*80)
        
        # Load 24-month data
        csv_file = self.data_dir / f"{symbol}_24month_1h_extended.csv"
        if not csv_file.exists():
            logger.error(f"Data file not found: {csv_file}")
            return {}
        
        df = pd.read_csv(csv_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        total_candles = len(df)
        candles_per_month = 720  # 24 hours * 30 days
        total_months = total_candles / candles_per_month
        
        logger.info(f"Total data: {total_candles} candles ({total_months:.1f} months)")
        
        train_candles = int(12 * candles_per_month)  # 12 months
        test_candles = int(6 * candles_per_month)    # 6 months
        roll_candles = int(6 * candles_per_month)    # Roll forward by 6 months
        
        # Generate walk-forward splits
        splits = []
        
        # Fold 1: Months 1-12 train, 13-18 test
        if total_candles >= train_candles + test_candles:
            train_df = df.iloc[:train_candles].copy()
            test_df = df.iloc[train_candles:train_candles + test_candles].copy()
            splits.append((
                "Months 1-12",
                "Months 13-18",
                train_df,
                test_df
            ))
        
        # Fold 2: Months 7-18 train, 19-24 test
        if total_candles >= roll_candles + train_candles + test_candles:
            train_df = df.iloc[roll_candles:roll_candles + train_candles].copy()
            test_df = df.iloc[roll_candles + train_candles:roll_candles + train_candles + test_candles].copy()
            splits.append((
                "Months 7-18",
                "Months 19-24",
                train_df,
                test_df
            ))
        
        logger.info(f"Generated {len(splits)} walk-forward folds")
        
        results = []
        
        for fold_idx, (train_period, test_period, train_df, test_df) in enumerate(splits, 1):
            logger.info("")
            logger.info(f"FOLD {fold_idx}: {train_period} (train) → {test_period} (test)")
            
            # Backtest on test set
            metrics = self.backtest_regime_strategy(test_df)
            
            if metrics:
                logger.info(f"  Sharpe: {metrics['sharpe']:.4f}")
                logger.info(f"  Annual Return: {100*metrics['annual_return']:.2f}%")
                logger.info(f"  Volatility: {100*metrics['volatility']:.2f}%")
                logger.info(f"  Max DD: {100*metrics['max_drawdown']:.2f}%")
                logger.info(f"  Win Rate: {100*metrics['win_rate']:.1f}%")
            
            results.append({
                'fold': fold_idx,
                'train_period': train_period,
                'test_period': test_period,
                'metrics': metrics
            })
        
        # Summary
        logger.info("")
        logger.info("="*80)
        logger.info("EXTENDED WALK-FORWARD SUMMARY")
        logger.info("="*80)
        
        sharpe_values = [r['metrics'].get('sharpe', 0) for r in results if r['metrics']]
        annual_returns = [r['metrics'].get('annual_return', 0) for r in results if r['metrics']]
        max_dds = [r['metrics'].get('max_drawdown', 0) for r in results if r['metrics']]
        
        if sharpe_values:
            mean_sharpe = np.mean(sharpe_values)
            std_sharpe = np.std(sharpe_values)
            mean_return = np.mean(annual_returns)
            mean_dd = np.mean(max_dds)
            
            logger.info(f"\nSharpe Statistics:")
            logger.info(f"  Mean: {mean_sharpe:.4f}")
            logger.info(f"  Std: {std_sharpe:.4f}")
            logger.info(f"  Min/Max: {min(sharpe_values):.4f} / {max(sharpe_values):.4f}")
            
            logger.info(f"\nReturn Statistics:")
            logger.info(f"  Mean Annual: {100*mean_return:.2f}%")
            logger.info(f"  Min/Max: {100*min(annual_returns):.2f}% / {100*max(annual_returns):.2f}%")
            
            logger.info(f"\nDrawdown Statistics:")
            logger.info(f"  Mean Max DD: {100*mean_dd:.2f}%")
            logger.info(f"  Worst: {100*min(max_dds):.2f}%")
            
            # Verdict
            logger.info("")
            if mean_sharpe > 0.3:
                logger.info(f"✅ SIGNAL PERSISTENT (Mean Sharpe {mean_sharpe:.4f} > 0.3)")
                logger.info("   Edge validated across 24-month multiple folds!")
            elif mean_sharpe > 0:
                logger.info(f"⚠️ WEAK SIGNAL (Mean Sharpe {mean_sharpe:.4f}, positive but < 0.3)")
                logger.info("   Edge exists but needs improvement")
            else:
                logger.info(f"❌ NO SIGNAL (Mean Sharpe {mean_sharpe:.4f} < 0)")
                logger.info("   Edge does not persist out-of-sample")
        
        return {
            'symbol': symbol,
            'folds': results,
            'mean_sharpe': float(np.mean(sharpe_values)) if sharpe_values else 0,
            'std_sharpe': float(np.std(sharpe_values)) if sharpe_values else 0,
            'mean_return': float(np.mean(annual_returns)) if annual_returns else 0,
            'mean_max_dd': float(np.mean(max_dds)) if max_dds else 0,
        }


if __name__ == '__main__':
    validator = ExtendedWalkForwardValidator()
    
    # Run for both symbols
    btc_results = validator.run_extended_walk_forward('BTCUSDT')
    eth_results = validator.run_extended_walk_forward('ETHUSDT')
    
    # Save results
    all_results = {
        'BTCUSDT': btc_results,
        'ETHUSDT': eth_results,
        'timestamp': pd.Timestamp.now().isoformat(),
    }
    
    output_file = Path('validation_outputs') / 'extended_walk_forward_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"\n✅ Extended walk-forward validation complete!")
    logger.info(f"Results saved to {output_file}")
