"""
Regime-Based Exposure Backtest

Test: Does dynamic exposure control improve risk-adjusted returns?

Strategy:
- Detect LOW_VOL_TRENDING regime (high confidence)
- Increase exposure to 1.5-2.0x in those periods
- Reduce exposure in other regimes

Metric: Sharpe ratio improvement vs static exposure
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RegimeExposureBacktest:
    """Backtest regime-based exposure control"""
    
    def __init__(self, output_dir: str = "validation_outputs"):
        self.output_dir = Path(output_dir)
    
    def backtest_symbol(self, symbol: str) -> dict:
        """
        Backtest regime exposure on single symbol.
        
        Returns:
            Dict with backtest metrics
        """
        
        # Load regime data
        data_file = self.output_dir / f"{symbol}_with_regime_and_exposure.csv"
        if not data_file.exists():
            logger.error(f"Data file not found: {data_file}")
            return {}
        
        df = pd.read_csv(data_file)
        logger.info(f"Loaded {len(df)} candles for {symbol}")
        
        # Strategy 1: Static exposure (1.0x all the time)
        df['static_return'] = df['return'] * 1.0
        
        # Strategy 2: Dynamic exposure based on regime
        df['dynamic_return'] = df['return'] * df['exposure_smooth']
        
        # Strategy 3: Aggressive in LOW_VOL_TRENDING only
        df['aggressive_exposure'] = 1.0
        low_vol_trending = (df['regime'] == 'LOW_VOL_TRENDING')
        df.loc[low_vol_trending, 'aggressive_exposure'] = 2.0
        df['aggressive_return'] = df['return'] * df['aggressive_exposure']
        
        # Calculate cumulative returns
        df['static_cumsum'] = (1 + df['static_return']).cumprod()
        df['dynamic_cumsum'] = (1 + df['dynamic_return']).cumprod()
        df['aggressive_cumsum'] = (1 + df['aggressive_return']).cumprod()
        
        # Metrics
        def calc_metrics(returns_series, name):
            returns = returns_series.dropna()
            if len(returns) == 0:
                return {}
            
            total_return = (returns + 1).prod() - 1
            annual_return = (1 + total_return) ** (252 * 24 / len(returns)) - 1  # 5m to annual
            volatility = returns.std() * np.sqrt(252 * 24)  # 5m to annual
            sharpe = annual_return / volatility if volatility > 0 else 0
            max_dd = np.min(np.minimum.accumulate(1 + returns.cumsum()) - 1)
            
            return {
                'total_return': float(total_return),
                'annual_return': float(annual_return),
                'volatility': float(volatility),
                'sharpe': float(sharpe),
                'max_drawdown': float(max_dd),
                'win_rate': float((returns > 0).sum() / len(returns)),
            }
        
        static_metrics = calc_metrics(df['static_return'], 'Static')
        dynamic_metrics = calc_metrics(df['dynamic_return'], 'Dynamic')
        aggressive_metrics = calc_metrics(df['aggressive_return'], 'Aggressive')
        
        logger.info("")
        logger.info(f"BACKTEST RESULTS: {symbol}")
        logger.info("")
        
        logger.info("STATIC EXPOSURE (1.0x all the time):")
        for key, val in static_metrics.items():
            if isinstance(val, float):
                logger.info(f"  {key}: {val:.4f}")
        
        logger.info("")
        logger.info("DYNAMIC EXPOSURE (0.5x - 2.0x based on regime):")
        for key, val in dynamic_metrics.items():
            if isinstance(val, float):
                logger.info(f"  {key}: {val:.4f}")
        
        logger.info("")
        logger.info("AGGRESSIVE EXPOSURE (2.0x in LOW_VOL_TRENDING, 1.0x else):")
        for key, val in aggressive_metrics.items():
            if isinstance(val, float):
                logger.info(f"  {key}: {val:.4f}")
        
        # Improvement
        if static_metrics.get('sharpe', 0) != 0:
            dynamic_improvement = (dynamic_metrics.get('sharpe', 0) / static_metrics['sharpe'] - 1) * 100
            aggressive_improvement = (aggressive_metrics.get('sharpe', 0) / static_metrics['sharpe'] - 1) * 100
        else:
            dynamic_improvement = 0
            aggressive_improvement = 0
        
        logger.info("")
        logger.info("IMPROVEMENT:")
        logger.info(f"  Dynamic vs Static Sharpe: {dynamic_improvement:+.1f}%")
        logger.info(f"  Aggressive vs Static Sharpe: {aggressive_improvement:+.1f}%")
        
        # LOW_VOL_TRENDING analysis
        low_vol_trending_df = df[df['regime'] == 'LOW_VOL_TRENDING']
        if len(low_vol_trending_df) > 0:
            logger.info("")
            logger.info(f"LOW_VOL_TRENDING ANALYSIS:")
            logger.info(f"  Count: {len(low_vol_trending_df)} ({100*len(low_vol_trending_df)/len(df):.1f}%)")
            logger.info(f"  Mean return: {100*low_vol_trending_df['return'].mean():.4f}%")
            logger.info(f"  Win rate: {100*(low_vol_trending_df['return'] > 0).sum() / len(low_vol_trending_df):.1f}%")
        
        return {
            'symbol': symbol,
            'n_candles': len(df),
            'static': static_metrics,
            'dynamic': dynamic_metrics,
            'aggressive': aggressive_metrics,
            'dynamic_improvement_pct': float(dynamic_improvement),
            'aggressive_improvement_pct': float(aggressive_improvement),
        }
    
    def run_backtest(self, symbols: list = None):
        """Run backtest for all symbols"""
        
        if symbols is None:
            symbols = ['BTCUSDT', 'ETHUSDT']
        
        logger.info("=" * 80)
        logger.info("REGIME-BASED EXPOSURE BACKTEST")
        logger.info("=" * 80)
        logger.info("")
        logger.info("Hypothesis: Dynamic exposure control improves Sharpe ratio")
        logger.info("")
        
        all_results = {}
        
        for symbol in symbols:
            logger.info("=" * 80)
            result = self.backtest_symbol(symbol)
            all_results[symbol] = result
        
        # Summary
        logger.info("")
        logger.info("=" * 80)
        logger.info("BACKTEST SUMMARY")
        logger.info("=" * 80)
        logger.info("")
        
        for symbol, result in all_results.items():
            if result:
                static_sharpe = result['static']['sharpe']
                dynamic_sharpe = result['dynamic']['sharpe']
                aggressive_sharpe = result['aggressive']['sharpe']
                
                logger.info(f"{symbol}:")
                logger.info(f"  Static Sharpe: {static_sharpe:.4f}")
                logger.info(f"  Dynamic Sharpe: {dynamic_sharpe:.4f} ({result['dynamic_improvement_pct']:+.1f}%)")
                logger.info(f"  Aggressive Sharpe: {aggressive_sharpe:.4f} ({result['aggressive_improvement_pct']:+.1f}%)")
                logger.info("")
        
        # Decision
        avg_improvement = np.mean([r.get('aggressive_improvement_pct', 0) for r in all_results.values()])
        
        logger.info("=" * 80)
        logger.info("DECISION")
        logger.info("=" * 80)
        logger.info("")
        
        if avg_improvement > 5:
            logger.info("✅ SIGNIFICANT IMPROVEMENT")
            logger.info("   Regime-based exposure control meaningfully improves Sharpe")
            logger.info("   Recommend: Integrate into live trading system")
            verdict = "PROCEED"
        elif avg_improvement > 0:
            logger.info("✅ MODEST IMPROVEMENT")
            logger.info("   Regime-based sizing shows positive edge")
            logger.info("   Recommend: Further optimization before live deployment")
            verdict = "OPTIMIZE"
        else:
            logger.info("❌ NO IMPROVEMENT")
            logger.info("   Dynamic sizing does not improve returns")
            logger.info("   Recommend: Return to static sizing or try alternative approach")
            verdict = "RECONSIDER"
        
        logger.info("")
        logger.info(f"Average Sharpe improvement: {avg_improvement:+.1f}%")
        logger.info("")
        
        # Save results
        output_file = self.output_dir / "regime_exposure_backtest_results.json"
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=float)
        logger.info(f"✅ Results saved to {output_file}")
        
        return all_results, verdict


if __name__ == '__main__':
    
    logger.info("")
    logger.info("🎯 REGIME-BASED EXPOSURE BACKTEST")
    logger.info("")
    
    backtester = RegimeExposureBacktest(output_dir='validation_outputs')
    results, verdict = backtester.run_backtest()
    
    logger.info("")
    logger.info(f"Final Verdict: {verdict}")
