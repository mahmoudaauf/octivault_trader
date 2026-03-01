"""
Deep Structural Imbalance Diagnostic (6-Month Real Data)

Purpose: Test if funding rate extremes predict liquidation cascades
on REAL Binance data over 6 months.

Success Criterion: P(cascade | extreme funding) exceeds 3× baseline
"""

import pandas as pd
import numpy as np
from scipy import stats
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DeepStructuralDiagnostic:
    """Analyze structural imbalances on real 6-month data"""
    
    def __init__(self, data_dir: str = "validation_outputs"):
        self.data_dir = Path(data_dir)
        self.output_dir = self.data_dir / "diagnostic_deep_structural_results"
        self.output_dir.mkdir(exist_ok=True)
    
    def run_diagnostic(self, symbols: list = None):
        """Run full diagnostic"""
        
        if symbols is None:
            symbols = ['BTCUSDT', 'ETHUSDT']
        
        logger.info("=" * 80)
        logger.info("DEEP STRUCTURAL IMBALANCE DIAGNOSTIC (6-MONTH REAL DATA)")
        logger.info("=" * 80)
        logger.info("")
        logger.info("Testing: P(cascade | extreme funding rate) > 3× baseline?")
        logger.info("")
        
        all_results = {}
        
        for symbol in symbols:
            logger.info("=" * 80)
            logger.info(f"{symbol}")
            logger.info("=" * 80)
            
            # Load data
            data_file = self.data_dir / f"{symbol}_6month_1h_structural.csv"
            if not data_file.exists():
                logger.error(f"Data file not found: {data_file}")
                continue
            
            df = pd.read_csv(data_file)
            logger.info(f"Loaded {len(df)} candles from {data_file.name}")
            
            # Calculate features
            df['funding_rate_pct'] = df['funding_rate'] * 100 if 'funding_rate' in df.columns else 0.0
            df['price_return'] = df['close'].pct_change()
            df['abs_return'] = df['price_return'].abs()
            
            # Define extreme conditions
            extreme_thresh_95 = df['funding_rate_pct'].quantile(0.95)
            elevated_thresh_90 = df['funding_rate_pct'].quantile(0.90)
            
            # Define cascade: 2%+ move within next 3 hours
            df['future_max_return'] = df['abs_return'].rolling(window=3, min_periods=1).max().shift(-2)
            df['is_cascade'] = df['future_max_return'] > 0.02
            
            logger.info("")
            logger.info("FUNDING RATE STATISTICS:")
            logger.info(f"  Mean: {df['funding_rate_pct'].mean():.4f}%")
            logger.info(f"  Std:  {df['funding_rate_pct'].std():.4f}%")
            logger.info(f"  Min:  {df['funding_rate_pct'].min():.4f}%")
            logger.info(f"  Max:  {df['funding_rate_pct'].max():.4f}%")
            
            logger.info("")
            logger.info("EXTREME CONDITION THRESHOLDS:")
            logger.info(f"  Extreme (95th): {extreme_thresh_95:.4f}%")
            logger.info(f"  Elevated (90th): {elevated_thresh_90:.4f}%")
            
            # Count conditions
            n_extreme = (df['funding_rate_pct'] > extreme_thresh_95).sum()
            n_elevated = (df['funding_rate_pct'] > elevated_thresh_90).sum()
            
            logger.info("")
            logger.info("CONDITION FREQUENCY:")
            logger.info(f"  Extreme funding:  {n_extreme:4d} ({100*n_extreme/len(df):5.1f}%)")
            logger.info(f"  Elevated funding: {n_elevated:4d} ({100*n_elevated/len(df):5.1f}%)")
            
            # Cascade analysis
            n_cascades = df['is_cascade'].sum()
            cascade_rate_overall = n_cascades / len(df)
            
            logger.info("")
            logger.info("CASCADE STATISTICS:")
            logger.info(f"  Total cascades (2%+ in 3h): {n_cascades} ({100*cascade_rate_overall:.1f}%)")
            
            # Conditional probability
            extreme_rows = df['funding_rate_pct'] > extreme_thresh_95
            cascades_after_extreme = df[extreme_rows]['is_cascade'].sum()
            prob_cascade_given_extreme = cascades_after_extreme / max(1, extreme_rows.sum())
            
            lift = prob_cascade_given_extreme / max(0.001, cascade_rate_overall)
            
            logger.info("")
            logger.info("CONDITIONAL PROBABILITY ANALYSIS:")
            logger.info(f"  P(cascade | extreme funding) = {100*prob_cascade_given_extreme:.1f}%")
            logger.info(f"  P(cascade | baseline)       = {100*cascade_rate_overall:.1f}%")
            logger.info(f"  Lift = {lift:.2f}x")
            
            # Verdict
            logger.info("")
            if lift > 3.0:
                logger.info("  ✅ STRONG SIGNAL: Extreme funding predicts cascades >3× baseline")
                verdict = "STRONG_SIGNAL"
            elif lift > 1.5:
                logger.info("  ⚠️  MODERATE SIGNAL: Extreme funding has some predictive power")
                verdict = "MODERATE_SIGNAL"
            else:
                logger.info("  ❌ WEAK SIGNAL: Extreme funding does not predict cascades")
                verdict = "WEAK_SIGNAL"
            
            # Save results
            results = {
                'symbol': symbol,
                'period': '6 months (Aug 2025 - Feb 2026)',
                'timeframe': '1H',
                'n_candles': len(df),
                'funding_rate': {
                    'mean': float(df['funding_rate_pct'].mean()),
                    'std': float(df['funding_rate_pct'].std()),
                    'min': float(df['funding_rate_pct'].min()),
                    'max': float(df['funding_rate_pct'].max()),
                    'p95': float(extreme_thresh_95),
                    'p90': float(elevated_thresh_90),
                },
                'conditions': {
                    'extreme_count': int(n_extreme),
                    'extreme_pct': float(100*n_extreme/len(df)),
                    'elevated_count': int(n_elevated),
                    'elevated_pct': float(100*n_elevated/len(df)),
                },
                'cascades': {
                    'total_cascades': int(n_cascades),
                    'cascade_rate': float(cascade_rate_overall),
                    'cascade_threshold': '2%+ move in next 3 hours',
                },
                'conditional_probability': {
                    'p_cascade_given_extreme': float(prob_cascade_given_extreme),
                    'p_cascade_baseline': float(cascade_rate_overall),
                    'lift': float(lift),
                },
                'verdict': verdict
            }
            
            all_results[symbol] = results
            
            logger.info("")
        
        # Summary
        logger.info("")
        logger.info("=" * 80)
        logger.info("SUMMARY")
        logger.info("=" * 80)
        
        for symbol, res in all_results.items():
            verdict = res['verdict']
            lift = res['conditional_probability']['lift']
            logger.info(f"{symbol}: {verdict} (lift: {lift:.2f}x)")
        
        # Save to JSON
        output_file = self.output_dir / "diagnostic_deep_structural_6m_results.json"
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=float)
        logger.info(f"✅ Results saved to {output_file}")
        
        # Decision
        logger.info("")
        logger.info("=" * 80)
        logger.info("DECISION")
        logger.info("=" * 80)
        
        verdicts = [res['verdict'] for res in all_results.values()]
        lifts = [res['conditional_probability']['lift'] for res in all_results.values()]
        
        if all(v == "STRONG_SIGNAL" for v in verdicts):
            logger.info("✅ PROCEED TO LSTM TRAINING")
            logger.info("   Funding rate extremes predict cascades across all symbols")
            logger.info("   Next: Train LSTM on structural features")
        elif any(v in ["STRONG_SIGNAL", "MODERATE_SIGNAL"] for v in verdicts):
            logger.info("⚠️  CONDITIONAL PROCEED")
            logger.info("   Signal detected on some symbols, weak on others")
            logger.info("   Recommend feature engineering before LSTM training")
        else:
            logger.info("❌ ABANDON STRUCTURAL APPROACH")
            logger.info("   Funding rate extremes do NOT predict cascades")
            logger.info("   Signal < 3× baseline across all symbols")
            logger.info("   Recommendation: Pivot to alternative alpha sources")
        
        logger.info("")
        logger.info(f"Average lift: {np.mean(lifts):.2f}x")
        logger.info("")
        
        return all_results


if __name__ == '__main__':
    diagnostic = DeepStructuralDiagnostic()
    results = diagnostic.run_diagnostic()
