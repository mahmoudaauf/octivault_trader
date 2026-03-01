"""
PHASE 9.5 DIAGNOSTIC: CONDITIONAL PROBABILITY ANALYSIS

Goal: Determine if there is ANY signal to learn

Question: Does P(expansion | feature_state) vary meaningfully from P(expansion)?

If answer is YES → Signal exists, worth modeling further
If answer is NO  → No conditional structure, abandon this approach

Method:
  1. For each feature (return, volatility, momentum, position)
  2. Bin into quintiles (5 buckets)
  3. Calculate P(expansion | feature in bucket)
  4. Compare to baseline P(expansion)
  5. Measure information gain / chi-square significance
  6. Report which features have predictive power

Key Insight:
  If NO feature shows meaningful conditional probability variation,
  then LSTM training is futile.
  
  But if SOME features DO vary,
  we know signal exists and can be extracted.
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from scipy import stats
from dataclasses import dataclass
from typing import Dict, Tuple, List

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger('ConditionalProbabilityDiagnostic')


@dataclass
class DiagnosticConfig:
    """Configuration for diagnostic"""
    event_data_dir: str = "validation_outputs"
    output_dir: str = "validation_outputs"
    bins: int = 5  # Quintiles


class ConditionalProbabilityAnalyzer:
    """Analyze if features have predictive signal"""
    
    def __init__(self, config: DiagnosticConfig):
        self.config = config
    
    def analyze_symbol(self, symbol: str, df: pd.DataFrame) -> Dict:
        """Analyze conditional probabilities for one symbol"""
        
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"Analyzing: {symbol}")
        logger.info("=" * 80)
        
        # Calculate baseline
        baseline_expansion_rate = np.mean(df['event_target'] == 1)
        logger.info(f"Baseline P(expansion) = {baseline_expansion_rate*100:.2f}%")
        logger.info("")
        
        # Calculate features
        df['return'] = df['close'].pct_change()
        df['volatility'] = df['return'].rolling(window=5).std()
        df['bar_direction'] = (df['close'] > df['open']).astype(int)
        df['momentum'] = df['bar_direction'].rolling(window=5).mean()
        
        # Price position
        df['high_20'] = df['high'].rolling(window=20).max()
        df['low_20'] = df['low'].rolling(window=20).min()
        df['position'] = (df['close'] - df['low_20']) / (df['high_20'] - df['low_20'])
        df['position'] = df['position'].fillna(0.5).clip(0, 1)
        
        results = {
            'baseline_expansion_rate': float(baseline_expansion_rate),
            'features': {}
        }
        
        # Analyze each feature
        features_to_analyze = ['return', 'volatility', 'momentum', 'position']
        
        for feature_name in features_to_analyze:
            logger.info(f"Feature: {feature_name}")
            logger.info("-" * 80)
            
            # Get feature values (skip NaN)
            mask = df[feature_name].notna() & df['event_target'].notna()
            X = df.loc[mask, feature_name].values
            y = df.loc[mask, 'event_target'].values
            
            if len(X) < 50:
                logger.info(f"  ⚠️  Insufficient data ({len(X)} < 50)")
                continue
            
            # Create bins
            try:
                bins = pd.qcut(X, q=self.config.bins, duplicates='drop')
            except:
                logger.info(f"  ⚠️  Cannot bin feature (insufficient variation)")
                continue
            
            # Calculate conditional probabilities
            contingency = pd.crosstab(bins, y)
            
            conditional_probs = []
            chi_square_stats = []
            bin_labels = []
            bin_sizes = []
            
            for bin_idx in contingency.index:
                if 1 not in contingency.columns:
                    expansion_count = 0
                else:
                    expansion_count = contingency.loc[bin_idx, 1] if 1 in contingency.columns else 0
                
                total_count = contingency.loc[bin_idx].sum()
                bin_sizes.append(int(total_count))
                
                if total_count > 0:
                    prob = expansion_count / total_count
                    conditional_probs.append(prob)
                    
                    # Expected count under independence
                    expected = total_count * baseline_expansion_rate
                    chi_stat = ((expansion_count - expected) ** 2) / (expected + 1e-10)
                    chi_square_stats.append(chi_stat)
                else:
                    conditional_probs.append(0)
                    chi_square_stats.append(0)
                
                bin_labels.append(f"Q{len(conditional_probs)}")
            
            # Calculate information metrics
            chi_square_total = np.sum(chi_square_stats)
            p_value = 1 - stats.chi2.cdf(chi_square_total, df=self.config.bins - 1) if chi_square_total > 0 else 1.0
            
            # Calculate information gain (entropy reduction)
            baseline_entropy = -baseline_expansion_rate * np.log2(baseline_expansion_rate + 1e-10) - \
                               (1 - baseline_expansion_rate) * np.log2(1 - baseline_expansion_rate + 1e-10)
            
            conditional_entropy = 0
            total_samples = np.sum(bin_sizes)
            for idx, (prob, size) in enumerate(zip(conditional_probs, bin_sizes)):
                if prob > 0 and prob < 1:
                    h = -prob * np.log2(prob) - (1 - prob) * np.log2(1 - prob)
                    conditional_entropy += (size / total_samples) * h
                elif prob == 0 or prob == 1:
                    # Pure bins have 0 entropy
                    conditional_entropy += 0
            
            information_gain = baseline_entropy - conditional_entropy
            
            # Log results
            logger.info(f"  Conditional P(expansion) by quintile:")
            max_prob = max(conditional_probs)
            min_prob = min(conditional_probs)
            prob_range = max_prob - min_prob
            
            for label, prob, size in zip(bin_labels, conditional_probs, bin_sizes):
                marker = ""
                if prob > baseline_expansion_rate * 1.2:
                    marker = " ↑ HIGHER"
                elif prob < baseline_expansion_rate * 0.8:
                    marker = " ↓ LOWER"
                logger.info(f"    {label}: {prob*100:5.1f}% (n={size:3d}){marker}")
            
            logger.info(f"")
            logger.info(f"  Range: {min_prob*100:.1f}% → {max_prob*100:.1f}% (Δ={prob_range*100:.1f}%)")
            logger.info(f"  Chi-Square: {chi_square_total:.2f}, p-value: {p_value:.4f}")
            logger.info(f"  Information Gain: {information_gain:.4f} bits")
            logger.info("")
            
            # Assess significance
            if p_value < 0.05 and prob_range > 0.05:
                significance = "✅ SIGNIFICANT"
            elif prob_range > 0.03:
                significance = "⚠️  MARGINAL"
            else:
                significance = "❌ NO SIGNAL"
            
            logger.info(f"  Assessment: {significance}")
            logger.info("")
            
            results['features'][feature_name] = {
                'baseline_expansion_rate': float(baseline_expansion_rate),
                'conditional_by_quintile': {
                    label: float(prob) 
                    for label, prob in zip(bin_labels, conditional_probs)
                },
                'bin_sizes': bin_sizes,
                'min_prob': float(min_prob),
                'max_prob': float(max_prob),
                'prob_range': float(prob_range),
                'chi_square': float(chi_square_total),
                'p_value': float(p_value),
                'information_gain': float(information_gain),
                'significance': significance
            }
        
        return results


class DiagnosticRunner:
    """Run full diagnostic"""
    
    def __init__(self, config: DiagnosticConfig):
        self.config = config
        self.all_results = {}
    
    def run(self):
        """Execute diagnostic for all symbols"""
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("CONDITIONAL PROBABILITY DIAGNOSTIC")
        logger.info("=" * 80)
        logger.info("")
        logger.info("Question: Does P(expansion | feature) vary meaningfully?")
        logger.info("")
        logger.info("If YES → Signal exists to learn")
        logger.info("If NO  → Abandon expansion modeling")
        logger.info("")
        
        # Load event-based targets
        data_dir = Path(self.config.event_data_dir)
        event_files = list(data_dir.glob("*_event_based_targets.csv"))
        
        if not event_files:
            logger.error("No event files found")
            return
        
        for file_path in sorted(event_files):
            symbol = file_path.stem.replace('_event_based_targets', '')
            
            # Load events
            df_events = pd.read_csv(file_path)
            
            # Load OHLCV
            ohlcv_file = Path(self.config.event_data_dir) / f"{symbol}_5m_with_60m_labels.csv"
            if not ohlcv_file.exists():
                logger.error(f"OHLCV not found: {ohlcv_file}")
                continue
            
            df_ohlcv = pd.read_csv(ohlcv_file)
            
            # Merge
            df = pd.merge(df_ohlcv, df_events[['timestamp', 'event_target']], 
                         on='timestamp', how='inner')
            
            # Analyze
            analyzer = ConditionalProbabilityAnalyzer(self.config)
            results = analyzer.analyze_symbol(symbol, df)
            self.all_results[symbol] = results
        
        # Save results
        output_file = Path(self.config.output_dir) / "diagnostic_conditional_probabilities.json"
        with open(output_file, 'w') as f:
            json.dump(self.all_results, f, indent=2)
        
        logger.info("=" * 80)
        logger.info("DIAGNOSTIC SUMMARY")
        logger.info("=" * 80)
        logger.info("")
        
        self._print_summary()
    
    def _print_summary(self):
        """Print summary assessment"""
        
        for symbol, results in self.all_results.items():
            logger.info(f"{symbol}:")
            
            significant_features = []
            marginal_features = []
            no_signal_features = []
            
            for feat_name, feat_results in results.get('features', {}).items():
                sig = feat_results['significance']
                if "SIGNIFICANT" in sig:
                    significant_features.append(feat_name)
                elif "MARGINAL" in sig:
                    marginal_features.append(feat_name)
                else:
                    no_signal_features.append(feat_name)
            
            logger.info(f"  Significant: {', '.join(significant_features) if significant_features else 'NONE'}")
            logger.info(f"  Marginal:    {', '.join(marginal_features) if marginal_features else 'NONE'}")
            logger.info(f"  No signal:   {', '.join(no_signal_features) if no_signal_features else 'NONE'}")
            logger.info("")
        
        logger.info("=" * 80)
        logger.info("🎯 INTERPRETATION")
        logger.info("=" * 80)
        logger.info("")
        
        all_no_signal = all(
            all("NO SIGNAL" in feat_results['significance'] 
                for feat_results in results.get('features', {}).values())
            for results in self.all_results.values()
        )
        
        all_significant = all(
            any("SIGNIFICANT" in feat_results['significance'] 
                for feat_results in results.get('features', {}).values())
            for results in self.all_results.values()
        )
        
        if all_no_signal:
            logger.info("❌ NO CONDITIONAL STRUCTURE DETECTED")
            logger.info("")
            logger.info("Conclusion: Features do NOT predict expansion vs reversal")
            logger.info("Action: ABANDON event-based expansion modeling")
            logger.info("")
            logger.info("Next Steps:")
            logger.info("  1. Expand feature engineering significantly")
            logger.info("  2. Include order flow, liquidation, volume spikes")
            logger.info("  3. Or: Use longer-horizon directional targets")
            logger.info("  4. Or: Shift to regime detection instead")
        
        elif all_significant:
            logger.info("✅ STRONG CONDITIONAL STRUCTURE DETECTED")
            logger.info("")
            logger.info("Conclusion: Features DO predict expansion vs reversal")
            logger.info("Action: LSTM should work with improved feature engineering")
            logger.info("")
            logger.info("Next Steps:")
            logger.info("  1. Use identified significant features")
            logger.info("  2. Consider ensemble methods (not just LSTM)")
            logger.info("  3. Feature interactions and non-linear combinations")
            logger.info("  4. Increase lookback to capture longer dependencies")
        
        else:
            logger.info("⚠️  PARTIAL SIGNAL DETECTED")
            logger.info("")
            logger.info("Conclusion: Some features show conditional structure")
            logger.info("Action: Worth exploring further, but not guaranteed")
            logger.info("")
            logger.info("Next Steps:")
            logger.info("  1. Focus LSTM on significant features only")
            logger.info("  2. Simplify model (fewer parameters)")
            logger.info("  3. Use identified features with XGBoost first")
            logger.info("  4. Validate on fresh data before scaling up")
        
        logger.info("")


if __name__ == "__main__":
    config = DiagnosticConfig()
    runner = DiagnosticRunner(config)
    runner.run()
