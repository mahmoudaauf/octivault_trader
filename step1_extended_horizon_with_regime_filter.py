"""
PHASE 9.5 STEP 1 - EXTENDED: 60m Labels with Regime-Aware Filtering

Strategy Pivot (All Three):
  1. Extend horizon from 30m (6 candles) → 60m (12 candles)
  2. Add volatility regime classification
  3. Drop ALL predictions in low-volatility regimes (keep only normal/high)

Rationale:
  - Real data showed 30m moves too small (0.15-0.19% vs 0.30% needed)
  - 60m gives √2 = 1.41x improvement (expected 0.21-0.27% moves)
  - Regime filtering removes noise, focuses on tradeable conditions
  - Result: Higher quality labels with meaningful signal

Key Changes from Original Step 1:
  - Horizon: 6 candles (30m) → 12 candles (60m)
  - Regime detection: 3-state (low/normal/high volatility)
  - Filter: Drop low-vol regime entirely (keep only normal/high)
  - Expected result: 25-35% positive targets, >= 0.25% median move
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('Step1_ExtendedRegimeAware')


@dataclass
class Step1ExtendedConfig:
    """Configuration for extended Step 1 with regime filtering"""
    
    # Data paths
    data_dir: str = "data/historical"
    output_dir: str = "validation_outputs"
    
    # Horizon: 12 candles = 60 minutes (extended from 6 candles = 30m)
    prediction_horizon_candles: int = 12
    
    # Edge threshold: still 0.20% but now on 60m moves
    edge_threshold_pct: float = 0.0020
    
    # Regime detection: rolling window volatility
    regime_window_candles: int = 20  # 100-minute rolling window
    
    # Volatility regimes: percentile thresholds
    low_vol_percentile: float = 0.33   # Bottom 33% = low volatility
    high_vol_percentile: float = 0.67  # Top 33% = high volatility
    
    # Filtering: keep only these regimes
    regimes_to_keep: List[str] = None  # Will default to ["normal", "high"]
    
    # Data quality
    min_data_points: int = 100
    
    def __post_init__(self):
        if self.regimes_to_keep is None:
            self.regimes_to_keep = ["normal", "high"]


class RegimeDetector:
    """Detect volatility regimes (low/normal/high)"""
    
    def __init__(self, window: int, low_pctl: float, high_pctl: float):
        self.window = window
        self.low_pctl = low_pctl
        self.high_pctl = high_pctl
    
    def detect_regimes(self, df: pd.DataFrame) -> pd.Series:
        """
        Classify each candle into volatility regime
        
        Returns:
            Series with values: "low", "normal", "high"
        """
        # Calculate rolling volatility (std of returns)
        df_copy = df.copy()
        df_copy['return'] = df_copy['close'].pct_change()
        df_copy['volatility'] = df_copy['return'].rolling(window=self.window).std()
        
        # Get percentile thresholds (ignoring NaNs from rolling window)
        vol_values = df_copy['volatility'].dropna()
        if len(vol_values) < 10:
            # Not enough data for regime detection
            return pd.Series('normal', index=df.index)
        
        low_threshold = vol_values.quantile(self.low_pctl)
        high_threshold = vol_values.quantile(self.high_pctl)
        
        # Classify into regimes
        regimes = pd.Series('normal', index=df.index)
        regimes[df_copy['volatility'] <= low_threshold] = 'low'
        regimes[df_copy['volatility'] >= high_threshold] = 'high'
        
        return regimes, low_threshold, high_threshold
    
    def get_regime_summary(self, regimes: pd.Series) -> Dict:
        """Get summary of regime distribution"""
        counts = regimes.value_counts()
        return {
            'low': int(counts.get('low', 0)),
            'normal': int(counts.get('normal', 0)),
            'high': int(counts.get('high', 0)),
            'total': len(regimes)
        }


class DataLoader:
    """Load OHLCV data from CSV files"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
    
    def load_symbol_data(self, symbol: str) -> Tuple[pd.DataFrame, bool]:
        """
        Load OHLCV data for a symbol
        
        Returns:
            (DataFrame, success_bool)
        """
        csv_path = self.data_dir / f"{symbol}_5m.csv"
        
        if not csv_path.exists():
            logger.error(f"❌ Data file not found: {csv_path}")
            return None, False
        
        try:
            df = pd.read_csv(csv_path)
            
            # Validate columns
            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                logger.error(f"❌ Missing required columns in {csv_path}")
                return None, False
            
            # Ensure correct data types
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Check for NaNs
            if df[required_cols].isna().any().any():
                logger.error(f"❌ Found NaN values in {symbol} data")
                return None, False
            
            logger.info(f"✅ Loaded {len(df)} rows for {symbol}")
            return df, True
            
        except Exception as e:
            logger.error(f"❌ Error loading {symbol}: {e}")
            return None, False
    
    def get_available_symbols(self) -> List[str]:
        """Auto-detect symbols from data directory"""
        csv_files = list(self.data_dir.glob("*_5m.csv"))
        symbols = [f.stem.replace("_5m", "") for f in csv_files]
        return sorted(symbols)


class LabelConstructor:
    """Construct 60m labels with regime filtering"""
    
    def __init__(self, config: Step1ExtendedConfig):
        self.horizon = config.prediction_horizon_candles
        self.threshold = config.edge_threshold_pct
        self.regime_detector = RegimeDetector(
            window=config.regime_window_candles,
            low_pctl=config.low_vol_percentile,
            high_pctl=config.high_vol_percentile
        )
        self.regimes_to_keep = config.regimes_to_keep
    
    def build_labels(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Build 60m labels with regime filtering
        
        Process:
          1. Detect volatility regimes
          2. Calculate forward 60m returns
          3. Create labels (0/1)
          4. Filter out low-volatility regimes
        
        Returns:
            (labeled_df, metadata)
        """
        df_copy = df.copy()
        n = len(df_copy)
        
        # Step 1: Detect regimes
        logger.info(f"Detecting volatility regimes (window={self.regime_detector.window})...")
        regimes, low_thresh, high_thresh = self.regime_detector.detect_regimes(df_copy)
        df_copy['regime'] = regimes
        
        regime_summary = self.regime_detector.get_regime_summary(regimes)
        logger.info(f"  Regime distribution: Low={regime_summary['low']}, "
                   f"Normal={regime_summary['normal']}, "
                   f"High={regime_summary['high']}")
        
        # Step 2: Initialize label columns
        df_copy['forward_return_60m'] = np.nan
        df_copy['target_60m'] = np.nan
        df_copy['forward_return_valid'] = False
        
        # Step 3: Calculate forward returns and labels
        logger.info(f"Constructing 60m labels (horizon={self.horizon} candles, "
                   f"threshold={self.threshold*100:.2f}%)...")
        
        valid_count = 0
        for i in range(n - self.horizon):
            entry_price = df_copy['close'].iloc[i]
            exit_price = df_copy['close'].iloc[i + self.horizon]
            cumulative_return = (exit_price - entry_price) / entry_price
            
            df_copy.loc[i, 'forward_return_60m'] = cumulative_return
            df_copy.loc[i, 'target_60m'] = 1 if cumulative_return > self.threshold else 0
            df_copy.loc[i, 'forward_return_valid'] = True
            
            valid_count += 1
        
        logger.info(f"✅ Created {valid_count} valid labels (with {self.horizon}-candle horizon)")
        
        # Step 4: Filter to keep only selected regimes
        logger.info(f"Filtering to keep only regimes: {self.regimes_to_keep}...")
        
        # Mark rows to keep
        df_copy['keep_for_training'] = df_copy['regime'].isin(self.regimes_to_keep)
        
        kept_count = df_copy['keep_for_training'].sum()
        dropped_count = (~df_copy['keep_for_training']).sum()
        
        logger.info(f"  Keeping {kept_count} labels in tradeable regimes")
        logger.info(f"  Dropping {dropped_count} labels in low-volatility regime")
        
        # Create filtered dataset (for metrics)
        df_filtered = df_copy[df_copy['keep_for_training']].copy()
        
        return df_copy, df_filtered, {
            'total_candles': n,
            'valid_labels_total': valid_count,
            'valid_labels_after_filter': len(df_filtered),
            'regime_distribution': regime_summary,
            'volatility_thresholds': {
                'low_vol_threshold': float(low_thresh),
                'high_vol_threshold': float(high_thresh)
            }
        }


class TargetAnalyzer:
    """Analyze target distribution and statistics"""
    
    @staticmethod
    def analyze_distribution(df: pd.DataFrame, threshold: float) -> Dict:
        """Analyze return distribution and target statistics"""
        
        # Get valid labels
        valid_mask = df['forward_return_valid'] == True
        valid_returns = df[valid_mask]['forward_return_60m']
        valid_targets = df[valid_mask]['target_60m']
        
        if len(valid_returns) == 0:
            return {}
        
        # Return statistics
        stats = {
            'total_observations': int(valid_mask.sum()),
            'mean': float(valid_returns.mean()),
            'median': float(valid_returns.median()),
            'std': float(valid_returns.std()),
            'min': float(valid_returns.min()),
            'max': float(valid_returns.max()),
            'percentiles': {
                'p5': float(valid_returns.quantile(0.05)),
                'p10': float(valid_returns.quantile(0.10)),
                'p25': float(valid_returns.quantile(0.25)),
                'p50': float(valid_returns.quantile(0.50)),
                'p75': float(valid_returns.quantile(0.75)),
                'p90': float(valid_returns.quantile(0.90)),
                'p95': float(valid_returns.quantile(0.95)),
            },
            'target_distribution': {
                'positive': float((valid_targets == 1).sum() / len(valid_targets)),
                'negative_or_equal': float((valid_targets == 0).sum() / len(valid_targets))
            },
            'absolute_move_stats': {
                'median': float(valid_returns.abs().median()),
                'mean': float(valid_returns.abs().mean()),
                'p25': float(valid_returns.abs().quantile(0.25)),
                'p75': float(valid_returns.abs().quantile(0.75)),
                'p90': float(valid_returns.abs().quantile(0.90)),
                'p95': float(valid_returns.abs().quantile(0.95)),
            }
        }
        
        return stats
    
    @staticmethod
    def print_report(symbol: str, stats: Dict, threshold: float):
        """Print formatted analysis report"""
        
        if not stats:
            logger.warning(f"No statistics for {symbol}")
            return
        
        logger.info("=" * 73)
        logger.info(f"60M CUMULATIVE RETURN ANALYSIS - {symbol}")
        logger.info("=" * 73)
        
        logger.info(f"\nTotal observations:      {stats['total_observations']}")
        
        logger.info(f"\nReturn Statistics:")
        logger.info(f"  Mean:                     {stats['mean']*100:>8.4f}%")
        logger.info(f"  Median:                   {stats['median']*100:>8.4f}%")
        logger.info(f"  Std Dev:                  {stats['std']*100:>8.4f}%")
        logger.info(f"  Min:                      {stats['min']*100:>8.4f}%")
        logger.info(f"  Max:                      {stats['max']*100:>8.4f}%")
        
        logger.info(f"\nReturn Percentiles:")
        for p in ['p5', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95']:
            logger.info(f"  {p:>3}:                   {stats['percentiles'][p]*100:>8.4f}%")
        
        logger.info(f"\nTarget Distribution (threshold={threshold*100:.2f}%):")
        logger.info(f"  Positive (> threshold):      {stats['target_distribution']['positive']*100:>6.1f}%")
        logger.info(f"  Negative (≤ threshold):      {stats['target_distribution']['negative_or_equal']*100:>6.1f}%")
        
        # Validation check
        pos_ratio = stats['target_distribution']['positive']
        if 0.25 <= pos_ratio <= 0.45:
            logger.info("Validation: ✅ REALISTIC (25-45% positive expected)")
        elif 0.20 <= pos_ratio < 0.25:
            logger.info("Validation: ⚠️  BORDERLINE (<25% positive)")
        elif pos_ratio > 0.45:
            logger.info("Validation: ⚠️  BORDERLINE (>45% positive)")
        else:
            logger.info("Validation: ❌ POOR (<20% positive)")
        
        logger.info(f"\nAbsolute Move Distribution:")
        logger.info(f"  Mean:                     {stats['absolute_move_stats']['mean']*100:>8.4f}%")
        logger.info(f"  Median:                   {stats['absolute_move_stats']['median']*100:>8.4f}%")
        logger.info(f"  Std Dev:                  {stats['std']*100:>8.4f}%")
        logger.info(f"  P25:                      {stats['absolute_move_stats']['p25']*100:>8.4f}%")
        logger.info(f"  P75:                      {stats['absolute_move_stats']['p75']*100:>8.4f}%")
        logger.info(f"  P90:                      {stats['absolute_move_stats']['p90']*100:>8.4f}%")
        logger.info(f"  P95:                      {stats['absolute_move_stats']['p95']*100:>8.4f}%")
        
        # Expected move validation
        theoretical_move = 0.3082 * np.sqrt(12)  # BTC volatility from real data
        actual_move = stats['absolute_move_stats']['median']
        ratio = actual_move / theoretical_move
        
        logger.info(f"\nExpected Move Validation (√12 × 0.31%):")
        logger.info(f"  Theoretical:              {theoretical_move*100:>8.4f}%")
        logger.info(f"  Actual Median:            {actual_move*100:>8.4f}%")
        logger.info(f"  Ratio:                       {ratio:>6.2f}x")
        
        if actual_move >= 0.0025:
            logger.info("Validation: ✅ VIABLE (moves large enough for trading)")
        else:
            logger.info("Validation: ⚠️  SMALL (moves below ideal)")
        
        logger.info("=" * 73)


class OutputWriter:
    """Save labeled data and analysis to disk"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_labeled_data(self, df: pd.DataFrame, symbol: str) -> bool:
        """Save labeled data to CSV"""
        try:
            csv_path = self.output_dir / f"{symbol}_5m_with_60m_labels.csv"
            df.to_csv(csv_path, index=False)
            logger.info(f"✅ Saved labeled data: {csv_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Error saving labeled data: {e}")
            return False
    
    def save_analysis(self, stats: Dict, symbol: str) -> bool:
        """Save analysis to JSON"""
        try:
            json_path = self.output_dir / f"{symbol}_60m_label_analysis.json"
            with open(json_path, 'w') as f:
                json.dump(stats, f, indent=2)
            logger.info(f"✅ Saved analysis: {json_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Error saving analysis: {e}")
            return False


class Step1ExtendedRunner:
    """Orchestrate extended Step 1 pipeline"""
    
    def __init__(self, config: Step1ExtendedConfig):
        self.config = config
        self.loader = DataLoader(config.data_dir)
        self.constructor = LabelConstructor(config)
        self.writer = OutputWriter(config.output_dir)
    
    def run(self) -> Dict:
        """Execute full pipeline"""
        
        logger.info("\n" + "=" * 73)
        logger.info("PHASE 9.5 VALIDATION - STEP 1 EXTENDED: 60M LABELS WITH REGIME FILTER")
        logger.info("=" * 73)
        
        # Get symbols
        symbols = self.loader.get_available_symbols()
        if not symbols:
            logger.error("❌ No data files found in " + self.config.data_dir)
            return {}
        
        logger.info(f"\nProcessing symbols: {symbols}")
        
        # Print configuration
        logger.info("\n" + "=" * 73)
        logger.info("STEP 1 CONFIGURATION")
        logger.info("=" * 73)
        logger.info(f"Data Directory:              {self.config.data_dir}")
        logger.info(f"Output Directory:            {self.config.output_dir}")
        logger.info(f"Prediction Horizon:          {self.config.prediction_horizon_candles} candles (60m)")
        logger.info(f"Edge Threshold:              {self.config.edge_threshold_pct*100:.2f}%")
        logger.info(f"Regime Window:               {self.config.regime_window_candles} candles (100m)")
        logger.info(f"Keep Regimes:                {self.config.regimes_to_keep}")
        logger.info(f"Min Data Points Required:    {self.config.min_data_points}")
        logger.info(f"Symbols to Process:          {symbols}")
        logger.info("=" * 73 + "\n")
        
        results = {}
        
        for symbol in symbols:
            logger.info(f"\n{'=' * 73}")
            logger.info(f"Processing: {symbol}")
            logger.info("=" * 73)
            
            # Load data
            df, success = self.loader.load_symbol_data(symbol)
            if not success:
                results[symbol] = {'status': 'FAILED', 'reason': 'Load error'}
                continue
            
            # Build labels
            df_labeled, df_filtered, metadata = self.constructor.build_labels(df)
            
            # Analyze
            stats = TargetAnalyzer.analyze_distribution(
                df_filtered, 
                self.config.edge_threshold_pct
            )
            stats.update(metadata)
            
            TargetAnalyzer.print_report(symbol, stats, self.config.edge_threshold_pct)
            
            # Save outputs
            csv_ok = self.writer.save_labeled_data(df_labeled, symbol)
            json_ok = self.writer.save_analysis(stats, symbol)
            
            results[symbol] = {
                'status': 'SUCCESS' if (csv_ok and json_ok) else 'FAILED',
                'csv_path': str(self.config.output_dir + f"/{symbol}_5m_with_60m_labels.csv"),
                'analysis_path': str(self.config.output_dir + f"/{symbol}_60m_label_analysis.json"),
                'stats': stats
            }
        
        # Summary
        logger.info("\n" + "=" * 73)
        logger.info("STEP 1 EXTENDED COMPLETION SUMMARY")
        logger.info("=" * 73)
        
        successful = sum(1 for r in results.values() if r['status'] == 'SUCCESS')
        failed = sum(1 for r in results.values() if r['status'] == 'FAILED')
        
        logger.info(f"\nResults:")
        logger.info(f"  ✅ Successful: {successful}")
        logger.info(f"  ❌ Failed:     {failed}")
        logger.info(f"  Total:         {len(results)}")
        
        logger.info(f"\nOutputs saved to: {self.config.output_dir}")
        
        logger.info("\nNext Steps:")
        logger.info("  1. Review labeled data files in validation_outputs")
        logger.info("  2. Check analysis JSON files for regime-filtered metrics")
        logger.info("  3. Verify: Positive 25-45%, Median move >= 0.25%")
        logger.info("  4. If PASS: Proceed to Step 2 (Regime Sensitivity)")
        logger.info("  5. If FAIL: Debug and retry with adjusted regimes")
        
        logger.info("=" * 73 + "\n")
        
        # Save summary
        summary_path = self.config.output_dir + "/step1_extended_results.json"
        try:
            with open(summary_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"✅ Results saved to: {summary_path}")
        except Exception as e:
            logger.error(f"❌ Error saving results: {e}")
        
        return results


# Main execution
if __name__ == "__main__":
    config = Step1ExtendedConfig(
        data_dir="data/historical",
        output_dir="validation_outputs",
        prediction_horizon_candles=12,  # 60 minutes
        edge_threshold_pct=0.0020,      # 0.20%
        regime_window_candles=20,       # 100-minute rolling window
        low_vol_percentile=0.33,        # Bottom 33%
        high_vol_percentile=0.67,       # Top 33%
        regimes_to_keep=["normal", "high"]  # Drop low-vol entirely
    )
    
    runner = Step1ExtendedRunner(config)
    results = runner.run()
