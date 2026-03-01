#!/usr/bin/env python3
"""
PHASE 9.5 VALIDATION STEP 1: Construct 30m Cumulative Return Labels
====================================================================

Purpose:
  Create ground truth targets for 30m cumulative return prediction.
  This allows us to train and test if the model can learn the pattern.

Input:
  - 5m OHLCV data (from data/historical/*.csv)
  
Output:
  - CSV files with 30m cumulative return labels
  - Analysis report of target distribution
  - Validation checks

Execution Time: 5-10 minutes

Decision Gate:
  ✅ PASS if:
    - Can construct valid labels (no errors)
    - Target distribution is ~30-40% positive
    - Forward move distribution is >= 0.37% median
  ❌ FAIL if:
    - Cannot construct labels (data issues)
    - Target distribution < 20% or > 50% positive
    - Forward moves are too small (< 0.30% median)

"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
import json
from datetime import datetime

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Step1_ConstructLabels")

# ============================================================================
# CONFIG
# ============================================================================

@dataclass
class Step1Config:
    """Configuration for Step 1 label construction"""
    
    # Data paths
    data_dir: str = "data/historical"
    output_dir: str = "validation_outputs"
    
    # Label construction parameters
    prediction_horizon_candles: int = 6  # 6 × 5m = 30m
    edge_threshold_pct: float = 0.0020   # 0.20% minimum return
    min_data_points: int = 100           # Need at least this many for valid labels
    
    # Analysis parameters
    forward_move_theoretical_pct: float = 0.0037  # √6 × 0.15%
    
    # Symbols to process
    symbols: List[str] = None  # If None, auto-detect from data directory
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = self._auto_detect_symbols()
    
    def _auto_detect_symbols(self) -> List[str]:
        """Detect available symbols from data directory"""
        data_path = Path(self.data_dir)
        if not data_path.exists():
            logger.warning(f"Data directory {self.data_dir} not found")
            return []
        
        symbols = set()
        for csv_file in data_path.glob("*_5m.csv"):
            symbol = csv_file.stem.replace("_5m", "")
            symbols.add(symbol)
        
        return sorted(list(symbols))


# ============================================================================
# STEP 1A: LOAD DATA
# ============================================================================

class DataLoader:
    """Load and validate 5m OHLCV data"""
    
    def __init__(self, config: Step1Config):
        self.config = config
        self.data_dir = Path(config.data_dir)
    
    def load_symbol_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Load 5m OHLCV for a symbol.
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
            or None if loading fails
        """
        filepath = self.data_dir / f"{symbol}_5m.csv"
        
        if not filepath.exists():
            logger.warning(f"Data file not found: {filepath}")
            return None
        
        try:
            df = pd.read_csv(filepath)
            
            # Standardize columns
            df.columns = [c.lower() for c in df.columns]
            
            # Validate required columns
            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                logger.error(f"Missing columns in {symbol}. Required: {required_cols}")
                return None
            
            # Sort by timestamp and reset index
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Convert close to numeric (in case of string)
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            if df['close'].isna().any():
                logger.warning(f"Found NaN values in close price for {symbol}")
                df = df.dropna(subset=['close'])
            
            logger.info(f"✅ Loaded {len(df)} rows for {symbol}")
            return df
        
        except Exception as e:
            logger.error(f"Error loading {symbol}: {e}")
            return None


# ============================================================================
# STEP 1B: CONSTRUCT 30M CUMULATIVE RETURN LABELS
# ============================================================================

class LabelConstructor:
    """Construct binary labels for 30m cumulative return"""
    
    def __init__(self, config: Step1Config):
        self.config = config
    
    def build_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build 30m cumulative return labels.
        
        Process:
          1. For each timestep i
          2. Look forward prediction_horizon_candles (6 candles = 30m)
          3. Calculate cumulative return: (close[i+6] - close[i]) / close[i]
          4. Label as 1 if return > edge_threshold_pct, else 0
          5. Mark label as valid if enough forward data exists
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added columns:
              - forward_return_30m: Actual 30m cumulative return
              - target_30m: Binary label (1 = positive return)
              - forward_return_valid: Whether label is valid
        """
        
        df_copy = df.copy()
        n = len(df_copy)
        horizon = self.config.prediction_horizon_candles
        threshold = self.config.edge_threshold_pct
        
        logger.info(f"Constructing 30m labels (horizon={horizon} candles, threshold={threshold:.4%})...")
        
        # Initialize new columns
        df_copy['forward_return_30m'] = np.nan
        df_copy['target_30m'] = np.nan
        df_copy['forward_return_valid'] = False
        
        # For each point in history
        valid_count = 0
        for i in range(n - horizon):
            entry_price = df_copy['close'].iloc[i]
            exit_price = df_copy['close'].iloc[i + horizon]
            
            # Calculate cumulative return
            cumulative_return = (exit_price - entry_price) / entry_price
            
            # Store results
            df_copy.loc[i, 'forward_return_30m'] = cumulative_return
            df_copy.loc[i, 'target_30m'] = 1 if cumulative_return > threshold else 0
            df_copy.loc[i, 'forward_return_valid'] = True
            valid_count += 1
        
        logger.info(f"✅ Created {valid_count} valid labels")
        
        # Remove rows without valid labels
        df_valid = df_copy[df_copy['forward_return_valid'] == True].copy().reset_index(drop=True)
        
        if len(df_valid) < self.config.min_data_points:
            logger.error(f"Only {len(df_valid)} valid labels, need {self.config.min_data_points}")
            return None
        
        return df_valid


# ============================================================================
# STEP 1C: ANALYZE TARGET DISTRIBUTION
# ============================================================================

class TargetAnalyzer:
    """Analyze distribution of constructed labels"""
    
    def __init__(self, config: Step1Config):
        self.config = config
    
    def analyze_distribution(self, df: pd.DataFrame) -> Dict:
        """
        Analyze the distribution of 30m forward returns and targets.
        
        Returns:
            Dictionary with analysis results
        """
        
        forward_returns = df['forward_return_30m'].dropna()
        
        if len(forward_returns) == 0:
            logger.error("No valid forward returns found")
            return None
        
        # Calculate statistics
        stats = {
            'total_observations': len(forward_returns),
            'mean': forward_returns.mean(),
            'median': forward_returns.median(),
            'std': forward_returns.std(),
            'min': forward_returns.min(),
            'max': forward_returns.max(),
            'percentiles': {
                'p05': np.percentile(forward_returns, 5),
                'p10': np.percentile(forward_returns, 10),
                'p25': np.percentile(forward_returns, 25),
                'p50': np.percentile(forward_returns, 50),
                'p75': np.percentile(forward_returns, 75),
                'p90': np.percentile(forward_returns, 90),
                'p95': np.percentile(forward_returns, 95),
            }
        }
        
        # Target distribution
        target_dist = df['target_30m'].value_counts(normalize=True).sort_index()
        stats['target_distribution'] = {
            'negative_or_equal': float(target_dist.get(0, 0)),
            'positive': float(target_dist.get(1, 0)),
        }
        
        # Absolute moves
        absolute_moves = forward_returns.abs()
        stats['absolute_move_stats'] = {
            'mean': float(absolute_moves.mean()),
            'median': float(absolute_moves.median()),
            'std': float(absolute_moves.std()),
            'p25': float(np.percentile(absolute_moves, 25)),
            'p50': float(np.percentile(absolute_moves, 50)),
            'p75': float(np.percentile(absolute_moves, 75)),
            'p90': float(np.percentile(absolute_moves, 90)),
            'p95': float(np.percentile(absolute_moves, 95)),
        }
        
        return stats
    
    def print_report(self, symbol: str, stats: Dict):
        """Print formatted analysis report"""
        
        if stats is None:
            return
        
        print("\n" + "=" * 80)
        print(f"30M CUMULATIVE RETURN ANALYSIS - {symbol}")
        print("=" * 80)
        
        fr = stats
        print(f"\nTotal observations:      {fr['total_observations']}")
        
        print(f"\nReturn Statistics:")
        print(f"  Mean:                  {fr['mean']:>10.4%}")
        print(f"  Median:                {fr['median']:>10.4%}")
        print(f"  Std Dev:               {fr['std']:>10.4%}")
        print(f"  Min:                   {fr['min']:>10.4%}")
        print(f"  Max:                   {fr['max']:>10.4%}")
        
        print(f"\nReturn Percentiles:")
        p = fr['percentiles']
        for pct in [5, 10, 25, 50, 75, 90, 95]:
            key = f'p{pct:02d}'
            val = p[key]
            print(f"  {pct:2d}th:                 {val:>10.4%}")
        
        td = fr['target_distribution']
        print(f"\nTarget Distribution (threshold={self.config.edge_threshold_pct:.4%}):")
        print(f"  Positive (> threshold): {td['positive']:>10.1%}")
        print(f"  Negative (≤ threshold): {td['negative_or_equal']:>10.1%}")
        
        # Validation
        pos_ratio = td['positive']
        if 0.25 < pos_ratio < 0.45:
            status = "✅ REALISTIC"
            note = "(30-40% positive expected)"
        elif pos_ratio < 0.25:
            status = "⚠️  WARNING"
            note = "(<25% positive - check data quality)"
        elif pos_ratio > 0.45:
            status = "⚠️  WARNING"
            note = "(>45% positive - threshold may be too low)"
        else:
            status = "❓ CHECK"
            note = ""
        
        print(f"\nValidation: {status} {note}")
        
        # Absolute move analysis
        am = fr['absolute_move_stats']
        theoretical = self.config.forward_move_theoretical_pct
        actual_median = am['median']
        
        print(f"\nAbsolute Move Distribution:")
        print(f"  Mean:                  {am['mean']:>10.4%}")
        print(f"  Median:                {am['median']:>10.4%}")
        print(f"  Std Dev:               {am['std']:>10.4%}")
        print(f"  P25:                   {am['p25']:>10.4%}")
        print(f"  P50:                   {am['p50']:>10.4%}")
        print(f"  P75:                   {am['p75']:>10.4%}")
        print(f"  P90:                   {am['p90']:>10.4%}")
        print(f"  P95:                   {am['p95']:>10.4%}")
        
        print(f"\nExpected Move Validation (√6 × 0.15%):")
        print(f"  Theoretical:           {theoretical:>10.4%}")
        print(f"  Actual Median:         {actual_median:>10.4%}")
        print(f"  Ratio:                 {actual_median/theoretical:>10.2f}x")
        
        if actual_median >= 0.0030:
            status_move = "✅ VIABLE"
            note_move = "(supports 30m horizon)"
        else:
            status_move = "⚠️  WARNING"
            note_move = "(moves smaller than expected)"
        
        print(f"\nValidation: {status_move} {note_move}")
        
        print("=" * 80 + "\n")


# ============================================================================
# STEP 1D: SAVE OUTPUT
# ============================================================================

class OutputWriter:
    """Save results to disk"""
    
    def __init__(self, config: Step1Config):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_labeled_data(self, symbol: str, df: pd.DataFrame) -> str:
        """Save labeled DataFrame to CSV"""
        
        output_path = self.output_dir / f"{symbol}_5m_with_30m_labels.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"✅ Saved labeled data: {output_path}")
        
        return str(output_path)
    
    def save_analysis(self, symbol: str, stats: Dict) -> str:
        """Save analysis results to JSON"""
        
        output_path = self.output_dir / f"{symbol}_30m_label_analysis.json"
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        logger.info(f"✅ Saved analysis: {output_path}")
        
        return str(output_path)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

class Step1Runner:
    """Orchestrate Step 1 execution"""
    
    def __init__(self, config: Step1Config):
        self.config = config
        self.loader = DataLoader(config)
        self.constructor = LabelConstructor(config)
        self.analyzer = TargetAnalyzer(config)
        self.writer = OutputWriter(config)
        self.results = {}
    
    def run(self):
        """Execute Step 1 for all symbols"""
        
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 9.5 VALIDATION - STEP 1: CONSTRUCT 30M LABELS")
        logger.info("=" * 80 + "\n")
        
        # Detect symbols if needed
        if not self.config.symbols:
            self.config.symbols = self.config._auto_detect_symbols()
            if not self.config.symbols:
                logger.error("No symbols found in data directory")
                return False
        
        logger.info(f"Processing symbols: {self.config.symbols}\n")
        
        success_count = 0
        fail_count = 0
        
        for symbol in self.config.symbols:
            logger.info(f"\n{'='*80}")
            logger.info(f"Processing: {symbol}")
            logger.info("=" * 80)
            
            try:
                # Load data
                df = self.loader.load_symbol_data(symbol)
                if df is None:
                    fail_count += 1
                    continue
                
                # Construct labels
                df_labeled = self.constructor.build_labels(df)
                if df_labeled is None:
                    fail_count += 1
                    continue
                
                # Analyze
                stats = self.analyzer.analyze_distribution(df_labeled)
                if stats is None:
                    fail_count += 1
                    continue
                
                # Print report
                self.analyzer.print_report(symbol, stats)
                
                # Save outputs
                csv_path = self.writer.save_labeled_data(symbol, df_labeled)
                json_path = self.writer.save_analysis(symbol, stats)
                
                # Record results
                self.results[symbol] = {
                    'status': 'SUCCESS',
                    'csv_path': csv_path,
                    'analysis_path': json_path,
                    'stats': stats,
                }
                
                success_count += 1
            
            except Exception as e:
                logger.exception(f"Error processing {symbol}: {e}")
                self.results[symbol] = {
                    'status': 'FAILED',
                    'error': str(e),
                }
                fail_count += 1
        
        # Final summary
        self._print_summary(success_count, fail_count)
        
        return success_count > 0
    
    def _print_summary(self, success_count: int, fail_count: int):
        """Print final summary"""
        
        print("\n" + "=" * 80)
        print("STEP 1 COMPLETION SUMMARY")
        print("=" * 80)
        
        print(f"\nResults:")
        print(f"  ✅ Successful: {success_count}")
        print(f"  ❌ Failed:     {fail_count}")
        print(f"  Total:         {success_count + fail_count}")
        
        print(f"\nOutputs saved to: {self.config.output_dir}")
        
        print(f"\nNext Steps:")
        if success_count > 0:
            print(f"  1. Review labeled data files in {self.config.output_dir}")
            print(f"  2. Check analysis JSON files for target distribution")
            print(f"  3. Verify:")
            print(f"     - Target distribution: 25-45% positive")
            print(f"     - Median absolute move: >= 0.30%")
            print(f"  4. If PASS: Proceed to Step 2 (Expected Move Validation)")
            print(f"  5. If FAIL: Debug data or reconsider horizon")
        else:
            print(f"  ❌ All symbols failed. Check data files and try again.")
        
        print("\n" + "=" * 80 + "\n")
        
        # Save results
        results_path = Path(self.config.output_dir) / "step1_results.json"
        with open(results_path, 'w') as f:
            # Convert non-serializable objects
            results_clean = {}
            for symbol, result in self.results.items():
                result_clean = result.copy()
                if 'stats' in result_clean and result_clean['stats'] is not None:
                    # Already dict, should be JSON-serializable
                    pass
                results_clean[symbol] = result_clean
            
            json.dump(results_clean, f, indent=2, default=str)
        
        logger.info(f"✅ Results saved to: {results_path}")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    
    # Create config
    config = Step1Config()
    
    # Print config
    print("\n" + "=" * 80)
    print("STEP 1 CONFIGURATION")
    print("=" * 80)
    print(f"\nData Directory:              {config.data_dir}")
    print(f"Output Directory:            {config.output_dir}")
    print(f"Prediction Horizon:          {config.prediction_horizon_candles} candles (30m)")
    print(f"Edge Threshold:              {config.edge_threshold_pct:.4%}")
    print(f"Min Data Points Required:    {config.min_data_points}")
    print(f"Symbols to Process:          {config.symbols}")
    print("=" * 80 + "\n")
    
    # Run Step 1
    runner = Step1Runner(config)
    success = runner.run()
    
    # Exit with appropriate code
    exit(0 if success else 1)
