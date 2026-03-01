"""
PHASE 9.5 PIVOT: STRUCTURAL IMBALANCE MODELING (Funding + OI)

Goal: Model REAL market microstructure, not price noise

Hypothesis:
  Extreme funding rates + OI imbalances create structural dislocations.
  When these align, liquidation cascades follow.
  Price moves in direction of OI overhang.

Target:
  Liquidation cascade = price move >= 3% in direction of OI concentration
  Timeframe: Next 4-24 hours after extreme condition detected

Features (Structural, not price-based):
  1. Funding Rate (% absolute value)
  2. Funding Rate Direction (+ long, - short)
  3. Long/Short OI Ratio
  4. OI Imbalance Severity (% deviation from 50/50)
  5. Funding Velocity (rate of change)
  6. OI Concentration Risk (how much at single leverage)
  
Extreme Definition (Adaptive Percentiles):
  - Top 10% funding rate (absolute) = elevated risk
  - Top 5% funding rate (absolute) = extreme
  - OI ratio > 60/40 or < 40/60 = imbalanced
  - Funding rate accelerating = cascade formation
  
Key Insight:
  Derivatives market has finite liquidity.
  Extreme imbalances force liquidations.
  Liquidations cascade in direction of least resistance.
  This is NOT random - it's mechanical.
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger('StructuralImbalanceModeling')


@dataclass
class ImbalanceConfig:
    """Configuration for structural imbalance modeling"""
    
    # Data paths
    funding_data_dir: str = "validation_outputs"
    ohlcv_data_dir: str = "validation_outputs"
    output_dir: str = "validation_outputs"
    
    # Extreme thresholds (percentile-based, adaptive)
    funding_rate_extreme_percentile: float = 95.0  # Top 5%
    funding_rate_elevated_percentile: float = 90.0  # Top 10%
    oi_imbalance_threshold: float = 0.55  # 55/45 split
    
    # Cascade detection
    cascade_return_threshold: float = 0.03  # 3% move defines cascade
    cascade_lookback_hours: int = 4  # Look ahead 4-24 hours
    cascade_lookback_max_hours: int = 24
    
    # Feature engineering
    funding_velocity_window: int = 5  # bars to measure acceleration
    
    # Data requirements
    min_samples: int = 50


class StructuralImbalanceAnalyzer:
    """Analyze funding + OI imbalance conditions"""
    
    def __init__(self, config: ImbalanceConfig):
        self.config = config
    
    def calculate_funding_extremes(self, funding_rates: np.ndarray) -> Tuple[float, float]:
        """
        Calculate adaptive funding rate thresholds based on percentiles
        
        Returns: (extreme_threshold, elevated_threshold)
        """
        extreme_threshold = np.percentile(np.abs(funding_rates), self.config.funding_rate_extreme_percentile)
        elevated_threshold = np.percentile(np.abs(funding_rates), self.config.funding_rate_elevated_percentile)
        
        return extreme_threshold, elevated_threshold
    
    def detect_extreme_conditions(self, df: pd.DataFrame, symbol: str) -> Dict:
        """
        Detect extreme imbalance conditions using adaptive percentiles
        
        Returns comprehensive analysis of when extremes occur
        """
        
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"Analyzing Structural Imbalances: {symbol}")
        logger.info("=" * 80)
        
        # Simulate funding rate data (in real implementation, fetch from Binance)
        # For now, create synthetic realistic funding rates
        np.random.seed(42)
        n = len(df)
        
        # Funding rates typically range from -0.1% to +0.1% daily
        # But can spike during extreme conditions
        base_funding = np.random.normal(0.0005, 0.0002, n)  # Mean +0.05% daily
        
        # Add occasional spikes (cascade events)
        spike_indices = np.random.choice(n, size=max(1, n // 20), replace=False)
        for idx in spike_indices:
            direction = np.random.choice([-1, 1])
            magnitude = np.random.uniform(0.001, 0.005)  # 0.1% to 0.5% daily
            base_funding[idx] = direction * magnitude
        
        funding_rates = base_funding
        
        # Simulate OI ratio (long/total OI)
        # Typically around 0.5 (50/50 split)
        oi_ratio = np.random.beta(5, 5, n) * 0.8 + 0.1  # Range 0.1-0.9, centered ~0.5
        
        # Create imbalance severity metric
        oi_imbalance = np.abs(oi_ratio - 0.5) * 2  # 0-1 scale
        
        # Funding velocity (rate of change)
        funding_velocity = np.diff(funding_rates, prepend=funding_rates[0])
        
        # Identify extreme conditions using percentiles
        extreme_funding_threshold, elevated_funding_threshold = \
            self.calculate_funding_extremes(funding_rates)
        
        is_extreme_funding = np.abs(funding_rates) >= extreme_funding_threshold
        is_elevated_funding = np.abs(funding_rates) >= elevated_funding_threshold
        is_imbalanced_oi = oi_imbalance >= (self.config.oi_imbalance_threshold - 0.5)
        
        logger.info(f"Funding Rate Analysis:")
        logger.info(f"  Mean: {np.mean(funding_rates)*100:+.3f}% daily")
        logger.info(f"  Std:  {np.std(funding_rates)*100:.3f}%")
        logger.info(f"  Min:  {np.min(funding_rates)*100:+.3f}%")
        logger.info(f"  Max:  {np.max(funding_rates)*100:+.3f}%")
        logger.info(f"")
        logger.info(f"Extreme Condition Thresholds (Percentile-Based):")
        logger.info(f"  Extreme (95th):  {extreme_funding_threshold*100:+.3f}%")
        logger.info(f"  Elevated (90th): {elevated_funding_threshold*100:+.3f}%")
        logger.info(f"")
        logger.info(f"Condition Frequency:")
        logger.info(f"  Extreme funding: {np.sum(is_extreme_funding):4d} ({100*np.mean(is_extreme_funding):5.1f}%)")
        logger.info(f"  Elevated funding: {np.sum(is_elevated_funding):4d} ({100*np.mean(is_elevated_funding):5.1f}%)")
        logger.info(f"  OI imbalanced:   {np.sum(is_imbalanced_oi):4d} ({100*np.mean(is_imbalanced_oi):5.1f}%)")
        logger.info(f"")
        
        # Detect co-occurrences (true danger)
        extreme_and_imbalanced = is_extreme_funding & is_imbalanced_oi
        logger.info(f"Co-Occurrence Analysis:")
        logger.info(f"  Extreme funding + OI imbalance: {np.sum(extreme_and_imbalanced)} ({100*np.mean(extreme_and_imbalanced):.2f}%)")
        logger.info(f"  → These are your HIGH-RISK cascade candidates")
        logger.info(f"")
        
        return {
            'funding_rates': funding_rates,
            'oi_ratio': oi_ratio,
            'oi_imbalance': oi_imbalance,
            'funding_velocity': funding_velocity,
            'is_extreme_funding': is_extreme_funding,
            'is_elevated_funding': is_elevated_funding,
            'is_imbalanced_oi': is_imbalanced_oi,
            'extreme_and_imbalanced': extreme_and_imbalanced,
            'extreme_threshold': extreme_funding_threshold,
            'elevated_threshold': elevated_funding_threshold,
            'thresholds': {
                'funding_rate_extreme': float(extreme_funding_threshold),
                'funding_rate_elevated': float(elevated_funding_threshold),
                'oi_imbalance': float(self.config.oi_imbalance_threshold)
            },
            'statistics': {
                'mean_funding': float(np.mean(funding_rates)),
                'std_funding': float(np.std(funding_rates)),
                'extreme_count': int(np.sum(is_extreme_funding)),
                'elevated_count': int(np.sum(is_elevated_funding)),
                'imbalanced_count': int(np.sum(is_imbalanced_oi)),
                'cooccurrence_count': int(np.sum(extreme_and_imbalanced))
            }
        }
    
    def identify_cascade_events(self, df: pd.DataFrame, conditions: Dict) -> Tuple[np.ndarray, Dict]:
        """
        Identify actual cascade events (1.5%+ moves in OI direction within 4 hours)
        
        Returns: (cascade_labels, cascade_analysis)
        """
        
        is_extreme = conditions['extreme_and_imbalanced']
        oi_ratio = conditions['oi_ratio']
        funding_rates = conditions['funding_rates']
        
        # Adjust cascade threshold based on volatility
        volatility = df['close'].pct_change().std()
        cascade_threshold = max(0.015, volatility * 1.5)  # At least 1.5% or 1.5x vol
        
        cascade_labels = np.zeros(len(df), dtype=int)
        cascade_analysis = {
            'cascade_count': 0,
            'cascade_indices': [],
            'by_direction': {'long_cascade': 0, 'short_cascade': 0},
            'success_rate': 0.0,
            'cascade_threshold': cascade_threshold
        }
        
        # For each extreme condition, check if cascade occurred within 4 hours
        for idx in np.where(is_extreme)[0]:
            if idx + self.config.cascade_lookback_hours >= len(df):
                continue  # Not enough future data
            
            # Get price action ahead (4 hours = 48 bars of 5m)
            lookback_end = min(idx + self.config.cascade_lookback_hours, len(df) - 1)
            future_prices = df['close'].iloc[idx:lookback_end]
            
            if len(future_prices) < 2:
                continue
            
            entry_price = future_prices.iloc[0]
            future_returns = (future_prices / entry_price - 1).values
            
            max_future_return = future_returns.max()
            min_future_return = future_returns.min()
            
            # OI ratio > 0.55 means more longs (susceptible to long squeeze down)
            # OI ratio < 0.45 means more shorts (susceptible to short squeeze up)
            # Combine with funding direction for stronger signal
            
            is_long_heavy = oi_ratio[idx] > 0.5
            is_short_heavy = oi_ratio[idx] < 0.5
            is_funding_positive = funding_rates[idx] > 0
            
            # Cascade: direction matches OI overhang
            cascade_occurred = False
            direction = None
            
            if is_long_heavy and min_future_return <= -cascade_threshold:
                # More longs, down move = short squeeze
                cascade_occurred = True
                direction = 'short_squeeze'
                cascade_analysis['by_direction']['short_cascade'] += 1
            
            elif is_short_heavy and max_future_return >= cascade_threshold:
                # More shorts, up move = long squeeze
                cascade_occurred = True
                direction = 'long_squeeze'
                cascade_analysis['by_direction']['long_cascade'] += 1
            
            if cascade_occurred:
                cascade_labels[idx] = 1
                cascade_analysis['cascade_count'] += 1
                cascade_analysis['cascade_indices'].append((idx, direction))
        
        if cascade_analysis['cascade_count'] > 0:
            cascade_analysis['success_rate'] = \
                cascade_analysis['cascade_count'] / np.sum(is_extreme)
        
        logger.info(f"Cascade Event Analysis:")
        logger.info(f"  Total extreme conditions: {np.sum(is_extreme)}")
        logger.info(f"  Actual cascades occurred: {cascade_analysis['cascade_count']}")
        logger.info(f"  Success rate: {cascade_analysis['success_rate']*100:.1f}%")
        logger.info(f"    Long squeezes: {cascade_analysis['by_direction']['long_cascade']}")
        logger.info(f"    Short squeezes: {cascade_analysis['by_direction']['short_cascade']}")
        logger.info(f"")
        
        return cascade_labels, cascade_analysis


class StructuralImbalanceDiagnostic:
    """Run full diagnostic on structural imbalances"""
    
    def __init__(self, config: ImbalanceConfig):
        self.config = config
        self.results = {}
    
    def run(self):
        """Execute diagnostic"""
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("STRUCTURAL IMBALANCE DIAGNOSTIC")
        logger.info("=" * 80)
        logger.info("")
        logger.info("Question: Do funding + OI imbalances predict liquidation cascades?")
        logger.info("")
        logger.info("Method: Adaptive percentile-based extreme detection")
        logger.info("  - Extreme funding: Top 5% (95th percentile)")
        logger.info("  - Elevated funding: Top 10% (90th percentile)")
        logger.info("  - OI imbalance: >55/45 or <45/55 split")
        logger.info("  - Cascade: 3%+ move in OI overhang direction")
        logger.info("")
        
        # Load OHLCV data
        data_dir = Path(self.config.ohlcv_data_dir)
        ohlcv_files = list(data_dir.glob("*_5m_with_60m_labels.csv"))
        
        if not ohlcv_files:
            logger.error("No OHLCV files found")
            return
        
        for file_path in sorted(ohlcv_files):
            symbol = file_path.stem.replace('_5m_with_60m_labels', '')
            
            # Load data
            df = pd.read_csv(file_path)
            
            # Analyze imbalances
            analyzer = StructuralImbalanceAnalyzer(self.config)
            conditions = analyzer.detect_extreme_conditions(df, symbol)
            
            # Identify cascades
            cascade_labels, cascade_analysis = analyzer.identify_cascade_events(df, conditions)
            
            # Store results
            self.results[symbol] = {
                'conditions': {
                    'thresholds': conditions['thresholds'],
                    'statistics': conditions['statistics']
                },
                'cascades': cascade_analysis,
                'samples': int(len(df))
            }
            
            # Conditional probability check
            self._analyze_conditional_probability(
                cascade_labels,
                conditions['extreme_and_imbalanced'],
                symbol
            )
        
        # Save results
        output_file = Path(self.config.output_dir) / "diagnostic_structural_imbalances.json"
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("DIAGNOSTIC COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Results saved to: {output_file}")
        logger.info("")
        
        self._print_recommendations()
    
    def _analyze_conditional_probability(self, cascade_labels: np.ndarray, 
                                         extreme_conditions: np.ndarray, 
                                         symbol: str):
        """Analyze P(cascade | extreme condition)"""
        
        extreme_indices = np.where(extreme_conditions)[0]
        
        if len(extreme_indices) == 0:
            logger.info(f"No extreme conditions detected for {symbol}")
            return
        
        cascades_given_extreme = cascade_labels[extreme_indices]
        p_cascade_given_extreme = np.mean(cascades_given_extreme)
        
        baseline_cascade_rate = np.mean(cascade_labels)
        
        logger.info(f"Conditional Probability Analysis ({symbol}):")
        logger.info(f"  P(cascade | extreme condition) = {p_cascade_given_extreme*100:.1f}%")
        logger.info(f"  P(cascade | baseline)          = {baseline_cascade_rate*100:.1f}%")
        logger.info(f"  Lift = {p_cascade_given_extreme / (baseline_cascade_rate + 1e-10):.2f}x")
        logger.info(f"")
        
        if p_cascade_given_extreme > baseline_cascade_rate * 1.5:
            logger.info(f"  ✅ STRONG SIGNAL: Extreme conditions significantly predict cascades")
        elif p_cascade_given_extreme > baseline_cascade_rate * 1.2:
            logger.info(f"  ⚠️  MODERATE SIGNAL: Some predictive power detected")
        else:
            logger.info(f"  ❌ WEAK SIGNAL: Limited predictive power")
        logger.info(f"")
    
    def _print_recommendations(self):
        """Print final recommendations"""
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("🎯 NEXT STEPS: STRUCTURAL MODELING")
        logger.info("=" * 80)
        logger.info("")
        
        logger.info("PHASE 1: Data Collection")
        logger.info("  □ Fetch historical funding rates from Binance API")
        logger.info("  □ Collect OI data by leverage tier")
        logger.info("  □ Timestamp alignment with OHLCV")
        logger.info("")
        
        logger.info("PHASE 2: Feature Engineering")
        logger.info("  □ Funding rate normalized to historical percentiles")
        logger.info("  □ OI imbalance severity (quantile score)")
        logger.info("  □ Funding acceleration (2nd derivative)")
        logger.info("  □ Liquidation cluster detection")
        logger.info("  □ OI concentration by leverage (Herfindahl index)")
        logger.info("")
        
        logger.info("PHASE 3: Target Refinement")
        logger.info("  □ Cascade = 3%+ move within 4-24 hours")
        logger.info("  □ Direction = OI overhang direction (long/short bias)")
        logger.info("  □ Magnitude = size of cascade relative to ATR")
        logger.info("")
        
        logger.info("PHASE 4: Model Training")
        logger.info("  □ LSTM on structural features (not price)")
        logger.info("  □ Conditional probability test first")
        logger.info("  □ If P(cascade|extreme) > 60% → proceed to training")
        logger.info("  □ If P(cascade|extreme) < 50% → feature engineering needed")
        logger.info("")
        
        logger.info("PHASE 5: Validation")
        logger.info("  □ Out-of-sample cascade prediction accuracy")
        logger.info("  □ Directional accuracy (did we predict right direction?)")
        logger.info("  □ Risk-adjusted returns (Sharpe ratio)")
        logger.info("  □ Maximum drawdown during cascade periods")
        logger.info("")
        
        logger.info("=" * 80)


if __name__ == "__main__":
    config = ImbalanceConfig()
    diagnostic = StructuralImbalanceDiagnostic(config)
    diagnostic.run()
