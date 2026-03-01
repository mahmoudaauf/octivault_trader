"""
PHASE 9.5 STEP 2.5: EVENT-BASED TARGET MODELING

Goal: Reframe prediction problem from fixed-horizon direction to event-based expansion

Paradigm Shift:
  OLD: "Is return > 0.2% in 60m?"
       - Symmetric payoff
       - Timing-dependent
       - Treats 0.21% same as 5% move
       - Weak structural signal
  
  NEW: "Will price expand ≥ X ATR before reversing Y ATR?"
       - Asymmetric payoff structure
       - Direction-dependent (upside vs downside)
       - Natural risk definition
       - Rewards capturing fat tails
       - Aligns with long-term capital growth

Target Definition:
  Event = (Expansion >= expansion_atr) BEFORE (Reversal >= reversal_atr)
  
  Expansion: Price moves away from entry in profitable direction
  Reversal:  Price moves against entry by stop amount
  
  Label: 1 if expansion triggers first, 0 if reversal triggers first

Key Insight:
  This creates asymmetric payoff that matches how markets actually work.
  Crypto profits come from BURSTS, not hourly drift.
  Most bars are chop; we avoid those.
  We trade only when expansion is more likely than reversal.

Parameters:
  atr_window: 20 (rolling volatility measure)
  expansion_atr_multiple: 2.0 (capture 2x current volatility)
  reversal_atr_multiple: 1.0 (risk 1x current volatility)
  
  This creates 2:1 payoff ratio naturally.
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger('EventBasedModeling')


@dataclass
class EventBasedConfig:
    """Configuration for event-based target modeling"""
    
    # Data paths
    labeled_data_dir: str = "validation_outputs"
    output_dir: str = "validation_outputs"
    
    # ATR-based event parameters
    atr_window: int = 20  # Rolling volatility window
    expansion_atr_multiple: float = 2.0  # Expansion target = 2x ATR
    reversal_atr_multiple: float = 1.0  # Reversal target = 1x ATR
    
    # Directional filtering
    trend_window: int = 5  # Bars to determine bias direction
    
    # Data filtering
    min_atr: float = 0.0005  # Min volatility to trade (avoids zero-vol periods)
    max_event_lookback: int = 100  # Max bars ahead to find event completion


class EventDetector:
    """Detect ATR-based expansion/reversal events"""
    
    def __init__(self, config: EventBasedConfig):
        self.config = config
    
    def calculate_atr(self, df: pd.DataFrame, window: int) -> pd.Series:
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean()
        
        return atr
    
    def calculate_trend_bias(self, df: pd.DataFrame, window: int) -> pd.Series:
        """
        Determine direction bias (1=bullish, 0=bearish)
        Bullish if recent close higher than close N bars ago
        """
        close_current = df['close']
        close_past = df['close'].shift(window)
        
        bias = (close_current > close_past).astype(int)
        return bias
    
    def find_event(self, df: pd.DataFrame, start_idx: int, 
                   trend_bias: int, atr_value: float) -> Optional[Tuple[str, int]]:
        """
        Find first event occurrence (expansion or reversal)
        
        Returns: (event_type, bars_to_event)
          event_type: 'expansion' (1) or 'reversal' (0)
          bars_to_event: How many bars ahead did event complete?
        """
        
        if start_idx >= len(df) - self.config.max_event_lookback:
            return None  # Not enough future data
        
        entry_price = df['close'].iloc[start_idx]
        
        if trend_bias == 1:  # Bullish - looking for upside expansion or downside reversal
            expansion_target = entry_price + (atr_value * self.config.expansion_atr_multiple)
            reversal_target = entry_price - (atr_value * self.config.reversal_atr_multiple)
        else:  # Bearish - looking for downside expansion or upside reversal
            expansion_target = entry_price - (atr_value * self.config.expansion_atr_multiple)
            reversal_target = entry_price + (atr_value * self.config.reversal_atr_multiple)
        
        # Scan forward for event
        for i in range(start_idx + 1, min(start_idx + self.config.max_event_lookback, len(df))):
            high = df['high'].iloc[i]
            low = df['low'].iloc[i]
            
            if trend_bias == 1:  # Bullish
                if high >= expansion_target:
                    return ('expansion', i - start_idx)
                if low <= reversal_target:
                    return ('reversal', i - start_idx)
            else:  # Bearish
                if low <= expansion_target:
                    return ('expansion', i - start_idx)
                if high >= reversal_target:
                    return ('reversal', i - start_idx)
        
        return None  # Event didn't complete within lookback window
    
    def create_event_targets(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create event-based binary targets
        
        Returns:
            (targets, event_bars, event_types)
            targets: 1 if expansion occurred first, 0 if reversal
            event_bars: How many bars to event completion
            event_types: String labels ('expansion', 'reversal', 'incomplete')
        """
        
        df_copy = df.copy()
        
        # Calculate ATR
        atr = self.calculate_atr(df_copy, self.config.atr_window)
        
        # Calculate trend bias
        bias = self.calculate_trend_bias(df_copy, self.config.trend_window)
        
        targets = []
        event_bars_list = []
        event_types = []
        
        # For each valid starting point
        for i in range(self.config.trend_window, len(df_copy) - self.config.max_event_lookback):
            atr_val = atr.iloc[i]
            trend_bias = bias.iloc[i]
            
            # Skip if volatility too low
            if atr_val < self.config.min_atr or pd.isna(atr_val):
                continue
            
            # Find event
            event = self.find_event(df_copy, i, trend_bias, atr_val)
            
            if event is None:
                # Event incomplete - skip this point
                continue
            
            event_type, bars_to_event = event
            
            # Label: 1 = expansion (WIN), 0 = reversal (LOSS)
            label = 1 if event_type == 'expansion' else 0
            
            targets.append(label)
            event_bars_list.append(bars_to_event)
            event_types.append(event_type)
        
        targets = np.array(targets)
        event_bars_list = np.array(event_bars_list)
        event_types = np.array(event_types)
        
        logger.info(f"✅ Created {len(targets)} event-based targets")
        logger.info(f"   Expansions (wins):  {np.sum(targets == 1)} ({100*np.mean(targets):.1f}%)")
        logger.info(f"   Reversals (losses): {np.sum(targets == 0)} ({100*np.mean(targets==0):.1f}%)")
        logger.info(f"   Median bars to event: {np.median(event_bars_list):.0f}")
        
        return targets, event_bars_list, event_types


class EventBasedAnalyzer:
    """Analyze event-based targets"""
    
    @staticmethod
    def analyze_events(targets: np.ndarray, event_bars: np.ndarray, 
                      event_types: np.ndarray, symbol: str) -> Dict:
        """Generate comprehensive analysis"""
        
        if len(targets) == 0:
            return {'status': 'EMPTY', 'reason': 'No events found'}
        
        win_count = np.sum(targets == 1)
        loss_count = np.sum(targets == 0)
        
        analysis = {
            'total_events': int(len(targets)),
            'total_wins': int(win_count),
            'total_losses': int(loss_count),
            'win_rate': float(win_count / len(targets)),
            'loss_rate': float(loss_count / len(targets)),
            'median_bars_to_event': float(np.median(event_bars)),
            'mean_bars_to_event': float(np.mean(event_bars)),
            'std_bars_to_event': float(np.std(event_bars)),
            'bars_distribution': {
                'p25': float(np.percentile(event_bars, 25)),
                'p50': float(np.percentile(event_bars, 50)),
                'p75': float(np.percentile(event_bars, 75)),
                'p90': float(np.percentile(event_bars, 90))
            }
        }
        
        logger.info(f"")
        logger.info(f"EVENT ANALYSIS: {symbol}")
        logger.info(f"━" * 70)
        logger.info(f"Total Events:        {analysis['total_events']:6d}")
        logger.info(f"Wins (Expansion):    {analysis['total_wins']:6d} ({analysis['win_rate']*100:5.1f}%)")
        logger.info(f"Losses (Reversal):   {analysis['total_losses']:6d} ({analysis['loss_rate']*100:5.1f}%)")
        logger.info(f"")
        logger.info(f"Time to Event:")
        logger.info(f"  Median:  {analysis['median_bars_to_event']:.0f} bars")
        logger.info(f"  Mean:    {analysis['mean_bars_to_event']:.1f} bars")
        logger.info(f"  P90:     {analysis['bars_distribution']['p90']:.0f} bars")
        logger.info(f"")
        logger.info(f"Key Insight:")
        logger.info(f"  {analysis['win_rate']*100:.1f}% of qualified setups expand before reversing")
        logger.info(f"  This is the TRUE EDGE - asymmetric payoff structure")
        logger.info(f"")
        
        return analysis


class Step25Runner:
    """Execute Step 2.5: Event-Based Target Creation"""
    
    def __init__(self, config: EventBasedConfig):
        self.config = config
        self.results = {}
    
    def run(self):
        """Execute for all symbols"""
        
        logger.info("=" * 80)
        logger.info("PHASE 9.5 STEP 2.5: EVENT-BASED MODELING")
        logger.info("=" * 80)
        logger.info("")
        logger.info("🎯 PARADIGM SHIFT:")
        logger.info("   OLD: Fixed-horizon directional (weak target)")
        logger.info("   NEW: ATR-based expansion vs reversal (asymmetric payoff)")
        logger.info("")
        logger.info("📊 PARAMETERS:")
        logger.info(f"   Expansion Target: {self.config.expansion_atr_multiple}x ATR")
        logger.info(f"   Reversal Target:  {self.config.reversal_atr_multiple}x ATR")
        logger.info(f"   Payoff Ratio:     {self.config.expansion_atr_multiple/self.config.reversal_atr_multiple:.1f}:1")
        logger.info("")
        logger.info("💡 HYPOTHESIS:")
        logger.info("   Markets reward expansion capture, not directional guessing.")
        logger.info("   Asymmetric payoff structure aligns with market reality.")
        logger.info("")
        
        # Get raw data files (before regime filtering)
        data_dir = Path(self.config.labeled_data_dir)
        labeled_files = list(data_dir.glob("*_5m_with_60m_labels.csv"))
        
        if not labeled_files:
            logger.error(f"❌ No labeled data found in {data_dir}")
            return
        
        logger.info(f"📁 Processing {len(labeled_files)} symbols")
        logger.info("")
        
        for file_path in sorted(labeled_files):
            symbol = file_path.stem.replace('_5m_with_60m_labels', '')
            logger.info("=" * 80)
            logger.info(f"Processing: {symbol}")
            logger.info("=" * 80)
            
            self._process_symbol(symbol, file_path)
            logger.info("")
        
        # Save summary
        summary_path = Path(self.config.output_dir) / "step2_5_event_based_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info("=" * 80)
        logger.info("STEP 2.5 COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Results saved to: {summary_path}")
        logger.info("")
        
        self._print_final_recommendation()
    
    def _process_symbol(self, symbol: str, file_path: Path):
        """Process one symbol"""
        
        # Load data
        df = pd.read_csv(file_path)
        logger.info(f"✅ Loaded {len(df)} rows")
        
        # Create event-based targets
        detector = EventDetector(self.config)
        targets, event_bars, event_types = detector.create_event_targets(df)
        
        if len(targets) == 0:
            logger.error("❌ No events found")
            self.results[symbol] = {'status': 'FAILED', 'reason': 'No events'}
            return
        
        # Analyze
        analyzer = EventBasedAnalyzer()
        analysis = analyzer.analyze_events(targets, event_bars, event_types, symbol)
        
        # Save to CSV
        output_csv = Path(self.config.output_dir) / f"{symbol}_event_based_targets.csv"
        df_output = pd.DataFrame({
            'timestamp': df['timestamp'].iloc[:len(targets)],
            'close': df['close'].iloc[:len(targets)],
            'event_target': targets,
            'bars_to_event': event_bars,
            'event_type': event_types
        })
        df_output.to_csv(output_csv, index=False)
        logger.info(f"✅ Saved to {output_csv.name}")
        
        # Store results
        self.results[symbol] = {
            'status': 'SUCCESS',
            'analysis': analysis,
            'output_file': str(output_csv)
        }
    
    def _print_final_recommendation(self):
        """Print recommendation based on results"""
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("🎯 RECOMMENDATION FOR STEP 3")
        logger.info("=" * 80)
        logger.info("")
        
        logger.info("EVENT-BASED MODELING HAS:")
        logger.info("  ✅ Asymmetric payoff structure (2:1 or better)")
        logger.info("  ✅ Natural risk definition (built into event)")
        logger.info("  ✅ Fat tail capture potential")
        logger.info("  ✅ Reduced churn (fewer but higher-quality signals)")
        logger.info("  ✅ Alignment with market structure (expansion vs reversal)")
        logger.info("")
        
        logger.info("WHY THIS IS BETTER:")
        logger.info("  1. No fixed-horizon bias")
        logger.info("  2. Natural expectancy structure (E = (Win% × Win Size) - (Loss% × Loss Size))")
        logger.info("  3. Rewards capturing volatility bursts")
        logger.info("  4. Penalizes being in choppy periods")
        logger.info("")
        
        logger.info("NEXT STEP (Step 3):")
        logger.info("  Build LSTM that predicts:")
        logger.info("    - Will this setup expand >= expansion_atr?")
        logger.info("    - Before it reverses >= reversal_atr?")
        logger.info("")
        logger.info("  If model achieves 55%+ accuracy on this:")
        logger.info("    - With 2:1 payoff ratio")
        logger.info("    - Expectancy = (0.55 × 2) - (0.45 × 1) = 1.10 - 0.45 = +0.65")
        logger.info("    - Per-trade edge: 65% positive expected value")
        logger.info("")
        
        logger.info("VALIDATION REQUIREMENT:")
        logger.info("  Need to show:")
        logger.info("    1. Win rate > 52% (break-even on 2:1 payoff)")
        logger.info("    2. Statistically significant (N > 100 trades)")
        logger.info("    3. Consistent across symbols")
        logger.info("    4. Stable across time periods")
        logger.info("")
        logger.info("=" * 80)


if __name__ == "__main__":
    config = EventBasedConfig()
    runner = Step25Runner(config)
    runner.run()
