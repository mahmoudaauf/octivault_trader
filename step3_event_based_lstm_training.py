"""
PHASE 9.5 STEP 3: EVENT-BASED LSTM TRAINING

Goal: Train LSTM to predict expansion vs reversal events

Target: Binary classification
  1 = Price will expand >= expansion_atr BEFORE reversing >= reversal_atr
  0 = Price will reverse >= reversal_atr BEFORE expanding >= expansion_atr

Success Criteria:
  Accuracy >= 55% (break-even on 2:1 payoff is 52%)
  If achieved: Edge = (0.55 × 2) - (0.45 × 1) = +0.65 per trade

Architecture:
  - LSTM with 2-3 layers
  - Dropout for regularization
  - Temporal features: past returns, volatility, trend
  - No future information leakage
  - Chronological train/test split (NO shuffling)

Key Insight:
  If model achieves 55%+ on this asymmetric target,
  it demonstrates REAL understanding of market structure,
  not just directional bias.
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, List
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger('Step3_EventBasedLSTM')

# Try to import TensorFlow
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("⚠️  TensorFlow not available.")


@dataclass
class Step3Config:
    """Configuration for event-based LSTM training"""
    
    # Data paths
    event_data_dir: str = "validation_outputs"
    output_dir: str = "validation_outputs"
    
    # Feature engineering
    lookback_candles: int = 10  # 50 minutes of 5m bars
    
    # Model architecture
    lstm_units: int = 32
    lstm_layers: int = 2
    dropout_rate: float = 0.2
    learning_rate: float = 0.001
    
    # Training
    epochs: int = 50
    batch_size: int = 16
    
    # Split ratios (CHRONOLOGICAL)
    train_ratio: float = 0.60
    val_ratio: float = 0.20
    test_ratio: float = 0.20
    
    # Success thresholds
    min_accuracy: float = 0.52  # Break-even on 2:1 payoff
    target_accuracy: float = 0.55  # Target edge


class EventBasedFeatureEngineer:
    """Create temporal features from event-based data"""
    
    def __init__(self, lookback: int):
        self.lookback = lookback
    
    def engineer_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create features from price data
        
        Features per timestep:
          1. Return (5m close-to-close % change)
          2. Volatility (rolling 5-bar standard deviation)
          3. Momentum (ratio of positive to negative bars)
          4. Price position (close relative to 20-bar high/low)
          
        Returns:
            (X features shape [N, lookback, features], y targets)
        """
        
        df_copy = df.copy()
        
        # Calculate returns
        df_copy['return'] = df_copy['close'].pct_change()
        
        # Calculate volatility (rolling std)
        df_copy['volatility'] = df_copy['return'].rolling(window=5).std()
        
        # Calculate momentum (direction bias)
        df_copy['bar_direction'] = (df_copy['close'] > df_copy['open']).astype(int)
        df_copy['momentum'] = df_copy['bar_direction'].rolling(window=5).mean()  # Ratio of up bars
        
        # Price position in range (0-1 scale)
        df_copy['high_20'] = df_copy['high'].rolling(window=20).max()
        df_copy['low_20'] = df_copy['low'].rolling(window=20).min()
        df_copy['position'] = (df_copy['close'] - df_copy['low_20']) / (df_copy['high_20'] - df_copy['low_20'])
        df_copy['position'] = df_copy['position'].fillna(0.5).clip(0, 1)
        
        # Create sequences
        X = []
        y = []
        
        for i in range(self.lookback, len(df_copy)):
            # Get lookback window of features
            seq = df_copy[['return', 'volatility', 'momentum', 'position']].iloc[i-self.lookback:i].values
            
            # Skip if NaN
            if np.isnan(seq).any():
                continue
            
            X.append(seq)
            
            # Get target
            if 'event_target' in df_copy.columns:
                y.append(df_copy['event_target'].iloc[i])
            else:
                y.append(0)  # Default
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"✅ Created {len(X)} feature sequences ({len(y) - np.sum(y)} negatives, {np.sum(y)} positives)")
        
        return X, y
    
    def split_chronological(self, X: np.ndarray, y: np.ndarray,
                           train_ratio: float, val_ratio: float) -> Tuple:
        """
        Split chronologically (NO shuffling to prevent look-ahead bias)
        """
        
        n = len(X)
        train_idx = int(n * train_ratio)
        val_idx = int(n * (train_ratio + val_ratio))
        
        X_train, y_train = X[:train_idx], y[:train_idx]
        X_val, y_val = X[train_idx:val_idx], y[train_idx:val_idx]
        X_test, y_test = X[val_idx:], y[val_idx:]
        
        logger.info(f"✅ Chronological split (maintaining temporal order):")
        logger.info(f"   Train: {len(X_train)} ({train_ratio*100:.0f}%)")
        logger.info(f"   Val:   {len(X_val)} ({val_ratio*100:.0f}%)")
        logger.info(f"   Test:  {len(X_test)} ({(1-train_ratio-val_ratio)*100:.0f}%)")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)


class EventBasedLSTMModel:
    """LSTM for event-based expansion prediction"""
    
    def __init__(self, config: Step3Config):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.history = None
    
    def build(self, input_shape: Tuple[int, int]) -> None:
        """Build LSTM model"""
        
        if not TF_AVAILABLE:
            logger.warning("⚠️  TensorFlow not available")
            return
        
        layers = [
            LSTM(self.config.lstm_units, activation='relu', input_shape=input_shape, return_sequences=True)
        ]
        
        # Add additional LSTM layers
        for _ in range(self.config.lstm_layers - 1):
            layers.append(Dropout(self.config.dropout_rate))
            layers.append(LSTM(self.config.lstm_units // 2, activation='relu', return_sequences=False))
        
        layers.extend([
            Dropout(self.config.dropout_rate),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        self.model = Sequential(layers)
        
        self.model.compile(
            optimizer=Adam(learning_rate=self.config.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info(f"✅ Model built: {self.model.count_params()} parameters")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """Train model"""
        
        if not TF_AVAILABLE:
            return {}
        
        logger.info(f"🔄 Training for {self.config.epochs} epochs...")
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            verbose=0
        )
        
        logger.info(f"✅ Training complete")
        
        return {
            'final_train_acc': float(self.history.history['accuracy'][-1]),
            'final_val_acc': float(self.history.history['val_accuracy'][-1])
        }
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate on test set"""
        
        if not TF_AVAILABLE:
            return self._fallback_stats(y_test)
        
        # Predictions
        y_pred_probs = self.model.predict(X_test, verbose=0).flatten()
        y_pred = (y_pred_probs >= 0.5).astype(int)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        try:
            auc = roc_auc_score(y_test, y_pred_probs)
        except:
            auc = 0.5
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        # Calculate expectancy
        payoff_ratio = 2.0  # 2:1 expansion:reversal
        expectancy = (accuracy * payoff_ratio) - ((1 - accuracy) * 1.0)
        
        logger.info(f"")
        logger.info(f"📊 TEST SET RESULTS")
        logger.info(f"━" * 70)
        logger.info(f"Accuracy:        {accuracy*100:6.2f}%")
        logger.info(f"Precision:       {precision*100:6.2f}%")
        logger.info(f"Recall:          {recall*100:6.2f}%")
        logger.info(f"F1:              {f1*100:6.2f}%")
        logger.info(f"AUC:             {auc:6.4f}")
        logger.info(f"")
        logger.info(f"💰 EXPECTED VALUE (2:1 payoff)")
        logger.info(f"  E = (Win% × 2) - (Loss% × 1)")
        logger.info(f"  E = ({accuracy:.2%} × 2) - ({1-accuracy:.2%} × 1)")
        logger.info(f"  E = {expectancy:.4f} ({expectancy*100:.2f}% per trade)")
        logger.info(f"")
        
        if accuracy >= 0.55:
            logger.info(f"✅ PASS: Accuracy {accuracy*100:.2f}% exceeds 55% threshold")
            logger.info(f"   Per-trade edge: +{expectancy*100:.1f}%")
            status = "PASS"
        elif accuracy >= 0.52:
            logger.info(f"⚠️  MARGINAL: Accuracy {accuracy*100:.2f}% near break-even")
            logger.info(f"   Per-trade edge: +{expectancy*100:.1f}% (marginal)")
            status = "MARGINAL"
        else:
            logger.info(f"❌ FAIL: Accuracy {accuracy*100:.2f}% below break-even")
            status = "FAIL"
        
        logger.info(f"")
        logger.info(f"🎯 Confusion Matrix:")
        logger.info(f"   True Negatives:  {tn:4d} | False Positives: {fp:4d}")
        logger.info(f"   False Negatives: {fn:4d} | True Positives:  {tp:4d}")
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'auc': float(auc),
            'expectancy': float(expectancy),
            'status': status,
            'confusion_matrix': {
                'tn': int(tn), 'fp': int(fp),
                'fn': int(fn), 'tp': int(tp)
            },
            'class_distribution': {
                'negative': int(np.sum(y_test == 0)),
                'positive': int(np.sum(y_test == 1))
            }
        }
    
    def _fallback_stats(self, y_test: np.ndarray) -> Dict:
        """Fallback without TensorFlow"""
        baseline = max(np.mean(y_test == 0), np.mean(y_test == 1))
        return {
            'accuracy': float(baseline),
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'auc': 0.5,
            'expectancy': 0.0,
            'status': 'UNKNOWN'
        }


class Step3Runner:
    """Execute Step 3: Event-based LSTM training"""
    
    def __init__(self, config: Step3Config):
        self.config = config
        self.results = {}
    
    def run(self):
        """Execute for all symbols"""
        
        logger.info("=" * 80)
        logger.info("PHASE 9.5 STEP 3: EVENT-BASED LSTM TRAINING")
        logger.info("=" * 80)
        logger.info("")
        logger.info("🎯 OBJECTIVE")
        logger.info("   Predict: Will price expand >= 2x ATR before reversing 1x ATR?")
        logger.info("")
        logger.info("📊 SUCCESS CRITERIA")
        logger.info("   Accuracy >= 55% (2:1 payoff break-even = 52%)")
        logger.info("   Expected Value >= +0.60 per trade (0.55×2 - 0.45×1 = 0.65)")
        logger.info("")
        logger.info("🏗️  ARCHITECTURE")
        logger.info(f"   LSTM({self.config.lstm_units}) → Dropout → LSTM({self.config.lstm_units//2}) → Dense(16) → Dense(1)")
        logger.info(f"   Lookback: {self.config.lookback_candles} candles (50 min)")
        logger.info(f"   Features: return, volatility, momentum, position")
        logger.info("")
        
        # Load event-based targets
        data_dir = Path(self.config.event_data_dir)
        event_files = list(data_dir.glob("*_event_based_targets.csv"))
        
        if not event_files:
            logger.error(f"❌ No event-based targets found in {data_dir}")
            logger.error("   Please run step2_5_event_based_modeling.py first")
            return
        
        logger.info(f"📁 Found {len(event_files)} event-based target files")
        logger.info("")
        
        for file_path in sorted(event_files):
            symbol = file_path.stem.replace('_event_based_targets', '')
            logger.info("=" * 80)
            logger.info(f"Processing: {symbol}")
            logger.info("=" * 80)
            
            self._process_symbol(symbol, file_path)
            logger.info("")
        
        # Save results
        summary_path = Path(self.config.output_dir) / "step3_event_lstm_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info("=" * 80)
        logger.info("STEP 3 COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Results saved to: {summary_path}")
        logger.info("")
        
        self._print_recommendations()
    
    def _process_symbol(self, symbol: str, file_path: Path):
        """Process one symbol"""
        
        # Load event targets
        df_events = pd.read_csv(file_path)
        logger.info(f"✅ Loaded {len(df_events)} event-based targets")
        
        # Load original OHLCV data
        ohlcv_file = Path(self.config.event_data_dir) / f"{symbol}_5m_with_60m_labels.csv"
        if not ohlcv_file.exists():
            logger.error(f"❌ OHLCV file not found: {ohlcv_file}")
            self.results[symbol] = {'status': 'FAILED', 'reason': 'OHLCV data not found'}
            return
        
        df_ohlcv = pd.read_csv(ohlcv_file)
        logger.info(f"✅ Loaded {len(df_ohlcv)} OHLCV records")
        
        # Merge on timestamp
        df = pd.merge(df_ohlcv, df_events[['timestamp', 'event_target']], on='timestamp', how='inner')
        logger.info(f"✅ Merged to {len(df)} aligned records")
        
        # Engineer features
        engineer = EventBasedFeatureEngineer(self.config.lookback_candles)
        X, y = engineer.engineer_features(df)
        
        if len(X) < 100:
            logger.error(f"❌ Insufficient sequences ({len(X)} < 100)")
            self.results[symbol] = {'status': 'FAILED', 'reason': 'Insufficient sequences'}
            return
        
        # Split chronologically
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = \
            engineer.split_chronological(X, y, self.config.train_ratio, self.config.val_ratio)
        
        # Normalize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
        X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        
        # Build and train
        model = EventBasedLSTMModel(self.config)
        model.build((X_train_scaled.shape[1], X_train_scaled.shape[2]))
        
        if TF_AVAILABLE:
            train_info = model.train(X_train_scaled, y_train, X_val_scaled, y_val)
            eval_results = model.evaluate(X_test_scaled, y_test)
        else:
            train_info = {}
            eval_results = model.evaluate(X_test_scaled, y_test)
        
        # Store
        self.results[symbol] = {
            'status': 'SUCCESS',
            'sequences': int(len(X)),
            'train_samples': int(len(X_train)),
            'val_samples': int(len(X_val)),
            'test_samples': int(len(X_test)),
            'class_dist_train': {
                'negative': int(np.sum(y_train == 0)),
                'positive': int(np.sum(y_train == 1))
            },
            'class_dist_test': eval_results.get('class_distribution', {}),
            'training': train_info,
            'evaluation': eval_results
        }
    
    def _print_recommendations(self):
        """Print recommendations"""
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("📈 STEP 3 RESULTS & RECOMMENDATIONS")
        logger.info("=" * 80)
        logger.info("")
        
        for symbol, result in self.results.items():
            if result['status'] != 'SUCCESS':
                logger.info(f"{symbol}: ❌ FAILED ({result.get('reason', 'Unknown')})")
                continue
            
            accuracy = result['evaluation'].get('accuracy', 0)
            expectancy = result['evaluation'].get('expectancy', 0)
            status = result['evaluation'].get('status', 'UNKNOWN')
            
            logger.info(f"{symbol}:")
            logger.info(f"  Status:     {status}")
            logger.info(f"  Accuracy:   {accuracy*100:.2f}%")
            logger.info(f"  Expectancy: +{expectancy*100:.2f}% per trade")
        
        logger.info("")
        logger.info("🎯 NEXT STEPS:")
        logger.info("  1. If PASS (acc >= 55%):")
        logger.info("     → Proceed to Step 4: Backtesting")
        logger.info("     → Validate on out-of-sample data")
        logger.info("     → Test across different timeframes")
        logger.info("")
        logger.info("  2. If MARGINAL (52-55%):")
        logger.info("     → Consider feature refinement")
        logger.info("     → Try ensemble methods")
        logger.info("     → Expand parameter search")
        logger.info("")
        logger.info("  3. If FAIL (acc < 52%):")
        logger.info("     → Revisit event definition")
        logger.info("     → Try different ATR multiples")
        logger.info("     → Add additional features")
        logger.info("")
        logger.info("=" * 80)


if __name__ == "__main__":
    config = Step3Config()
    runner = Step3Runner(config)
    runner.run()
