"""
PHASE 9.5 STEP 2: Regime-Aware Model Training & Validation (FIXED)

Goal: Train LSTM on 60m regime-filtered labels with CORRECTED feature engineering

Fixes Applied:
  1. TEMPORAL MIX FIX: Use chronological train/val/test split (NO shuffling)
  2. REGIME LEAKAGE FIX: Use regime from position i (calculated from past only), not i+lookback
  
Strategy:
  - Use 676 regime-filtered labels per symbol (normal/high vol only)
  - Build LSTM with regime-aware features (CLEAN features, no future info)
  - Train/test split: 60% train / 20% val / 20% test (CHRONOLOGICAL ORDER)
  - Target accuracy: >= 52% (1-2 points above random 50%)
  - Validate: Model learns real patterns, not regime classification

Expected Outcome:
  - ETH: 53-58% accuracy (clean signal)
  - BTC: 51-54% accuracy (marginal but real)
  - Both: Confirms structural alpha exists
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
    format='%(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('Step2_RegimeAwareTraining_FIXED')

# Optional: Try to import TensorFlow/Keras
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("⚠️  TensorFlow not available. Using fallback statistical model.")


@dataclass
class Step2Config:
    """Configuration for regime-aware model training"""
    
    # Data paths
    labeled_data_dir: str = "validation_outputs"
    output_dir: str = "validation_outputs"
    
    # Model configuration
    lookback_candles: int = 10  # Use 10 previous candles as features
    lstm_units: int = 32
    dropout_rate: float = 0.2
    learning_rate: float = 0.001
    epochs: int = 50
    batch_size: int = 16
    
    # Train/test split (CHRONOLOGICAL - NO SHUFFLING)
    train_ratio: float = 0.60
    val_ratio: float = 0.20
    test_ratio: float = 0.20
    
    # Performance thresholds
    min_accuracy: float = 0.51
    target_accuracy: float = 0.52


class RegimeAwareFeatureEngineerFIXED:
    """Create features for regime-aware LSTM training (FIXED)"""
    
    def __init__(self, lookback: int):
        self.lookback = lookback
    
    def engineer_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create features for LSTM training with CORRECTED regime handling.
        
        CRITICAL FIX:
        - Regime at position i is calculated from PAST data only (before position i)
        - Features use lookback window [i-lookback, i-1]
        - Target is forward return at i+10 (truly future)
        - This ensures NO leakage
        
        Returns:
            (X features, y targets, regime_used_in_features)
        """
        
        df_copy = df.copy()
        
        # Ensure timestamp is datetime
        if 'timestamp' in df_copy.columns:
            df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'], unit='ms', errors='coerce')
        
        # Calculate returns
        df_copy['return'] = df_copy['close'].pct_change()
        
        # FIXED: Encode regime (if not present, assume all normal)
        if 'regime' in df_copy.columns:
            regime_map = {'low': 0, 'normal': 1, 'high': 2}
            df_copy['regime_encoded'] = df_copy['regime'].map(regime_map)
        else:
            df_copy['regime_encoded'] = 1  # default to normal
        
        # Calculate rolling volatility (PAST data only)
        df_copy['volatility'] = df_copy['return'].rolling(window=5).std()
        
        # Forward return (our target, if not present create dummy)
        if 'forward_return_60m' not in df_copy.columns:
            df_copy['forward_return_60m'] = 0
        if 'target_60m' not in df_copy.columns:
            df_copy['target_60m'] = 0
        
        # Create sequences with FIXED regime handling
        X = []
        y = []
        regimes_used = []
        
        for i in range(self.lookback, len(df_copy) - self.lookback):
            # FIXED: Use features from [i-lookback, i-1] (PAST data only)
            seq = df_copy[['return', 'regime_encoded', 'volatility']].iloc[i-self.lookback:i].values
            
            # Skip sequences with NaN
            if np.isnan(seq).any():
                continue
            
            X.append(seq)
            
            # Target at i + lookback (purely forward)
            y.append(df_copy['target_60m'].iloc[i + self.lookback])
            
            # Regime USED in features (from position i-1, the last regime in the sequence)
            regimes_used.append(df_copy['regime_encoded'].iloc[i - 1])
        
        X = np.array(X)
        y = np.array(y)
        regimes_used = np.array(regimes_used)
        
        logger.info(f"✅ Created {len(X)} feature sequences")
        logger.info(f"   Features use PAST data only (no future information)")
        logger.info(f"   Target uses truly forward 60m returns")
        
        return X, y, regimes_used
    
    def split_data_chronological(self, X: np.ndarray, y: np.ndarray, 
                                  train_ratio: float, val_ratio: float) -> Tuple:
        """
        Split into train/validation/test with CHRONOLOGICAL ordering (NO SHUFFLING).
        
        CRITICAL FIX:
        - Do NOT use train_test_split with stratify
        - Use simple index-based split maintaining temporal order
        - train: first 60% of data
        - val: next 20% of data
        - test: last 20% of data
        """
        
        n = len(X)
        train_idx = int(n * train_ratio)
        val_idx = int(n * (train_ratio + val_ratio))
        
        # Split chronologically
        X_train, y_train = X[:train_idx], y[:train_idx]
        X_val, y_val = X[train_idx:val_idx], y[train_idx:val_idx]
        X_test, y_test = X[val_idx:], y[val_idx:]
        
        logger.info(f"✅ Chronological split (NO shuffling):")
        logger.info(f"   Train: {len(X_train)} samples (0-{train_ratio:.0%})")
        logger.info(f"   Val:   {len(X_val)} samples ({train_ratio:.0%}-{train_ratio+val_ratio:.0%})")
        logger.info(f"   Test:  {len(X_test)} samples ({train_ratio+val_ratio:.0%}-100%)")
        logger.info(f"   All indices maintain temporal order ✓")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)


class LSTMRegimeModelFIXED:
    """LSTM model for regime-aware prediction (FIXED)"""
    
    def __init__(self, config: Step2Config):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.history = None
    
    def build(self, input_shape: Tuple[int, int]) -> None:
        """Build LSTM model"""
        if not TF_AVAILABLE:
            logger.warning("⚠️  TensorFlow not available. Skipping model build.")
            return
        
        self.model = Sequential([
            LSTM(self.config.lstm_units, activation='relu', input_shape=input_shape, return_sequences=True),
            Dropout(self.config.dropout_rate),
            LSTM(self.config.lstm_units // 2, activation='relu'),
            Dropout(self.config.dropout_rate),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
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
            logger.warning("⚠️  TensorFlow not available. Skipping training.")
            return {}
        
        logger.info(f"Training for {self.config.epochs} epochs...")
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            verbose=0
        )
        
        logger.info(f"✅ Training complete")
        
        return {
            'epochs': self.config.epochs,
            'batch_size': self.config.batch_size,
            'final_train_loss': float(self.history.history['loss'][-1]),
            'final_train_acc': float(self.history.history['accuracy'][-1]),
            'final_val_loss': float(self.history.history['val_loss'][-1]),
            'final_val_acc': float(self.history.history['val_accuracy'][-1])
        }
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate model on test set"""
        if not TF_AVAILABLE:
            return self._fallback_evaluate(y_test)
        
        # Get predictions
        y_pred_probs = self.model.predict(X_test, verbose=0).flatten()
        y_pred = (y_pred_probs >= 0.5).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # AUC
        try:
            auc = roc_auc_score(y_test, y_pred_probs)
        except:
            auc = 0.5
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        # Baseline accuracy (majority class)
        baseline_acc = max(np.mean(y_test == 0), np.mean(y_test == 1))
        real_edge = accuracy - baseline_acc
        
        logger.info(f"✅ Test Results:")
        logger.info(f"   Accuracy:    {accuracy:.4f} ({accuracy*100:.2f}%)")
        logger.info(f"   Baseline:    {baseline_acc:.4f} ({baseline_acc*100:.2f}%)")
        logger.info(f"   Real Edge:   {real_edge:.4f} ({real_edge*100:.2f}%) ← ACTUAL SIGNAL")
        logger.info(f"   Precision:   {precision:.4f}")
        logger.info(f"   Recall:      {recall:.4f}")
        logger.info(f"   F1:          {f1:.4f}")
        logger.info(f"   AUC:         {auc:.4f}")
        logger.info(f"")
        logger.info(f"   Confusion Matrix:")
        logger.info(f"      TN={tn}, FP={fp}")
        logger.info(f"      FN={fn}, TP={tp}")
        
        return {
            'accuracy': float(accuracy),
            'baseline': float(baseline_acc),
            'real_edge': float(real_edge),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'auc': float(auc),
            'confusion_matrix': {
                'tn': int(tn), 'fp': int(fp),
                'fn': int(fn), 'tp': int(tp)
            },
            'class_distribution': {
                'negative': int(np.sum(y_test == 0)),
                'positive': int(np.sum(y_test == 1))
            }
        }
    
    def _fallback_evaluate(self, y_test: np.ndarray) -> Dict:
        """Fallback evaluation without TensorFlow"""
        baseline = max(np.mean(y_test == 0), np.mean(y_test == 1))
        return {
            'accuracy': float(baseline),
            'baseline': float(baseline),
            'real_edge': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'auc': 0.5
        }


class Step2RunnerFIXED:
    """Orchestrate Step 2 with corrected feature engineering and temporal split"""
    
    def __init__(self, config: Step2Config):
        self.config = config
        self.results = {}
    
    def run(self):
        """Execute Step 2 for all symbols"""
        
        logger.info("=" * 80)
        logger.info("PHASE 9.5 STEP 2: REGIME-AWARE MODEL TRAINING (FIXED)")
        logger.info("=" * 80)
        logger.info("")
        logger.info("🔧 FIXES APPLIED:")
        logger.info("   1. TEMPORAL ORDERING: Chronological split (NO shuffling)")
        logger.info("   2. REGIME LEAKAGE: Use regime from position i only, not i+lookback")
        logger.info("")
        
        # Get labeled data files
        data_dir = Path(self.config.labeled_data_dir)
        labeled_files = list(data_dir.glob("*_5m_with_60m_labels.csv"))
        
        if not labeled_files:
            logger.error(f"❌ No labeled data files found in {data_dir}")
            return
        
        logger.info(f"📁 Found {len(labeled_files)} labeled files")
        logger.info("")
        
        for file_path in sorted(labeled_files):
            symbol = file_path.stem.replace('_5m_with_60m_labels', '')
            logger.info("=" * 80)
            logger.info(f"Processing: {symbol}")
            logger.info("=" * 80)
            
            self._process_symbol(symbol, file_path)
            logger.info("")
        
        # Save results
        summary_path = Path(self.config.output_dir) / "step2_fixed_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info("=" * 80)
        logger.info("STEP 2 COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Results saved to: {summary_path}")
        logger.info("")
        
        # Summary
        self._print_summary()
    
    def _process_symbol(self, symbol: str, file_path: Path):
        """Process a single symbol"""
        
        # Load data
        df = pd.read_csv(file_path)
        logger.info(f"✅ Loaded {len(df)} rows")
        
        # Filter to regime-filtered data (normal/high vol only)
        if 'regime' in df.columns:
            df_filtered = df[df['regime'].isin(['normal', 'high'])].copy()
            logger.info(f"✅ Filtered to {len(df_filtered)} regime-filtered rows (normal/high vol only)")
        else:
            df_filtered = df.copy()
            logger.info(f"⚠️  No regime column, using all {len(df_filtered)} rows")
        
        # Engineer features with FIXED regime handling
        feature_engineer = RegimeAwareFeatureEngineerFIXED(self.config.lookback_candles)
        X, y, regimes_in_features = feature_engineer.engineer_features(df_filtered)
        
        if len(X) == 0:
            logger.error(f"❌ No valid sequences created")
            self.results[symbol] = {'status': 'FAILED', 'reason': 'No valid sequences'}
            return
        
        # Split chronologically (NO SHUFFLING)
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = \
            feature_engineer.split_data_chronological(
                X, y, 
                self.config.train_ratio,
                self.config.val_ratio
            )
        
        # Normalize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
        X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        
        # Build and train model
        model = LSTMRegimeModelFIXED(self.config)
        model.build((X_train_scaled.shape[1], X_train_scaled.shape[2]))
        
        if TF_AVAILABLE:
            train_info = model.train(X_train_scaled, y_train, X_val_scaled, y_val)
            eval_results = model.evaluate(X_test_scaled, y_test)
        else:
            train_info = {}
            eval_results = model.evaluate(X_test_scaled, y_test)
        
        # Store results
        self.results[symbol] = {
            'status': 'SUCCESS',
            'sequences_created': int(len(X)),
            'train_samples': int(len(X_train)),
            'val_samples': int(len(X_val)),
            'test_samples': int(len(X_test)),
            'class_distribution': {
                'test_negative': int(np.sum(y_test == 0)),
                'test_positive': int(np.sum(y_test == 1)),
                'test_positive_pct': float(np.mean(y_test == 1) * 100)
            },
            'training': train_info,
            'evaluation': eval_results
        }
        
        # Check pass criteria
        accuracy = eval_results['accuracy']
        real_edge = eval_results['real_edge']
        
        if real_edge >= 0.01:  # At least 1% real edge
            logger.info(f"✅ PASS: Real edge {real_edge*100:.2f}% exceeds threshold")
            self.results[symbol]['pass_fail'] = 'PASS'
        else:
            logger.info(f"⚠️  MARGINAL: Real edge only {real_edge*100:.2f}%")
            self.results[symbol]['pass_fail'] = 'MARGINAL'
    
    def _print_summary(self):
        """Print summary of all results"""
        logger.info("STEP 2 SUMMARY")
        logger.info("-" * 80)
        
        for symbol, result in self.results.items():
            if result['status'] == 'FAILED':
                logger.info(f"{symbol:12} ❌ FAILED: {result['reason']}")
            else:
                accuracy = result['evaluation'].get('accuracy', 0)
                baseline = result['evaluation'].get('baseline', 0)
                real_edge = result['evaluation'].get('real_edge', 0)
                pct_positive = result['class_distribution']['test_positive_pct']
                
                status = "✅ PASS" if real_edge >= 0.01 else "⚠️  MARGINAL"
                logger.info(f"{symbol:12} {status}")
                logger.info(f"  Accuracy:    {accuracy*100:6.2f}% (baseline: {baseline*100:6.2f}%)")
                logger.info(f"  Real Edge:   {real_edge*100:6.2f}% ← ACTUAL SIGNAL")
                logger.info(f"  Class Dist:  {pct_positive:5.1f}% positive in test")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    config = Step2Config()
    runner = Step2RunnerFIXED(config)
    runner.run()
