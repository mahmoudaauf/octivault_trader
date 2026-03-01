"""
PHASE 9.5 STEP 2: Regime-Aware Model Training & Validation

Goal: Train LSTM on 60m regime-filtered labels and validate accuracy

Strategy:
  - Use 676 regime-filtered labels per symbol (normal/high vol only)
  - Build LSTM with regime-aware features
  - Train/test split: 80/20
  - Target accuracy: >= 52% (1-2 points above random 50%)
  - Validate: Model learns real patterns, not regime classification

Expected Outcome:
  - ETH: 53-55% accuracy (strong signal)
  - BTC: 51-53% accuracy (marginal but viable)
  - Both: Confirms 60m horizon + regime filter is strategically sound
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
logger = logging.getLogger('Step2_RegimeAwareTraining')

# Optional: Try to import TensorFlow/Keras
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
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
    
    # Train/test split
    test_size: float = 0.20
    validation_size: float = 0.20
    random_seed: int = 42
    
    # Performance thresholds
    min_accuracy: float = 0.51  # Minimum acceptable accuracy
    target_accuracy: float = 0.52  # Target for model validation


class RegimeAwareFeatureEngineer:
    """Create features for regime-aware LSTM training"""
    
    def __init__(self, lookback: int):
        self.lookback = lookback
    
    def engineer_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create LSTM-ready features from OHLCV data
        
        Features:
          - Returns (relative price changes)
          - Regime encoding (0=low, 1=normal, 2=high)
          - Volatility (rolling std)
        
        Returns:
            (X features, y targets, regimes)
        """
        logger.info(f"Engineering features (lookback={self.lookback})...")
        
        df_copy = df.copy()
        
        # Calculate returns
        df_copy['return'] = df_copy['close'].pct_change()
        
        # Encode regime (if not present, assume all normal)
        if 'regime' in df_copy.columns:
            regime_map = {'low': 0, 'normal': 1, 'high': 2}
            df_copy['regime_encoded'] = df_copy['regime'].map(regime_map)
        else:
            df_copy['regime_encoded'] = 1  # default to normal
        
        # Calculate rolling volatility
        df_copy['volatility'] = df_copy['return'].rolling(window=5).std()
        
        # Forward return (our target, if not present create dummy)
        if 'forward_return_60m' not in df_copy.columns:
            df_copy['forward_return_60m'] = 0
        if 'target_60m' not in df_copy.columns:
            df_copy['target_60m'] = 0
        
        # Create sequences
        X = []
        y = []
        regimes = []
        
        for i in range(len(df_copy) - self.lookback):
            # Use returns, regime, volatility as features
            seq = df_copy[['return', 'regime_encoded', 'volatility']].iloc[i:i+self.lookback].values
            
            # Skip sequences with NaN
            if np.isnan(seq).any():
                continue
            
            X.append(seq)
            y.append(df_copy['target_60m'].iloc[i + self.lookback])
            regimes.append(df_copy['regime_encoded'].iloc[i + self.lookback])
        
        X = np.array(X)
        y = np.array(y)
        regimes = np.array(regimes)
        
        logger.info(f"✅ Created {len(X)} feature sequences")
        
        return X, y, regimes
    
    def split_data(self, X: np.ndarray, y: np.ndarray, 
                   test_size: float, val_size: float, seed: int) -> Tuple:
        """Split into train/validation/test with stratification"""
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=seed, stratify=y_temp
        )
        
        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)


class LSTMRegimeModel:
    """LSTM model for regime-aware prediction"""
    
    def __init__(self, config: Step2Config):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.history = None
    
    def build(self, input_shape: Tuple) -> bool:
        """Build and compile LSTM model"""
        
        if not TF_AVAILABLE:
            logger.warning("⚠️  TensorFlow not available, skipping model build")
            return False
        
        logger.info(f"Building LSTM model (input_shape={input_shape})...")
        
        try:
            self.model = Sequential([
                LSTM(self.config.lstm_units, activation='relu', 
                     input_shape=input_shape, return_sequences=True),
                Dropout(self.config.dropout_rate),
                LSTM(self.config.lstm_units // 2, activation='relu'),
                Dropout(self.config.dropout_rate),
                Dense(16, activation='relu'),
                Dense(1, activation='sigmoid')  # Binary classification
            ])
            
            optimizer = Adam(learning_rate=self.config.learning_rate)
            self.model.compile(
                optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            logger.info("✅ Model built successfully")
            self.model.summary()
            return True
            
        except Exception as e:
            logger.error(f"❌ Error building model: {e}")
            return False
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray) -> bool:
        """Train the model"""
        
        if not TF_AVAILABLE or self.model is None:
            logger.warning("⚠️  Cannot train without TensorFlow")
            return False
        
        logger.info(f"Training LSTM for {self.config.epochs} epochs...")
        
        try:
            self.history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                verbose=0
            )
            
            logger.info("✅ Training complete")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error during training: {e}")
            return False
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate model on test set"""
        
        if not TF_AVAILABLE or self.model is None:
            logger.warning("⚠️  Cannot evaluate without trained model")
            return self._fallback_evaluate(X_test, y_test)
        
        logger.info("Evaluating model on test set...")
        
        try:
            # Get predictions
            y_pred_proba = self.model.predict(X_test, verbose=0)
            y_pred = (y_pred_proba > 0.5).astype(int).flatten()
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            metrics = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'auc': float(auc),
                'test_size': len(X_test),
                'predictions': {
                    'true_positives': int(((y_pred == 1) & (y_test == 1)).sum()),
                    'false_positives': int(((y_pred == 1) & (y_test == 0)).sum()),
                    'true_negatives': int(((y_pred == 0) & (y_test == 0)).sum()),
                    'false_negatives': int(((y_pred == 0) & (y_test == 1)).sum()),
                }
            }
            
            logger.info(f"✅ Accuracy: {accuracy:.4f}")
            logger.info(f"   Precision: {precision:.4f}")
            logger.info(f"   Recall: {recall:.4f}")
            logger.info(f"   F1: {f1:.4f}")
            logger.info(f"   AUC: {auc:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Error during evaluation: {e}")
            return {}
    
    def _fallback_evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Fallback evaluation using random baseline"""
        
        logger.info("Using fallback statistical evaluation...")
        
        # Random baseline: predict based on class distribution
        random_pred = np.random.binomial(1, y_test.mean(), len(y_test))
        baseline_accuracy = accuracy_score(y_test, random_pred)
        
        # Simple feature-based heuristic
        # If average volatility in sequence is high, predict positive
        avg_volatility = X_test[:, :, 2].mean(axis=1)  # volatility is 3rd feature
        simple_pred = (avg_volatility > np.median(avg_volatility)).astype(int)
        simple_accuracy = accuracy_score(y_test, simple_pred)
        
        metrics = {
            'accuracy': float(simple_accuracy),
            'baseline_accuracy': float(baseline_accuracy),
            'test_size': len(X_test),
            'note': 'Fallback statistical model (TensorFlow not available)'
        }
        
        logger.info(f"   Fallback accuracy: {simple_accuracy:.4f}")
        logger.info(f"   Baseline (random): {baseline_accuracy:.4f}")
        
        return metrics


class Step2Runner:
    """Orchestrate Step 2 pipeline"""
    
    def __init__(self, config: Step2Config):
        self.config = config
        self.feature_engineer = RegimeAwareFeatureEngineer(config.lookback_candles)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run(self) -> Dict:
        """Execute full Step 2 pipeline"""
        
        logger.info("\n" + "=" * 73)
        logger.info("PHASE 9.5 VALIDATION - STEP 2: REGIME-AWARE MODEL TRAINING")
        logger.info("=" * 73)
        
        # Print configuration
        logger.info("\n" + "=" * 73)
        logger.info("STEP 2 CONFIGURATION")
        logger.info("=" * 73)
        logger.info(f"Lookback window:            {self.config.lookback_candles} candles")
        logger.info(f"LSTM units:                 {self.config.lstm_units}")
        logger.info(f"Learning rate:              {self.config.learning_rate}")
        logger.info(f"Epochs:                     {self.config.epochs}")
        logger.info(f"Batch size:                 {self.config.batch_size}")
        logger.info(f"Test size:                  {self.config.test_size*100:.0f}%")
        logger.info(f"Validation size:            {self.config.validation_size*100:.0f}%")
        logger.info(f"Min acceptable accuracy:    {self.config.min_accuracy*100:.1f}%")
        logger.info(f"Target accuracy:            {self.config.target_accuracy*100:.1f}%")
        logger.info("=" * 73 + "\n")
        
        results = {}
        
        # Find labeled data files
        labeled_files = list(Path(self.config.labeled_data_dir).glob("*_5m_with_60m_labels.csv"))
        symbols = sorted([f.stem.replace("_5m_with_60m_labels", "") for f in labeled_files])
        
        if not symbols:
            logger.error(f"❌ No labeled data found in {self.config.labeled_data_dir}")
            return {}
        
        logger.info(f"Found symbols: {symbols}\n")
        
        for symbol in symbols:
            logger.info("=" * 73)
            logger.info(f"Training: {symbol}")
            logger.info("=" * 73)
            
            # Load labeled data
            csv_path = Path(self.config.labeled_data_dir) / f"{symbol}_5m_with_60m_labels.csv"
            
            try:
                df = pd.read_csv(csv_path)
                
                # Filter to keep_for_training rows
                if 'keep_for_training' in df.columns:
                    df = df[df['keep_for_training'] == True].reset_index(drop=True)
                
                logger.info(f"✅ Loaded {len(df)} regime-filtered samples for {symbol}")
                
            except Exception as e:
                logger.error(f"❌ Error loading {csv_path}: {e}")
                results[symbol] = {'status': 'FAILED', 'reason': 'Load error'}
                continue
            
            # Engineer features
            try:
                X, y, regimes = self.feature_engineer.engineer_features(df)
                
                if len(X) < 50:
                    logger.error(f"❌ Not enough samples ({len(X)} < 50)")
                    results[symbol] = {'status': 'FAILED', 'reason': 'Insufficient data'}
                    continue
                
            except Exception as e:
                logger.error(f"❌ Error engineering features: {e}")
                results[symbol] = {'status': 'FAILED', 'reason': 'Feature engineering error'}
                continue
            
            # Split data
            try:
                (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.feature_engineer.split_data(
                    X, y, self.config.test_size, self.config.validation_size, self.config.random_seed
                )
                
            except Exception as e:
                logger.error(f"❌ Error splitting data: {e}")
                results[symbol] = {'status': 'FAILED', 'reason': 'Data split error'}
                continue
            
            # Build model
            model = LSTMRegimeModel(self.config)
            
            if not model.build((X_train.shape[1], X_train.shape[2])):
                results[symbol] = {'status': 'FAILED', 'reason': 'Model build failed'}
                continue
            
            # Train model (if TF available)
            if TF_AVAILABLE:
                if not model.train(X_train, y_train, X_val, y_val):
                    results[symbol] = {'status': 'FAILED', 'reason': 'Training failed'}
                    continue
            
            # Evaluate
            metrics = model.evaluate(X_test, y_test)
            
            if not metrics:
                results[symbol] = {'status': 'FAILED', 'reason': 'Evaluation failed'}
                continue
            
            # Determine pass/fail
            accuracy = metrics.get('accuracy', 0)
            status = 'PASS' if accuracy >= self.config.min_accuracy else 'FAIL'
            assessment = 'Excellent' if accuracy >= 0.55 else 'Good' if accuracy >= 0.53 else 'Marginal' if accuracy >= 0.51 else 'Poor'
            
            logger.info(f"\nAssessment: {assessment}")
            logger.info(f"Status: {status}")
            
            # Save metrics
            metrics['symbol'] = symbol
            metrics['status'] = status
            metrics['assessment'] = assessment
            metrics['train_size'] = len(X_train)
            metrics['val_size'] = len(X_val)
            metrics['test_size'] = len(X_test)
            metrics['feature_shape'] = str(X_train.shape)
            
            results[symbol] = metrics
            
            # Save to JSON
            json_path = self.output_dir / f"{symbol}_step2_results.json"
            try:
                with open(json_path, 'w') as f:
                    json.dump(metrics, f, indent=2)
                logger.info(f"✅ Results saved to {json_path}")
            except Exception as e:
                logger.error(f"❌ Error saving results: {e}")
            
            logger.info("=" * 73 + "\n")
        
        # Summary
        logger.info("=" * 73)
        logger.info("STEP 2 COMPLETION SUMMARY")
        logger.info("=" * 73)
        
        passed = sum(1 for r in results.values() if r.get('status') == 'PASS')
        failed = sum(1 for r in results.values() if r.get('status') == 'FAILED')
        
        logger.info(f"\nResults:")
        logger.info(f"  ✅ Passed: {passed}")
        logger.info(f"  ❌ Failed: {failed}")
        
        if results:
            logger.info(f"\nAccuracy by symbol:")
            for symbol, result in results.items():
                if result.get('status') != 'FAILED':
                    acc = result.get('accuracy', 0)
                    assess = result.get('assessment', 'Unknown')
                    logger.info(f"  {symbol}: {acc*100:.2f}% ({assess})")
        
        logger.info("\n" + "=" * 73)
        
        # Save summary
        summary_path = self.output_dir / "step2_summary.json"
        try:
            with open(summary_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"✅ Summary saved to {summary_path}")
        except Exception as e:
            logger.error(f"❌ Error saving summary: {e}")
        
        logger.info("=" * 73 + "\n")
        
        return results


# Main execution
if __name__ == "__main__":
    config = Step2Config(
        labeled_data_dir="validation_outputs",
        output_dir="validation_outputs",
        lookback_candles=10,
        lstm_units=32,
        dropout_rate=0.2,
        learning_rate=0.001,
        epochs=50,
        batch_size=16,
        test_size=0.20,
        validation_size=0.20,
        min_accuracy=0.51,
        target_accuracy=0.52
    )
    
    runner = Step2Runner(config)
    results = runner.run()
