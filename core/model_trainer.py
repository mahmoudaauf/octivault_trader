
import logging
import os
import time
import numpy as np
from typing import Any, Dict, Optional, Union
from pathlib import Path

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, Callback
except ImportError:
    tf = None
    EarlyStopping = None
    Callback = object

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import CalibratedClassifierCV
except ImportError:
    StandardScaler = None
    CalibratedClassifierCV = None

# Using ModelManager's helper to build paths if needed, 
# or we can redefine it locally to keep it standalone.
# We'll rely on the caller to handle paths or import from model_manager.
from core.model_manager import build_model_path, save_model, model_exists

class ModelTrainer:
    """
    Supervised Learning Trainer for binary classification.
    Trains a model to predict BUY (1) or HOLD/SELL (0) actions based on market data states.
    """
    def __init__(self, symbol: str, timeframe: str = "5m", input_lookback: int = 20, 
                 epochs: int = 15, learning_rate: float = 0.001,
                 agent_name: str = "TrendHunter", model_manager: Any = None):
        self.logger = logging.getLogger(f"ModelTrainer_{symbol}")
        self.symbol = symbol
        self.timeframe = timeframe
        self.input_lookback = input_lookback
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.agent_name = agent_name
        self.model_manager = model_manager
        self.batch_size = max(8, int(os.getenv("ML_TRAIN_BATCH_SIZE", "32") or 32))
        self.max_train_rows = int(os.getenv("ML_TRAIN_MAX_ROWS", "256") or 256)
        self.progress_every = max(1, int(os.getenv("ML_TRAIN_LOG_EVERY_EPOCHS", "1") or 1))
        self.early_stop_patience = max(0, int(os.getenv("ML_TRAIN_EARLY_STOP_PATIENCE", "2") or 2))
        self.early_stop_min_delta = float(os.getenv("ML_TRAIN_EARLY_STOP_MIN_DELTA", "0.0005") or 0.0005)
        self.cpu_epoch_cap = max(1, int(os.getenv("ML_TRAIN_CPU_EPOCH_CAP", "15") or 15))
        
        # New improvements
        self.label_threshold_pct = float(os.getenv("ML_TRAIN_LABEL_THRESHOLD_PCT", "0.0005") or 0.0005)
        self.use_gru = bool(os.getenv("ML_TRAIN_USE_GRU", "true").lower() == "true")
        self.gru_units_1 = max(8, int(os.getenv("ML_TRAIN_GRU_UNITS_1", "24") or 24))
        self.gru_units_2 = max(4, int(os.getenv("ML_TRAIN_GRU_UNITS_2", "12") or 12))
        self.dense_units = max(4, int(os.getenv("ML_TRAIN_DENSE_UNITS", "8") or 8))
        
        # PHASE 3: Regime-aware label thresholds
        self.regime_aware_labels_enabled = bool(os.getenv("ML_REGIME_AWARE_LABELS_ENABLED", "true").lower() == "true")
        self.label_trend_threshold_pct = float(os.getenv("ML_LABEL_TREND_THRESHOLD_PCT", "0.0020") or 0.0020)
        self.label_sideways_threshold_pct = float(os.getenv("ML_LABEL_SIDEWAYS_THRESHOLD_PCT", "0.0010") or 0.0010)
        self.label_extreme_threshold_pct = float(os.getenv("ML_LABEL_EXTREME_THRESHOLD_PCT", "0.0030") or 0.0030)
        
        # Feature scaling persistence
        self.feature_scalers = {}  # Will store sklearn scalers
        
        # Probability calibration
        self.calibration_method = os.getenv("ML_TRAIN_CALIBRATION_METHOD", "isotonic")  # isotonic or sigmoid
        self.enable_calibration = bool(
            os.getenv("ML_TRAIN_ENABLE_CALIBRATION", "false").strip().lower() in {"1", "true", "yes", "on"}
        )
        
        self.model = None
        self._last_train_metrics: Dict[str, Any] = {}

        if tf is None:
            self.logger.warning("TensorFlow not available. Training will be disabled.")

    def _build_model(self, state_shape):
        if tf is None: return None
        
        # Lightweight GRU architecture for CPU efficiency
        layers = []
        if self.use_gru:
            self.logger.info(f"Building lightweight GRU model for {self.symbol} (units: {self.gru_units_1}, {self.gru_units_2}, dense: {self.dense_units})")
            layers.append(GRU(self.gru_units_1, input_shape=state_shape, return_sequences=True, 
                             kernel_regularizer=tf.keras.regularizers.l2(0.001)))
            layers.append(Dropout(0.3))
            layers.append(GRU(self.gru_units_2, return_sequences=False,
                             kernel_regularizer=tf.keras.regularizers.l2(0.001)))
        else:
            # Fallback to lightweight LSTM if GRU disabled
            self.logger.info(f"Building lightweight LSTM model for {self.symbol} (units: {self.gru_units_1}, {self.gru_units_2}, dense: {self.dense_units})")
            layers.append(LSTM(self.gru_units_1, input_shape=state_shape, return_sequences=True,
                              kernel_regularizer=tf.keras.regularizers.l2(0.001)))
            layers.append(Dropout(0.3))
            layers.append(LSTM(self.gru_units_2, return_sequences=False,
                              kernel_regularizer=tf.keras.regularizers.l2(0.001)))
            
        layers.append(Dropout(0.3))
        layers.append(Dense(self.dense_units, activation='relu', 
                           kernel_regularizer=tf.keras.regularizers.l2(0.001)))
        layers.append(Dense(1, activation='sigmoid')) # Binary classification: Buy probability
        
        model = Sequential(layers)
        model.compile(
            loss='binary_crossentropy',
            optimizer=Adam(learning_rate=self.learning_rate, clipnorm=1.0),
            metrics=['accuracy'],
        )
        return model

    def _save_training_metadata(self, model_path: str):
        """Save training metadata including scalers and configuration."""
        import pickle

        model_path_obj = Path(model_path)
        metadata_path = model_path_obj.with_name(f"{model_path_obj.stem}_metadata.pkl")
        try:
            metadata = {
                'feature_scalers': self.feature_scalers,
                'label_threshold_pct': self.label_threshold_pct,
                'input_lookback': self.input_lookback,
                'model_version': self.timeframe,
                'use_gru': self.use_gru,
                'architecture': {
                    'gru_units_1': self.gru_units_1,
                    'gru_units_2': self.gru_units_2,
                    'dense_units': self.dense_units
                },
                'calibration_method': self.calibration_method,
                'training_config': {
                    'learning_rate': self.learning_rate,
                    'batch_size': self.batch_size,
                    'max_train_rows': self.max_train_rows,
                    'enable_calibration': self.enable_calibration,
                },
                'training_metrics': dict(self._last_train_metrics or {}),
                'model_last_trained_ts': float(time.time()),
                'model_val_accuracy': (
                    float(self._last_train_metrics.get('val_accuracy'))
                    if (self._last_train_metrics or {}).get('val_accuracy') is not None
                    else None
                ),
            }
            
            with metadata_path.open('wb') as f:
                pickle.dump(metadata, f)
                
            self.logger.info("Training metadata saved for %s at %s", self.symbol, metadata_path)
        except Exception as e:
            self.logger.warning("Failed to save training metadata for %s: %s", self.symbol, e)

    def _has_gpu(self) -> bool:
        if tf is None:
            return False
        try:
            return bool(tf.config.list_physical_devices("GPU"))
        except Exception:
            return False

    def _infer_regime_from_volatility(self, df, window: int = 20) -> str:
        """
        PHASE 3: Infer volatility regime from historical data.
        
        Uses ATR (Average True Range) relative to price to classify regime:
        - extreme: ATR/price > 2.0% (very volatile)
        - high: ATR/price 1.0-2.0%
        - medium: ATR/price 0.5-1.0%
        - low: ATR/price < 0.5%
        """
        try:
            if len(df) < window:
                return "medium"  # Default
            
            # Calculate ATR
            df_copy = df.copy()
            df_copy["tr"] = np.maximum(
                df_copy["high"] - df_copy["low"],
                np.maximum(
                    np.abs(df_copy["high"] - df_copy["close"].shift(1)),
                    np.abs(df_copy["low"] - df_copy["close"].shift(1))
                )
            )
            atr = df_copy["tr"].rolling(window=window).mean().iloc[-1]
            price = float(df_copy["close"].iloc[-1])
            
            if price <= 0:
                return "medium"
            
            atr_pct = atr / price
            
            if atr_pct > 0.02:
                return "extreme"
            elif atr_pct > 0.01:
                return "high"
            elif atr_pct > 0.005:
                return "medium"
            else:
                return "low"
        except Exception as e:
            self.logger.debug("Regime inference failed: %s", e)
            return "medium"

    @staticmethod
    def _last_history_metric(history: Dict[str, Any], keys: Union[str, tuple]) -> Optional[float]:
        key_list = [keys] if isinstance(keys, str) else list(keys)
        for key in key_list:
            vals = history.get(key)
            if not vals:
                continue
            try:
                return float(vals[-1])
            except Exception:
                continue
        return None

    def persist_model(self, model_path: Optional[Union[str, Path]] = None) -> bool:
        if self.model is None:
            self.logger.warning("No trained model to persist for %s.", self.symbol)
            return False
        try:
            resolved = Path(model_path) if model_path is not None else build_model_path(
                self.agent_name,
                self.symbol,
                self.timeframe,
            )
            save_model(self.model, resolved)
            self._save_training_metadata(str(resolved))
            saved = bool(model_exists(resolved))
            if saved:
                self.logger.info("Model and metadata saved for %s at %s", self.symbol, resolved)
            else:
                self.logger.warning("Model save check failed for %s at %s", self.symbol, resolved)
            return saved
        except Exception as e:
            self.logger.warning("Failed to persist model for %s: %s", self.symbol, e)
            return False

    def train_model(
        self,
        df,
        task: str = "supervised_learning",
        epochs: Optional[int] = None,
        max_rows: Optional[int] = None,
        save_model_artifact: bool = True,
        return_metrics: bool = False,
    ):
        """
        Main entry point to train the model on the provided DataFrame.
        This blocking call runs the training loop.
        """
        def _ret(ok: bool, reason: str, **extra):
            payload: Dict[str, Any] = {
                "ok": bool(ok),
                "reason": str(reason),
                "symbol": self.symbol,
                "timeframe": self.timeframe,
            }
            payload.update(extra)
            if return_metrics:
                return payload
            return bool(ok)

        if tf is None:
            self.logger.error("Cannot train: TensorFlow missing.")
            return _ret(False, "tensorflow_missing")
        if pd is None:
            self.logger.error("Cannot train: pandas missing.")
            return _ret(False, "pandas_missing")
        if df is None:
            self.logger.warning("Cannot train %s: dataframe is None.", self.symbol)
            return _ret(False, "data_none")
        if not isinstance(df, pd.DataFrame):
            try:
                df = pd.DataFrame(df)
            except Exception:
                self.logger.warning("Cannot train %s: unable to coerce training data to DataFrame.", self.symbol)
                return _ret(False, "invalid_dataframe")
            
        if df is None or len(df) < (self.input_lookback + 50):
            self.logger.warning(f"Insufficient data for training {self.symbol} (rows={len(df) if df is not None else 0}).")
            return _ret(False, "insufficient_rows", rows=int(len(df) if df is not None else 0))

        if task != "supervised_learning":
            self.logger.warning(f"Unsupported task: {task}")
            return _ret(False, "unsupported_task")

        effective_max_rows = int(self.max_train_rows)
        if max_rows is not None:
            try:
                effective_max_rows = int(max_rows)
            except Exception:
                self.logger.warning(
                    "Invalid max_rows override for %s (%s); using configured cap=%d.",
                    self.symbol,
                    str(max_rows),
                    int(self.max_train_rows),
                )
                effective_max_rows = int(self.max_train_rows)

        if effective_max_rows > 0 and len(df) > effective_max_rows:
            old_rows = len(df)
            df = df.tail(effective_max_rows).copy()
            self.logger.info(
                "Training rows capped for %s: %d -> %d",
                self.symbol,
                old_rows,
                len(df),
            )

        if len(df) < (self.input_lookback + 50):
            self.logger.warning(
                "Insufficient rows after cap for %s (rows=%d, need>=%d).",
                self.symbol,
                len(df),
                self.input_lookback + 50,
            )
            return _ret(False, "insufficient_rows_after_cap", rows=int(len(df)))

        epochs = int(epochs or self.epochs or 1)
        has_gpu = self._has_gpu()
        if not has_gpu:
            epochs = min(epochs, self.cpu_epoch_cap)
        device = "gpu" if has_gpu else "cpu"
        self.logger.info(
            "Starting training for %s (epochs=%d lookback=%d device=%s)...",
            self.symbol,
            epochs,
            self.input_lookback,
            device,
        )
        
        # Prepare Data features
        # If engineered features are provided, use all numeric columns (except timestamp).
        # This keeps training aligned with inference input space.
        if "close" not in df.columns:
            self.logger.error("Training DataFrame missing required 'close' column for label computation.")
            return _ret(False, "missing_close_column")

        numeric_cols = [c for c in df.columns if c != "timestamp" and pd.api.types.is_numeric_dtype(df[c])]
        if not numeric_cols:
            self.logger.error("No numeric feature columns available for training.")
            return _ret(False, "no_numeric_features")

        # Keep core OHLCV first (if present) for deterministic ordering, then engineered columns.
        core_cols = [c for c in ("open", "high", "low", "close", "volume") if c in numeric_cols]
        extra_cols = [c for c in numeric_cols if c not in core_cols]
        feature_cols = core_cols + extra_cols if core_cols else list(numeric_cols)

        raw_model_df = df[feature_cols].copy()
        raw_model_df = raw_model_df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)

        self.logger.info(
            "Training %s with %d features (lookback=%d).",
            self.symbol,
            len(feature_cols),
            self.input_lookback,
        )

        # Create labels: next close > current close by threshold -> 1 else 0
        df_copy = df.copy()
        df_copy["future_return"] = df_copy["close"].pct_change().shift(-1)
        
        # PHASE 3: Regime-aware label generation
        if self.regime_aware_labels_enabled:
            regime = self._infer_regime_from_volatility(df)
            if regime == "trend":
                threshold = self.label_trend_threshold_pct
            elif regime == "sideways":
                threshold = self.label_sideways_threshold_pct
            elif regime == "extreme":
                threshold = self.label_extreme_threshold_pct
            else:  # medium or unknown
                threshold = self.label_threshold_pct
            
            df_copy["label"] = (df_copy["future_return"] > threshold).astype(int)
            self.logger.info(
                f"Regime-aware labels: regime={regime} threshold={threshold:.6f} "
                f"positive samples: {df_copy['label'].sum()}/{len(df_copy)}"
            )
        else:
            # Fallback: use static threshold
            df_copy["label"] = (df_copy["future_return"] > self.label_threshold_pct).astype(int)
            self.logger.info(f"Label threshold: {self.label_threshold_pct:.6f}, positive samples: {df_copy['label'].sum()}/{len(df_copy)}")

        X = []
        y = []

        # Build windows that include the current bar i and predict i+1 move.
        # This keeps training aligned with live inference, which uses the latest bar.
        for i in range(self.input_lookback - 1, len(raw_model_df) - 1):
            start_idx = i - self.input_lookback + 1
            window = raw_model_df.iloc[start_idx:i + 1].values
            X.append(window)
            y.append(df_copy.iloc[i]["label"])

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if X.size == 0 or y.size == 0:
            self.logger.warning("No training samples generated for %s.", self.symbol)
            return _ret(False, "no_training_samples")
        if np.unique(y).size < 2:
            self.logger.warning("Label set is degenerate for %s; skipping retrain.", self.symbol)
            return _ret(False, "degenerate_labels")

        sample_count = int(X.shape[0])
        val_count = max(1, int(round(sample_count * 0.1)))
        min_train_count = max(16, int(self.batch_size))
        if sample_count - val_count < min_train_count:
            val_count = max(0, sample_count - min_train_count)
        has_validation = bool(sample_count >= 64 and val_count > 0)
        if has_validation:
            split_idx = sample_count - val_count
            X_train_raw, X_val_raw = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
        else:
            X_train_raw, y_train = X, y
            X_val_raw, y_val = None, None

        # Leakage fix: fit feature scalers on training split only, then transform train/val.
        self.feature_scalers = {}
        if StandardScaler is not None:
            X_train = np.empty_like(X_train_raw, dtype=np.float32)
            X_val = np.empty_like(X_val_raw, dtype=np.float32) if X_val_raw is not None else None
            for feat_idx, col in enumerate(feature_cols):
                scaler = StandardScaler()
                train_flat = X_train_raw[:, :, feat_idx].reshape(-1, 1)
                scaler.fit(train_flat)
                X_train[:, :, feat_idx] = scaler.transform(train_flat).reshape(
                    X_train_raw.shape[0], X_train_raw.shape[1]
                )
                if X_val_raw is not None and X_val is not None:
                    val_flat = X_val_raw[:, :, feat_idx].reshape(-1, 1)
                    X_val[:, :, feat_idx] = scaler.transform(val_flat).reshape(
                        X_val_raw.shape[0], X_val_raw.shape[1]
                    )
                self.feature_scalers[col] = scaler
        else:
            self.logger.warning("sklearn not available, using manual scaling (train-split fit only)")
            X_train = np.empty_like(X_train_raw, dtype=np.float32)
            X_val = np.empty_like(X_val_raw, dtype=np.float32) if X_val_raw is not None else None
            for feat_idx, col in enumerate(feature_cols):
                train_flat = X_train_raw[:, :, feat_idx].reshape(-1)
                col_mean = float(np.mean(train_flat))
                col_std = float(np.std(train_flat) + 1e-8)
                X_train[:, :, feat_idx] = ((X_train_raw[:, :, feat_idx] - col_mean) / col_std).astype(np.float32)
                if X_val_raw is not None and X_val is not None:
                    X_val[:, :, feat_idx] = ((X_val_raw[:, :, feat_idx] - col_mean) / col_std).astype(np.float32)
                self.feature_scalers[col] = {"mean": col_mean, "std": col_std}

        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        if X_val is not None:
            X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        self.model = self._build_model((self.input_lookback, len(feature_cols)))
        if self.model is None:
            return _ret(False, "model_build_failed")

        callbacks = []
        if EarlyStopping is not None and self.early_stop_patience > 0:
            callbacks.append(
                EarlyStopping(
                    monitor="val_loss" if has_validation else "loss",
                    patience=int(self.early_stop_patience),
                    min_delta=float(self.early_stop_min_delta),
                    restore_best_weights=True,
                )
            )

        trainer_logger = self.logger
        progress_every = int(self.progress_every)

        class _EpochProgress(Callback):
            def on_train_begin(self, logs=None):
                trainer_logger.info(
                    "Training progress %s: begin epochs=%d samples=%d features=%d batch=%d",
                    self.model.name if self.model else "model",
                    epochs,
                    int(X_train.shape[0]),
                    int(X_train.shape[-1]),
                    int(self.params.get("batch_size", 0) or 0),
                )

            def on_epoch_end(self, epoch, logs=None):
                ep = int(epoch) + 1
                if (ep % progress_every) != 0 and ep != epochs:
                    return
                payload = logs or {}
                trainer_logger.info(
                    "Training progress %s: epoch=%d/%d loss=%.6f val_loss=%.6f acc=%.4f val_acc=%.4f",
                    self.model.name if self.model else "model",
                    ep,
                    epochs,
                    float(payload.get("loss", 0.0) or 0.0),
                    float(payload.get("val_loss", 0.0) or 0.0),
                    float(payload.get("accuracy", payload.get("acc", 0.0)) or 0.0),
                    float(payload.get("val_accuracy", payload.get("val_acc", 0.0)) or 0.0),
                )

        callbacks.append(_EpochProgress())

        fit_kwargs = {
            "x": X_train,
            "y": y_train,
            "epochs": int(epochs),
            "batch_size": int(self.batch_size),
            "verbose": 0,
            "callbacks": callbacks,
            "shuffle": False,
        }
        if has_validation and X_val is not None and y_val is not None:
            fit_kwargs["validation_data"] = (X_val, y_val)

        # Add class imbalance weighting
        unique_labels, counts = np.unique(y_train, return_counts=True)
        total_samples = len(y_train)
        class_weights = {}
        for label, count in zip(unique_labels, counts):
            class_weights[int(label)] = total_samples / (len(unique_labels) * count)
        
        if len(class_weights) > 1:
            fit_kwargs["class_weight"] = class_weights
            self.logger.info("Applied class weights for %s: %s", self.symbol, 
                           {k: f"{v:.2f}" for k, v in class_weights.items()})

        history = self.model.fit(**fit_kwargs)
        history_map = history.history if hasattr(history, "history") else {}

        # Apply probability calibration
        if self.enable_calibration and CalibratedClassifierCV is not None:
            self.logger.warning(
                "ML_TRAIN_ENABLE_CALIBRATION=true for %s but sklearn calibrator is disabled for sequence models; skipping.",
                self.symbol,
            )

        final_loss = self._last_history_metric(history_map, "loss")
        final_accuracy = self._last_history_metric(history_map, ("accuracy", "acc"))
        final_val_loss = self._last_history_metric(history_map, "val_loss")
        final_val_accuracy = self._last_history_metric(history_map, ("val_accuracy", "val_acc"))

        self._last_train_metrics = {
            "loss": final_loss,
            "accuracy": final_accuracy,
            "val_loss": final_val_loss,
            "val_accuracy": final_val_accuracy,
            "epochs": int(epochs),
            "rows": int(len(df)),
            "samples_total": int(sample_count),
            "train_samples": int(X_train.shape[0]),
            "val_samples": int(X_val.shape[0]) if X_val is not None else 0,
        }

        saved = False
        resolved_model_path = build_model_path(self.agent_name, self.symbol, self.timeframe)
        if save_model_artifact:
            saved = self.persist_model(model_path=resolved_model_path)
        else:
            self.logger.info("Skipping model persistence for %s (save_model_artifact=False).", self.symbol)

        return _ret(
            True,
            "trained",
            rows=int(len(df)),
            samples_total=int(sample_count),
            train_samples=int(X_train.shape[0]),
            val_samples=int(X_val.shape[0]) if X_val is not None else 0,
            epochs=int(epochs),
            loss=final_loss,
            accuracy=final_accuracy,
            val_loss=final_val_loss,
            val_accuracy=final_val_accuracy,
            used_validation=bool(has_validation),
            model_path=str(resolved_model_path),
            saved=bool(saved),
            max_rows=int(effective_max_rows),
        )
