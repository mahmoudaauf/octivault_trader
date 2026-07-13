import logging
import os
import platform
import time
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

try:
    import tensorflow as tf
    from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam

    try:
        from tensorflow.keras.optimizers.legacy import Adam as _LegacyAdamCls
        # Keras 3 raises ImportError at instantiation time, not import time — test it
        _LegacyAdamCls(learning_rate=0.001)
        LegacyAdam = _LegacyAdamCls
    except Exception:
        LegacyAdam = None
    from tensorflow.keras.callbacks import Callback, EarlyStopping
except ImportError:
    tf = None
    EarlyStopping = None
    Callback = object
    LegacyAdam = None

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.utils.class_weight import compute_class_weight
except ImportError:
    StandardScaler = None
    CalibratedClassifierCV = None
    compute_class_weight = None

# Using ModelManager's helper to build paths if needed,
# or we can redefine it locally to keep it standalone.
# We'll rely on the caller to handle paths or import from model_manager.
from src.l5_strategy.model_manager import build_model_path, model_exists, save_model


class ModelTrainer:
    """
    Supervised Learning Trainer for binary classification.
    Trains a model to predict BUY (1) or HOLD/SELL (0) actions based on market data states.
    """

    def __init__(
        self,
        symbol: str,
        timeframe: str = "5m",
        input_lookback: int = 20,
        epochs: int = 15,
        learning_rate: float = 0.001,
        agent_name: str = "TrendHunter",
        model_manager: Any = None,
    ):
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
        self.early_stop_min_delta = float(
            os.getenv("ML_TRAIN_EARLY_STOP_MIN_DELTA", "0.0005") or 0.0005
        )
        self.cpu_epoch_cap = max(1, int(os.getenv("ML_TRAIN_CPU_EPOCH_CAP", "15") or 15))

        # New improvements
        self.label_threshold_pct = float(
            os.getenv("ML_TRAIN_LABEL_THRESHOLD_PCT", "0.0005") or 0.0005
        )
        self.use_gru = bool(os.getenv("ML_TRAIN_USE_GRU", "true").lower() == "true")
        self.gru_units_1 = max(8, int(os.getenv("ML_TRAIN_GRU_UNITS_1", "24") or 24))
        self.gru_units_2 = max(4, int(os.getenv("ML_TRAIN_GRU_UNITS_2", "12") or 12))
        self.dense_units = max(4, int(os.getenv("ML_TRAIN_DENSE_UNITS", "8") or 8))

        # PHASE 3: Regime-aware label thresholds
        self.regime_aware_labels_enabled = bool(
            os.getenv("ML_REGIME_AWARE_LABELS_ENABLED", "true").lower() == "true"
        )
        self.label_trend_threshold_pct = float(
            os.getenv("ML_LABEL_TREND_THRESHOLD_PCT", "0.0020") or 0.0020
        )
        self.label_sideways_threshold_pct = float(
            os.getenv("ML_LABEL_SIDEWAYS_THRESHOLD_PCT", "0.0010") or 0.0010
        )
        self.label_extreme_threshold_pct = float(
            os.getenv("ML_LABEL_EXTREME_THRESHOLD_PCT", "0.0030") or 0.0030
        )

        # IMPROVED LABELING: Triple Barrier Method
        self.use_triple_barrier_labels = bool(
            os.getenv("ML_USE_TRIPLE_BARRIER_LABELS", "true").lower() == "true"
        )
        self.triple_barrier_fee_pct = float(
            os.getenv("ML_TRIPLE_BARRIER_FEE_PCT", "0.001") or 0.001
        )
        self.triple_barrier_slippage_pct = float(
            os.getenv("ML_TRIPLE_BARRIER_SLIPPAGE_PCT", "0.0005") or 0.0005
        )
        self.triple_barrier_buffer_pct = float(
            os.getenv("ML_TRIPLE_BARRIER_BUFFER_PCT", "0.0005") or 0.0005
        )
        self.triple_barrier_lookforward = max(
            1, int(os.getenv("ML_TRIPLE_BARRIER_LOOKFORWARD_BARS", "5") or 5)
        )
        # Tier-2 trend-aware barriers: the label now requires the UP target to be hit
        # BEFORE a stop (path-dependent), so transient up-wiggles inside a downtrend are
        # no longer labelled BUY. Stop barrier = stop_mult × the profit threshold. An
        # optional trend filter further refuses BUY labels when price is below a long MA.
        self.triple_barrier_stop_mult = float(
            os.getenv("ML_TRIPLE_BARRIER_STOP_MULT", "1.0") or 1.0
        )
        self.triple_barrier_trend_filter = (
            os.getenv("ML_TRIPLE_BARRIER_TREND_FILTER", "true").lower() == "true"
        )
        self.triple_barrier_trend_ma = max(
            2, int(os.getenv("ML_TRIPLE_BARRIER_TREND_MA", "50") or 50)
        )

        # Feature scaling persistence
        self.feature_scalers = {}  # Will store sklearn scalers
        self.feature_columns: list[str] = []

        # Probability calibration
        self.calibration_method = os.getenv(
            "ML_TRAIN_CALIBRATION_METHOD", "isotonic"
        )  # isotonic or sigmoid
        self.enable_calibration = bool(
            os.getenv("ML_TRAIN_ENABLE_CALIBRATION", "false").strip().lower()
            in {"1", "true", "yes", "on"}
        )

        self.model = None
        self._last_train_metrics: dict[str, Any] = {}

        if tf is None:
            self.logger.warning("TensorFlow not available. Training will be disabled.")

    def _build_optimizer(self):
        """
        Build optimizer with a sensible Apple Silicon default.
        On M-series macOS, legacy Adam is materially faster than v2 optimizer.
        Override with ML_TRAIN_USE_LEGACY_ADAM=true/false (default: auto).
        """
        use_legacy = False
        if LegacyAdam is not None:
            pref = str(os.getenv("ML_TRAIN_USE_LEGACY_ADAM", "auto") or "auto").strip().lower()
            if pref in {"1", "true", "yes", "on"}:
                use_legacy = True
            elif pref in {"0", "false", "no", "off"}:
                use_legacy = False
            else:
                is_apple_arm = (
                    platform.system().lower() == "darwin"
                    and platform.machine().lower() in {"arm64", "aarch64"}
                )
                use_legacy = bool(is_apple_arm)

        if use_legacy:
            self.logger.info(
                "Using legacy Adam optimizer for improved Apple Silicon training performance."
            )
            try:
                return LegacyAdam(learning_rate=self.learning_rate, clipnorm=1.0)
            except Exception as exc:
                self.logger.warning(
                    "Legacy Adam unavailable at runtime (%s); falling back to standard Adam.",
                    exc,
                )
        return Adam(learning_rate=self.learning_rate, clipnorm=1.0)

    def _build_model(self, state_shape):
        if tf is None:
            return None

        # Lightweight GRU architecture for CPU efficiency
        layers = []
        if self.use_gru:
            self.logger.info(
                f"Building lightweight GRU model for {self.symbol} (units: {self.gru_units_1}, {self.gru_units_2}, dense: {self.dense_units})"
            )
            layers.append(
                GRU(
                    self.gru_units_1,
                    input_shape=state_shape,
                    return_sequences=True,
                    kernel_regularizer=tf.keras.regularizers.l2(0.001),
                )
            )
            layers.append(Dropout(0.3))
            layers.append(
                GRU(
                    self.gru_units_2,
                    return_sequences=False,
                    kernel_regularizer=tf.keras.regularizers.l2(0.001),
                )
            )
        else:
            # Fallback to lightweight LSTM if GRU disabled
            self.logger.info(
                f"Building lightweight LSTM model for {self.symbol} (units: {self.gru_units_1}, {self.gru_units_2}, dense: {self.dense_units})"
            )
            layers.append(
                LSTM(
                    self.gru_units_1,
                    input_shape=state_shape,
                    return_sequences=True,
                    kernel_regularizer=tf.keras.regularizers.l2(0.001),
                )
            )
            layers.append(Dropout(0.3))
            layers.append(
                LSTM(
                    self.gru_units_2,
                    return_sequences=False,
                    kernel_regularizer=tf.keras.regularizers.l2(0.001),
                )
            )

        layers.append(Dropout(0.3))
        layers.append(
            Dense(
                self.dense_units,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(0.001),
            )
        )
        layers.append(Dense(1, activation="sigmoid"))  # Binary classification: Buy probability

        model = Sequential(layers)
        model.compile(
            loss="binary_crossentropy",
            optimizer=self._build_optimizer(),
            metrics=["accuracy"],
        )
        return model

    def _save_training_metadata(self, model_path: str):
        """Save training metadata including scalers and configuration."""
        import pickle

        model_path_obj = Path(model_path)
        metadata_path = model_path_obj.with_name(f"{model_path_obj.stem}_metadata.pkl")
        try:
            metadata = {
                "feature_scalers": self.feature_scalers,
                "feature_columns": list(self.feature_columns),
                "label_threshold_pct": self.label_threshold_pct,
                "input_lookback": self.input_lookback,
                "model_version": self.timeframe,
                "use_gru": self.use_gru,
                "architecture": {
                    "gru_units_1": self.gru_units_1,
                    "gru_units_2": self.gru_units_2,
                    "dense_units": self.dense_units,
                },
                "calibration_method": self.calibration_method,
                "training_config": {
                    "learning_rate": self.learning_rate,
                    "batch_size": self.batch_size,
                    "max_train_rows": self.max_train_rows,
                    "enable_calibration": self.enable_calibration,
                },
                "training_metrics": dict(self._last_train_metrics or {}),
                "model_last_trained_ts": float(time.time()),
                "model_val_accuracy": (
                    float(self._last_train_metrics.get("val_accuracy"))
                    if (self._last_train_metrics or {}).get("val_accuracy") is not None
                    else None
                ),
            }

            with metadata_path.open("wb") as f:
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
                    np.abs(df_copy["low"] - df_copy["close"].shift(1)),
                ),
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

    def _create_labels_triple_barrier(
        self,
        df,
        fee_pct: float = 0.001,
        slippage_pct: float = 0.0005,
        buffer_pct: float = 0.0005,
        lookforward_bars: int = 5,
        volatility_window: int = 20,
    ) -> np.ndarray:
        """
        IMPROVED LABELING: Triple Barrier Method (Real Quant Standard)

        Creates labels based on:
        1. Forward return > (fees + slippage + buffer) AND within N bars
        2. Volatility-normalized thresholds
        3. Realistic transaction costs

        Returns:
            np.ndarray: Binary labels (1=BUY profitable, 0=HOLD/NO trade)
        """
        try:
            df_copy = df.copy()

            # Calculate volatility (ATR-based)
            df_copy["tr"] = np.maximum(
                df_copy["high"] - df_copy["low"],
                np.maximum(
                    np.abs(df_copy["high"] - df_copy["close"].shift(1)),
                    np.abs(df_copy["low"] - df_copy["close"].shift(1)),
                ),
            )
            df_copy["atr"] = df_copy["tr"].rolling(window=volatility_window).mean()
            df_copy["volatility"] = df_copy["atr"] / df_copy["close"]

            # Cost threshold: fees + slippage + buffer (normalized)
            cost_threshold = fee_pct + slippage_pct + buffer_pct

            # Extract numpy arrays once (path-dependent barrier walk needs per-bar OHLC).
            close = df_copy["close"].to_numpy(dtype=np.float64)
            high = df_copy["high"].to_numpy(dtype=np.float64)
            low = df_copy["low"].to_numpy(dtype=np.float64)
            vol = df_copy["volatility"].to_numpy(dtype=np.float64)
            stop_mult = float(self.triple_barrier_stop_mult)
            use_trend = bool(self.triple_barrier_trend_filter)
            trend_ma = (
                df_copy["close"].rolling(window=self.triple_barrier_trend_ma).mean().to_numpy(dtype=np.float64)
                if use_trend
                else None
            )

            labels = np.zeros(len(df_copy), dtype=np.float32)
            n_rows = len(df_copy)

            for i in range(n_rows - lookforward_bars):
                current_price = close[i]
                current_vol = vol[i]
                if current_price <= 0 or np.isnan(current_vol):
                    labels[i] = 0
                    continue

                # Trend filter: don't label BUY when price is below the long MA (downtrend).
                if use_trend and trend_ma is not None:
                    ma = trend_ma[i]
                    if not np.isnan(ma) and current_price < ma:
                        labels[i] = 0
                        continue

                # Barriers: profit target (up) and stop (down), vol-adjusted.
                profit_threshold = cost_threshold + (current_vol * 0.5)
                stop_threshold = -profit_threshold * stop_mult

                # Path-dependent walk: whichever barrier is touched FIRST decides the
                # label. Within a bar we check the stop first (pessimistic — we cannot
                # know intra-bar order), so a falling-then-bouncing bar is NOT a BUY.
                label = 0
                for j in range(i + 1, min(i + lookforward_bars + 1, n_rows)):
                    ret_low = (low[j] - current_price) / current_price
                    if ret_low <= stop_threshold:
                        label = 0  # stop hit first → not a profitable buy
                        break
                    ret_high = (high[j] - current_price) / current_price
                    if ret_high >= profit_threshold:
                        label = 1  # profit hit before stop → BUY
                        break
                labels[i] = label

            # Log distribution
            unique, counts = np.unique(labels, return_counts=True)
            label_dist = dict(zip(unique.astype(int).tolist(), counts.tolist()))
            self.logger.info(
                f"[ML DEBUG] Triple Barrier (path-dependent) Labels: fee={fee_pct:.4f} "
                f"slippage={slippage_pct:.4f} buffer={buffer_pct:.4f} stop_mult={stop_mult:.2f} "
                f"trend_filter={use_trend} lookforward={lookforward_bars} dist={label_dist}"
            )

            return labels
        except Exception as e:
            self.logger.warning(
                f"Triple Barrier labeling failed: {e}, falling back to simple threshold"
            )
            return None

    @staticmethod
    def _last_history_metric(history: dict[str, Any], keys: Union[str, tuple]) -> Optional[float]:
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
            resolved = (
                Path(model_path)
                if model_path is not None
                else build_model_path(
                    self.agent_name,
                    self.symbol,
                    self.timeframe,
                )
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

    def evaluate_holdout(
        self,
        df,
        *,
        probability_threshold: float = 0.55,
        round_trip_cost_pct: float = 0.0030,
        horizon_bars: Optional[int] = None,
    ) -> dict[str, Any]:
        """Evaluate the trained model without fitting on held-out observations.

        A signal is entered at the prediction bar's close and exited at the
        configured horizon close. Net returns include the complete round-trip
        fee/slippage assumption. Existing training scalers are transformed only;
        they are never refit on holdout data.
        """
        result: dict[str, Any] = {
            "ok": False,
            "reason": "unknown",
            "samples": 0,
            "signals": 0,
            "wins": 0,
            "accuracy": 0.0,
            "win_rate": 0.0,
            "expectancy_pct": 0.0,
            "profit_factor": 0.0,
            "winners_per_day": 0.0,
            "round_trip_cost_pct": float(round_trip_cost_pct),
        }
        if pd is None:
            result["reason"] = "pandas_missing"
            return result
        if self.model is None:
            result["reason"] = "model_missing"
            return result
        if not isinstance(df, pd.DataFrame):
            try:
                df = pd.DataFrame(df)
            except Exception:
                result["reason"] = "invalid_dataframe"
                return result

        feature_cols = list(self.feature_columns or self.feature_scalers.keys())
        if not feature_cols:
            result["reason"] = "feature_schema_missing"
            return result
        missing = [col for col in feature_cols if col not in df.columns]
        if missing:
            result["reason"] = f"missing_features:{','.join(missing[:5])}"
            return result
        if "close" not in df.columns:
            result["reason"] = "missing_close_column"
            return result

        horizon = max(1, int(horizon_bars or self.triple_barrier_lookforward))
        if len(df) < self.input_lookback + horizon:
            result["reason"] = "insufficient_rows"
            return result

        raw = (
            df[feature_cols]
            .replace([np.inf, -np.inf], np.nan)
            .ffill()
            .bfill()
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
        )
        scaled = np.empty_like(raw, dtype=np.float32)
        for feat_idx, col in enumerate(feature_cols):
            scaler = self.feature_scalers.get(col)
            if scaler is None:
                result["reason"] = f"scaler_missing:{col}"
                return result
            values = raw[:, feat_idx].reshape(-1, 1)
            if hasattr(scaler, "transform"):
                transformed = scaler.transform(values).reshape(-1)
            elif isinstance(scaler, dict) and "mean" in scaler and "std" in scaler:
                transformed = (values.reshape(-1) - float(scaler["mean"])) / max(
                    float(scaler["std"]), 1e-8
                )
            else:
                result["reason"] = f"invalid_scaler:{col}"
                return result
            scaled[:, feat_idx] = transformed
        scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)

        end_indices = list(range(self.input_lookback - 1, len(df) - horizon))
        windows = np.asarray(
            [scaled[i - self.input_lookback + 1 : i + 1] for i in end_indices],
            dtype=np.float32,
        )
        try:
            predictions = self.model.predict(windows, verbose=0)
        except TypeError:
            predictions = self.model.predict(windows)
        probabilities = np.asarray(predictions, dtype=np.float64).reshape(-1)
        if len(probabilities) != len(end_indices):
            result["reason"] = "prediction_shape_mismatch"
            return result

        labels = self._create_labels_triple_barrier(
            df,
            fee_pct=self.triple_barrier_fee_pct,
            slippage_pct=self.triple_barrier_slippage_pct,
            buffer_pct=self.triple_barrier_buffer_pct,
            lookforward_bars=horizon,
        )
        if labels is None:
            result["reason"] = "label_generation_failed"
            return result

        closes = df["close"].to_numpy(dtype=np.float64)
        predicted_buy = probabilities >= float(probability_threshold)
        y_true = np.asarray([labels[i] for i in end_indices], dtype=np.int8)
        accuracy = float(np.mean(predicted_buy.astype(np.int8) == y_true))
        net_returns = np.asarray(
            [
                (closes[i + horizon] / closes[i]) - 1.0 - float(round_trip_cost_pct)
                for i, buy in zip(end_indices, predicted_buy)
                if buy and closes[i] > 0.0
            ],
            dtype=np.float64,
        )
        wins = net_returns[net_returns > 0.0]
        losses = net_returns[net_returns < 0.0]
        gross_profit = float(wins.sum())
        gross_loss = abs(float(losses.sum()))
        profit_factor = gross_profit / gross_loss if gross_loss > 0.0 else (
            float("inf") if gross_profit > 0.0 else 0.0
        )

        if isinstance(df.index, pd.DatetimeIndex) and len(df.index):
            span_days = max(1, int((df.index[-1].normalize() - df.index[0].normalize()).days) + 1)
        else:
            span_days = max(1, int(np.ceil(len(df) / 288.0)))
        result.update(
            ok=True,
            reason="evaluated",
            samples=int(len(end_indices)),
            signals=int(len(net_returns)),
            wins=int(len(wins)),
            accuracy=accuracy,
            win_rate=float(len(wins) / len(net_returns)) if len(net_returns) else 0.0,
            expectancy_pct=float(net_returns.mean() * 100.0) if len(net_returns) else 0.0,
            profit_factor=float(profit_factor),
            winners_per_day=float(len(wins) / span_days),
            probability_threshold=float(probability_threshold),
            horizon_bars=int(horizon),
            calendar_days=int(span_days),
        )
        return result

    def train_model(
        self,
        df,
        task: str = "supervised_learning",
        epochs: Optional[int] = None,
        max_rows: Optional[int] = None,
        save_model_artifact: bool = True,
        return_metrics: bool = False,
        fear_greed_score: Optional[int] = None,
    ):
        """
        Main entry point to train the model on the provided DataFrame.
        This blocking call runs the training loop.

        Args:
            fear_greed_score: Current Fear & Greed Index (0-100). When <25 (Extreme Fear),
                label thresholds are raised to require bigger moves for BUY labels,
                and recent candles are upsampled to emphasize fear-period patterns.
        """

        def _ret(ok: bool, reason: str, **extra):
            payload: dict[str, Any] = {
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
                self.logger.warning(
                    "Cannot train %s: unable to coerce training data to DataFrame.", self.symbol
                )
                return _ret(False, "invalid_dataframe")

        if df is None or len(df) < (self.input_lookback + 50):
            self.logger.warning(
                f"Insufficient data for training {self.symbol} (rows={len(df) if df is not None else 0})."
            )
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
            self.logger.error(
                "Training DataFrame missing required 'close' column for label computation."
            )
            return _ret(False, "missing_close_column")

        numeric_cols = [
            c for c in df.columns if c != "timestamp" and pd.api.types.is_numeric_dtype(df[c])
        ]
        if not numeric_cols:
            self.logger.error("No numeric feature columns available for training.")
            return _ret(False, "no_numeric_features")

        # Keep core OHLCV first (if present) for deterministic ordering, then engineered columns.
        core_cols = [c for c in ("open", "high", "low", "close", "volume") if c in numeric_cols]
        extra_cols = [c for c in numeric_cols if c not in core_cols]
        feature_cols = core_cols + extra_cols if core_cols else list(numeric_cols)
        self.feature_columns = list(feature_cols)

        raw_model_df = df[feature_cols].copy()
        raw_model_df = raw_model_df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)

        self.logger.info(
            "Training %s with %d features (lookback=%d).",
            self.symbol,
            len(feature_cols),
            self.input_lookback,
        )

        # Create labels using improved Triple Barrier Method or fallback to simple threshold
        df_copy = df.copy()
        df_copy["future_return"] = df_copy["close"].pct_change().shift(-1)

        # Fear & Greed regime awareness
        # Extreme fear: require bigger moves to label as BUY (filters out noise during panic).
        # Normal market: standard thresholds apply.
        _fg = int(fear_greed_score) if fear_greed_score is not None else 50
        _fg_normalised = (_fg - 50.0) / 50.0  # maps 0→-1.0, 50→0.0, 100→+1.0
        if _fg < 25:
            _fg_regime = "EXTREME_FEAR"
            _fg_label_multiplier = 2.5   # need 2.5× bigger move to confirm BUY in extreme fear
            _fg_upsample_recent = 3      # recent candles weighted 3× during extreme fear
        elif _fg < 45:
            _fg_regime = "FEAR"
            _fg_label_multiplier = 1.5
            _fg_upsample_recent = 2
        else:
            _fg_regime = "NORMAL"
            _fg_label_multiplier = 1.0
            _fg_upsample_recent = 1
        if _fg_label_multiplier != 1.0:
            self.triple_barrier_fee_pct = self.triple_barrier_fee_pct * _fg_label_multiplier
            self.triple_barrier_slippage_pct = self.triple_barrier_slippage_pct * _fg_label_multiplier
            self.triple_barrier_buffer_pct = self.triple_barrier_buffer_pct * _fg_label_multiplier
            self.label_threshold_pct = self.label_threshold_pct * _fg_label_multiplier
            self.label_trend_threshold_pct = self.label_trend_threshold_pct * _fg_label_multiplier
            self.logger.info(
                "[FG-AWARE] %s F&G=%d (%s) — label threshold ×%.1f (need bigger move to confirm BUY)",
                self.symbol, _fg, _fg_regime, _fg_label_multiplier,
            )

        # Add F&G as a constant feature so model learns sentiment context
        df_copy["fear_greed_norm"] = float(_fg_normalised)
        _ = "fear_greed_norm" not in raw_model_df.columns  # unused; FG kept for label-weighting only

        # IMPROVED LABELING: Try Triple Barrier first
        if self.use_triple_barrier_labels:
            triple_barrier_labels = self._create_labels_triple_barrier(
                df_copy,
                fee_pct=self.triple_barrier_fee_pct,
                slippage_pct=self.triple_barrier_slippage_pct,
                buffer_pct=self.triple_barrier_buffer_pct,
                lookforward_bars=self.triple_barrier_lookforward,
            )
            if triple_barrier_labels is not None:
                df_copy["label"] = triple_barrier_labels
                self.logger.info("Using Triple Barrier Labeling (improved method)")
            else:
                # Fallback to regime-aware
                self.logger.warning("Triple Barrier failed, falling back to regime-aware labels")
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
                    df_copy["label"] = (df_copy["future_return"] > self.label_threshold_pct).astype(
                        int
                    )
                    self.logger.info(
                        f"Label threshold: {self.label_threshold_pct:.6f}, positive samples: {df_copy['label'].sum()}/{len(df_copy)}"
                    )
        # PHASE 3: Regime-aware label generation (fallback)
        elif self.regime_aware_labels_enabled:
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
            # Simple threshold fallback
            df_copy["label"] = (df_copy["future_return"] > self.label_threshold_pct).astype(int)
            self.logger.info(
                f"Label threshold: {self.label_threshold_pct:.6f}, positive samples: {df_copy['label'].sum()}/{len(df_copy)}"
            )

        # NOTE: F&G is used for label-weighting (above) but NOT injected into model input
        # to keep feature dimensions consistent with live inference (which expects len(feature_cols)).

        X = []
        y = []
        sample_weights = []  # for recent-candle upsampling

        # Build windows that include the current bar i and predict i+1 move.
        # This keeps training aligned with live inference, which uses the latest bar.
        total_bars = len(raw_model_df) - 1
        recent_start = max(0, total_bars - int(total_bars * 0.3))  # last 30% = "recent"
        for i in range(self.input_lookback - 1, total_bars):
            start_idx = i - self.input_lookback + 1
            window = raw_model_df.iloc[start_idx : i + 1].values
            X.append(window)
            y.append(df_copy.iloc[i]["label"])
            # Upsample recent candles during fear — model should weight current conditions more
            sample_weights.append(_fg_upsample_recent if i >= recent_start else 1)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        sample_weights = np.asarray(sample_weights, dtype=np.float32)
        if X.size == 0 or y.size == 0:
            self.logger.warning("No training samples generated for %s.", self.symbol)
            return _ret(False, "no_training_samples")
        if np.unique(y).size < 2:
            # Bootstrap a balanced target from forward-return ordering. This keeps
            # training numerically possible when a fixed threshold produces only
            # one class. The untouched cost-adjusted holdout gate still prevents a
            # model trained on weak/flat moves from being deployed.
            sample_returns = (
                df_copy["future_return"]
                .iloc[self.input_lookback - 1 : total_bars]
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
                .to_numpy(dtype=np.float64)
            )
            median_return = float(np.median(sample_returns))
            fallback_y = (sample_returns > median_return).astype(np.float32)
            method = "median"
            if np.unique(fallback_y).size < 2:
                order = np.argsort(sample_returns, kind="stable")
                fallback_y = np.zeros(len(sample_returns), dtype=np.float32)
                fallback_y[order[len(order) // 2 :]] = 1.0
                method = "rank"
            y = fallback_y
            self.logger.warning(
                "Degenerate labels for %s; using %s forward-return split for training only.",
                self.symbol,
                method,
            )

        # === DEBUG LABEL DISTRIBUTION ===
        unique, counts = np.unique(y, return_counts=True)
        label_dist = dict(zip(unique.astype(int).tolist(), counts.tolist()))
        self.logger.info(f"[ML DEBUG] Label distribution for {self.symbol}: {label_dist}")
        # ================================

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

        # DEBUG validation distribution
        if has_validation and y_val is not None:
            unique_val, counts_val = np.unique(y_val, return_counts=True)
            val_dist = dict(zip(unique_val.astype(int).tolist(), counts_val.tolist()))
            self.logger.info(f"[ML DEBUG] Validation distribution for {self.symbol}: {val_dist}")

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
            self.logger.warning(
                "sklearn not available, using manual scaling (train-split fit only)"
            )
            X_train = np.empty_like(X_train_raw, dtype=np.float32)
            X_val = np.empty_like(X_val_raw, dtype=np.float32) if X_val_raw is not None else None
            for feat_idx, col in enumerate(feature_cols):
                train_flat = X_train_raw[:, :, feat_idx].reshape(-1)
                col_mean = float(np.mean(train_flat))
                col_std = float(np.std(train_flat) + 1e-8)
                X_train[:, :, feat_idx] = (
                    (X_train_raw[:, :, feat_idx] - col_mean) / col_std
                ).astype(np.float32)
                if X_val_raw is not None and X_val is not None:
                    X_val[:, :, feat_idx] = (
                        (X_val_raw[:, :, feat_idx] - col_mean) / col_std
                    ).astype(np.float32)
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

        # Build per-sample weights combining class balance + fear-regime upsampling.
        # We use sample_weight instead of class_weight so both can be applied together.
        unique_labels = np.unique(y_train)
        if compute_class_weight is not None:
            cw_arr = compute_class_weight(class_weight="balanced", classes=unique_labels, y=y_train)
            class_weights = dict(zip(unique_labels.astype(int), cw_arr))
        else:
            unique_labels, counts = np.unique(y_train, return_counts=True)
            total_samples = len(y_train)
            class_weights = {int(l): total_samples / (len(unique_labels) * c)
                             for l, c in zip(unique_labels, counts)}

        # Combine: per-sample weight = class_weight × fear_upsample_weight
        train_sample_weights = sample_weights[:len(y_train)].copy()
        for idx, label in enumerate(y_train):
            train_sample_weights[idx] *= class_weights.get(int(label), 1.0)
        # Normalize so total weight equals number of samples
        _w_sum = train_sample_weights.sum()
        if _w_sum > 0:
            train_sample_weights = train_sample_weights / _w_sum * len(y_train)
        fit_kwargs["sample_weight"] = train_sample_weights
        self.logger.info(
            "Applied combined weights for %s: class=%s fear_upsample=×%d (F&G=%d %s)",
            self.symbol,
            {k: f"{v:.2f}" for k, v in class_weights.items()},
            _fg_upsample_recent, _fg, _fg_regime,
        )

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
            self.logger.info(
                "Skipping model persistence for %s (save_model_artifact=False).", self.symbol
            )

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
