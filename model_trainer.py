import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Input # Import Input layer
from keras.callbacks import EarlyStopping # Import EarlyStopping
import logging

logger = logging.getLogger("ModelTrainer")

class ModelTrainer:
    def __init__(self, symbol, timeframe, agent_name="MLForecaster", model_manager=None):
        self.symbol = symbol
        self.timeframe = timeframe
        self.agent_name = agent_name
        self.model_manager = model_manager  # ✅ Optional dependency
        self.model = None
        self.timesteps = 50
        self.features = None  # Will be inferred from engineered feature matrix

    def build_model(self):
        """
        Builds an LSTM classifier predicting probability of positive net move.
        """
        if self.features is None:
            raise ValueError("Features must be set before building the model.")

        self.model = Sequential([
            Input(shape=(self.timesteps, self.features)),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1, activation="sigmoid"),  # Probability output
        ])

        self.model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        return self.model

    def train_model(self, df: pd.DataFrame):
        """
        Trains the model using the provided DataFrame.
        Adds EarlyStopping with validation_split.
        Handles model saving via ModelManager if available.
        """
        if df.empty:
            logger.warning(f"No data to train for {self.symbol}.")
            return

        if "target" not in df.columns:
            logger.error("DataFrame must contain a 'target' column (0/1 classification).")
            return

        feature_cols = [c for c in df.columns if c != "target"]
        data = df[feature_cols].values
        targets = df["target"].values

        self.features = len(feature_cols)

        X, y = [], []
        for i in range(len(data) - self.timesteps):
            X.append(data[i:i + self.timesteps])
            y.append(targets[i + self.timesteps])

        if not X:
            logger.warning(
                f"Not enough data to create sequences for {self.symbol} with timesteps {self.timesteps}."
            )
            return

        X = np.array(X)
        y = np.array(y)

        # Build model if not already built
        if self.model is None:
            self.model = self.build_model()

        # Define callbacks with EarlyStopping
        callbacks = [EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)]

        # Chronological train/validation split (no leakage)
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        if len(X_val) == 0:
            logger.warning(f"Not enough samples for validation split on {self.symbol}. Skipping training.")
            return

        epochs = 50
        batch_size = 32

        logger.info(
            f"Starting model training for {self.symbol} | "
            f"train_samples={len(X_train)} val_samples={len(X_val)} "
            f"features={self.features} timesteps={self.timesteps}"
        )

        history = self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            shuffle=False,
            verbose=0
        )

        logger.info(f"Model training for {self.symbol} finished. Final loss: {history.history['loss'][-1]:.4f}")
        if 'val_loss' in history.history:
             logger.info(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")

        # Save model via ModelManager (if provided)
        if self.model_manager: # ✅ Save model via ModelManager
            model_path = self.model_manager.build_model_path(
                agent_name=self.agent_name,
                symbol=self.symbol,
                version=self.timeframe
            )
            self.model_manager.save_model(self.model, model_path)
            logger.info(f"Model saved for {self.symbol} at {model_path}")

# Note: This file (core/model_trainer.py) assumes 'ModelManager' and other core components
# are properly defined and importable in your project structure.
