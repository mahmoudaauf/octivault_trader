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
        self.timesteps = 50 # Example: define timesteps
        self.features = 1   # Example: define features (e.g., if only 'close' price is used)

    def build_model(self):
        """
        Builds a Sequential Keras model with LSTM layers.
        Uses Input layer to correctly handle RNN input shape.
        """
        self.model = Sequential([
            Input(shape=(self.timesteps, self.features)), # ✅ Use Input layer
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dense(1, activation="linear"), # Assuming output_dim is 1 for regression
        ])
        self.model.compile(optimizer="adam", loss="mse", metrics=["loss"]) # for regression
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

        # Prepare data for LSTM: X (features), y (target)
        # This is a simplified example; you'll need to adapt it to your actual data prep
        # For instance, if 'df' contains OHLCV, you might use 'close' price for prediction
        # Example: Using 'close' price as feature and next 'close' as target
        # This needs to be adjusted based on your actual feature engineering and target definition.

        # Example: Simple windowing for time series (replace with your actual logic)
        # Ensure df has a numerical column like 'Close'
        if 'Close' not in df.columns:
            logger.error("DataFrame must contain a 'Close' column for this example.")
            return

        data = df['Close'].values.reshape(-1, 1) # Assuming 'Close' as the feature
        
        X, y = [], []
        for i in range(len(data) - self.timesteps):
            X.append(data[i:i + self.timesteps])
            y.append(data[i + self.timesteps]) # Predicting the next value

        if not X:
            logger.warning(f"Not enough data to create sequences for {self.symbol} with timesteps {self.timesteps}.")
            return

        X = np.array(X)
        y = np.array(y)

        # Build model if not already built
        if self.model is None:
            self.model = self.build_model()

        # Define callbacks with EarlyStopping
        callbacks = [EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)] # ✅ EarlyStopping

        # Train the model
        epochs = 50 # Example: Set number of epochs
        batch_size = 32 # Example: Set batch size
        
        logger.info(f"Starting model training for {self.symbol} with {len(X)} samples.")
        history = self.model.fit(X, y, epochs=epochs, batch_size=batch_size,
                                 validation_split=0.2,  # ✅ Add validation split
                                 callbacks=callbacks, shuffle=True, verbose=0) # verbose=0 to reduce output

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
