
import logging
import os
import random
import time
import numpy as np
from collections import deque
from typing import Any
from pathlib import Path

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
except ImportError:
    tf = None

try:
    import pandas as pd
except ImportError:
    pd = None

# Using ModelManager's helper to build paths if needed, 
# or we can redefine it locally to keep it standalone.
# We'll rely on the caller to handle paths or import from model_manager.
from core.model_manager import build_model_path, save_model, model_exists

class ModelTrainer:
    """
    Reinforcement Learning Trainer using DQN (Deep Q-Network).
    Trains a model to predict BUY (1) or HOLD/SELL (0) actions based on market data states.
    """
    def __init__(self, symbol: str, timeframe: str = "5m", input_lookback: int = 20, 
                 epsilon_decay_steps: int = 10000, epochs: int = 5, learning_rate: float = 0.001,
                 agent_name: str = "TrendHunter", model_manager: Any = None):
        self.logger = logging.getLogger(f"ModelTrainer_{symbol}")
        self.symbol = symbol
        self.timeframe = timeframe
        self.input_lookback = input_lookback
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.agent_name = agent_name
        self.model_manager = model_manager
        
        self.model = None
        self.target_model = None
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0 # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / epsilon_decay_steps
        self.update_target_freq = 100
        self.batch_size = 32

        if tf is None:
            self.logger.warning("TensorFlow not available. Training will be disabled.")

    def _build_model(self, state_shape):
        if tf is None: return None
        
        # Determine model type from environment/config if possible, fallback to LSTM
        model_type = os.getenv("MODEL_TYPE", "LSTM").upper()
        
        layers = []
        if model_type == "GRU":
            self.logger.info(f"Building GRU-based model for {self.symbol}")
            layers.append(GRU(64, input_shape=state_shape, return_sequences=True))
            layers.append(Dropout(0.2))
            layers.append(GRU(32, return_sequences=False))
        else:
            self.logger.info(f"Building LSTM-based model for {self.symbol}")
            layers.append(LSTM(64, input_shape=state_shape, return_sequences=True))
            layers.append(Dropout(0.2))
            layers.append(LSTM(32, return_sequences=False))
            
        layers.append(Dropout(0.2))
        layers.append(Dense(24, activation='relu'))
        layers.append(Dense(2, activation='linear')) # [Q(Hold), Q(Buy)]
        
        model = Sequential(layers)
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def _get_state(self, data, index):
        # Extract a window of 'input_lookback' steps ending at 'index'
        start_idx = index - self.input_lookback + 1
        if start_idx < 0:
            # Pad with first row if strictly needed, or just return zeros
            window = np.zeros((self.input_lookback, data.shape[1]))
            # Fill existing
            valid_len = index + 1
            window[-valid_len:] = data.iloc[:index+1].values
        else:
            window = data.iloc[start_idx : index+1].values
            
        return window.reshape(1, self.input_lookback, data.shape[1])

    def train_model(self, df, task="reinforcement_learning", epochs=None):
        """
        Main entry point to train the model on the provided DataFrame.
        This blocking call runs the training loop.
        """
        if tf is None:
            self.logger.error("Cannot train: TensorFlow missing.")
            return False
            
        if df is None or len(df) < (self.input_lookback + 50):
            self.logger.warning(f"Insufficient data for training {self.symbol} (rows={len(df) if df is not None else 0}).")
            return False

        if task != "reinforcement_learning":
            self.logger.warning(f"Unsupported task: {task}")
            return False

        epochs = epochs or self.epochs
        self.logger.info(f"Starting training for {self.symbol} ({epochs} epochs)...")
        
        # Prepare Data features
        # Assuming df has 'open','high','low','close','volume'
        # Normalize/Scale (Simple MinMax-ish for speed here, ideally use a robust scaler)
        model_df = df[['open', 'high', 'low', 'close', 'volume']].copy()
        for c in model_df.columns:
            model_df[c] = (model_df[c] - model_df[c].mean()) / (model_df[c].std() + 1e-8)

        state_shape = (self.input_lookback, model_df.shape[1])
        
        self.model = self._build_model(state_shape)
        self.target_model = self._build_model(state_shape)
        self.target_model.set_weights(self.model.get_weights())

        start_ts = time.time()
        
        # simplified training loop
        for e in range(epochs):
            total_reward = 0
            # Walk forward through data
            # Skip lookback period
            for i in range(self.input_lookback, len(model_df) - 1):
                state = self._get_state(model_df, i)
                
                # Epsilon-greedy action
                if np.random.rand() <= self.epsilon:
                    action = np.random.randint(2)
                else:
                    q_values = self.model.predict(state, verbose=0)
                    action = np.argmax(q_values[0])

                # Reward: (Next Close - Current Close) if Buy (1), else 0 (Hold)
                # Note: This is a very simplified reward function for "Trend Following"
                curr_close = df.iloc[i]['close']
                next_close = df.iloc[i+1]['close']
                pct_change = (next_close - curr_close) / curr_close
                
                if action == 1: # Buy
                    reward = pct_change * 100.0 # e.g. 1% move = 1.0 reward
                else: 
                    # Hold/Sell - Avoid loss is good? Missed gain is bad?
                    # Simplified: 0 reward for doing nothing, or penalized missed gain?
                    reward = 0.0

                total_reward += reward
                
                # Store
                next_state = self._get_state(model_df, i+1)
                done = (i == len(model_df) - 2)
                self.memory.append((state, action, reward, next_state, done))
                
                # Replay Learning
                if len(self.memory) > self.batch_size:
                    minibatch = random.sample(self.memory, self.batch_size)
                    states_mb = np.vstack([x[0] for x in minibatch])
                    actions_mb = np.array([x[1] for x in minibatch])
                    rewards_mb = np.array([x[2] for x in minibatch])
                    next_states_mb = np.vstack([x[3] for x in minibatch])
                    dones_mb = np.array([x[4] for x in minibatch])
                    
                    target_q = self.model.predict(states_mb, verbose=0)
                    next_target_q = self.target_model.predict(next_states_mb, verbose=0)
                    
                    for k in range(self.batch_size):
                        if dones_mb[k]:
                            target_q[k][actions_mb[k]] = rewards_mb[k]
                        else:
                            target_q[k][actions_mb[k]] = rewards_mb[k] + self.gamma * np.amax(next_target_q[k])
                    
                    self.model.fit(states_mb, target_q, epochs=1, verbose=0)

                # Decay Epsilon
                if self.epsilon > self.epsilon_min:
                    self.epsilon -= self.epsilon_decay
                
                # Update Target
                if i % self.update_target_freq == 0:
                    self.target_model.set_weights(self.model.get_weights())
            
            self.logger.info(f"Epoch {e+1}/{epochs} | Reward: {total_reward:.4f} | Epsilon: {self.epsilon:.4f}")

        # Save Result
        path = build_model_path(self.agent_name, self.symbol, version=self.timeframe)
        save_model(self.model, path)
        self.logger.info(f"Training finished in {time.time() - start_ts:.1f}s. Model saved to {path}")
        return True
