import logging
import os
import json
import time
import re
from collections import defaultdict

try:
    import tensorflow as tf
except ImportError:
    tf = None
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Tuple
from collections import deque
import random
import os
import time

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ModelManager")
logger.setLevel(logging.INFO)

# -------------------------------
# Helpers for extensions & paths
# -------------------------------
KERAS_EXT = ".keras"
LEGACY_EXT = ".h5"
_LOG_THROTTLE_SEC = 120.0
_LAST_LOG_TS: Dict[str, float] = defaultdict(float)
_INCOMPATIBLE_QUARANTINE_DIR = "_incompatible_quarantine"


def _is_truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on", "y"}


def _auto_cleanup_incompatible_enabled() -> bool:
    # Enabled by default so legacy/incompatible artifacts are auto-cleaned and retrained.
    return _is_truthy(os.getenv("MODEL_AUTO_CLEANUP_INCOMPATIBLE", "true"))


def _classify_model_load_error(exc: Exception) -> Tuple[bool, str]:
    """
    Classify model-load exceptions that should trigger quarantine cleanup.
    """
    msg = str(exc or "")
    lower = msg.lower()

    signature_map: Dict[str, List[str]] = {
        "legacy_inputlayer_batch_shape": [
            "error when deserializing class 'inputlayer'",
            "unrecognized keyword arguments: ['batch_shape']",
            "unrecognized keyword arguments: ['batch_input_shape']",
        ],
        "legacy_keras_deserialization": [
            "could not deserialize",
            "error when deserializing",
            "unknown layer",
        ],
        "corrupt_or_invalid_model_file": [
            "file signature not found",
            "no model config found",
            "unable to open file",
            "bad marshal data",
        ],
    }

    for reason, needles in signature_map.items():
        if any(needle in lower for needle in needles):
            return True, reason
    return False, "unclassified_load_error"


def _safe_reason_token(reason: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", str(reason or "").lower()).strip("_")
    token = token[:48]
    return token or "incompatible_model"


def _quarantine_model_artifact(path: Path, reason: str) -> Optional[Path]:
    """
    Move an incompatible/corrupt model artifact to a quarantine folder.
    Sidecar metadata is moved as well when present.
    """
    src = Path(path)
    if not src.exists():
        return None
    try:
        qdir = _ensure_models_dir(src.parent / _INCOMPATIBLE_QUARANTINE_DIR)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        token = _safe_reason_token(reason)
        dst = qdir / f"{src.stem}__{stamp}__{token}{src.suffix}"
        i = 1
        while dst.exists():
            dst = qdir / f"{src.stem}__{stamp}__{token}_{i}{src.suffix}"
            i += 1

        src.rename(dst)

        # Move optional metadata sidecar when available.
        meta_src = src.with_name(f"{src.stem}_metadata.pkl")
        if meta_src.exists():
            meta_dst = qdir / f"{meta_src.stem}__{stamp}__{token}{meta_src.suffix}"
            j = 1
            while meta_dst.exists():
                meta_dst = qdir / f"{meta_src.stem}__{stamp}__{token}_{j}{meta_src.suffix}"
                j += 1
            try:
                meta_src.rename(meta_dst)
            except Exception as me:
                logger.warning("⚠️ Failed moving metadata sidecar to quarantine (%s): %s", meta_src, me)

        return dst
    except Exception as qe:
        logger.warning("⚠️ Failed to quarantine incompatible model artifact (%s): %s", src, qe)
        return None


def _log_throttled(level: str, key: str, msg: str, *args):
    """Emit repetitive model-load logs at most once per throttle window."""
    now = time.time()
    k = str(key or "")
    last = float(_LAST_LOG_TS.get(k, 0.0) or 0.0)
    if (now - last) < _LOG_THROTTLE_SEC:
        return
    _LAST_LOG_TS[k] = now
    if level == "warning":
        logger.warning(msg, *args)
    elif level == "error":
        logger.error(msg, *args)
    else:
        logger.info(msg, *args)

def _ensure_models_dir(p: Union[str, Path]) -> Path:
    """Ensures the given directory path exists and returns it as a Path object."""
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def _with_ext(path: Path, ext: str) -> Path:
    """Changes the suffix of a given Path object to the specified extension."""
    return path.with_suffix(ext)

def _paired_paths(path: Path) -> Tuple[Path, Path]:
    """
    Returns a tuple of (keras_path, h5_path) for a given base path,
    ignoring its original extension.
    """
    base = path.with_suffix("") # Remove any existing suffix
    return base.with_suffix(KERAS_EXT), base.with_suffix(LEGACY_EXT)

# -------------------------------
# Public API
# -------------------------------
def build_model_path(agent_name: str, symbol: str, version: str = "v2",
                     model_dir: Union[str, Path, None] = None,
                     use_legacy_h5: bool = False) -> Path:
    """
    Constructs a standardized file path for a Keras model.
    Defaults to the modern .keras format. Set `use_legacy_h5=True`
    only if you need to interoperate with older code that expects HDF5.

    CRITICAL: Always uses absolute paths to ensure models are found
    regardless of the current working directory.

    Args:
        agent_name (str): The name of the agent (e.g., "MLForecaster", "RLAgent").
        symbol (str): The trading symbol (e.g., "BTC-USD").
        version (str): A version string for the model (e.g., "5m", "v1").
        model_dir (Union[str, Path, None]): Optional. The base directory for models.
                                            If None, defaults to "models/" in project root.
        use_legacy_h5 (bool): If True, forces the use of the .h5 extension.

    Returns:
        Path: The constructed absolute file path for the model.
    """
    # If no model_dir provided, construct absolute path to models/ in project root
    if model_dir is None:
        try:
            # Get the absolute path to this file (core/model_manager.py)
            this_file = Path(__file__).resolve()
            # Navigate up to project root: core/model_manager.py -> core/ -> octivault_trader/
            project_root = this_file.parent.parent
            model_dir = project_root / "models"
        except Exception:
            # Fallback: use relative path if __file__ resolution fails
            model_dir = Path("models")
    
    # Convert to absolute path and ensure directory exists
    base_model_dir = _ensure_models_dir(Path(model_dir).resolve())
    ext = LEGACY_EXT if use_legacy_h5 else KERAS_EXT
    agent_key = "".join(ch for ch in str(agent_name or "").lower() if ch.isalnum()) or "agent"
    symbol_key = str(symbol or "").replace("/", "").replace("-", "").upper() or "SYMBOL"
    version_key = str(version or "v2").strip() or "v2"
    model_filename = f"{agent_key}_{symbol_key}_{version_key}{ext}"
    return base_model_dir / model_filename

def save_model(model, path: Path):
    """
    Saves a Keras model to the given path. By default, it saves in the native
    Keras format (.keras). If the path explicitly ends with .h5, it will
    save in HDF5 format and log a deprecation warning.

    Args:
        model: The Keras model to save.
        path (Path): The file path where the model will be saved.
    """
    try:
        if path.suffix.lower() == LEGACY_EXT:
            logger.warning("Saving in legacy HDF5 (.h5). Prefer the native Keras format (.keras).")
            _ensure_models_dir(path.parent)
            model.save(path) # Legacy .h5 format is still supported
        else:
            # Always ensure saving with the modern .keras extension
            keras_path = _with_ext(path, KERAS_EXT)
            _ensure_models_dir(keras_path.parent)
            model.save(keras_path)
            path = keras_path # Update path to reflect the actual saved file
        logger.info(f"💾 Model saved to: {path}")
    except Exception as e:
        logger.error(f"❌ Failed to save model to {path}: {e}")

def _try_load(path: Path) -> Any:
    """
    Attempts to load a Keras model from the given path.
    Returns the model if successful, otherwise None, logging any errors.
    """
    if tf is None: return None
    try:
        if path.exists() and path.stat().st_size > 0:
            return tf.keras.models.load_model(path)
    except Exception as e:
        should_quarantine, reason = _classify_model_load_error(e)
        if should_quarantine and _auto_cleanup_incompatible_enabled():
            quarantined = _quarantine_model_artifact(path, reason)
            if quarantined is not None:
                _log_throttled(
                    "warning",
                    f"model_quarantined:{str(path)}",
                    "🧹 Auto-cleaned incompatible model: %s -> %s (reason=%s). Will retrain using fresh artifact.",
                    str(path),
                    str(quarantined),
                    str(reason),
                )
                return None
        logger.error(f"❌ Error loading model from {path}: {e}")
    return None

def load_model(path: Path):
    """
    Loads a Keras model from the given path with graceful fallback.
    It prioritizes loading based on the provided path's extension,
    then attempts the alternative extension if the primary fails or is missing.

    Args:
        path (Path): The file path from which to load the model.
                     Can be a .keras, .h5, or a base name.

    Returns:
        tf.keras.Model or None: The loaded Keras model, or None if loading fails.
    """
    primary = Path(path)
    keras_path, h5_path = _paired_paths(path)

    candidates: list[Path] = []
    if primary.suffix.lower() == KERAS_EXT:
        candidates = [primary, h5_path] # Prefer .keras, then try .h5
    elif primary.suffix.lower() == LEGACY_EXT:
        candidates = [primary, keras_path] # Prefer .h5, then try .keras
    else:
        # No extension or unknown extension provided, try .keras first then .h5
        candidates = [keras_path, h5_path]

    available_candidates = [p for p in candidates if p.exists() and p.stat().st_size > 0]
    if tf is None:
        # Distinguish runtime dependency failure from true missing-model failure.
        if available_candidates:
            _log_throttled(
                "warning",
                f"tf_missing:{str(primary)}",
                "⚠️ TensorFlow unavailable; model file exists but cannot be loaded: %s",
                str(available_candidates[0]),
            )
        else:
            _log_throttled(
                "info",
                f"model_missing:{str(primary)}",
                "ℹ️ Model file not found yet (expected for bootstrap): %s",
                str(primary),
            )
        return None

    for p in candidates:
        m = _try_load(p)
        if m is not None:
            logger.info(f"✅ Model loaded from: {p}")
            return m

    remaining_candidates = [p for p in candidates if p.exists() and p.stat().st_size > 0]
    if remaining_candidates:
        _log_throttled(
            "warning",
            f"model_corrupt_or_incompatible:{str(primary)}",
            "⚠️ Failed to load existing model (possibly incompatible/corrupt); tried: %s",
            ", ".join(str(c) for c in candidates),
        )
    else:
        _log_throttled(
            "info",
            f"model_missing:{str(primary)}",
            "ℹ️ Model file not found yet (expected for bootstrap): %s",
            ", ".join(str(c) for c in candidates),
        )
    return None

def model_exists(path: Path) -> bool:
    """
    Checks if a model file exists and is not empty, looking for both
    .keras and .h5 extensions based on the given path's base name.

    Args:
        path (Path): The base file path to check (its extension will be ignored).

    Returns:
        bool: True if either the .keras or .h5 model file exists and is not empty,
              False otherwise.
    """
    keras_path, h5_path = _paired_paths(path)
    # Check the original path if it has a valid suffix, then the paired paths
    check_paths = [path] if path.suffix.lower() in [KERAS_EXT, LEGACY_EXT] else []
    check_paths.extend([keras_path, h5_path])

    for p in check_paths:
        if p.exists() and p.stat().st_size > 0:
            return True
    return False

def safe_load_model(path: Path):
    """
    Safely loads a Keras model from a given path if it exists and is valid.
    This function uses the same graceful fallback behavior as `load_model()`,
    checking both .keras and .h5 files.

    Args:
        path (Path): The path to the saved model.

    Returns:
        tf.keras.Model or None: The loaded model, or None if it fails.
    """
    if model_exists(path):
        return load_model(path)
    else:
        logger.warning(f"⚠️ Model path does not exist or is empty (checked .keras/.h5): {path}")
        return None

# --- ModelManager Class Wrapper ---
class ModelManager:
    """
    A wrapper class for managing Keras model persistence (save/load/check existence).
    It abstracts away the file extension details, prioritizing the modern .keras format
    while providing backward compatibility with .h5 files.
    """
    def __init__(self, config):
        self.config = config
        logger.info("ModelManager initialized.")

    def build_model_path(self, agent_name: str, symbol: str, version: str = "v2", model_dir: Union[str, Path, None] = None, use_legacy_h5: bool = False) -> Path:
        """Proxies to the global `build_model_path` function."""
        return build_model_path(agent_name, symbol, version, model_dir, use_legacy_h5)

    def save_model(self, model, path: Path):
        """Proxies to the global `save_model` function."""
        return save_model(model, path)

    def load_model(self, path: Path):
        """Proxies to the global `load_model` function."""
        return load_model(path)

    def safe_load_model(self, path: Path):
        """Proxies to the global `safe_load_model` function."""
        return safe_load_model(path)

    def model_exists(self, path: Path) -> bool:
        """Proxies to the global `model_exists` function."""
        return model_exists(path)

# --- ModelTrainer Class (Commented out as per original request, for context only) ---
# This section is typically part of the model_trainer.py, but was present in the provided model_manager.py snippet.
# For clarity, the actual ModelTrainer class will be defined in core/model_trainer.py.
# class ModelTrainer:
#     def __init__(self, symbol: str, timeframe: str = "5m", input_lookback: int = 20, epsilon_decay_steps: int = 10000, epochs: int = 10):
#         self.symbol = symbol
#         self.timeframe = timeframe
#         self.input_lookback = input_lookback
#         self.epochs = epochs
#         self.model = None
#         self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
#         self.loss_fn = tf.keras.losses.MeanSquaredError()
#         self.memory = deque(maxlen=2000) # Experience replay buffer
#         self.gamma = 0.95 # Discount factor
#         self.epsilon = 1.0 # Exploration-exploitation trade-off
#         self.epsilon_min = 0.01
#         self.epsilon_decay = (self.epsilon - self.epsilon_min) / epsilon_decay_steps
#         self.target_model = None
#         self.update_target_model_freq = 100 # Update target model every N steps

#     def _build_dqn_model(self, state_shape: tuple):
#         model = Sequential([
#             LSTM(64, input_shape=state_shape, return_sequences=True),
#             Dropout(0.2),
#             LSTM(32, return_sequences=False),
#             Dropout(0.2),
#             Dense(24, activation='relu'),
#             Dense(2, activation='linear') # 2 actions: buy, sell/hold
#         ])
#         return model

#     def _get_state(self, data: pd.DataFrame, index: int) -> np.ndarray:
#         if index < self.input_lookback:
#             # Pad with zeros or handle appropriately for start of data
#             padded_data = np.zeros((self.input_lookback, data.shape[1]))
#             padded_data[self.input_lookback - index -1 : ] = data.iloc[:index+1].values
#             state = padded_data
#         else:
#             state = data.iloc[index - self.input_lookback + 1 : index + 1].values
#         return state.reshape(1, self.input_lookback, data.shape[1])

#     def _choose_action(self, state: np.ndarray) -> int:
#         if np.random.rand() <= self.epsilon:
#             return np.random.randint(2) # Explore: 0 for buy, 1 for sell/hold
#         q_values = self.model.predict(state, verbose=0)[0]
#         return np.argmax(q_values) # Exploit: choose best action

#     def _remember(self, state, action, reward, next_state, done):
#         self.memory.append((state, action, reward, next_state, done))

#     def _replay(self, batch_size: int):
#         if len(self.memory) < batch_size:
#             return

#         minibatch = random.sample(self.memory, batch_size)
#         states, actions, rewards, next_states, dones = zip(*minibatch)

#         states = np.vstack(states)
#         next_states = np.vstack(next_states)

#         target_q_values = self.model.predict(states, verbose=0)
#         next_target_q_values = self.target_model.predict(next_states, verbose=0)

#         for i in range(batch_size):
#             if dones[i]:
#                 target_q_values[i][actions[i]] = rewards[i]
#             else:
#                 target_q_values[i][actions[i]] = rewards[i] + self.gamma * np.amax(next_target_q_values[i])

#         self.model.fit(states, target_q_values, epochs=1, verbose=0)

#         if self.epsilon > self.epsilon_min:
#             self.epsilon -= self.epsilon_decay

#     def train_model(self, df: pd.DataFrame, task: str = "reinforcement_learning", epochs: int = 10):
#         if task == "reinforcement_learning":
#             logger.info(f"Starting Reinforcement Learning training for {self.symbol}...")
#             # Scale data for RL
#             scaler = MinMaxScaler()
#             scaled_df = pd.DataFrame(scaler.fit_transform(df[["open", "high", "low", "close", "volume"]]), columns=df.columns)

#             state_shape = (self.input_lookback, scaled_df.shape[1])
#             self.model = self._build_dqn_model(state_shape)
#             self.target_model = self._build_dqn_model(state_shape)
#             self.target_model.set_weights(self.model.get_weights())

#             batch_size = 32
#             for e in range(epochs):
#                 total_reward = 0
#                 for i in range(len(scaled_df) - 1):
#                     state = self._get_state(scaled_df, i)
#                     action = self._choose_action(state)

#                     # Simulate environment step (simplified)
#                     # Reward logic: positive if action leads to profit, negative otherwise
#                     # This is a placeholder; real reward calculation would be more complex
#                     current_close = scaled_df.iloc[i]["close"]
#                     next_close = scaled_df.iloc[i+1]["close"]
#                     reward = 0
#                     if action == 0: # Buy
#                         if next_close > current_close:
#                             reward = 1 # Profit
#                         else:
#                             reward = -1 # Loss
#                     elif action == 1: # Sell/Hold
#                         if next_close < current_close:
#                             reward = 1 # Profit from short or avoiding loss
#                         else:
#                             reward = -1 # Missed opportunity or loss

#                     total_reward += reward
#                     next_state = self._get_state(scaled_df, i + 1)
#                     done = (i == len(scaled_df) - 2) # Done if it's the last step

#                     self._remember(state, action, reward, next_state, done)
#                     self._replay(batch_size)

#                     if i % self.update_target_model_freq == 0:
#                         self.target_model.set_weights(self.model.get_weights())

#                 logger.info(f"Epoch {e+1}/{epochs}, Total Reward: {total_reward}, Epsilon: {self.epsilon:.2f}")

#             model_path = build_model_path("RLAgent", self.symbol, version=self.timeframe)
#             save_model(self.model, model_path)
#             logger.info(f"RL Model trained and saved to {model_path}")
#             return True
#         else:
#             logger.warning(f"Unsupported training task: {task}")
#             return False

# # Example usage (from original model_manager.py, for demonstration purposes)
# if __name__ == "__main__":
#     num_data_points = 200
#     dummy_data = {
#         'open': np.linspace(100, 150, num_data_points) + np.random.randn(num_data_points) * 2,
#         'high': np.linspace(105, 155, num_data_points) + np.random.randn(num_data_points) * 2,
#         'low': np.linspace(95, 145, num_data_points) + np.random.randn(num_data_points) * 2,
#         'close': np.linspace(100, 150, num_data_points) + np.random.randn(num_data_points) * 2,
#         'volume': np.linspace(1000, 5000, num_data_points) + np.random.randn(num_data_points) * 200
#     }
#     df = pd.DataFrame(dummy_data)

#     # Initialize trainer with specific parameters
#     trainer = ModelTrainer(
#         symbol="BTC-USD",
#         timeframe="5m",
#         input_lookback=20,
#         epsilon_decay_steps=num_data_points * 5, # Decay over several passes of the data
#         epochs=5 # Run fewer epochs for quick demo
#     )

#     print("\n" + "-" * 60)
#     print("🚀 Starting Fully Developed ModelTrainer (RL) Demonstration...")
#     print("-" * 60)

#     # 1. Test training with reinforcement_learning task
#     success_rl = trainer.train_model(df, task="reinforcement_learning", epochs=trainer.epochs)
#     print(f"RL training successful: {success_rl}")

#     # 2. Test with an unsupported task
#     success_unsupported = trainer.train_model(df, task="forecasting")
#     print(f"Unsupported task training successful (expected False): {success_unsupported}")

#     # 3. Clean up the created model file
#     model_path_to_clean = build_model_path("RLAgent", "BTC-USD", "5m")
#     if model_exists(model_path_to_clean):
#         os.remove(model_path_to_clean)
#         print(f"🗑️ Cleaned up model file: {model_path_to_clean}")
