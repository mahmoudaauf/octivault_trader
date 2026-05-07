"""
Model Manager — Loads, caches, and validates TensorFlow Keras models.

Supports both .keras and .h5 formats with automatic fallback.
Handles corrupt/incompatible models via quarantine and cleanup.
"""

import logging
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, Union

try:
    import tensorflow as tf
except ImportError:
    tf = None

logger = logging.getLogger(__name__)

KERAS_EXT = ".keras"
LEGACY_EXT = ".h5"
_LOG_THROTTLE_SEC = 120.0
_LAST_LOG_TS: dict[str, float] = defaultdict(float)
_INCOMPATIBLE_QUARANTINE_DIR = "_incompatible_quarantine"


def _is_truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on", "y"}


def _auto_cleanup_incompatible_enabled() -> bool:
    return _is_truthy(os.getenv("MODEL_AUTO_CLEANUP_INCOMPATIBLE", "true"))


def _classify_model_load_error(exc: Exception) -> tuple[bool, str]:
    msg = str(exc or "")
    lower = msg.lower()

    signature_map: dict[str, list[str]] = {
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
                logger.warning(
                    "Failed moving metadata sidecar to quarantine (%s): %s", meta_src, me
                )

        return dst
    except Exception as qe:
        logger.warning("Failed to quarantine incompatible model artifact (%s): %s", src, qe)
        return None


def _log_throttled(level: str, key: str, msg: str, *args):
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
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _with_ext(path: Path, ext: str) -> Path:
    return path.with_suffix(ext)


def _paired_paths(path: Path) -> tuple[Path, Path]:
    base = path.with_suffix("")
    return base.with_suffix(KERAS_EXT), base.with_suffix(LEGACY_EXT)


def build_model_path(
    agent_name: str,
    symbol: str,
    version: str = "v2",
    model_dir: Union[str, Path, None] = None,
    use_legacy_h5: bool = False,
) -> Path:
    """
    Constructs a standardized file path for a Keras model.
    Always uses absolute paths to ensure models are found regardless of CWD.

    Args:
        agent_name: The name of the agent (e.g., "MLForecaster").
        symbol: The trading symbol (e.g., "BTCUSDT").
        version: A version string for the model (e.g., "5m", "v2").
        model_dir: Optional base directory for models. If None, defaults to "models/" in project root.
        use_legacy_h5: If True, forces .h5 extension.

    Returns:
        Path: The constructed absolute file path for the model.
    """
    if model_dir is None:
        try:
            this_file = Path(__file__).resolve()
            project_root = this_file.parent.parent.parent
            model_dir = project_root / "models"
        except Exception:
            model_dir = Path("models")

    base_model_dir = _ensure_models_dir(Path(model_dir).resolve())
    ext = LEGACY_EXT if use_legacy_h5 else KERAS_EXT
    agent_key = "".join(ch for ch in str(agent_name or "").lower() if ch.isalnum()) or "agent"
    symbol_key = str(symbol or "").replace("/", "").replace("-", "").upper() or "SYMBOL"
    version_key = str(version or "v2").strip() or "v2"
    model_filename = f"{agent_key}_{symbol_key}_{version_key}{ext}"
    return base_model_dir / model_filename


def save_model(model, path: Path):
    """Saves a Keras model to the given path in .keras format (or .h5 if explicitly requested)."""
    try:
        if path.suffix.lower() == LEGACY_EXT:
            logger.warning("Saving in legacy HDF5 (.h5). Prefer native Keras format (.keras).")
            _ensure_models_dir(path.parent)
            model.save(path)
        else:
            keras_path = _with_ext(path, KERAS_EXT)
            _ensure_models_dir(keras_path.parent)
            model.save(keras_path)
            path = keras_path
        logger.info("Model saved to: %s", path)
    except Exception as e:
        logger.error("Failed to save model to %s: %s", path, e)


def _try_load(path: Path) -> Any:
    """Attempts to load a Keras model from the given path. Returns None on failure."""
    if tf is None:
        return None
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
                    f"model_quarantined:{path!s}",
                    "Auto-cleaned incompatible model: %s -> %s (reason=%s)",
                    str(path),
                    str(quarantined),
                    str(reason),
                )
                return None
        logger.error("Error loading model from %s: %s", path, e)
    return None


def load_model(path: Path):
    """
    Loads a Keras model from the given path with graceful fallback.
    Tries .keras first, then .h5 as fallback.

    Args:
        path: The file path from which to load the model (can be .keras, .h5, or base name).

    Returns:
        tf.keras.Model or None: The loaded model, or None if loading fails.
    """
    primary = Path(path)
    keras_path, h5_path = _paired_paths(path)

    candidates: list[Path] = []
    if primary.suffix.lower() == KERAS_EXT:
        candidates = [primary, h5_path]
    elif primary.suffix.lower() == LEGACY_EXT:
        candidates = [primary, keras_path]
    else:
        candidates = [keras_path, h5_path]

    available_candidates = [p for p in candidates if p.exists() and p.stat().st_size > 0]
    if tf is None:
        if available_candidates:
            _log_throttled(
                "warning",
                f"tf_missing:{primary!s}",
                "TensorFlow unavailable; model file exists but cannot be loaded: %s",
                str(available_candidates[0]),
            )
        else:
            _log_throttled(
                "info",
                f"model_missing:{primary!s}",
                "Model file not found yet (expected for bootstrap): %s",
                str(primary),
            )
        return None

    for p in candidates:
        m = _try_load(p)
        if m is not None:
            logger.info("Model loaded from: %s", p)
            return m

    remaining_candidates = [p for p in candidates if p.exists() and p.stat().st_size > 0]
    if remaining_candidates:
        _log_throttled(
            "warning",
            f"model_corrupt_or_incompatible:{primary!s}",
            "Failed to load existing model (possibly incompatible/corrupt); tried: %s",
            ", ".join(str(c) for c in candidates),
        )
    else:
        _log_throttled(
            "info",
            f"model_missing:{primary!s}",
            "Model file not found yet (expected for bootstrap): %s",
            ", ".join(str(c) for c in candidates),
        )
    return None


def model_exists(path: Path) -> bool:
    """
    Checks if a model file exists and is not empty.

    Args:
        path: The base file path to check (extension will be ignored).

    Returns:
        bool: True if either .keras or .h5 model file exists and is not empty.
    """
    keras_path, h5_path = _paired_paths(path)
    check_paths = [path] if path.suffix.lower() in [KERAS_EXT, LEGACY_EXT] else []
    check_paths.extend([keras_path, h5_path])

    return any(p.exists() and p.stat().st_size > 0 for p in check_paths)


def safe_load_model(path: Path):
    """
    Safely loads a Keras model from a given path if it exists and is valid.

    Args:
        path: The path to the saved model.

    Returns:
        tf.keras.Model or None: The loaded model, or None if it fails.
    """
    if model_exists(path):
        return load_model(path)
    else:
        logger.warning("Model path does not exist or is empty (checked .keras/.h5): %s", path)
        return None


class ModelManager:
    """
    Wrapper class for managing Keras model persistence (save/load/check existence).
    Abstracts away file extension details, prioritizing .keras format with .h5 fallback.
    """

    def __init__(self, config):
        self.config = config
        logger.info("ModelManager initialized")

    def build_model_path(
        self,
        agent_name: str,
        symbol: str,
        version: str = "v2",
        model_dir: Union[str, Path, None] = None,
        use_legacy_h5: bool = False,
    ) -> Path:
        """Proxies to the global build_model_path function."""
        return build_model_path(agent_name, symbol, version, model_dir, use_legacy_h5)

    def save_model(self, model, path: Path):
        """Proxies to the global save_model function."""
        return save_model(model, path)

    def load_model(self, path: Path):
        """Proxies to the global load_model function."""
        return load_model(path)

    def safe_load_model(self, path: Path):
        """Proxies to the global safe_load_model function."""
        return safe_load_model(path)

    def model_exists(self, path: Path) -> bool:
        """Proxies to the global model_exists function."""
        return model_exists(path)
