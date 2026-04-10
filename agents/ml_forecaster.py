import os
import asyncio
import logging
import csv
import time
import contextlib
import numpy as np
from datetime import datetime
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List, Set
from pathlib import Path
from collections import defaultdict, deque
import inspect  # <-- added

try:
    from core.agent_optimizer import load_tuned_params
except Exception as _e:
    logging.getLogger("MLForecaster").warning(
        "load_tuned_params import failed; using empty tuned params fallback: %s",
        _e,
        exc_info=True,
    )
    def load_tuned_params(_name: str):  # type: ignore
        return {}

try:
    from core.component_status_logger import log_component_status
except Exception as _e:
    logging.getLogger("MLForecaster").warning(
        "log_component_status import failed; using no-op fallback: %s",
        _e,
        exc_info=True,
    )
    def log_component_status(*_args, **_kwargs):  # type: ignore
        return None

try:
    from core.stubs import is_fresh  # freshness gate
except Exception as _e:
    logging.getLogger("MLForecaster").warning(
        "is_fresh import failed; using permissive freshness fallback: %s",
        _e,
        exc_info=True,
    )
    async def is_fresh(*_args, **_kwargs):  # type: ignore
        return True

try:
    from core.model_trainer import ModelTrainer
except Exception as _e:
    ModelTrainer = None  # type: ignore
    logging.getLogger("MLForecaster").warning(
        "ModelTrainer import failed; training/retrain will be disabled for MLForecaster: %s",
        _e,
        exc_info=True,
    )

try:
    from utils.shared_state_tools import fee_bps
except Exception as _e:
    logging.getLogger("MLForecaster").warning(
        "fee_bps import failed; using fallback taker fee bps=10: %s",
        _e,
        exc_info=True,
    )
    def fee_bps(*_args, **_kwargs):  # type: ignore
        return 10.0

logger = logging.getLogger("MLForecaster")
if not logger.handlers:
    logger.setLevel(logging.INFO)


class MLForecaster:

    async def _get_market_data_safe(self, symbol: str, timeframe: str):
        fn = getattr(self.shared_state, "get_market_data", None)
        if not callable(fn):
            return None
        data = fn(symbol, timeframe)
        if asyncio.iscoroutine(data):
            data = await data
        return data
    """
    ML-based forecaster that PRODUCES signals and delegates execution to MetaController.
    - Emits via meta_controller.receive_signal(name, symbol, payload, confidence)
    - Respects confidence/freshness
    - No direct order placement; signal-only architecture
    """

    agent_type = "strategy"

    def __init__(self, shared_state, execution_manager, config, tp_sl_engine=None, **kwargs):
        self.name = kwargs.get("name", "MLForecaster")
        self.shared_state = shared_state
        self.execution_manager = execution_manager
        self.config = config
        self.tp_sl_engine = tp_sl_engine

        self.logger = logging.getLogger(self.name)
        self.interval = float(getattr(config, "ML_FORECASTER_INTERVAL", 60.0))

        # Freshness & emission controls
        self.fresh_max_age_s = float(getattr(self.config, "FRESH_MAX_AGE_S", 120.0))
        self.allow_sell_without_position = bool(getattr(self.config, "ALLOW_SELL_WITHOUT_POSITION", False))

        # Optional deps
        self.market_data_feed = kwargs.get("market_data_feed")
        self.meta_controller = kwargs.get("meta_controller")
        self.model_manager = kwargs.get("model_manager")
        self.symbol_manager = kwargs.get("symbol_manager")
        self.exchange_client = kwargs.get("exchange_client")
        self.database_manager = kwargs.get("database_manager")
        self.agent_schedule = kwargs.get("agent_schedule")

        # Log initialization state
        self.logger.info(
            f"[{self.name}] Initialized with: model_manager={'✓' if self.model_manager else '✗ MISSING'}, "
            f"meta_controller={'✓' if self.meta_controller else '✗'}, "
            f"symbol_manager={'✓' if self.symbol_manager else '✗'}, "
            f"market_data_feed={'✓' if self.market_data_feed else '✗'}"
        )

        # Symbols & tf
        self.symbols: List[str] = list(kwargs.get("symbols", []) or [])
        self.timeframe = kwargs.get("timeframe", "5m")

        # Tuned params (safe)
        # Move defining _cfg later or call it via self._cfg
        try:
            tuned_global = load_tuned_params(self.name) or {}
        except Exception:
            tuned_global = {}

        self._tuned_global = tuned_global
        self.window_size = int(tuned_global.get("window_size", self._cfg("WINDOW_SIZE", 60)))
        self._lookback_default = max(
            60,
            int(self._cfg("ML_LOOKBACK_DEFAULT", os.getenv("ML_LOOKBACK_DEFAULT", 60)) or 60),
        )
        self._lookback_min = max(
            40,
            int(self._cfg("ML_LOOKBACK_MIN", os.getenv("ML_LOOKBACK_MIN", 40)) or 40),
        )
        self._lookback_max = max(
            int(self._lookback_min),
            int(self._cfg("ML_LOOKBACK_MAX", os.getenv("ML_LOOKBACK_MAX", 160)) or 160),
        )

        # Caching / performance
        self.model_cache: Dict[str, Any] = {}
        self._model_mtime: Dict[str, float] = {}
        self._predict_fns: Dict[Tuple[str, int, int], Any] = {}  # (model_path, lookback, feature_dim) -> tf.function
        self._retrained_feature_models: Set[str] = set()

        # Concurrency & limits
        self.max_concurrency = int(getattr(self.config, "MLF_MAX_CONCURRENCY", 6))
        self.symbol_timeout_s = float(getattr(self.config, "MLF_SYMBOL_TIMEOUT_S", 15.0))
        self.predict_timeout_s = float(getattr(self.config, "MLF_PREDICT_TIMEOUT_S", 5.0))
        self.max_symbols_per_tick = 5
        self._sem = asyncio.Semaphore(self.max_concurrency)
        self._stop_event = asyncio.Event()
        
        # ARCHITECTURAL FIX: Signal collection buffer for generate_signals()
        self._collected_signals: List[Dict[str, Any]] = []

        # Confidence calibration / EV floor state
        self._conf_eval_enabled = self._cfg_bool("ML_CONF_EVAL_ENABLED", True)
        # Expected-move horizon (minutes). Default to 1h to avoid noise-trading on micro moves.
        self._conf_horizon_min = float(self._cfg("ML_CONF_EVAL_HORIZON_MIN", 60.0) or 60.0)
        self._conf_bucket_size = float(self._cfg("ML_CONF_BUCKET_SIZE", 0.05) or 0.05)
        self._conf_min_bucket_samples = int(self._cfg("ML_CONF_MIN_BUCKET_SAMPLES", 8) or 8)
        self._conf_min_total_samples = int(self._cfg("ML_CONF_MIN_TOTAL_SAMPLES", 40) or 40)
        self._conf_recalibrate_sec = float(self._cfg("ML_CONF_RECALIBRATE_SEC", 300.0) or 300.0)
        self._conf_floor_ema_alpha = float(self._cfg("ML_CONF_REQUIRED_EMA_ALPHA", 0.35) or 0.35)
        self._conf_floor_min = float(self._cfg("ML_DYNAMIC_CONF_FLOOR_MIN", 0.35) or 0.35)
        self._conf_floor_max = float(self._cfg("ML_DYNAMIC_CONF_FLOOR_MAX", 0.95) or 0.95)
        self._conf_hard_emit_floor = float(self._cfg("ML_MIN_CONF_EMIT_HARD", 0.05) or 0.05)
        self._regime_vol_low_pct = float(self._cfg("ML_REGIME_VOL_LOW_PCT", 0.0045) or 0.0045)
        self._regime_vol_high_pct = float(self._cfg("ML_REGIME_VOL_HIGH_PCT", 0.0150) or 0.0150)
        self._expected_move_fallback_pct = float(self._cfg("ML_EXPECTED_MOVE_FALLBACK_PCT", 0.0065) or 0.0065)
        self._expected_move_min_pct = float(self._cfg("ML_EXPECTED_MOVE_MIN_PCT", 0.0010) or 0.0010)
        self._expected_move_max_pct = float(self._cfg("ML_EXPECTED_MOVE_MAX_PCT", 0.0500) or 0.0500)
        
        # Regime-dependent horizon scaling (professional recommendation)
        # Bull regime: shorter horizon (60m), wider expected moves available
        # Normal regime: standard horizon (120m), moderate expected moves
        # Bear regime: longer horizon (disabled for longs)
        self._regime_horizon_map = {
            "bull": float(self._cfg("ML_REGIME_BULL_HORIZON_MIN", 60.0) or 60.0),
            "normal": float(self._cfg("ML_REGIME_NORMAL_HORIZON_MIN", 120.0) or 120.0),
            "bear": float(self._cfg("ML_REGIME_BEAR_HORIZON_MIN", 9999.0) or 9999.0),  # Effectively disable
            "trend": float(self._cfg("ML_REGIME_TREND_HORIZON_MIN", 90.0) or 90.0),
            "sideways": float(self._cfg("ML_REGIME_SIDEWAYS_HORIZON_MIN", 240.0) or 240.0),
            "high_vol": float(self._cfg("ML_REGIME_HIGHVOL_HORIZON_MIN", 60.0) or 60.0),
            "low_vol": float(self._cfg("ML_REGIME_LOWVOL_HORIZON_MIN", 240.0) or 240.0),
        }
        self._institutional_regime_filter_enabled = self._cfg_bool(
            "ML_INSTITUTIONAL_REGIME_FILTER_ENABLED",
            True,
        )
        self._inst_block_bear_buy = self._cfg_bool("ML_INST_BLOCK_BEAR_BUY", True)
        self._inst_high_vol_min_conf = float(self._cfg("ML_INST_HIGH_VOL_MIN_CONF", 0.72) or 0.72)
        self._inst_sideways_min_conf = float(self._cfg("ML_INST_SIDEWAYS_MIN_CONF", 0.68) or 0.68)
        self._inst_normal_min_conf = float(self._cfg("ML_INST_NORMAL_MIN_CONF", 0.58) or 0.58)
        self._inst_bull_min_conf = float(self._cfg("ML_INST_BULL_MIN_CONF", 0.55) or 0.55)
        self._inst_high_vol_ev_mult = float(self._cfg("ML_INST_HIGH_VOL_EV_MULT", 2.4) or 2.4)
        self._inst_sideways_ev_mult = float(self._cfg("ML_INST_SIDEWAYS_EV_MULT", 2.1) or 2.1)
        self._inst_normal_ev_mult = float(self._cfg("ML_INST_NORMAL_EV_MULT", 1.8) or 1.8)
        self._inst_bull_ev_mult = float(self._cfg("ML_INST_BULL_EV_MULT", 1.6) or 1.6)
        self._expected_move_calibration_enabled = self._cfg_bool(
            "ML_EXPECTED_MOVE_CALIBRATION_ENABLED",
            True,
        )
        self._expected_move_calib_alpha = float(self._cfg("ML_EXPECTED_MOVE_CALIB_ALPHA", 0.15) or 0.15)
        self._expected_move_calib_min_mult = float(self._cfg("ML_EXPECTED_MOVE_CALIB_MIN_MULT", 0.5) or 0.5)
        self._expected_move_calib_max_mult = float(self._cfg("ML_EXPECTED_MOVE_CALIB_MAX_MULT", 1.8) or 1.8)
        self._expected_move_calib_min_samples = int(self._cfg("ML_EXPECTED_MOVE_CALIB_MIN_SAMPLES", 25) or 25)
        self._expected_move_calib_global = 1.0
        self._expected_move_calib_by_regime: Dict[str, float] = {}
        self._expected_move_calib_counts: Dict[str, int] = {}
        self._sideways_conf_compression_enabled = self._cfg_bool("ML_SIDEWAYS_CONF_COMPRESSION_ENABLED", True)
        self._sideways_conf_compressed_floor = float(
            self._cfg("ML_SIDEWAYS_CONF_COMPRESSED_FLOOR", 0.65) or 0.65
        )
        self._conf_report_dir = str(self._cfg("ML_CONF_REPORT_DIR", "artifacts") or "artifacts")
        self._conf_plot_enabled = self._cfg_bool("ML_CONF_PLOT_ENABLED", True)
        self._conf_print_rows = int(self._cfg("ML_CONF_PRINT_ROWS", 20) or 20)
        self._conf_max_pending = int(self._cfg("ML_CONF_MAX_PENDING_SAMPLES", 2500) or 2500)
        self._conf_max_completed = int(self._cfg("ML_CONF_MAX_COMPLETED_SAMPLES", 5000) or 5000)
        self._conf_backtest_on_startup = self._cfg_bool("ML_CONF_BACKTEST_ON_STARTUP", True)
        self._conf_backtest_stride = int(self._cfg("ML_CONF_BACKTEST_STRIDE", 1) or 1)
        self._conf_backtest_max_per_symbol = int(
            self._cfg("ML_CONF_BACKTEST_MAX_SAMPLES_PER_SYMBOL", 300) or 300
        )
        self._pending_conf_samples: deque = deque(maxlen=max(100, self._conf_max_pending))
        self._completed_conf_samples: deque = deque(maxlen=max(200, self._conf_max_completed))
        self._last_conf_recalc_ts = 0.0
        self._dynamic_required_conf: Optional[float] = None
        self._dynamic_required_conf_by_regime: Dict[str, float] = {}
        self._startup_backtest_submitted = False
        self._startup_backtest_task: Optional[asyncio.Task] = None
        self._startup_backtest_done = False

        # Regime-aware feature stack (pattern guesser -> probabilistic edge detector)
        self._legacy_feature_columns: List[str] = ["open", "high", "low", "close", "volume"]
        self._edge_feature_columns: List[str] = [
            "returns_1",
            "returns_3",
            "returns_5",
            "atr_pct",
            "realized_vol_20",
            "realized_vol_50",
            "volatility_ratio",
            "volatility_zscore",
            "ema9_dist",
            "ema21_dist",
            "ema50_dist",
            "ema_slope",
            "ema_cross_dist",
            "rsi14",
            "roc10",
            "macd_hist",
            "volume_zscore",
            "volume_spike_ratio",
            "range_zscore",
            "rolling_range_expansion",
            "trend_strength",
            "trend_flag",
            "sideways_flag",
            "high_vol_flag",
        ]
        self._feature_force_upgrade = self._cfg_bool("ML_FEATURE_FORCE_UPGRADE", True)
        self._auto_retrain_feature_mismatch = self._cfg_bool("ML_AUTO_RETRAIN_FEATURE_MISMATCH", True)

        # Keep inference non-blocking: training is queued in background.
        self.train_cooldown_s = float(getattr(self.config, "MLF_TRAIN_COOLDOWN_S", 180.0))
        self.feature_upgrade_cooldown_s = float(
            getattr(self.config, "MLF_FEATURE_UPGRADE_COOLDOWN_S", 900.0)
        )
        self.train_timeout_s = float(getattr(self.config, "MLF_TRAIN_TIMEOUT_S", 300.0))
        self.max_background_trains = int(getattr(self.config, "MLF_MAX_BACKGROUND_TRAINS", 1))
        self._train_sem = asyncio.Semaphore(max(1, self.max_background_trains))
        self._train_tasks: Dict[str, asyncio.Task] = {}
        self._train_last_attempt_ts: Dict[Tuple[str, str], float] = {}
        self._feature_upgrade_pending: Set[str] = set()

        # Retrain orchestration guards (avoid hot-loop retraining on CPU).
        self._retrain_lock = asyncio.Lock()
        self._retrain_last_end_ts = 0.0
        self._retrain_rr_cursor = 0
        self._retrain_min_gap_s = float(
            self._cfg("ML_RETRAIN_MIN_GAP_S", os.getenv("ML_RETRAIN_MIN_GAP_S", 900.0)) or 900.0
        )
        self._retrain_symbol_timeout_s = float(
            self._cfg("ML_RETRAIN_SYMBOL_TIMEOUT_S", os.getenv("ML_RETRAIN_SYMBOL_TIMEOUT_S", 180.0)) or 180.0
        )
        self._retrain_run_budget_s = float(
            self._cfg("ML_RETRAIN_RUN_BUDGET_S", os.getenv("ML_RETRAIN_RUN_BUDGET_S", 240.0)) or 240.0
        )
        self._retrain_max_rows = int(
            self._cfg("ML_RETRAIN_MAX_ROWS", os.getenv("ML_RETRAIN_MAX_ROWS", 500)) or 500
        )
        self._retrain_default_epochs = int(
            self._cfg("ML_RETRAIN_EPOCHS", os.getenv("ML_RETRAIN_EPOCHS", 3)) or 3
        )
        self._retrain_cpu_epoch_cap = int(
            self._cfg("ML_RETRAIN_CPU_EPOCH_CAP", os.getenv("ML_RETRAIN_CPU_EPOCH_CAP", 15)) or 15
        )
        self._retrain_min_val_acc = float(
            self._cfg("ML_RETRAIN_MIN_VAL_ACC", os.getenv("ML_RETRAIN_MIN_VAL_ACC", 0.52)) or 0.52
        )
        self._full_train_on_startup = self._cfg_bool("ML_FULL_TRAIN_ON_STARTUP", True)
        self._full_train_force_first_boot = self._cfg_bool("ML_FULL_TRAIN_FORCE_FIRST_BOOT", False)
        self._full_train_target_rows = int(
            self._cfg("ML_FULL_TRAIN_TARGET_ROWS", os.getenv("ML_FULL_TRAIN_TARGET_ROWS", 3000)) or 3000
        )
        self._full_train_max_rows = int(
            self._cfg("ML_FULL_TRAIN_MAX_ROWS", os.getenv("ML_FULL_TRAIN_MAX_ROWS", 10000)) or 10000
        )
        self._full_train_epochs = int(
            self._cfg("ML_FULL_TRAIN_EPOCHS", os.getenv("ML_FULL_TRAIN_EPOCHS", 15)) or 15
        )
        self._full_train_cpu_epoch_cap = int(
            self._cfg("ML_FULL_TRAIN_CPU_EPOCH_CAP", os.getenv("ML_FULL_TRAIN_CPU_EPOCH_CAP", 20)) or 20
        )
        self._full_train_interval_s = float(
            self._cfg("ML_FULL_TRAIN_INTERVAL_S", os.getenv("ML_FULL_TRAIN_INTERVAL_S", 86400.0)) or 86400.0
        )
        self._full_train_min_rows = int(
            self._cfg("ML_FULL_TRAIN_MIN_ROWS", os.getenv("ML_FULL_TRAIN_MIN_ROWS", 3000)) or 3000
        )
        self._retrain_max_rows = max(500, int(self._retrain_max_rows))
        self._retrain_default_epochs = max(3, min(5, int(self._retrain_default_epochs)))
        self._retrain_cpu_epoch_cap = max(3, int(self._retrain_cpu_epoch_cap))
        self._retrain_min_val_acc = max(0.52, min(0.99, float(self._retrain_min_val_acc)))
        self._full_train_min_rows = max(int(self._full_train_min_rows), 3000)
        self._full_train_target_rows = max(int(self._full_train_target_rows), int(self._full_train_min_rows))
        self._full_train_max_rows = max(int(self._full_train_max_rows), int(self._full_train_target_rows))
        self._full_train_epochs = max(15, min(20, int(self._full_train_epochs)))
        self._full_train_cpu_epoch_cap = max(15, int(self._full_train_cpu_epoch_cap))
        self._lookback_default = max(60, int(self._lookback_default))
        self._lookback_min = max(40, min(int(self._lookback_default), int(self._lookback_min)))
        self._lookback_max = max(int(self._lookback_default), int(self._lookback_max))
        self._inst_high_vol_min_conf = max(self._conf_floor_min, min(self._conf_floor_max, float(self._inst_high_vol_min_conf)))
        self._inst_sideways_min_conf = max(self._conf_floor_min, min(self._conf_floor_max, float(self._inst_sideways_min_conf)))
        self._inst_normal_min_conf = max(self._conf_floor_min, min(self._conf_floor_max, float(self._inst_normal_min_conf)))
        self._inst_bull_min_conf = max(self._conf_floor_min, min(self._conf_floor_max, float(self._inst_bull_min_conf)))
        self._inst_high_vol_ev_mult = max(1.0, float(self._inst_high_vol_ev_mult))
        self._inst_sideways_ev_mult = max(1.0, float(self._inst_sideways_ev_mult))
        self._inst_normal_ev_mult = max(1.0, float(self._inst_normal_ev_mult))
        self._inst_bull_ev_mult = max(1.0, float(self._inst_bull_ev_mult))
        self._expected_move_calib_alpha = max(0.01, min(1.0, float(self._expected_move_calib_alpha)))
        self._expected_move_calib_min_mult = max(0.1, float(self._expected_move_calib_min_mult))
        self._expected_move_calib_max_mult = max(
            float(self._expected_move_calib_min_mult),
            float(self._expected_move_calib_max_mult),
        )
        self._expected_move_calib_min_samples = max(1, int(self._expected_move_calib_min_samples))
        self._full_train_last_ts: Dict[str, float] = {}
        self._startup_full_train_pending: Set[str] = set()
        self._startup_full_train_done: Set[str] = set()

    @property
    def min_conf(self) -> float:
        """Dynamic access to minimum signal confidence (Phase A)."""
        base = float(self._tuned_global.get("ML_MIN_CONF_EMIT", self._cfg("ML_MIN_CONF_EMIT", 0.55)))
        dyn = None
        try:
            dyn_cfg = getattr(self.shared_state, "dynamic_config", {}) or {}
            dyn = dyn_cfg.get("ML_DYNAMIC_REQUIRED_CONF")
        except Exception:
            dyn = None
        if dyn is None:
            dyn = self._dynamic_required_conf
        if dyn is None:
            return max(self._conf_floor_min, min(self._conf_floor_max, base))
        try:
            return max(self._conf_floor_min, min(self._conf_floor_max, float(dyn)))
        except Exception:
            return max(self._conf_floor_min, min(self._conf_floor_max, base))

    def _cfg(self, key: str, default: Any = None) -> Any:
        # 1. Check SharedState for live/dynamic overrides
        if hasattr(self.shared_state, "dynamic_config"):
            val = self.shared_state.dynamic_config.get(key)
            if val is not None:
                return val

        # 2. Fallback to static config (env or file)
        if isinstance(self.config, dict):
            return self.config.get(key, default)
        return getattr(self.config, key, default)

    def _cfg_bool(self, key: str, default: bool = False) -> bool:
        raw = self._cfg(key, default)
        if isinstance(raw, bool):
            return raw
        if raw is None:
            return bool(default)
        return str(raw).strip().lower() in {"1", "true", "yes", "on"}

    def _resolve_lookback(self, tuned: Optional[Dict[str, Any]] = None) -> int:
        raw = None
        if isinstance(tuned, dict) and tuned.get("lookback") is not None:
            raw = tuned.get("lookback")
        if raw is None:
            raw = self._cfg("LOOKBACK", self._lookback_default)
        try:
            lb = int(raw or self._lookback_default)
        except Exception:
            lb = int(self._lookback_default)
        return max(int(self._lookback_min), min(int(self._lookback_max), int(lb)))

    async def _has_position(self, symbol: str) -> bool:
        """Return True if we hold a positive position for symbol. Safe across sync/async branches."""
        qty = 0.0
        try:
            fn = getattr(self.shared_state, "get_position_qty", None) or getattr(self.shared_state, "get_position_quantity", None)
            if callable(fn):
                res = fn(symbol)
                qty = await res if asyncio.iscoroutine(res) else (res or 0.0)
        except Exception:
            qty = 0.0
        try:
            return float(qty) > 0
        except Exception:
            return False

        log_component_status(self.name, "Initialized")
        self.logger.info(
            f"{self.name} initialized (tf={self.timeframe}, interval={self.interval}s, "
            f"conc={self.max_concurrency}, symbol_timeout={self.symbol_timeout_s}s)"
        )

    # ---------------- Lifecycle ----------------

    async def stop(self):
        self.logger.info(f"🚓 {self.name} stopping...")
        self._stop_event.set()
        for task in list(self._train_tasks.values()):
            if task and not task.done():
                task.cancel()

    # ARCHITECTURAL FIX: run_loop() REMOVED
    # Strategy agents MUST NOT self-schedule
    # AgentManager calls generate_signals() on a central tick

    async def generate_signals(self):
        """Generate signals for all symbols. Called by AgentManager."""
        # Collect signals from run_once()
        self._collected_signals = []
        await self.run_once()
        # Return collected signals
        signals = self._collected_signals
        self._collected_signals = []
        return signals

    # ---------------- Core pass ----------------

    async def run_once(self):
        # Early skip if market data is not ready
        self.logger.info(f"[{self.name}] run_once starting. SharedState ID: {id(self.shared_state)}")
        if hasattr(self.shared_state, "is_market_data_ready"):
            try:
                if not self.shared_state.is_market_data_ready():
                    self.logger.warning(f"[{self.name}] Market data not ready; skipping tick.")
                    return
            except Exception:
                pass
        if self._conf_eval_enabled:
            try:
                await self._finalize_matured_confidence_samples()
                await self._maybe_recalibrate_confidence_floor(source="live")
            except Exception:
                self.logger.debug("[%s] confidence eval loop failed", self.name, exc_info=True)
        self.logger.debug(f"[{self.name}] Executing run_once.")
        if True:
            syms = await self._safe_get_symbols()
            if set(syms) != set(self.symbols):
                self.symbols = list(syms)
                self.logger.info(f"[{self.name}] Updated symbols: {len(self.symbols)}")
        # 🔧 Immediate Fix (Safe + Correct)
        # Symbol auto-sync from SharedState before startup backtest
        if not self.symbols:
            try:
                syms = await self._safe_get_symbols()
                if syms:
                    self.symbols = list(syms)
                    self.logger.info(f"[{self.name}] Synced symbols from SharedState: {self.symbols}")
            except Exception:
                pass
        if not self.symbols:
            self.logger.warning(f"[{self.name}] No symbols to process.")
            return
        await self._maybe_schedule_startup_backtest()

        # ARCHITECTURAL FIX: Process ALL symbols, not a subset
        # AgentManager contract: generate_signals() must scan ALL symbols per tick
        batch = self.symbols[:self.max_symbols_per_tick]

        async def _one(sym: str):
            async with self._sem:
                try:
                    return await asyncio.wait_for(self.run(sym), timeout=self.symbol_timeout_s)
                except asyncio.TimeoutError:
                    self.logger.warning(
                        f"[{self.name}] {sym} timed out after {self.symbol_timeout_s}s; dropping to NO_DECISION."
                    )
                    return {"action": "hold", "confidence": 0.0, "reason": "timeout"}
                except Exception:
                    self.logger.exception("[%s] %s processing failed; dropping to NO_DECISION.", self.name, sym)
                    return {"action": "hold", "confidence": 0.0, "reason": "processing_error"}

        # Capture results (currently unused but good practice)
        results = await asyncio.gather(*[_one(s) for s in batch], return_exceptions=False)
        self.logger.debug(f"[{self.name}] Finished run_once (processed={len(batch)} symbols, generated={len(self._collected_signals)} signals).")

    # ---------------- Helpers ----------------

    async def _safe_get_symbols(self) -> List[str]:
        """
        SharedState.get_accepted_symbols may return dict/list and may be sync/async across branches.
        Normalize to List[str].
        """
        try:
            getter = getattr(self.shared_state, "get_accepted_symbols", None)
            if callable(getter):
                res = getter()
                if asyncio.iscoroutine(res):
                    res = await res
            else:
                res = []
            if isinstance(res, dict):
                return list(res.keys())
            return list(res or [])
        except Exception:
            return []

    def _cleanup_train_tasks(self) -> None:
        done_keys: List[str] = []
        for model_path, task in list(self._train_tasks.items()):
            if not task.done():
                continue
            done_keys.append(model_path)
            with contextlib.suppress(asyncio.CancelledError):
                exc = task.exception()
                if exc:
                    self.logger.warning(
                        "[%s] background training task failed for %s: %s",
                        self.name,
                        model_path,
                        exc,
                    )
        for model_path in done_keys:
            self._train_tasks.pop(model_path, None)

    def _train_reason_group(self, reason: str) -> str:
        text = str(reason or "").lower()
        if text.startswith("feature_dim_upgrade") or text.startswith("architecture_upgrade"):
            return "feature_upgrade"
        return "train"

    def _train_cooldown_for_reason(self, reason: str) -> float:
        grp = self._train_reason_group(reason)
        if grp == "feature_upgrade":
            return max(0.0, float(self.feature_upgrade_cooldown_s or 0.0))
        return max(0.0, float(self.train_cooldown_s or 0.0))

    def _schedule_background_training(
        self,
        *,
        symbol: str,
        timeframe: str,
        lookback: int,
        train_df: pd.DataFrame,
        model_path: str,
        reason: str,
    ) -> Tuple[bool, str]:
        if ModelTrainer is None:
            return False, "model_trainer_unavailable"
        self._cleanup_train_tasks()
        if model_path in self._train_tasks and not self._train_tasks[model_path].done():
            return False, "training_in_progress"

        group = self._train_reason_group(reason)
        now_ts = time.time()
        cd = self._train_cooldown_for_reason(reason)
        key = (model_path, group)
        last_ts = float(self._train_last_attempt_ts.get(key, 0.0) or 0.0)
        if cd > 0 and (now_ts - last_ts) < cd:
            remain = max(0.0, cd - (now_ts - last_ts))
            return False, f"training_cooldown_{remain:.1f}s"
        self._train_last_attempt_ts[key] = now_ts

        if len(train_df) < (int(lookback) + 50):
            return False, "insufficient_rows_for_training"

        train_df_copy = train_df.copy()
        reason_text = str(reason)
        if group == "feature_upgrade":
            self._feature_upgrade_pending.add(model_path)

        async def _runner() -> None:
            start_ts = time.time()
            ok = False
            try:
                async with self._train_sem:
                    trainer = ModelTrainer(
                        symbol=symbol,
                        timeframe=timeframe,
                        input_lookback=int(lookback),
                        agent_name=self.name,
                        model_manager=self.model_manager,
                    )
                    loop = asyncio.get_running_loop()
                    ok = bool(
                        await asyncio.wait_for(
                            loop.run_in_executor(None, trainer.train_model, train_df_copy),
                            timeout=max(1.0, float(self.train_timeout_s)),
                        )
                    )
            except asyncio.TimeoutError:
                self.logger.warning(
                    "[%s] background training timeout for %s (%s) after %.1fs",
                    self.name,
                    symbol,
                    reason_text,
                    float(self.train_timeout_s),
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                self.logger.exception(
                    "[%s] background training exception for %s (%s)",
                    self.name,
                    symbol,
                    reason_text,
                )
            finally:
                self._feature_upgrade_pending.discard(model_path)

            if ok:
                self.model_cache.pop(model_path, None)
                self._model_mtime.pop(model_path, None)
                self._predict_fns = {k: v for k, v in self._predict_fns.items() if k[0] != model_path}
                if group == "feature_upgrade":
                    self._retrained_feature_models.add(model_path)
                self.logger.info(
                    "[%s] background training completed for %s (%s) in %.2fs",
                    self.name,
                    symbol,
                    reason_text,
                    max(0.0, time.time() - start_ts),
                )
            else:
                self.logger.warning(
                    "[%s] background training failed for %s (%s)",
                    self.name,
                    symbol,
                    reason_text,
                )

        task = asyncio.create_task(_runner(), name=f"{self.name}:train:{symbol}:{group}")
        self._train_tasks[model_path] = task
        self.logger.warning(
            "[%s] queued background training for %s (%s)",
            self.name,
            symbol,
            reason_text,
        )
        return True, "training_queued"

    def _has_gpu(self) -> bool:
        try:
            import tensorflow as tf  # local import: keep module import cheap
            return bool(tf.config.list_physical_devices("GPU"))
        except Exception:
            return False

    def _candle_ts_ms(self, candle: Any) -> int:
        try:
            ts = None
            if isinstance(candle, dict):
                ts = candle.get("timestamp")
                if ts is None:
                    ts = candle.get("ts")
                if ts is None and isinstance(candle.get("k"), dict):
                    ts = candle["k"].get("t")
            elif isinstance(candle, (list, tuple)) and len(candle) >= 1:
                ts = candle[0]
            if ts is None:
                return 0
            val = float(ts)
            if val <= 0:
                return 0
            if val < 1e12:
                val *= 1000.0
            return int(val)
        except Exception:
            return 0

    def _dedupe_sort_ohlcv(self, rows: List[Any]) -> List[Any]:
        if not rows:
            return []
        by_ts: Dict[int, Any] = {}
        for row in rows:
            ts_ms = self._candle_ts_ms(row)
            if ts_ms <= 0:
                continue
            by_ts[ts_ms] = row
        if not by_ts:
            return []
        return [by_ts[k] for k in sorted(by_ts.keys())]

    async def _fetch_exchange_ohlcv_batch(
        self,
        symbol: str,
        timeframe: str,
        limit: int,
        end_time_ms: Optional[int] = None,
    ) -> List[Any]:
        ec = self.exchange_client or getattr(self.market_data_feed, "exchange_client", None)
        getter = getattr(ec, "get_ohlcv", None) if ec is not None else None
        if not callable(getter):
            return []
        lim = max(1, int(limit))

        def _candidates(include_end_time: bool) -> List[Dict[str, Any]]:
            base = [
                {"symbol": symbol, "interval": timeframe, "limit": lim},
                {"symbol": symbol, "timeframe": timeframe, "limit": lim},
                {"symbol": symbol, "tf": timeframe, "limit": lim},
            ]
            if include_end_time and end_time_ms is not None:
                for item in base:
                    item["end_time"] = int(end_time_ms)
                base.append({"symbol": symbol, "interval": timeframe, "limit": lim, "endTime": int(end_time_ms)})
            return base

        tried_with_end = False
        for kwargs in _candidates(include_end_time=True):
            try:
                tried_with_end = tried_with_end or ("end_time" in kwargs or "endTime" in kwargs)
                res = getter(**kwargs)
                res = await res if asyncio.iscoroutine(res) else res
                return list(res or []) if isinstance(res, list) else []
            except TypeError:
                continue
            except Exception:
                self.logger.debug(
                    "[%s] get_ohlcv failed for %s kwargs=%s",
                    self.name,
                    symbol,
                    kwargs,
                    exc_info=True,
                )
                return []

        # Positional fallback.
        try:
            res = getter(symbol, timeframe, lim)
            res = await res if asyncio.iscoroutine(res) else res
            return list(res or []) if isinstance(res, list) else []
        except Exception:
            if tried_with_end:
                self.logger.debug(
                    "[%s] get_ohlcv positional fallback failed for %s",
                    self.name,
                    symbol,
                    exc_info=True,
                )
            return []

    async def _fetch_training_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        *,
        target_rows: int,
        max_rows: int,
    ) -> List[Any]:
        target = max(1, int(target_rows))
        cap = max(target, int(max_rows))
        collected: List[Any] = []
        end_time_ms: Optional[int] = None

        for _ in range(24):
            if len(collected) >= cap:
                break
            req = min(1000, max(100, cap - len(collected)))
            batch = await self._fetch_exchange_ohlcv_batch(
                symbol=symbol,
                timeframe=timeframe,
                limit=req,
                end_time_ms=end_time_ms,
            )
            if not batch:
                break
            cleaned = self._dedupe_sort_ohlcv(batch)
            if not cleaned:
                break

            prior_end = end_time_ms
            oldest_batch_ts = self._candle_ts_ms(cleaned[0])
            collected = self._dedupe_sort_ohlcv(cleaned + collected)
            if len(collected) > cap:
                collected = collected[-cap:]

            if len(cleaned) < req or oldest_batch_ts <= 0:
                break

            next_end = int(oldest_batch_ts - 1)
            if prior_end is not None and next_end >= prior_end:
                # Endpoint likely ignores pagination markers; avoid infinite looping.
                break
            end_time_ms = next_end

        # Fallback: merge with in-memory market data for symbols/timeframes already hydrated.
        if len(collected) < target:
            live_ohlcv = await self._get_market_data_safe(symbol, timeframe)
            if isinstance(live_ohlcv, list) and live_ohlcv:
                collected = self._dedupe_sort_ohlcv(collected + list(live_ohlcv))
                if len(collected) > cap:
                    collected = collected[-cap:]

        return list(collected[-cap:]) if collected else []

    def _build_train_df_from_ohlcv(self, ohlcv: List[Any], row_cap: int) -> Optional[pd.DataFrame]:
        feature_df = self._build_edge_feature_frame(ohlcv)
        if feature_df is None or feature_df.empty:
            return None
        train_cols = self._training_feature_columns()
        train_df = feature_df[train_cols].copy()
        cap = max(1, int(row_cap))
        if len(train_df) > cap:
            train_df = train_df.tail(cap).copy().reset_index(drop=True)
        return train_df

    def _load_model_metadata(self, model_path: str) -> Dict[str, Any]:
        try:
            import pickle

            base = Path(str(model_path))
            metadata_path = base.with_name(f"{base.stem}_metadata.pkl")
            if not metadata_path.exists() or metadata_path.stat().st_size <= 0:
                return {}
            with metadata_path.open("rb") as f:
                payload = pickle.load(f)
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    def _model_val_accuracy_from_metadata(self, metadata: Optional[Dict[str, Any]]) -> Optional[float]:
        if not isinstance(metadata, dict) or not metadata:
            return None
        candidates = [
            metadata.get("model_val_accuracy"),
            (metadata.get("training_metrics") or {}).get("val_accuracy"),
            (metadata.get("last_train_metrics") or {}).get("val_accuracy"),
        ]
        for val in candidates:
            if val is None:
                continue
            try:
                return float(val)
            except Exception:
                continue
        return None

    def _passes_retrain_quality_guard(
        self,
        train_result: Dict[str, Any],
        *,
        has_existing_model: bool,
        prior_val_accuracy: Optional[float] = None,
    ) -> Tuple[bool, str]:
        val_acc_raw = train_result.get("val_accuracy")
        if val_acc_raw is None:
            return False, "missing_val_accuracy"
        try:
            val_acc = float(val_acc_raw)
        except Exception:
            return False, "invalid_val_accuracy"
        if val_acc < float(self._retrain_min_val_acc):
            return False, f"val_accuracy_below_guard:{val_acc:.4f}<{float(self._retrain_min_val_acc):.4f}"
        if not has_existing_model:
            return True, "no_existing_model_threshold_passed"
        baseline = 0.0
        if prior_val_accuracy is not None:
            try:
                baseline = float(prior_val_accuracy)
            except Exception:
                baseline = 0.0
        if val_acc <= baseline:
            return False, f"no_improvement:{val_acc:.4f}<={baseline:.4f}"
        return True, "quality_ok"

    def _refresh_model_cache_for_path(self, model_path: str) -> bool:
        self.model_cache.pop(model_path, None)
        self._model_mtime.pop(model_path, None)
        self._predict_fns = {k: v for k, v in self._predict_fns.items() if k[0] != model_path}
        reloaded = self.model_manager.safe_load_model(model_path) if self.model_manager else None
        if reloaded is None:
            return False
        self.model_cache[model_path] = reloaded
        try:
            self._model_mtime[model_path] = float(os.path.getmtime(model_path))
        except Exception:
            self._model_mtime[model_path] = time.time()
        self._retrained_feature_models.add(model_path)
        return True

    def _effective_adaptive_epochs(self) -> int:
        epochs = max(3, min(5, int(self._retrain_default_epochs or 3)))
        if not self._has_gpu():
            epochs = min(epochs, max(3, int(self._retrain_cpu_epoch_cap or 3)))
        return max(3, min(5, int(epochs)))

    def _effective_full_train_epochs(self) -> int:
        epochs = max(15, min(20, int(self._full_train_epochs or 15)))
        if not self._has_gpu():
            epochs = min(epochs, max(15, int(self._full_train_cpu_epoch_cap or 15)))
        return max(15, min(20, int(epochs)))

    def _is_full_train_due(self, symbol: str, model_path: str, now_ts: float, force_startup: bool = False) -> bool:
        if not self.model_manager:
            return False
        if not self.model_manager.model_exists(model_path):
            return True

        metadata = self._load_model_metadata(model_path)
        prev_val_acc = self._model_val_accuracy_from_metadata(metadata)
        if prev_val_acc is not None and float(prev_val_acc) < float(self._retrain_min_val_acc):
            return True

        if force_startup and self._full_train_force_first_boot and model_path not in self._startup_full_train_done:
            return True

        interval = float(self._full_train_interval_s or 0.0)
        if interval <= 0:
            return False

        sym = str(symbol or "").upper()
        last_full_ts = float(self._full_train_last_ts.get(sym, 0.0) or 0.0)
        try:
            last_full_ts = max(last_full_ts, float(os.path.getmtime(model_path)))
        except Exception:
            pass
        return (now_ts - last_full_ts) >= interval

    def _schedule_startup_full_training(
        self,
        *,
        symbol: str,
        timeframe: str,
        lookback: int,
        model_path: str,
        reason: str,
    ) -> Tuple[bool, str]:
        if not self._full_train_on_startup:
            return False, "full_training_startup_disabled"
        if ModelTrainer is None:
            return False, "model_trainer_unavailable"

        self._cleanup_train_tasks()
        if model_path in self._train_tasks and not self._train_tasks[model_path].done():
            return False, "training_in_progress"
        if model_path in self._startup_full_train_pending:
            return False, "full_training_pending"
        if model_path in self._startup_full_train_done and not self._full_train_force_first_boot:
            return False, "full_training_already_done"

        key = (model_path, "full_bootstrap")
        now_ts = time.time()
        cd = max(60.0, float(self.train_cooldown_s or 0.0))
        last_ts = float(self._train_last_attempt_ts.get(key, 0.0) or 0.0)
        if cd > 0 and (now_ts - last_ts) < cd:
            remain = max(0.0, cd - (now_ts - last_ts))
            return False, f"training_cooldown_{remain:.1f}s"
        self._train_last_attempt_ts[key] = now_ts
        self._startup_full_train_pending.add(model_path)

        async def _runner() -> None:
            start_ts = time.time()
            reason_text = str(reason)
            sym = str(symbol or "").upper()
            try:
                async with self._train_sem:
                    has_existing_model = bool(self.model_manager and self.model_manager.model_exists(model_path))
                    prior_metadata = self._load_model_metadata(model_path) if has_existing_model else {}
                    prior_val_acc = self._model_val_accuracy_from_metadata(prior_metadata)
                    target_rows = max(int(self._full_train_target_rows), int(lookback) + 50)
                    min_rows_required = max(int(self._full_train_min_rows), int(lookback) + 50)
                    ohlcv = await self._fetch_training_ohlcv(
                        sym,
                        timeframe,
                        target_rows=target_rows,
                        max_rows=int(self._full_train_max_rows),
                    )
                    if not ohlcv or len(ohlcv) < min_rows_required:
                        self.logger.warning(
                            "[%s] full training skipped for %s (%s): insufficient_history rows=%d need>=%d",
                            self.name,
                            sym,
                            reason_text,
                            int(len(ohlcv) if isinstance(ohlcv, list) else 0),
                            int(min_rows_required),
                        )
                        return

                    train_df = self._build_train_df_from_ohlcv(ohlcv, int(self._full_train_max_rows))
                    if train_df is None or train_df.empty or len(train_df) < (int(lookback) + 50):
                        self.logger.warning(
                            "[%s] full training skipped for %s (%s): insufficient_features rows=%d",
                            self.name,
                            sym,
                            reason_text,
                            int(len(train_df) if train_df is not None else 0),
                        )
                        return

                    epochs = self._effective_full_train_epochs()
                    trainer = ModelTrainer(
                        symbol=sym,
                        timeframe=timeframe,
                        input_lookback=int(lookback),
                        epochs=int(epochs),
                        agent_name=self.name,
                        model_manager=self.model_manager,
                    )
                    loop = asyncio.get_running_loop()
                    result = await asyncio.wait_for(
                        loop.run_in_executor(
                            None,
                            lambda: trainer.train_model(
                                train_df,
                                task="supervised_learning",
                                epochs=int(epochs),
                                max_rows=int(self._full_train_max_rows),
                                save_model_artifact=False,
                                return_metrics=True,
                            ),
                        ),
                        timeout=max(1.0, float(self.train_timeout_s)),
                    )
                    if not isinstance(result, dict):
                        result = {"ok": bool(result), "reason": "legacy_result"}
                    if not bool(result.get("ok")):
                        self.logger.warning(
                            "[%s] full training failed for %s (%s): %s",
                            self.name,
                            sym,
                            reason_text,
                            str(result.get("reason", "train_failed")),
                        )
                        return

                    guard_ok, guard_reason = self._passes_retrain_quality_guard(
                        result,
                        has_existing_model=has_existing_model,
                        prior_val_accuracy=prior_val_acc,
                    )
                    if not guard_ok:
                        self.logger.warning(
                            "[%s] full training candidate rejected for %s (%s): %s",
                            self.name,
                            sym,
                            reason_text,
                            guard_reason,
                        )
                        return

                    if not trainer.persist_model(model_path=model_path):
                        self.logger.warning(
                            "[%s] full training save failed for %s (%s)",
                            self.name,
                            sym,
                            reason_text,
                        )
                        return
                    if not self._refresh_model_cache_for_path(model_path):
                        self.logger.error(
                            "[%s] full training reload failed for %s (%s)",
                            self.name,
                            sym,
                            reason_text,
                        )
                        return

                    self._full_train_last_ts[sym] = time.time()
                    self._startup_full_train_done.add(model_path)
                    self.logger.info(
                        "[%s] full training completed for %s (%s) in %.2fs rows=%d epochs=%d val_acc=%s",
                        self.name,
                        sym,
                        reason_text,
                        max(0.0, time.time() - start_ts),
                        int(result.get("rows", 0) or 0),
                        int(result.get("epochs", 0) or 0),
                        str(result.get("val_accuracy")),
                    )
            except asyncio.TimeoutError:
                self.logger.warning(
                    "[%s] full training timeout for %s (%s) after %.1fs",
                    self.name,
                    sym,
                    reason_text,
                    float(self.train_timeout_s),
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                self.logger.exception(
                    "[%s] full training exception for %s (%s)",
                    self.name,
                    sym,
                    reason_text,
                )
            finally:
                self._startup_full_train_pending.discard(model_path)

        task = asyncio.create_task(_runner(), name=f"{self.name}:full_train:{symbol}")
        self._train_tasks[model_path] = task
        return True, "full_training_queued"

    async def retrain(self) -> Dict[str, Any]:
        """
        Phase 9-compatible async retrain entrypoint.
        Runs blocking training in an executor, then explicitly reloads saved models.
        """
        start_ts = time.time()
        summary: Dict[str, Any] = {
            "ok": True,
            "symbols_total": 0,
            "success": 0,
            "failed": 0,
            "skipped": 0,
            "guard_rejected": 0,
            "full_runs": 0,
            "adaptive_runs": 0,
            "duration_sec": 0.0,
        }

        if ModelTrainer is None:
            self.logger.warning("[%s:Retrain] skipped: model_trainer_unavailable", self.name)
            summary["ok"] = False
            summary["duration_sec"] = float(max(0.0, time.time() - start_ts))
            return summary

        if not self.model_manager:
            self.logger.warning("[%s:Retrain] skipped: missing model_manager", self.name)
            summary["ok"] = False
            summary["duration_sec"] = float(max(0.0, time.time() - start_ts))
            return summary

        if self._retrain_lock.locked():
            self.logger.info("[%s:Retrain] skipped: prior retrain still in progress", self.name)
            summary["skipped"] += 1
            summary["duration_sec"] = float(max(0.0, time.time() - start_ts))
            return summary

        now_ts = time.time()
        if self._retrain_min_gap_s > 0 and (now_ts - float(self._retrain_last_end_ts or 0.0)) < self._retrain_min_gap_s:
            remain = max(0.0, self._retrain_min_gap_s - (now_ts - float(self._retrain_last_end_ts or 0.0)))
            self.logger.info(
                "[%s:Retrain] skipped: cooldown_active remaining=%.1fs",
                self.name,
                remain,
            )
            summary["skipped"] += 1
            summary["duration_sec"] = float(max(0.0, time.time() - start_ts))
            return summary

        async with self._retrain_lock:
            try:
                try:
                    symbols = list(self.symbols or [])
                    if not symbols:
                        symbols = await self._safe_get_symbols()
                    symbols = [str(s or "").upper() for s in symbols if str(s or "").strip()]
                except Exception:
                    symbols = []

                max_symbols = int(
                    self._cfg("ML_RETRAIN_MAX_SYMBOLS_PER_RUN", os.getenv("ML_RETRAIN_MAX_SYMBOLS_PER_RUN", 1)) or 1
                )
                max_symbols = max(1, max_symbols)
                if symbols:
                    cursor = int(self._retrain_rr_cursor % len(symbols))
                    ordered = symbols[cursor:] + symbols[:cursor]
                    symbols = ordered[:max_symbols]
                    self._retrain_rr_cursor = (cursor + max_symbols) % len(ordered)

                summary["symbols_total"] = len(symbols)
                if not symbols:
                    self.logger.info("[%s:Retrain] skipped: no symbols available", self.name)
                    summary["duration_sec"] = float(max(0.0, time.time() - start_ts))
                    return summary

                run_budget_s = max(
                    30.0,
                    float(self._retrain_run_budget_s or 0.0),
                )
                timeout_s = max(
                    30.0,
                    min(float(self.train_timeout_s or 300.0), float(self._retrain_symbol_timeout_s or 180.0)),
                )
                has_gpu = self._has_gpu()
                adaptive_epochs = self._effective_adaptive_epochs()
                full_epochs = self._effective_full_train_epochs()

                self.logger.info(
                    "[%s:Retrain] Retrain start: symbols=%d timeframe=%s timeout=%.1fs run_budget=%.1fs adaptive_rows=%d adaptive_epochs=%d full_rows_target=%d full_rows_cap=%d full_epochs=%d guard=val_acc>=%.2f device=%s",
                    self.name,
                    len(symbols),
                    str(self.timeframe),
                    timeout_s,
                    run_budget_s,
                    int(self._retrain_max_rows),
                    int(adaptive_epochs),
                    int(self._full_train_target_rows),
                    int(self._full_train_max_rows),
                    int(full_epochs),
                    float(self._retrain_min_val_acc),
                    "gpu" if has_gpu else "cpu",
                )

                loop = asyncio.get_running_loop()

                for idx, sym in enumerate(symbols):
                    if (time.time() - start_ts) >= run_budget_s:
                        remaining = max(0, len(symbols) - idx)
                        summary["skipped"] += remaining
                        self.logger.warning(
                            "[%s:Retrain] run budget exhausted after %.1fs; skipped_remaining=%d",
                            self.name,
                            max(0.0, time.time() - start_ts),
                            remaining,
                        )
                        break

                    per_start = time.time()
                    try:
                        tuned = {}
                        try:
                            tuned = load_tuned_params(f"{self.name}_{sym}_{self.timeframe}") or {}
                        except Exception:
                            tuned = {}
                        lookback = self._resolve_lookback(tuned)
                        if not self.model_manager:
                            self.logger.warning(f"[{self.name}] model_manager not available for {sym}")
                            continue
                        model_path = self.model_manager.build_model_path(
                            agent_name=self.name,
                            symbol=sym,
                            version=self.timeframe,
                        )
                        has_existing_model = bool(self.model_manager.model_exists(model_path))
                        prior_metadata = self._load_model_metadata(model_path) if has_existing_model else {}
                        prior_val_acc = self._model_val_accuracy_from_metadata(prior_metadata)
                        full_due = self._is_full_train_due(sym, model_path, time.time(), force_startup=False)

                        if full_due:
                            tier = "full"
                            target_rows = max(int(self._full_train_target_rows), int(lookback) + 50)
                            min_rows_required = max(int(self._full_train_min_rows), int(lookback) + 50)
                            ohlcv = await self._fetch_training_ohlcv(
                                sym,
                                self.timeframe,
                                target_rows=target_rows,
                                max_rows=int(self._full_train_max_rows),
                            )
                            if not isinstance(ohlcv, list) or len(ohlcv) < min_rows_required:
                                summary["skipped"] += 1
                                self.logger.info(
                                    "[%s:Retrain] Retrain finish symbol=%s tier=%s status=skipped reason=insufficient_ohlcv rows=%d need>=%d",
                                    self.name,
                                    sym,
                                    tier,
                                    int(len(ohlcv) if isinstance(ohlcv, list) else 0),
                                    int(min_rows_required),
                                )
                                continue
                            train_df = self._build_train_df_from_ohlcv(ohlcv, int(self._full_train_max_rows))
                            tier_epochs = int(full_epochs)
                            tier_max_rows = int(self._full_train_max_rows)
                        else:
                            tier = "adaptive"
                            ohlcv = await self._get_market_data_safe(sym, self.timeframe)
                            if not isinstance(ohlcv, list) or len(ohlcv) < (lookback + 5):
                                summary["skipped"] += 1
                                self.logger.info(
                                    "[%s:Retrain] Retrain finish symbol=%s tier=%s status=skipped reason=insufficient_ohlcv",
                                    self.name,
                                    sym,
                                    tier,
                                )
                                continue
                            train_df = self._build_train_df_from_ohlcv(ohlcv, int(self._retrain_max_rows))
                            tier_epochs = int(adaptive_epochs)
                            tier_max_rows = int(self._retrain_max_rows)

                        if train_df is None or train_df.empty or len(train_df) < (lookback + 50):
                            summary["skipped"] += 1
                            self.logger.info(
                                "[%s:Retrain] Retrain finish symbol=%s tier=%s status=skipped reason=insufficient_features rows=%d",
                                self.name,
                                sym,
                                tier,
                                int(len(train_df) if train_df is not None else 0),
                            )
                            continue

                        self.logger.info(
                            "[%s:Retrain] Retrain start symbol=%s tier=%s lookback=%d rows=%d epochs=%d path=%s",
                            self.name,
                            sym,
                            tier,
                            lookback,
                            int(len(train_df)),
                            int(tier_epochs),
                            str(model_path),
                        )

                        trainer = ModelTrainer(
                            symbol=sym,
                            timeframe=self.timeframe,
                            input_lookback=int(lookback),
                            epochs=int(tier_epochs),
                            agent_name=self.name,
                            model_manager=self.model_manager,
                        )
                        result = await asyncio.wait_for(
                            loop.run_in_executor(
                                None,
                                lambda: trainer.train_model(
                                    train_df,
                                    task="supervised_learning",
                                    epochs=int(tier_epochs),
                                    max_rows=int(tier_max_rows),
                                    save_model_artifact=False,
                                    return_metrics=True,
                                ),
                            ),
                            timeout=timeout_s,
                        )
                        if not isinstance(result, dict):
                            result = {"ok": bool(result), "reason": "legacy_result"}

                        if not bool(result.get("ok")):
                            summary["failed"] += 1
                            self.logger.warning(
                                "[%s:Retrain] Retrain finish symbol=%s tier=%s status=failed reason=%s duration=%.2fs",
                                self.name,
                                sym,
                                tier,
                                str(result.get("reason", "train_failed")),
                                max(0.0, time.time() - per_start),
                            )
                            continue

                        guard_ok, guard_reason = self._passes_retrain_quality_guard(
                            result,
                            has_existing_model=has_existing_model,
                            prior_val_accuracy=prior_val_acc,
                        )
                        if not guard_ok:
                            summary["guard_rejected"] += 1
                            summary["skipped"] += 1
                            self.logger.warning(
                                "[%s:Retrain] Retrain finish symbol=%s tier=%s status=discarded reason=%s val_acc=%s threshold=%.2f",
                                self.name,
                                sym,
                                tier,
                                guard_reason,
                                str(result.get("val_accuracy")),
                                float(self._retrain_min_val_acc),
                            )
                            continue

                        if not trainer.persist_model(model_path=model_path):
                            summary["failed"] += 1
                            self.logger.warning(
                                "[%s:Retrain] Retrain finish symbol=%s tier=%s status=failed reason=model_save_failed",
                                self.name,
                                sym,
                                tier,
                            )
                            continue

                        if not self._refresh_model_cache_for_path(model_path):
                            summary["failed"] += 1
                            self.logger.error(
                                "[%s:Retrain] Retrain finish symbol=%s tier=%s status=failed reason=model_reload_failed",
                                self.name,
                                sym,
                                tier,
                            )
                            continue

                        if tier == "full":
                            self._full_train_last_ts[sym] = time.time()
                            self._startup_full_train_done.add(model_path)
                            summary["full_runs"] += 1
                        else:
                            summary["adaptive_runs"] += 1

                        summary["success"] += 1
                        self.logger.info(
                            "[%s:Retrain] Retrain finish symbol=%s tier=%s status=ok duration=%.2fs val_acc=%s",
                            self.name,
                            sym,
                            tier,
                            max(0.0, time.time() - per_start),
                            str(result.get("val_accuracy")),
                        )

                    except asyncio.TimeoutError:
                        summary["failed"] += 1
                        self.logger.warning(
                            "[%s:Retrain] Retrain finish symbol=%s status=timeout timeout=%.1fs",
                            self.name,
                            sym,
                            timeout_s,
                        )
                    except asyncio.CancelledError:
                        self.logger.warning(
                            "[%s:Retrain] Retrain finish symbol=%s status=cancelled",
                            self.name,
                            sym,
                        )
                        raise
                    except Exception as e:
                        summary["failed"] += 1
                        self.logger.exception(
                            "[%s:Retrain] Retrain finish symbol=%s status=error err=%s",
                            self.name,
                            sym,
                            str(e),
                        )

                summary["duration_sec"] = float(max(0.0, time.time() - start_ts))
                self.logger.info(
                    "[%s:Retrain] Retrain finish: symbols=%d success=%d failed=%d skipped=%d guard_rejected=%d full_runs=%d adaptive_runs=%d duration=%.2fs",
                    self.name,
                    int(summary.get("symbols_total", 0) or 0),
                    int(summary.get("success", 0) or 0),
                    int(summary.get("failed", 0) or 0),
                    int(summary.get("skipped", 0) or 0),
                    int(summary.get("guard_rejected", 0) or 0),
                    int(summary.get("full_runs", 0) or 0),
                    int(summary.get("adaptive_runs", 0) or 0),
                    float(summary.get("duration_sec", 0.0) or 0.0),
                )
                return summary
            finally:
                self._retrain_last_end_ts = time.time()

    def _std_row(self, r) -> Optional[List[float]]:
        """
        Accept either short-key (o/h/l/c/v) or long-key (open/high/low/close/volume) dicts,
        or a sequence ending with [open, high, low, close, volume].
        """
        try:
            if isinstance(r, dict):
                d = r.get("k", r)
                o = d.get("o", d.get("open"))
                h = d.get("h", d.get("high"))
                l = d.get("l", d.get("low"))
                c = d.get("c", d.get("close", d.get("last", d.get("price"))))
                v = d.get("v", d.get("volume"))
                if None in (o, h, l, c, v):
                    return None
                return [float(o), float(h), float(l), float(c), float(v)]
            else:
                seq = list(r)
                if len(seq) >= 6:
                    seq = seq[-5:]  # keep last 5 numbers
                if len(seq) == 5:
                    return [float(x) for x in seq]
        except Exception:
            return None
        return None

    def _to_ohlcv_dataframe(self, ohlcv: List[Any]) -> pd.DataFrame:
        rows: List[Dict[str, float]] = []
        for candle in (ohlcv or []):
            std = self._std_row(candle)
            if std is None:
                continue
            rows.append(
                {
                    "timestamp": self._extract_row_timestamp(candle),
                    "open": float(std[0]),
                    "high": float(std[1]),
                    "low": float(std[2]),
                    "close": float(std[3]),
                    "volume": float(std[4]),
                }
            )
        if not rows:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        df = pd.DataFrame(rows)
        for col in self._legacy_feature_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=["open", "high", "low", "close"])
        if df.empty:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0).clip(lower=0.0)
        return df.reset_index(drop=True)

    def _rolling_zscore(
        self,
        series: pd.Series,
        window: int,
        min_periods: Optional[int] = None,
        clip: float = 8.0,
    ) -> pd.Series:
        w = max(2, int(window))
        mp = max(2, int(min_periods or max(5, w // 4)))
        s = pd.to_numeric(series, errors="coerce")
        mean = s.rolling(w, min_periods=mp).mean()
        std = s.rolling(w, min_periods=mp).std(ddof=0).replace(0.0, np.nan)
        z = (s - mean) / std
        z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return z.clip(lower=-abs(float(clip)), upper=abs(float(clip)))

    def _rsi_normalized(self, close: pd.Series, period: int = 14) -> pd.Series:
        p = max(2, int(period))
        delta = close.diff()
        gains = delta.clip(lower=0.0)
        losses = (-delta).clip(lower=0.0)
        avg_gain = gains.ewm(alpha=1.0 / p, adjust=False, min_periods=p).mean()
        avg_loss = losses.ewm(alpha=1.0 / p, adjust=False, min_periods=p).mean()
        rs = avg_gain / (avg_loss + 1e-12)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        # Center and scale to roughly [-1, 1] for stationarity.
        return ((rsi - 50.0) / 50.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1.0, 1.0)

    def _build_edge_feature_frame(self, ohlcv: List[Any]) -> pd.DataFrame:
        """
        Build a regime-aware, volatility-aware feature frame.
        Output keeps legacy OHLCV columns and adds engineered edge features.
        """
        df = self._to_ohlcv_dataframe(ohlcv)
        if df.empty:
            return df

        close = pd.to_numeric(df["close"], errors="coerce")
        high = pd.to_numeric(df["high"], errors="coerce")
        low = pd.to_numeric(df["low"], errors="coerce")
        volume = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0).clip(lower=0.0)

        returns_1 = close.pct_change(1)
        returns_3 = close.pct_change(3)
        returns_5 = close.pct_change(5)

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(14, min_periods=3).mean()
        atr_pct = atr / close.replace(0.0, np.nan)

        rv20 = returns_1.rolling(20, min_periods=5).std(ddof=0)
        rv50 = returns_1.rolling(50, min_periods=10).std(ddof=0)
        volatility_ratio = rv20 / rv50.replace(0.0, np.nan)

        ema9 = close.ewm(span=9, adjust=False).mean()
        ema21 = close.ewm(span=21, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()
        ema9_dist = (close - ema9) / close.replace(0.0, np.nan)
        ema21_dist = (close - ema21) / close.replace(0.0, np.nan)
        ema50_dist = (close - ema50) / close.replace(0.0, np.nan)
        ema_slope = ema21.pct_change(3)
        ema_cross_dist = (ema9 - ema21) / close.replace(0.0, np.nan)

        rsi14 = self._rsi_normalized(close, period=14)
        roc10 = close.pct_change(10)

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        macd_hist = (macd - macd_signal) / close.replace(0.0, np.nan)

        range_pct = (high - low) / close.replace(0.0, np.nan)
        range_zscore = self._rolling_zscore(range_pct, window=50)
        range_base = range_pct.rolling(20, min_periods=5).mean()
        rolling_range_expansion = range_pct / range_base.replace(0.0, np.nan)

        volume_zscore = self._rolling_zscore(volume, window=50)
        vol_base = volume.rolling(20, min_periods=5).mean()
        volume_spike_ratio = volume / vol_base.replace(0.0, np.nan)

        volatility_zscore = self._rolling_zscore(atr_pct, window=50)
        trend_strength = (close - ema50).abs() / close.replace(0.0, np.nan)

        high_vol_flag = (atr_pct >= float(self._regime_vol_high_pct)).astype(float)
        sideways_flag = (atr_pct <= float(self._regime_vol_low_pct)).astype(float)
        trend_flag = ((high_vol_flag < 0.5) & (sideways_flag < 0.5)).astype(float)

        feature_map = {
            "returns_1": returns_1,
            "returns_3": returns_3,
            "returns_5": returns_5,
            "atr_pct": atr_pct,
            "realized_vol_20": rv20,
            "realized_vol_50": rv50,
            "volatility_ratio": volatility_ratio,
            "volatility_zscore": volatility_zscore,
            "ema9_dist": ema9_dist,
            "ema21_dist": ema21_dist,
            "ema50_dist": ema50_dist,
            "ema_slope": ema_slope,
            "ema_cross_dist": ema_cross_dist,
            "rsi14": rsi14,
            "roc10": roc10,
            "macd_hist": macd_hist,
            "volume_zscore": volume_zscore,
            "volume_spike_ratio": volume_spike_ratio,
            "range_zscore": range_zscore,
            "rolling_range_expansion": rolling_range_expansion,
            "trend_strength": trend_strength,
            "trend_flag": trend_flag,
            "sideways_flag": sideways_flag,
            "high_vol_flag": high_vol_flag,
        }
        for col, values in feature_map.items():
            df[col] = pd.to_numeric(values, errors="coerce")

        for col in self._edge_feature_columns:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            if col.endswith("_flag"):
                df[col] = df[col].clip(lower=0.0, upper=1.0)
            else:
                df[col] = df[col].clip(lower=-12.0, upper=12.0)

        for col in self._legacy_feature_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=["open", "high", "low", "close"])
        if df.empty:
            return pd.DataFrame(columns=["timestamp"] + self._legacy_feature_columns + self._edge_feature_columns)
        df["volume"] = df["volume"].fillna(0.0).clip(lower=0.0)
        return df.reset_index(drop=True)

    def _training_feature_columns(self) -> List[str]:
        """Canonical training/inference column order used for full-feature models."""
        return list(dict.fromkeys(self._legacy_feature_columns + self._edge_feature_columns))

    def _model_input_shape(self, model: Any) -> Tuple[Optional[int], Optional[int]]:
        try:
            shape = getattr(model, "input_shape", None)
            if isinstance(shape, list) and shape:
                shape = shape[0]
            if not shape or len(shape) < 3:
                return None, None
            lookback = int(shape[-2]) if shape[-2] is not None else None
            dim = int(shape[-1]) if shape[-1] is not None else None
            return lookback, dim
        except Exception:
            return None, None

    def _model_input_lookback(self, model: Any) -> Optional[int]:
        lookback, _ = self._model_input_shape(model)
        return lookback

    def _model_input_feature_count(self, model: Any) -> Optional[int]:
        _, dim = self._model_input_shape(model)
        return dim

    def _model_has_expected_architecture(self, model: Any) -> bool:
        """Check if model has the expected GRU architecture (not old LSTM)."""
        try:
            if model is None:
                return False
            # Check if model has GRU layers (new architecture) vs LSTM layers (old)
            has_gru = any("gru" in layer.name.lower() for layer in model.layers)
            has_lstm = any("lstm" in layer.name.lower() for layer in model.layers)
            # If it has GRU, it's good. If it has LSTM but no GRU, it's old architecture.
            return has_gru or not has_lstm
        except Exception:
            return False

    def _resolve_input_columns_for_model(self, expected_dim: Optional[int]) -> Tuple[List[str], str]:
        full_cols = self._training_feature_columns()
        full_dim = len(full_cols)
        edge_dim = len(self._edge_feature_columns)
        legacy_dim = len(self._legacy_feature_columns)

        if expected_dim is None or expected_dim <= 0:
            return list(full_cols), "full_default"
        if expected_dim == full_dim:
            return list(full_cols), "full_ohlcv_edge"
        if expected_dim == edge_dim:
            return list(self._edge_feature_columns), "edge_native"
        if expected_dim == legacy_dim:
            return list(self._legacy_feature_columns), "legacy_ohlcv"
        if expected_dim < full_dim:
            # Best-effort compatibility for previously customized shapes.
            return list(full_cols[:expected_dim]), f"full_truncated_{expected_dim}"
        return [], f"unsupported_dim_{expected_dim}"

    def _build_model_input_tensor(
        self,
        feature_df: pd.DataFrame,
        lookback: int,
        input_columns: List[str],
    ) -> Optional[np.ndarray]:
        lb = max(2, int(lookback))
        if feature_df is None or feature_df.empty or len(feature_df) < lb:
            return None
        cols = list(input_columns or self._edge_feature_columns)
        for col in cols:
            if col not in feature_df.columns:
                feature_df[col] = 0.0
        window = feature_df[cols].tail(lb).copy()
        window = window.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        arr = np.asarray(window.values, dtype=np.float32)
        if arr.shape != (lb, len(cols)):
            return None
        return arr.reshape((1, lb, len(cols)))

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64)
        if arr.size == 0:
            return arr
        arr = arr - np.max(arr)
        exp_arr = np.exp(arr)
        denom = float(np.sum(exp_arr))
        if denom <= 0:
            return np.zeros_like(exp_arr, dtype=np.float64)
        return exp_arr / denom

    def _decode_model_output(self, y_raw: np.ndarray) -> Tuple[str, float, np.ndarray, str]:
        """
        Decode model outputs to (action, confidence, normalized_probs, schema).

        Supported schemas:
        - 1 value: scalar buy score/probability -> HOLD/BUY
        - 2 values: RL/DQN Q-values [HOLD, BUY] -> HOLD/BUY
        - >=3 values: class logits/probs [BUY, SELL, HOLD] (first 3 used)
        """
        arr = np.asarray(y_raw, dtype=np.float64).reshape(-1)
        if arr.size == 0:
            return "hold", 0.0, np.asarray([], dtype=np.float64), "empty"

        if arr.size == 1:
            val = float(arr[0])
            if 0.0 <= val <= 1.0:
                p_buy = val
            else:
                # Convert unconstrained score/logit to probability.
                p_buy = 1.0 / (1.0 + np.exp(-val))
            p_buy = float(np.clip(p_buy, 0.0, 1.0))
            action = "buy" if p_buy >= 0.5 else "hold"
            conf = float(max(p_buy, 1.0 - p_buy))
            probs = np.asarray([1.0 - p_buy, p_buy], dtype=np.float64)
            return action, conf, probs, "scalar_hold_buy"

        if arr.size == 2:
            # ModelTrainer defines Dense(2): [Q(HOLD), Q(BUY)].
            probs = self._softmax(arr)
            idx = int(np.argmax(probs))
            action = "hold" if idx == 0 else "buy"
            conf = float(np.clip(np.max(probs), 0.0, 1.0))
            return action, conf, probs, "q2_hold_buy"

        probs = self._softmax(arr[:3])
        idx = int(np.argmax(probs))
        action = ["buy", "sell", "hold"][idx]
        conf = float(np.clip(np.max(probs), 0.0, 1.0))
        return action, conf, probs, "cls3_buy_sell_hold"

    def _round_trip_cost_pct(self) -> float:
        taker_bps = float(fee_bps(self.shared_state, "taker") or 10.0)
        slippage_bps = float(
            self._cfg("EXIT_SLIPPAGE_BPS", self._cfg("CR_PRICE_SLIPPAGE_BPS", 0.0)) or 0.0
        )
        return ((taker_bps * 2.0) + slippage_bps) / 10000.0

    def _confidence_bucket_floor(self, conf: float, bucket_size: Optional[float] = None) -> float:
        size = float(bucket_size or self._conf_bucket_size or 0.05)
        size = max(0.01, min(0.50, size))
        c = max(0.0, min(0.999999, float(conf or 0.0)))
        idx = int(c / size)
        return round(idx * size, 4)

    def _magnitude_aware_confidence(self, base_confidence: float, expected_move_pct: float) -> float:
        """
        PHASE 2: Magnitude-aware confidence scoring.
        
        Boosts confidence based on magnitude of expected move relative to threshold.
        Model learns magnitude implicitly through this feedback mechanism.
        
        Formula: adjusted_conf = base_conf * min(1.0 + magnitude_factor, MAX_BOOST)
        Where magnitude_factor = (expected_move_pct - threshold) / threshold
        
        Example:
        - threshold=0.15%, expected_move=0.15% → magnitude_factor=0.0 → adjusted=base_conf
        - threshold=0.15%, expected_move=0.30% → magnitude_factor=1.0 → adjusted=base_conf*1.5
        - threshold=0.15%, expected_move=0.45% → magnitude_factor=2.0 → capped at base_conf*2.0
        """
        if not self._cfg("ML_MAGNITUDE_CONFIDENCE_ENABLED", True):
            return float(base_confidence)
        
        try:
            base_threshold = float(self._cfg("ML_MAGNITUDE_BASE_THRESHOLD", 0.0015) or 0.0015)
            multiplier = float(self._cfg("ML_MAGNITUDE_MULTIPLIER", 1.5) or 1.5)
            
            # Ensure we have positive expected move
            move = float(expected_move_pct or 0.0)
            if move <= base_threshold:
                return float(base_confidence)  # No boost if below threshold
            
            # Calculate magnitude factor (how much move exceeds threshold)
            magnitude_factor = (move - base_threshold) / max(base_threshold, 1e-9)
            
            # Boost confidence (capped at 2.0x multiplier to avoid extreme confidence)
            boost = min(multiplier, 1.0 + magnitude_factor)
            adjusted_conf = float(base_confidence) * float(boost)
            
            # Clip to valid range [0, 1]
            adjusted_conf = max(0.0, min(1.0, adjusted_conf))
            
            self.logger.debug(
                "[%s] magnitude_aware_conf: base=%.3f move=%.5f threshold=%.5f factor=%.2f boost=%.2f adjusted=%.3f",
                self.name,
                base_confidence,
                move,
                base_threshold,
                magnitude_factor,
                boost,
                adjusted_conf,
            )
            
            return float(adjusted_conf)
        except Exception as e:
            self.logger.debug("[%s] magnitude_aware_confidence failed: %s", self.name, e)
            return float(base_confidence)

    async def _safe_current_price(self, symbol: str) -> Optional[float]:
        try:
            p = float((getattr(self.shared_state, "latest_prices", {}) or {}).get(symbol, 0.0) or 0.0)
            if p > 0:
                return p
        except Exception:
            pass
        try:
            safe_price = getattr(self.shared_state, "safe_price", None)
            if callable(safe_price):
                res = safe_price(symbol, default=0.0)
                res = await res if asyncio.iscoroutine(res) else res
                p = float(res or 0.0)
                if p > 0:
                    return p
        except Exception:
            pass
        return None

    def _extract_row_timestamp(self, row: Any) -> Optional[float]:
        try:
            ts = None
            if isinstance(row, dict):
                ts = row.get("timestamp")
                if ts is None:
                    k = row.get("k")
                    if isinstance(k, dict):
                        ts = k.get("t")
            elif isinstance(row, (list, tuple)) and len(row) >= 1:
                ts = row[0]
            if ts is None:
                return None
            val = float(ts)
            if val > 1e12:
                val /= 1000.0
            return val
        except Exception:
            return None

    def _timeframe_seconds(self, timeframe: str) -> float:
        tf = str(timeframe or self.timeframe or "5m").lower()
        if tf.endswith("m"):
            try:
                return float(int(tf[:-1]) * 60)
            except Exception:
                return 300.0
        if tf.endswith("h"):
            try:
                return float(int(tf[:-1]) * 3600)
            except Exception:
                return 3600.0
        if tf.endswith("d"):
            return 86400.0
        return 300.0

    def _infer_candle_seconds(self, ohlcv: List[Any], fallback_tf: str) -> float:
        ts_vals: List[float] = []
        for row in ohlcv[-40:]:
            ts = self._extract_row_timestamp(row)
            if ts:
                ts_vals.append(ts)
        if len(ts_vals) >= 2:
            diffs = [b - a for a, b in zip(ts_vals[:-1], ts_vals[1:]) if b > a]
            if diffs:
                med = float(np.median(np.asarray(diffs, dtype=np.float64)))
                if med > 0:
                    return med
        return self._timeframe_seconds(fallback_tf)

    def _calc_signal_pnl_pct(self, action: str, entry_price: float, exit_price: float, net_cost_pct: float) -> Tuple[float, float]:
        a = str(action or "").upper()
        if entry_price <= 0 or exit_price <= 0:
            return 0.0, 0.0
        if a == "BUY":
            gross = (exit_price - entry_price) / entry_price
        elif a == "SELL":
            gross = (entry_price - exit_price) / entry_price
        else:
            gross = 0.0
        net = gross - float(net_cost_pct or 0.0)
        return float(gross), float(net)

    def _normalize_regime(self, regime: Any) -> str:
        r = str(regime or "").strip().lower()
        if r in {"bull", "bullish", "risk_on"}:
            return "bull"
        if r in {"bear", "bearish", "risk_off"}:
            return "bear"
        if r in {"high", "high_vol", "volatile", "volatility_high", "extreme", "extreme_vol", "crisis"}:
            return "high_vol"
        if r in {"sideways", "chop", "range", "low_vol", "flat", "compression"}:
            return "sideways"
        if r in {"trend", "trending"}:
            return "trend"
        if r in {"uptrend"}:
            return "bull"
        if r in {"downtrend"}:
            return "bear"
        if r in {"normal", "neutral", "medium", "mid_vol"}:
            return "normal"
        return "normal"

    def _atr_from_numeric_rows(self, rows: List[List[float]], lookback: int = 14) -> float:
        try:
            if len(rows) < lookback + 1:
                return 0.0
            window = rows[-(lookback + 1):]
            prev_c = float(window[0][3])
            trs: List[float] = []
            for i in range(1, len(window)):
                h = float(window[i][1])
                l = float(window[i][2])
                tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
                trs.append(tr)
                prev_c = float(window[i][3])
            if not trs:
                return 0.0
            return float(np.mean(np.asarray(trs, dtype=np.float64)))
        except Exception:
            return 0.0

    def _rv_pct_from_numeric_rows(self, rows: List[List[float]], lookback: int = 20) -> float:
        try:
            if len(rows) < lookback + 1:
                return 0.0
            closes = np.asarray([float(r[3]) for r in rows[-(lookback + 1):]], dtype=np.float64)
            if np.any(closes <= 0):
                return 0.0
            rets = (closes[1:] / closes[:-1]) - 1.0
            if rets.size < 2:
                return 0.0
            return float(np.std(rets, ddof=1))
        except Exception:
            return 0.0

    def _infer_regime_from_numeric_rows(self, rows: List[List[float]]) -> str:
        if not rows:
            return "normal"
        close = float(rows[-1][3] or 0.0)
        if close <= 0:
            return "normal"
        atr = self._atr_from_numeric_rows(rows, lookback=14)
        atr_pct = (atr / close) if atr > 0 else 0.0
        low = max(1e-6, float(self._regime_vol_low_pct))
        high = max(low + 1e-6, float(self._regime_vol_high_pct))
        if atr_pct >= high:
            return "high_vol"
        if atr_pct <= low:
            return "sideways"
        return "trend"

    def _estimate_expected_move_pct_from_rows(self, rows: List[List[float]], horizon_steps: int) -> float:
        """
        Better expected move estimator:
        blend ATR-based move scaling + realized-vol scaling for the target horizon.
        """
        if not rows:
            return float(self._expected_move_fallback_pct)
        close = float(rows[-1][3] or 0.0)
        if close <= 0:
            return float(self._expected_move_fallback_pct)
        atr = self._atr_from_numeric_rows(rows, lookback=14)
        atr_pct = (atr / close) if atr > 0 else 0.0
        rv_pct = self._rv_pct_from_numeric_rows(rows, lookback=20)
        h = max(1.0, float(horizon_steps))
        h_sqrt = float(np.sqrt(h))
        atr_move = atr_pct * max(1.0, min(3.0, h_sqrt))
        rv_move = rv_pct * h_sqrt
        expected = (0.6 * atr_move) + (0.4 * rv_move)
        if expected <= 0:
            expected = float(self._expected_move_fallback_pct)
        expected = max(float(self._expected_move_min_pct), min(float(self._expected_move_max_pct), float(expected)))
        return float(expected)

    def _derive_expected_move_pct(self, records: List[Dict[str, Any]]) -> float:
        vals: List[float] = []
        for rec in records:
            em = float(rec.get("expected_move_pct", 0.0) or 0.0)
            if em <= 0:
                em = abs(float(rec.get("gross_move_pct", 0.0) or 0.0))
            if em > 0:
                vals.append(em)
        if not vals:
            return float(self._expected_move_fallback_pct)
        med = float(np.median(np.asarray(vals, dtype=np.float64)))
        return max(float(self._expected_move_min_pct), min(float(self._expected_move_max_pct), med))

    def _expected_move_calibration_multiplier(self, regime: str) -> float:
        if not self._expected_move_calibration_enabled:
            return 1.0
        rg = self._normalize_regime(regime)
        global_mult = float(self._expected_move_calib_global or 1.0)
        regime_mult = float(self._expected_move_calib_by_regime.get(rg, global_mult) or global_mult)
        regime_count = int(self._expected_move_calib_counts.get(rg, 0) or 0)
        if regime_count < int(self._expected_move_calib_min_samples):
            mult = global_mult
        else:
            # Regime estimate dominates after enough samples; blend still keeps continuity.
            mult = (0.7 * regime_mult) + (0.3 * global_mult)
        return max(
            float(self._expected_move_calib_min_mult),
            min(float(self._expected_move_calib_max_mult), float(mult)),
        )

    def _update_expected_move_calibration(self, matured: List[Dict[str, Any]]) -> None:
        if not self._expected_move_calibration_enabled or not matured:
            return
        alpha = max(0.01, min(1.0, float(self._expected_move_calib_alpha or 0.15)))
        for rec in matured:
            expected = float(rec.get("expected_move_pct", 0.0) or 0.0)
            realized = abs(float(rec.get("gross_move_pct", 0.0) or 0.0))
            if expected <= 0.0 or realized <= 0.0:
                continue
            ratio = realized / max(expected, 1e-9)
            ratio = max(float(self._expected_move_calib_min_mult), min(float(self._expected_move_calib_max_mult), ratio))

            prev_global = float(self._expected_move_calib_global or 1.0)
            self._expected_move_calib_global = ((1.0 - alpha) * prev_global) + (alpha * ratio)
            self._expected_move_calib_global = max(
                float(self._expected_move_calib_min_mult),
                min(float(self._expected_move_calib_max_mult), float(self._expected_move_calib_global)),
            )

            rg = self._normalize_regime(rec.get("regime", "normal"))
            prev_rg = float(self._expected_move_calib_by_regime.get(rg, self._expected_move_calib_global))
            new_rg = ((1.0 - alpha) * prev_rg) + (alpha * ratio)
            new_rg = max(float(self._expected_move_calib_min_mult), min(float(self._expected_move_calib_max_mult), float(new_rg)))
            self._expected_move_calib_by_regime[rg] = float(new_rg)
            self._expected_move_calib_counts[rg] = int(self._expected_move_calib_counts.get(rg, 0) or 0) + 1

    def _institutional_regime_gate(
        self,
        *,
        action: str,
        regime: str,
        confidence: float,
        required_conf: float,
        expected_move_pct: float,
        round_trip_cost_ev_pct: float,
    ) -> Tuple[bool, float, str]:
        if not self._institutional_regime_filter_enabled:
            return True, float(required_conf), "inst_filter_disabled"
        if str(action or "").upper() != "BUY":
            return True, float(required_conf), "inst_filter_not_buy"

        rg = self._normalize_regime(regime)
        conf_floor = float(required_conf)
        if rg == "bear" and self._inst_block_bear_buy:
            return False, conf_floor, "regime_block_bear"

        regime_conf_floor = {
            "high_vol": float(self._inst_high_vol_min_conf),
            "sideways": float(self._inst_sideways_min_conf),
            "normal": float(self._inst_normal_min_conf),
            "trend": float(self._inst_normal_min_conf),
            "bull": float(self._inst_bull_min_conf),
        }.get(rg, float(self._inst_normal_min_conf))
        conf_floor = max(float(conf_floor), float(regime_conf_floor))

        regime_ev_mult = {
            "high_vol": float(self._inst_high_vol_ev_mult),
            "sideways": float(self._inst_sideways_ev_mult),
            "normal": float(self._inst_normal_ev_mult),
            "trend": float(self._inst_normal_ev_mult),
            "bull": float(self._inst_bull_ev_mult),
        }.get(rg, float(self._inst_normal_ev_mult))
        min_move = max(0.0, float(round_trip_cost_ev_pct) * float(regime_ev_mult))

        if float(expected_move_pct or 0.0) < float(min_move):
            return False, conf_floor, f"inst_move_floor:{float(expected_move_pct or 0.0):.5f}<{float(min_move):.5f}"
        if float(confidence or 0.0) < float(conf_floor):
            return False, conf_floor, f"inst_conf_floor:{float(confidence or 0.0):.4f}<{float(conf_floor):.4f}"
        return True, conf_floor, "inst_regime_ok"

    def _scale_expected_move_by_regime(
        self,
        base_expected_move_pct: float,
        regime: str,
        horizon_minutes: float,
    ) -> float:
        """
        Scale expected move based on regime and trading horizon.
        
        Professional recommendation:
        - Bull (60m): Market moving, wider expected moves available
        - Normal (120m): Standard horizon, moderate expected moves
        - Bear (disabled): Market stalled, no positive EV for longs
        
        Longer horizon → need wider expected move (sqrt scaling)
        Shorter horizon → narrower expected move acceptable
        
        Args:
            base_expected_move_pct: Raw expected move (e.g., 0.0065)
            regime: Market regime (bull, normal, bear, high_vol, low_vol)
            horizon_minutes: Target horizon in minutes
        
        Returns:
            Regime-scaled expected move
        """
        regime_norm = self._normalize_regime(regime)
        
        # Get target horizon for this regime
        target_horizon = self._regime_horizon_map.get(regime_norm, 120.0)
        
        # If market is in a regime we shouldn't trade (e.g., bear),
        # return 0 to signal "disabled"
        if target_horizon >= 9999.0:
            return 0.0
        
        # Scale expected move based on horizon ratio
        # longer_horizon → need wider move
        # Formula: scaled_move = base_move × sqrt(horizon_ratio)
        if target_horizon > 0 and horizon_minutes > 0:
            horizon_ratio = float(target_horizon) / max(1.0, float(horizon_minutes))
            horizon_scale = float(np.sqrt(max(0.1, horizon_ratio)))
        else:
            horizon_scale = 1.0
        
        scaled = float(base_expected_move_pct) * float(horizon_scale)
        
        # Clamp to min/max
        return max(
            float(self._expected_move_min_pct),
            min(float(self._expected_move_max_pct), scaled)
        )

    def _derive_regime_views(
        self,
        records: List[Dict[str, Any]],
        bucket_size: float,
        round_trip_cost_pct: float,
    ) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        by_regime: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for rec in records:
            rg = self._normalize_regime(rec.get("regime", "normal"))
            by_regime[rg].append(rec)
        min_samples = max(10, int(self._conf_min_total_samples * 0.25))
        for rg, arr in by_regime.items():
            if len(arr) < min_samples:
                continue
            rows = self._build_bucket_rows(arr, bucket_size=bucket_size)
            if not rows:
                continue
            expected_move_pct = self._derive_expected_move_pct(arr)
            deriv = self._derive_required_confidence(rows, expected_move_pct, round_trip_cost_pct)
            out[rg] = {
                "rows": rows,
                "derivation": deriv,
                "samples": len(arr),
                "expected_move_pct": expected_move_pct,
            }
        return out

    def _required_conf_for_regime(self, regime: str) -> float:
        rg = self._normalize_regime(regime)
        cfg_map = {}
        try:
            cfg_map = (getattr(self.shared_state, "dynamic_config", {}) or {}).get("ML_DYNAMIC_REQUIRED_CONF_BY_REGIME", {}) or {}
        except Exception:
            cfg_map = {}
        merged: Dict[str, float] = dict(self._dynamic_required_conf_by_regime or {})
        if isinstance(cfg_map, dict):
            for k, v in cfg_map.items():
                try:
                    merged[self._normalize_regime(k)] = float(v)
                except Exception:
                    pass
        base_floor = float(self.min_conf)
        chosen = None
        if rg in merged:
            chosen = float(merged[rg])
        elif "trend" in merged:
            chosen = float(merged["trend"])
        else:
            chosen = base_floor

        # Risk-first guard: high-vol regime must never reduce required floor.
        if rg in {"high_vol"}:
            chosen = max(chosen, base_floor)
        return max(self._conf_floor_min, min(self._conf_floor_max, float(chosen)))

    async def _live_regime_and_expected_move(self, symbol: str) -> Tuple[str, float]:
        regime = "normal"
        try:
            get_regime = getattr(self.shared_state, "get_volatility_regime", None)
            if callable(get_regime):
                rr = get_regime(symbol, timeframe=self.timeframe)
                rr = await rr if asyncio.iscoroutine(rr) else rr
                if isinstance(rr, dict):
                    regime = self._normalize_regime(rr.get("regime", "normal"))
        except Exception:
            regime = "normal"

        expected_move_pct = float(self._expected_move_fallback_pct)
        try:
            ohlcv = await self._get_market_data_safe(symbol, self.timeframe)
            if isinstance(ohlcv, list):
                rows = [self._std_row(c) for c in ohlcv]
                rows = [r for r in rows if r is not None]
                if rows:
                    candle_sec = max(1.0, self._infer_candle_seconds(ohlcv, self.timeframe))
                    
                    # Get regime-specific horizon (professional recommendation)
                    regime_norm = self._normalize_regime(regime)
                    target_horizon_min = self._regime_horizon_map.get(regime_norm, 120.0)
                    
                    # Check if regime should be disabled (bear regime)
                    if target_horizon_min >= 9999.0:
                        # Bear regime: no longs possible
                        return regime_norm, 0.0
                    
                    # Calculate horizon steps using regime-specific horizon
                    horizon_steps = max(1, int(round((float(target_horizon_min) * 60.0) / candle_sec)))
                    
                    # Estimate base expected move
                    base_expected_move = self._estimate_expected_move_pct_from_rows(rows, horizon_steps)
                    
                    # Scale by regime
                    expected_move_pct = self._scale_expected_move_by_regime(
                        base_expected_move,
                        regime,
                        float(target_horizon_min),
                    )

                    if regime == "normal":
                        regime = self._infer_regime_from_numeric_rows(rows)
                    regime_norm = self._normalize_regime(regime)
                    calib_mult = self._expected_move_calibration_multiplier(regime_norm)
                    expected_move_pct = float(expected_move_pct) * float(calib_mult)
                    expected_move_pct = max(
                        float(self._expected_move_min_pct),
                        min(float(self._expected_move_max_pct), float(expected_move_pct)),
                    )
        except Exception:
            expected_move_pct = float(self._expected_move_fallback_pct)

        return self._normalize_regime(regime), float(expected_move_pct)

    def _build_bucket_rows(self, records: List[Dict[str, Any]], bucket_size: Optional[float] = None) -> List[Dict[str, Any]]:
        grouped: Dict[float, List[Dict[str, Any]]] = defaultdict(list)
        for rec in records:
            conf = float(rec.get("confidence", 0.0) or 0.0)
            bf = self._confidence_bucket_floor(conf, bucket_size=bucket_size)
            grouped[bf].append(rec)

        rows: List[Dict[str, Any]] = []
        size = float(bucket_size or self._conf_bucket_size or 0.05)
        for low in sorted(grouped.keys()):
            arr = grouped[low]
            if not arr:
                continue
            sample_count = len(arr)
            wins = sum(1 for x in arr if int(x.get("outcome", 0) or 0) > 0)
            avg_net = float(np.mean(np.asarray([float(x.get("net_pnl_pct", 0.0) or 0.0) for x in arr], dtype=np.float64)))
            avg_gross = float(np.mean(np.asarray([abs(float(x.get("gross_move_pct", 0.0) or 0.0)) for x in arr], dtype=np.float64)))
            avg_expected_move = float(
                np.mean(
                    np.asarray(
                        [
                            float(x.get("expected_move_pct", abs(float(x.get("gross_move_pct", 0.0) or 0.0)) or 0.0) or 0.0)
                            for x in arr
                        ],
                        dtype=np.float64,
                    )
                )
            )
            mean_conf = float(np.mean(np.asarray([float(x.get("confidence", 0.0) or 0.0) for x in arr], dtype=np.float64)))
            rows.append(
                {
                    "bucket_low": float(low),
                    "bucket_high": float(min(1.0, low + size)),
                    "samples": int(sample_count),
                    "wins": int(wins),
                    "win_rate": float(wins / sample_count) if sample_count else 0.0,
                    "avg_net_pnl_pct": avg_net,
                    "avg_gross_move_pct": avg_gross,
                    "avg_expected_move_pct": avg_expected_move,
                    "mean_conf": mean_conf,
                }
            )
        return rows

    def _derive_required_confidence(self, rows: List[Dict[str, Any]], expected_move_pct: float, round_trip_cost_pct: float) -> Dict[str, Any]:
        if expected_move_pct <= 0:
            break_even_prob = 1.0
        else:
            break_even_prob = max(0.0, min(1.0, round_trip_cost_pct / expected_move_pct))

        min_ev_conf = None
        min_cal_conf = None
        for row in rows:
            if int(row.get("samples", 0) or 0) < self._conf_min_bucket_samples:
                continue
            if float(row.get("avg_net_pnl_pct", 0.0) or 0.0) > 0 and min_ev_conf is None:
                min_ev_conf = float(row.get("bucket_low", 0.0) or 0.0)
            if float(row.get("win_rate", 0.0) or 0.0) >= break_even_prob and min_cal_conf is None:
                min_cal_conf = float(row.get("bucket_low", 0.0) or 0.0)

        fallback = max(0.0, min(1.0, break_even_prob))
        required_conf = max(
            fallback,
            float(min_ev_conf if min_ev_conf is not None else fallback),
            float(min_cal_conf if min_cal_conf is not None else fallback),
        )
        required_conf = max(self._conf_floor_min, min(self._conf_floor_max, required_conf))

        return {
            "break_even_prob": float(break_even_prob),
            "min_positive_ev_conf": min_ev_conf,
            "min_calibrated_conf": min_cal_conf,
            "required_conf": float(required_conf),
        }

    async def _apply_dynamic_required_conf(
        self,
        required_conf: float,
        derivation: Dict[str, Any],
        source: str,
        regime_views: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        alpha = max(0.01, min(1.0, float(self._conf_floor_ema_alpha or 0.35)))
        if self._dynamic_required_conf is None:
            smoothed = float(required_conf)
        else:
            smoothed = ((1.0 - alpha) * float(self._dynamic_required_conf)) + (alpha * float(required_conf))
        smoothed = max(self._conf_floor_min, min(self._conf_floor_max, smoothed))
        self._dynamic_required_conf = smoothed

        be_by_regime: Dict[str, float] = {}
        if isinstance(regime_views, dict):
            for rg, view in regime_views.items():
                try:
                    req = float(((view or {}).get("derivation") or {}).get("required_conf", smoothed))
                    rg_key = self._normalize_regime(rg)
                    prev = float(self._dynamic_required_conf_by_regime.get(rg_key, req) or req)
                    rg_smoothed = ((1.0 - alpha) * prev) + (alpha * req)
                    rg_smoothed = max(self._conf_floor_min, min(self._conf_floor_max, rg_smoothed))
                    self._dynamic_required_conf_by_regime[rg_key] = rg_smoothed
                    be_by_regime[rg_key] = float(
                        (((view or {}).get("derivation") or {}).get("break_even_prob", 0.0) or 0.0)
                    )
                except Exception:
                    continue

        min_ev_conf = derivation.get("min_positive_ev_conf")
        min_cal_conf = derivation.get("min_calibrated_conf")
        update_payload = {
            "ML_DYNAMIC_REQUIRED_CONF": float(smoothed),
            "ML_DYNAMIC_REQUIRED_CONF_BY_REGIME": dict(self._dynamic_required_conf_by_regime),
            "ML_BREAK_EVEN_PROB": float(derivation.get("break_even_prob", 0.0) or 0.0),
            "ML_BREAK_EVEN_PROB_BY_REGIME": be_by_regime,
            "ML_MIN_POSITIVE_EV_CONF": float(min_ev_conf) if min_ev_conf is not None else -1.0,
            "ML_MIN_CALIBRATED_CONF": float(min_cal_conf) if min_cal_conf is not None else -1.0,
            "ML_CONF_DERIVATION_SOURCE": str(source),
            "ML_CONF_HORIZON_MIN": float(self._conf_horizon_min),
            "ML_CONF_LAST_RECALIBRATED_TS": time.time(),
        }
        try:
            update_dynamic = getattr(self.shared_state, "update_dynamic_config", None)
            if callable(update_dynamic):
                res = update_dynamic(update_payload)
                if asyncio.iscoroutine(res):
                    await res
            else:
                if not hasattr(self.shared_state, "dynamic_config") or getattr(self.shared_state, "dynamic_config", None) is None:
                    self.shared_state.dynamic_config = {}
                self.shared_state.dynamic_config.update(update_payload)
        except Exception:
            self.logger.debug("[%s] failed to apply dynamic confidence floor", self.name, exc_info=True)

        self.logger.info(
            "[%s:ConfFloor] source=%s required=%.4f (break_even=%.4f min_ev=%s min_cal=%s)",
            self.name,
            source,
            smoothed,
            float(derivation.get("break_even_prob", 0.0) or 0.0),
            f"{float(derivation.get('min_positive_ev_conf')):.4f}" if derivation.get("min_positive_ev_conf") is not None else "None",
            f"{float(derivation.get('min_calibrated_conf')):.4f}" if derivation.get("min_calibrated_conf") is not None else "None",
        )

    def _write_confidence_report_artifacts(
        self,
        rows: List[Dict[str, Any]],
        derivation: Dict[str, Any],
        source: str,
        regime_views: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        if not rows:
            return
        out_dir = Path(self._conf_report_dir)
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            return

        csv_path = out_dir / "ml_forecaster_confidence_buckets.csv"
        try:
            with csv_path.open("w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "bucket_low",
                        "bucket_high",
                        "samples",
                        "wins",
                        "win_rate",
                        "mean_conf",
                        "avg_gross_move_pct",
                        "avg_expected_move_pct",
                        "avg_net_pnl_pct",
                    ],
                )
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
        except Exception:
            self.logger.debug("[%s] failed writing confidence bucket CSV", self.name, exc_info=True)

        if isinstance(regime_views, dict) and regime_views:
            regime_csv = out_dir / "ml_forecaster_confidence_buckets_by_regime.csv"
            try:
                with regime_csv.open("w", newline="") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=[
                            "regime",
                            "bucket_low",
                            "bucket_high",
                            "samples",
                            "wins",
                            "win_rate",
                            "mean_conf",
                            "avg_gross_move_pct",
                            "avg_expected_move_pct",
                            "avg_net_pnl_pct",
                        ],
                    )
                    writer.writeheader()
                    for rg, view in regime_views.items():
                        rg_norm = self._normalize_regime(rg)
                        for row in (view or {}).get("rows", []) or []:
                            rr = dict(row)
                            rr["regime"] = rg_norm
                            writer.writerow(rr)
            except Exception:
                self.logger.debug("[%s] failed writing regime confidence CSV", self.name, exc_info=True)

        if not self._conf_plot_enabled:
            return
        try:
            import matplotlib.pyplot as plt  # type: ignore

            xs = [float(r.get("mean_conf", 0.0) or 0.0) for r in rows]
            ys = [float(r.get("win_rate", 0.0) or 0.0) for r in rows]
            weights = [max(1.0, float(r.get("samples", 1) or 1.0)) for r in rows]
            min_w = min(weights) if weights else 1.0
            max_w = max(weights) if weights else 1.0
            if max_w <= min_w:
                sizes = [50.0 for _ in weights]
            else:
                sizes = [40.0 + 160.0 * ((w - min_w) / (max_w - min_w)) for w in weights]

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(xs, ys, s=sizes, alpha=0.75, color="#1f77b4", label="Empirical win rate")
            ax.plot([0.0, 1.0], [0.0, 1.0], "--", color="#9e9e9e", label="Perfect calibration")
            be = float(derivation.get("break_even_prob", 0.0) or 0.0)
            req = float(derivation.get("required_conf", 0.0) or 0.0)
            ax.axhline(be, color="#e74c3c", linestyle=":", linewidth=1.5, label=f"Break-even p={be:.3f}")
            ax.axvline(req, color="#2ca02c", linestyle=":", linewidth=1.5, label=f"Required conf={req:.3f}")
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            ax.set_xlabel("Predicted confidence")
            ax.set_ylabel("Observed win rate")
            ax.set_title(f"{self.name} calibration ({source})")
            ax.grid(alpha=0.2)
            ax.legend(loc="best")
            fig.tight_layout()
            fig.savefig(out_dir / "ml_forecaster_calibration_curve.png", dpi=150)
            plt.close(fig)
        except Exception:
            self.logger.debug("[%s] matplotlib calibration plot skipped", self.name, exc_info=True)

    def _log_backtest_rows(
        self,
        rows: List[Dict[str, Any]],
        records: List[Dict[str, Any]],
        source: str,
        derivation: Dict[str, Any],
        regime_views: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        self.logger.info("[%s:ConfBacktest] source=%s samples=%d", self.name, source, len(records))
        recent = records[-max(0, self._conf_print_rows):] if self._conf_print_rows > 0 else []
        for rec in recent:
            self.logger.info(
                "[%s:ConfSample] conf=%.3f outcome=%s horizon=%.1fmin net_pnl=%.3f%% exp_move=%.3f%% regime=%s symbol=%s action=%s",
                self.name,
                float(rec.get("confidence", 0.0) or 0.0),
                "WIN" if int(rec.get("outcome", 0) or 0) > 0 else "LOSS",
                float(rec.get("horizon_min", self._conf_horizon_min) or self._conf_horizon_min),
                float(rec.get("net_pnl_pct", 0.0) or 0.0) * 100.0,
                float(rec.get("expected_move_pct", 0.0) or 0.0) * 100.0,
                str(rec.get("regime", "normal")),
                str(rec.get("symbol", "")),
                str(rec.get("action", "")),
            )

        for row in rows:
            self.logger.info(
                "[%s:ConfBucket] %.2f-%.2f n=%d win=%.1f%% ev=%.3f%% move=%.3f%% exp=%.3f%%",
                self.name,
                float(row.get("bucket_low", 0.0)),
                float(row.get("bucket_high", 0.0)),
                int(row.get("samples", 0) or 0),
                float(row.get("win_rate", 0.0) or 0.0) * 100.0,
                float(row.get("avg_net_pnl_pct", 0.0) or 0.0) * 100.0,
                float(row.get("avg_gross_move_pct", 0.0) or 0.0) * 100.0,
                float(row.get("avg_expected_move_pct", 0.0) or 0.0) * 100.0,
            )
        self.logger.info(
            "[%s:ConfBacktest] required_conf=%.4f break_even=%.4f min_positive_ev=%s",
            self.name,
            float(derivation.get("required_conf", 0.0) or 0.0),
            float(derivation.get("break_even_prob", 0.0) or 0.0),
            f"{float(derivation.get('min_positive_ev_conf')):.4f}" if derivation.get("min_positive_ev_conf") is not None else "None",
        )
        if isinstance(regime_views, dict):
            for rg, view in regime_views.items():
                dd = (view or {}).get("derivation", {}) or {}
                self.logger.info(
                    "[%s:ConfRegime] regime=%s samples=%d req=%.4f break_even=%.4f",
                    self.name,
                    self._normalize_regime(rg),
                    int((view or {}).get("samples", 0) or 0),
                    float(dd.get("required_conf", 0.0) or 0.0),
                    float(dd.get("break_even_prob", 0.0) or 0.0),
                )

    async def _finalize_matured_confidence_samples(self) -> int:
        if not self._conf_eval_enabled or not self._pending_conf_samples:
            return 0
        horizon_sec = max(60.0, float(self._conf_horizon_min) * 60.0)
        now = time.time()
        matured: List[Dict[str, Any]] = []
        retained: deque = deque(maxlen=self._pending_conf_samples.maxlen)
        rt_cost_pct = self._round_trip_cost_pct()

        while self._pending_conf_samples:
            rec = self._pending_conf_samples.popleft()
            ts = float(rec.get("ts", 0.0) or 0.0)
            if ts <= 0 or (now - ts) < horizon_sec:
                retained.append(rec)
                continue
            symbol = str(rec.get("symbol") or "")
            action = str(rec.get("action") or "").upper()
            entry_price = float(rec.get("entry_price", 0.0) or 0.0)
            confidence = float(rec.get("confidence", 0.0) or 0.0)
            if not symbol or entry_price <= 0 or action not in {"BUY", "SELL"}:
                continue
            exit_price = await self._safe_current_price(symbol)
            if exit_price is None or exit_price <= 0:
                retained.append(rec)
                continue
            gross_pct, net_pct = self._calc_signal_pnl_pct(action, entry_price, float(exit_price), rt_cost_pct)
            expected_move_pct = float(rec.get("expected_move_pct", 0.0) or 0.0)
            if expected_move_pct <= 0:
                fallback = abs(gross_pct) if abs(gross_pct) > 0 else float(self._expected_move_fallback_pct)
                expected_move_pct = max(
                    float(self._expected_move_min_pct),
                    min(float(self._expected_move_max_pct), float(fallback)),
                )
            matured.append(
                {
                    "symbol": symbol,
                    "action": action,
                    "confidence": confidence,
                    "entry_price": entry_price,
                    "exit_price": float(exit_price),
                    "gross_move_pct": gross_pct,
                    "net_pnl_pct": net_pct,
                    "outcome": 1 if net_pct > 0 else 0,
                    "expected_move_pct": expected_move_pct,
                    "regime": self._normalize_regime(rec.get("regime", "normal")),
                    "horizon_min": float((now - ts) / 60.0),
                    "ts": ts,
                    "evaluated_ts": now,
                }
            )

        self._pending_conf_samples = retained
        for rec in matured:
            self._completed_conf_samples.append(rec)
        if matured:
            self._update_expected_move_calibration(matured)
        return len(matured)

    async def _maybe_recalibrate_confidence_floor(self, source: str = "live") -> Optional[Dict[str, Any]]:
        if not self._conf_eval_enabled:
            return None
        now = time.time()
        if (now - float(self._last_conf_recalc_ts or 0.0)) < self._conf_recalibrate_sec:
            return None
        if len(self._completed_conf_samples) < self._conf_min_total_samples:
            return None
        self._last_conf_recalc_ts = now
        records = list(self._completed_conf_samples)
        rows = self._build_bucket_rows(records, bucket_size=self._conf_bucket_size)
        if not rows:
            return None
        rt_cost_pct = self._round_trip_cost_pct()
        expected_move_pct = self._derive_expected_move_pct(records)
        derivation = self._derive_required_confidence(rows, expected_move_pct, rt_cost_pct)
        regime_views = self._derive_regime_views(records, self._conf_bucket_size, rt_cost_pct)
        await self._apply_dynamic_required_conf(
            float(derivation["required_conf"]),
            derivation,
            source=source,
            regime_views=regime_views,
        )
        self._log_backtest_rows(rows, records, source=source, derivation=derivation, regime_views=regime_views)
        self._write_confidence_report_artifacts(rows, derivation, source=source, regime_views=regime_views)
        return {"rows": rows, "records": records, "derivation": derivation, "regime_views": regime_views}

    async def _maybe_schedule_startup_backtest(self) -> None:
        if self._startup_backtest_submitted or self._startup_backtest_done:
            return
        if not self._conf_eval_enabled or not self._conf_backtest_on_startup:
            return
        if not self.symbols:
            return

        self._startup_backtest_submitted = True

        async def _runner() -> None:
            try:
                await self.backtest_confidence_buckets(
                    symbols=list(self.symbols),
                    horizon_min=self._conf_horizon_min,
                    max_samples_per_symbol=self._conf_backtest_max_per_symbol,
                    bucket_size=self._conf_bucket_size,
                    source="startup_backtest",
                )
            except Exception:
                self.logger.exception("[%s] startup confidence backtest failed", self.name)
            finally:
                self._startup_backtest_done = True

        self._startup_backtest_task = asyncio.create_task(_runner(), name=f"{self.name}:conf_backtest")

    async def backtest_confidence_buckets(
        self,
        symbols: Optional[List[str]] = None,
        horizon_min: Optional[float] = None,
        max_samples_per_symbol: Optional[int] = None,
        bucket_size: Optional[float] = None,
        source: str = "manual",
    ) -> Dict[str, Any]:
        """
        Historical confidence backtest:
        - Replays model predictions across available OHLCV windows
        - Computes outcome after N minutes and fee-adjusted net PnL
        - Buckets by confidence for calibration and EV analysis
        """
        if not self.model_manager:
            return {"ok": False, "reason": "missing_model_manager"}

        syms = [str(s).upper() for s in (symbols or self.symbols or [])]
        if not syms:
            syms = await self._safe_get_symbols()
        if not syms:
            return {"ok": False, "reason": "no_symbols"}

        hz_min = float(horizon_min or self._conf_horizon_min or 15.0)
        stride = max(1, int(self._conf_backtest_stride or 1))
        max_per_sym = int(max_samples_per_symbol or self._conf_backtest_max_per_symbol or 300)
        tf = self.timeframe
        rt_cost_pct = self._round_trip_cost_pct()

        records: List[Dict[str, Any]] = []
        for sym in syms:
            try:
                if not self.model_manager:
                    self.logger.warning(f"[{self.name}] model_manager not available for backtest of {sym}")
                    continue
                model_path = self.model_manager.build_model_path(agent_name=self.name, symbol=sym, version=tf)
                if not self.model_manager.model_exists(model_path):
                    continue
                try:
                    mtime = os.path.getmtime(model_path)
                except Exception:
                    mtime = 0.0
                model = self.model_cache.get(model_path)
                if model is None or self._model_mtime.get(model_path) != mtime:
                    model = self.model_manager.safe_load_model(model_path)
                    if model is None:
                        continue
                    self.model_cache[model_path] = model
                    self._model_mtime[model_path] = mtime

                tuned = {}
                try:
                    tuned = load_tuned_params(f"{self.name}_{sym}_{tf}") or {}
                except Exception:
                    tuned = {}
                lookback_cfg = self._resolve_lookback(tuned)
                expected_lookback = self._model_input_lookback(model)
                effective_lookback = (
                    int(expected_lookback)
                    if expected_lookback is not None and int(expected_lookback) > 1
                    else int(lookback_cfg)
                )

                ohlcv = await self._get_market_data_safe(sym, tf)
                if not isinstance(ohlcv, list) or len(ohlcv) < (effective_lookback + 5):
                    continue
                feature_df = self._build_edge_feature_frame(ohlcv)
                if feature_df is None or feature_df.empty or len(feature_df) < (effective_lookback + 5):
                    continue
                rows = [self._std_row(c) for c in ohlcv]
                rows = [r for r in rows if r is not None]
                if len(rows) < (effective_lookback + 5):
                    continue
                expected_dim = self._model_input_feature_count(model)
                input_cols, feature_mode = self._resolve_input_columns_for_model(expected_dim)
                if not input_cols:
                    self.logger.debug(
                        "[%s] backtest skipped %s due unsupported model feature dim=%s",
                        self.name,
                        sym,
                        str(expected_dim),
                    )
                    continue
                if expected_dim is not None and len(input_cols) != int(expected_dim):
                    self.logger.debug(
                        "[%s] backtest skipped %s due feature map mismatch expected=%s mapped=%d",
                        self.name,
                        sym,
                        str(expected_dim),
                        len(input_cols),
                    )
                    continue
                for col in input_cols:
                    if col not in feature_df.columns:
                        feature_df[col] = 0.0

                candle_sec = max(1.0, self._infer_candle_seconds(ohlcv, tf))
                horizon_steps = max(1, int(round((hz_min * 60.0) / candle_sec)))
                max_i = len(rows) - horizon_steps
                if max_i <= effective_lookback:
                    continue
                idxs = list(range(effective_lookback, max_i, stride))
                if max_per_sym > 0 and len(idxs) > max_per_sym:
                    idxs = idxs[-max_per_sym:]
                if not idxs:
                    continue

                windows: List[np.ndarray] = []
                valid_idxs: List[int] = []
                for i in idxs:
                    window = feature_df[input_cols].iloc[i - effective_lookback:i].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
                    if window.shape != (effective_lookback, len(input_cols)):
                        continue
                    windows.append(np.asarray(window, dtype=np.float32))
                    valid_idxs.append(i)
                if not windows:
                    continue
                X_batch = np.asarray(windows, dtype=np.float32)
                if X_batch.ndim != 3:
                    continue
                y_raw_batch = model.predict(X_batch, verbose=0)
                if y_raw_batch is None:
                    continue

                for j, i in enumerate(valid_idxs):
                    y_raw = np.asarray(y_raw_batch[j], dtype=np.float64)
                    action, confidence, probs, _schema = self._decode_model_output(y_raw)
                    if probs.size == 0:
                        continue
                    if action not in ("buy", "sell"):
                        continue
                    entry_price = float(rows[i - 1][3])  # close at prediction bar
                    exit_price = float(rows[i + horizon_steps - 1][3])  # close after horizon
                    if entry_price <= 0 or exit_price <= 0:
                        continue
                    context_rows = rows[:i]
                    regime = self._infer_regime_from_numeric_rows(context_rows)
                    expected_move_pct = self._estimate_expected_move_pct_from_rows(context_rows, horizon_steps)
                    gross_pct, net_pct = self._calc_signal_pnl_pct(action.upper(), entry_price, exit_price, rt_cost_pct)
                    records.append(
                        {
                            "symbol": sym,
                            "action": action.upper(),
                            "confidence": confidence,
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "gross_move_pct": gross_pct,
                            "net_pnl_pct": net_pct,
                            "outcome": 1 if net_pct > 0 else 0,
                            "expected_move_pct": expected_move_pct,
                            "regime": regime,
                            "horizon_min": hz_min,
                            "feature_mode": feature_mode,
                        }
                    )
            except Exception:
                self.logger.debug("[%s] confidence backtest failed for %s", self.name, sym, exc_info=True)

        if not records:
            return {"ok": False, "reason": "no_records"}

        rows = self._build_bucket_rows(records, bucket_size=bucket_size or self._conf_bucket_size)
        if not rows:
            return {"ok": False, "reason": "no_buckets"}

        expected_move_pct = self._derive_expected_move_pct(records)
        derivation = self._derive_required_confidence(rows, expected_move_pct, rt_cost_pct)
        regime_views = self._derive_regime_views(records, bucket_size or self._conf_bucket_size, rt_cost_pct)
        await self._apply_dynamic_required_conf(
            float(derivation["required_conf"]),
            derivation,
            source=source,
            regime_views=regime_views,
        )
        self._log_backtest_rows(rows, records, source=source, derivation=derivation, regime_views=regime_views)
        self._write_confidence_report_artifacts(rows, derivation, source=source, regime_views=regime_views)
        for rec in records[-self._conf_max_completed:]:
            self._completed_conf_samples.append(rec)

        return {
            "ok": True,
            "records": len(records),
            "rows": rows,
            "derivation": derivation,
            "horizon_min": hz_min,
            "source": source,
        }

    async def _collect_signal(self, symbol: str, action: str, confidence: float, reason: str, extras: Optional[Dict[str, Any]] = None):
        """
        ARCHITECTURAL FIX: Build signal dict instead of emitting to Meta.
        AgentManager will collect these and forward them.
        """
        if action.upper() not in ("BUY", "SELL"):
            return
            
        # DYNAMIC THRESHOLD (P9 Profit Feedback)
        # Check system-wide aggression to relax standards if behind profit target
        agg_factor = 1.0
        if hasattr(self.shared_state, "get_dynamic_param"):
            agg_factor = float(self.shared_state.get_dynamic_param("aggression_factor", 1.0))
        if float(confidence) < float(self._conf_hard_emit_floor):
            return

        regime, expected_move_pct = await self._live_regime_and_expected_move(symbol)

        # PHASE 2: Apply magnitude-aware confidence adjustment
        # Boosts confidence if expected move is significantly above threshold
        confidence = self._magnitude_aware_confidence(float(confidence), float(expected_move_pct or 0.0))

        # Hard EV gate: require expected_move >= 2x round-trip costs.
        round_trip_cost_pct = float(self._round_trip_cost_pct())
        buffer_bps = 0.0
        try:
            buffer_bps = float(self._cfg("TP_MIN_BUFFER_BPS", 0.0) or 0.0)
        except Exception:
            buffer_bps = 0.0
        round_trip_cost_ev_pct = float(round_trip_cost_pct) + (float(buffer_bps) / 10000.0)
        
        ev_positive = False
        if action.upper() == "BUY":
            # PHASE H: Dynamic EV Multiplier based on volatility regime
            base_mult = float(self._cfg("EV_HARD_SAFETY_MULT", 1.6) or 1.6)
            if self._cfg("EV_DYNAMIC_MULT_ENABLED", True):
                regime_lower = (regime or "").lower()
                if regime_lower in ("extreme", "extreme_vol", "crisis"):
                    ev_mult = float(self._cfg("EV_MULT_EXTREME_VOL", 2.2))
                elif regime_lower in ("high", "high_vol"):
                    ev_mult = float(self._cfg("EV_MULT_HIGH_VOL", 2.0))
                elif regime_lower in ("medium", "normal"):
                    ev_mult = float(self._cfg("EV_MULT_MED_VOL", 1.7))
                elif regime_lower in ("low", "compression"):
                    ev_mult = float(self._cfg("EV_MULT_LOW_VOL", 1.4))
                else:
                    ev_mult = float(self._cfg("EV_MULT_DEFAULT", base_mult))
            else:
                ev_mult = float(base_mult)
            required_move_pct = float(round_trip_cost_ev_pct) * float(ev_mult)
            ev_positive = float(expected_move_pct or 0.0) >= float(required_move_pct)
            if not ev_positive:
                self.logger.info(
                    "[%s] BUY suppressed for %s — expected_move %.4f%% < required %.4f%% "
                    "(mult=%.2f round_trip=%.4f%%).",
                    self.name,
                    symbol,
                    float(expected_move_pct) * 100.0,
                    float(required_move_pct) * 100.0,
                    float(ev_mult),
                    float(round_trip_cost_ev_pct) * 100.0,
                )
                return

        effective_min_conf = float(self._required_conf_for_regime(regime))
        if agg_factor > 1.0:
            effective_min_conf = max(self._conf_floor_min, effective_min_conf / max(1.0, agg_factor))
        logger.info(
            "[MLForecaster:EV] symbol=%s conf=%.3f expected_move=%.5f round_trip_cost=%.5f break_even=%.3f",
            symbol,
            confidence,
            expected_move_pct,
            round_trip_cost_ev_pct,
            max(0.0, min(1.0, round_trip_cost_ev_pct / max(float(expected_move_pct or 0.0), 1e-9))),
        )
        break_even_prob = max(
            0.0,
            min(
                1.0,
                float(round_trip_cost_ev_pct) / max(float(expected_move_pct or 0.0), 1e-9),
            ),
        )
        required_conf = max(effective_min_conf, break_even_prob)
        if self._sideways_conf_compression_enabled and str(regime) == "sideways":
            compressed_floor = max(
                float(break_even_prob),
                max(
                    float(self._conf_floor_min),
                    min(float(self._conf_floor_max), float(self._sideways_conf_compressed_floor)),
                ),
            )
            if compressed_floor < required_conf:
                self.logger.info(
                    "[MLForecaster:EV] sideways confidence compression applied: required %.3f -> %.3f (break_even=%.3f floor=%.3f)",
                    required_conf,
                    compressed_floor,
                    break_even_prob,
                    float(self._sideways_conf_compressed_floor),
                )
            required_conf = compressed_floor

        inst_ok, inst_required_conf, inst_reason = self._institutional_regime_gate(
            action=action,
            regime=regime,
            confidence=float(confidence),
            required_conf=float(required_conf),
            expected_move_pct=float(expected_move_pct),
            round_trip_cost_ev_pct=float(round_trip_cost_ev_pct),
        )
        required_conf = max(float(required_conf), float(inst_required_conf))
        if not inst_ok:
            self.logger.info(
                "[%s] BUY suppressed by institutional regime gate for %s (regime=%s reason=%s conf=%.3f req=%.3f exp_move=%.4f%%)",
                self.name,
                symbol,
                str(regime),
                str(inst_reason),
                float(confidence),
                float(required_conf),
                float(expected_move_pct) * 100.0,
            )
            return

        # Adaptive sideways regime handling: allow BUY if EV positive AND confidence >= required_conf
        if (
            action.upper() == "BUY"
            and bool(self._cfg("DISABLE_SIDEWAYS_REGIME_TRADING", True))
            and str(regime) == "sideways"
        ):
            if ev_positive and confidence >= required_conf:
                self.logger.info(
                    "[%s] BUY allowed in sideways regime for %s (EV positive, conf=%.3f >= required=%.3f)",
                    self.name,
                    symbol,
                    confidence,
                    required_conf,
                )
            else:
                self.logger.info(
                    "[%s] BUY suppressed in sideways regime for %s (EV=%s conf=%.3f < required=%.3f)",
                    self.name,
                    symbol,
                    "positive" if ev_positive else "negative",
                    confidence,
                    required_conf,
                )
                return

        if not await is_fresh(self.shared_state, symbol, max_age_sec=self.fresh_max_age_s):
            self.logger.debug(f"[{self.name}] Stale; skip {symbol}")
            return

        # Optional SELL guard
        if action.upper() == "SELL" and not self.allow_sell_without_position:
            try:
                if not await self._has_position(symbol):
                    self.logger.info(f"[{self.name}] Skip SELL for {symbol} — no position (signal suppressed).")
                    return
            except Exception:
                pass
                
        # Quote hint
        qh = getattr(
            self.config,
            "EMIT_BUY_QUOTE",
            getattr(self.config, "MIN_ENTRY_USDT", getattr(self.config, "DEFAULT_PLANNED_QUOTE", 25.0)),
        )
        try:
            if isinstance(qh, dict):
                qh = float(qh.get(symbol, qh.get("default", 25.0)))
            else:
                qh = float(qh)
        except Exception:
            qh = 25.0

        # Build signal dictionary first (contains all metadata including _expected_move_pct)
        signal = {
            "symbol": symbol,
            "action": action.upper(),
            "side": action.upper(),  # Alias for compatibility
            "confidence": max(0.0, min(1.0, float(confidence or 0.0))),
            "reason": reason,
            "quote": qh,
            "quote_hint": qh,  # Alias
            "horizon_hours": float(getattr(self.config, "DEFAULT_SIGNAL_HORIZON_H", 6.0)),
            "agent": self.name,
            "_required_conf": float(required_conf),
            "_confidence_gap": float(confidence - required_conf),
            "_tradeability_hint": "pass" if float(confidence) >= float(required_conf) else "below_required_conf",
            "_break_even_prob": float(break_even_prob),
            "_expected_move_pct": float(expected_move_pct),
            "_regime": str(regime),
        }
        if extras:
            signal.update(extras)

        # Mandatory P9 Signal Contract: Emit to Signal Bus with full signal dict (preserves _expected_move_pct)
        # PHASE 3 CONSOLIDATION: Use add_strategy_signal path (fallback to add_agent_signal removed)
        if hasattr(self.shared_state, "add_strategy_signal"):
            try:
                await self.shared_state.add_strategy_signal(symbol, signal)
            except Exception as e:
                self.logger.warning(f"[{self.name}] Failed to emit strategy signal: {e}")

        # GAP FIX A: Validate quote against min_notional BEFORE buffering
        # Prevents sub-5 USDT signals from reaching MetaController
        MIN_NOTIONAL_FLOOR = float(getattr(self.config, "MIN_NOTIONAL_FLOOR", 5.0))
        if signal["quote"] < MIN_NOTIONAL_FLOOR * 0.8:  # 80% headroom for fees
            self.logger.warning(
                f"[{self.name}] Signal quote {signal['quote']:.2f} < min_notional {MIN_NOTIONAL_FLOOR:.2f}; filtering out"
            )
            return  # Don't emit sub-minimum signals

        # Add to collection buffer (AgentManager will forward to Meta)
        self._collected_signals.append(signal)
        self.logger.info(
            "[%s] SIGNAL: %s %s conf=%.2f req=%.2f break_even=%.2f regime=%s hint=%s",
            self.name,
            symbol,
            signal["action"],
            float(confidence),
            float(required_conf),
            float(break_even_prob),
            str(regime),
            str(signal.get("_tradeability_hint", "")),
        )

        # Record confidence sample for horizon-based EV calibration.
        if self._conf_eval_enabled:
            try:
                entry_price = await self._safe_current_price(symbol)
                if entry_price is not None and entry_price > 0:
                    self._pending_conf_samples.append(
                        {
                            "symbol": str(symbol).upper(),
                            "action": str(action).upper(),
                            "confidence": float(confidence),
                            "entry_price": float(entry_price),
                            "expected_move_pct": float(expected_move_pct),
                            "regime": str(regime),
                            "ts": time.time(),
                        }
                    )
            except Exception:
                self.logger.debug("[%s] failed to queue confidence sample for %s", self.name, symbol, exc_info=True)

    # ---------------- Run per symbol ----------------

    async def run(self, symbol: str):
        cur_sym = symbol.upper()
        cur_tf = self.timeframe
        self.logger.debug(f"🚀 {self.name} run for {cur_sym} @ {cur_tf}")

        if not self.model_manager:
            return {"action": "hold", "confidence": 0.0, "reason": "missing_model_manager"}

        # Per-symbol tuned params
        tuned = {}
        try:
            tuned = load_tuned_params(f"{self.name}_{cur_sym}_{cur_tf}") or {}
        except Exception:
            tuned = {}
        lookback = self._resolve_lookback(tuned)
        confidence_threshold = float(tuned.get("confidence_threshold", getattr(self.config, "CONFIDENCE_THRESHOLD", 0.7)))
        is_active = bool(tuned.get("active", True))
        self.logger.debug(f"[{self.name}] Tuned for {cur_sym}: lookback={lookback}, active={is_active}")

        if not is_active:
            self.logger.info(f"⚠️ {self.name} for {cur_sym}@{cur_tf} is deactivated.")
            return {"action": "hold", "confidence": 0.0, "reason": "Deactivated"}

        # Ensure model_manager is available
        if not self.model_manager:
            return {"action": "hold", "confidence": 0.0, "reason": "ModelManager not available"}

        # Ensure model path
        model_path = self.model_manager.build_model_path(agent_name=self.name, symbol=cur_sym, version=cur_tf)

        # Data fetch + feature build
        try:
            ohlcv = await self._get_market_data_safe(cur_sym, cur_tf)
            if ohlcv is None:
                return {"action": "hold", "confidence": 0.0, "reason": "OHLCV None"}
            if not isinstance(ohlcv, list) or len(ohlcv) < lookback:
                return {"action": "hold", "confidence": 0.0, "reason": "Insufficient OHLCV"}
            feature_df = self._build_edge_feature_frame(ohlcv)
            if feature_df is None or feature_df.empty or len(feature_df) < lookback:
                return {"action": "hold", "confidence": 0.0, "reason": "Insufficient engineered features"}
        except Exception as e:
            self.logger.error(f"❌ OHLCV fetch failed for {cur_sym}@{cur_tf}: {e}", exc_info=True)
            return {"action": "hold", "confidence": 0.0, "reason": "Data fetch error"}

        train_cols = self._training_feature_columns()
        train_df = feature_df[train_cols].copy()

        # Create model if absent.
        if not self.model_manager.model_exists(model_path):
            scheduled, train_state = self._schedule_startup_full_training(
                symbol=cur_sym,
                timeframe=cur_tf,
                lookback=lookback,
                model_path=model_path,
                reason="startup_missing_model",
            )
            if not scheduled:
                # Fallback to adaptive background training when full-tier bootstrap is disabled/unavailable.
                scheduled, train_state = self._schedule_background_training(
                    symbol=cur_sym,
                    timeframe=cur_tf,
                    lookback=lookback,
                    train_df=train_df,
                    model_path=model_path,
                    reason="missing_model_fallback",
                )
            if not scheduled:
                self.logger.debug(
                    "[%s] skipped training schedule for %s: %s",
                    self.name,
                    cur_sym,
                    train_state,
                )
            return {
                "action": "hold",
                "confidence": 0.0,
                "reason": "Bootstrap training scheduled" if scheduled else train_state,
            }

        # Cached load with mtime guard
        model = None
        try:
            mtime = os.path.getmtime(model_path)
        except Exception:
            mtime = 0.0
        cached = self.model_cache.get(model_path)
        if cached and self._model_mtime.get(model_path) == mtime:
            model = cached
        else:
            model = self.model_manager.safe_load_model(model_path)
            if model is not None:
                self.model_cache[model_path] = model
                self._model_mtime[model_path] = mtime
        if model is None:
            return {"action": "hold", "confidence": 0.0, "reason": "Model not available"}

        # Optional startup refresh: run one full-tier retrain on first boot (or if full-train interval elapsed).
        if self._full_train_on_startup:
            full_due_startup = self._is_full_train_due(
                cur_sym,
                model_path,
                time.time(),
                force_startup=True,
            )
            if full_due_startup:
                scheduled_full, full_state = self._schedule_startup_full_training(
                    symbol=cur_sym,
                    timeframe=cur_tf,
                    lookback=lookback,
                    model_path=model_path,
                    reason="startup_refresh_full_train",
                )
                if scheduled_full:
                    self.logger.info(
                        "[%s] queued startup full-tier refresh for %s",
                        self.name,
                        cur_sym,
                    )
                elif full_state not in {
                    "full_training_pending",
                    "training_in_progress",
                    "full_training_already_done",
                }:
                    self.logger.debug(
                        "[%s] startup full-tier refresh for %s not queued: %s",
                        self.name,
                        cur_sym,
                        full_state,
                    )

        desired_dim = len(self._training_feature_columns())
        expected_lookback = self._model_input_lookback(model)
        expected_dim = self._model_input_feature_count(model)
        effective_lookback = (
            int(expected_lookback)
            if expected_lookback is not None and int(expected_lookback) > 1
            else int(lookback)
        )
        if len(feature_df) < effective_lookback:
            self.logger.warning(
                "[%s] Skip %s: model lookback=%d exceeds feature rows=%d",
                self.name,
                cur_sym,
                int(effective_lookback),
                int(len(feature_df)),
            )
            return {"action": "hold", "confidence": 0.0, "reason": "model_lookback_mismatch"}
        needs_upgrade = bool(
            self._feature_force_upgrade
            and expected_dim is not None
            and expected_dim != desired_dim
        )
        
        # Check for architecture mismatch (old LSTM vs new GRU)
        has_expected_arch = self._model_has_expected_architecture(model)
        needs_retrain_arch = not has_expected_arch
        
        retrain_reason = (
            f"feature_dim_upgrade_{expected_dim}_to_{desired_dim}" if needs_upgrade else "architecture_upgrade_to_gru"
        )
        if (
            (needs_upgrade or needs_retrain_arch)
            and self._auto_retrain_feature_mismatch
            and model_path not in self._retrained_feature_models
            and model_path not in self._feature_upgrade_pending
        ):
            _scheduled, _state = self._schedule_background_training(
                symbol=cur_sym,
                timeframe=cur_tf,
                lookback=lookback,
                train_df=train_df,
                model_path=model_path,
                reason=retrain_reason,
            )
            if not _scheduled:
                self.logger.debug(
                    "[%s] feature-upgrade retrain not queued for %s: %s",
                    self.name,
                    cur_sym,
                    _state,
                )
            else:
                self.logger.warning(
                    "[%s] %s uses legacy feature dim=%s; queued upgrade to dim=%s.",
                    self.name,
                    cur_sym,
                    str(expected_dim),
                    str(desired_dim),
                )

        # Hard safety gate: do not run inference on unsupported/legacy architecture while upgrade is enforced.
        if needs_retrain_arch and self._feature_force_upgrade:
            self.logger.warning(
                "[%s] Skip %s inference: legacy architecture detected (expected GRU).",
                self.name,
                cur_sym,
            )
            return {"action": "hold", "confidence": 0.0, "reason": "model_arch_mismatch"}

        input_cols, feature_mode = self._resolve_input_columns_for_model(expected_dim)
        if not input_cols:
            self.logger.warning(
                "[%s] Skip %s inference: unsupported model feature dim=%s model_path=%s",
                self.name,
                cur_sym,
                str(expected_dim),
                str(model_path),
            )
            return {"action": "hold", "confidence": 0.0, "reason": "model_shape_unsupported"}
        if expected_dim is not None and len(input_cols) != int(expected_dim):
            self.logger.warning(
                "[%s] Skip %s inference: mapped feature dim mismatch expected=%s mapped=%d mode=%s",
                self.name,
                cur_sym,
                str(expected_dim),
                len(input_cols),
                feature_mode,
            )
            return {"action": "hold", "confidence": 0.0, "reason": "model_feature_mismatch"}

        X = self._build_model_input_tensor(feature_df, effective_lookback, input_cols)
        if X is None:
            self.logger.error(
                "[%s] Failed input tensor build for %s (lookback=%d model_lookback=%s mode=%s expected_dim=%s mapped_dim=%d).",
                self.name,
                cur_sym,
                effective_lookback,
                str(expected_lookback),
                feature_mode,
                str(expected_dim),
                len(input_cols),
            )
            return {"action": "hold", "confidence": 0.0, "reason": "Input formatting error"}
        if expected_dim is not None and int(X.shape[-1]) != int(expected_dim):
            self.logger.warning(
                "[%s] Skip %s inference: tensor feature dim mismatch expected=%s got=%d",
                self.name,
                cur_sym,
                str(expected_dim),
                int(X.shape[-1]),
            )
            return {"action": "hold", "confidence": 0.0, "reason": "model_feature_mismatch"}
        if expected_lookback is not None and int(X.shape[1]) != int(expected_lookback):
            self.logger.warning(
                "[%s] Skip %s inference: tensor lookback mismatch expected=%s got=%d",
                self.name,
                cur_sym,
                str(expected_lookback),
                int(X.shape[1]),
            )
            return {"action": "hold", "confidence": 0.0, "reason": "model_lookback_mismatch"}
        feature_dim = int(X.shape[-1])

        # Predict
        try:
            import tensorflow as tf  # local import to avoid import-time cost when unused
            key = (model_path, effective_lookback, feature_dim)
            predict_fn = self._predict_fns.get(key)
            spec = tf.TensorSpec(shape=[None, effective_lookback, feature_dim], dtype=tf.float32)
            if predict_fn is None:
                @tf.function(input_signature=[spec], reduce_retracing=True)
                def _predict_fn(x):
                    return model(x, training=False)
                self._predict_fns[key] = _predict_fn
                predict_fn = _predict_fn

            # Safety: ensure cached function was built with same lookback
            try:
                _ = predict_fn.get_concrete_function(
                    tf.TensorSpec(shape=[None, effective_lookback, feature_dim], dtype=tf.float32)
                )
            except Exception:
                # Rebuild if signature mismatched for any reason (including architecture changes)
                self.logger.info(f"Rebuilding prediction function for {cur_sym} due to signature mismatch")
                @tf.function(input_signature=[spec], reduce_retracing=True)
                def _predict_fn(x):
                    return model(x, training=False)
                self._predict_fns[key] = _predict_fn
                predict_fn = _predict_fn

            loop = asyncio.get_running_loop()

            def _infer():
                x_tf = tf.convert_to_tensor(X, dtype=tf.float32)
                x_tf = tf.ensure_shape(x_tf, [None, effective_lookback, feature_dim])
                return predict_fn(x_tf).numpy()[0]

            y_raw = await loop.run_in_executor(None, _infer)
            
            action, confidence, probs, schema = self._decode_model_output(y_raw)
            
            if np.max(y_raw) > 1.0 or np.min(y_raw) < 0.0:
                self.logger.info(
                    f"[{self.name}] {cur_sym}: logits={y_raw} schema={schema} feature_mode={feature_mode} "
                    f"lb={effective_lookback} fdim={feature_dim} probs={probs} "
                    f"-> action={action}, conf={confidence:.2f}"
                )
            else:
                self.logger.info(
                    f"[{self.name}] {cur_sym}: schema={schema} feature_mode={feature_mode} "
                    f"lb={effective_lookback} fdim={feature_dim} probs={probs} action={action}, conf={confidence:.2f}"
                )
        except Exception as e:
            self.logger.error(f"❌ Prediction failed for {cur_sym}: {e}", exc_info=True)
            return {"action": "hold", "confidence": 0.0, "reason": "Prediction failure"}

        # Filters (sentiment / regime / CoT echo if present)
        try:
            sentiment_val = await self.shared_state.get_sentiment(cur_sym)
        except Exception:
            sentiment_val = None
        sentiment = float(sentiment_val) if sentiment_val is not None else 0.0

        reginfo = None
        try:
            reginfo = await self.shared_state.get_volatility_regime(cur_sym, timeframe=self.timeframe)
        except Exception:
            reginfo = None
        regime = self._normalize_regime((reginfo or {}).get("regime", "normal"))

        cot_txt = ""
        try:
            get_cot = getattr(self.shared_state, "get_cot_explanation", None)
            if callable(get_cot):
                cot_txt = (await get_cot(cur_sym, self.name)) or ""
        except Exception:
            cot_txt = ""

        cot_num = 1 if "yes" in cot_txt.lower() else (-1 if "no" in cot_txt.lower() else 0)
        self.logger.info(
            f"[{self.name}] Filters: Sentiment={sentiment:.2f}, VolatilityRegime={regime}, CoT='{cot_txt}', CoT_Numeric={cot_num}"
        )

        if (sentiment < -0.5) or (regime in {"high_vol", "bear"}) or (cot_num < 0):
            action = "hold"

        self.logger.info(f"[{self.name}] Final decision for {cur_sym} => Action: {action.upper()}, Confidence: {confidence:.2f}")

        # Store a brief CoT/debug explanation for UI/debuggers
        try:
            await self.shared_state.set_cot_explanation(
                cur_sym,
                text=f"Pred={action} conf={confidence:.2f} on features shape={X.shape}",
                source=self.name,
            )
        except Exception:
            pass

        # ═══════════════════════════════════════════════════════════════════════
        # ML POSITION SCALING: Calculate position scale based on buy probability
        # ═══════════════════════════════════════════════════════════════════════
        position_scale = 1.0  # Default: no scaling
        
        if action.upper() == "BUY":
            # Extract buy probability from the model output
            # The confidence here represents our prediction strength
            prob = float(confidence)
            
            # Tiered position scaling based on confidence bands
            if prob >= 0.75:
                position_scale = 1.5  # 50% larger position
            elif prob >= 0.65:
                position_scale = 1.2  # 20% larger position
            elif prob >= 0.55:
                position_scale = 1.0  # Standard position size
            elif prob >= 0.45:
                position_scale = 0.8  # 20% smaller position
            else:
                position_scale = 0.6  # 40% smaller position
            
            # Store ML position scale in SharedState for downstream use by MetaController
            try:
                if hasattr(self.shared_state, "set_ml_position_scale"):
                    await self.shared_state.set_ml_position_scale(cur_sym, position_scale)
                    self.logger.info(
                        f"[{self.name}] ML position scale stored for {cur_sym}: {position_scale:.2f}x "
                        f"(confidence={prob:.2f})"
                    )
            except Exception as e:
                self.logger.warning(f"[{self.name}] Failed to store ML position scale: {e}")

        # --------- Emission (signal-only) ---------
        await self._collect_signal(
            symbol=cur_sym,
            action=action,
            confidence=confidence,
            reason="ML model prediction",
            extras={
                "_feature_mode": str(feature_mode),
                "_feature_dim": int(feature_dim),
                "_model_schema": str(schema),
            },
        )

        # 🔄 LIGHTWEIGHT SIGNAL OUTCOME TRACKING
        if action == "buy":
            try:
                now = time.time()
                # Phase 4: Use memory-first pattern for price lookup
                price_now = None
                if hasattr(self.shared_state, "prices"):
                    price_now = self.shared_state.prices.get(cur_sym.upper())
                if not price_now and hasattr(self.shared_state, "get_price"):
                    # Fallback to method if available
                    try:
                        price_now = self.shared_state.get_price(cur_sym)
                    except Exception:
                        price_now = None
                
                if hasattr(self.shared_state, "register_signal_outcome"):
                    self.shared_state.register_signal_outcome({
                        "symbol": cur_sym,
                        "timestamp": now,
                        "price_at_signal": price_now,
                        "confidence": confidence,
                        "agent": "MLForecaster"
                    })
            except Exception:
                pass

        return {"action": action, "confidence": confidence, "reason": "Prediction processed"}
