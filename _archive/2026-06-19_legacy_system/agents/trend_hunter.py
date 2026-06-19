# =============================
# Imports
# =============================
import asyncio
import inspect
import logging
import os
import time
from functools import partial
from math import inf
from typing import Any, Optional

try:
    import tensorflow as tf
except ImportError:
    tf = None


# =============================
# Utilities
# =============================
async def _await_maybe(x):
    """Await if coroutine-like, otherwise return as-is."""
    return await x if inspect.isawaitable(x) else x


# =============================
# TA-Lib Detection
# =============================
_HAS_TALIB = True
try:
    pass
except Exception:
    _HAS_TALIB = False


# =============================
# Local Imports
# =============================
from src.l0_core.component_status_logger import log_component_status
from src.l5_strategy.model_manager import build_model_path, safe_load_model

try:
    from utils.ta_indicators import calculate_volume_surge as _calc_volume_surge

    _HAS_TA_INDICATORS = True
except Exception:
    _HAS_TA_INDICATORS = False
    _calc_volume_surge = None
try:
    from utils.tuned_params import get_tuned_params as _get_tuned_params

    _HAS_TUNED_PARAMS = True
except Exception:
    _HAS_TUNED_PARAMS = False
    _get_tuned_params = None


# =============================
# Constants
# =============================
AGENT_NAME = "TrendHunter"


# =============================
# Logging (idempotent setup)
# =============================
logger = logging.getLogger(AGENT_NAME)
logger.setLevel(logging.DEBUG)

_log_path = f"logs/agents/{AGENT_NAME.lower()}.log"
os.makedirs(os.path.dirname(_log_path), exist_ok=True)

if not any(
    isinstance(h, logging.FileHandler) and getattr(h, "_trendhunter", False)
    for h in logger.handlers
):
    fh = logging.FileHandler(_log_path)
    fh._trendhunter = True  # mark to avoid duplicates
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] [%(name)s] %(message)s"))
    logger.addHandler(fh)


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


# =============================
# Agent
# =============================
class TrendHunter:
    # Accept short or long OHLCV keys and normalize to [o,h,l,c,v]
    def _std_row(self, r):
        try:
            if isinstance(r, dict):
                d = r
                o = d.get("o", d.get("open"))
                h = d.get("h", d.get("high"))
                l = d.get("l", d.get("low"))
                c = d.get("c", d.get("close", d.get("last", d.get("price"))))
                v = d.get("v", d.get("volume"))
                if None in (o, h, l, c, v):
                    return None
                return [float(o), float(h), float(l), float(c), float(v)]
            seq = list(r)
            if len(seq) >= 6:
                seq = seq[-5:]
            if len(seq) == 5:
                return [float(x) for x in seq]
        except Exception:
            return None
        return None

    async def _get_market_data_safe(self, symbol: str, timeframe: str):
        fn = getattr(self.shared_state, "get_market_data", None)
        if not callable(fn):
            return None
        res = fn(symbol, timeframe)
        return (await res) if asyncio.iscoroutine(res) else res

    async def _prefilter_symbol(self, symbol: str) -> bool:
        """Best-effort: True if symbol tradable and affordable given minNotional cap."""
        ec = getattr(self, "exchange_client", None)
        if not ec:
            return True
        try:
            if getattr(self, "require_trading_status", True) and hasattr(ec, "symbol_info"):
                info = ec.symbol_info(symbol)
                info = await info if inspect.isawaitable(info) else info
                if not info:
                    logger.warning("[%s] No symbol_info for %s; skipping.", self.name, symbol)
                    return False
                status = str(info.get("status", "TRADING")).upper()
                if status != "TRADING":
                    logger.warning("[%s] %s status is %s; skipping.", self.name, symbol, status)
                    return False
                # Parse MIN_NOTIONAL from dict or list formats
                min_notional = None
                filters = info.get("filters") or {}
                if isinstance(filters, dict) and "MIN_NOTIONAL" in filters:
                    try:
                        min_notional = float(filters["MIN_NOTIONAL"])
                    except Exception:
                        min_notional = None
                elif isinstance(filters, list):
                    for f in filters:
                        if isinstance(f, dict) and f.get("filterType") == "MIN_NOTIONAL":
                            try:
                                min_notional = float(f.get("minNotional", inf))
                            except Exception:
                                min_notional = None
                            break
                cap = float(getattr(self, "max_per_trade_usdt", 100.0))
                if min_notional is not None and min_notional > cap:
                    logger.warning(
                        "[%s] %s MIN_NOTIONAL %.4f exceeds cap %.2f; deferring to execution layer.",
                        self.name,
                        symbol,
                        min_notional,
                        cap,
                    )
            return True
        except Exception:
            logger.debug("[%s] prefilter failed for %s", self.name, symbol, exc_info=True)
            return False

    async def generate_signals(self) -> list[dict[str, Any]]:
        """
        Main entry point for signal generation.
        TrendHunter is not yet fully implemented, returning empty signals.
        """
        return []

    agent_type = "strategy"  # required by AgentManager

    def __init__(
        self,
        shared_state,
        market_data_feed,
        execution_manager,
        config,
        tp_sl_engine,
        model_manager,
        timeframe: str = "5m",
        symbols: Optional[list[str]] = None,
        name: str = AGENT_NAME,
        # optional/wired deps
        symbol: Optional[str] = None,
        market_data: Any = None,
        meta_controller: Any = None,
        symbol_manager: Any = None,
        exchange_client: Any = None,
        database_manager: Any = None,
        agent_schedule: Any = None,
        **kwargs,
    ):
        self.shared_state = shared_state
        self.market_data_feed = market_data_feed
        self.execution_manager = execution_manager
        self.config = config
        self.tp_sl_engine = tp_sl_engine
        self.model_manager = model_manager
        self.name = name
        self.timeframe = timeframe
        self.base_ccy = getattr(self.config, "BASE_CURRENCY", "USDT")

        # optional injections
        self.symbol = symbol
        self.market_data = market_data
        self.meta_controller = meta_controller
        self.symbol_manager = symbol_manager
        self.exchange_client = exchange_client
        self.database_manager = database_manager
        self.agent_schedule = agent_schedule

        # tuned params + safe config access
        self._tuned_cache = self._load_tuned()
        self.ema_fast = int(self._cfg("EMA_FAST", self._tuned_cache.get("ema_fast", 12)))
        self.ema_slow = int(self._cfg("EMA_SLOW", self._tuned_cache.get("ema_slow", 26)))
        # Risk-aware prefilter knobs for HYG guards (Dynamic)
        self.require_trading_status = bool(self._cfg("REQUIRE_TRADING_STATUS", True))

        # Per-agent knobs

        # Default now includes "high" since VolatilityRegimeDetector only emits
        # "low" | "normal" | "high" — excluding "high" caused TrendHunter to
        # silently skip ALL symbols in volatile crypto markets.
        _allowed_regimes = self._cfg("TRENDHUNTER_ALLOWED_REGIMES", ["", "low", "moderate", "high"])
        if isinstance(_allowed_regimes, str):
            _allowed_regimes = [r.strip().lower() for r in _allowed_regimes.split(",") if r.strip()]
        self.allowed_regimes = set(
            [str(r).lower() for r in (_allowed_regimes or ["", "low", "moderate", "high"])]
        )
        # Back-compat: map legacy "moderate" to the current "normal" regime label.
        if "moderate" in self.allowed_regimes and "normal" not in self.allowed_regimes:
            self.allowed_regimes.add("normal")

        # symbol and model caches
        self.symbols = symbols  # can be None → lazy load
        self._accepted_snapshot: Optional[list[str]] = None
        self._accepted_snapshot_ts: float = 0.0
        self._snapshot_ttl_sec = 15.0  # throttle SharedState hits

        self.model_cache: dict[str, Optional[Any]] = {}

        # if symbols provided, pre-warm model cache
        if self.symbols:
            for s in self.symbols:
                self._ensure_model_cache_key(s)

        self.trades_count = 0
        self.win_count = 0
        self.loss_count = 0
        self._collected_signals: list[dict[str, Any]] = []
        self._collecting_for_agent_manager = False
        self._training_in_progress: set[str] = set()
        self._retrain_last_attempt_ts: dict[str, float] = {}
        self._retrain_last_failure_ts: dict[str, float] = {}

        log_component_status(self.name, "Initialized")
        logger.info(
            "🚀 %s initialized (timeframe=%s, symbols=%d)",
            self.name,
            self.timeframe,
            len(self.symbols or []),
        )

    # ------------- helpers -------------
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

    @property
    def min_conf(self) -> float:
        """Dynamic access to minimum signal confidence (PHASE A)."""
        return float(
            self._cfg(
                "TREND_MIN_CONF",
                self._cfg("TRENDHUNTER_MIN_SIGNAL_CONF", self._cfg("MIN_SIGNAL_CONF", 0.35)),
            )
        )

    @property
    def max_per_trade_usdt(self) -> float:
        """Dynamic access to maximum trade size."""
        return float(self._cfg("MAX_PER_TRADE_USDT", 100.0))

    def _load_tuned(self) -> dict[str, Any]:
        try:
            from src.l5_strategy.agent_optimizer import load_tuned_params

            result = load_tuned_params(self.name) or {}
            if result:
                return result
        except Exception:
            pass
        # Fallback: utils.tuned_params (symbol-agnostic agent-level params)
        if _HAS_TUNED_PARAMS and _get_tuned_params is not None:
            try:
                return _get_tuned_params(self.name) or {}
            except Exception:
                pass
        return {}

    def _ensure_model_cache_key(self, sym: str) -> None:
        if sym in self.model_cache:
            return
        try:
            path = build_model_path(self.name, sym, version=self.timeframe)
            model = safe_load_model(path)
            if model is None:
                logger.info(
                    "[%s] No pre-trained model for %s (will train on first run).", self.name, sym
                )
            self.model_cache[sym] = model
        except Exception as e:
            logger.debug("[%s] safe_load_model failed for %s: %s", self.name, sym, e)
            self.model_cache[sym] = None

    def _is_auto_train_enabled(self) -> bool:
        return _as_bool(self._cfg("TREND_AUTO_TRAIN", self._cfg("AUTO_TRAIN", False)))

    def _normalize_training_rows(self, data: Any) -> list[dict[str, float]]:
        """
        Normalize OHLCV payload into canonical rows expected by ModelTrainer:
        {'timestamp','open','high','low','close','volume'}.
        """
        rows: list[dict[str, float]] = []
        if not isinstance(data, list):
            return rows

        for idx, item in enumerate(data):
            try:
                ts = float(idx)
                if isinstance(item, dict):
                    o = item.get("o", item.get("open"))
                    h = item.get("h", item.get("high"))
                    l = item.get("l", item.get("low"))
                    c = item.get("c", item.get("close", item.get("price")))
                    v = item.get("v", item.get("volume"))
                    ts = float(item.get("ts", item.get("timestamp", item.get("t", idx))) or idx)
                else:
                    seq = list(item)
                    if len(seq) >= 6:
                        ts = float(seq[0])
                        o, h, l, c, v = seq[1], seq[2], seq[3], seq[4], seq[5]
                    elif len(seq) == 5:
                        o, h, l, c, v = seq
                    else:
                        continue

                row = {
                    "timestamp": float(ts),
                    "open": float(o),
                    "high": float(h),
                    "low": float(l),
                    "close": float(c),
                    "volume": float(v),
                }
                rows.append(row)
            except Exception:
                continue

        # Keep last value for duplicate timestamps and preserve chronological order.
        if not rows:
            return rows
        dedup: dict[float, dict[str, float]] = {}
        for r in rows:
            dedup[float(r["timestamp"])] = r
        return [dedup[k] for k in sorted(dedup.keys())]

    async def _retrain_if_needed(self, symbol: str) -> bool:
        """
        If model is missing, trigger background training if AUTO_TRAIN is enabled.
        Returns True if model exists (ready to predict), False if training/missing.
        """
        if self.model_cache.get(symbol) is not None:
            return True

        # Check if already training
        if symbol in self._training_in_progress:
            return False

        if tf is None:
            logger.debug(
                "[%s] Model missing for %s; TensorFlow unavailable, staying in indicator-only mode.",
                self.name,
                symbol,
            )
            return True

        # Check config to authorize CPU-heavy training
        if not self._is_auto_train_enabled():
            logger.debug(
                "[%s] Model missing for %s; TREND_AUTO_TRAIN/AUTO_TRAIN disabled.",
                self.name,
                symbol,
            )
            return True  # Fallback to heuristic logic if training disabled

        now_ts = time.time()
        cooldown_s = max(0.0, float(self._cfg("TREND_RETRAIN_COOLDOWN_S", 0.0) or 0.0))
        last_attempt = float(self._retrain_last_attempt_ts.get(symbol, 0.0) or 0.0)
        if cooldown_s > 0 and (now_ts - last_attempt) < cooldown_s:
            remain = max(0.0, cooldown_s - (now_ts - last_attempt))
            logger.debug(
                "[%s] Retrain cooldown active for %s (remaining=%.1fs).",
                self.name,
                symbol,
                remain,
            )
            return False

        fail_backoff_s = max(0.0, float(self._cfg("TREND_RETRAIN_FAIL_BACKOFF_S", 1800.0) or 0.0))
        last_fail = float(self._retrain_last_failure_ts.get(symbol, 0.0) or 0.0)
        if fail_backoff_s > 0 and last_fail > 0 and (now_ts - last_fail) < fail_backoff_s:
            remain = max(0.0, fail_backoff_s - (now_ts - last_fail))
            logger.debug(
                "[%s] Retrain failure backoff active for %s (remaining=%.1fs).",
                self.name,
                symbol,
                remain,
            )
            return False

        # Trigger Background Training
        self._retrain_last_attempt_ts[symbol] = now_ts
        self._training_in_progress.add(symbol)
        logger.info(f"[{self.name}] 🧠 Triggering background training for {symbol}...")

        asyncio.create_task(self._run_background_training(symbol))
        return False  # Skip this tick while training

    async def _run_background_training(self, symbol: str):
        try:
            lookback = int(self._cfg("TRENDHUNTER_RETRAIN_LOOKBACK", 100) or 100)
            min_rows = max(lookback + 50, int(self._cfg("TREND_RETRAIN_MIN_BARS", 220) or 220))
            fetch_limit = max(
                min_rows + 50, int(self._cfg("TREND_RETRAIN_FETCH_LIMIT", 750) or 750)
            )
            max_rows = max(min_rows, int(self._cfg("TREND_RETRAIN_MAX_ROWS", 1200) or 1200))

            # 1) Start from cached shared-state OHLCV.
            cached = await self._get_market_data_safe(symbol, self.timeframe)
            rows = self._normalize_training_rows(cached)

            # 2) If cache is shallow, request a deeper pull directly from exchange.
            exchange_client = self.exchange_client or getattr(
                self.execution_manager, "exchange_client", None
            )
            if len(rows) < min_rows and exchange_client and hasattr(exchange_client, "get_klines"):
                try:
                    raw = await exchange_client.get_klines(
                        symbol, self.timeframe, limit=int(fetch_limit)
                    )
                    exchange_rows = self._normalize_training_rows(raw)
                    if len(exchange_rows) > len(rows):
                        rows = exchange_rows
                except Exception as e:
                    logger.debug(
                        "[%s] Exchange backfill failed for %s: %s",
                        self.name,
                        symbol,
                        e,
                        exc_info=True,
                    )

            if len(rows) < min_rows:
                self._retrain_last_failure_ts[symbol] = time.time()
                logger.warning(
                    "[%s] Not enough data to train %s (rows=%d need>=%d).",
                    self.name,
                    symbol,
                    len(rows),
                    min_rows,
                )
                return

            train_rows = rows[-int(max_rows) :]

            # 3) Run Trainer in ThreadPool to avoid blocking main loop.
            from src.l5_strategy.model_trainer import ModelTrainer

            trainer = ModelTrainer(
                symbol,
                timeframe=self.timeframe,
                input_lookback=lookback,
                agent_name=self.name,
                model_manager=self.model_manager,
            )

            loop = asyncio.get_running_loop()
            train_call = partial(
                trainer.train_model,
                train_rows,
                max_rows=int(max_rows),
                return_metrics=True,
            )

            result = await loop.run_in_executor(None, train_call)
            ok = bool(result.get("ok")) if isinstance(result, dict) else False
            if ok:
                logger.info(f"[{self.name}] Training completed successfully for {symbol}.")
                # Optionally reload model into cache here
                self._ensure_model_cache_key(symbol)
            else:
                logger.warning(f"[{self.name}] Training failed or incomplete for {symbol}.")

            self._training_in_progress.discard(symbol)
        except Exception as e:
            logger.error(
                f"[{self.name}] Exception in _run_background_training for {symbol}: {e}",
                exc_info=True,
            )
            self._training_in_progress.discard(symbol)


def _trendhunter_eof_marker():
    pass
