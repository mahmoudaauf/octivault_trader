
# =============================
# Imports
# =============================
import asyncio
import logging
import os
import time
import inspect
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Set
from math import inf

import numpy as np
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
    import talib
except Exception:
    _HAS_TALIB = False


# =============================
# Local Imports
# =============================
from core.stubs import TradeIntent, ExecOrder
from core.model_manager import load_model as _load_model, build_model_path, safe_load_model
from core.component_status_logger import log_component_status
from utils.indicators import compute_ema, compute_macd
from utils.volatility_adjusted_confidence import (
    compute_heuristic_confidence,
    categorize_signal,
    get_signal_quality_metrics,
)

from agents.edge_calculator import compute_agent_edge, merge_signal_with_edge  # ALPHA AMPLIFIER
import time




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

if not any(isinstance(h, logging.FileHandler) and getattr(h, "_trendhunter", False) for h in logger.handlers):
    fh = logging.FileHandler(_log_path)
    fh._trendhunter = True  # mark to avoid duplicates
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] [%(name)s] %(message)s"))
    logger.addHandler(fh)

    
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
            if getattr(self, 'require_trading_status', True) and hasattr(ec, "symbol_info"):
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
                cap = float(getattr(self, 'max_per_trade_usdt', 100.0))
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
        symbols: Optional[List[str]] = None,
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
        self.allowed_regimes = set([str(r).lower() for r in (_allowed_regimes or ["", "low", "moderate", "high"])])
        # Back-compat: map legacy "moderate" to the current "normal" regime label.
        if "moderate" in self.allowed_regimes and "normal" not in self.allowed_regimes:
            self.allowed_regimes.add("normal")

        # symbol and model caches
        self.symbols = symbols  # can be None → lazy load
        self._accepted_snapshot: Optional[List[str]] = None
        self._accepted_snapshot_ts: float = 0.0
        self._snapshot_ttl_sec = 15.0  # throttle SharedState hits

        self.model_cache: Dict[str, Optional[Any]] = {}

        # if symbols provided, pre-warm model cache
        if self.symbols:
            for s in self.symbols:
                self._ensure_model_cache_key(s)

        self.trades_count = 0
        self.win_count = 0
        self.loss_count = 0
        self._collected_signals: List[Dict[str, Any]] = []
        self._collecting_for_agent_manager = False

        log_component_status(self.name, "Initialized")
        logger.info("🚀 %s initialized (timeframe=%s, symbols=%d)", self.name, self.timeframe, len(self.symbols or []))

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
                self._cfg("TRENDHUNTER_MIN_SIGNAL_CONF",
                    self._cfg("MIN_SIGNAL_CONF", 0.55))
            )
        )

    @property
    def max_per_trade_usdt(self) -> float:
        """Dynamic access to maximum trade size."""
        return float(self._cfg("MAX_PER_TRADE_USDT", 100.0))

    def _load_tuned(self) -> Dict[str, Any]:
        try:
            from core.agent_optimizer import load_tuned_params
            return load_tuned_params(self.name) or {}
        except Exception:
            return {}

    def _ensure_model_cache_key(self, sym: str) -> None:
        if sym in self.model_cache:
            return
        try:
            path = build_model_path(self.name, sym, version=self.timeframe)
            model = safe_load_model(path)
            if model is None:
                logger.info("[%s] No pre-trained model for %s (will train on first run).", self.name, sym)
            self.model_cache[sym] = model
        except Exception as e:
            logger.debug("[%s] safe_load_model failed for %s: %s", self.name, sym, e)
            self.model_cache[sym] = None

        self._training_in_progress: Set[str] = set()

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

        # Check config to authorize CPU-heavy training
        if not self._cfg("AUTO_TRAIN", False):
            return True # Fallback to heuristic logic if training disabled

        # Trigger Background Training
        self._training_in_progress.add(symbol)
        logger.info(f"[{self.name}] 🧠 Triggering background training for {symbol}...")
        
        asyncio.create_task(self._run_background_training(symbol))
        return False # Skip this tick while training

    async def _run_background_training(self, symbol: str):
        try:
            # 1. Fetch Data
            df = await self.shared_state.get_market_data(symbol, self.timeframe, limit=500)
            if df is None or len(df) < 200:
                logger.warning(f"[{self.name}] Not enough data to train {symbol}")
                return

            # 2. Run Trainer in ThreadPool to avoid blocking Main Loop
            from core.model_trainer import ModelTrainer
            trainer = ModelTrainer(symbol, timeframe=self.timeframe)
            
            loop = asyncio.get_running_loop()
            success = await loop.run_in_executor(None, trainer.train_model, df)
            
            if success:
                logger.info(f"[{self.name}] Training success for {symbol}. Reloading model.")
                self._ensure_model_cache_key(symbol)
            else:
                logger.warning(f"[{self.name}] Training failed for {symbol}.")
                
        except Exception as e:
            logger.error(f"[{self.name}] Background training crashed for {symbol}: {e}")
        finally:
             if symbol in self._training_in_progress:
                 self._training_in_progress.remove(symbol)

    # ------------- lifecycle -------------
    async def generate_signals(self) -> List[Any]:
        self._collected_signals = []
        self._collecting_for_agent_manager = True
        try:
            await self.load_symbols()
            await self.run_once()
            return self._collected_signals
        finally:
            self._collecting_for_agent_manager = False

    async def _publish_trade_intent(self, payload: Dict[str, Any]) -> bool:
        """Publish canonical TradeIntent for standalone agent loops."""
        event_bus = getattr(self.shared_state, "event_bus", None)
        publish = getattr(event_bus, "publish", None)
        if callable(publish):
            try:
                await publish("events.trade.intent", TradeIntent(**payload))
                logger.info(
                    "[%s] Published TradeIntent: %s %s",
                    self.name,
                    payload.get("symbol"),
                    payload.get("side"),
                )
                return True
            except Exception:
                logger.warning(
                    "[%s] Failed to publish TradeIntent for %s",
                    self.name,
                    payload.get("symbol"),
                    exc_info=True,
                )
        emit_event = getattr(self.shared_state, "emit_event", None)
        if callable(emit_event):
            try:
                await _await_maybe(emit_event("TradeIntent", dict(payload)))
                return True
            except Exception:
                logger.warning(
                    "[%s] Fallback TradeIntent emit failed for %s",
                    self.name,
                    payload.get("symbol"),
                    exc_info=True,
                )
        return False

    async def load_symbols(self) -> None:
        """Load accepted symbols from SharedState with a small TTL throttle."""
        now = time.time()
        # if self.symbols is not None:
        #     # user-pre-specified list → still ensure cache keys
        #     for s in self.symbols:
        #         self._ensure_model_cache_key(s)
        #     return

        if (now - self._accepted_snapshot_ts) < self._snapshot_ttl_sec and self._accepted_snapshot:
            self.symbols = list(self._accepted_snapshot)
            return

        accepted = {}
        getter = getattr(self.shared_state, "get_accepted_symbols", None)
        if callable(getter):
            try:
                res = getter(full=True)
            except TypeError:
                res = getter()
            accepted = await res if asyncio.iscoroutine(res) else (res or {})
        if not isinstance(accepted, dict):
            snap = getattr(self.shared_state, "get_accepted_symbols_snapshot", None)
            if callable(snap):
                r = snap()
                accepted = await r if asyncio.iscoroutine(r) else (r or {})
            if not isinstance(accepted, dict):
                accepted = {s: {} for s in (accepted or [])}

        self._accepted_snapshot = list(accepted.keys())
        self._accepted_snapshot_ts = now
        self.symbols = list(accepted.keys())
        logger.info("[%s] 🔄 Loaded %d symbols from SharedState.", self.name, len(self.symbols))

        # warm model cache lazily to avoid startup spikes
        for s in self.symbols:
            self._ensure_model_cache_key(s)

    async def run_once(self) -> None:
        await self.load_symbols()
        logger.info("[%s] run_once start.", self.name)

        try:
            if hasattr(self.shared_state, "is_market_data_ready"):
                ready = await _await_maybe(self.shared_state.is_market_data_ready())
                if not ready:
                    logger.warning("[%s] Market data not ready, skipping.", self.name)
                    return
        except Exception:
            pass
        if not self.symbols:
            logger.warning("[%s] No symbols configured or fetched, skipping.", self.name)
            return

        # bounded concurrency to accelerate processing without overload
        max_conc = int(self._cfg("AGENT_MAX_CONCURRENCY", 6))
        sem = asyncio.Semaphore(max_conc)

        async def _guarded(sym: str):
            async with sem:
                await self._process_symbol(sym)

        await asyncio.gather(*[_guarded(s) for s in self.symbols])

        await self.shared_state.update_component_status(
            component=self.name, status="Operational", detail="run_once completed"
        )
        logger.info("[%s] run_once end.", self.name)

    async def run(self) -> None:
        logger.info("📈 [%s] run() started.", self.__class__.__name__)
        await self.run_once()
        logger.info("[%s] run() finished.", self.name)

    async def run_loop(self) -> None:
        await self.load_symbols()
        interval = int(self._cfg("AGENT_LOOP_INTERVAL", 60))
        logger.info("[%s] 🔁 Starting run_loop @ %ss", self.name, interval)
        try:
            while True:
                try:
                    await self.run_once()
                except Exception as e:
                    logger.error("[%s] ❌ Error in run_loop: %s", self.name, e, exc_info=True)
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            logger.info("[%s] run_loop cancelled; exiting cleanly.", self.name)
            raise

    # ------------- per-symbol flow -------------
    async def _process_symbol(self, symbol: str) -> None:
        logger.debug("[%s] Processing %s.", self.name, symbol)

        # Readiness gating decoupled (Spec Point 5)
        # Agents evaluate even if OpsPlane is gated to facilitate escalation triggers
        # (Prefilter remains to ensure symbol is active/notIONAL OK)
        try:
            logger.debug("[%s] Entering prefilter for %s", self.name, symbol)
            if not await self._prefilter_symbol(symbol):
                logger.debug("[%s] Prefilter REJECTED %s", self.name, symbol)
                return
            logger.debug("[%s] Prefilter PASSED for %s", self.name, symbol)
        except Exception as e:
            logger.debug("[%s] Prefilter error for %s: %s", self.name, symbol, e, exc_info=True)
            return
            
        # volatility guard
        reginfo = None
        regime_tf = str(self._cfg("VOLATILITY_REGIME_TIMEFRAME", self.timeframe) or self.timeframe)
        sym_u = str(symbol).replace("/", "").upper()
        try:
            logger.debug("[%s] Checking volatility regime for %s", self.name, symbol)
            reginfo = await self.shared_state.get_volatility_regime(sym_u, timeframe=regime_tf)
            if not reginfo:
                reginfo = await self.shared_state.get_volatility_regime("GLOBAL", timeframe=regime_tf)
        except Exception as e:
            logger.debug("[%s] Volatility check error for %s: %s", self.name, symbol, e)
            reginfo = None

        regime = (reginfo or {}).get("regime", "")
        regime = str(regime).lower() if regime else ""
        if not regime:
            logger.debug(
                "[%s] Volatility regime not ready for %s (tf=%s), defaulting to NORMAL",
                self.name,
                symbol,
                regime_tf,
            )
            regime = "normal"
        logger.debug("[%s] Regime for %s: %s", self.name, symbol, regime)
        if self.allowed_regimes and regime not in self.allowed_regimes:
            logger.debug("[%s] %s skipped due to disallowed volatility regime: %s", self.name, symbol, regime)
            return
            
        # ensure model presence
        is_ml_capable = False
        logger.debug("[%s] Checking ML capability for %s", self.name, symbol)
        if self.model_cache.get(symbol) is None:
            await self._retrain_if_needed(symbol)
            # We no longer skip if retraining is in progress or missing; we fallback to heuristic.
            if self.model_cache.get(symbol) is not None:
                is_ml_capable = True
        else:
            is_ml_capable = True
            
        logger.debug("[%s] Generating signal for %s (ML=%s)", self.name, symbol, is_ml_capable)
        action, confidence, reason = await self._generate_signal(symbol, is_ml_capable=is_ml_capable)
        
        # Guard freshness and only emit actionable/confident signals
        act = action.upper()
        # Use local helper (global function)
        if act in ("BUY", "SELL"):
            # Route actionable signals to AgentManager collection path.
            await self._submit_signal(symbol, act, float(confidence), reason)
        else:
            logger.debug("[%s] Not emitting for %s: action=%s, conf=%.2f (<%.2f)",
                         self.name, symbol, act, float(confidence), float(self.min_conf))

        # health detail for last symbol processed in this task
        await self.shared_state.update_component_status(
            component=self.name,
            status="Operational",
            detail=f"Last signal: {action} ({confidence:.2f}), regime={regime or ''}",
        )
 
    async def _get_regime_aware_confidence(self, symbol: str, timeframe: str = None) -> str:
        """
        Get volatility regime for signal confidence calculation.
        
        Preferentially uses high timeframe (1h) brain regime for entry decisions,
        falls back to symbol/global regime on the current timeframe.
        
        Returns:
            Regime string: "trend", "sideways", "high_vol", "bear", "normal", etc.
        """
        if timeframe is None:
            timeframe = self.timeframe
        
        try:
            sym_u = str(symbol).replace("/", "").upper()
            
            # Try 1h regime first (longer-term context)
            try:
                reginfo_1h = await self.shared_state.get_volatility_regime(sym_u, timeframe="1h")
                if not reginfo_1h:
                    reginfo_1h = await self.shared_state.get_volatility_regime("GLOBAL", timeframe="1h")
                regime = (reginfo_1h or {}).get("regime", "").lower() if reginfo_1h else None
                if regime:
                    return regime
            except Exception:
                pass
            
            # Fallback to symbol regime on current timeframe
            try:
                reginfo = await self.shared_state.get_volatility_regime(sym_u, timeframe=timeframe)
                if not reginfo:
                    reginfo = await self.shared_state.get_volatility_regime("GLOBAL", timeframe=timeframe)
                regime = (reginfo or {}).get("regime", "").lower() if reginfo else None
                if regime:
                    return regime
            except Exception:
                pass
            
            # Fallback to normal if no regime available
            return "normal"
            
        except Exception as e:
            logger.debug("[%s] Failed to get regime for %s: %s", self.name, symbol, e)
            return "normal"

    def _get_regime_scaling_factors(self, regime: str) -> Dict[str, float]:
        """
        Regime-based scaling factors instead of binary gating.
        
        Returns dict with:
        - position_size_mult: scale position size
        - tp_target_mult: scale TP distance
        - excursion_requirement_mult: scale minimum price movement required
        - trail_mult: scale trailing SL aggressiveness
        - confidence_boost: confidence adjustment
        
        Architecture:
        - trending (uptrend/downtrend): full size, full TP, aggressive trail
        - normal/neutral: baseline (1.0x)
        - sideways/chop: reduced size, reduced TP, increased excursion requirement
        - bear: defensive posture (slightly reduced)
        - high_vol: slightly reduced size, wider trails
        """
        regime_norm = str(regime or "normal").lower()
        
        # Default: no scaling
        default_scaling = {
            "position_size_mult": 1.0,
            "tp_target_mult": 1.0,
            "excursion_requirement_mult": 1.0,
            "trail_mult": 1.0,
            "confidence_boost": 0.0,
            "regime": regime_norm,
        }
        
        # Trending regimes: aggressive
        if regime_norm in {"trend", "uptrend", "downtrend"}:
            return {
                "position_size_mult": 1.0,      # Full size
                "tp_target_mult": 1.0,          # Full TP
                "excursion_requirement_mult": 0.85,  # Easier to trigger
                "trail_mult": 1.3,              # Aggressive trailing
                "confidence_boost": 0.05,       # +5% confidence bonus
                "regime": regime_norm,
            }
        
        # High volatility: be cautious with position size but aggressive with trails
        elif regime_norm in {"high_vol", "high"}:
            return {
                "position_size_mult": 0.80,     # 80% size
                "tp_target_mult": 1.05,         # Slightly wider TP
                "excursion_requirement_mult": 1.0,
                "trail_mult": 1.2,              # Moderately aggressive trailing
                "confidence_boost": 0.0,
                "regime": regime_norm,
            }
        
        # Sideways/Chop: significantly reduced
        elif regime_norm in {"sideways", "chop", "range"}:
            return {
                "position_size_mult": 0.50,     # 50% size
                "tp_target_mult": 0.60,         # 60% of normal TP
                "excursion_requirement_mult": 1.4,  # Require more movement
                "trail_mult": 0.9,              # Tighter trailing (less aggressive)
                "confidence_boost": -0.05,      # -5% confidence penalty
                "regime": regime_norm,
            }
        
        # Bear market: defensive
        elif regime_norm in {"bear", "bearish", "downtrend"}:
            return {
                "position_size_mult": 0.60,     # 60% size
                "tp_target_mult": 0.80,         # 80% TP
                "excursion_requirement_mult": 1.2,  # Harder to trigger
                "trail_mult": 0.95,             # Very tight trailing
                "confidence_boost": -0.08,      # -8% confidence penalty
                "regime": regime_norm,
            }
        
        # Default/Unknown: baseline
        else:
            return default_scaling

    async def _submit_signal(self, symbol: str, action: str, confidence: float, reason: str) -> None:
        action_upper = action.upper().strip()
        if action_upper == "HOLD":
            logger.debug("[%s] HOLD filtered for %s", self.name, symbol)
            return
        min_conf = float(self.min_conf)
        if action_upper == "SELL":
            min_conf = float(self._cfg("TREND_MIN_CONF_SELL", self._cfg("SELL_MIN_CONF", min_conf)))
        if float(confidence) < min_conf:
            logger.debug("[%s] Low-conf filtered for %s: %.2f < %.2f", self.name, symbol, confidence, min_conf)
            return
        
        # Regime-aware scaling for BUY signals (no binary gating, scaling instead)
        regime_1h = "normal"  # default
        regime_scaling = {"position_size_mult": 1.0, "tp_target_mult": 1.0, "excursion_requirement_mult": 1.0, "trail_mult": 1.0, "confidence_boost": 0.0}
        
        if action_upper == "BUY":
            try:
                sym_u = str(symbol).replace("/", "").upper()
                # Get 1h regime (brain)
                try:
                    reginfo_1h = await self.shared_state.get_volatility_regime(sym_u, timeframe="1h")
                    if not reginfo_1h:
                        reginfo_1h = await self.shared_state.get_volatility_regime("GLOBAL", timeframe="1h")
                    regime_1h = (reginfo_1h or {}).get("regime", "").lower() if reginfo_1h else "normal"
                except Exception as e:
                    logger.debug("[%s] Failed to get 1h regime for %s: %s", self.name, symbol, e)
                    regime_1h = "normal"
                
                # Get regime scaling factors (no binary gate)
                regime_scaling = self._get_regime_scaling_factors(regime_1h)
                
                # Apply confidence adjustment based on regime
                adjusted_confidence = float(confidence) + regime_scaling["confidence_boost"]
                
                # Re-check confidence after regime adjustment
                if adjusted_confidence < min_conf:
                    logger.info(
                        "[%s] BUY filtered for %s — regime=%s reduces confidence from %.2f → %.2f (below min %.2f)",
                        self.name,
                        symbol,
                        regime_1h,
                        float(confidence),
                        adjusted_confidence,
                        min_conf,
                    )
                    return
                
                # Only block on bear regime if institutional filter is enabled
                allow_bear_with_high_conf = bool(self._cfg("TREND_ALLOW_BEAR_IF_HIGH_CONF", False))
                if (regime_1h == "bear" and 
                    not (allow_bear_with_high_conf and adjusted_confidence >= 0.85)):
                    logger.info(
                        "[%s] BUY filtered for %s — 1h regime is BEAR (pos_size will be reduced 40% if proceeded)",
                        self.name,
                        symbol,
                    )
                    return
                
                # Log scaling factors being applied
                logger.info(
                    "[%s] BUY allowed for %s | regime=%s | "
                    "pos_size_mult=%.2f tp_mult=%.2f excursion_mult=%.2f trail_mult=%.2f | "
                    "conf_orig=%.2f → adjusted=%.2f",
                    self.name,
                    symbol,
                    regime_1h,
                    regime_scaling["position_size_mult"],
                    regime_scaling["tp_target_mult"],
                    regime_scaling["excursion_requirement_mult"],
                    regime_scaling["trail_mult"],
                    float(confidence),
                    adjusted_confidence,
                )
                
            except Exception as e:
                logger.debug("[%s] Regime scaling error for %s: %s (proceeding with baseline)", self.name, symbol, e)
                regime_scaling = self._get_regime_scaling_factors("normal")
        
        if action_upper == "SELL":
            pos_qty = 0.0
            gpq = getattr(self.shared_state, "get_position_quantity", None)
            if callable(gpq):
                r = gpq(symbol)
                pos_qty = await r if asyncio.iscoroutine(r) else (r or 0.0)
            if pos_qty <= 0:
                logger.info("[%s] Skip SELL for %s — no position.", self.name, symbol)
                return

        quote_hint = (
            float(
                self._cfg(
                    "EMIT_BUY_QUOTE",
                    self._cfg("MIN_ENTRY_USDT", self._cfg("MIN_ENTRY_QUOTE_USDT", 10.0)),
                )
            )
            if action_upper == "BUY"
            else None
        )

        # GAP FIX A: Validate quote against min_notional BEFORE buffering (TrendHunter version)
        if quote_hint is not None:
            MIN_NOTIONAL_FLOOR = float(getattr(self.config, "MIN_NOTIONAL_FLOOR", 5.0))
            if quote_hint < MIN_NOTIONAL_FLOOR * 0.8:  # 80% headroom for fees
                logger.warning(
                    f"[{self.name}] Signal quote {quote_hint:.2f} < min_notional {MIN_NOTIONAL_FLOOR:.2f}; filtering out"
                )
                return  # Don't emit sub-minimum quotes

        # Compute expected move based on TP distance, ML forecast, ATR, and historical ROI
        expected_move_pct = await self._compute_expected_move_pct(symbol, action_upper)
        
        # ALPHA AMPLIFIER: Compute edge score for composite aggregation
        edge = compute_agent_edge(
            agent_name=self.name,
            action=action_upper,
            confidence=float(confidence),
            expected_move_pct=expected_move_pct,
            symbol=symbol,
            timeframe=self.timeframe,
        )
        
        # Single-path emission: buffer for AgentManager collection only.
        # Avoid dual bus+buffer publication because cache key collisions can clobber metadata.
        signal = {
            "symbol": symbol,
            "action": action_upper,
            "side": action_upper,
            "confidence": float(confidence),
            "reason": reason,
            "quote": quote_hint,
            "quote_hint": quote_hint,
            "horizon_hours": 6.0,
            "agent": self.name,
            "expected_move_pct": float(expected_move_pct),  # Alpha signal for EV gate
            "edge": float(edge),  # ALPHA AMPLIFIER: Composite edge for institutional selection
            # NEW: Regime-based scaling factors for MetaController/ExecutionManager
            "_regime_scaling": dict(regime_scaling) if regime_scaling else {},
            "_regime": regime_1h if action_upper == "BUY" else "normal",
        }
        self._collected_signals.append(signal)
        logger.info(
            "[%s] Buffered %s for %s (conf=%.2f, exp_move=%.2f%%, edge=%.3f, regime=%s, pos_scale=%.2f)",
            self.name, action_upper, symbol, confidence, expected_move_pct, edge,
            regime_1h if action_upper == "BUY" else "N/A",
            regime_scaling.get("position_size_mult", 1.0) if regime_scaling else 1.0
        )
        # Always publish TradeIntent to event bus (needed for MetaController drain)
        intent_payload = {
            "symbol": str(symbol).replace("/", "").upper(),
            "side": action_upper,
            "planned_quote": quote_hint,
            "quote_hint": quote_hint,
            "confidence": float(confidence),
            "agent": self.name,
            "tag": f"strategy/{self.name}",
            "reason": str(reason),
            "rationale": str(reason),
            "timeframe": self.timeframe,
            "timestamp": time.time(),
            "policy_context": {
                "expected_move_pct": float(expected_move_pct),
                "edge": float(edge),
                "_regime": regime_1h if action_upper == "BUY" else "normal",
                "_regime_scaling": dict(regime_scaling) if regime_scaling else {},
            },
        }
        await self._publish_trade_intent(intent_payload)

    async def _generate_signal(self, symbol: str, is_ml_capable: bool = False) -> Tuple[str, float, str]:
        """
        Generate a trading signal. If is_ml_capable is True and a model exists,
        use the model for prediction. Otherwise, fallback to MACD/EMA heuristic.
        """
        data = await self._get_market_data_safe(symbol, self.timeframe)
        if data is None:
            return "HOLD", 0.0, "OHLCV None"

        rows = [self._std_row(r) for r in data]
        rows = [r for r in rows if r is not None]

        # 1) Try ML Prediction if capable
        if is_ml_capable:
            model = self.model_cache.get(symbol)
            lookback = int(self._cfg("TRENDHUNTER_RETRAIN_LOOKBACK", 100))
            if model and len(rows) >= lookback:
                try:
                    # Prepare input window
                    window = np.asarray([rows[-lookback:]], dtype=np.float32)
                    pred = model.predict(window, verbose=0)[0]
                    # pred is [Prob(Up), Prob(Down), Prob(Flat)]
                    idx = np.argmax(pred)
                    conf = float(pred[idx])
                    
                    if idx == 0: # Up
                        return "BUY", conf, f"ML Prediction (Up, conf={conf:.2f})"
                    elif idx == 1: # Down
                        return "SELL", conf, f"ML Prediction (Down, conf={conf:.2f})"
                    else:
                        return "HOLD", conf, "ML Prediction (Flat)"
                except Exception as e:
                    logger.warning("[%s] ML prediction failed for %s: %s. Falling back to heuristic.", self.name, symbol, e)
        data = await self._get_market_data_safe(symbol, self.timeframe)
        if data is None:
            return "HOLD", 0.0, "OHLCV None"

        rows = [self._std_row(r) for r in data]
        rows = [r for r in rows if r is not None]

        min_required = 50
        cfg_min = self._cfg("TRENDHUNTER_MIN_DATA", 50)
        if isinstance(cfg_min, dict):
            min_required = int(cfg_min.get(symbol, cfg_min.get("default", 50)))
        elif isinstance(cfg_min, int):
            min_required = cfg_min

        closes = np.asarray([r[3] for r in rows], dtype=float)  # r[3] is 'c'
        fast = int(self._cfg("TRENDHUNTER_EMA_SHORT", self.ema_fast))
        slow = int(self._cfg("TRENDHUNTER_EMA_LONG", self.ema_slow))
        # Ensure adequate lookback for indicator stability (EMA slow + MACD signal + padding)
        macd_signal = 9
        min_required = max(int(min_required), int(slow + macd_signal + 3))
        if len(rows) < min_required:
            return "HOLD", 0.1, f"Insufficient OHLCV ({len(rows)}<{min_required})"

        try:
            if _HAS_TALIB:
                ema_short = talib.EMA(closes, timeperiod=fast)
                ema_long = talib.EMA(closes, timeperiod=slow)
                macd_line, sig_line, hist = talib.MACD(closes)
            else:
                ema_short = compute_ema(closes, fast)
                ema_long = compute_ema(closes, slow)
                macd_line, sig_line, hist = compute_macd(closes)
            # Guard against short series/NaNs at the tail
            tail_vals = [np.asarray(ema_short)[-1], np.asarray(ema_long)[-1], np.asarray(hist)[-1]]
            # np.isnan works elementwise on numpy scalars; convert to float just in case
            if any(np.isnan(float(v)) for v in tail_vals):
                return "HOLD", 0.0, "Indicator NaN"
        except Exception as e:
            logger.error("[%s] Indicator calc failed for %s: %s", self.name, symbol, e, exc_info=True)
            return "HOLD", 0.0, "Indicator error"

        # 2) Fallback to MACD/EMA Heuristic with Volatility-Adjusted Confidence
        s_val = float(np.asarray(ema_short)[-1])
        l_val = float(np.asarray(ema_long)[-1])
        h_val = float(np.asarray(hist)[-1])
        
        logger.debug("[%s] Heuristic check for %s: EMA_S=%.2f EMA_L=%.2f HIST=%.6f", 
                     self.name, symbol, s_val, l_val, h_val)
        
        # Get regime for volatility-aware confidence (CRITICAL FIX)
        regime = await self._get_regime_aware_confidence(symbol)
        
        # Compute volatility-adjusted confidence instead of hardcoded 0.70
        h_conf = compute_heuristic_confidence(
            hist_value=h_val,
            hist_values=np.asarray(hist[-50:], dtype=float) if len(hist) >= 50 else np.asarray(hist, dtype=float),
            regime=regime,
            closes=closes[-50:] if len(closes) >= 50 else closes,
        )
        
        # Get metrics for detailed logging
        metrics = get_signal_quality_metrics(
            hist_values=np.asarray(hist[-50:], dtype=float) if len(hist) >= 50 else np.asarray(hist, dtype=float),
            regime=regime,
            closes=closes[-50:] if len(closes) >= 50 else closes,
        )
        
        # Determine action from histogram sign
        if h_val > 0:
            action = "BUY"
        elif h_val < 0:
            action = "SELL"
        else:
            return "HOLD", 0.0, "No clear heuristic signal"
        
        # Log confidence breakdown for debugging
        logger.info(
            "[%s] %s heuristic for %s (regime=%s) | "
            "mag=%.4f accel=%.4f raw=%.3f → adj=%.3f (floor=%.2f) → final=%.3f",
            self.name,
            action,
            symbol,
            regime,
            metrics.get("histogram_magnitude", 0),
            metrics.get("histogram_acceleration", 0),
            metrics.get("raw_confidence", 0),
            metrics.get("adjusted_confidence", 0),
            metrics.get("regime_floor", 0),
            h_conf,
        )
        
        return action, h_conf, f"Heuristic MACD {action.title()} (hist={h_val:.6f}, conf={h_conf:.3f}, regime={regime})"

    async def _compute_expected_move_pct(self, symbol: str, action: str) -> float:
        """
        Compute expected move percentage based on:
        1. TP/SL distance from current price
        2. ML forecast (if available)
        3. ATR multiple volatility adjustment
        4. Historical ROI on similar setups
        
        Returns: Expected move as percentage (0-100+)
        
        This is the TRUE alpha signal for EV gate evaluation,
        replacing the fallback ATR-only approach.
        """
        try:
            # Get current market data
            data = await self._get_market_data_safe(symbol, self.timeframe)
            if not data or len(data) == 0:
                logger.debug("[%s] No market data for %s, using ATR fallback", self.name, symbol)
                return 1.5  # Conservative fallback
            
            rows = [self._std_row(r) for r in data]
            rows = [r for r in rows if r is not None]
            
            if len(rows) == 0:
                return 1.5
            
            current_close = float(rows[-1][3])  # r[3] is close
            
            # === Component 1: TP/SL Distance ===
            tp_pct = 0.0
            try:
                if self.tp_sl_engine:
                    tp, sl = self.tp_sl_engine.calculate_tp_sl(symbol, current_close)
                    if action == "BUY" and tp > current_close:
                        tp_pct = ((tp - current_close) / current_close) * 100
                    elif action == "SELL" and sl < current_close:
                        tp_pct = ((current_close - sl) / current_close) * 100
                    logger.debug("[%s] TP distance for %s: %.2f%%", self.name, symbol, tp_pct)
            except Exception as e:
                logger.debug("[%s] TP/SL calculation failed for %s: %s", self.name, symbol, e)
                tp_pct = 0.0
            
            # CRITICAL FIX: Prevent TP component from collapsing to zero
            # If TP calculation fails/returns invalid, fallback to ATR-based estimate
            # This ensures the 40% TP weight contributes meaningfully to expected_move
            if tp_pct <= 0:
                # Calculate ATR first to use as reference (computed below)
                # For now, mark that we need fallback—ATR will be computed next
                tp_pct = None  # Signal to use ATR fallback after ATR calculation
            
            # === Component 2: ATR Multiple Volatility ===
            atr_pct = 1.5  # Default conservative move
            try:
                closes = np.asarray([r[3] for r in rows], dtype=float)
                highs = np.asarray([r[1] for r in rows], dtype=float)
                lows = np.asarray([r[2] for r in rows], dtype=float)
                
                if _HAS_TALIB:
                    atr_val = talib.ATR(highs, lows, closes, timeperiod=14)[-1]
                else:
                    # Manual ATR: (H-L, L-PC, PC-H) with exponential smoothing
                    tr_list = []
                    for i in range(len(highs)):
                        tr = max(
                            highs[i] - lows[i],
                            abs(highs[i] - closes[i-1]) if i > 0 else 0,
                            abs(lows[i] - closes[i-1]) if i > 0 else 0
                        )
                        tr_list.append(tr)
                    atr_val = np.mean(tr_list[-14:]) if len(tr_list) >= 14 else np.mean(tr_list)
                
                atr_pct = (atr_val / current_close) * 100
                atr_pct = max(1.5, atr_pct)  # Floor at 1.5%
                logger.debug("[%s] ATR-based move for %s: %.2f%%", self.name, symbol, atr_pct)
            except Exception as e:
                logger.debug("[%s] ATR calculation failed for %s: %s", self.name, symbol, e)
                atr_pct = 1.5
            
            # CRITICAL FIX: Apply TP fallback if it collapsed to zero
            # Use ATR * 1.2 as conservative estimate for TP distance
            # This prevents the 40% TP weight from completely collapsing
            if tp_pct is None or tp_pct <= 0:
                tp_pct = atr_pct * 1.2
                logger.warning("[%s] TP calculation invalid for %s; using ATR fallback: %.2f%%", 
                             self.name, symbol, tp_pct)
            
            # === Component 3: ML Forecast (if available) ===
            ml_pct = 0.0
            ml_weight = 0.0
            try:
                if self.model_cache.get(symbol):
                    model = self.model_cache.get(symbol)
                    lookback = int(self._cfg("TRENDHUNTER_RETRAIN_LOOKBACK", 100))
                    if len(rows) >= lookback:
                        window = np.asarray([rows[-lookback:]], dtype=np.float32)
                        pred = model.predict(window, verbose=0)[0]
                        
                        # pred[0] = Up probability, pred[1] = Down probability
                        # Use predicted direction's probability as confidence
                        if action == "BUY":
                            ml_confidence = float(pred[0])  # Up probability
                        else:
                            ml_confidence = float(pred[1])  # Down probability
                        
                        # Map confidence to expected move magnitude (0.5 to 4.0%)
                        ml_pct = 1.5 + (ml_confidence * 2.5)
                        ml_weight = 0.15  # 15% weight for ML component
                        logger.debug("[%s] ML predicted move for %s: %.2f%% (conf=%.2f)", self.name, symbol, ml_pct, ml_confidence)
            except Exception as e:
                logger.debug("[%s] ML forecast failed for %s: %s", self.name, symbol, e)
                ml_pct = 0.0
                ml_weight = 0.0
            
            # === Component 4: Historical ROI on similar setups ===
            # (Simplified: use recent win rate as proxy for setup quality)
            roi_pct = 1.0
            try:
                if self.trades_count > 5:
                    win_rate = self.win_count / max(self.trades_count, 1)
                    # High win rate → higher expected move on future trades
                    # Low win rate → conservative
                    roi_pct = 1.0 + (win_rate * 2.0)  # Range: 1.0-3.0%
                    logger.debug("[%s] Historical ROI factor for %s: %.2f%% (wins=%d/%d)", 
                                self.name, symbol, roi_pct, self.win_count, self.trades_count)
            except Exception as e:
                logger.debug("[%s] Historical ROI calculation failed: %s", self.name, e)
                roi_pct = 1.0
            
            # === Aggregate Expected Move ===
            # Weighted average: TP (40%) + ATR (30%) + ML (15% base, +15% if available) + Historical (15%)
            weights = {
                "tp": (tp_pct, 0.40),
                "atr": (atr_pct, 0.30),
                "ml": (ml_pct, 0.15 + ml_weight),  # Boost ML weight if available (0.15 or 0.30)
                "roi": (roi_pct, 0.15),
            }
            
            # Properly normalize: sum all weights and divide once at the end
            total_weight = sum(w for _, w in weights.values())
            if total_weight == 0:
                return 1.5  # Fallback
            
            # Correct weighted average: sum(component * weight) / total_weight
            # This ensures consistent results whether ML is available or not
            weighted_sum = sum(val * w for val, w in weights.values())
            expected_move = weighted_sum / total_weight
            
            # CRITICAL FIX #5: Remove redundant EV_MULTIPLIER from TrendHunter
            # ExecutionManager already applies regime-based multiplier (1.3/1.8/2.0)
            # Having TWO multipliers creates hidden 2.14x barrier in normal regime
            # Solution: Let ExecutionManager be the sole EV gate authority
            # expected_move stays as raw weighted average (no additional scaling)
            
            logger.info("[%s] Expected move for %s (%s): %.2f%% (TP=%.2f, ATR=%.2f, ML=%.2f, ROI=%.2f) [RAW, no EV multiplier applied]",
                       self.name, symbol, action, expected_move, tp_pct, atr_pct, ml_pct, roi_pct)
            
            return max(0.5, expected_move)  # Floor at 0.5%
            
        except Exception as e:
            logger.warning("[%s] Expected move calculation failed for %s: %s, using fallback", self.name, symbol, e)
            return 1.5  # Conservative fallback


def _iso_now():
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def _emit_health(ss, component: str, status: str, message: str):
    """Fire-and-forget health event; works if emit_event is sync or async."""
    try:
        if ss and hasattr(ss, "emit_event"):
            res = ss.emit_event(
                "HealthStatus",
                {
                    "component": component,
                    "status": status,
                    "message": message,
                    "timestamp": _iso_now(),
                },
            )
            if asyncio.iscoroutine(res):
                try:
                    asyncio.create_task(res)
                except Exception:
                    pass
    except Exception:
        # Never let telemetry crash the agent loop
        pass

def _norm_trade_intent(agent: str, x, default_ttl: float = 30.0):
    """Normalize various signal dicts/objects into a P9 trade-intent dict."""
    def g(obj, k, d=None):
        return getattr(obj, k, None) if hasattr(obj, k) else (obj.get(k, d) if isinstance(obj, dict) else d)

    symbol = g(x, "symbol")
    side = g(x, "side")
    if not symbol or not side:
        return None

    confidence = float(g(x, "confidence", 0.0) or 0.0)
    ttl_sec = float(g(x, "ttl_sec", default_ttl) or default_ttl)
    tag = g(x, "tag") or f"strategy/{agent}"
    quote_hint = g(x, "quote_hint")
    quantity = g(x, "quantity") or g(x, "qty_hint")
    rationale = g(x, "rationale")
    ts = g(x, "ts") or time.time()

    return {
        "symbol": symbol,
        "side": side,
        "agent": agent,
        "confidence": confidence,
        "ttl_sec": ttl_sec,
        "tag": tag,
        "quote_hint": quote_hint,
        "quantity": quantity,
        "rationale": rationale,
        "ts": float(ts),
    }

def _norm_exec_order(source: str, x, tag_default: str):
    """Normalize order-like dicts/objects into an ExecOrder-ish dict."""
    def g(obj, k, d=None):
        return getattr(obj, k, None) if hasattr(obj, k) else (obj.get(k, d) if isinstance(obj, dict) else d)

    symbol = g(x, "symbol")
    side = g(x, "side")
    if not symbol or not side:
        return None

    quantity = g(x, "quantity") or g(x, "qty") or g(x, "qty_hint")
    quote_hint = g(x, "quote_hint")
    tag = g(x, "tag") or tag_default

    eo = {
        "symbol": symbol,
        "side": side,
        "quantity": quantity,
        "source": source,
        "tag": tag,
    }
    if quote_hint is not None:
        eo["quote_hint"] = quote_hint
    return eo

# ===== P9 Strategy Normalization Wrappers (added) =====
try:
    _CLASS = TrendHunter
except NameError:
    _CLASS = None

if _CLASS is not None:
    if hasattr(_CLASS, "collect_signals"):
        if not hasattr(_CLASS, "_collect_signals_raw"):
            _CLASS._collect_signals_raw = _CLASS.collect_signals

        async def collect_signals(self):
            out = await _CLASS._collect_signals_raw(self)
            arr = out if isinstance(out, (list, tuple)) else ([out] if out is not None else [])
            norm = [y for y in ([_norm_trade_intent(getattr(self, "name", "TrendHunter"), x) for x in arr]) if y]
            _emit_health(getattr(self, "shared_state", None), "TrendHunter", "Running", f"signals={len(norm)}")
            return norm
        _CLASS.collect_signals = collect_signals

    if not hasattr(_CLASS, "warmup"):
        async def warmup(self):
            _emit_health(getattr(self, "shared_state", None), "TrendHunter", "Running", "warmup ok")
        _CLASS.warmup = warmup

    if not hasattr(_CLASS, "health"):
        def health(self):
            return {"component": "TrendHunter", "status": "Running", "timestamp": _iso_now()}
        _CLASS.health = health
