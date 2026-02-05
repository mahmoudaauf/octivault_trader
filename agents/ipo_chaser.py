import os
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger("IPOChaser")  # inherit level from app

# Light optional import helper (avoid import-time hard failures)
def _lazy_imports():
    mods = {}
    try:
        import numpy as np
        mods["np"] = np
    except Exception:
        mods["np"] = None
    try:
        import pandas as pd
        mods["pd"] = pd
    except Exception:
        mods["pd"] = None
    return mods

def _safe_bool_tuple(res) -> Tuple[bool, str]:
    if isinstance(res, tuple):
        ok = bool(res[0])
        reason = str(res[1]) if len(res) > 1 else ""
        return ok, reason
    return bool(res), ""

class IPOChaser:
    """
    IPO-focused discovery + (optional) ML signaler.

    - Constructor deps are OPTIONAL so Phase-3 probing/registration won't crash.
    - Discovery path (propose newly listed *USDT pairs*) is lightweight.
    - ML path (run single-symbol decision) is lazy-imported and guarded.
    """
    agent_type = "discovery"

    def __init__(
        self,
        shared_state: Any,
        config: Any,
        exchange_client: Optional[Any] = None,
        symbol_manager: Optional[Any] = None,
        execution_manager: Optional[Any] = None,
        tp_sl_engine: Optional[Any] = None,
        interval: int = 900,
    ):
        self.shared_state = shared_state
        self.config = config
        self.exchange_client = exchange_client
        self.symbol_manager = symbol_manager
        self.execution_manager = execution_manager
        self.tp_sl_engine = tp_sl_engine

        self.interval = max(30, int(interval))
        self.name = "IPOChaser"
        self.is_discovery_agent = True
        self.timeframe = str(getattr(self.config, "IPO_TIMEFRAME", "5m"))

        # runtime caches for ML mode
        self.model_cache: Dict[str, Any] = {}
        self._model_locks: Dict[str, asyncio.Lock] = {}
        self._model_ttl = int(getattr(self.config, "MODEL_TTL_SECONDS", 0))  # 0 = no TTL
        self._lookback_default = 50
        # lifecycle & concurrency
        self._stop_event = asyncio.Event()
        self._task = None
        self._lock = asyncio.Lock()
        self._running = False

        logger.info(
            f"[{self.name}] Initialized "
            f"(has_exch={self.exchange_client is not None}, "
            f"has_symmgr={self.symbol_manager is not None}, "
            f"has_exec={self.execution_manager is not None}, "
            f"has_tpsl={self.tp_sl_engine is not None}, "
            f"interval={self.interval}s)"
        )

    # -------------------- Discovery (lightweight) --------------------

    async def start(self):
        """
        P9 contract: start() spawns the periodic discovery loop once (idempotent).
        """
        if getattr(self, "_task", None) and not self._task.done():
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(self.run_loop(), name="agent.ipo_chaser")
        logger.info(f"[{self.name}] start() launched background loop.")

    async def stop(self):
        """
        P9 contract: stop() requests the loop to end and waits for it.
        """
        self._stop_event.set()
        t = getattr(self, "_task", None)
        self._task = None
        if t:
            try:
                t.cancel()
                try:
                    await asyncio.wait_for(t, timeout=float(getattr(self.config, "STOP_JOIN_TIMEOUT_S", 5.0)))
                except asyncio.CancelledError:
                    pass
            except Exception:
                logger.debug(f"[{self.name}] stop wait failed", exc_info=True)
        logger.info(f"[{self.name}] stopped.")

    async def run_once(self):
        """
        One-shot discovery pass:
          - calls exchange_client.get_new_listings()
          - proposes *USDT pairs* to SymbolManager
        Reentrancy-guarded to avoid overlapping runs during startup scheduling.
        """
        if self._running:
            logger.debug(f"[{self.name}] run_once skipped: already running")
            return
        async with self._lock:
            if self._running:
                return
            self._running = True
            try:
                if not self.exchange_client:
                    logger.info(f"[{self.name}] Skipping discovery: exchange_client not wired.")
                    return
                if not self.symbol_manager:
                    logger.info(f"[{self.name}] Skipping discovery: symbol_manager not wired.")
                    return
                if not hasattr(self.exchange_client, "get_new_listings"):
                    logger.warning(f"[{self.name}] Exchange client lacks get_new_listings().")
                    return

                logger.info(f"[{self.name}] run_once: scanning new listings...")
                try:
                    listings = await asyncio.wait_for(self.exchange_client.get_new_listings(), timeout=15)
                except asyncio.TimeoutError:
                    logger.warning(f"[{self.name}] ⏰ get_new_listings timeout.")
                    return
                except Exception as e:
                    logger.exception(f"[{self.name}] get_new_listings error: {e}")
                    return

                if not listings:
                    logger.info(f"[{self.name}] No IPO listings found.")
                    return

                usdt_pairs = [s for s in listings if isinstance(s, str) and s.endswith("USDT")]
                if not usdt_pairs:
                    logger.info(f"[{self.name}] No USDT IPO pairs in listings.")
                    return

                logger.info(f"[{self.name}] Found USDT IPO symbols: {usdt_pairs}")
                for sym in usdt_pairs:
                    try:
                        res = await asyncio.wait_for(
                            self.symbol_manager.propose_symbol(sym, source=self.name, metadata={"reason": "IPO candidate"}),
                            timeout=10,
                        )
                        ok, reason = _safe_bool_tuple(res)
                        if ok:
                            logger.info(f"✅ Accepted {sym} from {self.name}.")
                        else:
                            logger.warning(f"❌ {sym} rejected from {self.name}: {reason or 'no reason'}")
                    except asyncio.TimeoutError:
                        logger.warning(f"[{self.name}] ⏰ propose_symbol timeout for {sym}")
                    except Exception as e:
                        logger.exception(f"[{self.name}] propose_symbol error for {sym}: {e}")
                    await asyncio.sleep(0.05)  # tiny spacing
            finally:
                self._running = False

    async def run_loop(self):
        logger.info(f"🚀 [{self.name}] run_loop started. Interval: {self.interval}s")
        while not self._stop_event.is_set():
            try:
                await self.run_once()
            except Exception as e:
                logger.error(f"[{self.name}] discovery error: {e}", exc_info=True)
            await asyncio.sleep(self.interval + min(1.0, 0.05 * self.interval))  # mild jitter

    async def run_discovery(self):
        """Alias so AgentManager can call a one-shot discovery in Phase-3."""
        await self.run_once()

    # -------------------- Optional ML signaler (lazy deps) --------------------

    async def run(self, symbol: Optional[str], **kwargs) -> Tuple[Dict[str, Any], float]:
        """
        Single-symbol decision path (optional ML).
        Safe even if ML deps are not present; returns HOLD.
        """
        if not symbol or symbol == "N/A":
            return self._signal("hold", 0.0, reason="Invalid symbol"), 0.0

        if not self.enabled:
            return self._signal("hold", 0.0, reason="Deactivated"), 0.0

        ohlcv = await self._get_ohlcv(symbol, lookback)
        if ohlcv is None:
            return self._signal("hold", 0.0, reason="Insufficient/Malformed OHLCV"), 0.0

        model = await self._ensure_model(symbol, ohlcv)
        if model is None:
            return self._signal("hold", 0.0, reason="Model unavailable"), 0.0

        predicted_gain = await self._predict_gain(model, ohlcv)
        action = "buy" if predicted_gain >= gain_threshold else "hold"
        sig = self._signal(action, predicted_gain, **kwargs)

        # publish signal to bus if available
        await self._emit_signal(symbol, sig)
        if action == "buy":
            await self._maybe_execute_buy(symbol, sig)
        return sig, float(sig["confidence"])

    # -------------------- Internals (ML path) --------------------

    def _signal(self, action: str, predicted_gain: float, **kwargs) -> Dict[str, Any]:
        safe_meta = {k: v for k, v in (kwargs or {}).items() if not asyncio.iscoroutine(v)}
        return {
            "source": self.name,
            "action": str(action).lower(),
            "confidence": float(predicted_gain),
            "timestamp": datetime.utcnow().isoformat(),
            "meta": {"predicted_gain": float(predicted_gain), **safe_meta},
        }

    @property
    def enabled(self) -> bool:
        return bool(self._cfg("IPOCHASER_ENABLED", True))

    @property
    def gain_threshold(self) -> float:
        return float(self._cfg("IPO_GAIN_THRESHOLD", 0.01))

    @property
    def lookback(self) -> int:
        return int(self._cfg("IPO_LOOKBACK", self._lookback_default))

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

    def _load_tuned(self, key: str) -> Dict[str, Any]:
        try:
            from core.agent_optimizer import load_tuned_params
            return load_tuned_params(key) or {}
        except Exception:
            return {}

    async def _get_ohlcv(self, symbol: str, lookback: int):
        mods = _lazy_imports()
        np = mods["np"]
        if not np:
            logger.debug(f"[{self.name}] numpy not available; skipping ML path.")
            return None

        # try shared cache first
        try:
            cached = (getattr(self.shared_state, "historical_data", {}) or {}).get(symbol, {}).get(self.timeframe, [])
        except Exception:
            cached = []

        if len(cached) < lookback:
            if not self.exchange_client or not hasattr(self.exchange_client, "get_ohlcv"):
                return None
            try:
                cached = await asyncio.wait_for(
                    self.exchange_client.get_ohlcv(symbol, timeframe=self.timeframe, limit=lookback),
                    timeout=15,
                )
            except asyncio.TimeoutError:
                logger.warning(f"[{self.name}] ⏰ get_ohlcv timeout for {symbol}")
                return None
            except Exception as e:
                logger.warning(f"[{self.name}] get_ohlcv error for {symbol}: {e}")
                return None

        if not isinstance(cached, list) or len(cached) < lookback:
            return None

        sample = cached[-lookback:]
        req = ("open", "high", "low", "close", "volume")
        if not all(isinstance(c, dict) and all(k in c for k in req) for c in sample):
            return None

        arr = np.asarray([[c["open"], c["high"], c["low"], c["close"], c["volume"]] for c in sample], dtype=np.float32)
        return arr.reshape(1, lookback, 5)

    async def _ensure_model(self, symbol: str, ohlcv_np):
        path = self._model_path(symbol)
        # fast path
        if symbol in self.model_cache and not self._model_stale(path):
            return self.model_cache[symbol]

        lock = self._model_locks.setdefault(symbol, asyncio.Lock())
        async with lock:
            if symbol in self.model_cache and not self._model_stale(path):
                return self.model_cache[symbol]

            # (train if missing/stale)
            if self._model_stale(path) or not self._model_exists(path):
                ok = await self._train_model(symbol, ohlcv_np, path)
                if not ok:
                    return None

            model = self._safe_load_model(path)
            if model is not None:
                self.model_cache[symbol] = model
            return model

    def _model_path(self, symbol: str) -> str:
        try:
            from core.model_manager import build_model_path
            return build_model_path(self.name, symbol)
        except Exception:
            # fallback location
            return os.path.join("models", f"{self.name}_{symbol}.keras")

    def _model_exists(self, path: str) -> bool:
        try:
            from core.model_manager import model_exists
            return bool(model_exists(path))
        except Exception:
            return os.path.exists(path)

    def _model_stale(self, path: str) -> bool:
        if self._model_ttl <= 0:
            return False
        try:
            mtime = os.path.getmtime(path)
            return (mtime + self._model_ttl) < datetime.utcnow().timestamp()
        except Exception:
            return True

    async def _train_model(self, symbol: str, ohlcv_np, path: str) -> bool:
        mods = _lazy_imports()
        pd = mods["pd"]
        if not pd:
            logger.warning(f"[{self.name}] pandas not available; cannot train.")
            return False
        try:
            from core.model_trainer import ModelTrainer
            from core.model_manager import save_model
        except Exception as e:
            logger.warning(f"[{self.name}] Trainer/manager imports failed: {e}")
            return False

        try:
            df = pd.DataFrame(ohlcv_np.reshape(-1, 5), columns=["open", "high", "low", "close", "volume"])
            trainer = ModelTrainer(symbol, self.timeframe, model_dir=os.path.dirname(path) or "models")
            trained = trainer.train_model(df)
            if trained not in (True, False, None):
                try:
                    save_model(trained, path)
                except Exception:
                    logger.debug(f"[{self.name}] save_model failed (trainer likely persisted).", exc_info=True)
            return True
        except Exception as e:
            logger.error(f"[{self.name}] Training failed for {symbol}: {e}", exc_info=True)
            return False

    def _safe_load_model(self, path: str):
        try:
            from core.model_manager import safe_load_model
            return safe_load_model(path)
        except Exception as e:
            logger.error(f"[{self.name}] safe_load_model failed: {e}")
            return None

    async def _predict_gain(self, model, ohlcv_np) -> float:
        try:
            loop = asyncio.get_running_loop()
            out = await loop.run_in_executor(None, lambda: float(model.predict(ohlcv_np, verbose=0)[0][0]))
            if not (out == out) or out == float("inf") or out == float("-inf"):
                return 0.0
            return max(min(out, 5.0), -5.0)
        except Exception as e:
            logger.error(f"[{self.name}] Prediction failed: {e}", exc_info=True)
            return 0.0

    async def _emit_signal(self, symbol: str, signal: Dict[str, Any]) -> None:
        # Mandatory P9 Signal Contract: Emit to Signal Bus
        if hasattr(self.shared_state, "add_agent_signal"):
            try:
                await self.shared_state.add_agent_signal(
                    symbol=symbol,
                    agent=self.name,
                    side=signal.get("action", "hold").upper(),
                    confidence=float(signal.get("confidence", 0.0)),
                    ttl_sec=300,
                    tier="B",
                    rationale=signal.get("reason", "IPO Chaser ML prediction")
                )
            except Exception as e:
                logger.warning(f"[{self.name}] Failed to emit to signal bus: {e}")

    async def _maybe_execute_buy(self, symbol: str, signal: Dict[str, Any]) -> None:
        if not self.execution_manager:
            return
        price = await self._get_price(symbol)
        if price is None or price <= 0:
            return

        tp, sl = None, None
        if self.tp_sl_engine:
            try:
                tp, sl = self.tp_sl_engine.calculate_tp_sl(symbol, price)
            except Exception:
                logger.debug(f"[{self.name}] TP/SL calc error", exc_info=True)

        try:
            usdt_bal = float((getattr(self.shared_state, "balances", {}) or {}).get("USDT", 0.0) or 0.0)
            qty = round((usdt_bal / price) * 0.1, 6) if usdt_bal > 10 else 0.0
            if qty <= 0:
                return

            await self.execution_manager.execute_trade(
                symbol=symbol,
                side="buy",
                qty=qty,
                mode="market",
                take_profit=tp,
                stop_loss=sl,
                comment=f"{self.name}_strategy",
            )
        except Exception:
            logger.debug(f"[{self.name}] execute_trade failed", exc_info=True)

    async def _get_price(self, symbol: str) -> Optional[float]:
        try:
            if hasattr(self.shared_state, "get_latest_price_safe"):
                return await self.shared_state.get_latest_price_safe(symbol)

            md = getattr(self.shared_state, "market_data", {}) or {}
            px = (md.get(symbol) or {}).get("close")
            if px:
                return float(px)

            if self.exchange_client and hasattr(self.exchange_client, "get_current_price"):
                try:
                    return float(await asyncio.wait_for(self.exchange_client.get_current_price(symbol), timeout=10))
                except asyncio.TimeoutError:
                    return None
        except Exception:
            logger.debug(f"[{self.name}] price lookup failed", exc_info=True)
        return None
