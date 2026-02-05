import asyncio
import logging
import os
import time
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf  # retrain uses to_thread; non-blocking for loop

_HAS_TALIB = True
try:
    import talib
except Exception:
    _HAS_TALIB = False

from utils.status_logger import log_component_status
from utils.indicators import compute_ema, compute_macd
from core.model_manager import safe_load_model, build_model_path
from agents.signal_utils import emit_to_meta, is_fresh  # New import for signal utilities
from core.baseline_trading_kernel import ExecOrder

AGENT_NAME = "TrendHunter"

# -------------------------
# Logging (idempotent setup)
# -------------------------
logger = logging.getLogger(AGENT_NAME)
logger.setLevel(logging.INFO)

_log_path = f"logs/agents/{AGENT_NAME.lower()}.log"
os.makedirs(os.path.dirname(_log_path), exist_ok=True)

if not any(isinstance(h, logging.FileHandler) and getattr(h, "_trendhunter", False) for h in logger.handlers):
    fh = logging.FileHandler(_log_path)
    fh._trendhunter = True  # mark to avoid duplicates
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] [%(name)s] %(message)s"))
    logger.addHandler(fh)

# -------------------------
# Agent
# -------------------------
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
        self.min_conf = float(self._cfg("MIN_SIGNAL_CONF", self._tuned_cache.get("MIN_SIGNAL_CONF", 0.55)))

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

        log_component_status(self.name, "Initialized")
        logger.info("🚀 %s initialized (timeframe=%s, symbols=%d)", self.name, self.timeframe, len(self.symbols or []))

    # ------------- helpers -------------
    def _cfg(self, key: str, default: Any = None) -> Any:
        if isinstance(self.config, dict):
            return self.config.get(key, default)
        return getattr(self.config, key, default)

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
            path = build_model_path(self.name, sym)
            model = safe_load_model(path)
            if model is None:
                logger.info("[%s] No pre-trained model for %s (will train on first run).", self.name, sym)
            self.model_cache[sym] = model
        except Exception as e:
            logger.debug("[%s] safe_load_model failed for %s: %s", self.name, sym, e)
            self.model_cache[sym] = None

    # ------------- lifecycle -------------
    async def generate_signals(self) -> List[Any]:
        await self.load_symbols()
        await self.run_once()
        return []

    async def load_symbols(self) -> None:
        """Load accepted symbols from SharedState with a small TTL throttle."""
        now = time.time()
        if self.symbols is not None:
            # user-pre-specified list → still ensure cache keys
            for s in self.symbols:
                self._ensure_model_cache_key(s)
            return

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
            if hasattr(self.shared_state, "is_market_data_ready") and not self.shared_state.is_market_data_ready():
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

        # volatility guard
        reginfo = None
        try:
            reginfo = await self.shared_state.get_volatility_regime(symbol, timeframe=self.timeframe)
        except Exception:
            reginfo = None
        regime = (reginfo or {}).get("regime", "").lower()
        if regime == "high":
            logger.info("[%s] %s skipped due to high volatility.", self.name, symbol)
            return

        # ensure model presence (optional for your current simple signal logic)
        if self.model_cache.get(symbol) is None:
            ok = await self._retrain_if_needed(symbol)
            if not ok:
                return

        action, confidence, reason = await self._generate_signal(symbol)
        
        # Guard freshness and only emit actionable/confident signals
        act = action.upper()
        if not await is_fresh(self.shared_state, symbol, max_age_sec=120):
            logger.debug(f"[{self.name}] Stale data; skip {symbol}")
        else:
            if act in ("BUY", "SELL") and float(confidence) >= float(self.min_conf):
                await emit_to_meta(
                    self.meta_controller,
                    self.name,
                    symbol,
                    action=act,
                    confidence=float(confidence),
                    quote=10.0 if act == "BUY" else None,
                    horizon_hours=6.0,
                )
            else:
                logger.debug("[%s] Not emitting: action=%s, conf=%.2f (<%.2f)",
                             self.name, act, float(confidence), float(self.min_conf))


        # optional execution
        if action.upper() in ("BUY", "SELL") and float(confidence) >= float(self.min_conf):
            try:
                await self._maybe_execute(symbol, action, float(confidence), reason)
            except Exception as e:
                logger.error("[%s] ❌ Execution block failed for %s: %s", self.name, symbol, e, exc_info=True)

        # health detail for last symbol processed in this task
        await self.shared_state.update_component_status(
            component=self.name,
            status="Operational",
            detail=f"Last signal: {action} ({confidence:.2f})",
        )

    async def _maybe_execute(self, symbol: str, action: str, confidence: float, reason: str) -> None:
        # fetch price via shared_state helper with exchange fallback
        get_price_cb = getattr(self.execution_manager, "get_current_price", None)
        if self.execution_manager and hasattr(self.execution_manager, "exchange_client"):
            get_price_cb = getattr(self.execution_manager.exchange_client, "get_price", get_price_cb)

        price = await self.shared_state.get_latest_price(symbol)
        if (not price or price <= 0) and callable(get_price_cb):
            try:
                fetched = await asyncio.to_thread(get_price_cb, symbol)
                if fetched and fetched > 0:
                    await self.shared_state.update_latest_price(symbol, float(fetched))
                    price = float(fetched)
            except Exception:
                price = price or 0.0
        if not price or price <= 0:
            logger.info("[%s] %s – price_unavailable; skip execution.", self.name, symbol)
            return

        # normalize side to canonical lowercase ("buy" | "sell")
        side = "buy" if action.strip().upper() == "BUY" else ("sell" if action.strip().upper() == "SELL" else "")
        if side not in ("buy", "sell"):
            logger.debug("[%s] Non-actionable side for %s: %s", self.name, symbol, action)
            return
        
        if side == "buy":
            # spendable budget
            spendable = await self.shared_state.get_spendable_balance(self.base_ccy)
            if not spendable or spendable <= 0:
                logger.info("[%s] %s – no spendable %s; skip buy.", self.name, symbol, self.base_ccy)
                return

            # fractional budget, clamped to [0..1]
            fraction = float(self._cfg("BUY_FRACTION_OF_SPENDABLE", 0.10))
            desired_quote = float(spendable) * max(0.0, min(1.0, fraction))

            # pull filters to learn min entry constraint
            nf = await self.execution_manager.ensure_symbol_filters_ready(symbol)
            step, min_qty, max_qty, tick, min_notional = self.execution_manager._extract_filter_vals(nf)
            cfg_min_entry = (
                float(getattr(getattr(self.config, "GLOBAL", object()), "MIN_ENTRY_QUOTE_USDT", 10.0))
                if not isinstance(self.config, dict)
                else float(self.config.get("GLOBAL", {}).get("MIN_ENTRY_QUOTE_USDT", 10.0))
            )
            min_entry = max(float(min_notional or 0.0), cfg_min_entry)

            # ensure we try at least the minimum
            quote_to_spend = max(desired_quote, min_entry)

            # ask EM to confirm affordability (fees + headroom)
            ok, reason = await self.execution_manager.explain_afford_market_buy(symbol, quote_to_spend)
            if not ok:
                ok_min, reason_min = await self.execution_manager.explain_afford_market_buy(symbol, min_entry)
                if not ok_min:
                    logger.info("[%s] %s – cannot afford min entry (%s); skip buy.", self.name, symbol, reason_min)
                    return
                quote_to_spend = float(min_entry)

            order = ExecOrder(
                symbol=symbol,
                side="buy",
                planned_quote=float(quote_to_spend),
                tag=f"meta-{AGENT_NAME}",
            )
        else:
            pos_qty = float(await self.shared_state.get_position_quantity(symbol) or 0.0)
            if pos_qty <= 0:
                logger.info("[%s] Skip SELL for %s — no position.", self.name, symbol)
                return

            nf = await self.execution_manager.ensure_symbol_filters_ready(symbol)
            step, min_qty, max_qty, tick, min_notional = self.execution_manager._extract_filter_vals(nf)

            q_qty = self.execution_manager.exchange_client.quantize_qty(symbol, float(pos_qty))
            if max_qty > 0 and q_qty > max_qty:
                q_qty = self.execution_manager.exchange_client.quantize_qty(symbol, max_qty)
            if q_qty <= 0:
                logger.info("[%s] Skip SELL for %s — qty rounds to zero.", self.name, symbol)
                return

            notional = float(q_qty) * float(price)
            if min_notional and notional < float(min_notional):
                logger.info("[%s] Skip SELL for %s — notional %.4f < minNotional %.4f.",
                            self.name, symbol, notional, float(min_notional))
                return

            order = ExecOrder(
                symbol=symbol,
                side="sell",
                quantity=float(q_qty),
                tag=f"meta-{AGENT_NAME}",
            )

        await self.execution_manager.place(order)
        if order.side == "buy" and order.planned_quote is not None:
            logger.info("[%s] ✅ Order submitted via EM.place for %s (buy, quote=%.2f)",
                        self.name, symbol, float(order.planned_quote))
        else:
            logger.info("[%s] ✅ Order submitted via EM.place for %s (sell, qty=%.6f)",
                        self.name, symbol, float(order.quantity or 0.0))


    async def _submit_signal(self, symbol: str, action: str, confidence: float, reason: str) -> None:
        # basic filters
        if action.upper() == "HOLD":
            logger.debug("[%s] HOLD filtered for %s", self.name, symbol)
            return
        if float(confidence) < float(self.min_conf):
            logger.debug("[%s] Low-conf filtered for %s: %.2f < %.2f", self.name, symbol, confidence, self.min_conf)
            return
        if action.upper() == "SELL":
            pos_qty = 0.0
            gpq = getattr(self.shared_state, "get_position_quantity", None)
            if callable(gpq):
                r = gpq(symbol)
                pos_qty = await r if asyncio.iscoroutine(r) else (r or 0.0)
            if pos_qty <= 0:
                logger.info("[%s] Skip SELL for %s — no position.", self.name, symbol)
                return

        signal = {
            "agent_name": self.name,
            "symbol": symbol,
            "action": action,
            "confidence": float(confidence),
            "reason": reason,
            "timestamp": time.time(),
        }

        if getattr(self, "meta_controller", None):
            await self.meta_controller.receive_signal(
                agent_name=self.name,
                symbol=symbol,
                signal={
                    "action": signal["action"],
                    "confidence": signal["confidence"],
                    "reason": signal["reason"],
                },
            )
            logger.info("[%s] Submitted %s for %s to MetaController (conf=%.2f)", self.name, action, symbol, confidence)
        else:
            try:
                await self.shared_state.push_signal(symbol, signal)
                logger.info("[%s] Submitted %s for %s via SharedState (conf=%.2f)", self.name, action, symbol, confidence)
            except Exception as e:
                logger.error("[%s] Failed to submit signal for %s: %s", self.name, symbol, e, exc_info=True)

    async def _generate_signal(self, symbol: str) -> Tuple[str, float, str]:
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

        if len(rows) < min_required:
            return "HOLD", 0.1, f"Insufficient OHLCV ({len(rows)}<{min_required})"

        closes = np.asarray([r[3] for r in rows], dtype=float)  # r[3] is 'c'
        fast = int(self._cfg("TRENDHUNTER_EMA_SHORT", self.ema_fast))
        slow = int(self._cfg("TRENDHUNTER_EMA_LONG", self.ema_slow))

        try:
            if _HAS_TALIB:
                ema_short = talib.EMA(closes, timeperiod=fast)
                ema_long = talib.EMA(closes, timeperiod=slow)
                macd_line, sig_line, hist = talib.MACD(closes)
            else:
                ema_short = compute_ema(closes, fast)
                ema_long = compute_ema(closes, slow)
                macd_line, sig_line, hist = compute_macd(closes)
        except Exception as e:
            logger.error("[%s] Indicator calc failed for %s: %s", self.name, symbol, e, exc_info=True)
            return "HOLD", 0.0, "Indicator error"

        if ema_short[-1] > ema_long[-1] and hist[-1] > 0:
            return "BUY", 0.8, "Bullish MACD"
        if ema_short[-1] < ema_long[-1] and hist[-1] < 0:
            return "SELL", 0.8, "Bearish MACD"
        return "HOLD", 0.0, "No clear signal"

    async def _retrain_if_needed(self, symbol: str) -> bool:
        data = await self._get_market_data_safe(symbol, self.timeframe)
        if data is None:
            logger.info("[%s] OHLCV None for %s@%s; skip retrain.", self.name, symbol, self.timeframe)
            self.model_cache[symbol] = None
            return False

        rows = [self._std_row(r) for r in data]
        rows = [r for r in rows if r is not None]

        lookback = int(self._cfg("TRENDHUNTER_RETRAIN_LOOKBACK", 100))
        if len(rows) < lookback:
            logger.info("[%s] Cannot retrain %s; insufficient data (%d<%d).", self.name, symbol, len(rows), lookback)
            self.model_cache[symbol] = None
            return False

        X, y = [], []
        for i in range(lookback, len(rows)):
            window = rows[i - lookback : i]
            X.append(window)
            future = rows[i][3]         # c
            current = window[-1][3]     # c
            y.append([1,0,0] if future > current else [0,1,0] if future < current else [0,0,1])

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(lookback, 5)),
                tf.keras.layers.LSTM(32),
                tf.keras.layers.Dense(3, activation="softmax"),
            ]
        )
        model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])

        try:
            await asyncio.to_thread(model.fit, X, y, epochs=5, batch_size=32, verbose=0)
            path = build_model_path(self.name, symbol)
            self.model_manager.save_model(model, path)
            self.model_cache[symbol] = model
            logger.info("[%s] Retrained and saved model for %s → %s", self.name, symbol, path)
            return True
        except Exception as e:
            logger.error("[%s] Retrain failed for %s: %s", self.name, symbol, e, exc_info=True)
            self.model_cache[symbol] = None
            return False



# ===== P9 Spec Helpers (added) =====
import asyncio, time
from datetime import datetime

def _iso_now():
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def _emit_health(ss, component: str, status: str, message: str):
    try:
        if ss and hasattr(ss, "emit_event"):
            res = ss.emit_event("HealthStatus", {
                "component": component,
                "status": status,
                "message": message,
                "timestamp": _iso_now()
            })
            if asyncio.iscoroutine(res):
                try:
                    asyncio.create_task(res)
                except Exception:
                    pass
    except Exception:
        pass

def _norm_trade_intent(agent: str, x, default_ttl: float = 30.0):
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
    def g(obj, k, d=None):
        return getattr(obj, k, None) if hasattr(obj, k) else (obj.get(k, d) if isinstance(obj, dict) else d)
    symbol = g(x, "symbol")
    side = g(x, "side")
    quantity = g(x, "quantity") or g(x, "qty") or g(x, "qty_hint")
    quote_hint = g(x, "quote_hint")
    tag = g(x, "tag") or tag_default
    if not symbol or not side:
        return None
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
            _emit_health(getattr(self, "shared_state", None), "TrendHunter", "Running", f"signals={{len(norm)}}")
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
