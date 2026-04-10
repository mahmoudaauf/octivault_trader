# core/market_data_feed.py
from __future__ import annotations

import asyncio
import contextlib
import random
import time
import math
from typing import Any, Dict, List, Optional, Iterable, Tuple, Callable
import logging

# ── Optional enums from SharedState (don’t hard-fail if missing) ──────────────
try:
    from core.shared_state import Component as _ComponentEnum  # optional
except Exception:
    _ComponentEnum = None


try:
    from core.shared_state import HealthCode as _HealthEnum  # optional
except Exception:
    _HealthEnum = None

# ── ExchangeClient lazy public bootstrap (optional) ───────────────────────────
try:
    from core.exchange_client import ensure_public_bootstrap as _ensure_ec_public
except Exception:
    _ensure_ec_public = None

# ── WebSocket market data feed (optional, new hybrid architecture) ────────────
try:
    from core.market_data_websocket import MarketDataWebSocket
except Exception:
    MarketDataWebSocket = None  # type: ignore


API_AUTH_ERR_CODES = {-2015, -2014}        # Invalid key/permissions/signature
API_RATELIMIT_ERR_CODES = {-1003, -1015, -1021}   # Rate limit, too many requests, time skew


class MarketDataFeed:
    """
    Octivault Trader (P9) — MarketDataFeed (resilient build)
      • Warmup: bulk OHLCV + last price for accepted symbols.
      • Steady loop: refresh prices + incremental candles with bounded concurrency + jitter.
      • Observability: periodic health pings (works with enums or plain strings).
      • Hygiene: ascending ts, 6-field OHLCV [ts,o,h,l,c,v] (epoch seconds float).
      • Optional ATR warm cache on 1h.
    """

    # ---- late wiring hooks (safe no-ops if unused) ----
    def set_exchange_client(self, ec: Any) -> None:
        """Allow AppContext to inject/replace the ExchangeClient after construction."""
        self.exchange_client = ec
        try:
            self._exch_ready_evt.set()
        except Exception:
            pass

    def set_shared_state(self, ss: Any) -> None:
        """Allow AppContext to inject/replace the SharedState after construction."""
        self.shared_state = ss

    def __init__(
        self,
        shared_state: Any,
        exchange_client: Any,
        *,
        config: Optional[Dict[str, Any]] = None,
        ohlcv_timeframes: Optional[List[str]] = None,
        ohlcv_limit: int = 100,
        poll_interval: float = 15.0,
        max_concurrency: int = 8,
        logger: Optional[logging.Logger] = None,
        compute_atr: bool = True,
        health_cadence_sec: float = 10.0,
        warmup_timeout_sec: float = 45.0,
        retry_after_no_symbols_sec: float = 2.0,
        max_retry_backoff_sec: float = 20.0,
        min_bars_required: int = 50,
        readiness_emit: bool = True,
        per_symbol_readiness: bool = True,
        **_,
    ) -> None:
        # Initialize logger FIRST before any logging calls
        self._logger = logger or logging.getLogger("MarketDataFeed")
        if self._logger.level == logging.NOTSET:
            self._logger.setLevel(logging.INFO)
        
        self.shared_state = shared_state
        self.exchange_client = exchange_client
        self._ec_public_ok: bool = True  # allow public-only bootstrap for unsigned endpoints

        cfg = config or {}
        self.config = cfg
        
        # Helper to access config whether it's a dict or object
        def _cfg(key, default=None):
            if isinstance(cfg, dict):
                return cfg.get(key, default)
            return getattr(cfg, key, default)
        
        tfs = (
            _cfg("ohlcv_timeframes")
            or _cfg("SUPPORTED_TIMEFRAMES")
            or ohlcv_timeframes
            or ["1m", "5m"]  # default now includes 5m
        )
        if isinstance(tfs, str):
            tfs = [t.strip() for t in tfs.split(",") if t.strip()]
        self.timeframes: List[str] = [str(t).strip() for t in tfs]
        # MDF_OHLCV_LIMIT supports higher candle fetches for better training data
        raw_limit = _cfg("MDF_OHLCV_LIMIT") or _cfg("ohlcv_limit", ohlcv_limit)
        try:
            limit = int(raw_limit)
        except Exception:
            limit = 5000
        if limit < 50:
            limit = 50
        self.ohlcv_limit: int = limit
        self.poll_interval: float = float(_cfg("poll_interval", poll_interval))
        self.max_concurrency: int = int(_cfg("max_concurrency", max_concurrency))
        self.compute_atr: bool = bool(_cfg("compute_atr", compute_atr))
        self.jitter_max: float = float(_cfg("jitter_max", 0.7))
        self.health_cadence_sec: float = float(_cfg("health_cadence_sec", health_cadence_sec))
        self.warmup_timeout_sec: float = float(_cfg("warmup_timeout_sec", warmup_timeout_sec))
        self.retry_after_no_symbols_sec: float = float(_cfg("retry_after_no_symbols_sec", retry_after_no_symbols_sec))
        self.max_retry_backoff_sec: float = float(_cfg("max_retry_backoff_sec", max_retry_backoff_sec))
        self.max_retry_attempts: int = int(_cfg("max_retry_attempts", 6))
        self.min_bars_required: int = int(_cfg("min_bars_required", min_bars_required))
        self.readiness_emit: bool = bool(_cfg("readiness_emit", readiness_emit))
        self.per_symbol_readiness: bool = bool(_cfg("per_symbol_readiness", per_symbol_readiness))
        self._declared_ready: bool = False

        # WebSocket market data feed (NEW: hybrid architecture)
        self.enable_websocket: bool = bool(_cfg("ENABLE_WEBSOCKET", True))
        self.websocket_feed: Optional[Any] = None  # MarketDataWebSocket instance
        self._websocket_task: Optional[asyncio.Task] = None
        if self.enable_websocket and MarketDataWebSocket:
            self._logger.info("[MDF] WebSocket support enabled (hybrid mode)")
        
        # Helper to access config
        self._cfg = _cfg

        # Internal event to wait for late wiring of the exchange client (AppContext P3.65+)
        self._exch_ready_evt: asyncio.Event = asyncio.Event()
        if self.exchange_client is not None:
            self._exch_ready_evt.set()

        self._stop = asyncio.Event()
        self._run_loop_entered = False  # guard against multiple concurrent run() entries

        self._health_task: Optional[asyncio.Task] = None
        self._last_error_ts: float = 0.0
        self._empty_symbol_cycles: int = 0
        self._missing_exchange_cycles: int = 0
        self._poll_cycle: int = 0
        self._known_symbols: set[str] = set()
        self._backfill_tasks: Dict[str, asyncio.Task] = {}

        # Initialize symbols dict for WebSocket feed (hybrid mode)
        self.symbols: Dict[str, Any] = {}  # Will be populated at runtime from _get_accepted_symbols()

        # Resolve component and code values (enum if present, else strings)
        self._component_key = getattr(_ComponentEnum, "MARKET_DATA_FEED", "MarketDataFeed")
        self._code_ok = getattr(_HealthEnum, "OK", "OK")
        self._code_warn = getattr(_HealthEnum, "WARN", "WARN")
        self._code_error = getattr(_HealthEnum, "ERROR", "ERROR")

    async def _await_exchange_ready(self, timeout: float) -> bool:
        """
        Wait for an injected exchange client capable of public endpoints.
        Returns True when ready, False on timeout.
        """
        if self.exchange_client is not None:
            return True
        try:
            await asyncio.wait_for(self._exch_ready_evt.wait(), timeout=timeout)
            return self.exchange_client is not None
        except asyncio.TimeoutError:
            return False

    async def _start_websocket(self) -> None:
        """
        Start the WebSocket market data feed for hybrid architecture.
        Runs in background; initializes WebSocket if enabled and available.
        Gracefully handles if WebSocket component not available (REST fallback).
        """
        if not self.enable_websocket or not MarketDataWebSocket:
            self._logger.debug("[MDF] WebSocket disabled or unavailable")
            return

        try:
            self._logger.info("[MDF] Starting WebSocket feed (hybrid mode)...")
            self.websocket_feed = MarketDataWebSocket(
                shared_state=self.shared_state,
                logger=self._logger,
                is_testnet=bool(self._cfg("IS_TESTNET", False)),
            )
            self._websocket_task = asyncio.create_task(self.websocket_feed.start())
            self._logger.info("[MDF] WebSocket feed started in background")
        except Exception as e:
            self._logger.warning(
                "[MDF] Failed to start WebSocket feed, falling back to REST: %s", str(e)
            )
            self.websocket_feed = None
            self._websocket_task = None

    # -------------------- utils --------------------

    @staticmethod
    def _sanitize_ohlcv(rows: Iterable[Iterable[float]]) -> List[List[float]]:
        rows = [r for r in rows if r is not None and len(r) >= 6]
        rows.sort(key=lambda r: float(r[0]))
        return [[float(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])] for r in rows]

    @staticmethod
    async def _maybe_await(val):
        if asyncio.iscoroutine(val):
            return await val
        return val

    @staticmethod
    def _normalize_symbol_payload(payload: Any) -> List[str]:
        if isinstance(payload, dict):
            raw = list(payload.keys())
        elif isinstance(payload, (list, tuple, set)):
            raw = list(payload)
        else:
            raw = []
        out: List[str] = []
        for item in raw:
            sym = str(item or "").strip().upper()
            if sym:
                out.append(sym)
        return out

    @staticmethod
    def _coerce_positive_price(raw_price: Any) -> float:
        """Best-effort coercion of exchange price payloads to a positive finite float."""
        candidate = raw_price
        if isinstance(raw_price, dict):
            for key in ("price", "lastPrice", "last_price", "close", "c"):
                if key in raw_price:
                    candidate = raw_price.get(key)
                    break
        try:
            val = float(candidate)
        except Exception:
            return 0.0
        if not math.isfinite(val) or val <= 0:
            return 0.0
        return val

    async def _inject_latest_price(self, sym: str, price: float) -> bool:
        """
        Inject latest price into SharedState using resilient fallback chain:
        1) update_latest_price
        2) update_last_price
        3) direct map write (guarded by prices lock when available)
        """
        ss = self.shared_state
        write_methods = ("update_latest_price", "update_last_price")
        for name in write_methods:
            fn = getattr(ss, name, None)
            if not callable(fn):
                continue
            try:
                await self._maybe_await(fn(sym, price))
                return True
            except Exception:
                self._logger.warning(
                    "[MDF] %s failed for %s price=%.10f",
                    name,
                    sym,
                    price,
                    exc_info=True,
                )

        # Last-resort fallback to keep price cache live even if wrappers fail.
        try:
            norm_fn = getattr(ss, "_norm_sym", None)
            norm_sym = norm_fn(sym) if callable(norm_fn) else str(sym or "").upper()
            now = time.time()
            lock = None
            try:
                lock = getattr(ss, "_locks", {}).get("prices")
            except Exception:
                lock = None

            if isinstance(lock, asyncio.Lock):
                async with lock:
                    if isinstance(getattr(ss, "latest_prices", None), dict):
                        ss.latest_prices[norm_sym] = float(price)
                    if isinstance(getattr(ss, "_last_tick_timestamps", None), dict):
                        ss._last_tick_timestamps[norm_sym] = now
                    if isinstance(getattr(ss, "_price_cache", None), dict):
                        ss._price_cache[norm_sym] = (float(price), now)
            else:
                if isinstance(getattr(ss, "latest_prices", None), dict):
                    ss.latest_prices[norm_sym] = float(price)
                if isinstance(getattr(ss, "_last_tick_timestamps", None), dict):
                    ss._last_tick_timestamps[norm_sym] = now
                if isinstance(getattr(ss, "_price_cache", None), dict):
                    ss._price_cache[norm_sym] = (float(price), now)

            self._logger.warning(
                "[MDF] injected latest price via direct fallback for %s price=%.10f",
                norm_sym,
                price,
            )
            return True
        except Exception:
            self._logger.error(
                "[MDF] failed to inject latest price for %s price=%.10f",
                sym,
                price,
                exc_info=True,
            )
            return False

    async def _set_health(self, code, msg: str, metrics: Optional[Dict[str, Any]] = None):
        """
        Send health via SharedState.set_component_health if available.
        Works with enums or strings; never crashes the caller.
        Includes WebSocket metrics if available (NEW).
        """
        try:
            merged_metrics = metrics or {}
            
            # ⚡ Include WebSocket metrics if available (NEW)
            if self.websocket_feed and hasattr(self.websocket_feed, "get_stats"):
                try:
                    ws_stats = await self._maybe_await(self.websocket_feed.get_stats())
                    if isinstance(ws_stats, dict):
                        merged_metrics["websocket"] = ws_stats
                except Exception:
                    pass
            
            fn = getattr(self.shared_state, "set_component_health", None)
            if not fn:
                return
            await self._maybe_await(fn(self._component_key, code, msg, metrics=merged_metrics))
        except Exception:
            self._logger.debug("set_component_health failed", exc_info=True)

    async def _get_accepted_symbols(self) -> List[str]:
        getters = ("get_accepted_symbols", "get_accepted_symbols_snapshot")
        for name in getters:
            try:
                fn = getattr(self.shared_state, name, None)
                if not callable(fn):
                    continue
                payload = await self._maybe_await(fn())
                syms = self._normalize_symbol_payload(payload)
                if syms:
                    return syms
            except Exception:
                self._logger.debug("%s failed", name, exc_info=True)
        try:
            syms = self._normalize_symbol_payload(getattr(self.shared_state, "accepted_symbols", {}))
            if syms:
                return syms
        except Exception:
            self._logger.debug("accepted_symbols fallback failed", exc_info=True)
        return []

    async def _maybe_set_ready(self):
        """Best-effort signal to SharedState that market data is ready."""
        self._logger.warning(f"[MDF DEBUG] SharedState ID: {id(self.shared_state)}")
        if self._declared_ready or not self.readiness_emit:
            return
        try:
            # Prefer explicit setter if present
            fn = getattr(self.shared_state, "set_market_data_ready", None)
            if callable(fn):
                await self._maybe_await(fn(True))
                self._declared_ready = True
                return
            # Fall back to setting an asyncio.Event if exposed
            evt = getattr(self.shared_state, "market_data_ready", None)
            if isinstance(evt, asyncio.Event):
                evt.set()
                self._declared_ready = True
        except Exception:
            self._logger.debug("set_market_data_ready failed", exc_info=True)

    async def _get_exchange_client(self) -> Any:
        """
        Ensure an ExchangeClient is available. If not injected yet, lazily bootstrap a
        public-only client so unsigned GET endpoints (price/klines/exchangeInfo) work.
        """
        ec = getattr(self, "exchange_client", None)
        if ec is not None:
            return ec
        if _ensure_ec_public is not None and getattr(self, "_ec_public_ok", False):
            try:
                ec = await _ensure_ec_public(
                    config=getattr(self, "config", None),
                    logger=self._logger,
                    app=getattr(self, "app", None),
                    shared_state=self.shared_state,
                )
                self.exchange_client = ec
                try:
                    self._exch_ready_evt.set()
                except Exception:
                    pass
                return ec
            except Exception:
                pass
        return getattr(self, "exchange_client", None)

    async def _symbol_meets_depth(self, sym: str) -> bool:
        try:
            has_fn = getattr(self.shared_state, "has_ohlcv", None)
            count_fn = getattr(self.shared_state, "get_ohlcv_count", None)
            for tf in self.timeframes:
                enough = False
                if callable(has_fn):
                    enough = bool(await self._maybe_await(has_fn(sym, tf, self.min_bars_required)))
                elif callable(count_fn):
                    n = await self._maybe_await(count_fn(sym, tf))
                    enough = int(n or 0) >= self.min_bars_required
                if not enough:
                    return False
            return True
        except Exception:
            self._logger.debug("_symbol_meets_depth failed for %s", sym, exc_info=True)
            return False

    async def _mark_symbol_ready(self, sym: str) -> None:
        if not self.per_symbol_readiness:
            return
        try:
            fn = getattr(self.shared_state, "mark_symbol_data_ready", None)
            if callable(fn):
                await self._maybe_await(fn(sym))
                return
            emit = getattr(self.shared_state, "emit_event", None)
            if callable(emit):
                await self._maybe_await(emit("SymbolDataReady", {"symbol": sym}))
        except Exception:
            self._logger.debug("mark_symbol_ready failed for %s", sym, exc_info=True)

    async def _schedule_symbol_backfill(self, symbols: List[str]) -> None:
        """
        Schedule full-window backfill for newly accepted symbols.
        Keeps symbol ATR/indicator paths warm instead of waiting on tail polling.
        """
        for raw_sym in symbols:
            sym = str(raw_sym or "").strip().upper()
            if not sym:
                continue
            existing = self._backfill_tasks.get(sym)
            if existing and not existing.done():
                continue
            try:
                if await self._symbol_meets_depth(sym):
                    continue
            except Exception:
                pass

            self._logger.info("[MDF] scheduling accepted-symbol backfill for %s", sym)
            task = asyncio.create_task(self.on_symbol_accepted(sym), name=f"mdf.backfill[{sym}]")
            self._backfill_tasks[sym] = task

            def _done(t: asyncio.Task, symbol: str = sym) -> None:
                self._backfill_tasks.pop(symbol, None)
                try:
                    exc = t.exception()
                except asyncio.CancelledError:
                    return
                except Exception:
                    return
                if exc is not None:
                    self._logger.warning("[MDF] accepted-symbol backfill failed for %s: %s", symbol, exc)

            task.add_done_callback(_done)

    async def _cancel_backfill_tasks(self) -> None:
        if not self._backfill_tasks:
            return
        tasks = list(self._backfill_tasks.values())
        for t in tasks:
            if not t.done():
                t.cancel()
        with contextlib.suppress(Exception):
            await asyncio.gather(*tasks, return_exceptions=True)
        self._backfill_tasks.clear()

    # -------------------- error handling --------------------

    @staticmethod
    def _extract_api_code(err: Exception) -> Optional[int]:
        code = getattr(err, "code", None)
        if isinstance(code, int):
            return code
        s = str(err)
        try:
            if "code=" in s:
                frag = s.split("code=")[1].split(")")[0]
                return int(frag)
        except Exception:
            pass
        return None

    def _classify_error(self, err: Exception) -> Tuple[str, Dict[str, Any]]:
        code = self._extract_api_code(err)
        kind = "ExternalAPIError"
        if code in API_AUTH_ERR_CODES:
            kind = "AuthError"
        elif code in API_RATELIMIT_ERR_CODES:
            kind = "RateLimit"
            # 🎛️ Notify governor of rate limit
            try:
                if (hasattr(self, 'shared_state') and self.shared_state and 
                    hasattr(self.shared_state, '_app_context') and self.shared_state._app_context):
                    app = self.shared_state._app_context
                    if hasattr(app, 'capital_symbol_governor') and app.capital_symbol_governor:
                        app.capital_symbol_governor.mark_api_rate_limited()
            except Exception:
                pass  # Silently fail if governor unavailable
        elif isinstance(err, asyncio.TimeoutError):
            kind = "Timeout"
        meta = {"error": str(err)}
        if code is not None:
            meta["code"] = code
        return kind, meta

    @staticmethod
    def _parse_op_label(what: str) -> Dict[str, Any]:
        """
        Parse op label like "warmup.get_ohlcv[BTCUSDT,1m]" or
        "poll.get_price[ETHUSDT]" into structured fields.
        """
        out: Dict[str, Any] = {"op": str(what)}
        try:
            label = str(what)
            if "[" in label and "]" in label:
                op = label.split("[", 1)[0]
                args = label.split("[", 1)[1].rsplit("]", 1)[0]
                parts = [p.strip() for p in args.split(",") if p.strip()]
                out["op"] = op
                if parts:
                    out["symbol"] = parts[0]
                if len(parts) > 1:
                    out["timeframe"] = parts[1]
            else:
                out["op"] = label
        except Exception:
            return {"op": str(what)}
        return out

    async def _with_retries(self, coro_fn: Callable[[], asyncio.Future], what: str):
        # exponential backoff w/ jitter; capped
        attempt = 0
        while not self._stop.is_set():
            try:
                return await coro_fn()
            except Exception as e:
                kind, meta = self._classify_error(e)
                try:
                    self._last_error_ts = time.time()
                except Exception:
                    pass
                op_meta = self._parse_op_label(what)
                if op_meta:
                    meta = {**meta, **op_meta}
                await self._set_health(self._code_error, f"{what}:{kind}", metrics=meta)
                # If it's an auth error, back off harder and cap attempts (prevents hot-looping on -2015)
                if kind == "AuthError":
                    base = 6.0
                    if attempt >= 3:
                        raise
                else:
                    base = 1.0
                attempt += 1
                if attempt >= max(1, int(getattr(self, "max_retry_attempts", 6) or 6)):
                    raise
                backoff = min(self.max_retry_backoff_sec, base * (2 ** min(attempt, 6)))
                backoff += random.uniform(0.0, 0.5)
                await asyncio.sleep(backoff)

    # -------------------- lifecycle --------------------

    async def start(self):
        """Starts: health ticker → warmup → run loop (all in the caller task)."""
        self._health_task = asyncio.create_task(self._health_loop(), name="mdf.health")
        try:
            await self._set_health(self._code_ok, "start")
            await self.warmup()
            await self.run()
        finally:
            await self._set_health(self._code_warn, "stopped")
            await self._cancel_backfill_tasks()
            if self._health_task:
                self._health_task.cancel()
                with contextlib.suppress(Exception):
                    await self._health_task

    async def _health_loop(self):
        # Report baseline health independently of run()/warmup() progress
        try:
            while not self._stop.is_set():
                syms = await self._get_accepted_symbols()
                try:
                    ec = await self._get_exchange_client()
                    if ec is not None and hasattr(ec, "api_key") and not getattr(ec, "api_key", None):
                        recent_err = False
                        try:
                            recent_err = (time.time() - float(self._last_error_ts or 0.0)) <= (self.health_cadence_sec * 2.0)
                        except Exception:
                            recent_err = False
                        code = self._code_warn if recent_err else self._code_ok
                        msg = "heartbeat(public-only, recent_errors)" if recent_err else "heartbeat(public-only)"
                        await self._set_health(code, msg, metrics={"symbols": len(syms)})
                        await asyncio.sleep(self.health_cadence_sec)
                        continue
                except Exception:
                    pass
                recent_err = False
                try:
                    recent_err = (time.time() - float(self._last_error_ts or 0.0)) <= (self.health_cadence_sec * 2.0)
                except Exception:
                    recent_err = False
                code = self._code_warn if recent_err else self._code_ok
                msg = "heartbeat(recent_errors)" if recent_err else "heartbeat"
                await self._set_health(code, msg, metrics={"symbols": len(syms)})
                await asyncio.sleep(self.health_cadence_sec)
        except asyncio.CancelledError:
            pass
        except Exception:
            self._logger.debug("health_loop crashed", exc_info=True)

    # -------------------- warmup --------------------

    async def warmup(self) -> None:
        await self._set_health(self._code_ok, "warmup:start")

        # Ensure we have an exchange client; lazily bootstrap a public-only client if needed
        ec = await self._get_exchange_client()
        if ec is None:
            await self._set_health(self._code_warn, "warmup:no_exchange_client")
            return

        symbols = await self._get_accepted_symbols()
        if not symbols:
            # Wait for accepted symbols readiness (phase P5 gate)
            try:
                ready_evt = getattr(self.shared_state, "accepted_symbols_ready", None)
                if isinstance(ready_evt, asyncio.Event):
                    await asyncio.wait_for(ready_evt.wait(), timeout=self.warmup_timeout_sec)
                    symbols = await self._get_accepted_symbols()
                else:
                    await asyncio.sleep(self.retry_after_no_symbols_sec)
                    symbols = await self._get_accepted_symbols()
            except asyncio.TimeoutError:
                await self._set_health(self._code_warn, "warmup:timeout_waiting_accepted_symbols")
                return

        if not symbols:
            await self._set_health(self._code_warn, "warmup:no_accepted_symbols")
            return

        sem = asyncio.Semaphore(self.max_concurrency)

        async def _bulk_add_ohlcv(sym: str, tf: str, rows: Iterable[Iterable[float]]):
            for r in rows:
                bar = {
                    "ts": float(r[0]),
                    "o": float(r[1]),
                    "h": float(r[2]),
                    "l": float(r[3]),
                    "c": float(r[4]),
                    "v": float(r[5]),
                }
                await self._maybe_await(self.shared_state.add_ohlcv(sym, tf, bar))

        async def _load_symbol(sym: str):
            async with sem:
                # Full window for each timeframe
                for tf in self.timeframes:
                    async def _fetch_ohlcv():
                        return await ec.get_ohlcv(sym, tf, limit=self.ohlcv_limit)
                    rows = await self._with_retries(_fetch_ohlcv, f"warmup.get_ohlcv[{sym},{tf}]")
                    rows = self._sanitize_ohlcv(rows or [])
                    if rows:
                        await _bulk_add_ohlcv(sym, tf, rows)

                # Last price
                async def _fetch_price():
                    return await ec.get_current_price(sym)
                price = await self._with_retries(_fetch_price, f"warmup.get_price[{sym}]")
                price_f = self._coerce_positive_price(price)
                if price_f > 0:
                    await self._inject_latest_price(sym, price_f)
                else:
                    self._logger.warning("[MDF] warmup price invalid for %s: %r", sym, price)

                # Optional ATR on 1h
                if self.compute_atr and "1h" in self.timeframes:
                    try:
                        _ = await self._maybe_await(self.shared_state.calc_atr(sym, "1h", 14))
                    except Exception:
                        # Non-fatal if not supported
                        pass

                # If this symbol meets depth, emit per-symbol readiness
                try:
                    if await self._symbol_meets_depth(sym):
                        await self._mark_symbol_ready(sym)
                except Exception:
                    self._logger.debug("per-symbol readiness check failed for %s", sym, exc_info=True)

        # Execute warmup with bounded concurrency
        t0 = time.perf_counter()
        results = await asyncio.gather(*(_load_symbol(s) for s in symbols), return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                kind, meta = self._classify_error(r)
                await self._set_health(self._code_error, f"warmup.task:{kind}", metrics=meta)
        latency_ms = int((time.perf_counter() - t0) * 1000)

        # Validate OHLCV depth and declare readiness if all tfs meet minimum bars
        all_ok = True
        has_fn = getattr(self.shared_state, "has_ohlcv", None)
        count_fn = getattr(self.shared_state, "get_ohlcv_count", None)
        missing: Dict[str, Dict[str, int]] = {}
        for s in symbols:
            for tf in self.timeframes:
                enough = False
                try:
                    if callable(has_fn):
                        enough = bool(await self._maybe_await(has_fn(s, tf, self.min_bars_required)))
                    elif callable(count_fn):
                        n = await self._maybe_await(count_fn(s, tf))
                        enough = int(n or 0) >= self.min_bars_required
                    else:
                        # If no API exists, assume ok if we fetched anything
                        get_fn = getattr(self.shared_state, "get_ohlcv", None)
                        if callable(get_fn):
                            rows = await self._maybe_await(get_fn(s, tf, limit=self.min_bars_required))
                            enough = len(list(rows or [])) >= self.min_bars_required
                except Exception:
                    self._logger.debug("depth check failed for %s %s", s, tf, exc_info=True)
                if not enough:
                    all_ok = False
                    missing.setdefault(s, {})[tf] = self.min_bars_required
        if all_ok:
            await self._set_health(self._code_ok, "warmup:depth_ok", metrics={"symbols": len(symbols), "min_bars": self.min_bars_required})
            await self._maybe_set_ready()
        else:
            await self._set_health(self._code_warn, "warmup:insufficient_bars", metrics={"missing": {k: sorted(v.keys()) for k, v in missing.items()}, "min_bars": self.min_bars_required})
            # Allow partial readiness to unblock degraded-mode trading
            try:
                for s in symbols:
                    if await self._symbol_meets_depth(s):
                        await self._maybe_set_ready()
                        break
            except Exception:
                self._logger.debug("partial readiness check failed", exc_info=True)

        # Readiness: SharedState flips MarketDataReady internally once thresholds are met.
        await self._set_health(self._code_ok, "warmup:done", metrics={"symbols": len(symbols), "latency_ms": latency_ms})
        self._known_symbols = {str(s or "").strip().upper() for s in symbols if str(s or "").strip()}

    async def on_symbol_accepted(self, sym: str) -> None:
        """Blocking backfill for a single symbol; can be wired to an AcceptedSymbol event."""
        try:
            ec = await self._get_exchange_client()
            if ec is None:
                kind, meta = ("Timeout", {"error": "exchange_client_missing"})
                await self._set_health(self._code_error, f"on_symbol_accepted:{kind}", metrics=meta)
                return
            # Fetch full window for each timeframe
            for tf in self.timeframes:
                async def _fetch_ohlcv():
                    return await ec.get_ohlcv(sym, tf, limit=self.ohlcv_limit)
                rows = await self._with_retries(_fetch_ohlcv, f"accept.get_ohlcv[{sym},{tf}]")
                rows = self._sanitize_ohlcv(rows or [])
                for r in rows:
                    bar = {"ts": float(r[0]), "o": float(r[1]), "h": float(r[2]), "l": float(r[3]), "c": float(r[4]), "v": float(r[5])}
                    await self._maybe_await(self.shared_state.add_ohlcv(sym, tf, bar))

            # Price
            async def _fetch_price():
                return await ec.get_current_price(sym)
            price = await self._with_retries(_fetch_price, f"accept.get_price[{sym}]")
            price_f = self._coerce_positive_price(price)
            if price_f > 0:
                await self._inject_latest_price(sym, price_f)
            else:
                self._logger.warning("[MDF] accept price invalid for %s: %r", sym, price)

            # ATR keep-warm if configured
            if self.compute_atr and "1h" in self.timeframes:
                with contextlib.suppress(Exception):
                    await self._maybe_await(self.shared_state.calc_atr(sym, "1h", 14))

            # Per-symbol readiness + opportunistic global readiness
            if await self._symbol_meets_depth(sym):
                await self._mark_symbol_ready(sym)
            if self.readiness_emit and not self._declared_ready:
                try:
                    symbols = await self._get_accepted_symbols()
                    depth_ok = True
                    for s in symbols:
                        if not await self._symbol_meets_depth(s):
                            depth_ok = False
                            break
                    if depth_ok:
                        await self._maybe_set_ready()
                except Exception:
                    self._logger.debug("global readiness check failed in on_symbol_accepted", exc_info=True)
        except Exception as e:
            kind, meta = self._classify_error(e)
            await self._set_health(self._code_error, f"on_symbol_accepted:{kind}", metrics=meta)

    # -------------------- steady loop --------------------

    async def run(self) -> None:
        # GUARD: Prevent multiple concurrent run loop instances.
        # Multiple callers (AppContext._start_with_timeout, main_phased, main_live)
        # may invoke run() — only the first one should proceed.
        if self._run_loop_entered:
            self._logger.warning(
                "[MDF] run() called but loop already running — ignoring duplicate entry"
            )
            return
        self._run_loop_entered = True

        # ⚡ Start WebSocket feed for hybrid market data (NEW)
        await self._start_websocket()

        self._logger.info("MarketDataFeed run loop entered")
        syms = await self._get_accepted_symbols()
        self._logger.info(f"[MDF] loop started | accepted_symbols={syms}")

        while not self._stop.is_set():
            self._poll_cycle += 1
            symbols = await self._get_accepted_symbols()
            current_symbols = {str(s or "").strip().upper() for s in symbols if str(s or "").strip()}
            new_symbols = sorted(current_symbols - self._known_symbols)
            if new_symbols:
                self._logger.info("[MDF] accepted-symbol delta detected; backfill=%s", new_symbols)
                await self._schedule_symbol_backfill(new_symbols)
            self._known_symbols = current_symbols

            # 🔎 1) Log symbols at every cycle
            self._logger.warning("[DEBUG_MDF] symbols=%s", symbols)

            if not symbols:
                self._empty_symbol_cycles += 1
                if self._empty_symbol_cycles == 1 or self._empty_symbol_cycles % 10 == 0:
                    self._logger.warning(
                        "[MDF] poll cycle has no accepted symbols (streak=%d)",
                        self._empty_symbol_cycles,
                    )
                    await self._set_health(
                        self._code_warn,
                        "poll:no_symbols",
                        metrics={"streak": self._empty_symbol_cycles},
                    )
                await asyncio.sleep(self.retry_after_no_symbols_sec)
                continue
            self._empty_symbol_cycles = 0

            sem = asyncio.Semaphore(self.max_concurrency)

            tasks = [
                asyncio.create_task(self._poll_symbol(sym, sem), name=f"mdf.poll[{sym}]")
                for sym in symbols
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            poll_errors = 0
            no_client = 0
            no_price = 0
            price_updates = 0
            bars_added = 0

            for result in results:
                if isinstance(result, Exception):
                    poll_errors += 1
                    self._logger.warning("[MDF] poll task exception: %s", result)
                    continue
                if not isinstance(result, dict):
                    continue
                reason = str(result.get("reason") or "")
                if reason == "no_exchange_client":
                    no_client += 1
                elif reason == "price_unavailable":
                    no_price += 1
                if bool(result.get("price_updated")):
                    price_updates += 1
                bars_added += int(result.get("bars_added", 0) or 0)

            if no_client > 0:
                self._missing_exchange_cycles += 1
            else:
                self._missing_exchange_cycles = 0

            if poll_errors > 0:
                await self._set_health(
                    self._code_warn,
                    "poll:task_errors",
                    metrics={
                        "symbols": len(symbols),
                        "errors": poll_errors,
                        "price_updates": price_updates,
                        "bars_added": bars_added,
                    },
                )
            elif no_client == len(symbols):
                await self._set_health(
                    self._code_warn,
                    "poll:no_exchange_client",
                    metrics={"symbols": len(symbols), "missing_exchange_cycles": self._missing_exchange_cycles},
                )
                if self._missing_exchange_cycles == 1 or self._missing_exchange_cycles % 5 == 0:
                    self._logger.warning(
                        "[MDF] exchange client unavailable across all symbols (streak=%d)",
                        self._missing_exchange_cycles,
                    )
            elif price_updates == 0 and bars_added == 0:
                await self._set_health(
                    self._code_warn,
                    "poll:no_data",
                    metrics={
                        "symbols": len(symbols),
                        "no_price_symbols": no_price,
                    },
                )
                self._logger.warning(
                    "[MDF] poll cycle produced no market-data writes (symbols=%d, no_price=%d)",
                    len(symbols),
                    no_price,
                )
            else:
                await self._set_health(
                    self._code_ok,
                    "poll",
                    metrics={
                        "symbols": len(symbols),
                        "price_updates": price_updates,
                        "bars_added": bars_added,
                    },
                )

            await asyncio.sleep(self.poll_interval)

    async def _poll_symbol(self, sym: str, sem: asyncio.Semaphore) -> Dict[str, Any]:
        async with sem:
            ec = await self._get_exchange_client()
            if ec is None:
                # Late wiring race: skip this cycle for this symbol
                return {
                    "symbol": sym,
                    "price_updated": False,
                    "bars_added": 0,
                    "reason": "no_exchange_client",
                }
            # Last price
            price = None
            price_updated = False
            bars_added = 0
            try:
                # ⚡ Try WebSocket price first (hybrid mode) - NEW
                if self.websocket_feed and self.enable_websocket:
                    try:
                        # Check if WebSocket has provided this price
                        norm_fn = getattr(self.shared_state, "_norm_sym", None)
                        norm_sym = norm_fn(sym) if callable(norm_fn) else str(sym or "").upper()
                        cache = getattr(self.shared_state, "_price_cache", None)
                        if isinstance(cache, dict) and norm_sym in cache:
                            price_tuple = cache.get(norm_sym)
                            if isinstance(price_tuple, (tuple, list)) and len(price_tuple) >= 1:
                                price = price_tuple[0]
                                self._logger.debug("[MDF] using WebSocket price for %s: %.10f", sym, price)
                    except Exception:
                        pass
                
                # Fall back to REST if WebSocket price not available
                if price is None:
                    async def _fetch_price():
                        return await ec.get_current_price(sym)
                    price = await self._with_retries(_fetch_price, f"poll.get_price[{sym}]")
            except Exception:
                price = None
            price_f = self._coerce_positive_price(price)
            if price_f > 0:
                try:
                    price_updated = bool(await self._inject_latest_price(sym, price_f))
                    # 🔎 3) Log price write
                    self._logger.warning("[DEBUG_MDF] price update %s = %s", sym, price_f)
                except Exception:
                    self._logger.debug("update_last_price failed for %s", sym, exc_info=True)

            # Lightweight tail refresh on each timeframe
            # 🔎 2) Log before fetching OHLCV
            self._logger.warning("[DEBUG_MDF] fetching OHLCV for %s", sym)
            for tf in self.timeframes:
                try:
                    async def _fetch_tail():
                        return await ec.get_ohlcv(sym, tf, limit=3)
                    rows = await self._with_retries(_fetch_tail, f"poll.get_ohlcv[{sym},{tf}]")
                    rows = self._sanitize_ohlcv(rows or [])
                    for r in rows:
                        bar = {
                            "ts": float(r[0]),
                            "o": float(r[1]),
                            "h": float(r[2]),
                            "l": float(r[3]),
                            "c": float(r[4]),
                            "v": float(r[5]),
                        }
                        await self._maybe_await(self.shared_state.add_ohlcv(sym, tf, bar))
                        bars_added += 1
                except Exception:
                    self._logger.debug("poll.get_ohlcv failed for %s %s", sym, tf, exc_info=True)

            # Optional ATR keep-warm
            if self.compute_atr and "1h" in self.timeframes:
                try:
                    _ = await self._maybe_await(self.shared_state.calc_atr(sym, "1h", 14))
                except Exception:
                    pass

            # TEMPORARY DEBUG — remove after confirming data flow is healthy
            self._logger.debug(
                "[MDF:POLL] %s price=%s tfs=%s",
                sym, price, self.timeframes,
            )
            return {
                "symbol": sym,
                "price_updated": price_updated,
                "bars_added": bars_added,
                "reason": "" if (price_updated or bars_added > 0) else "price_unavailable",
            }

    # -------------------- control --------------------

    async def stop(self):
        self._stop.set()
        await self._cancel_backfill_tasks()
        
        # ⚡ Clean up WebSocket feed (NEW)
        if self.websocket_feed:
            try:
                await self.websocket_feed.stop()
            except Exception as e:
                self._logger.debug("[MDF] WebSocket cleanup error: %s", str(e))
        
        if self._websocket_task and not self._websocket_task.done():
            self._websocket_task.cancel()
            try:
                await self._websocket_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                self._logger.debug("[MDF] WebSocket task cancel error: %s", str(e))
