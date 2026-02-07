# core/market_data_feed.py
from __future__ import annotations

import asyncio
import contextlib
import random
import time
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
        ohlcv_limit: int = 300,
        poll_interval: float = 3.0,
        max_concurrency: int = 8,
        logger: Optional[logging.Logger] = None,
        compute_atr: bool = True,
        health_cadence_sec: float = 10.0,
        warmup_timeout_sec: float = 45.0,
        retry_after_no_symbols_sec: float = 2.0,
        max_retry_backoff_sec: float = 20.0,
        min_bars_required: int = 300,
        readiness_emit: bool = True,
        per_symbol_readiness: bool = True,
        **_,
    ) -> None:
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
        raw_limit = _cfg("ohlcv_limit", ohlcv_limit)
        try:
            limit = int(raw_limit)
        except Exception:
            limit = 250
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

        # Internal event to wait for late wiring of the exchange client (AppContext P3.65+)
        self._exch_ready_evt: asyncio.Event = asyncio.Event()
        if self.exchange_client is not None:
            self._exch_ready_evt.set()

        self._stop = asyncio.Event()
        self._logger = logger or logging.getLogger("MarketDataFeed")
        if self._logger.level == logging.NOTSET:
            self._logger.setLevel(logging.INFO)

        self._health_task: Optional[asyncio.Task] = None
        self._last_error_ts: float = 0.0

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

    async def _set_health(self, code, msg: str, metrics: Optional[Dict[str, Any]] = None):
        """
        Send health via SharedState.set_component_health if available.
        Works with enums or strings; never crashes the caller.
        """
        try:
            fn = getattr(self.shared_state, "set_component_health", None)
            if not fn:
                return
            await self._maybe_await(fn(self._component_key, code, msg, metrics=metrics or {}))
        except Exception:
            self._logger.debug("set_component_health failed", exc_info=True)

    async def _get_accepted_symbols(self) -> List[str]:
        try:
            syms = await self._maybe_await(self.shared_state.get_accepted_symbols())
            return list(syms or [])
        except Exception:
            self._logger.debug("get_accepted_symbols failed", exc_info=True)
            return []

    async def _maybe_set_ready(self):
        """Best-effort signal to SharedState that market data is ready."""
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
                if price is not None:
                    await self._maybe_await(self.shared_state.update_last_price(sym, float(price)))

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
            if price is not None:
                await self._maybe_await(self.shared_state.update_last_price(sym, float(price)))

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
        self._logger.info("📡 MarketDataFeed run loop entered")
        syms = await self._get_accepted_symbols()
        self._logger.warning(f"[MDF] loop entered | accepted_symbols={syms}")

        while True:
            symbols = await self._get_accepted_symbols()

            self._logger.debug(f"[MDF] symbols={symbols}")

            if not symbols:
                await asyncio.sleep(1)
                continue

            ec = await self._get_exchange_client()
            if ec is None:
                await asyncio.sleep(1)
                continue

            for symbol in symbols:
                try:
                    price = await ec.get_current_price(symbol)
                    await self._maybe_await(self.shared_state.update_last_price(symbol, float(price)))

                    for tf in self.timeframes:
                        ohlcv = await ec.get_ohlcv(symbol, tf, limit=3)
                        ohlcv = self._sanitize_ohlcv(ohlcv or [])
                        for bar_data in ohlcv:
                            bar = {
                                "ts": float(bar_data[0]),
                                "o": float(bar_data[1]),
                                "h": float(bar_data[2]),
                                "l": float(bar_data[3]),
                                "c": float(bar_data[4]),
                                "v": float(bar_data[5]),
                            }
                            await self._maybe_await(self.shared_state.add_ohlcv(symbol, tf, bar))
                except Exception as e:
                    self._logger.debug(f"Failed to poll {symbol}: {e}")

            await self._set_health(self._code_ok, "poll", metrics={"symbols": len(symbols)})

            await asyncio.sleep(self.poll_interval)

    async def _poll_symbol(self, sym: str, sem: asyncio.Semaphore) -> None:
        async with sem:
            ec = await self._get_exchange_client()
            if ec is None:
                # Late wiring race: skip this cycle for this symbol
                return
            # Last price
            price = None
            try:
                async def _fetch_price():
                    return await ec.get_current_price(sym)
                price = await self._with_retries(_fetch_price, f"poll.get_price[{sym}]")
            except Exception:
                price = None
            if price is not None:
                await self._maybe_await(self.shared_state.update_last_price(sym, float(price)))

            # Lightweight tail refresh on each timeframe
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
                except Exception:
                    pass

            # Optional ATR keep-warm
            if self.compute_atr and "1h" in self.timeframes:
                try:
                    _ = await self._maybe_await(self.shared_state.calc_atr(sym, "1h", 14))
                except Exception:
                    pass

    # -------------------- control --------------------

    async def stop(self):
        self._stop.set()
