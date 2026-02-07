if True:
    """
        Component: ExchangeClient
        Contract: core:ExchangeClient:v2.1.0
        Phase: P3 (bootstrap) → P9 (runtime)
        Author: Octivault

        Design Alignment (P9 invariants)
        - Single order path → _request → _send_* → /api (MARKET/quantity or quoteOrderQty)
        - Order tagging → newClientOrderId carries `octi-<ts>-<tag>`
        - Hygiene guards → NOTIONAL/MIN_NOTIONAL, MARKET_LOT_SIZE/LOT_SIZE.stepSize, per-symbol throttle, weight guard
            - Paper/live parity → mirrored fills, balances, events
            - Health/events → _report_status() → "events.health.status", plus "events.summary" lines
            - Caches → exchangeInfo, normalized filters, price micro-cache, account cache, IPO cache
    """

    # =============================
    # Imports
    # =============================
    import asyncio
    import hashlib
    import hmac
    import json
    import os
    import time
    import urllib.parse
    from collections import defaultdict, deque
    from datetime import datetime, timezone
    from decimal import Decimal, ROUND_DOWN
    from typing import Any, Dict, Optional, List, Tuple
    import logging

    import aiohttp
    from binance.async_client import AsyncClient
    import random
    try:
        from binance.exceptions import BinanceAPIException as _BinanceAPIException  # type: ignore

        class BinanceAPIException(_BinanceAPIException):
            def __init__(self, message: str, code: Optional[int] = None):
                self.message = message
                self.code = code
                try:
                    super().__init__(message, code=code)
                except TypeError:
                    try:
                        super().__init__(message)
                    except Exception:
                        Exception.__init__(self, message)
    except Exception:
        try:
            from core.stubs import BinanceAPIException  # type: ignore
        except Exception:
            class BinanceAPIException(Exception):
                def __init__(self, message: str, code: Optional[int] = None):
                    super().__init__(message)
                    self.message = message
                    self.code = code

    # -----------------------------------------
    # Import-compat shim for legacy module path
    # -----------------------------------------
    import sys as _sys
    from pathlib import Path as _Path
    try:
        _P = _Path(__file__).resolve().parent
        if str(_P) not in _sys.path:
            _sys.path.insert(0, str(_P))
    except Exception:
        pass
    _sys.modules.setdefault("core.exchange_client", _sys.modules[__name__])

    # ---------------------------
    # Module-level lazy singleton
    # ---------------------------

    _GLOBAL_EXCHANGE_CLIENT: Optional["ExchangeClient"] = None

    def get_global_exchange_client(
        *,
        config: Optional[dict] = None,
        logger: Optional[logging.Logger] = None,
        app: Optional[Any] = None,
        shared_state: Optional[Any] = None,
    ) -> "ExchangeClient":
        """
        Return a module-level ExchangeClient instance, creating it if necessary.
        Useful for early-phase consumers (e.g., AppContext P4) when the DI container
        hasn't built the client yet.
        """
        global _GLOBAL_EXCHANGE_CLIENT
        if _GLOBAL_EXCHANGE_CLIENT is None:
            _GLOBAL_EXCHANGE_CLIENT = ExchangeClient(
                config=config or {},
                logger=logger or logging.getLogger("octi.ExchangeClient"),
                app=app,
                shared_state=shared_state,
            )
        return _GLOBAL_EXCHANGE_CLIENT

    async def ensure_public_bootstrap(
        *,
        config: Optional[dict] = None,
        logger: Optional[logging.Logger] = None,
        app: Optional[Any] = None,
        shared_state: Optional[Any] = None,
    ) -> "ExchangeClient":
        """
        Ensure a public-only ExchangeClient is available and public endpoints can be called.
        Does NOT require API keys and is safe during early phases.
        """
        client = get_global_exchange_client(
            config=config, logger=logger, app=app, shared_state=shared_state
        )
        await client._ensure_started_public()
        return client

    # Convenience helpers for early-phase symbol/exchange info access
    async def public_get_exchange_info(
        *,
        config: Optional[dict] = None,
        logger: Optional[logging.Logger] = None,
        app: Optional[Any] = None,
        shared_state: Optional[Any] = None,
    ) -> Dict[str, Any]:
        client = await ensure_public_bootstrap(
            config=config, logger=logger, app=app, shared_state=shared_state
        )
        return await client.get_exchange_info()

    async def public_get_symbol_info(
        symbol: str,
        *,
        config: Optional[dict] = None,
        logger: Optional[logging.Logger] = None,
        app: Optional[Any] = None,
        shared_state: Optional[Any] = None,
    ) -> Optional[Dict[str, Any]]:
        client = await ensure_public_bootstrap(
            config=config, logger=logger, app=app, shared_state=shared_state
        )
        return await client.get_symbol_info(symbol)


    # Explicit export list for importers
    __all__ = [
        "ExchangeClient",
        "get_global_exchange_client",
        "ensure_public_bootstrap",
        "public_get_exchange_info",
        "public_get_symbol_info",
    ]


    # Project exceptions (keep your original locations)
    try:
        from core.stubs import ExecutionError
    except Exception:
        class ExecutionError(Exception):
            pass

    class NetworkException(Exception):
        pass

    class ExchangeClient:
        async def symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
            """Compatibility alias used by some agents (e.g., WalletScanner)."""
            return await self.get_symbol_info(symbol)
        async def _resync_time(self) -> None:
            """
            Best-effort server time resync used when Binance returns -1021.
            """
            try:
                t = await self._request("GET", "/api/v3/time", api="spot_api")
                server_ms = int(t.get("serverTime", 0))
                local_ms = int(time.time() * 1000)
                self._time_offset_ms = server_ms - local_ms
            except Exception:
                # Non-fatal; keep previous offset
                return
        # ---- tiny in-memory hot caches (low TTL) ----
        _BOOK_TTL = 0.75  # seconds
        _INFO_TTL = 10.0  # seconds (symbol info/tradable probe microcache)
        async def get_exchange_info(self) -> Dict[str, Any]:
            """Public getter used by SymbolManager; ensures cache is populated."""
            await self._sync_exchange_info()
            return self._exchange_info or {}

        async def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
            """Lookup a single symbol from cached exchangeInfo; refresh if needed, with microcache."""
            sym = self._norm_symbol(symbol)
            # microcache
            ent = getattr(self, "_sym_info_cache", {}).get(sym)
            if ent and (time.time() - ent["ts"] < self._INFO_TTL):
                return ent["val"]
            await self._sync_exchange_info()
            info = (self._exchange_info or {}).get("symbols", [])
            for s in info:
                if str(s.get("symbol", "")).upper() == sym:
                    self._sym_info_cache = getattr(self, "_sym_info_cache", {})
                    self._sym_info_cache[sym] = {"ts": time.time(), "val": s}
                    return s
            # Optional: try a direct query if not found in cache
            try:
                data = await self._request("GET", "/api/v3/exchangeInfo", {"symbol": sym}, api="spot_api")
                syms = (data or {}).get("symbols", [])
                val = syms[0] if syms else None
                self._sym_info_cache = getattr(self, "_sym_info_cache", {})
                self._sym_info_cache[sym] = {"ts": time.time(), "val": val}
                return val
            except Exception:
                return None

        # --------- discovery/wallet shims expected by agents ---------

        async def get_account_balances(self) -> dict:
            """
            WalletScanner expects this. Wraps get_spot_balances() and returns:
            { 'ASSET': {'free': float, 'locked': float}, ... }
            """
            try:
                return await self.get_spot_balances()
            except Exception:
                self.logger.warning("[ExchangeClient] get_account_balances fallback → empty dict", exc_info=True)
                return {}

        async def get_all_tickers(self) -> List[Dict[str, Any]]:
            """
            SymbolScreener expects this. Compatible shape with keys:
            symbol, priceChangePercent, quoteVolume, volume.
            Internally reuses get_24hr_tickers().
            """
            return await self.get_24hr_tickers()

        async def get_24hr_stats(self) -> Dict[str, Dict[str, Any]]:
            """
            SymbolScreener.run_discovery() expects a plural 'stats' map.
            Return { SYMBOL: { 'volume': float, 'priceChangePercent': float, ... }, ... }.
            """
            tickers = await self.get_24hr_tickers()
            out: Dict[str, Dict[str, Any]] = {}
            for t in tickers:
                try:
                    out[t["symbol"]] = {
                        "symbol": t["symbol"],
                        "lastPrice": float(t.get("lastPrice", 0.0)),
                        "priceChangePercent": float(t.get("priceChangePercent", 0.0)),
                        "quoteVolume": float(t.get("quoteVolume", 0.0)),
                        "volume": float(t.get("volume", 0.0)),
                    }
                except Exception:
                    continue
            return out

        # --------- helpers used by SymbolManager filters ---------

        def symbol_exists_cached(self, symbol: str) -> bool:
            """
            Cheap existence/trading check using cached exchangeInfo.
            """
            sym = self._norm_symbol(symbol)
            if not self._exchange_info:
                return False
            try:
                for s in self._exchange_info.get("symbols", []):
                    if str(s.get("symbol", "")).upper() == sym:
                        return s.get("status") in ("TRADING", "BREAK")
            except Exception:
                pass
            return False

        async def symbol_exists(self, symbol: str) -> bool:
            """
            Fallback existence/trading check with an API hit.
            """
            sym = self._norm_symbol(symbol)
            try:
                await self._sync_exchange_info()
                if self.symbol_exists_cached(sym):
                    return True
                # direct query for robustness
                data = await self._request("GET", "/api/v3/exchangeInfo", {"symbol": sym}, api="spot_api")
                syms = (data or {}).get("symbols", [])
                if not syms:
                    return False
                status = syms[0].get("status")
                return status in ("TRADING", "BREAK")
            except Exception:
                return False

        async def is_stable_asset(self, asset: str) -> bool:
            """
            Optional helper used by SymbolManager to avoid stable-coin bases.
            Heuristic set; extend if you need more.
            """
            return str(asset).upper() in {"USDT", "USDC", "FDUSD", "BUSD", "TUSD", "DAI"}

        # --------- IPO/new listings (used by IPOChaser & SymbolManager.recent) ---------

        async def get_new_listings(self) -> List[str]:
            """
            Lightweight heuristic: we don't have a perfect 'listing date' on public testnet.
            Return an empty list on testnet; on mainnet you could implement scraping/alt feed.
            """
            if self.testnet:
                self.logger.info("[ExchangeClient] get_new_listings: testnet → returning [].")
                return []
            # Minimal mainnet heuristic (safe default). Replace with a real feed if available.
            try:
                await self._sync_exchange_info()
                # As a placeholder, return symbols that are TRADING and quoted in USDT (you can refine).
                return [
                    s["symbol"] for s in self._exchange_info.get("symbols", [])
                    if s.get("status") in ("TRADING", "BREAK") and s.get("quoteAsset") == "USDT"
                ][:20]
            except Exception:
                return []

        async def get_new_listings_cached(self) -> List[str]:
            """
            Cached wrapper for get_new_listings(); keep same signature agents call.
            """
            # Very small TTL cache; you can promote to a real cache if needed.
            if not hasattr(self, "_ipo_cache") or (time.time() - getattr(self, "_ipo_cache_ts", 0) > 300):
                listings = await self.get_new_listings()
                self._ipo_cache = listings
                self._ipo_cache_ts = time.time()
            return list(self._ipo_cache or [])

        # --------- compatibility aliases for agents ---------
        async def get_ticker_price(self, symbol: str) -> float:
            """Alias for WalletScanner: returns current price."""
            return await self.get_current_price(symbol)

        async def get_price(self, symbol: str) -> float:
            """Alias for agents that expect get_price()."""
            return await self.get_current_price(symbol)

        def has_symbol(self, symbol: str) -> bool:
            """Cheap membership check using cached exchangeInfo."""
            sym = self._norm_symbol(symbol)
            if not self._exchange_info:
                return False
            try:
                return any(str(s.get("symbol", "")).upper() == sym for s in self._exchange_info.get("symbols", []))
            except Exception:
                return False

        async def is_tradable(self, symbol: str) -> bool:
            """Best-effort tradability check using cached exchangeInfo (refresh on demand, microcached)."""
            sym = self._norm_symbol(symbol)
            # microcache
            tcache = getattr(self, "_tradable_cache", {}).get(sym)
            if tcache and (time.time() - tcache["ts"] < self._INFO_TTL):
                return bool(tcache["val"])
            await self._sync_exchange_info()
            ok = False
            try:
                for s in self._exchange_info.get("symbols", []):
                    if str(s.get("symbol", "")).upper() == sym:
                        ok = s.get("status") in ("TRADING", "BREAK")
                        break
            except Exception:
                ok = False
            self._tradable_cache = getattr(self, "_tradable_cache", {})
            self._tradable_cache[sym] = {"ts": time.time(), "val": ok}
            return ok

        def get_known_quotes(self) -> List[str]:
            """Expose known quote assets for WalletScanner."""
            return list(self._known_quotes)

        # --- additional compatibility shims for older callers ---
        async def get_balances(self) -> dict:
            """
            Legacy alias used by older ops scripts/tests.
            Mirrors spot balances: { 'ASSET': {'free': float, 'locked': float}, ... }.
            """
            return await self.get_spot_balances()

        async def get_symbol_price(self, symbol: str) -> float:
            """
            Legacy alias → current price for a symbol.
            """
            return await self.get_current_price(symbol)

        async def get_order_book_ticker(self, symbol: str) -> Dict[str, str]:
            """
            Legacy alias that returns the /api/v3/ticker/bookTicker shape:
            {'bidPrice': '...', 'askPrice': '...'} using the internal best bid/ask.
            """
            bid, ask = await self.get_best_bid_ask(symbol)
            out: Dict[str, str] = {}
            if bid is not None:
                out["bidPrice"] = f"{bid:.16g}"
            if ask is not None:
                out["askPrice"] = f"{ask:.16g}"
            return out

        """Client for Binance spot (testnet/live) with P9-compliant order path."""

        async def get_order_status(self, symbol: str, client_order_id: str) -> Optional[dict]:
            sym = self._norm_symbol(symbol)
            try:
                raw = await self._request(
                    "GET", "/api/v3/order",
                    {"symbol": sym, "origClientOrderId": client_order_id},
                    signed=True, api="spot_api"
                )
                return raw
            except Exception:
                return None

        def __init__(
            self,
            config: Optional[dict] = None,
            logger: Optional[logging.Logger] = None,
            app: Optional[Any] = None,
            shared_state: Optional[Any] = None,
            exchange_client: Optional[Any] = None,  # ignored; keeps compatibility with _try_construct kwargs
            api_key: Optional[str] = None,
            api_secret: Optional[str] = None,
            testnet: Optional[bool] = None,
            paper_trade: Optional[bool] = None,
        ):
            # basic refs
            self.config = config or {}
            self.shared_state = shared_state
            self.app = app

            # logger
            self.logger = logger if hasattr(logger, "level") else logging.getLogger("octi.ExchangeClient")
            if self.logger.level == logging.NOTSET:
                self.logger.setLevel(logging.INFO)

            # resolve API keys from args -> config -> env
            def _cfg(key, default=None):
                try:
                    if hasattr(self.config, key):
                        return getattr(self.config, key)
                except Exception:
                    pass
                if isinstance(self.config, dict) and key in self.config:
                    return self.config.get(key, default)
                return os.getenv(key, default)

            def _cfg_bool(key, default=False):
                v = _cfg(key, default)
                if isinstance(v, bool):
                    return v
                if isinstance(v, str):
                    s = v.strip().lower()
                    if s in ("1", "true", "yes", "y", "on"):
                        return True
                    if s in ("0", "false", "no", "n", "off"):
                        return False
                return bool(v)

            self.api_key = api_key or _cfg("API_KEY") or _cfg("BINANCE_API_KEY")
            self.api_secret = api_secret or _cfg("API_SECRET") or _cfg("BINANCE_API_SECRET")

            # runtime modes (robust env parsing so "False" doesn't evaluate truthy)
            if testnet is None:
                testnet = _cfg_bool("TESTNET_MODE", False)
            if paper_trade is None:
                paper_trade = _cfg_bool("PAPER_MODE", False)

            self.testnet = bool(testnet)
            self.paper_trade = bool(paper_trade)
            if self.paper_trade:
                self.logger.info("Paper trading mode is enabled. No real orders will be placed.")

            # FIX 1: Validate API keys early before AsyncClient initialization
            if self.paper_trade:
                self.logger.info("[EC] Paper trading mode - using public endpoints only")
                self.api_key = "paper_key"
                self.api_secret = "paper_secret"
            else:
                if not self.api_key or not self.api_secret:
                    raise RuntimeError(
                        "[EC] CRITICAL: Binance API keys not found!\n"
                        "Required: BINANCE_API_KEY and BINANCE_API_SECRET in .env or environment\n"
                        f"Found: api_key={'present' if self.api_key else 'MISSING'}, "
                        f"api_secret={'present' if self.api_secret else 'MISSING'}\n"
                        "Check your .env file or environment variables."
                    )
                self.logger.info(f"[EC] API keys validated (key_len={len(self.api_key)}, secret_len={len(self.api_secret)})")

            # aiohttp/binance client handles
            self.client: Optional[AsyncClient] = None
            self.session: Optional[aiohttp.ClientSession] = None
            self._ready = False

            # API base URLs (spot /api + /sapi, UM futures kept available)
            # Allow override from config or environment, else derive by testnet flag.
            base_override = (
                (getattr(self.config, "BINANCE_BASE_URL", "") if not isinstance(self.config, dict) else self.config.get("BINANCE_BASE_URL", ""))
                or os.getenv("BINANCE_BASE_URL", "").strip()
            )

            if self.testnet:
                spot_api = "https://testnet.binance.vision"
                spot_sapi = "https://testnet.binance.vision"  # limited on testnet
                um_fut = "https://testnet.binancefuture.com"
            else:
                spot_api = "https://api.binance.com"
                spot_sapi = "https://api.binance.com"
                um_fut = "https://fapi.binance.com"

            if base_override:
                # If user explicitly provided a base, apply it to spot endpoints.
                spot_api = base_override
                spot_sapi = base_override

            self.base_url_spot_api = spot_api
            self.base_url_spot_sapi = spot_sapi
            self.base_url_um = um_fut

            # Define known quote assets for agent compatibility
            self._known_quotes = {"USDT", "FDUSD", "USDC", "BUSD", "TUSD", "DAI"}

            # caches & tuning
            self.price_cache: Dict[str, Tuple[float, float]] = {}  # (price, ts)
            self.symbol_filters: Dict[str, Dict[str, Any]] = {}
            self._exchange_info: Optional[Dict[str, Any]] = None
            self._exchange_info_timestamp: float = 0
            self._sync_lock = asyncio.Lock()
            self._paper_trade_orders: Dict[str, Dict[str, Any]] = {}
            self._time_offset_ms: int = 0  # server - local
            self._weight_counters = defaultdict(lambda: {"ts": 0.0, "w": 0})
            self._weight_window = float(_cfg("WEIGHT_WINDOW_SEC", 1.0))  # seconds
            self._weight_limit = int(_cfg("WEIGHT_LIMIT_PER_WINDOW", 10))    # conservative per second
            self._path_weight_overrides = dict(_cfg("PATH_WEIGHTS", {"/api/v3/klines": 2}) or {})
            self._acct_cache = None
            self._acct_cache_ts = 0.0
            self._acct_ttl = float(_cfg("ACCT_CACHE_TTL_SEC", 5.0))  # seconds
            self.fee_buffer_bps = int(_cfg("FEE_BUFFER_BPS", 10))
            self._px_ttl = float(_cfg("PRICE_MICROCACHE_TTL", 1.0))

            # Binance recvWindow parameter (milliseconds) used for signed requests
            self.recv_window_ms = int(_cfg("RECV_WINDOW_MS", 5000))
            # 24h ticker cache (avoid AttributeError when accessed before first population)
            self._ticker_24h_cache: Dict[str, Dict[str, Any]] = {}
            self._ticker_24h_ttl_sec = float(_cfg("TICKER_24H_TTL_SEC", 15.0))

            # Circuit breakers
            class _Breaker:
                def __init__(self, window_sec=60, open_error_rate_pct=10, min_requests=10, half_open_after_sec=15):
                    self.window_sec = window_sec
                    self.open_error_rate_pct = open_error_rate_pct
                    self.min_requests = min_requests
                    self.half_open_after_sec = half_open_after_sec
                    self.events = deque(maxlen=1000)  # (ts, ok_bool) - limit memory usage
                    self.state = "CLOSED"
                    self.open_since = 0.0

                def _prune(self, now):
                    while self.events and now - self.events[0][0] > self.window_sec:
                        self.events.popleft()

                def record(self, ok: bool):
                    now = time.time()
                    self.events.append((now, ok))
                    self._prune(now)
                    if self.state == "CLOSED":
                        if len(self.events) >= self.min_requests:
                            err = 1 - (sum(1 for _, ok in self.events if ok) / len(self.events))
                            if err * 100 >= self.open_error_rate_pct:
                                self.state = "OPEN"
                                self.open_since = now
                    elif self.state == "HALF_OPEN":
                        self.state = "CLOSED" if ok else "OPEN"
                        if not ok:
                            self.open_since = now

                def allow(self) -> bool:
                    now = time.time()
                    if self.state == "OPEN":
                        if now - self.open_since >= self.half_open_after_sec:
                            self.state = "HALF_OPEN"
                            return True  # one probe
                        return False
                    return True

            self._submit_breaker = _Breaker()
            self._query_breaker = _Breaker()

        # ------------- tiny utils -------------
        async def _maybe_await(self, v):
            if asyncio.iscoroutine(v):
                return await v
            return v

        def _now_iso(self):
            return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        def _norm_symbol(self, symbol: str) -> str:
            return (symbol or "").upper().replace("/", "")

        def _round_step(self, value: Decimal, step: str) -> Decimal:
            q = Decimal(step)
            return (value // q) * q

        # ------------- observability -------------
        async def _emit_summary(self, event: str, **kvs):
            # One-line roll-up per spec §3.17
            try:
                line = {"component": "ExchangeClient", "event": event, **kvs}
                self.logger.info("SUMMARY %s", line)
                if hasattr(self.shared_state, "emit_event"):
                    await self._maybe_await(self.shared_state.emit_event("events.summary", line))
            except Exception:
                pass

        async def _report_status(self, level: str, details: Optional[dict] = None):
            """Emit P9 HealthStatus: topic=events.health.status, schema={component,level,details,ts}"""
            payload = {
                "component": "ExchangeClient",
                "level": level,                     # "OK" | "DEGRADED" | "ERROR"
                "details": details or {},
                "ts": self._now_iso(),
            }
            try:
                if hasattr(self.shared_state, "emit_event"):
                    await self._maybe_await(self.shared_state.emit_event("events.health.status", payload))
            except Exception:
                self.logger.debug("emit_event failed for events.health.status", exc_info=True)

        # public passthrough to SharedState event bus (compat)
        async def emit_event(self, event_type: str, payload: Dict[str, Any]):
            try:
                if hasattr(self.shared_state, "emit_event"):
                    v = self.shared_state.emit_event(event_type, payload)
                    if asyncio.iscoroutine(v):
                        await v
            except Exception:
                self.logger.debug("emit_event passthrough failed", exc_info=True)

        # ------------- public helpers -------------
        async def get_all_symbols(self) -> List[str]:
            """
            Return tradable spot symbols (uppercased) using cached exchangeInfo when possible.
            AppContext uses this to prune accepted symbols.
            """
            await self._sync_exchange_info()
            info = self._exchange_info or {}
            out: List[str] = []
            try:
                for s in info.get("symbols", []):
                    status = s.get("status")
                    if status in ("TRADING", "BREAK"):
                        out.append(str(s.get("symbol", "")).upper())
            except Exception:
                # Fallback: refresh from API
                try:
                    data = await self._request("GET", "/api/v3/exchangeInfo", api="spot_api")
                    for s in data.get("symbols", []):
                        if s.get("status") in ("TRADING", "BREAK"):
                            out.append(str(s.get("symbol", "")).upper())
                except Exception:
                    return []
            return out

        async def get_exchange_filters(self, symbol: str) -> dict:
            """Returns normalized filters for a symbol, refreshing cache if needed."""
            await self._sync_exchange_info()
            f = self.symbol_filters.get(self._norm_symbol(symbol), {})
            notional = f.get("NOTIONAL") or f.get("MIN_NOTIONAL") or {}
            lot = self._pick_lot_filter(f)
            price_filter = f.get("PRICE_FILTER", {})
            return {
                "min_notional": float(notional.get("minNotional", 0.0)),
                "step_size":     str(lot.get("stepSize", "0.000001")),
                "min_qty":       float(lot.get("minQty", 0)),
                "tick_size":     str(price_filter.get("tickSize", "0.000001")),
                "quote_precision": f.get("_precisions", {}).get("quotePrecision"),
            }

        async def ensure_symbol_filters_ready(self, symbol: Optional[str] = None) -> dict:
            """
            Ensure raw Binance filters are cached. If `symbol` is provided, return its filters;
            otherwise just refresh the cache and return an empty dict.
            """
            await self._sync_exchange_info()
            if symbol:
                return self.symbol_filters.get(self._norm_symbol(symbol), {})
            return {}
        async def get_symbol_filters_raw(self, symbol: str) -> Dict[str, Any]:
            """Raw Binance filters map for a symbol (e.g., NOTIONAL/MIN_NOTIONAL, MARKET_LOT_SIZE/LOT_SIZE, PRICE_FILTER)."""
            await self._sync_exchange_info()
            return self.symbol_filters.get(self._norm_symbol(symbol), {})

        async def get_symbol_filters(self, symbol: str) -> Dict[str, Any]:
            """Normalized filters (min_notional, step_size, tick_size, quote_precision)."""
            return await self.get_exchange_filters(symbol)

        async def get_min_notional(self, symbol: str) -> float:
            """Return the exchange minNotional (NOTIONAL/MIN_NOTIONAL) for a symbol."""
            await self._sync_exchange_info()
            f = self.symbol_filters.get(self._norm_symbol(symbol), {})
            try:
                notional = f.get("NOTIONAL") or f.get("MIN_NOTIONAL") or {}
                return float(notional.get("minNotional", 0.0))
            except Exception:
                return 0.0

        @property
        def is_started(self) -> bool:
            return (self.session is not None) and (self.client is not None)

        async def start(self):
            """Canonical lifecycle entrypoint (AppContext expects start/close)."""
            if self.is_started:
                return
            from aiohttp import ClientTimeout, TCPConnector
            connector = TCPConnector(limit=50, ttl_dns_cache=300)
            self.session = aiohttp.ClientSession(timeout=ClientTimeout(total=15), connector=connector)
            # Allow starting in public-only mode if keys are absent; signed endpoints will fail gracefully.
            if not self.api_key or not self.api_secret:
                self.logger.warning("ExchangeClient starting without API keys — public endpoints only.")
                self.client = await AsyncClient.create(api_key="", api_secret="", testnet=self.testnet)
            else:
                self.client = await AsyncClient.create(self.api_key, self.api_secret, testnet=self.testnet)
            self.logger.info("Exchange client connected.")
            self._ready = True
            
            # FIX 5: Verify authentication and time sync before proceeding
            try:
                # Time sync to avoid -1021 errors
                t = await self._request("GET", "/api/v3/time", api="spot_api")
                server_ms = int(t.get("serverTime", 0))
                local_ms = int(time.time() * 1000)
                self._time_offset_ms = server_ms - local_ms
                self.logger.info(f"[EC:StartupSync] Server time synced (offset: {self._time_offset_ms}ms)")
                
                # Verify authentication (non-paper mode only)
                if not self.paper_trade and (self.api_key and self.api_key != "paper_key"):
                    try:
                        acct = await self.client.get_account()
                        can_trade = acct.get("canTrade", False)
                        self.logger.info(f"[EC:Authentication] Verified (trading_enabled={can_trade})")
                    except Exception as auth_err:
                        if "401" in str(auth_err) or "Unauthorized" in str(auth_err):
                            raise RuntimeError(f"[EC] Authentication failed - invalid API keys: {auth_err}")
                        # IP restrictions or other non-auth errors are non-fatal
                        self.logger.warning(f"[EC] Cannot verify authentication (might be IP restriction): {auth_err}")
                
                # Health check
                await self._request("GET", "/api/v3/ping", api="spot_api")
                await self._report_status("OK", {"event": "start_connected"})
            except Exception as e:
                self._ready = False
                await self._report_status("ERROR", {"event": "start_failed", "reason": str(e)})
                raise

        async def close(self):
            """Canonical lifecycle exit (AppContext calls this on shutdown)."""
            try:
                if self.client:
                    await self.client.close_connection()
            finally:
                if self.session and not self.session.closed:
                    await self.session.close()
            self.client = None
            self.session = None
            self._ready = False
            self.logger.info("Exchange client disconnected.")

        @property
        def is_ready(self) -> bool:
            return self._ready and self.session is not None and not self.session.closed and self.client is not None

        async def healthcheck(self) -> Dict[str, Any]:
            """
            Lightweight health check that pings the API and validates time sync.
            Returns a dict: {"ok": bool, "latency_ms": float, "time_offset_ms": int}
            Emits an events.health.status payload as well.
            """
            started = time.time()
            try:
                await self._request("GET", "/api/v3/ping", api="spot_api")
                t = await self._request("GET", "/api/v3/time", api="spot_api")
                server_ms = int(t.get("serverTime", 0))
                local_ms = int(time.time() * 1000)
                offset = server_ms - local_ms
                latency_ms = (time.time() - started) * 1000.0
                result = {"ok": True, "latency_ms": latency_ms, "time_offset_ms": offset}
                await self._report_status("OK", {"event": "healthcheck_ok", **result})
                return result
            except Exception as e:
                latency_ms = (time.time() - started) * 1000.0
                result = {"ok": False, "latency_ms": latency_ms, "error": str(e)}
                await self._report_status("ERROR", {"event": "healthcheck_error", **result})
                return result

        async def _ensure_started_public(self):
            """
            Lazily bootstrap a public-only session so unsigned GETs work during early phases
            (e.g., AppContext P4 calls that fetch exchangeInfo before P3 has completed).
            Safe to call multiple times.
            """
            if self.is_started:
                return
            try:
                from aiohttp import ClientTimeout, TCPConnector
                connector = TCPConnector(limit=50, ttl_dns_cache=300)
                self.session = aiohttp.ClientSession(timeout=ClientTimeout(total=15), connector=connector)
                # Public-only AsyncClient (empty keys) is fine for unsigned endpoints
                self.client = await AsyncClient.create(api_key="", api_secret="", testnet=self.testnet)
                self._ready = True
                await self._report_status("OK", {"event": "lazy_public_bootstrap"})
                self.logger.info("ExchangeClient lazy public bootstrap initialized.")
            except Exception as e:
                self.logger.warning("ExchangeClient lazy public bootstrap failed: %s", e, exc_info=True)
                raise

        # ------------- unified HTTP wrapper -------------
        async def _request(
            self, method: str, path: str, params: Optional[Dict[str, Any]] = None, *, signed: bool = False, api: str = "spot_api"
        ) -> Any:
            """
            Unified request wrapper.
            api: "spot_api" | "spot_sapi" | "um_futures"
            """
            # Allow lazy public bootstrap for unsigned/public GETs used during early phases
            if not self.is_started:
                if not signed and method.upper() == "GET":
                    await self._ensure_started_public()
                else:
                    raise RuntimeError("ExchangeClient session not started. Call await start() first.")

            if api == "spot_api":
                base = self.base_url_spot_api
            elif api == "spot_sapi":
                base = self.base_url_spot_sapi
            elif api == "um_futures":
                base = self.base_url_um
            else:
                raise ValueError(f"Unknown api family: {api}")

            url = f"{base}{path}"
            headers = {"User-Agent": "octivault-trader/2.1"}
            if self.api_key:
                headers["X-MBX-APIKEY"] = self.api_key
            params = params or {}

            # configurable weight guard per path
            weight = self._path_weight_overrides.get(path, 2 if path.startswith("/api/v3/klines") else 1)
            now = time.time()
            bucket = self._weight_counters[path]
            if now - bucket["ts"] > self._weight_window:
                bucket["ts"], bucket["w"] = now, 0
            if bucket["w"] + weight > self._weight_limit:
                await asyncio.sleep(max(0.0, self._weight_window - (now - bucket["ts"])))
                bucket["ts"], bucket["w"] = time.time(), 0
            bucket["w"] += weight

            if signed:
                if not (self.api_key and self.api_secret):
                    raise BinanceAPIException(
                        "Signed endpoint requires API_KEY and API_SECRET (Binance -2015 likely).",
                        code=-2015
                    )
                params["timestamp"] = int(time.time() * 1000 + self._time_offset_ms)
                params.setdefault("recvWindow", self.recv_window_ms)
                query_string = urllib.parse.urlencode(params)
                signature = hmac.new(
                    self.api_secret.encode("utf-8"),
                    query_string.encode("utf-8"),
                    hashlib.sha256,
                ).hexdigest()
                params["signature"] = signature

            # exponential backoff + jitter for transient cases
            backoffs = [0.2, 0.5, 1.0, 2.0]
            last_err = None
            for delay in [0.0, *backoffs]:
                if delay:
                    await asyncio.sleep(delay + random.uniform(0, delay/4))
                try:
                    async with self.session.request(method, url, headers=headers, params=params) as response:
                        text = await response.text()
                        if response.status >= 400:
                            # Try parse JSON error and surface native Binance codes
                            try:
                                err = json.loads(text) if text else {}
                            except Exception:
                                err = {"msg": text}
                            code = err.get("code", response.status)
                            msg = err.get("msg", text)
                            if code == -2015:
                                self.logger.error("Binance -2015 on %s %s (api=%s): %s", method, path, api, msg)
                            # Handle time skew (-1021): resync once and retry on next backoff iteration
                            if code == -1021:
                                try:
                                    await self._resync_time()
                                except Exception:
                                    pass
                                # Continue to next retry iteration
                                continue
                            # honor server-suggested retry for RL
                            if response.status in (418, 429):
                                try:
                                    ra = float(response.headers.get("Retry-After", "0"))
                                    if ra > 0:
                                        await asyncio.sleep(min(ra, 3.0))
                                except Exception:
                                    pass
                            raise BinanceAPIException(msg, code=code)
                        return json.loads(text) if text else {}
                except aiohttp.ClientResponseError as e:
                    last_err = e
                    # rate-limit
                    if e.status in (418, 429):
                        try:
                            ra = float(e.headers.get("Retry-After", "0"))
                            if ra > 0:
                                await asyncio.sleep(min(ra, 2.0))
                        except Exception:
                            pass
                        continue
                    if e.status == 400:
                        raise BinanceAPIException(f"API error: {getattr(e, 'message', str(e))}", code=e.status)
                    if e.status in (408,) or (500 <= e.status < 600):
                        continue
                    raise NetworkException(f"{method} {path} (api={api}) failed: {e}") from e
                except aiohttp.ClientError as e:
                    last_err = e
                    continue
            raise NetworkException(f"{method} {path} (api={api}) network error after retries: {last_err}")

        # ------------- market data helpers -------------
        async def get_24hr_tickers(self) -> List[Dict[str, Any]]:
            """
            Cached view of /api/v3/ticker/24hr with a short TTL.
            Returns a list of dicts with at least: symbol, lastPrice, priceChangePercent, quoteVolume, volume.
            """
            now = time.time()
            ts = getattr(self, "_ticker_24h_cache_ts", 0.0)
            if self._ticker_24h_cache and (now - ts) < self._ticker_24h_ttl_sec:
                return list(self._ticker_24h_cache.values())

            data = await self._request("GET", "/api/v3/ticker/24hr", api="spot_api")
            cache: Dict[str, Dict[str, Any]] = {}
            for t in data or []:
                try:
                    sym = str(t.get("symbol", "")).upper()
                    cache[sym] = {
                        "symbol": sym,
                        "lastPrice": float(t.get("lastPrice", 0.0)),
                        "priceChangePercent": float(t.get("priceChangePercent", 0.0)),
                        "quoteVolume": float(t.get("quoteVolume", 0.0)),
                        "volume": float(t.get("volume", 0.0)),
                    }
                except Exception:
                    continue
            self._ticker_24h_cache = cache
            self._ticker_24h_cache_ts = now
            return list(cache.values())

        async def get_ticker_24h_normalized(self, symbol: str) -> Dict[str, Any]:
            """
            Return a normalized 24h ticker dict with snake_case keys for a single symbol.
            Keys: symbol, last_price, price_change_pct, base_volume, quote_volume, trade_count, ts_exchange.
            """
            sym = self._norm_symbol(symbol)
            # Try cache first
            now = time.time()
            ts = getattr(self, "_ticker_24h_cache_ts", 0.0)
            if self._ticker_24h_cache and (now - ts) < self._ticker_24h_ttl_sec:
                t = self._ticker_24h_cache.get(sym)
                if t:
                    return {
                        "symbol": sym,
                        "last_price": float(t.get("lastPrice", 0.0)),
                        "price_change_pct": float(t.get("priceChangePercent", 0.0)),
                        "base_volume": float(t.get("volume", 0.0)),
                        "quote_volume": float(t.get("quoteVolume", 0.0)),
                        "trade_count": int(t.get("count", 0)) if isinstance(t.get("count", 0), (int, float)) else 0,
                        "ts_exchange": 0.0,
                    }
            # Fetch single from API if not in cache
            raw = await self._request("GET", "/api/v3/ticker/24hr", {"symbol": sym}, api="spot_api")
            def _f(x, default=0.0):
                try:
                    return float(x)
                except Exception:
                    return default
            out = {
                "symbol": sym,
                "last_price": _f(raw.get("lastPrice")),
                "price_change_pct": _f(raw.get("priceChangePercent")),
                "base_volume": _f(raw.get("volume")),
                "quote_volume": _f(raw.get("quoteVolume")),
                "trade_count": int(raw.get("count") or 0),
                "ts_exchange": (_f(raw.get("closeTime")) / 1000.0) if raw.get("closeTime") else 0.0,
            }
            # also refresh the cache entry for consistency
            try:
                self._ticker_24h_cache[self._norm_symbol(symbol)] = {
                    "symbol": sym,
                    "lastPrice": out["last_price"],
                    "priceChangePercent": out["price_change_pct"],
                    "quoteVolume": out["quote_volume"],
                    "volume": out["base_volume"],
                }
                self._ticker_24h_cache_ts = now
            except Exception:
                pass
            return out

        async def get_24hr_volume(self, symbol: str) -> float:
            """
            WalletScanner expects this: return 24h *quote* volume (e.g., USDT) for the symbol.
            Falls back to direct /api/v3/ticker/24hr if cache miss.
            """
            sym = self._norm_symbol(symbol)
            # Use cached multi-ticker view when available
            now = time.time()
            ts = getattr(self, "_ticker_24h_cache_ts", 0.0)
            if self._ticker_24h_cache and (now - ts) < self._ticker_24h_ttl_sec:
                t = self._ticker_24h_cache.get(sym)
                if t is not None:
                    try:
                        return float(t.get("quoteVolume", 0.0))
                    except Exception:
                        return 0.0
            # Fallback: single call
            try:
                raw = await self._request("GET", "/api/v3/ticker/24hr", {"symbol": sym}, api="spot_api")
                return float(raw.get("quoteVolume", 0.0))
            except Exception:
                return 0.0

        async def get_best_bid_ask(self, symbol: str) -> Tuple[Optional[float], Optional[float]]:
            """
            Return (bid, ask) with a tiny microcache to avoid double hits during validation/submit.
            """
            sym = self._norm_symbol(symbol)
            now = time.time()
            cache = getattr(self, "_book_cache", {})
            ent = cache.get(sym)
            if ent and (now - ent["ts"] < self._BOOK_TTL):
                return ent["bid"], ent["ask"]
            try:
                data = await self._request("GET", "/api/v3/ticker/bookTicker", {"symbol": sym}, api="spot_api")
                bid = float(data.get("bidPrice")) if data and data.get("bidPrice") else None
                ask = float(data.get("askPrice")) if data and data.get("askPrice") else None
                cache[sym] = {"ts": now, "bid": bid, "ask": ask}
                self._book_cache = cache
                return bid, ask
            except Exception:
                return None, None

        async def get_current_price(self, symbol: str) -> float:
            """Retrieves and caches the current price of a symbol."""
            sym = self._norm_symbol(symbol)
            now = time.time()
            # Micro-cache with configurable TTL (default 1s)
            if sym in self.price_cache and now - self.price_cache[sym][1] < self._px_ttl:
                return self.price_cache[sym][0]

            try:
                response = await self._request("GET", "/api/v3/ticker/price", {"symbol": sym}, api="spot_api")
                price = float(response["price"])
                self.price_cache[sym] = (price, now)
                return price
            except Exception as e:
                raise NetworkException(f"Failed to get price for {sym}: {e}")

        async def get_ohlcv(
            self,
            symbol: str,
            interval: str = "1h",
            limit: int = 500,
        ) -> List[List[float]]:
            """
            Return OHLCV rows from /api/v3/klines.
            Each row = [openTime, open, high, low, close, volume] as floats.
            """
            sym = self._norm_symbol(symbol)
            data = await self._request("GET", "/api/v3/klines", {"symbol": sym, "interval": interval, "limit": limit}, api="spot_api")
            return [[float(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5])] for x in data]

        async def get_klines(self, symbol: str, interval: str = "1h", limit: int = 500) -> List[List[float]]:
            """
            Compatibility alias returning raw klines as lists (Binance shape preserved).
            """
            sym = self._norm_symbol(symbol)
            data = await self._request("GET", "/api/v3/klines", {"symbol": sym, "interval": interval, "limit": limit}, api="spot_api")
            return data

        async def _sync_exchange_info(self):
            """Fetches and caches exchange information and symbol filters."""
            # Ensure we can hit public endpoints even if start() hasn't run yet
            if not self.is_started:
                try:
                    await self._ensure_started_public()
                except Exception:
                    pass
            now = time.time()
            # Cache for 12 hours
            if self._exchange_info and now - self._exchange_info_timestamp < 12 * 3600:
                return

            async with self._sync_lock:
                if self._exchange_info and now - self._exchange_info_timestamp < 12 * 3600:
                    return

                try:
                    response = await self._request("GET", "/api/v3/exchangeInfo", api="spot_api")
                    self._exchange_info = response
                    self._exchange_info_timestamp = now
                    self.symbol_filters = {}
                    for s in response.get("symbols", []):
                        sym = s["symbol"]
                        filters = {f["filterType"]: f for f in s.get("filters", [])}
                        # Attach precisions from symbol info:
                        filters["_precisions"] = {
                            "baseAssetPrecision":  s.get("baseAssetPrecision"),
                            "quotePrecision":      s.get("quotePrecision"),
                            "quoteAssetPrecision": s.get("quoteAssetPrecision"),
                            "baseCommissionPrecision":  s.get("baseCommissionPrecision"),
                            "quoteCommissionPrecision": s.get("quoteCommissionPrecision"),
                        }
                        self.symbol_filters[sym] = filters
                    self.logger.info("Exchange info synchronized.")
                    await self._report_status("OK", {"event": "exchange_info_synced"})
                except Exception as e:
                    self.logger.error("Failed to sync exchange info: %s", e)
                    await self._report_status("ERROR", {"event": "exchange_info_sync_failed", "reason": str(e)})

        def _pick_lot_filter(self, filters: dict) -> dict:
            # P9 Fix 4: Prefer LOT_SIZE for micro trading
            if not isinstance(filters, dict):
                return {}
            return filters.get("LOT_SIZE") or filters.get("MARKET_LOT_SIZE") or {}

        def _get_quantity_precision(self, symbol: str) -> int:
            """Derives quantity precision from the (market) lot step size as a string."""
            symbol = self._norm_symbol(symbol)
            filters = self.symbol_filters.get(symbol, {})
            lot = self._pick_lot_filter(filters)
            step_size = str(lot.get("stepSize", "1"))
            return max(0, -Decimal(step_size).as_tuple().exponent)

        async def _validate_order(self, symbol: str, quantity: float, side: str):
            """Validates order against exchange filters (uses MARKET_LOT_SIZE/NOTIONAL when available)."""
            symbol = self._norm_symbol(symbol)
            filters = self.symbol_filters.get(symbol, {})
            lot = self._pick_lot_filter(filters)
            min_qty = float(lot.get("minQty", 0))
            max_qty = float(lot.get("maxQty", float("inf")))
            step_size = str(lot.get("stepSize", "0.000001"))
            notional = filters.get("NOTIONAL") or filters.get("MIN_NOTIONAL") or {}
            min_notional = float(notional.get("minNotional", 0))

            if quantity < min_qty:
                raise ValueError(f"MinNotional/LOT_SIZE violation: qty {quantity} < minQty {min_qty}")
            if quantity > max_qty:
                raise ValueError(f"LOT_SIZE violation: qty {quantity} > maxQty {max_qty}")
            # Snap to stepSize (string-safe)
            q = Decimal(str(quantity))
            s = Decimal(step_size)
            snapped = (q / s).quantize(Decimal("1"), rounding=ROUND_DOWN) * s
            qty_f = float(snapped)
            # Check notional against book prices (ask for BUY, bid for SELL; fallback to last)
            try:
                bid, ask = await self.get_best_bid_ask(symbol)
                px = ask if str(side).upper() == "BUY" else (bid or ask)
                if px is None:
                    px = await self.get_current_price(symbol)
                if qty_f * float(px) < min_notional:
                    raise ValueError(f"MIN_NOTIONAL violation: {qty_f} * {px:.8f} < {min_notional}")
            except Exception:
                self.logger.warning("Price unavailable for MIN_NOTIONAL check; proceeding with caution.")
            return qty_f

        # ------------- canonical order path -------------
        async def market_buy(self, symbol: str, quote_amount: float, *, tag: str = "meta") -> dict:
            if not self.is_started:
                await self.start()
            if not self.client:
                raise RuntimeError("ExchangeClient not started; AsyncClient unavailable")
            sym = self._norm_symbol(symbol)
            ts_ms = int(time.time() * 1000)
            safe_tag = (tag or "meta_unknown").replace("/", "_")
            base_coid = f"octi-{ts_ms}-{safe_tag}"
            client_order_id = "".join(
                ch if (ch.isalnum() or ch in "_-") else "_" for ch in base_coid
            )[:36]
            return await self.client.create_order(
                symbol=sym,
                side="BUY",
                type="MARKET",
                quoteOrderQty=float(quote_amount),
                newClientOrderId=client_order_id,
            )

        async def place_market_order(
            self,
            symbol: str,
            side: str,
            *,
            quantity: Optional[float] = None,
            quote: Optional[float] = None,
            tag: str = "",
            clientOrderId: Optional[str] = None,
            _timeInForce: Optional[str] = None,  # unused for MARKET on Binance
            max_slippage_bps: Optional[int] = None,
        ) -> dict:
            """
            Canonical MARKET order entrypoint (spec §3.6, §3.19).
            Supports either `quantity` or `quoteOrderQty`.
            """
            sym = self._norm_symbol(symbol)

            # Fee-safety padding for quote orders (defaults to config or 10 bps)
            fee_bps = self.fee_buffer_bps
            if quote is not None:
                quote = float(Decimal(str(quote)) * (Decimal(1) - Decimal(fee_bps) / Decimal(10_000)))

            # Hygiene filters
            filters = await self.get_exchange_filters(sym)
            min_notional = float(filters.get("min_notional", 0))
            step_size = str(filters.get("step_size", "0.000001"))

            # Build idempotent clientOrderId with tag
            ts_ms = int(time.time() * 1000)
            # Binance newClientOrderId only allows: A-Z, a-z, 0-9, underscore, hyphen
            # Replace '/' with '_' to avoid ExternalAPIError
            safe_tag = (tag or 'meta_unknown').replace("/", "_")
            base_coid = "".join(ch if (ch.isalnum() or ch in "_-") else "_" for ch in (clientOrderId or f"octi-{ts_ms}-{safe_tag}"))[:36]

            # Validate required order tag per P9 contract
            self._validate_order_tag(tag)

            # Circuit breaker check (submit)
            if not self._submit_breaker.allow():
                reason = "circuit_open_submit"
                await self._emit_summary("ORDER_BLOCKED", symbol=sym, side=side, tag=tag, status="ERROR", reason=reason, error_code="CB_OPEN")
                return self._normalize_exec_result(base_coid, {"status": "REJECTED"}, False, error_code="CB_OPEN", error_msg="Submit breaker open")

            try:
                # Ensure spend meets minNotional (if we have a last price)
                px = await self.get_current_price(sym)
                if quote is not None:
                    spend = quote
                else:
                    if quantity is None:
                        raise ValueError("Either quantity or quote must be provided")
                    qty_dec = self._round_step(Decimal(str(quantity)), step_size)
                    # P9 Fix 3: Hard fail if qty becomes zero after rounding
                    if qty_dec <= 0:
                        raise ValueError("ZeroQuantityAfterRounding")
                    spend = float(qty_dec) * float(px)

                if spend < (min_notional * 1.0):
                    await self._emit_summary("ORDER_REJECTED", symbol=sym, side=side, tag=tag, status="ERROR", reason="MinNotionalViolation", error_code="MIN_NOTIONAL")
                    return self._normalize_exec_result(base_coid, {"status": "REJECTED"}, False, error_code="MIN_NOTIONAL", error_msg="Below minNotional")

                # Submit MARKET order (quantity OR quote)
                payload = {"symbol": sym, "side": side.upper(), "type": "MARKET", "newClientOrderId": base_coid}
                # NOTE: timeInForce is not used with MARKET on Binance; do not include it.
                if quote is not None:
                    if side.upper() != "BUY":
                        raise ValueError("quoteOrderQty is only supported for MARKET BUY in this build.")
                    
                    # P9 Fix: Convert quote to quantity to avoid "quoteOrderQty not supported" exchange errors
                    # We use the current price (px) to estimate the quantity needed.
                    if px <= 0:
                        raise ValueError(f"Invalid price {px} for quote-to-quantity conversion")
                    
                    raw_qty = float(quote) / float(px)
                    qty_dec = self._round_step(Decimal(str(raw_qty)), step_size)
                    
                    if qty_dec <= 0:
                        raise ValueError("ZeroQuantityAfterRounding (quote conversion)")
                    
                    payload["quantity"] = str(qty_dec)
                    # Ensure quoteOrderQty is NOT sent
                else:
                    qty_dec = self._round_step(Decimal(str(quantity)), step_size)
                    if qty_dec <= 0:
                        raise ValueError("ZeroQuantityAfterRounding")
                    payload["quantity"] = str(qty_dec)

                self.logger.debug("Submitting order payload=%s", payload)
                await self._emit_summary("ORDER_SENT", symbol=sym, side=side, tag=tag, status="SENT")

                # Capture reference price before submit for optional slippage check
                try:
                    bid, ask = await self.get_best_bid_ask(sym)
                    ref_px = (ask if side.upper() == "BUY" else bid) or px
                except Exception:
                    ref_px = px

                raw = await self._request("POST", "/api/v3/order", payload, signed=True, api="spot_api")
                self._submit_breaker.record(True)
                res = self._normalize_exec_result(base_coid, raw, True)
                res.setdefault("hygiene", {})
                res["hygiene"].update({"fee_safety": True, "min_notional_ok": True})
                # Invalidate balances cache after order acceptance/fill
                self._acct_cache_ts = 0.0

                # Optional soft slippage warning (do not reject here)
                try:
                    if max_slippage_bps and ref_px and res.get("price"):
                        slip_bps = (abs(res["price"] - ref_px) / ref_px) * 10_000
                        if slip_bps > max_slippage_bps:
                            await self._emit_summary(
                                "SLIPPAGE_WARN", symbol=sym, side=side, tag=tag,
                                status="WARN", reason="max_slippage_bps_exceeded",
                                slippage_bps=round(slip_bps, 2)
                            )
                except Exception:
                    pass

                await self._emit_summary(
                    "ORDER_FILLED" if res["status"] == "FILLED" else "ORDER_ACCEPTED",
                    symbol=sym, side=side, qty=res["executedQty"], price=res["price"], tag=tag,
                    status=res["status"], reason="filled" if res["status"] == "FILLED" else "accepted"
                )
                return res

            except Exception as e:
                self._submit_breaker.record(False)
                code = getattr(e, "code", None)
                if code == -1013:
                    # Filter failure (e.g., MIN_NOTIONAL) → map to MIN_NOTIONAL for clearer observability
                    await self._emit_summary(
                        "ORDER_REJECTED", symbol=sym, side=side, tag=tag, status="ERROR",
                        reason="MinNotionalViolation", error_code="MIN_NOTIONAL"
                    )
                    return self._normalize_exec_result(
                        base_coid, {"status": "REJECTED"}, False,
                        error_code="MIN_NOTIONAL", error_msg=str(e)
                    )
                # Idempotent status check (query breaker)
                if self._query_breaker.allow():
                    try:
                        qraw = await self.get_order_status(sym, base_coid)
                        if qraw is None:
                            raise RuntimeError("order_status_unknown")
                        self._query_breaker.record(True)
                        res = self._normalize_exec_result(base_coid, qraw, True)
                        # Invalidate balances cache if exchange confirms order existence
                        self._acct_cache_ts = 0.0
                        await self._emit_summary("ORDER_STATUS_RECOVERED", symbol=sym, side=side, tag=tag, status=res["status"], reason="recovered")
                        return res
                    except Exception:
                        self._query_breaker.record(False)
                await self._emit_summary(
                    "ORDER_REJECTED", symbol=sym, side=side, tag=tag, status="ERROR",
                    reason="ExternalAPIError", error_code="EXTERNAL_API"
                )
                return self._normalize_exec_result(base_coid, {"status": "REJECTED"}, False, error_code="EXTERNAL_API", error_msg=str(e))

        def _normalize_exec_result(self, order_id: str, raw: dict, status_ok: bool, error_code=None, error_msg=None) -> dict:
            # Canonical ExecResult (§3.7)
            res = {
                "order_id": order_id,
                "ok": bool(status_ok),
                "status": raw.get("status", "REJECTED"),
                "executedQty": float(raw.get("executedQty", 0.0)),
                "price": float(
                    raw.get("price")
                    or (raw.get("fills", [{}])[0].get("price", 0.0))
                    or raw.get("avgPrice", 0.0)
                    or 0.0
                ),
                "cummulativeQuoteQty": float(raw.get("cummulativeQuoteQty", 0.0)),
                "fills": [
                    {
                        "qty": float(f.get("qty", 0.0)),
                        "price": float(f.get("price", 0.0)),
                        "commission": float(f.get("commission", 0.0) or 0.0),
                        "commissionAsset": f.get("commissionAsset") or f.get("commission_asset"),
                    }
                    for f in raw.get("fills", [])
                ],
                "exchange_order_id": raw.get("orderId") or raw.get("order_id") or "",
                "error_code": error_code,
                "error_msg": error_msg,
                "ts_exchange": raw.get("transactTimeIso") or self._now_iso(),
            }
            return res

        # ------------- legacy send path (kept for compat) -------------
        # DEPRECATED: kept for backward compatibility only; disabled unless ALLOW_LEGACY_ORDER_PATH is set.
        async def send_order(
            self,
            symbol: str,
            side: str,
            quantity: Optional[float] = None,
            quote: Optional[float] = None,
            tag: str = "",
            clientOrderId: Optional[str] = None,
            _timeInForce: Optional[str] = None,  # unused for MARKET on Binance
            max_slippage_bps: Optional[int] = None,
        ) -> str:
            """Legacy path is disabled by default to preserve the single order path (P9)."""
            if not bool(self.config.get("ALLOW_LEGACY_ORDER_PATH", False)):
                raise NotImplementedError("Legacy order path is disabled. Use place_market_order().")
            symbol = self._norm_symbol(symbol)
            if self.paper_trade:
                return await self._send_paper_trade_order(symbol, side, quantity, quote, tag)
            # If explicitly enabled, fall back to old behavior:
            if (quantity is None and quote is None) or (quantity is not None and quote is not None):
                raise ValueError("Must specify either quantity or quote, but not both.")
            if quote is not None and side.upper() != "BUY":
                raise ValueError("Quote orders are only supported for market buy.")
            return await (self._send_real_order_quantity(symbol, side, quantity, tag) if quantity is not None
                        else self._send_real_order_quote(symbol, side, quote, tag))

        async def _send_real_order_quantity(
            self,
            symbol: str,
            side: str,
            quantity: float,
            tag: str,
            client_order_id: Optional[str] = None,
        ) -> str:
            await self._sync_exchange_info()
            await self._validate_order(symbol, quantity, side)

            params = {"symbol": symbol, "side": side.upper(), "type": "MARKET"}
            # Snap to step according to filters (prefer MARKET_LOT_SIZE for MARKET orders)
            filters = self.symbol_filters.get(symbol, {})
            lot = self._pick_lot_filter(filters)
            step_size = str(lot.get("stepSize", "0.000001"))
            qty_dec = self._round_step(Decimal(str(quantity)), step_size)
            if qty_dec <= 0:
                raise ValueError("ZeroQuantityAfterRounding")
            params["quantity"] = str(qty_dec)

            cid = client_order_id or f"octi-{int(time.time()*1000)}-{tag or 'meta'}"
            params["newClientOrderId"] = str(cid)[:36]
            raw = await self._request("POST", "/api/v3/order", params, signed=True, api="spot_api")
            return str(raw.get("orderId"))

        async def _send_real_order_quote(
            self,
            symbol: str,
            side: str,
            quote: float,
            tag: str,
            client_order_id: Optional[str] = None,
        ) -> str:
            await self._sync_exchange_info()
            # Enforce min notional on quote directly
            filters = self.symbol_filters.get(symbol, {})
            notional = filters.get("NOTIONAL") or filters.get("MIN_NOTIONAL") or {}
            min_notional = float(notional.get("minNotional", 0))
            if quote < min_notional:
                raise ValueError(f"MIN_NOTIONAL violation: quote {quote} < {min_notional}")

            # Use quote asset precision from exchangeInfo
            qp = int(filters.get("_precisions", {}).get("quotePrecision", 0) or 0)
            q = Decimal(str(quote)).quantize(Decimal(10) ** -qp, rounding=ROUND_DOWN)
            params = {"symbol": symbol, "side": side.upper(), "type": "MARKET", "quoteOrderQty": str(q)}
            cid = client_order_id or f"octi-{int(time.time()*1000)}-{tag or 'meta'}"
            params["newClientOrderId"] = str(cid)[:36]
            raw = await self._request("POST", "/api/v3/order", params, signed=True, api="spot_api")
            return str(raw.get("orderId"))

        async def _send_paper_trade_order(
            self,
            symbol: str,
            side: str,
            quantity: Optional[float] = None,
            quote: Optional[float] = None,
            tag: str = "",
        ) -> str:
            """Simulates an order for paper trading."""
            order_id = f"paper-{int(time.time() * 1000)}"
            fill_price = await self.get_current_price(symbol)
            fee = 0.001  # 0.1% paper fee

            if quote:
                quantity = float(quote) / float(fill_price)

            cid = f"octi-{order_id}-{tag or 'paper'}"[:36]
            order = {
                "orderId": order_id,
                "clientOrderId": cid,
                "symbol": symbol,
                "side": side.upper(),
                "status": "FILLED",
                "executedQty": quantity,
                "cummulativeQuoteQty": (quantity or 0.0) * float(fill_price),
                "price": float(fill_price),
                "fills": [{"price": float(fill_price), "qty": float(quantity or 0.0), "commission": fee}]
            }
            self._paper_trade_orders[str(order_id)] = order
            return str(order_id)

        # --------- P9: strict tag validator ----------
        def _validate_order_tag(self, tag: str) -> None:
            if not tag:
                raise ValueError("Missing order tag")

            # Configurable allowlists (handle both dict and object configs)
            cfg_exact = set(getattr(self.config, "ORDER_TAG_ALLOWLIST", None) or 
                        (self.config.get("ORDER_TAG_ALLOWLIST", []) if hasattr(self.config, "get") else []))
            cfg_prefix = set(getattr(self.config, "ORDER_TAG_PREFIX_ALLOWLIST", None) or 
                            (self.config.get("ORDER_TAG_PREFIX_ALLOWLIST", []) if hasattr(self.config, "get") else []))

            # Built-in safe tags
            builtin_exact = {
                "balancer", "liquidation", "tp_sl", "recovery", "rebalance",
                "screener", "risk", "backtest"
            }
            builtin_prefix = {"meta/", "strategy/", "agent/"}

            if (tag in builtin_exact) or any(tag.startswith(p) for p in builtin_prefix):
                return
            if (tag in cfg_exact) or any(tag.startswith(p) for p in cfg_prefix):
                return

            raise ValueError(f"Invalid order tag: {tag}")

        # ------------- balances -------------
        async def get_spot_balances(self) -> Dict[str, Dict[str, float]]:
            """Spot balances via /api/v3/account (safe; no SAPI permission required)."""
            data = await self._request("GET", "/api/v3/account", signed=True, api="spot_api")
            out = {}
            for b in data.get("balances", []):
                free = float(b.get("free", 0))
                locked = float(b.get("locked", 0))
                if free or locked:
                    out[b["asset"]] = {"free": free, "locked": locked}
            return out

        async def get_account_balance(self, asset: str) -> dict:
            """
            Lightweight balance lookup used by ExecutionManager.
            Returns {'free': float, 'locked': float} or zeros if not found.
            """
            try:
                now = time.time()
                # Return from cache if fresh
                if self._acct_cache and (now - self._acct_cache_ts) < self._acct_ttl:
                    bal = self._acct_cache.get(asset.upper())
                    if bal is not None:
                        return {"free": float(bal.get("free", 0.0)), "locked": float(bal.get("locked", 0.0))}
                # Refresh cache
                data = await self._request("GET", "/api/v3/account", signed=True, api="spot_api")
                cache = {}
                for b in data.get("balances", []):
                    free = float(b.get("free", 0))
                    locked = float(b.get("locked", 0))
                    if free or locked:
                        cache[b["asset"].upper()] = {"free": free, "locked": locked}
                self._acct_cache = cache
                self._acct_cache_ts = now
                bal = cache.get(asset.upper(), {"free": 0.0, "locked": 0.0})
                return {"free": float(bal.get("free", 0.0)), "locked": float(bal.get("locked", 0.0))}
            except BinanceAPIException:
                # Signed request without keys etc.; return zeros to allow callers to soft-handle
                return {"free": 0.0, "locked": 0.0}
            except Exception:
                self.logger.debug("get_account_balance failed", exc_info=True)
                return {"free": 0.0, "locked": 0.0}
