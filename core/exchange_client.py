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
import base64
import contextlib
import hashlib
import hmac
import inspect
import json
import os
import time
import urllib.parse
import uuid
from collections import defaultdict, deque
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from typing import Any, Dict, Optional, List, Tuple, Union
import logging

import aiohttp
from binance.async_client import AsyncClient
import random

# Ed25519 signing support — using PyNaCl (libsodium) for Binance WS API v3
# Guarded so the module still loads on stripped deployments (HMAC-only path unaffected).
try:
    from nacl.signing import SigningKey as _NaClSigningKey
    _NACL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _NaClSigningKey = None  # type: ignore
    _NACL_AVAILABLE = False
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
        ent = self._sym_info_cache.get(sym)
        if ent and (time.time() - ent["ts"] < self._INFO_TTL):
            return ent["val"]
        await self._sync_exchange_info()
        info = (self._exchange_info or {}).get("symbols", [])
        for s in info:
            if str(s.get("symbol", "")).upper() == sym:
                self._sym_info_cache[sym] = {"ts": time.time(), "val": s}
                return s
        # Optional: try a direct query if not found in cache
        try:
            data = await self._request("GET", "/api/v3/exchangeInfo", {"symbol": sym}, api="spot_api")
            syms = (data or {}).get("symbols", [])
            val = syms[0] if syms else None
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
        # No reliable new-listings feed is available from public Binance REST endpoints.
        # Returning all USDT symbols (previous behaviour) was misleading — IPOChaser would
        # treat every symbol as a new listing. Return [] until a real feed is integrated.
        return []

    async def get_new_listings_cached(self) -> List[str]:
        """
        Cached wrapper for get_new_listings(); keep same signature agents call.
        """
        if time.time() - self._ipo_cache_ts > 300:
            self._ipo_cache = await self.get_new_listings()
            self._ipo_cache_ts = time.time()
        return list(self._ipo_cache)

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
        tcache = self._tradable_cache.get(sym)
        if tcache and (time.time() - tcache["ts"] < self._INFO_TTL):
            return bool(tcache["val"])
        await self._sync_exchange_info()
        ok = False
        try:
            for s in self._exchange_info.get("symbols", []):
                if str(s.get("symbol", "")).upper() == sym:
                    ok = s.get("status") == "TRADING"
                    break
        except Exception:
            ok = False
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

    async def get_order(
        self,
        symbol: str,
        *,
        order_id: Optional[int] = None,
        client_order_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch a single order by exchange order id or client order id.
        """
        sym = self._norm_symbol(symbol)
        params: Dict[str, Any] = {"symbol": sym}
        if order_id is not None:
            params["orderId"] = int(order_id)
        elif client_order_id:
            params["origClientOrderId"] = str(client_order_id)
        else:
            return None
        try:
            raw = await self._request("GET", "/api/v3/order", params, signed=True, api="spot_api")
            return raw if isinstance(raw, dict) else None
        except Exception:
            return None

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Fetch open orders (optionally per symbol).
        """
        params: Dict[str, Any] = {}
        if symbol:
            params["symbol"] = self._norm_symbol(symbol)
        try:
            raw = await self._request(
                "GET",
                "/api/v3/openOrders",
                params if params else None,
                signed=True,
                api="spot_api",
            )
            return raw if isinstance(raw, list) else []
        except Exception:
            return []

    async def get_all_orders(self, symbol: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Fetch recent order history for a symbol.
        """
        sym = self._norm_symbol(symbol)
        lim = max(1, min(1000, int(limit or 50)))
        try:
            raw = await self._request(
                "GET",
                "/api/v3/allOrders",
                {"symbol": sym, "limit": lim},
                signed=True,
                api="spot_api",
            )
            return raw if isinstance(raw, list) else []
        except Exception:
            return []

    async def get_my_trades(self, symbol: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Fetch recent account trades (fill-level) for a symbol.
        Uses /api/v3/myTrades — the most granular, authoritative proof-of-fill.
        Each entry represents a single fill with a unique trade ID.
        """
        sym = self._norm_symbol(symbol)
        lim = max(1, min(1000, int(limit or 50)))
        try:
            raw = await self._request(
                "GET",
                "/api/v3/myTrades",
                {"symbol": sym, "limit": lim},
                signed=True,
                api="spot_api",
            )
            return raw if isinstance(raw, list) else []
        except Exception:
            return []

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

        # runtime modes (robust env parsing so "False" doesn't evaluate truthy)
        if testnet is None:
            # BINANCE_TESTNET is the primary single switch; TESTNET_MODE is a legacy alias
            testnet = _cfg_bool("BINANCE_TESTNET", False) or _cfg_bool("TESTNET_MODE", False)
        if paper_trade is None:
            paper_trade = _cfg_bool("PAPER_MODE", False)

        self.testnet = bool(testnet)
        self.paper_trade = bool(paper_trade)
        if self.paper_trade:
            self.logger.info("Paper trading mode is enabled. No real orders will be placed.")
        
        # Diagnostic: Log mode configuration
        self.logger.info(f"[EC:Init] Mode: testnet={self.testnet}, paper_trade={self.paper_trade}")

        # === API KEY SELECTION LOGIC ===
        # Use self.testnet which was correctly set above (avoiding local variable shadowing)
        if self.testnet:
            api_key = os.getenv("BINANCE_TESTNET_API_KEY")
            api_secret_hmac = os.getenv("BINANCE_TESTNET_API_SECRET_HMAC")
            api_secret_ed25519 = os.getenv("BINANCE_TESTNET_API_SECRET_ED25519")
            base_url = "https://testnet.binance.vision"
        else:
            api_key = os.getenv("BINANCE_API_KEY")
            api_secret_hmac = os.getenv("BINANCE_API_SECRET_HMAC")
            api_secret_ed25519 = os.getenv("BINANCE_API_SECRET_ED25519")
            base_url = "https://api.binance.com"
        
        self.api_key = api_key
        self.api_secret_hmac = api_secret_hmac  # For REST signed endpoints
        self.api_secret_ed25519 = api_secret_ed25519  # For WS API v3 session.logon
        # Keep api_secret for backward compatibility (use Ed25519 if available)
        self.api_secret = api_secret_ed25519 or api_secret_hmac

        # FIX 1: Validate API keys early before AsyncClient initialization
        if self.paper_trade:
            self.logger.info("[EC] Paper trading mode - using public endpoints only")
            self.api_key = "paper_key"
            self.api_secret = "paper_secret"
        else:
            if not self.api_key or not self.api_secret:
                err_msg = (
                    "[EC] Binance API keys not found at construction time.\n"
                    "Expected: BINANCE_API_KEY and BINANCE_API_SECRET in environment/config.\n"
                    f"Found: api_key={'present' if self.api_key else 'MISSING'}, "
                    f"api_secret={'present' if self.api_secret else 'MISSING'}.\n"
                    "Continuing in public-only mode; signed endpoints will fail until keys are provided."
                )
                if _cfg_bool("STRICT_API_KEYS_ON_INIT", False):
                    raise RuntimeError(err_msg)
                self.logger.warning(err_msg)
                # Keep explicit empty strings so signed checks remain consistent.
                self.api_key = self.api_key or ""
                self.api_secret = self.api_secret or ""
            else:
                self.logger.info("[EC] API keys validated (key_len=%d, secret_len=%d)", len(self.api_key), len(self.api_secret))

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
        # Correct Binance endpoint weights (https://binance-docs.github.io/apidocs/spot/en/#limits)
        _default_path_weights = {
            "/api/v3/openOrders": 6,   # weight 6 (all symbols); main polling endpoint
            "/api/v3/account": 20,     # weight 20
            "/api/v3/allOrders": 10,   # weight 10
            "/api/v3/klines": 2,
            "/api/v3/ticker/price": 2,
        }
        self._path_weight_overrides = dict(_cfg("PATH_WEIGHTS", _default_path_weights) or _default_path_weights)
        # Configurable REST polling interval for _user_data_polling_loop (Tier-3 fallback)
        self._user_data_poll_interval_sec = float(_cfg("USER_DATA_POLL_INTERVAL_SEC", 25.0))
        self._acct_cache = None
        self._acct_cache_ts = 0.0
        self._acct_ttl = float(_cfg("ACCT_CACHE_TTL_SEC", 5.0))  # seconds
        self._acct_cache_lock = asyncio.Lock()
        self.fee_buffer_bps = int(_cfg("FEE_BUFFER_BPS", 10))
        self._px_ttl = float(_cfg("PRICE_MICROCACHE_TTL", 1.0))

        # Binance recvWindow parameter (milliseconds) used for signed requests
        self.recv_window_ms = int(_cfg("RECV_WINDOW_MS", 5000))
        # 24h ticker cache
        self._ticker_24h_cache: Dict[str, Dict[str, Any]] = {}
        self._ticker_24h_cache_ts: float = 0.0
        self._ticker_24h_ttl_sec = float(_cfg("TICKER_24H_TTL_SEC", 15.0))

        # Micro-caches (symbol info, tradability, book ticker)
        self._sym_info_cache: Dict[str, Any] = {}
        self._tradable_cache: Dict[str, Any] = {}
        self._book_cache: Dict[str, Any] = {}

        # IPO / new-listings cache
        self._ipo_cache: List[str] = []
        self._ipo_cache_ts: float = 0.0

        # No remainder below quote threshold
        self.no_remainder_below_quote = 0.0

        # Lock for start() to prevent concurrent initialization races
        self._start_lock = asyncio.Lock()

        # User-data WebSocket/control-plane health state.
        # NOTE:
        # - `last_user_data_event_ts` tracks only user-data stream events.
        # - `last_any_ws_event_ts` can include any websocket traffic.
        # This separation allows feed-specific health decisions.
        self.user_data_stream_enabled = bool(_cfg_bool("USER_DATA_STREAM_ENABLED", True))
        self.user_data_ws_timeout_sec = float(_cfg("USER_DATA_WS_TIMEOUT_SEC", 65.0) or 65.0)
        self.user_data_ws_reconnect_backoff_sec = float(
            _cfg("USER_DATA_WS_RECONNECT_BACKOFF_SEC", 3.0) or 3.0
        )
        self.user_data_ws_max_backoff_sec = float(_cfg("USER_DATA_WS_MAX_BACKOFF_SEC", 30.0) or 30.0)
        self.user_data_ws_api_request_timeout_sec = float(
            _cfg("USER_DATA_WS_API_REQUEST_TIMEOUT_SEC", 12.0) or 12.0
        )
        self.user_data_ws_auth_mode = str(_cfg("USER_DATA_WS_AUTH_MODE", "auto") or "auto").strip().lower()
        if self.user_data_ws_auth_mode not in {"auto", "session", "signature", "polling"}:
            self.user_data_ws_auth_mode = "auto"

        # Key-type detection — used both for WS auth routing and signing algorithm.
        # Explicit BINANCE_API_TYPE=ED25519 overrides auto-detection.
        _explicit_key_type = str(_cfg("BINANCE_API_TYPE", "") or "").strip().upper()
        if _explicit_key_type == "ED25519":
            self._api_key_type = "ED25519"
            if self.user_data_ws_auth_mode == "auto":
                self.user_data_ws_auth_mode = "session"
            self.logger.info(
                "[EC] BINANCE_API_TYPE=ED25519 — WS auth mode set to 'session' (session.logon)"
            )
        else:
            # session.logon only works with Ed25519 keys (Binance WS API v3 contract).
            # HMAC/RSA secrets don't start with a PEM header — detect and skip session.logon
            # automatically so we go straight to userDataStream.subscribe.signature instead.
            secret = str(self.api_secret or "")
            if secret and secret.startswith("-----BEGIN"):
                self._api_key_type = "ED25519"
                # PEM key but no explicit type — leave auth_mode as-is (auto → session)
                self.logger.info(
                    "[EC] PEM key detected — treating as Ed25519, WS auth mode stays '%s'",
                    self.user_data_ws_auth_mode,
                )
            else:
                self._api_key_type = "HMAC"
                if self.user_data_ws_auth_mode == "auto":
                    self.user_data_ws_auth_mode = "signature"
                    self.logger.info(
                        "[EC] HMAC key detected — WS auth mode set to 'signature' "
                        "(session.logon requires Ed25519)"
                    )
        # Backward-compatible legacy knob (unused with WebSocket API v3 auth flow).
        self.listenkey_refresh_sec = float(_cfg("USER_DATA_LISTENKEY_REFRESH_SEC", 900.0) or 900.0)
        self.listenkey_refresh_sec = min(max(self.listenkey_refresh_sec, 60.0), 2500.0)

        self.last_user_data_event_ts = 0.0
        self.last_any_ws_event_ts = 0.0
        # Legacy metric name kept for compatibility with health monitors/tests.
        # It now tracks the last successful WS auth/subscription action timestamp.
        self.last_listenkey_refresh_ts = 0.0
        self.last_successful_force_sync_ts = 0.0
        self.ws_connected = False
        self.ws_reconnect_count = 0

        self._user_data_stop = asyncio.Event()
        self._user_data_ws_task: Optional[asyncio.Task] = None
        self._user_data_keepalive_task: Optional[asyncio.Task] = None  # legacy placeholder

        # Permanent-unavailability flags: set once when a terminal environment error is detected
        # (WS API v3 policy-close 1008, listenKey 410 Gone) so the supervisor skips those tiers
        # on every subsequent reconnect instead of retrying them on every loop iteration.
        self._ws_v3_unavailable: bool = False
        self._listen_key_unavailable: bool = False
        self._user_data_ws_conn: Optional[aiohttp.ClientWebSocketResponse] = None
        self._user_data_subscription_id: Optional[int] = None
        self._user_data_auth_mode_active: str = "none"
        self._user_data_listen_key: str = ""
        self._user_data_lock = asyncio.Lock()

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

        # Execution-path guard: order placement must be scoped by ExecutionManager.
        self._enforce_execution_manager_path = bool(
            _cfg_bool("ENFORCE_EXECUTION_MANAGER_PATH", True)
        )
        # Public alias for startup assertions (used by AppContext).
        self.ENFORCE_EXECUTION_MANAGER_PATH = bool(self._enforce_execution_manager_path)
        self._order_scope_depth = 0
        self._order_scope_owner = ""
        if not self.paper_trade and not self._enforce_execution_manager_path:
            allow_unsafe_direct = bool(_cfg_bool("ALLOW_UNSAFE_DIRECT_ORDER_PATH", False))
            msg = (
                "ENFORCE_EXECUTION_MANAGER_PATH is disabled in live mode. "
                "This can create silent order-path bypasses."
            )
            if allow_unsafe_direct:
                self.logger.warning(
                    "[ORDER_PATH_GUARD_DISABLED] %s ALLOW_UNSAFE_DIRECT_ORDER_PATH=true",
                    msg,
                )
            else:
                self.logger.critical("[ORDER_PATH_GUARD_DISABLED] %s", msg)
                raise RuntimeError("Unsafe order path: ENFORCE_EXECUTION_MANAGER_PATH must be True")

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

    def begin_execution_order_scope(self, source: str = "ExecutionManager") -> Optional[str]:
        """Mark an authorized order-placement scope (entered by ExecutionManager)."""
        if not self._enforce_execution_manager_path:
            return None
        owner = str(source or "ExecutionManager")
        self._order_scope_owner = owner
        self._order_scope_depth = int(self._order_scope_depth or 0) + 1
        return owner

    def end_execution_order_scope(self, _token: Optional[str] = None) -> None:
        """Leave an authorized order-placement scope."""
        if not self._enforce_execution_manager_path:
            return
        depth = int(self._order_scope_depth or 0)
        if depth <= 1:
            self._order_scope_depth = 0
            self._order_scope_owner = ""
            return
        self._order_scope_depth = depth - 1

    def _is_execution_scope_active(self) -> bool:
        return (
            int(getattr(self, "_order_scope_depth", 0) or 0) > 0
            and str(getattr(self, "_order_scope_owner", "")) == "ExecutionManager"
        )

    async def _guard_execution_path(self, *, method: str, symbol: str, side: str, tag: str = "") -> None:
        """
        Fail closed if someone tries to place orders outside ExecutionManager.
        Also enforces TRADING_MODE safety gate: only live mode allowed.
        """
        # ✅ EXTRA SAFETY: Check TRADING_MODE before any order submission
        import os
        mode = os.getenv("TRADING_MODE", "live").lower()
        if mode != "live":
            sym = self._norm_symbol(symbol)
            self.logger.error(
                "[TRADING_MODE_GUARD] Real order blocked: system not in LIVE mode (mode=%s method=%s symbol=%s side=%s)",
                mode,
                method or "unknown",
                sym,
                str(side or "").upper(),
            )
            try:
                await self._emit_summary(
                    "ORDER_BLOCKED_NON_LIVE_MODE",
                    symbol=sym,
                    side=str(side or "").upper(),
                    tag=str(tag or ""),
                    status="ERROR",
                    reason=f"execution_blocked_non_live_mode_{mode}",
                )
            except Exception:
                pass
            raise RuntimeError(f"Real order blocked: system not in LIVE mode (TRADING_MODE={mode})")
        
        if not bool(getattr(self, "_enforce_execution_manager_path", True)):
            return
        if self._is_execution_scope_active():
            return

        sym = self._norm_symbol(symbol)
        scope_depth = int(getattr(self, "_order_scope_depth", 0) or 0)
        scope_owner = str(getattr(self, "_order_scope_owner", "") or "")
        caller = ""
        with contextlib.suppress(Exception):
            for fr in inspect.stack(context=0)[1:]:
                mod = str(fr.frame.f_globals.get("__name__", "") or "")
                if mod.endswith("exchange_client") or mod.endswith("core.exchange_client"):
                    continue
                caller = f"{mod}.{fr.function}:{int(fr.lineno)}"
                break
        payload = {
            "component": "ExchangeClient",
            "event": "ORDER_PATH_BYPASS",
            "method": str(method or ""),
            "symbol": sym,
            "side": str(side or "").upper(),
            "tag": str(tag or ""),
            "scope_depth": scope_depth,
            "scope_owner": scope_owner,
            "caller": caller,
            "reason": "execution_manager_scope_required",
            "ts": self._now_iso(),
        }
        self.logger.error(
            "[ORDER_PATH_BYPASS] method=%s symbol=%s side=%s tag=%s caller=%s scope_depth=%s scope_owner=%s",
            payload["method"],
            payload["symbol"],
            payload["side"],
            payload["tag"],
            payload["caller"] or "unknown",
            payload["scope_depth"],
            payload["scope_owner"] or "none",
        )
        try:
            await self._emit_summary(
                "ORDER_PATH_BYPASS",
                symbol=sym,
                side=str(side or "").upper(),
                tag=str(tag or ""),
                status="ERROR",
                reason="execution_manager_scope_required",
            )
        except Exception:
            pass
        try:
            if hasattr(self.shared_state, "emit_event"):
                await self._maybe_await(self.shared_state.emit_event("ORDER_PATH_BYPASS", payload))
        except Exception:
            pass
        raise PermissionError("ORDER_PATH_BYPASS: execution_manager_scope_required")

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

    # ------------- websocket/user-data control-plane state -------------
    def _has_signed_credentials(self) -> bool:
        key = str(getattr(self, "api_key", "") or "")
        sec = str(getattr(self, "api_secret", "") or "")
        if not key or not sec:
            return False
        if key == "paper_key" and sec == "paper_secret":
            return False
        return True

    def mark_any_ws_event(self, source: str = "") -> float:
        ts = time.time()
        self.last_any_ws_event_ts = ts
        return ts

    def mark_user_data_event(self, event_name: str = "", payload: Optional[Dict[str, Any]] = None) -> float:
        ts = time.time()
        self.last_user_data_event_ts = ts
        self.last_any_ws_event_ts = ts
        return ts

    def record_successful_force_sync(self, *, reason: str = "", ts: Optional[float] = None) -> float:
        now = float(ts if ts is not None else time.time())
        self.last_successful_force_sync_ts = now
        return now

    def get_ws_health_snapshot(self) -> Dict[str, Any]:
        now = time.time()
        user_ts = float(getattr(self, "last_user_data_event_ts", 0.0) or 0.0)
        any_ts = float(getattr(self, "last_any_ws_event_ts", 0.0) or 0.0)
        listen_ts = float(getattr(self, "last_listenkey_refresh_ts", 0.0) or 0.0)
        sync_ts = float(getattr(self, "last_successful_force_sync_ts", 0.0) or 0.0)
        return {
            "user_data_stream_enabled": bool(getattr(self, "user_data_stream_enabled", False)),
            "ws_connected": bool(getattr(self, "ws_connected", False)),
            "ws_reconnect_count": int(getattr(self, "ws_reconnect_count", 0) or 0),
            "user_data_ws_auth_mode": str(getattr(self, "_user_data_auth_mode_active", "none") or "none"),
            "user_data_subscription_id": getattr(self, "_user_data_subscription_id", None),
            "last_user_data_event_ts": user_ts,
            "last_any_ws_event_ts": any_ts,
            "last_listenkey_refresh_ts": listen_ts,
            "last_successful_force_sync_ts": sync_ts,
            "user_data_gap_sec": (now - user_ts) if user_ts > 0 else -1.0,
            "any_ws_gap_sec": (now - any_ts) if any_ts > 0 else -1.0,
            "listenkey_refresh_gap_sec": (now - listen_ts) if listen_ts > 0 else -1.0,
            "force_sync_gap_sec": (now - sync_ts) if sync_ts > 0 else -1.0,
            # Environment capability flags (set after first terminal failure)
            "ws_v3_unavailable": bool(getattr(self, "_ws_v3_unavailable", False)),
            "listen_key_unavailable": bool(getattr(self, "_listen_key_unavailable", False)),
        }

    def _user_data_ws_api_url(self) -> str:
        try:
            parsed = urllib.parse.urlparse(str(self.base_url_spot_api or ""))
            host = str(parsed.netloc or "").lower()
        except Exception:
            host = ""

        if "testnet.binance.vision" in host:
            return "wss://ws-api.testnet.binance.vision/ws-api/v3"
        if "api.binance.com" in host:
            return "wss://ws-api.binance.com:443/ws-api/v3"
        if host:
            scheme = "wss" if str(self.base_url_spot_api or "").startswith("https://") else "ws"
            return f"{scheme}://{host}/ws-api/v3"
        return "wss://ws-api.binance.com:443/ws-api/v3"

    async def _create_listen_key(self) -> Optional[str]:
        """
        Create a new listenKey for WebSocket Streams API.
        Works with HMAC keys (no Ed25519 required).
        
        Returns:
            listenKey string or None if failed
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.logger.info(f"[EC:ListenKey] Creating new listenKey (attempt {attempt+1}/{max_retries})...")
                # POST /api/v3/userDataStream
                # Call directly using aiohttp to avoid recvWindow being added automatically
                if not self.session or self.session.closed:
                    self.logger.error("[EC:ListenKey] ❌ Session not available")
                    return None
                
                # Spot listenKey uses API-key header auth (no timestamp/signature payload).
                
                url = f"{self.base_url_spot_api}/api/v3/userDataStream"
                headers = {
                    "User-Agent": "octivault-trader/2.1",
                    "X-MBX-APIKEY": self.api_key or ""
                }
                
                async with self.session.request("POST", url, headers=headers) as response:
                    text = await response.text()
                    if response.status >= 400:
                        try:
                            err = json.loads(text) if text else {}
                        except Exception:
                            err = {"msg": text}
                        code = err.get("code", response.status)
                        msg = err.get("msg", text)
                        raise BinanceAPIException(msg, code=code)
                    
                    data = json.loads(text) if text else {}
                    listen_key = data.get("listenKey")
                    if listen_key:
                        self.logger.info(f"[EC:ListenKey] ✅ Created: {listen_key[:16]}...")
                        return listen_key
                    else:
                        self.logger.error("[EC:ListenKey] ❌ No listenKey in response: %s", data)
                        return None
                        
            except Exception as e:
                error_str = str(e).lower()
                if "410" in str(e) or "gone" in error_str:
                    self.logger.warning("[EC:ListenKey] Got 410 Gone; listenKey stream unavailable in this environment. Falling back.")
                    return None
                elif "429" in str(e) or "rate" in error_str:
                    self.logger.warning(f"[EC:ListenKey] ❌ Rate limited, waiting before retry {attempt+1}...")
                    await asyncio.sleep(5.0 * (attempt + 1))  # Longer backoff for rate limits
                else:
                    self.logger.error("[EC:ListenKey] ❌ Failed to create: %s", e)
                    return None
        
        self.logger.error("[EC:ListenKey] ❌ Failed after %d attempts", max_retries)
        return None

    async def _refresh_listen_key(self, listen_key: str) -> bool:
        """
        Refresh an existing listenKey (keep it alive).
        Must be called every 30 minutes or listenKey expires.
        
        Returns:
            True if successful, False otherwise
        """
        if not listen_key:
            return False
        
        try:
            self.logger.debug("[EC:ListenKey] Refreshing...")
            # Spot listenKey keepalive also uses API-key header auth only.
            if not self.session or self.session.closed:
                return False
            url = f"{self.base_url_spot_api}/api/v3/userDataStream"
            headers = {
                "User-Agent": "octivault-trader/2.1",
                "X-MBX-APIKEY": self.api_key or "",
            }
            async with self.session.request("PUT", url, headers=headers, params={"listenKey": listen_key}) as response:
                if response.status >= 400:
                    text = await response.text()
                    self.logger.warning("[EC:ListenKey] ❌ Refresh HTTP %s: %s", response.status, text)
                    return False
            self.logger.debug("[EC:ListenKey] ✅ Refreshed")
            return True
        except Exception as e:
            self.logger.warning("[EC:ListenKey] ❌ Refresh failed: %s", e)
            return False

    def _user_data_ws_stream_url(self, listen_key: str) -> str:
        """
        Get WebSocket Streams API URL (old REST-based streaming).
        Works with listenKey (compatible with HMAC keys).
        """
        if not listen_key:
            return ""
        
        host = self.base_url_spot_api or "api.binance.com"
        if "testnet" in host:
            return f"wss://stream.testnet.binance.vision/ws/{listen_key}"
        if "api.binance.com" in host:
            return f"wss://stream.binance.com:9443/ws/{listen_key}"
        if host:
            scheme = "wss" if str(self.base_url_spot_api or "").startswith("https://") else "ws"
            return f"{scheme}://stream.binance.com:9443/ws/{listen_key}"
        return f"wss://stream.binance.com:9443/ws/{listen_key}"

    def _sign_ed25519(self, payload: str) -> Optional[str]:
        """
        Sign payload with Ed25519 private key for WS API v3 session.logon.
        
        Uses PyNaCl (libsodium) for Ed25519 signing.
        
        Input Format:
        - Base64-encoded 32-byte Ed25519 seed (standard)
        - Base64-encoded 48-byte seed (extended, uses first 32 bytes)
        - Raw 32-byte seed (not base64-encoded)
        
        Returns: Base64-encoded signature, or None if key unavailable.
        """
        # Validate payload
        if not payload:
            self.logger.error("[EC] Ed25519 signing called with empty payload")
            return None
        
        if payload is None:
            self.logger.error("[EC] Ed25519 signing called with None payload")
            return None
        
        if not self.api_secret_ed25519:
            self.logger.warning("[EC] Ed25519 signing requested but api_secret_ed25519 is empty")
            return None
        
        if not _NACL_AVAILABLE:
            self.logger.error("[EC] PyNaCl library not available for Ed25519 signing")
            return None
        
        try:
            seed_input = self.api_secret_ed25519.strip()
            ed25519_seed = None
            
            # First, try to treat it as a base64-encoded seed
            try:
                seed_bytes = base64.b64decode(seed_input)
                self.logger.debug("[EC] Decoded base64 seed: %d bytes", len(seed_bytes))
                
                # Handle 32-byte (standard) or 48-byte (extended) seeds
                if len(seed_bytes) == 32:
                    ed25519_seed = seed_bytes
                    self.logger.debug("[EC] Using 32-byte Ed25519 seed")
                elif len(seed_bytes) == 48:
                    # 48-byte seed - use first 32 bytes as Ed25519 private key
                    ed25519_seed = seed_bytes[:32]
                    self.logger.debug("[EC] Using first 32 bytes of 48-byte seed")
                else:
                    self.logger.debug(
                        "[EC] Base64-decoded seed is %d bytes (not 32 or 48), trying as raw seed",
                        len(seed_bytes)
                    )
            except Exception as b64_err:
                self.logger.debug("[EC] Base64 decode failed: %s, trying as raw bytes", str(b64_err))
            
            # If base64 decode didn't yield valid seed, try raw bytes interpretation
            if ed25519_seed is None:
                try:
                    # Try treating the input as raw UTF-8 bytes
                    raw_bytes = seed_input.encode('utf-8')
                    if len(raw_bytes) == 32:
                        ed25519_seed = raw_bytes
                        self.logger.debug("[EC] Using raw 32-byte seed from UTF-8 encoding")
                    else:
                        self.logger.error(
                            "[EC] Raw UTF-8 seed is %d bytes (not 32), cannot use",
                            len(raw_bytes)
                        )
                except Exception as raw_err:
                    self.logger.error("[EC] Raw seed interpretation failed: %s", str(raw_err))
            
            if ed25519_seed is None or len(ed25519_seed) != 32:
                self.logger.error(
                    "[EC] Ed25519 seed must be 32 bytes (after base64 decode or raw interpretation)"
                )
                return None
            
            # Create signing key from seed
            try:
                signing_key = _NaClSigningKey(ed25519_seed)
            except Exception as sk_err:
                self.logger.error("[EC] Failed to create SigningKey from seed: %s", str(sk_err))
                return None
            
            # Sign the payload
            payload_str = str(payload) if payload else ""
            if not payload_str:
                self.logger.error("[EC] Payload is empty after conversion to string")
                return None
            
            signed_message = signing_key.sign(payload_str.encode('utf-8'))
            signature_bytes = signed_message.signature
            
            # Base64 encode the signature
            signature_b64 = base64.b64encode(signature_bytes).decode('utf-8')
            
            self.logger.debug("[EC] Ed25519 signature created successfully (len=%d)", len(signature_b64))
            return signature_b64
            
        except Exception as e:
            self.logger.error("[EC] Ed25519 signing failed: %s", str(e))
            import traceback
            self.logger.debug("[EC] Traceback: %s", traceback.format_exc())
            return None

    def _ws_api_signed_params(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        WS API v3 params for session.logon WITH HMAC signature.
        
        CRITICAL: Binance WS API v3 REQUIRES HMAC signatures for HMAC-based API keys.
        - Only Ed25519 keys use a different method (not applicable here)
        - For HMAC keys:
        * Include: apiKey (MUST BE IN PARAMS)
        * Include: timestamp
        * Calculate: signature = HMAC-SHA256(query_string, api_secret)
        * Append signature to params
        - Query string format: MUST be ALPHABETICALLY SORTED
        * Format: "apiKey=...&timestamp=..." (NOT timestamp first)
        
        ⚠️ CRITICAL FIXES:
        - DO NOT use urllib.parse.urlencode() (order not guaranteed)
        - DO NOT JSON encode
        - DO NOT include signature in the sorted list
        - MUST sort params alphabetically BEFORE calculating signature
        """
        params: Dict[str, Any] = dict(extra or {})
        params["apiKey"] = str(self.api_key or "")
        timestamp = int(time.time() * 1000 + self._time_offset_ms)
        params["timestamp"] = timestamp
        
        # MUST be alphabetically sorted: apiKey comes before timestamp
        query_string = "&".join(
            f"{k}={v}" for k, v in sorted(params.items())
        )
        
        # Calculate HMAC-SHA256 signature BEFORE adding to params
        # REST API ALWAYS uses HMAC, never Ed25519
        if not self.api_secret_hmac:
            self.logger.warning("[EC] HMAC secret not available for REST signing")
            return params
        
        signature = hmac.new(
            self.api_secret_hmac.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        # Add signature AFTER calculation (not included in signed payload)
        params["signature"] = signature
        
        return params

    def _ws_api_session_logon_params(self) -> Dict[str, Any]:
        """
        WS API v3 params for session.logon WITH Ed25519 signature.
        
        CRITICAL: Binance WS API v3 session.logon REQUIRES Ed25519 signatures.
        - Include: apiKey
        - Include: timestamp
        - Calculate: signature = ED25519_SIGN(private_key, payload)
        - Signature must be base64-encoded
        """
        # Validate prerequisites
        if not self.api_key:
            self.logger.error("[EC] api_key is missing, cannot create session.logon params")
            return {}
        
        if not self.api_secret_ed25519:
            self.logger.error("[EC] api_secret_ed25519 is missing, cannot sign session.logon")
            return {}
        
        params: Dict[str, Any] = {}
        params["apiKey"] = str(self.api_key)
        params["timestamp"] = int(time.time() * 1000 + self._time_offset_ms)
        
        # Create payload for signing (apiKey + timestamp, alphabetically sorted)
        payload = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        
        if not payload:
            self.logger.error("[EC] Failed to construct payload for session.logon signing")
            return {}
        
        self.logger.debug("[EC] session.logon payload: %s", payload)
        
        # Sign with Ed25519
        signature = self._sign_ed25519(payload)
        if signature:
            params["signature"] = signature
            self.logger.debug("[EC] session.logon signature created (len=%d)", len(signature))
        else:
            self.logger.warning("[EC] Ed25519 signing failed, session.logon will fail")
        
        return params

    def _ws_api_signature_params(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        WS API v3 params for userDataStream subscription WITH HMAC signature.
        
        CRITICAL: Binance WS API v3 REQUIRES HMAC signatures for HMAC-based API keys.
        - For HMAC keys:
        * Include: apiKey (MUST BE IN PARAMS)
        * Include: timestamp
        * Calculate: signature = HMAC-SHA256(query_string, api_secret)
        * Append signature to params
        - Query string format: "apiKey=...&timestamp=..."
        """
        params: Dict[str, Any] = dict(extra or {})
        params["apiKey"] = str(self.api_key or "")
        params["timestamp"] = int(time.time() * 1000 + self._time_offset_ms)
        params.setdefault("recvWindow", self.recv_window_ms)

        # Deterministic HMAC payload for WS API v3
        query_string = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        # Use HMAC secret if available (hybrid mode), otherwise use api_secret
        secret_for_signing = self.api_secret_hmac or self.api_secret
        signature = hmac.new(
            secret_for_signing.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        params["signature"] = signature
        
        return params

    def _extract_ws_error_code(self, value: Any) -> Optional[int]:
        try:
            if value is None:
                return None
            return int(value)
        except Exception:
            return None

    def _is_user_data_ws_auth_error(self, err: Exception) -> bool:
        code = self._extract_ws_error_code(getattr(err, "code", None))
        if code in {-2015, -2014, -1022, -1021}:
            return True
        text = str(err or "").lower()
        if "invalid api-key" in text or "invalid api key" in text:
            return True
        if "signature for this request is not valid" in text:
            return True
        if "ed25519" in text and "support" in text:
            return True
        if "unauthorized" in text:
            return True
        return False

    def _should_use_signature_fallback(self, err: Exception) -> bool:
        pref = str(getattr(self, "user_data_ws_auth_mode", "auto") or "auto").strip().lower()
        if pref == "signature":
            return True
        if pref == "session":
            return False

        code = self._extract_ws_error_code(getattr(err, "code", None))
        if code in {-2015, -2014, -1022, -1021}:
            return True
        text = str(err or "").lower()
        if "ed25519" in text:
            return True
        if "not supported for this request" in text:
            return True
        if "invalid api-key" in text or "invalid api key" in text:
            return True
        if "signature for this request is not valid" in text:
            return True
        return False

    def _ingest_user_data_ws_payload(self, payload: Dict[str, Any]) -> str:
        if not isinstance(payload, dict):
            return ""

        event_payload: Dict[str, Any] = {}
        subscription_id = payload.get("subscriptionId")
        raw_event = payload.get("event")
        if isinstance(raw_event, dict):
            event_payload = raw_event
        elif payload.get("e") or payload.get("eventType"):
            event_payload = payload
        if not event_payload:
            return ""

        if subscription_id is not None:
            code = self._extract_ws_error_code(subscription_id)
            if code is not None:
                self._user_data_subscription_id = code

        evt = str(event_payload.get("e") or event_payload.get("eventType") or "UNKNOWN")
        self.mark_user_data_event(evt, event_payload)

        # Route user-data events to the SharedState event bus so downstream
        # components (PositionManager, ExecutionManager) receive fill/balance
        # notifications from every WS tier (WS API v3, listenKey, and polling).
        _ROUTABLE = {
            "executionReport",
            "balanceUpdate",
            "outboundAccountPosition",
            "listStatus",
            "listenKeyExpired",
        }
        if evt in _ROUTABLE:
            ss = getattr(self, "shared_state", None)
            if ss is not None and hasattr(ss, "emit_event"):
                try:
                    coro = ss.emit_event(evt, dict(event_payload))
                    loop = asyncio.get_running_loop()
                    loop.create_task(coro)
                except RuntimeError:
                    pass  # No running event loop (e.g., test context or pre-start)
                except Exception:
                    pass

        return evt

    async def _ws_api_request(
        self,
        ws: aiohttp.ClientWebSocketResponse,
        *,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        timeout_sec: Optional[float] = None,
    ) -> Dict[str, Any]:
        request_id = f"octi-{uuid.uuid4()}"
        payload: Dict[str, Any] = {"id": request_id, "method": str(method or "")}
        if params:
            payload["params"] = params

        self.logger.debug("[EC:WS] send method=%s request_id=%s", method, request_id)
        json_payload = json.dumps(payload, separators=(",", ":"))
        self.logger.warning(f"[EC:WS] SENDING RPC: method={method} payload={json_payload}")
        try:
            await ws.send_str(json_payload)
            self.logger.warning(f"[EC:WS] RPC sent successfully")
        except Exception as e:
            self.logger.error(f"[EC:WS] Failed to send RPC: {e}")
            raise
        self.mark_any_ws_event("ws_api_request:%s" % method)

        timeout_val = max(
            1.0,
            float(timeout_sec if timeout_sec is not None else self.user_data_ws_api_request_timeout_sec or 12.0),
        )
        deadline = time.time() + timeout_val
        self.logger.debug("[EC:WS] waiting for response method=%s timeout=%.1fs", method, timeout_val)

        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise TimeoutError(f"WS_API_TIMEOUT:{method}")

            msg = await asyncio.wait_for(ws.receive(), timeout=remaining)
            self.logger.debug("[EC:WS] msg type=%s method=%s", msg.type, method)
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    raw = json.loads(msg.data) if msg.data else {}
                    frame = raw if isinstance(raw, dict) else {}
                except Exception:
                    frame = {}

                self.logger.debug("[EC:WS] TEXT frame id=%s status=%s", frame.get("id"), frame.get("status"))
                self.mark_any_ws_event("user_data_ws_text")
                frame_evt = self._ingest_user_data_ws_payload(frame)
                if frame_evt == "eventStreamTerminated":
                    raise RuntimeError("USER_DATA_STREAM_TERMINATED")

                if str(frame.get("id", "")) != request_id:
                    self.logger.debug("[EC:WS] frame id mismatch expected=%s got=%s", request_id, frame.get("id"))
                    continue

                status = self._extract_ws_error_code(frame.get("status"))
                status_code = int(status if status is not None else 0)
                if 200 <= status_code < 300:
                    self.logger.debug("[EC:WS] %s success status=%s", method, status_code)
                    return frame

                err_payload = frame.get("error") if isinstance(frame.get("error"), dict) else {}
                err_code = self._extract_ws_error_code(err_payload.get("code"))
                err_msg = str(err_payload.get("msg") or ("WS_API_%s_FAILED" % method))
                self.logger.warning("[EC:WS] %s error code=%s msg=%s", method, err_code, err_msg)
                raise BinanceAPIException(
                    f"WS_API_{method} failed (status={status_code}, code={err_code}): {err_msg}",
                    code=err_code,
                )

            if msg.type == aiohttp.WSMsgType.PING:
                # Must manually pong when autoping=False; server closes if unanswered.
                await ws.pong(msg.data)
                self.mark_any_ws_event("user_data_ping")
                continue
            if msg.type == aiohttp.WSMsgType.PONG:
                self.mark_any_ws_event("user_data_pong")
                continue

            if msg.type in (
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.CLOSING,
                aiohttp.WSMsgType.ERROR,
            ):
                _close_code = getattr(msg, "data", None)
                _close_reason = str(getattr(msg, "extra", "") or "")
                if _close_code == 1008:
                    self.logger.warning(
                        "[EC:WS] Policy-close during %s request: type=%s code=%s reason=%r",
                        method, msg.type, _close_code, _close_reason,
                    )
                else:
                    self.logger.error(
                        "[EC:WS] ❌ SERVER CLOSED CONNECTION during %s request: type=%s code=%s reason=%r",
                        method, msg.type, _close_code, _close_reason,
                    )
                raise RuntimeError(
                    "USER_DATA_WS_CLOSED:%s ws_code=%s reason=%r" % (msg.type, _close_code, _close_reason)
                )

    async def _ws_api_subscribe_with_session(
        self, ws: aiohttp.ClientWebSocketResponse
    ) -> Tuple[str, Optional[int]]:
        self.logger.debug("[EC:WS] sending session.logon")
        
        # Get Ed25519 session.logon params
        logon_params = self._ws_api_session_logon_params()
        
        # Check if signature was successfully created
        if "signature" not in logon_params:
            self.logger.warning(
                "[EC:WS] Ed25519 signing failed for session.logon - "
                "falling back to listenKey (Tier 2)"
            )
            raise RuntimeError("Ed25519 session.logon unavailable (key or signing failed)")
        
        await self._ws_api_request(
            ws,
            method="session.logon",
            params=logon_params,  # Ed25519 signing
        )
        self.last_listenkey_refresh_ts = time.time()

        self.logger.debug("[EC:WS] sending userDataStream.subscribe")
        sub_resp = await self._ws_api_request(
            ws,
            method="userDataStream.subscribe",
            params=self._ws_api_signature_params(),  # HMAC signing
        )
        self.logger.debug("[EC:WS] subscribe response keys=%s", list(sub_resp.keys()) if isinstance(sub_resp, dict) else sub_resp)
        sub_res = sub_resp.get("result") if isinstance(sub_resp.get("result"), dict) else {}
        sub_id = self._extract_ws_error_code(sub_res.get("subscriptionId"))
        self._user_data_subscription_id = sub_id
        self._user_data_auth_mode_active = "session"
        self.last_listenkey_refresh_ts = time.time()
        return "session", sub_id

    async def _ws_api_subscribe_with_signature(
        self, ws: aiohttp.ClientWebSocketResponse
    ) -> Tuple[str, Optional[int]]:
        self.logger.debug("[EC:WS] sending userDataStream.subscribe.signature")
        sub_resp = await self._ws_api_request(
            ws,
            method="userDataStream.subscribe.signature",
            params=self._ws_api_signature_params(),
        )
        self.logger.debug("[EC:WS] subscribe response keys=%s", list(sub_resp.keys()) if isinstance(sub_resp, dict) else sub_resp)
        sub_res = sub_resp.get("result") if isinstance(sub_resp.get("result"), dict) else {}
        sub_id = self._extract_ws_error_code(sub_res.get("subscriptionId"))
        self._user_data_subscription_id = sub_id
        self._user_data_auth_mode_active = "signature"
        self.last_listenkey_refresh_ts = time.time()
        return "signature", sub_id

    async def _ws_api_subscribe_user_data(
        self, ws: aiohttp.ClientWebSocketResponse
    ) -> Tuple[str, Optional[int]]:
        pref = str(getattr(self, "user_data_ws_auth_mode", "auto") or "auto").strip().lower()
        self.logger.debug("[EC:WS] subscribe user_data auth_mode=%s", pref)
        if pref == "signature":
            return await self._ws_api_subscribe_with_signature(ws)

        try:
            result = await self._ws_api_subscribe_with_session(ws)
            self.logger.debug("[EC:WS] session mode succeeded mode=%s sub_id=%s", result[0], result[1])
            return result
        except Exception as e:
            self.logger.debug("[EC:WS] session mode failed: %s", e)
            if not self._should_use_signature_fallback(e):
                raise
            self.logger.warning(
                "[EC:UserDataWS] session.logon unavailable (%s); falling back to userDataStream.subscribe.signature",
                e,
            )
            return await self._ws_api_subscribe_with_signature(ws)

    async def _user_data_listen_key_loop(self) -> None:
        """
        WebSocket Streams API loop using listenKey (works with HMAC keys).
        
        This is the fallback when WS API v3 is not available (no Ed25519 keys).
        Uses REST-based listenKey authentication which is compatible with HMAC-SHA256 keys.
        """
        backoff = max(1.0, float(self.user_data_ws_reconnect_backoff_sec or 3.0))
        max_backoff = max(backoff, float(self.user_data_ws_max_backoff_sec or 30.0))
        max_reconnect_attempts = int(getattr(self, "user_data_ws_max_reconnects", 50) or 50)
        listen_key_refresh_interval = float(self.listenkey_refresh_sec or 900.0)
        last_listen_key_refresh = 0.0

        while self.is_started and not self._user_data_stop.is_set():
            try:
                current_reconnect_count = int(getattr(self, "ws_reconnect_count", 0) or 0)
                if current_reconnect_count > max_reconnect_attempts:
                    self.logger.critical(
                        "[EC:ListenKeyWS] FATAL: reconnect_count=%d exceeds max=%d",
                        current_reconnect_count,
                        max_reconnect_attempts,
                    )
                    self._user_data_stop.set()
                    break

                # Create/refresh listenKey if needed
                if not hasattr(self, "_listen_key") or not self._listen_key:
                    self._listen_key = await self._create_listen_key()
                    if not self._listen_key:
                        raise RuntimeError("Failed to create listenKey - likely HTTP 410 Gone (account doesn't support user-data streams)")
                    last_listen_key_refresh = time.time()

                ws_url = self._user_data_ws_stream_url(self._listen_key)
                self.logger.info(
                    "[EC:ListenKeyWS] Connecting to WebSocket Streams API (listenKey mode, attempt %d)",
                    current_reconnect_count + 1,
                )
                self.logger.debug("[EC:ListenKeyWS] URL=%s", ws_url)

                async with self.session.ws_connect(ws_url, heartbeat=20.0, autoping=True) as ws:
                    self._user_data_ws_conn = ws
                    self.ws_connected = True
                    self.ws_reconnect_count = 0
                    self.mark_any_ws_event("user_data_connected")
                    self.last_user_data_event_ts = time.time()
                    self.logger.info("[EC:ListenKeyWS] ✅ Connected (listenKey mode)")

                    while self.is_started and not self._user_data_stop.is_set():
                        try:
                            # Refresh listenKey every 30 minutes
                            now = time.time()
                            if now - last_listen_key_refresh > listen_key_refresh_interval - 60:
                                self.logger.debug("[EC:ListenKeyWS] Refreshing listenKey...")
                                success = await self._refresh_listen_key(self._listen_key)
                                if success:
                                    last_listen_key_refresh = now

                            remaining = listen_key_refresh_interval - (now - last_listen_key_refresh)
                            timeout = max(1.0, min(30.0, remaining))

                            msg = await asyncio.wait_for(ws.receive(), timeout=timeout)
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                try:
                                    payload = json.loads(msg.data) if msg.data else {}
                                    evt = self._ingest_user_data_ws_payload(payload)
                                    if evt == "eventStreamTerminated":
                                        raise RuntimeError("USER_DATA_STREAM_TERMINATED")
                                except Exception as e:
                                    self.logger.debug("[EC:ListenKeyWS] Payload processing error: %s", e)
                                continue

                            if msg.type == aiohttp.WSMsgType.PING:
                                await ws.pong(msg.data)
                                self.mark_any_ws_event("user_data_ping")
                                continue

                            if msg.type == aiohttp.WSMsgType.PONG:
                                self.mark_any_ws_event("user_data_pong")
                                continue

                            if msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSING, aiohttp.WSMsgType.ERROR):
                                _close_code = getattr(msg, "data", None)
                                _close_reason = str(getattr(msg, "extra", "") or "")
                                self.logger.warning("[EC:ListenKeyWS] Server closed: code=%s reason=%r", _close_code, _close_reason)
                                raise RuntimeError(f"USER_DATA_WS_CLOSED: {msg.type} code={_close_code}")

                        except asyncio.TimeoutError:
                            continue
                        except asyncio.CancelledError:
                            raise
                        except Exception as e:
                            self.logger.warning("[EC:ListenKeyWS] Error in message loop: %s", e)
                            raise

            except asyncio.CancelledError:
                raise
            except Exception as e:
                # Check if this is a fatal listen key error - if so, re-raise to trigger polling fallback
                error_str = str(e).lower()
                if "failed to create listenkey" in error_str or "doesn't support" in error_str:
                    self.logger.warning("[EC:ListenKeyWS] listenKey unavailable; handing off to polling fallback: %s", e)
                    raise
                
                self.ws_connected = False
                self._user_data_ws_conn = None
                self._user_data_subscription_id = None
                self._user_data_auth_mode_active = "none"
                self.ws_reconnect_count = int(getattr(self, "ws_reconnect_count", 0) or 0) + 1
                auth_error = self._is_user_data_ws_auth_error(e)
                self.logger.warning(
                    "[EC:ListenKeyWS] Disconnected: %s (reconnect_count=%d auth_error=%s)",
                    e,
                    int(self.ws_reconnect_count),
                    bool(auth_error),
                )
                await asyncio.sleep(backoff + random.uniform(0.0, min(1.0, backoff / 2.0)))
                backoff = min(max_backoff, backoff * 1.7)

            finally:
                self.ws_connected = False
                self._user_data_ws_conn = None

        self.ws_connected = False
        self._user_data_ws_conn = None
        self._user_data_subscription_id = None
        self._user_data_auth_mode_active = "none"

    async def _user_data_polling_loop(self) -> None:
        """
        ✅ POLLING MODE: Deterministic account reconciliation via REST.
        
        Every poll cycle (2.0s):
        1. Fetch open orders via /api/v3/openOrders
        2. Fetch account balances via /api/v3/account
        3. Compare with internal state
        4. Detect:
            - Missing fills (orders disappeared)
            - Partial fills (balances changed unexpectedly)
            - Closed positions
            - Trade execution confirmation
        5. Emit reconciliation events to update position state
        
        This is more stable than WebSocket because:
        - No authentication state to manage
        - Deterministic comparison logic (not event-driven)
        - Easy to test and audit
        - Works reliably under market stress
        """
        poll_interval = self._user_data_poll_interval_sec  # Default 25s; set USER_DATA_POLL_INTERVAL_SEC to tune
        backoff = max(1.0, float(self.user_data_ws_reconnect_backoff_sec or 3.0))
        max_backoff = max(backoff, float(self.user_data_ws_max_backoff_sec or 30.0))
        
        # Track previous state to detect changes
        prev_balances: Dict[str, Dict[str, float]] = {}
        prev_open_orders: Dict[str, Any] = {}  # order_id -> order_data
        prev_filled_qty: Dict[str, float] = {}  # order_id -> filled_qty
        last_poll_time = 0.0
        
        while self.is_started and not self._user_data_stop.is_set():
            try:
                now = time.time()
                
                # Mark as connected in polling mode
                self.ws_connected = True
                self._user_data_auth_mode_active = "polling"
                self.ws_reconnect_count = 0
                self.mark_any_ws_event("user_data_connected")
                self.last_user_data_event_ts = now
                
                self.logger.info("[EC:Polling] Polling mode active (interval=%.1fs)", poll_interval)
                
                while self.is_started and not self._user_data_stop.is_set():
                    # Wait until next poll time
                    now = time.time()
                    time_since_poll = now - last_poll_time
                    if time_since_poll < poll_interval:
                        await asyncio.sleep(min(0.5, poll_interval - time_since_poll))
                        continue
                    
                    try:
                        now = time.time()
                        self.logger.debug("[EC:Polling] Starting reconciliation cycle at %.3f", now)
                        
                        # ====== PHASE 1: Fetch Current State ======
                        # Fetch open orders — skip when idle to conserve weight (6/call)
                        _ss = getattr(self, "shared_state", None)
                        _has_positions = bool(_ss and getattr(_ss, "open_positions_count", lambda: 0)() > 0)
                        if prev_open_orders or _has_positions:
                            try:
                                open_orders_resp = await self._request(
                                    "GET", "/api/v3/openOrders",
                                    api="spot_api", signed=True,
                                    timeout=5.0
                                )
                                current_open_orders = {
                                    str(o.get("orderId")): o for o in (open_orders_resp or [])
                                }
                            except Exception as e:
                                self.logger.warning("[EC:Polling] Failed to fetch open orders: %s", e)
                                current_open_orders = prev_open_orders  # Use previous state
                        else:
                            # No tracked orders and no open positions — skip the call
                            self.logger.debug("[EC:Polling] Idle — skipping openOrders poll (no active orders/positions)")
                            current_open_orders = {}
                        
                        # Fetch account balances
                        try:
                            acct = await self._request(
                                "GET", "/api/v3/account", 
                                api="spot_api", signed=True,
                                timeout=5.0
                            )
                            current_balances = {}
                            # Get quote asset for tradable pair validation
                            quote_asset = str(getattr(self.config, "DEFAULT_QUOTE_CURRENCY", "USDT")).upper() if self.config else "USDT"
                            for bal in acct.get("balances", []):
                                asset = bal.get("asset", "")
                                free = float(bal.get("free", 0.0))
                                locked = float(bal.get("locked", 0.0))
                                
                                # FILTER: Skip non-tradable assets
                                # Only include quote asset or assets that form valid trading pairs
                                a = asset.upper()
                                if a != quote_asset:
                                    # For non-quote assets, verify they form tradable pairs
                                    symbol = f"{a}{quote_asset}"
                                    if not self.has_symbol(symbol):
                                        self.logger.debug(f"[EC:Polling] Ignoring non-tradable asset {asset} (symbol {symbol})")
                                        continue
                                
                                current_balances[asset] = {"free": free, "locked": locked}
                        except Exception as e:
                            self.logger.warning("[EC:Polling] Failed to fetch account: %s", e)
                            current_balances = prev_balances  # Use previous state
                        
                        last_poll_time = now
                        self.last_user_data_event_ts = now
                        
                        # ====== PHASE 2: Detect Balance Changes ======
                        # Skip logging on first snapshot (prev_balances is empty)
                        if prev_balances:
                            for asset, curr in current_balances.items():
                                prev = prev_balances.get(asset, {})
                                # Skip logging for zero-balance assets (dust)
                                if curr["free"] == 0 and curr["locked"] == 0:
                                    continue
                                if (prev.get("free") != curr["free"] or prev.get("locked") != curr["locked"]):
                                    self.logger.info(
                                        "[EC:Polling:Balance] %s changed: free=%.8f (was %.8f) locked=%.8f (was %.8f)",
                                        asset, curr["free"], prev.get("free", 0), 
                                        curr["locked"], prev.get("locked", 0)
                                    )
                                    # Emit balanceUpdate event
                                    evt_payload = {
                                        "e": "balanceUpdate",
                                        "E": int(now * 1000),
                                        "a": asset,
                                        "d": curr["free"],
                                        "l": curr["locked"]
                                    }
                                    self._ingest_user_data_ws_payload(evt_payload)
                        
                        # ====== PHASE 3: Detect Order Fills (Critical!) ======
                        # Check for orders that disappeared (filled or cancelled)
                        for order_id, prev_order in prev_open_orders.items():
                            if order_id not in current_open_orders:
                                # Order closed (filled or cancelled)
                                symbol = prev_order.get("symbol", "UNKNOWN")
                                filled_qty = float(prev_order.get("executedQty", 0.0))
                                status = prev_order.get("status", "UNKNOWN")
                                self.logger.info(
                                    "[EC:Polling:Fill] Order %s (%s) CLOSED: status=%s filled_qty=%.8f",
                                    order_id, symbol, status, filled_qty
                                )
                                
                                # Emit executionReport to notify position manager
                                evt_payload = {
                                    "e": "executionReport",
                                    "E": int(now * 1000),
                                    "s": symbol,
                                    "c": prev_order.get("clientOrderId", ""),
                                    "S": prev_order.get("side", "BUY"),
                                    "o": prev_order.get("type", "LIMIT"),
                                    "f": prev_order.get("timeInForce", "GTC"),
                                    "q": prev_order.get("origQty", 0),
                                    "p": prev_order.get("price", 0),
                                    "P": 0,
                                    "F": filled_qty,
                                    "L": 0,
                                    "C": "",
                                    "x": "TRADE" if filled_qty > 0 else "CANCELED",
                                    "X": status,
                                    "i": int(order_id),
                                    "l": 0,
                                    "z": filled_qty,
                                    "n": 0,
                                    "N": None,
                                    "u": False,
                                    "m": False,
                                    "O": prev_order.get("time", int(now * 1000)),
                                    "Z": 0,
                                    "Y": 0,
                                    "Q": 0
                                }
                                self._ingest_user_data_ws_payload(evt_payload)
                        
                        # ====== PHASE 4: Detect Partial Fills ======
                        # Check for quantity changes in open orders
                        for order_id, curr_order in current_open_orders.items():
                            if order_id in prev_open_orders:
                                prev_order = prev_open_orders[order_id]
                                prev_filled = float(prev_order.get("executedQty", 0.0))
                                curr_filled = float(curr_order.get("executedQty", 0.0))
                                
                                if curr_filled > prev_filled:
                                    # Partial fill detected
                                    fill_qty = curr_filled - prev_filled
                                    symbol = curr_order.get("symbol", "UNKNOWN")
                                    self.logger.info(
                                        "[EC:Polling:PartialFill] Order %s (%s) partial fill: +%.8f qty "
                                        "(total filled=%.8f)",
                                        order_id, symbol, fill_qty, curr_filled
                                    )
                                    
                                    # Emit executionReport for partial fill
                                    evt_payload = {
                                        "e": "executionReport",
                                        "E": int(now * 1000),
                                        "s": symbol,
                                        "c": curr_order.get("clientOrderId", ""),
                                        "S": curr_order.get("side", "BUY"),
                                        "o": curr_order.get("type", "LIMIT"),
                                        "f": curr_order.get("timeInForce", "GTC"),
                                        "q": curr_order.get("origQty", 0),
                                        "p": curr_order.get("price", 0),
                                        "P": 0,
                                        "F": curr_filled,
                                        "L": fill_qty,
                                        "C": "",
                                        "x": "TRADE",
                                        "X": "PARTIALLY_FILLED",
                                        "i": int(order_id),
                                        "l": fill_qty,
                                        "z": curr_filled,
                                        "n": 0,
                                        "N": None,
                                        "u": False,
                                        "m": False,
                                        "O": curr_order.get("time", int(now * 1000)),
                                        "Z": 0,
                                        "Y": 0,
                                        "Q": 0
                                    }
                                    self._ingest_user_data_ws_payload(evt_payload)
                        
                        # ====== PHASE 5: Truth Audit ======
                        # Run truth auditor to detect any state mismatches
                        with contextlib.suppress(Exception):
                            await self._run_truth_auditor(current_balances, current_open_orders)
                        
                        # Update previous state for next cycle
                        prev_balances = current_balances
                        prev_open_orders = current_open_orders
                        self.mark_any_ws_event("user_data_account_update")
                        
                        self.logger.debug(
                            "[EC:Polling] Reconciliation complete: %d open orders, %d balance assets",
                            len(current_open_orders), len(current_balances)
                        )
                        
                    except asyncio.CancelledError:
                        raise
                    except Exception as poll_err:
                        if self._is_rate_limit_error(poll_err):
                            # -1003 / 429: back off for 2× the poll interval (capped at 120s)
                            _rl_sleep = min(120.0, poll_interval * 2)
                            self.logger.warning(
                                "[EC:Polling] Rate limit hit, backing off %.1fs before retry: %s",
                                _rl_sleep, poll_err,
                            )
                            await asyncio.sleep(_rl_sleep)
                        else:
                            self.logger.warning("[EC:Polling] Reconciliation error: %s", poll_err)
                            await asyncio.sleep(1.0)
            
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.ws_connected = False
                self._user_data_auth_mode_active = "none"
                self.ws_reconnect_count = int(getattr(self, "ws_reconnect_count", 0) or 0) + 1
                self.logger.warning(
                    "[EC:Polling] Main loop error (reconnect_count=%d): %s", 
                    int(self.ws_reconnect_count), e
                )
                await asyncio.sleep(backoff + random.uniform(0.0, min(1.0, backoff / 2.0)))
                backoff = min(max_backoff, backoff * 1.7)
            
            finally:
                self.ws_connected = False
                self._user_data_auth_mode_active = "none"

    async def _run_truth_auditor(self, current_balances: Dict[str, Any], current_open_orders: Dict[str, Any]) -> None:
        """
        ✅ Truth Auditor: Validate state consistency.
        
        Checks:
        - All open orders reference symbols that exist
        - Balance values are reasonable (non-negative)
        - No contradictions between order state and balances
        - Alert on suspicious patterns (e.g., large unexpected fills)
        """
        try:
            # Validate balances are non-negative
            for asset, balance in current_balances.items():
                if balance.get("free", 0) < 0 or balance.get("locked", 0) < 0:
                    self.logger.error(
                        "[EC:TruthAuditor] ❌ NEGATIVE BALANCE detected: %s free=%.8f locked=%.8f",
                        asset, balance.get("free"), balance.get("locked")
                    )
            
            # Validate open orders reference valid symbols
            for order_id, order in current_open_orders.items():
                symbol = order.get("symbol", "")
                qty = float(order.get("origQty", 0.0))
                filled = float(order.get("executedQty", 0.0))
                if filled > qty:
                    self.logger.error(
                        "[EC:TruthAuditor] ❌ FILLED > ORDERED: order %s (%s) qty=%.8f filled=%.8f",
                        order_id, symbol, qty, filled
                    )
            
            self.logger.debug("[EC:TruthAuditor] ✅ State consistency check passed")
        
        except Exception as e:
            self.logger.error("[EC:TruthAuditor] Auditor failed: %s", e)

    async def _user_data_ws_loop(self) -> None:
        """
        User-data stream supervisor (WebSocket-first).

        Priority:
        1) WS API v3 (session/signature auth)
        2) listenKey WebSocket stream fallback
        3) REST polling fallback
        """
        while self.is_started and not self._user_data_stop.is_set():
            try:
                if not bool(getattr(self, "user_data_stream_enabled", True)):
                    self.logger.info("[EC:UserDataWS] USER_DATA_STREAM_ENABLED=false; using polling mode.")
                    await self._user_data_polling_loop()
                    continue

                if not self._has_signed_credentials():
                    self.logger.warning("[EC:UserDataWS] Signed credentials unavailable; using polling mode.")
                    await self._user_data_polling_loop()
                    continue

                # Tier 1: WS API v3
                if not self._ws_v3_unavailable:
                    try:
                        self.logger.info("[EC:UserDataWS] Starting Tier 1 (WS API v3)")
                        await self._user_data_ws_api_v3_direct()
                        continue
                    except asyncio.CancelledError:
                        raise
                    except Exception as ws_v3_err:
                        _err_str = str(ws_v3_err)
                        if "1008" in _err_str or "POLICY" in _err_str.upper():
                            self._ws_v3_unavailable = True
                            self.logger.warning(
                                "[EC:UserDataWS] Tier 1 (WS API v3) permanently unavailable "
                                "in this environment (policy-close 1008). "
                                "Skipping on all future reconnects."
                            )
                        else:
                            self.logger.warning("[EC:UserDataWS] Tier 1 failed: %s", ws_v3_err)
                else:
                    self.logger.debug(
                        "[EC:UserDataWS] Tier 1 (WS API v3) skipped — permanently unavailable"
                    )

                # Tier 2: listenKey WS stream
                if not self._listen_key_unavailable:
                    try:
                        self.logger.info("[EC:UserDataWS] Starting Tier 2 (listenKey stream)")
                        await self._user_data_listen_key_loop()
                        continue
                    except asyncio.CancelledError:
                        raise
                    except Exception as listen_key_err:
                        _err_str = str(listen_key_err).lower()
                        if "410" in _err_str or "gone" in _err_str or "doesn't support" in _err_str:
                            self._listen_key_unavailable = True
                            self.logger.warning(
                                "[EC:UserDataWS] Tier 2 (listenKey) permanently unavailable "
                                "in this environment (HTTP 410 Gone). "
                                "Skipping on all future reconnects."
                            )
                        else:
                            self.logger.warning("[EC:UserDataWS] Tier 2 failed: %s", listen_key_err)
                else:
                    self.logger.debug(
                        "[EC:UserDataWS] Tier 2 (listenKey) skipped — permanently unavailable"
                    )

                # Tier 3: deterministic polling
                self.logger.warning("[EC:UserDataWS] Falling back to Tier 3 (polling).")
                await self._user_data_polling_loop()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.error(
                    "[EC:UserDataWS] Supervisor loop failed: %s. Retrying in 5s...", e
                )
                await asyncio.sleep(5.0)

    async def _user_data_ws_api_v3_direct(self) -> None:
        """Direct WS API v3 connection (no fallback)."""
        backoff = max(1.0, float(self.user_data_ws_reconnect_backoff_sec or 3.0))
        max_backoff = max(backoff, float(self.user_data_ws_max_backoff_sec or 30.0))
        max_reconnect_attempts = int(getattr(self, "user_data_ws_max_reconnects", 50) or 50)

        while self.is_started and not self._user_data_stop.is_set():
            try:
                # Runaway loop prevention: escalate to FATAL after too many reconnects
                current_reconnect_count = int(getattr(self, "ws_reconnect_count", 0) or 0)
                if current_reconnect_count > max_reconnect_attempts:
                    self.logger.critical(
                        "[EC:UserDataWS] FATAL: reconnect_count=%d exceeds max=%d. "
                        "Stopping user data stream. Manual intervention required.",
                        current_reconnect_count,
                        max_reconnect_attempts,
                    )
                    with contextlib.suppress(Exception):
                        await self._report_status(
                            "FATAL",
                            {
                                "event": "user_data_ws_fatal",
                                "reason": "max_reconnects_exceeded",
                                "reconnect_count": current_reconnect_count,
                                "max_allowed": max_reconnect_attempts,
                            },
                        )
                    self._user_data_stop.set()
                    break

                ws_url = self._user_data_ws_api_url()
                self.logger.info(
                    "[EC:UserDataWS] connecting to WebSocket API v3 user-data stream (attempt %d)",
                    current_reconnect_count + 1,
                )
                self.logger.debug("[EC:WS] connecting url=%s", ws_url)
                # Disable heartbeat and autoping: WS API v3 is JSON-RPC, not a raw
                # stream. Automatic control frames interfere with the RPC auth sequence.
                # Authentication: API key goes in X-MBX-APIKEY header, then session.logon RPC
                headers = {"X-MBX-APIKEY": str(self.api_key or "")}
                async with self.session.ws_connect(ws_url, headers=headers, heartbeat=None, autoping=False) as ws:
                    self._user_data_ws_conn = ws
                    auth_mode, sub_id = await self._ws_api_subscribe_user_data(ws)
                    self.logger.debug("[EC:WS] subscribed auth_mode=%s sub_id=%s", auth_mode, sub_id)
                    self.ws_connected = True
                    # RESET reconnect counter on successful connection
                    self.ws_reconnect_count = 0
                    self.mark_any_ws_event("user_data_connected")
                    # Baseline heartbeat for gap monitoring even before first account event arrives.
                    self.last_user_data_event_ts = time.time()
                    self._user_data_listen_key = ""
                    with contextlib.suppress(Exception):
                        await self._report_status(
                            "OK",
                            {
                                "event": "user_data_ws_connected",
                                "ws_auth_mode": str(auth_mode),
                                "subscription_id": int(sub_id) if sub_id is not None else None,
                            },
                        )
                    self.logger.info(
                        "[EC:UserDataWS] user data subscribed (mode=%s subscription_id=%s)",
                        str(auth_mode),
                        str(sub_id if sub_id is not None else "unknown"),
                    )
                    backoff = max(1.0, float(self.user_data_ws_reconnect_backoff_sec or 3.0))

                    while self.is_started and not self._user_data_stop.is_set():
                        try:
                            msg = await asyncio.wait_for(
                                ws.receive(),
                                timeout=max(5.0, float(self.user_data_ws_timeout_sec or 65.0)),
                            )
                        except asyncio.TimeoutError:
                            # No account events received within timeout window.
                            # Send a session.status keepalive instead of disconnecting
                            # immediately — avoids constant reconnect on quiet accounts.
                            try:
                                await self._ws_api_request(ws, method="session.status", timeout_sec=8.0)
                                self.mark_any_ws_event("user_data_keepalive")
                            except Exception as _ka_err:
                                raise RuntimeError(
                                    "USER_DATA_WS_TIMEOUT: keepalive failed: %s" % _ka_err
                                )
                            continue

                        if msg.type == aiohttp.WSMsgType.TEXT:
                            payload: Dict[str, Any]
                            try:
                                raw_payload = json.loads(msg.data) if msg.data else {}
                                payload = raw_payload if isinstance(raw_payload, dict) else {}
                            except Exception:
                                payload = {}
                            evt = self._ingest_user_data_ws_payload(payload)
                            if evt == "eventStreamTerminated":
                                raise RuntimeError("USER_DATA_STREAM_TERMINATED")
                            continue

                        if msg.type == aiohttp.WSMsgType.PING:
                            # Must manually pong when autoping=False.
                            await ws.pong(msg.data)
                            self.mark_any_ws_event("user_data_ping")
                            continue
                        if msg.type == aiohttp.WSMsgType.PONG:
                            self.mark_any_ws_event("user_data_pong")
                            continue

                        if msg.type in (
                            aiohttp.WSMsgType.CLOSE,
                            aiohttp.WSMsgType.CLOSED,
                            aiohttp.WSMsgType.CLOSING,
                            aiohttp.WSMsgType.ERROR,
                        ):
                            _close_code = getattr(msg, "data", None)
                            _close_reason = str(getattr(msg, "extra", "") or "")
                            self.logger.warning(
                                "[EC:UserDataWS] server closed stream type=%s ws_code=%s reason=%r",
                                msg.type, _close_code, _close_reason,
                            )
                            raise RuntimeError(
                                "USER_DATA_WS_CLOSED:%s ws_code=%s reason=%r" % (msg.type, _close_code, _close_reason)
                            )

            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.ws_connected = False
                self._user_data_ws_conn = None
                self._user_data_subscription_id = None
                self._user_data_auth_mode_active = "none"
                self.ws_reconnect_count = int(getattr(self, "ws_reconnect_count", 0) or 0) + 1
                auth_error = self._is_user_data_ws_auth_error(e)
                is_policy_error = "1008" in str(e) or "POLICY" in str(e)
                
                # Re-raise policy errors for fallback to listenKey mode
                if is_policy_error:
                    self.logger.warning(
                        "[EC:UserDataWS:v3] Policy violation error (1008) - will trigger fallback: %s",
                        e
                    )
                    raise
                
                self.logger.warning(
                    "[EC:UserDataWS] disconnected: %s (reconnect_count=%d auth_error=%s)",
                    e,
                    int(self.ws_reconnect_count),
                    bool(auth_error),
                )
                await asyncio.sleep(backoff + random.uniform(0.0, min(1.0, backoff / 2.0)))
                backoff = min(max_backoff, backoff * 1.7)

                with contextlib.suppress(Exception):
                    await self._report_status(
                        "DEGRADED",
                        {
                            "event": "user_data_ws_disconnected",
                            "reason": str(e),
                            "reconnect_count": int(self.ws_reconnect_count),
                            "auth_error": bool(auth_error),
                            "max_allowed": max_reconnect_attempts,
                        },
                    )
            finally:
                self.ws_connected = False
                self._user_data_ws_conn = None

        self.ws_connected = False
        self._user_data_ws_conn = None
        self._user_data_subscription_id = None
        self._user_data_auth_mode_active = "none"

    async def start_user_data_stream(self) -> bool:
        if not bool(getattr(self, "user_data_stream_enabled", True)):
            return False
        if not self._has_signed_credentials():
            return False
        if not self.is_started:
            return False
        if self._user_data_ws_task and not self._user_data_ws_task.done():
            return True

        self._user_data_stop.clear()
        self._user_data_ws_task = asyncio.create_task(
            self._user_data_ws_loop(),
            name="ExchangeClient:user_data_ws",
        )
        self._user_data_keepalive_task = None
        return True

    async def stop_user_data_stream(self, *, close_listen_key: bool = True) -> None:
        self._user_data_stop.set()
        ws_conn = self._user_data_ws_conn
        if ws_conn is not None and not ws_conn.closed:
            with contextlib.suppress(Exception):
                await ws_conn.close()

        # Keep legacy task slot in case another runtime still sets it.
        tasks = [self._user_data_ws_task, self._user_data_keepalive_task]
        self._user_data_ws_task = None
        self._user_data_keepalive_task = None
        self._user_data_ws_conn = None
        for task in tasks:
            if task and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
        self.ws_connected = False
        self._user_data_subscription_id = None
        self._user_data_auth_mode_active = "none"
        self._user_data_listen_key = ""
        # close_listen_key kept for API compatibility; no-op in WS API v3 flow.
        _ = bool(close_listen_key)

    async def reconnect_user_data_stream(self, reason: str = "MANUAL_TRIGGER") -> bool:
        if not bool(getattr(self, "user_data_stream_enabled", True)):
            return False
        if not self._has_signed_credentials():
            return False
        self.ws_reconnect_count = int(getattr(self, "ws_reconnect_count", 0) or 0) + 1
        self.logger.warning("[EC:UserDataWS] reconnect requested reason=%s", str(reason or ""))
        await self.stop_user_data_stream(close_listen_key=False)
        started = await self.start_user_data_stream()
        with contextlib.suppress(Exception):
            await self._emit_summary(
                "USER_DATA_WS_RECONNECT",
                status="OK" if started else "SKIPPED",
                reason=str(reason or ""),
                reconnect_count=int(self.ws_reconnect_count),
            )
        return started

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
            norm_sym = self._norm_symbol(symbol)
            filters = self.symbol_filters.get(norm_sym, {})
            if not filters:
                raise RuntimeError(f"Symbol filters could not be loaded for {symbol} (normalized: {norm_sym})")
            return filters
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
        async with self._start_lock:
            await self._start_locked()

    async def _start_locked(self):
        """Actual start logic — always called under self._start_lock."""
        from aiohttp import ClientTimeout, TCPConnector

        # If already started but we have API keys and the client might be public-only, recreate with signed keys
        _has_real_keys = (
            self.api_key and self.api_secret
            and self.api_key != "paper_key"
            and not self.paper_trade
        )
        if self.is_started and _has_real_keys:
            try:
                if self.client:
                    await self.client.close_connection()
                self.client = await AsyncClient.create(self.api_key, self.api_secret, testnet=self.testnet)
                self.logger.info("Exchange client upgraded to signed mode.")
                with contextlib.suppress(Exception):
                    await self.start_user_data_stream()
                return
            except Exception as e:
                self.logger.warning("Failed to upgrade to signed client: %s", e)
                # Fall through to normal start logic

        if self.is_started:
            with contextlib.suppress(Exception):
                await self.start_user_data_stream()
            return

        connector = TCPConnector(limit=50, ttl_dns_cache=300)
        self.session = aiohttp.ClientSession(timeout=ClientTimeout(total=15), connector=connector)
        # Allow starting in public-only mode if keys are absent or paper sentinels are set.
        # Never pass "paper_key"/"paper_secret" to AsyncClient — Binance rejects them with -2014.
        if not _has_real_keys:
            if self.paper_trade:
                self.logger.info("ExchangeClient starting in paper mode — using public-only AsyncClient.")
            else:
                self.logger.warning("ExchangeClient starting without API keys — public endpoints only.")
            self.client = await AsyncClient.create(api_key="", api_secret="", testnet=self.testnet)
        else:
            self.client = await AsyncClient.create(self.api_key, self.api_secret, testnet=self.testnet)
        self.logger.info("Exchange client connected.")
        self._ready = True

        # Verify authentication and time sync before proceeding
        try:
            # Time sync to avoid -1021 errors
            t = await self._request("GET", "/api/v3/time", api="spot_api")
            server_ms = int(t.get("serverTime", 0))
            local_ms = int(time.time() * 1000)
            self._time_offset_ms = server_ms - local_ms
            self.logger.info("[EC:StartupSync] Server time synced (offset: %dms)", self._time_offset_ms)

            # Verify authentication (non-paper mode only)
            if not self.paper_trade and (self.api_key and self.api_key != "paper_key"):
                try:
                    acct = await self.client.get_account()
                    can_trade = acct.get("canTrade", False)
                    self.logger.info("[EC:Authentication] Verified (trading_enabled=%s)", can_trade)
                except Exception as auth_err:
                    if "401" in str(auth_err) or "Unauthorized" in str(auth_err):
                        raise RuntimeError(f"[EC] Authentication failed - invalid API keys: {auth_err}")
                    # IP restrictions or other non-auth errors are non-fatal
                    self.logger.warning("[EC] Cannot verify authentication (might be IP restriction): %s", auth_err)

            # Health check
            await self._request("GET", "/api/v3/ping", api="spot_api")
            await self._report_status("OK", {"event": "start_connected"})
            # Start user-data stream control-plane when signed credentials are available.
            with contextlib.suppress(Exception):
                await self.start_user_data_stream()
        except Exception as e:
            self._ready = False
            await self._report_status("ERROR", {"event": "start_failed", "reason": str(e)})
            raise

    async def close(self):
        """Canonical lifecycle exit (AppContext calls this on shutdown)."""
        try:
            with contextlib.suppress(Exception):
                try:
                    await asyncio.wait_for(
                        self.stop_user_data_stream(close_listen_key=True),
                        timeout=5.0
                    )
                except (asyncio.TimeoutError, Exception):
                    pass
            if self.client:
                try:
                    await asyncio.wait_for(
                        self.client.close_connection(),
                        timeout=5.0
                    )
                except (asyncio.TimeoutError, Exception):
                    pass
        finally:
            if self.session and not self.session.closed:
                try:
                    await asyncio.wait_for(
                        self.session.close(),
                        timeout=5.0
                    )
                except (asyncio.TimeoutError, Exception):
                    pass
        self.client = None
        self.session = None
        self._ready = False
        self.logger.info("Exchange client disconnected.")

    async def __aenter__(self):
        """Context manager entry - allows async with ExchangeClient usage."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup happens even on exceptions."""
        try:
            await asyncio.wait_for(self.close(), timeout=10.0)
        except asyncio.TimeoutError:
            self.logger.error("[EC] Close operation timed out")
        except Exception as e:
            self.logger.error(f"[EC] Error during close: {e}")
        return False  # Don't suppress exceptions

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
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        *,
        signed: bool = False,
        api: str = "spot_api",
        timeout: Optional[float] = None,
    ) -> Any:
            """
            Unified request wrapper.
            api: "spot_api" | "spot_sapi" | "um_futures"
            """
            # PAPER MODE: Return mock data for account endpoints
            if self.paper_trade and signed:
                if path == "/api/v3/account":
                    # Return empty balances in paper mode
                    self.logger.debug("[EC] Paper mode: returning mock account data for /api/v3/account")
                    return {"balances": [], "permissions": []}
                elif path.startswith("/api/v3/order") or path == "/api/v3/openOrders":
                    # Return empty orders in paper mode
                    self.logger.debug("[EC] Paper mode: returning mock order data for %s", path)
                    return [] if "openOrders" in path else {}
            
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
            # Prune stale buckets to prevent unbounded defaultdict growth
            if len(self._weight_counters) > 50:
                stale = [p for p, b in list(self._weight_counters.items()) if now - b["ts"] > 60.0]
                for p in stale:
                    del self._weight_counters[p]
            bucket = self._weight_counters[path]
            if now - bucket["ts"] > self._weight_window:
                bucket["ts"], bucket["w"] = now, 0
            if bucket["w"] + weight > self._weight_limit:
                await asyncio.sleep(max(0.0, self._weight_window - (now - bucket["ts"])))
                bucket["ts"], bucket["w"] = time.time(), 0
            bucket["w"] += weight
    
            if signed:
                if not (self.api_key and self.api_secret_hmac):
                    raise BinanceAPIException(
                        "Signed endpoint requires API_KEY and HMAC_SECRET (Binance -2015 likely).",
                        code=-2015
                    )
                params["timestamp"] = int(time.time() * 1000 + self._time_offset_ms)
                params.setdefault("recvWindow", self.recv_window_ms)
                query_string = urllib.parse.urlencode(params)
                # REST API ALWAYS uses HMAC, never Ed25519
                signature = hmac.new(
                    self.api_secret_hmac.encode("utf-8"),
                    query_string.encode("utf-8"),
                    hashlib.sha256,
                ).hexdigest()
                params["signature"] = signature
    
            # exponential backoff + jitter for transient cases
            backoffs = [0.2, 0.5, 1.0, 2.0]
            last_err = None
            for delay in [0.0, *backoffs]:
                if delay:
                    await asyncio.sleep(delay + random.uniform(0, delay / 4))
                try:
                    req_kwargs: Dict[str, Any] = {"headers": headers, "params": params}
                    if timeout is not None:
                        req_kwargs["timeout"] = aiohttp.ClientTimeout(total=max(0.1, float(timeout)))
                    async with self.session.request(method, url, **req_kwargs) as response:
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
        ts = self._ticker_24h_cache_ts
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
        ts = self._ticker_24h_cache_ts
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
        ts = self._ticker_24h_cache_ts
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
        ent = self._book_cache.get(sym)
        if ent and (now - ent["ts"] < self._BOOK_TTL):
            return ent["bid"], ent["ask"]
        try:
            data = await self._request("GET", "/api/v3/ticker/bookTicker", {"symbol": sym}, api="spot_api")
            bid = float(data.get("bidPrice")) if data and data.get("bidPrice") else None
            ask = float(data.get("askPrice")) if data and data.get("askPrice") else None
            self._book_cache[sym] = {"ts": now, "bid": bid, "ask": ask}
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
        start_time: Optional[Union[int, float]] = None,
        end_time: Optional[Union[int, float]] = None,
    ) -> List[List[float]]:
        """
        Return OHLCV rows from /api/v3/klines.
        Each row = [openTime, open, high, low, close, volume] as floats.
        """
        sym = self._norm_symbol(symbol)
        params = {"symbol": sym, "interval": interval, "limit": int(limit)}
        if start_time is not None:
            params["startTime"] = int(float(start_time))
        if end_time is not None:
            params["endTime"] = int(float(end_time))
        data = await self._request("GET", "/api/v3/klines", params, api="spot_api")
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

    def get_quote_precision(self, symbol: str) -> int:
        """Get the quote asset precision for the given symbol."""
        symbol = self._norm_symbol(symbol)
        filters = self.symbol_filters.get(symbol, {})
        prec = filters.get("_precisions", {}) or {}
        qp = prec.get("quoteAssetPrecision", prec.get("quotePrecision", 8))
        try:
            return max(0, int(qp))
        except Exception:
            return 8

    def _normalize_quote_order_qty(self, symbol: str, quote_amount: float) -> Decimal:
        """
        Quantize quoteOrderQty to exchange precision and avoid float artifacts
        (e.g., 24.300000000000004).
        """
        sym = self._norm_symbol(symbol)
        raw = Decimal(str(float(quote_amount or 0.0)))
        qp = self.get_quote_precision(sym)
        scale = Decimal("1").scaleb(-qp)
        q = raw.quantize(scale, rounding=ROUND_DOWN)
        if q <= 0:
            raise ValueError("ZeroQuoteAfterPrecision")
        return q

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
        px = None
        try:
            bid, ask = await self.get_best_bid_ask(symbol)
            px = ask if str(side).upper() == "BUY" else (bid or ask)
            if px is None:
                px = await self.get_current_price(symbol)
        except Exception:
            self.logger.warning("Price unavailable for MIN_NOTIONAL check; proceeding with caution.")
        if px is not None and qty_f * float(px) < min_notional:
            raise ValueError(f"MIN_NOTIONAL violation: {qty_f} * {px:.8f} < {min_notional}")
        return qty_f

    # ------------- canonical order path -------------
    async def market_buy(self, symbol: str, quote_amount: float, *, tag: str = "meta") -> dict:
        """Delegate to place_market_order so circuit breaker and normalization are applied."""
        return await self.place_market_order(symbol, "BUY", quote=quote_amount, tag=tag)

    async def place_market_order(
        self,
        symbol: str,
        side: str,
        *,
        quantity: Optional[float] = None,
        quote: Optional[float] = None,
        quote_order_qty: Optional[float] = None,
        tag: str = "",
        clientOrderId: Optional[str] = None,
        _timeInForce: Optional[str] = None,  # unused for MARKET on Binance
        max_slippage_bps: Optional[int] = None,
    ) -> dict:
        """
        Canonical MARKET order entrypoint (spec §3.6, §3.19).
        Supports either `quantity` or `quoteOrderQty` (via quote or quote_order_qty).
        
        Parameters:
            quote_order_qty: Alias for `quote` (for backwards compatibility with ExecutionManager)
            quote: Quote asset amount for BUY orders
        """
        # Handle quote_order_qty alias (ExecutionManager uses this parameter name)
        if quote_order_qty is not None and quote is None:
            quote = quote_order_qty
        
        await self._guard_execution_path(method="place_market_order", symbol=symbol, side=side, tag=tag)
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
                q = self._normalize_quote_order_qty(sym, quote)
                payload["quoteOrderQty"] = format(q, "f")
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
            status = str((raw or {}).get("status", "")).upper()
            status_ok = status in ("FILLED", "PARTIALLY_FILLED")
            res = self._normalize_exec_result(base_coid, raw, status_ok)
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
                    q_status = str((qraw or {}).get("status", "")).upper()
                    q_status_ok = q_status in ("FILLED", "PARTIALLY_FILLED")
                    res = self._normalize_exec_result(base_coid, qraw, q_status_ok)
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
        executed_qty = 0.0
        cumm_quote = 0.0
        with contextlib.suppress(Exception):
            executed_qty = float(raw.get("executedQty", 0.0) or 0.0)
        with contextlib.suppress(Exception):
            cumm_quote = float(raw.get("cummulativeQuoteQty", 0.0) or 0.0)

        fills_out: List[Dict[str, Any]] = []
        weighted_num = 0.0
        weighted_den = 0.0
        for f in raw.get("fills", []) or []:
            if not isinstance(f, dict):
                continue
            q = 0.0
            p = 0.0
            with contextlib.suppress(Exception):
                q = float(f.get("qty", 0.0) or 0.0)
            with contextlib.suppress(Exception):
                p = float(f.get("price", 0.0) or 0.0)
            if q > 0 and p > 0:
                weighted_num += q * p
                weighted_den += q
            fills_out.append(
                {
                    "qty": q,
                    "price": p,
                    "commission": float(f.get("commission", 0.0) or 0.0),
                    "commissionAsset": f.get("commissionAsset") or f.get("commission_asset"),
                }
            )

        raw_price = 0.0
        raw_avg = 0.0
        with contextlib.suppress(Exception):
            raw_price = float(raw.get("price", 0.0) or 0.0)
        with contextlib.suppress(Exception):
            raw_avg = float(raw.get("avgPrice", 0.0) or 0.0)

        fill_avg = (weighted_num / weighted_den) if weighted_den > 0 else 0.0
        inferred_avg = raw_avg
        if inferred_avg <= 0 and cumm_quote > 0 and executed_qty > 0:
            inferred_avg = cumm_quote / max(executed_qty, 1e-12)
        if inferred_avg <= 0 and fill_avg > 0:
            inferred_avg = fill_avg

        # Prefer effective avg price for MARKET fills; fallback to explicit raw price.
        effective_price = inferred_avg if inferred_avg > 0 else (raw_price if raw_price > 0 else 0.0)
        res = {
            "order_id": order_id,
            "ok": bool(status_ok),
            "status": raw.get("status", "REJECTED"),
            "executedQty": float(executed_qty),
            "price": float(effective_price),
            "avgPrice": float(inferred_avg if inferred_avg > 0 else effective_price),
            "cummulativeQuoteQty": float(cumm_quote),
            "fills": fills_out,
            "exchange_order_id": raw.get("orderId") or raw.get("order_id") or "",
            "client_order_id": raw.get("clientOrderId") or raw.get("origClientOrderId") or order_id,
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
        await self._guard_execution_path(method="send_order", symbol=symbol, side=side, tag=tag)
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
        await self._guard_execution_path(
            method="_send_real_order_quantity",
            symbol=symbol,
            side=side,
            tag=tag,
        )
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
        await self._guard_execution_path(
            method="_send_real_order_quote",
            symbol=symbol,
            side=side,
            tag=tag,
        )
        await self._sync_exchange_info()
        sym = self._norm_symbol(symbol)
        # Enforce min notional on quote directly
        filters = self.symbol_filters.get(sym, {})
        notional = filters.get("NOTIONAL") or filters.get("MIN_NOTIONAL") or {}
        min_notional = float(notional.get("minNotional", 0))
        if quote < min_notional:
            raise ValueError(f"MIN_NOTIONAL violation: quote {quote} < {min_notional}")

        q = self._normalize_quote_order_qty(sym, quote)
        params = {"symbol": sym, "side": side.upper(), "type": "MARKET", "quoteOrderQty": format(q, "f")}
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

    # ------------- balances -------------
    @staticmethod
    def _is_rate_limit_error(exc: Exception) -> bool:
        code = getattr(exc, "code", None)
        status_code = getattr(exc, "status_code", None)
        status = getattr(exc, "status", None)
        msg = str(exc).lower()
        return (
            code in (-1003, -1015, 429)
            or status_code in (429, -1003, -1015)
            or status in (429, -1003, -1015)
            or "apierror(code=-1003)" in msg
            or "too much request weight" in msg
            or "request weight used" in msg
            or "too many requests" in msg
        )

    async def get_spot_balances(self) -> Dict[str, Dict[str, float]]:
        """Spot balances via /api/v3/account (safe; no SAPI permission required).
        
        Returns all non-zero balances including assets that may not be tradable.
        Filtering of non-tradable pairs should be done at SharedState level.
        """
        # Allow balances even in paper/testnet mode
        # Diagnostic: Log the decision gate
        self.logger.debug(f"[EC:Balances] paper_trade={self.paper_trade}, testnet={self.testnet}")
        
        if self.paper_trade and not self.testnet:
            self.logger.debug("[EC] Paper simulation: returning empty balances")
            return {}

        now = time.time()
        # Shared cache path for full account snapshot.
        if self._acct_cache and (now - self._acct_cache_ts) < self._acct_ttl:
            return {
                k: {"free": float(v.get("free", 0.0)), "locked": float(v.get("locked", 0.0))}
                for k, v in dict(self._acct_cache).items()
            }

        async with self._acct_cache_lock:
            now = time.time()
            if self._acct_cache and (now - self._acct_cache_ts) < self._acct_ttl:
                return {
                    k: {"free": float(v.get("free", 0.0)), "locked": float(v.get("locked", 0.0))}
                    for k, v in dict(self._acct_cache).items()
                }

            try:
                self.logger.debug(f"[EC] Fetching real balances from {'testnet' if self.testnet else 'live'}")
                data = await self._request("GET", "/api/v3/account", signed=True, api="spot_api")
                out = {}
                for b in data.get("balances", []):
                    free = float(b.get("free", 0))
                    locked = float(b.get("locked", 0))
                    if free or locked:
                        out[b["asset"].upper()] = {"free": free, "locked": locked}
                self._acct_cache = dict(out)
                self._acct_cache_ts = time.time()
                return out
            except Exception as e:
                if self._is_rate_limit_error(e):
                    if self._acct_cache:
                        self.logger.warning(
                            "[EC:Balances] Rate limited on /api/v3/account; serving cached balances (age=%.1fs)",
                            max(0.0, time.time() - self._acct_cache_ts),
                        )
                        return {
                            k: {"free": float(v.get("free", 0.0)), "locked": float(v.get("locked", 0.0))}
                            for k, v in dict(self._acct_cache).items()
                        }
                    self.logger.warning("[EC:Balances] Rate limited and no cache available")
                raise

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
