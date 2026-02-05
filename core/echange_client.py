import aiohttp
import logging
import hmac
import hashlib
import time
import asyncio
from decimal import Decimal, ROUND_DOWN
from binance.async_client import AsyncClient
from binance.client import Client # Not directly used in async client, but good to keep if you have synchronous parts
from binance.exceptions import BinanceRequestException
from tenacity import wait_random
from typing import List, Dict, Any, Optional, Tuple, Union
from collections import defaultdict
import urllib.parse
import json
import functools
from datetime import datetime, timedelta
import decimal 
import math 
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, RetryError, wait_fixed

# Import BinanceAPIException and NetworkException from core.errors
from core.errors import BinanceAPIException, NetworkException


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# NEW THROTTLE DECORATOR AS PROVIDED BY THE USER
logger = logging.getLogger("Throttle")
def throttle(calls_per_second: int):
    interval = 1 / calls_per_second
    last_call = {}

    def decorator(func):
        @functools.wraps(func) # Maintain function metadata
        async def wrapper(*args, **kwargs):
            key = func.__name__
            now = time.monotonic()
            last = last_call.get(key, 0)
            wait_time = max(0, interval - (now - last))
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            last_call[key] = time.monotonic()
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Unexpected error in throttled function {func.__name__}: {e}", exc_info=True)
                raise
        return wrapper
    return decorator


class ExchangeClient:
    # Base URL for Binance API
    BASE_URL = "https://api.binance.com"
    TESTNET_BASE_URL = "https://testnet.binance.vision"

    # Define a set of symbols that are not actual Binance trading pairs
    INTERNAL_SYMBOLS = {"BALANCES", "REALIZED_PNL", "OPEN_POSITIONS"}

    def __init__(self, config, shared_state):
        self.config = config
        self.shared_state = shared_state
        self.logger = logging.getLogger("BinanceClient")
        # Ensure api_key and api_secret are correctly loaded based on paper mode
        self.api_key = self.config.BINANCE_API_KEY if not self.config.PAPER_MODE else self.config.BINANCE_TESTNET_API_KEY
        self.api_secret = self.config.BINANCE_API_SECRET if not self.config.PAPER_MODE else self.config.BINANCE_TESTNET_API_KEY
        self.testnet_mode = self.config.PAPER_MODE
        self.session = None # This will be the aiohttp session for direct signed requests
        self.client = None # This will be the AsyncClient instance
        self.base_currency = self.config.BASE_CURRENCY # Ensure base_currency is still set
        # Correctly set current_base_url based on paper mode for direct signed requests
        self.current_base_url = self.TESTNET_BASE_URL if self.config.PAPER_MODE else self.BASE_URL
        self.logger.info(f"BinanceClient initialized for {'PAPER' if self.config.PAPER_MODE else 'LIVE'} trading.")
        self.logger.info(f"Using base URL for signed requests: {self.current_base_url}")

        # Caches for exchange info and market stats
        self.exchange_info: Dict[str, Any] = {} # Stores exchange info by symbol
        self.market_stats: Dict[str, Dict] = {} # Stores market stats by symbol
        self.all_symbols_list: List[str] = [] # Stores a list of all tradable symbols
        self.new_listings_cache: List[str] = [] # Initialize new_listings_cache
        self.symbol_info: Dict[str, Any] = {} # Initialize symbol_info cache for new place_order
        self.balances: Dict[str, Dict[str, float]] = {} # Initialize balances cache
        self._symbols_set = set() # Initialize fast symbol set

        # New Listings Cache (used in get_new_listings_cached)
        self._cached_new_listings = None
        self._last_new_listing_fetch = None
        self._new_listing_cache_ttl = getattr(self.config, "IPO_LISTING_CACHE_TTL", 300)  # seconds
        # After self._new_listing_cache_ttl ...
        self.symbol_filters: Dict[str, Dict[str, float]] = {}  # normalized per-symbol filters
        self.unsupported_symbols: set[str] = set()


    async def initialize(self):
        """
        Initializes the aiohttp ClientSession and the Binance AsyncClient.
        Performs an initial connection test to verify API keys.
        Also populates initial exchange info and market stats caches.
        """
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=10)  # Session-wide timeout
            self.session = aiohttp.ClientSession(timeout=timeout)
            self.logger.debug("aiohttp ClientSession created for direct calls.")

        if self.client is None:
            try:
                # Initialize AsyncClient with loaded API key, secret, and testnet mode
                self.client = await AsyncClient.create(
                    api_key=self.api_key,
                    api_secret=self.api_secret,
                    testnet=self.testnet_mode
                )
                self.logger.info("✅ Binance client initialized.")
                # Optional: Log secrets being loaded (only log partial key for security)
                self.logger.info(f"🔑 Loaded API key: {self.api_key[:6]}..., Secret present: {bool(self.api_secret)}")


                # Test connection by getting account status (requires valid API keys)
                await self.client.get_account_status()
                self.logger.info("✅ ExchangeClient initialized and initial connection test successful.")

                # Populate exchange info cache
                raw_info = await self.client.get_exchange_info() # Use the throttled method
                self.exchange_info = {s['symbol']: s for s in raw_info.get('symbols', [])}
                # Build convenient caches (NO awaits, used everywhere)
                self.all_symbols_list = [s for s, meta in self.exchange_info.items()
                                         if meta.get("status") == "TRADING"]
                self._symbols_set = set(self.all_symbols_list)

                self.logger.info(f"Populated exchange info cache with {len(self.exchange_info)} symbols.")

                try:
                    filters_map = await self.get_symbol_filters()
                    if filters_map:
                        self.symbol_filters.update(filters_map)
                        self.logger.info(f"✅ Primed normalized symbol filters for {len(filters_map)} symbols.")
                except Exception as e:
                    self.logger.warning(f"⚠️ Could not prime symbol filters: {e}")

                # Populate market stats cache
                # Use get_all_24hr_stats for populating market_stats
                all_24hr_stats = await self.get_all_24hr_stats()
                self.market_stats = {symbol: {
                    "volume": stats["volume"],
                    "price": stats["price"],
                    "price_change": stats["price_change"],
                    # Add bid/ask/spread if needed for market_stats, otherwise populate from get_all_24hr_stats
                    "bidPrice": stats.get("bidPrice", 0), # Assuming get_all_24hr_stats might return these
                    "askPrice": stats.get("askPrice", 0), # Assuming get_all_24hr_stats might return these
                    "spread": abs(stats.get("askPrice", 0) - stats.get("bidPrice", 0)) if stats.get("askPrice") and stats.get("bidPrice") else None
                } for symbol, stats in all_24hr_stats.items()}
                self.logger.info(f"Populated market stats cache for {len(self.market_stats)} symbols.")

                # Populate new listings cache on initialization
                await self.get_new_listings()
                # Update initial balances
                await self.update_balances()


            except Exception as e:
                self.logger.error(f"❌ Failed to create or initialize AsyncClient or populate caches: {e}", exc_info=True)
                # --- Heartbeat status update (Section 5) ---
                self._report_status("ExchangeClient", "Failed", f"Initialization failed: {e}")
                self.client = None
                raise # Re-raise to indicate a critical initialization failure
        # --- Heartbeat status update (Section 5) ---
        self._report_status("ExchangeClient", "Running", "Client initialized and connected.")


    async def close(self):
        if hasattr(self, "session") and self.session and not self.session.closed:
            await self.session.close()
            self.logger.info("🔒 BinanceClient aiohttp session closed.")
            self.session = None
        if self.client:
            await self.client.close_connection()
            self.logger.info("🔒 BinanceClient AsyncClient connection closed.")
            self.client = None
        # --- Heartbeat status update (Section 5) ---
        self._report_status("ExchangeClient", "Stopped", "Client connection closed.")


    def _generate_signature(self, params: Dict[str, Any]) -> str:
        """
        Generates the HMAC SHA256 signature for the given parameters.
        Parameters are sorted and URL-encoded before signing.
        """
        query_string = urllib.parse.urlencode(params)
        m = hmac.new(self.api_secret.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256)
        return m.hexdigest()

    # --- START OF USER-REQUESTED MODIFICATION (SECTION 1) ---
    async def _send_signed_request(self, method: str, endpoint: str, params: dict) -> dict:
        if self.session is None or self.session.closed:
            self.logger.warning("Session closed. Reinitializing...")
            timeout = aiohttp.ClientTimeout(total=10)
            self.session = aiohttp.ClientSession(timeout=timeout)

        # Add timestamp and signature
        params['timestamp'] = int(time.time() * 1000)
        params['signature'] = self._generate_signature(params)

        headers = {
            'X-MBX-APIKEY': self.api_key
        }

        url = f"{self.current_base_url}{endpoint}"
        self.logger.debug(f"Sending signed request to {endpoint} with params: {params}")

        self.logger.error(f"📦 Final order payload: {params}")

        response = None
        try:
            # Use self.session.request for more generic HTTP method handling
            async with self.session.request(method, url, headers=headers, params=params) as response:
                if response.status != 200:
                    try:
                        error_msg = await response.text()
                    except Exception:
                        error_msg = "[No response body - likely connection closed early]"
                    self.logger.error(f"❌ Binance API error on {method} {endpoint}: Status {response.status} - {error_msg}")
                    # --- Heartbeat status update (Section 5) ---
                    self._report_status("ExchangeClient", "API Error", f"Signed request {endpoint} failed: Status {response.status}, Msg: {error_msg}")
                    raise BinanceAPIException(f"API Error: {error_msg}", code=response.status)
                return await response.json()
        except aiohttp.ClientResponseError as e:
            # This block might catch errors that were already handled by response.status != 200,
            # but it's good to have for unexpected ClientResponseErrors.
            self.logger.error(f"❌ Binance API error (ClientResponseError) on {method} {endpoint}: Status {e.status} - {str(e)}")
            # --- Heartbeat status update (Section 5) ---
            self._report_status("ExchangeClient", "API Error", f"ClientResponseError for {endpoint}: {str(e)}")
            raise BinanceAPIException(f"API Error: {str(e)}", code=e.status) from e
        except aiohttp.ClientConnectionError as e:
            self.logger.warning(f"🔁 Network error on {method} {endpoint}, retrying...: {e}")
            # --- Heartbeat status update (Section 5) ---
            self._report_status("ExchangeClient", "Network Error", f"Connection error for {endpoint}: {e}")
            if hasattr(self, "session") and self.session and not self.session.closed:
                await self.session.close()
            self.session = None
            raise e
        except Exception as e:
            self.logger.error(f"🔥 Unexpected error during signed request to {endpoint}: {e}", exc_info=True)
            # --- Heartbeat status update (Section 5) ---
            self._report_status("ExchangeClient", "Error", f"Unhandled exception in signed request {endpoint}: {e}")
            raise # Re-raise for general errors
    # --- END OF USER-REQUESTED MODIFICATION (SECTION 1) ---

    async def _send_public_request(self, http_method: str, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Sends a public (unsigned) request to a Binance API endpoint.
        This method is used for endpoints that do not require authentication.
        """
        if self.session is None:
            await self.initialize() # Ensure session is initialized

        if params is None:
            params = {}

        url = f"{self.current_base_url}{endpoint}"
        self.logger.debug(f"Sending {http_method} public request to {url} with params: {params}")

        try:
            if http_method == 'GET':
                async with self.session.get(url, params=params) as response:
                    response.raise_for_status()
                    return await response.json()
            else:
                raise ValueError(f"Unsupported HTTP method for public request: {http_method}")
        except aiohttp.ClientResponseError as e:
            self.logger.error(f"❌ HTTP error during public request to {endpoint}: Status {e.status}, Message: {e.message}", exc_info=True)
            raise BinanceAPIException(f"API Error: {e.message}", code=e.status) from e
        except aiohttp.ClientConnectionError as e:
            self.logger.error(f"❌ Connection error during public request to {endpoint}: {e}", exc_info=True)
            raise BinanceRequestException(f"Connection Error: {e}") from e
        except Exception as e:
            self.logger.error(f"🔥 Unexpected error during public request to {endpoint}: {e}", exc_info=True)
            raise # Re-raise for general errors


    async def get_open_positions(self) -> Dict[str, Dict[str, Any]]:
        """
        Fetches the user's open spot positions (assets with a balance > 0, excluding base currency).
        This method uses the AsyncClient's get_account() which handles signing internally.
        """
        self.logger.debug("DEBUG: Entering get_open_positions.")
        if not self.client:
            self.logger.error("❌ Cannot fetch open positions: AsyncClient not initialized.")
            return {}

        positions_data = {}
        try:
            self.logger.debug("DEBUG: Before calling get_account() for open positions.")
            account_info = await self.client.get_account()
            self.logger.debug("DEBUG: After calling get_account(). Account info retrieved.")

            balances_list = account_info.get('balances', [])
            self.logger.debug(f"DEBUG: Processing {len(balances_list)} balances for potential positions.")

            for asset_balance in balances_list:
                try:
                    asset_name = asset_balance['asset']
                    free_qty = float(asset_balance['free'])
                    locked_qty = float(asset_balance['locked'])
                    total_qty = free_qty + locked_qty
                    self.logger.debug(f"DEBUG: Checking asset: {asset_name}, total_qty: {total_qty}, free: {free_qty}, locked: {locked_qty}")

                    if asset_name != self.base_currency and total_qty > 0: # Use self.base_currency
                        symbol = f"{asset_name}{self.base_currency}" # Use self.base_currency
                        self.logger.debug(f"DEBUG: Found relevant asset '{asset_name}', constructing symbol: {symbol}")

                        # Get latest price for the symbol
                        price = await self.shared_state.get_latest_price(symbol)
                        self.logger.debug(f"DEBUG: Retrieved price for {symbol}: {price}")

                        positions_data[symbol] = {
                            "symbol": symbol,
                            "current_qty": total_qty,
                            "free_qty": free_qty,
                            "locked_qty": locked_qty,
                            "entry_price": price if price else 0.0, # Use latest price as 'entry' if historical entry isn't tracked
                            "side": "LONG", # All positive balances are considered LONG positions
                            "timestamp": int(time.time() * 1000)
                        }
                        self.logger.debug(f"DEBUG: Added position data for {symbol}.")
                except Exception as inner_e:
                    self.logger.error(f"❌ ERROR processing single asset '{asset_balance.get('asset', 'UNKNOWN_ASSET')}': {inner_e}", exc_info=True)

            self.logger.info(f"💰 Fetched open positions (held assets): {positions_data}")
            return positions_data

        except BinanceAPIException as e:
            self.logger.error(f"❌ Binance API error while fetching open positions (from balances): Code={e.code}, Msg={e.message}", exc_info=True)
            return {}
        except BinanceRequestException as e:
            self.logger.error(f"❌ Binance request error while fetching open positions (from balances): {e}", exc_info=True)
            return {}
        except Exception as e:
            self.logger.error(f"🔥 UNEXPECTED GENERAL ERROR while fetching open positions: {e}", exc_info=True)
            return {}

    # --- START OF DROP-IN place_order() REPLACEMENT ---
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1)) # --- USER-REQUESTED MODIFICATION (SECTION 2) ---
    async def place_order(self, symbol: str, side: str, quantity: float, order_type: str = "MARKET"):
        # Add a simple guard to avoid pseudo-symbols
        if symbol in self.INTERNAL_SYMBOLS:
            self.logger.warning(f"🚫 Skipping order for internal/virtual symbol: {symbol}")
            return None

        await self.ensure_symbol_info_loaded(symbol)
        info = self.symbol_info.get(symbol)

        if not info:
            self.logger.error(f"❌ No symbol info found for {symbol}.")
            # --- Heartbeat status update (Section 5) ---
            self._report_status("ExchangeClient", "Order Failed", f"No symbol info for {symbol}.")
            return None

        try:
            # --- OLD MIN_NOTIONAL KeyError check removed as get_min_notional handles it
            
            # Load filters using helper methods
            min_qty = float(self.get_min_qty_from_lot_size(symbol)) # Assuming a new helper for min_qty from LOT_SIZE
            step_size_str = self.get_step_size(symbol)
            step_size = float(step_size_str)
            
            price = await self.get_current_price(symbol)
            notional = price * quantity

            # --- Strict minNotional check ---
            try:
                min_notional = self.get_min_notional(symbol)
            except BinanceAPIException:
                self._report_status("ExchangeClient", "Order Skipped", f"minNotional unavailable for {symbol}.")
                return None
            if notional < min_notional:
                self.logger.info(f"⏭️ Skipping {symbol}: notional {notional:.2f} < minNotional {min_notional:.2f}.")
                self._report_status("ExchangeClient", "Order Skipped", f"Notional below minNotional for {symbol}.")
                return None

            # Adjust quantity to match stepSize
            # Use raw quantity here before truncation for floor calculation
            adjusted_qty_float = self.floor_to_step(quantity, step_size) # Use the new floor_to_step
            
            # --- Use truncate_to_step for final quantity string (Requested modification) ---
            quantity_str = self.truncate_to_step(adjusted_qty_float, step_size_str)
            # Convert back to float for internal calculations (e.g., notional check)
            adjusted_qty = float(quantity_str) 
            # --- END OF USER-REQUESTED MODIFICATION (QUANTITY SANITIZATION) ---


            # Skip if adjusted quantity < minQty
            if adjusted_qty < min_qty:
                self.logger.warning(f"❌ Adjusted quantity {adjusted_qty} for {symbol} < minQty {min_qty}. Skipping {symbol}.")
                # --- Heartbeat status update (Section 5) ---
                self._report_status("ExchangeClient", "Order Skipped", f"Adjusted quantity too small for {symbol} ({adjusted_qty}).")
                return None

            # Re-check notional after quantity adjustment
            final_notional = adjusted_qty * price
            # Skip if notional < minNotional
            if final_notional < min_notional:
                self.logger.warning(f"🚫 Notional {final_notional:.2f} < minNotional {min_notional} for {symbol}. Skipping {symbol}.")
                # --- Heartbeat status update (Section 5) ---
                self._report_status("ExchangeClient", "Order Skipped", f"Final notional too small for {symbol} ({final_notional:.2f}).")
                return None

            # --- Improve Balance Guarding (Section 2) ---
            if side.upper() == "BUY":
                # For BUY order, we need quote currency balance (e.g., USDT) >= final_notional
                base_asset_balance_info = self.balances.get(self.base_currency, {'free': 0.0}) # Use self.balances cache
                available_balance = base_asset_balance_info.get('free', 0.0)
                if available_balance < final_notional:
                    self.logger.warning(f"💰 Insufficient {self.base_currency} balance for {symbol} BUY. Needed: {final_notional:.2f}, Available: {available_balance:.2f}. Skipping {symbol}.")
                    # --- Heartbeat status update (Section 5) ---
                    self._report_status("ExchangeClient", "Order Skipped", f"Insufficient {self.base_currency} for {symbol} BUY.")
                    return None
            elif side.upper() == "SELL":
                # For SELL order, we need the traded asset balance >= adjusted_qty
                traded_asset = symbol.replace(self.base_currency, "") # e.g., BTC from BTCUSDT
                traded_asset_balance_info = self.balances.get(traded_asset, {'free': 0.0}) # Use self.balances cache
                available_balance = traded_asset_balance_info.get('free', 0.0)
                if available_balance < adjusted_qty:
                    self.logger.warning(f"📦 Insufficient {traded_asset} balance for {symbol} SELL. Needed: {adjusted_qty:.6f}, Available: {available_balance:.6f}. Skipping {symbol}.")
                    # --- Heartbeat status update (Section 5) ---
                    self._report_status("ExchangeClient", "Order Skipped", f"Insufficient {traded_asset} for {symbol} SELL.")
                    return None
            else:
                self.logger.error(f"❌ Invalid order side: {side}. Skipping order for {symbol}.")
                # --- Heartbeat status update (Section 5) ---
                self._report_status("ExchangeClient", "Order Skipped", f"Invalid order side for {symbol}: {side}.")
                return None
            # --- END Improve Balance Guarding (Section 2) ---


            # Final payload
            payload = {
                "symbol": symbol,
                "side": side.upper(),
                "type": order_type,
                "quantity": quantity_str, # Use the precisely formatted string quantity
                "timestamp": int(time.time() * 1000)
            }

            self.logger.error(f"📦 Final order payload: {payload}")

            endpoint = "/api/v3/order"
            order_response = await self._send_signed_request("POST", endpoint, payload)
            # --- Heartbeat status update (Section 5) ---
            self._report_status("ExchangeClient", "Order Placed", f"Successfully placed order for {symbol}.")
            return order_response

        except Exception as e:
            self.logger.error(f"🔥 Exception placing order for {symbol}: {e}", exc_info=True)
            # --- Heartbeat status update (Section 5) ---
            self._report_status("ExchangeClient", "Order Failed", f"Exception placing order for {symbol}: {e}")
            return None
    # --- END OF DROP-IN place_order() REPLACEMENT ---
    

    async def get_all_open_positions(self) -> Dict[str, Dict[str, Any]]:
        """
        Aggregates all open positions. Currently, only spot positions are considered.
        """
        all_positions = {}
        spot_positions = await self.get_open_positions()
        all_positions.update(spot_positions)
        return all_positions

    # --- START OF NEW METHOD: update_balances ---
    async def update_balances(self):
        """
        Updates the cached balances by fetching them from the exchange.
        """
        balances = await self.get_account_balances() # get_account_balances provides filtered non-zero balances
        self.balances = balances
        self.logger.debug(f"Updated balances cache: {self.balances}")
        return balances
    # --- END OF NEW METHOD: update_balances ---

    # Add near update_balances()
    async def refresh_balances(self):
        """Alias for update_balances() to satisfy callers."""
        return await self.update_balances()

    async def get_available_balance(self, asset: Optional[str] = None, currency: Optional[str] = None, **_ignored) -> float:
        """
        Return spendable balance for an asset (e.g., 'USDT', 'BTC').
        Accepts either `asset=` or legacy `currency=`.
        """
        try:
            # Lazy refresh if cache empty
            if not getattr(self, "balances", None):
                if hasattr(self, "update_balances") and callable(self.update_balances):
                    await self.update_balances()

            key = (asset or currency or "").upper()
            if not key:
                return 0.0

            # balances as dict: {'USDT': {'free': '12.3', 'locked': '0.1', ...}, ...}
            if isinstance(self.balances, dict):
                entry = self.balances.get(key)
                if isinstance(entry, dict):
                    free = float(entry.get("free", 0) or entry.get("available", 0) or 0)
                    # If you track reservations, subtract them; otherwise locked is irrelevant to “available”
                    reserved = float(entry.get("reserved", 0) or 0)
                    return max(0.0, free - reserved)

            # balances as list: [{'asset': 'USDT', 'free': '12.3', ...}, ...]
            if isinstance(self.balances, list):
                for b in self.balances:
                    if (b.get("asset") or "").upper() == key:
                        free = float(b.get("free", 0) or b.get("available", 0) or 0)
                        reserved = float(b.get("reserved", 0) or 0)
                        return max(0.0, free - reserved)

        except Exception as e:
            if hasattr(self, "logger"):
                self.logger.error(f"[ExchangeClient] get_available_balance({asset or currency}) failed: {e}", exc_info=True)

        return 0.0

    # --- START OF NEW HELPER METHOD: ensure_symbol_info_loaded ---
    async def ensure_symbol_info_loaded(self, symbol: str):
        """
        Ensures that the symbol's exchange information (filters) is loaded into cache.
        """
        if symbol not in self.symbol_info:
            self.logger.info(f"Loading symbol info for {symbol} into cache.")
            info = await self.get_symbol_info(symbol)
            if info:
                self.symbol_info[symbol] = info
                # Also cache normalized filters
                nf = self._normalize_symbol_filters(info.get("filters", []))
                if nf:
                    self.symbol_filters[symbol] = nf
            else:
                self.logger.error(f"Failed to load symbol info for {symbol}.")
    # --- END OF NEW HELPER METHOD: ensure_symbol_info_loaded ---

    # --- START OF NEW HELPER METHOD: get_min_notional ---
    def _normalize_symbol_filters(self, raw_filters: list) -> dict:
        """
        Normalize Binance filters into a single dict.
        keys: stepSize, minQty, maxQty, tickSize, minNotional
        Supports both MIN_NOTIONAL and NOTIONAL schemas.
        """
        result = {}
        for f in raw_filters:
            t = f.get("filterType")
            if t == "LOT_SIZE":
                result["stepSize"] = float(f.get("stepSize", 0) or 0)
                result["minQty"]   = float(f.get("minQty", 0) or 0)
                result["maxQty"]   = float(f.get("maxQty", 0) or 0)
            elif t == "PRICE_FILTER":
                result["tickSize"] = float(f.get("tickSize", 0) or 0)
            elif t in ("MIN_NOTIONAL", "NOTIONAL"):
                # handle both variants
                val = f.get("minNotional") or f.get("notional")
                if val is not None:
                    result["minNotional"] = float(val)
        return result

    def get_min_notional(self, symbol: str) -> float:
        """
        Strictly returns minNotional or raises.
        """
        # Prefer normalized cache
        f = self.symbol_filters.get(symbol)
        if f and "minNotional" in f:
            return float(f["minNotional"])

        # Fallback to symbol_info if not in cache
        info = self.symbol_info.get(symbol)
        if info:
            nf = self._normalize_symbol_filters(info.get("filters", []))
            if "minNotional" in nf:
                # cache it
                self.symbol_filters[symbol] = {**self.symbol_filters.get(symbol, {}), **nf}
                return float(nf["minNotional"])

        # If still missing, mark unsupported and stop
        self.logger.error(f"❌ MIN_NOTIONAL/NOTIONAL not found for {symbol}. Marking as unsupported.")
        self.unsupported_symbols.add(symbol)
        raise BinanceAPIException(f"Missing minNotional for {symbol}.", code=-1013)
    # --- END OF NEW HELPER METHOD: get_min_notional ---

    # Public method to get normalized filters per symbol
    async def get_symbol_filters_map(self, symbol: str) -> dict:
        """Return normalized filters dict for a single symbol: {stepSize, minQty, maxQty, tickSize, minNotional}"""
        info = await self.get_symbol_info(symbol)
        if not info:
            return {}
        return self._normalize_symbol_filters(info.get("filters", []))

    # Quantity rounding (floor_to_step)
    def floor_to_step(self, qty: float, step: float) -> float:
        if step <= 0:
            return qty
        steps = int(qty / step)
        return steps * step

    # --- START OF NEW HELPER METHOD: get_step_size ---
    def get_step_size(self, symbol: str) -> str:
        """
        Retrieves the stepSize for a given symbol from cached info.
        Returns "0.00000001" (a common small step size) if not found, with a warning.
        """
        info = self.symbol_info.get(symbol)
        if not info:
            self.logger.warning(f"⚠️ No symbol info found for {symbol} when getting step_size. Using fallback '0.00000001'.")
            return "0.00000001" # Fallback to a very small step size
        filters = {f["filterType"]: f for f in info["filters"]}
        lot_size_filter = filters.get("LOT_SIZE")
        if lot_size_filter:
            return lot_size_filter.get("stepSize", "0.00000001")
        self.logger.warning(f"⚠️ LOT_SIZE filter not found for {symbol}. Using fallback '0.00000001'.")
        return "0.00000001"
    # --- END OF NEW HELPER METHOD: get_step_size ---

    # --- START OF USER-REQUESTED MODIFICATION (QUANTITY SANITIZATION) ---
    def truncate_to_step(self, quantity: float, step_size: str) -> str:
        """
        Helper to truncate a quantity to the nearest valid step size precision.
        Uses Decimal for accurate floating-point arithmetic.
        """
        # Convert step_size string to Decimal for accurate precision calculation
        step_decimal = Decimal(step_size)
        
        # Calculate precision from step_size (e.g., "0.001" -> 3)
        # Using as_tuple().exponent * -1 gives the number of decimal places
        precision = step_decimal.as_tuple().exponent * -1
        
        # Quantize the quantity to the step_size precision, rounding down
        quant = Decimal(quantity)
        return str(quant.quantize(step_decimal, rounding=ROUND_DOWN))
    # --- END OF USER-REQUESTED MODIFICATION (QUANTITY SANITIZATION) ---

    def round_step_size(self, quantity, step_size):
        """
        Helper to round a quantity to the nearest valid step size.
        """
        # Calculate precision from step_size (e.g., 0.001 -> 3, 1.0 -> 0)
        # Using Decimal for log10 to handle very small step_sizes precisely
        precision = int(round(-math.log10(Decimal(str(step_size))), 0))
        return round(quantity, precision)

    # --- START OF USER-REQUESTED MODIFICATION (SECTION 1) ---
    def sanitize_quantity(self, qty: float) -> str:
        """
        Formats a float quantity into a string with sufficient precision,
        removing unnecessary trailing zeros and decimal points.
        """
        # Format to a high precision to avoid scientific notation for small numbers
        # Then rstrip '0' and '.' to get the exact required format.
        return format(qty, '.20f').rstrip('0').rstrip('.') or '0' # Ensure "0" if all are zeros
    # --- END OF USER-REQUESTED MODIFICATION (SECTION 1) ---

    # Assuming a helper for min_qty from LOT_SIZE is desired
    def get_min_qty_from_lot_size(self, symbol: str) -> float:
        """
        Retrieves the minQty from the LOT_SIZE filter for a given symbol.
        Returns 0.0 if not found, with a warning.
        """
        info = self.symbol_info.get(symbol)
        if not info:
            self.logger.warning(f"⚠️ No symbol info found for {symbol} when getting min_qty from LOT_SIZE.")
            return 0.0
        filters = {f["filterType"]: f for f in info["filters"]}
        lot_size_filter = filters.get("LOT_SIZE")
        if lot_size_filter:
            return float(lot_size_filter.get("minQty", 0.0))
        self.logger.warning(f"⚠️ LOT_SIZE filter (minQty) not found for {symbol}. Using fallback 0.0.")
        return 0.0

    def adjust_quantity_to_step(self, symbol, quantity, filters):
        """
        Adjusts a raw quantity to match Binance LOT_SIZE filters.
        """
        if not filters:
            self.logger.warning(f"❌ No LOT_SIZE filters provided for {symbol}.")
            return 0.0

        step_size = float(filters["stepSize"])
        min_qty = float(filters["minQty"])
        max_qty = float(filters["maxQty"])

        adjusted_qty = (int(float(quantity) / step_size)) * step_size
        adjusted_qty = round(adjusted_qty, int(-math.log10(step_size)))

        if adjusted_qty < min_qty:
            self.logger.warning(
                f"❌ Adjusted quantity {adjusted_qty} for {symbol} < minQty {min_qty}. Skipping."
            )
            return 0.0

        if adjusted_qty > max_qty:
            self.logger.warning(
                f"⚠️ Adjusted quantity {adjusted_qty} > maxQty {max_qty} for {symbol}. Capping."
            )
            adjusted_qty = max_qty

        return adjusted_qty

    async def place_market_order(self, symbol: str, side: str, quote_quantity: float):
        endpoint = "/api/v3/order"

        # STRICT minNotional check
        try:
            min_notional = self.get_min_notional(symbol)
        except BinanceAPIException:
            self._report_status("ExchangeClient", "Order Skipped", f"minNotional unavailable for {symbol}.")
            return None
        if quote_quantity < min_notional:
            self.logger.info(f"⏭️ Skipping {symbol}: quoteOrderQty {quote_quantity:.2f} < minNotional {min_notional:.2f}.")
            self._report_status("ExchangeClient", "Order Skipped", f"quoteOrderQty below minNotional for {symbol}.")
            return None

        try:
            # Primary attempt: use quoteOrderQty
            params = {
                "symbol": symbol,
                "side": side,
                "type": "MARKET",
                "quoteOrderQty": round(quote_quantity, 2), # quoteOrderQty typically doesn't need sanitize_quantity as it's a monetary value
                "timestamp": int(time.time() * 1000)
            }
            self.logger.info(f"🧾 Attempting order with quoteOrderQty for {symbol}: {params}")
            try: # Nested try-except for ClientConnectionError
                order_response = await self._send_signed_request("POST", endpoint, params)
                # --- Heartbeat status update (Section 5) ---
                self._report_status("ExchangeClient", "Order Placed", f"Market order (quoteOrderQty) placed for {symbol}.")
                return order_response
            except aiohttp.ClientConnectionError as conn_err:
                self.logger.warning(f"🔌 Connection error on quoteOrderQty attempt for {symbol}. Retrying with fallback. Error: {conn_err}")
                raise BinanceAPIException(None, -1, f"ClientConnectionError: {conn_err}")  # Force fallback
            except BinanceAPIException as e: # Catch API exceptions for the first attempt
                if getattr(e, "code", None) in (-2010, "-2010"):
                    try:
                        await self.refresh_balances()
                    except Exception:
                        pass
                    if side.upper() == "BUY":
                        self.logger.warning(f"⛔ Insufficient funds for {symbol} on quote path. Handing back to caller.")
                        return None
                self.logger.warning(f"⚠️ quoteOrderQty failed for {symbol}. Error: {e}. Retrying with quantity fallback...")
                raise # Re-raise to trigger the outer try-except for fallback


        except BinanceAPIException as e: # This catches the re-raised exception from the first attempt
            try:
                symbol_info = await self.get_symbol_info(symbol)
                price = await self.get_current_price(symbol)

                # --- START OF USER-REQUESTED MODIFICATION (ROOT FIX) ---
                base_quantity = quote_quantity / price
                # Extract filters
                filters = {f["filterType"]: f for f in symbol_info.get("filters", [])}
                lot_filter = filters.get("LOT_SIZE", {})

                step_size_str = lot_filter.get("stepSize", "UNKNOWN")
                min_qty_str = lot_filter.get("minQty", "UNKNOWN")
                max_qty_str = lot_filter.get("maxQty", "UNKNOWN")

                self.logger.warning(f"[LOT_SIZE] stepSize={step_size_str}, minQty={min_qty_str}, maxQty={max_qty_str}")
                self.logger.warning(f"[PRICE] {price}, quote_quantity={quote_quantity}, base_quantity={base_quantity}")

                # Truncate quantity
                quantity_str = self.truncate_to_step(base_quantity, step_size_str)
                quantity_float = float(quantity_str)

                min_qty = float(min_qty_str)
                # Auto-correct to minQty if needed
                if quantity_float < min_qty:
                    self.logger.warning(f"⚠️ Quantity {quantity_float} < minQty {min_qty} — adjusting up.")
                    quantity_float = min_qty
                    quantity_str = self.truncate_to_step(quantity_float, step_size_str) # Re-truncate after adjustment

                # Final notional check
                final_notional = quantity_float * price
                if final_notional < min_notional:
                    self.logger.warning(f"⚠️ Notional {final_notional:.2f} < minNotional {min_notional}. Skipping {symbol}.")
                    return None
                # --- END OF USER-REQUESTED MODIFICATION (ROOT FIX) ---

                fallback_params = {
                    "symbol": symbol,
                    "side": side,
                    "type": "MARKET",
                    "quantity": quantity_str, # Use the precisely formatted string quantity
                    "timestamp": int(time.time() * 1000)
                }

                self.logger.error(f"📤 Order retry payload for {symbol}: {fallback_params}")
                order_response = await self._send_signed_request("POST", endpoint, fallback_params)
                self.logger.info(f"✅ Retried market order succeeded using quantity for {symbol}.") # New log
                # --- Heartbeat status update (Section 5) ---
                self._report_status("ExchangeClient", "Order Placed", f"Market order (quantity fallback) placed for {symbol}.")
                return order_response

            except Exception as inner_error:
                self.logger.exception(f"❌ Fallback to quantity also failed for {symbol}: {inner_error}")
                # --- Heartbeat status update (Section 5) ---
                self._report_status("ExchangeClient", "Order Failed", f"Market order fallback failed for {symbol}: {inner_error}")
                # --- Improve Fallback Retry Handling (Section 3) ---
                self.logger.error(f"🚫 Trade aborted: Both market order attempts failed for {symbol}. Error: {inner_error}")
                # Optional: Cooldown to avoid retry storm
                # await asyncio.sleep(5) # Sleep for 5 seconds before next attempt for this trade
                return None # Gracefully abort

        except Exception as unexpected_error:
            self.logger.exception(f"❌ Unexpected error placing market order for {symbol}: {unexpected_error}")
            # --- Heartbeat status update (Section 5) ---
            self._report_status("ExchangeClient", "Order Failed", f"Unexpected error in market order for {symbol}: {unexpected_error}")
            return None # Gracefully abort


    @throttle(2) # 1 / 0.5 = 2 calls/second
    async def cancel_order(self, symbol: str, orderId: Union[int, str]) -> Dict[str, Any]:
        """
        Cancels a single open order by symbol and order ID.
        This method uses a signed DELETE request.
        """
        if self.config.PAPER_MODE:
            self.logger.info(f"[PAPER MODE] Simulating order cancellation for {symbol}, orderId {orderId}")
            # --- Heartbeat status update (Section 5) ---
            self._report_status("ExchangeClient", "Simulated Order Canceled", f"Simulated cancellation for {symbol} order {orderId}.")
            return {"symbol": symbol, "orderId": orderId, "status": "CANCELED"}

        endpoint = "/api/v3/order"
        params = {
            "symbol": symbol,
            "orderId": int(orderId) # Ensure orderId is an integer
        }

        try:
            response = await self._send_signed_request('DELETE', endpoint, params)
            self.logger.info(f"✅ Order {orderId} for {symbol} successfully cancelled: {response}")
            # --- Heartbeat status update (Section 5) ---
            self._report_status("ExchangeClient", "Order Canceled", f"Order {orderId} canceled for {symbol}.")
            return response
        except BinanceAPIException as e:
            self.logger.error(f"❌ Failed to cancel order {orderId} for {symbol}: {e.code} - {e.message}", exc_info=True)
            # --- Heartbeat status update (Section 5) ---
            self._report_status("ExchangeClient", "Order Cancellation Failed", f"Failed to cancel {symbol} order {orderId}: {e.message}")
            raise
        except Exception as e:
            self.logger.error(f"🔥 An unexpected error occurred while canceling order: {e}", exc_info=True)
            # --- Heartbeat status update (Section 5) ---
            self._report_status("ExchangeClient", "Error", f"Unhandled exception canceling order {orderId} for {symbol}: {e}")
            raise

    # --- START OF NEW METHOD: get_order_status ---
    async def get_order_status(self, symbol: str, order_id: Union[str, int]) -> dict:
        """
        Fetches the status of a specific order by symbol and order ID.
        """
        endpoint = "/api/v3/order"
        params = {
            "symbol": symbol,
            "orderId": int(order_id)
        }

        try:
            order = await self._send_signed_request("GET", endpoint, params)
            self.logger.info(f"🔄 Polled order status for {symbol} | Order ID: {order_id} | Status: {order.get('status')}")
            return order
        except Exception as e:
            self.logger.error(f"🔥 Failed to fetch order status for {symbol}, orderId {order_id}: {e}", exc_info=True)
            # --- Heartbeat status update (Section 5) ---
            self._report_status("ExchangeClient", "Order Status Fetch Failed", f"Failed to get status for {symbol} order {order_id}: {e}")
            return {}
    # --- END OF NEW METHOD: get_order_status ---


    async def get_all_open_positions(self) -> Dict[str, Dict[str, Any]]:
        """
        Aggregates all open positions. Currently, only spot positions are considered.
        """
        all_positions = {}
        spot_positions = await self.get_open_positions()
        all_positions.update(spot_positions)
        return all_positions

    @throttle(1) # 1 / 1.0 = 1 call/second
    async def get_balances(self) -> Dict[str, Dict[str, float]]:
        """
        Fetches the user's balances for all assets with a positive free or locked balance.
        Returns a dictionary where keys are asset names and values are dictionaries
        containing 'free' and 'locked' quantities.
        This method uses the AsyncClient's get_account() which handles signing internally.
        """
        if not self.client:
            self.logger.error("❌ Cannot fetch balances: AsyncClient not initialized.")
            return {}

        balances = {}
        try:
            account_info = await self.client.get_account()
            for asset_balance in account_info.get('balances', []):
                asset_name = asset_balance['asset']
                free_qty = float(asset_balance['free'])
                locked_qty = float(asset_balance['locked'])
                
                if free_qty > 0 or locked_qty > 0:
                    balances[asset_name] = {
                        "free": free_qty,
                        "locked": locked_qty,
                        "total": free_qty + locked_qty
                    }
            self.logger.info(f"💰 Fetched current balances: {balances}")
            return balances
        except BinanceAPIException as e:
            self.logger.error(f"❌ Binance API error while fetching balances: {e.status_code} - {e.message}", exc_info=True)
            # --- Heartbeat status update (Section 5) ---
            self._report_status("ExchangeClient", "Balance Fetch Failed", f"API error fetching balances: {e.message}")
            return {}
        except BinanceRequestException as e:
            self.logger.error(f"❌ Binance request error while fetching balances: {e}", exc_info=True)
            # --- Heartbeat status update (Section 5) ---
            self._report_status("ExchangeClient", "Balance Fetch Failed", f"Request error fetching balances: {e}")
            return {}
        except Exception as e:
            self.logger.error(f"🔥 An unexpected error occurred while fetching balances: {e}", exc_info=True)
            # --- Heartbeat status update (Section 5) ---
            self._report_status("ExchangeClient", "Error", f"Unhandled exception fetching balances: {e}")
            return {}

    @throttle(1) # 1 / 1.0 = 1 call/second
    async def get_account_balances(self) -> dict:
        """
        Retrieves actual wallet balances from Binance, filtering for non-zero balances only.

        Returns:
            dict: A dictionary mapping asset symbols to their 'free' and 'locked' balances.
        """
        try:
            account_info = await self.client.get_account()
            balances = {
                item["asset"]: {
                    "free": float(item["free"]), # Ensure float conversion
                    "locked": float(item["locked"]) # Ensure float conversion
                }
                for item in account_info["balances"]
                if float(item["free"]) + float(item["locked"]) > 0
            }
            self.logger.info(f"[ExchangeClient] ✅ Retrieved {len(balances)} non-zero balances from Binance.")
            return balances
        except Exception as e:
            self.logger.error(f"[ExchangeClient] ❌ Failed to fetch account balances: {e}", exc_info=True)
            self._report_status("ExchangeClient", "Balance Fetch Failed", f"Failed to fetch account balances: {e}")
            return {}

    @throttle(1) # 1 / 1.0 = 1 call/second
    async def get_all_balances(self) -> Dict[str, float]:
        """
        Fetch all non-zero total balances from the Binance account.
        Returns a dictionary where keys are asset names and values are the total balance (free + locked).
        """
        if not self.client:
            self.logger.error("❌ Cannot fetch all balances: AsyncClient not initialized.")
            return {}
        try:
            account_info = await self.client.get_account()
            balances = {
                asset['asset']: float(asset['free']) + float(asset['locked'])
                for asset in account_info['balances']
                if float(asset['free']) + float(asset['locked']) > 0
            }
            self.logger.info(f"💰 Fetched all balances: {balances}")
            return balances
        except Exception as e:
            self.logger.error(f"❌ Failed to fetch all balances: {e}", exc_info=True)
            self._report_status("ExchangeClient", "Balance Fetch Failed", f"Failed to fetch all balances: {e}")
            return {}

    @throttle(1) # 1 / 1.0 = 1 call/second
    async def get_account_balance(self, asset: str) -> Dict[str, float]:
        """
        Fetches the free, locked, and total balance for a specific asset.
        Returns a dictionary with 'free', 'locked', and 'total' quantities.
        Returns {'free': 0.0, 'locked': 0.0, 'total': 0.0} if the asset is not found.
        """
        self.logger.debug(f"Fetching balance for asset: {asset}")
        # Use the cached balances if available, otherwise fetch
        if not self.balances:
            await self.update_balances() # Ensure balances are loaded

        balance_data = self.balances.get(asset, {'free': 0.0, 'locked': 0.0}) # Access from self.balances
        return {
            'free': float(balance_data.get('free', 0.0)),
            'locked': float(balance_data.get('locked', 0.0)),
            'total': float(balance_data.get('free', 0.0)) + float(balance_data.get('locked', 0.0))
        }

    @throttle(1) # 1 / 1.0 = 1 call/second
    async def get_all_symbols(self) -> List[str]:
        """
        Fetches all trading symbols from Binance and returns a list of symbol strings.
        Example: ["BTCUSDT", "ETHUSDT", ...]
        """
        self.logger.info("📡 [get_all_symbols] Fetching symbols from Binance...")

        try:
            info = await self.client.get_exchange_info()
            symbols = [
                s["symbol"]
                for s in info["symbols"]
                if s["symbol"].endswith("USDT") and s["status"] == "TRADING"
            ]
            self.logger.info(f"✅ [get_all_symbols] Retrieved {len(symbols)} symbols.")
            return symbols
        except Exception as e:
            self.logger.exception(f"❌ Failed to get symbols from Binance: {e}")
            # --- Heartbeat status update (Section 5) ---
            self._report_status("ExchangeClient", "Symbol Fetch Failed", f"Failed to get all symbols: {e}")
            return []

    @throttle(0.2) # 1 / 5.0 = 0.2 calls/second
    async def symbol_exists(self, symbol: str) -> bool:
        """
        Checks if a given symbol exists and is tradable on Binance.
        """
        if not self.client:
            self.logger.warning("❌ Binance client not initialized for symbol existence check.")
            return False
        try:
            # Use cached exchange_info for faster lookup
            if not self.exchange_info:
                # Fallback if cache is empty, though initialize() should populate it
                exchange_info_response = await self.get_exchange_info()
                self.exchange_info = {s['symbol']: s for s in exchange_info_response.get('symbols', [])}
            
            return symbol in self.exchange_info and self.exchange_info[symbol].get("status") == "TRADING"
        except Exception as e:
            self.logger.error(f"Error checking if symbol exists: {e}", exc_info=True)
            # --- Heartbeat status update (Section 5) ---
            self._report_status("ExchangeClient", "Symbol Check Failed", f"Error checking if {symbol} exists: {e}")
            return False

    def symbol_exists_cached(self, symbol: str) -> bool:
        """O(1) membership check using preloaded exchange info; no await, no throttling."""
        try:
            return symbol in self._symbols_set
        except Exception:
            return False

    def get_cached_price(self, symbol: str) -> float:
        """Return last known price from market_stats cache, or 0.0 if missing."""
        try:
            return float(self.market_stats.get(symbol, {}).get("price", 0.0))
        except Exception:
            return 0.0

    @throttle(0.2) # 1 / 5.0 = 0.2 calls/second
    async def validate_symbol(self, symbol: str) -> Tuple[bool, str]:
        """
        Validate that a symbol exists and is active on Binance.
        Returns: (True, reason) or (False, error_reason)
        """
        try:
            # The method now expects a string directly, no need for dict extraction
            symbol_str = symbol

            if not symbol_str:
                self.logger.warning(f"⚠️ Empty symbol string passed for validation.")
                return False, "Symbol string is empty"

            info = self.exchange_info.get(symbol_str)
            if not info:
                self.logger.warning(f"⚠️ Symbol {symbol_str} not found in Binance exchange_info.")
                return False, f"Symbol {symbol_str} not found in exchange_info"

            if info.get("status", "").lower() != "trading":
                self.logger.warning(f"⚠️ Symbol {symbol_str} exists but is not active for trading (status: {info.get('status')}).")
                return False, f"Symbol {symbol_str} not active for trading"
            
            # Base asset check (e.g., must be a USDT pair)
            if not symbol_str.endswith(self.base_currency):
                self.logger.warning(f"⚠️ Rejected symbol {symbol_str} - Reason: Not a {self.base_currency} pair.")
                return False, f"Not {self.base_currency} pair"

            # Reject leveraged or derivative tokens
            lower_symbol = symbol_str.lower()
            blacklist_keywords = ["upusdt", "downusdt", "bullusdt", "bearusdt", "3lusdt", "3susdt"]
            if any(x in lower_symbol for x in blacklist_keywords):
                self.logger.warning(f"⚠️ Rejected symbol {symbol_str} - Reason: Leveraged/ETF symbol.")
                return False, f"Rejected due to leveraged/ETF symbol: {symbol_str}"

            # Optional: Minimum volume check (using market_stats cache)
            stats = self.market_stats.get(symbol_str)
            if stats:
                volume = float(stats.get("volume", 0))
                min_volume_threshold = getattr(self.config, 'SYMBOL_MIN_VOLUME', 100_000)
                if volume < min_volume_threshold:
                    self.logger.warning(f"⚠️ Rejected symbol {symbol_str} - Reason: Low volume ({volume}).")
                    return False, f"Low volume: {volume}"
            else:
                self.logger.debug(f"DEBUG: No market stats found for {symbol_str}. Skipping volume check.")
                # Decide if you want to reject if stats are missing or proceed
                # For now, we proceed if stats are missing, but log a warning.

            self.logger.info(f"✅ Symbol {symbol_str} validated successfully. Reason: OK")
            return True, "Valid"
        
        except Exception as e:
            self.logger.error(f"❌ Exception during symbol validation for {symbol}: {e}", exc_info=True)
            return False, f"Exception: {e}"

    @throttle(1/60) # 1 / 60.0 = 0.0166... calls/second
    async def get_exchange_info(self) -> Dict[str, Any]:
        """
        Fetches exchange information including all symbols and trading rules.
        """
        if not self.client:
            self.logger.error("[ExchangeClient] ❌ Cannot fetch exchange info: Binance client not initialized.")
            return {}
        try:
            response = await self.client.get_exchange_info()
            self.logger.info(f"[ExchangeClient] 📊 Fetched exchange info from Binance.")
            return response
        except Exception as e:
            self.logger.error(f"[ExchangeClient] ❌ Failed to fetch exchange info: {e}", exc_info=True)
            return {}

    @throttle(2) # 1 / 0.5 = 2 calls/second
    async def get_market_stats(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Fetches 24hr ticker data (volume, price, spread) for each symbol.
        Returns a dict: {symbol: {volume, price, spread}}
        This method is now primarily used for populating the market_stats cache
        and can be replaced by get_all_24hr_stats for comprehensive data.
        """
        stats = {}
        # Use self.current_base_url for API calls
        url = self.current_base_url + "/api/v3/ticker/24hr"

        # Use the existing session if available, otherwise create a new one
        session_to_use = self.session if self.session else aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
        
        try:
            async with session_to_use.get(url) as response:
                response.raise_for_status() # Raise an exception for bad status codes
                data = await response.json()

            for item in data:
                symbol = item.get("symbol")
                if symbol not in symbols:
                    continue

                try:
                    volume = float(item.get("quoteVolume", 0))
                    last_price = float(item.get("lastPrice", 0))
                    bid = float(item.get("bidPrice", 0))
                    ask = float(item.get("askPrice", 0))
                    spread = abs(ask - bid) if ask and bid else None

                    stats[symbol] = {
                        "volume": volume,
                        "price": last_price,
                        "spread": spread,
                        "bidPrice": bid, # Include bid and ask for completeness
                        "askPrice": ask
                    }
                except Exception as e:
                    self.logger.warning(f"⚠️ Malformed 24hr ticker data for symbol {symbol}: {e}", exc_info=True)
                    continue  # Malformed entry
            self.logger.info(f"✅ Fetched market stats for {len(stats)} symbols.")
            return stats
        except aiohttp.ClientResponseError as e:
            self.logger.error(f"❌ HTTP error fetching market stats: Status {e.status}, Message: {e.message}", exc_info=True)
            return {}
        except aiohttp.ClientConnectionError as e:
            self.logger.error(f"❌ Connection error fetching market stats: {e}", exc_info=True)
            return {}
        except Exception as e:
            self.logger.error(f"🔥 Unexpected error fetching market stats: {e}", exc_info=True)
            return {}
        finally:
            # Close session only if it was created within this method
            if session_to_use is not self.session and not session_to_use.closed:
                await session_to_use.close()

    @throttle(1) # 1 / 1.0 = 1 call/second
    async def get_all_24hr_stats(self) -> Dict[str, Dict]:
        """
        Fetches all 24hr ticker data (volume, price, price_change, etc.) at once.
        Returns a dict: {symbol: {price, volume, price_change, ...}}
        """
        endpoint = "/api/v3/ticker/24hr"
        url = self.current_base_url + endpoint

        session_to_use = self.session if self.session else aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))

        result = {}
        try:
            async with session_to_use.get(url) as response:
                response.raise_for_status()
                data = await response.json()

            for item in data:
                symbol = item.get("symbol")
                if symbol:
                    try:
                        result[symbol] = {
                            "price": float(item.get("lastPrice", 0)),
                            "volume": float(item.get("quoteVolume", 0)),
                            "price_change": float(item.get("priceChangePercent", 0)),
                            "highPrice": float(item.get("highPrice", 0)),
                            "lowPrice": float(item.get("lowPrice", 0)),
                            "openPrice": float(item.get("openPrice", 0)),
                            "bidPrice": float(item.get("bidPrice", 0)),
                            "askPrice": float(item.get("askPrice", 0)),
                            "weightedAvgPrice": float(item.get("weightedAvgPrice", 0)),
                            "prevClosePrice": float(item.get("prevClosePrice", 0)),
                            "volume_base_asset": float(item.get("volume", 0)), # Base asset volume
                            "count": int(item.get("count", 0)), # Number of trades
                        }
                    except Exception as e:
                        self.logger.warning(f"⚠️ Malformed 24hr stats data for symbol {symbol}: {e}", exc_info=True)
                        continue # Skip this malformed entry
            self.logger.info(f"✅ Fetched all 24hr stats for {len(result)} symbols.")
            return result
        except aiohttp.ClientResponseError as e:
            self.logger.error(f"❌ HTTP error fetching all 24hr stats: Status {e.status}, Message: {e.message}", exc_info=True)
            return {}
        except aiohttp.ClientConnectionError as e:
            self.logger.error(f"❌ Connection error fetching all 24hr stats: {e}", exc_info=True)
            return {}
        except Exception as e:
            self.logger.error(f"🔥 Unexpected error fetching all 24hr stats: {e}", exc_info=True)
            return {}
        finally:
            if session_to_use is not self.session and not session_to_use.closed:
                await session_to_use.close()

    @throttle(1) # 1 / 1.0 = 1 call/second
    async def get_24hr_tickers(self) -> List[Dict]:
        """
        Fetch 24hr ticker price change statistics for all symbols.
        Returns: List of dicts like [{'symbol': 'BTCUSDT', 'volume': '...', ...}, ...]
        """
        if not self.client:
            self.logger.error("❌ Cannot fetch 24hr tickers: AsyncClient not initialized.")
            return []
        try:
            # Use the AsyncClient's get_ticker() method to fetch all 24hr tickers
            # This method returns a list of dictionaries, each representing a ticker
            tickers = await self.client.get_ticker()
            return tickers
        except Exception as e:
            self.logger.error(f"❌ Failed to fetch 24hr tickers: {e}", exc_info=True)
            return []


    @throttle(2) # 1 / 0.5 = 2 calls/second
    async def get_recent_trades(self, symbol: str, limit: int = 500) -> List[Dict[str, Any]]:
        """
        Fetches recent trades for a given symbol.
        """
        if symbol in self.INTERNAL_SYMBOLS:
            self.logger.warning(f"[ExchangeClient] Skipping recent trades fetch for virtual symbol: {symbol}")
            return []
        if not self.client:
            self.logger.error("❌ Cannot fetch recent trades: AsyncClient not initialized.")
            return []
        try:
            self.logger.debug(f"Fetching recent trades for {symbol}, limit={limit}")
            trades = await self.client.get_recent_trades(symbol=symbol, limit=limit)
            return trades
        except Exception as e:
            self.logger.error(f"❌ Failed to fetch recent trades for {symbol}: {e}", exc_info=True)
            return []

    @throttle(1) # 1 / 1.0 = 1 call/second
    async def get_ohlcv(self, symbol: str, timeframe: str = "1m", limit: int = 500) -> Optional[List[Dict[str, Any]]]:
        """
        Fetches OHLCV data for a given symbol and timeframe.

        Args:
            symbol (str): Trading pair symbol like 'BTCUSDT'
            timeframe (str): Candlestick interval, e.g., '1m', '5m', '1h'
            limit (int): Number of bars to retrieve

        Returns:
            Optional[List[Dict]]: List of candlestick data in standardized format, or None on failure.
        """
        # 🚫 Skip virtual/internal symbols that do not exist on Binance
        if symbol in self.INTERNAL_SYMBOLS:
            self.logger.warning(f"[ExchangeClient] Skipping OHLCV fetch for virtual symbol: {symbol}")
            return None # Return None as per Optional[List[Dict]]
        if not self.client:
            self.logger.error("❌ Cannot fetch OHLCV: AsyncClient not initialized.")
            return None # Return None as per Optional[List[Dict]]

        try:
            self.logger.info(f"Fetching OHLCV for {symbol}@{timeframe}, limit={limit}")
            # Use self.client.get_klines as fetch_ohlcv is not a direct method on AsyncClient
            response = await self.client.get_klines(symbol=symbol, interval=timeframe, limit=limit)

            # Expected Binance format: [timestamp, open, high, low, close, volume, ...]
            ohlcv_list = []
            for candle in response:
                ohlcv_list.append({
                    "timestamp": candle[0],
                    "open": float(candle[1]),
                    "high": float(candle[2]),
                    "low": float(candle[3]),
                    "close": float(candle[4]),
                    "volume": float(candle[5]),
                })

            return ohlcv_list

        except Exception as e:
            self.logger.error(f"❌ Failed to fetch OHLCV for {symbol}@{timeframe}: {e}", exc_info=True)
            return None # Return None on failure

    @throttle(10) # 1 / 0.1 = 10 calls/second
    async def get_price(self, symbol: str) -> float:
        """
        Fetches the current price of a symbol using the Ticker endpoint.
        Returns 0.0 on failure.
        """
        self.logger.debug(f"Fetching price for {symbol}")
        try:
            ticker_data = await self.client.get_symbol_ticker(symbol=symbol)
            return float(ticker_data.get('price', 0.0))
        except Exception as e:
            self.logger.error(f"❌ Failed to get price for {symbol}: {e}", exc_info=True)
            return 0.0

    async def get_current_price(self, symbol: str) -> float:
        """
        A wrapper for get_price, providing a more descriptive name.
        """
        return await self.get_price(symbol)

    @throttle(10) # 1 / 0.1 = 10 calls/second
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Fetches the latest ticker information for a specific symbol.
        This method uses the AsyncClient's get_symbol_ticker() method.
        It also includes a guard for invalid pseudo-symbols.
        """
        # 🚫 Skip virtual/internal symbols that do not exist on Binance
        if symbol in self.INTERNAL_SYMBOLS:
            self.logger.warning(f"[ExchangeClient] Skipping non-tradeable symbol: {symbol}")
            return {} # Return empty dict to match existing error return type for get_ticker
        try:
            self.logger.debug(f"Fetching ticker for {symbol}")
            ticker = await self.client.get_symbol_ticker(symbol=symbol)
            return ticker
        except Exception as e:
            self.logger.error(f"❌ Failed to fetch ticker for {symbol}: {e}", exc_info=True)
            return {}

    @throttle(1) # 1 / 1.0 = 1 call/second
    async def get_all_tickers(self) -> List[Dict[str, Any]]:
        """
        Fetches the latest ticker information for all symbols.
        This method uses the AsyncClient's get_all_tickers() method.
        """
        try:
            self.logger.debug(f"Fetching all tickers.")
            tickers = await self.client.get_all_tickers()
            return tickers
        except Exception as e:
            self.logger.error(f"❌ Failed to fetch all tickers: {e}", exc_info=True)
            return []

    @throttle(10) # 1 / 0.1 = 10 calls/second
    async def get_24hr_volume(self, symbol: str) -> Optional[float]: # Changed return type hint to Optional[float]
        """
        Fetches the 24-hour trading volume for a specific symbol.
        This method uses the AsyncClient's get_ticker(symbol=symbol)
        and extracts the 'quoteVolume'.
        """
        # 🚫 Skip virtual/internal symbols that do not exist on Binance
        if symbol in self.INTERNAL_SYMBOLS:
            self.logger.warning(f"[ExchangeClient] Skipping 24hr volume fetch for virtual symbol: {symbol}")
            return None # Return None as per Optional[float]
        try:
            self.logger.debug(f"Fetching 24hr volume for {symbol}")
            # get_ticker returns 24hr ticker information, including quoteVolume
            ticker_data = await self.client.get_ticker(symbol=symbol)
            # The 'quoteVolume' field represents the 24hr trading volume in the quote asset.
            volume = float(ticker_data.get("quoteVolume", 0.0))
            self.logger.info(f"✅ 24hr volume for {symbol}: {volume}")
            return volume
        except Exception as e:
            self.logger.warning(f"[ExchangeClient] Failed to fetch 24hr volume for {symbol}: {e}") # Changed to warning as per user's example
            return None # Return None as per Optional[float]

    @throttle(0.2) # 1 / 5.0 = 0.2 calls/second
    async def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves symbol information from Binance's exchange info endpoint,
        with strict validation for trading status.
        """
        try:
            # Use _send_public_request for direct API call to exchangeInfo endpoint as it's a public endpoint
            info = await self._send_public_request("GET", "/api/v3/exchangeInfo", params={"symbol": symbol.upper()})
            
            # The response for a specific symbol will have 'symbols' as a list with one item
            symbol_data = info.get("symbols", [])[0] 
            
            if symbol_data["status"] != "TRADING":
                self.logger.warning(f"[ExchangeClient] Symbol {symbol} found but status is {symbol_data['status']}. Skipping.")
                return None
            self.logger.info(f"[ExchangeClient] ✅ Fetched symbol info for {symbol}.")
            return symbol_data
        except IndexError:
            self.logger.warning(f"[ExchangeClient] Symbol {symbol} not found in exchange info.")
            return None
        except Exception as e:
            self.logger.warning(f"[ExchangeClient] Failed to fetch symbol info for {symbol}: {e}", exc_info=True)
            return None
    
    @throttle(0.2)
    async def get_symbol_filters_for_symbol(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Fetches only the trading filters for a given symbol.
        """
        symbol_info = await self.get_symbol_info(symbol)
        if symbol_info:
            return symbol_info.get("filters", [])
        return []

    async def get_symbol_filters(self) -> Dict[str, Dict]:
        """Retrieve minNotional and stepSize for all symbols."""
        try:
            data = await self._send_public_request("GET", "/api/v3/exchangeInfo")
            filters_map = {}
            for s in data.get("symbols", []):
                min_notional = None
                min_qty = max_qty = stepSize = tickSize = 0.0
                for f in s.get("filters", []):
                    t = f.get("filterType")
                    if t in ("NOTIONAL", "MIN_NOTIONAL"):
                        min_notional = float(f.get("minNotional", 0) or 0)
                    elif t == "LOT_SIZE":
                        min_qty  = float(f.get("minQty", 0) or 0)
                        max_qty  = float(f.get("maxQty", 0) or 0)
                        stepSize = float(f.get("stepSize", 0) or 0)
                    elif t == "PRICE_FILTER":
                        tickSize = float(f.get("tickSize", 0) or 0)

                filters_map[s["symbol"]] = {
                    "minNotional": min_notional or 0.0,
                    "minQty": min_qty,
                    "maxQty": max_qty,
                    "stepSize": stepSize,
                    "tickSize": tickSize,
                }
            self.logger.info(f"✅ Loaded symbol filters for {len(filters_map)} symbols from exchange.")
            return filters_map
        except Exception as e:
            self.logger.error(f"❌ Failed to fetch symbol filters: {e}", exc_info=True)
            return {}


    @throttle(2) # 1 / 0.5 = 2 calls/second
    async def get_order_books(self, symbol: str, limit: int = 50) -> dict:
        """
        Fetches order book for a given symbol.

        Args:
            symbol (str): The trading pair symbol (e.g., "BTCUSDT").
            limit (int): Number of orders to retrieve per side (default is 50).

        Returns:
            dict: Dictionary containing bids and asks, e.g.:
                  {"bids": [[price, qty], ...], "asks": [[price, qty], ...]}
        """
        # 🚫 Skip virtual/internal symbols that do not exist on Binance
        if symbol in self.INTERNAL_SYMBOLS:
            self.logger.warning(f"[ExchangeClient] Skipping order book fetch for virtual symbol: {symbol}")
            return {"bids": [], "asks": []}
        if not self.client:
            self.logger.error("❌ Cannot fetch order book: AsyncClient not initialized.")
            return {"bids": [], "asks": []}
        try:
            # Using self.client.get_order_book to fetch order book data
            result = await self.client.get_order_book(symbol=symbol, limit=limit)
            return {
                "bids": result.get("bids", []),
                "asks": result.get("asks", [])
            }
        except Exception as e:
            self.logger.error(f"❌ Failed to get order book for {symbol}: {e}", exc_info=True)
            return {"bids": [], "asks": []}


    @throttle(1) # 1 / 1.0 = 1 call/second
    async def get_all_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Fetches all open orders for a specific symbol, or for all symbols if none is provided.
        This method uses the custom _send_signed_request method for explicit HMAC signing.
        """
        if self.config.PAPER_MODE:
            self.logger.info(f"[PAPER MODE] Simulating get_all_open_orders. No actual API call will be made.")
            return [] # In paper mode, we assume no open orders unless explicitly simulated elsewhere

        try:
            params = {}
            if symbol:
                params['symbol'] = symbol
            self.logger.info(f"Attempting to fetch all open orders for symbol: {symbol if symbol else 'ALL'}")

            # Endpoint for fetching open orders
            endpoint = "/api/v3/openOrders"
            open_orders = await self._send_signed_request('GET', endpoint, params)

            self.logger.info(f"✅ Successfully fetched open orders: {open_orders}")
            return open_orders
        except BinanceAPIException as e:
            self.logger.error(f"❌ Binance API error fetching open orders: {e.status_code} - {e.message}", exc_info=True)
            return []
        except BinanceRequestException as e:
            self.logger.error(f"❌ Binance request error fetching open orders: {e}", exc_info=True)
            return []
        except Exception as e:
            self.logger.error(f"🔥 An unexpected error occurred while fetching open orders: {e}", exc_info=True)
            return []

    @throttle(1) # 1 / 1.0 = 1 call/second
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Fetches open (unfilled) orders from Binance.
        If symbol is provided, filters by that symbol; otherwise fetches all open orders.
        This method uses the AsyncClient's get_open_orders() which handles signing internally.
        """
        if not self.client:
            self.logger.error("❌ Cannot fetch open orders: AsyncClient not initialized.")
            return []

        try:
            if symbol:
                self.logger.debug(f"Fetching open orders for {symbol}")
                orders = await self.client.get_open_orders(symbol=symbol)
            else:
                self.logger.debug("Fetching all open orders.")
                orders = await self.client.get_open_orders()
            self.logger.info(f"📦 Retrieved {len(orders)} open orders{' for ' + symbol if symbol else ''}.")
            return orders
        except BinanceAPIException as e:
            self.logger.error(f"❌ Binance API error fetching open orders: {e.status_code} - {e.message}", exc_info=True)
        except BinanceRequestException as e:
            self.logger.error(f"❌ Binance request error fetching open orders: {e}", exc_info=True)
        except Exception as e:
            self.logger.error(f"🔥 Unexpected error fetching open orders: {e}", exc_info=True)
        return []

    @throttle(1/300) # 1 / 300.0 = 0.0033... calls/second
    async def get_new_listings_cached(self) -> List[str]:
        """
        Returns a cached list of recently listed USDT symbols.
        You should refresh this from Binance API at startup or on interval.
        """
        # Using the new _cached_new_listings and _last_new_listing_fetch for caching logic
        now = time.time()
        if self._cached_new_listings and self._last_new_listing_fetch:
            age = now - self._last_new_listing_fetch
            if age < self._new_listing_cache_ttl:
                self.logger.info(f"[ExchangeClient] Returning cached new listings (valid for {self._new_listing_cache_ttl - age:.2f}s).")
                return self._cached_new_listings

        # Fallback to fetching if cache expired or not available
        self.logger.info("[ExchangeClient] New listings cache expired or not available. Fetching fresh data.")
        listings = await self.get_new_listings()
        self._cached_new_listings = listings
        self._last_new_listing_fetch = time.time()
        return listings

    @throttle(1/300) # 1 / 300.0 = 0.0033... calls/second
    async def get_new_listings(self) -> List[str]:
        """
        Fetches recently listed USDT symbols from Binance API using `onboardDate`.
        Falls back to last few USDT symbols if listing date is unavailable.
        """
        try:
            # Using _send_public_request as it's a public endpoint
            response = await self._send_public_request("GET", "/api/v3/exchangeInfo")
            all_symbols = response.get("symbols", [])

            now = datetime.utcnow()
            recent_symbols = []

            for symbol_info in all_symbols:
                symbol = symbol_info.get("symbol", "")
                quote = symbol_info.get("quoteAsset", "")
                status = symbol_info.get("status", "")
                listing_time = symbol_info.get("onboardDate", 0)  # In ms

                # ✅ Primary filter using onboardDate (if available)
                if (
                    quote == "USDT"
                    and status == "TRADING"
                    and symbol.endswith("USDT")
                    and listing_time > 0
                    and (now - datetime.utcfromtimestamp(listing_time / 1000)) < timedelta(hours=getattr(self.config, "IPO_LISTING_WINDOW_HOURS", 6))
                ):
                    recent_symbols.append(symbol)

            # ✅ Fallback if onboardDate not available or empty result
            if not recent_symbols and getattr(self.config, "ALLOW_FALLBACK_LISTINGS", True):
                self.logger.warning("⚠️ No IPO listings found using onboardDate. Falling back to heuristic detection.")
                fallback_symbols = sorted([
                    s["symbol"] for s in all_symbols
                    if s.get("quoteAsset") == "USDT" and s.get("status") == "TRADING" and s["symbol"].endswith("USDT")
                ])[-10:]  # Last 10 USDT symbols
                recent_symbols = fallback_symbols

            self.new_listings_cache = recent_symbols
            self.logger.info(f"[ExchangeClient] ✅ Fetched and cached {len(recent_symbols)} recent listings: {recent_symbols}")
            self.logger.info(f"[ExchangeClient] IPO listings (fresh): {recent_symbols}") # Added debug log
            return recent_symbols

        except Exception as e:
            self.logger.error(f"[ExchangeClient] ❌ Failed to fetch new listings: {e}", exc_info=True)
            return []

    # --- Heartbeat status reporting helper (Section 5) ---
    def _report_status(self, component_name: str, status: str, detail: str = ""):
        """
        Reports the status of a component to a central monitoring system (e.g., SharedState).
        This method is a placeholder; its actual implementation depends on your SharedState's design.
        """
        # Example of how it might integrate with a SharedState:
        # if self.shared_state and hasattr(self.shared_state, 'update_component_status'):
        #     self.shared_state.update_component_status(component_name, status, detail)
        # else:
        self.logger.info(f"[Heartbeat] {component_name} Status: {status} - {detail}")

