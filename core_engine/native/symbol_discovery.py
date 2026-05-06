"""
Native L0: Symbol Discovery Engine

Auto-detects tradable symbols from Binance exchange info without hardcoding.
Filters by trading status, quote asset, and 24h volume.

Pattern ported from src/l3_portfolio/symbol_manager.py.
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class NativeSymbolDiscovery:
    """
    Auto-discovers USDT trading pairs from Binance exchange info.

    Filters:
    - Trading status = TRADING
    - Quote asset = USDT (configurable via base_currency)
    - Minimum 24h volume threshold
    """

    def __init__(
        self,
        exchange_client: Any,
        base_currency: str = "USDT",
        min_24h_volume: float = 100000.0,
        max_symbols: int = 30,
    ) -> None:
        self._client = exchange_client
        self._base = base_currency.upper()
        self._min_volume = min_24h_volume
        self._max_symbols = max_symbols
        self._exchange_info_cache: dict[str, dict[str, Any]] = {}
        self._cache_ts: float = 0.0
        self._cache_ttl_sec: float = 900.0

    async def discover(self) -> list[str]:
        """
        Return a list of symbols sorted by exchange status.

        Fetches exchange info once per TTL period; caches result.
        Note: Volume filtering is disabled as quoteVolume in exchange_info is not 24h volume.
        """
        await self._ensure_exchange_info()

        out: list[str] = []

        for symbol, info in self._exchange_info_cache.items():
            if not isinstance(info, dict):
                continue

            # Filter 1: Must be tradable
            if info.get("status") != "TRADING":
                continue

            # Filter 2: Spot trading allowed
            if info.get("isSpotTradingAllowed") is False:
                continue

            # Filter 3: Quote asset must match base currency
            if (info.get("quoteAsset") or "").upper() != self._base:
                continue

            out.append(symbol)

        # Sort alphabetically, cap at max_symbols
        out.sort()
        symbols = out[: self._max_symbols]

        logger.info(
            "🔍 Symbol discovery: %d tradable %s pairs, selected %d",
            len(out),
            self._base,
            len(symbols),
        )

        return symbols

    async def _ensure_exchange_info(self) -> None:
        """Fetch exchange info once per TTL period."""
        now = time.time()
        if self._exchange_info_cache and (now - self._cache_ts) < self._cache_ttl_sec:
            return

        try:
            info = await self._client.get_exchange_info()
            if not isinstance(info, dict) or "symbols" not in info:
                logger.warning("Invalid exchange info response; retaining cached data")
                return

            # Rebuild cache as {symbol: info}
            self._exchange_info_cache = {
                sym_info["symbol"]: sym_info
                for sym_info in info.get("symbols", [])
                if isinstance(sym_info, dict) and sym_info.get("symbol")
            }
            self._cache_ts = now
            logger.debug("💾 Exchange info cached: %d symbols", len(self._exchange_info_cache))
        except Exception as e:
            logger.error("Failed to fetch exchange info: %s", e)
