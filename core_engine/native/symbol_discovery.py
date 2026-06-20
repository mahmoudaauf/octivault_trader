"""
Native L0: Symbol Discovery Engine

Auto-detects tradable symbols by scanning wallet balance.
Only trades pairs where you already hold the base asset.
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class NativeSymbolDiscovery:
    """
    Auto-discovers USDT trading pairs based on wallet holdings.

    Scans your balance and creates symbol list from assets you own.
    Only generates signals for pairs you already hold (wallet-based discovery).
    """

    def __init__(
        self,
        exchange_client: Any,
        base_currency: str = "USDT",
        shared_state: Any | None = None,
        min_scan_interval_sec: float = 900.0,
        empty_scan_retry_sec: float = 300.0,
    ) -> None:
        self._client = exchange_client
        self._base = base_currency.upper()
        self._shared_state = shared_state
        self._min_scan_interval_sec = max(1.0, float(min_scan_interval_sec))
        self._empty_scan_retry_sec = max(1.0, float(empty_scan_retry_sec))
        self._last_scan_ts: float = 0.0
        self._last_empty_scan_ts: float = 0.0
        self._cached_symbols: list[str] = []

    def _fallback_symbols_from_state(self) -> list[str]:
        symbols: set[str] = set(self._cached_symbols)
        state = self._shared_state
        if state is None:
            return sorted(symbols)

        accepted = getattr(state, "accepted_symbols", set()) or set()
        symbols.update(str(sym).upper() for sym in accepted if sym)

        positions = getattr(state, "positions", {}) or {}
        symbols.update(str(sym).upper() for sym in positions if sym)

        balances = getattr(state, "balance", {}) or {}
        for asset, qty in balances.items():
            if str(asset).upper() == self._base:
                continue
            if float(qty or 0.0) > 0:
                symbols.add(f"{str(asset).upper()}{self._base}")

        return sorted(symbols)

    async def discover(self) -> list[str]:
        """
        Scan wallet balance and return trading symbols.

        Returns pairs for all non-zero holdings (excluding USDT itself).
        Example: if you hold ETH, BNB, SOL, returns [ETHUSDT, BNBUSDT, SOLUSDT]
        """
        try:
            if (
                self._shared_state is not None
                and float(getattr(self._shared_state, "exchange_throttle_until_ts", 0.0) or 0.0)
                > time.time()
            ):
                logger.warning("Persisted throttle window active; skipping wallet scan this cycle")
                return self._fallback_symbols_from_state()
            if hasattr(self._client, "is_throttled") and self._client.is_throttled():
                logger.warning("Exchange throttled; skipping wallet scan this cycle")
                return self._fallback_symbols_from_state()
            now = time.time()
            if self._cached_symbols and (now - self._last_scan_ts) < self._min_scan_interval_sec:
                logger.info(
                    "Using cached discovered symbols (%d); next wallet scan in %.0fs",
                    len(self._cached_symbols),
                    self._min_scan_interval_sec - (now - self._last_scan_ts),
                )
                return list(self._cached_symbols)
            if (
                not self._cached_symbols
                and self._last_empty_scan_ts > 0
                and (now - self._last_empty_scan_ts) < self._empty_scan_retry_sec
            ):
                logger.debug(
                    "Recent empty wallet scan; deferring retry for %.0fs",
                    self._empty_scan_retry_sec - (now - self._last_empty_scan_ts),
                )
                return self._fallback_symbols_from_state()
            balance = await self._client.get_balance()
            self._last_scan_ts = now
            if not balance:
                self._last_empty_scan_ts = now
                logger.warning("No balance data available; returning empty symbols")
                return self._fallback_symbols_from_state()

            # Find all non-zero holdings and build USDT pairs
            # Use live prices to filter out dust (< $1 notional) that causes phantom signal loops
            prices: dict = {}
            if self._shared_state is not None:
                prices = dict(getattr(self._shared_state, "prices", {}) or {})
                price_cache = dict(getattr(self._shared_state, "price_cache", {}) or {})
                prices = {**price_cache, **prices}  # live prices take precedence
            _MIN_NOTIONAL_USD = 1.0  # ignore dust below $1

            symbols: list[str] = []
            for asset, qty in sorted(balance.items()):
                # Skip USDT itself (quote asset, not tradable)
                if asset.upper() == self._base:
                    continue
                # Skip zero balances and dust
                qty_f = float(qty)
                if qty_f <= 0:
                    continue
                sym = f"{asset.upper()}{self._base}"
                price = float(prices.get(sym, 0.0) or 0.0)
                if price > 0 and qty_f * price < _MIN_NOTIONAL_USD:
                    logger.debug(
                        "[SymbolDiscovery] Skipping dust %s qty=%.8f notional=$%.4f",
                        sym, qty_f, qty_f * price,
                    )
                    continue
                symbols.append(sym)

            logger.info(
                "🔍 Wallet scan: discovered %d symbols from your holdings: %s",
                len(symbols),
                symbols,
            )
            self._cached_symbols = list(symbols)
            if not symbols:
                self._last_empty_scan_ts = now
            return symbols

        except Exception as e:
            logger.error("Failed to scan wallet balance: %s", e)
            return self._fallback_symbols_from_state()
