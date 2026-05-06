"""
Native L0: Symbol Discovery Engine

Auto-detects tradable symbols by scanning wallet balance.
Only trades pairs where you already hold the base asset.
"""

from __future__ import annotations

import logging
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
    ) -> None:
        self._client = exchange_client
        self._base = base_currency.upper()

    async def discover(self) -> list[str]:
        """
        Scan wallet balance and return trading symbols.

        Returns pairs for all non-zero holdings (excluding USDT itself).
        Example: if you hold ETH, BNB, SOL, returns [ETHUSDT, BNBUSDT, SOLUSDT]
        """
        try:
            balance = await self._client.get_balance()
            if not balance:
                logger.warning("No balance data available; returning empty symbols")
                return []

            # Find all non-zero holdings and build USDT pairs
            symbols: list[str] = []
            for asset, qty in sorted(balance.items()):
                # Skip USDT itself (quote asset, not tradable)
                if asset.upper() == self._base:
                    continue
                # Skip zero balances and dust
                if float(qty) <= 0:
                    continue
                symbols.append(f"{asset.upper()}{self._base}")

            logger.info(
                "🔍 Wallet scan: discovered %d symbols from your holdings: %s",
                len(symbols),
                symbols,
            )
            return symbols

        except Exception as e:
            logger.error("Failed to scan wallet balance: %s", e)
            return []
