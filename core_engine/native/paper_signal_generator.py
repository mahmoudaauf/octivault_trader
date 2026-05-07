"""
Paper Mode Signal Generator — Auto-generates test signals for paper trading.

In paper mode, we don't have real agents running (they require external APIs,
ML models, etc.). This generator produces synthetic BUY/SELL signals based on
simple price action and random selection to bootstrap the trading system.

Used ONLY in paper mode testing to enable end-to-end cycle validation.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any

logger = logging.getLogger(__name__)


class PaperModeSignalGenerator:
    """Generates synthetic trading signals for paper mode testing."""

    def __init__(
        self,
        config: Any | None = None,
        symbols: list[str] | None = None,
        enabled: bool = True,
    ):
        """
        Initialize the paper signal generator.

        Args:
            config: Configuration object
            symbols: List of symbols to generate signals for
            enabled: Whether to generate signals (default True)
        """
        self.config = config or {}
        self.symbols = symbols or [
            "BTCUSDT",
            "ETHUSDT",
            "BNBUSDT",
            "SOLUSDT",
            "XRPUSDT",
            "ADAUSDT",
            "LINKUSDT",
            "DOGEUSDT",
            "AVAXUSDT",
            "PEPEUSDT",
        ]
        self.enabled = enabled
        self._last_signal_ts: dict[str, float] = {}
        self._signal_count = 0
        logger.info(
            "[PaperSignalGenerator] Initialized (%d symbols, enabled=%s)",
            len(self.symbols),
            enabled,
        )

    async def generate_signals(self, shared_state: Any | None = None) -> list[dict[str, Any]]:
        """
        Generate synthetic trading signals.

        Returns:
            List of signal dictionaries with symbol, action, confidence, quote
        """
        if not self.enabled:
            return []

        signals: list[dict[str, Any]] = []
        now = time.time()

        # Generate signals for 20% of symbols (2 per cycle on average for 10 symbols)
        # Reduced from 50% to avoid rate-limiting on startup. Binance enforces
        # strict cooldown on signed requests; multiple orders trigger 15s+ blocking.
        num_signals = max(1, len(self.symbols) // 5)
        selected_symbols = random.sample(self.symbols, min(num_signals, len(self.symbols)))

        for symbol in selected_symbols:
            # Rate limit: don't generate for the same symbol too frequently (1 sec for paper mode)
            last_ts = self._last_signal_ts.get(symbol, 0.0)
            if now - last_ts < 1.0:  # At least 1 second between signals per symbol
                continue

            # Random action: 60% BUY, 40% SELL (bias toward buying during growth phase)
            action = "BUY" if random.random() < 0.6 else "SELL"

            # Confidence: varies from 0.60 to 0.85 (high enough to pass thresholds)
            confidence = random.uniform(0.60, 0.85)

            # Quote amount (Binance min is $10)
            quote = float(getattr(self.config, "DEFAULT_PLANNED_QUOTE", 25.0))

            signal = {
                "symbol": symbol,
                "action": action,
                "signal_type": action,  # BUY or SELL
                "confidence": confidence,
                "edge_score": confidence,  # Use confidence as edge_score for decision engine
                "edge": confidence,  # Legacy field for fallback
                "quote": quote,
                "timestamp": now,
                "source": "PaperModeSignalGenerator",
            }

            signals.append(signal)
            self._last_signal_ts[symbol] = now
            self._signal_count += 1

            logger.debug(
                "[PaperSignalGenerator] Generated %s signal for %s (conf=%.2f, quote=%.2f)",
                action,
                symbol,
                confidence,
                quote,
            )

        if signals:
            logger.info("[PaperSignalGenerator] Generated %d signals this cycle", len(signals))

        return signals

    def get_signal_count(self) -> int:
        """Get total signals generated so far."""
        return self._signal_count

    def reset(self) -> None:
        """Reset state for testing."""
        self._last_signal_ts.clear()
        self._signal_count = 0
