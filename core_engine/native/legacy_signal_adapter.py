"""
Legacy Signal Adapter — Bridges legacy signal agents to native signal pipeline.

Wraps MLForecaster and SymbolScreener to produce signals compatible with
NativeSignalEngine. Handles async conversion and error resilience.

Usage::

    adapter = LegacySignalAdapter(
        ml_forecaster=ml_forecaster,
        symbol_screener=symbol_screener,
        shared_state=shared_state,
    )
    signals = await adapter.get_signals()
    # ⇒ list[dict] with fields: symbol, action, confidence, edge_score, quote, etc.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


class LegacySignalAdapter:
    """Adapts legacy signal agents (MLForecaster, SymbolScreener) to native signal format."""

    def __init__(
        self,
        ml_forecaster: Optional[Any] = None,
        symbol_screener: Optional[Any] = None,
        shared_state: Optional[Any] = None,
        timeout_sec: float = 10.0,
    ):
        """
        Initialize adapter.

        Args:
            ml_forecaster: MLForecaster instance (optional)
            symbol_screener: SymbolScreener instance (optional)
            shared_state: NativeSharedState for context
            timeout_sec: Timeout for signal generation
        """
        self._ml_forecaster = ml_forecaster
        self._symbol_screener = symbol_screener
        self._shared_state = shared_state
        self._timeout_sec = max(1.0, float(timeout_sec))

        logger.info(
            "[LegacySignalAdapter] Initialized (forecaster=%s screener=%s)",
            "✓" if ml_forecaster else "✗",
            "✓" if symbol_screener else "✗",
        )

    async def get_signals(self) -> list[dict[str, Any]]:
        """
        Get signals from legacy agents and convert to native format.

        Returns:
            List of signal dicts with: symbol, action, confidence, edge_score, quote, timestamp, source
        """
        signals: list[dict[str, Any]] = []

        # Collect from MLForecaster
        if self._ml_forecaster:
            try:
                fc_signals = await self._safely_call(
                    self._ml_forecaster.generate_signals,
                    "MLForecaster.generate_signals",
                )
                if fc_signals:
                    signals.extend(self._normalize_forecaster_signals(fc_signals))
            except Exception as e:
                logger.warning("Failed to get MLForecaster signals: %s", e)

        # SymbolScreener is discovery only, doesn't produce action signals
        # It proposes new symbols via shared_state.symbol_proposals

        if signals:
            logger.debug("[LegacySignalAdapter] Collected %d signals", len(signals))

        return signals

    async def _safely_call(self, coro_or_fn: Any, name: str) -> Any:
        """Safely call async function with timeout."""
        import asyncio

        try:
            result = coro_or_fn()
            if asyncio.iscoroutine(result):
                result = await asyncio.wait_for(result, timeout=self._timeout_sec)
            return result
        except asyncio.TimeoutError:
            logger.warning(
                "[LegacySignalAdapter] Timeout calling %s (%.1fs)", name, self._timeout_sec
            )
            return None
        except Exception as e:
            logger.debug("[LegacySignalAdapter] Error calling %s: %s", name, e)
            return None

    def _normalize_forecaster_signals(
        self, raw_signals: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Normalize MLForecaster signals to native format.

        Input fields: symbol, signal_type, confidence, expected_move, timestamp, etc.
        Output fields: symbol, action, confidence, edge_score, quote, timestamp, source
        """
        normalized: list[dict[str, Any]] = []

        for sig in raw_signals:
            if not isinstance(sig, dict):
                continue

            try:
                symbol = str(sig.get("symbol", "")).upper()
                if not symbol:
                    continue

                # MLForecaster currently emits ``action``/``side`` in buffered
                # signals, while some older paths use ``signal_type``.
                signal_type = str(
                    sig.get("signal_type", sig.get("action", sig.get("side", "BUY")))
                ).upper()
                action = "BUY" if signal_type in ("BUY", "LONG") else "SELL"

                # Confidence
                confidence = float(sig.get("confidence", 0.5))
                edge_score = float(sig.get("edge_score", confidence))

                # Expected move / quote (ML forecaster doesn't compute quote, use default)
                quote = float(sig.get("quote", 12.0))  # Default from config

                # Timestamp
                timestamp = float(sig.get("timestamp", time.time()))

                normalized_sig = {
                    "symbol": symbol,
                    "action": action,
                    "signal_type": action,  # For consistency
                    "confidence": confidence,
                    "edge_score": edge_score,
                    "edge": edge_score,  # Legacy field
                    "quote": quote,
                    "timestamp": timestamp,
                    "source": "MLForecaster",
                    # Pass through metadata for diagnostics
                    "expected_move": float(sig.get("expected_move", 0.0)),
                    "regime": str(sig.get("regime", "NORMAL")),
                }

                normalized.append(normalized_sig)
                logger.debug(
                    "[LegacySignalAdapter] Normalized %s %s (conf=%.2f)",
                    symbol,
                    action,
                    confidence,
                )

            except Exception as e:
                logger.debug("[LegacySignalAdapter] Failed to normalize signal: %s", e)
                continue

        return normalized
