"""
Signal Manager Bridge — Integrates legacy SignalManager with native stack.

Acts as a bridge between:
1. Legacy signal_manager (from src/l5_strategy/)
2. Native signal generation (PaperModeSignalGenerator in paper mode)
3. Native app_ctx and shared_state

Provides unified interface for getting signals regardless of source.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class SignalManagerBridge:
    """
    Bridges legacy SignalManager with native stack.
    Combines signals from multiple sources (legacy agents, paper mode, etc.).
    """

    def __init__(
        self,
        config: Any | None = None,
        legacy_signal_manager: Any | None = None,
        paper_generator: Any | None = None,
        shared_state: Any | None = None,
        ml_forecaster: Any | None = None,
        symbol_screener: Any | None = None,
        timeout_sec: float = 10.0,
    ):
        """
        Initialize the signal manager bridge.

        Args:
            config: Configuration object
            legacy_signal_manager: Legacy SignalManager instance (from src/)
            paper_generator: PaperModeSignalGenerator for synthetic signals
            shared_state: NativeSharedState for accessing runtime state
            ml_forecaster: MLForecaster instance (optional)
            symbol_screener: SymbolScreener instance (optional)
        """
        self.config = config or {}
        self.legacy_sm = legacy_signal_manager
        self.paper_gen = paper_generator
        self.shared_state = shared_state
        self._signal_count = 0
        self._sources_active: dict[str, bool] = {
            "legacy": bool(legacy_signal_manager),
            "paper_mode": bool(paper_generator),
            "ml_forecaster": bool(ml_forecaster),
            "symbol_screener": bool(symbol_screener),
        }

        # Initialize legacy signal adapter if agents provided
        self._adapter = None
        if ml_forecaster or symbol_screener:
            try:
                from .legacy_signal_adapter import LegacySignalAdapter

                self._adapter = LegacySignalAdapter(
                    ml_forecaster=ml_forecaster,
                    symbol_screener=symbol_screener,
                    shared_state=shared_state,
                    timeout_sec=timeout_sec,
                )
                self._sources_active["legacy_adapter"] = True
            except Exception as e:
                logger.warning("[SignalManagerBridge] Failed to initialize legacy adapter: %s", e)

        logger.info(
            "[SignalManagerBridge] Initialized (legacy=%s, paper_mode=%s, forecaster=%s, screener=%s)",
            "yes" if self.legacy_sm else "no",
            "yes" if self.paper_gen else "no",
            "yes" if ml_forecaster else "no",
            "yes" if symbol_screener else "no",
        )

    async def get_all_signals(self) -> list[dict[str, Any]]:
        """
        Get all signals from all active sources.

        Returns:
            Combined list of signals from all sources (legacy, paper mode, forecaster, etc.)
        """
        signals: list[dict[str, Any]] = []

        # Get signals from legacy signal_manager (if available)
        if self.legacy_sm and hasattr(self.legacy_sm, "get_all_signals"):
            try:
                legacy_signals = self.legacy_sm.get_all_signals()
                if legacy_signals:
                    signals.extend(legacy_signals)
                    logger.debug(
                        "[SignalManagerBridge] Got %d signals from legacy manager",
                        len(legacy_signals),
                    )
            except Exception as e:
                logger.debug("[SignalManagerBridge] Error getting legacy signals: %s", e)

        # Get signals from legacy signal adapter (MLForecaster, SymbolScreener)
        if self._adapter:
            try:
                adapter_signals = await self._adapter.get_signals()
                if adapter_signals:
                    signals.extend(adapter_signals)
                    logger.debug(
                        "[SignalManagerBridge] Got %d signals from legacy adapter",
                        len(adapter_signals),
                    )
            except Exception as e:
                logger.debug("[SignalManagerBridge] Error getting adapter signals: %s", e)

        # Get synthetic signals from paper mode generator (if available and enabled)
        if self.paper_gen and hasattr(self.paper_gen, "generate_signals"):
            try:
                paper_signals = await self.paper_gen.generate_signals(self.shared_state)
                if paper_signals:
                    signals.extend(paper_signals)
                    logger.debug(
                        "[SignalManagerBridge] Got %d signals from paper generator",
                        len(paper_signals),
                    )
            except Exception as e:
                logger.debug("[SignalManagerBridge] Error getting paper signals: %s", e)

        self._signal_count += len(signals)
        return signals

    async def get_signals_for_symbol(self, symbol: str) -> list[dict[str, Any]]:
        """
        Get all signals for a specific symbol.

        Args:
            symbol: Trading symbol

        Returns:
            List of signals for the symbol from all sources
        """
        all_signals = await self.get_all_signals()
        return [s for s in all_signals if s.get("symbol") == symbol]

    def receive_signal(self, agent_name: str, symbol: str, signal: dict[str, Any]) -> bool:
        """
        Accept a signal from an agent (for runtime signal injection).

        Args:
            agent_name: Name of the agent sending the signal
            symbol: Trading symbol
            signal: Signal dictionary

        Returns:
            True if signal was accepted, False otherwise
        """
        if not self.legacy_sm or not hasattr(self.legacy_sm, "receive_signal"):
            logger.debug(
                "[SignalManagerBridge] Ignoring signal from %s: legacy manager not available",
                agent_name,
            )
            return False

        try:
            result = self.legacy_sm.receive_signal(agent_name, symbol, signal)
            if result:
                self._signal_count += 1
            return result
        except Exception as e:
            logger.debug("[SignalManagerBridge] Error receiving signal from %s: %s", agent_name, e)
            return False

    def cleanup_expired_signals(self) -> int:
        """Clean up expired signals from legacy manager cache."""
        if not self.legacy_sm or not hasattr(self.legacy_sm, "cleanup_expired_signals"):
            return 0
        try:
            return self.legacy_sm.cleanup_expired_signals()
        except Exception as e:
            logger.debug("[SignalManagerBridge] Error cleaning up signals: %s", e)
            return 0

    def get_signal_count(self) -> int:
        """Get total signals processed by this bridge."""
        return self._signal_count

    def get_sources_status(self) -> dict[str, bool]:
        """Get status of signal sources (for monitoring)."""
        return dict(self._sources_active)
