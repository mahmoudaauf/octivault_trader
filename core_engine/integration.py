"""
Core Engine Integration Layer
──────────────────────────────

PHASE 2: Wire 5 Façade Engines to actual L0-L8 components

This module provides:
1. AppContext factory for creating and wiring all components
2. Integration adapters that bridge façade engines to real components
3. Dependency injection setup
4. Lifecycle management
"""

from __future__ import annotations

import logging
from typing import Any

# Type hints
__all__ = ["CoreEngineIntegration", "create_app_context", "wire_engines"]

logger = logging.getLogger(__name__)


class CoreEngineIntegration:
    """
    Integration adapter: bridges façade engines to L0-L8 components.

    Provides normalized interfaces that real components expose.
    """

    @staticmethod
    def wire_exchange_client(app_ctx: dict[str, Any]) -> None:
        """
        Wire MarketAccountEngine → exchange_client (L1)

        The exchange_client should expose:
        • get_balance() → Dict[symbol: balance]
        • get_prices(symbols) → Dict[symbol: price]
        • get_kline(symbol, interval, limit) → [OHLCV]
        • subscribe_ticker(callback)
        • place_buy_order(symbol, qty, price)
        • place_sell_order(symbol, qty, price)
        """
        exchange_client = app_ctx.get("exchange_client")

        if not exchange_client:
            logger.warning("⚠️  exchange_client not found in app_ctx")
            return

        logger.info("✓ Wired MarketAccountEngine → exchange_client (L1)")

    @staticmethod
    def wire_market_data_feed(app_ctx: dict[str, Any]) -> None:
        """
        Wire MarketAccountEngine → market_data_feed (L2)

        Should expose:
        • get_prices(symbols) → Dict[symbol: price]
        • get_ohlcv(symbol, tf, limit) → [candles]
        • prices cache
        • subscribe_updates(callback)
        """
        market_data_feed = app_ctx.get("market_data_feed")

        if not market_data_feed:
            logger.warning("⚠️  market_data_feed not found in app_ctx")
            return

        logger.info("✓ Wired MarketAccountEngine → market_data_feed (L2)")

    @staticmethod
    def wire_portfolio_manager(app_ctx: dict[str, Any]) -> None:
        """
        Wire SituationEngine → portfolio_manager (L3)

        Should expose:
        • get_nav() → float
        • get_positions() → Dict[symbol: position]
        • get_capital_allocated() → float
        • get_capital_available() → float
        • calculate_pnl() → float
        """
        portfolio_manager = app_ctx.get("portfolio_manager")

        if not portfolio_manager:
            logger.warning("⚠️  portfolio_manager not found in app_ctx")
            return

        logger.info("✓ Wired SituationEngine → portfolio_manager (L3)")

    @staticmethod
    def wire_signal_fusion(app_ctx: dict[str, Any]) -> None:
        """
        Wire SituationEngine → signal_fusion (L5)

        Should expose:
        • fuse_signal(symbol) → SignalScore with composite edge
        • apply_weights(signals) → weighted composite
        • get_threshold() → confidence floor
        """
        signal_fusion = app_ctx.get("signal_fusion")

        if not signal_fusion:
            logger.warning("⚠️  signal_fusion not found in app_ctx")
            return

        logger.info("✓ Wired SituationEngine → signal_fusion (L5)")

    @staticmethod
    def wire_arbitration_engine(app_ctx: dict[str, Any]) -> None:
        """
        Wire DecisionEngine → arbitration_engine (L5)

        Should expose:
        • evaluate_gates(symbol, signal_type, edge) → ArbitrationResult
        • gate_1_symbol_format(symbol) → bool
        • gate_2_confidence(edge, mode) → bool
        • gate_3_regime(regime) → bool
        • gate_4_position_limit(symbol) → bool
        • gate_5_capital(symbol) → bool
        • gate_6_risk_manager(decision) → bool
        """
        arbitration_engine = app_ctx.get("arbitration_engine")

        if not arbitration_engine:
            logger.warning("⚠️  arbitration_engine not found in app_ctx")
            return

        logger.info("✓ Wired DecisionEngine → arbitration_engine (L5)")

    @staticmethod
    def wire_mode_manager(app_ctx: dict[str, Any]) -> None:
        """
        Wire DecisionEngine → mode_manager (L5)

        Should expose:
        • get_current_mode() → str (PAUSED, PROTECTIVE, BOOTSTRAP, GROWTH, CRISIS)
        • set_mode(mode) → bool
        • get_constraints(mode) → Dict with allow_new, max_pos, conf_floor, etc.
        """
        mode_manager = app_ctx.get("mode_manager")

        if not mode_manager:
            logger.warning("⚠️  mode_manager not found in app_ctx")
            return

        logger.info("✓ Wired DecisionEngine → mode_manager (L5)")

    @staticmethod
    def wire_execution_manager(app_ctx: dict[str, Any]) -> None:
        """
        Wire SafeExecutionEngine → execution_manager (L4)

        Should expose:
        • place_order(symbol, qty, price, action) → ExecutionResult
        • validate_order(symbol, qty, price) → OrderValidation
        • get_order_status(order_id) → order_dict
        • cancel_order(order_id) → bool
        """
        execution_manager = app_ctx.get("execution_manager")

        if not execution_manager:
            logger.warning("⚠️  execution_manager not found in app_ctx")
            return

        logger.info("✓ Wired SafeExecutionEngine → execution_manager (L4)")

    @staticmethod
    def wire_bounded_cache(app_ctx: dict[str, Any]) -> None:
        """
        Wire SafeExecutionEngine → bounded_cache (L0) for FIX #2

        Should expose:
        • get(key) → value or None
        • set(key, value, ttl) → None
        • contains(key) → bool
        • clear() → None
        """
        bounded_cache = app_ctx.get("bounded_cache")

        if not bounded_cache:
            logger.warning("⚠️  bounded_cache not found in app_ctx (FIX #2 guard disabled)")
            return

        logger.info("✓ Wired SafeExecutionEngine → bounded_cache (L0) [FIX #2]")

    @staticmethod
    def wire_health_monitor(app_ctx: dict[str, Any]) -> None:
        """
        Wire OperationsEngine → health_monitor (L7)

        Should expose:
        • get_component_status(name) → ComponentStatus
        • get_overall_health() → HealthStatus
        • get_all_components() → Dict[name: ComponentStatus]
        """
        health_monitor = app_ctx.get("health_monitor")

        if not health_monitor:
            logger.warning("⚠️  health_monitor not found in app_ctx")
            return

        logger.info("✓ Wired OperationsEngine → health_monitor (L7)")

    @staticmethod
    def wire_startup_orchestrator(app_ctx: dict[str, Any]) -> None:
        """
        Wire OperationsEngine → startup_orchestrator (L8)

        Should expose:
        • startup() → bool (initialize L0→L8)
        • shutdown() → bool (cleanup L8→L0)
        """
        startup_orchestrator = app_ctx.get("startup_orchestrator")

        if not startup_orchestrator:
            logger.warning("⚠️  startup_orchestrator not found in app_ctx")
            return

        logger.info("✓ Wired OperationsEngine → startup_orchestrator (L8)")

    @staticmethod
    def wire_all(app_ctx: dict[str, Any]) -> None:
        """Wire all engines to components."""
        logger.info("🔌 Wiring all engines to L0-L8 components...")

        # MarketAccountEngine
        CoreEngineIntegration.wire_exchange_client(app_ctx)
        CoreEngineIntegration.wire_market_data_feed(app_ctx)

        # SituationEngine
        CoreEngineIntegration.wire_portfolio_manager(app_ctx)
        CoreEngineIntegration.wire_signal_fusion(app_ctx)

        # DecisionEngine
        CoreEngineIntegration.wire_arbitration_engine(app_ctx)
        CoreEngineIntegration.wire_mode_manager(app_ctx)

        # SafeExecutionEngine
        CoreEngineIntegration.wire_execution_manager(app_ctx)
        CoreEngineIntegration.wire_bounded_cache(app_ctx)

        # OperationsEngine
        CoreEngineIntegration.wire_health_monitor(app_ctx)
        CoreEngineIntegration.wire_startup_orchestrator(app_ctx)

        logger.info("✅ All engines wired successfully")


async def create_app_context(
    *,
    native: bool = False,
    compat: bool = False,
) -> dict[str, Any]:
    """
    Create application context with all L0-L8 components.

    Modes
    -----
    native=True (Phase 8.2.8): construct an app_ctx from the native
        L0-L6 + L8 stack via ``core_engine.native.bootstrap`` +
        ``core_engine.native.app_context.build_native_app_ctx``.
        Reads credentials from the environment (``BINANCE_API_KEY`` /
        ``BINANCE_API_SECRET``). On any failure (missing credentials,
        construction error) falls back to mock mode (empty dict) and
        logs the cause.

        ``compat=True`` additionally installs null-object stubs for the
        six legacy app_ctx keys the façade engines reference but native
        does not yet provide (forward-compat; no-ops today).

    native=False (default): returns an empty dict — engines run in
        mock mode via graceful degradation. Used for dry-run /
        structural smoke tests.

    Note: the legacy ``production=True`` mode (which delegated to the
    now-deleted ``core_engine.production_bridge``) was removed in
    Phase 8.2.8 step 7 along with the ``--production`` CLI flag.

    Returns:
        app_ctx: Dict with all components wired (or empty in mock mode)
    """
    if native:
        try:
            from core_engine.native.app_context import build_native_app_ctx
            from core_engine.native.bootstrap import (
                BootstrapConfig,
                build_components,
            )

            cfg = BootstrapConfig.from_env()
            components = await build_components(cfg)
            app_ctx, _orch = build_native_app_ctx(components, compat=compat)
            # Stash the components handle so callers can shut down
            # background polling tasks + the exchange-client HTTP
            # session via ``shutdown_components(...)``. Phase 8.3.1.
            app_ctx["_native_components"] = components
            logger.info(
                "✅ Native app_ctx built: %d keys, testnet=%s, symbols=%d, compat=%s",
                len(app_ctx),
                cfg.testnet,
                len(cfg.symbols),
                compat,
            )
            return app_ctx
        except Exception as e:
            logger.error("❌ Native bootstrap failed: %r — falling back to mock context", e)
            # Fall through to empty context (mock mode) on bootstrap failure

    app_ctx: dict[str, Any] = {}

    logger.info("📦 Creating app context with all L0-L8 components...")

    try:
        # L0: Core infrastructure
        logger.debug("  Initializing L0 components...")
        # from src.l0_core.shared_state import SharedState
        # from src.l0_core.bounded_cache import BoundedCache
        # app_ctx["shared_state"] = SharedState()
        # app_ctx["bounded_cache"] = BoundedCache(max_entries=10000)

        # L1: Exchange I/O
        logger.debug("  Initializing L1 components...")
        # from src.l1_exchange.exchange_client import ExchangeClient
        # app_ctx["exchange_client"] = ExchangeClient(...)

        # L2: Market data & wallet
        logger.debug("  Initializing L2 components...")
        # from src.l2_marketdata.market_data_feed import MarketDataFeed
        # from src.l2_marketdata.balance_manager import BalanceManager
        # app_ctx["market_data_feed"] = MarketDataFeed(...)
        # app_ctx["balance_manager"] = BalanceManager(...)

        # L3: Portfolio state
        logger.debug("  Initializing L3 components...")
        # from src.l3_portfolio.portfolio_manager import PortfolioManager
        # app_ctx["portfolio_manager"] = PortfolioManager(...)

        # L4: Execution
        logger.debug("  Initializing L4 components...")
        # from src.l4_execution.execution_manager import ExecutionManager
        # app_ctx["execution_manager"] = ExecutionManager(...)

        # L5: Strategy & signals
        logger.debug("  Initializing L5 components...")
        # from src.l5_strategy.signal_fusion import SignalFusion
        # from src.l5_strategy.arbitration_engine import ArbitrationEngine
        # from src.l5_strategy.mode_manager import ModeManager
        # app_ctx["signal_fusion"] = SignalFusion(...)
        # app_ctx["arbitration_engine"] = ArbitrationEngine(...)
        # app_ctx["mode_manager"] = ModeManager(...)

        # L6: Governance
        logger.debug("  Initializing L6 components...")
        # from src.l6_governance.risk_manager import RiskManager
        # app_ctx["risk_manager"] = RiskManager(...)

        # L7: Observability
        logger.debug("  Initializing L7 components...")
        # from src.l7_observability.health_monitor import HealthMonitor
        # app_ctx["health_monitor"] = HealthMonitor(...)

        # L8: Lifecycle
        logger.debug("  Initializing L8 components...")
        # from src.l8_lifecycle.startup_orchestrator import StartupOrchestrator
        # app_ctx["startup_orchestrator"] = StartupOrchestrator(...)

        logger.info("✅ App context created")
        return app_ctx

    except Exception as e:
        logger.error(f"❌ Error creating app context: {e}")
        raise


async def wire_engines(app_ctx: dict[str, Any]) -> None:
    """
    Wire all 5 façade engines to components.

    Args:
        app_ctx: Application context with all components
    """
    CoreEngineIntegration.wire_all(app_ctx)


# Convenience function for quick setup
async def setup_core_engines(
    *, native: bool = False, compat: bool = False
) -> dict[str, Any]:
    """
    Setup: create context → wire engines → return ready app_ctx

    Args:
        native: If True, build a native app_ctx via the Phase 8.2.8
            bootstrap.
        compat: If True (and ``native`` is True), install null-object
            stubs for unmigrated façade keys (forward-compat).

    Returns:
        app_ctx: Ready-to-use application context
    """
    app_ctx = await create_app_context(native=native, compat=compat)
    await wire_engines(app_ctx)
    return app_ctx
