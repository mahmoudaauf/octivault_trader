"""
Native L0-L8 application context (Phase 8.2.8)

This module is the **canonical** seam for assembling a native app_ctx.
It replaced the legacy ``core_engine.production_bridge`` (deleted in
Phase 8.2.8 step 6).

Scope (today)
-------------
* Pure assembly: takes already-constructed native components and
  composes them into a single ``app_ctx`` dict + a wired
  ``NativeOrchestrator``.
* Does **not** itself construct exchange clients, balance sync, or
  signal engines. That bootstrap (API keys, env-specific config) is
  the caller's responsibility — see ``PHASE_8_2_8_PREP.md`` for the
  remaining work needed before this can fully replace the legacy bridge.

Why no live bootstrap here?
---------------------------
The legacy bridge runs ``check_prerequisites() + initialize_components()``
which performs Binance auth, WS connect, etc. The native equivalent
needs a dedicated bootstrap module owning credentials and lifecycle.
Adding that here would conflate assembly with I/O setup. Keeping the
factory pure makes it trivially testable and reusable across paper /
live / backtest modes.

Usage::

    from core_engine.native import (
        NativeBalanceSync, NativeDecisionEngine, NativeExecutor,
        NativeMarketData, NativeSharedState, NativeSignalEngine,
        NativeTelemetry,
    )
    from core_engine.native.app_context import build_native_app_ctx

    components = NativeComponents(
        shared_state=NativeSharedState(),
        market_data=md,
        signal_engine=sig,
        decision_engine=dec,
        executor=exe,
        balance_sync=bsync,
        telemetry=NativeTelemetry(),
    )
    app_ctx, orch = build_native_app_ctx(components)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .balance_sync import NativeBalanceSync
from .decisions import NativeDecisionEngine
from .executor import NativeExecutor
from .market_data import NativeMarketData
from .observability import NativeTelemetry
from .orchestrator import NativeOrchestrator
from .shared_state import NativeSharedState
from .signals import NativeSignalEngine

# ----------------------------------------------------------------------
# app_ctx key contract
# ----------------------------------------------------------------------
# Public, stable keys the 5 facade engines may consume from app_ctx.
# Mirrors a *subset* of what the legacy bridge once exposed — only
# what the native stack currently provides. Missing keys remain absent;
# engines must continue to graceful-degrade.
NATIVE_CTX_KEYS: tuple[str, ...] = (
    "shared_state",  # L0
    "exchange_client",  # L1 (NativeExchangeClient; consumed by safe_execution_engine)
    "balance_manager",  # L1 (NativeBalanceSync; legacy key name preserved)
    "market_data_feed",  # L2 (NativeMarketData; legacy key name preserved)
    "signal_manager",  # L3 (NativeSignalEngine; legacy key name preserved)
    "decision_engine",  # L4 (native-only)
    "execution_manager",  # L5 (NativeExecutor; legacy key name preserved)
    "mode_manager",  # L5 strategy state machine (native)
    "signal_fusion",  # L5 native adapter over signal aggregation
    "arbitration_engine",  # L5 native adapter over gating/risk stack
    "market_regime_detector",  # L2 native market regime adapter
    "telemetry",  # L6 (native-only)
    "health_monitor",  # L7 native health monitor
    "portfolio_manager",  # L3 (NativePortfolioManager; replaces compat stub in 8.3.7)
    "capital_allocator",  # L6 (NativeCapitalAllocator; allocates capital for trades)
    "position_manager",  # L3 (NativePositionManager; replaces compat stub in 8.3.8)
    "tp_sl_engine",  # L4 (NativeTPSLEngine; replaces compat stub in 8.3.9)
    "safety_order_manager",  # L4 (NativeSafetyOrderManager; replaces compat stub in 8.3.10)
    "recovery_engine",  # L4 (NativeRecoveryEngine; replaces compat stub in 8.3.11)
    "watchdog",  # L7 (NativeWatchdog; replaces compat stub in 8.3.12 — last one)
    "trade_journal",  # L0 (NativeTradeJournal; observability — crash-safe audit trail)
    "prometheus_exporter",  # L7 (NativePrometheusExporter; observability — Grafana metrics)
    "fill_tracker",  # L3 (NativeFillTracker; detects fills and updates positions)
    "_native_orchestrator",  # L8 handle
    "_native_mode",  # marker flag
)


@dataclass(frozen=True)
class NativeComponents:
    """
    Pre-constructed native components, ready to be assembled.

    All required components are instances of the L0-L6 native classes.
    ``telemetry`` is optional but recommended.
    """

    shared_state: NativeSharedState
    market_data: NativeMarketData
    signal_engine: NativeSignalEngine
    decision_engine: NativeDecisionEngine
    executor: NativeExecutor
    balance_sync: NativeBalanceSync | None = None
    telemetry: NativeTelemetry | None = None
    polling_coordinator: Any | None = (
        None  # NativePollingCoordinator | None (legacy-style staggered polling)
    )
    portfolio_accessor: Any | None = None  # callable | None (kept loose)
    exchange_client: Any | None = None  # NativeExchangeClient | duck-typed stub
    telemetry_exporter: Any | None = None  # NativeTelemetryExporter | None
    runtime_state_exporter: Any | None = None  # NativeRuntimeStateExporter | None
    portfolio_manager: Any | None = None  # NativePortfolioManager | None
    capital_allocator: Any | None = None  # NativeCapitalAllocator | None
    position_manager: Any | None = None  # NativePositionManager | None
    tp_sl_engine: Any | None = None  # NativeTPSLEngine | None
    safety_order_manager: Any | None = None  # NativeSafetyOrderManager | None
    recovery_engine: Any | None = None  # NativeRecoveryEngine | None
    watchdog: Any | None = None  # NativeWatchdog | None
    trade_journal: Any | None = None  # NativeTradeJournal | None
    prometheus_exporter: Any | None = None  # NativePrometheusExporter | None
    fill_tracker: Any | None = None  # NativeFillTracker | None
    adaptive_capital_engine: Any | None = None  # NativeAdaptiveCapitalEngine | None
    objective_feedback_controller: Any | None = None  # NativeObjectiveFeedbackController | None
    mode_manager: Any | None = None  # NativeModeManager | None
    signal_fusion: Any | None = None  # NativeSignalFusion | None
    arbitration_engine: Any | None = None  # NativeArbitrationEngine | None
    market_regime_detector: Any | None = None  # NativeMarketRegimeDetector | None
    health_monitor: Any | None = None  # NativeHealthMonitor | None
    symbol_discovery: Any | None = None  # NativeSymbolDiscovery | None
    market_data_ws: Any | None = None  # NativeMarketDataWebSocket | None (zero API rate limits)
    position_hydration_engine: Any | None = (
        None  # NativePositionHydrationEngine | None (L0, Phase 8.4)
    )
    startup_state_machine: Any | None = None  # NativeStartupStateMachine | None (L0, Phase 8.4)
    signal_manager_bridge: Any | None = (
        None  # SignalManagerBridge | None (L3, integrates legacy + paper signals)
    )
    ml_forecaster: Any | None = None  # MLForecaster | None (legacy agent, L3)
    symbol_screener: Any | None = None  # SymbolScreener | None (legacy agent, L3)


def build_native_app_ctx(
    components: NativeComponents,
    *,
    compat: bool = False,
) -> tuple[dict[str, Any], NativeOrchestrator]:
    """
    Assemble a native ``app_ctx`` and wire a ``NativeOrchestrator``.

    Parameters
    ----------
    components
        Pre-constructed native L0-L6 instances.
    compat
        Deprecated no-op kept for signature stability. Historically this
        installed null-object stubs for the six legacy ``app_ctx`` keys
        not yet covered by native impls (``portfolio_manager``,
        ``position_manager``, ``tp_sl_engine``, ``safety_order_manager``,
        ``recovery_engine``, ``watchdog``). All six were replaced in
        Phases 8.3.7-8.3.12 and ``core_engine.native.compat`` was
        retired. The parameter remains so existing callers keep working
        unchanged; it is silently ignored.

    Returns
    -------
    (app_ctx, orchestrator)
        ``app_ctx`` exposes the keys listed in ``NATIVE_CTX_KEYS``.
        ``orchestrator`` is a fully-wired ``NativeOrchestrator`` that
        records into ``components.telemetry`` if provided.
    """
    del compat  # deprecated no-op (G5; see docstring)
    orch = NativeOrchestrator(
        market_data=components.market_data,
        signal_engine=components.signal_engine,
        decision_engine=components.decision_engine,
        executor=components.executor,
        balance_sync=components.balance_sync,
        shared_state=components.shared_state,
        portfolio_accessor=components.portfolio_accessor,
        telemetry=components.telemetry,
        watchdog=components.watchdog,
        fill_tracker=components.fill_tracker,
        tp_sl_engine=components.tp_sl_engine,
        objective_feedback_controller=components.objective_feedback_controller,
        mode_manager=components.mode_manager,
        symbol_discovery=components.symbol_discovery,
        market_data_ws=components.market_data_ws,
        polling_coordinator=components.polling_coordinator,
        position_hydration_engine=components.position_hydration_engine,
        startup_state_machine=components.startup_state_machine,
    )

    app_ctx: dict[str, Any] = {
        "shared_state": components.shared_state,
        "balance_manager": components.balance_sync,
        "market_data_feed": components.market_data,
        "signal_manager": components.signal_engine,
        "decision_engine": components.decision_engine,
        "execution_manager": components.executor,
        "_native_orchestrator": orch,
        "_native_mode": True,
    }
    if components.telemetry is not None:
        app_ctx["telemetry"] = components.telemetry
    if components.exchange_client is not None:
        app_ctx["exchange_client"] = components.exchange_client
    if components.portfolio_manager is not None:
        app_ctx["portfolio_manager"] = components.portfolio_manager
    if components.capital_allocator is not None:
        app_ctx["capital_allocator"] = components.capital_allocator
    if components.mode_manager is not None:
        app_ctx["mode_manager"] = components.mode_manager
    if components.signal_fusion is not None:
        app_ctx["signal_fusion"] = components.signal_fusion
    if components.arbitration_engine is not None:
        app_ctx["arbitration_engine"] = components.arbitration_engine
    if components.market_regime_detector is not None:
        app_ctx["market_regime_detector"] = components.market_regime_detector
    if components.health_monitor is not None:
        app_ctx["health_monitor"] = components.health_monitor
    if components.position_manager is not None:
        app_ctx["position_manager"] = components.position_manager
    if components.tp_sl_engine is not None:
        app_ctx["tp_sl_engine"] = components.tp_sl_engine
    if components.safety_order_manager is not None:
        app_ctx["safety_order_manager"] = components.safety_order_manager
    if components.recovery_engine is not None:
        app_ctx["recovery_engine"] = components.recovery_engine
    if components.watchdog is not None:
        app_ctx["watchdog"] = components.watchdog
    if components.trade_journal is not None:
        app_ctx["trade_journal"] = components.trade_journal
    if components.prometheus_exporter is not None:
        app_ctx["prometheus_exporter"] = components.prometheus_exporter
    if components.fill_tracker is not None:
        app_ctx["fill_tracker"] = components.fill_tracker
    if components.signal_manager_bridge is not None:
        app_ctx["signal_manager_bridge"] = components.signal_manager_bridge

    return app_ctx, orch


__all__ = [
    "NATIVE_CTX_KEYS",
    "NativeComponents",
    "build_native_app_ctx",
]
