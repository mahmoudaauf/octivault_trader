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
from .compat import register_compat_stubs
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
    "telemetry",  # L6 (native-only)
    "portfolio_manager",  # L3 (NativePortfolioManager; replaces compat stub in 8.3.7)
    "position_manager",  # L3 (NativePositionManager; replaces compat stub in 8.3.8)
    "tp_sl_engine",  # L4 (NativeTPSLEngine; replaces compat stub in 8.3.9)
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
    balance_sync: NativeBalanceSync
    telemetry: NativeTelemetry | None = None
    portfolio_accessor: Any | None = None  # callable | None (kept loose)
    exchange_client: Any | None = None  # NativeExchangeClient | duck-typed stub
    telemetry_exporter: Any | None = None  # NativeTelemetryExporter | None
    portfolio_manager: Any | None = None  # NativePortfolioManager | None
    position_manager: Any | None = None  # NativePositionManager | None
    tp_sl_engine: Any | None = None  # NativeTPSLEngine | None


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
        If True, install null-object stubs for the six legacy keys the
        façade engines reference but native does not yet provide
        (see ``core_engine.native.compat``). Default False keeps the
        pure-factory path stub-free for unit tests.

    Returns
    -------
    (app_ctx, orchestrator)
        ``app_ctx`` exposes the keys listed in ``NATIVE_CTX_KEYS``.
        ``orchestrator`` is a fully-wired ``NativeOrchestrator`` that
        records into ``components.telemetry`` if provided.
    """
    orch = NativeOrchestrator(
        market_data=components.market_data,
        signal_engine=components.signal_engine,
        decision_engine=components.decision_engine,
        executor=components.executor,
        balance_sync=components.balance_sync,
        shared_state=components.shared_state,
        portfolio_accessor=components.portfolio_accessor,
        telemetry=components.telemetry,
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
    if components.position_manager is not None:
        app_ctx["position_manager"] = components.position_manager
    if components.tp_sl_engine is not None:
        app_ctx["tp_sl_engine"] = components.tp_sl_engine

    if compat:
        register_compat_stubs(app_ctx)

    return app_ctx, orch


__all__ = [
    "NATIVE_CTX_KEYS",
    "NativeComponents",
    "build_native_app_ctx",
]
