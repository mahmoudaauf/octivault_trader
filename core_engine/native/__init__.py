"""
Native module: L0 (8.2.1) + L1 (8.2.2) + L2 (8.2.3) + L3 (8.2.4) + L4 (8.2.5) + L5 (8.2.6) + L6 (8.2.7) + L8 (8.2.9)

L0 - Utilities: NativeSharedState, NativeTimeUtils, ConfigLoader, NativeRetryManager
L1 - Exchange: NativeExchangeClient, NativeBalanceSync, NativeOrderExecution
L2 - Market Data: NativeMarketData
L3 - Signals: NativeSignalEngine, Signal, AggregatedSignal
L4 - Decisions: NativeDecisionEngine, Decision, PortfolioSnapshot
L5 - Execution: NativeExecutor, ExecutionResult, ExecutionStatus
L6 - Observability: NativeTelemetry
L8 - Orchestrator: NativeOrchestrator, CycleMetrics
"""

# ── L0 ───────────────────────────────────────────────────────────────
# ── App context (8.2.8 prep) ─────────────────────────────────────────
# ── L6 ───────────────────────────────────────────────────────────────
from .adaptive_capital_engine import NativeAdaptiveCapitalEngine
from .app_context import NATIVE_CTX_KEYS, NativeComponents, build_native_app_ctx
from .arbitration_engine import NativeArbitrationEngine
from .balance_sync import NativeBalanceSync
from .balance_validator import AllocationLedgerEntry, AllocationStatus, NativeBalanceValidator

# ── Bootstrap (8.2.8) ────────────────────────────────────────────────
from .bootstrap import (
    BootstrapConfig,
    ExchangeClientFactory,
    build_components,
    shutdown_components,
)
from .capital_allocator import NativeCapitalAllocator
from .capital_policy import compute_spendable_quote, prune_reservations
from .concentration_guard import ConcentrationCheck, NativeConcentrationGuard
from .config_loader import ConfigLoader, get_config

# ── L4 ───────────────────────────────────────────────────────────────
from .decisions import Decision, NativeDecisionEngine, PortfolioSnapshot
from .daily_compounding import DailyCompoundingPolicy, DailyCompoundingState

# ── Observability (Legacy features ported) ──────────────────────────
from .error_types import (
    BootstrapError,
    ErrorCategory,
    ErrorRecovery,
    ErrorSeverity,
    ExchangeError,
    ExecutionError,
    NetworkError,
    OctiError,
    PortfolioError,
    SignalError,
    ValidationError,
)

# ── L1 ───────────────────────────────────────────────────────────────
from .exchange_client import ExchangeClientError, NativeExchangeClient
from .futures_exchange_client import (
    FuturesExchangeClientError,
    NativeFuturesExchangeClient,
)

# ── L5 ───────────────────────────────────────────────────────────────
from .executor import ExecutionResult, ExecutionStatus, NativeExecutor

# ── L3 ───────────────────────────────────────────────────────────────
from .fill_tracker import NativeFillTracker
from .health_monitor import NativeHealthMonitor

# ── L2 ───────────────────────────────────────────────────────────────
from .market_data import NativeMarketData
from .market_regime_detector import NativeMarketRegimeDetector
from .math_utils import (
    calmar_ratio,
    cumulative_returns,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    volatility,
    win_rate,
)
from .mode_manager import NativeMode, NativeModeManager

# ── L2 (Feedback Control) ────────────────────────────────────────────
from .objective_feedback_controller import NativeObjectiveFeedbackController

# ── L6 ───────────────────────────────────────────────────────────────
from .observability import NativeTelemetry

# ── L8 ───────────────────────────────────────────────────────────────
from .orchestrator import CycleMetrics, NativeOrchestrator
from .order_execution import NativeOrderExecution, OrderResult
from .prometheus_exporter import MetricsSnapshot, NativePrometheusExporter
from .regime_gate import NativeRegimeGate, RegimeDecision
from .retry_manager import (
    RETRY_AGGRESSIVE,
    RETRY_FAST,
    RETRY_NO_JITTER,
    RETRY_STANDARD,
    NativeRetryManager,
)
from .runtime_state import NativeRuntimeStateExporter, load_runtime_state
from .shared_state import NativeSharedState, Order, Position

# ── L3 ───────────────────────────────────────────────────────────────
from .signal_fusion import NativeSignalFusion
from .signals import AggregatedSignal, NativeSignalEngine, Signal
from .symbol_discovery import NativeSymbolDiscovery
from .time_utils import NativeTimeUtils
from .trade_journal import NativeTradeJournal

__all__ = [
    # L0
    "NativeSharedState",
    "DailyCompoundingPolicy",
    "DailyCompoundingState",
    "Position",
    "Order",
    "NativeTimeUtils",
    "ConfigLoader",
    "get_config",
    "NativeRetryManager",
    "RETRY_FAST",
    "RETRY_STANDARD",
    "RETRY_AGGRESSIVE",
    "RETRY_NO_JITTER",
    "NativeRuntimeStateExporter",
    "load_runtime_state",
    # L1
    "NativeExchangeClient",
    "ExchangeClientError",
    "NativeFuturesExchangeClient",
    "FuturesExchangeClientError",
    "NativeBalanceSync",
    "NativeBalanceValidator",
    "AllocationStatus",
    "AllocationLedgerEntry",
    "NativeOrderExecution",
    "OrderResult",
    # L2
    "NativeMarketData",
    "NativeMode",
    "NativeModeManager",
    "NativeMarketRegimeDetector",
    "NativeRegimeGate",
    "RegimeDecision",
    "NativeSymbolDiscovery",
    "NativeSignalFusion",
    "NativeArbitrationEngine",
    # L3
    "NativeSignalEngine",
    "Signal",
    "AggregatedSignal",
    "NativeFillTracker",
    # L4
    "NativeDecisionEngine",
    "Decision",
    "PortfolioSnapshot",
    # L5
    "NativeExecutor",
    "ExecutionResult",
    "ExecutionStatus",
    # L6
    "NativeAdaptiveCapitalEngine",
    "NativeCapitalAllocator",
    "compute_spendable_quote",
    "prune_reservations",
    "NativeConcentrationGuard",
    "ConcentrationCheck",
    "NativeTelemetry",
    "NativeHealthMonitor",
    # L2 (Feedback Control)
    "NativeObjectiveFeedbackController",
    # L8
    "NativeOrchestrator",
    "CycleMetrics",
    # App context (8.2.8 prep)
    "NATIVE_CTX_KEYS",
    "NativeComponents",
    "build_native_app_ctx",
    # Bootstrap (8.2.8)
    "BootstrapConfig",
    "ExchangeClientFactory",
    "build_components",
    "shutdown_components",
    # Observability (Legacy features ported)
    "OctiError",
    "ErrorSeverity",
    "ErrorCategory",
    "ErrorRecovery",
    "BootstrapError",
    "ExchangeError",
    "ExecutionError",
    "SignalError",
    "PortfolioError",
    "ValidationError",
    "NetworkError",
    "NativeTradeJournal",
    "MetricsSnapshot",
    "NativePrometheusExporter",
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "max_drawdown",
    "cumulative_returns",
    "win_rate",
    "profit_factor",
    "volatility",
]

__version__ = "0.8.2"
__phase__ = "8.2.8-prep"
