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
from .app_context import NATIVE_CTX_KEYS, NativeComponents, build_native_app_ctx
from .balance_sync import NativeBalanceSync

# ── Bootstrap (8.2.8) ────────────────────────────────────────────────
from .bootstrap import (
    BootstrapConfig,
    ExchangeClientFactory,
    build_components,
    shutdown_components,
)

# ── L6 ───────────────────────────────────────────────────────────────
from .capital_allocator import NativeCapitalAllocator
from .config_loader import ConfigLoader, get_config

# ── L4 ───────────────────────────────────────────────────────────────
from .decisions import Decision, NativeDecisionEngine, PortfolioSnapshot

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

# ── L5 ───────────────────────────────────────────────────────────────
from .executor import ExecutionResult, ExecutionStatus, NativeExecutor

# ── L3 ───────────────────────────────────────────────────────────────
from .fill_tracker import NativeFillTracker

# ── L2 ───────────────────────────────────────────────────────────────
from .market_data import NativeMarketData
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

# ── L6 ───────────────────────────────────────────────────────────────
from .observability import NativeTelemetry

# ── L8 ───────────────────────────────────────────────────────────────
from .orchestrator import CycleMetrics, NativeOrchestrator
from .order_execution import NativeOrderExecution, OrderResult
from .prometheus_exporter import MetricsSnapshot, NativePrometheusExporter
from .retry_manager import (
    RETRY_AGGRESSIVE,
    RETRY_FAST,
    RETRY_NO_JITTER,
    RETRY_STANDARD,
    NativeRetryManager,
)
from .shared_state import NativeSharedState, Order, Position

# ── L3 ───────────────────────────────────────────────────────────────
from .signals import AggregatedSignal, NativeSignalEngine, Signal
from .time_utils import NativeTimeUtils
from .trade_journal import NativeTradeJournal

__all__ = [
    # L0
    "NativeSharedState",
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
    # L1
    "NativeExchangeClient",
    "ExchangeClientError",
    "NativeBalanceSync",
    "NativeOrderExecution",
    "OrderResult",
    # L2
    "NativeMarketData",
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
    "NativeCapitalAllocator",
    "NativeTelemetry",
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
