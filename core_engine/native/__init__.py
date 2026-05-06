"""
Native module: L0 (8.2.1) + L1 exchange (8.2.2) + L2 market data (8.2.3) + L3 signals (8.2.4)

L0 (Phase 8.2.1) — Utilities
    NativeSharedState, NativeTimeUtils, ConfigLoader, NativeRetryManager

L1 (Phase 8.2.2) — Exchange Integration
    NativeExchangeClient, NativeBalanceSync, NativeOrderExecution

L2 (Phase 8.2.3) — Market Data
    NativeMarketData

L3 (Phase 8.2.4) — Signal Engine
    NativeSignalEngine, Signal, AggregatedSignal
"""

# ── L0 ───────────────────────────────────────────────────────────────
from .balance_sync import NativeBalanceSync
from .config_loader import ConfigLoader, get_config

# ── L1 ───────────────────────────────────────────────────────────────
from .exchange_client import ExchangeClientError, NativeExchangeClient

# ── L2 ───────────────────────────────────────────────────────────────
from .market_data import NativeMarketData
from .order_execution import NativeOrderExecution, OrderResult
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
]

__version__ = "0.4.0"
__phase__ = "8.2.4"
