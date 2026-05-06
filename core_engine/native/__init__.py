"""
Native module: L0 utilities (Phase 8.2.1) + L1 exchange/balance/orders (Phase 8.2.2)

L0 (Phase 8.2.1) — Utilities
    NativeSharedState, NativeTimeUtils, ConfigLoader, NativeRetryManager

L1 (Phase 8.2.2) — Exchange Integration
    NativeExchangeClient, NativeBalanceSync, NativeOrderExecution
"""

# ── L0 ───────────────────────────────────────────────────────────────
from .balance_sync import NativeBalanceSync
from .config_loader import ConfigLoader, get_config

# ── L1 ───────────────────────────────────────────────────────────────
from .exchange_client import ExchangeClientError, NativeExchangeClient
from .order_execution import NativeOrderExecution, OrderResult
from .retry_manager import (
    RETRY_AGGRESSIVE,
    RETRY_FAST,
    RETRY_NO_JITTER,
    RETRY_STANDARD,
    NativeRetryManager,
)
from .shared_state import NativeSharedState, Order, Position
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
]

__version__ = "0.2.0"
__phase__ = "8.2.2"
