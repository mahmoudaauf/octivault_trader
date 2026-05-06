"""
Native L0: Core utilities module (Phase 8.2.1)

This module provides lightweight native implementations of critical L0 components,
replacing legacy code with minimal, focused implementations.

Exports:
    NativeSharedState: In-memory state management
    NativeTimeUtils: Time utilities (static methods)
    ConfigLoader: Configuration management
    NativeRetryManager: Async retry with exponential backoff
"""

from .shared_state import NativeSharedState, Position, Order
from .time_utils import NativeTimeUtils
from .config_loader import ConfigLoader, get_config
from .retry_manager import (
    NativeRetryManager,
    RETRY_FAST,
    RETRY_STANDARD,
    RETRY_AGGRESSIVE,
    RETRY_NO_JITTER,
)

__all__ = [
    # Shared State
    "NativeSharedState",
    "Position",
    "Order",
    
    # Time
    "NativeTimeUtils",
    
    # Config
    "ConfigLoader",
    "get_config",
    
    # Retry
    "NativeRetryManager",
    "RETRY_FAST",
    "RETRY_STANDARD",
    "RETRY_AGGRESSIVE",
    "RETRY_NO_JITTER",
]

__version__ = "0.1.0"
__phase__ = "8.2.1"
