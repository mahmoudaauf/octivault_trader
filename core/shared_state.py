# =============================
# core/shared_state.py — Octivault P9 SharedState
# =============================
"""
ARCHITECTURE MAINTENANCE NOTE:
When making changes to data structures, event contracts, or state management,
please update the ARCHITECTURE.md file in the project root to reflect these changes.
Key areas to review: data architecture, event bus, and component interfaces.
"""
from __future__ import annotations

# ---- Standard Library Imports ----
import asyncio
import contextlib
import logging
import time
from datetime import datetime
from decimal import getcontext
from collections import deque, defaultdict
from contextlib import asynccontextmanager
from functools import wraps
from typing import Any, Dict, List, Set, Tuple, Optional, Callable, TypedDict, TYPE_CHECKING
from dataclasses import dataclass, asdict, is_dataclass
from enum import Enum

# ---- Optional Third-Party Imports ----
try:
    import pandas as pd  # type: ignore
except ImportError:
    pd = None

# ---- Type-Only Imports (avoid circular imports) ----
if TYPE_CHECKING:
    from core.app_context import AppContext, log_structured_error

# ---- Module Metadata ----
__version__ = "2.0.1"
__component__ = "core.shared_state"
__contract_id__ = "core:SharedState:v2.0.0"

__all__ = ["SharedState", "SharedStateConfig", "HealthCode", "Component", "SharedStateError", "ErrorCode", "CircuitBreaker", "CircuitBreakerState", "PortfolioState", "BootstrapMetrics", "DustPosition", "DustRegistry", "MergeOperation", "MergeImpact", "PositionMerger", "TradeExecution", "TradingCoordinator", "OHLCVBar", "SellableLine"]

# ---- Decimal Precision ----
getcontext().prec = 28

# =============================
# Enums, Dataclasses, and Types
# =============================

class HealthCode(Enum):
    OK = "ok"
    WARN = "warn"
    ERROR = "error"

class PositionState(Enum):
    ACTIVE = "ACTIVE"
    DUST_LOCKED = "DUST_LOCKED"
    LIQUIDATING = "LIQUIDATING"

class ExecutionResult(Enum):
    FILLED = "FILLED"
    PARTIAL = "PARTIAL"
    REJECTED = "REJECTED"
    BLOCKED = "BLOCKED"

class Component(Enum):
    MARKET_DATA_FEED = "MarketDataFeed"
    EXECUTION_MANAGER = "ExecutionManager"
    META_CONTROLLER  = "MetaController"
    AGENT_MANAGER    = "AgentManager"
    RISK_MANAGER     = "RiskManager"
    PNL_CALCULATOR   = "PnLCalculator"
    PERFORMANCE_MON  = "PerformanceEvaluator"
    APP_CONTEXT      = "AppContext"

class OHLCVBar(TypedDict):
    ts: float
    o: float
    h: float
    l: float
    c: float
    v: float


# Dataclass for sellable inventory line
@dataclass
class SellableLine:
    symbol: str
    base_asset: str
    quote_asset: str
    qty: float
    est_quote_value: float
    price: float
    filters: Dict[str, Any]
    reason: str

@dataclass
class PendingPositionIntent:
    symbol: str
    side: str  # BUY | SELL
    target_quote: float
    accumulated_quote: float
    min_notional: float
    ttl_sec: int
    source_agent: str
    state: str = "ACCUMULATING"  # ACCUMULATING | READY | EXECUTED | EXPIRED
    created_at: float = 0.0

    def __post_init__(self):
        if not self.created_at:
            self.created_at = time.time()

@dataclass
class SharedStateConfig:
    max_event_log_size: int = 3000
    max_signal_buffer_size: int = 1500
    max_trade_history_size: int = 3000
    max_performance_samples: int = 500
    cache_ttl_seconds: int = 60
    price_cache_ttl_seconds: int = 300
    filter_cache_ttl_seconds: int = 1800
    reservation_default_ttl: int = 30
    memory_optimization_interval: int = 300
    wallet_sync_interval: int = 120
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_timeout: int = 60

    # --- New runtime knobs (P9 QoL) ---
    quote_asset: str = "USDT"  # Canonical quote asset for all capital evaluation
    quote_reserve_ratio: float = 0.10  # default reserve ratio for quote when computing spendable
    quote_min_reserve: float = 0.0     # hard floor for quote reserve
    auto_positions_from_balances: bool = True  # mirror wallet (non-quote) into positions
    dust_min_quote_usdt: float = 5.0   # minimum notional to treat as non-dust
    dust_liquidation_enabled: bool = True  # allow listing dust as sellable inventory
    DUST_POSITION_QTY: float = 0.0001
    liq_queue_maxsize: int = 1000      # maximum size for liquidation queue

    # --- Active symbols fallback behavior (helps agents like LiquidationAgent) ---
    active_symbols_fallback_from_positions: bool = True  # include currently held positions if accepted list is small/empty
    active_symbols_default_limit: int = 0  # 0 = unlimited; if >0, truncate get_active_symbols() output

    # --- Shadow Mode Configuration (P9 Virtual Trading) ---
    trading_mode: str = "live"  # "live" | "shadow" — when shadow, ExecutionManager simulates fills
    shadow_slippage_bps: float = 0.02  # ±2 basis points slippage in shadow mode (0.02 = 0.02%)
    shadow_min_run_rate_usdt_24h: float = 15.0  # Min run rate to allow shadow → live switch
    shadow_max_drawdown_pct: float = 0.10  # Max drawdown % (10%) to allow shadow → live switch

class ErrorCode(Enum):
    EXTERNAL_API_ERROR = "external_api_error"
    MIN_NOTIONAL_VIOLATION = "min_notional_violation"
    FEE_SAFETY_VIOLATION = "fee_safety_violation"
    RISK_CAP_EXCEEDED = "risk_cap_exceeded"
    INSUFFICIENT_BALANCE = "insufficient_balance"
    INTEGRITY_ERROR = "integrity_error"
    CONFIGURATION_ERROR = "configuration_error"
    TIMEOUT_ERROR = "timeout_error"

class CircuitBreakerState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class PortfolioState(Enum):
    """
    Portfolio state machine for dust loop elimination (Phase 1).
    Distinguishes between empty, dust-only, active, recovering, and cold bootstrap states.
    """
    EMPTY_PORTFOLIO = "EMPTY_PORTFOLIO"
    PORTFOLIO_WITH_DUST = "PORTFOLIO_WITH_DUST"
    PORTFOLIO_ACTIVE = "PORTFOLIO_ACTIVE"
    PORTFOLIO_RECOVERING = "PORTFOLIO_RECOVERING"
    COLD_BOOTSTRAP = "COLD_BOOTSTRAP"


class BootstrapMetrics:
    """
    Phase 2: Persistent storage for bootstrap metrics.
    
    Persists bootstrap history to JSON file so that restart doesn't reset
    bootstrap detection. This prevents the system from repeatedly attempting
    bootstrap after the first trade has executed.
    
    Prevents the dust loop by ensuring bootstrap only happens on true first run,
    not on every restart where metrics were lost.
    """
    
    def __init__(self, db_path: str = None):
        """
        Initialize bootstrap metrics storage.
        
        Args:
            db_path: Path to directory where bootstrap_metrics.json will be stored.
                    If None, uses current working directory.
        """
        import os
        import json
        
        self.json = json
        self.os = os
        
        if db_path is None:
            db_path = os.getcwd()
        
        self.db_path = str(db_path)
        self.metrics_file = os.path.join(self.db_path, "bootstrap_metrics.json")
        self.logger = logging.getLogger(__name__)
        
        # Load existing metrics from disk
        self._cached_metrics = self._load_or_empty()
    
    def _load_or_empty(self) -> Dict[str, Any]:
        """Load metrics from JSON file or return empty dict if not found."""
        try:
            if self.os.path.exists(self.metrics_file):
                with open(self.metrics_file, 'r') as f:
                    data = self.json.load(f)
                    self.logger.debug(f"[BootstrapMetrics] Loaded from {self.metrics_file}")
                    return data if isinstance(data, dict) else {}
        except Exception as e:
            self.logger.warning(f"[BootstrapMetrics] Failed to load metrics: {e}")
        
        return {}
    
    def _write(self, data: Dict[str, Any]) -> None:
        """Atomically write metrics to JSON file."""
        try:
            # Ensure directory exists
            self.os.makedirs(self.db_path, exist_ok=True)
            
            # Write to temp file first (atomic write)
            temp_file = self.metrics_file + ".tmp"
            with open(temp_file, 'w') as f:
                self.json.dump(data, f, indent=2)
            
            # Atomic rename
            if self.os.path.exists(self.metrics_file):
                self.os.remove(self.metrics_file)
            self.os.rename(temp_file, self.metrics_file)
            
            self.logger.debug(f"[BootstrapMetrics] Wrote to {self.metrics_file}")
        except Exception as e:
            self.logger.error(f"[BootstrapMetrics] Failed to write metrics: {e}")
    
    def save_first_trade_at(self, timestamp: float) -> None:
        """
        Record when the first trade was executed.
        
        Should be called exactly once after the first successful trade fill.
        Subsequent calls are ignored (idempotent).
        
        Args:
            timestamp: Unix timestamp of first trade
        """
        if self._cached_metrics.get("first_trade_at") is None:
            self._cached_metrics["first_trade_at"] = timestamp
            self._cached_metrics["startup_time"] = time.time()
            self._write(self._cached_metrics)
            self.logger.info(f"[BootstrapMetrics] First trade recorded at {timestamp}")
        else:
            self.logger.debug(f"[BootstrapMetrics] First trade already recorded, ignoring")
    
    def get_first_trade_at(self) -> Optional[float]:
        """
        Get the timestamp when first trade was executed.
        
        Returns:
            Unix timestamp if first trade has occurred, None otherwise
        """
        return self._cached_metrics.get("first_trade_at")
    
    def save_trade_executed(self) -> None:
        """
        Increment the total trade execution counter.
        
        Should be called after every successful trade fill.
        """
        current_count = self._cached_metrics.get("total_trades_executed", 0)
        self._cached_metrics["total_trades_executed"] = current_count + 1
        self._write(self._cached_metrics)
        self.logger.debug(f"[BootstrapMetrics] Trade count: {current_count + 1}")
    
    def get_total_trades_executed(self) -> int:
        """
        Get the total number of trades executed.
        
        Returns:
            Count of trades executed (0 if none)
        """
        return self._cached_metrics.get("total_trades_executed", 0)
    
    def reload(self) -> None:
        """
        Reload metrics from disk. Useful for testing or manual sync.
        """
        self._cached_metrics = self._load_or_empty()
        self.logger.info(f"[BootstrapMetrics] Reloaded from disk")
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """
        Get all persisted bootstrap metrics.
        
        Returns:
            Dictionary of all metrics currently stored
        """
        return dict(self._cached_metrics)


@dataclass
class DustPosition:
    """
    Represents a tracked dust position with lifecycle information.
    Used by DustRegistry to track individual dust positions.
    """
    symbol: str
    quantity: float
    notional_usd: float
    created_at: float
    status: str = "NEW"  # NEW, HEALING, HEALED, ABANDONED
    healing_attempts: int = 0
    last_healing_attempt_at: Optional[float] = None
    first_healing_attempt_at: Optional[float] = None
    healing_days_elapsed: float = 0.0
    max_healing_days: float = 30.0
    circuit_breaker_enabled: bool = False
    circuit_breaker_tripped_at: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "notional_usd": self.notional_usd,
            "created_at": self.created_at,
            "status": self.status,
            "healing_attempts": self.healing_attempts,
            "last_healing_attempt_at": self.last_healing_attempt_at,
            "first_healing_attempt_at": self.first_healing_attempt_at,
            "healing_days_elapsed": self.healing_days_elapsed,
            "max_healing_days": self.max_healing_days,
            "circuit_breaker_enabled": self.circuit_breaker_enabled,
            "circuit_breaker_tripped_at": self.circuit_breaker_tripped_at,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DustPosition":
        """Create DustPosition from dictionary."""
        return cls(**data)


class DustRegistry:
    """
    Phase 3: Persistent storage and tracking of dust position lifecycle.
    
    Tracks dust positions through their lifecycle (creation → detection → healing → resolution)
    and maintains a circuit breaker to prevent repeated healing attempts for the same position.
    
    Persists to JSON file so dust tracking survives system restart.
    Prevents the dust loop by ensuring dust isn't repeatedly healed without progress.
    """
    
    def __init__(self, db_path: str = None):
        """
        Initialize dust registry.
        
        Args:
            db_path: Path to directory where dust_registry.json will be stored.
                    If None, uses current working directory.
        """
        import os
        import json
        
        self.json = json
        self.os = os
        
        if db_path is None:
            db_path = os.getcwd()
        
        self.db_path = str(db_path)
        self.registry_file = os.path.join(self.db_path, "dust_registry.json")
        self.logger = logging.getLogger(__name__)
        
        # Load existing registry from disk
        self._cached_registry = self._load_or_empty()
    
    def _load_or_empty(self) -> Dict[str, Any]:
        """Load registry from JSON file or return empty dict if not found."""
        try:
            if self.os.path.exists(self.registry_file):
                with open(self.registry_file, 'r') as f:
                    data = self.json.load(f)
                    self.logger.debug(f"[DustRegistry] Loaded from {self.registry_file}")
                    return data if isinstance(data, dict) else {"dust_positions": {}, "metadata": {}}
        except Exception as e:
            self.logger.warning(f"[DustRegistry] Failed to load registry: {e}")
        
        return {"dust_positions": {}, "metadata": {}}
    
    def _write(self, data: Dict[str, Any]) -> None:
        """Atomically write registry to JSON file."""
        try:
            # Ensure directory exists
            self.os.makedirs(self.db_path, exist_ok=True)
            
            # Write to temp file first (atomic write)
            temp_file = self.registry_file + ".tmp"
            with open(temp_file, 'w') as f:
                self.json.dump(data, f, indent=2)
            
            # Atomic rename
            self.os.replace(temp_file, self.registry_file)
            self.logger.debug(f"[DustRegistry] Persisted to {self.registry_file}")
        except Exception as e:
            self.logger.error(f"[DustRegistry] Failed to write registry: {e}")
    
    def mark_position_as_dust(self, symbol: str, quantity: float, notional_usd: float) -> None:
        """
        Record a position as dust.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC')
            quantity: Quantity of the position
            notional_usd: USD value of the position
        """
        if "dust_positions" not in self._cached_registry:
            self._cached_registry["dust_positions"] = {}
        
        # Only create new entry if not already tracked
        if symbol not in self._cached_registry["dust_positions"]:
            dust_pos = DustPosition(
                symbol=symbol,
                quantity=quantity,
                notional_usd=notional_usd,
                created_at=time.time(),
                status="NEW"
            )
            self._cached_registry["dust_positions"][symbol] = dust_pos.to_dict()
            self._write(self._cached_registry)
            self.logger.info(f"[DustRegistry] Marked {symbol} as dust (${notional_usd:.2f})")
    
    def mark_healing_started(self, symbol: str) -> None:
        """Record that healing attempt started for a dust position."""
        if "dust_positions" not in self._cached_registry:
            return
        
        if symbol in self._cached_registry["dust_positions"]:
            pos = self._cached_registry["dust_positions"][symbol]
            pos["status"] = "HEALING"
            if pos.get("first_healing_attempt_at") is None:
                pos["first_healing_attempt_at"] = time.time()
            pos["last_healing_attempt_at"] = time.time()
            self._write(self._cached_registry)
    
    def record_healing_attempt(self, symbol: str) -> None:
        """Increment healing attempt counter."""
        if "dust_positions" not in self._cached_registry:
            return
        
        if symbol in self._cached_registry["dust_positions"]:
            pos = self._cached_registry["dust_positions"][symbol]
            pos["healing_attempts"] = pos.get("healing_attempts", 0) + 1
            pos["last_healing_attempt_at"] = time.time()
            
            # Calculate days elapsed since first healing attempt
            if pos.get("first_healing_attempt_at"):
                elapsed = time.time() - pos["first_healing_attempt_at"]
                pos["healing_days_elapsed"] = elapsed / (24 * 3600)
            
            self._write(self._cached_registry)
    
    def mark_healing_complete(self, symbol: str) -> None:
        """Mark dust position as successfully healed."""
        if "dust_positions" not in self._cached_registry:
            return
        
        if symbol in self._cached_registry["dust_positions"]:
            pos = self._cached_registry["dust_positions"][symbol]
            pos["status"] = "HEALED"
            self._write(self._cached_registry)
            self.logger.info(f"[DustRegistry] Marked {symbol} as HEALED")
    
    def should_attempt_healing(self, symbol: str) -> bool:
        """
        Check if healing should be attempted for a dust position.
        
        Returns False if:
        - Position not tracked as dust
        - Circuit breaker is tripped
        - Already healed
        
        Returns True if healing should be attempted.
        """
        if "dust_positions" not in self._cached_registry:
            return False
        
        if symbol not in self._cached_registry["dust_positions"]:
            return False
        
        pos = self._cached_registry["dust_positions"][symbol]
        
        # Don't attempt if already healed
        if pos.get("status") == "HEALED":
            return False
        
        # Don't attempt if circuit breaker is tripped
        if pos.get("circuit_breaker_enabled") and pos.get("circuit_breaker_tripped_at") is not None:
            return False
        
        return True
    
    def trip_circuit_breaker(self, symbol: str) -> None:
        """
        Prevent further healing attempts for this dust position.
        Circuit breaker is tripped when healing appears ineffective.
        """
        if "dust_positions" not in self._cached_registry:
            return
        
        if symbol in self._cached_registry["dust_positions"]:
            pos = self._cached_registry["dust_positions"][symbol]
            pos["circuit_breaker_enabled"] = True
            pos["circuit_breaker_tripped_at"] = time.time()
            self._write(self._cached_registry)
            self.logger.warning(f"[DustRegistry] Circuit breaker TRIPPED for {symbol}")
    
    def reset_circuit_breaker(self, symbol: str) -> None:
        """Reset circuit breaker to allow healing to resume."""
        if "dust_positions" not in self._cached_registry:
            return
        
        if symbol in self._cached_registry["dust_positions"]:
            pos = self._cached_registry["dust_positions"][symbol]
            pos["circuit_breaker_tripped_at"] = None
            self._write(self._cached_registry)
            self.logger.info(f"[DustRegistry] Circuit breaker RESET for {symbol}")
    
    def is_circuit_breaker_tripped(self, symbol: str) -> bool:
        """Check if circuit breaker is tripped for this position."""
        if "dust_positions" not in self._cached_registry:
            return False
        
        if symbol not in self._cached_registry["dust_positions"]:
            return False
        
        pos = self._cached_registry["dust_positions"][symbol]
        return pos.get("circuit_breaker_enabled", False) and pos.get("circuit_breaker_tripped_at") is not None
    
    def is_dust_position(self, symbol: str) -> bool:
        """Check if a position is tracked as dust."""
        if "dust_positions" not in self._cached_registry:
            return False
        return symbol in self._cached_registry["dust_positions"]
    
    def get_dust_status(self, symbol: str) -> Optional[str]:
        """Get current status of dust position."""
        if "dust_positions" not in self._cached_registry:
            return None
        
        if symbol in self._cached_registry["dust_positions"]:
            return self._cached_registry["dust_positions"][symbol].get("status")
        
        return None
    
    def get_healing_attempts(self, symbol: str) -> int:
        """Get number of healing attempts for dust position."""
        if "dust_positions" not in self._cached_registry:
            return 0
        
        if symbol in self._cached_registry["dust_positions"]:
            return self._cached_registry["dust_positions"][symbol].get("healing_attempts", 0)
        
        return 0
    
    def get_dust_position_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get all tracking data for a dust position."""
        if "dust_positions" not in self._cached_registry:
            return None
        
        if symbol in self._cached_registry["dust_positions"]:
            return dict(self._cached_registry["dust_positions"][symbol])
        
        return None
    
    def mark_healed(self, symbol: str) -> None:
        """Remove dust position from tracking after successful healing."""
        if "dust_positions" not in self._cached_registry:
            return
        
        if symbol in self._cached_registry["dust_positions"]:
            # Mark as HEALED but don't delete (keep history)
            self._cached_registry["dust_positions"][symbol]["status"] = "HEALED"
            self._write(self._cached_registry)
    
    def cleanup_abandoned_dust(self, days_threshold: float = 30.0) -> List[str]:
        """
        Remove dust that hasn't improved in N days.
        
        Returns list of symbols that were cleaned up.
        """
        if "dust_positions" not in self._cached_registry:
            return []
        
        cleaned = []
        symbols_to_remove = []
        
        for symbol, pos in self._cached_registry["dust_positions"].items():
            # Check if dust has been healing for longer than threshold without being healed
            if pos.get("status") == "HEALING":
                healing_days = pos.get("healing_days_elapsed", 0.0)
                if healing_days > days_threshold:
                    symbols_to_remove.append(symbol)
                    cleaned.append(symbol)
                    self.logger.info(f"[DustRegistry] Cleaning up abandoned dust: {symbol} ({healing_days:.1f} days)")
        
        # Remove cleaned symbols
        for symbol in symbols_to_remove:
            pos = self._cached_registry["dust_positions"][symbol]
            pos["status"] = "ABANDONED"
        
        if cleaned:
            self._write(self._cached_registry)
        
        return cleaned
    
    def get_all_dust_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get all tracked dust positions."""
        if "dust_positions" not in self._cached_registry:
            return {}
        return dict(self._cached_registry.get("dust_positions", {}))
    
    def get_dust_summary(self) -> Dict[str, Any]:
        """Get summary statistics of dust registry."""
        if "dust_positions" not in self._cached_registry:
            return {
                "total_dust_symbols": 0,
                "total_dust_notional": 0.0,
                "by_status": {},
            }
        
        positions = self._cached_registry.get("dust_positions", {})
        
        summary = {
            "total_dust_symbols": len(positions),
            "total_dust_notional": 0.0,
            "by_status": {
                "NEW": 0,
                "HEALING": 0,
                "HEALED": 0,
                "ABANDONED": 0,
            },
            "circuit_breakers_tripped": 0,
        }
        
        for symbol, pos in positions.items():
            summary["total_dust_notional"] += pos.get("notional_usd", 0.0)
            status = pos.get("status", "NEW")
            if status in summary["by_status"]:
                summary["by_status"][status] += 1
            if pos.get("circuit_breaker_tripped_at") is not None:
                summary["circuit_breakers_tripped"] += 1
        
        return summary
    
    def reload(self) -> None:
        """Manually reload registry from disk."""
        self._cached_registry = self._load_or_empty()
        self.logger.info(f"[DustRegistry] Reloaded from disk")
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all persisted dust registry data."""
        return dict(self._cached_registry)


@dataclass
class MergeOperation:
    """
    Represents a merge operation between positions or orders.
    Used by PositionMerger to track consolidation operations.
    """
    symbol: str
    source_quantity: float
    target_quantity: float
    source_entry_price: float
    target_entry_price: float
    merged_quantity: float
    merged_entry_price: float
    merge_type: str = "POSITION_MERGE"  # POSITION_MERGE, ORDER_MERGE, CONSOLIDATION
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "symbol": self.symbol,
            "source_quantity": self.source_quantity,
            "target_quantity": self.target_quantity,
            "source_entry_price": self.source_entry_price,
            "target_entry_price": self.target_entry_price,
            "merged_quantity": self.merged_quantity,
            "merged_entry_price": self.merged_entry_price,
            "merge_type": self.merge_type,
            "timestamp": self.timestamp,
        }


@dataclass
class MergeImpact:
    """
    Analysis of merge impact on position metrics.
    """
    symbol: str
    cost_basis_change: float
    new_average_entry: float
    quantity_change: float
    order_count_reduction: int
    estimated_slippage: float = 0.0
    feasibility_score: float = 1.0  # 0-1, higher = better merge
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "cost_basis_change": self.cost_basis_change,
            "new_average_entry": self.new_average_entry,
            "quantity_change": self.quantity_change,
            "order_count_reduction": self.order_count_reduction,
            "estimated_slippage": self.estimated_slippage,
            "feasibility_score": self.feasibility_score,
        }


class PositionMerger:
    """
    Phase 4: Position consolidation and merging.
    
    Analyzes fragmented positions and consolidates them to reduce complexity,
    lower trading costs, and improve capital efficiency.
    
    Prevents the dust loop by automatically merging dust fragments into
    consolidated positions before trading.
    """
    
    def __init__(self):
        """Initialize position merger."""
        self.logger = logging.getLogger(__name__)
        self.merge_history: List[MergeOperation] = []
        self.merge_threshold_usd = 1.0  # Minimum notional for merge
        self.max_entry_price_deviation = 0.05  # 5% max deviation
    
    def identify_merge_candidates(self, positions: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
        """
        Identify positions that are candidates for merging.
        
        Groups positions by symbol and returns symbols with multiple positions.
        
        Args:
            positions: Dict of symbol -> position details
            
        Returns:
            Dict mapping symbol -> list of position IDs that can be merged
        """
        candidates = {}
        
        # Group by symbol
        symbols_with_multiples = {}
        for pos_id, pos_data in positions.items():
            symbol = pos_data.get("symbol", "")
            if symbol:
                if symbol not in symbols_with_multiples:
                    symbols_with_multiples[symbol] = []
                symbols_with_multiples[symbol].append(pos_id)
        
        # Find symbols with multiple positions
        for symbol, pos_ids in symbols_with_multiples.items():
            if len(pos_ids) > 1:
                candidates[symbol] = pos_ids
        
        return candidates
    
    def calculate_weighted_entry_price(self, positions: List[Dict[str, Any]]) -> float:
        """
        Calculate volume-weighted average entry price for positions.
        
        Args:
            positions: List of position dicts with quantity and entry_price
            
        Returns:
            Weighted average entry price
        """
        total_notional = 0.0
        total_quantity = 0.0
        
        for pos in positions:
            qty = abs(pos.get("quantity", 0.0))
            entry = pos.get("entry_price", 0.0)
            notional = qty * entry
            total_notional += notional
            total_quantity += qty
        
        if total_quantity == 0:
            return 0.0
        
        return total_notional / total_quantity
    
    def calculate_merge_impact(self, symbol: str, positions: List[Dict[str, Any]]) -> MergeImpact:
        """
        Calculate the impact of merging positions.
        
        Args:
            symbol: Trading symbol
            positions: List of positions to merge
            
        Returns:
            MergeImpact with analysis
        """
        if not positions:
            return MergeImpact(
                symbol=symbol,
                cost_basis_change=0.0,
                new_average_entry=0.0,
                quantity_change=0.0,
                order_count_reduction=0
            )
        
        # Calculate merged quantity and entry
        total_quantity = sum(abs(p.get("quantity", 0.0)) for p in positions)
        current_avg_entry = self.calculate_weighted_entry_price(positions)
        
        # Calculate cost basis change
        original_cost = sum(abs(p.get("quantity", 0.0)) * p.get("entry_price", 0.0) for p in positions)
        merged_cost = total_quantity * current_avg_entry
        cost_basis_change = merged_cost - original_cost
        
        # Order reduction
        order_count_reduction = len(positions) - 1
        
        # Estimate slippage as percentage of notional (assume 0.1% per order merged)
        # So for 2 orders, slippage = 0.1% total, for 3 orders = 0.2%, etc.
        slippage_percentage = order_count_reduction * 0.001
        estimated_slippage = (total_quantity * current_avg_entry) * slippage_percentage
        
        # Feasibility score (0-1): higher is better
        # Based on: position count, total quantity, deviation consistency
        position_score = min(len(positions) / 5.0, 1.0)  # More positions = higher score
        # Use 1.0 as base quantity threshold (1 unit or more is good)
        quantity_score = min(total_quantity / 1.0, 1.0) if total_quantity > 0 else 0.0
        
        # Check entry price consistency
        entry_prices = [p.get("entry_price", 0.0) for p in positions]
        max_price = max(entry_prices) if entry_prices else 0.0
        min_price = min(entry_prices) if entry_prices else 0.0
        
        if max_price > 0:
            deviation = (max_price - min_price) / max_price
            consistency_score = max(0.0, 1.0 - deviation)
        else:
            consistency_score = 1.0
        
        feasibility_score = (position_score + quantity_score + consistency_score) / 3.0
        
        return MergeImpact(
            symbol=symbol,
            cost_basis_change=cost_basis_change,
            new_average_entry=current_avg_entry,
            quantity_change=total_quantity,
            order_count_reduction=order_count_reduction,
            estimated_slippage=estimated_slippage,
            feasibility_score=feasibility_score
        )
    
    def validate_merge(self, position1: Dict[str, Any], position2: Dict[str, Any]) -> bool:
        """
        Validate that two positions can be safely merged.
        
        Args:
            position1: First position
            position2: Second position
            
        Returns:
            True if merge is valid, False otherwise
        """
        # Check symbols match
        if position1.get("symbol") != position2.get("symbol"):
            return False
        
        # Check entry prices are compatible
        entry1 = position1.get("entry_price", 0.0)
        entry2 = position2.get("entry_price", 0.0)
        
        if entry1 == 0 or entry2 == 0:
            return False
        
        deviation = abs(entry1 - entry2) / max(entry1, entry2)
        if deviation > self.max_entry_price_deviation:
            self.logger.warning(f"[PositionMerger] Entry price deviation too high: {deviation:.2%}")
            return False
        
        # Both must have valid quantities
        qty1 = position1.get("quantity", 0.0)
        qty2 = position2.get("quantity", 0.0)
        if qty1 == 0 or qty2 == 0:
            return False
        
        return True
    
    def merge_positions(self, symbol: str, positions: List[Dict[str, Any]]) -> Optional[MergeOperation]:
        """
        Merge multiple positions into a single consolidated position.
        
        Args:
            symbol: Trading symbol
            positions: List of positions to merge
            
        Returns:
            MergeOperation if successful, None otherwise
        """
        if len(positions) < 2:
            return None
        
        # Validate all positions can be merged
        for i in range(len(positions) - 1):
            if not self.validate_merge(positions[i], positions[i + 1]):
                self.logger.warning(f"[PositionMerger] Cannot merge {symbol}: validation failed")
                return None
        
        # Calculate merged position
        source_qty = positions[0].get("quantity", 0.0)
        source_price = positions[0].get("entry_price", 0.0)
        
        total_qty = sum(abs(p.get("quantity", 0.0)) for p in positions)
        merged_price = self.calculate_weighted_entry_price(positions)
        
        # Create merge operation
        operation = MergeOperation(
            symbol=symbol,
            source_quantity=source_qty,
            target_quantity=sum(abs(p.get("quantity", 0.0)) for p in positions[1:]),
            source_entry_price=source_price,
            target_entry_price=positions[1].get("entry_price", 0.0),
            merged_quantity=total_qty,
            merged_entry_price=merged_price,
            merge_type="POSITION_MERGE"
        )
        
        # Track operation
        self.merge_history.append(operation)
        self.logger.info(f"[PositionMerger] Merged {len(positions)} positions for {symbol}: "
                        f"{total_qty} @ {merged_price:.2f}")
        
        return operation
    
    def should_merge(self, symbol: str, positions: List[Dict[str, Any]]) -> bool:
        """
        Determine if positions should be merged.
        
        Args:
            symbol: Trading symbol
            positions: List of positions
            
        Returns:
            True if positions should be merged
        """
        if len(positions) < 2:
            return False
        
        # Calculate impact
        impact = self.calculate_merge_impact(symbol, positions)
        
        # Merge if:
        # 1. Feasibility score > 0.6
        # 2. Cost basis change < 1% of position notional
        # 3. Slippage < 0.5% of position notional
        if impact.feasibility_score < 0.6:
            return False
        
        total_notional = impact.quantity_change * impact.new_average_entry
        if total_notional > 0:
            cost_pct = abs(impact.cost_basis_change) / total_notional
            if cost_pct > 0.01:
                return False
            
            slippage_pct = impact.estimated_slippage / total_notional
            if slippage_pct > 0.005:
                return False
        
        return True
    
    def consolidate_dust(self, symbol: str, positions: List[Dict[str, Any]], 
                        dust_threshold: float = 1.0) -> Optional[MergeOperation]:
        """
        Consolidate dust positions (small positions) into a single position.
        
        Args:
            symbol: Trading symbol
            positions: List of positions for this symbol
            dust_threshold: Notional value threshold for dust ($)
            
        Returns:
            MergeOperation if successful
        """
        dust_positions = []
        
        for pos in positions:
            qty = abs(pos.get("quantity", 0.0))
            entry = pos.get("entry_price", 0.0)
            notional = qty * entry
            
            if notional < dust_threshold:
                dust_positions.append(pos)
        
        if len(dust_positions) < 2:
            return None
        
        return self.merge_positions(symbol, dust_positions)
    
    def get_merge_summary(self) -> Dict[str, Any]:
        """Get summary of all merge operations."""
        if not self.merge_history:
            return {
                "total_merges": 0,
                "symbols_merged": 0,
                "total_quantity_consolidated": 0.0,
                "orders_eliminated": 0,
            }
        
        return {
            "total_merges": len(self.merge_history),
            "symbols_merged": len(set(m.symbol for m in self.merge_history)),
            "total_quantity_consolidated": sum(m.merged_quantity for m in self.merge_history),
            "orders_eliminated": sum(m.merged_quantity for m in self.merge_history),
            "last_merge": self.merge_history[-1].to_dict(),
        }
    
    def reset_history(self) -> None:
        """Clear merge history."""
        self.merge_history.clear()
        self.logger.info("[PositionMerger] History cleared")

@dataclass
class CircuitBreaker:
    failure_count: int = 0
    last_failure_time: float = 0
    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    failure_threshold: int = 5
    timeout: int = 60
    def should_allow_request(self) -> bool:
        if self.state == CircuitBreakerState.CLOSED:
            return True
        if self.state == CircuitBreakerState.OPEN:
            if time.time() - self.last_failure_time > self.timeout:
                self.state = CircuitBreakerState.HALF_OPEN
                return True
            return False
        return True
    def record_success(self) -> None:
        self.failure_count = 0
        self.state = CircuitBreakerState.CLOSED
    def record_failure(self) -> None:
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN

class SharedStateError(Exception):
    def __init__(self, message: str, error_code: ErrorCode = ErrorCode.INTEGRITY_ERROR):
        super().__init__(message)
        self.error_code = error_code

# =============================
# Utility Functions & Decorators
# =============================

def track_performance(method):
    @wraps(method)
    async def wrapper(self, *args, **kwargs):
        if not hasattr(self, "_performance_stats"):
            return await method(self, *args, **kwargs)
        start = time.time()
        try:
            return await method(self, *args, **kwargs)
        finally:
            dt = time.time() - start
            name = method.__name__
            samples = self._performance_stats["method_call_times"][name]
            samples.append(dt)
            max_samples = self.config.max_performance_samples
            if len(samples) > max_samples:
                self._performance_stats["method_call_times"][name] = samples[-max_samples//2:]
    return wrapper

# =============================
# Module-level utility functions
# =============================

async def _safe_await(maybe):
    """Await the value if it's awaitable, otherwise return it directly."""
    if asyncio.iscoroutine(maybe):
        return await maybe
    return maybe


class _SharedStateEventBus:
    """
    Thin topic bus adapter over SharedState's event log/queue fanout.
    Exposes the explicit API requested by strategist routing:
      - publish(topic, payload)
      - subscribe(subscriber_name, max_queue)
      - unsubscribe(subscriber_name)
    """

    def __init__(self, shared_state: "SharedState"):
        self._ss = shared_state

    def _to_event_dict(self, payload: Any) -> Dict[str, Any]:
        if payload is None:
            return {}
        if isinstance(payload, dict):
            return dict(payload)
        if is_dataclass(payload):
            try:
                return asdict(payload)
            except Exception:
                return dict(getattr(payload, "__dict__", {}) or {})
        if hasattr(payload, "__dict__"):
            return dict(getattr(payload, "__dict__", {}) or {})
        return {"payload": payload}

    async def publish(self, topic: str, payload: Any) -> None:
        data = self._to_event_dict(payload)
        if "ts" not in data and "timestamp" not in data:
            data["ts"] = time.time()
        await self._ss.emit_event(str(topic or ""), data)

    async def subscribe(self, subscriber_name: str, max_queue: int = 1000) -> asyncio.Queue:
        return await self._ss.subscribe_events(subscriber_name, max_queue=max_queue)

    async def unsubscribe(self, subscriber_name: str) -> None:
        await self._ss.unsubscribe(subscriber_name)

class SharedState:

    # ---- Compatibility helpers (TPSLEngine/Watchdog) ----
    def update_timestamp(self, component: str) -> None:
        """
        Best-effort timestamp updater for component freshness tracking.
        Updates component_statuses and lightweight last-seen mirrors.
        """
        try:
            ts = time.time()
            payload = self.component_statuses.get(component) or {"status": "Unknown", "message": "", "timestamp": ts}
            payload["timestamp"] = ts
            self.component_statuses[component] = payload
            self.component_last_seen[component] = ts
            self.timestamps[component] = ts
        except Exception:
            pass

    def get_accepted_symbols_ready_event(self):
        return self.accepted_symbols_ready_event

    def get_market_data_ready_event(self):
        return self.market_data_ready_event

    def get_market_data_for_symbol(self, symbol: str) -> Dict[str, Dict[str, Any]]:
        """
        Normalize market_data to the {tf: {ohlcv: [...]}} shape expected by TPSLEngine.
        Supports either dict-by-symbol or tuple-keyed (symbol, tf) storage.
        """
        sym = str(symbol).upper()
        md = getattr(self, "market_data", {}) or {}
        # If already in symbol-keyed format
        if isinstance(md, dict) and sym in md and isinstance(md.get(sym), dict):
            return md.get(sym) or {}
        # Tuple-keyed format: {(symbol, tf): ohlcv}
        out: Dict[str, Dict[str, Any]] = {}
        if isinstance(md, dict):
            for (k, v) in md.items():
                try:
                    if isinstance(k, tuple) and len(k) >= 2:
                        k_sym, k_tf = str(k[0]).upper(), str(k[1])
                        if k_sym == sym:
                            out.setdefault(k_tf, {})["ohlcv"] = v
                except Exception:
                    continue
        return out
    
    def set_profit_guard(self, guard_fn: Callable[[Dict[str, Any]], Any]) -> None:
        """Register the ProfitTargetEngine's global check."""
        self._profit_guard = guard_fn
        
    async def profit_target_ok(self, min_usdt_per_hour: float = 10.0) -> bool:
        """
        Global profit target check used by ExecutionManager.
        Delegates to _profit_guard if present.
        """
        if self._profit_guard:
            try:
                # We await it if it's async (which it is: check_global_compliance)
                res = self._profit_guard({"min_usdt_per_hour": min_usdt_per_hour})
                if hasattr(res, "__await__"):
                    return await res
                return bool(res)
            except Exception as e:
                self.logger.warning(f"Profit guard check failed: {e}")
                return True # Fail open
        return True # Fail open if no guard

    # Synchronous fallback for status reporting (for threads/no event loop)
    def update_system_health(self, component: str, status: str, message: str = "", detail: str | None = None):
        ts = time.time()
        payload = {"status": status, "message": message or (detail or ""), "timestamp": ts}
        self.component_statuses[component] = payload
        self.system_health = payload
        self.metrics["last_health_update"] = ts
        # Update ops-plane readiness on first healthy reports (sync path)
        cname = str(component)
        status_l = str(status).lower()
        if cname in ("PnLCalculator", "PerformanceEvaluator") and status_l in ("ok", "healthy", "running", "operational"):
            self._ops_first_report[cname] = True
            # best-effort: schedule async check without awaiting (since we're in sync context)
            try:
                coro = self._maybe_set_ops_plane_ready()
                loop = asyncio.get_running_loop()
                loop.create_task(coro)
            except RuntimeError:
                # no running loop available; skip scheduling
                pass
            except Exception:
                pass
        # No await here (sync context) — it's fine to skip emit_event in this path

    # ---- Component status API (CSL + Watchdog friendly) ----
    async def update_component_status(self, component: str, status: str, detail: str = "", *, timestamp: float | None = None):
        ts = float(timestamp or time.time())
        payload = {"status": status, "message": detail, "timestamp": ts}
        async with self._lock_context("global"):
            self.component_statuses[component] = payload
            # keep a lightweight system mirror (optional)
            self.system_health = payload
            self.metrics["last_health_update"] = ts
        await self.emit_event("HealthStatus", {"component": component, **payload})
        # Ops-plane readiness tracking for first healthy reports
        cname = str(component)
        status_l = str(status).lower()
        if cname in ("PnLCalculator", "PerformanceEvaluator") and status_l in ("ok", "healthy", "running", "operational"):
            self._ops_first_report[cname] = True
            await self._maybe_set_ops_plane_ready()

    async def register_component(self, component: str, initial_status: str = "Initialized", detail: str = "Registered"):
        """Simple wrapper to register a component and set its initial status."""
        await self.update_component_status(component, initial_status, detail)

    # back-compat alias used by CSL
    async def set_component_status(self, component: str, status: str, detail: str, *, timestamp: float | None = None):
        await self.update_component_status(component, status, detail, timestamp=timestamp)

    # snapshot reader used by Watchdog (if present)
    def get_component_status_snapshot(self) -> dict:
        # return a shallow copy so readers can index directly by component name
        return dict(self.component_statuses)


    async def _maybe_set_ops_plane_ready(self) -> None:
        if self._ops_first_report.get("PnLCalculator") and self._ops_first_report.get("PerformanceEvaluator"):
            if not self.ops_plane_ready_event.is_set():
                self.ops_plane_ready_event.set()
                self.metrics["ops_plane_ready_at"] = time.time()
                await self.emit_event("OpsPlaneReady", {"timestamp": self.metrics["ops_plane_ready_at"]})

    @asynccontextmanager
    async def _lock_context(self, lock_name: str):
        """
        Async lock wrapper that records how long we waited to acquire the lock.
        Usage: async with self._lock_context("global"): ...
        """
        lock = self._locks[lock_name]
        t0 = time.time()
        await lock.acquire()
        try:
            wait = time.time() - t0
            self._performance_stats["lock_wait_times"][lock_name].append(wait)
            yield
        finally:
            lock.release()

    def __init__(self, config: Optional[Dict | Any]=None, database_manager=None, exchange_client: Optional[Any]=None, app: Optional[Any]=None) -> None:
        # Logger must be initialized FIRST (before any self.logger calls)
        self.logger = logging.getLogger("SharedState")
        
        # Config initialization
        self.config = SharedStateConfig()
        if config:
            if isinstance(config, dict):
                items = config.items()
            else:
                try:
                    items = vars(config).items()
                except Exception:
                    items = []
            
            for k, v in items:
                if hasattr(self.config, k):
                    setattr(self.config, k, v)

        # Database
        self._database_manager = database_manager

        # Dynamic Configuration Overrides (Memory-resident)
        self.dynamic_config: Dict[str, Any] = {}
        self._exchange_client = exchange_client
        self._app = app  # AppContext reference for accessing governor
        
        self._profit_guard: Optional[Callable[[Dict[str, Any]], Any]] = None  # P9 Integration

        # Phase gates & event log
        self.accepted_symbols_ready_event = asyncio.Event()
        self.balances_ready_event = asyncio.Event()
        self.market_data_ready_event = asyncio.Event()
        self.nav_ready_event = asyncio.Event()
        self.ops_plane_ready_event = asyncio.Event()
        self.replan_request_event = asyncio.Event()  # P9: Re-plan trigger
        self._ops_first_report = {"PnLCalculator": False, "PerformanceEvaluator": False}
        
        self._event_log: deque = deque(maxlen=self.config.max_event_log_size)

        # Metrics & health
        self.metrics: Dict[str, Any] = {
            "startup_time": time.time(),  # TIER 2: Cold-bootstrap duration tracking
            "balances_updated_at": 0.0,
            "balances_ready": False,
            "nav_ready": False,
            "last_health_update": 0.0,
            "total_operations": 0,
            "error_counts": defaultdict(int),
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "nav": 0.0,
            "first_trade_at": None,  # BOOTSTRAP: Timestamp of first successful trade
            "total_trades_executed": 0,  # BOOTSTRAP: Count of executed trades
            "bootstrap_completed": False,  # BOOTSTRAP: Cosmetic latch after first successful trade
            "trades_tier_a": 0,          # Frequency Engineering: Tier A trade count
            "trades_tier_b": 0,          # Frequency Engineering: Tier B trade count
            "idle_ticks_count": 0,       # Count of cycles with no trade activity
            "total_holding_time_sec": 0.0, # Sum of holding times for completed trades
            "completed_trades_count": 0,  # Number of trades that have been closed
            "capital_utilization_pct": 0.0, # % of NAV currently in positions
            "ops_plane_ready_at": 0.0,
            "dust_registry_size": 0,
            "dust_origin_breakdown": {},
            "policy_conflicts": {
                "single_authority_vs_economic": 0,
                "economic_vs_phase2_grace": 0,
                "accumulating_protection_blocks": 0,
                "capital_floor_blocks": 0,
            },
            "capital_stable": False,
            "capital_stability_reason": "unknown",
            "current_mode": "BOOTSTRAP",
            "governance_decision": {},
        }
        
        # Phase 2: Persistent Bootstrap Metrics
        # Initialize persistent storage for bootstrap history
        db_path = getattr(self.config, "DB_PATH", None) or getattr(self.config, "DATABASE_PATH", None)
        self.bootstrap_metrics = BootstrapMetrics(db_path=db_path)
        
        # Load persisted metrics into in-memory metrics dict (for backward compatibility)
        persisted = self.bootstrap_metrics.get_all_metrics()
        if persisted.get("first_trade_at") is not None:
            self.metrics["first_trade_at"] = persisted["first_trade_at"]
        if persisted.get("total_trades_executed", 0) > 0:
            self.metrics["total_trades_executed"] = persisted["total_trades_executed"]
        
        # Phase 3: Dust Registry Lifecycle
        # Initialize persistent storage for dust position tracking
        self.dust_lifecycle_registry = DustRegistry(db_path=db_path)
        
        # Phase 4: Position Merger & Consolidation
        # Initialize position merger for consolidating fragmented positions
        self.position_merger = PositionMerger()
        
        # Phase 5: Trading Coordinator Integration
        # Initialize trading coordinator for unified trade execution
        # Import here to avoid circular imports
        from core.trading_coordinator import TradingCoordinator
        self.trading_coordinator = TradingCoordinator(self)
        
        # Health mirrors
        self.component_statuses: Dict[str, Dict[str, Any]] = {}
        self.system_health: Dict[str, Any] = {"status": "unknown", "message": "", "timestamp": 0.0}
        self.component_last_seen: Dict[str, float] = {}
        self.timestamps: Dict[str, float] = {}

        # Symbols
        self.symbols: Dict[str, Dict[str, Any]] = {}
        self.accepted_symbols: Dict[str, Dict[str, Any]] = {}
        self.symbol_filters: Dict[str, Dict[str, Any]] = {}

        # Market data
        self.latest_prices: Dict[str, float] = {}
        self.market_data: Dict[Tuple[str, str], List[OHLCVBar]] = {}
        self._last_tick_timestamps: Dict[str, float] = {}
        self._price_cache: Dict[str, Tuple[float, float]] = {}
        self._atr_cache: Dict[Tuple[str, str, int], float] = {}

        # Portfolio
        self.open_trades: Dict[str, Dict[str, Any]] = {}
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.balances: Dict[str, Dict[str, float]] = {}
        self.trade_history: deque = deque(maxlen=self.config.max_trade_history_size)
        self._realized_pnl: deque = deque(maxlen=4096)
        self.trade_count: int = 0
        self._avg_price_cache: Dict[str, float] = {}
        # Exit tracking (anti-churn / re-entry guard)
        self.last_exit_reason: Dict[str, str] = {}
        self.last_exit_ts: Dict[str, float] = {}
        self.last_exit_source: Dict[str, str] = {}

        # Quote asset (used by spendable-quote helpers)
        # Canonical quote asset from config
        self.quote_asset: str = str(getattr(config, 'quote_asset', 'USDT')).upper() if config else 'USDT'
        self.logger.info(f"[SS:Init] Quote asset configured: {self.quote_asset}")

        # ===== SHADOW MODE: Virtual Portfolio (P9) =====
        # When TRADING_MODE == "shadow", these track virtual balances & positions
        # Real balances remain in self.balances (untouched)
        self.virtual_balances: Dict[str, float] = {}  # Shadow-only balances
        self.virtual_positions: Dict[str, Dict[str, Any]] = {}  # Shadow-only positions
        self.virtual_realized_pnl: float = 0.0  # Cumulative realized PnL in shadow mode
        self.virtual_unrealized_pnl: float = 0.0  # Mark-to-market unrealized PnL in shadow
        self.virtual_nav: float = 0.0  # Net asset value in shadow mode
        self._shadow_mode_start_time: float = 0.0  # When shadow mode was activated
        self._shadow_mode_high_water_mark: float = 0.0  # Peak virtual NAV for drawdown calc
        self.trading_mode: str = str(getattr(self.config, 'trading_mode', 'live') or 'live')

        # Dust register tracks tiny, non-economical positions we may want to liquidate opportunistically
        self.dust_registry: Dict[str, Dict[str, Any]] = {}

        self._liq_requests = asyncio.Queue(maxsize=self.config.liq_queue_maxsize)

        # ===== PHASE 3: Dust Cleanup Tracking =====
        self.dust_cleanup_attempts = {}        # symbol → attempt count
        self.dust_cleanup_last_try = {}        # symbol → last attempt timestamp
        self._dust_first_seen = {}             # symbol → first_seen timestamp for age tracking
        self.dust_cleanup_max_attempts = 3     # max cleanup attempts before giving up
        self.dust_cleanup_retry_cooldown_sec = 300  # 5 minute retry cooldown
        self.bypass_portfolio_flat_for_dust = False  # Flag to bypass flat checks for dust cleanup

        # Risk & misc
        self.exposure_target = 0.25  # Increased to 25% NAV for profit activation
        self.cooldowns = {}
        self.active_liquidations = set()
        self.exit_in_progress: Dict[str, bool] = {}  # Symbol-level exit lock
        self.dust_operation_symbols: Dict[str, float] = {}  # Dust op timestamps by symbol
        self.risk_based_quote: Dict[str, float] = {}  # Risk-sized quote per symbol
        self.rebalance_targets: Set[str] = set()

        # Agent state
        self.volatility_regimes = {}
        self.sentiment_scores = {}
        self.agent_scores = {}
        self.cot_explanations = {}
        self.ml_position_scale = {}  # Symbol -> position scale multiplier from ML model
        # Back-compat aliases for components expecting singular names
        self.volatility_state = self.volatility_regimes
        self.sentiment_score = self.sentiment_scores

        # ═══════════════════════════════════════════════════════════════════════════════
        # SIGNAL BUFFER CONSENSUS: Adaptive Signal Window for Multi-Agent Fusion
        # ═══════════════════════════════════════════════════════════════════════════════
        # Allows signals to accumulate over time window instead of requiring instant alignment
        # Dramatically increases trading activity (10-20x) while maintaining risk controls
        
        self.signal_consensus_buffer: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        # Format: symbol → [signal_dict, signal_dict, ...]
        # Each signal must have: timestamp, action, agent, confidence
        
        self.signal_buffer_window_sec = 20.0  # Window for signal accumulation (default 20s)
        self.signal_buffer_max_age_sec = 30.0  # Max age before signal expires (default 30s)
        self.signal_buffer_max_signals_per_symbol = 20  # Max signals to keep per symbol
        
        # Agent weights for consensus voting (DIRECTIONAL VOTES ONLY - MLForecaster excluded)
        # MLForecaster is used for position sizing only, NOT directional consensus
        self.agent_consensus_weights: Dict[str, float] = {
            "TrendHunter": 0.50,      # 50% weight
            "DipSniper": 0.50,        # 50% weight
            # MLForecaster: NOT included - position sizing only
        }
        
        self.signal_consensus_threshold = 0.60  # Minimum score needed (0.0 - 1.0)
        self.signal_consensus_min_confidence = 0.55  # Minimum confidence for consideration
        
        # Statistics
        self.signal_buffer_stats = {
            "signals_received": 0,
            "consensus_trades_triggered": 0,
            "consensus_failures": 0,
            "buffer_flushes": 0,
            "last_consensus_check": 0.0,
        }

        # Seed a default volatility regime so agents have a baseline before the detector loop warms up.
        try:
            cfg = config or {}
            if isinstance(cfg, dict):
                default_tf = str(cfg.get("VOLATILITY_REGIME_TIMEFRAME", "5m") or "5m")
                default_regime = str(cfg.get("VOLATILITY_REGIME_DEFAULT", "normal") or "normal")
            else:
                default_tf = str(getattr(cfg, "VOLATILITY_REGIME_TIMEFRAME", "5m") or "5m")
                default_regime = str(getattr(cfg, "VOLATILITY_REGIME_DEFAULT", "normal") or "normal")
            default_regime = default_regime.lower()
            self.volatility_regimes.setdefault("GLOBAL", {})[default_tf] = {
                "regime": default_regime,
                "atrp": 0.0,
                "timestamp": time.time(),
            }
            self.metrics.setdefault("volatility_regime", default_regime)
            self.metrics.setdefault("volatility_regime_atrp", 0.0)
        except Exception:
            pass

        # Signals & alerts
        self.latest_signals_by_symbol: Dict[str, Dict[str, Dict[str, Any]]] = {} # sym -> agent -> signal
        self._pending_position_intents: Dict[Tuple[str, str], PendingPositionIntent] = {}  # (symbol, side) -> Intent
        self._latest_allocation_plan = {}  # Authoritative capital plans (P9)
        self._signal_buffer = deque(maxlen=self.config.max_signal_buffer_size)
        self.alerts = deque(maxlen=1000)
        self._pending_reservation_requests = [] # Pending P9 meta-healing requests

        # 🔄 LIGHTWEIGHT SIGNAL OUTCOME TRACKING
        self._signal_outcomes = []  # List of signal outcome records for periodic evaluation

        # Liquidity reservations
        self._quote_reservations = {}
        self._authoritative_reservations: Dict[str, float] = {} # Per-agent authoritative budget (P9 Strict)
        self._authoritative_reservation_ts: Dict[str, float] = {}  # When each budget was last set
        self._capital_failures: Dict[str, float] = {}  # agent_id -> timestamp
        self._portfolio_reset_done = False  # One-time portfolio reset guard

        # Async infra
        self._locks = {
            "global": asyncio.Lock(),
            "prices": asyncio.Lock(),
            "balances": asyncio.Lock(),
            "positions": asyncio.Lock(),
            "signals": asyncio.Lock(),
            "market_data": asyncio.Lock(),
            "metrics": asyncio.Lock(),
        }
        self._background_tasks: Dict[str, Optional[asyncio.Task]] = {
            "memory_optimization": None,
            "wallet_sync": None,
        }

        # Clocks, subscribers (logger already initialized at start of __init__)
        self._trading_start_time = datetime.utcnow()
        self._start_time_unix = time.time()
        self._start_monotonic = time.monotonic()
        self._cache_enabled = True
        self._subscribers: Dict[str, asyncio.Queue] = {}
        self.event_bus = _SharedStateEventBus(self)

        # Perf stats
        self._performance_stats = {
            "lock_wait_times": defaultdict(lambda: deque(maxlen=self.config.max_performance_samples)),
            "method_call_times": defaultdict(lambda: deque(maxlen=self.config.max_performance_samples)),
            "cache_hit_rates": defaultdict(lambda: {"hits": 0, "misses": 0}),
        }

        # Circuit breakers
        self._circuit_breakers: Dict[str, CircuitBreaker] = {
            "exchange": CircuitBreaker(failure_threshold=self.config.circuit_breaker_failure_threshold, timeout=self.config.circuit_breaker_timeout),
            "database": CircuitBreaker(failure_threshold=self.config.circuit_breaker_failure_threshold, timeout=self.config.circuit_breaker_timeout),
        }

        # Rejection tracking: (symbol, side, reason) -> count (P9 Deadlock Prevention)
        self.rejection_counters: Dict[Tuple[str, str, str], int] = defaultdict(int)
        self.rejection_timestamps: Dict[Tuple[str, str, str], float] = {}
        
        # 🔒 DUST RETIREMENT RULE (Mandatory Invariant)
        # Prevent dust positions from entering infinite rejection loops
        self.permanent_dust: Set[str] = set()  # Symbols marked as irrevocable dust
        self.dust_retirement_rejection_threshold: int = 3  # After N rejections, dust is PERMANENT
        self.dust_unhealable: Dict[str, str] = {}  # symbol -> reason; positions excluded from dust healing
        self._price_history: Dict[str, Any] = {}  # symbol -> deque/list of recent prices (used by get_market_state)

    @property
    def avg_holding_time_sec(self) -> float:
        """Frequency Engineering: Calculated average holding time."""
        total = self.metrics.get("total_holding_time_sec", 0.0)
        count = self.metrics.get("completed_trades_count", 0)
        if count <= 0: return 0.0
        return round(total / count, 2)

    async def initialize_from_database(self):
        """Legacy compatibility method for phased initialization."""
        self.logger.info("Initializing SharedState from database (legacy path)...")
        # Logic is now handled by RecoveryEngine, but we provide this stub to prevent crashes
        # if old main.py scripts are used (like on EC2).
        if self._database_manager:
            try:
                # Attempt to load a simple snapshot if available
                snapshot = await self._database_manager.load_shared_state_snapshot()
                if snapshot:
                    self.logger.info("SharedState snapshot loaded.")
            except Exception as e:
                self.logger.warning(f"Could not load legacy snapshot: {e}")

    async def add_agent_signal(self, symbol: str, agent: str, side: str, confidence: float, ttl_sec: int = 300, tier: str = "B", rationale: str = "", **extra_fields) -> None:
        """
        P9 Mandatory Signal Contract:
        Every trading agent must call this when it emits a signal.
        This is the shared 'signal bus' used by MetaController and other evaluators.
        
        Extra fields (e.g., _expected_move_pct, expected_edge_bps) are injected into the signal
        for downstream use by PolicyManager and ExecutionManager.
        """
        sym = self._norm_sym(symbol)
        now = time.time()
        sig = {
            "symbol": sym,
            "agent": agent,
            "side": side.upper(),
            "action": side.upper(),
            "confidence": float(confidence),
            "ttl_sec": int(ttl_sec),
            "tier": tier,
            "rationale": rationale,
            "ts": now,
            "timestamp": now,
        }
        
        # Inject optional extra signal fields (e.g., expected_edge_bps, _expected_move_pct)
        if extra_fields:
            for k, v in extra_fields.items():
                sig[k] = v
        
        # P9 Core storage (latest_signals_by_symbol)
        async with self._lock_context("signals"):
            if sym not in self.latest_signals_by_symbol:
                self.latest_signals_by_symbol[sym] = {}
            self.latest_signals_by_symbol[sym][agent] = sig
            self._signal_buffer.append(sig)
            
        # Emit event for downstream visibility
        if hasattr(self, "emit_event"):
            await self.emit_event("AgentSignal", sig)
        
        self.logger.info(f"📡 [Bus] Signal added for {sym} by {agent}: {side} (conf={confidence:.2f}, tier={tier})")

    async def add_strategy_signal(self, symbol: str, signal: Dict[str, Any]) -> None:
        """
        Append a signal to the internal signal buffer.
        P9: Preserves multi-agent signals per symbol.
        """
        async with self._lock_context("signals"):
            sym = self._norm_sym(symbol)
            agent = signal.get("agent", "UnknownAgent")
            self._signal_buffer.append(signal)
            
            if sym not in self.latest_signals_by_symbol:
                self.latest_signals_by_symbol[sym] = {}
            self.latest_signals_by_symbol[sym][agent] = signal

    # -------------------
    # Pending Position Accumulation (P9 Phase 4)
    # -------------------
    async def record_position_intent(self, intent: PendingPositionIntent) -> None:
        """Upsert a pending position intent for accumulation."""
        async with self._lock_context("signals"):
            key = (intent.symbol.upper(), intent.side.upper())
            self._pending_position_intents[key] = intent
            self.logger.info(f"Recorded pending intent for {key}: target={intent.target_quote}")
            if self._database_manager:
                with contextlib.suppress(Exception):
                    await self._database_manager.save_pending_intent({
                        "symbol": intent.symbol, "side": intent.side,
                        "target_quote": intent.target_quote, "accumulated_quote": intent.accumulated_quote,
                        "min_notional": intent.min_notional, "ttl_sec": intent.ttl_sec,
                        "source_agent": intent.source_agent, "state": intent.state,
                        "created_at": intent.created_at
                    })

    def get_pending_intent(self, symbol: str, side: str) -> Optional[PendingPositionIntent]:
        """Read-only access to a pending intent (no lock for speed)."""
        key = (symbol.upper(), side.upper())
        return self._pending_position_intents.get(key)

    def has_pending_intent(self, symbol: str, side: str = "BUY") -> bool:
        """Check if an active intent bucket exists for this symbol/side."""
        intent = self.get_pending_intent(symbol, side)
        return intent is not None and intent.state == "ACCUMULATING"

    def get_accumulated_quote(self, symbol: str, side: str = "BUY") -> float:
        """Return the current amount saved in the intent bucket for this symbol/side."""
        intent = self.get_pending_intent(symbol, side)
        if intent and intent.state == "ACCUMULATING":
            return float(intent.accumulated_quote)
        return 0.0

    async def add_to_accumulation(self, symbol: str, side: str, amount: float) -> float:
        """Add to the accumulated quote amount for an existing intent."""
        async with self._lock_context("signals"):
            key = (symbol.upper(), side.upper())
            intent = self._pending_position_intents.get(key)
            if intent:
                intent.accumulated_quote += amount
                if intent.state == "EXPIRED":
                    intent.state = "ACCUMULATING" # revive on new capital
                
                if self._database_manager:
                    with contextlib.suppress(Exception):
                        await self._database_manager.save_pending_intent({
                            "symbol": intent.symbol, "side": intent.side,
                            "target_quote": intent.target_quote, "accumulated_quote": intent.accumulated_quote,
                            "min_notional": intent.min_notional, "ttl_sec": intent.ttl_sec,
                            "source_agent": intent.source_agent, "state": intent.state,
                            "created_at": intent.created_at
                        })
                return intent.accumulated_quote
            return 0.0

    async def mark_intent_ready(self, symbol: str, side: str) -> bool:
        """
        Atomic claim of intent for execution (CAS-style).
        Returns True if this caller successfully flipped the state to READY.
        """
        async with self._lock_context("signals"):
            key = (symbol.upper(), side.upper())
            intent = self._pending_position_intents.get(key)
            if intent and intent.state == "ACCUMULATING":
                intent.state = "READY"
                if self._database_manager:
                    with contextlib.suppress(Exception):
                        await self._database_manager.save_pending_intent({
                            "symbol": intent.symbol, "side": intent.side,
                            "target_quote": intent.target_quote, "accumulated_quote": intent.accumulated_quote,
                            "min_notional": intent.min_notional, "ttl_sec": intent.ttl_sec,
                            "source_agent": intent.source_agent, "state": intent.state,
                            "created_at": intent.created_at
                        })
                return True
            # G019: IntentNotReady gate - NOT claimed or already ready - ADD INFO LOG
            if intent:
                self.logger.info(f"[EXEC_BLOCK] gate=INTENT_NOT_READY reason=ALREADY_CLAIMED state={intent.state} symbol={symbol} side={side} component=SharedState action=SKIP_ACCUMULATION")
            return False

    # -------------------
    # Dynamic Runtime Parameters (System Blackboard)
    # -------------------
    async def set_dynamic_param(self, key: str, value: Any) -> None:
        """Set a global runtime parameter (e.g. aggression_factor)."""
        # No lock needed for atomic dictionary operations in Python, 
        # but standardized access is good.
        self.dynamic_config[key] = value
        
    def get_dynamic_param(self, key: str, default: Any = None) -> Any:
        """Get a global runtime parameter."""
        return self.dynamic_config.get(key, default)

    def _cfg(self, key: str, default: Any = None) -> Any:
        """Resolve config from dynamic overrides or static config object."""
        if key in self.dynamic_config:
            return self.dynamic_config.get(key, default)
        if isinstance(self.config, dict):
            return self.config.get(key, default)
        return getattr(self.config, key, default)

    def is_intent_valid(self, symbol: str, side: str) -> bool:
        """Condition B: Check if the market intent is still valid based on current signals. (Point 4)"""
        sym = symbol.upper()
        # P9: Multi-agent signal lookup
        per_agent = self.latest_signals_by_symbol.get(sym, {})
        if not per_agent:
            return False
        
        # Take the most-recent signal by timestamp across all agents
        signal = max(per_agent.values(), key=lambda s: float(s.get("timestamp", 0.0)))
            
        # 1. Action Alignment
        action = str(signal.get("action", "") or signal.get("side", "")).upper()
        if action != side.upper():
            return False
            
        # 2. Confidence (Point 4)
        conf = float(signal.get("confidence", 0.0))
        min_conf = float(signal.get("min_confidence", 0.5))
        if conf < min_conf:
            return False
            
        # 3. Age / TTL (Point 4)
        sig_ts = float(signal.get("timestamp", 0.0))
        if sig_ts > 0:
            age = time.time() - sig_ts
            max_age = float(signal.get("ttl", 300)) # Default 5 min signal life
            if age > max_age:
                return False
                
        return True

    async def clear_pending_intent(self, symbol: str, side: str) -> None:
        """Remove an intent after execution or cancellation."""
        async with self._lock_context("signals"):
            key = (symbol.upper(), side.upper())
            if key in self._pending_position_intents:
                del self._pending_position_intents[key]
                if self._database_manager:
                    with contextlib.suppress(Exception):
                        await self._database_manager.delete_pending_intent(symbol.upper(), side.upper())

    async def expire_old_intents(self, now: float) -> None:
        """Cleanup logic for stale or invalid intents (Point 5)."""
        async with self._lock_context("signals"):
            to_del = []
            for key, intent in self._pending_position_intents.items():
                if intent.state == "ACCUMULATING":
                    # Time-based expiry
                    if now - intent.created_at > intent.ttl_sec:
                        to_del.append(key)
                        self.logger.info(f"TTL expired for {key}. Dropping intent.")
                    # Market validity (Condition B / Point 5)
                    elif not self.is_intent_valid(intent.symbol, intent.side):
                        to_del.append(key)
                        self.logger.info(f"Market validity lost for {key}. Dropping intent.")
            
            for key in to_del:
                if key in self._pending_position_intents:
                    del self._pending_position_intents[key]
                    if self._database_manager:
                        with contextlib.suppress(Exception):
                            await self._database_manager.delete_pending_intent(key[0], key[1])

    async def load_pending_intents_from_db(self) -> None:
        """Hydrate memory registry from persisted DB state on startup."""
        if not self._database_manager:
            return
        
        try:
            intents_data = await self._database_manager.load_pending_intents()
            async with self._lock_context("signals"):
                for data in intents_data:
                    # Construct from DB dict
                    intent = PendingPositionIntent(
                        symbol=data["symbol"],
                        side=data["side"],
                        target_quote=data["target_quote"],
                        accumulated_quote=data["accumulated_quote"],
                        min_notional=data["min_notional"],
                        ttl_sec=data["ttl_sec"],
                        source_agent=data["source_agent"],
                        state=data["state"],
                        created_at=data["created_at"]
                    )
                    key = (intent.symbol.upper(), intent.side.upper())
                    self._pending_position_intents[key] = intent
            self.logger.info(f"Hydrated {len(intents_data)} pending position intents from DB.")
        except Exception as e:
            self.logger.error(f"Failed to load pending intents from DB: {e}")

    def get_unified_score(self, symbol: str) -> float:
        """
        Compute a consistent, cross-component score for a symbol.
        Architect's refinement: Multi-factor professional scoring
        40% conviction + 20% volatility + 20% momentum + 20% liquidity
        """
        symbol = symbol.upper()
        
        # Factor 1: Base Conviction (AI agent scores) - 40%
        conviction = self.agent_scores.get(symbol, 0.5)
        
        # Factor 2: Market Regime (Volatility) - 20%
        # Normalize regime: bear=0.8, neutral=1.0, bull=1.2
        # volatility_regimes is nested: {symbol: {timeframe: {regime, atrp, timestamp}}}
        # Extract regime from the primary timeframe (5m is default)
        regime_name = "neutral"
        try:
            symbol_regimes = self.volatility_regimes.get(symbol, {})
            if isinstance(symbol_regimes, dict):
                # Try primary timeframe first (5m), then fall back to any available
                regime_data = symbol_regimes.get("5m") or symbol_regimes.get("1m") or next(iter(symbol_regimes.values()), None)
                if isinstance(regime_data, dict):
                    regime_name = regime_data.get("regime", "neutral").lower()
        except Exception:
            pass  # Fall back to neutral
        
        volatility_score = 1.0
        if regime_name == "bull": 
            volatility_score = 1.2
        elif regime_name == "bear": 
            volatility_score = 0.8
        
        # Factor 3: Momentum (Sentiment + Price Action) - 20%
        # Normalize sentiment from -1..+1 to 0..1 range
        sentiment = self.sentiment_scores.get(symbol, 0.0)
        momentum_score = (sentiment + 1.0) / 2.0  # Converts -1..1 to 0..1
        
        # Factor 4: Liquidity (Volume + Spread) - 20%
        # ⚡ ARCHITECT REFINEMENT #2: Include volume in scoring, not rejection
        # FIX: latest_prices is Dict[str, float], not Dict[str, Dict]
        # Use accepted_symbols for volume/liquidity data if available
        # NOTE: Some discovery agents populate with minimal metadata (just status/notional)
        # This is OK - we gracefully degrade to neutral liquidity score
        liquidity_score = 0.5  # Default neutral liquidity
        try:
            symbol_info = self.accepted_symbols.get(symbol, {})
            if isinstance(symbol_info, dict):
                # Try to extract liquidity metrics from symbol info
                # Try multiple key names for compatibility with different data sources
                quote_volume = float(
                    symbol_info.get("quote_volume") 
                    or symbol_info.get("volume") 
                    or symbol_info.get("24h_volume")
                    or symbol_info.get("quote_volume_usd", 0) 
                    or 0
                )
                spread = float(symbol_info.get("spread") or symbol_info.get("bid_ask_spread", 0.01) or 0.01)
                
                # Only compute liquidity score if we have meaningful volume
                # If volume is 0 (missing metadata), keep neutral 0.5
                if quote_volume > 0:
                    liquidity_score = min(quote_volume / 100000, 1.0) * max(0, 1.0 - min(spread, 0.05))
                # else: keep liquidity_score = 0.5 (neutral)
        except (TypeError, ValueError, AttributeError):
            # Fall back to neutral liquidity if any error (type mismatch, conversion error, etc.)
            liquidity_score = 0.5
        
        # Professional multi-factor composite
        # This is the approach used by hedge funds and market makers
        composite = (
            conviction * 0.40 +          # 40% AI signal strength
            volatility_score * 0.20 +    # 20% market regime (bull/bear)
            momentum_score * 0.20 +      # 20% trend strength
            liquidity_score * 0.20       # 20% tradability (includes volume!)
        )
        
        return float(composite)

    def get_symbol_scores(self) -> Dict[str, float]:
        """Returns a snapshot of unified scores for all known symbols."""
        all_syms = set(self.latest_prices.keys()) | set(self.positions.keys()) | set(self.accepted_symbols.keys())
        return {s: self.get_unified_score(s) for s in all_syms if s}

    def calibrate_confidence(self, raw_conf: float, agent: str = "unknown") -> float:
        """
        Calibrate raw ML confidence to prevent overconfidence.
        
        - Clamps output to max 0.95 to prevent false certainty
        - Uses historical win_rate when available to calibrate
        - Logs warning if raw confidence is 1.0 (likely uncalibrated)
        """
        MAX_CONFIDENCE = 0.95  # Never allow 100% confidence
        
        # Warning for suspiciously high confidence
        if raw_conf >= 0.99:
            self.logger.debug(f"[ConfidenceCalibration] High raw confidence {raw_conf:.3f} from {agent} - capping at {MAX_CONFIDENCE}")
        
        # Try to use historical win_rate for calibration
        try:
            kpi = getattr(self, "kpi_metrics", None) or {}
            agent_stats = (kpi.get("per_agent") or {}).get(agent, {})
            historical_win_rate = float(agent_stats.get("win_rate", 0.0))
            
            if historical_win_rate > 0:
                # Blend raw ML confidence with historical performance
                # calibrated = 0.7 * raw + 0.3 * historical_win_rate
                calibrated = 0.7 * raw_conf + 0.3 * historical_win_rate
            else:
                # No history - apply a conservative discount
                calibrated = raw_conf * 0.90
        except Exception:
            calibrated = raw_conf * 0.90
        
        return min(MAX_CONFIDENCE, max(0.0, calibrated))

    def get_balance_snapshot(self) -> Dict[str, Dict[str, float]]:
        """Return a shallow copy of all balances."""
        return dict(self.balances)

    def get_nav_quote(self) -> float:
        """Return the current NAV in quote asset (USDT).
        
        CRITICAL SHADOW MODE FIX:
        In shadow mode, use WALLET_VALUE directly instead of positions + free.
        This prevents double-counting because positions are derived from wallet balances.
        
        Normal mode: NAV = sum(all_quote_balances) + sum(all_positions_at_market_price)
        Shadow mode: NAV = wallet_value (computed from actual exchange balances)
        
        FIX #3: Support multiple quote assets (USDT, BUSD, FDUSD, etc)
        BOOTSTRAP FIX: When NAV calculates to 0 (cold start, no positions),
        return free quote as the bootstrap NAV to unblock first trade.
        """
        # CRITICAL FIX: In shadow mode, use wallet value directly
        # This prevents double-counting when positions are hydrated from wallet
        is_shadow_mode = getattr(self, "_shadow_mode", False)
        
        nav = 0.0
        
        # FIX #3: Support list of quote assets (multi-quote accounts)
        quote_assets = getattr(self, "quote_assets", None)
        if not quote_assets:
            # Fallback to singular quote_asset for backward compatibility
            quote_assets = [getattr(self, "quote_asset", "USDT").upper()]
        else:
            quote_assets = [q.upper() for q in (quote_assets if isinstance(quote_assets, list) else [quote_assets])]
        
        free_total = 0.0
        locked_total = 0.0
        quote_balances: Dict[str, Dict[str, float]] = {}
        
        # Sum ALL quote assets from balances
        for asset, b in self.balances.items():
            a = asset.upper()
            if a in quote_assets:
                free = float(b.get("free", 0.0))
                locked = float(b.get("locked", 0.0))
                free_total += free
                locked_total += locked
                quote_balances[a] = {"free": free, "locked": locked}
                self.logger.debug(f"[NAV] Quote asset {a}: free={free}, locked={locked}")
        
        nav += free_total + locked_total
        
        # SHADOW MODE FIX: If in shadow mode, return wallet value directly
        # Do NOT add positions because they are derived from wallet balances
        if is_shadow_mode:
            self.logger.info(
                f"[NAV] Shadow mode: using wallet_value={nav:.2f} "
                f"(not adding positions to prevent double-count)"
            )
            return nav
        
        # Add ALL position values (no filtering by trade floor or position size)
        # This is required so NAV accurately reflects total portfolio value
        has_positions = False
        for sym, pos in self.positions.items():
            qty = float(pos.get("quantity", 0.0))
            if qty <= 0: 
                continue
            has_positions = True
            px = float(self.latest_prices.get(sym) or pos.get("mark_price") or pos.get("entry_price") or 0.0)
            if px > 0:
                nav += qty * px  # Include ALL positions, even if below MIN_ECONOMIC_TRADE_USDT
        
        # FIX #3: BOOTSTRAP FIX - If NAV is 0 but we have free quote, use it as bootstrap NAV
        if nav <= 0 and free_total > 0 and not has_positions:
            self.logger.info(f"[BOOTSTRAP] NAV=0, using free quote total as bootstrap NAV: {free_total:.2f}")
            return free_total
            
        self.logger.debug(
            f"[NAV] Total: {nav:.2f} | "
            f"Quotes: {quote_balances} | "
            f"Positions: {len(self.positions)} | "
            f"Assets: {len(self.balances)}"
        )
        return nav

    async def get_nav(self) -> float:
        """Async-compatible NAV getter for legacy callers."""
        try:
            return float(self.get_nav_quote())
        except Exception:
            return 0.0

    def get_active_allocation_plan(self) -> Dict[str, Any]:
        """P9: Public getter for the latest authoritative allocation plan."""
        return dict(getattr(self, "_latest_allocation_plan", {}) or {})
    def set_readiness_flag(self, flag: str, value: bool = True) -> None:
        """Set/clear a readiness event by name and emit a lightweight event. No global resets here."""
        flag_map = {
            "accepted_symbols_ready": self.accepted_symbols_ready_event,
            "balances_ready": self.balances_ready_event,
            "market_data_ready": self.market_data_ready_event,
            "ops_plane_ready": self.ops_plane_ready_event,
            "nav_ready": self.nav_ready_event,
        }
        ev = flag_map.get(flag)
        if not ev:
            return
        if value:
            ev.set()
        else:
            ev.clear()
        # mirror to metrics for observability
        if flag == "balances_ready":
            self.metrics["balances_ready"] = bool(value)
        try:
            coro = self.emit_event("ReadinessFlagChanged", {"flag": flag, "value": bool(value), "ts": time.time()})
            loop = asyncio.get_running_loop()
            loop.create_task(coro)
        except (RuntimeError, Exception):
            pass

    def get_target_exposure(self) -> float:
        """Return target exposure (0.0 to 1.0), preferring dynamic config."""
        return float(self.dynamic_config.get("TARGET_EXPOSURE", self.exposure_target))

    def set_target_exposure(self, value: float) -> None:
        self.exposure_target = float(value)
        self.dynamic_config["TARGET_EXPOSURE"] = float(value)

    # -------- exchange client plumbing --------
    @property
    def exchange_client(self):
        return self._exchange_client
    @exchange_client.setter
    def exchange_client(self, client):
        self._exchange_client = client
    async def set_exchange_client(self, client):
        self._exchange_client = client

    # -------- convenience shims expected by MDF/AppContext --------
    @property
    def accepted_symbols_ready(self) -> asyncio.Event:
        """Compatibility alias used by MDF."""
        return self.accepted_symbols_ready_event

    def is_market_data_ready(self) -> bool:
        """Probed by AppContext after MDF warmup."""
        return self.market_data_ready_event.is_set()

    def is_balances_ready(self) -> bool:
        """Probed by components that need wallet state."""
        return self.balances_ready_event.is_set()

    async def get_config(self, key: str, default: Any = None) -> Any:
        """
        Retrieves a value from the dynamic configuration store.
        """
        return self.dynamic_config.get(key, default)

    async def set_config(self, key: str, value: Any) -> None:
        """
        Updates a dynamic configuration value and emits an event.
        """
        self.dynamic_config[key] = value
        self.logger.info(f"Dynamic Config Updated: {key} = {value}")
        try:
            coro = self.emit_event("DynamicConfigChanged", {"key": key, "value": value, "ts": time.time()})
            loop = asyncio.get_running_loop()
            loop.create_task(coro)
        except Exception:
            pass

    async def update_dynamic_config(self, mapping: Dict[str, Any]) -> None:
        """
        Bulk updates the dynamic configuration and emits an event.
        """
        self.dynamic_config.update(mapping)
        self.logger.info(f"Dynamic Config Bulk Updated: {list(mapping.keys())}")
        try:
            coro = self.emit_event("DynamicConfigChanged", {"mapping": mapping, "ts": time.time()})
            loop = asyncio.get_running_loop()
            loop.create_task(coro)
        except Exception:
            pass

    # --- New helpers required by MDF/AppContext contracts (P9) ---
    def has_ohlcv(self, symbol: str, timeframe: str, min_bars: int = 1) -> bool:
        """
        Return True if we have at least `min_bars` OHLCV rows for (symbol, timeframe).
        MDF/AppContext probe this during warmup/readiness.
        """
        try:
            sym = self._norm_sym(symbol)
            return self.get_ohlcv_count(sym, timeframe) >= int(min_bars)
        except Exception:
            return False

    async def set_market_data_ready(self, value: bool = True) -> None:
        """
        Explicit toggler used by MarketDataFeed._maybe_set_ready().
        Mirrors to the internal event and emits a lightweight 'MarketDataReady' when enabling.
        """
        self.set_readiness_flag("market_data_ready", bool(value))
        if value:
            try:
                syms = list(self.accepted_symbols.keys())
                await self.emit_event("MarketDataReady", {"symbols": syms, "timeframe": "auto", "min_bars": getattr(self.config, "min_bars_required", 0)})
            except Exception:
                pass

    async def mark_symbol_data_ready(self, symbol: str) -> None:
        """
        Per-symbol readiness signal used by MDF warmup/run loops.
        """
        try:
            sym = self._norm_sym(symbol)
            await self.emit_event("SymbolDataReady", {"symbol": sym, "ts": time.time()})
        except Exception:
            pass

    @property
    def balances_ready(self) -> bool:
        """
        Back-compat boolean probed by AppContext.
        Mirrors balances_ready_event.is_set().
        """
        return self.balances_ready_event.is_set()

    @property
    def nav_ready(self) -> bool:
        """
        Compatibility boolean probed by AppContext.
        Mirrors nav_ready_event.is_set().
        """
        return self.nav_ready_event.is_set()
    async def set_nav_ready(self, value: bool = True) -> None:
        """
        Explicit toggler used by AppContext or internal logic once NAV is computable.
        """
        self.set_readiness_flag("nav_ready", bool(value))
        if value:
            try:
                await self.emit_event("NavReady", {"ts": time.time()})
            except Exception:
                pass

    async def _maybe_set_nav_ready(self) -> None:
        """
        Consider NAV 'ready' once balances are available; optionally require any price basis.
        This avoids AppContext gating forever on NAVNotReady in public-only or low-funds modes.
        """
        try:
            if not self.balances_ready_event.is_set():
                return
            # If you want to be stricter, require any last price or any non-zero NAV:
            # has_price = bool(self.latest_prices)
            # if not has_price: return
            if not self.nav_ready_event.is_set():
                self.nav_ready_event.set()
                self.metrics["nav_ready"] = True
                await self.emit_event("NavReady", {"ts": time.time()})
        except Exception:
            pass

    async def free_usdt(self) -> float:
        """Spendable quote funds after reserve policy. Delegates to get_spendable_balance()."""
        try:
            return await self.get_spendable_quote(
                self.quote_asset,
                reserve_ratio=self.config.quote_reserve_ratio,
                min_reserve=self.config.quote_min_reserve,
            )
        except Exception:
            # Fallback to raw free balance if anything goes wrong
            bal = await self.get_balance(self.quote_asset)
            return float(bal.get("free", 0.0))

    # ========== CAPITAL STATE MANAGEMENT (FIX #6) ==========
    
    async def hard_reset_capital_state(self) -> None:
        """
        MANDATORY: Call on every manual restart.
        Clears ALL stale capital state before MetaController starts.
        
        FIX #6: Eliminates carryover capital locks that prevent trading.
        """
        async with self._lock_context("global"):
            # Clear all reservations
            self._quote_reservations.clear()
            self.logger.info("[SS:HardReset] Cleared all quote reservations")
            
            # Clear all pending intents
            self._pending_position_intents.clear()
            self.logger.info("[SS:HardReset] Cleared all pending position intents")
            
            # Clear locked capital tracking
            self._authoritative_reservations.clear()
            self._authoritative_reservation_ts.clear()
            self._capital_failures.clear()
            self.logger.info("[SS:HardReset] Cleared locked capital state")
            
            # Force sync from exchange
            await self.sync_authoritative_balance(force=True)
            self.logger.info("[SS:HardReset] Force-synced balances from Binance")
            
            # === CRITICAL: Rehydrate capital from wallet ===
            # After clearing stale reservations, recompute free capital from actual exchange balance
            # This ensures MetaController sees the REAL available capital, not a stale value
            try:
                quote_asset = str(self.config.quote_asset or "USDT").upper()
                actual_balance = await self.get_balance(quote_asset)
                wallet_free = float(actual_balance.get("free", 0.0))
                
                # Compute spendable (free - safety buffer)
                reserve_ratio = float(self.config.quote_reserve_ratio or 0.10)
                safety_buffer = wallet_free * reserve_ratio
                spendable = max(0.0, wallet_free - safety_buffer)
                
                self.logger.warning(
                    "[SS:HardReset:CapitalRehydration] "
                    "Wallet: free=%.2f USDT, buffer=%.2f USDT (%.0f%%), spendable=%.2f USDT",
                    wallet_free, safety_buffer, reserve_ratio * 100, spendable
                )
                
                # Trigger one calculation of spendable_balance to cache fresh value
                # This forces get_spendable_balance() to see zero reservations (since we cleared them)
                fresh_spendable = await self.get_spendable_balance(quote_asset)
                self.logger.warning(
                    "[SS:HardReset:CapitalRehydration] "
                    "get_spendable_balance()=%s returned %.2f USDT (should match computed spendable)",
                    quote_asset, fresh_spendable
                )
            except Exception as e:
                self.logger.error(
                    "[SS:HardReset:CapitalRehydration] Failed to rehydrate capital: %s",
                    e, exc_info=True
                )
            
            self.logger.warning(
                "[SS:HardReset] ⚠️ HARD CAPITAL RESET COMPLETE - "
                "All reservations, intents, and locks cleared. "
                "Capital rehydrated from wallet. "
                "System ready for MetaController startup."
            )

    async def reset_portfolio_state_once(self) -> None:
        """
        One-time portfolio reset:
        - Clear ghost positions (positions not backed by wallet balances)
        - Clear reserved capital (stale reservations/intents)
        - Keep actual balances untouched (re-sync only)
        """
        if getattr(self, "_portfolio_reset_done", False):
            self.logger.info("[SS:PortfolioReset] Skipped (already executed once)")
            return

        self._portfolio_reset_done = True
        self.logger.warning("[SS:PortfolioReset] Starting one-time portfolio reset (ghost positions + reservations)")

        # Clear stale capital state + re-sync balances
        await self.hard_reset_capital_state()

        # Snapshot balances for ghost detection (do NOT mutate balances)
        balances_snapshot = dict(self.balances)
        positions_snapshot = dict(self.positions)

        quote_assets = getattr(self, "quote_assets", None)
        if not quote_assets:
            quote_assets = [getattr(self, "quote_asset", "USDT").upper()]
        else:
            quote_assets = [q.upper() for q in (quote_assets if isinstance(quote_assets, list) else [quote_assets])]

        def infer_base(sym: str) -> Optional[str]:
            s = self._norm_sym(sym)
            for q in quote_assets:
                if s.endswith(q) and len(s) > len(q):
                    return s[: -len(q)]
            return None

        ghost_symbols: list[str] = []
        for sym, pos in positions_snapshot.items():
            qty = float(pos.get("quantity", 0.0))
            status = str(pos.get("status", "")).upper()
            if qty <= 0 or status in {"CLOSED", "DUST", "PERMANENT_DUST"}:
                ghost_symbols.append(sym)
                continue

            base = infer_base(sym)
            if not base:
                continue
            bal = balances_snapshot.get(base.upper(), {})
            bal_total = float(bal.get("free", 0.0)) + float(bal.get("locked", 0.0))
            if bal_total <= 0:
                ghost_symbols.append(sym)

        if ghost_symbols:
            async with self._lock_context("positions"):
                for sym in ghost_symbols:
                    self.positions.pop(self._norm_sym(sym), None)
            async with self._lock_context("global"):
                for sym in ghost_symbols:
                    self.open_trades.pop(self._norm_sym(sym), None)
                    self.dust_registry.pop(self._norm_sym(sym), None)

            self.metrics["dust_registry_size"] = len(self.dust_registry)
            self.logger.warning(
                "[SS:PortfolioReset] Cleared ghost positions: %s",
                ", ".join(sorted(set(ghost_symbols)))
            )
        else:
            self.logger.info("[SS:PortfolioReset] No ghost positions detected")

        # Rehydrate positions from wallet (if enabled) to ensure consistency
        # CRITICAL: Never hydrate positions from balances in shadow mode
        if (
            getattr(self.config, "auto_positions_from_balances", True)
            and self.trading_mode != "shadow"
        ):
            await self.hydrate_positions_from_balances()

        # Log resulting spendable capital snapshot
        try:
            quote_asset = getattr(self, "quote_asset", "USDT")
            spendable = await self.get_spendable_balance(quote_asset)
            self.logger.warning(
                "[SS:PortfolioReset] Completed. spendable_%s=%.2f",
                str(quote_asset).upper(),
                float(spendable or 0.0),
            )
        except Exception:
            pass

    async def authoritative_wallet_sync(self) -> Dict[str, Any]:
        """
        Authoritative wallet sync (exchange is source of truth).
        - Hard-sync balances from exchange
        - Clear in-memory positions/reservations/intents/locks
        - Rebuild positions from non-zero balances
        - Recompute invested capital, free capital, unrealized PnL
        """
        if not self._exchange_client:
            self.logger.warning("[SS:AuthoritativeSync] No exchange client attached; skipping")
            return {"balances": {}, "positions": {}, "invested_capital": 0.0, "free_capital": 0.0}

        # Clear in-memory capital + intent state
        async with self._lock_context("global"):
            self._quote_reservations.clear()
            self._authoritative_reservations.clear()
            self._authoritative_reservation_ts.clear()
            self._capital_failures.clear()
            self._pending_reservation_requests.clear()
            self._pending_position_intents.clear()
        async with self._lock_context("positions"):
            self.positions.clear()
        async with self._lock_context("global"):
            self.open_trades.clear()
            self.dust_registry.clear()

        self.metrics["dust_registry_size"] = 0

        # Fetch balances from exchange (hard sync)
        await self.sync_authoritative_balance(force=True)

        quote_assets = getattr(self, "quote_assets", None)
        if not quote_assets:
            quote_assets = [getattr(self, "quote_asset", "USDT").upper()]
        else:
            quote_assets = [q.upper() for q in (quote_assets if isinstance(quote_assets, list) else [quote_assets])]

        quote_asset = quote_assets[0]
        balances_snapshot = dict(self.balances)

        invested_capital = 0.0
        unrealized_pnl = 0.0
        rebuilt_positions: Dict[str, Dict[str, Any]] = {}

        for asset, data in balances_snapshot.items():
            a = asset.upper()
            if a in quote_assets:
                continue
            qty = float(data.get("free", 0.0)) + float(data.get("locked", 0.0))
            if qty <= 0:
                continue

            sym = f"{a}{quote_asset}"
            if hasattr(self._exchange_client, "has_symbol") and not self._exchange_client.has_symbol(sym):
                continue

            price = 0.0
            try:
                if hasattr(self._exchange_client, "get_current_price"):
                    price = float(await self._exchange_client.get_current_price(sym) or 0.0)
                elif hasattr(self._exchange_client, "get_symbol_price"):
                    price = float(await self._exchange_client.get_symbol_price(sym) or 0.0)
            except Exception:
                price = 0.0

            avg_price = price if price > 0 else 0.0
            position_value = qty * avg_price if avg_price > 0 else 0.0
            significant_floor = float(await self.get_significant_position_floor(sym) or 0.0)
            is_significant = bool(position_value >= significant_floor and position_value > 0.0)
            if is_significant:
                invested_capital += position_value
            rebuilt_positions[sym] = {
                "quantity": qty,
                "avg_price": avg_price,
                "mark_price": price,
                "value_usdt": float(position_value),
                "significant_floor_usdt": float(significant_floor),
                "status": "SIGNIFICANT" if is_significant else "DUST",
                "state": PositionState.ACTIVE.value if is_significant else PositionState.DUST_LOCKED.value,
                "is_significant": bool(is_significant),
                "is_dust": not bool(is_significant),
                "_is_dust": not bool(is_significant),
                "open_position": bool(is_significant),
                "_mirrored": True,
            }
            if price > 0:
                self.latest_prices[sym] = price

        # Apply rebuilt positions
        for sym, pos in rebuilt_positions.items():
            await self.update_position(sym, pos)
            if not bool(pos.get("is_significant", False)):
                self.record_dust(
                    sym,
                    float(pos.get("quantity", 0.0) or 0.0),
                    origin="wallet_balance_sync",
                    context={
                        "source": "hard_reset_authoritative_sync",
                        "value_usdt": float(pos.get("value_usdt", 0.0) or 0.0),
                        "significant_floor_usdt": float(pos.get("significant_floor_usdt", 0.0) or 0.0),
                    },
                )
                self.open_trades.pop(sym, None)
            else:
                self.dust_registry.pop(sym, None)

        # Recompute free capital from quote balance
        quote_bal = balances_snapshot.get(quote_asset, {})
        quote_total = float(quote_bal.get("free", 0.0)) + float(quote_bal.get("locked", 0.0))
        free_capital = max(0.0, quote_total - invested_capital)

        # Unrealized PnL from price vs avg_price
        for sym, pos in rebuilt_positions.items():
            avg = float(pos.get("avg_price", 0.0))
            px = float(pos.get("mark_price", 0.0))
            qty = float(pos.get("quantity", 0.0))
            if avg > 0 and px > 0:
                unrealized_pnl += (px - avg) * qty

        if isinstance(self.metrics, dict):
            self.metrics["invested_capital"] = float(invested_capital)
            self.metrics["capital_free"] = float(free_capital)
            self.metrics["unrealized_pnl"] = float(unrealized_pnl)

        self.logger.warning(
            "[SS:AuthoritativeSync] Done | positions=%d invested=%.2f free=%.2f quote=%s",
            len(rebuilt_positions), invested_capital, free_capital, quote_asset
        )

        return {
            "balances": balances_snapshot,
            "positions": rebuilt_positions,
            "invested_capital": invested_capital,
            "free_capital": free_capital,
        }

    async def get_free_and_reserved(self) -> tuple:
        """
        FIX #6, Step 2: Returns (binance_free, system_reserved).
        
        Binance free is the SOURCE OF TRUTH.
        System reserved = sum of our tracked reservations.
        
        The difference tells us if we have orphaned reservations.
        """
        bal = await self.get_balance("USDT")
        binance_free = float(bal.get("free", 0.0))
        
        # Sum of our reservations
        reservations = self._quote_reservations.get("USDT", [])
        system_reserved = sum(float(r.get("amount", 0.0)) for r in reservations)
        
        return binance_free, system_reserved

    async def classify_positions_by_size(self) -> Dict[str, List[str]]:
        """
        FIX #6, Step 3: Classify positions into SIGNIFICANT and DUST based on minNotional.

        ✅ FIX #1: Use position's own price data when market price unavailable

        Critical for portfolio "flat" detection.
        Dust positions:
        - Do NOT block capital allocation
        - Do NOT count toward occupied capital
        - Can be liquidated opportunistically
        """
        significant = []
        dust = []

        # ARCHITECTURE FIX: Branch on trading_mode to get correct positions source
        positions_source = self.virtual_positions if self.trading_mode == "shadow" else self.positions
        
        # Snapshot keys to avoid mutation-during-iteration issues
        position_keys = list(positions_source.keys())

        for symbol in position_keys:
            try:
                async with self._lock_context("positions"):
                    position = positions_source.get(symbol)
                    if not position:
                        continue
                    position = dict(position)  # Work on a copy

                qty = float(position.get("quantity", 0.0) or position.get("qty", 0.0) or 0.0)
                if qty <= 0:
                    continue

                significant_floor = float(await self.get_significant_position_floor(symbol) or 0.0)
                position_value = float(self._estimate_position_value_usdt(symbol, position) or 0.0)
                self.logger.warning(
                    "[DEBUG:CLASSIFY] %s qty=%.8f value=%.4f floor=%.4f latest_price=%.4f avg_price=%s entry_price=%s",
                    symbol,
                    qty,
                    position_value,
                    significant_floor,
                    float(self.latest_prices.get(self._norm_sym(symbol), 0.0) or 0.0),
                    position.get("avg_price"),
                    position.get("entry_price"),
                )
                position["value_usdt"] = float(position_value)
                position["significant_floor_usdt"] = float(significant_floor)

                if position_value >= significant_floor and position_value > 0:
                    significant.append(symbol)
                    position["status"] = "SIGNIFICANT"
                    position["state"] = PositionState.ACTIVE.value
                    position["capital_occupied"] = float(position_value)
                    position["is_significant"] = True
                    position["is_dust"] = False
                    position["_is_dust"] = False
                    position["open_position"] = True
                    async with self._lock_context("positions"):
                        positions_source[symbol] = position
                    self.dust_registry.pop(self._norm_sym(symbol), None)
                else:
                    dust.append(symbol)
                    self.mark_as_dust(symbol)
                    self.logger.debug(
                        "[SS:Dust] %s value=%.4f < floor=%.4f -> DUST_LOCKED",
                        symbol,
                        position_value,
                        significant_floor,
                    )
            except Exception as e:
                self.logger.warning(f"[SS:Dust] Error classifying {symbol}: {e}")
                dust.append(symbol)
                self.mark_as_dust(symbol)

        return {"significant": significant, "dust": dust}

    def mark_as_dust(self, symbol: str) -> None:
        """
        Mark a position as dust.
        
        FIX #6, Step 3: Dust positions:
        - Do NOT block capital allocation
        - Do NOT count toward portfolio "occupied" capital
        - CAN be liquidated opportunistically
        """
        sym = self._norm_sym(symbol)
        if sym in self.positions:
            pos = self.positions[sym]
            pos["status"] = "DUST"
            pos["capital_occupied"] = 0.0
            pos["state"] = PositionState.DUST_LOCKED.value
            pos["is_significant"] = False
            pos["is_dust"] = True
            pos["_is_dust"] = True
            pos["open_position"] = False
            pos["accumulation_locked"] = False
            self.positions[sym] = pos

        # Hard invariant: dust never lives in open_trades.
        self.open_trades.pop(sym, None)

        # Record in dust registry
        accepted = sym in self.accepted_symbols or sym in self.symbols
        origin = "strategy_portfolio" if accepted else "external_untracked"
        qty = float((self.positions.get(sym, {}) or {}).get("quantity", 0.0) or 0.0)
        self.record_dust(
            sym,
            qty,
            origin=origin,
            context={"source": "mark_as_dust", "accepted_symbol": accepted},
        )

        self.logger.debug("[SS:Dust] Marked %s as DUST - does not block capital", sym)

    def mark_as_permanent_dust(self, symbol: str) -> None:
        """
        🔒 DUST RETIREMENT RULE: Mark a position as PERMANENT_DUST.
        
        Once a dust position has been rejected >= N times, it's retired permanently:
        - Cannot be re-activated
        - Excluded from rejection counters
        - Excluded from liquidation queue
        - Excluded from capital accounting
        - Future operations on this symbol will skip it
        
        Critical for preventing infinite rejection loops.
        """
        sym = symbol.upper()
        if sym not in self.permanent_dust:
            self.permanent_dust.add(sym)
            
            if sym in self.positions:
                self.positions[sym]["status"] = "PERMANENT_DUST"
                self.positions[sym]["capital_occupied"] = 0.0
                self.positions[sym]["state"] = PositionState.DUST_LOCKED.value
            
            # Clear all rejection counters for this symbol to reset the cycle
            keys_to_del = [k for k in self.rejection_counters.keys() if k[0] == sym]
            for k in keys_to_del:
                self.rejection_counters.pop(k, None)
                self.rejection_timestamps.pop(k, None)
            
            self.logger.info(f"[SS:DUST_RETIRED] {sym} marked PERMANENT_DUST (irrevocable, retirement complete)")

    def is_permanent_dust(self, symbol: str) -> bool:
        """Check if a position is marked as PERMANENT_DUST (retired)."""
        return symbol.upper() in self.permanent_dust

    def get_permanent_dust_positions(self) -> List[str]:
        """Get list of all PERMANENT_DUST positions (retired positions)."""
        return list(self.permanent_dust)

    async def get_significant_position_count(self) -> int:
        """
        FIX #6, Step 4: Count positions that are SIGNIFICANT (>= minNotional).
        
        Dust positions are NOT counted.
        """
        classification = await self.classify_positions_by_size()
        return len(classification["significant"])

    async def get_occupied_capital(self) -> float:
        """
        FIX #6, Step 3: Total capital occupied by SIGNIFICANT positions only.
        
        Dust does NOT contribute to occupied capital.
        """
        occupied = 0.0
        for symbol, position in self.positions.items():
            try:
                if position.get("status") == "DUST":
                    # Dust doesn't occupy capital
                    continue
                
                # Get position value
                qty = float(position.get("quantity", 0.0))
                price = await self.get_latest_price(symbol)
                if qty > 0 and price and price > 0:
                    occupied += qty * float(price)
            except Exception:
                pass
        
        return occupied

    async def get_portfolio_status(self) -> Dict[str, Any]:
        """
        FIX #6: Get authoritative portfolio status with correct dust handling.
        
        Returns complete portfolio picture with dust correctly excluded.
        """
        free_usdt = await self.free_usdt()
        occupied = await self.get_occupied_capital()
        significant_count = await self.get_significant_position_count()
        
        # Classify positions
        classification = await self.classify_positions_by_size()
        dust_count = len(classification["dust"])
        
        # ===== CRITICAL ASSERTION: Catch illegal state =====
        # If is_flat=True, there should be ZERO SIGNIFICANT positions.
        # Dust positions are allowed (and expected after some trades).
        is_flat = significant_count == 0
        
        # We only assert if something is fundamentally wrong: is_flat=True but significant_count > 0
        # (This is technically impossible due to the line above, but serves as a placeholder for state tracking)
        assert not (is_flat and significant_count > 0), (
            f"ILLEGAL STATE: is_flat={is_flat} but significant_count={significant_count} > 0 | "
            f"dust={dust_count}, total_positions={len(self.positions)}"
        )
        
        return {
            "free_usdt": free_usdt,
            "occupied_capital": occupied,
            "significant_positions": significant_count,
            "dust_positions": dust_count,
            "portfolio_flat": significant_count == 0,
            "total_capital": free_usdt + occupied,
        }

    # ===== STEP 2: DECOUPLE CONCEPTS (is_flat, is_starved, is_full, has_significant_positions) =====

    async def is_flat(self) -> bool:
        """
        ✅ DECOUPLED CONCEPT #1: Portfolio is FLAT
        
        Definition: Portfolio has ZERO positions with qty > 0
        
        This is INDEPENDENT of:
        • Capital availability
        • Margin usage
        • Trade history
        
        Returns: True if total_positions == 0, False otherwise
        """
        significant_count = await self.get_significant_position_count()
        return significant_count == 0

    async def is_starved(self) -> bool:
        """
        ✅ DECOUPLED CONCEPT #2: Portfolio is STARVED
        
        Definition: Free quote capital < minimum safe threshold
        
        This is INDEPENDENT of:
        • Flatness (can be flat AND rich, or full AND starved)
        • Position count
        • Trade status
        
        Returns: True if free_usdt < minimum threshold, False otherwise
        """
        free_usdt = await self.free_usdt()
        min_threshold = float(getattr(self.config, "dust_min_quote_usdt", 10.0) or 10.0)
        return free_usdt < min_threshold

    async def is_full(self) -> bool:
        """
        ✅ DECOUPLED CONCEPT #3: Portfolio is FULL
        
        Definition: Occupied capital is approaching max exposure limit
        
        This is INDEPENDENT of:
        • Flatness (can be flat AND full if previous positions still valued high)
        • Capital availability
        • Significant position count
        
        Returns: True if occupied_capital / total_capital > exposure_target, False otherwise
        """
        free_usdt = await self.free_usdt()
        occupied = await self.get_occupied_capital()
        total = free_usdt + occupied
        
        if total <= 0:
            return False  # Can't be "full" if no capital
        
        exposure_pct = occupied / total
        exposure_target = float(getattr(self, "exposure_target", 0.25) or 0.25)  # 25% default
        
        return exposure_pct > exposure_target

    async def has_significant_positions(self) -> bool:
        """
        ✅ DECOUPLED CONCEPT #4: Portfolio HAS SIGNIFICANT POSITIONS
        
        Definition: Count of non-dust positions > 0
        
        This is INDEPENDENT of:
        • Capital state (starved, full, rich)
        • Trade lifecycle
        • Dust positions
        
        Returns: True if significant_positions > 0, False otherwise
        """
        significant_count = await self.get_significant_position_count()
        return significant_count > 0

    async def build_affordability_probe(
        self,
        symbol: str,
        *,
        planned_quote: Optional[float] = None,
        safety_factor: float = 1.10,
        min_notional_override: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Compute whether current free quote can afford trading 'symbol'.
        Returns a dict with keys:
        symbol, ok, amount, code, planned_quote, required_min_quote
        - ok: True if planned_quote >= required_min_quote (when known)
        - amount: if ok==False, amount of quote still missing to satisfy requirement
        - code: 'QUOTE_LT_MIN_NOTIONAL' if exchange minNotional gating; else 'INSUFFICIENT_QUOTE'
        - planned_quote: the quote amount considered (typically free_usdt after reserve policy)
        - required_min_quote: minNotional * safety_factor if available, else None
        """
        sym = self._norm_sym(symbol)
        try:
            # Determine planned quote if caller didn't pass it
            if planned_quote is None:
                planned_quote = await self.free_usdt()
            planned_quote = float(planned_quote or 0.0)

            exit_info = await self.compute_symbol_exit_floor(
                sym,
                fee_bps=self._cfg("EXIT_FEE_BPS", self._cfg("CR_FEE_BPS", 10.0)),
                slippage_bps=self._cfg("EXIT_SLIPPAGE_BPS", self._cfg("CR_PRICE_SLIPPAGE_BPS", 15.0)),
                min_notional_override=min_notional_override,
            )

            required_min_quote: Optional[float] = None
            code = "INSUFFICIENT_QUOTE"
            ok = True
            gap = 0.0

            min_exit_quote = float(exit_info.get("min_exit_quote") or 0.0)
            if min_exit_quote > 0:
                required_min_quote = float(min_exit_quote) * float(safety_factor or 1.0)
                if planned_quote < required_min_quote:
                    ok = False
                    gap = float(required_min_quote - planned_quote)
                    code = "QUOTE_LT_MIN_NOTIONAL"
                else:
                    ok = True
                    gap = 0.0
                    code = "OK"
            else:
                # No exit floor known – treat as generic insufficiency only if no quote at all
                if planned_quote <= 0:
                    ok = False
                    gap = 0.0
                    code = "INSUFFICIENT_QUOTE"
                else:
                    ok = True
                    gap = 0.0
                    code = "OK"

            return {
                "symbol": sym,
                "ok": bool(ok),
                "amount": float(gap),
                "code": code,
                "planned_quote": planned_quote,
                "required_min_quote": required_min_quote,
            }
        except Exception:
            # On error, be conservative and signal not OK with generic code
            return {
                "symbol": sym,
                "ok": False,
                "amount": 0.0,
                "code": "INSUFFICIENT_QUOTE",
                "planned_quote": float(planned_quote or 0.0) if planned_quote is not None else 0.0,
                "required_min_quote": None,
            }

    async def affordability_snapshot(
        self,
        symbol: str,
        *,
        planned_quote: Optional[float] = None,
        min_free_quote_floor_usdt: float = 6.0,
        floor_factor: float = 1.2,
        safety_factor: float = 1.10
    ) -> Dict[str, Any]:
        """
        Build a snapshot used by AppContext startup readiness logs.
        Includes:
        - MarketDataReady
        - FreeUSDT
        - AffordabilityProbe (for the given symbol)
        - PlannedQuoteUsed
        - StartupSanity (coverage, floors)
        """
        try:
            # Planned/spendable quote to use in probe
            if planned_quote is None:
                planned_quote = await self.free_usdt()
            planned_quote = float(planned_quote or 0.0)

            probe = await self.build_affordability_probe(
                symbol,
                planned_quote=planned_quote,
                safety_factor=safety_factor
            )

            # Coverage sanity: if we have any symbol filters cached, assume coverage OKish
            # (AppContext may overwrite with its own precise computation)
            coverage_pct = 100.0 if self.symbol_filters else 0.0

            # Cache a single raw-free balance lookup to avoid duplicate awaits
            bal = await self.get_balance(self.quote_asset)
            free_raw = float(bal.get("free", 0.0))
            snapshot = {
                "MarketDataReady": self.market_data_ready_event.is_set(),
                "BalancesReady": self.balances_ready_event.is_set(),
                "NavReady": self.nav_ready_event.is_set(),
                "FreeUSDT": free_raw,
                "AffordabilityProbe": probe,
                "PlannedQuoteUsed": planned_quote,
                "StartupSanity": {
                    "filters_coverage_pct": coverage_pct,
                    "required_coverage_pct": 80.0,
                    "min_free_quote_floor_usdt": float(min_free_quote_floor_usdt),
                    "free_usdt": free_raw,
                    "floor_factor": float(floor_factor),
                },
            }
            return snapshot
        except Exception:
            # Return a minimal snapshot if anything goes wrong, but never raise here.
            return {
                "MarketDataReady": self.market_data_ready_event.is_set(),
                "BalancesReady": self.balances_ready_event.is_set(),
                "NavReady": self.nav_ready_event.is_set(),
                "FreeUSDT": 0.0,
                "AffordabilityProbe": {
                    "symbol": self._norm_sym(symbol),
                    "ok": False,
                    "amount": 0.0,
                    "code": "ERROR",
                    "planned_quote": float(planned_quote or 0.0) if planned_quote is not None else 0.0,
                    "required_min_quote": None,
                },
                "PlannedQuoteUsed": float(planned_quote or 0.0) if planned_quote is not None else 0.0,
                "StartupSanity": {
                    "filters_coverage_pct": 0.0,
                    "required_coverage_pct": 80.0,
                    "min_free_quote_floor_usdt": float(min_free_quote_floor_usdt),
                    "free_usdt": 0.0,
                    "floor_factor": float(floor_factor),
                },
            }

    async def set_component_health(
        self, component: Component, code: HealthCode, message: str, *, metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        ts = time.time()
        component_key = component.value if hasattr(component, "value") else str(component)
        code_val = code.value if hasattr(code, "value") else str(code)
        data = {
            "status": code_val,
            "message": message,
            "timestamp": ts,
            "metrics": metrics or {},
        }
        self.component_statuses[component_key] = data
        self.system_health = data
        self.metrics["last_health_update"] = ts
        await self.emit_event("HealthStatus", {"component": component_key, **data})
        if component_key == Component.MARKET_DATA_FEED.value and code_val in (HealthCode.ERROR.value, HealthCode.WARN.value):
            self.set_readiness_flag("market_data_ready", False)

    async def emit_health(self, component: str, status: str, reason: str = "", meta: Optional[Dict[str, Any]] = None):
        """Small wrapper used by AppContext contract validation & boot logs."""
        ts = time.time()
        payload = {"status": status, "message": reason, "timestamp": ts, "metrics": meta or {}}
        self.component_statuses[component] = payload
        self.system_health = payload
        self.metrics["last_health_update"] = ts
        await self.emit_event("HealthStatus", {"component": component, **payload})

    # -------- symbol management --------
    async def set_accepted_symbols(self, symbols: Dict[str, Dict[str, Any]], *, allow_shrink: bool = False, merge_mode: bool = False, source: Optional[str] = None) -> None:
        """
        Set or merge accepted symbols into the universe.
        
        Args:
            symbols: Dict of symbol -> metadata to set/merge
            allow_shrink: If False, reject updates that would shrink the universe (unless merge_mode=True)
            merge_mode: If True, merge incoming symbols with existing (additive). If False, replace (default).
            source: Source identifier for logging (e.g., "SymbolScreener", "WalletScannerAgent")
        
        Additive mode (merge_mode=True):
            - Used by discovery agents (SymbolScreener, IPOChaser)
            - Merges incoming symbols with existing ones
            - Cap is applied AFTER merge
            - Shrink rejection is bypassed (since we're adding)
        
        Replace mode (merge_mode=False):
            - Legacy behavior: incoming symbols replace current universe
            - Shrink protection still applies
            - Used for initialization and WalletScannerAgent updates
        """
        if not isinstance(symbols, dict):
            raise SharedStateError("symbols must be a dictionary", ErrorCode.CONFIGURATION_ERROR)
        
        async with self._lock_context("global"):
            current_count = len(self.accepted_symbols)
            
            # === MERGE vs REPLACE LOGIC ===
            if merge_mode:
                # ADDITIVE MODE: Merge incoming with existing
                working_symbols = dict(self.accepted_symbols)  # Start with current
                incoming_count = len(symbols)
                
                # Merge in new symbols (updates if already exist)
                for raw_sym, meta in symbols.items():
                    symbol = self._norm_sym(raw_sym)
                    working_symbols[symbol] = dict(meta or {})
                
                final_count = len(working_symbols)
                self.logger.info(
                    f"[SS] 🔄 MERGE MODE: {current_count} + {incoming_count} = {final_count} symbols (source={source})"
                )
            else:
                # REPLACEMENT MODE: Incoming replaces current
                working_symbols = dict(symbols)
                final_count = len(working_symbols)
                
                # === STRICT SHRINK REJECTION (only in replace mode) ===
                if not allow_shrink and final_count < current_count:
                    self.logger.warning(
                        "[SS] Rejecting shrink because allow_shrink=False. "
                        f"Current={current_count}, Incoming={final_count}, Source={source}"
                    )
                    return
                
                self.logger.info(
                    f"[SS] 🔄 REPLACE MODE: {current_count} → {final_count} symbols (source={source})"
                )
            
            # === CANONICAL GOVERNOR ENFORCEMENT ===
            # Apply governor cap at the authoritative store (SharedState)
            # This ensures NO component can bypass the cap, regardless of code path
            try:
                if hasattr(self, '_app') and self._app and hasattr(self._app, 'capital_symbol_governor'):
                    governor = self._app.capital_symbol_governor
                    if governor:
                        cap = await governor.compute_symbol_cap()
                        
                        if cap is not None and len(working_symbols) > cap:
                            self.logger.info(
                                f"🎛️ CANONICAL GOVERNOR: {len(working_symbols)} → {cap} symbols (at SharedState)"
                            )
                            symbol_items = list(working_symbols.items())
                            working_symbols = dict(symbol_items[:cap])
            except Exception as e:
                self.logger.warning(f"⚠️ Canonical governor enforcement failed: {e}")
            
            # === BUILD FINAL SYMBOL SET ===
            # In replace mode, we remove symbols not in the incoming set (but protect wallet_force)
            # In merge mode, we keep all existing symbols (just add new ones)
            if not merge_mode:
                # REPLACE: Remove symbols not in incoming (but protect wallet_force)
                wanted = {self._norm_sym(k) for k in symbols.keys()}
                current_keys = set(self.accepted_symbols.keys())
                for s in (current_keys - wanted):
                    meta = self.accepted_symbols.get(s, {})
                    # Wallet-force symbols are sticky: only remove if source is WalletScannerAgent
                    if meta.get("accept_policy") == "wallet_force" and source != "WalletScannerAgent":
                        self.logger.debug("🛡️ Protected wallet_force symbol %s from removal", s)
                        continue
                    
                    self.accepted_symbols.pop(s, None)
                    self.symbols.pop(s, None)
            
            # === NORMALIZE & INSERT SYMBOLS ===
            wallet_forced = []
            normal_accepted = []
            for raw_sym, meta in working_symbols.items():
                symbol = self._norm_sym(raw_sym)
                m = dict(meta or {})
                if source: m["source"] = source
                # WalletScannerAgent bypass logic
                if source == "WalletScannerAgent":
                    m["accept_policy"] = "wallet_force"
                    self.accepted_symbols[symbol] = m
                    self.symbols.setdefault(symbol, {}).update(m)
                    wallet_forced.append(symbol)
                else:
                    # Normal path for SymbolScreener, IPOChaser, etc.
                    self.accepted_symbols[symbol] = m
                    self.symbols.setdefault(symbol, {}).update(m)
                    normal_accepted.append(symbol)
            # Emit event with more detail if WalletScannerAgent was the source
            if source == "WalletScannerAgent":
                sym_list = list(self.accepted_symbols.keys())
                await self.emit_event(
                    "AcceptedSymbolsUpdated",
                    {
                        "count": len(self.accepted_symbols),
                        "wallet_forced": wallet_forced,
                        "accept_policy": "wallet_force",
                        "symbols": sym_list,
                        "source": source,
                    }
                )
                # Publish a high-level bus event that downstream agents (e.g., LiquidationAgent) may subscribe to
                await self.publish_event("wallet_scan.accepted", {"symbols": sym_list, "count": len(sym_list)})
            else:
                sym_list = list(self.accepted_symbols.keys())
                await self.emit_event("AcceptedSymbolsUpdated", {"count": len(self.accepted_symbols), "symbols": sym_list, "source": source or "normal"})
                # Publish a SymbolManager-shaped topic to improve cross-component compatibility
                await self.publish_event("symbol_manager.accepted.updated", {"symbols": sym_list, "count": len(sym_list)})
        
        if not self.accepted_symbols_ready_event.is_set():
            self.accepted_symbols_ready_event.set()
            sym_list = list(self.accepted_symbols.keys())
            payload = {
                "count": len(self.accepted_symbols),
                "accept_policy": "wallet_force" if source == "WalletScannerAgent" else "normal",
                "symbols": sym_list,
            }
            await self.emit_event("AcceptedSymbolsReady", payload)
            # Also publish a SymbolManager-style readiness topic that some agents listen to
            await self.publish_event("symbol_manager.accepted.ready", payload)
    def get_accepted_symbol_list(self) -> List[str]:
        """Return the normalized list of currently accepted symbols (read-only view)."""
        return list(self.accepted_symbols.keys())

    def ensure_symbol_caches_consistent(self) -> Dict[str, int]:
        """
        Ensure internal symbol caches are consistent:
        - self.symbols contains every key in self.accepted_symbols
        Returns a small dict of counters for observability.
        """
        added = 0
        for s, meta in self.accepted_symbols.items():
            if s not in self.symbols:
                self.symbols[s] = dict(meta)
                added += 1
        return {"accepted": len(self.accepted_symbols), "symbols": len(self.symbols), "symbols_added": added}

    async def get_accepted_symbols(self) -> Dict[str, Dict[str, Any]]:
        return dict(self.accepted_symbols)

    def get_accepted_symbols_snapshot(self) -> Dict[str, Dict[str, Any]]:
        """P9: Synchronous snapshot of accepted symbols."""
        return dict(self.accepted_symbols)
    async def get_symbols(self) -> List[str]:
        return list(self.accepted_symbols.keys())
    def _norm_sym(self, s: str) -> str:
        return (s or "").upper().replace("/", "")
    def _norm_tf(self, timeframe: str) -> str:
        return str(timeframe or "").strip().lower()

    def _get_dynamic_significant_floor(self) -> float:
        """
        ALIGNMENT FIX: Calculate dynamic significant floor based on risk-based trade sizing.
        
        Ensures MIN_POSITION_VALUE ≤ SIGNIFICANT_FLOOR ≤ MIN_RISK_BASED_TRADE
        
        Logic:
        1. Base floor from config: SIGNIFICANT_POSITION_FLOOR (default 25.0)
        2. Risk-based trade size: Calculated from equity and risk %
        3. Dynamic floor: min(base_floor, risk_trade_size) to avoid slot accounting mismatch
        
        Returns: Dynamic significant floor in USDT
        """
        try:
            # Get base configuration floor
            base_floor = float(
                self._cfg("SIGNIFICANT_POSITION_FLOOR", 25.0) or 25.0
            )
            
            # Get equity for risk-based calculation
            equity = float(getattr(self, "total_equity", 0.0) or 0.0)
            if equity <= 0:
                # No equity yet, use base floor
                return base_floor
            
            # Calculate risk-based trade size
            # Risk per trade: % of available equity
            risk_pct_per_trade = float(self._cfg("RISK_PCT_PER_TRADE", 0.01) or 0.01)  # Default 1%
            risk_amount_usd = equity * risk_pct_per_trade
            
            # Assume a typical SL distance (1% from entry) for conservative floor calculation
            # This ensures the dynamic floor aligns with expected position sizes
            typical_sl_pct = 0.01  # 1% stop loss
            typical_risk_trade_size = risk_amount_usd / typical_sl_pct if typical_sl_pct > 0 else base_floor
            
            # Dynamic floor: align with risk sizing, capped at base floor
            dynamic_floor = min(base_floor, typical_risk_trade_size)
            
            # Ensure floor doesn't go below MIN_POSITION_VALUE
            min_position_value = float(self._cfg("MIN_POSITION_VALUE_USDT", 10.0) or 10.0)
            dynamic_floor = max(min_position_value, dynamic_floor)
            
            return dynamic_floor
            
        except Exception as e:
            self.logger.warning(f"[SS] Error calculating dynamic significant floor: {e}, using base 25.0")
            return 25.0

    def calculate_capital_floor(self, nav: float = 0.0, trade_size: float = 0.0) -> float:
        """
        Calculate dynamic capital floor based on NAV and trade size.
        
        Formula: capital_floor = max(8, NAV * 0.12, trade_size * 0.5)
        
        This ensures:
        - Absolute minimum of $8 (maintenance buffer)
        - At least 12% of NAV reserved (NAV-based safety)
        - At least 50% of typical trade size reserved (trade viability)
        
        Args:
            nav: Net Asset Value in USDT (uses self.nav if not provided)
            trade_size: Typical trade size in USDT (uses configured trade amount if not provided)
            
        Returns:
            Capital floor in USDT
        """
        try:
            # Use provided NAV or get from state
            if nav <= 0:
                nav = float(getattr(self, "nav", 0.0) or 0.0)
            
            # Use provided trade_size or get from config
            if trade_size <= 0:
                trade_size = float(self._cfg("TRADE_AMOUNT_USDT", self._cfg("DEFAULT_PLANNED_QUOTE", 30.0)) or 30.0)
            
            # Calculate three floor candidates
            absolute_min = 8.0
            nav_based = nav * 0.12
            trade_based = trade_size * 0.5
            
            # Capital floor is the maximum of all three components
            capital_floor = max(absolute_min, nav_based, trade_based)
            
            return capital_floor
            
        except Exception as e:
            self.logger.warning(f"[SS] Error calculating capital floor: {e}, using base 8.0")
            return 8.0

    def _significant_position_floor_from_min_notional(self, min_notional: float = 0.0) -> float:
        """Canonical significant-position floor used across Meta/SharedState/TPSL.
        
        FIX #7: Now uses dynamic floor to align with risk-based trade sizing.
        This prevents slot accounting mismatches where SIGNIFICANT_FLOOR > actual_risk_trade_size
        """
        # Get dynamic significant floor based on equity and risk parameters
        dynamic_floor = self._get_dynamic_significant_floor()
        
        # Fallback to static config if dynamic calculation is unavailable
        strategy_floor = float(
            self._cfg(
                "SIGNIFICANT_POSITION_FLOOR",
                self._cfg(
                    "MIN_SIGNIFICANT_POSITION_USDT",
                    self._cfg("MIN_SIGNIFICANT_USD", 25.0),
                ),
            )
            or 25.0
        )
        
        min_position_value = float(self._cfg("MIN_POSITION_VALUE_USDT", 10.0) or 10.0)
        
        # Use dynamic floor as primary, with fallbacks to exchange min_notional and min_position_value
        return max(float(min_notional or 0.0), min_position_value, dynamic_floor)

    def _cached_min_notional(self, symbol: str) -> float:
        try:
            filters = dict(self.symbol_filters.get(self._norm_sym(symbol), {}) or {})
            _step, _min_qty, _tick, min_notional = self._extract_symbol_filter_values(filters)
            return float(min_notional or 0.0)
        except Exception:
            return 0.0

    async def get_significant_position_floor(self, symbol: str) -> float:
        """Async floor resolver that prefers live exchange trade rules."""
        min_notional = 0.0
        try:
            _lot_step, min_notional = await self.compute_symbol_trade_rules(symbol)
        except Exception:
            min_notional = 0.0
        if min_notional <= 0:
            min_notional = self._cached_min_notional(symbol)
        return float(self._significant_position_floor_from_min_notional(min_notional))

    def _estimate_position_value_usdt(
        self,
        symbol: str,
        position_data: Dict[str, Any],
        price_hint: float = 0.0,
    ) -> float:
        """Best-effort mark-to-market position value."""
        pos = position_data if isinstance(position_data, dict) else {}
        qty = float(pos.get("quantity", 0.0) or pos.get("qty", 0.0) or 0.0)
        if qty <= 0:
            return 0.0
        sym = self._norm_sym(symbol)
        price = float(price_hint or 0.0)
        if price <= 0:
            price = float(self.latest_prices.get(sym, 0.0) or 0.0)
        if price <= 0:
            price = float(
                pos.get("mark_price", 0.0)
                or pos.get("avg_price", 0.0)
                or pos.get("entry_price", 0.0)
                or pos.get("price", 0.0)
                or 0.0
            )
        if price > 0:
            return float(qty * price)
        return float(pos.get("value_usdt", 0.0) or 0.0)

    def classify_position_snapshot(
        self,
        symbol: str,
        position_data: Dict[str, Any],
        *,
        floor_hint: float = 0.0,
        price_hint: float = 0.0,
    ) -> Tuple[bool, float, float]:
        """
        Sync significance check for runtime paths that cannot await.
        Returns: (is_open_significant, value_usdt, significant_floor)
        """
        pos = position_data if isinstance(position_data, dict) else {}
        value_usdt = self._estimate_position_value_usdt(symbol, pos, price_hint=price_hint)
        floor = float(floor_hint or pos.get("significant_floor_usdt", 0.0) or 0.0)
        if floor <= 0:
            floor = self._significant_position_floor_from_min_notional(self._cached_min_notional(symbol))
        is_open = bool(value_usdt >= max(floor, 0.0) and value_usdt > 0.0)
        return is_open, float(value_usdt), float(floor)

    def get_ohlcv_count(self, symbol: str, timeframe: str) -> int:
        sym = self._norm_sym(symbol)
        tf = self._norm_tf(timeframe)
        rows = self.market_data.get((sym, tf))
        if rows is None:
            rows = self.market_data.get((sym, str(timeframe or "").strip()))
        if rows is None:
            rows = self.market_data.get((symbol, timeframe))
        return len(rows or [])

    def have_min_bars(self, symbols: list[str], timeframe: str, min_bars: int) -> bool:
        return all(self.get_ohlcv_count(s, timeframe) >= min_bars for s in symbols)

    async def _maybe_set_market_data_ready(self, *, timeframe: str = "5m", min_bars: int = 50) -> None:
        if self.accepted_symbols and not self.market_data_ready_event.is_set():
            syms = list(self.accepted_symbols.keys())
            if self.have_min_bars(syms, timeframe, min_bars):
                self.market_data_ready_event.set()
                self.logger.warning(
                    "[DEBUG_MDF_SET] shared_state_id=%s event_id=%s",
                    id(self),
                    id(self.market_data_ready_event),
                )
                await self.emit_event("MarketDataReady", {"symbols": syms, "timeframe": timeframe, "min_bars": min_bars})

    def is_symbol_tradable(self, symbol: str) -> bool:
        return self._norm_sym(symbol) in self.accepted_symbols

    # -------- price management --------
    async def compute_symbol_trade_rules(self, symbol: str) -> Tuple[float, float]:
        """
        Return (lot_step, min_notional) for a symbol, supporting both RAW (Binance-shaped)
        and normalized filter schemas. If an exchange client is present, refresh cache first.
        """
        sym = self._norm_sym(symbol)
        f = await self._fetch_and_cache_symbol_filters(sym)
        step_size, _min_qty, _tick_size, min_notional = self._extract_symbol_filter_values(f)
        return step_size, min_notional

    async def _fetch_and_cache_symbol_filters(self, sym: str) -> Dict[str, Any]:
        """Fetch symbol filters from exchange (raw → normalized fallback) and cache in symbol_filters."""
        f = dict(self.symbol_filters.get(sym, {}))
        if self._exchange_client:
            try:
                if hasattr(self._exchange_client, "ensure_symbol_filters_ready"):
                    await self._exchange_client.ensure_symbol_filters_ready(sym)
                raw = await self._exchange_client.get_symbol_filters_raw(sym) if hasattr(self._exchange_client, "get_symbol_filters_raw") else {}
            except Exception:
                raw = {}
            if isinstance(raw, dict) and raw:
                f = dict(raw)
                self.symbol_filters[sym] = dict(raw)
            elif not f:
                try:
                    norm = await self._exchange_client.get_symbol_filters(sym) if hasattr(self._exchange_client, "get_symbol_filters") else {}
                except Exception:
                    norm = {}
                if isinstance(norm, dict) and norm:
                    f = {"_normalized": dict(norm)}
                    self.symbol_filters[sym] = dict(f)
        return f

    def _extract_symbol_filter_values(self, filters: Dict[str, Any]) -> Tuple[float, float, float, float]:
        """Return (step_size, min_qty, tick_size, min_notional) from raw or normalized filters."""
        f = filters or {}
        lot = f.get("LOT_SIZE") or f.get("MARKET_LOT_SIZE") or {}
        price = f.get("PRICE_FILTER") or {}
        notional = f.get("MIN_NOTIONAL") or f.get("NOTIONAL") or {}
        norm = f.get("_normalized", {})

        step_size = float(
            lot.get("stepSize")
            or f.get("stepSize")
            or norm.get("step_size")
            or 0.0
        )
        min_qty = float(
            lot.get("minQty")
            or f.get("minQty")
            or norm.get("min_qty")
            or norm.get("min_quantity")
            or 0.0
        )
        tick_size = float(
            price.get("tickSize")
            or f.get("tickSize")
            or norm.get("tick_size")
            or 0.0
        )
        min_notional = float(
            notional.get("minNotional")
            or f.get("minNotional")
            or norm.get("min_notional")
            or 0.0
        )
        return step_size, min_qty, tick_size, min_notional

    async def compute_symbol_exit_floor(
        self,
        symbol: str,
        *,
        price: Optional[float] = None,
        fee_bps: Optional[float] = None,
        slippage_bps: Optional[float] = None,
        min_notional_override: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Compute the minimum safe quote required to exit a position immediately.
        Uses exchange filters + fee/slippage buffers to ensure SELL feasibility.
        """
        sym = self._norm_sym(symbol)

        # Get best-effort price
        if price is None:
            try:
                price = await self.get_latest_price(sym)
            except Exception:
                price = None
        if (not price or price <= 0) and self._exchange_client:
            try:
                if hasattr(self._exchange_client, "get_current_price"):
                    price = float(await self._exchange_client.get_current_price(sym))
                elif hasattr(self._exchange_client, "get_ticker_price"):
                    price = float(await self._exchange_client.get_ticker_price(sym))
            except Exception:
                price = price or 0.0

        # Fetch filters (refresh cache if supported)
        f = await self._fetch_and_cache_symbol_filters(sym)
        step_size, min_qty, tick_size, min_notional = self._extract_symbol_filter_values(f)
        if min_notional_override is not None:
            min_notional = float(min_notional_override)

        fee_bps_val = float(fee_bps if fee_bps is not None else self._cfg("EXIT_FEE_BPS", self._cfg("CR_FEE_BPS", 10.0)))
        slippage_bps_val = float(slippage_bps if slippage_bps is not None else self._cfg("EXIT_SLIPPAGE_BPS", self._cfg("CR_PRICE_SLIPPAGE_BPS", 15.0)))
        fee_buffer = max(0.0, fee_bps_val) / 10000.0
        slippage_buffer = max(0.0, slippage_bps_val) / 10000.0

        min_notional_floor = float(min_notional or 0.0) * (1.0 + fee_buffer + slippage_buffer)
        min_qty_floor = 0.0
        if min_qty and price and price > 0:
            min_qty_floor = float(min_qty) * float(price) * (1.0 + fee_buffer)

        # --- Entry floor components (exit-feasibility + fees + volatility) ---
        # Expected round-trip fee (in quote) = min_notional × round_trip_fee_rate
        try:
            round_trip_fee_bps = float(self._cfg("ROUND_TRIP_FEE_BPS", 0.0) or 0.0)
        except Exception:
            round_trip_fee_bps = 0.0
        if round_trip_fee_bps <= 0:
            round_trip_fee_bps = max(0.0, float(fee_bps_val) * 2.0)
        expected_round_trip_fee = float(min_notional or 0.0) * (round_trip_fee_bps / 10000.0)
        round_trip_fee_rate = float(round_trip_fee_bps / 10000.0) if round_trip_fee_bps > 0 else 0.0

        # Volatility-adjusted min move (quote) using ATR% when available
        atr = 0.0
        if price and price > 0:
            try:
                if hasattr(self, "calc_atr"):
                    atr = float(await self.calc_atr(sym, "5m", 14) or 0.0)
                    if atr <= 0:
                        atr = float(await self.calc_atr(sym, "1m", 14) or 0.0)
            except Exception:
                atr = 0.0
        try:
            fallback_atr_pct = float(self._cfg("TPSL_FALLBACK_ATR_PCT", 0.01) or 0.01)
        except Exception:
            fallback_atr_pct = 0.01
        try:
            atr_mult = float(self._cfg("ENTRY_MIN_MOVE_ATR_MULT", 1.0) or 1.0)
        except Exception:
            atr_mult = 1.0
        atr_pct = (float(atr) / float(price)) if (atr and price and price > 0) else 0.0
        volatility_move_pct = max(float(fallback_atr_pct), float(atr_pct) * float(atr_mult))
        volatility_adjusted_min_move = float(min_notional or 0.0) * float(volatility_move_pct)

        # Profitability sizing: require expected move × position size to exceed fees × multiplier
        expected_move_fee_mult = float(self._cfg("ENTRY_EXPECTED_MOVE_FEE_MULT", 1.6) or 1.6)
        expected_move_fee_mult = max(1.6, float(expected_move_fee_mult))
        profitability_floor = 0.0
        if round_trip_fee_rate > 0:
            required_move_pct = float(expected_move_fee_mult) * float(round_trip_fee_rate)
            denom = max(float(volatility_move_pct), 1e-9)
            profitability_mult = max(1.0, float(required_move_pct) / denom)
            profitability_floor = float(min_notional or 0.0) * float(profitability_mult)

        min_exit_quote = max(min_notional_floor, min_qty_floor)
        min_entry_quote = max(
            float(min_notional or 0.0),
            float(3.0 * expected_round_trip_fee),
            float(volatility_adjusted_min_move),
            float(profitability_floor),
        )
        return {
            "min_exit_quote": float(min_exit_quote),
            "min_entry_quote": float(min_entry_quote),
            "min_notional": float(min_notional or 0.0),
            "min_qty": float(min_qty or 0.0),
            "step_size": float(step_size or 0.0),
            "tick_size": float(tick_size or 0.0),
            "price": float(price or 0.0),
            "fee_bps": float(fee_bps_val),
            "slippage_bps": float(slippage_bps_val),
            "round_trip_fee_bps": float(round_trip_fee_bps),
            "expected_round_trip_fee": float(expected_round_trip_fee),
            "volatility_move_pct": float(volatility_move_pct),
            "volatility_adjusted_min_move": float(volatility_adjusted_min_move),
            "expected_move_fee_mult": float(expected_move_fee_mult),
            "profitability_floor": float(profitability_floor),
        }

    async def compute_min_entry_quote(
        self,
        symbol: str,
        *,
        default_quote: Optional[float] = None,
        price: Optional[float] = None,
        fee_bps: Optional[float] = None,
        slippage_bps: Optional[float] = None,
        min_notional_override: Optional[float] = None,
    ) -> float:
        """Compute the dynamic minimum entry quote based on exit feasibility."""
        base_quote = float(default_quote if default_quote is not None else self._cfg("DEFAULT_PLANNED_QUOTE", self._cfg("MIN_ENTRY_QUOTE_USDT", 0.0)))
        exit_info = await self.compute_symbol_exit_floor(
            symbol,
            price=price,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            min_notional_override=min_notional_override,
        )
        min_entry_quote = float(exit_info.get("min_entry_quote", 0.0))
        min_exit_quote = float(exit_info.get("min_exit_quote", 0.0))
        return max(float(min_exit_quote), float(min_entry_quote), float(base_quote or 0.0))
    
    @track_performance
    async def update_latest_price(self, symbol: str, price: float) -> None:
        sym = self._norm_sym(symbol)
        p = float(price)
        if p <= 0:
            raise SharedStateError(f"Invalid price for {symbol}: {price}", ErrorCode.CONFIGURATION_ERROR)
        async with self._lock_context("prices"):
            self.latest_prices[sym] = p
            self._last_tick_timestamps[sym] = time.time()
            self._price_cache[sym] = (p, time.time())

    async def update_last_price(self, symbol: str, price: float) -> None:
        """Alias used by MDF to update last price."""
        await self.update_latest_price(symbol, price)

    async def get_latest_price(self, symbol: str) -> Optional[float]:
        return self.latest_prices.get(self._norm_sym(symbol))
    async def get_all_prices(self) -> Dict[str, float]:
        return dict(self.latest_prices)

    async def ensure_latest_prices_coverage(self, price_fetcher: Callable[[str], Any]) -> None:
        """
        Ensure self.latest_prices has coverage for all relevant symbols (accepted + held).
        If missing, uses price_fetcher(symbol) to populate.
        """
        # 1. Gather all candidates
        candidates = set(self.accepted_symbols.keys())
        
        # Add from balances (wallet)
        if hasattr(self, "balances") and self.balances:
            for asset, amt in self.balances.items():
                if asset != "USDT": # assume USDT base
                    candidates.add(f"{asset}USDT")
        
        # Add from positions
        if hasattr(self, "positions") and self.positions:
            for s in self.positions.keys():
                candidates.add(s)

        # 2. Check coverage
        missing = [s for s in candidates if s not in self.latest_prices]
        if not missing:
            return

        # 3. Fetch missing
        logging.getLogger("SharedState").info(f"Populating price cache for {len(missing)} symbols...")
        for sym in missing:
            try:
                p = await price_fetcher(sym)
                if p is None and asyncio.iscoroutine(p):
                    p = await p
                
                if p:
                    await self.update_latest_price(sym, float(p))
            except Exception:
                pass

    # -------- OHLCV ingestion (required by MDF) --------
    async def add_ohlcv(self, symbol: str, timeframe: str, bar: OHLCVBar) -> None:
        """
        Append/merge a single OHLCV bar ensuring ascending ts and 6-field hygiene.
        bar keys: ts,o,h,l,c,v  (epoch seconds float)
        """
        sym = self._norm_sym(symbol)
        tf = self._norm_tf(timeframe)
        key = (sym, tf)
        b = {
            "ts": float(bar["ts"]),
            "o": float(bar["o"]),
            "h": float(bar["h"]),
            "l": float(bar["l"]),
            "c": float(bar["c"]),
            "v": float(bar["v"]),
        }
        async with self._lock_context("market_data"):
            # Canonicalize legacy non-normalized keys on write.
            legacy_key = (symbol, timeframe)
            if legacy_key != key and legacy_key in self.market_data and key not in self.market_data:
                self.market_data[key] = list(self.market_data.get(legacy_key) or [])
                self.market_data.pop(legacy_key, None)
            lst = self.market_data.setdefault(key, [])
            if lst and abs(lst[-1]["ts"] - b["ts"]) < 1e-9:
                lst[-1] = b
            else:
                lst.append(b)
                if len(lst) >= 2 and lst[-2]["ts"] > lst[-1]["ts"]:
                    lst.sort(key=lambda r: r["ts"])
            # Invalidate ATR cache entries for this (symbol, timeframe)
            self._atr_cache = {
                k: v
                for k, v in self._atr_cache.items()
                if not (
                    (k[0] == sym and k[1] == tf)
                    or (k[0] == symbol and k[1] == timeframe)
                )
            }
        # price keep-warm
        await self.update_latest_price(sym, b["c"])
        # Do not set MarketDataReady here; rely on coverage check
        await self._maybe_set_market_data_ready()

    async def set_market_data(self, symbol: str, timeframe: str, ohlcv_data: List[Dict[str, Any]]) -> None:
        """Batch set (not used by MDF warmup, but kept for completeness)."""
        sym = self._norm_sym(symbol)
        tf = self._norm_tf(timeframe)
        key = (sym, tf)
        norm: List[OHLCVBar] = []
        for r in ohlcv_data or []:
            if {"ts","o","h","l","c","v"} <= r.keys():
                norm.append(OHLCVBar(ts=float(r["ts"]), o=float(r["o"]), h=float(r["h"]), l=float(r["l"]), c=float(r["c"]), v=float(r["v"])) )
            else:
                ts = float(r.get("ts") or r.get("timestamp") or r.get("time") or 0.0)
                norm.append(OHLCVBar(
                    ts=ts,
                    o=float(r.get("o") or r.get("open") or 0.0),
                    h=float(r.get("h") or r.get("high") or 0.0),
                    l=float(r.get("l") or r.get("low")  or 0.0),
                    c=float(r.get("c") or r.get("close") or 0.0),
                    v=float(r.get("v") or r.get("volume") or 0.0),
                ))
        norm.sort(key=lambda x: x["ts"])
        async with self._lock_context("market_data"):
            legacy_key = (symbol, timeframe)
            if legacy_key != key:
                self.market_data.pop(legacy_key, None)
            self.market_data[key] = norm
            # Invalidate ATR cache entries for this (symbol, timeframe)
            self._atr_cache = {
                k: v
                for k, v in self._atr_cache.items()
                if not (
                    (k[0] == sym and k[1] == tf)
                    or (k[0] == symbol and k[1] == timeframe)
                )
            }
        if norm:
            await self.update_latest_price(sym, norm[-1]["c"])
        # Do not set MarketDataReady here; rely on coverage check
        await self._maybe_set_market_data_ready()

    async def get_market_data(self, symbol: str, timeframe: str) -> Optional[List[OHLCVBar]]:
        sym = self._norm_sym(symbol)
        tf = self._norm_tf(timeframe)
        rows = self.market_data.get((sym, tf))
        if rows is None:
            rows = self.market_data.get((sym, str(timeframe or "").strip()))
        if rows is None:
            rows = self.market_data.get((symbol, timeframe))
        return rows

    # -------- ATR utility (used by MDF warm cache) --------
    async def calc_atr(self, symbol: str, timeframe: str, period: int = 14) -> Optional[float]:
        sym = self._norm_sym(symbol)
        tf = self._norm_tf(timeframe)
        key = (sym, tf)
        rows = self.market_data.get(key)
        if rows is None:
            rows = self.market_data.get((sym, str(timeframe or "").strip()))
        if rows is None:
            rows = self.market_data.get((symbol, timeframe))
        rows = rows or []
        if len(rows) < max(2, period+1):
            return None
        cache_key = (sym, tf, period)
        if cache_key in self._atr_cache:
            return self._atr_cache[cache_key]
        # Compute True Range & ATR
        trs: List[float] = []
        for i in range(1, len(rows)):
            c_prev = rows[i-1]["c"]
            h = rows[i]["h"]
            l = rows[i]["l"]
            c = rows[i]["c"]
            tr = max(h - l, abs(h - c_prev), abs(l - c_prev))
            trs.append(tr)
        if len(trs) < period:
            return None
        # Wilder's smoothing or simple average for first ATR
        atr = sum(trs[-period:]) / float(period)
        self._atr_cache[cache_key] = float(atr)
        return float(atr)

    # -------- balances / portfolio --------

    # small numeric helper used by inventory scanner
    @staticmethod
    def _round_step(value: float, step: float) -> float:
        try:
            if step <= 0:
                return float(value)
            return float((int(value / step)) * step)
        except Exception:
            return float(value)
    @track_performance
    async def update_balances(self, balances: Dict[str, Dict[str, float]]) -> None:
        """Update balances with change detection and reservation reconciliation.
        
        FIX #2: Reconciliation logic prevents phantom capital loss on sync
        """
        if not isinstance(balances, dict):
            raise SharedStateError("balances must be a dictionary", ErrorCode.CONFIGURATION_ERROR)
        changed_assets: List[str] = []
        async with self._lock_context("balances"):
            for asset, data in balances.items():
                if not isinstance(data, dict):
                    continue
                a = asset.upper()
                new_free = max(0.0, float(data.get("free", 0.0)))
                new_locked = max(0.0, float(data.get("locked", 0.0)))
                
                # Get previous balance
                prev = self.balances.get(a)
                prev_free = float(prev.get("free", 0.0)) if prev else 0.0
                prev_locked = float(prev.get("locked", 0.0)) if prev else 0.0
                
                # FIX #2: Only update if there's an actual change
                if prev and (prev_free == new_free and prev_locked == new_locked):
                    # No change, skip
                    continue
                
                changed_assets.append(a)
                
                # FIX #2: Reconcile against reservations before overwrite
                if a in self._quote_reservations:
                    reserved = sum(r.get("amount", 0.0) for r in self._quote_reservations[a])
                    expected_free = max(0.0, prev_free - reserved) if prev else 0.0
                    
                    # Warn if discrepancy is larger than tolerance (e.g. 1%)
                    tolerance = abs(expected_free * 0.01) if expected_free > 0 else 0.01
                    if abs(new_free - expected_free) > tolerance:
                        self.logger.warning(
                            f"[SS:BalanceReconciliation] Asset={a}: "
                            f"expected_free={expected_free:.4f}, "
                            f"actual_free={new_free:.4f}, "
                            f"reserved={reserved:.4f}, "
                            f"delta={new_free - expected_free:.4f}"
                        )
                
                # FIX #2: Only update changed fields, preserve metadata
                if prev:
                    prev.update({"free": new_free, "locked": new_locked})
                    self.balances[a] = prev
                else:
                    self.balances[a] = {"free": new_free, "locked": new_locked}
                
                self.logger.debug(f"[SS:BalanceUpdate] {a}: free={new_free}, locked={new_locked}")
            self.metrics["balances_updated_at"] = time.time()
            # Mark balances ready on first successful update
            if not self.balances_ready_event.is_set():
                self.balances_ready_event.set()
                self.metrics["balances_ready"] = True
                await self.emit_event("BalancesReady", {"assets": list(self.balances.keys())})
            # NAV becomes derivable once balances are present; mark as ready.
            await self._maybe_set_nav_ready()
        # Emit a summary event outside the lock
        if changed_assets:
            await self.emit_event("BalancesUpdated", {"assets": changed_assets, "count": len(self.balances)})
        # Optionally mirror balances into spot positions for inventory/liq workflows
        # CRITICAL: Never hydrate positions from balances in shadow mode
        # In shadow mode, positions are managed entirely by virtual_positions
        # and must not be overwritten by exchange balances
        try:
            if (
                getattr(self.config, "auto_positions_from_balances", True)
                and self.trading_mode != "shadow"
            ):
                await self.hydrate_positions_from_balances()
        except Exception as e:
            self.logger.warning(f"hydrate_positions_from_balances failed: {e}")

    async def get_balance(self, asset: str) -> Dict[str, float]:
        """P9: Authoritative balance retrieval with mandatory freshness check."""
        a = asset.upper()
        # If balance is missing or older than 3 seconds, we could trigger a refresh
        # but for Meta tick performance we rely on the background syncer.
        return self.balances.get(a, {"free": 0.0, "locked": 0.0})

    async def sync_authoritative_balance(self, force: bool = False) -> None:
        """
        P9: Force a hard sync of balances from the exchange to prevent phantom capital.
        
        FIX #6: Added force parameter for hard_reset_capital_state() startup sequence.
        When force=True, bypasses any throttling and force-refreshes immediately.
        
        SURGICAL FIX #2: In shadow mode, treat real balances as read-only reference snapshot.
        Never overwrite self.balances in shadow mode - all trading must use virtual ledgers.
        This prevents exchange corrections from wiping out shadow positions.
        """
        if self._exchange_client and hasattr(self._exchange_client, "get_spot_balances"):
            try:
                new_bals = await self._exchange_client.get_spot_balances()
                if new_bals:
                    async with self._lock_context("balances"):
                        # SURGICAL FIX #2: Only update real balances if NOT in shadow mode
                        # In shadow mode, self.balances is read-only reference snapshot
                        if self.trading_mode != "shadow":
                            # FIX #5: NORMALIZE KEYS TO UPPERCASE - sync_authoritative_balance bypass bug
                            # Problem: Exchange returns {"usdt": {...}, "btc": {...}} (lowercase)
                            # But get_balance() looks for uppercase keys, causing mismatches
                            # Solution: Normalize all keys to uppercase like update_balances() does
                            for asset, data in new_bals.items():
                                if isinstance(data, dict):
                                    a = asset.upper()
                                    self.balances[a] = data
                        # Always update last sync timestamp (even in shadow mode)
                        self.last_balance_sync = time.time()
                        # FIX #2: Ensure balances_ready_event is set, even if sync_authoritative_balance called directly
                        # Problem: Only update_balances() was setting the ready event, not sync_authoritative_balance()
                        # This caused MetaController to block indefinitely waiting for balances to be marked ready
                        if not self.balances_ready_event.is_set():
                            self.balances_ready_event.set()
                            self.metrics["balances_ready"] = True
                    log_level = "warning" if force else "info"
                    msg = "[SS] Authoritative balance sync complete (FORCE)" if force else "[SS] Authoritative balance sync complete."
                    if self.trading_mode == "shadow":
                        msg += " [SHADOW MODE - balances not updated, virtual ledger is authoritative]"
                    if log_level == "warning":
                        self.logger.warning(msg)
                    else:
                        self.logger.info(msg)
            except Exception as e:
                self.logger.error(f"[SS] Failed to sync authoritative balance: {e}")

    async def init_virtual_portfolio_from_real_snapshot(self) -> None:
        """
        Initialize virtual portfolio (shadow mode) from real balance snapshot.
        
        P9 SHADOW MODE: Copies real balances to virtual ledger, enabling simulated trading
        without touching real Binance positions. Called once at boot if TRADING_MODE="shadow".
        
        Key behaviors:
        - Copies real balances to virtual_balances
        - Initializes virtual_positions as empty (start fresh)
        - Records shadow mode start time for monitoring
        - Sets high water mark for drawdown calculation
        - Emits PortfolioSnapshot (virtual) for observability
        """
        if self.trading_mode != "shadow":
            self.logger.info("[SS:ShadowMode] Not initializing virtual portfolio (trading_mode != shadow)")
            return
        
        try:
            async with self._lock_context("balances"):
                # Copy real balances to virtual ledger
                self.virtual_balances = {
                    asset: {"free": bal.get("free", 0.0), "locked": bal.get("locked", 0.0)}
                    for asset, bal in self.balances.items()
                }
                
                # Initialize virtual positions (empty, starting fresh)
                self.virtual_positions = {}
                
                # Initialize PnL tracking
                self.virtual_realized_pnl = 0.0
                self.virtual_unrealized_pnl = 0.0
                
                # Compute initial NAV (all in quote asset)
                quote_bal = self.virtual_balances.get(self.quote_asset, {}).get("free", 0.0)
                self.virtual_nav = float(quote_bal)
                self._shadow_mode_high_water_mark = self.virtual_nav
                
                # Record start time
                self._shadow_mode_start_time = time.time()
                
                self.logger.info(
                    f"[SS:ShadowMode] Virtual portfolio initialized. "
                    f"Quote balance: {quote_bal:.2f}, NAV: {self.virtual_nav:.2f}"
                )
                
                # Emit event for observability
                await self.emit_event("ShadowModeInitialized", {
                    "virtual_balances": {k: v.get("free", 0.0) for k, v in self.virtual_balances.items()},
                    "virtual_nav": self.virtual_nav,
                    "start_time": self._shadow_mode_start_time,
                })
        except Exception as e:
            self.logger.error(f"[SS:ShadowMode] Failed to initialize virtual portfolio: {e}", exc_info=True)

    def get_virtual_balance(self, asset: str) -> Dict[str, float]:
        """Get virtual balance for an asset (shadow mode only)."""
        return self.virtual_balances.get(asset.upper(), {"free": 0.0, "locked": 0.0})

    async def get_spendable_balance(self, asset: str, *, reserve_ratio: Optional[float] = None, min_reserve: Optional[float] = None) -> float:

        """CANONICAL: Compute spendable balance for an asset after reserves and reservation cleanup.

        All other spendable-balance methods delegate here:
          free_usdt() -> get_spendable_quote() -> get_spendable_balance()
          get_free_quote() -> get_spendable_quote() -> get_spendable_balance()
          get_spendable_usdt() -> get_spendable_balance()
        """
        a = asset.upper()
        bal = await self.get_balance(a)
        
        # FIX #3: Diagnostic logging for quote asset mismatch detection
        # If balance is zero, log available assets to help diagnose quote_asset mismatch
        if (not bal.get("free", 0.0)) and (not bal.get("locked", 0.0)):
            available_assets = list(self.balances.keys())
            self.logger.warning(
                f"[SS:BalanceWarning] Queried asset {a} not found (zero balance). "
                f"Available assets in balances: {available_assets}. "
                f"This may indicate quote_asset configuration mismatch between MetaController and SharedState."
            )
        
        # FIX #1: Read both free and locked amounts properly
        free = float(bal.get("free", 0.0))
        locked = float(bal.get("locked", 0.0))
        total = free + locked
        
        # Log for audit trail
        self.logger.debug(f"[SS:Balance] {a}: free={free}, locked={locked}, total={total}")
        
        # For spendable calculation, typically only "free" is immediately spendable
        # but we track both for inventory purposes
        available = free  # The actually-spendable amount (not locked)
        
        rr = self.config.quote_reserve_ratio if reserve_ratio is None else float(reserve_ratio)
        mr = self.config.quote_min_reserve if min_reserve is None else float(min_reserve)

        now = time.time()
        # LAYER 8: AGGRESSIVE TTL CLEANUP - Emergency fix for stale reservation locks
        # Problem: Old reservations with bad/missing TTL can block all capital
        # Solution: 1) Remove expired (TTL passed), 2) Remove >60s old, 3) Remove missing TTL
        all_reservations = self._quote_reservations.get(a, [])
        cleaned_reservations = []
        freed_amount = 0.0
        
        max_reservation_age_sec = 90  # Hard ceiling: no reservation survives >90s

        for r in all_reservations:
            expires_at = r.get("expires_at", 0)

            # Skip if missing expires_at (invalid reservation)
            if not expires_at or expires_at <= 0:
                freed_amount += float(r.get("amount", 0.0))
                continue

            # Skip if already expired (TTL passed)
            if expires_at <= now:
                freed_amount += float(r.get("amount", 0.0))
                continue

            # Skip if created too long ago (emergency force-expire).
            # Use stored created_at when available; fall back to default TTL estimate.
            created_at = r.get("created_at", 0)
            if not created_at or created_at <= 0:
                # Legacy reservation without created_at: estimate from default TTL
                created_at = expires_at - float(self.config.reservation_default_ttl)
            age_sec = now - created_at
            if age_sec > max_reservation_age_sec:
                freed_amount += float(r.get("amount", 0.0))
                continue

            # Valid reservation - keep it
            cleaned_reservations.append(r)
        
        self._quote_reservations[a] = cleaned_reservations
        
        # Log cleanup if capital was freed
        if freed_amount > 0.01:
            self.logger.info(f"[SS:Cleanup] Purged stale reservations. Freed: ${freed_amount:.2f} (count={len(all_reservations)-len(cleaned_reservations)})")
        
        reserved = sum(float(r.get("amount", 0.0)) for r in cleaned_reservations)

        # CRITICAL FIX: Bootstrap deadlock prevention (Fix #4)
        # When portfolio is completely flat (no reserved capital) and balance is critically low,
        # relax safety reserve to minimal $0.50 to allow first trade to execute
        spendable_with_full_reserve = available - reserved - max(available * rr, mr)
        
        if reserved == 0 and spendable_with_full_reserve < 5.0 and available > 5.0:
            # Flat portfolio with starved capital: use minimal reserve ($0.50) instead of full
            # This allows the first BUY to execute when startup has consumed all capital via safety reserves
            self.logger.info(f"[SS:BootstrapFix] Flat portfolio with capital starvation. Using minimal reserve. Available: ${available:.2f} → Spendable: ${max(0.0, available - reserved - 0.50):.2f}")
            return max(0.0, available - reserved - 0.50)
        
        safety_reserve = max(available * rr, mr)
        return max(0.0, available - reserved - safety_reserve)

    async def get_free_balance(self, asset: str) -> float:
        """Compatibility alias: return raw free balance for an asset."""
        bal = await self.get_balance(asset)
        try:
            return float(bal.get("free", 0.0))
        except Exception:
            return 0.0

    async def force_cleanup_expired_reservations(
        self,
        asset: str = "USDT",
        max_age_sec: float = 60.0
    ) -> tuple:
        """
        EMERGENCY CLEANUP: Nuclear option for capital-starved situations.
        Force-removes ANY reservation older than max_age_sec seconds, regardless of TTL.
        
        Args:
            asset: Quote asset to clean (default: USDT)
            max_age_sec: Age threshold in seconds (default: 60.0)
        
        Returns: (count_removed, capital_freed)
        """
        a = asset.upper()
        all_reservations = self._quote_reservations.get(a, [])
        now = time.time()
        
        cleaned = []
        freed = 0.0
        removed = 0
        
        for r in all_reservations:
            expires_at = r.get("expires_at", 0)
            created_at = r.get("created_at", 0)
            if not created_at or created_at <= 0:
                # Legacy reservation: estimate creation from default TTL
                created_at = (expires_at - float(self.config.reservation_default_ttl)) if expires_at > 0 else 0
            age_sec = (now - created_at) if created_at > 0 else float('inf')

            # Force remove if older than max_age_sec or invalid
            if age_sec > max_age_sec or expires_at <= 0:
                freed += float(r.get("amount", 0.0))
                removed += 1
            else:
                cleaned.append(r)
        
        self._quote_reservations[a] = cleaned
        
        if removed > 0:
            self.logger.warning(f"[SS:EmergencyCleanup] Force-removed {removed} old reservations. Freed: ${freed:.2f}")
        
        return (removed, freed)

    async def get_spendable_quote(self, asset: str, *, reserve_ratio: float = 0.10, min_reserve: float = 0.0) -> float:
        """Alias → get_spendable_balance(). Used by ml_forecaster, free_usdt(), get_free_quote()."""
        return await self.get_spendable_balance(asset, reserve_ratio=reserve_ratio, min_reserve=min_reserve)

    async def get_free_quote(self) -> float:
        """Alias → get_spendable_quote(quote_asset). Used by execution_logic, meta_controller."""
        return await self.get_spendable_quote(
            self.quote_asset,
            reserve_ratio=self.config.quote_reserve_ratio,
            min_reserve=self.config.quote_min_reserve,
        )

    async def get_spendable_usdt(self) -> float:
        """Alias → get_spendable_balance(quote_asset). Used by scaling, liquidation, meta_controller."""
        return await self.get_spendable_balance(self.quote_asset)

    async def get_non_quote_positions(self) -> Dict[str, Dict[str, Any]]:
        """
        Returns a shallow copy of positions that are NOT the quote asset (e.g., non-USDT).
        Used by LiquidationAgent to discover what can be liquidated to free quote.
        """
        # Positions are tracked by trading symbol (e.g., BTCUSDT). Filter any positions with qty>0
        # and ignore synthetic or quote-only tickers.
        out: Dict[str, Dict[str, Any]] = {}
        for sym, pos in self.positions.items():
            try:
                qty = float(pos.get("quantity", 0.0))
            except Exception:
                qty = 0.0
            if qty > 0.0:
                out[sym] = dict(pos)
        return out

    # -------- Dust register helpers --------
    def _infer_dust_origin(self, symbol: str) -> str:
        sym = self._norm_sym(symbol)
        if sym in self.accepted_symbols or sym in self.symbols:
            return "strategy_portfolio"
        return "external_untracked"

    def _update_dust_origin_metrics(self) -> None:
        counts: Dict[str, int] = defaultdict(int)
        for data in self.dust_registry.values():
            origin = str(data.get("origin") or "unknown")
            counts[origin] += 1
        self.metrics["dust_origin_breakdown"] = dict(counts)

    def record_dust(
        self,
        symbol: str,
        qty: float,
        *,
        origin: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register dust metadata so we can distinguish strategy vs. trash origins.

        NOTE: This is sync because it's called from many sync paths (mark_as_dust,
        hydrate_positions_from_balances, etc). Safe in single-threaded asyncio since
        it contains no await points — all dict writes execute atomically within one
        event loop tick.
        """
        try:
            sym = self._norm_sym(symbol)
            now = time.time()
            entry = dict(self.dust_registry.get(sym, {}))
            first_seen = float(entry.get("first_seen") or entry.get("timestamp") or now)

            entry.update({
                "qty": float(max(qty, 0.0)),
                "timestamp": entry.get("timestamp", first_seen),
                "first_seen": first_seen,
                "last_seen": now,
                "state": PositionState.DUST_LOCKED.value,
                "origin": origin or entry.get("origin") or self._infer_dust_origin(sym),
            })

            if context:
                existing_ctx = dict(entry.get("context") or {})
                existing_ctx.update(context)
                entry["context"] = existing_ctx

            # Write dust_registry first, then update position state.
            # Both are single dict assignments — atomic in CPython's GIL.
            self.dust_registry[sym] = entry

            pos = self.positions.get(sym)
            if pos is not None:
                pos["state"] = PositionState.DUST_LOCKED.value

            if sym not in self._dust_first_seen:
                self._dust_first_seen[sym] = first_seen

            self.metrics["dust_registry_size"] = len(self.dust_registry)
            self._update_dust_origin_metrics()
        except Exception:
            pass

    async def get_dust_registry_snapshot(self) -> Dict[str, Dict[str, Any]]:
        """Return a shallow copy of the current dust register."""
        return dict(self.dust_registry)

    async def get_dust_origin_breakdown(self) -> Dict[str, int]:
        """Expose the current dust-origin histogram for monitoring/telemetry."""
        return dict(self.metrics.get("dust_origin_breakdown", {}))

    async def prune_reservations(self) -> None:
        """
        CRITICAL: Remove all expired quote reservations to recover locked capital.
        This prevents deadlock when reservations from failed/stuck orders consume all spendable balance.
        """
        try:
            now = time.time()
            total_before = sum(
                sum(float(r.get("amount", 0.0)) for r in reservations)
                for reservations in self._quote_reservations.values()
            )
            
            # Filter all reservations, remove expired ones
            for asset in list(self._quote_reservations.keys()):
                reservations = self._quote_reservations[asset]
                valid = [r for r in reservations if r.get("expires_at", 0) > now]
                if valid:
                    self._quote_reservations[asset] = valid
                else:
                    self._quote_reservations.pop(asset, None)
            
            total_after = sum(
                sum(float(r.get("amount", 0.0)) for r in reservations)
                for reservations in self._quote_reservations.values()
            )
            
            recovered = total_before - total_after
            if recovered > 0:
                self.logger.warning(
                    f"[SharedState:Prune] Cleared stale reservations: recovered {recovered:.2f} USDT "
                    f"(was {total_before:.2f}, now {total_after:.2f})"
                )
                await self.emit_event("ReservationsPruned", {
                    "recovered": float(recovered),
                    "before": float(total_before),
                    "after": float(total_after),
                    "ts": now
                })
        except Exception as e:
            self.logger.warning(f"[SharedState:Prune] Error pruning reservations: {e}", exc_info=True)

        # Also prune stale per-agent authoritative budgets
        self.prune_authoritative_reservations()

    def prune_authoritative_reservations(self, max_age_sec: float = 300.0) -> int:
        """Remove per-agent budgets that have not been refreshed within max_age_sec.

        CapitalAllocator re-writes all agent budgets every cycle, so any entry
        older than ~5 minutes was set by a now-dead cycle and should be released.
        Returns the count of pruned entries.
        """
        if not hasattr(self, "_authoritative_reservation_ts"):
            return 0
        now = time.time()
        stale = [
            agent for agent, ts in self._authoritative_reservation_ts.items()
            if (now - ts) > max_age_sec
        ]
        for agent in stale:
            self._authoritative_reservations.pop(agent, None)
            self._authoritative_reservation_ts.pop(agent, None)
        if stale:
            self.logger.warning(
                "[SS:PruneAuthRes] Pruned %d stale authoritative reservations (age >%.0fs): %s",
                len(stale), max_age_sec, stale,
            )
        return len(stale)

    async def prune_dust_registry(self, ttl_days: float = 7.0) -> None:
        """Drop dust entries not seen for `ttl_days`."""
        try:
            horizon = time.time() - max(0.0, float(ttl_days)) * 86400.0
            drop = [s for s, d in self.dust_registry.items() if float(d.get("last_seen", 0.0)) < horizon]
            for s in drop:
                self.dust_registry.pop(s, None)
            self.metrics["dust_registry_size"] = len(self.dust_registry)
            self._update_dust_origin_metrics()
            if drop:
                await self.emit_event("DustRegistryPruned", {"dropped": drop, "remaining": len(self.dust_registry)})
        except Exception:
            pass

    async def get_sellable_inventory(
        self,
        *,
        min_quote_value: Optional[float] = None,
        include_dust: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """
        Build a list of positions that can be sold to free quote.
        Uses symbol filters (LOT_SIZE, MIN_NOTIONAL) and current price when available.
        Returns a list of dicts with keys:
        symbol, base_asset, quote_asset, qty, est_quote_value, price, filters, reason
        """
        if min_quote_value is None:
            min_quote_value = float(getattr(self.config, "dust_min_quote_usdt", 5.0) or 0.0)
        if include_dust is None:
            include_dust = bool(getattr(self.config, "dust_liquidation_enabled", True))
        results: List[Dict[str, Any]] = []
        # Snapshot to avoid long-held locks
        positions = dict(self.positions)
        prices = await self.get_all_prices()
        quote = self.quote_asset.upper()

        # Attempt to fetch/ensure filters from the exchange client if we are missing some
        try:
            if self._exchange_client and hasattr(self._exchange_client, "ensure_symbol_filters_ready"):
                await self._exchange_client.ensure_symbol_filters_ready()
        except Exception:
            pass

        for symbol, pos in positions.items():
            try:
                qty = float(pos.get("quantity", 0.0))
            except Exception:
                qty = 0.0
            if qty <= 0:
                continue

            sym = self._norm_sym(symbol)
            # Use suffix slice for base asset (avoid accidental replacements)
            if sym.endswith(quote):
                base_asset = sym[:-len(quote)]
            else:
                base_asset = sym
            quote_asset = quote

            # Fetch filters (raw → normalized fallback, cached)
            f = await self._fetch_and_cache_symbol_filters(sym)
            lot_step, _min_qty, _tick_size, min_notional = self._extract_symbol_filter_values(f)

            # Current price
            px = float(prices.get(sym) or pos.get("mark_price") or pos.get("entry_price") or 0.0)
            # Compute eff_qty before value and use it for est_quote_value
            eff_qty = self._round_step(qty, lot_step) if lot_step > 0 else qty
            if px <= 0:
                reason = "no_price"
                est_quote_value = 0.0
            else:
                est_quote_value = eff_qty * px
                reason = "ok"

            # ===== PHASE 3: Track dust eligibility for cleanup =====
            dust_cleanup_eligible = False

            # Decide eligibility
            eligible = True
            if eff_qty <= 0:
                eligible = False
                reason = "qty_below_step"
            origin_hint = (self.dust_registry.get(sym) or {}).get("origin")
            default_origin = origin_hint or ("strategy_portfolio" if sym in positions else "external_untracked")

            if px > 0 and min_notional > 0 and est_quote_value < min_notional:
                self.record_dust(
                    sym,
                    eff_qty,
                    origin=default_origin,
                    context={"source": "sellable_inventory", "reason": "below_min_notional"},
                )
                dust_cleanup_eligible = True  # ← PHASE 3: Mark for cleanup even if below threshold
                if not include_dust and not self.bypass_portfolio_flat_for_dust:
                    # Only skip if we're NOT doing dust cleanup
                    eligible = False
                    reason = "below_min_notional"
            if px > 0 and est_quote_value < float(min_quote_value or 0.0):
                self.record_dust(
                    sym,
                    eff_qty,
                    origin=default_origin,
                    context={"source": "sellable_inventory", "reason": "below_threshold"},
                )
                dust_cleanup_eligible = True  # ← PHASE 3: Mark for cleanup even if below threshold
                if not include_dust and not self.bypass_portfolio_flat_for_dust:
                    # Only skip if we're NOT doing dust cleanup
                    eligible = False
                    reason = "below_threshold"

            # ===== PHASE 3 FIX: Always include dust when in cleanup mode =====
            if dust_cleanup_eligible and self.bypass_portfolio_flat_for_dust:
                eligible = True  # Force eligible for dust cleanup
                reason = "dust_cleanup"

            if not eligible:
                continue

            results.append({
                "symbol": sym,
                "base_asset": base_asset,
                "quote_asset": quote_asset,
                "qty": eff_qty,
                "est_quote_value": est_quote_value,
                "price": px,
                "filters": f,
                "reason": reason,
            })

        # Sort largest first to help the LiquidationAgent free capital quickly
        results.sort(key=lambda r: r.get("est_quote_value", 0.0), reverse=True)
        return results

    # ===== PHASE 3: Dust Cleanup Methods =====

    async def get_dust_cleanup_candidates(
        self,
        min_age_sec: int = 300,  # At least 5 minutes old
        max_attempts: int = 3,
        attempt_cooldown_sec: int = 300
    ) -> List[Dict[str, Any]]:
        """
        ===== PHASE 3: Get dust positions eligible for cleanup SELL =====
        
        Returns positions that:
        1. Are in dust_registry (below minNotional)
        2. Haven't exceeded max cleanup attempts
        3. Are old enough (min_age_sec)
        4. Are past cooldown from last attempt
        5. Have qty > 0 (not already closed)
        
        These can be sold REGARDLESS of portfolio state.
        """
        now = time.time()
        candidates = []
        
        # Get all positions that are currently in dust register
        for symbol in list(self.dust_registry.keys()):
            try:
                pos = await self.get_position(symbol)
                if not pos:
                    continue
                    
                qty = float(pos.get("quantity", 0.0))
                if qty <= 0:
                    # Already liquidated, remove from dust register
                    self.dust_registry.pop(symbol, None)
                    self.metrics["dust_registry_size"] = len(self.dust_registry)
                    self._update_dust_origin_metrics()
                    continue
                
                # Check attempt count
                attempt_count = self.dust_cleanup_attempts.get(symbol, 0)
                if attempt_count >= max_attempts:
                    self.logger.info(
                        f"[Phase3] {symbol} dust cleanup: max attempts ({max_attempts}) reached"
                    )
                    continue
                
                # Check age (must be old enough to be dust)
                first_seen = self._dust_first_seen.get(symbol, now)
                age_sec = now - first_seen
                if age_sec < min_age_sec:
                    continue
                
                # Check cooldown from last attempt
                last_try = self.dust_cleanup_last_try.get(symbol, 0)
                time_since_last_try = now - last_try
                if time_since_last_try < attempt_cooldown_sec and attempt_count > 0:
                    continue
                
                # This dust position is eligible for cleanup
                px = float(pos.get("mark_price") or pos.get("entry_price") or 0.0)
                est_value = qty * px if px > 0 else 0.0
                
                candidates.append({
                    "symbol": symbol,
                    "qty": qty,
                    "price": px,
                    "est_quote_value": est_value,
                    "age_sec": age_sec,
                    "attempt_count": attempt_count,
                    "reason": "dust_cleanup_eligible"
                })
            except Exception as e:
                self.logger.warning(f"Error checking dust cleanup for {symbol}: {e}")
                continue
        
        # Sort by age (oldest first) and attempt count (fewer attempts first)
        candidates.sort(key=lambda x: (-x["age_sec"], x["attempt_count"]))
        
        return candidates


    async def mark_dust_cleanup_attempted(self, symbol: str) -> None:
        """
        ===== PHASE 3: Record that we attempted to cleanup a dust position =====
        """
        sym = self._norm_sym(symbol)
        self.dust_cleanup_attempts[sym] = self.dust_cleanup_attempts.get(sym, 0) + 1
        self.dust_cleanup_last_try[sym] = time.time()
        
        self.logger.info(
            f"[Phase3] Dust cleanup attempt #{self.dust_cleanup_attempts[sym]} for {sym}"
        )


    async def clear_dust_cleanup_state(self, symbol: str) -> None:
        """
        ===== PHASE 3: Clear dust cleanup tracking after successful SELL =====
        """
        sym = self._norm_sym(symbol)
        self.dust_cleanup_attempts.pop(sym, None)
        self.dust_cleanup_last_try.pop(sym, None)
        
        self.logger.info(f"[Phase3] Dust cleanup state cleared for {sym}")


    async def enable_dust_cleanup_mode(self) -> None:
        """
        ===== PHASE 3: Enable bypass of portfolio_flat checks for dust cleanup SELL =====
        
        When enabled:
        • get_sellable_inventory() marks dust positions as eligible
        • get_dust_cleanup_candidates() returns all eligible dust
        • SELL signals for dust will not be blocked by portfolio state
        • Only dust positions are affected (normal positions unaffected)
        """
        self.bypass_portfolio_flat_for_dust = True
        self.logger.info("[Phase3] Dust cleanup mode ENABLED - dust SELL bypass active")


    async def disable_dust_cleanup_mode(self) -> None:
        """
        ===== PHASE 3: Disable dust cleanup bypass =====
        
        Normal portfolio state checks resume for all positions.
        """
        self.bypass_portfolio_flat_for_dust = False
        self.logger.info("[Phase3] Dust cleanup mode DISABLED - normal checks resumed")


    async def is_dust_cleanup_mode_enabled(self) -> bool:
        """
        ===== PHASE 3: Query whether dust cleanup mode is active =====
        """
        return self.bypass_portfolio_flat_for_dust

    async def hydrate_positions_from_balances(self) -> None:
        """
        Mirror non-quote wallet balances into spot positions using the configured quote asset.
        If a symbol like BASE+QUOTE (e.g., BTCUSDT) exists in `self.symbols` or `self.accepted_symbols`,
        create/update a position entry with quantity equal to wallet free amount (do not touch avg_price).
        """
        quote = self.quote_asset.upper()
        # Snapshot balances to avoid holding the balances lock while touching positions
        snapshot = dict(self.balances)
        changed: list[str] = []
        for asset, data in snapshot.items():
            a = asset.upper()
            if a == quote:
                continue
            free_qty = float(data.get("free", 0.0))
            if free_qty <= 0:
                # If we previously had a mirrored position, clear it
                sym = f"{a}{quote}"
                if sym in self.positions and self.positions.get(sym, {}).get("_mirrored", False):
                    await self.update_position(sym, {
                        "quantity": 0.0,
                        "avg_price": 0.0,
                        "_mirrored": True,
                        "status": "CLOSED"  # CRITICAL: Mark as CLOSED so open_positions_count() doesn't count it
                    })
                    self.open_trades.pop(sym, None)
                    changed.append(sym)
                continue
            sym = f"{a}{quote}"
            if sym not in self.symbols and sym not in self.accepted_symbols:
                # Skip unknown trading pairs
                continue
            prev = self.positions.get(sym, {})
            if float(prev.get("quantity", 0.0)) != free_qty or not prev.get("_mirrored"):
                pos = dict(prev)
                price = float(self.latest_prices.get(sym, 0.0) or 0.0)
                if price <= 0 and self._exchange_client:
                    try:
                        getter = getattr(self._exchange_client, "get_current_price", None) or getattr(
                            self._exchange_client, "get_symbol_price", None
                        )
                        if callable(getter):
                            price = float(await getter(sym) or 0.0)
                    except Exception:
                        price = 0.0
                position_value = float(free_qty * price) if price > 0 else float(prev.get("value_usdt", 0.0) or 0.0)
                significant_floor = float(await self.get_significant_position_floor(sym) or 0.0)
                is_significant = bool(position_value >= significant_floor and position_value > 0.0)
                
                # ===== BEST PRACTICE: ENTRY PRICE IMMUTABILITY =====
                # Entry price is the original trade price and MUST NEVER CHANGE.
                # Only avg_price can change during scaling.
                # Strategy: Use existing entry_price if available, fallback to avg_price ONLY if missing.
                reconstructed_entry_price = pos.get("entry_price")
                
                if reconstructed_entry_price is None:
                    reconstructed_entry_price = pos.get("avg_price")
                
                # LAST RESORT ONLY: Use current price if no historical data
                if reconstructed_entry_price is None:
                    reconstructed_entry_price = price
                
                reconstructed_entry_price = float(reconstructed_entry_price or 0.0)
                
                # ===== avg_price: Can change during scaling =====
                # Prefer existing avg_price, fallback to reconstructed_entry_price
                avg_price = pos.get("avg_price")
                
                if avg_price is None:
                    avg_price = reconstructed_entry_price
                
                avg_price = float(avg_price or 0.0)
                
                pos.update({
                    "quantity": free_qty,
                    "avg_price": avg_price,
                    "entry_price": reconstructed_entry_price,  # IMMUTABLE: Never changes after creation
                    "mark_price": float(price or pos.get("mark_price", 0.0) or 0.0),
                    "value_usdt": float(position_value),
                    "significant_floor_usdt": float(significant_floor),
                    "is_significant": bool(is_significant),
                    "is_dust": not bool(is_significant),
                    "_is_dust": not bool(is_significant),
                    "open_position": bool(is_significant),
                    "state": PositionState.ACTIVE.value if is_significant else PositionState.DUST_LOCKED.value,
                    "_mirrored": True,
                    "status": "SIGNIFICANT" if is_significant else "DUST",
                })
                
                await self.update_position(sym, pos)
                if not is_significant:
                    self.record_dust(
                        sym,
                        free_qty,
                        origin="wallet_balance_sync",
                        context={
                            "source": "hydrate_positions_from_balances",
                            "value_usdt": float(position_value),
                            "significant_floor_usdt": float(significant_floor),
                        },
                    )
                    self.open_trades.pop(sym, None)
                else:
                    self.dust_registry.pop(sym, None)
                changed.append(sym)
        if changed:
            await self.emit_event("PositionsMirroredFromBalances", {"symbols": changed, "count": len(changed)})

    async def hydrate_balances_from_exchange(self) -> bool:
        """Pull balances from the attached exchange client (if any) and update local state.
        Returns True on success, False otherwise."""
        try:
            if not self._exchange_client:
                return False
            # Prefer a generic shim if the client exposes it; otherwise fall back to spot balances.
            if hasattr(self._exchange_client, "get_account_balances"):
                bal = await self._exchange_client.get_account_balances()
            elif hasattr(self._exchange_client, "get_spot_balances"):
                bal = await self._exchange_client.get_spot_balances()
            elif hasattr(self._exchange_client, "get_balances"):
                bal = await self._exchange_client.get_balances()
            else:
                return False
            if isinstance(bal, dict):
                await self.update_balances(bal)
                return True
            return False
        except Exception as e:
            self.logger.warning(f"hydrate_balances_from_exchange failed: {e}")
            return False

    async def _wallet_sync_loop(self) -> None:
        """Background task: periodically refresh balances from the exchange client."""
        interval = int(getattr(self.config, "wallet_sync_interval", 120) or 120)
        while True:
            try:
                ok = await self.hydrate_balances_from_exchange()
                if not ok:
                    # Avoid hot loop if no client or failure
                    await asyncio.sleep(max(2, interval))
                else:
                    await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Wallet sync loop error: {e}")
                await asyncio.sleep(interval)

    async def get_portfolio_snapshot(self) -> Dict[str, Any]:
        """
        🔥 CRITICAL: Get live portfolio snapshot directly from Binance
        DO NOT use stale cached prices. Sync with actual Binance balances.
        
        FIX: Reconcile positions with open_trades to prevent double-counting
        
        ⚠️ SHADOW MODE BRANCH: In shadow mode, return virtual portfolio (not Binance)
        """
        # SHADOW MODE: use virtual portfolio only
        if self.trading_mode == "shadow":
            return await self._get_shadow_portfolio_snapshot()
        
        # LIVE MODE: Continue with Binance-backed snapshot
        # 1. REFRESH balances from Binance (authoritative source)
        try:
            if hasattr(self._exchange_client, "get_account_balances"):
                live_balances = await self._exchange_client.get_account_balances()
                if live_balances:
                    self.balances = live_balances
            elif hasattr(self._exchange_client, "get_balances"):
                live_balances = await self._exchange_client.get_balances()
                if live_balances:
                    self.balances = live_balances
        except Exception as e:
            logging.getLogger("SharedState").warning(f"Failed to refresh balances: {e}")
        
        # 2. RECONCILE: Sync open_trades with actual positions from balances
        # This prevents double-count where open_trades qty ≠ position qty
        # NOTE: Use timestamp-based tolerance to avoid false reconciliation due to API lag
        try:
            if isinstance(self.open_trades, dict):
                current_time = time.time()
                for sym in list(self.open_trades.keys()):
                    # Get position quantity from balances
                    asset = sym.replace("USDT", "").upper() if "USDT" in sym else ""
                    if not asset:
                        continue
                    
                    bal_qty = 0.0
                    for asset_key, bal in (self.balances or {}).items():
                        if asset_key.upper() == asset:
                            bal_qty = float(bal.get("free", 0.0)) + float(bal.get("locked", 0.0))
                            break
                    
                    if bal_qty > 0:
                        # Update open_trade quantity to match actual balance
                        ot = self.open_trades.get(sym, {})
                        if isinstance(ot, dict):
                            old_qty = float(ot.get("quantity", 0.0))
                            qty_mismatch = abs(old_qty - bal_qty)
                            
                            if qty_mismatch > 0.00000001:  # Threshold for floating point
                                # CRITICAL: Check age of the fill
                                # If very recent (< 5 sec), might be Binance API lag
                                # Trust our record over stale Binance API
                                pos_data = self.positions.get(sym, {})
                                last_fill_ts = float(pos_data.get("last_fill_ts", 0.0) or 0.0)
                                fill_age = current_time - last_fill_ts if last_fill_ts > 0 else 999.0
                                
                                if fill_age < 5.0:
                                    # Recent fill: trust our record, skip reconciliation
                                    logging.getLogger("SharedState").debug(
                                        f"[RECONCILE:SKIP] {sym}: recent fill (age={fill_age:.1f}s), "
                                        f"trusting local qty={old_qty:.8f} over balance qty={bal_qty:.8f}"
                                    )
                                    continue
                                else:
                                    # Old position: trust Binance, reconcile
                                    logging.getLogger("SharedState").warning(
                                        f"[RECONCILE] {sym}: open_trade qty={old_qty:.8f} → balance qty={bal_qty:.8f} "
                                        f"(fill_age={fill_age:.1f}s)"
                                    )
                                    ot["quantity"] = bal_qty
                                    self.open_trades[sym] = ot
                    else:
                        # Position closed - remove from open_trades
                        self.open_trades.pop(sym, None)
        except Exception as e:
            logging.getLogger("SharedState").warning(f"Failed to reconcile open_trades: {e}")
        
        # 3. REFRESH positions by querying Binance (authoritative)
        try:
            # Clear positions and rebuild from actual Binance balances
            self.positions = {}
            for asset, bal in self.balances.items():
                if asset.upper() == self.quote_asset.upper():
                    continue  # Skip USDT, add it later
                qty = float(bal.get("free", 0.0)) + float(bal.get("locked", 0.0))
                if qty > 0:
                    sym = f"{asset}USDT"
                    # Get live price for this symbol
                    try:
                        if hasattr(self._exchange_client, "get_current_price"):
                            price = await self._exchange_client.get_current_price(sym)
                        else:
                            price = self.latest_prices.get(sym, 0.0)
                        if price:
                            self.positions[sym] = {
                                "symbol": sym,
                                "quantity": qty,
                                "current_price": float(price),
                                "mark_price": float(price),
                                "entry_price": float(price),
                                "avg_price": float(price),  # Use current price (safer)
                            }
                    except Exception:
                        pass
        except Exception as e:
            logging.getLogger("SharedState").warning(f"Failed to rebuild positions: {e}")
        
        # 4. GET LIVE PRICES (fresh from exchange_client if possible)
        prices = await self.get_all_prices()
        if hasattr(self._exchange_client, "get_ticker") and self.positions:
            try:
                for sym in list(self.positions.keys()):
                    try:
                        tick = await self._exchange_client.get_ticker(sym)
                        if tick and tick.get("last"):
                            prices[sym] = float(tick["last"])
                    except Exception:
                        pass
            except Exception:
                pass
        
        # 5. CALCULATE NAV (Net Asset Value)
        nav = 0.0
        unreal = 0.0
        
        # Add USDT balance (quote asset)
        for asset, b in self.balances.items():
            if asset.upper() == self.quote_asset.upper():
                nav += float(b.get("free", 0.0)) + float(b.get("locked", 0.0))
        
        # Add crypto positions at LIVE prices
        for sym, pos in self.positions.items():
            qty = float(pos.get("quantity", 0.0))
            if qty <= 0: continue
            
            # CRITICAL: Use live price, fallback to last known price
            px = float(prices.get(sym) or pos.get("mark_price") or pos.get("current_price") or 0.0)
            if px <= 0:
                continue  # Skip if no price available
            
            nav += qty * px
            
            # Unrealized PnL: (current - entry) * qty
            # Use current price for entry if no avg_price available
            avg = float(pos.get("avg_price") or pos.get("entry_price") or px)
            if avg > 0 and px > 0:
                unreal += (px - avg) * qty
        
        self.metrics["nav"] = nav
        self.metrics["unrealized_pnl"] = unreal
        if not self.nav_ready_event.is_set():
            self.nav_ready_event.set()
            self.metrics["nav_ready"] = True
            await self.emit_event("NavReady", {"ts": time.time(), "source": "portfolio_snapshot"})
        
        return {
            "ts": time.time(), "nav": nav,
            "realized_pnl": float(self.metrics.get("realized_pnl", 0.0)),
            "unrealized_pnl": unreal,
            "balances": dict(self.balances),
            "positions": dict(self.positions),
            "prices": prices,
        }

    async def _get_shadow_portfolio_snapshot(self) -> Dict[str, Any]:
        """
        Shadow Mode: Return virtual portfolio snapshot (NOT Binance)
        
        In shadow mode:
        - Authority = virtual_balances
        - Authority = virtual_positions
        - Authority = virtual_nav
        
        Never pull from Binance in shadow mode.
        """
        # Calculate NAV from virtual portfolio
        nav = 0.0
        unreal = 0.0
        
        # Add virtual USDT balance (quote asset)
        for asset, bal in self.virtual_balances.items():
            if asset.upper() == self.quote_asset.upper():
                nav += float(bal.get("free", 0.0)) + float(bal.get("locked", 0.0))
        
        # Add virtual crypto positions at LIVE prices
        for sym, pos in self.virtual_positions.items():
            qty = float(pos.get("quantity", 0.0))
            if qty <= 0:
                continue
            
            # Use live price, fallback to position's mark_price
            px = float(self.latest_prices.get(sym) or pos.get("mark_price") or pos.get("current_price") or 0.0)
            if px <= 0:
                continue  # Skip if no price available
            
            nav += qty * px
            
            # Unrealized PnL: (current - entry) * qty
            avg = float(pos.get("avg_price") or pos.get("entry_price") or px)
            if avg > 0 and px > 0:
                unreal += (px - avg) * qty
        
        # Update metrics
        self.metrics["nav"] = nav
        self.metrics["unrealized_pnl"] = unreal
        self.virtual_nav = nav  # Sync virtual_nav with calculated NAV
        
        if not self.nav_ready_event.is_set():
            self.nav_ready_event.set()
            self.metrics["nav_ready"] = True
            await self.emit_event("NavReady", {"ts": time.time(), "source": "shadow_portfolio_snapshot"})
        
        # Return virtual portfolio (NOT live balances/positions)
        return {
            "ts": time.time(),
            "nav": nav,
            "realized_pnl": float(self.metrics.get("realized_pnl", 0.0)),
            "unrealized_pnl": unreal,
            "balances": dict(self.virtual_balances),  # SHADOW ONLY
            "positions": dict(self.virtual_positions),  # SHADOW ONLY
            "prices": self.latest_prices,
            "mode": "shadow",  # Flag this as shadow mode
        }

    # ---- Rejection Tracking (Deadlock Prevention) ----
    async def record_rejection(self, symbol: str, side: str, reason: str, source: str = "Unknown"):
        """P9: Record a trade rejection/block for deadlock detection."""
        sym = str(symbol).upper()
        sd = str(side).upper()
        rea = str(reason).upper()
        src = str(source)
        key = (sym, sd, rea)
        self.rejection_counters[key] += 1
        self.rejection_timestamps[key] = time.time()
        
        # Track in rejection history (bounded deque)
        if not hasattr(self, "rejection_history"):
            self.rejection_history = deque(maxlen=100)
        self.rejection_history.append({
            "symbol": sym, "side": sd, "reason": rea, "source": src,
            "count": self.rejection_counters[key], "ts": time.time()
        })
        
        rej_count = self.rejection_counters[key]
        # Structured [EXEC_REJECT] log format matching LOOP_SUMMARY pattern
        self.logger.info(f"[EXEC_REJECT] symbol={sym} side={sd} reason={rea} count={rej_count} action=RETRY")
        
        if hasattr(self, "emit_event"):
            await self.emit_event("TradeRejection", {
                "symbol": sym, "side": sd, "reason": rea, "source": src,
                "count": rej_count, "ts": time.time()
            })

    # ---- TIER 2: Policy Conflict Tracking ----
    def record_policy_conflict(self, conflict_type: str) -> None:
        """Track policy conflict metrics for monitoring and alerting."""
        try:
            policy_conflicts = self.metrics.get("policy_conflicts", {})
            if conflict_type in policy_conflicts:
                policy_conflicts[conflict_type] = policy_conflicts.get(conflict_type, 0) + 1
        except Exception:
            pass

    def register_signal_outcome(self, record: Dict[str, Any]) -> None:
        """Register a signal outcome for tracking price movement after emission.
        
        Args:
            record: Dict with keys: symbol, timestamp, price_at_signal, confidence, agent
        """
        try:
            self._signal_outcomes.append(record)
        except Exception:
            pass

    def get_policy_conflict_summary(self) -> Dict[str, int]:
        """Return current policy conflict metrics for observability."""
        try:
            return dict(self.metrics.get("policy_conflicts", {}))
        except Exception:
            return {}

    def get_rejection_count(self, symbol: str, side: str, reason: Optional[str] = None) -> int:
        """P9: Get count of rejections for a symbol/side combo with TTL decay (5 min).
        
        CRITICAL FIX for G022 (Rejection Infinite Loop):
        - Rejection counters now have 5-minute TTL
        - After 5 min without new rejection, counter resets to 0
        - Prevents deadlock where rejection_count >= 3 blocks forever
        """
        s = str(symbol).upper()
        sd = str(side).upper()
        rej_ttl_sec = 300.0  # 5 minutes
        now = time.time()
        
        if reason:
            key = (s, sd, str(reason).upper())
        else:
            key = (s, sd, "")  # Will aggregate across reasons below
        
        # Apply TTL decay
        if reason:
            ts = self.rejection_timestamps.get(key, now)
            if now - ts > rej_ttl_sec:
                # Rejection counter expired; reset
                self.rejection_counters[key] = 0
                self.rejection_timestamps[key] = now
                return 0
            return self.rejection_counters.get(key, 0)
        
        # Total for symbol/side across all reasons (aggregate with TTL)
        total = 0
        expired_keys = []
        for k, v in self.rejection_counters.items():
            if k[0] == s and k[1] == sd:
                ts = self.rejection_timestamps.get(k, now)
                if now - ts > rej_ttl_sec:
                    expired_keys.append(k)
                else:
                    total += v
        
        # Clean up expired entries
        for k in expired_keys:
            self.rejection_counters[k] = 0
            self.logger.debug(f"[SharedState] Rejection counter TTL expired for {k}, reset to 0")
        
        return total

    def get_total_rejections(self) -> int:
        """P9: Get total rejection count across all symbols."""
        return sum(self.rejection_counters.values())

    def get_max_rejection_count(self) -> Tuple[Optional[Tuple[str, str, str]], int]:
        """P9: Get the key with highest rejection count."""
        if not self.rejection_counters:
            return None, 0
        max_key = max(self.rejection_counters.keys(), key=lambda k: self.rejection_counters[k])
        return max_key, self.rejection_counters[max_key]

    def is_symbol_blocked(self, symbol: str, side: str, threshold: int = 3) -> bool:
        """P9: Check if a symbol/side combo is blocked (exceeds threshold)."""
        count = self.get_rejection_count(symbol, side)
        return count >= threshold

    async def clear_rejections(self, symbol: str, side: str):
        """P9: Clear rejection counts (e.g. after successful trade)."""
        s = str(symbol).upper()
        sd = str(side).upper()
        keys_to_del = [k for k in self.rejection_counters.keys() if k[0] == s and k[1] == sd]
        for k in keys_to_del:
            self.rejection_counters.pop(k, None)
            self.rejection_timestamps.pop(k, None)

    async def is_economically_ready(self, min_executable_symbols: int = 1, threshold: int = 3) -> bool:
        """
        P9 ECONOMIC READINESS: Check if at least N symbols are executable.
        Returns False if all top symbols are blocked.
        """
        accepted = list(self.accepted_symbols.keys())
        if not accepted:
            return False
        
        executable_count = 0
        for sym in accepted:
            if not self.is_symbol_blocked(sym, "BUY", threshold):
                executable_count += 1
                if executable_count >= min_executable_symbols:
                    return True
        return False


    async def record_fill(self, symbol: str, side: str, qty: float, price: float, fee_quote: float = 0.0, fee_base: float = 0.0, tier: Optional[str] = None) -> Dict[str, Any]:
        side_u = (side or "").upper()
        qty = float(qty)
        price = float(price)
        if qty <= 0 or price <= 0: return {"realized_pnl_delta": 0.0}
        pos = dict(self.positions.get(symbol, {}))
        cur_qty = float(pos.get("quantity", 0.0))
        avg = float(pos.get("avg_price", self._avg_price_cache.get(symbol, 0.0) or 0.0))
        realized = 0.0
        fee_quote = float(fee_quote or 0.0)
        fee_base = float(fee_base or 0.0)
        
        if side_u == "BUY":
            # Spot BUY fees can be charged in base or quote. Track both as base-equivalent
            # and keep position qty net of base-fee deductions.
            fee_quote_equiv_base = (fee_quote / price) if fee_quote > 0 and price > 0 else 0.0
            buy_fee_base = float(pos.get("buy_fee_base", 0.0) or 0.0) + fee_base + fee_quote_equiv_base
            net_qty = max(0.0, qty - fee_base)
            if net_qty <= 0:
                return {"realized_pnl_delta": 0.0}
            new_qty = cur_qty + net_qty
            new_avg = ((cur_qty * avg) + (net_qty * price)) / max(new_qty, 1e-12)
            pos.update({
                "quantity": new_qty,
                "avg_price": new_avg,
                "last_fill_ts": time.time(),
                "buy_fee_base": buy_fee_base,
                "value_usdt": new_qty * price,  # Update position value
            })
            self._avg_price_cache[symbol] = new_avg
            
            # Frequency Engineering: Track tier count
            if tier == "A":
                self.metrics["trades_tier_a"] += 1
            elif tier == "B":
                self.metrics["trades_tier_b"] += 1
                
        elif side_u == "SELL":
            close_qty = min(qty, cur_qty)
            sell_fee_quote = fee_quote
            if qty > 0 and close_qty > 0 and close_qty < qty:
                sell_fee_quote *= (close_qty / max(qty, 1e-12))
            buy_fee_quote = 0.0
            if close_qty > 0 and avg > 0:
                buy_fee_base_total = float(pos.get("buy_fee_base", 0.0) or 0.0)
                if cur_qty > 0 and buy_fee_base_total > 0:
                    allocated_buy_fee_base = buy_fee_base_total * (close_qty / cur_qty)
                    buy_fee_quote = allocated_buy_fee_base * avg
                realized = (price - avg) * close_qty - float(sell_fee_quote or 0.0) - float(buy_fee_quote or 0.0)
            new_qty = max(0.0, cur_qty - qty)
            if new_qty == 0:
                pos.pop("buy_fee_base", None)
            else:
                buy_fee_base = float(pos.get("buy_fee_base", 0.0) or 0.0)
                if cur_qty > 0 and buy_fee_base > 0:
                    pos["buy_fee_base"] = buy_fee_base * (new_qty / cur_qty)
            pos.update({"quantity": new_qty, "avg_price": avg if new_qty > 0 else 0.0, "last_fill_ts": time.time()})
            if new_qty > 0:
                pos["value_usdt"] = new_qty * price  # Update position value for remaining quantity
            
            # Frequency Engineering: Track holding time
            ot = self.open_trades.get(symbol)
            if ot and "opened_at" in ot:
                duration = time.time() - ot["opened_at"]
                self.metrics["total_holding_time_sec"] += duration
                self.metrics["completed_trades_count"] += 1
        else:
            return {"realized_pnl_delta": 0.0}

        current_qty = float(pos.get("quantity", 0.0) or 0.0)
        significant_floor = float(await self.get_significant_position_floor(symbol) or 0.0)
        position_value = float(current_qty * price) if current_qty > 0 and price > 0 else 0.0
        if position_value <= 0:
            position_value = float(self._estimate_position_value_usdt(symbol, pos, price_hint=price) or 0.0)
        is_significant = bool(position_value >= significant_floor and position_value > 0.0)
        pos["value_usdt"] = float(position_value)
        pos["significant_floor_usdt"] = float(significant_floor)
        pos["is_significant"] = bool(is_significant)
        pos["is_dust"] = not bool(is_significant)
        pos["_is_dust"] = not bool(is_significant)
        pos["open_position"] = bool(is_significant)
        pos["capital_occupied"] = float(position_value) if is_significant else 0.0
        if current_qty > 0:
            pos["state"] = PositionState.ACTIVE.value if is_significant else PositionState.DUST_LOCKED.value
            pos["status"] = "SIGNIFICANT" if is_significant else "DUST"
        else:
            pos["state"] = PositionState.DUST_LOCKED.value
            pos["status"] = "CLOSED"

        await self.update_position(symbol, pos)

        # Hard invariant: only significant positions are represented in open_trades.
        if side_u == "BUY" and current_qty > 0 and is_significant:
            now_ts = time.time()
            ot = self.open_trades.get(symbol) if isinstance(self.open_trades, dict) else None
            if not isinstance(ot, dict):
                ot = {
                    "symbol": symbol,
                    "position": "long",
                    "entry_price": price,
                    "quantity": current_qty,
                    "opened_at": now_ts,
                    "created_at": now_ts,
                    "tier": tier,
                }
            else:
                ot.setdefault("opened_at", now_ts)
                ot.setdefault("created_at", now_ts)
                ot["quantity"] = float(current_qty)
                ot.setdefault("entry_price", price)
                if tier is not None:
                    ot["tier"] = tier
            self.open_trades[symbol] = ot
        else:
            self.open_trades.pop(symbol, None)

        if current_qty > 0 and not is_significant:
            self.record_dust(
                symbol,
                current_qty,
                origin="execution_fill",
                context={
                    "source": "record_fill",
                    "value_usdt": float(position_value),
                    "significant_floor_usdt": float(significant_floor),
                },
            )
        elif current_qty > 0 and is_significant:
            self.dust_registry.pop(self._norm_sym(symbol), None)
            
        now_ts = time.time()
        async with self._lock_context("metrics"):
            self.metrics["realized_pnl"] = float(self.metrics.get("realized_pnl", 0.0)) + realized
            total_realized = self.metrics["realized_pnl"]
        self._realized_pnl.append((now_ts, realized))
        self.trade_history.append({
            "ts": now_ts, "symbol": symbol, "side": side_u, "qty": qty, "price": price, "fee_quote": fee_quote,
            "fee_base": fee_base, "realized_delta": realized, "tier": tier
        })
        self.trade_count += 1
        await self.emit_event("RealizedPnlUpdated", {"realized_pnl": total_realized, "pnl_delta": realized, "symbol": symbol})
        return {"realized_pnl_delta": realized, "realized_pnl_total": total_realized}

    async def record_trade(self, symbol: str, side: str, qty: float, price: float, fee_quote: float = 0.0, fee_base: float = 0.0, tier: Optional[str] = None) -> Dict[str, Any]:
        """Compatibility alias for ExecutionManager post-fill tracking."""
        return await self.record_fill(symbol, side, qty, price, fee_quote=fee_quote, fee_base=fee_base, tier=tier)

    async def increment_realized_pnl(self, delta: float) -> None:
        """Atomically increment metrics['realized_pnl'] under the metrics lock.

        Used by ExecutionManager as the safe fallback when record_fill/record_trade
        is unavailable. Prevents the unprotected direct dict write race condition.
        """
        async with self._lock_context("metrics"):
            current = float(self.metrics.get("realized_pnl", 0.0) or 0.0)
            self.metrics["realized_pnl"] = current + delta
        try:
            self._realized_pnl.append((time.time(), delta))
        except AttributeError:
            self.logger.error(
                "[SharedState] _realized_pnl deque missing — recreating. Prior PnL history lost."
            )
            self._realized_pnl = deque(maxlen=4096)
            self._realized_pnl.append((time.time(), delta))

    def increment_idle_ticks(self) -> None:
        """Frequency Engineering: Track periods of no trading activity."""
        self.metrics["idle_ticks_count"] += 1

    async def update_utilization_metric(self) -> float:
        """Frequency Engineering: Update capital utilization percentage."""
        try:
            nav = await _safe_await(self.get_nav_quote())
            if not nav or nav <= 0:
                self.metrics["capital_utilization_pct"] = 0.0
                return 0.0
            
            total_pos_value = 0.0
            for symbol, pos in self.positions.items():
                qty = float(pos.get("quantity", 0.0))
                price = self.latest_prices.get(symbol, 0.0)
                if qty > 0 and price > 0:
                    total_pos_value += qty * price
            
            utilization = (total_pos_value / nav) * 100.0
            self.metrics["capital_utilization_pct"] = round(utilization, 2)
            return self.metrics["capital_utilization_pct"]
        except Exception:
            return 0.0

    @track_performance
    async def update_position(self, symbol: str, position_data: Dict[str, Any]) -> None:
        if not isinstance(position_data, dict):
            raise SharedStateError("position_data must be a dictionary", ErrorCode.CONFIGURATION_ERROR)
        sym = self._norm_sym(symbol)
        async with self._lock_context("positions"):
            # P9: Enforce state defaults
            if "state" not in position_data:
                # Inherit DUST_LOCKED if already in registry
                if sym in self.dust_registry:
                    position_data["state"] = PositionState.DUST_LOCKED.value
                else:
                    position_data["state"] = PositionState.ACTIVE.value
            
            # ===== Ensure status/state fields are consistent =====
            # Use classify_position_snapshot to align with the canonical
            # SIGNIFICANT/DUST vocabulary used everywhere else.
            if "status" not in position_data:
                qty = float(position_data.get("quantity", 0.0) or 0.0)
                if qty > 0:
                    is_sig, val, floor = self.classify_position_snapshot(sym, position_data)
                    position_data["status"] = "SIGNIFICANT" if is_sig else "DUST"
                    position_data["state"] = PositionState.ACTIVE.value if is_sig else PositionState.DUST_LOCKED.value
                    position_data["is_significant"] = is_sig
                    position_data["is_dust"] = not is_sig
                    position_data["_is_dust"] = not is_sig
                    position_data["open_position"] = is_sig
                    position_data.setdefault("value_usdt", val)
                    position_data.setdefault("significant_floor_usdt", floor)
                else:
                    position_data["status"] = "CLOSED"
                    position_data["state"] = PositionState.DUST_LOCKED.value
            
            # ===== POSITION INVARIANT ENFORCEMENT =====
            # CRITICAL ARCHITECTURE: Enforce the global invariant:
            # quantity > 0 → entry_price > 0
            # This protects ALL downstream modules (ExecutionManager, RiskManager, RotationExitAuthority,
            # ProfitGate, ScalingEngine, etc.) from deadlock due to missing entry_price.
            qty = float(position_data.get("quantity", 0.0) or 0.0)
            if qty > 0:
                entry = position_data.get("entry_price")
                avg = position_data.get("avg_price")
                mark = position_data.get("mark_price")
                
                if not entry or entry <= 0:
                    # Reconstruct entry_price from available sources
                    position_data["entry_price"] = float(avg or mark or 0.0)
                    
                    # Diagnostic warning so bugs never hide silently
                    self.logger.warning(
                        "[PositionInvariant] entry_price missing for %s — reconstructed from avg_price/mark_price",
                        sym
                    )
                
                # ===== ULTIMATE GUARD: Fail loudly if entry_price is still invalid =====
                # This is the strongest possible check - prevents corrupt state from propagating.
                final_entry = position_data.get("entry_price")
                if not final_entry or final_entry <= 0:
                    raise ValueError(
                        f"[CRITICAL INVARIANT] Cannot create/update position {sym} with qty={qty} "
                        f"but entry_price={final_entry}. Entry price MUST be > 0 for open positions."
                    )
            
            # ARCHITECTURE FIX: In shadow mode, update virtual_positions instead of positions
            if self.trading_mode == "shadow":
                self.virtual_positions[sym] = dict(position_data)
                self.positions[sym] = dict(position_data)  # ← ADD THIS
            else:
                self.positions[sym] = dict(position_data)

    async def force_close_all_open_lots(self, symbol: str, reason: str = "") -> None:
        """Force-close any open lots for a symbol by clearing position and open_trades."""
        sym = self._norm_sym(symbol)
        try:
            pos = dict(self.positions.get(sym, {}) or {})
            pos["quantity"] = 0.0
            pos["status"] = "CLOSED"
            if reason:
                pos["closed_reason"] = str(reason)
            pos["closed_at"] = time.time()
            await self.update_position(sym, pos)
        except Exception:
            pass
        try:
            if isinstance(self.open_trades, dict):
                self.open_trades.pop(sym, None)
        except Exception:
            pass
        try:
            await self.emit_event("ForceCloseAllOpenLots", {"symbol": sym, "reason": reason})
        except Exception:
            pass

    async def close_position(self, symbol: str, reason: str = "") -> None:
        """Canonical close helper for position finalization."""
        await self.force_close_all_open_lots(symbol, reason=reason)

    async def mark_position_closed(
        self,
        symbol: str,
        qty: float,
        price: float,
        reason: str = "SELL_FILLED",
        tag: str = "",
    ) -> None:
        """Record a filled SELL by reducing or closing the position and cleaning open_trades."""
        sym = self._norm_sym(symbol)
        exec_qty = float(qty or 0.0)
        exec_price = float(price or 0.0)

        try:
            pos = dict(self.positions.get(sym, {}) or {})
        except Exception:
            pos = {}

        cur_qty = float(pos.get("quantity", 0.0) or 0.0)
        new_qty = max(0.0, cur_qty - exec_qty) if cur_qty > 0 and exec_qty > 0 else cur_qty

        # 🔥 CRITICAL: Log position closure BEFORE modifying state
        if new_qty <= 0 and cur_qty > 0:
            logger = logging.getLogger(self.__class__.__name__)
            logger.critical(
                "[SS:MarkPositionClosed] POSITION FULLY CLOSED: symbol=%s cur_qty=%.10f "
                "exec_qty=%.10f exec_price=%.8f reason=%s tag=%s",
                sym, cur_qty, exec_qty, exec_price, reason, tag
            )
            # Journal position closure
            with contextlib.suppress(Exception):
                if hasattr(self, "_journal") and callable(getattr(self, "_journal")):
                    self._journal("POSITION_MARKED_CLOSED", {
                        "symbol": sym,
                        "prev_qty": cur_qty,
                        "executed_qty": exec_qty,
                        "executed_price": exec_price,
                        "remaining_qty": new_qty,
                        "reason": reason,
                        "tag": tag,
                        "timestamp": time.time(),
                    })

        if pos:
            pos["quantity"] = new_qty
            if new_qty <= 0:
                pos["status"] = "CLOSED"
                pos["closed_at"] = time.time()
                pos["closed_reason"] = str(reason or "SELL_FILLED")
                pos["closed_price"] = exec_price
                pos["closed_qty"] = exec_qty
                if tag:
                    pos["closed_tag"] = str(tag)
            await self.update_position(sym, pos)

        try:
            ot = self.open_trades if isinstance(self.open_trades, dict) else {}
            tr = dict(ot.get(sym, {}) or {})
            if tr:
                tr_qty = float(tr.get("quantity", 0.0) or 0.0)
                tr_new_qty = max(0.0, tr_qty - exec_qty) if tr_qty > 0 and exec_qty > 0 else tr_qty
                if tr_new_qty <= 0:
                    # ⚠️ IMPORTANT: Log removal of open_trades entry
                    logger = logging.getLogger(self.__class__.__name__)
                    logger.warning(
                        "[SS:OpenTradesRemoved] Removing from open_trades: symbol=%s qty=%.10f reason=%s",
                        sym, tr_qty, reason
                    )
                    ot.pop(sym, None)
                else:
                    tr["quantity"] = tr_new_qty
                    ot[sym] = tr
        except Exception:
            pass

        try:
            await self.emit_event("PositionClosed", {
                "symbol": sym,
                "qty": exec_qty,
                "price": exec_price,
                "reason": str(reason or "SELL_FILLED"),
                "tag": str(tag or ""),
                "timestamp": time.time(),
            })
        except Exception:
            pass

    async def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        return self.positions.get(self._norm_sym(symbol))

    async def get_position_quantity(self, symbol: str) -> float:
        p = await self.get_position(symbol)
        if not p:
            return 0.0
        qty = float(p.get("quantity", 0.0))
        fee_base = float(p.get("buy_fee_base", 0.0) or 0.0)
        return max(0.0, qty - fee_base)

    # -------- Sentiment/Signals/Regimes --------
    async def set_volatility_regime(self, symbol: str, timeframe: str, regime: str, atrp: Optional[float] = None) -> None:
        async with self._lock_context("global"):
            self.volatility_regimes.setdefault(symbol, {})[timeframe] = {"regime": regime, "atrp": atrp, "timestamp": time.time()}
    async def get_volatility_regime(self, symbol: str, timeframe: str, max_age_seconds: int = 3600) -> Optional[Dict[str, Any]]:
        d = self.volatility_regimes.get(symbol, {}).get(timeframe)
        if not d: return None
        if time.time() - d["timestamp"] > max_age_seconds: return None
        return d
    async def set_sentiment(self, symbol: str, score: float) -> None:
        async with self._lock_context("signals"):
            self.sentiment_scores[symbol] = (float(score), time.time())
    async def get_sentiment(self, symbol: str, max_age_seconds: int = 1800) -> Optional[float]:
        s = self.sentiment_scores.get(symbol)
        if not s: return None
        score, ts = s
        return score if time.time() - ts <= max_age_seconds else None

    # -------- ML Position Scaling --------
    async def set_ml_position_scale(self, symbol: str, scale: float) -> None:
        """
        Store ML model position scale multiplier for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            scale: Position scale multiplier (1.0 = no change, 1.5 = 50% larger, 0.8 = 20% smaller)
        """
        async with self._lock_context("signals"):
            self.ml_position_scale[symbol] = (float(scale), time.time())

    async def get_ml_position_scale(self, symbol: str, default: float = 1.0) -> float:
        """
        Get ML model position scale multiplier for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            default: Default scale if not found (default 1.0 = no scaling)
            
        Returns:
            Position scale multiplier as float
        """
        s = self.ml_position_scale.get(symbol)
        if not s:
            return float(default)
        scale, ts = s
        # Scale is valid; return it (no expiry check as scales are meant to persist per-signal)
        return float(scale)

    async def push_signal(self, symbol: str, signal_data: Dict[str, Any]) -> None:
        """P9: Legacy compatibility shim for push_signal."""
        await self.add_strategy_signal(symbol, signal_data)

    async def get_latest_signal(self, symbol: str) -> Optional[Dict[str, Any]]:
        """P9: Returns the first available agent signal for this symbol."""
        sym = self._norm_sym(symbol)
        per_agent = self.latest_signals_by_symbol.get(sym, {})
        if per_agent:
            return list(per_agent.values())[0]
        return None
    async def get_recent_signals(self, symbol: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        arr = list(self._signal_buffer)[-limit:]
        return [s for s in arr if s.get("symbol") == symbol] if symbol else arr

    # -------- Liquidity reservations --------
    async def reserve_liquidity(self, asset: str, amount: float, ttl_seconds: int = None) -> str:
        ttl = ttl_seconds if ttl_seconds is not None else self.config.reservation_default_ttl
        now = time.time()
        rid = f"{asset}_{now}_{amount}"
        async with self._lock_context("balances"):
            self._quote_reservations.setdefault(asset.upper(), []).append({
                "id": rid,
                "amount": float(amount),
                "created_at": now,
                "expires_at": now + ttl,
            })
        return rid
    async def release_liquidity(self, asset: str, reservation_id: str) -> bool:
        async with self._lock_context("balances"):
            arr = self._quote_reservations.get(asset.upper(), [])
            for i, r in enumerate(arr):
                if r.get("id") == reservation_id:
                    arr.pop(i)
                    return True
        return False

    async def rollback_liquidity(self, asset: str, reservation_id: str) -> bool:
        """
        PHASE 2: Rollback (cancel) a liquidity reservation without releasing it.
        
        Used when an order fails to fill or when execution is cancelled.
        Identical to release_liquidity() but with explicit semantic meaning.
        
        Args:
            asset: Quote asset (e.g., 'USDT')
            reservation_id: Reservation ID returned from reserve_liquidity()
        
        Returns:
            bool: True if reservation was found and rolled back, False otherwise
        """
        async with self._lock_context("balances"):
            arr = self._quote_reservations.get(asset.upper(), [])
            for i, r in enumerate(arr):
                if r.get("id") == reservation_id:
                    arr.pop(i)
                    return True
        return False

    # -------- Liquidation requests (consumed by LiquidationAgent) --------
    async def request_liquidation(self, symbol: str, reason: str = "", *, min_quote_target: float | None = None) -> None:
        """
        Enqueue a liquidation request for a specific symbol. The LiquidationAgent will
        pick this up.
        """
        try:
            req = {"symbol": symbol, "reason": reason, "ts": time.time(), "min_quote_target": min_quote_target}
            self.active_liquidations.add(self._norm_sym(symbol))
            # Non-blocking push
            if not self._liq_requests.full():
                self._liq_requests.put_nowait(req)
        except Exception:
            pass

    def request_reservation_adjustment(self, agent: str, delta: float, reason: str = "") -> bool:
        """
        P9: Meta-Healing request. MetaController requests a budget change, but 
        the SharedState/Allocator remains the ultimate authority.
        
        This method NO LONGER mutates reservations directly. It queues them for the Allocator.
        """
        self.logger.info(f"[SharedState] Reservation adjustment REQUESTED by {agent}: delta={delta:+.2f} ({reason})")
        
        req = {
            "agent": agent,
            "delta": float(delta),
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
        # Add to non-authoritative pending list
        self._pending_reservation_requests.append(req)
        
        # Trigger immediate re-plan to let Allocator process the request
        self.replan_request_event.set()
        
        # Emit event for audit/subscribers
        try:
            if hasattr(self, "emit_event"):
                res = self.emit_event("ReservationAdjustmentRequest", req)
                if asyncio.iscoroutine(res):
                    asyncio.create_task(res)
        except Exception:
            self.logger.exception("Failed to emit ReservationAdjustmentRequest event")
            
        return True

    async def get_next_liquidation_request(self) -> Optional[Dict[str, Any]]:
        if self._liq_requests.empty():
            return None
        return await self._liq_requests.get()

    def clear_liquidation_flag(self, symbol: str) -> None:
        try:
            self.active_liquidations.discard(self._norm_sym(symbol))
        except Exception:
            pass

    # -------- Authoritative Capital Reservations (P9 Strict) --------
    def set_authoritative_reservation(self, agent_id: str, amount: float) -> None:
        """
        Set the authoritative capital budget for an agent.
        This is the source of truth for 'executable capital'.
        Only CapitalAllocator should call this.
        """
        self._authoritative_reservations[agent_id] = float(max(0.0, amount))
        self._authoritative_reservation_ts[agent_id] = time.time()

    def apply_reservation_batch(self, adjustments: List[Dict[str, Any]]) -> None:
        """
        P9: Authoritatively apply a batch of reservation adjustments (deltas).
        This replaces direct dictionary mutation by external components.
        """
        for req in adjustments:
            agent = req.get("agent")
            delta = float(req.get("delta", 0.0))
            if not agent: continue
            
            old_val = self._authoritative_reservations.get(agent, 0.0)
            new_val = float(max(0.0, old_val + delta))
            self._authoritative_reservations[agent] = new_val

    def set_authoritative_reservations(self, reservations: Dict[str, float]) -> None:
        """
        ISSUE 5 FIX: Atomically set multiple authoritative capital budgets.
        This ensures all-or-nothing semantics for crash safety.
        """
        validated = {}
        now = time.time()
        for agent_id, amount in reservations.items():
            validated[agent_id] = float(max(0.0, amount))

        # Atomic update: replace entire dict
        self._authoritative_reservations = validated
        self._authoritative_reservation_ts = {k: now for k in validated}

    def get_pending_reservation_requests(self, drain: bool = False) -> List[Dict[str, Any]]:
        """
        P9: Retrieve pending reservation requests from external components.
        If drain=True, the requests are cleared from the queue.
        """
        if not hasattr(self, "_pending_reservation_requests"):
            return []
            
        requests = self._pending_reservation_requests
        if drain:
            # Drain the list (atomic drain if possible, simple clear for now)
            to_process = list(requests)
            requests.clear()
            return to_process
        return list(requests)

    def get_authoritative_reservation(self, agent_id: str) -> float:
        """
        Get the currently authorized capital budget for an agent.
        Returns 0.0 if no budget is allocated.
        """
        return self._authoritative_reservations.get(agent_id, 0.0)

    def get_authoritative_reservations(self) -> Dict[str, float]:
        """
        P9: Get a copy of all authoritative reservations.
        """
        return dict(self._authoritative_reservations)

    def get_total_authoritative_reservations(self) -> float:
        """Return sum of all authoritative agent budgets."""
        return sum(self._authoritative_reservations.values())
    # -------- Capital Failure Tracking (Hysteresis) --------
    def report_agent_capital_failure(self, agent_id: str) -> None:
        """Record that an agent failed to execute due to capital constraints."""
        self._capital_failures[agent_id] = time.time()

    def get_agent_capital_failure(self, agent_id: str) -> float:
        """Return timestamp of last capital failure for this agent, or 0.0."""
        return self._capital_failures.get(agent_id, 0.0)

    def clear_agent_capital_failure(self, agent_id: str) -> None:
        """Clear the capital failure record for an agent (re-enabling)."""
        self._capital_failures.pop(agent_id, None)

    # -------- Events & health --------
    async def emit_event(self, event_name: str, event_data: Dict[str, Any]) -> None:
        """Structured event emission path; persists in-memory and notifies subscribers."""
        # Persistent storage for specific critical events (e.g. AllocationPlan)
        if event_name == "AllocationPlan":
            self._latest_allocation_plan = dict(event_data or {})
            self.logger.info("[SS] Captured latest AllocationPlan: pool=%.2f", event_data.get("pool_quote", 0))

        ts = event_data.get("ts") or event_data.get("timestamp") or time.time()
        ev_obj = {"name": event_name, "data": event_data, "timestamp": ts}
        self._event_log.append(ev_obj)
        
        # P9 Fix: Ensure subscribers are NOTIFIED
        await self.publish_event(event_name, event_data)

    async def wait_for_event(self, event_name: str) -> None:
        """Wait for a named event to be set."""
        event_map = {
            "AcceptedSymbolsReady": self.accepted_symbols_ready_event,
            "BalancesReady": self.balances_ready_event,
            "MarketDataReady": self.market_data_ready_event,
            "NavReady": self.nav_ready_event,
            "OpsPlaneReady": self.ops_plane_ready_event,
        }
        event = event_map.get(event_name)
        if event:
            await event.wait()
        else:
            self.logger.warning(f"Unknown event name for wait_for_event: {event_name}")
    async def get_recent_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        return list(self._event_log)[-limit:]
    async def publish_event(self, name: str, data: Dict[str, Any]) -> None:
        if not self._subscribers: return
        ev = {"name": name, "data": data, "timestamp": time.time()}
        for q in list(self._subscribers.values()):
            try: q.put_nowait(ev)
            except Exception: pass
    async def subscribe_events(self, subscriber_name: str, max_queue: int = 1000) -> asyncio.Queue:
        q = asyncio.Queue(maxsize=max_queue)
        self._subscribers[subscriber_name] = q
        return q
    async def unsubscribe(self, subscriber_name: str) -> None:
        self._subscribers.pop(subscriber_name, None)

    # -------- Perf & maintenance --------
    def get_readiness_snapshot(self) -> Dict[str, bool]:
        """Lightweight, synchronous readiness view for Watchdog/StrategyManager."""
        opr = self.is_ops_plane_ready()  # Use the bootstrap-aware method
        lifecycle = self.get_system_lifecycle_state()
            
        return {
            "accepted_symbols_ready": self.accepted_symbols_ready_event.is_set(),
            "balances_ready": self.balances_ready_event.is_set(),
            "market_data_ready": self.market_data_ready_event.is_set(),
            "ops_plane_ready": opr,
            "lifecycle_state": lifecycle,  # BOOTSTRAP, LIVE_IDLE, or ACTIVE
            "is_bootstrap": self.is_bootstrap_mode(),
        }

    def mark_bootstrap_signal_validated(self) -> None:
        """
        🔧 BOOTSTRAP COMPLETION FIX: Mark bootstrap complete when first signal is validated
        
        CRITICAL: Bootstrap should complete on SIGNAL VALIDATION, not trade execution.
        
        Problem:
        - In shadow mode, signals are validated but NO trade is executed
        - If bootstrap only completes on trade execution, shadow mode DEADLOCKS forever
        - System waits for first trade, but shadow mode has no orders to fill
        
        Solution:
        - Complete bootstrap when MetaController validates the first signal
        - Set first_signal_validated_at timestamp
        - Prevent bootstrap logic from re-firing on subsequent validations
        
        Usage:
        - Called by MetaController.propose_exposure_directive() after signal validation passes
        - Called BEFORE execution (so shadow mode works too)
        - Idempotent: safe to call multiple times
        """
        if self.metrics.get("first_signal_validated_at") is not None:
            # Already marked, skip (idempotent)
            return
        
        now = time.time()
        self.metrics["first_signal_validated_at"] = now
        self.metrics["bootstrap_completed"] = True
        
        # Also persist to bootstrap_metrics for restart safety
        self.bootstrap_metrics._cached_metrics["first_signal_validated_at"] = now
        self.bootstrap_metrics._cached_metrics["bootstrap_completed"] = True
        self.bootstrap_metrics._write(self.bootstrap_metrics._cached_metrics)
        
        self.logger.warning(
            "[BOOTSTRAP] ✅ Bootstrap completed by first signal validation at %.1f "
            "(not waiting for trade execution). Shadow mode deadlock prevented.",
            now
        )

    def is_cold_bootstrap(self) -> bool:
        """
        Returns True ONLY when ALL of these conditions are met:
          1. Total historical trades == 0 (no prior execution history) [PERSISTED]
          2. No database/persistence file exists (true first-ever launch)
          3. COLD_BOOTSTRAP_ENABLED flag is explicitly True (opt-in)
          4. LIVE_MODE is NOT True (live systems never force-bootstrap)

        Phase 2 Enhancement: Checks persisted bootstrap metrics, so restart
        doesn't reset bootstrap detection. This prevents bootstrap re-entry.
        
        This prevents bootstrap logic (forced seed trades, sell blocks, confidence
        overrides) from firing on restarts, when a DB exists, or in live trading.
        Startup must be pure reconciliation — no forced action.
        
        🔧 BOOTSTRAP FIX: Also checks for first signal validation.
        If any signal has been validated (even without trade execution),
        bootstrap is considered complete. This prevents shadow mode deadlock.
        """
        # Condition 1: Zero historical trades [NOW PERSISTED]
        # Phase 2: Check both in-memory and persisted metrics
        # 🔧 BOOTSTRAP FIX: Also check for signal validation (not just trade execution)
        has_signal_or_trade_history = (
            self.metrics.get("first_trade_at") is not None
            or self.metrics.get("first_signal_validated_at") is not None
            or self.metrics.get("total_trades_executed", 0) > 0
            or self.bootstrap_metrics.get_first_trade_at() is not None
            or self.bootstrap_metrics.get_total_trades_executed() > 0
        )
        if has_signal_or_trade_history:
            return False

        # Condition 2: No persistent state (DB file / snapshot) exists
        try:
            import os
            db_path = getattr(self, "_db_path", None) or getattr(self.config, "DB_PATH", None) or getattr(self.config, "DATABASE_PATH", None)
            if db_path and os.path.exists(str(db_path)):
                return False
            snapshot_path = getattr(self.config, "SNAPSHOT_PATH", None) or getattr(self.config, "STATE_SNAPSHOT_FILE", None)
            if snapshot_path and os.path.exists(str(snapshot_path)):
                return False
        except Exception:
            pass

        # Condition 3: Explicit opt-in flag required
        cold_enabled = False
        try:
            v = getattr(self.config, "COLD_BOOTSTRAP_ENABLED", None)
            if v is None:
                import os as _os
                v = _os.getenv("COLD_BOOTSTRAP_ENABLED")
            if v is not None:
                cold_enabled = str(v).strip().lower() in ("1", "true", "yes", "on")
        except Exception:
            cold_enabled = False
        if not cold_enabled:
            return False

        # Condition 4: Must NOT be in live trading mode
        try:
            is_live = bool(getattr(self.config, "LIVE_MODE", False))
            if is_live:
                return False
        except Exception:
            pass

        return True

    def get_cold_bootstrap_duration_sec(self) -> float:
        """
        TIER 2: Cold-bootstrap duration clarification.
        Returns min(30 seconds, time_until_first_successful_trade) as per policy audit.
        Duration ends when first trade executes successfully.
        """
        if self.metrics.get("first_trade_at") is not None:
            start_time = self.metrics.get("startup_time", 0.0)
            if start_time > 0:
                duration = self.metrics["first_trade_at"] - start_time
                return min(30.0, duration)
        return 0.0  # Not yet in bootstrap or already past first trade

    def is_bootstrap_mode(self) -> bool:
        """Backward compatibility alias for is_cold_bootstrap()."""
        return self.is_cold_bootstrap()

    def _is_position_significant(self, symbol: str, qty: float) -> bool:
        """
        Phase 1: Helper method to determine if a position is significant (not dust).
        A position is significant if its notional value >= PERMANENT_DUST_USDT_THRESHOLD.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            qty: Position quantity in base asset
            
        Returns:
            True if position notional >= threshold, False otherwise
        """
        # Get configured threshold (default $1.0)
        threshold = float(getattr(self.config, "PERMANENT_DUST_USDT_THRESHOLD", 1.0))
        
        try:
            # Get current price for the symbol
            current_price = self.latest_price(symbol)
            if current_price is None or current_price <= 0:
                self.logger.warning(f"Cannot determine significance for {symbol}: no price available")
                return False
            
            # Calculate notional value
            notional_value = abs(qty) * current_price
            
            # Check if significant
            is_significant = notional_value >= threshold
            
            if not is_significant:
                self.logger.debug(
                    f"Position {symbol} qty={qty} is DUST: notional=${notional_value:.2f} < ${threshold:.2f}"
                )
            
            return is_significant
        except Exception as e:
            self.logger.warning(f"Error checking position significance for {symbol}: {e}")
            # On error, assume significant to avoid false positives for dust
            return True

    async def get_portfolio_state(self) -> str:
        """
        Phase 1: Portfolio State Machine - distinguishes between dust and empty.
        
        Returns one of 5 states as string representation of PortfolioState enum:
        - "COLD_BOOTSTRAP": Never traded before (is_cold_bootstrap() = True)
        - "EMPTY_PORTFOLIO": No positions and no dust at all
        - "PORTFOLIO_WITH_DUST": Only dust positions exist (all positions < threshold)
        - "PORTFOLIO_ACTIVE": Significant positions exist
        - "PORTFOLIO_RECOVERING": Error state during recovery
        
        This breaks the dust loop at step 2 by preventing bootstrap when dust exists.
        """
        if self.is_cold_bootstrap():
            return PortfolioState.COLD_BOOTSTRAP.value

        try:
            all_positions = self.get_open_positions()
            
            # No positions at all
            if not all_positions:
                self.logger.info("[SS:PortState] Portfolio is EMPTY_PORTFOLIO: no positions")
                return PortfolioState.EMPTY_PORTFOLIO.value
            
            # Separate significant from dust positions
            significant_positions = []
            dust_positions = []
            
            for position in all_positions:
                symbol = position.get("symbol")
                qty = float(position.get("qty", 0.0))
                
                if self._is_position_significant(symbol, qty):
                    significant_positions.append(position)
                else:
                    dust_positions.append(position)
            
            # Determine state based on position types
            if significant_positions:
                self.logger.info(
                    "[SS:PortState] Portfolio is PORTFOLIO_ACTIVE: "
                    "significant_positions=%d, dust_positions=%d",
                    len(significant_positions), len(dust_positions)
                )
                return PortfolioState.PORTFOLIO_ACTIVE.value
            elif dust_positions:
                self.logger.warning(
                    "[SS:PortState] Portfolio is PORTFOLIO_WITH_DUST: "
                    "only_dust_positions=%d (no significant positions)",
                    len(dust_positions)
                )
                return PortfolioState.PORTFOLIO_WITH_DUST.value
            else:
                # Shouldn't reach here but defensive
                self.logger.info("[SS:PortState] Portfolio is EMPTY_PORTFOLIO: all_positions_were_dust")
                return PortfolioState.EMPTY_PORTFOLIO.value
                
        except Exception as e:
            self.logger.error(f"get_portfolio_state check failed: {e}, defaulting to RECOVERING")
            return PortfolioState.PORTFOLIO_RECOVERING.value

    async def is_portfolio_flat(self) -> bool:
        """
        Returns True only when portfolio is completely empty (not dust-only).
        Note: This now properly distinguishes dust-only from empty states.
        """
        state = await self.get_portfolio_state()
        
        # Portfolio is flat only if empty
        is_flat = state == PortfolioState.EMPTY_PORTFOLIO.value
        
        if is_flat:
            self.logger.debug("[SS:IsFlat] Portfolio is FLAT - completely empty")
        else:
            self.logger.debug(f"[SS:IsFlat] Portfolio NOT flat - state={state}")
        
        return is_flat

    # ===== PHASE 2: CAPITAL STATE DETECTION =====
    async def get_capital_state(self) -> str:
        """
        PHASE 2 ENHANCEMENT: Returns capital availability state.
        States:
        - SUFFICIENT: Free capital >= min_viable_quote
        - INSUFFICIENT: No free capital available
        - FRAGMENTED: Capital locked in many small reservations
        - RESERVED: Capital locked but should recover soon
        """
        try:
            spendable = await self.free_usdt() if hasattr(self, "free_usdt") else 0.0
            min_viable = float(getattr(self.config, "MIN_EXECUTABLE_QUOTE", 10.0))
            
            # Count active reservations
            total_reserved = 0.0
            reservations_count = 0
            try:
                for sym, reservations in getattr(self, "_quote_reservations", {}).items():
                    if isinstance(reservations, list):
                        for r in reservations:
                            total_reserved += float(r.get("amount", 0.0))
                            reservations_count += 1
                    elif isinstance(reservations, dict):
                        total_reserved += float(reservations.get("amount", 0.0))
                        reservations_count += 1
            except Exception:
                pass
            
            if spendable >= min_viable:
                return "SUFFICIENT"
            elif total_reserved > spendable * 2:
                return "FRAGMENTED"
            elif total_reserved > 0:
                return "RESERVED"
            else:
                return "INSUFFICIENT"
        except Exception as e:
            self.logger.debug(f"get_capital_state failed: {e}")
            return "INSUFFICIENT"

    # ===== PHASE 2: MARKET STATE DETECTION =====
    async def get_market_state(self) -> str:
        """
        PHASE 2 ENHANCEMENT: Returns market liquidity/volatility state.
        States:
        - NORMAL: Standard bid-ask spread, moderate volatility
        - LOW_LIQUIDITY: Wide spreads, low order book depth
        - HIGH_VOLATILITY: High price movement, rapid changes
        """
        try:
            # Simple heuristic: check bid-ask spreads and recent volatility
            volatility_threshold = 0.05  # 5% in recent candles
            spread_threshold_bps = 30  # 30 basis points = 0.3%
            
            # Estimate from recent prices and bid-ask data
            recent_volatility = 0.0
            try:
                # Check last 10 price observations for volatility
                price_history = getattr(self, "_price_history", {})
                if isinstance(price_history, dict):
                    for prices in list(price_history.values())[:10]:
                        if isinstance(prices, (list, deque)) and len(prices) >= 2:
                            # Calculate simple volatility
                            arr = list(prices)
                            if len(arr) > 1:
                                pct_changes = [abs((arr[i] - arr[i-1]) / arr[i-1]) for i in range(1, len(arr))]
                                if pct_changes:
                                    recent_volatility = max(recent_volatility, sum(pct_changes) / len(pct_changes))
            except Exception:
                pass
            
            if recent_volatility > volatility_threshold:
                return "HIGH_VOLATILITY"
            
            # Check bid-ask spreads if available
            try:
                spreads = getattr(self, "_bid_ask_spreads", {})
                if spreads:
                    avg_spread = sum(spreads.values()) / len(spreads)
                    if avg_spread > (spread_threshold_bps / 10000):
                        return "LOW_LIQUIDITY"
            except Exception:
                pass
            
            return "NORMAL"
        except Exception as e:
            self.logger.debug(f"get_market_state failed: {e}")
            return "NORMAL"

    def get_system_lifecycle_state(self) -> str:
        """Returns BOOTSTRAP, LIVE_IDLE, or ACTIVE based on trading history and current state."""
        if self.is_cold_bootstrap():
            return "BOOTSTRAP"
        
        # Check if we have active positions
        has_positions = False
        try:
            for p in self.positions.values():
                if float(p.get("quantity", 0.0)) > 0:
                    has_positions = True
                    break
        except Exception:
            pass
        
        if has_positions:
            return "ACTIVE"
        return "LIVE_IDLE"

    async def is_circuit_breaker_open(self, component: str = "exchange") -> bool:
        """P9: Explicit check for circuit breaker status."""
        cb = self._circuit_breakers.get(component)
        if cb and cb.state == CircuitBreakerState.OPEN:
            return True
        return False

    def is_ops_plane_ready(self) -> bool:
        """
        Unified Ops-Plane Readiness Check (P9).
        Central logic for deciding if trading is architecturally safe.
        """
        # 1. CIRCUIT BREAKER CHECK
        if self._circuit_breakers["exchange"].state == CircuitBreakerState.OPEN:
            self.logger.warning("[SS] OpsPlane: Exchange circuit breaker OPEN. Denying readiness.")
            return False

        # 2. BOOTSTRAP MODE: Relaxed requirements
        if self.is_bootstrap_mode():
            has_symbols = len(self.get_accepted_symbols_snapshot()) > 0
            st = str(self.component_statuses.get("ExecutionManager", {}).get("status", "")).lower()
            exec_healthy = st in ("healthy", "running", "ok", "initialized", "operational")
            return has_symbols and exec_healthy
        
        # 3. LIVE MODE: Hard dependency on event trigger
        if not self.ops_plane_ready_event.is_set():
            return False
            
        # 4. ACTIVE LIFECYCLE: Must have either budget or skin in the game
        if sum(self._authoritative_reservations.values()) > 0:
            return True
            
        for p in self.positions.values():
            if float(p.get("quantity", 0.0)) > 0:
                return True
            
        return False

    def get_latest_signals_by_symbol(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """P9: Jurisdictional getter for latest signals."""
        return dict(self.latest_signals_by_symbol)

    async def get_performance_stats(self) -> Dict[str, Any]:
        """
        Return aggregated performance and cache stats for observability dashboards.
        Safe to call at any time; does not require external I/O.
        """
        stats: Dict[str, Any] = {}
        # Method timings
        for name, times in self._performance_stats["method_call_times"].items():
            if times:
                L = list(times)
                stats[f"method_{name}"] = {
                    "calls": len(L),
                    "avg_time": sum(L) / len(L),
                    "max_time": max(L),
                    "min_time": min(L),
                }
        # Cache hit/miss rates
        for name, rates in self._performance_stats["cache_hit_rates"].items():
            total = rates["hits"] + rates["misses"]
            if total > 0:
                stats[f"cache_{name}"] = {
                    "hit_rate": rates["hits"] / total,
                    "hits": rates["hits"],
                    "misses": rates["misses"],
                }
        # Memory footprint
        stats["memory"] = {
            "event_log_size": len(self._event_log),
            "signal_buffer_size": len(self._signal_buffer),
            "trade_history_size": len(self.trade_history),
            "price_cache_size": len(self._price_cache),
        }
        return stats

    async def start_background_tasks(self) -> None:
        if not self._background_tasks["memory_optimization"] or self._background_tasks["memory_optimization"].done():
            self._background_tasks["memory_optimization"] = asyncio.create_task(self._memory_optimization_loop(), name="SharedState.memory_optimization")
        # Start wallet sync only if an exchange client is available
        if self._exchange_client and (not self._background_tasks.get("wallet_sync") or self._background_tasks["wallet_sync"].done()):
            self._background_tasks["wallet_sync"] = asyncio.create_task(self._wallet_sync_loop(), name="SharedState.wallet_sync")
    async def _memory_optimization_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(self.config.memory_optimization_interval)
                await self._optimize_memory()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Memory optimization failed: {e}")
    async def _optimize_memory(self) -> None:
        now = time.time()
        # P9 Phase 4: Intent-level accumulation cleanup
        await self.expire_old_intents(now)
        # prune stale price cache
        stale = [s for s, (_, ts) in self._price_cache.items() if now - ts > self.config.price_cache_ttl_seconds]
        for s in stale: self._price_cache.pop(s, None)
        # prune expired reservations
        for asset, arr in list(self._quote_reservations.items()):
            valid = [r for r in arr if r.get("expires_at", 0) > now]
            if valid: self._quote_reservations[asset] = valid
            else: self._quote_reservations.pop(asset, None)

    # -------- Shutdown --------
    async def shutdown(self) -> None:
        tasks = [t for t in self._background_tasks.values() if t and not t.done()]
        for t in tasks: t.cancel()
        if tasks:
            try: await asyncio.gather(*tasks, return_exceptions=True)
            except Exception: pass
        self.logger.info("SharedState shutdown completed")
    def get_positions(self) -> Dict[str, Dict[str, Any]]:
        """
        ✅ CANONICAL: Get all positions (both open and closed).

        Returns:
            Dict mapping symbol → position_data
        """
        return dict(self.positions)

    # Alias kept for callers that used the old name
    get_positions_by_symbol = get_positions
    
    # ----------- Additional helpers & wrappers -----------

    async def safe_price(self, symbol: str, default: float = 0.0) -> float:
        """Return price for symbol or default if not available."""
        return float(self.latest_prices.get(self._norm_sym(symbol), default))

    async def get_symbol_filters_cached(self, symbol: str) -> Dict[str, Any]:
        """Return a shallow copy of symbol filters for symbol (if any)."""
        return dict(self.symbol_filters.get(self._norm_sym(symbol), {}))

    def get_positions_snapshot(self) -> Dict[str, Dict[str, Any]]:
        """Return a shallow copy of all positions, branching by trading mode."""
        if self.trading_mode == "shadow":
            return dict(self.virtual_positions)
        return dict(self.positions)

    def is_ops_plane_ready(self) -> bool:
        """
        Idempotent, centralized readiness check for OpsPlaneReady emission.
        
        Returns True if:
        1. Authoritative reservations have positive budget allocated, OR
        2. Positions snapshot contains at least one position with qty > 0
        
        This is the authoritative source of truth for ops plane readiness.
        No early returns, no race conditions, no timing-dependent logic.
        """
        try:
            # Check 1: Authoritative reservations
            reservations = self.get_authoritative_reservations()
            if sum(reservations.values()) > 0:
                return True
            
            # Check 2: Positions snapshot
            snap = self.get_positions_snapshot()
            for p in snap.values():
                if float(p.get("quantity", 0.0)) > 0:
                    return True
            
            return False
        except Exception:
            # Defensive: treat any exception as "not ready"
            return False

    def _sync_heal_position_states(self) -> None:
        """
        Self-heal stale position classification state.
        Ensures dust positions are properly marked and don't block routing/capacity.
        Call this explicitly rather than hiding mutations inside getters.
        """
        for sym, pos_data in list(self.positions.items()):
            if not isinstance(pos_data, dict):
                continue
            qty = float(pos_data.get("quantity", 0.0) or pos_data.get("qty", 0.0) or 0.0)
            if qty <= 0:
                continue
            is_open, value_usdt, floor = self.classify_position_snapshot(sym, pos_data)
            if not is_open:
                pos_data["status"] = "DUST"
                pos_data["state"] = PositionState.DUST_LOCKED.value
                pos_data["is_significant"] = False
                pos_data["is_dust"] = True
                pos_data["_is_dust"] = True
                pos_data["open_position"] = False
                pos_data["capital_occupied"] = 0.0
                pos_data["value_usdt"] = float(value_usdt)
                pos_data["significant_floor_usdt"] = float(floor)
                self.positions[sym] = pos_data
                self.open_trades.pop(sym, None)
            else:
                pos_data["status"] = "SIGNIFICANT"
                pos_data["state"] = PositionState.ACTIVE.value
                pos_data["is_significant"] = True
                pos_data["is_dust"] = False
                pos_data["_is_dust"] = False
                pos_data["open_position"] = True
                pos_data["capital_occupied"] = float(value_usdt)
                pos_data["value_usdt"] = float(value_usdt)
                pos_data["significant_floor_usdt"] = float(floor)
                self.positions[sym] = pos_data

    def get_open_positions(self) -> Dict[str, Dict[str, Any]]:
        """
        Return only OPEN SIGNIFICANT positions.

        Canonical invariant:
        position is OPEN iff position_value_usdt >= significant_position_floor.

        Side effect: heals stale position states (dust mis-classified as significant).
        
        ARCHITECTURE FIX: Branches by trading_mode to return correct positions source.
        """
        self._sync_heal_position_states()
        result = {}
        
        # Branch by trading mode to use correct positions source
        positions_source = self.virtual_positions if self.trading_mode == "shadow" else self.positions
        
        for sym, pos_data in list(positions_source.items()):
            if not isinstance(pos_data, dict):
                continue
            qty = float(pos_data.get("quantity", 0.0) or pos_data.get("qty", 0.0) or 0.0)
            if qty <= 0:
                continue
            if pos_data.get("is_significant", False) and pos_data.get("open_position", False):
                result[sym] = pos_data
        return result

    def get_position_qty(self, symbol: str) -> float:
        """Return the quantity for a position, or 0.0 if not present."""
        p = self.positions.get(self._norm_sym(symbol))
        if not p:
            return 0.0
        qty = float(p.get("quantity", 0.0))
        fee_base = float(p.get("buy_fee_base", 0.0) or 0.0)
        return max(0.0, qty - fee_base)

    def record_exit_reason(self, symbol: str, reason: str, source: Optional[str] = None) -> None:
        """Record the last exit reason for a symbol (used for anti-churn gating)."""
        try:
            sym = self._norm_sym(symbol)
            now = time.time()
            self.last_exit_reason[sym] = str(reason)
            self.last_exit_ts[sym] = float(now)
            if source is not None:
                self.last_exit_source[sym] = str(source)
        except Exception:
            pass

    def get_last_exit_reason(self, symbol: str) -> Optional[str]:
        """Return the last recorded exit reason for a symbol, if any."""
        try:
            return self.last_exit_reason.get(self._norm_sym(symbol))
        except Exception:
            return None

    def get_last_exit_ts(self, symbol: str) -> float:
        """Return the last exit timestamp for a symbol, or 0.0 if unknown."""
        try:
            return float(self.last_exit_ts.get(self._norm_sym(symbol), 0.0) or 0.0)
        except Exception:
            return 0.0

    def open_positions_count(self) -> int:
        """
        Return count of OPEN positions.

        ✅ FIX: EXCLUDE DUST positions from count.
        Dust positions do NOT count toward portfolio occupancy.
        Only SIGNIFICANT positions count.

        AUTHORITATIVE FLAT CHECK: position.status in {"OPEN", "PARTIALLY_FILLED", "SIGNIFICANT"}
        and NOT marked as dust
        """
        return int(len(self.get_open_positions()))


    # ---- Optional helpers ----
    async def push_agent_signal(self, agent: str, symbol: str, signal_data: Dict[str, Any]) -> None:
        """Push a signal tagged with agent name."""
        sd = dict(signal_data)
        sd["agent"] = agent
        await self.push_signal(symbol, sd)

    def get_active_symbols(self, *, limit: Optional[int] = None) -> List[str]:
        """
        Return a prioritized list of symbols for agents:
        1) accepted_symbols (wallet-forced + normal)
        2) positions we currently hold (so liquidation/management never misses them)
        3) any other known symbols cached in self.symbols
        The list is de-duplicated while preserving the above priority order.
        If `limit` is provided or `config.active_symbols_default_limit > 0`, the result is truncated.
        """
        # Keep internal caches consistent so agents always see the full list
        try:
            self.ensure_symbol_caches_consistent()
        except Exception:
            pass
        seen: Set[str] = set()
        out: List[str] = []

        # 1) Accepted symbols first
        for s in self.accepted_symbols.keys():
            ss = self._norm_sym(s)
            if ss not in seen:
                out.append(ss)
                seen.add(ss)

        # 2) Fallback to open positions (ensures agents see all held inventory)
        if getattr(self.config, "active_symbols_fallback_from_positions", True):
            for s in self.positions.keys():
                ss = self._norm_sym(s)
                if ss not in seen:
                    out.append(ss)
                    seen.add(ss)

        # 3) Wallet assets direct (Ensures symbols we hold are always visible even if not in accepted/symbol sets)
        try:
            quote_asset = getattr(self.config, "quote_asset", "USDT")
            for asset in self.balances.keys():
                if asset.upper() == quote_asset.upper(): continue
                sym = f"{asset.upper()}{quote_asset.upper()}"
                ss = self._norm_sym(sym)
                if ss not in seen:
                    out.append(ss)
                    seen.add(ss)
        except Exception:
            pass

        # 4) Finally any other known symbols (cache)
        for s in self.symbols.keys():
            ss = self._norm_sym(s)
            if ss not in seen:
                out.append(ss)
                seen.add(ss)

        # Optional truncation
        if limit is None:
            limit = int(getattr(self.config, "active_symbols_default_limit", 0) or 0)
        
        if limit > 0:
            out = out[:limit]
        return out

    def get_analysis_symbols(self) -> List[str]:
        """
        Full, untruncated universe for trader analysis.
        NEVER limited.
        """
        return self.get_active_symbols(limit=0)

    def is_symbol_temporarily_blocked(self, symbol: str, side: str, window_seconds: int = 60) -> bool:
        """Return True if symbol/side had any rejection within the last `window_seconds`.
        This implements a short cooldown to avoid immediate repeated attempts that are guaranteed to fail.
        """
        s = str(symbol).upper()
        sd = str(side).upper()
        now = time.time()
        for (sym_k, side_k, reason_k), ts in list(self.rejection_timestamps.items()):
            if sym_k == s and side_k == sd and (now - ts) <= float(window_seconds):
                return True
        return False

    # ═══════════════════════════════════════════════════════════════════════════════
    # SIGNAL BUFFER CONSENSUS: Adaptive Signal Window Methods
    # ═══════════════════════════════════════════════════════════════════════════════
    
    def add_signal_to_consensus_buffer(self, symbol: str, signal: Dict[str, Any]) -> None:
        """Add a signal to the consensus buffer with timestamp."""
        try:
            symbol = str(symbol).upper()
            # Ensure signal has timestamp
            if "ts" not in signal or signal["ts"] is None:
                signal["ts"] = time.time()
            
            # Add to buffer
            self.signal_consensus_buffer[symbol].append(signal)
            
            # Keep only max signals
            max_signals = int(self.signal_buffer_max_signals_per_symbol)
            if len(self.signal_consensus_buffer[symbol]) > max_signals:
                self.signal_consensus_buffer[symbol] = self.signal_consensus_buffer[symbol][-max_signals:]
            
            # Update stats
            self.signal_buffer_stats["signals_received"] += 1
            
            self.logger.debug(
                "[SignalBuffer:ADD] Symbol %s: signal from %s (action=%s, conf=%.2f, ts=%.1f)",
                symbol, signal.get("agent", "Unknown"), signal.get("action", "?"),
                signal.get("confidence", 0.0), signal.get("ts", 0.0)
            )
        except Exception as e:
            self.logger.warning("[SignalBuffer:ADD] Error adding signal for %s: %s", symbol, e)
    
    def get_valid_buffered_signals(self, symbol: str, window_sec: Optional[float] = None) -> List[Dict[str, Any]]:
        """Get all signals for symbol within the time window."""
        try:
            symbol = str(symbol).upper()
            if symbol not in self.signal_consensus_buffer:
                return []
            
            window = float(window_sec or self.signal_buffer_window_sec)
            now = time.time()
            max_age = float(self.signal_buffer_max_age_sec)
            
            valid_signals = []
            for sig in self.signal_consensus_buffer[symbol]:
                sig_ts = float(sig.get("ts", now))
                age = now - sig_ts
                
                # Keep if within window AND not expired
                if age <= window or age <= max_age:
                    valid_signals.append(sig)
            
            return valid_signals
        except Exception as e:
            self.logger.warning("[SignalBuffer:GET] Error getting signals for %s: %s", symbol, e)
            return []
    
    def compute_consensus_score(self, symbol: str, action: str = "BUY", 
                               window_sec: Optional[float] = None) -> Tuple[float, int]:
        """
        Compute weighted consensus score for a symbol/action.
        IMPORTANT: MLForecaster is NOT included in directional voting (only position sizing)
        
        Returns: (score, signal_count)
        - score: 0.0 to 1.0 (sum of agent weights that agree)
        - signal_count: number of signals used (excluding MLForecaster)
        """
        try:
            symbol = str(symbol).upper()
            action = str(action).upper()
            
            # Get valid signals within window
            valid_signals = self.get_valid_buffered_signals(symbol, window_sec)
            if not valid_signals:
                return 0.0, 0
            
            # Filter by action and minimum confidence
            # CRITICAL: Exclude MLForecaster from directional consensus
            matching_signals = [
                s for s in valid_signals
                if str(s.get("action", "")).upper() == action
                and float(s.get("confidence", 0.0)) >= float(self.signal_consensus_min_confidence)
                and str(s.get("agent", "Unknown")).upper() != "MLFORECASTER"  # Exclude MLForecaster
            ]
            
            if not matching_signals:
                return 0.0, 0
            
            # Compute weighted score (only from TrendHunter and DipSniper)
            score = 0.0
            for sig in matching_signals:
                agent = str(sig.get("agent", "Unknown"))
                weight = self.agent_consensus_weights.get(agent, 0.0)  # Default 0.0 (no weight if unknown)
                score += weight
            
            # Clamp to [0, 1]
            score = min(1.0, max(0.0, score))
            
            self.logger.debug(
                "[SignalBuffer:CONSENSUS] %s %s: score=%.2f signals=%d threshold=%.2f (MLForecaster excluded from voting)",
                symbol, action, score, len(matching_signals), float(self.signal_consensus_threshold)
            )
            
            return score, len(matching_signals)
        except Exception as e:
            self.logger.warning("[SignalBuffer:CONSENSUS] Error computing consensus for %s: %s", symbol, e)
            return 0.0, 0
    
    def check_consensus_reached(self, symbol: str, action: str = "BUY",
                               window_sec: Optional[float] = None) -> bool:
        """Check if consensus threshold is reached for symbol/action."""
        try:
            score, count = self.compute_consensus_score(symbol, action, window_sec)
            threshold = float(self.signal_consensus_threshold)
            reached = score >= threshold
            
            if reached:
                self.signal_buffer_stats["consensus_trades_triggered"] += 1
                self.logger.warning(
                    "[SignalBuffer:REACHED] ✅ CONSENSUS REACHED for %s %s (score=%.2f >= threshold=%.2f)",
                    symbol, action, score, threshold
                )
            else:
                self.signal_buffer_stats["consensus_failures"] += 1
            
            return reached
        except Exception as e:
            self.logger.warning("[SignalBuffer:CHECK] Error checking consensus for %s: %s", symbol, e)
            return False
    
    def get_consensus_signal(self, symbol: str, action: str = "BUY",
                            window_sec: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        Get best consensus signal if threshold reached.
        Returns the highest-confidence signal of matching action type.
        """
        try:
            symbol = str(symbol).upper()
            
            # Check if consensus reached
            if not self.check_consensus_reached(symbol, action, window_sec):
                return None
            
            # Get all valid matching signals
            valid_signals = self.get_valid_buffered_signals(symbol, window_sec)
            matching = [
                s for s in valid_signals
                if str(s.get("action", "")).upper() == str(action).upper()
            ]
            
            if not matching:
                return None
            
            # Return highest confidence signal (or best-weighted signal)
            best_sig = max(matching, key=lambda s: float(s.get("confidence", 0.0)))
            
            # Mark with consensus metadata
            best_sig["_consensus_reached"] = True
            best_sig["_consensus_signal_count"] = len(matching)
            best_sig["_from_buffer"] = True
            
            self.logger.warning(
                "[SignalBuffer:MERGED] %s %s consensus signal selected (agent=%s, conf=%.2f, sig_count=%d)",
                symbol, action, best_sig.get("agent", "Unknown"),
                best_sig.get("confidence", 0.0), len(matching)
            )
            
            return best_sig
        except Exception as e:
            self.logger.warning("[SignalBuffer:GET] Error getting consensus signal for %s: %s", symbol, e)
            return None
    
    def clear_buffer_for_symbol(self, symbol: str) -> None:
        """Clear all buffered signals for a symbol (after trade execution)."""
        try:
            symbol = str(symbol).upper()
            if symbol in self.signal_consensus_buffer:
                count = len(self.signal_consensus_buffer[symbol])
                self.signal_consensus_buffer[symbol].clear()
                self.signal_buffer_stats["buffer_flushes"] += 1
                self.logger.info("[SignalBuffer:CLEAR] Cleared %d signals for %s", count, symbol)
        except Exception as e:
            self.logger.warning("[SignalBuffer:CLEAR] Error clearing buffer for %s: %s", symbol, e)
    
    def cleanup_expired_signals(self) -> None:
        """Remove all expired signals from buffer (call periodically)."""
        try:
            now = time.time()
            max_age = float(self.signal_buffer_max_age_sec)
            total_removed = 0
            
            for symbol in list(self.signal_consensus_buffer.keys()):
                buffer = self.signal_consensus_buffer[symbol]
                initial_count = len(buffer)
                
                # Keep only non-expired signals
                self.signal_consensus_buffer[symbol] = [
                    s for s in buffer
                    if (now - float(s.get("ts", now))) <= max_age
                ]
                
                removed = initial_count - len(self.signal_consensus_buffer[symbol])
                total_removed += removed
                
                if removed > 0:
                    self.logger.debug(
                        "[SignalBuffer:CLEANUP] Removed %d expired signals for %s",
                        removed, symbol
                    )
            
            if total_removed > 0:
                self.logger.info("[SignalBuffer:CLEANUP] Total expired signals removed: %d", total_removed)
        except Exception as e:
            self.logger.warning("[SignalBuffer:CLEANUP] Error in cleanup: %s", e)
    
    def get_buffer_stats_snapshot(self) -> Dict[str, Any]:
        """Get current buffer statistics."""
        return {
            **self.signal_buffer_stats,
            "buffer_size": {
                str(sym): len(sigs)
                for sym, sigs in self.signal_consensus_buffer.items()
            },
            "timestamp": time.time(),
        }

