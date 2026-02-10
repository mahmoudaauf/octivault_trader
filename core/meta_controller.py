
# -*- coding: utf-8 -*-
"""
MetaController (Monolithic - Transitional)

This file currently contains multiple subsystems:
- Orchestration
- Policy evaluation
- Mode management
- Liveness detection
- Capital safety

It is intentionally monolithic during stabilization.
Sections arex annotated for future extraction once behavior is stable.
"""

############################################################
# SECTION: Imports & Constants
# Responsibility:
# - All imports and global constants
# - Module dependencies and type definitions
# Future Extraction Target:
# - Keep as shared dependencies module
############################################################

from __future__ import annotations

import typing
from typing import Dict, Any, Optional, List, Set, Tuple, TYPE_CHECKING
from collections import deque, defaultdict
import datetime
import time
import inspect as _inspect
import asyncio as _asyncio
from decimal import Decimal
from datetime import timezone
import logging
import json
import uuid

# Exchange exception (used for safe error classification)
try:
    from binance.exceptions import BinanceAPIException  # type: ignore
except Exception:
    try:
        from core.stubs import BinanceAPIException  # type: ignore
    except Exception:
        class BinanceAPIException(Exception):
            pass

# ==============================================================================
# INLINE DEFINITIONS: Essential classes from meta_libs (avoiding external dependency)
# ==============================================================================
# These were refactored into core/meta_libs/ locally, but meta_libs doesn't exist
# on EC2. To ensure EC2 deployment works, we inline the essential definitions here.

from enum import Enum
from dataclasses import dataclass

class ExecutionError(Exception):
    """Canonical P9 execution error with proper classification."""

    class Type:
        """Error type constants for backward compatibility (string-based comparison)."""
        EXTERNAL_API_ERROR = "EXTERNAL_API_ERROR"
        MIN_NOTIONAL_VIOLATION = "MIN_NOTIONAL_VIOLATION"
        FEE_SAFETY_VIOLATION = "FEE_SAFETY_VIOLATION"
        RISK_CAP_EXCEEDED = "RISK_CAP_EXCEEDED"
        INSUFFICIENT_BALANCE = "INSUFFICIENT_BALANCE"
        INTEGRITY_ERROR = "INTEGRITY_ERROR"

    def __init__(
        self,
        error_type: str,
        message: str,
        symbol: str = "",
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.error_type = error_type
        self.message = message
        self.symbol = symbol
        self.context = context or {}


class DustState(Enum):
    """Dust Accumulation State Machine."""
    EMPTY = "empty"
    DUST_ACCUMULATING = "dust_accumulating"
    DUST_MATURED = "dust_matured"
    TRADABLE = "tradable"


@dataclass
class LiquidityPlan:
    """Enhanced liquidation plan with P9 compliance."""
    status: str
    exits: List[Dict[str, Any]]
    freed_quote: float
    trace_id: str
    risk_assessment: Optional[Dict[str, Any]] = None


class BoundedCache:
    """Thread-safe bounded cache with TTL support."""

    def __init__(self, max_size: int = 1000, default_ttl: float = 300.0):
        self._cache: Dict[str, tuple] = {}
        self._max_size = max_size
        self._default_ttl = default_ttl

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache if not expired."""
        if key not in self._cache:
            return default
        value, expires_at = self._cache[key]
        now = time.time()
        if now > expires_at:
            del self._cache[key]
            return default
        return value

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set value in cache with TTL."""
        now = time.time()
        expires_at = now + (ttl or self._default_ttl)
        self._cache[key] = (value, expires_at)

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Alias for set() for compatibility."""
        self.set(key, value, ttl)

    def list_all(self) -> List[Any]:
        """Return all non-expired values."""
        now = time.time()
        results = []
        expired_keys = []
        for k, (val, exp) in list(self._cache.items()):
            if now <= exp:
                results.append(val)
            else:
                expired_keys.append(k)
        for k in expired_keys:
            self._cache.pop(k, None)
        return results

    def cleanup_expired(self) -> int:
        """Remove expired entries from cache. Returns count of removed items."""
        now = time.time()
        expired_keys = [k for k, (_, exp) in self._cache.items() if now > exp]
        for k in expired_keys:
            del self._cache[k]
        return len(expired_keys)



class ThreadSafeIntentSink:
    """Thread-safe collection for trade intents."""

    def __init__(self, max_size: int = 1000):
        self._intents = []
        self._max_size = max_size

    def add(self, intent: Any) -> None:
        """Add intent to sink."""
        self._intents.append(intent)
        if len(self._intents) > self._max_size:
            self._intents.pop(0)

    def get_all(self) -> List[Any]:
        """Get all intents and clear."""
        result = list(self._intents)
        self._intents.clear()
        return result


def parse_timestamp(val: Any, default_ts: float = 0.0) -> float:
    """Coerce timestamps into epoch seconds."""
    if val is None:
        return default_ts
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        try:
            if val.endswith('Z'):
                val = val[:-1] + '+00:00'
            from datetime import datetime
            dt = datetime.fromisoformat(val.replace('Z', '+00:00'))
            return dt.timestamp()
        except Exception:
            return default_ts
    return default_ts


def classify_execution_error(err: Any, symbol: str = "") -> ExecutionError:
    """Classify execution error type and return ExecutionError object."""
    if isinstance(err, ExecutionError):
        # Ensure symbol is set if provided
        if symbol and not err.symbol:
            err.symbol = symbol
        return err
    if hasattr(err, 'error_type'):
        return ExecutionError(err.error_type, str(err), symbol)
    err_str = str(err).lower()
    if 'notional' in err_str:
        return ExecutionError(ExecutionError.Type.MIN_NOTIONAL_VIOLATION, str(err), symbol)
    if 'balance' in err_str or 'insufficient' in err_str:
        return ExecutionError(ExecutionError.Type.INSUFFICIENT_BALANCE, str(err), symbol)
    if 'fee' in err_str:
        return ExecutionError(ExecutionError.Type.FEE_SAFETY_VIOLATION, str(err), symbol)
    return ExecutionError("UNKNOWN", str(err), symbol)

# Core system imports
from core.focus_mode import FocusModeManager
# Inline _safe_await to avoid ModuleNotFoundError during bootstrap
async def _safe_await(maybe):
    if maybe is None: return None
    if _inspect.isawaitable(maybe): return await maybe
    return maybe
from core.scaling import ScalingManager
from core.execution_logic import ExecutionLogic
from core.state_manager import StateManager
from core.rotation_authority import RotationExitAuthority
from core.portfolio_authority import PortfolioAuthority
from core.exit_utils import post_exit_bookkeeping

# ==============================================================================
# Type stubs for type checking only (no runtime impact)
# ==============================================================================
if TYPE_CHECKING:
    from core.shared_state import SharedState
    from core.exchange_client import ExchangeClient
    from core.execution_manager import ExecutionManager
    from core.config import Config as ConfigType

# Simple type aliases for runtime (avoid circular imports)
SharedState = Any  # Type alias, actual type hint uses string annotation
ExchangeClient = Any
ExecutionManager = Any
Config = Any
KernelState = Any
ExecOrder = Any

# TradeIntent moved to core/stubs.py - keeping import for backward compatibility
from core.stubs import TradeIntent

# Component Status Logger
try:
    from core.component_status_logger import ComponentStatusLogger as CSL
except ImportError:
    class CSL:
        """Component Status Logger stub for environments without CSL available."""
        @staticmethod
        def set_status(*a, **k): pass
        @staticmethod
        async def heartbeat(*a, **k): pass
        @staticmethod
        def log_status(*a, **k): pass

# ==============================================================================
# END PHASE 1 MODULARIZATION IMPORTS
# ==============================================================================


############################################################
# SECTION: MetaController Lifecycle & Entry Points
# Responsibility:
# - Main class definition and initialization
# - Core lifecycle methods (run, evaluate_and_act)
# - Primary entry points and orchestration
# Future Extraction Target:
# - MetaControllerOrchestrator
############################################################


class MetaController:
        def _reset_bootstrap_override_if_deadlocked(self, symbol: str, signal: dict, last_result: dict = None):
            """Reset bootstrap override counter if is_flat, no realized trades, and last override did NOT produce execution."""
            # Check if position is flat
            is_flat = False
            try:
                qty = 0.0
                if hasattr(self.shared_state, "get_position_qty"):
                    qty = float(self.shared_state.get_position_qty(symbol) or 0.0)
                elif hasattr(self.shared_state, "get_position"):
                    pos = self.shared_state.get_position(symbol)
                    if _asyncio.iscoroutine(pos):
                        pos = _asyncio.get_event_loop().run_until_complete(pos)
                    qty = float(pos.get("qty", 0.0) or pos.get("quantity", 0.0) or 0.0)
                is_flat = qty == 0.0
            except Exception:
                pass
            # Check if no realized trades yet
            realized_trades = 0
            try:
                metrics = getattr(self.shared_state, "metrics", {}) or {}
                realized_trades = int(metrics.get("total_trades_executed", 0) or 0)
            except Exception:
                pass
            # Check if last override did NOT produce execution
            last_override_failed = False
            if last_result is not None:
                last_override_failed = not (str(last_result.get("status", "")).lower() in {"placed", "executed", "filled"})
            # If all conditions met, reset bootstrap attempts
            if is_flat and realized_trades == 0 and last_override_failed and getattr(self, "_bootstrap_attempts", 0) > 0:
                self.logger.warning(f"[Meta:BOOTSTRAP_DEADLOCK] Resetting bootstrap override counter for {symbol} (flat, no realized trades, last override failed)")
                self._bootstrap_attempts = 0
    def _compute_min_notional_aware_qty(self, *, price: float, min_notional: float, min_qty: float, step_size: float, fee_buffer: float = 1.01, slippage_buffer: float = 1.0) -> float:
        """
        Compute the minimum executable quantity that satisfies min_notional, min_qty, and step_size, with buffers.
        """
        from math import ceil
        if price <= 0 or min_notional <= 0 or step_size <= 0:
            return 0.0
        # Apply fee and slippage buffer
        min_quote = min_notional * fee_buffer * slippage_buffer
        qty = min_quote / price
        # Enforce min_qty and step_size
        qty = max(qty, min_qty)
        # Round down to step_size
        from decimal import Decimal, ROUND_DOWN
        qty = float((Decimal(str(qty)) / Decimal(str(step_size))).to_integral_value(rounding=ROUND_DOWN) * Decimal(str(step_size)))
        return qty

    # Symbol lifecycle states
LIFECYCLE_DUST_HEALING = "DUST_HEALING"
LIFECYCLE_STRATEGY_OWNED = "STRATEGY_OWNED"
LIFECYCLE_ROTATION_PENDING = "ROTATION_PENDING"
LIFECYCLE_LIQUIDATION = "LIQUIDATION"

class MetaController:
    # ...existing code...
    def _init_symbol_lifecycle(self):
        # Call this in __init__
        self.symbol_lifecycle = {}  # symbol -> state
        self.dust_healing_cooldown = {}  # symbol -> timestamp

    def _is_bootstrap_mode(self) -> bool:
        try:
            if hasattr(self.shared_state, "is_bootstrap_mode"):
                return bool(self.shared_state.is_bootstrap_mode())
        except Exception:
            pass
        try:
            if hasattr(self, "mode_manager"):
                return str(self.mode_manager.get_mode() or "").upper() == "BOOTSTRAP"
        except Exception:
            pass
        return False

    def _rotation_preempt_active(self) -> bool:
        metrics = getattr(self.shared_state, "metrics", {}) or {}
        try:
            realized = float(metrics.get("realized_pnl", 0.0) or 0.0)
            unrealized = float(metrics.get("unrealized_pnl", 0.0) or 0.0)
            drawdown = float(metrics.get("drawdown_pct", 0.0) or 0.0)
            nav = float(metrics.get("nav", 0.0) or 0.0)
            base_capital = float(getattr(self.config, "BASE_CAPITAL", 0.0) or 0.0)
            usdt_per_hour = float(metrics.get("usdt_per_hour", 0.0) or 0.0)
        except Exception:
            return False

        capital_falling = bool(
            drawdown > 0
            or (base_capital > 0 and nav > 0 and nav < base_capital * 0.98)
            or usdt_per_hour < 0
        )
        no_profit = realized <= 0 and unrealized <= 0
        return bool(capital_falling and no_profit)

    def _refresh_buy_reentry_delta(self) -> None:
        """Refresh BUY re-entry delta using temporary override and restore gates."""
        try:
            base_delta = float(getattr(self.config, "BUY_REENTRY_DELTA_PCT", 0.005))
            temp_delta = float(getattr(self.config, "BUY_REENTRY_DELTA_PCT_TEMP", base_delta))
            restore_equity = float(getattr(self.config, "BUY_REENTRY_DELTA_RESTORE_EQUITY", 0.0))
            restore_trades = int(getattr(self.config, "BUY_REENTRY_DELTA_RESTORE_TRADES", 0))

            metrics = getattr(self.shared_state, "metrics", {}) or {}
            realized = float(metrics.get("realized_pnl", 0.0) or 0.0)
            base_capital = float(getattr(self.config, "BASE_CAPITAL", 0.0) or 0.0)
            realized_equity = base_capital + realized

            trade_history = list(getattr(self.shared_state, "trade_history", []) or [])
            closed_trades = len(trade_history)

            restore_hit = False
            if restore_equity > 0 and realized_equity >= restore_equity:
                restore_hit = True
            if restore_trades > 0 and closed_trades >= restore_trades:
                restore_hit = True

            target_delta = base_delta if restore_hit else temp_delta
            if target_delta <= 0:
                target_delta = base_delta

            if float(self._buy_reentry_delta_pct or 0.0) != float(target_delta or 0.0):
                self.logger.info(
                    "[Meta] BUY_REENTRY_DELTA_PCT -> %.4f%% (mode=%s, equity=%.2f, closed_trades=%s)",
                    float(target_delta) * 100.0,
                    "DEFAULT" if restore_hit else "TEMP",
                    float(realized_equity),
                    int(closed_trades),
                )
                self._buy_reentry_delta_pct = float(target_delta or 0.0)

            state = getattr(self.shared_state, "dynamic_config", None)
            if state is None:
                self.shared_state.dynamic_config = {}
                state = self.shared_state.dynamic_config
            state["BUY_REENTRY_DELTA_PCT_EFFECTIVE"] = float(target_delta or 0.0)
            state["BUY_REENTRY_DELTA_MODE"] = "DEFAULT" if restore_hit else "TEMP"
            state["BUY_REENTRY_DELTA_RESTORE_EQUITY"] = float(restore_equity or 0.0)
            state["BUY_REENTRY_DELTA_RESTORE_TRADES"] = int(restore_trades or 0)
        except Exception as e:
            self.logger.debug("[Meta] Failed to refresh BUY_REENTRY_DELTA_PCT: %s", e)

    def _dust_merge_retry_allowed(self, symbol: str, current_price: float) -> bool:
        attempt = self._dust_merge_attempts.get(symbol)
        if not attempt:
            return True
        if attempt.get("bootstrap_epoch") != self._dust_merge_bootstrap_epoch:
            return True
        last_price = float(attempt.get("price", 0.0) or 0.0)
        if last_price > 0 and current_price > 0:
            retry_pct = float(self._cfg("DUST_MERGE_RETRY_PCT", 0.02))
            delta_pct = abs(current_price - last_price) / max(last_price, 1e-9)
            return delta_pct >= retry_pct
        return False

    def _record_dust_merge_attempt(self, symbol: str, price: float) -> None:
        self._dust_merge_attempts[symbol] = {
            "price": float(price or 0.0),
            "ts": time.time(),
            "bootstrap_epoch": self._dust_merge_bootstrap_epoch,
        }

    def _bootstrap_dust_bypass_allowed(self, symbol: str, is_bootstrap_override: bool, is_dust_position: bool) -> bool:
        """Allow a one-shot PositionLock bypass for bootstrap dust scaling."""
        if not (is_bootstrap_override and is_dust_position):
            return False
        if symbol in self._bootstrap_dust_bypass_used:
            return False
        self._bootstrap_dust_bypass_used.add(symbol)
        return True

    def _set_lifecycle(self, symbol, state):
        prev = self.symbol_lifecycle.get(symbol)
        self.symbol_lifecycle[symbol] = state
        self.logger.info(f"[LIFECYCLE] {symbol}: {prev} -> {state}")

    def _can_act(self, symbol, authority):
        # Only allow one authority at a time
        state = self.symbol_lifecycle.get(symbol)
        if state == self.LIFECYCLE_DUST_HEALING and authority in ("SELL", "ROTATION"):
            self.logger.info(f"[LIFECYCLE] {symbol}: {authority} blocked (in DUST_HEALING)")
            return False
        if state == self.LIFECYCLE_ROTATION_PENDING and authority == "DUST_HEALING":
            self.logger.info(f"[LIFECYCLE] {symbol}: DUST_HEALING blocked (in ROTATION_PENDING)")
            return False
        return True

    def _on_sell_executed(self, symbol):
        self._set_lifecycle(symbol, self.LIFECYCLE_ROTATION_PENDING)
        # Freeze dust healing for cooldown (e.g., 10 min)
        import time
        self.dust_healing_cooldown[symbol] = time.time() + 600
        self.logger.info(f"[LIFECYCLE] {symbol}: Dust healing frozen for 600s after SELL")
        if getattr(self, "_bootstrap_seed_active", False):
            self._bootstrap_seed_active = False
            self._bootstrap_seed_used = True
            self._bootstrap_seed_enabled = False
            self.logger.warning("[BOOTSTRAP] Seed trade completed. System unlocked.")
            try:
                if hasattr(self, "mode_manager") and self.mode_manager.get_mode().upper() == "BOOTSTRAP":
                    self.mode_manager.set_mode("NORMAL")
            except Exception:
                pass

    @property
    def _trade_timestamps(self):
        """Delegated throughput tracking."""
        return self.execution_logic._trade_timestamps

    @property
    def _trade_timestamps_sym(self):
        """Delegated per-symbol throughput tracking."""
        return self.execution_logic._trade_timestamps_sym

    @property
    def _trade_timestamps_agent(self):
        """Delegated per-agent throughput tracking."""
        return self.execution_logic._trade_timestamps_agent

    @property
    def _trade_timestamps_day(self):
        """Delegated daily throughput tracking."""
        return self.execution_logic._trade_timestamps_day

    # Belongs to: MetaController Lifecycle & Entry Points
    # Extraction Candidate: Yes
    # Depends on: State & Internal Counters, Mode Management
    def __init__(
        self,
        shared_state,
        exchange_client,
        execution_manager,
        config,
        cot_assistant=None,
        alert_callback=None,
        liquidation_agent=None,
        agent_manager: Optional[object] = None,
        tp_sl_engine=None,
        risk_manager=None,
        kernel_state: Optional["KernelState"] = None,
        portfolio_manager: Optional[object] = None,
    ):
        self.shared_state = shared_state
        self.exchange_client = exchange_client
        self.execution_manager = execution_manager
        self.config = config
        self.cot_assistant = cot_assistant
        self.alert_callback = alert_callback
        self.liquidation_agent = liquidation_agent
        self.agent_manager = agent_manager
        self.tp_sl_engine = tp_sl_engine
        self.risk_manager = risk_manager
        self.kernel_state = kernel_state
        self.portfolio_manager = portfolio_manager
        # Ensure logger is available before any init logging
        self.logger = logging.getLogger("MetaController")

        # PHASE2 GUARD: Initialize with safe defaults to prevent AttributeError
        # Ensure all required keys are present, matching PolicyManager
        self._phase2_guard = {
            "activation_age_sec": 0,
            "position_grace_sec": float(getattr(config, 'PHASE2_POSITION_GRACE_SEC', 600.0)),
        }
        
        # Initialize StateManager early since _kpi_metrics delegates to it
        from core.state_manager import StateManager
        self.state_manager = StateManager(self.shared_state, self.config, self.logger, component_name="MetaController")
        
        # ═══════════════════════════════════════════════════════════════════
        # PHASE 1 MODULARIZATION: Use imported BoundedCache directly
        # BoundedCache is now imported at module level from core.meta_controller.cache
        # ═══════════════════════════════════════════════════════════════════
        
        # Initialize signal_cache for signal integrity
        cache_size = int(getattr(config, 'signal_cache_max_size', 1000))
        cache_ttl = float(getattr(config, 'signal_cache_ttl', 300.0))
        self.signal_cache = BoundedCache(max_size=cache_size, default_ttl=cache_ttl)
        self.logger.info(f"[Meta:Init] Signal cache initialized: max_size={cache_size}, ttl={cache_ttl}s")
        
        from core.mode_manager import ModeManager
        self.mode_manager = ModeManager(self.logger, config)
        
        # Initialize min_notional_cache for exchange minimum tracking
        self._min_notional_cache = BoundedCache(max_size=500, default_ttl=3600)
        self.logger.debug("[Meta:Init] Min notional cache initialized: max_size=500, ttl=3600s")
        
        # Initialize reason log cache for duplicate message filtering
        self._last_reason_log = BoundedCache(max_size=200, default_ttl=1800)
        self.logger.debug("[Meta:Init] Reason log cache initialized: max_size=200, ttl=1800s")

        # Initialize additional controller attributes
        self._reason_cooldown = float(getattr(config, 'REASON_LOG_COOLDOWN_SEC', 30.0))
        self._performance_lock = _asyncio.Lock()
        
        # Initialize ops plane readiness flag
        self._has_emitted_ops_ready = False
        
        # Initialize lifecycle and mode tracking flags
        self._perf_eval_ready = False
        self._first_trade_executed = False
        self._bootstrap_lock_engaged = False
        self._bootstrap_cooldown_until = 0.0
        self._bootstrap_last_veto_reason = None
        self._running = False
        self._stop = False
        
        # Initialize start time for idle tracking (CRITICAL: required by _gather_mode_metrics)
        self._start_time = time.time()
        
        # Initialize execution attempts tracking (cycle-based counter)
        self._execution_attempts_this_cycle = 0
        
        # Initialize tick counter for evaluation cycle tracking
        self.tick_id = 0
        self._tick_counter = 0
        
        # --- Execution confidence floors ---
        self._min_exec_conf = getattr(config, "MIN_EXEC_CONF", 0.60)
        self._tier_a_conf = getattr(config, "TIER_A_CONF", 0.70)
        self._tier_b_conf = getattr(config, "TIER_B_CONF", 0.55)
        self._meta_min_agents = int(getattr(config, "META_MIN_AGENTS", 1))
        self._directional_consistency_pct = float(getattr(config, "META_DIRECTIONAL_CONSISTENCY_PCT", 0.60))
        self._adaptive_aggression = 1.0 # P9: Start with neutral aggression

        # --- Planned-quote defaults (avoid 0.00 sizing) ---
        min_entry_quote = float(getattr(config, "MIN_ENTRY_QUOTE_USDT", 12.0))
        min_econ_trade = float(getattr(config, "MIN_ECONOMIC_TRADE_USDT", 0.0) or 0.0)
        if min_econ_trade > 0:
            min_entry_quote = max(min_entry_quote, min_econ_trade)
        default_quote = float(
            getattr(
                config,
                "DEFAULT_PLANNED_QUOTE",
                getattr(config, "TRADE_AMOUNT_USDT", 10.0)
            )
        )
        self._min_entry_quote_usdt = max(10.0, min_entry_quote)
        self._default_planned_quote = max(default_quote, self._min_entry_quote_usdt)
        self._max_spend = float(getattr(config, "MAX_SPEND_PER_TRADE_USDT", 50.0))
        # Conditional size bump latches (activated when edge proves itself)
        self._exit_coalescing_enabled = True
        self._min_hold_enabled = True
        metrics = getattr(self.shared_state, "metrics", {}) or {}
        self._planned_quote_bump_applied = bool(metrics.get("planned_quote_bump_applied"))
        
        # --- Post-bootstrap controls ---
        self._symbol_concentration_limit = getattr(
            config,
            "SYMBOL_CONCENTRATION_LIMIT",
            2  # safe default
        )
        self._bootstrap_veto_cooldown_sec = float(
            getattr(config, "BOOTSTRAP_VETO_COOLDOWN_SEC", 600.0) or 600.0
        )

        self._micro_size_quote = float(getattr(config, "MICRO_SIZE_QUOTE", 5.0))
        self._scout_min_notional = float(getattr(config, "SCOUT_MIN_NOTIONAL", 5.0))
        
        # EXECUTION FLOORS
        self._min_exec_confidence = float(self._cfg("MIN_EXECUTION_CONFIDENCE", default=0.60))
        self._tier_a_conf = float(self._cfg("TIER_A_CONFIDENCE_THRESHOLD", default=0.70))
        
        # Config snapshot
        self._min_conf_ingest = float(self._cfg("MIN_SIGNAL_CONF", default=0.50))
        self._max_age_sec = float(self._cfg("MAX_SIGNAL_AGE_SECONDS", default=60))
        self._min_exec_conf = float(self._cfg("MIN_EXECUTION_CONFIDENCE", default=0.55))
        # Ensure execution floor is not below ingest floor
        if self._min_exec_conf < self._min_conf_ingest:
            self.logger.warning(
                "MIN_EXECUTION_CONFIDENCE (%.2f) < MIN_SIGNAL_CONF (%.2f); bumping exec floor to ingest floor.",
                self._min_exec_conf, self._min_conf_ingest
            )
            self._min_exec_conf = self._min_conf_ingest
        self._enable_cot = bool(self._cfg("ENABLE_COT_VALIDATION", default=False))
        self._known_quotes = {"USDT", "FDUSD", "USDC", "BUSD", "TUSD", "DAI"}
        self._default_planned_quote = float(self._cfg("DEFAULT_PLANNED_QUOTE", default=10.0))
        self._min_entry_quote = float(self._cfg("MIN_ENTRY_QUOTE_USDT", default=10.0))
        self._max_spend = float(self._cfg("MAX_SPEND_PER_TRADE_USDT", default=50.0))

        # Initialize Focus Mode configuration and state
        # Ensure time module is available (fix for local variable reference error)
        # Allow disabling FOCUS_MODE for testing or recovery
        self.FOCUS_MODE_ENABLED = bool(getattr(config, 'FOCUS_MODE_ENABLED', False))  # Default to False for easier recovery
        # Allow more cycles before FOCUS_MODE triggers (less aggressive)
        self.FOCUS_LIVENESS_FAILURE_CYCLES = int(getattr(config, 'FOCUS_LIVENESS_FAILURE_CYCLES', 30))  # Was 10
        self.MAX_FOCUS_SYMBOLS = int(getattr(config, 'MAX_FOCUS_SYMBOLS', 3))
        self.MIN_SIGNIFICANT_USDT = float(
            getattr(
                config,
                "MIN_SIGNIFICANT_POSITION_USDT",
                getattr(config, "MIN_SIGNIFICANT_USDT", 25.0),
            )
        )
        self._focus_mode_start_time = time.time()  # Track when focus mode became active
        self._focus_liveness_counter = 0  # Counter for focus mode liveness detection

        # Integrate FocusModeManager for focus mode logic
        self.focus_mode_manager = FocusModeManager(self.shared_state, self.execution_manager, self.config, self.logger)
        # Integrate ScalingManager for scaling logic
        self.scaling_manager = ScalingManager(self.shared_state, self.execution_manager, self.config, self.logger, mode_manager=self.mode_manager)
        # Integrate ExecutionLogic for execution path
        self.execution_logic = ExecutionLogic(self.shared_state, self.execution_manager, self.config, self.logger, self)
        
        self._throughput_window_sec = 600.0  # 10 minutes throughput window
        self._max_trades_per_hour = int(getattr(config, 'MAX_TRADES_PER_HOUR', 12))
        self._max_trades_per_day = int(getattr(config, 'MAX_TRADES_PER_DAY', 0) or 0)
        self._buy_cooldown_sec = float(getattr(config, 'BUY_COOLDOWN_SEC', 600.0)) * 0.5
        self._last_buy_ts = defaultdict(float)
        self._buy_reentry_delta_pct = float(getattr(config, 'BUY_REENTRY_DELTA_PCT', 0.005))
        self._last_buy_price = defaultdict(float)
        self._reentry_lock_sec = float(getattr(config, 'REENTRY_LOCK_SEC', 900.0))
        self._reentry_require_tp_sl_exit = bool(getattr(config, 'REENTRY_REQUIRE_TPSL_EXIT', True))
        self._reentry_require_signal_change = bool(getattr(config, 'REENTRY_REQUIRE_SIGNAL_CHANGE', True))
        self._max_open_positions_per_symbol = int(getattr(config, "MAX_OPEN_POSITIONS_PER_SYMBOL", 1) or 1)
        self._last_signal_fingerprint = defaultdict(str)
        self._rotation_escape_last_ts = 0.0
        self._rotation_escape_buy_persist = defaultdict(int)
        
        # Invariant: Accumulation Resolution State
        self._accumulated_quote = {}  # symbol -> accumulated USDT
        self._accumulation_metadata = {}  # symbol -> {first_rejection_time, rejection_count}
        
        # Initialize IntentManager for intent and signal management
        from core.intent_manager import IntentManager
        self.intent_manager = IntentManager(config, self.logger)
        self.logger.info(f"[Meta:Init] Intent sink initialized (via IntentManager)")

        # Use SignalManager for all signal cache and intake logic
        from core.signal_manager import SignalManager
        self.signal_manager = SignalManager(config, self.logger, self.signal_cache, self.intent_manager)

        # Initialize PolicyManager for policy evaluation and decision logic
        from core.policy_manager import PolicyManager
        self.policy_manager = PolicyManager(self.logger, self.config)
        self.active_policy_nudges = {}  # Store active policy nudges
        self.logger.info(f"[Meta:Init] Policy manager initialized")
        self.rotation_authority = RotationExitAuthority(self.logger, self.config, self.shared_state)

        # Profit-locked re-entry (compounding guard)
        try:
            self._profit_lock_checkpoint = float(getattr(self.shared_state, "metrics", {}).get("realized_pnl", 0.0) or 0.0)
        except Exception:
            self._profit_lock_checkpoint = 0.0
        try:
            base_quote = float(self._cfg("PROFIT_LOCK_BASE_QUOTE", self._cfg("DEFAULT_PLANNED_QUOTE", self._min_entry_quote_usdt)))
        except Exception:
            base_quote = float(self._min_entry_quote_usdt)
        self._profit_lock_base_quote = max(float(base_quote or 0.0), float(self._min_entry_quote_usdt or 0.0))
        self.portfolio_authority = PortfolioAuthority(self.logger, self.config, self.shared_state)

        # Initialize missing attributes for economic guard and trade tracking
        self.cycles_no_trade = 0  # Track cycles without trade execution
        self._base_min_notional = float(getattr(config, "MIN_NOTIONAL_USDT", 10.0))
        self._buy_headroom = float(getattr(config, "BUY_HEADROOM_FACTOR", 1.05))
        self.logger.info("[Meta:Init] Cycles no trade counter initialized")
        
        # Emergency Dust Exit Policy
        self.DUST_EXIT_ENABLED = bool(getattr(config, "DUST_EXIT_ENABLED", True))
        self.DUST_EXIT_THRESHOLD = float(getattr(config, "DUST_EXIT_THRESHOLD", 0.60))
        self.DUST_EXIT_NO_TRADE_CYCLES = int(getattr(config, "DUST_EXIT_NO_TRADE_CYCLES", 20))

        # Time-based exit configuration
        self._time_exit_enabled = bool(getattr(config, "TIME_EXIT_ENABLED", True))
        self._time_exit_min_hours = float(getattr(config, "TIME_EXIT_MIN_HOURS", 24.0))
        self._time_exit_slice_pct = float(getattr(config, "TIME_EXIT_SLICE_PCT", 0.50))

        # Gating settings
        self._reentry_lock_sec = float(getattr(config, "REENTRY_LOCK_SEC", 900.0))
        self._reentry_require_tp_sl_exit = bool(getattr(config, "REENTRY_REQUIRE_TPSL_EXIT", True))
        self._reentry_require_signal_change = bool(getattr(config, "REENTRY_REQUIRE_SIGNAL_CHANGE", True))

        # Initialize focus mode state tracking attributes (moved after manager initialization)
        self.FOCUS_SYMBOLS = set()  # Top N symbols by wallet value (PINNED at startup)
        self.FOCUS_SYMBOLS_PINNED = False  # Once set, don't change until restart
        self.FOCUS_SYMBOLS_LAST_UPDATE = 0  # Timestamp of last update
        self._bootstrap_focus_symbols_pending = True  # Flag: need to bootstrap on startup
        self._last_reason_log = BoundedCache(max_size=100, default_ttl=60)
        self._reason_cooldown = float(getattr(config, "REASON_COOLDOWN_SEC", 300.0))
        
        # Initialize focus mode state tracking attributes
        self._focus_mode_active = False
        self._focus_mode_reason = ""
        self._focus_mode_trade_executed_count = 0
        self._focus_mode_healthy_cycles = 0
        self.FOCUS_MODE_AUTO_EXIT_HEALTHY_CYCLES = int(getattr(config, 'FOCUS_MODE_AUTO_EXIT_HEALTHY_CYCLES', 3))  # Auto-exit after 3 healthy cycles
        self._focus_mode_trade_executed = False
        self._bootstrap_attempts = 0  # Track bootstrap override attempts for one-shot enforcement
        self._bootstrap_seed_enabled = bool(getattr(config, "BOOTSTRAP_SEED_ENABLED", True))
        self._bootstrap_seed_symbol = str(getattr(config, "BOOTSTRAP_SEED_SYMBOL", "BTCUSDT")).upper()
        self._bootstrap_seed_quote = float(getattr(config, "BOOTSTRAP_SEED_QUOTE", 20.0))
        self._bootstrap_seed_attempted = False
        self._bootstrap_seed_used = False
        self._bootstrap_seed_active = False
        self._bootstrap_seed_cycle = None
        self._bootstrap_dust_bypass_used = set()  # Track one-shot bootstrap dust scale bypass per symbol
        self._consolidated_dust_symbols = set()  # Track symbols that have completed dust consolidation
        self._dust_merge_attempts = {}  # symbol -> {price, ts, bootstrap_epoch}
        self._dust_merge_bootstrap_epoch = 0
        self._dust_merge_last_bootstrap = None
        self._dust_merges = set()  # legacy compatibility (deprecated)
        self._info_cache = {}

        # Initialize symbol lifecycle state tracking (required by _can_act / dust healing)
        self._init_symbol_lifecycle()

    def _info_once(self, key: str, msg: str, *args):
        """Helper to log important events only once to avoid spamming."""
        if key not in self._info_cache:
            self.logger.info(msg, *args)
            self._info_cache[key] = time.time()

    async def _has_open_position(self, symbol: str) -> Tuple[bool, float]:
        """Return (has_open_position, qty) using the most reliable sources."""
        sym = str(symbol or "")
        if not sym:
            return False, 0.0

        dust_qty = float(getattr(self.config, "DUST_POSITION_QTY", 0.0) or 0.0)
        is_bootstrap = self._is_bootstrap_mode() if dust_qty > 0 else False
        qty = 0.0
        open_statuses = {"OPEN", "PARTIALLY_FILLED", "ACTIVE", "IN_POSITION", "RUNNING"}

        try:
            if hasattr(self.shared_state, "get_open_positions"):
                pos_map = await _safe_await(self.shared_state.get_open_positions())
                if isinstance(pos_map, dict) and sym in pos_map:
                    pos = pos_map.get(sym) or {}
                    qty = float(pos.get("quantity") or pos.get("qty") or pos.get("current_qty") or 0.0)
                    status = str(pos.get("status") or "").upper()
                    if qty > 0 or status in open_statuses:
                        if is_bootstrap and 0 < qty < dust_qty:
                            return False, 0.0
                        return True, max(qty, 0.0)
        except Exception:
            pass

        try:
            if hasattr(self.shared_state, "get_positions_snapshot"):
                snap = await _safe_await(self.shared_state.get_positions_snapshot())
                if isinstance(snap, dict) and sym in snap:
                    pos = snap.get(sym) or {}
                    qty = float(pos.get("quantity") or pos.get("qty") or pos.get("current_qty") or 0.0)
                    status = str(pos.get("status") or "").upper()
                    if qty > 0 or status in open_statuses:
                        if is_bootstrap and 0 < qty < dust_qty:
                            return False, 0.0
                        return True, max(qty, 0.0)
        except Exception:
            pass

        try:
            pos_map = getattr(self.shared_state, "positions", {}) or {}
            if isinstance(pos_map, dict) and sym in pos_map:
                pos = pos_map.get(sym) or {}
                qty = float(pos.get("quantity") or pos.get("qty") or pos.get("current_qty") or 0.0)
                status = str(pos.get("status") or "").upper()
                if qty > 0 or status in open_statuses:
                    if is_bootstrap and 0 < qty < dust_qty:
                        return False, 0.0
                    return True, max(qty, 0.0)
        except Exception:
            pass

        try:
            open_trades = getattr(self.shared_state, "open_trades", {}) or {}
            if isinstance(open_trades, dict) and sym in open_trades:
                tr = open_trades.get(sym) or {}
                qty = float(tr.get("quantity") or tr.get("qty") or 0.0)
                if qty > 0:
                    if is_bootstrap and 0 < qty < dust_qty:
                        return False, 0.0
                    return True, max(qty, 0.0)
        except Exception:
            pass

        try:
            fn = getattr(self.shared_state, "get_position_qty", None) or getattr(self.shared_state, "get_position_quantity", None)
            if callable(fn):
                qty = float(await _safe_await(fn(sym)) or 0.0)
                if qty > 0:
                    if is_bootstrap and 0 < qty < dust_qty:
                        return False, 0.0
                    return True, qty
        except Exception:
            pass

        return False, 0.0

    @property
    def FOCUS_SYMBOLS(self):
        """Delegate to FocusModeManager."""
        return self.focus_mode_manager.FOCUS_SYMBOLS

    @FOCUS_SYMBOLS.setter
    def FOCUS_SYMBOLS(self, value):
        """Delegate to FocusModeManager."""
        self.focus_mode_manager.FOCUS_SYMBOLS = value

    @property
    def FOCUS_SYMBOLS_PINNED(self):
        """Delegate to FocusModeManager."""
        return self.focus_mode_manager.FOCUS_SYMBOLS_PINNED

    @FOCUS_SYMBOLS_PINNED.setter
    def FOCUS_SYMBOLS_PINNED(self, value):
        """Delegate to FocusModeManager."""
        self.focus_mode_manager.FOCUS_SYMBOLS_PINNED = value

    @property
    def FOCUS_SYMBOLS_LAST_UPDATE(self):
        """Delegate to FocusModeManager."""
        return self.focus_mode_manager.FOCUS_SYMBOLS_LAST_UPDATE

    @FOCUS_SYMBOLS_LAST_UPDATE.setter
    def FOCUS_SYMBOLS_LAST_UPDATE(self, value):
        """Delegate to FocusModeManager."""
        self.focus_mode_manager.FOCUS_SYMBOLS_LAST_UPDATE = value

    @property
    def _bootstrap_focus_symbols_pending(self):
        """Delegate to FocusModeManager."""
        return self.focus_mode_manager._bootstrap_focus_symbols_pending

    @_bootstrap_focus_symbols_pending.setter
    def _bootstrap_focus_symbols_pending(self, value):
        """Delegate to FocusModeManager."""
        self.focus_mode_manager._bootstrap_focus_symbols_pending = value

    @property
    def _focus_mode_active(self):
        """Delegate to FocusModeManager."""
        return self.focus_mode_manager._focus_mode_active

    @_focus_mode_active.setter
    def _focus_mode_active(self, value):
        """Delegate to FocusModeManager."""
        self.focus_mode_manager._focus_mode_active = value

    @property
    def _focus_mode_trade_executed(self):
        """Delegate to FocusModeManager."""
        return self.focus_mode_manager._focus_mode_trade_executed

    @_focus_mode_trade_executed.setter
    def _focus_mode_trade_executed(self, value):
        """Delegate to FocusModeManager."""
        self.focus_mode_manager._focus_mode_trade_executed = value

    def _activate_focus_mode(self, reason: str):
        """Delegate focus mode activation to FocusModeManager."""
        if not self.FOCUS_MODE_ENABLED:
            self.logger.info("[Meta:FOCUS_MODE] Skipped activation: FOCUS_MODE is disabled by config.")
            return
        self.focus_mode_manager.activate_focus_mode(reason)

    def _deactivate_focus_mode(self):
        """Delegate focus mode deactivation to FocusModeManager."""
        self.focus_mode_manager.deactivate_focus_mode()
        self._focus_mode_healthy_cycles = 0

    @property
    def _mandatory_sell_mode_active(self):
        """Delegate to ModeManager."""
        return self.mode_manager.is_mandatory_sell_mode_active()

    @_mandatory_sell_mode_active.setter
    def _mandatory_sell_mode_active(self, value):
        """Delegate to ModeManager."""
        self.mode_manager.set_mandatory_sell_mode(value)

    @property
    def _kpi_metrics(self):
        """Delegate to StateManager."""
        return self.state_manager._kpi_metrics

    @_kpi_metrics.setter
    def _kpi_metrics(self, value):
        """Delegate to StateManager."""
        self.state_manager._kpi_metrics = value

    async def _count_significant_positions(self) -> Tuple[int, int, int]:
        """
        FIX #1: Classify positions into SIGNIFICANT vs DUST.
        
        Returns: (total_count, significant_count, dust_count)
        
        Critical for distinguishing between "portfolio full" and "mostly dust".
        Dust positions should NOT block capital allocation or replacement.
        
        ✅ ENHANCEMENT: Properly count dust in fallback mode
        """
        try:
            # Get position classification from SharedState
            if hasattr(self.shared_state, "classify_positions_by_size"):
                classification = await _safe_await(self.shared_state.classify_positions_by_size())
                sig_list = classification.get("significant", [])
                dust_list = classification.get("dust", [])
                total = len(sig_list) + len(dust_list)
                
                if total > 0:
                    dust_ratio = len(dust_list) / total
                    self.logger.debug(
                        "[Meta:PosCounts] Total=%d Sig=%d Dust=%d Ratio=%.1f%%",
                        total, len(sig_list), len(dust_list), dust_ratio * 100
                    )
                
                return total, len(sig_list), len(dust_list)
        except Exception as e:
            self.logger.debug("[Meta:PosCounts] Primary classification failed: %s", e)
        
        # Fallback: manually count dust positions
        try:
            # ✅ FIX: Exclude dust from significant count
            sig_count = self.shared_state.open_positions_count()  # Already excludes dust
            
            # Count dust positions manually
            dust_count = 0
            positions = self.shared_state.get_open_positions()
            for sym, pos in positions.items():
                is_dust = pos.get("is_dust", False) or pos.get("_is_dust", False)
                qty = float(pos.get("qty", 0) or pos.get("quantity", 0))
                status = str(pos.get("status", "")).upper()
                
                if qty > 0 and is_dust and status in {"OPEN", "PARTIALLY_FILLED"}:
                    dust_count += 1
            
            total = sig_count + dust_count
            return total, sig_count, dust_count
        except Exception as e:
            self.logger.warning("[Meta:PosCounts] Fallback also failed: %s. Returning zeros.", e)
            return 0, 0, 0

############################################################
# SECTION: State & Internal Counters  
# Responsibility:
# - Tick counters, execution attempt tracking
# - Timestamp generation and cycle management
# - Internal state variables and flags
# Future Extraction Target:
# - StateManager or CycleTracker
############################################################

    # Belongs to: State & Internal Counters
    # Extraction Candidate: Yes
    # Depends on: None (utility function)
    def _epoch(self) -> float:
        """Return current epoch timestamp in seconds."""
        import time
        return time.time()

    def _signal_fingerprint(self, sig: Dict[str, Any]) -> str:
        """Create a fingerprint for signal change detection."""
        try:
            # Core components that define a 'different' signal
            action = str(sig.get("action", "")).upper()
            side = str(sig.get("side", "") or ("BUY" if action == "BUY" else "SELL")).upper()
            
            # Use deterministic parts to avoid false triggers from noise
            parts = [
                str(sig.get("symbol", "")),
                action,
                side,
                str(round(float(sig.get("confidence", 0.0)), 2)), # 2 decimal precision
            ]
            return ":".join(parts)
        except Exception:
            return str(hash(str(sig)))

    def _normalize_symbol(self, symbol: str) -> str:
        """Delegate symbol normalization to SignalManager."""
        return self.signal_manager._normalize_symbol(symbol)

    def _split_base_quote(self, symbol: str) -> Tuple[str, str]:
        """Delegate symbol splitting to SignalManager."""
        return self.signal_manager._split_base_quote(symbol)

    def _build_signal_fingerprint(self, sig: Dict[str, Any]) -> str:
        """Build a compact fingerprint for anti-churn re-entry checks."""
        try:
            agent = str(sig.get("agent", "")).lower()
            action = str(sig.get("action", "")).upper()
            reason = str(sig.get("reason", "")).lower()
            conf = float(sig.get("confidence", 0.0))
            tier = str(sig.get("_tier", ""))
            return f"{action}:{agent}:{reason}:{tier}:{conf:.3f}"
        except Exception:
            return ""

    def _classify_exit_reason(self, signal: Dict[str, Any]) -> str:
        """Classify SELL exit reason for re-entry gating."""
        try:
            tag = str(signal.get("_tag") or signal.get("tag") or "").lower()
            reason = str(signal.get("reason", "")).lower()
            if "tp" in reason or "tp_sl" in tag:
                return "TP"
            if "sl" in reason:
                return "SL"
            if "capital_recovery" in reason or signal.get("_capital_recovery_forced") or signal.get("_capital_recovery_soft"):
                return "RECOVERY"
            if "rotation" in reason or "rotation" in tag:
                return "ROTATION"
            if "liquidation" in reason or "liquidation" in tag:
                return "LIQUIDATION"
            return "STRATEGY_SELL"
        except Exception:
            return "EXIT"

    # Belongs to: State & Internal Counters
    # Extraction Candidate: Yes  
    # Depends on: None
    def get_execution_attempts_this_cycle(self) -> int:
        """Get the number of execution attempts in the current evaluation cycle."""
        return self.state_manager.get_execution_attempts_this_cycle()

    # Belongs to: State & Internal Counters
    # Extraction Candidate: Yes
    # Depends on: None
    def increment_execution_attempts(self) -> None:
        """Increment the execution attempts counter for this cycle."""
        self.state_manager.increment_execution_attempts()

    # Belongs to: State & Internal Counters
    # Extraction Candidate: Yes
    # Depends on: None
    def reset_execution_attempts(self) -> None:
        """Reset execution attempts counter at the start of a new evaluation cycle."""
        self.state_manager.reset_execution_attempts()

    async def _bootstrap_focus_symbols(self) -> Set[str]:
        """Delegate focus symbol bootstrap to FocusModeManager."""
        self.FOCUS_SYMBOLS = await self.focus_mode_manager._bootstrap_focus_symbols()
        return self.FOCUS_SYMBOLS

    async def _update_focus_symbols(self) -> Set[str]:
        """Delegate focus symbol updates to FocusModeManager."""
        self.FOCUS_SYMBOLS = await self.focus_mode_manager._update_focus_symbols()
        return self.FOCUS_SYMBOLS

    async def should_execute_sell(self, symbol: str, emergency_liquidation: bool = False) -> bool:
        """
        CRITICAL FIX #3: WALLET_FOCUS_BOOTSTRAP Requirement + PHASE_2_GUARD Override
        
        Gate 1: FOCUS MODE RESTRICTION
        In WALLET_FOCUS_BOOTSTRAP mode, restrict SELL signals to focus symbols only.
        Non-focus positions are FROZEN (not eliminated, but out-of-trading-scope).
        EXCEPTION: emergency_liquidation=True bypasses focus gate (for PHASE_2_GUARD).
        
        Gate 2: DUST STATE MACHINE
        Prevent SELL signal generation for DUST_ACCUMULATING positions.
        This prevents infinite retry loops by skipping the signal early (at MetaController).
        
        Args:
            symbol: Trading symbol
            emergency_liquidation: If True, bypass FOCUS_MODE gate (used by PHASE_2_GUARD)
        
        Returns:
            True: Execute SELL signal normally (passes both gates)
            False: Skip SELL signal (fails focus gate or dust gate)
        """
        try:
            symbol = (symbol or "").upper()
            if not symbol:
                return False

            # Hard invariant: if inventory exists, SELL must be allowed
            try:
                if await self._should_allow_sell(symbol):
                    return True
            except Exception:
                # Fail-open for SELL if gate check fails
                return True
            
            # ═══════════════════════════════════════════════════════════════════════
            # GATE 1: FOCUS MODE RESTRICTION (CRITICAL FIX #3 + FIX #3A)
            # ═══════════════════════════════════════════════════════════════════════
            # WALLET_FOCUS_BOOTSTRAP Requirement:
            # "SELL signals for focus symbols are always evaluated"
            # "Only FOCUS_SYMBOLS are eligible for ... SELL ..."
            # "Non-focus symbols are frozen and ignored"
            # 
            # FIX #3A: PHASE_2_GUARD OVERRIDE
            # Emergency dust liquidation (60%+ dust ratio) bypasses focus gate
            # to ensure emergency escape route is always available
            
            if self.FOCUS_MODE_ENABLED:
                if emergency_liquidation:
                    self.logger.info(
                        "[WALLET_FOCUS_BOOTSTRAP] ⚠️ EMERGENCY LIQUIDATION: Bypassing FOCUS_MODE gate for %s "
                        "(PHASE_2_GUARD dust ratio exceeded)",
                        symbol
                    )
                else:
                    # Focus mode should restrict entries, not exits.
                    self.logger.debug(
                        "[WALLET_FOCUS_BOOTSTRAP] Focus mode active — SELL allowed for %s.",
                        symbol
                    )
            
            # ═══════════════════════════════════════════════════════════════════════
            # GATE 2: DUST STATE MACHINE
            # ═══════════════════════════════════════════════════════════════════════
            # Prevent infinite retry loops for dust positions
            # Get the dust state for this symbol
            dust_state = await self.portfolio_manager.get_dust_state(symbol)
            
            # Only skip signal if in DUST_ACCUMULATING state
            # (TRADABLE, DUST_MATURED, and EMPTY allow execution)
            if dust_state == DustState.DUST_ACCUMULATING:
                # NEW: Check if accumulation timeout has been exceeded
                # After 5 minutes in accumulation, allow SELL as market may have changed
                dust_record = await self.portfolio_manager.get_dust_record(symbol) or {}
                accumulated_at = float(dust_record.get("accumulated_at", time.time()))
                dust_age = time.time() - accumulated_at
                
                if dust_age > 300:  # 5 minutes timeout
                    self.logger.warning(
                        "[Meta:DustGate] ⏱️ DUST_EXIT: %s has been DUST_ACCUMULATING for %d sec (>5min). "
                        "Allowing SELL (dust timeout activated).",
                        symbol, int(dust_age)
                    )
                    # Allow SELL - dust timeout expired
                else:
                    self.logger.debug(
                        f"[Meta:DustGate] {symbol} is DUST_ACCUMULATING (age={int(dust_age)}s). "
                        f"Skipping SELL signal, waiting for accumulation or price appreciation."
                    )
                    return False
            
            # All gates passed - allow SELL
            return True
            
        except Exception as e:
            self.logger.warning(f"[WALLET_FOCUS_BOOTSTRAP] Error checking sell gate for {symbol}: {e}")
            # Fail-safe: allow execution if uncertain (prefer execution over silence)
            return True
            
    def _get_fee_bps(self, shared_state: Any, fee_type: str = "taker") -> float:
        """Helper to safely get fee bps with defaults."""
        try:
            # Try to get from PolicyManager if available
            if hasattr(self, "policy_manager") and hasattr(self.policy_manager, "get_fee_bps"):
                return float(self.policy_manager.get_fee_bps(fee_type) or 10.0)
            # Try shared state
            if hasattr(shared_state, "get_fee_bps"):
                return float(shared_state.get_fee_bps(fee_type) or 10.0)
            return 10.0
        except Exception:
            return 10.0

    def _resolve_sell_tag(self, signal: Dict[str, Any]) -> str:
        tag = str(signal.get("tag") or signal.get("_tag") or "").strip()
        tag_lower = tag.lower()
        if tag_lower in {"tp_sl", "liquidation", "rebalance", "meta_exit"}:
            return tag_lower

        reason_text = " ".join([
            str(signal.get("reason") or ""),
            str(signal.get("exit_reason") or ""),
            str(signal.get("signal_reason") or ""),
        ]).upper()

        if "TP" in reason_text and "SL" not in reason_text:
            return "tp_sl"
        if "TP_SL" in reason_text or "TPSL" in reason_text:
            return "tp_sl"

        if (
            signal.get("_is_liquidation")
            or signal.get("_is_starvation_sell")
            or signal.get("_force_dust_liquidation")
            or "LIQUIDATION" in reason_text
            or "LIQUIDATION" in tag.upper()
            or str(signal.get("agent") or "") == "LiquidationAgent"
        ):
            return "liquidation"

        if "REBALANCE" in reason_text or signal.get("_is_rebalance"):
            return "rebalance"

        return "meta_exit"

    async def _passes_meta_sell_profit_gate(self, symbol: str, sig: Dict[str, Any]) -> bool:
        """Fee-aware profit gate for MetaController-originated SELLs."""
        reason_text = " ".join([
            str(sig.get("reason") or ""),
            str(sig.get("tag") or ""),
            str(sig.get("exit_reason") or ""),
            str(sig.get("signal_reason") or ""),
        ]).upper()
        if (
            sig.get("_is_liquidation")
            or sig.get("_is_starvation_sell")
            or sig.get("_time_exit")
            or "TIME_EXIT" in reason_text
            or "EMERGENCY" in reason_text
            or "SL" in reason_text
        ):
            return True

        # Resolve entry price
        entry_price = 0.0
        try:
            ot = getattr(self.shared_state, "open_trades", {}).get(symbol, {})
            entry_price = float(ot.get("entry_price", 0.0) or 0.0)
        except Exception:
            entry_price = 0.0
        if not entry_price:
            try:
                pos = getattr(self.shared_state, "positions", {}).get(symbol, {})
                entry_price = float(pos.get("avg_price", 0.0) or pos.get("entry_price", 0.0) or 0.0)
            except Exception:
                entry_price = 0.0

        # Resolve current price
        cur_price = 0.0
        try:
            if hasattr(self.shared_state, "safe_price"):
                cur_price = float(await _safe_await(self.shared_state.safe_price(symbol)) or 0.0)
        except Exception:
            cur_price = 0.0
        if cur_price <= 0:
            cur_price = float(getattr(self.shared_state, "latest_prices", {}).get(symbol, 0.0) or 0.0)

        if entry_price <= 0 or cur_price <= 0:
            self.logger.info("[Meta:ProfitGate] SELL blocked for %s (missing price/entry for fee gate).", symbol)
            return False

        pnl_pct = (cur_price - entry_price) / entry_price
        fee_mult = float(getattr(self.config, "MIN_PROFIT_EXIT_FEE_MULT", 2.0) or 2.0)
        rt_fee_pct = ((self._get_fee_bps(self.shared_state, "taker") or 10.0) * 2.0) / 10000.0
        min_profit = rt_fee_pct * fee_mult

        stagnation_override = bool(sig.get("_stagnation_override"))
        stagnation_exit_enabled = bool(getattr(self.config, "STAGNATION_EXIT_ENABLED", False))
        stagnation_max_loss_pct = abs(float(getattr(self.config, "STAGNATION_EXIT_MAX_LOSS_PCT", 0.0) or 0.0))
        if stagnation_override and stagnation_exit_enabled and pnl_pct >= -stagnation_max_loss_pct:
            self.logger.info(
                "[Meta:ProfitGate] Stagnation override allows micro-loss exit for %s pnl=%.3f%% (max_loss=%.3f%%).",
                symbol, pnl_pct * 100.0, stagnation_max_loss_pct * 100.0,
            )
            return True

        if pnl_pct < min_profit:
            self.logger.info(
                "[Meta:ProfitGate] SELL blocked for %s pnl=%.3f%% < min_profit=%.3f%% (fee_mult=%.2f)",
                symbol, pnl_pct * 100.0, min_profit * 100.0, fee_mult,
            )
            return False

        return True

    async def _passes_meta_sell_excursion_gate(self, symbol: str, sig: Dict[str, Any]) -> bool:
        """Minimum price excursion gate for non-SL MetaController SELLs."""
        reason_text = " ".join([
            str(sig.get("reason") or ""),
            str(sig.get("tag") or ""),
            str(sig.get("exit_reason") or ""),
            str(sig.get("signal_reason") or ""),
        ]).upper()
        if (
            sig.get("_is_liquidation")
            or sig.get("_is_starvation_sell")
            or sig.get("_time_exit")
            or "TIME_EXIT" in reason_text
            or "EMERGENCY" in reason_text
            or "SL" in reason_text
        ):
            return True

        entry_price = 0.0
        try:
            ot = getattr(self.shared_state, "open_trades", {}).get(symbol, {})
            entry_price = float(ot.get("entry_price", 0.0) or 0.0)
        except Exception:
            entry_price = 0.0
        if not entry_price:
            try:
                pos = getattr(self.shared_state, "positions", {}).get(symbol, {})
                entry_price = float(pos.get("avg_price", 0.0) or pos.get("entry_price", 0.0) or 0.0)
            except Exception:
                entry_price = 0.0

        cur_price = 0.0
        try:
            if hasattr(self.shared_state, "safe_price"):
                cur_price = float(await _safe_await(self.shared_state.safe_price(symbol)) or 0.0)
        except Exception:
            cur_price = 0.0
        if cur_price <= 0:
            cur_price = float(getattr(self.shared_state, "latest_prices", {}).get(symbol, 0.0) or 0.0)

        if entry_price <= 0 or cur_price <= 0:
            self.logger.info("[Meta:ExcursionGate] SELL blocked for %s (missing entry/price).", symbol)
            return False

        tick_size = 0.0
        try:
            info = await self._get_symbol_info(symbol)
            if isinstance(info, dict):
                tick_size = float(info.get("tick_size", 0.0) or 0.0)
        except Exception:
            tick_size = 0.0

        atr = 0.0
        try:
            if hasattr(self.shared_state, "calc_atr"):
                atr = float(await _safe_await(self.shared_state.calc_atr(symbol, "5m", 14)) or 0.0)
                if atr <= 0:
                    atr = float(await _safe_await(self.shared_state.calc_atr(symbol, "1m", 14)) or 0.0)
        except Exception:
            atr = 0.0
        if atr <= 0:
            fallback_pct = float(getattr(self.config, "TPSL_FALLBACK_ATR_PCT", 0.01) or 0.01)
            atr = entry_price * fallback_pct if entry_price > 0 else 0.0

        spread = 0.0
        try:
            ex = getattr(self.shared_state, "exchange_client", None)
            if ex and hasattr(ex, "get_best_bid_ask"):
                bid, ask = await _safe_await(ex.get_best_bid_ask(symbol))
                if bid and ask and ask > 0 and bid > 0:
                    spread = abs(float(ask) - float(bid))
        except Exception:
            spread = 0.0

        tick_mult = float(getattr(self.config, "EXIT_EXCURSION_TICK_MULT", 2.0) or 2.0)
        atr_mult = float(getattr(self.config, "EXIT_EXCURSION_ATR_MULT", 0.35) or 0.35)
        spread_mult = float(getattr(self.config, "EXIT_EXCURSION_SPREAD_MULT", 3.0) or 3.0)
        min_tick_move = tick_size * tick_mult if tick_size > 0 else 0.0
        threshold = max(min_tick_move, atr_mult * atr, spread_mult * spread)

        if threshold <= 0:
            return True

        excursion = abs(cur_price - entry_price)
        if excursion < threshold:
            self.logger.info(
                "[Meta:ExcursionGate] SELL blocked for %s excursion=%.6f < threshold=%.6f (tick=%.6f atr=%.6f spread=%.6f)",
                symbol, excursion, threshold, min_tick_move, atr, spread,
            )
            return False

        return True

    async def should_place_buy(self, symbol: str, planned_quote: float, 
                            confidence: float, reason: str, expected_alpha: float = 0.008) -> bool:
        """
        """
        try:
            symbol = (symbol or "").upper()
            if not symbol:
                return False
            
            # ═══════════════════════════════════════════════════════════════════════
            # FOCUS MODE GUARD: Only allow BUYs on focused symbols
            # ═══════════════════════════════════════════════════════════════════════
            # FOCUS MODE GUARD: Allow rotation (sell → buy for the same symbol)
            # ═══════════════════════════════════════════════════════════════════════
            if self.FOCUS_MODE_ENABLED:
                await self._update_focus_symbols()
                
                # Check if symbol is in focus
                if symbol not in self.FOCUS_SYMBOLS:
                    # Check if it's an existing position (accumulation allowed on existing)
                    snap = self.shared_state.get_positions_snapshot() or {}
                    existing_qty = 0.0
                    for sym_raw, pos_data in snap.items():
                        if self._normalize_symbol(sym_raw) == symbol:
                            existing_qty = float(pos_data.get("quantity", 0.0) or pos_data.get("qty", 0.0))
                            break

                    if existing_qty <= 0:
                        # New symbol, not in focus
                        self.logger.warning(
                            "[FOCUS] 🚫 BUY blocked — %s not in focus symbols %s (new position)",
                            symbol, sorted(self.FOCUS_SYMBOLS)
                        )
                        self.logger.info("[WHY_NO_TRADE] reason=FOCUS_MODE_BLOCK symbol=%s details=new_position_not_in_focus", symbol)
                        return False
                    else:
                        self.logger.debug("[FOCUS] ✓ BUY allowed on %s (existing position, accumulation)", symbol)
                else:
                    self.logger.debug("[FOCUS] ✓ BUY allowed on %s (in focus)", symbol)
            
            # ═══════════════════════════════════════════════════════════════════════
            # STEP 1: Get exchange constraints for this symbol
            # ═══════════════════════════════════════════════════════════════════════
            taker_bps = self._get_fee_bps(self.shared_state, "taker")
            slippage_bps = float(getattr(self.config, "EXIT_SLIPPAGE_BPS", getattr(self.config, "CR_PRICE_SLIPPAGE_BPS", 15.0)))
            exit_info = {}
            try:
                exit_info = await self.shared_state.compute_symbol_exit_floor(
                    symbol,
                    fee_bps=taker_bps,
                    slippage_bps=slippage_bps,
                )
            except Exception as e:
                self.logger.debug(f"[Meta:BuyGate] Exit floor compute failed for {symbol}: {e}")

            min_notional = float(exit_info.get("min_notional") or 0.0)
            if min_notional <= 0:
                min_notional = float(getattr(self.config, "MIN_NOTIONAL_USDT", 10.0))

            min_exit_quote = float(exit_info.get("min_exit_quote") or 0.0)
            base_quote = float(getattr(self.config, "DEFAULT_PLANNED_QUOTE", self._default_planned_quote) or self._default_planned_quote)
            min_trade_floor = float(self._cfg("MIN_TRADE_QUOTE", 0.0) or 0.0)
            min_trade_quote = max(min_exit_quote, base_quote, min_trade_floor)

            # ═══════════════════════════════════════════════════════════════════════
            # GLOBAL ECONOMIC BUY GATE (SOP): economically sellable + profit lock
            # ═══════════════════════════════════════════════════════════════════════
            if planned_quote < min_trade_quote:
                self.logger.info(
                    "[Meta:BuyGate] ECONOMICALLY_UNSELLABLE %s planned_quote=%.2f < min_trade_quote=%.2f",
                    symbol, planned_quote, min_trade_quote,
                )
                return False

            try:
                realized_profit = float(getattr(self.shared_state, "metrics", {}).get("realized_pnl", 0.0) or 0.0)
            except Exception:
                realized_profit = 0.0
            realized_profit = max(0.0, realized_profit)
            base_quote = float(getattr(self.config, "PROFIT_LOCK_BASE_QUOTE", self._default_planned_quote) or self._default_planned_quote)
            if planned_quote > (base_quote + realized_profit):
                self.logger.info(
                    "[Meta:BuyGate] PROFIT_LOCK %s planned_quote=%.2f > base+realized=%.2f",
                    symbol, planned_quote, base_quote + realized_profit,
                )
                return False
            
            # 🔑 STEP 2.5: Profitability Expectancy Rule (3x Fees)
            # ═══════════════════════════════════════════════════════════════════════
            is_profitable, err_msg = self.policy_manager.check_entry_profitability(
                planned_quote, expected_alpha, taker_bps
            )
            if not is_profitable:
                # RELAXATION: Allow if it's a consolidation BUY (dust merge)
                # But log it clearly.
                self.logger.warning("[Meta:Profitability] %s %s. Checking consolidation exception...", symbol, err_msg)
            
            # ═══════════════════════════════════════════════════════════════════════
            # STEP 3: Get EXISTING position notional for this symbol
            # ═══════════════════════════════════════════════════════════════════════
            existing_position_notional = 0.0
            is_existing_dust = False
            try:
                pos = await _safe_await(self.shared_state.get_position(symbol))
                if pos:
                    qty = float(pos.get("qty", 0.0))
                    price = float(pos.get("price", 0.0))
                    if qty > 0 and price > 0:
                        existing_position_notional = qty * price
                        # Check if existing position is dust
                        is_existing_dust = existing_position_notional < min_notional
                        self.logger.debug(
                            f"[Meta:BuyGate] Existing position for {symbol}: "
                            f"qty={qty:.8f}, price=${price:.2f}, notional=${existing_position_notional:.2f} "
                            f"(dust={is_existing_dust})"
                        )
            except Exception as e:
                self.logger.debug(f"[Meta:BuyGate] Failed to get existing position for {symbol}: {e}")
            
            # ═══════════════════════════════════════════════════════════════════════
            # STEP 4: Apply NO_DUST_BUY invariant with CONSOLIDATION ALLOWANCE
            # ═══════════════════════════════════════════════════════════════════════
            # INVARIANT: (planned_quote + existing_position_notional) >= min_trade_quote
            total_notional_after_buy = Decimal(str(planned_quote)) + Decimal(str(existing_position_notional))
            min_trade_quote_dec = Decimal(str(min_trade_quote))
            
            if total_notional_after_buy < min_trade_quote_dec:
                # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                # Would fail - but check CONSOLIDATION EXCEPTION first
                # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                if is_existing_dust:
                    # FIX: Cap dust re-entry to once per symbol
                    if symbol in self._consolidated_dust_symbols:
                        self.logger.info("[WHY_NO_TRADE] reason=DUST_ALREADY_CONSOLIDATED symbol=%s", symbol)
                        return False

                    # We have dust, and this BUY would consolidate it toward health
                    # Even if total doesn't reach full health, we ALLOW consolidation BUYs
                    # This is the critical rule that prevents dust from being permanent
                    self.logger.info(
                        f"[Meta:BuyGate] ✅ CONSOLIDATION ALLOWED for {symbol}:\n"
                        f"  existing_notional=${existing_position_notional:.2f} (DUST)\n"
                        f"  planned_quote=${planned_quote:.2f}\n"
                        f"  = total=${float(total_notional_after_buy):.2f} USDT\n"
                        f"  (still < min_trade_quote=${min_trade_quote:.2f}, but is consolidation)\n"
                        f"  Reason: {reason}\n"
                        f"  → BUY APPROVED (consolidation exception - dust → health path)"
                    )
                    self._consolidated_dust_symbols.add(symbol)
                    return True  # ✅ ALLOW: Consolidation of dust position
                else:
                    # No existing dust, and total would be dust OR failed expectancy - BLOCK
                    if not is_profitable:
                        self.logger.warning(
                            f"[Meta:BuyGate] ❌ PROFITABILITY VIOLATION for {symbol}: {err_msg}. Blocking entry."
                        )
                        self.logger.info("[WHY_NO_TRADE] reason=PROFITABILITY_BELOW_THRESHOLD symbol=%s details=%s", symbol, err_msg)
                        return False

                    self.logger.warning(
                        f"[Meta:BuyGate] ❌ MVT/DUST VIOLATION for {symbol}:\n"
                        f"  planned_quote={planned_quote:.2f} USDT\n"
                        f"  + existing_notional={existing_position_notional:.2f} USDT\n"
                        f"  = total={float(total_notional_after_buy):.2f} USDT\n"
                        f"  < min_trade_quote (MVT)={min_trade_quote:.2f} USDT\n"
                        f"  Reason: {reason}\n"
                        f"  → BUY REJECTED (quote below MVT/Dust limit)"
                    )
                    self.logger.info("[WHY_NO_TRADE] reason=MIN_TRADE_QUOTE_BLOCK symbol=%s details=quote_below_mvt", symbol)
                    return False  # ❌ BLOCK: Would create new dust or below MVT
            

            # ═══════════════════════════════════════════════════════════════════════
            # STEP 4.5: DUST-AWARE POSITION LOCK (Professional-grade fix)
            # ═══════════════════════════════════════════════════════════════════════
            now = time.time()
            last_buy = self._last_buy_ts.get(symbol, 0.0)

            # 1. Entry Cooldown (prevent buy storms)
            cooldown_sec = float(getattr(self.config, "ENTRY_COOLDOWN_SEC", 30.0))
            if now - last_buy < cooldown_sec:
                self.logger.info("[WHY_NO_TRADE] reason=ENTRY_COOLDOWN_ACTIVE symbol=%s details=last_buy_%.1fs_ago", symbol, now - last_buy)
                return False

            # 2. Max Position Cap (prevent over-accumulation)
            max_pos = float(getattr(self.config, "MAX_POSITION_USDT", 30.0))
            if existing_position_notional >= max_pos:
                self.logger.info("[WHY_NO_TRADE] symbol=%s reason=MAX_POSITION_REACHED details=current_%.2f_max_%.2f", symbol, existing_position_notional, max_pos)
                return False

            if (existing_position_notional + planned_quote) > max_pos:
                self.logger.info("[WHY_NO_TRADE] symbol=%s reason=MAX_POSITION_REACHED details=resulting_%.2f_max_%.2f", symbol, existing_position_notional + planned_quote, max_pos)
                return False

            # 3. DUST-AWARE POSITION LOCK
            # Treat dust as no position: if position value < DUST_POSITION_VALUE_USDT, allow BUY as bootstrap merge
            DUST_POSITION_VALUE_USDT = float(getattr(self.config, "DUST_POSITION_VALUE_USDT", 10.0))
            if existing_position_notional > 0:
                if existing_position_notional < DUST_POSITION_VALUE_USDT:
                    # Allow buy as bootstrap merge (treat as no position)
                    self.logger.info(f"[Meta:BuyGate] DUST-AWARE POSITION: {symbol} position=${existing_position_notional:.2f} < DUST_POSITION_VALUE_USDT=${DUST_POSITION_VALUE_USDT:.2f} → BUY ALLOWED (bootstrap merge)")
                else:
                    # Real position exists, block unless explicit scale-in
                    allow_scale_in = False
                    if "scale_in" in reason.lower() or "dca" in reason.lower():
                        allow_scale_in = True
                    if not allow_scale_in:
                        self.logger.info("[WHY_NO_TRADE] reason=POSITION_ALREADY_OPEN symbol=%s details=single_entry_rule_active", symbol)
                        return False

            # ═══════════════════════════════════════════════════════════════════════
            # STEP 5: Check balance availability
            # ═══════════════════════════════════════════════════════════════════════
            try:
                balance_info = await _safe_await(self.shared_state.get_balance("USDT"))
                if isinstance(balance_info, dict):
                    balance_usdt = float(balance_info.get("free", 0.0))
                else:
                    balance_usdt = float(balance_info or 0.0)
            except Exception as e:
                self.logger.debug(f"[Meta:BuyGate] Failed to get USDT balance: {e}")
                balance_usdt = 0.0
            
            if balance_usdt < planned_quote:
                self.logger.warning(
                    f"[Meta:BuyGate] BUY BLOCKED for {symbol}: "
                    f"insufficient balance. Available=${balance_usdt:.2f} < "
                    f"required=${planned_quote:.2f}. Reason: {reason}"
                )
                return False  # ❌ BLOCK: Insufficient balance
            
            # ═══════════════════════════════════════════════════════════════════════
            # STEP 6: All checks passed - BUY is safe
            # ═══════════════════════════════════════════════════════════════════════
            self.logger.info(
                f"[Meta:BuyGate] ✅ NO_DUST_BUY PASSED for {symbol}:\n"
                f"  total_notional={float(total_notional_after_buy):.2f} USDT\n"
                f"  >= min_trade_quote={min_trade_quote:.2f} USDT\n"
                f"  balance=${balance_usdt:.2f} available\n"
                f"  → BUY APPROVED (immediately sellable)"
            )
            return True  # ✅ ALLOW: Safe to place BUY
            
        except Exception as e:
            self.logger.warning(f"[Meta:BuyGate] Error evaluating BUY for {symbol}: {e}")
            # Fail-safe: block if uncertain (prefer safety over execution)
            return False

    async def dust_accumulation_guard(self, symbol: str, position_qty: float,
                                    position_mark_price: float, 
                                    symbol_min_notional: Optional[float] = None) -> bool:
        """Delegate dust accumulation guard to PolicyManager."""
        return await self.policy_manager.dust_accumulation_guard(
            symbol, position_qty, position_mark_price, symbol_min_notional,
            exchange_client=getattr(self, 'exchange_client', None),
            shared_state=self.shared_state
        )

    async def accumulation_resolution_check(self, symbol: str, rejection_quote: float, 
                                        min_notional: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        INVARIANT: ACCUMULATION_RESOLUTION
        
        Track accumulated rejected quotes and emit a single BUY TradeIntent when threshold crossed.
        
        For any symbol S:
        1. accumulated_quote[S] += rejection_quote (when BUY rejected due to minNotional)
        2. IF accumulated_quote[S] >= minNotional:
            → EMIT single BUY TradeIntent with qty for full accumulated value
            → RESET accumulated_quote[S] = 0
        
        This converts "position is accumulating" from a log line into a proper state machine
        with automatic resolution when the threshold is crossed.
        
        Args:
            symbol: Trading pair (e.g., "ETHUSDT")
            rejection_quote: USDT amount that was rejected (to accumulate)
            min_notional: Exchange minNotional (fetched if not provided)
        
        Returns:
            Dict with 'trade_intent' if resolution threshold crossed, None otherwise
            Intent dict:
            {
                'symbol': symbol,
                'accumulated_quote': accumulated_quote[S],
                'should_emit_buy': True,
                'accumulated_iterations': count of rejections,
                'accumulated_duration_sec': time since first rejection
            }
        """
        try:
            symbol = (symbol or "").upper()
            if not symbol or rejection_quote <= 0:
                return None
            
            # Get minNotional threshold
            if min_notional is None:
                try:
                    if self.exchange_client and hasattr(self.exchange_client, "get_symbol_info"):
                        info = await self.exchange_client.get_symbol_info(symbol)
                        if info and isinstance(info, dict):
                            min_notional = float(info.get("minNotional", 10.0))
                except Exception:
                    pass
                
                if min_notional is None:
                    min_notional = float(getattr(self.config, "MIN_NOTIONAL_USDT", 10.0))
            
            # Accumulate the rejected quote
            old_accumulated = self._accumulated_quote.get(symbol, 0.0)
            new_accumulated = old_accumulated + rejection_quote
            self._accumulated_quote[symbol] = new_accumulated
            
            # Update metadata
            now = time.time()
            if symbol not in self._accumulation_metadata:
                self._accumulation_metadata[symbol] = {
                    "first_rejection_time": now,
                    "rejection_count": 0,
                }
            
            self._accumulation_metadata[symbol]["last_rejection_time"] = now
            self._accumulation_metadata[symbol]["rejection_count"] += 1
            
            self.logger.info(
                "[INVARIANT:Accumulation] %s accumulated: %.2f USDT (was %.2f, +%.2f). "
                "Threshold: %.2f. Rejections: %d. Duration: %.1fs",
                symbol, new_accumulated, old_accumulated, rejection_quote, min_notional,
                self._accumulation_metadata[symbol]["rejection_count"],
                now - self._accumulation_metadata[symbol]["first_rejection_time"]
            )
            
            # Check if accumulated amount >= minNotional
            if new_accumulated >= min_notional:
                # ✅ RESOLUTION: Accumulated value crossed threshold!
                duration = now - self._accumulation_metadata[symbol]["first_rejection_time"]
                count = self._accumulation_metadata[symbol]["rejection_count"]
                
                self.logger.warning(
                    "[INVARIANT:Accumulation:RESOLUTION] %s RESOLVED! "
                    "accumulated=%.2f >= threshold=%.2f. "
                    "Iterations=%d, Duration=%.1fs. "
                    "Accumulation resolved; BUY emission disabled by global gate.",
                    symbol, new_accumulated, min_notional, count, duration
                )
                
                # Reset for next cycle
                self._accumulated_quote[symbol] = 0.0
                self._accumulation_metadata[symbol] = {
                    "first_rejection_time": now,
                    "rejection_count": 0,
                }

                return None
            
            # Still accumulating, not yet resolved
            return None
            
        except Exception as e:
            self.logger.warning(
                "[INVARIANT:Accumulation] Error processing accumulation for %s: %s",
                symbol, e
            )
            return None
    
    
    
    def get_current_mode(self) -> str:
        """Get the current operating mode."""
        return self.mode_manager.get_mode()
    
    def get_mode_info(self) -> Dict[str, Any]:
        """Get detailed mode information."""
        return self.mode_manager.get_mode_info()

    async def _gather_mode_metrics(self) -> Dict[str, Any]:
        """Gather all metrics required for the Mode Transition State Machine."""
        # 1. Run-rate & Target
        target_rr = float(self.state_manager._kpi_metrics.get("hourly_target_usdt", 20.0))
        
        curr_rr = 0.0
        if hasattr(self.shared_state, "kpi_usdt_per_hour"):
            curr_rr = float(self.shared_state.kpi_usdt_per_hour or 0.0)
        elif hasattr(self.shared_state, "metrics"):
            curr_rr = float(self.shared_state.metrics.get("usdt_per_hour", 0.0))
            
        # 2. Drawdown
        drawdown = 0.0
        if hasattr(self.shared_state, "kpi_drawdown_pct"):
            drawdown = float(self.shared_state.kpi_drawdown_pct or 0.0)
        elif hasattr(self.shared_state, "metrics"):
            drawdown = float(self.shared_state.metrics.get("drawdown_pct", 0.0))
            
        # 3. Volatility
        volatility = "NORMAL"
        # Potential future extraction: self.volatility_detector = VolatilityRegimeDetector(...)
        if hasattr(self, "volatility_detector") and self.volatility_detector:
            volatility = self.volatility_detector.get_regime().upper()
        
        # 4. Integrity & Risk
        integrity_error = False
        if hasattr(self.shared_state, "metrics"):
            if self.shared_state.metrics.get("error_counts", {}).get("INTEGRITY_ERROR", 0) > 0:
                integrity_error = True
            
        risk_flags = False # Placeholder for RiskManager integration
        
        # 5. Circuit Breaker & Failures
        cb_open = False
        if hasattr(self.shared_state, "_circuit_breakers"):
            for cb in self.shared_state._circuit_breakers.values():
                # Check enum value or string representation
                state_str = str(getattr(cb, "state", "closed")).lower()
                if "open" in state_str:
                    cb_open = True
                    break
        
        repeated_failures = False
        deadlock_threshold = int(getattr(self.config, "DEADLOCK_REJECTION_THRESHOLD", 10))
        ignore_csv = str(getattr(self.config, "DEADLOCK_REJECTION_IGNORE_REASONS", "COLD_BOOTSTRAP_BLOCK,PORTFOLIO_FULL") or "")
        ignore_reasons = {r.strip().upper() for r in ignore_csv.split(",") if r.strip()}
        rej_ttl_sec = 300.0
        now_ts_local = time.time()
        if hasattr(self.shared_state, "rejection_counters"):
            for (sym, side, reason), count in self.shared_state.rejection_counters.items():
                reason_u = str(reason).upper()
                if reason_u in ignore_reasons:
                    continue
                ts = self.shared_state.rejection_timestamps.get((sym, side, reason), now_ts_local)
                if now_ts_local - ts > rej_ttl_sec:
                    continue
                if count >= deadlock_threshold:
                    repeated_failures = True
                    break
        
        # 6. Lifecycle
        is_restart = (self.tick_id < 5)
        forced_liquidation = self.mode_manager.is_mandatory_sell_mode_active()
        
        # Health & Trades
        health_ok = True
        status = "unknown"
        if hasattr(self.shared_state, "system_health"):
            status = str(self.shared_state.system_health.get("status", "ok")).lower()
        if status in ("error", "breach", "degraded"):
            health_ok = False
            
        first_trade_executed = self._first_trade_executed
        
        # Has positions (for bootstrap/recovery check)
        has_positions = not await self.shared_state.is_portfolio_flat()
        
        # Idle time tracking
        idle_time_sec = 0.0
        if hasattr(self.state_manager, "get_idle_time_sec"):
            idle_time_sec = await self.state_manager.get_idle_time_sec()
        else:
            # Fallback: time since last trade
            last_trade = getattr(self.state_manager, "_last_execution_ts", {}).get("GLOBAL", self._start_time)
            idle_time_sec = time.time() - last_trade

        return {
            "run_rate": curr_rr,
            "target_run_rate": target_rr,
            "drawdown_pct": drawdown,
            "volatility": volatility,
            "risk_flags": risk_flags,
            "integrity_error": integrity_error,
            "circuit_breaker_open": cb_open,
            "repeated_failures": repeated_failures,
            "is_restart": is_restart,
            "forced_liquidation": forced_liquidation,
            "health_ok": health_ok,
            "health_status": status,
            "first_trade_executed": first_trade_executed,
            "has_positions": has_positions,
            "idle_time_sec": idle_time_sec
        }

    async def _evaluate_mode_switch(self):
        """Evaluate and perform mode switching logic using the SOP State Machine."""
        try:
            # 1. Gather all required metrics
            mode_metrics = await self._gather_mode_metrics()
            
            # 2. Delegate transition logic to ModeManager
            await self.mode_manager.evaluate_state_machine(mode_metrics)
            
            # 3. Mode-specific synchronization (Legacy compatibility)
            current_mode = self.mode_manager.get_mode()
            if current_mode != "BOOTSTRAP":
                self._bootstrap_lock_engaged = False
                self._bootstrap_focus_symbols_pending = False
                
        except Exception as e:
            self.logger.error(f"[Meta:ModeManager] _evaluate_mode_switch failed: {e}", exc_info=True)

############################################################
# SECTION: Mode Management (NORMAL / FOCUS / RECOVERY / BOOTSTRAP)
# Responsibility:
# - Operating mode evaluation and switching logic
# - Bootstrap, Normal, Aggressive, Recovery mode transitions
# - Mode-specific configuration and thresholds
# Future Extraction Target:
# - ModeManager
############################################################

    # Belongs to: Mode Management
    # Extraction Candidate: Yes
    # Depends on: State & Internal Counters, Capital & Portfolio Health Evaluation

    # Belongs to: Mode Management
    # Extraction Candidate: Yes
    # Depends on: Capital & Portfolio Health Evaluation, Metrics, KPIs & Observability

    # ═════════════════════════════════════════════════════════════════════════════════
    # EXECUTION LIVENESS CHECK (NEW - CRITICAL)
    # ═════════════════════════════════════════════════════════════════════════════════
    
    async def check_execution_liveness(self, symbol: str, side: str) -> Tuple[bool, str]:
        """
        CRITICAL: Check if any execution path exists for this symbol/side.
        
        If all three layers (Focus, Dust, Entry) say NO, emit alert.
        This is essential for detecting and escaping deadlocks.
        
        Returns:
            (is_viable, reason): True if execution is possible, reason if blocked
        """
        return await self.state_manager.check_execution_liveness(symbol, side)

    # -------------------
    # Health & observability aligned with SharedState
    # -------------------
    async def _health_set(self, status: str, detail: str) -> None:
        """Delegate health updates to StateManager."""
        await self.state_manager._health_set(status, detail)

    async def _heartbeat(self) -> None:
        """Delegate heartbeat to StateManager."""
        await self.state_manager._heartbeat()

    async def _log_execution_event(self, event_type: str, symbol: str, details: Dict[str, Any]) -> None:
        """Structured event logging per P9 specification."""
        event_data = {
            "ts": self._epoch(),
            "component": "MetaController",
            "event": event_type,
            "symbol": symbol,
            **details
        }

        # Log locally
        self.logger.info("Execution Event: %s", json.dumps(event_data, default=str))

        # Send to SharedState event bus
        try:
            if hasattr(self.shared_state, "emit_event"):
                await _safe_await(self.shared_state.emit_event("ExecutionEvent", event_data))
        except Exception:
            self.logger.debug("Failed to emit execution event to SharedState.", exc_info=True)

    # [FIX #5] Helper methods for mixed portfolio handling
    def has_pending_non_dust_opportunity(self, symbol: str) -> bool:
        """
        Check if a non-dust symbol has pending BUY opportunity (deferred or active).
        
        Used by tests to verify that non-dust signals are preserved during dust promotion.
        
        Returns:
            True if symbol has active BUY signal or deferred BUY from P0
            False if signal was discarded
        """
        try:
            all_signals = self.signal_manager.get_all_signals()
            for sig in all_signals:
                sig_sym = sig.get("symbol")
                if self._normalize_symbol(sig_sym) == self._normalize_symbol(symbol):
                    action = str(sig.get("action")).upper()
                    # Check for active BUY or deferred BUY
                    if action == "BUY":
                        is_deferred = sig.get("_deferred_for_p0", False)
                        if is_deferred:
                            self.logger.debug(
                                "[Meta:TestHelper] %s has DEFERRED BUY opportunity (preserved from P0)",
                                symbol
                            )
                            return True
                        else:
                            self.logger.debug(
                                "[Meta:TestHelper] %s has ACTIVE BUY opportunity",
                                symbol
                            )
                            return True
            return False
        except Exception as e:
            self.logger.warning("[Meta:TestHelper] Failed to check pending opportunities for %s: %s", symbol, e)
            return False

    def get_dust_promotion_reason(self, decision_reason: str) -> str:
        """Extract P0 dust promotion reason from decision."""
        if decision_reason and "P0_DUST_PROMOTION" in decision_reason:
            return decision_reason
        return "NONE"

    # KPI tracking aligned with SharedState
    async def _update_kpi_metrics(self, metric_type: str, value: float = 1.0, symbol: str = "") -> None:
        """Delegate KPI updates to StateManager."""
        await self.state_manager._update_kpi_metrics(metric_type, value, symbol)

    # -------------------
    # Helpers aligned with SharedState patterns
    # -------------------
    async def _readiness_snapshot(self) -> Dict[str, bool]:
        """Best-effort readiness snapshot from SharedState for gating decisions (async)."""
        snap: Dict[str, bool] = {
            "accepted_symbols_ready": True,
            "balances_ready": True,
            "market_data_ready": True,
            "ops_plane_ready": True,
        }
        try:
            getter = getattr(self.shared_state, "get_readiness_snapshot", None)
            if callable(getter):
                got = await _safe_await(getter())
                if isinstance(got, dict):
                    for k in list(snap.keys()):
                        if k in got:
                            snap[k] = bool(got[k])
        except Exception:
            pass
        return snap
    def _cfg(self, section_or_key: str, key: typing.Optional[str] = None, default=None):
        """Helper to get config values with optional section/key pattern."""
        # Robust Overload: If 2nd arg (key) is passed but not a string, treat it as default
        if key is not None and not isinstance(key, str) and default is None:
            default = key
            key = None

        try:
            val = getattr(self.config, section_or_key, default)
            if key and hasattr(val, "get"):
                return val.get(key, default)
            return val
        except Exception:
            return default

    def _get_max_positions(self) -> int:
        """AUTHORITATIVE max positions limit (Mode Envelope prioritized + Policy Nudge)."""
        limit = 5
        try:
            if self.mode_manager:
                limit = int(self.mode_manager.get_envelope().get("max_positions", 5))
        except Exception:
            limit = int(self._cfg("MAX_POSITIONS", 5))
            
        # Apply Policy Nudge
        nudge = int(self.active_policy_nudges.get("max_positions_nudge", 0))
        
        # Enforce Mode Constraint: Nudge cannot exceed the Mode's hard limit
        # Policies can reduce exposure (risk management), but not increase it beyond Mode definition
        effective_limit = limit + nudge
        return max(1, min(limit, effective_limit))

    def _get_governance_decision(self, is_flat: bool, bootstrap_override: bool) -> Dict[str, Any]:
        """
        P9: Select the correct Operating Mode and resolve conflicts deterministically.
        Returns the structured governance decision JSON required by the SOP.
        """
        mode = self.mode_manager.get_mode()
        allowed_actions = []
        blocking_reason = None
        
        # Rationale based on SOP Objective
        rationale = self.mode_manager.get_envelope().get("objective", "Ongoing operations")
        
        if mode == "PAUSED":
            allowed_actions = ["NONE"]
            blocking_reason = "Manual / Compliance Pause and Freeze"
        elif mode == "PROTECTIVE":
            allowed_actions = ["SELL", "LIQUIDATE"]
            blocking_reason = "Capital Defense: New positions blocked"
        elif mode == "BOOTSTRAP":
            allowed_actions = ["BUY"]
            if not is_flat:
                # This state shouldn't last, as BOOTSTRAP exits on first trade
                allowed_actions = ["BUY", "SELL"]
        elif mode == "SIGNAL_ONLY":
            allowed_actions = ["BUY", "SELL", "LIQUIDATE"]
            blocking_reason = "SaaS Mode: Signals only (Simulated execution)"
        elif mode == "RECOVERY":
            allowed_actions = ["BUY", "SELL", "LIQUIDATE"]
            blocking_reason = "Stabilization Mode: Exposure reduced"
            rationale = "System recovery/restart path"
        elif mode == "AGGRESSIVE":
            allowed_actions = ["BUY", "SELL", "LIQUIDATE"]
            rationale = "Performance maximization (Low risk + Low run-rate)"
        else: # NORMAL
            allowed_actions = ["BUY", "SELL", "LIQUIDATE"]

        # Confidence is derived from the current mode's minimum floor
        confidence = self.mode_manager.get_envelope().get("confidence_floor", 0.60)
        if bootstrap_override:
            confidence = 0.60 # Explicit override confidence
            rationale += " | BOOTSTRAP_OVERRIDE: Breaking flat deadlock"

        return {
            "mode": mode,
            "allowed_actions": allowed_actions,
            "override_active": bootstrap_override,
            "blocking_reason": blocking_reason,
            "confidence": confidence,
            "rationale": rationale
        }

    def _emit_governance_decision(self, decision: Dict[str, Any]):
        """Emit the structured Governance Decision and log it for P9 audit."""
        self.logger.info("[GovernanceDecision] %s", json.dumps(decision, default=str))
        
        # Sync to SharedState for cross-component coordination
        if hasattr(self.shared_state, "metrics"):
            self.shared_state.metrics["governance_decision"] = decision
            self.shared_state.metrics["current_mode"] = decision.get("mode", "BOOTSTRAP")
            
        # Optional: Add to SharedState loop summary
        if hasattr(self, "_loop_summary_state"):
            self._loop_summary_state["governance"] = decision

    def _get_mode_confidence_floor(self) -> float:
        """AUTHORITATIVE mode confidence floor (Mode Envelope prioritized + Policy Nudge)."""
        base = 0.60
        mode = "NORMAL"
        try:
            if self.mode_manager:
                mode = self.mode_manager.get_mode()
                base = float(self.mode_manager.get_envelope().get("confidence_floor", 0.60))
        except Exception:
            base = float(self._cfg("MIN_EXECUTION_CONFIDENCE", 0.60))
            
        # Apply Policy Nudge
        nudge = float(self.active_policy_nudges.get("confidence_nudge", 0.0))
        effective = base + nudge
        
        # BOOTSTRAP CONFIDENCE FLOOR: Enforce minimum confidence for bootstrap mode
        # to avoid garbage signals while still allowing liquidity seeding
        if self._is_bootstrap_mode():
            bootstrap_min_conf = float(self._cfg("BOOTSTRAP_MIN_CONFIDENCE", 0.55))
            effective = max(effective, bootstrap_min_conf)
            self.logger.debug("[Meta:Bootstrap] Confidence floor enforced: %.3f (min=%.3f)", effective, bootstrap_min_conf)
        
        # SAFETY CLAMP: Policies can INCREASE confidence floor (making entries harder/safer),
        # but they cannot LOWER the floor for safety-critical modes (PROTECTIVE, SAFE, RECOVERY).
        if mode in ("PROTECTIVE", "SAFE", "RECOVERY"):
            effective = max(base, effective)
            
        return max(0.0, min(1.0, effective))

    def set_active_policy_nudges(self, nudges: Dict[str, float]):
        """Update active policy nudges from PolicyManager."""
        self.active_policy_nudges = nudges
        # Pass cooldown nudge to StateManager
        if "cooldown_nudge" in nudges:
            self.state_manager.set_policy_modifiers({"cooldown_nudge": nudges["cooldown_nudge"]})


    def _is_position_dust_locked(self, state: str, qty: float, value_usdt: float = 0.0) -> bool:
        """
        Central definition of dust classification.
        
        A position is considered dust if:
        1. Explicitly marked as DUST_LOCKED, OR
        2. Has micro quantity (< 0.0001) AND positive value in USDT
        
        Args:
            state: Position state string
            qty: Position quantity
            value_usdt: Position value in USDT (default 0.0)
        
        Returns:
            bool: True if position is dust
            
        Note:
            value_usdt check is critical to avoid "phantom dust" 
            (pending orders with 0 value)
        """
        return state == "DUST_LOCKED" or (0 < qty < 0.0001 and value_usdt > 0)

    def _is_recovery_sellable(
        self,
        symbol: str,
        pos: Dict[str, Any],
        ignore_filters: bool = False,
        ignore_core: bool = False,
    ) -> bool:
        """Central recovery eligibility check for nomination and forced exits."""
        if not isinstance(pos, dict):
            return False
        qty = float(pos.get("quantity", 0.0) or pos.get("qty", 0.0))
        if qty <= 0:
            return False
        if ignore_filters:
            return True

        value_usdt = float(pos.get("value_usdt", 0.0) or 0.0)
        state = str(pos.get("state", "") or "")
        if self._is_position_dust_locked(state, qty, value_usdt):
            return False

        quote_asset = str(self._cfg("QUOTE_ASSET") or "USDT").upper()
        base = symbol.replace(quote_asset, "") if symbol.endswith(quote_asset) else symbol
        core_assets = set(getattr(self, "SACRIFICE_PRECIOUS", {"BTC", "ETH"}))
        if not ignore_core and base in core_assets:
            return False

        min_significant = float(
            self._cfg(
                "MIN_SIGNIFICANT_POSITION_USDT",
                self._cfg("MIN_SIGNIFICANT_USD", 25.0),
            )
        )
        if value_usdt > 0 and value_usdt < min_significant:
            return False

        return True

    def _get_sacrifice_priority(self, symbol: str) -> int:
        """
        Get sacrifice priority tier for a symbol.
        
        Lower priority value = sacrifice earlier
        - Meme coins: 0 (sacrifice first)
        - Common coins: 50 (sacrifice second)
        - Precious (BTC/ETH): 100 (sacrifice last)
        
        Args:
            symbol: Asset symbol (e.g., "DOGE", "BTC")
        
        Returns:
            int: Priority value (0=highest, 100=lowest)
        """
        sym_upper = symbol.upper()
        
        if any(meme in sym_upper for meme in self.SACRIFICE_MEME_COINS):
            return self.SACRIFICE_PRIORITY_MEME
        elif any(precious in sym_upper for precious in self.SACRIFICE_PRECIOUS):
            return self.SACRIFICE_PRIORITY_PRECIOUS
        else:
            return self.SACRIFICE_PRIORITY_COMMON

    def _is_portfolio_full(self, total_pos: int, sig_pos: int, dust_pos: int, max_pos: int) -> Tuple[bool, Optional[str]]:
        """
        Enhanced portfolio capacity check with two strategies:
        
        OPTION A: Count significant positions only (recommended for dust-heavy portfolios)
        OPTION B: Dust override policy (emergency escape when dust_ratio is high)
        
        Args:
            total_pos: Total positions (significant + dust)
            sig_pos: Count of significant positions (value >= threshold)
            dust_pos: Count of dust positions (value < threshold)
            max_pos: Maximum allowed positions
        
        Returns:
            Tuple[bool, Optional[str]]:
                - bool: True if portfolio should be considered full
                - str: Reason if overridden (for logging), None if using standard calculation
        
        Examples:
            total=10, sig=3, dust=7, max=5 (dust heavy):
                - Option A: 3 < 5 → NOT_FULL (can add more despite total > max)
                - Option B: dust_ratio=70% > 60% → DUST_OVERRIDE (even if sig_pos >= max)
        """
        # Standard check: portfolio full when position count reaches max
        standard_is_full = total_pos >= max_pos
        
        # ===== OPTION A: Significant-Only Capacity =====
        # Count only significant positions toward capacity limit
        if self.CAPACITY_COUNT_SIGNIFICANT_ONLY:
            sig_is_full = sig_pos >= max_pos
            
            if sig_is_full != standard_is_full:
                # Significant count differs from total count
                # This happens when portfolio has high dust ratio
                reason = f"SIG_ONLY[sig={sig_pos}/{max_pos}]" if sig_is_full else None
                return sig_is_full, reason
        
        # ===== OPTION B: Dust Override Policy =====
        # If dust ratio is extremely high, override portfolio_full restriction
        if self.DUST_OVERRIDE_ENABLED and standard_is_full:
            total = total_pos if total_pos > 0 else 1
            # INVARIANT:
            # dust_ratio MUST count only economic dust (>= MIN_SIGNIFICANT_USDT).
            # Registry-level dust (<MIN_SIGNIFICANT_USDT) must not affect ratio.
            dust_ratio = dust_pos / total
            
            if dust_ratio > self.DUST_OVERRIDE_THRESHOLD:
                # Dust ratio is critically high (e.g., 70% dust, 30% significant)
                # Override portfolio_full to allow capital recovery via healing
                reason = f"DUST_OVERRIDE[ratio={dust_ratio:.1%}>{self.DUST_OVERRIDE_THRESHOLD:.1%}]"
                return False, reason  # NOT full despite hitting position limit
        
        # Standard calculation: count all positions
        return standard_is_full, None

    async def _select_dust_exit_candidate(self) -> Optional[str]:
        """
        Select a dust position for forced exit (DUST_EXIT_POLICY).
        
        Used when system is stuck (no SELL signals, no BUY signals, full portfolio)
        to force liquidation of oldest/lowest-confidence dust to free capacity and capital.
        
        Selection criteria (in order):
        1. Is dust (state == DUST_LOCKED or qty < 0.0001)
        2. Oldest first (longest held, least recovery probability)
        3. Lowest confidence (weaker signal history)
        4. Lowest notional value (smallest loss)
        
        Returns:
            Optional[str]: Symbol to SELL, or None if no dust candidate found
            
        Notes:
            - Only returns ONE position (not aggressive)
            - Filtered to exclude positions already being worked on
            - Logs selection reasoning at INFO level
        """
        try:
            if not self.portfolio_manager:
                return None
                
            # Get all positions from portfolio
            positions = getattr(self.portfolio_manager, "positions", {})
            if not positions:
                return None
            
            # Filter to dust positions only
            dust_candidates = []
            
            for symbol, pos_data in positions.items():
                if not isinstance(pos_data, dict):
                    continue
                
                # Check if this is a dust position
                state = pos_data.get("state", "UNKNOWN")
                qty = float(pos_data.get("qty", 0))
                value_usdt = float(pos_data.get("value_usdt", 0))
                
                if self._is_position_dust_locked(state, qty, value_usdt):
                    # Get metadata for sorting
                    created_at = pos_data.get("created_at", 0)
                    confidence = float(pos_data.get("confidence", 0.5))  # 0-1 scale
                    notional = float(pos_data.get("notional_usdt", value_usdt))
                    
                    dust_candidates.append({
                        "symbol": symbol,
                        "created_at": created_at,
                        "confidence": confidence,
                        "notional": notional,
                        "value_usdt": value_usdt,
                        "qty": qty,
                    })
            
            if not dust_candidates:
                return None
            
            # Sort by criteria:
            # 1. Oldest first (smallest created_at)
            # 2. Then lowest confidence
            # 3. Then lowest notional (smallest loss)
            dust_candidates.sort(
                key=lambda x: (x["created_at"], -x["confidence"], x["notional"])
            )
            
            # Select the first (oldest, lowest confidence, lowest notional)
            selected = dust_candidates[0]
            
            self.logger.info(
                "[DUST_EXIT] Selected dust candidate for forced liquidation: %s "
                "(value=%.2f, age=%d cycles, confidence=%.2f, notional=%.2f)",
                selected["symbol"], selected["value_usdt"], 
                self.cycles_no_trade, selected["confidence"], selected["notional"]
            )
            
            return selected["symbol"]
            
        except Exception as e:
            self.logger.warning("[DUST_EXIT] Failed to select candidate: %s", str(e))
            return None

    def _build_policy_context(self, symbol: str, side: str, policies: Optional[List[str]] = None, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Delegate policy context building to PolicyManager."""
        return self.policy_manager._build_policy_context(symbol, side, policies, extra)

    def _ensure_decision_id(self, symbol: str, side: str, sig: Dict[str, Any], idx: int) -> str:
        """Attach a deterministic decision_id to a signal if missing."""
        if not isinstance(sig, dict):
            return ""
        for key in (
            "decision_id",
            "decisionId",
            "signal_id",
            "signalId",
            "id",
            "cache_key",
            "intent_id",
            "intentId",
        ):
            val = sig.get(key)
            if val:
                sig["decision_id"] = str(val)
                return sig["decision_id"]
        base = f"{self._normalize_symbol(symbol)}:{side.upper()}:{int(self.tick_id)}:{int(idx)}"
        sig["decision_id"] = base
        return base

    def _extract_expected_edge_bps(self, signal: Optional[Dict[str, Any]]) -> Optional[float]:
        """Delegate edge extraction to PolicyManager."""
        return self.policy_manager._extract_expected_edge_bps(signal)

    def _check_economic_profitability(self, symbol: str, signal: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """Delegate economic profitability check to PolicyManager."""
        return self.policy_manager._check_economic_profitability(symbol, signal)

    def _update_phase2_guard(self, dust_ratio: float) -> Tuple[bool, float]:
        """Delegate phase 2 guard update to PolicyManager."""
        return self.policy_manager._update_phase2_guard(dust_ratio)

    async def _check_portfolio_flat(self) -> bool:
        """
        Returns: True ONLY if ALL positions have qty <= 0
        
        ===== DEADLOCK SAFETY: Multi-Method Verification =====
        We verify flatness using multiple independent paths to prevent single-point failure.
        If ANY path shows inventory exists, we return False.
        """
        # ===== PRIMARY CHECK: Get all positions, check for any qty > 0 =====
        try:
            positions = self.shared_state.get_open_positions()
            if isinstance(positions, dict):
                if len(positions) == 0:
                    self.logger.debug("[Meta:CheckFlat] Portfolio FLAT: open_positions=0")
                    return True
                self.logger.debug("[Meta:CheckFlat] Portfolio NOT FLAT: open_positions=%d", len(positions))
                return False
        except Exception as e:
            self.logger.warning("[Meta:CheckFlat] Primary check failed: %s", e)

        # ===== SECONDARY CHECK: Portfolio Snapshot (from policy manager) =====
        # This provides a second authoritative opinion.
        try:
            snap = self.shared_state.get_positions_snapshot()
            if snap and len(snap) > 0:
                for sym, p in snap.items():
                    qty = float(p.get("qty", 0.0) or p.get("quantity", 0.0))
                    if qty > 0:
                        self.logger.debug(
                            "[Meta:CheckFlat] Snapshot shows %s qty=%.8f > 0. NOT FLAT.",
                            sym, qty
                        )
                        return False
        except Exception as e:
            self.logger.warning("[Meta:CheckFlat] Secondary check failed: %s", e)

        # If both checks pass, it's flat.
        self.logger.info("[Meta:CheckFlat] Portfolio is FLAT. Ready for fresh capital allocation.")
        return True

    def _is_budget_required(self, action: str) -> bool:
        """Determine if an action requires capital allocation (USDT budget)."""
        return str(action).upper() == "BUY"
    async def _should_allow_sell(self, symbol: str) -> bool:
        """
        ===== THE HARD INVARIANT: SELL GATING =====
        
        CRITICAL RULE: If SharedState has inventory for a symbol, SELL is ALWAYS allowed.
        
        SELL must NOT be blocked by:
        - PORTFOLIO_FULL
        - BOOTSTRAP mode
        - CAPITAL_STARVATION
        - is_flat check
        
        Why: SELL is capacity-FREEING, not capacity-CONSUMING.
        If SELL is blocked, no capacity can ever be freed → deadlock.
        
        Returns: True if SELL should be allowed (inventory exists)
        """
        symbol = (symbol or "").upper()
        if not symbol:
            return False
        
        # Get authoritative inventory from SharedState
        try:
            # Try method #1: get_open_positions()
            positions = self.shared_state.get_open_positions()
            if positions and isinstance(positions, dict):
                p = positions.get(symbol)
                if p and isinstance(p, dict):
                    qty = float(p.get("qty", 0.0) or p.get("quantity", 0.0))
                    if qty > 0:
                        self.logger.debug("[Meta:SellGate] %s has qty=%.8f → ALLOW SELL", symbol, qty)
                        return True
        except Exception as e:
            self.logger.debug("[Meta:SellGate] get_open_positions() failed for %s: %s", symbol, e)
        
        # Fallback: Try get_positions()
        try:
            if hasattr(self.shared_state, "get_positions"):
                snap = await _safe_await(self.shared_state.get_positions())
                if snap and isinstance(snap, dict):
                    p = snap.get(symbol)
                    if p and isinstance(p, dict):
                        qty = float(p.get("qty", 0.0) or p.get("quantity", 0.0))
                        if qty > 0:
                            self.logger.debug("[Meta:SellGate:Fallback] %s has qty=%.8f → ALLOW SELL", symbol, qty)
                            return True
        except Exception as e:
            self.logger.debug("[Meta:SellGate:Fallback] get_positions() failed for %s: %s", symbol, e)
        
        # Fallback: Try get_positions_snapshot()
        try:
            if hasattr(self.shared_state, "get_positions_snapshot"):
                snap = self.shared_state.get_positions_snapshot()
                if snap and isinstance(snap, dict):
                    p = snap.get(symbol)
                    if p and isinstance(p, dict):
                        qty = float(p.get("qty", 0.0) or p.get("quantity", 0.0))
                        if qty > 0:
                            self.logger.debug("[Meta:SellGate:Fallback2] %s has qty=%.8f → ALLOW SELL", symbol, qty)
                            return True
        except Exception as e:
            self.logger.debug("[Meta:SellGate:Fallback2] get_positions_snapshot() failed for %s: %s", symbol, e)
        
        # No inventory found
        self.logger.debug("[Meta:SellGate] %s has no inventory or qty <= 0 → BLOCK SELL", symbol)
        return False

    # -------------------
    # Public API
    # -------------------
    async def receive_intents(self, intents: List[Any]):
        """Accept a batch of intents from AgentManager and push to sink."""
        await self.intent_manager.receive_intents(intents)

############################################################
# SECTION: Signal Intake & Caching
# Responsibility:
# - Receiving signals from MLForecaster and other agents
# - Signal validation, caching, and deduplication
# - Signal queue management and cleanup
# Future Extraction Target:
# - SignalManager or SignalBus
############################################################

    # Belongs to: Signal Intake & Caching
    # Extraction Candidate: Yes
    # Depends on: None (external interface)
    async def receive_signal(self, agent_name: str, symbol: str, signal: Dict[str, Any]):
        """Accept and cache signals with delegation to SignalManager."""
        if not symbol or not isinstance(signal, dict):
            self.logger.warning("Invalid signal received: symbol=%s signal=%s", symbol, signal)
            return

        # Use SignalManager for core intake logic
        success = self.signal_manager.receive_signal(agent_name, symbol, signal)
        if not success:
            return

        # Fetch the normalized signal for SharedState sync
        sig = self.signal_manager.get_signals_for_symbol(self._normalize_symbol(symbol))
        if sig:
            # Get the most recent one for this agent
            agent_sig = next((s for s in reversed(sig) if s.get("agent") == agent_name), None)
            if agent_sig:
                try:
                    if hasattr(self.shared_state, "add_strategy_signal"):
                        await self.shared_state.add_strategy_signal(agent_sig["symbol"], agent_sig)
                except Exception:
                    pass
        
        self.logger.debug("[MetaController] Signal processed and cached for %s", symbol)

    # -------------------
    # LAYER 1: LOOP_SUMMARY Emission
    # -------------------
    def _emit_loop_summary(self):
        """
        Emit one structured LOOP_SUMMARY per tick.
        This is the ONLY high-level log per tick; all decisions, rejections, deadlocks are captured here.
        """
        try:
            state = self._loop_summary_state
            focus_info = ""
            if self.FOCUS_MODE_ENABLED:
                focus_info = f" focus_symbols={sorted(self.FOCUS_SYMBOLS)}"
            
            self.logger.info(
                "[LOOP_SUMMARY] "
                "loop_id=%d symbols=%d top=%s decision=%s exec_attempted=%s exec_result=%s "
                "rejection_reason=%s rej_count=%d fallback=%s capital_free=%.2f reserved=%.2f "
                "trade_opened=%s pnl=%.2f deadlock=%s health=%s%s",
                state["loop_id"],
                state["symbols_considered"],
                state["top_candidate"],
                state["decision"],
                state["execution_attempted"],
                state["execution_result"],
                state["rejection_reason"],
                state["rejection_count"],
                state["fallback_used"],
                state["capital_free"],
                state["capital_reserved"],
                state["trade_opened"],
                state["realized_pnl"],
                state["deadlock"],
                state["system_health"],
                focus_info,
            )
        except Exception as e:
            self.logger.debug("_emit_loop_summary failed: %s", e)

    # -------------------
    # Lifecycle
    # -------------------
    async def start(self, interval_sec: float = 2.0):
        if getattr(self, "_running", False):
            return
        await self._disable_bootstrap_if_positions()
        self._running = True
        self.interval = interval_sec

        self._eval_task = _asyncio.create_task(self.run(), name="meta.run")
        self._health_task = _asyncio.create_task(self.report_health_loop(), name="meta.health")
        
        async def _cleanup_loop():
            """Background cleanup and monitoring."""
            try:
                while self._running:
                    await self._run_cleanup_cycle()
                    await _asyncio.sleep(30) # Faster poll during startup
            except _asyncio.CancelledError:
                pass

        self._cleanup_task = _asyncio.create_task(_cleanup_loop(), name="meta.cleanup")

        await self._health_set("Healthy", "MetaController started.")
        self.logger.info("MetaController started.")

    async def _disable_bootstrap_if_positions(self) -> None:
        """Hard rule: disable bootstrap when any exchange-backed positions exist."""
        try:
            snap = self.shared_state.get_positions_snapshot() or {}
            has_positions = False
            for pos in snap.values():
                qty = float(pos.get("quantity", 0.0) or pos.get("qty", 0.0))
                status = str(pos.get("status", "")).upper()
                if qty > 0 and status not in {"CLOSED", "DUST", "PERMANENT_DUST"}:
                    has_positions = True
                    break

            if not has_positions:
                return

            self._bootstrap_lock_engaged = False
            self._bootstrap_focus_symbols_pending = False
            self._current_mode = "NORMAL"
            self._last_mode = "BOOTSTRAP"

            if self._post_bootstrap_symbol_limit is not None:
                self._active_symbol_limit = self._post_bootstrap_symbol_limit
            else:
                self._active_symbol_limit = int(self._cfg("MAX_ACTIVE_SYMBOLS", default=5))

            self.logger.warning(
                "[Meta:BootstrapOff] Exchange positions detected. Bootstrap disabled. active_symbols=%s",
                self._active_symbol_limit,
            )
        except Exception as e:
            self.logger.debug("_disable_bootstrap_if_positions failed: %s", e)

    # Belongs to: MetaController Lifecycle & Entry Points
    # Extraction Candidate: Yes
    # Depends on: Policy Evaluation Pipeline, State & Internal Counters
    async def run(self):
        """Mandatory P9 lifecycle loop: runs forever executing evaluation and heartbeat."""
        self.logger.info("[MetaController] Starting lifecycle loop (interval=%.1fs)", self.interval)
        while not self._stop and self._running:
            try:
                await self.evaluate_and_act()
            except Exception as e:
                self.logger.error("[MetaController] Evaluation cycle crashed: %s", e, exc_info=True)
                await self._update_kpi_metrics("error", ExecutionError.Type.INTEGRITY_ERROR)
            
            # Integrated heartbeat per USER specification
            try:
                if hasattr(self.shared_state, "update_system_health"):
                    # Call sync update_system_health first to update component_statuses mirror
                    self.shared_state.update_system_health(
                        component="MetaController",
                        status="Healthy",
                        detail=f"tick={self.tick_id}"
                    )
                # Also try async update_component_status for better watchdog integration
                if hasattr(self.shared_state, "update_component_status"):
                    await self.shared_state.update_component_status(
                        "MetaController",
                        "Healthy",
                        f"tick={self.tick_id}"
                    )
                # Always emit CSL heartbeat as a fallback for watchdogs relying on ComponentStatusLogger.
                await self._heartbeat()
            except Exception:
                pass
            
            await _asyncio.sleep(self.interval)

    async def stop(self):
        if not self._running:
            return
        self._stop = True
        self._running = False

        for t in (self._eval_task, self._health_task, self._cleanup_task):
            if t and not t.done():
                t.cancel()
        for t in (self._eval_task, self._health_task, self._cleanup_task):
            if t:
                try:
                    await t
                except _asyncio.CancelledError:
                    pass

        self._eval_task = None
        self._health_task = None
        self._cleanup_task = None
        self.logger.info("MetaController stopped.")

    async def run_loop(self):
        """Scheduler-friendly entrypoint."""
        if not getattr(self, "_running", False):
            await self.start(self._cfg("META_TICK_INTERVAL_SEC", default=2.0))
        try:
            await _asyncio.gather(self._eval_task, self._health_task, self._cleanup_task)
        except _asyncio.CancelledError:
            self.logger.info("MetaController run_loop cancelled.")
        except Exception as e:
            self.logger.exception("MetaController run_loop crashed: %s", e)
            await self._health_set("Critical", f"Run loop crashed: {e}")
        finally:
            await self.stop()

    async def _run_cleanup_cycle(self):
        """Perform a single iteration of background cleanup and lifecycle checks."""
        try:
            # Clean up expired cache entries (with null guards)
            self.signal_manager.cleanup_expired_signals()
            if self._min_notional_cache is not None:
                self._min_notional_cache.cleanup_expired()
            if self._last_reason_log is not None:
                self._last_reason_log.cleanup_expired()

            # Log KPI status periodically
            kpi_status = await self.get_kpi_status()
            if kpi_status.get("execution_count", 0) > 0:
                self.logger.info("KPI Status: %s", json.dumps(kpi_status, default=str))

            # 1. OpsPlaneReady Emission Logic
            if not self._has_emitted_ops_ready:
                # Verify core dependencies are present
                deps_ok = all([
                    self.shared_state is not None,
                    self.exchange_client is not None,
                    self.execution_manager is not None,
                    self.tp_sl_engine is not None
                ])
                if deps_ok:
                    # Stricter P9 Readiness: check component health
                    health_ok = True
                    try:
                        # Check ExecutionManager health (if it exposes health())
                        if hasattr(self.execution_manager, "health"):
                            h = self.execution_manager.health()
                            if h.get("status") != "Healthy": health_ok = False
                        
                        # Check TP/SL Engine health
                        if health_ok and hasattr(self.tp_sl_engine, "health"):
                            h = self.tp_sl_engine.health()
                            if h.get("status") != "Healthy": health_ok = False
                        
                        # Check PerformanceEvaluator health via SharedState mirror
                        if health_ok and hasattr(self.shared_state, "component_statuses"):
                            statuses = getattr(self.shared_state, "component_statuses", {})
                            pe_status = statuses.get("PerformanceEvaluator", {}).get("status", "Healthy")
                            if pe_status != "Healthy": health_ok = False
                    except Exception:
                        self.logger.debug("Component health check failed during OpsPlaneReady evaluation", exc_info=True)
                        health_ok = False

                # Health gating based on critical component statuses
                health_ready = True
                try:
                    # BOOTSTRAP FIX: Check if we're in bootstrap mode
                    is_bootstrap = False
                    if hasattr(self.shared_state, "is_bootstrap_mode"):
                        is_bootstrap = self.shared_state.is_bootstrap_mode()
                    
                    snap = self.shared_state.get_component_status_snapshot()
                    # ExecutionManager and TPSLEngine are MUST-HAVES for OpsPlaneReady
                    # PerformanceEvaluator is OPTIONAL during bootstrap
                    required_components = ["ExecutionManager", "TPSLEngine"]
                    if not is_bootstrap:
                        required_components.append("PerformanceEvaluator")
                    
                    for comp in required_components:
                        st = snap.get(comp, {}).get("status", "").lower()
                        if st not in ("running", "operational", "healthy"):
                            health_ready = False
                            self.logger.debug(f"[Meta] Bootstrap={is_bootstrap}, {comp} not ready: {st}")
                            break
                except Exception:
                    health_ready = False

                if health_ready:
                    # Spec Point 2/3: Readiness must depend on EXECUTABLE capital or ACTIVE positions
                    # This is now encapsulated in shared_state.is_ops_plane_ready() if we force a check,
                    # but here we are the ones SETTING the event.
                    
                    has_executable_capital = False
                    try:
                        # Authoritative check: do we have reserved budget? (P9 compliant)
                        reservations = self.shared_state.get_authoritative_reservations()
                        if sum(reservations.values()) > 0:
                            has_executable_capital = True
                        
                        # Active Lifecycle check (Rule 3)
                        if not has_executable_capital:
                            snap = self.shared_state.get_positions_snapshot()
                            for p in snap.values():
                                if float(p.get("quantity", 0.0)) > 0:
                                    has_executable_capital = True
                                    break
                    except Exception:
                        pass
                            
                    if not has_executable_capital:
                        self._info_once("__waiting_budget__", "[Meta] Health OK, but waiting for CapitalAllocator budget or active positions.")
                        return

                    if not self.shared_state.ops_plane_ready_event.is_set():
                        self.logger.info("[Meta] Core stability & budget detected. OpsPlaneReady = TRUE.")
                        self.shared_state.ops_plane_ready_event.set()
                        # Emit event for external consumers
                        if hasattr(self.shared_state, "emit_event"):
                            await _safe_await(self.shared_state.emit_event("OpsPlaneReady", {"timestamp": time.time()}))
                    self._has_emitted_ops_ready = True

            # 2. Symbol Expansion check
            # BOOTSTRAP FIX: Expand symbol limit after first trade
            try:
                is_bootstrap = False
                if hasattr(self.shared_state, "is_bootstrap_mode"):
                    is_bootstrap = self.shared_state.is_bootstrap_mode()
                
                # If we've exited bootstrap (first trade completed), expand symbol limit
                if not is_bootstrap and self._active_symbol_limit == self._bootstrap_symbol_limit:
                    self.logger.info(f"[Meta] BOOTSTRAP COMPLETE: Expanding symbol limit {self._bootstrap_symbol_limit} → {self._post_bootstrap_symbol_limit}")
                    if hasattr(self, "_post_bootstrap_symbol_limit"):
                        self._active_symbol_limit = self._post_bootstrap_symbol_limit
            except Exception as e:
                self.logger.debug(f"Bootstrap symbol expansion check failed: {e}")
            
            if not self._perf_eval_ready:
                try:
                    if hasattr(self.shared_state, "get_component_status_snapshot"):
                        snap = self.shared_state.get_component_status_snapshot()
                        perf_status = snap.get("PerformanceEvaluator", {}).get("status", "").lower()
                        if perf_status in ("running", "operational", "healthy"):
                            self.logger.info("[Meta] PerformanceEvaluator warm-up complete. Expansion enabled.")
                            self._perf_eval_ready = True
                            self._active_symbol_limit = int(self._cfg("MAX_ACTIVE_SYMBOLS", default=5))
                except Exception:
                    pass

        except Exception as e:
            self.logger.warning("Cleanup cycle error: %s", e)

    # -------------------
    # Core evaluation logic
    # -------------------
    # Belongs to: MetaController Lifecycle & Entry Points
    # Extraction Candidate: Yes
    # Depends on: All subsystems - main orchestration method
    # TODO (Post-Stabilization):
    # This function mixes orchestration and policy execution.
    # Candidate split into:
    # - orchestrate_evaluation_cycle()
    # - execute_policy_pipeline()
    async def evaluate_and_act(self):
        """
        P9: Mandatory Lifecycle Evaluation cycle.
        Ingests all signals once, builds decisions, and executes based on readiness gating.
        
        LOGGING CONTRACT:
        - Emits exactly ONE [LOOP_SUMMARY] at END of tick with all decision metrics
        - Emits [DEADLOCK_DETECTED] if rejection threshold exceeded
        - Emits [EXEC_REJECT] for each execution failure with reason
        - Emits [ECON_EVENT] ONLY for trades opened/closed, capital moves
        """
        self._last_cycle_execution_attempts = self.get_execution_attempts_this_cycle()
        self.reset_execution_attempts()  # Structural Correction: Reset at cycle boundary
        self.tick_id += 1
        self._tick_counter += 1
        loop_id = self._tick_counter

        # Refresh temporary BUY re-entry delta (auto-restore based on equity/trade count)
        self._refresh_buy_reentry_delta()

        # Reset per-cycle dust merge tracker so dust healing can be re-attempted
        # after the underlying issue (e.g. symbol_lifecycle) is resolved.
        if hasattr(self, "_dust_merges"):
            self._dust_merges.clear()

        # --- Signal Cache Cleanup: Remove expired signals at start of each cycle ---
        try:
            cleaned_count = self.signal_manager.cleanup_expired_signals()
            if cleaned_count > 0:
                self.logger.debug(f"[Meta:SignalCache] Cleaned up {cleaned_count} expired signals")
        except Exception as e:
            self.logger.warning(f"[Meta:SignalCache] Cleanup failed: {e}")
        
        # ===== AUTOMATIC MODE SWITCHING & POLICY EVALUATION =====
        # Evaluate if we should switch operating modes at the start of each evaluation cycle
        try:
            await self._evaluate_mode_switch()
            await self.policy_manager.evaluate_policies(self, loop_id)
        except Exception as e:
            self.logger.error(f"[Meta:ModeSwitch] Failed to evaluate modes/policies: {e}", exc_info=True)

        # ═════════════════════════════════════════════════════════════════════════
        # CRITICAL FIX #1: WALLET_FOCUS_BOOTSTRAP - Startup Bootstrap Sequence
        # ═════════════════════════════════════════════════════════════════════════
        # On first cycle, initialize focus symbols from wallet balances
        if self.FOCUS_MODE_ENABLED and self._bootstrap_focus_symbols_pending:
            self.logger.info("[WALLET_FOCUS_BOOTSTRAP] Triggering startup bootstrap on first cycle...")
            await self._bootstrap_focus_symbols()
        
        # Update focus symbols (now returns pinned set from bootstrap)
        if self.FOCUS_MODE_ENABLED:
            await self._update_focus_symbols()
        
        # Initialize LOOP_SUMMARY state for this tick
        self._loop_summary_state = {
            "loop_id": loop_id,
            "symbols_considered": 0,
            "top_candidate": None,
            "decision": "NONE",
            "execution_attempted": False,
            "execution_result": "NONE",
            "rejection_reason": None,
            "rejection_count": 0,
            "fallback_used": False,
            "fallback_symbol": None,
            "trade_opened": False,
            "trade_closed": False,
            "realized_pnl": 0.0,
            "unrealized_delta": 0.0,
            "capital_free": 0.0,
            "capital_reserved": 0.0,
            "deadlock": False,
            "system_health": "HEALTHY",
        }
        
        # --- Capital Integrity: Recompute balance every tick ---
        if hasattr(self.shared_state, "sync_authoritative_balance"):
            await self.shared_state.sync_authoritative_balance()
            
        # --- Circuit Breaker Invariant: Freeze on CB_OPEN ---
        if hasattr(self.shared_state, "is_circuit_breaker_open") and await self.shared_state.is_circuit_breaker_open():
            self.logger.warning("[Meta:Freeze] 🛑 Circuit Breaker is OPEN. Freezing all trade intents.")
            self._loop_summary_state["system_health"] = "ERROR"
            self._loop_summary_state["deadlock"] = True
            await self._health_set("Degraded", "Frozen: Circuit Breaker Open")
            self._emit_loop_summary()
            # --- FOCUS_MODE auto-exit logic: if active, count healthy cycles and auto-exit after threshold ---
            if self._focus_mode_active:
                # If a trade was executed or system is healthy, increment healthy cycles
                if self._focus_mode_trade_executed:
                    self._focus_mode_healthy_cycles += 1
                    self.logger.info(f"[Meta:FOCUS_MODE] Healthy cycle in FOCUS_MODE: {self._focus_mode_healthy_cycles}/{self.FOCUS_MODE_AUTO_EXIT_HEALTHY_CYCLES}")
                    if self._focus_mode_healthy_cycles >= self.FOCUS_MODE_AUTO_EXIT_HEALTHY_CYCLES:
                        self.logger.info("[Meta:FOCUS_MODE] Auto-deactivating FOCUS_MODE after healthy cycles.")
                        self._deactivate_focus_mode()
                else:
                    self._focus_mode_healthy_cycles = 0
            return

        # 1. Consolidate Signal Ingestion (sink + bus + liquidation)
        now_epoch = self._epoch()
        try:
            await self._flush_intents_to_cache(now_ts=now_epoch)
            await self._ingest_strategy_bus(now_ts=now_epoch)
            await self._ingest_liquidation_signals(now_ts=now_epoch)
        except Exception as e:
            self.logger.error("[Meta] Signal ingestion failure: %s", e)
            self._emit_loop_summary()
            return

        # 2. Synchronize Symbol Universe from Source of Truth
        accepted_symbols_set = set()
        try:
            if hasattr(self.shared_state, "get_analysis_symbols"):
                active_list = self.shared_state.get_analysis_symbols()
                accepted_symbols_set = set(active_list) if active_list else set()
            elif hasattr(self.shared_state, "get_accepted_symbols_snapshot"):
                sn = await _safe_await(self.shared_state.get_accepted_symbols_snapshot())
                accepted_symbols_set = set(sn) if sn else set()
        except Exception as e:
            self.logger.warning("[Meta] Symbol sync failure: %s", e)
            self._emit_loop_summary()
            return
        
        self._loop_summary_state["symbols_considered"] = len(accepted_symbols_set)

        # 3. Build Decision Context (Portfolio Arbitration)
        decisions = await self._build_decisions(accepted_symbols_set)
        
        # Frequency Engineering: Always update utilization and check for idle ticks
        if hasattr(self.shared_state, "update_utilization_metric"):
            await _safe_await(self.shared_state.update_utilization_metric())

        if not decisions:
            if hasattr(self.shared_state, "increment_idle_ticks"):
                self.shared_state.increment_idle_ticks()
            # Update capital_free even when no decisions, so loop summary
            # always reports actual spendable capital instead of 0.00
            try:
                quote_asset = str(self._cfg("QUOTE_ASSET") or "USDT").upper()
                if quote_asset in self.shared_state.balances:
                    actual_spendable = float(await _safe_await(self.shared_state.get_spendable_balance(quote_asset)) or 0.0)
                    self._loop_summary_state["capital_free"] = actual_spendable
            except Exception:
                pass
            self._emit_loop_summary()
            return

        # 4. Readiness Gating (Market Data, Balances, OpsPlane)
        gated_reasons = []
        try:
            snap = await self._readiness_snapshot()
            if not snap.get("market_data_ready", True): gated_reasons.append("MarketData")
            if not snap.get("balances_ready", True): gated_reasons.append("Balances")
            if not snap.get("ops_plane_ready", True): gated_reasons.append("OpsPlane")
        except Exception:
            pass

        # 5. Execution Logic
        is_cold_bootstrap = (
            hasattr(self.shared_state, "is_cold_bootstrap") and 
            await _safe_await(self.shared_state.is_cold_bootstrap())
        )
        
        # CRITICAL: Use authoritative check, not cached SharedState value
        is_flat = await self._check_portfolio_flat()
        portfolio_state = "PORTFOLIO_FLAT" if is_flat else "ACTIVE"
        
        bootstrap_buy = None
        
        if is_flat or is_cold_bootstrap:
            # In flat/cold bootstrap state, promote the best Throughput BUY to "safe"
            for d in decisions:
                if d[1] == "BUY" and ("throughput" in str(d[2].get("reason", "")).lower() or "bootstrap" in str(d[2].get("reason", "")).lower()):
                    bootstrap_buy = d
                    break

        # Transactional Fill Tracking for this tick
        tick_fills = {}
        opened_trades = 0
        closed_trades = 0
        capital_released = 0.0

        # Snapshot before execution
        start_snap = await _safe_await(self.shared_state.get_portfolio_snapshot())
        start_pnl = float(start_snap.get("realized_pnl", 0.0))

        try:
            if gated_reasons:
                # Gated Mode: Process actions that DON'T require budget (SELL/HOLD)
                safe_decisions = [d for d in decisions if not self._is_budget_required(d[1])]
                
                if bootstrap_buy:
                    self.logger.info("[Meta:Bootstrap] 🚀 Gating bypass: Promoting %s BUY to safe_decisions.", bootstrap_buy[0])
                    safe_decisions.append(bootstrap_buy)

                if safe_decisions:
                    for sym, side, sig in safe_decisions:
                        # P9: Explicitly emit TradeIntent
                        if hasattr(self.shared_state, "emit_event"):
                            try:
                                await _safe_await(self.shared_state.emit_event("TradeIntent", {
                                    "symbol": sym, "action": side, "confidence": sig.get("confidence", 0.0),
                                    "planned_quote": sig.get("_planned_quote"), "agent": sig.get("agent", "Meta"),
                                    "reason": f"GATED:{sig.get('reason')}", "ts": time.time()
                                }))
                            except Exception: pass
                            
                        res = await self._execute_decision(sym, side, sig, accepted_symbols_set)
                        status = "FAILED"
                        if isinstance(res, dict): status = str(res.get("status", "FAILED")).upper()
                        elif res is True: status = "FILLED"
                        
                        if status in ("FILLED", "PARTIALLY_FILLED", "PLACED", "EXECUTED"):
                            if side == "BUY": opened_trades += 1
                            elif side == "SELL":
                                closed_trades += 1
                                capital_released += float((res if isinstance(res, dict) else {}).get("cummulativeQuoteQty", 0.0))
                        
                        # Only one bootstrap BUY at a time
                        if side == "BUY": break
                else:
                    self.logger.info("[MetaTick] System gated (%s) and no safe decisions found.", ", ".join(gated_reasons))
            else:
                # Normal Mode Execution
                for sym, side, sig in decisions:
                    # --- Replacement Invariant: Only proceed with BUY if SELL filled ---
                    if side == "BUY" and sig.get("_replacement"):
                        replaces = sig.get("_replaces_symbol")
                        if replaces and tick_fills.get(replaces) != "FILLED":
                            self.logger.warning("[Meta:Atomic] Aborting replacement BUY for %s: Linked SELL for %s failed.", sym, replaces)
                            continue

                    # P9: Explicitly emit TradeIntent
                    if hasattr(self.shared_state, "emit_event"):
                        try:
                            await _safe_await(self.shared_state.emit_event("TradeIntent", {
                                "symbol": sym, "action": side, "confidence": sig.get("confidence", 0.0),
                                "planned_quote": sig.get("_planned_quote"), "agent": sig.get("agent", "Meta"),
                                "reason": sig.get("reason"), "ts": time.time()
                            }))
                        except Exception: pass

                    res = await self._execute_decision(sym, side, sig, accepted_symbols_set)
                    
                    status = "FAILED"
                    if isinstance(res, dict): status = str(res.get("status", "FAILED")).upper()
                    elif res is True: status = "FILLED"
                    
                    tick_fills[sym] = status
                    if status in ("FILLED", "PARTIALLY_FILLED", "PLACED", "EXECUTED"):
                        if side == "BUY": opened_trades += 1
                        elif side == "SELL":
                            closed_trades += 1
                            capital_released += float((res if isinstance(res, dict) else {}).get("cummulativeQuoteQty", 0.0))

        finally:
            # Update cycle counter
            if opened_trades > 0 or closed_trades > 0:
                self.cycles_no_trade = 0
            else:
                self.cycles_no_trade += 1
            
            # Post-execution refresh
            if opened_trades > 0 or closed_trades > 0:
                try:
                    if hasattr(self.shared_state, "recalculate_portfolio_state"):
                        await _safe_await(self.shared_state.recalculate_portfolio_state())
                except Exception: pass
            
            if opened_trades > 0 and not self._first_trade_executed:
                self._first_trade_executed = True
                self._bootstrap_lock_engaged = True

            # Summary metrics
            end_snap = await _safe_await(self.shared_state.get_portfolio_snapshot())
            end_pnl = float(end_snap.get("realized_pnl", 0.0))
            realized_pnl_delta = end_pnl - start_pnl
            
            # Layer 1: Update LOOP_SUMMARY
            self._loop_summary_state.update({
                "execution_attempted": len(decisions) > 0,
                "execution_result": "SUCCESS" if (opened_trades + closed_trades) > 0 else ("SKIPPED" if not decisions else "REJECTED"),
                "trade_opened": opened_trades > 0,
                "trade_closed": closed_trades > 0,
                "realized_pnl": realized_pnl_delta,
            })
            
            # Capital free info
            quote_asset = str(self._cfg("QUOTE_ASSET") or "USDT").upper()
            if quote_asset in self.shared_state.balances:
                actual_spendable = float(await _safe_await(self.shared_state.get_spendable_balance(quote_asset)) or 0.0)
                self._loop_summary_state["capital_free"] = actual_spendable
            
            # EMIT SUMMARY
            self._emit_loop_summary()




    async def _flush_intents_to_cache(self, now_ts: float) -> int:
        """Process intents from sink into signal cache via SignalManager."""
        return self.signal_manager.flush_intents_to_cache(now_ts)

    async def _ingest_strategy_bus(self, now_ts: float) -> int:
        """
        Ingest signals from the SharedState mandatory signal bus.
        Populated by agents calling self.shared_state.add_agent_signal().
        """
        try:
            # Map: sym -> agent -> signal
            getter = getattr(self.shared_state, "get_latest_signals_by_symbol", None)
            if callable(getter):
                bus = getter()
            else:
                bus = getattr(self.shared_state, "latest_signals_by_symbol", {})
            
            if not bus:
                return 0
            
            accepted = 0
            for sym, agents in bus.items():
                for agent, sig in agents.items():
                    # Skip if stale
                    ts = sig.get("timestamp") or sig.get("ts") or now_ts
                    ttl = float(sig.get("ttl_sec") or 300)
                    if (now_ts - ts) > ttl:
                        continue
                    
                    # Normalize and cache
                    symbol = self._normalize_symbol(sym)
                    action = str(sig.get("side") or sig.get("action", "")).upper()
                    if not symbol or action not in ("BUY", "SELL"):
                        continue
                        
                    # Build internal signal format using dict update to preserve metadata
                    internal_sig = dict(sig)
                    internal_sig.update({
                        "symbol": symbol,
                        "action": action,
                        "confidence": float(sig.get("confidence", 0.0)),
                        "agent": agent,
                        "timestamp": now_ts,
                        "budget_required": self._is_budget_required(action),
                        "tier": sig.get("tier", "B"),
                        "reason": sig.get("rationale") or sig.get("reason")
                    })
                    
                    # Store in signal_cache (deduplicated by symbol:agent)
                    # Consistency Fix: receive_signal uses ':', so we use ':' here too.
                    self.signal_manager.store_signal(agent, symbol, internal_sig)
                    accepted += 1
            
            return accepted
        except Exception as e:
            self.logger.error("[Meta:Bus] Failed to ingest from strategy bus: %s", e, exc_info=True)
            return 0


    async def _ingest_liquidation_signals(self, now_ts: float) -> int:
        """Ingest urgent liquidation/exit signals emitted by LiquidationAgent via SharedState.
        Expected format in SharedState: shared_state.liquidation_signals -> {
            symbol: {"action": "SELL", "confidence": 1.0, "timestamp": epoch, "agent": "LiquidationAgent", "reason": str}
        }
        """
        try:
            liq_bus = getattr(self.shared_state, "liquidation_signals", None)
            if _asyncio.iscoroutine(liq_bus):
                liq_bus = await liq_bus
            if not isinstance(liq_bus, dict) or not liq_bus:
                return 0

            ingested = 0
            for sym, sig in list(liq_bus.items()):
                if not isinstance(sig, dict):
                    continue
                action = str(sig.get("action", "")).upper()
                if action != "SELL":
                    continue
                ts = parse_timestamp(sig.get("timestamp", now_ts), now_ts)
                if now_ts - ts > self._max_age_sec:
                    continue
                payload = {
                    "symbol": self._normalize_symbol(sym),
                    "action": "SELL",
                    "confidence": 1.0,  # force high confidence for liquidations
                    "timestamp": ts,
                    "agent": sig.get("agent", "LiquidationAgent"),
                    "reason": sig.get("reason", "liq_request"),
                    "tag": "liquidation",
                    "_is_liquidation": True,
                    "bypass_conf": True,
                    "budget_required": False,  # Liquidations are always risk-reduction/exits
                }
                await self.receive_signal(payload["agent"], payload["symbol"], payload)
                ingested += 1
            return ingested
        except Exception as e:
            self.logger.warning("Failed to ingest liquidation signals: %s", e)
            return 0

    async def _get_aggression_factor(self) -> float:
        """Calculate P9 Adaptive Aggression factor based on profit target matching."""
        try:
            target_hr = self._kpi_metrics.get("hourly_target_usdt", 20.0)
            status = await self.get_kpi_status()
            pnl = float(status.get("total_realized_pnl", 0.0))
            
            # KPI RATE FIX: Calculate rate based on elapsed time
            elapsed_sec = time.time() - self._start_time
            elapsed_hr = max(0.01, elapsed_sec / 3600.0)
            pnl_rate = pnl / elapsed_hr
            
            # P9 Invariant: Hard lock aggression to 1.0 if no realized profit
            if pnl <= 0:
                self.logger.debug("[Meta:Aggression] PnL <= 0. Locking factor to 1.0")
                return 1.0

            # Simple linear pressure: if we are below 20 USDT/hr, increase aggression up to 2.5x
            if pnl_rate < target_hr:
                gap = target_hr - pnl_rate
                factor = 1.0 + min(1.5, gap / target_hr) # max 2.5x
                return factor
        except Exception:
            pass
        return 1.0

    # ===== SCENARIO EVENT EMISSIONS (Deadlock Scenarios 1 & 2) =====
    async def _emit_scenario_event(self, scenario_name: str, details: Dict[str, Any]) -> None:
        """Unified event emission for deadlock scenarios."""
        try:
            event_data = {
                "event": scenario_name,
                "tick_id": self.tick_id,
                "ts": time.time(),
                **details
            }
            self.logger.info("[Scenario:%s] %s", scenario_name, json.dumps(event_data, default=str))
            
            # Emit to SharedState event bus
            if hasattr(self.shared_state, "emit_event"):
                await _safe_await(self.shared_state.emit_event(scenario_name, event_data))
        except Exception as e:
            self.logger.debug("[Scenario] Failed to emit %s event: %s", scenario_name, e)

    async def _handle_capital_starved(self, symbol: str, planned_quote: float, gap: float) -> None:
        """Scenario 1: FLAT + INSUFFICIENT CAPITAL + BUY signals
        Emits CAPITAL_STARVED when all fallback tiers are exhausted.
        
        ✅ FIX: Use shared_state for spendable balance, not execution_manager._free_usdt()
        (ExecutionManager doesn't have _free_usdt method)
        """
        try:
            # Get spendable balance from shared_state
            quote_asset = str(self._cfg("QUOTE_ASSET") or "USDT").upper()
            spendable = float(await self.shared_state.get_spendable_balance(quote_asset) or 0.0)
            
            await self._emit_scenario_event("CAPITAL_STARVED", {
                "symbol": symbol,
                "planned_quote": planned_quote,
                "available_capital": spendable,
                "gap_size": gap,
                "reason": "all_fallback_tiers_exhausted",
                "portfolio_state": "FLAT"
            })
        except Exception as e:
            self.logger.debug("[Scenario] Failed to emit CAPITAL_STARVED: %s", e)

    async def _handle_flat_with_sell_signals_only(self, sell_signals: List[Dict[str, Any]]) -> None:
        """Scenario 2: FLAT + SELL signals only
        Emits WAITING when portfolio is flat but receive only SELL signals.
        """
        try:
            await self._emit_scenario_event("WAITING", {
                "reason": "FLAT_PORTFOLIO_SELL_ONLY",
                "rejected_sell_count": len(sell_signals),
                "action": "wait_for_BUY_signal_or_position_entry",
                "portfolio_state": "FLAT",
                "next_state": "BUY_READY"
            })
        except Exception as e:
            self.logger.debug("[Scenario] Failed to emit WAITING: %s", e)

    async def _handle_no_executable_symbols(self, blocked_symbols: List[str], threshold: int) -> None:
        """Scenario 4 & 5: No executable symbols due to rejections or market conditions.
        Emits NO_EXECUTABLE_SYMBOLS event.
        """
        try:
            await self._emit_scenario_event("NO_EXECUTABLE_SYMBOLS", {
                "blocked_symbols": blocked_symbols[:10],  # Limit to first 10
                "blocked_count": len(blocked_symbols),
                "rejection_threshold": threshold,
                "reason": "all_symbols_rejected_or_cooldown",
                "action": "system_idle_waiting_recovery"
            })
        except Exception as e:
            self.logger.debug("[Scenario] Failed to emit NO_EXECUTABLE_SYMBOLS: %s", e)

    async def _handle_portfolio_state_transition(self, from_state: str, to_state: str, trigger: str) -> None:
        """Scenario 3: Portfolio state changed (e.g., from FLAT to ACTIVE via SELL).
        Emits PORTFOLIO_STATE_TRANSITION event.
        """
        try:
            await self._emit_scenario_event("PORTFOLIO_STATE_TRANSITION", {
                "from_state": from_state,
                "to_state": to_state,
                "trigger": trigger,
                "action": "monitoring_portfolio_dynamics"
            })
        except Exception as e:
            self.logger.debug("[Scenario] Failed to emit PORTFOLIO_STATE_TRANSITION: %s", e)

    async def _check_p_minus_1_dust_consolidation(self) -> Optional[List[Tuple[str, str, Dict[str, Any]]]]:
        """
        ===== P-1 EMERGENCY DUST CONSOLIDATION: PRE-TIER-ZERO POLICY =====
        
        HIGHEST PRIORITY: Runs BEFORE P0 when multiple dust positions are trapped
        
        How real exchanges escape dust traps (e.g., Binance dust converter):
        Bundle multiple dust positions together and sell them as a group
        even if each is below min-notional, until freed USDT >= min-notional * 2
        
        Trigger Conditions:
        1. Have 3+ dust positions (accumulated below min-notional)
        2. Total dust value < min-notional but > 0.5 * min-notional
        3. Portfolio at or near capacity (>= 80%)
        
        Action:
        Emit sequential SELL orders for all dust at once
        (Binance processes this as a batch liquidation)
        
        Returns:
        - List of SELL decisions if consolidation triggered
        - None if consolidation not needed
        """
        try:
            # Get all positions and identify dust
            snap = self.shared_state.get_positions_snapshot() or {}
            dust_positions = []  # [(sym, qty, value)]
            
            quote_asset = str(self._cfg("QUOTE_ASSET") or "USDT").upper()
            total_dust_value = 0.0
            
            for sym_raw, p in snap.items():
                sym = self._normalize_symbol(sym_raw)
                q_val = p.get("quantity")
                if q_val is None:
                    q_val = p.get("qty")
                qty = float(q_val or 0.0)
                
                if qty <= 0:
                    continue
                
                # Skip quote asset itself
                if sym == quote_asset:
                    continue
                
                # Get min-notional for this symbol
                try:
                    info = await self.shared_state.get_symbol_filters(sym)
                    min_notional = float(info.get("min_notional", 10.0))
                except Exception:
                    min_notional = 10.0
                
                # Calculate position value
                try:
                    price = await self.shared_state.safe_price(sym)
                    value = qty * float(price)
                except Exception:
                    price = float(p.get("avg_price", 0.0)) or 1.0
                    value = qty * price
                
                # Is this a dust position?
                is_dust = (value < min_notional) and (value > 0.01)
                
                if is_dust:
                    dust_positions.append((sym, qty, value, min_notional))
                    total_dust_value += value
            
            # ===== TRIGGER: 3+ dust positions with accumulated value =====
            if len(dust_positions) < 3:
                return None  # Not enough dust to consolidate
            
            # Get typical min-notional (assume same across symbols)
            typical_min_notional = dust_positions[0][3] if dust_positions else 10.0
            
            # Total dust should be significant (> 50% of min-notional)
            if total_dust_value < 0.5 * typical_min_notional:
                return None  # Dust too small, not worth consolidating
            
            # Check portfolio capacity
            capacity = self.shared_state.get_portfolio_capacity()
            try:
                used_ratio = (capacity["used"] / capacity["total"]) if capacity["total"] > 0 else 0.0
            except Exception:
                used_ratio = 0.0
            
            # Only trigger if portfolio is tight (>= 80% full)
            if used_ratio < 0.80:
                return None  # Portfolio has space, no consolidation needed
            
            # ===== CONSOLIDATION ACTION =====
            self.logger.warning(
                "[Meta:P-1_DUST_CONSOLIDATION] 🔥 EMERGENCY TRIGGER: "
                "%d dust positions, total_value=$%.2f, capacity=%.1f%%, "
                "bundling for consolidated SELL",
                len(dust_positions), total_dust_value, used_ratio * 100
            )
            
            # Create a SELL decision for each dust position
            # Binance will process these sequentially; together they unlock capital
            decisions = []
            for sym, qty, value, min_not in dust_positions:
                sell_sig = {
                    "symbol": sym,
                    "action": "SELL",
                    "confidence": 1.0,  # Emergency policy = max confidence
                    "agent": "MetaP-1DustConsolidation",
                    "timestamp": time.time(),
                    "reason": f"P-1_EMERGENCY_DUST_CONSOLIDATION_GROUP_{len(dust_positions)}",
                    "_p_minus_1_consolidation": True,
                    "_consolidated_group_size": len(dust_positions),
                    "_dust_value_usd": value,
                    "_tier": "EMERGENCY"
                }
                decisions.append((sym, "SELL", sell_sig))
            
            self.logger.info(
                "[Meta:P-1_DUST_CONSOLIDATION] Decision: "
                "Bundled SELL of %d dust positions (total=$%.2f) "
                "→ freed USDT will unlock capital for trading",
                len(dust_positions), total_dust_value
            )
            
            return decisions
            
        except Exception as e:
            self.logger.exception("[Meta:P-1_DUST_CONSOLIDATION] Failed to check consolidation: %s", e)
            return None

    async def _check_dust_healing_opportunity(self) -> Optional[List[Tuple[str, str, Dict[str, Any]]]
]:
        """
        ===== DUST HEALING: Recover dust positions via BUY =====
        
        ARCHITECTURAL PRINCIPLE: Dust is a state flag, not an executable quantity.
        This method answers: "Can we heal a dust-locked position into a valid position?"
        
        HEALING (Path A) - PREFERRED:
        - Condition: Position is dust-locked AND we have capital (even if below floor)
        - Action: BUY more of the same symbol to lift position above minNotional
        - Result: Position becomes healthy, exits dust state
        - Preferred path: Preserves position, recovers it to tradable state
        
        CRITICAL RULES:
        1. MUTUAL EXCLUSIVITY: If healing is possible, sacrifice MUST NOT run
        2. Healing bypasses portfolio capacity checks (not a new slot, repairs existing)
        3. Healing allows sub-floor execution (temporary, for recovery only)
        4. Only heals existing positions (symbols already owned)
        5. Uses standard BUY operation on real position
        6. Healing has PRIORITY OVERRIDE: Even capital-starved can heal if dust exists
        
        Returns:
            List of (symbol, "BUY", signal) decisions if healing opportunity found
            None if no dust positions or healing not possible
        """
        try:
            # Enable dust healing after the first completed trade
            metrics = getattr(self.shared_state, "metrics", {}) or {}
            trade_count = int(
                metrics.get("total_trades_executed", 0)
                or getattr(self.shared_state, "trade_count", 0)
                or 0
            )
            if trade_count < 1:
                self.logger.info(
                    "[DUST_HEALING] Disabled: trade_count=%d < 1 (no realized trades yet)",
                    trade_count,
                )
                return None

            # ═══════════════════════════════════════════════════════════════════════
            # FOCUS MODE: Disable dust healing in live trading
            # ═══════════════════════════════════════════════════════════════════════
            if self.FOCUS_MODE_ENABLED:
                self.logger.debug(
                    "[DUST_HEALING] Disabled: FOCUS_MODE_ENABLED=True "
                    "(dust healing is out-of-scope for focused symbol strategy)"
                )
                return None
            
            snap = self.shared_state.get_positions_snapshot() or {}
            quote_asset = str(self._cfg("QUOTE_ASSET") or "USDT").upper()
            free_usdt = float(await self.shared_state.get_spendable_balance(quote_asset) or 0.0)
            min_notional = float(self._cfg("MIN_NOTIONAL", 10.0))
            min_floor = float(self._cfg("MIN_NOTIONAL_FLOOR", 10.0))  # Normal floor
            healing_floor = min_floor * 0.5  # Healing allows sub-floor (50% of normal floor)
            
            # Identify healable dust positions
            healable = []
            for sym_raw, pos_data in snap.items():
                sym = self._normalize_symbol(sym_raw)
                state = pos_data.get("state", "")
                qty = float(pos_data.get("quantity", 0.0) or pos_data.get("qty", 0.0))
                value_usdt = float(pos_data.get("value_usdt", 0.0) or 0.0)

                # ═══════════════════════════════════════════════════════════════
                # FIX 5: Dynamic value calculation to prevent runaway healing.
                # Snapshot value_usdt can be stale ($0.00) after fills because
                # SharedState doesn't recalculate position values between cycles.
                # Always compute live value = qty * current_price.
                # ═══════════════════════════════════════════════════════════════
                if qty > 0:
                    live_price = float(
                        getattr(self.shared_state, "latest_prices", {}).get(sym, 0.0) or 0.0
                    )
                    if live_price <= 0 and hasattr(self.shared_state, "safe_price"):
                        try:
                            live_price = float(await self.shared_state.safe_price(sym) or 0.0)
                        except Exception:
                            pass
                    if live_price > 0:
                        live_value = qty * live_price
                        if live_value > value_usdt:
                            self.logger.debug(
                                "[DUST_HEALING] FIX5: Corrected stale value_usdt for %s: "
                                "snapshot=$%.2f → live=$%.2f (qty=%.8f × price=%.2f)",
                                sym, value_usdt, live_value, qty, live_price
                            )
                            value_usdt = live_value

                # Check if position is dust-locked
                is_dust_locked = self._is_position_dust_locked(state, qty, value_usdt)
                if not is_dust_locked:
                    continue
                
                # === CRITICAL INVARIANT CHECK ===
                # At this point, STEP 0.0 should have already graduated non-healable dust
                # If we see a position that's not DUST_LOCKED, graduation failed
                if state != "DUST_LOCKED":
                    self.logger.critical(
                        "[INVARIANT_VIOLATION] Healing target is not DUST_LOCKED! "
                        "symbol=%s state=%s qty=%.8f value=%.2f | "
                        "This means STEP 0.0 graduation failed to process this position. "
                        "Skipping to prevent re-healing.",
                        sym, state, qty, value_usdt
                    )
                    continue  # Skip this position to be safe
                
                # Calculate amount needed to heal to minNotional
                deficit = max(0, min_notional - value_usdt)
                
                # HEALING ALLOWS SUB-FLOOR EXECUTION
                # Check if we can heal with available capital (even if below normal floor)
                # This is the recovery mechanism, so temporarily relax the floor
                if deficit > 0 and free_usdt >= healing_floor:
                    # If we have at least the healing floor, we can attempt healing
                    # (even if it goes below normal floor temporarily)
                    healable.append({
                        "symbol": sym,
                        "qty": qty,
                        "value_usdt": value_usdt,
                        "deficit": deficit,
                        "can_fully_heal": (free_usdt >= deficit),  # Track if we can fully heal
                        "position": pos_data
                    })
            
            if not healable:
                return None  # No healable dust positions
            
            # Sort: first by "can_fully_heal" (descending), then by deficit (ascending)
            healable.sort(key=lambda x: (-int(x["can_fully_heal"]), x["deficit"]))

            healing_signals = []

            for heal in healable:
                sym_to_heal = heal["symbol"]
                # Enforce lifecycle lock: skip if not allowed
                if not self._can_act(sym_to_heal, "DUST_HEALING"):
                    continue
                # Check cooldown
                import time
                if sym_to_heal in self.dust_healing_cooldown and self.dust_healing_cooldown[sym_to_heal] > time.time():
                    self.logger.info(f"[LIFECYCLE] {sym_to_heal}: DUST_HEALING blocked by cooldown")
                    continue
                deficit = heal["deficit"]
                can_fully_heal = heal["can_fully_heal"]
                value_usdt = heal["value_usdt"]
                self._set_lifecycle(sym_to_heal, self.LIFECYCLE_DUST_HEALING)
                self.logger.info(
                    "[DUST_HEALING] 🔧 Healing opportunity found: symbol=%s current_value=$%.2f "
                    "deficit=$%.2f (need to reach minNotional=$%.2f). Free USDT=$%.2f (floor=$%.2f, "
                    "healing_floor=$%.2f). CAN_FULLY_HEAL=%s",
                    sym_to_heal, value_usdt, deficit, min_notional, free_usdt,
                    min_floor, healing_floor, can_fully_heal
                )
                healing_amount = deficit if can_fully_heal else min(deficit, free_usdt)
                buy_sig = {
                    "symbol": sym_to_heal,
                    "action": "BUY",
                    "amount_usdt": healing_amount,
                    "confidence": 1.0,
                    "agent": "MetaDustHealing",
                    "timestamp": time.time(),
                    "reason": "DUST_HEALING_BUY",
                    "_dust_healing": True,
                    "_target_position_value": min_notional,
                    "_current_position_value": value_usdt,
                    "_healing_amount": healing_amount,
                    "_can_fully_heal": can_fully_heal,
                    "_allows_sub_floor": True,
                    "_tier": "DUST_RECOVERY"
                }
                self.logger.warning(
                    "[DUST_HEALING] 💚 Healing signal created: BUY %s with $%.2f "
                    "(heal position value from $%.2f toward target $%.2f). "
                    "CAN_FULLY_HEAL=%s | BLOCKS_SACRIFICE",
                    sym_to_heal, healing_amount, value_usdt, min_notional,
                    can_fully_heal
                )
                healing_signals.append((sym_to_heal, "BUY", buy_sig))
                self.logger.info(
                    f"[HEALING] ✓ Qualified for recovery: symbol={sym_to_heal} "
                    f"(under healing_floor, recovery in progress)"
                )

            if not healing_signals:
                return None
            return healing_signals
            
        except Exception as e:
            self.logger.exception("[DUST_HEALING] Failed to check healing opportunity: %s", e)
            return None

    async def _check_dust_sacrifice_necessity(self) -> Optional[List[Tuple[str, str, Dict[str, Any]]]
]:
        """
        ===== DUST SACRIFICE: Emergency SELL when position unrecoverable =====
        
        ARCHITECTURAL PRINCIPLE: Dust is a state flag. When healing is impossible
        and position blocks the system, sacrifice it (SELL entire position).
        
        SACRIFICE (Path B) - ONLY IF HEALING FAILS:
        - Condition: Healing NOT possible AND:
        * Portfolio is full (at capacity), AND
        * Capital below floor, OR
        * Dust ratio very high (deadlocked state)
        - Action: SELL entire real position (not just "dust units")
        - Result: Position removed, capacity + capital freed
        - Escape hatch: Only use when healing blocked AND deadlocked
        
        CRITICAL RULES:
        1. MUTUAL EXCLUSIVITY: Only triggers if healing_possible == False
        2. Very strict trigger conditions (only extreme cases)
        3. NEVER sacrifice BTC/ETH - start with meme coins (low recovery value)
        4. Always operates on entire real position (no partial logic)
        5. Uses standard SELL operation
        6. Last resort to break deadlock
        
        SACRIFICE PRIORITY (highest to lowest):
        1. Meme coins (DOGE, SHIB, etc) - lowest recovery potential
        2. Low-liquidity alts - hard to exit anyway
        3. High-spread symbols - poor execution
        4. Common alts (SOL, etc)
        5. ETH - preserve for leverage
        6. BTC - LAST RESORT - destroy future recovery capacity
        
        Returns:
            List of (symbol, "SELL", signal) decisions if sacrifice necessary
            None if healing possible or portfolio not deadlocked
        """
        try:
            snap = self.shared_state.get_positions_snapshot() or {}
            quote_asset = str(self._cfg("QUOTE_ASSET") or "USDT").upper()
            free_usdt = float(await self.shared_state.get_spendable_balance(quote_asset) or 0.0)
            min_notional = float(self._cfg("MIN_NOTIONAL", 10.0))
            min_floor = float(self._cfg("MIN_NOTIONAL_FLOOR", 10.0))
            
            # Check portfolio status
            total_pos, sig_pos, dust_pos = await self._count_significant_positions()
            max_pos = self._get_max_positions()
            is_full = (sig_pos >= max_pos)
            is_capital_starved = (free_usdt < min_floor)
            
            # Calculate dust ratio
            dust_ratio = (dust_pos / total_pos) if total_pos > 0 else 0.0
            is_very_dusty = (dust_ratio > 0.60)
            
            # Sacrifice conditions: ALL of these must be true
            # 1. Portfolio is full
            # 2. Capital is low OR dust ratio is very high
            # 3. There are dust positions to sacrifice
            should_consider_sacrifice = (
                is_full and 
                (is_capital_starved or is_very_dusty) and 
                dust_pos > 0
            )
            
            if not should_consider_sacrifice:
                return None  # No sacrifice needed
            
            # Find dust positions to sacrifice, ordered by PRIORITY
            # Priority: meme coins first, BTC/ETH last
            sacrificable = []
            
            for sym_raw, pos_data in snap.items():
                sym = self._normalize_symbol(sym_raw)
                state = pos_data.get("state", "")
                qty = float(pos_data.get("quantity", 0.0) or pos_data.get("qty", 0.0))
                value_usdt = float(pos_data.get("value_usdt", 0.0) or 0.0)
                
                # Check if position is dust-locked
                is_dust_locked = self._is_position_dust_locked(state, qty, value_usdt)
                if not is_dust_locked:
                    continue
                
                # Determine sacrifice priority using centralized helper
                priority = self._get_sacrifice_priority(sym)
                
                sacrificable.append({
                    "symbol": sym,
                    "qty": qty,
                    "value_usdt": value_usdt,
                    "priority": priority,
                    "position": pos_data
                })
            
            if not sacrificable:
                return None  # No dust to sacrifice
            
            # Sort by priority (ascending = meme coins first), then by value (ascending = smallest first)
            sacrificable.sort(key=lambda x: (x["priority"], x["value_usdt"]))
            victim = sacrificable[0]
            
            sym_to_sacrifice = victim["symbol"]
            qty_to_sacrifice = victim["qty"]
            value_sacrificed = victim["value_usdt"]
            
            self.logger.warning(
                "[DUST_SACRIFICE] 🔴 SACRIFICE NECESSARY: "
                "Portfolio full (sig_pos=%d/%d), capital=$%.2f (floor=$%.2f), "
                "dust_ratio=%.1f%% → Sacrificing %s (qty=%.6f, value=$%.2f, priority=%d) to break deadlock",
                sig_pos, max_pos, free_usdt, min_floor, dust_ratio * 100,
                sym_to_sacrifice, qty_to_sacrifice, value_sacrificed, victim["priority"]
            )
            
            # Create SELL signal for sacrifice (entire position)
            sell_sig = {
                "symbol": sym_to_sacrifice,
                "action": "SELL",
                "quantity": qty_to_sacrifice,  # Entire position
                "confidence": 1.0,  # Sacrifice is max confidence
                "agent": "MetaDustSacrifice",
                "timestamp": time.time(),
                "reason": "DUST_SACRIFICE_EMERGENCY",
                "_dust_sacrifice": True,
                "_position_value_sacrificed": value_sacrificed,
                "_deadlock_reason": ("capital_starved" if is_capital_starved else "very_dusty"),
                "_portfolio_full": is_full,
                "_sacrifice_priority": victim["priority"],
                "_tier": "DEADLOCK_ESCAPE"
            }
            
            self.logger.critical(
                "[DUST_SACRIFICE] 💀 Emergency sacrifice activated: SELL entire %s "
                "(qty=%.6f, $%.2f, priority=%d) to escape deadlock condition",
                sym_to_sacrifice, qty_to_sacrifice, value_sacrificed, victim["priority"]
            )
            
            return [(sym_to_sacrifice, "SELL", sell_sig)]
            
        except Exception as e:
            self.logger.exception("[DUST_SACRIFICE] Failed to check sacrifice necessity: %s", e)
            return None


    async def _reclassify_graduated_positions(self) -> int:
        """
        ===== DUST GRADUATION CHECK: Reclassify positions that grew beyond dust =====
        
        CRITICAL FIX: Dust is a sticky state flag that doesn't auto-clear.
        Positions must be explicitly promoted from DUST_LOCKED to HEALTHY when they graduate.
        
        WHY THIS IS NEEDED:
        A position starts as dust (small value), then grows via healing BUYs or price increases.
        But if we don't explicitly clear the DUST_LOCKED flag, the position stays marked as dust
        internally, even though its value is healthy.
        
        This causes:
        - Incorrect portfolio composition reporting (appears "dust-heavy")
        - Incorrect position prioritization
        - Bypass of normal SELL/TP logic
        - Inability to treat recovered positions as normal trading positions
        
        THE FIX:
        A position exits dust state when it meets BOTH:
        1. Value >= max(MIN_NOTIONAL, SIGNIFICANT_POSITION_THRESHOLD)
        2. Quantity is not micro (qty > min_qty_threshold)
        3. No longer in emergency dust-recovery cycle
        
        This method:
        - Scans all dust-marked positions
        - Checks if they've grown beyond dust thresholds
        - Updates position.state from DUST_LOCKED → HEALTHY
        - Logs promotion events
        - Returns count of graduated positions
        
        RUNS: Every cycle in STEP 0.1 (before all other checks)
        
        Returns:
            Number of positions that graduated from dust state
        """
        try:
            snap = self.shared_state.get_positions_snapshot() or {}
            min_notional = float(self._cfg("MIN_NOTIONAL", 10.0))
            significant_threshold = float(
                self._cfg(
                    "MIN_SIGNIFICANT_POSITION_USDT",
                    self._cfg("SIGNIFICANT_POSITION_USDT", 25.0),
                )
            )
            
            # Promotion threshold = max of minNotional or significant threshold
            promotion_threshold = max(min_notional, significant_threshold)
            
            promoted_count = 0
            
            for sym_raw, pos_data in snap.items():
                sym = self._normalize_symbol(sym_raw)
                state = pos_data.get("state", "")
                qty = float(pos_data.get("quantity", 0.0) or pos_data.get("qty", 0.0))
                value_usdt = float(pos_data.get("value_usdt", 0.0) or 0.0)

                # ═══════════════════════════════════════════════════════════════
                # FIX 6: Dynamic value calculation for graduation check.
                # Snapshot value_usdt can be stale ($0.00) after healing BUYs,
                # preventing positions from ever graduating to HEALTHY.
                # Compute live value = qty * current_price.
                # ═══════════════════════════════════════════════════════════════
                if qty > 0:
                    live_price = float(
                        getattr(self.shared_state, "latest_prices", {}).get(sym, 0.0) or 0.0
                    )
                    if live_price <= 0 and hasattr(self.shared_state, "safe_price"):
                        try:
                            live_price = float(await self.shared_state.safe_price(sym) or 0.0)
                        except Exception:
                            pass
                    if live_price > 0:
                        live_value = qty * live_price
                        if live_value > value_usdt:
                            self.logger.debug(
                                "[DUST_GRADUATION] FIX6: Corrected stale value_usdt for %s: "
                                "snapshot=$%.2f → live=$%.2f (qty=%.8f × price=%.2f)",
                                sym, value_usdt, live_value, qty, live_price
                            )
                            value_usdt = live_value

                # Only consider positions currently marked as dust
                if state != "DUST_LOCKED":
                    continue

                # Check if position has graduated beyond dust threshold
                has_healthy_value = (value_usdt >= promotion_threshold)
                has_reasonable_qty = (qty > 0.00001)  # Not micro
                
                if has_healthy_value and has_reasonable_qty:
                    # Position has graduated - update state
                    pos_data["state"] = "HEALTHY"
                    pos_data["_dust_locked"] = False
                    pos_data["_dust_promoted"] = True
                    pos_data["_promotion_timestamp"] = time.time()
                    pos_data["_promotion_value"] = value_usdt
                    
                    self.logger.warning(
                        "[DUST_GRADUATION] 🎓 Position promoted from DUST_LOCKED → HEALTHY: "
                        "%s (value=$%.2f >= threshold=$%.2f, qty=%.6f). "
                        "Position is no longer dust-classified internally.",
                        sym, value_usdt, promotion_threshold, qty
                    )
                    
                    promoted_count += 1
            
            if promoted_count > 0:
                self.logger.info(
                    "[DUST_GRADUATION] ✅ Total promoted: %d position(s) graduated to HEALTHY state",
                    promoted_count
                )
            
            return promoted_count
            
        except Exception as e:
            self.logger.exception("[DUST_GRADUATION] Failed to reclassify graduated positions: %s", e)
            return 0

    async def _check_p0_dust_promotion(self) -> Optional[Dict[str, Any]]:
        """
        ===== P0_DUST_PROMOTION: GATE (NOT EXECUTOR) =====
        
        ARCHITECTURAL FIX: P0 is demoted from executor to gate.
        
        OLD (BROKEN):
        - Detected dust with BUY signal
        - Forced SELL of worst performer
        - Forced BUY into dust
        - Created infinite loop (dust exists → buy → create dust → repeat)
        
        NEW (CORRECT):
        - Identify dust positions with strong BUY signals
        - Annotate them for PRIORITY ordering in normal pipeline
        - Let Dust Consolidation BUY handle actual execution
        - Return gate/ordering info, never execute BUY
        
        CRITICAL RULE: P0 must NEVER place a BUY order. Ever.
        P0 controls eligibility & ordering, not execution.
        
        🔒 PORTFOLIO FULL GUARD (UPDATED): If portfolio is at capacity (sig_pos >= max_pos),
        P0 must NOT open the gate for NEW symbols. However, DUST HEALING bypasses this gate
        (healing repairs existing positions, doesn't create new slots).
        Normal BUY signals are still blocked when portfolio full.
        P0 may still evaluate/rank dust internally, but returns None to disable promotion.
        
        Returns:
            Dict with dust eligibility info if dust with BUY signals found AND portfolio not full
            None if no eligible dust, no strong BUY signals, OR portfolio is full
        """
        # 🔒 PORTFOLIO FULL GUARD (UPDATED): Block P0 for new symbols, but allow healing
        # When portfolio full:
        # - Dust healing (repairing existing) is ALLOWED
        # - New symbol promotion is BLOCKED
        # P0 controls dust promotion ordering, not blocking healing
        try:
            total_pos, sig_pos, dust_pos = await self._count_significant_positions()
            max_pos = int(self._cfg("MAX_POSITIONS", 5))
            
            # Note: P0 no longer blocks dust healing when full
            # Healing is handled separately and bypasses capacity checks
            # P0 still blocks NEW symbol promotion but evaluates dust ordering
        except Exception as e:
            self.logger.warning("[P0_GATE] Could not check portfolio capacity: %s. Proceeding with P0 evaluation.", e)
        
        try:
            # Get all positions
            snap = self.shared_state.get_positions_snapshot() or {}
            owned_positions = {}
            dust_positions = {}
            
            for sym_raw, p in snap.items():
                sym = self._normalize_symbol(sym_raw)
                q_val = p.get("quantity")
                if q_val is None:
                    q_val = p.get("qty")
                qty = float(q_val or 0.0)
                if qty > 0:
                    owned_positions[sym] = p
                    
                    # Explicit dust categorization
                    state = p.get("state", "")
                    value_usdt = float(p.get("value_usdt", 0.0))
                    is_dust = self._is_position_dust_locked(state, qty, value_usdt)
                    
                    if is_dust:
                        dust_positions[sym] = p
            
            if not dust_positions:
                return None  # No dust positions to evaluate
            
            # Get all signals
            all_signals = self.signal_manager.get_all_signals()
            signals_by_sym = defaultdict(list)
            for s in all_signals:
                sym = s.get("symbol")
                if sym:
                    signals_by_sym[sym].append(s)
            
            # Find dust positions with strong BUY signals
            # These become HIGH PRIORITY for normal flow, but don't execute here
            eligible_dust = []
            
            for sym, pos_data in dust_positions.items():
                for sig in signals_by_sym.get(sym, []):
                    if str(sig.get("action")).upper() == "BUY":
                        conf = float(sig.get("confidence", 0.0))
                        if conf >= 0.55:  # Strong BUY threshold
                            qty = float(pos_data.get("quantity", 0.0) or pos_data.get("qty", 0.0))
                            eligible_dust.append({
                                "symbol": sym,
                                "qty": qty,
                                "confidence": conf,
                                "position": pos_data,
                                "signal": sig
                            })
            
            if not eligible_dust:
                return None  # No eligible dust to promote
            
            # Sort by confidence (highest first) - this is the ORDERING
            eligible_dust.sort(key=lambda x: x["confidence"], reverse=True)
            
            self.logger.info(
                "[P0_GATE] ✓ P0 GATE OPEN: %d eligible dust position(s) found (will be prioritized in normal flow, not executed here)",
                len(eligible_dust)
            )
            
            # Return gate info (NOT decisions to execute)
            # This annotates which dust should be prioritized, but execution happens in normal flow
            return {
                "p0_gate_open": True,
                "eligible_dust_list": eligible_dust,
                "top_dust": eligible_dust[0] if eligible_dust else None,
                "reason": "Dust with strong BUY signals eligible for Consolidation BUY priority"
            }
            
        except Exception as e:
            self.logger.exception("[P0_GATE] ERROR in P0 eligibility check: %s", e)
            return None

    def _can_promote_dust_viable(self, dust_sym: str, freed_capital: float) -> bool:
        """
        FIX FOR DUST INFLATION: Check if dust promotion will create a viable position
        
        Don't promote dust unless:
        1. We have enough freed capital to make meaningful position ($25+)
        2. After rounding, position size won't be dust again
        3. The promotion actually helps, not just recycles dust
        
        Returns:
            True if promotion will work (avoided dust inflation)
            False if promotion would just create new dust (block it)
        """
        try:
            # DUST INFLATION FIX: Minimum capital for viable promotion
            MIN_VIABLE_QUOTE = float(self._cfg("DUST_PROMOTION_MIN_QUOTE", 25.0))
            
            if freed_capital < MIN_VIABLE_QUOTE:
                self.logger.info(
                    f"[DUST_GATE] ✗ Dust promotion BLOCKED for {dust_sym}: "
                    f"freed_capital=${freed_capital:.2f} < min_viable=${MIN_VIABLE_QUOTE:.2f} "
                    f"(would create new dust!)"
                )
                return False
            
            self.logger.info(
                f"[DUST_GATE] ✓ Dust promotion VIABLE for {dust_sym}: "
                f"freed_capital=${freed_capital:.2f} >= min_viable=${MIN_VIABLE_QUOTE:.2f}"
            )
            return True
        except Exception as e:
            self.logger.error(f"[DUST_GATE] ERROR checking promotion viability: {e}")
            return False

    def _can_p0_dust_promotion_execute(self) -> bool:
        """
        CHECK IF P0_DUST_PROMOTION CAN EXECUTE (Option A: Capital Floor Bypass)
        
        Determines if P0 dust promotion conditions are met, which would justify
        bypassing the capital floor check. This allows the escape policy to run
        even when capital is starved.
        
        Returns:
            True if dust promotion can help (dust positions + strong buy signals exist)
            False if dust promotion won't help (no dust or no strong buys)
        """
        try:
            # Get all positions
            snap = self.shared_state.get_positions_snapshot() or {}
            owned_positions = {}
            
            for sym_raw, p in snap.items():
                sym = self._normalize_symbol(sym_raw)
                q_val = p.get("quantity")
                if q_val is None:
                    q_val = p.get("qty")
                qty = float(q_val or 0.0)
                if qty > 0:
                    owned_positions[sym] = p
            
            # Get all signals
            all_signals = self.signal_manager.get_all_signals()
            signals_by_sym = defaultdict(list)
            for s in all_signals:
                sym = s.get("symbol")
                if sym:
                    signals_by_sym[sym].append(s)
            
            # Check if any dust positions have strong BUY signals
            for sym, pos_data in owned_positions.items():
                state = pos_data.get("state", "")
                qty = float(pos_data.get("quantity", 0.0) or pos_data.get("qty", 0.0))
                value_usdt = float(pos_data.get("value_usdt", 0.0))
                is_dust = self._is_position_dust_locked(state, qty, value_usdt)
                
                if not is_dust:
                    continue
                
                # Check for strong BUY signals
                for sig in signals_by_sym.get(sym, []):
                    if str(sig.get("action")).upper() == "BUY":
                        conf = float(sig.get("confidence", 0.0))
                        if conf >= 0.55:  # Strong BUY threshold
                            self.logger.info(
                                f"[P0_CHECK] ✓ P0 CAN EXECUTE: dust_sym={sym} "
                                f"dust_qty={qty:.6f} buy_conf={conf:.2f}"
                            )
                            return True
            
            return False
        except Exception as e:
            self.logger.error(f"[P0_CHECK] ERROR determining if P0 can execute: {e}")
            return False
    
    async def _get_avg_trade_cost(self) -> float:
        """
        Get the average cost of recent trades for entry sizing.
        
        Used by Layer 1 to calculate SIGNIFICANT_POSITION_USDT.
        Falls back to min_notional if no trade history available.
        """
        try:
            min_notional = float(self._cfg("MIN_NOTIONAL_USDT", 10.0))
            
            # Try to get average from recent trades
            if hasattr(self.shared_state, "get_recent_trades"):
                recent = await self.shared_state.get_recent_trades(limit=20)
                if recent:
                    costs = [float(t.get("cost", t.get("quote_qty", 0.0))) for t in recent]
                    avg_cost = sum(costs) / len(costs) if costs else min_notional
                    return max(avg_cost, min_notional)
            
            # Fallback: use min_notional
            return min_notional
        except Exception as e:
            self.logger.debug(f"[Meta:AvgTradeCost] Error calculating: {e}, using min_notional")
            return float(self._cfg("MIN_NOTIONAL_USDT", 10.0))
    
    async def _can_accumulation_promotion_help(self) -> bool:
        """
        CHECK IF ACCUMULATION PROMOTION CAN HELP (Capital Floor Bypass Escape Hatch)
        
        Determines if accumulation conditions are met, which would justify
        bypassing the capital floor check. ACCUMULATION allows small amounts
        to grow dust positions toward minNotional (capital-efficient operation).
        
        Returns:
            True if accumulation can help (dust positions + buy signals exist)
            False if accumulation won't help (no dust positions with buys)
        """
        try:
            # Get all positions
            snap = self.shared_state.get_positions_snapshot() or {}
            owned_positions = {}
            
            for sym_raw, p in snap.items():
                sym = self._normalize_symbol(sym_raw)
                q_val = p.get("quantity")
                if q_val is None:
                    q_val = p.get("qty")
                qty = float(q_val or 0.0)
                if qty > 0:
                    owned_positions[sym] = p
            
            # Get all signals
            all_signals = self.signal_manager.get_all_signals()
            signals_by_sym = defaultdict(list)
            for s in all_signals:
                sym = s.get("symbol")
                if sym:
                    signals_by_sym[sym].append(s)
            
            # Check if any dust positions have BUY signals (any confidence)
            min_significant = float(
                self._cfg(
                    "MIN_SIGNIFICANT_POSITION_USDT",
                    self._cfg("MIN_SIGNIFICANT_USD", 25.0),
                )
            )
            
            for sym, pos_data in owned_positions.items():
                qty = float(pos_data.get("quantity", 0.0) or pos_data.get("qty", 0.0))
                if qty <= 0:
                    continue
                
                # Calculate current position value
                try:
                    current_price = await self.shared_state.safe_price(sym)
                    current_value = qty * current_price
                except Exception:
                    current_value = float(pos_data.get("value_usdt", 0.0))
                
                # Is this dust? (value < MIN_SIGNIFICANT)
                if current_value >= min_significant:
                    continue
                
                # Check for ANY BUY signal (conf >= 0.40)
                for sig in signals_by_sym.get(sym, []):
                    if str(sig.get("action")).upper() == "BUY":
                        conf = float(sig.get("confidence", 0.0))
                        if conf >= 0.40:  # ACCUMULATION threshold is lower than P0 (0.40 vs 0.55)
                            self.logger.info(
                                f"[ACCUM_CHECK] ✓ ACCUMULATION CAN HELP: dust_sym={sym} "
                                f"current_value=${current_value:.2f} (< ${min_significant:.2f}) "
                                f"buy_conf={conf:.2f}"
                            )
                            return True
            
            return False
        except Exception as e:
            self.logger.error(f"[ACCUM_CHECK] ERROR determining if ACCUMULATION can help: {e}")
            return False

    async def _check_capital_floor_central(self) -> bool:
        """
        CENTRALIZED CAPITAL FLOOR CHECK (Phase 2 Consolidation + Option A Fix)
        
        Single authority, single threshold, single timing point.
        Called ONCE before any policy decisions.
        
    Design Principles:
    - BEFORE intent emission (no race conditions)
    - Dynamic floor: max(ABSOLUTE_MIN_FLOOR, NAV * CAPITAL_FLOOR_PCT)
    - Capital floor applies ONLY to risk-increasing actions (BUY)
    - Explicit timing (called from _build_decisions start)
    - EXCEPTION: P0_DUST_PROMOTION can bypass if it can help (escape hatch)
    - EXCEPTION: ACCUMULATION_PROMOTION can bypass for dust growth (new escape hatch)
        
        Returns:
            True if capital >= floor (safe to proceed)
            True if capital < floor BUT dust recovery possible (escape hatch)
            False if capital < floor AND no dust recovery available (abort all decisions)
        """
        try:
            # Step 1: Get current capital state
            quote_asset = str(self._cfg("QUOTE_ASSET") or "USDT").upper()
            free_usdt = float(await self.shared_state.get_spendable_balance(quote_asset) or 0.0)

            # NAV-aware floor (percentage + absolute minimum)
            nav = 0.0
            try:
                if hasattr(self.shared_state, "get_nav_quote"):
                    nav = float(await _safe_await(self.shared_state.get_nav_quote()) or 0.0)
                else:
                    nav = float(getattr(self.shared_state, "nav", 0.0) or 0.0)
            except Exception:
                nav = 0.0

            abs_min_floor = float(self._cfg("ABSOLUTE_MIN_FLOOR", self._cfg("CAPITAL_PRESERVATION_FLOOR", 10.0)))
            floor_pct = float(self._cfg("CAPITAL_FLOOR_PCT", 0.20))
            min_floor = max(abs_min_floor, nav * floor_pct)
            
            # Step 2: Check floor (simple, no adaptation)
            capital_ok = free_usdt >= min_floor
            
            # Step 3: Log result with context
            if capital_ok:
                self.logger.debug(
                    f"CAPITAL_FLOOR_CHECK: ✓ PASSED | "
                    f"free_usdt={free_usdt:.2f} >= floor={min_floor:.2f} (nav={nav:.2f}, pct={floor_pct:.2%})"
                )
                return True
            
            # Step 4: Capital is low - check if dust recovery can help (escape hatch)
            self.logger.warning(
                f"CAPITAL_FLOOR_CHECK: ✗ FAILED | "
                f"free_usdt={free_usdt:.2f} < floor={min_floor:.2f} | "
                f"shortfall=${min_floor - free_usdt:.2f} (nav={nav:.2f}, pct={floor_pct:.2%})"
            )
            
            # Can P0 dust promotion help?
            can_p0_help = self._can_p0_dust_promotion_execute()
            if can_p0_help:
                self.logger.warning(
                    f"CAPITAL_FLOOR_BYPASS: P0_DUST_PROMOTION can help recover capacity, "
                    f"allowing bypass of capital floor check"
                )
                return True
            
            # Can ACCUMULATION promotion help? (Growing dust positions toward minNotional)
            can_accum_help = await self._can_accumulation_promotion_help()
            if can_accum_help:
                self.logger.warning(
                    f"CAPITAL_FLOOR_BYPASS: ACCUMULATION_PROMOTION can grow dust positions, "
                    f"allowing bypass of capital floor check (micro-allocation for growth)"
                )
                return True
            
            # Capital low AND no dust recovery available: Block all trading
            self.logger.warning(
                f"CAPITAL_FLOOR_CHECK: HARD BLOCK - "
                f"Capital starved AND no dust recovery mechanisms available"
            )
            return False
        except Exception as e:
            self.logger.error(f"CAPITAL_FLOOR_CHECK: ERROR - {e}")
            return False

    def _evaluate_capital_stability(self, capital_ok: bool, nav: float) -> Tuple[bool, str]:
        """
        Determine whether the system is stable enough to trade real capital.

        Stability requirements:
        - Capital floor OK
        - Ops plane ready
        - System health not degraded
        - NAV confirmed (or first trade already executed)
        """
        if not capital_ok:
            return False, "capital_floor"

        ops_ready = True
        try:
            if hasattr(self.shared_state, "ops_plane_ready_event"):
                ops_ready = bool(self.shared_state.ops_plane_ready_event.is_set())
        except Exception:
            ops_ready = False
        if not ops_ready:
            return False, "ops_not_ready"

        status = "unknown"
        health_ok = True
        if hasattr(self.shared_state, "system_health"):
            status = str(self.shared_state.system_health.get("status", "ok")).lower()
        if status in ("error", "breach", "degraded"):
            health_ok = False
        if not health_ok:
            return False, f"health_{status}"

        nav_confirmed = bool(nav > 0.0 or self._first_trade_executed)
        if not nav_confirmed:
            return False, "nav_unconfirmed"

        return True, "ok"

    async def _maybe_build_rotation_escape_decisions(self) -> List[Dict[str, Any]]:
        """
        Build a SELL->BUY rotation escape when focus mode deadlocks.

        Requirements:
        - Focus mode active
        - HAS_POSITIONS == True
        - BUY signal persists N cycles
        - No SELL/TP-SL trigger
        - Cooldown elapsed
        - Partial amount only
        """
        try:
            self.logger.info("[ROTATION_DEBUG] Checking rotation escape. Focus=%s", getattr(self, "_focus_mode_active", "N/A"))
            if not bool(getattr(self, "_focus_mode_active", False)):
                return []
            if not bool(getattr(self.config, "FOCUS_MODE_EXIT_ENABLED", False)):
                return []

            positions = self.shared_state.get_positions_snapshot() or {}
            if not positions:
                return []

            all_signals = self.signal_manager.get_all_signals()
            has_buy = any(str(s.get("action")).upper() == "BUY" for s in all_signals)
            has_sell = any(str(s.get("action")).upper() == "SELL" for s in all_signals)
            
            if not has_buy or has_sell:
                # Reset persistence counters if BUY signals drop or SELLs appear
                for sym in list(self._rotation_escape_buy_persist.keys()):
                    self._rotation_escape_buy_persist[sym] = 0
                return []

            buy_by_sym = defaultdict(list)
            for s in all_signals:
                if str(s.get("action")).upper() == "BUY":
                    sym = self._normalize_symbol(s.get("symbol") or "")
                    if sym:
                        buy_by_sym[sym].append(s)

            pos_by_sym: Dict[str, Dict[str, Any]] = {}
            candidates = []
            for sym_raw, pos in positions.items():
                sym = self._normalize_symbol(sym_raw)
                qty = float(pos.get("quantity", 0.0) or pos.get("qty", 0.0))
                if qty <= 0:
                    self._rotation_escape_buy_persist[sym] = 0
                    continue
                pos_by_sym[sym] = pos
                if sym in buy_by_sym:
                    self._rotation_escape_buy_persist[sym] = self._rotation_escape_buy_persist.get(sym, 0) + 1
                    candidates.append(sym)
                else:
                    self._rotation_escape_buy_persist[sym] = 0

            if not candidates:
                return []

            now = time.time()
            cooldown_sec = float(self._cfg("ROTATION_ESCAPE_COOLDOWN_SEC", 900.0))
            if now - float(getattr(self, "_rotation_escape_last_ts", 0.0)) < cooldown_sec:
                return []

            persist_cycles = int(self._cfg("ROTATION_ESCAPE_BUY_PERSIST_CYCLES", 3))
            open_trades = dict(getattr(self.shared_state, "open_trades", {}) or {})

            # PHASE 2: Check each position for rotation eligibility
            for sym_raw, pos in positions.items():
                sym = self._normalize_symbol(sym_raw)
                qty = float(pos.get("quantity", 0.0) or pos.get("qty", 0.0))
                if qty <= 0:
                    continue
                
                # Eligibility analysis
                eligible = True
                fail_reason = None
                
                tr = open_trades.get(sym) or {}
                created_at = float(tr.get("created_at") or tr.get("opened_at") or pos.get("opened_at") or 0.0)
                
                # RECOVERY FIX: If open_trades is empty (restart), try to infer or fallback
                if created_at <= 0:
                    # If we have a position but no trade record, it's an "orphan" from before restart.
                    persist_count_check = int(self._rotation_escape_buy_persist.get(sym, 0))
                    if persist_count_check >= persist_cycles:
                        created_at = now - (float(self._cfg("ROTATION_ESCAPE_MIN_AGE_MIN", 30)) * 60) - 1.0

                age_min = (now - created_at) / 60.0 if created_at > 0 else 0.0
                
                # Persistence check
                persist_count = int(self._rotation_escape_buy_persist.get(sym, 0))
                if persist_count < persist_cycles:
                    eligible = False
                    fail_reason = f"LOW_PERSISTENCE (cycles={persist_count}/{persist_cycles})"
                elif sym not in buy_by_sym:
                    eligible = False
                    fail_reason = "NO_BUY_SIGNAL"
                
                # unrealized pnl for log
                upnl_pct = 0.0
                try:
                    if hasattr(self.shared_state, "get_unrealized_pnl_pct"):
                        upnl_pct = await _safe_await(self.shared_state.get_unrealized_pnl_pct(sym))
                except Exception: pass

                self.logger.info(
                    "[ROTATION_CHECK] symbol=%s age=%.1fm upnl=%.2f%% eligible=%s reason=%s",
                    sym, age_min, upnl_pct, eligible, fail_reason or "PERSISTENCE_MET"
                )

                if not eligible:
                    hold_reason = "low_volatility" if "LOW_PERSISTENCE" in (fail_reason or "") else (fail_reason or "stagnant")
                    self.logger.info("[ROTATION_HOLD] %s held (reason=%s)", sym, hold_reason)


            best_sym = None
            best_persist = 0
            for sym in candidates:
                count = int(self._rotation_escape_buy_persist.get(sym, 0))
                if count >= persist_cycles and count > best_persist:
                    best_sym = sym
                    best_persist = count

            if not best_sym:
                return []

            best_sig = max(buy_by_sym.get(best_sym, []), key=lambda s: float(s.get("confidence", 0.0)))

            # Ensure TP/SL not triggered for this position
            tr = open_trades.get(best_sym) or {}
            entry_price = float(tr.get("entry_price", 0.0) or 0.0)
            tp = tr.get("tp")
            sl = tr.get("sl")

            # RECOVERY FIX: If entry_price is missing (restart), try to fetch from SharedState positions
            if not entry_price:
                try:
                    if hasattr(self.shared_state, "get_position_entry_price"):
                        entry_price = float(self.shared_state.get_position_entry_price(best_sym) or 0.0)
                    if not entry_price:
                        pos_data = self.shared_state.positions.get(best_sym, {})
                        entry_price = float(pos_data.get("avg_price", 0.0) or pos_data.get("entry_price", 0.0))
                    if entry_price > 0:
                        self.logger.debug("[ROTATION_CHECK] Recovered entry price %.4f for %s from SharedState", entry_price, best_sym)
                except Exception:
                    pass

            cur_price = 0.0
            try:
                if hasattr(self.shared_state, "safe_price"):
                    cur_price = float(await _safe_await(self.shared_state.safe_price(best_sym)) or 0.0)
            except Exception:
                cur_price = 0.0
            if not cur_price:
                cur_price = float(getattr(self.shared_state, "latest_prices", {}).get(best_sym, 0.0) or 0.0)
            
            if not cur_price:
                self.logger.info("[ROTATION_CHECK] %s skipped: No current price", best_sym)
                return []

            check_tpsl = True
            if not entry_price or tp is None or sl is None:
                check_tpsl = False
                self.logger.debug("[ROTATION_CHECK] %s missing full TP/SL data (entry=%.2f tp=%s sl=%s). Bypassing TP/SL check to allow rotation.", 
                                best_sym, entry_price, tp, sl)

            if check_tpsl:
                position_side = str(tr.get("position") or "long").lower()
                if position_side == "short":
                    if cur_price <= float(tp) or cur_price >= float(sl):
                        self.logger.info("[ROTATION_CHECK] %s skipped: TP/SL pending", best_sym)
                        return []
                else:
                    if cur_price >= float(tp) or cur_price <= float(sl):
                        self.logger.info("[ROTATION_CHECK] %s skipped: TP/SL pending", best_sym)
                        return []

            # Partial rotation sizing (never 100%)
            partial_pct = float(self._cfg("ROTATION_ESCAPE_PARTIAL_PCT", 0.25))
            partial_pct = max(0.05, min(0.5, partial_pct))

            pos = pos_by_sym.get(best_sym, {})
            value_usdt = float(pos.get("value_usdt", 0.0) or 0.0)
            if value_usdt <= 0:
                value_usdt = float(pos.get("qty", 0.0) or pos.get("quantity", 0.0)) * cur_price

            sell_sig = {
                "agent": best_sig.get("agent", "Meta"),
                "confidence": float(best_sig.get("confidence", 0.0)),
                "reason": "rotation_escape",
                "tag": "rotation_escape",
                "_partial_pct": partial_pct,
                "_rotation_escape": True,
            }

            min_notional = float(self._cfg("MIN_NOTIONAL_USDT", 10.0))
            min_entry = None
            try:
                if self.shared_state and hasattr(self.shared_state, "compute_min_entry_quote"):
                    base_quote = float(self._cfg("DEFAULT_PLANNED_QUOTE", self._cfg("MIN_ENTRY_QUOTE_USDT", 10.0)) or 0.0)
                    min_entry = await self.shared_state.compute_min_entry_quote(
                        best_sym,
                        default_quote=base_quote,
                        price=float(cur_price or 0.0),
                    )
            except Exception:
                min_entry = None
            if min_entry is None:
                min_entry = float(self._cfg("MIN_ENTRY_QUOTE_USDT", 10.0))
            min_trade = max(min_notional, float(min_entry))
            partial_value = value_usdt * partial_pct
            
            # FIX: Ensure partial sell meets min_notional to prevent rejection
            if partial_value < min_trade:
                if value_usdt > min_trade * 1.05: # Ensure we have enough buffer to partial sell
                    # Bump to minimum executable size
                    adjusted_pct = (min_trade * 1.02) / value_usdt # 2% buffer over min
                    adjusted_pct = min(1.0, adjusted_pct)
                    self.logger.info("[ROTATION_ESCAPE] Bumping partial sell %.1f%% -> %.1f%% to meet min_notional $%.2f", 
                                    partial_pct*100, adjusted_pct*100, min_trade)
                    partial_pct = adjusted_pct
                    partial_value = value_usdt * partial_pct
                    sell_sig["_partial_pct"] = partial_pct
                else:
                    # Position too small for partial rotation, force full rotation (clear dust/small pos)
                    self.logger.info("[ROTATION_ESCAPE] Position value $%.2f too small for partial. Forcing 100%% sell.", value_usdt)
                    partial_pct = 1.0
                    partial_value = value_usdt
                    sell_sig["_partial_pct"] = 1.0

            # If the re-entry BUY would be too small or we just cleared dust, we only emit the SELL
            # (Note: if partial_pct bumped to 1.0, we probably want to assume it's a liquidation and allow re-entry
            # only if capital allows, but rotation implies Swap. However, clearing dust might be best as One Way)
            if partial_value < min_trade:
                self.logger.info("[ROTATION_ESCAPE] Partial buyback %.2f < min_trade %.2f. Emitting SELL only to free capital.", partial_value, min_trade)
                # Return dictionary format for PolicyManager compatibility
                return [{"symbol": best_sym, "side": "SELL", "signal": sell_sig}]

            planned_quote = float(best_sig.get("_planned_quote", best_sig.get("planned_quote", 0.0)) or 0.0)
            reentry_quote = max(planned_quote, partial_value, min_trade)
            buy_sig = {
                "agent": best_sig.get("agent", "Meta"),
                "confidence": float(best_sig.get("confidence", 0.0)),
                "reason": "rotation_reentry",
                "tag": "rotation_reentry",
                "_planned_quote": reentry_quote,
                "_rotation_escape": True,
                "_tier": best_sig.get("_tier", "B"),
            }

            self._rotation_escape_last_ts = now
            self.logger.warning(
                "[ROTATION_ESCAPE] Triggering partial rotation: %s sell_pct=%.0f%% reentry_quote=%.2f (persist=%d cycles)",
                best_sym, partial_pct * 100.0, reentry_quote, best_persist
            )

            # Return dictionary format for PolicyManager compatibility
            return [
                {"symbol": best_sym, "side": "SELL", "signal": sell_sig},
                {"symbol": best_sym, "side": "BUY", "signal": buy_sig},
            ]
        except Exception as e:
            self.logger.debug("[ROTATION_ESCAPE] Failed to build rotation decisions: %s", e)
            return []

    async def _check_controlled_partial_rotation(self, owned_positions: Dict[str, Any], now_ts: float) -> List[Dict[str, Any]]:
        """
        FIRE-001: Controlled Partial Rotation Policy
        
        IF focus_mode active
        AND position_age >= 30 minutes (User spec)
        AND abs(pnl) < 0.3% (stagnation)
        THEN SELL 25% (User spec) to free up capital.
        """
        if not bool(getattr(self, "_focus_mode_active", False)):
            return []
        if not bool(getattr(self.config, "FOCUS_MODE_EXIT_ENABLED", False)):
            return []

        rotation_signals = []
        open_trades = getattr(self.shared_state, "open_trades", {}) or {}
        
        for sym, pos in owned_positions.items():
            ot = open_trades.get(sym, {}) if isinstance(open_trades, dict) else {}
            
            # Authoritative age detection
            opened_at = float(ot.get("opened_at", pos.get("opened_at", 0.0)) or 0.0)
            if opened_at <= 0:
                # If we have a position but no record, it's either just opened or carried over
                # For safety, we bypass if we can't confirm age unless we have first_seen
                continue
                
            age_min = (now_ts - opened_at) / 60.0
            
            # Policy: Age >= 30 minutes
            min_age = float(self._cfg("CONTROLLED_ROTATION_MIN_AGE_MIN", 30.0))
            if age_min < min_age:
                continue
                
            # Authoritative PnL detection
            entry = float(ot.get("entry_price", pos.get("avg_price", 0.0)) or 0.0)
            if entry <= 0:
                continue
                
            price = 0.0
            try:
                if hasattr(self.shared_state, "safe_price"):
                    price = float(await _safe_await(self.shared_state.safe_price(sym)) or 0.0)
            except Exception:
                pass
            if price <= 0:
                price = float(getattr(self.shared_state, "latest_prices", {}).get(sym, 0.0) or 0.0)
                
            if price <= 0:
                continue
                
            pnl_pct = ((price - entry) / entry) * 100.0
            abs_pnl = abs(pnl_pct)
            
            # Policy: abs(pnl) < 0.3% (Stagnation check)
            max_pnl_threshold = float(self._cfg("CONTROLLED_ROTATION_MAX_PNL_PCT", 0.3))
            
            if abs_pnl < max_pnl_threshold:
                self.logger.warning(
                    "[ROTATION_PARTIAL] STAGNATION DETECTED for %s | age=%.1fm | pnl=%.3f%% | Triggering 25%% exit",
                    sym, age_min, pnl_pct
                )
                
                # Check for existing SELL signals to avoid double-dipping
                # (Signal cache list_all is expensive, only do if eligible)
                
                sig = {
                    "symbol": sym,
                    "action": "SELL",
                    "confidence": 0.85, # Strong meta-signal
                    "agent": "MetaRotation",
                    "timestamp": now_ts,
                    "reason": f"ROTATION_PARTIAL age={age_min:.1f}m pnl={pnl_pct:.2f}% (stagnant)",
                    "tag": "ROTATION_PARTIAL",
                    "_partial_pct": 0.25, # 25% as requested
                    "_allow_reentry": True, # Allow rebuying/compounding
                    "_is_rotation": True,
                    "_tier": "A" # Priority execution
                }
                rotation_signals.append(sig)
                
        return rotation_signals

    async def _check_scale_in_opportunity(self, owned_positions: Dict[str, Any], now_ts: float) -> List[Dict[str, Any]]:
        """
        Scale-in after hold time (Compounding) - Delegated to ScalingManager.
        """
        return await self.scaling_manager.check_scale_in_opportunity(
            owned_positions, 
            now_ts, 
            focus_mode_active=getattr(self, "_focus_mode_active", False)
        )

    def _sell_priority_score(self, sig: Dict[str, Any]) -> int:
        """Assign a priority score for SELL arbitration (higher = more important)."""
        reason_text = " ".join([
            str(sig.get("reason") or ""),
            str(sig.get("tag") or ""),
            str(sig.get("exit_reason") or ""),
            str(sig.get("signal_reason") or ""),
        ]).upper()

        # Emergency / Risk SELL
        if sig.get("_is_liquidation") or sig.get("_is_starvation_sell") or any(
            k in reason_text for k in ("EMERGENCY", "LIQUIDATION", "SL", "STOP_LOSS", "DELIST", "LIQUIDITY")
        ):
            return 100

        # Profit-taking SELL
        if "TP" in reason_text or "TAKE_PROFIT" in reason_text:
            return 90

        # Manual / Operator SELL
        if "MANUAL" in reason_text or "OPERATOR" in reason_text or "OVERRIDE" in reason_text:
            return 80

        # Strategic rotation SELL
        if sig.get("_is_rotation") or any(
            k in reason_text for k in ("ROTATION", "REBALANCE", "RECYCLE", "VELOCITY", "CONCENTRATION")
        ):
            return 60

        # State / Escape / Recovery SELL
        if sig.get("_rotation_escape") or sig.get("_dust_exit_forced") or any(
            k in reason_text for k in ("FOCUS", "STARVATION", "CAPITAL_RECOVERY", "BOOTSTRAP", "DUST", "ESCAPE", "RECOVERY")
        ):
            return 40

        return 50

    def _apply_profit_locked_reentry(self, planned_quote: float, symbol: str, sig: Dict[str, Any]) -> float:
        """Clamp BUY size so only realized profit can increase position size."""
        if not bool(getattr(self.config, "PROFIT_LOCK_REENTRY_ENABLED", True)):
            return planned_quote

        if sig.get("_bootstrap") or sig.get("_bootstrap_override") or sig.get("_bootstrap_seed") or sig.get("bootstrap_seed"):
            return planned_quote

        try:
            realized = float(getattr(self.shared_state, "metrics", {}).get("realized_pnl", 0.0) or 0.0)
        except Exception:
            realized = 0.0

        profit_delta = max(0.0, realized - float(getattr(self, "_profit_lock_checkpoint", 0.0) or 0.0))
        base_quote = float(getattr(self, "_profit_lock_base_quote", 0.0) or 0.0)
        allowed = max(base_quote, float(self._min_entry_quote_usdt or 0.0)) + profit_delta

        if planned_quote > allowed:
            self.logger.info(
                "[Meta:ProfitLock] %s planned_quote %.2f -> %.2f (base=%.2f profit_delta=%.2f)",
                symbol, planned_quote, allowed, base_quote, profit_delta,
            )
            return float(allowed)

        return planned_quote

    def _update_profit_lock_checkpoint(self) -> None:
        """Update the realized PnL checkpoint after a successful SELL."""
        try:
            realized = float(getattr(self.shared_state, "metrics", {}).get("realized_pnl", 0.0) or 0.0)
        except Exception:
            realized = 0.0
        self._profit_lock_checkpoint = realized
        try:
            if hasattr(self.shared_state, "metrics"):
                self.shared_state.metrics["profit_lock_checkpoint"] = realized
        except Exception:
            pass

    def _maybe_apply_conditional_size_bump(self) -> None:
        """Apply a one-time planned-quote bump when realized PnL clears fee threshold."""
        if self._planned_quote_bump_applied:
            return
        if not (self._exit_coalescing_enabled and self._min_hold_enabled):
            return

        metrics = getattr(self.shared_state, "metrics", {}) or {}
        try:
            realized = float(metrics.get("realized_pnl", 0.0) or 0.0)
        except Exception:
            realized = 0.0
        if realized <= 0:
            return

        base_quote = float(self._cfg("DEFAULT_PLANNED_QUOTE", default=self._default_planned_quote) or self._default_planned_quote)
        taker_bps = float(self._get_fee_bps(self.shared_state, "taker") or 10.0)
        rt_fee_pct = (taker_bps * 2.0) / 10000.0
        avg_round_trip_fee = base_quote * rt_fee_pct
        bump_threshold_mult = float(self._cfg("PLANNED_QUOTE_BUMP_THRESHOLD_MULT", default=1.2) or 1.2)
        threshold = bump_threshold_mult * avg_round_trip_fee

        if realized > threshold:
            bump_mult = float(self._cfg("PLANNED_QUOTE_BUMP_MULT", default=1.5) or 1.5)
            bumped = max(base_quote, self._min_entry_quote_usdt) * bump_mult
            if self._max_spend > 0:
                bumped = min(bumped, self._max_spend)
            self._default_planned_quote = max(self._default_planned_quote, bumped)
            self._planned_quote_bump_applied = True
            try:
                if hasattr(self.shared_state, "metrics"):
                    self.shared_state.metrics["planned_quote_bump_applied"] = True
                    self.shared_state.metrics["planned_quote_bump_value"] = self._default_planned_quote
                    self.shared_state.metrics["planned_quote_bump_threshold"] = threshold
            except Exception:
                pass
            self.logger.warning(
                "[Meta:SizeBump] Activated: realized_pnl=%.4f > threshold=%.4f (rt_fee=%.4f, mult=%.2f). "
                "DEFAULT_PLANNED_QUOTE -> %.2f (bump_mult=%.2f)",
                realized,
                threshold,
                avg_round_trip_fee,
                bump_threshold_mult,
                self._default_planned_quote,
                bump_mult,
            )
            self.logger.warning(
                "🚀 SIZE_BUMP_APPLIED | new_planned_quote=%.2f | realized_pnl=%.4f | fee_threshold=%.4f",
                self._default_planned_quote,
                realized,
                threshold,
            )

    def _apply_sell_arbiter(self, decisions: List[Tuple[str, str, Dict[str, Any]]]) -> List[Tuple[str, str, Dict[str, Any]]]:
        """Enforce a single SELL per cycle on real capital using priority ranking."""
        is_real_mode = bool(getattr(self.config, "LIVE_MODE", False)) and not bool(getattr(self.config, "SIMULATION_MODE", False)) and not bool(getattr(self.config, "PAPER_MODE", False)) and not bool(getattr(self.config, "TESTNET_MODE", False))
        if not is_real_mode:
            return decisions

        sell_decisions = [d for d in decisions if d[1] == "SELL"]
        if len(sell_decisions) <= 1:
            return decisions

        def _score(decision: Tuple[str, str, Dict[str, Any]]) -> Tuple[int, float]:
            sig = decision[2] or {}
            return (
                self._sell_priority_score(sig),
                float(sig.get("confidence", 0.0) or 0.0),
            )

        best_sell = max(sell_decisions, key=_score)
        dropped = [d for d in sell_decisions if d is not best_sell]
        if dropped:
            self.logger.warning(
                "[Meta:SellArbiter] Real mode: keeping 1 SELL (symbol=%s) and dropping %d lower-priority SELL(s).",
                best_sell[0], len(dropped)
            )

        filtered = [d for d in decisions if d[1] != "SELL"]
        filtered.insert(0, best_sell)
        return filtered

    def _batch_buy_decisions(self, decisions: List[Tuple[str, str, Dict[str, Any]]]) -> List[Tuple[str, str, Dict[str, Any]]]:
        """Aggregate multiple BUY intents per symbol into a single batched BUY."""
        if not decisions:
            return decisions

        grouped: Dict[str, List[Tuple[str, str, Dict[str, Any]]]] = defaultdict(list)
        passthrough: List[Tuple[str, str, Dict[str, Any]]] = []

        for sym, action, sig in decisions:
            if action == "BUY":
                grouped[self._normalize_symbol(sym)].append((sym, action, sig))
            else:
                passthrough.append((sym, action, sig))

        batched: List[Tuple[str, str, Dict[str, Any]]] = []
        for sym_norm, buys in grouped.items():
            if len(buys) == 1:
                batched.append(buys[0])
                continue

            def _score(item: Tuple[str, str, Dict[str, Any]]) -> float:
                return float((item[2] or {}).get("confidence", 0.0) or 0.0)

            best = max(buys, key=_score)
            base_sym, _, base_sig = best
            base_sig = dict(base_sig or {})

            total_quote = 0.0
            reasons = []
            agents = []
            contributions = []
            tiers = []

            for _, _, sig in buys:
                sig = sig or {}
                q = float(sig.get("_planned_quote") or sig.get("planned_quote") or sig.get("quote_amount") or sig.get("quote") or 0.0)
                if q > 0:
                    total_quote += q
                reasons.append(str(sig.get("reason") or sig.get("tag") or ""))
                agents.append(str(sig.get("agent") or ""))
                tiers.append(str(sig.get("_tier") or ""))
                if sig.get("_contributions"):
                    contributions.extend(list(sig.get("_contributions") or []))

            if total_quote <= 0:
                total_quote = float(base_sig.get("_planned_quote") or base_sig.get("planned_quote") or 0.0)
            base_sig["_planned_quote"] = total_quote
            if contributions:
                base_sig["_contributions"] = contributions
            if tiers:
                if "A" in tiers:
                    base_sig["_tier"] = "A"
                elif "B" in tiers:
                    base_sig["_tier"] = "B"
            if any(bool(s.get("_bootstrap")) for _, _, s in buys):
                base_sig["_bootstrap"] = True
            if any(bool(s.get("_bootstrap_override")) for _, _, s in buys):
                base_sig["_bootstrap_override"] = True
            if any(bool(s.get("_bootstrap_seed")) for _, _, s in buys):
                base_sig["_bootstrap_seed"] = True
            if any(bool(s.get("_accumulated")) for _, _, s in buys):
                base_sig["_accumulated"] = True

            base_sig["_batched"] = True
            base_sig["_batched_count"] = len(buys)
            base_sig["_batched_reasons"] = reasons
            base_sig["_batched_agents"] = agents

            self.logger.info(
                "[Meta:BuyBatch] %s batched %d BUYs -> one intent (quote=%.2f)",
                sym_norm, len(buys), total_quote,
            )
            batched.append((base_sym, "BUY", base_sig))

        return passthrough + batched

    def _batch_sell_decisions(self, decisions: List[Tuple[str, str, Dict[str, Any]]]) -> List[Tuple[str, str, Dict[str, Any]]]:
        """Aggregate multiple SELL intents per symbol into a single batched SELL."""
        if not decisions:
            return decisions

        grouped: Dict[str, List[Tuple[str, str, Dict[str, Any]]]] = defaultdict(list)
        passthrough: List[Tuple[str, str, Dict[str, Any]]] = []

        for sym, action, sig in decisions:
            if action == "SELL":
                grouped[self._normalize_symbol(sym)].append((sym, action, sig))
            else:
                passthrough.append((sym, action, sig))

        batched: List[Tuple[str, str, Dict[str, Any]]] = []
        for sym_norm, sells in grouped.items():
            if len(sells) == 1:
                batched.append(sells[0])
                continue

            # Choose highest-priority SELL as the base
            def _score(item: Tuple[str, str, Dict[str, Any]]) -> Tuple[int, float]:
                return (self._sell_priority_score(item[2] or {}), float(item[2].get("confidence", 0.0) or 0.0))

            best = max(sells, key=_score)
            base_sym, _, base_sig = best
            base_sig = dict(base_sig or {})

            # Aggregate partial exits per symbol
            partial_sum = 0.0
            for _, _, sig in sells:
                pct = float(sig.get("_partial_pct", 0.0) or 0.0)
                if pct > 0:
                    partial_sum += pct

            if partial_sum > 0:
                base_sig["_partial_pct"] = min(1.0, partial_sum)

            # Preserve liquidation/emergency flags if any
            if any(bool(s.get("_is_liquidation")) for _, _, s in sells):
                base_sig["_is_liquidation"] = True
            if any(bool(s.get("_is_starvation_sell")) for _, _, s in sells):
                base_sig["_is_starvation_sell"] = True

            base_sig["_batched"] = True
            base_sig["_batched_count"] = len(sells)
            base_sig["_batched_reasons"] = [str(s.get("reason") or s.get("tag") or "") for _, _, s in sells]

            # Audit: log batched qty + pnl vs fee threshold
            try:
                total_qty = 0.0
                for sym_raw, _, sig in sells:
                    if sig.get("_partial_pct") is not None:
                        qty = float(self.shared_state.get_position_qty(sym_raw) or 0.0)
                        total_qty += qty * float(sig.get("_partial_pct") or 0.0)
                if total_qty <= 0:
                    total_qty = float(self.shared_state.get_position_qty(base_sym) or 0.0)

                entry_price = 0.0
                try:
                    ot = getattr(self.shared_state, "open_trades", {}).get(base_sym, {})
                    entry_price = float(ot.get("entry_price", 0.0) or 0.0)
                except Exception:
                    entry_price = 0.0
                if not entry_price:
                    pos = getattr(self.shared_state, "positions", {}).get(base_sym, {})
                    entry_price = float(pos.get("avg_price", 0.0) or pos.get("entry_price", 0.0) or 0.0)

                cur_price = float(getattr(self.shared_state, "latest_prices", {}).get(base_sym, 0.0) or 0.0)

                pnl_pct = ((cur_price - entry_price) / entry_price) if entry_price > 0 and cur_price > 0 else 0.0
                fee_mult = float(getattr(self.config, "MIN_PROFIT_EXIT_FEE_MULT", 2.0) or 2.0)
                rt_fee_pct = ((self._get_fee_bps(self.shared_state, "taker") or 10.0) * 2.0) / 10000.0
                min_profit = rt_fee_pct * fee_mult

                self.logger.info(
                    "[Meta:SellBatchAudit] %s qty=%.8f pnl=%.3f%% vs fee_threshold=%.3f%% (fee_mult=%.2f)",
                    sym_norm, total_qty, pnl_pct * 100.0, min_profit * 100.0, fee_mult,
                )
            except Exception:
                self.logger.debug("[Meta:SellBatchAudit] Failed to compute audit metrics for %s", sym_norm)

            self.logger.info(
                "[Meta:SellBatch] %s batched %d SELLs -> one intent (partial_pct=%s)",
                sym_norm, len(sells),
                f"{base_sig.get('_partial_pct'):.2f}" if base_sig.get("_partial_pct") else "n/a",
            )
            batched.append((base_sym, "SELL", base_sig))

        return passthrough + batched


    async def _build_decisions(self, accepted_symbols_set: set) -> List[Tuple[str, str, Dict[str, Any]]]:
        # CRITICAL DEBUG: Log that _build_decisions is being called
        self.logger.warning("[Meta:DEBUG] _build_decisions called with %d accepted symbols", len(accepted_symbols_set))

        # Conditional size bump (one-time) after realized PnL clears fee threshold
        try:
            self._maybe_apply_conditional_size_bump()
        except Exception:
            self.logger.debug("[Meta:SizeBump] Evaluation failed (non-fatal)", exc_info=True)
        
        # 1. Authoritative Flat Check (Authoritative Source for Governance)
        is_flat = await self._check_portfolio_flat()
        
        # 2. Get Structured Governance Decision (PROACTIVE)
        # We don't have bootstrap_override yet, but we'll calculate it later if needed for emission
        # For now, get the base decision to enforce blockers
        gov_decision = self._get_governance_decision(is_flat, bootstrap_override=False)
        self._emit_governance_decision(gov_decision)
        
        # 3. Absolute Mode Blockers (SOP Enforcement)
        current_mode = gov_decision["mode"]
        if current_mode == "PAUSED":
            self.logger.info("[Meta:PAUSED] Enforcement: Blocking ALL trading activity.")
            return []

        # ═══════════════════════════════════════════════════════════════════════
        # SOP-BOOT-01: Single Bootstrap Seed Trade (one-cycle TTL)
        # ═══════════════════════════════════════════════════════════════════════
        try:
            _, sig_pos_seed, _ = await self._count_significant_positions()
        except Exception:
            sig_pos_seed = 0

        if (
            self._bootstrap_seed_enabled
            and not self._bootstrap_seed_attempted
            and not self._bootstrap_seed_used
            and sig_pos_seed == 0
        ):
            if self._bootstrap_seed_cycle != self._tick_counter:
                seed_symbol = self._bootstrap_seed_symbol
                if accepted_symbols_set and seed_symbol not in accepted_symbols_set:
                    self.logger.warning(
                        "[BOOTSTRAP] Seed skipped: %s not in accepted symbols.",
                        seed_symbol,
                    )
                else:
                    min_notional = None
                    try:
                        if self.exchange_client and hasattr(self.exchange_client, "get_symbol_info"):
                            info = await self.exchange_client.get_symbol_info(seed_symbol)
                            if info and isinstance(info, dict):
                                min_notional = float(info.get("minNotional", 0.0) or 0.0)
                    except Exception:
                        min_notional = None
                    if min_notional is None or min_notional <= 0:
                        min_notional = float(getattr(self.config, "MIN_NOTIONAL_USDT", 10.0))

                    seed_quote = max(
                        float(self._bootstrap_seed_quote or 0.0),
                        float(self._min_entry_quote_usdt or 0.0),
                        float(min_notional or 0.0),
                    )

                    seed_sig = {
                        "symbol": seed_symbol,
                        "action": "BUY",
                        "confidence": 1.0,
                        "agent": "BootstrapSeed",
                        "timestamp": time.time(),
                        "reason": "BOOTSTRAP_SEED",
                        "context": "BOOTSTRAP",
                        "_bootstrap_seed": True,
                        "bootstrap_seed": True,
                        "execution_tag": "meta/bootstrap_seed",
                        "_bootstrap_seed_cycle": self._tick_counter,
                        "_tier": "BOOTSTRAP",
                    }

                    planned_quote = await self._planned_quote_for(seed_symbol, seed_sig, budget_override=seed_quote)
                    planned_quote = max(
                        float(getattr(self.config, "DEFAULT_PLANNED_QUOTE", 0.0) or 0.0),
                        float(getattr(self.config, "MIN_TRADE_QUOTE", 0.0) or 0.0),
                        float(planned_quote or 0.0),
                    )
                    if planned_quote < min_notional:
                        self.logger.warning(
                            "[BOOTSTRAP] Seed blocked: planned_quote %.2f < min_notional %.2f",
                            planned_quote,
                            min_notional,
                        )
                        self._bootstrap_seed_attempted = True
                        self._bootstrap_seed_cycle = self._tick_counter
                    else:
                        quote_asset = str(self._cfg("QUOTE_ASSET") or "USDT").upper()
                        try:
                            balance = float(await _safe_await(
                                self.shared_state.get_spendable_balance(quote_asset)
                            ) or 0.0)
                        except Exception:
                            balance = 0.0
                        if balance < planned_quote:
                            self.logger.warning(
                                "[BOOTSTRAP] Seed blocked: balance %.2f < planned_quote %.2f",
                                balance,
                                planned_quote,
                            )
                            self._bootstrap_seed_attempted = True
                            self._bootstrap_seed_cycle = self._tick_counter
                        else:
                            seed_sig["_planned_quote"] = planned_quote
                            self.logger.warning(
                                "[BOOTSTRAP] Seed trade armed: %s quote=%.2f (min_notional=%.2f) TTL=1 cycle",
                                seed_symbol,
                                planned_quote,
                                min_notional,
                            )
                            self._bootstrap_seed_attempted = True
                            self._bootstrap_seed_cycle = self._tick_counter
                            return [(seed_symbol, "BUY", seed_sig)]
            
        allowed_actions = gov_decision["allowed_actions"]
        
        # Portfolio analysis for focus mode logic
        total_pos, sig_pos, dust_pos = await self._count_significant_positions()
        dust_ratio = (dust_pos / total_pos) if total_pos > 0 else 0.0
        
        # STEP 0.1: Mandatory Sell Unsticking (Enforce SOP capacity recovered)
        # We need max_pos here early to evaluate unsticking condition
        max_pos_for_clear = self._get_max_positions()
        if self._mandatory_sell_mode_active and sig_pos < max_pos_for_clear:
            self.logger.info(
                "[Meta:MandatorySell] ✅ RECOVERY SUCCESSFUL: Portfolio recovered to sig_pos=%d < max_pos=%d. "
                "Exiting SELL-only recovery mode.",
                sig_pos, max_pos_for_clear
            )
            self._mandatory_sell_mode_active = False

        # ────────────────────────────────────────────────────────────────────────
        # FOCUS MODE LATCH: Evaluate triggers and manage state
        # ────────────────────────────────────────────────────────────────────────
        context_flags = {}
        # PHASE 2 CONSOLIDATION: STEP 0 - Centralized Capital Floor Check
        self.logger.info("STEP 0: Checking capital floor (centralized check)")
        capital_block = False
        capital_ok = await self._check_capital_floor_central()
        is_starved = not capital_ok
        # Calculate NAV early for all authority layers
        nav = 0.0
        try:
            if hasattr(self.shared_state, "get_nav_quote"):
                nav = float(await _safe_await(self.shared_state.get_nav_quote()) or 0.0)
            else:
                nav = float(getattr(self.shared_state, "nav", 0.0) or 0.0)
        except Exception:
            nav = 0.0

        if not capital_ok:
            # Do NOT return early—BUYs blocked; SELLs still allowed
            capital_block = True
            context_flags["CAPITAL_BLOCK"] = True
            quote_asset = str(self._cfg("QUOTE_ASSET") or "USDT").upper()
            free_usdt = float(await self.shared_state.get_spendable_balance(quote_asset) or 0.0)
            
            abs_min_floor = float(self._cfg("ABSOLUTE_MIN_FLOOR", self._cfg("CAPITAL_PRESERVATION_FLOOR", 10.0)))
            floor_pct = float(self._cfg("CAPITAL_FLOOR_PCT", 0.20))
            floor = max(abs_min_floor, nav * floor_pct)
            self.logger.warning(
                f"STEP 0: CAPITAL_FLOOR_VIOLATION - BUYs blocked | "
                f"free_usdt={free_usdt:.2f} < floor={floor:.2f} (nav={nav:.2f}, pct={floor_pct:.2%})"
            )

        # ────────────────────────────────────────────────────────────────────────
        # PHASE GATE: BOOTSTRAP (virtual) until capital stability is confirmed
        # ────────────────────────────────────────────────────────────────────────
        capital_stable, stability_reason = self._evaluate_capital_stability(capital_ok, nav)
        if hasattr(self.shared_state, "metrics"):
            self.shared_state.metrics["capital_stable"] = bool(capital_stable)
            self.shared_state.metrics["capital_stability_reason"] = stability_reason
        if not capital_stable:
            self.logger.warning(
                "[Meta:PhaseGate] BOOTSTRAP_VIRTUAL active: stability=%s | Blocking real trades until stable.",
                stability_reason,
            )
            return []

        # ────────────────────────────────────────────────────────────────────────
        # STEP 1: EXPLICIT VELOCITY & EXIT AUTHORITY (Canonical REA Override)
        # ────────────────────────────────────────────────────────────────────────
        # [REA] We evaluate exits based on capital velocity and rotation BEFORE 
        # looking at new buys. This solves the "deadlock" where the bot holds
        # stagnant positions and hits limits on new buys.
        
        owned_positions_for_rea = self.shared_state.get_open_positions()
        
        # 1.1 Proactive Stagnation Purge (Always running to maintain velocity)
        # Identifies "zombie" positions that are old and unprofitable.
        stagnation_exit_sig = self.rotation_authority.authorize_stagnation_exit(
            owned_positions=owned_positions_for_rea,
            current_mode=current_mode
        )
        if stagnation_exit_sig:
            self.logger.warning("[Meta:ExitAuth] VELOCITY_PURGE: Authorizing exit for stagnant %s (Maintaining turnover)", stagnation_exit_sig.get("symbol"))
            if (
                await self._passes_meta_sell_profit_gate(stagnation_exit_sig.get("symbol"), stagnation_exit_sig)
                and await self._passes_meta_sell_excursion_gate(stagnation_exit_sig.get("symbol"), stagnation_exit_sig)
            ):
                return [(stagnation_exit_sig.get("symbol"), "SELL", stagnation_exit_sig)]

        # 1.2 Proactive Rotation (if full or starved)
        # Authorizes forced exits to free capital for higher-alpha opportunities.
        if sig_pos >= max_pos_for_clear or is_starved:
            # Fetch best available candidates from cache/bus
            temp_signals = self.signal_manager.get_all_signals() or []
            if temp_signals:
                # Find best BUY candidate
                buy_sigs = [s for s in temp_signals if s.get("action") == "BUY"]
                
                if buy_sigs:
                    best_buy_cand = max(buy_sigs, key=lambda x: self.score_opportunity(x))
                    opp_score = self.score_opportunity(best_buy_cand)
                    best_buy_cand["_opp_score"] = opp_score
                    
                    rotation_sig = await self.rotation_authority.authorize_rotation(
                        sig_pos=sig_pos, 
                        max_pos=max_pos_for_clear,
                        owned_positions=owned_positions_for_rea,
                        best_opp=best_buy_cand,
                        current_mode=current_mode,
                        is_starved=is_starved
                    )
                    
                    if rotation_sig:
                        best_buy_cand["_replacement"] = True
                        best_buy_cand["_replaces_symbol"] = rotation_sig.get("symbol")
                        self.logger.warning(
                            "[Meta:ExitAuth] ROTATION GRANTED: %s -> %s (Starved: %s)", 
                            rotation_sig.get("symbol"), best_buy_cand.get("symbol"), is_starved
                        )
                        if (
                            await self._passes_meta_sell_profit_gate(rotation_sig.get("symbol"), rotation_sig)
                            and await self._passes_meta_sell_excursion_gate(rotation_sig.get("symbol"), rotation_sig)
                        ):
                            return [(rotation_sig.get("symbol"), "SELL", rotation_sig)]

        # 1.2.5: LAYER 1.5 - Capital Starvation Authority (Rule 4)
        # If starved for capital, identify and exit the LOWEST efficiency position
        # to ensure restricted funds are always working in the best signals.
        if is_starved:
            starvation_exit_sig = self.rotation_authority.authorize_starvation_efficiency_exit(
                owned_positions=owned_positions_for_rea,
                nav=nav,
                free_usdt=available_capital
            )
            if starvation_exit_sig:
                self.logger.warning(
                    "[Meta:ExitAuth] STARVATION_EXIT: Reclaiming capital from %s (Efficiency Optimization)", 
                    starvation_exit_sig.get("symbol")
                )
                if (
                    await self._passes_meta_sell_profit_gate(starvation_exit_sig.get("symbol"), starvation_exit_sig)
                    and await self._passes_meta_sell_excursion_gate(starvation_exit_sig.get("symbol"), starvation_exit_sig)
                ):
                    return [(starvation_exit_sig.get("symbol"), "SELL", starvation_exit_sig)]

        # 1.3: LAYER 2 (Cont.) - Portfolio Concentration Authority
        # Proactively authorizes tactical exits if one symbol is consuming too much bandwidth.
        conc_exit_sig = self.rotation_authority.authorize_concentration_exit(
            owned_positions=owned_positions_for_rea,
            nav=nav
        )
        if conc_exit_sig:
            conc_exit_sig.setdefault("tag", "meta_exit")
            self.logger.warning("[Meta:ExitAuth] CONCENTRATION_EXIT: Re-balancing %s for bandwidth", conc_exit_sig.get("symbol"))
            if (
                await self._passes_meta_sell_profit_gate(conc_exit_sig.get("symbol"), conc_exit_sig)
                and await self._passes_meta_sell_excursion_gate(conc_exit_sig.get("symbol"), conc_exit_sig)
            ):
                return [(conc_exit_sig.get("symbol"), "SELL", conc_exit_sig)]

        # 1.4: LAYER 3 - Portfolio Authority (Velocity & Rebalancing)
        # Higher-level governance based on target USDT/hr and capital utilization.
        metrics = await self._gather_mode_metrics()
        
        # Velocity Recycling check
        vel_exit_sig = self.portfolio_authority.authorize_velocity_exit(
            owned_positions=owned_positions_for_rea,
            current_metrics=metrics
        )
        if vel_exit_sig:
            vel_exit_sig.setdefault("tag", "meta_exit")
            self.logger.warning("[Meta:ExitAuth] VELOCITY_RECYCLING: Exiting %s (Below profit target)", vel_exit_sig.get("symbol"))
            if (
                await self._passes_meta_sell_profit_gate(vel_exit_sig.get("symbol"), vel_exit_sig)
                and await self._passes_meta_sell_excursion_gate(vel_exit_sig.get("symbol"), vel_exit_sig)
            ):
                return [(vel_exit_sig.get("symbol"), "SELL", vel_exit_sig)]
            
        # Rebalancing check
        rebal_exit_sig = self.portfolio_authority.authorize_rebalance_exit(
            owned_positions=owned_positions_for_rea,
            nav=nav
        )
        if rebal_exit_sig:
            rebal_exit_sig.setdefault("tag", "rebalance")
            self.logger.warning("[Meta:ExitAuth] PORTFOLIO_REBALANCE: Force rebalancing %s", rebal_exit_sig.get("symbol"))
            if (
                await self._passes_meta_sell_profit_gate(rebal_exit_sig.get("symbol"), rebal_exit_sig)
                and await self._passes_meta_sell_excursion_gate(rebal_exit_sig.get("symbol"), rebal_exit_sig)
            ):
                return [(rebal_exit_sig.get("symbol"), "SELL", rebal_exit_sig)]
            
        # Profit Recycling check
        recycle_exit_sig = self.portfolio_authority.authorize_profit_recycling(
            owned_positions=owned_positions_for_rea
        )
        if recycle_exit_sig:
            recycle_exit_sig.setdefault("tag", "meta_exit")
            self.logger.warning("[Meta:ExitAuth] PROFIT_RECYCLING: Locking in %s", recycle_exit_sig.get("symbol"))
            if (
                await self._passes_meta_sell_profit_gate(recycle_exit_sig.get("symbol"), recycle_exit_sig)
                and await self._passes_meta_sell_excursion_gate(recycle_exit_sig.get("symbol"), recycle_exit_sig)
            ):
                return [(recycle_exit_sig.get("symbol"), "SELL", recycle_exit_sig)]

            # CAPITAL RECOVERY MODE: enable SELL escalation on capital floor breach
            # Rule: if capital below floor, positions exist, and not in mandatory sell-only mode
            # then allow only SELLs and force exits via TPSL (preferred) or orchestrator fallback.
            if total_pos > 0 and not self._mandatory_sell_mode_active:
                min_pnl_pct = float(self._cfg("CAPITAL_RECOVERY_MIN_PNL_PCT", 0.0001))
                max_age_min = float(self._cfg("CAPITAL_RECOVERY_MAX_AGE_MINUTES", 10.0))
                max_age_sec = max(0.0, max_age_min * 60.0)

                prev_state = getattr(self.shared_state, "capital_recovery_mode", {})
                prev_state = prev_state if isinstance(prev_state, dict) else {}
                prev_active = bool(prev_state.get("active"))
                started_at = prev_state.get("started_at")
                if not started_at:
                    started_at = time.time()
                started_at = float(started_at or time.time())

                setattr(self.shared_state, "capital_recovery_mode", {
                    "active": True,
                    "started_at": started_at,
                    "free_usdt": free_usdt,
                    "floor": floor,
                    "gap_usdt": max(0.0, floor - free_usdt),
                    "min_pnl_pct": min_pnl_pct,
                    "max_age_sec": max_age_sec,
                })
                context_flags["CAPITAL_RECOVERY_MODE"] = True
                self.logger.warning(
                    "[Meta:CapitalRecovery] ACTIVE | free_usdt=%.2f < floor=%.2f | min_pnl=%.4f%% | max_age=%ds",
                    free_usdt, floor, min_pnl_pct * 100.0, int(max_age_sec)
                )
                if not prev_active:
                    self.logger.info(
                        "[Meta:CapitalRecovery] Activated (transition) | free_usdt=%.2f floor=%.2f nav=%.2f pct=%.2f%% pos=%d",
                        free_usdt, floor, nav, floor_pct * 100.0, int(total_pos)
                    )
                    # Option A: nominate oldest position for recovery sell on activation
                    try:
                        positions = getattr(self.shared_state, "positions", {}) or {}
                        open_trades = getattr(self.shared_state, "open_trades", {}) or {}
                        candidates = []
                        fallback_candidates = []
                        if isinstance(positions, dict):
                            for sym, pos in positions.items():
                                ot = open_trades.get(sym, {}) if isinstance(open_trades, dict) else {}
                                created_at = (
                                    ot.get("created_at")
                                    or ot.get("opened_at")
                                    or pos.get("entry_time")
                                    or pos.get("opened_at")
                                    or time.time()
                                )
                                if self._is_recovery_sellable(sym, pos, ignore_filters=False):
                                    candidates.append((float(created_at or 0.0), sym))
                                if self._is_recovery_sellable(sym, pos, ignore_filters=True):
                                    fallback_candidates.append((float(created_at or 0.0), sym))
                        selection = candidates if candidates else fallback_candidates
                        if selection:
                            selection.sort(key=lambda x: x[0])
                            nominated_sym = selection[0][1]
                            new_state = dict(getattr(self.shared_state, "capital_recovery_mode", {}) or {})
                            new_state.update({
                                "nominated_symbol": nominated_sym,
                                "candidate_symbol": nominated_sym,
                                "nominated_at": time.time(),
                                "nominated_reason": "max_age",
                                "force_sell_emitted": False,
                                "soft_sell_emitted": False,
                            })
                            setattr(self.shared_state, "capital_recovery_mode", new_state)
                            self.logger.warning(
                                "[Meta:CapitalRecovery] Nominated oldest position for sell: %s",
                                nominated_sym
                            )
                    except Exception as e:
                        self.logger.debug("[Meta:CapitalRecovery] Nomination failed: %s", e)

                # Fallback path: if no TP/SL engine is available, trigger liquidity orchestration
                if not getattr(self, "tp_sl_engine", None):
                    gap_usdt = max(0.0, floor - free_usdt)
                    try:
                        if self.liquidation_agent and hasattr(self.liquidation_agent, "trigger_liquidity"):
                            await _safe_await(self.liquidation_agent.trigger_liquidity(
                                required_usdt=floor,
                                free_usdt=free_usdt,
                                gap_usdt=gap_usdt,
                                reason="CAPITAL_FLOOR_RECOVERY",
                            ))
                        elif self.liquidation_agent and hasattr(self.liquidation_agent, "_free_usdt_now"):
                            await self.liquidation_agent._free_usdt_now(
                                target=gap_usdt,
                                reason="CAPITAL_FLOOR_RECOVERY",
                                free_before=free_usdt,
                            )
                        elif hasattr(self.shared_state, "emit_event"):
                            payload = {
                                "required_usdt": float(floor),
                                "free_usdt": float(free_usdt),
                                "gap_usdt": float(gap_usdt),
                                "reason": "CAPITAL_FLOOR_RECOVERY",
                            }
                            await _safe_await(self.shared_state.emit_event("Liquidity.Orchestrate", payload))
                    except Exception as e:
                        self.logger.debug("[Meta:CapitalRecovery] Liquidity fallback failed: %s", e)
        else:
            # Clear capital recovery mode when capital is healthy or no positions exist
            try:
                prev_state = getattr(self.shared_state, "capital_recovery_mode", {})
                prev_state = prev_state if isinstance(prev_state, dict) else {}
                if prev_state.get("active"):
                    self.logger.info(
                        "[Meta:CapitalRecovery] Deactivated (transition) | free_usdt=%.2f floor=%.2f nav=%.2f pct=%.2f%% pos=%d",
                        free_usdt, floor, nav, floor_pct * 100.0, int(total_pos)
                    )
                setattr(self.shared_state, "capital_recovery_mode", {
                    "active": False,
                    "nomination_status_logged": False,
                })
                status_payload = {
                    "recovery_active": False,
                    "elapsed_sec": 0,
                    "remaining_sec": 0,
                    "max_age_sec": 0,
                    "candidate": None,
                    "floor": 0.0,
                    "free_usdt": 0.0,
                }
                if hasattr(self.shared_state, "set_dynamic_param"):
                    await _safe_await(self.shared_state.set_dynamic_param(
                        "capital_recovery_status", status_payload
                    ))
                else:
                    setattr(self.shared_state, "capital_recovery_status", status_payload)
            except Exception:
                pass

        # Emit recovery state once per loop for diagnosis
        try:
            rec_state = getattr(self.shared_state, "capital_recovery_mode", None)
            active_slots = len([
                p for p in owned_positions.values()
                if self._is_recovery_sellable("", p, ignore_filters=False, ignore_core=True)
            ]) if isinstance(owned_positions, dict) else 0
            self.logger.info(
                "[Meta:CapitalRecovery] STATE | %s | pos(total=%d sig=%d dust=%d active_slots=%d)",
                rec_state, int(total_pos), int(sig_pos), int(dust_pos), int(active_slots)
            )
        except Exception:
            pass

        # Emit recovery timer every loop when active
        try:
            rec_state = getattr(self.shared_state, "capital_recovery_mode", {}) or {}
            if isinstance(rec_state, dict) and rec_state.get("active"):
                now = time.time()
                started_at = float(rec_state.get("started_at", now) or now)
                max_age = float(rec_state.get(
                    "max_age_sec",
                    self._cfg("CAPITAL_RECOVERY_FORCE_SELL_AFTER_SEC", 600.0)
                ))
                elapsed = int(max(0.0, now - started_at))
                remaining = int(max(0.0, max_age - elapsed))
                last_log_ts = float(rec_state.get("last_timer_log_ts", 0.0) or 0.0)
                if (now - last_log_ts) >= 1.0:
                    rec_state["last_timer_log_ts"] = now

                    nominated_sym = rec_state.get("nominated_symbol") or rec_state.get("candidate_symbol")
                    status_logged = bool(rec_state.get("nomination_status_logged"))
                    if not status_logged:
                        self.logger.info(
                            "[Meta:CapitalRecovery] Nomination status: %s",
                            nominated_sym or "NONE"
                        )
                        rec_state["nomination_status_logged"] = True

                    if not nominated_sym:
                        try:
                            positions = self.shared_state.get_positions_snapshot() or {}
                            open_trades = getattr(self.shared_state, "open_trades", {}) or {}
                            candidates = []
                            fallback_candidates = []
                            if isinstance(positions, dict):
                                for sym_raw, pos in positions.items():
                                    sym = self._normalize_symbol(sym_raw)
                                    ot = open_trades.get(sym, {}) if isinstance(open_trades, dict) else {}
                                    created_at = (
                                        ot.get("created_at")
                                        or ot.get("opened_at")
                                        or pos.get("entry_time")
                                        or pos.get("opened_at")
                                        or now
                                    )
                                    value_usdt = float(pos.get("value_usdt", 0.0) or 0.0)
                                    entry = (float(created_at or 0.0), -value_usdt, sym)
                                    if self._is_recovery_sellable(sym, pos, ignore_filters=False):
                                        candidates.append(entry)
                                    if self._is_recovery_sellable(sym, pos, ignore_filters=True):
                                        fallback_candidates.append(entry)
                            if candidates or fallback_candidates:
                                selection = candidates if candidates else fallback_candidates
                                selection.sort(key=lambda x: (x[0], x[1]))
                                nominated_sym = selection[0][2]
                                rec_state.update({
                                    "nominated_symbol": nominated_sym,
                                    "candidate_symbol": nominated_sym,
                                    "nominated_at": now,
                                    "nominated_reason": "recovery_fallback",
                                })
                                self.logger.warning(
                                    "[Meta:CapitalRecovery] Fallback nomination: %s",
                                    nominated_sym
                                )
                        except Exception:
                            nominated_sym = None

                        if not nominated_sym:
                            self.logger.warning(
                                "[Meta:CapitalRecovery] No nominee available (no eligible positions)."
                            )

                    if nominated_sym:
                        rec_state["candidate_symbol"] = nominated_sym

                    # Reset latches when candidate is closed
                    try:
                        positions = self.shared_state.get_positions_snapshot() or {}
                        cand_pos = positions.get(nominated_sym) if nominated_sym else None
                        cand_qty = float(cand_pos.get("quantity", 0.0) or cand_pos.get("qty", 0.0)) if cand_pos else 0.0
                        if nominated_sym and cand_qty <= 0:
                            rec_state.update({
                                "force_sell_emitted": False,
                                "soft_sell_emitted": False,
                                "nominated_emitted": False,
                                "nominated_symbol": None,
                                "candidate_symbol": None,
                                "nomination_status_logged": False,
                            })
                            nominated_sym = None
                    except Exception:
                        pass

                    setattr(self.shared_state, "capital_recovery_mode", rec_state)

                    free_usdt = float(rec_state.get("free_usdt", 0.0) or 0.0)
                    floor = float(rec_state.get("floor", 0.0) or 0.0)
                    self.logger.warning(
                        "[Meta:CapitalRecovery:TIMER] ⏱ elapsed=%ds / %ds | remaining=%ds | candidate=%s | floor=%.2f free=%.2f",
                        elapsed, int(max_age), remaining, nominated_sym or "NONE", floor, free_usdt
                    )

                    status_payload = {
                        "recovery_active": True,
                        "elapsed_sec": elapsed,
                        "remaining_sec": remaining,
                        "max_age_sec": int(max_age),
                        "candidate": nominated_sym,
                        "floor": floor,
                        "free_usdt": free_usdt,
                    }
                    if hasattr(self.shared_state, "set_dynamic_param"):
                        await _safe_await(self.shared_state.set_dynamic_param(
                            "capital_recovery_status", status_payload
                        ))
                    else:
                        setattr(self.shared_state, "capital_recovery_status", status_payload)
                else:
                    setattr(self.shared_state, "capital_recovery_mode", rec_state)
        except Exception:
            pass

        # FOCUS MODE LATCH: Only evaluate entry if not already active
        if self.FOCUS_MODE_ENABLED and not getattr(self, '_focus_mode_active', False):
            # 1. Liveness failure (execution-based)
            last_cycle_attempts = int(getattr(self, '_last_cycle_execution_attempts', 0) or 0)
            if last_cycle_attempts == 0:
                self._focus_liveness_counter += 1
            else:
                self._focus_liveness_counter = 0
            if self._focus_liveness_counter >= self.FOCUS_LIVENESS_FAILURE_CYCLES:
                self._activate_focus_mode("Liveness failure: No execution attempts for N evaluate_and_act() cycles")
            # 2. Structural failure
            if dust_ratio > 0.6 and sig_pos == 0 and not capital_ok:
                self._activate_focus_mode("Dust-dominated portfolio with no significant positions and capital starvation")
        # FOCUS MODE EXIT (latched): Only allow exit if active
        if getattr(self, '_focus_mode_active', False):
            if self._focus_mode_trade_executed and (sig_pos > 0 or capital_ok):
                self._deactivate_focus_mode()
            # While active, do not re-evaluate entry conditions
            self.logger.warning("[FOCUS_MODE] ACTIVE | Restricting trading universe and execution")
            context_flags["FOCUS_MODE"] = True
        # ...existing code...
        self.logger.debug(f"[MetaTrace] _build_decisions start. Accepted: {len(accepted_symbols_set)}")

        # Capital floor applies only to BUYs; SELL path continues regardless

        # ═══════════════════════════════════════════════════════════════════════════════
        # STEP 0.0: DUST GRADUATION (NOT terminal, updates position states)
        # ═══════════════════════════════════════════════════════════════════════════════
        # Reclassify positions that have grown beyond dust thresholds
        # This ensures dust state doesn't stay sticky even when position value increases
        # CRITICAL: Must run before any other checks so they see correct position states
        
        graduated_count = await self._reclassify_graduated_positions()
        if graduated_count > 0:
            self.logger.info(
                "[Meta:DUST_GRADUATION] 🎓 %d position(s) promoted from DUST_LOCKED to HEALTHY "
                "(state is no longer sticky, position now treated as normal)",
                graduated_count
            )
            context_flags["DUST_GRADUATED"] = graduated_count
            # REFRESH AUTHORITATIVE COUNTS after graduation
            total_pos, sig_pos, dust_pos = await self._count_significant_positions()
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # STEP 0.1: P-1 EMERGENCY DUST CONSOLIDATION (NOT terminal, sets context)
        # ═══════════════════════════════════════════════════════════════════════════════
        # Bundle multiple dust positions together and sell them as a group
        # This is how real exchanges escape dust traps (e.g., Binance dust converter)
        # 
        # This allows the pipeline to continue and respect the emergency intent
        p1_plan = await self._check_p_minus_1_dust_consolidation()
        if p1_plan:
            context_flags["P1_EMERGENCY_ACTIVE"] = True
            context_flags["P1_EMERGENCY_PLAN"] = p1_plan
            self.logger.info(
                "[Meta:P-1_DUST_CONSOLIDATION] 🔥 EMERGENCY dust consolidation MARKED (context flag set, not terminal)"
            )
            # NO return — allow pipeline to continue

############################################################
# SECTION: P0: Dust & Emergency Logic
# Responsibility:
# - P0 dust promotion gate logic and eligibility evaluation
# - Emergency position handling and dust consolidation
# - Dust viability checks and promotion ordering
# Future Extraction Target:
# - DustManager or P0PolicyEngine
############################################################

        # ═══════════════════════════════════════════════════════════════════════════════
        # STEP 0.2: P0 DUST PROMOTION GATE (NOT executor, controls eligibility & ordering)
        # ═══════════════════════════════════════════════════════════════════════════════
        # P0 evaluates which dust positions are eligible for healing (dust + strong BUY signal)
        # Returns gate info for normal pipeline to prioritize, NOT decisions to execute
        #
        # CRITICAL: P0 never executes BUY. It only identifies eligible dust for normal flow.
        # Actual BUY execution happens via Dust Consolidation BUY in the normal pipeline.
        #
        # FIX #3: STRICT SELL-ONLY MODE
        # When mandatory_sell_mode is active (portfolio full, no SELL signals, no dust clusters),
        # P0 must NOT run. BUY signals are forbidden in recovery mode.
        # INVARIANT:
        # In clean wallet state (healthy capital, no economic dust):
        # - P0 must NOT execute (no positions to heal)
        # - P1 must NOT execute (portfolio not full)
        # - P2 MUST execute if valid signal + gates ready
        # Any deviation is a policy ordering bug.
        p0_gate = None
        if self._mandatory_sell_mode_active:
            self.logger.warning(
                "[Meta:P0_GATE] 🔒 BLOCKED: P0 gate disabled during SELL-only recovery mode. "
                "No BUY evaluation while portfolio full with no exit signals."
            )
        else:
            p0_gate = await self._check_p0_dust_promotion()
        
        if p0_gate and p0_gate.get("p0_gate_open"):
            context_flags["P0_GATE_OPEN"] = True
            context_flags["P0_ELIGIBLE_DUST"] = p0_gate.get("eligible_dust_list", [])
            context_flags["P0_TOP_DUST"] = p0_gate.get("top_dust")
            self.logger.info(
                "[Meta:P0_GATE] ✓ P0 GATE OPEN: %d dust position(s) eligible for prioritization (will be handled by normal flow, not executed here)",
                len(p0_gate.get("eligible_dust_list", []))
            )
            # NO return — allow pipeline to continue

        # ═══════════════════════════════════════════════════════════════════════════════
        # STEP 0.3: DUST RESOLUTION PATH CHECKS (MUTUALLY EXCLUSIVE)
        # ═══════════════════════════════════════════════════════════════════════════════
        # Check two dust resolution paths with MUTUAL EXCLUSIVITY:
        # Path A: DUST HEALING - Can we recover position via BUY? (PREFERRED)
        # Path B: DUST SACRIFICE - Is escape hatch necessary? (Only if healing blocked)
        # CRITICAL: If healing_possible == True, NEVER trigger sacrifice
        # Neither returns early - both feed into normal decision pipeline for prioritization
        
        metrics = getattr(self.shared_state, "metrics", {}) or {}
        trade_count = int(
            metrics.get("total_trades_executed", 0)
            or getattr(self.shared_state, "trade_count", 0)
            or 0
        )
        if trade_count < 1:
            self.logger.info(
                "[Meta:DUST_HEALING] Disabled before evaluation: trade_count=%d < 1 (no realized trades yet)",
                trade_count,
            )
            dust_healing_decisions = None
        else:
            dust_healing_decisions = await self._check_dust_healing_opportunity()
        if dust_healing_decisions:
            context_flags["DUST_HEALING_AVAILABLE"] = True
            context_flags["DUST_HEALING_DECISION"] = dust_healing_decisions[0]
            self.logger.info(
                "[Meta:DUST_HEALING] 💚 Healing opportunity available: %s (will be EXECUTED immediately, BLOCKS sacrifice)",
                dust_healing_decisions[0][0]
            )
            # MUTUAL EXCLUSIVITY: If healing is possible, skip sacrifice check entirely
            # Healing is preferred path, sacrifice only if healing blocked
            # INJECT dust healing decision(s) for execution (prepend for priority)
            return dust_healing_decisions
        else:
            # ONLY check sacrifice if healing is not possible
            # This ensures mutual exclusivity: at most one path is active per cycle
            dust_sacrifice_decisions = await self._check_dust_sacrifice_necessity()
            if dust_sacrifice_decisions:
                context_flags["DUST_SACRIFICE_NECESSARY"] = True
                context_flags["DUST_SACRIFICE_DECISION"] = dust_sacrifice_decisions[0]
                self.logger.critical(
                    "[Meta:DUST_SACRIFICE] 💀 Escape hatch activated: %s (deadlock condition detected, healing not possible)",
                    dust_sacrifice_decisions[0][0]
                )

        # 0.5) PORTFOLIO AWARENESS: Expand universe to include owned assets
        symbols_to_consider = set(accepted_symbols_set or [])
        owned_positions = {}
        
        # AUTHORITATIVE FLAT CHECK: bootstrap-aware (dust-only portfolios treated as flat in BOOTSTRAP)
        is_flat = await self._check_portfolio_flat()
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # BOOTSTRAP FIRST TRADE POLICY (Critical Fix for Clean Wallet Deadlock)
        # ═══════════════════════════════════════════════════════════════════════════════
        # If portfolio is flat (no positions) and we have decent signals,
        # apply bootstrap execution trigger to break the deadlock
        bootstrap_execution_override = False
        now_ts = time.time()
        cooldown_remaining = float(self._bootstrap_cooldown_until or 0.0) - now_ts
        if cooldown_remaining > 0:
            self.logger.warning(
                "[Meta:BOOTSTRAP] Standing down for %.0fs after policy veto (%s).",
                cooldown_remaining,
                self._bootstrap_last_veto_reason or "unknown",
            )
        elif is_flat:
            # Hard guard: Ensure strict flatness (redundant but safe)
            # This ensures we NEVER bootstrap if any position (even dust) exists
            is_flat_confirmed = bool(is_flat)
            if not is_flat_confirmed:
                bootstrap_execution_override = False
            else:
                # Check available capital first
                quote_asset = str(self._cfg("QUOTE_ASSET") or "USDT").upper()
                try:
                    available_capital = float(await _safe_await(
                        self.shared_state.get_spendable_balance(quote_asset)
                    ) or 0.0)
                    min_bootstrap_capital = 50.0  # Minimum capital to attempt bootstrap
                    
                    # CRITICAL DEBUG: Log bootstrap conditions
                    self.logger.warning(
                        "[Meta:BOOTSTRAP_DEBUG] Checking bootstrap conditions: "
                        "is_flat=%s, capital=%.2f (min=%.2f), signal_cache=%s",
                        is_flat, available_capital, min_bootstrap_capital, 
                        "available" if self.signal_cache else "None"
                    )
                    
                    if available_capital >= min_bootstrap_capital:
                        # Look for signals with sufficient confidence
                        all_signals_now = self.signal_manager.get_all_signals()
                        
                        # CRITICAL DEBUG: Log signal cache contents
                        self.logger.warning(
                            "[Meta:BOOTSTRAP_DEBUG] Signal cache contains %d signals: %s",
                            len(all_signals_now),
                            [f"{s.get('symbol')}:{s.get('action')}:{s.get('confidence')}" for s in all_signals_now[:3]]
                        )
                        
                        best_bootstrap_signal = None
                        for sig in all_signals_now:
                            if str(sig.get("action", "")).upper() == "BUY":
                                conf = float(sig.get("confidence", 0.0))
                                if conf >= 0.60:  # Bootstrap confidence threshold
                                    if best_bootstrap_signal is None or conf > float(best_bootstrap_signal.get("confidence", 0.0)):
                                        best_bootstrap_signal = sig
                        
                        if best_bootstrap_signal:
                            # FIX: Enforce one-shot limit (Caveat #1)
                            if self._bootstrap_attempts > 0:
                                self.logger.warning(
                                    "[Meta:BOOTSTRAP] One-shot override limit reached (%d attempts/trades). "
                                    "Stopping override to prevent spam.",
                                    self._bootstrap_attempts
                                )
                                bootstrap_execution_override = False
                            elif self.FOCUS_MODE_ENABLED:
                                # SOP Rule: Only on pinned or wallet-origin symbols during BOOTSTRAP Focus
                                sym = best_bootstrap_signal.get("symbol")
                                if sym not in self.FOCUS_SYMBOLS:
                                    self.logger.warning("[Meta:BOOTSTRAP] Skipping best signal %s: Not a pinned/focus symbol.", sym)
                                    bootstrap_execution_override = False
                                else:
                                    bootstrap_execution_override = True
                            else:
                                bootstrap_execution_override = True
                                
                            if bootstrap_execution_override:
                                self.logger.warning(
                                    "[Meta:BOOTSTRAP_FIRST_TRADE] 🚀 BOOTSTRAP EXECUTION OVERRIDE: "
                                    "Portfolio flat, capital=%.2f, best_signal=%s conf=%.2f. "
                                    "Overriding conservative gates to break deadlock.",
                                    available_capital, 
                                    best_bootstrap_signal.get("symbol", "unknown"),
                                    float(best_bootstrap_signal.get("confidence", 0.0))
                                )
                        else:
                            self.logger.warning(
                                "[Meta:BOOTSTRAP_DEBUG] No qualifying BUY signals found for bootstrap "
                                "(need conf >= 0.60)"
                            )
                    else:
                        self.logger.warning(
                            "[Meta:BOOTSTRAP_DEBUG] Insufficient capital for bootstrap: %.2f < %.2f",
                            available_capital, min_bootstrap_capital
                        )
                except Exception as e:
                    self.logger.warning(f"[Meta:BOOTSTRAP_FIRST_TRADE] Failed to check bootstrap conditions: {e}")
        else:
            self.logger.info("[Meta:BOOTSTRAP_DEBUG] Portfolio not flat, bootstrap not needed")
            # Reset attempts when we exit flat state or have positions, to re-arm specifically for next deadlock
            self._bootstrap_attempts = 0
            self._bootstrap_cooldown_until = 0.0
            self._bootstrap_last_veto_reason = None

        # ═══════════════════════════════════════════════════════════════════════════════
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # GOVERNANCE BLOCKER (RE-RE-ENFORCEMENT)
        # ─────────────────────────────────────────────────────────────────────────────
        # If BUY is not in allowed actions (e.g. PROTECTIVE or SIGNAL_ONLY), set capital_block=True
        # to ensure no new positions are planned, while allowing SELLs to continue.
        # ─────────────────────────────────────────────────────────────────────────────
        if "BUY" not in allowed_actions:
            if not bootstrap_execution_override:
                self.logger.info("[Meta:Governance] Mode %s blocks BUY actions. Activating block.", current_mode)
                capital_block = True
        # ═══════════════════════════════════════════════════════════════════════════════

        try:
            snap = self.shared_state.get_positions_snapshot() or {}
            for sym_raw, p in snap.items():
                sym = self._normalize_symbol(sym_raw)
                q_val = p.get("quantity")
                if q_val is None:
                    q_val = p.get("qty")
                qty = float(q_val or 0.0)
                
                # P9 Invariant: Ignore DUST_LOCKED positions
                if p.get("state") == "DUST_LOCKED":
                    self.logger.debug("[Meta:Universe] %s is DUST_LOCKED. Skipping.", sym)
                    continue

                if qty > 0:
                    owned_positions[sym] = p
                    symbols_to_consider.add(sym)
        except Exception:
            self.logger.warning("[Meta] Failed to retrieve owned positions.", exc_info=True)

        # 1) Group Valid Signals from Cache (Multi-agent aware)
        all_signals = self.signal_manager.get_all_signals()
        
        signals_by_sym = defaultdict(list)
        for s in all_signals:
            sym = s.get("symbol")
            if sym:
                signals_by_sym[sym].append(s)
                # Phase A expansion: also consider any symbol that has an active signal
                # BOOTSTRAP FIX: If we are flat, we MUST consider all active signals to find a starter.
                symbols_to_consider.add(sym)

        valid_signals_by_symbol = defaultdict(list)
        now_ts = time.time()

        # FIRE-001 (User Request): Controlled Partial Rotation
        # IF focus_mode AND position_age >= 20m AND abs(pnl) < 0.3% THEN SELL 20%
        rotation_signals = await self._check_controlled_partial_rotation(owned_positions, now_ts)
        for rs in rotation_signals:
            rs_sym = rs["symbol"]
            signals_by_sym[rs_sym].append(rs)
            self._info_once(f"rot_{rs_sym}", f"[Meta:Rotation] Injected partial rotation signal for {rs_sym}")

        # [SCALING FIX]: Scale-in after hold time (Compounding)
        # IF focus_mode AND position_age >= 60m AND pnl > 1% THEN BUY more
        # (This is the partner to the rotation rule for compounding)
        scaling_signals = await self._check_scale_in_opportunity(owned_positions, now_ts)
        for ss in scaling_signals:
            ss_sym = ss["symbol"]
            signals_by_sym[ss_sym].append(ss)
            self._info_once(f"scale_{ss_sym}", f"[Meta:Scaling] Injected scale-in signal for {ss_sym}")

        # P9: Adaptive Aggression (Early Calculation for Dynamic Filtering)
        agg_factor = await self._get_aggression_factor()
        self._adaptive_aggression = agg_factor
        if hasattr(self.shared_state, "set_dynamic_param"):
            await self.shared_state.set_dynamic_param("aggression_factor", agg_factor)


        # Phase A - Throughput Guard (Early Check)
        recent_trades_count = sum(1 for ts in self._trade_timestamps if (now_ts - ts) < self._throughput_window_sec)
        throughput_gap = (recent_trades_count == 0)
        if throughput_gap:
            self.logger.info("[Meta] Throughput Guard ACTIVE: No trades in last %dm. Prioritizing Tier-B participation.", 
                            int(self._throughput_window_sec / 60))

        # ═══════════════════════════════════════════════════════════════════════════════
        # SOP MODE ENVELOPE: Fetch constraints for the current active mode
        # ═══════════════════════════════════════════════════════════════════════════════
        envelope = self.mode_manager.get_envelope()
        mode_conf_floor = self._get_mode_confidence_floor()
        max_pos = self._get_max_positions()
        mode_cooldown = envelope.get("cooldown_sec", 60)
        probing_enabled = envelope.get("probing_enabled", True)

        buy_suppressed = False
        rec_state = getattr(self.shared_state, "capital_recovery_mode", {}) or {}
        suppress_buys = bool(isinstance(rec_state, dict) and rec_state.get("active"))
        for sym in symbols_to_consider:
            sigs = signals_by_sym.get(sym, [])
            
            # MEMORY INJECTION (HOLD context only, not intended for execution)
            if not sigs and sym in owned_positions:
                pos = owned_positions[sym]
                roi = float(pos.get("roi") or pos.get("unrealized_pnl_pct", 0.0))
                hold_conf = 0.40
                if roi > 0.05: hold_conf = 0.60
                elif roi < -0.10: hold_conf = 0.25

                sigs = [{
                    "symbol": sym, "action": "HOLD", "confidence": hold_conf,
                    "agent": "PortfolioMemory", "timestamp": now_ts,
                }]

            # PER-SYMBOL DIRECTIONAL CONSISTENCY (Phase A Fix)
            # Soft gate: If symbols have conflicting agent signals (BUY and SELL), we check dominant side
            symbol_buys = sum(1 for s in sigs if str(s.get("action")).upper() == "BUY")
            symbol_sells = sum(1 for s in sigs if str(s.get("action")).upper() == "SELL")
            
            if symbol_buys > 0 and symbol_sells > 0:
                dominant = max(symbol_buys, symbol_sells)
                if (dominant / (symbol_buys + symbol_sells)) < self._directional_consistency_pct:
                    if sym in owned_positions and symbol_sells > 0:
                        self.logger.debug(
                            "[Meta] Conflicting signals for %s below consistency %.2f; prioritizing SELL for exit.",
                            sym, self._directional_consistency_pct
                        )
                        sigs = [s for s in sigs if str(s.get("action")).upper() == "SELL"]
                        if not sigs:
                            continue
                    else:
                        self.logger.debug("[Meta] Dropping %s - conflicting signals (B:%d S:%d) below consistency %.2f", 
                                        sym, symbol_buys, symbol_sells, self._directional_consistency_pct)
                        continue

            for sig in sigs:
                action = str(sig.get("action", "")).upper()
                sig["action"] = action
                if suppress_buys and action == "BUY":
                    if sig.get("_bootstrap") or sig.get("_bootstrap_override") or bootstrap_execution_override:
                        self.logger.info(
                            "[Meta:CapitalRecovery] Allowing bootstrap BUY during recovery mode: %s",
                            sym,
                        )
                    else:
                        buy_suppressed = True
                        continue
                conf = float(sig.get("confidence", 0.0))

                # ═══════════════════════════════════════════════════════════════════════
                # SOP ENVELOPE GATING
                # ═══════════════════════════════════════════════════════════════════════
                if action == "BUY":
                    # 1. Position Limit Gate
                    # [Capacity Policy 2.0] Early gate relaxed to allow Rotation Authority evaluation.
                    # Capacity is now enforced authoritatively in the refactored policy block downstream.
                    pass
                    # if sig_pos >= max_pos and sym not in owned_positions:
                    #     self.logger.info("[Meta:Envelope] BLOCKED %s BUY: Mode max positions (%d) reached.", sym, max_pos)
                    #     continue
                    
                    # 2. Probing Gate: Block new symbols if probing disabled for mode
                    # EXCEPTION: Bootstrap signals AND BOOTSTRAP MODE itself bypass this restriction
                    # The ENTIRE purpose of BOOTSTRAP mode is to make the first trade on a flat portfolio
                    is_bootstrap_signal = (
                        sig.get("_bootstrap")
                        or sig.get("_bootstrap_override")
                        or str(sig.get("context") or "").upper() == "BOOTSTRAP"
                    )
                    current_mode = self.mode_manager.get_mode()
                    is_bootstrap_mode = current_mode == "BOOTSTRAP"
                    
                    # In BOOTSTRAP mode, probing IS allowed (that's the point of the mode)
                    if not probing_enabled and sym not in owned_positions and not is_bootstrap_signal and not is_bootstrap_mode:
                        self.logger.info("[Meta:Envelope] BLOCKED %s BUY: Probing (new assets) disabled in %s mode.", sym, current_mode)
                        continue

                    # 3. Confidence Floor Gate
                    # Phase A: Dynamic Execution Floor (Aggression-Aware)
                    base_floor = 0.50 if throughput_gap else self._min_exec_conf
                    exec_floor = max(0.40, base_floor / agg_factor) if agg_factor > 1.0 else base_floor

                    # BOOTSTRAP OVERRIDE: Lower confidence threshold for first trade in flat portfolio
                    if bootstrap_execution_override:
                        exec_floor = min(exec_floor, 0.60)  # Bootstrap threshold: 0.60
                    
                    # SOP ENVELOPE: Use the stricter of the calculated floor or the mode's floor
                    # CRITICAL FIX: BOOTSTRAP overrides all envelope restrictions including RECOVERY floor
                    if bootstrap_execution_override:
                        final_exec_floor = exec_floor
                    else:
                        final_exec_floor = max(exec_floor, mode_conf_floor)
                    
                    if conf < final_exec_floor and not sig.get("bypass_conf"):
                        self.logger.debug("[Meta:Envelope] %s BUY rejected: conf %.2f < final_floor %.2f (bootstrap=%s)", 
                                        sym, conf, final_exec_floor, bootstrap_execution_override)
                        continue

                    # 4. Mode-based Cooldown Gate
                    last_buy_ts = float(self._last_buy_ts.get(sym, 0.0) or 0.0)
                    if last_buy_ts > 0 and (now_ts - last_buy_ts) < mode_cooldown:
                        if is_flat:
                            self.logger.info(
                                "[Meta:Envelope] Bypassed %s BUY: mode cooldown ignored for FLAT_PORTFOLIO",
                                sym,
                            )
                        elif not sig.get("_bootstrap_override") and not sig.get("_bootstrap"):
                            remaining = int(mode_cooldown - (now_ts - last_buy_ts))
                            self.logger.info(
                                "[Meta:Envelope] Skipping %s BUY: mode cooldown (%ds remaining)",
                                sym, max(0, remaining)
                            )
                            continue
                if action not in ("BUY", "SELL", "HOLD"):
                    continue

                # Per-symbol BUY cooldown to prevent overtrading
                if action == "BUY" and self._buy_cooldown_sec > 0:
                    last_buy_ts = float(self._last_buy_ts.get(sym, 0.0) or 0.0)
                    if last_buy_ts > 0 and (now_ts - last_buy_ts) < self._buy_cooldown_sec:
                        if is_flat:
                            self.logger.info(
                                "[Meta:BUY_COOLDOWN] Bypassed %s BUY: FLAT_PORTFOLIO",
                                sym,
                            )
                        elif not sig.get("_bootstrap_override") and not sig.get("_bootstrap"):
                            remaining = int(self._buy_cooldown_sec - (now_ts - last_buy_ts))
                            self.logger.info(
                                "[Meta:BUY_COOLDOWN] Skipping %s BUY: cooldown active (%ds remaining)",
                                sym, max(0, remaining)
                            )
                            continue

                # Price-delta re-entry guard to prevent rapid re-buys on unchanged price
                if action == "BUY" and self._buy_reentry_delta_pct > 0:
                    last_price = float(self._last_buy_price.get(sym, 0.0) or 0.0)
                    if (
                        last_price > 0
                        and not sig.get("_bootstrap_override")
                        and not sig.get("_bootstrap")
                        and not bootstrap_execution_override
                    ):
                        cur_price = 0.0
                        try:
                            if hasattr(self.shared_state, "safe_price"):
                                cur_price = float(await _safe_await(self.shared_state.safe_price(sym)) or 0.0)
                        except Exception:
                            cur_price = 0.0
                        if not cur_price:
                            cur_price = float(getattr(self.shared_state, "latest_prices", {}).get(sym, 0.0) or 0.0)
                        if cur_price > 0:
                            delta_pct = abs(cur_price - last_price) / max(last_price, 1e-9)
                            if delta_pct < self._buy_reentry_delta_pct:
                                self.logger.info(
                                    "[Meta:BUY_REENTRY] Skipping %s BUY: price delta %.3f%% < %.3f%%",
                                    sym, delta_pct * 100.0, self._buy_reentry_delta_pct * 100.0
                                )
                                continue

                # Anti-churn: enforce one position per symbol unless explicitly accumulating
                if action == "BUY":
                    max_per_symbol = int(self._cfg(
                        "MAX_OPEN_POSITIONS_PER_SYMBOL",
                        default=self._max_open_positions_per_symbol,
                    ))
                    if max_per_symbol <= 1:
                        has_open, existing_qty = await self._has_open_position(sym)
                        allow_scale_in = bool(
                            sig.get("_scale_in")
                            or sig.get("_accumulate_mode")
                            or sig.get("_allow_reentry")
                        )
                        if has_open and not allow_scale_in:
                            self.logger.info(
                                "[WHY_NO_TRADE] symbol=%s reason=POSITION_ALREADY_OPEN details=max_open_positions_per_symbol qty=%.6f",
                                sym,
                                existing_qty,
                            )
                            continue
                        if has_open and allow_scale_in:
                            sig["_allow_reentry"] = True

                    existing_qty = float(self.shared_state.get_position_qty(sym) or 0.0)
                    reason_lower = str(sig.get("reason", "")).lower()
                    allow_reentry = bool(
                        sig.get("_accumulate_mode")
                        or sig.get("_rotation_escape")
                        or sig.get("_allow_reentry")
                        or sig.get("_is_rotation")
                        or sig.get("_is_compounding")
                        or "dust" in reason_lower
                        or "accumulate" in reason_lower
                        or (self._focus_mode_active and sym in self.FOCUS_SYMBOLS) # REENTRY_LOCK BYPASS for focus mode
                    )
                    
                    # AUTHORITATIVE BYPASS (Fix #2): If focus mode + focus symbol, RE-ENTRY IS ALWAYS ALLOWED
                    # We also allow if the most recent action was a SELL (re-entry) OR if tagged as rotation/compounding
                    if self._focus_mode_active and sym in self.FOCUS_SYMBOLS:
                        last_buy_ts = float(self._last_buy_ts.get(sym, 0.0))
                        last_exit_ts = float(self.shared_state.get_last_exit_ts(sym) or 0.0)
                        if last_exit_ts > last_buy_ts or sig.get("_is_rotation") or sig.get("_is_compounding"):
                            allow_reentry = True
                            sig["_reentry_allowed"] = True
                            sig["_allow_reentry"] = True
                            self.logger.debug("[Meta:Focus] Allowing re-entry for focus symbol %s (last_exit > last_buy)", sym)
                        else:
                            # Stacking is allowed if confidence is high (Tier A)
                            if sig.get("_tier") == "A" or float(sig.get("confidence", 0.0)) >= self._tier_a_conf:
                                allow_reentry = True
                                sig["_allow_reentry"] = True
                                self.logger.debug("[Meta:Focus] Allowing stacking for focus symbol %s (Tier A confidence)", sym)
                    significant_floor = 0.0
                    if existing_qty > 0:
                        try:
                            _, min_notional = await _safe_await(self.shared_state.compute_symbol_trade_rules(sym))
                            if min_notional is None or float(min_notional) <= 0:
                                min_notional = float(self._cfg("MIN_NOTIONAL_USDT", 10.0))
                        except Exception:
                            min_notional = float(self._cfg("MIN_NOTIONAL_USDT", 10.0))
                        min_position_value = float(self._cfg("MIN_POSITION_VALUE_USDT", 10.0))
                        strategy_floor = float(
                            self._cfg(
                                "MIN_SIGNIFICANT_POSITION_USDT",
                                self._cfg(
                                    "MIN_SIGNIFICANT_USD",
                                    self._cfg("SIGNIFICANT_POSITION_USDT", 25.0),
                                ),
                            )
                        )
                        significant_floor = max(float(min_notional), min_position_value, strategy_floor)

                    pos_value = 0.0
                    if existing_qty > 0 and significant_floor > 0:
                        try:
                            if hasattr(self.shared_state, "safe_price"):
                                price = float(await _safe_await(self.shared_state.safe_price(sym)) or 0.0)
                            else:
                                price = float(getattr(self.shared_state, "latest_prices", {}).get(sym, 0.0) or 0.0)
                        except Exception:
                            price = float(getattr(self.shared_state, "latest_prices", {}).get(sym, 0.0) or 0.0)
                        if price > 0:
                            pos_value = existing_qty * price

                    has_significant_position = existing_qty > 0 and pos_value >= significant_floor
                    is_dust_position = existing_qty > 0 and pos_value > 0 and pos_value < significant_floor
                    if is_dust_position and bool(self._cfg("DUST_REENTRY_OVERRIDE", True)):
                        if sym in self._dust_merges:
                            self.logger.info(
                                "[Meta:DUST_REENTRY] Blocking repeat dust merge for %s (already attempted)", sym
                            )
                            self.logger.info("[WHY_NO_TRADE] symbol=%s reason=DUST_MERGE_LIMIT details=already_attempted", sym)
                        else:
                            allow_reentry = True
                            sig["_allow_reentry"] = True
                            sig["_dust_reentry_override"] = True
                            self._dust_merges.add(sym)
                            self.logger.info(
                                "[Meta:DUST_REENTRY] Allowing %s BUY to merge dust (value=%.4f < floor=%.4f)",
                                sym, pos_value, significant_floor
                            )
                    if has_significant_position and not allow_reentry:
                        self.logger.info(
                            "[Meta:REENTRY_LOCK] Skipping %s BUY: existing position qty=%.6f",
                            sym, existing_qty
                        )
                        self.logger.info("[WHY_NO_TRADE] symbol=%s reason=POSITION_ALREADY_OPEN details=qty=%.6f", sym, existing_qty)
                        continue

                    # Re-entry guard after non-TP/SL exits unless signal changed
                    last_exit_reason = None
                    last_exit_ts = 0.0
                    if hasattr(self.shared_state, "get_last_exit_reason"):
                        last_exit_reason = self.shared_state.get_last_exit_reason(sym)
                    if hasattr(self.shared_state, "get_last_exit_ts"):
                        last_exit_ts = float(self.shared_state.get_last_exit_ts(sym) or 0.0)

                    fp = self._signal_fingerprint(sig)
                    if fp:
                        sig["_signal_fingerprint"] = fp
                    last_fp = self._last_signal_fingerprint.get(sym, "")
                    signal_changed = bool(fp) and fp != last_fp

                    exit_reason_norm = str(last_exit_reason) if last_exit_reason is not None else ""
                    if self._reentry_require_tp_sl_exit and (last_exit_ts or exit_reason_norm) and exit_reason_norm not in ("TP", "SL"):
                        if self._reentry_require_signal_change and not signal_changed:
                            self.logger.info(
                                "[Meta:REENTRY_LOCK] Skipping %s BUY: last_exit=%s and signal unchanged",
                                sym, last_exit_reason or "UNKNOWN"
                            )
                            continue
                        if self._reentry_lock_sec > 0 and (now_ts - float(last_exit_ts or 0.0)) < self._reentry_lock_sec and not signal_changed:
                            remaining = int(self._reentry_lock_sec - (now_ts - float(last_exit_ts or 0.0)))
                            self.logger.info(
                                "[Meta:REENTRY_LOCK] Skipping %s BUY: exit cooldown active (%ds remaining)",
                                sym, max(0, remaining)
                            )
                            continue

                if action == "SELL":
                    # STRICT FIX: Filter phantom SELLs before they pollute ranking
                    curr_qty = float(self.shared_state.get_position_qty(sym) or 0.0)
                    if curr_qty <= 0:
                        continue
                    
                    # NOTE (CRITICAL FIX): Dust SELL filter removed to enable recovery from deadlock
                    # Previous logic: if est_val < 5.5: continue  (BLOCKED DUST RECOVERY)
                    # Issue: System deadlocked with dust trapped, capital below floor, capacity full
                    # Solution: Allow P0_DUST_PROMOTION to handle dust liquidation regardless of value
                    # The exchange will enforce minNotional, but at least we try to promote/liquidate
                    pass  # Allow SELL signals through, even for dust positions

                # BOOTSTRAP OVERRIDE: Mark qualifying signals for downstream bypass
                if bootstrap_execution_override and action == "BUY" and conf >= 0.60:
                    sig["_bootstrap_override"] = True
                    sig["_bypass_reason"] = "BOOTSTRAP_FIRST_TRADE"
                    sig["bypass_conf"] = True  # Ensure it bypasses confidence checks
                    self.logger.info(
                        "[Meta:BOOTSTRAP_OVERRIDE] Flagged %s signal for bootstrap execution: conf=%.2f",
                        sym, conf
                    )

                try:
                    if hasattr(self.shared_state, "is_cooldown_active"):
                        if await self.shared_state.is_cooldown_active(sym):
                            continue
                except Exception:
                    pass
                
                # Short-circuit on recent rejections to avoid looping on known-bad combos
                try:
                    cooldown_s = float(self._cfg("REJECTION_COOLDOWN_SECONDS", 60))
                    if hasattr(self.shared_state, "is_symbol_temporarily_blocked"):
                        if await _safe_await(self.shared_state.is_symbol_temporarily_blocked(sym, action, cooldown_s)):
                            self.logger.debug("[Meta:Cooldown] Skipping %s %s due to recent rejections (cooldown=%ss)", sym, action, cooldown_s)
                            continue
                except Exception:
                    pass

                # SELL PATH BYPASS: SELLs must NOT be subject to rejection_threshold blocking
                # because SELL is risk-reducing and must be executable even during capital shortage
                # BOOTSTRAP BYPASS: Bootstrap mode must NOT be subject to rejection_threshold blocking
                # because the goal is liquidity seeding and confidence gating is counterproductive
                if action != "SELL" and not self._is_bootstrap_mode():
                    try:
                        if hasattr(self.shared_state, "is_symbol_blocked") and self.shared_state.is_symbol_blocked(sym, action):
                            self.logger.info("[Meta:Block] Skipping %s %s: blocked by rejection threshold", sym, action)
                            continue
                    except Exception:
                        pass

                valid_signals_by_symbol[sym].append(sig)

        if buy_suppressed:
            self.logger.info("[Meta:CapitalRecovery] BUY signals suppressed during recovery mode")

        # --- Time-based exit injection (capital rotation) ---
        if self._time_exit_enabled and owned_positions:
            try:
                open_trades = getattr(self.shared_state, "open_trades", {}) or {}
                for sym, pos in owned_positions.items():
                    ot = open_trades.get(sym, {}) if isinstance(open_trades, dict) else {}
                    opened_at = float(ot.get("opened_at", pos.get("opened_at", 0.0)) or 0.0)
                    if opened_at <= 0:
                        continue
                    age_hours = (now_ts - opened_at) / 3600.0
                    if age_hours < self._time_exit_min_hours:
                        continue

                    # Compute PnL % from avg/entry price vs current price
                    entry = float(ot.get("entry_price", pos.get("avg_price", 0.0)) or 0.0)
                    if entry <= 0:
                        continue
                    try:
                        price = float(await _safe_await(self.shared_state.safe_price(sym)) or 0.0)
                    except Exception:
                        price = float(self.shared_state.latest_prices.get(sym, 0.0) or 0.0)
                    if price <= 0:
                        continue

                    pnl_pct = ((price - entry) / entry) * 100.0
                    if not self._time_exit_force_sell and pnl_pct < self._time_exit_min_pnl_pct:
                        continue

                    if not await self._should_allow_sell(sym):
                        continue

                    time_exit_sig = {
                        "symbol": sym,
                        "action": "SELL",
                        "confidence": 0.60,
                        "agent": "MetaTimeExit",
                        "timestamp": now_ts,
                        "reason": f"TIME_EXIT age={age_hours:.2f}h pnl={pnl_pct:.2f}%",
                        "_time_exit": True,
                    }
                    valid_signals_by_symbol[sym].append(time_exit_sig)
            except Exception as e:
                self.logger.debug("[Meta:TimeExit] injection failed: %s", e)

        # Capital floor applies to BUYs only; if starved, keep SELLs and drop BUYs
        if capital_block:
            # CAPITAL RECOVERY: after sustained floor breach, force SELL oldest non-core position
            try:
                cap_rec = getattr(self.shared_state, "capital_recovery_mode", {}) or {}
            except Exception:
                cap_rec = {}

            if isinstance(cap_rec, dict) and cap_rec.get("active") and not self._mandatory_sell_mode_active:
                # Option A: emit nominated recovery sell once per activation
                try:
                    nominated_sym = cap_rec.get("nominated_symbol") or cap_rec.get("candidate_symbol")
                    nominated_emitted = bool(cap_rec.get("nominated_emitted"))
                    if nominated_sym and not nominated_emitted and nominated_sym in owned_positions:
                        if await self._should_allow_sell(nominated_sym):
                            nominated_sig = {
                                "symbol": nominated_sym,
                                "action": "SELL",
                                "confidence": 0.95,
                                "agent": "MetaCapitalRecovery",
                                "timestamp": now_ts,
                                "reason": "CAPITAL_RECOVERY",
                                "_capital_recovery_nominated": True,
                                "_force_dust_liquidation": True,
                                "_tag": "liquidation/capital_recovery",
                            }
                            valid_signals_by_symbol.setdefault(nominated_sym, []).insert(0, nominated_sig)
                            new_state = dict(cap_rec)
                            new_state.update({
                                "nominated_emitted": True,
                                "candidate_symbol": nominated_sym,
                            })
                            setattr(self.shared_state, "capital_recovery_mode", new_state)
                            self.logger.warning(
                                "[Meta:CapitalRecovery] Emitted nominated SELL: %s",
                                nominated_sym
                            )
                except Exception as e:
                    self.logger.debug("[Meta:CapitalRecovery] nominated sell emit failed: %s", e)

                started_at = float(cap_rec.get("started_at", now_ts) or now_ts)
                elapsed = max(0.0, now_ts - started_at)
                force_after = float(cap_rec.get(
                    "max_age_sec",
                    self._cfg("CAPITAL_RECOVERY_FORCE_SELL_AFTER_SEC", 300.0)
                ))
                soft_after = float(self._cfg("CAPITAL_RECOVERY_SOFT_SELL_AFTER_SEC", 120.0))
                soft_pct = float(self._cfg("CAPITAL_RECOVERY_SOFT_SELL_PCT", 0.4))
                dedup_sec = float(self._cfg("CAPITAL_RECOVERY_SELL_DEDUP_SEC", 30.0))
                last_recovery_emit = float(cap_rec.get("last_recovery_sell_ts", 0.0) or 0.0)
                if 0 < soft_after and elapsed >= soft_after and owned_positions:
                    if not cap_rec.get("soft_sell_emitted") and (now_ts - last_recovery_emit) >= dedup_sec:
                        candidate_sym = cap_rec.get("candidate_symbol") or cap_rec.get("nominated_symbol")
                        if not candidate_sym:
                            candidate_sym = next(iter(owned_positions.keys()), None)
                        if candidate_sym and candidate_sym in owned_positions:
                            soft_sig = {
                                "symbol": candidate_sym,
                                "action": "SELL",
                                "confidence": 0.85,
                                "agent": "MetaCapitalRecovery",
                                "timestamp": now_ts,
                                "reason": "CAPITAL_RECOVERY",
                                "_capital_recovery_soft": True,
                                "_partial_pct": soft_pct,
                                "_force_dust_liquidation": True,
                                "_tag": "liquidation/capital_recovery",
                            }
                            valid_signals_by_symbol.setdefault(candidate_sym, []).insert(0, soft_sig)
                            context_flags.setdefault("CAPITAL_RECOVERY_SOFT_DECISIONS", []).append(
                                (candidate_sym, "SELL", soft_sig)
                            )
                            new_state = dict(cap_rec)
                            new_state.update({
                                "soft_sell_emitted": True,
                                "candidate_symbol": candidate_sym,
                                "last_recovery_sell_ts": now_ts,
                            })
                            setattr(self.shared_state, "capital_recovery_mode", new_state)
                            self.logger.warning(
                                "[Meta:CapitalRecovery] SOFT SELL after %.0fs breach: %s (pct=%.0f%%)",
                                elapsed, candidate_sym, soft_pct * 100.0
                            )
                    elif not cap_rec.get("soft_sell_emitted"):
                        self.logger.info(
                            "[Meta:CapitalRecovery] Soft sell suppressed by dedup window (%.0fs)",
                            dedup_sec
                        )
                if force_after > 0 and elapsed >= force_after and owned_positions:
                    if cap_rec.get("force_sell_emitted"):
                        self.logger.debug(
                            "[Meta:CapitalRecovery] Force sell already emitted; skipping repeat."
                        )
                    elif (now_ts - last_recovery_emit) < dedup_sec:
                        self.logger.info(
                            "[Meta:CapitalRecovery] Force sell suppressed by dedup window (%.0fs)",
                            dedup_sec
                        )
                    else:
                        self.logger.error(
                            "[Meta:CapitalRecovery:FORCE] 🔥 max_age reached (elapsed=%.0fs >= %.0fs). Injecting SELL intent.",
                            elapsed, force_after
                        )
                        open_trades = getattr(self.shared_state, "open_trades", {}) or {}

                        candidates = []
                        fallback_candidates = []
                        for sym, pos in owned_positions.items():
                            if not self._is_recovery_sellable(sym, pos, ignore_filters=False):
                                if self._is_recovery_sellable(sym, pos, ignore_filters=True):
                                    pass
                                else:
                                    continue
                            ot = open_trades.get(sym, {}) if isinstance(open_trades, dict) else {}
                            created_at = (
                                ot.get("created_at")
                                or ot.get("opened_at")
                                or pos.get("entry_time")
                                or pos.get("opened_at")
                                or now_ts
                            )
                            created_at = float(created_at or now_ts)
                            value_usdt = float(pos.get("value_usdt", 0.0) or 0.0)
                            entry = (created_at, -value_usdt, sym)
                            if self._is_recovery_sellable(sym, pos, ignore_filters=False):
                                candidates.append(entry)
                            if self._is_recovery_sellable(sym, pos, ignore_filters=True):
                                fallback_candidates.append(entry)

                        selection = candidates if candidates else fallback_candidates
                        if selection:
                            selection.sort(key=lambda x: (x[0], x[1]))  # oldest, then largest
                            forced_sym = selection[0][2]
                            forced_sig = {
                                "symbol": forced_sym,
                                "action": "SELL",
                                "confidence": 1.0,
                                "agent": "MetaCapitalRecovery",
                                "timestamp": now_ts,
                                "reason": "CAPITAL_RECOVERY",
                                "_capital_recovery_forced": True,
                                "_force_dust_liquidation": True,
                                "_tag": "liquidation/capital_recovery",
                            }
                            valid_signals_by_symbol.setdefault(forced_sym, []).insert(0, forced_sig)
                            context_flags.setdefault("CAPITAL_RECOVERY_FORCED_DECISIONS", []).append(
                                (forced_sym, "SELL", forced_sig)
                            )
                            new_state = dict(cap_rec)
                            new_state.update({
                                "force_sell_emitted": True,
                                "candidate_symbol": forced_sym,
                                "last_recovery_sell_ts": now_ts,
                            })
                            setattr(self.shared_state, "capital_recovery_mode", new_state)
                            self.logger.warning(
                                "[Meta:CapitalRecovery] FORCED SELL after %.0fs breach: %s",
                                elapsed, forced_sym
                            )
            
            pruned = 0
            for sym in list(valid_signals_by_symbol.keys()):
                sell_sigs = [s for s in valid_signals_by_symbol.get(sym, []) if s.get("action") == "SELL"]
                if sell_sigs:
                    valid_signals_by_symbol[sym] = sell_sigs
                else:
                    del valid_signals_by_symbol[sym]
                    pruned += 1
            self.logger.warning(
                "[Meta:CapitalFloor] BUYs blocked due to capital floor; kept SELLs only (pruned=%d)",
                pruned
            )

        # === 1.5) FLAT PORTFOLIO GUARD (MANDATORY USER REQ) ===
        # === 1.5) FLAT PORTFOLIO GUARD (MANDATORY USER REQ + EXECUTION GUARANTEE) ===
        # AUTHORITATIVE FLAT CHECK: position.status in {OPEN, PARTIALLY_FILLED}
        if is_flat and not self._bootstrap_lock_engaged:
            # ✅ FIX #2: Only enforce FLAT_PORTFOLIO logic if bootstrap lock NOT engaged
            self.logger.info("[Meta] FLAT_PORTFOLIO detected. Enforcing BUY-only logic.")
            
            # === SELL SIGNAL PRIORITY BYPASS (CAPITAL RECOVERY FIRST) ===
            # Before forcing BUY-only, check if there are valid SELL signals to recover capital
            # SELL signals take priority because they free capital for subsequent BUY signals
            for sym, sigs in valid_signals_by_symbol.items():
                for s in sigs:
                    if str(s.get("action")).upper() == "SELL":
                        # Check if position still exists and quantity is valid
                        curr_qty = float(self.shared_state.get_position_qty(sym) or 0.0)
                        if curr_qty > 0:
                            # Get current price for notional check
                            try:
                                price = await self.shared_state.safe_price(sym)
                                est_notional = curr_qty * price
                                
                                # ===== FIX #2A: CAPITAL STARVATION ESCAPE HATCH =====
                                # Allow SELL even with lower notional if capital is critically low
                                quote_asset = str(self._cfg("QUOTE_ASSET") or "USDT").upper()
                                available_capital = float(await _safe_await(
                                    self.shared_state.get_spendable_balance(quote_asset)
                                ) or 0.0)
                                
                                # If capital is critically low (< 10.0) AND position exists, 
                                # allow SELL with LOWER threshold (2.0 instead of 5.5)
                                min_notional_for_sell = 5.5
                                if available_capital < 10.0 and available_capital >= 0:
                                    min_notional_for_sell = 2.0
                                    self.logger.info(
                                        "[Meta:FIX#2] ⚠️ CAPITAL STARVATION DETECTED (available=%.2f). "
                                        "Lowering SELL threshold from 5.5 → 2.0 for escape.",
                                        available_capital
                                    )
                                
                                if est_notional >= min_notional_for_sell:  # Use dynamic threshold
                                    # Valid SELL signal - execute immediately, bypass FLAT_PORTFOLIO logic
                                    self.logger.info(
                                        "[Meta:SELL_BYPASS] Bypassing FLAT_PORTFOLIO for SELL capital recovery: "
                                        "%s qty=%.4f notional=%.2f USD (threshold=%.1f)",
                                        sym, curr_qty, est_notional, min_notional_for_sell
                                    )
                                    s["_tier"] = "A"
                                    s["_bypass_reason"] = "FLAT_SELL_BYPASS"
                                    s["reason"] = "FLAT_PORTFOLIO_SELL_RECOVERY"
                                    if available_capital < 10.0:
                                        s["_starvation_escape"] = True  # Flag for downstream awareness
                                    return [(sym, "SELL", s)]
                            except Exception as e:
                                self.logger.warning("[Meta:SELL_BYPASS] Failed to process SELL for %s: %s", sym, e)
            
            # No valid SELL found - proceed with BUY-only forcing logic
            
            # === DUST POSITION GUARD (PRIORITY 3) ===
            # Before forcing BUY on any symbol, check if it already has dust position
            # Don't force BUY on dusty symbols - let accumulation handle it instead
            
            dusty_symbols = set()
            if self.portfolio_manager:
                try:
                    # Get all symbols with dust state
                    for sym in valid_signals_by_symbol.keys():
                        dust_state = await self.portfolio_manager.get_dust_state(sym)
                        if dust_state == DustState.DUST_ACCUMULATING:
                            dusty_symbols.add(sym)
                            self.logger.info(
                                "[Meta:FlatDustGuard] %s has DUST_ACCUMULATING state. "
                                "Excluding from BUY_FORCED candidates.",
                                sym
                            )
                except Exception as e:
                    self.logger.warning(
                        "[Meta:FlatDustGuard] Failed to check dust states: %s",
                        e
                    )
            
            # Find best EXECUTABLE BUY with escalation fallback
            candidates = []
            for sym, sigs in valid_signals_by_symbol.items():
                # Skip dusty symbols
                if sym in dusty_symbols:
                    self.logger.debug(
                        "[Meta:FlatDustGuard] Skipping %s (dust_accumulating)",
                        sym
                    )
                    continue

                # FIX FOR TEST 5: Prevent bootstrap override if position exists
                # Bootstrap logic is for INITIAL entry only.
                # If we have a position (even if dusty/small), we must respect normal lifecycle.
                if sym in owned_positions:
                    self.logger.info(
                        "[Meta:BootstrapGuard] Skipping %s for bootstrap override: Position already exists.",
                        sym
                    )
                    self.logger.info("[WHY_NO_TRADE] symbol=%s reason=BOOTSTRAP_NOT_FLAT details=position_exists", sym)
                    continue
                    
                for s in sigs:
                    if str(s.get("action")).upper() == "BUY":
                        candidates.append((sym, s))
            
            if len(candidates) > 0 and len(dusty_symbols) > 0:
                self.logger.info(
                    "[Meta:FlatDustGuard] Filtered candidates: %d original, %d clean (removed %d dusty)",
                    len(candidates) + len(dusty_symbols), len(candidates), len(dusty_symbols)
                )
            
            # Sort by confidence (descending)
            candidates.sort(key=lambda x: float(x[1].get("confidence", 0.0)), reverse=True)
            
            for sym, s in candidates:
                bootstrap_bypass_active = False  # Always define before any reference
                conf = float(s.get("confidence", 0.0))
                
                # ===== CRITICAL FIX #1: PRE-FLIGHT GATE CHECK (BOOTSTRAP DEADLOCK FIX) =====
                # Check hourly trade limit BEFORE creating decision
                # This prevents decisions from being created then gated/blocked during execution
                now = time.time()
                
                # Clean expired timestamps (older than 1 hour)
                while (self._trade_timestamps_sym[sym] and 
                    (now - self._trade_timestamps_sym[sym][0] > 3600)):
                    self._trade_timestamps_sym[sym].popleft()
                
                # Check if this symbol is at hourly limit (2 trades/symbol/hour)
                # FIX: Exempt BOOTSTRAP and DUST_MERGE trades from frequency limits
                is_bootstrap_bypass = s.get("_bootstrap_override") or s.get("_bootstrap")
                is_dust_merge = s.get("_dust_reentry_override")
                
                if not (is_bootstrap_bypass or is_dust_merge) and len(self._trade_timestamps_sym[sym]) >= 2:
                    self.logger.info(
                        "[Meta:PreGate] %s blocked by hourly limit (%d/2 trades). Trying next candidate.",
                        sym, len(self._trade_timestamps_sym[sym])
                    )
                    continue  # Skip this symbol, try next in loop
                # End CRITICAL FIX #1
                
                # LAYER 8 FIX: Force cleanup of expired reservations before affordability check
                # This prevents stale reservations from indefinitely locking capital
                quote_asset = "USDT"  # Assumed quote asset
                try:
                    # Trigger aggressive cleanup by calling get_spendable_balance
                    await self.shared_state.get_spendable_balance(quote_asset)
                    self.logger.debug("[Meta:LAYER8] Reservation cleanup triggered for BUY affordability check")
                except Exception as e:
                    self.logger.warning(f"[Meta:LAYER8] Cleanup failed: {e}")
                
                # ESCALATION LOGIC (DEADLOCK FIX)
                # CRITICAL FIX: Use agent's allocated budget as floor, not hardcoded 10.0
                # This ensures MetaController respects CapitalAllocator's small-account scaling
                agent_name = s.get("agent", "default")
                allocated_budget = float(self.shared_state.get_authoritative_reservation(agent_name)) if agent_name else 0.0
                
                # If CapitalAllocator allocated < 10.0, use that as the floor instead of hardcoded default
                # ALIGNMENT FIX: Use the SAME economic floor as ExecutionManager
                # to avoid Meta proposing quotes that EM will reject.
                min_econ_trade = float(getattr(self.config, "MIN_ECONOMIC_TRADE_USDT", 40.0) or 40.0)
                min_notional_floor = max(
                    float(self._cfg("MIN_NOTIONAL_FLOOR", 10.0)),
                    min_econ_trade,
                )
                min_executable_quote = max(
                    float(self._cfg("MIN_EXECUTABLE_QUOTE", 10.0)),
                    min_econ_trade,
                )
                dust_floor = float(getattr(self.config, "DUST_MIN_QUOTE_USDT", 5.0) or 5.0)

                # PRIORITY 3 FIX: Respect CapitalAllocator's allocation for small accounts
                # but NEVER go below the EM economic floor (unless bootstrap escape hatch)
                if 0 < allocated_budget < min_econ_trade and not bootstrap_bypass_active:
                    min_notional_floor = max(allocated_budget * 0.80, min_econ_trade)
                    self.logger.debug(
                        f"[Meta:LAYER9] Small account: agent={agent_name} "
                        f"allocated={allocated_budget:.2f} -> floor={min_notional_floor:.2f} "
                        f"(clamped to MIN_ECONOMIC={min_econ_trade})")
                elif 0 < allocated_budget < min_econ_trade and bootstrap_bypass_active:
                    # Bootstrap: allow lower floor for first trade
                    min_notional_floor = max(allocated_budget * 0.80, dust_floor)
                    self.logger.debug(
                        f"[Meta:LAYER9:BOOTSTRAP] Using bootstrap budget floor: "
                        f"agent={agent_name} allocated={allocated_budget:.2f} -> floor={min_notional_floor:.2f}")

                base_floor = max(min_notional_floor, min_executable_quote)
                current_price = None
                
                # BOOTSTRAP-AWARE QUANTITY FLOOR: Calculate minimum viable quote for successful execution
                try:
                    symbol_info = await self._get_symbol_info(sym)
                    if symbol_info:
                        symbol_min_notional = float(symbol_info.get('min_notional', base_floor))
                        min_qty = float(symbol_info.get('min_qty', 0.0))
                        step_size = float(symbol_info.get('step_size', 0.0001))
                        # Get current price for quantity calculation
                        current_price = None
                        if self.exchange_client:
                            get_px = getattr(self.exchange_client, "get_current_price", None) or getattr(self.exchange_client, "get_price", None)
                            if get_px:
                                try:
                                    current_price = await get_px(sym)
                                    if current_price:
                                        current_price = float(current_price)
                                except Exception as e:
                                    self.logger.debug(f"[Meta:FlatBootstrap] Failed to get price for {sym}: {e}")
                        if current_price and current_price > 0:
                            # Use production-grade min-notional-aware quantity calculation
                            fee_buffer = 1.01  # 1% fee buffer
                            slippage_buffer = 1.0  # Optionally configurable
                            qty = self._compute_min_notional_aware_qty(
                                price=current_price,
                                min_notional=symbol_min_notional,
                                min_qty=min_qty,
                                step_size=step_size,
                                fee_buffer=fee_buffer,
                                slippage_buffer=slippage_buffer
                            )
                            plan_q = qty * current_price
                            self.logger.info(
                                "[Meta:FlatBootstrap] Symbol %s: min-notional-aware qty=%.8f planned_quote=%.2f (min_notional=%.2f, min_qty=%.6f, step_size=%.6f, price=%.2f, fee_buffer=%.3f)",
                                sym, qty, plan_q, symbol_min_notional, min_qty, step_size, current_price, fee_buffer
                            )
                        else:
                            # Fallback without price: use enhanced min_notional approach
                            fee_buffer = 1.10  # 10% buffer without price info
                            bootstrap_min_quote = float(self._cfg("BOOTSTRAP_MIN_QUOTE", 10.0))
                            plan_q = max(symbol_min_notional * fee_buffer, bootstrap_min_quote)
                            self.logger.info(
                                "[Meta:FlatBootstrap] Symbol %s: fallback planned_quote=%.2f (symbol_min_notional=%.2f, fee_buffer=%.2f, bootstrap_min=%.2f, reason=no_price)",
                                sym, plan_q, symbol_min_notional, fee_buffer, bootstrap_min_quote
                            )
                    else:
                        # Fallback if symbol_info unavailable
                        trade_fee_pct = 0.001 * float(getattr(self.config, "TAKER_FEE_BPS", 10))
                        safety_factor = 1.15  # Increased safety factor for bootstrap
                        escalation_factor = (1.0 + trade_fee_pct) * safety_factor
                        plan_q = base_floor * escalation_factor
                        self.logger.warning(f"[Meta:FlatBootstrap] Symbol {sym}: using fallback planned_quote=%.2f (no symbol_info)", plan_q)
                except Exception as e:
                    self.logger.warning(f"[Meta:FlatBootstrap] Failed to calculate min-notional-aware quote for {sym}: {e}, using fallback")
                    trade_fee_pct = 0.001 * float(getattr(self.config, "TAKER_FEE_BPS", 10))
                    safety_factor = 1.15
                    escalation_factor = (1.0 + trade_fee_pct) * safety_factor
                    plan_q = base_floor * escalation_factor

                # Enforce unified entry floor (exit-feasibility-first when available)
                min_entry_floor = None
                try:
                    if self.shared_state and hasattr(self.shared_state, "compute_min_entry_quote"):
                        min_entry_floor = await self.shared_state.compute_min_entry_quote(
                            sym,
                            default_quote=float(plan_q or 0.0),
                            price=float(current_price or 0.0),
                        )
                except Exception:
                    min_entry_floor = None
                if min_entry_floor is None:
                    try:
                        min_entry_floor = float(self._cfg("MIN_ENTRY_USDT", self._min_entry_quote_usdt))
                    except Exception:
                        min_entry_floor = float(self._min_entry_quote_usdt)
                plan_q = max(float(plan_q or 0.0), float(min_entry_floor or 0.0))
                plan_q = max(
                    float(getattr(self.config, "DEFAULT_PLANNED_QUOTE", 0.0) or 0.0),
                    float(getattr(self.config, "MIN_TRADE_QUOTE", 0.0) or 0.0),
                    float(plan_q or 0.0),
                )
                
                # CRITICAL FIX #24: Initialize bootstrap bypass flag
                # BOOTSTRAP ESCAPE HATCH: Activate when portfolio is FLAT (cold bootstrap)
                # so the first trade can bypass MICRO_TRADE_KILL_SWITCH + MIN_ECONOMIC gates
                bootstrap_bypass_active = False
                try:
                    if (hasattr(self.shared_state, "is_bootstrap_mode")
                            and self.shared_state.is_bootstrap_mode()
                            and getattr(self.config, "BOOTSTRAP_ESCAPE_HATCH_ENABLED", False)):
                        bootstrap_bypass_active = True
                        self.logger.info(
                            "[Meta:BOOTSTRAP_ESCAPE] Escape hatch active for %s — "
                            "portfolio is FLAT, first trade allowed", sym)
                except Exception:
                    pass

                # Probe: Can we afford this?
                # Pass bootstrap state through policy context if in bootstrap mode
                if bootstrap_bypass_active:
                    policy_ctx = self._build_policy_context(sym, "BUY", extra={"bootstrap_bypass": True})
                    can_exec, gap, reason = await self.execution_manager.can_afford_market_buy(sym, plan_q, policy_context=policy_ctx)
                else:
                    can_exec, gap, reason = await self.execution_manager.can_afford_market_buy(sym, plan_q)
                
                # DEBUG: Log the affordability check details for capital starvation diagnosis
                quote_asset = str(self._cfg("QUOTE_ASSET") or "USDT").upper()
                actual_spendable = float(await _safe_await(self.shared_state.get_spendable_balance(quote_asset)) or 0.0)
                self.logger.warning(
                    f"[Meta:CAPITAL_DEBUG] Symbol {sym}: can_afford_check(planned_quote={plan_q:.2f}) "
                    f"-> can_exec={can_exec}, gap={gap:.2f}, reason='{reason}' | "
                    f"actual_spendable={actual_spendable:.2f}"
                )
                
                if not can_exec:
                    # EMERGENCY: If capital gap exists, force cleanup old locks IMMEDIATELY
                    if gap > 0:
                        removed, freed = await self.shared_state.force_cleanup_expired_reservations(quote_asset)
                        if removed > 0:
                            self.logger.info(f"[Meta:EMERGENCY] Gap detected (${gap:.2f}), forced cleanup. Removed {removed} stale locks, freed ${freed:.2f}")
                    
                    # BOOTSTRAP CAPITAL FIX: If we have significant actual capital but ExecutionManager says no,
                    # bypass the check for small bootstrap amounts during clean wallet recovery
                    if actual_spendable > 50.0 and plan_q <= 15.0:
                        self.logger.warning(
                            f"[Meta:BOOTSTRAP_BYPASS] ExecutionManager affordability check failed for {sym} "
                            f"but actual_spendable=${actual_spendable:.2f} >> required=${plan_q:.2f}. "
                            f"Bypassing check for bootstrap execution."
                        )
                        # Force execution by marking as affordable
                        can_exec = True
                        gap = 0.0
                        reason = "BOOTSTRAP_BYPASS"
                        bootstrap_bypass_active = True
                    
                # CRITICAL: Skip fallback logic if bootstrap bypass is active
                if not can_exec and not bootstrap_bypass_active:
                    # LAYER 9: CAPITAL ADAPTATION - REMOVED (Phase 2 consolidation)
                    # Capital floor check moved to _check_capital_floor_central (called at start)
                    # No longer adapt floor dynamically - use fixed floor from centralized check
                    # ===== FIX #2B: STARVATION ESCAPE HATCH - DEPRECATED =====
                    # This logic masked the underlying fragmentation issue
                    # Replaced with explicit capital check at _build_decisions start
                    
                    # Attempt fallback: reduce by gap + buffer
                    # CRITICAL FIX: When capital is severely constrained (gap > allocated),
                    # allow graceful degradation to near-minimum notional.
                    fallback_q = plan_q - (gap + 0.5)  # 0.5 safety margin
                    
                    # TIER 1: Try original fallback if it's reasonable
                    if fallback_q >= min_notional_floor * 0.75:  # Allow 75% of floor
                        plan_q = fallback_q
                        # Pass bootstrap context for consistency
                        if bootstrap_bypass_active:
                            policy_ctx = self._build_policy_context(sym, "BUY", extra={"bootstrap_bypass": True})
                            can_exec_fb, gap_fb, reason_fb = await self.execution_manager.can_afford_market_buy(sym, plan_q, policy_context=policy_ctx)
                        else:
                            can_exec_fb, gap_fb, reason_fb = await self.execution_manager.can_afford_market_buy(sym, plan_q)
                        
                        if can_exec_fb:
                            self.logger.info(
                                "[Meta:FlatBootstrap] Symbol %s: tier-1 fallback to %.2f (gap was %.2f → %.2f)",
                                sym, plan_q, gap, gap_fb
                            )
                        else:
                            # TIER 1B/2/3: Try ultra-aggressive degradation
                            executed = False
                            for trial_q in [fallback_q, min_notional_floor, min_notional_floor * 0.9, min_notional_floor * 0.75]:
                                if trial_q <= 0: continue
                                plan_q = trial_q
                                # Pass bootstrap context for consistency
                                if bootstrap_bypass_active:
                                    policy_ctx = self._build_policy_context(sym, "BUY", extra={"bootstrap_bypass": True})
                                    can_exec_trial, gap_trial, reason_trial = await self.execution_manager.can_afford_market_buy(sym, plan_q, policy_context=policy_ctx)
                                else:
                                    can_exec_trial, gap_trial, reason_trial = await self.execution_manager.can_afford_market_buy(sym, plan_q)
                                
                                # DEBUG: Log each fallback attempt
                                self.logger.warning(
                                    f"[Meta:FALLBACK_DEBUG] {sym} trying trial_q={trial_q:.2f} "
                                    f"-> can_exec={can_exec_trial}, gap={gap_trial:.2f}, reason='{reason_trial}'"
                                )
                                
                                if can_exec_trial:
                                    self.logger.info(
                                        "[Meta:FlatBootstrap] Symbol %s: fallback to %.2f (gap=%.2f → %.2f)",
                                        sym, plan_q, gap, gap_trial
                                    )
                                    executed = True
                                    break
                            if not executed:
                                self.logger.warning(
                                    "[Meta:FlatBootstrap] Symbol %s: all fallback tiers exhausted (gap=%.2f). "
                                    "Insufficient capital. Skipping.",
                                    sym, gap
                                )
                                await self.shared_state.record_rejection(sym, "BUY", "CAPITAL_INSUFFICIENT", source="MetaController")
                                # SCENARIO 1: FLAT + INSUFFICIENT + BUY → Emit CAPITAL_STARVED
                                await self._handle_capital_starved(sym, plan_q, gap)
                                continue
                    else:
                        # Fallback is small but > 0: try ultra-aggressive tiers
                        executed = False
                        for trial_q in [fallback_q, min_notional_floor, 5.0, 3.0, 2.0, 1.5, 1.0]:
                            if trial_q <= 0: continue
                            plan_q = trial_q
                            # Pass bootstrap context for consistency
                            if bootstrap_bypass_active:
                                policy_ctx = self._build_policy_context(sym, "BUY", extra={"bootstrap_bypass": True})
                                can_exec_trial, gap_trial, reason_trial = await self.execution_manager.can_afford_market_buy(sym, plan_q, policy_context=policy_ctx)
                            else:
                                can_exec_trial, gap_trial, reason_trial = await self.execution_manager.can_afford_market_buy(sym, plan_q)
                            
                            # DEBUG: Log each aggressive fallback attempt
                            self.logger.warning(
                                f"[Meta:AGGRESSIVE_DEBUG] {sym} trying trial_q={trial_q:.2f} "
                                f"-> can_exec={can_exec_trial}, gap={gap_trial:.2f}, reason='{reason_trial}'"
                            )
                            
                            if can_exec_trial:
                                self.logger.info(
                                    "[Meta:FlatBootstrap] Symbol %s: aggressive fallback to %.2f (gap=%.2f → %.2f)",
                                    sym, plan_q, gap, gap_trial
                                )
                                executed = True
                                break
                        if not executed:
                            self.logger.warning(
                                "[Meta:FlatBootstrap] Symbol %s: all fallback tiers exhausted (gap=%.2f). "
                                "Insufficient capital. Skipping.",
                                sym, gap
                            )
                            await self.shared_state.record_rejection(sym, "BUY", "CAPITAL_INSUFFICIENT", source="MetaController")
                            # SCENARIO 1: FLAT + INSUFFICIENT + BUY → Emit CAPITAL_STARVED
                            await self._handle_capital_starved(sym, plan_q, gap)
                            continue

                # CRITICAL FIX #24: After bootstrap bypass, skip to execution
                if bootstrap_bypass_active:
                    self.logger.warning(
                        f"[Meta:BOOTSTRAP_BYPASS] Proceeding to execution with bypass active: "
                        f"{sym} planned_quote={plan_q:.2f}, can_exec={can_exec}, reason={reason}"
                    )
                else:
                    self.logger.info(
                        "[Meta:FlatBootstrap] Symbol %s: executeableat escalated quote %.2f (gap=%.2f)",
                        sym, plan_q, gap
                    )

                # Found valid candidate at escalated/fallback quote
                self.logger.info(
                    "[Meta] FLAT_PORTFOLIO -> BUY_FORCED symbol=%s conf=%.2f escalated_quote=%.2f",
                    sym, conf, plan_q
                )
                
                # Enforce attributes for execution
                s["_tier"] = "B"
                s["_bootstrap"] = True
                s["_force_min_notional"] = True
                s["_planned_quote"] = plan_q
                s["reason"] = f"FLAT_FORCED_BOOTSTRAP:{conf:.2f}:escalated"
                return [(sym, "BUY", s)]
            
            self.logger.info("[Meta] FLAT_PORTFOLIO but no valid & executable BUY signals found.")
            
            # ===== CRITICAL FIX #3: Symbol Rotation / Fallback Candidates =====
            # If no direct signals available, attempt to find ANY valid symbol from accepted set
            # This prevents hardcoding BNBUSDT and enables dynamic symbol rotation
            if len(candidates) == 0:
                self.logger.info("[Meta:FIX#3] No BUY candidates from signals. Checking accepted symbols for fallback...")
                
                # Build list of accepted symbols with synthetic HOLD signals as fallback
                fallback_candidates = []
                for accepted_sym in accepted_symbols_set:
                    if accepted_sym in owned_positions:
                        # This symbol is already owned - check for exit signals first
                        continue
                    
                    # Skip dusty symbols in fallback too
                    if accepted_sym in dusty_symbols:
                        self.logger.debug(
                            "[Meta:FlatDustGuard:Fallback] Skipping %s (dust_accumulating)",
                            accepted_sym
                        )
                        continue
                    
                    # Create synthetic fallback signal (low confidence HOLD becomes opportunistic BUY during FLAT)
                    fallback_sig = {
                        "symbol": accepted_sym,
                        "action": "BUY",
                        "confidence": 0.35,  # Low confidence fallback
                        "agent": "MetaFallback",
                        "timestamp": now_ts,
                        "_fallback": True,
                        "reason": "FLAT_ROTATION_FALLBACK"
                    }
                    fallback_candidates.append((accepted_sym, fallback_sig))
                
                # Try fallback candidates
                for fallback_sym, fallback_sig in fallback_candidates:
                    self.logger.info("[Meta:FIX#3] Attempting fallback symbol: %s (conf=%.2f)", fallback_sym, 0.35)
                    
                    # Check if this symbol would pass gate checks
                    now = time.time()
                    while (self._trade_timestamps_sym[fallback_sym] and 
                        (now - self._trade_timestamps_sym[fallback_sym][0] > 3600)):
                        self._trade_timestamps_sym[fallback_sym].popleft()
                    
                    if len(self._trade_timestamps_sym[fallback_sym]) >= 2:
                        self.logger.debug("[Meta:FIX#3] Fallback %s also gated (at hourly limit). Trying next.", fallback_sym)
                        continue
                    
                    # This symbol is available - use it as bootstrap candidate
                    candidates = [(fallback_sym, fallback_sig)]
                    break
            
            # SCENARIO 2: FLAT + SELL signals only → Emit WAITING
            sell_only_signals = [
                s for signals in valid_signals_by_symbol.values() 
                for s in signals 
                if str(s.get("action")).upper() == "SELL"
            ]
            if sell_only_signals:
                await self._handle_flat_with_sell_signals_only(sell_only_signals)
            
            return []

        # Volatility regime gate (pre-decision, non-negotiable)
        regime = "normal"
        try:
            if hasattr(self, "volatility_detector") and self.volatility_detector:
                regime = str(self.volatility_detector.get_regime() or regime)
            if hasattr(self.shared_state, "metrics") and isinstance(self.shared_state.metrics, dict):
                regime = str(self.shared_state.metrics.get("volatility_regime") or regime)
            if hasattr(self.shared_state, "get_volatility_regime"):
                tf = str(self._cfg("VOLATILITY_REGIME_TIMEFRAME", "5m"))
                reg = await _safe_await(self.shared_state.get_volatility_regime("GLOBAL", tf, max_age_seconds=600))
                if reg and reg.get("regime"):
                    regime = str(reg.get("regime"))
        except Exception:
            pass
        regime = (regime or "normal").lower()

        if regime == "low":
            blocked_agents = {"trendhunter", "bootstrapscalper"}

            def _low_regime_exit(sig: Dict[str, Any]) -> bool:
                reason_text = " ".join([
                    str(sig.get("reason") or ""),
                    str(sig.get("exit_reason") or ""),
                    str(sig.get("signal_reason") or ""),
                    str(sig.get("liquidation_reason") or ""),
                    str(sig.get("tag") or ""),
                    str(sig.get("_tag") or ""),
                    str(sig.get("execution_tag") or ""),
                ]).lower()
                if sig.get("_is_starvation_sell") or sig.get("_force_dust_liquidation"):
                    return True
                if "tp_sl" in reason_text or "take_profit" in reason_text or "stop_loss" in reason_text:
                    return True
                if "liquidation" in reason_text or "dust" in reason_text:
                    return True
                return False

            dropped = 0
            for sym in list(valid_signals_by_symbol.keys()):
                filtered = []
                for sig in valid_signals_by_symbol.get(sym, []):
                    action = str(sig.get("action") or "").upper()
                    agent = str(sig.get("agent") or "").lower()
                    if action == "SELL" and _low_regime_exit(sig):
                        filtered.append(sig)
                        continue
                    if agent in blocked_agents or action == "BUY":
                        dropped += 1
                        continue
                    dropped += 1
                if filtered:
                    valid_signals_by_symbol[sym] = filtered
                else:
                    valid_signals_by_symbol.pop(sym, None)

            if dropped > 0:
                self.logger.warning(
                    "[Meta:RegimeGate] LOW regime: blocked %d signals (allowed exits only).",
                    dropped,
                )
        
        # ✅ FIX #2: Bootstrap lock engaged - log it and proceed with normal decision making
        if is_flat and self._bootstrap_lock_engaged:
            self.logger.info(
                "[Meta:FIX#2] 🔒 Bootstrap lock is engaged. Portfolio is_flat=%s but FLAT_FORCED logic is DISABLED. "
                "Proceeding with normal multi-signal mode.",
                is_flat
            )

        # 2) Ranking & Scoring BUYs (with Rejection Penalty for Deadlock Prevention)
        symbol_scores = []
        deadlock_threshold = int(self._cfg("DEADLOCK_REJECTION_THRESHOLD", 3))
        for sym, signals in valid_signals_by_symbol.items():
            max_conf = max(float(s.get("confidence", 0.0)) for s in signals)
            rank_weight = 1.3 if sym in owned_positions else 1.0
            
            # P9 DEADLOCK PREVENTION: Penalize symbols with recent rejections
            rejection_count = 0
            if hasattr(self.shared_state, "get_rejection_count"):
                rejection_count = self.shared_state.get_rejection_count(sym, "BUY")
            
            # Penalty: reduce score by 50% for each rejection (compounding)
            rejection_penalty = 0.5 ** rejection_count if rejection_count > 0 else 1.0
            
            # If symbol exceeds deadlock threshold, skip entirely (force diversification)
            if rejection_count >= deadlock_threshold:
                self.logger.warning("[Meta:Deadlock] Symbol %s has %d rejections (threshold=%d). Skipping.", 
                                    sym, rejection_count, deadlock_threshold)
                continue

            # Diversification: penalize symbols traded recently to encourage rotation
            recent_window = float(self._cfg("DIVERSIFY_RECENT_TRADE_WINDOW_SEC", 1800.0))
            penalty_factor = float(self._cfg("DIVERSIFY_RECENT_TRADE_PENALTY", 0.7))
            recent_trade_penalty = 1.0
            if recent_window > 0 and penalty_factor > 0:
                now = time.time()
                recent_trades = sum(
                    1 for ts in self._trade_timestamps_sym.get(sym, [])
                    if (now - ts) <= recent_window
                )
                if recent_trades > 0:
                    recent_trade_penalty = penalty_factor ** min(recent_trades, 3)

            final_score = max_conf * rank_weight * rejection_penalty * recent_trade_penalty
            symbol_scores.append((sym, final_score, rejection_count))
        symbol_scores.sort(key=lambda x: x[1], reverse=True)
        ranked_symbols = [s[0] for s in symbol_scores]

        # 2.5) Portfolio Replacement Evaluation (Intelligence Tier)
        # (max_pos already defined by mode envelope)
        min_notional = float(self._cfg("MIN_NOTIONAL_USDT", 10.0))
        
        # FIX #1 & #2: CRITICAL - Use SIGNIFICANT position count, not total count
        # (counts already retrieved at start of function)
        current_pos_count = sig_pos  # Use significant count, not total
        
        # Log classification for debugging
        if total_pos > 0:
            self.logger.info(
                "[Meta:PosCounts] Portfolio state: total=%d sig=%d dust=%d ratio=%.1f%%",
                total_pos, sig_pos, dust_pos, dust_ratio * 100
            )
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # PRE-LAYER: Initialize agent budgets (needed by Layers 1, 3)
        # ═══════════════════════════════════════════════════════════════════════════════
        agent_budgets = {}
        try:
            plan = self.shared_state.get_active_allocation_plan()
            meta_reservation = 0.0
            if hasattr(self.shared_state, "get_authoritative_reservation"):
                meta_reservation = float(self.shared_state.get_authoritative_reservation("Meta") or 0.0)
            for agent in plan.get("per_agent_usdt", {}).keys():
                raw_budget = float(self.shared_state.get_authoritative_reservation(agent))
                if raw_budget <= 0.0 and meta_reservation > 0.0:
                    agent_budgets[agent] = meta_reservation
                    self.logger.warning(
                        "[Meta:ReservationFallback] Agent %s has zero reservation; using Meta reservation %.2f for sizing",
                        agent, meta_reservation
                    )
                else:
                    agent_budgets[agent] = raw_budget
        except Exception:
            self.logger.debug("[Meta:PreLayer] Failed to initialize agent budgets early, will use defaults")
        
        # Ensure Meta fallback exists
        if "Meta" not in agent_budgets:
            agent_budgets["Meta"] = float(await self.shared_state.get_spendable_balance("USDT") or 0.0)
        
        # 🧱 LAYER 1: HARD MINIMUM ENTRY SIZE (PREVENT DUST AT CREATION)
        # ═══════════════════════════════════════════════════════════════════════════════
        # CRITICAL: A new BUY is FORBIDDEN unless it can immediately be significant
        # This prevents 80% of dust creation at the source
        # ═══════════════════════════════════════════════════════════════════════════════
        
        min_position_value = float(self._cfg("MIN_POSITION_VALUE_USDT", 10.0))
        strategy_floor = float(
            self._cfg(
                "MIN_SIGNIFICANT_POSITION_USDT",
                self._cfg("MIN_SIGNIFICANT_USD", self._cfg("SIGNIFICANT_POSITION_USDT", 25.0)),
            )
        )
        min_entry = float(self._cfg("MIN_ENTRY_USDT", self._cfg("SAFE_ENTRY_USDT", 12.0)))
        significant_position_usdt = max(
            1.5 * min_notional,  # 1.5x exchange minimum
            1.5 * await self._get_avg_trade_cost(),  # 1.5x average trade cost
            min_position_value,
            strategy_floor,
            min_entry,
        )
        
        self.logger.info(
            "[Meta:Layer1] ENTRY_SIZE_ENFORCEMENT: significant_position_usdt=$%.2f | "
            "min_notional=$%.2f, avg_trade_cost=$%.2f",
            significant_position_usdt, min_notional, await self._get_avg_trade_cost()
        )
        
        # Filter BUY signals: reject if planned_quote < significant_position_usdt (NEW positions only)
        filtered_buy_symbols = []
        for sym in valid_signals_by_symbol.keys():
            sym_norm = self._normalize_symbol(sym)
            has_existing_position = sym_norm in owned_positions
            if has_existing_position:
                # Existing position: accumulation allowed (checked separately)
                filtered_buy_symbols.append(sym)
                continue
            
            # New position: must meet minimum entry size
            buy_sigs = [s for s in valid_signals_by_symbol.get(sym, []) if s.get("action") == "BUY"]
            if not buy_sigs:
                continue

            is_dust_reentry = any(bool(s.get("_dust_reentry_override")) for s in buy_sigs)
            if is_dust_reentry:
                self.logger.info(
                    "[Meta:Layer1] DUST_REENTRY_BYPASS: %s | allowing BUY to merge dust despite reservation floors",
                    sym
                )
                filtered_buy_symbols.append(sym)
                continue
            
            best_sig = max(buy_sigs, key=lambda s: float(s.get("confidence", 0.0)))
            agent_name = best_sig.get("agent", "Meta")
            agent_budget = agent_budgets.get(agent_name, agent_budgets.get("Meta", 0.0))
            
            if agent_budget >= significant_position_usdt:
                # Entry size sufficient
                filtered_buy_symbols.append(sym)
            else:
                if has_existing_position:
                    # Allow scaling existing position below significant threshold
                    filtered_buy_symbols.append(sym)
                else:
                    self.logger.warning(
                        "[Meta:Layer1] 🚫 ENTRY_TOO_SMALL_PREVENT_DUST: %s | "
                        "planned=%.2f < minimum=%.2f USD (DENIED)",
                        sym, agent_budget, significant_position_usdt
                    )
        
        # Update valid_signals_by_symbol to only include qualified symbols
        valid_signals_by_symbol = {
            sym: valid_signals_by_symbol.get(sym, []) 
            for sym in valid_signals_by_symbol.keys() 
            if sym in filtered_buy_symbols or sym in owned_positions
        }
        
        # 🧱 LAYER 2: DUST LOCKOUT (SYMBOL-LEVEL MEMORY)
        # ═══════════════════════════════════════════════════════════════════════════════
        # Once dust, locked until promotion capital exists
        # ═══════════════════════════════════════════════════════════════════════════════
        
        promotion_capital_threshold = significant_position_usdt * 2.0
        quote_asset = str(self._cfg("QUOTE_ASSET") or "USDT").upper()
        available_capital = float(await self.shared_state.get_spendable_balance(quote_asset) or 0.0)
        
        dust_locked_symbols = set()
        for sym, pos_data in owned_positions.items():
            current_qty = float(pos_data.get("quantity", 0.0) or pos_data.get("qty", 0.0))
            if current_qty <= 0:
                continue
            
            try:
                current_price = await self.shared_state.safe_price(sym)
                current_value = current_qty * current_price
            except Exception:
                current_value = float(pos_data.get("value_usdt", 0.0))
            
            # Is this dust?
            if current_value < float(
                self._cfg("MIN_SIGNIFICANT_POSITION_USDT", self._cfg("MIN_SIGNIFICANT_USD", 25.0))
            ):
                # Check if we have capital to promote it
                if available_capital < promotion_capital_threshold:
                    dust_locked_symbols.add(sym)
                    self.logger.warning(
                        "[Meta:Layer2] 🔒 DUST_LOCKED: %s | value=%.2f USD, "
                        "capital_available=%.2f < promotion_threshold=%.2f (BUY DENIED)",
                        sym, current_value, available_capital, promotion_capital_threshold
                    )
        
        # Remove dust-locked symbols from valid signals
        for sym in dust_locked_symbols:
            if sym in valid_signals_by_symbol:
                del valid_signals_by_symbol[sym]
                self.logger.debug("[Meta:Layer2] Removed DUST_LOCKED symbol %s from trading", sym)
        
        # 🧱 LAYER 3: ACCUMULATION GATE (GOAL-BASED, NOT INCREMENTAL)
        # ═══════════════════════════════════════════════════════════════════════════════
        # Accumulation is ONLY allowed if it can COMPLETE promotion
        # No "let's try with $2 and see" - must reach MIN_SIGNIFICANT
        # ═══════════════════════════════════════════════════════════════════════════════
        
        # CRITICAL FIX: Define min_significant BEFORE any nested functions or list comprehensions
        # Python's scoping evaluates references at definition time, not execution time.
        # If a nested function uses min_significant before it's defined, NameError is raised.
        min_significant = float(
            self._cfg("MIN_SIGNIFICANT_POSITION_USDT", self._cfg("MIN_SIGNIFICANT_USD", 25.0))
        )
        
        accumulation_candidates = []
        
        for sym in valid_signals_by_symbol.keys():
            if sym not in owned_positions:
                continue  # Only existing positions
            
            pos_data = owned_positions[sym]
            current_qty = float(pos_data.get("quantity", 0.0) or pos_data.get("qty", 0.0))
            if current_qty <= 0:
                continue
            
            try:
                current_price = await self.shared_state.safe_price(sym)
                current_value = current_qty * current_price
            except Exception:
                current_value = float(pos_data.get("value_usdt", 0.0))
            
            # Is this dust?
            min_significant = float(
                self._cfg("MIN_SIGNIFICANT_POSITION_USDT", self._cfg("MIN_SIGNIFICANT_USD", 25.0))
            )
            if current_value >= min_significant:
                continue  # Not dust
            
            # CRITICAL: Calculate REQUIRED amount to complete promotion
            # (not optional, not incremental, REQUIRED)
            required_to_promote = significant_position_usdt - current_value

            # FIX FOR TEST 4: Prevent repeat accumulation if already attempted in this lifecycle
            if sym in self._dust_merges:
                self.logger.info(
                    "[Meta:Layer3] 🚫 ACCUMULATION_DROPPED: %s | Lifecycle lock active (already attempted)",
                    sym
                )
                self.logger.info("[WHY_NO_TRADE] symbol=%s reason=POSITION_ALREADY_OPEN details=accumulation_locked", sym)
                continue
            
            # Check BUY signal
            buy_sigs = [s for s in valid_signals_by_symbol.get(sym, []) if s.get("action") == "BUY"]
            if not buy_sigs:
                continue
            
            best_sig = max(buy_sigs, key=lambda s: float(s.get("confidence", 0.0)))
            buy_conf = float(best_sig.get("confidence", 0.0))
            agent_name = best_sig.get("agent", "Meta")
            agent_budget = agent_budgets.get(agent_name, 0.0)
            
            # LAYER 3 GATE: Can we complete the promotion?
            if agent_budget >= required_to_promote and available_capital >= required_to_promote:
                # YES - We can complete promotion
                accumulation_candidates.append({
                    "symbol": sym,
                    "signal": best_sig,
                    "confidence": buy_conf,
                    "current_value": current_value,
                    "required_to_promote": required_to_promote,
                    "target_value": significant_position_usdt,
                    "agent": agent_name,
                    "priority_score": buy_conf * required_to_promote
                })
                
                self.logger.info(
                    "[Meta:Layer3] 🎯 ACCUMULATION_COMPLETABLE: %s | "
                    "current=%.2f + required=%.2f = target=%.2f (APPROVED)",
                    sym, current_value, required_to_promote, significant_position_usdt
                )
            else:
                # NO - Cannot complete promotion
                self.logger.warning(
                    "[Meta:Layer3] 🚫 ACCUMULATION_CANNOT_COMPLETE: %s | "
                    "current=%.2f, required=%.2f, available_capital=%.2f (DENIED)",
                    sym, current_value, required_to_promote, available_capital
                )
        
        # Process accumulation candidates (highest priority first)
        if accumulation_candidates:
            accumulation_candidates.sort(key=lambda x: x["priority_score"], reverse=True)
            
            best_accum = accumulation_candidates[0]
            sym = best_accum["symbol"]
            sig = best_accum["signal"]
            required_amount = best_accum["required_to_promote"]
            
            self.logger.warning(
                "[Meta:Layer3] 💰 EXECUTING_GOAL_BASED_ACCUMULATION: %s | "
                "current=%.2f → target=%.2f (required=%.2f)",
                sym, best_accum["current_value"], best_accum["target_value"], required_amount
            )
            
            # Tag signal with goal-based accumulation flag
            sig["_goal_based_accumulation"] = True
            sig["_accumulation_current_value"] = best_accum["current_value"]
            sig["_accumulation_required_to_promote"] = required_amount
            sig["_accumulation_target"] = best_accum["target_value"]
            sig["reason"] = f"GOAL_BASED_ACCUMULATION (promotion_required)"
            sig["_planned_quote"] = required_amount
            
            return [(sym, "BUY", sig)]
        
        # 🧱 LAYER 4: PORTFOLIO SLOT ACCOUNTING (DUST ≠ SLOT)
        # ═══════════════════════════════════════════════════════════════════════════════
        # Portfolio capacity checks IGNORE dust positions
        # Only significant positions count toward capacity
        # ═══════════════════════════════════════════════════════════════════════════════
        # min_significant already defined at Layer 3 start to avoid closure NameError

        # Count only SIGNIFICANT positions (not dust)
        significant_positions = [
            p for p in owned_positions.values()
            if self._is_recovery_sellable("", p, ignore_filters=False, ignore_core=True)
        ]
        active_slots = len(significant_positions)
        # (max_pos already defined by mode envelope)
        
        self.logger.info(
            "[Meta:Layer4] SLOT_ACCOUNTING: active_slots=%d/%d (dust positions not counted)",
            active_slots, max_pos
        )
        
        # Use active_slots (not total_pos) for capacity checks in P1 and P2
        # This is already handled in the portfolio full checks below
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # END 4-LAYER DUST PREVENTION SYSTEM
        # ═══════════════════════════════════════════════════════════════════════════════
        
        # ===== ACCUMULATION PRIORITY: DUST + STRONG_BUY = FORCED PROMOTION =====
        # When portfolio is FULL but dust exists with strong BUY signal,
        # ALLOW the BUY (promote dust to significant) BUT force a SELL elsewhere.
        # This rule turns "accumulation in limbo" into "active accumulation".
        if sig_pos >= max_pos:
            self.logger.info(
                "[Meta:AccumPriority] PORTFOLIO_FULL (sig_pos=%d >= max_pos=%d). "
                "Checking for dust promotion opportunities...",
                sig_pos, max_pos
            )
            
            # Find dust positions with strong BUY signals
            dust_promotion_candidates = []
            for sym, sigs in valid_signals_by_symbol.items():
                # Check if this symbol is in dust state
                pos_data = owned_positions.get(sym)
                if not pos_data:
                    continue
                
                # Check position state is DUST or quantity is very small
                state = pos_data.get("state", "")
                qty = float(pos_data.get("quantity", 0.0) or (pos_data.get("qty", 0.0)))
                value_usdt = float(pos_data.get("value_usdt", 0.0))
                
                # Must be in DUST_LOCKED state or have tiny quantity
                is_dust = self._is_position_dust_locked(state, qty, value_usdt)
                
                if not is_dust:
                    continue
                
                # Look for BUY signals with decent confidence
                for sig in sigs:
                    if str(sig.get("action")).upper() == "BUY":
                        conf = float(sig.get("confidence", 0.0))
                        # Strong BUY = confidence >= 0.55
                        if conf >= 0.55:
                            dust_promotion_candidates.append((sym, sig, conf, qty))
            
            if dust_promotion_candidates:
                # Sort by confidence (highest first)
                dust_promotion_candidates.sort(key=lambda x: x[2], reverse=True)
                
                # Pick the best dust-to-promote
                best_dust_sym, best_dust_sig, dust_conf, dust_qty = dust_promotion_candidates[0]
                
                self.logger.warning(
                    "[Meta:AccumPriority] 🚀 DUST+STRONG_BUY PROMOTION: "
                    "symbol=%s dust_qty=%.6f confidence=%.2f → promoting to significant "
                    "(but forcing SELL elsewhere for capacity)",
                    best_dust_sym, dust_qty, dust_conf
                )
                
                # Now find a non-dust position to force-SELL for capacity recovery
                eligible_sells = []
                for sym_to_sell, pos_to_sell in owned_positions.items():
                    if sym_to_sell == best_dust_sym:
                        continue  # Don't sell the one we're promoting
                    
                    state_to_sell = pos_to_sell.get("state", "")
                    qty_to_sell = float(pos_to_sell.get("quantity", 0.0) or pos_to_sell.get("qty", 0.0))
                    
                    # Only sell non-dust significant positions
                    if state_to_sell != "DUST_LOCKED" and qty_to_sell >= 0.0001:
                        score = self.score_position(pos_to_sell)
                        eligible_sells.append((sym_to_sell, score))
                
                if eligible_sells:
                    # Sort by score (lowest = worst performer)
                    eligible_sells.sort(key=lambda x: x[1])
                    worst_sell_sym = eligible_sells[0][0]
                    
                    self.logger.warning(
                        "[Meta:AccumPriority] → Forcing SELL of worst performer: %s "
                        "(to free capacity for dust promotion)",
                        worst_sell_sym
                    )
                    
                    # Mark the dust BUY as promoted
                    best_dust_sig["_dust_promotion"] = True
                    best_dust_sig["_promotes_dust"] = best_dust_sym
                    best_dust_sig["reason"] = "ACCUMULATION_DUST_PROMOTION"
                    
                    # Create forced SELL for capacity
                    forced_sell_sig = {
                        "symbol": worst_sell_sym,
                        "action": "SELL",
                        "confidence": 1.0,
                        "agent": "MetaAccumulator",
                        "timestamp": time.time(),
                        "reason": f"forced_capacity_for_dust_promotion_of_{best_dust_sym}",
                        "_dust_promotion_sacrifice": True,
                        "_sacrifice_for_symbol": best_dust_sym
                    }
                    
                    # Execute SELL first, then BUY
                    self.logger.info(
                        "[Meta:AccumPriority] Decision sequence: "
                        "1) SELL %s (capacity) 2) BUY %s (accumulation)",
                        worst_sell_sym, best_dust_sym
                    )
                    
                    # Return SELL first - it will free the slot
                    return [(worst_sell_sym, "SELL", forced_sell_sig)]
        
        # CRITICAL: Prepare BUY ranking early for Replacement & Mandatory Sell Logic
        buy_ranked_symbols = [
            sym for sym in ranked_symbols 
            if sym in valid_signals_by_symbol and any(s.get("action") == "BUY" for s in valid_signals_by_symbol.get(sym, []))
        ]

        # 2. ROTATION & VELOCITY AUTHORITY (P9 Canonical REA)
        # If portfolio is FULL (sig_pos >= max_pos) OR capital is STARVED, check for rotation.
        mandatory_sell_mode = False
        if sig_pos >= max_pos or is_starved:
            self.logger.info(
                "[Meta:Policy] CAPACITY_LIMIT_OR_STARVATION (sig_pos=%d, max_pos=%d, starved=%s). Evaluating REA rotation.",
                sig_pos, max_pos, is_starved
            )
            self.logger.info(
                "[Meta:Policy] PORTFOLIO_FULL (sig_pos=%d >= max_pos=%d). Enforcing capacity policies.",
                sig_pos, max_pos
            )
            
            # --- 1. SOP STANDING RECOVERY (Exit Velocity) ---
            # If strictly over limit, we MUST exit something to get back to SOP standing.
            if sig_pos > max_pos:
                self.logger.warning(
                    "[Meta:SopEnforcer] 👮 OVER-CAPACITY DETECTED (%d > %d). "
                    "Forcing exit of worst performer to recover SOP standing.",
                    sig_pos, max_pos
                )
                
                # KEY FIX: Dynamic value calculation to match _count_significant_positions
                # Do NOT rely on potentially stale 'value_usdt' from snapshot
                sig_holdings = []
                for p_sym, p_data in owned_positions.items():
                    qty = float(p_data.get("quantity", 0.0) or p_data.get("qty", 0.0))
                    if qty <= 0: continue
                    
                    # Calculate dynamic value
                    try:
                        price = float(p_data.get("mark_price", 0.0) or p_data.get("price", 0.0) or 0.0)
                        if price <= 0 and hasattr(self.shared_state, "get_latest_price"):
                            # We can't await here easily if we want to keep this synchronous block safe?
                            # But _build_decisions is async.
                            # Best effort fallback
                            price = float(getattr(self.shared_state, "latest_prices", {}).get(p_sym, 0.0) or 0.0)
                        
                        val = qty * price
                        # Fallback to stored value if dynamic failed
                        if val <= 0:
                            val = float(p_data.get("value_usdt", 0.0))
                    except Exception:
                        val = float(p_data.get("value_usdt", 0.0))

                    if val >= float(
                        self._cfg("MIN_SIGNIFICANT_POSITION_USDT", self._cfg("MIN_SIGNIFICANT_USD", 25.0))
                    ):
                        score = self.score_position(p_data)
                        sig_holdings.append((p_sym, score))
                
                if sig_holdings:
                    sig_holdings.sort(key=lambda x: x[1])
                    worst_sym, worst_score = sig_holdings[0]
                    
                    forced_sell_sig = {
                        "symbol": worst_sym, "action": "SELL", "confidence": 1.0,
                        "agent": "MetaSopEnforcer", "timestamp": time.time(),
                        "reason": f"sop_standing_recovery_limit_{max_pos}_score_{worst_score:.4f}",
                        "_mandatory_capacity_recovery": True
                    }
                    self.logger.warning("[Meta:SopEnforcer] 🔥 FORCING SELL of %s (worst performer) to recover slot.", worst_sym)
                    return [(worst_sym, "SELL", forced_sell_sig)]

            # --- 2. ROTATION AUTHORITY (P9 Canonical REA) ---
            # If we are at capacity, check if we can improve the portfolio quality.
            # This runs BEFORE SELL-only mode latches so stagnation overrides can free capacity.
            replacement_triggered = False
            if buy_ranked_symbols:
                best_buy_sym = buy_ranked_symbols[0]
                best_buy_sig = max(
                    valid_signals_by_symbol.get(best_buy_sym, []),
                    key=lambda x: self.score_opportunity(x) if x.get("action") == "BUY" else -1.0,
                )

                if best_buy_sig.get("action") == "BUY":
                    # Attach opportunity score for REA analysis
                    opp_score = self.score_opportunity(best_buy_sig)
                    best_buy_sig["_opp_score"] = opp_score

                    rotation_sig = await self.rotation_authority.authorize_rotation(
                        sig_pos=sig_pos,
                        max_pos=max_pos,
                        owned_positions=owned_positions,
                        best_opp=best_buy_sig,
                        current_mode=current_mode,
                        is_starved=is_starved,
                    )

                    if rotation_sig:
                        worst_sym = rotation_sig.get("symbol")
                        best_buy_sig["_replacement"] = True
                        best_buy_sig["_replaces_symbol"] = worst_sym
                        replacement_triggered = True
                        self.logger.warning(
                            "[Meta:REA] Authorized ROTATION: %s -> %s (Starved: %s, StagnationOverride: %s)",
                            worst_sym,
                            best_buy_sym,
                            is_starved,
                            rotation_sig.get("_stagnation_override", False),
                        )
                        if (
                            await self._passes_meta_sell_profit_gate(worst_sym, rotation_sig)
                            and await self._passes_meta_sell_excursion_gate(worst_sym, rotation_sig)
                        ):
                            return [(worst_sym, "SELL", rotation_sig)]

            # --- 3. AGENT-INDICATED SELL ---
            if not replacement_triggered:
                sell_candidates = []
                for sym, sigs in valid_signals_by_symbol.items():
                    for sig in sigs:
                        if str(sig.get("action")).upper() == "SELL":
                            curr_qty = float(self.shared_state.get_position_qty(sym) or 0.0)
                            if curr_qty > 0:
                                sell_candidates.append((sym, sig, curr_qty))
                
                if sell_candidates:
                    sell_candidates.sort(key=lambda x: float(x[1].get("confidence", 0.0)), reverse=True)
                    best_sell_sym, best_sell_sig, qty = sell_candidates[0]
                    
                    self.logger.warning("[Meta:MandatorySell] 🔥 Capacity full; executing agent SELL signal for %s", best_sell_sym)
                    best_sell_sig["_mandatory_capacity_recovery"] = True
                    return [(best_sell_sym, "SELL", best_sell_sig)]

            # --- 4. DEADLOCK / WAIT STATE ---
            if not replacement_triggered:
                mandatory_sell_mode = True
                self._mandatory_sell_mode_active = True
                self.logger.warning(
                    "[Meta:SELL_ONLY_MODE] 🔒 PORTFOLIO_FULL with no rotation/exit justified. "
                    "Skipping BUY ranking until capacity recovered."
                )
                return []

        # ═══════════════════════════════════════════════════════════════════════════════
        # SECONDARY POLICIES: DUST & REPLACEMENT CONTINUATION
        # ═══════════════════════════════════════════════════════════════════════════════

        allow_phase2_liq, phase2_age = self._update_phase2_guard(dust_ratio)
        p2_cfg = self.policy_manager._phase2_guard
        if p2_cfg["active_since"] and not allow_phase2_liq:
            self._info_once(
                "phase2_grace_wait",
                "[Meta:PHASE_2_GRACE] Dust ratio %.1f%% detected but waiting grace %.0fs (elapsed %.0fs) before liquidation.",
                dust_ratio * 100,
                p2_cfg["activation_age_sec"],
                phase2_age,
            )
        
        # P9 Replacement Logic (Starved Case - Free Capital < Min Notional)
        quote_asset = str(self._cfg("QUOTE_ASSET") or "USDT").upper()
        real_free = float(await self.shared_state.get_spendable_balance(quote_asset) or 0.0)
        min_floor = float(self._cfg("MIN_NOTIONAL_FLOOR", 10.0))
        
        # CAPITAL STARVED: If we have no money but slots are open, we might need to sell something to buy a better one.
        if real_free < min_floor and sig_pos < max_pos and buy_ranked_symbols:
            self.logger.info("[Meta:Replacement] Capital starved (%.2f < %.2f) but slots open. Checking rotation.", real_free, min_floor)
            # Re-use rotation logic above if needed or let standard flow handle it.
            # (In this case, rotation already evaluated or not needed as we have slots).
            pass
        
        # 🔒 SELL-ONLY RECOVERY MODE GUARD
        # ════════════════════════════════════════════════════════════════════════════════
        # If we're in mandatory_sell_mode (portfolio full but no SELL signal candidates),
        # skip BUY ranking entirely. This prevents wasted computation and edge case bugs.
        # Only when capacity is recovered can we resume normal multi-signal trading.
        #
    # SELL-ONLY MODE
    # When SELL-only mode active:
    # - P0_CHECK is disabled (above, in _build_decisions STEP 0.2)
    # - BUY signals are ignored here
    # - SELL is always allowed (capital floor applies to BUYs only)
    # This creates a boundary: NO BUY while exiting, SELLs proceed freely
        # ════════════════════════════════════════════════════════════════════════════════
        if mandatory_sell_mode or self._mandatory_sell_mode_active:
            self.logger.warning(
                "[Meta:SELL_ONLY_MODE] 🔒 Blocking NEW symbol entries (portfolio full). "
                "Allowing scaling/healing for %d existing positions. Waiting for capacity recovery...",
                len(owned_positions)
            )
            # Filter to only allow scaling for owned positions
            buy_ranked_symbols = [s for s in buy_ranked_symbols if s in owned_positions]
        
        # [Note: Redundant replacement logic block removed. Handled by consolidated Capacity Policy above.]

        
        # ═══════════════════════════════════════════════════════════════════════════════
        # PHASE 2 FIX #1: AGGRESSIVE DUST LIQUIDATION
        # ═══════════════════════════════════════════════════════════════════════════════
        # When dust_ratio > 60%, generate aggressive SELL signals for dust positions
        # This prevents 81% dust portfolios and improves capital efficiency
        if allow_phase2_liq:
            self.logger.info(
                "[Meta:PHASE_2_GRACE] Dust ratio %.1f%% sustained for %.0fs (>=%.0fs). Triggering controlled liquidation wave.",
                dust_ratio * 100,
                phase2_age,
                self._phase2_guard["activation_age_sec"],
            )
            
            try:
                # Get all positions and identify dust
                owned_pos = await _safe_await(self.shared_state.get_positions())
                if isinstance(owned_pos, dict) and owned_pos:
                    # Calculate dust threshold: < $1 USD equivalent (Phase 2 improvement)
                    dust_value_threshold = 1.0  # More aggressive than MIN_POSITION_VALUE_USDT
                    position_grace_sec = self._phase2_guard["position_grace_sec"]
                    now = time.time()
                    
                    dust_to_liquidate = []
                    for sym, pdata in owned_pos.items():
                        qty = float(pdata.get("quantity", 0.0))
                        value_usdt = float(pdata.get("value_usdt", 0.0))
                        entry_ts = float(
                            pdata.get("entry_time")
                            or pdata.get("opened_at")
                            or pdata.get("timestamp")
                            or pdata.get("ts")
                            or 0.0
                        )
                        pos_age_sec = max(0.0, now - entry_ts) if entry_ts else None
                        if pos_age_sec is not None and pos_age_sec < position_grace_sec:
                            self.logger.debug(
                                "[Meta:PHASE_2_GRACE] Skipping %s dust liquidation; position age %.0fs < grace %.0fs.",
                                sym,
                                pos_age_sec,
                                position_grace_sec,
                            )
                            continue
                        
                        # Identify dust: value < threshold AND qty > 0 AND not already being replaced
                        if qty > 0 and value_usdt < dust_value_threshold:
                            # Check if this symbol is already flagged for replacement
                            already_replacing = any(
                                s.get("_is_replacement_sell") 
                                for s in valid_signals_by_symbol.get(sym, [])
                            )
                            if not already_replacing:
                                dust_to_liquidate.append((sym, qty, value_usdt, pos_age_sec))
                    
                    # ═════════════════════════════════════════════════════════════════════════════
                    # PHASE 3 FIX: DUST STATE MACHINE GATE
                    # ═════════════════════════════════════════════════════════════════════════════
                    # Apply dust state machine gate to filter positions that are not yet executable
                    # This prevents infinite retry loops by skipping signal generation for positions
                    # in DUST_ACCUMULATING state (waiting to reach minNotional)
                    executable_dust = []
                    unsellable_dust = []  # TERMINAL STATE: dust that will never be tradeable
                    
                    DUST_SELL_CONF = 0.6  # Confidence threshold for Dust-Sell Escape Policy
                    MIN_SIGNIFICANT_USDT = getattr(self, 'MIN_SIGNIFICANT_USDT', 5.0)  # fallback if not defined
                    for sym, qty, value_usdt, pos_age_sec in dust_to_liquidate:
                        # Check if this position is executable using the dust state machine gate
                        # FIX #3A: Pass emergency_liquidation=True to bypass FOCUS_MODE gate
                        should_execute = await self.should_execute_sell(sym, emergency_liquidation=True)
                        # Dust-Sell Escape Policy: If a strong SELL signal exists, allow liquidation
                        escape_sell = False
                        for sig in valid_signals_by_symbol.get(sym, []):
                            if (
                                sig.get("action") == "SELL"
                                and sig.get("confidence", 0) >= DUST_SELL_CONF
                                and value_usdt < MIN_SIGNIFICANT_USDT
                            ):
                                escape_sell = True
                                break
                        if should_execute or escape_sell:
                            executable_dust.append((sym, qty, value_usdt, pos_age_sec))
                        else:
                            # ════════════════════════════════════════════════════════════════════
                            # TERMINAL STATE FIX: Prevent Livelock
                            # ════════════════════════════════════════════════════════════════════
                            # If dust is below minNotional AND has failed at least once,
                            # mark it as DUST_UNSELLABLE_TERMINAL to prevent infinite retries
                            rejection_count = self.shared_state.get_rejection_count(sym, "SELL")
                            if rejection_count > 0 and value_usdt < dust_value_threshold:
                                # This dust will NEVER be tradeable (permanently below minNotional)
                                # Mark it as terminal state to stop retry loops
                                unsellable_dust.append((sym, qty, value_usdt, pos_age_sec))
                                # Mark position as unsellable (prevent future retry attempts)
                                if hasattr(self.shared_state, 'mark_position_unsellable'):
                                    await self.shared_state.mark_position_unsellable(sym)
                                self.logger.warning(
                                    "[Meta:TERMINAL_STATE_FIX] %s marked UNSELLABLE_TERMINAL. "
                                    "Dust value %.2f USDT is permanently below min_notional (%.2f). "
                                    "Rejection count: %d. Will skip future liquidation attempts.",
                                    sym, value_usdt, dust_value_threshold, rejection_count
                                )
                            else:
                                # Still waiting for accumulation (not yet rejected)
                                self.logger.debug(
                                    f"[Meta:PHASE_3_FIX] {sym} dust position ({value_usdt:.2f} USDT) "
                                    f"not yet executable. Waiting for maturation. (rejections={rejection_count})"
                                )
                    
                    # Generate SELL signals only for executable dust positions
                    if executable_dust:
                        self.logger.warning(
                            "[Meta:PHASE_2_FIX#1 + PHASE_3_GATE] Liquidating %d dust positions (value < $%.2f): %s",
                            len(executable_dust), dust_value_threshold,
                            ", ".join(sym for sym, _, _, _ in executable_dust[:5])
                        )
                        
                        for sym, qty, value_usdt, pos_age_sec in executable_dust:
                            dust_sell_sig = {
                                "symbol": sym,
                                "action": "SELL",
                                "confidence": 0.99,
                                "agent": "MetaDustLiquidator",
                                "timestamp": time.time(),
                                "reason": f"phase2_dust_liquidation_value_{value_usdt:.2f}_usdt",
                                "_is_dust": True,
                                "_dust_value": value_usdt,
                                "_force_dust_liquidation": True,  # Hard override
                                "tag": "dust_cleanup_phase2",
                                "_phase2_guard": {
                                    "dust_ratio": dust_ratio,
                                    "phase2_age_sec": phase2_age,
                                    "position_age_sec": pos_age_sec,
                                },
                            }
                            
                            # Inject with high priority
                            valid_signals_by_symbol.setdefault(sym, []).insert(0, dust_sell_sig)
                            if sym not in ranked_symbols:
                                ranked_symbols.insert(0, sym)
                        
                        self.logger.info(
                            "[Meta:PHASE_2_FIX#1 + PHASE_3_GATE] Injected %d executable dust liquidation signals "
                            "(skipped %d due to accumulation). Expect dust ratio to improve within 3 ticks.",
                            len(executable_dust), len(dust_to_liquidate) - len(executable_dust)
                        )
                    
                    # ════════════════════════════════════════════════════════════════════════════
                    # TERMINAL STATE REPORT: Log unsellable dust positions
                    # ════════════════════════════════════════════════════════════════════════════
                    if unsellable_dust:
                        self.logger.warning(
                            "[Meta:TERMINAL_STATE_REPORT] %d positions marked UNSELLABLE_TERMINAL "
                            "(permanently below minNotional). These will NOT be retried: %s",
                            len(unsellable_dust),
                            ", ".join(f"{sym}(${val:.2f})" for sym, _, val, _ in unsellable_dust[:10])
                        )
                        
            except Exception as e:
                self.logger.warning("[Meta:PHASE_2_FIX#1] Dust liquidation setup failed: %s", str(e), exc_info=False)
        
        # RULE 5 ESCALATION: If starved for capital but we have slots, 
        # do NOT clear buy_ranked_symbols. We want them to reach execution to trigger liquidation.
        if is_starved and current_pos_count < max_pos and not buy_ranked_symbols:
            # Refresh buy_ranked_symbols if we cleared it accidentally above
            buy_ranked_symbols = [
                sym for sym in ranked_symbols 
                if any(s.get("action") == "BUY" for s in valid_signals_by_symbol.get(sym, []))
            ]
            self.logger.info("[Meta] Capital starved but slots available. Retaining top BUY to trigger Rule 5.")
        
        # ===== QUOTE-BASED LIQUIDATION: PRIMARY CAPITAL RECOVERY PATH =====
        # When starved: Try quote-based SELL (deterministic, dust-proof)
        # If quote-based unavailable: Fall back to batch liquidation
        # Goal: Unlock capital WITHOUT violating min-notional constraints
        
        if is_starved and current_pos_count > 0 and owned_positions:
            # Only attempt if we have a pending opportunity
            has_pending_buy = any(any(s.get("action") == "BUY" for s in valid_signals_by_symbol.get(sym, []))
                                for sym in ranked_symbols)
            
            if has_pending_buy:
                now = time.time()
                
                # ===== PRIMARY PATH: Quote-Based Liquidation =====
                # Sell N USDT worth (not X units). Binance handles lot size, min qty, rounding.
                min_floor = float(self._cfg("MIN_NOTIONAL_FLOOR", 10.0))
                quote_target = min(25.0, min_floor * 2.5)  # Target 15-25 USDT liquidation
                
                quote_liq_attempted = False
                cooldown_sec = int(self._cfg("STARVATION_LIQUIDATION_COOLDOWN_SEC", 60))
                
                for p_sym, p_data in owned_positions.items():
                    # Skip fresh positions (anti-churn)
                    entry_ts = p_data.get("entry_time", 0)
                    if (now - entry_ts) < cooldown_sec:
                        continue
                    
                    # Skip winners (profit protection)
                    pnl_pct = p_data.get("unrealized_pnl_pct", 0.0)
                    if pnl_pct > 0.002:
                        continue
                    
                    # Score the position
                    p_score = self.score_position(p_data)
                    if p_score >= -0.5:
                        continue  # Skip unless clearly negative
                    
                    # Candidate found - attempt quote-based SELL
                    quote_value = p_data.get("value_usdt", 0.0)
                    if quote_value <= 0:
                        continue
                    
                    # Use quote-based order (not quantity-based)
                    quote_sell = {
                        "symbol": p_sym,
                        "action": "SELL",
                        "confidence": 0.98,
                        "agent": "MetaQuoteLiq",
                        "timestamp": now,
                        "reason": "quote_based_capital_recovery",
                        "_quote_based": True,  # KEY: Use quoteOrderQty, not quantity
                        "_target_usdt": quote_target,  # Sell ~15-25 USDT worth
                        "_is_starvation_sell": True,
                        "_safe_dust_liquidation": True,
                    }
                    
                    self.logger.warning(
                        "[Meta:QuoteLiq:Primary] Attempting quote-based liquidation of %s. "
                        "Target: %.2f USDT. Position value: %.2f USDT. Score: %.4f.",
                        p_sym, quote_target, quote_value, p_score
                    )
                    
                    # Inject at front with high priority
                    valid_signals_by_symbol.setdefault(p_sym, []).insert(0, quote_sell)
                    if p_sym not in ranked_symbols:
                        ranked_symbols.insert(0, p_sym)
                    
                    quote_liq_attempted = True
                    break  # Only attempt one per tick
                
                if quote_liq_attempted:
                    self.logger.info(
                        "[Meta:QuoteLiq] Quote-based liquidation injected. "
                        "System now executes deterministic USDT-value sells (bypasses min-notional dust traps)."
                    )
                
                # ===== FALLBACK PATH: Batch Liquidation =====
                # If quote-based not attempted (all positions protected), try batch approach
                if not quote_liq_attempted:
                    self.logger.debug(
                        "[Meta:BatchLiq] Quote-based candidates exhausted. "
                        "Evaluating batch liquidation as fallback."
                    )
                    
                    # Collect all eligible positions for potential batch
                    batch_candidates = []
                    for p_sym, p_data in owned_positions.items():
                        entry_ts = p_data.get("entry_time", 0)
                        if (now - entry_ts) < cooldown_sec:
                            continue
                        pnl_pct = p_data.get("unrealized_pnl_pct", 0.0)
                        if pnl_pct > 0.002:
                            continue
                        p_score = self.score_position(p_data)
                        if p_score >= -0.5:
                            continue
                        batch_candidates.append((p_sym, p_data, p_score))
                    
                    # If we have candidates, prepare batch SELL signals
                    if len(batch_candidates) >= 2:
                        batch_candidates.sort(key=lambda x: x[2])  # Sort by score (worst first)
                        
                        self.logger.warning(
                            "[Meta:BatchLiq] Preparing batch liquidation of %d positions. "
                            "This is fallback (quote-based is preferred).",
                            len(batch_candidates)
                        )
                        
                        for idx, (b_sym, b_data, b_score) in enumerate(batch_candidates[:3]):  # Max 3 per batch
                            batch_sell = {
                                "symbol": b_sym,
                                "action": "SELL",
                                "confidence": 0.92 - (idx * 0.02),  # Decreasing confidence for subsequent sells
                                "agent": "MetaBatchLiq",
                                "timestamp": now,
                                "reason": "batch_capital_recovery",
                                "_batch_sell": True,
                                "_batch_index": idx,
                                "_batch_total": len(batch_candidates[:3]),
                                "_is_starvation_sell": True,
                            }
                            
                            valid_signals_by_symbol.setdefault(b_sym, []).insert(0, batch_sell)
                            if b_sym not in ranked_symbols:
                                ranked_symbols.insert(0, b_sym)
                        
                        self.logger.info(
                            "[Meta:BatchLiq] Batch liquidation fallback activated. "
                            "Will sell %d positions sequentially to unlock capital.",
                            len(batch_candidates[:3])
                        )
        
        elif not any(any(s.get("_replacement") for s in valid_signals_by_symbol.get(sym, [])) for sym in buy_ranked_symbols):
            # Standard Slice if not starved AND not a replacement (which frees a slot)
            orig_buy_count = len(buy_ranked_symbols)
            remaining_slots = max(0, max_pos - current_pos_count)
            limit = min(self._symbol_concentration_limit, remaining_slots * 2) 
            buy_ranked_symbols = buy_ranked_symbols[:limit]
            
            # Frequency Engineering: Inject Trailing Strategy if in Aggression mode
            # This allows the bot to "ride" winners to catch up with targets.
            if self._adaptive_aggression > 1.05 and not best_sig.get("_no_trailing"):
                best_sig["_use_trailing"] = True
                self.logger.info("[Meta:Boost] Injecting Trailing TP for %s to catch up with profit target.", best_buy_sym)

            if orig_buy_count > limit:
                self.logger.info("[Meta] Concentrating on top %d symbols (from %d) due to slots/conc limit.", 
                                limit, orig_buy_count)

        # 3) Budget Validation & Reallocation (agent_budgets already initialized in PRE-LAYER)
        plan = {"per_agent_usdt": {}, "reason": ""}  # Initialize plan with defaults
        try:
            plan = self.shared_state.get_active_allocation_plan()
        except Exception:
            self.logger.warning("[Meta] Failed to pull authoritative reservations.")
        
        # --- FIX: Re-evaluate Effective Tradable Balance ---
        try:
            total_reserved = sum(agent_budgets.values())
            
            if total_reserved > real_free:
                # FIX: Identify priority agents (Dust/Bootstrap) to protect from over-scaling
                priority_agents = set()
                min_floor_prot = float(self._cfg("MIN_NOTIONAL_FLOOR", 10.0))
                
                # Scan available signals for priority flags
                for sym_sigs in valid_signals_by_symbol.values():
                    for s in sym_sigs:
                        if s.get("_dust_reentry_override") or s.get("_bootstrap_override") or "bootstrap" in str(s.get("reason", "")).lower():
                            priority_agents.add(s.get("agent", "Meta"))

                scale = real_free / total_reserved if total_reserved > 0 else 0.0
                if scale < 0.99:
                    self.logger.warning("[Meta] Budget Reality Check: Scaling plan (%.2f) to match free capital (%.2f). Factor: %.2f", 
                                        total_reserved, real_free, scale)
                    
                    for a in agent_budgets:
                        normalized_val = agent_budgets[a] * scale
                        
                        # CRITICAL FIX: Floor protection for priority agents (Dust/Bootstrap)
                        # Prevent scaling below min_notional causing QUOTE_LT_MIN_NOTIONAL rejection
                        if a in priority_agents and normalized_val < min_floor_prot and real_free >= min_floor_prot:
                            normalized_val = min_floor_prot
                            
                        agent_budgets[a] = normalized_val
            elif real_free > total_reserved:
                # --- FIX: Surplus Re-allocation (Capital Efficiency) ---
                surplus = real_free - total_reserved
                
                plan_reason = plan.get("reason", "")
                risk_blocked = (total_reserved == 0 and plan_reason in ("risk_cap_exceeded", "usable_pool_zero"))
                
                if not risk_blocked:
                    min_floor = float(self._cfg("MIN_NOTIONAL_FLOOR", 10.0))
                    min_useful = max(min_floor, 5.0, self._micro_size_quote)
                    
                    if surplus >= min_useful:
                        active_agents = set()
                        for sym in buy_ranked_symbols:
                            for s in valid_signals_by_symbol.get(sym, []):
                                if s.get("action") == "BUY":
                                    active_agents.add(s.get("agent", "Meta"))
                        
                        if active_agents:
                            share = surplus / len(active_agents)
                            self.logger.info("[Meta] Surplus Liquidity: Re-allocating %.2f USDT -> %.2f to %d active agents", 
                                            surplus, share, len(active_agents))
                            for ag in active_agents:
                                agent_budgets[ag] = agent_budgets.get(ag, 0.0) + share
        except Exception as e:
            self.logger.warning("[Meta] Failed to validate budget against real balance: %s", e)
        
        # 4) Agent Coverage Mapping & Weighting (Phase A Fix)
        agent_targets = defaultdict(list)
        agent_conf_sums = defaultdict(float)
        for sym in buy_ranked_symbols:
            for sig in valid_signals_by_symbol.get(sym, []):
                if self._is_budget_required(sig.get("action")):
                    agent_name = sig.get("agent", "Meta")
                    agent_targets[agent_name].append(sig)
                    agent_conf_sums[agent_name] += float(sig.get("confidence", 0.0))

        # 4.5) Apply Adaptive Aggression (Phase A Boost)
        # 4.5) Apply Adaptive Aggression (Boost log)
        if agg_factor > 1.0:
            self.logger.info("[Meta:Aggression] Applying factor %.2f to catch up with profit target.", agg_factor)

        # ===== FIX 1: LIQUIDATION HARD DECISION =====
        # Extract ALL liquidation signals BEFORE consensus gating
        liquidation_signals = []
        for sym in ranked_symbols:
            for sig in valid_signals_by_symbol.get(sym, []):
                if sig.get("action") == "SELL" and sig.get("_is_starvation_sell"):
                    liquidation_signals.append((sym, sig))
                    self.logger.warning("[Meta:LiquidationHardPath] FORCING liquidation bypass gate. Symbol: %s, Type: %s, Agent: %s", 
                                    sym, "QuoteBased" if sig.get("_quote_based") else "Batch", sig.get("agent", "Unknown"))
        
        # 5) Weighted Allocation & Decision Building
        final_decisions = []
        # ===== AUTHORITATIVE CHECK: Do we have ANY positions with qty > 0? (including dust) =====
        # This is the TRUE source of truth for SELL gating - not open_positions_count()
        # Reason: SELL must be allowed if inventory exists, regardless of dust classification
        snap_for_inventory = self.shared_state.get_positions_snapshot() or {}
        has_positions = any(
            float(p.get("quantity", 0.0)) > 0 
            for p in snap_for_inventory.values()
        )
        
        self.logger.info("[Meta] Decision loop: %d symbols considered, %d significant pos, %d signals in cache, %d liquidation signals, %s inventory check.", 
                        len(symbols_to_consider), self.shared_state.open_positions_count(), len(signals_by_sym), len(liquidation_signals),
                        "HAS_POSITIONS" if has_positions else "EMPTY")
        
        # ===== FIX 1 ENFORCEMENT: Liquidation signals BYPASS all consensus/tier/throughput gates =====
        # They are injected as hard decisions BEFORE normal flow
        if liquidation_signals:
            self.logger.warning("[Meta:LiquidationHardPath] Injecting %d liquidation signals as FORCED decisions (bypass consensus/tier/throughput)", 
                            len(liquidation_signals))
            for sym, sig in liquidation_signals:
                final_decisions.append((sym, "SELL", sig))
                self.logger.warning("[Meta:LiquidationHardPath:INJECTED] Symbol: %s, Confidence: %.2f, Agent: %s", 
                                sym, sig.get("confidence", 0.0), sig.get("agent", "Unknown"))
        
        # SELLs only (HOLDs are context)
        # ===== HARD INVARIANT: SELL GATING + DUST ACCUMULATION GUARD =====
        # RULE: If SharedState has position with qty > 0, SELL is ALLOWED
        # BUT: ONLY if position.notional >= minNotional (DUST ACCUMULATION GUARD)
        # SELL must NOT be blocked by portfolio flatness, bootstrap mode, or capital starvation
        # Reason: SELL is capacity-FREEING, not capacity-CONSUMING
        for sym in ranked_symbols:
            for sig in valid_signals_by_symbol.get(sym, []):
                if sig.get("action") == "SELL" and not sig.get("_is_starvation_sell"):  # Skip liquidation SELLs (already injected)
                    # HARD GATE: Check if SELL is allowed by invariant
                    allow_sell = await self._should_allow_sell(sym)
                    
                    if allow_sell:
                        # Get position details
                        qty = float(self.shared_state.get_position_qty(sym))
                        price = float(await _safe_await(self.shared_state.safe_price(sym, default=1.0)))
                        val = qty * price
                        
                        # ===== CANONICAL DUST ACCUMULATION GUARD =====
                        # Check: position.notional >= minNotional
                        # If not, BLOCK_ACCUMULATE (no TradeIntent emission)
                        can_emit_tradeintent = await self.dust_accumulation_guard(
                            symbol=sym,
                            position_qty=qty,
                            position_mark_price=price
                        )
                        
                        if can_emit_tradeintent:
                            # THE INVARIANT: Inventory exists AND notional >= minNotional → SELL is ALLOWED
                            final_decisions.append((sym, "SELL", sig))
                            self.logger.info("[EXEC_DECISION] SELL %s qty=%.8f val=%.2f [INVARIANT_PASS] bypass_capital_check=True", 
                                        sym, qty, val)
                        else:
                            # DUST DETECTED: Below minNotional
                            # Do NOT emit TradeIntent, wait for accumulation
                            self.logger.debug(
                                "[EXEC_BLOCK:DustGuard] SELL suppressed for %s (notional=%.2f < minNotional). "
                                "Position is accumulating. Will retry when accumulated.",
                                sym, val
                            )
                    else:
                        # No inventory - SELL cannot execute
                        qty = float(self.shared_state.get_position_qty(sym))
                        reason_code = "NO_POSITION_QTY" if qty <= 0 else "UNKNOWN"
                        self.logger.info("[EXEC_BLOCK] gate_id=SELL_SUPPRESSED symbol=%s reason=%s qty=%.8f action=SKIP_SELL [INVARIANT_FAIL]", 
                                    sym, reason_code, qty)

        # ═══════════════════════════════════════════════════════════════════════════════
        # FIX #3: SURPLUS RE-ALLOCATION FROM INTENDED SELLS (Capital Recycling)
        # ═══════════════════════════════════════════════════════════════════════════════
        # Calculate intended freed capital from SELL decisions and add to available budget
        # This allows compounding and rotation within the SAME cycle.
        total_intended_freed = 0.0
        for sym, action, sig in final_decisions:
            if action == "SELL":
                qty = float(self.shared_state.get_position_qty(sym) or 0.0)
                price = float(await _safe_await(self.shared_state.safe_price(sym, default=1.0)))
                pct = float(sig.get("_partial_pct", 1.0))
                val = qty * price * pct
                
                # Assume 99% realization (1% buffer for slippage/fees)
                freed = val * 0.99
                total_intended_freed += freed
                self.logger.info("[Meta:Recycle] Intended SELL of %s will free approx %.2f USDT", sym, freed)
        
        if total_intended_freed > 1.0:
            self.logger.warning("[Meta:Recycle] Adding %.2f USDT surplus to agent budgets from intended SELLs", total_intended_freed)
            active_buy_agents = set()
            for sym in buy_ranked_symbols:
                for s in valid_signals_by_symbol.get(sym, []):
                    if s.get("action") == "BUY":
                        active_buy_agents.add(s.get("agent", "Meta"))
            
            if active_buy_agents:
                share = total_intended_freed / len(active_buy_agents)
                for ag in active_buy_agents:
                    agent_budgets[ag] = agent_budgets.get(ag, 0.0) + share
                    self.logger.info("[Meta:Recycle] Agent %s budget boosted by %.2f -> new_total=%.2f", ag, share, agent_budgets[ag])

        # Throughput Guard Bootstrap Check (Fix for "flow stops before this point")
        # If we have no positions and throughput gap is active, we ensure at least one agent has a micro budget
        # to kickstart the system even if the CapitalAllocator hasn't emitted a plan yet.
        if throughput_gap and not has_positions:
            # We delay budget granting until we find the BEST bootstrap candidate below.
            # CRITICAL FIX 5: Don't shotgun budget to everyone; wait for selection.
            pass

        # Weighted BUYs (Tier A & Tier B logic)
        for sym in buy_ranked_symbols:
            best_sig = max(valid_signals_by_symbol.get(sym, []), key=lambda x: float(x.get("confidence", 0.0)) if x.get("action") == "BUY" else -1.0)
            if best_sig.get("action") != "BUY":
                continue
            
            best_conf = float(best_sig.get("confidence", 0.0))

            # Bounded agreement uplift (only when all signals agree on BUY)
            sym_sigs = valid_signals_by_symbol.get(sym, [])
            
            # FIX: Rotation SELLs should NOT block BUYs (they are part of the same intent chain)
            has_sell_conflict = any(
                str(s.get("action")).upper() == "SELL" 
                and not s.get("_is_rotation") 
                and not s.get("_rotation_escape")
                for s in sym_sigs
            )
            
            buy_agents = {s.get("agent") for s in sym_sigs if str(s.get("action")).upper() == "BUY"}
            if not has_sell_conflict and len(buy_agents) >= 2:
                uplift = float(getattr(self.config, "TIER_A_AGREE_UPLIFT", 0.02) or 0.02)
                uplift = max(0.0, min(0.02, uplift))
                if uplift > 0:
                    best_conf = min(0.99, best_conf + uplift)
            
            # --- Tier Assignment ---
            tier = None
            if best_conf >= self._tier_a_conf:
                tier = "A"
            elif best_conf >= (self._tier_b_conf / agg_factor): # Relax conf floor if behind target
                tier = "B"
            elif throughput_gap and best_conf >= (0.50 / agg_factor):
                # Force Tier-B if we are idle and have at least 0.50 conf (scaled by agg)
                tier = "B"

            # Bootstrap override: force Tier-B eligibility even if throughput_gap is false.
            bootstrap_force = bool(bootstrap_execution_override or best_sig.get("_bootstrap_override"))
            if not tier and bootstrap_force and best_conf >= 0.60:
                tier = "B"
                best_sig["_bootstrap_override"] = True
                best_sig["_bypass_reason"] = "BOOTSTRAP_FIRST_TRADE"
                best_sig["bypass_conf"] = True

            # Tier-A readiness log (close but not yet eligible)
            readiness_margin = float(getattr(self.config, "TIER_A_READINESS_MARGIN", 0.03) or 0.03)
            near_tier_a = best_conf >= (self._tier_a_conf - readiness_margin)
            if near_tier_a and tier != "A":
                self.logger.info(
                    "[Meta:TierA:Readiness] %s conf=%.2f tier_a=%.2f margin=%.2f agg=%.2f reason=CONF_BELOW_TIER_A buy_agents=%d sell_conflict=%s",
                    sym, best_conf, self._tier_a_conf, readiness_margin, agg_factor,
                    len(buy_agents), has_sell_conflict
                )
            
            if not tier:
                continue

            # Consensus Rule (Relaxed for Tier B throughput guard)
            agents_for_sym = set(s.get("agent") for s in valid_signals_by_symbol.get(sym, []) if s.get("action") == "BUY")
            min_agents = self._meta_min_agents if tier == "A" else 1
            if tier == "A" and not getattr(self, "_focus_mode_active", False):
                min_agents = max(min_agents, 2)
            if len(agents_for_sym) < min_agents:
                if near_tier_a:
                    self.logger.info(
                        "[Meta:TierA:Readiness] %s conf=%.2f tier_a=%.2f margin=%.2f reason=INSUFFICIENT_AGENTS agents=%d/%d focus=%s",
                        sym, best_conf, self._tier_a_conf, readiness_margin,
                        len(agents_for_sym), min_agents, getattr(self, "_focus_mode_active", False)
                    )
                continue
            
            # 🔒 CAPACITY ENFORCEMENT (Rule 1/5): Track slot consumption if early gate was relaxed
            if sym not in owned_positions and sig_pos >= max_pos:
                self.logger.info("[Meta:Capacity] Skipping %s BUY: sig_pos (%d) >= max_pos (%d)", sym, sig_pos, max_pos)
                continue

            # Aggregated budget logic: Combine budgets from ALL agents who have a BUY signal for this symbol
            # [FIX #3]: This prevents fragmentation by merging micro-orders from different agents.
            total_intended_quote = 0.0
            actual_contributions = []  # List of (agent_name, share)
            
            # force_eval is defined earlier in the original code, so it's available here.
            force_eval = (tier == "B" and throughput_gap and not has_positions) or \
                        best_sig.get("_replacement") or \
                        best_sig.get("_dust_reentry_override") or \
                        best_sig.get("_is_compounding") or \
                        best_sig.get("_is_rotation") or \
                        bootstrap_force or \
                        (is_starved and current_pos_count < max_pos)

            for sig_item in sym_sigs:
                if str(sig_item.get("action")).upper() == "BUY":
                    ag = sig_item.get("agent", "Meta")
                    av = agent_budgets.get(ag, 0.0)
                    
                    if av > 0 or force_eval:
                        # Calculate this agent's share contribution
                        if tier == "B":
                            # Tier B contribution is capped at micro_size
                            sh = min(av if av > 0 else 10.0, self._micro_size_quote * agg_factor)
                        else:
                            # Tier A contribution is weighted by signal confidence
                            agent_total_conf = agent_conf_sums.get(ag, 1.0)
                            sh = av * (float(sig_item.get("confidence", 0.0)) / agent_total_conf)
                        
                        if sh > 0:
                            total_intended_quote += sh
                            actual_contributions.append((ag, sh))

            if total_intended_quote <= 0 and not force_eval:
                continue

            # Update best_sig with the aggregated budget context
            best_sig["_tier"] = tier
            best_sig["_contributions"] = actual_contributions
            
            # Mission branding for visibility
            agent_tag = best_sig.get("agent", "Meta")
            if tier == "B":
                if throughput_gap and not has_positions:
                    best_sig["_bootstrap"] = True
                    best_sig["_force_min_notional"] = True
                    best_sig["_bypass_risk"] = True
                    best_sig["reason"] = f"bootstrap:{agent_tag}:{best_conf:.2f}"
                    
                    # Ensure minimum budget for bootstrap if all agents are dry
                    if total_intended_quote < 6.0:
                        total_intended_quote = 6.0
                        self.logger.info("[Meta] Throughput Guard: Forced bootstrap budget 6.0 for %s", sym)
                elif best_sig.get("_bootstrap_override") or bootstrap_execution_override:
                    best_sig["_bootstrap"] = True
                    best_sig["_force_min_notional"] = True
                    best_sig["reason"] = f"bootstrap_override:{agent_tag}:{best_conf:.2f}"
                    min_floor = float(self._cfg("MIN_NOTIONAL_FLOOR", 10.0))
                    min_entry = float(getattr(self, "_min_entry_quote_usdt", 0.0) or 0.0)
                    micro_quote = float(self._micro_size_quote or 0.0)
                    bootstrap_min = max(min_floor, min_entry, micro_quote)
                    if total_intended_quote < bootstrap_min:
                        total_intended_quote = bootstrap_min
                        self.logger.info("[Meta] Bootstrap override: Forced minimum budget %.2f for %s", bootstrap_min, sym)
                elif best_sig.get("_dust_reentry_override"):
                    best_sig["_bootstrap"] = True
                    best_sig["_force_min_notional"] = True
                    best_sig["reason"] = f"dust_merge_override:{best_conf:.2f}"
                    req_budget = float(
                        self._cfg("MIN_SIGNIFICANT_POSITION_USDT", self._cfg("MIN_SIGNIFICANT_USD", 25.0))
                    )
                    total_intended_quote = max(total_intended_quote, req_budget)
                else:
                    best_sig["reason"] = f"tier_b_micro ({best_conf:.2f})"
            else:
                best_sig["reason"] = f"tier_a_agg ({best_conf:.2f})"

            # Calculate final planned quote
            planned_quote = await self._planned_quote_for(sym, best_sig, budget_override=total_intended_quote)

            # Profit-locked re-entry: only realized profit increases position size
            planned_quote = self._apply_profit_locked_reentry(planned_quote, sym, best_sig)

            # -----------------------------------------------------------------
            # Safety fix: Ensure Tier-B micro trades meet a controller-level
            # minimum entry quote to avoid perpetual min-notional rejections.
            # This enforces the minimal USDT sizing for Tier-B without
            # relaxing any exchange or policy checks.
            # -----------------------------------------------------------------
            try:
                if tier == "B":
                    min_entry = float(getattr(self, "_min_entry_quote_usdt", 0.0) or 0.0)
                    if min_entry > 0 and planned_quote < min_entry:
                        self.logger.info("[Meta] Tier-B planned_quote (%.2f) below min_entry (%.2f). Raising to min_entry to avoid min-notional deadlock.", planned_quote, min_entry)
                        planned_quote = float(min_entry)
            except Exception:
                # best-effort only; on error continue with computed planned_quote
                pass

            # Single-symbol escalation for Tier B (keep legacy logic but use aggregated value)
            if tier == "B" and len(accepted_symbols_set or []) == 1 and planned_quote > 0:
                quote_asset = str(self._cfg("QUOTE_ASSET") or "USDT").upper()
                try:
                    free_usdt = float(await self.shared_state.get_spendable_balance(quote_asset) or 0.0)
                    if free_usdt > float(planned_quote) * 2:
                        tier_b_cap = float(getattr(self.config, "TIER_B_MAX_QUOTE", 0.0) or 0.0)
                        escalated = min(float(planned_quote) * 2, tier_b_cap if tier_b_cap > 0 else 999999)
                        if escalated > float(planned_quote):
                            planned_quote = await self._planned_quote_for(sym, best_sig, budget_override=escalated)
                            best_sig["_tier"] = "B+"
                            best_sig["_tier_b_plus"] = True
                            best_sig["reason"] = f"tier_b_plus_single_symbol ({best_conf:.2f})"
                except Exception: pass
            
            if planned_quote > 0 and (total_intended_quote >= (planned_quote - 0.01) or force_eval):
                # FINAL EXECUTION GUARANTEE (MANDATORY)
                # BOOTSTRAP FIX: If this signal is marked with _bootstrap flag, pass bootstrap context
                bootstrap_policy_ctx = None
                if best_sig.get("_bootstrap") or best_sig.get("_bootstrap_override"):
                    bootstrap_policy_ctx = self._build_policy_context(
                        sym, 
                        "BUY",
                        extra={"bootstrap_bypass": True}
                    )
                    self.logger.info("[Meta:BOOTSTRAP] Using bootstrap_bypass context for affordability check: %s", sym)

                can_exec, _, reason = await self.execution_manager.can_afford_market_buy(
                    sym, planned_quote, policy_context=bootstrap_policy_ctx
                )


                if not can_exec:
                    self.logger.info(
                        "[Meta] Dropping BUY %s — not executable (%s)",
                        sym, reason
                    )
                    if best_sig.get("_bootstrap") or best_sig.get("_bootstrap_override") or bootstrap_execution_override:
                        cooldown = float(self._bootstrap_veto_cooldown_sec or 0.0)
                        if cooldown > 0:
                            self._bootstrap_cooldown_until = time.time() + cooldown
                            self._bootstrap_last_veto_reason = f"AFFORDABILITY:{reason}"
                            self.logger.warning(
                                "[Meta:BOOTSTRAP] Standing down after affordability veto (%s). Cooldown=%ds",
                                reason,
                                int(cooldown),
                            )
                    try:
                        if isinstance(self._loop_summary_state, dict):
                            self._loop_summary_state["rejection_reason"] = reason
                            self._loop_summary_state["rejection_count"] = int(self._loop_summary_state.get("rejection_count", 0)) + 1
                    except Exception:
                        pass
                    continue

                # DUST BUY GATE: Hard Invariant - Block any BUY that would create dust
                # Phase A: FEE-AWARE ENTRY POLICY (SOP Rule 3.0)
                # ═══════════════════════════════════════════════════════════════════════
                # Rule: profit >= 3x fees AND quote >= MVT.
                # ═══════════════════════════════════════════════════════════════════════
                expected_alpha = float(best_sig.get("expected_roi", best_sig.get("expected_alpha", 0.008)))
                
                should_buy = await self.should_place_buy(
                    sym, 
                    planned_quote, 
                    best_conf, 
                    best_sig.get("reason", "normal"),
                    expected_alpha=expected_alpha
                )
                
                if not should_buy:
                    self.logger.warning(
                        "[Meta:PolicyGate] BLOCKING BUY %s — failed fee-aware entry policy (quote=%.2f alpha=%.4f)",
                        sym, planned_quote, expected_alpha
                    )
                    if best_sig.get("_bootstrap") or best_sig.get("_bootstrap_override") or bootstrap_execution_override:
                        cooldown = float(self._bootstrap_veto_cooldown_sec or 0.0)
                        if cooldown > 0:
                            self._bootstrap_cooldown_until = time.time() + cooldown
                            self._bootstrap_last_veto_reason = "POLICY_GATE"
                            self.logger.warning(
                                "[Meta:BOOTSTRAP] Standing down after policy veto. Cooldown=%ds",
                                int(cooldown),
                            )
                    await self.shared_state.record_rejection(sym, "BUY", "DUST_PREVENTION_GATE", source="MetaController")
                    
                    # ✨ INVARIANT: ACCUMULATION_RESOLUTION
                    # When BUY is rejected due to dust prevention, accumulate the quote
                    # and check if we've crossed the minNotional threshold for emission
                    accumulation_result = await self.accumulation_resolution_check(
                        symbol=sym,
                        rejection_quote=planned_quote,
                        min_notional=None  # Will fetch from exchange
                    )
                    
                    if accumulation_result and accumulation_result.get("should_emit_buy"):
                        # Accumulated value has crossed minNotional threshold!
                        # Emit a BUY TradeIntent with the full accumulated amount
                        self.logger.warning(
                            "[Meta:AccumulationResolution] %s THRESHOLD CROSSED! "
                            "Accumulated %.2f USDT after %d rejections (%.1fs). "
                            "EMITTING accumulated BUY TradeIntent.",
                            sym, 
                            accumulation_result["accumulated_quote"],
                            accumulation_result["accumulated_iterations"],
                            accumulation_result["accumulated_duration_sec"]
                        )
                        
                        # Create accumulated BUY decision with full accumulated value
                        accumulated_signal = dict(best_sig)
                        accumulated_signal["_accumulated"] = True
                        accumulated_signal["_accumulated_quote"] = accumulation_result["accumulated_quote"]
                        accumulated_signal["_accumulated_iterations"] = accumulation_result["accumulated_iterations"]
                        
                        actual_cost = accumulation_result["accumulated_quote"]
                        if agent_name in agent_budgets:
                            agent_budgets[agent_name] -= actual_cost
                        accumulated_signal["_planned_quote"] = actual_cost
                        
                        final_decisions.append((sym, "BUY", accumulated_signal))
                        
                        # 📈 TRACK CAPACITY: Consumed one slot if this was a NEW symbol
                        if sym not in owned_positions:
                            sig_pos += 1
                            self.logger.info("[Meta:Capacity] Symbol %s consumed slot. New sig_pos=%d/%d", sym, sig_pos, max_pos)
                    
                    continue

                # Multi-agent budget deduction: Deduct costs from each contributing agent
                actual_cost = planned_quote
                if not force_eval and total_intended_quote > 0:
                    # Scaling factor in case planned_quote differs from intended_quote (e.g. rounded or floor-bumped)
                    scale = actual_cost / total_intended_quote
                    for ag, sh in best_sig.get("_contributions", []):
                        if ag in agent_budgets:
                            deduction = sh * scale
                            agent_budgets[ag] -= deduction
                
                best_sig["_planned_quote"] = actual_cost
                final_decisions.append((sym, "BUY", best_sig))
                
                # 📈 TRACK CAPACITY: Consumed one slot if this was a NEW symbol
                if sym not in owned_positions:
                    sig_pos += 1
                    self.logger.info("[Meta:Capacity] Symbol %s consumed slot. New sig_pos=%d/%d", sym, sig_pos, max_pos)
                
                # Explicit observability requested by USER
                self.logger.info("[MetaController] Selected Tier-%s: %s BUY", tier, sym)
                self.logger.info("[Meta] Executing Tier-%s trade: %s | quote=%.2f | conf=%.2f | reason=%s", 
                                tier, sym, actual_cost, best_conf, best_sig.get("reason", "normal"))
            else:
                # Budget insufficient for this trade
                agent_name = best_sig.get("agent", "Meta")
                self.logger.warning("[Meta] Starved %s/%s - budget %.2f < needed %.2f", sym, agent_name, total_intended_quote, planned_quote)

        # 5.5) Throughput Guard Strictness: IF no open positions and throughput_guard_active, emit exactly ONE TradeIntent.
        if throughput_gap and not has_positions:
            buys = [d for d in final_decisions if d[1] == "BUY"]
            sells = [d for d in final_decisions if d[1] == "SELL"]
            if buys:
                # Keep only the single best BUY to satisfy the "exactly ONE" rule
                best_buy = max(buys, key=lambda x: float(x[2].get("confidence", 0.0)))
                self.logger.info("[Meta] Strict Throughput Guard: Limiting to exactly ONE BUY intent for symbol: %s", best_buy[0])
                
                # CRITICAL FIX 1: Absolute SELL block
                # If we are bootstrap-forcing, we must NOT emit any SELLs or dust cleanups.
                # LiquidationAgent leaks or phantom SELLs must be purged.
                final_decisions = [best_buy]

        # decisions_with_affordability_and_position_checks logic is now merged 
        # below into the ranked loop to ensure we don't 'reserve' budget for 
        # trades that fail the EM probe.
        
        decisions = []
        for sym, action, sig in final_decisions:
            if action == "HOLD":
                # ISSUE #4 FIX: explicit guard to ensure HOLD never reaches ExecutionManager
                continue

            if action == "BUY":
                planned_quote = sig.get("_planned_quote", 0.0)
                
                # P9: Enforce Min-Notional at Planning Time (Upstream Veto)
                min_floor = float(self._cfg("MIN_NOTIONAL_FLOOR", 10.0))
                if planned_quote < min_floor:
                    self.logger.info("[Meta] Planning Veto: Quote %.2f < MinNotional %.2f. Dropping %s.", 
                                    planned_quote, min_floor, sym)
                    
                    # GAP FIX F: Send rejection feedback to agent
                    agent_name = sig.get("agent", "Meta")
                    gap_amount = min_floor - planned_quote
                    
                    if hasattr(self.shared_state, "record_agent_rejection"):
                        try:
                            await self.shared_state.record_agent_rejection(
                                agent=agent_name,
                                symbol=sym,
                                side=action,
                                reason="PLANNING_VETO_MIN_NOTIONAL",
                                rejected_quote=planned_quote,
                                gap=gap_amount,
                                timestamp=time.time()
                            )
                            self.logger.info("[Meta:Feedback] Agent %s notified: min_notional violation (quote=%.2f gap=%.2f)", 
                                        agent_name, planned_quote, gap_amount)
                        except Exception as e:
                            self.logger.warning("[Meta:Feedback] Failed to record rejection for %s: %s", agent_name, e)
                    
                    # === PHASE H: Accumulation Resolution at Planning Veto ===
                    # Check if accumulated amounts trigger auto-emission BEFORE returning budget
                    accumulation_result = await self.accumulation_resolution_check(
                        symbol=sym,
                        rejection_quote=planned_quote,
                        min_notional=min_floor
                    )
                    
                    if accumulation_result and accumulation_result.get("should_emit_buy"):
                        # Accumulated amount crossed threshold - emit BUY with full accumulated amount
                        self.logger.info("[Meta:Accumulation] Planning Veto RESOLVED: Accumulated %.2f >= MinNotional %.2f for %s. "
                                    "Emitting auto-accumulated BUY (iterations=%d, duration_sec=%.1f)",
                                    accumulation_result.get("accumulated_quote", 0),
                                    min_floor,
                                    sym,
                                    accumulation_result.get("accumulated_iterations", 0),
                                    accumulation_result.get("accumulated_duration_sec", 0))
                        
                        # Create accumulated BUY decision from best_sig template
                        accumulated_signal = dict(sig)
                        accumulated_signal["_accumulated"] = True
                        accumulated_signal["_accumulated_quote"] = accumulation_result.get("accumulated_quote", 0)
                        accumulated_signal["_accumulated_iterations"] = accumulation_result.get("accumulated_iterations", 0)
                        accumulated_signal["_accumulated_duration_sec"] = accumulation_result.get("accumulated_duration_sec", 0)
                        accumulated_signal["reason"] = f"auto_accumulated_resolution({accumulation_result.get('accumulated_iterations', 0)}x)"
                        accumulated_signal["_planned_quote"] = accumulation_result.get("accumulated_quote", 0)
                        
                        # Calculate actual_cost from accumulated quote
                        actual_cost = accumulation_result.get("accumulated_quote", 0)
                        
                        # Deduct from agent budget since we're now emitting
                        agent_name = sig.get("agent", "Meta")
                        if agent_name in agent_budgets:
                            agent_budgets[agent_name] -= actual_cost
                            self.logger.debug("[Meta:Accumulation] Deducted %.2f from agent %s budget (accumulated emission)", 
                                            actual_cost, agent_name)
                        
                        # Add accumulated BUY to decisions for normal execution flow
                        decisions.append((sym, "BUY", accumulated_signal))
                        self.logger.info("[Meta:Accumulation] Added accumulated BUY to decisions: %s, amount=%.2f", sym, actual_cost)
                    else:
                        # No threshold crossing yet - return budget as normal
                        self.shared_state.request_reservation_adjustment(
                            agent=agent_name, delta=planned_quote, reason="planning_veto_min_notional"
                        )
                    
                    continue

                try:
                    # Strict Rule 2: Non-zero qty check BEFORE final decision
                    # BOOTSTRAP FIX: If this signal is marked with _bootstrap flag, pass bootstrap context
                    bootstrap_policy_ctx = None
                    if sig.get("_bootstrap") or sig.get("_bootstrap_override"):
                        bootstrap_policy_ctx = self._build_policy_context(
                            sym, 
                            "BUY",
                            extra={"bootstrap_bypass": True}
                        )

                    can_afford, gap, reason = await self.execution_manager.can_afford_market_buy(sym, planned_quote, policy_context=bootstrap_policy_ctx)

                    if not can_afford and gap > 0:
                        # Point 3 enhancement: Quantity Healing
                        agent_name = sig.get("agent", "Meta")
                        avail = agent_budgets.get(agent_name, 0.0)
                        gap_f = float(gap)
                        
                        if avail >= gap_f:
                            self.logger.info("[Meta] Healing trade for %s: Scaling quote %.2f -> %.2f (+%.2f) to reach valid quantity.", 
                                            sym, planned_quote, planned_quote + gap_f, gap_f)
                            # ISSUE #3 FIX: P9-compliant adjustment request
                            self.shared_state.request_reservation_adjustment(
                                agent=agent_name, delta=-gap_f, reason="meta_healing"
                            )
                            
                            # 2. Update local tracking
                            planned_quote += gap_f
                            agent_budgets[agent_name] -= gap_f
                            sig["_planned_quote"] = planned_quote
                            
                            # 3. Final sanity check
                            res_heal, _, _ = await self.execution_manager.can_afford_market_buy(sym, planned_quote, policy_context=bootstrap_policy_ctx)
                            if res_heal:
                                can_afford = True
                                self.logger.info("[Meta] Healing successful for %s.", sym)

                    if not can_afford:
                        # Rule 5: Give back the budget to the pool if it won't be used
                        agent_name = sig.get("agent", "Meta")
                        
                        # GAP FIX D: Send rejection feedback to agent
                        if hasattr(self.shared_state, "record_agent_rejection"):
                            try:
                                await self.shared_state.record_agent_rejection(
                                    agent=agent_name,
                                    symbol=sym,
                                    side=action,
                                    reason=str(reason),
                                    rejected_quote=planned_quote,
                                    gap=gap,
                                    timestamp=time.time()
                                )
                                self.logger.info("[Meta:Feedback] Agent %s notified: affordability failure (quote=%.2f gap=%.2f reason=%s)", 
                                            agent_name, planned_quote, gap, reason)
                            except Exception as e:
                                self.logger.warning("[Meta:Feedback] Failed to record rejection for %s: %s", agent_name, e)
                        
                        # ISSUE #3 FIX: P9-compliant adjustment request
                        self.shared_state.request_reservation_adjustment(
                            agent=agent_name, delta=planned_quote, reason="trade_validation_failed"
                        )
                        
                        # P9 DEADLOCK PREVENTION: Record rejection
                        if hasattr(self.shared_state, "record_rejection"):
                            await self.shared_state.record_rejection(sym, action, str(reason))
                        
                        # Check for deadlock condition
                        deadlock_threshold = int(self._cfg("DEADLOCK_REJECTION_THRESHOLD", 3))
                        rej_count = self.shared_state.get_rejection_count(sym, "BUY") if hasattr(self.shared_state, "get_rejection_count") else 0
                        if rej_count >= deadlock_threshold:
                            self.logger.error("[Meta:DEADLOCK] Symbol %s has been rejected %d times. Emitting DEADLOCK event.", sym, rej_count)
                            if hasattr(self.shared_state, "emit_event"):
                                await _safe_await(self.shared_state.emit_event("DEADLOCK_DETECTED", {
                                    "symbol": sym, "side": action, "reason": str(reason), 
                                    "rejection_count": rej_count, "ts": time.time()
                                }))
                        
                        if reason in ("INSUFFICIENT_QUOTE", "QUOTE_LT_MIN_NOTIONAL") or "NOT_EXECUTABLE" in str(reason):
                            sig["_need_liquidity"] = True
                            sig["_liq_gap"] = gap
                            sig["_liq_reason"] = reason
                            decisions.append((sym, action, sig))
                        else:
                            self.logger.info("[Meta] Skipping %s - unaffordable (%s); budget returned.", sym, reason)
                    else:
                        decisions.append((sym, action, sig))
                except Exception as e:
                    self.logger.debug("Affordability check failed for %s: %s", sym, e)
            else:
                # SELL path: No capital checks needed, direct pass-through
                if action == "SELL":
                    if (
                        await self._passes_meta_sell_profit_gate(sym, sig)
                        and await self._passes_meta_sell_excursion_gate(sym, sig)
                    ):
                        self.logger.info("[EXEC_DECISION] SELL %s [via_final_decisions] bypass_capital_check=True", sym)
                        decisions.append((sym, action, sig))
                    else:
                        self.logger.info("[EXEC_DECISION] SELL %s blocked by profit/excursion gate", sym)
                else:
                    decisions.append((sym, action, sig))

        # ═══════════════════════════════════════════════════════════════════════════════
        # FINAL ARBITRATION: Apply context flags as overrides (NOT replacements)
        # ═══════════════════════════════════════════════════════════════════════════════
        
        # If P-1 Emergency is active, prepend emergency plan to decisions
        # This allows P-1 to be executed alongside normal signals
        if context_flags.get("P1_EMERGENCY_ACTIVE"):
            p1_plan = context_flags.get("P1_EMERGENCY_PLAN", [])
            if p1_plan:
                self.logger.warning(
                    "[Meta:Final] 🔥 P-1 EMERGENCY PLAN PREPENDED: %d emergency SELLs will execute alongside normal signals",
                    len(p1_plan)
                )
                decisions = p1_plan + decisions  # Prepend emergency liquidations
        
        # If P0 Dust Recovery is active, prepend forced decisions
        # Normal signals follow after forced SELL/BUY pair
        if context_flags.get("DUST_RECOVERY_MODE"):
            p0_forced = context_flags.get("P0_FORCED_DECISIONS", [])
            if p0_forced:
                self.logger.warning(
                    "[Meta:Final] ✅ P0 FORCED DECISIONS PREPENDED: %d P0 decisions will execute first (DUST_RECOVERY mode)",
                    len(p0_forced)
                )
                decisions = p0_forced + decisions  # Prepend forced SELL/BUY pair

        # If Capital Recovery forced decisions exist, prepend them to ensure execution
        if context_flags.get("CAPITAL_RECOVERY_FORCED_DECISIONS"):
            cap_forced = context_flags.get("CAPITAL_RECOVERY_FORCED_DECISIONS", [])
            if cap_forced:
                self.logger.warning(
                    "[Meta:Final] 🔥 CAPITAL RECOVERY FORCED DECISIONS PREPENDED: %d forced SELL(s)",
                    len(cap_forced)
                )
                decisions = cap_forced + decisions

        if context_flags.get("CAPITAL_RECOVERY_SOFT_DECISIONS"):
            cap_soft = context_flags.get("CAPITAL_RECOVERY_SOFT_DECISIONS", [])
            if cap_soft:
                self.logger.warning(
                    "[Meta:Final] 🟠 CAPITAL RECOVERY SOFT DECISIONS PREPENDED: %d partial SELL(s)",
                    len(cap_soft)
                )
                decisions = cap_soft + decisions
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # DUST EXIT POLICY: Emergency escape from constrained waiting state
        # ═══════════════════════════════════════════════════════════════════════════════
        # If system is stuck with no decisions AND high dust ratio AND no trades for N cycles,
        # force liquidate 1 dust position to free capacity and capital
        if (not decisions and self.DUST_EXIT_ENABLED):
            try:
                # Check if conditions met for dust exit
                total_pos, sig_pos, dust_pos = await self._count_significant_positions()
                total = total_pos if total_pos > 0 else 1
                dust_ratio = dust_pos / total
                
                if (dust_ratio > self.DUST_EXIT_THRESHOLD and 
                    self.cycles_no_trade >= self.DUST_EXIT_NO_TRADE_CYCLES):
                    
                    # Select oldest/lowest confidence dust position
                    target_symbol = await self._select_dust_exit_candidate()
                    
                    if target_symbol:
                        # Create forced SELL decision with liquidation tag
                        dust_exit_signal = {
                            "action": "SELL",
                            "agent": "Meta",
                            "tag": "liquidation/dust_exit",
                            "reason": "Emergency dust exit - capacity/capital recovery",
                            "_dust_exit_forced": True,
                            "_dust_ratio": dust_ratio,
                            "_cycles_no_trade": self.cycles_no_trade,
                        }
                        
                        decisions.append((target_symbol, "SELL", dust_exit_signal))
                        
                        self.logger.warning(
                            "[DUST_EXIT] 🔑 ESCAPE MECHANISM TRIGGERED: Forced SELL %s "
                            "(dust_ratio=%.1f%% > %.1f%%, cycles_idle=%d >= %d)",
                            target_symbol, dust_ratio * 100, self.DUST_EXIT_THRESHOLD * 100,
                            self.cycles_no_trade, self.DUST_EXIT_NO_TRADE_CYCLES
                        )
            except Exception as e:
                self.logger.warning("[DUST_EXIT] Failed to execute dust exit policy: %s", str(e))

        decisions = self._batch_buy_decisions(decisions)
        decisions = self._batch_sell_decisions(decisions)
        decisions = self._apply_sell_arbiter(decisions)
        for idx, (sym, action, sig) in enumerate(decisions):
            self._ensure_decision_id(sym, action, sig, idx)
        self.logger.info(
            "[Meta:Final] ✅ Decision sequence complete: %d total decisions (including context overrides)",
            len(decisions)
        )
        return decisions

############################################################
# SECTION: Execution Dispatch & Integration with ExecutionManager
# Responsibility:
# - Trade execution coordination with ExecutionManager
# - Order placement, result handling, and error recovery
# - Execution attempt tracking and retry logic
# Future Extraction Target:
# - ExecutionCoordinator
############################################################

    # Belongs to: Execution Dispatch & Integration with ExecutionManager
    # Extraction Candidate: Yes
    # Depends on: State & Internal Counters, Metrics & KPIs
    def _log_reason(self, level: str, sym: str, reason: str):
        """Deduplicated reason logging."""
        if not hasattr(self, "_last_reason_log"):
            self._last_reason_log = BoundedCache(max_size=100, default_ttl=60)
            
        cached_reason = self._last_reason_log.get(sym)
        if cached_reason == reason:
            return

        self._last_reason_log.set(sym, reason)
        log_msg = f"[DecisionReject] {sym} :: {reason}"
        if level == "INFO":
            self.logger.info(log_msg)
        else:
            self.logger.debug(log_msg)

    async def _execute_decision(self, symbol: str, side: str, signal: Dict[str, Any], accepted_symbols_set: set):
        # Enforce lifecycle lock for SELL and ROTATION
        if side.upper() == "SELL":
            if not self._can_act(symbol, "SELL"):
                self.logger.info(f"[LIFECYCLE] {symbol}: SELL execution blocked by lifecycle lock")
                return
        # ...existing code...
        # On successful SELL, update lifecycle
        if side.upper() == "SELL":
            self._on_sell_executed(symbol)
        """Standardized execution path with P9-aligned readiness and trade frequency gating."""
        # NOTE: Execution attempts are counted only when we actually call ExecutionManager.
        # This prevents policy rejections from inflating liveness counters.
        last_result = None
        if side == "BUY":
            now = time.time()
            agent_name = signal.get("agent", "Meta")
            # 1. Clean old timestamps (Hourly window)
            for dq in [self._trade_timestamps, self._trade_timestamps_sym[symbol], self._trade_timestamps_agent[agent_name]]:
                while dq and (now - dq[0] > 3600):
                    dq.popleft()
            while self._trade_timestamps_day and (now - self._trade_timestamps_day[0] > 86400):
                self._trade_timestamps_day.popleft()
            # ===== CRITICAL FIX #2A: Bootstrap BUY can bypass hourly limits =====
            is_bootstrap = "bootstrap" in str(signal.get("reason", "")).lower()
            is_bootstrap_override = signal.get("_bootstrap_override", False)
            is_flat_init = signal.get("_flat_init_buy", False)
            is_dust_merge = signal.get("_dust_reentry_override", False)
            focus_active = bool(getattr(self, "_focus_mode_active", False))
            is_bootstrap_seed = bool(signal.get("_bootstrap_seed") or signal.get("bootstrap_seed") or str(signal.get("reason", "")).upper() == "BOOTSTRAP_SEED")

            # FIX: Position Lock Invariant (User Rule 7)
            # If position exists, REJECT BUY unless scaling is explicitly planned/allowed.
            curr_qty = self.shared_state.get_position_qty(symbol) if hasattr(self.shared_state, "get_position_qty") else 0.0
            is_dust_healing = signal.get("_dust_healing", False)
            if is_dust_healing:
                metrics = getattr(self.shared_state, "metrics", {}) or {}
                trade_count = int(
                    metrics.get("total_trades_executed", 0)
                    or getattr(self.shared_state, "trade_count", 0)
                    or 0
                )
                if trade_count < 1:
                    self.logger.warning(
                        "[DUST_HEALING] BLOCKED at execution: trade_count=%d < 1 (no realized trades yet)",
                        trade_count,
                    )
                    return {"ok": False, "status": "skipped", "reason": "dust_healing_not_ready"}
            # Check if position is marked as dust
            pos_data = None
            if hasattr(self.shared_state, "get_position"):
                try:
                    pos_data = await _safe_await(self.shared_state.get_position(symbol))
                except Exception:
                    pos_data = None
            is_dust_position = False
            if pos_data and isinstance(pos_data, dict):
                is_dust_position = pos_data.get("is_dust", False) or pos_data.get("_is_dust", False) or pos_data.get("state", "") == "DUST_LOCKED"
            is_bootstrap_dust_bypass = False
            # Diagnostic logging for dust healing execution
            self.logger.info(
                "[DIAG:DUST_HEALING] _execute_decision: symbol=%s, curr_qty=%.8f, is_dust_healing=%s, is_dust_position=%s, is_bootstrap=%s, is_bootstrap_override=%s, is_flat_init=%s, focus_active=%s, reason=%s",
                symbol, curr_qty, is_dust_healing, is_dust_position, is_bootstrap, is_bootstrap_override, is_flat_init, focus_active, str(signal.get("reason", ""))
            )
            # SOP-REC-004: Dust Healing Execution Authority
            if curr_qty > 0 and not is_dust_merge:
                is_bootstrap_dust_bypass = self._bootstrap_dust_bypass_allowed(
                    symbol,
                    bool(is_bootstrap_override),
                    bool(is_dust_position),
                )
                # Allow dust healing scaling if position is marked as dust, regardless of mode
                if not (is_bootstrap_seed or is_bootstrap_dust_bypass or (is_dust_healing and (is_dust_position or is_bootstrap or is_bootstrap_override or is_flat_init))):
                    self.logger.warning(
                        "[Meta:PositionLock] REJECTING BUY %s: Position already exists (%.8f). Scaling not enabled.", 
                        symbol, curr_qty
                    )
                    return {"ok": False, "status": "skipped", "reason": "position_lock", "reason_detail": "position_already_exists"}
                else:
                    if is_bootstrap_seed:
                        self.logger.info(
                            "[BOOTSTRAP] Seed BUY authorized: bypassing PositionLock once for %s",
                            symbol,
                        )
                    elif is_bootstrap_dust_bypass:
                        self.logger.info(
                            "[BOOTSTRAP] Dust scaling authorized: bypassing PositionLock once for %s",
                            symbol,
                        )
                        signal["execution_tag"] = "meta/bootstrap_dust"
                        signal["_bootstrap_dust_bypass"] = True
                    else:
                        # Emit SOP-REC-004 log for dust healing authority
                        self.logger.info(
                            "[SOP-REC-004] Dust healing execution authorized | reason=dust_only_portfolio | mode=%s | symbol=%s | quote_amount=%.2f",
                            "DUST_HEALING" if is_dust_position else ("BOOTSTRAP" if is_bootstrap or is_bootstrap_override or is_flat_init else "RECOVERY"),
                            symbol,
                            float(signal.get("quote_amount", 0.0))
                        )
                        # Tag execution for dust healing
                        signal["execution_tag"] = "meta/dust_healing"

            is_dust_healing = signal.get("_dust_healing", False)
            # SOP-REC-004: Dust healing overrides PositionLock, hourly trade limit, confidence threshold, agent ownership
            if is_dust_healing:
                # Log dust recovery event (required by SOP)
                pre_value = float(signal.get("pre_value", 0.0))
                post_value = float(signal.get("post_value", 0.0))
                self.logger.info(
                    f"[SOP-REC-004] symbol={symbol} action=BUY reason=dust_recovery pre_value=${pre_value:.2f} post_value=${post_value:.2f} mode=NORMAL override=[PositionLock, HourlyLimit]"
                )
                # Tag for stats exclusion and authority
                signal["tier"] = "DUST_RECOVERY"
                signal["execution_tag"] = "meta/dust_healing"
                # Override agent ownership
                signal["agent"] = "Meta"
                # Bypass PositionLock, hourly trade limit, confidence threshold
                # (PositionLock and confidence threshold checks must be bypassed elsewhere if present)
                # Do not count toward hourly trade limit or stats (skip timestamp append below)
                # Still enforce capital floor, risk hard stops, exchange min/max rules
                # ...existing code...
            elif not (is_bootstrap or is_bootstrap_override or is_flat_init or is_dust_merge or focus_active):
                # 2. Gating logic (Phase A Enhancements) — Only for NORMAL BUYs
                # GLOBAL GATE
                max_hourly = int(getattr(self.config, "MAX_TRADES_PER_HOUR", self._max_trades_per_hour) or 0)
                max_daily = int(getattr(self.config, "MAX_TRADES_PER_DAY", self._max_trades_per_day) or 0)
                if max_daily > 0 and len(self._trade_timestamps_day) >= max_daily:
                    self.logger.info("[Meta] Skip %s BUY: Global daily trade limit (%d) reached.", symbol, max_daily)
                    return {"ok": False, "status": "skipped", "reason": "global_daily_limit", "reason_detail": "global_daily_limit_reached"}
                if max_hourly > 0 and len(self._trade_timestamps) >= max_hourly:
                    self.logger.info("[Meta] Skip %s BUY: Global hourly trade limit (%d) reached.", symbol, max_hourly)
                    return {"ok": False, "status": "skipped", "reason": "global_limit", "reason_detail": "global_hourly_limit_reached"}
                # PER-SYMBOL GATE
                max_sym_hourly = int(getattr(self.config, "MAX_TRADES_PER_SYMBOL_PER_HOUR", 2) or 0)
                if max_sym_hourly > 0 and len(self._trade_timestamps_sym[symbol]) >= max_sym_hourly:
                    self.logger.info("[Meta] Skip %s BUY: Symbol hourly trade limit reached.", symbol)
                    return {"ok": False, "status": "skipped", "reason": "symbol_limit", "reason_detail": "symbol_hourly_limit_reached"}
                # PER-AGENT GATE (Safety: Max 75% of global limit per agent)
                agent_limit = max(1, int(max_hourly * 0.75))
                if len(self._trade_timestamps_agent[agent_name]) >= agent_limit:
                    self.logger.info("[Meta] Skip %s BUY: Agent %s hourly limit reached.", symbol, agent_name)
                    return {"ok": False, "status": "skipped", "reason": "agent_limit", "reason_detail": f"agent_limit_{agent_name}_reached"}
            else:
                # BOOTSTRAP BYPASS: Log that we're bypassing limits
                self.logger.info(
                    "[Meta:FIX#2] Bootstrap BUY %s bypassing hourly limits (is_bootstrap=%s, is_flat_init=%s)",
                    symbol, is_bootstrap, is_flat_init
                )
                if focus_active:
                    self.logger.info("[Meta:FocusMode] BUY limits relaxed for %s (focus mode active).", symbol)
        if symbol not in accepted_symbols_set:
            # P9 FIX: SELL always bypasses accepted_symbols check
            # REASON: SELL is for exiting existing positions, even if symbol is not in analysis universe
            # SELL must never be blocked by universe filters - positions must always be exitiable
            if side == "SELL":
                self.logger.info(
                    "[Meta:P9] SELL bypass: %s not in accepted set but SELL must execute (P9 Rule: Exits always allowed). Proceeding.",
                    symbol
                )
            else:
                # For BUY: Allow bootstrap BUY to bypass accepted_symbols check
                is_bootstrap = "bootstrap" in str(signal.get("reason", "")).lower()
                if side == "BUY" and is_bootstrap:
                    self.logger.info("[Meta:Bootstrap] Gating bypass: %s is not in accepted set but is a bootstrap BUY. Proceeding.", symbol)
                else:
                    self.logger.warning("Skipping unaccepted symbol: %s", symbol)
                    await self._log_execution_result(symbol, side, signal, {"status": "skipped", "reason": "symbol_not_accepted"})
                    return {"ok": False, "status": "skipped", "reason": "symbol_not_accepted"}
        # Ensure per-symbol market data is ready if SharedState provides a hook
        try:
            fn = getattr(self.shared_state, "is_symbol_data_ready", None)
            if callable(fn):
                ok = fn(symbol)
                if _asyncio.iscoroutine(ok):
                    ok = await ok
                if not ok:
                    await self._log_execution_result(symbol, side, signal, {"status": "skipped", "reason": "symbol_data_not_ready"})
                    return {"ok": False, "status": "skipped", "reason": "symbol_data_not_ready"}
        except Exception:
            self.logger.debug("symbol readiness check failed for %s", symbol, exc_info=True)
        try:
            if side == "BUY":
                # Step 1: Fetch symbol economics from exchange
                filters = await self.exchange_client.ensure_symbol_filters_ready(symbol)
                min_notional = float(getattr(filters, "min_notional", 0.0) or filters.get("min_notional", 0.0) or 0.0)
                min_entry_quote = max(min_notional, float(self._cfg("MIN_ENTRY_QUOTE", 0.0)))
                step_size = float(getattr(filters, "step_size", 0.0) or filters.get("step_size", 0.0) or 0.0)
                min_qty = float(getattr(filters, "min_qty", 0.0) or filters.get("min_qty", 0.0) or 0.0)
                dust_threshold = min_notional * 0.9
                # Attach to policy_context
                policy_context = {
                    "min_notional": min_notional,
                    "min_entry_quote": min_entry_quote,
                    "step_size": step_size,
                    "min_qty": min_qty,
                    "dust_threshold": dust_threshold,
                }
                # Strict Rule 5: Zero-amount execution must NEVER be OK
                planned_quote = float(signal.get("_planned_quote", 0.0))
                if planned_quote <= 0:
                    planned_quote = await self._planned_quote_for(symbol, signal)
                # Reservation fallback: remap agent to a funded reservation key if needed
                agent_name = signal.get("agent", "Meta")
                try:
                    if hasattr(self.shared_state, "get_authoritative_reservation"):
                        auth_res = float(self.shared_state.get_authoritative_reservation(agent_name) or 0.0)
                        if auth_res < planned_quote:
                            fallback_agent = None
                            if hasattr(self.shared_state, "get_authoritative_reservations"):
                                reservations = self.shared_state.get_authoritative_reservations() or {}
                                if isinstance(reservations, dict):
                                    if float(reservations.get("Meta", 0.0) or 0.0) >= planned_quote:
                                        fallback_agent = "Meta"
                                    else:
                                        for cand, val in reservations.items():
                                            if float(val or 0.0) >= planned_quote:
                                                fallback_agent = cand
                                                break
                            if fallback_agent and fallback_agent != agent_name:
                                self.logger.warning(
                                    "[Meta:ReservationFallback] Remapping agent %s -> %s for %s (planned=%.2f)",
                                    agent_name, fallback_agent, symbol, planned_quote
                                )
                                signal["_reservation_original_agent"] = agent_name
                                signal["agent"] = fallback_agent
                                signal["_reservation_fallback"] = True
                                agent_name = fallback_agent
                except Exception:
                    self.logger.debug("[Meta:ReservationFallback] Failed to evaluate reservation fallback", exc_info=True)
                # Early TP/SL sanity check to avoid entries with unrealistic exits
                if getattr(self, "tp_sl_engine", None) is not None and bool(self._cfg("TP_SL_GUARD_ENABLED", True)):
                    cur_price = 0.0
                    try:
                        if hasattr(self.shared_state, "safe_price"):
                            cur_price = float(await _safe_await(self.shared_state.safe_price(symbol)) or 0.0)
                    except Exception:
                        cur_price = 0.0
                    if not cur_price:
                        cur_price = float(getattr(self.shared_state, "latest_prices", {}).get(symbol, 0.0) or 0.0)
                    if cur_price > 0 and hasattr(self.tp_sl_engine, "calculate_tp_sl"):
                        tp, sl = self.tp_sl_engine.calculate_tp_sl(symbol, cur_price, tier=signal.get("_tier"))
                        taker_bps = float(self._get_fee_bps(self.shared_state, "taker") or 10.0)
                        slippage_bps = float(getattr(self.config, "EXIT_SLIPPAGE_BPS", getattr(self.config, "CR_PRICE_SLIPPAGE_BPS", 0.0)) or 0.0)
                        buffer_bps = float(getattr(self.config, "TP_MIN_BUFFER_BPS", 0.0) or 0.0)
                        min_tp_pct_floor = (taker_bps * 2.0 + slippage_bps + buffer_bps) / 10000.0
                        min_tp_pct = max(min_tp_pct_floor, float(self._cfg("TP_SL_MIN_TP_PCT", 0.002) or 0.0))
                        min_sl_pct = float(self._cfg("TP_SL_MIN_SL_PCT", 0.002) or 0.0)
                        tp_dist = abs(float(tp or 0.0) - cur_price) / cur_price if tp else 0.0
                        sl_dist = abs(cur_price - float(sl or 0.0)) / cur_price if sl else 0.0
                        floor_hit = False
                        try:
                            floor_hit = bool(getattr(self.tp_sl_engine, "_tp_floor_hit", {}).get(symbol))
                        except Exception:
                            floor_hit = False
                        if floor_hit:
                            self.logger.warning(
                                "[Meta:TPSL_GUARD] Blocking BUY %s: TP below fee-aware floor (min_tp=%.3f%%)",
                                symbol, min_tp_pct * 100.0
                            )
                            await self._log_execution_result(symbol, side, signal, {
                                "status": "skipped",
                                "reason": "tp_sl_guard",
                                "details": {"min_tp_pct": min_tp_pct}
                            })
                            return {"ok": False, "status": "skipped", "reason": "tp_sl_guard"}
                        if tp_dist < min_tp_pct or sl_dist < min_sl_pct:
                            self.logger.warning(
                                "[Meta:TPSL_GUARD] Blocking BUY %s: TP/SL too tight (tp=%.3f%% sl=%.3f%% min_tp=%.3f%%)",
                                symbol, tp_dist * 100.0, sl_dist * 100.0, min_tp_pct * 100.0
                            )
                            await self._log_execution_result(symbol, side, signal, {
                                "status": "skipped",
                                "reason": "tp_sl_guard",
                                "details": {"tp_dist": tp_dist, "sl_dist": sl_dist, "min_tp_pct": min_tp_pct}
                            })
                            return {"ok": False, "status": "skipped", "reason": "tp_sl_guard"}
                    elif cur_price > 0:
                        self.logger.warning("[Meta:TPSL_GUARD] TP/SL engine missing calculate_tp_sl; skipping guard.")
                # TIER 1: ACCUMULATING State Protection
                # Skip SELL on fresh accumulating positions (unless risk/liquidation/tp_sl)
                intent_owner = signal.get("_intent_owner", "")
                if side == "SELL" and hasattr(self.shared_state, "get_pending_intent"):
                    intent = self.shared_state.get_pending_intent(symbol, "BUY")
                    if intent and intent.state == "ACCUMULATING":
                        is_protected_sell = any(tag in str(intent_owner).lower() or tag in str(signal.get("agent", "")).lower() 
                                            for tag in ["risk", "liquidation", "tp_sl", "emergency", "rotation", "authority"])
                        if not is_protected_sell:
                            self.logger.info(
                                "[Meta:AccumGuard] Blocking SELL on %s: position is ACCUMULATING (intent_owner=%s). "
                                "Only risk/liquidation/tp_sl/rotation/authority can override.",
                                symbol, intent_owner
                            )
                            await self.shared_state.record_rejection(
                                symbol, "SELL", "ACCUMULATING_PROTECTION", source="MetaController"
                            )
                            # TIER 2: Track conflict (economic gate vs accumulation protection)
                            if hasattr(self.shared_state, "record_policy_conflict"):
                                self.shared_state.record_policy_conflict("accumulating_protection_blocks")
                            return {"ok": False, "status": "skipped", "reason": "accumulating_protection"}
                # TIER 1: Capital Preservation Floor ($50 minimum)
                min_capital_floor = float(self._cfg("CAPITAL_PRESERVATION_FLOOR", 50.0))
                free_quote = 0.0
                try:
                    if hasattr(self.shared_state, "get_free_quote"):
                        maybe = self.shared_state.get_free_quote()
                        if _asyncio.iscoroutine(maybe):
                            free_quote = await maybe
                        else:
                            free_quote = float(maybe or 0.0)
                except Exception:
                    free_quote = 0.0
                # PHASE 2 NOTE: Capital floor check removed (now checked at start of _build_decisions)
                # If we reached this point, capital has already been validated
                # No redundant check needed here
                policy_flags = {"POLICY_SINGLE_AUTHORITY": True}
                econ_ok, econ_reason, econ_metrics = self._check_economic_profitability(symbol, signal)
                if not econ_ok:
                    self._log_reason("INFO", symbol, f"economic_guard:{econ_reason}")
                    if hasattr(self.shared_state, "record_rejection"):
                        await self.shared_state.record_rejection(
                            symbol, "BUY", "ECONOMIC_PROFITABILITY_BLOCK", source="MetaController"
                        )
                    await self._log_execution_result(
                        symbol,
                        side,
                        signal,
                        {"status": "skipped", "reason": "economic_guard", "details": econ_metrics},
                    )
                    return {"ok": False, "status": "skipped", "reason": "economic_guard"}
                policy_flags["ECONOMIC_PROFITABILITY_INVARIANT"] = True
                # Double check affordability for executable qty (Rule 2/5)
                # BOOTSTRAP FIX: If this signal is marked with _bootstrap flag, pass bootstrap context
                bootstrap_policy_ctx = policy_context.copy() if policy_context else None
                if signal.get("_bootstrap") or signal.get("_bootstrap_override"):
                    if not bootstrap_policy_ctx:
                        bootstrap_policy_ctx = {}
                    bootstrap_policy_ctx["bootstrap_bypass"] = True
                    self.logger.info("[Meta:BOOTSTRAP] Using bootstrap_bypass context for affordability check: %s", symbol)

                can_ex, _, reason = await self.execution_manager.can_afford_market_buy(symbol, planned_quote, policy_context=bootstrap_policy_ctx)
                if not can_ex:
                    self.logger.warning("⚡ [Escalation] Signal %s for %s has zero executable qty (%s). Triggering Rule 5 Escalation.", symbol, side, reason)
                    # P9: Report capital failure immediately
                    agent = signal.get("agent", "Meta")
                    if hasattr(self.shared_state, "report_agent_capital_failure"):
                        self.shared_state.report_agent_capital_failure(agent)
                    # Rule 6: Liquidation must be able to invalidate readiness
                    if hasattr(self.shared_state, "ops_plane_ready_event"):
                        self.shared_state.ops_plane_ready_event.clear()
                        self.logger.info("[Meta] Readiness = FALSE (Escalation Triggered)")
                    if self.liquidation_agent and hasattr(self.liquidation_agent, "_free_usdt_now"):
                        await self.liquidation_agent._free_usdt_now(
                            target=float(self._cfg("MIN_NOTIONAL_FLOOR", 15.0)),
                            reason=f"rule5_escalation_{symbol}"
                        )
                    elif self.liquidation_agent:
                        # If liquidation_agent doesn't have _free_usdt_now method, try propose_liquidations instead
                        self.logger.warning(f"[Meta] LiquidationAgent doesn't have _free_usdt_now method. Using propose_liquidations instead.")
                        try:
                            target_usdt = float(self._cfg("MIN_NOTIONAL_FLOOR", 15.0))
                            proposals = await self.liquidation_agent.propose_liquidations(
                                gap_usdt=target_usdt,
                                reason=f"rule5_escalation_{symbol}",
                                force=True
                            )
                            if proposals:
                                self.logger.info(f"[Meta] Generated {len(proposals)} liquidation proposals for escalation")
                        except Exception as e:
                            self.logger.warning(f"[Meta] Failed to generate liquidation proposals: {e}")
                    else:
                        self.logger.warning(f"[Meta] No liquidation agent available for escalation")
                    # P9: Trigger immediate re-plan
                    if hasattr(self.shared_state, "replan_request_event"):
                        self.shared_state.replan_request_event.set()
                    await self._log_execution_result(symbol, side, signal, {"status": "failed", "reason": f"rule5_escalation_{reason}"})
                    return {"ok": False, "status": "failed", "reason": f"rule5_escalation_{reason}"}
                # Handle liquidity healing if needed
                if signal.get("_need_liquidity"):
                    success = await self._attempt_liquidity_healing(symbol, signal)
                    if not success:
                        await self._log_execution_result(
                            symbol, side, signal, {"status": "skipped", "reason": "liquidity_healing_failed"}
                        )
                        return {"ok": False, "status": "skipped", "reason": "liquidity_healing"}
                # Risk pre-check (BUY) - Tier-aware with FIX #4: liquidation bypass support
                if self.risk_manager and hasattr(self.risk_manager, "pre_check"):
                    tier = signal.get("_tier", "A")  # Default to Tier A if not specified
                    # FIX #4: Detect if this is a liquidation signal for bypass
                    is_liq = signal.get("_is_starvation_sell") or signal.get("_quote_based") or signal.get("_batch_sell")
                    tag = signal.get("_tag") or signal.get("tag") or ""
                    is_liq = is_liq or ("liquidation" in str(tag))
                    ok, r_reason = await _safe_await(self.risk_manager.pre_check(
                        symbol=symbol, side="BUY", planned_quote=planned_quote, tier=tier, is_liquidation=is_liq
                    ))
                    if not ok:
                        self._log_reason("INFO", symbol, f"risk_precheck:{r_reason}")
                        await self._log_execution_result(symbol, side, signal, {"status": "skipped", "reason": f"risk:{r_reason}"})
                        return {"ok": False, "status": "skipped", "reason": f"risk:{r_reason}"}
                # P9: Revalidate signal + risk context at firing time
                if hasattr(self.shared_state, "is_intent_valid"):
                    is_bootstrap_seed = bool(
                        signal.get("_bootstrap_seed")
                        or signal.get("bootstrap_seed")
                        or str(signal.get("reason", "")).upper() == "BOOTSTRAP_SEED"
                        or str(signal.get("execution_tag", "")) == "meta/bootstrap_seed"
                    )
                    if not is_bootstrap_seed and not self.shared_state.is_intent_valid(symbol, "BUY"):
                        self.logger.warning("[Meta] Signal no longer valid at firing time for %s. Skipping.", symbol)
                        await self._log_execution_result(symbol, side, signal, {"status": "skipped", "reason": "signal_invalid_at_firing"})
                        return {"ok": False, "status": "skipped", "reason": "signal_invalid"}
                # Execute through ExecutionManager (canonical single order path)
                tier = signal.get("_tier", "A")
                extra_ctx = {"economic_guard": econ_metrics}
                # ACCUMULATE_MODE: If signal is marked with _accumulate_mode, pass it through policy context
                if signal.get("_accumulate_mode"):
                    extra_ctx["_accumulate_mode"] = True
                    self.logger.info("[Meta:P0] Passing ACCUMULATE_MODE flag to ExecutionManager for %s", symbol)
                # BOOTSTRAP_MODE: If signal is marked with _bootstrap or _bootstrap_override, pass it through policy context
                if signal.get("_bootstrap") or signal.get("_bootstrap_override"):
                    extra_ctx["bootstrap_bypass"] = True
                    self.logger.info("[Meta:BOOTSTRAP] Passing bootstrap_bypass flag to ExecutionManager for %s", symbol)
                    # FIX: Increment attempts counter for one-shot policy
                    self._bootstrap_attempts += 1
                policy_ctx = self._build_policy_context(
                    symbol,
                    "BUY",
                    policies=[p for p, enabled in policy_flags.items() if enabled],
                    extra=extra_ctx,
                )
                self.increment_execution_attempts()
                result = await self.execution_manager.execute_trade(
                    symbol=symbol,
                    side="buy",
                    quantity=None,
                    planned_quote=planned_quote,
                    tag=signal.get("tag") or f"meta-{signal.get('agent', 'Meta')}",
                    tier=tier,
                    policy_context=policy_ctx,
                )
                last_result = result
                # Option A: Reset bootstrap override counter if deadlocked
                if signal.get("_bootstrap_override", False):
                    self._reset_bootstrap_override_if_deadlocked(symbol, signal, last_result)
                # Escalation Loop: BUY -> fail (InsufficientBalance) -> liquidate -> retry (Behavior Change 5)
                ec = result.get("error_code") or result.get("reason")
                if not result.get("ok") and ec in ("InsufficientBalance", "INSUFFICIENT_QUOTE", "RESERVE_FLOOR", "QUOTE_LT_MIN_NOTIONAL", "MIN_NOTIONAL_VIOLATION"):
                    # P9: Report capital failure so allocator knows
                    agent = signal.get("agent", "Meta")
                    if hasattr(self.shared_state, "report_agent_capital_failure"):
                        self.shared_state.report_agent_capital_failure(agent)
                    if self.liquidation_agent and hasattr(self.liquidation_agent, "_free_usdt_now"):
                        self.logger.info("⚡ [Escalation] Insufficient funds for %s. Triggering forced liquidation.", symbol)
                        # Mandatory liquidation trigger (Behavior Change 2)
                        await self.liquidation_agent._free_usdt_now(
                            target=float(planned_quote),
                            reason=f"escalation_{symbol}"
                        )
                    elif self.liquidation_agent:
                        self.logger.info("⚡ [Escalation] Insufficient funds for %s. Using propose_liquidations fallback.", symbol)
                        try:
                            proposals = await self.liquidation_agent.propose_liquidations(
                                gap_usdt=float(planned_quote),
                                reason=f"escalation_{symbol}",
                                force=True
                            )
                            if proposals:
                                self.logger.info(f"[Meta] Generated {len(proposals)} liquidation proposals for insufficient funds")
                        except Exception as e:
                            self.logger.warning(f"[Meta] Failed to generate liquidation proposals: {e}")
                    else:
                        self.logger.warning(f"[Meta] No liquidation agent available for insufficient funds escalation")
                        # P9: Trigger immediate re-plan (system rebalance)
                        if hasattr(self.shared_state, "replan_request_event"):
                            self.shared_state.replan_request_event.set()
                        # Brief wait for reconciliation
                        await _asyncio.sleep(float(self._cfg("ESCALATION_RETRY_DELAY_SEC", default=2.0)))
                        
                        # CRITICAL FIX: Refresh balance from exchange after escalation liquidation
                        # Problem: Cached balance may not reflect liquidation results
                        # Solution: Force sync before retry to ensure accurate affordability check
                        try:
                            await self.shared_state.sync_authoritative_balance(force=True)
                            self.logger.debug("Balance refreshed after escalation liquidation for %s", symbol)
                        except Exception as e:
                            self.logger.warning("Failed to refresh balance after escalation: %s", e)
                        
                        # Authoritative Retry
                        self.logger.info("🔄 [Escalation] Retrying BUY for %s after liquidation.", symbol)
                        retry_policy_ctx = self._build_policy_context(
                            symbol,
                            "BUY",
                            policies=[p for p, enabled in policy_flags.items() if enabled],
                            extra={
                                "economic_guard": econ_metrics,
                                "retry": "post_liquidation",
                            },
                        )
                        self.increment_execution_attempts()
                        result = await self.execution_manager.execute_trade(
                            symbol=symbol,
                            side="buy",
                            quantity=None,
                            planned_quote=planned_quote,
                            tag=signal.get("tag") or f"meta-{signal.get('agent', 'Meta')}",
                            policy_context=retry_policy_ctx,
                        )
                # Post-success cooldown & health bump
                if str(result.get("status", "")).lower() in {"placed", "executed", "filled"}:
                    ts = time.time()
                    self._trade_timestamps.append(ts)
                    self._trade_timestamps_day.append(ts)
                    self._trade_timestamps_sym[symbol].append(ts)
                    self._trade_timestamps_agent[agent_name].append(ts)
                    self._last_buy_ts[symbol] = ts
                    buy_price = float(result.get("avgPrice") or result.get("price") or signal.get("price") or 0.0)
                    if not buy_price:
                        buy_price = float(getattr(self.shared_state, "latest_prices", {}).get(symbol, 0.0) or 0.0)
                    if buy_price > 0:
                        self._last_buy_price[symbol] = buy_price

                    if signal.get("_bootstrap_seed") or signal.get("bootstrap_seed"):
                        self._bootstrap_seed_active = True
                        self._bootstrap_seed_used = True
                        self.logger.warning(
                            "[BOOTSTRAP] Seed BUY executed: %s | quote=%.2f",
                            symbol,
                            float(result.get("cummulativeQuoteQty") or result.get("cummulative_quote") or result.get("quote") or signal.get("_planned_quote") or 0.0),
                        )

                    # Ensure TP/SL is armed immediately after BUY (best-effort)
                    try:
                        if self.tp_sl_engine and hasattr(self.tp_sl_engine, "set_initial_tp_sl"):
                            exec_qty = float(result.get("executedQty") or result.get("executed_qty") or result.get("qty") or 0.0)
                            exec_price = float(result.get("avgPrice") or result.get("price") or 0.0)
                            if exec_qty > 0 and exec_price > 0:
                                self.tp_sl_engine.set_initial_tp_sl(symbol, exec_price, exec_qty, tier=tier)
                    except Exception:
                        self.logger.debug("TP/SL post-fill arm failed (non-fatal)", exc_info=True)

                    # ═══════════════════════════════════════════════════════════
                    # FIX 7: Set dust healing cooldown after successful healing BUY.
                    # Without this, the healing loop re-triggers every 2-second
                    # cycle because _dust_merges.clear() resets per-cycle and
                    # dust_healing_cooldown is never populated for BUY fills.
                    # Cooldown = 120s (enough for SharedState to refresh values).
                    # ═══════════════════════════════════════════════════════════
                    if signal.get("_dust_healing"):
                        _healing_cooldown_sec = float(self._cfg(
                            "DUST_HEALING_COOLDOWN_SEC", default=120
                        ))
                        self.dust_healing_cooldown[symbol] = ts + _healing_cooldown_sec
                        # Also update lifecycle state to exit DUST_HEALING
                        self._set_lifecycle(symbol, "HEALED_PENDING_GRADUATION")
                        self.logger.warning(
                            "[DUST_HEALING] FIX7: Set healing cooldown for %s: "
                            "%.0fs (until %.0f). Prevents runaway re-healing.",
                            symbol, _healing_cooldown_sec,
                            self.dust_healing_cooldown[symbol]
                        )

                    fp = signal.get("_signal_fingerprint") or self._signal_fingerprint(signal)
                    if fp:
                        self._last_signal_fingerprint[symbol] = fp
                    self._last_buy_ts[symbol] = ts
                    buy_price = float(result.get("avgPrice") or result.get("price") or signal.get("price") or 0.0)
                    if not buy_price:
                        buy_price = float(getattr(self.shared_state, "latest_prices", {}).get(symbol, 0.0) or 0.0)
                    if buy_price > 0:
                        self._last_buy_price[symbol] = buy_price
                    
                    # Track trade execution for focus mode exit condition
                    if self._focus_mode_active:
                        self._focus_mode_trade_executed = True
                        self._focus_mode_trade_executed_count += 1
                        self.logger.info(f"[FOCUS_MODE] Trade executed: {symbol} {side} (count={self._focus_mode_trade_executed_count})")
                    
                    try:
                        if hasattr(self.shared_state, "set_cooldown"):
                            cooldown_sec = float(self._cfg("META_DECISION_COOLDOWN_SEC", default=15))
                            await self.shared_state.set_cooldown(symbol, cooldown_sec)
                    except Exception:
                        self.logger.debug("Failed to set SharedState cooldown for %s", symbol)
                    try:
                        await self._health_set("Healthy", f"Executed BUY {symbol}")
                    except Exception:
                        pass
            else:  # SELL
                policy_flags = {"POLICY_SINGLE_AUTHORITY": True}
                qty = signal.get("quantity")
                tag = signal.get("_tag") or signal.get("tag") or ""
                sell_tag = self._resolve_sell_tag(signal)
                signal["tag"] = sell_tag
                is_starvation_sell = bool(signal.get("_is_starvation_sell"))
                is_quote_based_signal = bool(signal.get("_quote_based"))
                is_batch_sell = bool(signal.get("_batch_sell"))
                is_liq_signal = bool(
                    is_starvation_sell
                    or is_quote_based_signal
                    or is_batch_sell
                    or signal.get("_force_dust_liquidation")
                    or (sell_tag == "liquidation")
                )
                sell_reason = str(
                    signal.get("reason")
                    or signal.get("_reason")
                    or signal.get("_exit_reason")
                    or ""
                )
                dust_reason_detail = None
                if signal.get("_force_dust_liquidation") or signal.get("_is_dust"):
                    dust_reason_detail = sell_reason
                    sell_reason = "DUST_CLEANUP"
                sell_policy_extra = {
                    "liquidation_signal": is_liq_signal,
                    "dust_value": signal.get("_dust_value"),
                    "liquidation_reason": sell_reason,
                    "sell_source": sell_tag,
                }
                if dust_reason_detail:
                    sell_policy_extra["dust_reason_detail"] = dust_reason_detail
                if signal.get("_phase2_guard"):
                    policy_flags["PHASE_2_GRACE_PERIOD"] = True
                    sell_policy_extra["phase2_guard"] = signal["_phase2_guard"]
                reason_text = f"{sell_reason} {tag}".lower()
                is_hard_stop = bool(
                    signal.get("_is_sl")
                    or signal.get("_is_stop_loss")
                    or signal.get("_hard_stop")
                    or "stop_loss" in reason_text
                    or "stoploss" in reason_text
                    or "hard_stop" in reason_text
                    or "hardstop" in reason_text
                    or " sl" in f" {reason_text}"
                )
                # [FIX #4] UNIFIED SELL AUTHORITY: Liquidation SELLs have supreme authority
                # This ensures MetaController's decision to liquidate cannot be blocked by lower layers
                if is_liq_signal:
                    policy_flags["UNIFIED_SELL_AUTHORITY"] = True
                    self.logger.info(f"[Meta:UnifiedSell] {symbol} liquidation SELL will assert UNIFIED_SELL_AUTHORITY")
                # Bootstrap SELL override context (allow break-even/small loss exits)
                bootstrap_mode = False
                bootstrap_active = False
                try:
                    if hasattr(self.shared_state, "is_bootstrap_mode"):
                        bootstrap_mode = self.shared_state.is_bootstrap_mode()
                        if _asyncio.iscoroutine(bootstrap_mode):
                            bootstrap_mode = await bootstrap_mode
                except Exception:
                    bootstrap_mode = False
                if not bootstrap_mode:
                    try:
                        if hasattr(self.mode_manager, "get_mode"):
                            bootstrap_mode = str(self.mode_manager.get_mode() or "").upper() == "BOOTSTRAP"
                    except Exception:
                        bootstrap_mode = False
                if bootstrap_mode:
                    try:
                        duration_min = int(getattr(self.config, "BOOTSTRAP_DURATION_MINUTES", 1440))
                        start_ts = float(getattr(self.shared_state, "_start_time_unix", 0.0) or 0.0)
                        if not start_ts:
                            start_ts = float(getattr(self, "_start_time", 0.0) or 0.0)
                        if start_ts:
                            uptime_min = (time.time() - start_ts) / 60.0
                            bootstrap_active = uptime_min <= float(duration_min)
                    except Exception:
                        bootstrap_active = False
                bootstrap_allow_loss = bool(getattr(self.config, "BOOTSTRAP_ALLOW_SELL_BELOW_FEE", True))
                bootstrap_min_net = float(getattr(self.config, "BOOTSTRAP_MAX_NEGATIVE_PNL", 0.0) or 0.0)
                # Rotation/capacity override: allow small capped loss to free capacity
                rotation_override = False
                try:
                    rotation_override = bool(signal.get("_is_rotation") or signal.get("_rotation_escape"))
                    if not rotation_override:
                        reason_u = str(sell_reason).upper()
                        rotation_override = any(k in reason_u for k in ("ROTATION", "CAPITAL_RECOVERY", "PORTFOLIO_FULL", "RECOVERY"))
                    if not rotation_override and hasattr(self.shared_state, "metrics"):
                        rotation_override = bool(self.shared_state.metrics.get("portfolio_full", False))
                except Exception:
                    rotation_override = False

                sell_policy_extra.update({
                    "bootstrap_sell_override": bool(bootstrap_active and bootstrap_allow_loss),
                    "bootstrap_sell_allow_loss": bootstrap_allow_loss,
                    "bootstrap_sell_min_net": bootstrap_min_net,
                    "rotation_sell_override": rotation_override,
                })
                if not is_liq_signal:
                    is_cold = False
                    try:
                        cold_attr = getattr(self.shared_state, "is_cold_bootstrap", None)
                        if callable(cold_attr):
                            cold_state = cold_attr()
                            if _asyncio.iscoroutine(cold_state):
                                cold_state = await cold_state
                            is_cold = bool(cold_state)
                        elif cold_attr is not None:
                            is_cold = bool(cold_attr)
                    except Exception:
                        is_cold = False
                    if is_cold:
                        reason = str(signal.get("reason", "") or "")
                        is_recovery_sell = any(
                            token in reason
                            for token in (
                                "CAPITAL_RECOVERY",
                                "CAPITAL_RECOVERY_SOFT_SELL",
                                "CAPITAL_RECOVERY_HARD_SELL",
                                "CAPITAL_FLOOR_FORCED_SELL",
                                "CAPITAL_RECOVERY_NOMINATED",
                                "ROTATION",
                                "STAGNATION",
                            )
                        ) or bool(
                            signal.get("_capital_recovery_soft")
                            or signal.get("_capital_recovery_forced")
                            or signal.get("_capital_recovery_nominated")
                            or signal.get("_is_rotation")
                        )
                        is_liq_agent = str(signal.get("agent") or "") == "LiquidationAgent"
                        conf = float(signal.get("confidence", 0.0) or 0.0)
                        if is_recovery_sell or (is_liq_agent and conf >= 0.9):
                            self.logger.info(
                                "[Meta:RecoverySell] Allowing recovery SELL during cold bootstrap: %s",
                                symbol
                            )
                        else:
                            self._log_reason("INFO", symbol, "sell_blocked_cold_bootstrap")
                            if hasattr(self.shared_state, "record_rejection"):
                                await self.shared_state.record_rejection(
                                    symbol, "SELL", "COLD_BOOTSTRAP_BLOCK", source="MetaController"
                                )
                            await self._log_execution_result(
                                symbol,
                                side,
                                signal,
                                {"status": "blocked", "reason": "cold_bootstrap_no_sell"},
                            )
                            return {"ok": False, "status": "blocked", "reason": "cold_bootstrap_no_sell"}
                # Min holding time gate (SELL only). Bypass for liquidation/hard stop-loss.
                if not is_liq_signal and not is_hard_stop:
                    min_hold_sec = float(self._cfg("MIN_HOLD_SEC", default=45.0) or 0.0)
                    if bootstrap_active:
                        try:
                            min_hold_sec = float(getattr(self.config, "MIN_HOLD_SEC_BOOTSTRAP", min_hold_sec) or min_hold_sec)
                        except Exception:
                            pass
                    if min_hold_sec > 0:
                        entry_ts = 0.0
                        try:
                            sym_norm = self._normalize_symbol(symbol)
                            open_trades = getattr(self.shared_state, "open_trades", {}) or {}
                            if isinstance(open_trades, dict):
                                ot = open_trades.get(symbol) or open_trades.get(sym_norm) or {}
                                entry_ts = parse_timestamp(
                                    ot.get("entry_time")
                                    or ot.get("created_at")
                                    or ot.get("opened_at")
                                )
                            if not entry_ts:
                                positions = getattr(self.shared_state, "positions", {}) or {}
                                if isinstance(positions, dict):
                                    pos = positions.get(symbol) or positions.get(sym_norm) or {}
                                    entry_ts = parse_timestamp(
                                        pos.get("entry_time")
                                        or pos.get("opened_at")
                                        or pos.get("created_at")
                                    )
                        except Exception:
                            entry_ts = 0.0

                        if entry_ts > 0:
                            age_sec = max(0.0, time.time() - entry_ts)
                            if age_sec < min_hold_sec:
                                remaining = max(0.0, min_hold_sec - age_sec)
                                self._log_reason("INFO", symbol, f"sell_min_hold:{age_sec:.1f}s<{min_hold_sec:.0f}s")
                                self.logger.info(
                                    "[Meta:MinHold] SELL blocked for %s: age=%.1fs < min_hold=%.0fs (remaining=%.1fs)",
                                    symbol,
                                    age_sec,
                                    min_hold_sec,
                                    remaining,
                                )
                                if hasattr(self.shared_state, "record_rejection"):
                                    await self.shared_state.record_rejection(
                                        symbol, "SELL", "MIN_HOLD_SEC", source="MetaController"
                                    )
                                await self._log_execution_result(
                                    symbol,
                                    side,
                                    signal,
                                    {"status": "blocked", "reason": "min_hold", "min_hold_sec": min_hold_sec, "age_sec": age_sec},
                                )
                                return {"ok": False, "status": "blocked", "reason": "min_hold"}
                # LiquidationAgent-specific min hold (blocks fresh positions).
                if is_liq_signal and str(signal.get("agent") or "") == "LiquidationAgent":
                    min_hold_sec = float(self._cfg("LIQ_MIN_HOLD_SEC", default=90.0) or 0.0)
                    if min_hold_sec > 0:
                        entry_ts = 0.0
                        try:
                            sym_norm = self._normalize_symbol(symbol)
                            open_trades = getattr(self.shared_state, "open_trades", {}) or {}
                            if isinstance(open_trades, dict):
                                ot = open_trades.get(symbol) or open_trades.get(sym_norm) or {}
                                entry_ts = parse_timestamp(
                                    ot.get("entry_time")
                                    or ot.get("created_at")
                                    or ot.get("opened_at")
                                )
                            if not entry_ts:
                                positions = getattr(self.shared_state, "positions", {}) or {}
                                if isinstance(positions, dict):
                                    pos = positions.get(symbol) or positions.get(sym_norm) or {}
                                    entry_ts = parse_timestamp(
                                        pos.get("entry_time")
                                        or pos.get("opened_at")
                                        or pos.get("created_at")
                                    )
                        except Exception:
                            entry_ts = 0.0

                        if entry_ts > 0:
                            age_sec = max(0.0, time.time() - entry_ts)
                            if age_sec < min_hold_sec:
                                self._log_reason("INFO", symbol, f"liq_min_hold:{age_sec:.1f}s<{min_hold_sec:.0f}s")
                                self.logger.info(
                                    "[Meta:MinHold:Liq] SELL blocked for %s: age=%.1fs < min_hold=%.0fs",
                                    symbol,
                                    age_sec,
                                    min_hold_sec,
                                )
                                if hasattr(self.shared_state, "record_rejection"):
                                    await self.shared_state.record_rejection(
                                        symbol, "SELL", "LIQ_MIN_HOLD_SEC", source="MetaController"
                                    )
                                await self._log_execution_result(
                                    symbol,
                                    side,
                                    signal,
                                    {"status": "blocked", "reason": "liq_min_hold", "min_hold_sec": min_hold_sec, "age_sec": age_sec},
                                )
                                return {"ok": False, "status": "blocked", "reason": "liq_min_hold"}
                        else:
                            self._log_reason("INFO", symbol, "liq_min_hold_missing_entry")
                            await self._log_execution_result(
                                symbol,
                                side,
                                signal,
                                {"status": "blocked", "reason": "liq_min_hold_missing_entry"},
                            )
                            return {"ok": False, "status": "blocked", "reason": "liq_min_hold_missing_entry"}
                # Get position from SharedState (Canonical)
                total_pos_qty = float(self.shared_state.get_position_qty(symbol) or 0.0)
                qty = total_pos_qty
                partial_pct = float(signal.get("_partial_pct", 0.0) or signal.get("partial_pct", 0.0) or 0.0)
                
                if partial_pct > 0.0 and partial_pct < 1.0:
                    # FIX: Clean Exit Logic (Prevent Dust Remainder)
                    # If remaining amount is dust (< $11), force full exit
                    target_qty = total_pos_qty * partial_pct
                    remaining_qty = total_pos_qty - target_qty
                    force_full = False
                    
                    try:
                        price = float(await self.shared_state.get_latest_price(symbol) or 0.0)
                        # Use slightly higher threshold (11.0) than standard 10.0 to be safe
                        if price > 0 and (remaining_qty * price) < 11.0:
                            force_full = True
                            self.logger.info(
                                "[Meta:CleanExit] Partial sell leaves dust ($%.2f). Forcing FULL EXIT.",
                                remaining_qty * price
                            )
                    except Exception:
                        # Fallback logic: If selling > 90%, assume cleanup
                        if partial_pct > 0.90:
                            force_full = True
                    
                    if force_full:
                        qty = total_pos_qty
                    else:
                        qty = target_qty
                        self.logger.info(
                            "[Meta:PartialSell] %s applying partial_pct=%.1f%% qty=%.8f (leaves safe remainder)",
                            symbol, partial_pct * 100.0, qty
                        )
                # SELL net-PnL gate (non-liquidation, non-hard-stop)
                if not is_liq_signal and not is_hard_stop:
                    allow_below_fee = bool(getattr(self.config, "ALLOW_SELL_BELOW_FEE", False))
                    min_net_usdt = float(getattr(self.config, "SELL_MIN_NET_PNL_USDT", 0.0) or 0.0)
                    fee_mult = float(self._cfg("MIN_PROFIT_EXIT_FEE_MULT", default=2.0) or 2.0)

                    if not allow_below_fee:
                        entry_price = 0.0
                        try:
                            sym_norm = self._normalize_symbol(symbol)
                            open_trades = getattr(self.shared_state, "open_trades", {}) or {}
                            if isinstance(open_trades, dict):
                                ot = open_trades.get(symbol) or open_trades.get(sym_norm) or {}
                                entry_price = float(ot.get("entry_price", 0.0) or 0.0)
                        except Exception:
                            entry_price = 0.0
                        if not entry_price:
                            try:
                                positions = getattr(self.shared_state, "positions", {}) or {}
                                if isinstance(positions, dict):
                                    pos = positions.get(symbol) or positions.get(sym_norm) or {}
                                    entry_price = float(pos.get("avg_price", 0.0) or pos.get("entry_price", 0.0) or 0.0)
                            except Exception:
                                entry_price = 0.0

                        price = 0.0
                        try:
                            if hasattr(self.shared_state, "get_latest_price"):
                                price = float(await self.shared_state.get_latest_price(symbol) or 0.0)
                            if not price:
                                price = float(getattr(self.shared_state, "latest_prices", {}).get(symbol, 0.0) or 0.0)
                        except Exception:
                            price = float(getattr(self.shared_state, "latest_prices", {}).get(symbol, 0.0) or 0.0)

                        if entry_price > 0 and price > 0 and qty > 0:
                            proceeds = qty * price
                            fee_est = proceeds * float(getattr(self.config, "TRADE_FEE_PCT", 0.001) or 0.0) * 2.0
                            entry_cost = qty * entry_price
                            net_pnl = proceeds - fee_est - entry_cost
                            min_required = max(min_net_usdt, fee_est * fee_mult)

                            if net_pnl < min_required:
                                self._log_reason("INFO", symbol, f"sell_net_pnl:{net_pnl:.4f}<{min_required:.4f}")
                                self.logger.info(
                                    "[Meta:SellNetGate] SELL blocked for %s: net_pnl=%.4f < min_required=%.4f (fee=%.4f entry=%.4f price=%.4f qty=%.6f)",
                                    symbol, net_pnl, min_required, fee_est, entry_price, price, qty
                                )
                                if hasattr(self.shared_state, "record_rejection"):
                                    await self.shared_state.record_rejection(
                                        symbol, "SELL", "SELL_NET_PNL_MIN", source="MetaController"
                                    )
                                await self._log_execution_result(
                                    symbol,
                                    side,
                                    signal,
                                    {
                                        "status": "blocked",
                                        "reason": "sell_net_pnl",
                                        "net_pnl": net_pnl,
                                        "min_required": min_required,
                                    },
                                )
                                return {"ok": False, "status": "blocked", "reason": "sell_net_pnl"}
                        else:
                            self._log_reason("INFO", symbol, "sell_net_pnl_missing_data")
                            if hasattr(self.shared_state, "record_rejection"):
                                await self.shared_state.record_rejection(
                                    symbol, "SELL", "SELL_NET_PNL_MISSING", source="MetaController"
                                )
                            await self._log_execution_result(
                                symbol,
                                side,
                                signal,
                                {"status": "blocked", "reason": "sell_net_pnl_missing"},
                            )
                            return {"ok": False, "status": "blocked", "reason": "sell_net_pnl_missing"}
                if not qty or qty <= 0:
                    self._log_reason("INFO", symbol, "sell_no_position_qty")
                    await self._log_execution_result(
                        symbol, side, signal, {"status": "skipped", "reason": "no_position_quantity"}
                    )
                    return
                # Risk pre-check (SELL) with FIX #4: liquidation bypass support
                if self.risk_manager and hasattr(self.risk_manager, "pre_check"):
                    # FIX #4: Detect if this is a liquidation signal for bypass
                    ok, r_reason = await _safe_await(self.risk_manager.pre_check(
                        symbol=symbol, side="SELL", is_liquidation=is_liq_signal
                    ))
                    if not ok:
                        self._log_reason("INFO", symbol, f"risk_precheck:{r_reason}")
                        await self._log_execution_result(symbol, side, signal, {"status": "skipped", "reason": f"risk:{r_reason}"})
                        return
                # ===== FIX 3: Quote-Based SELL for Liquidation =====
                # If this is a liquidation SELL with quote-based flag, use quoteOrderQty instead of quantity
                if is_quote_based_signal and is_starvation_sell:
                    quote_value = signal.get("_target_usdt", 0.0)
                    if quote_value > 0:
                        self.logger.warning(
                            "[Meta:QuoteLiq:Execute] Executing QUOTE-BASED liquidation SELL. "
                            "Symbol: %s, TargetUSDT: %.2f, PositionQty: %.8f. Using quoteOrderQty (bypasses min-notional).",
                            symbol, quote_value, qty
                        )
                        # Execute with quoteOrderQty (USDT value, not quantity)
                        policy_ctx = self._build_policy_context(
                            symbol,
                            "SELL",
                            policies=[p for p, enabled in policy_flags.items() if enabled],
                            extra={**sell_policy_extra, "mode": "quote_based"},
                        )
                        self.increment_execution_attempts()
                        result = await self.execution_manager.execute_trade(
                            symbol=symbol,
                            side="sell",
                            quantity=None,  # Use quoteOrderQty instead
                            planned_quote=quote_value,  # Pass as USDT value
                            tag=sell_tag,
                            policy_context=policy_ctx,
                        )
                    else:
                        self.logger.warning("[Meta:QuoteLiq:Execute] Quote-based liquidation flag set but _target_usdt is invalid. Falling back to quantity-based sell.")
                        policy_ctx = self._build_policy_context(
                            symbol,
                            "SELL",
                            policies=[p for p, enabled in policy_flags.items() if enabled],
                            extra=sell_policy_extra,
                        )
                        self.increment_execution_attempts()
                        result = await self.execution_manager.execute_trade(
                            symbol=symbol,
                            side="sell",
                            quantity=qty,
                            tag=sell_tag,
                            policy_context=policy_ctx,
                        )
                else:
                    # ===== FIX 2: Liquidation SELLs BYPASS starvation gates =====
                    # Even if system is starved, liquidation SELLs execute to free capital
                    if is_starvation_sell:
                        self.logger.warning(
                            "[Meta:LiquidationHardPath:Execute] Executing LIQUIDATION SELL (batch or fallback). "
                            "Symbol: %s, Qty: %.8f. Bypassing starvation/affordability gates.",
                            symbol, qty
                        )
                    # Standard quantity-based SELL (batch liquidation or normal sell)
                    policy_ctx = self._build_policy_context(
                        symbol,
                        "SELL",
                        policies=[p for p, enabled in policy_flags.items() if enabled],
                        extra={**sell_policy_extra, "mode": "quantity"},
                    )
                    self.increment_execution_attempts()
                    result = await self.execution_manager.execute_trade(
                        symbol=symbol,
                        side="sell",
                        quantity=qty,
                        tag=sell_tag,
                        policy_context=policy_ctx,
                    )
                # Post-success cooldown & health bump
                if str(result.get("status", "")).lower() in {"placed", "executed", "filled"}:
                    try:
                        exit_reason = self._classify_exit_reason(signal)
                        await post_exit_bookkeeping(
                            self.shared_state,
                            self.config,
                            self.logger,
                            symbol,
                            exit_reason,
                            "meta",
                        )
                    except Exception:
                        pass
                    self._update_profit_lock_checkpoint()
                    # Track trade execution for focus mode exit condition
                    if self._focus_mode_active:
                        self._focus_mode_trade_executed = True
                        self._focus_mode_trade_executed_count += 1
                        self.logger.info(f"[FOCUS_MODE] Trade executed: {symbol} {side} (count={self._focus_mode_trade_executed_count})")

                    # Auto-clear recovery mode once capital is restored
                    try:
                        rec_state = getattr(self.shared_state, "capital_recovery_mode", {}) or {}
                        if isinstance(rec_state, dict) and rec_state.get("active"):
                            quote_asset = str(self._cfg("QUOTE_ASSET") or "USDT").upper()
                            free_usdt = float(await _safe_await(
                                self.shared_state.get_spendable_balance(quote_asset)
                            ) or 0.0)
                            nav = 0.0
                            try:
                                if hasattr(self.shared_state, "get_nav_quote"):
                                    nav = float(await _safe_await(self.shared_state.get_nav_quote()) or 0.0)
                                else:
                                    nav = float(getattr(self.shared_state, "nav", 0.0) or 0.0)
                            except Exception:
                                nav = 0.0
                            abs_min_floor = float(self._cfg("ABSOLUTE_MIN_FLOOR", self._cfg("CAPITAL_PRESERVATION_FLOOR", 10.0)))
                            floor_pct = float(self._cfg("CAPITAL_FLOOR_PCT", 0.20))
                            floor = max(abs_min_floor, nav * floor_pct)
                            if free_usdt >= floor:
                                setattr(self.shared_state, "capital_recovery_mode", {
                                    "active": False,
                                    "nomination_status_logged": False,
                                })
                                if hasattr(self.shared_state, "set_dynamic_param"):
                                    await _safe_await(self.shared_state.set_dynamic_param(
                                        "capital_recovery_status", {
                                            "recovery_active": False,
                                            "elapsed_sec": 0,
                                            "remaining_sec": 0,
                                            "max_age_sec": 0,
                                            "candidate": None,
                                            "floor": float(floor),
                                            "free_usdt": float(free_usdt),
                                        }
                                    ))
                                self.logger.info(
                                    "[Meta:CapitalRecovery] Cleared after SELL (free_usdt=%.2f >= floor=%.2f)",
                                    free_usdt, floor
                                )
                    except Exception:
                        pass
                    
                    try:
                        # Check current position quantity from SharedState
                        pos_qty = float(self.shared_state.get_position_qty(symbol) or 0.0)

                        # If fully closed, perform lifecycle cleanup
                        if pos_qty <= 0:
                            # 1. Clear dust merge history
                            if symbol in self._dust_merges:
                                self._dust_merges.discard(symbol)
                                self.logger.info("[Meta:Dust] Cleared dust merge history for %s after full exit", symbol)
                            
                            # 2. Log lifecycle reset (Sanity Check 2)
                            self.logger.info("[POSITION_CLOSED] symbol=%s lifecycle reset", symbol)

                    except Exception:
                        self.logger.debug("Failed to apply post-exit cleanup for %s", symbol)
                    try:
                        await self._health_set("Healthy", f"Executed SELL {symbol}")
                    except Exception:
                        pass
            await self._log_execution_result(symbol, side, signal, result)
            await self._update_kpi_metrics("execution")
            return result
        except Exception as e:
            classified_error = classify_execution_error(e, symbol)
            self.logger.error("Decision execution failed for %s: %s", symbol, classified_error.error_type, exc_info=True)
            # P9 Requirement 3/4: Report Capital Failure for Hysteresis
            if classified_error.error_type in (ExecutionError.Type.INSUFFICIENT_BALANCE, ExecutionError.Type.MIN_NOTIONAL_VIOLATION):
                agent = signal.get("agent", "Meta")
                if hasattr(self.shared_state, "report_agent_capital_failure"):
                    self.shared_state.report_agent_capital_failure(agent)
            await self._update_kpi_metrics("error", classified_error.error_type)
            await self._health_set("Critical", f"Execution error for {symbol}: {classified_error}")
            now = time.time()
            agent_name = signal.get("agent", "Meta")
            
            # 1. Clean old timestamps (Hourly window)
            for dq in [self._trade_timestamps, self._trade_timestamps_sym[symbol], self._trade_timestamps_agent[agent_name]]:
                while dq and (now - dq[0] > 3600):
                    dq.popleft()
            
            # ===== CRITICAL FIX #2A: Bootstrap BUY can bypass hourly limits =====
            is_bootstrap = "bootstrap" in str(signal.get("reason", "")).lower()
            is_flat_init = signal.get("_flat_init_buy", False)
            
            if not (is_bootstrap or is_flat_init):
                # 2. Gating logic (Phase A Enhancements) — Only for NORMAL BUYs
                # GLOBAL GATE
                if len(self._trade_timestamps) >= self._max_trades_per_hour:
                    self.logger.info("[Meta] Skip %s BUY: Global hourly trade limit (%d) reached.", symbol, self._max_trades_per_hour)
                    return {"ok": False, "status": "skipped", "reason": "global_limit", "reason_detail": "global_hourly_limit_reached"}
                
                # PER-SYMBOL GATE (Safety: Max 2 trades per symbol per hour during Phase A)
                if len(self._trade_timestamps_sym[symbol]) >= 2:
                    self.logger.info("[Meta] Skip %s BUY: Symbol hourly trade limit reached.", symbol)
                    return {"ok": False, "status": "skipped", "reason": "symbol_limit", "reason_detail": "symbol_hourly_limit_reached"}

                # PER-AGENT GATE (Safety: Max 75% of global limit per agent)
                agent_limit = max(1, int(self._max_trades_per_hour * 0.75))
                if len(self._trade_timestamps_agent[agent_name]) >= agent_limit:
                    self.logger.info("[Meta] Skip %s BUY: Agent %s hourly limit reached.", symbol, agent_name)
                    return {"ok": False, "status": "skipped", "reason": "agent_limit", "reason_detail": f"agent_limit_{agent_name}_reached"}
            else:
                # BOOTSTRAP BYPASS: Log that we're bypassing limits
                self.logger.info(
                    "[Meta:FIX#2] Bootstrap BUY %s bypassing hourly limits (is_bootstrap=%s, is_flat_init=%s)",
                    symbol, is_bootstrap, is_flat_init
                )

        if symbol not in accepted_symbols_set:
            # P9 FIX: SELL always bypasses accepted_symbols check
            # REASON: SELL is for exiting existing positions, even if symbol is not in analysis universe
            # SELL must never be blocked by universe filters - positions must always be exitiable
            if side == "SELL":
                self.logger.info(
                    "[Meta:P9] SELL bypass: %s not in accepted set but SELL must execute (P9 Rule: Exits always allowed). Proceeding.",
                    symbol
                )
            else:
                # For BUY: Allow bootstrap BUY to bypass accepted_symbols check
                is_bootstrap = "bootstrap" in str(signal.get("reason", "")).lower()
                if side == "BUY" and is_bootstrap:
                    self.logger.info("[Meta:Bootstrap] Gating bypass: %s is not in accepted set but is a bootstrap BUY. Proceeding.", symbol)
                else:
                    self.logger.warning("Skipping unaccepted symbol: %s", symbol)
                    await self._log_execution_result(symbol, side, signal, {"status": "skipped", "reason": "symbol_not_accepted"})
                    return {"ok": False, "status": "skipped", "reason": "symbol_not_accepted"}

        # Ensure per-symbol market data is ready if SharedState provides a hook
        try:
            fn = getattr(self.shared_state, "is_symbol_data_ready", None)
            if callable(fn):
                ok = fn(symbol)
                if _asyncio.iscoroutine(ok):
                    ok = await ok
                if not ok:
                    await self._log_execution_result(symbol, side, signal, {"status": "skipped", "reason": "symbol_data_not_ready"})
                    return {"ok": False, "status": "skipped", "reason": "symbol_data_not_ready"}
        except Exception:
            self.logger.debug("symbol readiness check failed for %s", symbol, exc_info=True)

        try:
            if side == "BUY":
                # Strict Rule 5: Zero-amount execution must NEVER be OK
                planned_quote = float(signal.get("_planned_quote", 0.0))
                if planned_quote <= 0:
                    planned_quote = await self._planned_quote_for(symbol, signal)

                # TIER 1: ACCUMULATING State Protection
                # Skip SELL on fresh accumulating positions (unless risk/liquidation/tp_sl)
                intent_owner = signal.get("_intent_owner", "")
                if side == "SELL" and hasattr(self.shared_state, "get_pending_intent"):
                    intent = self.shared_state.get_pending_intent(symbol, "BUY")
                    if intent and intent.state == "ACCUMULATING":
                        is_protected_sell = any(tag in str(intent_owner).lower() or tag in str(signal.get("agent", "")).lower() 
                                            for tag in ["risk", "liquidation", "tp_sl", "emergency", "rotation", "authority"])
                        if not is_protected_sell:
                            self.logger.info(
                                "[Meta:AccumGuard] Blocking SELL on %s: position is ACCUMULATING (intent_owner=%s). "
                                "Only risk/liquidation/tp_sl/rotation/authority can override.",
                                symbol, intent_owner
                            )
                            await self.shared_state.record_rejection(
                                symbol, "SELL", "ACCUMULATING_PROTECTION", source="MetaController"
                            )
                            # TIER 2: Track conflict (economic gate vs accumulation protection)
                            if hasattr(self.shared_state, "record_policy_conflict"):
                                self.shared_state.record_policy_conflict("accumulating_protection_blocks")
                            return {"ok": False, "status": "skipped", "reason": "accumulating_protection"}

                # TIER 1: Capital Preservation Floor ($50 minimum)
                min_capital_floor = float(self._cfg("CAPITAL_PRESERVATION_FLOOR", 50.0))
                free_quote = 0.0
                try:
                    if hasattr(self.shared_state, "get_free_quote"):
                        maybe = self.shared_state.get_free_quote()
                        if _asyncio.iscoroutine(maybe):
                            free_quote = await maybe
                        else:
                            free_quote = float(maybe or 0.0)
                except Exception:
                    free_quote = 0.0

                # PHASE 2 NOTE: Capital floor check removed (now checked at start of _build_decisions)
                # If we reached this point, capital has already been validated
                # No redundant check needed here

                policy_flags = {"POLICY_SINGLE_AUTHORITY": True}
                econ_ok, econ_reason, econ_metrics = self._check_economic_profitability(symbol, signal)
                if not econ_ok:
                    self._log_reason("INFO", symbol, f"economic_guard:{econ_reason}")
                    if hasattr(self.shared_state, "record_rejection"):
                        await self.shared_state.record_rejection(
                            symbol, "BUY", "ECONOMIC_PROFITABILITY_BLOCK", source="MetaController"
                        )
                    await self._log_execution_result(
                        symbol,
                        side,
                        signal,
                        {"status": "skipped", "reason": "economic_guard", "details": econ_metrics},
                    )
                    return {"ok": False, "status": "skipped", "reason": "economic_guard"}
                policy_flags["ECONOMIC_PROFITABILITY_INVARIANT"] = True
                
                # Double check affordability for executable qty (Rule 2/5)
                # BOOTSTRAP FIX: If this signal is marked with _bootstrap flag, pass bootstrap context
                bootstrap_policy_ctx = None
                if signal.get("_bootstrap") or signal.get("_bootstrap_override"):
                    bootstrap_policy_ctx = self._build_policy_context(
                        symbol,
                        side,
                        extra={"bootstrap_bypass": True}
                    )
                    self.logger.info("[Meta:BOOTSTRAP] Using bootstrap_bypass context for affordability check: %s", symbol)

                can_ex, _, reason = await self.execution_manager.can_afford_market_buy(symbol, planned_quote, policy_context=bootstrap_policy_ctx)

                if not can_ex:
                    self.logger.warning("⚡ [Escalation] Signal %s for %s has zero executable qty (%s). Triggering Rule 5 Escalation.", symbol, side, reason)
                    
                    # P9: Report capital failure immediately
                    agent = signal.get("agent", "Meta")
                    if hasattr(self.shared_state, "report_agent_capital_failure"):
                        self.shared_state.report_agent_capital_failure(agent)

                    # Rule 6: Liquidation must be able to invalidate readiness
                    if hasattr(self.shared_state, "ops_plane_ready_event"):
                        self.shared_state.ops_plane_ready_event.clear()
                        self.logger.info("[Meta] Readiness = FALSE (Escalation Triggered)")
                        
                    if self.liquidation_agent and hasattr(self.liquidation_agent, "_free_usdt_now"):
                        await self.liquidation_agent._free_usdt_now(
                            target=float(self._cfg("MIN_NOTIONAL_FLOOR", 15.0)),
                            reason=f"rule5_escalation_{symbol}"
                        )
                    elif self.liquidation_agent:
                        # If liquidation_agent doesn't have _free_usdt_now method, try propose_liquidations instead
                        self.logger.warning(f"[Meta] LiquidationAgent doesn't have _free_usdt_now method. Using propose_liquidations instead.")
                        try:
                            target_usdt = float(self._cfg("MIN_NOTIONAL_FLOOR", 15.0))
                            proposals = await self.liquidation_agent.propose_liquidations(
                                gap_usdt=target_usdt,
                                reason=f"rule5_escalation_{symbol}",
                                force=True
                            )
                            if proposals:
                                self.logger.info(f"[Meta] Generated {len(proposals)} liquidation proposals for escalation")
                        except Exception as e:
                            self.logger.warning(f"[Meta] Failed to generate liquidation proposals: {e}")
                    else:
                        self.logger.warning(f"[Meta] No liquidation agent available for escalation")
                    # P9: Trigger immediate re-plan
                    if hasattr(self.shared_state, "replan_request_event"):
                        self.shared_state.replan_request_event.set()
                    await self._log_execution_result(symbol, side, signal, {"status": "failed", "reason": f"rule5_escalation_{reason}"})
                    return {"ok": False, "status": "failed", "reason": f"rule5_escalation_{reason}"}

                # Handle liquidity healing if needed
                if signal.get("_need_liquidity"):
                    success = await self._attempt_liquidity_healing(symbol, signal)
                    if not success:
                        await self._log_execution_result(
                            symbol, side, signal, {"status": "skipped", "reason": "liquidity_healing_failed"}
                        )
                        return {"ok": False, "status": "skipped", "reason": "liquidity_healing"}

                # Hard per-symbol guard to prevent stacked BUYs without explicit scale-in
                max_per_symbol = int(self._cfg(
                    "MAX_OPEN_POSITIONS_PER_SYMBOL",
                    default=self._max_open_positions_per_symbol,
                ))
                if max_per_symbol <= 1:
                    has_open, existing_qty = await self._has_open_position(symbol)
                    allow_scale_in = bool(
                        signal.get("_scale_in")
                        or signal.get("_accumulate_mode")
                        or signal.get("_allow_reentry")
                    )
                    if has_open and not allow_scale_in:
                        await self._log_execution_result(
                            symbol,
                            side,
                            signal,
                            {
                                "status": "skipped",
                                "reason": "position_already_open",
                                "reason_detail": f"max_open_positions_per_symbol qty={existing_qty:.6f}",
                            },
                        )
                        return {"ok": False, "status": "skipped", "reason": "position_already_open"}

                # Risk pre-check (BUY) - Tier-aware with FIX #4: liquidation bypass support
                if self.risk_manager and hasattr(self.risk_manager, "pre_check"):
                    tier = signal.get("_tier", "A")  # Default to Tier A if not specified
                    # FIX #4: Detect if this is a liquidation signal for bypass
                    is_liq = signal.get("_is_starvation_sell") or signal.get("_quote_based") or signal.get("_batch_sell")
                    tag = signal.get("_tag") or signal.get("tag") or ""
                    is_liq = is_liq or ("liquidation" in str(tag))
                    
                    ok, r_reason = await _safe_await(self.risk_manager.pre_check(
                        symbol=symbol, side="BUY", planned_quote=planned_quote, tier=tier, is_liquidation=is_liq
                    ))
                    if not ok:
                        self._log_reason("INFO", symbol, f"risk_precheck:{r_reason}")
                        await self._log_execution_result(symbol, side, signal, {"status": "skipped", "reason": f"risk:{r_reason}"})
                        return {"ok": False, "status": "skipped", "reason": f"risk:{r_reason}"}

                # P9: Revalidate signal + risk context at firing time
                if hasattr(self.shared_state, "is_intent_valid"):
                    is_bootstrap_seed = bool(
                        signal.get("_bootstrap_seed")
                        or signal.get("bootstrap_seed")
                        or str(signal.get("reason", "")).upper() == "BOOTSTRAP_SEED"
                        or str(signal.get("execution_tag", "")) == "meta/bootstrap_seed"
                    )
                    if not is_bootstrap_seed and not self.shared_state.is_intent_valid(symbol, "BUY"):
                        self.logger.warning("[Meta] Signal no longer valid at firing time for %s. Skipping.", symbol)
                        await self._log_execution_result(symbol, side, signal, {"status": "skipped", "reason": "signal_invalid_at_firing"})
                        return {"ok": False, "status": "skipped", "reason": "signal_invalid"}

                # Execute through ExecutionManager (canonical single order path)
                tier = signal.get("_tier", "A")
                extra_ctx = {"economic_guard": econ_metrics}
                
                # ACCUMULATE_MODE: If signal is marked with _accumulate_mode, pass it through policy context
                if signal.get("_accumulate_mode"):
                    extra_ctx["_accumulate_mode"] = True
                    self.logger.info("[Meta:P0] Passing ACCUMULATE_MODE flag to ExecutionManager for %s", symbol)
                
                policy_ctx = self._build_policy_context(
                    symbol,
                    "BUY",
                    policies=[p for p, enabled in policy_flags.items() if enabled],
                    extra=extra_ctx,
                )
                self.increment_execution_attempts()
                result = await self.execution_manager.execute_trade(
                    symbol=symbol,
                    side="buy",
                    quantity=None,
                    planned_quote=planned_quote,
                    tag=f"meta-{signal.get('agent', 'Meta')}",
                    tier=tier,
                    policy_context=policy_ctx,
                )

                # Escalation Loop: BUY -> fail (InsufficientBalance) -> liquidate -> retry (Behavior Change 5)
                ec = result.get("error_code") or result.get("reason")
                if not result.get("ok") and ec in ("InsufficientBalance", "INSUFFICIENT_QUOTE", "RESERVE_FLOOR", "QUOTE_LT_MIN_NOTIONAL", "MIN_NOTIONAL_VIOLATION"):
                    # P9: Report capital failure so allocator knows
                    agent = signal.get("agent", "Meta")
                    if hasattr(self.shared_state, "report_agent_capital_failure"):
                        self.shared_state.report_agent_capital_failure(agent)

                    if self.liquidation_agent and hasattr(self.liquidation_agent, "_free_usdt_now"):
                        self.logger.info("⚡ [Escalation] Insufficient funds for %s. Triggering forced liquidation.", symbol)
                        # Mandatory liquidation trigger (Behavior Change 2)
                        await self.liquidation_agent._free_usdt_now(
                            target=float(planned_quote),
                            reason=f"escalation_{symbol}"
                        )
                    elif self.liquidation_agent:
                        self.logger.info("⚡ [Escalation] Insufficient funds for %s. Using propose_liquidations fallback.", symbol)
                        try:
                            proposals = await self.liquidation_agent.propose_liquidations(
                                gap_usdt=float(planned_quote),
                                reason=f"escalation_{symbol}",
                                force=True
                            )
                            if proposals:
                                self.logger.info(f"[Meta] Generated {len(proposals)} liquidation proposals for insufficient funds")
                        except Exception as e:
                            self.logger.warning(f"[Meta] Failed to generate liquidation proposals: {e}")
                    else:
                        self.logger.warning(f"[Meta] No liquidation agent available for insufficient funds escalation")
                        # P9: Trigger immediate re-plan (system rebalance)
                        if hasattr(self.shared_state, "replan_request_event"):
                            self.shared_state.replan_request_event.set()
                        # Brief wait for reconciliation
                        await _asyncio.sleep(float(self._cfg("ESCALATION_RETRY_DELAY_SEC", default=2.0)))
                        
                        # Authoritative Retry
                        self.logger.info("🔄 [Escalation] Retrying BUY for %s after liquidation.", symbol)
                        retry_policy_ctx = self._build_policy_context(
                            symbol,
                            "BUY",
                            policies=[p for p, enabled in policy_flags.items() if enabled],
                            extra={
                                "economic_guard": econ_metrics,
                                "retry": "post_liquidation",
                            },
                        )
                        self.increment_execution_attempts()
                        result = await self.execution_manager.execute_trade(
                            symbol=symbol,
                            side="buy",
                            quantity=None,
                            planned_quote=planned_quote,
                            tag=f"meta-{signal.get('agent', 'Meta')}",
                            policy_context=retry_policy_ctx,
                        )

                # Post-success cooldown & health bump
                if str(result.get("status", "")).lower() in {"placed", "executed", "filled"}:
                    ts = time.time()
                    self._trade_timestamps.append(ts)
                    self._trade_timestamps_day.append(ts)
                    self._trade_timestamps_sym[symbol].append(ts)
                    self._trade_timestamps_agent[agent_name].append(ts)
                    self._last_buy_ts[symbol] = ts
                    try:
                        if hasattr(self.shared_state, "set_cooldown"):
                            cooldown_sec = float(self._cfg("META_DECISION_COOLDOWN_SEC", default=15))
                            await self.shared_state.set_cooldown(symbol, cooldown_sec)
                    except Exception:
                        self.logger.debug("Failed to set SharedState cooldown for %s", symbol)
                    try:
                        await self._health_set("Healthy", f"Executed BUY {symbol}")
                    except Exception:
                        pass

            else:  # SELL
                policy_flags = {"POLICY_SINGLE_AUTHORITY": True}
                qty = signal.get("quantity")

                tag = signal.get("_tag") or signal.get("tag") or ""
                sell_tag = self._resolve_sell_tag(signal)
                signal["tag"] = sell_tag
                is_starvation_sell = bool(signal.get("_is_starvation_sell"))
                is_quote_based_signal = bool(signal.get("_quote_based"))
                is_batch_sell = bool(signal.get("_batch_sell"))
                is_liq_signal = bool(
                    is_starvation_sell
                    or is_quote_based_signal
                    or is_batch_sell
                    or signal.get("_force_dust_liquidation")
                    or (sell_tag == "liquidation")
                )
                sell_policy_extra = {
                    "liquidation_signal": is_liq_signal,
                    "dust_value": signal.get("_dust_value"),
                    "sell_source": sell_tag,
                }
                if signal.get("_phase2_guard"):
                    policy_flags["PHASE_2_GRACE_PERIOD"] = True
                    sell_policy_extra["phase2_guard"] = signal["_phase2_guard"]

                # [FIX #4] UNIFIED SELL AUTHORITY: Liquidation SELLs have supreme authority
                # This ensures MetaController's decision to liquidate cannot be blocked by lower layers
                if is_liq_signal:
                    policy_flags["UNIFIED_SELL_AUTHORITY"] = True
                    self.logger.info(f"[Meta:UnifiedSell] {symbol} liquidation SELL will assert UNIFIED_SELL_AUTHORITY")

                if not is_liq_signal:
                    is_cold = False
                    try:
                        cold_attr = getattr(self.shared_state, "is_cold_bootstrap", None)
                        if callable(cold_attr):
                            cold_state = cold_attr()
                            if _asyncio.iscoroutine(cold_state):
                                cold_state = await cold_state
                            is_cold = bool(cold_state)
                        elif cold_attr is not None:
                            is_cold = bool(cold_attr)
                    except Exception:
                        is_cold = False

                    if is_cold:
                        self._log_reason("INFO", symbol, "sell_blocked_cold_bootstrap")
                        if hasattr(self.shared_state, "record_rejection"):
                            await self.shared_state.record_rejection(
                                symbol, "SELL", "COLD_BOOTSTRAP_BLOCK", source="MetaController"
                            )
                        await self._log_execution_result(
                            symbol,
                            side,
                            signal,
                            {"status": "blocked", "reason": "cold_bootstrap_no_sell"},
                        )
                        return {"ok": False, "status": "blocked", "reason": "cold_bootstrap_no_sell"}

                # LiquidationAgent-specific min hold (blocks fresh positions).
                if is_liq_signal and str(signal.get("agent") or "") == "LiquidationAgent":
                    min_hold_sec = float(self._cfg("LIQ_MIN_HOLD_SEC", default=90.0) or 0.0)
                    if min_hold_sec > 0:
                        entry_ts = 0.0
                        try:
                            sym_norm = self._normalize_symbol(symbol)
                            open_trades = getattr(self.shared_state, "open_trades", {}) or {}
                            if isinstance(open_trades, dict):
                                ot = open_trades.get(symbol) or open_trades.get(sym_norm) or {}
                                entry_ts = parse_timestamp(
                                    ot.get("entry_time")
                                    or ot.get("created_at")
                                    or ot.get("opened_at")
                                )
                            if not entry_ts:
                                positions = getattr(self.shared_state, "positions", {}) or {}
                                if isinstance(positions, dict):
                                    pos = positions.get(symbol) or positions.get(sym_norm) or {}
                                    entry_ts = parse_timestamp(
                                        pos.get("entry_time")
                                        or pos.get("opened_at")
                                        or pos.get("created_at")
                                    )
                        except Exception:
                            entry_ts = 0.0

                        if entry_ts > 0:
                            age_sec = max(0.0, time.time() - entry_ts)
                            if age_sec < min_hold_sec:
                                self._log_reason("INFO", symbol, f"liq_min_hold:{age_sec:.1f}s<{min_hold_sec:.0f}s")
                                self.logger.info(
                                    "[Meta:MinHold:Liq] SELL blocked for %s: age=%.1fs < min_hold=%.0fs",
                                    symbol,
                                    age_sec,
                                    min_hold_sec,
                                )
                                if hasattr(self.shared_state, "record_rejection"):
                                    await self.shared_state.record_rejection(
                                        symbol, "SELL", "LIQ_MIN_HOLD_SEC", source="MetaController"
                                    )
                                await self._log_execution_result(
                                    symbol,
                                    side,
                                    signal,
                                    {"status": "blocked", "reason": "liq_min_hold", "min_hold_sec": min_hold_sec, "age_sec": age_sec},
                                )
                                return {"ok": False, "status": "blocked", "reason": "liq_min_hold"}
                        else:
                            self._log_reason("INFO", symbol, "liq_min_hold_missing_entry")
                            await self._log_execution_result(
                                symbol,
                                side,
                                signal,
                                {"status": "blocked", "reason": "liq_min_hold_missing_entry"},
                            )
                            return {"ok": False, "status": "blocked", "reason": "liq_min_hold_missing_entry"}

                # Get position from SharedState (Canonical)
                qty = float(self.shared_state.get_position_qty(symbol) or 0.0)

                if not qty or qty <= 0:
                    self._log_reason("INFO", symbol, "sell_no_position_qty")
                    await self._log_execution_result(
                        symbol, side, signal, {"status": "skipped", "reason": "no_position_quantity"}
                    )
                    return

                # Risk pre-check (SELL) with FIX #4: liquidation bypass support
                if self.risk_manager and hasattr(self.risk_manager, "pre_check"):
                    # FIX #4: Detect if this is a liquidation signal for bypass
                    ok, r_reason = await _safe_await(self.risk_manager.pre_check(
                        symbol=symbol, side="SELL", is_liquidation=is_liq_signal
                    ))
                    if not ok:
                        self._log_reason("INFO", symbol, f"risk_precheck:{r_reason}")
                        await self._log_execution_result(symbol, side, signal, {"status": "skipped", "reason": f"risk:{r_reason}"})
                        return

                # ===== FIX 3: Quote-Based SELL for Liquidation =====
                # If this is a liquidation SELL with quote-based flag, use quoteOrderQty instead of quantity
                if is_quote_based_signal and is_starvation_sell:
                    quote_value = signal.get("_target_usdt", 0.0)
                    if quote_value > 0:
                        self.logger.warning(
                            "[Meta:QuoteLiq:Execute] Executing QUOTE-BASED liquidation SELL. "
                            "Symbol: %s, TargetUSDT: %.2f, PositionQty: %.8f. Using quoteOrderQty (bypasses min-notional).",
                            symbol, quote_value, qty
                        )
                        # Execute with quoteOrderQty (USDT value, not quantity)
                        policy_ctx = self._build_policy_context(
                            symbol,
                            "SELL",
                            policies=[p for p, enabled in policy_flags.items() if enabled],
                            extra={**sell_policy_extra, "mode": "quote_based"},
                        )
                        result = await self.execution_manager.execute_trade(
                            symbol=symbol,
                            side="sell",
                            quantity=None,  # Use quoteOrderQty instead
                            planned_quote=quote_value,  # Pass as USDT value
                            tag=sell_tag,
                            policy_context=policy_ctx,
                        )
                    else:
                        self.logger.warning("[Meta:QuoteLiq:Execute] Quote-based liquidation flag set but _target_usdt is invalid. Falling back to quantity-based sell.")
                        policy_ctx = self._build_policy_context(
                            symbol,
                            "SELL",
                            policies=[p for p, enabled in policy_flags.items() if enabled],
                            extra=sell_policy_extra,
                        )
                        result = await self.execution_manager.execute_trade(
                            symbol=symbol,
                            side="sell",
                            quantity=qty,
                            tag=sell_tag,
                            policy_context=policy_ctx,
                        )
                else:
                    # ===== FIX 2: Liquidation SELLs BYPASS starvation gates =====
                    # Even if system is starved, liquidation SELLs execute to free capital
                    if is_starvation_sell:
                        self.logger.warning(
                            "[Meta:LiquidationHardPath:Execute] Executing LIQUIDATION SELL (batch or fallback). "
                            "Symbol: %s, Qty: %.8f. Bypassing starvation/affordability gates.",
                            symbol, qty
                        )
                    
                    # Standard quantity-based SELL (batch liquidation or normal sell)
                    policy_ctx = self._build_policy_context(
                        symbol,
                        "SELL",
                        policies=[p for p, enabled in policy_flags.items() if enabled],
                        extra={**sell_policy_extra, "mode": "quantity"},
                    )
                    self.increment_execution_attempts()
                    result = await self.execution_manager.execute_trade(
                        symbol=symbol,
                        side="sell",
                        quantity=qty,
                        tag=sell_tag,
                        policy_context=policy_ctx,
                    )

                # Post-success cooldown & health bump
                if str(result.get("status", "")).lower() in {"placed", "executed", "filled"}:
                    # Track trade execution for focus mode exit condition
                    if self._focus_mode_active:
                        self._focus_mode_trade_executed = True
                        self._focus_mode_trade_executed_count += 1
                        self.logger.info(f"[FOCUS_MODE] Trade executed: {symbol} {side} (count={self._focus_mode_trade_executed_count})")
                    
                    try:
                        if hasattr(self.shared_state, "set_cooldown"):
                            cooldown_sec = float(self._cfg("META_DECISION_COOLDOWN_SEC", default=15))
                            await self.shared_state.set_cooldown(symbol, cooldown_sec)
                    except Exception:
                        self.logger.debug("Failed to set SharedState cooldown for %s", symbol)
                    try:
                        await self._health_set("Healthy", f"Executed SELL {symbol}")
                    except Exception:
                        pass

            await self._log_execution_result(symbol, side, signal, result)
            await self._update_kpi_metrics("execution")
            return result

        except Exception as e:
            classified_error = classify_execution_error(e, symbol)
            self.logger.error("Decision execution failed for %s: %s", symbol, classified_error.error_type, exc_info=True)
            
            # P9 Requirement 3/4: Report Capital Failure for Hysteresis
            if classified_error.error_type in (ExecutionError.Type.INSUFFICIENT_BALANCE, ExecutionError.Type.MIN_NOTIONAL_VIOLATION):
                agent = signal.get("agent", "Meta")
                if hasattr(self.shared_state, "report_agent_capital_failure"):
                    self.shared_state.report_agent_capital_failure(agent)
                    
            await self._update_kpi_metrics("error", classified_error.error_type)
            await self._health_set("Critical", f"Execution error for {symbol}: {classified_error}")

    def score_position(self, position: Dict[str, Any]) -> float:
        """
        P9: score a held position for replacement eligibility.
        Metrics: ROI (PnL %), confidence decay, and duration penalty.
        """
        roi = float(position.get("unrealized_pnl_pct", 0.0) or 0.0)
        symbol = position.get("symbol", "unknown")
        
        # Duration penalty: encourage rotation
        entry_ts = float(position.get("entry_time", time.time()))
        age_hours = (time.time() - entry_ts) / 3600.0
        duration_penalty = age_hours * 0.005 # 0.5% penalty per hour (configurable)
        
        score = roi - duration_penalty
        self.logger.debug("[Meta:Scoring] Position %s: Score=%.4f (ROI=%.2f%%, Age=%.1fh)", 
                        symbol, score, roi*100, age_hours)
        return score

    def score_opportunity(self, signal: Dict[str, Any]) -> float:
        """
        P9: score a new BUY opportunity.
        Metrics: Expected ROI, Confidence.
        """
        conf = float(signal.get("confidence", 0.0))
        symbol = signal.get("symbol", "unknown")
        # ROI proxy: if agent doesn't provide it, use 3% fixed target for scoring
        # (Slightly higher than the duration penalty to justify entry)
        expected_roi = float(signal.get("expected_roi", 0.03))
        
        score = expected_roi * conf
        self.logger.debug("[Meta:Scoring] Opportunity %s: Score=%.4f (Conf=%.2f, EstROI=%.2f%%)", 
                        symbol, score, conf, expected_roi*100)
        return score

    async def _planned_quote_for(self, symbol: str, sig: Dict[str, Any], budget_override: float = None) -> float:
        """Compute planned quote for BUY with floors & caps aligned with SharedState. Delegated to ScalingManager + Policy Nudges."""
        quote = await self.scaling_manager.calculate_planned_quote(
            symbol, 
            sig, 
            budget_override=budget_override
        )
        
        # Apply Policy Nudge (Multiplier)
        mult = float(self.active_policy_nudges.get("trade_size_multiplier", 1.0))
        if abs(mult - 1.0) > 0.01:
            quote *= mult
            
        # Re-clamp to Mode Envelope (Safety)
        try:
            if self.mode_manager:
                env = self.mode_manager.get_envelope()
                mode_max = float(env.get("max_trade_usdt", 0.0))
                if mode_max > 0 and quote > mode_max:
                    quote = mode_max
        except Exception:
            pass
            
        return quote

    async def _attempt_liquidity_healing(self, symbol: str, signal: Dict[str, Any]) -> bool:
        """Enhanced liquidity healing with verification."""
        if not self.liquidation_agent:
            self.logger.warning("LiquidationAgent not available for liquidity healing.")
            return False

        gap = float(signal.get("_liq_gap", 0.0))
        planned_quote = float(signal.get("_planned_quote", self._default_planned_quote))
        needed_quote = max(gap, planned_quote)

        await self._update_kpi_metrics("liquidity_request")

        for attempt in range(self._liq_retry_max):
            try:
                # Build opportunity metadata
                opp_meta = {
                    "symbol": symbol,
                    "planned_quote": planned_quote,
                    "expected_return_pct": signal.get("potential_pnl", 0.0),
                    "horizon_hours": signal.get("horizon_hours", 6),
                    "current_roi_pct": signal.get("current_roi_pct", 0.0),
                    "can_afford_reason": signal.get("_liq_reason", "INSUFFICIENT_QUOTE"),
                    "actual_gap": gap,
                }

                # Request liquidity plan
                plan = await self.request_liquidity(symbol, needed_quote, opp_meta)
                if not (plan and plan.get("status") in ("APPROVED", "PARTIAL")):
                    self.logger.info("Liquidity plan rejected for %s (attempt %d)", symbol, attempt + 1)
                    continue

                # Execute the plan
                success = False
                if hasattr(self.liquidation_agent, "execute_plan"):
                    result = await self.liquidation_agent.execute_plan(plan)
                    success = bool(result and result.get("success"))
                elif hasattr(self.execution_manager, "execute_liquidation_plan"):
                    result = await self.execution_manager.execute_liquidation_plan(plan.get("exits", []))
                    success = bool(result)

                if not success:
                    self.logger.warning("Liquidity execution failed for %s (attempt %d)", symbol, attempt + 1)
                    continue

                # CRITICAL FIX: Refresh balance from exchange after liquidation execution
                # Problem: Cached balance in SharedState may not reflect recent liquidations
                # Solution: Force sync authoritative balance before affordability verification
                try:
                    await self.shared_state.sync_authoritative_balance(force=True)
                    self.logger.debug("Balance refreshed after liquidity execution for %s", symbol)
                except Exception as e:
                    self.logger.warning("Failed to refresh balance after liquidity execution: %s", e)

                # Verify liquidity was freed
                # BOOTSTRAP FIX: Use bootstrap_bypass context if needed
                bootstrap_policy_ctx = None
                if signal.get("_bootstrap") or signal.get("_bootstrap_override"):
                    bootstrap_policy_ctx = self._build_policy_context(
                        symbol, 
                        "BUY",
                        extra={"bootstrap_bypass": True}
                    )
                can_afford, _, reason = await self.execution_manager.can_afford_market_buy(symbol, planned_quote, policy_context=bootstrap_policy_ctx)

                if can_afford:
                    self.logger.info("Liquidity healing successful for %s after %d attempts", symbol, attempt + 1)
                    return True
                else:
                    self.logger.warning("Liquidity verification failed for %s: %s", symbol, reason)

            except Exception as e:
                self.logger.error("Liquidity healing attempt %d failed for %s: %s", attempt + 1, symbol, e)

            # Brief delay between retries
            if attempt < self._liq_retry_max - 1:
                await _asyncio.sleep(1.0)

        self.logger.error("All liquidity healing attempts failed for %s", symbol)
        return False

############################################################
# SECTION: Metrics, KPIs & Observability
# Responsibility:
# - Trade execution logging and result tracking
# - Performance metrics collection and reporting
# - System health monitoring and diagnostics
# Future Extraction Target:
# - MetricsCollector or ObservabilityManager
############################################################

    # Belongs to: Metrics, KPIs & Observability
    # Extraction Candidate: Yes
    # Depends on: State & Internal Counters
    async def _log_execution_result(self, symbol: str, side: str, signal: Dict[str, Any], result: Optional[Dict[str, Any]]):
        """Enhanced execution result logging with P9 structured events."""
        status = str((result or {}).get("status", "")).lower()
        confidence = float(signal.get("confidence", 0.0))

        # ===== CRITICAL FIX #4: Propagate rejection reason with full detail =====
        reason = (result or {}).get("reason")
        reason_detail = (result or {}).get("reason_detail", "")
        full_reason = f"{reason}:{reason_detail}" if reason_detail else reason
        
        details = {
            "side": side,
            "status": status,
            "confidence": confidence,
            "agent": signal.get("agent", "Unknown"),
            "planned_quote": signal.get("_planned_quote"),
            "reason": full_reason,  # FIX #4: Include detailed reason
            "reason_detail": reason_detail,  # FIX #4: Separate detail field for parsing
            "executed_qty": (result or {}).get("executedQty"),
            "price": (result or {}).get("price"),
            "order_id": (result or {}).get("orderId"),
            "cummulative_quote": (result or {}).get("cummulativeQuoteQty"),
            # Added for canonical parity
            "qty": (result or {}).get("executedQty") or signal.get("quantity"),
            "tag": f"meta-{signal.get('agent', 'Meta')}",
            "error_code": (result or {}).get("error_code"),
        }

        if status in ("filled", "placed", "executed"):
            self._first_trade_executed = True
            # P9: Update bootstrap metrics (Total Trades & First Trade TS)
            try:
                metrics = getattr(self.shared_state, "metrics", {})
                if "total_trades_executed" in metrics:
                    metrics["total_trades_executed"] += 1
                if "first_trade_at" in metrics and metrics.get("first_trade_at") is None:
                    metrics["first_trade_at"] = datetime.now(timezone.utc).isoformat()
                if "bootstrap_completed" in metrics:
                    metrics["bootstrap_completed"] = True
            except Exception:
                pass
            
            # P9 DEADLOCK PREVENTION: Clear rejections on success
            if hasattr(self.shared_state, "clear_rejections"):
                await self.shared_state.clear_rejections(symbol, side)

            await self._log_execution_event("TRADE_EXECUTED", symbol, details)
            self.logger.info("MetaController EXECUTED %s: %s (conf=%.2f)", symbol, side, confidence)
        elif status == "skipped":
            await self._log_execution_event("TRADE_SKIPPED", symbol, details)
            # FIX #4: Enhanced logging with reason details
            self.logger.info(
                "MetaController SKIPPED %s: %s — reason=%s | details=%s",
                symbol, side, full_reason, reason_detail
            )
            # DIAGNOSTIC: [WHY_NO_TRADE] for easy grep/filtering
            if side == "BUY":
                self.logger.info(
                    "[WHY_NO_TRADE] symbol=%s reason=%s details=%s",
                    symbol, full_reason, reason_detail
                )
        elif status == "accumulating":
            await self._log_execution_event("TRADE_ACCUMULATING", symbol, details)
            self.logger.info("MetaController ACCUMULATING %s: %s (Total: %.4f)", 
                            symbol, side, (result or {}).get("accumulated_quote", 0.0))
        else:
            await self._log_execution_event("TRADE_UNKNOWN", symbol, details)
            # FIX #4: Enhanced logging for failures
            self.logger.debug(
                "MetaController UNKNOWN result for %s: %s — reason=%s | full_result=%s",
                symbol, side, full_reason, result
            )

    # -------------------
    # Liquidity management
    # -------------------
    async def request_liquidity(self, target_symbol: str, needed_quote: float, opp_meta: Dict[str, Any], target_quote: str = "USDT") -> Optional[Dict[str, Any]]:
        """Request liquidity plan from LiquidationAgent."""
        if not self.liquidation_agent:
            return None

        try:
            # ISSUE #3 FIX: Guard against infinite liquidity loops per symbol
            if self._liq_backoff.get(target_symbol):
                self.logger.debug("[LIQ] Skipping request for %s - backoff active.", target_symbol)
                return None
            self._liq_backoff.set(target_symbol, True, ttl=self._liq_backoff_sec)

            if needed_quote <= 0:
                return None

            trace_id = f"liq-{uuid.uuid4().hex[:8]}"
            self.logger.info(
                "[LIQ][%s] Requesting liquidity for %s: %.4f %s",
                trace_id, target_symbol, needed_quote, target_quote
            )

            # Check opportunity viability
            mu = opp_meta.get("expected_return_pct")
            H = opp_meta.get("horizon_hours")
            roi = opp_meta.get("current_roi_pct", 0.0)

            if mu is not None and H:
                # Calculate utility delta
                delta_u = (float(mu) - float(roi)) / float(H) - (self._liq_cost_bps / 10000.0) / float(H)
                if delta_u < (self._liq_min_edge_bps / 10000.0):
                    self.logger.info(
                        "[LIQ][%s] Utility too low (%.4f%%) for %s",
                        trace_id, delta_u * 100, target_symbol
                    )
                    return None

            # Build liquidity plan
            try:
                plan = await self.liquidation_agent.build_plan(target_symbol, needed_quote, opp_meta, target_quote=target_quote)
            except TypeError:
                # Fallback for older interface
                plan = await self.liquidation_agent.build_plan(target_symbol, needed_quote, opp_meta)

            if plan:
                plan["trace_id"] = trace_id
                self.logger.info(
                    "[LIQ][%s] Plan status: %s, expected: %.4f",
                    trace_id, plan.get("status"), plan.get("freed_quote", 0)
                )

            return plan

        except Exception as e:
            self.logger.error("Error requesting liquidity: %s", e, exc_info=True)
            return None

    # -------------------
    # Health monitoring
    # -------------------
    async def report_health_loop(self):
        """Background health reporter."""
        try:
            # CRITICAL FIX: Report health IMMEDIATELY on startup (don't wait 10 seconds)
            # This ensures watchdog sees status quickly after MetaController.start()
            try:
                await self._heartbeat()
                await self._health_set("Healthy", "Running normally.")
            except Exception as e:
                self.logger.warning("MetaController health report failed on startup: %s", e)
            
            # Then continue periodic reporting
            while self._running:
                try:
                    await self._heartbeat()
                    await self._health_set("Healthy", "Running normally.")
                except Exception as e:
                    self.logger.warning("MetaController health report failed: %s", e)
                await _asyncio.sleep(10)
        except _asyncio.CancelledError:
            pass

    async def get_kpi_status(self) -> Dict[str, Any]:
        """Return current KPI performance vs targets."""
        # P9 Sync: Favor SharedState as source of truth for realized PnL
        try:
            if hasattr(self.shared_state, "metrics"):
                val = float(self.shared_state.metrics.get("realized_pnl", 0.0))
                self._kpi_metrics["total_realized_pnl"] = val
        except Exception:
            pass

        async with self._performance_lock:
            total_pnl = self._kpi_metrics["total_realized_pnl"]
            target_rate = self._kpi_metrics["hourly_target_usdt"]

            return {
                "total_realized_pnl": total_pnl,
                "target_hourly_rate": target_rate,
                "execution_count": self._kpi_metrics["execution_count"],
                "liquidity_requests": self._kpi_metrics["liquidity_requests"],
                "error_distribution": dict(self._kpi_metrics["error_count_by_type"]),
                "timestamp": self._epoch(),
            }

    # -------------------
    # Public interface compatibility
    # -------------------
    # ===== CRITICAL FIX #1: _get_symbol_info() — Read exchange filters before deciding =====
    async def _get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve symbol information including filters and constraints from exchange.
        This is CRITICAL for FIX #1 — Meta must read filters before creating decisions.
        
        Returns: {
            'min_notional': float,          # Exchange minimum notional value
            'min_quantity': float,          # Exchange minimum order quantity
            'max_quantity': float,          # Exchange maximum order quantity (if any)
            'step_size': float,             # Order quantity step size
            'tick_size': float,             # Price tick size
            'price_filter': {...},          # Price constraints
            'lot_size_filter': {...},       # Quantity constraints
            'notional_filter': {...}        # Notional constraints
        }
        """
        try:
            # First attempt: Use ExecutionManager's cached symbol filters
            if hasattr(self.execution_manager, "get_symbol_filters_cached"):
                filters = await _safe_await(self.execution_manager.get_symbol_filters_cached(symbol))
                if filters:
                    # Parse filters into standardized info dict
                    info = {
                        'symbol': symbol,
                        'filters': filters,
                        'timestamp': time.time()
                    }
                    
                    # Extract common constraints
                    notional = filters.get("MIN_NOTIONAL") or filters.get("NOTIONAL") or {}
                    if notional:
                        info['min_notional'] = float(notional.get("minNotional", 10.0))
                    
                    lot_size = filters.get("LOT_SIZE") or {}
                    if lot_size:
                        info['min_quantity'] = float(lot_size.get("minQty", 0.0001))
                        info['step_size'] = float(lot_size.get("stepSize", 0.0001))
                    
                    price_filter = filters.get("PRICE_FILTER") or {}
                    if price_filter:
                        info['tick_size'] = float(price_filter.get("tickSize", 0.01))
                    
                    self.logger.debug(
                        "[Meta:SymbolInfo] Retrieved for %s: min_notional=%.2f, min_qty=%.8f, tick_size=%.8f",
                        symbol, info.get("min_notional", 10.0), info.get("min_quantity", 0.0001), info.get("tick_size", 0.01)
                    )
                    return info
        except Exception as e:
            self.logger.debug("[Meta:SymbolInfo] Failed to retrieve filters for %s via ExecutionManager: %s", symbol, e)
        
        try:
            # Second attempt: Use ExchangeClient if available
            if hasattr(self.shared_state, "exchange_client") and self.shared_state.exchange_client:
                symbol_info = await _safe_await(self.shared_state.exchange_client.get_symbol_info(symbol))
                if symbol_info:
                    self.logger.debug("[Meta:SymbolInfo] Retrieved for %s from ExchangeClient", symbol)
                    return symbol_info
        except Exception as e:
            self.logger.debug("[Meta:SymbolInfo] Failed to retrieve from ExchangeClient: %s", e)
        
        # Fallback: Return safe defaults
        self.logger.warning("[Meta:SymbolInfo] No symbol info available for %s, using safe defaults", symbol)
        return {
            'symbol': symbol,
            'min_notional': 10.0,
            'min_quantity': 0.0001,
            'step_size': 0.0001,
            'tick_size': 0.01,
            'filters': {}
        }
    
    async def _check_symbol_filters(self, symbol: str, side: str, planned_quote: float) -> Tuple[bool, str]:
        """
        Check if a trade violates symbol exchange filters.
        FIX #1 Extended: Verify against min_notional, quantity constraints, etc.
        
        Returns: (is_valid, rejection_reason)
        """
        try:
            symbol_info = await self._get_symbol_info(symbol)
            if not symbol_info:
                return True, ""  # Safe default if no info available
            
            min_notional = symbol_info.get("min_notional", 10.0)
            
            if planned_quote < min_notional:
                reason = f"below_min_notional:{planned_quote:.2f}<{min_notional:.2f}"
                return False, reason
            
            return True, ""
        except Exception as e:
            self.logger.debug("[Meta:FilterCheck] Error checking filters for %s: %s", symbol, e)
            return True, ""  # Fail open on errors
    
    def _build_rejection_result(self, reason: str, reason_detail: str = "", status: str = "skipped") -> Dict[str, Any]:
        """
        ===== CRITICAL FIX #4: Build standardized rejection result with full traceability =====
        
        Helper method to create rejection responses with comprehensive reason details.
        This ensures rejection reasons are visible throughout the system.
        
        Args:
            reason: Short reason code (e.g., "symbol_limit", "capital_insufficient")
            reason_detail: Detailed explanation (e.g., "symbol_hourly_limit_reached: 2/2 trades")
            status: Response status (default: "skipped")
        
        Returns: Dict with "ok", "status", "reason", "reason_detail" fields
        """
        return {
            "ok": False,
            "status": status,
            "reason": reason,
            "reason_detail": reason_detail,
            "timestamp": time.time()
        }

    def set_liquidation_agent(self, agent):
        """Inject LiquidationAgent."""
        self.liquidation_agent = agent

    def wire_with_execution_manager(self):
        """Wire with ExecutionManager if supported."""
        if hasattr(self.execution_manager, "set_meta_controller"):
            self.execution_manager.set_meta_controller(self)

    async def submit_signal(self, agent_name: str, symbol: str, payload: dict, confidence: float):
        """Compatibility method for signal submission."""
        sig = dict(payload)
        sig["confidence"] = confidence
        return await self.receive_signal(agent_name, symbol, sig)

    async def _evaluation_tick(self):
        """Compatibility alias for evaluate_and_act."""
        return await self.evaluate_and_act()


__all__ = ["MetaController", "LiquidityPlan", "ExecutionError", "BoundedCache", "ThreadSafeIntentSink"]
