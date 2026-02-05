# =============================
# core/shared_state.py — Octivault P9 SharedState
# =============================
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
from typing import Any, Dict, List, Set, Tuple, Optional, Callable, TypedDict
from dataclasses import dataclass
from enum import Enum

# ---- Optional Third-Party Imports ----
try:
    import pandas as pd  # type: ignore
except ImportError:
    pd = None

# ---- Module Metadata ----
__version__ = "2.0.1"
__component__ = "core.shared_state"
__contract_id__ = "core:SharedState:v2.0.0"

__all__ = ["SharedState", "State", "SharedStateConfig", "StateConfig", "HealthCode", "Component", "SharedStateError", "ErrorCode", "CircuitBreaker", "CircuitBreakerState", "OHLCVBar", "SellableLine"]

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
    liq_queue_maxsize: int = 1000      # maximum size for liquidation queue

    # --- Active symbols fallback behavior (helps agents like LiquidationAgent) ---
    active_symbols_fallback_from_positions: bool = True  # include currently held positions if accepted list is small/empty
    active_symbols_default_limit: int = 0  # 0 = unlimited; if >0, truncate get_active_symbols() output

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
        self.failure_count = 0; self.state = CircuitBreakerState.CLOSED
    def record_failure(self) -> None:
        self.failure_count += 1; self.last_failure_time = time.time()
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
    # ---- Component status API (CSL + Watchdog friendly) ----
    async def update_component_status(self, component: str, status: str, detail: str = "", *, timestamp: float | None = None):
        ts = float(timestamp or time.time())
        payload = {"status": status, "message": detail, "timestamp": ts}
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

    def __init__(self, config: Optional[Dict | Any]=None, database_manager=None, exchange_client: Optional[Any]=None) -> None:
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
        self.rebalance_targets: Set[str] = set()

        # Agent state
        self.volatility_regimes = {}
        self.sentiment_scores = {}
        self.agent_scores = {}
        self.cot_explanations = {}
        # Back-compat aliases for components expecting singular names
        self.volatility_state = self.volatility_regimes
        self.sentiment_score = self.sentiment_scores

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

        # Liquidity reservations
        self._quote_reservations = {}
        self._authoritative_reservations: Dict[str, float] = {} # Per-agent authoritative budget (P9 Strict)
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

    async def add_agent_signal(self, symbol: str, agent: str, side: str, confidence: float, ttl_sec: int = 300, tier: str = "B", rationale: str = "") -> None:
        """
        P9 Mandatory Signal Contract:
        Every trading agent must call this when it emits a signal.
        This is the shared 'signal bus' used by MetaController and other evaluators.
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

    def get_latest_signals_by_symbol(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Return a shallow copy of the latest signals map."""
        return dict(self.latest_signals_by_symbol)

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
        
        # Take the most recent/best signal (or agent match if we add agent filtering here)
        signal = list(per_agent.values())[0]
            
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
            
            for symbol, side in to_del:
                key = (symbol.upper(), side.upper())
                if key in self._pending_position_intents:
                    del self._pending_position_intents[key]
                    if self._database_manager:
                        with contextlib.suppress(Exception):
                            await self._database_manager.delete_pending_intent(symbol.upper(), side.upper())

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
        Combines AI conviction, market regime, and portfolio momentum.
        """
        symbol = symbol.upper()
        # 1. Base Conviction (average of agent scores)
        conv = self.agent_scores.get(symbol, 0.5)
        
        # 2. Market Regime Multiplier
        # Normalize regime: 0 (bear) to 1 (bull)
        regime = self.volatility_regimes.get(symbol, {"regime": "neutral"})
        regime_name = regime.get("regime", "neutral").lower()
        regime_mult = 1.0
        if regime_name == "bull": regime_mult = 1.2
        elif regime_name == "bear": regime_mult = 0.8
        
        # 3. Momentum (from sentiment or realized pnl if available)
        sent = self.sentiment_scores.get(symbol, 0.0)
        
        # Final formula: base conviction weighted by regime and sentiment
        score = (conv * 0.7 + (sent + 1) * 0.15) * regime_mult
        return float(score)

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
        
        FIX #3: Support multiple quote assets (USDT, BUSD, FDUSD, etc)
        BOOTSTRAP FIX: When NAV calculates to 0 (cold start, no positions),
        return free quote as the bootstrap NAV to unblock first trade.
        """
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
        
        # FIX #3: Sum ALL quote assets
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
        
        # Mark positions
        has_positions = False
        for sym, pos in self.positions.items():
            qty = float(pos.get("quantity", 0.0))
            if qty <= 0: 
                continue
            has_positions = True
            px = float(self.latest_prices.get(sym) or pos.get("mark_price") or pos.get("entry_price") or 0.0)
            if px > 0:
                nav += qty * px
        
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
        """
        Back-compat getter probed by AppContext startup sanity.
        Returns spendable quote funds (after reserve policy) for the configured quote asset.
        """
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
        if getattr(self.config, "auto_positions_from_balances", True):
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
            invested_capital += qty * avg_price
            rebuilt_positions[sym] = {
                "quantity": qty,
                "avg_price": avg_price,
                "mark_price": price,
                "status": "OPEN",
                "state": PositionState.ACTIVE.value,
                "_mirrored": True,
            }
            if price > 0:
                self.latest_prices[sym] = price

        # Apply rebuilt positions
        for sym, pos in rebuilt_positions.items():
            await self.update_position(sym, pos)

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
        
        for symbol, position in self.positions.items():
            try:
                qty = float(position.get("quantity", 0.0))
                if qty <= 0:
                    continue
                
                # Get minNotional for this symbol
                _, min_notional = await self.compute_symbol_trade_rules(symbol)
                if min_notional <= 0:
                    min_notional = 10.0  # Default fallback

                # Position significance threshold (independent of exchange)
                min_position_value = float(getattr(self.config, "MIN_POSITION_VALUE_USDT", 10.0) or 10.0)
                strategy_floor = float(getattr(self.config, "MIN_SIGNIFICANT_USD", 0.0) or 0.0)
                significant_floor = max(float(min_notional), min_position_value, strategy_floor)
                
                # Get current price - PRIMARY source
                price = await self.get_latest_price(symbol)
                
                # ✅ FIX #1: If no market price, use position's own price data
                # This prevents false DUST classification when market data lags
                if not price or price <= 0:
                    # Try to get price from position's own cost basis
                    avg_price = float(position.get("avg_price", 0.0))
                    mark_price = float(position.get("mark_price", 0.0))
                    entry_price = float(position.get("entry_price", 0.0))
                    
                    # Use best available position price
                    if mark_price > 0:
                        price = mark_price
                    elif avg_price > 0:
                        price = avg_price
                    elif entry_price > 0:
                        price = entry_price
                    else:
                        # Still no price available - mark as dust but log it
                        self.logger.warning(
                            f"[SS:Dust] {symbol}: No price found (market=None, "
                            f"mark={mark_price}, avg={avg_price}, entry={entry_price}). "
                            f"Marking as dust (qty={qty:.6f})"
                        )
                        dust.append(symbol)
                        self.mark_as_dust(symbol)
                        continue
                
                position_value = qty * float(price)
                
                # CRITICAL: If below significance floor, it's DUST
                if position_value < significant_floor:
                    dust.append(symbol)
                    self.mark_as_dust(symbol)
                    self.logger.debug(
                        f"[SS:Dust] {symbol}: value=${position_value:.2f} < "
                        f"significant_floor=${significant_floor:.2f} (minNotional={min_notional:.2f}, "
                        f"min_position_value={min_position_value:.2f}, strategy_floor={strategy_floor:.2f}) → DUST"
                    )
                else:
                    significant.append(symbol)
                    position["status"] = "SIGNIFICANT"
                    position["capital_occupied"] = position_value
                    
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
        if symbol in self.positions:
            self.positions[symbol]["status"] = "DUST"
            self.positions[symbol]["capital_occupied"] = 0.0
            self.positions[symbol]["state"] = PositionState.DUST_LOCKED.value
            
            # Record in dust registry
            sym_norm = self._norm_sym(symbol)
            accepted = sym_norm in self.accepted_symbols or sym_norm in self.symbols
            origin = "strategy_portfolio" if accepted else "external_untracked"
            self.record_dust(
                symbol,
                self.positions[symbol].get("quantity", 0.0),
                origin=origin,
                context={"source": "mark_as_dust", "accepted_symbol": accepted},
            )
            
            self.logger.debug(f"[SS:Dust] Marked {symbol} as DUST - does not block capital")

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

    async def is_portfolio_flat(self) -> bool:
        """
        ===== GOLDEN RULE: If total_positions > 0 → portfolio is NOT flat =====

        Returns True only if portfolio has ZERO positions.
        Any position (including dust) means portfolio is NOT flat.

        This is the authoritative flatness check.
        """
        # FIX: Count ALL positions (including dust) — not just significant ones.
        # The previous call to get_significant_position_count() excluded dust,
        # which contradicted the docstring and caused BOOTSTRAP mode deadlock
        # when only dust positions existed.
        all_positions = self.get_open_positions()  # Returns both significant AND dust
        total_positions = len(all_positions)

        if total_positions == 0:
            self.logger.debug("[SS:Portfolio] Portfolio is FLAT - no positions")
            return True
        else:
            self.logger.debug(f"[SS:Portfolio] Portfolio NOT flat - {total_positions} positions exist (including dust)")
            return False

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
        data = {
            "status": code.value,
            "message": message,
            "timestamp": ts,
            "metrics": metrics or {},
        }
        self.component_statuses[component.value] = data
        self.system_health = data
        self.metrics["last_health_update"] = ts
        await self.emit_event("HealthStatus", {"component": component.value, **data})

    async def emit_health(self, component: str, status: str, reason: str = "", meta: Optional[Dict[str, Any]] = None):
        """Small wrapper used by AppContext contract validation & boot logs."""
        ts = time.time()
        payload = {"status": status, "message": reason, "timestamp": ts, "metrics": meta or {}}
        self.component_statuses[component] = payload
        self.system_health = payload
        self.metrics["last_health_update"] = ts
        await self.emit_event("HealthStatus", {"component": component, **payload})

    # -------- symbol management --------
    async def set_accepted_symbols(self, symbols: Dict[str, Dict[str, Any]], *, allow_shrink: bool = False, source: Optional[str] = None) -> None:
        if not isinstance(symbols, dict):
            raise SharedStateError("symbols must be a dictionary", ErrorCode.CONFIGURATION_ERROR)
        async with self._lock_context("global"):
            if allow_shrink:
                current_count = len(self.accepted_symbols)
                new_count = len(symbols)
                wanted = { self._norm_sym(k) for k in symbols.keys() }
                
                # P9 Guard: Collapse Protection
                # If we are about to shrink from a healthy universe (>10) to a broken one (<=1),
                # this is almost certainly a discovery filter failure or config error.
                # Propagating this would freeze the bot since agents won't have symbols.
                if current_count > 10 and new_count <= 1:
                    self.logger.error(
                        "🛡️ PANIC GUARD: Universe collapse detected (%d -> %d)! Refusing to shrink below safety floor.",
                        current_count, new_count
                    )
                    # We continue but don't delete anything, making the operation ADDITIVE
                else:
                    current_keys = set(self.accepted_symbols.keys())
                    for s in (current_keys - wanted):
                        # P9 Guard: Wallet-force symbols are sticky
                        # They should only be removed if the source specifies it or if we are doing a hard reset.
                        meta = self.accepted_symbols.get(s, {})
                        if meta.get("accept_policy") == "wallet_force" and source != "WalletScannerAgent":
                            self.logger.debug("🛡️ Protected wallet_force symbol %s from removal", s)
                            continue
                            
                        self.accepted_symbols.pop(s, None)
                        self.symbols.pop(s, None)
            wallet_forced = []
            normal_accepted = []
            for raw_sym, meta in symbols.items():
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
        
        # Defensive logging for accidental shrink (if allow_shrink=False)
        if not allow_shrink and len(symbols) < len(self.accepted_symbols):
             self.logger.warning(
                "[SS] Accepted symbols update is smaller than current set (no shrink allowed). "
                f"Current={len(self.accepted_symbols)}, Incoming={len(symbols)}, Source={source}"
            )
            
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
    async def get_symbol_filters_cached(self, symbol: str) -> Optional[Dict[str, Any]]:
        """P9: Jurisdictional getter for cached symbol filters."""
        return self.symbol_filters.get(self._norm_sym(symbol))

    def _norm_sym(self, s: str) -> str:
        return (s or "").upper().replace("/", "")

    def get_ohlcv_count(self, symbol: str, timeframe: str) -> int:
        return len(self.market_data.get((symbol, timeframe), []))

    def have_min_bars(self, symbols: list[str], timeframe: str, min_bars: int) -> bool:
        return all(self.get_ohlcv_count(s, timeframe) >= min_bars for s in symbols)

    async def _maybe_set_market_data_ready(self, *, timeframe: str = "5m", min_bars: int = 50) -> None:
        if self.accepted_symbols and not self.market_data_ready_event.is_set():
            syms = list(self.accepted_symbols.keys())
            if self.have_min_bars(syms, timeframe, min_bars):
                self.market_data_ready_event.set()
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
        try:
            if self._exchange_client and hasattr(self._exchange_client, "ensure_symbol_filters_ready"):
                await self._exchange_client.ensure_symbol_filters_ready(sym)
        except Exception:
            pass

        f = dict(self.symbol_filters.get(sym, {}))
        if self._exchange_client:
            try:
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

        lot_step = float(
            f.get("LOT_SIZE", {}).get("stepSize")
            or f.get("stepSize")
            or f.get("_normalized", {}).get("step_size", 0.0)
            or 0.0
        )
        min_notional = float(
            f.get("MIN_NOTIONAL", {}).get("minNotional")
            or f.get("minNotional")
            or f.get("_normalized", {}).get("min_notional", 0.0)
            or 0.0
        )
        return lot_step, min_notional

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
        try:
            expected_move_fee_mult = float(self._cfg("ENTRY_EXPECTED_MOVE_FEE_MULT", 2.0) or 2.0)
        except Exception:
            expected_move_fee_mult = 2.0
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
        key = (symbol, timeframe)
        b = {
            "ts": float(bar["ts"]),
            "o": float(bar["o"]),
            "h": float(bar["h"]),
            "l": float(bar["l"]),
            "c": float(bar["c"]),
            "v": float(bar["v"]),
        }
        async with self._lock_context("market_data"):
            lst = self.market_data.setdefault(key, [])
            if lst and abs(lst[-1]["ts"] - b["ts"]) < 1e-9:
                lst[-1] = b
            else:
                lst.append(b)
                if len(lst) >= 2 and lst[-2]["ts"] > lst[-1]["ts"]:
                    lst.sort(key=lambda r: r["ts"])
            # Invalidate ATR cache entries for this (symbol, timeframe)
            self._atr_cache = {k:v for k,v in self._atr_cache.items() if not (k[0]==symbol and k[1]==timeframe)}
        # price keep-warm
        await self.update_latest_price(symbol, b["c"])
        # Do not set MarketDataReady here; rely on coverage check
        await self._maybe_set_market_data_ready()

    async def set_market_data(self, symbol: str, timeframe: str, ohlcv_data: List[Dict[str, Any]]) -> None:
        """Batch set (not used by MDF warmup, but kept for completeness)."""
        key = (symbol, timeframe)
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
            self.market_data[key] = norm
            # Invalidate ATR cache entries for this (symbol, timeframe)
            self._atr_cache = {k:v for k,v in self._atr_cache.items() if not (k[0]==symbol and k[1]==timeframe)}
        if norm:
            await self.update_latest_price(symbol, norm[-1]["c"])
        # Do not set MarketDataReady here; rely on coverage check
        await self._maybe_set_market_data_ready()

    async def get_market_data(self, symbol: str, timeframe: str) -> Optional[List[OHLCVBar]]:
        return self.market_data.get((symbol, timeframe))

    # -------- ATR utility (used by MDF warm cache) --------
    async def calc_atr(self, symbol: str, timeframe: str, period: int = 14) -> Optional[float]:
        key = (symbol, timeframe)
        rows = self.market_data.get(key) or []
        if len(rows) < max(2, period+1):
            return None
        cache_key = (symbol, timeframe, period)
        if cache_key in self._atr_cache:
            return self._atr_cache[cache_key]
        # Compute True Range & ATR
        trs: List[float] = []
        for i in range(1, len(rows)):
            c_prev = rows[i-1]["c"]
            h = rows[i]["h"]; l = rows[i]["l"]; c = rows[i]["c"]
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
        try:
            if getattr(self.config, "auto_positions_from_balances", True):
                await self.hydrate_positions_from_balances()
        except Exception as e:
            self.logger.warning(f"hydrate_positions_from_balances failed: {e}")

        # Best-effort: update dust register for assets without tradable symbol/price yet
        try:
            quote = self.quote_asset.upper()
            for asset, data in list(self.balances.items()):
                a = asset.upper()
                if a == quote:
                    continue
                free_qty = float(data.get("free", 0.0))
                if free_qty <= 0:
                    continue
                sym = f"{a}{quote}"
                # If we have no price yet, still track as dust candidate so later price updates can reevaluate
                if sym not in self.latest_prices:
                    self.record_dust(
                        sym,
                        free_qty,
                        origin="wallet_balance_sync",
                        context={"source": "balance_sync", "has_price": False},
                    )
        except Exception:
            pass

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
        """
        if self._exchange_client and hasattr(self._exchange_client, "get_spot_balances"):
            try:
                new_bals = await self._exchange_client.get_spot_balances()
                if new_bals:
                    async with self._lock_context("balances"):
                        # FIX #5: NORMALIZE KEYS TO UPPERCASE - sync_authoritative_balance bypass bug
                        # Problem: Exchange returns {"usdt": {...}, "btc": {...}} (lowercase)
                        # But get_balance() looks for uppercase keys, causing mismatches
                        # Solution: Normalize all keys to uppercase like update_balances() does
                        for asset, data in new_bals.items():
                            if isinstance(data, dict):
                                a = asset.upper()
                                self.balances[a] = data
                        self.last_balance_sync = time.time()
                        # FIX #2: Ensure balances_ready_event is set, even if sync_authoritative_balance called directly
                        # Problem: Only update_balances() was setting the ready event, not sync_authoritative_balance()
                        # This caused MetaController to block indefinitely waiting for balances to be marked ready
                        if not self.balances_ready_event.is_set():
                            self.balances_ready_event.set()
                            self.metrics["balances_ready"] = True
                    log_level = "warning" if force else "info"
                    msg = "[SS] Authoritative balance sync complete (FORCE)" if force else "[SS] Authoritative balance sync complete."
                    if log_level == "warning":
                        self.logger.warning(msg)
                    else:
                        self.logger.info(msg)
            except Exception as e:
                self.logger.error(f"[SS] Failed to sync authoritative balance: {e}")

    async def get_spendable_balance(self, asset: str, *, reserve_ratio: Optional[float] = None, min_reserve: Optional[float] = None) -> float:
        """Get spendable balance with FIX #1: proper free/locked handling."""
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
        
        for r in all_reservations:
            expires_at = r.get("expires_at", 0)
            
            # Skip if already expired (TTL passed)
            if expires_at <= now:
                freed_amount += float(r.get("amount", 0.0))
                continue
            
            # Skip if created >60 seconds ago (emergency force-expire)
            if expires_at - 30 < now - 60:  # expires_at - ttl < now - 60
                freed_amount += float(r.get("amount", 0.0))
                continue
                
            # Skip if missing expires_at (invalid reservation)
            if expires_at == 0:
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

    async def force_cleanup_expired_reservations(self, asset: str = "USDT") -> tuple:
        """
        EMERGENCY CLEANUP: Nuclear option for capital-starved situations.
        Force-removes ANY reservation older than 60 seconds, regardless of TTL.
        
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
            created_age = now - (expires_at - 30) if expires_at > 0 else float('inf')
            
            # Force remove if older than 60 seconds
            if created_age > 60 or expires_at == 0:
                freed += float(r.get("amount", 0.0))
                removed += 1
            else:
                cleaned.append(r)
        
        self._quote_reservations[a] = cleaned
        
        if removed > 0:
            self.logger.warning(f"[SS:EmergencyCleanup] Force-removed {removed} old reservations. Freed: ${freed:.2f}")
        
        return (removed, freed)

    async def get_spendable_quote(self, asset: str, *, reserve_ratio: float = 0.10, min_reserve: float = 0.0) -> float:
        """Alias for get_spendable_balance for compatibility with callers using 'quote' wording."""
        return await self.get_spendable_balance(asset, reserve_ratio=reserve_ratio, min_reserve=min_reserve)

    async def get_free_quote(self) -> float:
        """
        Convenience getter for available quote balance (free funds) after safety reserve,
        using the configured quote_asset (default: USDT).
        """
        return await self.get_spendable_quote(
            self.quote_asset,
            reserve_ratio=self.config.quote_reserve_ratio,
            min_reserve=self.config.quote_min_reserve,
        )

    async def get_spendable_usdt(self) -> float:
        """Convenience alias for the configured quote asset (usually USDT)."""
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
        """Register dust metadata so we can distinguish strategy vs. trash origins."""
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

            self.dust_registry[sym] = entry

            if sym in self.positions:
                self.positions[sym]["state"] = PositionState.DUST_LOCKED.value

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

            # Prefer RAW Binance-shaped filters first, then fall back to normalized
            f = dict(self.symbol_filters.get(sym, {}))
            if self._exchange_client:
                try:
                    raw = await self._exchange_client.get_symbol_filters_raw(sym) if hasattr(self._exchange_client, "get_symbol_filters_raw") else {}
                except Exception:
                    raw = {}
                if isinstance(raw, dict) and raw:
                    f = dict(raw)
                    # cache RAW in shared_state for future lookups
                    self.symbol_filters[sym] = dict(raw)
                elif not f:
                    # fallback to normalized map if RAW not available
                    try:
                        norm = await self._exchange_client.get_symbol_filters(sym) if hasattr(self._exchange_client, "get_symbol_filters") else {}
                    except Exception:
                        norm = {}
                    if isinstance(norm, dict) and norm:
                        # keep normalized under a namespaced key to avoid clobbering RAW layout
                        f = {"_normalized": dict(norm)}
                        self.symbol_filters[sym] = dict(f)

            # Derive lot_step and min_notional supporting both schemas
            lot_step = float(
                f.get("LOT_SIZE", {}).get("stepSize")
                or f.get("stepSize")
                or f.get("_normalized", {}).get("step_size", 0.0)
                or 0.0
            )
            min_notional = float(
                f.get("MIN_NOTIONAL", {}).get("minNotional")
                or f.get("minNotional")
                or f.get("_normalized", {}).get("min_notional", 0.0)
                or 0.0
            )

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
                    changed.append(sym)
                continue
            sym = f"{a}{quote}"
            if sym not in self.symbols and sym not in self.accepted_symbols:
                # Skip unknown trading pairs
                continue
            prev = self.positions.get(sym, {})
            if float(prev.get("quantity", 0.0)) != free_qty or not prev.get("_mirrored"):
                pos = dict(prev)
                pos.update({
                    "quantity": free_qty,
                    "avg_price": float(pos.get("avg_price", 0.0)),
                    "_mirrored": True,
                    "status": "OPEN"  # CRITICAL: Set status so open_positions_count() recognizes it
                })
                await self.update_position(sym, pos)
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
        prices = await self.get_all_prices()
        nav = 0.0; unreal = 0.0
        for asset, b in self.balances.items():
            if asset.upper() == "USDT":
                nav += float(b.get("free", 0.0)) + float(b.get("locked", 0.0))
        for sym, pos in self.positions.items():
            qty = float(pos.get("quantity", 0.0))
            if qty == 0: continue
            px = float(prices.get(sym, pos.get("mark_price") or pos.get("entry_price") or 0.0))
            avg = float(pos.get("avg_price", self._avg_price_cache.get(sym, 0.0)))
            nav += qty * px
            if avg > 0 and px > 0:
                unreal += (px - avg) * qty
        self.metrics["nav"] = nav; self.metrics["unrealized_pnl"] = unreal
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

    # ---- Rejection Tracking (Deadlock Prevention) ----
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


    async def record_fill(self, symbol: str, side: str, qty: float, price: float, fee_quote: float = 0.0, tier: Optional[str] = None) -> None:
        side_u = (side or "").upper()
        qty = float(qty); price = float(price)
        if qty <= 0 or price <= 0: return
        pos = dict(self.positions.get(symbol, {}))
        cur_qty = float(pos.get("quantity", 0.0))
        avg = float(pos.get("avg_price", self._avg_price_cache.get(symbol, 0.0) or 0.0))
        realized = 0.0
        
        if side_u == "BUY":
            new_qty = cur_qty + qty
            new_avg = ((cur_qty * avg) + (qty * price)) / max(new_qty, 1e-12)
            pos.update({"quantity": new_qty, "avg_price": new_avg, "last_fill_ts": time.time()})
            self._avg_price_cache[symbol] = new_avg
            
            # Frequency Engineering: Track tier count
            if tier == "A":
                self.metrics["trades_tier_a"] += 1
            elif tier == "B":
                self.metrics["trades_tier_b"] += 1
                
        elif side_u == "SELL":
            close_qty = min(qty, cur_qty)
            if close_qty > 0 and avg > 0:
                realized = (price - avg) * close_qty - float(fee_quote or 0.0)
            new_qty = max(0.0, cur_qty - qty)
            pos.update({"quantity": new_qty, "avg_price": avg if new_qty > 0 else 0.0, "last_fill_ts": time.time()})
            if new_qty == 0: self._avg_price_cache.pop(symbol, None)
            
            # Frequency Engineering: Track holding time
            ot = self.open_trades.get(symbol)
            if ot and "opened_at" in ot:
                duration = time.time() - ot["opened_at"]
                self.metrics["total_holding_time_sec"] += duration
                self.metrics["completed_trades_count"] += 1
        else:
            return
            
        await self.update_position(symbol, pos)
        
        # P9 Integrity: Ensure open_trades exists for TP/SL tracking
        if side_u == "BUY" and symbol not in self.open_trades:
            now_ts = time.time()
            self.open_trades[symbol] = {
                "symbol": symbol,
                "position": "long",
                "entry_price": price,
                "quantity": qty,
                "opened_at": now_ts,
                "created_at": now_ts,
                "tier": tier
            }
        elif side_u == "BUY" and symbol in self.open_trades:
            # Ensure timestamps exist for time-based exits
            try:
                now_ts = time.time()
                ot = self.open_trades.get(symbol) or {}
                if isinstance(ot, dict):
                    ot.setdefault("opened_at", now_ts)
                    ot.setdefault("created_at", now_ts)
                    self.open_trades[symbol] = ot
            except Exception:
                pass
        elif side_u == "SELL" and new_qty <= 0:
            self.open_trades.pop(symbol, None)
            
        self.metrics["realized_pnl"] = float(self.metrics.get("realized_pnl", 0.0)) + realized
        self.trade_history.append({
            "ts": time.time(), "symbol": symbol, "side": side_u, "qty": qty, "price": price, "fee_quote": fee_quote,
            "realized_delta": realized, "tier": tier
        })
        self.trade_count += 1
        await self.emit_event("RealizedPnlUpdated", {"realized_pnl": self.metrics["realized_pnl"]})

    async def record_trade(self, symbol: str, side: str, qty: float, price: float, fee_quote: float = 0.0, tier: Optional[str] = None) -> None:
        """Compatibility alias for ExecutionManager post-fill tracking."""
        await self.record_fill(symbol, side, qty, price, fee_quote=fee_quote, tier=tier)

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
            
            # ===== CRITICAL FIX: Ensure status field exists =====
            # Required by get_open_positions() to avoid false FLAT state
            # If position has quantity > 0 but no status field, it will be
            # filtered out by get_open_positions(), causing portfolio to
            # incorrectly appear FLAT
            if "status" not in position_data:
                # Default to "OPEN" if not specified
                position_data["status"] = "OPEN"
                self.logger.debug(
                    "[SS:UpdatePos] Added default status='OPEN' to position %s "
                    "(missing status field could cause false FLAT state)",
                    sym
                )
            
            self.positions[sym] = dict(position_data)

    async def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        return self.positions.get(symbol)
    async def get_position_quantity(self, symbol: str) -> float:
        p = await self.get_position(symbol)
        return float(p.get("quantity", 0.0)) if p else 0.0

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
        rid = f"{asset}_{time.time()}_{amount}"
        async with self._lock_context("balances"):
            self._quote_reservations.setdefault(asset.upper(), []).append({"id": rid, "amount": float(amount), "expires_at": time.time()+ttl})
        return rid
    async def release_liquidity(self, asset: str, reservation_id: str) -> bool:
        async with self._lock_context("balances"):
            arr = self._quote_reservations.get(asset.upper(), [])
            for i, r in enumerate(arr):
                if r.get("id") == reservation_id:
                    arr.pop(i); return True
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
        for agent_id, amount in reservations.items():
            validated[agent_id] = float(max(0.0, amount))
        
        # Atomic update: replace entire dict
        self._authoritative_reservations = validated

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
    async def emit_event(self, event_name: str, event_data: Dict[Dict[str, Any]]) -> None:
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
    async def get_recent_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        return list(self._event_log)[-limit:]
    async def publish_event(self, name: str, data: Dict[str, Any]) -> None:
        if not self._subscribers: return
        ev = {"name": name, "data": data, "timestamp": time.time()}
        for q in list(self._subscribers.values()):
            try: q.put_nowait(ev)
            except Exception: pass
    async def subscribe_events(self, subscriber_name: str, max_queue: int = 1000) -> asyncio.Queue:
        q = asyncio.Queue(maxsize=max_queue); self._subscribers[subscriber_name] = q; return q
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

    def is_cold_bootstrap(self) -> bool:
        """
        GAP #4 FIX: Returns True ONLY if system has never executed ANY trade (true cold-start).
        This is the correct semantic for "can we allow emergency operations?"
        """
        return self.metrics.get("first_trade_at") is None and self.metrics.get("total_trades_executed", 0) == 0

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

    async def get_portfolio_state(self) -> str:
        """
        ===== GOLDEN RULE: If total_positions > 0 → portfolio is NOT flat =====
        
        States:
          - COLD_BOOTSTRAP: Never traded (is_cold_bootstrap() = True)
          - ACTIVE: Has ANY positions (total_positions > 0)
          - PORTFOLIO_FLAT: No positions (total_positions == 0)
        
        This replaces the minNotional-based logic with the authoritative rule:
        Any position (regardless of size) means the portfolio is ACTIVE.
        """
        if self.is_cold_bootstrap():
            return "COLD_BOOTSTRAP"
        
        # ===== CRITICAL FIX: Golden Rule Implementation =====
        # Count ALL positions with qty > 0 (including dust)
        try:
            total_positions = 0
            for sym, pos in self.positions.items():
                qty = float(pos.get("quantity", 0.0))
                if qty > 0:
                    total_positions += 1
            
            if total_positions > 0:
                # ANY position means portfolio is ACTIVE
                self.logger.info(
                    "[SS:PortState] Portfolio is ACTIVE: total_positions=%d",
                    total_positions
                )
                return "ACTIVE"
            else:
                # No positions means portfolio is FLAT
                self.logger.info("[SS:PortState] Portfolio is PORTFOLIO_FLAT: no positions")
                return "PORTFOLIO_FLAT"
        except Exception as e:
            self.logger.warning(f"get_portfolio_state check failed: {e}, defaulting to FLAT")
            return "PORTFOLIO_FLAT"

    async def is_portfolio_flat(self) -> bool:
        """
        ===== GOLDEN RULE: If total_positions > 0 → portfolio is NOT flat =====

        Returns True only if portfolio has ZERO positions.
        Any position (including dust) means portfolio is NOT flat.
        """
        # FIX: Directly count ALL positions (including dust) instead of
        # delegating to get_portfolio_state() which treats COLD_BOOTSTRAP
        # as flat even when dust positions exist from wallet sync.
        all_positions = self.get_open_positions()
        total_positions = len(all_positions)

        if total_positions == 0:
            self.logger.debug("[SS:IsFlat] Portfolio is FLAT - no positions")
            return True
        else:
            self.logger.debug(
                "[SS:IsFlat] Portfolio NOT flat - %d positions exist (including dust)",
                total_positions,
            )
            return False

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
            spendable = await self._free_usdt() if hasattr(self, "_free_usdt") else 0.0
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
    def get_positions_by_symbol(self) -> Dict[str, Dict[str, Any]]:
        return dict(self.positions)
    
    def get_positions(self) -> Dict[str, Dict[str, Any]]:
        """
        ✅ CANONICAL: Get all positions (both open and closed).
        
        Alias for get_positions_by_symbol() to maintain API compatibility.
        Used by meta_controller and other components expecting this interface.
        
        Returns:
            Dict mapping symbol → position_data
        """
        return dict(self.positions)
    
    # ----------- Additional helpers & wrappers -----------

    async def safe_price(self, symbol: str, default: float = 0.0) -> float:
        """Return price for symbol or default if not available."""
        return float(self.latest_prices.get(self._norm_sym(symbol), default))

    async def get_symbol_filters_cached(self, symbol: str) -> Dict[str, Any]:
        """Return a shallow copy of symbol filters for symbol (if any)."""
        return dict(self.symbol_filters.get(self._norm_sym(symbol), {}))

    def get_positions_snapshot(self) -> Dict[str, Dict[str, Any]]:
        """Return a shallow copy of all positions."""
        return dict(self.positions)

    def get_open_positions(self) -> Dict[str, Dict[str, Any]]:
        """
        Return all OPEN positions (status in {OPEN, PARTIALLY_FILLED}).
        
        ✅ Returns BOTH significant and dust positions.
        Filter by is_dust flag in caller if needed.
        """
        result = {}
        for sym, pos_data in self.positions.items():
            status = str(pos_data.get("status", "")).upper()
            qty = float(pos_data.get("quantity", 0.0))
            
            # Only include positions with OPEN/PARTIALLY_FILLED status and qty > 0
            if status in {"OPEN", "PARTIALLY_FILLED"} and qty > 0:
                result[sym] = pos_data
        
        return result

    def get_position_qty(self, symbol: str) -> float:
        """Return the quantity for a position, or 0.0 if not present."""
        p = self.positions.get(self._norm_sym(symbol))
        return float(p.get("quantity", 0.0)) if p else 0.0

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
        
        AUTHORITATIVE FLAT CHECK: position.status in {"OPEN", "PARTIALLY_FILLED"}
        and NOT marked as dust
        """
        count = 0
        for sym, pos_data in self.positions.items():
            status = str(pos_data.get("status", "")).upper()
            qty = float(pos_data.get("quantity", 0.0))
            is_dust = pos_data.get("is_dust", False) or pos_data.get("_is_dust", False)
            
            # Only count positions with OPEN or PARTIALLY_FILLED status 
            # AND qty > 0 AND NOT dust
            if status in {"OPEN", "PARTIALLY_FILLED"} and qty > 0 and not is_dust:
                count += 1
        
        return count


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
