from dataclasses import dataclass
from typing import Optional, Any, Dict
import time

@dataclass
class TradeIntent:
    """
    Canonical domain object representing a validated trading decision approved for execution.
    
    Single source of truth for all trading intents across all agents:
    - MetaController intents
    - Liquidation intents
    - Compounding intents
    - Rebalancer intents
    - Arbitrage intents
    - Manual override intents
    
    ExecutionManager receives TradeIntent, not loose parameters.
    This eliminates coupling between controllers and execution logic.
    """
    symbol: str
    side: str
    quantity: Optional[float] = None
    planned_quote: Optional[float] = None
    confidence: float = 0.0
    trace_id: Optional[str] = None
    tier: Optional[str] = None
    is_liquidation: bool = False
    policy_context: Optional[Dict[str, Any]] = None
    agent: Optional[str] = None
    tag: str = "meta"
    reason: str = ""  # CRITICAL: Added to support _determine_execution_tier in MetaController
    timestamp: Optional[float] = None
    execution_mode: str = "live"  # "live" or "shadow"
    
    # Legacy hints (keep for backwards compatibility during transition)
    qty_hint: Optional[float] = None
    quote_hint: Optional[float] = None
    rationale: Optional[str] = None
    ttl_sec: int = 30
    timeframe: Optional[str] = None
    
    def __post_init__(self):
        """Ensure timestamp is set."""
        if self.timestamp is None:
            self.timestamp = time.time()

@dataclass
class ExecOrder:
    symbol: str
    side: str
    qty: float = 0.0
    price: Optional[float] = None
    type: str = "MARKET"
    tag: Optional[str] = None
    planned_quote: float = 0.0
    status: str = "NEW"
    id: Optional[str] = None

@dataclass
class MetaDecision:
    """
    Typed decision object with full source traceability and audit trail.
    
    Replaces the legacy tuple-based decision format (symbol, side, dict).
    
    Purpose:
    - Type-safe decision representation
    - Full traceability from intent → decision → order
    - Gate application tracking (which gates filtered, which passed)
    - Enriched context for execution and audit
    - Support for decision rejection reasons
    
    Usage:
        decision = MetaDecision(
            symbol="BTCUSDT",
            side="BUY",
            confidence=0.85,
            planned_quote=100.0,
            source_intent=trade_intent,
            trace_id="intent-123-abc",
            applied_gates=["min_confidence", "concentration"],
            rejection_reasons=[],
            execution_tier="immediate",
            enrichment={"dip_percent": 2.5},
            timestamp=time.time(),
            policy_context={"max_position": 1000},
            rationale="DipSniper: dip 2.5% below EMA",
        )
    """
    # Core decision
    symbol: str                                    # BTCUSDT
    side: str                                      # "BUY" or "SELL"
    confidence: float                              # 0.0-1.0
    planned_quote: float                           # USDT amount
    
    # Source traceability
    source_intent: 'TradeIntent'                   # Original intent that created this decision
    trace_id: str                                  # Unique ID for end-to-end tracing
    
    # Gate processing
    applied_gates: list = None                     # ["min_confidence", "concentration", "position_limit"]
    rejection_reasons: list = None                 # ["confidence_below_min", "position_limit_exceeded"]
    
    # Execution context
    execution_tier: str = "immediate"              # "immediate", "pending", "deferred", "rejected"
    
    # Enriched data
    enrichment: Dict[str, Any] = None              # {"dip_percent": 2.5, "bb_breach": True, ...}
    
    # Metadata
    timestamp: Optional[float] = None              # When decision was made
    policy_context: Optional[Dict[str, Any]] = None  # Policy parameters that influenced decision
    rationale: str = ""                            # Human-readable explanation
    
    def __post_init__(self):
        """Initialize defaults and validate."""
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.applied_gates is None:
            self.applied_gates = []
        if self.rejection_reasons is None:
            self.rejection_reasons = []
        if self.enrichment is None:
            self.enrichment = {}
        if self.policy_context is None:
            self.policy_context = {}
        
        # Validation
        if self.side not in ("BUY", "SELL"):
            raise ValueError(f"Invalid side: {self.side}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be 0.0-1.0, got {self.confidence}")
        if self.execution_tier not in ("immediate", "pending", "deferred", "rejected"):
            raise ValueError(f"Invalid execution_tier: {self.execution_tier}")
    
    @property
    def is_rejected(self) -> bool:
        """Check if this decision was rejected."""
        return self.execution_tier == "rejected"
    
    @property
    def is_approved(self) -> bool:
        """Check if this decision is approved for execution."""
        return not self.is_rejected
    
    def add_gate(self, gate_name: str) -> None:
        """Record that a gate was applied."""
        if gate_name not in self.applied_gates:
            self.applied_gates.append(gate_name)
    
    def add_rejection_reason(self, reason: str) -> None:
        """Record a rejection reason."""
        if reason not in self.rejection_reasons:
            self.rejection_reasons.append(reason)
        self.execution_tier = "rejected"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "symbol": self.symbol,
            "side": self.side,
            "confidence": self.confidence,
            "planned_quote": self.planned_quote,
            "source_intent_symbol": self.source_intent.symbol,
            "source_intent_agent": self.source_intent.agent,
            "trace_id": self.trace_id,
            "applied_gates": self.applied_gates,
            "rejection_reasons": self.rejection_reasons,
            "execution_tier": self.execution_tier,
            "enrichment": self.enrichment,
            "timestamp": self.timestamp,
            "policy_context": self.policy_context,
            "rationale": self.rationale,
        }


from typing import Callable
import inspect as _inspect

async def maybe_await(obj):
    if _inspect.isawaitable(obj):
        return await obj
    return obj

async def maybe_call(obj: Any, method_name: str, *args, **kwargs):
    if not obj:
        return None
    fn = getattr(obj, method_name, None)
    if not callable(fn):
        return None
    res = fn(*args, **kwargs)
    if _inspect.isawaitable(res):
        return await res
    return res

import time

class KernelState:
    def __init__(self):
        self.metrics = {}

class MetaPolicy:
    def __init__(self, state, min_conf=0.0):
        self.state = state
        self.min_conf = min_conf
        
    def evaluate(self, signal):
        # Basic pass-through policy
        conf = signal.get("confidence", 0.0)
        if conf >= self.min_conf:
            return True
        return False

async def is_fresh(shared_state, symbol, max_age_sec=300):
    if not shared_state:
        return False
    ts = shared_state._last_tick_timestamps.get(symbol)
    if ts:
        return (time.time() - ts) < max_age_sec
    return False

class BinanceAPIException(Exception):
    def __init__(self, message: str, code: Optional[int] = None):
        super().__init__(message)
        self.message = message
        self.code = code

class ExecutionError(Exception):
    pass

class HealthStatus:
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    STOPPED = "STOPPED"
    DEGRADED = "DEGRADED"
    CRITICAL = "CRITICAL"

def resilient_trade(component_name="Component", max_attempts=3):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            last_err = None
            for i in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_err = e
                    await asyncio.sleep(0.5 * (i + 1))
            raise last_err
        return wrapper
    return decorator




