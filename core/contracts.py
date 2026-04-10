# core/contracts.py
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union
import time

class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"

@dataclass
class TradeIntent:
    """
    Standard trade intent contract.
    
    Design Rules (P9 Architecture):
    ✅ AGENT SETS: symbol, side, quantity, quote, confidence, reason, agent, tag
    ❌ AGENT DOES NOT SET: planned_quote, trace_id, tier, is_liquidation, policy_context
    
    MetaController ENRICHES the intent with:
    - trace_id: Decision traceability link
    - tier: Execution tier (BOT_POSITION, RECOVERY, DUST_RECOVERY, etc.)
    - policy_context: Policy metadata dict (NOT agent-owned)
    
    ExecutionManager NEVER TRUSTS these fields — it uses SharedState for authority:
    - classification (from position.classification, not intent)
    """
    symbol: str
    side: Union[OrderSide, str] = OrderSide.BUY
    quote: Optional[float] = None
    quantity: Optional[float] = None
    confidence: Optional[float] = None
    reason: str = ""
    agent: str = ""
    tag: str = ""
    ts_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    
    # Execution metadata (set by MetaController/ExecutionManager, NOT agents)
    planned_quote: Optional[float] = None
    trace_id: Optional[str] = None
    tier: Optional[str] = None
    is_liquidation: bool = False
    policy_context: Optional[dict] = None

    def __init__(self, symbol: str, side: Union[OrderSide, str] = OrderSide.BUY,
                 quote: Optional[float] = None, quantity: Optional[float] = None,
                 confidence: Optional[float] = None, reason: str = "",
                 agent: str = "", tag: str = "", ts_ms: Optional[int] = None,
                 planned_quote: Optional[float] = None,
                 trace_id: Optional[str] = None,
                 tier: Optional[str] = None,
                 is_liquidation: bool = False,
                 policy_context: Optional[dict] = None,
                 **legacy):
        action = legacy.pop("action", None)
        if action is not None and side == OrderSide.BUY:
            side = action
        if isinstance(side, str):
            s = side.strip().lower()
            if s not in ("buy", "sell"):
                raise ValueError(f"Invalid side/action: {side}")
            side = OrderSide.BUY if s == "buy" else OrderSide.SELL

        object.__setattr__(self, "symbol", symbol.upper())
        object.__setattr__(self, "side", side)
        object.__setattr__(self, "quote", quote)
        object.__setattr__(self, "quantity", quantity)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "agent", agent)
        object.__setattr__(self, "tag", tag)
        object.__setattr__(self, "ts_ms", ts_ms if ts_ms is not None else int(time.time() * 1000))
        
        # Execution metadata defaults
        object.__setattr__(self, "planned_quote", planned_quote or quote)
        object.__setattr__(self, "trace_id", trace_id)
        object.__setattr__(self, "tier", tier)
        object.__setattr__(self, "is_liquidation", is_liquidation)
        object.__setattr__(self, "policy_context", policy_context or {})
