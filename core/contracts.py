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
    symbol: str
    side: Union[OrderSide, str] = OrderSide.BUY
    quote: Optional[float] = None
    quantity: Optional[float] = None
    confidence: Optional[float] = None
    reason: str = ""
    agent: str = ""
    tag: str = ""
    ts_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    def __init__(self, symbol: str, side: Union[OrderSide, str] = OrderSide.BUY,
                 quote: Optional[float] = None, quantity: Optional[float] = None,
                 confidence: Optional[float] = None, reason: str = "",
                 agent: str = "", tag: str = "", ts_ms: Optional[int] = None, **legacy):
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
