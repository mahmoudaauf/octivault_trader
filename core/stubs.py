from dataclasses import dataclass
from typing import Optional, Any, Dict

@dataclass
class TradeIntent:
    symbol: str
    side: str
    qty_hint: Optional[float] = None
    quote_hint: Optional[float] = None
    agent: Optional[str] = None
    confidence: float = 0.0
    rationale: Optional[str] = None
    ttl_sec: int = 30
    tag: Optional[str] = None
    timeframe: Optional[str] = None

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

import asyncio
from typing import Callable

async def maybe_await(obj):
    if asyncio.iscoroutine(obj):
        return await obj
    return obj

async def maybe_call(obj: Any, method_name: str, *args, **kwargs):
    if not obj:
        return None
    fn = getattr(obj, method_name, None)
    if not callable(fn):
        return None
    res = fn(*args, **kwargs)
    if asyncio.iscoroutine(res):
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





