# core/integration_guard.py
import time, uuid, logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional, Literal, Any, Tuple

logger = logging.getLogger("IntegrationGuard")

# ---------- Schema / versions ----------
@dataclass
class Versions:
    agent_signal: int = 3
    exec_request: int = 2
    exec_result: int = 1

class SchemaRegistry:
    REQUIRED = Versions().__dict__
    def __init__(self):
        self.reported: Dict[str, Dict[str,int]] = {}
    def report(self, component: str, versions: Dict[str,int]):
        self.reported[component] = versions
    def validate_all(self) -> list[str]:
        issues=[]
        for comp, vers in self.reported.items():
            for k, req in self.REQUIRED.items():
                if vers.get(k) != req:
                    issues.append(f"{comp}:{k} expected {req} got {vers.get(k)}")
        return issues

schema_registry = SchemaRegistry()

# ---------- Freshness helpers ----------
def too_old(ts_sec: float, budget_sec: float) -> Tuple[bool, float]:
    if not ts_sec: return (True, float("inf"))
    age = time.time() - float(ts_sec)
    return (age > budget_sec, age)

# ---------- Build standardized messages ----------
def new_op_id() -> str:
    return str(uuid.uuid4())

def build_agent_signal(
    shared_state: Any, symbol: str, action: Literal["BUY","SELL","HOLD"], confidence: float,
    agent_name: str, extra: Optional[dict]=None
) -> dict:
    md = getattr(shared_state, "market_data", {})
    ohlcv_last_ts = 0.0
    try:
        ohlcv_last_ts = float(md.get(symbol, {}).get("last_updated", 0.0))
    except Exception:
        pass
    payload = {
        "version": Versions().agent_signal,
        "op_id": new_op_id(),
        "ts": datetime.now(timezone.utc).isoformat(),
        "agent": agent_name,
        "symbol": symbol,
        "action": action.upper(),
        "confidence": float(confidence),
        "ohlcv_last_ts": ohlcv_last_ts,
    }
    if extra: payload.update(extra)
    return payload

# ---------- Pre-trade checks (shared by agents & execution) ----------
def compute_notional(price: float, qty: float, mode: Literal["QUOTE","BASE"]) -> float:
    return qty if mode == "QUOTE" else (price * qty)

def validate_trade(
    symbol: str, price: float, qty: float, mode: Literal["QUOTE","BASE"],
    wallet_free_usdt: float, filters: dict
) -> Tuple[bool, str]:
    if price <= 0: return False, "price_invalid"
    notional = compute_notional(price, qty, mode)
    min_notional = float(filters.get("minNotional", 0.0))
    if notional < min_notional: return False, "below_minNotional"
    step = float(filters.get("stepSize", 0.0))
    if mode == "BASE" and step > 0:
        # respect lot size step
        if abs((qty / step) - round(qty / step)) > 1e-9:
            return False, "qty_step_violation"
    if mode == "QUOTE" and notional > wallet_free_usdt:
        return False, "insufficient_usdt"
    return True, "ok"

# ---------- Watchdog utilities ----------
def max_market_age(shared_state: Any, symbols: list[str]) -> float:
    md = getattr(shared_state, "market_data", {})
    now = time.time()
    ages=[]
    for s in symbols or []:
        try:
            ts = float(md.get(s, {}).get("last_updated", 0.0))
            if ts > 0: ages.append(now - ts)
        except Exception:
            pass
    return max(ages) if ages else -1.0

# ---------- NullMetaController (safe sink) ----------
class NullMetaController:
    def __init__(self):
        self.logger = logging.getLogger("NullMetaController")
    async def forward(self, *_args, **_kwargs):
        self.logger.warning("Dropping signal: MetaController not wired.")
        return False

# ---------- Small filter refresh helper for MetaController ----------
async def ensure_symbol_filters_fresh(meta, symbol: str):
    try:
        ex = getattr(meta, "exchange_client", None)
        if ex and hasattr(ex, "ensure_symbol_info_loaded"):
            await ex.ensure_symbol_info_loaded(symbol)
        if ex and hasattr(ex, "get_symbol_filters"):
            f = await ex.get_symbol_filters(symbol)
            if f and "minNotional" in f:
                # Keep MetaController cache aligned if it exists
                if hasattr(meta, "_min_notional_cache"):
                    meta._min_notional_cache[symbol] = float(f["minNotional"])
    except Exception:
        logging.getLogger("IntegrationGuard").debug("Filter refresh failed for %s", symbol, exc_info=True)
