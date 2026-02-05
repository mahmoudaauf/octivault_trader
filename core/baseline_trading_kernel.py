from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Literal
import time
# Back-compat shim
from .contracts import TradeIntent, OrderSide  # يمرر نفس الكلاس للاسم القديم

# ======= Enums / Aliases =======
Side = Literal["buy", "sell"]
Action = Literal["BUY", "SELL", "HOLD", "NO_DECISION"]
Venue = Literal["binance_spot"]

# ======= Contracts =======
@dataclass
class TradeIntent:
    symbol: str
    action: Action                    # BUY / SELL (never HOLD on timeout)
    confidence: float                 # 0..1
    agent: str                        # e.g., "MLForecaster", "DipSniper"
    tag: str = "signal"
    ttl_sec: int = 30                 # after this, Meta must drop it
    planned_quote: Optional[float] = None  # BUY budget in quote (USDT)
    planned_qty: Optional[float] = None    # explicit qty (for SELL/liquidation)
    created_ts: float = field(default_factory=lambda: time.time())

    def is_expired(self) -> bool:
        return (time.time() - self.created_ts) > self.ttl_sec

@dataclass
class ExecOrder:
    symbol: str
    side: Side
    venue: Venue = "binance_spot"
    tag: str = "meta"
    planned_quote: Optional[float] = None
    quantity: Optional[float] = None       # ExecutionManager will round ↓ to lotSize

@dataclass
class OrderResult:
    ok: bool
    symbol: str
    side: Side
    reason: str = ""
    txid: Optional[str] = None
    fills: Optional[List[Dict[str, Any]]] = None
    notional_quote: Optional[float] = None

# ======= Shared Gates & State (thin interface) =======
class Readiness:
    def __init__(self):
        self.market_data_ready = False
        self.accepted_symbols_ready = False
        self.trading_enabled = False

class KernelState:
    """
    Minimal surface the rest of the system uses.
    Your existing SharedState can implement/compose this.
    """
    def __init__(self):
        self.readiness = Readiness()
        self.valid_symbols: set[str] = set()
        self.latest_prices: Dict[str, float] = {}
        self.balances: Dict[str, float] = {}
        self.nav_quote: float = 0.0
        self.ghost_blacklist: set[str] = set()

    def set_nav_quote(self, v: float): self.nav_quote = max(0.0, float(v))
    def get_nav_quote(self) -> float: return float(self.nav_quote)

    def is_symbol_tradable(self, symbol: str) -> bool:
        return symbol in self.valid_symbols and symbol not in self.ghost_blacklist

# ======= Policy: timeouts & ghosts =======
def coerce_timeout_to_nodecision() -> Action:
    # Agents must emit NO_DECISION on timeout (never HOLD)
    return "NO_DECISION"

def blacklist_ghost(state: KernelState, symbol: str):
    state.ghost_blacklist.add(symbol)

# ======= Meta arbitration (reference implementation) =======
class MetaPolicy:
    def __init__(self, state: KernelState, min_conf: float = 0.60):
        self.s = state
        self.min_conf = float(min_conf)
        self.cooldown_sec = 15
        self._last_exec_ts: Dict[str, float] = {}

    def decide(self, intents: List[TradeIntent]) -> Optional[ExecOrder]:
        # Gates
        r = self.s.readiness
        if not (r.market_data_ready and r.accepted_symbols_ready and r.trading_enabled):
            return None

        # Keep only fresh, actionable intents
        intents = [i for i in intents
                   if not i.is_expired()
                   and i.action in ("BUY", "SELL")
                   and i.confidence >= self.min_conf
                   and self.s.is_symbol_tradable(i.symbol)]

        if not intents:
            return None

        # Pick the most recent highest-confidence
        intents.sort(key=lambda x: (x.confidence, x.created_ts), reverse=True)
        intent = intents[0]

        # Cooldown per symbol
        last = self._last_exec_ts.get(intent.symbol, 0)
        if time.time() - last < self.cooldown_sec:
            return None

        # Build ExecOrder
        if intent.action == "BUY":
            eo = ExecOrder(symbol=intent.symbol, side="buy", tag=f"meta:{intent.agent}",
                           planned_quote=intent.planned_quote)
        else:
            qty = intent.planned_qty  # for sell/liquidation; EM will compute if None
            eo = ExecOrder(symbol=intent.symbol, side="sell", tag=f"meta:{intent.agent}",
                           quantity=qty)

        self._last_exec_ts[intent.symbol] = time.time()
        return eo

# ======= ExecutionManager facade (exchange handled elsewhere) =======
class ExecutionFacade:
    """
    Thin facade the rest of the app should use.
    Your existing ExecutionManager can conform to this surface.
    """
    def __init__(self, state: KernelState, exchange_client, risk_manager=None):
        self.s = state
        self.ex = exchange_client
        self.risk = risk_manager

    async def place(self, order: ExecOrder) -> OrderResult:
        # Reject ghosts/invalid right here
        if not self.s.is_symbol_tradable(order.symbol):
            return OrderResult(ok=False, symbol=order.symbol, side=order.side, reason="invalid_symbol")

        # Risk required in LIVE (paper/sim may bypass)
        if self.risk and not self.risk.allow(order):
            return OrderResult(ok=False, symbol=order.symbol, side=order.side, reason="risk_veto")

        # Ensure symbol filters exist & compute qty if quote given
        try:
            await self.ex.ensure_symbol_filters_ready(order.symbol)
        except Exception as e:
            return OrderResult(ok=False, symbol=order.symbol, side=order.side, reason=f"filters_missing:{e}")

        if order.side == "buy" and order.planned_quote:
            res = await self.ex.place_market_order_quote(order.symbol, order.planned_quote, tag=order.tag)
        else:
            # SELL path: compute qty if not provided (sell all free balance)
            qty = order.quantity
            if qty is None:
                qty = await self.ex.get_sellable_free_qty(order.symbol)
            res = await self.ex.place_market_order_qty(order.symbol, qty, side=order.side, tag=order.tag)

        # Normalize result
        if res.get("ok"):
            return OrderResult(ok=True, symbol=order.symbol, side=order.side,
                               txid=res.get("orderId"), notional_quote=res.get("cummulativeQuoteQty"),
                               fills=res.get("fills"))
        else:
            return OrderResult(ok=False, symbol=order.symbol, side=order.side,
                               reason=res.get("reason", "unknown"))

# ======= Baseline kernel wiring (reference) =======
class TradingKernel:
    """
    Reference wiring used by main_phased/app_context.
    """
    def __init__(self, state: KernelState, meta: MetaPolicy, execf: ExecutionFacade):
        self.s = state
        self.meta = meta
        self.execf = execf

    async def on_agent_batch(self, intents: List[TradeIntent]):
        # drop HOLD/timeouts
        intents = [i for i in intents if i.action in ("BUY", "SELL")]
        order = self.meta.decide(intents)
        if not order:
            return {"accepted": 0, "reason": "no_decision"}
        result = await self.execf.place(order)
        return {"accepted": 1, "order": order.__dict__, "result": result.__dict__}
