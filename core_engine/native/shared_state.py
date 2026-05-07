"""
Native L0: Lightweight in-memory state manager (replaces 1,200-line legacy SharedState)

This module provides minimal, focused state management for the trading system.
Trade-offs: No invariant checking, no event system, no quota reservations.
Those responsibilities are pushed to execution layer (L3).

Performance: ~75% less code than legacy, instant initialization (no Binance I/O).
"""

import asyncio
from dataclasses import dataclass
from typing import Optional

from .capital_policy import prune_reservations


@dataclass
class Position:
    """Single position snapshot"""

    symbol: str
    qty: float
    entry_price: float
    mark_price: float

    @property
    def unrealized_pnl_pct(self) -> float:
        """Unrealized P&L percentage"""
        if self.entry_price <= 0:
            return 0.0
        return ((self.mark_price - self.entry_price) / self.entry_price) * 100

    @property
    def position_value(self) -> float:
        """Position value in USDT"""
        return self.qty * self.mark_price


@dataclass
class Order:
    """Single order snapshot"""

    order_id: str
    symbol: str
    side: str  # "BUY" or "SELL"
    qty: float
    price: float
    status: str  # "PENDING", "FILLED", "CANCELED"
    timestamp_ms: int = 0


class NativeSharedState:
    """
    Minimal in-memory state replacement for legacy SharedState.

    Responsibilities:
    - Store current NAV (Net Asset Value in USDT)
    - Store positions by symbol
    - Store balance (free and invested)
    - Store prices (latest market prices)
    - Track accepted symbols for trading
    - Signal readiness when positions hydrated

    Not responsible for (moved to other layers):
    - Position invariant checking (execution layer)
    - Event emission (use callbacks)
    - Quota reservations (execution layer)
    - Symbol convergence (strategy layer)
    """

    def __init__(self):
        # Essential state
        self.nav_usdt: float = 0.0
        self.free_balance_usdt: float = 0.0
        self.invested_capital_usdt: float = 0.0
        self.balance: dict[str, float] = {}
        self.prices: dict[str, float] = {}
        self.market_data: dict[tuple[str, str], list[dict]] = {}
        self.market_data_ready: bool = False
        self._last_tick_timestamps: dict[str, float] = {}

        # Position tracking
        self.positions: dict[str, Position] = {}  # symbol -> Position
        self.open_orders: dict[str, Order] = {}  # order_id -> Order
        self.price_cache: dict[str, float] = {}  # symbol -> latest_price

        # Symbol tracking
        self.accepted_symbols: set[str] = set()
        self.dust_symbols: set[str] = set()

        # Hydration state (lazy initialization to avoid event loop requirement)
        self._ready_event: Optional[asyncio.Event] = None
        self._hydrated = False

        # Feedback loop state (ObjectiveFeedbackController + AdaptiveCapitalEngine)
        self.metrics: dict = {
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "session_elapsed_h": 0.0,
            "peak_nav": 0.0,
            "trades_in_window": 0,
            "win_rate_window": 0.5,
            "avg_fee_bps": 0.0,
            "avg_slippage_bps": 0.0,
            "avg_net_profit_bps": 0.0,
            "last_update_ts": 0.0,
        }
        self.session_anchor_nav: float = 0.0  # NAV at session start
        self.runtime_overrides: dict = {}  # OFC writes: confidence_floor, size_multiplier, etc.
        self.trading_halted: bool = False  # OFC kill-switch
        self.trade_history: dict[str, list] = {}  # symbol -> list of closed-trade records
        self._session_start_ts: float = 0.0  # Session start time for OFC elapsed calc
        self.current_mode: str = "BOOTSTRAP"
        self.current_mode_reason: str = ""
        self.exchange_throttled: bool = False
        self.exchange_throttle_reason: str = ""
        self.exchange_throttle_until_ts: float = 0.0
        self.quote_reservations: dict[str, list[dict]] = {}

    # ==================== NAV Management ====================

    def update_nav(self, nav: float):
        """Update NAV (single source of truth)"""
        self.nav_usdt = max(0.0, nav)

    def get_nav(self) -> float:
        """Get current NAV"""
        return self.nav_usdt

    def update_balance(self, free: float, invested: float):
        """Update balance breakdown"""
        self.free_balance_usdt = max(0.0, free)
        self.invested_capital_usdt = max(0.0, invested)
        # Auto-update NAV from balance
        self.nav_usdt = self.free_balance_usdt + self.invested_capital_usdt

    def update_balance_map(self, balances: dict[str, float]) -> None:
        """Store raw exchange/free balances and keep canonical USDT fields aligned."""
        self.balance = {str(k).upper(): float(v or 0.0) for k, v in (balances or {}).items()}
        free_usdt = float(self.balance.get("USDT", 0.0) or 0.0)
        invested = self.get_portfolio_value()
        self.update_balance(free_usdt, invested)

    # ==================== Position Management ====================

    def update_position(self, symbol: str, qty: float, entry: float, current: float):
        """Update position (no invariant checking)"""
        if qty > 1e-8:  # Minimal dust threshold
            self.positions[symbol] = Position(
                symbol=symbol, qty=qty, entry_price=entry, mark_price=current
            )
        else:
            # Position closed or dust
            self.positions.pop(symbol, None)

    def close_position(self, symbol: str):
        """Mark position as closed"""
        self.positions.pop(symbol, None)

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position by symbol"""
        return self.positions.get(symbol)

    def get_position_qty(self, symbol: str) -> float:
        """Return position quantity for object- or dict-backed positions."""
        pos = self.positions.get(symbol)
        if pos is None:
            return 0.0
        qty = getattr(pos, "qty", None)
        if qty is None and isinstance(pos, dict):
            qty = pos.get("qty", 0.0)
        return float(qty or 0.0)

    def get_position_quantity(self, symbol: str) -> float:
        """Compatibility alias for legacy callers."""
        return self.get_position_qty(symbol)

    def get_all_positions(self) -> dict[str, Position]:
        """Get all positions"""
        return self.positions.copy()

    def get_portfolio_value(self) -> float:
        """Total value of all positions"""
        total = 0.0
        for pos in self.positions.values():
            if isinstance(pos, dict):
                qty = float(pos.get("qty", 0.0) or 0.0)
                mark = float(
                    pos.get("mark_price", pos.get("current_price", pos.get("avg_price", 0.0)))
                    or 0.0
                )
                total += qty * mark
            else:
                total += float(getattr(pos, "position_value", 0.0) or 0.0)
        return total

    # ==================== Order Management ====================

    def add_order(self, order_id: str, symbol: str, side: str, qty: float, price: float):
        """Track new order"""
        self.open_orders[order_id] = Order(
            order_id=order_id, symbol=symbol, side=side, qty=qty, price=price, status="PENDING"
        )

    def mark_order_filled(self, order_id: str):
        """Mark order as filled"""
        if order_id in self.open_orders:
            self.open_orders[order_id].status = "FILLED"

    def mark_order_canceled(self, order_id: str):
        """Mark order as canceled"""
        if order_id in self.open_orders:
            self.open_orders[order_id].status = "CANCELED"

    def remove_order(self, order_id: str):
        """Remove order from tracking"""
        self.open_orders.pop(order_id, None)

    def get_open_order_count(self) -> int:
        """Count pending orders"""
        return sum(1 for o in self.open_orders.values() if o.status == "PENDING")

    # ==================== Price Management ====================

    def update_price(self, symbol: str, price: float):
        """Update latest price for symbol"""
        if price > 0:
            import time

            symbol_key = str(symbol or "").upper()
            now = time.time()
            self.price_cache[symbol_key] = price
            self.prices[symbol_key] = price
            self._last_tick_timestamps[symbol_key] = now

    def get_price(self, symbol: str) -> float:
        """Get latest price for symbol"""
        return self.price_cache.get(symbol, 0.0)

    # ==================== Symbol Management ====================

    def set_accepted_symbols(self, symbols: list[str]):
        """Set symbols to trade"""
        self.accepted_symbols = set(symbols)

    def add_accepted_symbol(self, symbol: str):
        """Add symbol to trading universe"""
        self.accepted_symbols.add(symbol)

    def remove_accepted_symbol(self, symbol: str):
        """Remove symbol from trading universe"""
        self.accepted_symbols.discard(symbol)

    def get_accepted_symbols(self) -> set[str]:
        """Get symbols to trade"""
        return self.accepted_symbols.copy()

    def mark_dust(self, symbol: str):
        """Mark symbol as dust (below threshold)"""
        self.dust_symbols.add(symbol)

    def is_dust(self, symbol: str) -> bool:
        """Check if symbol is dust"""
        return symbol in self.dust_symbols

    # ==================== Hydration ====================

    async def wait_ready(self):
        """Wait until positions hydrated"""
        # Lazy-create event only when needed (ensures we're in async context)
        if self._ready_event is None:
            self._ready_event = asyncio.Event()
        if not self._hydrated:
            await self._ready_event.wait()

    def mark_ready(self):
        """Signal positions are ready"""
        self._hydrated = True
        # Only set event if it was created
        if self._ready_event is not None:
            self._ready_event.set()

    def is_ready(self) -> bool:
        """Check if positions hydrated"""
        return self._hydrated

    # ==================== Summary ====================

    def get_portfolio_summary(self) -> dict:
        """Get complete portfolio snapshot"""
        return {
            "nav_usdt": self.nav_usdt,
            "free_balance_usdt": self.free_balance_usdt,
            "invested_capital_usdt": self.invested_capital_usdt,
            "position_count": len(self.positions),
            "open_order_count": self.get_open_order_count(),
            "portfolio_value": self.get_portfolio_value(),
            "symbols_active": len(self.accepted_symbols),
            "symbols_dust": len(self.dust_symbols),
            "exchange_throttled": self.exchange_throttled,
        }

    def set_exchange_throttle(
        self,
        throttled: bool,
        *,
        reason: str = "",
        until_ts: float = 0.0,
    ) -> None:
        self.exchange_throttled = bool(throttled)
        self.exchange_throttle_reason = str(reason or "")
        self.exchange_throttle_until_ts = max(0.0, float(until_ts or 0.0))

    # ==================== Reservations ====================

    def reserve_quote(
        self,
        asset: str,
        amount: float,
        *,
        ttl_sec: float = 30.0,
        reservation_id: str = "",
        created_at: float = 0.0,
    ) -> None:
        asset_key = str(asset or "USDT").upper()
        amount = max(0.0, float(amount or 0.0))
        if amount <= 0:
            return
        created = float(created_at or 0.0)
        if created <= 0:
            import time

            created = time.time()
        expires = created + max(1.0, float(ttl_sec or 30.0))
        arr = self.quote_reservations.setdefault(asset_key, [])
        arr.append(
            {
                "reservation_id": str(reservation_id or ""),
                "amount": amount,
                "created_at": created,
                "expires_at": expires,
            }
        )

    def release_quote_reservation(self, asset: str, reservation_id: str) -> None:
        asset_key = str(asset or "USDT").upper()
        if asset_key not in self.quote_reservations:
            return
        rid = str(reservation_id or "")
        self.quote_reservations[asset_key] = [
            r
            for r in self.quote_reservations.get(asset_key, [])
            if str(r.get("reservation_id", "")) != rid
        ]

    def prune_quote_reservations(
        self,
        asset: str,
        *,
        now_ts: float = 0.0,
        max_age_sec: float = 90.0,
        default_ttl_sec: float = 30.0,
    ) -> float:
        asset_key = str(asset or "USDT").upper()
        kept, freed = prune_reservations(
            self.quote_reservations.get(asset_key, []),
            now_ts=now_ts or None,
            max_age_sec=max_age_sec,
            default_ttl_sec=default_ttl_sec,
        )
        self.quote_reservations[asset_key] = kept
        return float(freed)

    def reserved_quote_total(self, asset: str, *, prune: bool = True) -> float:
        asset_key = str(asset or "USDT").upper()
        if prune:
            self.prune_quote_reservations(asset_key)
        return sum(
            float(r.get("amount", 0.0) or 0.0) for r in self.quote_reservations.get(asset_key, [])
        )

    # ==================== Feedback Loop ====================

    def append_trade_record(self, symbol: str, record: dict) -> None:
        """Append a closed-trade record for adaptive engine consumption.
        Capped at 200 per symbol to prevent unbounded growth."""
        lst = self.trade_history.setdefault(symbol, [])
        lst.append(record)
        if len(lst) > 200:
            lst.pop(0)

    async def emit_event(self, name: str, payload: dict) -> None:
        """Emit event for telemetry/journaling.
        Stub: subclasses can override to add event handling."""
        pass
