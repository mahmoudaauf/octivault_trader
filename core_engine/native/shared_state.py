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
        # Real-time top-of-book (from @bookTicker): the demand/supply layer that
        # PRECEDES indicators. symbol -> {bid, bid_qty, ask, ask_qty, spread_pct,
        # imbalance, ts}. imbalance = bid_qty/(bid_qty+ask_qty): >0.5 demand-heavy,
        # <0.5 supply-heavy (ask wall). Populated by WebSocketMarketData.
        self.order_book: dict[str, dict] = {}

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
        self.previous_nav_usdt: float = 0.0   # NAV from previous evaluation cycle
        self.peak_nav_usdt: float = 0.0        # All-time-high NAV (persisted across restarts via metrics)
        self.protection_floor_usdt: float = 0.0  # Ratchet floor set by NAVProtectionEngine
        self.nav_protection_state: dict = {}   # Latest NAVProtectionState.to_dict()
        self.last_nav_attribution: dict = {}    # Last NAVAttribution.to_dict() for delta tracking
        self.recovery_state: dict = {}          # Portfolio recovery mode flags
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

    def update_nav_protection(self, attribution: dict, protection_state: dict) -> None:
        """Called by NAVProtectionEngine after each evaluation."""
        self.nav_protection_state = dict(protection_state or {})
        peak = float((protection_state or {}).get("peak_nav_usdt", 0.0) or 0.0)
        if peak > self.peak_nav_usdt:
            self.peak_nav_usdt = peak
            self.metrics["peak_nav"] = peak
            self.metrics["peak_nav_ts"] = __import__("time").time()
        floor = float((protection_state or {}).get("protection_floor_usdt", 0.0) or 0.0)
        if floor > self.protection_floor_usdt:
            self.protection_floor_usdt = floor

    def update_nav(self, nav: float):
        """Update NAV (single source of truth). Captures previous value for attribution."""
        if self.nav_usdt > 0:
            self.previous_nav_usdt = self.nav_usdt
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
        """Store raw exchange/free balances, reconcile position quantities, and align NAV."""
        self.balance = {str(k).upper(): float(v or 0.0) for k, v in (balances or {}).items()}
        free_usdt = float(self.balance.get("USDT", 0.0) or 0.0)

        _NON_TRADE_ASSETS = {"USDT", "BNB", "BTC"}
        _MIN_NOTIONAL_FOR_POSITION = 5.0  # ignore confirmed dust (below $5 means truly worthless)

        # Reconcile position quantities against exchange balances.
        # This corrects partial fills and removes sold positions automatically.
        for sym in list(self.positions.keys()):
            asset = str(sym).upper().replace("USDT", "")
            exchange_qty = float(self.balance.get(asset, 0.0) or 0.0)
            pos = self.positions[sym]
            held_qty = float(getattr(pos, "qty", 0.0) if not isinstance(pos, dict) else pos.get("qty", 0.0))
            price = float((getattr(self, "prices", {}) or {}).get(sym, 0.0) or 0.0)
            if exchange_qty <= 1e-8 or (price > 0 and exchange_qty * price < _MIN_NOTIONAL_FOR_POSITION):
                # Exchange has none, or notional dropped below MIN_NOTIONAL — treat as dust/closed
                self.positions.pop(sym, None)
            elif abs(exchange_qty - held_qty) / max(held_qty, 1e-8) > 0.01:
                # Quantity differs by >1% — update to real exchange amount
                entry = float(getattr(pos, "entry_price", 0.0) if not isinstance(pos, dict) else pos.get("entry_price", 0.0))
                mark = float(getattr(pos, "mark_price", entry) if not isinstance(pos, dict) else pos.get("mark_price", entry))
                self.update_position(sym, exchange_qty, entry, mark)

        # Surface positions that exist on exchange but are unknown to the bot.
        # Covers manual buys, external transfers, and fills the bot missed.
        live_prices = getattr(self, "prices", {}) or {}
        for asset, qty in self.balance.items():
            sym = asset.upper() + "USDT"
            if asset in _NON_TRADE_ASSETS or sym in self.positions or qty <= 1e-8:
                continue
            price = float(live_prices.get(sym, 0.0) or 0.0)
            if price == 0:
                # Try price_cache as fallback (populated by REST price refresh)
                price = float((getattr(self, "price_cache", {}) or {}).get(sym, 0.0) or 0.0)
            if price > 0 and qty * price < _MIN_NOTIONAL_FOR_POSITION:
                continue  # confirmed dust by notional
            if price == 0 and qty < 1.0:
                continue  # likely dust (can't verify notional without price)
            # Register with available price (entry = current price; 0 if price not yet loaded)
            import logging as _logging
            _logging.getLogger("SharedState").info(
                "[SharedState] External position detected: %s qty=%.6f (~$%.2f) — adding to registry",
                sym, qty, qty * price,
            )
            self.update_position(sym, qty, price, price)

        invested = self.get_portfolio_value()
        # Also count any non-USDT balances not tracked as positions (dust / excluded holdings)
        # so NAV reflects true account value even when positions are filtered out.
        live_prices = getattr(self, "prices", {}) or {}
        for asset, qty in self.balance.items():
            if asset in _NON_TRADE_ASSETS:
                continue
            sym = asset.upper() + "USDT"
            if sym in self.positions:
                continue  # already counted in get_portfolio_value()
            price = float(live_prices.get(sym, 0.0) or 0.0)
            if price > 0 and float(qty or 0) > 1e-8:
                invested += float(qty) * price
        # Don't zero out invested_capital when positions exist but prices aren't loaded yet.
        if invested <= 0 and self.positions and self.invested_capital_usdt > 0:
            invested = self.invested_capital_usdt
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
        """Total value of all positions, using live prices where available."""
        total = 0.0
        live_prices = getattr(self, "price_cache", {}) or {}
        for symbol, pos in self.positions.items():
            sym_upper = str(symbol or "").upper()
            if isinstance(pos, dict):
                qty = float(pos.get("qty", 0.0) or 0.0)
                mark = float(
                    live_prices.get(sym_upper)
                    or live_prices.get(symbol)
                    or pos.get("mark_price")
                    or pos.get("current_price")
                    or pos.get("avg_price")
                    or pos.get("entry_price")  # fallback: use cost basis
                    or 0.0
                )
                total += qty * mark
            else:
                qty = float(getattr(pos, "qty", 0.0) or 0.0)
                mark = float(
                    live_prices.get(sym_upper)
                    or live_prices.get(symbol)
                    or getattr(pos, "mark_price", 0.0)
                    or getattr(pos, "entry_price", 0.0)  # fallback: use cost basis
                    or 0.0
                )
                total += qty * mark if mark > 0 else float(getattr(pos, "position_value", 0.0) or 0.0)
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

    def get_price_age(self, symbol: str) -> float:
        """Seconds since the last price tick for symbol. inf if never seen."""
        import time

        symbol_key = str(symbol or "").upper()
        ts = self._last_tick_timestamps.get(symbol_key)
        if not ts:
            return float("inf")
        return max(0.0, time.time() - float(ts))

    # ==================== Order Book / Flow ====================

    def update_book(self, symbol: str, bid: float, bid_qty: float,
                    ask: float, ask_qty: float) -> None:
        """Store real-time top-of-book from @bookTicker. Computes spread_pct and
        imbalance (bid-vs-ask resting size) — the supply/demand state that leads price."""
        import time

        symbol_key = str(symbol or "").upper()
        if bid <= 0 or ask <= 0:
            return
        mid = (bid + ask) / 2.0
        spread_pct = (ask - bid) / mid if mid > 0 else 0.0
        total_qty = bid_qty + ask_qty
        imbalance = (bid_qty / total_qty) if total_qty > 0 else 0.5
        self.order_book[symbol_key] = {
            "bid": bid, "bid_qty": bid_qty, "ask": ask, "ask_qty": ask_qty,
            "spread_pct": spread_pct, "imbalance": imbalance, "ts": time.time(),
        }

    def get_book(self, symbol: str) -> dict:
        """Latest top-of-book snapshot for symbol (empty dict if unseen)."""
        return self.order_book.get(str(symbol or "").upper(), {})

    def get_spread_pct(self, symbol: str) -> float:
        """Real-time bid/ask spread fraction (e.g. 0.001 = 0.1%). 0.0 if unseen."""
        return float(self.get_book(symbol).get("spread_pct", 0.0) or 0.0)

    def get_book_imbalance(self, symbol: str) -> float:
        """Top-of-book size imbalance: bid_qty/(bid_qty+ask_qty). >0.5 demand-heavy
        (buyers resting), <0.5 supply-heavy (ask wall). 0.5 (neutral) if unseen."""
        return float(self.get_book(symbol).get("imbalance", 0.5) or 0.5)

    def get_book_age(self, symbol: str) -> float:
        """Seconds since the last book update. inf if never seen."""
        import time

        ts = self.get_book(symbol).get("ts")
        if not ts:
            return float("inf")
        return max(0.0, time.time() - float(ts))

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

    def get_market_data(self, symbol: str, timeframe: str = "1m") -> list:
        """Return kline buffer for a symbol/timeframe. Pre-fetch populates this with 3000 rows."""
        return self.market_data.get((symbol, timeframe)) or []

    async def get_sentiment(self, symbol: str = None) -> float:
        """Return normalized Fear & Greed score as sentiment.
        -1.0 = extreme fear, 0.0 = neutral, +1.0 = extreme greed."""
        fg = getattr(self, "fear_greed_score", None)
        if fg is None:
            return 0.0
        return (float(fg) - 50.0) / 50.0

    async def emit_event(self, name: str, payload: dict) -> None:
        """Emit event for telemetry/journaling.
        Stub: subclasses can override to add event handling."""
        pass
