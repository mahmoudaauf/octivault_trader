"""
Native L3: Fill Tracker (Phase 8.2.3)

Detects order fills and updates positions in SharedState.
Replaces portion of legacy position_manager.py that handled fill-to-position routing.

Design:
- Polls exchange for recent trades (user-initiated fills)
- Updates SharedState positions on detected fills
- Emits fill events for telemetry/journal
- Reconciles with order tracking (dedup via exchange_order_id)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class Fill:
    """Single fill event from exchange"""

    exchange_trade_id: int  # Unique trade ID from exchange
    symbol: str
    side: str  # "BUY" or "SELL"
    quantity: float
    price: float
    fee_qty: float
    fee_asset: str
    timestamp_ms: int
    commission: float = 0.0  # Denominated in ``fee_asset`` (not necessarily quote)


class NativeFillTracker:
    """
    Detect fills and update positions.

    Usage::

        tracker = NativeFillTracker(
            exchange_client=client,
            shared_state=ss,
            poll_interval_sec=5.0
        )
        await tracker.start()
        # Fills detected every 5 seconds, positions auto-updated
    """

    def __init__(
        self,
        *,
        exchange_client: Any,  # NativeExchangeClient
        shared_state: Any,  # NativeSharedState
        poll_interval_sec: float = 5.0,
        tp_sl_engine: Any = None,  # NativeTPSLEngine (optional, for arm_position on fill)
    ):
        self._exchange = exchange_client
        self._shared_state = shared_state
        self._poll_interval = max(1.0, poll_interval_sec)
        self._tp_sl_engine = tp_sl_engine

        self._running = False
        self._poll_task: Optional[asyncio.Task[None]] = None
        self._last_trade_id: dict[str, int] = {}  # symbol → highest seen trade_id
        self._processed_trades: set[int] = set()  # dedup gate
        self._lock = asyncio.Lock()

    def set_tp_sl_engine(self, engine: Any) -> None:
        """Late-inject TP/SL engine after construction (avoids circular bootstrap order)."""
        self._tp_sl_engine = engine

    # ──────────────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────────────
    async def start(self) -> None:
        """Start background fill detection loop."""
        if self._running:
            return
        self._running = True
        self._poll_task = asyncio.create_task(self._run_poll_loop())
        logger.info("🎯 FillTracker started (poll_interval=%.1fs)", self._poll_interval)

    async def stop(self) -> None:
        """Stop fill detection."""
        self._running = False
        logger.info("FillTracker stopped.")

    # ──────────────────────────────────────────────────────────────────
    # Core fill detection
    # ──────────────────────────────────────────────────────────────────
    async def _run_poll_loop(self) -> None:
        """Background task: detect fills periodically."""
        while self._running:
            try:
                await self._detect_fills()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("FillTracker error: %s", e, exc_info=True)
            await asyncio.sleep(self._poll_interval)

    async def _detect_fills(self) -> None:
        """Query exchange for recent trades, update positions."""
        if not self._shared_state:
            return

        # Get symbols to monitor
        symbols = self._shared_state.get_accepted_symbols()
        if not symbols:
            return

        async with self._lock:
            new_fills = []
            for sym in symbols:
                try:
                    fills = await self._fetch_recent_trades(sym)
                    if fills:
                        new_fills.extend(fills)
                except Exception as e:
                    logger.warning("Failed to fetch trades for %s: %s", sym, e)

            # Process new fills (update positions, dedup)
            for fill in new_fills:
                if fill.exchange_trade_id not in self._processed_trades:
                    await self._process_fill(fill)
                    self._processed_trades.add(fill.exchange_trade_id)

    async def _fetch_recent_trades(self, symbol: str) -> list[Fill]:
        """Query exchange for recent trades on symbol."""
        if not hasattr(self._exchange, "get_my_trades"):
            return []

        try:
            # Binance side field is isBuyer (bool), not a "side" string — resolve
            # it into the "BUY"/"SELL" shape this class works in before parsing.
            raw_trades = await self._exchange.get_my_trades(
                symbol,
                limit=10,
            )
            if not raw_trades:
                return []

            fills = []
            for trade in raw_trades:
                # Parse exchange response
                try:
                    side = trade.get("side")
                    if side is None:
                        side = "BUY" if trade.get("isBuyer") else "SELL"
                    fill = Fill(
                        exchange_trade_id=int(trade.get("id") or 0),
                        symbol=str(trade.get("symbol") or symbol).upper(),
                        side=str(side or "BUY").upper(),
                        quantity=float(trade.get("qty") or trade.get("quantity") or 0.0),
                        price=float(trade.get("price") or 0.0),
                        fee_qty=float(trade.get("commissionQty") or 0.0),
                        fee_asset=str(trade.get("commissionAsset") or ""),
                        timestamp_ms=int(trade.get("time") or 0),
                        commission=float(trade.get("commission") or 0.0),
                    )
                    if fill.quantity > 0 and fill.price > 0:
                        fills.append(fill)
                except Exception:
                    continue

            return fills
        except Exception as e:
            logger.warning("get_my_trades failed: %s", e)
            return []

    async def _process_fill(self, fill: Fill) -> None:
        """Update position after fill, emit event."""
        try:
            if fill.side.upper() == "BUY":
                await self._process_buy_fill(fill)
            elif fill.side.upper() == "SELL":
                await self._process_sell_fill(fill)
        except Exception as e:
            logger.error("Failed to process fill: %s", e, exc_info=True)

    def _commission_quote(self, fill: Fill) -> float:
        """Convert Binance's commission asset to quote currency when possible."""
        commission = max(0.0, float(fill.commission or 0.0))
        if commission <= 0:
            return 0.0
        symbol = fill.symbol.upper()
        fee_asset = fill.fee_asset.upper()
        quote_asset = "USDT" if symbol.endswith("USDT") else ""
        base_asset = symbol[: -len(quote_asset)] if quote_asset else ""
        if fee_asset == quote_asset:
            return commission
        if fee_asset == base_asset:
            return commission * float(fill.price or 0.0)
        price_cache = getattr(self._shared_state, "price_cache", {}) or {}
        conversion = float(price_cache.get(f"{fee_asset}{quote_asset}", 0.0) or 0.0)
        return commission * conversion if conversion > 0 else 0.0

    def _record_fee_metric(self, *, quote_value: float, commission_quote: float) -> float:
        fee_bps = (
            commission_quote / quote_value * 10_000.0
            if quote_value > 0 and commission_quote > 0
            else 0.0
        )
        metrics = getattr(self._shared_state, "metrics", None)
        if not isinstance(metrics, dict) or fee_bps <= 0:
            return fee_bps
        samples = int(metrics.get("fee_samples", 0) or 0)
        previous = float(metrics.get("avg_fee_bps", 0.0) or 0.0)
        metrics["avg_fee_bps"] = ((previous * samples) + fee_bps) / (samples + 1)
        metrics["fee_samples"] = samples + 1
        metrics["last_update_ts"] = time.time()
        return fee_bps

    async def _process_buy_fill(self, fill: Fill) -> None:
        """Update position after BUY fill."""
        sym = fill.symbol.upper()

        # Get current position (if any)
        current_pos = self._shared_state.get_position(sym)
        if current_pos:
            # Add to existing position (average in)
            old_qty = current_pos.qty
            old_cost = old_qty * current_pos.entry_price

            new_qty = old_qty + fill.quantity
            new_cost = old_cost + (fill.quantity * fill.price)
            new_entry_price = new_cost / new_qty if new_qty > 0 else fill.price

            self._shared_state.update_position(
                symbol=sym,
                qty=new_qty,
                entry=new_entry_price,
                current=fill.price,
            )
        else:
            # New position
            self._shared_state.update_position(
                symbol=sym,
                qty=fill.quantity,
                entry=fill.price,
                current=fill.price,
            )

        # Arm TP/SL for the new/updated position (records entry_ts for time-based logic)
        if self._tp_sl_engine is not None:
            try:
                tp, sl = self._tp_sl_engine.arm_position(
                    sym, fill.price, entry_ts=fill.timestamp_ms / 1000.0
                )
                pos_obj = self._shared_state.get_position(sym)
                if pos_obj is not None and isinstance(pos_obj, dict):
                    pos_obj["tp"] = tp
                    pos_obj["sl"] = sl
                    pos_obj["entry_ts"] = fill.timestamp_ms / 1000.0
            except Exception as _tpsl_e:
                logger.warning("[FillTracker] arm_position failed for %s: %s", sym, _tpsl_e)

        # Track fee in basis points (1 bps = 0.01%)
        quote_value = fill.quantity * fill.price
        commission_quote = self._commission_quote(fill)
        fee_bps = self._record_fee_metric(
            quote_value=quote_value, commission_quote=commission_quote
        )

        # Emit event
        await self._emit_fill_event(
            fill_type="BUY_FILLED",
            symbol=sym,
            qty=fill.quantity,
            price=fill.price,
            commission=fill.commission,
        )

        logger.info(
            "[BUY_FILLED] %s qty=%.8f price=%.8f commission=%.8f fee_bps=%.2f",
            sym,
            fill.quantity,
            fill.price,
            fill.commission,
            fee_bps,
        )

    async def _process_sell_fill(self, fill: Fill) -> None:
        """Update position after SELL fill."""
        sym = fill.symbol.upper()

        # Get current position
        current_pos = self._shared_state.get_position(sym)
        if not current_pos or current_pos.qty <= 0:
            # SELL without position (should not happen, but log it)
            logger.warning("[SELL_FILLED] %s but no position", sym)
            return

        # Reduce or close position
        remaining_qty = current_pos.qty - fill.quantity
        if remaining_qty > 1e-8:
            # Partial close
            self._shared_state.update_position(
                symbol=sym,
                qty=remaining_qty,
                entry=current_pos.entry_price,
                current=fill.price,
            )
        else:
            # Full close — clear TP/SL tracking state
            self._shared_state.close_position(sym)
            if self._tp_sl_engine is not None:
                try:
                    self._tp_sl_engine.close_position_tracking(sym)
                except Exception as _e:
                    logger.warning(
                        "[FillTracker] close_position_tracking failed for %s: %s", sym, _e
                    )

        # Calculate realized P&L
        commission_quote = self._commission_quote(fill)
        realized_pnl = (
            (fill.price - current_pos.entry_price) * fill.quantity - commission_quote
        )

        # Emit event
        await self._emit_fill_event(
            fill_type="SELL_FILLED",
            symbol=sym,
            qty=fill.quantity,
            price=fill.price,
            commission=fill.commission,
            realized_pnl=realized_pnl,
        )

        # Append trade record for adaptive engine feedback loop
        record = {
            "ts": time.time(),
            "realized_delta": realized_pnl,
            "fee_quote": commission_quote,
            "hold_sec": 0.0,  # TODO: track entry_ts in Position for accurate hold time
        }
        if hasattr(self._shared_state, "append_trade_record"):
            self._shared_state.append_trade_record(sym, record)

        # Track fee in basis points
        quote_value = fill.quantity * fill.price
        fee_bps = self._record_fee_metric(
            quote_value=quote_value, commission_quote=commission_quote
        )

        # Update metrics for ObjectiveFeedbackController
        if hasattr(self._shared_state, "metrics"):
            m = self._shared_state.metrics
            if hasattr(self._shared_state, "record_realized_pnl_event"):
                self._shared_state.record_realized_pnl_event(realized_pnl)
            else:
                m["realized_pnl"] = m.get("realized_pnl", 0.0) + realized_pnl
            m["trades_in_window"] = m.get("trades_in_window", 0) + 1
            m["trades_since_ofc_check"] = m.get("trades_since_ofc_check", 0) + 1
            m["last_update_ts"] = time.time()
            # Rolling win_rate using exponential moving average
            prev_wr = m.get("win_rate_window", 0.5)
            m["win_rate_window"] = prev_wr * 0.9 + (1.0 if realized_pnl > 0 else 0.0) * 0.1

        logger.info(
            "[SELL_FILLED] %s qty=%.8f price=%.8f commission=%.8f fee_bps=%.2f pnl=%.8f",
            sym,
            fill.quantity,
            fill.price,
            fill.commission,
            fee_bps,
            realized_pnl,
        )

    # ──────────────────────────────────────────────────────────────────
    # Events
    # ──────────────────────────────────────────────────────────────────
    async def _emit_fill_event(
        self,
        fill_type: str,
        symbol: str,
        qty: float,
        price: float,
        commission: float = 0.0,
        realized_pnl: float = 0.0,
    ) -> None:
        """Emit fill event for telemetry/journaling."""
        try:
            if not hasattr(self._shared_state, "emit_event"):
                return

            payload = {
                "symbol": symbol,
                "fill_type": fill_type,
                "qty": float(qty),
                "price": float(price),
                "commission": float(commission),
                "timestamp": time.time(),
            }
            if fill_type == "SELL_FILLED":
                payload["realized_pnl"] = float(realized_pnl)

            # Emit event (may be sync or async)
            res = self._shared_state.emit_event(fill_type, payload)
            if asyncio.iscoroutine(res):
                await res
        except Exception as e:
            logger.debug("Failed to emit fill event: %s", e)
