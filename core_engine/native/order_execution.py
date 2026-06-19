"""
Native L1: Order Execution (Phase 8.2.2)

Stateless order management. Replaces ~500-line legacy OrderExecution
module with a focused ~200-line implementation.

Design choices
--------------
* No internal queues. Caller is responsible for ordering.
* Local dict tracks open orders by client_order_id.
* All exchange interaction goes through NativeExchangeClient.
* Returns OrderResult dataclass for clean consumption by upper layers.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from core_engine.native.exchange_client import (
    ExchangeClientError,
    NativeExchangeClient,
)

logger = logging.getLogger(__name__)


@dataclass
class OrderResult:
    """Outcome of a single place_order call."""

    success: bool
    symbol: str
    side: str
    quantity: float
    order_type: str
    client_order_id: str
    exchange_order_id: Optional[int] = None
    status: str = "PENDING"
    price: Optional[float] = None
    avg_price: Optional[float] = None  # real volume-weighted fill price (market orders)
    executed_qty: Optional[float] = None  # actual filled base qty (may be < requested on partial fill)
    error: Optional[str] = None
    raw: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


class NativeOrderExecution:
    """
    Thin order executor on top of :class:`NativeExchangeClient`.

    Responsibilities
    ----------------
    * Generate stable client_order_id when caller does not supply one.
    * Translate exchange responses into :class:`OrderResult`.
    * Track open orders locally (best-effort; exchange is source of truth).
    """

    def __init__(self, client: NativeExchangeClient) -> None:
        self._client = client
        self._open: dict[str, OrderResult] = {}

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────
    async def place_market_buy(
        self, symbol: str, quantity: float, *, client_order_id: Optional[str] = None
    ) -> OrderResult:
        return await self._place(symbol, "BUY", quantity, "MARKET", None, client_order_id)

    async def place_market_sell(
        self, symbol: str, quantity: float, *, client_order_id: Optional[str] = None
    ) -> OrderResult:
        return await self._place(symbol, "SELL", quantity, "MARKET", None, client_order_id)

    async def place_limit_buy(
        self,
        symbol: str,
        quantity: float,
        price: float,
        *,
        client_order_id: Optional[str] = None,
    ) -> OrderResult:
        return await self._place(symbol, "BUY", quantity, "LIMIT", price, client_order_id)

    async def place_limit_sell(
        self,
        symbol: str,
        quantity: float,
        price: float,
        *,
        client_order_id: Optional[str] = None,
    ) -> OrderResult:
        return await self._place(symbol, "SELL", quantity, "LIMIT", price, client_order_id)

    async def cancel(self, symbol: str, client_order_id: str) -> bool:
        """Cancel by client_order_id. Returns True on success."""
        try:
            await self._client.cancel_order(symbol, client_order_id=client_order_id)
            self._open.pop(client_order_id, None)
            return True
        except ExchangeClientError as e:
            logger.warning("cancel failed for %s: %s", client_order_id, e)
            return False

    async def refresh_status(self, symbol: str, client_order_id: str) -> OrderResult:
        """Fetch latest status from exchange and update local cache."""
        raw = await self._client.get_order(symbol, client_order_id=client_order_id)
        order = self._open.get(client_order_id)
        if order is None:
            order = OrderResult(
                success=True,
                symbol=symbol,
                side=str(raw.get("side", "")),
                quantity=float(raw.get("origQty", 0.0) or 0.0),
                order_type=str(raw.get("type", "")),
                client_order_id=client_order_id,
            )
        order.exchange_order_id = raw.get("orderId")
        order.status = str(raw.get("status", order.status))
        order.raw = raw
        if str(raw.get("price", "")) not in ("", "0", "0.0", "0.00000000"):
            try:
                order.price = float(raw["price"])
            except (TypeError, ValueError):
                pass

        if order.status in ("FILLED", "CANCELED", "EXPIRED", "REJECTED"):
            self._open.pop(client_order_id, None)
        else:
            self._open[client_order_id] = order
        return order

    def open_orders(self) -> list[OrderResult]:
        """Snapshot of currently-tracked open orders."""
        return list(self._open.values())

    # ──────────────────────────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _new_client_order_id() -> str:
        # Binance allows up to 36 chars. Prefix for traceability.
        return f"oct-{uuid.uuid4().hex[:24]}"

    async def _place(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str,
        price: Optional[float],
        client_order_id: Optional[str],
    ) -> OrderResult:
        coid = client_order_id or self._new_client_order_id()
        result = OrderResult(
            success=False,
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            client_order_id=coid,
            price=price,
        )
        try:
            raw = await self._client.place_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=order_type,
                price=price,
                client_order_id=coid,
            )
        except ExchangeClientError as e:
            result.error = str(e)
            result.status = "ERROR"
            return result

        result.success = True
        result.raw = raw
        result.exchange_order_id = raw.get("orderId")
        result.status = str(raw.get("status", "PENDING"))
        # Derive the real volume-weighted fill price from the exchange response.
        # MARKET orders don't carry a meaningful `price`; the true average is
        # cummulativeQuoteQty / executedQty (or the weighted `fills` array).
        result.avg_price = self._extract_avg_fill_price(raw)
        if result.avg_price and result.avg_price > 0:
            result.price = result.avg_price
        # Capture the ACTUAL filled quantity — a partial fill leaves us holding less
        # than requested; registering the requested qty over-states the position and
        # causes -2010 on the eventual sell. Fall back to requested only if absent.
        try:
            _eq = float(raw.get("executedQty", 0.0) or 0.0)
            result.executed_qty = _eq if _eq > 0 else None
        except (TypeError, ValueError):
            result.executed_qty = None
        # Track only if not already terminal
        if result.status not in ("FILLED", "CANCELED", "EXPIRED", "REJECTED"):
            self._open[coid] = result
        return result

    @staticmethod
    def _extract_avg_fill_price(raw: dict[str, Any]) -> Optional[float]:
        """Compute the volume-weighted average fill price from a Binance order response."""
        try:
            # Preferred: cummulativeQuoteQty / executedQty (exact, fee-inclusive notional)
            executed_qty = float(raw.get("executedQty", 0.0) or 0.0)
            cqq = float(raw.get("cummulativeQuoteQty", 0.0) or 0.0)
            if executed_qty > 0 and cqq > 0:
                return cqq / executed_qty
            # Fallback: weight the individual fills
            fills = raw.get("fills") or []
            if fills:
                tot_qty = 0.0
                tot_quote = 0.0
                for f in fills:
                    fq = float(f.get("qty", 0.0) or 0.0)
                    fp = float(f.get("price", 0.0) or 0.0)
                    tot_qty += fq
                    tot_quote += fq * fp
                if tot_qty > 0 and tot_quote > 0:
                    return tot_quote / tot_qty
        except (TypeError, ValueError, ZeroDivisionError):
            pass
        return None
