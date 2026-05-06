"""
Native L4: Safety order (OCO) manager (Phase 8.3.10).

Per-symbol One-Cancels-Other (OCO) intent store. Replaces the
``compat.py`` null stub for the ``safety_order_manager`` app_ctx key,
satisfying the contract consumed by
``SafeExecutionEngine.place_safety_order``.

Design philosophy
-----------------
Spot Binance exposes a true OCO endpoint (``/api/v3/order/oco``) but
``NativeExchangeClient`` does not yet implement it (only ``place_order``
for MARKET/LIMIT exists). Rather than block this phase on a new
exchange-client method, the manager operates as an **intent store**:
``place_oco`` records a paired (TP/SL) intent and best-effort places a
LIMIT TP-side order via the existing exchange client when one is
provided. The SL leg is tracked as a logical intent that the (existing)
``NativeTPSLEngine`` already evaluates each cycle via
``check_exit_levels``. This keeps L4 cohesive: the SL leg fires through
the same path the TP/SL engine already uses, while the TP leg goes on
the book as a resting LIMIT order whenever possible.

When no exchange client is wired (unit tests, paper-trade mode, or
``NativeExchangeClient`` is missing), the intent is still recorded with
``status="SIMULATED"`` so callers can observe it. This mirrors the
legacy "graceful degradation" behaviour of ``SafeExecutionEngine``.

API contract
------------
* ``async place_oco(symbol, quantity, take_profit, stop_loss, *, side="SELL") -> dict``
      — record the OCO; return ExecutionResult-shaped dict
* ``async cancel_oco(symbol) -> bool``       — cancel resting TP order + drop intent
* ``get_oco(symbol) -> dict | None``         — observability snapshot
* ``list_active() -> list[str]``             — symbols with active intents
* ``health() -> dict``                       — counters
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .exchange_client import NativeExchangeClient

logger = logging.getLogger(__name__)


@dataclass
class _OCOIntent:
    """Per-symbol OCO state."""

    symbol: str
    side: str  # exit side ("SELL" for long positions, "BUY" for shorts)
    quantity: float
    take_profit: float
    stop_loss: float
    status: str  # "ACTIVE" | "SIMULATED" | "CANCELED" | "FAILED"
    tp_order_id: str | None = None
    error: str | None = None
    created_ts: float = field(default_factory=time.time)


class NativeSafetyOrderManager:
    """Per-symbol OCO intent store with best-effort exchange placement."""

    def __init__(
        self,
        *,
        exchange_client: NativeExchangeClient | None = None,
        min_order_usdt: float = 10.0,
    ) -> None:
        if min_order_usdt <= 0:
            raise ValueError(f"min_order_usdt must be > 0, got {min_order_usdt}")
        self._exchange_client = exchange_client
        self._min_order_usdt = float(min_order_usdt)
        self._intents: dict[str, _OCOIntent] = {}

        # Health counters
        self._placed = 0
        self._simulated = 0
        self._failed = 0
        self._canceled = 0

    # ------------------------------------------------------------------
    # Place / cancel
    # ------------------------------------------------------------------
    async def place_oco(
        self,
        symbol: str,
        quantity: float,
        take_profit: float,
        stop_loss: float,
        *,
        side: str = "SELL",
    ) -> dict[str, Any]:
        """
        Record an OCO intent and best-effort place the TP leg.

        Returns an ExecutionResult-shaped dict consumable by
        ``SafeExecutionEngine.place_safety_order`` without further
        adaptation.
        """
        result: dict[str, Any] = {
            "success": False,
            "symbol": symbol,
            "action": "OCO",
            "quantity": quantity,
            "order_id": None,
            "status": "FAILED",
            "error_message": None,
            "timestamp": time.time(),
        }

        # ---- input validation ----
        if quantity <= 0:
            result["error_message"] = f"quantity must be > 0, got {quantity}"
            self._failed += 1
            return result
        if take_profit <= 0 or stop_loss <= 0:
            result["error_message"] = f"tp ({take_profit}) and sl ({stop_loss}) must both be > 0"
            self._failed += 1
            return result
        side_u = side.upper()
        if side_u not in ("BUY", "SELL"):
            result["error_message"] = f"invalid side: {side!r}"
            self._failed += 1
            return result
        # For a SELL OCO (long exit), TP > SL must hold.
        if side_u == "SELL" and take_profit <= stop_loss:
            result["error_message"] = (
                f"SELL OCO requires take_profit > stop_loss; "
                f"got tp={take_profit}, sl={stop_loss}"
            )
            self._failed += 1
            return result
        # Notional sanity
        if quantity * take_profit < self._min_order_usdt:
            result["error_message"] = (
                f"notional {quantity * take_profit:.2f} < min_order_usdt " f"{self._min_order_usdt}"
            )
            self._failed += 1
            return result

        # ---- best-effort exchange placement (TP leg only) ----
        intent = _OCOIntent(
            symbol=symbol,
            side=side_u,
            quantity=float(quantity),
            take_profit=float(take_profit),
            stop_loss=float(stop_loss),
            status="SIMULATED",
        )

        if self._exchange_client is not None:
            try:
                resp = await self._exchange_client.place_order(
                    symbol=symbol,
                    side=side_u,
                    quantity=float(quantity),
                    order_type="LIMIT",
                    price=float(take_profit),
                )
                tp_id = str(resp.get("orderId") or resp.get("clientOrderId") or "")
                intent.tp_order_id = tp_id or None
                intent.status = "ACTIVE"
                self._placed += 1
                result["order_id"] = intent.tp_order_id
                result["status"] = "PENDING"
                result["success"] = True
            except Exception as exc:
                logger.warning("OCO TP-leg place_order failed for %s: %s", symbol, exc)
                intent.status = "FAILED"
                intent.error = str(exc)
                self._failed += 1
                result["error_message"] = str(exc)
                # Persist the failed intent for observability before returning.
                self._intents[symbol] = intent
                return result
        else:
            self._simulated += 1
            result["status"] = "SIMULATED"
            result["success"] = True

        self._intents[symbol] = intent
        return result

    async def cancel_oco(self, symbol: str) -> bool:
        """
        Cancel the resting TP order (if any) and drop the intent.

        Returns True if an intent existed and was removed (regardless of
        whether the exchange cancel itself succeeded), False if no
        intent was tracked for *symbol*.
        """
        intent = self._intents.pop(symbol, None)
        if intent is None:
            return False

        self._canceled += 1
        if self._exchange_client is not None and intent.tp_order_id and intent.status == "ACTIVE":
            try:
                await self._exchange_client.cancel_order(symbol=symbol, order_id=intent.tp_order_id)
            except Exception as exc:
                logger.warning(
                    "OCO TP-leg cancel failed for %s (order=%s): %s",
                    symbol,
                    intent.tp_order_id,
                    exc,
                )
        return True

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------
    def get_oco(self, symbol: str) -> dict[str, Any] | None:
        """Return a serialisable snapshot of the OCO intent for *symbol*."""
        intent = self._intents.get(symbol)
        if intent is None:
            return None
        return {
            "symbol": intent.symbol,
            "side": intent.side,
            "quantity": intent.quantity,
            "take_profit": intent.take_profit,
            "stop_loss": intent.stop_loss,
            "status": intent.status,
            "tp_order_id": intent.tp_order_id,
            "error": intent.error,
            "created_ts": intent.created_ts,
        }

    def list_active(self) -> list[str]:
        """Symbols whose OCO intent is in ACTIVE or SIMULATED state."""
        return [
            sym for sym, intent in self._intents.items() if intent.status in ("ACTIVE", "SIMULATED")
        ]

    def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "tracked_symbols": len(self._intents),
            "active": len(self.list_active()),
            "placed": self._placed,
            "simulated": self._simulated,
            "canceled": self._canceled,
            "failed": self._failed,
            "min_order_usdt": self._min_order_usdt,
            "exchange_wired": self._exchange_client is not None,
        }


__all__ = ["NativeSafetyOrderManager"]
