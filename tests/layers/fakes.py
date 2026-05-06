"""
tests/layers/fakes.py
=====================

Layer-boundary fakes used by per-layer tests. Each fake satisfies the
Protocol-shaped interface declared in LOGICAL_LAYERED_ARCHITECTURE.md and
records every call so tests can assert on interaction patterns.

These fakes are intentionally minimal and have NO network, DB, or filesystem
side effects.
"""
from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional


# ----------------------------------------------------------------------------
# L1 fake: Exchange I/O
# ----------------------------------------------------------------------------
class FakeExchange:
    """Implements IExchangeClient + IOrderCache."""

    def __init__(
        self,
        balances: Optional[dict[str, dict[str, float]]] = None,
        klines: Optional[list[list[float]]] = None,
        fail_first_n: int = 0,
    ):
        self._balances = balances or {
            "USDT": {"free": 10_000.0, "locked": 0.0},
            "BTC": {"free": 0.0, "locked": 0.0},
        }
        self._open_positions: list = []
        self._klines = klines or [[0, 1, 2, 0.5, 1.5, 100.0]] * 50
        self._fail_first_n = fail_first_n
        self._calls = 0
        self.placed_orders: list = []
        self.cancelled: list = []
        self.cache: dict[str, dict[str, Any]] = {}

    # IExchangeClient
    async def get_balances(self) -> dict[str, Any]:
        self._calls += 1
        if self._calls <= self._fail_first_n:
            raise ConnectionError("simulated transient failure")
        # Deep copy: snapshots must be independent (L1 invariant)
        return {k: dict(v) for k, v in self._balances.items()}

    async def get_open_positions(self) -> list:
        return list(self._open_positions)

    async def place_order(self, order: dict[str, Any]) -> dict[str, Any]:
        oid = order.get("id") or f"O-{uuid.uuid4().hex[:8]}"
        filled = {
            **order,
            "id": oid,
            "status": "FILLED",
            "filled_qty": order.get("quantity", 0.0),
            "filled_price": order.get("price", 0.0),
            "ts": time.time(),
        }
        self.placed_orders.append(filled)
        self.cache[oid] = filled
        return filled

    async def cancel_order(self, oid: str) -> bool:
        self.cancelled.append(oid)
        return self.cache.pop(oid, None) is not None

    async def get_klines(self, symbol: str, tf: str, n: int) -> list:
        return list(self._klines[:n])

    # IOrderCache
    def upsert(self, order: dict[str, Any]) -> None:
        self.cache[order["id"]] = order

    def get(self, oid: str) -> Optional[dict[str, Any]]:
        return self.cache.get(oid)

    def reconcile(self, exchange_orders: list) -> dict[str, Any]:
        ex_ids = {o["id"] for o in exchange_orders}
        local_ids = set(self.cache.keys())
        return {"missing_local": ex_ids - local_ids, "stale_local": local_ids - ex_ids}


# ----------------------------------------------------------------------------
# L3 fake: Portfolio Authority
# ----------------------------------------------------------------------------
@dataclass
class _Reservation:
    token: str
    symbol: str
    qty: float
    reason: str
    released: bool = False


class FakePortfolio:
    """Implements IPortfolioAuthority with strict reserve/release accounting."""

    def __init__(self, cash: float = 10_000.0):
        self._cash = cash
        self._trading: dict[str, dict[str, float]] = {}
        self._external: dict[str, dict[str, float]] = {}
        self._reservations: dict[str, _Reservation] = {}
        self.journal: list[dict[str, Any]] = []

    @property
    def total(self) -> float:
        traded = sum(p["qty"] * p["price"] for p in self._trading.values())
        ext = sum(p["qty"] * p["price"] for p in self._external.values())
        reserved = sum(r.qty for r in self._reservations.values() if not r.released)
        return self._cash + reserved + traded + ext

    def buckets(self) -> dict[str, float]:
        traded = sum(p["qty"] * p["price"] for p in self._trading.values())
        ext = sum(p["qty"] * p["price"] for p in self._external.values())
        reserved = sum(r.qty for r in self._reservations.values() if not r.released)
        return {"CASH": self._cash, "RESERVED": reserved, "TRADING": traded, "EXTERNAL": ext}

    def positions(self) -> dict[str, dict[str, Any]]:
        return {**self._trading, **{k: {**v, "external": True} for k, v in self._external.items()}}

    def classify(self, asset: str) -> str:
        if asset in self._external:
            return "EXTERNAL"
        if asset in self._trading:
            return "TRADING"
        return "CASH" if asset == "USDT" else "UNKNOWN"

    def reserve(self, sym: str, qty: float, reason: str) -> Optional[str]:
        if qty <= 0 or qty > self._cash:
            return None
        token = f"R-{uuid.uuid4().hex[:8]}"
        self._cash -= qty
        self._reservations[token] = _Reservation(token, sym, qty, reason)
        self.journal.append(
            {"event": "RESERVE", "token": token, "sym": sym, "qty": qty, "reason": reason}
        )
        return token

    def release(self, token: str) -> bool:
        r = self._reservations.get(token)
        if r is None or r.released:
            return False
        r.released = True
        self._cash += r.qty
        self.journal.append({"event": "RELEASE", "token": token, "qty": r.qty})
        return True

    def apply_fill(self, token: str, symbol: str, qty: float, price: float) -> bool:
        r = self._reservations.get(token)
        if r is None or r.released:
            return False
        r.released = True
        self._trading[symbol] = {"qty": qty, "price": price}
        self.journal.append(
            {
                "event": "FILL",
                "token": token,
                "sym": symbol,
                "qty": qty,
                "price": price,
            }
        )
        return True

    def add_external(self, asset: str, qty: float, price: float) -> None:
        self._external[asset] = {"qty": qty, "price": price}

    def force_mutate_external(self, asset: str) -> None:
        # Used by tests to verify the invariant rejects this path
        raise PermissionError("EXTERNAL positions are read-only — invariant violated")


# ----------------------------------------------------------------------------
# L6 fake: Policy Gate
# ----------------------------------------------------------------------------
class FakePolicyGate:
    """Implements IPolicyGate. `predicate(intent) -> (approved, reason)`."""

    def __init__(
        self,
        predicate: Optional[Callable[[dict[str, Any]], tuple]] = None,
        max_position_usdt: float = 5_000.0,
    ):
        self._predicate = predicate
        self.max_position_usdt = max_position_usdt
        self.approved: list = []
        self.vetoed: list = []

    def approve(self, intent: dict[str, Any]):
        if self._predicate is not None:
            ok, reason = self._predicate(intent)
        else:
            notional = intent.get("qty", 0.0) * intent.get("price", 0.0)
            if notional > self.max_position_usdt:
                ok, reason = False, f"notional {notional} > cap {self.max_position_usdt}"
            else:
                ok, reason = True, None
        if ok:
            approved = {**intent, "approved": True}
            self.approved.append(approved)
            return approved
        else:
            veto = {
                "intent_id": intent.get("id"),
                "reason": reason,
                "cap_breached": "max_position_usdt",
            }
            self.vetoed.append(veto)
            return veto

    def caps(self) -> dict[str, Any]:
        return {"max_position_usdt": self.max_position_usdt}


# ----------------------------------------------------------------------------
# L7 fake: Metrics + Alerts
# ----------------------------------------------------------------------------
class FakeMetrics:
    """Implements IMetricsSink + IAlertBus by recording every call."""

    def __init__(self, raise_on: Optional[str] = None):
        self.gauges: list = []
        self.counters: list = []
        self.histograms: list = []
        self.alerts: list = []
        self._raise_on = raise_on

    def gauge(self, name: str, value: float, labels: Optional[dict] = None):
        if self._raise_on == "gauge":
            raise RuntimeError("simulated metrics failure")
        self.gauges.append((name, value, labels or {}))

    def counter(self, name: str, inc: float = 1.0, labels: Optional[dict] = None):
        self.counters.append((name, inc, labels or {}))

    def histogram(self, name: str, value: float, labels: Optional[dict] = None):
        self.histograms.append((name, value, labels or {}))

    def emit(self, severity: str, source_layer: str, msg: str, ctx: dict):
        self.alerts.append({"severity": severity, "layer": source_layer, "msg": msg, "ctx": ctx})
