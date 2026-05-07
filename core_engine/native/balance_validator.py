"""
Native balance validator.

Slim native port of the legacy pre-allocation guard. It protects against
over-deployment and keeps a small in-memory audit trail of capital commits and
releases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from time import time
from typing import Any


class AllocationStatus(str, Enum):
    SUCCESS = "success"
    INSUFFICIENT_BALANCE = "insufficient_balance"
    CIRCUIT_BREAKER_OPEN = "circuit_breaker_open"
    INVALID_AMOUNT = "invalid_amount"
    FAILED = "failed"


@dataclass
class AllocationLedgerEntry:
    ts: float
    symbol: str
    side: str
    amount: float
    order_id: str
    status: str
    total_balance: float = 0.0
    allocated_after: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)


class NativeBalanceValidator:
    def __init__(
        self,
        *,
        max_deployment_ratio: float = 0.98,
        max_failed_before_circuit: int = 5,
    ) -> None:
        self.total_balance = 0.0
        self.allocated_balance = 0.0
        self.reserved_balance = 0.0
        self.circuit_breaker_open = False
        self.failed_allocations = 0
        self.max_failed_before_circuit = max(1, int(max_failed_before_circuit))
        self.max_deployment_ratio = max(0.0, min(1.0, float(max_deployment_ratio)))
        self.allocation_ledger: list[AllocationLedgerEntry] = []

    def set_total_balance(self, balance: float) -> None:
        self.total_balance = max(0.0, float(balance or 0.0))

    def get_available_balance(self) -> float:
        return max(0.0, self.total_balance - self.allocated_balance - self.reserved_balance)

    def validate_allocation(
        self,
        *,
        amount: float,
        symbol: str,
        side: str,
        order_id: str = "",
    ) -> tuple[bool, AllocationStatus, str]:
        try:
            if self.circuit_breaker_open:
                return False, AllocationStatus.CIRCUIT_BREAKER_OPEN, "circuit breaker open"
            if amount <= 0:
                return False, AllocationStatus.INVALID_AMOUNT, f"invalid amount {amount}"
            available = self.get_available_balance()
            if amount > available:
                return (
                    False,
                    AllocationStatus.INSUFFICIENT_BALANCE,
                    f"need {amount:.2f}, available {available:.2f}",
                )
            reserved_after = self.allocated_balance + amount + self.reserved_balance
            if (
                self.total_balance > 0
                and reserved_after > self.total_balance * self.max_deployment_ratio
            ):
                return (
                    False,
                    AllocationStatus.INSUFFICIENT_BALANCE,
                    "allocation exceeds deployment ratio",
                )
            return True, AllocationStatus.SUCCESS, "ok"
        except Exception as e:
            return False, AllocationStatus.FAILED, str(e)

    def commit_allocation(self, *, amount: float, symbol: str, side: str, order_id: str) -> bool:
        if amount > self.get_available_balance():
            self.failed_allocations += 1
            self._check_circuit_breaker()
            return False
        self.allocated_balance += max(0.0, float(amount or 0.0))
        self.allocation_ledger.append(
            AllocationLedgerEntry(
                ts=time(),
                symbol=symbol,
                side=side,
                amount=float(amount or 0.0),
                order_id=order_id,
                status="committed",
                total_balance=self.total_balance,
                allocated_after=self.allocated_balance,
            )
        )
        return True

    def release_allocation(
        self,
        *,
        amount: float,
        symbol: str,
        order_id: str,
        reason: str = "position_closed",
    ) -> bool:
        amount = max(0.0, float(amount or 0.0))
        if amount > self.allocated_balance:
            return False
        self.allocated_balance -= amount
        self.failed_allocations = 0
        self.allocation_ledger.append(
            AllocationLedgerEntry(
                ts=time(),
                symbol=symbol,
                side="RELEASE",
                amount=amount,
                order_id=order_id,
                status=f"released_{reason}",
                total_balance=self.total_balance,
                allocated_after=self.allocated_balance,
            )
        )
        return True

    def recent_entries(self, limit: int = 20) -> list[AllocationLedgerEntry]:
        return self.allocation_ledger[-max(1, int(limit)) :]

    def _check_circuit_breaker(self) -> None:
        if self.failed_allocations >= self.max_failed_before_circuit:
            self.circuit_breaker_open = True
