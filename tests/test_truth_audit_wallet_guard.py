"""
Sanity tests for the TruthAuditor wallet-truth guard (run-#6 fix).

Validates that ExchangeTruthAuditor._apply_recovered_fill no longer
phantom-closes a position whose base asset is still held in the wallet.

Cases:
  T1  Wallet holds full pos_qty → mark_position_closed is SKIPPED.
  T2  Wallet holds 99%+ (fee tolerance) → mark_position_closed is SKIPPED.
  T3  Wallet empty (real missed fill) → mark_position_closed PROCEEDS.
  T4  Env flag TRUTH_AUDIT_WALLET_GUARD=0 → guard bypassed (legacy behavior).
  T5  Wallet at 50% of pos_qty → close PROCEEDS (real partial reconciliation).
"""

from __future__ import annotations

import asyncio
import os
import types

import pytest

from src.l1_exchange.exchange_truth_auditor import ExchangeTruthAuditor


class _StubSharedState:
    """Minimal shared-state stub capturing mark_position_closed calls."""

    def __init__(self, pos_qty: float) -> None:
        self._pos_qty = pos_qty
        self.record_trade_calls: list = []
        self.mark_closed_calls: list = []

    def get_position_qty(self, sym: str) -> float:
        return self._pos_qty

    async def record_trade(self, *args, **kwargs):
        self.record_trade_calls.append((args, kwargs))

    async def mark_position_closed(self, **kwargs):
        self.mark_closed_calls.append(kwargs)


def _make_auditor(balances: dict, ss, base: str = "ETH", quote: str = "USDT") -> ExchangeTruthAuditor:
    """Build a bare ExchangeTruthAuditor with stubbed helpers."""
    a = ExchangeTruthAuditor.__new__(ExchangeTruthAuditor)
    a.shared_state = ss
    a.logger = types.SimpleNamespace(
        warning=lambda *a, **k: None,
        debug=lambda *a, **k: None,
        error=lambda *a, **k: None,
        info=lambda *a, **k: None,
    )
    a.dust_threshold = 1e-8

    async def _balances():
        return balances

    a._get_exchange_balances = _balances  # type: ignore[assignment]
    a._split_base_quote = lambda sym: (base, quote)  # type: ignore[assignment]

    async def _maybe_call(ss, name, *args, **kwargs):
        fn = getattr(ss, name, None)
        if fn is None:
            return None
        res = fn(*args, **kwargs)
        if asyncio.iscoroutine(res):
            return await res
        return res

    a._maybe_call = _maybe_call  # type: ignore[assignment]
    return a


def _order(side: str = "SELL", sym: str = "ETHUSDT", qty: float = 0.01) -> dict:
    return {
        "symbol": sym,
        "side": side,
        "executedQty": qty,
        "avgPrice": 2000.0,
        "fee_quote": 0.0,
        "fee_base": 0.0,
    }


@pytest.mark.asyncio
async def test_T1_wallet_full_skips_close(monkeypatch):
    monkeypatch.setenv("TRUTH_AUDIT_WALLET_GUARD", "1")
    pos_qty = 0.0108
    ss = _StubSharedState(pos_qty=pos_qty)
    a = _make_auditor(ss=ss, balances={"ETH": {"free": pos_qty}})
    await a._apply_recovered_fill(_order(qty=pos_qty), "missed_fill_recovery", False)
    assert ss.mark_closed_calls == [], "guard should skip close when wallet still holds pos"


@pytest.mark.asyncio
async def test_T2_wallet_99pct_tolerance_skips(monkeypatch):
    monkeypatch.setenv("TRUTH_AUDIT_WALLET_GUARD", "1")
    pos_qty = 0.0108
    ss = _StubSharedState(pos_qty=pos_qty)
    # Wallet has 99.5% (fee shaved) — should still skip
    a = _make_auditor(ss=ss, balances={"ETH": {"free": pos_qty * 0.995}})
    await a._apply_recovered_fill(_order(qty=pos_qty), "missed_fill_recovery", False)
    assert ss.mark_closed_calls == [], "guard should tolerate <=1% wallet shortfall"


@pytest.mark.asyncio
async def test_T3_wallet_empty_proceeds_with_close(monkeypatch):
    monkeypatch.setenv("TRUTH_AUDIT_WALLET_GUARD", "1")
    pos_qty = 0.0108
    ss = _StubSharedState(pos_qty=pos_qty)
    a = _make_auditor(ss=ss, balances={"ETH": {"free": 0.0}})
    await a._apply_recovered_fill(_order(qty=pos_qty), "missed_fill_recovery", False)
    assert len(ss.mark_closed_calls) == 1, "real missed fill: close must proceed"
    assert ss.mark_closed_calls[0]["symbol"] == "ETHUSDT"


@pytest.mark.asyncio
async def test_T4_flag_off_bypasses_guard(monkeypatch):
    monkeypatch.setenv("TRUTH_AUDIT_WALLET_GUARD", "0")
    pos_qty = 0.0108
    ss = _StubSharedState(pos_qty=pos_qty)
    a = _make_auditor(ss=ss, balances={"ETH": {"free": pos_qty}})
    await a._apply_recovered_fill(_order(qty=pos_qty), "missed_fill_recovery", False)
    # Legacy behavior: close even though wallet still holds qty
    assert len(ss.mark_closed_calls) == 1, "with guard disabled, legacy close must run"


@pytest.mark.asyncio
async def test_T5_wallet_50pct_proceeds(monkeypatch):
    monkeypatch.setenv("TRUTH_AUDIT_WALLET_GUARD", "1")
    pos_qty = 0.0108
    ss = _StubSharedState(pos_qty=pos_qty)
    a = _make_auditor(ss=ss, balances={"ETH": {"free": pos_qty * 0.5}})
    await a._apply_recovered_fill(_order(qty=pos_qty), "missed_fill_recovery", False)
    assert len(ss.mark_closed_calls) == 1, "real 50% wallet shortfall: close must proceed"
