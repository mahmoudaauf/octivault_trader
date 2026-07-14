"""
Regression tests for a real bug found 2026-07-14 while running the live system:
arbitration_engine.py's gate_6_risk_manager daily-loss check was reading
shared_state.metrics["realized_pnl"] -- a LIFETIME CUMULATIVE total -- as if
it were today's PnL. Once enough historical loss had accumulated, this
permanently tripped the 2% daily-loss circuit breaker on every cycle
regardless of what actually happened that day (observed live:
"daily loss 99.66% exceeds limit 2.00%" against a flat-NAV account with a
large historical loss, blocking every BUY).

Fix: NativeSharedState.record_realized_pnl_event() maintains a separate
UTC-day-scoped "realized_pnl_today" bucket alongside the lifetime one, and
arbitration_engine.py's _portfolio_snapshot() now computes daily_pnl_pct from
that day-scoped bucket instead.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from core_engine.native.arbitration_engine import NativeArbitrationEngine
from core_engine.native.decisions import NativeDecisionEngine
from core_engine.native.shared_state import NativeSharedState


# ── NativeSharedState.record_realized_pnl_event ─────────────────────────────

def test_record_realized_pnl_event_updates_both_lifetime_and_today():
    ss = NativeSharedState()
    ss.record_realized_pnl_event(-5.0)
    ss.record_realized_pnl_event(2.0)
    assert ss.metrics["realized_pnl"] == pytest.approx(-3.0)
    assert ss.metrics["realized_pnl_today"] == pytest.approx(-3.0)


def test_realized_pnl_today_resets_on_utc_day_change(monkeypatch):
    ss = NativeSharedState()
    ss.record_realized_pnl_event(-50.0)
    assert ss.metrics["realized_pnl_today"] == pytest.approx(-50.0)
    assert ss.metrics["realized_pnl"] == pytest.approx(-50.0)

    # Simulate the next UTC day by directly rewriting the stored date, since
    # record_realized_pnl_event always stamps "today" from real time.
    ss.metrics["realized_pnl_today_date"] = "2000-01-01"
    ss.record_realized_pnl_event(1.0)

    # Lifetime total keeps accumulating...
    assert ss.metrics["realized_pnl"] == pytest.approx(-49.0)
    # ...but today's bucket resets and only reflects the new day's trade.
    assert ss.metrics["realized_pnl_today"] == pytest.approx(1.0)


# ── NativeArbitrationEngine._portfolio_snapshot daily_pnl_pct scoping ───────

def _make_shared_state_with_history(*, lifetime_loss: float, today_pnl: float = 0.0):
    ss = NativeSharedState()
    ss.free_balance_usdt = 57.85
    ss.balance = {"USDT": 57.85}
    ss.session_anchor_nav = 57.85
    ss.metrics["realized_pnl"] = lifetime_loss
    if today_pnl != 0.0:
        ss.record_realized_pnl_event(today_pnl)
    else:
        # No trade closed today -- bucket must be treated as 0, not stale.
        ss.metrics["realized_pnl_today"] = 0.0
        ss.metrics["realized_pnl_today_date"] = ""
    return ss


def _make_engine(ss):
    de = NativeDecisionEngine(daily_loss_limit_pct=2.0)
    return NativeArbitrationEngine(shared_state=ss, decision_engine=de)


def test_large_historical_loss_does_not_trip_daily_gate_when_today_is_flat():
    """The exact bug scenario: a large lifetime cumulative loss (~99% of NAV)
    must NOT be read as today's loss when no trade has closed today."""
    ss = _make_shared_state_with_history(lifetime_loss=-57.67)
    snapshot = _make_engine(ss)._portfolio_snapshot()
    assert snapshot.daily_pnl_pct == pytest.approx(0.0)


def test_gate_6_risk_manager_not_blocked_by_historical_loss():
    ss = _make_shared_state_with_history(lifetime_loss=-57.67)
    engine = _make_engine(ss)
    assert engine.gate_6_risk_manager(check_exposure=False) is True


def test_todays_actual_loss_still_correctly_trips_the_gate():
    """A genuine today loss exceeding the 2% limit must still block --
    this fix must not silently disable the daily-loss circuit breaker."""
    ss = _make_shared_state_with_history(lifetime_loss=-57.67, today_pnl=-5.0)  # -8.6% of 57.85
    engine = _make_engine(ss)
    assert engine.gate_6_risk_manager(check_exposure=False) is False


def test_todays_small_loss_within_limit_does_not_trip_gate():
    ss = _make_shared_state_with_history(lifetime_loss=-57.67, today_pnl=-0.5)  # ~-0.86% of 57.85
    engine = _make_engine(ss)
    assert engine.gate_6_risk_manager(check_exposure=False) is True


def test_stale_prior_day_bucket_not_mistaken_for_today():
    """If realized_pnl_today_date doesn't match today, the stored value must
    be ignored (treated as 0), not read as if it were current."""
    ss = NativeSharedState()
    ss.free_balance_usdt = 100.0
    ss.session_anchor_nav = 100.0
    ss.metrics["realized_pnl"] = -90.0
    ss.metrics["realized_pnl_today"] = -90.0
    ss.metrics["realized_pnl_today_date"] = "2020-01-01"  # stale
    snapshot = _make_engine(ss)._portfolio_snapshot()
    assert snapshot.daily_pnl_pct == pytest.approx(0.0)
