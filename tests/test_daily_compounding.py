from datetime import datetime, timezone

import pytest

from core_engine.native.daily_compounding import DailyCompoundingPolicy


def _utc(day: int) -> datetime:
    return datetime(2026, 1, day, 0, 5, tzinfo=timezone.utc)


def test_intraday_profit_does_not_raise_sizing_nav() -> None:
    policy = DailyCompoundingPolicy()
    assert policy.sizing_nav(100.0, has_open_positions=False, now=_utc(1)) == 100.0
    assert policy.sizing_nav(110.0, has_open_positions=False, now=_utc(1)) == 100.0


def test_intraday_loss_reduces_sizing_nav_immediately() -> None:
    policy = DailyCompoundingPolicy()
    policy.sizing_nav(100.0, has_open_positions=False, now=_utc(1))
    assert policy.sizing_nav(95.0, has_open_positions=False, now=_utc(1)) == 95.0


def test_next_day_flat_portfolio_compounds_net_nav() -> None:
    policy = DailyCompoundingPolicy()
    policy.sizing_nav(100.0, has_open_positions=False, now=_utc(1))
    assert policy.sizing_nav(105.0, has_open_positions=False, now=_utc(2)) == 105.0
    assert policy.snapshot()["sizing_nav_usdt"] == 105.0


def test_rollover_waits_for_open_positions_to_close() -> None:
    policy = DailyCompoundingPolicy()
    policy.sizing_nav(100.0, has_open_positions=False, now=_utc(1))

    assert policy.sizing_nav(110.0, has_open_positions=True, now=_utc(2)) == 100.0
    assert policy.snapshot()["pending_rollover"] is True
    assert policy.sizing_nav(108.0, has_open_positions=False, now=_utc(2)) == 108.0
    assert policy.snapshot()["pending_rollover"] is False


def test_state_survives_restart(tmp_path) -> None:
    state_path = tmp_path / "compound.json"
    first = DailyCompoundingPolicy(state_path=state_path)
    first.sizing_nav(100.0, has_open_positions=False, now=_utc(1))

    restored = DailyCompoundingPolicy(state_path=state_path)
    assert restored.sizing_nav(110.0, has_open_positions=False, now=_utc(1)) == 100.0


def test_disabled_policy_uses_live_nav() -> None:
    policy = DailyCompoundingPolicy(enabled=False)
    assert policy.sizing_nav(110.0, has_open_positions=True, now=_utc(1)) == 110.0


def test_naive_datetime_is_treated_as_utc() -> None:
    policy = DailyCompoundingPolicy()
    naive = datetime(2026, 1, 1, 12, 0)
    assert policy.sizing_nav(100.0, has_open_positions=False, now=naive) == pytest.approx(100.0)


# ── Remediation item #17: NAV-protection floor connection ──────────────────

def test_protection_floor_caps_sizing_nav_below_committed_nav() -> None:
    """A protection floor closer to current NAV than the committed sizing NAV
    must win — sizing must never imply risking capital below the floor."""
    policy = DailyCompoundingPolicy()
    policy.sizing_nav(100.0, has_open_positions=False, now=_utc(1))
    # Committed sizing NAV is 100.0; a floor of 98.0 means only $2 is risk-eligible.
    result = policy.sizing_nav(
        100.0, has_open_positions=False, now=_utc(1), protection_floor_usdt=98.0,
    )
    assert result == pytest.approx(2.0)


def test_protection_floor_zero_is_a_no_op() -> None:
    policy = DailyCompoundingPolicy()
    policy.sizing_nav(100.0, has_open_positions=False, now=_utc(1))
    assert policy.sizing_nav(
        100.0, has_open_positions=False, now=_utc(1), protection_floor_usdt=0.0,
    ) == 100.0


def test_protection_floor_above_current_nav_yields_zero_sizing() -> None:
    """Deep drawdown: floor exceeds current NAV — no capital is risk-eligible,
    matching NAVProtectionEngine's own max(0.0, ...) clamp on available_profit_to_risk_usdt."""
    policy = DailyCompoundingPolicy()
    policy.sizing_nav(100.0, has_open_positions=False, now=_utc(1))
    result = policy.sizing_nav(
        90.0, has_open_positions=False, now=_utc(1), protection_floor_usdt=95.0,
    )
    assert result == 0.0


def test_protection_floor_also_applies_on_first_seed_and_disabled_policy() -> None:
    # First-ever call (no prior state) still respects the floor.
    policy = DailyCompoundingPolicy()
    assert policy.sizing_nav(
        100.0, has_open_positions=False, now=_utc(1), protection_floor_usdt=97.0,
    ) == pytest.approx(3.0)

    # Disabled policy: floor still applies (it's a separate risk control, not
    # part of the daily-rollover mechanism being disabled).
    disabled = DailyCompoundingPolicy(enabled=False)
    assert disabled.sizing_nav(
        110.0, has_open_positions=True, now=_utc(1), protection_floor_usdt=100.0,
    ) == pytest.approx(10.0)
