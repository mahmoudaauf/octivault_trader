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
