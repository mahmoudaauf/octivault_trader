"""
Regression tests for a real bug found 2026-07-14: ObjectiveFeedbackController's
kill-switch (_trip_kill_switch) set shared_state.trading_halted=True with no
code anywhere ever reading it back to False -- once tripped, every BUY was
permanently blocked until a manual process restart, no matter how much
drawdown recovered. Fixed via _maybe_resume_kill_switch(), which resumes once
drawdown recovers (dd_error <= 0) AND a cooldown has held since the trip.
"""
from __future__ import annotations

import pytest

from core_engine.native.objective_feedback_controller import (
    ObjectiveFeedbackController,
    Telemetry,
)


class _SharedState:
    def __init__(self) -> None:
        self.runtime_overrides = {}
        self.positions = {}
        self.trading_halted = False


def _make_controller(tmp_path, **cfg_overrides):
    class _Cfg:
        pass

    cfg = _Cfg()
    for k, v in cfg_overrides.items():
        setattr(cfg, k, v)
    ss = _SharedState()
    controller = ObjectiveFeedbackController(
        config=cfg, shared_state=ss, artefact_path=str(tmp_path / "ofc.json"),
    )
    return controller, ss


def _telemetry(*, drawdown_pct: float, trades_in_window: int = 5) -> Telemetry:
    return Telemetry(
        ok=True, nav=100.0, nav_anchor=100.0, elapsed_h=5.0,
        trades_in_window=trades_in_window, avg_net_profit_bps=5.0,
        drawdown_pct=drawdown_pct,
    )


@pytest.mark.asyncio
async def test_kill_switch_trips_after_two_consecutive_breaches(tmp_path):
    controller, ss = _make_controller(tmp_path, OBJ_MAX_DRAWDOWN_PCT=0.06)

    async def breach():
        return _telemetry(drawdown_pct=10.0)  # 10% > 6% limit

    controller._measure = breach  # type: ignore[method-assign]
    r1 = await controller.step()
    assert ss.trading_halted is False  # first breach: not yet 2 consecutive
    r2 = await controller.step()
    assert ss.trading_halted is True
    assert r2["halted"] is True


@pytest.mark.asyncio
async def test_kill_switch_does_not_resume_before_cooldown_elapses(tmp_path, monkeypatch):
    controller, ss = _make_controller(
        tmp_path, OBJ_MAX_DRAWDOWN_PCT=0.06, OBJ_KILL_SWITCH_RESUME_COOLDOWN_S=1800,
    )
    ss.trading_halted = True
    ss._trading_halted_since = 1_000_000.0

    monkeypatch.setattr(
        "core_engine.native.objective_feedback_controller.time.time",
        lambda: 1_000_100.0,  # only 100s elapsed, cooldown is 1800s
    )

    async def recovered():
        return _telemetry(drawdown_pct=1.0)  # back under the 6% limit

    controller._measure = recovered  # type: ignore[method-assign]
    result = await controller.step()

    assert ss.trading_halted is True  # still halted -- cooldown not elapsed
    assert result["halted"] is True


@pytest.mark.asyncio
async def test_kill_switch_auto_resumes_after_drawdown_recovers_and_cooldown_holds(
    tmp_path, monkeypatch
):
    controller, ss = _make_controller(
        tmp_path, OBJ_MAX_DRAWDOWN_PCT=0.06, OBJ_KILL_SWITCH_RESUME_COOLDOWN_S=1800,
    )
    ss.trading_halted = True
    ss._trading_halted_since = 1_000_000.0

    monkeypatch.setattr(
        "core_engine.native.objective_feedback_controller.time.time",
        lambda: 1_000_000.0 + 1801.0,  # cooldown elapsed
    )

    async def recovered():
        return _telemetry(drawdown_pct=1.0)

    controller._measure = recovered  # type: ignore[method-assign]
    result = await controller.step()

    assert ss.trading_halted is False
    assert result["halted"] is False


@pytest.mark.asyncio
async def test_kill_switch_does_not_resume_while_drawdown_still_breached(tmp_path, monkeypatch):
    """The dd_error<=0 precondition must hold -- cooldown elapsing alone must
    never resume trading while drawdown is still over the limit."""
    controller, ss = _make_controller(
        tmp_path, OBJ_MAX_DRAWDOWN_PCT=0.06, OBJ_KILL_SWITCH_RESUME_COOLDOWN_S=1800,
    )
    ss.trading_halted = True
    ss._trading_halted_since = 1_000_000.0

    monkeypatch.setattr(
        "core_engine.native.objective_feedback_controller.time.time",
        lambda: 1_000_000.0 + 999_999.0,  # plenty of cooldown elapsed
    )

    async def still_breached():
        return _telemetry(drawdown_pct=10.0)  # still over 6% limit

    controller._measure = still_breached  # type: ignore[method-assign]
    result = await controller.step()

    assert ss.trading_halted is True
    assert result["halted"] is True


@pytest.mark.asyncio
async def test_halted_with_no_trip_timestamp_does_not_auto_resume(tmp_path):
    """If trading_halted was set True by something other than
    _trip_kill_switch (no _trading_halted_since recorded), do not guess --
    require manual intervention rather than silently resuming."""
    controller, ss = _make_controller(tmp_path, OBJ_MAX_DRAWDOWN_PCT=0.06)
    ss.trading_halted = True  # no _trading_halted_since set at all

    async def recovered():
        return _telemetry(drawdown_pct=1.0)

    controller._measure = recovered  # type: ignore[method-assign]
    result = await controller.step()

    assert ss.trading_halted is True
    assert result["halted"] is True


@pytest.mark.asyncio
async def test_never_halted_is_a_no_op_for_resume_logic(tmp_path):
    controller, ss = _make_controller(tmp_path, OBJ_MAX_DRAWDOWN_PCT=0.06)

    async def healthy():
        return _telemetry(drawdown_pct=1.0)

    controller._measure = healthy  # type: ignore[method-assign]
    result = await controller.step()

    assert ss.trading_halted is False
    assert result["halted"] is False


@pytest.mark.asyncio
async def test_halted_true_while_still_breached_but_not_yet_two_consecutive(tmp_path):
    """Regression: `halted` in the returned record must reflect the CURRENT
    shared_state.trading_halted, not just 'did this exact step freshly trip
    it' -- a step with dd_error>0 but only 1 consecutive breach so far must
    still report halted=True if a PRIOR step already set trading_halted."""
    controller, ss = _make_controller(tmp_path, OBJ_MAX_DRAWDOWN_PCT=0.06)
    ss.trading_halted = True
    ss._trading_halted_since = 1.0

    async def still_breached_first_consecutive():
        return _telemetry(drawdown_pct=10.0)

    controller._measure = still_breached_first_consecutive  # type: ignore[method-assign]
    result = await controller.step()

    assert controller.state.consecutive_dd_breaches == 1  # not yet 2 -- _trip_kill_switch not called
    assert ss.trading_halted is True  # but still correctly reported halted
    assert result["halted"] is True
