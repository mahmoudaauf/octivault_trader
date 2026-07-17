"""
Opt-in pace-chasing in ObjectiveFeedbackController (2026-07-17).

Context: the OFC originally had a pace_error -> size/throughput control path.
It was deliberately removed (d_size <= 0 always, d_thru hardwired to 0) with an
in-code rationale that idle periods must never raise size or lower the
confidence floor. These flags re-enable it at an operator's explicit request,
who was shown the governing arithmetic first: a pace controller has NO FIXED
POINT on a negative-expectancy generator, so it ramps to its clamp and pins.

These tests exist to prove the CONTAINMENT holds — that the ramp cannot escape
the pre-existing knob_ranges clamps, that de-risking always beats pace-chasing,
and above all that the DEFAULT configuration is completely inert.
"""
from __future__ import annotations

from types import SimpleNamespace

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


def _controller(tmp_path, **flags) -> ObjectiveFeedbackController:
    """Explicit flags always win over the real OS environment (never fall
    through to `os.environ`, per _cfg_bool()'s config->env->default order) so
    these tests stay hermetic regardless of ambient .env state. Necessary
    because .env now has all three OBJ_PACE_*_ENABLED permanently true for
    production (see memory: ofc-pace-chasing-enabled) — a bare `config=None`
    default here would silently pick that up whenever something upstream in
    the same pytest session imports main.py (which calls load_dotenv())."""
    base = {
        "OBJ_PACE_SIZE_ENABLED": False,
        "OBJ_PACE_GATE_ENABLED": False,
        "OBJ_PACE_THRU_ENABLED": False,
    }
    base.update(flags)
    cfg = SimpleNamespace(**base)
    c = ObjectiveFeedbackController(
        config=cfg,
        shared_state=_SharedState(),
        artefact_path=str(tmp_path / "ofc.json"),
    )
    c.state.last_knobs = {
        "confidence_floor": 0.68,
        "size_multiplier": 1.0,
        "target_throughput_per_hour": 10.0,
    }
    return c


def _telemetry(*, nav=100.0, anchor=100.0, trades=1, dd_pct=0.0) -> Telemetry:
    return Telemetry(
        ok=True,
        nav=nav,
        nav_anchor=anchor,
        elapsed_h=1.0,
        trades_in_window=trades,
        avg_net_profit_bps=5.0,
        drawdown_pct=dd_pct,
    )


def _attach(controller, tel: Telemetry):
    async def measure() -> Telemetry:
        return tel

    controller._measure = measure  # type: ignore[method-assign]


class TestDefaultIsInert:
    @pytest.mark.asyncio
    async def test_flags_default_off_means_no_pace_intervention(self, tmp_path) -> None:
        """Merging this must change nothing until deliberately enabled."""
        c = _controller(tmp_path)
        _attach(c, _telemetry())  # flat NAV -> behind the +2%/day target
        result = await c.step()

        assert result["errors"]["pace_pct_per_h"] < 0, "precondition: must be behind"
        assert result["pace_applied"] == {}
        assert result["knobs_after"]["size_multiplier"] == 1.0
        assert result["knobs_after"]["confidence_floor"] == 0.68


class TestPaceSizing:
    @pytest.mark.asyncio
    async def test_behind_target_raises_size_when_enabled(self, tmp_path) -> None:
        c = _controller(tmp_path, OBJ_PACE_SIZE_ENABLED=True)
        _attach(c, _telemetry())
        result = await c.step()

        assert result["pace_applied"].get("size", 0) > 0
        assert result["knobs_after"]["size_multiplier"] > 1.0

    @pytest.mark.asyncio
    async def test_size_never_exceeds_hard_cap_however_far_behind(self, tmp_path) -> None:
        """The containment guarantee. An unreachable target means the error never
        closes, so the controller pushes every step forever — the clamp is the
        only thing standing between that and unbounded size."""
        c = _controller(tmp_path, OBJ_PACE_SIZE_ENABLED=True)
        cap = c.knob_ranges["size_multiplier"][1]
        # Catastrophically behind, and iterate far past the point of pinning.
        _attach(c, _telemetry(nav=50.0, anchor=100.0))
        for _ in range(200):
            result = await c.step()
            assert result["knobs_after"]["size_multiplier"] <= cap

        assert result["knobs_after"]["size_multiplier"] == pytest.approx(cap)

    @pytest.mark.asyncio
    async def test_ahead_of_target_applies_no_pace_boost(self, tmp_path) -> None:
        c = _controller(tmp_path, OBJ_PACE_SIZE_ENABLED=True)
        # +10% NAV in 1h — far ahead of the ~0.083%/h target.
        _attach(c, _telemetry(nav=110.0, anchor=100.0))
        result = await c.step()

        assert result["errors"]["pace_pct_per_h"] > 0, "precondition: must be ahead"
        assert result["pace_applied"] == {}


class TestPaceGateLoosening:
    @pytest.mark.asyncio
    async def test_behind_target_lowers_conf_floor_when_enabled(self, tmp_path) -> None:
        c = _controller(tmp_path, OBJ_PACE_GATE_ENABLED=True)
        _attach(c, _telemetry())
        result = await c.step()

        assert result["pace_applied"].get("conf", 0) < 0
        assert result["knobs_after"]["confidence_floor"] < 0.68

    @pytest.mark.asyncio
    async def test_conf_floor_never_breaches_empirical_breakeven_floor(self, tmp_path) -> None:
        """OBJ_CONF_FLOOR_MIN (0.65) is documented in-code as the empirical
        breakeven — below it, observed win-rate is ~30%. Pace-chasing must never
        punch through it, however far behind target the system is."""
        c = _controller(tmp_path, OBJ_PACE_GATE_ENABLED=True)
        floor = c.knob_ranges["confidence_floor"][0]
        _attach(c, _telemetry(nav=50.0, anchor=100.0))
        for _ in range(200):
            result = await c.step()
            assert result["knobs_after"]["confidence_floor"] >= floor

        assert result["knobs_after"]["confidence_floor"] == pytest.approx(floor)


class TestDeRiskAlwaysWins:
    @pytest.mark.asyncio
    async def test_drawdown_suppresses_pace_size_up(self, tmp_path) -> None:
        """Being in drawdown AND behind target is the single most dangerous
        combination: the pace term wants to size up exactly when the de-risk
        term wants to size down. De-risking must win unconditionally."""
        c = _controller(tmp_path, OBJ_PACE_SIZE_ENABLED=True, OBJ_PACE_GATE_ENABLED=True)
        # drawdown_pct well over the 6% limit -> dd_error > 0
        _attach(c, _telemetry(nav=50.0, anchor=100.0, dd_pct=20.0))
        result = await c.step()

        assert result["errors"]["drawdown_pct_over_limit"] > 0
        assert result["pace_applied"] == {}, "pace must not fire while de-risking"
        assert result["knobs_after"]["size_multiplier"] < 1.0, "must de-risk, not size up"


class TestThroughput:
    @pytest.mark.asyncio
    async def test_thru_stays_hardwired_when_flag_off(self, tmp_path) -> None:
        c = _controller(tmp_path, OBJ_PACE_SIZE_ENABLED=True)  # size on, thru off
        _attach(c, _telemetry())
        result = await c.step()

        assert "thru" not in result["pace_applied"]
        assert result["knobs_after"]["target_throughput_per_hour"] == 10.0

    @pytest.mark.asyncio
    async def test_thru_raised_when_enabled_and_behind(self, tmp_path) -> None:
        c = _controller(tmp_path, OBJ_PACE_THRU_ENABLED=True)
        _attach(c, _telemetry())
        result = await c.step()

        assert result["pace_applied"].get("thru", 0) > 0
        assert result["knobs_after"]["target_throughput_per_hour"] > 10.0
