from types import SimpleNamespace

import pytest

from core_engine.native.adaptive_capital_engine import AdaptiveCapitalEngine
from core_engine.native.objective_feedback_controller import (
    ObjectiveFeedbackController,
    Telemetry,
)
from core_engine.native.tp_sl_engine import NativeTPSLEngine


class _SharedState:
    def __init__(self) -> None:
        self.runtime_overrides = {}
        self.positions = {}


@pytest.mark.asyncio
async def test_controller_does_not_chase_pace_with_lower_quality_or_more_size(tmp_path) -> None:
    shared_state = _SharedState()
    controller = ObjectiveFeedbackController(
        shared_state=shared_state,
        artefact_path=str(tmp_path / "ofc.json"),
    )
    controller.state.last_knobs = {
        "confidence_floor": 0.65,
        "size_multiplier": 1.0,
        "target_throughput_per_hour": 0.5,
    }

    async def measure() -> Telemetry:
        return Telemetry(
            ok=True,
            nav=100.0,
            nav_anchor=100.0,
            elapsed_h=1.0,
            trades_in_window=1,
            avg_net_profit_bps=5.0,
        )

    controller._measure = measure  # type: ignore[method-assign]
    result = await controller.step()

    assert result["errors"]["pace_pct_per_h"] < 0
    assert result["knobs_after"]["confidence_floor"] == 0.65
    assert result["knobs_after"]["size_multiplier"] == 1.0


@pytest.mark.asyncio
async def test_controller_holds_quality_knobs_when_no_trade_qualifies(tmp_path) -> None:
    controller = ObjectiveFeedbackController(
        shared_state=_SharedState(),
        artefact_path=str(tmp_path / "ofc.json"),
    )
    controller.state.last_knobs = {
        "confidence_floor": 0.70,
        "size_multiplier": 0.8,
        "target_throughput_per_hour": 0.5,
    }

    async def measure() -> Telemetry:
        return Telemetry(
            ok=True,
            nav=100.0,
            nav_anchor=100.0,
            elapsed_h=1.0,
            trades_in_window=0,
        )

    controller._measure = measure  # type: ignore[method-assign]
    result = await controller.step()

    assert result["knobs_after"]["confidence_floor"] == 0.70
    assert result["knobs_after"]["size_multiplier"] == 0.8


def test_adaptive_sizing_does_not_boost_risk_for_idle_or_low_throughput() -> None:
    config = SimpleNamespace(
        ADAPTIVE_CAPITAL_ENGINE_ENABLED=True,
        ADAPTIVE_RISK_FRACTION_MIN=0.01,
        ADAPTIVE_RISK_FRACTION_MAX=0.50,
        DEFAULT_PLANNED_QUOTE=10.0,
    )
    engine = AdaptiveCapitalEngine(config)

    decision = engine.evaluate(
        symbol="BTCUSDT",
        nav=1_000.0,
        free_capital=1_000.0,
        base_risk_fraction=0.20,
        volatility_pct=0.01,
        drawdown_pct=0.0,
        fee_bps=10.0,
        slippage_bps=5.0,
        min_notional=10.0,
        slot_utilization=0.0,
        throughput_per_hour=0.0,
        target_throughput_per_hour=10.0,
        trade_history=[],
        now_ts=1_000.0,
    )

    assert decision.risk_fraction == pytest.approx(0.20)
    assert "throughput_low_up" not in decision.reasons
    assert "idle_capital_boost" not in decision.reasons


def test_tp_sl_uses_same_configured_round_trip_cost_as_entry_gate(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    shared_state = SimpleNamespace(positions={}, prices={}, metrics={})
    config = SimpleNamespace(
        taker_fee_bps=8.0,
        exit_slippage_bps=9.0,
        TP_ATR_MULT=1.5,
        SL_ATR_MULT=1.5,
        TARGET_RISK_PCT=2.0,
        ATR_LOOKBACK=14,
        MIN_ATR_PCT=0.005,
        TPSL_VOL_ADAPTATION_ENABLED=True,
        VOL_PRESSURE_SCALE=0.35,
        MIN_NOTIONAL_SAFETY=10.0,
        TPSL_AUTO_ARM_ENABLED=True,
    )

    engine = NativeTPSLEngine(shared_state, config)

    assert engine._round_trip_cost_pct == pytest.approx(0.0025)
