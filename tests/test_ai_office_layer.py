from __future__ import annotations

import pytest

from ai_office.ai_office_manager import AIOfficeManager
from ai_office.policy_bus import PolicyBus
from ai_office.schemas import AIOfficeRecommendation
from core_engine.native.shared_state import NativeSharedState


@pytest.mark.asyncio
async def test_policy_bus_rejects_unsafe_actions() -> None:
    state = NativeSharedState()
    bus = PolicyBus(state)
    rec = AIOfficeRecommendation(
        source="test",
        recommendation_type="risk_policy",
        confidence=0.9,
        reason="bad",
        severity="CRITICAL",
        target_component="RiskManager",
        suggested_action="place_order",
    )

    decision = await bus.validate(rec)

    assert decision.accepted is False
    assert decision.reason == "unsafe_action:place_order"


@pytest.mark.asyncio
async def test_ai_office_snapshot_builds_from_native_shared_state() -> None:
    state = NativeSharedState()
    state.nav_usdt = 100.0
    state.previous_nav_usdt = 98.0
    state.peak_nav_usdt = 110.0
    state.free_balance_usdt = 40.0
    state.metrics["realized_pnl"] = 5.0
    state.metrics["unrealized_pnl"] = 3.0
    state.metrics["volatility_regime"] = "trend"
    state.nav_protection_state["protection_mode"] = "NORMAL"
    state.positions = {"BTCUSDT": {"qty": 0.1}, "ETHUSDT": {"qty": 1.0}}
    state.position_recovery = {
        "BTCUSDT": {"status": "STALE", "notional_usdt": 20.0},
        "ETHUSDT": {"status": "DUST", "notional_usdt": 5.0},
    }
    manager = AIOfficeManager(shared_state=state, review_interval_sec=28800)

    snapshot = await manager.build_snapshot()

    assert snapshot.nav_usdt == 100.0
    assert snapshot.free_usdt == 40.0
    assert snapshot.locked_usdt == 60.0
    assert snapshot.positions_count == 2
    assert snapshot.stale_positions_count == 1
    assert snapshot.dust_ratio == pytest.approx(0.05)


@pytest.mark.asyncio
async def test_ai_office_routes_pause_new_buys_to_risk_manager() -> None:
    state = NativeSharedState()

    class DummyRisk:
        def __init__(self) -> None:
            self.frozen = False

        def freeze_trading(self, reason: str = "") -> None:
            self.frozen = True

    risk = DummyRisk()
    manager = AIOfficeManager(shared_state=state, risk_manager=risk)
    rec = AIOfficeRecommendation(
        source="test",
        recommendation_type="risk_policy",
        confidence=0.9,
        reason="drawdown",
        severity="CRITICAL",
        target_component="RiskManager",
        suggested_action="pause_new_buys",
    )
    decision = await manager._bus.validate(rec)
    assert decision.accepted is True

    await manager._route(decision)
    assert risk.frozen is True
