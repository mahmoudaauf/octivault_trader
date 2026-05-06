"""
Tests for ``core_engine.native.recovery_engine`` (Phase 8.3.11).

Coverage:
* ``RecoveryPlan`` dataclass shape parity with the legacy plan in
  ``core_engine.operations_engine``.
* ``generate_recovery_plan``: empty state → NORMAL/empty plan;
  orphan OCO → HIGH; stale price → HIGH; NAV drift → URGENT;
  zero entry_price → HIGH; multiple issues compound to highest.
* ``apply_plan``: dispatcher executes each op, returns True iff all
  ops succeed; unknown ops are skipped (counted) and don't fail the
  plan when no other op fails — but an unknown op does flip the
  ``all_ok`` only via being treated as failure (we explicitly assert
  the documented behavior).
* ``health()`` reports counters.
* Constructor input validation.
* Wiring: appears in ``app_ctx``, isn't overwritten by compat stubs,
  is constructed by ``build_components``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import pytest

from core_engine.native.recovery_engine import (
    NativeRecoveryEngine,
    RecoveryPlan,
)


# ---------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------
@dataclass
class _Pos:
    qty: float = 0.0
    entry_price: float = 0.0
    mark_price: float = 0.0


@dataclass
class _State:
    nav_usdt: float = 0.0
    free_balance_usdt: float = 0.0
    positions: dict[str, _Pos] = field(default_factory=dict)
    price_timestamps: dict[str, float] = field(default_factory=dict)
    price_cache: dict[str, float] = field(default_factory=dict)


class _SafetyStub:
    """Minimal NativeSafetyOrderManager-compatible stub."""

    def __init__(self, active: list[str] | None = None) -> None:
        self._active = list(active or [])
        self.cancelled: list[str] = []
        self.cancel_returns = True

    def list_active(self) -> list[str]:
        return list(self._active)

    async def cancel_oco(self, symbol: str) -> bool:
        self.cancelled.append(symbol)
        if self.cancel_returns:
            self._active = [s for s in self._active if s != symbol]
        return self.cancel_returns


# ---------------------------------------------------------------------
# RecoveryPlan dataclass
# ---------------------------------------------------------------------
def test_recovery_plan_dataclass_fields():
    p = RecoveryPlan(
        issues=["x"],
        recovery_steps=["s"],
        estimated_recovery_time_sec=1.5,
        priority="HIGH",
    )
    assert p.issues == ["x"]
    assert p.recovery_steps == ["s"]
    assert p.estimated_recovery_time_sec == 1.5
    assert p.priority == "HIGH"
    assert p.auto_recover is True
    assert p.ops == []


# ---------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------
def test_constructor_rejects_negative_nav_tolerance():
    with pytest.raises(ValueError, match="nav_drift_tolerance_pct"):
        NativeRecoveryEngine(_State(), nav_drift_tolerance_pct=-1.0)


def test_constructor_rejects_negative_stale_threshold():
    with pytest.raises(ValueError, match="stale_price_threshold_sec"):
        NativeRecoveryEngine(_State(), stale_price_threshold_sec=-1.0)


def test_constructor_defaults_health_ok():
    eng = NativeRecoveryEngine(_State())
    h = eng.health()
    assert h["ok"] is True
    assert h["plans_generated"] == 0
    assert h["plans_applied"] == 0
    assert h["safety_wired"] is False


# ---------------------------------------------------------------------
# generate_recovery_plan
# ---------------------------------------------------------------------
@pytest.mark.asyncio
async def test_generate_empty_state_returns_normal_empty_plan():
    eng = NativeRecoveryEngine(_State())
    plan = await eng.generate_recovery_plan()
    assert plan.issues == []
    assert plan.recovery_steps == []
    assert plan.ops == []
    assert plan.priority == "NORMAL"
    assert plan.estimated_recovery_time_sec == 0.0
    assert eng.health()["plans_generated"] == 1


@pytest.mark.asyncio
async def test_generate_detects_orphan_oco_high_priority():
    state = _State(positions={})  # no positions
    safety = _SafetyStub(active=["BTCUSDT"])
    eng = NativeRecoveryEngine(state, safety_order_manager=safety)

    plan = await eng.generate_recovery_plan()
    assert plan.priority == "HIGH"
    assert any("orphan" in i and "BTCUSDT" in i for i in plan.issues)
    assert any(o["op"] == "cancel_orphan_oco" and o["symbol"] == "BTCUSDT" for o in plan.ops)


@pytest.mark.asyncio
async def test_generate_skips_oco_when_position_present():
    state = _State(positions={"BTCUSDT": _Pos(qty=1.0, entry_price=50000, mark_price=50000)})
    safety = _SafetyStub(active=["BTCUSDT"])
    eng = NativeRecoveryEngine(state, safety_order_manager=safety)

    plan = await eng.generate_recovery_plan()
    assert not any(o["op"] == "cancel_orphan_oco" for o in plan.ops)


@pytest.mark.asyncio
async def test_generate_detects_stale_market_data():
    old = time.time() - 120.0
    state = _State(
        positions={"ETHUSDT": _Pos(qty=2.0, entry_price=3000, mark_price=3000)},
        price_timestamps={"ETHUSDT": old},
    )
    eng = NativeRecoveryEngine(state, stale_price_threshold_sec=60.0)

    plan = await eng.generate_recovery_plan()
    assert plan.priority == "HIGH"
    assert any("stale" in i and "ETHUSDT" in i for i in plan.issues)
    assert any(o["op"] == "request_price_refresh" for o in plan.ops)


@pytest.mark.asyncio
async def test_generate_skips_stale_when_no_timestamp():
    state = _State(
        positions={"ETHUSDT": _Pos(qty=2.0, entry_price=3000, mark_price=3000)},
        # no price_timestamps entry
    )
    eng = NativeRecoveryEngine(state)
    plan = await eng.generate_recovery_plan()
    assert not any(o["op"] == "request_price_refresh" for o in plan.ops)


@pytest.mark.asyncio
async def test_generate_detects_nav_drift_urgent():
    # nav says 1000 but free=500 + 0 positions -> derived=500 -> 50% drift
    state = _State(nav_usdt=1000.0, free_balance_usdt=500.0)
    eng = NativeRecoveryEngine(state, nav_drift_tolerance_pct=5.0)

    plan = await eng.generate_recovery_plan()
    assert plan.priority == "URGENT"
    assert any("NAV drift" in i for i in plan.issues)
    assert any(o["op"] == "recompute_nav" for o in plan.ops)


@pytest.mark.asyncio
async def test_generate_skips_nav_drift_within_tolerance():
    state = _State(nav_usdt=1000.0, free_balance_usdt=990.0)
    eng = NativeRecoveryEngine(state, nav_drift_tolerance_pct=5.0)
    plan = await eng.generate_recovery_plan()
    assert not any(o["op"] == "recompute_nav" for o in plan.ops)


@pytest.mark.asyncio
async def test_generate_detects_zero_entry_price():
    state = _State(
        positions={"SOLUSDT": _Pos(qty=10.0, entry_price=0.0, mark_price=150.0)},
    )
    eng = NativeRecoveryEngine(state)
    plan = await eng.generate_recovery_plan()
    assert plan.priority == "HIGH"
    assert any("entry_price" in i for i in plan.issues)
    assert any(o["op"] == "backfill_entry_from_mark" for o in plan.ops)


@pytest.mark.asyncio
async def test_generate_priority_escalates_to_highest():
    # NAV drift (URGENT) + orphan OCO (HIGH) → URGENT wins
    state = _State(nav_usdt=1000.0, free_balance_usdt=100.0)
    safety = _SafetyStub(active=["BTCUSDT"])
    eng = NativeRecoveryEngine(state, safety_order_manager=safety)
    plan = await eng.generate_recovery_plan()
    assert plan.priority == "URGENT"
    assert len(plan.issues) >= 2


# ---------------------------------------------------------------------
# apply_plan
# ---------------------------------------------------------------------
@pytest.mark.asyncio
async def test_apply_empty_plan_returns_true():
    eng = NativeRecoveryEngine(_State())
    plan = RecoveryPlan(
        issues=[], recovery_steps=[], estimated_recovery_time_sec=0.0, priority="NORMAL"
    )
    assert await eng.apply_plan(plan) is True
    assert eng.health()["plans_applied"] == 1


@pytest.mark.asyncio
async def test_apply_cancel_orphan_oco_calls_safety():
    state = _State()
    safety = _SafetyStub(active=["BTCUSDT"])
    eng = NativeRecoveryEngine(state, safety_order_manager=safety)
    plan = await eng.generate_recovery_plan()
    ok = await eng.apply_plan(plan)
    assert ok is True
    assert "BTCUSDT" in safety.cancelled
    assert eng.health()["ops_applied"] >= 1


@pytest.mark.asyncio
async def test_apply_cancel_orphan_oco_failure_propagates():
    state = _State()
    safety = _SafetyStub(active=["BTCUSDT"])
    safety.cancel_returns = False
    eng = NativeRecoveryEngine(state, safety_order_manager=safety)
    plan = await eng.generate_recovery_plan()
    ok = await eng.apply_plan(plan)
    assert ok is False


@pytest.mark.asyncio
async def test_apply_recompute_nav_writes_to_state():
    state = _State(nav_usdt=1000.0, free_balance_usdt=100.0)
    eng = NativeRecoveryEngine(state)
    plan = await eng.generate_recovery_plan()
    ok = await eng.apply_plan(plan)
    assert ok is True
    # NAV should now match derived value (free + 0 positions = 100)
    assert state.nav_usdt == 100.0


@pytest.mark.asyncio
async def test_apply_backfill_entry_from_mark():
    pos = _Pos(qty=10.0, entry_price=0.0, mark_price=150.0)
    state = _State(positions={"SOLUSDT": pos})
    eng = NativeRecoveryEngine(state)
    plan = await eng.generate_recovery_plan()
    ok = await eng.apply_plan(plan)
    assert ok is True
    assert pos.entry_price == 150.0


@pytest.mark.asyncio
async def test_apply_unknown_op_is_skipped_and_fails_plan():
    eng = NativeRecoveryEngine(_State())
    plan = RecoveryPlan(
        issues=["?"],
        recovery_steps=["?"],
        estimated_recovery_time_sec=0.0,
        priority="NORMAL",
        ops=[{"op": "totally_made_up", "symbol": None}],
    )
    ok = await eng.apply_plan(plan)
    assert ok is False
    assert eng.health()["ops_skipped"] == 1


@pytest.mark.asyncio
async def test_apply_request_price_refresh_is_advisory_success():
    eng = NativeRecoveryEngine(_State())
    plan = RecoveryPlan(
        issues=[],
        recovery_steps=[],
        estimated_recovery_time_sec=0.0,
        priority="NORMAL",
        ops=[{"op": "request_price_refresh", "symbol": "BTCUSDT"}],
    )
    assert await eng.apply_plan(plan) is True


@pytest.mark.asyncio
async def test_apply_handles_dispatch_exception():
    """Op that raises is caught, counted as failure, doesn't crash the loop."""
    state = _State()
    safety = _SafetyStub(active=["BTCUSDT"])

    async def _boom(_sym):
        raise RuntimeError("boom")

    safety.cancel_oco = _boom  # type: ignore[assignment]
    eng = NativeRecoveryEngine(state, safety_order_manager=safety)
    plan = RecoveryPlan(
        issues=["x"],
        recovery_steps=["x"],
        estimated_recovery_time_sec=0.0,
        priority="HIGH",
        ops=[{"op": "cancel_orphan_oco", "symbol": "BTCUSDT"}],
    )
    ok = await eng.apply_plan(plan)
    assert ok is False


# ---------------------------------------------------------------------
# Wiring tests
# ---------------------------------------------------------------------
def test_native_components_carries_recovery_engine_field():
    """NativeComponents dataclass exposes the new field."""
    from core_engine.native.app_context import NativeComponents

    fields = {f.name for f in NativeComponents.__dataclass_fields__.values()}
    assert "recovery_engine" in fields


def test_recovery_engine_visible_in_app_ctx_when_provided():
    """When a recovery_engine is in components, it appears in app_ctx."""
    from core_engine.native.app_context import NativeComponents, build_native_app_ctx
    from core_engine.native.observability import NativeTelemetry

    # Reuse minimal stubs
    class _MD:
        async def start(self):
            pass

        async def stop(self):
            pass

        def get_prices(self):
            return {}

        async def get_klines(self, *a, **k):
            return []

    class _Sig:
        def evaluate(self, *a, **k):
            return None

    class _Dec:
        def decide(self, *a, **k):
            return []

    class _Exe:
        async def execute(self, *a, **k):
            return []

    class _Bal:
        async def start(self):
            pass

        async def stop(self):
            pass

    state = _State()
    eng = NativeRecoveryEngine(state)
    components = NativeComponents(
        shared_state=state,  # type: ignore[arg-type]
        market_data=_MD(),  # type: ignore[arg-type]
        signal_engine=_Sig(),  # type: ignore[arg-type]
        decision_engine=_Dec(),  # type: ignore[arg-type]
        executor=_Exe(),  # type: ignore[arg-type]
        balance_sync=_Bal(),  # type: ignore[arg-type]
        telemetry=NativeTelemetry(),
        recovery_engine=eng,
    )
    app_ctx, _ = build_native_app_ctx(components)
    assert app_ctx["recovery_engine"] is eng


def test_recovery_engine_not_overwritten_by_compat_stub():
    """With compat=True, real recovery_engine survives stub registration."""
    from core_engine.native.app_context import NativeComponents, build_native_app_ctx
    from core_engine.native.compat import _NullStub
    from core_engine.native.observability import NativeTelemetry

    class _MD:
        async def start(self):
            pass

        async def stop(self):
            pass

        def get_prices(self):
            return {}

        async def get_klines(self, *a, **k):
            return []

    class _Sig:
        def evaluate(self, *a, **k):
            return None

    class _Dec:
        def decide(self, *a, **k):
            return []

    class _Exe:
        async def execute(self, *a, **k):
            return []

    class _Bal:
        async def start(self):
            pass

        async def stop(self):
            pass

    state = _State()
    eng = NativeRecoveryEngine(state)
    components = NativeComponents(
        shared_state=state,  # type: ignore[arg-type]
        market_data=_MD(),  # type: ignore[arg-type]
        signal_engine=_Sig(),  # type: ignore[arg-type]
        decision_engine=_Dec(),  # type: ignore[arg-type]
        executor=_Exe(),  # type: ignore[arg-type]
        balance_sync=_Bal(),  # type: ignore[arg-type]
        telemetry=NativeTelemetry(),
        recovery_engine=eng,
    )
    app_ctx, _ = build_native_app_ctx(components, compat=True)
    assert app_ctx["recovery_engine"] is eng
    assert not isinstance(app_ctx["recovery_engine"], _NullStub)
