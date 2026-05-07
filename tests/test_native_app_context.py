"""
Tests for Phase 8.2.8 (preparation):
- core_engine.native.app_context.build_native_app_ctx
- core_engine.production_bridge deprecation warning
"""

from __future__ import annotations

import pytest

from core_engine.native.app_context import (
    NATIVE_CTX_KEYS,
    NativeComponents,
    build_native_app_ctx,
)
from core_engine.native.observability import NativeTelemetry
from core_engine.native.orchestrator import NativeOrchestrator


# ---------------------------------------------------------------------
# stubs (avoid heavy native instantiation; use duck-typed objects that
# satisfy NativeOrchestrator's requirements)
# ---------------------------------------------------------------------
class _MD:
    async def start(self):
        pass

    async def stop(self):
        pass

    def get_prices(self):
        return {}

    async def get_klines(self, *a, **kw):
        return []


class _Sig:
    def evaluate(self, *a, **kw):
        return None


class _Dec:
    def decide(self, *a, **kw):
        return []


class _Exe:
    async def execute(self, *a, **kw):
        return []


class _Bal:
    async def start(self):
        pass

    async def stop(self):
        pass


class _State:
    def __init__(self):
        self.nav = 1234.0


def _components(*, telemetry: NativeTelemetry | None = None) -> NativeComponents:
    # NativeComponents type-hints concrete classes; we bypass with cast-by-init
    # because the dataclass has no runtime isinstance enforcement (frozen=True only).
    return NativeComponents(
        shared_state=_State(),  # type: ignore[arg-type]
        market_data=_MD(),  # type: ignore[arg-type]
        signal_engine=_Sig(),  # type: ignore[arg-type]
        decision_engine=_Dec(),  # type: ignore[arg-type]
        executor=_Exe(),  # type: ignore[arg-type]
        balance_sync=_Bal(),  # type: ignore[arg-type]
        telemetry=telemetry,
    )


# ---------------------------------------------------------------------
# build_native_app_ctx
# ---------------------------------------------------------------------
def test_build_native_app_ctx_returns_ctx_and_orchestrator():
    app_ctx, orch = build_native_app_ctx(_components())
    assert isinstance(app_ctx, dict)
    assert isinstance(orch, NativeOrchestrator)


def test_build_native_app_ctx_exposes_documented_keys():
    app_ctx, _ = build_native_app_ctx(_components())
    # All non-optional keys present
    required = {
        "shared_state",
        "balance_manager",
        "market_data_feed",
        "signal_manager",
        "decision_engine",
        "execution_manager",
        "_native_orchestrator",
        "_native_mode",
    }
    assert required.issubset(app_ctx.keys())
    assert app_ctx["_native_mode"] is True


def test_build_native_app_ctx_omits_telemetry_when_none():
    app_ctx, _ = build_native_app_ctx(_components(telemetry=None))
    assert "telemetry" not in app_ctx


def test_build_native_app_ctx_includes_telemetry_when_provided():
    t = NativeTelemetry()
    app_ctx, _ = build_native_app_ctx(_components(telemetry=t))
    assert app_ctx["telemetry"] is t


def test_build_native_app_ctx_omits_exchange_client_when_none():
    app_ctx, _ = build_native_app_ctx(_components())
    assert "exchange_client" not in app_ctx


def test_build_native_app_ctx_includes_exchange_client_when_provided():
    sentinel = object()
    components = NativeComponents(
        shared_state=_State(),  # type: ignore[arg-type]
        market_data=_MD(),  # type: ignore[arg-type]
        signal_engine=_Sig(),  # type: ignore[arg-type]
        decision_engine=_Dec(),  # type: ignore[arg-type]
        executor=_Exe(),  # type: ignore[arg-type]
        balance_sync=_Bal(),  # type: ignore[arg-type]
        exchange_client=sentinel,
    )
    app_ctx, _ = build_native_app_ctx(components)
    assert app_ctx["exchange_client"] is sentinel


def test_orchestrator_handle_is_same_instance_in_ctx():
    app_ctx, orch = build_native_app_ctx(_components())
    assert app_ctx["_native_orchestrator"] is orch


def test_native_ctx_keys_constant_advertises_stable_contract():
    # Sanity: declared keys cover everything we actually populate
    expected_subset = {
        "shared_state",
        "balance_manager",
        "market_data_feed",
        "signal_manager",
        "decision_engine",
        "execution_manager",
        "signal_fusion",
        "arbitration_engine",
        "market_regime_detector",
        "telemetry",
        "health_monitor",
        "_native_orchestrator",
        "_native_mode",
    }
    assert expected_subset.issubset(set(NATIVE_CTX_KEYS))


@pytest.mark.asyncio
async def test_orchestrator_built_by_factory_runs_one_cycle():
    t = NativeTelemetry()
    _ctx, orch = build_native_app_ctx(_components(telemetry=t))
    metrics = await orch.run_cycle()
    assert metrics.cycle_num == 1
    assert len(t) == 1
    assert t.latest() is metrics


# ---------------------------------------------------------------------
# production_bridge removal sanity (Phase 8.2.8 step 6)
# ---------------------------------------------------------------------
def test_production_bridge_module_is_removed():
    """The legacy bridge module must not be importable any more."""
    import importlib

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("core_engine.production_bridge")
