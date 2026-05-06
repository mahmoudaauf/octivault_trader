"""
Tests for Phase 8.2.8 compat stubs (core_engine.native.compat).

These verify:
* The null-object behaves correctly (truthy, async no-op methods).
* `register_compat_stubs` only fills missing keys.
* `build_native_app_ctx(..., compat=True)` installs all six stubs.
* `build_native_app_ctx(..., compat=False)` (default) installs none.
* `create_app_context(native=True, compat=True)` propagates the flag.
"""

from __future__ import annotations

import pytest

from core_engine.native.app_context import (
    NativeComponents,
    build_native_app_ctx,
)
from core_engine.native.compat import (
    COMPAT_KEYS,
    _NullStub,
    make_compat_stubs,
    register_compat_stubs,
)
from core_engine.native.observability import NativeTelemetry


# ---------------------------------------------------------------------
# stubs (re-used from test_native_app_context.py shape)
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


def _components() -> NativeComponents:
    return NativeComponents(
        shared_state=_State(),  # type: ignore[arg-type]
        market_data=_MD(),  # type: ignore[arg-type]
        signal_engine=_Sig(),  # type: ignore[arg-type]
        decision_engine=_Dec(),  # type: ignore[arg-type]
        executor=_Exe(),  # type: ignore[arg-type]
        balance_sync=_Bal(),  # type: ignore[arg-type]
        telemetry=NativeTelemetry(),
    )


# ---------------------------------------------------------------------
# _NullStub
# ---------------------------------------------------------------------
def test_null_stub_is_truthy():
    s = _NullStub("portfolio_manager")
    assert bool(s) is True
    if not s:
        pytest.fail("NullStub should be truthy so `if xxx:` checks pass")


def test_null_stub_repr_includes_name():
    s = _NullStub("watchdog")
    assert "watchdog" in repr(s)


@pytest.mark.asyncio
async def test_null_stub_async_method_returns_none():
    s = _NullStub("recovery_engine")
    result = await s.recover()
    assert result is None


@pytest.mark.asyncio
async def test_null_stub_async_method_accepts_arbitrary_args():
    s = _NullStub("safety_order_manager")
    # façades will pass positional + kwargs; must not raise
    result = await s.place_oco("BTCUSDT", 0.1, 50000.0, take_profit=51000.0)
    assert result is None


def test_null_stub_method_identity_is_stable():
    """Repeated attribute access must return the same callable (identity)."""
    s = _NullStub("watchdog")
    a = s.check_liveness
    b = s.check_liveness
    assert a is b


def test_null_stub_distinct_methods_are_distinct():
    s = _NullStub("watchdog")
    assert s.method_one is not s.method_two


# ---------------------------------------------------------------------
# make_compat_stubs / register_compat_stubs
# ---------------------------------------------------------------------
def test_make_compat_stubs_covers_all_compat_keys():
    stubs = make_compat_stubs()
    assert set(stubs.keys()) == set(COMPAT_KEYS)
    assert all(isinstance(v, _NullStub) for v in stubs.values())


def test_register_compat_stubs_fills_missing_keys():
    app_ctx: dict = {}
    register_compat_stubs(app_ctx)
    for key in COMPAT_KEYS:
        assert key in app_ctx
        assert isinstance(app_ctx[key], _NullStub)


def test_register_compat_stubs_does_not_overwrite_existing():
    sentinel = object()
    app_ctx: dict = {"watchdog": sentinel}
    register_compat_stubs(app_ctx)
    assert app_ctx["watchdog"] is sentinel  # untouched
    # but other compat keys should now be present
    assert isinstance(app_ctx["recovery_engine"], _NullStub)


def test_compat_keys_excludes_dropped_keys():
    """The 5 keys decided as drops must NOT be in COMPAT_KEYS."""
    dropped = {
        "risk_manager",
        "signal_fusion",
        "arbitration_engine",
        "startup_orchestrator",
        "performance_monitor",
    }
    assert dropped.isdisjoint(set(COMPAT_KEYS))


# ---------------------------------------------------------------------
# build_native_app_ctx integration
# ---------------------------------------------------------------------
def test_build_native_app_ctx_compat_false_by_default():
    app_ctx, _ = build_native_app_ctx(_components())
    for key in COMPAT_KEYS:
        assert key not in app_ctx, f"compat key {key!r} leaked into default app_ctx"


def test_build_native_app_ctx_compat_true_installs_stubs():
    app_ctx, _ = build_native_app_ctx(_components(), compat=True)
    for key in COMPAT_KEYS:
        assert key in app_ctx
        assert isinstance(app_ctx[key], _NullStub)
        # Native-managed keys must remain unaffected
    assert "_native_mode" in app_ctx
    assert app_ctx["_native_mode"] is True


@pytest.mark.asyncio
async def test_build_native_app_ctx_compat_true_does_not_break_cycle():
    app_ctx, orch = build_native_app_ctx(_components(), compat=True)
    metrics = await orch.run_cycle()
    assert metrics.cycle_num == 1
    # Stubs are present but orchestrator never touches them
    assert isinstance(app_ctx["portfolio_manager"], _NullStub)


# ---------------------------------------------------------------------
# integration.create_app_context propagation
# ---------------------------------------------------------------------
@pytest.mark.asyncio
async def test_create_app_context_propagates_compat_flag(monkeypatch):
    """``create_app_context(native=True, compat=True)`` must install stubs."""
    from core_engine import integration
    from core_engine.native import bootstrap as bs

    # Reuse the stub factory pattern from test_integration_native_wiring.py
    class _StubExchangeClient:
        async def get_klines(self, *a, **kw):
            return []

        async def get_prices(self, *a, **kw):
            return {}

        async def get_balance(self, *a, **kw):
            return {}

        async def place_order(self, *a, **kw):
            return {"orderId": "x", "status": "FILLED"}

        async def cancel_order(self, *a, **kw):
            return {}

        async def close(self):
            pass

    monkeypatch.setattr(bs, "_default_exchange_factory", lambda _cfg: _StubExchangeClient())
    monkeypatch.setenv("BINANCE_API_KEY", "k")
    monkeypatch.setenv("BINANCE_API_SECRET", "s")
    monkeypatch.setenv("SYMBOLS", "BTCUSDT")

    app_ctx = await integration.create_app_context(native=True, compat=True)
    assert app_ctx.get("_native_mode") is True
    for key in COMPAT_KEYS:
        assert key in app_ctx, f"compat key {key!r} missing when compat=True"
        assert isinstance(app_ctx[key], _NullStub)


@pytest.mark.asyncio
async def test_create_app_context_native_without_compat_omits_stubs(monkeypatch):
    from core_engine import integration
    from core_engine.native import bootstrap as bs

    class _StubExchangeClient:
        async def get_klines(self, *a, **kw):
            return []

        async def get_prices(self, *a, **kw):
            return {}

        async def get_balance(self, *a, **kw):
            return {}

        async def place_order(self, *a, **kw):
            return {"orderId": "x", "status": "FILLED"}

        async def cancel_order(self, *a, **kw):
            return {}

        async def close(self):
            pass

    monkeypatch.setattr(bs, "_default_exchange_factory", lambda _cfg: _StubExchangeClient())
    monkeypatch.setenv("BINANCE_API_KEY", "k")
    monkeypatch.setenv("BINANCE_API_SECRET", "s")
    monkeypatch.setenv("SYMBOLS", "BTCUSDT")

    app_ctx = await integration.create_app_context(native=True)  # compat default False
    for key in COMPAT_KEYS:
        assert key not in app_ctx, f"compat key {key!r} leaked when compat=False"
