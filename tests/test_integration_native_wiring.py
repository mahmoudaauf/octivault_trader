"""
Tests for ``core_engine.integration.create_app_context(native=True)``.

Validates that the Phase 8.2.8 wiring of native bootstrap + native
app_context produces a correctly-shaped app_ctx, falls back to mock
mode on failure, and takes precedence over the legacy production
bridge path.
"""

from __future__ import annotations

from typing import Any

import pytest

from core_engine.integration import create_app_context, setup_core_engines


class _StubExchangeClient:
    """No-network drop-in for NativeExchangeClient."""

    def __init__(self, *_, **__):
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1

    async def get_account(self) -> dict[str, Any]:
        return {"balances": []}

    async def get_ticker_prices(self) -> dict[str, float]:
        return {}

    async def get_klines(self, *a, **kw) -> list[Any]:
        return []


@pytest.fixture
def patched_native_factory(monkeypatch):
    """
    Replace the default exchange-client factory used by build_components
    so that calling create_app_context(native=True) does NOT hit the
    network. We patch at the module level so the import inside
    create_app_context picks up our version.
    """
    from core_engine.native import bootstrap as bs

    def _factory(_cfg):
        return _StubExchangeClient()

    monkeypatch.setattr(bs, "_default_exchange_factory", _factory)
    return _factory


@pytest.fixture
def env_with_creds(monkeypatch):
    monkeypatch.setenv("BINANCE_API_KEY", "test-key")
    monkeypatch.setenv("BINANCE_API_SECRET", "test-secret")
    monkeypatch.setenv("BINANCE_TESTNET", "true")
    monkeypatch.setenv("SYMBOLS", "BTCUSDT,ETHUSDT")


# ---------------------------------------------------------------------
# native=True success path
# ---------------------------------------------------------------------
@pytest.mark.asyncio
async def test_native_true_builds_app_ctx_with_documented_keys(
    env_with_creds, patched_native_factory
):
    app_ctx = await create_app_context(native=True)
    for k in (
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
    ):
        assert k in app_ctx, f"missing key: {k}"
    assert app_ctx["_native_mode"] is True


@pytest.mark.asyncio
async def test_native_true_orchestrator_can_run_cycle(env_with_creds, patched_native_factory):
    app_ctx = await create_app_context(native=True)
    orch = app_ctx["_native_orchestrator"]
    metrics = await orch.run_cycle()
    assert metrics.cycle_num == 1
    # telemetry recorded the cycle
    assert len(app_ctx["telemetry"]) == 1


@pytest.mark.asyncio
async def test_setup_core_engines_forwards_native_flag(env_with_creds, patched_native_factory):
    app_ctx = await setup_core_engines(native=True)
    assert app_ctx.get("_native_mode") is True


# ---------------------------------------------------------------------
# Phase 8.3.1: shutdown handle exposed in app_ctx
# ---------------------------------------------------------------------
@pytest.mark.asyncio
async def test_native_components_handle_in_app_ctx(env_with_creds, patched_native_factory):
    """The bootstrap handle must be reachable via _native_components so
    callers (main.py) can run shutdown_components() on cleanup."""
    from core_engine.native.app_context import NativeComponents
    from core_engine.native.bootstrap import shutdown_components

    app_ctx = await create_app_context(native=True)
    components = app_ctx.get("_native_components")
    assert components is not None, "missing _native_components handle"
    assert isinstance(components, NativeComponents)

    # shutdown_components must be idempotent and not raise
    await shutdown_components(components)
    # second call must also be a no-op
    await shutdown_components(components)


@pytest.mark.asyncio
async def test_native_shutdown_closes_exchange_client(env_with_creds, patched_native_factory):
    from core_engine.native.bootstrap import shutdown_components

    app_ctx = await create_app_context(native=True)
    components = app_ctx["_native_components"]
    client = components.exchange_client
    # The stub records close_calls; before shutdown it's untouched.
    assert getattr(client, "close_calls", 0) == 0
    await shutdown_components(components)
    assert client.close_calls >= 1


# ---------------------------------------------------------------------
# native=True failure -> mock fallback
# ---------------------------------------------------------------------
@pytest.mark.asyncio
async def test_native_true_falls_back_to_mock_when_creds_missing(monkeypatch, caplog):
    # Ensure no creds in env
    monkeypatch.delenv("BINANCE_API_KEY", raising=False)
    monkeypatch.delenv("BINANCE_API_SECRET", raising=False)
    import logging

    with caplog.at_level(logging.ERROR, logger="core_engine.integration"):
        app_ctx = await create_app_context(native=True)
    assert app_ctx == {}
    assert any("Native bootstrap failed" in rec.getMessage() for rec in caplog.records)


@pytest.mark.asyncio
async def test_native_true_falls_back_when_factory_raises(env_with_creds, monkeypatch, caplog):
    from core_engine.native import bootstrap as bs

    def _bad_factory(_cfg):
        raise RuntimeError("simulated client construction failure")

    monkeypatch.setattr(bs, "_default_exchange_factory", _bad_factory)

    import logging

    with caplog.at_level(logging.ERROR, logger="core_engine.integration"):
        app_ctx = await create_app_context(native=True)
    assert app_ctx == {}
    assert any("Native bootstrap failed" in rec.getMessage() for rec in caplog.records)


# ---------------------------------------------------------------------
# default mode unchanged
# ---------------------------------------------------------------------
@pytest.mark.asyncio
async def test_default_mode_returns_empty_dict():
    app_ctx = await create_app_context()
    assert app_ctx == {}
