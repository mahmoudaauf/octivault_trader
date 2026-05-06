"""
Tests for Phase 8.2.8 native bootstrap.

Covers:
* BootstrapConfig defaults + from_env validation
* env var coercion (bool/int/float, malformed values)
* build_components produces a wired NativeComponents
* injection seam (exchange_client_factory) avoids real network setup
* shutdown_components stops pollers & closes the HTTP session
* end-to-end: bootstrap -> build_native_app_ctx -> orchestrator runs cycle
"""

from __future__ import annotations

from typing import Any

import pytest

from core_engine.native.app_context import build_native_app_ctx
from core_engine.native.bootstrap import (
    BootstrapConfig,
    build_components,
    shutdown_components,
)


# ---------------------------------------------------------------------
# stub exchange client (no network)
# ---------------------------------------------------------------------
class _StubExchangeClient:
    """Drop-in replacement for NativeExchangeClient — no I/O."""

    def __init__(self, *_, **__):
        self._closed = False
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1
        self._closed = True

    # NativeBalanceSync probes these on the prime call; we make them no-ops.
    async def get_account(self) -> dict[str, Any]:
        return {"balances": []}

    async def get_ticker_prices(self) -> dict[str, float]:
        return {}

    async def get_klines(self, *a, **kw) -> list[Any]:
        return []


def _stub_factory(_cfg: BootstrapConfig) -> Any:
    return _StubExchangeClient()


def _min_cfg(**overrides: Any) -> BootstrapConfig:
    base: dict[str, Any] = {
        "api_key": "k",
        "api_secret": "s",
        "testnet": True,
        "symbols": ["BTCUSDT"],
    }
    base.update(overrides)
    return BootstrapConfig(**base)


# ---------------------------------------------------------------------
# BootstrapConfig
# ---------------------------------------------------------------------
def test_bootstrap_config_defaults():
    cfg = BootstrapConfig(api_key="k", api_secret="s")
    assert cfg.api_key == "k"
    assert cfg.api_secret == "s"
    assert cfg.testnet is False
    assert cfg.symbols  # non-empty default list
    assert cfg.market_data_poll_sec == 2.0
    assert cfg.balance_poll_sec == 5.0
    assert cfg.telemetry_capacity == 1024


def test_bootstrap_config_is_frozen():
    cfg = BootstrapConfig(api_key="k", api_secret="s")
    from dataclasses import FrozenInstanceError

    with pytest.raises(FrozenInstanceError):
        cfg.api_key = "other"  # type: ignore[misc]


def test_from_env_requires_credentials():
    with pytest.raises(ValueError, match="BINANCE_API_KEY"):
        BootstrapConfig.from_env(env={})
    with pytest.raises(ValueError, match="BINANCE_API_KEY"):
        BootstrapConfig.from_env(env={"BINANCE_API_KEY": "k"})  # missing secret
    with pytest.raises(ValueError, match="BINANCE_API_KEY"):
        BootstrapConfig.from_env(env={"BINANCE_API_KEY": "  ", "BINANCE_API_SECRET": "s"})


def test_from_env_parses_full_environment():
    env = {
        "BINANCE_API_KEY": "key123",
        "BINANCE_API_SECRET": "secret456",
        "BINANCE_TESTNET": "true",
        "SYMBOLS": "BTCUSDT, ETHUSDT ,SOLUSDT",
        "MARKET_DATA_POLL_SEC": "1.5",
        "KLINES_CACHE_SIZE": "32",
        "STALE_THRESHOLD_SEC": "20",
        "BALANCE_POLL_SEC": "3",
        "KELLY_FRACTION": "0.5",
        "MAX_POSITION_PCT": "10",
        "MAX_CONCURRENT_POSITIONS": "5",
        "MIN_ORDER_USDT": "20",
        "MAX_DRAWDOWN_PCT": "15",
        "DAILY_LOSS_LIMIT_PCT": "8",
        "RISK_PER_SYMBOL_PCT": "3",
        "SIGNAL_COOLDOWN_SEC": "60",
        "TELEMETRY_CAPACITY": "256",
        "DURATION_SEC": "1800",
        "REQUEST_TIMEOUT_SEC": "5",
    }
    cfg = BootstrapConfig.from_env(env=env)
    assert cfg.api_key == "key123"
    assert cfg.api_secret == "secret456"
    assert cfg.testnet is True
    assert cfg.symbols == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    assert cfg.market_data_poll_sec == 1.5
    assert cfg.klines_cache_size == 32
    assert cfg.balance_poll_sec == 3.0
    assert cfg.kelly_fraction == 0.5
    assert cfg.max_concurrent_positions == 5
    assert cfg.telemetry_capacity == 256
    assert cfg.duration_sec == 1800.0
    assert cfg.request_timeout_sec == 5.0


def test_from_env_falls_back_on_malformed_numbers():
    env = {
        "BINANCE_API_KEY": "k",
        "BINANCE_API_SECRET": "s",
        "MARKET_DATA_POLL_SEC": "not-a-float",
        "KLINES_CACHE_SIZE": "abc",
        "BINANCE_TESTNET": "weird",
    }
    cfg = BootstrapConfig.from_env(env=env)
    assert cfg.market_data_poll_sec == 2.0  # default
    assert cfg.klines_cache_size == 64  # default
    assert cfg.testnet is False  # default


def test_from_env_bool_variants():
    base = {"BINANCE_API_KEY": "k", "BINANCE_API_SECRET": "s"}
    for truthy in ("true", "True", "1", "yes", "ON"):
        cfg = BootstrapConfig.from_env(env={**base, "BINANCE_TESTNET": truthy})
        assert cfg.testnet is True, truthy
    for falsy in ("false", "0", "no", "off", ""):
        cfg = BootstrapConfig.from_env(env={**base, "BINANCE_TESTNET": falsy})
        assert cfg.testnet is False, falsy


def test_from_env_handles_empty_symbols_string():
    cfg = BootstrapConfig.from_env(
        env={"BINANCE_API_KEY": "k", "BINANCE_API_SECRET": "s", "SYMBOLS": ""}
    )
    # Empty SYMBOLS string -> empty list (caller decides whether that's fatal)
    assert cfg.symbols == []


# ---------------------------------------------------------------------
# build_components
# ---------------------------------------------------------------------
@pytest.mark.asyncio
async def test_build_components_returns_wired_native_components():
    cfg = _min_cfg()
    components = await build_components(cfg, exchange_client_factory=_stub_factory)
    # All required fields populated
    assert components.shared_state is not None
    assert components.market_data is not None
    assert components.signal_engine is not None
    assert components.decision_engine is not None
    assert components.executor is not None
    assert components.balance_sync is not None
    assert components.telemetry is not None
    # Telemetry capacity propagated
    assert components.telemetry.capacity == cfg.telemetry_capacity
    # Exchange client surfaced as first-class field (used by safe_execution_engine)
    assert components.exchange_client is not None


@pytest.mark.asyncio
async def test_build_components_propagates_decision_config():
    cfg = _min_cfg(
        kelly_fraction=0.4,
        max_position_size_pct=8.0,
        max_concurrent_positions=3,
        min_order_usdt=25.0,
        max_drawdown_pct=12.0,
        daily_loss_limit_pct=6.0,
        risk_per_symbol_pct=2.5,
    )
    components = await build_components(cfg, exchange_client_factory=_stub_factory)
    de = components.decision_engine
    assert de.kelly_fraction == 0.4
    assert de.max_position_size_pct == 8.0
    assert de.max_concurrent_positions == 3
    assert de.min_order_usdt == 25.0
    assert de.max_drawdown_pct == 12.0
    assert de.daily_loss_limit_pct == 6.0
    assert de.risk_per_symbol_pct == 2.5


@pytest.mark.asyncio
async def test_build_components_uses_injected_exchange_client():
    seen_cfg: list[BootstrapConfig] = []
    stub = _StubExchangeClient()

    def factory(cfg):
        seen_cfg.append(cfg)
        return stub

    cfg = _min_cfg()
    components = await build_components(cfg, exchange_client_factory=factory)
    assert len(seen_cfg) == 1 and seen_cfg[0] is cfg
    # Both balance_sync and market_data should share the same client instance.
    assert components.balance_sync._client is stub
    assert components.market_data._client is stub


# ---------------------------------------------------------------------
# shutdown_components
# ---------------------------------------------------------------------
@pytest.mark.asyncio
async def test_shutdown_components_closes_client_and_stops_pollers():
    stub = _StubExchangeClient()
    cfg = _min_cfg()
    components = await build_components(cfg, exchange_client_factory=lambda _c: stub)
    # Pollers were never started -> stop() must still be safe (idempotent)
    await shutdown_components(components)
    assert stub.close_calls == 1


@pytest.mark.asyncio
async def test_shutdown_components_idempotent():
    stub = _StubExchangeClient()
    components = await build_components(_min_cfg(), exchange_client_factory=lambda _c: stub)
    await shutdown_components(components)
    await shutdown_components(components)
    # client.close was called twice but tolerated; no exception.
    assert stub.close_calls == 2


# ---------------------------------------------------------------------
# end-to-end: bootstrap -> app_ctx -> cycle
# ---------------------------------------------------------------------
@pytest.mark.asyncio
async def test_bootstrap_then_app_ctx_then_cycle_runs():
    stub = _StubExchangeClient()
    components = await build_components(_min_cfg(), exchange_client_factory=lambda _c: stub)
    app_ctx, orch = build_native_app_ctx(components)
    # Documented native ctx keys present
    for k in (
        "shared_state",
        "balance_manager",
        "market_data_feed",
        "signal_manager",
        "decision_engine",
        "execution_manager",
        "telemetry",
        "_native_orchestrator",
        "_native_mode",
    ):
        assert k in app_ctx, f"missing key: {k}"

    metrics = await orch.run_cycle()
    assert metrics.cycle_num == 1
    # telemetry recorded the cycle
    assert len(components.telemetry) == 1
    await shutdown_components(components)
