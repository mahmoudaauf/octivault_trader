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

import json
import time
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
        self.restored_throttle_until_ts = 0.0
        self.restored_throttle_reason = ""

    async def close(self) -> None:
        self.close_calls += 1
        self._closed = True

    def restore_throttle_state(self, *, until_ts: float = 0.0, reason: str = "") -> None:
        self.restored_throttle_until_ts = float(until_ts or 0.0)
        self.restored_throttle_reason = str(reason or "")

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
        "runtime_state_path": "",
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
    assert cfg.symbols == []
    assert cfg.market_data_poll_sec == 2.0
    assert cfg.balance_poll_sec == 5.0
    assert cfg.telemetry_capacity == 1024
    assert cfg.runtime_state_path == "runtime_state_snapshot.json"


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
        "QUOTE_RESERVE_RATIO": "0.2",
        "QUOTE_MIN_RESERVE_USDT": "5",
        "MAX_TOTAL_EXPOSURE_PCT": "55",
        "CONFIDENCE_FLOOR": "0.6",
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
    assert cfg.quote_reserve_ratio == 0.2
    assert cfg.quote_min_reserve_usdt == 5.0
    assert cfg.max_total_exposure_pct == 55.0
    assert cfg.confidence_floor == 0.6
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
    assert components.balance_sync is None
    assert components.polling_coordinator is not None
    assert components.telemetry is not None
    assert components.mode_manager is not None
    assert components.signal_fusion is not None
    assert components.arbitration_engine is not None
    assert components.market_regime_detector is not None
    assert components.health_monitor is not None
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
    assert components.exchange_client is stub
    assert components.market_data._client is stub


@pytest.mark.asyncio
async def test_build_components_restores_throttle_into_exchange_client(tmp_path):
    snapshot = tmp_path / "runtime_state.json"
    until_ts = time.time() + 300.0
    snapshot.write_text(
        json.dumps(
            {
                "exchange_throttled": True,
                "exchange_throttle_reason": "persisted 418",
                "exchange_throttle_until_ts": until_ts,
            }
        )
    )
    stub = _StubExchangeClient()
    cfg = _min_cfg(runtime_state_path=str(snapshot))

    components = await build_components(cfg, exchange_client_factory=lambda _c: stub)
    assert components.shared_state.exchange_throttled is True
    assert components.shared_state.exchange_throttle_until_ts == until_ts
    assert stub.restored_throttle_until_ts == until_ts
    assert stub.restored_throttle_reason == "persisted 418"


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
        "mode_manager",
        "telemetry",
        "_native_orchestrator",
        "_native_mode",
    ):
        assert k in app_ctx, f"missing key: {k}"

    metrics = await orch.run_cycle()
    assert metrics.cycle_num == 1
    # telemetry recorded the cycle
    assert len(components.telemetry) == 1


# ---------------------------------------------------------------------
# portfolio_accessor (Phase 8.2.8 step 5a follow-up)
# ---------------------------------------------------------------------
@pytest.mark.asyncio
async def test_build_components_attaches_portfolio_accessor():
    stub = _StubExchangeClient()
    components = await build_components(_min_cfg(), exchange_client_factory=lambda _c: stub)
    assert components.portfolio_accessor is not None
    snap = components.portfolio_accessor()
    # Duck-typed contract NativeDecisionEngine.decide() relies on:
    assert hasattr(snap, "nav")
    assert hasattr(snap, "nav_peak")
    assert hasattr(snap, "balance")
    assert hasattr(snap, "positions")


@pytest.mark.asyncio
async def test_portfolio_accessor_falls_back_to_balance_when_nav_zero():
    """
    Early in a session (or in offline smoke), nothing has populated
    ``shared_state.nav_usdt``. The accessor must derive NAV from the
    USDT balance so the drawdown gate doesn't spuriously fire 100%.
    """
    from core_engine.native.balance_sync import NativeBalanceSync
    from core_engine.native.bootstrap import _make_portfolio_accessor
    from core_engine.native.shared_state import NativeSharedState

    state = NativeSharedState()
    # nav_usdt left at 0.0 (default)
    bs = NativeBalanceSync(_StubExchangeClient(), poll_interval_sec=99.0)
    bs._balances = {"USDT": 1000.0, "BTC": 0.0}  # simulate post-poll state

    accessor = _make_portfolio_accessor(state, bs)
    snap = accessor()
    assert snap.nav == 1000.0  # derived from USDT balance
    assert snap.nav_peak >= snap.nav  # peak tracked


@pytest.mark.asyncio
async def test_portfolio_accessor_prefers_shared_state_nav_when_set():
    from core_engine.native.balance_sync import NativeBalanceSync
    from core_engine.native.bootstrap import _make_portfolio_accessor
    from core_engine.native.shared_state import NativeSharedState

    state = NativeSharedState()
    state.nav_usdt = 5_000.0
    bs = NativeBalanceSync(_StubExchangeClient(), poll_interval_sec=99.0)
    bs._balances = {"USDT": 1_000.0}  # would derive 1000 if fallback fired

    accessor = _make_portfolio_accessor(state, bs)
    snap = accessor()
    assert snap.nav == 5_000.0  # canonical nav_usdt wins


@pytest.mark.asyncio
async def test_portfolio_accessor_tracks_peak_monotonically():
    from core_engine.native.balance_sync import NativeBalanceSync
    from core_engine.native.bootstrap import _make_portfolio_accessor
    from core_engine.native.shared_state import NativeSharedState

    state = NativeSharedState()
    bs = NativeBalanceSync(_StubExchangeClient(), poll_interval_sec=99.0)
    accessor = _make_portfolio_accessor(state, bs)

    state.nav_usdt = 100.0
    s1 = accessor()
    state.nav_usdt = 200.0
    s2 = accessor()
    state.nav_usdt = 50.0  # drawdown
    s3 = accessor()

    assert s1.nav_peak == 100.0
    assert s2.nav_peak == 200.0
    assert s3.nav_peak == 200.0  # peak does not retreat


@pytest.mark.asyncio
async def test_portfolio_accessor_extracts_position_qty_from_position_objects():
    from core_engine.native.balance_sync import NativeBalanceSync
    from core_engine.native.bootstrap import _make_portfolio_accessor
    from core_engine.native.shared_state import NativeSharedState, Position

    state = NativeSharedState()
    state.positions = {
        "BTCUSDT": Position(symbol="BTCUSDT", qty=0.5, entry_price=50_000.0, mark_price=51_000.0),
    }
    bs = NativeBalanceSync(_StubExchangeClient(), poll_interval_sec=99.0)
    accessor = _make_portfolio_accessor(state, bs)

    snap = accessor()
    assert snap.positions == {"BTCUSDT": 0.5}


def test_bootstrap_config_parses_balance_min_refresh_interval() -> None:
    env = {
        "BINANCE_API_KEY": "k",
        "BINANCE_API_SECRET": "s",
        "BALANCE_MIN_REFRESH_INTERVAL_SEC": "120",
    }
    cfg = BootstrapConfig.from_env(env=env)
    assert cfg.balance_min_refresh_interval_sec == 120.0


# ---------------------------------------------------------------------
# telemetry exporter wiring (Phase 8.3.3)
# ---------------------------------------------------------------------
@pytest.mark.asyncio
async def test_telemetry_exporter_disabled_when_path_unset():
    """Default config has empty TELEMETRY_EXPORT_PATH → no exporter."""
    components = await build_components(_min_cfg(), exchange_client_factory=_stub_factory)
    assert components.telemetry_exporter is None
    await shutdown_components(components)


@pytest.mark.asyncio
async def test_telemetry_exporter_enabled_when_path_set(tmp_path):
    out = tmp_path / "telemetry.json"
    cfg = _min_cfg(
        telemetry_export_path=str(out),
        telemetry_export_interval_sec=0.05,
    )
    components = await build_components(cfg, exchange_client_factory=_stub_factory)
    try:
        assert components.telemetry_exporter is not None
        assert components.telemetry_exporter.output_path == out
        assert components.telemetry_exporter.interval_sec == 0.05
        # Exporter is already started by build_components.
        assert components.telemetry_exporter._task is not None
    finally:
        await shutdown_components(components)
    # File exists after shutdown (final snapshot is best-effort).
    assert out.exists()


@pytest.mark.asyncio
async def test_shutdown_components_stops_telemetry_exporter(tmp_path):
    out = tmp_path / "telemetry.json"
    cfg = _min_cfg(
        telemetry_export_path=str(out),
        telemetry_export_interval_sec=0.05,
    )
    components = await build_components(cfg, exchange_client_factory=_stub_factory)
    exporter = components.telemetry_exporter
    assert exporter is not None
    await shutdown_components(components)
    assert exporter._task is None  # cleared on stop
    # Idempotent — second shutdown must not raise.
    await shutdown_components(components)


def test_from_env_parses_telemetry_export_settings(monkeypatch):
    monkeypatch.setenv("BINANCE_API_KEY", "k")
    monkeypatch.setenv("BINANCE_API_SECRET", "s")
    monkeypatch.setenv("TELEMETRY_EXPORT_PATH", "/tmp/octi_telemetry.json")
    monkeypatch.setenv("TELEMETRY_EXPORT_INTERVAL_SEC", "30")
    cfg = BootstrapConfig.from_env()
    assert cfg.telemetry_export_path == "/tmp/octi_telemetry.json"
    assert cfg.telemetry_export_interval_sec == 30.0


def test_from_env_telemetry_export_defaults():
    cfg = BootstrapConfig.from_env({"BINANCE_API_KEY": "k", "BINANCE_API_SECRET": "s"})
    assert cfg.telemetry_export_path == ""
    assert cfg.telemetry_export_interval_sec == 10.0
