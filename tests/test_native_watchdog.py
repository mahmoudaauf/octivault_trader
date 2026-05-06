"""
Tests for ``core_engine.native.watchdog`` (Phase 8.3.12).

Coverage:
* Constructor input validation (5 guards).
* ``record_heartbeat`` updates timestamp and counter; ``ok=False``
  records error in sliding window without skipping the timestamp.
* ``check_liveness`` cold-start grace + post-heartbeat decay.
* ``detect_anomalies`` per-detector: stale heartbeat, stale market
  data, stale balance sync, missing exchange client, error rate.
* ``health()`` reports counters and configured thresholds.
* Wiring: appears in ``app_ctx``, isn't overwritten (no compat stub
  exists for it anymore — COMPAT_KEYS is empty after 8.3.12).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import pytest

from core_engine.native.watchdog import NativeWatchdog


# ---------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------
@dataclass
class _State:
    price_timestamps: dict[str, float] = field(default_factory=dict)
    last_md_update_ts: float = 0.0


@dataclass
class _BalanceStub:
    last_sync_ts: float = 0.0


@dataclass
class _MarketDataStub:
    last_poll_ts: float = 0.0


# ---------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------
def test_constructor_rejects_zero_liveness_timeout():
    with pytest.raises(ValueError, match="liveness_timeout_sec"):
        NativeWatchdog(_State(), liveness_timeout_sec=0)


def test_constructor_rejects_negative_md_timeout():
    with pytest.raises(ValueError, match="market_data_timeout_sec"):
        NativeWatchdog(_State(), market_data_timeout_sec=-1)


def test_constructor_rejects_negative_balance_timeout():
    with pytest.raises(ValueError, match="balance_sync_timeout_sec"):
        NativeWatchdog(_State(), balance_sync_timeout_sec=-1)


def test_constructor_rejects_error_rate_out_of_range():
    with pytest.raises(ValueError, match="error_rate_threshold"):
        NativeWatchdog(_State(), error_rate_threshold=1.5)


def test_constructor_rejects_zero_window_size():
    with pytest.raises(ValueError, match="error_window_size"):
        NativeWatchdog(_State(), error_window_size=0)


# ---------------------------------------------------------------------
# Heartbeat
# ---------------------------------------------------------------------
def test_record_heartbeat_updates_counter_and_timestamp():
    wd = NativeWatchdog(_State())
    assert wd.health()["heartbeats_recorded"] == 0
    wd.record_heartbeat()
    h = wd.health()
    assert h["heartbeats_recorded"] == 1
    assert h["last_heartbeat_age_sec"] is not None
    assert h["last_heartbeat_age_sec"] >= 0


def test_record_heartbeat_failed_cycle_still_updates_timestamp():
    wd = NativeWatchdog(_State())
    wd.record_heartbeat(ok=False)
    h = wd.health()
    assert h["heartbeats_recorded"] == 1
    assert h["last_heartbeat_age_sec"] is not None


# ---------------------------------------------------------------------
# check_liveness
# ---------------------------------------------------------------------
@pytest.mark.asyncio
async def test_check_liveness_cold_start_grace_returns_true():
    wd = NativeWatchdog(_State(), liveness_timeout_sec=1.0)
    assert await wd.check_liveness() is True
    assert wd.health()["liveness_checks"] == 1


@pytest.mark.asyncio
async def test_check_liveness_returns_true_when_recent_heartbeat():
    wd = NativeWatchdog(_State(), liveness_timeout_sec=10.0)
    wd.record_heartbeat()
    assert await wd.check_liveness() is True


@pytest.mark.asyncio
async def test_check_liveness_returns_false_when_heartbeat_stale():
    wd = NativeWatchdog(_State(), liveness_timeout_sec=0.05)
    wd.record_heartbeat()
    time.sleep(0.1)
    assert await wd.check_liveness() is False


# ---------------------------------------------------------------------
# detect_anomalies — individual detectors
# ---------------------------------------------------------------------
@pytest.mark.asyncio
async def test_detect_anomalies_clean_slate_returns_empty():
    state = _State(last_md_update_ts=time.time())
    wd = NativeWatchdog(
        state,
        balance_sync=_BalanceStub(last_sync_ts=time.time()),
        exchange_client=object(),
    )
    wd.record_heartbeat()
    assert await wd.detect_anomalies() == []


@pytest.mark.asyncio
async def test_detect_anomalies_flags_stale_heartbeat():
    wd = NativeWatchdog(_State(), liveness_timeout_sec=0.05, exchange_client=object())
    wd.record_heartbeat()
    time.sleep(0.1)
    out = await wd.detect_anomalies()
    assert any("heartbeat stale" in s for s in out)


@pytest.mark.asyncio
async def test_detect_anomalies_flags_stale_market_data():
    state = _State(price_timestamps={"BTCUSDT": time.time() - 120})
    wd = NativeWatchdog(state, market_data_timeout_sec=60.0, exchange_client=object())
    out = await wd.detect_anomalies()
    assert any("market data stale" in s for s in out)


@pytest.mark.asyncio
async def test_detect_anomalies_uses_top_level_md_field_when_no_per_symbol():
    state = _State(last_md_update_ts=time.time() - 120)
    wd = NativeWatchdog(state, market_data_timeout_sec=60.0, exchange_client=object())
    out = await wd.detect_anomalies()
    assert any("market data stale" in s for s in out)


@pytest.mark.asyncio
async def test_detect_anomalies_silent_when_no_md_tracking():
    """No price_timestamps + no last_md_update_ts → no md anomaly emitted."""
    wd = NativeWatchdog(_State(), exchange_client=object())
    out = await wd.detect_anomalies()
    assert not any("market data" in s for s in out)


@pytest.mark.asyncio
async def test_detect_anomalies_flags_stale_balance_sync():
    bal = _BalanceStub(last_sync_ts=time.time() - 120)
    wd = NativeWatchdog(
        _State(),
        balance_sync=bal,
        balance_sync_timeout_sec=60.0,
        exchange_client=object(),
    )
    out = await wd.detect_anomalies()
    assert any("balance sync stale" in s for s in out)


@pytest.mark.asyncio
async def test_detect_anomalies_flags_missing_exchange_client():
    wd = NativeWatchdog(_State())  # exchange_client=None default
    out = await wd.detect_anomalies()
    assert any("exchange_client unavailable" in s for s in out)


@pytest.mark.asyncio
async def test_detect_anomalies_flags_high_cycle_error_rate():
    wd = NativeWatchdog(
        _State(),
        error_rate_threshold=0.4,
        error_window_size=10,
        exchange_client=object(),
    )
    # 6 errors out of 10 = 60% > 40% threshold
    for _ in range(4):
        wd.record_heartbeat(ok=True)
    for _ in range(6):
        wd.record_heartbeat(ok=False)
    out = await wd.detect_anomalies()
    assert any("cycle error rate" in s for s in out)


@pytest.mark.asyncio
async def test_detect_anomalies_silent_when_window_too_small():
    """Need at least max(3, maxlen//2) cycles before evaluating rate."""
    wd = NativeWatchdog(
        _State(),
        error_rate_threshold=0.1,
        error_window_size=10,
        exchange_client=object(),
    )
    wd.record_heartbeat(ok=False)  # 1 cycle, 100% error — but below window threshold
    wd.record_heartbeat(ok=False)  # 2 cycles
    out = await wd.detect_anomalies()
    assert not any("cycle error rate" in s for s in out)


@pytest.mark.asyncio
async def test_detect_anomalies_counts_aggregate():
    wd = NativeWatchdog(_State(), liveness_timeout_sec=0.05)  # exchange_client=None
    wd.record_heartbeat()
    time.sleep(0.1)
    out = await wd.detect_anomalies()
    assert len(out) >= 2  # heartbeat stale + missing exchange_client
    assert wd.health()["anomalies_detected"] == len(out)
    assert wd.health()["anomaly_sweeps"] == 1


# ---------------------------------------------------------------------
# Wiring tests
# ---------------------------------------------------------------------
def test_native_components_carries_watchdog_field():
    from core_engine.native.app_context import NativeComponents

    fields = {f.name for f in NativeComponents.__dataclass_fields__.values()}
    assert "watchdog" in fields


def test_watchdog_visible_in_app_ctx_when_provided():
    from core_engine.native.app_context import NativeComponents, build_native_app_ctx
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
    wd = NativeWatchdog(state)
    components = NativeComponents(
        shared_state=state,  # type: ignore[arg-type]
        market_data=_MD(),  # type: ignore[arg-type]
        signal_engine=_Sig(),  # type: ignore[arg-type]
        decision_engine=_Dec(),  # type: ignore[arg-type]
        executor=_Exe(),  # type: ignore[arg-type]
        balance_sync=_Bal(),  # type: ignore[arg-type]
        telemetry=NativeTelemetry(),
        watchdog=wd,
    )
    app_ctx, _ = build_native_app_ctx(components)
    assert app_ctx["watchdog"] is wd


def test_compat_keys_empty_after_8_3_12():
    """Final acceptance: COMPAT_KEYS is empty; compat module is a no-op."""
    from core_engine.native.compat import COMPAT_KEYS, make_compat_stubs

    assert COMPAT_KEYS == ()
    assert make_compat_stubs() == {}
