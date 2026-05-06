"""
Tests for Native L6 (Phase 8.2.7) - NativeTelemetry.

Covers:
* ring-buffer recording (bounded capacity, eviction order)
* read-only accessors (latest, history, len, capacity)
* aggregate summary schema (empty + populated)
* phase_breakdown averaging
* percentile correctness (p50, p95, edge cases)
* structured log_cycle adapter
* orchestrator integration: telemetry receives a CycleMetrics per cycle
"""

from __future__ import annotations

import logging

import pytest

from core_engine.native.observability import NativeTelemetry, _percentile
from core_engine.native.orchestrator import CycleMetrics, NativeOrchestrator


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def _mk_metrics(
    n: int,
    *,
    duration_ms: float = 100.0,
    nav: float = 1000.0,
    signals: int = 1,
    decisions: int = 1,
    executions: int = 1,
    successes: int = 1,
    failures: int = 0,
    errors: list[str] | None = None,
    phase_times: dict[str, float] | None = None,
) -> CycleMetrics:
    return CycleMetrics(
        cycle_num=n,
        duration_ms=duration_ms,
        nav=nav,
        signals_count=signals,
        decisions_count=decisions,
        executions_count=executions,
        execution_successes=successes,
        execution_failures=failures,
        phase_times=phase_times or {"read": 10.0, "execute": 50.0},
        errors=errors or [],
    )


# ---------------------------------------------------------------------
# constructor / ring buffer
# ---------------------------------------------------------------------
def test_capacity_must_be_positive():
    with pytest.raises(ValueError):
        NativeTelemetry(capacity=0)
    with pytest.raises(ValueError):
        NativeTelemetry(capacity=-1)


def test_capacity_property_and_initial_state():
    t = NativeTelemetry(capacity=128)
    assert t.capacity == 128
    assert len(t) == 0
    assert t.latest() is None
    assert t.history() == []


def test_record_and_latest():
    t = NativeTelemetry(capacity=4)
    m1 = _mk_metrics(1)
    m2 = _mk_metrics(2)
    t.record(m1)
    t.record(m2)
    assert len(t) == 2
    assert t.latest() is m2
    hist = t.history()
    assert hist == [m1, m2]
    # history() returns a copy
    hist.append("sentinel")  # type: ignore[arg-type]
    assert len(t) == 2


def test_ring_buffer_evicts_oldest_when_full():
    t = NativeTelemetry(capacity=3)
    for i in range(5):
        t.record(_mk_metrics(i))
    assert len(t) == 3
    nums = [m.cycle_num for m in t.history()]
    assert nums == [2, 3, 4]


def test_clear_resets_buffer():
    t = NativeTelemetry(capacity=4)
    t.record(_mk_metrics(1))
    t.record(_mk_metrics(2))
    t.clear()
    assert len(t) == 0
    assert t.latest() is None


# ---------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------
def test_summary_empty_buffer_has_stable_schema():
    t = NativeTelemetry()
    s = t.summary()
    expected_keys = {
        "count",
        "avg_duration_ms",
        "p50_duration_ms",
        "p95_duration_ms",
        "max_duration_ms",
        "total_signals",
        "total_decisions",
        "total_executions",
        "total_successes",
        "total_failures",
        "total_errors",
        "error_rate",
        "success_rate",
        "latest_nav",
    }
    assert set(s.keys()) == expected_keys
    assert s["count"] == 0
    assert s["avg_duration_ms"] == 0.0
    assert s["error_rate"] == 0.0
    assert s["success_rate"] == 0.0


def test_summary_aggregates_correctly():
    t = NativeTelemetry()
    t.record(
        _mk_metrics(
            1,
            duration_ms=100.0,
            nav=1000.0,
            signals=2,
            decisions=1,
            executions=1,
            successes=1,
            failures=0,
        )
    )
    t.record(
        _mk_metrics(
            2,
            duration_ms=200.0,
            nav=1010.0,
            signals=3,
            decisions=2,
            executions=2,
            successes=1,
            failures=1,
            errors=["BoomError: x"],
        )
    )
    t.record(
        _mk_metrics(
            3,
            duration_ms=150.0,
            nav=1020.0,
            signals=1,
            decisions=1,
            executions=1,
            successes=1,
            failures=0,
        )
    )
    s = t.summary()
    assert s["count"] == 3
    assert s["avg_duration_ms"] == pytest.approx((100 + 200 + 150) / 3)
    assert s["max_duration_ms"] == 200.0
    assert s["total_signals"] == 6
    assert s["total_decisions"] == 4
    assert s["total_executions"] == 4
    assert s["total_successes"] == 3
    assert s["total_failures"] == 1
    assert s["total_errors"] == 1
    assert s["error_rate"] == pytest.approx(1 / 3)
    assert s["success_rate"] == pytest.approx(3 / 4)
    assert s["latest_nav"] == 1020.0


def test_summary_success_rate_zero_when_no_executions():
    t = NativeTelemetry()
    t.record(_mk_metrics(1, executions=0, successes=0, failures=0))
    s = t.summary()
    assert s["success_rate"] == 0.0
    assert s["total_executions"] == 0


# ---------------------------------------------------------------------
# percentiles
# ---------------------------------------------------------------------
def test_percentile_basic():
    vals = sorted([10.0, 20.0, 30.0, 40.0, 50.0])
    assert _percentile(vals, 0.0) == 10.0
    assert _percentile(vals, 1.0) == 50.0
    assert _percentile(vals, 0.5) == pytest.approx(30.0)


def test_percentile_single_element():
    assert _percentile([42.0], 0.95) == 42.0


def test_percentile_empty_returns_zero():
    assert _percentile([], 0.5) == 0.0


def test_percentile_rejects_invalid_q():
    with pytest.raises(ValueError):
        _percentile([1.0, 2.0], 1.5)
    with pytest.raises(ValueError):
        _percentile([1.0, 2.0], -0.1)


def test_summary_p95_matches_distribution():
    t = NativeTelemetry()
    for i in range(1, 101):
        t.record(_mk_metrics(i, duration_ms=float(i)))
    s = t.summary()
    # p95 of 1..100 (linear interp) = 1 + 0.95*99 = 95.05
    assert s["p95_duration_ms"] == pytest.approx(95.05)
    assert s["p50_duration_ms"] == pytest.approx(50.5)


# ---------------------------------------------------------------------
# phase breakdown
# ---------------------------------------------------------------------
def test_phase_breakdown_empty():
    assert NativeTelemetry().phase_breakdown() == {}


def test_phase_breakdown_averages_per_phase():
    t = NativeTelemetry()
    t.record(_mk_metrics(1, phase_times={"read": 10.0, "execute": 50.0}))
    t.record(_mk_metrics(2, phase_times={"read": 30.0, "execute": 70.0}))
    t.record(_mk_metrics(3, phase_times={"read": 20.0}))  # no execute this cycle
    pb = t.phase_breakdown()
    assert pb["read"] == pytest.approx((10 + 30 + 20) / 3)
    assert pb["execute"] == pytest.approx((50 + 70) / 2)


# ---------------------------------------------------------------------
# logging adapter
# ---------------------------------------------------------------------
def test_log_cycle_emits_record(caplog):
    t = NativeTelemetry()
    log = logging.getLogger("native.l6.test")
    m = _mk_metrics(
        7, duration_ms=123.4, nav=999.5, signals=2, decisions=2, executions=1, successes=1
    )
    with caplog.at_level(logging.INFO, logger="native.l6.test"):
        t.log_cycle(m, log)
    assert any("cycle=00007" in rec.getMessage() for rec in caplog.records)


# ---------------------------------------------------------------------
# orchestrator integration
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
        self.nav = 1000.0


def _make_orch(telemetry=None) -> NativeOrchestrator:
    return NativeOrchestrator(
        market_data=_MD(),
        signal_engine=_Sig(),
        decision_engine=_Dec(),
        executor=_Exe(),
        balance_sync=_Bal(),
        shared_state=_State(),
        telemetry=telemetry,
    )


@pytest.mark.asyncio
async def test_orchestrator_records_into_telemetry_when_provided():
    t = NativeTelemetry(capacity=16)
    orch = _make_orch(telemetry=t)
    metrics = await orch.run_cycle()
    assert len(t) == 1
    assert t.latest() is metrics
    summary = t.summary()
    assert summary["count"] == 1


@pytest.mark.asyncio
async def test_orchestrator_works_without_telemetry():
    orch = _make_orch(telemetry=None)
    m = await orch.run_cycle()
    assert m.cycle_num == 1


@pytest.mark.asyncio
async def test_orchestrator_run_loop_records_each_cycle():
    t = NativeTelemetry(capacity=16)
    orch = _make_orch(telemetry=t)
    metrics = await orch.run_loop(max_cycles=5)
    assert len(metrics) == 5
    assert len(t) == 5
    nums = [m.cycle_num for m in t.history()]
    assert nums == [1, 2, 3, 4, 5]


@pytest.mark.asyncio
async def test_telemetry_record_failure_does_not_break_cycle():
    class _BadTelemetry:
        def record(self, _m):
            raise RuntimeError("telemetry down")

    orch = _make_orch(telemetry=_BadTelemetry())
    # Should not raise; cycle still completes and returns metrics.
    m = await orch.run_cycle()
    assert m.cycle_num == 1
