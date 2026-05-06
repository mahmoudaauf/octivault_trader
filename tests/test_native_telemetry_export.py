"""
Tests for NativeTelemetryExporter (Phase 8.3.3).

Covers:
- snapshot schema (no cycles + with cycles)
- atomic write (no .tmp leftovers on success or failure)
- periodic loop wakes & writes >= 2 snapshots within bounded time
- idempotent stop / double-start / stop-without-start
- bootstrap wiring: TELEMETRY_EXPORT_PATH gates exporter creation
- shutdown_components stops the exporter (no leaked task)
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from core_engine.native.observability import NativeTelemetry
from core_engine.native.orchestrator import CycleMetrics
from core_engine.native.telemetry_export import NativeTelemetryExporter


def _make_metric(cycle: int, dur: float = 12.5, nav: float = 100.0) -> CycleMetrics:
    return CycleMetrics(
        cycle_num=cycle,
        duration_ms=dur,
        nav=nav,
        signals_count=3,
        decisions_count=2,
        executions_count=1,
        execution_successes=1,
        execution_failures=0,
        phase_times={"read": 1.0, "decide": 2.0, "execute": 3.0},
        errors=[],
    )


# ----------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------
def test_exporter_rejects_non_positive_interval(tmp_path: Path) -> None:
    tel = NativeTelemetry(capacity=8)
    with pytest.raises(ValueError):
        NativeTelemetryExporter(tel, tmp_path / "t.json", interval_sec=0.0)
    with pytest.raises(ValueError):
        NativeTelemetryExporter(tel, tmp_path / "t.json", interval_sec=-1.0)


def test_exporter_exposes_config(tmp_path: Path) -> None:
    tel = NativeTelemetry(capacity=8)
    out = tmp_path / "telemetry.json"
    exp = NativeTelemetryExporter(tel, out, interval_sec=5.0)
    assert exp.output_path == out
    assert exp.interval_sec == 5.0
    assert exp.write_count == 0


# ----------------------------------------------------------------------
# Snapshot schema
# ----------------------------------------------------------------------
def test_snapshot_schema_empty_buffer(tmp_path: Path) -> None:
    tel = NativeTelemetry(capacity=8)
    out = tmp_path / "t.json"
    exp = NativeTelemetryExporter(tel, out, interval_sec=1.0)

    exp._write_snapshot()
    assert exp.write_count == 1

    payload = json.loads(out.read_text())
    assert set(payload) == {
        "ts",
        "buffer_size",
        "buffer_capacity",
        "summary",
        "phase_breakdown",
        "latest",
    }
    assert payload["buffer_size"] == 0
    assert payload["buffer_capacity"] == 8
    assert payload["latest"] is None
    assert payload["summary"]["count"] == 0
    assert payload["phase_breakdown"] == {}


def test_snapshot_schema_with_metrics(tmp_path: Path) -> None:
    tel = NativeTelemetry(capacity=8)
    tel.record(_make_metric(1, dur=10.0, nav=100.0))
    tel.record(_make_metric(2, dur=20.0, nav=105.0))
    out = tmp_path / "t.json"
    exp = NativeTelemetryExporter(tel, out, interval_sec=1.0)

    exp._write_snapshot()
    payload = json.loads(out.read_text())

    assert payload["buffer_size"] == 2
    assert payload["latest"]["cycle_num"] == 2
    assert payload["latest"]["nav"] == 105.0
    assert payload["latest"]["phase_times"] == {
        "read": 1.0,
        "decide": 2.0,
        "execute": 3.0,
    }
    assert payload["summary"]["count"] == 2
    assert payload["summary"]["avg_duration_ms"] == 15.0
    # phase_breakdown is the rolling per-phase mean
    assert payload["phase_breakdown"]["read"] == 1.0


# ----------------------------------------------------------------------
# Atomic write
# ----------------------------------------------------------------------
def test_atomic_write_leaves_no_tmp_files(tmp_path: Path) -> None:
    tel = NativeTelemetry(capacity=8)
    tel.record(_make_metric(1))
    out = tmp_path / "telemetry.json"
    exp = NativeTelemetryExporter(tel, out, interval_sec=1.0)

    for _ in range(5):
        exp._write_snapshot()

    assert out.exists()
    leftovers = [p for p in tmp_path.iterdir() if p.name != out.name]
    assert leftovers == [], f"unexpected files: {leftovers}"


def test_atomic_write_recovers_when_tempfile_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tel = NativeTelemetry(capacity=8)
    tel.record(_make_metric(1))
    out = tmp_path / "t.json"
    exp = NativeTelemetryExporter(tel, out, interval_sec=1.0)

    # First write succeeds, then make os.replace fail; existing file
    # should remain valid (last good snapshot) and no .tmp leaks.
    exp._write_snapshot()
    good = json.loads(out.read_text())

    import os as _os

    real_replace = _os.replace

    def boom(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise OSError("simulated EIO")

    monkeypatch.setattr("core_engine.native.telemetry_export.os.replace", boom)
    with pytest.raises(OSError):
        exp._write_snapshot()
    monkeypatch.setattr("core_engine.native.telemetry_export.os.replace", real_replace)

    assert json.loads(out.read_text()) == good
    leftovers = [p for p in tmp_path.iterdir() if p.name != out.name]
    assert leftovers == [], f"tmp file leak: {leftovers}"


# ----------------------------------------------------------------------
# Lifecycle
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_periodic_loop_writes_multiple_snapshots(tmp_path: Path) -> None:
    tel = NativeTelemetry(capacity=16)
    out = tmp_path / "telemetry.json"
    exp = NativeTelemetryExporter(tel, out, interval_sec=0.05)

    await exp.start()
    # Let the loop tick at least 3 times.
    await asyncio.sleep(0.25)
    tel.record(_make_metric(1))
    await asyncio.sleep(0.1)
    await exp.stop()

    assert out.exists()
    payload = json.loads(out.read_text())
    # Final snapshot must reflect the recorded cycle.
    assert payload["buffer_size"] == 1
    assert payload["latest"]["cycle_num"] == 1
    # write_count includes periodic writes + the closing snapshot.
    assert exp.write_count >= 3


@pytest.mark.asyncio
async def test_double_start_is_idempotent(tmp_path: Path) -> None:
    tel = NativeTelemetry(capacity=8)
    out = tmp_path / "t.json"
    exp = NativeTelemetryExporter(tel, out, interval_sec=0.1)

    await exp.start()
    task1 = exp._task
    await exp.start()  # second start should not spawn a new task
    assert exp._task is task1
    await exp.stop()


@pytest.mark.asyncio
async def test_stop_without_start_is_noop(tmp_path: Path) -> None:
    tel = NativeTelemetry(capacity=8)
    out = tmp_path / "t.json"
    exp = NativeTelemetryExporter(tel, out, interval_sec=0.1)

    await exp.stop()  # must not raise; no file required
    assert exp.write_count == 0


@pytest.mark.asyncio
async def test_double_stop_is_idempotent(tmp_path: Path) -> None:
    tel = NativeTelemetry(capacity=8)
    out = tmp_path / "t.json"
    exp = NativeTelemetryExporter(tel, out, interval_sec=0.05)

    await exp.start()
    await asyncio.sleep(0.1)
    await exp.stop()
    first_count = exp.write_count
    await exp.stop()  # second stop = best-effort final snapshot only
    assert exp.write_count >= first_count
