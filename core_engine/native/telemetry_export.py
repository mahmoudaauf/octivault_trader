"""
Native L6: Telemetry exporter (Phase 8.3.3)

Periodic JSON snapshot writer for ``NativeTelemetry``. Runs as a
background ``asyncio.Task`` that wakes every ``interval_sec``, builds
``{summary, phase_breakdown, latest, ts}``, and atomically writes it to
``output_path``.

Design choices
--------------
* Pure stdlib (json, asyncio, pathlib, tempfile). No prometheus / OTel.
* Atomic write: ``tempfile`` + ``os.replace`` (POSIX atomic on same fs).
* Bounded work: snapshot is a single ring-buffer scan, dominated by
  ``len(buf)`` (≤ telemetry_capacity).
* Idempotent shutdown: ``stop()`` may be called any number of times
  before or after ``start()`` and is safe across cancellation races.
* No swallowed errors *silently*: write failures log a warning, the
  loop continues so a transient disk full doesn't kill the bot.

Usage::

    exporter = NativeTelemetryExporter(
        telemetry=components.telemetry,
        output_path=Path("runs/native/telemetry.json"),
        interval_sec=10.0,
    )
    await exporter.start()
    ...
    await exporter.stop()
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from .observability import NativeTelemetry

logger = logging.getLogger(__name__)


class NativeTelemetryExporter:
    """Periodic JSON snapshot writer for ``NativeTelemetry``."""

    def __init__(
        self,
        telemetry: NativeTelemetry,
        output_path: Path,
        interval_sec: float = 10.0,
    ) -> None:
        if interval_sec <= 0:
            raise ValueError(f"interval_sec must be > 0, got {interval_sec}")
        self._telemetry = telemetry
        self._output_path = Path(output_path)
        self._interval_sec = float(interval_sec)
        self._task: asyncio.Task[None] | None = None
        self._stopping: asyncio.Event | None = None  # Lazy initialization
        self._write_count = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def start(self) -> None:
        """Spawn the background snapshot loop. Idempotent."""
        if self._task is not None and not self._task.done():
            return
        # Lazy-create the event when we're in async context
        if self._stopping is None:
            self._stopping = asyncio.Event()
        self._stopping.clear()
        # Ensure parent directory exists up-front so first export succeeds.
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._task = asyncio.create_task(self._run_loop(), name="native-telemetry-exporter")
        logger.info(
            "telemetry exporter started: path=%s interval=%.1fs",
            self._output_path,
            self._interval_sec,
        )

    async def stop(self) -> None:
        """Signal the loop to exit and await its completion. Idempotent."""
        # Ensure event is created before setting it
        if self._stopping is None:
            self._stopping = asyncio.Event()
        self._stopping.set()
        task = self._task
        if task is None:
            return
        if not task.done():
            try:
                # Wake the loop early instead of waiting for the next tick.
                await asyncio.wait_for(task, timeout=self._interval_sec + 1.0)
            except asyncio.TimeoutError:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        self._task = None
        # Best-effort final snapshot so callers see the closing state.
        try:
            self._write_snapshot()
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("telemetry exporter: final snapshot failed: %r", e)
        logger.info(
            "telemetry exporter stopped: writes=%d path=%s",
            self._write_count,
            self._output_path,
        )

    # ------------------------------------------------------------------
    # Introspection (test surface)
    # ------------------------------------------------------------------
    @property
    def write_count(self) -> int:
        return self._write_count

    @property
    def output_path(self) -> Path:
        return self._output_path

    @property
    def interval_sec(self) -> float:
        return self._interval_sec

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    async def _run_loop(self) -> None:
        """Wake every ``interval_sec`` and write a snapshot until stopped."""
        while not self._stopping.is_set():
            try:
                self._write_snapshot()
            except Exception as e:
                logger.warning(
                    "telemetry exporter: write to %s failed: %r",
                    self._output_path,
                    e,
                )
            try:
                await asyncio.wait_for(self._stopping.wait(), timeout=self._interval_sec)
            except asyncio.TimeoutError:
                continue  # normal interval roll-over

    def _build_payload(self) -> dict[str, Any]:
        latest = self._telemetry.latest()
        latest_dict: dict[str, Any] | None
        if latest is None:
            latest_dict = None
        else:
            latest_dict = {
                "cycle_num": latest.cycle_num,
                "duration_ms": latest.duration_ms,
                "nav": latest.nav,
                "signals_count": latest.signals_count,
                "decisions_count": latest.decisions_count,
                "executions_count": latest.executions_count,
                "execution_successes": latest.execution_successes,
                "execution_failures": latest.execution_failures,
                "phase_times": dict(latest.phase_times),
                "errors": list(latest.errors),
                "ts": latest.ts,
            }
        return {
            "ts": time.time(),
            "buffer_size": len(self._telemetry),
            "buffer_capacity": self._telemetry.capacity,
            "summary": self._telemetry.summary(),
            "phase_breakdown": self._telemetry.phase_breakdown(),
            "latest": latest_dict,
        }

    def _write_snapshot(self) -> None:
        """Atomically write the current snapshot to ``output_path``."""
        payload = self._build_payload()
        # Atomic write: temp file in the same directory, then os.replace.
        target = self._output_path
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            prefix=target.name + ".",
            suffix=".tmp",
            dir=str(target.parent),
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(payload, f, indent=2, sort_keys=True, default=str)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, target)
        except Exception:
            # Cleanup the temp file on failure; never leak partial files.
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
        self._write_count += 1


__all__ = ["NativeTelemetryExporter"]
