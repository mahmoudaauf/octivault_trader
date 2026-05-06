"""
Native L6: Observability (Phase 8.2.7)

Lightweight, dependency-free telemetry for the native trading cycle.
Records ``CycleMetrics`` produced by ``NativeOrchestrator`` and exposes
rolling-window aggregates for dashboards / health checks.

Design choices
--------------
* Pure stdlib. No prometheus / OpenTelemetry coupling.
* Bounded memory: ring buffer with configurable capacity (default 1024).
* Read-only aggregates; never mutates the recorded metrics.
* Optional structured-log adapter (``log_cycle``).
* Safe for hot path: O(1) record, O(N) summary (N = window size).

Usage::

    telemetry = NativeTelemetry(capacity=2048)
    orch = NativeOrchestrator(..., telemetry=telemetry)
    await orch.run_loop(max_cycles=100)
    snap = telemetry.summary()
    # {'count': 100, 'avg_duration_ms': 187.3, 'p95_duration_ms': 240.1, ...}
"""

from __future__ import annotations

import logging
from collections import deque
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .orchestrator import CycleMetrics


class NativeTelemetry:
    """Bounded ring-buffer telemetry aggregator for ``CycleMetrics``."""

    def __init__(self, capacity: int = 1024) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {capacity}")
        self._capacity = capacity
        self._buf: deque[CycleMetrics] = deque(maxlen=capacity)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------
    def record(self, metrics: CycleMetrics) -> None:
        """Append a cycle's metrics to the ring buffer (O(1))."""
        self._buf.append(metrics)

    def clear(self) -> None:
        """Drop all recorded metrics."""
        self._buf.clear()

    # ------------------------------------------------------------------
    # Read-only access
    # ------------------------------------------------------------------
    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        return len(self._buf)

    def latest(self) -> CycleMetrics | None:
        """Return the most-recently recorded cycle, or None if empty."""
        if not self._buf:
            return None
        return self._buf[-1]

    def history(self) -> list[CycleMetrics]:
        """Return a shallow copy of the buffer (oldest -> newest)."""
        return list(self._buf)

    # ------------------------------------------------------------------
    # Aggregates
    # ------------------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        """
        Rolling aggregates over the current buffer window.

        Empty buffer returns a zero-filled summary so callers can rely on
        a stable schema.
        """
        n = len(self._buf)
        if n == 0:
            return {
                "count": 0,
                "avg_duration_ms": 0.0,
                "p50_duration_ms": 0.0,
                "p95_duration_ms": 0.0,
                "max_duration_ms": 0.0,
                "total_signals": 0,
                "total_decisions": 0,
                "total_executions": 0,
                "total_successes": 0,
                "total_failures": 0,
                "total_errors": 0,
                "error_rate": 0.0,
                "success_rate": 0.0,
                "latest_nav": 0.0,
            }

        durations = sorted(m.duration_ms for m in self._buf)
        total_sig = sum(m.signals_count for m in self._buf)
        total_dec = sum(m.decisions_count for m in self._buf)
        total_exe = sum(m.executions_count for m in self._buf)
        total_succ = sum(m.execution_successes for m in self._buf)
        total_fail = sum(m.execution_failures for m in self._buf)
        total_err = sum(len(m.errors) for m in self._buf)

        return {
            "count": n,
            "avg_duration_ms": sum(durations) / n,
            "p50_duration_ms": _percentile(durations, 0.50),
            "p95_duration_ms": _percentile(durations, 0.95),
            "max_duration_ms": durations[-1],
            "total_signals": total_sig,
            "total_decisions": total_dec,
            "total_executions": total_exe,
            "total_successes": total_succ,
            "total_failures": total_fail,
            "total_errors": total_err,
            "error_rate": total_err / n,
            "success_rate": (total_succ / total_exe) if total_exe else 0.0,
            "latest_nav": self._buf[-1].nav,
        }

    def phase_breakdown(self) -> dict[str, float]:
        """
        Average milliseconds spent in each phase across the buffer.

        Returns empty dict if buffer is empty.
        """
        n = len(self._buf)
        if n == 0:
            return {}
        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        for m in self._buf:
            for phase, ms in m.phase_times.items():
                totals[phase] = totals.get(phase, 0.0) + ms
                counts[phase] = counts.get(phase, 0) + 1
        return {phase: totals[phase] / counts[phase] for phase in totals}

    # ------------------------------------------------------------------
    # Structured logging adapter
    # ------------------------------------------------------------------
    def log_cycle(
        self,
        metrics: CycleMetrics,
        logger: logging.Logger,
        level: int = logging.INFO,
    ) -> None:
        """Emit a single structured log line for the given cycle."""
        logger.log(
            level,
            (
                "cycle=%05d duration_ms=%.1f nav=%.2f "
                "signals=%d decisions=%d executions=%d "
                "ok=%d fail=%d errors=%d"
            ),
            metrics.cycle_num,
            metrics.duration_ms,
            metrics.nav,
            metrics.signals_count,
            metrics.decisions_count,
            metrics.executions_count,
            metrics.execution_successes,
            metrics.execution_failures,
            len(metrics.errors),
        )


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _percentile(sorted_values: list[float], q: float) -> float:
    """
    Linear-interpolation percentile on a pre-sorted list.

    q in [0.0, 1.0]. Returns 0.0 for empty input (defensive; callers guard).
    """
    if not sorted_values:
        return 0.0
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0,1], got {q}")
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    pos = q * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac
