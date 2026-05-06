"""
Native L7: Watchdog (Phase 8.3.12).

Liveness + anomaly detection for the running native stack. Final
native implementation behind the ``watchdog`` app_ctx key consumed
by ``OperationsEngine.check_liveness`` /
``OperationsEngine.detect_anomalies``.

This was the **last** compat-stub replacement; ``core_engine.native.compat``
was subsequently retired (G5 acceptance gate from ``PHASE_8_3_PLAN.md``).

Responsibility
--------------
Two read-only async methods serving the OperationsEngine surface:

* ``check_liveness()`` — fast (sub-millisecond) boolean predicate.
  Returns True iff the orchestrator has emitted a heartbeat within the
  configured ``liveness_timeout_sec`` window.
* ``detect_anomalies()`` — slower diagnostic sweep returning a list of
  human-readable anomaly strings. Empty list means healthy.

The orchestrator is expected to call ``record_heartbeat()`` once per
loop cycle. Wiring is *optional*: a watchdog with zero heartbeats is
considered alive (cold-start grace) so the very first cycle never
trips the liveness gate.

Anomaly checks
--------------
1. **Stale orchestrator heartbeat** — last heartbeat older than
   ``liveness_timeout_sec``.
2. **Stale market data** — ``shared_state.last_md_update_ts`` (or
   derived from ``price_timestamps``) older than
   ``market_data_timeout_sec``.
3. **Stale balance sync** — ``balance_sync.last_sync_ts`` older than
   ``balance_sync_timeout_sec``.
4. **Exchange client missing** — ``exchange_client`` was None at
   construction (degraded mode).
5. **Cycle error rate** — ``cycle_errors / cycles_observed`` exceeds
   ``error_rate_threshold`` over the last sliding window.

Out of scope
------------
* Process-level kill / restart — owned by ``auto_recovery.py``.
* Pager / alerting — owned by L7 ``alert_system``.
* Recovery actions — owned by ``NativeRecoveryEngine`` (8.3.11).
  This module only *observes*; the recovery engine *acts*.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .balance_sync import NativeBalanceSync
    from .market_data import NativeMarketData
    from .shared_state import NativeSharedState

logger = logging.getLogger(__name__)


class NativeWatchdog:
    """In-process liveness + anomaly detector."""

    def __init__(
        self,
        shared_state: NativeSharedState,
        *,
        balance_sync: NativeBalanceSync | None = None,
        market_data: NativeMarketData | None = None,
        exchange_client: Any | None = None,
        liveness_timeout_sec: float = 30.0,
        market_data_timeout_sec: float = 60.0,
        balance_sync_timeout_sec: float = 60.0,
        error_rate_threshold: float = 0.5,
        error_window_size: int = 20,
    ) -> None:
        if liveness_timeout_sec <= 0:
            raise ValueError(f"liveness_timeout_sec must be > 0, got {liveness_timeout_sec}")
        if market_data_timeout_sec <= 0:
            raise ValueError(f"market_data_timeout_sec must be > 0, got {market_data_timeout_sec}")
        if balance_sync_timeout_sec <= 0:
            raise ValueError(
                f"balance_sync_timeout_sec must be > 0, got {balance_sync_timeout_sec}"
            )
        if not 0.0 <= error_rate_threshold <= 1.0:
            raise ValueError(f"error_rate_threshold must be in [0, 1], got {error_rate_threshold}")
        if error_window_size <= 0:
            raise ValueError(f"error_window_size must be > 0, got {error_window_size}")

        self._state = shared_state
        self._balance = balance_sync
        self._md = market_data
        self._exchange = exchange_client

        self._liveness_to = float(liveness_timeout_sec)
        self._md_to = float(market_data_timeout_sec)
        self._bal_to = float(balance_sync_timeout_sec)
        self._err_threshold = float(error_rate_threshold)

        # Heartbeat tracking
        self._last_heartbeat_ts: float = 0.0
        self._heartbeats_recorded: int = 0

        # Sliding window of cycle outcomes (True=ok, False=error)
        self._cycle_log: deque[bool] = deque(maxlen=int(error_window_size))

        # Health counters
        self._liveness_checks: int = 0
        self._anomaly_sweeps: int = 0
        self._anomalies_detected: int = 0

    # ------------------------------------------------------------------
    # Heartbeat surface (called by orchestrator each cycle — optional)
    # ------------------------------------------------------------------
    def record_heartbeat(self, *, ok: bool = True) -> None:
        """
        Note that the main loop completed a cycle.

        ``ok=False`` records a failed cycle (counted toward the error
        rate). Heartbeat timestamp is updated regardless so a crashed
        cycle still postpones the stale-heartbeat alarm.
        """
        self._last_heartbeat_ts = time.time()
        self._heartbeats_recorded += 1
        self._cycle_log.append(bool(ok))

    # ------------------------------------------------------------------
    # OperationsEngine contract
    # ------------------------------------------------------------------
    async def check_liveness(self) -> bool:
        """
        Returns True iff the orchestrator is alive.

        * Cold-start grace: True if no heartbeats have ever been
          recorded (the orchestrator just hasn't wired us up yet).
        * Otherwise: True iff age of last heartbeat <
          ``liveness_timeout_sec``.
        """
        self._liveness_checks += 1
        if self._heartbeats_recorded == 0:
            return True  # cold start grace
        age = time.time() - self._last_heartbeat_ts
        return age < self._liveness_to

    async def detect_anomalies(self) -> list[str]:
        """
        Run a full diagnostic sweep, return a list of anomaly strings.

        Empty list → healthy. Each entry is human-readable and
        suitable for logging or alert-system ingestion.
        """
        self._anomaly_sweeps += 1
        out: list[str] = []
        now = time.time()

        # 1) Stale heartbeat
        if self._heartbeats_recorded > 0:
            age = now - self._last_heartbeat_ts
            if age >= self._liveness_to:
                out.append(
                    f"orchestrator heartbeat stale ({age:.1f}s >= " f"{self._liveness_to:.1f}s)"
                )

        # 2) Stale market data (look at newest timestamp across symbols)
        md_age = self._market_data_age(now)
        if md_age is not None and md_age >= self._md_to:
            out.append(
                f"market data stale (newest tick {md_age:.1f}s old " f">= {self._md_to:.1f}s)"
            )

        # 3) Stale balance sync
        bal_age = self._balance_sync_age(now)
        if bal_age is not None and bal_age >= self._bal_to:
            out.append(f"balance sync stale ({bal_age:.1f}s >= " f"{self._bal_to:.1f}s)")

        # 4) Exchange client missing
        if self._exchange is None:
            out.append("exchange_client unavailable (degraded mode)")

        # 5) Cycle error rate (only meaningful once window has data)
        if len(self._cycle_log) >= max(3, self._cycle_log.maxlen // 2):
            errors = sum(1 for ok in self._cycle_log if not ok)
            rate = errors / len(self._cycle_log)
            if rate > self._err_threshold:
                out.append(
                    f"cycle error rate {rate:.0%} over last "
                    f"{len(self._cycle_log)} cycles "
                    f"(threshold {self._err_threshold:.0%})"
                )

        if out:
            self._anomalies_detected += len(out)
        return out

    # ------------------------------------------------------------------
    # Health / observability
    # ------------------------------------------------------------------
    def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "heartbeats_recorded": self._heartbeats_recorded,
            "last_heartbeat_age_sec": (
                time.time() - self._last_heartbeat_ts if self._heartbeats_recorded > 0 else None
            ),
            "liveness_checks": self._liveness_checks,
            "anomaly_sweeps": self._anomaly_sweeps,
            "anomalies_detected": self._anomalies_detected,
            "cycle_window_size": len(self._cycle_log),
            "liveness_timeout_sec": self._liveness_to,
            "market_data_timeout_sec": self._md_to,
            "balance_sync_timeout_sec": self._bal_to,
            "error_rate_threshold": self._err_threshold,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _market_data_age(self, now: float) -> float | None:
        """
        Return age of the freshest market-data tick, or None if no
        tracking data is available.
        """
        # Prefer explicit per-symbol timestamps on shared_state.
        timestamps = getattr(self._state, "price_timestamps", None) or {}
        if timestamps:
            newest = max((float(t) for t in timestamps.values() if t), default=0.0)
            if newest > 0:
                return now - newest

        # Fallback: a top-level last-update field if the L0 state
        # exposes one (defensive — name varies historically).
        for attr in ("last_md_update_ts", "last_price_update_ts"):
            ts = float(getattr(self._state, attr, 0.0) or 0.0)
            if ts > 0:
                return now - ts

        # Final fallback: ask the market_data component directly.
        if self._md is not None:
            ts = float(getattr(self._md, "last_poll_ts", 0.0) or 0.0)
            if ts > 0:
                return now - ts

        return None

    def _balance_sync_age(self, now: float) -> float | None:
        """Return age of the last successful balance sync, or None."""
        if self._balance is None:
            return None
        for attr in ("last_sync_ts", "last_poll_ts", "last_update_ts"):
            ts = float(getattr(self._balance, attr, 0.0) or 0.0)
            if ts > 0:
                return now - ts
        return None


__all__ = ["NativeWatchdog"]
