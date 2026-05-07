"""
Lightweight cadence scheduler for multi-speed orchestration.

Keeps the existing loop shape but allows different engine phases to run at
different cadences without creating a second orchestrator.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(frozen=True)
class CadenceConfig:
    fast_sec: float = 2.0
    account_sec: float = 30.0
    decision_sec: float = 30.0
    scenario_sec: float = 60.0
    rebalance_sec: float = 600.0
    dust_cleanup_sec: float = 900.0
    performance_sec: float = 3600.0
    health_sec: float = 10.0
    summary_sec: float = 60.0


class CadenceScheduler:
    """Tracks when a named cadence lane is due."""

    def __init__(self, config: CadenceConfig | None = None) -> None:
        self.config = config or CadenceConfig()
        self._last_run: dict[str, float] = {}

    def is_due(self, lane: str, *, now: float | None = None) -> bool:
        now = float(now if now is not None else time.time())
        interval = self._interval_for(lane)
        last = self._last_run.get(lane)
        if last is None:
            return True
        return (now - last) >= interval

    def mark(self, lane: str, *, now: float | None = None) -> None:
        self._last_run[lane] = float(now if now is not None else time.time())

    def _interval_for(self, lane: str) -> float:
        mapping = {
            "fast": self.config.fast_sec,
            "account": self.config.account_sec,
            "decision": self.config.decision_sec,
            "scenario": self.config.scenario_sec,
            "rebalance": self.config.rebalance_sec,
            "dust_cleanup": self.config.dust_cleanup_sec,
            "performance": self.config.performance_sec,
            "health": self.config.health_sec,
            "summary": self.config.summary_sec,
        }
        return float(mapping.get(lane, self.config.fast_sec))


__all__ = ["CadenceConfig", "CadenceScheduler"]
