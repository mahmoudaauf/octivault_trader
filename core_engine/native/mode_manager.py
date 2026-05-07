"""
Native mode manager.

Lightweight port of the legacy state-machine idea. The goal is not to clone
the old subsystem, but to provide durable operating modes that reflect account
state and recent health, and can be consumed by the native decision layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class NativeMode:
    name: str
    max_positions: int
    confidence_floor: float
    max_trade_usdt: float


class NativeModeManager:
    def __init__(
        self,
        *,
        bootstrap_trade_usdt: float = 20.0,
        recovery_trade_usdt: float = 50.0,
        normal_trade_usdt: float = 150.0,
    ) -> None:
        self.bootstrap_trade_usdt = max(1.0, float(bootstrap_trade_usdt))
        self.recovery_trade_usdt = max(1.0, float(recovery_trade_usdt))
        self.normal_trade_usdt = max(1.0, float(normal_trade_usdt))
        self._manual_mode: str | None = None
        self._last_mode: str = "BOOTSTRAP"

    def evaluate(
        self, *, nav: float, metrics: dict[str, Any], state: Any | None = None
    ) -> NativeMode:
        if self._manual_mode:
            constraints = self.get_constraints(self._manual_mode, nav=nav)
            mode = NativeMode(
                self._manual_mode,
                int(constraints["max_positions"]),
                float(constraints["confidence_floor"]),
                float(constraints["max_trade_usdt"]),
            )
            self._last_mode = mode.name
            return mode
        drawdown_pct = float(metrics.get("drawdown_pct", 0.0) or 0.0)
        repeated_failures = bool(metrics.get("repeated_failures", False))
        manual_pause = bool(metrics.get("manual_pause", False))
        health_ok = bool(metrics.get("health_ok", True))
        first_trade_executed = bool(metrics.get("first_trade_executed", False))
        has_positions = bool(metrics.get("has_positions", False))

        if manual_pause:
            mode_name = "PAUSED"
        elif repeated_failures or drawdown_pct >= 5.0:
            mode_name = "SAFE"
        elif drawdown_pct >= 2.0:
            mode_name = "PROTECTIVE"
        elif not health_ok:
            mode_name = "RECOVERY"
        elif nav < 100.0 and not has_positions and not first_trade_executed:
            mode_name = "BOOTSTRAP"
        elif nav < 500.0:
            mode_name = "RECOVERY"
        elif nav < 2000.0:
            mode_name = "NORMAL"
        else:
            mode_name = "GROWTH"

        constraints = self.get_constraints(mode_name, nav=nav)
        mode = NativeMode(
            mode_name,
            int(constraints["max_positions"]),
            float(constraints["confidence_floor"]),
            float(constraints["max_trade_usdt"]),
        )
        self._last_mode = mode.name
        return mode

    async def get_current_mode(self) -> str:
        return self._manual_mode or self._last_mode

    async def set_mode(self, mode: str) -> bool:
        mode_name = str(mode or "").upper()
        if mode_name not in {
            "PAUSED",
            "SAFE",
            "PROTECTIVE",
            "RECOVERY",
            "BOOTSTRAP",
            "NORMAL",
            "GROWTH",
        }:
            return False
        self._manual_mode = mode_name
        self._last_mode = mode_name
        return True

    def get_constraints(self, mode: str, *, nav: float = 0.0) -> dict[str, float | int | bool]:
        mode_name = str(mode or "").upper()
        if mode_name == "PAUSED":
            return {
                "allow_new": False,
                "max_positions": 0,
                "confidence_floor": 1.0,
                "max_trade_usdt": 0.0,
            }
        if mode_name == "SAFE":
            return {
                "allow_new": True,
                "max_positions": 1,
                "confidence_floor": 0.90,
                "max_trade_usdt": min(30.0, max(1.0, nav * 0.03)),
            }
        if mode_name == "PROTECTIVE":
            return {
                "allow_new": True,
                "max_positions": 2,
                "confidence_floor": 0.60,
                "max_trade_usdt": min(50.0, max(1.0, nav * 0.05)),
            }
        if mode_name == "RECOVERY":
            return {
                "allow_new": True,
                "max_positions": 2,
                "confidence_floor": 0.50,
                "max_trade_usdt": min(
                    self.recovery_trade_usdt,
                    max(1.0, nav * 0.08 if nav > 0 else self.recovery_trade_usdt),
                ),
            }
        if mode_name == "NORMAL":
            return {
                "allow_new": True,
                "max_positions": 3,
                "confidence_floor": 0.45,
                "max_trade_usdt": self.normal_trade_usdt,
            }
        if mode_name == "GROWTH":
            return {
                "allow_new": True,
                "max_positions": 5,
                "confidence_floor": 0.40,
                "max_trade_usdt": max(25.0, nav * 0.15 if nav > 0 else 25.0),
            }
        return {
            "allow_new": True,
            "max_positions": 1,
            "confidence_floor": 0.50,
            "max_trade_usdt": self.bootstrap_trade_usdt,
        }
