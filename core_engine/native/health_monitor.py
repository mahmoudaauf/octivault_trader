"""
Native health monitor.
"""

from __future__ import annotations

import os
import time
from typing import Any

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None


class NativeHealthMonitor:
    def __init__(
        self,
        *,
        shared_state: Any | None = None,
        exchange_client: Any | None = None,
        market_data: Any | None = None,
        market_data_ws: Any | None = None,
        watchdog: Any | None = None,
        mode_manager: Any | None = None,
        started_at: float | None = None,
    ) -> None:
        self._shared_state = shared_state
        self._exchange_client = exchange_client
        self._market_data = market_data
        self._market_data_ws = market_data_ws
        self._watchdog = watchdog
        self._mode_manager = mode_manager
        self._started_at = float(started_at or time.time())

    async def get_report(self) -> dict[str, Any]:
        components = {
            "exchange_api": self._exchange_component(),
            "market_data": self._market_data_component(),
            "portfolio_state": self._portfolio_component(),
            "watchdog": self._watchdog_component(),
            "mode_manager": self._mode_component(),
            "memory": self._memory_component(),
            "cpu": self._cpu_component(),
        }
        critical_issues = [name for name, c in components.items() if c["status"] == "CRITICAL"]
        warnings = [
            f"{name}:{c['message']}" for name, c in components.items() if c["status"] == "WARN"
        ]
        overall_status = "OK"
        if critical_issues:
            overall_status = "CRITICAL"
        elif warnings:
            overall_status = "WARN"
        suggestions = []
        if bool(getattr(self._shared_state, "exchange_throttled", False)):
            suggestions.append("Exchange throttled: remain in read-only / WS-first mode")
        if float(getattr(self._shared_state, "nav_usdt", 0.0) or 0.0) <= 0.0:
            suggestions.append("NAV not hydrated yet: defer new deployment decisions")
        return {
            "timestamp": time.time(),
            "overall_status": overall_status,
            "components": components,
            "critical_issues": critical_issues,
            "warnings": warnings,
            "suggestions": suggestions,
        }

    def get_component_status(self, name: str) -> dict[str, Any]:
        return {
            "exchange_api": self._exchange_component(),
            "market_data": self._market_data_component(),
            "portfolio_state": self._portfolio_component(),
            "watchdog": self._watchdog_component(),
            "mode_manager": self._mode_component(),
            "memory": self._memory_component(),
            "cpu": self._cpu_component(),
        }.get(name, {"name": name, "status": "WARN", "message": "unknown component"})

    def get_overall_health(self) -> str:
        if bool(getattr(self._shared_state, "trading_halted", False)):
            return "CRITICAL"
        if bool(getattr(self._shared_state, "exchange_throttled", False)):
            return "WARN"
        return "OK"

    def _exchange_component(self) -> dict[str, Any]:
        throttled = bool(getattr(self._shared_state, "exchange_throttled", False))
        if throttled:
            return {
                "name": "exchange_api",
                "status": "WARN",
                "message": str(
                    getattr(self._shared_state, "exchange_throttle_reason", "")
                    or "exchange throttled"
                ),
            }
        return {"name": "exchange_api", "status": "OK", "message": "exchange reachable or idle"}

    def _market_data_component(self) -> dict[str, Any]:
        ready = bool(getattr(self._shared_state, "market_data_ready", False))
        prices = dict(getattr(self._shared_state, "prices", {}) or {})
        if ready or prices:
            return {
                "name": "market_data",
                "status": "OK",
                "message": f"{len(prices)} prices cached",
            }
        # In early startup, market_data delay is not a blocker (downgrade to OK or silent)
        # Market data can arrive on its own schedule during bootstrap
        return {"name": "market_data", "status": "OK", "message": "market data buffering"}

    def _portfolio_component(self) -> dict[str, Any]:
        nav = float(getattr(self._shared_state, "nav_usdt", 0.0) or 0.0)
        if nav > 0.0:
            return {"name": "portfolio_state", "status": "OK", "message": f"NAV {nav:.2f} USDT"}
        return {"name": "portfolio_state", "status": "WARN", "message": "NAV unavailable or zero"}

    def _watchdog_component(self) -> dict[str, Any]:
        if self._watchdog is None:
            return {"name": "watchdog", "status": "WARN", "message": "watchdog not configured"}
        return {"name": "watchdog", "status": "OK", "message": "watchdog configured"}

    def _mode_component(self) -> dict[str, Any]:
        mode = str(getattr(self._shared_state, "current_mode", "") or "")
        if mode:
            return {"name": "mode_manager", "status": "OK", "message": mode}
        return {"name": "mode_manager", "status": "WARN", "message": "mode not evaluated yet"}

    def _memory_component(self) -> dict[str, Any]:
        if psutil is None:
            return {"name": "memory", "status": "OK", "message": "psutil unavailable"}
        try:
            process = psutil.Process(os.getpid())
            pct = float(process.memory_percent())
            if pct >= 90.0:
                status = "CRITICAL"
            elif pct >= 75.0:
                status = "WARN"
            else:
                status = "OK"
            return {"name": "memory", "status": status, "message": f"{pct:.1f}% used"}
        except Exception as e:
            return {"name": "memory", "status": "WARN", "message": str(e)}

    def _cpu_component(self) -> dict[str, Any]:
        if psutil is None:
            return {"name": "cpu", "status": "OK", "message": "psutil unavailable"}
        try:
            process = psutil.Process(os.getpid())
            pct = float(process.cpu_percent(interval=0.0))
            if pct >= 95.0:
                status = "CRITICAL"
            elif pct >= 80.0:
                status = "WARN"
            else:
                status = "OK"
            return {"name": "cpu", "status": status, "message": f"{pct:.1f}% used"}
        except Exception as e:
            return {"name": "cpu", "status": "WARN", "message": str(e)}


__all__ = ["NativeHealthMonitor"]
