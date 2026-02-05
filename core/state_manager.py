"""
StateManager subsystem extracted from MetaController.
Handles liveness, cooldowns, health, and system state transitions.
"""

import time
import asyncio as _asyncio
from typing import Optional, Dict, Any
from collections import defaultdict

# Component Status Logger
try:
    from core.component_status_logger import ComponentStatusLogger as CSL
except ImportError:
    class CSL:
        """Component Status Logger stub for environments without CSL available."""
        @staticmethod
        def set_status(*a, **k): pass
        @staticmethod
        async def heartbeat(*a, **k): pass
        @staticmethod
        def log_status(*a, **k): pass

class StateManager:
    def __init__(self, shared_state, config, logger, component_name: str = "StateManager"):
        self.shared_state = shared_state
        self.config = config
        self.logger = logger
        self.component_name = component_name

        # Execution attempts tracking
        self._execution_attempts_this_cycle = 0

        # Liveness tracking
        self._last_execution_ts = {}
        self._execution_cooldowns = {}

        # KPI metrics tracking
        self._kpi_metrics = {
            "total_realized_pnl": 0.0,
            "execution_count": 0,
            "liquidity_requests": 0,
            "consecutive_zero_execution_ticks": 0,
            "last_executed_tick": 0,
            "deadlock_detected": False,
            "hourly_target_usdt": 20.0,
            "error_count_by_type": defaultdict(int)
        }
        self._performance_lock = _asyncio.Lock()
        
        # Policy modifiers (Soft Controllers)
        self.policy_modifiers = {"cooldown_nudge": 0.0}

    def set_policy_modifiers(self, modifiers: Dict[str, Any]):
        """Set policy modifiers for soft control."""
        if modifiers:
            self.policy_modifiers.update(modifiers)

    def get_execution_attempts_this_cycle(self):
        """Get the number of execution attempts in the current cycle."""
        return self._execution_attempts_this_cycle

    def increment_execution_attempts(self):
        """Increment the execution attempts counter for this cycle."""
        self._execution_attempts_this_cycle += 1

    def reset_execution_attempts(self):
        """Reset the execution attempts counter for a new cycle."""
        self._execution_attempts_this_cycle = 0

    async def check_execution_liveness(self, symbol=None, side=None):
        """
        Check execution liveness for a symbol/side combination.
        Returns (is_viable, reason)
        """
        try:
            # Check cooldowns
            if symbol and side:
                cooldown_key = f"{symbol}_{side}"
                last_ts = self._last_execution_ts.get(cooldown_key, 0)
                
                # Base cooldown + Policy Nudge
                base_cooldown = getattr(self.config, 'EXECUTION_COOLDOWN_SEC', 60)
                nudge = self.policy_modifiers.get("cooldown_nudge", 0.0)
                cooldown_sec = max(0.0, base_cooldown + nudge)

                if time.time() - last_ts < cooldown_sec:
                    return False, f"COOLDOWN_ACTIVE_{cooldown_sec:.1f}s"

            # Check global execution rate limits
            max_attempts = getattr(self.config, 'MAX_EXECUTION_ATTEMPTS_PER_CYCLE', 5)
            if self._execution_attempts_this_cycle >= max_attempts:
                return False, f"MAX_ATTEMPTS_EXCEEDED_{max_attempts}"

            # Check circuit breaker
            if hasattr(self.shared_state, 'is_circuit_breaker_open'):
                if await self.shared_state.is_circuit_breaker_open():
                    return False, "CIRCUIT_BREAKER_OPEN"

            return True, "VIABLE"

        except Exception as e:
            self.logger.warning(f"StateManager liveness check failed: {e}")
            return True, "LIVENESS_CHECK_FAILED"

    def update_liveness(self, symbol=None, side=None):
        """Update liveness tracking after successful execution."""
        if symbol and side:
            cooldown_key = f"{symbol}_{side}"
            self._last_execution_ts[cooldown_key] = time.time()

    def update_health(self, status, detail):
        """Update system health status."""
        try:
            fn = getattr(self.shared_state, "update_system_health", None)
            if callable(fn):
                fn(component="StateManager", status=status, detail=detail)
            else:
                if hasattr(self.shared_state, "system_health"):
                    self.shared_state.system_health["StateManager"] = {
                        "status": status,
                        "detail": detail,
                        "ts": time.time()
                    }
        except Exception:
            self.logger.debug("Fallback update_system_health failed.", exc_info=True)

    def check_cooldown(self, symbol=None):
        """Check if a symbol is in cooldown."""
        if not symbol:
            return False

        last_ts = self._last_execution_ts.get(symbol, 0)
        cooldown_sec = getattr(self.config, 'SYMBOL_COOLDOWN_SEC', 300)

        return time.time() - last_ts < cooldown_sec

    def get_system_status(self):
        """Get comprehensive system status."""
        return {
            "execution_attempts": self._execution_attempts_this_cycle,
            "last_execution_ts": self._last_execution_ts.copy(),
            "cooldowns": self._execution_cooldowns.copy(),
            "kpi_metrics": dict(self._kpi_metrics),  # Convert defaultdict to regular dict
            "timestamp": time.time()
        }

    async def _health_set(self, status: str, detail: str) -> None:
        """Update system health status."""
        try:
            CSL.log_status(component=self.component_name, status=status, detail=detail)
        except Exception:
            self.logger.debug("CSL.log_status failed.", exc_info=True)
        try:
            fn = getattr(self.shared_state, "update_system_health", None)
            if callable(fn):
                # RULE: Use _safe_await to handle both sync and async SharedState implementations
                from .core_utils import _safe_await
                await _safe_await(fn(component=self.component_name, status=status, detail=detail))
            else:
                if hasattr(self.shared_state, "system_health"):
                    if not isinstance(self.shared_state.system_health, dict):
                        self.shared_state.system_health = {}
                    self.shared_state.system_health[self.component_name] = {"status": status, "detail": detail, "ts": time.time()}
        except Exception:
            self.logger.debug("Fallback update_system_health failed.", exc_info=True)

    async def _heartbeat(self) -> None:
        """System heartbeat functionality."""
        try:
            await CSL.heartbeat(self.component_name, "OK")
        except Exception:
            try:
                CSL.log_status(self.component_name, "Running", "OK")
            except Exception:
                self.logger.debug("CSL.log_status fallback failed.", exc_info=True)
        try:
            # RULE: Use _health_set wrapper for safe, timestamped status updates
            await self._health_set("Healthy", "OK")
        except Exception:
            self.logger.debug("_health_set fallback failed in heartbeat.", exc_info=True)

    async def _update_kpi_metrics(self, metric_type: str, value: float = 1.0, symbol: str = "") -> None:
        """Update KPI metrics aligned with SharedState tracking and global metrics."""
        async with getattr(self, '_performance_lock', _asyncio.Lock()):
            # 1. Internal tracking
            if metric_type == "realized_pnl":
                self._kpi_metrics["total_realized_pnl"] += value
            elif metric_type == "execution":
                self._kpi_metrics["execution_count"] += 1
            elif metric_type == "liquidity_request":
                self._kpi_metrics["liquidity_requests"] += 1
            elif metric_type == "error":
                error_type = str(value) if isinstance(value, type) else "unknown"
                self._kpi_metrics["error_count_by_type"][error_type] += 1

            # 2. Global metrics reporting
            try:
                from core.metrics import increment_counter, record_value
                if metric_type == "realized_pnl":
                    record_value("realized_pnl", value)
                else:
                    increment_counter(f"meta_{metric_type}")
            except ImportError:
                pass

            # 3. SharedState sync
            if hasattr(self.shared_state, "update_kpi_metric"):
                try:
                    from .core_utils import _safe_await
                    await _safe_await(self.shared_state.update_kpi_metric(metric_type, value))
                except Exception:
                    pass
