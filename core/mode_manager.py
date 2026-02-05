"""
ModeManager subsystem extracted from MetaController.
Handles mode switching, mode evaluation, and mode state tracking (NORMAL, FOCUS, RECOVERY, BOOTSTRAP).
"""
from typing import Dict, Any, Set, Optional
import time

class ModeManager:
    def __init__(self, logger, config):
        self.logger = logger
        self.config = config
        
        # Mode state tracking
        self._current_mode = "BOOTSTRAP"  # Start in bootstrap mode
        self._last_mode = None
        self._mode_switch_count = 0
        self._mode_switch_timestamps = {}
        
        # ═══════════════════════════════════════════════════════════════════
        # MODES SOP MATRIX (HARD ENVELOPE)
        # ═══════════════════════════════════════════════════════════════════
        self._SOP_MATRIX = {
            "SAFE": {
                "max_trade_usdt": 30.0,
                "max_positions": 1,
                "confidence_floor": 0.85,
                "cooldown_sec": 300,
                "probing_enabled": False,
                "objective": "Capital protection"
            },
            "PROTECTIVE": {
                "max_trade_usdt": 0.0,  # Blocked
                "max_positions": 0,    # Blocked
                "confidence_floor": 0.95, # Strictly high (Liquidation only basically)
                "cooldown_sec": 0,
                "probing_enabled": False,
                "objective": "Capital defense (New Positions Blocked)"
            },
            "NORMAL": {
                "max_trade_usdt": 150.0,
                "max_positions": 3,
                "confidence_floor": 0.65,
                "cooldown_sec": 120,
                "probing_enabled": True,
                "objective": "Balanced trading (Default Profit Engine)"
            },
            "AGGRESSIVE": {
                "max_trade_usdt": 300.0,
                "max_positions": 5,
                "confidence_floor": 0.55,
                "cooldown_sec": 60,
                "probing_enabled": True,
                "objective": "Velocity recovery"
            },
            "RECOVERY": {
                "max_trade_usdt": 50.0,
                "max_positions": 1,
                "confidence_floor": 0.80,
                "cooldown_sec": 240,
                "probing_enabled": False,
                "objective": "System stabilization (Fault Recovery)"
            },
            "BOOTSTRAP": {
                "max_trade_usdt": 20.0,
                "max_positions": 1,
                "confidence_floor": 0.70,
                "cooldown_sec": 60,
                "probing_enabled": False,
                "objective": "Initial funding (Bootstrap)"
            },
            "PAUSED": {
                "max_trade_usdt": 0.0,
                "max_positions": 0,
                "confidence_floor": 1.0,
                "cooldown_sec": 0,
                "probing_enabled": False,
                "objective": "Human interaction required (Paused)"
            },
            "SIGNAL_ONLY": {
                "max_trade_usdt": 1000.0, # Simulated capacity
                "max_positions": 10,
                "confidence_floor": 0.40, # Wide funnel for signals
                "cooldown_sec": 0,
                "probing_enabled": True,
                "objective": "SaaS/API Signal Emission (No Execution)"
            }
        }
        
        # Mode-specific symbol limits ( legacy compatibility )
        self._bootstrap_symbol_limit = 1 
        self._active_symbol_limit = self._bootstrap_symbol_limit 
        
        # Mode configuration thresholds
        self._mode_config = {
            "min_mode_duration_sec": float(getattr(config, 'MIN_MODE_DURATION_SEC', 30.0)),
        }
        
        # Mode flags
        self._mandatory_sell_mode_active = False
        
        # Condition tracking for hysteresis
        self._condition_start_times = {} # condition_name -> start_timestamp
        
        # Event emission placeholder: can be replaced with a real event bus or callback system
        self._event_handlers = []
        
        self.logger.info(f"[ModeManager:Init] Initialized with mode: {self._current_mode}")

    async def evaluate_state_machine(self, metrics: Dict[str, Any]):
        """
        Evaluate mode transition conditions based on SOP State Machine.
        """
        now = time.time()
        current = self._current_mode
        status = str(metrics.get("health_status", "unknown")).lower()
        
        # Helper: Async flat check if metrics are missing
        async def _check_is_flat_async(m):
            # This is a fallback; usually metrics.has_positions is provided
            return m.get("has_positions") is False

        # =====================================================================
        # 1. IMMEDIATE TRANSITIONS (Priority: BOOTSTRAP > RECOVERY > Safety)
        # =====================================================================
        
        # 1. BOOTSTRAP GATE (Highest Precedence)
        # IF portfolio is flat AND (startup OR prolonged_idle) -> BOOTSTRAP
        is_flat = metrics.get("has_positions") is False
        
        # Double check if metrics missing (defensive)
        if "has_positions" not in metrics:
            is_flat = await self._check_is_flat_async(metrics)
        idle_time = float(metrics.get("idle_time_sec", 0.0))
        prolonged_idle = idle_time > float(getattr(self.config, "PROLONGED_IDLE_SECONDS", 3600))
        
        if is_flat and (metrics.get("is_restart") or prolonged_idle):
            if current != "BOOTSTRAP":
                self.logger.warning("[ModeManager:SOP] 🟦 Triggering BOOTSTRAP: portfolio_flat=True, startup=%s, idle=%ds", 
                                   metrics.get("is_restart"), int(idle_time))
                self.set_mode("BOOTSTRAP")
                return

        # 2. Circuit Breaker Opens -> SAFE (Immediate)
        if metrics.get("circuit_breaker_open", False):
             if current != "SAFE":
                  self.logger.warning("[EmergencySOP] 🛑 Circuit Breaker OPEN -> Switching to SAFE immediately.")
                  self.set_mode("SAFE")
                  return

        # 3. Manual Pause / Compliance -> PAUSED (Immediate)
        if metrics.get("manual_pause", False) or "paused" in status:
             if current != "PAUSED":
                  self.logger.warning("[SOP:Authority] ⏸️ Manual Pause Detected -> Switching to PAUSED mode.")
                  self.set_mode("PAUSED")
                  return

        # 4. Manual Operator Freeze (via Health/Status) -> SAFE (Immediate)
        if "frozen" in status or metrics.get("manual_freeze", False):
             if current != "SAFE":
                  self.logger.warning("[EmergencySOP] 🛑 Manual Freeze Detected -> Switching to SAFE immediately.")
                  self.set_mode("SAFE")
                  return

        # 4. Reservation Corruption (Integrity Error) -> SAFE (Immediate)
        if metrics.get("integrity_error", False):
             if current != "SAFE":
                  self.logger.warning("[EmergencySOP] 🛑 Integrity/Reservation Corruption -> Switching to SAFE immediately.")
                  self.set_mode("SAFE")
                  return

        # 5. Repeated HYG Failures -> PROTECTIVE (Immediate)
        if metrics.get("repeated_failures", False):
             if current not in ("SAFE", "PROTECTIVE"):
                  self.logger.warning("[EmergencySOP] ⚠️ Repeated Outcomes/HYG Failures -> Switching to PROTECTIVE immediately.")
                  self.set_mode("PROTECTIVE")
                  return

        # 6. RECOVERY Triggers (Restart / Health Faults / Forced Liquidation)
        # RECOVERY: stabilize, don’t freeze.
        # Triggers: Restart, Exchange/API fault (health_ok=False), Inconsistent state.
        # NOTE: BOOTSTRAP check above already handled flat-portfolio restart.
        health_ok = metrics.get("health_ok", True)
        is_restart = metrics.get("is_restart", False)
        forced_liq = metrics.get("forced_liquidation", False)
        first_trades = metrics.get("first_trade_executed", False)
        
        if ((is_restart and not first_trades) or not health_ok or forced_liq):
            if current not in ("RECOVERY", "BOOTSTRAP", "SAFE", "PAUSED"):
                self.logger.warning("[ModeManager:SOP] 🟨 Triggering RECOVERY: restart=%s, health_ok=%s, forced_liq=%s, status=%s", 
                                   is_restart, health_ok, forced_liq, status)
                self.set_mode("RECOVERY")
                return

        # 7. Drawdown Breach -> PROTECTIVE (SOP REQUIREMENT)
        if current in ("NORMAL", "AGGRESSIVE"):
            protective_dd_limit = float(getattr(self.config, "PROTECTIVE_DRAWDOWN_LIMIT", 2.0))
            if metrics.get("drawdown_pct", 0.0) >= protective_dd_limit:
                 self.logger.warning("[ModeManager:SOP] 🛡️ Drawdown Breach -> Switching to PROTECTIVE (%.2f%%)", metrics.get("drawdown_pct"))
                 self.set_mode("PROTECTIVE")
                 return
                 
        # 8. Hard Drawdown Breach -> SAFE (Absolute floor)
        if current in ("PROTECTIVE", "NORMAL", "AGGRESSIVE"):
            hard_dd_limit = float(getattr(self.config, "HARD_DRAWDOWN_LIMIT", 5.0))
            if metrics.get("drawdown_pct", 0.0) >= hard_dd_limit:
                self.logger.warning("[ModeManager:SOP] 🛑 Hard Drawdown Breach -> Switching to SAFE (%.2f%%)", metrics.get("drawdown_pct"))
                self.set_mode("SAFE")
                return

        # =====================================================================
        # 2. CONDITION-BASED TRANSITIONS (Hysteresis)
        # =====================================================================

        # NORMAL -> AGGRESSIVE (30 min hysteresis)
        if current == "NORMAL":
            target_rr = float(metrics.get("target_run_rate", 20.0))
            rr_low = float(metrics.get("run_rate", 0.0)) < (0.7 * target_rr)
            dd_low = float(metrics.get("drawdown_pct", 0.0)) < 1.0
            
            if rr_low and dd_low:
                if self._check_condition_persistence("normal_to_aggressive", now, 1800):
                    self.logger.info("[ModeManager:SOP] NORMAL -> AGGRESSIVE: Low run-rate and drawdown persistence met.")
                    self.set_mode("AGGRESSIVE")
                    return
            else:
                self._reset_condition("normal_to_aggressive")

        # AGGRESSIVE -> NORMAL (20 min hysteresis)
        if current == "AGGRESSIVE":
            target_rr = float(metrics.get("target_run_rate", 20.0))
            rr_ok = float(metrics.get("run_rate", 0.0)) >= target_rr
            
            if rr_ok:
                if self._check_condition_persistence("aggressive_to_normal", now, 1200):
                    self.logger.info("[ModeManager:SOP] AGGRESSIVE -> NORMAL: Target run-rate recovery persistence met.")
                    self.set_mode("NORMAL")
                    return
            else:
                self._reset_condition("aggressive_to_normal")

        # NORMAL -> PROTECTIVE (Volatility/Risk)
        if current == "NORMAL":
            vol_high = str(metrics.get("volatility", "")).upper() == "HIGH"
            risk_flags = metrics.get("risk_flags", False)
            
            if vol_high or risk_flags:
                # SOP requirement: Gradual transition (persistence check) for both Volatility and Risk Flags
                if self._check_condition_persistence("normal_to_protective", now, 300):
                    self.logger.info("[ModeManager:SOP] NORMAL -> PROTECTIVE: Volatility/Risk flags persistence met.")
                    self.set_mode("PROTECTIVE")
                    return
            else:
                self._reset_condition("normal_to_protective")

        # RECOVERY -> NORMAL (Hysteresis)
        if current == "RECOVERY":
            health_ok = metrics.get("health_ok", False)
            first_trades = metrics.get("first_trade_executed", False)
            
            if health_ok and first_trades:
                stabilization_sec = int(getattr(self.config, "RECOVERY_STABILIZATION_MIN", 10)) * 60
                if self._check_condition_persistence("recovery_to_normal", now, stabilization_sec):
                    self.logger.info("[ModeManager:SOP] RECOVERY -> NORMAL: Health stabilization met.")
                    self.set_mode("NORMAL")
                    return
            else:
                self._reset_condition("recovery_to_normal")

        # PROTECTIVE -> RECOVERY (Stabilization)
        if current == "PROTECTIVE":
            health_ok = metrics.get("health_ok", False)
            # PnL stabilize: DD < threshold (e.g. 1.5% if trigger was 2%)
            dd_stable = float(metrics.get("drawdown_pct", 0.0)) < 1.5
            
            if health_ok and dd_stable:
                if self._check_condition_persistence("protective_to_recovery", now, 600): # 10 min stabilization
                    self.logger.info("[ModeManager:SOP] 🛡️ -> 🟨 PROTECTIVE -> RECOVERY: Health and PnL stabilized.")
                    self.set_mode("RECOVERY")
                    return
            else:
                self._reset_condition("protective_to_recovery")

        # BOOTSTRAP -> NORMAL
        if current == "BOOTSTRAP":
            if metrics.get("first_trade_executed", False) or metrics.get("has_positions", False):
                self.logger.info("[ModeManager:SOP] BOOTSTRAP -> NORMAL: First trade or position detected.")
                self.transition_from_bootstrap()
                return

    def _check_condition_persistence(self, name: str, now: float, duration_sec: float) -> bool:
        """Check if a named condition has persisted for the required duration."""
        if name not in self._condition_start_times:
            self._condition_start_times[name] = now
            return False
        
        elapsed = now - self._condition_start_times[name]
        return elapsed >= duration_sec

    def _reset_condition(self, name: str):
        """Reset a persistence timer."""
        if name in self._condition_start_times:
            del self._condition_start_times[name]


    def get_mode(self) -> str:
        """Get the current trading mode."""
        return self._current_mode

    def get_envelope(self) -> Dict[str, Any]:
        """Get the hard envelope (constraints) for the current mode."""
        return self._SOP_MATRIX.get(self._current_mode, self._SOP_MATRIX["NORMAL"])

    def set_mode(self, mode: str):
        """Set the current trading mode and emit an event if changed."""
        mode = mode.upper()
        if mode not in self._SOP_MATRIX:
            self.logger.warning(f"[ModeManager] Attempted to set invalid mode: {mode}")
            return

        if mode != self._current_mode:
            old_mode = self._current_mode
            self._current_mode = mode
            self._last_mode = old_mode
            self._mode_switch_count += 1
            self._mode_switch_timestamps[mode] = time.time()
            
            # Update active symbol limit based on envelope
            self._active_symbol_limit = self._SOP_MATRIX[mode]["max_positions"]
            
            self.logger.info(f"[ModeManager] Mode changed from {old_mode} to {mode} | Objective: {self._SOP_MATRIX[mode]['objective']}")
            self._emit_event('mode_changed', {'old_mode': old_mode, 'new_mode': mode})

    def get_mode_info(self) -> Dict[str, Any]:
        """Get detailed mode information."""
        info = {
            "current_mode": self._current_mode,
            "last_mode": self._last_mode,
            "mode_switch_count": self._mode_switch_count,
            "mode_switch_timestamps": self._mode_switch_timestamps.copy(),
            "active_symbol_limit": self._active_symbol_limit,
            "envelope": self.get_envelope()
        }
        return info

    def get_active_symbol_limit(self) -> int:
        """Get the current active symbol limit based on mode."""
        return self._active_symbol_limit

    def set_mandatory_sell_mode(self, active: bool):
        """Set mandatory sell mode flag."""
        self._mandatory_sell_mode_active = active

    def is_mandatory_sell_mode_active(self) -> bool:
        """Check if mandatory sell mode is active."""
        return self._mandatory_sell_mode_active

    def transition_from_bootstrap(self):
        """Transition from bootstrap mode to normal mode."""
        if self._current_mode == "BOOTSTRAP":
            self.set_mode("NORMAL")
            self.logger.info(f"[ModeManager] Bootstrap transition complete. Active symbol limit: {self._active_symbol_limit}")

    def register_event_handler(self, handler):
        """Register an event handler callback for mode events."""
        self._event_handlers.append(handler)

    def _emit_event(self, event_type, payload):
        """Emit an event to all registered handlers (placeholder for event bus)."""
        for handler in self._event_handlers:
            try:
                handler(event_type, payload)
            except Exception as e:
                self.logger.debug(f"ModeManager event handler error: {e}")
