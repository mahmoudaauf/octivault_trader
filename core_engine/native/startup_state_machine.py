"""
Native L0: Startup State Machine (Phase 8.4.2)

Enforces a strict state progression from restart to trading-ready:

    BOOTING
      ↓ (dependencies initialized)
    HYDRATING
      ↓ (portfolio reconstructed from journal/exchange)
    RECONCILING
      ↓ (balances validated against orders/fills)
    VALIDATING
      ↓ (NAV sanity checks, TP/SL restoration)
    READY
      ↓ (trading allowed)

Critical rule: NEW BUY DECISIONS ONLY ALLOWED IN READY STATE

This prevents the dangerous scenario where:
  - System restarts
  - Balances loaded but positions unknown
  - Trading engine active
  - Opens new positions while existing positions unprotected
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class StartupState(Enum):
    """System startup lifecycle state."""

    BOOTING = "BOOTING"  # Dependencies being initialized
    HYDRATING = "HYDRATING"  # Reconstructing position state from journal/exchange
    RECONCILING = "RECONCILING"  # Validating balance consistency
    VALIDATING = "VALIDATING"  # Checking NAV, TP/SL, reserved capital
    READY = "READY"  # Trading allowed
    FAILED = "FAILED"  # Startup failed, manual intervention needed


@dataclass
class StateTransition:
    """Record of a state transition."""

    from_state: StartupState
    to_state: StartupState
    reason: str
    timestamp: float = field(default_factory=time.time)
    error: Optional[str] = None


class NativeStartupStateMachine:
    """
    Orchestrates the startup sequence and gates trading until READY.

    Usage:
        sm = NativeStartupStateMachine(decision_engine)
        sm.set_callback(StartupState.HYDRATING, hydration_engine.hydrate)
        sm.set_callback(StartupState.RECONCILING, reconciler.reconcile)
        sm.set_callback(StartupState.VALIDATING, validator.validate)

        await sm.run_startup()  # Runs full sequence

        # During trading:
        if sm.is_ready():  # True only in READY state
            await decision_engine.evaluate_buy_signals()
    """

    def __init__(self, decision_engine: Optional[Any] = None) -> None:
        """
        Initialize the state machine.

        Args:
            decision_engine: DecisionEngine to gate (buy decisions blocked until READY)
        """
        self._state = StartupState.BOOTING
        self._decision_engine = decision_engine
        self._transitions: list[StateTransition] = []
        self._callbacks: dict[StartupState, Callable] = {}
        self._lock = asyncio.Lock()
        self._startup_time = time.time()

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def is_ready(self) -> bool:
        """True if system is ready for trading (in READY state)."""
        return self._state == StartupState.READY

    def is_healthy(self) -> bool:
        """True if system is in READY or later (not FAILED)."""
        return self._state != StartupState.FAILED and self._state in (
            StartupState.READY,
            StartupState.VALIDATING,
        )

    def current_state(self) -> StartupState:
        """Get current startup state."""
        return self._state

    def startup_elapsed_sec(self) -> float:
        """Elapsed time since startup began."""
        return time.time() - self._startup_time

    def set_callback(self, state: StartupState, callback: Callable) -> None:
        """Register a callback to run when entering a state."""
        self._callbacks[state] = callback
        logger.info(f"Registered startup callback for {state.value}")

    def can_buy(self) -> bool:
        """
        Gate for BUY decisions.

        Returns True only if system is in READY state.
        Used by DecisionEngine to reject BUY decisions during startup.
        """
        if not self.is_ready():
            logger.debug(
                f"BUY blocked: system not ready (state={self._state.value}, "
                f"elapsed={self.startup_elapsed_sec():.1f}s)"
            )
            return False
        return True

    # ──────────────────────────────────────────────────────────────────────────
    # Startup sequence
    # ──────────────────────────────────────────────────────────────────────────

    async def run_startup(self, timeout_sec: float = 60.0) -> bool:
        """
        Run the full startup sequence.

        Progression: BOOTING → HYDRATING → RECONCILING → VALIDATING → READY

        Returns:
            True if startup completed successfully, False if failed
        """
        logger.info("🚀 Starting system initialization sequence...")

        try:
            # BOOTING → HYDRATING (dependencies ready)
            logger.info("📝 Phase 1: Booting (waiting for dependencies)...")
            await self._transition_to(
                StartupState.HYDRATING,
                reason="Dependencies initialized",
            )
            await self._run_state_callback(StartupState.HYDRATING, timeout_sec)

            # HYDRATING → RECONCILING (positions reconstructed)
            logger.info("🔄 Phase 2: Reconciling (validating balance consistency)...")
            await self._transition_to(
                StartupState.RECONCILING,
                reason="Position hydration complete",
            )
            await self._run_state_callback(StartupState.RECONCILING, timeout_sec)

            # RECONCILING → VALIDATING (consistency checked)
            logger.info("✓ Phase 3: Validating (checking NAV and TP/SL)...")
            await self._transition_to(
                StartupState.VALIDATING,
                reason="Balance consistency verified",
            )
            await self._run_state_callback(StartupState.VALIDATING, timeout_sec)

            # VALIDATING → READY (all checks passed)
            logger.info("✅ Phase 4: Ready (trading enabled)...")
            await self._transition_to(
                StartupState.READY,
                reason="All startup checks passed",
            )

            elapsed = self.startup_elapsed_sec()
            logger.info(f"🎉 Startup complete in {elapsed:.1f}s. System ready for trading!")
            return True

        except asyncio.TimeoutError:
            await self._transition_to(
                StartupState.FAILED,
                reason="Startup timeout",
                error="One or more startup phases exceeded timeout",
            )
            return False

        except Exception as e:
            await self._transition_to(
                StartupState.FAILED,
                reason="Startup error",
                error=str(e),
            )
            logger.exception(f"❌ Startup failed: {e}")
            return False

    async def fail_startup(self, reason: str, error: Optional[str] = None) -> None:
        """Manually transition to FAILED state."""
        await self._transition_to(StartupState.FAILED, reason=reason, error=error)

    # ──────────────────────────────────────────────────────────────────────────
    # Internal machinery
    # ──────────────────────────────────────────────────────────────────────────

    async def _transition_to(
        self,
        next_state: StartupState,
        reason: str = "",
        error: Optional[str] = None,
    ) -> None:
        """
        Transition to a new state (thread-safe).

        Logs the transition and records it for audit.
        """
        async with self._lock:
            old_state = self._state
            self._state = next_state

            transition = StateTransition(
                from_state=old_state,
                to_state=next_state,
                reason=reason,
                error=error,
            )
            self._transitions.append(transition)

            log_level = logging.WARNING if error else logging.INFO
            logger.log(
                log_level,
                f"State transition: {old_state.value} → {next_state.value} ({reason})"
                + (f" — Error: {error}" if error else ""),
            )

            # Gate: if we fail, immediately disable BUY
            if next_state == StartupState.FAILED and self._decision_engine:
                logger.warning("⚠️  BUY decisions will be blocked (startup failed)")

    async def _run_state_callback(self, state: StartupState, timeout_sec: float = 30.0) -> None:
        """Run the callback for a state (if registered) with timeout."""
        callback = self._callbacks.get(state)
        if not callback:
            logger.debug(f"No callback registered for {state.value}")
            return

        logger.info(f"  Running startup task for {state.value}...")
        try:
            result = callback()
            if asyncio.iscoroutine(result):
                result = await asyncio.wait_for(result, timeout=timeout_sec)
            logger.info(f"  ✅ {state.value} task completed successfully")
        except asyncio.TimeoutError:
            logger.error(f"  ❌ {state.value} task timed out after {timeout_sec}s")
            raise
        except Exception as e:
            logger.error(f"  ❌ {state.value} task failed: {e}")
            raise

    # ──────────────────────────────────────────────────────────────────────────
    # Diagnostics
    # ──────────────────────────────────────────────────────────────────────────

    def get_transition_history(self) -> list[StateTransition]:
        """Get full transition history for debugging."""
        return self._transitions.copy()

    def diagnostic_summary(self) -> str:
        """Return a human-readable diagnostic summary."""
        lines = [
            "═" * 70,
            "STARTUP STATE MACHINE DIAGNOSTIC",
            "═" * 70,
            f"Current state: {self._state.value}",
            f"Is ready: {self.is_ready()}",
            f"Can buy: {self.can_buy()}",
            f"Elapsed: {self.startup_elapsed_sec():.1f}s",
            "",
            "Transition history:",
        ]

        for i, t in enumerate(self._transitions, 1):
            elapsed = t.timestamp - self._startup_time
            lines.append(f"  {i}. [{elapsed:6.1f}s] {t.from_state.value} → {t.to_state.value}")
            if t.reason:
                lines.append(f"     Reason: {t.reason}")
            if t.error:
                lines.append(f"     Error: {t.error}")

        lines.append("═" * 70)
        return "\n".join(lines)


__all__ = [
    "NativeStartupStateMachine",
    "StartupState",
    "StateTransition",
]
