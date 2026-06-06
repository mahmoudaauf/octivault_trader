"""
ObjectiveFeedbackController — L2 closed-loop controller that auto-calibrates
runtime knobs to keep the bot tracking the +2%/day NAV objective.

Design contract (see OBJECTIVE_FEEDBACK_PLAN.md):
  • Runs every CHECKPOINT_HEARTBEAT_S seconds (default 900 = 15 min).
  • Reads observed pace / drawdown / throughput / economics from SharedState.
  • Bounded PI control on three knobs:
        confidence_floor, size_multiplier, target_throughput_per_hour
  • Writes to shared_state.runtime_overrides (hot-reload dict).
  • Hard kill-switch on drawdown breach (≥5%).
  • Refuses to act on stale / missing telemetry (logs STARVED, no-op).

This module is import-safe: it never raises out of its background task.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

# ------------------------------------------------------------------ #
#  Tunables (overridable via env)                                    #
# ------------------------------------------------------------------ #

DEFAULTS = {
    # Cadence
    "CHECKPOINT_HEARTBEAT_S": 900,  # 15 min
    # Set-points (derived from objective contract)
    "OBJ_DAILY_TARGET_PCT": 0.02,  # +2%/day
    "OBJ_HOURLY_TARGET_PCT": 0.02 / 24,  # ≈0.0833%/h
    "OBJ_MAX_DRAWDOWN_PCT": 0.05,  # 5% kill-switch
    "OBJ_MIN_NET_EDGE_BPS": 5.0,  # avg net profit must beat 5 bps
    # Knob ranges  (clamped)
    # Floor minimum of 0.65 is the empirical breakeven threshold — signals below this
    # have shown ~30% win rate which is deeply unprofitable. Never go below 0.65.
    "OBJ_CONF_FLOOR_MIN": 0.50,
    "OBJ_CONF_FLOOR_MAX": 0.72,
    "OBJ_SIZE_MULT_MIN": 0.50,
    "OBJ_SIZE_MULT_MAX": 1.50,
    "OBJ_THRU_MIN": 2.0,  # trades / hour
    "OBJ_THRU_MAX": 60.0,
    # PI gains (small — we run every 15 min, want gentle motion)
    "OBJ_KP_CONF": 6.0,  # Δconf per (%/h) of pace error
    "OBJ_KI_CONF": 0.5,
    "OBJ_KP_SIZE": 4.0,
    "OBJ_KP_THRU": 50.0,
    "OBJ_KP_DD_PENALTY": 200.0,  # heavy penalty if dd_error > 0
    # Telemetry freshness
    "OBJ_TELEMETRY_MAX_AGE_S": 1800,  # 2 × heartbeat
}


def _cfg(config: Any, key: str) -> float:
    """Read from config attr, then env, then DEFAULTS."""
    if config is not None and hasattr(config, key):
        v = getattr(config, key)
        if v is not None:
            return float(v) if not isinstance(v, bool) else v
    env = os.getenv(key)
    if env is not None:
        try:
            return float(env)
        except ValueError:
            pass
    return DEFAULTS[key]


# ------------------------------------------------------------------ #
#  Telemetry snapshot                                                #
# ------------------------------------------------------------------ #


@dataclass
class Telemetry:
    ok: bool
    age_s: float = 0.0
    nav: float = 0.0
    nav_anchor: float = 0.0  # NAV at session/day start
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    drawdown_pct: float = 0.0
    elapsed_h: float = 0.0
    trades_in_window: int = 0
    win_rate: float = 0.0
    avg_fee_bps: float = 0.0
    avg_slippage_bps: float = 0.0
    avg_net_profit_bps: float = 0.0
    missing: list = field(default_factory=list)


# ------------------------------------------------------------------ #
#  PI state                                                          #
# ------------------------------------------------------------------ #


@dataclass
class ControllerState:
    integral_pace: float = 0.0
    last_action_ts: float = 0.0
    consecutive_dd_breaches: int = 0
    last_knobs: dict[str, float] = field(default_factory=dict)
    history: list = field(default_factory=list)


# ------------------------------------------------------------------ #
#  Controller                                                        #
# ------------------------------------------------------------------ #


class ObjectiveFeedbackController:
    component_name = "ObjectiveFeedbackController"

    def __init__(
        self,
        config: Any = None,
        shared_state: Any = None,
        profit_target_engine: Any = None,
        logger: Optional[logging.Logger] = None,
        artefact_path: Optional[str] = None,
    ):
        self.config = config
        self.ss = shared_state
        self.pte = profit_target_engine
        self.logger = logger or logging.getLogger(self.component_name)

        self.heartbeat_s = int(_cfg(config, "CHECKPOINT_HEARTBEAT_S"))
        self.daily_target = _cfg(config, "OBJ_DAILY_TARGET_PCT")
        self.hourly_target = _cfg(config, "OBJ_HOURLY_TARGET_PCT")
        self.dd_max = _cfg(config, "OBJ_MAX_DRAWDOWN_PCT")
        self.min_edge_bps = _cfg(config, "OBJ_MIN_NET_EDGE_BPS")
        self.telemetry_max_age = _cfg(config, "OBJ_TELEMETRY_MAX_AGE_S")

        self.knob_ranges = {
            "confidence_floor": (
                _cfg(config, "OBJ_CONF_FLOOR_MIN"),
                _cfg(config, "OBJ_CONF_FLOOR_MAX"),
            ),
            "size_multiplier": (
                _cfg(config, "OBJ_SIZE_MULT_MIN"),
                _cfg(config, "OBJ_SIZE_MULT_MAX"),
            ),
            "target_throughput_per_hour": (
                _cfg(config, "OBJ_THRU_MIN"),
                _cfg(config, "OBJ_THRU_MAX"),
            ),
        }

        self.gains = {
            "Kp_conf": _cfg(config, "OBJ_KP_CONF"),
            "Ki_conf": _cfg(config, "OBJ_KI_CONF"),
            "Kp_size": _cfg(config, "OBJ_KP_SIZE"),
            "Kp_thru": _cfg(config, "OBJ_KP_THRU"),
            "Kp_dd": _cfg(config, "OBJ_KP_DD_PENALTY"),
        }

        self.state = ControllerState()
        self.state.last_knobs = {
            "confidence_floor": 0.65,
            "size_multiplier": 1.00,
            "target_throughput_per_hour": 12.0,
        }

        self._task: Optional[asyncio.Task] = None
        self._stopping = False
        self._session_peak_nav: float = 0.0  # intra-session peak; never loaded from metrics

        self.artefact_path = Path(
            artefact_path or os.getenv("OBJ_ARTEFACT_PATH", "objective_controller_state.json")
        )

        # Restore only the knobs from the previous session.
        # consecutive_dd_breaches and integral_pace must NOT carry over — a stale
        # breach count from a previous crash causes an immediate kill-switch on restart
        # even when the current session has no drawdown at all.
        self._restore_knobs_from_artefact()

        self.logger.info(
            "[OFC] Initialised — daily=%.2f%% hourly=%.4f%% dd_max=%.2f%% hb=%ds",
            self.daily_target * 100,
            self.hourly_target * 100,
            self.dd_max * 100,
            self.heartbeat_s,
        )

    # ----------------------------------------------------------------
    # Lifecycle
    # ----------------------------------------------------------------

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._stopping = False
        self._task = asyncio.create_task(self._run_loop(), name="ObjectiveFeedbackLoop")
        self.logger.info("[OFC] Started — heartbeat every %ds", self.heartbeat_s)

    async def stop(self) -> None:
        self._stopping = True
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        self._persist_state()
        self.logger.info("[OFC] Stopped")

    async def _run_loop(self) -> None:
        # Initial small delay so SharedState is warm
        await asyncio.sleep(min(60, self.heartbeat_s))
        while not self._stopping:
            try:
                await self.step()
            except Exception as e:  # pragma: no cover
                self.logger.exception("[OFC] step error: %s", e)
            await asyncio.sleep(self.heartbeat_s)

    # ----------------------------------------------------------------
    # One control step (also callable from tests / scripts)
    # ----------------------------------------------------------------

    async def step(self) -> dict[str, Any]:
        """One full measure → decide → act cycle. Returns a record dict."""
        now = time.time()
        tel = await self._measure()

        if not tel.ok:
            self.logger.warning(
                "[OFC] STARVED — telemetry missing/stale (age=%.0fs, missing=%s) — no-op",
                tel.age_s,
                tel.missing,
            )
            return {"status": "starved", "missing": tel.missing}

        # ---- 1. Errors ------------------------------------------------
        observed_pace_pct_h = self._observed_pace_pct_h(tel)
        pace_error = observed_pace_pct_h - (self.hourly_target * 100)  # % per hour
        dd_error = max(0.0, tel.drawdown_pct - self.dd_max * 100)  # % over limit
        edge_error_bps = self.min_edge_bps - tel.avg_net_profit_bps  # >0 = bad
        # avg_net_profit_bps carries stale EMA data from previous sessions.
        # Suppress edge penalty until we have fresh trades (at least 3h or 5 fills)
        # to avoid false tightening from historical bad data.
        if tel.elapsed_h < 3.0 or tel.trades_in_window < 5:
            edge_error_bps = min(0.0, edge_error_bps)

        # ---- 2. Kill-switch ------------------------------------------
        halted = False
        if dd_error > 0:
            self.state.consecutive_dd_breaches += 1
            if self.state.consecutive_dd_breaches >= 2:
                halted = await self._trip_kill_switch(tel, dd_error)
        else:
            self.state.consecutive_dd_breaches = 0

        # ---- 3. PI update --------------------------------------------
        # When there are zero trades (regime block, cooldown, etc.) and no drawdown,
        # the system is idle by design — not underperforming. Skip PI update to prevent
        # integral windup and slowly decay floor back to min so it doesn't ratchet up.
        # EXCEPTION: if the portfolio is capacity-constrained (many positions or high
        # exposure), the idle signal is a capacity signal, not a quality signal — do NOT
        # decay the floor. Decaying the floor when all slots are taken causes low-quality
        # entries as soon as a slot opens.
        _ss_cap = self.ss
        _positions_count = 0
        _exposure_ratio = 0.0
        if _ss_cap is not None:
            try:
                _positions_count = len(getattr(_ss_cap, "positions", {}) or {})
                _metrics_cap = getattr(_ss_cap, "metrics", {}) or {}
                _nav_cap = await _safe_get(_ss_cap, "get_nav", default=1.0) or 1.0
                _locked_cap = float(_metrics_cap.get("locked_usdt", 0.0) or 0.0)
                _exposure_ratio = _locked_cap / _nav_cap if _nav_cap > 0 else 0.0
            except Exception:
                pass
        _capacity_constrained = _positions_count >= 10 or _exposure_ratio >= 0.45

        no_trades_idle = tel.trades_in_window == 0 and dd_error == 0.0
        if no_trades_idle and _capacity_constrained:
            # Portfolio is full — 0 trades means no slots, not low quality.
            # Hold floor steady so we don't open slots to mediocre signals.
            self.logger.debug(
                "[OFC] idle (0 trades) but capacity-constrained (positions=%d, exposure=%.0f%%) — holding floor steady",
                _positions_count,
                _exposure_ratio * 100,
            )
            d_conf = 0.0
            d_size = 0.0
            d_thru = 0.0
        elif no_trades_idle:
            self.logger.debug(
                "[OFC] idle (0 trades, no DD) — decaying floor toward min, holding other knobs"
            )
            conf_min = self.knob_ranges["confidence_floor"][0]
            current_floor = self.state.last_knobs["confidence_floor"]
            # Gentle decay: move 10% toward floor_min each OFC step so ratchet reverses
            d_conf = (conf_min - current_floor) * 0.10
            d_size = 0.0
            d_thru = 0.0
        else:
            # Anti-windup: only integrate when not at saturation
            self.state.integral_pace = _clamp(self.state.integral_pace + pace_error, -2.0, 2.0)

            d_conf = (
                +self.gains["Kp_conf"] * pace_error * 0.01  # behind→lower floor, ahead→raise
                + self.gains["Ki_conf"] * self.state.integral_pace * 0.01
                + self.gains["Kp_dd"] * (dd_error * 0.01)  # raise floor on DD
                + (edge_error_bps / 1000.0)  # poor edge → raise floor
            )

            d_size = -self.gains["Kp_size"] * pace_error * 0.01 - self.gains["Kp_dd"] * (
                dd_error * 0.01
            )

            d_thru = (
                +self.gains["Kp_thru"] * pace_error * 0.01
                - self.gains["Kp_dd"] * (dd_error * 0.01) * 10.0
            )

        # ---- 4. Apply (clamped) --------------------------------------
        new_knobs = dict(self.state.last_knobs)
        new_knobs["confidence_floor"] = _clamp(
            self.state.last_knobs["confidence_floor"] + d_conf,
            *self.knob_ranges["confidence_floor"],
        )
        new_knobs["size_multiplier"] = _clamp(
            self.state.last_knobs["size_multiplier"] + d_size,
            *self.knob_ranges["size_multiplier"],
        )
        new_knobs["target_throughput_per_hour"] = _clamp(
            self.state.last_knobs["target_throughput_per_hour"] + d_thru,
            *self.knob_ranges["target_throughput_per_hour"],
        )

        # If halted, force conservative knobs
        if halted:
            new_knobs["size_multiplier"] = self.knob_ranges["size_multiplier"][0]
            new_knobs["target_throughput_per_hour"] = self.knob_ranges[
                "target_throughput_per_hour"
            ][0]

        await self._publish(new_knobs, halted=halted)

        record = {
            "ts": now,
            "telemetry": asdict(tel),
            "errors": {
                "pace_pct_per_h": pace_error,
                "drawdown_pct_over_limit": dd_error,
                "net_edge_shortfall_bps": edge_error_bps,
            },
            "knobs_before": self.state.last_knobs,
            "knobs_after": new_knobs,
            "halted": halted,
            "integral_pace": self.state.integral_pace,
        }
        self.state.last_knobs = new_knobs
        self.state.last_action_ts = now
        self.state.history.append(record)
        if len(self.state.history) > 200:
            self.state.history = self.state.history[-200:]
        self._persist_state()

        self.logger.info(
            "[OFC] step pace_err=%+.4f%%/h dd_err=%+.2f%% edge_err=%+.1fbps "
            "→ conf=%.3f size=%.2f thru=%.1f%s",
            pace_error,
            dd_error,
            edge_error_bps,
            new_knobs["confidence_floor"],
            new_knobs["size_multiplier"],
            new_knobs["target_throughput_per_hour"],
            " HALTED" if halted else "",
        )
        return record

    # ----------------------------------------------------------------
    # Measurement
    # ----------------------------------------------------------------

    async def _measure(self) -> Telemetry:
        ss = self.ss
        if ss is None:
            return Telemetry(ok=False, missing=["shared_state"])

        missing = []

        nav = await _safe_get(ss, "get_nav", default=0.0)
        if nav <= 0:
            missing.append("nav")

        metrics = getattr(ss, "metrics", {}) or {}
        realized = float(metrics.get("realized_pnl", 0.0) or 0.0)
        unreal = float(metrics.get("unrealized_pnl", 0.0) or 0.0)

        # Anchor NAV — try PTE first, then ss attr, then current nav
        nav_anchor = 0.0
        if self.pte is not None:
            nav_anchor = float(getattr(self.pte, "_daily_anchor_nav", 0.0) or 0.0)
        if nav_anchor <= 0:
            nav_anchor = float(getattr(ss, "session_anchor_nav", 0.0) or nav)
        if nav_anchor <= 0:
            missing.append("nav_anchor")

        elapsed_h = float(metrics.get("session_elapsed_h", 0.0) or 0.0)
        if elapsed_h <= 0:
            # fall back: derive from PTE start_time
            if self.pte is not None and getattr(self.pte, "_start_time", 0):
                elapsed_h = max((time.time() - self.pte._start_time) / 3600.0, 1e-3)
            else:
                missing.append("elapsed_h")

        # Track peak_nav in-memory so stale all-time highs from prior sessions never
        # contaminate this session. On first call, seed from session anchor (not metrics).
        if self._session_peak_nav <= 0:
            self._session_peak_nav = float(nav_anchor or nav)
        self._session_peak_nav = max(self._session_peak_nav, nav)
        peak_nav = self._session_peak_nav
        dd_pct = 0.0 if peak_nav <= 0 else max(0.0, (peak_nav - nav) / peak_nav * 100.0)

        trades = int(metrics.get("trades_in_window", 0) or 0)
        win_rate = float(metrics.get("win_rate_window", 0.0) or 0.0)
        fee_bps = float(metrics.get("avg_fee_bps", 0.0) or 0.0)
        slip_bps = float(metrics.get("avg_slippage_bps", 0.0) or 0.0)
        net_bps = float(metrics.get("avg_net_profit_bps", 0.0) or 0.0)

        last_update = float(metrics.get("last_update_ts", 0.0) or 0.0)
        age = (time.time() - last_update) if last_update > 0 else 0.0
        if last_update > 0 and age > self.telemetry_max_age:
            missing.append(f"stale_telemetry({age:.0f}s)")

        ok = (not missing) and nav > 0 and nav_anchor > 0
        return Telemetry(
            ok=ok,
            age_s=age,
            nav=nav,
            nav_anchor=nav_anchor,
            realized_pnl=realized,
            unrealized_pnl=unreal,
            drawdown_pct=dd_pct,
            elapsed_h=max(elapsed_h, 1e-3),
            trades_in_window=trades,
            win_rate=win_rate,
            avg_fee_bps=fee_bps,
            avg_slippage_bps=slip_bps,
            avg_net_profit_bps=net_bps,
            missing=missing,
        )

    @staticmethod
    def _observed_pace_pct_h(tel: Telemetry) -> float:
        """Realised NAV growth %/hour relative to anchor."""
        if tel.nav_anchor <= 0 or tel.elapsed_h <= 0:
            return 0.0
        total_pct = (tel.nav - tel.nav_anchor) / tel.nav_anchor * 100.0
        return total_pct / tel.elapsed_h

    # ----------------------------------------------------------------
    # Publish & kill-switch
    # ----------------------------------------------------------------

    async def _publish(self, knobs: dict[str, float], halted: bool) -> None:
        ss = self.ss
        if ss is None:
            return
        # Hot-reload dict consumed by AdaptiveCapitalEngine / MetaController.
        if not hasattr(ss, "runtime_overrides") or ss.runtime_overrides is None:
            try:
                ss.runtime_overrides = {}
            except Exception:
                return
        ss.runtime_overrides.update(
            {
                "CONFIDENCE_FLOOR": knobs["confidence_floor"],
                "SIZE_MULTIPLIER": knobs["size_multiplier"],
                "TARGET_THROUGHPUT_PER_HOUR": knobs["target_throughput_per_hour"],
                "OBJECTIVE_HALTED": bool(halted),
                "_objective_fb_updated": time.time(),
            }
        )

        # Best-effort event emission
        if hasattr(ss, "emit_event"):
            try:
                await ss.emit_event(
                    "ObjectiveFeedback",
                    {
                        "knobs": knobs,
                        "halted": halted,
                    },
                )
            except Exception:
                pass

    async def _trip_kill_switch(self, tel: Telemetry, dd_error: float) -> bool:
        self.logger.error(
            "[OFC] 🚨 KILL-SWITCH — drawdown %.2f%% exceeds limit by %.2f%% "
            "(consecutive=%d). Halting new BUYs.",
            tel.drawdown_pct,
            dd_error,
            self.state.consecutive_dd_breaches,
        )
        ss = self.ss
        if ss is not None:
            try:
                import time as _t

                ss.trading_halted = True
                ss._trading_halted_since = _t.time()  # timestamp for auto-resume
            except Exception:
                pass
            if hasattr(ss, "emit_event"):
                try:
                    await ss.emit_event(
                        "ObjectiveKillSwitch",
                        {
                            "reason": "max_drawdown_breach",
                            "drawdown_pct": tel.drawdown_pct,
                            "limit_pct": self.dd_max * 100,
                        },
                    )
                except Exception:
                    pass
        return True

    # ----------------------------------------------------------------
    # Persistence (so we survive restarts and have an audit trail)
    # ----------------------------------------------------------------

    def _restore_knobs_from_artefact(self) -> None:
        """Load only last_knobs from the artefact — never restore breach counters or integral state."""
        try:
            if self.artefact_path.exists():
                data = json.loads(self.artefact_path.read_text())
                knobs = data.get("last_knobs")
                if isinstance(knobs, dict):
                    self.state.last_knobs = knobs
                    self.logger.info(
                        "[OFC] Restored knobs from artefact: conf_floor=%.2f size_mult=%.2f",
                        float(knobs.get("confidence_floor", 0.65)),
                        float(knobs.get("size_multiplier", 1.0)),
                    )
                # Deliberately NOT restoring: consecutive_dd_breaches, integral_pace, history_tail
                # Those are session-scoped — stale values cause false kill-switches on restart.
        except Exception:
            self.logger.debug("[OFC] _restore_knobs_from_artefact: no artefact or parse error")

    def _persist_state(self) -> None:
        try:
            payload = {
                "last_action_ts": self.state.last_action_ts,
                "integral_pace": self.state.integral_pace,
                "consecutive_dd_breaches": self.state.consecutive_dd_breaches,
                "last_knobs": self.state.last_knobs,
                "history_tail": self.state.history[-20:],
                "set_points": {
                    "daily_target_pct": self.daily_target,
                    "hourly_target_pct": self.hourly_target,
                    "dd_max_pct": self.dd_max,
                },
            }
            self.artefact_path.write_text(json.dumps(payload, indent=2, default=str))
        except Exception:  # pragma: no cover
            self.logger.debug("[OFC] persist_state failed", exc_info=True)


# ------------------------------------------------------------------ #
#  Helpers                                                           #
# ------------------------------------------------------------------ #


def _clamp(x: float, lo: float, hi: float) -> float:
    if math.isnan(x) or math.isinf(x):
        return (lo + hi) / 2
    return max(lo, min(hi, x))


async def _safe_get(obj: Any, attr: str, default: Any = None) -> Any:
    try:
        f = getattr(obj, attr, None)
        if f is None:
            return default
        v = f() if callable(f) else f
        if hasattr(v, "__await__"):
            v = await v
        return v if v is not None else default
    except Exception:
        return default


# Alias for native stack compatibility
NativeObjectiveFeedbackController = ObjectiveFeedbackController
