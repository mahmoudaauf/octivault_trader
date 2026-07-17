"""
ObjectiveFeedbackController — L2 closed-loop controller that auto-calibrates
runtime knobs to keep the bot tracking the +2%/day NAV objective.

Design contract (see OBJECTIVE_FEEDBACK_PLAN.md):
  • Runs every CHECKPOINT_HEARTBEAT_S seconds (default 900 = 15 min).
  • Reads observed pace / drawdown / throughput / economics from SharedState.
  • Bounded quality/risk control on three knobs:
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
    "OBJ_MAX_DRAWDOWN_PCT": 0.06,  # 6% kill-switch — staggered above NAV FREEZE_BUY (4%)
    "OBJ_MIN_NET_EDGE_BPS": 5.0,  # avg net profit must beat 5 bps
    # Knob ranges  (clamped)
    # Floor minimum of 0.65 is the empirical breakeven threshold — signals below this
    # have shown ~30% win rate which is deeply unprofitable. Never go below 0.65.
    "OBJ_CONF_FLOOR_MIN": 0.65,
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
    # ── Pace-chasing (OPT-IN, all default OFF) ─────────────────────────────
    # These re-enable the pace_error -> knob path that step() otherwise zeroes.
    # Read the long block comment in step()'s "PI update" section before
    # enabling any of them: a pace controller has NO FIXED POINT on a
    # negative-expectancy trade generator (being behind target is *caused by*
    # losing; sizing up enlarges the loss; the error never closes), so these
    # will ramp to their clamp and pin there. That is expected behavior, not a
    # bug. They exist because the operator asked for them with that arithmetic
    # in hand. Containment is the pre-existing knob_ranges clamps:
    # size_multiplier <= OBJ_SIZE_MULT_MAX (1.50) and confidence_floor >=
    # OBJ_CONF_FLOOR_MIN (0.65).
    "OBJ_PACE_SIZE_ENABLED": False,  # allow d_size > 0 when behind target
    "OBJ_PACE_GATE_ENABLED": False,  # allow d_conf < 0 (loosen floor) when behind
    "OBJ_PACE_THRU_ENABLED": False,  # un-hardwire d_thru
    # Telemetry freshness
    "OBJ_TELEMETRY_MAX_AGE_S": 1800,  # 2 × heartbeat
    # Kill-switch auto-resume: minimum time the halt must hold once drawdown
    # has recovered below the limit, before BUYs are allowed again. Prevents
    # rapid flip-flopping right at the threshold; does NOT bypass the
    # requirement that drawdown actually recover first (see step()).
    "OBJ_KILL_SWITCH_RESUME_COOLDOWN_S": 1800,  # 30 min
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


def _cfg_bool(config: Any, key: str) -> bool:
    """Boolean config: attr, then env, then DEFAULTS.

    Booleans need their own reader — _cfg() coerces through float(), so an env
    value of "true"/"yes"/"on" raises ValueError and silently falls back to the
    default, which for an opt-in safety flag is a trap (you'd set it and it
    wouldn't take). Accepts the same tokens as the rest of the codebase.
    """
    if config is not None and hasattr(config, key):
        v = getattr(config, key)
        if v is not None:
            return bool(v)
    env = os.getenv(key)
    if env is not None:
        return env.strip().lower() in ("1", "true", "yes", "on")
    return bool(DEFAULTS[key])


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
        self.kill_switch_resume_cooldown_s = _cfg(config, "OBJ_KILL_SWITCH_RESUME_COOLDOWN_S")

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

        # Opt-in pace-chasing. All default False -> step()'s control law is
        # byte-for-byte its previous behavior unless deliberately enabled.
        self.pace_size_enabled = _cfg_bool(config, "OBJ_PACE_SIZE_ENABLED")
        self.pace_gate_enabled = _cfg_bool(config, "OBJ_PACE_GATE_ENABLED")
        self.pace_thru_enabled = _cfg_bool(config, "OBJ_PACE_THRU_ENABLED")
        if self.pace_size_enabled or self.pace_gate_enabled or self.pace_thru_enabled:
            self.logger.warning(
                "[OFC] PACE-CHASING ENABLED (size=%s gate=%s thru=%s). This re-enables "
                "a control path that was deliberately removed. It cannot converge on a "
                "negative-expectancy generator and will ramp to its clamp (size<=%.2f, "
                "conf_floor>=%.2f) and pin there. Intended and operator-authorised.",
                self.pace_size_enabled, self.pace_gate_enabled, self.pace_thru_enabled,
                self.knob_ranges["size_multiplier"][1],
                self.knob_ranges["confidence_floor"][0],
            )

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
        # Suppress all control actions for first 30 min — pace estimate is
        # meaningless over a tiny elapsed window and causes wild swings.
        if tel.elapsed_h < 0.5:
            self.logger.info(
                "[OFC] warm-up (elapsed=%.1fmin < 30min) — no-op this step",
                tel.elapsed_h * 60,
            )
            return {"status": "warmup", "elapsed_h": tel.elapsed_h}

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
        # `halted` must always reflect the CURRENT shared_state.trading_halted
        # (ground truth), not just "did this exact step freshly trip it" --
        # otherwise a still-breached-but-not-yet-2-consecutive step, or a step
        # that neither trips nor resumes, would under-report an already-active
        # halt and skip the "force conservative knobs" branch below.
        if dd_error > 0:
            self.state.consecutive_dd_breaches += 1
            if self.state.consecutive_dd_breaches >= 2:
                await self._trip_kill_switch(tel, dd_error)
        else:
            self.state.consecutive_dd_breaches = 0
            # Auto-resume (2026-07-14 fix): _trip_kill_switch previously set
            # trading_halted=True with no code anywhere ever reading it back
            # to False -- a tripped kill-switch blocked every BUY permanently
            # until a manual process restart, no matter how much drawdown
            # recovered. Resume only once drawdown is back under the limit
            # (dd_error <= 0, this branch) AND a minimum cooldown has held
            # since the trip, to avoid flip-flopping right at the threshold.
            await self._maybe_resume_kill_switch()
        halted = bool(getattr(self.ss, "trading_halted", False)) if self.ss is not None else False

        # ---- 3. PI update --------------------------------------------
        # BASE LAW (always active, unchanged): de-risk only.
        #
        # Original rationale, preserved verbatim because it is still correct and
        # still governs the default configuration:
        #   "Frequency is an evaluation target, never an entry quota.  Zero trades can
        #   legitimately mean that no setup cleared the quality gates, so idle periods
        #   must not lower confidence, increase size, or shorten cooldowns."
        no_trades_idle = tel.trades_in_window == 0 and dd_error == 0.0
        if no_trades_idle:
            self.logger.debug("[OFC] idle (0 trades, no DD) — holding quality/risk knobs")
            self.state.integral_pace = 0.0
            d_conf = 0.0
            d_size = 0.0
            d_thru = 0.0
        else:
            # Pace remains telemetry only.  Confidence may relax slightly only when
            # observed NET edge is already above its floor; it may never be lowered
            # merely because the system is behind a P&L or throughput target.
            self.state.integral_pace = 0.0
            d_conf = (
                self.gains["Kp_dd"] * (dd_error * 0.01)  # raise floor on DD
                + (edge_error_bps / 1000.0)  # poor edge → raise floor
            )
            # The controller can de-risk, but never sizes up to chase a target.
            d_size = -self.gains["Kp_dd"] * (dd_error * 0.01)
            d_thru = 0.0

        # ---- 3b. PACE-CHASING (opt-in; all flags default False) -------
        # Re-enables the pace_error -> knob path that the base law above zeroes.
        # This deliberately overrides the removal quoted above; the operator
        # asked for it with the following arithmetic explicitly in hand:
        #
        #   A pace controller has NO FIXED POINT on a negative-expectancy trade
        #   generator. Being behind target is *caused by* the -EV strategy
        #   losing. Sizing up a losing bet enlarges the loss, so the error never
        #   closes, so the controller pushes harder — until it pins at its
        #   clamp. Measured: the ML forecaster is -0.27%/trade over 3,140
        #   backtested samples; at 2x size that is -0.54%/trade. +2%/day is
        #   unreachable from a negative edge at ANY size.
        #
        # Containment is the pre-existing knob_ranges clamps applied in step 4
        # (size_multiplier <= 1.50, confidence_floor >= 0.65 — the latter
        # documented as the empirical breakeven; below it win-rate ~30%).
        # Deliberately PURE PROPORTIONAL: integral_pace stays pinned at 0.0, so
        # there is no accumulator to wind up. The per-step deltas are small and
        # accumulate through last_knobs, which the clamps bound.
        pace_applied = {}
        behind = pace_error < 0.0
        if behind and dd_error <= 0.0:
            # dd_error > 0 means the de-risk term is active. Never let pace
            # fight it: de-risking always wins, regardless of pace flags.
            if self.pace_size_enabled:
                _d = -self.gains["Kp_size"] * pace_error * 0.01  # pace_error<0 -> positive
                d_size += _d
                pace_applied["size"] = _d
            if self.pace_gate_enabled:
                _d = self.gains["Kp_conf"] * pace_error * 0.01  # pace_error<0 -> negative
                d_conf += _d
                pace_applied["conf"] = _d
            if self.pace_thru_enabled:
                _d = -self.gains["Kp_thru"] * pace_error * 0.01
                d_thru += _d
                pace_applied["thru"] = _d
        if pace_applied:
            self.logger.info(
                "[OFC:PACE] behind by %.4f%%/h — applying %s (idle=%s). Base knobs: %s",
                -pace_error, pace_applied, no_trades_idle, self.state.last_knobs,
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
            # Empty dict whenever pace-chasing is off (the default) or the
            # system is on/ahead of pace — so any non-empty value in the
            # telemetry record is a real, auditable pace intervention.
            "pace_applied": pace_applied,
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

        # 2026-07-14 fix: trades_in_window is a forever-incrementing counter
        # (only reset on process restart) despite its name -- reading it here
        # meant the "fresh trades" edge-error suppression and the "genuinely
        # idle" detection below both permanently stopped working after the
        # first ~5 trades of a session, regardless of how idle the bot
        # actually was afterward. trades_since_ofc_check is a twin counter
        # incremented at the same write sites, but consumed (reset to 0) by
        # this method every call, giving true "since we last checked" window
        # semantics without changing trades_in_window's meaning for its other
        # reader (orchestrator.py's first_trade_executed flag).
        trades = int(metrics.get("trades_since_ofc_check", 0) or 0)
        metrics["trades_since_ofc_check"] = 0
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

    async def _maybe_resume_kill_switch(self) -> bool:
        """Auto-resume the kill-switch once drawdown has recovered and a
        cooldown has held. Returns True if still halted (either not yet
        eligible to resume, or resume itself failed), False if resumed or
        never halted. Only called from the dd_error<=0 branch of step() --
        i.e. drawdown recovery is already a precondition by construction;
        this method only adds the cooldown gate on top."""
        ss = self.ss
        if ss is None or not bool(getattr(ss, "trading_halted", False)):
            return False
        halted_since = float(getattr(ss, "_trading_halted_since", 0.0) or 0.0)
        if halted_since <= 0.0:
            # Halted with no recorded trip time (e.g. set by something other
            # than _trip_kill_switch) -- do not auto-resume state we don't
            # understand the origin of; require manual intervention.
            return True
        elapsed = time.time() - halted_since
        if elapsed < self.kill_switch_resume_cooldown_s:
            return True
        try:
            ss.trading_halted = False
            ss._trading_halted_since = 0.0
        except Exception:
            return True
        self.logger.warning(
            "[OFC] ✅ KILL-SWITCH auto-resumed — drawdown recovered and held "
            "for %.0fs (cooldown %.0fs). New BUYs allowed again.",
            elapsed, self.kill_switch_resume_cooldown_s,
        )
        if hasattr(ss, "emit_event"):
            try:
                await ss.emit_event(
                    "ObjectiveKillSwitchResumed",
                    {"reason": "drawdown_recovered", "halted_for_s": elapsed},
                )
            except Exception:
                pass
        return False

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
