"""
Compound Growth Engine
======================
Drives the system toward +2%/day compound growth on NAV via:

  1. Hourly NAV checkpoints — measure actual vs target pace
  2. Four-tier kill-switch system — protect capital automatically
  3. 50% profit reinvestment pool — scale position sizes as NAV grows
  4. Fee-aware edge validation — every trade must clear 2.5× round-trip cost
  5. Pace signals — emit BEHIND/ON_PACE/AHEAD to MetaController

Architecture
------------
CompoundGrowthEngine is a singleton instantiated in MASTER Layer 6.
MetaController reads three values from it every trade:
  • is_kill_switch_active()  → block BUY if True
  • get_position_size_mult() → scale position size by reinvestment pool
  • get_pace_status()        → advisory signal (can tighten/loosen filters)
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger("CompoundEngine")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class PaceStatus(str, Enum):
    AHEAD = "AHEAD"  # Exceeding 2%/day pace → maintain parameters
    ON_PACE = "ON_PACE"  # Within ±25% of target → normal operation
    BEHIND = "BEHIND"  # Behind pace → can slightly loosen entry filters
    CRITICAL = "CRITICAL"  # Far behind — kill-switch territory


@dataclass
class Checkpoint:
    ts: float
    nav: float
    target_nav: float  # What NAV should be at this time for +2%/day
    gap_pct: float  # (nav - target_nav) / target_nav
    pace_status: PaceStatus
    trades_since_last: int
    realized_pnl_since_last: float


@dataclass
class KillSwitch:
    reason: str
    triggered_at: float
    resume_at: float  # 0 = permanent until manual reset
    nav_at_trigger: float
    detail: str = ""

    def is_expired(self) -> bool:
        return self.resume_at > 0 and time.time() >= self.resume_at

    def time_remaining_sec(self) -> float:
        if self.resume_at <= 0:
            return float("inf")
        return max(0.0, self.resume_at - time.time())


@dataclass
class ReinvestmentLedger:
    """
    Tracks the 50% reinvestment pool.
    Every time a trade closes with profit, 50% goes into the pool.
    The pool is drawn down as position sizes are scaled up.
    """

    pool_usdt: float = 0.0  # Accumulated reinvestable capital
    total_profits_seen: float = 0.0
    total_reinvested: float = 0.0
    total_withdrawn: float = 0.0  # 50% kept / taken out

    def record_profit(self, profit_usdt: float, reinvest_rate: float = 0.50):
        if profit_usdt <= 0:
            return
        reinvest = profit_usdt * reinvest_rate
        withdraw = profit_usdt - reinvest
        self.pool_usdt += reinvest
        self.total_profits_seen += profit_usdt
        self.total_reinvested += reinvest
        self.total_withdrawn += withdraw
        logger.info(
            "[Compound:Reinvest] profit=+$%.4f reinvest=+$%.4f pool=$%.4f "
            "total_reinvested=$%.4f total_withdrawn=$%.4f",
            profit_usdt,
            reinvest,
            self.pool_usdt,
            self.total_reinvested,
            self.total_withdrawn,
        )

    def draw(self, amount: float) -> float:
        """Draw from pool (e.g. to scale a position). Returns amount drawn."""
        drawn = min(amount, self.pool_usdt)
        self.pool_usdt -= drawn
        return drawn


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------


class CompoundGrowthEngine:
    """
    Singleton engine for +2%/day compound growth.

    Instantiate once in MASTER, inject into MetaController, call:
      engine.on_trade_closed(symbol, pnl_usdt)
      engine.get_position_size_mult(base_size_usdt, current_nav) -> float
      engine.is_kill_switch_active() -> bool
      engine.get_pace_status() -> PaceStatus
    """

    def __init__(self, config: Any = None, shared_state: Any = None):
        self.config = config
        self.shared_state = shared_state
        self._lock = asyncio.Lock()

        # Core target
        self.daily_target_pct = float(self._cfg("COMPOUND_DAILY_TARGET_PCT", 0.02))
        self.reinvest_rate = float(self._cfg("COMPOUND_REINVEST_RATE", 0.50))
        self.checkpoint_sec = float(self._cfg("COMPOUND_CHECKPOINT_SEC", 3600))

        # Risk bounds
        self.max_nav_at_risk_pct = float(self._cfg("COMPOUND_MAX_NAV_AT_RISK_PCT", 0.60))
        self.min_edge_fee_mult = float(self._cfg("COMPOUND_MIN_EDGE_FEE_MULT", 2.5))

        # Kill-switch thresholds
        self.ks_daily_dd_pct = float(self._cfg("KILLSWITCH_DAILY_DD_PCT", 0.05))
        self.ks_session_floor_pct = float(self._cfg("KILLSWITCH_SESSION_FLOOR_PCT", 0.10))
        self.ks_consec_losses = int(self._cfg("KILLSWITCH_CONSEC_LOSSES", 3))
        self.ks_fee_drain_pct = float(self._cfg("KILLSWITCH_FEE_DRAIN_PCT", 0.01))
        self.ks_pause_sec = float(self._cfg("KILLSWITCH_PAUSE_SEC", 1800))

        # State
        self._session_open_nav: Optional[float] = None
        self._daily_open_nav: Optional[float] = None
        self._last_checkpoint_ts: float = 0.0
        self._checkpoints: list[Checkpoint] = []
        self._kill_switches: list[KillSwitch] = []
        self._reinvestment = ReinvestmentLedger()

        # Loss streak tracking
        self._consecutive_losses: int = 0
        self._trades_today: int = 0
        self._pnl_today: float = 0.0
        self._fees_today: float = 0.0

        # Session start
        self._session_start = time.time()
        logger.info(
            "[Compound:Init] target=+%.1f%%/day reinvest=%.0f%% "
            "max_risk=%.0f%% DD_kill=%.1f%% losses_kill=%d",
            self.daily_target_pct * 100,
            self.reinvest_rate * 100,
            self.max_nav_at_risk_pct * 100,
            self.ks_daily_dd_pct * 100,
            self.ks_consec_losses,
        )

    def _cfg(self, key: str, default):
        if self.config is not None:
            return getattr(self.config, key, default)
        return default

    # ------------------------------------------------------------------
    # Public API used by MetaController
    # ------------------------------------------------------------------

    def initialise_nav(self, nav: float):
        """Call once after first NAV read at startup."""
        if self._session_open_nav is None:
            self._session_open_nav = nav
            self._daily_open_nav = nav
            self._last_checkpoint_ts = time.time()
            logger.info("[Compound:Init] Session open NAV: $%.2f", nav)

    def on_trade_closed(self, symbol: str, pnl_usdt: float, fee_usdt: float = 0.0):
        """
        Must be called after every closed position.
        Updates streak, reinvestment pool, daily PnL, and kill-switch checks.
        """
        self._trades_today += 1
        self._pnl_today += pnl_usdt
        self._fees_today += fee_usdt

        # Reinvestment pool
        if pnl_usdt > 0:
            self._reinvestment.record_profit(pnl_usdt, self.reinvest_rate)
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1

        logger.info(
            "[Compound:Trade] %s pnl=%+.4f streak_losses=%d " "pnl_today=%+.4f trades_today=%d",
            symbol,
            pnl_usdt,
            self._consecutive_losses,
            self._pnl_today,
            self._trades_today,
        )

        # Evaluate kill-switches after every trade
        self._evaluate_kill_switches()

    def is_kill_switch_active(self, current_nav: Optional[float] = None) -> bool:
        """Returns True if any active (non-expired) kill-switch is set."""
        # Expire resolved switches
        expired = [ks for ks in self._kill_switches if ks.is_expired()]
        for ks in expired:
            logger.info(
                "[Compound:KillSwitch:CLEARED] reason=%s was_triggered_at=%.0f",
                ks.reason,
                ks.triggered_at,
            )
        self._kill_switches = [ks for ks in self._kill_switches if not ks.is_expired()]

        # Check NAV-based switches that need current NAV
        if current_nav is not None:
            self._evaluate_kill_switches(current_nav)

        active = [ks for ks in self._kill_switches]
        if active:
            for ks in active:
                logger.warning(
                    "[Compound:KillSwitch:ACTIVE] %s — %s  resume_in=%.0fs",
                    ks.reason,
                    ks.detail,
                    ks.time_remaining_sec(),
                )
            return True
        return False

    def get_position_size_mult(self, base_size_usdt: float, current_nav: float) -> float:
        """
        Returns a multiplier for position sizing based on reinvestment pool.

        Logic:
        - Base: 1.0× (no change)
        - If reinvestment pool has capital: can scale up toward 1.5×
        - Hard cap: total position cannot exceed max_nav_at_risk_pct of NAV
        - Kill-switch active: return 0.0 (no new trades)
        """
        if self.is_kill_switch_active(current_nav):
            return 0.0

        # Pool-based scale-up: for every $10 in pool, allow +0.1× up to 1.5×
        pool = self._reinvestment.pool_usdt
        scale_bonus = min(0.5, pool / max(base_size_usdt * 5, 1.0))
        mult = 1.0 + scale_bonus

        # Hard cap: don't exceed max_nav_at_risk
        max_size = current_nav * self.max_nav_at_risk_pct
        if base_size_usdt * mult > max_size:
            mult = max_size / max(base_size_usdt, 1.0)

        mult = max(0.5, min(1.5, mult))  # clamp to [0.5, 1.5]

        if abs(mult - 1.0) > 0.05:
            logger.debug(
                "[Compound:Sizing] base=$%.2f mult=%.2f pool=$%.2f nav=$%.2f",
                base_size_usdt,
                mult,
                pool,
                current_nav,
            )
        return mult

    def get_pace_status(self, current_nav: Optional[float] = None) -> PaceStatus:
        """
        How are we tracking vs +2%/day pace at this point in the session?
        Uses elapsed session time to pro-rate the daily target.
        """
        if self._session_open_nav is None or self._session_open_nav <= 0:
            return PaceStatus.ON_PACE

        elapsed_h = (time.time() - self._session_start) / 3600.0
        if elapsed_h < 0.25:  # first 15 min: no data yet
            return PaceStatus.ON_PACE

        # Pro-rated target: fraction of 24h elapsed × daily target
        day_fraction = min(1.0, elapsed_h / 24.0)
        target_gain = self._session_open_nav * self.daily_target_pct * day_fraction
        target_nav = self._session_open_nav + target_gain

        nav = current_nav
        if nav is None:
            try:
                nav = float(getattr(self.shared_state, "nav", 0) or 0)
            except Exception:
                nav = self._session_open_nav

        if nav <= 0:
            return PaceStatus.ON_PACE

        gap_pct = (nav - target_nav) / self._session_open_nav

        if gap_pct >= 0.005:  # >+0.5% ahead of pace
            return PaceStatus.AHEAD
        elif gap_pct >= -0.005:  # within ±0.5%
            return PaceStatus.ON_PACE
        elif gap_pct >= -0.015:  # 0.5-1.5% behind
            return PaceStatus.BEHIND
        else:  # >1.5% behind
            return PaceStatus.CRITICAL

    def take_checkpoint(self, current_nav: float) -> Checkpoint:
        """Record a NAV checkpoint and log pace status."""
        elapsed_h = (time.time() - self._session_start) / 3600.0
        day_fraction = min(1.0, elapsed_h / 24.0)
        open_nav = self._session_open_nav or current_nav
        target_nav = open_nav * (1.0 + self.daily_target_pct * day_fraction)
        gap_pct = (current_nav - target_nav) / open_nav if open_nav > 0 else 0.0
        pace = self.get_pace_status(current_nav)

        trades_since = self._trades_today - (
            self._checkpoints[-1].trades_since_last if self._checkpoints else 0
        )
        pnl_since = self._pnl_today - sum(c.realized_pnl_since_last for c in self._checkpoints)

        cp = Checkpoint(
            ts=time.time(),
            nav=current_nav,
            target_nav=target_nav,
            gap_pct=gap_pct,
            pace_status=pace,
            trades_since_last=trades_since,
            realized_pnl_since_last=pnl_since,
        )
        self._checkpoints.append(cp)
        self._last_checkpoint_ts = time.time()

        status_emoji = {"AHEAD": "🟢", "ON_PACE": "✅", "BEHIND": "🟡", "CRITICAL": "🔴"}
        logger.info(
            "[Compound:Checkpoint] %s elapsed=%.1fh  NAV=$%.2f  target=$%.2f  "
            "gap=%+.2f%%  trades=%d  pnl=%+.4f  pool=$%.4f  losses_streak=%d",
            status_emoji.get(pace.value, ""),
            elapsed_h,
            current_nav,
            target_nav,
            gap_pct * 100,
            self._trades_today,
            self._pnl_today,
            self._reinvestment.pool_usdt,
            self._consecutive_losses,
        )
        return cp

    def validate_trade_edge(self, expected_move_pct: float, fee_rt_pct: float = 0.002) -> bool:
        """
        Returns False if expected move doesn't clear the minimum fee-adjusted edge.
        Prevents entering trades where fees eat more than the expected profit.
        """
        min_required = fee_rt_pct * self.min_edge_fee_mult
        if expected_move_pct < min_required:
            logger.debug(
                "[Compound:EdgeCheck] REJECT expected=%.3f%% < min=%.3f%% (%.1f× fees)",
                expected_move_pct * 100,
                min_required * 100,
                self.min_edge_fee_mult,
            )
            return False
        return True

    def get_summary(self) -> dict[str, Any]:
        """Return a dict suitable for dashboard / logging."""
        pace = self.get_pace_status()
        open_nav = self._session_open_nav or 0.0
        return {
            "session_open_nav": open_nav,
            "daily_target_pct": self.daily_target_pct,
            "reinvest_rate": self.reinvest_rate,
            "reinvest_pool_usdt": self._reinvestment.pool_usdt,
            "total_reinvested": self._reinvestment.total_reinvested,
            "total_withdrawn": self._reinvestment.total_withdrawn,
            "pnl_today_usdt": self._pnl_today,
            "trades_today": self._trades_today,
            "fees_today_usdt": self._fees_today,
            "consecutive_losses": self._consecutive_losses,
            "kill_switches_active": [ks.reason for ks in self._kill_switches],
            "pace_status": pace.value,
            "checkpoints": len(self._checkpoints),
        }

    # ------------------------------------------------------------------
    # Kill-switch evaluation
    # ------------------------------------------------------------------

    def _evaluate_kill_switches(self, current_nav: Optional[float] = None):
        """Check all kill-switch conditions and arm any that are triggered."""
        if self._session_open_nav is None:
            return

        nav = current_nav
        if nav is None:
            try:
                nav = float(getattr(self.shared_state, "nav", 0) or 0)
            except Exception:
                nav = None

        open_nav = self._session_open_nav

        # 1. Daily drawdown: -5% from session open
        if nav is not None and nav > 0:
            dd_pct = (nav - open_nav) / open_nav
            if dd_pct <= -self.ks_daily_dd_pct:
                self._arm_kill_switch(
                    reason="DAILY_DRAWDOWN",
                    pause_sec=self.ks_pause_sec,
                    nav=nav,
                    detail=f"NAV dropped {dd_pct*100:.1f}% from open ${open_nav:.2f} → ${nav:.2f}",
                )

        # 2. Session floor: -10% from open (permanent until manual reset)
        if nav is not None and nav > 0:
            floor_pct = (nav - open_nav) / open_nav
            if floor_pct <= -self.ks_session_floor_pct:
                self._arm_kill_switch(
                    reason="SESSION_FLOOR",
                    pause_sec=0,  # permanent
                    nav=nav,
                    detail=f"NAV hit session floor {floor_pct*100:.1f}% — manual restart required",
                )

        # 3. Consecutive loss streak
        if self._consecutive_losses >= self.ks_consec_losses:
            self._arm_kill_switch(
                reason="LOSS_STREAK",
                pause_sec=self.ks_pause_sec,
                nav=nav or open_nav,
                detail=f"{self._consecutive_losses} consecutive losses — cooling off",
            )

        # 4. Fee drain: fees > 1% of starting NAV today
        if open_nav > 0 and self._fees_today / open_nav >= self.ks_fee_drain_pct:
            self._arm_kill_switch(
                reason="FEE_DRAIN",
                pause_sec=self.ks_pause_sec / 2,  # shorter pause
                nav=nav or open_nav,
                detail=f"Fees today ${self._fees_today:.4f} = "
                f"{self._fees_today/open_nav*100:.2f}% of NAV — reduce frequency",
            )

    def _arm_kill_switch(self, reason: str, pause_sec: float, nav: float, detail: str):
        """Arm a kill-switch if one with this reason isn't already active."""
        existing = [ks for ks in self._kill_switches if ks.reason == reason]
        if existing:
            return  # already armed

        ks = KillSwitch(
            reason=reason,
            triggered_at=time.time(),
            resume_at=time.time() + pause_sec if pause_sec > 0 else 0.0,
            nav_at_trigger=nav,
            detail=detail,
        )
        self._kill_switches.append(ks)
        emoji = "🛑" if pause_sec == 0 else "⚠️"
        logger.warning(
            "[Compound:KillSwitch:ARMED] %s %s — %s  " "pause=%.0fs (resume=%s)",
            emoji,
            reason,
            detail,
            pause_sec,
            "NEVER (manual reset)" if pause_sec == 0 else f"{pause_sec/60:.0f}m",
        )

    def reset_loss_streak(self):
        """Call after a successful trade to clear loss-streak kill-switch."""
        if self._consecutive_losses > 0:
            self._consecutive_losses = 0
        # Remove LOSS_STREAK kill-switch (will be removed on next check too)
        self._kill_switches = [ks for ks in self._kill_switches if ks.reason != "LOSS_STREAK"]

    def manual_reset(self, reason: str = "ALL"):
        """Manually clear kill-switches (e.g. after reviewing situation)."""
        before = len(self._kill_switches)
        if reason == "ALL":
            self._kill_switches.clear()
        else:
            self._kill_switches = [ks for ks in self._kill_switches if ks.reason != reason]
        logger.warning(
            "[Compound:ManualReset] Cleared %d kill-switch(es) (reason=%s)",
            before - len(self._kill_switches),
            reason,
        )

    # ------------------------------------------------------------------
    # Async background loop (hourly checkpoints)
    # ------------------------------------------------------------------

    async def run_checkpoint_loop(self):
        """Background task: take hourly checkpoints and log pace."""
        logger.info("[Compound:CheckpointLoop] Started (interval=%.0fs)", self.checkpoint_sec)
        while True:
            try:
                await asyncio.sleep(self.checkpoint_sec)
                nav = 0.0
                try:
                    nav = float(getattr(self.shared_state, "nav", 0) or 0)
                    if nav <= 0:
                        balances = getattr(self.shared_state, "balances", {}) or {}
                        for asset, data in balances.items():
                            if str(asset).upper() == "USDT":
                                nav += float((data or {}).get("free", 0) or 0)
                                nav += float((data or {}).get("locked", 0) or 0)
                except Exception as e:
                    logger.debug("[Compound:CheckpointLoop] NAV read error: %s", e)

                if nav > 0:
                    if self._session_open_nav is None:
                        self.initialise_nav(nav)
                    cp = self.take_checkpoint(nav)
                    if cp.pace_status == PaceStatus.CRITICAL:
                        logger.warning(
                            "[Compound:CheckpointLoop] 🔴 CRITICAL pace — "
                            "%.2f%% behind target. Reviewing kill-switches.",
                            abs(cp.gap_pct) * 100,
                        )
            except asyncio.CancelledError:
                logger.info("[Compound:CheckpointLoop] Stopped.")
                break
            except Exception as e:
                logger.error("[Compound:CheckpointLoop] Error: %s", e)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_engine: Optional[CompoundGrowthEngine] = None


def get_engine(config=None, shared_state=None) -> CompoundGrowthEngine:
    """Return the module-level singleton (create on first call)."""
    global _engine
    if _engine is None:
        _engine = CompoundGrowthEngine(config=config, shared_state=shared_state)
    return _engine
