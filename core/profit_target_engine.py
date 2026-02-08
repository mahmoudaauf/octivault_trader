"""
ProfitTargetEngine — daily target tracking, per-cycle risk cap, and compounding throttle.

Wired into SharedState via `check_global_compliance` (called as a profit guard).
Constructed by AppContext with: config, logger, app, shared_state.

Config knobs (all overridable via .env):
  PROFIT_TARGET_DAILY_PCT          – daily NAV target (default 2%)
  PROFIT_TARGET_MAX_RISK_PER_CYCLE – max risk as fraction of NAV per eval cycle (default 0.5%)
  PROFIT_TARGET_COMPOUND_THROTTLE  – fraction of excess profit reinvested (default 50%)
  PROFIT_TARGET_BASE_USD_PER_HOUR  – hard USD/h floor (0 = use ratio-based)
  PROFIT_TARGET_GRACE_MINUTES      – startup grace period where guard is open
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional


class ProfitTargetEngine:
    """Lightweight profit-target guard & compounding throttle."""

    component_name = "ProfitTargetEngine"

    def __init__(
        self,
        config: Any = None,
        logger: Optional[logging.Logger] = None,
        app: Any = None,
        shared_state: Any = None,
    ):
        self.config = config
        self.shared_state = shared_state
        self.app = app
        self.logger = logger or logging.getLogger(self.component_name)

        # --- Config knobs ---
        self.daily_target_pct = float(
            getattr(config, "PROFIT_TARGET_DAILY_PCT", 0.02) or 0.02
        )
        self.max_risk_per_cycle = float(
            getattr(config, "PROFIT_TARGET_MAX_RISK_PER_CYCLE", 0.005) or 0.005
        )
        self.compound_throttle = float(
            getattr(config, "PROFIT_TARGET_COMPOUND_THROTTLE", 0.5) or 0.5
        )
        self.base_usd_per_hour = float(
            getattr(config, "PROFIT_TARGET_BASE_USD_PER_HOUR", 0.0) or 0.0
        )
        self.grace_minutes = float(
            getattr(config, "PROFIT_TARGET_GRACE_MINUTES", 30.0) or 30.0
        )

        # Internal bookkeeping
        self._start_time = time.time()
        self._daily_realized: float = 0.0
        self._daily_anchor_nav: float = 0.0
        self._day_start_ts: float = 0.0
        self._last_check_ts: float = 0.0

        self.logger.info(
            "[PTE] Initialized — daily_target=%.2f%%, max_risk/cycle=%.2f%%, "
            "compound_throttle=%.0f%%, grace=%dm",
            self.daily_target_pct * 100,
            self.max_risk_per_cycle * 100,
            self.compound_throttle * 100,
            int(self.grace_minutes),
        )

    # ------------------------------------------------------------------
    # Public API — wired as SharedState profit guard
    # ------------------------------------------------------------------

    async def check_global_compliance(self, context: Dict[str, Any]) -> bool:
        """
        Called by SharedState.profit_target_ok() on every BUY attempt.

        Returns True  → trade allowed
        Returns False → trade blocked (daily target already met or risk cap hit)

        During the startup grace period, always returns True (fail-open).
        """
        try:
            now = time.time()
            elapsed_min = (now - self._start_time) / 60.0

            # Grace period: always allow trading during startup warm-up
            if elapsed_min < self.grace_minutes:
                return True

            # Refresh daily anchor if day rolled
            self._maybe_roll_day(now)

            nav = await self._get_nav()
            if nav <= 0:
                return True  # fail open — cannot compute targets without NAV

            # 1. Daily target check
            daily_target_usd = nav * self.daily_target_pct
            realized_today = await self._get_realized_today()

            if realized_today >= daily_target_usd:
                self.logger.info(
                    "[PTE] Daily target MET (realized=$%.2f >= target=$%.2f). "
                    "Blocking new BUYs.",
                    realized_today,
                    daily_target_usd,
                )
                return False

            # 2. Per-cycle risk cap
            max_risk_usd = nav * self.max_risk_per_cycle
            unrealized = await self._get_unrealized()
            if unrealized < -max_risk_usd:
                self.logger.info(
                    "[PTE] Cycle risk cap breached (unrealized=$%.2f, cap=$-%.2f). "
                    "Blocking new BUYs.",
                    unrealized,
                    max_risk_usd,
                )
                return False

            # 3. Compounding throttle — if realized today exceeds target,
            #    only allow `compound_throttle` fraction of excess to be reinvested.
            #    (This is informational — the actual quote scaling happens in
            #    CapitalAllocator; here we just don't block.)
            self._last_check_ts = now
            return True

        except Exception as e:
            self.logger.warning("[PTE] check_global_compliance error: %s — fail open", e)
            return True  # fail open

    # ------------------------------------------------------------------
    # Compounding helper — called by CapitalAllocator for quote sizing
    # ------------------------------------------------------------------

    def get_compounding_factor(self) -> float:
        """
        Returns a multiplier [0.0, 1.0] for how much of excess profit
        should be reinvested this cycle.

        1.0 = reinvest everything (target not yet met)
        compound_throttle = partial reinvestment (target exceeded)
        """
        try:
            nav = self._daily_anchor_nav
            if nav <= 0:
                return 1.0
            target = nav * self.daily_target_pct
            if self._daily_realized >= target and target > 0:
                return self.compound_throttle
            return 1.0
        except Exception:
            return 1.0

    def get_max_risk_quote(self, nav: float) -> float:
        """Max USDT a single cycle should risk (position sizing cap)."""
        return nav * self.max_risk_per_cycle if nav > 0 else 0.0

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _maybe_roll_day(self, now: float) -> None:
        """Reset daily counters if calendar day changed (UTC)."""
        import datetime as _dt

        today_start = _dt.datetime.utcnow().replace(
            hour=0, minute=0, second=0, microsecond=0
        ).timestamp()
        if self._day_start_ts < today_start:
            self._daily_realized = 0.0
            self._daily_anchor_nav = 0.0
            self._day_start_ts = today_start
            self.logger.info("[PTE] Day rolled — counters reset.")

    async def _get_nav(self) -> float:
        try:
            if hasattr(self.shared_state, "get_nav"):
                val = self.shared_state.get_nav()
                if hasattr(val, "__await__"):
                    val = await val
                nav = float(val or 0.0)
                if nav > 0 and self._daily_anchor_nav <= 0:
                    self._daily_anchor_nav = nav
                return nav
        except Exception:
            pass
        return 0.0

    async def _get_realized_today(self) -> float:
        try:
            metrics = getattr(self.shared_state, "metrics", {}) or {}
            return float(metrics.get("realized_pnl", 0.0) or 0.0)
        except Exception:
            return 0.0

    async def _get_unrealized(self) -> float:
        try:
            metrics = getattr(self.shared_state, "metrics", {}) or {}
            return float(metrics.get("unrealized_pnl", 0.0) or 0.0)
        except Exception:
            return 0.0
