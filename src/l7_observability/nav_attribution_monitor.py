"""
NavAttributionMonitor (L7)
==========================

Continuously tracks NAV fluctuation and explains *why* the bot is growing or
decaying. Pure read-only observer — never mutates positions, balances, or
issues trades. Writes its verdict to ``shared_state.metrics["nav_trend"]``
so other components (capital_governor, mode_manager, dashboard) can react.

Decomposition (per snapshot):
    ΔNAV = ΔRealized + ΔUnrealized + ΔWalletExt − ΔFees

where:
    ΔRealized   : delta of ss.realized_pnl_total between snapshots
    ΔFees       : sum of fee_quote in trade_history added since last snap
    ΔUnrealized : delta of metrics["unrealized_pnl"]
    ΔWalletExt  : residual = ΔNAV − the three above (deposits, withdrawals,
                  wallet-side balance drift not captured by trade history)

Trend verdict (rolling 5-min window, configurable):
    GROWING  : slope ≥ +epsilon AND realized contribution dominant
    FLAT     : |slope| < epsilon
    DECAYING : slope ≤ −epsilon
    CHURNING : |Δfees| / |ΔNAV| ≥ churn_ratio (fee-eating-PnL pathology)

Single periodic log line:
    [NavAttribution] NAV $103.17→$103.05 (Δ=-0.12, slope=-0.09/min)
        verdict=DECAYING  realized=-0.10 fees=-0.04 unreal=+0.02 ext=0.00
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import Any, Deque, Dict, Optional, Tuple

logger = logging.getLogger("NavAttributionMonitor")


class NavAttributionMonitor:
    """Read-only NAV trend & attribution tracker."""

    def __init__(
        self,
        shared_state: Any,
        config: Any = None,
        *,
        interval_sec: float = 30.0,
        window_sec: float = 300.0,
        flat_epsilon_per_min: float = 0.02,  # $/min (configurable per NAV scale)
        churn_ratio: float = 0.6,            # |fees| / |ΔNAV| > 0.6 → CHURNING
    ) -> None:
        self.ss = shared_state
        self.cfg = config
        self.interval_sec = float(
            getattr(config, "NAV_ATTRIBUTION_INTERVAL_SEC", interval_sec) or interval_sec
        )
        self.window_sec = float(
            getattr(config, "NAV_ATTRIBUTION_WINDOW_SEC", window_sec) or window_sec
        )
        self.flat_epsilon_per_min = float(
            getattr(config, "NAV_TREND_FLAT_EPSILON_PER_MIN", flat_epsilon_per_min)
            or flat_epsilon_per_min
        )
        self.churn_ratio = float(
            getattr(config, "NAV_TREND_CHURN_RATIO", churn_ratio) or churn_ratio
        )
        self.logger = logger

        # Rolling snapshots: (ts, nav, realized_total, fees_total, unrealized)
        self._snapshots: Deque[Tuple[float, float, float, float, float]] = deque()
        self._task: Optional[asyncio.Task] = None
        self._running = False

    # ----------------------------------------------------------------- helpers
    def _read_nav(self) -> float:
        try:
            m = getattr(self.ss, "metrics", {}) or {}
            v = m.get("nav") or m.get("total_equity")
            return float(v or 0.0)
        except Exception:
            return 0.0

    def _read_realized_total(self) -> float:
        # Primary: ss.realized_pnl (running total). Fallback: sum trade_history.
        try:
            v = getattr(self.ss, "realized_pnl", None)
            if v is not None:
                return float(v or 0.0)
        except Exception:
            pass
        try:
            return float(
                sum(
                    float(t.get("realized_delta", 0.0) or 0.0)
                    for t in (getattr(self.ss, "trade_history", []) or [])
                )
            )
        except Exception:
            return 0.0

    def _read_fees_total(self) -> float:
        try:
            return float(
                sum(
                    float(t.get("fee_quote", 0.0) or 0.0)
                    for t in (getattr(self.ss, "trade_history", []) or [])
                )
            )
        except Exception:
            return 0.0

    def _read_unrealized(self) -> float:
        try:
            return float((getattr(self.ss, "metrics", {}) or {}).get("unrealized_pnl", 0.0) or 0.0)
        except Exception:
            return 0.0

    # ----------------------------------------------------------------- core
    def _take_snapshot(self) -> Tuple[float, float, float, float, float]:
        ts = time.time()
        nav = self._read_nav()
        realized = self._read_realized_total()
        fees = self._read_fees_total()
        unreal = self._read_unrealized()
        snap = (ts, nav, realized, fees, unreal)

        self._snapshots.append(snap)
        # Trim to window
        cutoff = ts - max(self.window_sec, self.interval_sec * 2)
        while self._snapshots and self._snapshots[0][0] < cutoff:
            self._snapshots.popleft()
        return snap

    @staticmethod
    def _slope_per_min(snapshots: Deque[Tuple[float, float, float, float, float]]) -> float:
        """Linear slope of NAV vs time, expressed in $/min."""
        if len(snapshots) < 2:
            return 0.0
        first_ts, first_nav = snapshots[0][0], snapshots[0][1]
        last_ts, last_nav = snapshots[-1][0], snapshots[-1][1]
        dt = last_ts - first_ts
        if dt <= 1e-6:
            return 0.0
        return (last_nav - first_nav) / dt * 60.0

    def _attribute(
        self,
        prev: Tuple[float, float, float, float, float],
        curr: Tuple[float, float, float, float, float],
    ) -> Dict[str, float]:
        d_nav = curr[1] - prev[1]
        d_realized = curr[2] - prev[2]
        d_fees = curr[3] - prev[3]
        d_unreal = curr[4] - prev[4]
        # External / unexplained: deposits, withdrawals, wallet drift, mark-px refresh,
        # adoption-engine SELL realized externally, etc.
        d_external = d_nav - d_realized + d_fees - d_unreal
        return {
            "d_nav": d_nav,
            "d_realized": d_realized,
            "d_fees": d_fees,
            "d_unrealized": d_unreal,
            "d_external": d_external,
        }

    def _classify(self, slope_per_min: float, attr: Dict[str, float]) -> str:
        eps = self.flat_epsilon_per_min
        d_nav = attr["d_nav"]
        d_fees = attr["d_fees"]

        # CHURNING: fees eating > churn_ratio of any meaningful NAV swing
        if abs(d_nav) > 1e-3 and (d_fees / max(abs(d_nav), 1e-9)) > self.churn_ratio:
            return "CHURNING"
        if slope_per_min > eps:
            return "GROWING"
        if slope_per_min < -eps:
            return "DECAYING"
        return "FLAT"

    async def _cycle(self) -> None:
        snap = self._take_snapshot()
        if len(self._snapshots) < 2:
            return  # Need at least two points

        prev = self._snapshots[-2]
        attr = self._attribute(prev, snap)
        slope = self._slope_per_min(self._snapshots)
        verdict = self._classify(slope, attr)

        # Publish to shared_state.metrics for downstream consumers
        try:
            metrics = getattr(self.ss, "metrics", None)
            if isinstance(metrics, dict):
                metrics["nav_trend"] = verdict
                metrics["nav_slope_per_min"] = float(slope)
                metrics["nav_attribution_last"] = {
                    "ts": snap[0],
                    "nav": snap[1],
                    "verdict": verdict,
                    "slope_per_min": float(slope),
                    "window_sec": float(self.window_sec),
                    **{k: float(v) for k, v in attr.items()},
                }
        except Exception:
            self.logger.debug("nav_trend metrics publish failed", exc_info=True)

        # Single, structured periodic log line
        prev_nav, curr_nav = prev[1], snap[1]
        # Emit at INFO normally; warn if DECAYING/CHURNING (operator attention)
        line = (
            "[NavAttribution] NAV $%.2f->$%.2f (Δ=%+0.4f, slope=%+0.4f/min, win=%ds) "
            "verdict=%s  realized=%+0.4f fees=%+0.4f unreal=%+0.4f ext=%+0.4f"
            % (
                prev_nav,
                curr_nav,
                attr["d_nav"],
                slope,
                int(self.window_sec),
                verdict,
                attr["d_realized"],
                -attr["d_fees"],   # negate so log shows fees as cost (negative)
                attr["d_unrealized"],
                attr["d_external"],
            )
        )
        if verdict in ("DECAYING", "CHURNING"):
            self.logger.warning(line)
        else:
            self.logger.info(line)

    # ----------------------------------------------------------------- lifecycle
    async def _run_forever(self) -> None:
        self._running = True
        # Initial settle delay so wallet hydration & first trade history load
        await asyncio.sleep(min(15.0, self.interval_sec))
        try:
            while self._running:
                try:
                    await self._cycle()
                except Exception as e:
                    self.logger.warning("[NavAttribution] cycle error: %s", e)
                await asyncio.sleep(self.interval_sec)
        except asyncio.CancelledError:
            self._running = False
            raise

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._task = asyncio.create_task(self._run_forever(), name="ops.nav_attribution")
        self.logger.info(
            "[NavAttribution] started (interval=%.1fs window=%.0fs eps=%.4f$/min churn=%.2f)",
            self.interval_sec, self.window_sec, self.flat_epsilon_per_min, self.churn_ratio,
        )

    async def stop(self) -> None:
        self._running = False
        t = self._task
        self._task = None
        if t:
            t.cancel()
            try:
                await asyncio.wait_for(t, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

    # ----------------------------------------------------------------- public API
    def latest_verdict(self) -> Dict[str, Any]:
        try:
            return dict((getattr(self.ss, "metrics", {}) or {}).get("nav_attribution_last", {}))
        except Exception:
            return {}
