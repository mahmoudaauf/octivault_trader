"""Native L0: Daily Target Monitor (remediation item #18, 2026-07-14).

Tracks progress toward the "N qualified, net-profitable, compoundable trades
per day" operational target. Confirmed absent from the codebase by the
2026-07-14 audit (docs/audit/daily_target_assessment.md §4) — this is the
first implementation.

Design constraints (binding, per docs/audit/remediation_plan.md item #18):
- READ-ONLY with respect to gates. This module has no method that can change
  ConfFloor, PERSIST_GATE, risk limits, or any trading decision. It only
  records observations and computes a progress classification from them.
  There is deliberately no setter for any gate/threshold anywhere in this file.
- Built on top of net-of-fee realized P&L (executor.py's compute_net_trade_pnl,
  remediation items #3/#4) — NOT gross P&L. A monitor built on gross PnL would
  report a wrong "profitable trades" count, which is worse than no monitor.
- Distinguishes strategy-side ("no qualified signal today") from wiring-side
  ("signal qualified but blocked/failed downstream") causes, so an operator
  reading this doesn't reach for the wrong fix.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ── Progress-state vocabulary (docs/audit/daily_target_assessment.md §3) ───
NOT_STARTED = "NOT_STARTED"
MONITORING = "MONITORING"
OPPORTUNITIES_AVAILABLE = "OPPORTUNITIES_AVAILABLE"
NO_QUALIFIED_OPPORTUNITIES = "NO_QUALIFIED_OPPORTUNITIES"
EXECUTION_BLOCKED = "EXECUTION_BLOCKED"
RISK_FROZEN = "RISK_FROZEN"
DATA_DEGRADED = "DATA_DEGRADED"
PARTIAL_PROGRESS = "PARTIAL_PROGRESS"
TARGET_ACHIEVED = "TARGET_ACHIEVED"
TARGET_MISSED_MARKET_CONDITIONS = "TARGET_MISSED_MARKET_CONDITIONS"
TARGET_MISSED_SYSTEM_FAILURE = "TARGET_MISSED_SYSTEM_FAILURE"
TARGET_MISSED_STRATEGY_PERFORMANCE = "TARGET_MISSED_STRATEGY_PERFORMANCE"


@dataclass
class DailyTargetState:
    date: str = ""  # UTC ISO date, e.g. "2026-07-14"
    target_trades: int = 5

    # Funnel counters (docs/audit/daily_trade_funnel.md stages)
    opportunities_detected: int = 0
    signals_generated: int = 0
    signals_qualified: int = 0       # passed PERSIST_GATE + ConfFloor
    signals_risk_approved: int = 0   # arbitration/decision allowed=True
    signals_rejected: int = 0        # allowed=False, with reason recorded separately
    orders_submitted: int = 0
    entries_filled: int = 0
    duplicate_suppressed: int = 0

    # Outcomes (net-of-fee, from executor.py's canonical compute_net_trade_pnl)
    trades_closed: int = 0
    net_profitable_trades: int = 0
    losing_trades: int = 0
    breakeven_trades: int = 0
    compoundable_trades: int = 0

    gross_pnl_usdt: float = 0.0
    net_pnl_usdt: float = 0.0
    fees_usdt: float = 0.0

    # Rejection-reason tally for the strategy-vs-wiring split (see progress_state()).
    rejection_reasons: dict = field(default_factory=dict)

    last_update_ts: float = 0.0


class NativeDailyTargetMonitor:
    """Tracks same-UTC-day progress toward the daily profitable-trade target.

    Every ``record_*`` method is purely additive bookkeeping — none of them
    can influence a live trading decision. Call sites in implementations.py /
    executor.py treat this exactly like a metrics sink (same pattern as
    ``shared_state.metrics``), never as an input to gating logic.
    """

    def __init__(
        self,
        *,
        target_trades: int = 5,
        state_path: Optional[str] = None,
        history_path: Optional[str] = None,
        now: Optional[datetime] = None,
    ) -> None:
        """``now`` seeds the initial day boundary. It exists because seeding the
        constructor from the wall clock while every ``record_*`` call takes an
        explicit ``now`` makes the class untestable with fixed dates: a monitor
        built "today" but fed day-14 events would spuriously roll over and
        archive an empty real-today row on its first call.
        """
        self.target_trades = max(1, int(target_trades))
        self.state_path = Path(state_path) if state_path else None
        self.history_path = Path(history_path) if history_path else None
        self.state = DailyTargetState(date=self._today(now), target_trades=self.target_trades)
        self._load(now=now)

    # ── Day boundary ────────────────────────────────────────────────────
    @staticmethod
    def _today(now: Optional[datetime] = None) -> str:
        instant = now or datetime.now(timezone.utc)
        if instant.tzinfo is None:
            instant = instant.replace(tzinfo=timezone.utc)
        return instant.astimezone(timezone.utc).date().isoformat()

    def _maybe_rollover(self, now: Optional[datetime] = None) -> None:
        today = self._today(now)
        # Roll FORWARD only (`>`, not `!=`), matching daily_compounding.py's
        # `today > self.state.sizing_date`. `!=` also fired on a backwards date
        # (clock correction, or a fixed-date test), which would archive a
        # partial day and reset — losing real data for no benefit.
        if self.state.date and today > self.state.date:
            self._archive_day(self.state)
            self.state = DailyTargetState(date=today, target_trades=self.target_trades)
            self._persist()
        elif not self.state.date:
            self.state.date = today

    # ── Recording (additive only — no gate access anywhere below) ──────
    def record_opportunity(self, symbol: str, *, now: Optional[datetime] = None) -> None:
        self._maybe_rollover(now)
        self.state.opportunities_detected += 1
        self._touch()

    def record_signal(
        self, symbol: str, *, qualified: bool, now: Optional[datetime] = None
    ) -> None:
        self._maybe_rollover(now)
        self.state.signals_generated += 1
        if qualified:
            self.state.signals_qualified += 1
        self._touch()

    def record_decision(
        self,
        symbol: str,
        *,
        allowed: bool,
        blocked_reason: str = "",
        now: Optional[datetime] = None,
    ) -> None:
        self._maybe_rollover(now)
        if allowed:
            self.state.signals_risk_approved += 1
        else:
            self.state.signals_rejected += 1
            if blocked_reason:
                self.state.rejection_reasons[blocked_reason] = (
                    self.state.rejection_reasons.get(blocked_reason, 0) + 1
                )
        self._touch()

    def record_duplicate_suppressed(self, symbol: str, *, now: Optional[datetime] = None) -> None:
        self._maybe_rollover(now)
        self.state.duplicate_suppressed += 1
        self._touch()

    def record_order_submitted(self, symbol: str, *, now: Optional[datetime] = None) -> None:
        self._maybe_rollover(now)
        self.state.orders_submitted += 1
        self._touch()

    def record_entry_filled(self, symbol: str, *, now: Optional[datetime] = None) -> None:
        self._maybe_rollover(now)
        self.state.entries_filled += 1
        self._touch()

    def record_trade_closed(
        self,
        symbol: str,
        *,
        gross_pnl: float,
        net_pnl: float,
        fees: float = 0.0,
        compoundable: bool = False,
        now: Optional[datetime] = None,
    ) -> None:
        """Record one closed trade's outcome. ``net_pnl`` MUST be net-of-fee
        (executor.py's compute_net_trade_pnl output) — never pass a gross
        figure here, it would silently corrupt the daily target's own
        net-profitable count."""
        self._maybe_rollover(now)
        self.state.trades_closed += 1
        self.state.gross_pnl_usdt += float(gross_pnl or 0.0)
        self.state.net_pnl_usdt += float(net_pnl or 0.0)
        self.state.fees_usdt += float(fees or 0.0)
        if net_pnl > 1e-9:
            self.state.net_profitable_trades += 1
            if compoundable:
                self.state.compoundable_trades += 1
        elif net_pnl < -1e-9:
            self.state.losing_trades += 1
        else:
            self.state.breakeven_trades += 1
        self._touch()
        logger.info(
            "[DailyTargetMonitor] %s closed: net_pnl=%.4f (progress %d/%d net-profitable today)",
            symbol, net_pnl, self.state.net_profitable_trades, self.target_trades,
        )

    def _touch(self) -> None:
        self.state.last_update_ts = time.time()
        self._persist()

    # ── Progress classification ──────────────────────────────────────────
    def progress_state(
        self,
        *,
        risk_frozen: bool = False,
        data_degraded: bool = False,
        market_unfavorable: bool = False,
        system_error_detected: bool = False,
    ) -> str:
        """Classify current same-day progress. The boolean flags are supplied
        by the caller from its own domain knowledge (e.g. NAV protection mode,
        market-data staleness, an exception counter) — this monitor does not
        infer them from trading internals it doesn't own, to avoid guessing.
        """
        s = self.state
        if s.net_profitable_trades >= self.target_trades:
            return TARGET_ACHIEVED
        if system_error_detected:
            return TARGET_MISSED_SYSTEM_FAILURE
        if risk_frozen:
            return RISK_FROZEN
        if data_degraded:
            return DATA_DEGRADED
        if s.trades_closed == 0 and s.signals_generated == 0:
            return NOT_STARTED
        if s.net_profitable_trades > 0:
            return PARTIAL_PROGRESS
        if s.signals_risk_approved > 0 and s.orders_submitted == 0:
            return EXECUTION_BLOCKED
        if s.signals_qualified == 0 and s.signals_generated > 0:
            # Nothing has cleared quality gates — a strategy-side signature
            # (e.g. ConfFloor/PERSIST_GATE holding everything back), not a
            # wiring failure. Do not conflate with EXECUTION_BLOCKED.
            if market_unfavorable:
                return TARGET_MISSED_MARKET_CONDITIONS
            return NO_QUALIFIED_OPPORTUNITIES
        if s.signals_generated > 0:
            return MONITORING
        return TARGET_MISSED_STRATEGY_PERFORMANCE

    def strategy_vs_wiring_summary(self) -> dict[str, object]:
        """Explicit split requested by daily_target_assessment.md §4/§5: separate
        'no qualified signal today' (strategy-side) from 'qualified but blocked
        downstream' (wiring-side), so a reader doesn't fix the wrong layer."""
        s = self.state
        return {
            "strategy_side_blocked": max(0, s.signals_generated - s.signals_qualified),
            "wiring_side_blocked": max(0, s.signals_qualified - s.signals_risk_approved),
            "execution_side_blocked": max(0, s.signals_risk_approved - s.orders_submitted),
            "fill_side_blocked": max(0, s.orders_submitted - s.entries_filled),
            "rejection_reasons": dict(s.rejection_reasons),
        }

    def snapshot(self, **progress_kwargs) -> dict[str, object]:
        out = asdict(self.state)
        out["progress_state"] = self.progress_state(**progress_kwargs)
        out["progress_pct"] = min(
            100.0, 100.0 * self.state.net_profitable_trades / self.target_trades
        )
        out["remaining_qualified_trades_required"] = max(
            0, self.target_trades - self.state.net_profitable_trades
        )
        out["strategy_vs_wiring"] = self.strategy_vs_wiring_summary()
        return out

    # ── Persistence (same pattern as daily_compounding.py) ──────────────
    def _already_archived(self, date: str) -> bool:
        """True if ``date`` already has a row in the history file.

        Guards against double-archiving if the process restarts more than once
        before ``_persist()`` overwrites the stale state file.
        """
        if not date or self.history_path is None or not self.history_path.exists():
            return False
        try:
            with open(self.history_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        if str(json.loads(line).get("date", "")) == date:
                            return True
                    except json.JSONDecodeError:
                        continue
        except OSError:
            return False
        return False

    def _load(self, now: Optional[datetime] = None) -> None:
        if self.state_path is None or not self.state_path.exists():
            return
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            date = str(payload.get("date", "") or "")
            known_fields = {f for f in DailyTargetState.__dataclass_fields__}
            filtered = {k: v for k, v in payload.items() if k in known_fields}
            if date == self._today(now):
                # Only restore same-day state — a stale prior day's counters
                # should never silently carry into a new day's progress.
                self.state = DailyTargetState(**filtered)
            elif date:
                # A PRIOR day's state. This branch used to just DROP it, so any
                # restart across the UTC midnight boundary permanently lost that
                # day from the history file — daily_target_history.jsonl was only
                # complete because the process happened to stay alive overnight.
                # Archive it before moving on (idempotently, in case of repeated
                # restarts), then persist today's fresh state over the stale file.
                prior = DailyTargetState(**filtered)
                if not self._already_archived(prior.date):
                    self._archive_day(prior)
                    logger.info(
                        "[DailyTargetMonitor] archived prior day %s recovered from state "
                        "file on startup (restart crossed UTC midnight)", prior.date,
                    )
                self._persist()
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as e:
            logger.warning("[DailyTargetMonitor] invalid state file ignored: %s (%s)", self.state_path, e)

    def _persist(self) -> None:
        if self.state_path is None:
            return
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
            temp_path.write_text(
                json.dumps(asdict(self.state), indent=2) + "\n", encoding="utf-8",
            )
            os.replace(temp_path, self.state_path)
        except OSError:
            logger.warning("[DailyTargetMonitor] failed to persist state: %s", self.state_path)

    def _archive_day(self, finished: DailyTargetState) -> None:
        if self.history_path is None:
            return
        try:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.history_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(finished)) + "\n")
        except OSError:
            logger.warning("[DailyTargetMonitor] failed to archive day: %s", self.history_path)
