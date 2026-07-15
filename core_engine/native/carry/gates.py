"""
Native carry gates (Phase 4 of the funding-carry native-wiring plan).

Rather than threading bypass flags through NativeArbitrationEngine's 11-gate
sequence (arbitration_engine.py) -- 7 of those 11 gates are either
meaningless or actively wrong for a market-neutral, funding-driven strategy
(gate_4 position-limit and gate_9 pace-limiter don't understand a hedge pair
as one unit; gate_10 no-average-down and gate_11 downtrend-veto are
directionally irrelevant for a delta-neutral position; gate_3 regime,
gate_7 cooldown, and gate_8 performance-tradeability are tuned for
directional ML signals, not funding thresholds) -- carry gets its own small,
self-contained gate stack here, porting carry_paper_trader.py's
already-working safety_checks()/inline entry-close logic almost verbatim.

The ONE check shared with the existing engine is the account-wide kill
switch (shared_state.trading_halted) -- genuinely strategy-agnostic (if the
whole account is halted, e.g. by ObjectiveFeedbackController's kill-switch,
BOTH strategies must respect that). Deliberately NOT reusing
NativeArbitrationEngine.gate_6_risk_manager() wholesale: most of that
method's logic (Fear&Greed pause, NAV-protection FREEZE_BUY/RECOVERY mode,
directional exposure checks) is tightly coupled to the spot strategy's own
capital/sizing model and doesn't apply the same way to a market-neutral
funding harvest -- reusing it wholesale would pull in spot-specific
semantics under a "shared, strategy-agnostic" label that don't actually fit.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Optional

from .state import CarrySharedState


@dataclass
class CarryGateDecision:
    allowed: bool
    reason: str = ""


class CarryGateEngine:
    def __init__(
        self,
        *,
        carry_state: CarrySharedState,
        shared_state: Any = None,
        entry_bps: float = 6.0,
        exit_bps: float = 1.0,
        positive_only: bool = True,
        max_positions: int = 5,
        max_total_usd: float = 5000.0,
        max_hold_h: float = 360.0,
        max_drawdown_pct: float = 5.0,
        liq_buffer_pct: float = 15.0,
        kill_file: str = "logs/native_carry.stop",
    ) -> None:
        self._carry_state = carry_state
        self._shared_state = shared_state
        self.entry = float(entry_bps) / 10000.0
        self.exit = float(exit_bps) / 10000.0
        self.positive_only = bool(positive_only)
        self.max_positions = max(1, int(max_positions))
        self.max_total_usd = max(0.0, float(max_total_usd))
        self.max_hold_h = max(0.0, float(max_hold_h))
        self.max_drawdown_pct = max(0.0, float(max_drawdown_pct))
        self.liq_buffer_pct = max(0.0, float(liq_buffer_pct))
        self.kill_file = kill_file

    # ──────────────────────────────────────────────────────────────────
    # Account-wide check (the one gate shared with the existing engine)
    # ──────────────────────────────────────────────────────────────────
    def _account_halted(self) -> bool:
        return bool(getattr(self._shared_state, "trading_halted", False)) if self._shared_state else False

    def _killed(self) -> bool:
        return os.path.exists(self.kill_file)

    # ──────────────────────────────────────────────────────────────────
    # Drawdown auto-halt — ported from carry_paper_trader.py's
    # _current_drawdown_pct()/safety_checks(), using CarrySharedState's
    # ledger instead of reading the file directly.
    # ──────────────────────────────────────────────────────────────────
    def current_drawdown_pct(self) -> float:
        """Drawdown (%) below the peak of the cumulative net-P&L equity curve."""
        cum = peak = 0.0
        for trade in self._carry_state.read_ledger():
            try:
                cum += float(trade.get("net_pct", 0.0))
            except (TypeError, ValueError):
                continue
            peak = max(peak, cum)
        return max(0.0, peak - cum)

    def check_drawdown_halt(self) -> bool:
        """Returns True (and touches the kill file) if drawdown from peak has
        breached max_drawdown_pct. Mirrors safety_checks()'s auto-halt: this
        blocks new opens on the NEXT evaluate_open() call and lets
        evaluate_close() force-close existing positions, rather than
        unwinding anything synchronously here."""
        dd = self.current_drawdown_pct()
        if dd >= self.max_drawdown_pct and not self._killed():
            os.makedirs(os.path.dirname(self.kill_file) or ".", exist_ok=True)
            open(self.kill_file, "w").close()
            return True
        return self._killed()

    # ──────────────────────────────────────────────────────────────────
    # Liquidation-buffer guard — ported from safety_checks()'s live-mode
    # per-symbol check. Caller (the poller/executor, which has the real
    # futures_position_information() data) supplies mark/liq prices; this
    # class stays free of any exchange-client dependency.
    # ──────────────────────────────────────────────────────────────────
    def is_near_liquidation(self, *, mark_price: float, liquidation_price: float) -> bool:
        if mark_price <= 0 or liquidation_price <= 0:
            return False
        buf_pct = abs(mark_price - liquidation_price) / mark_price * 100.0
        return buf_pct < self.liq_buffer_pct

    # ──────────────────────────────────────────────────────────────────
    # Open / close decisions
    # ──────────────────────────────────────────────────────────────────
    def evaluate_open(self, symbol: str, funding_rate: float) -> CarryGateDecision:
        if self._account_halted():
            return CarryGateDecision(False, "account_trading_halted")
        if self._killed():
            return CarryGateDecision(False, "carry_kill_file_present")
        if self._carry_state.get_open_hedge(symbol) is not None:
            return CarryGateDecision(False, "already_open")
        if self.positive_only and funding_rate <= 0:
            return CarryGateDecision(False, "negative_funding_v1_unsupported")
        if abs(funding_rate) < self.entry:
            return CarryGateDecision(False, "funding_below_entry_threshold")
        if self._carry_state.open_count() >= self.max_positions:
            return CarryGateDecision(False, "max_positions_reached")
        return CarryGateDecision(True, "ok")

    def check_notional_budget(self, candidate_notional_usd: float) -> CarryGateDecision:
        """Separate from evaluate_open() because the candidate's actual
        NAV-aware notional (see carry_paper_trader.py's _resolve_notional
        pattern) is only known after sizing, not at the initial funding-check
        stage."""
        if self._carry_state.locked_capital_usd() + candidate_notional_usd > self.max_total_usd:
            return CarryGateDecision(False, "max_total_notional_exceeded")
        return CarryGateDecision(True, "ok")

    def evaluate_close(self, symbol: str, current_funding: float, *, now: Optional[float] = None) -> CarryGateDecision:
        pos = self._carry_state.get_open_hedge(symbol)
        if pos is None:
            return CarryGateDecision(False, "not_open")
        if self._killed():
            return CarryGateDecision(True, "kill_file_present")
        held_h = pos.held_h(now=now)
        if held_h >= self.max_hold_h:
            return CarryGateDecision(True, "max_hold_exceeded")
        if abs(current_funding) < self.exit:
            return CarryGateDecision(True, "funding_normalized")
        return CarryGateDecision(False, "hold")


__all__ = ["CarryGateEngine", "CarryGateDecision"]
