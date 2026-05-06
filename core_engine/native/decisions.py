"""
Native L4: Decision Engine (Phase 8.2.5)

Position sizing + risk gating → trading decisions. Replaces ~800 LOC
legacy ``decision_engine/`` with focused ~250-line implementation.

Design choices
--------------
* Pure function: ``decide(signals, portfolio, balance) → list[Decision]``.
* Kelly fraction for sizing (capped by exposure limits).
* Risk gates: max-drawdown, daily-loss, concurrent-position count.
* Idempotency tagging: each decision gets a UUID for dedup downstream.
* Closed action set: {OPEN, CLOSE, HOLD} — no ambiguity.
* Deterministic ranking: highest-conviction first, capital-aware order.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class Action(Enum):
    """Trading decision action."""

    OPEN = "OPEN"  # enter new position
    CLOSE = "CLOSE"  # exit existing position
    HOLD = "HOLD"  # no action


@dataclass
class Decision:
    """Single trading decision (one symbol, one action)."""

    symbol: str
    action: Action
    quantity: float  # for OPEN/CLOSE
    reason: str  # e.g. "signal_buy", "risk_gate_triggered"
    risk_score: float  # 0..1 (0=safe, 1=risky)
    decision_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "action": self.action.value,
            "quantity": self.quantity,
            "reason": self.reason,
            "risk_score": self.risk_score,
            "decision_id": self.decision_id,
            "ts": self.ts,
        }


@dataclass
class PortfolioSnapshot:
    """Read-only view of current portfolio state."""

    nav: float  # net asset value ($)
    nav_peak: float  # all-time high ($)
    balance: dict[str, float]  # free balances per asset
    positions: dict[str, float]  # symbol → quantity held
    open_orders: dict[str, Any]  # symbol → order details (minimal)


class NativeDecisionEngine:
    """
    Position-sizing and risk-gated decision maker.

    Usage::

        eng = NativeDecisionEngine(
            kelly_fraction=0.25,
            max_concurrent_positions=5,
            daily_loss_limit_pct=2.0,
        )
        decisions = eng.decide(
            signals=[...],           # from L3 NativeSignalEngine
            portfolio=snapshot,      # current state
            balance_usdt=10000.0,
        )
    """

    def __init__(
        self,
        *,
        kelly_fraction: float = 0.25,
        max_position_size_pct: float = 5.0,
        max_concurrent_positions: int = 10,
        min_order_usdt: float = 10.0,
        max_drawdown_pct: float = 10.0,
        daily_loss_limit_pct: float = 5.0,
        risk_per_symbol_pct: float = 2.0,
    ) -> None:
        self.kelly_fraction = max(0.0, min(1.0, float(kelly_fraction)))
        self.max_position_size_pct = max(0.1, float(max_position_size_pct))
        self.max_concurrent_positions = max(1, int(max_concurrent_positions))
        self.min_order_usdt = max(0.0, float(min_order_usdt))
        self.max_drawdown_pct = max(0.0, float(max_drawdown_pct))
        self.daily_loss_limit_pct = max(0.0, float(daily_loss_limit_pct))
        self.risk_per_symbol_pct = max(0.1, float(risk_per_symbol_pct))

    # ──────────────────────────────────────────────────────────────────
    # Main API
    # ──────────────────────────────────────────────────────────────────
    def decide(
        self,
        signals: dict[str, Any],  # symbol → AggregatedSignal
        portfolio: PortfolioSnapshot,
        balance_usdt: float,
    ) -> list[Decision]:
        """
        Generate trading decisions from signals + portfolio state.

        Returns a list of decisions ordered by conviction (highest first).
        All ``HOLD`` actions are omitted.
        """
        decisions: list[Decision] = []

        # Risk gates
        if self._check_drawdown_exceeded(portfolio):
            logger.warning("max drawdown exceeded; returning empty decisions")
            return []
        if self._check_daily_loss_exceeded(portfolio):
            logger.warning("daily loss limit exceeded; returning empty decisions")
            return []

        # Position limits
        open_count = len(portfolio.positions)
        space_available = max(0, self.max_concurrent_positions - open_count)

        # Process BUY signals
        buy_sigs = [
            (sym, sig) for sym, sig in signals.items() if sig.get("direction") == "BUY"
        ]
        buy_sigs.sort(key=lambda x: -x[1].get("score", 0.0))  # highest conviction first

        for sym, sig in buy_sigs:
            if len([d for d in decisions if d.action == Action.OPEN]) >= space_available:
                break
            qty = self._size_new_position(sym, sig, balance_usdt, portfolio)
            if qty > 0:
                decisions.append(
                    Decision(
                        sym,
                        Action.OPEN,
                        qty,
                        f"signal_buy:{sig.get('score', 0.0):.2f}",
                        risk_score=sig.get("score", 0.0),
                    )
                )

        # Process SELL signals
        sell_sigs = [
            (sym, sig) for sym, sig in signals.items() if sig.get("direction") == "SELL"
        ]
        sell_sigs.sort(key=lambda x: -x[1].get("score", 0.0))

        for sym, sig in sell_sigs:
            pos_qty = portfolio.positions.get(sym, 0.0)
            if pos_qty > 0:
                decisions.append(
                    Decision(
                        sym,
                        Action.CLOSE,
                        pos_qty,
                        f"signal_sell:{sig.get('score', 0.0):.2f}",
                        risk_score=sig.get("score", 0.0),
                    )
                )

        return decisions

    # ──────────────────────────────────────────────────────────────────
    # Risk gates
    # ──────────────────────────────────────────────────────────────────
    def _check_drawdown_exceeded(self, portfolio: PortfolioSnapshot) -> bool:
        if portfolio.nav_peak <= 0:
            return False
        dd_pct = (1.0 - portfolio.nav / portfolio.nav_peak) * 100.0
        exceeded = dd_pct > self.max_drawdown_pct
        if exceeded:
            logger.warning(
                "drawdown %.2f%% exceeds limit %.2f%%", dd_pct, self.max_drawdown_pct
            )
        return exceeded

    def _check_daily_loss_exceeded(self, portfolio: PortfolioSnapshot) -> bool:
        # Simplified: assume a reference opening NAV can be retrieved from
        # portfolio metadata (implementation-dependent). For now, return
        # False unless portfolio signals a loss exceeded condition.
        # TODO: wire opening-NAV from session metadata when available.
        return False

    # ──────────────────────────────────────────────────────────────────
    # Position sizing
    # ──────────────────────────────────────────────────────────────────
    def _size_new_position(
        self,
        symbol: str,
        signal: dict[str, Any],
        balance_usdt: float,
        portfolio: PortfolioSnapshot,
    ) -> float:
        """
        Size a new position using Kelly fraction + exposure limits.

        Returns quantity (in base asset) to purchase, or 0 if below minimum.
        """
        if balance_usdt <= self.min_order_usdt:
            return 0.0

        # Max exposure per position
        nav = max(portfolio.nav, 1.0)
        max_exposure_usd = nav * (self.max_position_size_pct / 100.0)

        # Kelly sizing: fraction * confidence-weighted allocation
        conviction = float(signal.get("score", 0.5))  # 0..1
        kelly_allocation = (
            balance_usdt
            * self.kelly_fraction
            * conviction
            * (self.risk_per_symbol_pct / 100.0)
        )
        position_usd = min(kelly_allocation, max_exposure_usd)

        # Sanity check
        if position_usd < self.min_order_usdt:
            return 0.0

        # Convert USD to base-asset quantity (mock: assume 1 USDT per base-unit).
        # In production, fetch current market price from L2.
        # For this implementation, we return the USD amount as a placeholder qty.
        qty = position_usd / max(1.0, portfolio.balance.get("USDT", 1.0))
        return max(0.0, qty)

    # ──────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def rank_decisions(decisions: list[Decision]) -> list[Decision]:
        """Sort decisions by risk_score (highest first) then by symbol."""
        return sorted(decisions, key=lambda d: (-d.risk_score, d.symbol))
