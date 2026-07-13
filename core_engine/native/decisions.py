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
from typing import Any

from .capital_policy import compute_spendable_quote
from .concentration_guard import NativeConcentrationGuard
from .regime_gate import NativeRegimeGate

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
    daily_pnl_pct: float = 0.0  # realized P&L for the current session/day as %
    mode_name: str = ""


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
        min_order_usdt: float = 1.0,
        max_drawdown_pct: float = 10.0,
        daily_loss_limit_pct: float = 2.0,
        risk_per_symbol_pct: float = 20.0,
        min_notional_usdt: float = 10.0,
        quote_reserve_ratio: float = 0.10,
        quote_min_reserve_usdt: float = 0.0,
        max_total_exposure_pct: float = 60.0,
        confidence_floor: float = 0.50,
        max_cluster_exposure_pct: float = 40.0,
        cluster_map: dict[str, str] | None = None,
    ) -> None:
        self.kelly_fraction = max(0.0, min(1.0, float(kelly_fraction)))
        self.max_position_size_pct = max(0.1, float(max_position_size_pct))
        self.max_concurrent_positions = max(1, int(max_concurrent_positions))
        self.min_order_usdt = max(0.0, float(min_order_usdt))
        self.max_drawdown_pct = max(0.0, float(max_drawdown_pct))
        self.daily_loss_limit_pct = max(0.0, float(daily_loss_limit_pct))
        self.risk_per_symbol_pct = max(0.1, float(risk_per_symbol_pct))
        self.min_notional_usdt = max(0.0, float(min_notional_usdt))
        self.quote_reserve_ratio = max(0.0, float(quote_reserve_ratio))
        self.quote_min_reserve_usdt = max(0.0, float(quote_min_reserve_usdt))
        self.max_total_exposure_pct = max(0.0, float(max_total_exposure_pct))
        self.confidence_floor = max(0.0, min(1.0, float(confidence_floor)))
        self.max_cluster_exposure_pct = max(0.0, float(max_cluster_exposure_pct))
        self._concentration_guard = NativeConcentrationGuard(
            max_cluster_exposure_pct=self.max_cluster_exposure_pct,
            cluster_map=cluster_map,
        )
        self._regime_gate = NativeRegimeGate()

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
        spendable_usdt = self._compute_spendable_quote(balance_usdt)
        mode = self._resolve_mode(portfolio)

        # Risk gates
        if self._check_drawdown_exceeded(portfolio):
            logger.warning("max drawdown exceeded; returning empty decisions")
            return []
        if self._check_daily_loss_exceeded(portfolio):
            logger.warning("daily loss limit exceeded; returning empty decisions")
            return []
        exposure_exceeded = self._check_total_exposure_exceeded(portfolio, spendable_usdt)
        if exposure_exceeded:
            logger.warning("total exposure exceeded; skipping new OPEN decisions")

        # Position limits
        open_count = len(portfolio.positions)
        mode_max_positions = min(self.max_concurrent_positions, mode["max_positions"])
        space_available = max(0, mode_max_positions - open_count)

        # Process BUY signals
        buy_sigs = [(sym, sig) for sym, sig in signals.items() if sig.get("direction") == "BUY"]
        ranked_buys = self._rank_buy_signals(buy_sigs, portfolio)

        for sym, sig in ranked_buys:
            if exposure_exceeded:
                break
            if len([d for d in decisions if d.action == Action.OPEN]) >= space_available:
                break
            regime_decision = self._regime_gate.evaluate(sig)
            if not regime_decision.allowed:
                logger.info("regime gate blocked %s: %s", sym, regime_decision.reason)
                continue
            if not self._passes_buy_filters(sig, mode):
                continue
            if not self._passes_regime_adjusted_confidence(sig, mode, regime_decision):
                continue
            qty = self._size_new_position(sym, sig, spendable_usdt, portfolio, mode)
            price_map = self._extract_price_map(portfolio, signals)
            concentration = self._concentration_guard.check_new_position(
                symbol=sym,
                proposed_quote=qty,
                portfolio=portfolio,
                price_map=price_map,
            )
            if not concentration.allowed:
                logger.info(
                    "cluster exposure blocked %s: cluster=%s exposure=%.2f%% limit=%.2f%%",
                    sym,
                    concentration.cluster,
                    concentration.cluster_exposure_pct,
                    self.max_cluster_exposure_pct,
                )
                continue
            logger.info(
                "🎯 Size %s: score=%.2f qty=%.6f (min=%.2f spendable=%.2f mode=%s)",
                sym,
                sig.get("score", 0.0),
                qty,
                self.min_order_usdt,
                spendable_usdt,
                mode["name"],
            )
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
        sell_sigs = [(sym, sig) for sym, sig in signals.items() if sig.get("direction") == "SELL"]
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

        # Capital freeing: Liquidate dust holdings when needed & opportunity is good
        # This allows the system to recycle existing capital instead of waiting for deposits
        # Only sell when: (1) we need capital for strong BUY signals, (2) asset has weak signal
        if spendable_usdt < self.min_order_usdt and len(ranked_buys) > 0:
            best_candidate = None
            best_score = 1.0  # Lower is better (we want weak signals)

            for asset, qty in portfolio.balance.items():
                if asset == "USDT" or qty <= 0:
                    continue

                # Check this asset's signal quality
                asset_symbol = f"{asset}USDT"
                asset_signal = signals.get(asset_symbol, {})
                signal_direction = asset_signal.get("direction", "HOLD")
                signal_score = float(asset_signal.get("score", 0.0))

                # Prioritize SELL signals (best opportunity), then HOLD with low conviction
                if signal_direction == "SELL":
                    # Strong exit signal = best time to free capital
                    priority_score = 0.0 + signal_score
                elif signal_direction == "HOLD":
                    # Neutral = safe to exit
                    priority_score = 0.5 + signal_score
                else:
                    # BUY signal = hold, don't sell
                    priority_score = 2.0

                # Consider asset size (prefer dust over large holdings)
                is_dust = qty < 0.001 or (portfolio.nav > 0 and (qty * 1.0) < portfolio.nav * 0.02)

                # Only consider selling if:
                # 1. Not a strong BUY signal
                # 2. Small holding (dust) OR weak signal
                if priority_score < 2.0 and (is_dust or signal_score < 0.5):
                    if priority_score < best_score:
                        best_candidate = (asset_symbol, asset, qty, signal_direction, signal_score)
                        best_score = priority_score

            # Execute capital freeing if we found a good candidate
            if best_candidate:
                asset_symbol, asset, qty, direction, score = best_candidate
                logger.info(
                    "💰 CAPITAL FREEING: %s qty=%.8f (signal=%s score=%.2f) → frees capital for BUY opportunity",
                    asset_symbol,
                    qty,
                    direction,
                    score,
                )
                decisions.append(
                    Decision(
                        asset_symbol,
                        Action.CLOSE,
                        qty,
                        f"capital_freeing:{direction.lower()}:{score:.2f}",
                        risk_score=0.2,  # Very low risk for capital freeing
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
            logger.warning("drawdown %.2f%% exceeds limit %.2f%%", dd_pct, self.max_drawdown_pct)
        return exceeded

    def _check_daily_loss_exceeded(self, portfolio: PortfolioSnapshot) -> bool:
        daily_pnl_pct = float(getattr(portfolio, "daily_pnl_pct", 0.0) or 0.0)
        # ``daily_pnl_pct`` is signed: losses are negative and gains are positive.
        # Using abs() here incorrectly halted trading after a sufficiently profitable
        # day.  Compare only the loss magnitude so the cap cannot become a profit cap.
        daily_loss_pct = max(0.0, -daily_pnl_pct)
        exceeded = daily_loss_pct > self.daily_loss_limit_pct
        if exceeded:
            logger.warning(
                "daily loss %.2f%% exceeds limit %.2f%%",
                daily_loss_pct,
                self.daily_loss_limit_pct,
            )
        return exceeded

    def _check_total_exposure_exceeded(
        self,
        portfolio: PortfolioSnapshot,
        spendable_usdt: float,
    ) -> bool:
        nav = max(float(portfolio.nav or 0.0), 1.0)
        invested = max(0.0, nav - max(0.0, spendable_usdt))
        exposure_pct = (invested / nav) * 100.0
        exceeded = exposure_pct >= self.max_total_exposure_pct
        if exceeded:
            logger.info(
                "total exposure %.2f%% at/above limit %.2f%%",
                exposure_pct,
                self.max_total_exposure_pct,
            )
        return exceeded

    # ──────────────────────────────────────────────────────────────────
    # Position sizing
    # ──────────────────────────────────────────────────────────────────
    def _size_new_position(
        self,
        symbol: str,
        signal: dict[str, Any],
        balance_usdt: float,
        portfolio: PortfolioSnapshot,
        mode: dict[str, Any],
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
            balance_usdt * self.kelly_fraction * conviction * (self.risk_per_symbol_pct / 100.0)
        )
        position_usd = min(kelly_allocation, max_exposure_usd, float(mode["max_trade_usdt"]))

        # Sanity check
        if position_usd < max(self.min_order_usdt, self.min_notional_usdt):
            return 0.0

        # Return USD allocation directly (actual price conversion happens in executor)
        # This allows capital_allocator and executor to use current market prices
        return max(0.0, position_usd)

    def _compute_spendable_quote(self, free_usdt: float) -> float:
        return compute_spendable_quote(
            free_usdt,
            reserve_ratio=self.quote_reserve_ratio,
            min_reserve=self.quote_min_reserve_usdt,
        )

    def _count_active_tradable_positions(self, snapshot: Any) -> int:
        """Count positions with non-zero qty, excluding BNB dust."""
        positions = getattr(snapshot, "positions", {}) or {}
        count = 0
        for sym, pos in positions.items():
            qty = float(getattr(pos, "qty", 0) or 0)
            if qty > 0 and not (sym == "BNBUSDT" and qty * float(getattr(pos, "entry_price", 0) or 0) < 1.0):
                count += 1
        return count

    def _is_slot_blocking_position(self, symbol: str, snapshot: Any) -> bool:
        """Return True if an existing position should block a new BUY slot for this symbol."""
        positions = getattr(snapshot, "positions", {}) or {}
        pos = positions.get(symbol)
        if pos is None:
            return False
        qty = float(getattr(pos, "qty", 0) or 0)
        return qty > 0

    def _resolve_mode(self, portfolio: PortfolioSnapshot) -> dict[str, Any]:
        nav = float(portfolio.nav or 0.0)
        mode_name = str(getattr(portfolio, "mode_name", "") or "").upper()
        if mode_name == "PAUSED":
            return {
                "name": "PAUSED",
                "max_positions": 0,
                "confidence_floor": 1.0,
                "max_trade_usdt": 0.0,
            }
        if mode_name == "SAFE":
            return {
                "name": "SAFE",
                "max_positions": 1,
                "confidence_floor": max(0.90, self.confidence_floor),
                "max_trade_usdt": max(self.min_notional_usdt, 30.0),
            }
        if mode_name == "PROTECTIVE":
            return {
                "name": "PROTECTIVE",
                "max_positions": 2,
                "confidence_floor": max(0.60, self.confidence_floor),
                "max_trade_usdt": max(self.min_notional_usdt, 50.0),
            }
        if mode_name == "RECOVERY":
            return {
                "name": "RECOVERY",
                "max_positions": 5,
                "confidence_floor": max(0.50, self.confidence_floor),
                "max_trade_usdt": max(self.min_notional_usdt, 50.0),
            }
        if mode_name == "NORMAL":
            return {
                "name": "NORMAL",
                "max_positions": 3,
                "confidence_floor": max(0.45, self.confidence_floor - 0.05),
                "max_trade_usdt": max(self.min_notional_usdt, 150.0),
            }
        if mode_name == "GROWTH":
            return {
                "name": "GROWTH",
                "max_positions": 5,
                "confidence_floor": max(0.40, self.confidence_floor - 0.10),
                "max_trade_usdt": max(self.min_notional_usdt, portfolio_nav_cap(nav=nav)),
            }
        if nav < 100.0:
            return {
                "name": "BOOTSTRAP",
                "max_positions": 3,
                "confidence_floor": max(0.50, self.confidence_floor),
                "max_trade_usdt": max(self.min_notional_usdt, 20.0),
            }
        if nav < 500.0:
            return {
                "name": "RECOVERY",
                "max_positions": 5,
                "confidence_floor": max(0.50, self.confidence_floor),
                "max_trade_usdt": max(self.min_notional_usdt, 50.0),
            }
        if nav < 2000.0:
            return {
                "name": "NORMAL",
                "max_positions": 5,
                "confidence_floor": max(0.45, self.confidence_floor - 0.05),
                "max_trade_usdt": max(self.min_notional_usdt, 150.0),
            }
        return {
            "name": "GROWTH",
            "max_positions": 5,
            "confidence_floor": max(0.40, self.confidence_floor - 0.10),
            "max_trade_usdt": max(self.min_notional_usdt, portfolio_nav_cap(nav=nav)),
        }

    def _passes_buy_filters(self, signal: dict[str, Any], mode: dict[str, Any]) -> bool:
        confidence = float(signal.get("confidence", signal.get("score", 0.0)) or 0.0)
        return confidence >= float(mode["confidence_floor"])

    def _passes_regime_adjusted_confidence(
        self,
        signal: dict[str, Any],
        mode: dict[str, Any],
        regime_decision: Any,
    ) -> bool:
        confidence = float(signal.get("confidence", signal.get("score", 0.0)) or 0.0)
        floor = float(mode["confidence_floor"]) + float(
            getattr(regime_decision, "confidence_floor_bump", 0.0) or 0.0
        )
        return confidence >= min(1.0, floor)

    def _rank_buy_signals(
        self,
        buy_sigs: list[tuple[str, dict[str, Any]]],
        portfolio: PortfolioSnapshot,
    ) -> list[tuple[str, dict[str, Any]]]:
        def _rank_key(item: tuple[str, dict[str, Any]]) -> tuple[float, float, float, str]:
            sym, sig = item
            score = float(sig.get("score", 0.0) or 0.0)
            confidence = float(sig.get("confidence", score) or score)
            liquidity = float(sig.get("liquidity_score", 0.0) or 0.0)
            held_penalty = -1.0 if sym in portfolio.positions else 0.0
            return (
                score + confidence * 0.5 + liquidity * 0.25 + held_penalty,
                confidence,
                score,
                sym,
            )

        return sorted(buy_sigs, key=_rank_key, reverse=True)

    def _extract_price_map(
        self,
        portfolio: PortfolioSnapshot,
        signals: dict[str, Any],
    ) -> dict[str, float]:
        prices: dict[str, float] = {}
        maybe_prices = getattr(portfolio, "prices", None)
        if isinstance(maybe_prices, dict):
            for sym, px in maybe_prices.items():
                try:
                    prices[str(sym).upper()] = float(px or 0.0)
                except Exception:
                    continue
        for sym, sig in signals.items():
            if not isinstance(sig, dict):
                continue
            try:
                px = float(sig.get("price", 0.0) or 0.0)
            except Exception:
                px = 0.0
            if px > 0:
                prices[str(sym).upper()] = px
        return prices

    # ──────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def rank_decisions(decisions: list[Decision]) -> list[Decision]:
        """Sort decisions by risk_score (highest first) then by symbol."""
        return sorted(decisions, key=lambda d: (-d.risk_score, d.symbol))


def portfolio_nav_cap(nav: float) -> float:
    return max(25.0, nav * 0.15)
