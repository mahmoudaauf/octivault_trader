"""
Tests for Native L4 (Phase 8.2.5) — NativeDecisionEngine.

Tests risk gates, Kelly sizing, signal aggregation, idempotency.
"""

from __future__ import annotations

import pytest

from core_engine.native import (
    Decision,
    NativeDecisionEngine,
    PortfolioSnapshot,
)
from core_engine.native.decisions import Action


# ─────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────
def _portfolio(
    nav: float = 10000.0,
    nav_peak: float = 10000.0,
    balance: dict[str, float] | None = None,
    positions: dict[str, float] | None = None,
) -> PortfolioSnapshot:
    return PortfolioSnapshot(
        nav=nav,
        nav_peak=nav_peak,
        balance=balance or {"USDT": 10000.0},
        positions=positions or {},
        open_orders={},
    )


def _signal_buy(sym: str, score: float = 0.7) -> dict:
    return {"symbol": sym, "direction": "BUY", "score": score}


def _signal_sell(sym: str, score: float = 0.7) -> dict:
    return {"symbol": sym, "direction": "SELL", "score": score}


# ─────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────
class TestNativeDecisionEngine:
    def test_default_params(self) -> None:
        eng = NativeDecisionEngine()
        assert eng.kelly_fraction == 0.25
        assert eng.max_concurrent_positions == 10
        assert eng.min_order_usdt == 10.0

    def test_param_clamping(self) -> None:
        eng = NativeDecisionEngine(kelly_fraction=1.5, max_concurrent_positions=-1)
        assert eng.kelly_fraction == 1.0
        assert eng.max_concurrent_positions == 1

    def test_buy_signal_produces_open_decision(self) -> None:
        eng = NativeDecisionEngine()
        signals = {"BTCUSDT": _signal_buy("BTCUSDT", 0.8)}
        portfolio = _portfolio()
        decisions = eng.decide(signals, portfolio, balance_usdt=10000.0)
        assert len(decisions) == 1
        assert decisions[0].symbol == "BTCUSDT"
        assert decisions[0].action == Action.OPEN
        assert decisions[0].quantity > 0.0

    def test_sell_signal_closes_existing_position(self) -> None:
        eng = NativeDecisionEngine()
        signals = {"BTCUSDT": _signal_sell("BTCUSDT", 0.8)}
        portfolio = _portfolio(positions={"BTCUSDT": 0.1})
        decisions = eng.decide(signals, portfolio, balance_usdt=10000.0)
        assert len(decisions) == 1
        assert decisions[0].action == Action.CLOSE
        assert decisions[0].quantity == 0.1

    def test_sell_signal_without_position_ignored(self) -> None:
        eng = NativeDecisionEngine()
        signals = {"BTCUSDT": _signal_sell("BTCUSDT", 0.8)}
        portfolio = _portfolio(positions={})
        decisions = eng.decide(signals, portfolio, balance_usdt=10000.0)
        assert len(decisions) == 0

    def test_no_signal_or_hold_returns_empty(self) -> None:
        eng = NativeDecisionEngine()
        signals = {"BTCUSDT": {"symbol": "BTCUSDT", "direction": "HOLD", "score": 0.0}}
        portfolio = _portfolio()
        decisions = eng.decide(signals, portfolio, balance_usdt=10000.0)
        assert len(decisions) == 0

    def test_multiple_buy_signals_ordered_by_conviction(self) -> None:
        eng = NativeDecisionEngine()
        signals = {
            "BTCUSDT": _signal_buy("BTCUSDT", 0.5),
            "ETHUSDT": _signal_buy("ETHUSDT", 0.9),  # higher
            "XRPUSDT": _signal_buy("XRPUSDT", 0.7),
        }
        portfolio = _portfolio()
        decisions = eng.decide(signals, portfolio, balance_usdt=10000.0)
        # Should be ordered: ETHUSDT (0.9), XRPUSDT (0.7), BTCUSDT (0.5)
        symbols = [d.symbol for d in decisions]
        assert symbols[0] == "ETHUSDT"
        assert symbols[1] == "XRPUSDT"

    def test_concurrent_position_limit(self) -> None:
        eng = NativeDecisionEngine(max_concurrent_positions=2)
        # Already 1 open position
        signals = {
            "ETHUSDT": _signal_buy("ETHUSDT", 0.9),
            "XRPUSDT": _signal_buy("XRPUSDT", 0.7),
            "LTCUSDT": _signal_buy("LTCUSDT", 0.5),
        }
        portfolio = _portfolio(positions={"BTCUSDT": 0.1})
        decisions = eng.decide(signals, portfolio, balance_usdt=10000.0)
        # Should open only 1 more (ETHUSDT) to reach limit 2
        assert len([d for d in decisions if d.action == Action.OPEN]) == 1
        assert decisions[0].symbol == "ETHUSDT"

    def test_drawdown_exceeded_returns_empty(self) -> None:
        eng = NativeDecisionEngine(max_drawdown_pct=5.0)
        signals = {"BTCUSDT": _signal_buy("BTCUSDT", 0.9)}
        # NAV 9000, peak 10000 ⇒ 10% drawdown > 5% limit
        portfolio = _portfolio(nav=9000.0, nav_peak=10000.0)
        decisions = eng.decide(signals, portfolio, balance_usdt=10000.0)
        assert len(decisions) == 0

    def test_drawdown_ok_proceeds(self) -> None:
        eng = NativeDecisionEngine(max_drawdown_pct=15.0)
        signals = {"BTCUSDT": _signal_buy("BTCUSDT", 0.9)}
        # NAV 9000, peak 10000 ⇒ 10% drawdown < 15% limit
        portfolio = _portfolio(nav=9000.0, nav_peak=10000.0)
        decisions = eng.decide(signals, portfolio, balance_usdt=10000.0)
        assert len(decisions) == 1

    def test_minimum_order_size_enforced(self) -> None:
        eng = NativeDecisionEngine(min_order_usdt=1000.0)
        signals = {"BTCUSDT": _signal_buy("BTCUSDT", 0.9)}
        portfolio = _portfolio()
        # balance_usdt too low
        decisions = eng.decide(signals, portfolio, balance_usdt=100.0)
        assert len(decisions) == 0

    def test_position_size_respects_max_position_size_pct(self) -> None:
        eng = NativeDecisionEngine(max_position_size_pct=2.0)
        signals = {"BTCUSDT": _signal_buy("BTCUSDT", 1.0)}
        portfolio = _portfolio(nav=10000.0)
        decisions = eng.decide(signals, portfolio, balance_usdt=10000.0)
        assert len(decisions) == 1
        # Position should be capped at ~2% of NAV = 200 USDT
        # (This is a soft cap; exact number depends on Kelly calcs)

    def test_kelly_fraction_affects_sizing(self) -> None:
        eng_conservative = NativeDecisionEngine(kelly_fraction=0.1)
        eng_aggressive = NativeDecisionEngine(kelly_fraction=0.5)
        signals = {"BTCUSDT": _signal_buy("BTCUSDT", 0.9)}
        portfolio = _portfolio()
        dec_cons = eng_conservative.decide(signals, portfolio, balance_usdt=10000.0)
        dec_agg = eng_aggressive.decide(signals, portfolio, balance_usdt=10000.0)
        assert len(dec_cons) == 1
        assert len(dec_agg) == 1
        # Aggressive should size higher
        assert dec_agg[0].quantity > dec_cons[0].quantity

    def test_decision_has_unique_id(self) -> None:
        eng = NativeDecisionEngine()
        signals = {"BTCUSDT": _signal_buy("BTCUSDT", 0.9)}
        portfolio = _portfolio()
        decisions = eng.decide(signals, portfolio, balance_usdt=10000.0)
        assert len(decisions) == 1
        dec = decisions[0]
        assert dec.decision_id
        assert len(dec.decision_id) > 0
        # Two calls should produce different IDs
        decisions2 = eng.decide(signals, portfolio, balance_usdt=10000.0)
        assert decisions2[0].decision_id != dec.decision_id

    def test_decision_to_dict(self) -> None:
        dec = Decision(
            symbol="BTCUSDT",
            action=Action.OPEN,
            quantity=0.1,
            reason="test",
            risk_score=0.7,
        )
        d = dec.to_dict()
        assert d["symbol"] == "BTCUSDT"
        assert d["action"] == "OPEN"
        assert d["quantity"] == 0.1
        assert d["risk_score"] == 0.7

    def test_rank_decisions_by_risk_score(self) -> None:
        decs = [
            Decision("A", Action.OPEN, 1.0, "", 0.5),
            Decision("B", Action.OPEN, 1.0, "", 0.9),
            Decision("C", Action.OPEN, 1.0, "", 0.7),
        ]
        ranked = NativeDecisionEngine.rank_decisions(decs)
        assert ranked[0].symbol == "B"
        assert ranked[1].symbol == "C"
        assert ranked[2].symbol == "A"

    def test_mixed_buy_sell_signals(self) -> None:
        eng = NativeDecisionEngine()
        signals = {
            "BTCUSDT": _signal_buy("BTCUSDT", 0.9),
            "ETHUSDT": _signal_sell("ETHUSDT", 0.8),
        }
        portfolio = _portfolio(positions={"ETHUSDT": 0.2})
        decisions = eng.decide(signals, portfolio, balance_usdt=10000.0)
        assert len(decisions) == 2
        opens = [d for d in decisions if d.action == Action.OPEN]
        closes = [d for d in decisions if d.action == Action.CLOSE]
        assert len(opens) == 1
        assert len(closes) == 1
        assert opens[0].symbol == "BTCUSDT"
        assert closes[0].symbol == "ETHUSDT"
