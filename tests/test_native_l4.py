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
from core_engine.native.arbitration_engine import NativeArbitrationEngine
from core_engine.native.decisions import Action
from core_engine.native.signal_fusion import NativeSignalFusion
from core_engine.native.signals import NativeSignalEngine


# ─────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────
def _portfolio(
    nav: float = 10000.0,
    nav_peak: float = 10000.0,
    balance: dict[str, float] | None = None,
    positions: dict[str, float] | None = None,
    prices: dict[str, float] | None = None,
) -> PortfolioSnapshot:
    snap = PortfolioSnapshot(
        nav=nav,
        nav_peak=nav_peak,
        balance=balance or {"USDT": 10000.0},
        positions=positions or {},
        open_orders={},
    )
    snap.prices = prices or {}
    return snap


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
        assert eng.min_order_usdt == 1.0

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

    def test_small_nav_bootstrap_mode_blocks_extra_slots(self) -> None:
        eng = NativeDecisionEngine(min_notional_usdt=1.0)
        signals = {
            "BTCUSDT": _signal_buy("BTCUSDT", 0.8),
            "ETHUSDT": _signal_buy("ETHUSDT", 0.9),
        }
        portfolio = _portfolio(nav=80.0, nav_peak=80.0, balance={"USDT": 80.0})
        decisions = eng.decide(signals, portfolio, balance_usdt=80.0)
        opens = [d for d in decisions if d.action == Action.OPEN]
        # BOOTSTRAP now allows up to 3 concurrent positions (raised from 1 for capital efficiency)
        assert len(opens) == 2
        symbols = {d.symbol for d in opens}
        assert "ETHUSDT" in symbols

    def test_paused_mode_blocks_open_decisions(self) -> None:
        eng = NativeDecisionEngine()
        signals = {"BTCUSDT": _signal_buy("BTCUSDT", 0.95)}
        portfolio = _portfolio()
        portfolio.mode_name = "PAUSED"
        decisions = eng.decide(signals, portfolio, balance_usdt=10000.0)
        assert decisions == []

    def test_protective_mode_raises_confidence_floor(self) -> None:
        eng = NativeDecisionEngine(confidence_floor=0.50)
        signals = {"BTCUSDT": {"symbol": "BTCUSDT", "direction": "BUY", "score": 0.55}}
        portfolio = _portfolio()
        portfolio.mode_name = "PROTECTIVE"
        decisions = eng.decide(signals, portfolio, balance_usdt=10000.0)
        assert decisions == []

    def test_regime_gate_blocks_low_liquidity_buy(self) -> None:
        eng = NativeDecisionEngine()
        signals = {
            "BTCUSDT": {
                "symbol": "BTCUSDT",
                "direction": "BUY",
                "score": 0.9,
                "confidence": 0.9,
                "regime": "low_liquidity",
                "liquidity_score": 0.1,
            }
        }
        portfolio = _portfolio()
        decisions = eng.decide(signals, portfolio, balance_usdt=10000.0)
        assert decisions == []

    def test_regime_gate_tightens_confidence_in_volatile_mode(self) -> None:
        eng = NativeDecisionEngine(confidence_floor=0.50)
        signals = {
            "BTCUSDT": {
                "symbol": "BTCUSDT",
                "direction": "BUY",
                "score": 0.49,
                "confidence": 0.49,
                "regime": "volatile",
                "volatility_score": 0.95,
                "liquidity_score": 0.8,
            }
        }
        portfolio = _portfolio()
        decisions = eng.decide(signals, portfolio, balance_usdt=10000.0)
        assert decisions == []

    def test_regime_gate_allows_high_confidence_volatile_buy(self) -> None:
        eng = NativeDecisionEngine(confidence_floor=0.50)
        signals = {
            "BTCUSDT": {
                "symbol": "BTCUSDT",
                "direction": "BUY",
                "score": 0.75,
                "confidence": 0.75,
                "regime": "volatile",
                "volatility_score": 0.95,
                "liquidity_score": 0.8,
            }
        }
        portfolio = _portfolio()
        decisions = eng.decide(signals, portfolio, balance_usdt=10000.0)
        opens = [d for d in decisions if d.action == Action.OPEN]
        assert len(opens) == 1

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

    def test_reserve_policy_reduces_spendable_quote(self) -> None:
        eng = NativeDecisionEngine(quote_reserve_ratio=0.10, quote_min_reserve_usdt=10.0)
        assert eng._compute_spendable_quote(100.0) == 90.0
        assert eng._compute_spendable_quote(20.0) == 19.0

    def test_daily_loss_gate_blocks_new_decisions(self) -> None:
        eng = NativeDecisionEngine(daily_loss_limit_pct=2.0)
        signals = {"BTCUSDT": _signal_buy("BTCUSDT", 0.9)}
        portfolio = PortfolioSnapshot(
            nav=1000.0,
            nav_peak=1000.0,
            balance={"USDT": 1000.0},
            positions={},
            open_orders={},
            daily_pnl_pct=-3.0,
        )
        decisions = eng.decide(signals, portfolio, balance_usdt=1000.0)
        assert decisions == []

    def test_daily_loss_gate_does_not_block_profitable_day(self) -> None:
        eng = NativeDecisionEngine(daily_loss_limit_pct=2.0)
        signals = {"BTCUSDT": _signal_buy("BTCUSDT", 0.9)}
        portfolio = PortfolioSnapshot(
            nav=1030.0,
            nav_peak=1030.0,
            balance={"USDT": 1030.0},
            positions={},
            open_orders={},
            daily_pnl_pct=3.0,
        )
        decisions = eng.decide(signals, portfolio, balance_usdt=1030.0)
        assert len(decisions) == 1

    def test_exposure_gate_blocks_new_buys(self) -> None:
        eng = NativeDecisionEngine(max_total_exposure_pct=60.0)
        signals = {"BTCUSDT": _signal_buy("BTCUSDT", 0.9)}
        portfolio = _portfolio(nav=1000.0, balance={"USDT": 100.0})
        decisions = eng.decide(signals, portfolio, balance_usdt=100.0)
        assert decisions == []

    def test_position_size_respects_max_position_size_pct(self) -> None:
        eng = NativeDecisionEngine(max_position_size_pct=2.0)
        signals = {"BTCUSDT": _signal_buy("BTCUSDT", 1.0)}
        portfolio = _portfolio(nav=10000.0)
        decisions = eng.decide(signals, portfolio, balance_usdt=10000.0)
        assert len(decisions) == 1
        # Position should be capped at ~2% of NAV = 200 USDT
        # (This is a soft cap; exact number depends on Kelly calcs)

    def test_cluster_exposure_gate_blocks_correlated_buy(self) -> None:
        eng = NativeDecisionEngine(
            max_position_size_pct=5.0,
            max_cluster_exposure_pct=30.0,
            min_notional_usdt=1.0,
        )
        signals = {"ETHUSDT": {**_signal_buy("ETHUSDT", 0.9), "price": 3000.0}}
        portfolio = _portfolio(
            nav=1000.0,
            nav_peak=1000.0,
            balance={"USDT": 500.0},
            positions={"BTCUSDT": 0.3},
            prices={"BTCUSDT": 1000.0},
        )
        decisions = eng.decide(signals, portfolio, balance_usdt=500.0)
        assert decisions == []

    def test_cluster_exposure_gate_allows_unrelated_buy(self) -> None:
        eng = NativeDecisionEngine(
            max_position_size_pct=5.0,
            max_cluster_exposure_pct=40.0,
            min_notional_usdt=1.0,
        )
        signals = {"DOGEUSDT": {**_signal_buy("DOGEUSDT", 0.9), "price": 0.2}}
        portfolio = _portfolio(
            nav=1000.0,
            nav_peak=1000.0,
            balance={"USDT": 500.0},
            positions={"BTCUSDT": 0.3},
            prices={"BTCUSDT": 1000.0},
        )
        decisions = eng.decide(signals, portfolio, balance_usdt=500.0)
        opens = [d for d in decisions if d.action == Action.OPEN]
        assert len(opens) == 1
        assert opens[0].symbol == "DOGEUSDT"

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

    @pytest.mark.asyncio
    async def test_arbitration_engine_blocks_low_confidence_buy(self) -> None:
        shared_state = type(
            "_SS",
            (),
            {
                "current_mode": "SAFE",
                "runtime_overrides": {},
                "free_balance_usdt": 100.0,
                "balance": {"USDT": 100.0},
                "positions": {},
                "price_cache": {},
                "metrics": {"peak_nav": 100.0, "realized_pnl": 0.0},
                "session_anchor_nav": 100.0,
                "open_orders": {},
                "trading_halted": False,
            },
        )()
        mode_manager = type(
            "_MM",
            (),
            {
                "get_constraints": lambda self, mode, nav=0.0: {
                    "confidence_floor": 0.9,
                    "max_positions": 1,
                    "max_trade_usdt": 30.0,
                    "allow_new": True,
                }
            },
        )()
        arb = NativeArbitrationEngine(
            shared_state=shared_state,
            decision_engine=NativeDecisionEngine(confidence_floor=0.5, min_order_usdt=1.0),
            signal_fusion=None,
            mode_manager=mode_manager,
        )
        result = await arb.evaluate("BTCUSDT", "BUY", 0.4)
        assert result["passed"] is False
        assert "gate_2_confidence" in result["blocking_gates"]

    @pytest.mark.asyncio
    async def test_arbitration_engine_uses_fused_regime_signal(self) -> None:
        decision_engine = NativeDecisionEngine(confidence_floor=0.5, min_order_usdt=1.0)
        shared_state = type(
            "_SS",
            (),
            {
                "current_mode": "NORMAL",
                "runtime_overrides": {},
                "free_balance_usdt": 100.0,
                "balance": {"USDT": 100.0},
                "positions": {},
                "price_cache": {},
                "metrics": {"peak_nav": 100.0, "realized_pnl": 0.0},
                "session_anchor_nav": 100.0,
                "open_orders": {},
                "trading_halted": False,
            },
        )()
        market_data = type(
            "_MD",
            (),
            {
                "_klines": {
                    ("BTCUSDT", "1m", 64): (0.0, [[0, 0, 0, 0, 100.0, 0] for _ in range(60)])
                }
            },
        )()
        signal_engine = NativeSignalEngine(enabled=["ma_cross"])
        fusion = NativeSignalFusion(
            signal_engine=signal_engine,
            market_data=market_data,
            shared_state=shared_state,
        )
        arb = NativeArbitrationEngine(
            shared_state=shared_state,
            decision_engine=decision_engine,
            signal_fusion=fusion,
            mode_manager=None,
        )
        result = await arb.evaluate("BTCUSDT", "BUY", 0.7)
        assert "gate_3_regime" in result["gates_status"]
