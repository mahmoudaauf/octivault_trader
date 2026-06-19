from __future__ import annotations

import json
import time

import pytest

from core_engine.native.decisions import Action, NativeDecisionEngine, PortfolioSnapshot
from core_engine.native.portfolio_recovery import PortfolioRecoveryEngine
from core_engine.native.shared_state import NativeSharedState


def _portfolio(
    *,
    nav: float,
    balance: dict[str, float],
    positions: dict[str, float],
) -> PortfolioSnapshot:
    return PortfolioSnapshot(
        nav=nav,
        nav_peak=nav,
        balance=balance,
        positions=positions,
        open_orders={},
    )


@pytest.mark.asyncio
async def test_restart_with_unknown_entry_blocks_buy_and_activates_recovery(tmp_path) -> None:
    state = NativeSharedState()
    state.balance = {"USDT": 5.0, "BTC": 0.001}
    state.prices = {}
    recovery = PortfolioRecoveryEngine(shared_state=state, trade_journal_dir=str(tmp_path))

    snapshot = await recovery.refresh(force=True)

    assert snapshot["recovery_mode_active"] is True
    assert snapshot["buy_blocked"] is True
    assert snapshot["positions"]["BTCUSDT"]["entry_price_confidence"] == "UNKNOWN"
    assert "MIN_TRADE_NOT_MET" in snapshot["reason"]


@pytest.mark.asyncio
async def test_restart_with_local_journal_restores_entry_and_pnl(tmp_path) -> None:
    journal = tmp_path / "trade_journal_20260508.jsonl"
    journal.write_text(
        json.dumps(
            {
                "epoch": time.time() - 7200,  # 2 hours ago, past the 45-min min-hold
                "event": "ORDER_FILLED",
                "symbol": "ETHUSDT",
                "side": "BUY",
                "qty": 1.0,
                "price": 100.0,
            }
        )
        + "\n"
    )
    state = NativeSharedState()
    state.mark_ready()
    state.balance = {"USDT": 50.0, "ETH": 1.0}
    state.prices = {"ETHUSDT": 120.0}
    recovery = PortfolioRecoveryEngine(shared_state=state, trade_journal_dir=str(tmp_path))

    snapshot = await recovery.refresh(force=True)
    pos = snapshot["positions"]["ETHUSDT"]

    assert pos["avg_entry_price"] == pytest.approx(100.0)
    assert pos["entry_price_confidence"] == "HIGH"
    # Fee-adjusted PnL: 0.1% buy + 0.1% sell; (120*0.999 - 100*1.001) / (100*1.001) * 100
    assert pos["unrealized_pnl_pct"] == pytest.approx(19.760239760239763, rel=1e-4)
    assert pos["status"] == "PROFITABLE"


@pytest.mark.asyncio
async def test_low_free_usdt_selects_recovery_candidate() -> None:
    state = NativeSharedState()
    state.mark_ready()
    state.balance = {"USDT": 1.0, "SOL": 1.0}
    state.prices = {"SOLUSDT": 50.0}
    state.position_recovery = {
        "SOLUSDT": {
            "symbol": "SOLUSDT",
            "entry_price": 60.0,
            "entry_time": time.time() - 3600,
        }
    }
    recovery = PortfolioRecoveryEngine(shared_state=state, trade_journal_dir="logs")
    snapshot = await recovery.refresh(force=True)

    assert snapshot["buy_blocked"] is True
    assert snapshot["selected_symbol"] == "SOLUSDT"
    assert snapshot["selected_action"] in {"SELL_RECOVERY", "SELL_PROFIT", "SELL_STALE"}


@pytest.mark.asyncio
async def test_throttled_startup_defers_hydration_and_keeps_buy_blocked() -> None:
    state = NativeSharedState()
    state.set_exchange_throttle(True, reason="418", until_ts=time.time() + 120)
    state.balance = {"USDT": 0.0}
    recovery = PortfolioRecoveryEngine(shared_state=state, trade_journal_dir="logs")

    snapshot = await recovery.refresh(force=True)

    assert snapshot["hydration_deferred"] is True
    assert snapshot["ready"] is True
    assert snapshot["buy_blocked"] is True


@pytest.mark.asyncio
async def test_stale_positions_alone_do_not_keep_buy_blocked_when_capital_is_healthy() -> None:
    journal = None
    state = NativeSharedState()
    state.mark_ready()
    state.balance = {"USDT": 60.0, "XRP": 10.0}
    state.prices = {"XRPUSDT": 1.0}
    state.position_recovery = {
        "XRPUSDT": {
            "symbol": "XRPUSDT",
            "entry_price": 1.1,
            "entry_time": time.time() - (48 * 3600),
        }
    }
    recovery = PortfolioRecoveryEngine(shared_state=state, trade_journal_dir="logs")

    snapshot = await recovery.refresh(force=True)

    assert snapshot["recovery_mode_active"] is False
    assert snapshot["buy_blocked"] is False
    assert snapshot["selected_symbol"] == "XRPUSDT"
    assert snapshot["selected_action"] == "SELL_STALE"
    assert "STALE_POSITIONS" in snapshot["reason"]


def test_sell_reason_gate_allows_recovery_without_profit() -> None:
    state = NativeSharedState()
    state.mark_ready()
    state.balance = {"USDT": 1.0, "DOGE": 100.0}
    state.prices = {"DOGEUSDT": 0.2}
    state.position_recovery = {
        "DOGEUSDT": {
            "symbol": "DOGEUSDT",
            "entry_price": 0.3,
            "entry_time": time.time() - 7200,
        }
    }
    recovery = PortfolioRecoveryEngine(shared_state=state, trade_journal_dir="logs")
    state.recovery_state = {
        "buy_blocked": True,
        "reason": "LOW_FREE_USDT",
        "recovery_mode_active": True,
        "selected_symbol": "DOGEUSDT",
        "max_recovery_sells_per_cycle": 1,
        "last_recovery_decision_ts": 0.0,
    }
    state.position_recovery["DOGEUSDT"].update(
        {
            "symbol": "DOGEUSDT",
            "qty": 100.0,
            "notional_usdt": 20.0,
            "status": "WEAK",
            "reason": "negative_momentum",
            "entry_price_confidence": "MEDIUM",
        }
    )
    eng = NativeDecisionEngine(shared_state=state, portfolio_recovery=recovery)
    portfolio = _portfolio(nav=21.0, balance=state.balance, positions={"DOGEUSDT": 100.0})

    decisions = eng.decide({"BTCUSDT": {"direction": "BUY", "score": 0.9}}, portfolio, 1.0)

    assert all(d.action != Action.OPEN for d in decisions)
    closes = [d for d in decisions if d.action == Action.CLOSE]
    assert closes
    assert closes[0].sell_reason == "CAPITAL_RECOVERY"


def test_take_profit_sell_reason_requires_positive_signal_reason_only() -> None:
    assert NativeDecisionEngine._map_signal_sell_reason({"reason": "take_profit"}) == "TAKE_PROFIT"
    assert NativeDecisionEngine._map_signal_sell_reason({"reason": "stop_loss"}) == "STOP_LOSS"
    assert NativeDecisionEngine._map_recovery_status_to_sell_reason("DUST") == "DUST_CLEANUP"
    assert NativeDecisionEngine._map_recovery_status_to_sell_reason("STALE") == "STALE_EXIT"


def test_dust_remnants_do_not_block_new_position_slots() -> None:
    state = NativeSharedState()
    state.mark_ready()
    state.balance = {"USDT": 83.25, "DOGE": 12.0}
    state.position_recovery = {
        "DOGEUSDT": {
            "symbol": "DOGEUSDT",
            "qty": 12.0,
            "status": "DUST",
            "notional_usdt": 2.4,
            "sellable": False,
        }
    }
    engine = NativeDecisionEngine(
        shared_state=state,
        min_order_usdt=1.0,
        min_notional_usdt=10.0,
        max_concurrent_positions=1,
        max_position_size_pct=100.0,
        risk_per_symbol_pct=100.0,
        kelly_fraction=1.0,
    )
    portfolio = _portfolio(
        nav=85.65,
        balance=state.balance,
        positions={"DOGEUSDT": 12.0},
    )

    decisions = engine.decide(
        {"BTCUSDT": {"direction": "BUY", "score": 0.9, "confidence": 0.9}},
        portfolio,
        83.25,
    )

    opens = [d for d in decisions if d.action == Action.OPEN]
    assert opens
    assert opens[0].symbol == "BTCUSDT"


def test_remnant_reactivation_requires_strong_signal_and_meaningful_size() -> None:
    state = NativeSharedState()
    state.mark_ready()
    state.balance = {"USDT": 40.0, "DOGE": 12.0}
    state.position_recovery = {
        "DOGEUSDT": {
            "symbol": "DOGEUSDT",
            "qty": 12.0,
            "status": "DUST",
            "notional_usdt": 2.4,
            "sellable": False,
        }
    }
    engine = NativeDecisionEngine(
        shared_state=state,
        min_order_usdt=1.0,
        min_notional_usdt=10.0,
        max_concurrent_positions=1,
        max_position_size_pct=100.0,
        risk_per_symbol_pct=100.0,
        kelly_fraction=1.0,
        max_cluster_exposure_pct=100.0,
    )
    portfolio = _portfolio(
        nav=42.4,
        balance=state.balance,
        positions={"DOGEUSDT": 12.0},
    )

    weak = engine.decide(
        {"DOGEUSDT": {"direction": "BUY", "score": 0.72, "confidence": 0.72}},
        portfolio,
        40.0,
    )
    strong = engine.decide(
        {"DOGEUSDT": {"direction": "BUY", "score": 0.92, "confidence": 0.92}},
        portfolio,
        40.0,
    )

    assert all(d.action != Action.OPEN for d in weak)
    strong_opens = [d for d in strong if d.action == Action.OPEN]
    assert strong_opens
    assert strong_opens[0].symbol == "DOGEUSDT"


def test_selective_rebalance_can_reactivate_best_remnant_over_weaker_fresh_symbol() -> None:
    state = NativeSharedState()
    state.mark_ready()
    state.balance = {"USDT": 55.0, "DOGE": 12.0}
    state.position_recovery = {
        "DOGEUSDT": {
            "symbol": "DOGEUSDT",
            "qty": 12.0,
            "status": "DUST",
            "notional_usdt": 2.4,
            "sellable": False,
        }
    }
    state.nav_protection_state["allow_buy"] = True
    state.recovery_state["recovery_mode_active"] = False
    engine = NativeDecisionEngine(
        shared_state=state,
        min_order_usdt=1.0,
        min_notional_usdt=10.0,
        max_concurrent_positions=2,
        max_position_size_pct=100.0,
        risk_per_symbol_pct=100.0,
        kelly_fraction=1.0,
        max_cluster_exposure_pct=100.0,
    )
    portfolio = _portfolio(
        nav=57.4,
        balance=state.balance,
        positions={"DOGEUSDT": 12.0},
    )

    decisions = engine.decide(
        {
            "DOGEUSDT": {
                "direction": "BUY",
                "score": 0.86,
                "confidence": 0.86,
                "entry_quality": 0.70,
            },
            "ADAUSDT": {
                "direction": "BUY",
                "score": 0.78,
                "confidence": 0.78,
                "entry_quality": 0.58,
            },
        },
        portfolio,
        55.0,
    )

    opens = [d for d in decisions if d.action == Action.OPEN]
    assert opens
    assert opens[0].symbol == "DOGEUSDT"


def test_selective_rebalance_does_not_override_stronger_fresh_candidate() -> None:
    state = NativeSharedState()
    state.mark_ready()
    state.balance = {"USDT": 55.0, "DOGE": 12.0}
    state.position_recovery = {
        "DOGEUSDT": {
            "symbol": "DOGEUSDT",
            "qty": 12.0,
            "status": "DUST",
            "notional_usdt": 2.4,
            "sellable": False,
        }
    }
    state.nav_protection_state["allow_buy"] = True
    state.recovery_state["recovery_mode_active"] = False
    engine = NativeDecisionEngine(
        shared_state=state,
        min_order_usdt=1.0,
        min_notional_usdt=10.0,
        max_concurrent_positions=2,
        max_position_size_pct=100.0,
        risk_per_symbol_pct=100.0,
        kelly_fraction=1.0,
        max_cluster_exposure_pct=100.0,
    )
    portfolio = _portfolio(
        nav=57.4,
        balance=state.balance,
        positions={"DOGEUSDT": 12.0},
    )

    decisions = engine.decide(
        {
            "DOGEUSDT": {
                "direction": "BUY",
                "score": 0.82,
                "confidence": 0.82,
                "entry_quality": 0.70,
            },
            "ADAUSDT": {
                "direction": "BUY",
                "score": 0.90,
                "confidence": 0.90,
                "entry_quality": 0.70,
            },
        },
        portfolio,
        55.0,
    )

    opens = [d for d in decisions if d.action == Action.OPEN]
    assert opens
    assert opens[0].symbol == "ADAUSDT"
