from __future__ import annotations

import json
import time
from unittest.mock import MagicMock

import pytest

from core_engine.native.arbitration_engine import NativeArbitrationEngine
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


# ── Below: rewritten 2026-07-14. The original 6 tests here constructed
# NativeDecisionEngine(shared_state=..., portfolio_recovery=...) and called
# _map_signal_sell_reason/_map_recovery_status_to_sell_reason -- none of
# which ever existed on this class (confirmed via full git history), and no
# equivalent exists anywhere on the live path either. These tests describe
# a dust-remnant-reactivation feature that was speced but never built.
# Building it for real would mean carving a signal-strength-based exception
# into arbitration_engine.py's $2 anti-stacking rebuy gate (REBUY_BLOCK_NOTIONAL),
# which was deliberately hardened after a real overconcentration incident
# (see the gate_4_position_limit comment re: "the WLFI/DOGE concentration").
# Per a 2026-07-14 decision, that gate stays as-is; these tests are rewritten
# to assert the REAL, current, verified behavior instead.


def test_dust_position_does_not_block_new_symbol_slot(tmp_path) -> None:
    """gate_4_position_limit's own slot-counting (arbitration_engine.py) already
    excludes positions below min_notional_usdt ($10) from the active-position
    count -- confirmed live behavior, no change needed. A $2.40 DOGEUSDT dust
    remnant must not consume the single available slot for a fresh BTCUSDT buy."""
    ss = NativeSharedState()
    ss.positions = {"DOGEUSDT": {"qty": 12.0}}
    ss.prices = {"DOGEUSDT": 0.2}  # notional = 12.0 * 0.2 = $2.40 -- below $10 threshold

    de = MagicMock()
    de.min_notional_usdt = 10.0
    de.max_concurrent_positions = 1
    de._resolve_mode = MagicMock(return_value={"max_positions": 1})

    engine = NativeArbitrationEngine(shared_state=ss, decision_engine=de)
    engine._arb_state_path = str(tmp_path / "arb_state.json")
    engine._load_streak_state()

    assert engine.gate_4_position_limit("BTCUSDT") is True


def test_rebuy_block_applies_regardless_of_signal_strength(tmp_path) -> None:
    """The anti-stacking rebuy gate (REBUY_BLOCK_NOTIONAL, default $2) blocks
    re-buying an already-held symbol unconditionally once its notional is at
    or above the threshold -- it takes no signal/confidence input at all, so
    a "strong" signal cannot override it. This is deliberate (see the
    WLFI/DOGE concentration incident documented in gate_4_position_limit)."""
    ss = NativeSharedState()
    ss.positions = {"DOGEUSDT": {"qty": 12.0}}
    ss.prices = {"DOGEUSDT": 0.2}  # notional = $2.40 -- at/above the $2 rebuy threshold

    de = MagicMock()
    de.min_notional_usdt = 10.0
    de.max_concurrent_positions = 5
    de._resolve_mode = MagicMock(return_value={"max_positions": 5})

    engine = NativeArbitrationEngine(shared_state=ss, decision_engine=de)
    engine._arb_state_path = str(tmp_path / "arb_state.json")
    engine._load_streak_state()

    # gate_4_position_limit has no signal-strength parameter -- it blocks
    # purely on held notional, so there is no "strong signal" input that
    # could change this outcome.
    assert engine.gate_4_position_limit("DOGEUSDT") is False


def test_held_symbol_always_ranked_below_fresh_candidate_regardless_of_score() -> None:
    """NativeDecisionEngine._rank_buy_signals applies a flat -1.0 held_penalty
    to any symbol already in portfolio.positions, regardless of qty/status --
    there is no dust-aware "reactivate the best remnant" ranking bonus. A held
    DOGEUSDT with a HIGHER raw score than a fresh ADAUSDT still ranks lower,
    because the -1.0 penalty dwarfs any realistic score gap."""
    engine = NativeDecisionEngine(
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
        balance={"USDT": 55.0, "DOGE": 12.0},
        positions={"DOGEUSDT": 12.0},
    )

    decisions = engine.decide(
        {
            "DOGEUSDT": {"direction": "BUY", "score": 0.86, "confidence": 0.86},
            "ADAUSDT": {"direction": "BUY", "score": 0.78, "confidence": 0.78},
        },
        portfolio,
        55.0,
    )

    opens = [d for d in decisions if d.action == Action.OPEN]
    assert opens
    assert opens[0].symbol == "ADAUSDT", (
        "held_penalty must keep the already-held symbol ranked below a fresh "
        f"candidate even with a higher raw score; got opens={[d.symbol for d in opens]}"
    )


def test_fresh_candidate_wins_over_held_symbol_with_lower_score_too() -> None:
    """Same mechanism as above, with a fresh candidate that also has the
    higher raw score -- the held symbol loses either way."""
    engine = NativeDecisionEngine(
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
        balance={"USDT": 55.0, "DOGE": 12.0},
        positions={"DOGEUSDT": 12.0},
    )

    decisions = engine.decide(
        {
            "DOGEUSDT": {"direction": "BUY", "score": 0.82, "confidence": 0.82},
            "ADAUSDT": {"direction": "BUY", "score": 0.90, "confidence": 0.90},
        },
        portfolio,
        55.0,
    )

    opens = [d for d in decisions if d.action == Action.OPEN]
    assert opens
    assert opens[0].symbol == "ADAUSDT"
