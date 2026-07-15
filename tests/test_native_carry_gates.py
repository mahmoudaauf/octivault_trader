"""
Tests for CarryGateEngine (Phase 4 of the funding-carry native-wiring plan).
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from core_engine.native.carry.gates import CarryGateEngine
from core_engine.native.carry.state import CarrySharedState


def _carry_state(tmp_path) -> CarrySharedState:
    return CarrySharedState(
        state_path=str(tmp_path / "carry_state.json"),
        ledger_path=str(tmp_path / "carry_ledger.jsonl"),
    )


def _engine(tmp_path, *, shared_state=None, **overrides) -> CarryGateEngine:
    cs = _carry_state(tmp_path)
    kwargs = dict(
        carry_state=cs,
        shared_state=shared_state,
        entry_bps=6.0,
        exit_bps=1.0,
        positive_only=True,
        max_positions=5,
        max_total_usd=5000.0,
        max_hold_h=360.0,
        max_drawdown_pct=5.0,
        liq_buffer_pct=15.0,
        kill_file=str(tmp_path / "carry.stop"),
    )
    kwargs.update(overrides)
    return CarryGateEngine(**kwargs)


class TestEvaluateOpen:
    def test_allows_qualifying_positive_funding(self, tmp_path) -> None:
        eng = _engine(tmp_path)
        d = eng.evaluate_open("BTCUSDT", 0.0007)  # 7bps, above 6bps entry
        assert d.allowed is True

    def test_blocks_below_entry_threshold(self, tmp_path) -> None:
        eng = _engine(tmp_path)
        d = eng.evaluate_open("BTCUSDT", 0.0003)  # 3bps, below 6bps entry
        assert d.allowed is False
        assert d.reason == "funding_below_entry_threshold"

    def test_blocks_negative_funding_under_positive_only(self, tmp_path) -> None:
        eng = _engine(tmp_path)
        d = eng.evaluate_open("BTCUSDT", -0.0008)
        assert d.allowed is False
        assert d.reason == "negative_funding_v1_unsupported"

    def test_allows_negative_funding_when_positive_only_disabled(self, tmp_path) -> None:
        eng = _engine(tmp_path, positive_only=False)
        d = eng.evaluate_open("BTCUSDT", -0.0008)
        assert d.allowed is True

    def test_blocks_when_already_open(self, tmp_path) -> None:
        eng = _engine(tmp_path)
        eng._carry_state.open_hedge(
            "BTCUSDT", entry_funding=0.0007, perp_qty=0.01, spot_qty=0.01, notional_usd=500.0
        )
        d = eng.evaluate_open("BTCUSDT", 0.0007)
        assert d.allowed is False
        assert d.reason == "already_open"

    def test_blocks_at_max_positions(self, tmp_path) -> None:
        eng = _engine(tmp_path, max_positions=1)
        eng._carry_state.open_hedge(
            "BTCUSDT", entry_funding=0.0007, perp_qty=0.01, spot_qty=0.01, notional_usd=500.0
        )
        d = eng.evaluate_open("ETHUSDT", 0.0009)
        assert d.allowed is False
        assert d.reason == "max_positions_reached"

    def test_blocks_when_account_halted(self, tmp_path) -> None:
        ss = SimpleNamespace(trading_halted=True)
        eng = _engine(tmp_path, shared_state=ss)
        d = eng.evaluate_open("BTCUSDT", 0.0007)
        assert d.allowed is False
        assert d.reason == "account_trading_halted"

    def test_allows_when_account_not_halted(self, tmp_path) -> None:
        ss = SimpleNamespace(trading_halted=False)
        eng = _engine(tmp_path, shared_state=ss)
        d = eng.evaluate_open("BTCUSDT", 0.0007)
        assert d.allowed is True

    def test_blocks_when_no_shared_state_provided_does_not_crash(self, tmp_path) -> None:
        """shared_state is optional -- must not raise, must default to not-halted."""
        eng = _engine(tmp_path, shared_state=None)
        d = eng.evaluate_open("BTCUSDT", 0.0007)
        assert d.allowed is True

    def test_blocks_when_kill_file_present(self, tmp_path) -> None:
        eng = _engine(tmp_path)
        open(eng.kill_file, "w").close()
        d = eng.evaluate_open("BTCUSDT", 0.0007)
        assert d.allowed is False
        assert d.reason == "carry_kill_file_present"


class TestNotionalBudget:
    def test_allows_within_budget(self, tmp_path) -> None:
        eng = _engine(tmp_path, max_total_usd=1000.0)
        d = eng.check_notional_budget(500.0)
        assert d.allowed is True

    def test_blocks_over_budget(self, tmp_path) -> None:
        eng = _engine(tmp_path, max_total_usd=1000.0)
        eng._carry_state.open_hedge(
            "BTCUSDT", entry_funding=0.0007, perp_qty=0.01, spot_qty=0.01, notional_usd=800.0
        )
        d = eng.check_notional_budget(500.0)
        assert d.allowed is False
        assert d.reason == "max_total_notional_exceeded"


class TestEvaluateClose:
    def test_not_open_symbol(self, tmp_path) -> None:
        eng = _engine(tmp_path)
        d = eng.evaluate_close("BTCUSDT", 0.0007)
        assert d.allowed is False
        assert d.reason == "not_open"

    def test_holds_while_funding_still_extreme_and_not_max_hold(self, tmp_path) -> None:
        eng = _engine(tmp_path)
        eng._carry_state.open_hedge(
            "BTCUSDT", entry_funding=0.0007, perp_qty=0.01, spot_qty=0.01, notional_usd=500.0, now=time.time()
        )
        d = eng.evaluate_close("BTCUSDT", 0.0009, now=time.time() + 3600)
        assert d.allowed is False
        assert d.reason == "hold"

    def test_closes_when_funding_normalized(self, tmp_path) -> None:
        eng = _engine(tmp_path)
        eng._carry_state.open_hedge(
            "BTCUSDT", entry_funding=0.0007, perp_qty=0.01, spot_qty=0.01, notional_usd=500.0, now=time.time()
        )
        d = eng.evaluate_close("BTCUSDT", 0.00005, now=time.time() + 3600)  # below 1bps exit
        assert d.allowed is True
        assert d.reason == "funding_normalized"

    def test_closes_when_max_hold_exceeded_even_if_funding_still_extreme(self, tmp_path) -> None:
        eng = _engine(tmp_path, max_hold_h=10.0)
        entry_ts = time.time() - 100 * 3600
        eng._carry_state.open_hedge(
            "BTCUSDT", entry_funding=0.0007, perp_qty=0.01, spot_qty=0.01, notional_usd=500.0, now=entry_ts
        )
        d = eng.evaluate_close("BTCUSDT", 0.0009)  # still extreme funding
        assert d.allowed is True
        assert d.reason == "max_hold_exceeded"

    def test_closes_when_kill_file_present_regardless_of_funding(self, tmp_path) -> None:
        eng = _engine(tmp_path)
        eng._carry_state.open_hedge(
            "BTCUSDT", entry_funding=0.0007, perp_qty=0.01, spot_qty=0.01, notional_usd=500.0, now=time.time()
        )
        open(eng.kill_file, "w").close()
        d = eng.evaluate_close("BTCUSDT", 0.0009, now=time.time() + 3600)  # extreme funding, would normally hold
        assert d.allowed is True
        assert d.reason == "kill_file_present"


class TestDrawdownHalt:
    def test_no_ledger_no_drawdown(self, tmp_path) -> None:
        eng = _engine(tmp_path)
        assert eng.current_drawdown_pct() == 0.0
        assert eng.check_drawdown_halt() is False
        assert not eng._killed()

    def test_all_winning_trades_no_drawdown(self, tmp_path) -> None:
        eng = _engine(tmp_path)
        eng._carry_state.record_closed_trade("BTCUSDT", held_h=10, accrued_funding_pct=0.5, net_pct=0.3, exit_funding=0.0001)
        eng._carry_state.record_closed_trade("ETHUSDT", held_h=8, accrued_funding_pct=0.4, net_pct=0.2, exit_funding=0.0001)
        assert eng.current_drawdown_pct() == 0.0

    def test_drawdown_from_peak_computed_correctly(self, tmp_path) -> None:
        eng = _engine(tmp_path)
        # cum: +2.0 (peak=2.0), then -1.5 (cum=0.5) -> drawdown = 2.0 - 0.5 = 1.5
        eng._carry_state.record_closed_trade("A", held_h=1, accrued_funding_pct=2.0, net_pct=2.0, exit_funding=0.0001)
        eng._carry_state.record_closed_trade("B", held_h=1, accrued_funding_pct=0.0, net_pct=-1.5, exit_funding=0.0001)
        assert eng.current_drawdown_pct() == pytest.approx(1.5)

    def test_halt_triggers_and_touches_kill_file_when_drawdown_breached(self, tmp_path) -> None:
        eng = _engine(tmp_path, max_drawdown_pct=1.0)
        eng._carry_state.record_closed_trade("A", held_h=1, accrued_funding_pct=2.0, net_pct=2.0, exit_funding=0.0001)
        eng._carry_state.record_closed_trade("B", held_h=1, accrued_funding_pct=0.0, net_pct=-1.5, exit_funding=0.0001)
        assert eng.check_drawdown_halt() is True
        assert eng._killed()
        # A subsequent open must now be blocked by the kill file this created.
        d = eng.evaluate_open("BTCUSDT", 0.0007)
        assert d.allowed is False
        assert d.reason == "carry_kill_file_present"

    def test_no_halt_when_drawdown_below_threshold(self, tmp_path) -> None:
        eng = _engine(tmp_path, max_drawdown_pct=10.0)
        eng._carry_state.record_closed_trade("A", held_h=1, accrued_funding_pct=2.0, net_pct=2.0, exit_funding=0.0001)
        eng._carry_state.record_closed_trade("B", held_h=1, accrued_funding_pct=0.0, net_pct=-1.5, exit_funding=0.0001)
        assert eng.check_drawdown_halt() is False
        assert not eng._killed()


class TestLiquidationBuffer:
    def test_not_near_liquidation_when_far_apart(self, tmp_path) -> None:
        eng = _engine(tmp_path, liq_buffer_pct=15.0)
        assert eng.is_near_liquidation(mark_price=50000.0, liquidation_price=30000.0) is False

    def test_near_liquidation_within_buffer(self, tmp_path) -> None:
        eng = _engine(tmp_path, liq_buffer_pct=15.0)
        # 10% away from liq price -- inside the 15% buffer
        assert eng.is_near_liquidation(mark_price=50000.0, liquidation_price=45000.0) is True

    def test_zero_prices_are_safe_not_near(self, tmp_path) -> None:
        eng = _engine(tmp_path)
        assert eng.is_near_liquidation(mark_price=0.0, liquidation_price=0.0) is False
