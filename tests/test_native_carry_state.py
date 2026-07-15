"""
Tests for CarrySharedState (Phase 2 of the funding-carry native-wiring plan).

This is the highest-risk arithmetic in the whole plan (NAV-injection seam),
so coverage here is deliberately thorough: lifecycle, persistence
round-trip, locked-capital summation, net-exposure sign math (the part most
likely to be silently wrong), and the ledger schema.
"""

from __future__ import annotations

import pytest

from core_engine.native.carry.state import (
    DEFAULT_LEDGER_PATH,
    DEFAULT_STATE_PATH,
    CarrySharedState,
    HedgePosition,
)


def _state(tmp_path) -> CarrySharedState:
    return CarrySharedState(
        state_path=str(tmp_path / "carry_state.json"),
        ledger_path=str(tmp_path / "carry_ledger.jsonl"),
    )


class TestHedgeLifecycle:
    def test_open_hedge_positive_funding_is_short_perp(self, tmp_path) -> None:
        s = _state(tmp_path)
        pos = s.open_hedge(
            "BTCUSDT", entry_funding=0.0006, perp_qty=0.01, spot_qty=0.01, notional_usd=500.0
        )
        assert pos.direction == "short_perp"
        assert s.get_open_hedge("BTCUSDT") is pos
        assert s.open_count() == 1
        assert s.open_symbols() == ["BTCUSDT"]

    def test_open_hedge_negative_funding_is_long_perp(self, tmp_path) -> None:
        """Kept symmetric for interface completeness even though v1's
        POSITIVE_ONLY restriction means this branch isn't reachable via the
        real strategy logic (see HedgePosition docstring)."""
        s = _state(tmp_path)
        pos = s.open_hedge(
            "ETHUSDT", entry_funding=-0.0003, perp_qty=1.0, spot_qty=1.0, notional_usd=1000.0
        )
        assert pos.direction == "long_perp"

    def test_open_hedge_duplicate_symbol_raises(self, tmp_path) -> None:
        s = _state(tmp_path)
        s.open_hedge("BTCUSDT", entry_funding=0.0006, perp_qty=0.01, spot_qty=0.01, notional_usd=500.0)
        with pytest.raises(ValueError):
            s.open_hedge("BTCUSDT", entry_funding=0.0006, perp_qty=0.01, spot_qty=0.01, notional_usd=500.0)

    def test_close_hedge_returns_and_removes(self, tmp_path) -> None:
        s = _state(tmp_path)
        s.open_hedge("BTCUSDT", entry_funding=0.0006, perp_qty=0.01, spot_qty=0.01, notional_usd=500.0)
        closed = s.close_hedge("BTCUSDT")
        assert closed is not None
        assert closed.symbol == "BTCUSDT"
        assert s.open_count() == 0
        assert s.get_open_hedge("BTCUSDT") is None

    def test_close_hedge_unknown_symbol_returns_none(self, tmp_path) -> None:
        s = _state(tmp_path)
        assert s.close_hedge("NOSUCHUSDT") is None

    def test_held_h_computes_from_entry_ts(self) -> None:
        pos = HedgePosition(
            symbol="BTCUSDT", entry_ts=1000.0, entry_funding=0.0006,
            direction="short_perp", perp_qty=0.01, spot_qty=0.01, notional_usd=500.0,
        )
        assert pos.held_h(now=1000.0 + 7200.0) == pytest.approx(2.0)


class TestPersistence:
    def test_state_survives_reload(self, tmp_path) -> None:
        state_path = str(tmp_path / "carry_state.json")
        ledger_path = str(tmp_path / "carry_ledger.jsonl")
        s1 = CarrySharedState(state_path=state_path, ledger_path=ledger_path)
        s1.open_hedge("BTCUSDT", entry_funding=0.0006, perp_qty=0.01, spot_qty=0.01, notional_usd=500.0)

        s2 = CarrySharedState(state_path=state_path, ledger_path=ledger_path)
        restored = s2.get_open_hedge("BTCUSDT")
        assert restored is not None
        assert restored.perp_qty == pytest.approx(0.01)
        assert restored.notional_usd == pytest.approx(500.0)
        assert restored.direction == "short_perp"

    def test_close_persists_removal_across_reload(self, tmp_path) -> None:
        state_path = str(tmp_path / "carry_state.json")
        ledger_path = str(tmp_path / "carry_ledger.jsonl")
        s1 = CarrySharedState(state_path=state_path, ledger_path=ledger_path)
        s1.open_hedge("BTCUSDT", entry_funding=0.0006, perp_qty=0.01, spot_qty=0.01, notional_usd=500.0)
        s1.close_hedge("BTCUSDT")

        s2 = CarrySharedState(state_path=state_path, ledger_path=ledger_path)
        assert s2.open_count() == 0

    def test_missing_state_file_starts_empty(self, tmp_path) -> None:
        s = CarrySharedState(
            state_path=str(tmp_path / "does_not_exist.json"),
            ledger_path=str(tmp_path / "carry_ledger.jsonl"),
        )
        assert s.open_count() == 0

    def test_corrupt_state_file_starts_empty_not_raises(self, tmp_path) -> None:
        p = tmp_path / "carry_state.json"
        p.write_text("{not valid json")
        s = CarrySharedState(state_path=str(p), ledger_path=str(tmp_path / "l.jsonl"))
        assert s.open_count() == 0

    def test_default_paths_are_distinct_from_standalone_script(self) -> None:
        """Regression guard: the standalone carry_paper_trader.py daemon
        writes logs/carry_state.json and logs/carry_ledger.jsonl. This
        module must never default to those exact paths, or two independent
        processes could race-write the same files once this is wired into
        the live runtime while the standalone daemon is still running."""
        assert DEFAULT_STATE_PATH != "logs/carry_state.json"
        assert DEFAULT_LEDGER_PATH != "logs/carry_ledger.jsonl"


class TestNAVIntegrationSeam:
    def test_locked_capital_usd_sums_across_open_hedges(self, tmp_path) -> None:
        s = _state(tmp_path)
        s.open_hedge("BTCUSDT", entry_funding=0.0006, perp_qty=0.01, spot_qty=0.01, notional_usd=500.0)
        s.open_hedge("ETHUSDT", entry_funding=0.0008, perp_qty=1.0, spot_qty=1.0, notional_usd=300.0)
        assert s.locked_capital_usd() == pytest.approx(800.0)

    def test_locked_capital_usd_zero_when_flat(self, tmp_path) -> None:
        s = _state(tmp_path)
        assert s.locked_capital_usd() == 0.0

    def test_locked_capital_usd_excludes_closed_hedges(self, tmp_path) -> None:
        s = _state(tmp_path)
        s.open_hedge("BTCUSDT", entry_funding=0.0006, perp_qty=0.01, spot_qty=0.01, notional_usd=500.0)
        s.close_hedge("BTCUSDT")
        assert s.locked_capital_usd() == 0.0

    def test_net_exposure_near_zero_for_well_hedged_position(self, tmp_path) -> None:
        """The whole point of delta-neutral carry: equal qty on both legs at
        (near-)equal prices should net out to ~0 exposure."""
        s = _state(tmp_path)
        s.open_hedge("BTCUSDT", entry_funding=0.0006, perp_qty=0.01, spot_qty=0.01, notional_usd=500.0)
        net, healthy = s.net_exposure_usd({"BTCUSDT": {"perp": 50000.0, "spot": 50000.0}})
        assert healthy is True
        assert net == pytest.approx(0.0, abs=1e-9)

    def test_net_exposure_nonzero_when_perp_and_spot_prices_diverge(self, tmp_path) -> None:
        """A real (small) basis between perp and spot should show up as a
        real (small) net exposure, not be silently hidden."""
        s = _state(tmp_path)
        s.open_hedge("BTCUSDT", entry_funding=0.0006, perp_qty=0.01, spot_qty=0.01, notional_usd=500.0)
        # perp trading at a premium to spot (common when funding is positive)
        net, healthy = s.net_exposure_usd({"BTCUSDT": {"perp": 50100.0, "spot": 50000.0}})
        assert healthy is True
        # spot_value(500.00) - perp_value(501.00) = -1.00
        assert net == pytest.approx(-1.00, abs=1e-6)

    def test_net_exposure_missing_price_marks_unhealthy(self, tmp_path) -> None:
        s = _state(tmp_path)
        s.open_hedge("BTCUSDT", entry_funding=0.0006, perp_qty=0.01, spot_qty=0.01, notional_usd=500.0)
        s.open_hedge("ETHUSDT", entry_funding=0.0004, perp_qty=1.0, spot_qty=1.0, notional_usd=300.0)
        # Only BTCUSDT price supplied -- ETHUSDT is missing.
        net, healthy = s.net_exposure_usd({"BTCUSDT": {"perp": 50000.0, "spot": 50000.0}})
        assert healthy is False

    def test_net_exposure_empty_book_is_zero_and_healthy(self, tmp_path) -> None:
        s = _state(tmp_path)
        net, healthy = s.net_exposure_usd({})
        assert net == 0.0
        assert healthy is True

    def test_net_exposure_long_perp_sign_is_symmetric(self, tmp_path) -> None:
        s = _state(tmp_path)
        s.open_hedge("BTCUSDT", entry_funding=-0.0003, perp_qty=0.01, spot_qty=0.01, notional_usd=500.0)
        net, healthy = s.net_exposure_usd({"BTCUSDT": {"perp": 50000.0, "spot": 50000.0}})
        assert healthy is True
        assert net == pytest.approx(1000.0, abs=1e-6)  # spot(500) + perp(500), same-sign for long_perp


class TestLedger:
    def test_record_and_read_closed_trade(self, tmp_path) -> None:
        s = _state(tmp_path)
        s.record_closed_trade(
            "BTCUSDT", held_h=12.5, accrued_funding_pct=0.24, net_pct=0.0,
            exit_funding=0.00005, mode="paper",
        )
        trades = s.read_ledger()
        assert len(trades) == 1
        t = trades[0]
        assert t["symbol"] == "BTCUSDT"
        assert t["held_h"] == 12.5
        assert t["net_pct"] == 0.0
        assert t["mode"] == "paper"
        assert "ts" in t

    def test_read_ledger_missing_file_returns_empty(self, tmp_path) -> None:
        s = _state(tmp_path)
        assert s.read_ledger() == []

    def test_read_ledger_skips_corrupt_lines(self, tmp_path) -> None:
        s = _state(tmp_path)
        s.record_closed_trade("BTCUSDT", held_h=1.0, accrued_funding_pct=0.1, net_pct=0.0, exit_funding=0.0)
        with open(s.ledger_path, "a") as f:
            f.write("not valid json\n")
        s.record_closed_trade("ETHUSDT", held_h=2.0, accrued_funding_pct=0.2, net_pct=0.1, exit_funding=0.0)
        trades = s.read_ledger()
        assert len(trades) == 2
        assert [t["symbol"] for t in trades] == ["BTCUSDT", "ETHUSDT"]
