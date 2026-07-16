from __future__ import annotations

import core_engine.native.symbol_performance_tracker as perf_tracker_module
import core_engine.native.symbol_rotator as rotator_module
from core_engine.native.shared_state import NativeSharedState
from core_engine.native.symbol_performance_tracker import SymbolPerformanceTracker
from core_engine.native.symbol_rotator import SymbolRotator, _get_win_rate


def test_rotation_pins_held_symbol_without_queueing_forced_exit(monkeypatch) -> None:
    state = NativeSharedState()
    state.metrics["symbol_regimes"] = {"NEWUSDT": "UPTREND"}
    state.update_price("OLDUSDT", 20.0)
    state.update_position("OLDUSDT", qty=1.0, entry=19.0, current=20.0)
    state.set_accepted_symbols(["OLDUSDT"])
    monkeypatch.setattr(rotator_module, "_available_model_symbols", lambda: ["NEWUSDT"])

    rotator = SymbolRotator(state, fallback_symbols=["OLDUSDT"])
    rotator._current_universe = ["OLDUSDT"]
    rotator._rotate()

    assert "NEWUSDT" in rotator.current_universe
    assert "OLDUSDT" in rotator.current_universe
    assert state.accepted_symbols == {"NEWUSDT", "OLDUSDT"}
    assert rotator.pop_pending_exits() == {}


def test_rotation_removes_unheld_symbol_from_entry_universe(monkeypatch) -> None:
    state = NativeSharedState()
    state.metrics["symbol_regimes"] = {"NEWUSDT": "UPTREND"}
    state.set_accepted_symbols(["OLDUSDT"])
    monkeypatch.setattr(rotator_module, "_available_model_symbols", lambda: ["NEWUSDT"])

    rotator = SymbolRotator(state, fallback_symbols=["OLDUSDT"])
    rotator._current_universe = ["OLDUSDT"]
    rotator._rotate()

    assert rotator.current_universe == ["NEWUSDT"]
    assert state.accepted_symbols == {"NEWUSDT"}
    assert rotator.pop_pending_exits() == {}


def test_get_win_rate_returns_neutral_for_none_tracker() -> None:
    assert _get_win_rate("BTCUSDT", None) == 0.5


def test_get_win_rate_reflects_real_track_record(tmp_path, monkeypatch) -> None:
    """Live bug fixed 2026-07-16: _get_win_rate() called perf_tracker.symbol_info(),
    a method that doesn't exist on SymbolPerformanceTracker -- always raised
    AttributeError, silently caught, always returning the neutral 0.5 default
    regardless of any real trade history. This is the single heaviest-weighted
    factor (0.30) in SymbolRotator's own scoring formula, so it contributed zero
    actual differentiation between symbols. Fixed to call quality_score(), which
    already exists and returns exactly the rolling win-rate this function wants.
    """
    # Isolate from the real, live bot's logs/symbol_perf.json -- SymbolPerformanceTracker
    # takes no state_path override, so redirect the module-level constant instead of
    # ever writing test data (a prior version of this test did, by accident) into
    # production state.
    monkeypatch.setattr(perf_tracker_module, "_STATE_PATH", str(tmp_path / "symbol_perf.json"))

    tracker = SymbolPerformanceTracker()
    # 4 wins, 1 loss -- above _MIN_TRADES(3), well above neutral.
    for pnl in (1.0, 1.0, 1.0, 1.0, -1.0):
        tracker.record_trade("WINNERUSDT", pnl)

    win_rate = _get_win_rate("WINNERUSDT", tracker)
    assert win_rate == tracker.quality_score("WINNERUSDT")
    assert win_rate > 0.6, f"expected a real win-rate reflecting the 4/5 track record, got {win_rate}"


def test_get_win_rate_survives_a_tracker_without_the_expected_method() -> None:
    """Fail-open, not fail-crash, if the tracker ever lacks quality_score() for any
    reason (mirrors the original function's own defensive try/except intent)."""

    class _BrokenTracker:
        pass

    assert _get_win_rate("BTCUSDT", _BrokenTracker()) == 0.5
