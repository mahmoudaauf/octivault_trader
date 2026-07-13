from __future__ import annotations

import core_engine.native.symbol_rotator as rotator_module
from core_engine.native.shared_state import NativeSharedState
from core_engine.native.symbol_rotator import SymbolRotator


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
