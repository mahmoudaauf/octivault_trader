"""
Regression test for a real bug found 2026-07-14: gate_9's global buy-pace
history (_global_buy_history) was never persisted to disk, unlike its sibling
_global_sl_history -- so it silently reset to empty on every process restart,
letting a fresh burst of up to max_buys_in_window BUYs through immediately
post-restart on top of whatever already happened just before the restart.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from core_engine.native.arbitration_engine import NativeArbitrationEngine
from core_engine.native.shared_state import NativeSharedState


def _make_arb(state_path):
    ss = NativeSharedState()
    de = MagicMock()
    de.min_notional_usdt = 10.0
    de.max_concurrent_positions = 3
    de._is_slot_blocking_position = MagicMock(return_value=False)
    de._resolve_mode = MagicMock(return_value={"max_positions": 3})
    engine = NativeArbitrationEngine(shared_state=ss, decision_engine=de)
    # __init__ calls _load_streak_state() against the default logs/arb_state.json
    # path before we can override it -- point at the test path and reload so
    # the "restart" simulation actually reads from the right file.
    engine._arb_state_path = str(state_path)
    engine._load_streak_state()
    return engine


def test_global_buy_history_survives_a_restart(tmp_path):
    state_path = tmp_path / "arb_state.json"

    first = _make_arb(state_path)
    first.record_buy("BTCUSDT")
    first.record_buy("ETHUSDT")
    assert len(first._global_buy_history) == 2

    restarted = _make_arb(state_path)
    assert len(restarted._global_buy_history) == 2


def test_global_buy_history_prunes_entries_older_than_2h_on_restore(tmp_path):
    import json
    import time

    state_path = tmp_path / "arb_state.json"
    now = time.time()
    state_path.write_text(json.dumps({
        "global_buy_history": [now - 10000, now - 100],  # one stale (>2h), one fresh
    }))

    engine = _make_arb(state_path)
    assert len(engine._global_buy_history) == 1


def test_global_buy_history_absent_from_disk_defaults_empty(tmp_path):
    state_path = tmp_path / "arb_state.json"
    state_path.write_text("{}")
    engine = _make_arb(state_path)
    assert engine._global_buy_history == []
