from __future__ import annotations

import json
from pathlib import Path

from core_engine.native.runtime_state import NativeRuntimeStateExporter, load_runtime_state
from core_engine.native.shared_state import NativeSharedState


def test_runtime_state_export_and_restore(tmp_path: Path) -> None:
    state = NativeSharedState()
    state.update_balance_map({"USDT": 123.45, "BTC": 0.01})
    state.update_nav(150.0)
    state.update_price("BTCUSDT", 65000.0)
    state.update_position("BTCUSDT", qty=0.01, entry=60000.0, current=65000.0)
    state.set_exchange_throttle(True, reason="418 ban", until_ts=9999999999.0)

    out = tmp_path / "runtime_state.json"
    exp = NativeRuntimeStateExporter(state, out, interval_sec=1.0)
    exp._write_snapshot()

    payload = json.loads(out.read_text())
    assert payload["nav_usdt"] == 150.0
    assert payload["balance"]["USDT"] == 123.45
    assert payload["exchange_throttled"] is True
    assert payload["last_known_good_account_state"]["nav_usdt"] == 150.0

    restored = NativeSharedState()
    assert load_runtime_state(restored, out) is True
    assert restored.nav_usdt == 150.0
    assert restored.balance["USDT"] == 123.45
    assert restored.prices["BTCUSDT"] == 65000.0
    assert "BTCUSDT" in restored._last_tick_timestamps
    assert restored.exchange_throttled is True
    assert restored.get_position("BTCUSDT") is not None


def test_runtime_state_restore_prefers_last_known_good_when_current_is_zero_and_throttled(
    tmp_path: Path,
) -> None:
    out = tmp_path / "runtime_state.json"
    payload = {
        "ts": 2.0,
        "nav_usdt": 0.0,
        "free_balance_usdt": 0.0,
        "invested_capital_usdt": 0.0,
        "balance": {},
        "prices": {"BTCUSDT": 65000.0},
        "positions": {},
        "session_anchor_nav": 0.0,
        "exchange_throttled": True,
        "exchange_throttle_reason": "418 ban",
        "exchange_throttle_until_ts": 9999999999.0,
        "metrics": {},
        "last_known_good_account_state": {
            "ts": 1.0,
            "nav_usdt": 222.0,
            "free_balance_usdt": 111.0,
            "invested_capital_usdt": 111.0,
            "balance": {"USDT": 111.0},
            "session_anchor_nav": 200.0,
        },
    }
    out.write_text(json.dumps(payload))

    restored = NativeSharedState()
    assert load_runtime_state(restored, out) is True
    assert restored.nav_usdt == 222.0
    assert restored.free_balance_usdt == 111.0
    assert restored.balance["USDT"] == 111.0
    # session_anchor_nav is intentionally zeroed on restore — main.py sets it from live NAV
    # on the first cycle so phantom drawdowns from prior sessions don't lock the OFC floor.
    assert restored.session_anchor_nav == 0.0
    assert restored.exchange_throttled is True
