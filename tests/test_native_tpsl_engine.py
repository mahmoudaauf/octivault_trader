from __future__ import annotations

import time

from core_engine.native.shared_state import NativeSharedState
from core_engine.native.tp_sl_engine import NativeTPSLEngine


class _Cfg:
    TP_ATR_MULT = 1.5
    SL_ATR_MULT = 1.5
    TARGET_RISK_PCT = 2.0
    ATR_LOOKBACK = 14
    MIN_ATR_PCT = 0.001  # very small floor so ATR-based SL tests control the value precisely
    TPSL_VOL_ADAPTATION_ENABLED = True
    VOL_PRESSURE_SCALE = 0.35
    MIN_NOTIONAL_SAFETY = 10.0
    TPSL_AUTO_ARM_ENABLED = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _engine_with_regime(regime: str) -> tuple[NativeTPSLEngine, NativeSharedState]:
    state = NativeSharedState()
    state.metrics = {"market_regime": regime}
    state.prices = {"TESTUSDT": 100.0}
    engine = NativeTPSLEngine(state, _Cfg())
    engine._startup_grace_until = 0.0  # disable grace period in all tests
    return engine, state


def _make_position(entry: float, tp: float, sl: float) -> dict:
    return {"symbol": "TESTUSDT", "qty": 1.0, "entry_price": entry, "tp": tp, "sl": sl}


# ---------------------------------------------------------------------------
# ATR-based SL tests
# ---------------------------------------------------------------------------


def test_calculate_tp_sl_uses_atr_for_slow_mover() -> None:
    """Low-ATR symbol should produce SL near the 1.0% floor."""
    state = NativeSharedState()
    # Inject a small ATR directly into shared_state.market_data
    state.market_data = {"SLOWUSDT": {"atr": 0.5}}  # 0.5 absolute on a $100 entry = 0.5% ATR
    state.prices = {"SLOWUSDT": 100.0}
    engine = NativeTPSLEngine(state, _Cfg())

    tp, sl = engine.calculate_tp_sl("SLOWUSDT", 100.0)

    sl_pct = (100.0 - sl) / 100.0
    # ATR pct = 0.5/100 = 0.5%, SL_ATR_MULT=1.5 → 0.75% → clamped to floor 1.0%
    assert abs(sl_pct - 0.010) < 1e-9, f"Expected SL ~1.0%, got {sl_pct*100:.3f}%"
    # TP = max(sl_pct*2.0, atr*1.5) = max(2.0%, 0.75%) = 2.0%, floored at 1.5% → 2.0%
    tp_pct = (tp - 100.0) / 100.0
    assert abs(tp_pct - 0.020) < 1e-9, f"Expected TP ~2.0%, got {tp_pct*100:.3f}%"


def test_calculate_tp_sl_uses_atr_for_volatile_mover() -> None:
    """High-ATR symbol should produce SL near the 2.5% ceiling."""
    state = NativeSharedState()
    state.market_data = {"VOLAUSDT": {"atr": 2.0}}  # 2.0 on $100 = 2.0% ATR
    state.prices = {"VOLAUSDT": 100.0}
    engine = NativeTPSLEngine(state, _Cfg())

    tp, sl = engine.calculate_tp_sl("VOLAUSDT", 100.0)

    sl_pct = (100.0 - sl) / 100.0
    # ATR pct = 2.0/100 = 2.0%, SL_ATR_MULT=1.5 → 3.0% → clamped to ceiling 2.5%
    assert abs(sl_pct - 0.025) < 1e-9, f"Expected SL ~2.5%, got {sl_pct*100:.3f}%"


def test_calculate_tp_sl_medium_atr_between_floor_and_ceiling() -> None:
    """Medium-ATR symbol should produce SL proportional to ATR, unclamped."""
    state = NativeSharedState()
    state.market_data = {"MEDUSDT": {"atr": 1.0}}  # 1.0% ATR on $100 entry
    state.prices = {"MEDUSDT": 100.0}
    engine = NativeTPSLEngine(state, _Cfg())

    tp, sl = engine.calculate_tp_sl("MEDUSDT", 100.0)

    sl_pct = (100.0 - sl) / 100.0
    # ATR pct = 1.0%, SL_ATR_MULT=1.5 → 1.5% → within [1.0%, 2.5%] → unclamped
    assert abs(sl_pct - 0.015) < 1e-9, f"Expected SL ~1.5%, got {sl_pct*100:.3f}%"


def test_calculate_tp_sl_fallback_no_data_reasonable_sl() -> None:
    """When no price/candle data is available, SL should stay within bounds."""
    state = NativeSharedState()
    # No market_data, no klines, no prices → _compute_atr returns 0 → min_atr kicks in
    engine = NativeTPSLEngine(state, _Cfg())

    tp, sl = engine.calculate_tp_sl("NOUSDT", 100.0)

    sl_pct = (100.0 - sl) / 100.0
    assert (
        engine._SL_FLOOR_PCT <= sl_pct <= engine._SL_CEILING_PCT
    ), f"SL {sl_pct*100:.2f}% out of [{engine._SL_FLOOR_PCT*100}%, {engine._SL_CEILING_PCT*100}%]"
    # TP must be at least 2:1 vs SL and within [1.5%, 6%]
    tp_pct = (tp - 100.0) / 100.0
    assert 0.015 <= tp_pct <= 0.06, f"TP {tp_pct*100:.2f}% outside [1.5%, 6%]"
    assert tp_pct >= sl_pct * 2.0 - 1e-9, f"TP {tp_pct*100:.2f}% < 2x SL {sl_pct*100:.2f}%"


def test_calculate_tp_sl_price_fallback_gives_reasonable_sl() -> None:
    """Price-only fallback (no candles) produces ATR=0.8% → SL=1.2%."""
    state = NativeSharedState()
    state.prices = {"PRICEUSDT": 100.0}
    # no market_data / klines → falls through to price fallback 0.8%
    engine = NativeTPSLEngine(state, _Cfg())

    tp, sl = engine.calculate_tp_sl("PRICEUSDT", 100.0)

    sl_pct = (100.0 - sl) / 100.0
    # atr = 100 * 0.008 = 0.8%, SL_ATR_MULT=1.5 → 1.2% → within bounds
    assert abs(sl_pct - 0.012) < 1e-9, f"Expected SL ~1.2%, got {sl_pct*100:.3f}%"


# ---------------------------------------------------------------------------
# Regime-aware trailing stop tests
# ---------------------------------------------------------------------------


def test_trailing_stop_uptrend_activates_at_2pct() -> None:
    """In UPTREND, trailing activates at +2.0% and fires 1.0% below peak.

    For trailing to fire: peak * (1 - distance) must stay above activation threshold.
    Constraint: peak/entry >= (1 + activation) / (1 - distance) = 1.02/0.99 = 1.0303.
    We use peak=105 (5% above entry) so trail=103.95 is above activation floor 102.0.
    """
    engine, state = _engine_with_regime("UPTREND")
    entry = 100.0
    engine._entry_prices["TESTUSDT"] = entry
    engine._entry_timestamps["TESTUSDT"] = time.time() - 60

    pos = _make_position(entry, tp=108.0, sl=97.5)

    # Price at +1.9% — below activation threshold of 2.0% → trailing not yet active
    result = engine.check_triggers("TESTUSDT", pos, current_price=101.9)
    assert result is None, f"Should not activate below 2.0%, got {result}"

    # Price rises to +5% (peak=105), then drops to 103.9 (profit=3.9% >= 2.0% activation,
    # and 103.9 <= 105 * 0.99 = 103.95 → fires)
    engine._peak_prices["TESTUSDT"] = 105.0
    result = engine.check_triggers("TESTUSDT", pos, current_price=103.9)
    assert result == "TRAILING_STOP", f"Expected TRAILING_STOP, got {result}"


def test_trailing_stop_choppy_activates_at_0pt8pct() -> None:
    """In CHOPPY, trailing activates at +0.8% — faster lock-in.

    Constraint: peak/entry >= (1 + 0.008) / (1 - 0.004) = 1.008/0.996 = 1.01205.
    Peak=102 (2% above entry) → trail=101.592; current=101.5 satisfies both conditions.
    """
    engine, state = _engine_with_regime("CHOPPY")
    entry = 100.0
    engine._entry_prices["TESTUSDT"] = entry
    engine._entry_timestamps["TESTUSDT"] = time.time() - 60

    pos = _make_position(entry, tp=108.0, sl=97.5)

    # Price at +0.7% — below CHOPPY activation of 0.8% → no trailing
    result = engine.check_triggers("TESTUSDT", pos, current_price=100.7)
    assert result is None

    # Peak=102, current=101.5 (profit=1.5% >= 0.8%; 101.5 <= 102 * 0.996 = 101.592) → fires
    engine._peak_prices["TESTUSDT"] = 102.0
    result = engine.check_triggers("TESTUSDT", pos, current_price=101.5)
    assert result == "TRAILING_STOP", f"Expected TRAILING_STOP in CHOPPY, got {result}"


def test_trailing_stop_downtrend_activates_at_0pt6pct() -> None:
    """In DOWNTREND, trailing activates at +0.6% — survival mode.

    Constraint: peak/entry >= (1 + 0.006) / (1 - 0.003) = 1.006/0.997 = 1.00903.
    Peak=101.5 (1.5% above entry) → trail=101.196; current=101.0 satisfies both.
    """
    engine, state = _engine_with_regime("DOWNTREND")
    entry = 100.0
    engine._entry_prices["TESTUSDT"] = entry
    engine._entry_timestamps["TESTUSDT"] = time.time() - 60

    pos = _make_position(entry, tp=108.0, sl=97.5)

    # Peak=101.5, current=101.0 (profit=1.0% >= 0.6%; 101.0 <= 101.5 * 0.997 = 101.195) → fires
    engine._peak_prices["TESTUSDT"] = 101.5
    result = engine.check_triggers("TESTUSDT", pos, current_price=101.0)
    assert result == "TRAILING_STOP", f"Expected TRAILING_STOP in DOWNTREND, got {result}"


def test_trailing_stop_unknown_regime_uses_trending_default() -> None:
    """Unknown/empty regime falls back to TRENDING behaviour (1.5%, 0.8%).

    Constraint: peak/entry >= (1 + 0.015) / (1 - 0.008) = 1.015/0.992 = 1.02318.
    Peak=104 (4% above entry) → trail=103.168; current=103.1 satisfies both.
    """
    engine, state = _engine_with_regime("")
    entry = 100.0
    engine._entry_prices["TESTUSDT"] = entry
    engine._entry_timestamps["TESTUSDT"] = time.time() - 60

    pos = _make_position(entry, tp=108.0, sl=97.5)

    # +1.4% — below TRENDING activation of 1.5% → no trailing
    result = engine.check_triggers("TESTUSDT", pos, current_price=101.4)
    assert result is None

    # Peak=104, current=103.1 (profit=3.1% >= 1.5%; 103.1 <= 104 * 0.992 = 103.168) → fires
    engine._peak_prices["TESTUSDT"] = 104.0
    result = engine.check_triggers("TESTUSDT", pos, current_price=103.1)
    assert result == "TRAILING_STOP", f"Expected TRAILING_STOP for default regime, got {result}"


def test_all_regime_params_within_valid_bounds() -> None:
    """Every regime entry has activation >= distance (can't trail before activating)."""
    for regime, (activation, distance) in NativeTPSLEngine._REGIME_TRAIL_PARAMS.items():
        assert (
            activation > distance
        ), f"{regime}: activation ({activation}) must exceed distance ({distance})"
        assert activation >= 0.005, f"{regime}: activation too small ({activation})"
        assert distance >= 0.002, f"{regime}: distance too small ({distance})"


def test_persisted_sl_survives_restart_unchanged() -> None:
    """Best practice: persisted SL must be restored exactly, not recalculated.

    A position armed at entry with SL=98.0 should still have SL=98.0 after a
    simulated restart (_load_tpsl_state), even if ATR would produce a different
    value now.  The symbol must also be in _armed_symbols so auto_arm skips it.
    """
    import json
    import os
    import tempfile

    entry = 100.0
    original_sl = 98.0  # 2.0% SL set at entry
    original_tp = 108.0

    with tempfile.TemporaryDirectory() as tmpdir:
        state_path = os.path.join(tmpdir, "tpsl_state.json")

        # Simulate a saved state from a previous session
        saved = {
            "tp_levels": {"TESTUSDT": original_tp},
            "sl_levels": {"TESTUSDT": original_sl},
            "entry_timestamps": {"TESTUSDT": time.time() - 3600},
            "peak_prices": {"TESTUSDT": entry},
            "entry_prices": {"TESTUSDT": entry},
            "saved_at": time.time(),
        }
        with open(state_path, "w") as f:
            json.dump(saved, f)

        # Simulate restart: create a fresh engine pointing at the saved state file
        state = NativeSharedState()
        state.positions = {"TESTUSDT": {"symbol": "TESTUSDT", "qty": 1.0, "entry_price": entry}}
        state.prices = {"TESTUSDT": 101.0}
        # Inject candle data that would produce a DIFFERENT SL if recalculated
        state.market_data[("TESTUSDT", "1m")] = [
            {"time": i, "open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 1000}
            for i in range(20)
        ]  # ATR ~1.0% → would produce SL=98.5 if recalculated (different from 98.0)

        engine = NativeTPSLEngine(state, _Cfg())
        # Fully reset internal state so production logs/tpsl_state.json (loaded in __init__)
        # doesn't pollute this test — we want to test only our fixture file.
        engine._armed_symbols.clear()
        engine._tp_levels.clear()
        engine._sl_levels.clear()
        engine._entry_timestamps.clear()
        engine._peak_prices.clear()
        engine._entry_prices.clear()
        engine._tpsl_state_path = state_path
        engine._load_tpsl_state()  # restore from saved state

        # After _load_tpsl_state(), TESTUSDT must be in _armed_symbols so that
        # _auto_arm_existing_positions() skips recalculation for it.
        assert (
            "TESTUSDT" in engine._armed_symbols
        ), "Restored symbol must be in _armed_symbols so auto-arm skips recalculation"
        assert engine._sl_levels.get("TESTUSDT") == original_sl, (
            f"SL was modified on load: got {engine._sl_levels.get('TESTUSDT')}, "
            f"expected persisted {original_sl}"
        )
        assert engine._tp_levels.get("TESTUSDT") == original_tp
        assert engine._entry_prices.get("TESTUSDT") == entry


def test_sl_ceiling_never_exceeded_by_atr_formula() -> None:
    """Even with an extreme ATR, SL must never exceed the 2.5% ceiling."""
    state = NativeSharedState()
    state.market_data = {"EXTREMEUSDT": {"atr": 50.0}}  # 50% ATR on $100 entry
    state.prices = {"EXTREMEUSDT": 100.0}
    engine = NativeTPSLEngine(state, _Cfg())

    _, sl = engine.calculate_tp_sl("EXTREMEUSDT", 100.0)

    sl_pct = (100.0 - sl) / 100.0
    assert (
        sl_pct <= engine._SL_CEILING_PCT + 1e-9
    ), f"SL ceiling violated: {sl_pct*100:.2f}% > {engine._SL_CEILING_PCT*100}%"


def test_recalculate_aged_positions_tightens_quick_winner_under_nav_protection() -> None:
    state = NativeSharedState()
    now = time.time()
    state.positions = {
        "BTCUSDT": {
            "symbol": "BTCUSDT",
            "qty": 1.0,
            "entry_price": 100.0,
            "current_price": 101.2,
            "tp": 102.5,
            "sl": 98.5,
        }
    }
    state.prices = {"BTCUSDT": 101.2}
    state.position_setup_context = {
        "BTCUSDT": {
            "setup_family": "continuation",
            "regime": "trend",
            "confidence": 0.78,
            "entry_quality": 0.72,
            "entry_ts": now - (45 * 60),
        }
    }
    state.nav_protection_state = {
        "allow_tp_sl_adjustment": True,
        "protection_mode": "FLOATING_GAIN_PROTECTION",
        "suggested_actions": ["TIGHTEN_TP_SL", "PARTIAL_TAKE_PROFIT"],
    }

    engine = NativeTPSLEngine(state, _Cfg())
    engine._tpsl_state_path = "/tmp/test_tpsl_isolated.json"  # isolate from live state file
    engine._tp_levels.clear()
    engine._sl_levels.clear()
    engine._entry_timestamps.clear()
    engine._entry_timestamps["BTCUSDT"] = now - (45 * 60)

    updates = engine.recalculate_aged_positions()

    assert "BTCUSDT" in updates
    assert state.positions["BTCUSDT"]["tp"] < 102.5
    assert state.positions["BTCUSDT"]["sl"] > 100.0
    assert "protection" in updates["BTCUSDT"]["reason"]


def test_recalculate_aged_positions_does_not_tighten_fresh_flat_position() -> None:
    state = NativeSharedState()
    now = time.time()
    state.positions = {
        "ETHUSDT": {
            "symbol": "ETHUSDT",
            "qty": 1.0,
            "entry_price": 100.0,
            "current_price": 100.3,
            "tp": 102.0,
            "sl": 98.5,
        }
    }
    state.prices = {"ETHUSDT": 100.3}
    state.position_setup_context = {
        "ETHUSDT": {
            "setup_family": "continuation",
            "regime": "trend",
            "confidence": 0.75,
            "entry_quality": 0.70,
            "entry_ts": now - (10 * 60),
        }
    }
    state.nav_protection_state = {
        "allow_tp_sl_adjustment": True,
        "protection_mode": "FLOATING_GAIN_PROTECTION",
        "suggested_actions": ["TIGHTEN_TP_SL"],
    }

    engine = NativeTPSLEngine(state, _Cfg())
    engine._entry_timestamps["ETHUSDT"] = now - (10 * 60)

    updates = engine.recalculate_aged_positions()

    assert updates == {}
    assert state.positions["ETHUSDT"]["tp"] == 102.0
    assert state.positions["ETHUSDT"]["sl"] == 98.5


# ---------------------------------------------------------------------------
# Dynamic TP/SL — Position object support + regime-aware widening
# ---------------------------------------------------------------------------

from core_engine.native.shared_state import Position


def test_check_triggers_supports_position_objects():
    """check_triggers works with Position objects, not just dicts."""
    engine, state = _engine_with_regime("TRENDING")
    entry = 1.00
    tp = 1.025
    sl = 0.975
    engine._tp_levels["POSOBJ"] = tp
    engine._sl_levels["POSOBJ"] = sl
    engine._entry_timestamps["POSOBJ"] = time.time() - 100

    pos_obj = Position(symbol="POSOBJ", qty=10.0, entry_price=entry, mark_price=tp + 0.001)

    result = engine.check_triggers("POSOBJ", pos_obj, tp + 0.001)
    assert result == "TP_HIT", f"Expected TP_HIT, got {result}"


def test_check_triggers_position_object_sl():
    """SL triggers correctly on Position objects."""
    engine, state = _engine_with_regime("TRENDING")
    entry = 1.00
    tp = 1.025
    sl = 0.975
    engine._tp_levels["POSOBJ"] = tp
    engine._sl_levels["POSOBJ"] = sl
    engine._entry_timestamps["POSOBJ"] = time.time() - 100

    pos_obj = Position(symbol="POSOBJ", qty=10.0, entry_price=entry, mark_price=sl - 0.001)

    result = engine.check_triggers("POSOBJ", pos_obj, sl - 0.001)
    assert result == "SL_HIT", f"Expected SL_HIT, got {result}"


def test_recalculate_aged_positions_handles_position_objects():
    """recalculate_aged_positions doesn't skip Position objects."""
    state = NativeSharedState()
    state.metrics = {"market_regime": "TRENDING"}
    state.prices = {"BIOOBJ": 0.028}
    state.nav_protection_state = {}

    # Simulate a 3-hour old position as a Position object (after polling refresh)
    entry = 0.030
    pos_obj = Position(symbol="BIOOBJ", qty=660.0, entry_price=entry, mark_price=0.028)
    state.positions["BIOOBJ"] = pos_obj

    engine = NativeTPSLEngine(state, _Cfg())
    engine._startup_grace_until = 0.0
    engine._entry_timestamps["BIOOBJ"] = time.time() - (3 * 3600)
    original_tp = entry * 1.024
    engine._tp_levels["BIOOBJ"] = original_tp
    engine._sl_levels["BIOOBJ"] = entry * 0.961

    updates = engine.recalculate_aged_positions()

    # Should have processed and tightened (3h >= 2h threshold → tp=+1.5%)
    assert "BIOOBJ" in updates, f"Expected BIOOBJ in updates, got {updates}"
    assert updates["BIOOBJ"]["tp"] < original_tp, \
        f"TP should have tightened: {updates['BIOOBJ']['tp']:.6f} < {original_tp:.6f}"


def test_maybe_widen_tp_trending_regime():
    """Dynamic TP widening triggers for TRENDING regime on underwater position."""
    engine, state = _engine_with_regime("TRENDING")
    result = engine._maybe_widen_tp(
        symbol="BIOUSDT",
        entry_price=0.030,
        current_price=0.029,   # underwater
        current_tp=0.030 * 1.024,   # tight 2.4% TP
        current_sl=0.030 * 0.961,
        age_sec=3600,           # 1 hour old
        profit_pct=-0.033,
        regime="TRENDING",
    )
    assert result is not None, "Expected TP widening for TRENDING + underwater position"
    assert result["tp"] > 0.030 * 1.024, "New TP should be wider than original"
    assert "TRENDING" in result["reason"]


def test_maybe_widen_tp_no_widen_for_profitable():
    """Dynamic TP widening does NOT apply to profitable positions (trailing handles those)."""
    engine, state = _engine_with_regime("TRENDING")
    result = engine._maybe_widen_tp(
        symbol="BIOUSDT",
        entry_price=0.030,
        current_price=0.032,   # +6.7% profitable
        current_tp=0.030 * 1.06,
        current_sl=0.030 * 0.98,
        age_sec=3600,
        profit_pct=0.067,
        regime="TRENDING",
    )
    assert result is None, "Should not widen TP for profitable position"


def test_maybe_widen_tp_no_widen_for_downtrend():
    """Dynamic TP widening is suppressed in DOWNTREND."""
    engine, state = _engine_with_regime("DOWNTREND")
    result = engine._maybe_widen_tp(
        symbol="BIOUSDT",
        entry_price=0.030,
        current_price=0.028,
        current_tp=0.030 * 1.024,
        current_sl=0.030 * 0.975,
        age_sec=3600,
        profit_pct=-0.067,
        regime="DOWNTREND",
    )
    assert result is None, "Should not widen TP in DOWNTREND"


def test_maybe_widen_tp_uptrend_sets_10pct():
    """UPTREND regime sets TP to 10% above entry."""
    engine, state = _engine_with_regime("UPTREND")
    entry = 0.056
    result = engine._maybe_widen_tp(
        symbol="WLFIUSDT",
        entry_price=entry,
        current_price=entry * 0.99,
        current_tp=entry * 1.024,
        current_sl=entry * 0.975,
        age_sec=1800,
        profit_pct=-0.01,
        regime="UPTREND",
    )
    assert result is not None
    assert abs(result["tp"] - entry * 1.10) < 1e-8, \
        f"UPTREND TP should be +10%, got {result['tp']:.6f}"


def test_force_exit_triggers_on_aged_position_object():
    """TIME_FORCE_EXIT fires on Position objects after _AGE_FORCE_EXIT_SEC (3h
    default, TPSL_FORCE_EXIT_H) when unprofitable. Layer 2 (time-based force
    exit) is checked before Layer 1 (static TP/SL) in check_triggers(), so once
    the age threshold is crossed this fires ahead of SL_HIT even though the
    price here has also breached the static SL level.
    """
    engine, state = _engine_with_regime("CHOPPY")
    entry = 0.030
    current = 0.028  # -6.7% unprofitable

    pos_obj = Position(symbol="STALE", qty=660.0, entry_price=entry, mark_price=current)
    engine._tp_levels["STALE"] = entry * 1.024
    engine._sl_levels["STALE"] = entry * 0.975
    engine._entry_timestamps["STALE"] = time.time() - (3.5 * 3600)  # past the 3h default

    result = engine.check_triggers("STALE", pos_obj, current)
    assert result == "TIME_FORCE_EXIT", f"Expected TIME_FORCE_EXIT, got {result}"


def test_hydrated_position_with_no_tp_sl_gets_auto_armed_on_first_check(): # noqa: E501
    """Remediation item #12: position_hydration_engine.py's apply_to_shared_state()
    writes tp=None/sl=None for any restart-recovered position it couldn't restore
    a prior TP/SL for (the common real-restart case, since shared_state.positions
    is empty before hydration runs). This must not leave the position permanently
    unprotected — check_triggers()'s auto-arm branch (this file's "AUTO-ARM" log
    line) must arm it on the very first call once the startup grace period ends.
    """
    engine, state = _engine_with_regime("CHOPPY")
    state.positions["BTCUSDT"] = {
        "symbol": "BTCUSDT", "qty": 1.0, "entry_price": 100.0,
        "current_price": 100.0, "mark_price": 100.0,
        "tp": None, "sl": None, "lifecycle": "ACTIVE",
    }
    pos = state.positions["BTCUSDT"]

    assert engine._tp_levels.get("BTCUSDT", 0) == 0  # not armed yet
    engine.check_triggers("BTCUSDT", pos, current_price=100.0)

    assert engine._tp_levels.get("BTCUSDT", 0) > 0, "hydrated position was not auto-armed"
    assert engine._sl_levels.get("BTCUSDT", 0) > 0, "hydrated position was not auto-armed"
    assert engine._tp_levels["BTCUSDT"] > 100.0


def test_check_triggers_prefers_armed_entry_price_over_corrupted_position_dict():
    """Live incident 2026-07-16: polling_coordinator's fill-reconciliation averaged
    together fills from a previous, already-closed round trip on the same symbol,
    inflating position["entry_price"] far from the real fill price (e.g. DASH
    34.78 -> 36.93, AIXBT 0.01933 -> 0.03116). check_triggers() must use the price
    arm_position() actually recorded, not whatever polling_coordinator's averaging
    wrote into the position dict -- mirroring recalculate_aged_positions()'s
    already-correct preference for the same reason.

    Uses trailing-stop activation as the discriminator: profit_pct = (current -
    entry_price) / entry_price feeds directly into whether the trailing stop
    arms at all (RANGING: activates at +0.8%). Relative to the real entry (100)
    a small rally clears that bar; relative to the corrupted entry (150) the
    exact same price looks like a ~33% LOSS and never clears any positive
    activation threshold -- so the trailing stop only ever arms, and can only
    ever fire, if the real armed price was actually used.
    """
    engine, state = _engine_with_regime("RANGING")
    real_entry = 100.0
    engine.arm_position("TESTUSDT", real_entry)

    # Simulate polling_coordinator's corruption: position dict now claims a wildly
    # different entry price than what was actually armed.
    corrupted_entry = 150.0
    pos = {"symbol": "TESTUSDT", "qty": 1.0, "entry_price": corrupted_entry}

    # Step 1: rally to +1.5% (relative to the REAL entry) -- clears RANGING's 0.8%
    # trailing activation and should record a peak at 101.5.
    result_1 = engine.check_triggers("TESTUSDT", pos, current_price=101.5)
    assert result_1 is None, f"expected no exit yet on the rally leg, got {result_1!r}"
    assert engine._peak_prices["TESTUSDT"] == 101.5, (
        "trailing stop never activated -- entry_price resolution likely used the "
        "corrupted position-dict value (150) instead of the real armed price (100)"
    )

    # Step 2: retrace to 100.9 -- more than RANGING's 0.5% trail distance below the
    # 101.5 peak (100.9 <= 101.5*(1-0.005)=100.9925), should fire TRAILING_STOP.
    result_2 = engine.check_triggers("TESTUSDT", pos, current_price=100.9)
    assert result_2 == "TRAILING_STOP", (
        f"expected TRAILING_STOP on the retrace leg, got {result_2!r}"
    )
