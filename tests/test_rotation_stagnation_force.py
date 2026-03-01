import time
from types import SimpleNamespace

from core.rotation_authority import RotationExitAuthority


def _build_cfg():
    return SimpleNamespace(
        ROTATION_BASE_ALPHA_GAP=0.005,
        ROTATION_WINNER_PROTECTION_PNL=0.002,
        ROTATION_WINNER_EXTRA_ALPHA=0.03,
        MAX_HOLD_SEC=60.0,
        STAGNATION_HOLD_MULT=1.0,
        STAGNATION_AGE_SEC=60.0,
        STAGNATION_PNL_THRESHOLD=0.0025,
        STAGNATION_STREAK_LIMIT=2,
        STAGNATION_CONTINUATION_MIN_SCORE=0.65,
        STAGNATION_FORCE_ROTATION_ENABLED=True,
        STAGNATION_FORCE_ROTATION_CONSEC_CYCLES=2,
        STAGNATION_FORCE_ROTATION_MIN_AGE_MULT=1.0,
        STAGNATION_FORCE_ROTATION_PNL_BAND=0.0025,
        STAGNATION_FORCE_ROTATION_SELL_FRACTION=0.40,
    )


def _build_pos():
    old_ts = time.time() - 3600.0
    return {
        "BTCUSDT": {
            "symbol": "BTCUSDT",
            "entry_time": old_ts,
            "opened_at": old_ts,
            "unrealized_pnl_pct": 0.0001,
            "state": "OPEN",
        }
    }


def test_stagnation_forced_rotation_triggers_after_consecutive_cycles():
    logger = SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None, debug=lambda *a, **k: None)
    rea = RotationExitAuthority(
        logger=logger,
        config=_build_cfg(),
        shared_state=SimpleNamespace(is_cold_bootstrap=lambda: False),
    )

    # Cycle 1: streak warms up, no forced action yet.
    res1 = rea.authorize_stagnation_exit(_build_pos(), current_mode="SAFE")
    assert res1 is None

    # Cycle 2: streak reaches threshold -> forced rotation signal.
    res2 = rea.authorize_stagnation_exit(_build_pos(), current_mode="SAFE")
    assert res2 is not None
    assert res2.get("reason") == "FORCED_ROTATION_STAGNATION"
    assert res2.get("tag") == "meta-rotation_authority"
    assert bool(res2.get("_is_rotation")) is True
    assert bool(res2.get("_forced")) is True
    assert bool(res2.get("_stagnation_force")) is True
    assert int(res2.get("_stagnation_streak", 0)) >= 2
    assert abs(float(res2.get("target_fraction")) - 0.40) < 1e-6


def test_forced_rotation_disabled_during_cold_bootstrap():
    logger = SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None, debug=lambda *a, **k: None)
    rea = RotationExitAuthority(
        logger=logger,
        config=_build_cfg(),
        shared_state=SimpleNamespace(is_cold_bootstrap=lambda: True),
    )

    res1 = rea.authorize_stagnation_exit(_build_pos(), current_mode="SAFE")
    res2 = rea.authorize_stagnation_exit(_build_pos(), current_mode="SAFE")
    assert res1 is None
    assert res2 is None


def test_forced_rotation_skips_when_continuation_is_strong():
    logger = SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None, debug=lambda *a, **k: None)
    rea = RotationExitAuthority(
        logger=logger,
        config=_build_cfg(),
        shared_state=SimpleNamespace(is_cold_bootstrap=lambda: False),
    )
    pos = _build_pos()
    pos["BTCUSDT"]["continuation_score"] = 0.90
    res1 = rea.authorize_stagnation_exit(pos, current_mode="SAFE")
    res2 = rea.authorize_stagnation_exit(pos, current_mode="SAFE")
    assert res1 is None
    assert res2 is None


def test_stagnation_disabled_during_startup_grace():
    logger = SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None, debug=lambda *a, **k: None)
    # Simulate very recent startup (within grace window)
    recent_start = time.time() - (5 * 60)  # started 5 minutes ago
    cfg = _build_cfg()
    # Ensure grace window is larger than elapsed
    setattr(cfg, "STARTUP_STAGNATION_GRACE_MINUTES", 30)
    shared = SimpleNamespace(is_cold_bootstrap=lambda: False, _start_time_unix=recent_start, metrics={"startup_time": recent_start})
    rea = RotationExitAuthority(logger=logger, config=cfg, shared_state=shared)

    # Even though positions are old, the startup grace prevents purge
    res1 = rea.authorize_stagnation_exit(_build_pos(), current_mode="SAFE")
    assert res1 is None
    res2 = rea.authorize_stagnation_exit(_build_pos(), current_mode="SAFE")
    assert res2 is None
