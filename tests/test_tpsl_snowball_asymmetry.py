from types import SimpleNamespace

from core.tp_sl_engine import TPSLEngine


def _build_cfg():
    return SimpleNamespace(
        TPSL_CHECK_INTERVAL=10,
        MAX_CONCURRENT_CLOSES=5,
        TPSL_FALLBACK_ATR_PCT=0.01,
        TPSL_DEBOUNCE_CLOSE_SEC=5.0,
        TPSL_PRICE_STALE_SEC=120.0,
        TPSL_OHLCV_STALE_SEC=300.0,
        TPSL_MIN_NOTIONAL_SAFETY=1.0,
        TPSL_STRATEGY="hybrid_atr_time",
        TPSL_CALC_MODEL="atr_pct",
        TPSL_STATUS_TIMEOUT_SEC=1.0,
        TPSL_PROFIT_AUDIT=False,
        TPSL_PROFIT_AUDIT_SEC=300.0,
        TPSL_TIME_EXIT_ENABLED=False,
        TPSL_RV_LOOKBACK=20,
        TPSL_VOL_LOW_ATR_PCT=0.0045,
        TPSL_VOL_HIGH_ATR_PCT=0.0150,
        TPSL_VOL_TARGET_ATR_PCT=0.0090,
        TPSL_DYNAMIC_RR_MIN=1.30,
        TPSL_DYNAMIC_RR_MAX=3.00,
        TP_ATR_MULT=1.4,
        SL_ATR_MULT=1.0,
        TARGET_RR_RATIO=1.8,
        TP_PCT_MIN=0.003,
        TP_PCT_MAX=0.20,
        SL_PCT_MIN=0.003,
        SL_PCT_MAX=0.05,
        EXIT_SLIPPAGE_BPS=5.0,
        TP_MIN_BUFFER_BPS=4.0,
        TP_SL_MIN_RR=1.3,
        TRAILING_ATR_MULT=1.5,
        RISK_PCT_PER_TRADE=0.01,
        TIER_B_RISK_PCT=0.005,
        TPSL_SNOWBALL_ASYMMETRY_ENABLED=True,
        TPSL_SPREAD_ADAPTIVE_ENABLED=True,
        TPSL_SPREAD_TIGHT_BPS=6.0,
        TPSL_SPREAD_HIGH_BPS=18.0,
        TPSL_SPREAD_EXTREME_BPS=45.0,
        TPSL_SPREAD_RR_BONUS_MAX=0.18,
        TPSL_SPREAD_RR_DISCOUNT_MAX=0.06,
        TPSL_SPREAD_TP_FLOOR_MULT=2.0,
        TPSL_ASYMMETRIC_TP_ENABLED=True,
        TPSL_ASYM_TP_TREND_BONUS=0.12,
        TPSL_ASYM_TP_HIGH_VOL_BONUS=0.08,
        TPSL_ASYM_TP_CHOP_DISCOUNT=0.08,
        TPSL_ASYM_TP_SENTIMENT_WEIGHT=0.06,
        TPSL_ASYM_TP_PHASE_GAP_WEIGHT=0.20,
        TPSL_ASYM_TP_PHASE_BONUS_CAP=0.12,
        TPSL_ASYM_TP_MIN_BIAS=0.92,
        TPSL_ASYM_TP_MAX_BIAS=1.35,
        TPSL_ASYM_TP_TIER_B_CAP=1.08,
        COMPOUNDING_TPSL_PHASE_PROFILES={
            "PHASE_1_SEED": {"tp_mult": 1.00, "sl_mult": 1.00, "rr_bonus": 0.00},
            "PHASE_4_SNOWBALL": {"tp_mult": 1.25, "sl_mult": 0.90, "rr_bonus": 0.25},
        },
    )


def _build_shared_state(phase_name: str, bid_ask=None):
    best_bid_ask = {}
    if bid_ask is not None:
        best_bid_ask["BTCUSDT"] = {"bid": float(bid_ask[0]), "ask": float(bid_ask[1])}
    return SimpleNamespace(
        open_trades={},
        latest_prices={},
        sentiment_score={"BTCUSDT": 0.0},
        volatility_state={},
        dynamic_config={"COMPOUNDING_PHASE": phase_name},
        market_data={"BTCUSDT": {"atr": 20.0, "5m": {"atr": 20.0, "ohlcv": []}}},
        metrics={"total_trades_executed": 5},
        total_equity=1000.0,
        symbol_filters={},
        best_bid_ask=best_bid_ask,
    )


def test_tpsl_phase4_is_more_asymmetric_for_growth():
    cfg = _build_cfg()

    engine_p1 = TPSLEngine(
        shared_state=_build_shared_state("PHASE_1_SEED"),
        config=cfg,
        execution_manager=SimpleNamespace(),
    )
    tp1, sl1 = engine_p1.calculate_tp_sl("BTCUSDT", 1000.0, tier="A")

    engine_p4 = TPSLEngine(
        shared_state=_build_shared_state("PHASE_4_SNOWBALL"),
        config=cfg,
        execution_manager=SimpleNamespace(),
    )
    tp4, sl4 = engine_p4.calculate_tp_sl("BTCUSDT", 1000.0, tier="A")

    assert tp1 is not None and sl1 is not None
    assert tp4 is not None and sl4 is not None

    tp_dist_p1 = float(tp1) - 1000.0
    tp_dist_p4 = float(tp4) - 1000.0
    sl_dist_p1 = 1000.0 - float(sl1)
    sl_dist_p4 = 1000.0 - float(sl4)

    # Snowball phase should target bigger upside with tighter downside.
    assert tp_dist_p4 > tp_dist_p1
    assert sl_dist_p4 < sl_dist_p1


def test_phase2_tp_boost_increases_tp_without_widening_sl():
    base_cfg = _build_cfg()
    base_cfg.COMPOUNDING_TPSL_PHASE_PROFILES = {
        "PHASE_2_TRACTION": {"tp_mult": 1.40, "sl_mult": 0.95, "rr_bonus": 0.12},
    }
    base_cfg.TP_PHASE_MULTIPLIERS = {"PHASE_2_TRACTION": 1.40}
    base_cfg.SL_PHASE_MULTIPLIERS = {"PHASE_2_TRACTION": 0.95}

    boost_cfg = _build_cfg()
    boost_cfg.COMPOUNDING_TPSL_PHASE_PROFILES = {
        "PHASE_2_TRACTION": {"tp_mult": 1.54, "sl_mult": 0.95, "rr_bonus": 0.16},
    }
    boost_cfg.TP_PHASE_MULTIPLIERS = {"PHASE_2_TRACTION": 1.54}
    boost_cfg.SL_PHASE_MULTIPLIERS = {"PHASE_2_TRACTION": 0.95}

    state = _build_shared_state("PHASE_2_TRACTION")
    entry = 1000.0
    tp1, sl1 = TPSLEngine(shared_state=state, config=base_cfg, execution_manager=SimpleNamespace()).calculate_tp_sl("BTCUSDT", entry, tier="A")
    tp2, sl2 = TPSLEngine(shared_state=_build_shared_state("PHASE_2_TRACTION"), config=boost_cfg, execution_manager=SimpleNamespace()).calculate_tp_sl("BTCUSDT", entry, tier="A")

    assert tp1 is not None and sl1 is not None and tp2 is not None and sl2 is not None
    assert float(tp2) > float(tp1)
    assert abs(float(sl2) - float(sl1)) < 1e-9


def test_spread_aware_tp_expands_when_spread_is_wide():
    cfg = _build_cfg()
    cfg.TPSL_SPREAD_TIGHT_BPS = 3.0
    cfg.TPSL_SPREAD_HIGH_BPS = 10.0
    cfg.TPSL_SPREAD_EXTREME_BPS = 40.0
    cfg.TPSL_SPREAD_RR_BONUS_MAX = 0.25
    cfg.TPSL_SPREAD_RR_DISCOUNT_MAX = 0.0

    entry = 1000.0
    tp_tight, sl_tight = TPSLEngine(
        shared_state=_build_shared_state("PHASE_1_SEED", bid_ask=(999.95, 1000.05)),
        config=cfg,
        execution_manager=SimpleNamespace(),
    ).calculate_tp_sl("BTCUSDT", entry, tier="A")

    tp_wide, sl_wide = TPSLEngine(
        shared_state=_build_shared_state("PHASE_1_SEED", bid_ask=(998.0, 1002.0)),
        config=cfg,
        execution_manager=SimpleNamespace(),
    ).calculate_tp_sl("BTCUSDT", entry, tier="A")

    assert tp_tight is not None and sl_tight is not None
    assert tp_wide is not None and sl_wide is not None
    assert float(tp_wide) > float(tp_tight)
    assert abs(float(sl_wide) - float(sl_tight)) < 1e-9


def test_spread_floor_enforces_minimum_tp_distance():
    cfg = _build_cfg()
    cfg.TPSL_SPREAD_TP_FLOOR_MULT = 4.0
    cfg.TP_PCT_MIN = 0.001
    cfg.SL_PCT_MIN = 0.001
    cfg.TP_ATR_MULT = 0.8
    cfg.SL_ATR_MULT = 0.8

    state = _build_shared_state("PHASE_1_SEED", bid_ask=(999.0, 1001.0))
    state.market_data["BTCUSDT"]["atr"] = 1.0
    state.market_data["BTCUSDT"]["5m"]["atr"] = 1.0

    entry = 1000.0
    tp, _sl = TPSLEngine(
        shared_state=state,
        config=cfg,
        execution_manager=SimpleNamespace(),
    ).calculate_tp_sl("BTCUSDT", entry, tier="A")

    assert tp is not None
    tp_dist = float(tp) - entry
    expected_floor = abs(1001.0 - 999.0) * 4.0
    assert tp_dist >= expected_floor - 1e-9


def test_asymmetric_tp_bias_expands_upside_without_widening_sl():
    cfg_base = _build_cfg()
    cfg_base.TPSL_ASYMMETRIC_TP_ENABLED = False

    cfg_asym = _build_cfg()
    cfg_asym.TPSL_ASYMMETRIC_TP_ENABLED = True
    cfg_asym.TPSL_ASYM_TP_TREND_BONUS = 0.28
    cfg_asym.TPSL_ASYM_TP_HIGH_VOL_BONUS = 0.20
    cfg_asym.TPSL_ASYM_TP_SENTIMENT_WEIGHT = 0.10
    cfg_asym.TPSL_ASYM_TP_CHOP_DISCOUNT = 0.0
    cfg_asym.TPSL_ASYM_TP_PHASE_GAP_WEIGHT = 0.35

    state_base = _build_shared_state("PHASE_4_SNOWBALL", bid_ask=(999.95, 1000.05))
    state_base.volatility_state["BTCUSDT"] = "trend"
    state_base.sentiment_score["BTCUSDT"] = 0.90

    state_asym = _build_shared_state("PHASE_4_SNOWBALL", bid_ask=(999.95, 1000.05))
    state_asym.volatility_state["BTCUSDT"] = "trend"
    state_asym.sentiment_score["BTCUSDT"] = 0.90

    entry = 1000.0
    tp_base, sl_base = TPSLEngine(
        shared_state=state_base,
        config=cfg_base,
        execution_manager=SimpleNamespace(),
    ).calculate_tp_sl("BTCUSDT", entry, tier="A")

    tp_asym, sl_asym = TPSLEngine(
        shared_state=state_asym,
        config=cfg_asym,
        execution_manager=SimpleNamespace(),
    ).calculate_tp_sl("BTCUSDT", entry, tier="A")

    assert tp_base is not None and sl_base is not None
    assert tp_asym is not None and sl_asym is not None
    assert float(tp_asym) > float(tp_base)
    assert abs(float(sl_asym) - float(sl_base)) < 1e-9
