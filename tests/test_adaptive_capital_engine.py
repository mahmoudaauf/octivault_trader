from types import SimpleNamespace

from core.adaptive_capital_engine import AdaptiveCapitalEngine


def _cfg():
    return SimpleNamespace(
        ADAPTIVE_CAPITAL_ENGINE_ENABLED=True,
        ADAPTIVE_PERF_REVIEW_SEC=0.0,
        ADAPTIVE_RISK_FRACTION_MIN=0.05,
        ADAPTIVE_RISK_FRACTION_MAX=0.35,
        ADAPTIVE_DRAWDOWN_SOFT_PCT=2.0,
        ADAPTIVE_DRAWDOWN_HARD_PCT=4.0,
        ADAPTIVE_HIGH_VOL_PCT=0.015,
        ADAPTIVE_LOW_VOL_PCT=0.004,
        ADAPTIVE_THROUGHPUT_LOW_RATIO=0.50,
        ADAPTIVE_IDLE_FREE_CAPITAL_PCT=0.60,
        ADAPTIVE_IDLE_TIME_SEC=1800.0,
        ADAPTIVE_WIN_STREAK_TRADES=3,
        ADAPTIVE_LOSS_STREAK_TRADES=3,
        ADAPTIVE_WIN_STREAK_RISK_BONUS=0.10,
        ADAPTIVE_LOSS_STREAK_RISK_PENALTY=0.18,
        ADAPTIVE_WIN_RATE_BONUS_THRESHOLD=0.60,
        ADAPTIVE_WIN_RATE_PENALTY_THRESHOLD=0.45,
        ADAPTIVE_WIN_RATE_BONUS=0.08,
        ADAPTIVE_WIN_RATE_PENALTY=0.10,
        ADAPTIVE_FEE_GROSS_THRESHOLD=0.35,
        ADAPTIVE_FEE_GROSS_BONUS=0.06,
        ADAPTIVE_ECON_MIN_NOTIONAL_MULT=1.20,
        ADAPTIVE_ECON_TARGET_PROFIT_PCT=0.004,
        ADAPTIVE_MIN_QUOTE_BUFFER_MULT=1.20,
        MIN_NET_PROFIT_AFTER_FEES=0.004,
        MAX_HOLD_TIME_SEC=1800.0,
        MAX_POSITION_EXPOSURE_PERCENTAGE=0.20,
    )


def _trade(ts: float, pnl: float, fee: float = 0.02, hold: float = 300.0):
    return {
        "ts": ts,
        "realized_delta": pnl,
        "fee_quote": fee,
        "hold_time_sec": hold,
    }


def test_risk_fraction_increases_with_wins_and_idle_capital():
    cfg = _cfg()
    engine = AdaptiveCapitalEngine(cfg)
    now = 1_000_000.0
    hist = [
        _trade(now - 3700, 0.20),
        _trade(now - 3600, 0.15),
        _trade(now - 3500, 0.18),
    ]
    decision = engine.evaluate(
        symbol="BTCUSDT",
        nav=167.0,
        free_capital=130.0,
        base_risk_fraction=0.20,
        volatility_pct=0.006,
        drawdown_pct=0.3,
        fee_bps=10.0,
        slippage_bps=10.0,
        min_notional=20.0,
        slot_utilization=0.25,
        throughput_per_hour=0.02,
        target_throughput_per_hour=0.20,
        trade_history=hist,
        now_ts=now,
    )
    assert decision.risk_fraction > 0.20
    assert decision.min_trade_quote >= 24.0
    assert "idle_capital_boost" in decision.reasons


def test_risk_fraction_decreases_with_losses_drawdown_and_vol():
    cfg = _cfg()
    engine = AdaptiveCapitalEngine(cfg)
    now = 2_000_000.0
    hist = [
        _trade(now - 200, -0.20),
        _trade(now - 150, -0.18),
        _trade(now - 100, -0.15),
    ]
    decision = engine.evaluate(
        symbol="ETHUSDT",
        nav=220.0,
        free_capital=90.0,
        base_risk_fraction=0.22,
        volatility_pct=0.03,
        drawdown_pct=5.0,
        fee_bps=10.0,
        slippage_bps=10.0,
        min_notional=20.0,
        slot_utilization=0.80,
        throughput_per_hour=0.25,
        target_throughput_per_hour=0.20,
        trade_history=hist,
        now_ts=now,
    )
    assert decision.risk_fraction < 0.22
    assert "drawdown_hard" in decision.reasons
    assert "high_vol_down" in decision.reasons


def test_economic_floor_is_nav_bounded():
    cfg = _cfg()
    engine = AdaptiveCapitalEngine(cfg)
    decision = engine.evaluate(
        symbol="BTCUSDT",
        nav=167.0,
        free_capital=120.0,
        base_risk_fraction=0.18,
        volatility_pct=0.008,
        drawdown_pct=0.5,
        fee_bps=10.0,
        slippage_bps=10.0,
        min_notional=20.0,
        slot_utilization=0.10,
        throughput_per_hour=0.10,
        target_throughput_per_hour=0.20,
        trade_history=[],
        now_ts=3_000_000.0,
    )
    # Floor should clear notional buffer but stay bounded by NAV position cap.
    assert decision.min_trade_quote >= 24.0
    assert decision.min_trade_quote <= (167.0 * 0.20) + 1e-9

