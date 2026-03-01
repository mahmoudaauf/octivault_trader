import pytest
from types import SimpleNamespace

from core.scaling import ScalingManager


class _ScalingSharedState:
    def __init__(self, total_equity: float = 750.0, bootstrap_equity: float = 300.0):
        self.dynamic_config = {}
        self.metrics = {"realized_pnl": 350.0, "drawdown_pct": 0.0}
        self.trade_history = [{"realized_delta": 12.0} for _ in range(25)]
        self.market_data = {"BTCUSDT": {"5m": {"atr": 22.0}}}
        self.latest_prices = {"BTCUSDT": 1000.0}
        self.risk_based_quote = {}
        self.total_equity = float(total_equity)
        self.bootstrap_equity = float(bootstrap_equity)
        self.emitted = []

    async def get_nav_quote(self):
        return float(self.total_equity)

    async def get_authoritative_reservation(self, _agent_name):
        return 0.0

    async def compute_min_entry_quote(self, _symbol, default_quote=0.0):
        return max(10.0, float(default_quote or 0.0))

    def emit_event(self, name, data):
        self.emitted.append((name, data))


class _ScalingExecutionManager:
    async def get_symbol_filters_cached(self, _symbol):
        return {"MIN_NOTIONAL": {"minNotional": "10.0"}}


class _ModeManager:
    def __init__(self, mode="NORMAL"):
        self._mode = mode

    def get_mode(self):
        return self._mode

    def get_envelope(self):
        return {"max_trade_usdt": 220.0, "max_positions": 3}


@pytest.mark.asyncio
async def test_growth_curve_phase_and_dynamic_quote_applied():
    cfg = SimpleNamespace(
        BASE_CAPITAL=400.0,
        COMPOUNDING_ENABLED=True,
        SCALING_EQUITY_TIERS=[],
        DEFAULT_PLANNED_QUOTE=80.0,
        MIN_TRADE_QUOTE=40.0,
        MAX_TRADE_QUOTE=250.0,
        MIN_NOTIONAL_USDT=10.0,
        BUY_HEADROOM_FACTOR=1.05,
        SCOUT_MIN_NOTIONAL=5.0,
        MAX_SPEND_PER_TRADE_USDT=250.0,
        RISK_PCT_PER_TRADE=0.01,
        TIER_B_RISK_PCT=0.005,
        SL_ATR_MULT=1.0,
        SL_PCT_MIN=0.003,
        SL_PCT_MAX=0.02,
        TPSL_VOL_TARGET_ATR_PCT=0.009,
        COMPOUNDING_GROWTH_CURVE_ENABLED=True,
        GROWTH_PHASE_THRESHOLDS=[
            {"name": "PHASE_1_SEED", "min_ratio": 0.0, "max_ratio": 1.25},
            {"name": "PHASE_4_SNOWBALL", "min_ratio": 2.50, "max_ratio": None},
        ],
        PHASE_SIZE_MULTIPLIERS={
            "PHASE_1_SEED": 0.90,
            "PHASE_4_SNOWBALL": 1.25,
        },
        PHASE_MAX_TRADE_CAP={
            "PHASE_1_SEED": 100.0,
            "PHASE_4_SNOWBALL": 220.0,
        },
        COMPOUNDING_MAX_DRAWDOWN_PCT=2.5,
        COMPOUNDING_MIN_POSITIVE_STREAK=0,
        COMPOUNDING_GROWTH_PHASES=[
            {
                "name": "PHASE_1_SEED",
                "min_equity": 0.0,
                "max_equity": 299.99,
                "quote_mult": 0.9,
                "risk_mult": 0.9,
                "min_quote": 40.0,
                "max_quote": 100.0,
            },
            {
                "name": "PHASE_4_SNOWBALL",
                "min_equity": 700.0,
                "max_equity": None,
                "quote_mult": 1.25,
                "risk_mult": 1.16,
                "min_quote": 90.0,
                "max_quote": 220.0,
            },
        ],
        DYNAMIC_POSITION_SIZING_ENABLED=True,
        DYNAMIC_SIZE_CONF_FLOOR_MULT=0.85,
        DYNAMIC_SIZE_CONF_CEIL_MULT=1.25,
        DYNAMIC_SIZE_VOL_FLOOR_MULT=0.70,
        DYNAMIC_SIZE_VOL_CEIL_MULT=1.35,
        DYNAMIC_SIZE_MOMENTUM_MULT=0.45,
        DYNAMIC_SIZE_BLEND_WEIGHT=0.65,
        DYNAMIC_SIZE_UPSIDE_CAP_MULT=2.25,
        DYNAMIC_SIZE_MOMENTUM_LOOKBACK_TRADES=20,
    )

    ss = _ScalingSharedState()
    logger = SimpleNamespace(info=lambda *a, **k: None, debug=lambda *a, **k: None, warning=lambda *a, **k: None)
    mgr = ScalingManager(
        shared_state=ss,
        execution_manager=_ScalingExecutionManager(),
        config=cfg,
        logger=logger,
        mode_manager=_ModeManager(),
    )

    sig = {"symbol": "BTCUSDT", "agent": "MLForecaster", "action": "BUY", "confidence": 0.91}
    quote = await mgr.calculate_planned_quote("BTCUSDT", sig)

    assert ss.dynamic_config.get("compounding_phase") == "PHASE_4_SNOWBALL"
    assert ss.dynamic_config.get("COMPOUNDING_PHASE") == "PHASE_4_SNOWBALL"
    assert abs(float(ss.dynamic_config.get("COMPOUNDING_PHASE_RATIO", 0.0)) - 2.5) < 1e-6
    assert any(ev[0] == "CompoundingPhaseChanged" for ev in ss.emitted)
    assert quote >= 90.0
    assert quote <= 220.0
    assert quote > 80.0


@pytest.mark.asyncio
async def test_phase_change_increases_quote_deterministically():
    cfg = SimpleNamespace(
        BASE_CAPITAL=400.0,
        COMPOUNDING_ENABLED=True,
        COMPOUNDING_GROWTH_CURVE_ENABLED=True,
        GROWTH_PHASE_THRESHOLDS=[
            {"name": "PHASE_1_SEED", "min_ratio": 0.0, "max_ratio": 1.25},
            {"name": "PHASE_4_SNOWBALL", "min_ratio": 2.50, "max_ratio": None},
        ],
        PHASE_SIZE_MULTIPLIERS={"PHASE_1_SEED": 0.90, "PHASE_4_SNOWBALL": 1.25},
        PHASE_MAX_TRADE_CAP={"PHASE_1_SEED": 100.0, "PHASE_4_SNOWBALL": 220.0},
        COMPOUNDING_GROWTH_PHASES=[
            {"name": "PHASE_1_SEED", "quote_mult": 0.90, "risk_mult": 0.90, "min_quote": 40.0, "max_quote": 100.0},
            {"name": "PHASE_4_SNOWBALL", "quote_mult": 1.25, "risk_mult": 1.16, "min_quote": 90.0, "max_quote": 220.0},
        ],
        SCALING_EQUITY_TIERS=[],
        DEFAULT_PLANNED_QUOTE=80.0,
        MIN_TRADE_QUOTE=40.0,
        MAX_TRADE_QUOTE=250.0,
        MIN_NOTIONAL_USDT=10.0,
        BUY_HEADROOM_FACTOR=1.05,
        SCOUT_MIN_NOTIONAL=5.0,
        MAX_SPEND_PER_TRADE_USDT=250.0,
        RISK_PCT_PER_TRADE=0.01,
        TIER_B_RISK_PCT=0.005,
        SL_ATR_MULT=1.0,
        SL_PCT_MIN=0.003,
        SL_PCT_MAX=0.02,
        TPSL_VOL_TARGET_ATR_PCT=0.009,
        DYNAMIC_POSITION_SIZING_ENABLED=False,
    )

    ss = _ScalingSharedState(total_equity=500.0, bootstrap_equity=500.0)  # ratio = 1.0 => phase 1
    logger = SimpleNamespace(info=lambda *a, **k: None, debug=lambda *a, **k: None, warning=lambda *a, **k: None)
    mgr = ScalingManager(
        shared_state=ss,
        execution_manager=_ScalingExecutionManager(),
        config=cfg,
        logger=logger,
        mode_manager=_ModeManager("NORMAL"),
    )

    sig = {"symbol": "BTCUSDT", "agent": "MLForecaster", "action": "BUY", "confidence": 0.90}
    q_phase1 = await mgr.calculate_planned_quote("BTCUSDT", sig)
    assert ss.dynamic_config.get("compounding_phase") == "PHASE_1_SEED"

    ss.total_equity = 1300.0  # ratio = 2.6 => phase 4
    q_phase4 = await mgr.calculate_planned_quote("BTCUSDT", sig)
    assert ss.dynamic_config.get("compounding_phase") == "PHASE_4_SNOWBALL"
    assert q_phase4 > q_phase1


@pytest.mark.asyncio
async def test_compounding_disabled_in_safe_mode():
    cfg = SimpleNamespace(
        BASE_CAPITAL=400.0,
        COMPOUNDING_ENABLED=True,
        COMPOUNDING_GROWTH_CURVE_ENABLED=True,
        GROWTH_PHASE_THRESHOLDS=[
            {"name": "PHASE_1_SEED", "min_ratio": 0.0, "max_ratio": 1.25},
            {"name": "PHASE_4_SNOWBALL", "min_ratio": 2.50, "max_ratio": None},
        ],
        PHASE_SIZE_MULTIPLIERS={"PHASE_1_SEED": 0.90, "PHASE_4_SNOWBALL": 1.25},
        PHASE_MAX_TRADE_CAP={"PHASE_1_SEED": 100.0, "PHASE_4_SNOWBALL": 220.0},
        COMPOUNDING_GROWTH_PHASES=[
            {"name": "PHASE_1_SEED", "quote_mult": 0.90, "risk_mult": 0.90, "min_quote": 40.0, "max_quote": 100.0},
            {"name": "PHASE_4_SNOWBALL", "quote_mult": 1.25, "risk_mult": 1.16, "min_quote": 90.0, "max_quote": 220.0},
        ],
        SCALING_EQUITY_TIERS=[],
        DEFAULT_PLANNED_QUOTE=80.0,
        MIN_TRADE_QUOTE=40.0,
        MAX_TRADE_QUOTE=250.0,
        MIN_NOTIONAL_USDT=10.0,
        BUY_HEADROOM_FACTOR=1.05,
        SCOUT_MIN_NOTIONAL=5.0,
        MAX_SPEND_PER_TRADE_USDT=250.0,
        RISK_PCT_PER_TRADE=0.01,
        TIER_B_RISK_PCT=0.005,
        SL_ATR_MULT=1.0,
        SL_PCT_MIN=0.003,
        SL_PCT_MAX=0.02,
        TPSL_VOL_TARGET_ATR_PCT=0.009,
        DYNAMIC_POSITION_SIZING_ENABLED=False,
    )
    ss = _ScalingSharedState(total_equity=1300.0, bootstrap_equity=500.0)
    logger = SimpleNamespace(info=lambda *a, **k: None, debug=lambda *a, **k: None, warning=lambda *a, **k: None)
    mgr = ScalingManager(
        shared_state=ss,
        execution_manager=_ScalingExecutionManager(),
        config=cfg,
        logger=logger,
        mode_manager=_ModeManager("SAFE"),
    )
    sig = {"symbol": "BTCUSDT", "agent": "MLForecaster", "action": "BUY", "confidence": 0.90}
    quote = await mgr.calculate_planned_quote("BTCUSDT", sig)
    assert bool(ss.dynamic_config.get("COMPOUNDING_GROWTH_ACTIVE")) is False
    assert quote <= 80.0


@pytest.mark.asyncio
async def test_medium_acceleration_profile_targets_30_to_45_usdt_band():
    cfg = SimpleNamespace(
        BASE_CAPITAL=150.0,
        COMPOUNDING_ENABLED=True,
        COMPOUNDING_GROWTH_CURVE_ENABLED=True,
        GROWTH_PHASE_THRESHOLDS=[
            {"name": "PHASE_1_SEED", "min_ratio": 0.0, "max_ratio": 1.25},
            {"name": "PHASE_4_SNOWBALL", "min_ratio": 2.50, "max_ratio": None},
        ],
        PHASE_SIZE_MULTIPLIERS={"PHASE_1_SEED": 1.20, "PHASE_4_SNOWBALL": 1.30},
        PHASE_MAX_TRADE_CAP={"PHASE_1_SEED": 60.0, "PHASE_4_SNOWBALL": 150.0},
        COMPOUNDING_GROWTH_PHASES=[
            {"name": "PHASE_1_SEED", "quote_mult": 1.20, "risk_mult": 1.20, "min_quote": 25.0, "max_quote": 60.0},
            {"name": "PHASE_4_SNOWBALL", "quote_mult": 1.30, "risk_mult": 1.10, "min_quote": 80.0, "max_quote": 150.0},
        ],
        SCALING_EQUITY_TIERS=[],
        DEFAULT_PLANNED_QUOTE=30.0,
        MIN_TRADE_QUOTE=10.0,
        MAX_TRADE_QUOTE=250.0,
        MIN_NOTIONAL_USDT=10.0,
        BUY_HEADROOM_FACTOR=1.05,
        SCOUT_MIN_NOTIONAL=5.0,
        MAX_SPEND_PER_TRADE_USDT=150.0,
        RISK_PCT_PER_TRADE=0.01,
        TIER_B_RISK_PCT=0.005,
        SL_ATR_MULT=1.0,
        SL_PCT_MIN=0.003,
        SL_PCT_MAX=0.02,
        TPSL_VOL_TARGET_ATR_PCT=0.009,
        DYNAMIC_POSITION_SIZING_ENABLED=True,
        DYNAMIC_SIZE_UPSIDE_CAP_MULT=3.0,
        DYNAMIC_RISK_BUDGET_PCT=0.25,
        DYNAMIC_RISK_BUDGET_PCT_TIER_B=0.10,
        DYNAMIC_CONFIDENCE_MIN=0.65,
        DYNAMIC_CONFIDENCE_MAX=0.90,
    )
    ss = _ScalingSharedState(total_equity=150.0, bootstrap_equity=150.0)
    ss.market_data = {"BTCUSDT": {"5m": {"atr": 2.0}}}
    ss.latest_prices = {"BTCUSDT": 100.0}
    logger = SimpleNamespace(info=lambda *a, **k: None, debug=lambda *a, **k: None, warning=lambda *a, **k: None)
    mgr = ScalingManager(
        shared_state=ss,
        execution_manager=_ScalingExecutionManager(),
        config=cfg,
        logger=logger,
        mode_manager=_ModeManager("NORMAL"),
    )
    sig_low = {"symbol": "BTCUSDT", "agent": "MLForecaster", "action": "BUY", "confidence": 0.65}
    sig_high = {"symbol": "BTCUSDT", "agent": "MLForecaster", "action": "BUY", "confidence": 0.90}
    q_low = await mgr.calculate_planned_quote("BTCUSDT", sig_low)
    q_high = await mgr.calculate_planned_quote("BTCUSDT", sig_high)
    assert 30.0 <= q_low <= 45.0
    assert 30.0 <= q_high <= 45.0
    assert q_high >= q_low
