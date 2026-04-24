from dotenv import load_dotenv, find_dotenv

# Load repo .env, but keep explicit shell/runtime env vars authoritative.
# This is critical for live-session overrides (e.g. SYMBOLS, feature flags, durations).
load_dotenv(dotenv_path=find_dotenv(usecwd=True), override=False)

import os
import logging
import json
from types import SimpleNamespace
from typing import Optional


logger = logging.getLogger("Config")
MASK = "****"
logger.info("Config module stamp: 2025-10-04T15:35Z / notional floors logging ENABLED")


class Config:
    """
    Centralized configuration registry for Octivault Trader.
    - LIVE-safe defaults; everything can be overridden via .env
    - Adds explicit Binance routing (region + account_type) with clear logging
    """

    # ---------- Static (class-level) ----------
    DEFAULT_WATCHDOG_INTERVAL = 10
    TIMEFRAMES = ["1m", "5m"]
    REQUIRED_TIMEFRAMES_FOR_AGENTS = ["1m", "5m"]
    MIN_BARS_FOR_MARKET_READY = 150
    REQUIRED_SYMBOL_COVERAGE_FOR_READY = 0.70
    # Universe-layer cap (discovery/watchlist breadth). Must stay independent from
    # position/allocation limits such as MAX_POSITIONS_TOTAL.
    MAX_UNIVERSE_SYMBOLS = 30
    MAX_POSITIONS_TOTAL = 2
    
    # ---------- Phase 1: Safe Upgrade - Symbol Rotation ----------
    # Soft bootstrap lock: prevents rotation for duration after trade
    BOOTSTRAP_SOFT_LOCK_ENABLED = True
    BOOTSTRAP_SOFT_LOCK_DURATION_SEC = 3600  # 1 hour
    
    # Replacement multiplier: score threshold to trigger rotation
    # e.g., 1.10 means candidate must be 10% better than current to rotate
    SYMBOL_REPLACEMENT_MULTIPLIER = 1.10
    
    # Universe size limits (must fit within discovery cap)
    MAX_ACTIVE_SYMBOLS = 5       # Do not exceed this many active symbols
    MIN_ACTIVE_SYMBOLS = 3       # Maintain at least this many
    
    # Symbol screener targets (Phase 1 implementation)
    SCREENER_MIN_PROPOSALS = 20   # Minimum candidates to propose
    SCREENER_MAX_PROPOSALS = 30   # Maximum candidates to propose
    SCREENER_MIN_VOLUME = 100000  # $100k minimum 24h volume (lowered from 1M for better discovery)
    SCREENER_MIN_PRICE = 0.01     # Filter dust coins

    # ---------- Volatility regime (ATR%) ----------
    VOLATILITY_REGIME_TIMEFRAME = "5m"
    VOLATILITY_REGIME_ATR_PERIOD = 14
    VOLATILITY_REGIME_UPDATE_SEC = 15.0
    VOLATILITY_REGIME_LOW_PCT = 0.0025
    VOLATILITY_REGIME_HIGH_PCT = 0.006
    VOLATILITY_REGIME_DEFAULT = "normal"
    LOW_REGIME_IGNORE_BREACH = True
    LOW_REGIME_TARGET_MULT = 0.25

    # ---------- Capital velocity (REA / stagnation) ----------
    ROTATION_COOLDOWN_SEC = 180
    ROTATION_BASE_ALPHA_GAP = 0.003
    ROTATION_WINNER_PROTECTION_PNL = 0.005
    ROTATION_WINNER_EXTRA_ALPHA = 0.02
    STAGNATION_HOLD_MULT = 2.0
    STAGNATION_PNL_BAND = 0.001
    STAGNATION_OVERRIDE_ENABLED = True
    STAGNATION_EXIT_ENABLED = True
    STAGNATION_EXIT_MAX_LOSS_PCT = 0.004  # was 0.0002 (0.02%) — hair-trigger that turned every dip into a realized loss

    # ---------- CAPITAL PROFILES (Dynamic NAV-based switching) ----------
    # P9 Architecture: Profile system maintains integrity while adapting to capital scale
    # 
    # The system automatically switches profiles based on Net Asset Value (NAV):
    # 
    # BOOTSTRAP_GROWTH (NAV < 500 USDT)
    #   Objective: Unblock engine, accumulate learning data, enable compounding
    #   EV Multiplier: 1.4 (permissive gate to unlock alpha)
    #   Sizing: Adaptive 12-24 USDT (micro capital aware)
    #   Profit Lock: Disabled (priority = learning)
    #   Warning: Per-trade PnL will be small. This is expected.
    #   Goal: Enable market learning, not magical profits.
    #
    # INSTITUTIONAL (NAV >= 500 USDT)
    #   Objective: Proven profitability with discipline
    #   EV Multiplier: 2.0 (strict gate)
    #   Sizing: Scales to 30+ USDT (institutional standard)
    #   Profit Lock: Enabled (harvest earned edge)
    #   Discipline: Full P9 constraints applied
    #   Goal: Sustainable long-term compounding
    
    CAPITAL_PROFILE_NAV_THRESHOLD = 500.0  # Switch threshold (USDT)
    # Capital governor bracket thresholds
    CAPITAL_MICRO_THRESHOLD = 500.0
    CAPITAL_SMALL_THRESHOLD = 2000.0
    CAPITAL_MEDIUM_THRESHOLD = 10000.0
    # Micro-bracket position/rotation policy (override via env without code change)
    CAPITAL_MICRO_MAX_ACTIVE_SYMBOLS = 3
    CAPITAL_MICRO_CORE_PAIRS = 2
    CAPITAL_MICRO_MAX_ROTATING_SLOTS = 1
    CAPITAL_MICRO_MAX_CONCURRENT_POSITIONS = 2
    CAPITAL_MICRO_ALLOW_ROTATION = True
    CAPITAL_MICRO_SYMBOL_REPLACEMENT_MULTIPLIER = 1.35
    CAPITAL_MICRO_SOFT_LOCK_DURATION_SEC = 3600
    # Liveness guard: prevents deadlock-prone micro profiles (1 slot + no rotation)
    # from stalling execution while still preserving conservative risk posture.
    CAPITAL_MICRO_ENFORCE_LIVENESS = True
    # Adaptive micro-capacity escape hatch for POSITION_ALREADY_OPEN pressure
    CAPITAL_MICRO_ADAPTIVE_CAPACITY_ENABLED = True
    CAPITAL_MICRO_ADAPTIVE_PRESSURE_TRIGGER = 6
    CAPITAL_MICRO_ADAPTIVE_WINDOW_SEC = 300.0
    CAPITAL_MICRO_ADAPTIVE_MAX_CONCURRENT_POSITIONS = 2
    CAPITAL_MICRO_ADAPTIVE_MAX_ROTATING_SLOTS = 1
    CAPITAL_MICRO_ADAPTIVE_SOFT_LOCK_DURATION_SEC = 900
    CAPITAL_MICRO_ADAPTIVE_SYMBOL_REPLACEMENT_MULTIPLIER = 1.35
    CAPITAL_MICRO_ADAPTIVE_SELL_DEADLOCK_ENABLED = True
    CAPITAL_MICRO_ADAPTIVE_SELL_DEADLOCK_TRIGGER = 5
    CAPITAL_MICRO_ADAPTIVE_SELL_DEADLOCK_WINDOW_SEC = 300.0
    CAPITAL_MICRO_ADAPTIVE_SELL_DEADLOCK_REASONS = "PORTFOLIO_PNL_IMPROVEMENT,SELL_DYNAMIC_EDGE_MIN,CLOSE_NOT_SUBMITTED,SELL_NET_PNL_MIN"
    
    # Profile definitions (can be extended for intermediate phases)
    CAPITAL_PROFILES = {
        "BOOTSTRAP_GROWTH": {
            "ev_multiplier": 1.4,
            "default_planned_quote": 24.0,
            "min_trade_quote": 12.0,
            "min_order_usdt": 12.0,
            "enable_profit_lock": False,
            "enable_strict_gating": False,
            "description": "Learning phase: unblock trading, enable compounding",
        },
        "INSTITUTIONAL": {
            "ev_multiplier": 2.0,
            "default_planned_quote": 30.0,
            "min_trade_quote": 20.0,
            "min_order_usdt": 20.0,
            "enable_profit_lock": True,
            "enable_strict_gating": True,
            "description": "Production phase: proven edge, institutional discipline",
        },
    }
    
    def get_capital_profile(self, nav: float) -> dict:
        """
        Dynamically select capital profile based on NAV.
        
        Args:
            nav: Net Asset Value in USDT
            
        Returns:
            Profile dict with EV multiplier, sizing, gates
        """
        if nav < self.CAPITAL_PROFILE_NAV_THRESHOLD:
            profile_name = "BOOTSTRAP_GROWTH"
        else:
            profile_name = "INSTITUTIONAL"
        
        profile = self.CAPITAL_PROFILES.get(profile_name, self.CAPITAL_PROFILES["BOOTSTRAP_GROWTH"])
        logger.info(
            f"[Profile] NAV={nav:.2f} → {profile_name} "
            f"(EV×{profile['ev_multiplier']}, quote={profile['default_planned_quote']:.0f})"
        )
        return profile

    # ---------- Min-notional avoidance (P9 Defaults - profile-overrideable) ----------
    # NOTE: These are the DEFAULT values for BOOTSTRAP phase.
    #   When capital < 500: Uses BOOTSTRAP_GROWTH profile (12-24 USDT)
    #   When capital >= 500: Uses INSTITUTIONAL profile (20-30+ USDT)
    # 
    # The get_capital_profile() method dynamically returns appropriate values.
    MIN_ORDER_USDT = 12.0
    QUOTE_MIN_NOTIONAL = 12.0
    EXECUTION_MIN_NOTIONAL_QUOTE = 12.0
    SAFE_ENTRY_USDT = 12.0
    MIN_POSITION_VALUE_USDT = 10.0
    MIN_POSITION_USDT = 24.0
    MIN_POSITION_MIN_NOTIONAL_MULT = 2.0
    MIN_ENTRY_USDT = 24.0
    # Reduced from 15.0 to 10.0 to allow larger discovery universe on small accounts
    # UURE smart cap = floor(equity × exposure / min_entry)
    # Example: $200 × 0.6 / 10 = 12 symbols (vs. 4 with 25)
    # With env override via MIN_ENTRY_QUOTE_USDT, can be tuned per account
    MIN_ENTRY_QUOTE_USDT = 10.0
    DEFAULT_PLANNED_QUOTE = 24.0
    EMIT_BUY_QUOTE = 24.0
    MIN_TRADE_QUOTE = 12.0
    MAX_TRADE_QUOTE = 250.0
    MIN_SIGNIFICANT_POSITION_USDT = 20.0
    SIGNIFICANT_POSITION_FLOOR = MIN_SIGNIFICANT_POSITION_USDT
    EXEC_PROBE_QUOTE = 12.0
    MAX_HOLD_SEC = 3600.0      # 1 hour — was 30 min, caused forced exits during temporary dips
    MAX_HOLD_TIME_SEC = 3600.0
    LIQ_ORCH_MIN_USDT_FLOOR = 5.0
    LIQ_ORCH_MIN_USDT_TARGET = 6.0
    DUST_MIN_QUOTE_USDT = 5.0
    DUST_POSITION_QTY = 0.0001
    PERMANENT_DUST_USDT_THRESHOLD = 1.0
    DUST_LIQUIDATION_ENABLED = True
    DUST_REENTRY_OVERRIDE = True
    ALLOW_ENTRY_BELOW_SIGNIFICANT_FLOOR = False
    # Dust priority system thresholds (Reuse > Aggregate > Cleanup)
    DUST_AGGREGATE_THRESHOLD_HOURS = 4.0   # hold dust this long before triggering cleanup
    DUST_STALL_THRESHOLD_HOURS = 4.0       # used by DustMonitor to classify STALLED status
    DUST_CRITICAL_THRESHOLD_HOURS = 8.0   # used by DustMonitor to classify CRITICAL status
    STRICT_ACCOUNTING_INTEGRITY = False
    STRICT_OBSERVABILITY_EVENTS = False
    CAPITAL_ALLOCATOR_SHARED_WALLET = True

    EXIT_EXCURSION_TICK_MULT = 2.0
    EXIT_EXCURSION_ATR_MULT = 0.35
    EXIT_EXCURSION_SPREAD_MULT = 3.0
    EXIT_FEE_BPS = 10.0
    EXIT_SLIPPAGE_BPS = 10.0
    TP_MIN_BUFFER_BPS = 8.0
    TP_MICRO_NOTIONAL_USDT = 25.0
    TP_MICRO_EXTRA_BPS = 20.0
    TP_ATR_MULT = 1.5  # Increased for RR viability (was 1.0)
    SL_ATR_MULT = 1.0  # ATR base for SL (was 1.1)
    TARGET_RR_RATIO = 1.8  # Target risk-reward ratio (TP = RR × SL)
    TRAILING_ATR_MULT = 1.5
    TPSL_RV_LOOKBACK = 20
    TPSL_VOL_LOW_ATR_PCT = 0.0045
    TPSL_VOL_HIGH_ATR_PCT = 0.0150
    TPSL_VOL_TARGET_ATR_PCT = 0.0090
    TPSL_DYNAMIC_RR_MIN = 1.35
    TPSL_DYNAMIC_RR_MAX = 2.60
    TRAILING_ACTIVATE_R_MULT = 0.60
    TPSL_PROFILE = "balanced"
    TPSL_PROFILE_PRESETS = {
        "scalp": {
            "TARGET_RR_RATIO": 1.45,
            "TP_ATR_MULT": 1.20,
            "SL_ATR_MULT": 0.95,
            "TRAILING_ATR_MULT": 1.10,
            "TPSL_VOL_LOW_ATR_PCT": 0.0035,
            "TPSL_VOL_HIGH_ATR_PCT": 0.0120,
            "TPSL_VOL_TARGET_ATR_PCT": 0.0075,
            "TPSL_DYNAMIC_RR_MIN": 1.20,
            "TPSL_DYNAMIC_RR_MAX": 2.00,
            "TP_PCT_MIN": 0.0020,
            "TP_PCT_MAX": 0.0120,
            "SL_PCT_MIN": 0.0020,
            "SL_PCT_MAX": 0.0080,
            "TRAILING_ACTIVATE_R_MULT": 0.70,
        },
        "balanced": {
            "TARGET_RR_RATIO": 1.80,
            "TP_ATR_MULT": 1.50,
            "SL_ATR_MULT": 1.00,
            "TRAILING_ATR_MULT": 1.50,
            "TPSL_VOL_LOW_ATR_PCT": 0.0045,
            "TPSL_VOL_HIGH_ATR_PCT": 0.0150,
            "TPSL_VOL_TARGET_ATR_PCT": 0.0090,
            "TPSL_DYNAMIC_RR_MIN": 1.35,
            "TPSL_DYNAMIC_RR_MAX": 2.60,
            "TP_PCT_MIN": 0.0080,  # Fee-aware: round-trip ~0.30%, must clear with margin
            "TP_PCT_MAX": 0.0250,
            "SL_PCT_MIN": 0.0030,
            "SL_PCT_MAX": 0.0100,
            "TRAILING_ACTIVATE_R_MULT": 0.60,
        },
        "swing": {
            "TARGET_RR_RATIO": 2.20,
            "TP_ATR_MULT": 1.90,
            "SL_ATR_MULT": 1.10,
            "TRAILING_ATR_MULT": 1.90,
            "TPSL_VOL_LOW_ATR_PCT": 0.0050,
            "TPSL_VOL_HIGH_ATR_PCT": 0.0180,
            "TPSL_VOL_TARGET_ATR_PCT": 0.0110,
            "TPSL_DYNAMIC_RR_MIN": 1.50,
            "TPSL_DYNAMIC_RR_MAX": 3.00,
            "TP_PCT_MIN": 0.0040,
            "TP_PCT_MAX": 0.0250,
            "SL_PCT_MIN": 0.0040,
            "SL_PCT_MAX": 0.0120,
            "TRAILING_ACTIVATE_R_MULT": 0.55,
        },
    }
    TP_PCT_MIN = 0.008  # Fee-aware minimum: round-trip friction ~0.30%, TP must clear it with margin
    TP_PCT_MAX = 0.025  # Increased maximum TP for asymmetry headroom
    # Spread-adaptive TP shaping (microstructure-aware TP optimization)
    TPSL_SPREAD_ADAPTIVE_ENABLED = True
    TPSL_SPREAD_TIGHT_BPS = 6.0
    TPSL_SPREAD_HIGH_BPS = 18.0
    TPSL_SPREAD_EXTREME_BPS = 45.0
    TPSL_SPREAD_RR_BONUS_MAX = 0.18
    TPSL_SPREAD_RR_DISCOUNT_MAX = 0.06
    TPSL_SPREAD_TP_FLOOR_MULT = 2.0
    # Risk-based position sizing
    RISK_PCT_PER_TRADE = 0.01  # 1% of equity risk per trade
    TIER_B_RISK_PCT = 0.005  # 0.5% for micro trades
    TIER_B_MAX_QUOTE = 40.0  # Max quote for tier B trades
    # Entry economics guards
    MIN_PLANNED_QUOTE_FEE_MULT = 3.0
    MIN_PROFIT_EXIT_FEE_MULT = 2.5
    MIN_ECONOMIC_TRADE_USDT = 30.0
    MIN_NET_PROFIT_AFTER_FEES = 0.004
    STRICT_PROFIT_ONLY_SELLS = False
    # Pre-trade effect + micro-backtest gate
    PRETRADE_EFFECT_GUARD_ENABLED = True
    PRETRADE_ALLOW_MISSING_EXPECTED_MOVE = False
    PRETRADE_ALLOW_MISSING_EXPECTED_MOVE_SELL = True
    PRETRADE_SELL_REQUIRE_EXPECTED_EDGE = False
    PRETRADE_SELL_GATE_ENFORCED = False
    PRETRADE_ALLOW_BOOTSTRAP_BYPASS = False
    PRETRADE_DIRECTIVE_GATE_ENABLED = True
    PRETRADE_EFFECT_BUFFER_BPS = 5.0
    PRETRADE_MIN_EXPECTED_NET_PCT = 0.0012
    PRETRADE_MIN_EXPECTED_NET_USDT = 0.04
    PRETRADE_RECENT_REALIZED_WINDOW = 30
    PRETRADE_MIN_REALIZED_SAMPLES = 10
    PRETRADE_MIN_REALIZED_WIN_RATE = 0.50
    PRETRADE_MICRO_BACKTEST_ENABLED = True
    PRETRADE_MICRO_BACKTEST_TIMEFRAME = "5m"
    PRETRADE_MICRO_BACKTEST_LOOKBACK_BARS = 96
    PRETRADE_MICRO_BACKTEST_HORIZON_BARS = 3
    PRETRADE_MICRO_BACKTEST_MIN_SAMPLES = 12
    PRETRADE_MICRO_BACKTEST_REQUIRE_SAMPLES = True
    PRETRADE_MICRO_BACKTEST_MIN_WIN_RATE = 0.52
    PRETRADE_MICRO_BACKTEST_MIN_AVG_NET_PCT = 0.0002
    PRETRADE_MICRO_BACKTEST_DEADLOCK_RELAX_ENABLED = True
    PRETRADE_MICRO_BACKTEST_DEADLOCK_REJECTION_TRIGGER = 6
    PRETRADE_MICRO_BACKTEST_DEADLOCK_STEP_REJECTIONS = 4
    PRETRADE_MICRO_BACKTEST_DEADLOCK_RELAX_WIN_STEP = 0.015
    PRETRADE_MICRO_BACKTEST_DEADLOCK_MIN_WIN_RATE_FLOOR = 0.45
    PRETRADE_MICRO_BACKTEST_DEADLOCK_MIN_AVG_NET_FLOOR_PCT = -0.0003
    PRETRADE_MICRO_BACKTEST_DEADLOCK_DISABLE_SAMPLE_REQUIRE_AFTER_STEPS = 2
    PRETRADE_MICRO_BACKTEST_DEADLOCK_FORCE_BYPASS_AFTER_REJECTIONS = 16
    PRETRADE_MICRO_BACKTEST_DEADLOCK_FORCE_BYPASS_MIN_CONF = 0.68
    PRETRADE_MICRO_BACKTEST_DEADLOCK_FORCE_BYPASS_MIN_EXPECTED_NET_PCT = 0.0022
    PRETRADE_MICRO_BACKTEST_DEADLOCK_FORCE_BYPASS_MIN_EXPECTED_NET_USDT = 0.025
    PRETRADE_MICRO_BACKTEST_STARVATION_CYCLES_TRIGGER = 10
    PRETRADE_MICRO_BACKTEST_STARVATION_REJECTION_TRIGGER = 6
    PRETRADE_MICRO_BACKTEST_STARVATION_FORCE_BYPASS_MIN_CONF = 0.60
    PRETRADE_MICRO_BACKTEST_STARVATION_FORCE_BYPASS_MIN_EXPECTED_NET_PCT = 0.0015
    PRETRADE_MICRO_BACKTEST_STARVATION_FORCE_BYPASS_MIN_EXPECTED_NET_USDT = 0.02
    PRETRADE_NET_USDT_DEADLOCK_RELAX_ENABLED = True
    PRETRADE_NET_USDT_DEADLOCK_REJECTION_TRIGGER = 6
    PRETRADE_NET_USDT_DEADLOCK_STEP_REJECTIONS = 4
    PRETRADE_NET_USDT_DEADLOCK_RELAX_STEP_FACTOR = 0.85
    PRETRADE_NET_USDT_DEADLOCK_MIN_USDT_FLOOR = 0.02
    PRETRADE_STALL_RELAX_ENABLED = True
    PRETRADE_STALL_RELAX_CYCLES_TRIGGER = 10
    PRETRADE_STALL_RELAX_STEP_CYCLES = 5
    PRETRADE_STALL_RELAX_NET_PCT_STEP = 0.00012
    PRETRADE_STALL_RELAX_NET_USDT_STEP = 0.005
    PRETRADE_STALL_RELAX_WIN_RATE_STEP = 0.01
    PRETRADE_STALL_RELAX_MAX_STEPS = 6
    PRETRADE_STALL_RELAX_MIN_NET_PCT_FLOOR = 0.0006
    PRETRADE_STALL_RELAX_MIN_NET_USDT_FLOOR = 0.02
    PRETRADE_STALL_RELAX_MIN_WIN_RATE_FLOOR = 0.45
    PRETRADE_STALL_RELAX_MIN_BT_WIN_RATE_FLOOR = 0.46
    PRETRADE_MICRO_BACKTEST_SOFT_GATE_ENABLED = True
    PRETRADE_MICRO_BACKTEST_SOFT_GATE_MIN_SAMPLES = 24
    PRETRADE_MICRO_BACKTEST_SOFT_GATE_RELAX_WIN_RATE = 0.03
    PRETRADE_MICRO_BACKTEST_SOFT_GATE_RELAX_AVG_NET_PCT = 0.00015
    PRETRADE_MICRO_BACKTEST_SOFT_GATE_ALLOW_MISSING_AFTER_CYCLES = 16
    SELL_DYNAMIC_EDGE_GATE_ENABLED = True
    SELL_DYNAMIC_SLIPPAGE_MIN_PCT = 0.0003
    SELL_DYNAMIC_SLIPPAGE_ATR_MULT = 0.05
    SELL_DYNAMIC_VOL_BUFFER_ATR_MULT = 0.15
    SELL_DYNAMIC_STRATEGIC_BUFFER_PCT = 0.0
    SELL_DYNAMIC_MIN_USDT_FLOOR = 0.12
    SELL_DYNAMIC_ATR_TIMEFRAME = "5m"
    SELL_DYNAMIC_ATR_PERIOD = 14
    SELL_DYNAMIC_FALLBACK_ATR_PCT = 0.0
    SELL_DYNAMIC_REGIME_HIGH_MULT = 1.3
    SELL_DYNAMIC_REGIME_LOW_MULT = 0.8
    SELL_DYNAMIC_LEGACY_NET_PCT_GUARD = False
    MIN_PORTFOLIO_IMPROVEMENT_USD = 0.05
    MIN_PORTFOLIO_IMPROVEMENT_PCT = 0.0015
    CLOSE_ESCAPE_HATCH_ENABLED = True
    CLOSE_ESCAPE_HATCH_TRIGGER_COUNT = 4
    CLOSE_ESCAPE_HATCH_WINDOW_SEC = 900.0
    CLOSE_ESCAPE_HATCH_MIN_AGE_SEC = 120.0
    CLOSE_ESCAPE_HATCH_RETRY_COOLDOWN_SEC = 180.0
    CLOSE_ESCAPE_HATCH_BUY_LOCK_TRIGGER = 0
    CLOSE_ESCAPE_HATCH_ALLOW_REASONS = "portfolio_pnl_improvement,sell_dynamic_edge_below_min,sell_net_pnl_below_min,sell_net_pct_below_min"
    TRADEABILITY_FILL_STALL_RELAX_ENABLED = True
    TRADEABILITY_FILL_STALL_WINDOW_SEC = 900.0
    TRADEABILITY_FILL_STALL_MAX_STEPS = 6
    TRADEABILITY_FILL_STALL_MIN_RECENT_TRADES = 1
    TRADEABILITY_FILL_STALL_RELAX_STEP = 0.008
    TRADEABILITY_FILL_STALL_RELAX_MAX = 0.045
    TRADEABILITY_FILL_STALL_MEDIUM_RATIO_STEP = 0.010
    TRADEABILITY_FILL_STALL_MEDIUM_RATIO_MAX_DROP = 0.10
    BOOTSTRAP_VETO_COOLDOWN_SEC = 600.0
    # Micro-trade kill switch (low equity + low volatility)
    MICRO_TRADE_KILL_SWITCH_ENABLED = True
    MICRO_TRADE_KILL_EQUITY_MAX = 150.0
    MICRO_TRADE_KILL_ATR_FEE_MULT = 1.0
    MICRO_TRADE_KILL_FALLBACK_ATR_PCT = 0.0
    # Bootstrap escape hatch: allow ONE trade to bypass kill-switch + min-economic
    # when portfolio is FLAT (cold bootstrap). Auto-disables after first trade.
    BOOTSTRAP_ESCAPE_HATCH_ENABLED = True
    BUY_COOLDOWN_SEC = 300
    ENTRY_COOLDOWN_SEC = 120
    MAX_OPEN_POSITIONS_PER_SYMBOL = 1
    BUY_REENTRY_MIN_DELTA_PCT = 0.001
    BUY_REENTRY_DELTA_PCT = 0.002
    BUY_REENTRY_DELTA_PCT_TEMP = 0.0015
    BUY_REENTRY_DELTA_RESTORE_EQUITY = 220.0
    BUY_REENTRY_DELTA_RESTORE_TRADES = 3

    NO_TICK_THRESHOLD_SECONDS = 300
    RECOVERY_COOLDOWN_SECONDS = 180
    RECOVERY_STARTUP_GRACE_SECONDS = 180
    GLOBAL_RECOVER_MIN_GAP = 15
    MDF_RESTART_MIN_GAP = 20

    TARGET_PROFIT_USDT_PER_HOUR = 0.0 # Deprecated: use NAV * TARGET_PROFIT_RATIO_PER_HOUR
    TARGET_PROFIT_LOOKBACK_MIN = 60
    TARGET_PROFIT_CHECK_SEC = 60
    UPTIME_GRACE_PERIOD_MIN = 30 # Ignore noise during startup
    TARGET_PROFIT_RATIO_PER_HOUR = 0.0008 # 0.08% per hour of NAV (baseline 0.08%–0.12%)
    # ProfitTargetEngine knobs
    PROFIT_TARGET_DAILY_PCT = 0.02          # 2% daily NAV target
    PROFIT_TARGET_MAX_RISK_PER_CYCLE = 0.005 # 0.5% max risk per evaluation cycle
    PROFIT_TARGET_COMPOUND_THROTTLE = 0.5   # 50% of excess profit reinvested (rest banked)
    PROFIT_TARGET_BASE_USD_PER_HOUR = 0.0   # 0 = use ratio-based target; >0 overrides
    PROFIT_TARGET_GRACE_MINUTES = 30        # Ignore target enforcement during startup
    # Grace period to prevent stagnation purges immediately after startup
    # Units: minutes (default 30 minutes)
    STARTUP_STAGNATION_GRACE_MINUTES = float(os.getenv("STARTUP_STAGNATION_GRACE_MINUTES", "30"))
    TPSL_PROFIT_AUDIT = True
    TPSL_PROFIT_AUDIT_SEC = 300
    MAX_TRADES_PER_DAY = 0
    TP_TARGET_PCT = 0.0
    SL_CAP_PCT = 0.0

    # ---------- Equity-tier scaling (Realized equity ladder) ----------
    EQUITY_TIER_BLOCK_ON_RECOVERY = True
    EQUITY_TIER_BLOCK_ON_LIQUIDATION = True
    EQUITY_TIER_BLOCK_ON_OPEN_POSITIONS = True
    EQUITY_TIER_USE_REALIZED_ONLY = True
    SCALING_EQUITY_TIERS = [
        {
            "name": "200-249",
            "min": 200.0,
            "max": 249.99,
            "planned_quote": 20.0,
            "max_positions": 1,
            "risk_mode": "NORMAL",
            "max_daily_trades": 6,
            "hold_minutes": 30,
            "tp_target_pct": 0.006,
            "sl_cap_pct": -0.004,
            "goal": "Prove profitability > fees",
        },
        {
            "name": "250-299",
            "min": 250.0,
            "max": 299.99,
            "planned_quote": 25.0,
            "max_positions": 1,
            "risk_mode": "NORMAL",
            "max_daily_trades": 8,
            "hold_minutes": 30,
            "tp_target_pct": 0.007,
            "sl_cap_pct": -0.0045,
            "goal": "Fee efficiency + consistency",
        },
        {
            "name": "300-349",
            "min": 300.0,
            "max": 349.99,
            "planned_quote": 30.0,
            "max_positions": 2,
            "risk_mode": "NORMAL",
            "max_daily_trades": 10,
            "hold_minutes": 45,
            "tp_target_pct": 0.008,
            "sl_cap_pct": -0.005,
            "goal": "Parallel exposure without churn",
        },
        {
            "name": "350-449",
            "min": 350.0,
            "max": 449.99,
            "planned_quote": 35.0,
            "max_positions": 2,
            "risk_mode": "CONTROLLED",
            "max_daily_trades": 12,
            "hold_minutes": 60,
            "tp_target_pct": 0.009,
            "sl_cap_pct": -0.0055,
            "goal": "Compounding speed without drawdowns",
        },
        {
            "name": "450+",
            "min": 450.0,
            "max": None,
            "planned_quote": 40.0,
            "max_positions": 3,
            "risk_mode": "CONTROLLED",
            "max_daily_trades": 14,
            "hold_minutes": 60,
            "tp_target_pct": 0.010,
            "sl_cap_pct": -0.006,
            "goal": "Prove scalability before automation",
        },
    ]

    # ---------- Compounding growth curve (Phase 1 -> Phase 4) ----------
    COMPOUNDING_ENABLED = True
    COMPOUNDING_GROWTH_CURVE_ENABLED = True  # backward-compat alias
    # Ratio uses: growth_ratio = total_equity / bootstrap_equity
    GROWTH_PHASE_THRESHOLDS = [
        {"name": "PHASE_1_SEED", "min_ratio": 0.0, "max_ratio": 1.25},
        {"name": "PHASE_2_TRACTION", "min_ratio": 1.25, "max_ratio": 1.75},
        {"name": "PHASE_3_ACCELERATE", "min_ratio": 1.75, "max_ratio": 2.50},
        {"name": "PHASE_4_SNOWBALL", "min_ratio": 2.50, "max_ratio": None},
    ]
    PHASE_SIZE_MULTIPLIERS = {
        "PHASE_1_SEED": 1.20,
        "PHASE_2_TRACTION": 1.10,
        "PHASE_3_ACCELERATE": 1.28,
        # Phase 4 prioritizes capital defense over further aggression.
        "PHASE_4_SNOWBALL": 1.12,
    }
    PHASE_MAX_TRADE_CAP = {
        "PHASE_1_SEED": 45.0,
        "PHASE_2_TRACTION": 140.0,
        "PHASE_3_ACCELERATE": 210.0,
        "PHASE_4_SNOWBALL": 180.0,
    }
    COMPOUNDING_MAX_DRAWDOWN_PCT = 2.5
    COMPOUNDING_MIN_POSITIVE_STREAK = 0
    COMPOUNDING_GROWTH_PHASES = [
        {
            "name": "PHASE_1_SEED",
            "min_equity": 0.0,
            "max_equity": 249.99,
            "min_momentum": -9999.0,
            "quote_mult": 1.20,
            "risk_mult": 1.20,
            "min_quote": 25.0,
            "max_quote": 45.0,
            "tp_asym_mult": 1.20,
            "sl_asym_mult": 1.00,
            "rr_bonus": 0.04,
        },
        {
            "name": "PHASE_2_TRACTION",
            "min_equity": 250.0,
            "max_equity": 399.99,
            "min_momentum": 0.0,
            "quote_mult": 1.10,
            "risk_mult": 1.10,
            "min_quote": 55.0,
            "max_quote": 140.0,
            "tp_asym_mult": 1.40,
            "sl_asym_mult": 0.95,
            "rr_bonus": 0.12,
        },
        {
            "name": "PHASE_3_ACCELERATE",
            "min_equity": 400.0,
            "max_equity": 699.99,
            "min_momentum": 0.0,
            "quote_mult": 1.28,
            "risk_mult": 1.22,
            "min_quote": 70.0,
            "max_quote": 210.0,
            "tp_asym_mult": 1.60,
            "sl_asym_mult": 0.90,
            "rr_bonus": 0.22,
        },
        {
            "name": "PHASE_4_SNOWBALL",
            "min_equity": 700.0,
            "max_equity": None,
            "min_momentum": 0.0,
            "quote_mult": 1.12,
            "risk_mult": 1.05,
            "min_quote": 80.0,
            "max_quote": 180.0,
            "tp_asym_mult": 1.30,
            "sl_asym_mult": 0.75,
            "rr_bonus": 0.10,
        },
    ]

    # ---------- Dynamic position sizing ----------
    DYNAMIC_POSITION_SIZING_ENABLED = True
    DYNAMIC_SIZE_CONF_FLOOR_MULT = 0.85
    DYNAMIC_SIZE_CONF_CEIL_MULT = 1.25
    DYNAMIC_SIZE_VOL_FLOOR_MULT = 0.70
    DYNAMIC_SIZE_VOL_CEIL_MULT = 1.35
    DYNAMIC_SIZE_MOMENTUM_MULT = 0.45
    DYNAMIC_SIZE_BLEND_WEIGHT = 0.65
    DYNAMIC_SIZE_UPSIDE_CAP_MULT = 2.25
    DYNAMIC_SIZE_MOMENTUM_LOOKBACK_TRADES = 20
    DYNAMIC_RISK_BUDGET_PCT = 0.30
    DYNAMIC_RISK_BUDGET_PCT_TIER_B = 0.10
    DYNAMIC_CONFIDENCE_MIN = 0.65
    DYNAMIC_CONFIDENCE_MAX = 0.90
    STABLE_RISK_BUDGET_ENABLED = True
    STABLE_RISK_BUDGET_MULT = 1.15
    STABLE_RISK_MIN_POSITIVE_STREAK = 3
    STABLE_RISK_MAX_DRAWDOWN_PCT = 1.5
    ADAPTIVE_CAPITAL_ENGINE_ENABLED = True
    ADAPTIVE_PERF_REVIEW_SEC = 3600.0
    ADAPTIVE_RISK_FRACTION_MIN = 0.05
    ADAPTIVE_RISK_FRACTION_MAX = 0.35
    ADAPTIVE_DRAWDOWN_SOFT_PCT = 2.0
    ADAPTIVE_DRAWDOWN_HARD_PCT = 4.0
    ADAPTIVE_HIGH_VOL_PCT = 0.015
    ADAPTIVE_LOW_VOL_PCT = 0.004
    ADAPTIVE_THROUGHPUT_LOW_RATIO = 0.50
    ADAPTIVE_IDLE_FREE_CAPITAL_PCT = 0.60
    ADAPTIVE_IDLE_TIME_SEC = 1800.0
    ADAPTIVE_WIN_STREAK_TRADES = 3
    ADAPTIVE_LOSS_STREAK_TRADES = 3
    ADAPTIVE_WIN_STREAK_RISK_BONUS = 0.10
    ADAPTIVE_LOSS_STREAK_RISK_PENALTY = 0.18
    ADAPTIVE_WIN_RATE_BONUS_THRESHOLD = 0.60
    ADAPTIVE_WIN_RATE_PENALTY_THRESHOLD = 0.45
    ADAPTIVE_WIN_RATE_BONUS = 0.08
    ADAPTIVE_WIN_RATE_PENALTY = 0.10
    ADAPTIVE_FEE_GROSS_THRESHOLD = 0.35
    ADAPTIVE_FEE_GROSS_BONUS = 0.06
    ADAPTIVE_ECON_MIN_NOTIONAL_MULT = 1.20
    ADAPTIVE_ECON_TARGET_PROFIT_PCT = 0.004
    ADAPTIVE_MIN_QUOTE_BUFFER_MULT = 1.20

    # ---------- Stagnation forced rotation ----------
    STAGNATION_AGE_SEC = 1500.0  # 25m
    STAGNATION_PNL_THRESHOLD = 0.0030
    STAGNATION_STREAK_LIMIT = 2
    STAGNATION_CONTINUATION_MIN_SCORE = 0.65
    STAGNATION_FORCE_ROTATION_ENABLED = True
    STAGNATION_FORCE_ROTATION_CONSEC_CYCLES = 2  # backward-compat alias
    STAGNATION_FORCE_ROTATION_MIN_AGE_MULT = 2.5
    STAGNATION_FORCE_ROTATION_PNL_BAND = 0.0030  # backward-compat alias
    STAGNATION_FORCE_ROTATION_SELL_FRACTION = 0.40

    # ---------- TP/SL snowball asymmetry ----------
    TPSL_SNOWBALL_ASYMMETRY_ENABLED = True
    TP_PHASE_MULTIPLIERS = {
        "PHASE_1_SEED": 1.20,
        "PHASE_2_TRACTION": 1.54,
        "PHASE_3_ACCELERATE": 1.60,
        "PHASE_4_SNOWBALL": 1.30,
    }
    SL_PHASE_MULTIPLIERS = {
        "PHASE_1_SEED": 1.00,
        "PHASE_2_TRACTION": 0.95,
        "PHASE_3_ACCELERATE": 0.90,
        "PHASE_4_SNOWBALL": 0.75,
    }
    COMPOUNDING_TPSL_PHASE_PROFILES = {
        "PHASE_1_SEED": {"tp_mult": 1.20, "sl_mult": 1.00, "rr_bonus": 0.04},
        "PHASE_2_TRACTION": {"tp_mult": 1.54, "sl_mult": 0.95, "rr_bonus": 0.16},
        "PHASE_3_ACCELERATE": {"tp_mult": 1.60, "sl_mult": 0.90, "rr_bonus": 0.22},
        "PHASE_4_SNOWBALL": {"tp_mult": 1.30, "sl_mult": 0.75, "rr_bonus": 0.10},
    }

    ALLOW_DEBUG_SYMBOLS: bool = False
    TRENDHUNTER_MIN_DATA = 50
    TRENDHUNTER_EMA_SHORT = 12
    TRENDHUNTER_EMA_LONG = 26
    TRENDHUNTER_EMA_SIGNAL = 9
    AUTO_TRAIN = True
    SWING_AUTO_TRAIN = True
    TREND_AUTO_TRAIN = True
    ENABLE_COT_VALIDATION = True  # keep CoT guard path enabled
    DEFAULT_AGENT_DIRECT_EXECUTION = False  # global safe default; agents read AGENT_EXECUTION

    # ═══════════════════════════════════════════════════════════════════════════════
    # CAPITAL VELOCITY OPTIMIZER CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════════════════
    ENABLE_CAPITAL_VELOCITY_OPTIMIZATION = True     # Master switch
    VELOCITY_GAP_THRESHOLD_PCT = 0.5                # Min % per hour to consider rotating
    VELOCITY_MIN_POSITION_AGE_HOURS = 0.25          # Min hold time (15 min)
    VELOCITY_HOLDING_COST_FEE_BPS = 10.0            # Estimated round-trip fee in basis points
    VELOCITY_CONFIDENCE_MIN = 0.55                  # Min ML confidence to estimate velocity

    # ----- helper -----
    @staticmethod
    def _mask(s: str, head: int = 4, tail: int = 2) -> str:
        if not s:
            return ""
        if len(s) <= head + tail:
            return MASK
        return f"{s[:head]}{MASK}{s[-tail:]}"

    def __init__(self, config_file_path: Optional[str] = None):
        logger.info("⚙️ Config initialized (LIVE-safe defaults). Loading .env overrides...")

        # --- P9 SHADOW MODE: Load trading mode (live or shadow) ---
        self.trading_mode = os.getenv("TRADING_MODE", "live").lower()

        # --- read early knobs with class defaults as fallbacks
        self.EXEC_PROBE_QUOTE = float(os.getenv("EXEC_PROBE_QUOTE", str(Config.EXEC_PROBE_QUOTE)))
        self.LIQ_ORCH_MIN_USDT_FLOOR = float(os.getenv("LIQ_ORCH_MIN_USDT_FLOOR", str(Config.LIQ_ORCH_MIN_USDT_FLOOR)))
        self.LIQ_ORCH_MIN_USDT_TARGET = float(os.getenv("LIQ_ORCH_MIN_USDT_TARGET", str(Config.LIQ_ORCH_MIN_USDT_TARGET)))

        # Optional orchestration defaults (env-bridge) so AppContext helpers can read them via _cfg()
        try:
            # Prefer agent mode by default; AppContext reads this via _cfg()
            os.environ.setdefault("LIQUIDITY_ORCHESTRATION_MODE", "agent")
            os.environ.setdefault("LIQ_ORCH_ENABLE", "true")
            # Trigger orchestration only for meaningful gaps; avoid spam via debounce
            os.environ.setdefault("LIQ_ORCH_MIN_GAP_USDT", "1.50")
            os.environ.setdefault("LIQ_ORCH_CONSECUTIVE_TICKS", "2")
            os.environ.setdefault("LIQ_ORCH_DEBOUNCE_SEC", "240")
            os.environ.setdefault("LIQUIDITY_NOTICE_COOLDOWN_SEC", "120")
            # CashRouter buffer headroom (used only if mode=cash_router)
            os.environ.setdefault("CASH_ROUTER_USDT_BUFFER", "0.10")
            # Startup / phase timing tweaks
            os.environ.setdefault("P4_MARKET_DATA_START_TIMEOUT_SEC", "90.0")
            # Poll interval for startup gates; AppContext will use if not provided explicitly
            os.environ.setdefault("STARTUP_POLL_INTERVAL_SEC", "2.0")
        except Exception:
            pass

        # Bridge affordability scout knobs for AppContext
        try:
            os.environ.setdefault("AFFORD_SCOUT_ENABLE", "true")
            os.environ.setdefault("AFFORD_SCOUT_INTERVAL_SEC", "15")
            os.environ.setdefault("AFFORD_SCOUT_JITTER_PCT", "0.10")
            os.environ.setdefault("AFFORD_SCOUT_ONLY_USDT", "true")
        except Exception:
            pass

        # Bridge CashRouter knobs (provide safe LIVE defaults; overridable via .env)
        try:
            os.environ.setdefault("CR_ENABLE", "true")
            os.environ.setdefault("CR_ALLOW_POSITION_FREE", "true")
            # Keep core assets protected; allow freeing from smaller alts
            os.environ.setdefault("CR_PROTECTED_ASSETS", "BTC,ETH,BNB,SOL,USDT")
            # Execution + accounting safety
            os.environ.setdefault("CR_MAX_ACTIONS", "3")
            os.environ.setdefault("CR_SWEEP_DUST_MIN", "1.0")
            os.environ.setdefault("CR_PRICE_SLIPPAGE_BPS", "15")
            os.environ.setdefault("CR_FEE_BPS", "10")
            os.environ.setdefault("CR_EPSILON_USDT", "0.02")
            # Prefer using official stable conversions when available
            os.environ.setdefault("CR_ENABLE_REDEEM_STABLES", "true")
            os.environ.setdefault("CR_STABLE_SYMBOLS", "FDUSD,BUSD,USDC")
            # Spread cap for router paths (used as fallback if LIQUIDATION_SPREAD_CAP_BPS not set)
            os.environ.setdefault("CR_SPREAD_CAP_BPS", "12.0")
        except Exception:
            pass

        # Exit feasibility defaults (fall back to CashRouter buffers)
        self.EXIT_FEE_BPS = float(os.getenv("EXIT_FEE_BPS", os.getenv("CR_FEE_BPS", str(Config.EXIT_FEE_BPS))))
        self.EXIT_SLIPPAGE_BPS = float(os.getenv("EXIT_SLIPPAGE_BPS", os.getenv("CR_PRICE_SLIPPAGE_BPS", str(Config.EXIT_SLIPPAGE_BPS))))

        # Mirror global liquidation spread cap to CashRouter cap unless CR cap explicitly set
        try:
            _liq_cap = os.getenv("LIQUIDATION_SPREAD_CAP_BPS")
            if _liq_cap and not os.getenv("CR_SPREAD_CAP_BPS"):
                os.environ["CR_SPREAD_CAP_BPS"] = _liq_cap
        except Exception:
            pass

        # Build a small CASH_ROUTER namespace for downstream logging/inspection
        _protected_csv = os.getenv("CR_PROTECTED_ASSETS", "BTC,ETH,BNB,SOL,USDT")
        _protected = [s.strip().upper() for s in _protected_csv.split(",") if s.strip()]
        _stables_csv = os.getenv("CR_STABLE_SYMBOLS", "FDUSD,BUSD,USDC")
        _stables = [s.strip().upper() for s in _stables_csv.split(",") if s.strip()]

        self.CASH_ROUTER = SimpleNamespace(
            ENABLE=os.getenv("CR_ENABLE", "true").lower() == "true",
            ALLOW_POSITION_FREE=os.getenv("CR_ALLOW_POSITION_FREE", "true").lower() == "true",
            PROTECTED_ASSETS=_protected,
            MAX_ACTIONS=int(os.getenv("CR_MAX_ACTIONS", "3")),
            SWEEP_DUST_MIN=float(os.getenv("CR_SWEEP_DUST_MIN", "1.0")),
            ENABLE_REDEEM_STABLES=os.getenv("CR_ENABLE_REDEEM_STABLES", "true").lower() == "true",
            STABLE_SYMBOLS=_stables,
            SPREAD_CAP_BPS=float(os.getenv("CR_SPREAD_CAP_BPS", os.getenv("LIQUIDATION_SPREAD_CAP_BPS", "12.0"))),
            EPSILON_USDT=float(os.getenv("CR_EPSILON_USDT", "0.02")),
        )

        self.BASE_CURRENCY = os.getenv("BASE_CURRENCY", "USDT").upper()

        # ---------- BINANCE EXCHANGE CONFIGURATION ----------
        # API Keys and testnet configuration MUST be loaded here
        self.TESTNET_MODE = os.getenv("BINANCE_TESTNET", "false").lower() == "true"
        self.BINANCE_ACCOUNT_TYPE = os.getenv("BINANCE_ACCOUNT_TYPE", "spot").lower()
        self.BINANCE_REGION = os.getenv("BINANCE_REGION", "global").lower()
        
        # Load API keys for LIVE or TESTNET based on TESTNET_MODE
        if self.TESTNET_MODE:
            self.BINANCE_API_KEY = os.getenv("BINANCE_TESTNET_API_KEY", "")
            self.BINANCE_API_SECRET = os.getenv("BINANCE_TESTNET_API_SECRET_HMAC", "") or os.getenv("BINANCE_TESTNET_API_SECRET", "")
            self.BINANCE_API_SECRET_ED25519 = os.getenv("BINANCE_TESTNET_API_SECRET_ED25519", "")
        else:
            self.BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
            self.BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET_HMAC", "") or os.getenv("BINANCE_API_SECRET", "")
            self.BINANCE_API_SECRET_ED25519 = os.getenv("BINANCE_API_SECRET_ED25519", "")
        
        # Determine API base URL
        if self.TESTNET_MODE:
            self.BINANCE_BASE_URL = "https://testnet.binance.vision"
        else:
            if self.BINANCE_ACCOUNT_TYPE == "futures":
                self.BINANCE_BASE_URL = "https://fapi.binance.com"
            elif self.BINANCE_REGION == "us":
                self.BINANCE_BASE_URL = "https://api.binance.us"
            else:
                self.BINANCE_BASE_URL = "https://api.binance.com"
        
        logger.info(f"[Config] Binance loaded: testnet={self.TESTNET_MODE}, account_type={self.BINANCE_ACCOUNT_TYPE}, region={self.BINANCE_REGION}, api_key_loaded={len(self.BINANCE_API_KEY) > 0}")

        # ---------- Universe & allocation defaults ----------
        # Structural separation:
        # - MAX_UNIVERSE_SYMBOLS controls discovery/watchlist breadth.
        # - MAX_POSITIONS_TOTAL controls concurrent capital deployment.
        self.MAX_UNIVERSE_SYMBOLS = int(
            os.getenv("MAX_UNIVERSE_SYMBOLS", str(getattr(Config, "MAX_UNIVERSE_SYMBOLS", 30)))
        )
        if self.MAX_UNIVERSE_SYMBOLS <= 0:
            self.MAX_UNIVERSE_SYMBOLS = int(getattr(Config, "MAX_UNIVERSE_SYMBOLS", 30))
        # Legacy aliases kept for compatibility with existing modules.
        self.discovery_top_n_symbols = int(
            os.getenv("DISCOVERY_TOP_N_SYMBOLS", str(self.MAX_UNIVERSE_SYMBOLS))
        )
        self.MAX_ACTIVE_SYMBOLS = int(
            os.getenv("MAX_ACTIVE_SYMBOLS", str(self.MAX_UNIVERSE_SYMBOLS))
        )
        self.MAX_POSITIONS_TOTAL = int(
            os.getenv("MAX_POSITIONS_TOTAL", str(getattr(Config, "MAX_POSITIONS_TOTAL", 2)))
        )
        if self.MAX_POSITIONS_TOTAL <= 0:
            self.MAX_POSITIONS_TOTAL = int(getattr(Config, "MAX_POSITIONS_TOTAL", 2))

        # Capital governor thresholds (override-able via env)
        self.CAPITAL_MICRO_THRESHOLD = float(
            os.getenv("CAPITAL_MICRO_THRESHOLD", str(getattr(Config, "CAPITAL_MICRO_THRESHOLD", 500.0)))
        )
        self.CAPITAL_SMALL_THRESHOLD = float(
            os.getenv("CAPITAL_SMALL_THRESHOLD", str(getattr(Config, "CAPITAL_SMALL_THRESHOLD", 2000.0)))
        )
        self.CAPITAL_MEDIUM_THRESHOLD = float(
            os.getenv("CAPITAL_MEDIUM_THRESHOLD", str(getattr(Config, "CAPITAL_MEDIUM_THRESHOLD", 10000.0)))
        )
        self.CAPITAL_MICRO_MAX_ACTIVE_SYMBOLS = int(
            os.getenv("CAPITAL_MICRO_MAX_ACTIVE_SYMBOLS", str(getattr(Config, "CAPITAL_MICRO_MAX_ACTIVE_SYMBOLS", 3)))
        )
        self.CAPITAL_MICRO_CORE_PAIRS = int(
            os.getenv("CAPITAL_MICRO_CORE_PAIRS", str(getattr(Config, "CAPITAL_MICRO_CORE_PAIRS", 2)))
        )
        self.CAPITAL_MICRO_MAX_ROTATING_SLOTS = int(
            os.getenv("CAPITAL_MICRO_MAX_ROTATING_SLOTS", str(getattr(Config, "CAPITAL_MICRO_MAX_ROTATING_SLOTS", 1)))
        )
        self.CAPITAL_MICRO_MAX_CONCURRENT_POSITIONS = int(
            os.getenv("CAPITAL_MICRO_MAX_CONCURRENT_POSITIONS", str(getattr(Config, "CAPITAL_MICRO_MAX_CONCURRENT_POSITIONS", 2)))
        )
        self.CAPITAL_MICRO_ALLOW_ROTATION = os.getenv(
            "CAPITAL_MICRO_ALLOW_ROTATION", str(getattr(Config, "CAPITAL_MICRO_ALLOW_ROTATION", True))
        ).lower() == "true"
        self.CAPITAL_MICRO_SYMBOL_REPLACEMENT_MULTIPLIER = float(
            os.getenv(
                "CAPITAL_MICRO_SYMBOL_REPLACEMENT_MULTIPLIER",
                str(getattr(Config, "CAPITAL_MICRO_SYMBOL_REPLACEMENT_MULTIPLIER", 1.35)),
            )
        )
        self.CAPITAL_MICRO_SOFT_LOCK_DURATION_SEC = int(
            os.getenv(
                "CAPITAL_MICRO_SOFT_LOCK_DURATION_SEC",
                str(getattr(Config, "CAPITAL_MICRO_SOFT_LOCK_DURATION_SEC", 3600)),
            )
        )
        self.CAPITAL_MICRO_ENFORCE_LIVENESS = os.getenv(
            "CAPITAL_MICRO_ENFORCE_LIVENESS",
            str(getattr(Config, "CAPITAL_MICRO_ENFORCE_LIVENESS", True)),
        ).lower() == "true"
        self.CAPITAL_MICRO_ADAPTIVE_CAPACITY_ENABLED = os.getenv(
            "CAPITAL_MICRO_ADAPTIVE_CAPACITY_ENABLED",
            str(getattr(Config, "CAPITAL_MICRO_ADAPTIVE_CAPACITY_ENABLED", True)),
        ).lower() == "true"
        self.CAPITAL_MICRO_ADAPTIVE_PRESSURE_TRIGGER = int(
            os.getenv(
                "CAPITAL_MICRO_ADAPTIVE_PRESSURE_TRIGGER",
                str(getattr(Config, "CAPITAL_MICRO_ADAPTIVE_PRESSURE_TRIGGER", 6)),
            )
        )
        self.CAPITAL_MICRO_ADAPTIVE_WINDOW_SEC = float(
            os.getenv(
                "CAPITAL_MICRO_ADAPTIVE_WINDOW_SEC",
                str(getattr(Config, "CAPITAL_MICRO_ADAPTIVE_WINDOW_SEC", 300.0)),
            )
        )
        self.CAPITAL_MICRO_ADAPTIVE_MAX_CONCURRENT_POSITIONS = int(
            os.getenv(
                "CAPITAL_MICRO_ADAPTIVE_MAX_CONCURRENT_POSITIONS",
                str(getattr(Config, "CAPITAL_MICRO_ADAPTIVE_MAX_CONCURRENT_POSITIONS", 2)),
            )
        )
        self.CAPITAL_MICRO_ADAPTIVE_MAX_ROTATING_SLOTS = int(
            os.getenv(
                "CAPITAL_MICRO_ADAPTIVE_MAX_ROTATING_SLOTS",
                str(getattr(Config, "CAPITAL_MICRO_ADAPTIVE_MAX_ROTATING_SLOTS", 1)),
            )
        )
        self.CAPITAL_MICRO_ADAPTIVE_SOFT_LOCK_DURATION_SEC = int(
            os.getenv(
                "CAPITAL_MICRO_ADAPTIVE_SOFT_LOCK_DURATION_SEC",
                str(getattr(Config, "CAPITAL_MICRO_ADAPTIVE_SOFT_LOCK_DURATION_SEC", 900)),
            )
        )
        self.CAPITAL_MICRO_ADAPTIVE_SYMBOL_REPLACEMENT_MULTIPLIER = float(
            os.getenv(
                "CAPITAL_MICRO_ADAPTIVE_SYMBOL_REPLACEMENT_MULTIPLIER",
                str(getattr(Config, "CAPITAL_MICRO_ADAPTIVE_SYMBOL_REPLACEMENT_MULTIPLIER", 1.35)),
            )
        )
        self.CAPITAL_MICRO_ADAPTIVE_SELL_DEADLOCK_ENABLED = os.getenv(
            "CAPITAL_MICRO_ADAPTIVE_SELL_DEADLOCK_ENABLED",
            str(getattr(Config, "CAPITAL_MICRO_ADAPTIVE_SELL_DEADLOCK_ENABLED", True)),
        ).lower() == "true"
        self.CAPITAL_MICRO_ADAPTIVE_SELL_DEADLOCK_TRIGGER = int(
            os.getenv(
                "CAPITAL_MICRO_ADAPTIVE_SELL_DEADLOCK_TRIGGER",
                str(getattr(Config, "CAPITAL_MICRO_ADAPTIVE_SELL_DEADLOCK_TRIGGER", 5)),
            )
        )
        self.CAPITAL_MICRO_ADAPTIVE_SELL_DEADLOCK_WINDOW_SEC = float(
            os.getenv(
                "CAPITAL_MICRO_ADAPTIVE_SELL_DEADLOCK_WINDOW_SEC",
                str(getattr(Config, "CAPITAL_MICRO_ADAPTIVE_SELL_DEADLOCK_WINDOW_SEC", 300.0)),
            )
        )
        self.CAPITAL_MICRO_ADAPTIVE_SELL_DEADLOCK_REASONS = str(
            os.getenv(
                "CAPITAL_MICRO_ADAPTIVE_SELL_DEADLOCK_REASONS",
                str(getattr(Config, "CAPITAL_MICRO_ADAPTIVE_SELL_DEADLOCK_REASONS", "")),
            )
        )

        # ---------- Phase 1: Safe Upgrade - Symbol Rotation ----------
        self.BOOTSTRAP_SOFT_LOCK_ENABLED = os.getenv("BOOTSTRAP_SOFT_LOCK_ENABLED", "true").lower() == "true"
        self.BOOTSTRAP_SOFT_LOCK_DURATION_SEC = int(
            os.getenv("BOOTSTRAP_SOFT_LOCK_DURATION_SEC", "3600")
        )
        self.SYMBOL_REPLACEMENT_MULTIPLIER = float(
            os.getenv("SYMBOL_REPLACEMENT_MULTIPLIER", "1.10")
        )
        self.MIN_ACTIVE_SYMBOLS = int(
            os.getenv("MIN_ACTIVE_SYMBOLS", "3")
        )
        
        # Symbol screener (Phase 1)
        self.SCREENER_MIN_PROPOSALS = int(
            os.getenv("SCREENER_MIN_PROPOSALS", "20")
        )
        self.SCREENER_MAX_PROPOSALS = int(
            os.getenv("SCREENER_MAX_PROPOSALS", "30")
        )
        self.SCREENER_MIN_VOLUME = float(
            os.getenv("SCREENER_MIN_VOLUME", "100000")
        )
        self.SCREENER_MIN_PRICE = float(
            os.getenv("SCREENER_MIN_PRICE", "0.01")
        )

        # Dynamic Symbol Discovery Settings
        self.DISCOVERY = SimpleNamespace(
            # Discovery breadth is decoupled from allocation limits.
            TOP_N_SYMBOLS=int(os.getenv("DISCOVERY_TOP_N_SYMBOLS", str(self.MAX_UNIVERSE_SYMBOLS))),
            ACCEPT_NEW_SYMBOLS=os.getenv("DISCOVERY_ACCEPT_NEW_SYMBOLS", "true").lower() == "true",
            MIN_24H_VOL=float(os.getenv("MIN_TRADE_VOLUME", "100000.0")),
            # Dynamic blacklist for leveraged/ETP tokens
            DISALLOW_SUFFIXES=os.getenv("DISALLOW_SUFFIXES", f"UP{self.BASE_CURRENCY},DOWN{self.BASE_CURRENCY},BULL{self.BASE_CURRENCY},BEAR{self.BASE_CURRENCY}").split(","),
        )

        # AI Model Settings
        self.MODEL_TYPE = os.getenv("MODEL_TYPE", "LSTM") # Options: LSTM, GRU

        # Accept SYMBOLS from env (comma-separated), normalize to UPPER and strip
        _symbols_env = os.getenv("SYMBOLS", "")
        self.SYMBOLS = [s.strip().upper() for s in _symbols_env.split(",") if s.strip()]

        # Database Settings
        self.DATABASE_PATH = os.getenv("DATABASE_PATH", "octivault_trader.db")

        # ---------- Modes, region/type, keys, endpoints ... (unchanged)

        # ---------- Trading Capital / KPI ----------
        base_cap_str = os.getenv("BASE_CAPITAL", "")
        self.BASE_CAPITAL = float(base_cap_str) if base_cap_str else None
        self.BASE_TARGET_PER_HOUR = float(os.getenv("BASE_TARGET_PER_HOUR", 20.0))


        # ---------- Equity-tier scaling ----------
        self.EQUITY_TIER_BLOCK_ON_RECOVERY = os.getenv(
            "EQUITY_TIER_BLOCK_ON_RECOVERY",
            str(Config.EQUITY_TIER_BLOCK_ON_RECOVERY)
        ).lower() == "true"
        self.EQUITY_TIER_BLOCK_ON_LIQUIDATION = os.getenv(
            "EQUITY_TIER_BLOCK_ON_LIQUIDATION",
            str(Config.EQUITY_TIER_BLOCK_ON_LIQUIDATION)
        ).lower() == "true"
        self.EQUITY_TIER_BLOCK_ON_OPEN_POSITIONS = os.getenv(
            "EQUITY_TIER_BLOCK_ON_OPEN_POSITIONS",
            str(Config.EQUITY_TIER_BLOCK_ON_OPEN_POSITIONS)
        ).lower() == "true"
        self.EQUITY_TIER_USE_REALIZED_ONLY = os.getenv(
            "EQUITY_TIER_USE_REALIZED_ONLY",
            str(Config.EQUITY_TIER_USE_REALIZED_ONLY)
        ).lower() == "true"
        self.SCALING_EQUITY_TIERS = Config.SCALING_EQUITY_TIERS

        # ---------- Compounding growth curve (Phase 1 -> Phase 4) ----------
        self.COMPOUNDING_ENABLED = os.getenv(
            "COMPOUNDING_ENABLED",
            str(Config.COMPOUNDING_ENABLED),
        ).lower() == "true"
        # Backward-compatible alias for old flag consumers.
        self.COMPOUNDING_GROWTH_CURVE_ENABLED = os.getenv(
            "COMPOUNDING_GROWTH_CURVE_ENABLED",
            str(self.COMPOUNDING_ENABLED and Config.COMPOUNDING_GROWTH_CURVE_ENABLED),
        ).lower() == "true"

        growth_json = os.getenv("COMPOUNDING_GROWTH_PHASES_JSON", "").strip()
        if growth_json:
            try:
                parsed = json.loads(growth_json)
                self.COMPOUNDING_GROWTH_PHASES = parsed if isinstance(parsed, list) and parsed else Config.COMPOUNDING_GROWTH_PHASES
            except Exception:
                self.COMPOUNDING_GROWTH_PHASES = Config.COMPOUNDING_GROWTH_PHASES
        else:
            self.COMPOUNDING_GROWTH_PHASES = Config.COMPOUNDING_GROWTH_PHASES

        thresholds_json = os.getenv("GROWTH_PHASE_THRESHOLDS_JSON", "").strip()
        if thresholds_json:
            try:
                parsed_thresholds = json.loads(thresholds_json)
                self.GROWTH_PHASE_THRESHOLDS = parsed_thresholds if isinstance(parsed_thresholds, list) and parsed_thresholds else Config.GROWTH_PHASE_THRESHOLDS
            except Exception:
                self.GROWTH_PHASE_THRESHOLDS = Config.GROWTH_PHASE_THRESHOLDS
        else:
            self.GROWTH_PHASE_THRESHOLDS = Config.GROWTH_PHASE_THRESHOLDS

        size_map_json = os.getenv("PHASE_SIZE_MULTIPLIERS_JSON", "").strip()
        if size_map_json:
            try:
                parsed_size = json.loads(size_map_json)
                self.PHASE_SIZE_MULTIPLIERS = parsed_size if isinstance(parsed_size, dict) and parsed_size else Config.PHASE_SIZE_MULTIPLIERS
            except Exception:
                self.PHASE_SIZE_MULTIPLIERS = Config.PHASE_SIZE_MULTIPLIERS
        else:
            self.PHASE_SIZE_MULTIPLIERS = Config.PHASE_SIZE_MULTIPLIERS

        phase_cap_json = os.getenv("PHASE_MAX_TRADE_CAP_JSON", "").strip()
        if phase_cap_json:
            try:
                parsed_cap = json.loads(phase_cap_json)
                self.PHASE_MAX_TRADE_CAP = parsed_cap if isinstance(parsed_cap, dict) and parsed_cap else Config.PHASE_MAX_TRADE_CAP
            except Exception:
                self.PHASE_MAX_TRADE_CAP = Config.PHASE_MAX_TRADE_CAP
        else:
            self.PHASE_MAX_TRADE_CAP = Config.PHASE_MAX_TRADE_CAP

        self.COMPOUNDING_MAX_DRAWDOWN_PCT = float(
            os.getenv(
                "COMPOUNDING_MAX_DRAWDOWN_PCT",
                str(Config.COMPOUNDING_MAX_DRAWDOWN_PCT),
            )
        )
        self.COMPOUNDING_MIN_POSITIVE_STREAK = int(
            os.getenv(
                "COMPOUNDING_MIN_POSITIVE_STREAK",
                str(Config.COMPOUNDING_MIN_POSITIVE_STREAK),
            )
        )

        # ---------- Dynamic position sizing ----------
        self.DYNAMIC_POSITION_SIZING_ENABLED = os.getenv(
            "DYNAMIC_POSITION_SIZING_ENABLED",
            str(Config.DYNAMIC_POSITION_SIZING_ENABLED),
        ).lower() == "true"
        self.DYNAMIC_SIZE_CONF_FLOOR_MULT = float(
            os.getenv("DYNAMIC_SIZE_CONF_FLOOR_MULT", str(Config.DYNAMIC_SIZE_CONF_FLOOR_MULT))
        )
        self.DYNAMIC_SIZE_CONF_CEIL_MULT = float(
            os.getenv("DYNAMIC_SIZE_CONF_CEIL_MULT", str(Config.DYNAMIC_SIZE_CONF_CEIL_MULT))
        )
        self.DYNAMIC_SIZE_VOL_FLOOR_MULT = float(
            os.getenv("DYNAMIC_SIZE_VOL_FLOOR_MULT", str(Config.DYNAMIC_SIZE_VOL_FLOOR_MULT))
        )
        self.DYNAMIC_SIZE_VOL_CEIL_MULT = float(
            os.getenv("DYNAMIC_SIZE_VOL_CEIL_MULT", str(Config.DYNAMIC_SIZE_VOL_CEIL_MULT))
        )
        self.DYNAMIC_SIZE_MOMENTUM_MULT = float(
            os.getenv("DYNAMIC_SIZE_MOMENTUM_MULT", str(Config.DYNAMIC_SIZE_MOMENTUM_MULT))
        )
        self.DYNAMIC_SIZE_BLEND_WEIGHT = float(
            os.getenv("DYNAMIC_SIZE_BLEND_WEIGHT", str(Config.DYNAMIC_SIZE_BLEND_WEIGHT))
        )
        self.DYNAMIC_SIZE_UPSIDE_CAP_MULT = float(
            os.getenv("DYNAMIC_SIZE_UPSIDE_CAP_MULT", str(Config.DYNAMIC_SIZE_UPSIDE_CAP_MULT))
        )
        self.DYNAMIC_SIZE_MOMENTUM_LOOKBACK_TRADES = int(
            os.getenv(
                "DYNAMIC_SIZE_MOMENTUM_LOOKBACK_TRADES",
                str(Config.DYNAMIC_SIZE_MOMENTUM_LOOKBACK_TRADES),
            )
        )
        self.DYNAMIC_RISK_BUDGET_PCT = float(
            os.getenv("DYNAMIC_RISK_BUDGET_PCT", str(Config.DYNAMIC_RISK_BUDGET_PCT))
        )
        self.DYNAMIC_RISK_BUDGET_PCT_TIER_B = float(
            os.getenv("DYNAMIC_RISK_BUDGET_PCT_TIER_B", str(Config.DYNAMIC_RISK_BUDGET_PCT_TIER_B))
        )
        self.DYNAMIC_CONFIDENCE_MIN = float(
            os.getenv("DYNAMIC_CONFIDENCE_MIN", str(Config.DYNAMIC_CONFIDENCE_MIN))
        )
        self.DYNAMIC_CONFIDENCE_MAX = float(
            os.getenv("DYNAMIC_CONFIDENCE_MAX", str(Config.DYNAMIC_CONFIDENCE_MAX))
        )
        self.MIN_EXECUTION_CONFIDENCE = float(
            os.getenv("MIN_EXECUTION_CONFIDENCE", "0.6")
        )
        self.MIN_SIGNAL_CONF = float(
            os.getenv("MIN_SIGNAL_CONF", "0.5")
        )
        self.STABLE_RISK_BUDGET_ENABLED = os.getenv(
            "STABLE_RISK_BUDGET_ENABLED",
            str(Config.STABLE_RISK_BUDGET_ENABLED),
        ).lower() == "true"
        self.STABLE_RISK_BUDGET_MULT = float(
            os.getenv("STABLE_RISK_BUDGET_MULT", str(Config.STABLE_RISK_BUDGET_MULT))
        )
        self.STABLE_RISK_MIN_POSITIVE_STREAK = int(
            os.getenv(
                "STABLE_RISK_MIN_POSITIVE_STREAK",
                str(Config.STABLE_RISK_MIN_POSITIVE_STREAK),
            )
        )
        self.STABLE_RISK_MAX_DRAWDOWN_PCT = float(
            os.getenv(
                "STABLE_RISK_MAX_DRAWDOWN_PCT",
                str(Config.STABLE_RISK_MAX_DRAWDOWN_PCT),
            )
        )
        self.ADAPTIVE_CAPITAL_ENGINE_ENABLED = os.getenv(
            "ADAPTIVE_CAPITAL_ENGINE_ENABLED",
            str(Config.ADAPTIVE_CAPITAL_ENGINE_ENABLED),
        ).lower() == "true"
        self.ADAPTIVE_PERF_REVIEW_SEC = float(
            os.getenv("ADAPTIVE_PERF_REVIEW_SEC", str(Config.ADAPTIVE_PERF_REVIEW_SEC))
        )
        self.ADAPTIVE_RISK_FRACTION_MIN = float(
            os.getenv("ADAPTIVE_RISK_FRACTION_MIN", str(Config.ADAPTIVE_RISK_FRACTION_MIN))
        )
        self.ADAPTIVE_RISK_FRACTION_MAX = float(
            os.getenv("ADAPTIVE_RISK_FRACTION_MAX", str(Config.ADAPTIVE_RISK_FRACTION_MAX))
        )
        self.ADAPTIVE_DRAWDOWN_SOFT_PCT = float(
            os.getenv("ADAPTIVE_DRAWDOWN_SOFT_PCT", str(Config.ADAPTIVE_DRAWDOWN_SOFT_PCT))
        )
        self.ADAPTIVE_DRAWDOWN_HARD_PCT = float(
            os.getenv("ADAPTIVE_DRAWDOWN_HARD_PCT", str(Config.ADAPTIVE_DRAWDOWN_HARD_PCT))
        )
        self.ADAPTIVE_HIGH_VOL_PCT = float(
            os.getenv("ADAPTIVE_HIGH_VOL_PCT", str(Config.ADAPTIVE_HIGH_VOL_PCT))
        )
        self.ADAPTIVE_LOW_VOL_PCT = float(
            os.getenv("ADAPTIVE_LOW_VOL_PCT", str(Config.ADAPTIVE_LOW_VOL_PCT))
        )
        self.ADAPTIVE_THROUGHPUT_LOW_RATIO = float(
            os.getenv("ADAPTIVE_THROUGHPUT_LOW_RATIO", str(Config.ADAPTIVE_THROUGHPUT_LOW_RATIO))
        )
        self.ADAPTIVE_IDLE_FREE_CAPITAL_PCT = float(
            os.getenv("ADAPTIVE_IDLE_FREE_CAPITAL_PCT", str(Config.ADAPTIVE_IDLE_FREE_CAPITAL_PCT))
        )
        self.ADAPTIVE_IDLE_TIME_SEC = float(
            os.getenv("ADAPTIVE_IDLE_TIME_SEC", str(Config.ADAPTIVE_IDLE_TIME_SEC))
        )
        self.ADAPTIVE_WIN_STREAK_TRADES = int(
            os.getenv("ADAPTIVE_WIN_STREAK_TRADES", str(Config.ADAPTIVE_WIN_STREAK_TRADES))
        )
        self.ADAPTIVE_LOSS_STREAK_TRADES = int(
            os.getenv("ADAPTIVE_LOSS_STREAK_TRADES", str(Config.ADAPTIVE_LOSS_STREAK_TRADES))
        )
        self.ADAPTIVE_WIN_STREAK_RISK_BONUS = float(
            os.getenv("ADAPTIVE_WIN_STREAK_RISK_BONUS", str(Config.ADAPTIVE_WIN_STREAK_RISK_BONUS))
        )
        self.ADAPTIVE_LOSS_STREAK_RISK_PENALTY = float(
            os.getenv("ADAPTIVE_LOSS_STREAK_RISK_PENALTY", str(Config.ADAPTIVE_LOSS_STREAK_RISK_PENALTY))
        )
        self.ADAPTIVE_WIN_RATE_BONUS_THRESHOLD = float(
            os.getenv("ADAPTIVE_WIN_RATE_BONUS_THRESHOLD", str(Config.ADAPTIVE_WIN_RATE_BONUS_THRESHOLD))
        )
        self.ADAPTIVE_WIN_RATE_PENALTY_THRESHOLD = float(
            os.getenv("ADAPTIVE_WIN_RATE_PENALTY_THRESHOLD", str(Config.ADAPTIVE_WIN_RATE_PENALTY_THRESHOLD))
        )
        self.ADAPTIVE_WIN_RATE_BONUS = float(
            os.getenv("ADAPTIVE_WIN_RATE_BONUS", str(Config.ADAPTIVE_WIN_RATE_BONUS))
        )
        self.ADAPTIVE_WIN_RATE_PENALTY = float(
            os.getenv("ADAPTIVE_WIN_RATE_PENALTY", str(Config.ADAPTIVE_WIN_RATE_PENALTY))
        )
        self.ADAPTIVE_FEE_GROSS_THRESHOLD = float(
            os.getenv("ADAPTIVE_FEE_GROSS_THRESHOLD", str(Config.ADAPTIVE_FEE_GROSS_THRESHOLD))
        )
        self.ADAPTIVE_FEE_GROSS_BONUS = float(
            os.getenv("ADAPTIVE_FEE_GROSS_BONUS", str(Config.ADAPTIVE_FEE_GROSS_BONUS))
        )
        self.ADAPTIVE_ECON_MIN_NOTIONAL_MULT = float(
            os.getenv("ADAPTIVE_ECON_MIN_NOTIONAL_MULT", str(Config.ADAPTIVE_ECON_MIN_NOTIONAL_MULT))
        )
        self.ADAPTIVE_ECON_TARGET_PROFIT_PCT = float(
            os.getenv("ADAPTIVE_ECON_TARGET_PROFIT_PCT", str(Config.ADAPTIVE_ECON_TARGET_PROFIT_PCT))
        )
        self.ADAPTIVE_MIN_QUOTE_BUFFER_MULT = float(
            os.getenv("ADAPTIVE_MIN_QUOTE_BUFFER_MULT", str(Config.ADAPTIVE_MIN_QUOTE_BUFFER_MULT))
        )

        # ---------- Stagnation forced rotation ----------
        self.STAGNATION_FORCE_ROTATION_ENABLED = os.getenv(
            "STAGNATION_FORCE_ROTATION_ENABLED",
            str(Config.STAGNATION_FORCE_ROTATION_ENABLED),
        ).lower() == "true"
        self.STAGNATION_AGE_SEC = float(
            os.getenv("STAGNATION_AGE_SEC", str(Config.STAGNATION_AGE_SEC))
        )
        self.STAGNATION_PNL_THRESHOLD = float(
            os.getenv("STAGNATION_PNL_THRESHOLD", str(Config.STAGNATION_PNL_THRESHOLD))
        )
        self.STAGNATION_STREAK_LIMIT = int(
            os.getenv("STAGNATION_STREAK_LIMIT", str(Config.STAGNATION_STREAK_LIMIT))
        )
        self.STAGNATION_CONTINUATION_MIN_SCORE = float(
            os.getenv(
                "STAGNATION_CONTINUATION_MIN_SCORE",
                str(Config.STAGNATION_CONTINUATION_MIN_SCORE),
            )
        )
        self.STAGNATION_FORCE_ROTATION_CONSEC_CYCLES = int(
            os.getenv(
                "STAGNATION_FORCE_ROTATION_CONSEC_CYCLES",
                str(self.STAGNATION_STREAK_LIMIT or Config.STAGNATION_FORCE_ROTATION_CONSEC_CYCLES),
            )
        )
        self.STAGNATION_FORCE_ROTATION_MIN_AGE_MULT = float(
            os.getenv(
                "STAGNATION_FORCE_ROTATION_MIN_AGE_MULT",
                str(Config.STAGNATION_FORCE_ROTATION_MIN_AGE_MULT),
            )
        )
        self.STAGNATION_FORCE_ROTATION_PNL_BAND = float(
            os.getenv(
                "STAGNATION_FORCE_ROTATION_PNL_BAND",
                str(self.STAGNATION_PNL_THRESHOLD or Config.STAGNATION_FORCE_ROTATION_PNL_BAND),
            )
        )
        self.STAGNATION_FORCE_ROTATION_SELL_FRACTION = float(
            os.getenv(
                "STAGNATION_FORCE_ROTATION_SELL_FRACTION",
                str(Config.STAGNATION_FORCE_ROTATION_SELL_FRACTION),
            )
        )

        # ---------- TP/SL snowball asymmetry ----------
        self.TPSL_SNOWBALL_ASYMMETRY_ENABLED = os.getenv(
            "TPSL_SNOWBALL_ASYMMETRY_ENABLED",
            str(Config.TPSL_SNOWBALL_ASYMMETRY_ENABLED),
        ).lower() == "true"

        tp_map_json = os.getenv("TP_PHASE_MULTIPLIERS_JSON", "").strip()
        if tp_map_json:
            try:
                parsed_tp_map = json.loads(tp_map_json)
                self.TP_PHASE_MULTIPLIERS = parsed_tp_map if isinstance(parsed_tp_map, dict) and parsed_tp_map else Config.TP_PHASE_MULTIPLIERS
            except Exception:
                self.TP_PHASE_MULTIPLIERS = Config.TP_PHASE_MULTIPLIERS
        else:
            self.TP_PHASE_MULTIPLIERS = Config.TP_PHASE_MULTIPLIERS

        sl_map_json = os.getenv("SL_PHASE_MULTIPLIERS_JSON", "").strip()
        if sl_map_json:
            try:
                parsed_sl_map = json.loads(sl_map_json)
                self.SL_PHASE_MULTIPLIERS = parsed_sl_map if isinstance(parsed_sl_map, dict) and parsed_sl_map else Config.SL_PHASE_MULTIPLIERS
            except Exception:
                self.SL_PHASE_MULTIPLIERS = Config.SL_PHASE_MULTIPLIERS
        else:
            self.SL_PHASE_MULTIPLIERS = Config.SL_PHASE_MULTIPLIERS

        tpsl_profiles_json = os.getenv("COMPOUNDING_TPSL_PHASE_PROFILES_JSON", "").strip()
        if tpsl_profiles_json:
            try:
                parsed_profiles = json.loads(tpsl_profiles_json)
                if isinstance(parsed_profiles, dict) and parsed_profiles:
                    self.COMPOUNDING_TPSL_PHASE_PROFILES = parsed_profiles
                else:
                    self.COMPOUNDING_TPSL_PHASE_PROFILES = Config.COMPOUNDING_TPSL_PHASE_PROFILES
            except Exception:
                self.COMPOUNDING_TPSL_PHASE_PROFILES = Config.COMPOUNDING_TPSL_PHASE_PROFILES
        else:
            self.COMPOUNDING_TPSL_PHASE_PROFILES = Config.COMPOUNDING_TPSL_PHASE_PROFILES

        # ---- Small-cap profile (auto-on for <= $250 unless overridden) ----
        _small_cap_env = os.getenv("SMALL_CAP_PROFILE")
        _small_cap_on = (
            (_small_cap_env or "").lower() == "true"
            if _small_cap_env is not None
            else (self.BASE_CAPITAL is not None and self.BASE_CAPITAL <= 250.0)
        )
        self.SMALL_CAP_PROFILE = _small_cap_on
        if _small_cap_on:
            # Simplify for ~$200–$250 NAV: fewer slots, sane per-trade size, longer re-entry delay
            os.environ.setdefault("MAX_POSITIONS_TOTAL", "2")
            os.environ.setdefault("META_SYMBOL_CONCENTRATION_LIMIT", "1")
            os.environ.setdefault("TRADE_AMOUNT_USDT", "30.0")
            os.environ.setdefault("DEFAULT_PLANNED_QUOTE", "30.0")
            os.environ.setdefault("EMIT_BUY_QUOTE", "30.0")
            os.environ.setdefault("REENTRY_LOCK_SEC", "1200")

        self.TRADE_AMOUNT_USDT = float(os.getenv("TRADE_AMOUNT_USDT", 30.0))
        # Capital floor (NAV-aware): max(ABSOLUTE_MIN_FLOOR, NAV * CAPITAL_FLOOR_PCT)
        self.ABSOLUTE_MIN_FLOOR = float(os.getenv("ABSOLUTE_MIN_FLOOR", "10.0"))
        self.CAPITAL_FLOOR_PCT = float(os.getenv("CAPITAL_FLOOR_PCT", "0.20"))
        # Entry sizing defaults (aligned to unified buy size; override via .env)
        self.MIN_ENTRY_QUOTE_USDT = float(os.getenv("MIN_ENTRY_QUOTE_USDT", str(Config.MIN_ENTRY_QUOTE_USDT)))
        self.DEFAULT_PLANNED_QUOTE = float(os.getenv("DEFAULT_PLANNED_QUOTE", str(Config.DEFAULT_PLANNED_QUOTE)))
        self.EMIT_BUY_QUOTE = float(os.getenv("EMIT_BUY_QUOTE", str(Config.EMIT_BUY_QUOTE)))

        # ---------- Profit-locked re-entry (compounding guard) ----------
        self.PROFIT_LOCK_REENTRY_ENABLED = os.getenv("PROFIT_LOCK_REENTRY_ENABLED", "true").lower() == "true"
        self.PROFIT_LOCK_BASE_QUOTE = float(os.getenv("PROFIT_LOCK_BASE_QUOTE", str(self.DEFAULT_PLANNED_QUOTE)))

        # IMPORTANT: fallback remains small-account friendly (5.0). Override via .env as needed.
        self.MIN_ORDER_USDT = float(os.getenv("MIN_ORDER_USDT", str(Config.MIN_ORDER_USDT)))
        self.SAFE_ENTRY_USDT = float(os.getenv("SAFE_ENTRY_USDT", str(Config.SAFE_ENTRY_USDT)))
        self.MIN_POSITION_VALUE_USDT = float(os.getenv("MIN_POSITION_VALUE_USDT", str(Config.MIN_POSITION_VALUE_USDT)))
        self.MIN_POSITION_USDT = float(os.getenv("MIN_POSITION_USDT", str(Config.MIN_POSITION_USDT)))
        self.MIN_POSITION_MIN_NOTIONAL_MULT = float(os.getenv("MIN_POSITION_MIN_NOTIONAL_MULT", str(Config.MIN_POSITION_MIN_NOTIONAL_MULT)))
        self.MIN_ENTRY_USDT = float(os.getenv("MIN_ENTRY_USDT", str(Config.MIN_ENTRY_USDT)))
        self.MIN_TRADE_QUOTE = float(os.getenv("MIN_TRADE_QUOTE", str(Config.MIN_TRADE_QUOTE)))
        self.MAX_TRADE_QUOTE = float(os.getenv("MAX_TRADE_QUOTE", str(Config.MAX_TRADE_QUOTE)))
        self.MIN_SIGNIFICANT_POSITION_USDT = float(
            os.getenv("MIN_SIGNIFICANT_POSITION_USDT", str(Config.MIN_SIGNIFICANT_POSITION_USDT))
        )
        self.SIGNIFICANT_POSITION_FLOOR = float(
            os.getenv("SIGNIFICANT_POSITION_FLOOR", str(self.MIN_SIGNIFICANT_POSITION_USDT))
        )
        if self.SIGNIFICANT_POSITION_FLOOR <= 0:
            self.SIGNIFICANT_POSITION_FLOOR = float(self.MIN_SIGNIFICANT_POSITION_USDT)
        # Keep legacy key aligned with canonical significant floor.
        self.MIN_SIGNIFICANT_POSITION_USDT = float(self.SIGNIFICANT_POSITION_FLOOR)
        if self.MAX_TRADE_QUOTE and self.MAX_TRADE_QUOTE < self.MIN_TRADE_QUOTE:
            logger.warning(
                "MAX_TRADE_QUOTE (%.2f) < MIN_TRADE_QUOTE (%.2f). Bumping max to min.",
                self.MAX_TRADE_QUOTE,
                self.MIN_TRADE_QUOTE,
            )
            self.MAX_TRADE_QUOTE = float(self.MIN_TRADE_QUOTE)
        
        # FIX #3: Entry-sizing floor alignment
        # Config defaults should match SIGNIFICANT_POSITION_FLOOR to avoid runtime normalization churn
        required_entry_floor = float(max(self.MIN_POSITION_USDT, self.SIGNIFICANT_POSITION_FLOOR))
        if self.MIN_ENTRY_USDT < required_entry_floor:
            logger.warning(
                "[Config:EntryFloor] MIN_ENTRY_USDT (%.2f) < floor (max(MIN_POSITION_USDT=%.2f, SIGNIFICANT_POSITION_FLOOR=%.2f)=%.2f). "
                "Bumping MIN_ENTRY_USDT to align config intent with runtime expectations.",
                self.MIN_ENTRY_USDT,
                self.MIN_POSITION_USDT,
                self.SIGNIFICANT_POSITION_FLOOR,
                required_entry_floor,
            )
            self.MIN_ENTRY_USDT = float(required_entry_floor)
        if self.MIN_ENTRY_QUOTE_USDT < self.MIN_ENTRY_USDT:
            logger.warning(
                "[Config:EntryFloor] MIN_ENTRY_QUOTE_USDT (%.2f) < MIN_ENTRY_USDT (%.2f). Bumping MIN_ENTRY_QUOTE_USDT to keep planners/execution aligned.",
                self.MIN_ENTRY_QUOTE_USDT,
                self.MIN_ENTRY_USDT,
            )
            self.MIN_ENTRY_QUOTE_USDT = float(self.MIN_ENTRY_USDT)
        if self.DEFAULT_PLANNED_QUOTE < self.MIN_ENTRY_USDT:
            logger.warning(
                "DEFAULT_PLANNED_QUOTE (%.2f) < MIN_ENTRY_USDT (%.2f). Bumping default planned quote.",
                self.DEFAULT_PLANNED_QUOTE,
                self.MIN_ENTRY_USDT,
            )
            self.DEFAULT_PLANNED_QUOTE = float(self.MIN_ENTRY_USDT)
        if self.EMIT_BUY_QUOTE < self.MIN_ENTRY_USDT:
            logger.warning(
                "EMIT_BUY_QUOTE (%.2f) < MIN_ENTRY_USDT (%.2f). Bumping emitted buy quote.",
                self.EMIT_BUY_QUOTE,
                self.MIN_ENTRY_USDT,
            )
            self.EMIT_BUY_QUOTE = float(self.MIN_ENTRY_USDT)
        self.MAX_HOLD_TIME_SEC = float(os.getenv("MAX_HOLD_TIME_SEC", str(Config.MAX_HOLD_TIME_SEC)))
        self.MAX_HOLD_SEC = float(os.getenv("MAX_HOLD_SEC", str(self.MAX_HOLD_TIME_SEC)))
        self.EXIT_EXCURSION_TICK_MULT = float(os.getenv("EXIT_EXCURSION_TICK_MULT", str(Config.EXIT_EXCURSION_TICK_MULT)))
        self.EXIT_EXCURSION_ATR_MULT = float(os.getenv("EXIT_EXCURSION_ATR_MULT", str(Config.EXIT_EXCURSION_ATR_MULT)))
        self.EXIT_EXCURSION_SPREAD_MULT = float(os.getenv("EXIT_EXCURSION_SPREAD_MULT", str(Config.EXIT_EXCURSION_SPREAD_MULT)))
        self.BUY_COOLDOWN_SEC = float(os.getenv("BUY_COOLDOWN_SEC", str(Config.BUY_COOLDOWN_SEC)))
        self.ENTRY_COOLDOWN_SEC = float(os.getenv("ENTRY_COOLDOWN_SEC", str(Config.ENTRY_COOLDOWN_SEC)))
        self.BUY_REENTRY_MIN_DELTA_PCT = float(os.getenv("BUY_REENTRY_MIN_DELTA_PCT", str(Config.BUY_REENTRY_MIN_DELTA_PCT)))
        self.BUY_REENTRY_DELTA_PCT = float(os.getenv("BUY_REENTRY_DELTA_PCT", str(Config.BUY_REENTRY_DELTA_PCT)))
        self.BUY_REENTRY_DELTA_PCT_TEMP = float(os.getenv("BUY_REENTRY_DELTA_PCT_TEMP", str(Config.BUY_REENTRY_DELTA_PCT_TEMP)))
        self.BUY_REENTRY_DELTA_RESTORE_EQUITY = float(os.getenv("BUY_REENTRY_DELTA_RESTORE_EQUITY", str(Config.BUY_REENTRY_DELTA_RESTORE_EQUITY)))
        self.BUY_REENTRY_DELTA_RESTORE_TRADES = int(os.getenv("BUY_REENTRY_DELTA_RESTORE_TRADES", str(Config.BUY_REENTRY_DELTA_RESTORE_TRADES)))
        # Entry economics guards
        self.MIN_PLANNED_QUOTE_FEE_MULT = float(os.getenv("MIN_PLANNED_QUOTE_FEE_MULT", str(Config.MIN_PLANNED_QUOTE_FEE_MULT)))
        self.MIN_PROFIT_EXIT_FEE_MULT = float(os.getenv("MIN_PROFIT_EXIT_FEE_MULT", str(Config.MIN_PROFIT_EXIT_FEE_MULT)))
        self.MIN_ECONOMIC_TRADE_USDT = float(os.getenv("MIN_ECONOMIC_TRADE_USDT", str(Config.MIN_ECONOMIC_TRADE_USDT)))
        self.MIN_NET_PROFIT_AFTER_FEES = float(os.getenv("MIN_NET_PROFIT_AFTER_FEES", str(Config.MIN_NET_PROFIT_AFTER_FEES)))
        self.STRICT_PROFIT_ONLY_SELLS = os.getenv(
            "STRICT_PROFIT_ONLY_SELLS",
            str(Config.STRICT_PROFIT_ONLY_SELLS)
        ).lower() == "true"
        # Pre-trade effect + micro-backtest gate
        self.PRETRADE_EFFECT_GUARD_ENABLED = os.getenv(
            "PRETRADE_EFFECT_GUARD_ENABLED",
            str(Config.PRETRADE_EFFECT_GUARD_ENABLED)
        ).lower() == "true"
        self.PRETRADE_ALLOW_MISSING_EXPECTED_MOVE = os.getenv(
            "PRETRADE_ALLOW_MISSING_EXPECTED_MOVE",
            str(Config.PRETRADE_ALLOW_MISSING_EXPECTED_MOVE)
        ).lower() == "true"
        self.PRETRADE_ALLOW_MISSING_EXPECTED_MOVE_SELL = os.getenv(
            "PRETRADE_ALLOW_MISSING_EXPECTED_MOVE_SELL",
            str(Config.PRETRADE_ALLOW_MISSING_EXPECTED_MOVE_SELL)
        ).lower() == "true"
        self.PRETRADE_SELL_REQUIRE_EXPECTED_EDGE = os.getenv(
            "PRETRADE_SELL_REQUIRE_EXPECTED_EDGE",
            str(Config.PRETRADE_SELL_REQUIRE_EXPECTED_EDGE)
        ).lower() == "true"
        self.PRETRADE_SELL_GATE_ENFORCED = os.getenv(
            "PRETRADE_SELL_GATE_ENFORCED",
            str(Config.PRETRADE_SELL_GATE_ENFORCED)
        ).lower() == "true"
        self.PRETRADE_ALLOW_BOOTSTRAP_BYPASS = os.getenv(
            "PRETRADE_ALLOW_BOOTSTRAP_BYPASS",
            str(Config.PRETRADE_ALLOW_BOOTSTRAP_BYPASS)
        ).lower() == "true"
        self.PRETRADE_DIRECTIVE_GATE_ENABLED = os.getenv(
            "PRETRADE_DIRECTIVE_GATE_ENABLED",
            str(Config.PRETRADE_DIRECTIVE_GATE_ENABLED)
        ).lower() == "true"
        self.PRETRADE_EFFECT_BUFFER_BPS = float(
            os.getenv("PRETRADE_EFFECT_BUFFER_BPS", str(Config.PRETRADE_EFFECT_BUFFER_BPS))
        )
        self.PRETRADE_MIN_EXPECTED_NET_PCT = float(
            os.getenv("PRETRADE_MIN_EXPECTED_NET_PCT", str(Config.PRETRADE_MIN_EXPECTED_NET_PCT))
        )
        self.PRETRADE_MIN_EXPECTED_NET_USDT = float(
            os.getenv("PRETRADE_MIN_EXPECTED_NET_USDT", str(Config.PRETRADE_MIN_EXPECTED_NET_USDT))
        )
        self.PRETRADE_RECENT_REALIZED_WINDOW = int(
            os.getenv("PRETRADE_RECENT_REALIZED_WINDOW", str(Config.PRETRADE_RECENT_REALIZED_WINDOW))
        )
        self.PRETRADE_MIN_REALIZED_SAMPLES = int(
            os.getenv("PRETRADE_MIN_REALIZED_SAMPLES", str(Config.PRETRADE_MIN_REALIZED_SAMPLES))
        )
        self.PRETRADE_MIN_REALIZED_WIN_RATE = float(
            os.getenv("PRETRADE_MIN_REALIZED_WIN_RATE", str(Config.PRETRADE_MIN_REALIZED_WIN_RATE))
        )
        self.PRETRADE_MICRO_BACKTEST_ENABLED = os.getenv(
            "PRETRADE_MICRO_BACKTEST_ENABLED",
            str(Config.PRETRADE_MICRO_BACKTEST_ENABLED)
        ).lower() == "true"
        self.PRETRADE_MICRO_BACKTEST_TIMEFRAME = str(
            os.getenv("PRETRADE_MICRO_BACKTEST_TIMEFRAME", str(Config.PRETRADE_MICRO_BACKTEST_TIMEFRAME))
        )
        self.PRETRADE_MICRO_BACKTEST_LOOKBACK_BARS = int(
            os.getenv("PRETRADE_MICRO_BACKTEST_LOOKBACK_BARS", str(Config.PRETRADE_MICRO_BACKTEST_LOOKBACK_BARS))
        )
        self.PRETRADE_MICRO_BACKTEST_HORIZON_BARS = int(
            os.getenv("PRETRADE_MICRO_BACKTEST_HORIZON_BARS", str(Config.PRETRADE_MICRO_BACKTEST_HORIZON_BARS))
        )
        self.PRETRADE_MICRO_BACKTEST_MIN_SAMPLES = int(
            os.getenv("PRETRADE_MICRO_BACKTEST_MIN_SAMPLES", str(Config.PRETRADE_MICRO_BACKTEST_MIN_SAMPLES))
        )
        self.PRETRADE_MICRO_BACKTEST_REQUIRE_SAMPLES = os.getenv(
            "PRETRADE_MICRO_BACKTEST_REQUIRE_SAMPLES",
            str(Config.PRETRADE_MICRO_BACKTEST_REQUIRE_SAMPLES)
        ).lower() == "true"
        self.PRETRADE_MICRO_BACKTEST_MIN_WIN_RATE = float(
            os.getenv("PRETRADE_MICRO_BACKTEST_MIN_WIN_RATE", str(Config.PRETRADE_MICRO_BACKTEST_MIN_WIN_RATE))
        )
        self.PRETRADE_MICRO_BACKTEST_MIN_AVG_NET_PCT = float(
            os.getenv("PRETRADE_MICRO_BACKTEST_MIN_AVG_NET_PCT", str(Config.PRETRADE_MICRO_BACKTEST_MIN_AVG_NET_PCT))
        )
        self.PRETRADE_MICRO_BACKTEST_DEADLOCK_RELAX_ENABLED = os.getenv(
            "PRETRADE_MICRO_BACKTEST_DEADLOCK_RELAX_ENABLED",
            str(Config.PRETRADE_MICRO_BACKTEST_DEADLOCK_RELAX_ENABLED),
        ).lower() == "true"
        self.PRETRADE_MICRO_BACKTEST_DEADLOCK_REJECTION_TRIGGER = int(
            os.getenv(
                "PRETRADE_MICRO_BACKTEST_DEADLOCK_REJECTION_TRIGGER",
                str(Config.PRETRADE_MICRO_BACKTEST_DEADLOCK_REJECTION_TRIGGER),
            )
        )
        self.PRETRADE_MICRO_BACKTEST_DEADLOCK_STEP_REJECTIONS = int(
            os.getenv(
                "PRETRADE_MICRO_BACKTEST_DEADLOCK_STEP_REJECTIONS",
                str(Config.PRETRADE_MICRO_BACKTEST_DEADLOCK_STEP_REJECTIONS),
            )
        )
        self.PRETRADE_MICRO_BACKTEST_DEADLOCK_RELAX_WIN_STEP = float(
            os.getenv(
                "PRETRADE_MICRO_BACKTEST_DEADLOCK_RELAX_WIN_STEP",
                str(Config.PRETRADE_MICRO_BACKTEST_DEADLOCK_RELAX_WIN_STEP),
            )
        )
        self.PRETRADE_MICRO_BACKTEST_DEADLOCK_MIN_WIN_RATE_FLOOR = float(
            os.getenv(
                "PRETRADE_MICRO_BACKTEST_DEADLOCK_MIN_WIN_RATE_FLOOR",
                str(Config.PRETRADE_MICRO_BACKTEST_DEADLOCK_MIN_WIN_RATE_FLOOR),
            )
        )
        self.PRETRADE_MICRO_BACKTEST_DEADLOCK_MIN_AVG_NET_FLOOR_PCT = float(
            os.getenv(
                "PRETRADE_MICRO_BACKTEST_DEADLOCK_MIN_AVG_NET_FLOOR_PCT",
                str(Config.PRETRADE_MICRO_BACKTEST_DEADLOCK_MIN_AVG_NET_FLOOR_PCT),
            )
        )
        self.PRETRADE_MICRO_BACKTEST_DEADLOCK_DISABLE_SAMPLE_REQUIRE_AFTER_STEPS = int(
            os.getenv(
                "PRETRADE_MICRO_BACKTEST_DEADLOCK_DISABLE_SAMPLE_REQUIRE_AFTER_STEPS",
                str(Config.PRETRADE_MICRO_BACKTEST_DEADLOCK_DISABLE_SAMPLE_REQUIRE_AFTER_STEPS),
            )
        )
        self.PRETRADE_MICRO_BACKTEST_DEADLOCK_FORCE_BYPASS_AFTER_REJECTIONS = int(
            os.getenv(
                "PRETRADE_MICRO_BACKTEST_DEADLOCK_FORCE_BYPASS_AFTER_REJECTIONS",
                str(Config.PRETRADE_MICRO_BACKTEST_DEADLOCK_FORCE_BYPASS_AFTER_REJECTIONS),
            )
        )
        self.PRETRADE_MICRO_BACKTEST_DEADLOCK_FORCE_BYPASS_MIN_CONF = float(
            os.getenv(
                "PRETRADE_MICRO_BACKTEST_DEADLOCK_FORCE_BYPASS_MIN_CONF",
                str(Config.PRETRADE_MICRO_BACKTEST_DEADLOCK_FORCE_BYPASS_MIN_CONF),
            )
        )
        self.PRETRADE_MICRO_BACKTEST_DEADLOCK_FORCE_BYPASS_MIN_EXPECTED_NET_PCT = float(
            os.getenv(
                "PRETRADE_MICRO_BACKTEST_DEADLOCK_FORCE_BYPASS_MIN_EXPECTED_NET_PCT",
                str(Config.PRETRADE_MICRO_BACKTEST_DEADLOCK_FORCE_BYPASS_MIN_EXPECTED_NET_PCT),
            )
        )
        self.PRETRADE_MICRO_BACKTEST_DEADLOCK_FORCE_BYPASS_MIN_EXPECTED_NET_USDT = float(
            os.getenv(
                "PRETRADE_MICRO_BACKTEST_DEADLOCK_FORCE_BYPASS_MIN_EXPECTED_NET_USDT",
                str(Config.PRETRADE_MICRO_BACKTEST_DEADLOCK_FORCE_BYPASS_MIN_EXPECTED_NET_USDT),
            )
        )
        self.PRETRADE_MICRO_BACKTEST_STARVATION_CYCLES_TRIGGER = int(
            os.getenv(
                "PRETRADE_MICRO_BACKTEST_STARVATION_CYCLES_TRIGGER",
                str(Config.PRETRADE_MICRO_BACKTEST_STARVATION_CYCLES_TRIGGER),
            )
        )
        self.PRETRADE_MICRO_BACKTEST_STARVATION_REJECTION_TRIGGER = int(
            os.getenv(
                "PRETRADE_MICRO_BACKTEST_STARVATION_REJECTION_TRIGGER",
                str(Config.PRETRADE_MICRO_BACKTEST_STARVATION_REJECTION_TRIGGER),
            )
        )
        self.PRETRADE_MICRO_BACKTEST_STARVATION_FORCE_BYPASS_MIN_CONF = float(
            os.getenv(
                "PRETRADE_MICRO_BACKTEST_STARVATION_FORCE_BYPASS_MIN_CONF",
                str(Config.PRETRADE_MICRO_BACKTEST_STARVATION_FORCE_BYPASS_MIN_CONF),
            )
        )
        self.PRETRADE_MICRO_BACKTEST_STARVATION_FORCE_BYPASS_MIN_EXPECTED_NET_PCT = float(
            os.getenv(
                "PRETRADE_MICRO_BACKTEST_STARVATION_FORCE_BYPASS_MIN_EXPECTED_NET_PCT",
                str(Config.PRETRADE_MICRO_BACKTEST_STARVATION_FORCE_BYPASS_MIN_EXPECTED_NET_PCT),
            )
        )
        self.PRETRADE_MICRO_BACKTEST_STARVATION_FORCE_BYPASS_MIN_EXPECTED_NET_USDT = float(
            os.getenv(
                "PRETRADE_MICRO_BACKTEST_STARVATION_FORCE_BYPASS_MIN_EXPECTED_NET_USDT",
                str(Config.PRETRADE_MICRO_BACKTEST_STARVATION_FORCE_BYPASS_MIN_EXPECTED_NET_USDT),
            )
        )
        self.PRETRADE_NET_USDT_DEADLOCK_RELAX_ENABLED = os.getenv(
            "PRETRADE_NET_USDT_DEADLOCK_RELAX_ENABLED",
            str(Config.PRETRADE_NET_USDT_DEADLOCK_RELAX_ENABLED),
        ).lower() == "true"
        self.PRETRADE_NET_USDT_DEADLOCK_REJECTION_TRIGGER = int(
            os.getenv(
                "PRETRADE_NET_USDT_DEADLOCK_REJECTION_TRIGGER",
                str(Config.PRETRADE_NET_USDT_DEADLOCK_REJECTION_TRIGGER),
            )
        )
        self.PRETRADE_NET_USDT_DEADLOCK_STEP_REJECTIONS = int(
            os.getenv(
                "PRETRADE_NET_USDT_DEADLOCK_STEP_REJECTIONS",
                str(Config.PRETRADE_NET_USDT_DEADLOCK_STEP_REJECTIONS),
            )
        )
        self.PRETRADE_NET_USDT_DEADLOCK_RELAX_STEP_FACTOR = float(
            os.getenv(
                "PRETRADE_NET_USDT_DEADLOCK_RELAX_STEP_FACTOR",
                str(Config.PRETRADE_NET_USDT_DEADLOCK_RELAX_STEP_FACTOR),
            )
        )
        self.PRETRADE_NET_USDT_DEADLOCK_MIN_USDT_FLOOR = float(
            os.getenv(
                "PRETRADE_NET_USDT_DEADLOCK_MIN_USDT_FLOOR",
                str(Config.PRETRADE_NET_USDT_DEADLOCK_MIN_USDT_FLOOR),
            )
        )
        self.PRETRADE_STALL_RELAX_ENABLED = os.getenv(
            "PRETRADE_STALL_RELAX_ENABLED",
            str(Config.PRETRADE_STALL_RELAX_ENABLED),
        ).lower() == "true"
        self.PRETRADE_STALL_RELAX_CYCLES_TRIGGER = int(
            os.getenv(
                "PRETRADE_STALL_RELAX_CYCLES_TRIGGER",
                str(Config.PRETRADE_STALL_RELAX_CYCLES_TRIGGER),
            )
        )
        self.PRETRADE_STALL_RELAX_STEP_CYCLES = int(
            os.getenv(
                "PRETRADE_STALL_RELAX_STEP_CYCLES",
                str(Config.PRETRADE_STALL_RELAX_STEP_CYCLES),
            )
        )
        self.PRETRADE_STALL_RELAX_NET_PCT_STEP = float(
            os.getenv(
                "PRETRADE_STALL_RELAX_NET_PCT_STEP",
                str(Config.PRETRADE_STALL_RELAX_NET_PCT_STEP),
            )
        )
        self.PRETRADE_STALL_RELAX_NET_USDT_STEP = float(
            os.getenv(
                "PRETRADE_STALL_RELAX_NET_USDT_STEP",
                str(Config.PRETRADE_STALL_RELAX_NET_USDT_STEP),
            )
        )
        self.PRETRADE_STALL_RELAX_WIN_RATE_STEP = float(
            os.getenv(
                "PRETRADE_STALL_RELAX_WIN_RATE_STEP",
                str(Config.PRETRADE_STALL_RELAX_WIN_RATE_STEP),
            )
        )
        self.PRETRADE_STALL_RELAX_MAX_STEPS = int(
            os.getenv(
                "PRETRADE_STALL_RELAX_MAX_STEPS",
                str(Config.PRETRADE_STALL_RELAX_MAX_STEPS),
            )
        )
        self.PRETRADE_STALL_RELAX_MIN_NET_PCT_FLOOR = float(
            os.getenv(
                "PRETRADE_STALL_RELAX_MIN_NET_PCT_FLOOR",
                str(Config.PRETRADE_STALL_RELAX_MIN_NET_PCT_FLOOR),
            )
        )
        self.PRETRADE_STALL_RELAX_MIN_NET_USDT_FLOOR = float(
            os.getenv(
                "PRETRADE_STALL_RELAX_MIN_NET_USDT_FLOOR",
                str(Config.PRETRADE_STALL_RELAX_MIN_NET_USDT_FLOOR),
            )
        )
        self.PRETRADE_STALL_RELAX_MIN_WIN_RATE_FLOOR = float(
            os.getenv(
                "PRETRADE_STALL_RELAX_MIN_WIN_RATE_FLOOR",
                str(Config.PRETRADE_STALL_RELAX_MIN_WIN_RATE_FLOOR),
            )
        )
        self.PRETRADE_STALL_RELAX_MIN_BT_WIN_RATE_FLOOR = float(
            os.getenv(
                "PRETRADE_STALL_RELAX_MIN_BT_WIN_RATE_FLOOR",
                str(Config.PRETRADE_STALL_RELAX_MIN_BT_WIN_RATE_FLOOR),
            )
        )
        self.PRETRADE_MICRO_BACKTEST_SOFT_GATE_ENABLED = os.getenv(
            "PRETRADE_MICRO_BACKTEST_SOFT_GATE_ENABLED",
            str(Config.PRETRADE_MICRO_BACKTEST_SOFT_GATE_ENABLED),
        ).lower() == "true"
        self.PRETRADE_MICRO_BACKTEST_SOFT_GATE_MIN_SAMPLES = int(
            os.getenv(
                "PRETRADE_MICRO_BACKTEST_SOFT_GATE_MIN_SAMPLES",
                str(Config.PRETRADE_MICRO_BACKTEST_SOFT_GATE_MIN_SAMPLES),
            )
        )
        self.PRETRADE_MICRO_BACKTEST_SOFT_GATE_RELAX_WIN_RATE = float(
            os.getenv(
                "PRETRADE_MICRO_BACKTEST_SOFT_GATE_RELAX_WIN_RATE",
                str(Config.PRETRADE_MICRO_BACKTEST_SOFT_GATE_RELAX_WIN_RATE),
            )
        )
        self.PRETRADE_MICRO_BACKTEST_SOFT_GATE_RELAX_AVG_NET_PCT = float(
            os.getenv(
                "PRETRADE_MICRO_BACKTEST_SOFT_GATE_RELAX_AVG_NET_PCT",
                str(Config.PRETRADE_MICRO_BACKTEST_SOFT_GATE_RELAX_AVG_NET_PCT),
            )
        )
        self.PRETRADE_MICRO_BACKTEST_SOFT_GATE_ALLOW_MISSING_AFTER_CYCLES = int(
            os.getenv(
                "PRETRADE_MICRO_BACKTEST_SOFT_GATE_ALLOW_MISSING_AFTER_CYCLES",
                str(Config.PRETRADE_MICRO_BACKTEST_SOFT_GATE_ALLOW_MISSING_AFTER_CYCLES),
            )
        )
        self.MIN_PORTFOLIO_IMPROVEMENT_USD = float(os.getenv("MIN_PORTFOLIO_IMPROVEMENT_USD", str(Config.MIN_PORTFOLIO_IMPROVEMENT_USD)))
        self.MIN_PORTFOLIO_IMPROVEMENT_PCT = float(os.getenv("MIN_PORTFOLIO_IMPROVEMENT_PCT", str(Config.MIN_PORTFOLIO_IMPROVEMENT_PCT)))
        self.CLOSE_ESCAPE_HATCH_ENABLED = os.getenv(
            "CLOSE_ESCAPE_HATCH_ENABLED",
            str(Config.CLOSE_ESCAPE_HATCH_ENABLED),
        ).lower() == "true"
        self.CLOSE_ESCAPE_HATCH_TRIGGER_COUNT = int(
            os.getenv("CLOSE_ESCAPE_HATCH_TRIGGER_COUNT", str(Config.CLOSE_ESCAPE_HATCH_TRIGGER_COUNT))
        )
        self.CLOSE_ESCAPE_HATCH_WINDOW_SEC = float(
            os.getenv("CLOSE_ESCAPE_HATCH_WINDOW_SEC", str(Config.CLOSE_ESCAPE_HATCH_WINDOW_SEC))
        )
        self.CLOSE_ESCAPE_HATCH_MIN_AGE_SEC = float(
            os.getenv("CLOSE_ESCAPE_HATCH_MIN_AGE_SEC", str(Config.CLOSE_ESCAPE_HATCH_MIN_AGE_SEC))
        )
        self.CLOSE_ESCAPE_HATCH_RETRY_COOLDOWN_SEC = float(
            os.getenv(
                "CLOSE_ESCAPE_HATCH_RETRY_COOLDOWN_SEC",
                str(Config.CLOSE_ESCAPE_HATCH_RETRY_COOLDOWN_SEC),
            )
        )
        self.CLOSE_ESCAPE_HATCH_BUY_LOCK_TRIGGER = int(
            os.getenv("CLOSE_ESCAPE_HATCH_BUY_LOCK_TRIGGER", str(Config.CLOSE_ESCAPE_HATCH_BUY_LOCK_TRIGGER))
        )
        self.CLOSE_ESCAPE_HATCH_ALLOW_REASONS = str(
            os.getenv("CLOSE_ESCAPE_HATCH_ALLOW_REASONS", str(Config.CLOSE_ESCAPE_HATCH_ALLOW_REASONS))
        )
        self.TRADEABILITY_FILL_STALL_RELAX_ENABLED = os.getenv(
            "TRADEABILITY_FILL_STALL_RELAX_ENABLED",
            str(Config.TRADEABILITY_FILL_STALL_RELAX_ENABLED),
        ).lower() == "true"
        self.TRADEABILITY_FILL_STALL_WINDOW_SEC = float(
            os.getenv("TRADEABILITY_FILL_STALL_WINDOW_SEC", str(Config.TRADEABILITY_FILL_STALL_WINDOW_SEC))
        )
        self.TRADEABILITY_FILL_STALL_MAX_STEPS = int(
            os.getenv("TRADEABILITY_FILL_STALL_MAX_STEPS", str(Config.TRADEABILITY_FILL_STALL_MAX_STEPS))
        )
        self.TRADEABILITY_FILL_STALL_MIN_RECENT_TRADES = int(
            os.getenv(
                "TRADEABILITY_FILL_STALL_MIN_RECENT_TRADES",
                str(Config.TRADEABILITY_FILL_STALL_MIN_RECENT_TRADES),
            )
        )
        self.TRADEABILITY_FILL_STALL_RELAX_STEP = float(
            os.getenv("TRADEABILITY_FILL_STALL_RELAX_STEP", str(Config.TRADEABILITY_FILL_STALL_RELAX_STEP))
        )
        self.TRADEABILITY_FILL_STALL_RELAX_MAX = float(
            os.getenv("TRADEABILITY_FILL_STALL_RELAX_MAX", str(Config.TRADEABILITY_FILL_STALL_RELAX_MAX))
        )
        self.TRADEABILITY_FILL_STALL_MEDIUM_RATIO_STEP = float(
            os.getenv(
                "TRADEABILITY_FILL_STALL_MEDIUM_RATIO_STEP",
                str(Config.TRADEABILITY_FILL_STALL_MEDIUM_RATIO_STEP),
            )
        )
        self.TRADEABILITY_FILL_STALL_MEDIUM_RATIO_MAX_DROP = float(
            os.getenv(
                "TRADEABILITY_FILL_STALL_MEDIUM_RATIO_MAX_DROP",
                str(Config.TRADEABILITY_FILL_STALL_MEDIUM_RATIO_MAX_DROP),
            )
        )
        self.BOOTSTRAP_VETO_COOLDOWN_SEC = float(os.getenv("BOOTSTRAP_VETO_COOLDOWN_SEC", str(Config.BOOTSTRAP_VETO_COOLDOWN_SEC)))
        self.CAPITAL_ALLOCATOR_SHARED_WALLET = str(
            os.getenv("CAPITAL_ALLOCATOR_SHARED_WALLET", str(Config.CAPITAL_ALLOCATOR_SHARED_WALLET))
        ).strip().lower() in {"1", "true", "yes", "on"}

        # Medium-acceleration guardrail:
        # If dynamic risk budget is elevated, keep floors aligned so planned quote can express 30-45 USDT sizing.
        if float(self.DYNAMIC_RISK_BUDGET_PCT or 0.0) >= 0.20:
            self.MIN_ENTRY_QUOTE_USDT = min(float(self.MIN_ENTRY_QUOTE_USDT), 30.0)
            self.DEFAULT_PLANNED_QUOTE = min(float(self.DEFAULT_PLANNED_QUOTE), 30.0)
            self.MIN_TRADE_QUOTE = min(float(self.MIN_TRADE_QUOTE), 20.0)
            self.MIN_ECONOMIC_TRADE_USDT = min(float(self.MIN_ECONOMIC_TRADE_USDT), 30.0)
            self.MIN_ORDER_USDT = min(float(self.MIN_ORDER_USDT), 20.0)

        self.MICRO_TRADE_KILL_SWITCH_ENABLED = os.getenv(
            "MICRO_TRADE_KILL_SWITCH_ENABLED",
            str(Config.MICRO_TRADE_KILL_SWITCH_ENABLED)
        ).lower() == "true"
        self.MICRO_TRADE_KILL_EQUITY_MAX = float(os.getenv("MICRO_TRADE_KILL_EQUITY_MAX", str(Config.MICRO_TRADE_KILL_EQUITY_MAX)))
        self.MICRO_TRADE_KILL_ATR_FEE_MULT = float(os.getenv("MICRO_TRADE_KILL_ATR_FEE_MULT", str(Config.MICRO_TRADE_KILL_ATR_FEE_MULT)))
        self.MICRO_TRADE_KILL_FALLBACK_ATR_PCT = float(os.getenv("MICRO_TRADE_KILL_FALLBACK_ATR_PCT", str(Config.MICRO_TRADE_KILL_FALLBACK_ATR_PCT)))
        self.BOOTSTRAP_ESCAPE_HATCH_ENABLED = os.getenv(
            "BOOTSTRAP_ESCAPE_HATCH_ENABLED",
            str(Config.BOOTSTRAP_ESCAPE_HATCH_ENABLED)
        ).lower() == "true"
        self.MAX_OPEN_POSITIONS_PER_SYMBOL = int(
            os.getenv("MAX_OPEN_POSITIONS_PER_SYMBOL", str(Config.MAX_OPEN_POSITIONS_PER_SYMBOL))
        )

        # ProfitTargetEngine knobs
        self.PROFIT_TARGET_DAILY_PCT = float(os.getenv("PROFIT_TARGET_DAILY_PCT", str(Config.PROFIT_TARGET_DAILY_PCT)))
        self.PROFIT_TARGET_MAX_RISK_PER_CYCLE = float(os.getenv("PROFIT_TARGET_MAX_RISK_PER_CYCLE", str(Config.PROFIT_TARGET_MAX_RISK_PER_CYCLE)))
        self.PROFIT_TARGET_COMPOUND_THROTTLE = float(os.getenv("PROFIT_TARGET_COMPOUND_THROTTLE", str(Config.PROFIT_TARGET_COMPOUND_THROTTLE)))
        self.PROFIT_TARGET_BASE_USD_PER_HOUR = float(os.getenv("PROFIT_TARGET_BASE_USD_PER_HOUR", str(Config.PROFIT_TARGET_BASE_USD_PER_HOUR)))
        self.PROFIT_TARGET_GRACE_MINUTES = float(os.getenv("PROFIT_TARGET_GRACE_MINUTES", str(Config.PROFIT_TARGET_GRACE_MINUTES)))

        # Stagnation micro-loss exit (Phase-1.5)
        self.STAGNATION_EXIT_ENABLED = os.getenv(
            "STAGNATION_EXIT_ENABLED",
            str(Config.STAGNATION_EXIT_ENABLED)
        ).lower() == "true"
        self.STAGNATION_EXIT_MAX_LOSS_PCT = float(
            os.getenv("STAGNATION_EXIT_MAX_LOSS_PCT", str(Config.STAGNATION_EXIT_MAX_LOSS_PCT))
        )

        # ... other sections ...

        # ---------- Wallet / Positions & Dust ----------
        self.AUTO_POSITION_FROM_BALANCES = os.getenv("AUTO_POSITION_FROM_BALANCES", "True").lower() == "true"

        # IMPORTANT: fallback remains small-account friendly (5.0). Override via .env as needed.
        self.DUST_MIN_QUOTE_USDT = float(os.getenv("DUST_MIN_QUOTE_USDT", str(Config.DUST_MIN_QUOTE_USDT)))
        self.DUST_POSITION_QTY = float(os.getenv("DUST_POSITION_QTY", str(Config.DUST_POSITION_QTY)))
        self.PERMANENT_DUST_USDT_THRESHOLD = float(os.getenv("PERMANENT_DUST_USDT_THRESHOLD", str(Config.PERMANENT_DUST_USDT_THRESHOLD)))
        # Dust priority system: Reuse > Aggregate > Cleanup
        self.DUST_AGGREGATE_THRESHOLD_HOURS = float(os.getenv("DUST_AGGREGATE_THRESHOLD_HOURS", str(Config.DUST_AGGREGATE_THRESHOLD_HOURS)))
        self.DUST_STALL_THRESHOLD_HOURS = float(os.getenv("DUST_STALL_THRESHOLD_HOURS", str(Config.DUST_STALL_THRESHOLD_HOURS)))
        self.DUST_CRITICAL_THRESHOLD_HOURS = float(os.getenv("DUST_CRITICAL_THRESHOLD_HOURS", str(Config.DUST_CRITICAL_THRESHOLD_HOURS)))

        # Ensure dust controls exist (some logs assume these attributes)
        # ✅ Standardized to lowercase for consistent runtime naming
        self.dust_liquidation_enabled = os.getenv(
            "DUST_LIQUIDATION_ENABLED",
            str(Config.DUST_LIQUIDATION_ENABLED),
        ).lower() == "true"
        # Allow dust positions to bypass re-entry lock (merge dust back into tradable size)
        self.dust_reentry_override = os.getenv(
            "DUST_REENTRY_OVERRIDE",
            str(Config.DUST_REENTRY_OVERRIDE),
        ).lower() == "true"
        # 🔧 NEW: Guard to prevent opening new trades below significant floor unless explicitly allowed
        self.allow_entry_below_significant_floor = os.getenv(
            "ALLOW_ENTRY_BELOW_SIGNIFICANT_FLOOR",
            str(Config.ALLOW_ENTRY_BELOW_SIGNIFICANT_FLOOR),
        ).lower() == "true"
        # Back-compat aliases used by older callsites/scripts.
        self.DUST_LIQUIDATION_ENABLED = bool(self.dust_liquidation_enabled)
        self.DUST_REENTRY_OVERRIDE = bool(self.dust_reentry_override)
        self.ALLOW_ENTRY_BELOW_SIGNIFICANT_FLOOR = bool(self.allow_entry_below_significant_floor)
        logger.info(
            "🧹 Dust authority → reentry_override=%s | liquidation_enabled=%s | entry_floor_guard=%s",
            self.dust_reentry_override,
            self.dust_liquidation_enabled,
            not self.allow_entry_below_significant_floor,  # Show guard status (True = guard enabled)
        )
        # Provide a minimal DUST_REGISTER namespace if not defined elsewhere
        if not hasattr(self, "DUST_REGISTER"):
            from types import SimpleNamespace as _SN
            self.DUST_REGISTER = _SN(
                ENABLE=os.getenv("DUST_REGISTER_ENABLE", "false").lower() == "true",
                TTL_DAYS=int(os.getenv("DUST_REGISTER_TTL_DAYS", "7")),
                CHECK_INTERVAL_S=int(os.getenv("DUST_REGISTER_CHECK_INTERVAL_S", "900")),
                TRY_EXCHANGE_CONVERT=os.getenv("DUST_TRY_EXCHANGE_CONVERT", "false").lower() == "true",
            )

        # ---------- Agent-Level Thresholds (Phase A) ----------
        self.ML_MIN_CONF_EMIT = float(os.getenv("ML_MIN_CONF_EMIT", "0.57"))
        # Optional SELL-specific confidence floor (lower to enable exits without loosening BUYs)
        self.SELL_MIN_CONF = float(os.getenv("SELL_MIN_CONF", "0.50"))
        self.ML_MIN_CONF_EMIT_SELL = float(os.getenv("ML_MIN_CONF_EMIT_SELL", str(self.SELL_MIN_CONF)))
        self.ML_GOOD_CONF = float(os.getenv("ML_GOOD_CONF", "0.65"))
        self.ML_STRONG_CONF = float(os.getenv("ML_STRONG_CONF", "0.75"))
        self.ML_CONF_BACKTEST_ON_STARTUP = os.getenv("ML_CONF_BACKTEST_ON_STARTUP", "true").lower() == "true"
        self.AUTO_TRAIN = str(os.getenv("AUTO_TRAIN", str(Config.AUTO_TRAIN))).strip().lower() in {"1", "true", "yes", "on"}
        self.SWING_AUTO_TRAIN = str(
            os.getenv("SWING_AUTO_TRAIN", str(self.AUTO_TRAIN))
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.TREND_AUTO_TRAIN = str(
            os.getenv("TREND_AUTO_TRAIN", str(self.AUTO_TRAIN))
        ).strip().lower() in {"1", "true", "yes", "on"}
        logger.info(
            "Auto-train toggles: AUTO_TRAIN=%s SWING_AUTO_TRAIN=%s TREND_AUTO_TRAIN=%s",
            self.AUTO_TRAIN,
            self.SWING_AUTO_TRAIN,
            self.TREND_AUTO_TRAIN,
        )

        self.TREND_MIN_CONF = float(os.getenv("TREND_MIN_CONF", "0.60"))
        self.TREND_MIN_CONF_SELL = float(os.getenv("TREND_MIN_CONF_SELL", str(self.SELL_MIN_CONF)))
        self.TREND_STRONG_CONF = float(os.getenv("TREND_STRONG_CONF", "0.70"))

        self.DIP_THRESHOLD_PERCENT = float(os.getenv("DIP_THRESHOLD_PERCENT", "2.0"))
        self.MIN_SIGNAL_CONF = float(os.getenv("MIN_SIGNAL_CONF", "0.60"))
        self.VOLUME_SPIKE_MULTIPLIER = float(os.getenv("VOLUME_SPIKE_MULTIPLIER", "1.5"))

        # ---------- Deadlock sensitivity ----------
        self.DEADLOCK_REJECTION_THRESHOLD = int(os.getenv("DEADLOCK_REJECTION_THRESHOLD", "10"))
        self.DEADLOCK_REJECTION_IGNORE_REASONS = os.getenv(
            "DEADLOCK_REJECTION_IGNORE_REASONS",
            (
                "COLD_BOOTSTRAP_BLOCK,PORTFOLIO_FULL,EXPECTED_MOVE_LT_ROUND_TRIP_COST,POSITION_ALREADY_OPEN,"
                "CONF_BELOW_REQUIRED,NET_USDT_BELOW_THRESHOLD,PRETRADE_EFFECT_GATE:NET_USDT_BELOW_THRESHOLD,"
                "MICRO_BACKTEST_WIN_RATE_BELOW_THRESHOLD,"
                "PRETRADE_EFFECT_GATE:MICRO_BACKTEST_WIN_RATE_BELOW_THRESHOLD,"
                "MICRO_BACKTEST_INSUFFICIENT_SAMPLES"
            ),
        )

        # ---------- Global Economic Thresholds ----------
        self.CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.75"))
        self.ARBITRAGE_THRESHOLD = float(os.getenv("ARBITRAGE_THRESHOLD", "0.002"))
        self.DUST_EXIT_THRESHOLD = float(os.getenv("DUST_EXIT_THRESHOLD", "0.60"))
        self.SFA_SIGNAL_CONFIDENCE_THRESHOLD = float(os.getenv("SFA_SIGNAL_CONFIDENCE_THRESHOLD", "0.60"))
        self.TIER_A_CONFIDENCE_THRESHOLD = float(os.getenv("TIER_A_CONFIDENCE_THRESHOLD", "0.75"))

        # ---------- MetaController Execution (Phase A) ----------
        self.META_MIN_AGENTS = int(os.getenv("META_MIN_AGENTS", "1"))
        self.META_MIN_CONF = float(os.getenv("META_MIN_CONF", "0.75"))
        self.MAX_TRADES_PER_HOUR = int(os.getenv("MAX_TRADES_PER_HOUR", "8"))
        self.MAX_TRADES_PER_DAY = int(os.getenv("MAX_TRADES_PER_DAY", str(Config.MAX_TRADES_PER_DAY)))
        self.MAX_TRADES_PER_SYMBOL_PER_HOUR = int(os.getenv("MAX_TRADES_PER_SYMBOL_PER_HOUR", "2"))
        self.FOCUS_MODE_EXIT_ENABLED = os.getenv("FOCUS_MODE_EXIT_ENABLED", "false").lower() == "true"
        self.META_DIRECTIONAL_CONSISTENCY_PCT = float(os.getenv("META_DIRECTIONAL_CONSISTENCY_PCT", "0.65"))
        # Diversification knobs (penalize recently traded symbols when ranking BUYs)
        self.DIVERSIFY_RECENT_TRADE_WINDOW_SEC = float(os.getenv("DIVERSIFY_RECENT_TRADE_WINDOW_SEC", "1800"))
        self.DIVERSIFY_RECENT_TRADE_PENALTY = float(os.getenv("DIVERSIFY_RECENT_TRADE_PENALTY", "0.7"))
        # Small, bounded confidence uplift for multi-agent agreement (applies only when no opposing signals)
        self.TIER_A_AGREE_UPLIFT = float(os.getenv("TIER_A_AGREE_UPLIFT", "0.02"))
        # Tier-A readiness log margin (how close to tier A before we emit readiness log)
        self.TIER_A_READINESS_MARGIN = float(os.getenv("TIER_A_READINESS_MARGIN", "0.03"))

        # ---------- Risk & TP/SL ----------
        self.MAX_RISK_PER_TRADE = float(os.getenv("MAX_RISK_PER_TRADE", "0.02"))
        self.BASE_MIN_NOTIONAL = float(os.getenv("BASE_MIN_NOTIONAL", "5.0"))
        self.SCOUT_MIN_NOTIONAL = float(os.getenv("SCOUT_MIN_NOTIONAL", "6.0"))
        self.MIN_NOTIONAL_USDT = float(os.getenv("MIN_NOTIONAL_USDT", "10.0"))
        self.MAX_DAILY_LOSS_PCT = float(
            os.getenv("MAX_DAILY_LOSS_PCT", os.getenv("MAX_DAILY_LOSS", "0.10"))
        )
        self.MAX_POSITION_EXPOSURE_PCT = float(
            os.getenv(
                "MAX_POSITION_EXPOSURE_PCT",
                os.getenv(
                    "MAX_POSITION_EXPOSURE_PERCENTAGE",
                    os.getenv("MAX_SYMBOL_EXPOSURE", "0.20"),
                ),
            )
        )
        self.MAX_POSITION_EXPOSURE_PERCENTAGE = float(self.MAX_POSITION_EXPOSURE_PCT)
        self.MAX_TOTAL_EXPOSURE_PCT = float(
            os.getenv(
                "MAX_TOTAL_EXPOSURE_PCT",
                os.getenv(
                    "MAX_TOTAL_EXPOSURE_PERCENTAGE",
                    os.getenv("MAX_PORTFOLIO_EXPOSURE", "0.60"),
                ),
            )
        )
        self.MAX_TOTAL_EXPOSURE_PERCENTAGE = float(self.MAX_TOTAL_EXPOSURE_PCT)
        _env_exec_reserve = float(os.getenv("EXECUTION_MIN_FREE_RESERVE_USDT", "0.0") or 0.0)
        _env_liq_buffer = float(os.getenv("MIN_LIQUIDITY_BUFFER", str(_env_exec_reserve)) or _env_exec_reserve)
        self.MIN_LIQUIDITY_BUFFER = float(max(_env_exec_reserve, _env_liq_buffer))
        self.EXECUTION_MIN_FREE_RESERVE_USDT = float(max(_env_exec_reserve, self.MIN_LIQUIDITY_BUFFER))

        self.TP_ATR_MULT = float(os.getenv("TP_ATR_MULT", str(Config.TP_ATR_MULT)))
        self.SL_ATR_MULT = float(os.getenv("SL_ATR_MULT", str(Config.SL_ATR_MULT)))
        self.TARGET_RR_RATIO = float(os.getenv("TARGET_RR_RATIO", str(Config.TARGET_RR_RATIO)))
        self.TRAILING_ATR_MULT = float(os.getenv("TRAILING_ATR_MULT", str(Config.TRAILING_ATR_MULT)))
        self.TPSL_RV_LOOKBACK = int(os.getenv("TPSL_RV_LOOKBACK", str(Config.TPSL_RV_LOOKBACK)))
        self.TPSL_VOL_LOW_ATR_PCT = float(os.getenv("TPSL_VOL_LOW_ATR_PCT", str(Config.TPSL_VOL_LOW_ATR_PCT)))
        self.TPSL_VOL_HIGH_ATR_PCT = float(os.getenv("TPSL_VOL_HIGH_ATR_PCT", str(Config.TPSL_VOL_HIGH_ATR_PCT)))
        self.TPSL_VOL_TARGET_ATR_PCT = float(os.getenv("TPSL_VOL_TARGET_ATR_PCT", str(Config.TPSL_VOL_TARGET_ATR_PCT)))
        self.TPSL_DYNAMIC_RR_MIN = float(os.getenv("TPSL_DYNAMIC_RR_MIN", str(Config.TPSL_DYNAMIC_RR_MIN)))
        self.TPSL_DYNAMIC_RR_MAX = float(os.getenv("TPSL_DYNAMIC_RR_MAX", str(Config.TPSL_DYNAMIC_RR_MAX)))
        self.TRAILING_ACTIVATE_R_MULT = float(os.getenv("TRAILING_ACTIVATE_R_MULT", str(Config.TRAILING_ACTIVATE_R_MULT)))
        self.TPSL_PROFILE = str(os.getenv("TPSL_PROFILE", str(Config.TPSL_PROFILE))).strip().lower()
        self.TPSL_SPREAD_ADAPTIVE_ENABLED = os.getenv(
            "TPSL_SPREAD_ADAPTIVE_ENABLED",
            str(Config.TPSL_SPREAD_ADAPTIVE_ENABLED),
        ).lower() == "true"
        self.TPSL_SPREAD_TIGHT_BPS = float(
            os.getenv("TPSL_SPREAD_TIGHT_BPS", str(Config.TPSL_SPREAD_TIGHT_BPS))
        )
        self.TPSL_SPREAD_HIGH_BPS = float(
            os.getenv("TPSL_SPREAD_HIGH_BPS", str(Config.TPSL_SPREAD_HIGH_BPS))
        )
        self.TPSL_SPREAD_EXTREME_BPS = float(
            os.getenv("TPSL_SPREAD_EXTREME_BPS", str(Config.TPSL_SPREAD_EXTREME_BPS))
        )
        self.TPSL_SPREAD_RR_BONUS_MAX = float(
            os.getenv("TPSL_SPREAD_RR_BONUS_MAX", str(Config.TPSL_SPREAD_RR_BONUS_MAX))
        )
        self.TPSL_SPREAD_RR_DISCOUNT_MAX = float(
            os.getenv("TPSL_SPREAD_RR_DISCOUNT_MAX", str(Config.TPSL_SPREAD_RR_DISCOUNT_MAX))
        )
        self.TPSL_SPREAD_TP_FLOOR_MULT = float(
            os.getenv("TPSL_SPREAD_TP_FLOOR_MULT", str(Config.TPSL_SPREAD_TP_FLOOR_MULT))
        )

        # Profile-aware defaults:
        # Apply preset values only when explicit env keys are NOT provided.
        preset = Config.TPSL_PROFILE_PRESETS.get(self.TPSL_PROFILE)
        if preset:
            for key, val in preset.items():
                if key in os.environ:
                    continue
                setattr(self, key, val)
        elif self.TPSL_PROFILE:
            logger.warning("Unknown TPSL_PROFILE=%r. Using explicit/default TPSL values.", self.TPSL_PROFILE)
        # Minimum acceptable RR for guardrails (can be stricter than target for specific paths)
        self.TP_SL_MIN_RR = float(os.getenv("TP_SL_MIN_RR", "1.4"))
        # First-cycle profitability uplift for cold bootstrap / first trade
        self.FIRST_CYCLE_TP_BOOST_MULT = float(os.getenv("FIRST_CYCLE_TP_BOOST_MULT", "1.15"))
        self.FIRST_CYCLE_SL_TIGHTEN_MULT = float(os.getenv("FIRST_CYCLE_SL_TIGHTEN_MULT", "0.90"))
        # Optional percent clamps for TP/SL distances (fallbacks for ATR variance)
        self.TP_PCT_MIN = float(os.getenv("TP_PCT_MIN", str(Config.TP_PCT_MIN)))
        self.TP_PCT_MAX = float(os.getenv("TP_PCT_MAX", str(Config.TP_PCT_MAX)))
        self.SL_PCT_MIN = float(os.getenv("SL_PCT_MIN", "0.003"))  # 0.30%
        self.SL_PCT_MAX = float(os.getenv("SL_PCT_MAX", "0.010"))  # 1.00%
        self.TP_TARGET_PCT = float(os.getenv("TP_TARGET_PCT", str(Config.TP_TARGET_PCT)))
        self.SL_CAP_PCT = float(os.getenv("SL_CAP_PCT", str(Config.SL_CAP_PCT)))
        self.TP_MIN_BUFFER_BPS = float(os.getenv("TP_MIN_BUFFER_BPS", str(Config.TP_MIN_BUFFER_BPS)))
        self.TP_MICRO_NOTIONAL_USDT = float(os.getenv("TP_MICRO_NOTIONAL_USDT", str(Config.TP_MICRO_NOTIONAL_USDT)))
        self.TP_MICRO_EXTRA_BPS = float(os.getenv("TP_MICRO_EXTRA_BPS", str(Config.TP_MICRO_EXTRA_BPS)))
        logger.info(
            "ResolvedProfitabilityCanon exit_fee_bps=%.2f exit_slip_bps=%.2f tp_buf_bps=%.2f min_net=%.4f tp_min=%.4f tp_max=%.4f tp_atr=%.2f sl_atr=%.2f entry_fee_mult=%.2f exit_fee_mult=%.2f",
            float(getattr(self, "EXIT_FEE_BPS", 0.0) or 0.0),
            float(getattr(self, "EXIT_SLIPPAGE_BPS", 0.0) or 0.0),
            float(getattr(self, "TP_MIN_BUFFER_BPS", 0.0) or 0.0),
            float(getattr(self, "MIN_NET_PROFIT_AFTER_FEES", 0.0) or 0.0),
            float(getattr(self, "TP_PCT_MIN", 0.0) or 0.0),
            float(getattr(self, "TP_PCT_MAX", 0.0) or 0.0),
            float(getattr(self, "TP_ATR_MULT", 0.0) or 0.0),
            float(getattr(self, "SL_ATR_MULT", 0.0) or 0.0),
            float(getattr(self, "MIN_PLANNED_QUOTE_FEE_MULT", 0.0) or 0.0),
            float(getattr(self, "MIN_PROFIT_EXIT_FEE_MULT", 0.0) or 0.0),
        )
        logger.info(
            "TPSLProfile profile=%s rr=%.2f tp_atr=%.2f sl_atr=%.2f trail_atr=%.2f trail_activate_r=%.2f vol_low=%.4f vol_high=%.4f",
            str(getattr(self, "TPSL_PROFILE", "balanced")),
            float(getattr(self, "TARGET_RR_RATIO", 0.0) or 0.0),
            float(getattr(self, "TP_ATR_MULT", 0.0) or 0.0),
            float(getattr(self, "SL_ATR_MULT", 0.0) or 0.0),
            float(getattr(self, "TRAILING_ATR_MULT", 0.0) or 0.0),
            float(getattr(self, "TRAILING_ACTIVATE_R_MULT", 0.0) or 0.0),
            float(getattr(self, "TPSL_VOL_LOW_ATR_PCT", 0.0) or 0.0),
            float(getattr(self, "TPSL_VOL_HIGH_ATR_PCT", 0.0) or 0.0),
        )

        # ---------- SELL economic gates ----------
        # If False, non-liquidation SELLs must clear fee-aware net PnL gate.
        self.ALLOW_SELL_BELOW_FEE = os.getenv("ALLOW_SELL_BELOW_FEE", "false").lower() == "true"
        # Minimum net PnL (USDT) required for non-liquidation SELLs (fees included).
        self.SELL_MIN_NET_PNL_USDT = float(os.getenv("SELL_MIN_NET_PNL_USDT", "0.05"))
        self.SELL_DYNAMIC_EDGE_GATE_ENABLED = os.getenv(
            "SELL_DYNAMIC_EDGE_GATE_ENABLED", str(Config.SELL_DYNAMIC_EDGE_GATE_ENABLED)
        ).lower() == "true"
        self.SELL_DYNAMIC_SLIPPAGE_MIN_PCT = float(
            os.getenv("SELL_DYNAMIC_SLIPPAGE_MIN_PCT", str(Config.SELL_DYNAMIC_SLIPPAGE_MIN_PCT))
        )
        self.SELL_DYNAMIC_SLIPPAGE_ATR_MULT = float(
            os.getenv("SELL_DYNAMIC_SLIPPAGE_ATR_MULT", str(Config.SELL_DYNAMIC_SLIPPAGE_ATR_MULT))
        )
        self.SELL_DYNAMIC_VOL_BUFFER_ATR_MULT = float(
            os.getenv("SELL_DYNAMIC_VOL_BUFFER_ATR_MULT", str(Config.SELL_DYNAMIC_VOL_BUFFER_ATR_MULT))
        )
        self.SELL_DYNAMIC_STRATEGIC_BUFFER_PCT = float(
            os.getenv("SELL_DYNAMIC_STRATEGIC_BUFFER_PCT", str(Config.SELL_DYNAMIC_STRATEGIC_BUFFER_PCT))
        )
        self.SELL_DYNAMIC_MIN_USDT_FLOOR = float(
            os.getenv("SELL_DYNAMIC_MIN_USDT_FLOOR", str(Config.SELL_DYNAMIC_MIN_USDT_FLOOR))
        )
        self.SELL_DYNAMIC_ATR_TIMEFRAME = str(
            os.getenv("SELL_DYNAMIC_ATR_TIMEFRAME", str(Config.SELL_DYNAMIC_ATR_TIMEFRAME))
        )
        self.SELL_DYNAMIC_ATR_PERIOD = int(
            os.getenv("SELL_DYNAMIC_ATR_PERIOD", str(Config.SELL_DYNAMIC_ATR_PERIOD))
        )
        self.SELL_DYNAMIC_FALLBACK_ATR_PCT = float(
            os.getenv("SELL_DYNAMIC_FALLBACK_ATR_PCT", str(Config.SELL_DYNAMIC_FALLBACK_ATR_PCT))
        )
        self.SELL_DYNAMIC_REGIME_HIGH_MULT = float(
            os.getenv("SELL_DYNAMIC_REGIME_HIGH_MULT", str(Config.SELL_DYNAMIC_REGIME_HIGH_MULT))
        )
        self.SELL_DYNAMIC_REGIME_LOW_MULT = float(
            os.getenv("SELL_DYNAMIC_REGIME_LOW_MULT", str(Config.SELL_DYNAMIC_REGIME_LOW_MULT))
        )
        self.SELL_DYNAMIC_LEGACY_NET_PCT_GUARD = os.getenv(
            "SELL_DYNAMIC_LEGACY_NET_PCT_GUARD", str(Config.SELL_DYNAMIC_LEGACY_NET_PCT_GUARD)
        ).lower() == "true"

        # Bootstrap SELL override: allow break-even/small-loss exits to free capital
        self.BOOTSTRAP_ALLOW_SELL_BELOW_FEE = os.getenv("BOOTSTRAP_ALLOW_SELL_BELOW_FEE", "true").lower() == "true"
        self.BOOTSTRAP_MAX_NEGATIVE_PNL = float(os.getenv("BOOTSTRAP_MAX_NEGATIVE_PNL", "-0.03"))
        self.BOOTSTRAP_DURATION_MINUTES = int(os.getenv("BOOTSTRAP_DURATION_MINUTES", "1440"))

        # ---------- Size bump tuning ----------
        # Multiplier applied when size bump activates (conditional planned_quote bump)
        self.PLANNED_QUOTE_BUMP_MULT = float(os.getenv("PLANNED_QUOTE_BUMP_MULT", "1.5"))
        # Threshold multiplier for realized PnL vs round-trip fees to trigger bump
        self.PLANNED_QUOTE_BUMP_THRESHOLD_MULT = float(os.getenv("PLANNED_QUOTE_BUMP_THRESHOLD_MULT", "1.2"))

        # ---------- TP/SL enforcement & recovery guard ----------
        self.TPSL_AUTO_ARM_ON_STARTUP = os.getenv("TPSL_AUTO_ARM_ON_STARTUP", "true").lower() == "true"
        self.CAPITAL_RECOVERY_TPSL_GUARD = os.getenv("CAPITAL_RECOVERY_TPSL_GUARD", "true").lower() == "true"

        # ---------- Anti-churn re-entry guards ----------
        self.REENTRY_LOCK_SEC = float(os.getenv("REENTRY_LOCK_SEC", "300"))
        self.REENTRY_REQUIRE_TPSL_EXIT = os.getenv("REENTRY_REQUIRE_TPSL_EXIT", "true").lower() == "true"
        self.REENTRY_REQUIRE_SIGNAL_CHANGE = os.getenv("REENTRY_REQUIRE_SIGNAL_CHANGE", "false").lower() == "true"

        # ---------- Min-hold guard for SELLs ----------
        self.MIN_HOLD_SEC = float(os.getenv("MIN_HOLD_SEC", "300"))
        # Bootstrap min-hold override (allow fast exits during bootstrap)
        self.MIN_HOLD_SEC_BOOTSTRAP = float(os.getenv("MIN_HOLD_SEC_BOOTSTRAP", "30"))
        # Optional price expansion gate (exit only after sufficient move)
        self.MIN_HOLD_PRICE_MOVE_PCT = float(os.getenv("MIN_HOLD_PRICE_MOVE_PCT", "0.4"))

        # ---------- Micro-profit cycle (time exit) ----------
        # Tighten early time-exit for Tier-B capital rotation.
        # 0.25h = 15 minutes; 0.0005 = 0.05%
        self.MICRO_PROFIT_CYCLE_ENABLED = os.getenv("MICRO_PROFIT_CYCLE_ENABLED", "false").lower() == "true"
        self.MICRO_PROFIT_CYCLE_MIN_AGE_HOURS = float(os.getenv("MICRO_PROFIT_CYCLE_MIN_AGE_HOURS", "0.25"))
        self.MICRO_PROFIT_CYCLE_MAX_AGE_HOURS = float(os.getenv("MICRO_PROFIT_CYCLE_MAX_AGE_HOURS", "4.0"))
        self.MICRO_PROFIT_CYCLE_MIN_PNL_PCT = float(os.getenv("MICRO_PROFIT_CYCLE_MIN_PNL_PCT", "0.0005"))  # 0.05%

        # ---------- Phase 9: Wealth engine (scaling/compounding) ----------
        # Lower friction for safe compounding while preventing startup flip.
        # SCALE_IN_MIN_AGE_MIN: minimum position age in minutes before scaling in (default 3 min)
        # SCALE_IN_MIN_PNL_PCT: minimum pnl (in percent units) required to consider scaling in (default 0.2 == 0.2%)
        self.SCALE_IN_MIN_AGE_MIN = float(os.getenv("SCALE_IN_MIN_AGE_MIN", "3.0"))
        self.SCALE_IN_MIN_PNL_PCT = float(os.getenv("SCALE_IN_MIN_PNL_PCT", "0.2"))

    # ---------- Endpoint resolver (NEW) ----------
    def _resolve_base_url(self, region: str, account_type: str, testnet: bool) -> str:
        region = region.lower()
        account_type = account_type.lower()

        if testnet:
            # Official testnets:
            # Spot testnet: https://testnet.binance.vision
            # UM Futures testnet: https://testnet.binancefuture.com
            if account_type == "spot":
                return "https://testnet.binance.vision"
            return "https://testnet.binancefuture.com"

        # Live endpoints
        if account_type == "spot":
            if region == "us":
                return "https://api.binance.us"
            return "https://api.binance.com"
        else:
            # UM Futures: Binance.US does not support UM futures
            # Always use global futures endpoint
            return "https://fapi.binance.com"

    def validate_exchange_settings(self) -> None:
        if self.BINANCE_ACCOUNT_TYPE not in ("spot", "futures"):
            raise ValueError(f"BINANCE_ACCOUNT_TYPE must be 'spot' or 'futures', got {self.BINANCE_ACCOUNT_TYPE!r}")
        if self.BINANCE_REGION not in ("global", "us"):
            raise ValueError(f"BINANCE_REGION must be 'global' or 'us', got {self.BINANCE_REGION!r}")

        if self.BINANCE_ACCOUNT_TYPE == "futures" and self.BINANCE_REGION == "us" and not self.TESTNET_MODE:
            logger.warning(
                "⚠️ You selected BINANCE_REGION='us' with BINANCE_ACCOUNT_TYPE='futures'. "
                "Binance.US does not provide UM Futures. Using global UM futures endpoint "
                "(https://fapi.binance.com) — your API key must be from binance.com and your server IP must be whitelisted there."
            )

        if not self.TESTNET_MODE and (not self.BINANCE_API_KEY or not self.BINANCE_API_SECRET):
            logger.warning("⚠️ No live API keys found. Private calls will fail (e.g., balances/orders).")

        if self.TESTNET_MODE and (not self.BINANCE_TESTNET_API_KEY or not self.BINANCE_TESTNET_API_SECRET):
            logger.warning("⚠️ TESTNET_MODE is True but testnet keys are missing.")

    def _log_effective_exchange_cfg(self) -> None:
        live_key_masked = self._mask(self.BINANCE_API_KEY)
        test_key_masked = self._mask(self.BINANCE_TESTNET_API_KEY)

        logger.info(
            "🔌 Exchange scope → region=%s | account_type=%s | testnet=%s | base_url=%s (spot=api.binance.com / futures=fapi.binance.com)",
            self.BINANCE_REGION, self.BINANCE_ACCOUNT_TYPE, self.TESTNET_MODE, self.BINANCE_BASE_URL
        )
        logger.info(
            "🔑 Keys loaded → live_key=%s | testnet_key=%s",
            live_key_masked or "(none)", test_key_masked or "(none)"
        )
        logger.info(
            "🧰 Wallet-sync & dust → auto_position_from_balances=%s | dust_liq_enabled=%s | dust_min_quote=%.2f",
            self.AUTO_POSITION_FROM_BALANCES,
            self.dust_liquidation_enabled,
            self.DUST_MIN_QUOTE_USDT,
        )
        logger.info("🛡️ HYG guards → max_per_trade_usdt=%.2f | require_trading_status=%s | watchdog_warn_cooldown=%ss",
                    self.MAX_PER_TRADE_USDT, self.REQUIRE_TRADING_STATUS, self.WATCHDOG_WARN_COOLDOWN_SEC)
        logger.info("💵 Capital → target_free_usdt=%.2f", getattr(self.CAPITAL, "TARGET_FREE_USDT", 0.0))
        logger.info(
            "🛒 Execution → min_notional_quote=%.2f | maker_grace_s=%.2f | allow_taker_within_bps=%.2f | max_conc=%s | min_free_reserve_usdt=%.2f | no_remainder_below_quote=%.2f",
            getattr(self.EXECUTION, "MIN_NOTIONAL_QUOTE", 0.0),
            getattr(self.EXECUTION, "MAKER_GRACE_S", 0.0),
            getattr(self.EXECUTION, "ALLOW_TAKER_IF_WITHIN_BPS", 0.0),
            getattr(self.EXECUTION, "MAX_CONCURRENCY", "-"),
            getattr(self.EXECUTION, "MIN_FREE_RESERVE_USDT", 0.0),
            getattr(self.EXECUTION, "NO_REMAINDER_BELOW_QUOTE", 0.0),
        )
        logger.info(
            "🧪 Notional floors → probe=%.2f | liq_floor=%.2f | liq_target=%.2f",
            self.EXEC_PROBE_QUOTE,
            self.LIQ_ORCH_MIN_USDT_FLOOR,
            self.LIQ_ORCH_MIN_USDT_TARGET,
        )
        logger.info("🧮 Effective notional floor (all paths) = %.2f (baseline=%s)", self.MIN_ORDER_USDT, os.getenv("MIN_NOTIONAL_BASELINE", "5.0"))
        logger.info(
            "🧹 Dust register → enable=%s | ttl_days=%s | check_interval_s=%s | try_exchange_convert=%s",
            getattr(self.DUST_REGISTER, "ENABLE", False),
            getattr(self.DUST_REGISTER, "TTL_DAYS", "-"),
            getattr(self.DUST_REGISTER, "CHECK_INTERVAL_S", "-"),
            getattr(self.DUST_REGISTER, "TRY_EXCHANGE_CONVERT", False),
        )
        logger.info(
            "💧 CashRouter → enable=%s | allow_position_free=%s | protected=%s | max_actions=%s | sweep_min=%.2f | redeem_stables=%s | stables=%s | spread_cap_bps=%.2f | epsilon_usdt=%.2f",
            getattr(self.CASH_ROUTER, "ENABLE", False),
            getattr(self.CASH_ROUTER, "ALLOW_POSITION_FREE", False),
            ",".join(getattr(self.CASH_ROUTER, "PROTECTED_ASSETS", [])),
            getattr(self.CASH_ROUTER, "MAX_ACTIONS", "-"),
            getattr(self.CASH_ROUTER, "SWEEP_DUST_MIN", 0.0),
            getattr(self.CASH_ROUTER, "ENABLE_REDEEM_STABLES", False),
            ",".join(getattr(self.CASH_ROUTER, "STABLE_SYMBOLS", [])),
            getattr(self.CASH_ROUTER, "SPREAD_CAP_BPS", 0.0),
            getattr(self.CASH_ROUTER, "EPSILON_USDT", 0.0),
        )
        logger.info(
            "🧯 Liquidation → enable=%s | run_interval_s=%s | daily_cost_budget_bps=%.2f | spread_cap_bps=%.2f | priority=%s | stop_when_free_usdt_gte_target=%s",
            getattr(self.LIQUIDATION, "ENABLE", False),
            getattr(self.LIQUIDATION, "RUN_INTERVAL_S", "-"),
            getattr(self.LIQUIDATION, "DAILY_COST_BUDGET_BPS", 0.0),
            getattr(self.LIQUIDATION, "SPREAD_CAP_BPS", 0.0),
            getattr(self.LIQUIDATION, "PRIORITY", "-"),
            getattr(self.LIQUIDATION, "STOP_WHEN_FREE_USDT_GTE_TARGET", False),
        )
        logger.info(
            "🔍 Discovery → top_n_symbols=%s | accept_new_symbols=%s | min_24h_vol=%.2f",
            getattr(self.DISCOVERY, "TOP_N_SYMBOLS", "-"),
            getattr(self.DISCOVERY, "ACCEPT_NEW_SYMBOLS", False),
            getattr(self.DISCOVERY, "MIN_24H_VOL", 0.0),
        )
        logger.info(
            "📏 StrategyManager.OrderGuard → require_free_usdt_gte=%.2f | cooldown_s_per_symbol=%s",
            getattr(self.STRATEGY_MANAGER.ORDER_GUARD, "REQUIRE_FREE_USDT_GTE", 0.0),
            getattr(self.STRATEGY_MANAGER.ORDER_GUARD, "COOLDOWN_S_PER_SYMBOL", "-"),
        )

        logger.info("⏳ Ready-gating → wait_ready_secs=%s | gates=%s", getattr(self, "WAIT_READY_SECS", 0), ",".join(getattr(self, "GATE_READY_ON", [])))
        logger.info("🚦 StartupSanity → filters_coverage_pct>=%.1f%% | min_free_quote_factor=%.2f",
                    getattr(self.STARTUP, "FILTERS_COVERAGE_PCT", 80.0),
                    getattr(self.STARTUP, "MIN_FREE_QUOTE_FACTOR", 1.2))
        logger.info("🔧 Router/Exec floor check → floor=%.2f (EXEC_PROBE_QUOTE=%.2f, QUOTE_MIN_NOTIONAL=%.2f, DUST_MIN_QUOTE=%.2f)",
                    self.MIN_ORDER_USDT, self.EXEC_PROBE_QUOTE, self.EXECUTION.MIN_NOTIONAL_QUOTE, self.DUST_MIN_QUOTE_USDT)

    # ---------- Static fallbacks (kept for compatibility with older imports) ----------
    GLOBAL = globals().get("GLOBAL") or SimpleNamespace(
        MIN_ENTRY_QUOTE_USDT=20.0,
            DEFAULT_PLANNED_QUOTE=30.0,
        MAX_SPEND_PER_TRADE_USDT=50.0,
    )
