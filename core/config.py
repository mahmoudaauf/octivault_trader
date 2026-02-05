from dotenv import load_dotenv, find_dotenv
# Force-load the .env closest to your repo (not the caller's cwd), and override any stale env
load_dotenv(dotenv_path=find_dotenv(usecwd=True), override=True)

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
        STAGNATION_EXIT_MAX_LOSS_PCT = 0.0002

        # ---------- Min-notional avoidance (NEW) ----------
        # NOTE: Keep all min-notional knobs aligned via .env:
        #   MIN_ORDER_USDT, QUOTE_MIN_NOTIONAL, EXECUTION_MIN_NOTIONAL_QUOTE, DUST_MIN_QUOTE_USDT, EXEC_PROBE_QUOTE.
        #   On Binance spot, use a slight cushion (e.g., 5.10–5.25 USDT) to avoid fee/rounding rejections.
        MIN_ORDER_USDT = 5.0
        SAFE_ENTRY_USDT = 12.0
        MIN_POSITION_VALUE_USDT = 10.0
        MIN_ENTRY_USDT = 25.0
        MIN_TRADE_QUOTE = 50.0
        MAX_TRADE_QUOTE = 250.0
        EXEC_PROBE_QUOTE = 10.5
        MAX_HOLD_TIME_SEC = 1800.0
        LIQ_ORCH_MIN_USDT_FLOOR = 5.0
        LIQ_ORCH_MIN_USDT_TARGET = 6.0
        DUST_MIN_QUOTE_USDT = 5.0

        EXIT_EXCURSION_TICK_MULT = 2.0
        EXIT_EXCURSION_ATR_MULT = 0.35
        EXIT_EXCURSION_SPREAD_MULT = 3.0
        EXIT_FEE_BPS = 10.0
        EXIT_SLIPPAGE_BPS = 15.0
        # Entry economics guards
        MIN_PLANNED_QUOTE_FEE_MULT = 2.5
        MIN_ECONOMIC_TRADE_USDT = 20.0
        MIN_NET_PROFIT_AFTER_FEES = 0.0035
        MIN_PORTFOLIO_IMPROVEMENT_USD = 0.05
        MIN_PORTFOLIO_IMPROVEMENT_PCT = 0.0015
        BOOTSTRAP_VETO_COOLDOWN_SEC = 600.0
        # Micro-trade kill switch (low equity + low volatility)
        MICRO_TRADE_KILL_SWITCH_ENABLED = True
        MICRO_TRADE_KILL_EQUITY_MAX = 150.0
        MICRO_TRADE_KILL_ATR_FEE_MULT = 1.0
        MICRO_TRADE_KILL_FALLBACK_ATR_PCT = 0.0
        BUY_COOLDOWN_SEC = 60
        ENTRY_COOLDOWN_SEC = 60
        BUY_REENTRY_DELTA_PCT = 0.005
        BUY_REENTRY_DELTA_PCT_TEMP = 0.002
        BUY_REENTRY_DELTA_RESTORE_EQUITY = 220.0
        BUY_REENTRY_DELTA_RESTORE_TRADES = 3

        NO_TICK_THRESHOLD_SECONDS = 300
        RECOVERY_COOLDOWN_SECONDS = 180
        RECOVERY_STARTUP_GRACE_SECONDS = 180
        GLOBAL_RECOVER_MIN_GAP = 15
        MDF_RESTART_MIN_GAP = 20

        TARGET_PROFIT_USDT_PER_HOUR = 20.0 # Goal: 20 USDT/hour through frequency
        TARGET_PROFIT_LOOKBACK_MIN = 60
        TARGET_PROFIT_CHECK_SEC = 60
        UPTIME_GRACE_PERIOD_MIN = 30 # Ignore noise during startup
        TARGET_PROFIT_RATIO_PER_HOUR = 0.002 # 0.2% per hour of NAV
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

        ALLOW_DEBUG_SYMBOLS: bool = False
        TRENDHUNTER_MIN_DATA = 50
        TRENDHUNTER_EMA_SHORT = 12
        TRENDHUNTER_EMA_LONG = 26
        TRENDHUNTER_EMA_SIGNAL = 9
        ENABLE_COT_VALIDATION = True  # keep CoT guard path enabled
        DEFAULT_AGENT_DIRECT_EXECUTION = False  # global safe default; agents read AGENT_EXECUTION

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

            # Dynamic Symbol Discovery Settings
            self.DISCOVERY = SimpleNamespace(
                # Default to top 10 by volume if not specified (Phase 1 minimal fix)
                TOP_N_SYMBOLS=int(os.getenv("DISCOVERY_TOP_N_SYMBOLS", "10")),
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
            self.BASE_CAPITAL = float(os.getenv("BASE_CAPITAL", 400.0))
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

            # ---- Small-cap profile (auto-on for <= $250 unless overridden) ----
            _small_cap_env = os.getenv("SMALL_CAP_PROFILE")
            _small_cap_on = (
                (_small_cap_env or "").lower() == "true"
                if _small_cap_env is not None
                else self.BASE_CAPITAL <= 250.0
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
            self.MIN_ENTRY_QUOTE_USDT = float(os.getenv("MIN_ENTRY_QUOTE_USDT", str(Config.MIN_ENTRY_USDT)))
            self.DEFAULT_PLANNED_QUOTE = float(os.getenv("DEFAULT_PLANNED_QUOTE", "30.0"))
            self.EMIT_BUY_QUOTE = float(os.getenv("EMIT_BUY_QUOTE", "30.0"))

            # ---------- Profit-locked re-entry (compounding guard) ----------
            self.PROFIT_LOCK_REENTRY_ENABLED = os.getenv("PROFIT_LOCK_REENTRY_ENABLED", "true").lower() == "true"
            self.PROFIT_LOCK_BASE_QUOTE = float(os.getenv("PROFIT_LOCK_BASE_QUOTE", str(self.DEFAULT_PLANNED_QUOTE)))

            # IMPORTANT: fallback remains small-account friendly (5.0). Override via .env as needed.
            self.MIN_ORDER_USDT = float(os.getenv("MIN_ORDER_USDT", str(Config.MIN_ORDER_USDT)))
            self.SAFE_ENTRY_USDT = float(os.getenv("SAFE_ENTRY_USDT", str(Config.SAFE_ENTRY_USDT)))
            self.MIN_POSITION_VALUE_USDT = float(os.getenv("MIN_POSITION_VALUE_USDT", str(Config.MIN_POSITION_VALUE_USDT)))
            self.MIN_ENTRY_USDT = float(os.getenv("MIN_ENTRY_USDT", str(Config.MIN_ENTRY_USDT)))
            self.MIN_TRADE_QUOTE = float(os.getenv("MIN_TRADE_QUOTE", str(Config.MIN_TRADE_QUOTE)))
            self.MAX_TRADE_QUOTE = float(os.getenv("MAX_TRADE_QUOTE", str(Config.MAX_TRADE_QUOTE)))
            if self.MAX_TRADE_QUOTE and self.MAX_TRADE_QUOTE < self.MIN_TRADE_QUOTE:
                logger.warning(
                    "MAX_TRADE_QUOTE (%.2f) < MIN_TRADE_QUOTE (%.2f). Bumping max to min.",
                    self.MAX_TRADE_QUOTE,
                    self.MIN_TRADE_QUOTE,
                )
                self.MAX_TRADE_QUOTE = float(self.MIN_TRADE_QUOTE)
            self.MAX_HOLD_TIME_SEC = float(os.getenv("MAX_HOLD_TIME_SEC", str(Config.MAX_HOLD_TIME_SEC)))
            self.EXIT_EXCURSION_TICK_MULT = float(os.getenv("EXIT_EXCURSION_TICK_MULT", str(Config.EXIT_EXCURSION_TICK_MULT)))
            self.EXIT_EXCURSION_ATR_MULT = float(os.getenv("EXIT_EXCURSION_ATR_MULT", str(Config.EXIT_EXCURSION_ATR_MULT)))
            self.EXIT_EXCURSION_SPREAD_MULT = float(os.getenv("EXIT_EXCURSION_SPREAD_MULT", str(Config.EXIT_EXCURSION_SPREAD_MULT)))
            self.BUY_COOLDOWN_SEC = float(os.getenv("BUY_COOLDOWN_SEC", str(Config.BUY_COOLDOWN_SEC)))
            self.ENTRY_COOLDOWN_SEC = float(os.getenv("ENTRY_COOLDOWN_SEC", str(Config.ENTRY_COOLDOWN_SEC)))
            self.BUY_REENTRY_DELTA_PCT = float(os.getenv("BUY_REENTRY_DELTA_PCT", str(Config.BUY_REENTRY_DELTA_PCT)))
            self.BUY_REENTRY_DELTA_PCT_TEMP = float(os.getenv("BUY_REENTRY_DELTA_PCT_TEMP", str(Config.BUY_REENTRY_DELTA_PCT_TEMP)))
            self.BUY_REENTRY_DELTA_RESTORE_EQUITY = float(os.getenv("BUY_REENTRY_DELTA_RESTORE_EQUITY", str(Config.BUY_REENTRY_DELTA_RESTORE_EQUITY)))
            self.BUY_REENTRY_DELTA_RESTORE_TRADES = int(os.getenv("BUY_REENTRY_DELTA_RESTORE_TRADES", str(Config.BUY_REENTRY_DELTA_RESTORE_TRADES)))
            # Entry economics guards
            self.MIN_PLANNED_QUOTE_FEE_MULT = float(os.getenv("MIN_PLANNED_QUOTE_FEE_MULT", str(Config.MIN_PLANNED_QUOTE_FEE_MULT)))
            self.MIN_ECONOMIC_TRADE_USDT = float(os.getenv("MIN_ECONOMIC_TRADE_USDT", str(Config.MIN_ECONOMIC_TRADE_USDT)))
            self.MIN_NET_PROFIT_AFTER_FEES = float(os.getenv("MIN_NET_PROFIT_AFTER_FEES", str(Config.MIN_NET_PROFIT_AFTER_FEES)))
            self.MIN_PORTFOLIO_IMPROVEMENT_USD = float(os.getenv("MIN_PORTFOLIO_IMPROVEMENT_USD", str(Config.MIN_PORTFOLIO_IMPROVEMENT_USD)))
            self.MIN_PORTFOLIO_IMPROVEMENT_PCT = float(os.getenv("MIN_PORTFOLIO_IMPROVEMENT_PCT", str(Config.MIN_PORTFOLIO_IMPROVEMENT_PCT)))
            self.BOOTSTRAP_VETO_COOLDOWN_SEC = float(os.getenv("BOOTSTRAP_VETO_COOLDOWN_SEC", str(Config.BOOTSTRAP_VETO_COOLDOWN_SEC)))
            self.MICRO_TRADE_KILL_SWITCH_ENABLED = os.getenv(
                "MICRO_TRADE_KILL_SWITCH_ENABLED",
                str(Config.MICRO_TRADE_KILL_SWITCH_ENABLED)
            ).lower() == "true"
            self.MICRO_TRADE_KILL_EQUITY_MAX = float(os.getenv("MICRO_TRADE_KILL_EQUITY_MAX", str(Config.MICRO_TRADE_KILL_EQUITY_MAX)))
            self.MICRO_TRADE_KILL_ATR_FEE_MULT = float(os.getenv("MICRO_TRADE_KILL_ATR_FEE_MULT", str(Config.MICRO_TRADE_KILL_ATR_FEE_MULT)))
            self.MICRO_TRADE_KILL_FALLBACK_ATR_PCT = float(os.getenv("MICRO_TRADE_KILL_FALLBACK_ATR_PCT", str(Config.MICRO_TRADE_KILL_FALLBACK_ATR_PCT)))

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

            # Ensure dust controls exist (some logs assume these attributes)
            self.DUST_LIQUIDATION_ENABLED = os.getenv("DUST_LIQUIDATION_ENABLED", "false").lower() == "true"
            # Allow dust positions to bypass re-entry lock (merge dust back into tradable size)
            self.DUST_REENTRY_OVERRIDE = os.getenv("DUST_REENTRY_OVERRIDE", "true").lower() == "true"
            logger.info(
                "🧹 Dust authority → reentry_override=%s | liquidation_enabled=%s",
                self.DUST_REENTRY_OVERRIDE,
                self.DUST_LIQUIDATION_ENABLED,
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
                "COLD_BOOTSTRAP_BLOCK,PORTFOLIO_FULL",
            )

            # ---------- MetaController Execution (Phase A) ----------
            self.META_MIN_AGENTS = int(os.getenv("META_MIN_AGENTS", "1"))
            self.META_MIN_CONF = float(os.getenv("META_MIN_CONF", "0.70"))
            self.MAX_TRADES_PER_HOUR = int(os.getenv("MAX_TRADES_PER_HOUR", "10"))  # Increased for 20 USDT/h target
            self.MAX_TRADES_PER_DAY = int(os.getenv("MAX_TRADES_PER_DAY", str(Config.MAX_TRADES_PER_DAY)))
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

            self.TP_ATR_MULT = float(os.getenv("TP_ATR_MULT", "1.2"))
            self.SL_ATR_MULT = float(os.getenv("SL_ATR_MULT", "0.8"))
            # Optional percent clamps for TP/SL distances (fallbacks for ATR variance)
            self.TP_PCT_MIN = float(os.getenv("TP_PCT_MIN", "0.003"))  # 0.30%
            self.TP_PCT_MAX = float(os.getenv("TP_PCT_MAX", "0.006"))  # 0.60%
            self.SL_PCT_MIN = float(os.getenv("SL_PCT_MIN", "0.005"))  # 0.50%
            self.SL_PCT_MAX = float(os.getenv("SL_PCT_MAX", "0.008"))  # 0.80%
            self.TP_TARGET_PCT = float(os.getenv("TP_TARGET_PCT", str(Config.TP_TARGET_PCT)))
            self.SL_CAP_PCT = float(os.getenv("SL_CAP_PCT", str(Config.SL_CAP_PCT)))
            self.TP_MIN_BUFFER_BPS = float(os.getenv("TP_MIN_BUFFER_BPS", "5"))

            # ---------- SELL economic gates ----------
            # If False, non-liquidation SELLs must clear fee-aware net PnL gate.
            self.ALLOW_SELL_BELOW_FEE = os.getenv("ALLOW_SELL_BELOW_FEE", "false").lower() == "true"
            # Minimum net PnL (USDT) required for non-liquidation SELLs (fees included).
            self.SELL_MIN_NET_PNL_USDT = float(os.getenv("SELL_MIN_NET_PNL_USDT", "0.05"))

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
            self.MIN_HOLD_SEC = float(os.getenv("MIN_HOLD_SEC", "120"))
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

            # ---------- Capital recovery (floor-triggered SELL escalation) ----------
            # Force small-profit or time-based exits when capital floor is violated.
            # 0.0002 = 0.02%; 10 minutes default max age.
            self.CAPITAL_RECOVERY_MIN_PNL_PCT = float(os.getenv("CAPITAL_RECOVERY_MIN_PNL_PCT", "0.0002"))
            self.CAPITAL_RECOVERY_MAX_AGE_MINUTES = float(os.getenv("CAPITAL_RECOVERY_MAX_AGE_MINUTES", "10"))
            self.CAPITAL_RECOVERY_FORCE_SELL_AFTER_SEC = float(os.getenv("CAPITAL_RECOVERY_FORCE_SELL_AFTER_SEC", "600"))

            # ---------- Meta time-exit (capital rotation) ----------
            self.TIME_EXIT_ENABLED = os.getenv("TIME_EXIT_ENABLED", "true").lower() == "true"
            self.TIME_EXIT_MIN_SECONDS = float(os.getenv("TIME_EXIT_MIN_SECONDS", "0"))
            self.TIME_EXIT_MIN_MINUTES = float(os.getenv("TIME_EXIT_MIN_MINUTES", "10"))
            self.TIME_EXIT_MIN_HOURS = float(os.getenv("TIME_EXIT_MIN_HOURS", "0"))
            self.TIME_EXIT_MIN_PNL_PCT = float(os.getenv("TIME_EXIT_MIN_PNL_PCT", "0.0"))
            self.TIME_EXIT_FORCE_SELL = os.getenv("TIME_EXIT_FORCE_SELL", "true").lower() == "true"
            
            # ---------- Phase A - Frequency Engineering ----------
            # Tiered Confidence System
            self.META_TIER_A_CONF = float(os.getenv("META_TIER_A_CONF", "0.70"))  # Confidence baseline
            self.META_TIER_B_CONF = float(os.getenv("META_TIER_B_CONF", "0.50"))  # Micro-sizing threshold
            self.META_MICRO_SIZE_USDT = float(os.getenv("META_MICRO_SIZE_USDT", "25.0"))  # Micro trade size
            
            # Symbol Concentration
            self.META_SYMBOL_CONCENTRATION_LIMIT = int(os.getenv("META_SYMBOL_CONCENTRATION_LIMIT", "2"))
            self.MAX_POSITIONS_TOTAL = int(os.getenv("MAX_POSITIONS_TOTAL", "2")) # Limit open positions
            
            # RiskManager - Tier B Allowance
            self.TIER_B_MAX_QUOTE = float(os.getenv("TIER_B_MAX_QUOTE", "12.0"))  # Max quote for micro trades
            min_econ_floor = float(getattr(self, "MIN_ECONOMIC_TRADE_USDT", 0.0) or 0.0)
            if min_econ_floor > 0 and self.TIER_B_MAX_QUOTE < min_econ_floor:
                logger.warning(
                    "TIER_B_MAX_QUOTE (%.2f) < MIN_ECONOMIC_TRADE_USDT (%.2f). Bumping Tier-B cap to match economic floor.",
                    self.TIER_B_MAX_QUOTE,
                    min_econ_floor,
                )
                self.TIER_B_MAX_QUOTE = float(min_econ_floor)
            
            # Portfolio Replacement
            self.REPLACEMENT_THRESHOLD_ROI = float(os.getenv("REPLACEMENT_THRESHOLD_ROI", "0.02")) # 2% gap required
            
            # TP/SL Engine - Tier B Fast Exits
            self.TIER_B_TTL_SEC = int(os.getenv("TIER_B_TTL_SEC", "300"))  # 5 minutes max holding time
            
            # Scaling Target USDT based on NAV
            self.TARGET_PROFIT_RATIO_PER_HOUR = float(os.getenv("TARGET_PROFIT_RATIO_PER_HOUR", "0.002"))
            self.UPTIME_GRACE_PERIOD_MIN = int(os.getenv("UPTIME_GRACE_PERIOD_MIN", "30"))

            # ---------- Execution policy ----------
            self.EXECUTION = SimpleNamespace(
                MIN_ORDER_USDT=self.MIN_ORDER_USDT,
                SAFE_ENTRY_USDT=self.SAFE_ENTRY_USDT,
                MIN_POSITION_VALUE_USDT=self.MIN_POSITION_VALUE_USDT,
                MIN_ENTRY_USDT=self.MIN_ENTRY_USDT,
                MAX_HOLD_TIME_SEC=self.MAX_HOLD_TIME_SEC,
                QUOTE_MIN_NOTIONAL=float(os.getenv("QUOTE_MIN_NOTIONAL", str(self.MIN_ORDER_USDT))),
                RESPECT_EXCHANGE_MIN_NOTIONAL=os.getenv("RESPECT_EXCHANGE_MIN_NOTIONAL", "true").lower() == "true",
                ALLOW_PARTIAL_FILLS=os.getenv("ALLOW_PARTIAL_FILLS", "true").lower() == "true",
                MIN_NOTIONAL_QUOTE=float(os.getenv("EXECUTION_MIN_NOTIONAL_QUOTE",
                                    os.getenv("MIN_NOTIONAL_QUOTE", str(self.MIN_ORDER_USDT)))),
                MAKER_GRACE_S=float(os.getenv("EXECUTION_MAKER_GRACE_S", "1.5")),
                ALLOW_TAKER_IF_WITHIN_BPS=float(os.getenv("EXECUTION_ALLOW_TAKER_IF_WITHIN_BPS", "3.0")),
                MAX_CONCURRENCY=int(os.getenv("EXECUTION_MAX_CONCURRENCY", os.getenv("MAX_CONCURRENCY", "8"))),
                MIN_FREE_RESERVE_USDT=float(os.getenv("EXECUTION_MIN_FREE_RESERVE_USDT", "0.50")),
                NO_REMAINDER_BELOW_QUOTE=float(os.getenv("EXECUTION_NO_REMAINDER_BELOW_QUOTE", "2.0")),
            )

            # ---------- Capital Allocator ----------
            _alloc_ranges_raw = os.getenv("CAPITAL_ALLOCATOR_AGENT_ALLOC_RANGES", "")
            if _alloc_ranges_raw:
                try:
                    _alloc_ranges = json.loads(_alloc_ranges_raw)
                except Exception:
                    _alloc_ranges = {}
            else:
                # Default: MLForecaster gets 40-60% of free capital
                _alloc_ranges = {"MLForecaster": [0.40, 0.60]}

            self.CAPITAL_ALLOCATOR = {
                "ENABLED": os.getenv("CAPITAL_ALLOCATOR_ENABLED", "true").lower() == "true",  # Changed default to "true"
                "INTERVAL_MIN": float(os.getenv("CAPITAL_ALLOCATOR_INTERVAL_MIN", "15")),
                "AGENT_ALLOC_RANGES": _alloc_ranges,
            }

            # --- SINGLE place to enforce the unified notional floor (do this AFTER EXECUTION exists)
            # Allow a configurable baseline so small-account runs are not silently clamped by a hard literal.
            _baseline = float(os.getenv("MIN_NOTIONAL_BASELINE", "5.10"))
            _floor = max(
                float(self.MIN_ORDER_USDT),
                float(getattr(self.EXECUTION, "QUOTE_MIN_NOTIONAL", 0.0)),
                float(getattr(self.EXECUTION, "MIN_NOTIONAL_QUOTE", 0.0)),
                float(self.EXEC_PROBE_QUOTE),
                _baseline,  # safety baseline (Binance 1–5 USDT tier compatible)
            )
            self.MIN_ORDER_USDT = _floor
            self.EXEC_PROBE_QUOTE = _floor
            self.EXECUTION.QUOTE_MIN_NOTIONAL = _floor
            self.EXECUTION.MIN_NOTIONAL_QUOTE = _floor

            # (Optional) keep dust aligned to floor only if the user did NOT set it explicitly in the environment.
            # This respects .env values like DUST_MIN_QUOTE_USDT=5.0 even when floor is higher.
            if "DUST_MIN_QUOTE_USDT" not in os.environ:
                self.DUST_MIN_QUOTE_USDT = max(self.DUST_MIN_QUOTE_USDT, _floor)

            # Log after enforcement to reflect final values
            logger.info(
                "🧮 Effective notional floor (all paths) = %.2f | probe=%.2f | quote_min=%.2f | baseline=%.2f",
                self.MIN_ORDER_USDT, self.EXEC_PROBE_QUOTE, self.EXECUTION.MIN_NOTIONAL_QUOTE, _baseline
            )
            self.LIVE_MODE = os.getenv("LIVE_MODE", "True").lower() == "true"
            self.SIMULATION_MODE = os.getenv("SIMULATION_MODE", "False").lower() == "true"
            self.PAPER_MODE = os.getenv("PAPER_MODE", "False").lower() == "true"
            self.TESTNET_MODE = os.getenv("TESTNET_MODE", "False").lower() == "true"

            # ---------- Region / Account Type (NEW) ----------
            # BINANCE_REGION: 'global' or 'us'
            self.BINANCE_REGION = os.getenv("BINANCE_REGION", "global").strip().lower()
            # BINANCE_ACCOUNT_TYPE: 'spot' or 'futures' (UM)
            self.BINANCE_ACCOUNT_TYPE = os.getenv("BINANCE_ACCOUNT_TYPE", "spot").strip().lower()

            # ---------- Binance Keys ----------
            self.BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
            self.BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")
            self.BINANCE_TESTNET_API_KEY = os.getenv("BINANCE_TESTNET_API_KEY", "")
            self.BINANCE_TESTNET_API_SECRET = os.getenv("BINANCE_TESTNET_API_SECRET", "")

            # ---------- Resolve endpoints (NEW) ----------
            # Allow manual override, otherwise derive from region+type+testnet.
            self.BINANCE_BASE_URL = os.getenv("BINANCE_BASE_URL", "").strip()
            if not self.BINANCE_BASE_URL:
                self.BINANCE_BASE_URL = self._resolve_base_url(
                    region=self.BINANCE_REGION,
                    account_type=self.BINANCE_ACCOUNT_TYPE,
                    testnet=self.TESTNET_MODE,
                )

            # ---------- Intervals & Timeouts ----------
            self.WATCHDOG_INTERVAL = int(os.getenv("WATCHDOG_INTERVAL", os.getenv("WATCHDOG_CHECK_INTERVAL_SECONDS", str(self.DEFAULT_WATCHDOG_INTERVAL))))
            self.WATCHDOG_WARN_COOLDOWN_SEC = int(os.getenv("WATCHDOG_WARN_COOLDOWN_SEC", 60))
            self.HEARTBEAT_INTERVAL = int(os.getenv("HEARTBEAT_INTERVAL", 30))
            self.EXECUTION_INTERVAL = float(os.getenv("EXECUTION_INTERVAL", 1.0))
            self.DIAGNOSTICS_INTERVAL = int(os.getenv("DIAGNOSTICS_INTERVAL", 60))
            self.AGENT_MANAGER_INTERVAL = int(os.getenv("AGENT_MANAGER_INTERVAL", 60))
            self.META_CONTROLLER_INTERVAL = int(os.getenv("META_CONTROLLER_INTERVAL", 30))
            self.TPSL_ENGINE_INTERVAL = int(os.getenv("TPSL_ENGINE_INTERVAL", 10))
            self.META_SIGNAL_COOLDOWN = int(os.getenv("META_SIGNAL_COOLDOWN", 10))
            self.RECOVERY_CHECK_INTERVAL = int(os.getenv("RECOVERY_CHECK_INTERVAL", 10))
            self.RECOVERY_MAX_INACTIVE_TIME = int(os.getenv("RECOVERY_MAX_INACTIVE_TIME", 60))
            # Expose AppContext startup timeout knob with a safer default
            self.START_TIMEOUT_SEC = float(os.getenv("START_TIMEOUT_SEC", "30"))
            self.SHUTDOWN_TIMEOUT_SECONDS = int(os.getenv("SHUTDOWN_TIMEOUT_SECONDS", 10))
            self.DASHBOARD_ENABLED = os.getenv("DASHBOARD_ENABLED", "true").lower() == "true"

            # ---- Readiness gating knobs ----
            self.WAIT_READY_SECS = int(os.getenv("WAIT_READY_SECS", "90"))
            self.GATE_READY_ON = [s.strip() for s in os.getenv("GATE_READY_ON", "market_data,execution,capital,exchange,startup_sanity").split(",") if s.strip()]
            # Bridge: keep a plain string version for AppContext._cfg("GATE_READY_ON", ...) which expects a CSV string
            try:
                _gate_csv = ",".join(self.GATE_READY_ON) if isinstance(self.GATE_READY_ON, list) else str(self.GATE_READY_ON)
                os.environ["GATE_READY_ON"] = _gate_csv
            except Exception:
                pass

            # ---- Startup namespace for thresholds ----
            self.STARTUP = SimpleNamespace(
                FILTERS_COVERAGE_PCT=float(os.getenv("STARTUP.filters_coverage_pct", "80")),
                MIN_FREE_QUOTE_FACTOR=float(os.getenv("STARTUP.min_free_quote_factor", "1.2")),
            )
            # Bridge: publish dotted-key env vars for AppContext._cfg("STARTUP.*") lookups
            try:
                os.environ.setdefault("STARTUP.filters_coverage_pct", str(getattr(self.STARTUP, "FILTERS_COVERAGE_PCT", 80.0)))
                os.environ.setdefault("STARTUP.min_free_quote_factor", str(getattr(self.STARTUP, "MIN_FREE_QUOTE_FACTOR", 1.2)))
            except Exception:
                pass

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
                self.DUST_LIQUIDATION_ENABLED,
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
            DEFAULT_PLANNED_QUOTE=20.0,
            MAX_SPEND_PER_TRADE_USDT=50.0,
        )
