#!/usr/bin/env python3
"""
🎯 MASTER SYSTEM ORCHESTRATOR
================================

Unified entry point for complete Octi AI Trading Bot system.
Properly initializes all 7 layers in correct order and runs them coordinated.

This is the CORRECT way to run the system:
1. Validates all prerequisites
2. Initializes layers in dependency order
3. Runs all components concurrently
4. Monitors system health
5. Graceful shutdown on errors

USAGE:
    export APPROVE_LIVE_TRADING=YES
    python3 🎯_MASTER_SYSTEM_ORCHESTRATOR.py [--duration 24] [--paper]

Environment Variables:
    APPROVE_LIVE_TRADING=YES        ← Required for real trading
    TRADING_DURATION_HOURS=24       ← How long to run (default: 24)
    PAPER_TRADING=false             ← Set to 'true' for paper trading

Exit codes:
    0 = Successful shutdown
    1 = Initialization failed
    2 = Configuration error
    3 = Prerequisite check failed
"""

import asyncio
import inspect
import logging
import os
import sys
import signal
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging():
    """Configure comprehensive logging"""
    log_dir = Path("/tmp")
    log_file = log_dir / "octivault_master_orchestrator.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)-8s] %(name)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

logger = setup_logging()

# ============================================================================
# IMPORTS
# ============================================================================

try:
    from dotenv import load_dotenv
    from core.config import Config as CoreConfig
    from core.exchange_client import ExchangeClient
    from core.shared_state import SharedState
    from core.polling_coordinator import PollingCoordinator, PollingConfig
    from core.signal_manager import SignalManager
    from core.execution_manager import ExecutionManager
    from core.risk_manager import RiskManager
    from core.tp_sl_engine import TPSLEngine  # ← CRITICAL: TP/SL monitoring for position management
    from core.meta_controller import MetaController
    from core.agent_manager import AgentManager
    from core.market_data_feed import MarketDataFeed  # ← CRITICAL: Add market data streaming
    from core.health_monitor import HealthMonitor
    from core.watchdog import Watchdog
    from core.heartbeat import Heartbeat
    
    # OPTIONAL COMPONENTS (Enhanced System Features)
    try:
        from core.recovery_engine import RecoveryEngine
        HAS_RECOVERY_ENGINE = True
    except Exception:
        HAS_RECOVERY_ENGINE = False

    try:
        from core.startup_orchestrator import StartupOrchestrator
        HAS_STARTUP_ORCHESTRATOR = True
    except Exception:
        HAS_STARTUP_ORCHESTRATOR = False
        logger.warning("⚠️ StartupOrchestrator not found — state hydration will be skipped")

    try:
        from core.cash_router import CashRouter
        HAS_CASH_ROUTER = True
    except Exception:
        HAS_CASH_ROUTER = False

    try:
        from core.external_adoption_engine import ExternalAdoptionEngine
        HAS_EXTERNAL_ADOPTION_ENGINE = True
    except Exception:
        HAS_EXTERNAL_ADOPTION_ENGINE = False

    try:
        from core.dead_capital_healer import DeadCapitalHealer
        HAS_DEAD_CAPITAL_HEALER = True
    except Exception:
        HAS_DEAD_CAPITAL_HEALER = False
    
    try:
        from core.bootstrap_manager import BootstrapDustBypassManager
        HAS_BOOTSTRAP_MANAGER = True
    except Exception:
        HAS_BOOTSTRAP_MANAGER = False
    
    try:
        from core.performance_monitor import PerformanceMonitor
        HAS_PERFORMANCE_MONITOR = True
    except Exception:
        HAS_PERFORMANCE_MONITOR = False

    try:
        from core.profit_target_engine import ProfitTargetEngine
        HAS_PROFIT_TARGET_ENGINE = True
    except Exception:
        HAS_PROFIT_TARGET_ENGINE = False
    
    try:
        from core.capital_allocator import CapitalAllocator
        HAS_CAPITAL_ALLOCATOR = True
    except Exception:
        HAS_CAPITAL_ALLOCATOR = False
    
    try:
        from core.symbol_manager import SymbolManager
        HAS_SYMBOL_MANAGER = True
    except Exception:
        HAS_SYMBOL_MANAGER = False

    try:
        from core.universe_rotation_engine import UniverseRotationEngine
        HAS_UNIVERSE_ROTATION_ENGINE = True
    except Exception:
        HAS_UNIVERSE_ROTATION_ENGINE = False

    try:
        from utils.pnl_calculator import PnLCalculator
        HAS_PNL_CALCULATOR = True
    except Exception:
        HAS_PNL_CALCULATOR = False

    try:
        from core.performance_evaluator import PerformanceEvaluator
        HAS_PERFORMANCE_EVALUATOR = True
    except Exception:
        HAS_PERFORMANCE_EVALUATOR = False

    try:
        from core.portfolio_balancer import PortfolioBalancer
        HAS_PORTFOLIO_BALANCER = True
    except Exception:
        HAS_PORTFOLIO_BALANCER = False

    try:
        from core.liquidation_orchestrator import LiquidationOrchestrator
        HAS_LIQUIDATION_ORCHESTRATOR = True
    except Exception:
        HAS_LIQUIDATION_ORCHESTRATOR = False

    try:
        from core.alert_system import AlertSystem
        HAS_ALERT_SYSTEM = True
    except Exception:
        HAS_ALERT_SYSTEM = False

    try:
        from core.exchange_truth_auditor import ExchangeTruthAuditor
        HAS_EXCHANGE_TRUTH_AUDITOR = True
    except Exception:
        HAS_EXCHANGE_TRUTH_AUDITOR = False

    try:
        from core.position_merger_enhanced import PositionMergerEnhanced
        HAS_POSITION_MERGER_ENHANCED = True
    except Exception:
        HAS_POSITION_MERGER_ENHANCED = False

    try:
        from core.rebalancing_engine import RebalancingEngine, AllocationTarget, RebalanceStrategy
        HAS_REBALANCING_ENGINE = True
    except Exception:
        HAS_REBALANCING_ENGINE = False

    try:
        from core.volatility_regime import VolatilityRegimeDetector
        HAS_VOLATILITY_REGIME = True
    except Exception:
        HAS_VOLATILITY_REGIME = False

    try:
        from core.compounding_engine import CompoundingEngine
        HAS_COMPOUNDING_ENGINE = True
    except Exception:
        HAS_COMPOUNDING_ENGINE = False
    
    try:
        from core.three_bucket_manager import ThreeBucketPortfolioManager
        HAS_THREE_BUCKET_MANAGER = True
    except Exception:
        HAS_THREE_BUCKET_MANAGER = False
    
    try:
        from core.portfolio_segmentation import PortfolioSegmentationManager
        HAS_PORTFOLIO_SEGMENTATION = True
    except Exception:
        HAS_PORTFOLIO_SEGMENTATION = False

    # =========================================================================
    # EXTENSION MODULES — All previously un-wired scripts now activated
    # =========================================================================

    # --- Utility helpers ---
    try:
        from utils.logging_setup import setup_logging as util_setup_logging
        HAS_UTIL_LOGGING = True
    except Exception:
        HAS_UTIL_LOGGING = False

    try:
        from utils.ta_indicators import calculate_ema, calculate_rsi, calculate_volume_surge
        HAS_TA_INDICATORS = True
    except Exception:
        HAS_TA_INDICATORS = False

    try:
        from utils.ohlcv_cache import load_ohlcv_from_cache, save_ohlcv_to_csv, fetch_and_cache_ohlcv
        HAS_OHLCV_CACHE = True
    except Exception:
        HAS_OHLCV_CACHE = False

    try:
        from utils.tuned_params import get_tuned_params, get_symbol_volatility_class, get_adaptive_learning_rate
        HAS_TUNED_PARAMS = True
    except Exception:
        HAS_TUNED_PARAMS = False

    # --- Contracts & constants ---
    try:
        from core.config_constants import TimeoutConstants, RetryConstants, CapitalConstants, ConfidenceConstants, get_all_constants
        HAS_CONFIG_CONSTANTS = True
    except Exception:
        HAS_CONFIG_CONSTANTS = False

    try:
        from core.config_validator import validate_config_on_startup
        HAS_CONFIG_VALIDATOR = True
    except Exception:
        HAS_CONFIG_VALIDATOR = False

    try:
        from core.contracts import TradeIntent as ContractTradeIntent, OrderSide as ContractOrderSide
        HAS_CONTRACTS = True
    except Exception:
        HAS_CONTRACTS = False

    try:
        from core.baseline_trading_kernel import KernelState, MetaPolicy, Readiness
        HAS_BASELINE_KERNEL = True
    except Exception:
        HAS_BASELINE_KERNEL = False

    try:
        from core.layer_contracts import LayerInput, LayerOutput, WalletLayerContract, PortfolioLayerContract
        HAS_LAYER_CONTRACTS = True
    except Exception:
        HAS_LAYER_CONTRACTS = False

    # --- Market intelligence ---
    try:
        from core.market_regime_detector import MarketRegimeDetector, MarketRegime
        HAS_MARKET_REGIME_DETECTOR = True
    except Exception:
        HAS_MARKET_REGIME_DETECTOR = False

    try:
        from core.market_regime_integration import RegimeAwareMediator
        HAS_REGIME_INTEGRATION = True
    except Exception:
        HAS_REGIME_INTEGRATION = False

    try:
        from core.discovery_coordinator import DiscoveryCoordinator
        HAS_DISCOVERY_COORDINATOR = True
    except Exception:
        HAS_DISCOVERY_COORDINATOR = False

    # --- Capital & risk management ---
    try:
        from core.capital_symbol_governor import CapitalSymbolGovernor
        HAS_CAPITAL_SYMBOL_GOVERNOR = True
    except Exception:
        HAS_CAPITAL_SYMBOL_GOVERNOR = False

    try:
        from core.reserve_manager import ReserveManager
        HAS_RESERVE_MANAGER = True
    except Exception:
        HAS_RESERVE_MANAGER = False

    # --- Execution & order management ---
    try:
        from core.order_cache_manager import OrderCacheManager
        HAS_ORDER_CACHE_MANAGER = True
    except Exception:
        HAS_ORDER_CACHE_MANAGER = False

    try:
        from core.action_router import ActionRouter
        HAS_ACTION_ROUTER = True
    except Exception:
        HAS_ACTION_ROUTER = False

    try:
        from core.exit_arbitrator import ExitArbitrator, get_arbitrator
        HAS_EXIT_ARBITRATOR = True
    except Exception:
        HAS_EXIT_ARBITRATOR = False

    try:
        from core.balance_sync_backoff import BalanceSyncRetryManager, BalanceSyncCoordinator
        HAS_BALANCE_SYNC_BACKOFF = True
    except Exception:
        HAS_BALANCE_SYNC_BACKOFF = False

    try:
        from core.retry_manager import RetryManager
        HAS_RETRY_MANAGER = True
    except Exception:
        HAS_RETRY_MANAGER = False

    # --- Position & portfolio management ---
    try:
        from core.position_manager import PositionManager
        HAS_POSITION_MANAGER = True
    except Exception:
        HAS_POSITION_MANAGER = False

    try:
        from core.portfolio_manager import PortfolioManager
        HAS_PORTFOLIO_MANAGER = True
    except Exception:
        HAS_PORTFOLIO_MANAGER = False

    # --- State & synchronization ---
    try:
        from core.state_synchronizer import StateSynchronizer, StateSyncronizationTask
        HAS_STATE_SYNCHRONIZER = True
    except Exception:
        HAS_STATE_SYNCHRONIZER = False

    try:
        from core.layer_orchestrator import LayerOrchestrator
        HAS_LAYER_ORCHESTRATOR = True
    except Exception:
        HAS_LAYER_ORCHESTRATOR = False

    # --- Market data ---
    try:
        from core.ws_market_data import WebSocketMarketData
        HAS_WS_MARKET_DATA = True
    except Exception:
        HAS_WS_MARKET_DATA = False

    # --- Observability ---
    try:
        from core.prometheus_exporter import SafetyGuardMetrics, get_metrics
        HAS_PROMETHEUS_EXPORTER = True
    except Exception:
        HAS_PROMETHEUS_EXPORTER = False

    try:
        from core.health_check_manager import HealthCheckManager
        HAS_HEALTH_CHECK_MANAGER = True
    except Exception:
        HAS_HEALTH_CHECK_MANAGER = False

    try:
        from core.health_endpoints import HealthEndpoints, register_health_endpoints
        HAS_HEALTH_ENDPOINTS = True
    except Exception:
        HAS_HEALTH_ENDPOINTS = False

    try:
        from core.apm_instrument import APMInstrument, get_apm_instrument
        HAS_APM_INSTRUMENT = True
    except Exception:
        HAS_APM_INSTRUMENT = False

    try:
        from core.dashboard import initialize_dashboard
        HAS_DASHBOARD = True
    except Exception:
        HAS_DASHBOARD = False

    # --- Advanced / conditional ---
    try:
        from core.replay_engine import DeterministicReplayEngine
        HAS_REPLAY_ENGINE = True
    except Exception:
        HAS_REPLAY_ENGINE = False

    try:
        from core.chaos_monkey import ChaosMonkey, get_chaos_monkey
        HAS_CHAOS_MONKEY = True
    except Exception:
        HAS_CHAOS_MONKEY = False

except ImportError as e:
    logger.error(f"❌ Failed to import core modules: {e}")
    sys.exit(1)

load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

class OrchestratorConfig:
    """Configuration for master orchestrator"""
    
    def __init__(self):
        # Load full production config once and use it as a fallback source.
        # This keeps orchestrator lightweight while preserving all core knobs.
        self._base_config = None
        try:
            self._base_config = CoreConfig()
        except Exception as e:
            logger.warning("⚠️ Could not initialize core Config fallback: %s", e, exc_info=True)

        self.live_mode = os.getenv("LIVE_MODE", "True").lower() == "true"
        self.paper_trading = os.getenv("PAPER_TRADING", "false").lower() == "true"
        self.testnet_mode = os.getenv("TESTNET_MODE", "False").lower() == "true"
        self.duration_hours = float(os.getenv("TRADING_DURATION_HOURS", "24"))
        self.polling_interval_sec = int(os.getenv("POLLING_INTERVAL_SECONDS", "30"))
        self.controller_cycle_sec = int(os.getenv("META_CONTROLLER_CYCLE_SECONDS", "10"))
        self.heartbeat_interval_sec = int(os.getenv("HEARTBEAT_INTERVAL_SECONDS", "30"))
        # Capital policy: keep a fixed reserve in quote (USDT) for sustainable compounding.
        self.quote_asset = str(os.getenv("QUOTE_ASSET", "USDT")).upper()
        self.capital_floor_pct = float(os.getenv("CAPITAL_FLOOR_PCT", "0.20"))
        self.absolute_min_floor = float(os.getenv("ABSOLUTE_MIN_FLOOR", "10.0"))
        reserve_ratio_env = os.getenv("QUOTE_RESERVE_RATIO", "").strip()
        reserve_ratio = float(reserve_ratio_env) if reserve_ratio_env else self.capital_floor_pct
        self.quote_reserve_ratio = max(self.capital_floor_pct, reserve_ratio)
        self.quote_min_reserve = float(os.getenv("QUOTE_MIN_RESERVE", str(self.absolute_min_floor)))
        # Compatibility aliases for modules that read uppercase config keys.
        self.QUOTE_ASSET = self.quote_asset
        self.CAPITAL_FLOOR_PCT = self.capital_floor_pct
        self.ABSOLUTE_MIN_FLOOR = self.absolute_min_floor
        self.QUOTE_RESERVE_RATIO = self.quote_reserve_ratio
        self.QUOTE_MIN_RESERVE = self.quote_min_reserve
        # Signal injector is useful for paper tests, but dangerous in live mode.
        default_injector = "true" if self.paper_trading else "false"
        self.enable_signal_injector = os.getenv("ENABLE_SIGNAL_INJECTOR", default_injector).lower() == "true"
        # Optional advanced portfolio engines (off by default; enable explicitly).
        self.enable_volatility_regime = os.getenv("ENABLE_VOLATILITY_REGIME", "false").lower() == "true"
        self.enable_compounding_engine = os.getenv("ENABLE_COMPOUNDING_ENGINE", "false").lower() == "true"
        self.enable_alert_system = os.getenv("ENABLE_ALERT_SYSTEM", "false").lower() == "true"
        self.enable_exchange_truth_auditor = os.getenv("ENABLE_EXCHANGE_TRUTH_AUDITOR", "false").lower() == "true"
        self.truth_auditor_mode = os.getenv("TRUTH_AUDITOR_MODE", "continuous").lower()
        self.enable_position_merger_enhanced = os.getenv("ENABLE_POSITION_MERGER_ENHANCED", "false").lower() == "true"
        self.enable_rebalancing_engine = os.getenv("ENABLE_REBALANCING_ENGINE", "false").lower() == "true"
        self.position_merger_interval_sec = float(os.getenv("POSITION_MERGER_INTERVAL_SEC", "300"))
        self.rebalancing_engine_loop_sec = float(os.getenv("REBALANCING_ENGINE_LOOP_SEC", "300"))
        self.rebalancing_engine_strategy = os.getenv("REBALANCING_ENGINE_STRATEGY", "DRIFT_THRESHOLD").upper()
        self.enable_universe_rotation = os.getenv("ENABLE_UNIVERSE_ROTATION", "true").lower() == "true"
        self.uure_interval_sec = float(os.getenv("UURE_INTERVAL_SEC", "300"))
        # Profit discipline mode: force non-emergency SELLs to pass fee-aware profit gate.
        self.STRICT_PROFIT_ONLY_SELLS = os.getenv("STRICT_PROFIT_ONLY_SELLS", "false").lower() == "true"
        # Capital governor thresholds (forwarded to MetaController/CapitalGovernor).
        self.CAPITAL_MICRO_THRESHOLD = float(os.getenv("CAPITAL_MICRO_THRESHOLD", "500"))
        self.CAPITAL_SMALL_THRESHOLD = float(os.getenv("CAPITAL_SMALL_THRESHOLD", "2000"))
        self.CAPITAL_MEDIUM_THRESHOLD = float(os.getenv("CAPITAL_MEDIUM_THRESHOLD", "10000"))
        self.TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

    def __getattr__(self, name: str):
        """
        Fallback to full core Config attributes for any field not defined here.
        This prevents thin-orchestrator config drift from silently disabling
        capital-quality controls (floors, dust thresholds, sizing, etc.).
        """
        base = self.__dict__.get("_base_config")
        if base is not None and hasattr(base, name):
            return getattr(base, name)
        raise AttributeError(f"{self.__class__.__name__} has no attribute '{name}'")
        
    def validate(self) -> bool:
        """Validate configuration consistency"""
        logger.info("Validating configuration...")
        
        if self.testnet_mode and not self.paper_trading:
            logger.error("❌ TESTNET_MODE=True but PAPER_TRADING=false")
            logger.error("   Either set TESTNET_MODE=False for real trading")
            logger.error("   Or set PAPER_TRADING=true for testnet")
            return False
        
        if not self.live_mode and not self.paper_trading:
            logger.error("❌ LIVE_MODE=False and PAPER_TRADING=false")
            logger.error("   Must choose either live or paper mode")
            return False
        
        if not (0.0 < self.capital_floor_pct < 1.0):
            logger.error("❌ CAPITAL_FLOOR_PCT must be between 0 and 1")
            return False
        
        if not (0.0 <= self.quote_reserve_ratio < 1.0):
            logger.error("❌ QUOTE_RESERVE_RATIO must be between 0 and <1")
            return False
        
        mode_str = "🔴 LIVE TRADING" if self.live_mode else (
            "📝 PAPER TRADING" if self.paper_trading else "❓ UNKNOWN"
        )
        logger.info(f"✅ Mode: {mode_str}")
        logger.info(f"✅ Duration: {self.duration_hours} hours")
        logger.info(f"✅ Polling interval: {self.polling_interval_sec}s")
        logger.info(
            "✅ Capital reserve policy: keep %.1f%% in %s (min reserve $%.2f)",
            self.capital_floor_pct * 100.0,
            self.quote_asset,
            self.quote_min_reserve,
        )
        logger.info(f"✅ Spendable reserve ratio: {self.quote_reserve_ratio * 100.0:.1f}%")
        logger.info(f"✅ Signal injector: {'ENABLED' if self.enable_signal_injector else 'DISABLED'}")
        logger.info(f"✅ Volatility regime: {'ENABLED' if self.enable_volatility_regime else 'DISABLED'}")
        logger.info(f"✅ Compounding engine: {'ENABLED' if self.enable_compounding_engine else 'DISABLED'}")
        logger.info(f"✅ Alert system: {'ENABLED' if self.enable_alert_system else 'DISABLED'}")
        logger.info(
            "✅ Exchange truth auditor: %s (%s)",
            "ENABLED" if self.enable_exchange_truth_auditor else "DISABLED",
            self.truth_auditor_mode,
        )
        logger.info(
            "✅ Position merger enhanced: %s (interval=%.1fs)",
            "ENABLED" if self.enable_position_merger_enhanced else "DISABLED",
            self.position_merger_interval_sec,
        )
        logger.info(
            "✅ Rebalancing engine: %s (loop=%.1fs strategy=%s)",
            "ENABLED" if self.enable_rebalancing_engine else "DISABLED",
            self.rebalancing_engine_loop_sec,
            self.rebalancing_engine_strategy,
        )
        logger.info(
            "✅ Universe rotation engine: %s (interval=%.1fs)",
            "ENABLED" if self.enable_universe_rotation else "DISABLED",
            self.uure_interval_sec,
        )
        logger.info(
            "✅ Strict profit-only sells: %s",
            "ENABLED" if self.STRICT_PROFIT_ONLY_SELLS else "DISABLED",
        )
        
        return True

# ============================================================================
# MASTER ORCHESTRATOR
# ============================================================================

class MasterSystemOrchestrator:
    """
    Unified entry point for complete trading system.
    
    Initialization Sequence:
    1. Verify prerequisites
    2. Initialize Layer 1: ExchangeClient
    3. Initialize Layer 2: SharedState
    4. Initialize Layer 3B: SignalManager, RiskManager
    5. Initialize Layer 5: ExecutionManager
    6. Initialize Layer 3B: MetaController (decisions)
    7. Initialize Layer 3A: AgentManager (strategies)
    8. Initialize synchronization: PollingCoordinator
    9. Initialize monitoring: HealthMonitor, Watchdog, Heartbeat
    10. Start all components concurrently
    """
    
    def __init__(self):
        self.config = OrchestratorConfig()
        self.logger = logging.getLogger(__name__)
        
        # Components (initialized in order)
        self.exchange_client: Optional[ExchangeClient] = None
        self.shared_state: Optional[SharedState] = None
        self.signal_manager: Optional[SignalManager] = None
        self.risk_manager: Optional[RiskManager] = None
        self.execution_manager: Optional[ExecutionManager] = None
        self.tp_sl_engine: Optional[TPSLEngine] = None  # ← CRITICAL: TP/SL monitoring for exits
        self.meta_controller: Optional[MetaController] = None
        self.agent_manager: Optional[AgentManager] = None
        self.market_data_feed: Optional[MarketDataFeed] = None  # ← CRITICAL: Real-time market data
        self.polling_coordinator: Optional[PollingCoordinator] = None
        self.health_monitor: Optional[HealthMonitor] = None
        self.watchdog: Optional[Watchdog] = None
        self.heartbeat: Optional[Heartbeat] = None
        
        # OPTIONAL COMPONENTS (Enhanced Features)
        self.recovery_engine: Optional[object] = None
        self.bootstrap_manager: Optional[object] = None
        self.performance_monitor: Optional[object] = None
        self.profit_target_engine: Optional[object] = None
        self.capital_allocator: Optional[object] = None
        self.symbol_manager: Optional[object] = None
        self.pnl_calculator: Optional[object] = None
        self.performance_evaluator: Optional[object] = None
        self.portfolio_balancer: Optional[object] = None
        self.liquidation_orchestrator: Optional[object] = None
        self.volatility_regime: Optional[object] = None
        self.compounding_engine: Optional[object] = None
        self.alert_system: Optional[object] = None
        self.exchange_truth_auditor: Optional[object] = None
        self.position_merger_enhanced: Optional[object] = None
        self.rebalancing_engine: Optional[object] = None
        self.universe_rotation_engine: Optional[object] = None
        self.three_bucket_manager: Optional[ThreeBucketPortfolioManager] = None
        self.segmentation_manager: Optional[object] = None
        self.startup_orchestrator: Optional[object] = None
        self.cash_router: Optional[object] = None
        self.external_adoption_engine: Optional[object] = None
        self.dead_capital_healer: Optional[object] = None
        self._mdf_warmup_task: Optional[asyncio.Task] = None

        # EXTENSION COMPONENTS (all previously un-wired modules)
        self.market_regime_detector: Optional[object] = None
        self.regime_mediator: Optional[object] = None
        self.discovery_coordinator: Optional[object] = None
        self.capital_symbol_governor: Optional[object] = None
        self.reserve_manager: Optional[object] = None
        self.order_cache_manager: Optional[object] = None
        self.action_router: Optional[object] = None
        self.exit_arbitrator: Optional[object] = None
        self.balance_sync_coordinator: Optional[object] = None
        self.retry_manager: Optional[object] = None
        self.position_manager: Optional[object] = None
        self.portfolio_manager: Optional[object] = None
        self.state_synchronizer: Optional[object] = None
        self.state_sync_task: Optional[object] = None
        self.layer_orchestrator: Optional[object] = None
        self.ws_market_data: Optional[object] = None
        self.prometheus_metrics: Optional[object] = None
        self.health_check_manager: Optional[object] = None
        self.apm_instrument: Optional[object] = None
        self.chaos_monkey: Optional[object] = None
        self.replay_engine: Optional[object] = None
        
        # Tracking
        self.start_time: Optional[datetime] = None
        self.running = False
        self.tasks: list = []
        
    # ========================================================================
    # PREREQUISITE CHECKS
    # ========================================================================
    
    def check_prerequisites(self) -> bool:
        """Check all prerequisites before initializing system"""
        logger.info("=" * 80)
        logger.info("PHASE 0: PREREQUISITE CHECKS")
        logger.info("=" * 80)
        
        checks_passed = 0
        checks_total = 0
        
        # Check 1: Configuration valid
        checks_total += 1
        if not self.config.validate():
            logger.error("❌ Configuration validation failed")
            return False
        checks_passed += 1
        
        # Check 2: Live trading approval (if live mode)
        checks_total += 1
        if self.config.live_mode:
            approval = os.getenv("APPROVE_LIVE_TRADING", "").upper()
            if approval != "YES":
                logger.error("❌ Live trading not approved")
                logger.error("   Set: export APPROVE_LIVE_TRADING=YES")
                return False
            logger.info("✅ Live trading approval confirmed")
            checks_passed += 1
        else:
            logger.info("✅ Paper trading mode - no approval needed")
            checks_passed += 1
        
        # Check 3: API keys configured
        checks_total += 1
        api_key = os.getenv("BINANCE_API_KEY")
        api_secret_hmac = os.getenv("BINANCE_API_SECRET_HMAC")
        api_secret_ed25519 = os.getenv("BINANCE_API_SECRET_ED25519")
        
        if not api_key:
            logger.error("❌ BINANCE_API_KEY not configured")
            return False
        if not (api_secret_hmac or api_secret_ed25519):
            logger.error("❌ No BINANCE_API_SECRET configured")
            logger.error("   Set either BINANCE_API_SECRET_HMAC or BINANCE_API_SECRET_ED25519")
            return False
        logger.info("✅ API keys configured")
        checks_passed += 1
        
        # Check 4: Directory structure
        checks_total += 1
        required_dirs = ["core", "agents", "utils", "config", "logs"]
        for dir_name in required_dirs:
            if not (project_root / dir_name).exists():
                logger.error(f"❌ Required directory missing: {dir_name}")
                return False
        logger.info("✅ Directory structure valid")
        checks_passed += 1
        
        logger.info(f"\n✅ Prerequisite checks: {checks_passed}/{checks_total} PASSED\n")
        return True
    
    # ========================================================================
    # COMPONENT INITIALIZATION
    # ========================================================================
    
    async def initialize_components(self) -> bool:
        """Initialize all components in dependency order"""
        try:
            logger.info("=" * 80)
            logger.info("PHASE 1: COMPONENT INITIALIZATION")
            logger.info("=" * 80)
            
            # LAYER 1: Exchange Integration
            logger.info("\n[1/9] LAYER 1: Exchange Integration (ExchangeClient)")
            self.exchange_client = ExchangeClient()
            await self.exchange_client.start()  # ← CRITICAL: Start the session
            balance = await self.exchange_client.get_spot_balances()
            usdt_balance = balance.get('USDT', {}).get('free', 0)
            logger.info(f"✅ ExchangeClient initialized")
            logger.info(f"   Real balance: ${usdt_balance:.2f} USDT")

            # LAYER 1.5: CONFIG VALIDATION & CONSTANTS
            if HAS_CONFIG_VALIDATOR:
                try:
                    validate_config_on_startup(self.config)
                    logger.info("✅ Config validated (ConfigValidator)")
                except Exception as _cv_e:
                    logger.warning("⚠️  ConfigValidator: %s (non-fatal)", _cv_e)

            if HAS_CONFIG_CONSTANTS:
                try:
                    _consts = get_all_constants()
                    logger.info("✅ ConfigConstants loaded: %d constant groups", len(_consts))
                except Exception as _cc_e:
                    logger.warning("⚠️  ConfigConstants: %s", _cc_e)

            # LAYER 1.6: RETRY MANAGER (wraps exchange calls with smart backoff)
            if HAS_RETRY_MANAGER:
                try:
                    self.retry_manager = RetryManager()
                    logger.info("✅ RetryManager initialized")
                except Exception as _rm_e:
                    logger.warning("⚠️  RetryManager: %s", _rm_e)

            # LAYER 1.7: CHAOS MONKEY (resilience testing — only if explicitly enabled)
            if HAS_CHAOS_MONKEY:
                try:
                    _chaos_enabled = bool(getattr(self.config, "CHAOS_MONKEY_ENABLED", False))
                    self.chaos_monkey = ChaosMonkey(
                        enabled=_chaos_enabled,
                        injection_rate=float(getattr(self.config, "CHAOS_MONKEY_RATE", 0.01)),
                    )
                    if _chaos_enabled:
                        logger.warning("⚠️  ChaosMonkey ACTIVE (injection_rate=%.2f) — resilience testing mode", float(getattr(self.config, "CHAOS_MONKEY_RATE", 0.01)))
                    else:
                        logger.info("✅ ChaosMonkey initialized (disabled — enable with CHAOS_MONKEY_ENABLED=true)")
                except Exception as _cm_e:
                    logger.warning("⚠️  ChaosMonkey: %s", _cm_e)

            # LAYER 2: Shared State
            logger.info("\n[2/9] LAYER 2: Shared State (Central Hub)")
            self.shared_state = SharedState(
                config=self.config,
                exchange_client=self.exchange_client
            )
            # Enforce reserve policy in canonical spendable-balance path.
            self.shared_state.config.quote_asset = self.config.quote_asset
            self.shared_state.config.quote_reserve_ratio = self.config.quote_reserve_ratio
            self.shared_state.config.quote_min_reserve = self.config.quote_min_reserve
            self.shared_state.balances["USDT"] = {"free": usdt_balance, "locked": 0}
            logger.info(f"✅ SharedState initialized")
            logger.info(f"   Portfolio synced from exchange")

            # LAYER 2.1: BALANCE SYNC BACKOFF (wraps authoritative balance calls)
            if HAS_BALANCE_SYNC_BACKOFF:
                try:
                    self.balance_sync_coordinator = BalanceSyncCoordinator(
                        logger=logger,
                        component_name="MASTER",
                    )
                    logger.info("✅ BalanceSyncCoordinator initialized")
                except Exception as _bsb_e:
                    logger.warning("⚠️  BalanceSyncCoordinator: %s", _bsb_e)

            # LAYER 2.2: PROMETHEUS METRICS (observability)
            if HAS_PROMETHEUS_EXPORTER:
                try:
                    self.prometheus_metrics = SafetyGuardMetrics()
                    logger.info("✅ PrometheusExporter (SafetyGuardMetrics) initialized")
                except Exception as _prom_e:
                    logger.warning("⚠️  PrometheusExporter: %s", _prom_e)

            # LAYER 2.3: APM INSTRUMENTATION (distributed tracing)
            if HAS_APM_INSTRUMENT:
                try:
                    self.apm_instrument = get_apm_instrument()
                    logger.info("✅ APMInstrument initialized (tracing=%s)", self.apm_instrument.enabled if self.apm_instrument else False)
                except Exception as _apm_e:
                    logger.warning("⚠️  APMInstrument: %s", _apm_e)

            # BOOTSTRAP: Seed default symbols if none present
            logger.info("\n[BOOTSTRAP] Seeding default symbols...")
            try:
                from core.bootstrap_symbols import bootstrap_default_symbols
                await bootstrap_default_symbols(self.shared_state, logger)
                logger.info("✅ Symbol bootstrap complete")
            except Exception as e:
                logger.warning("⚠️ Symbol bootstrap failed (will retry during agent startup): %s", e, exc_info=True)

            # ────────────────────────────────────────────────────────────────
            # LAYER 2.5: STATE HYDRATION — StartupOrchestrator
            # Must run BEFORE MetaController starts so positions are visible.
            # ────────────────────────────────────────────────────────────────
            logger.info("\n[2.5/9] LAYER 2.5: State Hydration (StartupOrchestrator)")
            if HAS_STARTUP_ORCHESTRATOR and HAS_RECOVERY_ENGINE:
                try:
                    # Pre-instantiate RecoveryEngine here (it is also added to
                    # the optional section below for its background monitoring loop).
                    _se_recovery = RecoveryEngine(
                        config=self.config,
                        shared_state=self.shared_state,
                        exchange_client=self.exchange_client,
                        database_manager=None,
                        sstools=None,
                        pnl_calculator=None,
                        logger=logger,
                    )
                    # Pre-instantiate ExchangeTruthAuditor if enabled
                    _se_auditor = None
                    if self.config.enable_exchange_truth_auditor and HAS_EXCHANGE_TRUTH_AUDITOR:
                        try:
                            _se_auditor = ExchangeTruthAuditor(
                                config=self.config,
                                logger=logger,
                                shared_state=self.shared_state,
                                exchange_client=self.exchange_client,
                                mode="startup_only",
                            )
                        except Exception as _e:
                            logger.warning("⚠️  ExchangeTruthAuditor pre-init failed: %s", _e)

                    self.startup_orchestrator = StartupOrchestrator(
                        config=self.config,
                        shared_state=self.shared_state,
                        exchange_client=self.exchange_client,
                        recovery_engine=_se_recovery,
                        exchange_truth_auditor=_se_auditor,
                        logger=logger,
                    )
                    _hydration_ok = await self.startup_orchestrator.execute_startup_sequence()
                    if _hydration_ok:
                        logger.info("✅ StartupOrchestrator complete — positions hydrated, state ready")
                        # Store the pre-built recovery engine so optional section reuses it
                        self.recovery_engine = _se_recovery
                        if _se_auditor is not None:
                            self.exchange_truth_auditor = _se_auditor
                    else:
                        logger.warning(
                            "⚠️  StartupOrchestrator reported incomplete hydration — "
                            "system will continue from available state"
                        )
                        self.recovery_engine = _se_recovery
                except Exception as e:
                    logger.warning(
                        "⚠️  StartupOrchestrator failed (%s) — continuing with available state", e,
                        exc_info=True,
                    )
            else:
                logger.warning(
                    "⚠️  StartupOrchestrator or RecoveryEngine not available — "
                    "state hydration skipped (positions may be empty on restart)"
                )

            # Restore rejection counters from previous session (deadlock-relief continuity)
            self._load_rejection_counters()

            # ── Clear stale cross-session rejection counters so old counts (with
            # missing timestamps that default to now) don't trigger repeated_failures=True
            # → PROTECTIVE mode on the very first MetaController tick.
            try:
                if hasattr(self.shared_state, "rejection_counters") and self.shared_state.rejection_counters:
                    cleared = len(self.shared_state.rejection_counters)
                    self.shared_state.rejection_counters.clear()
                    if hasattr(self.shared_state, "rejection_timestamps"):
                        self.shared_state.rejection_timestamps.clear()
                    logger.info(
                        "✅ [RejCounters] Cleared %d stale cross-session rejection counter(s) to prevent "
                        "false PROTECTIVE mode trigger at startup",
                        cleared
                    )
            except Exception as _rce:
                logger.warning("⚠️  [RejCounters] Could not clear rejection counters: %s", _rce)

            # ── Pre-seed dust blacklist so permanently unsellable positions are
            # skipped from cycle 1, without needing a failed SELL attempt first.
            try:
                import json as _json
                _dust_bl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snapshots", "dust_blacklist.json")
                if os.path.exists(_dust_bl_path):
                    with open(_dust_bl_path, "r") as _f:
                        _dust_bl = _json.load(_f)
                    if _dust_bl and isinstance(_dust_bl, dict):
                        if not hasattr(self.shared_state, "dust_unhealable") or self.shared_state.dust_unhealable is None:
                            self.shared_state.dust_unhealable = {}
                        self.shared_state.dust_unhealable.update(_dust_bl)
                        logger.info(
                            "✅ [DustBlacklist] Pre-seeded %d symbol(s) as UNHEALABLE_LT_MIN_NOTIONAL: %s",
                            len(_dust_bl), list(_dust_bl.keys())
                        )
            except Exception as _dbl_err:
                logger.warning("⚠️  [DustBlacklist] Failed to load dust_blacklist.json: %s", _dbl_err)

            # ── Allow entries below significant floor so dust-deduction on existing
            # micro-positions never blocks a valid BUY (overrides config default).
            try:
                self.shared_state.allow_entry_below_significant_floor = True
                logger.info("✅ [EntryFloor] allow_entry_below_significant_floor=True set on shared_state")
            except Exception as _efe:
                logger.warning("⚠️  [EntryFloor] Could not set allow_entry_below_significant_floor: %s", _efe)

            # ── Lower the capital floor ratio from 20% to 10% so $22+ free USDT
            # is enough to proceed with $15 trades.  calculate_capital_floor()
            # reads from shared_state.dynamic_config first, so this takes effect
            # immediately without a restart or .env change.
            try:
                if not hasattr(self.shared_state, "dynamic_config") or self.shared_state.dynamic_config is None:
                    self.shared_state.dynamic_config = {}
                self.shared_state.dynamic_config["CAPITAL_FLOOR_PCT"] = 0.10
                logger.info("✅ [CapFloor] CAPITAL_FLOOR_PCT=0.10 injected into dynamic_config (was 0.20)")
            except Exception as _cfe:
                logger.warning("⚠️  [CapFloor] Could not set CAPITAL_FLOOR_PCT: %s", _cfe)

            # ── Patch DEADLOCK_REJECTION_IGNORE_REASONS directly on the config object.
            # The config.pyc on this filesystem may pre-date our .py edits; using direct
            # attribute injection guarantees the new reasons reach MetaController regardless
            # of whether the .pyc or .py was loaded.  This is the single authoritative place.
            try:
                _ignore_reasons = (
                    "COLD_BOOTSTRAP_BLOCK,PORTFOLIO_FULL,EXPECTED_MOVE_LT_ROUND_TRIP_COST,"
                    "POSITION_ALREADY_OPEN,CONF_BELOW_REQUIRED,NET_USDT_BELOW_THRESHOLD,"
                    "PRETRADE_EFFECT_GATE:NET_USDT_BELOW_THRESHOLD,"
                    "MICRO_BACKTEST_WIN_RATE_BELOW_THRESHOLD,"
                    "PRETRADE_EFFECT_GATE:MICRO_BACKTEST_WIN_RATE_BELOW_THRESHOLD,"
                    "MICRO_BACKTEST_INSUFFICIENT_SAMPLES,"
                    # Normal capacity/policy gates — not system failures:
                    "ENTRY_POLICY_GATE,"
                    # Sell blocked by edge gate on losing position — expected during recovery:
                    "CLOSE_NOT_SUBMITTED"
                )
                self.config.DEADLOCK_REJECTION_IGNORE_REASONS = _ignore_reasons
                logger.info(
                    "✅ [DeadlockIgnore] DEADLOCK_REJECTION_IGNORE_REASONS patched on config "
                    "(added ENTRY_POLICY_GATE, CLOSE_NOT_SUBMITTED)"
                )
            except Exception as _dire:
                logger.warning("⚠️  [DeadlockIgnore] Could not patch ignore reasons: %s", _dire)

            # ── Patch additional config attributes that may be missing from stale .pyc
            try:
                # Force startup in RECOVERY so we don't inherit PROTECTIVE from a previous crash.
                # config.pyc may not have STARTUP_MODE_OVERRIDE — inject it directly.
                if not getattr(self.config, "STARTUP_MODE_OVERRIDE", None):
                    self.config.STARTUP_MODE_OVERRIDE = "RECOVERY"
                    logger.info("✅ [StartupMode] STARTUP_MODE_OVERRIDE=RECOVERY injected into config")
                # Raise the PROTECTIVE→RECOVERY drawdown threshold so accounts with historical
                # realized losses can still exit PROTECTIVE mode.
                if not getattr(self.config, "PROTECTIVE_DD_STABLE_THRESHOLD", None):
                    self.config.PROTECTIVE_DD_STABLE_THRESHOLD = 50.0
                    logger.info("✅ [StartupMode] PROTECTIVE_DD_STABLE_THRESHOLD=50.0 injected into config")
            except Exception as _sme:
                logger.warning("⚠️  [StartupMode] Could not patch startup mode attrs: %s", _sme)

            # LAYER 3C: Market Data Feed — must warm up BEFORE MetaController starts
            logger.info("\n[2.8/9] LAYER 2.8: Market Data Feed (early warm-up)")
            self.market_data_feed = MarketDataFeed(
                config=self.config,
                logger=logger,
                exchange_client=self.exchange_client,
                shared_state=self.shared_state,
                symbol_manager=None,
            )
            # Start market data streaming immediately so OHLCV cache populates
            # before MetaController's first decision cycle.
            # The task reference is stored on self so run_system() adds it to the monitor loop.
            self._mdf_warmup_task = asyncio.create_task(
                self.market_data_feed.run(), name="MarketDataFeed"
            )
            logger.info("✅ MarketDataFeed started (warming up OHLCV cache...)")
            await asyncio.sleep(6)  # 6s warm-up — allows initial bars to arrive
            logger.info("✅ MarketDataFeed warm-up complete")

            # LAYER 2.9: WEBSOCKET MARKET DATA (real-time price + kline streams)
            if HAS_WS_MARKET_DATA:
                try:
                    _ws_enabled = bool(getattr(self.config, "ENABLE_WS_MARKET_DATA", True))
                    if _ws_enabled:
                        self.ws_market_data = WebSocketMarketData(
                            shared_state=self.shared_state,
                            exchange_client=self.exchange_client,
                            ohlcv_timeframes=["1m", "5m", "1h"],
                            logger=logger,
                        )
                        asyncio.create_task(self.ws_market_data.start(), name="WebSocketMarketData")
                        logger.info("✅ WebSocketMarketData started (real-time streams: 1m/5m/1h)")
                    else:
                        logger.info("ℹ️  WebSocketMarketData disabled (ENABLE_WS_MARKET_DATA=false)")
                except Exception as _ws_e:
                    logger.warning("⚠️  WebSocketMarketData: %s (REST polling continues)", _ws_e)

            # LAYER 2.95: MARKET REGIME DETECTOR (ADX/RSI/ATR regime classification)
            if HAS_MARKET_REGIME_DETECTOR:
                try:
                    self.market_regime_detector = MarketRegimeDetector(
                        config=self.config,
                        logger=logger,
                    )
                    logger.info("✅ MarketRegimeDetector initialized (ADX/RSI/ATR regime detection)")
                except Exception as _mrd_e:
                    logger.warning("⚠️  MarketRegimeDetector: %s", _mrd_e)

            # LAYER 3B: Signal & Risk Management
            logger.info("\n[3/9] LAYER 3B: Signal Manager")
            self.signal_manager = SignalManager(
                config=self.config,
                logger=logger,
                shared_state=self.shared_state
            )
            logger.info(f"✅ SignalManager initialized")
            
            logger.info("\n[4/9] LAYER 4: Risk Manager")
            self.risk_manager = RiskManager(
                shared_state=self.shared_state,
                config=self.config,
                logger=logger
            )
            logger.info(f"✅ RiskManager initialized")
            
            # LAYER 5: Execution Manager
            logger.info("\n[5/9] LAYER 5: Execution Manager")
            self.execution_manager = ExecutionManager(
                shared_state=self.shared_state,
                exchange_client=self.exchange_client,
                config=self.config
            )
            logger.info(f"✅ ExecutionManager initialized")

            # LAYER 5.1: ORDER CACHE MANAGER (local order tracking, reduces API polling)
            if HAS_ORDER_CACHE_MANAGER:
                try:
                    self.order_cache_manager = OrderCacheManager(
                        shared_state=self.shared_state,
                        exchange_client=self.exchange_client,
                        config={"reconciliation_interval_sec": 30.0, "stale_order_timeout_sec": 300.0},
                        logger_=logger,
                    )
                    asyncio.create_task(self.order_cache_manager.start(), name="OrderCacheManager")
                    logger.info("✅ OrderCacheManager started (local order cache + reconciliation)")
                except Exception as _ocm_e:
                    logger.warning("⚠️  OrderCacheManager: %s", _ocm_e)

            # LAYER 5.2: ACTION ROUTER (conflict resolution for trade intents)
            if HAS_ACTION_ROUTER:
                try:
                    self.action_router = ActionRouter()
                    logger.info("✅ ActionRouter initialized (intent conflict resolution active)")
                except Exception as _ar_e:
                    logger.warning("⚠️  ActionRouter: %s", _ar_e)

            # LAYER 5.3: EXIT ARBITRATOR (deterministic exit priority: RISK > TP/SL > SIGNAL > ROTATION)
            if HAS_EXIT_ARBITRATOR:
                try:
                    self.exit_arbitrator = get_arbitrator()
                    logger.info("✅ ExitArbitrator initialized (priority: RISK>TP_SL>SIGNAL>ROTATION>REBALANCE)")
                except Exception as _ea_e:
                    logger.warning("⚠️  ExitArbitrator: %s", _ea_e)

            # LAYER 5.4: POSITION MANAGER (periodic exchange reconciliation)
            if HAS_POSITION_MANAGER:
                try:
                    self.position_manager = PositionManager(
                        config=self.config,
                        shared_state=self.shared_state,
                        exchange_client=self.exchange_client,
                    )
                    asyncio.create_task(self.position_manager.start(), name="PositionManager")
                    logger.info("✅ PositionManager started (exchange position sync every %ss)", getattr(self.config, "POSITION_SYNC_INTERVAL_S", 20))
                except Exception as _pm_e:
                    logger.warning("⚠️  PositionManager: %s", _pm_e)

            # LAYER 5.5: RESERVE MANAGER (capital reserve enforcement per regime)
            if HAS_RESERVE_MANAGER:
                try:
                    self.reserve_manager = ReserveManager(
                        config=self.config,
                        shared_state=self.shared_state,
                        logger=logger,
                    )
                    logger.info("✅ ReserveManager initialized (capital reserve protection active)")
                except Exception as _rsv_e:
                    logger.warning("⚠️  ReserveManager: %s", _rsv_e)

            # LAYER 5.6: PORTFOLIO MANAGER (treasury + dust detection)
            if HAS_PORTFOLIO_MANAGER:
                try:
                    self.portfolio_manager = PortfolioManager(
                        config=self.config,
                        shared_state=self.shared_state,
                        exchange_client=self.exchange_client,
                        database_manager=None,  # DB optional — skip persistence layer
                    )
                    logger.info("✅ PortfolioManager initialized (treasury + dust classification)")
                except Exception as _pfm_e:
                    logger.warning("⚠️  PortfolioManager: %s", _pfm_e)

            # LAYER 5B: TP/SL Engine (Position Management & Exit Signals)
            logger.info("\n[5B/9] LAYER 5B: TP/SL Engine (Exit Monitoring)")
            try:
                self.tp_sl_engine = TPSLEngine(
                    shared_state=self.shared_state,
                    config=self.config,
                    execution_manager=self.execution_manager,
                    logger=logger
                )
                logger.info(f"✅ TPSLEngine initialized")
            except Exception as e:
                logger.error(f"❌ TPSLEngine initialization failed: {e}", exc_info=True)
                logger.warning("⚠️  System will continue without TP/SL monitoring (HIGH RISK)")
                self.tp_sl_engine = None
            
            # LAYER 3B: Meta Controller (decisions)
            logger.info("\n[6/9] LAYER 3B: Meta Controller (Decision Engine)")
            
            # 🔥 CRITICAL FIX: Disable ProfitLock on first run (no realized PnL yet)
            # ProfitLock prevents trading when there are no profits to compound
            # This is appropriate for experienced systems with P&L history,
            # but blocks execution on bootstrap. Will be enabled after first profitable trade.
            setattr(self.config, "PROFIT_LOCK_REENTRY_ENABLED", False)
            logger.info("[6/9] 🔥 DISABLED ProfitLock (will enable after first profit)")
            
            self.meta_controller = MetaController(
                shared_state=self.shared_state,
                execution_manager=self.execution_manager,
                exchange_client=self.exchange_client,
                config=self.config,
                signal_manager=self.signal_manager  # CRITICAL: Share the same SignalManager so injected signals are visible!
            )
            logger.info(f"✅ MetaController initialized")

            # LAYER 6.1: REGIME-AWARE MEDIATOR (wires MarketRegimeDetector → MetaController/AgentManager/CapitalAllocator)
            if HAS_MARKET_REGIME_DETECTOR and HAS_REGIME_INTEGRATION and self.market_regime_detector is not None:
                try:
                    self.regime_mediator = RegimeAwareMediator(
                        config=self.config,
                        market_regime_detector=self.market_regime_detector,
                        meta_controller=self.meta_controller,
                        agent_manager=None,         # wired below after AgentManager init
                        capital_allocator=None,     # wired below after CapitalAllocator init
                        execution_manager=self.execution_manager,
                        logger=logger,
                    )
                    asyncio.create_task(self.regime_mediator.start(), name="RegimeAwareMediator")
                    logger.info("✅ RegimeAwareMediator started (regime→agent weights→capital allocation)")
                except Exception as _rmed_e:
                    logger.warning("⚠️  RegimeAwareMediator: %s", _rmed_e)

            # LAYER 6.2: CAPITAL SYMBOL GOVERNOR (dynamic symbol cap from capital/drawdown rules)
            if HAS_CAPITAL_SYMBOL_GOVERNOR:
                try:
                    self.capital_symbol_governor = CapitalSymbolGovernor(
                        shared_state=self.shared_state,
                        config=self.config,
                        logger=logger,
                    )
                    logger.info("✅ CapitalSymbolGovernor initialized (dynamic symbol cap based on equity)")
                except Exception as _csg_e:
                    logger.warning("⚠️  CapitalSymbolGovernor: %s", _csg_e)

            # LAYER 6.3: EXIT ARBITRATOR — wire into MetaController if supported
            if HAS_EXIT_ARBITRATOR and self.exit_arbitrator is not None:
                try:
                    if hasattr(self.meta_controller, "set_exit_arbitrator"):
                        self.meta_controller.set_exit_arbitrator(self.exit_arbitrator)
                        logger.info("✅ ExitArbitrator wired into MetaController")
                    else:
                        logger.info("✅ ExitArbitrator available (MetaController will use it via shared reference)")
                        setattr(self.meta_controller, "_exit_arbitrator", self.exit_arbitrator)
                except Exception as _eaw_e:
                    logger.warning("⚠️  ExitArbitrator wiring: %s", _eaw_e)

            # ── ExternalAdoptionEngine — adopt / manage pre-existing wallet holdings ──
            if HAS_EXTERNAL_ADOPTION_ENGINE:
                try:
                    _eae_cfg = {
                        "EXTERNAL_DUST_THRESHOLD_USDT": 10.0,
                        "EXTERNAL_ADOPTION_MIN_USDT": 5.0,   # low threshold — adopt small positions
                        "EXTERNAL_EXPOSURE_LIMIT_PCT": 0.40,
                        "EXTERNAL_HEDGE_SPEED_PCT": 5.0,
                    }
                    self.external_adoption_engine = ExternalAdoptionEngine(
                        shared_state=self.shared_state,
                        risk_manager=self.risk_manager,
                        execution_manager=self.execution_manager,
                        config=_eae_cfg,
                    )
                    self.meta_controller.set_external_adoption_engine(self.external_adoption_engine)
                    logger.info("✅ ExternalAdoptionEngine wired — existing holdings will be managed")
                except Exception as _eae_e:
                    logger.warning("⚠️  ExternalAdoptionEngine failed: %s", _eae_e)

            # THREE-BUCKET PORTFOLIO MANAGEMENT
            logger.info("\n[6.5/9] THREE-BUCKET PORTFOLIO MANAGEMENT")
            if HAS_THREE_BUCKET_MANAGER:
                try:
                    self.three_bucket_manager = ThreeBucketPortfolioManager(config=self.config)
                    logger.info("✅ ThreeBucketPortfolioManager initialized")
                except Exception as e:
                    logger.warning(f"⚠️  ThreeBucketPortfolioManager failed: {e}")
                    self.three_bucket_manager = None
            else:
                logger.warning("⚠️  ThreeBucketPortfolioManager import failed")

            # ── DeadCapitalHealer — liquidates dust/stale/orphaned positions ──
            if HAS_DEAD_CAPITAL_HEALER:
                try:
                    # Estimate initial equity from current USDT balance for threshold selection
                    _usdt_now = float(
                        (self.shared_state.balances or {}).get("USDT", {}).get("free", 50.0)
                        if isinstance((self.shared_state.balances or {}).get("USDT"), dict)
                        else (self.shared_state.balances or {}).get("USDT", 50.0)
                    )
                    self.dead_capital_healer = DeadCapitalHealer(config={
                        "total_equity": max(50.0, _usdt_now),
                        "batch_heal_enabled": True,
                        "max_liquidations": 5,
                    })
                    # Wire into ThreeBucketManager if available
                    if self.three_bucket_manager is not None and hasattr(
                        self.three_bucket_manager, "set_dead_capital_healer"
                    ):
                        self.three_bucket_manager.set_dead_capital_healer(self.dead_capital_healer)
                        logger.info("✅ DeadCapitalHealer wired into ThreeBucketManager")
                    else:
                        logger.info("✅ DeadCapitalHealer initialized (standalone)")
                except Exception as _dch_e:
                    logger.warning("⚠️  DeadCapitalHealer failed: %s", _dch_e)
            
            # CAPITAL RECYCLING: Portfolio Segmentation
            logger.info("\n[6.6/9] CAPITAL RECYCLING: Portfolio Segmentation")
            if HAS_PORTFOLIO_SEGMENTATION:
                try:
                    self.segmentation_manager = PortfolioSegmentationManager(
                        config=self.config,
                        shared_state=self.shared_state,
                        logger=logger
                    )
                    logger.info("✅ PortfolioSegmentationManager initialized")
                except Exception as e:
                    logger.warning(f"⚠️  PortfolioSegmentationManager failed: {e}")
                    self.segmentation_manager = None
            else:
                logger.warning("⚠️  PortfolioSegmentationManager import failed")
            
            # LAYER 3A: Agent Manager
            logger.info("\n[7/9] LAYER 3A: Agent Manager")
            self.agent_manager = AgentManager(
                shared_state=self.shared_state,
                exchange_client=self.exchange_client,
                execution_manager=self.execution_manager,
                market_data={},  # Will be populated during runtime
                config=self.config,
                meta_controller=self.meta_controller,
            )
            # Defensive rebind: keep AgentManager and MetaController explicitly linked.
            self.agent_manager.meta_controller = self.meta_controller
            logger.info(f"✅ AgentManager initialized")

            # SymbolManager must be initialized before discovery registration so
            # proposer agents (e.g., IPOChaser) receive a live symbol_manager.
            if HAS_SYMBOL_MANAGER and self.symbol_manager is None:
                try:
                    self.symbol_manager = SymbolManager(
                        exchange_client=self.exchange_client,
                        shared_state=self.shared_state,
                        config=self.config,
                        logger=logger
                    )
                    logger.info("✅ SymbolManager initialized (pre-wired for discovery)")
                except Exception as e:
                    logger.warning(f"⚠️  SymbolManager pre-wire failed (discovery quality reduced): {e}")
            
            # LAYER 3D: Discovery Agents (Proposer System)
            logger.info("\n[7.3/9] LAYER 3D: Discovery Agents (Proposer System)")
            try:
                from core.agent_registry import register_all_discovery_agents
                
                # Create app_context-like object for discovery agents
                class AppContext:
                    pass
                
                app_ctx = AppContext()
                app_ctx.shared_state = self.shared_state
                app_ctx.config = self.config
                app_ctx.exchange_client = self.exchange_client
                app_ctx.symbol_manager = self.symbol_manager
                app_ctx.execution_manager = self.execution_manager
                app_ctx.meta_controller = self.meta_controller
                app_ctx.tp_sl_engine = None
                
                # Register all discovery/proposer agents
                register_all_discovery_agents(self.agent_manager, app_ctx)
                
                discovery_agents = self.agent_manager.get_discovery_agents()
                logger.info(f"✅ Discovery Agents registered: {len(discovery_agents)} agents")
                for agent in discovery_agents:
                    agent_name = agent.name if hasattr(agent, 'name') else agent.__class__.__name__
                    logger.info(f"   • {agent_name}")
                    
            except Exception as e:
                logger.warning(f"⚠️  Discovery agents registration failed (non-fatal): {e}")
                logger.warning("   System will continue with strategy agents only")

            # LAYER 7.4: DISCOVERY COORDINATOR (deduplicates & rates discovery proposals)
            if HAS_DISCOVERY_COORDINATOR:
                try:
                    self.discovery_coordinator = DiscoveryCoordinator(
                        shared_state=self.shared_state,
                        config=self.config,
                        logger=logger,
                    )
                    logger.info("✅ DiscoveryCoordinator initialized (deduplication + quality gating for proposals)")
                except Exception as _dc_e:
                    logger.warning("⚠️  DiscoveryCoordinator: %s", _dc_e)

            # LAYER 7.5b: Wire regime mediator → agent_manager (now that agent_manager exists)
            if self.regime_mediator is not None:
                try:
                    self.regime_mediator.agent_manager = self.agent_manager
                    logger.info("✅ RegimeAwareMediator: AgentManager wired in")
                except Exception as _rmaw_e:
                    logger.warning("⚠️  RegimeAwareMediator agent_manager wire: %s", _rmaw_e)

            # LAYER 3C: Market Data Feed — already started at step [2.8/9]
            logger.info("\n[7.5/9] LAYER 3C: Market Data Feed (already running from step 2.8)")
            if self.market_data_feed is None:
                # Fallback: create if warm-up path failed
                self.market_data_feed = MarketDataFeed(
                    config=self.config,
                    logger=logger,
                    exchange_client=self.exchange_client,
                    shared_state=self.shared_state,
                    symbol_manager=None,
                )
                logger.warning("⚠️  MarketDataFeed fallback init (warm-up was skipped)")
            else:
                logger.info("✅ MarketDataFeed already running (warm-up complete)")
            
            # Synchronization: Polling Coordinator
            logger.info("\n[8/9] SYNCHRONIZATION: Polling Coordinator")
            polling_config = PollingConfig(
                balance_interval_sec=self.config.polling_interval_sec,
                open_orders_interval_sec=self.config.polling_interval_sec,
                position_interval_sec=self.config.polling_interval_sec,
                health_cadence_sec=5.0
            )
            self.polling_coordinator = PollingCoordinator(
                exchange_client=self.exchange_client,
                shared_state=self.shared_state,
                config=polling_config
            )
            logger.info(f"✅ PollingCoordinator initialized")
            logger.info(f"   Sync interval: {self.config.polling_interval_sec}s")

            # LAYER 8.1: STATE SYNCHRONIZER (reconciles SharedState vs component state)
            if HAS_STATE_SYNCHRONIZER:
                try:
                    self.state_synchronizer = StateSynchronizer(
                        shared_state=self.shared_state,
                        meta_controller=self.meta_controller,
                    )
                    self.state_sync_task = StateSyncronizationTask(
                        synchronizer=self.state_synchronizer,
                        interval_sec=float(getattr(self.config, "STATE_SYNC_INTERVAL_S", 60.0)),
                    )
                    asyncio.create_task(self.state_sync_task.start(), name="StateSynchronizer")
                    logger.info("✅ StateSynchronizer started (reconciles state every %ss)", getattr(self.config, "STATE_SYNC_INTERVAL_S", 60))
                except Exception as _ss_e:
                    logger.warning("⚠️  StateSynchronizer: %s", _ss_e)

            # LAYER 8.2: LAYER ORCHESTRATOR (Wallet → Portfolio → Strategy 3-layer cycle)
            if HAS_LAYER_ORCHESTRATOR:
                try:
                    self.layer_orchestrator = LayerOrchestrator(
                        shared_state=self.shared_state,
                        config=self.config,
                        wallet_scanner=getattr(self, "wallet_scanner_agent", None),
                        portfolio_manager=self.portfolio_manager,
                        strategy_executor=self.execution_manager,
                    )
                    asyncio.create_task(self.layer_orchestrator.start(), name="LayerOrchestrator")
                    logger.info("✅ LayerOrchestrator started (Wallet→Portfolio→Strategy cycle)")
                except Exception as _lo_e:
                    logger.warning("⚠️  LayerOrchestrator: %s", _lo_e)

            # LAYER 8.3: REPLAY ENGINE (deterministic event replay for analysis)
            if HAS_REPLAY_ENGINE:
                try:
                    _replay_enabled = bool(getattr(self.config, "REPLAY_ENGINE_ENABLED", False))
                    if _replay_enabled:
                        from core.event_store import get_event_store
                        _event_store = await get_event_store()
                        self.replay_engine = DeterministicReplayEngine(event_store=_event_store)
                        logger.info("✅ ReplayEngine initialized (deterministic event replay enabled)")
                    else:
                        logger.info("ℹ️  ReplayEngine available (disabled — enable with REPLAY_ENGINE_ENABLED=true)")
                except Exception as _re_e:
                    logger.warning("⚠️  ReplayEngine: %s", _re_e)

            # Monitoring
            logger.info("\n[9/9] MONITORING: Health & Watchdog")
            self.health_monitor = HealthMonitor()
            self.watchdog = Watchdog(shared_state=self.shared_state)
            self.heartbeat = Heartbeat(shared_state=self.shared_state)
            logger.info(f"✅ HealthMonitor initialized")
            logger.info(f"✅ Watchdog initialized")
            logger.info(f"✅ Heartbeat initialized")

            # LAYER 9.1: HEALTH CHECK MANAGER (structured health probes)
            if HAS_HEALTH_CHECK_MANAGER:
                try:
                    self.health_check_manager = HealthCheckManager()
                    logger.info("✅ HealthCheckManager initialized (structured component health probes)")
                except Exception as _hcm_e:
                    logger.warning("⚠️  HealthCheckManager: %s", _hcm_e)

            # LAYER 9.2: HEALTH ENDPOINTS (FastAPI /health /ready /live — activate via HEALTH_ENDPOINT_PORT)
            if HAS_HEALTH_ENDPOINTS:
                try:
                    _health_port = int(getattr(self.config, "HEALTH_ENDPOINT_PORT", 0))
                    if _health_port > 0:
                        import uvicorn
                        from fastapi import FastAPI as _FastAPI
                        _health_app = _FastAPI(title="OctiVault Health")
                        register_health_endpoints(_health_app)
                        asyncio.create_task(
                            asyncio.to_thread(uvicorn.run, _health_app, host="0.0.0.0", port=_health_port, log_level="warning"),
                            name="HealthEndpoints",
                        )
                        logger.info("✅ HealthEndpoints serving on port %d", _health_port)
                    else:
                        logger.info("ℹ️  HealthEndpoints available (set HEALTH_ENDPOINT_PORT=<port> to activate)")
                except Exception as _he_e:
                    logger.warning("⚠️  HealthEndpoints: %s", _he_e)

            # LAYER 9.3: DASHBOARD (FastAPI trading dashboard — activate via DASHBOARD_PORT)
            if HAS_DASHBOARD:
                try:
                    _dash_port = int(getattr(self.config, "DASHBOARD_PORT", 0))
                    if _dash_port > 0:
                        initialize_dashboard(
                            state=self.shared_state,
                            perf_monitor=getattr(self, "performance_monitor", None),
                            cap_alloc=getattr(self, "capital_allocator", None),
                            exec_mgr=self.execution_manager,
                        )
                        logger.info("✅ Dashboard initialized on port %d", _dash_port)
                    else:
                        logger.info("ℹ️  Dashboard available (set DASHBOARD_PORT=<port> to activate)")
                except Exception as _dash_e:
                    logger.warning("⚠️  Dashboard: %s", _dash_e)

            # LAYER 9.4: REPLAY ENGINE (deterministic event replay — activate via REPLAY_ENGINE_ENABLED=true)
            if HAS_REPLAY_ENGINE:
                try:
                    _replay_enabled = bool(getattr(self.config, "REPLAY_ENGINE_ENABLED", False))
                    if _replay_enabled:
                        from core.event_store import get_event_store as _get_es
                        _event_store = await _get_es()
                        self.replay_engine = DeterministicReplayEngine(event_store=_event_store)
                        logger.info("✅ ReplayEngine initialized (deterministic event replay active)")
                    else:
                        logger.info("ℹ️  ReplayEngine available (set REPLAY_ENGINE_ENABLED=true to activate)")
                except Exception as _re_e:
                    logger.warning("⚠️  ReplayEngine: %s", _re_e)

            # OPTIONAL COMPONENTS: Enhanced System Features
            logger.info("\n[OPTIONAL] ENHANCED FEATURES: Optional Components")
            
            # Symbol Manager (if available)
            if HAS_SYMBOL_MANAGER:
                if self.symbol_manager is None:
                    try:
                        self.symbol_manager = SymbolManager(
                            exchange_client=self.exchange_client,
                            shared_state=self.shared_state,
                            config=self.config,
                            logger=logger
                        )
                        logger.info("✅ SymbolManager initialized (optional)")
                    except Exception as e:
                        logger.warning(f"⚠️  SymbolManager failed (optional): {e}")
                else:
                    logger.info("ℹ️  SymbolManager already initialized (reused)")

            # Universe Rotation Engine (feature-flagged)
            if self.config.enable_universe_rotation:
                if HAS_UNIVERSE_ROTATION_ENGINE:
                    try:
                        self.universe_rotation_engine = UniverseRotationEngine(
                            shared_state=self.shared_state,
                            capital_governor=None,
                            config=self.config,
                            execution_manager=self.execution_manager,
                            meta_controller=self.meta_controller,
                            logger=logger,
                        )
                        if hasattr(self.universe_rotation_engine, "wire_runtime_dependencies"):
                            self.universe_rotation_engine.wire_runtime_dependencies(
                                capital_governor=None,
                                execution_manager=self.execution_manager,
                                meta_controller=self.meta_controller,
                            )
                        logger.info(
                            "✅ UniverseRotationEngine initialized (optional, interval=%.1fs)",
                            self.config.uure_interval_sec,
                        )
                    except Exception as e:
                        logger.warning(f"⚠️  UniverseRotationEngine failed (optional): {e}")
                else:
                    logger.warning(
                        "⚠️  ENABLE_UNIVERSE_ROTATION=true but core.universe_rotation_engine import failed"
                    )
            else:
                logger.info("ℹ️  UniverseRotationEngine disabled by feature flag")
            
            # Recovery Engine (if available)
            # RecoveryEngine requires: config, shared_state, exchange_client, database_manager, [optional: sstools, pnl_calculator, logger]
            if HAS_RECOVERY_ENGINE:
                try:
                    self.recovery_engine = RecoveryEngine(
                        config=self.config,
                        shared_state=self.shared_state,
                        exchange_client=self.exchange_client,
                        database_manager=None,  # Graceful fallback: RecoveryEngine will skip DB snapshot
                        sstools=None,
                        pnl_calculator=None,
                        logger=logger
                    )
                    logger.info("✅ RecoveryEngine initialized (optional)")
                except Exception as e:
                    logger.warning(f"⚠️  RecoveryEngine failed (optional): {e}")
            
            # Bootstrap Manager (if available)
            if HAS_BOOTSTRAP_MANAGER:
                try:
                    self.bootstrap_manager = BootstrapDustBypassManager()
                    logger.info("✅ BootstrapManager initialized (optional)")
                except Exception as e:
                    logger.warning(f"⚠️  BootstrapManager failed (optional): {e}")
            
            # Performance Monitor (if available)
            # PerformanceMonitor requires: cfg, shared_state, [optional: sstools, db, logger]
            if HAS_PERFORMANCE_MONITOR:
                try:
                    self.performance_monitor = PerformanceMonitor(
                        cfg=self.config,  # Note: parameter name is 'cfg' not 'config'
                        shared_state=self.shared_state,
                        sstools=None,
                        db=None,
                        logger=logger
                    )
                    logger.info("✅ PerformanceMonitor initialized (optional)")
                except Exception as e:
                    logger.warning(f"⚠️  PerformanceMonitor failed (optional): {e}")

            # Profit Target Engine (if available)
            if HAS_PROFIT_TARGET_ENGINE:
                try:
                    self.profit_target_engine = ProfitTargetEngine(
                        config=self.config,
                        logger=logger,
                        app=None,
                        shared_state=self.shared_state,
                    )
                    if hasattr(self.shared_state, "set_profit_guard") and hasattr(
                        self.profit_target_engine, "check_global_compliance"
                    ):
                        self.shared_state.set_profit_guard(
                            self.profit_target_engine.check_global_compliance
                        )
                        logger.info("✅ Profit guard wired to SharedState via ProfitTargetEngine")
                    logger.info("✅ ProfitTargetEngine initialized (optional)")
                except Exception as e:
                    logger.warning(f"⚠️  ProfitTargetEngine failed (optional): {e}")
            
            # Capital Allocator (if available)
            # CapitalAllocator requires: config, shared_state, risk_manager, [optional: sstools, logger, profit_target_engine, strategy_manager, agent_manager, liquidation_agent]
            if HAS_CAPITAL_ALLOCATOR:
                try:
                    self.capital_allocator = CapitalAllocator(
                        config=self.config,
                        shared_state=self.shared_state,
                        risk_manager=self.risk_manager,  # KEY: Now we pass risk_manager!
                        sstools=None,
                        logger=logger,
                        profit_target_engine=self.profit_target_engine,
                        strategy_manager=None,
                        agent_manager=self.agent_manager if hasattr(self, 'agent_manager') else None,
                        liquidation_agent=None
                    )
                    logger.info("✅ CapitalAllocator initialized (optional)")
                    # Wire CapitalAllocator into RegimeAwareMediator now that it exists
                    if self.regime_mediator is not None:
                        try:
                            self.regime_mediator.capital_allocator = self.capital_allocator
                            logger.info("✅ RegimeAwareMediator: CapitalAllocator wired in")
                        except Exception:
                            pass
                except Exception as e:
                    logger.warning(f"⚠️  CapitalAllocator failed (optional): {e}")

            # PnL Calculator (if available)
            if HAS_PNL_CALCULATOR:
                try:
                    self.pnl_calculator = PnLCalculator(
                        shared_state=self.shared_state,
                        config=self.config,
                        exchange_client=self.exchange_client,
                    )
                    logger.info("✅ PnLCalculator initialized (optional)")
                except Exception as e:
                    logger.warning(f"⚠️  PnLCalculator failed (optional): {e}")

            # Performance Evaluator (if available)
            if HAS_PERFORMANCE_EVALUATOR:
                try:
                    self.performance_evaluator = PerformanceEvaluator(
                        config=self.config,
                        shared_state=self.shared_state,
                        database_manager=None,
                        notification_manager=None,
                    )
                    logger.info("✅ PerformanceEvaluator initialized (optional)")
                except Exception as e:
                    logger.warning(f"⚠️  PerformanceEvaluator failed (optional): {e}")

            # Portfolio Balancer (if available)
            if HAS_PORTFOLIO_BALANCER:
                try:
                    self.portfolio_balancer = PortfolioBalancer(
                        shared_state=self.shared_state,
                        exchange_client=self.exchange_client,
                        execution_manager=self.execution_manager,
                        config=self.config,
                        meta_controller=self.meta_controller,
                    )
                    logger.info("✅ PortfolioBalancer initialized (optional)")
                except Exception as e:
                    logger.warning(f"⚠️  PortfolioBalancer failed (optional): {e}")

            # Liquidation Orchestrator (if available)
            if HAS_LIQUIDATION_ORCHESTRATOR:
                try:
                    liquidation_agent = None
                    if self.agent_manager and hasattr(self.agent_manager, "get_agent"):
                        liquidation_agent = self.agent_manager.get_agent("LiquidationAgent")
                        if liquidation_agent is None:
                            liquidation_agent = self.agent_manager.get_agent("liquidation_agent")

                    # ── Wire CashRouter so dust-sweep + stablecoin-redemption chain is live ──
                    if HAS_CASH_ROUTER:
                        try:
                            self.cash_router = CashRouter(
                                shared_state=self.shared_state,
                                exchange_client=self.exchange_client,
                                config=self.config,
                                execution_manager=self.execution_manager,
                            )
                            logger.info("✅ CashRouter initialized (dust sweep + stablecoin redemption active)")
                        except Exception as _cr_e:
                            logger.warning("⚠️  CashRouter init failed (%s) — LiquidationOrchestrator will run without it", _cr_e)
                    else:
                        logger.warning("⚠️  CashRouter not available — dust sweep chain disabled")

                    self.liquidation_orchestrator = LiquidationOrchestrator(
                        shared_state=self.shared_state,
                        liquidation_agent=liquidation_agent,
                        execution_manager=self.execution_manager,
                        cash_router=self.cash_router,          # ← now wired
                        meta_controller=self.meta_controller,
                        position_manager=None,
                        risk_manager=self.risk_manager,
                        min_usdt_target=float(self.config.quote_min_reserve),
                        min_usdt_floor=float(self.config.quote_min_reserve),
                    )

                    # Wire LiquidationOrchestrator back into MetaController so the
                    # capital-recovery task can call ensure_liquidity() directly.
                    if hasattr(self.meta_controller, "set_liquidation_orchestrator"):
                        self.meta_controller.set_liquidation_orchestrator(
                            self.liquidation_orchestrator
                        )
                        logger.info("✅ LiquidationOrchestrator wired into MetaController")
                    else:
                        # Attribute injection fallback
                        self.meta_controller._liquidation_orchestrator = self.liquidation_orchestrator
                        logger.info("✅ LiquidationOrchestrator injected into MetaController (attr)")

                    # Start async monitoring loop
                    if self.liquidation_orchestrator:
                        try:
                            asyncio.create_task(
                                self.liquidation_orchestrator._async_start(),
                                name="liquidation_orchestrator_main"
                            )
                            logger.info("✅ LiquidationOrchestrator started (async loop active)")
                        except Exception as e:
                            logger.warning(f"⚠️  LiquidationOrchestrator async start failed: {e}")
                except Exception as e:
                    logger.warning(f"⚠️  LiquidationOrchestrator failed (optional): {e}", exc_info=True)

            # Volatility Regime Detector (feature-flagged, if available)
            if self.config.enable_volatility_regime:
                if HAS_VOLATILITY_REGIME:
                    try:
                        accepted_symbols = await self.shared_state.get_accepted_symbols()
                        symbol_list = list((accepted_symbols or {}).keys())
                        self.volatility_regime = VolatilityRegimeDetector(
                            config=self.config,
                            logger=logger,
                            shared_state=self.shared_state,
                            symbols=symbol_list,
                        )
                        logger.info(
                            "✅ VolatilityRegimeDetector initialized (optional, symbols=%d)",
                            len(symbol_list),
                        )
                    except Exception as e:
                        logger.warning(f"⚠️  VolatilityRegimeDetector failed (optional): {e}")
                else:
                    logger.warning(
                        "⚠️  ENABLE_VOLATILITY_REGIME=true but core.volatility_regime import failed"
                    )
            else:
                logger.info("ℹ️  VolatilityRegimeDetector disabled by feature flag")

            # Compounding Engine (feature-flagged, if available)
            if self.config.enable_compounding_engine:
                if HAS_COMPOUNDING_ENGINE:
                    try:
                        self.compounding_engine = CompoundingEngine(
                            shared_state=self.shared_state,
                            exchange_client=self.exchange_client,
                            config=self.config,
                            execution_manager=self.execution_manager,
                        )
                        logger.info("✅ CompoundingEngine initialized (optional)")
                    except Exception as e:
                        logger.warning(f"⚠️  CompoundingEngine failed (optional): {e}")
                else:
                    logger.warning(
                        "⚠️  ENABLE_COMPOUNDING_ENGINE=true but core.compounding_engine import failed"
                    )
            else:
                logger.info("ℹ️  CompoundingEngine disabled by feature flag")

            # Alert System (feature-flagged, if available)
            if self.config.enable_alert_system:
                if HAS_ALERT_SYSTEM:
                    try:
                        self.alert_system = AlertSystem(
                            config=self.config,
                            logger=logger,
                        )
                        logger.info("✅ AlertSystem initialized (optional)")
                    except Exception as e:
                        logger.warning(f"⚠️  AlertSystem failed (optional): {e}")
                else:
                    logger.warning(
                        "⚠️  ENABLE_ALERT_SYSTEM=true but core.alert_system import failed"
                    )
            else:
                logger.info("ℹ️  AlertSystem disabled by feature flag")

            # Exchange Truth Auditor (feature-flagged, if available)
            if self.config.enable_exchange_truth_auditor:
                if HAS_EXCHANGE_TRUTH_AUDITOR:
                    try:
                        self.exchange_truth_auditor = ExchangeTruthAuditor(
                            config=self.config,
                            logger=logger,
                            shared_state=self.shared_state,
                            exchange_client=self.exchange_client,
                            mode=self.config.truth_auditor_mode,
                        )
                        if hasattr(self.exchange_truth_auditor, "set_execution_manager"):
                            self.exchange_truth_auditor.set_execution_manager(self.execution_manager)
                        logger.info(
                            "✅ ExchangeTruthAuditor initialized (optional, mode=%s)",
                            self.config.truth_auditor_mode,
                        )
                    except Exception as e:
                        logger.warning(f"⚠️  ExchangeTruthAuditor failed (optional): {e}")
                else:
                    logger.warning(
                        "⚠️  ENABLE_EXCHANGE_TRUTH_AUDITOR=true but core.exchange_truth_auditor import failed"
                    )
            else:
                logger.info("ℹ️  ExchangeTruthAuditor disabled by feature flag")

            # Position Merger Enhanced (feature-flagged, if available)
            if self.config.enable_position_merger_enhanced:
                if HAS_POSITION_MERGER_ENHANCED:
                    try:
                        self.position_merger_enhanced = PositionMergerEnhanced(
                            shared_state=self.shared_state,
                            exchange_client=self.exchange_client,
                            config=self.config,
                        )
                        logger.info("✅ PositionMergerEnhanced initialized (optional)")
                    except Exception as e:
                        logger.warning(f"⚠️  PositionMergerEnhanced failed (optional): {e}")
                else:
                    logger.warning(
                        "⚠️  ENABLE_POSITION_MERGER_ENHANCED=true but core.position_merger_enhanced import failed"
                    )
            else:
                logger.info("ℹ️  PositionMergerEnhanced disabled by feature flag")

            # Rebalancing Engine (feature-flagged, if available)
            if self.config.enable_rebalancing_engine:
                if HAS_REBALANCING_ENGINE:
                    try:
                        self.rebalancing_engine = RebalancingEngine(
                            shared_state=self.shared_state,
                            exchange_client=self.exchange_client,
                            config=self.config,
                            meta_controller=self.meta_controller,
                        )
                        logger.info("✅ RebalancingEngine initialized (optional)")
                    except Exception as e:
                        logger.warning(f"⚠️  RebalancingEngine failed (optional): {e}")
                else:
                    logger.warning(
                        "⚠️  ENABLE_REBALANCING_ENGINE=true but core.rebalancing_engine import failed"
                    )
            else:
                logger.info("ℹ️  RebalancingEngine disabled by feature flag")
            
            logger.info("\n" + "=" * 80)
            logger.info("✅ ALL COMPONENTS INITIALIZED SUCCESSFULLY")
            logger.info("=" * 80)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Component initialization failed: {e}", exc_info=True)
            return False

    def _launch_optional_component(
        self,
        component: Optional[object],
        name: str,
        lifecycle_methods: Optional[tuple] = None,
    ) -> bool:
        """
        Launch optional component using the first compatible lifecycle method.
        """
        if component is None:
            return False

        methods = lifecycle_methods or ("start", "run_forever", "run", "run_loop")
        for method_name in methods:
            method = getattr(component, method_name, None)
            if callable(method):
                try:
                    task_name = f"{name}:{method_name}"
                    result = method()
                    if inspect.isawaitable(result):
                        self.tasks.append(asyncio.create_task(result, name=task_name))
                        logger.info("✅ %s task added (%s)", name, method_name)
                    else:
                        logger.info("✅ %s launched via sync %s()", name, method_name)
                    return True
                except Exception as e:
                    logger.warning("⚠️  Could not start %s via %s: %s", name, method_name, e)
                    return False

        logger.info("ℹ️  %s has no runnable lifecycle method (start/run/run_forever/run_loop)", name)
        return False

    async def _stop_component(self, component: Optional[object], name: str) -> None:
        """
        Stop optional component using the first compatible shutdown method.
        """
        if component is None:
            return

        for method_name in ("stop", "shutdown", "close"):
            method = getattr(component, method_name, None)
            if callable(method):
                try:
                    logger.info("Shutting down %s via %s...", name, method_name)
                    maybe = method()
                    if asyncio.iscoroutine(maybe):
                        try:
                            await asyncio.wait_for(maybe, timeout=10.0)
                        except asyncio.TimeoutError:
                            logger.warning("⚠️  %s %s timed out after 10s; continuing shutdown", name, method_name)
                    logger.info("✅ %s shutdown", name)
                except Exception as e:
                    logger.warning("⚠️  %s %s failed: %s", name, method_name, e, exc_info=True)
                return

    async def _snapshot_positions_prices(self) -> tuple:
        """
        Best-effort snapshot used by optional analytics loops.
        Returns: (positions_by_symbol, prices_by_symbol)
        """
        positions = {}
        prices = {}
        if not self.shared_state:
            return positions, prices

        try:
            get_positions = getattr(self.shared_state, "get_positions_snapshot", None)
            if callable(get_positions):
                raw_positions = get_positions()
                if inspect.isawaitable(raw_positions):
                    raw_positions = await raw_positions
                if isinstance(raw_positions, dict):
                    for key, value in raw_positions.items():
                        if not isinstance(value, dict):
                            continue
                        symbol = str(value.get("symbol") or key).upper()
                        normalized = dict(value)
                        normalized["symbol"] = symbol
                        positions[symbol] = normalized
        except Exception as e:
            logger.debug("Optional loop position snapshot failed: %s", e, exc_info=True)

        try:
            get_prices = getattr(self.shared_state, "get_all_prices", None)
            if callable(get_prices):
                raw_prices = get_prices()
                if inspect.isawaitable(raw_prices):
                    raw_prices = await raw_prices
                if isinstance(raw_prices, dict):
                    prices = {str(k).upper(): float(v or 0.0) for k, v in raw_prices.items()}
            if not prices:
                raw_latest = getattr(self.shared_state, "latest_prices", {}) or {}
                if isinstance(raw_latest, dict):
                    prices = {str(k).upper(): float(v or 0.0) for k, v in raw_latest.items()}
        except Exception as e:
            logger.debug("Optional loop price snapshot failed: %s", e, exc_info=True)

        return positions, prices

    async def _run_position_merger_enhanced_loop(self) -> None:
        """
        Periodically run consolidation analysis for PositionMergerEnhanced.
        """
        interval = max(1.0, float(getattr(self.config, "position_merger_interval_sec", 300.0)))
        logger.info("🧩 PositionMergerEnhanced loop started (interval=%.1fs)", interval)
        while self.running and self.position_merger_enhanced is not None:
            try:
                positions, prices = await self._snapshot_positions_prices()
                if positions and prices:
                    report = await self.position_merger_enhanced.run_consolidation_cycle(positions, prices)
                    if getattr(report, "proposals_generated", 0) > 0 or getattr(report, "proposals_executed", 0) > 0:
                        logger.info(
                            "🧩 PositionMergerEnhanced cycle: proposals=%s executed=%s errors=%d",
                            getattr(report, "proposals_generated", 0),
                            getattr(report, "proposals_executed", 0),
                            len(getattr(report, "errors", []) or []),
                        )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("⚠️  PositionMergerEnhanced loop error: %s", e, exc_info=True)
            await asyncio.sleep(interval)
        logger.info("🧩 PositionMergerEnhanced loop stopped")

    async def _run_rebalancing_engine_loop(self) -> None:
        """
        Periodically run RebalancingEngine auto-rebalance checks.
        """
        interval = max(1.0, float(getattr(self.config, "rebalancing_engine_loop_sec", 300.0)))
        strategy_name = str(getattr(self.config, "rebalancing_engine_strategy", "DRIFT_THRESHOLD")).upper()
        logger.info("⚖️ RebalancingEngine loop started (interval=%.1fs strategy=%s)", interval, strategy_name)

        strategy = None
        if HAS_REBALANCING_ENGINE:
            strategy = getattr(RebalanceStrategy, strategy_name, RebalanceStrategy.DRIFT_THRESHOLD)

        while self.running and self.rebalancing_engine is not None:
            try:
                positions, prices = await self._snapshot_positions_prices()
                if not positions or not prices:
                    await asyncio.sleep(interval)
                    continue

                if not getattr(self.rebalancing_engine, "target_allocation", {}):
                    symbols = [s for s, p in positions.items() if float(p.get("quantity", 0.0)) > 0 and prices.get(s, 0.0) > 0]
                    if symbols:
                        eq_weight = 1.0 / len(symbols)
                        if HAS_REBALANCING_ENGINE:
                            targets = {
                                s: AllocationTarget(symbol=s, target_weight=eq_weight, priority=1)
                                for s in symbols
                            }
                            self.rebalancing_engine.set_target_allocation(targets)
                            logger.info("⚖️ RebalancingEngine seeded equal-weight targets for %d symbols", len(symbols))

                plan = await self.rebalancing_engine.auto_rebalance(
                    positions=positions,
                    prices=prices,
                    strategy=strategy,
                )
                if plan is not None:
                    logger.info(
                        "⚖️ RebalancingEngine cycle: strategy=%s orders=%d status=%s",
                        getattr(plan.strategy, "value", str(plan.strategy)),
                        len(getattr(plan, "rebalance_orders", []) or []),
                        getattr(plan.rebalance_status, "value", str(plan.rebalance_status)),
                    )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("⚠️  RebalancingEngine loop error: %s", e, exc_info=True)
            await asyncio.sleep(interval)
        logger.info("⚖️ RebalancingEngine loop stopped")

    async def _run_universe_rotation_loop(self) -> None:
        """
        Periodically execute UniverseRotationEngine to keep symbol selection dynamic.
        """
        interval = max(15.0, float(getattr(self.config, "uure_interval_sec", 300.0)))
        logger.info("🌐 UniverseRotationEngine loop started (interval=%.1fs)", interval)

        async def _run_once(stage: str) -> None:
            try:
                if self.universe_rotation_engine is None:
                    return
                result = self.universe_rotation_engine.compute_and_apply_universe()
                if inspect.isawaitable(result):
                    result = await result
                if isinstance(result, dict):
                    rotation = result.get("rotation", {}) or {}
                    logger.info(
                        "🌐 UURE %s cycle: added=%d removed=%d kept=%d universe=%d",
                        stage,
                        len(rotation.get("added", []) or []),
                        len(rotation.get("removed", []) or []),
                        len(rotation.get("kept", []) or []),
                        len(result.get("new_universe", []) or []),
                    )
                else:
                    logger.info("🌐 UURE %s cycle complete", stage)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("⚠️  UniverseRotationEngine %s cycle failed: %s", stage, e, exc_info=True)

        # Immediate cycle on startup so proposals are consumed quickly.
        await _run_once("startup")

        while self.running and self.universe_rotation_engine is not None:
            try:
                await asyncio.sleep(interval)
                await _run_once("periodic")
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("⚠️  UniverseRotationEngine loop error: %s", e, exc_info=True)

        logger.info("🌐 UniverseRotationEngine loop stopped")
    
    # ========================================================================
    # SYSTEM EXECUTION
    # ========================================================================
    
    async def run_system(self) -> bool:
        """Run complete trading system"""
        
        if not self.check_prerequisites():
            logger.error("❌ Prerequisite checks failed")
            return False
        
        if not await self.initialize_components():
            logger.error("❌ Component initialization failed")
            return False
        
        self.running = True
        self.start_time = datetime.now()
        
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 2: RUNNING TRADING SYSTEM")
        logger.info("=" * 80)
        logger.info(f"Started: {self.start_time}")
        logger.info(f"Duration: {self.config.duration_hours} hours")
        logger.info(f"Mode: {'🔴 LIVE TRADING' if self.config.live_mode else '📝 PAPER TRADING'}")
        logger.info("=" * 80 + "\n")
        
        try:
            # DEBUG: Pre-task creation state
            logger.warning("[DEBUG] About to create tasks. self.running=%s, signal_manager=%s, shared_state=%s", 
                          self.running, self.signal_manager is not None, self.shared_state is not None)
            
            # Create tasks for all components
            # NOTE: MarketDataFeed is already running from the [2.8/9] warm-up step.
            # Do NOT create a second task for it — that would spawn a duplicate stream.
            self.tasks = [
                asyncio.create_task(self.polling_coordinator.start(), name="PollingCoordinator"),
                # market_data_feed already running — task tracked in _mdf_task above
                asyncio.create_task(self.tp_sl_engine.start(), name="TPSLEngine") if self.tp_sl_engine else None,
                asyncio.create_task(self.meta_controller.start(), name="MetaController"),
                asyncio.create_task(self.agent_manager.start(), name="AgentManager"),
                asyncio.create_task(self.watchdog.run(), name="Watchdog"),
                asyncio.create_task(self.heartbeat.run(), name="Heartbeat"),
                asyncio.create_task(self._inject_signals_loop(), name="SignalInjector")
                if self.config.enable_signal_injector else None,
            ]
            # Add the MarketDataFeed warm-up task to the monitored set
            if hasattr(self, "_mdf_warmup_task") and self._mdf_warmup_task is not None:
                self.tasks.append(self._mdf_warmup_task)
                logger.info("✅ MarketDataFeed task added to monitor loop")

            if self.config.enable_signal_injector:
                logger.warning("[Orchestrator] SignalInjector ENABLED (test mode).")
            else:
                logger.info("[Orchestrator] SignalInjector disabled (production-safe).")
            
            # Add optional component tasks (supports start/run_forever/run/run_loop).
            optional_components = [
                ("RecoveryEngine", self.recovery_engine, None),
                ("BootstrapManager", self.bootstrap_manager, None),
                ("PerformanceMonitor", self.performance_monitor, None),
                ("ProfitTargetEngine", self.profit_target_engine, ("start",)),
                ("CapitalAllocator", self.capital_allocator, None),
                ("SymbolManager", self.symbol_manager, None),
                ("PnLCalculator", self.pnl_calculator, None),
                ("PerformanceEvaluator", self.performance_evaluator, None),
                ("PortfolioBalancer", self.portfolio_balancer, None),
                ("LiquidationOrchestrator", self.liquidation_orchestrator, None),
                # Prefer long-running loop methods so orchestrator owns cancellation directly.
                ("VolatilityRegimeDetector", self.volatility_regime, ("run", "run_loop", "run_forever", "start")),
                ("CompoundingEngine", self.compounding_engine, ("run_loop", "run", "run_forever", "start")),
                ("AlertSystem", self.alert_system, None),
                ("ExchangeTruthAuditor", self.exchange_truth_auditor, None),
                ("PositionMergerEnhanced", self.position_merger_enhanced, None),
                ("RebalancingEngine", self.rebalancing_engine, None),
                ("UniverseRotationEngine", self.universe_rotation_engine, None),
            ]
            for comp_name, comp, lifecycle_methods in optional_components:
                self._launch_optional_component(comp, comp_name, lifecycle_methods=lifecycle_methods)

            # Components without native run loops are orchestrated here.
            if self.position_merger_enhanced is not None:
                self.tasks.append(
                    asyncio.create_task(
                        self._run_position_merger_enhanced_loop(),
                        name="PositionMergerEnhanced:loop",
                    )
                )
                logger.info("✅ PositionMergerEnhanced task added (orchestrator loop)")
            if self.rebalancing_engine is not None:
                self.tasks.append(
                    asyncio.create_task(
                        self._run_rebalancing_engine_loop(),
                        name="RebalancingEngine:loop",
                    )
                )
                logger.info("✅ RebalancingEngine task added (orchestrator loop)")
            if self.universe_rotation_engine is not None:
                self.tasks.append(
                    asyncio.create_task(
                        self._run_universe_rotation_loop(),
                        name="UniverseRotationEngine:loop",
                    )
                )
                logger.info("✅ UniverseRotationEngine task added (orchestrator loop)")
            
            # Filter out None tasks
            self.tasks = [t for t in self.tasks if t is not None]
            
            # DEBUG: Post-task creation verification
            logger.warning("[DEBUG] Created %d tasks:", len(self.tasks))
            for t in self.tasks:
                logger.warning("[DEBUG]   - %s: done=%s, cancelled=%s", t.get_name(), t.done(), t.cancelled())
            
            # Calculate duration in seconds
            duration_seconds = self.config.duration_hours * 3600
            logger.info(f"⏱️  System will run for {self.config.duration_hours} hours ({duration_seconds}s)")
            logger.info("🟢 All components now running concurrently...")
            
            # Keep system running for full duration, allowing components to fail/restart without killing system
            try:
                start_time = asyncio.get_event_loop().time()
                elapsed = 0
                loop_count = 0
                logged_tasks = set()  # Track which tasks we've already logged
                
                while elapsed < duration_seconds:
                    loop_count += 1
                    remaining = duration_seconds - elapsed
                    check_timeout = min(5.0, remaining)  # Check every 5 seconds
                    
                    # Monitor tasks for failures - allows us to log but NOT exit
                    # Guard: if all tasks are already done asyncio.wait() returns
                    # instantly with no sleep → busy-loop that floods the log.
                    active_tasks = [t for t in self.tasks if not t.done()]
                    if not active_tasks:
                        # All tasks finished; just sleep and check elapsed time
                        await asyncio.sleep(min(5.0, remaining))
                        done, pending = set(), set()
                    else:
                        done, pending = await asyncio.wait(
                            active_tasks,
                            timeout=check_timeout,
                            return_when=asyncio.FIRST_COMPLETED
                        )

                    # Log any NEWLY completed tasks (only once per task)
                    for task in done:
                        if task not in logged_tasks:
                            logged_tasks.add(task)
                            try:
                                exc = task.exception()
                                if exc:
                                    logger.error(f"❌ Task {task.get_name()} failed with: {type(exc).__name__}: {exc}")
                            except (asyncio.CancelledError, asyncio.InvalidStateError):
                                pass

                    # Update elapsed time
                    elapsed = asyncio.get_event_loop().time() - start_time
                    hours_remaining = (duration_seconds - elapsed) / 3600

                    if loop_count % 12 == 0:  # Log every ~60 seconds (12 * 5s checks)
                        logger.info(f"⏱️  Running... ({elapsed/3600:.2f}h elapsed, {hours_remaining:.2f}h remaining) | Active tasks: {len(active_tasks)}")
                
                logger.info(f"\n✅ Trading session duration ({self.config.duration_hours}h) complete")
                logger.info("🛑 Initiating graceful shutdown...")
                
            except asyncio.TimeoutError:
                logger.info(f"\n✅ Trading session duration ({self.config.duration_hours}h) complete (timeout)")
                logger.info("🛑 Initiating graceful shutdown...")
            
            return True
            
        except KeyboardInterrupt:
            logger.info("\n🛑 SHUTDOWN REQUESTED (Ctrl+C)")
            return True
            
        except Exception as e:
            logger.error(f"❌ System error: {e}", exc_info=True)
            return False
            
        finally:
            await self.shutdown()
    
    # ========================================================================
    # SIGNAL INJECTION (for testing MetaController execution pipeline)
    # ========================================================================
    
    async def _inject_signals_loop(self):
        """
        Inject high-confidence test signals into the signal manager.
        This bypasses agents and tests if MetaController can execute.
        """
        import time
        import sys
        
        # IMMEDIATE DEBUG - this should be first log
        msg = "[SignalInjector] 🚀 Task started! signal_manager=%s, shared_state=%s, running=%s" % (
            self.signal_manager is not None, self.shared_state is not None, self.running)
        logger.warning(msg)
        print(msg, file=sys.stderr)
        
        # Wait for shared_state and signal_manager (but not too long)
        if not self.signal_manager or not self.shared_state:
            logger.error("[SignalInjector] ❌ Components not ready! signal_manager=%s, shared_state=%s", 
                          self.signal_manager, self.shared_state)
            return
        
        logger.warning("[SignalInjector] ✅ Components ready, starting injection loop")
        
        try:
            symbols = list((await self.shared_state.get_accepted_symbols()).keys()) if self.shared_state else []
            if not symbols:
                logger.warning("[SignalInjector] No symbols available for injection")
                return
            
            logger.warning("[SignalInjector] Got %d symbols: %s", len(symbols), symbols)
            
            cycle = 0
            injected_count = 0
            
            while self.running:
                cycle += 1
                now = time.time()
                
                # Inject BUY signal for each symbol
                for symbol in symbols:
                    signal = {
                        "action": "BUY",
                        "confidence": 0.75,
                        "reason": "Test signal (injected)",
                        "timestamp": now,
                        "quote": 10.0,
                    }
                    
                    if self.signal_manager.receive_signal("TEST_INJECTOR", symbol, signal):
                        injected_count += 1
                
                # ═══════════════════════════════════════════════════════════════
                # THREE-BUCKET PORTFOLIO MANAGEMENT
                # ═══════════════════════════════════════════════════════════════
                if self.three_bucket_manager and self.shared_state:
                    try:
                        # Get current portfolio state
                        positions = await self.shared_state.get_positions_snapshot()
                        total_equity = await self.shared_state.get_total_nav()
                        
                        # Update bucket classification
                        bucket_state = await self.three_bucket_manager.update_bucket_state(
                            positions=positions,
                            total_equity=total_equity
                        )
                        
                        # Check if healing should execute
                        if self.three_bucket_manager.should_execute_healing():
                            logger.warning("💀 Executing dead capital healing cycle...")
                            healing_result = await self.three_bucket_manager.execute_healing()
                            if healing_result:
                                logger.info(f"✅ Healing complete: {healing_result}")
                        
                        # Log status every 10 cycles
                        if cycle % 10 == 0:
                            self.three_bucket_manager.log_bucket_status()
                            self.three_bucket_manager.log_trading_gates()
                        
                        # =====================================================================
                        # CAPITAL RECYCLING: Analyze portfolio state
                        # =====================================================================
                        if self.segmentation_manager:
                            try:
                                # Get portfolio segmentation
                                segmentation = await self.segmentation_manager.segment_portfolio()
                                
                                # Log portfolio health (every 10 cycles)
                                if cycle % 10 == 0:
                                    logger.info(
                                        f"📊 PORTFOLIO SEGMENTATION\n"
                                        f"├─ Operating Cash: ${segmentation.operating_cash_usdt:.2f} "
                                        f"({segmentation.cash_ratio_pct:.1f}%)\n"
                                        f"├─ Productive: ${segmentation.productive_value:.2f} "
                                        f"({segmentation.productive_ratio_pct:.1f}%)\n"
                                        f"├─ Dead Capital: ${segmentation.dead_capital_value:.2f} "
                                        f"({segmentation.dead_ratio_pct:.1f}%)\n"
                                        f"├─ Health Score: {segmentation.portfolio_health_score:.0f}/100\n"
                                        f"├─ Fragmentation: {segmentation.fragmentation_index:.1f}/100\n"
                                        f"└─ Dust Ratio: {segmentation.dust_ratio_pct:.2f}%"
                                    )
                                    
                                    # Log immediate recycling actions
                                    if segmentation.immediate_actions:
                                        for action in segmentation.immediate_actions:
                                            logger.info(f"🔄 RECYCLING ACTION: {action}")
                                    
                                    # Log strategic recommendations
                                    if segmentation.strategic_recommendations:
                                        for rec in segmentation.strategic_recommendations:
                                            logger.info(f"💡 STRATEGY: {rec}")
                                    
                                    # Log capital efficiency metrics (every 20 cycles)
                                    if cycle % 20 == 0:
                                        logger.info(
                                            f"📈 CAPITAL EFFICIENCY\n"
                                            f"├─ Health Score: {segmentation.portfolio_health_score:.1f}/100\n"
                                            f"├─ Cash Ratio: {segmentation.cash_ratio_pct:.1f}%\n"
                                            f"├─ Productive Ratio: {segmentation.productive_ratio_pct:.1f}%\n"
                                            f"├─ Dead Ratio: {segmentation.dead_ratio_pct:.1f}%\n"
                                            f"├─ Fragmentation: {segmentation.fragmentation_index:.1f}/100\n"
                                            f"├─ Dust Ratio: {segmentation.dust_ratio_pct:.2f}%\n"
                                            f"└─ Position Count: {len(segmentation.holdings)}"
                                        )
                            
                            except Exception as e:
                                logger.debug(f"Portfolio segmentation analysis failed: {e}")
                        
                        # Get trading decision gates
                        gates = self.three_bucket_manager.get_trading_decision_gates()
                        
                        # CHECK: Can we trade new positions?
                        all_gates_pass = all(passed for passed, _ in gates.values())
                        if not all_gates_pass:
                            logger.warning("🚦 TRADING GATES FAILED - Skipping trade execution")
                            for gate_name, (passed, reason) in gates.items():
                                if not passed:
                                    logger.warning(f"   ❌ {gate_name}: {reason}")
                    
                    except Exception as e:
                        logger.warning(f"⚠️  Three-bucket update failed: {e}", exc_info=True)
                
                # Log every 1 cycle (every 2 seconds)
                all_signals = self.signal_manager.get_all_signals()
                logger.info(
                    "[SignalInjector] Cycle %d: Injected | Cache: %d signals | Total: %d injected",
                    cycle, len(all_signals), injected_count
                )
                
                # Wait before next cycle (inject every 2 seconds)
                await asyncio.sleep(2)
        
        except Exception as e:
            logger.error("[SignalInjector] Error: %s", e, exc_info=True)
        finally:
            logger.info(f"[SignalInjector] ✅ Ended after {cycle} cycles, {injected_count} signals injected")
    
    # ========================================================================
    # SHUTDOWN
    # ========================================================================
    
    # ──────────────────────────────────────────────────────────────────────────
    # Rejection counter persistence (survives restarts — prevents deadlock reset)
    # ──────────────────────────────────────────────────────────────────────────
    _REJECTION_PERSIST_FILE = Path(__file__).parent / "snapshots" / "rejection_counters.json"

    def _save_rejection_counters(self):
        """Persist rejection counters to disk so deadlock-relief state survives restart."""
        try:
            if self.shared_state is None:
                return
            counters = getattr(self.shared_state, "rejection_counters", {})
            if not counters:
                return
            self._REJECTION_PERSIST_FILE.parent.mkdir(parents=True, exist_ok=True)
            serialisable = {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in counters.items()}
            with open(self._REJECTION_PERSIST_FILE, "w") as fh:
                import json as _json
                _json.dump(serialisable, fh)
            logger.info("[Persist] Rejection counters saved (%d entries)", len(serialisable))
        except Exception as e:
            logger.debug("[Persist] Could not save rejection counters: %s", e)

    def _load_rejection_counters(self):
        """Reload rejection counters from disk into SharedState at boot."""
        try:
            if self.shared_state is None or not self._REJECTION_PERSIST_FILE.exists():
                return
            import json as _json
            with open(self._REJECTION_PERSIST_FILE) as fh:
                serialisable = _json.load(fh)
            counters = getattr(self.shared_state, "rejection_counters", None)
            if counters is None:
                return
            loaded = 0
            for composite_key, count in serialisable.items():
                parts = composite_key.split("|", 2)
                if len(parts) == 3:
                    key = (parts[0], parts[1], parts[2])
                    counters[key] = int(count)
                    loaded += 1
            logger.info("[Persist] Rejection counters restored (%d entries) — deadlock-relief state preserved", loaded)
        except Exception as e:
            logger.debug("[Persist] Could not load rejection counters: %s", e)

    async def shutdown(self):
        """Graceful shutdown of all components"""
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 3: GRACEFUL SHUTDOWN")
        logger.info("=" * 80)

        # Persist rejection counters before anything stops
        self._save_rejection_counters()

        self.running = False
        
        # Cancel all tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to finish
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        
        # Shutdown components
        try:
            await self._stop_component(self.tp_sl_engine, "TPSLEngine")
            await self._stop_component(self.market_data_feed, "MarketDataFeed")
            await self._stop_component(self.polling_coordinator, "PollingCoordinator")
            await self._stop_component(self.meta_controller, "MetaController")
            await self._stop_component(self.agent_manager, "AgentManager")
            await self._stop_component(self.watchdog, "Watchdog")
            await self._stop_component(self.heartbeat, "Heartbeat")

            # Optional components
            await self._stop_component(self.recovery_engine, "RecoveryEngine")
            await self._stop_component(self.bootstrap_manager, "BootstrapManager")
            await self._stop_component(self.performance_monitor, "PerformanceMonitor")
            await self._stop_component(self.profit_target_engine, "ProfitTargetEngine")
            await self._stop_component(self.capital_allocator, "CapitalAllocator")
            await self._stop_component(self.symbol_manager, "SymbolManager")
            await self._stop_component(self.pnl_calculator, "PnLCalculator")
            await self._stop_component(self.performance_evaluator, "PerformanceEvaluator")
            await self._stop_component(self.portfolio_balancer, "PortfolioBalancer")
            await self._stop_component(self.liquidation_orchestrator, "LiquidationOrchestrator")
            await self._stop_component(self.volatility_regime, "VolatilityRegimeDetector")
            await self._stop_component(self.compounding_engine, "CompoundingEngine")
            await self._stop_component(self.alert_system, "AlertSystem")
            await self._stop_component(self.exchange_truth_auditor, "ExchangeTruthAuditor")
            await self._stop_component(self.position_merger_enhanced, "PositionMergerEnhanced")
            await self._stop_component(self.rebalancing_engine, "RebalancingEngine")
            await self._stop_component(self.universe_rotation_engine, "UniverseRotationEngine")
            await self._stop_component(self.three_bucket_manager, "ThreeBucketPortfolioManager")
            await self._stop_component(self.segmentation_manager, "PortfolioSegmentationManager")
            await self._stop_component(self.exchange_client, "ExchangeClient")
            
            if self.execution_manager:
                logger.info("Closing all positions...")
                # TODO: Close all open positions
                logger.info("✅ All positions closed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        
        # Print final statistics
        if self.start_time:
            uptime = (datetime.now() - self.start_time).total_seconds() / 3600
            logger.info(f"\nUptime: {uptime:.2f} hours")
        
        logger.info("=" * 80)
        logger.info("✅ SYSTEM SHUTDOWN COMPLETE")
        logger.info("=" * 80)
    
    # ========================================================================
    # SIGNAL HANDLERS
    # ========================================================================
    
    def handle_signal(self, signum, frame):
        """Handle system signals (SIGTERM, SIGINT)"""
        logger.info(f"Received signal {signum}")
        if self.running:
            asyncio.create_task(self.shutdown())

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

async def main():
    """Main entry point"""
    
    # Register signal handlers
    orchestrator = MasterSystemOrchestrator()
    signal.signal(signal.SIGINT, orchestrator.handle_signal)
    signal.signal(signal.SIGTERM, orchestrator.handle_signal)
    
    # Run system
    success = await orchestrator.run_system()
    
    # Return appropriate exit code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
