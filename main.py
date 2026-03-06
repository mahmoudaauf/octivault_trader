from dotenv import load_dotenv
import os
load_dotenv(os.path.abspath(os.path.join(os.getcwd(), '.env')), override=True)
import asyncio
import logging

from core.config import Config
from core.shared_state import SharedState
from core.exchange_client import ExchangeClient
from core.database_manager import DatabaseManager
from core.portfolio_manager import PortfolioManager
from core.market_data_feed import MarketDataFeed
from core.strategy_manager import StrategyManager
from core.agent_manager import AgentManager
from core.risk_manager import RiskManager
from core.capital_allocator import CapitalAllocator
from core.execution_manager import ExecutionManager
from core.tp_sl_engine import TPSLEngine
from core.meta_controller import MetaController
from core.performance_monitor import PerformanceMonitor
from core.position_manager import PositionManager
from core.notification_manager import NotificationManager
from core.recovery_engine import RecoveryEngine
from core.alert_system import AlertSystem
from core.watchdog import Watchdog
from core.heartbeat import Heartbeat
from core.compounding_engine import CompoundingEngine
from core.volatility_regime import VolatilityRegimeDetector
from agents.cot_assistant import CoTAssistant
from agents.symbol_screener import SymbolScreener
from utils.logging_setup import setup_logging
from datetime import datetime
from utils.pid_manager import PIDManager
from core.symbol_manager import SymbolManager

# ⚡ ARCHITECT REFINEMENT #3: Import UniverseRotationEngine for ranking cycle
from core.universe_rotation_engine import UniverseRotationEngine
from core.capital_governor import CapitalSymbolGovernor

# Import RetrainingEngine
from core.retraining_engine import RetrainingEngine
from agents.ml_forecaster import MLForecaster # Importing MLForecaster as suggested for retrainable agents

# Add Imports for PnLCalculator and PerformanceEvaluator
from core.pnl_calculator import PnLCalculator
from core.performance_evaluator import PerformanceEvaluator

# Add imports for new symbol proposing agents
from agents.ipo_chaser import IPOChaser
from agents.wallet_scanner_agent import WalletScannerAgent
from agents.symbol_screener import SymbolScreener as ScreenerAgent # ✅ Fixed import
from agents.symbol_discoverer_agent import SymbolDiscovererAgent # Added/Fixed import for SymbolDiscovererAgent

# === REGIME TRADING INTEGRATION (Week 2 Implementation) ===
from core.regime_trading_integration import (
    RegimeTradingAdapter,
    RegimeTradingConfig,
    create_regime_trading_adapter,
)
from live_trading_system_architecture import SymbolConfig

logger = logging.getLogger("core.app_context")

class AppContext:
    def __init__(self, config: Config):
        self.config = config
        self.database_manager = None
        self.shared_state = None
        self.exchange_client = None
        self.notification_manager = None
        self.execution_manager = None
        self.tp_sl_engine = None
        self.risk_manager = None
        self.market_data_feed = None
        self.strategy_manager = None
        self.capital_allocator = None
        self.performance_monitor = None
        self.position_manager = None
        self.cot_assistant = None
        self.meta_controller = None
        self.agent_manager = None
        self.recovery_engine = None
        self.alert_system = None
        self.watchdog = None
        self.heartbeat = None
        self.compounding_engine = None
        self.volatility_regime = None
        self.symbol_screener = None
        self.pid_manager = PIDManager("logs/octivault_trader.pid")
        self.active_tasks = {}
        # Initialize new agent placeholders (existing ones)
        self.wallet_scanner = None # Retained, will be WalletScanner
        self.symbol_discoverer = None # Remains SymbolDiscovererAgent
        # Add SymbolManager placeholder
        self.symbol_manager = None
        # Add self.retraining_engine to AppContext.__init__()
        self.retraining_engine = None
        # Declare PnLCalculator and PerformanceEvaluator in AppContext.__init__()
        self.pnl_calculator = None
        self.performance_evaluator = None
        # Declare model_manager
        self.model_manager = None
        # Declare new symbol proposing agents
        self.ipo_chaser = None
        self.screener_agent = None
        
        # ⚡ ARCHITECT REFINEMENT #3: Add universe rotation engine for cycle separation
        self.universe_rotation_engine = None
        self.capital_symbol_governor = None
        
        # === REGIME TRADING INTEGRATION ===
        self.regime_trading_adapter = None
        self.enable_regime_trading = os.getenv("ENABLE_REGIME_TRADING", "false").lower() == "true"
        self.regime_trading_paper_mode = os.getenv("PAPER_TRADING", "true").lower() == "true"


    async def __aenter__(self):
        await self.initialize_all()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.shutdown()

    async def initialize_all(self):
        if not self.pid_manager.acquire_lock():
            raise RuntimeError("Another instance is already running.")

        self.database_manager = DatabaseManager(self.config)
        await self.database_manager.connect()

        self.shared_state = SharedState(self.config, self.database_manager)
        # ✅ Initialize SharedState.symbols from config
        self.shared_state.symbols = {s: {} for s in self.config.SYMBOLS}
        await self.shared_state.initialize_from_database()

        class NoOpNotificationManager:
            async def send_alert(self, message: str, level: str = "INFO"):
                logger.debug(f"[NoOpAlert] [{level}] {message}")
            async def start_listener(self):
                logger.debug("Notification listener disabled.")
            async def close(self):
                logger.debug("Notification manager closed.")

        self.notification_manager = NoOpNotificationManager()
        alert_callback = self.notification_manager.send_alert

        # Load API keys from environment
        BINANCE_API_KEY = os.environ.get('BINANCE_API_KEY')
        BINANCE_API_SECRET = os.environ.get('BINANCE_API_SECRET')

        self.exchange_client = ExchangeClient(
            config=self.config,
            shared_state=self.shared_state,
            api_key=BINANCE_API_KEY,
            api_secret=BINANCE_API_SECRET
        )
        await self.exchange_client.start()
        self.shared_state.exchange_client = self.exchange_client
        
        # Initialize SymbolManager (already present and good)
        self.symbol_manager = SymbolManager(self.shared_state, self.config, self.exchange_client)
        # ✅ Fix: Retroactively inject symbol_manager into shared_state
        self.shared_state.symbol_manager = self.symbol_manager

        self.execution_manager = ExecutionManager(self.config, self.shared_state, self.exchange_client, alert_callback)
        self.tp_sl_engine = TPSLEngine(self.shared_state, self.config, self.execution_manager)
        self.execution_manager.tp_sl_engine = self.tp_sl_engine  # Direct attribute assignment

        self.portfolio_manager = PortfolioManager(self.config, self.shared_state, self.exchange_client, self.database_manager, self.notification_manager)
        self.market_data_feed = MarketDataFeed(self.shared_state, self.exchange_client, config=self.config)
        self.strategy_manager = StrategyManager(self.shared_state, self.config, self.database_manager)
        self.risk_manager = RiskManager(self.shared_state, self.config, self.execution_manager)
        self.capital_allocator = CapitalAllocator(self.config, self.shared_state, self.strategy_manager, self.risk_manager)
        self.position_manager = PositionManager(self.config, self.shared_state, self.exchange_client)
        self.performance_monitor = PerformanceMonitor(self.shared_state, self.config)
        self.cot_assistant = CoTAssistant(self.shared_state, self.market_data_feed, self.execution_manager, self.tp_sl_engine, self.config, alert_callback_func=alert_callback)
        self.meta_controller = MetaController(self.shared_state, self.exchange_client, self.execution_manager, self.config, self.cot_assistant, alert_callback)
        # Updated AgentManager instantiation with keyword arguments
        self.agent_manager = AgentManager(
            shared_state=self.shared_state,
            market_data=self.market_data_feed,
            execution_manager=self.execution_manager,
            config=self.config,
            symbols=getattr(self.config, 'SYMBOLS', []),
            tp_sl_engine=self.tp_sl_engine,
            market_data_feed=self.market_data_feed,
            model_manager=self.model_manager,
            meta_controller=self.meta_controller,
            symbol_manager=self.symbol_manager,
            exchange_client=self.exchange_client,
            agent_schedule=getattr(self.config, 'AGENT_SESSION_SCHEDULE', {}),
        )
        await self.agent_manager.auto_register_agents()

        # Initialize RetrainingEngine inside initialize_all()
        # Gather retrainable agents
        retrainable_agents = [
            agent for agent in self.agent_manager.agents.values()
            if hasattr(agent, "retrain")
        ]

        self.retraining_engine = RetrainingEngine(
            shared_state=self.shared_state,
            market_data_feed=self.market_data_feed,
            agents=retrainable_agents,
            retrain_interval_hours=24  # configurable
        )

        # Fix for SymbolScreener initialization
        logger.info(f"[DEBUG] Instantiating SymbolScreener with symbol_manager={self.symbol_manager}")
        self.symbol_screener = SymbolScreener(self.shared_state, self.exchange_client, self.config, self.symbol_manager)
        self.recovery_engine = RecoveryEngine(self.config, self.shared_state, self.exchange_client, self.database_manager)
        self.alert_system = AlertSystem(self.config, self.notification_manager)
        self.watchdog = Watchdog(shared_state=self.shared_state, config=self.config, check_interval_seconds=getattr(self.config, 'WATCHDOG_CHECK_INTERVAL_SECONDS', 60))
        self.heartbeat = Heartbeat(self.shared_state, self.config.HEARTBEAT_INTERVAL)
        # Fix for VolatilityRegimeDetector initialization, adding 'symbols' argument
        self.volatility_regime = VolatilityRegimeDetector(
            self.shared_state,
            self.config,
            symbols=self.shared_state.symbols # Added symbols argument
        )
        self.compounding_engine = CompoundingEngine(self.shared_state, self.exchange_client, self.config, self.execution_manager)

        # Initialize SymbolDiscovererAgent, passing symbol_manager directly (this remains)
        self.symbol_discoverer = SymbolDiscovererAgent(
            shared_state=self.shared_state,
            config=self.config,
            exchange_client=self.exchange_client, # Renamed binance_client to exchange_client
            symbol_manager=self.symbol_manager # Pass symbol_manager directly
        )

        await self.shared_state.ensure_latest_prices_coverage(price_fetcher=self.exchange_client.get_current_price)

        # Instantiate PnLCalculator and PerformanceEvaluator
        # PnL Calculator
        self.pnl_calculator = PnLCalculator(self.shared_state, self.config)
        await self.shared_state.register_component("PnLCalculator")
        await self.shared_state.update_component_status("PnLCalculator", "Operational")
        self.active_tasks["pnl_calculator"] = asyncio.create_task(self.pnl_calculator.run())

        # Performance Evaluator
        self.performance_evaluator = PerformanceEvaluator(self.config, self.shared_state, database_manager=self.database_manager, notification_manager=self.notification_manager)
        await self.shared_state.register_component("PerformanceEvaluator")
        await self.shared_state.update_component_status("PerformanceEvaluator", "Operational")
        await self.performance_evaluator.start()
        if hasattr(self.performance_evaluator, "_task"):
            self.active_tasks["performance_evaluator"] = self.performance_evaluator._task

        # 🔁 Symbol Proposers Initialization
        self.ipo_chaser = IPOChaser(
            shared_state=self.shared_state,
            config=self.config,
            exchange_client=self.exchange_client,
            symbol_manager=self.symbol_manager
        )

        self.wallet_scanner = WalletScannerAgent( # Retained WalletScannerAgent
            shared_state=self.shared_state,
            config=self.config,
            exchange_client=self.exchange_client,
            symbol_manager=self.symbol_manager
        )

        self.screener_agent = ScreenerAgent(
            shared_state=self.shared_state,
            config=self.config,
            exchange_client=self.exchange_client,
            symbol_manager=self.symbol_manager
        )

        # ⚡ ARCHITECT REFINEMENT #3: Initialize UniverseRotationEngine for ranking cycle
        try:
            self.capital_symbol_governor = CapitalSymbolGovernor(
                config=self.config,
                shared_state=self.shared_state
            )
            logger.info("✅ CapitalSymbolGovernor initialized")
        except Exception as e:
            logger.warning(f"⚠️ CapitalSymbolGovernor initialization failed: {e}. Ranking cycle may be limited.")
        
        try:
            self.universe_rotation_engine = UniverseRotationEngine(
                shared_state=self.shared_state,
                capital_governor=self.capital_symbol_governor,
                config=self.config
            )
            logger.info("✅ UniverseRotationEngine initialized for ranking cycle")
        except Exception as e:
            logger.error(f"❌ UniverseRotationEngine initialization failed: {e}", exc_info=True)
            self.universe_rotation_engine = None


    async def start_all(self):
        """
        Orchestrates the startup of all necessary components and background tasks.
        """
        # Ensure OHLCV is loaded before anything else starts, with a timeout
        try:
            await asyncio.wait_for(self.market_data_feed.poll_all_symbols_and_timeframes(), timeout=15)
        except asyncio.TimeoutError:
            logger.warning("⚠️ MarketDataFeed loading timeout. Proceeding with background tasks anyway.")
        
        # === REGIME TRADING INTEGRATION ===
        # Initialize regime trading adapter if enabled
        if self.enable_regime_trading:
            await self._initialize_regime_trading()
        
        await self.start_background_tasks()
        logger.info("All application components and background tasks started.")


    async def _initialize_regime_trading(self):
        """Initialize the regime trading adapter (new system)"""
        try:
            logger.info("Initializing regime trading system...")
            
            # Create symbol configurations for regime trading
            regime_symbols = os.getenv("REGIME_SYMBOLS", "ETHUSDT").split(",")
            symbol_configs = {}
            
            for symbol in regime_symbols:
                symbol_configs[symbol] = SymbolConfig(
                    symbol=symbol,
                    enabled=True,
                    base_exposure=float(os.getenv(f"{symbol}_BASE_EXPOSURE", "1.0")),
                    alpha_exposure=float(os.getenv(f"{symbol}_ALPHA_EXPOSURE", "2.0")),
                    max_position_size_pct=float(os.getenv("MAX_POSITION_SIZE_PCT", "0.05")),
                    max_drawdown_threshold=float(os.getenv("MAX_DRAWDOWN_THRESHOLD", "0.30")),
                    daily_loss_limit=float(os.getenv("DAILY_LOSS_LIMIT", "0.02")),
                )
            
            # Create configuration
            regime_config = RegimeTradingConfig(
                enabled=True,
                paper_trading=self.regime_trading_paper_mode,
                symbols=symbol_configs,
                sync_interval_seconds=float(os.getenv("SYNC_INTERVAL_SECONDS", "60")),
            )
            
            # Create and initialize adapter
            self.regime_trading_adapter = await create_regime_trading_adapter(
                shared_state=self.shared_state,
                execution_manager=self.execution_manager,
                market_data_feed=self.market_data_feed,
                config=regime_config,
            )
            
            if self.regime_trading_adapter:
                logger.info("✅ Regime trading system initialized")
            else:
                logger.warning("⚠️  Regime trading initialization incomplete")
                self.enable_regime_trading = False
        
        except Exception as e:
            logger.error(f"❌ Regime trading initialization failed: {e}", exc_info=True)
            self.enable_regime_trading = False


    async def start_background_tasks(self):
        self.active_tasks.update({
            'market_data_feed': asyncio.create_task(self.market_data_feed.run()),
            'strategy_manager': asyncio.create_task(self.strategy_manager.start_periodic_analysis()),
            'risk_manager': asyncio.create_task(self.risk_manager.run()),
            'position_manager': asyncio.create_task(self.position_manager.run()),
            'performance_monitor': asyncio.create_task(self.performance_monitor.run()),
            'recovery_engine': asyncio.create_task(self.recovery_engine.run(self.config.SYMBOLS)),
            'tp_sl_engine': asyncio.create_task(self.tp_sl_engine.run()),
            'meta_controller': asyncio.create_task(self.meta_controller.run()),
            'watchdog': asyncio.create_task(self.watchdog.run()),
            'heartbeat': asyncio.create_task(self.heartbeat.run()),
            'compounding_engine': asyncio.create_task(self.compounding_engine.run()),
            # Fix: Changed run_all() to run() as the standard for continuous background tasks
            'volatility_regime': asyncio.create_task(self.volatility_regime.run()),
            'symbol_screener': asyncio.create_task(self.symbol_screener.start_periodic_screening()),
            'execution_order_monitoring': asyncio.create_task(self.execution_manager.start_order_monitoring()),
            # ⚡ ARCHITECT REFINEMENT #3: Add cycle separation
            'discovery_cycle': asyncio.create_task(self._discovery_cycle()),
            'ranking_cycle': asyncio.create_task(self._ranking_cycle()),
            # Trading cycle already exists as meta_controller
            # 🔥 This is what starts the agents
            'agent_manager': asyncio.create_task(self.agent_manager.run_loop()),
            
            # Start the new agents as background tasks
            'wallet_scanner': asyncio.create_task(self.wallet_scanner.scheduler()), # Now refers to WalletScanner
            'symbol_discoverer': asyncio.create_task(self.symbol_discoverer.run_loop()),
            'ipo_chaser': asyncio.create_task(self.ipo_chaser.run_loop()), # New task
            # Changed screener_agent to call start_periodic_screening()
            'screener_agent': asyncio.create_task(self.screener_agent.start_periodic_screening()),
            # Conditional Retraining Engine task
        })
        if self.config.RETRAINING_ENABLED:
            self.active_tasks['retraining_engine'] = asyncio.create_task(self.retraining_engine.run())

    # ⚡ ARCHITECT REFINEMENT #3: Separate discovery, ranking, and trading cycles
    
    async def _discovery_cycle(self):
        """
        Discovery cycle runs every 5 minutes.
        Market research phase: Find new trading candidates.
        Independent from ranking and trading to allow different frequencies.
        """
        logger.info("🔍 Discovery cycle initialized (runs every 5 minutes)")
        while True:
            try:
                logger.info("🔍 Starting discovery cycle")
                # Run all discovery agents
                await self.agent_manager.run_loop()  # This runs IPO chaser, wallet scanner, screener
                logger.info("✅ Discovery cycle complete - symbols fed to validation")
            except Exception as e:
                logger.error(f"❌ Discovery cycle failed: {e}", exc_info=True)
            await asyncio.sleep(300)  # Every 5 minutes
    
    async def _ranking_cycle(self):
        """
        Ranking cycle runs every 5 minutes.
        Portfolio management phase: Rank discovered symbols and update active universe.
        Independent from discovery and trading to allow periodic portfolio rebalancing.
        """
        logger.info("📊 Ranking cycle initialized (runs every 5 minutes)")
        while True:
            try:
                logger.info("📊 Starting UURE ranking cycle")
                # Compute and apply universe with new 40/20/20/20 scoring
                if hasattr(self, 'universe_rotation_engine') and self.universe_rotation_engine:
                    await self.universe_rotation_engine.compute_and_apply_universe()
                    logger.info("✅ Ranking cycle complete - active universe updated")
                else:
                    logger.warning("⚠️ UURE not available, skipping ranking cycle")
            except Exception as e:
                logger.error(f"❌ Ranking cycle failed: {e}", exc_info=True)
            await asyncio.sleep(300)  # Every 5 minutes
    
    async def _trading_cycle(self):
        """
        Trading cycle runs every 10 seconds.
        Execution phase: Evaluate current universe and execute trades.
        Frequent independent cycle for responsive market opportunity capture.
        """
        logger.info("🏃 Trading cycle initialized (runs every 10 seconds)")
        while True:
            try:
                # MetaController evaluates once per cycle
                await self.meta_controller.evaluate_once()
            except Exception as e:
                logger.error(f"❌ Trading cycle failed: {e}", exc_info=True)
            await asyncio.sleep(10)  # Every 10 seconds

    async def shutdown(self):
        logger.info("Shutting down application context.")
        for task in self.active_tasks.values():
            task.cancel()
        await asyncio.gather(*self.active_tasks.values(), return_exceptions=True)

        if self.exchange_client:
            await self.exchange_client.close()
        if self.database_manager:
            await self.database_manager.close()
        if self.notification_manager:
            await self.notification_manager.close()

        if self.pid_manager and self.pid_manager.is_locked():
            self.pid_manager.remove_pid_file()

        logger.info("AppContext shutdown complete.")


async def main():
    # Set up logging configuration
    log_path = f"logs/run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.log"
    setup_logging(log_path)

    # Load configuration
    config = Config()

    # Initialize application context and start services
    async with AppContext(config) as app:
        await app.start_all() # Replaced the previous two lines with a single call to start_all()

        # Application is now running; press Ctrl+C to exit
        logger.info("🏃‍♂️ Application is now running. Press Ctrl+C to exit.")
        # Keep the application alive
        while True:
            await asyncio.sleep(3600)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("✋ Received exit signal. Shutting down...")
