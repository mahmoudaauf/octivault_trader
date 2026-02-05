import asyncio
import logging
from core.config import Config
from core.exchange_client import ExchangeClient
from core.shared_state import SharedState
from core.database_manager import DatabaseManager
from core.market_data_feed import MarketDataFeed
from core.symbol_manager import SymbolManager
from core.execution_manager import ExecutionManager
from core.tp_sl_engine import TPSLEngine
from core.risk_manager import RiskManager
from core.portfolio_manager import PortfolioManager
from core.position_manager import PositionManager
from core.agent_manager import AgentManager
from core.meta_controller import MetaController
from core.performance_monitor import PerformanceMonitor
from core.compounding_engine import CompoundingEngine
from core.heartbeat import Heartbeat
from core.watchdog import Watchdog
from core.alert_system import AlertSystem
from core.recovery_engine import RecoveryEngine
from core.model_manager import ModelManager
from agents.wallet_scanner_agent import WalletScannerAgent
from agents.ipo_chaser import IPOChaser
from agents.symbol_screener import SymbolScreener
from agents.dip_sniper import DipSniper
from agents.trend_hunter import TrendHunter
from agents.ml_forecaster import MLForecaster

logger = logging.getLogger("AppContext")
logging.basicConfig(level=logging.INFO)

class AppContext:
    def __init__(self):
        self.phase_logs = []

    async def initialize_all(self, up_to_phase=9):
        self.config = Config()
        self.database_manager = DatabaseManager(self.config)
        self.exchange_client = ExchangeClient(self.config)
        self.shared_state = SharedState(self.config, self.database_manager)
        logger.info("✅ Phase 1 Complete: Core initialized")

        if up_to_phase >= 2:
            self.symbol_manager = SymbolManager(self.shared_state, self.config)
            self.discovery_agents = [
                WalletScannerAgent(self.config, self.shared_state, self.exchange_client, self.symbol_manager),
                IPOChaser(self.config, self.shared_state, self.exchange_client, self.symbol_manager),
                SymbolScreener(self.config, self.shared_state, self.exchange_client, self.symbol_manager)
            ]
            logger.info("✅ Phase 2 Complete: Discovery agents initialized")

        if up_to_phase >= 3:
            for agent in self.discovery_agents:
                await agent.run_once()
            self.symbol_manager.accept_proposed_symbols()
            logger.info(f"✅ Phase 3 Complete: Symbols accepted: {list(self.shared_state.symbols.keys())}")

        if up_to_phase >= 4:
            self.market_data_feed = MarketDataFeed(self.shared_state, self.exchange_client, self.config, list(self.shared_state.symbols.keys()))
            await self.market_data_feed.warm_up_indicators()
            logger.info("✅ Phase 4 Complete: MarketDataFeed warmed up")

        if up_to_phase >= 5:
            self.tp_sl_engine = TPSLEngine(self.shared_state, self.exchange_client, self.config)
            self.execution_manager = ExecutionManager(self.shared_state, self.exchange_client, self.tp_sl_engine, self.config)
            self.risk_manager = RiskManager(self.shared_state, self.config)
            self.position_manager = PositionManager(self.shared_state, self.config)
            self.portfolio_manager = PortfolioManager(self.shared_state, self.config)
            logger.info("✅ Phase 5 Complete: Execution infrastructure ready")

        if up_to_phase >= 6:
            self.model_manager = ModelManager(self.config)
            self.agent_manager = AgentManager(
                shared_state=self.shared_state,
                market_data=self.market_data_feed,
                execution_manager=self.execution_manager,
                config=self.config,
                tp_sl_engine=self.tp_sl_engine,
                market_data_feed=self.market_data_feed,
                model_manager=self.model_manager,
                symbol_manager=self.symbol_manager,
                exchange_client=self.exchange_client,
                database_manager=self.database_manager
            )
            self.agent_manager.auto_register_agents([DipSniper, TrendHunter, MLForecaster])
            logger.info("✅ Phase 6 Complete: Strategy agents registered")

        if up_to_phase >= 7:
            self.meta_controller = MetaController(self.shared_state, self.config, self.execution_manager)
            self.recovery_engine = RecoveryEngine(self.shared_state, self.config)
            logger.info("✅ Phase 7 Complete: Meta control layer initialized")

        if up_to_phase >= 8:
            self.watchdog = Watchdog(self.shared_state, self.config)
            self.heartbeat = Heartbeat(self.shared_state, self.config)
            self.performance_monitor = PerformanceMonitor(self.shared_state, self.config)
            self.compounding_engine = CompoundingEngine(self.shared_state, self.exchange_client, self.config)
            self.alert_system = AlertSystem(self.shared_state, self.config)
            logger.info("✅ Phase 8 Complete: Monitoring components initialized")

        if up_to_phase == 9:
            await asyncio.gather(
                self.market_data_feed.run(),
                self.agent_manager.run(),
                self.heartbeat.run(),
                self.watchdog.run(),
                self.performance_monitor.run(),
                self.compounding_engine.run()
            )
            logger.info("✅ Phase 9 Complete: All live loops running")
