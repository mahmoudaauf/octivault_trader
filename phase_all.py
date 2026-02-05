import asyncio
import logging
from core.config import Config
from core.shared_state import SharedState
from core.exchange_client import ExchangeClient
from core.symbol_manager import SymbolManager
from core.market_data_feed import MarketDataFeed
from core.portfolio_manager import PortfolioManager
from core.tp_sl_engine import TPSLEngine
from core.execution_manager import ExecutionManager
from core.position_manager import PositionManager
from core.risk_manager import RiskManager
from core.performance_monitor import PerformanceMonitor
from core.compounding_engine import CompoundingEngine
from core.meta_controller import MetaController
from core.agent_manager import AgentManager
from core.recovery_engine import RecoveryEngine
from core.notification_manager import NotificationManager
from core.alert_system import AlertSystem
from core.heartbeat import Heartbeat
from core.watchdog import Watchdog
from agents.wallet_scanner_agent import WalletScannerAgent
from agents.symbol_screener import SymbolScreener
from agents.symbol_discoverer_agent import SymbolDiscovererAgent
from agents.ipo_chaser import IPOChaser
from agents.cot_assistant import CoTAssistant

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PhasedStartup")

async def main():
    config = Config()
    shared_state = SharedState(config=config)

    # Phase 1: Core Initialization
    logger.info("🚀 Phase 1: Initializing Core Modules...")
    exchange_client = ExchangeClient(config)
    await exchange_client.initialize()

    symbol_manager = SymbolManager(config, shared_state)
    market_data_feed = MarketDataFeed(config, shared_state, exchange_client)
    portfolio_manager = PortfolioManager(config, shared_state)
    tp_sl_engine = TPSLEngine(config, shared_state, exchange_client)
    execution_manager = ExecutionManager(config, shared_state, exchange_client)
    position_manager = PositionManager(config, shared_state, exchange_client, tp_sl_engine)
    risk_manager = RiskManager(config, shared_state)
    performance_monitor = PerformanceMonitor(config, shared_state)
    compounding_engine = CompoundingEngine(config, shared_state)
    meta_controller = MetaController(config, shared_state)
    recovery_engine = RecoveryEngine(config, shared_state)
    notification_manager = NotificationManager(config, shared_state, config.NOTIFY_ENDPOINT)
    alert_system = AlertSystem(config, shared_state)
    heartbeat = Heartbeat(config, shared_state)
    watchdog = Watchdog(config, shared_state)

    logger.info("✅ Phase 1 complete.\n")

    # Phase 2: AgentManager Initialization
    logger.info("🚀 Phase 2: Initializing AgentManager...")
    agent_manager = AgentManager(
        config=config,
        shared_state=shared_state,
        exchange_client=exchange_client,
        symbol_manager=symbol_manager,
    )
    logger.info("✅ Phase 2 complete.\n")

    # Phase 3: Initialize Symbol Discovery Agents
    logger.info("🚀 Phase 3: Initializing Symbol Discovery Agents...")
    wallet_scanner = WalletScannerAgent(shared_state, config, exchange_client, symbol_manager)
    screener = SymbolScreener(shared_state, config, exchange_client, symbol_manager)
    discoverer = SymbolDiscovererAgent(shared_state, config, exchange_client, symbol_manager)
    ipo_chaser = IPOChaser(shared_state, config, exchange_client, symbol_manager)
    logger.info("✅ Phase 3 complete.\n")

    # Phase 4: Run Symbol Discovery Once
    logger.info("🚀 Phase 4: Running Symbol Discovery Agents (one-shot)...")
    await wallet_scanner.run_once()
    await screener.run_once()
    await discoverer.run_once()
    await ipo_chaser.run_once()
    logger.info("✅ Phase 4 complete.\n")

    # Phase 5: Validate and Fallback to Default Symbols
    logger.info("🚀 Phase 5: Finalizing Symbol Set...")
    symbols = shared_state.get_symbols()
    if not symbols:
        logger.warning("⚠️ No symbols found, using fallback config.SYMBOLS.")
        for sym in config.SYMBOLS:
            shared_state.add_symbol(sym)
    logger.info(f"✅ Symbols now tracked: {shared_state.get_symbols()}\n")

    # Phase 6: Market Data Feed Launch
    logger.info("🚀 Phase 6: Starting Market Data Feed for selected symbols...")
    await market_data_feed.prefetch_initial_data()
    logger.info("✅ Phase 6 complete.\n")

    # Phase 7: Final System Check
    logger.info("✅ Phase 7: Running MetaController test...")
    cot_assistant = CoTAssistant(config, shared_state)
    await meta_controller.run_once()
    logger.info("🎯 System Initialization Phases Complete. Ready for Trading!")

if __name__ == "__main__":
    asyncio.run(main())
