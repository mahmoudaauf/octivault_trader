import asyncio
import logging

from core.config import Config
from core.shared_state import SharedState
from core.exchange_client import BinanceClient
from core.market_data_feed import MarketDataFeed
from core.execution_manager import ExecutionManager
from core.recovery_engine import RecoveryEngine
from core.tp_sl_engine import TPSLEngine
from core.meta_controller import MetaController

from agents.ml_forecaster import MLForecaster
from agents.rl_strategist import RLStrategist
from agents.dip_sniper import DipSniper
from agents.ipo_chaser import IPOChaser
from agents.trend_hunter import TrendHunter
from agents.news_reactor import NewsReactor
from agents.liquidation_agent import LiquidationAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ModularTestRunner")

SYMBOLS = ["BTCUSDT", "ETHUSDT"]
INTERVALS = ["5m"]

async def test_phase_1_load_config():
    logger.info("🔧 Phase 1: Load Config")
    config = Config()
    logger.info(f"✅ LIVE_MODE: {config.LIVE_MODE}")
    return config

async def test_phase_2_init_shared_state(config):
    logger.info("🔧 Phase 2: Initialize Shared State")
    shared_state = SharedState(config)
    logger.info(f"✅ SharedState created with empty balances: {shared_state.balances}")
    return shared_state

async def test_phase_3_init_exchange_client(config):
    logger.info("🔧 Phase 3: Initialize Exchange Client")
    api_key = config.BINANCE_API_KEY
    api_secret = config.BINANCE_API_SECRET

    if not api_key or not api_secret:
        raise ValueError("❌ Binance API credentials are missing in the environment or config.")

    exchange_client = BinanceClient(api_key, api_secret)
    await exchange_client.initialize()

    logger.info("✅ BinanceClient initialized.")
    return exchange_client

async def test_phase_4_market_data_feed(shared_state, config, exchange_client):
    logger.info("🔧 Phase 4: Start Market Data Feed")
    market_data_feed = MarketDataFeed(shared_state, config, exchange_client, intervals=INTERVALS)
    await market_data_feed.get_recent_ohlcv(SYMBOLS[0], interval="5m")

    async def limited_polling():
        await market_data_feed.start_polling(SYMBOLS, interval_seconds=3)

    try:
        await asyncio.wait_for(limited_polling(), timeout=10)
    except asyncio.TimeoutError:
        logger.info("✅ Market data feed polling ran for 10 seconds and timed out as expected.")

    return market_data_feed

async def test_phase_5_execution_manager(shared_state, config):
    logger.info("🔧 Phase 5: Initialize Execution Manager")
    tp_sl_engine = TPSLEngine(shared_state, config)
    execution_manager = ExecutionManager(shared_state, config, tp_sl_engine)
    logger.info("✅ ExecutionManager ready.")
    return execution_manager, tp_sl_engine

async def test_phase_6_agents_init(shared_state, market_data_feed, execution_manager, config, tp_sl_engine):
    logger.info("🔧 Phase 6: Initialize Agents")
    agents = {
        "MLForecaster": MLForecaster(shared_state, market_data_feed, config, tp_sl_engine),
        "RLStrategist": RLStrategist(shared_state, market_data_feed, execution_manager, config),
        "DipSniper": DipSniper(shared_state, market_data_feed, execution_manager, config),
        "IPOChaser": IPOChaser(shared_state, market_data_feed, execution_manager, config),
        "TrendHunter": TrendHunter(shared_state, market_data_feed, execution_manager, config),
        "NewsReactor": NewsReactor(shared_state, market_data_feed, config),
        "LiquidationAgent": LiquidationAgent(shared_state, market_data_feed, execution_manager, config, tp_sl_engine)
    }
    logger.info(f"✅ {len(agents)} agents initialized.")
    return agents

async def test_phase_7_run_agents(agents):
    logger.info("🔧 Phase 7: Run Agents (Single Pass)")
    tasks = []
    for name, agent in agents.items():
        for symbol in SYMBOLS:
            if hasattr(agent, 'run'):
                tasks.append(agent.run(symbol))
    await asyncio.gather(*tasks)
    logger.info("✅ Agents executed.")

async def test_phase_8_meta_controller(shared_state, config):
    logger.info("🔧 Phase 8: MetaController Evaluation")
    controller = MetaController(shared_state, config)
    await controller.evaluate_signals(SYMBOLS[0])
    logger.info("✅ MetaController signal evaluation complete.")
    return controller

async def test_phase_9_recovery_engine(shared_state):
    logger.info("🔧 Phase 9: Start Recovery Engine Monitor (Short Test)")

    async def mock_restart():
        logger.info("🚨 [Test] Restart Triggered")

    async def mock_alert():
        logger.info("📣 [Test] Alert Triggered")

    engine = RecoveryEngine(shared_state, mock_restart, mock_alert, check_interval=3, max_inactive_time=5)
    await asyncio.wait_for(engine.monitor_health(), timeout=8)

async def main():
    config = await test_phase_1_load_config()
    shared_state = await test_phase_2_init_shared_state(config)
    exchange_client = await test_phase_3_init_exchange_client(config)
    market_data_feed = await test_phase_4_market_data_feed(shared_state, config, exchange_client)
    execution_manager, tp_sl_engine = await test_phase_5_execution_manager(shared_state, config)
    agents = await test_phase_6_agents_init(shared_state, market_data_feed, execution_manager, config, tp_sl_engine)
    await test_phase_7_run_agents(agents)
    await test_phase_8_meta_controller(shared_state, config)
    await test_phase_9_recovery_engine(shared_state)

if __name__ == "__main__":
    asyncio.run(main())
