import asyncio
import logging
import sys
import time
from core.config import Config
from core.shared_state import SharedState
from core.exchange_client import ExchangeClient
from core.market_data_feed import MarketDataFeed
from core.tp_sl_engine import TPSLEngine
from core.execution_manager import ExecutionManager
from core.meta_controller import MetaController
from core.capital_allocator import CapitalAllocator
from core.recovery_engine import RecoveryEngine
from stream.sentiment_stream import BinanceSentimentStream
from agents.ml_forecaster import MLForecaster
from agents.ipo_chaser import IPOChaser
from agents.dip_sniper import DipSniper
from agents.rl_strategist import RLStrategist
from agents.trend_hunter import TrendHunter
from agents.news_reactor import NewsReactor
from agents.liquidation_agent import LiquidationAgent
from utils.status_logger import log_component_status
from core.alert_system import AlertSystem

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SequentialLiveTest")

async def main():
    logger.info("🚀 Starting Octivault Trader - Sequential Live Test Mode")

    config = Config()
    config.LIVE_MODE = True

    shared_state = SharedState(config)
    exchange_client = ExchangeClient(config=config, shared_state=shared_state)
    await exchange_client.initialize()

    logger.info("🔍 Fetching symbols from Binance wallet...")
    symbols = await exchange_client.get_symbols_from_balance()
    symbols = [s.upper() for s in symbols]
    config.SYMBOLS = symbols
    shared_state.symbols = symbols
    logger.info(f"✅ Using symbols: {symbols}")

    market_data_feed = MarketDataFeed(config, shared_state, exchange_client, symbols)
    await market_data_feed.preload_historical_data()

    tp_sl_engine = TPSLEngine(shared_state, config, exchange_client)
    execution_manager = ExecutionManager(config, shared_state, exchange_client, tp_sl_engine)
    capital_allocator = CapitalAllocator(shared_state, config)
    alert_system = AlertSystem(config=config, shared_state=shared_state)
    recovery_engine = RecoveryEngine(shared_state=shared_state, config=config, alert_callback=alert_system.send_alert)
    meta_controller = MetaController(shared_state, config, exchange_client, execution_manager)
    sentiment_stream = BinanceSentimentStream(shared_state, config)

    agents = [
        MLForecaster(shared_state, market_data_feed, execution_manager, config, tp_sl_engine, symbols=symbols),
        IPOChaser(shared_state, market_data_feed, execution_manager, config, tp_sl_engine, symbols=symbols),
        DipSniper(shared_state, market_data_feed, execution_manager, config, tp_sl_engine, symbols=symbols),
        RLStrategist(shared_state, market_data_feed, execution_manager, config, tp_sl_engine, symbols=symbols),
        TrendHunter(shared_state, market_data_feed, execution_manager, config, tp_sl_engine, symbols=symbols),
        NewsReactor(shared_state, market_data_feed, execution_manager, config, tp_sl_engine, sentiment_stream=sentiment_stream, symbols=symbols),
        LiquidationAgent(shared_state, market_data_feed, execution_manager, config, tp_sl_engine, meta_controller, symbols=symbols)
    ]

    log_component_status("SequentialLiveTest", "Agents Initialized")

    logger.info("🔁 Running each agent once...")
    for agent in agents:
        if hasattr(agent, "run_once"):
            await agent.run_once()

    logger.info("🔍 Running MetaController once...")
    await meta_controller.run_once()

    logger.info("📈 Running ExecutionManager + CapitalAllocator once...")
    for symbol in symbols:
        action, confidence = meta_controller.get_best_action(symbol)
        qty = capital_allocator.get_quantity(symbol)
        if action in ["BUY", "SELL"] and qty > 0:
            await execution_manager.execute(symbol, action, qty, confidence)

    logger.info("🛡️ Running TPSLEngine check_orders once...")
    await tp_sl_engine.check_orders()

    logger.info("✅ Test complete. All modules validated sequentially in LIVE mode.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"❌ Unhandled exception: {e}", exc_info=True)
        sys.exit(1)
