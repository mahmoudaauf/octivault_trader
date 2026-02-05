import asyncio
import logging

from core.config import Config
from core.exchange_client import ExchangeClient
from core.shared_state import SharedState
from core.market_data_feed import MarketDataFeed
from core.symbol_manager import SymbolManager
from core.agent_manager import AgentManager
from core.meta_controller import MetaController
from core.tp_sl_engine import TPSLEngine
from core.execution_manager import ExecutionManager
from core.risk_manager import RiskManager
from core.position_manager import PositionManager
from core.portfolio_manager import PortfolioManager
from core.compounding_engine import CompoundingEngine
from core.performance_monitor import PerformanceMonitor
from core.alert_system import AlertSystem
from core.watchdog import Watchdog
from core.heartbeat import Heartbeat

from agent_registry import AGENT_CLASS_MAP  # Maps agent names to classes

# Optional:
# from core.notification_manager import NotificationManager

logger = logging.getLogger("LiveLauncher")
logging.basicConfig(level=logging.INFO)

async def run_live():
    # ─────────────────────────────────────────────
    # 🎛 Initialization
    # ─────────────────────────────────────────────
    config = Config()
    shared_state = SharedState(config=config, exchange_client=None)  # Temporarily set None
    # Now pass shared_state into ExchangeClient
    exchange_client = ExchangeClient(config=config, shared_state=shared_state)
    # Re-link exchange_client inside shared_state
    shared_state.exchange_client = exchange_client
    symbol_manager = SymbolManager(config=config)

    # ─────────────────────────────────────────────
    # 📡 Market Data
    # ─────────────────────────────────────────────
    market_data_feed = MarketDataFeed(shared_state=shared_state, exchange_client=exchange_client, config=config)
    await market_data_feed.load_historical_data()  # warm-up OHLCV

    # ─────────────────────────────────────────────
    # ⚙️ Core Trading Engines
    # ─────────────────────────────────────────────
    execution_manager = ExecutionManager(shared_state=shared_state, exchange_client=exchange_client, config=config)
    tp_sl_engine = TPSLEngine(shared_state=shared_state, execution_manager=execution_manager, config=config)
    risk_manager = RiskManager(shared_state=shared_state, execution_manager=execution_manager, config=config)
    position_manager = PositionManager(shared_state=shared_state, execution_manager=execution_manager, config=config)
    portfolio_manager = PortfolioManager(shared_state=shared_state, exchange_client=exchange_client, config=config)
    compounding_engine = CompoundingEngine(shared_state=shared_state, portfolio_manager=portfolio_manager, config=config)

    # ─────────────────────────────────────────────
    # 🧠 Strategy Agents
    # ─────────────────────────────────────────────
    agent_manager = AgentManager(
        shared_state=shared_state,
        execution_manager=execution_manager,
        tp_sl_engine=tp_sl_engine,
        config=config
    )

    # Register strategist agents
    for name, AgentClass in AGENT_CLASS_MAP.items():
        agent = AgentClass(
            shared_state=shared_state,
            execution_manager=execution_manager,
            config=config,
            tp_sl_engine=tp_sl_engine,
            name=name
        )
        await agent_manager.register(agent)

    # ─────────────────────────────────────────────
    # 🧠 MetaController
    # ─────────────────────────────────────────────
    meta_controller = MetaController(
        shared_state=shared_state,
        agent_manager=agent_manager,
        execution_manager=execution_manager,
        config=config
    )

    # ─────────────────────────────────────────────
    # 🩺 Monitoring
    # ─────────────────────────────────────────────
    performance_monitor = PerformanceMonitor(shared_state=shared_state, config=config)
    alert_system = AlertSystem(shared_state=shared_state, config=config)
    watchdog = Watchdog(shared_state=shared_state, config=config)
    heartbeat = Heartbeat(shared_state=shared_state, config=config)

    # Optional NotificationManager
    # notification_manager = NotificationManager(shared_state=shared_state, config=config)

    # ─────────────────────────────────────────────
    # 🚀 Launch All Components
    # ─────────────────────────────────────────────
    logger.info("✅ Launching all live tasks...")

    tasks = [
        asyncio.create_task(market_data_feed.run()),
        asyncio.create_task(agent_manager.run()),
        asyncio.create_task(meta_controller.run_loop()),
        asyncio.create_task(tp_sl_engine.monitor_trades()),
        asyncio.create_task(execution_manager.monitor_orders()),
        asyncio.create_task(performance_monitor.track()),
        asyncio.create_task(heartbeat.ping()),
        asyncio.create_task(watchdog.monitor()),
        asyncio.create_task(compounding_engine.run()),
        asyncio.create_task(position_manager.sync_positions())
        # asyncio.create_task(notification_manager.run()),  # if enabled
    ]

    await asyncio.gather(*tasks)

if __name__ == "__main__":
    try:
        asyncio.run(run_live())
    except KeyboardInterrupt:
        print("🛑 Gracefully shutting down Octivault Trader Live Mode.")
