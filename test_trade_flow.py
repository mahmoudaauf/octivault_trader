import asyncio
import logging

from core.config import Config
from core.database_manager import DatabaseManager
from core.shared_state import SharedState
from core.exchange_client import ExchangeClient
from core.execution_manager import ExecutionManager
from core.tp_sl_engine import TPSLEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestTradeFlow")

class DummyAgent:
    def __init__(self, name="TestAgent"):
        self.name = name

    def __str__(self):
        return self.name

async def test_trade_simulation():
    config = Config()
    database_manager = DatabaseManager(config=config)

    # Circular dependency workaround
    exchange_client = ExchangeClient(config=config, shared_state=None)
    shared_state = SharedState(config=config, database_manager=database_manager, exchange_client=exchange_client)
    exchange_client.shared_state = shared_state

    execution_manager = ExecutionManager(
        config=config,
        shared_state=shared_state,
        exchange_client=exchange_client
    )

    tp_sl_engine = TPSLEngine(
        shared_state=shared_state,
        exchange_client=exchange_client,
        config=config,
        execution_manager=execution_manager
    )

    # Test trade
    symbol = "DOGEUSDT"
    action = "buy"
    confidence = 0.95
    reason = "Test simulation"
    agent = DummyAgent()

    logger.info(f"🧪 Simulating {action.upper()} order for {symbol}")

    result = await execution_manager.execute_trade(
        symbol=symbol,
        action=action,
        confidence=confidence,
        reason=reason,
        agent=agent
    )

    logger.info(f"✅ Trade execution result: {result}")

if __name__ == "__main__":
    asyncio.run(test_trade_simulation())
