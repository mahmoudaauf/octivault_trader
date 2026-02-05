# integrity_checker.py

import inspect
import logging
import asyncio
import sys
import os

# Ensure current directory is in path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger("IntegrityChecker")
logging.basicConfig(level=logging.INFO)

class IntegrityChecker:
    def __init__(self, components, symbols, intervals):
        self.components = components
        self.symbols = symbols
        self.intervals = intervals

    def check_constructor_signatures(self):
        logger.info("\n🔧 Checking constructor signatures...")
        for name, instance in self.components.items():
            try:
                init_sig = inspect.signature(instance.__init__)
                logger.info(f"✅ {name}.__init__ => {init_sig}")
            except Exception as e:
                logger.error(f"❌ Failed to inspect {name} constructor: {e}")

    def check_shared_parameters(self):
        logger.info("\n🔗 Checking shared parameter availability...")
        required_attrs = ['shared_state', 'config']
        for name, instance in self.components.items():
            for attr in required_attrs:
                if hasattr(instance, attr):
                    logger.info(f"✅ {name} has '{attr}'")
                else:
                    logger.error(f"❌ {name} missing required attribute '{attr}'")

    def check_required_methods(self):
        logger.info("\n🔍 Checking for required methods...")
        method_map = {
            'ExecutionManager': ['place_order'],
            'MarketDataFeed': ['get_recent_ohlcv', 'start_polling'],
            'MetaController': ['evaluate_signals'],
            'AgentManager': ['start_agents', 'assign_agents'],
        }
        for name, methods in method_map.items():
            instance = self.components.get(name)
            if not instance:
                logger.warning(f"⚠️ {name} not provided in components.")
                continue
            for method in methods:
                if hasattr(instance, method):
                    logger.info(f"✅ {name}.{method}() found")
                else:
                    logger.error(f"❌ {name} missing method '{method}'")

    def check_symbol_interval_support(self):
        logger.info("\n📊 Validating symbol + interval data presence in shared_state.market_data...")
        shared_state = self.components.get("SharedState")
        if not shared_state:
            logger.error("❌ SharedState not in components.")
            return

        market_data = getattr(shared_state, "market_data", {})
        for symbol in self.symbols:
            for interval in self.intervals:
                try:
                    data = market_data[symbol][interval]
                    if data:
                        logger.info(f"✅ market_data[{symbol}][{interval}] found")
                    else:
                        logger.warning(f"⚠️ market_data[{symbol}][{interval}] is empty")
                except KeyError:
                    logger.error(f"❌ Missing market_data[{symbol}][{interval}]")

    def check_agent_data_structures(self):
        logger.info("\n🧠 Checking agent_signals, agent_scores, sentiment, drawdown per symbol...")
        shared_state = self.components.get("SharedState")
        if not shared_state:
            logger.error("❌ SharedState not in components.")
            return

        for symbol in self.symbols:
            checks = [
                (shared_state.agent_signals, 'agent_signals'),
                (shared_state.agent_scores, 'agent_scores'),
                (shared_state.sentiment_score, 'sentiment_score'),
                (shared_state.drawdown_tracker, 'drawdown_tracker')
            ]
            for store, label in checks:
                if symbol in store:
                    logger.info(f"✅ {label}[{symbol}] present")
                else:
                    logger.warning(f"⚠️ {label}[{symbol}] missing")

    def check_execution_mode_flags(self):
        logger.info("\n🚦 Checking execution mode flags in config...")
        config = self.components.get("Config")
        if not config:
            logger.error("❌ Config not in components.")
            return

        for flag in ['LIVE_MODE', 'SIMULATION_MODE', 'PAPER_MODE']:
            if hasattr(config, flag):
                logger.info(f"✅ config.{flag} = {getattr(config, flag)}")
            else:
                logger.warning(f"⚠️ config.{flag} is missing")

    def run_all(self):
        logger.info("\n🎯 Starting Full System Integrity Check...")
        self.check_constructor_signatures()
        self.check_shared_parameters()
        self.check_required_methods()
        self.check_symbol_interval_support()
        self.check_agent_data_structures()
        self.check_execution_mode_flags()
        logger.info("\n✅ Integration Check Complete.")


# Optional: Add this to your main.py to call dynamically
if __name__ == "__main__":
    from core.shared_state import SharedState
    from core.config import Config
    from core.market_data_feed import MarketDataFeed
    from core.execution_manager import ExecutionManager
    from core.meta_controller import MetaController
    from core.agent_manager import AgentManager

    class DummyTPSEngine:
        async def check_orders(self):
            pass

    config = Config()
    shared_state = SharedState(config)
    market_data_feed = MarketDataFeed(shared_state, config, exchange_client=None, intervals=["5m"])
    execution_manager = ExecutionManager(shared_state, config, tp_sl_engine=DummyTPSEngine())
    meta_controller = MetaController(shared_state, config)
    agent_manager = AgentManager(shared_state, config, execution_manager)

    components = {
        "Config": config,
        "SharedState": shared_state,
        "MarketDataFeed": market_data_feed,
        "ExecutionManager": execution_manager,
        "MetaController": meta_controller,
        "AgentManager": agent_manager
    }

    checker = IntegrityChecker(components, symbols=["BTCUSDT", "ETHUSDT"], intervals=["1m", "5m"])
    checker.run_all()
