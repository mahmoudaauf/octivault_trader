import asyncio
import logging
from datetime import datetime

# Adjust import paths as needed for your project layout
from core.execution_manager import ExecutionManager
from core.tp_sl_engine import TPSLEngine
from core.meta_controller import MetaController
from core.agent_manager import AgentManager
from core.risk_manager import RiskManager
from core.performance_monitor import PerformanceMonitor
from core.shared_state import SharedState
from core.config import Config

logger = logging.getLogger("PhaseDiagnostics")
logging.basicConfig(level=logging.INFO)

async def run_diagnostics(app_context):
    components = {
        "ExecutionManager": app_context.execution_manager,
        "MetaController": app_context.meta_controller,
        "AgentManager": app_context.agent_manager,
        "RiskManager": app_context.risk_manager,
        "TPSLEngine": app_context.tp_sl_engine,
        "PerformanceMonitor": app_context.performance_monitor,
        "MarketDataFeed": app_context.market_data_feed,
    }

    logger.info("🔍 Running Phase 8 & 9 Diagnostics — %s", datetime.now())

    # Check each component for existence and method availability
    for name, component in components.items():
        if component is None:
            logger.error(f"❌ {name} is not initialized!")
            continue

        # Check for async 'start' or 'run' method
        if hasattr(component, 'start') or hasattr(component, 'run'):
            logger.info(f"✅ {name} is initialized and has a start/run method.")
        else:
            logger.warning(f"⚠️ {name} is initialized but missing a start/run method.")

        # Check for health registration
        try:
            status = app_context.shared_state.get_component_status(name)
            if status:
                logger.info(f"📊 {name} Watchdog Status: {status}")
            else:
                logger.warning(f"🚨 {name} has not registered with Watchdog.")
        except Exception as e:
            logger.error(f"❌ Failed to check status for {name}: {e}")

    # Extra checks
    logger.info("🧪 Checking active agent count...")
    try:
        agents = app_context.agent_manager.get_registered_agents()
        logger.info(f"🧠 Registered Agents: {len(agents)}")
    except Exception as e:
        logger.warning(f"❌ Unable to retrieve agents: {e}")

    logger.info("✅ Phase 8 & 9 Diagnostics Complete.\n")

# Use this in your main runner
if __name__ == "__main__":
    import sys
    import os

    sys.path.append(os.getcwd())  # Ensure current dir is in path

    async def main():
        from core.app_context import AppContext
        config = Config()
        app_context = AppContext(config)
        await app_context.initialize()  # You might replace with actual phase init

        await run_diagnostics(app_context)

    asyncio.run(main())
