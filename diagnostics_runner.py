# diagnostics_runner.py
import asyncio
import logging
from core.config import Config
from core.app_context import AppContext  # Assuming this initializes all components safely

logger = logging.getLogger("Diagnostics")
logging.basicConfig(level=logging.INFO)

async def diagnostics_main():
    config = Config()
    app_context = AppContext(config)

    # Initialize all up to Phase 9 so everything is fully wired
    await app_context.initialize_all(up_to_phase=9)

    logger.info("✅ AppContext initialized with all components.")

    # Define component tests
    async def test_component(name, method):
        if not method or not callable(method):
            logger.warning(f"⚠️ {name} has no valid callable method to test.")
            return
        try:
            logger.info(f"🔍 Testing {name}...")
            await asyncio.wait_for(method(), timeout=10)
            logger.info(f"✅ {name} test completed successfully.")
        except asyncio.TimeoutError:
            logger.error(f"⏱️ {name} test timed out.")
        except Exception as e:
            logger.exception(f"❌ {name} test failed: {e}")

    # MetaController
    if app_context.meta_controller:
        await test_component("MetaController.run_once", app_context.meta_controller.run_once)

    # AgentManager
    if app_context.agent_manager:
        await test_component("AgentManager.run_once", getattr(app_context.agent_manager, "run_once", None))

    # ExecutionManager
    if app_context.execution_manager:
        await test_component("ExecutionManager.run_once", getattr(app_context.execution_manager, "run_once", None))

    # RiskManager
    if app_context.risk_manager:
        await test_component("RiskManager.run_once", getattr(app_context.risk_manager, "run_once", None))

    # TPSLEngine
    if app_context.tp_sl_engine:
        await test_component("TP/SLEngine.run_once", getattr(app_context.tp_sl_engine, "run_once", None))

    # MarketDataFeed
    if app_context.market_data_feed:
        await test_component("MarketDataFeed.test_poll", getattr(app_context.market_data_feed, "test_poll", None))

    # SharedState
    if app_context.shared_state:
        try:
            logger.info("🧠 SharedState symbol count: %s", len(app_context.shared_state.get_symbols()))
        except Exception as e:
            logger.error(f"❌ SharedState access failed: {e}")

    # Heartbeat
    if app_context.heartbeat:
        await test_component("Heartbeat.run", app_context.heartbeat.run)

    # Watchdog
    if app_context.watchdog:
        await test_component("Watchdog.run_loop", app_context.watchdog.run_loop)

    # PnLCalculator
    if app_context.pnl_calculator:
        await test_component("PnLCalculator.run_once", getattr(app_context.pnl_calculator, "run_once", None))

    # PerformanceMonitor
    if app_context.performance_monitor:
        await test_component("PerformanceMonitor.run_once", getattr(app_context.performance_monitor, "run_once", None))

    logger.info("✅ Diagnostics run complete.")

if __name__ == "__main__":
    asyncio.run(diagnostics_main())
