import asyncio
from core.diagnostics.system_summary_logger import system_summary_logger
from core.monitoring.watchdog_enhanced import Watchdog
from core.utils.structured_logger import get_logger
from core.context import AppContext  # assuming you have an AppContext for access to shared modules

logger = get_logger("MonitoringMain")

async def main():
    logger.info("🚀 Monitoring system started. Initializing...")

    app_context = AppContext()
    await app_context.initialize_all()  # assumes this sets up shared_state, agents, etc.

    # Launch monitoring tasks
    tasks = [
        asyncio.create_task(system_summary_logger(app_context)),
        asyncio.create_task(Watchdog(
            check_interval_seconds=10,
            config=app_context.config,
            shared_state=app_context.shared_state
        ).run())
    ]

    try:
        await asyncio.gather(*tasks)
    except Exception as e:
        logger.error(f"🔥 Monitoring crashed: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())
