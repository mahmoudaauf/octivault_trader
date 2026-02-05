import asyncio
from datetime import datetime, timedelta
from core.utils.structured_logger import get_logger

logger = get_logger("Watchdog")

class Watchdog:
    def __init__(self, check_interval_seconds, config, shared_state):
        self.check_interval = check_interval_seconds
        self.tolerance_time = timedelta(seconds=check_interval_seconds * 3)
        self.config = config
        self.shared_state = shared_state

    async def run(self):
        logger.info("🛡️ Watchdog initialized.")
        while True:
            try:
                now = datetime.utcnow()
                component_statuses = self.shared_state.get_component_statuses()  # dict: {component_name: timestamp}

                logger.info("\n🩺 COMPONENT HEALTH CHECK:")
                for component, last_ping in component_statuses.items():
                    delta = now - last_ping
                    status = self._evaluate_status(delta)
                    logger.info(f"- {component}: {status} ({delta.total_seconds():.1f}s ago)")

                missing = set(self.shared_state.expected_components) - set(component_statuses.keys())
                for missing_component in missing:
                    logger.warning(f"❌ {missing_component}: No heartbeat detected")

            except Exception as e:
                logger.error(f"🔥 Watchdog encountered an error: {e}", exc_info=True)

            await asyncio.sleep(self.check_interval)

    def _evaluate_status(self, delta):
        if delta < timedelta(seconds=self.check_interval * 1.5):
            return "✅ Healthy"
        elif delta < self.tolerance_time:
            return "⚠️ Slow"
        else:
            return "❌ Dead"
