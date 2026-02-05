# debug_watchdog.py
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict

logger = logging.getLogger("DebugWatchdog")
logger.setLevel(logging.DEBUG)

# ⏱ Configurable timeout (seconds)
COMPONENT_TIMEOUT = 45

# 🧠 In-memory registry for component heartbeats
component_heartbeats: Dict[str, datetime] = {}

def register_heartbeat(component: str):
    component_heartbeats[component] = datetime.utcnow()
    logger.debug(f"✅ Heartbeat received from {component} at {component_heartbeats[component].isoformat()}")

def get_stale_components():
    now = datetime.utcnow()
    return {
        comp: (now - ts).total_seconds()
        for comp, ts in component_heartbeats.items()
        if (now - ts).total_seconds() > COMPONENT_TIMEOUT
    }

async def debug_watchdog_loop():
    logger.info("🛡️ DebugWatchdog started. Monitoring component heartbeats...")

    while True:
        await asyncio.sleep(10)

        stale = get_stale_components()
        if stale:
            for comp, age in stale.items():
                logger.warning(f"⚠️ Component '{comp}' is stale! Last heartbeat {age:.1f}s ago.")
        else:
            logger.info("✅ All components are reporting within timeout.")
