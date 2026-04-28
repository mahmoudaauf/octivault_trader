# core/health.py

import logging
from datetime import datetime

logger = logging.getLogger("Health")

async def update_health(shared_state, component_name, status, detail=""):
    """
    Report health status of a component to the shared state.
    """
    timestamp = datetime.utcnow().isoformat()
    try:
        if hasattr(shared_state, "set_component_status"):
            await shared_state.set_component_status(
                component_name=component_name,
                status=status,
                detail=detail,
                timestamp=timestamp
            )
            logger.info(f"🩺 [Health] {component_name} → {status} | {detail}")
        else:
            logger.warning(f"⚠️ SharedState missing method: set_component_status()")
    except Exception as e:
        logger.error(f"❌ Failed to update health for {component_name}: {e}")
