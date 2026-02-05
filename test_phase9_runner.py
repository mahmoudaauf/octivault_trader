import asyncio
import logging
import os
import traceback

from core.config import Config
from core.app_context import AppContext

# ───────────────────────────────
# Setup basic logging
# ───────────────────────────────
log_path = "logs/test_phase9_debugger.log"
os.makedirs(os.path.dirname(log_path), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("Phase9Debugger")


async def debug_phase9():
    logger.info("🧪 Starting Phase 9 Debugging")

    try:
        # Initialize AppContext with Config
        logger.info("📦 Initializing AppContext...")
        config = Config()
        app_context = AppContext(config=config)

        # Run Phase 1–5 (symbol discovery, market data warmup)
        logger.info("🔁 Running Phases 1–5...")
        await app_context.run_phase1_logic()
        await app_context.run_phase2_logic()
        await app_context.run_phase3_logic()
        await app_context.run_phase4_logic()
        await app_context.run_phase5_logic()

        # Phase 6 – register strategy agents
        logger.info("🤖 Running Phase 6 (Agent Registration)...")
        await app_context.run_phase6_logic()

        # Phase 7 – initialize core components
        logger.info("⚙️ Running Phase 7 (Execution Core)...")
        await app_context.run_phase7_logic()

        # Phase 8 – no-op or warm-up
        logger.info("🌡️ Running Phase 8 (Prep)...")
        await app_context.run_phase8_logic()

        # Phase 9 – final live run
        logger.info("🚀 Running Phase 9 (Live Run)...")
        await app_context.run_phase9_logic()

    except Exception as e:
        logger.error(f"❌ Exception during Phase 9 debug: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(debug_phase9())
