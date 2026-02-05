import asyncio
import logging

from core.config import Config
from core.app_context import AppContext

async def reset():
    logging.basicConfig(level=logging.INFO)
    config = Config()

    app_context = AppContext(config=config)
    await app_context.initialize_all(up_to_phase=1)

    app_context.shared_state.symbols.clear()
    await app_context.database_manager.clear_symbols()

    print("✅ All active and stored symbols have been cleared.")

    await app_context.shutdown()

if __name__ == "__main__":
    asyncio.run(reset())
