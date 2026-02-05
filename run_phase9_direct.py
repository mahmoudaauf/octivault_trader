import asyncio
from core.config import Config
from core.app_context import AppContext
from core.app_context import _run_phase9_logic

async def main():
    config = Config()
    app_context = AppContext(config=config)

    print("🔥 Running full initialization up to Phase 9")
    await app_context.initialize_all(up_to_phase=9)

    print("🚀 Now calling _run_phase9_logic() directly")
    await _run_phase9_logic(app_context)

if __name__ == "__main__":
    asyncio.run(main())
