import asyncio
import logging
import sys # Added sys import for CLI argument parsing

from core.config import Config
from core.app_context import AppContext  # Make sure your script is structured to import this

logging.basicConfig(level=logging.INFO)

# Modified run function to accept a specific phase
async def run(phase):
    config = Config()
    
    # Initialize context but do NOT use `__aenter__`, we'll do it step-by-step
    app_context = AppContext(config=config)

    # Added print statement as requested
    print(f"🔥 Running up to Phase: {phase}")
    await app_context.initialize_all(up_to_phase=phase)
    
    if phase == 9:
        # Added print statement as requested
        print("🚀 Launching Phase 9 operational loops...")
        # Added log marker as requested
        print("⚠️ Phase 9: Calling start_background_tasks()")
        await app_context.start_background_tasks() # This must be called

        # Temporary fallback added as requested
        print("⚠️ Fallback: Directly calling _run_phase9_logic()")
        from core.app_context import _run_phase9_logic
        await _run_phase9_logic(app_context)

# Entry point
if __name__ == "__main__":
    # Get phase from command line argument, default to 1 if not provided
    phase = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    asyncio.run(run(phase))

# Note: The request also included adding 'logger.info("📍 Entered _run_phase9_logic()")'
# This change would need to be made inside the `_run_phase9_logic()` method
# within your `AppContext` class definition (likely in core/app_context.py),
# as that function is not part of this script.
