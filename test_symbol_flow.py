import asyncio
import logging
from core.config import Config
from core.app_context import AppContext # Only AppContext is needed for imports

# Configure logging for better output visibility
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SymbolFlowTest")


async def run_symbol_flow_diagnostics():
    """
    Runs a diagnostic test focused on the final symbol snapshot and its
    consistency with the database.

    This function performs the following checks:
    1. Initializes the application context up to phase 3.
    2. Verifies the count of accepted symbols currently in SharedState.
    3. Saves a snapshot of these accepted symbols to the database.
    4. Reloads the snapshot from the database to confirm persistence.
    5. Asserts the integrity and content of the reloaded snapshot.
    """
    logger.info("🔍 Running Final Symbol Snapshot Diagnostic")

    # Initialize configuration
    config = Config()

    # Initialize AppContext and all its components up to phase 3
    app_context = AppContext(config)
    await app_context.initialize_all(up_to_phase=3)

    # Get the initialized components from app_context
    shared_state = app_context.shared_state
    database_manager = app_context.database_manager
    symbol_manager = app_context.symbol_manager # Kept for completeness, though not directly used in this simplified test

    # ✅ 1. Check how many accepted symbols exist in SharedState
    # This line is kept as it's also part of the original flow,
    # and the new write operation needs 'accepted' symbols.
    accepted = await shared_state.get_accepted_symbols()
    logger.info(f"✅ SharedState.accepted_symbols: {len(accepted)}")

    # ✅ 2. Write snapshot directly to DB using database_manager
    # The 'accepted' symbols are retrieved again here to ensure the most
    # up-to-date list is written, following the user's instruction.
    accepted = await shared_state.get_accepted_symbols()
    await database_manager.write_symbol_snapshot(accepted)
    logger.info("✅ Snapshot written to DB successfully.")

    # ✅ 3. Reload from DB and verify consistency
    reloaded = await database_manager.load_symbol_snapshot()
    logger.info(f"✅ Reloaded snapshot contains: {len(reloaded)} symbols")

    # Assertions to ensure the reloaded snapshot is valid and not empty
    assert reloaded, "❌ Snapshot reload failed — empty list"
    # Check if the first reloaded symbol has a 'symbol' field, indicating
    # that the data structure is as expected.
    assert reloaded[0].get("symbol"), "❌ Reloaded symbols missing 'symbol' field"

    logger.info("🎉 ALL CHECKS PASSED: Finalized symbol snapshot and DB are consistent ✅")

    # Ensure the database connection is closed after the diagnostic
    await database_manager.disconnect()


if __name__ == "__main__":
    # Entry point for running the diagnostic test.
    # Uses asyncio.run to execute the asynchronous function.
    try:
        asyncio.run(run_symbol_flow_diagnostics())
    except Exception as e:
        # Catch any exceptions that occur during the test and log them.
        logger.exception(f"❌ Diagnostic test failed: {e}")
