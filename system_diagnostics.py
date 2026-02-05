import asyncio
import logging

# 1. ✅ Match to Live System Structure: Replace mocks with real components
# Uncomment these imports when the actual core modules are available in your project.
from core.config import Config
from core.exchange_client import ExchangeClient
from core.shared_state import SharedState
from core.database_manager import DatabaseManager
from core.symbol_manager import SymbolManager # 2. ✅ Add SymbolManager Check

# Set up basic logging to show INFO level messages
logging.basicConfig(level=logging.INFO)

# --- Mock Classes (Removed for actual integration, kept here as a reference for original structure) ---
# The mock classes (MockConfig, MockDatabaseManager, MockSharedState, MockExchangeClient)
# have been removed from the active code to allow integration with your live system's
# core components as per your request.

# --- Main Diagnostic Function ---

async def run_diagnostics():
    """
    Runs a series of diagnostic checks for the trading bot components.
    This includes initializing configuration, database, shared state,
    symbol manager, and exchange client, then attempting to fetch market data
    for all finalized symbols.
    """
    logging.info("Starting diagnostics...")

    # Initialize variables to None so they can be referenced in finally block even if
    # initialization fails.
    config = None
    db = None
    shared_state = None
    exchange_client = None
    symbol_manager = None

    try:
        # 1. ✅ Match to Live System Structure: Modify initialization to use real components
        config = Config()
        logging.info("Config initialized.")

        db = DatabaseManager(config)
        await db.connect()
        logging.info("DatabaseManager connected.")

        shared_state = SharedState(config, db)
        await shared_state.initialize_from_database()
        logging.info("SharedState initialized from database.")

        # 2. ✅ Add SymbolManager Check
        # Initialize SymbolManager after shared_state, as it often depends on it.
        # ExchangeClient is passed as it might be needed for symbol validation/fetching.
        exchange_client = ExchangeClient(config, shared_state) # Initialize ExchangeClient before SymbolManager
        await exchange_client.initialize()
        logging.info("ExchangeClient initialized.")

        # Removed: final_symbols = await symbol_manager.get_finalized_symbols()
        # Updated as per user request to use shared_state.get_active_symbols()
        final_symbols = shared_state.get_active_symbols()
        print(f"\n📌 Proposed/filtered symbols: {final_symbols}")

        # Use the finalized symbols for market data fetching
        symbols_to_fetch = final_symbols if final_symbols else config.SYMBOLS # Fallback to config.SYMBOLS if shared_state returns empty

        print(f"Attempting to fetch market data for: {symbols_to_fetch}")

        # Iterate and fetch market data for each symbol
        if symbols_to_fetch:
            for symbol in symbols_to_fetch:
                print(f"Fetching market data for: {symbol}")
                try:
                    # Updated the call to get_ohlcv as per user request
                    candles = await exchange_client.get_ohlcv(symbol, timeframe="1m")
                    print(f"Successfully received {len(candles)} candles for {symbol}")
                    # Optionally, print a sample of the data
                    if candles:
                        print(f"Sample candle for {symbol}: {candles[0]}")
                except Exception as e:
                    logging.error(f"Failed to fetch market data for {symbol}: {e}")
        else:
            print("No symbols configured or finalized to fetch market data.")

    except Exception as e:
        logging.critical(f"An unhandled error occurred during diagnostics: {e}", exc_info=True)
    finally:
        # 4. ✅ Wrap with try/finally: Ensure resources are closed even on exception
        if db:
            await db.close()
            logging.info("DatabaseManager connection closed.")
        if exchange_client:
            await exchange_client.close()
            logging.info("ExchangeClient connection closed.")
            # 3. ✅ Close aiohttp Sessions Properly
            # IMPORTANT: Ensure that your actual ExchangeClient.close() method
            # calls 'await self.session.close()' if it uses aiohttp.ClientSession
            # to prevent "Unclosed client session..." warnings.

        logging.info("Diagnostics completed.")

# Entry point for the script
if __name__ == "__main__":
    # Run the asynchronous diagnostic function
    asyncio.run(run_diagnostics())
