import asyncio
from core.config import Config
from core.database_manager import DatabaseManager
from core.shared_state import SharedState
from core.exchange_client import ExchangeClient

async def main():
    # Load config
    cfg = Config()
    
    # Initialize database manager
    db = DatabaseManager(cfg)
    await db.connect()

    # Initialize shared state
    shared_state = SharedState(cfg, db)
    
    # Set shared_state reference in database (needed for snapshots, PnL, etc.)
    db.shared_state = shared_state

    # Initialize exchange client
    exchange = ExchangeClient(cfg, shared_state)
    await exchange.initialize()

    print("✅ All core modules initialized successfully.")

    # Graceful cleanup
    await exchange.close()
    await db.close()

if __name__ == "__main__":
    asyncio.run(main())
