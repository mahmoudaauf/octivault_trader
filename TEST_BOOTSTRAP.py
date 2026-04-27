#!/usr/bin/env python3
"""Quick test of bootstrap function"""
import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import logging
logging.basicConfig(level=logging.WARNING, format="%(name)s - %(message)s")
logger = logging.getLogger("BootstrapTest")

async def test():
    # Import after path setup
    from core.config import Config
    from core.exchange_client import ExchangeClient
    from core.shared_state import SharedState
    from core.bootstrap_symbols import bootstrap_default_symbols
    
    logger.warning("=" * 80)
    logger.warning("BOOTSTRAP TEST")
    logger.warning("=" * 80)
    
    # Init
    config = Config()
    exchange = ExchangeClient(config=config, logger=logger)
    
    shared_state = SharedState(config=config, logger=logger, exchange_client=exchange)
    
    logger.warning("\nBefore bootstrap:")
    logger.warning("  Symbols: %d", len(shared_state.accepted_symbols))
    logger.warning("  Keys: %s", list(shared_state.accepted_symbols.keys()))
    
    # Bootstrap
    logger.warning("\nRunning bootstrap...")
    result = await bootstrap_default_symbols(shared_state, logger)
    
    logger.warning("\nAfter bootstrap:")
    logger.warning("  Result: %s", result)
    logger.warning("  Symbols: %d", len(shared_state.accepted_symbols))
    logger.warning("  Keys: %s", list(shared_state.accepted_symbols.keys()))
    
    if len(shared_state.accepted_symbols) > 0:
        logger.warning("\n✅ SUCCESS - Symbols were set!")
    else:
        logger.warning("\n❌ FAILURE - Symbols still empty!")

if __name__ == "__main__":
    asyncio.run(test())
