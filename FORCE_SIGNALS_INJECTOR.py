#!/usr/bin/env python3
"""
⚡ ITERATION 2: SIGNAL INJECTOR - Force signals into system to bypass agents

This tool bypasses all agent logic and directly injects high-confidence signals into
the MetaController's signal cache. This lets us test if MetaController can execute trades
when given valid signals, without waiting for agents to generate them.

Purpose:
- Test MetaController decision-making logic
- Verify trade execution path
- Unlock profitability test without agent dependencies

Strategy:
- Connect to existing system
- Load default symbols
- Inject high-confidence BUY signals every tick
- Monitor signal cache & trade execution
"""

import sys
import asyncio
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)-8s] %(name)s - %(message)s"
)
logger = logging.getLogger("SignalInjector")

def main():
    """Main signal injection loop"""
    logger.info("=" * 80)
    logger.info("⚡ SIGNAL INJECTOR - Force signals into system")
    logger.info("=" * 80)
    
    # 1. Load config
    try:
        from core.config import Config
        config = Config()
        logger.info("✅ Config loaded")
    except Exception as e:
        logger.error("❌ Config load failed: %s", e, exc_info=True)
        return
    
    # 2. Initialize exchange client
    try:
        from core.exchange_client import ExchangeClient
        exchange = ExchangeClient()
        logger.info("✅ Exchange initialized")
    except Exception as e:
        logger.error("❌ Exchange init failed: %s", e, exc_info=True)
        return
    
    # 3. Initialize SharedState
    try:
        from core.shared_state import SharedState
        shared_state = SharedState()
        logger.info("✅ SharedState initialized")
    except Exception as e:
        logger.error("❌ SharedState init failed: %s", e, exc_info=True)
        return
    
    # 4. Bootstrap symbols
    try:
        from core.bootstrap_symbols import bootstrap_default_symbols
        result = bootstrap_default_symbols(shared_state, logger)
        logger.info("✅ Symbols bootstrapped")
        
        # Verify symbols
        symbols = shared_state.get_accepted_symbols()
        logger.info("📊 Available symbols: %s", list(symbols.keys()) if symbols else "None")
        
        if not symbols:
            logger.error("❌ No symbols available after bootstrap!")
            return
    except Exception as e:
        logger.error("❌ Bootstrap failed: %s", e, exc_info=True)
        return
    
    # 5. Initialize SignalManager
    try:
        from core.signal_manager import SignalManager
        signal_mgr = SignalManager()
        logger.info("✅ SignalManager initialized")
    except Exception as e:
        logger.error("❌ SignalManager init failed: %s", e, exc_info=True)
        return
    
    # 6. Get MetaController (just to verify it initializes, don't start it)
    try:
        from core.meta_controller import MetaController
        meta = MetaController()
        logger.info("✅ MetaController initialized (not started)")
    except Exception as e:
        logger.error("❌ MetaController init failed: %s", e, exc_info=True)
        return
    
    # 7. Main injection loop
    logger.info("\n" + "=" * 80)
    logger.info("🔄 Starting signal injection loop...")
    logger.info("=" * 80 + "\n")
    
    symbols = list(shared_state.get_accepted_symbols().keys()) if shared_state.get_accepted_symbols() else []
    cycle = 0
    injection_count = 0
    
    try:
        while True:
            cycle += 1
            now = time.time()
            
            # Inject high-confidence BUY signal for each symbol
            injected_this_cycle = 0
            for symbol in symbols:
                signal = {
                    "direction": "BUY",
                    "confidence": 0.75,  # Above 0.60 bootstrap floor
                    "reason": "Injected test signal",
                    "timestamp": now,
                    "quote": 10.0,
                }
                
                if signal_mgr.receive_signal("SignalInjector", symbol, signal):
                    injected_this_cycle += 1
                    injection_count += 1
            
            # Log every 5 cycles
            if cycle % 5 == 0:
                all_signals = signal_mgr.get_all_signals()
                logger.info(
                    "[Cycle %d] 💉 Injected: %d signals | Cache: %d total | Injection total: %d",
                    cycle, injected_this_cycle, len(all_signals), injection_count
                )
            
            # Wait before next cycle
            time.sleep(2)
    
    except KeyboardInterrupt:
        logger.info("\n⏹️  Interrupted by user")
    except Exception as e:
        logger.error("❌ Injection loop error: %s", e, exc_info=True)
    finally:
        logger.info(f"✅ Injection loop ended after {cycle} cycles, {injection_count} signals injected")

if __name__ == "__main__":
    main()
