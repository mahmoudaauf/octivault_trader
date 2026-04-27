#!/usr/bin/env python3
"""
🔧 DIAGNOSTIC SCRIPT: Trace Signal Flow Through Gates

This script helps identify where signals are being dropped.
Run this to analyze the system behavior in real-time.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

async def diagnose_signal_flow():
    """Diagnose where signals are getting filtered out"""
    
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║              🔧 DIAGNOSTIC: Signal Flow Through Gates                     ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)
    
    try:
        from core.shared_state import SharedState
        from core.signal_manager import SignalManager
        from core.market_data_client import MarketDataClient
        from core.config import TradingConfig
        import logging
        
        # Setup logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s [%(levelname)s] %(name)s - %(message)s'
        )
        logger = logging.getLogger(__name__)
        
        print("✅ Loading configuration...")
        config = TradingConfig()
        
        print("✅ Initializing SharedState...")
        shared_state = SharedState(config=config, logger=logger)
        await shared_state.start()
        
        print("✅ Initializing MarketDataClient...")
        market_data = MarketDataClient(config=config, logger=logger)
        await market_data.start()
        
        print("✅ Initializing SignalManager...")
        signal_mgr = SignalManager(
            config=config,
            logger=logger,
            shared_state=shared_state
        )
        
        # Add test signals
        print("\n📝 Injecting test signals...")
        test_signals = [
            {"symbol": "BTCUSDT", "action": "BUY", "confidence": 0.75, "agent": "TEST_DIAGNOSTIC", "_source": "diagnostic"},
            {"symbol": "ETHUSDT", "action": "BUY", "confidence": 0.75, "agent": "TEST_DIAGNOSTIC", "_source": "diagnostic"},
            {"symbol": "SOLUSDT", "action": "BUY", "confidence": 0.75, "agent": "TEST_DIAGNOSTIC", "_source": "diagnostic"},
        ]
        
        for sig in test_signals:
            signal_mgr.add_signal(sig)
            print(f"  + Injected {sig['symbol']} BUY (conf={sig['confidence']})")
        
        # Retrieve and show
        print("\n📊 Signal retrieval test:")
        all_sigs = signal_mgr.get_all_signals()
        print(f"  ✅ Retrieved {len(all_sigs)} signals from cache")
        for sig in all_sigs:
            print(f"     - {sig.get('symbol')} {sig.get('action')} conf={sig.get('confidence')}")
        
        # Price check
        print("\n💹 Price data check:")
        for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
            price = market_data.get_price(sym)
            print(f"  {sym}: price={price}")
        
        # Balance check
        print("\n💰 Balance check:")
        balance = await shared_state.get_balance("USDT")
        print(f"  USDT Balance: {balance}")
        
        # Position check
        print("\n📍 Position check:")
        positions = shared_state.get_positions_snapshot()
        print(f"  Positions: {len(positions) if positions else 0}")
        for sym, pos in (positions or {}).items():
            qty = pos.get("qty") or pos.get("quantity") or 0
            print(f"    - {sym}: qty={qty}")
        
        print("\n✅ Diagnostic complete")
        
    except Exception as e:
        print(f"\n❌ Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    result = asyncio.run(diagnose_signal_flow())
    sys.exit(0 if result else 1)
