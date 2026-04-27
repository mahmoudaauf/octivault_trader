#!/usr/bin/env python3
"""Quick test: Can agents load DEFAULT_SYMBOLS fallback?"""
import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

async def test():
    # Test 1: Import DEFAULT_SYMBOLS
    print("Test 1: Import DEFAULT_SYMBOLS...")
    try:
        from core.bootstrap_symbols import DEFAULT_SYMBOLS
        print(f"  ✅ Imported {len(DEFAULT_SYMBOLS)} DEFAULT_SYMBOLS")
        print(f"  Keys: {list(DEFAULT_SYMBOLS.keys())}")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return
    
    # Test 2: Check if agent would use fallback
    print("\nTest 2: Simulate agent symbol loading...")
    accepted = {}  # Simulate empty accepted_symbols
    
    if not accepted:
        print("  detected empty accepted_symbols, using fallback...")
        try:
            from core.bootstrap_symbols import DEFAULT_SYMBOLS
            accepted = DEFAULT_SYMBOLS
            print(f"  ✅ Using {len(DEFAULT_SYMBOLS)} DEFAULT_SYMBOLS as fallback")
            print(f"  Fallback symbols: {list(accepted.keys())}")
        except Exception as e:
            print(f"  ❌ Fallback failed: {e}")
            accepted = {}
    
    symbols = list(accepted.keys())
    print(f"\n✅ Result: {len(symbols)} symbols available")
    print(f"   Ready for agents to use!")

if __name__ == "__main__":
    asyncio.run(test())
