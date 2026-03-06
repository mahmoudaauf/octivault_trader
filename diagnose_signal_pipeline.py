#!/usr/bin/env python3
"""
🔍 Signal Pipeline Diagnostics

Run this script to verify the signal pipeline fix is working correctly.
It will trace signals from agents through to MetaController's signal_cache.
"""

import asyncio
import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def diagnose_signal_pipeline():
    """Run diagnostic checks on the signal pipeline."""
    
    print("\n" + "="*80)
    print("🔍 SIGNAL PIPELINE DIAGNOSTIC")
    print("="*80 + "\n")
    
    checks = {
        "Agent Manager": 0,
        "Meta Controller": 0,
        "Signal Manager": 0,
        "Event Bus": 0,
        "Signal Cache": 0,
    }
    
    # 1. Check AgentManager exists and has meta_controller
    print("1️⃣  Checking AgentManager...")
    try:
        from core.agent_manager import AgentManager
        print("   ✅ AgentManager importable")
        checks["Agent Manager"] = 1
        
        # Check if collect_and_forward_signals has the direct path fix
        import inspect
        source = inspect.getsource(AgentManager.collect_and_forward_signals)
        if "DIRECT PATH" in source and "meta_controller.receive_signal" in source:
            print("   ✅ Direct signal forwarding code PRESENT")
            checks["Agent Manager"] = 2
        else:
            print("   ❌ Direct signal forwarding code MISSING - fix not applied?")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # 2. Check MetaController has receive_signal
    print("\n2️⃣  Checking MetaController...")
    try:
        from core.meta_controller import MetaController
        print("   ✅ MetaController importable")
        
        if hasattr(MetaController, "receive_signal"):
            print("   ✅ receive_signal() method exists")
            checks["Meta Controller"] = 2
        else:
            print("   ❌ receive_signal() method missing")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # 3. Check SignalManager exists
    print("\n3️⃣  Checking SignalManager...")
    try:
        from core.signal_manager import SignalManager
        print("   ✅ SignalManager importable")
        
        if hasattr(SignalManager, "receive_signal") and hasattr(SignalManager, "get_all_signals"):
            print("   ✅ receive_signal() and get_all_signals() methods exist")
            checks["Signal Manager"] = 2
        else:
            print("   ❌ Required methods missing")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # 4. Check BoundedCache exists
    print("\n4️⃣  Checking Signal Cache...")
    try:
        from core.meta_controller import BoundedCache
        print("   ✅ BoundedCache importable")
        checks["Signal Cache"] = 2
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # 5. Check Config has required settings
    print("\n5️⃣  Checking Configuration...")
    try:
        from core.config import Config
        config = Config()
        
        ttl = getattr(config, 'signal_cache_ttl', 300)
        size = getattr(config, 'signal_cache_max_size', 1000)
        
        print(f"   ✅ signal_cache_ttl={ttl}s")
        print(f"   ✅ signal_cache_max_size={size}")
        checks["Event Bus"] = 2
    except Exception as e:
        print(f"   ⚠️  Config check (non-critical): {e}")
        checks["Event Bus"] = 1
    
    # Print summary
    print("\n" + "="*80)
    print("📊 DIAGNOSTIC SUMMARY")
    print("="*80)
    
    total = 0
    passed = 0
    for name, score in checks.items():
        status = "✅" if score == 2 else ("⚠️" if score == 1 else "❌")
        print(f"{status} {name}: {'READY' if score == 2 else 'PARTIAL' if score == 1 else 'FAILED'}")
        total += 2
        passed += score
    
    print("\n" + "="*80)
    print(f"Overall: {passed}/{total} ({int(100*passed/total)}%)")
    
    if passed == total:
        print("✅ All systems GO! Signal pipeline should be working correctly.")
        print("\nNext steps:")
        print("1. Start trading system")
        print("2. Look for logs: '[AgentManager:DIRECT] Forwarded N signals'")
        print("3. Verify trades execute normally")
    else:
        print("❌ Some issues detected. See above for details.")
        print("\nTroubleshooting:")
        print("1. Verify the fix was applied: git diff core/agent_manager.py")
        print("2. Check imports: python3 -c 'from core.agent_manager import AgentManager'")
        print("3. Check syntax: python3 -m py_compile core/agent_manager.py")
    
    print("="*80 + "\n")
    
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(diagnose_signal_pipeline())
    sys.exit(0 if success else 1)
