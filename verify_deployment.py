#!/usr/bin/env python3
"""
Live Deployment Verification Script
Checks all systems before and after PRODUCTION_STARTUP.py launch
"""

import sys
import os
from pathlib import Path
import json
from datetime import datetime


def verify_required_files():
    """Verify all required files exist"""
    print("\n📋 REQUIRED FILES CHECK")
    print("=" * 80)
    
    required = {
        "system_state_manager.py": "State persistence engine",
        "auto_recovery.py": "Auto-recovery system",
        "live_integration.py": "Live environment initializer",
        "PRODUCTION_STARTUP.py": "Production entry point",
        "core/meta_controller.py": "Trading engine (5 fixes)",
        "monitoring/sandbox_monitor.py": "Monitoring system"
    }
    
    all_exist = True
    for file, purpose in required.items():
        exists = Path(file).exists()
        status = "✅" if exists else "❌"
        print(f"{status} {file:40} ({purpose})")
        if not exists:
            all_exist = False
    
    return all_exist


def verify_state_directory():
    """Verify state directory setup"""
    print("\n📁 STATE DIRECTORY CHECK")
    print("=" * 80)
    
    state_dir = Path("state")
    
    # Create if doesn't exist
    state_dir.mkdir(exist_ok=True)
    print(f"✅ state/ directory: {state_dir.resolve()}")
    
    # Check permissions
    if os.access(state_dir, os.W_OK):
        print(f"✅ state/ directory: WRITABLE")
    else:
        print(f"❌ state/ directory: NOT WRITABLE")
        return False
    
    # List expected state files
    expected = [
        "operational_state.json",
        "session_memory.json",
        "checkpoint.json",
        "recovery_state.json",
        "context.json"
    ]
    
    print("\n   Expected state files:")
    for f in expected:
        state_file = state_dir / f
        if state_file.exists():
            size = state_file.stat().st_size
            print(f"   ✅ {f:40} ({size} bytes)")
        else:
            print(f"   ⏳ {f:40} (will be created)")
    
    return True


def verify_imports():
    """Verify all required modules can be imported"""
    print("\n📦 MODULE IMPORTS CHECK")
    print("=" * 80)
    
    modules = [
        "system_state_manager",
        "auto_recovery",
        "live_integration",
        "asyncio",
        "json",
        "pathlib"
    ]
    
    all_import = True
    for module in modules:
        try:
            __import__(module)
            print(f"✅ {module:30} - IMPORTABLE")
        except ImportError as e:
            print(f"❌ {module:30} - IMPORT FAILED: {e}")
            all_import = False
    
    return all_import


def verify_core_system():
    """Verify core system functionality"""
    print("\n🔧 CORE SYSTEM CHECK")
    print("=" * 80)
    
    try:
        from system_state_manager import SystemStateManager
        print("✅ SystemStateManager: LOADABLE")
        
        # Try to instantiate
        mgr = SystemStateManager()
        print("✅ SystemStateManager: INSTANTIABLE")
        
        # Get system context
        ctx = mgr.get_system_context()
        print(f"✅ System context: {ctx['system_status']['current_phase']}")
        
    except Exception as e:
        print(f"❌ SystemStateManager error: {e}")
        return False
    
    try:
        from auto_recovery import check_and_recover
        print("✅ Auto-recovery: LOADABLE")
        
        result = check_and_recover()
        restart_detected = result.get('restart_detected', False)
        recovered = result.get('recovered', False)
        ready = result.get('context', {}).get('recovery_info', {}).get('safe_to_continue', False)
        print(f"✅ Recovery check: restart_detected={restart_detected}, recovered={recovered}, safe_to_continue={ready}")
        
    except Exception as e:
        print(f"❌ Auto-recovery error: {e}")
        return False
    
    return True


def verify_production_startup():
    """Verify PRODUCTION_STARTUP.py"""
    print("\n🚀 PRODUCTION STARTUP CHECK")
    print("=" * 80)
    
    try:
        import PRODUCTION_STARTUP
        print("✅ PRODUCTION_STARTUP: IMPORTABLE")
        
        # Check if it has required functions
        if hasattr(PRODUCTION_STARTUP, 'startup_production'):
            print("✅ startup_production(): FOUND")
        else:
            print("❌ startup_production(): NOT FOUND")
            return False
        
        if hasattr(PRODUCTION_STARTUP, 'main_trading_loop'):
            print("✅ main_trading_loop(): FOUND")
        else:
            print("❌ main_trading_loop(): NOT FOUND")
            return False
            
    except Exception as e:
        print(f"❌ PRODUCTION_STARTUP error: {e}")
        return False
    
    return True


def generate_deployment_report():
    """Generate deployment readiness report"""
    print("\n" + "=" * 80)
    print("DEPLOYMENT VERIFICATION REPORT")
    print("=" * 80)
    
    checks = {
        "Required files": verify_required_files,
        "State directory": verify_state_directory,
        "Module imports": verify_imports,
        "Core system": verify_core_system,
        "Production startup": verify_production_startup
    }
    
    results = {}
    for name, check_fn in checks.items():
        try:
            results[name] = check_fn()
        except Exception as e:
            print(f"\n⚠️  Error in {name}: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "=" * 80)
    print("DEPLOYMENT READINESS SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"\nChecks passed: {passed}/{total}")
    print()
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status:12} {name}")
    
    print()
    
    if all(results.values()):
        print("🎉 ALL CHECKS PASSED - READY FOR DEPLOYMENT!")
        print()
        print("Next steps:")
        print("  1. Run: python3 PRODUCTION_STARTUP.py")
        print("  2. Monitor: watch -n 5 'ls -lh state/'")
        print("  3. Verify: cat state/checkpoint.json | python3 -m json.tool")
        print()
        return 0
    else:
        print("⚠️  SOME CHECKS FAILED - PLEASE REVIEW ABOVE")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(generate_deployment_report())
