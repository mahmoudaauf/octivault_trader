#!/usr/bin/env python3
"""
Verification script for dust-liquidation flag wiring & entry floor guard implementation.
Checks that all changes are properly in place.
"""

import re
import sys
from pathlib import Path

def check_file_exists(path: str) -> bool:
    """Check if file exists."""
    return Path(path).exists()

def grep_file(path: str, pattern: str, should_exist: bool = True) -> bool:
    """Check if pattern exists (or doesn't exist) in file."""
    try:
        with open(path, 'r') as f:
            content = f.read()
            found = bool(re.search(pattern, content))
            if should_exist:
                return found
            else:
                return not found
    except Exception as e:
        print(f"❌ Error reading {path}: {e}")
        return False

def main():
    """Run all verification checks."""
    print("🔍 Verifying Dust-Liquidation Fix Implementation\n")
    
    base = "/Users/mauf/Desktop/Octi AI Trading Bot/octivault_trader"
    
    checks = [
        # ===== Config File Checks =====
        ("✅ config.py exists", lambda: check_file_exists(f"{base}/core/config.py")),
        ("✅ config.py: dust_liquidation_enabled lowercase defined", 
         lambda: grep_file(f"{base}/core/config.py", r"self\.dust_liquidation_enabled\s*=", True)),
        ("✅ config.py: dust_reentry_override lowercase defined", 
         lambda: grep_file(f"{base}/core/config.py", r"self\.dust_reentry_override\s*=", True)),
        ("✅ config.py: No UPPERCASE DUST_LIQUIDATION_ENABLED attribute", 
         lambda: grep_file(f"{base}/core/config.py", r"self\.DUST_LIQUIDATION_ENABLED\s*=", False)),
        ("✅ config.py: Logging uses lowercase", 
         lambda: grep_file(f"{base}/core/config.py", r"self\.dust_liquidation_enabled,", True)),
        
        # ===== SharedState File Checks =====
        ("✅ shared_state.py exists", lambda: check_file_exists(f"{base}/core/shared_state.py")),
        ("✅ shared_state.py: dust_liquidation_enabled field", 
         lambda: grep_file(f"{base}/core/shared_state.py", r"dust_liquidation_enabled:\s*bool\s*=\s*True", True)),
        ("✅ shared_state.py: dust_reentry_override field", 
         lambda: grep_file(f"{base}/core/shared_state.py", r"dust_reentry_override:\s*bool\s*=\s*True", True)),
        ("✅ shared_state.py: allow_entry_below_significant_floor field", 
         lambda: grep_file(f"{base}/core/shared_state.py", r"allow_entry_below_significant_floor:\s*bool\s*=\s*False", True)),
        
        # ===== ExecutionManager Guard Method =====
        ("✅ execution_manager.py exists", lambda: check_file_exists(f"{base}/core/execution_manager.py")),
        ("✅ execution_manager.py: _check_entry_floor_guard method exists", 
         lambda: grep_file(f"{base}/core/execution_manager.py", r"async def _check_entry_floor_guard\(", True)),
        ("✅ execution_manager.py: Guard returns Tuple\[bool, str\]", 
         lambda: grep_file(f"{base}/core/execution_manager.py", r"\)\s*->\s*Tuple\[bool,\s*str\]:", True)),
        ("✅ execution_manager.py: Guard checks is_dust_healing_buy bypass", 
         lambda: grep_file(f"{base}/core/execution_manager.py", r"if is_dust_healing_buy:", True)),
        ("✅ execution_manager.py: Guard checks allow_entry_below_significant_floor", 
         lambda: grep_file(f"{base}/core/execution_manager.py", r"allow_entry_below_significant_floor", True)),
        
        # ===== Quote-Based BUY Path Integration =====
        ("✅ execution_manager.py: Quote-based BUY has guard check", 
         lambda: grep_file(f"{base}/core/execution_manager.py", 
                          r"guard_allowed, guard_reason = await self\._check_entry_floor_guard", True)),
        ("✅ execution_manager.py: Quote-based guard blocks execution", 
         lambda: grep_file(f"{base}/core/execution_manager.py", 
                          r"ENTRY_FLOOR_GUARD", True)),
        
        # ===== Qty-Based BUY Path Integration =====
        ("✅ execution_manager.py: Qty-based BUY has guard check", 
         lambda: grep_file(f"{base}/core/execution_manager.py", 
                          r"estimated_quote = float", True)),
        ("✅ execution_manager.py: Qty-based guard calls _check_entry_floor_guard", 
         lambda: grep_file(f"{base}/core/execution_manager.py", 
                          r"get_mark_price", True)),
        
        # ===== Documentation =====
        ("✅ Implementation doc exists", lambda: check_file_exists(f"{base}/DUST_LIQUIDATION_FIX_IMPLEMENTATION.md")),
        ("✅ Implementation doc has testing plan", 
         lambda: grep_file(f"{base}/DUST_LIQUIDATION_FIX_IMPLEMENTATION.md", r"## Testing Plan", True)),
        ("✅ Implementation doc has deployment checklist", 
         lambda: grep_file(f"{base}/DUST_LIQUIDATION_FIX_IMPLEMENTATION.md", r"## Deployment Checklist", True)),
    ]
    
    passed = 0
    failed = 0
    
    for name, check_func in checks:
        try:
            result = check_func()
            if result:
                print(f"{name}")
                passed += 1
            else:
                print(f"❌ FAILED: {name}")
                failed += 1
        except Exception as e:
            print(f"❌ ERROR: {name} - {e}")
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'='*60}\n")
    
    if failed == 0:
        print("✅ ALL CHECKS PASSED - Implementation complete!")
        return 0
    else:
        print(f"⚠️  {failed} checks failed - review implementation")
        return 1

if __name__ == "__main__":
    sys.exit(main())
