#!/usr/bin/env python3
"""
PHASE 2 BOTTLENECK FIXES - VERIFICATION SCRIPT

Verifies that all three fixes have been properly implemented:
1. Recovery Exit Min-Hold Bypass
2. Micro Rotation Override
3. Entry-Sizing Config Alignment

Run this AFTER implementing the fixes to confirm deployment readiness.
"""

import os
import sys
import re
from pathlib import Path

# Colors for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'

def print_header(text):
    print(f"\n{BLUE}{BOLD}{'='*70}{RESET}")
    print(f"{BLUE}{BOLD}{text}{RESET}")
    print(f"{BLUE}{BOLD}{'='*70}{RESET}\n")

def print_success(text):
    print(f"{GREEN}✅ {text}{RESET}")

def print_error(text):
    print(f"{RED}❌ {text}{RESET}")

def print_warning(text):
    print(f"{YELLOW}⚠️  {text}{RESET}")

def print_info(text):
    print(f"{BLUE}ℹ️  {text}{RESET}")

def check_file_exists(filepath):
    """Check if file exists"""
    exists = os.path.exists(filepath)
    if exists:
        print_success(f"File found: {filepath}")
    else:
        print_error(f"File NOT found: {filepath}")
    return exists

def read_file(filepath):
    """Read file content"""
    try:
        with open(filepath, 'r') as f:
            return f.read()
    except Exception as e:
        print_error(f"Failed to read {filepath}: {e}")
        return None

def check_pattern_in_file(filepath, pattern, description):
    """Check if pattern exists in file"""
    content = read_file(filepath)
    if content is None:
        return False
    
    if isinstance(pattern, str):
        found = pattern in content
    else:  # regex pattern
        found = re.search(pattern, content) is not None
    
    if found:
        print_success(description)
    else:
        print_error(description)
    
    return found

# ============================================================================
# VERIFICATION FUNCTIONS
# ============================================================================

def verify_fix_1():
    """Verify Fix #1: Recovery Exit Min-Hold Bypass"""
    print_header("FIX #1: Recovery Exit Min-Hold Bypass")
    
    filepath = "core/meta_controller.py"
    results = []
    
    # Check 1.1: _safe_passes_min_hold has bypass parameter
    results.append(check_pattern_in_file(
        filepath,
        r"def _safe_passes_min_hold\(self.*bypass:\s*bool\s*=\s*False\)",
        "Check 1.1: _safe_passes_min_hold() has bypass: bool=False parameter"
    ))
    
    # Check 1.2: Bypass logic in function
    results.append(check_pattern_in_file(
        filepath,
        r"if bypass:\s+return True",
        "Check 1.2: Bypass logic returns True when bypass=True"
    ))
    
    # Check 1.3: Stagnation exit sets flag
    results.append(check_pattern_in_file(
        filepath,
        r'stagnation_exit_sig\["_bypass_min_hold"\]\s*=\s*True',
        "Check 1.3: Stagnation exit sets _bypass_min_hold flag"
    ))
    
    # Check 1.4: Liquidity restore sets flag
    results.append(check_pattern_in_file(
        filepath,
        r'liquidity_restore_sig\["_bypass_min_hold"\]\s*=\s*True',
        "Check 1.4: Liquidity restore exit sets _bypass_min_hold flag"
    ))
    
    # Check 1.5: Liquidity restore calls with bypass=True
    results.append(check_pattern_in_file(
        filepath,
        r'self\._safe_passes_min_hold\(.*bypass=True\)',
        "Check 1.5: Liquidity restore calls _safe_passes_min_hold with bypass=True"
    ))
    
    passed = sum(results)
    total = len(results)
    print_info(f"\nFix #1 Score: {passed}/{total}")
    return all(results)

def verify_fix_2():
    """Verify Fix #2: Micro Rotation Override"""
    print_header("FIX #2: Micro Rotation Override")
    
    filepath = "core/rotation_authority.py"
    results = []
    
    # Check 2.1: authorize_rotation has force_rotation parameter
    results.append(check_pattern_in_file(
        filepath,
        r"async def authorize_rotation\(.*force_rotation:\s*bool\s*=\s*False",
        "Check 2.1: authorize_rotation() has force_rotation: bool=False parameter"
    ))
    
    # Check 2.2: Force rotation precedence logic exists
    results.append(check_pattern_in_file(
        filepath,
        r"if owned_positions and not force_rotation:",
        "Check 2.2: Precedence check: if NOT force_rotation, apply MICRO restriction"
    ))
    
    # Check 2.3: Override branch exists
    results.append(check_pattern_in_file(
        filepath,
        r"elif owned_positions and force_rotation:",
        "Check 2.3: Override branch exists: elif force_rotation, override restriction"
    ))
    
    # Check 2.4: Override logging with emoji
    results.append(check_pattern_in_file(
        filepath,
        r"⚠️\s+MICRO restriction OVERRIDDEN",
        "Check 2.4: Override logging includes ⚠️ MICRO restriction OVERRIDDEN message"
    ))
    
    # Check 2.5: Precedence documentation
    results.append(check_pattern_in_file(
        filepath,
        r"PRECEDENCE:.*force_rotation",
        "Check 2.5: Precedence documentation added to method docstring"
    ))
    
    passed = sum(results)
    total = len(results)
    print_info(f"\nFix #2 Score: {passed}/{total}")
    return all(results)

def verify_fix_3():
    """Verify Fix #3: Entry-Sizing Config Alignment"""
    print_header("FIX #3: Entry-Sizing Config Alignment")
    
    filepath = ".env"
    results = []
    
    # Check all 7 entry-size parameters
    checks = [
        ("DEFAULT_PLANNED_QUOTE=25", "DEFAULT_PLANNED_QUOTE set to 25 USDT"),
        ("MIN_TRADE_QUOTE=25", "MIN_TRADE_QUOTE set to 25 USDT"),
        ("MIN_ENTRY_USDT=25", "MIN_ENTRY_USDT set to 25 USDT"),
        ("TRADE_AMOUNT_USDT=25", "TRADE_AMOUNT_USDT set to 25 USDT"),
        ("MIN_ENTRY_QUOTE_USDT=25", "MIN_ENTRY_QUOTE_USDT set to 25 USDT"),
        ("EMIT_BUY_QUOTE=25", "EMIT_BUY_QUOTE set to 25 USDT"),
        ("META_MICRO_SIZE_USDT=25", "META_MICRO_SIZE_USDT set to 25 USDT"),
    ]
    
    for i, (pattern, description) in enumerate(checks, 1):
        results.append(check_pattern_in_file(
            filepath,
            pattern,
            f"Check 3.{i}: {description}"
        ))
    
    # Check 3.8: Floor alignment comment
    results.append(check_pattern_in_file(
        filepath,
        r"Aligned with SIGNIFICANT_POSITION_FLOOR.*25 USDT",
        "Check 3.8: Floor alignment comment present in .env"
    ))
    
    # Check 3.9: FIX #3 marker in .env
    results.append(check_pattern_in_file(
        filepath,
        r"FIX #3:.*Entry-Sizing Config Alignment",
        "Check 3.9: FIX #3 marker added to .env"
    ))
    
    # Check 3.10: MIN_SIGNIFICANT_POSITION_USDT also aligned
    results.append(check_pattern_in_file(
        filepath,
        r"MIN_SIGNIFICANT_POSITION_USDT=25",
        "Check 3.10: MIN_SIGNIFICANT_POSITION_USDT set to 25 USDT"
    ))
    
    passed = sum(results)
    total = len(results)
    print_info(f"\nFix #3 Score: {passed}/{total}")
    return all(results)

def verify_compilation():
    """Verify that Python files compile cleanly"""
    print_header("COMPILATION & IMPORT CHECKS")
    
    results = []
    
    # Check core/meta_controller.py compiles
    try:
        import py_compile
        py_compile.compile("core/meta_controller.py", doraise=True)
        print_success("core/meta_controller.py compiles cleanly")
        results.append(True)
    except Exception as e:
        print_error(f"core/meta_controller.py compilation failed: {e}")
        results.append(False)
    
    # Check core/rotation_authority.py compiles
    try:
        py_compile.compile("core/rotation_authority.py", doraise=True)
        print_success("core/rotation_authority.py compiles cleanly")
        results.append(True)
    except Exception as e:
        print_error(f"core/rotation_authority.py compilation failed: {e}")
        results.append(False)
    
    return all(results)

def generate_summary(results):
    """Generate verification summary"""
    print_header("VERIFICATION SUMMARY")
    
    fix1, fix2, fix3, compilation = results
    
    checks_passed = sum([fix1, fix2, fix3, compilation])
    checks_total = len(results)
    
    if checks_passed == checks_total:
        print_success(f"ALL CHECKS PASSED ({checks_passed}/{checks_total})")
        print_info("\n✨ System is ready for deployment!")
        return True
    else:
        print_warning(f"INCOMPLETE VERIFICATION ({checks_passed}/{checks_total} passed)")
        print_info("\nPlease review the failures above and implement the required fixes.")
        return False

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    print(f"\n{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}PHASE 2 BOTTLENECK FIXES - VERIFICATION SCRIPT{RESET}")
    print(f"{BOLD}{'='*70}{RESET}")
    
    # Run all verifications
    fix1_ok = verify_fix_1()
    fix2_ok = verify_fix_2()
    fix3_ok = verify_fix_3()
    compilation_ok = verify_compilation()
    
    # Generate summary
    all_ok = generate_summary([fix1_ok, fix2_ok, fix3_ok, compilation_ok])
    
    # Print next steps
    if all_ok:
        print(f"\n{BLUE}{BOLD}NEXT STEPS:{RESET}")
        print("1. Deploy to staging environment")
        print("2. Run 30-minute warm-up test")
        print("3. Monitor logs for expected patterns:")
        print("   - [Meta:SafeMinHold] Bypassing min-hold check for forced recovery exit")
        print("   - [REA:authorize_rotation] ⚠️ MICRO restriction OVERRIDDEN")
        print("   - Entry orders with ~25 USDT size")
        print("4. Deploy to production if warm-up passes")
    else:
        print(f"\n{YELLOW}{BOLD}REMEDIATION REQUIRED:{RESET}")
        print("Please review the failed checks above and implement the required fixes.")
        print("Then run this verification script again.")
    
    sys.exit(0 if all_ok else 1)
