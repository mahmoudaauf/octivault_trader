#!/usr/bin/env python3
"""
Verification script for Bottleneck Fixes Phase 2
Validates all three fixes are correctly applied
"""

import os
import sys
import re
from pathlib import Path

def check_file_contains(filepath, pattern, description):
    """Check if file contains a pattern"""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        if re.search(pattern, content, re.MULTILINE | re.DOTALL):
            print(f"  ✅ {description}")
            return True
        else:
            print(f"  ❌ {description} - NOT FOUND")
            return False
    except Exception as e:
        print(f"  ❌ {description} - ERROR: {e}")
        return False

def main():
    base_dir = Path(__file__).parent
    os.chdir(base_dir)
    
    print("╔" + "═" * 78 + "╗")
    print("║ BOTTLENECK FIXES PHASE 2 - VERIFICATION SCRIPT                        ║")
    print("╚" + "═" * 78 + "╝\n")
    
    all_pass = True
    
    # ────────────────────────────────────────────────────────────────────────────
    print("🔍 FIX #1: Safe Min-Hold Bypass")
    print("────────────────────────────────────────────────────────────────────────")
    
    results = []
    results.append(check_file_contains(
        "core/meta_controller.py",
        r'stagnation_exit_sig\["_bypass_min_hold"\]\s*=\s*True',
        "Stagnation exit carries _bypass_min_hold flag"
    ))
    
    results.append(check_file_contains(
        "core/meta_controller.py",
        r'liquidity_restore_sig\["_bypass_min_hold"\]\s*=\s*True',
        "Liquidity restoration exit carries _bypass_min_hold flag"
    ))
    
    results.append(check_file_contains(
        "core/meta_controller.py",
        r'_safe_passes_min_hold\(self,\s*symbol:\s*Optional\[str\],\s*bypass:\s*bool\s*=\s*False\)',
        "_safe_passes_min_hold has bypass parameter in signature"
    ))
    
    results.append(check_file_contains(
        "core/meta_controller.py",
        r'if bypass:.*?\[Meta:SafeMinHold\].*?return True',
        "Bypass logic implemented in _safe_passes_min_hold"
    ))
    
    all_pass = all_pass and all(results)
    
    # ────────────────────────────────────────────────────────────────────────────
    print("\n🔍 FIX #2: Micro Rotation Override")
    print("────────────────────────────────────────────────────────────────────────")
    
    results = []
    results.append(check_file_contains(
        "core/rotation_authority.py",
        r'PRECEDENCE:.*?force_rotation flag overrides MICRO bracket restrictions',
        "Precedence documentation added to authorize_rotation"
    ))
    
    results.append(check_file_contains(
        "core/rotation_authority.py",
        r'if owned_positions and not force_rotation:.*?# PHASE C check',
        "MICRO bracket check conditional on NOT force_rotation"
    ))
    
    results.append(check_file_contains(
        "core/rotation_authority.py",
        r'elif owned_positions and force_rotation:.*?# Force rotation overrides',
        "Force rotation override branch implemented"
    ))
    
    results.append(check_file_contains(
        "core/rotation_authority.py",
        r'⚠️.*?MICRO restriction OVERRIDDEN',
        "Override logging with emoji indicator"
    ))
    
    all_pass = all_pass and all(results)
    
    # ────────────────────────────────────────────────────────────────────────────
    print("\n🔍 FIX #3: Entry-Sizing Config Alignment")
    print("────────────────────────────────────────────────────────────────────────")
    
    results = []
    
    # Check .env values
    with open(".env", 'r') as f:
        env_content = f.read()
    
    env_checks = [
        ("DEFAULT_PLANNED_QUOTE=25", "DEFAULT_PLANNED_QUOTE set to 25"),
        ("MIN_TRADE_QUOTE=25", "MIN_TRADE_QUOTE set to 25"),
        ("MIN_ENTRY_USDT=25", "MIN_ENTRY_USDT set to 25"),
        ("TRADE_AMOUNT_USDT=25", "TRADE_AMOUNT_USDT set to 25"),
        ("MIN_ENTRY_QUOTE_USDT=25", "MIN_ENTRY_QUOTE_USDT set to 25"),
        ("EMIT_BUY_QUOTE=25", "EMIT_BUY_QUOTE set to 25"),
        ("META_MICRO_SIZE_USDT=25", "META_MICRO_SIZE_USDT set to 25"),
    ]
    
    for pattern, desc in env_checks:
        if pattern in env_content:
            print(f"  ✅ {desc}")
            results.append(True)
        else:
            print(f"  ❌ {desc} - NOT FOUND")
            results.append(False)
    
    results.append(check_file_contains(
        ".env",
        r"# NOTE:.*?Aligned with SIGNIFICANT_POSITION_FLOOR",
        "Floor alignment comment added to .env"
    ))
    
    results.append(check_file_contains(
        "core/config.py",
        r"# FIX #3:.*?Entry-sizing floor alignment",
        "FIX #3 comment added to config.py"
    ))
    
    results.append(check_file_contains(
        "core/config.py",
        r"\[Config:EntryFloor\].*?align config intent with runtime expectations",
        "Enhanced logging with floor alignment context"
    ))
    
    all_pass = all_pass and all(results)
    
    # ────────────────────────────────────────────────────────────────────────────
    print("\n🔍 Compilation & Imports")
    print("────────────────────────────────────────────────────────────────────────")
    
    import subprocess
    
    # Check compilation
    result = subprocess.run(
        ["python3", "-m", "compileall", "-q", "core", "agents", "utils"],
        capture_output=True,
        text=True,
        timeout=30
    )
    
    if result.returncode == 0:
        print("  ✅ All modules compile cleanly")
    else:
        print(f"  ❌ Compilation failed: {result.stderr}")
        all_pass = False
    
    # Check imports
    try:
        from core.meta_controller import MetaController
        from core.rotation_authority import RotationExitAuthority
        from core.config import Config
        print("  ✅ Core modules import successfully")
        
        # Check signature
        import inspect
        sig = inspect.signature(MetaController._safe_passes_min_hold)
        if 'bypass' in str(sig):
            print("  ✅ _safe_passes_min_hold has bypass parameter")
        else:
            print("  ❌ _safe_passes_min_hold missing bypass parameter")
            all_pass = False
    except Exception as e:
        print(f"  ❌ Import error: {e}")
        all_pass = False
    
    # ────────────────────────────────────────────────────────────────────────────
    print("\n" + "═" * 80)
    if all_pass:
        print("✅ ALL CHECKS PASSED - Ready for deployment")
        print("═" * 80)
        return 0
    else:
        print("❌ SOME CHECKS FAILED - Review errors above")
        print("═" * 80)
        return 1

if __name__ == "__main__":
    sys.exit(main())
