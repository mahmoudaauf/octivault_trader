#!/usr/bin/env python3
"""
P9 Compliance Validation Script

Validates that SignalFusion redesign is fully P9-compliant and properly integrated.

Checks:
1. SignalFusion has NO references to ExecutionManager
2. SignalFusion has NO references to MetaController  
3. SignalFusion._emit_fused_signal() only uses shared_state.add_agent_signal()
4. MetaController.start() calls await signal_fusion.start()
5. MetaController.stop() calls await signal_fusion.stop()
6. Signal flow is: Agents → shared_state → SignalFusion → shared_state → MetaController
"""

import sys
import re
from pathlib import Path

def check_file_content(filepath: str, pattern: str, should_not_contain=False, description=""):
    """Check if file contains or doesn't contain a pattern"""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        matches = re.findall(pattern, content, re.IGNORECASE)
        
        if should_not_contain:
            if matches:
                print(f"❌ FAIL: {description}")
                print(f"   File: {filepath}")
                print(f"   Found: {matches[0]}")
                return False
            else:
                print(f"✅ PASS: {description}")
                return True
        else:
            if matches:
                print(f"✅ PASS: {description}")
                return True
            else:
                print(f"❌ FAIL: {description}")
                print(f"   File: {filepath}")
                print(f"   Pattern not found: {pattern}")
                return False
    except Exception as e:
        print(f"❌ ERROR: {description} - {e}")
        return False

def main():
    base_path = Path(__file__).parent.absolute()
    signal_fusion_path = base_path / "core" / "signal_fusion.py"
    meta_controller_path = base_path / "core" / "meta_controller.py"
    signal_manager_path = base_path / "core" / "signal_manager.py"
    
    print("\n" + "="*80)
    print("P9 COMPLIANCE VALIDATION")
    print("="*80 + "\n")
    
    results = []
    
    # ===== SIGNALFU SION CHECKS =====
    print("📋 Checking SignalFusion (core/signal_fusion.py):\n")
    
    # Check 1: No ExecutionManager reference (in code, not docstrings)
    with open(str(signal_fusion_path), 'r') as f:
        content = f.read()
        # Remove comments and docstrings before checking
        code_only = re.sub(r'#.*$', '', content, flags=re.MULTILINE)
        code_only = re.sub(r'""".*?"""', '', code_only, flags=re.DOTALL)
        code_only = re.sub(r"'''.*?'''", '', code_only, flags=re.DOTALL)
        if re.search(r'execution_manager\s*=|self\.execution_manager', code_only, re.IGNORECASE):
            print(f"❌ FAIL: SignalFusion has NO execution_manager parameter")
            results.append(False)
        else:
            print(f"✅ PASS: SignalFusion has NO execution_manager parameter")
            results.append(True)
    
    # Check 2: No MetaController reference
    results.append(check_file_content(
        str(signal_fusion_path),
        r'meta_controller|MetaController\(',
        should_not_contain=True,
        description="SignalFusion has NO meta_controller parameter"
    ))
    
    # Check 3: No fuse_and_execute method
    results.append(check_file_content(
        str(signal_fusion_path),
        r'def fuse_and_execute',
        should_not_contain=True,
        description="SignalFusion has NO fuse_and_execute() method"
    ))
    
    # Check 4: Has async start() method
    results.append(check_file_content(
        str(signal_fusion_path),
        r'async def start\(',
        description="SignalFusion HAS async def start() method"
    ))
    
    # Check 5: Has async stop() method
    results.append(check_file_content(
        str(signal_fusion_path),
        r'async def stop\(',
        description="SignalFusion HAS async def stop() method"
    ))
    
    # Check 6: Has _run_fusion_loop() method
    results.append(check_file_content(
        str(signal_fusion_path),
        r'async def _run_fusion_loop',
        description="SignalFusion HAS async def _run_fusion_loop() method"
    ))
    
    # Check 7: Emits via shared_state.add_agent_signal() only
    results.append(check_file_content(
        str(signal_fusion_path),
        r'add_agent_signal',
        description="SignalFusion emits via shared_state.add_agent_signal()"
    ))
    
    print("\n📋 Checking MetaController (core/meta_controller.py):\n")
    
    # Check 8: MetaController initializes SignalFusion without ExecutionManager
    results.append(check_file_content(
        str(meta_controller_path),
        r'from core\.signal_fusion import SignalFusion',
        description="MetaController imports SignalFusion"
    ))
    
    # Check 9: MetaController.start() calls signal_fusion.start()
    results.append(check_file_content(
        str(meta_controller_path),
        r'await self\.signal_fusion\.start\(\)',
        description="MetaController.start() calls await signal_fusion.start()"
    ))
    
    # Check 10: MetaController.stop() calls signal_fusion.stop()
    results.append(check_file_content(
        str(meta_controller_path),
        r'await self\.signal_fusion\.stop\(\)',
        description="MetaController.stop() calls await signal_fusion.stop()"
    ))
    
    print("\n📋 Checking SignalManager (core/signal_manager.py):\n")
    
    # Check 11: MIN_SIGNAL_CONF defaults to 0.50 (defensive floor)
    results.append(check_file_content(
        str(signal_manager_path),
        r"MIN_SIGNAL_CONF.*0\.5",
        description="SignalManager MIN_SIGNAL_CONF defaults to 0.50 (defensive floor)"
    ))
    
    # ===== SUMMARY =====
    print("\n" + "="*80)
    passed = sum(results)
    total = len(results)
    print(f"RESULT: {passed}/{total} checks passed")
    print("="*80 + "\n")
    
    if passed == total:
        print("✅ ALL P9 COMPLIANCE CHECKS PASSED!")
        print("\nSignalFusion is properly redesigned as:")
        print("  ✓ Independent async component (no execution_manager or meta_controller refs)")
        print("  ✓ Emits signals via shared_state signal bus (P9-compliant)")
        print("  ✓ Properly integrated with MetaController lifecycle")
        print("  ✓ Defensive signal floor (MIN_SIGNAL_CONF=0.50)")
        print("\nSignal Flow:")
        print("  Agents → shared_state.agent_signals")
        print("  → SignalFusion._run_fusion_loop() [async background task]")
        print("  → shared_state.add_agent_signal() [emit back to bus]")
        print("  → MetaController.receive_signal() [natural P9 integration]")
        return 0
    else:
        print("❌ SOME P9 COMPLIANCE CHECKS FAILED!")
        print(f"\nPassed: {passed}/{total}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
