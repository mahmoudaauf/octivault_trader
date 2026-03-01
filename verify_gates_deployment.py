#!/usr/bin/env python3
"""
Quick deployment verification script for protective gates.
Verifies that all three gates are implemented and callable.
"""

import sys
import ast
import asyncio
from pathlib import Path


def verify_gate_implementation(file_path: str) -> dict:
    """Verify that all three gates are implemented in the file."""
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    results = {
        'syntax_valid': False,
        'gate1_implemented': False,
        'gate2_implemented': False,
        'gate3_implemented': False,
        'gate1_integrated': False,
        'gate2_integrated': False,
        'gate3_integrated': False,
        'imports_present': False,
        'errors': []
    }
    
    # Check syntax
    try:
        ast.parse(content)
        results['syntax_valid'] = True
    except SyntaxError as e:
        results['errors'].append(f"Syntax Error: {e}")
        return results
    
    # Check imports
    if 'import asyncio' in content and 'import numpy as np' in content:
        results['imports_present'] = True
    else:
        results['errors'].append("Missing required imports (asyncio, numpy)")
    
    # Check Gate 1
    if 'async def _validate_volatility_gate(self, symbol: str) -> bool:' in content:
        results['gate1_implemented'] = True
        if 'await self._validate_volatility_gate(symbol)' in content:
            results['gate1_integrated'] = True
    else:
        results['errors'].append("Gate 1 (_validate_volatility_gate) not found")
    
    # Check Gate 2
    if 'async def _validate_edge_gate(self, symbol: str) -> bool:' in content:
        results['gate2_implemented'] = True
        if 'await self._validate_edge_gate(symbol)' in content:
            results['gate2_integrated'] = True
    else:
        results['errors'].append("Gate 2 (_validate_edge_gate) not found")
    
    # Check Gate 3
    if 'async def _validate_economic_gate(self, amount: float, num_symbols: int) -> bool:' in content:
        results['gate3_implemented'] = True
        if 'await self._validate_economic_gate(spendable' in content or \
           'await self._validate_economic_gate(amount' in content:
            results['gate3_integrated'] = True
    else:
        results['errors'].append("Gate 3 (_validate_economic_gate) not found")
    
    return results


def print_verification_report(results: dict, file_path: str) -> None:
    """Print a formatted verification report."""
    
    print("\n" + "="*80)
    print("🔍 PROTECTIVE GATES DEPLOYMENT VERIFICATION")
    print("="*80)
    
    print(f"\nFile: {file_path}")
    
    # Syntax check
    status = "✅ PASS" if results['syntax_valid'] else "❌ FAIL"
    print(f"\n{status} Syntax Validation")
    
    # Imports check
    status = "✅ PASS" if results['imports_present'] else "❌ FAIL"
    print(f"{status} Required Imports")
    
    # Gate 1
    impl_status = "✅" if results['gate1_implemented'] else "❌"
    integ_status = "✅" if results['gate1_integrated'] else "❌"
    print(f"\n{impl_status} Gate 1: Volatility Validator")
    print(f"   Implementation: {impl_status} {'FOUND' if results['gate1_implemented'] else 'MISSING'}")
    print(f"   Integration: {integ_status} {'INTEGRATED' if results['gate1_integrated'] else 'NOT INTEGRATED'}")
    
    # Gate 2
    impl_status = "✅" if results['gate2_implemented'] else "❌"
    integ_status = "✅" if results['gate2_integrated'] else "❌"
    print(f"\n{impl_status} Gate 2: Edge Validator")
    print(f"   Implementation: {impl_status} {'FOUND' if results['gate2_implemented'] else 'MISSING'}")
    print(f"   Integration: {integ_status} {'INTEGRATED' if results['gate2_integrated'] else 'NOT INTEGRATED'}")
    
    # Gate 3
    impl_status = "✅" if results['gate3_implemented'] else "❌"
    integ_status = "✅" if results['gate3_integrated'] else "❌"
    print(f"\n{impl_status} Gate 3: Economic Validator")
    print(f"   Implementation: {impl_status} {'FOUND' if results['gate3_implemented'] else 'MISSING'}")
    print(f"   Integration: {integ_status} {'INTEGRATED' if results['gate3_integrated'] else 'NOT INTEGRATED'}")
    
    # Overall status
    all_pass = all([
        results['syntax_valid'],
        results['imports_present'],
        results['gate1_implemented'],
        results['gate1_integrated'],
        results['gate2_implemented'],
        results['gate2_integrated'],
        results['gate3_implemented'],
        results['gate3_integrated']
    ])
    
    print("\n" + "="*80)
    if all_pass:
        print("✅ VERIFICATION PASSED - READY FOR DEPLOYMENT")
    else:
        print("❌ VERIFICATION FAILED - ISSUES DETECTED")
        if results['errors']:
            print("\nErrors:")
            for error in results['errors']:
                print(f"  • {error}")
    print("="*80 + "\n")
    
    return all_pass


def main():
    """Main verification function."""
    
    file_path = "core/compounding_engine.py"
    
    # Check if file exists
    if not Path(file_path).exists():
        print(f"❌ Error: {file_path} not found")
        sys.exit(1)
    
    # Run verification
    results = verify_gate_implementation(file_path)
    all_pass = print_verification_report(results, file_path)
    
    # Return appropriate exit code
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
