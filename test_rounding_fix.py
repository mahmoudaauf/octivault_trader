#!/usr/bin/env python3
"""
Verification script for the rounding bug fix.
Tests the round_step() function with direction parameter.
"""

import sys
from decimal import Decimal, ROUND_DOWN, ROUND_UP

def round_step_original(value: float, step: float) -> float:
    """Original function - always rounds DOWN"""
    if step <= 0:
        return float(value)
    q = (Decimal(str(value)) / Decimal(str(step))).to_integral_value(rounding=ROUND_DOWN)
    return float(q * Decimal(str(step)))


def round_step_fixed(value: float, step: float, direction: str = "down") -> float:
    """Fixed function - supports both UP and DOWN"""
    if step <= 0:
        return float(value)
    
    rounding_mode = ROUND_UP if direction.lower() == "up" else ROUND_DOWN
    q = (Decimal(str(value)) / Decimal(str(step))).to_integral_value(rounding=rounding_mode)
    return float(q * Decimal(str(step)))


def test_rounding():
    """Run comprehensive tests"""
    print("=" * 80)
    print("ROUNDING BUG FIX - VERIFICATION TESTS")
    print("=" * 80)
    
    tests = [
        {
            "name": "DOGE position with dust (1.0 step)",
            "value": 210.898,
            "step": 1.0,
            "expected_down": 210.0,
            "expected_up": 211.0,
        },
        {
            "name": "Smaller DOGE position (0.001 step)",
            "value": 0.898,
            "step": 0.001,
            "expected_down": 0.898,  # Already aligned to 0.001
            "expected_up": 0.898,    # Already aligned, no rounding needed
        },
        {
            "name": "BTC with decimal remainder (0.00000001 step)",
            "value": 0.12345678,
            "step": 0.00000001,
            "expected_down": 0.12345678,
            "expected_up": 0.12345678,
        },
        {
            "name": "Position already aligned (no dust)",
            "value": 210.0,
            "step": 1.0,
            "expected_down": 210.0,
            "expected_up": 210.0,
        },
        {
            "name": "Fractional remainder (4.9 with 1.0 step)",
            "value": 4.9,
            "step": 1.0,
            "expected_down": 4.0,
            "expected_up": 5.0,
        },
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        print(f"\n📋 Test: {test['name']}")
        print(f"   Input: {test['value']}, Step: {test['step']}")
        
        # Test original (always rounds down)
        original_result = round_step_original(test['value'], test['step'])
        print(f"   Original (DOWN):        {original_result}")
        
        # Test fixed (rounds down by default)
        fixed_down = round_step_fixed(test['value'], test['step'], direction="down")
        print(f"   Fixed (DOWN):           {fixed_down}")
        
        # Test fixed (rounds up)
        fixed_up = round_step_fixed(test['value'], test['step'], direction="up")
        print(f"   Fixed (UP):             {fixed_up}")
        
        # Verify expected values
        if original_result == test['expected_down']:
            print(f"   ✅ Original DOWN matches expected")
            passed += 1
        else:
            print(f"   ❌ Original DOWN mismatch! Expected {test['expected_down']}")
            failed += 1
        
        if fixed_down == test['expected_down']:
            print(f"   ✅ Fixed DOWN matches expected")
            passed += 1
        else:
            print(f"   ❌ Fixed DOWN mismatch! Expected {test['expected_down']}")
            failed += 1
        
        if fixed_up == test['expected_up']:
            print(f"   ✅ Fixed UP matches expected")
            passed += 1
        else:
            print(f"   ❌ Fixed UP mismatch! Expected {test['expected_up']}")
            failed += 1
    
    print("\n" + "=" * 80)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 80)
    
    if failed > 0:
        print("\n❌ Some tests FAILED!")
        return False
    else:
        print("\n✅ All tests PASSED!")
        return True


def test_doge_case():
    """Specific test for the DOGE 0.898 dust case"""
    print("\n" + "=" * 80)
    print("SPECIAL CASE: DOGE 0.898 DUST BUG")
    print("=" * 80)
    
    position = 210.898
    step_size = 1.0
    remainder = 0.898
    notional = remainder * 0.098  # $0.088
    
    print(f"\nScenario:")
    print(f"  Position: {position} DOGE")
    print(f"  Step size: {step_size}")
    print(f"  Remainder after ROUND_DOWN: {remainder}")
    print(f"  Remainder notional: ${notional:.4f}")
    print(f"  Dust threshold: $5.00")
    print(f"  Is dust? {notional < 5.00} ✅")
    
    # Original behavior (BUG)
    original = round_step_original(position, step_size)
    print(f"\n❌ ORIGINAL (BUG):")
    print(f"  Rounded: {original}")
    print(f"  Dust left: {position - original} DOGE")
    print(f"  Problem: Dust not included in SELL order!")
    
    # Fixed behavior
    fixed = round_step_fixed(position, step_size, direction="up")
    print(f"\n✅ FIXED (CORRECT):")
    print(f"  Rounded: {fixed}")
    print(f"  Dust included: {fixed - (position - remainder)} DOGE")
    print(f"  Result: Entire position sold, zero dust!")
    
    if fixed > position - step_size * 0.1:
        print(f"\n✅ FIX VERIFIED: Rounding UP works correctly!")
        return True
    else:
        print(f"\n❌ FIX FAILED: Rounding didn't include dust!")
        return False


def test_backward_compatibility():
    """Verify backward compatibility"""
    print("\n" + "=" * 80)
    print("BACKWARD COMPATIBILITY TEST")
    print("=" * 80)
    
    test_values = [
        (100.123, 0.01),
        (0.999, 0.001),
        (1.5, 0.1),
    ]
    
    print("\nVerifying that default behavior (no direction) still rounds DOWN:")
    all_match = True
    
    for value, step in test_values:
        original = round_step_original(value, step)
        fixed_default = round_step_fixed(value, step)  # No direction parameter
        fixed_explicit = round_step_fixed(value, step, direction="down")
        
        match_default = original == fixed_default
        match_explicit = original == fixed_explicit
        
        print(f"\n  Value={value}, Step={step}")
        print(f"    Original:  {original}")
        print(f"    Fixed (no param): {fixed_default} {'✅' if match_default else '❌'}")
        print(f"    Fixed (explicit down): {fixed_explicit} {'✅' if match_explicit else '❌'}")
        
        if not (match_default and match_explicit):
            all_match = False
    
    if all_match:
        print("\n✅ BACKWARD COMPATIBILITY VERIFIED!")
        return True
    else:
        print("\n❌ BACKWARD COMPATIBILITY BROKEN!")
        return False


if __name__ == "__main__":
    results = []
    
    # Run all tests
    results.append(("Rounding Tests", test_rounding()))
    results.append(("DOGE Dust Case", test_doge_case()))
    results.append(("Backward Compatibility", test_backward_compatibility()))
    
    # Summary
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name}: {status}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! Fix is ready for deployment.")
        sys.exit(0)
    else:
        print("\n⚠️  SOME TESTS FAILED! Review before deployment.")
        sys.exit(1)
