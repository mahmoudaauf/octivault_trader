#!/usr/bin/env python3
"""
Phase B Integration Test: Capital Governor Position Limits

Tests:
1. Position limit blocking (max 1 in MICRO)
2. Position limit allowing (0 < max)
3. Bracket scaling (SMALL allows 2)
4. Position count helper accuracy

USAGE:
    python test_phase_b_integration.py
"""

import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.capital_governor import CapitalGovernor
from core.config import Config


def test_capital_governor_initialization():
    """Test 1: Capital Governor can be initialized."""
    print("\n" + "="*70)
    print("TEST 1: Capital Governor Initialization")
    print("="*70)
    
    try:
        config = Config()
        governor = CapitalGovernor(config)
        print("✅ Capital Governor initialized successfully")
        print(f"   Instance: {governor}")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize Capital Governor: {e}")
        return False


def test_position_limits_micro():
    """Test 2: MICRO bracket returns max 1 position."""
    print("\n" + "="*70)
    print("TEST 2: Position Limits - MICRO Bracket ($350)")
    print("="*70)
    
    try:
        config = Config()
        governor = CapitalGovernor(config)
        
        nav = 350.0  # MICRO bracket
        limits = governor.get_position_limits(nav)
        
        print(f"NAV: ${nav}")
        print(f"Bracket: {limits['bracket']}")
        print(f"Max concurrent positions: {limits['max_concurrent_positions']}")
        print(f"Max active symbols: {limits['max_active_symbols']}")
        
        # Assertions
        assert limits["bracket"] == "micro", f"Expected bracket 'micro', got '{limits['bracket']}'"
        assert limits["max_concurrent_positions"] == 1, f"Expected 1 position, got {limits['max_concurrent_positions']}"
        
        print("✅ MICRO bracket limits verified: 1 position max")
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def test_position_limits_small():
    """Test 3: SMALL bracket returns max 2 positions."""
    print("\n" + "="*70)
    print("TEST 3: Position Limits - SMALL Bracket ($1,500)")
    print("="*70)
    
    try:
        config = Config()
        governor = CapitalGovernor(config)
        
        nav = 1500.0  # SMALL bracket (< $2000)
        limits = governor.get_position_limits(nav)
        
        print(f"NAV: ${nav}")
        print(f"Bracket: {limits['bracket']}")
        print(f"Max concurrent positions: {limits['max_concurrent_positions']}")
        print(f"Max active symbols: {limits['max_active_symbols']}")
        
        # Assertions
        assert limits["bracket"] == "small", f"Expected bracket 'small', got '{limits['bracket']}'"
        assert limits["max_concurrent_positions"] == 2, f"Expected 2 positions, got {limits['max_concurrent_positions']}"
        
        print("✅ SMALL bracket limits verified: 2 positions max")
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def test_position_limits_medium():
    """Test 4: MEDIUM bracket returns max 3 positions."""
    print("\n" + "="*70)
    print("TEST 4: Position Limits - MEDIUM Bracket ($5,000)")
    print("="*70)
    
    try:
        config = Config()
        governor = CapitalGovernor(config)
        
        nav = 5000.0  # MEDIUM bracket
        limits = governor.get_position_limits(nav)
        
        print(f"NAV: ${nav}")
        print(f"Bracket: {limits['bracket']}")
        print(f"Max concurrent positions: {limits['max_concurrent_positions']}")
        print(f"Max active symbols: {limits['max_active_symbols']}")
        
        # Assertions
        assert limits["bracket"] == "medium", f"Expected bracket 'medium', got '{limits['bracket']}'"
        assert limits["max_concurrent_positions"] == 3, f"Expected 3 positions, got {limits['max_concurrent_positions']}"
        
        print("✅ MEDIUM bracket limits verified: 3 positions max")
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def test_bracket_boundaries():
    """Test 5: Verify bracket boundaries are correct."""
    print("\n" + "="*70)
    print("TEST 5: Bracket Boundary Verification")
    print("="*70)
    
    try:
        config = Config()
        governor = CapitalGovernor(config)
        
        test_cases = [
            (100.0, "micro"),     # Below $500
            (500.0, "small"),     # At $500 boundary (>= $500)
            (1000.0, "small"),    # In SMALL range
            (1999.0, "small"),    # Just below $2000
            (2000.0, "medium"),   # At $2000 boundary (>= $2000)
            (5000.0, "medium"),   # In MEDIUM range
            (9999.0, "medium"),   # Just below $10000
            (10000.0, "large"),   # At $10000 boundary (>= $10000)
            (50000.0, "large"),   # Large account
        ]
        
        results = []
        for nav, expected_bracket in test_cases:
            limits = governor.get_position_limits(nav)
            bracket = limits["bracket"]
            status = "✓" if bracket == expected_bracket else "✗"
            results.append((status, nav, bracket, expected_bracket))
            
            if bracket != expected_bracket:
                raise AssertionError(f"NAV ${nav}: expected '{expected_bracket}', got '{bracket}'")
        
        print(f"\n{'Status':<5} {'NAV':<10} {'Bracket':<10} {'Expected':<10}")
        print("-" * 35)
        for status, nav, bracket, expected in results:
            print(f"{status:<5} ${nav:<9.1f} {bracket:<10} {expected:<10}")
        
        print("\n✅ All bracket boundaries verified")
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def test_position_sizing_micro():
    """Test 6: MICRO bracket position sizing is correct."""
    print("\n" + "="*70)
    print("TEST 6: Position Sizing - MICRO Bracket")
    print("="*70)
    
    try:
        config = Config()
        governor = CapitalGovernor(config)
        
        nav = 350.0
        sizing = governor.get_position_sizing(nav)
        
        print(f"NAV: ${nav}")
        print(f"Quote per position: ${sizing['quote_per_position']}")
        print(f"EV multiplier: {sizing['ev_multiplier']}")
        print(f"Enable profit lock: {sizing['enable_profit_lock']}")
        
        # Assertions
        assert sizing["quote_per_position"] == 12.0, f"Expected $12, got ${sizing['quote_per_position']}"
        assert sizing["ev_multiplier"] == 1.4, f"Expected EV 1.4, got {sizing['ev_multiplier']}"
        assert sizing["enable_profit_lock"] == False, f"Expected no profit lock in MICRO"
        
        print("✅ MICRO position sizing verified: $12 per position, 1.4x EV")
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def test_rotation_restriction_micro():
    """Test 7: MICRO bracket restricts rotation."""
    print("\n" + "="*70)
    print("TEST 7: Rotation Restriction - MICRO Bracket")
    print("="*70)
    
    try:
        config = Config()
        governor = CapitalGovernor(config)
        
        nav = 350.0
        is_restricted = governor.should_restrict_rotation(nav)
        
        print(f"NAV: ${nav}")
        print(f"Rotation restricted: {is_restricted}")
        
        # Assertions
        assert is_restricted == True, f"Expected rotation restriction in MICRO"
        
        print("✅ MICRO rotation restriction verified: rotation disabled for focused learning")
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "="*70)
    print("PHASE B INTEGRATION TEST SUITE")
    print("Capital Governor Position Limits")
    print("="*70)
    
    tests = [
        ("Initialization", test_capital_governor_initialization),
        ("Position Limits - MICRO", test_position_limits_micro),
        ("Position Limits - SMALL", test_position_limits_small),
        ("Position Limits - MEDIUM", test_position_limits_medium),
        ("Bracket Boundaries", test_bracket_boundaries),
        ("Position Sizing", test_position_sizing_micro),
        ("Rotation Restriction", test_rotation_restriction_micro),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, "PASS" if passed else "FAIL"))
        except Exception as e:
            print(f"❌ Test '{name}' crashed: {e}")
            results.append((name, "FAIL"))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, status in results if status == "PASS")
    total = len(results)
    
    for name, status in results:
        symbol = "✅" if status == "PASS" else "❌"
        print(f"{symbol} {name:<45} {status}")
    
    print("-" * 70)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Phase B integration is ready.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed.")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
