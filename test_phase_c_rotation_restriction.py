#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase C Integration Test Suite
Testing Capital Governor-based Rotation Restrictions

Tests:
  1. Rotation blocked in MICRO bracket
  2. Rotation allowed in SMALL bracket
  3. Rotation allowed in MEDIUM bracket
  4. Rotation allowed in LARGE bracket
  5. Stagnation-based rotation blocked in MICRO
  6. Graceful fallback on error
"""

import sys
import logging
from unittest.mock import Mock, MagicMock

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s: %(message)s'
)


class MockConfig:
    """Mock configuration object."""
    def __init__(self):
        self.ROTATION_BASE_ALPHA_GAP = 0.005
        self.ROTATION_WINNER_PROTECTION_PNL = 0.002
        self.ROTATION_WINNER_EXTRA_ALPHA = 0.03
        self.STAGNATION_FORCE_ROTATION_ENABLED = True
        self.HOLD_GRACE_SECONDS = 180.0


class MockSharedState:
    """Mock SharedState for testing."""
    def __init__(self, nav=350.0):
        self.nav = nav
        self.total_value = nav
        self.is_cold_bootstrap = False
        self.cold_bootstrap = False


def test_rotation_restriction_micro():
    """Test 1: Rotation blocked in MICRO bracket ($350)."""
    print("\n" + "="*70)
    print("TEST 1: Rotation Restriction - MICRO Bracket ($350)")
    print("="*70)
    
    try:
        # Import here to catch import errors
        from core.capital_governor import CapitalGovernor
        from core.rotation_authority import RotationExitAuthority
        
        config = MockConfig()
        shared_state = MockSharedState(nav=350.0)  # MICRO bracket
        logger = logging.getLogger("test")
        
        # Initialize Governor and Authority
        governor = CapitalGovernor(config)
        authority = RotationExitAuthority(logger, config, shared_state, capital_governor=governor)
        
        # Test: Check rotation restriction
        should_restrict, reason = authority.should_restrict_rotation("BTCUSDT")
        
        print(f"NAV: ${shared_state.nav}")
        print(f"Bracket: {governor.get_bracket(shared_state.nav).value}")
        print(f"Should restrict rotation: {should_restrict}")
        print(f"Reason: {reason}")
        
        if should_restrict and reason == "micro_bracket_restriction":
            print("✅ MICRO bracket rotation restriction VERIFIED")
            return True
        else:
            print("❌ Test failed: Expected rotation to be restricted in MICRO")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rotation_allowed_small():
    """Test 2: Rotation allowed in SMALL bracket ($1,500)."""
    print("\n" + "="*70)
    print("TEST 2: Rotation Allowed - SMALL Bracket ($1,500)")
    print("="*70)
    
    try:
        from core.capital_governor import CapitalGovernor
        from core.rotation_authority import RotationExitAuthority
        
        config = MockConfig()
        shared_state = MockSharedState(nav=1500.0)  # SMALL bracket
        logger = logging.getLogger("test")
        
        governor = CapitalGovernor(config)
        authority = RotationExitAuthority(logger, config, shared_state, capital_governor=governor)
        
        should_restrict, reason = authority.should_restrict_rotation("BTCUSDT")
        
        print(f"NAV: ${shared_state.nav}")
        print(f"Bracket: {governor.get_bracket(shared_state.nav).value}")
        print(f"Should restrict rotation: {should_restrict}")
        print(f"Reason: {reason}")
        
        if not should_restrict:
            print("✅ SMALL bracket rotation ALLOWED")
            return True
        else:
            print("❌ Test failed: Expected rotation to be allowed in SMALL")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rotation_allowed_medium():
    """Test 3: Rotation allowed in MEDIUM bracket ($5,000)."""
    print("\n" + "="*70)
    print("TEST 3: Rotation Allowed - MEDIUM Bracket ($5,000)")
    print("="*70)
    
    try:
        from core.capital_governor import CapitalGovernor
        from core.rotation_authority import RotationExitAuthority
        
        config = MockConfig()
        shared_state = MockSharedState(nav=5000.0)  # MEDIUM bracket
        logger = logging.getLogger("test")
        
        governor = CapitalGovernor(config)
        authority = RotationExitAuthority(logger, config, shared_state, capital_governor=governor)
        
        should_restrict, reason = authority.should_restrict_rotation("ETHUSDT")
        
        print(f"NAV: ${shared_state.nav}")
        print(f"Bracket: {governor.get_bracket(shared_state.nav).value}")
        print(f"Should restrict rotation: {should_restrict}")
        
        if not should_restrict:
            print("✅ MEDIUM bracket rotation ALLOWED")
            return True
        else:
            print("❌ Test failed: Expected rotation to be allowed in MEDIUM")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rotation_allowed_large():
    """Test 4: Rotation allowed in LARGE bracket ($50,000)."""
    print("\n" + "="*70)
    print("TEST 4: Rotation Allowed - LARGE Bracket ($50,000)")
    print("="*70)
    
    try:
        from core.capital_governor import CapitalGovernor
        from core.rotation_authority import RotationExitAuthority
        
        config = MockConfig()
        shared_state = MockSharedState(nav=50000.0)  # LARGE bracket
        logger = logging.getLogger("test")
        
        governor = CapitalGovernor(config)
        authority = RotationExitAuthority(logger, config, shared_state, capital_governor=governor)
        
        should_restrict, reason = authority.should_restrict_rotation("SOLUSDT")
        
        print(f"NAV: ${shared_state.nav}")
        print(f"Bracket: {governor.get_bracket(shared_state.nav).value}")
        print(f"Should restrict rotation: {should_restrict}")
        
        if not should_restrict:
            print("✅ LARGE bracket rotation ALLOWED")
            return True
        else:
            print("❌ Test failed: Expected rotation to be allowed in LARGE")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_stagnation_blocked_micro():
    """Test 5: Stagnation-based rotation blocked in MICRO."""
    print("\n" + "="*70)
    print("TEST 5: Stagnation-based Rotation Blocked - MICRO Bracket")
    print("="*70)
    
    try:
        from core.capital_governor import CapitalGovernor
        from core.rotation_authority import RotationExitAuthority
        
        config = MockConfig()
        shared_state = MockSharedState(nav=350.0)  # MICRO bracket
        logger = logging.getLogger("test")
        
        governor = CapitalGovernor(config)
        authority = RotationExitAuthority(logger, config, shared_state, capital_governor=governor)
        
        # Mock owned positions (prevents cold_bootstrap check)
        owned_positions = {
            "BTCUSDT": {
                "quantity": 0.00017,
                "entry_time": 1708000000,
                "unrealized_pnl_pct": -0.01,
                "state": "ACTIVE"
            }
        }
        
        # Call authorize_stagnation_exit (should return None due to MICRO restriction)
        result = authority.authorize_stagnation_exit(owned_positions, "NORMAL")
        
        print(f"NAV: ${shared_state.nav}")
        print(f"Bracket: {governor.get_bracket(shared_state.nav).value}")
        print(f"authorize_stagnation_exit result: {result}")
        print(f"Expected: None (blocked by PHASE C)")
        
        if result is None:
            print("✅ Stagnation rotation blocked in MICRO bracket")
            return True
        else:
            print("❌ Test failed: Expected stagnation rotation to be blocked")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_graceful_fallback():
    """Test 6: Governor initialization when passed None."""
    print("\n" + "="*70)
    print("TEST 6: Governor Initialization - Fallback Auto-Init")
    print("="*70)
    
    try:
        from core.rotation_authority import RotationExitAuthority
        
        config = MockConfig()
        shared_state = MockSharedState(nav=350.0)
        logger = logging.getLogger("test")
        
        # Initialize without passing Governor (should auto-init)
        authority = RotationExitAuthority(logger, config, shared_state, capital_governor=None)
        
        # Should initialize Governor automatically in __init__
        # For MICRO bracket, should restrict
        should_restrict, reason = authority.should_restrict_rotation("BTCUSDT")
        
        print(f"Capital Governor auto-initialized: {authority.capital_governor is not None}")
        print(f"NAV: ${shared_state.nav}")
        print(f"Should restrict rotation: {should_restrict}")
        print(f"Reason: {reason or '(empty)'}")
        
        # In MICRO bracket, should be restricted (verifies auto-init worked)
        if should_restrict and reason == "micro_bracket_restriction":
            print("✅ Governor auto-initialization verified")
            return True
        else:
            print("❌ Test failed: Expected Governor auto-init to work")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("PHASE C INTEGRATION TEST SUITE")
    print("Capital Governor-based Rotation Restrictions")
    print("="*70)
    
    tests = [
        ("Rotation blocked in MICRO", test_rotation_restriction_micro),
        ("Rotation allowed in SMALL", test_rotation_allowed_small),
        ("Rotation allowed in MEDIUM", test_rotation_allowed_medium),
        ("Rotation allowed in LARGE", test_rotation_allowed_large),
        ("Stagnation rotation blocked in MICRO", test_stagnation_blocked_micro),
        ("Governor auto-initialization", test_graceful_fallback),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"CRITICAL ERROR in {test_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print("-" * 70)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Phase C integration is ready.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
