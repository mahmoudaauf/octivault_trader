#!/usr/bin/env python3
"""
Test script for capital_floor implementation.
Tests the formula: capital_floor = max(8, NAV * 0.12, trade_size * 0.5)
"""

import sys
sys.path.insert(0, "/Users/mauf/Desktop/Octi AI Trading Bot/octivault_trader")

from core.shared_state import SharedState, SharedStateConfig

def test_capital_floor():
    """Test the calculate_capital_floor method with various scenarios."""
    
    config = SharedStateConfig()
    ss = SharedState(config)
    
    print("=" * 70)
    print("CAPITAL FLOOR FORMULA TEST: capital_floor = max(8, NAV * 0.12, trade_size * 0.5)")
    print("=" * 70)
    
    test_cases = [
        # (nav, trade_size, description)
        (100, 30, "Small account: NAV=100, trade_size=30"),
        (500, 30, "Mid account: NAV=500, trade_size=30"),
        (1000, 50, "Large account: NAV=1000, trade_size=50"),
        (2000, 100, "Very large account: NAV=2000, trade_size=100"),
        (50, 20, "Tiny account: NAV=50, trade_size=20"),
        (0, 0, "Bootstrap mode: NAV=0, trade_size=0 (uses defaults)"),
    ]
    
    for nav, trade_size, description in test_cases:
        result = ss.calculate_capital_floor(nav=nav, trade_size=trade_size)
        
        # Calculate components for display
        abs_min = 8.0
        nav_component = nav * 0.12 if nav > 0 else 0.0
        
        # Get actual trade_size used
        actual_trade_size = trade_size
        if trade_size <= 0:
            actual_trade_size = float(ss._cfg("TRADE_AMOUNT_USDT", 30.0) or 30.0)
        
        trade_component = actual_trade_size * 0.5
        
        print(f"\n{description}")
        print(f"  Inputs: NAV=${nav:.2f}, trade_size=${trade_size:.2f}")
        print(f"  Components:")
        print(f"    • Absolute minimum: ${abs_min:.2f}")
        print(f"    • NAV-based (12%): ${nav_component:.2f}")
        print(f"    • Trade-based (50%): ${trade_component:.2f}")
        print(f"  ✓ Capital floor: ${result:.2f}")
        
        # Verify the formula
        expected = max(abs_min, nav_component, trade_component)
        assert abs(result - expected) < 0.01, f"Expected {expected}, got {result}"
    
    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED")
    print("=" * 70)

if __name__ == "__main__":
    test_capital_floor()
