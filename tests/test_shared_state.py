import pytest
from core.shared_state import SharedState

def test_initial_balances():
    state = SharedState()
    assert isinstance(state.balances, dict)
    # Balances may be empty initially, so just check it's a dict
    assert isinstance(state.balances, dict)

def test_calculate_capital_floor():
    """Test capital floor calculation: max(8, NAV * 0.12, trade_size * 0.5)"""
    state = SharedState()
    
    # Test 1: Small NAV and trade size - should return absolute minimum of 8
    result = state.calculate_capital_floor(nav=10.0, trade_size=5.0)
    assert result == 8.0, f"Expected 8.0, got {result}"
    
    # Test 2: NAV-based floor is dominant
    result = state.calculate_capital_floor(nav=200.0, trade_size=10.0)
    expected = max(8.0, 200.0 * 0.12, 10.0 * 0.5)  # max(8, 24, 5) = 24
    assert result == expected, f"Expected {expected}, got {result}"
    
    # Test 3: Trade size-based floor is dominant
    result = state.calculate_capital_floor(nav=50.0, trade_size=100.0)
    expected = max(8.0, 50.0 * 0.12, 100.0 * 0.5)  # max(8, 6, 50) = 50
    assert result == expected, f"Expected {expected}, got {result}"
    
    # Test 4: All components equal - NAV-based dominates
    result = state.calculate_capital_floor(nav=66.67, trade_size=16.0)
    expected = max(8.0, 66.67 * 0.12, 16.0 * 0.5)  # max(8, ~8, 8) = 8
    assert abs(result - 8.0) < 0.1, f"Expected ~8.0, got {result}"
    
    # Test 5: Large NAV scenario - NAV-based dominates
    result = state.calculate_capital_floor(nav=1000.0, trade_size=100.0)
    expected = max(8.0, 1000.0 * 0.12, 100.0 * 0.5)  # max(8, 120, 50) = 120
    assert result == expected, f"Expected {expected}, got {result}"
    
    # Test 6: Default parameters (nav=0, trade_size=0) should use state values
    result = state.calculate_capital_floor()
    assert result >= 8.0, f"Floor should be at least 8.0, got {result}"

def test_capital_floor_recalculation_on_nav_change():
    """Test that capital floor recalculates when NAV changes (simulating cycle updates)"""
    state = SharedState()
    trade_size = 30.0
    
    # Cycle 1: NAV = 100
    floor_cycle1 = state.calculate_capital_floor(nav=100.0, trade_size=trade_size)
    expected_cycle1 = max(8.0, 100.0 * 0.12, trade_size * 0.5)  # max(8, 12, 15) = 15
    assert floor_cycle1 == expected_cycle1
    
    # Cycle 2: NAV increases to 500
    floor_cycle2 = state.calculate_capital_floor(nav=500.0, trade_size=trade_size)
    expected_cycle2 = max(8.0, 500.0 * 0.12, trade_size * 0.5)  # max(8, 60, 15) = 60
    assert floor_cycle2 == expected_cycle2
    assert floor_cycle2 > floor_cycle1, "Floor should increase with NAV"
    
    # Cycle 3: NAV decreases to 200
    floor_cycle3 = state.calculate_capital_floor(nav=200.0, trade_size=trade_size)
    expected_cycle3 = max(8.0, 200.0 * 0.12, trade_size * 0.5)  # max(8, 24, 15) = 24
    assert floor_cycle3 == expected_cycle3
    assert floor_cycle3 < floor_cycle2, "Floor should decrease with NAV"

def test_capital_floor_vs_free_usdt():
    """Test the comparison: free_usdt >= capital_floor for trading approval"""
    state = SharedState()
    
    # Scenario 1: Sufficient capital
    nav = 500.0
    trade_size = 30.0
    free_usdt = 100.0
    capital_floor = state.calculate_capital_floor(nav=nav, trade_size=trade_size)
    assert free_usdt >= capital_floor, f"Expected {free_usdt} >= {capital_floor}"
    # Trade would be allowed
    
    # Scenario 2: Insufficient capital
    nav = 500.0
    trade_size = 30.0
    free_usdt = 5.0
    capital_floor = state.calculate_capital_floor(nav=nav, trade_size=trade_size)
    assert free_usdt < capital_floor, f"Expected {free_usdt} < {capital_floor}"
    # Trade would be blocked
    
    # Scenario 3: Exactly at floor
    nav = 100.0
    trade_size = 30.0
    free_usdt = 15.0  # Exactly at floor
    capital_floor = state.calculate_capital_floor(nav=nav, trade_size=trade_size)
    assert free_usdt >= capital_floor, f"Expected {free_usdt} >= {capital_floor}"
    # Trade would be allowed (can spend remaining USDT)

