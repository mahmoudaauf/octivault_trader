#!/usr/bin/env python3
"""
Test suite for SignalManager NAV and Position Count sources.

Validates:
- Constructor accepts new parameters
- get_current_nav() retrieves NAV correctly
- get_position_count() retrieves position count correctly
- Graceful fallback when sources unavailable
"""

import logging
import sys
from typing import Dict, Any, Optional, Callable

# Mock classes for testing
class MockConfig:
    """Mock configuration object."""
    pass


class MockLogger:
    """Mock logger that tracks calls."""
    def __init__(self):
        self.calls = []
    
    def info(self, msg, *args, **kwargs):
        self.calls.append(('info', msg % args if args else msg))
    
    def warning(self, msg, *args, **kwargs):
        self.calls.append(('warning', msg % args if args else msg))
    
    def debug(self, msg, *args, **kwargs):
        self.calls.append(('debug', msg % args if args else msg))


class MockSharedState:
    """Mock SharedState with NAV and positions."""
    def __init__(self, nav: float = 1000.0, positions: Optional[Dict] = None):
        self.nav = nav
        self.portfolio_nav = nav * 1.1
        self.total_equity_usdt = nav * 1.05
        self._positions = positions or {
            "BTCUSDT": {"qty": 0.1, "quantity": 0.1},
            "ETHUSDT": {"qty": 0.5, "quantity": 0.5},
            "DUST": {"qty": 0.0, "quantity": 0.0},
        }
    
    def get_positions_snapshot(self) -> Dict[str, Dict]:
        """Return positions snapshot."""
        return self._positions


def test_constructor_without_sources():
    """Test SignalManager initialization without NAV/position sources."""
    from core.signal_manager import SignalManager
    
    print("Test 1: Constructor without sources...")
    config = MockConfig()
    logger = MockLogger()
    
    sm = SignalManager(config=config, logger=logger)
    
    assert sm.config == config
    assert sm.logger == logger
    assert sm.shared_state is None
    assert sm.position_count_source is None
    print("✓ PASS: Constructor works without sources\n")


def test_constructor_with_sources():
    """Test SignalManager initialization with NAV/position sources."""
    from core.signal_manager import SignalManager
    
    print("Test 2: Constructor with sources...")
    config = MockConfig()
    logger = MockLogger()
    shared_state = MockSharedState()
    position_source = lambda: 2
    
    sm = SignalManager(
        config=config,
        logger=logger,
        shared_state=shared_state,
        position_count_source=position_source
    )
    
    assert sm.shared_state == shared_state
    assert sm.position_count_source == position_source
    
    # Check for initialization log
    log_found = any("Initialized with NAV source" in str(call) for call in logger.calls)
    assert log_found, "Expected initialization log not found"
    print("✓ PASS: Constructor works with sources")
    print("  Initialization log:", [c for c in logger.calls if "Initialized" in str(c)][0][1])
    print()


def test_get_current_nav_without_source():
    """Test get_current_nav() when no source provided."""
    from core.signal_manager import SignalManager
    
    print("Test 3: get_current_nav() without source...")
    config = MockConfig()
    logger = MockLogger()
    
    sm = SignalManager(config=config, logger=logger)
    nav = sm.get_current_nav()
    
    assert nav == 0.0, f"Expected 0.0, got {nav}"
    print("✓ PASS: Returns 0.0 when no source configured\n")


def test_get_current_nav_with_source():
    """Test get_current_nav() retrieves NAV from shared_state."""
    from core.signal_manager import SignalManager
    
    print("Test 4: get_current_nav() with source...")
    config = MockConfig()
    logger = MockLogger()
    shared_state = MockSharedState(nav=2500.0)
    
    sm = SignalManager(config=config, logger=logger, shared_state=shared_state)
    nav = sm.get_current_nav()
    
    assert nav == 2500.0, f"Expected 2500.0, got {nav}"
    print(f"✓ PASS: Retrieved NAV = {nav} USDT\n")


def test_get_current_nav_fallback():
    """Test get_current_nav() fallback to portfolio_nav."""
    from core.signal_manager import SignalManager
    
    print("Test 5: get_current_nav() fallback chain...")
    config = MockConfig()
    logger = MockLogger()
    
    # Create shared_state with nav=0 to force fallback
    shared_state = MockSharedState(nav=0.0)
    shared_state.nav = None  # Remove nav attribute
    
    sm = SignalManager(config=config, logger=logger, shared_state=shared_state)
    nav = sm.get_current_nav()
    
    # Should fall back to portfolio_nav (nav * 1.1)
    expected = 0.0 * 1.1  # nav was 0.0, so portfolio_nav = 0.0
    # Actually, let's check with a proper value
    
    shared_state2 = MockSharedState(nav=1000.0)
    shared_state2.nav = None
    sm2 = SignalManager(config=config, logger=logger, shared_state=shared_state2)
    nav2 = sm2.get_current_nav()
    
    expected2 = 1000.0 * 1.1  # Falls back to portfolio_nav
    assert nav2 == expected2, f"Expected {expected2}, got {nav2}"
    print(f"✓ PASS: Fallback chain works, NAV = {nav2} USDT\n")


def test_get_position_count_without_source():
    """Test get_position_count() when no source provided."""
    from core.signal_manager import SignalManager
    
    print("Test 6: get_position_count() without source...")
    config = MockConfig()
    logger = MockLogger()
    
    sm = SignalManager(config=config, logger=logger)
    count = sm.get_position_count()
    
    assert count == 0, f"Expected 0, got {count}"
    print("✓ PASS: Returns 0 when no source configured\n")


def test_get_position_count_with_callable():
    """Test get_position_count() with callable source."""
    from core.signal_manager import SignalManager
    
    print("Test 7: get_position_count() with callable source...")
    config = MockConfig()
    logger = MockLogger()
    position_source = lambda: 3
    
    sm = SignalManager(
        config=config,
        logger=logger,
        position_count_source=position_source
    )
    count = sm.get_position_count()
    
    assert count == 3, f"Expected 3, got {count}"
    print(f"✓ PASS: Retrieved position count = {count}\n")


def test_get_position_count_from_shared_state():
    """Test get_position_count() counts positions from shared_state."""
    from core.signal_manager import SignalManager
    
    print("Test 8: get_position_count() from shared_state...")
    config = MockConfig()
    logger = MockLogger()
    
    shared_state = MockSharedState(positions={
        "BTCUSDT": {"qty": 0.1},
        "ETHUSDT": {"qty": 0.5},
        "BNBUSDT": {"qty": 0.0},  # Should not be counted (qty=0)
        "ADAUSDT": {"quantity": 1.0},  # Should be counted (qty falls back to quantity)
    })
    
    sm = SignalManager(config=config, logger=logger, shared_state=shared_state)
    count = sm.get_position_count()
    
    # BTC (0.1), ETH (0.5), ADA (1.0) = 3 positions (BNB is 0)
    assert count == 3, f"Expected 3, got {count}"
    print(f"✓ PASS: Counted {count} positions from shared_state\n")


def test_get_position_count_priority():
    """Test get_position_count() uses callable over shared_state."""
    from core.signal_manager import SignalManager
    
    print("Test 9: get_position_count() priority (callable > shared_state)...")
    config = MockConfig()
    logger = MockLogger()
    shared_state = MockSharedState()  # Has 2 positions
    position_source = lambda: 5  # Should take precedence
    
    sm = SignalManager(
        config=config,
        logger=logger,
        shared_state=shared_state,
        position_count_source=position_source
    )
    count = sm.get_position_count()
    
    assert count == 5, f"Expected 5 (from callable), got {count}"
    print(f"✓ PASS: Used callable source (5) over shared_state (2)\n")


def test_error_handling_nav():
    """Test get_current_nav() handles exceptions gracefully."""
    from core.signal_manager import SignalManager
    
    print("Test 10: Error handling in get_current_nav()...")
    config = MockConfig()
    logger = MockLogger()
    
    # Create a mock that raises exception
    class BadSharedState:
        @property
        def nav(self):
            raise RuntimeError("Simulated error")
    
    sm = SignalManager(
        config=config,
        logger=logger,
        shared_state=BadSharedState()
    )
    
    # Should not raise, should return 0.0
    nav = sm.get_current_nav()
    assert nav == 0.0, f"Expected 0.0 on error, got {nav}"
    
    # Check for error log
    debug_logs = [c for c in logger.calls if c[0] == 'debug']
    error_found = any("Failed to get NAV" in str(c) for c in debug_logs)
    assert error_found, "Expected error log not found"
    print(f"✓ PASS: Exception handled gracefully")
    print("  Error logged:", [c for c in debug_logs if "Failed to get NAV" in str(c)][0][1] if error_found else "")
    print()


def test_error_handling_position_count():
    """Test get_position_count() handles exceptions gracefully."""
    from core.signal_manager import SignalManager
    
    print("Test 11: Error handling in get_position_count()...")
    config = MockConfig()
    logger = MockLogger()
    
    def bad_position_source():
        raise RuntimeError("Simulated error")
    
    # Create a mock that raises exception
    class BadSharedState:
        def get_positions_snapshot(self):
            raise RuntimeError("Simulated error")
    
    sm = SignalManager(
        config=config,
        logger=logger,
        shared_state=BadSharedState(),
        position_count_source=bad_position_source
    )
    
    # Should not raise, should return 0
    count = sm.get_position_count()
    assert count == 0, f"Expected 0 on error, got {count}"
    
    # Check for error logs
    debug_logs = [c for c in logger.calls if c[0] == 'debug']
    errors_found = [c for c in debug_logs if "Failed to" in str(c)]
    assert len(errors_found) > 0, "Expected error logs not found"
    print(f"✓ PASS: Exception handled gracefully")
    print("  Errors logged:", len(errors_found))
    print()


def run_all_tests():
    """Run all tests."""
    print("=" * 80)
    print("SignalManager NAV & Position Count Source - Test Suite")
    print("=" * 80)
    print()
    
    tests = [
        test_constructor_without_sources,
        test_constructor_with_sources,
        test_get_current_nav_without_source,
        test_get_current_nav_with_source,
        test_get_current_nav_fallback,
        test_get_position_count_without_source,
        test_get_position_count_with_callable,
        test_get_position_count_from_shared_state,
        test_get_position_count_priority,
        test_error_handling_nav,
        test_error_handling_position_count,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            failed += 1
            print(f"✗ FAIL: {test.__name__}")
            print(f"  Error: {e}\n")
        except Exception as e:
            failed += 1
            print(f"✗ ERROR: {test.__name__}")
            print(f"  Exception: {e}\n")
    
    print("=" * 80)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 80)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
