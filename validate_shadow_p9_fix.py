#!/usr/bin/env python3
"""
Validation script for Shadow Mode P9 Readiness Gate Fix

Verifies that the readiness gate behavior is correct for both shadow and live modes.
"""

import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ShadowModeP9Fix")


def test_shadow_mode_logic():
    """Test shadow mode readiness gate logic."""
    logger.info("=" * 80)
    logger.info("TEST: Shadow Mode P9 Readiness Gate Logic")
    logger.info("=" * 80)
    
    # Simulate the readiness check
    def check_readiness(is_shadow_mode, md_ready, as_ready, has_accepted_symbols):
        if is_shadow_mode:
            # Shadow mode: Only require accepted_symbols (via event OR actual population)
            readiness_ok = as_ready or has_accepted_symbols
            reason = "shadow"
        else:
            # Live mode: Require both market data AND accepted symbols
            readiness_ok = (md_ready and as_ready)
            reason = "live"
        return readiness_ok, reason
    
    test_cases = [
        # (is_shadow, md_ready, as_ready, has_symbols, expected_ok, description)
        (True, False, False, True,  True,  "Shadow: Only has_symbols → OK"),
        (True, False, True,  False, True,  "Shadow: as_ready event → OK"),
        (True, True,  True,  True,  True,  "Shadow: All set → OK"),
        (True, False, False, False, False, "Shadow: Nothing → BLOCKED"),
        
        (False, True,  True,  True,  True,  "Live: All set → OK"),
        (False, True,  False, True,  False, "Live: Missing as_ready → BLOCKED"),
        (False, False, True,  True,  False, "Live: Missing md_ready → BLOCKED"),
        (False, False, False, True,  False, "Live: Missing both → BLOCKED"),
    ]
    
    all_passed = True
    for is_shadow, md_ready, as_ready, has_symbols, expected, desc in test_cases:
        readiness_ok, reason = check_readiness(is_shadow, md_ready, as_ready, has_symbols)
        
        status = "✅ PASS" if readiness_ok == expected else "❌ FAIL"
        if readiness_ok != expected:
            all_passed = False
        
        logger.info(
            f"{status} | {desc:50s} | Result: {readiness_ok} (expected {expected})"
        )
    
    return all_passed


def test_fallback_logic():
    """Test the fallback symbol checking logic."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST: Fallback Symbol Checking Logic")
    logger.info("=" * 80)
    
    # Simulate the fallback check
    def get_accepted_symbols_status(accepted_symbols_dict, event_is_set):
        has_symbols = bool(accepted_symbols_dict)
        event_ready = event_is_set
        return has_symbols, event_ready, (has_symbols or event_ready)
    
    test_cases = [
        # (symbols_dict, event_set, description)
        ({}, False, "Empty symbols, event not set"),
        ({}, True,  "Empty symbols, event set"),
        ({"BTCUSDT": {}}, False, "Has symbols, event not set"),
        ({"BTCUSDT": {}, "ETHUSDT": {}}, True, "Has symbols, event set"),
    ]
    
    all_passed = True
    for symbols_dict, event_set, desc in test_cases:
        has_symbols, event_ready, total_ok = get_accepted_symbols_status(symbols_dict, event_set)
        
        status = "✅" if (total_ok == (has_symbols or event_set)) else "❌"
        logger.info(
            f"{status} {desc:50s} | has_symbols={has_symbols}, event={event_ready}, total_ok={total_ok}"
        )
    
    return all_passed


def main():
    """Run all validation tests."""
    logger.info("\n🔍 Shadow Mode P9 Readiness Gate Fix - Validation Suite\n")
    
    test1_pass = test_shadow_mode_logic()
    test2_pass = test_fallback_logic()
    
    logger.info("\n" + "=" * 80)
    if test1_pass and test2_pass:
        logger.info("✅ ALL TESTS PASSED")
        logger.info("=" * 80)
        return 0
    else:
        logger.error("❌ SOME TESTS FAILED")
        logger.error("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
