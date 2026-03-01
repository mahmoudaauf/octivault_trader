#!/usr/bin/env python3
"""
Diagnostic test for SignalManager validation.
Helps identify why signals are being rejected.
"""
import logging
import sys
from unittest.mock import MagicMock

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(name)s - %(levelname)s - %(message)s'
)

# Create mock config
config = MagicMock()
config.MIN_SIGNAL_CONF = 0.10
config.MAX_SIGNAL_AGE_SECONDS = 60.0
config.DEFAULT_PLANNED_QUOTE = 10.0
config.signal_cache_max_size = 1000
config.signal_cache_ttl = 300.0

# Import after config is ready
from core.signal_manager import SignalManager

logger = logging.getLogger("TEST")
signal_manager = SignalManager(config, logger)

# Test cases
test_cases = [
    {
        "name": "Valid BTC/USDT signal",
        "agent": "TrendHunter",
        "symbol": "BTCUSDT",
        "signal": {"action": "BUY", "confidence": 0.75},
        "expected": True,
    },
    {
        "name": "Valid ETH/USDT signal",
        "agent": "DipSniper",
        "symbol": "ETHUSDT",
        "signal": {"action": "SELL", "confidence": 0.60},
        "expected": True,
    },
    {
        "name": "Low confidence signal (should pass with new floor)",
        "agent": "MLForecaster",
        "symbol": "ADAUSDT",
        "signal": {"action": "BUY", "confidence": 0.15},
        "expected": True,
    },
    {
        "name": "Very low confidence signal (should fail)",
        "agent": "SwingTrader",
        "symbol": "XRPUSDT",
        "signal": {"action": "BUY", "confidence": 0.05},
        "expected": False,
    },
    {
        "name": "Missing confidence (defaults to 0.0, should fail)",
        "agent": "LiquidationAgent",
        "symbol": "DOGEUSDT",
        "signal": {"action": "SELL"},
        "expected": False,
    },
    {
        "name": "Symbol with slash",
        "agent": "TrendHunter",
        "symbol": "BTC/USDT",
        "signal": {"action": "BUY", "confidence": 0.70},
        "expected": True,
    },
    {
        "name": "Invalid quote token (BTC/EUR)",
        "agent": "TestAgent",
        "symbol": "BTCEUR",
        "signal": {"action": "BUY", "confidence": 0.70},
        "expected": False,
    },
    {
        "name": "Too short symbol",
        "agent": "TestAgent",
        "symbol": "BTC",
        "signal": {"action": "BUY", "confidence": 0.70},
        "expected": False,
    },
    {
        "name": "Confidence > 1.0 (should be clamped)",
        "agent": "TestAgent",
        "symbol": "BTCUSDT",
        "signal": {"action": "BUY", "confidence": 1.25},
        "expected": True,
    },
    {
        "name": "Confidence = 0.10 (edge case)",
        "agent": "TestAgent",
        "symbol": "BTCUSDT",
        "signal": {"action": "BUY", "confidence": 0.10},
        "expected": True,
    },
]

print("\n" + "="*80)
print("SIGNAL MANAGER VALIDATION TESTS")
print("="*80 + "\n")

passed = 0
failed = 0

for tc in test_cases:
    result = signal_manager.receive_signal(tc["agent"], tc["symbol"], tc["signal"])
    status = "✓ PASS" if result == tc["expected"] else "✗ FAIL"
    
    if result == tc["expected"]:
        passed += 1
    else:
        failed += 1
    
    print(f"{status} | {tc['name']}")
    print(f"       Agent={tc['agent']}, Symbol={tc['symbol']}, Conf={tc['signal'].get('confidence', 'N/A')}")
    print(f"       Expected={tc['expected']}, Got={result}")
    
    # Check cache
    cached = signal_manager.get_signals_for_symbol(tc["symbol"].upper().replace("/", ""))
    if cached:
        print(f"       ✓ Signal cached: {len(cached)} signal(s)")
    print()

print("="*80)
print(f"Results: {passed} passed, {failed} failed out of {len(test_cases)} tests")
print("="*80 + "\n")

if failed > 0:
    sys.exit(1)
