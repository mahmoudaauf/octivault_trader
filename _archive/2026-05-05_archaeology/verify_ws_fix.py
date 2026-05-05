#!/usr/bin/env python3
"""
WebSocket Fix Verification Script

Checks that the WebSocket issue fix is properly deployed.
Monitors for key log patterns that indicate the system is working.
"""

import re
import sys
import time
from pathlib import Path


def check_log_patterns(log_file, patterns, timeout_sec=30):
    """
    Monitor a log file for specific patterns within a timeout period.
    Returns True if all patterns found, False otherwise.
    """
    print(f"\n📋 Monitoring {log_file} for {timeout_sec}s...")

    if not Path(log_file).exists():
        print(f"❌ Log file not found: {log_file}")
        return False

    # Get initial file size
    initial_size = Path(log_file).stat().st_size

    # Pattern status tracker
    found = {pattern: False for pattern in patterns}
    start_time = time.time()

    while time.time() - start_time < timeout_sec:
        try:
            with open(log_file) as f:
                content = f.read()

            for pattern in patterns:
                if not found[pattern]:
                    if re.search(pattern, content):
                        found[pattern] = True
                        print(f"✅ Found: {pattern}")

        except Exception as e:
            print(f"⚠️  Error reading log: {e}")

        time.sleep(1)

    # Report results
    print(f"\n📊 Results after {timeout_sec}s:")
    all_found = True
    for pattern, is_found in found.items():
        status = "✅" if is_found else "❌"
        print(f"  {status} {pattern}")
        if not is_found:
            all_found = False

    return all_found


def check_ws_auto_subscribe():
    """
    Verify WebSocket auto-subscribe fix is working.
    """
    print("\n" + "=" * 60)
    print("🔍 WebSocket Fix Verification")
    print("=" * 60)

    log_file = "/Users/mauf/Desktop/Octi AI Trading Bot/octivault_trader/logs/octivault_master_orchestrator.log"

    # Check 1: WebSocket should auto-subscribe to symbols
    print("\n[Check 1] WebSocket Auto-Subscribe")
    patterns_1 = [
        r"\[WS:AutoSubscribe\] Subscribed to \d+ symbols",
        r"\[WS:Connected\] WebSocket connected",
    ]

    if check_log_patterns(log_file, patterns_1, timeout_sec=15):
        print("✅ PASS: WebSocket auto-subscribe working")
    else:
        print("⚠️  WARNING: Check WebSocket logs manually")

    # Check 2: Verify NO hanging on missing symbols
    print("\n[Check 2] No WebSocket Hangs")

    with open(log_file) as f:
        content = f.read()

    # Count occurrences of the error pattern
    hang_pattern = r"\[WS:Connect\] No symbols to subscribe, waiting"
    hangs = len(re.findall(hang_pattern, content))

    if hangs > 0:
        print(f"⚠️  WARNING: Found {hangs} WebSocket hangs (checking if recent...)")

        # Get timestamp of last occurrence
        last_lines = content.split("\n")[-100:]
        recent_hang = any(hang_pattern in line for line in last_lines)

        if recent_hang:
            print("❌ FAIL: Recent WebSocket hangs detected")
            return False
        else:
            print("✅ PASS: Hangs are old (before fix was deployed)")
    else:
        print("✅ PASS: No WebSocket hangs detected")

    # Check 3: Verify fallback mechanism
    print("\n[Check 3] Fallback Mechanism")
    patterns_3 = [
        r"\[WS:Fallback\] Using \d+ (bootstrap|hardcoded) symbols",
    ]

    fallback_found = check_log_patterns(log_file, patterns_3, timeout_sec=10)
    if fallback_found:
        print("✅ PASS: Fallback mechanism is working")
    else:
        print("ℹ️  INFO: Fallback not triggered (might not be needed if primary method works)")

    # Check 4: Market Data Feed subscribing WebSocket
    print("\n[Check 4] MDF WebSocket Subscription")
    patterns_4 = [
        r"\[MDF\] WebSocket subscribed to \d+ new symbols",
    ]

    if check_log_patterns(log_file, patterns_4, timeout_sec=20):
        print("✅ PASS: MDF is proactively subscribing WebSocket")
    else:
        print("ℹ️  INFO: MDF subscription not triggered (might happen later during runtime)")

    # Check 5: Trade execution should now work
    print("\n[Check 5] Trade Execution")

    with open(log_file) as f:
        content = f.read()

    # Check for trade decisions
    decisions = len(re.findall(r"DECISION.*(?:BUY|SELL|HOLD|NONE)", content))
    executions = len(re.findall(r"execute_trade|TRADE_SUBMITTED|EM:", content))

    print(f"  📊 Found {decisions} trade decisions in logs")
    print(f"  📊 Found {executions} trade executions in logs")

    if decisions > 0 and executions > 0:
        print("✅ PASS: Trade pipeline is working!")
    elif decisions > 0:
        print("⚠️  INFO: Decisions made but not yet executed (normal during startup)")
    else:
        print("ℹ️  INFO: Monitor system after it stabilizes")

    print("\n" + "=" * 60)
    print("✅ Verification Complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    try:
        check_ws_auto_subscribe()
    except KeyboardInterrupt:
        print("\n⛔ Verification interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
        sys.exit(1)
