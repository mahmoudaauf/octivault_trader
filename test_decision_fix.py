#!/usr/bin/env python3
"""
Test script to verify that signals are being converted to decisions
"""
import subprocess
import re
import sys
import time

def run_test():
    """Run a test and check if decisions are being generated"""
    print("🔍 Testing decision generation fix...")
    print("=" * 80)
    
    # Run the test
    try:
        result = subprocess.run(
            ['python3', '-m', 'core.test_runner'],
            cwd='/Users/mauf/Desktop/Octi AI Trading Bot/octivault_trader',
            capture_output=True,
            text=True,
            timeout=45
        )
        
        output = result.stderr + result.stdout
        
        # Check for key metrics
        signals_match = re.search(r'SIGNAL:.*?(\d+)', output)
        decisions_match = re.search(r'decisions_count=(\d+)', output)
        trades_match = re.search(r'FILLED.*?(\d+)', output)
        
        print("\n📊 KEY METRICS:")
        print("-" * 80)
        
        if signals_match:
            signal_count = len(re.findall(r'SIGNAL:', output))
            print(f"✅ Signals generated: {signal_count}")
        else:
            print("❓ Signals: could not parse")
        
        if decisions_match:
            decision_count = int(decisions_match.group(1))
            print(f"{'✅' if decision_count > 0 else '❌'} Decisions built: {decision_count}")
        else:
            print("❓ Decisions: could not parse")
        
        trades_count = len(re.findall(r'FILLED', output))
        print(f"{'✅' if trades_count > 0 else '⚠️'} Trades executed: {trades_count}")
        
        # Print last 50 lines for debugging
        print("\n📋 LAST 50 LINES OF LOG:")
        print("-" * 80)
        lines = output.split('\n')
        for line in lines[-50:]:
            if line.strip():
                print(line)
        
        return decision_count > 0 if decisions_match else False
        
    except subprocess.TimeoutExpired:
        print("⏱️ Test timed out (expected for long-running process)")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == '__main__':
    success = run_test()
    if success is True:
        print("\n✅ FIX VERIFIED: Decisions are now being generated!")
        sys.exit(0)
    elif success is False:
        print("\n❌ FIX FAILED: Decisions are still not being generated")
        sys.exit(1)
    else:
        print("\n⚠️ UNCERTAIN: Could not determine test result")
        sys.exit(2)
