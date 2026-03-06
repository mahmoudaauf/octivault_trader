#!/usr/bin/env python3
"""
Verify that confidence hardcoding fix is working correctly.

Run AFTER restarting the bot to confirm:
1. Bot is actually using new code
2. Confidence values vary (not always 0.500)
3. Floor values correct for regime
"""

import subprocess
import time
import re
from pathlib import Path


def check_recent_logs(last_n_lines=50):
    """Check recent bot logs for confidence values."""
    log_file = Path("./logs/bot.log")
    
    if not log_file.exists():
        print(f"❌ Log file not found: {log_file}")
        return None
    
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()[-last_n_lines:]
    except Exception as e:
        print(f"❌ Error reading log: {e}")
        return None
    
    # Look for confidence log patterns
    pattern = r"(\w+) heuristic.*regime=(\w+).*mag=([\d.]+).*floor=([\d.]+).*final=([\d.]+)"
    
    matches = []
    for line in lines:
        m = re.search(pattern, line)
        if m:
            matches.append({
                'action': m.group(1),
                'regime': m.group(2),
                'magnitude': float(m.group(3)),
                'floor': float(m.group(4)),
                'confidence': float(m.group(5)),
            })
    
    return matches


def analyze_results(matches):
    """Analyze confidence values to verify fix is working."""
    if not matches:
        print("⚠️  No confidence log lines found. Log more signals or check log file location.")
        return False
    
    print(f"\n📊 Analyzed {len(matches)} signal generation logs\n")
    
    # Check 1: Confidence varies
    confidences = [m['confidence'] for m in matches]
    unique_confs = set(confidences)
    
    print(f"Unique confidence values: {sorted(unique_confs)}")
    
    if len(unique_confs) == 1 and list(unique_confs)[0] == 0.5:
        print("❌ FAIL: All confidence values are 0.500 (hardcoding bug still present)")
        return False
    
    if len(unique_confs) == 1:
        print(f"⚠️  WARNING: Only one unique confidence value: {list(unique_confs)[0]}")
    else:
        print(f"✅ PASS: Confidence values vary ({min(confidences):.3f} to {max(confidences):.3f})")
    
    # Check 2: Floor values are correct
    print(f"\nFloor values by regime:")
    floor_by_regime = {}
    for m in matches:
        regime = m['regime']
        if regime not in floor_by_regime:
            floor_by_regime[regime] = []
        floor_by_regime[regime].append(m['floor'])
    
    expected_floors = {
        'normal': 0.55,
        'trending': 0.65,
        'chop': 0.78,
        'sideways': 0.65,
        'bear': 0.55,
    }
    
    all_correct = True
    for regime, floors in sorted(floor_by_regime.items()):
        unique_floor = set(floors)
        expected = expected_floors.get(regime, '?')
        status = "✅" if len(unique_floor) == 1 and list(unique_floor)[0] == expected else "❌"
        print(f"  {status} {regime}: {unique_floor} (expected {expected})")
        if len(unique_floor) != 1 or list(unique_floor)[0] != expected:
            all_correct = False
    
    if all_correct:
        print("✅ PASS: All regime floors are correct")
    else:
        print("❌ FAIL: Some regime floors are incorrect")
    
    # Check 3: Magnitude > 0 for non-HOLD signals
    print(f"\nMagnitude distribution:")
    mags = [m['magnitude'] for m in matches]
    print(f"  Min: {min(mags):.4f}")
    print(f"  Max: {max(mags):.4f}")
    print(f"  Avg: {sum(mags)/len(mags):.4f}")
    
    zero_mags = [m for m in matches if m['magnitude'] == 0.0]
    if zero_mags:
        print(f"⚠️  {len(zero_mags)}/{len(matches)} signals have mag=0.0000")
    
    # Final verdict
    print("\n" + "="*60)
    if len(unique_confs) > 1 and all_correct:
        print("✅ SUCCESS: Confidence fix is working correctly!")
        print(f"   - Confidence values vary dynamically")
        print(f"   - Regime floors applied correctly")
        print(f"   - Bot is using new code")
        return True
    else:
        print("❌ FAILURE: Fix may not be deployed or bot not restarted")
        print(f"   - Confidence still hardcoded: {len(unique_confs) == 1}")
        print(f"   - Regime floors wrong: {not all_correct}")
        return False


if __name__ == "__main__":
    print("🔍 Confidence Hardcoding Fix Verification\n")
    print("Checking logs for evidence that fix is deployed and working...\n")
    
    matches = check_recent_logs(100)
    
    if matches:
        success = analyze_results(matches)
        exit(0 if success else 1)
    else:
        print("No logs found. Make sure:")
        print("1. Bot is running and generating signals")
        print("2. Log file is at ./logs/bot.log")
        exit(1)
