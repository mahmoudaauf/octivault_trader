#!/usr/bin/env python3
"""
Quick diagnostic analyzer for signal filtering issues.
Parses logs and shows signal flow through MetaController filters.
"""

import sys
import re
from collections import defaultdict
from pathlib import Path

def analyze_logs(log_file):
    """Analyze MetaController filtering logs."""
    if not Path(log_file).exists():
        print(f"❌ Log file not found: {log_file}")
        return
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    # Find all diagnostic lines
    signal_intake = []
    gate_traces = []
    gate_drops = defaultdict(list)
    gate_passed = []
    after_filter = []
    deadlock = []
    
    for line in lines:
        if '[Meta:SIGNAL_INTAKE]' in line:
            signal_intake.append(line.strip())
        elif '[Meta:GATE_TRACE]' in line:
            gate_traces.append(line.strip())
        elif '[Meta:GATE_DROP_' in line:
            # Extract gate type
            match = re.search(r'\[Meta:GATE_DROP_(\w+)\]', line)
            if match:
                gate_type = match.group(1)
                gate_drops[gate_type].append(line.strip())
        elif '[Meta:GATE_PASSED]' in line:
            gate_passed.append(line.strip())
        elif '[Meta:AFTER_FILTER]' in line:
            after_filter.append(line.strip())
        elif '[Meta:DEADLOCK_DIAGNOSTIC]' in line:
            deadlock.append(line.strip())
    
    # Print summary
    print("\n" + "="*80)
    print("SIGNAL FILTERING DIAGNOSTIC SUMMARY")
    print("="*80 + "\n")
    
    # Signal intake
    print(f"📥 SIGNAL INTAKE:")
    if signal_intake:
        print(f"   Found {len(signal_intake)} entries")
        for entry in signal_intake[-3:]:
            print(f"   {entry}")
    else:
        print("   ❌ No SIGNAL_INTAKE entries found!")
    
    print()
    
    # Gate traces
    print(f"🔍 GATE TRACES (signals entering filter):")
    if gate_traces:
        print(f"   Found {len(gate_traces)} signal entries")
        # Count by symbol
        symbols = defaultdict(int)
        for trace in gate_traces:
            match = re.search(r'Processing (\w+)', trace)
            if match:
                symbols[match.group(1)] += 1
        print(f"   By symbol: {dict(symbols)}")
    else:
        print("   ❌ No signals entered filter!")
    
    print()
    
    # Gate drops
    print(f"🚫 GATE DROPS (signals rejected):")
    if gate_drops:
        total_drops = sum(len(v) for v in gate_drops.values())
        print(f"   Total drops: {total_drops}")
        for gate_type in sorted(gate_drops.keys()):
            count = len(gate_drops[gate_type])
            print(f"   - {gate_type}: {count} drops")
            for drop in gate_drops[gate_type][-2:]:
                print(f"     {drop[:100]}...")
    else:
        print("   ✅ No gate drops detected!")
    
    print()
    
    # Gate passed
    print(f"✅ SIGNALS PASSED ALL GATES:")
    if gate_passed:
        print(f"   Found {len(gate_passed)} signals that passed")
        for passed in gate_passed[-3:]:
            print(f"   {passed}")
    else:
        print("   ❌ No signals passed all gates!")
    
    print()
    
    # After filter
    print(f"📊 AFTER-FILTER STATE:")
    if after_filter:
        print(f"   Found {len(after_filter)} entries")
        for entry in after_filter[-2:]:
            # Extract symbol count
            match = re.search(r'has (\d+) symbols', entry)
            if match:
                count = match.group(1)
                print(f"   → {count} symbols with valid signals")
                print(f"   {entry[:150]}...")
    else:
        print("   ❌ No AFTER_FILTER entries found!")
    
    print()
    
    # Deadlock diagnostic
    print(f"🔴 DEADLOCK DIAGNOSTICS:")
    if deadlock:
        print(f"   Found {len(deadlock)} deadlock messages")
        for msg in deadlock[-1:]:
            print(f"   {msg[:200]}...")
    else:
        print("   ✅ No deadlock messages (signals are flowing!)")
    
    print("\n" + "="*80)
    print("ANALYSIS:")
    print("="*80 + "\n")
    
    if not signal_intake:
        print("🔴 ISSUE: No signals being generated/cached by TrendHunter")
        print("   → Check TrendHunter agent logs")
    elif not gate_traces:
        print("🔴 ISSUE: Signals retrieved but not processed")
        print("   → Check for exceptions in signal loop")
    elif len(gate_passed) == 0 and total_drops > 0:
        print("🔴 ISSUE: ALL signals dropped by filters!")
        print(f"   → Primary gate dropping signals: {sorted(gate_drops.keys(), key=lambda k: len(gate_drops[k]), reverse=True)[0]}")
        print(f"   → Review filter settings and market conditions")
    elif len(gate_passed) > 0:
        print("✅ GOOD: Signals are passing through filters!")
        print("   → Check if decisions are being created from these signals")
        print("   → Check ExecutionManager logs for execution attempts")
    else:
        print("⚠️  Inconclusive - check logs manually")
    
    print()

if __name__ == "__main__":
    log_file = sys.argv[1] if len(sys.argv) > 1 else "logs/clean_run.log"
    analyze_logs(log_file)
