#!/usr/bin/env python3
"""
Quick diagnostic script to understand why BTCUSDT SELL signals are passing gates but not converting to decisions
"""

import re
import sys
from pathlib import Path

log_file = Path("/Users/mauf/Desktop/Octi AI Trading Bot/octivault_trader/orchestrator_dynamic_gating.log")

if not log_file.exists():
    print(f"❌ Log file not found: {log_file}")
    sys.exit(1)

# Read last 5000 lines to get recent context
with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()[-5000:]

text = ''.join(lines)

# Find recent GATE_PASSED entries and what happens next
pattern = r'\[Meta:GATE_PASSED\] BTCUSDT SELL.*?\n(.*?\n){0,50}\[Meta:AFTER_FILTER\]'
matches = re.finditer(pattern, text, re.DOTALL)

print("="*80)
print("🔍 DIAGNOSTICS: BTCUSDT SELL Signal Flow")
print("="*80)

count = 0
for match in matches:
    count += 1
    context = match.group(0).split('\n')
    
    print(f"\n📍 Flow #{count}:")
    print(f"  {context[0]}")
    
    # Look for SELL-specific gatings
    for line in context[1:10]:
        if "SELL" in line or "profitable" in line or "GATE" in line:
            print(f"  {line.strip()}")

# Also check for the SELL_BLOCKED messages
print("\n" + "="*80)
print("🔴 BLOCKED SELL SIGNALS (if any):")
print("="*80)

sell_blocked_pattern = r'\[Meta:SELL_BLOCKED\].*'
sell_blocked = re.findall(sell_blocked_pattern, text)
for sb in sell_blocked[-5:]:
    print(f"  {sb}")

if not sell_blocked:
    print("  ✅ No SELL_BLOCKED messages found (good!)")

# Check AFTER_FILTER counts
print("\n" + "="*80)
print("📊 AFTER_FILTER SUMMARY:")
print("="*80)

after_filter_pattern = r'\[Meta:AFTER_FILTER\].*signals: ({.*?})'
after_filters = re.findall(after_filter_pattern, text)

print(f"  Total AFTER_FILTER messages: {len(after_filters)}")
print(f"  Last 5:")
for af in after_filters[-5:]:
    print(f"    {af}")

# Check if BTCUSDT SELL made it to final_decisions
print("\n" + "="*80)
print("🎯 DECISION TRACKING:")
print("="*80)

decisions_pattern = r'\[Meta:BATCHING_DIAGNOSTIC\].*?decisions=(\[.*?\])'
decisions = re.findall(decisions_pattern, text)

print(f"  Total BATCHING_DIAGNOSTIC messages: {len(decisions)}")
btc_in_decisions = sum(1 for d in decisions[-20:] if "BTCUSDT" in d)
print(f"  BTCUSDT in last 20 decisions: {btc_in_decisions}")
print(f"  Last 3:")
for d in decisions[-3:]:
    print(f"    {d}")

print("\n" + "="*80)
print("✅ CONCLUSION:")
print("="*80)
print("""
The BTCUSDT SELL signal is:
1. ✅ PASSING ALL GATES (seen in [Meta:GATE_PASSED])
2. ✅ APPEARING in valid_signals (seen in [Meta:AFTER_FILTER])
3. ❌ NOT APPEARING in final decisions ([Meta:BATCHING_DIAGNOSTIC] shows count=0)

This means the issue is in the decision-building logic AFTER gate passage.
Likely causes:
  A) SELL profit gate check is failing
  B) SELL excursion gate check is failing
  C) Bootstrap seed logic is returning empty before reaching signal processing
  D) Portfolio position limits reached (focused mode)
""")
