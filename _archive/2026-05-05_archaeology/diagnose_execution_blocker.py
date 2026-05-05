#!/usr/bin/env python3
"""Diagnose why trades aren't executing despite having capital"""
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

def check_logs():
    """Analyze execution logs for blockers"""
    log_path = Path("/tmp/octivault_master_orchestrator.log")
    
    if not log_path.exists():
        print("❌ No orchestrator log found")
        return
    
    # Read last 500 lines
    with open(log_path, 'r') as f:
        lines = f.readlines()[-500:]
    
    print("=" * 80)
    print("EXECUTION BLOCKER DIAGNOSIS")
    print("=" * 80)
    
    # Count rejection reasons
    rejection_reasons = {}
    nav_values = []
    accumulated = {}
    
    for line in lines:
        if "RULE5_ESCALATION_INSUFFICIENT_QUOTE" in line:
            symbol = line.split("symbol=")[1].split()[0] if "symbol=" in line else "unknown"
            rejection_reasons[f"INSUFFICIENT_QUOTE_{symbol}"] = rejection_reasons.get(f"INSUFFICIENT_QUOTE_{symbol}", 0) + 1
        
        if "CapitalGovernor:PositionLimits" in line and "NAV=$" in line:
            nav = line.split("NAV=$")[1].split()[0] if "NAV=$" in line else "0"
            nav_values.append(float(nav))
        
        if "Accumulating" in line and "for" in line:
            parts = line.split()
            try:
                amount = float(parts[parts.index("Accumulating")+1])
                sym = parts[parts.index("for")+1]
                accumulated[sym] = accumulated.get(sym, 0) + amount
            except:
                pass
    
    print("\n📊 CAPITAL STATUS:")
    if nav_values:
        print(f"   Current NAV: ${nav_values[-1]:.2f}")
        print(f"   Min NAV (last 500 lines): ${min(nav_values):.2f}")
        print(f"   Max NAV (last 500 lines): ${max(nav_values):.2f}")
    
    print("\n⚠️  EXECUTION REJECTIONS (Recent):")
    for reason, count in sorted(rejection_reasons.items(), key=lambda x: x[1], reverse=True):
        print(f"   {reason}: {count} times")
    
    print("\n💰 ACCUMULATED QUOTES (Waiting to Execute):")
    for sym, amount in sorted(accumulated.items(), key=lambda x: x[1], reverse=True):
        print(f"   {sym}: ${amount:.2f}")
    
    print("\n🔍 INVESTIGATION:")
    
    # Check if there are active positions blocking execution
    if "RULE5_ESCALATION_INSUFFICIENT_QUOTE" in str(lines):
        print("   ❌ BLOCKER IDENTIFIED: RULE5_ESCALATION triggered")
        print("      → System trying to ACCUMULATE capital for positions")
        print("      → But accumulation blocked by insufficient quote rule")
        print("      → This creates a deadlock: can't accumulate = can't execute")
    
    # Check if this is a capital floor issue
    if nav_values and min(nav_values) < 5:
        print("   ⚠️  LOW CAPITAL WARNING: NAV went below $5 at some point")
        print("      → Capital floor constraint may be too restrictive")
    
    print("\n✅ RECOMMENDATIONS:")
    print("   1. Check RULE5_ESCALATION config in execution manager")
    print("   2. Verify accumulation min_quote threshold")
    print("   3. Check capital_floor calculation")
    print("   4. Consider enabling auto-correction for this blocker")

if __name__ == "__main__":
    check_logs()
