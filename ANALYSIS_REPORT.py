#!/usr/bin/env python3
"""
COMPREHENSIVE TRADING SYSTEM DIAGNOSTICS
Analyzes archived log for rejection patterns and trading issues
"""

import re
import json
from collections import defaultdict
from pathlib import Path

def analyze_logs():
    # Read the archived log
    log_file = Path("logs/trading_run_20260425T074834Z.log.archived")
    
    if not log_file.exists():
        print("❌ Log file not found")
        return
    
    print("\n" + "="*80)
    print("DIAGNOSTIC ANALYSIS - Trading System Performance Report")
    print("="*80)
    
    # Parse log file line by line (efficient memory usage)
    loop_count = 0
    exec_attempts = 0
    trades_opened = 0
    trades_rejected = 0
    rejection_reasons = defaultdict(int)
    decisions = defaultdict(int)
    exec_results = defaultdict(int)
    pnl_values = []
    last_loops = []
    
    print("\n📖 Scanning log file (this may take a moment)...")
    
    try:
        with open(log_file, 'r') as f:
            for line in f:
                if 'LOOP_SUMMARY' in line:
                    loop_count += 1
                    
                    # Extract metrics
                    loop_id = re.search(r'loop_id=(\d+)', line)
                    decision = re.search(r'decision=([A-Z]+)', line)
                    exec_attempted = re.search(r'exec_attempted=([A-Za-z]+)', line)
                    exec_result = re.search(r'exec_result=([A-Z_]+)', line)
                    pnl = re.search(r'pnl=([\d\.\-]+)', line)
                    trade_opened = re.search(r'trade_opened=([A-Za-z]+)', line)
                    reason = re.search(r'rejection_reason=([^\s]+)', line)
                    
                    # Count metrics
                    if exec_attempted and 'True' in exec_attempted.group(1):
                        exec_attempts += 1
                    if trade_opened and 'True' in trade_opened.group(1):
                        trades_opened += 1
                    else:
                        trades_rejected += 1
                    
                    if decision:
                        decisions[decision.group(1)] += 1
                    if exec_result:
                        exec_results[exec_result.group(1)] += 1
                    if reason and reason.group(1) != 'None':
                        rejection_reasons[reason.group(1)] += 1
                    if pnl:
                        pnl_val = float(pnl.group(1))
                        pnl_values.append(pnl_val)
                    
                    # Store last 5 loops
                    if len(last_loops) >= 5:
                        last_loops.pop(0)
                    last_loops.append({
                        'id': int(loop_id.group(1)) if loop_id else 0,
                        'decision': decision.group(1) if decision else '?',
                        'exec_attempted': exec_attempted.group(1) if exec_attempted else 'False',
                        'exec_result': exec_result.group(1) if exec_result else '?',
                        'pnl': float(pnl.group(1)) if pnl else 0.0,
                        'trade_opened': trade_opened.group(1) if trade_opened else 'False',
                        'rejection_reason': reason.group(1) if reason else 'None'
                    })
    
    except Exception as e:
        print(f"❌ Error reading log: {e}")
        return
    
    # Print summary
    print(f"\n📊 SUMMARY STATISTICS:")
    print(f"  Total loop cycles: {loop_count}")
    print(f"  Execution attempts: {exec_attempts}")
    print(f"  Successful trades: {trades_opened}")
    print(f"  Rejected trades: {trades_rejected}")
    print(f"  Execution success rate: {(trades_opened/max(exec_attempts,1)*100):.1f}%")
    
    if pnl_values:
        print(f"\n💰 PnL ANALYSIS:")
        print(f"  Final PnL: ${pnl_values[-1]:.2f} USDT")
        print(f"  Max PnL: ${max(pnl_values):.2f} USDT")
        print(f"  Min PnL: ${min(pnl_values):.2f} USDT")
    
    print(f"\n🎯 DECISION BREAKDOWN:")
    for decision, count in sorted(decisions.items(), key=lambda x: x[1], reverse=True):
        pct = (count / loop_count * 100) if loop_count > 0 else 0
        print(f"  {decision}: {count}x ({pct:.1f}%)")
    
    print(f"\n✅ EXECUTION RESULTS:")
    for result, count in sorted(exec_results.items(), key=lambda x: x[1], reverse=True):
        pct = (count / loop_count * 100) if loop_count > 0 else 0
        print(f"  {result}: {count}x ({pct:.1f}%)")
    
    if rejection_reasons:
        print(f"\n❌ REJECTION REASONS (Top 10):")
        for reason, count in sorted(rejection_reasons.items(), key=lambda x: x[1], reverse=True)[:10]:
            pct = (count / trades_rejected * 100) if trades_rejected > 0 else 0
            print(f"  {reason}: {count}x ({pct:.1f}%)")
    
    print(f"\n📋 LAST 5 LOOP CYCLES:")
    print(f"  {'ID':>3} | {'Decision':>7} | {'Exec Att':>8} | {'Result':>16} | {'Trade':>6} | {'PnL':>8}")
    print(f"  {'-'*3}+{'-'*7}+{'-'*8}+{'-'*16}+{'-'*6}+{'-'*8}")
    for loop in last_loops:
        print(f"  {loop['id']:>3} | {loop['decision']:>7} | {loop['exec_attempted']:>8} | {loop['exec_result']:>16} | {loop['trade_opened']:>6} | {loop['pnl']:>8.2f}")
    
    print("\n" + "="*80)
    print("🔍 KEY FINDINGS:")
    print("="*80)
    
    if trades_opened == 0:
        print("⚠️  CRITICAL: Zero trades executed despite many attempts")
        print("   This suggests a systematic rejection issue in the execution pipeline")
        print("   Next steps:")
        print("   1. Check allocation validation (INVALID_AMOUNT errors)")
        print("   2. Check execution manager pre-exec guards (ZERO_COMPUTED_AMOUNT)")
        print("   3. Check micro_backtest gate (win_rate threshold)")
        print("   4. Check exchange-level rejections (minNotional, etc)")
    
    if exec_results.get('REJECTED', 0) > 0:
        print(f"\n⚠️  {exec_results.get('REJECTED', 0)} execution rejections detected")
        print(f"   Most common reason: {max(rejection_reasons.items(), key=lambda x: x[1])[0] if rejection_reasons else 'N/A'}")
    
    print("\n" + "="*80)

if __name__ == '__main__':
    analyze_logs()
