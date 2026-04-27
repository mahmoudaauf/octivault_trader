#!/usr/bin/env python3
"""Fast diagnostics for trading system - reads last N lines efficiently"""

import subprocess
import re
import sys
from collections import defaultdict

def get_recent_logs(filename, num_lines=100):
    """Get last N lines from file efficiently"""
    try:
        result = subprocess.run(
            ['tail', '-n', str(num_lines), filename],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.stdout
    except Exception as e:
        print(f"Error reading logs: {e}")
        return ""

def extract_metrics(log_text):
    """Extract trading metrics from log lines"""
    loops = []
    rejections = defaultdict(int)
    exec_count = 0
    trades_count = 0
    total_pnl = 0.0
    
    for line in log_text.split('\n'):
        if 'LOOP_SUMMARY' in line:
            # Extract metrics
            loop_id = re.search(r'loop_id=(\d+)', line)
            decision = re.search(r'decision=([A-Z]+)', line)
            exec_attempted = re.search(r'exec_attempted=([A-Za-z]+)', line)
            exec_result = re.search(r'exec_result=([A-Z_]+)', line)
            pnl = re.search(r'pnl=([\d\.\-]+)', line)
            trade_opened = re.search(r'trade_opened=([A-Za-z]+)', line)
            reason = re.search(r'rejection_reason=([^\s]+)', line)
            
            loop_data = {
                'id': int(loop_id.group(1)) if loop_id else 0,
                'decision': decision.group(1) if decision else '?',
                'exec_attempted': exec_attempted.group(1) if exec_attempted else 'False',
                'exec_result': exec_result.group(1) if exec_result else '?',
                'pnl': float(pnl.group(1)) if pnl else 0.0,
                'trade_opened': trade_opened.group(1) if trade_opened else 'False',
                'rejection_reason': reason.group(1) if reason else 'None'
            }
            loops.append(loop_data)
            
            # Count metrics
            if 'True' in (exec_attempted.group(1) if exec_attempted else ''):
                exec_count += 1
            if 'True' in (trade_opened.group(1) if trade_opened else ''):
                trades_count += 1
            if pnl:
                total_pnl = float(pnl.group(1))
            if reason and reason.group(1) != 'None':
                rejections[reason.group(1)] += 1
    
    return loops, rejections, exec_count, trades_count, total_pnl

def main():
    print("\n" + "=" * 90)
    print("FAST DIAGNOSTICS - Trading System Analysis")
    print("=" * 90)
    
    # Get recent logs (last 500 lines)
    logs = get_recent_logs('logs/trading_run_20260425T074834Z.log', 500)
    
    if not logs:
        print("ERROR: Could not read logs")
        return
    
    loops, rejections, exec_count, trades_count, total_pnl = extract_metrics(logs)
    
    print(f"\n📊 LOOP STATISTICS:")
    print(f"  Total loops in last 500 lines: {len(loops)}")
    print(f"  Execution attempts: {exec_count}")
    print(f"  Trades opened: {trades_count}")
    print(f"  Current PnL: ${total_pnl:.2f} USDT")
    
    if rejections:
        print(f"\n❌ REJECTION REASONS (frequency):")
        for reason, count in sorted(rejections.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {reason}: {count}x")
    else:
        print(f"\n✅ No rejections found!")
    
    if loops:
        print(f"\n📋 LAST 10 LOOPS:")
        print(f"  {'ID':>3} | {'Decision':>6} | {'Attempted':>9} | {'Result':>15} | {'Trades':>6} | PnL {'':>7}")
        print(f"  {'-'*3}+{'-'*6}+{'-'*9}+{'-'*15}+{'-'*6}+{'-'*14}")
        
        for loop in loops[-10:]:
            print(f"  {loop['id']:>3} | {loop['decision']:>6} | {loop['exec_attempted']:>9} | {loop['exec_result']:>15} | {str(loop['trade_opened']):>6} | {loop['pnl']:>7.2f}")
    
    print("\n" + "=" * 90)

if __name__ == '__main__':
    main()
