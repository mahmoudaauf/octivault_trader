#!/usr/bin/env python3
"""Real-time trading diagnostics - checks for rejection patterns"""

import subprocess
import re
import time
import sys

def get_recent_log_lines(filename, num_lines=200):
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
        return f"Error: {e}"

def analyze_current_state():
    """Analyze current trading state"""
    log_file = "logs/trading_run_20260425T080527Z.log"
    
    text = get_recent_log_lines(log_file, 300)
    
    # Count patterns
    zero_blocks = len(re.findall(r'\[EM:ZERO_AMT_BLOCK\]', text))
    alloc_traces = len(re.findall(r'\[Meta:ALLOC_TRACE\]', text))
    loops = re.findall(r'LOOP_SUMMARY.*', text)
    
    print("\n" + "="*70)
    print("REAL-TIME TRADING STATE ANALYSIS")
    print("="*70)
    print(f"Log file: {log_file}")
    print(f"\n📊 Pattern counts (in last 300 lines):")
    print(f"  ZERO_AMT_BLOCK: {zero_blocks}")
    print(f"  ALLOC_TRACE: {alloc_traces}")
    print(f"  LOOP_SUMMARY: {len(loops)}")
    
    if loops:
        print(f"\n📋 Last 3 loops:")
        for loop in loops[-3:]:
            # Parse metrics
            m_id = re.search(r'loop_id=(\d+)', loop)
            m_res = re.search(r'exec_result=([A-Z_]+)', loop)
            m_pnl = re.search(r'pnl=([\d\.\-]+)', loop)
            m_trade = re.search(r'trade_opened=([A-Za-z]+)', loop)
            m_reason = re.search(r'rejection_reason=([^\s]+)', loop)
            
            loop_id = m_id.group(1) if m_id else "?"
            result = m_res.group(1) if m_res else "?"
            pnl = m_pnl.group(1) if m_pnl else "?"
            trade = m_trade.group(1) if m_trade else "?"
            reason = m_reason.group(1) if m_reason else "None"
            
            print(f"  Loop {loop_id}: result={result:15s} | trade={trade:5s} | pnl={pnl:7s} | reason={reason}")
    
    # Check for zero blocks
    if zero_blocks > 0:
        print(f"\n⚠️  CRITICAL: {zero_blocks} ZERO_AMOUNT blocks detected!")
        print("   This means ExecutionManager is blocking trades due to zero quantities.")
        print("   Root cause likely: zero planned_quote or zero final_qty")
    
    if alloc_traces > 0:
        print(f"\n✅ Allocation traces being logged ({alloc_traces} found)")
    
    print("\n" + "="*70)

if __name__ == '__main__':
    analyze_current_state()
