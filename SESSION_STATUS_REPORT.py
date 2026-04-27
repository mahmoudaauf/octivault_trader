#!/usr/bin/env python3
"""
2-HOUR SESSION STATUS REPORT
Real-time status snapshot of the trading system
"""

import subprocess
import json
import re
from datetime import datetime
from pathlib import Path

def run_cmd(cmd):
    """Run shell command safely"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        return result.stdout.strip()
    except:
        return ""

def main():
    print("\n" + "=" * 100)
    print("🎯 2-HOUR CHECKPOINT SESSION - STATUS REPORT".center(100))
    print("=" * 100)
    print(f"Report Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)
    
    # Check processes
    print("\n📊 SYSTEM PROCESSES:")
    procs = run_cmd("ps aux | grep -E 'MASTER_SYSTEM_ORCHESTRATOR|2HOUR_CHECKPOINT' | grep -v grep | wc -l")
    print(f"   ✅ Active Processes: {procs.strip()}")
    
    # Get PID
    pid_output = run_cmd("ps aux | grep MASTER_SYSTEM_ORCHESTRATOR | grep -v grep | awk '{print $2}'")
    if pid_output:
        print(f"   └─ Main PID: {pid_output}")
    
    # Get uptime from log
    log_output = run_cmd("grep 'Uptime:' /tmp/octivault_master_orchestrator.log 2>/dev/null | tail -1")
    
    print("\n📈 TRADING ACTIVITY:")
    
    # Count signals
    signals = run_cmd("grep -c 'Published TradeIntent' /tmp/octivault_master_orchestrator.log 2>/dev/null || echo 0")
    print(f"   ├─ Total Signals Generated: {signals.strip()}")
    
    # Count rejections
    rejections = run_cmd("grep -c 'EXEC_REJECT' /tmp/octivault_master_orchestrator.log 2>/dev/null || echo 0")
    print(f"   ├─ Execution Rejections: {rejections.strip()}")
    
    # Get latest signals
    print("\n   📋 Latest Signals (last 10):")
    latest_signals = run_cmd("grep 'Published TradeIntent' /tmp/octivault_master_orchestrator.log 2>/dev/null | tail -10 | awk -F': ' '{print $NF}'")
    if latest_signals:
        for i, signal in enumerate(latest_signals.split('\n'), 1):
            if signal.strip():
                print(f"      {i}. {signal.strip()}")
    
    # Get current positions
    print("\n💼 POSITION STATUS:")
    positions = run_cmd("grep -oE '[A-Z]{3,}USDT' /tmp/octivault_master_orchestrator.log 2>/dev/null | sort | uniq -c | sort -rn | head -10")
    if positions:
        print("   Active Symbols (most recent):")
        for line in positions.split('\n'):
            if line.strip():
                print(f"      {line.strip()}")
    
    # System health indicators
    print("\n❤️  SYSTEM HEALTH:")
    
    running = run_cmd("grep -c 'Status: Running' /tmp/octivault_master_orchestrator.log 2>/dev/null || echo 0")
    print(f"   ├─ Running Components: {running.strip()}")
    
    healthy = run_cmd("grep -c 'Status: Healthy' /tmp/octivault_master_orchestrator.log 2>/dev/null || echo 0")
    print(f"   ├─ Healthy Components: {healthy.strip()}")
    
    warnings = run_cmd("grep -c '\\[WARNING' /tmp/octivault_master_orchestrator.log 2>/dev/null || echo 0")
    print(f"   ├─ Warnings: {warnings.strip()}")
    
    errors = run_cmd("grep -c '\\[ERROR' /tmp/octivault_master_orchestrator.log 2>/dev/null || echo 0")
    print(f"   └─ Errors: {errors.strip()}")
    
    # Trading decision gates
    print("\n🚦 TRADING DECISION GATES:")
    
    buy_allowed = run_cmd("grep -c 'BUY allowed' /tmp/octivault_master_orchestrator.log 2>/dev/null || echo 0")
    print(f"   ├─ BUY Signals Generated: {buy_allowed.strip()}")
    
    sell_allowed = run_cmd("grep -c 'SELL allowed' /tmp/octivault_master_orchestrator.log 2>/dev/null || echo 0")
    print(f"   ├─ SELL Signals Generated: {sell_allowed.strip()}")
    
    # Risk management
    print("\n🛡️  RISK MANAGEMENT:")
    
    tp_sl_mentions = run_cmd("grep -c 'TPSLEngine\\|TP/SL' /tmp/octivault_master_orchestrator.log 2>/dev/null || echo 0")
    print(f"   ├─ TP/SL Monitoring: {tp_sl_mentions.strip()} mentions")
    
    capital_alloc = run_cmd("grep -c 'CapitalAllocator' /tmp/octivault_master_orchestrator.log 2>/dev/null || echo 0")
    print(f"   └─ Capital Allocation Events: {capital_alloc.strip()}")
    
    # Get rejection reasons
    print("\n📌 TOP REJECTION REASONS (last 20 minutes):")
    rejections_detail = run_cmd("grep 'EXEC_REJECT' /tmp/octivault_master_orchestrator.log 2>/dev/null | tail -20 | grep -oP 'reason=\\K[^ ]*' | sort | uniq -c | sort -rn")
    if rejections_detail:
        for line in rejections_detail.split('\n')[:5]:
            if line.strip():
                print(f"   • {line.strip()}")
    
    # Session timeline
    print("\n⏱️  SESSION TIMELINE:")
    
    start_time = run_cmd("head -1 /tmp/octivault_master_orchestrator.log 2>/dev/null | awk '{print $1}'")
    print(f"   ├─ Started: {start_time.strip()}")
    
    latest_time = run_cmd("tail -1 /tmp/octivault_master_orchestrator.log 2>/dev/null | awk '{print $1}'")
    print(f"   ├─ Latest Activity: {latest_time.strip()}")
    
    duration = run_cmd("grep -oP '\\d+:\\d+:\\d+' /tmp/octivault_master_orchestrator.log 2>/dev/null | tail -1")
    print(f"   └─ Duration: ~{duration.strip() if duration else 'N/A'}")
    
    # Checkpoint status
    print("\n✅ CHECKPOINT MONITORING:")
    checkpoint_file = Path("/Users/mauf/Desktop/Octi AI Trading Bot/octivault_trader/trading_2hour_checkpoint_session.log")
    if checkpoint_file.exists():
        checkpoints = run_cmd(f"grep -c 'CHECKPOINT' {checkpoint_file}")
        print(f"   └─ Checkpoints Completed: {checkpoints.strip()}")
    
    print("\n" + "=" * 100)
    print("✅ Session Status: RUNNING".center(100))
    print("=" * 100 + "\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
